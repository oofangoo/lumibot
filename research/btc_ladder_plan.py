"""E022: backtest of the "buy in tranches, trim at +20%, stop at -30%" plan for BTC on Bitkub (real BTC/THB daily candles), against plain buy-and-hold.

The plan is measured from P0 = the last finished daily close before the purchase ("today's price"). Everything is fixed here BEFORE looking at results and
all variants are reported:
  entry   all-in : the whole amount is bought at the next open.
          ladder : 25% bought at the next open, resting limit buys (good till horizon) for 25% each at P0 -15%, -25%, -35%.
  exit    none | trim: limit sell of 1/3 of the units held (sized each morning) at P0 +20%, once | stop: if a daily close is at or below P0 x (1 - X),
          sell everything at the next open and cancel the unfilled buys (X = 30% or 40%) | trim + stop.
  => 2 entries x 6 exit settings = 12 variants. Horizons: 91 days (the question was "next 3 months") and 365 days. No re-entry after a stop.
Cash waiting for ladder buys earns nothing. Result = value after the horizon of 1 THB of capital (position liquidated at the last close, 0.25% fee each way),
so cash that was never invested counts. Limit orders fill as LumiBot's backtesting broker fills them on daily bars: a buy limit fills at the open if the bar
opens at or below it, else at the limit if the low reaches it; a sell limit the mirror image. Stops are decided on closes and sell at the next open
(Bitkub has no stop orders, so a bot or you must do it).
Engine check: the full plan also runs as a LumiBot Strategy through BitkubBacktesting (market + resting limit orders, cancel, replace) and its fills must equal
the pandas model's; cross-start-day statistics have no LumiBot equivalent, so those stay in pandas. One run is drawn with LumiBot's indicators chart.
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\btc_ladder_plan.py
"""
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["LUMIBOT_DISABLE_UI"] = "true"

import lumibot  # noqa: E402

if "lumibot-dev" not in str(Path(lumibot.__file__)):
    sys.exit("This experiment needs the Bitkub-enabled LumiBot (lumibot-dev). Set PYTHONPATH to the dev clone; see the docstring.")

from lumibot.backtesting import BitkubBacktesting  # noqa: E402
from lumibot.entities import TradingFee  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import BTC, BUFFER, CACHE, THB, load_bitkub  # noqa: E402

FEE = 0.0025
DIPS = (0.15, 0.25, 0.35)
TRIM_GAIN = 0.20
PRICE = "1. BTC price, THB (log)"
VARIANTS = {}
for _entry in ("all-in now", "4 tranches"):
    for _label, _trim, _stop in (("hold", False, 0), ("trim +20%", True, 0), ("stop 30%", False, 0.30), ("trim + stop 30%", True, 0.30), ("stop 40%", False, 0.40), ("trim + stop 40%", True, 0.40)):
        VARIANTS[f"{_entry} | {_label}"] = {"ladder": _entry == "4 tranches", "trim": _trim, "stop": _stop}


# ------------------------------------------------------------------------------------------------ pandas model
def sim_plan(o, h, low, c, t0, plan, horizon, fee=FEE, engine_sizing=False, lag=0):
    """Returns dict(final, maxdd, stopped, invested, fills[(day_index, side, price)]). Capital = 1.0."""
    n = len(c)
    p0 = c[t0 - 1]
    tranche = 0.25 if plan["ladder"] else 1.0
    buf = BUFFER if engine_sizing else 1.0
    size_px = p0 if engine_sizing else o[t0]  # the engine sizes on the last finished close, the pandas study on the fill price
    units = buf * tranche / (size_px * (1 + fee))
    cash = 1.0 - units * o[t0] * (1 + fee)
    pending = [(p0 * (1 - d), buf * 0.25 / (p0 * (1 - d) * (1 + fee))) for d in DIPS] if plan["ladder"] else []
    fills, trimmed, stopped = [(t0, "buy", o[t0])], False, False
    end = min(t0 + horizon, n - 1)
    peak, maxdd = 1.0, 0.0
    for idx in range(t0, end):
        trim_qty = units / 3 if (plan["trim"] and not trimmed and idx > t0 and units > 0) else 0.0  # sized from the position known this morning
        for level, qty in list(pending):
            if low[idx] <= level:
                price = min(o[idx], level)
                cash -= qty * price * (1 + fee)
                units += qty
                pending.remove((level, qty))
                fills.append((idx, "buy", price))
        if trim_qty > 0:
            level = p0 * (1 + TRIM_GAIN)
            if h[idx] >= level:
                price = max(o[idx], level)
                cash += trim_qty * price * (1 - fee)
                units -= trim_qty
                trimmed = True
                fills.append((idx, "sell", price))
        eq = cash + units * c[idx]
        peak = max(peak, eq)
        maxdd = max(maxdd, 1 - eq / peak)
        if plan["stop"] and units > 0 and c[idx - lag] <= p0 * (1 - plan["stop"]) and idx + 1 <= end:
            cash += units * o[idx + 1] * (1 - fee)
            fills.append((idx + 1, "sell", o[idx + 1]))
            units, pending, stopped = 0.0, [], True
            break
    final = cash + units * c[end] * (1 - fee)
    invested = units * c[end] * (1 - fee) / final if final > 0 else 0.0
    return {"final": final, "maxdd": maxdd, "stopped": stopped, "invested": invested, "fills": fills, "mark": cash + units * c[end]}


def study(o, h, low, c, starts, horizon):
    out = {}
    for name, plan in VARIANTS.items():
        r = [sim_plan(o, h, low, c, t0, plan, horizon) for t0 in starts]
        out[name] = (np.array([x["final"] for x in r]), np.array([x["maxdd"] for x in r]), np.array([x["stopped"] for x in r]), np.array([x["invested"] for x in r]))
    return out


def table(res):
    rows = []
    for name, (m, dd, stopped, inv) in res.items():
        rows.append({"variant": name, "worst": f"{(m.min() - 1) * 100:+.0f}%", "10th pct": f"{(np.percentile(m, 10) - 1) * 100:+.0f}%", "median": f"{(np.median(m) - 1) * 100:+.0f}%",
                     "mean": f"{(m.mean() - 1) * 100:+.0f}%", "below start": f"{(m < 1).mean() * 100:.0f}%", "typical dip on the way": f"-{np.median(dd) * 100:.0f}%",
                     "worst dip": f"-{dd.max() * 100:.0f}%", "stop fired": f"{stopped.mean() * 100:.0f}%", "in BTC at end": f"{inv.mean() * 100:.0f}%"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ LumiBot engine strategy
class LadderPlan(Strategy):
    parameters = {"ladder": True, "trim": True, "stop": 0.30}

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.p0, self.started, self.done, self.trimmed, self.trim_order = None, False, False, False, None
        self.fills, self.last_stamp = [], None

    def _draw(self, stamp, close):
        self.add_line("BTC/THB close", close, color="#f1f3f5", width=2, plot_name=PRICE, dt=stamp)
        p = self.parameters
        if p["ladder"]:
            for d, color in zip(DIPS, ("#ffd43b", "#ffa94d", "#ff8787")):
                self.add_line(f"Buy limit at -{d:.0%}", self.p0 * (1 - d), color=color, style="dotted", plot_name=PRICE, dt=stamp)
        if p["trim"]:
            self.add_line(f"Trim limit at +{TRIM_GAIN:.0%}", self.p0 * (1 + TRIM_GAIN), color="#69db7c", style="dashed", plot_name=PRICE, dt=stamp)
        if p["stop"]:
            self.add_line(f"Stop: sell if a close is {p['stop']:.0%} under start", self.p0 * (1 - p["stop"]), color="#ff6b6b", style="dashed", plot_name=PRICE, dt=stamp)

    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 3, "day", quote=THB)
        stamp = bars.df.index[-1]
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        close = float(bars.df["close"].iloc[-1])
        p = self.parameters
        if not self.started:
            self.started, self.p0 = True, close
            budget = self.get_cash()
            tranche = 0.25 if p["ladder"] else 1.0
            self.submit_order(self.create_order(BTC, BUFFER * tranche * budget / (close * (1 + FEE)), "buy", quote=THB))
            if p["ladder"]:
                for d in DIPS:
                    level = close * (1 - d)
                    self.submit_order(self.create_order(BTC, BUFFER * 0.25 * budget / (level * (1 + FEE)), "buy", limit_price=level, quote=THB, time_in_force="gtc"))
            self._draw(stamp, close)
            return
        if self.done:
            return
        self._draw(stamp, close)
        pos = self.get_position(BTC)
        if p["stop"] and pos is not None and pos.quantity > 0 and close <= self.p0 * (1 - p["stop"]):
            self.cancel_open_orders()
            self.submit_order(self.create_order(BTC, pos.quantity, "sell", quote=THB))
            self.done = True
            return
        if p["trim"] and not self.trimmed and pos is not None and pos.quantity > 0:
            if self.trim_order is not None:
                self.cancel_order(self.trim_order)
            self.trim_order = self.create_order(BTC, pos.quantity / 3, "sell", limit_price=self.p0 * (1 + TRIM_GAIN), quote=THB, time_in_force="gtc")
            self.submit_order(self.trim_order)

    def on_filled_order(self, position, order, price, quantity, multiplier):
        when = pd.Timestamp(self.get_datetime()).tz_convert("UTC").normalize()
        side = str(order.side)
        self.fills.append((when, side, float(price)))
        if self.trim_order is not None and order.identifier == self.trim_order.identifier:
            self.trimmed = True
        kind = "Buy" if side == "buy" else ("Trim (limit sell)" if order.order_type == "limit" or str(order.order_type) == "limit" else "Stop sell")
        self.add_marker(kind, float(price), color="#51cf66" if side == "buy" else "#ff6b6b", symbol="triangle-up" if side == "buy" else "triangle-down", size=12,
                        plot_name=PRICE, dt=when, detail_text=f"{when.date()}: {kind} {quantity:.5f} BTC at {price:,.0f}")


def run_engine(start_day, plan, horizon, chart=False):
    fees = [TradingFee(percent_fee=FEE)]
    start = datetime(start_day.year, start_day.month, start_day.day)
    flags = dict(analyze_backtest=chart, show_plot=chart, show_tearsheet=False, save_tearsheet=chart, show_indicators=chart)
    results, strat = LadderPlan.run_backtest(
        BitkubBacktesting, backtesting_start=start, backtesting_end=start + timedelta(days=horizon), cache_dir=CACHE, quote_asset=THB, benchmark_asset=None,
        budget=100_000, buy_trading_fees=fees, sell_trading_fees=fees, parameters=plan, save_logfile=False, show_progress_bar=False, quiet_logs=True,
        save_stats_file=False, name="LadderPlan", **flags)
    return results, strat


def norm(fills):
    return sorted((str(d.date()), s, round(p, 2)) for d, s, p in fills)


def parity(px, o, h, low, c):
    print("\n=== LumiBot engine (BitkubBacktesting) vs the pandas model: same fills? ===")
    days, n = list(px.index), len(c)
    full = VARIANTS["4 tranches | trim + stop 30%"]
    scan = range(int(np.searchsorted(px.index, pd.Timestamp("2021-01-01", tz="UTC"))), n - 366, 30)
    wanted = {  # category -> (plan, predicate on the pandas result)
        "ladder buys then a stop": (full, lambda r: r["stopped"] and sum(f[1] == "buy" for f in r["fills"]) >= 3),
        "a dip buy then a trim": (full, lambda r: not r["stopped"] and any(f[1] == "buy" for f in r["fills"][1:]) and any(f[1] == "sell" for f in r["fills"])),
        "all four tranches then a 40% stop": (VARIANTS["4 tranches | stop 40%"], lambda r: r["stopped"] and sum(f[1] == "buy" for f in r["fills"]) == 4),
        "ladder, hold (dip buys only)": (VARIANTS["4 tranches | hold"], lambda r: sum(f[1] == "buy" for f in r["fills"]) >= 3),
        "all-in with trim and stop": (VARIANTS["all-in now | trim + stop 30%"], lambda r: len(r["fills"]) >= 2),
        "ladder, trim only": (VARIANTS["4 tranches | trim +20%"], lambda r: any(f[1] == "sell" for f in r["fills"])),
    }
    mismatches, wrong_differs, value_gaps = 0, False, []
    for label, (plan, pred) in wanted.items():
        pick = next((t for t in scan if pred(sim_plan(o, h, low, c, t, plan, 365, engine_sizing=True))), None)
        assert pick is not None, f"no start day found for case: {label}"
        r = sim_plan(o, h, low, c, pick, plan, 365, engine_sizing=True)
        wrong = sim_plan(o, h, low, c, pick, plan, 365, engine_sizing=True, lag=1)
        res, strat = run_engine(days[pick], plan, 365)
        eng, pan = norm(strat.fills), norm([(days[i].normalize(), s, p) for i, s, p in r["fills"]])
        same = len(eng) == len(pan) and all(a[:2] == b[:2] and abs(a[2] - b[2]) <= 1e-6 * b[2] for a, b in zip(eng, pan))
        mismatches += not same
        wrong_differs |= norm([(days[i].normalize(), s, p) for i, s, p in wrong["fills"]]) != eng
        print(f"  [{label}] start {days[pick].date()}: engine {len(eng)} fills, pandas {len(pan)} fills -> {'SAME' if same else 'DIFFERENT'}")
        for a in eng:
            print(f"      {a[0]} {a[1]:<4} {a[2]:>12,.0f}")
        if not same:
            print("      pandas:", pan)
    print(f"  cases that differ: {mismatches}; the deliberately wrong (1-day-late stop) model differs from the engine somewhere: {wrong_differs}")
    assert mismatches == 0 and wrong_differs, "engine check failed"


def main():
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 40)
    px = load_bitkub("1D")
    px.index = px.index.normalize()
    px = px[~px.index.duplicated(keep="last")]
    o, h, low, c = (px[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sma200 = px["close"].rolling(200).mean().to_numpy(float)
    n = len(c)
    first = int(np.searchsorted(px.index, pd.Timestamp("2019-01-01", tz="UTC")))
    since21 = int(np.searchsorted(px.index, pd.Timestamp("2021-06-01", tz="UTC")))
    print(f"BTC/THB {px.index[0].date()}..{px.index[-1].date()}, {n} days")

    parity(px, o, h, low, c)

    for horizon in (91, 365):
        starts_all = list(range(first, n - horizon))
        sets = {
            "ALL purchase days 2019-2025": starts_all,
            "purchase days since mid-2021": [t for t in starts_all if t >= since21],
            "purchase days when price was above its 200-day average (like today)": [t for t in starts_all if c[t - 1] > sma200[t - 1]],
        }
        for label, starts in sets.items():
            print(f"\n=== {horizon}-DAY horizon, {label}: {len(starts)} days (~{len(starts) / horizon:.1f} independent periods). Result = value of 1 THB of capital at the end, net of fees ===")
            print(table(study(o, h, low, c, starts, horizon)).to_string(index=False))

    last = c[-1]
    print(f"\n=== the plan in THB for a start today (P0 = last close {last:,.0f} on {px.index[-1].date()}; 10,000 THB = 4 x 2,500) ===")
    for d in DIPS:
        print(f"  limit buy 2,500 THB at {last * (1 - d):,.0f} (-{d:.0%})")
    print(f"  limit sell one third of the BTC held at {last * (1 + TRIM_GAIN):,.0f} (+{TRIM_GAIN:.0%})")
    for s in (0.30, 0.40):
        print(f"  stop {s:.0%}: if a daily close is at or below {last * (1 - s):,.0f}, sell everything at the next open")

    print("\n=== one illustrative run drawn by LumiBot: start 2025-10-06 (the all-time-high week), full plan, 365 days ===")
    plan = VARIANTS["4 tranches | trim + stop 30%"]
    res, strat = run_engine(pd.Timestamp("2025-10-06"), plan, 365, chart=True)
    print("fills:", [(str(d.date()), s, f"{p:,.0f}") for d, s, p in strat.fills])
    stats = {k: (float(v["drawdown"]) if k == "max_drawdown" else float(v)) for k, v in (res or {}).items() if k in ("total_return", "max_drawdown", "volatility")}
    print("LumiBot stats:", {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main()
