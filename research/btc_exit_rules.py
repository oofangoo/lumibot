"""E020: if you hold BTC bought on any past day, which sell rules would have covered you over the next year? (real Bitkub BTC/THB daily candles)

Bitkub only has market and limit orders (no stop, no OCO), so a rule is one of: a resting limit sell above the market (take-profit), or a market sell the
morning after a daily close breaks a level. Rules are fixed here, before looking at results, and all are reported:
  none | fixed stop 10/15/20/25/30% under the entry price | trailing stop 15/20/25/30% under the highest close since entry | sell below the 200-day average |
  take-profit limits (half at +50%; thirds at +30/+60/+100%) alone and with a 25% trailing stop on what is left.
A purchase on day t is filled at that day's open; every close from day t on can trigger a sell at the next open; a sold position stays in cash until the
year ends (no re-entry). Fee 0.25% each way, no slippage. Take-profit limits fill at their price when the day's high reaches it.
Engine check: the stop logic is also run as a LumiBot Strategy through BitkubBacktesting and its fills must equal the pandas model's (cross-start-day
statistics have no LumiBot equivalent, so those stay in pandas).
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\btc_exit_rules.py
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
HORIZON = 365
RULES = {
    "none (hold)": {},
    "fixed stop 10%": {"fixed": 0.10}, "fixed stop 15%": {"fixed": 0.15}, "fixed stop 20%": {"fixed": 0.20},
    "fixed stop 25%": {"fixed": 0.25}, "fixed stop 30%": {"fixed": 0.30},
    "trailing stop 15%": {"trail": 0.15}, "trailing stop 20%": {"trail": 0.20}, "trailing stop 25%": {"trail": 0.25}, "trailing stop 30%": {"trail": 0.30},
    "sell below 200-day average": {"sma200": True},
    "take-profit: half at +50%": {"tp": [(0.50, 0.5)]},
    "take-profit: thirds at +30/+60/+100%": {"tp": [(0.30, 1 / 3), (0.60, 1 / 3), (1.00, 1 / 3)]},
    "trailing 25% + half at +50%": {"trail": 0.25, "tp": [(0.50, 0.5)]},
    "trailing 25% + thirds at +30/+60/+100%": {"trail": 0.25, "tp": [(0.30, 1 / 3), (0.60, 1 / 3), (1.00, 1 / 3)]},
}


# ------------------------------------------------------------------------------------------------ pandas model
def sim_path(o, h, c, sma200, t0, rule, fee=FEE, horizon=HORIZON, engine_sizing=False, lag=0):
    """Hold from the open of day ``t0``. Returns (value of 1 THB at day t0+horizon, fills[(day_index, side, price)], sell_price_of_full_exit or None)."""
    n = len(c)
    if engine_sizing:  # what the LumiBot strategy does: size on the last finished close, 99.5% of cash, fill at the open
        units = BUFFER * 1.0 / (c[t0 - 1] * (1 + fee))
        cash = 1.0 - units * o[t0] * (1 + fee)
    else:
        units, cash = 1.0 / (o[t0] * (1 + fee)), 0.0
    entry, peak = o[t0], o[t0]
    base_units = units
    fills, tp_left, exit_price = [(t0, "buy", o[t0])], list(rule.get("tp", [])), None
    end = min(t0 + horizon, n - 1)
    for idx in range(t0, end):
        for gain, frac in list(tp_left):  # resting limit sells fill when the day's high reaches the level
            level = entry * (1 + gain)
            if units > 0 and h[idx] >= level:
                sell = min(units, base_units * frac)
                cash += sell * level * (1 - fee)
                units -= sell
                tp_left.remove((gain, frac))
                fills.append((idx, "tp", level))
        peak = max(peak, c[idx])
        # decisions use closes known `lag` days ago (lag=0 is the normal rule; lag=1 is the deliberately wrong variant used to prove the engine check can fail)
        ref = idx - lag
        trigger = False
        if units > 0 and ref >= t0:
            if "fixed" in rule and c[ref] <= entry * (1 - rule["fixed"]):
                trigger = True
            if "trail" in rule and c[ref] <= max(entry, c[t0: ref + 1].max()) * (1 - rule["trail"]):
                trigger = True
            if rule.get("sma200") and not np.isnan(sma200[ref]) and c[ref] < sma200[ref]:
                trigger = True
        if trigger and units > 0 and idx + 1 <= end:
            cash += units * o[idx + 1] * (1 - fee)
            fills.append((idx + 1, "sell", o[idx + 1]))
            exit_price, units = o[idx + 1], 0.0
            break
    final = cash + units * c[end] * (1 - fee)
    return final, fills, exit_price


def study(o, h, c, sma200, starts):
    """For every start day and rule: final multiple, whether the whole position was sold early, and whether BTC ended the year higher than that sale price."""
    out = {}
    for name, rule in RULES.items():
        mult, stopped, regret = [], [], []
        for t0 in starts:
            final, fills, exit_price = sim_path(o, h, c, sma200, t0, rule)
            mult.append(final)
            stopped.append(exit_price is not None)
            regret.append(exit_price is not None and c[min(t0 + HORIZON, len(c) - 1)] > exit_price)
        out[name] = (np.array(mult), np.array(stopped), np.array(regret))
    return out


def table(res, base="none (hold)"):
    rows = []
    for name, (m, stopped, regret) in res.items():
        rows.append({"rule": name, "worst": f"{(m.min() - 1) * 100:+.0f}%", "5th pct": f"{(np.percentile(m, 5) - 1) * 100:+.0f}%", "10th pct": f"{(np.percentile(m, 10) - 1) * 100:+.0f}%",
                     "median": f"{(np.median(m) - 1) * 100:+.0f}%", "mean": f"{(m.mean() - 1) * 100:+.0f}%", "ended below start": f"{(m < 1).mean() * 100:.0f}%",
                     "sold early": f"{stopped.mean() * 100:.0f}%", "...and BTC was higher at year end": f"{(regret.sum() / max(stopped.sum(), 1)) * 100:.0f}%" if stopped.any() else "-"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ LumiBot engine strategy (stop logic only)
class StopExit(Strategy):
    """Buys at the first iteration, then sells everything the morning after a daily close breaks the stop."""
    parameters = {"kind": "trail", "x": 0.25}

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.fills, self.entry, self.peak, self.bought, self.done = [], None, None, False, False

    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 3, "day", quote=THB)
        close = float(bars.df["close"].iloc[-1])
        if not self.bought:
            self.submit_order(self.create_order(BTC, BUFFER * self.get_cash() / (close * (1 + FEE)), "buy", quote=THB))
            self.bought = True
            return
        if self.entry is None or self.done:
            return
        self.peak = max(self.peak, close)
        kind, x = self.parameters["kind"], self.parameters["x"]
        trigger = close <= (self.peak if kind == "trail" else self.entry) * (1 - x)
        if trigger and self.get_position(BTC) is not None:
            self.submit_order(self.create_order(BTC, self.get_position(BTC).quantity, "sell", quote=THB))
            self.done = True

    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.fills.append((pd.Timestamp(self.get_datetime()).tz_convert("UTC").normalize(), str(order.side), float(price)))
        if self.entry is None and str(order.side) == "buy":
            self.entry = self.peak = float(price)


def engine_fills(start_day, kind, x):
    fees = [TradingFee(percent_fee=FEE)]
    start = datetime(start_day.year, start_day.month, start_day.day)
    _, strat = StopExit.run_backtest(
        BitkubBacktesting, backtesting_start=start, backtesting_end=start + timedelta(days=HORIZON), cache_dir=CACHE, quote_asset=THB, benchmark_asset=None,
        budget=100_000, buy_trading_fees=fees, sell_trading_fees=fees, parameters={"kind": kind, "x": x}, analyze_backtest=False, show_plot=False,
        show_tearsheet=False, save_tearsheet=False, save_logfile=False, show_progress_bar=False, quiet_logs=True, show_indicators=False, save_stats_file=False)
    return strat.fills


def parity(px, o, h, c, sma200):
    print("\n=== LumiBot engine (BitkubBacktesting) vs the pandas model: same fills? ===")
    days = list(px.index)
    cases = [("2021-11-10", "trail", 0.25), ("2025-10-06", "trail", 0.25), ("2023-01-06", "trail", 0.25), ("2021-11-10", "fixed", 0.20),
             ("2024-03-13", "fixed", 0.15), ("2022-05-01", "trail", 0.20)]
    mismatches, any_stop, wrong_variant_differs = 0, False, False
    for day, kind, x in cases:
        t0 = days.index(pd.Timestamp(day, tz="UTC"))
        rule = {kind: x}
        _, fills, _ = sim_path(o, h, c, sma200, t0, rule, engine_sizing=True)
        _, wrong, _ = sim_path(o, h, c, sma200, t0, rule, engine_sizing=True, lag=1)
        eng = engine_fills(pd.Timestamp(day), kind, x)
        pandas_fills = [(days[i].normalize(), side if side != "tp" else "sell", price) for i, side, price in fills]
        same = len(eng) == len(pandas_fills) and all(a[0] == b[0] and a[1] == b[1] and abs(a[2] - b[2]) <= 1e-6 * b[2] for a, b in zip(eng, pandas_fills))
        any_stop |= len(eng) > 1
        wrong_f = [(days[i].normalize(), side, price) for i, side, price in wrong]
        wrong_variant_differs |= (len(wrong_f) != len(eng)) or any(a[0] != b[0] for a, b in zip(eng, wrong_f))
        mismatches += not same
        print(f"  start {day} {kind} {x:.0%}: engine fills {len(eng)} {[f'{d.date()} {s} {p:,.0f}' for d, s, p in eng]} | pandas {'SAME' if same else 'DIFFERENT'}")
    print(f"  cases that differ: {mismatches}; at least one case triggered a sell: {any_stop}; the deliberately wrong (1-day-late) model differs from the engine somewhere: {wrong_variant_differs}")
    assert mismatches == 0 and any_stop and wrong_variant_differs, "engine check failed"


def main():
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    px = load_bitkub("1D")
    px.index = px.index.normalize()
    px = px[~px.index.duplicated(keep="last")]
    o, h, c = px["open"].to_numpy(float), px["high"].to_numpy(float), px["close"].to_numpy(float)
    sma200 = px["close"].rolling(200).mean().to_numpy(float)
    n = len(c)
    first = int(np.searchsorted(px.index, pd.Timestamp("2019-01-01", tz="UTC")))
    since21 = int(np.searchsorted(px.index, pd.Timestamp("2021-06-01", tz="UTC")))
    starts_all = list(range(first, n - HORIZON))
    print(f"BTC/THB {px.index[0].date()}..{px.index[-1].date()}; {len(starts_all)} possible purchase days with a full year ahead (2019-01 .. {px.index[n - 1 - HORIZON].date()})")

    parity(px, o, h, c, sma200)

    sets = {
        "ALL purchase days 2019-2025": starts_all,
        "purchase days since mid-2021": [t for t in starts_all if t >= since21],
        "purchase days when price was above its 200-day average (like today)": [t for t in starts_all if c[t - 1] > sma200[t - 1]],
    }
    for label, starts in sets.items():
        print(f"\n=== {label}: {len(starts)} days (~{len(starts) / HORIZON:.1f} independent years). Result = value of the position after one year vs 10,000 THB put in (net of fees) ===")
        print(table(study(o, h, c, sma200, starts)).to_string(index=False))

    last = c[-1]
    print(f"\n=== what each rule means in THB for a purchase at today's price {last:,.0f} (10,000 THB = {10000 / last:.5f} BTC) ===")
    for name, rule in RULES.items():
        parts = []
        if "fixed" in rule:
            parts.append(f"sell if a daily close is at or below {last * (1 - rule['fixed']):,.0f}")
        if "trail" in rule:
            parts.append(f"sell if a daily close is {rule['trail']:.0%} under the highest close since buying (starts at {last * (1 - rule['trail']):,.0f})")
        if rule.get("sma200"):
            parts.append(f"sell if a daily close is under the 200-day average (now {sma200[-1]:,.0f}, {sma200[-1] / last - 1:+.0%})")
        for g, f in rule.get("tp", []):
            parts.append(f"limit sell {f:.0%} of the position at {last * (1 + g):,.0f} (+{g:.0%})")
        print(f"  {name:<42} " + "; ".join(parts) if parts else f"  {name:<42} nothing")


if __name__ == "__main__":
    main()
