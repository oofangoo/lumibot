"""E019: the BTC boom-bust pattern and where this cycle stands, drawn with LumiBot's own chart engine.

A LumiBot Strategy runs over real Bitkub BTC/THB daily candles (BitkubBacktesting). Every day it records the close, the 50/200-day averages
(self.indicators) and the fall from the all-time high with Strategy.add_line; when the run ends it adds the cycle tops/lows (add_marker), the
price levels that matter, a one-year spread of outcomes taken from every past start day (add_line with future dates), and the last cycles
replayed on top of the current one (add_line with shifted dates). LumiBot's plot_indicators then writes the Plotly page
(logs/BtcCycleChart_*_indicators.html). Nothing here is a forecast: the future lines are what past years did, applied to today's price.
The strategy also buys and holds BTC from the start so LumiBot's trades output and tearsheet describe plain buy-and-hold.

Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\btc_cycle_chart.py
"""
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["LUMIBOT_DISABLE_UI"] = "true"  # write the HTML files, never open a browser

import lumibot  # noqa: E402

if "lumibot-dev" not in str(Path(lumibot.__file__)):
    sys.exit("This chart needs the Bitkub-enabled LumiBot (lumibot-dev). Set PYTHONPATH to the dev clone; see the docstring.")

from lumibot.backtesting import BitkubBacktesting  # noqa: E402
from lumibot.entities import TradingFee  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import BTC, BUFFER, CACHE, THB  # noqa: E402

PRICE, DROP, REPLAY = "1. BTC price, THB (log)", "2. Fall from the all-time high (%)", "3. Cycle replay (% from top)"  # "(log)": LumiBot draws a log axis
CYCLE_GAP_DAYS = 365  # a "cycle top" is an all-time high that stood for a year or more (or still stands)
FAN_DAYS, FAN_STEP = 365, 5
FAN_SINCE = pd.Timestamp("2021-06-01")  # the modern sample: after BTC's 2020-21 run-up
REPLAY_DAYS = 450  # the past lows came after 264 and 422 days; a little beyond shows how each recovery began


def cycle_tops(dates, closes, gap_days=CYCLE_GAP_DAYS):
    """Index of every all-time high that was not beaten for ``gap_days`` (or has not been beaten yet): the top of a boom-bust cycle."""
    running = np.maximum.accumulate(closes)
    highs = [i for i in range(len(closes)) if closes[i] >= running[i]]
    tops = []
    for k, i in enumerate(highs):
        next_high = highs[k + 1] if k + 1 < len(highs) else len(closes) - 1
        if (dates[next_high] - dates[i]).days >= gap_days:
            tops.append(i)
    return tops


def low_after_top(dates, closes, top):
    """Index of the lowest close between a cycle top and the next time price made a new high (or the end of the data)."""
    stop = len(closes)
    beaten = np.nonzero(closes[top + 1:] > closes[top])[0]
    if len(beaten):
        stop = top + 1 + int(beaten[0])
    return top + int(np.argmin(closes[top:stop]))


def fan_percentiles(dates, closes, since=FAN_SINCE, days=FAN_DAYS, step=FAN_STEP, pcts=(10, 50, 90)):
    """For each day d in 0..days: percentiles of price(t+d)/price(t)-1 over every start day t on or after ``since`` that has ``days`` of future."""
    since = pd.Timestamp(since)
    if dates.tz is not None and since.tzinfo is None:
        since = since.tz_localize(dates.tz)  # LumiBot stamps candles in UTC; compare like with like
    starts = np.array([i for i in range(len(closes) - days) if dates[i] >= since])
    out = {}
    for d in range(0, days + 1, step):
        ratio = closes[starts + d] / closes[starts] - 1
        out[d] = np.percentile(ratio, pcts)
    return out, len(starts)


def year_outcomes(dates, closes, new_high_gain, since, days=FAN_DAYS):
    """Share of start days (on or after ``since``) whose price a year later was: down more than 20%, between -20% and the old high, above the old high."""
    since = pd.Timestamp(since)
    if dates.tz is not None and since.tzinfo is None:
        since = since.tz_localize(dates.tz)
    starts = np.array([i for i in range(len(closes) - days) if dates[i] >= since])
    ratio = closes[starts + days] / closes[starts] - 1
    return {"down_20": float((ratio <= -0.20).mean()), "middle": float(((ratio > -0.20) & (ratio <= new_high_gain)).mean()),
            "new_high": float((ratio > new_high_gain).mean()), "start_days": int(len(starts)), "about_independent_years": round(len(starts) / days, 1)}


class BtcCycleChart(Strategy):
    """Buys BTC/THB once and holds it, recording the chart as it goes."""

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.dates, self.closes, self.peak = [], [], 0.0
        self.summary = {}

    # ------------------------------------------------------------------ every day: the facts LumiBot's data source shows us
    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 5, "day", quote=THB)
        if bars is None or bars.df.empty:
            return
        stamp, close = bars.df.index[-1], float(bars.df["close"].iloc[-1])
        # LumiBot stamps candles in New York time, which puts a 00:00 UTC candle on the previous calendar day; keep Bitkub's own UTC days
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        if self.dates and stamp <= self.dates[-1]:
            return  # no new finished candle yet
        self.dates.append(stamp)
        self.closes.append(close)
        self.peak = max(self.peak, close)

        self.add_line("BTC/THB close", close, color="#f1f3f5", width=2, plot_name=PRICE, dt=stamp)
        for length, color, style in ((50, "#4dabf7", "dotted"), (200, "#ffa94d", "dashed")):
            value = self.indicators.sma(BTC, length=length, quote=THB)
            if value is not None and math.isfinite(float(value)):
                self.add_line(f"{length}-day average", float(value), color=color, style=style, width=2, plot_name=PRICE, dt=stamp)
        self.add_line("Below the all-time high (%)", (close / self.peak - 1) * 100, color="#ff8787", width=2, plot_name=DROP, dt=stamp)

        if self.get_position(BTC) is None and len(self.dates) == 1:
            self.submit_order(self.create_order(BTC, BUFFER * self.get_cash() / (close * 1.0025), "buy", quote=THB))

    # ------------------------------------------------------------------ at the end: what the whole history says
    def on_strategy_end(self):
        dates = pd.DatetimeIndex(self.dates)
        closes = np.array(self.closes)
        today, last = dates[-1], closes[-1]
        tops = cycle_tops(dates, closes)
        cycles = []
        for i in tops:
            low = low_after_top(dates, closes, i)
            beaten = np.nonzero(closes[i + 1:] > closes[i])[0]  # first close above the old top
            back = i + 1 + int(beaten[0]) if len(beaten) else None
            cycles.append({"top": i, "low": low, "depth": closes[low] / closes[i] - 1, "days_to_low": (dates[low] - dates[i]).days,
                           "top_day": dates[i].date(), "low_day": dates[low].date(),
                           "back_day": dates[back].date() if back is not None else None, "days_to_back": (dates[back] - dates[i]).days if back is not None else None})
        now_cycle, past_cycles = cycles[-1], cycles[:-1]
        top_price = closes[now_cycle["top"]]

        # cycle tops and lows: on the price panel, and the depth of each fall on the fall-from-high panel
        for c in cycles:
            running_top = closes[: c["top"] + 1].max()
            low_text = f"{c['depth'] * 100:.0f}% below the top, {c['days_to_low']} days after it, on {c['low_day']}" + (" (lowest so far)" if c is now_cycle else "")
            self.add_marker("Cycle top", closes[c["top"]], color="#ff6b6b", symbol="triangle-down", size=14, plot_name=PRICE, dt=dates[c["top"]],
                            detail_text=f"Cycle top {c['top_day']}")
            self.add_marker("Cycle low", closes[c["low"]], color="#69db7c", symbol="triangle-up", size=14, plot_name=PRICE, dt=dates[c["low"]], detail_text=low_text)
            self.add_marker("Deepest fall of each cycle", (closes[c["low"]] / running_top - 1) * 100, color="#69db7c", symbol="triangle-up", size=14,
                            plot_name=DROP, dt=dates[c["low"]], detail_text=low_text)
        self.add_marker("Today", float(last), color="#ffd43b", symbol="diamond", size=14, plot_name=PRICE, dt=today,
                        detail_text=f"{today.date()}: {last:,.0f} THB, {(last / top_price - 1) * 100:.0f}% from the high")

        # price levels that matter, drawn from this cycle's top to a year past today
        end = today + pd.Timedelta(days=FAN_DAYS)
        levels = [("All-time high", top_price, "#adb5bd", "dotted", f"Highest close, {now_cycle['top_day']}"),
                  (f"Last low ({now_cycle['depth'] * 100:.0f}% from the high)", closes[now_cycle["low"]], "#69db7c", "dotted", f"Lowest close since the high, {now_cycle['low_day']}")]
        for c in past_cycles:
            levels.append((f"If it fell like the {c['top_day'].year} cycle ({c['depth'] * 100:.0f}%)", top_price * (1 + c["depth"]), "#da77f2", "dashed",
                           f"Price if this cycle fell as far from its top as the cycle topping on {c['top_day']} did"))
        for name, y, color, style, text in levels:
            for x in (dates[now_cycle["top"]], end):
                self.add_line(name, float(y), color=color, style=style, width=1, plot_name=PRICE, dt=x, detail_text=text)

        # a year ahead, the way past years went: every start day since mid-2021 (weak = 10th percentile, strong = 90th) applied to today's price
        fan, n_starts = fan_percentiles(dates, closes)
        all_history, _ = fan_percentiles(dates, closes, since=dates[0])
        for d, (lo, mid, hi) in fan.items():
            when = today + pd.Timedelta(days=d)
            for name, ratio, color, style, width in (("Weak year (10th percentile since mid-2021)", lo, "#ff6b6b", "solid", 3),
                                                     ("Typical year (median since mid-2021)", mid, "#ced4da", "solid", 3),
                                                     ("Strong year (90th percentile since mid-2021)", hi, "#51cf66", "solid", 3)):
                self.add_line(name, float(last * (1 + ratio)), color=color, style=style, width=width, plot_name=PRICE, dt=when,
                              detail_text=f"{ratio * 100:+.0f}% from today after {d} days (past years, not a forecast)")
            self.add_line("Typical year (median, all history)", float(last * (1 + all_history[d][1])), color="#868e96", style="dotted", width=2, plot_name=PRICE,
                          dt=when, detail_text=f"{all_history[d][1] * 100:+.0f}% from today after {d} days (all past years, not a forecast)")

        # the last cycles replayed from their tops onto this cycle's top date
        anchor = dates[now_cycle["top"]]
        palette = ["#74c0fc", "#b197fc", "#ffa8a8"]
        for k, c in enumerate(past_cycles):
            shift = anchor - dates[c["top"]]
            for j in range(0, REPLAY_DAYS + 1, 3):
                idx = c["top"] + j
                if idx >= len(closes):
                    break
                self.add_line(f"{c['top_day'].year} cycle, replayed from its top", float((closes[idx] / closes[c["top"]] - 1) * 100), color=palette[k % 3], width=2,
                              plot_name=REPLAY, dt=dates[idx] + shift, detail_text=f"Real date {dates[idx].date()}, day {j} after the top")
        for j in range(0, len(closes) - now_cycle["top"], 1):
            idx = now_cycle["top"] + j
            self.add_line("This cycle (top Oct 2025)", float((closes[idx] / top_price - 1) * 100), color="#ffd43b", width=4, plot_name=REPLAY, dt=dates[idx],
                          detail_text=f"Day {j} after the top")
        self.add_marker("Today (replay)", float((last / top_price - 1) * 100), color="#ffd43b", symbol="diamond", size=16, plot_name=REPLAY, dt=today,
                        detail_text=f"Day {(today - anchor).days} after the top")
        for c in past_cycles:
            self.add_marker("Past cycle low", float(c["depth"] * 100), color="#69db7c", symbol="triangle-up", size=14, plot_name=REPLAY,
                            dt=anchor + (dates[c["low"]] - dates[c["top"]]),
                            detail_text=f"{c['top_day'].year} cycle low: {c['depth'] * 100:.0f}% on day {c['days_to_low']} (real date {c['low_day']})")
        self.add_marker("This cycle's low so far", float(now_cycle["depth"] * 100), color="#69db7c", symbol="triangle-up", size=14, plot_name=REPLAY,
                        dt=dates[now_cycle["low"]], detail_text=f"{now_cycle['depth'] * 100:.0f}% on day {now_cycle['days_to_low']}")

        self.summary = {
            "today": str(today.date()), "last_close": float(last), "top_price": float(top_price), "n_days": len(closes), "fan_start_days": n_starts,
            "cycles": [{"top_day": str(c["top_day"]), "low_day": str(c["low_day"]), "depth": float(c["depth"]), "days_to_low": int(c["days_to_low"]),
                        "top_price": float(closes[c["top"]]), "low_price": float(closes[c["low"]]),
                        "back_day": str(c["back_day"]) if c["back_day"] else None, "days_to_back": c["days_to_back"]} for c in cycles],
            "fan": {int(d): [float(last * (1 + r)) for r in v] for d, v in fan.items() if d in (0, 90, 180, 270, 365)},
            "fan_pct_change": {int(d): [float(r) for r in v] for d, v in fan.items() if d in (0, 90, 180, 270, 365)},
            "all_history_median_365": float(all_history[FAN_DAYS][1]),
            "sma50": float(closes[-50:].mean()), "sma200": float(closes[-200:].mean()), "new_high_gain": float(top_price / last - 1),
            "outcomes_since_mid_2021": year_outcomes(dates, closes, top_price / last - 1, FAN_SINCE),
            "outcomes_all_history": year_outcomes(dates, closes, top_price / last - 1, dates[0]),
        }


def main():
    end = datetime(2026, 10, 11)  # the last finished candle is 2026-10-09 UTC; an iteration on 10-10 is the first that sees it
    fees = [TradingFee(percent_fee=0.0025)]
    results, strat = BtcCycleChart.run_backtest(
        BitkubBacktesting, backtesting_start=datetime(2018, 6, 1), backtesting_end=end, cache_dir=CACHE, quote_asset=THB, benchmark_asset=None,
        budget=100_000, buy_trading_fees=fees, sell_trading_fees=fees, name="BtcCycleChart",
        analyze_backtest=True, show_plot=True, show_tearsheet=False, save_tearsheet=True, save_logfile=False,
        show_progress_bar=False, quiet_logs=True, show_indicators=True, save_stats_file=False,
    )
    s = strat.summary
    print(f"\nrecorded {s['n_days']} daily candles through {s['today']}; last close {s['last_close']:,.0f} THB; top {s['top_price']:,.0f} THB")
    for c in s["cycles"]:
        print(f"  cycle top {c['top_day']} ({c['top_price']:,.0f}) -> low {c['low_day']} ({c['low_price']:,.0f}): {c['depth'] * 100:.0f}% after {c['days_to_low']} days")
    print(f"  one-year spread from today ({s['fan_start_days']} start days since mid-2021), THB / change:")
    for d in (90, 180, 365):
        print(f"    +{d:>3}d: " + "  ".join(f"{v:,.0f} ({p * 100:+.0f}%)" for v, p in zip(s["fan"][d], s["fan_pct_change"][d])))
    print(f"  all-history median after 365 days: {s['all_history_median_365'] * 100:+.0f}%")
    stats = {k: (float(v["drawdown"]) if k == "max_drawdown" else float(v)) for k, v in (results or {}).items() if k in ("cagr", "total_return", "max_drawdown", "volatility", "sharpe")}
    print("LumiBot buy-and-hold stats:", {k: round(v, 3) for k, v in stats.items()})
    out = Path("logs") / "BtcCycleChart_latest_summary.json"
    out.write_text(json.dumps({**s, "buy_and_hold": stats}, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
