"""E023 chart: today's chart next to its closest look-alikes from history, drawn with LumiBot's own chart engine.

A LumiBot Strategy runs over real daily candles (BitkubBacktesting for BTC/THB, AlpacaBacktesting for SPY), draws the price and its 200-day average with
add_line, and in on_strategy_end adds, from logs/pattern_analogs.json (written by pattern_study.py):
  panel 1  price. Each of the 10 look-alikes is rebased so its last day equals today's close and drawn over the last 60 days and the following H days (colored by
           what happened: red fell hard ... green surged); thick lines = middle and 10th/90th percentile of the look-alikes' following paths; grey dotted = the same
           for every window in history.
  panel 2  the shapes themselves (z-scored log price of the last 60 days): today in white, the look-alikes thin.
plot_indicators writes logs/PatternAnalogs<market>_*_indicators.html. The strategy refuses to draw if its last close differs from the one the study used.
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\pattern_chart.py BTC   (and SPY; after pattern_study.py)
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
os.environ["LUMIBOT_DISABLE_UI"] = "true"

import lumibot  # noqa: E402

if "lumibot-dev" not in str(Path(lumibot.__file__)):
    sys.exit("This chart needs the Bitkub-enabled LumiBot (lumibot-dev). Set PYTHONPATH to the dev clone; see the docstring.")

import pytz  # noqa: E402
from lumibot.backtesting import AlpacaBacktesting, BitkubBacktesting  # noqa: E402
from lumibot.entities import Asset, TradingFee  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import BTC, CACHE, THB  # noqa: E402

DATA = Path("logs") / "pattern_analogs.json"
COLORS = {"Falls hard": "#ff6b6b", "Weak": "#ffa94d", "Middling": "#adb5bd", "Strong": "#8ce99a", "Surges": "#38d9a9"}


class PatternChart(Strategy):
    parameters = {"market": "BTC"}

    def initialize(self):
        self.sleeptime = "1D"
        self.market_key = self.parameters["market"]
        if self.market_key == "BTC":
            self.set_market("24/7")
        self.asset = BTC if self.market_key == "BTC" else Asset("SPY")
        self.dates, self.closes, self.summary = [], [], {}
        self.panel1 = "1. Price and its closest look-alikes"
        self.panel2 = "2. The shapes (z-scored last 60 days)"

    def on_trading_iteration(self):
        kw = {"quote": THB} if self.market_key == "BTC" else {}
        bars = self.get_historical_prices(self.asset, 5, "day", **kw)
        if bars is None or bars.df.empty:
            return
        stamp, close = bars.df.index[-1], float(bars.df["close"].iloc[-1])
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        stamp = stamp.tz_localize(None).normalize()
        if self.dates and stamp <= self.dates[-1]:
            return
        self.dates.append(stamp)
        self.closes.append(close)
        self.add_line("Close", close, color="#f1f3f5", width=2, plot_name=self.panel1, dt=stamp.tz_localize("UTC"))
        value = self.indicators.sma(self.asset, length=200, **kw)
        if value is not None and math.isfinite(float(value)):
            self.add_line("200-day average", float(value), color="#ffa94d", style="dashed", width=2, plot_name=self.panel1, dt=stamp.tz_localize("UTC"))

    def on_strategy_end(self):
        d = json.loads(DATA.read_text(encoding="utf-8"))[self.market_key]
        seen, seen_close = self.dates[-1], self.closes[-1]
        recent = dict(zip(d["today"]["recent"]["dates"], d["today"]["recent"]["close"]))
        gap = (pd.Timestamp(d["last_date"]) - pd.Timestamp(seen)).days
        # LumiBot's backtest cannot see a day's close until the next iteration, which for stocks is the next session's open: it may be one session behind the study
        if str(seen.date()) not in recent or abs(seen_close / recent[str(seen.date())] - 1) > 0.005 or not 0 <= gap <= 4:
            raise RuntimeError(f"LumiBot's last candle {seen.date()} close {seen_close:,.2f} does not match the study's data (last day {d['last_date']}): rerun pattern_study.py")
        today, last = pd.Timestamp(d["last_date"]), float(d["last_close"])
        t = d["today"]
        H = d["horizon"]
        past = [pd.Timestamp(x, tz="UTC") for x in t["window_dates"]]
        if self.market_key == "BTC":
            future = [pd.Timestamp(today, tz="UTC") + pd.Timedelta(days=i) for i in range(H + 1)]
        else:
            future = [pd.Timestamp(today, tz="UTC")] + list(pd.bdate_range(today + pd.Timedelta(days=1), periods=H).tz_localize("UTC"))
        pstep, fstep = 2, 2
        for i, a in enumerate(t["analogs"], 1):
            name = f"Look-alike {i}: {a['start']} to {a['end']} ({a['fwd_return'] * 100:+.0f}% over the next {H})"
            color = COLORS[a["bucket"]]
            for j in range(0, len(past), pstep):
                self.add_line(name, float(last * math.exp(a["window_rebased"][j])), color=color, width=1, plot_name=self.panel1, dt=past[j],
                              detail_text=f"{a['start']} to {a['end']}: the look-alike, scaled to today's price")
            for k in list(range(0, H + 1, fstep)) + ([H] if H % fstep else []):
                self.add_line(name, float(last * math.exp(a["path"][k])), color=color, width=1, plot_name=self.panel1, dt=future[k],
                              detail_text=f"{a['bucket']}: {(math.exp(a['path'][k]) - 1) * 100:+.0f}% after {k} days")
        fan = np.array(t["analog_fan"])  # per day offset: p10 p25 p50 p75 p90 (log return)
        base = np.array(t["base_fan"])
        for src, label, cols, style, width in ((fan, "Look-alikes", ("#ff8787", "#ffffff", "#69db7c"), "solid", 3), (base, "All history", ("#868e96", "#868e96", "#868e96"), "dotted", 2)):
            for k in list(range(0, H + 1, fstep)) + ([H] if H % fstep else []):
                for idx, nm, col in ((0, "bad case (10th percentile)", cols[0]), (2, "middle", cols[1]), (4, "good case (90th percentile)", cols[2])):
                    self.add_line(f"{label}: {nm}", float(last * math.exp(src[k][idx])), color=col, style=style, width=width, plot_name=self.panel1, dt=future[k],
                                  detail_text=f"{(math.exp(src[k][idx]) - 1) * 100:+.0f}% after {k} days (from {len(t['analogs']) if label == 'Look-alikes' else 'every'} windows; not a prediction)")
        self.add_marker("Today", float(last), color="#ffd43b", symbol="diamond", size=14, plot_name=self.panel1, dt=pd.Timestamp(today, tz="UTC"),
                        detail_text=f"{today.date()}: {last:,.2f}")
        for j in range(0, len(past), pstep):
            self.add_line("Today's shape", float(t["current_z"][j]), color="#ffffff", width=3, plot_name=self.panel2, dt=past[j])
        for i, a in enumerate(t["analogs"], 1):
            for j in range(0, len(past), pstep):
                self.add_line(f"Shape of look-alike {i} ({a['start']})", float(a["z"][j]), color=COLORS[a["bucket"]], width=1, plot_name=self.panel2, dt=past[j],
                              detail_text=f"{a['start']} to {a['end']}; distance {a['distance']:.2f}")
        self.summary = {"today": str(today.date()), "last": float(last), "n_days": len(self.closes), "lumibot_last_candle": str(seen.date())}


def main():
    market = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
    common = dict(budget=100_000, name=f"PatternAnalogs{market}", analyze_backtest=True, show_plot=True, show_tearsheet=False, save_tearsheet=False, save_logfile=False,
                  show_progress_bar=False, quiet_logs=True, show_indicators=True, save_stats_file=False, parameters={"market": market})
    if market == "BTC":
        results, strat = PatternChart.run_backtest(BitkubBacktesting, backtesting_start=datetime(2025, 3, 1, 21, 0), backtesting_end=datetime(2026, 10, 15), cache_dir=CACHE,
                                                   quote_asset=THB, benchmark_asset=None, **common)
    else:
        ny = pytz.timezone("America/New_York")
        ret = PatternChart.backtest(AlpacaBacktesting, backtesting_start=ny.localize(datetime(2024, 9, 3)), backtesting_end=ny.localize(datetime(2026, 10, 13)),
                                               benchmark_asset=None, timestep="day", market="NYSE", config=None, warm_up_trading_days=260, auto_adjust=True, **common)
        strat = next((x for x in ret if isinstance(x, Strategy)), None)  # .backtest() returns more than (results, strategy)
    print("recorded", strat.summary if strat is not None else "(strategy object not returned; the chart guard ran inside on_strategy_end)")


if __name__ == "__main__":
    main()
