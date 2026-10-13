"""E021 chart: the next 91 days of BTC/THB as the forecast models see them, drawn with LumiBot's own chart engine.

A LumiBot Strategy runs over real Bitkub BTC/THB daily candles (BitkubBacktesting) and draws with Strategy.add_line / add_marker (dt= for the future dates,
plot_name= for subplots); LumiBot's plot_indicators writes the Plotly page (logs/Btc3mForecast_*_indicators.html). The model numbers come from
research/btc_3m_scenarios.py (logs/btc_3m_forecast.json); the strategy refuses to draw if its last close differs from the one the models used.
Panel 1: price with the 50/200-day averages and the spread of outcomes (10th/50th/90th percentile) of the two best-calibrated models (BOOT0 = zero drift,
BOOT = with the past bull-market drift) and the volatility-clustering model (FHS) that the walk-forward test found too narrow.
Panel 2: chance that the price has already dipped 15% / 25% / 35% below today's price at some time by day N (BOOT0). Not a forecast of what will happen.
The strategy buys and holds from the start so LumiBot's tearsheet describes the last 19 months of plain buy-and-hold.

Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\btc_3m_chart.py   (after btc_3m_scenarios.py)
"""
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["LUMIBOT_DISABLE_UI"] = "true"

import lumibot  # noqa: E402

if "lumibot-dev" not in str(Path(lumibot.__file__)):
    sys.exit("This chart needs the Bitkub-enabled LumiBot (lumibot-dev). Set PYTHONPATH to the dev clone; see the docstring.")

from lumibot.backtesting import BitkubBacktesting  # noqa: E402
from lumibot.entities import TradingFee  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import BTC, BUFFER, CACHE, THB  # noqa: E402

PRICE, DIP = "1. BTC price, THB (log)", "2. Chance it has dipped by day N (%)"
FORECAST = Path("logs") / "btc_3m_forecast.json"
MODEL_STYLE = {  # model -> (label, colors for p10/p50/p90, dash)
    "BOOT0": ("Zero-drift bootstrap", ("#ff8787", "#f1f3f5", "#69db7c"), "solid"),
    "BOOT": ("Bootstrap with past bull drift", ("#ffa94d", "#ced4da", "#38d9a9"), "dashed"),
    "FHS": ("Volatility-clustering sim (too narrow in tests)", ("#b197fc", "#868e96", "#b197fc"), "dotted"),
}


class Btc3mForecast(Strategy):
    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.dates, self.closes = [], []
        self.summary = {}

    def on_trading_iteration(self):
        bars = self.get_historical_prices(BTC, 5, "day", quote=THB)
        if bars is None or bars.df.empty:
            return
        stamp, close = bars.df.index[-1], float(bars.df["close"].iloc[-1])
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")  # keep Bitkub's own UTC days
        if self.dates and stamp <= self.dates[-1]:
            return
        self.dates.append(stamp)
        self.closes.append(close)
        self.last_iteration = self.get_datetime()
        self.add_line("BTC/THB close", close, color="#f1f3f5", width=2, plot_name=PRICE, dt=stamp)
        for length, color, style in ((50, "#4dabf7", "dotted"), (200, "#ffa94d", "dashed")):
            value = self.indicators.sma(BTC, length=length, quote=THB)
            if value is not None and math.isfinite(float(value)):
                self.add_line(f"{length}-day average", float(value), color=color, style=style, width=2, plot_name=PRICE, dt=stamp)
        if len(self.dates) == 1:
            self.submit_order(self.create_order(BTC, BUFFER * self.get_cash() / (close * 1.0025), "buy", quote=THB))

    def on_strategy_end(self):
        fc = json.loads(FORECAST.read_text(encoding="utf-8"))
        today, last = self.dates[-1], self.closes[-1]
        if str(today.date()) != fc["date"] or abs(last - fc["last_close"]) > 1:
            raise RuntimeError(f"LumiBot saw {today.date()} close {last:,.0f}, the models used {fc['date']} close {fc['last_close']:,.0f}: rerun btc_3m_scenarios.py")
        self.add_marker("Today", float(last), color="#ffd43b", symbol="diamond", size=14, plot_name=PRICE, dt=today, detail_text=f"{today.date()}: {last:,.0f} THB")
        for model, (label, (c_lo, c_mid, c_hi), style) in MODEL_STYLE.items():
            fan = fc["models"][model]["fan"]["price"]  # per day: [p10, p25, p50, p75, p90]
            for d in range(0, len(fan), 2):
                when = today + pd.Timedelta(days=d)
                for name, idx, color in ((f"{label}: bad case (10th pct)", 0, c_lo), (f"{label}: typical (median)", 2, c_mid), (f"{label}: good case (90th pct)", 4, c_hi)):
                    self.add_line(name, float(fan[d][idx]), color=color, style=style, width=3 if model == "BOOT0" else 2, plot_name=PRICE, dt=when,
                                  detail_text=f"{(fan[d][idx] / last - 1) * 100:+.0f}% from today after {d} days (before fee; model, not a prediction)")
        end = today + pd.Timedelta(days=fc["horizon_days"])
        for key, label in (("05", "5th"), ("25", "25th"), ("75", "75th"), ("95", "95th")):
            self.add_marker(f"All-model average, day 91, {label} percentile", float(fc["ens_price_quantiles"][key]), color="#ffd43b", symbol="circle-open", size=10, plot_name=PRICE, dt=end,
                            detail_text=f"All-model average, {label} percentile: {fc['ens_price_quantiles'][key]:,.0f} THB")
        curves = fc["models"]["BOOT0"]["dip_curve"]
        for pct, color in (("15", "#ffd43b"), ("25", "#ffa94d"), ("35", "#ff6b6b")):
            for d, p in enumerate(curves[pct]):
                if d % 2 == 0:
                    self.add_line(f"Dipped {pct}% below today's price (zero-drift model)", p * 100, color=color, width=3, plot_name=DIP, dt=today + pd.Timedelta(days=d),
                                  detail_text=f"{p * 100:.0f}% of simulated paths had been {pct}% or more below today's price by day {d}")
        self.summary = {"today": str(today.date()), "last": float(last), "n_days": len(self.closes), "last_iteration": str(self.last_iteration)}


def main():
    fees = [TradingFee(percent_fee=0.0025)]
    results, strat = Btc3mForecast.run_backtest(
        BitkubBacktesting, backtesting_start=datetime(2025, 3, 1, 21, 0), backtesting_end=datetime(2026, 10, 15), cache_dir=CACHE, quote_asset=THB, benchmark_asset=None,
        budget=100_000, buy_trading_fees=fees, sell_trading_fees=fees, name="Btc3mForecast",
        analyze_backtest=True, show_plot=True, show_tearsheet=False, save_tearsheet=True, save_logfile=False,
        show_progress_bar=False, quiet_logs=True, show_indicators=True, save_stats_file=False,
    )
    print("recorded", strat.summary)
    stats = {k: (float(v["drawdown"]) if k == "max_drawdown" else float(v)) for k, v in (results or {}).items() if k in ("cagr", "total_return", "max_drawdown", "volatility", "sharpe")}
    print("LumiBot buy-and-hold stats since 2025-03:", {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    main()
