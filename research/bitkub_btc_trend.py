"""E016: does a BTC trend rule work on real Bitkub BTC/THB prices, after Bitkub's fees?

Data: real Bitkub daily candles (BTC_THB, from the exchange's public API, cached in data/bitkub_cache).
Engine: LumiBot's new BitkubBacktesting source (strategies see finished candles only; fills at the current candle's open).
Rules (fixed BEFORE running, no tuning): hold BTC while the last finished close > SMA(n) of finished closes, else hold THB cash;
all-in/all-out; fee 0.25% per side; no stops (Bitkub has none); cash earns nothing.

Three things are produced:
  1. PARITY: the engine result vs an independent pandas model on the same real candles (validates the data source + engine).
  2. GRID: SMA length x signal lag x fee x window, on the pandas model (fast), plus buy-and-hold.
  3. CONTEXT: the same rule on BTC-USD (Yahoo) with the same fee, to isolate what the THB price series changes.

Run (uses the NEW LumiBot from the dev clone; the live bot's LumiBot is untouched):
    $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\bitkub_btc_trend.py
"""
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import lumibot  # noqa: E402

if "lumibot-dev" not in str(Path(lumibot.__file__)):
    sys.exit("This experiment needs the Bitkub-enabled LumiBot (lumibot-dev). Set PYTHONPATH to the dev clone; see the docstring.")

from lumibot.backtesting import BitkubBacktesting  # noqa: E402
from lumibot.entities import Asset, TradingFee  # noqa: E402
from lumibot.strategies import Strategy  # noqa: E402
from lumibot.tools.bitkub_history import BitkubHistory  # noqa: E402

WS = Path(__file__).resolve().parent.parent
CACHE = WS / "data" / "bitkub_cache"
BTC, THB = Asset("BTC", Asset.AssetType.CRYPTO), Asset("THB", Asset.AssetType.FOREX)
EVAL_START = pd.Timestamp("2019-01-01", tz="UTC")  # every SMA length has full look-back from here (Bitkub history starts 2018-05)
BUFFER = 0.995  # fraction of cash an order uses (the fill is at the open, the sizing at the last finished close)


# ------------------------------------------------------------------------------------------------ data
def load_bitkub(resolution="1D", symbol="BTC_THB"):
    history = BitkubHistory(cache_dir=CACHE)
    return history.get(symbol, resolution, int(datetime(2018, 5, 1).timestamp()), int(time.time()))


def load_yahoo():
    import yfinance as yf

    d = yf.download("BTC-USD", start="2017-01-01", progress=False, auto_adjust=True)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d.dropna(subset=["Close"]).rename(columns=str.lower)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index), tz="UTC")
    return d[["open", "high", "low", "close", "volume"]]


# ------------------------------------------------------------------------------------------------ independent pandas model
def pandas_trend(px, n, fee, lag=0, start=EVAL_START, end=None, fill=None, hold_only=False):
    """Decide on finished candles, trade at the open of the day's candle.

    Day t: mark = cash + units * close[t-1]; signal uses closes up to t-1-lag; a buy sizes on close[t-1] (what a strategy
    knows) and fills at the fill price (open[t] by default; ``fill`` can supply another per-day price, e.g. the 15:00 UTC
    hourly open). Returns (equity series indexed by day, trades list)."""
    days = px.index
    close, open_ = px["close"].to_numpy(), px["open"].to_numpy()
    fill_px = open_ if fill is None else fill.reindex(days).fillna(px["open"]).to_numpy()
    first = int(np.searchsorted(days, start))
    last = len(days) - 1 if end is None else int(np.searchsorted(days, end, side="right")) - 1
    cash, units, eq, trades, idx = 100_000.0, 0.0, [], [], []
    for t in range(max(first, n + 1 + lag), last + 1):
        eq.append(cash + units * close[t - 1])
        idx.append(days[t])
        ref = t - 1 - lag
        want_long = True if hold_only else close[ref] > close[ref - n + 1: ref + 1].mean()
        if want_long and units == 0 and cash > 0:
            qty = BUFFER * cash / (close[t - 1] * (1 + fee))
            cash -= qty * fill_px[t] * (1 + fee)
            units = qty
            trades.append((days[t], "buy", fill_px[t]))
        elif not want_long and units > 0:
            cash += units * fill_px[t] * (1 - fee)
            trades.append((days[t], "sell", fill_px[t]))
            units = 0.0
    return pd.Series(eq, index=pd.DatetimeIndex(idx)), trades


# ------------------------------------------------------------------------------------------------ metrics
def metrics(equity, capital=100_000):
    eq = equity.astype(float)
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1
    weekly = eq.resample("W-SUN").last().diff().dropna() / eq.iloc[0] * capital
    return {
        "total%": round(total * 100, 1), "CAGR%": round(((1 + total) ** (1 / years) - 1) * 100, 1) if years > 0 else np.nan,
        "MaxDD%": round(((eq / eq.cummax()) - 1).min() * 100, 1),
        "Sharpe": round(r.mean() / r.std() * np.sqrt(365), 2) if r.std() > 0 else np.nan,
        f"avg THB/wk on {capital // 1000}k": round(weekly.mean()), f"median THB/wk": round(weekly.median()),
        "wks>0%": round((weekly > 0).mean() * 100),
    }


# ------------------------------------------------------------------------------------------------ LumiBot engine strategy
class BtcTrend(Strategy):
    parameters = {"sma": 50, "lag": 0, "hold": False, "fee": 0.0025}

    def initialize(self):
        self.sleeptime = "1D"
        self.set_market("24/7")
        self.equity = []
        self.fills = []

    def on_trading_iteration(self):
        p = self.parameters
        n, lag = int(p["sma"]), int(p["lag"])
        bars = self.get_historical_prices(BTC, n + 1 + lag, "day", quote=THB)
        self.equity.append((self.get_datetime(), float(self.portfolio_value)))
        if bars is None or len(bars.df) < n + 1 + lag:
            return
        closes = bars.df["close"].to_numpy()
        ref = len(closes) - 1 - lag
        want_long = True if p["hold"] else closes[ref] > closes[ref - n + 1: ref + 1].mean()
        held = self.get_position(BTC)
        if want_long and held is None:
            qty = BUFFER * self.get_cash() / (float(closes[-1]) * (1 + float(p["fee"])))
            self.submit_order(self.create_order(BTC, qty, "buy", quote=THB))
        elif not want_long and held is not None:
            self.submit_order(self.create_order(BTC, held.quantity, "sell", quote=THB))

    def on_filled_order(self, position, order, price, quantity, multiplier):
        self.fills.append((self.get_datetime(), str(order.side), float(price), float(quantity)))


def engine_run(n, lag, hold, fee, start, end):
    fees = [TradingFee(percent_fee=fee)]
    _, strat = BtcTrend.run_backtest(
        BitkubBacktesting, backtesting_start=start, backtesting_end=end, cache_dir=CACHE, quote_asset=THB, benchmark_asset=None,
        budget=100_000, buy_trading_fees=fees, sell_trading_fees=fees, parameters={"sma": n, "lag": lag, "hold": hold, "fee": fee},
        analyze_backtest=False, show_plot=False, show_tearsheet=False, save_tearsheet=False, save_logfile=False,
        show_progress_bar=False, quiet_logs=True, show_indicators=False, save_stats_file=False,
    )
    eq = pd.Series({pd.Timestamp(d).tz_convert("UTC").normalize(): v for d, v in strat.equity})
    return eq, strat.fills


def main():
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    px = load_bitkub("1D")
    print(f"Bitkub BTC_THB daily candles: {len(px)} from {px.index[0].date()} to {px.index[-1].date()} (last FINISHED candle)")
    full_range = pd.date_range(px.index[0], px.index[-1], freq="D")
    print(f"  missing days inside the range: {len(full_range.difference(px.index))}  | zero/odd rows: "
          f"{int((px['volume'] <= 0).sum())} zero-volume candles")
    if len(full_range.difference(px.index)):
        print("  WARNING: gaps exist; a pandas/engine comparison needs a dense series. Gap dates:", list(full_range.difference(px.index))[:10])

    end = px.index[-1]
    fee = 0.0025
    results = []

    # ---- 1. parity: engine vs the independent pandas model, on the same real candles
    print("\n=== 1. PARITY: LumiBot engine (BitkubBacktesting) vs independent pandas model, real data, fee 0.25% ===")
    start_dt, end_dt = datetime(2019, 1, 1), (end + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=None)
    for label, n, lag, hold in (("buy&hold", 50, 0, True), ("SMA50 lag0", 50, 0, False), ("SMA100 lag0", 100, 0, False),
                                ("SMA50 lag1", 50, 1, False), ("SMA200 lag0", 200, 0, False)):
        t0 = time.time()
        eq_engine, fills = engine_run(n, lag, hold, fee, start_dt, end_dt)
        eq_model, trades = pandas_trend(px, n, fee, lag, start=EVAL_START, hold_only=hold)
        common = eq_engine.index.intersection(eq_model.index)
        a, b = eq_engine.loc[common], eq_model.loc[common]
        rel = ((a - b).abs() / b).max() * 100
        print(f"  {label:12s} days={len(common)} engine_final={a.iloc[-1]:>12,.0f} model_final={b.iloc[-1]:>12,.0f} "
              f"max rel diff={rel:.3f}%  trades engine/model={len(fills)}/{len(trades)}  ({time.time() - t0:.0f}s)")
        results.append({"section": "parity", "variant": label, "engine_final": a.iloc[-1], "model_final": b.iloc[-1], "max_rel_diff_%": rel,
                        "engine_trades": len(fills), "model_trades": len(trades)})

    # ---- 2. grid on the pandas model
    print("\n=== 2. GRID (pandas model, fee 0.25%, windows from 2019-01-01 unless stated) ===")
    windows = {"2019-now": "2019-01-01", "2021-06-now": "2021-06-01", "2023-now": "2023-01-01", "2019-2020": "2019-01-01|2020-12-31",
               "2021-2022": "2021-01-01|2022-12-31"}
    rows = []
    for wname, spec in windows.items():
        a, _, b = spec.partition("|")
        start, wend = pd.Timestamp(a, tz="UTC"), (pd.Timestamp(b, tz="UTC") if b else None)
        eq, _ = pandas_trend(px, 50, fee, 0, start=start, end=wend, hold_only=True)
        rows.append({"window": wname, "variant": "BUY&HOLD", **metrics(eq), "trades": 1})
        for n in (20, 30, 50, 100, 150, 200):
            for lag in (0, 1):
                eq, trades = pandas_trend(px, n, fee, lag, start=start, end=wend)
                rows.append({"window": wname, "variant": f"SMA{n} lag{lag}", **metrics(eq), "trades": len(trades)})
    grid = pd.DataFrame(rows)
    for wname in windows:
        print(f"\n--- {wname} ---")
        print(grid[grid.window == wname].drop(columns="window").to_string(index=False))
    results += [{"section": "grid", **r} for r in rows]

    # ---- fee sensitivity
    print("\n=== fee sensitivity: SMA50 and SMA100 (lag0), 2019-now ===")
    fr = []
    for f in (0.0010, 0.0025, 0.0050, 0.0075):
        for n in (50, 100):
            eq, trades = pandas_trend(px, n, f, 0)
            fr.append({"fee/side%": f * 100, "variant": f"SMA{n}", **metrics(eq), "trades": len(trades)})
    print(pd.DataFrame(fr).to_string(index=False))
    results += [{"section": "fees", **r} for r in fr]

    # ---- execution-time sensitivity: the live schedule trades ~15h after the daily candle closes (22:05 Bangkok = 15:05 UTC)
    print("\n=== execution timing: fill at the day's OPEN (00:00 UTC) vs the 15:00 UTC hourly open (the live 22:05 Bangkok schedule) ===")
    hourly = load_bitkub("60")
    h15 = hourly[hourly.index.hour == 15]
    fill15 = pd.Series(h15["open"].to_numpy(), index=h15.index.normalize())
    covered = fill15.reindex(px.index).notna().mean() * 100
    print(f"  hourly 15:00 UTC candle exists on {covered:.0f}% of days (others fall back to the day's open)")
    tr = []
    for n in (50, 100, 200):
        for label, fp in (("fill at 00:00 UTC open", None), ("fill at 15:00 UTC open", fill15)):
            eq, trades = pandas_trend(px, n, fee, 0, fill=fp)
            tr.append({"variant": f"SMA{n}", "execution": label, **metrics(eq), "trades": len(trades)})
    print(pd.DataFrame(tr).to_string(index=False))
    results += [{"section": "timing", **r} for r in tr]

    # ---- 3. context: the same rule on BTC-USD (Yahoo), same fee
    print("\n=== 3. CONTEXT: same rule on BTC-USD (Yahoo) vs BTC-THB (Bitkub), fee 0.25%, 2019-now ===")
    try:
        usd = load_yahoo()
        usd = usd.reindex(pd.date_range(usd.index[0], usd.index[-1], freq="D", tz="UTC")).ffill()  # Yahoo has missing days; context only
        cr = []
        for name, series in (("BTC-THB (Bitkub)", px), ("BTC-USD (Yahoo)", usd)):
            for n in (50, 100, 200):
                eq, trades = pandas_trend(series, n, fee, 0)
                cr.append({"market": name, "variant": f"SMA{n}", **metrics(eq), "trades": len(trades)})
            eq, _ = pandas_trend(series, 50, fee, 0, hold_only=True)
            cr.append({"market": name, "variant": "BUY&HOLD", **metrics(eq), "trades": 1})
        print(pd.DataFrame(cr).to_string(index=False))
        results += [{"section": "context", **r} for r in cr]
    except Exception as exc:  # context only; never block the main result
        print("  Yahoo comparison skipped:", type(exc).__name__, exc)

    pd.DataFrame(results).to_csv(WS / "research" / "bitkub_btc_trend_results.csv", index=False)
    print("\nwrote research/bitkub_btc_trend_results.csv")


if __name__ == "__main__":
    main()
