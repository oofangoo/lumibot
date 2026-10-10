"""E002: daily-bar scan of candidate assets with fixed (untuned) rules.

Rules are fixed up front so the 2024+ hold-out is a fair test:
  BH      buy and hold
  T200    long when close > SMA200 (signal at close t, earns return t+1), else cash
  T50     long when close > SMA50, else cash
Costs: 0.05% per side for stocks/ETFs, 0.25% per side for crypto.
Output: per asset/rule, in-sample (2018-2023) vs hold-out (2024-now): CAGR, maxDD, Sharpe,
mean weekly return, % positive weeks, and capital needed for $100/week at the mean weekly return.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ASSETS = ["BTC-USD", "SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "TQQQ"]
START, SPLIT = "2017-06-01", "2024-01-01"
COST = {"BTC-USD": 0.0025}


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.25 if r.index.dayofweek.nunique() == 7 else len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    per_year = 365.25 if r.index.dayofweek.nunique() == 7 else 252
    sharpe = r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else np.nan
    wk = (1 + r).resample("W-FRI").prod() - 1
    wk = wk[wk.index.isin(wk.index)]
    mw = wk.mean()
    return {
        "CAGR": cagr,
        "MaxDD": dd,
        "Sharpe": sharpe,
        "MeanWk": mw,
        "MedWk": wk.median(),
        "PosWks": (wk > 0).mean(),
        "WorstWk": wk.min(),
        "Cap$100/wk": (100 / mw) if mw > 0 else np.inf,
    }


def run(px: pd.Series, rule: str, cost: float) -> pd.Series:
    ret = px.pct_change()
    if rule == "BH":
        pos = pd.Series(1.0, index=px.index)
    else:
        n = 200 if rule == "T200" else 50
        pos = (px > px.rolling(n).mean()).astype(float)
    pos = pos.shift(1).fillna(0)  # trade next bar: no look-ahead
    turn = pos.diff().abs().fillna(0)
    return pos * ret - turn * cost


rows = []
for a in ASSETS:
    df = yf.download(a, start=START, auto_adjust=True, progress=False)
    if df.empty:
        print("NO DATA", a)
        continue
    px = df["Close"].squeeze().dropna()
    for rule in ("BH", "T200", "T50"):
        r = run(px, rule, COST.get(a, 0.0005))
        for name, seg in (("IS 17-23", r[: SPLIT]), ("HOLDOUT 24+", r[SPLIT:])):
            m = metrics(seg)
            if m:
                rows.append({"asset": a, "rule": rule, "period": name, **m})

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 500)
fmt = out.copy()
for c in ("CAGR", "MaxDD", "MeanWk", "MedWk", "PosWks", "WorstWk"):
    fmt[c] = (fmt[c] * 100).round(2)
fmt["Sharpe"] = fmt["Sharpe"].round(2)
fmt["Cap$100/wk"] = fmt["Cap$100/wk"].replace(np.inf, np.nan).round(0)
print(fmt.to_string(index=False))
out.to_csv("research/scan_daily_results.csv", index=False)
