"""E004: robustness of the 'above SMA-N -> risk asset else T-bills' rule.

Varies N (100/150/200/250) and the risk asset (QQQ since 1999, SPY since 1993), over the full history
and by sub-period (incl. dot-com crash, 2008, 2022). Cash leg earns the 13-week T-bill yield (^IRX).
Signal at close t, traded next bar. Cost 0.05% per switch.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

irx = yf.download("^IRX", start="1990-01-01", progress=False, auto_adjust=False)["Close"].squeeze()
cash_daily = (irx / 100 / 252).ffill()

PERIODS = {
    "FULL": (None, None),
    "2000-02 dotcom": ("2000-01-01", "2002-12-31"),
    "2007-09 GFC": ("2007-10-01", "2009-12-31"),
    "2010-19": ("2010-01-01", "2019-12-31"),
    "2020-22": ("2020-01-01", "2022-12-31"),
    "2023-now": ("2023-01-01", None),
}


def stats(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return eq.iloc[-1] ** (1 / yrs) - 1, (eq / eq.cummax() - 1).min()


rows = []
for sym, start in (("QQQ", "1999-03-10"), ("SPY", "1993-02-01")):
    px = yf.download(sym, start=start, progress=False, auto_adjust=True)["Close"].squeeze().dropna()
    ret = px.pct_change()
    cash = cash_daily.reindex(px.index).ffill().fillna(0)
    variants = {"BH": pd.Series(1.0, index=px.index)}
    for n in (100, 150, 200, 250):
        variants[f"SMA{n}"] = (px > px.rolling(n).mean()).astype(float)
    for name, sig in variants.items():
        pos = sig.shift(1).fillna(0)
        r = pos * ret + (1 - pos) * cash - pos.diff().abs().fillna(0) * 0.0005
        for pname, (a, b) in PERIODS.items():
            seg = r.loc[a:b] if a or b else r
            if len(seg) > 100:
                cagr, dd = stats(seg)
                rows.append({"asset": sym, "rule": name, "period": pname, "CAGR%": round(cagr * 100, 1), "MaxDD%": round(dd * 100, 1)})

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_rows", 500)
for sym in ("QQQ", "SPY"):
    for col in ("CAGR%", "MaxDD%"):
        t = out[out.asset == sym].pivot(index="rule", columns="period", values=col)[list(PERIODS)]
        print(f"\n{sym} - {col}")
        print(t.to_string())
out.to_csv("research/robustness_trend_results.csv", index=False)
