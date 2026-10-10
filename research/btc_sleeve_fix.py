"""E012: correct the BTC sleeve model. E005 simulated BTC on an equity-trading-day calendar (weekends dropped), so 'SMA50' was ~70
calendar days and weekend lows never triggered stops. Here BTC is simulated on its real 7-day calendar, cost 0.25%/side, cash earns 0.
Validation target: Lumibot's own backtest of the same rule on 2021-06-01..2026-09-30 (parity_check.py):
    SMA50, no stop  -> +176% total, max DD -59.6%      SMA50, 8% stop -> +106%, DD -66.9%
Then: how do other lengths / stop sizes / fee levels look, and what does the 40/40/20 portfolio become?
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "research")
import portfolio_backtest as pb  # noqa: E402

d = yf.download("BTC-USD", start="2014-10-01", progress=False, auto_adjust=True)
d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
d = d.dropna(subset=["Close"])
zero_cash = np.zeros(len(d))


def sleeve(n, stop, fee):
    return pb.sleeve_returns(d, n, fee, stop, zero_cash)


def trades(n, stop, fee, a, b):
    r = sleeve(n, stop, fee).loc[a:b]
    return r


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 365.25
    flips = (r != 0).sum()
    return {"total%": round((eq.iloc[-1] - 1) * 100), "CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "DD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(365), 2)}


print("=== validation vs Lumibot (2021-06-01..2026-09-30, fee 0.25%/side) ===")
for stop in (None, 0.08):
    print("pandas 7-day SMA50 stop=%s:" % stop, stats(sleeve(50, stop, 0.0025).loc["2021-06-01":"2026-09-30"]), " | Lumibot:", "+176% / DD -59.6%" if stop is None else "+106% / DD -66.9%")

rows = []
for per, (a, b) in {"2015-now": ("2015-01-01", None), "2021-06-now": ("2021-06-01", None), "2022-now (post-peak)": ("2022-01-01", None)}.items():
    for n in (50, 100, 150, 200):
        for stop in (None, 0.08, 0.15):
            for fee in (0.0010, 0.0025):
                s = stats(sleeve(n, stop, fee).loc[a:b])
                rows.append({"period": per, "SMA": n, "stop": stop or "none", "fee/side": fee, **s})
df = pd.DataFrame(rows)
bh = (1 + d["Close"].pct_change().dropna())
pd.set_option("display.width", 250, "display.max_rows", 500)
for per in df.period.unique():
    print(f"\n=== {per} (BTC buy&hold total {round((bh.loc['2015-01-01' if per=='2015-now' else ('2021-06-01' if per.startswith('2021') else '2022-01-01'):].prod()-1)*100)}%) ===")
    print(df[df.period == per].drop(columns="period").to_string(index=False))
