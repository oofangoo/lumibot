"""E013: E005 portfolio recomputed with the corrected BTC sleeve (real 7-day calendar, SMA in calendar days, fee 0.25%/side).
Stock sleeves unchanged (SPY/QQQ > SMA200, 8% stop, T-bills when off). Monthly rebalance to target weights.
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
zero = np.zeros(len(d))


def btc_on_equity_days(n, stop):
    r7 = pb.sleeve_returns(d, n, 0.0025, stop, zero)
    eq = (1 + r7).cumprod()
    return eq.reindex(pb.idx).ffill().pct_change().fillna(0)  # weekend moves compound into Monday


PER = {"2015-now": (None, None), "2021-06-now": ("2021-06-01", None), "2023-now": ("2023-01-01", None)}
cfgs = {"SPY buy&hold": None, "50/50/0 (no BTC)": ({"SPY": .5, "QQQ": .5, "BTC-USD": 0.0}, None, None)}
for n in (50, 100, 150):
    for bstop in (None, 0.08):
        for w, tag in (({"SPY": .4, "QQQ": .4, "BTC-USD": .2}, "40/40/20"), ({"SPY": .45, "QQQ": .45, "BTC-USD": .10}, "45/45/10")):
            cfgs[f"{tag} BTC SMA{n} stop={'none' if bstop is None else '8%'}"] = (w, n, bstop)

results = {}
for label, c in cfgs.items():
    if c is None:
        results[label] = pb.spy_bh
        continue
    w, n, bstop = c
    if n is not None:
        pb.sleeve_ret[(0.08, "BTC-USD")] = btc_on_equity_days(n, bstop)
    results[label] = pb.portfolio(0.08, w, True)

pd.set_option("display.width", 250, "display.max_rows", 500)
for pname, (a, b) in PER.items():
    rows = [pb.metrics(r.loc[a:b], k) for k, r in results.items()]
    print(f"\n=== {pname} ===")
    print(pd.DataFrame(rows).drop(columns=["worst wk %", "final $"]).to_string(index=False))
