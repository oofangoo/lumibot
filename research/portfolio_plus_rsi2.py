"""E008: does a small RSI(2) mean-reversion sleeve improve the E005 portfolio (40/40/20 monthly rebal, 8% stop)?

RSI2 sleeve: SPY, buy when RSI(2)<10 and close>SMA200, sell when close>SMA5; conservative next-open execution,
1bp/side, cash earns the T-bill rate while flat. Blends: 100/0, 90/10, 80/20, 70/30 (E005 / RSI2), rebalanced monthly.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "research")
import portfolio_backtest as pb  # noqa: E402  (recomputes the E005 sleeves; prints nothing when imported)

d = pb.data["SPY"].reindex(pb.idx).ffill()
close, opn = d["Close"], d["Open"]
o, i = opn / close.shift(1) - 1, close / opn - 1


def rsi2(c):
    x = c.diff()
    up, dn = x.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(), (-x.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


r2, sma200, sma5 = rsi2(close), close.rolling(200).mean(), close.rolling(5).mean()
h, state = [], 0
for k in range(len(close)):
    if state == 0 and r2.iloc[k] < 10 and close.iloc[k] > sma200.iloc[k]:
        state = 1
    elif state == 1 and close.iloc[k] > sma5.iloc[k]:
        state = 0
    h.append(state)
h = pd.Series(h, index=close.index)
chg = h.diff().abs().fillna(0)
cash = pd.Series(pb.cash_daily, index=pb.idx)
# conservative: signal at close t -> buy/sell at open t+1. Overnight held when flagged at the prior close.
rsi_ret = (h.shift(2).fillna(0) * o + h.shift(1).fillna(0) * i + (1 - h.shift(1).fillna(0)) * cash - chg.shift(1).fillna(0) * 1e-4).fillna(0)

A = pb.portfolio(0.08, {"SPY": .4, "QQQ": .4, "BTC-USD": .2}, True)
print("daily-return correlation  E005 vs RSI2 sleeve: %.2f" % A.corr(rsi_ret))
print("RSI2 sleeve alone (incl. T-bill on idle cash): CAGR %.1f%%" % (((1 + rsi_ret).prod() ** (252 / len(rsi_ret)) - 1) * 100))


def blend(w_rsi):
    out, cur, month = [], np.array([1 - w_rsi, w_rsi]), None
    base = np.array([1 - w_rsi, w_rsi])
    for t in A.index:
        if (t.year, t.month) != month:
            cur, month = base.copy(), (t.year, t.month)
        row = np.array([A[t], rsi_ret[t]])
        out.append(float((cur * row).sum()))
        cur = cur * (1 + row)
        cur = cur / cur.sum()
    return pd.Series(out, index=A.index)


results = {f"E005 {int((1-w)*100)} / RSI2 {int(w*100)}": blend(w) for w in (0.0, 0.1, 0.2, 0.3)}
for pname, (a, b) in pb.PERIODS.items():
    rows = [pb.metrics(r.loc[a:b], label) for label, r in results.items()]
    print(f"\n=== {pname} ===")
    print(pd.DataFrame(rows).to_string(index=False))
