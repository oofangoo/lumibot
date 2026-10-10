"""E005: combined portfolio of three independent trend sleeves on $50k, with/without broker-side resting stops.

Sleeves (initial weights): SPY>SMA200 40%, QQQ>SMA200 40%, BTC-USD>SMA50 20%. Otherwise the sleeve sits in cash
earning the T-bill rate (^IRX). Signal at close t; trade at close t+1 ... modelled as position(t+1) from signal(t),
earning close-to-close returns. Sleeves compound independently (no inter-sleeve rebalancing).

Stop model: after entry at price E (that day's close), a resting stop at E*(1-s). If a later bar's LOW <= stop, we exit at
min(stop, open) (gap-through fills at the open). After a stop-out the sleeve stays in cash until the trend signal turns
off and back on (a fresh entry). Daily bars only, so intraday path is approximated by low/open.
Costs: 0.05% per stock switch, 0.25% per BTC switch (each side).
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

START, CAPITAL = "2014-10-01", 50_000
SLEEVES = [("SPY", 200, 0.40, 0.0005), ("QQQ", 200, 0.40, 0.0005), ("BTC-USD", 50, 0.20, 0.0025)]
STOPS = [None, 0.08, 0.10, 0.15]


def load(sym):
    d = yf.download(sym, start=START, progress=False, auto_adjust=True)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d.dropna(subset=["Close"])


irx = yf.download("^IRX", start=START, progress=False, auto_adjust=False)["Close"].squeeze()


def sleeve_returns(d, n, cost, stop, cash_daily):
    px, lo, op = d["Close"].values, d["Low"].values, d["Open"].values
    sig = (d["Close"] > d["Close"].rolling(n).mean()).values
    rets = np.zeros(len(px))
    pos, entry, stopped = 0, np.nan, False
    for i in range(1, len(px)):
        r = 0.0
        if pos == 1:
            if stop is not None and lo[i] <= entry * (1 - stop):
                exit_px = min(entry * (1 - stop), op[i])
                r = exit_px / px[i - 1] - 1 - cost
                pos, stopped = 0, True
            else:
                r = px[i] / px[i - 1] - 1
        else:
            r = cash_daily[i]
        # Conservative timing: use YESTERDAY's close signal and trade at TODAY's close (one-day lag,
        # roughly what a bot running after the next open would get).
        s = sig[i - 1]
        if not s:
            stopped = False  # trend off: a later 'on' is a fresh entry
        want = 1 if (s and not stopped) else 0
        if want != pos:
            r -= cost
            pos = want
            if pos == 1:
                entry = px[i]
        rets[i] = r
    return pd.Series(rets, index=d.index)


data = {s: load(s) for s, *_ in SLEEVES}
idx = data["SPY"].index  # equity trading days; BTC reindexed to these
cash_daily = (irx / 100 / 252).reindex(idx).ffill().fillna(0).values

spy_px = data["SPY"]["Close"].reindex(idx)
spy_bh = spy_px.pct_change().fillna(0)


def metrics(r, label):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    wk = (1 + r).resample("W-FRI").prod() - 1
    return {
        "variant": label, "CAGR%": round(cagr * 100, 1), "MaxDD%": round(dd * 100, 1), "Sharpe": round(sharpe, 2),
        "$/wk avg": round(wk.mean() * CAPITAL), "$/wk med": round(wk.median() * CAPITAL),
        "wks>0 %": round((wk > 0).mean() * 100), "worst wk %": round(wk.min() * 100, 1),
        "final $": round(CAPITAL * eq.iloc[-1]),
    }


sleeve_ret = {}
for stop in STOPS:
    for sym, n, w, cost in SLEEVES:
        d = data[sym].reindex(idx).ffill()  # BTC trades weekends; reindexing to equity days keeps Friday->Monday gap intact
        sleeve_ret[(stop, sym)] = sleeve_returns(d, n, cost, stop, cash_daily)


def portfolio(stop, weights, rebalance_monthly):
    """Daily portfolio return. rebalance_monthly=False lets sleeves drift (the winner takes over)."""
    R = pd.DataFrame({s: sleeve_ret[(stop, s)] for s in weights})
    w0 = pd.Series(weights)
    if not rebalance_monthly:
        eq = (w0 * (1 + R).cumprod()).sum(axis=1)
        return eq.pct_change().fillna(0)
    out, cur = [], w0.copy()
    month = None
    for t, row in R.iterrows():
        if (t.year, t.month) != month:  # first trading day of a month: reset to target weights
            cur, month = w0.copy(), (t.year, t.month)
        port_r = float((cur * row).sum())
        cur = cur * (1 + row)
        cur = cur / cur.sum()
        out.append(port_r)
    return pd.Series(out, index=R.index)


CONFIGS = {
    "40/40/20 drift (as proposed)": ({"SPY": .4, "QQQ": .4, "BTC-USD": .2}, False),
    "40/40/20 monthly rebal": ({"SPY": .4, "QQQ": .4, "BTC-USD": .2}, True),
    "45/45/10 monthly rebal": ({"SPY": .45, "QQQ": .45, "BTC-USD": .10}, True),
    "50/50/0 monthly rebal (no BTC)": ({"SPY": .5, "QQQ": .5, "BTC-USD": 0.0}, True),
}
results = {}
for cname, (wts, reb) in CONFIGS.items():
    for stop in (None, 0.08):
        results[f"{cname} stop={'none' if stop is None else '8%'}"] = portfolio(stop, wts, reb)

PERIODS = {"FULL 2015-now": (None, None), "2015-19": (None, "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31"), "2023-now": ("2023-01-01", None)}
if __name__ == "__main__":
    for pname, (a, b) in PERIODS.items():
        rows = [metrics(spy_bh.loc[a:b], "SPY buy&hold")]
        for label, r in results.items():
            rows.append(metrics(r.loc[a:b], label))
        print(f"\n=== {pname} ===")
        print(pd.DataFrame(rows).to_string(index=False))
