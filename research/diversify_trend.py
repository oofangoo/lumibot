"""E015: does adding uncorrelated trend sleeves (GLD gold, IEF 7-10y Treasuries, TLT 20y+ Treasuries, DBC commodities)
improve on the live SPY/QQQ(/BTC) trend portfolio? Rules fixed BEFORE running (no tuning): each sleeve = held while the last
completed close > SMA200, else T-bills (^IRX); 8% resting stop below entry; 0.05%/side cost; monthly rebalance to target weights;
1-day signal lag. Sleeve logic is copied from portfolio_backtest.py (E005) so numbers are comparable.

Window: 2007-01..now (DBC starts 2006-02; SMA200 warm-up). Covers 2008, 2020, and 2022 (stocks AND bonds fell together).
BTC (from 2014-10) is modelled on its real 7-day calendar with SMA50, fee 0.25%/side (as in E012/E013) and only enters
the portfolios evaluated on 2015-now.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

START, EVAL_START, CAPITAL = "2005-01-01", "2007-01-01", 50_000
STOCK_COST, BTC_COST = 0.0005, 0.0025


def load(sym, start=START):
    d = yf.download(sym, start=start, progress=False, auto_adjust=True)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d.dropna(subset=["Close"])


def sleeve_returns(d, n, cost, stop, cash_daily):
    """Identical logic to portfolio_backtest.sleeve_returns (E005)."""
    px, lo, op = d["Close"].values, d["Low"].values, d["Open"].values
    sig = (d["Close"] > d["Close"].rolling(n).mean()).values
    rets = np.zeros(len(px))
    pos, entry, stopped = 0, np.nan, False
    for i in range(1, len(px)):
        if pos == 1:
            if stop is not None and lo[i] <= entry * (1 - stop):
                r = min(entry * (1 - stop), op[i]) / px[i - 1] - 1 - cost
                pos, stopped = 0, True
            else:
                r = px[i] / px[i - 1] - 1
        else:
            r = cash_daily[i]
        s = sig[i - 1]
        if not s:
            stopped = False
        want = 1 if (s and not stopped) else 0
        if want != pos:
            r -= cost
            pos = want
            if pos == 1:
                entry = px[i]
        rets[i] = r
    return pd.Series(rets, index=d.index)


def metrics(r, label):
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    wk = (1 + r).resample("W-FRI").prod() - 1
    return {"variant": label, "CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "MaxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2), "$/wk avg": round(wk.mean() * CAPITAL), "$/wk med": round(wk.median() * CAPITAL),
            "wks>0 %": round((wk > 0).mean() * 100)}


def portfolio(R, weights):
    """Monthly rebalance to target weights."""
    w0 = pd.Series(weights)
    R = R[list(weights)]
    out, cur, month = [], w0.copy(), None
    for t, row in R.iterrows():
        if (t.year, t.month) != month:
            cur, month = w0.copy(), (t.year, t.month)
        out.append(float((cur * row).sum()))
        cur = cur * (1 + row)
        cur = cur / cur.sum()
    return pd.Series(out, index=R.index)


def build(sma=200, stop=0.08):
    spy = load("SPY")
    idx = spy.index
    irx = yf.download("^IRX", start=START, progress=False, auto_adjust=False)["Close"].squeeze()
    cash = (irx / 100 / 252).reindex(idx).ffill().fillna(0).values
    R = {}
    for sym in ("SPY", "QQQ", "GLD", "IEF", "TLT", "DBC"):
        d = load(sym).reindex(idx).ffill()
        R[sym] = sleeve_returns(d, sma, STOCK_COST, stop, cash)
    btc = load("BTC-USD", "2014-09-01")
    r7 = sleeve_returns(btc, 50, BTC_COST, 0.08, np.zeros(len(btc)))
    R["BTC"] = (1 + r7).cumprod().reindex(idx).ffill().pct_change().fillna(0)
    return pd.DataFrame(R), spy["Close"].reindex(idx).pct_change().fillna(0), idx


NO_BTC = {
    "live stocks 50/50 (SPY,QQQ)": {"SPY": .5, "QQQ": .5},
    "SPY/QQQ/GLD/IEF 25 each": {"SPY": .25, "QQQ": .25, "GLD": .25, "IEF": .25},
    "SPY/QQQ/GLD/TLT 25 each": {"SPY": .25, "QQQ": .25, "GLD": .25, "TLT": .25},
    "SPY/QQQ/GLD/IEF/DBC 20 each": {"SPY": .2, "QQQ": .2, "GLD": .2, "IEF": .2, "DBC": .2},
    "SPY/QQQ 30 + GLD/IEF 20": {"SPY": .3, "QQQ": .3, "GLD": .2, "IEF": .2},
}
WITH_BTC = {
    "LIVE 40/40/20 (SPY,QQQ,BTC)": {"SPY": .4, "QQQ": .4, "BTC": .2},
    "SPY/QQQ 25 + GLD/IEF 15 + BTC 20": {"SPY": .25, "QQQ": .25, "GLD": .15, "IEF": .15, "BTC": .2},
    "SPY/QQQ 20 + GLD/IEF 20 + BTC 20": {"SPY": .2, "QQQ": .2, "GLD": .2, "IEF": .2, "BTC": .2},
    "SPY/QQQ/GLD/IEF/DBC 16 + BTC 20": {"SPY": .16, "QQQ": .16, "GLD": .16, "IEF": .16, "DBC": .16, "BTC": .2},
}

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 500)
    out = []
    for sma in (200, 150, 250):  # SMA200 is the pre-registered rule; 150/250 only check it is not a knife-edge
        R, spy_bh, idx = build(sma=sma)
        if sma == 200:
            print("sleeve return correlations (daily, 2007+):")
            print(R.loc[EVAL_START:].corr().round(2).to_string())
        for pname, (a, b, grp) in {"2007-now": (EVAL_START, None, NO_BTC), "2007-12": (EVAL_START, "2012-12-31", NO_BTC), "2013-19": ("2013-01-01", "2019-12-31", NO_BTC),
                                   "2020-now": ("2020-01-01", None, NO_BTC), "2022 (stocks+bonds down)": ("2022-01-01", "2022-12-31", NO_BTC),
                                   "2015-now +BTC": ("2015-01-01", None, WITH_BTC), "2021-06-now +BTC": ("2021-06-01", None, WITH_BTC)}.items():
            rows = [metrics(spy_bh.loc[a:b], "SPY buy&hold")]
            for label, w in grp.items():
                rows.append(metrics(portfolio(R.loc[a:b], w), label))
            df = pd.DataFrame(rows)
            df.insert(0, "period", pname)
            df.insert(0, "SMA", sma)
            out.append(df)
            if sma == 200:
                print(f"\n=== SMA{sma} | {pname} ===")
                print(df.drop(columns=["period", "SMA"]).to_string(index=False))
    res = pd.concat(out)
    res.to_csv("research/diversify_trend_results.csv", index=False)
    print("\n=== robustness: Sharpe by SMA length, 2007-now (no BTC) and 2015-now (+BTC) ===")
    piv = res[res.period.isin(["2007-now", "2015-now +BTC"])].pivot_table(index=["period", "variant"], columns="SMA", values=["Sharpe", "MaxDD%", "CAGR%"])
    print(piv.to_string())
