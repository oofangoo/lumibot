"""E006: faster-than-trend ideas testable on daily OHLC (SPY since 1993, QQQ since 1999), cost-sensitivity included.

  INTRADAY   buy open, sell close every day                      (baseline: is there anything in the trading day?)
  OVERNIGHT  buy close, sell next open every day                 (the 'overnight drift' effect)
  RSI2       Connors-style mean reversion: buy when RSI(2)<10 and close>SMA200, sell when close>SMA5
             two execution models: MOC (trade at the signal close; optimistic) and NEXTOPEN (conservative)
Costs are per side as a fraction of price: 0, 0.5bp, 1bp, 2bp. Fixed rules, nothing tuned.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
CAP = 50_000


def load(sym, start):
    d = yf.download(sym, start=start, progress=False, auto_adjust=True)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d.dropna(subset=["Close", "Open"])


def rsi2(close):
    d = close.diff()
    up, dn = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi2_hold(d):
    c = d["Close"]
    r, sma200, sma5 = rsi2(c), c.rolling(200).mean(), c.rolling(5).mean()
    hold, h = [], 0
    for i in range(len(c)):
        if h == 0 and r.iloc[i] < 10 and c.iloc[i] > sma200.iloc[i]:
            h = 1
        elif h == 1 and c.iloc[i] > sma5.iloc[i]:
            h = 0
        hold.append(h)
    return pd.Series(hold, index=c.index)


def stats(r, label, trades_per_year):
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    wk = (1 + r).resample("W-FRI").prod() - 1
    return {"rule": label, "CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), "MaxDD%": round((eq / eq.cummax() - 1).min() * 100, 1),
            "Sharpe": round(r.mean() / r.std() * np.sqrt(252), 2) if r.std() > 0 else np.nan,
            "trades/yr": round(trades_per_year), "$/wk avg": round(wk.mean() * CAP), "$/wk med": round(wk.median() * CAP),
            "wks>0%": round((wk > 0).mean() * 100)}


def run(sym, start):
    d = load(sym, start)
    o = d["Open"] / d["Close"].shift(1) - 1   # overnight (prev close -> open)
    i = d["Close"] / d["Open"] - 1            # intraday  (open -> close)
    rows = []
    for bp in (0, 0.5, 1, 2):
        c = bp / 1e4
        rows.append((f"INTRADAY  cost {bp}bp/side", i - 2 * c, 252))
        rows.append((f"OVERNIGHT cost {bp}bp/side", o - 2 * c, 252))
    h = rsi2_hold(d)
    chg = h.diff().abs().fillna(0)
    for bp in (1, 2):
        c = bp / 1e4
        moc = h.shift(1).fillna(0) * ((1 + o) * (1 + i) - 1) - chg.shift(1).fillna(0) * c   # signal at close t, own the close(t)->close(t+1) move
        nxt = h.shift(2).fillna(0) * o + h.shift(1).fillna(0) * i - chg.shift(1).fillna(0) * c  # conservative: buy next open
        n_tr = (chg.sum() / 2) / (len(d) / 252)
        rows.append((f"RSI2 MOC      cost {bp}bp/side", moc, n_tr * 2))
        rows.append((f"RSI2 NEXTOPEN cost {bp}bp/side", nxt, n_tr * 2))
    periods = {"FULL": (None, None), "to 2009": (None, "2009-12-31"), "2010-19": ("2010-01-01", "2019-12-31"), "2020-now": ("2020-01-01", None)}
    out = []
    for name, (a, b) in periods.items():
        for label, r, tpy in rows:
            seg = r.loc[a:b] if (a or b) else r
            if len(seg) > 200:
                out.append({"asset": sym, "period": name, **stats(seg, label, tpy)})
    in_mkt = h.mean()
    print(f"\n##### {sym}  (RSI2 in market {in_mkt*100:.0f}% of days; buy&hold CAGR for reference below)")
    bh = d["Close"].pct_change()
    print("buy&hold FULL:", stats(bh, "BH", 0))
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    for name in periods:
        print(f"\n--- {sym} {name} ---")
        print(df[df.period == name].drop(columns=["asset", "period"]).to_string(index=False))
    return df


all_ = pd.concat([run("SPY", "1993-02-01"), run("QQQ", "1999-03-10")])
all_.to_csv("research/scan_fast_daily_results.csv", index=False)
