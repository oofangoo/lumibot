"""E009: 'what to buy for ~90 days' sanity study. NOT a prediction: a momentum screen plus an honest look at how a
momentum pick's 90-day (63 trading day) outcomes have been distributed.

Universe: ~110 large US stocks (today's big names -> SURVIVORSHIP BIAS inflates the historical numbers; read them as
an upper bound). Score = 12-1 month momentum (return from t-252 to t-21 trading days), the classic cross-sectional factor.
Backtest: at each month-end 2006..2026 take the top 5, hold 63 trading days; compare with the equal-weight universe.
Then print today's screen with trend / volatility / drawdown context.
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
T = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B LLY JPM V UNH XOM MA COST HD PG JNJ ABBV WMT NFLX BAC CRM ORCL CVX KO MRK AMD PEP "
     "TMO ADBE LIN ACN MCD CSCO ABT WFC DHR TXN GE IBM QCOM CAT VZ AMGN NOW INTU ISRG PM DIS GS RTX SPGI T LOW UBER BKNG PFE NEE UNP AXP MS "
     "HON BLK C SYK TJX COP BA DE MMC PLD LMT CB MDT GILD ADP ADI SBUX VRTX MU AMAT PANW BX TMUS CI SO MO DUK INTC LRCX KLAC PYPL F GM MMM "
     "GD BMY USB CVS EMR FDX NKE SCHW CL MDLZ TGT BK DOW KHC COF AIG MET SPG").split()
px = yf.download(T, start="2004-06-01", progress=False, auto_adjust=True)["Close"].dropna(axis=1, how="all")
print(f"universe loaded: {px.shape[1]} tickers, {px.index[0].date()} .. {px.index[-1].date()}")

H, LB, SKIP, TOPN = 63, 252, 21, 5
mom = px.shift(SKIP) / px.shift(LB) - 1
fwd = px.shift(-H) / px - 1
month_ends = px.groupby([px.index.year, px.index.month]).tail(1).index
rows = []
for t in month_ends:
    if t < pd.Timestamp("2006-01-01") or pd.isna(fwd.loc[t]).all():
        continue
    m, f = mom.loc[t].dropna(), fwd.loc[t]
    m = m[f.reindex(m.index).notna()]
    if len(m) < 40:
        continue
    top = m.nlargest(TOPN).index
    rows.append({"date": t, "top5": f[top].mean(), "universe": f.dropna().mean(), "best_single": f[m.idxmax()]})
b = pd.DataFrame(rows).set_index("date")


def summ(s):
    return {"mean%": round(s.mean() * 100, 1), "median%": round(s.median() * 100, 1), "pos%": round((s > 0).mean() * 100),
            "p10%": round(s.quantile(.1) * 100, 1), "worst%": round(s.min() * 100, 1)}


print("\n90-day (63 trading-day) forward returns, monthly picks 2006-2026 (survivorship-biased upper bound):")
print(pd.DataFrame({"top-5 momentum": summ(b["top5"]), "#1 momentum stock only": summ(b["best_single"]), "equal-wt universe": summ(b["universe"])}).T.to_string())
print(f"top-5 beat the universe in {(b['top5'] > b['universe']).mean()*100:.0f}% of {len(b)} months")
for per, (x, y) in {"2006-2014": ("2006", "2014"), "2015-2026": ("2015", "2026")}.items():
    s = b.loc[x:y]
    print(f"  {per}: top-5 mean {s['top5'].mean()*100:.1f}% vs universe {s['universe'].mean()*100:.1f}%  (n={len(s)})")

# --- today's screen ---
last = px.index[-1]
cur = pd.DataFrame({
    "12-1m mom%": (mom.iloc[-1] * 100).round(0),
    "3m ret%": ((px.iloc[-1] / px.shift(63).iloc[-1] - 1) * 100).round(0),
    "vs 200d avg%": ((px.iloc[-1] / px.rolling(200).mean().iloc[-1] - 1) * 100).round(0),
    "ann vol%": (px.pct_change().tail(126).std() * np.sqrt(252) * 100).round(0),
    "1y maxDD%": px.tail(252).apply(lambda s: (s / s.cummax() - 1).min() * 100).round(0),
    "price": px.iloc[-1].round(2),
}).dropna()
print(f"\nToday's screen as of {last.date()} (top 12 by 12-1 momentum):")
print(cur.sort_values("12-1m mom%", ascending=False).head(12).to_string())
