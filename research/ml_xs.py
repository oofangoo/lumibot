"""E010: cross-sectional ML model that ranks ~110 large US stocks by expected 90-day (63 trading day) return.

Honest setup:
  * features use price history only (fundamentals from free sources are not point-in-time -> would leak the future)
  * features are cross-sectionally rank-normalised each month-end; target = forward 63-day return, ranked within month
  * WALK-FORWARD: for each test year Y (2012..2026) the models train only on months whose 63-day outcome was already
    known before Jan 1 of Y; they never see test data. Models: Ridge (linear) and a shallow, heavily regularised
    gradient-boosted tree. BASELINE to beat: plain 12-1 month momentum (the E009 rule).
  * metrics: monthly rank-IC, top-5 mean forward return vs the equal-weight universe and vs the momentum baseline
  * universe = today's big names -> survivorship-biased (flatters every method, including the baseline)
"""
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
T = ("AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B LLY JPM V UNH XOM MA COST HD PG JNJ ABBV WMT NFLX BAC CRM ORCL CVX KO MRK AMD PEP "
     "TMO ADBE LIN ACN MCD CSCO ABT WFC DHR TXN GE IBM QCOM CAT VZ AMGN NOW INTU ISRG PM DIS GS RTX SPGI T LOW UBER BKNG PFE NEE UNP AXP MS "
     "HON BLK C SYK TJX COP BA DE MMC PLD LMT CB MDT GILD ADP ADI SBUX VRTX MU AMAT PANW BX TMUS CI SO MO DUK INTC LRCX KLAC PYPL F GM MMM "
     "GD BMY USB CVS EMR FDX NKE SCHW CL MDLZ TGT BK DOW KHC COF AIG MET SPG").split()
px = yf.download(T + ["SPY"], start="2004-06-01", progress=False, auto_adjust=True)["Close"].dropna(axis=1, how="all")
spy = px.pop("SPY")
H = 63
ret = px.pct_change()

F = {
    "r1m": px / px.shift(21) - 1,
    "r3m": px / px.shift(63) - 1,
    "r6m": px / px.shift(126) - 1,
    "r12_1": px.shift(21) / px.shift(252) - 1,
    "vol3m": ret.rolling(63).std(),
    "vol12m": ret.rolling(252).std(),
    "from_high": px / px.rolling(252).max() - 1,
    "dist200": px / px.rolling(200).mean() - 1,
    "beta": ret.rolling(252).cov(spy.pct_change()).div(spy.pct_change().rolling(252).var(), axis=0),
}
fwd = px.shift(-H) / px - 1
me = px.groupby([px.index.year, px.index.month]).tail(1).index
me = me[me >= "2006-01-01"]

rows = []
for t in me:
    X = pd.DataFrame({k: v.loc[t] for k, v in F.items()}).dropna()
    if len(X) < 50:
        continue
    Xr = X.rank(pct=True) - 0.5
    y = fwd.loc[t].reindex(X.index)
    Xr["date"], Xr["fwd"] = t, y
    Xr["y_rank"] = y.rank(pct=True) - 0.5
    Xr["ticker"] = Xr.index
    rows.append(Xr)
P = pd.concat(rows, ignore_index=True)
feat = list(F)
print(f"panel: {len(P)} stock-months, {P['date'].nunique()} months, {P['ticker'].nunique()} tickers; features: {feat}")


def models():
    return {"ridge": Ridge(alpha=50.0),
            "gbm": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=200, l2_regularization=5.0, random_state=0)}


res = []
for Y in range(2012, 2027):
    cutoff = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=95)  # outcome window must have ended before the test year
    tr = P[(P["date"] <= cutoff) & P["y_rank"].notna()]
    te = P[(P["date"].dt.year == Y) & P["fwd"].notna()]
    if te.empty:
        continue
    ms = {n: m.fit(tr[feat], tr["y_rank"]) for n, m in models().items()}
    te = te.copy()
    for n, m in ms.items():
        te[n] = m.predict(te[feat])
    te["mom"] = te["r12_1"]
    for dt, g in te.groupby("date"):
        rec = {"date": dt, "universe": g["fwd"].mean()}
        for s in ("ridge", "gbm", "mom"):
            rec[f"ic_{s}"] = g[s].corr(g["fwd"], method="spearman")
            rec[f"top5_{s}"] = g.nlargest(5, s)["fwd"].mean()
            rec[f"q1_{s}"] = g.nlargest(max(1, len(g) // 5), s)["fwd"].mean()
        res.append(rec)
R = pd.DataFrame(res).set_index("date")
n = len(R)
print(f"\nwalk-forward out-of-sample: {n} monthly decisions, {R.index[0].date()} .. {R.index[-1].date()}")
tab = {}
for s in ("mom", "ridge", "gbm"):
    ic = R[f"ic_{s}"]
    tab[s] = {"mean IC": round(ic.mean(), 3), "IC t-stat*": round(ic.mean() / ic.std() * np.sqrt(n / 3), 2),
              "top-5 avg 90d %": round(R[f"top5_{s}"].mean() * 100, 1), "top-5 beats univ %": round((R[f"top5_{s}"] > R["universe"]).mean() * 100),
              "top-5 p10 %": round(R[f"top5_{s}"].quantile(.1) * 100, 1), "top-5 worst %": round(R[f"top5_{s}"].min() * 100, 1),
              "top-quintile avg %": round(R[f"q1_{s}"].mean() * 100, 1)}
tab["universe"] = {"top-5 avg 90d %": round(R["universe"].mean() * 100, 1)}
pd.set_option("display.width", 250)
print(pd.DataFrame(tab).T.to_string())
print("(*t-stat deflated for overlapping 90-day windows)")
for per, (a, b) in {"2012-2018": ("2012", "2018"), "2019-2026": ("2019", "2026")}.items():
    s = R.loc[a:b]
    print(f"  {per} (n={len(s)}): top-5 mom {s['top5_mom'].mean()*100:.1f}% | ridge {s['top5_ridge'].mean()*100:.1f}% | gbm {s['top5_gbm'].mean()*100:.1f}% | universe {s['universe'].mean()*100:.1f}%")
net = 0.002
print(f"\nafter a 0.2% round-trip cost per 90-day hold: mom {(R['top5_mom'].mean()-net)*100:.1f}%  ridge {(R['top5_ridge'].mean()-net)*100:.1f}%  gbm {(R['top5_gbm'].mean()-net)*100:.1f}%")

# --- final model on everything with a known outcome, score the latest month-end ---
full = P[P["y_rank"].notna()]
ridge = Ridge(alpha=50.0).fit(full[feat], full["y_rank"])
print("\nRidge coefficients (positive = higher rank today -> higher expected rank later):")
print(pd.Series(ridge.coef_, index=feat).round(3).sort_values().to_string())
gbm = models()["gbm"].fit(full[feat], full["y_rank"])
t = px.index[-1]
Xn = pd.DataFrame({k: v.loc[t] for k, v in F.items()}).dropna()
Xr = Xn.rank(pct=True) - 0.5
out = pd.DataFrame({"ridge": ridge.predict(Xr[feat]), "gbm": gbm.predict(Xr[feat]), "mom12_1%": (Xn["r12_1"] * 100).round(0),
                    "vol(ann)%": (Xn["vol3m"] * np.sqrt(252) * 100).round(0), "dist200%": (Xn["dist200"] * 100).round(0)}, index=Xn.index)
out["blend"] = out[["ridge", "gbm"]].rank(pct=True).mean(axis=1)
print(f"\nModel scores as of {t.date()} (top 10 by blend of ridge+gbm; higher = model expects relatively better 90d):")
print(out.sort_values("blend", ascending=False).head(10).round(3).to_string())
