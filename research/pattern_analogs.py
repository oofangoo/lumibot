"""E023 engine: "has the chart looked like this before, and what happened next?"  Plain numpy / scikit-learn (LumiBot has no pattern-matching tool; it supplies
the data, the backtest engine and the charts: see pattern_chart.py).

A pattern is the shape of the last W days of log price (z-scored, so only the shape matters, not the level or the size of the moves).
  * analogs      the K past windows whose shape is closest to today's (Euclidean distance), at least W days apart so each is a separate episode, and whose
                 following H days were already known at the time (no look-ahead).
  * scenarios    what happened over the next H days after each analog, grouped into five buckets defined by the asset's own history (fifths of every
                 H-day return ever seen), so the base rate of each bucket is 20% and any tilt is visible.
  * shape library k-means (k=6) on all window shapes, fitted on the first half of history only; the forward returns of every shape are then compared
                 between the first half (fit) and the second half (never seen).
  * honest test  walk-forward: at each past date use only earlier history to forecast the next H days, score against what happened, compare with the plain
                 history forecast, and compare with the same procedure run on made-up price series that have the same daily moves but no patterns.
All decisions (W=60, K=10, H=91 for BTC / 63 for SPY, k=6, quantile levels) are fixed before any result is looked at.
"""
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")

W = 60
K = 10
KMEANS_K = 6
QLEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
BUCKETS = ("Falls hard", "Weak", "Middling", "Strong", "Surges")


# ------------------------------------------------------------------------------------------------ shapes and analogs
def shape_matrix(lc, w=W):
    """Row i = z-scored log price of days i..i+w-1 (the window that ends at day i+w-1)."""
    win = sliding_window_view(np.asarray(lc, dtype=float), w)
    return (win - win.mean(1, keepdims=True)) / (win.std(1, keepdims=True) + 1e-12)


def find_analogs(Z, now_end, horizon, k=K, w=W, sep=None):
    """Windows ending at e <= now_end - horizon (outcome known), nearest to the window ending at now_end. Returns [(end_index, distance)], nearest first."""
    sep = sep or w
    cur = Z[now_end - w + 1]
    last_end = now_end - horizon
    if last_end < w - 1:
        return []
    cand = Z[: last_end - w + 2]  # window starts 0..last_end-w+1
    dist = np.sqrt(((cand - cur) ** 2).mean(1))
    order = np.argsort(dist)
    chosen = []
    for i in order:
        e = int(i) + w - 1
        if all(abs(e - c[0]) >= sep for c in chosen):
            chosen.append((e, float(dist[i])))
            if len(chosen) == k:
                break
    return chosen


def fwd_returns(lc, ends, horizon):
    lc = np.asarray(lc, dtype=float)
    ends = np.asarray(ends, dtype=int)
    return lc[ends + horizon] - lc[ends]


def fwd_paths(lc, ends, horizon):
    lc = np.asarray(lc, dtype=float)
    return np.array([lc[e: e + horizon + 1] - lc[e] for e in ends])  # shape (n, horizon+1), starts at 0


def bucket_edges(lc, horizon, upto_end, w=W):
    """Fifths of every H-day log return whose outcome was known by day upto_end."""
    ends = np.arange(w - 1, upto_end - horizon + 1)
    r = fwd_returns(lc, ends, horizon)
    return np.quantile(r, [0.2, 0.4, 0.6, 0.8])


def bucket_of(r, edges):
    return int(np.searchsorted(edges, r, side="right"))


# ------------------------------------------------------------------------------------------------ scoring
def pinball(qpred, y):
    d = y - qpred
    return float(np.mean(np.maximum(QLEVELS * d, (QLEVELS - 1) * d)))


def walk_forward(lc, horizon, origins, k=K, w=W, Z=None):
    """For each origin day T: forecast quantiles of the next-H log return from the analogs (history up to T only) and from plain history; score both."""
    Z = shape_matrix(lc, w) if Z is None else Z
    lc = np.asarray(lc, dtype=float)
    rows = []
    for T in origins:
        if T + horizon >= len(lc):
            continue
        an = find_analogs(Z, T, horizon, k, w)
        if len(an) < k:
            continue
        y = lc[T + horizon] - lc[T]
        ends_all = np.arange(w - 1, T - horizon + 1)
        emp = np.quantile(fwd_returns(lc, ends_all, horizon), QLEVELS)
        a_r = fwd_returns(lc, [e for e, _ in an], horizon)
        a_q = np.quantile(a_r, QLEVELS)
        rows.append({"T": int(T), "y": float(y), "loss_an": pinball(a_q, y), "loss_emp": pinball(emp, y), "med_an": float(np.median(a_r)), "med_emp": float(emp[3]),
                     "in80_an": bool(a_q[1] <= y <= a_q[5]), "in80_emp": bool(emp[1] <= y <= emp[5]), "dist": float(np.mean([d for _, d in an]))})
    return rows


def wf_summary(rows, rng, block=13, n_boot=2000):
    from scipy.stats import spearmanr

    la, le = np.array([r["loss_an"] for r in rows]), np.array([r["loss_emp"] for r in rows])
    y, med = np.array([r["y"] for r in rows]), np.array([r["med_an"] for r in rows])
    diff = la - le
    n = len(diff)
    nb = -(-n // block)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        boots.append(diff[idx].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rho = spearmanr(med, y)[0] if np.std(med) > 0 else 0.0
    return {"n": n, "loss_an": float(la.mean()), "loss_emp": float(le.mean()), "ratio": float(la.mean() / le.mean()), "diff": float(diff.mean()), "diff_lo": float(lo), "diff_hi": float(hi),
            "in80_an": float(np.mean([r["in80_an"] for r in rows])), "in80_emp": float(np.mean([r["in80_emp"] for r in rows])),
            "dir_hit": float(np.mean((med > 0) == (y > 0))), "up_share": float(np.mean(y > 0)), "rank_corr": float(rho),
            "mean_dist": float(np.mean([r["dist"] for r in rows]))}


def surrogate(lc, rng, block=1):
    """Same daily moves, no patterns: block-bootstrap of the real daily log returns (block=1: fully shuffled), rebuilt into a price path."""
    r = np.diff(np.asarray(lc, dtype=float))
    n = len(r)
    nb = -(-n // block)
    starts = rng.integers(0, n - block + 1, nb)
    rr = r[(starts[:, None] + np.arange(block)).ravel()[:n]]
    return np.concatenate([[lc[0]], lc[0] + np.cumsum(rr)])


# ------------------------------------------------------------------------------------------------ shape library
def name_shape(z):
    """Plain-language label from the average shape (z-scored log price over W days)."""
    z = np.asarray(z)
    n = len(z)
    slope = z[-1] - z[0]
    recent = z[-1] - z[-n // 4]  # last quarter of the window
    first = z[n // 2] - z[0]
    second = z[-1] - z[n // 2]
    if slope > 1.4:
        if recent < -0.15:
            return "Climb, then pullback"
        return "Climb that flattens out" if second < 0.35 * first else ("Accelerating climb" if second > 2 * first else "Steady climb")
    if slope < -1.4:
        if recent > 0.15:
            return "Slide, then bounce"
        return "Slide that flattens out" if second > 0.65 * first else ("Accelerating slide" if second < 2 * first else "Steady slide")
    if first < -0.8 and second > 0.8:
        return "V-shaped recovery"
    if first > 0.8 and second < -0.8:
        return "Peak and roll-over"
    if abs(slope) <= 1.4 and recent > 0.5:
        return "Late breakout"
    if abs(slope) <= 1.4 and recent < -0.5:
        return "Late slip"
    return "Sideways range"


def shape_library(lc, horizon, w=W, k=KMEANS_K, step=3):
    from sklearn.cluster import KMeans

    lc = np.asarray(lc, dtype=float)
    Z = shape_matrix(lc, w)
    n_end = len(lc) - 1
    ends = np.arange(w - 1, len(lc), step)
    split_end = (w - 1 + n_end) // 2  # first half = fit
    fit = ends[ends + horizon <= split_end]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z[fit - w + 1])
    lab = km.predict(Z[ends - w + 1])
    known = ends + horizon <= n_end
    r = np.full(len(ends), np.nan)
    r[known] = fwd_returns(lc, ends[known], horizon)
    first_half = ends + horizon <= split_end
    second_half = (ends > split_end) & known
    base = {"first": float(np.nanmedian(r[first_half])), "second": float(np.nanmedian(r[second_half]))}
    cur_lab = int(km.predict(Z[-1:])[0])
    out = []
    for c in range(k):
        m = lab == c
        cen = km.cluster_centers_[c]
        res = {"id": c, "name": name_shape(cen), "centroid": cen.tolist(), "today": c == cur_lab, "share": float(m.mean()),
               "band_lo": np.quantile(Z[ends[m] - w + 1], 0.1, axis=0).tolist(), "band_hi": np.quantile(Z[ends[m] - w + 1], 0.9, axis=0).tolist()}
        for key, mm in (("first", m & first_half), ("second", m & second_half)):
            rr = r[mm]
            res[key] = {"n": int(mm.sum()), "median": float(np.median(rr)) if len(rr) else None, "mean": float(rr.mean()) if len(rr) else None,
                        "up": float((rr > 0).mean()) if len(rr) else None, "episodes": float(mm.sum() * step / w)}
        out.append(res)
    out.sort(key=lambda d: d["centroid"][-1] - d["centroid"][0])
    return {"clusters": out, "base": base, "split_index": int(split_end), "today_cluster": cur_lab}
