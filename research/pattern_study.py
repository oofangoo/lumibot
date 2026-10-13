"""E023 study: find the closest historical look-alikes of today's chart for BTC/THB (Bitkub) and SPY (Yahoo), read the scenarios off what followed, and test
whether pattern matching beats plain history. Writes logs/pattern_analogs.json (read by pattern_chart.py and pattern_page.py). See pattern_analogs.py for the method.
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\pattern_study.py
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pattern_analogs as pa  # noqa: E402

WS = Path(__file__).resolve().parent.parent
LOGS = WS / "logs"
N_SURR = 30
MARKETS = {
    "BTC": {"label": "Bitcoin (BTC/THB, Bitkub)", "horizon": 91, "unit": "days", "first_origin": "2020-06-01", "step": 7},
    "SPY": {"label": "S&P 500 ETF (SPY)", "horizon": 63, "unit": "trading days", "first_origin": "2003-01-01", "step": 10},
}


def load_btc():
    from bitkub_btc_trend import load_bitkub

    px = load_bitkub("1D")
    px.index = px.index.normalize()
    px = px[~px.index.duplicated(keep="last")]
    return px["close"].astype(float)


def load_spy():
    cache = WS / "data" / "spy_yahoo.csv"
    if cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600:
        s = pd.read_csv(cache, index_col=0, parse_dates=True)["close"]
        return s.astype(float)
    import yfinance as yf

    d = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    s = d["Close"].dropna().astype(float)
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
    s.rename("close").to_frame().to_csv(cache)
    return s


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def study(name, cfg, close, rng):
    H = cfg["horizon"]
    lc = np.log(close.to_numpy(float))
    n = len(lc)
    now = n - 1
    dates = close.index
    Z = pa.shape_matrix(lc)
    out = {"label": cfg["label"], "horizon": H, "unit": cfg["unit"], "first_date": str(dates[0].date()), "last_date": str(dates[-1].date()), "n_days": n,
           "last_close": float(close.iloc[-1]), "window": pa.W, "k": pa.K}

    # ---- today's look-alikes
    an = pa.find_analogs(Z, now, H)
    ends = [e for e, _ in an]
    paths = pa.fwd_paths(lc, ends, H)  # (K, H+1) from the analog's last day
    edges = pa.bucket_edges(lc, H, now)
    rets = paths[:, -1]
    buckets = [pa.bucket_of(r, edges) for r in rets]
    cur_win = (lc[now - pa.W + 1: now + 1] - lc[now - pa.W + 1: now + 1].mean()) / lc[now - pa.W + 1: now + 1].std()
    analogs = []
    for (e, d), p in zip(an, paths):
        w0 = e - pa.W + 1
        win = lc[w0: e + 1] - lc[e]  # the analog's window, rebased so its last day = 0
        analogs.append({"start": str(dates[w0].date()), "end": str(dates[e].date()), "distance": d, "fwd_return": float(np.exp(p[-1]) - 1),
                        "fwd_low": float(np.exp(p.min()) - 1), "fwd_high": float(np.exp(p.max()) - 1), "bucket": pa.BUCKETS[pa.bucket_of(p[-1], edges)],
                        "window_rebased": win.tolist(), "path": p.tolist(), "z": ((lc[w0: e + 1] - lc[w0: e + 1].mean()) / lc[w0: e + 1].std()).tolist()})
    base_ends = np.arange(pa.W - 1, now - H + 1)
    base_r = pa.fwd_returns(lc, base_ends, H)
    shares = []
    for b, nm in enumerate(pa.BUCKETS):
        idx = [i for i, bb in enumerate(buckets) if bb == b]
        lo, hi = wilson(len(idx), len(an))
        lo_edge = None if b == 0 else float(np.exp(edges[b - 1]) - 1)
        hi_edge = None if b == 4 else float(np.exp(edges[b]) - 1)
        shares.append({"name": nm, "n": len(idx), "share": len(idx) / len(an), "ci_lo": lo, "ci_hi": hi, "base": 0.2, "from": lo_edge, "to": hi_edge,
                       "median_path": np.median(paths[idx], axis=0).tolist() if idx else None})
    fan_q = [0.1, 0.25, 0.5, 0.75, 0.9]
    base_paths = pa.fwd_paths(lc, base_ends[:: max(1, len(base_ends) // 400)], H)
    out["today"] = {
        "analogs": analogs, "shares": shares, "edges": [float(np.exp(e) - 1) for e in edges], "current_z": cur_win.tolist(),
        "window_dates": [str(d.date()) for d in dates[now - pa.W + 1: now + 1]],
        "analog_fan": np.quantile(paths, fan_q, axis=0).T.tolist(), "base_fan": np.quantile(base_paths, fan_q, axis=0).T.tolist(),
        "analog_quantiles": {f"{int(q * 100):02d}": float(np.exp(np.quantile(rets, q)) - 1) for q in pa.QLEVELS},
        "base_quantiles": {f"{int(q * 100):02d}": float(np.exp(np.quantile(base_r, q)) - 1) for q in pa.QLEVELS},
        "analog_up_share": float((rets > 0).mean()), "base_up_share": float((base_r > 0).mean()), "mean_distance": float(np.mean([d for _, d in an])),
        "recent": {"dates": [str(d.date()) for d in dates[-pa.W:]], "close": close.iloc[-pa.W:].tolist()},
    }

    # ---- walk-forward skill vs plain history, real series
    first = int(np.searchsorted(dates, pd.Timestamp(cfg["first_origin"], tz=dates.tz)))
    origins = list(range(first, n - H, cfg["step"]))
    rows = pa.walk_forward(lc, H, origins, Z=Z)
    real = pa.wf_summary(rows, rng)
    real["origins"] = len(rows)
    real["by_year"] = {}
    for r in rows:
        y = str(dates[r["T"]].year)
        d = real["by_year"].setdefault(y, [0.0, 0.0, 0])
        d[0] += r["loss_an"]
        d[1] += r["loss_emp"]
        d[2] += 1
    real["by_year"] = {y: {"ratio": v[0] / v[1], "n": v[2]} for y, v in real["by_year"].items()}
    real["rows"] = [{"date": str(dates[r["T"]].date()), "pred_median": float(np.exp(r["med_an"]) - 1), "actual": float(np.exp(r["y"]) - 1)} for r in rows]
    print(f"[{name}] walk-forward origins {len(rows)}: analog loss {real['loss_an']:.4f} vs plain history {real['loss_emp']:.4f} (ratio {real['ratio']:.3f}, diff CI {real['diff_lo']:+.4f}..{real['diff_hi']:+.4f}); "
          f"80% band hit {real['in80_an'] * 100:.0f}% vs {real['in80_emp'] * 100:.0f}%; direction {real['dir_hit'] * 100:.0f}% (up {real['up_share'] * 100:.0f}%); rank corr {real['rank_corr']:+.3f}", flush=True)
    out["walk_forward"] = real

    # ---- robustness: other window lengths and K
    rob = {}
    for w, k in ((30, 10), (120, 10), (60, 20)):
        Zw = pa.shape_matrix(lc, w)
        o2 = [t for t in origins if t > w + 5]
        r2 = pa.walk_forward(lc, H, o2, k=k, w=w, Z=Zw)
        s2 = pa.wf_summary(r2, rng)
        rob[f"W{w}_K{k}"] = {"ratio": s2["ratio"], "rank_corr": s2["rank_corr"], "dir_hit": s2["dir_hit"], "n": s2["n"]}
        print(f"[{name}] robustness W={w} K={k}: loss ratio {s2['ratio']:.3f}, rank corr {s2['rank_corr']:+.3f}, direction {s2['dir_hit'] * 100:.0f}%", flush=True)
    out["robustness"] = rob

    # ---- the same procedure on made-up series with the same daily moves but no patterns
    null = {}
    for label, block in (("shuffled", 1), ("block10", 10)):
        ratios, corrs, dists, hits = [], [], [], []
        for s in range(N_SURR):
            sl = pa.surrogate(lc, rng, block)
            rs = pa.walk_forward(sl, H, origins, Z=pa.shape_matrix(sl))
            ss = pa.wf_summary(rs, rng, n_boot=50)
            ratios.append(ss["ratio"])
            corrs.append(ss["rank_corr"])
            dists.append(ss["mean_dist"])
            hits.append(ss["dir_hit"])
        ratios, corrs = np.array(ratios), np.array(corrs)
        null[label] = {"ratios": ratios.tolist(), "rank_corrs": corrs.tolist(), "mean_dist": float(np.mean(dists)), "dir_hit_mean": float(np.mean(hits)),
                       "p_ratio": float(np.mean(ratios <= real["ratio"])), "p_corr": float(np.mean(corrs >= real["rank_corr"]))}
        print(f"[{name}] null '{label}' ({N_SURR} made-up series): loss ratio mean {ratios.mean():.3f} (5-95%: {np.percentile(ratios, 5):.3f}..{np.percentile(ratios, 95):.3f}); "
              f"real {real['ratio']:.3f} -> share of made-up series doing as well or better {null[label]['p_ratio'] * 100:.0f}%; rank corr real {real['rank_corr']:+.3f} vs made-up mean {corrs.mean():+.3f}; "
              f"look-alike distance real {real['mean_dist']:.3f} vs made-up {np.mean(dists):.3f}", flush=True)
    out["null"] = null

    # ---- shape library
    lib = pa.shape_library(lc, H)
    seen = {}
    for c in lib["clusters"]:
        seen[c["name"]] = seen.get(c["name"], 0) + 1
        if seen[c["name"]] > 1:
            c["name"] = f"{c['name']} ({seen[c['name']]})"
    lib["split_date"] = str(dates[lib["split_index"]].date())
    out["library"] = lib
    print(f"[{name}] shape library (fit on data up to {lib['split_date']}); today's shape: {[c['name'] for c in lib['clusters'] if c['today']]}", flush=True)
    for c in lib["clusters"]:
        f, s = c["first"], c["second"]
        fm = f"{(np.exp(f['median']) - 1) * 100:+.0f}%" if f["median"] is not None else "-"
        sm = f"{(np.exp(s['median']) - 1) * 100:+.0f}%" if s["median"] is not None else "-"
        print(f"    {c['name']:<22} share {c['share'] * 100:3.0f}%  first half n={f['n']:4d} (~{f['episodes']:.0f} episodes) median {fm:>5} up {((f['up'] or 0) * 100):3.0f}% | second half n={s['n']:4d} (~{s['episodes']:.0f}) median {sm:>5} up {((s['up'] or 0) * 100):3.0f}%{'  <- today' if c['today'] else ''}")
    print(f"    base median first half {(np.exp(lib['base']['first']) - 1) * 100:+.0f}%, second half {(np.exp(lib['base']['second']) - 1) * 100:+.0f}%")

    print(f"\n[{name}] TODAY {out['last_date']} close {out['last_close']:,.2f}; look-alikes (window start..end, distance, what the next {H} {cfg['unit']} did):")
    for a in analogs:
        print(f"    {a['start']}..{a['end']}  dist {a['distance']:.3f}  {a['fwd_return'] * 100:+5.0f}% (low {a['fwd_low'] * 100:+.0f}%, high {a['fwd_high'] * 100:+.0f}%)  {a['bucket']}")
    print("    scenario shares: " + ", ".join(f"{s['name']} {s['share'] * 100:.0f}% ({s['n']}/{len(an)}; 95% range {s['ci_lo'] * 100:.0f}-{s['ci_hi'] * 100:.0f}%)" for s in out["today"]["shares"]))
    print("    bucket edges (return over the horizon): " + ", ".join(f"{e * 100:+.0f}%" for e in out["today"]["edges"]))
    print("    look-alike quantiles " + str({k: f"{v * 100:+.0f}%" for k, v in out["today"]["analog_quantiles"].items()}) + " | all-history " + str({k: f"{v * 100:+.0f}%" for k, v in out["today"]["base_quantiles"].items()}))
    return out


def main():
    os.makedirs(LOGS, exist_ok=True)
    rng = np.random.default_rng(23)
    which = sys.argv[1:] or list(MARKETS)
    result = {}
    for name in which:
        close = load_btc() if name == "BTC" else load_spy()
        print(f"\n===== {name}: {close.index[0].date()}..{close.index[-1].date()}, {len(close)} days =====", flush=True)
        result[name] = study(name, MARKETS[name], close, rng)
    path = LOGS / "pattern_analogs.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.update(result)
    path.write_text(json.dumps(old), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
