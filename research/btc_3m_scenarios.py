"""E021: "if I buy BTC now, what are the likely scenarios over the next 3 months (91 days)?" on real Bitkub BTC/THB daily candles.

Part 1 WALK-FORWARD TEST: from 2020-06 every 7th day, each model (research/btc_forecast_models.py, fixed in advance) forecasts the distribution of the next
91-day return using ONLY data up to that day; we score the forecast against what really happened (pinball loss, interval coverage) and compare with the
naive baseline "the history so far" (EMP). Windows overlap, so ~25 independent quarters stand behind ~330 test points: block-bootstrap confidence
intervals are reported and the verdict follows them.
Part 2 TODAY: every model forecasts from the last finished candle: price quantiles, chance of ending down/up by X, chance of touching -15/-25/-35%
on the way (path models), 5 scenario bands. Written to logs/btc_3m_forecast.json for the LumiBot chart (research/btc_3m_chart.py).
Numbers are AFTER the 0.25% Bitkub fee on the buy and the sell. Not a forecast of what WILL happen: a calibrated description of how uncertain it is.

Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\btc_3m_scenarios.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btc_forecast_models import (ALL_MODELS, ML_MODELS, MIN_HISTORY, NET, PATH_MODELS, QLEVELS, H, QuantileML, feature_matrix, model_boot, model_emp,  # noqa: E402
                                 model_fhs, model_rw, path_stats, training_set)
from bitkub_btc_trend import load_bitkub  # noqa: E402

TEST_START = pd.Timestamp("2020-06-01", tz="UTC")
STEP = 7
BANDS = [("crash (worse than -30%)", -9, -0.30), ("bear (-30% to -10%)", -0.30, -0.10), ("sideways (-10% to +10%)", -0.10, 0.10),
         ("bull (+10% to +30%)", 0.10, 0.30), ("melt-up (better than +30%)", 0.30, 9)]


def run_models(closes, rng, ml, n_sims, features=None, t=None):
    """{model: (quantiles of the log return, paths or None)} using closes[:t+1]."""
    c = closes if t is None else closes[: t + 1]
    out = {"EMP": model_emp(c), "RW": model_rw(c, rng, n_sims), "BOOT": model_boot(c, rng, n_sims), "BOOT0": model_boot(c, rng, n_sims, demean=True),
           "FHS": model_fhs(c, rng, n_sims)}
    for name in ML_MODELS:
        out[name] = (ml[name].predict(features[len(c) - 1]), None)
    out["ENS"] = (np.mean([out[m][0] for m in ALL_MODELS], axis=0), None)
    return out


def pinball(q, y):
    """Mean pinball loss of the quantile vector q (levels QLEVELS) for the realized log return y."""
    d = y - q
    return float(np.mean(np.maximum(QLEVELS * d, (QLEVELS - 1) * d)))


def block_boot_ci(diff, block=13, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    m = len(diff)
    nb = -(-m // block)
    means = [diff[((rng.integers(0, m - block + 1, nb)[:, None]) + np.arange(block)).reshape(-1)[:m]].mean() for _ in range(n)]
    return np.percentile(means, [5, 95])


def walk_forward(closes, dates):
    lc = np.log(closes)
    feats = feature_matrix(closes)
    first = int(np.searchsorted(dates, TEST_START))
    ts = list(range(first, len(closes) - H, STEP))
    rows, ml, last_month = [], {}, None
    for t in ts:
        month = (dates[t].year, dates[t].month)
        if month != last_month:  # refit the machine-learning models monthly on outcomes already known on day t
            X, y = training_set(closes, feats, t)
            ml = {"QLIN": QuantileML("lin").fit(X, y), "QGBM": QuantileML("gbm").fit(X, y)}
            last_month = month
        res = run_models(closes, np.random.default_rng(t), ml, 2000, feats, t)
        y_real = lc[t + H] - lc[t]
        for name, (q, _) in res.items():
            rows.append({"t": t, "date": dates[t], "model": name, "loss": pinball(q, y_real), "y": y_real, "q": q})
    return pd.DataFrame(rows), len(ts)


def summarize_walk_forward(df, n_points):
    base = df[df.model == "EMP"].set_index("t")["loss"]
    out = []
    for name in ALL_MODELS + ("ENS",):
        d = df[df.model == name].set_index("t")
        q = np.stack(d["q"].values)
        y = d["y"].values
        cov = lambda lo, hi: float(((y >= q[:, lo]) & (y <= q[:, hi])).mean())  # noqa: E731
        diff = (d["loss"] - base.loc[d.index]).values  # negative = better than the baseline
        lo, hi = block_boot_ci(diff)
        halves = [float(np.mean(diff[d["date"].values < np.datetime64("2023-01-01")])), float(np.mean(diff[d["date"].values >= np.datetime64("2023-01-01")]))]
        out.append({"model": name, "mean loss": d["loss"].mean(), "vs EMP": d["loss"].mean() / base.loc[d.index].mean() - 1,
                    "90% CI of loss diff": f"[{lo:+.4f}, {hi:+.4f}]", "verdict": "better" if hi < 0 else ("worse" if lo > 0 else "no clear difference"),
                    "80% band hit (want 80)": cov(1, 5) * 100, "50% band hit (want 50)": cov(2, 4) * 100, "90% band hit (want 90)": cov(0, 6) * 100,
                    "pred median > 0 (%)": float((q[:, 3] > 0).mean()) * 100, "windows really up (%)": float((y > 0).mean()) * 100,
                    "direction right (%)": float(((q[:, 3] > 0) == (y > 0)).mean()) * 100, "rank corr (median vs real)": float(pd.Series(q[:, 3]).corr(pd.Series(y), method="spearman")), "loss diff 2020-22": halves[0], "loss diff 2023+": halves[1]})
    return pd.DataFrame(out)


def cdf_from_quantiles(q_net, x):
    """P(net return <= x) by interpolating the quantile vector; tails beyond 5%/95% are approximated at 2.5%/97.5%."""
    return float(np.interp(x, q_net, QLEVELS, left=0.025, right=0.975))


def today(closes, dates, fh):
    last = closes[-1]
    feats = feature_matrix(closes)
    X, y = training_set(closes, feats, len(closes) - 1)
    ml = {"QLIN": QuantileML("lin").fit(X, y), "QGBM": QuantileML("gbm").fit(X, y)}
    res = run_models(closes, np.random.default_rng(21), ml, 20000, feats)
    fc = {"date": str(dates[-1].date()), "last_close": float(last), "horizon_days": H, "models": {}}
    print(f"\n=== TODAY: last finished candle {dates[-1].date()}, BTC/THB {last:,.0f}; buy now, sell in {H} days, after 0.25% fee each way ===")
    rows = []
    for name, (q, paths) in res.items():
        net = np.exp(q) * NET - 1
        entry = {"net_quantiles": {f"{int(l * 100):02d}": float(v) for l, v in zip(QLEVELS, net)}}
        row = {"model": name, **{f"p{int(l * 100):02d}": f"{v * 100:+.0f}%" for l, v in zip(QLEVELS, net)}}
        row["10,000 THB -> median"] = f"{10000 * (1 + net[3]):,.0f}"
        row["P(end < 0)"] = f"{cdf_from_quantiles(net, 0) * 100:.0f}%"
        row["P(end < -20%)"] = f"{cdf_from_quantiles(net, -0.20) * 100:.0f}%"
        row["P(end > +20%)"] = f"{(1 - cdf_from_quantiles(net, 0.20)) * 100:.0f}%"
        entry["P_loss"] = cdf_from_quantiles(net, 0)
        entry["bands"] = {b[0]: cdf_from_quantiles(net, b[2]) - (cdf_from_quantiles(net, b[1]) if b[1] > -1 else 0.0) for b in BANDS}
        entry["bands"][BANDS[-1][0]] = 1 - cdf_from_quantiles(net, BANDS[-1][1])
        if paths is not None:
            ps = path_stats(paths)
            entry["path_stats"] = ps
            row.update({"touch -15%": f"{ps['touch_down_15'] * 100:.0f}%", "touch -25%": f"{ps['touch_down_25'] * 100:.0f}%", "touch -35%": f"{ps['touch_down_35'] * 100:.0f}%",
                        "touch +20%": f"{ps['touch_up_20'] * 100:.0f}%"})
            dipped = np.exp(np.minimum.accumulate(paths, axis=1)) - 1  # running low of each path, as a change from today
            entry["dip_curve"] = {str(int(d * 100)): [0.0] + [float((dipped[:, k] <= -d).mean()) for k in range(H)] for d in (0.15, 0.25, 0.35)}
            pct = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)
            entry["fan"] = {"days": list(range(0, H + 1)), "price": [[float(last)] * 5] + [[float(last * np.exp(v)) for v in pct[:, k]] for k in range(H)]}
        fc["models"][name] = entry
        rows.append(row)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nScenario shares by model (share of outcomes in each band, after fees):")
    print(pd.DataFrame({m: {b: f"{v * 100:.0f}%" for b, v in e["bands"].items()} for m, e in fc["models"].items()}).to_string())
    # price levels (THB) from the ensemble for the chart / reading
    q_ens = np.exp(res["ENS"][0]) * last
    fc["ens_price_quantiles"] = {f"{int(l * 100):02d}": float(v) for l, v in zip(QLEVELS, q_ens)}
    print("\nEnsemble price in 91 days (before fee), THB: " + "  ".join(f"p{int(l * 100)} {v:,.0f}" for l, v in zip(QLEVELS, q_ens)))
    return fc


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.4f}")
    px = load_bitkub("1D")
    closes, dates = px["close"].to_numpy(dtype=float), px.index
    print(f"BTC/THB daily candles {dates[0].date()}..{dates[-1].date()} ({len(closes)} days, {len(pd.date_range(dates[0], dates[-1], freq='D').difference(dates))} missing)")
    df, n_points = walk_forward(closes, dates)
    print(f"\n=== WALK-FORWARD: {n_points} forecast days (every {STEP}th day from {TEST_START.date()}, ~{n_points * STEP / H:.0f} independent quarters), scored on the real next-91-day return ===")
    tab = summarize_walk_forward(df, n_points)
    print(tab.to_string(index=False))
    fc = today(closes, dates, None)
    fc["walk_forward"] = json.loads(tab.to_json(orient="records"))
    Path("logs").mkdir(exist_ok=True)
    Path("logs/btc_3m_forecast.json").write_text(json.dumps(fc, indent=1), encoding="utf-8")
    print("\nwrote logs/btc_3m_forecast.json")


if __name__ == "__main__":
    main()
