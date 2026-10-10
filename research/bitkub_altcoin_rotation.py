"""E017b: can rotating into Bitkub altcoins get anywhere near 5-10% per week? Real Bitkub THB-market candles, rules fixed in advance.

Universe: every THB market that is tradable through the API (source == "exchange"), active OR stopped, with its real daily candles
(point-in-time: a coin only exists from its first candle). SURVIVORSHIP CAVEAT: coins that were fully removed from the exchange are not
in the symbol list, so their history is missing; this flatters every altcoin result.

Weekly cycle (Monday 00:00 UTC): decide from finished candles up to Sunday, trade at Monday's open, mark at the next Monday's open
(or the week's last candle if the next open does not exist; a coin with no candle at all that week is a total loss).
Costs: Bitkub fee 0.25%/side + assumed slippage on top (default 0.30%/side for altcoins), charged on traded weight only.

Variants (pre-specified, NOT tuned; all are reported): BTC buy&hold, BTC trend SMA50, equal-weight liquid coins, momentum top-5 (28d),
momentum top-5 with a BTC>SMA50 regime filter, top-1 weekly chase (7d), and a reversal control (bottom-5 by 7d).
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\bitkub_altcoin_rotation.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import CACHE  # noqa: E402
from lumibot.tools.bitkub_helpers import BitkubClient  # noqa: E402
from lumibot.tools.bitkub_history import BitkubHistory  # noqa: E402

FEE = 0.0025
MIN_LIQUIDITY_THB = 1_000_000  # trailing-30-day median daily traded value; a 10k THB order must be a small part of a day's volume
START = pd.Timestamp("2019-01-07", tz="UTC")  # first Monday of the evaluation window
CAPITAL = 10_000


def load_panel():
    """Daily open/close/volume panels for every tradable THB market (cached; first run downloads ~300 markets)."""
    client = BitkubClient()
    markets = [r["symbol"] for r in client.get_symbols() if r["quote_asset"] == "THB" and r["source"] == "exchange"]
    history = BitkubHistory(client=client, cache_dir=CACHE)
    now = int(time.time())
    opens, closes, vols, failed = {}, {}, {}, []
    t0 = time.time()
    for i, sym in enumerate(markets, 1):
        try:
            f = history.get(sym, "1D", int(pd.Timestamp("2018-05-01").timestamp()), now)
        except Exception as exc:
            failed.append((sym, type(exc).__name__))
            continue
        if f.empty:
            continue
        f.index = f.index.normalize()
        opens[sym], closes[sym], vols[sym] = f["open"], f["close"], f["volume"]
        if i % 50 == 0:
            print(f"  downloaded/cached {i}/{len(markets)} markets ({time.time() - t0:.0f}s)")
    print(f"  markets with candles: {len(closes)} of {len(markets)}; failed: {len(failed)} {failed[:3]}")
    return pd.DataFrame(opens).sort_index(), pd.DataFrame(closes).sort_index(), pd.DataFrame(vols).sort_index()


def run_strategy(name, open_, close, elig, pick, regime=None, slip=0.003, mondays=None):
    """pick(date, eligible) -> list of symbols to hold this week (equal weight). Returns the weekly net-return series."""
    rets, drift, held_counts, wipeouts = [], {}, [], 0
    for k, monday in enumerate(mondays[:-1]):
        nxt = mondays[k + 1]
        sunday = monday - pd.Timedelta(days=1)
        if sunday not in close.index or monday not in open_.index:
            rets.append((monday, 0.0))
            continue
        eligible = elig.get(monday, [])
        if (regime is not None and not regime.at[sunday]) or not eligible:
            target = {}  # regime says stay out (or nothing is tradable): hold cash
        else:
            chosen = pick(sunday, eligible)
            target = {s: 1.0 / len(chosen) for s in chosen}
        # turnover against last week's drifted weights; every traded unit of weight pays fee + slippage
        traded = sum(abs(target.get(s, 0.0) - drift.get(s, 0.0)) for s in set(target) | set(drift))
        cost = traded * (FEE + (slip if name != "BTC" else 0.0))
        gross, new_drift = 0.0, {}
        for s, w in target.items():
            week = close.loc[monday:nxt - pd.Timedelta(days=1), s].dropna()
            exit_px = open_.at[nxt, s] if nxt in open_.index and np.isfinite(open_.at[nxt, s]) else (week.iloc[-1] if len(week) else np.nan)
            r = (exit_px / open_.at[monday, s] - 1) if np.isfinite(exit_px) else -1.0  # no candle at all => assume the position was lost
            wipeouts += int(not np.isfinite(exit_px))
            gross += w * r
            new_drift[s] = w * (1 + r)
        total = 1.0 - sum(target.values()) + sum(new_drift.values())
        net = (gross - cost)
        rets.append((monday, net))
        drift = {s: v / total for s, v in new_drift.items()} if total > 0 else {}
        held_counts.append(len(target))
    out = pd.Series(dict(rets))
    out.attrs["wipeouts"], out.attrs["avg_names"] = wipeouts, float(np.mean(held_counts)) if held_counts else 0.0
    return out


def summarize(name, wr, since=None):
    if since is not None:
        wr = wr[wr.index >= since]
    eq = CAPITAL * (1 + wr).cumprod()
    years = len(wr) / 52.0
    roll26 = (eq / eq.shift(26)).dropna() - 1
    return {
        "variant": name, "weeks": len(wr), "mean/wk%": round(wr.mean() * 100, 2), "median/wk%": round(wr.median() * 100, 2),
        "wks>=5%": f"{(wr >= 0.05).mean() * 100:.0f}%", "wks>=10%": f"{(wr >= 0.10).mean() * 100:.0f}%", "wks<=-10%": f"{(wr <= -0.10).mean() * 100:.0f}%",
        "worst wk%": round(wr.min() * 100, 0), "CAGR%": round(((eq.iloc[-1] / CAPITAL) ** (1 / years) - 1) * 100) if years > 0 and eq.iloc[-1] > 0 else -100,
        "MaxDD%": round(((eq / eq.cummax()) - 1).min() * 100), "10k THB ends at": f"{eq.iloc[-1]:,.0f}",
        "26wk windows below start": f"{(roll26 < 0).mean() * 100:.0f}%", "avg names": round(wr.attrs.get("avg_names", np.nan), 1) if hasattr(wr, "attrs") else "",
    }


def main():
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("loading THB markets ...")
    open_, close, vol = load_panel()
    value = (vol * close)
    value30 = value.rolling(30, min_periods=20).median()
    ret28, ret7 = close / close.shift(28) - 1, close / close.shift(7) - 1
    btc = close["BTC_THB"]
    regime = (btc > btc.rolling(50).mean())  # known at the Sunday close
    mondays = [d for d in pd.date_range(START, close.index[-1], freq="W-MON")]
    print(f"panel: {close.shape[1]} coins, {close.index[0].date()}..{close.index[-1].date()}, {len(mondays)} Mondays from {START.date()}")
    elig = {}
    for m in mondays:
        sunday = m - pd.Timedelta(days=1)
        if m in open_.index and sunday in close.index:
            ok = (value30.loc[sunday] >= MIN_LIQUIDITY_THB) & np.isfinite(open_.loc[m]) & np.isfinite(close.loc[sunday])
            elig[m] = list(ok.index[ok.to_numpy()])
    eligible_counts = pd.Series({m: len(v) for m, v in elig.items()})
    print(f"liquid coins available per week: median {int(eligible_counts.median())}, min {int(eligible_counts.min())}, max {int(eligible_counts.max())}")

    def top(score, n):
        return lambda d, el: sorted((s for s in el if np.isfinite(score.at[d, s])), key=lambda s: -score.at[d, s])[:n] or ["BTC_THB"]

    def bottom(score, n):
        return lambda d, el: sorted((s for s in el if np.isfinite(score.at[d, s])), key=lambda s: score.at[d, s])[:n] or ["BTC_THB"]

    results = {}
    for slip in (0.0, 0.003, 0.01):
        specs = {
            "BTC buy&hold": run_strategy("BTC", open_, close, elig, lambda d, el: ["BTC_THB"], slip=slip, mondays=mondays),
            "BTC trend SMA50": run_strategy("BTC", open_, close, elig, lambda d, el: ["BTC_THB"], regime=regime, slip=slip, mondays=mondays),
            "equal-weight liquid coins": run_strategy("EW", open_, close, elig, lambda d, el: el or ["BTC_THB"], slip=slip, mondays=mondays),
            "momentum top-5 (28d)": run_strategy("MOM5", open_, close, elig, top(ret28, 5), slip=slip, mondays=mondays),
            "momentum top-5 + BTC>SMA50": run_strategy("MOM5R", open_, close, elig, top(ret28, 5), regime=regime, slip=slip, mondays=mondays),
            "chase top-1 (7d)": run_strategy("CH1", open_, close, elig, top(ret7, 1), slip=slip, mondays=mondays),
            "chase top-1 (7d) + BTC>SMA50": run_strategy("CH1R", open_, close, elig, top(ret7, 1), regime=regime, slip=slip, mondays=mondays),
            "reversal bottom-5 (7d) [control]": run_strategy("REV5", open_, close, elig, bottom(ret7, 5), slip=slip, mondays=mondays),
        }
        results[slip] = specs
    for slip, specs in results.items():
        print(f"\n=== 2019-now, fee 0.25% + altcoin slippage {slip * 100:g}%/side ===")
        print(pd.DataFrame([summarize(n, w) for n, w in specs.items()]).to_string(index=False))
    split = pd.Timestamp("2023-01-02", tz="UTC")
    for label, since, until in (("IN-SAMPLE-ish 2019-2022", None, split), ("OUT-OF-PERIOD 2023-now", split, None)):
        print(f"\n=== {label}, slippage 0.3%/side ===")
        rows = []
        for n, w in results[0.003].items():
            sub = w[(w.index >= (since or w.index[0])) & (w.index < (until or w.index[-1] + pd.Timedelta(days=1)))]
            sub.attrs.update(w.attrs)
            rows.append(summarize(n, sub))
        print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame({n: w for n, w in results[0.003].items()}).to_csv(Path(__file__).resolve().parent / "bitkub_altcoin_rotation_weekly.csv")
    print("\nwrote research/bitkub_altcoin_rotation_weekly.csv (weekly net returns, slippage 0.3%)")


if __name__ == "__main__":
    main()
