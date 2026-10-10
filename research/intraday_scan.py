"""E007: three intraday ideas on 5-minute SPY/QQQ bars (SIP, split-adjusted), 2016-01 .. 2026-10.

Rules are FIXED before looking at results (no tuning). Train/read period 2016-2021, out-of-sample TEST 2022-now.
Bars index i: 9:30+5*i, i=0..77. Signals use COMPLETED bars; trades fill at the NEXT bar's open (conservative).
Flat every night, 1x notional, one trade per day max. Cost per side 1bp / 2bp / 4bp (slippage + spread; commissions are 0).

  ORB    first 30 min = bars 0-5 define a range. First later bar CLOSING outside it triggers entry in that direction
         (long above / short below), stop at the opposite side of the range, exit at the 15:55 bar close.
  MOM    sign of (10:00 price / prior close - 1) predicts the final half hour (Gao et al.): enter bar 66 (15:30), exit close.
  VWAP   when a bar closes more than K below the running VWAP (bars>=12, i.e. after 10:30) buy next open (short mirror above);
         exit when a bar closes back at/through VWAP (next open) or at the close. K = 0.25% and 0.50% (both reported).
Variants: LONG-ONLY and LONG-SHORT.
"""
import numpy as np
import pandas as pd

ROOT = "data/"
CAP = 50_000
COSTS_BP = (1, 2, 4)


def days(sym):
    df = pd.read_parquet(f"{ROOT}{sym}_5min_sip.parquet")
    out, prev_close = [], None
    for d, g in df.groupby(df.index.date):
        if len(g) != 78 or g.index[0].strftime("%H:%M") != "09:30":
            prev_close = g["close"].iloc[-1] if len(g) else prev_close
            continue
        out.append(dict(date=pd.Timestamp(d), o=g["open"].values, h=g["high"].values, l=g["low"].values, c=g["close"].values,
                        v=g["volume"].values, prev_close=prev_close))
        prev_close = g["close"].iloc[-1]
    return out


def orb(D, long_short):
    t = []
    for d in D:
        hi, lo = d["h"][:6].max(), d["l"][:6].min()
        for i in range(6, 77):
            side = 1 if d["c"][i] > hi else (-1 if (long_short and d["c"][i] < lo) else 0)
            if side == 0:
                continue
            entry, stop = d["o"][i + 1], (lo if side == 1 else hi)
            exit_px = d["c"][77]
            for j in range(i + 1, 78):
                if side == 1 and d["l"][j] <= stop:
                    exit_px = min(stop, d["o"][j]); break
                if side == -1 and d["h"][j] >= stop:
                    exit_px = max(stop, d["o"][j]); break
            t.append((d["date"], side * (exit_px / entry - 1)))
            break
    return t


def mom(D, long_short):
    t = []
    for d in D:
        if d["prev_close"] is None:
            continue
        r = d["c"][5] / d["prev_close"] - 1
        side = 1 if r > 0 else (-1 if long_short and r < 0 else 0)
        if side:
            t.append((d["date"], side * (d["c"][77] / d["o"][66] - 1)))
    return t


def vwap(D, long_short, K):
    t = []
    for d in D:
        typ = (d["h"] + d["l"] + d["c"]) / 3
        vw = np.cumsum(typ * d["v"]) / np.maximum(np.cumsum(d["v"]), 1)
        for i in range(12, 76):
            dev = d["c"][i] / vw[i] - 1
            side = 1 if dev < -K else (-1 if (long_short and dev > K) else 0)
            if not side:
                continue
            entry, exit_px = d["o"][i + 1], d["c"][77]
            for j in range(i + 1, 77):
                if (side == 1 and d["c"][j] >= vw[j]) or (side == -1 and d["c"][j] <= vw[j]):
                    exit_px = d["o"][j + 1]; break
            t.append((d["date"], side * (exit_px / entry - 1)))
            break
    return t


def stats(trades, all_days, cost_bp, a, b):
    idx = pd.DatetimeIndex([d["date"] for d in all_days])
    s = pd.Series(0.0, index=idx)
    n = 0
    for dt, r in trades:
        s[dt] += r - 2 * cost_bp / 1e4
        n += 1
    s = s.loc[a:b]
    tr = [(dt, r - 2 * cost_bp / 1e4) for dt, r in trades if pd.Timestamp(a) <= dt <= pd.Timestamp(b)]
    if len(s) < 100 or not tr:
        return None
    eq = (1 + s).cumprod()
    yrs = len(s) / 252
    wk = (1 + s).resample("W-FRI").prod() - 1
    rr = np.array([r for _, r in tr])
    return dict(trades=len(tr), per_yr=round(len(tr) / yrs), win=round((rr > 0).mean() * 100), avg_bp=round(rr.mean() * 1e4, 1),
                CAGR=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1), DD=round((eq / eq.cummax() - 1).min() * 100, 1),
                Sharpe=round(s.mean() / s.std() * np.sqrt(252), 2) if s.std() > 0 else np.nan, wk_avg=round(wk.mean() * CAP), wk_med=round(wk.median() * CAP))


rows = []
for sym in ("SPY", "QQQ"):
    D = days(sym)
    print(f"{sym}: {len(D)} complete days {D[0]['date'].date()} .. {D[-1]['date'].date()}")
    strat = {}
    for ls, tag in ((False, "long-only"), (True, "long-short")):
        strat[f"ORB {tag}"] = orb(D, ls)
        strat[f"MOM {tag}"] = mom(D, ls)
        strat[f"VWAP K=0.25% {tag}"] = vwap(D, ls, 0.0025)
        strat[f"VWAP K=0.50% {tag}"] = vwap(D, ls, 0.0050)
    for name, tr in strat.items():
        for bp in COSTS_BP:
            for per, (a, b) in {"2016-21 (read)": ("2016-01-01", "2021-12-31"), "2022-now (TEST)": ("2022-01-01", "2026-12-31")}.items():
                st = stats(tr, D, bp, a, b)
                if st:
                    rows.append(dict(asset=sym, strategy=name, cost_bp=bp, period=per, **st))

df = pd.DataFrame(rows)
df.to_csv("research/intraday_scan_results.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 1000)
for sym in ("SPY", "QQQ"):
    for bp in (1, 2):
        print(f"\n===== {sym}, cost {bp}bp/side =====")
        t = df[(df.asset == sym) & (df.cost_bp == bp)].drop(columns=["asset", "cost_bp"])
        print(t.to_string(index=False))
