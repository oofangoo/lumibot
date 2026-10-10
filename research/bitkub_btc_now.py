"""E018: where is BTC right now, what did buy-and-hold deliver over different holding periods, and how did BTC behave after situations like today's?

Real data: Bitkub BTC/THB daily candles (2018-05..), and BTC-USD (Yahoo, 2014-09..) as a longer second opinion (more market cycles).
This is a description of history and base rates, NOT a forecast. Entry = the finished close of the day the state is observed; fee 0.25% on the
buy and on the final sell. Overlapping windows make neighbouring days highly dependent, so "independent samples" is reported as days / horizon.
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\bitkub_btc_now.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import load_bitkub, load_yahoo  # noqa: E402

FEE = 0.0025
NET = (1 - FEE) / (1 + FEE)  # buy once, sell once
HORIZONS = {"1 month": 30, "3 months": 91, "6 months": 182, "1 year": 365, "2 years": 730, "3 years": 1095, "4 years": 1460}


def snapshot(px, label):
    c = px["close"]
    last = c.iloc[-1]
    ath, ath_day = c.max(), c.idxmax()
    sma = {n: c.rolling(n).mean().iloc[-1] for n in (20, 50, 100, 200)}
    vol30 = np.log(c).diff().rolling(30).std() * np.sqrt(365)
    print(f"\n=== {label}: snapshot at the last FINISHED daily candle ({c.index[-1].date()}) ===")
    print(f"  close {last:,.0f}   |  ATH close {ath:,.0f} on {ath_day.date()} ({(c.index[-1] - ath_day).days} days ago)  |  drawdown from ATH {(last / ath - 1) * 100:.0f}%")
    print("  vs moving averages: " + "   ".join(f"SMA{n} {sma[n]:,.0f} ({(last / sma[n] - 1) * 100:+.1f}%)" for n in sma))
    print("  returns: " + "   ".join(f"{d}d {(last / c.iloc[-1 - d] - 1) * 100:+.0f}%" for d in (7, 30, 90, 180, 365)))
    print(f"  30-day realized volatility (annualized) {vol30.iloc[-1] * 100:.0f}%  = {(vol30.iloc[-1] <= vol30.dropna()).mean() * 100:.0f}th percentile from the top of its own history "
          f"(median {vol30.median() * 100:.0f}%)")
    above50, above200 = last > sma[50], last > sma[200]
    cross = "above" if sma[50] > sma[200] else "below"
    print(f"  trend state: price {'above' if above50 else 'below'} SMA50, {'above' if above200 else 'below'} SMA200; SMA50 is {cross} SMA200")
    # how long has the SMA50 signal been on, and how often does a signal this close to the line flip soon?
    sig = c > c.rolling(50).mean()
    run = (sig != sig.shift()).cumsum()
    print(f"  the SMA50 'buy' signal has been {'ON' if sig.iloc[-1] else 'OFF'} for {int(sig.groupby(run).cumcount().iloc[-1]) + 1} days")
    dist = (c / c.rolling(50).mean() - 1)
    near = (dist > 0) & (dist < dist.iloc[-1] * 1.5) & (dist > dist.iloc[-1] * 0.5)
    flips = [(sig.shift(-k).fillna(sig) != sig)[near].mean() for k in (14,)]
    print(f"  when BTC was only this far above its SMA50 ({dist.iloc[-1] * 100:.1f}%, band {dist.iloc[-1] * 50:.1f}-{dist.iloc[-1] * 150:.1f}%) the signal had flipped to OFF "
          f"14 days later {flips[0] * 100:.0f}% of the time (n={int(near.sum())} days)")
    return dict(above50=above50, above200=above200, dd=last / ath - 1)


def forward(px, h):
    c = px["close"]
    return (c.shift(-h) / c - 1) * NET


def describe(r, label, h):
    r = r.dropna()
    if len(r) < 20:
        return None
    return {"situation": label, "days": len(r), "~independent": max(1, round(len(r) / h)), "worst": f"{r.min() * 100:+.0f}%", "10th pct": f"{r.quantile(.1) * 100:+.0f}%",
            "median": f"{r.median() * 100:+.0f}%", "90th pct": f"{r.quantile(.9) * 100:+.0f}%", "best": f"{r.max() * 100:+.0f}%", "% positive": f"{(r > 0).mean() * 100:.0f}%"}


def episodes(mask):
    """Number of separate stretches (runs of consecutive days) a situation lasted."""
    return int(((mask) & (~mask.shift(fill_value=False))).sum())


def holding_table(px, label):
    print(f"\n=== {label}: buy on ANY day and hold (fee 0.25% each way) ===")
    rows = []
    for name, h in HORIZONS.items():
        d = describe(forward(px, h), "all start days", h)
        if d:
            years = h / 365
            d["median per year"] = f"{((1 + float(d['median'].rstrip('%')) / 100) ** (1 / years) - 1) * 100:+.0f}%"
            rows.append({"hold": name, **d})
    print(pd.DataFrame(rows).drop(columns=["situation"]).to_string(index=False))


def conditional_table(px, label, state):
    c = px["close"]
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    dd = c / c.cummax() - 1
    masks = {
        "all start days": pd.Series(True, index=c.index),
        f"SAME trend state as now ({'>' if state['above50'] else '<'}SMA50, {'>' if state['above200'] else '<'}SMA200)":
            ((c > sma50) == state["above50"]) & ((c > sma200) == state["above200"]) & sma200.notna(),
        "price > SMA50 (any SMA200)": (c > sma50) & sma50.notna(),
        "price < SMA200": (c < sma200) & sma200.notna(),
        f"drawdown from ATH between {abs(state['dd']) * 100 - 12:.0f}% and {abs(state['dd']) * 100 + 12:.0f}% (like now)":
            (dd < state["dd"] + 0.12) & (dd > state["dd"] - 0.12),
    }
    print(f"\n=== {label}: what happened after situations like today's (entry = that day's close) ===")
    for hname in ("3 months", "1 year", "2 years", "4 years"):
        h = HORIZONS[hname]
        rows = []
        for mname, m in masks.items():
            d = describe(forward(px, h)[m], mname, h)
            if d:
                d["episodes"] = episodes(m)
                rows.append(d)
        print(f"\n--- hold {hname} ---")
        print(pd.DataFrame(rows).to_string(index=False))


def thb_view(px):
    print("\n=== 10,000 THB bought today and held (BTC/THB history, all start days, fee 0.25% each way) ===")
    rows = []
    for name in ("1 year", "2 years", "4 years"):
        r = forward(px, HORIZONS[name]).dropna()
        rows.append({"hold": name, "worst start": f"{10000 * (1 + r.min()):,.0f}", "10th pct": f"{10000 * (1 + r.quantile(.1)):,.0f}",
                     "median": f"{10000 * (1 + r.median()):,.0f}", "90th pct": f"{10000 * (1 + r.quantile(.9)):,.0f}", "best start": f"{10000 * (1 + r.max()):,.0f}",
                     "ended below 10,000": f"{(r < 0).mean() * 100:.0f}%"})
    print(pd.DataFrame(rows).to_string(index=False))


def underwater(px, label):
    """How long did a buyer at an all-time high wait to get back to even? (and how deep was the hole)"""
    c = px["close"]
    peak = c.cummax()
    is_high = c >= peak
    print(chr(10) + f"=== {label}: buying at an all-time high, how long until you were back to even? ===")
    rows = []
    peaks = []  # a "peak" is an all-time high that is followed by at least 90 days without a new one
    prev = None
    for day in c.index[is_high]:
        if prev is not None and (day - prev).days > 90:
            peaks.append(prev)
        prev = day
    peaks.append(prev)  # the most recent high
    for p in peaks:
        after = c.loc[p:]
        recovered = after[after >= c.loc[p] * 1.0001].index
        recovered = recovered[recovered > p]
        trough = after.loc[: (recovered[0] if len(recovered) else after.index[-1])].min()
        rows.append({"high on": str(p.date()), "price": f"{c.loc[p]:,.0f}", "deepest loss after": f"{(trough / c.loc[p] - 1) * 100:.0f}%",
                     "back to even": (str(recovered[0].date()) + f" ({(recovered[0] - p).days / 365:.1f} yrs later)") if len(recovered) else "NOT YET"})
    print(pd.DataFrame(rows).to_string(index=False))


def lump_vs_dca(px, label, horizon=365, parts=4, gap=30):
    """Lump sum on day t vs the same money split into equal buys every ``gap`` days; both valued (and sold, fee 0.25%) ``horizon`` days after t."""
    c = px["close"]
    n = len(c)
    lump, dca = [], []
    for i in range(0, n - horizon):
        buy = c.iloc[i]
        lump.append(c.iloc[i + horizon] / buy * NET)
        units = 0.0
        for k in range(parts):
            units += (1.0 / parts) / (c.iloc[min(i + k * gap, n - 1)] * (1 + FEE))
        dca.append(units * c.iloc[i + horizon] * (1 - FEE))
    lump, dca = pd.Series(lump), pd.Series(dca)
    rows = [{"entry": f"all at once", "median": f"{(lump.median() - 1) * 100:+.0f}%", "10th pct": f"{(lump.quantile(.1) - 1) * 100:+.0f}%",
             "worst": f"{(lump.min() - 1) * 100:+.0f}%", "% ended below start": f"{(lump < 1).mean() * 100:.0f}%"},
            {"entry": f"{parts} equal buys, {gap} days apart", "median": f"{(dca.median() - 1) * 100:+.0f}%", "10th pct": f"{(dca.quantile(.1) - 1) * 100:+.0f}%",
             "worst": f"{(dca.min() - 1) * 100:+.0f}%", "% ended below start": f"{(dca < 1).mean() * 100:.0f}%"}]
    print(chr(10) + f"=== {label}: {horizon // 365}-year result of buying all at once vs spreading the entry ({(dca > lump).mean() * 100:.0f}% of start days the staged entry did better) ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    pd.set_option("display.width", 250, "display.max_columns", 30)
    thb = load_bitkub("1D")
    usd = load_yahoo()
    usd = usd.reindex(pd.date_range(usd.index[0], usd.index[-1], freq="D", tz="UTC")).ffill()  # Yahoo skips days; used for context only
    state_thb = snapshot(thb, "BTC / THB (Bitkub)")
    state_usd = snapshot(usd, "BTC / USD (Yahoo)")
    holding_table(thb, "BTC/THB (2018-05..now, ~2 market cycles)")
    holding_table(usd, "BTC/USD (2014-09..now, ~3 market cycles)")
    conditional_table(thb, "BTC/THB", state_thb)
    conditional_table(usd, "BTC/USD", state_usd)
    thb_view(thb)
    underwater(thb, "BTC/THB")
    underwater(usd, "BTC/USD")
    lump_vs_dca(thb, "BTC/THB")
    lump_vs_dca(usd, "BTC/USD")


if __name__ == "__main__":
    main()
