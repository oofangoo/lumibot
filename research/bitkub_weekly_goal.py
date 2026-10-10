"""E017a: what does a "5-10% per week" goal mean, and how close did BTC and the BTC trend rule ever get (real Bitkub BTC/THB)?

Reuses the cached Bitkub daily candles and the validated pandas trend model from bitkub_btc_trend.py. Fee 0.25%/side.
Run:  $env:PYTHONPATH='<path to the lumibot-dev clone>'; .\\run.ps1 research\\bitkub_weekly_goal.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bitkub_btc_trend import EVAL_START, load_bitkub, pandas_trend  # noqa: E402

FEE = 0.0025
START_CAPITAL = 10_000


def weekly_returns(equity: pd.Series) -> pd.Series:
    """Week = Monday-Sunday (UTC); return = last mark of the week over the last mark of the previous week."""
    w = equity.resample("W-SUN").last()
    return w.pct_change().dropna()


def dist(label, wr):
    return {
        "variant": label, "weeks": len(wr), "mean/wk%": round(wr.mean() * 100, 2), "median/wk%": round(wr.median() * 100, 2),
        "best wk%": round(wr.max() * 100, 1), "worst wk%": round(wr.min() * 100, 1),
        "wks>=+5%": f"{(wr >= 0.05).mean() * 100:.0f}%", "wks>=+10%": f"{(wr >= 0.10).mean() * 100:.0f}%",
        "wks<=-5%": f"{(wr <= -0.05).mean() * 100:.0f}%", "wks<=-10%": f"{(wr <= -0.10).mean() * 100:.0f}%",
        "wks>0": f"{(wr > 0).mean() * 100:.0f}%",
    }


def main():
    pd.set_option("display.width", 220, "display.max_columns", 30)
    px = load_bitkub("1D")
    print(f"BTC_THB daily candles {px.index[0].date()}..{px.index[-1].date()}\n")

    print("=== what a weekly goal compounds to (10,000 THB, no withdrawals, no losing weeks) ===")
    rows = []
    for g in (0.005, 0.01, 0.02, 0.05, 0.10):
        rows.append({"weekly": f"{g * 100:g}%", "per year": f"{(1 + g) ** 52:,.1f}x", "10,000 THB after 1 year": f"{START_CAPITAL * (1 + g) ** 52:,.0f}",
                     "after 2 years": f"{START_CAPITAL * (1 + g) ** 104:,.0f}", "= annual return": f"{((1 + g) ** 52 - 1) * 100:,.0f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

    for wname, start in (("2019-now (7.8 yrs, bull-dominated)", EVAL_START), ("2021-06-now (post-peak, 5.3 yrs)", pd.Timestamp("2021-06-01", tz="UTC"))):
        print(f"\n=== weekly return distribution, {wname} ===")
        rows = []
        bh, _ = pandas_trend(px, 50, FEE, 0, start=start, hold_only=True)
        rows.append(dist("BTC buy&hold", weekly_returns(bh)))
        for n in (50, 100, 200):
            eq, _ = pandas_trend(px, n, FEE, 0, start=start)
            rows.append(dist(f"BTC trend SMA{n}", weekly_returns(eq)))
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== how often did ANY 52-week window average >= 5% per week (i.e. >= 1,164%/yr)? ===")
    rows = []
    for label, eq in (("BTC buy&hold", pandas_trend(px, 50, FEE, 0, hold_only=True)[0]), ("BTC trend SMA50", pandas_trend(px, 50, FEE, 0)[0]),
                      ("BTC trend SMA100", pandas_trend(px, 100, FEE, 0)[0])):
        w = eq.resample("W-SUN").last()
        roll = (w / w.shift(52)).dropna() - 1  # 52-week total return for every starting week
        weekly_equiv = (1 + roll) ** (1 / 52) - 1
        rows.append({"variant": label, "52-wk windows": len(roll), "best 52-wk return": f"{roll.max() * 100:,.0f}%",
                     "median 52-wk return": f"{roll.median() * 100:,.0f}%", "worst 52-wk return": f"{roll.min() * 100:,.0f}%",
                     "windows >= 5%/wk avg": f"{(weekly_equiv >= 0.05).sum()} of {len(roll)}", "windows ending below start": f"{(roll < 0).mean() * 100:.0f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== what 10,000 THB really is, in costs (Bitkub min order ~10 THB; fee 0.25%/side) ===")
    print(f"  one round trip costs about {START_CAPITAL * FEE * 2:,.0f} THB (0.5% of the account) before any price move")
    for n in (50, 100, 200):
        _, trades = pandas_trend(px, n, FEE, 0)
        years = (px.index[-1] - EVAL_START).days / 365.25
        rt = len(trades) / 2 / years
        print(f"  SMA{n}: ~{rt:.1f} round trips/yr -> ~{rt * START_CAPITAL * FEE * 2:,.0f} THB/yr in fees on 10,000 THB ({rt * FEE * 2 * 100:.1f}% of the account)")




def start_52_weeks():
    """What would 10,000 THB have been after 52 weeks, for every possible starting week? (rolling windows, not an average)"""
    px = load_bitkub("1D")
    print("\n=== 10,000 THB after 52 weeks, for every starting week (BTC/THB, fee 0.25%/side) ===")
    rows = []
    for label, since in (("all starts 2019-now", EVAL_START), ("starts after 2021-06", pd.Timestamp("2021-06-01", tz="UTC"))):
        for name, kwargs in (("BTC buy&hold", dict(hold_only=True)), ("BTC trend SMA50", dict()), ("BTC trend SMA100", dict(n=100))):
            n = kwargs.pop("n", 50)
            eq, _ = pandas_trend(px, n, FEE, 0, start=EVAL_START, **kwargs)
            w = eq.resample("W-SUN").last()
            end_value = (w / w.shift(52) * START_CAPITAL).dropna()
            end_value = end_value[end_value.index - pd.Timedelta(weeks=52) >= since]
            q = end_value.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            rows.append({"starts": label, "variant": name, "windows": len(end_value), "worst": f"{end_value.min():,.0f}",
                         "10th pct": f"{q[0.1]:,.0f}", "25th pct": f"{q[0.25]:,.0f}", "median": f"{q[0.5]:,.0f}", "75th pct": f"{q[0.75]:,.0f}",
                         "90th pct": f"{q[0.9]:,.0f}", "best": f"{end_value.max():,.0f}", "ended below 10,000": f"{(end_value < START_CAPITAL).mean() * 100:.0f}%",
                         "needed for 5%/wk": f"{START_CAPITAL * 1.05 ** 52:,.0f}"})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
    start_52_weeks()
