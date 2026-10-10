"""E011: parity check - native LumiBot backtests (AlpacaBacktesting, daily bars, SIP/crypto data) vs the pandas simulation used in E005.

Same rule, same window, per sleeve: SPY>SMA200 (SHV when off), QQQ>SMA200 (SHV), BTC>SMA50 (cash); each with 8% stop and with no stop.
LumiBot: signal on the last completed daily close, orders filled by LumiBot's broker simulation, fees 0.05% (stocks) / 0.25% (BTC) per side.
Pandas: research/portfolio_backtest.py sleeve model (1-day-lag signal, fills at the close, stop vs daily low).
"""
import glob
import json
import os
import sys
from datetime import datetime

os.environ["MPLBACKEND"] = "Agg"  # no GUI backend: Tk breaks when LumiBot plots from worker threads across runs

import numpy as np
import pandas as pd

sys.path.insert(0, "strategies")
sys.path.insert(0, "research")
from lumi_trend_sleeve import run_backtest  # noqa: E402

START, END, BUDGET = datetime(2021, 6, 1), datetime(2026, 9, 30), 20_000
CASES = [("SPY", "stock", 200, "SHV"), ("QQQ", "stock", 200, "SHV"), ("BTC", "crypto", 50, None)]


def lumi(sym, kind, n, safe, stop):
    label = f"Parity_{sym}_{'stop' if stop else 'nostop'}"
    run_backtest(sym, kind, n, safe, stop, START, END, BUDGET, 0.0025 if kind == "crypto" else 0.0005, label)
    m = sorted(glob.glob(f"logs/{label}_*tearsheet_metrics.json"), key=os.path.getmtime)[-1]
    d = json.load(open(m))
    return d, m


def find(d, *keys):
    """tearsheet_metrics.json layout is not documented; search recursively for a key (case-insensitive substring)."""
    out = {}

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}/{k}")
        else:
            out[path.lower()] = x

    walk(d)
    for k in keys:
        for p, v in out.items():
            if k in p:
                return v
    return None


# --- pandas side (E005 model) ---
import portfolio_backtest as pb  # noqa: E402


def pandas_side(sym, stop):
    key = {"SPY": "SPY", "QQQ": "QQQ", "BTC": "BTC-USD"}[sym]
    r = pb.sleeve_ret[(stop, key)].loc[START:END]
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"total%": (eq.iloc[-1] - 1) * 100, "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "maxDD%": (eq / eq.cummax() - 1).min() * 100}


rows = []
for sym, kind, n, safe in CASES:
    for stop in (0.08, None):
        p = pandas_side(sym, stop)
        d, path = lumi(sym, kind, n, safe, stop)
        sm = d["scalar_metrics"]
        L = {k: sm[k]["Strategy"] * 100 for k in ("Total Return", "CAGR% (Annual Return)", "Max Drawdown")}
        rows.append({"sleeve": sym, "stop": stop or "none",
                     "LUMI total%": round(L["Total Return"], 1), "pandas total%": round(p["total%"], 1),
                     "LUMI CAGR%": round(L["CAGR% (Annual Return)"], 1), "pandas CAGR%": round(p["CAGR%"], 1),
                     "LUMI DD%": round(L["Max Drawdown"], 1), "pandas DD%": round(p["maxDD%"], 1)})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(df.to_string(index=False))
