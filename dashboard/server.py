"""Read-only local dashboard for the paper-trading workspace.

    .\\run.ps1 dashboard\\server.py          ->  http://127.0.0.1:8765

Safety: binds to 127.0.0.1 only, GET only (no way to place/cancel orders from here), refuses to start unless
ALPACA_IS_PAPER=true, and never sends API keys to the browser (the browser only sees /api/status JSON).

Bot-only performance is reconstructed from the bot's own fills (Alpaca order history since BOT_START) plus daily closes,
so it is not polluted by manual trades or legacy positions in the same paper account.
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "strategies"))
import daily_runner as dr  # noqa: E402  (loads .env; reuses sleeve definitions + signal code; main() is not run)

PORT = 8765
DOCS = {"diary": WS / "DIARY.md", "experiments": WS / "EXPERIMENT_LOG.md", "lumibot": WS / "LUMIBOT_NOTES.md"}
LOGDIR = WS / "logs"
LOGS = {"stocks decisions": LOGDIR / "lumi_stocks_decisions.log", "btc decisions": LOGDIR / "lumi_btc_decisions.log",
        "stocks run output": LOGDIR / "lumi_stocks.log", "btc run output": LOGDIR / "lumi_btc.log",
        "run status": LOGDIR / "run_status.csv", "old runner": LOGDIR / "daily_runner.log"}
MANAGED = dr.MANAGED                      # {"SPY","QQQ","SHV","BTCUSD"}
BUDGET = dr.CAPITAL                       # 50,000
STOP_PCT = dr.STOP_PCT                    # 0.08
BOT_START = "2026-10-09T10:05:00+00:00"   # orders before this predate the bot (incl. the old IREN bracket order)
DD_HALT = 0.20                            # strategy's per-group drawdown halt (lumi_portfolio.py)
GROUPS = {"stocks": {"state": WS / "state" / "lumi_stocks.json", "symbols": {"SPY", "QQQ", "SHV"}, "weight": 0.80, "task": "LumiTrend-Stocks-PAPER", "max_age_h": 100},
          "btc": {"state": WS / "state" / "lumi_btc.json", "symbols": {"BTCUSD"}, "weight": 0.20, "task": "LumiTrend-BTC-PAPER", "max_age_h": 30}}
SLEEVE_SYMBOL = {"SPY": "SPY", "QQQ": "QQQ", "BTC": "BTCUSD"}
SLEEVE_GROUP = {"SPY": "stocks", "QQQ": "stocks", "BTC": "btc"}
TASKS = ["LumiTrend-Stocks-PAPER", "LumiTrend-BTC-PAPER", "LumibotTrendPortfolio-PAPER"]  # last = old alpaca-py runner (must stay disabled)
OLD_TASK = "LumibotTrendPortfolio-PAPER"
# What the backtests said to expect (EXPERIMENT_LOG E013, 2021-06..now window, BTC SMA50 + 8% stops, 40/40/20 monthly rebalance)
EXPECT = {"cagr": 15.5, "maxdd": -24.2, "sharpe": 1.12, "wk_avg": 140, "wk_med": 60, "goal_wk": 100,
          "min_weeks": 4, "ideal_weeks": 8, "earliest_review": "2026-11-06", "ideal_review": "2026-12-04"}

assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing to start: ALPACA_IS_PAPER is not true"

from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest  # noqa: E402

tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)

_cache, _lock = {}, threading.Lock()


def cached(key, ttl, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    try:
        val = fn()
    except Exception as e:  # one failing section must not blank the whole page
        val = {"error": f"{type(e).__name__}: {e}"}
    with _lock:
        _cache[key] = (time.time(), val)
    return val


def f(x):
    return None if x is None else float(x)


def enum(x):
    return getattr(x, "value", str(x))


def ok(x):
    return isinstance(x, list)


def norm(sym):
    return sym.replace("/", "")


# ------------------------------------------------------------------------------------------------ broker data
def account():
    a = tc.get_account()
    eq, last = f(a.equity), f(a.last_equity)
    return {"equity": eq, "last_equity": last, "day_change": None if last is None else eq - last, "cash": f(a.cash),
            "buying_power": f(a.buying_power), "status": enum(a.status), "trading_blocked": bool(a.trading_blocked)}


def clock():
    c = tc.get_clock()
    return {"is_open": bool(c.is_open), "next_open": c.next_open.isoformat(), "next_close": c.next_close.isoformat()}


def positions():
    out = []
    for p in tc.get_all_positions():
        out.append({"symbol": p.symbol, "qty": f(p.qty), "avg_entry": f(p.avg_entry_price), "price": f(p.current_price),
                    "value": f(p.market_value), "pl": f(p.unrealized_pl), "plpc": f(p.unrealized_plpc), "bot": p.symbol in MANAGED})
    return out


def orders():
    out = []
    for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN)):
        out.append({"symbol": o.symbol, "side": enum(o.side), "type": enum(o.order_type), "qty": f(o.qty),
                    "stop": f(o.stop_price), "limit": f(o.limit_price), "tif": enum(o.time_in_force),
                    "status": enum(o.status), "created": o.created_at.isoformat() if o.created_at else None, "bot": norm(o.symbol) in MANAGED})
    return out


def order_history():
    out = []
    for o in tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=80, nested=True)):
        created = o.created_at.isoformat() if o.created_at else None
        out.append({"created": created, "symbol": o.symbol, "side": enum(o.side), "type": enum(o.order_type),
                    "klass": enum(o.order_class) if o.order_class else "simple", "legs": len(o.legs) if getattr(o, "legs", None) else 0,
                    "status": enum(o.status), "qty": f(o.qty), "fill": f(o.filled_avg_price), "legacy": bool(created and created < BOT_START),
                    "bot": norm(o.symbol) in MANAGED})
    return out


def account_history():
    h = tc.get_portfolio_history(GetPortfolioHistoryRequest(period="1M", timeframe="1D"))
    return [{"t": t, "equity": e} for t, e in zip(h.timestamp or [], h.equity or []) if e]  # skip pre-funding days (equity 0)


def bot_fills():
    """Every filled (even partially) order on the bot's symbols since BOT_START, oldest first."""
    since = datetime.fromisoformat(BOT_START)
    res = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=since, limit=500, nested=False))
    out = []
    for o in res:
        q = f(o.filled_qty) or 0.0
        if q > 0 and o.filled_at and f(o.filled_avg_price) and norm(o.symbol) in MANAGED:
            out.append({"ts": o.filled_at, "sym": norm(o.symbol), "side": enum(o.side), "qty": q, "price": f(o.filled_avg_price)})
    out.sort(key=lambda x: x["ts"])
    return {"fills": out, "truncated": len(res) >= 500}


def daily_closes(start):
    import yfinance as yf
    df = yf.download(["SPY", "QQQ", "SHV", "BTC-USD"], start=str(start), progress=False, auto_adjust=False)["Close"]
    df.columns = [c.replace("-", "") for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.sort_index()


# ------------------------------------------------------------------------------------------------ bot-only performance
def bot_performance(live_positions):
    fd = cached("fills", 60, bot_fills)
    if isinstance(fd, dict) and fd.get("error"):
        return fd
    fills = fd["fills"]
    if not fills:
        return {"points": [], "weeks": [], "note": "No bot fills yet."}
    first = fills[0]["ts"].astimezone(timezone.utc).date()
    today = datetime.now(timezone.utc).date()
    cl = cached("closes", 900, lambda: daily_closes(first - timedelta(days=10)))
    if isinstance(cl, dict) and cl.get("error"):
        return cl
    cl = cl.reindex(pd.date_range(cl.index.min(), pd.Timestamp(today))).ffill()
    spy_fills = [x for x in fills if x["sym"] == "SPY" and x["side"] == "buy"]
    spy_ref = spy_fills[0]["price"] if spy_fills else float(cl["SPY"].loc[pd.Timestamp(first):].dropna().iloc[0])

    pts = []
    for d in pd.date_range(first, today):
        qty, invested = {}, 0.0
        for x in fills:
            if x["ts"].astimezone(timezone.utc).date() <= d.date():
                sgn = 1 if x["side"] == "buy" else -1
                qty[x["sym"]] = qty.get(x["sym"], 0.0) + sgn * x["qty"]
                invested += sgn * x["qty"] * x["price"]
        value = sum(q * float(cl[s].loc[d]) for s, q in qty.items() if abs(q) > 1e-9 and s in cl.columns and pd.notna(cl[s].loc[d]))
        spy_px = float(cl["SPY"].loc[d]) if pd.notna(cl["SPY"].loc[d]) else spy_ref
        pts.append({"d": d.date().isoformat(), "nav": BUDGET - invested + value, "spy": BUDGET * spy_px / spy_ref, "invested": invested})
    if ok(live_positions) and pts:  # today's point uses live broker prices
        live_val = sum(p["value"] or 0 for p in live_positions if p["bot"])
        pts[-1]["nav"] = BUDGET - pts[-1]["invested"] + live_val
        spy_live = next((p["price"] for p in live_positions if p["symbol"] == "SPY" and p["price"]), None)
        if spy_live:
            pts[-1]["spy"] = BUDGET * spy_live / spy_ref
        pts[-1]["live"] = True

    navs = [BUDGET] + [p["nav"] for p in pts]
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    # calendar weeks (Mon-Sun): P&L = last NAV in week minus last NAV of the previous week (first week starts from the budget)
    weeks, prev_nav, prev_spy = [], BUDGET, BUDGET
    by_week = {}
    for p in pts:
        dt = datetime.fromisoformat(p["d"]).date()
        by_week[dt - timedelta(days=dt.weekday())] = p
    for mon in sorted(by_week):
        p = by_week[mon]
        weeks.append({"monday": mon.isoformat(), "end": p["d"], "start_nav": prev_nav, "end_nav": p["nav"], "pnl": p["nav"] - prev_nav,
                      "spy_pnl": p["spy"] - prev_spy, "in_progress": mon == today - timedelta(days=today.weekday())})
        prev_nav, prev_spy = p["nav"], p["spy"]
    now = pts[-1]
    full = [w for w in weeks if not w["in_progress"]]
    notional = sum(x["qty"] * x["price"] for x in fills)
    return {"points": pts, "weeks": weeks, "nav": now["nav"], "pnl": now["nav"] - BUDGET, "pnl_pct": (now["nav"] / BUDGET - 1) * 100,
            "spy_pnl": now["spy"] - BUDGET, "max_dd_pct": mdd * 100, "n_fills": len(fills), "notional": notional, "truncated": fd["truncated"],
            "avg_full_week": (sum(w["pnl"] for w in full) / len(full)) if full else None, "n_full_weeks": len(full)}


# ------------------------------------------------------------------------------------------------ signals, state, scheduler, logs
def btc_signal_alpaca(sl):
    """BTC daily closes from Alpaca (same source as the bot). Yahoo's BTC-USD series has been seen missing whole days."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cli = CryptoHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"])
    bars = cli.get_crypto_bars(CryptoBarsRequest(symbol_or_symbols="BTC/USD", timeframe=TimeFrame.Day,
                                                 start=datetime.now(timezone.utc) - timedelta(days=sl["sma"] + 25))).df
    close = bars["close"].droplevel(0) if isinstance(bars.index, pd.MultiIndex) else bars["close"]
    today = datetime.now(timezone.utc).date()
    close = close[[ts.date() < today for ts in close.index]]  # completed UTC days only
    if len(close) < sl["sma"] + 1:
        raise RuntimeError("not enough Alpaca BTC history")
    days = pd.Series([ts.date() for ts in close.index])
    if (days.diff().dropna().dt.days > 1).any():
        raise RuntimeError("gap in Alpaca BTC daily bars")
    last, sma = float(close.iloc[-1]), float(close.tail(sl["sma"]).mean())
    return last > sma, last, sma, close.index[-1].date()


def signals():
    out = {}
    for name, sl in dr.SLEEVES.items():
        try:
            try:
                if name != "BTC":
                    raise LookupError
                on, last, sma, bar = btc_signal_alpaca(sl)
                src = "Alpaca"
            except Exception:  # BTC falls back to Yahoo; stocks always use Yahoo (same as the old runner)
                on, last, sma, bar = dr.signal_for(sl)
                src = "Yahoo"
            out[name] = {"on": bool(on), "last": last, "sma": sma, "sma_days": sl["sma"], "bar": str(bar), "pct_from_sma": (last / sma - 1) * 100, "src": src}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def scheduler():
    names = ",".join(f"'{t}'" for t in TASKS)
    ps = ("$r=foreach($n in @(NAMES)){ $t=Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; if($t){ $i=Get-ScheduledTaskInfo -TaskName $n; "
          "[pscustomobject]@{name=$n; state=[string]$t.State; last=$i.LastRunTime.ToString('o'); result=$i.LastTaskResult; next=$i.NextRunTime.ToString('o')} } }; "
          "ConvertTo-Json -InputObject @($r)").replace("NAMES", names)
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return {"error": (r.stderr or "tasks not found").strip()[:200]}
    tasks = json.loads(r.stdout)
    for d in tasks:
        if d["last"].startswith("1999"):
            d["last"] = None  # Task Scheduler's "never ran" placeholder
    return {"tasks": tasks}


def read_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def run_history():
    p = LOGDIR / "run_status.csv"
    if not p.exists():
        return {"rows": [], "ok": 0, "failed": 0}
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    good = sum(1 for r in rows if r.get("exit_code") == "0")
    return {"rows": rows[-12:][::-1], "ok": good, "failed": len(rows) - good, "last": {g: next((r for r in reversed(rows) if r["group"] == g), None) for g in GROUPS}}


STABLE_BRANCH = "trading-workspace"


def code_version():
    """Which commit is the live bot running? The scheduler executes the working tree, so uncommitted edits to the live code matter."""
    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(WS), *a], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None
    if git("rev-parse", "--git-dir") is None:
        return {"tracked": False}
    dirty = (git("status", "--porcelain", "--", "strategies", "run.ps1", "run_scheduled.ps1") or "").splitlines()
    unpushed = git("rev-list", "--count", "@{u}..HEAD")
    return {"tracked": True, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "commit": git("rev-parse", "--short", "HEAD"),
            "describe": git("describe", "--tags", "--always"), "subject": git("log", "-1", "--format=%s"), "dirty": [ln[3:] for ln in dirty],
            "unpushed": int(unpushed) if unpushed and unpushed.isdigit() else None}


def merged_log(n=45):
    lines = []
    for tag, p in (("stocks", LOGS["stocks decisions"]), ("btc", LOGS["btc decisions"]), ("old", LOGS["old runner"])):
        if p.exists():
            for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]:
                lines.append((ln[:19], f"[{tag}] {ln}"))
    lines.sort(key=lambda x: x[0])
    return [x[1] for x in lines[-n:]]


def group_state(positions_):
    """Per-group live state: the strategy's state file plus its drawdown-halt headroom (same NAV estimate the strategy uses)."""
    out = {}
    held = {p["symbol"]: p for p in positions_} if ok(positions_) else {}
    for g, cfg in GROUPS.items():
        st = read_json(cfg["state"]) or {}
        capital = BUDGET * cfg["weight"]
        mine = [held[s] for s in cfg["symbols"] if s in held and held[s]["qty"]]
        value = sum(p["value"] or 0 for p in mine)
        cost = sum((p["qty"] or 0) * (p["avg_entry"] or 0) for p in mine)
        nav = value + max(0.0, capital - cost)
        hwm = st.get("hwm")
        trig = hwm * (1 - DD_HALT) if hwm else None
        out[g] = {"month": st.get("month"), "hwm": hwm, "halt_trigger": trig, "nav_est": nav, "capital": capital,
                  "halted": bool(trig and nav < trig), "headroom_pct": ((nav / trig - 1) * 100) if trig else None,
                  "rebalance_due": st.get("month") != datetime.now().strftime("%Y-%m"), "sleeves": st.get("sleeves", {})}
    return out


def protection(positions_, orders_):
    """Is every risk position covered by a resting sell stop for (almost) all of its quantity?"""
    if not ok(positions_) or not ok(orders_):
        return None
    out = []
    for sym in ("SPY", "QQQ", "BTCUSD"):
        p = next((x for x in positions_ if x["symbol"] == sym and x["qty"]), None)
        if not p:
            continue
        stops = [o for o in orders_ if norm(o["symbol"]) == sym and o["side"] == "sell" and o["type"] in ("stop", "stop_limit")]
        cover = sum(o["qty"] or 0 for o in stops)
        trig = max((o["stop"] for o in stops if o["stop"]), default=None)
        ratio = cover / p["qty"] if p["qty"] else 0
        out.append({"symbol": sym, "qty": p["qty"], "stop_qty": cover, "coverage": ratio, "stop": trig,
                    "limit": next((o["limit"] for o in stops if o["limit"]), None), "price": p["price"],
                    "drop_to_stop_pct": ((p["price"] / trig - 1) * 100) if trig and p["price"] else None,
                    "status": "protected" if ratio >= 0.999 else ("partial" if ratio > 0 else "none")})
    return out


def build_alerts(d):
    A = []

    def add(level, msg):
        A.append({"level": level, "msg": msg})
    a = d["account"]
    if isinstance(a, dict) and a.get("trading_blocked"):
        add("crit", "Alpaca reports trading_blocked on this account.")
    if (WS / "KILL").exists():
        add("warn", "KILL file present: the bot places no orders (stops at the broker stay).")
    prot = d["protection"]
    for p in prot or []:
        if p["status"] != "protected":
            add("crit", f"{p['symbol']}: {p['status'].upper()} stop protection (stop covers {p['coverage'] * 100:.0f}% of the position).")
    tasks = {t["name"]: t for t in (d["scheduler"].get("tasks", []) if isinstance(d["scheduler"], dict) else [])}
    if isinstance(d["scheduler"], dict) and d["scheduler"].get("error"):
        add("warn", f"Could not read Task Scheduler: {d['scheduler']['error']}")
    if OLD_TASK in tasks and tasks[OLD_TASK]["state"] != "Disabled":
        add("crit", f"Old runner task {OLD_TASK} is {tasks[OLD_TASK]['state']} - it must stay Disabled (two runners on one account).")
    now = datetime.now().astimezone()
    for g, cfg in GROUPS.items():
        t = tasks.get(cfg["task"])
        if not t:
            add("warn", f"Scheduled task {cfg['task']} not found.")
            continue
        if t["state"] == "Disabled":
            add("warn", f"{cfg['task']} is Disabled: the {g} sleeve is not being managed.")
        if t["last"] and t["result"] not in (0,):
            add("crit", f"{cfg['task']} last result code {t['result']} (non-zero = the run failed). Check the run output log.")
        if t["last"]:
            age_h = (now - datetime.fromisoformat(t["last"])).total_seconds() / 3600
            if age_h > cfg["max_age_h"] and t["state"] != "Disabled":
                add("warn", f"{g}: last scheduled run was {age_h:.0f}h ago (expected within {cfg['max_age_h']}h) - was the PC off?")
        elif d["run_history"].get("last", {}).get(g) is None and t["state"] != "Disabled":
            add("info", f"{g}: the scheduled task has not run yet (first run: {t['next'][:16].replace('T', ' ')}).")
    cv = d.get("code") or {}
    if cv.get("tracked") is False:
        add("warn", "The workspace is not under git: there is no history or backup of the live bot's code.")
    elif cv.get("tracked"):
        if cv["dirty"]:
            add("warn", f"Live bot code has UNCOMMITTED changes ({', '.join(cv['dirty'][:4])}): scheduled runs execute the working tree, not a reviewed commit.")
        if cv["branch"] != STABLE_BRANCH:
            add("warn", f"Live tree is on branch '{cv['branch']}', not the stable branch '{STABLE_BRANCH}'.")
        if cv.get("unpushed"):
            add("info", f"{cv['unpushed']} commit(s) not pushed to the remote yet (no off-machine backup of them).")
    rh = d["run_history"]
    if rh.get("failed"):
        add("warn", f"{rh['failed']} logged run(s) exited with a non-zero code (see run history).")
    gs = d["group_state"]
    for g, s in gs.items():
        if s["halted"]:
            add("warn", f"{g}: drawdown halt is ACTIVE (NAV {s['nav_est']:,.0f} below trigger {s['halt_trigger']:,.0f}); no new risk buys.")
        if s["hwm"] is None:
            add("info", f"{g}: drawdown high-water mark will be set on the next run.")
    sig = d["signals"]
    pos = {p["symbol"]: p for p in d["positions"]} if ok(d["positions"]) else {}
    if isinstance(sig, dict) and not sig.get("error"):
        for name, s in sig.items():
            if s.get("error"):
                add("warn", f"{name} signal: {s['error']}")
                continue
            sym = SLEEVE_SYMBOL[name]
            holding = sym in pos and (pos[sym]["value"] or 0) > 200
            stopped = bool(gs.get(SLEEVE_GROUP[name], {}).get("sleeves", {}).get(name, {}).get("stopped"))
            if s["on"] and not holding and not stopped:
                add("info", f"{name}: trend is ON but the position is not held - the next run will buy.")
            if (not s["on"]) and holding:
                add("warn", f"{name}: trend is OFF but still held - the next run should sell.")
            if abs(s["pct_from_sma"]) < 2.0:
                add("info", f"{name} is only {s['pct_from_sma']:+.1f}% from its {s['sma_days']}-day average - close to flipping.")
    for p in pos.values():
        if not p["bot"]:
            add("info", f"Non-bot position in the account: {p['symbol']} ({p['qty']:g}). The bot ignores it, and the account equity includes it.")
    if ok(d["orders"]):
        for o in d["orders"]:
            if not o["bot"]:
                add("info", f"Non-bot open order: {o['side']} {o['qty']:g} {o['symbol']} ({o['type']}, {o['status']}).")
    for k in ("account", "positions", "orders", "signals", "scheduler", "bot_perf"):
        if isinstance(d.get(k), dict) and d[k].get("error") and k not in ("scheduler",):
            add("warn", f"Data source problem ({k}): {d[k]['error']}")
    order = {"crit": 0, "warn": 1, "info": 2}
    A.sort(key=lambda x: order[x["level"]])
    return A


def status():
    t0 = time.time()
    pos = cached("pos", 15, positions)
    ords = cached("orders", 15, orders)
    sched = cached("sched", 60, scheduler)
    sig = cached("sig", 600, signals)
    held = {p["symbol"]: p["value"] for p in pos} if ok(pos) else {}
    targets = [{"sleeve": n, "weight": sl["weight"], "target": sl["weight"] * BUDGET, "symbol": norm(sl["alpaca"]), "held": held.get(norm(sl["alpaca"]), 0.0),
                "safe": sl["safe"], "safe_held": held.get(sl["safe"], 0.0) if sl["safe"] else None, "group": SLEEVE_GROUP[n]} for n, sl in dr.SLEEVES.items()]
    perf = cached("perf", 30, lambda: bot_performance(pos))
    rh = run_history()
    d = {"now": datetime.now().astimezone().isoformat(), "budget": BUDGET, "stop_pct": STOP_PCT, "expect": EXPECT, "bot_start": BOT_START,
         "account": cached("acct", 15, account), "clock": cached("clock", 30, clock), "positions": pos, "orders": ords,
         "order_history": cached("ohist", 30, order_history), "history": cached("hist", 120, account_history), "signals": sig,
         "targets": targets, "scheduler": sched, "run_history": rh, "log": merged_log(), "group_state": group_state(pos),
         "protection": protection(pos, ords), "bot_perf": perf, "code": cached("code", 20, code_version)}
    d["alerts"] = build_alerts(d)
    d["took_ms"] = int((time.time() - t0) * 1000)
    return d


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):  # blocks DNS-rebinding style access
            return self._send(403, "forbidden", "text/plain")
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (Path(__file__).parent / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._send(200, json.dumps(status(), default=str), "application/json")
        if u.path == "/api/doc":
            name = parse_qs(u.query).get("name", [""])[0]
            p = DOCS.get(name)
            return self._send(200, p.read_text(encoding="utf-8") if p and p.exists() else "(not found)", "text/plain; charset=utf-8")
        if u.path == "/api/logfile":  # whitelisted names only
            name = parse_qs(u.query).get("name", [""])[0]
            p = LOGS.get(name)
            txt = "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-300:]) if p and p.exists() else "(no such log yet)"
            return self._send(200, txt, "text/plain; charset=utf-8")
        if u.path == "/api/logs":
            return self._send(200, json.dumps(list(LOGS)), "application/json")
        self._send(404, "not found", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=status, daemon=True).start()  # warm the caches so the first page load is quick
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"Dashboard (read-only, paper) on http://127.0.0.1:{PORT}  - Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
