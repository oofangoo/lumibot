"""Daily runner for the 'trend portfolio' (E005): SPY>SMA200 40%, QQQ>SMA200 40%, BTC>SMA50 20% of a $50k budget.

Meant to be run once a day by Windows Task Scheduler (not a long-running bot). Each run:
  1. computes each sleeve's trend signal from the last COMPLETED daily close (one-day lag, as in the backtest)
  2. works out target dollars per instrument (risk asset when trend is on, SHV T-bills for the stock sleeves when off,
     cash for BTC when off), trades only on a signal change, a new calendar month (rebalance), or a missing position
  3. keeps a broker-side GTC stop 8% below the average entry on every risk position (survives the PC being off)

Safety: PAPER ONLY (refuses unless ALPACA_IS_PAPER=true), touches ONLY SPY, QQQ, SHV and BTC/USD, dry-run by default.
  .\\run.ps1 strategies\\daily_runner.py            # prints the plan, places nothing
  .\\run.ps1 strategies\\daily_runner.py --execute  # places paper orders
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent.parent
load_dotenv(HERE / ".env")

CAPITAL = 50_000
STOP_PCT = 0.08
MIN_TRADE_USD = 200
SLEEVES = {  # name: weight, SMA days, yahoo symbol, alpaca symbol, safe asset when trend off
    "SPY": dict(weight=0.40, sma=200, yahoo="SPY", alpaca="SPY", safe="SHV"),
    "QQQ": dict(weight=0.40, sma=200, yahoo="QQQ", alpaca="QQQ", safe="SHV"),
    "BTC": dict(weight=0.20, sma=50, yahoo="BTC-USD", alpaca="BTC/USD", safe=None),
}
MANAGED = {"SPY", "QQQ", "SHV", "BTCUSD"}  # position-symbol form; nothing else is ever touched
STATE_FILE = HERE / "state" / "trend_portfolio.json"
LOG_FILE = HERE / "logs" / "daily_runner.log"
EXECUTE = "--execute" in sys.argv
# --only BTC  (or SPY,QQQ): act on just those sleeves, e.g. BTC trades 24/7 while stocks wait for the open.
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else set()
ACTIVE = {n: sl for n, sl in SLEEVES.items() if not ONLY or n in ONLY}


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_rebalance_month": None, "sleeves": {k: {"signal": None, "holding": False, "stopped": False} for k in SLEEVES}}


def save_state(st):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=2))


def signal_for(sl):
    df = yf.download(sl["yahoo"], period="2y", progress=False, auto_adjust=True)
    close = df["Close"].squeeze().dropna()
    today = datetime.now(timezone.utc).date() if sl["yahoo"] == "BTC-USD" else pd.Timestamp.now(tz="America/New_York").date()
    close = close[[d.date() < today for d in close.index]]  # completed bars only
    if len(close) < sl["sma"] + 5:
        raise RuntimeError(f"not enough history for {sl['yahoo']}")
    age = (today - close.index[-1].date()).days
    if age > 5:
        raise RuntimeError(f"stale data for {sl['yahoo']}: last bar {close.index[-1].date()}")
    last, sma = float(close.iloc[-1]), float(close.tail(sl["sma"]).mean())
    return last > sma, last, sma, close.index[-1].date()


def main():
    assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: ALPACA_IS_PAPER is not true"
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoLatestTradeRequest, StockLatestTradeRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopLimitOrderRequest, StopOrderRequest

    key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"]
    tc = TradingClient(key, secret, paper=True)
    sdc, cdc = StockHistoricalDataClient(key, secret), CryptoHistoricalDataClient()
    log(f"=== run start (execute={EXECUTE}) ===")

    def price(sym):
        if sym == "BTC/USD":
            return float(cdc.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=sym))[sym].price)
        return float(sdc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym))[sym].price)

    st = load_state()
    positions = {p.symbol: p for p in tc.get_all_positions() if p.symbol in MANAGED}
    open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    now_month = f"{datetime.now():%Y-%m}"
    month_rebalance = st["last_rebalance_month"] != now_month

    # --- 1. signals + stop-out detection -------------------------------------------------------------
    plan, any_trade = {}, False
    for name, sl in ACTIVE.items():
        s = st["sleeves"][name]
        on, last, sma, bar = signal_for(sl)
        sym = sl["alpaca"].replace("/", "")
        if s["holding"] and sym not in positions and on:
            s["stopped"], s["holding"] = True, False
            log(f"{name}: position gone while trend still on -> stop-loss assumed triggered; sitting out until trend resets")
        if not on:
            s["stopped"] = False
        changed = s["signal"] is not None and s["signal"] != on
        log(f"{name}: last close {last:.2f} ({bar}) vs SMA{sl['sma']} {sma:.2f} -> trend {'ON' if on else 'OFF'}"
            f"{' (CHANGED)' if changed else ''}{' [stopped out]' if s['stopped'] else ''}")
        s["signal"] = bool(on)
        plan[name] = dict(on=on and not s["stopped"], changed=changed)
        any_trade |= changed

    # --- 2. target dollars per instrument ------------------------------------------------------------
    target = {}
    for sl in ACTIVE.values():
        target.setdefault(sl["alpaca"].replace("/", ""), 0.0)
        if sl["safe"]:
            target.setdefault(sl["safe"], 0.0)
    for name, sl in ACTIVE.items():
        dollars = sl["weight"] * CAPITAL
        if plan[name]["on"]:
            target[sl["alpaca"].replace("/", "")] += dollars
        elif sl["safe"]:
            target[sl["safe"]] += dollars

    orders = []
    for sym, tgt in target.items():
        cur = float(positions[sym].market_value) if sym in positions else 0.0
        delta = tgt - cur
        zero_gap = (cur == 0 and tgt > 0) or (tgt == 0 and cur > 0)
        sig_changed = any(plan[n]["changed"] and (SLEEVES[n]["alpaca"].replace("/", "") == sym or SLEEVES[n]["safe"] == sym) for n in plan)
        if abs(delta) >= MIN_TRADE_USD and (zero_gap or sig_changed or month_rebalance):
            orders.append((sym, delta, cur, tgt))
    log("targets: " + ", ".join(f"{k}=${v:,.0f}" for k, v in target.items()))
    if not orders:
        log("no trades needed")
    for sym, delta, cur, tgt in orders:
        log(f"PLAN {'BUY ' if delta > 0 else 'SELL'} {sym}: ${abs(delta):,.0f}  (now ${cur:,.0f} -> target ${tgt:,.0f})")

    # --- 3. execute (sells first so cash is there) ----------------------------------------------------
    if EXECUTE:
        for sym, delta, cur, tgt in sorted(orders, key=lambda x: x[1]):
            asym = "BTC/USD" if sym == "BTCUSD" else sym
            for o in open_orders:  # cancel our own resting stops on this symbol before changing the position
                if o.symbol.replace("/", "") == sym:
                    tc.cancel_order_by_id(o.id)
            if delta < 0:
                if tgt == 0 and sym in positions:
                    tc.close_position(sym)
                    log(f"submitted CLOSE {sym}")
                else:
                    qty = round(abs(delta) / price(asym), 6) if sym == "BTCUSD" else int(abs(delta) // price(asym))
                    if qty > 0:
                        tc.submit_order(MarketOrderRequest(symbol=asym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC if sym == "BTCUSD" else TimeInForce.DAY))
                        log(f"submitted SELL {qty} {asym}")
            else:
                p_ = price(asym)
                qty = round(delta / p_, 6) if sym == "BTCUSD" else int(delta // p_)
                if qty > 0:
                    tc.submit_order(MarketOrderRequest(symbol=asym, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC if sym == "BTCUSD" else TimeInForce.DAY))
                    log(f"submitted BUY {qty} {asym} (~${qty * p_:,.0f})")
        if orders:
            time.sleep(10)
        for name, sl in ACTIVE.items():
            sym = sl["alpaca"].replace("/", "")
            st["sleeves"][name]["holding"] = bool(target[sym] > 0)
        if not ONLY:  # a partial run must not mark the whole portfolio as rebalanced this month
            st["last_rebalance_month"] = now_month

        # --- 4. ensure a resting stop under every risk position ---------------------------------------
        positions = {p.symbol: p for p in tc.get_all_positions() if p.symbol in MANAGED}
        open_orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for sym in [sl["alpaca"].replace("/", "") for sl in ACTIVE.values()]:
            if sym not in positions:
                continue
            p = positions[sym]
            asym = "BTC/USD" if sym == "BTCUSD" else sym
            have_stop = [o for o in open_orders if o.symbol.replace("/", "") == sym and "stop" in str(o.order_type).lower()]
            held_qty = float(p.qty)
            if have_stop and abs(sum(float(o.qty) for o in have_stop) - held_qty) < 1e-6:
                continue
            for o in have_stop:
                tc.cancel_order_by_id(o.id)
            stop_px = round(float(p.avg_entry_price) * (1 - STOP_PCT), 2)
            if sym == "BTCUSD":  # Alpaca crypto has no plain stop order: use stop-limit a hair below
                req = StopLimitOrderRequest(symbol=asym, qty=round(held_qty, 6), side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                            stop_price=stop_px, limit_price=round(stop_px * 0.995, 2))
            else:
                req = StopOrderRequest(symbol=asym, qty=int(held_qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC, stop_price=stop_px)
            tc.submit_order(req)
            log(f"placed resting stop on {sym}: qty {held_qty} @ {stop_px}")
        save_state(st)
    else:
        log("dry run: nothing placed, state not saved")


if __name__ == "__main__":
    main()
