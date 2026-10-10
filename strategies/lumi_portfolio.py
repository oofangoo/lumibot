"""Trend portfolio as a native LumiBot strategy (backtest AND live-paper), replacing the hand-rolled alpaca-py runner.

Two groups run as two separate strategies (separate processes) because stocks and BTC have different market hours:
  stocks : SPY, QQQ  -> hold the risk asset while last completed close > SMA200, else SHV (T-bills)      market NYSE
  btc    : BTC       -> hold BTC while last completed close > SMA50, else cash                            market 24/7
Each sleeve gets weight x budget (default 40/40/20 of $50k). Trades only on a signal change, a new calendar month (rebalance),
or a missing position. A resting GTC stop sits stop_pct below every risk-asset fill; after a stop-out the sleeve waits for the
trend to turn off and on again. Same rules as research/portfolio_backtest.py / daily_runner.py.

Safety (code, not an agent): symbol allowlist; targets may never exceed the budget (fail closed); KILL file stops all trading;
drawdown halt blocks NEW risk buys; the account's other positions are never touched.

Backtest:  .\\run.ps1 strategies\\lumi_portfolio.py backtest stocks 2021-06-01 2026-09-30
Live:      .\\run.ps1 strategies\\lumi_portfolio.py live stocks       (paper; one iteration, then exits)
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy

WS = Path(__file__).resolve().parent.parent
KILL_FILE = WS / "KILL"
ALLOWED = {"SPY", "QQQ", "SHV", "BTC"}
GROUPS = {
    "stocks": {"market": "NYSE", "sleeves": {"SPY": dict(weight=0.40, sma=200, kind="stock", safe="SHV"),
                                              "QQQ": dict(weight=0.40, sma=200, kind="stock", safe="SHV")}},
    "btc": {"market": "24/7", "sleeves": {"BTC": dict(weight=0.20, sma=50, kind="crypto", safe=None)}},
}


class TrendPortfolio(Strategy):
    parameters = {"group": "stocks", "budget_total": 50_000, "stop_pct": 0.08, "min_trade_usd": 200, "dd_halt": 0.20, "use_state_file": True}

    # ---------------------------------------------------------------- helpers
    def log_message(self, message, *args, **kwargs):
        """Live runs also append every decision to logs/lumi_<group>_decisions.log (LumiBot prints nothing in run-once mode)."""
        if not self.is_backtesting:
            try:
                path = WS / "logs" / f"lumi_{self.parameters['group']}_decisions.log"
                path.parent.mkdir(exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
            except Exception:
                pass  # logging must never block trading
        return super().log_message(message, *args, **kwargs)

    def initialize(self):
        self.sleeptime = "1D"
        grp = GROUPS[self.parameters["group"]]
        if grp["market"] == "24/7":
            self.set_market("24/7")
        self.vars.mem = {"month": None, "hwm": None, "sleeves": {}}
        self._state_path = WS / "state" / f"lumi_{self.parameters['group']}.json"

    def _cfg(self):
        return GROUPS[self.parameters["group"]]["sleeves"]

    def _asset(self, sym):
        crypto = sym == "BTC"
        return Asset(symbol=sym, asset_type=Asset.AssetType.CRYPTO if crypto else Asset.AssetType.STOCK)

    def _quote(self, asset):
        return Asset(symbol="USD", asset_type=Asset.AssetType.FOREX) if asset.asset_type == Asset.AssetType.CRYPTO else None

    def _held(self, asset):
        pos = self.get_position(asset)
        return float(pos.quantity) if pos is not None and pos.quantity else 0.0

    def _price(self, asset):
        return self.get_last_price(asset, quote=self._quote(asset))

    def _load(self):
        if self.is_backtesting or not self.parameters["use_state_file"]:
            return self.vars.mem
        if self._state_path.exists():
            return json.loads(self._state_path.read_text(encoding="utf-8-sig"))  # tolerate a BOM if the file was edited by hand
        return {"month": None, "hwm": None, "sleeves": {}}

    def _save(self, st):
        if self.is_backtesting or not self.parameters["use_state_file"]:
            self.vars.mem = st
            return
        self._state_path.parent.mkdir(exist_ok=True)
        self._state_path.write_text(json.dumps(st, indent=2))

    def _signal(self, sym, sma):
        asset = self._asset(sym)
        bars = self.get_historical_prices(asset, sma + 8, "day", quote=self._quote(asset))
        if bars is None or len(bars.df) < sma + 1:
            return None
        df = bars.df
        if not self.is_backtesting:
            # live: completed daily bars only, as in the backtests. A bar is complete once its whole day has elapsed. (Comparing dates is
            # wrong for crypto: LumiBot labels the BTC day bar 00:00 UTC = 20:00 ET of the PREVIOUS date, so the still-forming bar never
            # matched "today" and every live BTC signal used a half-finished day. Verified with a probe on 2026-10-10.)
            now = self.get_datetime()
            now = now if df.index.tz is not None else now.replace(tzinfo=None)
            df = df[[ts + timedelta(days=1) <= now for ts in df.index]]
        close = df["close"]
        if len(close) < sma:
            return None
        return float(close.iloc[-1]) > float(close.tail(sma).mean()), float(close.iloc[-1]), float(close.tail(sma).mean())

    def _stop_orders(self, asset):
        return [o for o in self.get_orders() if o.asset.symbol == asset.symbol and o.is_active()
                and o.side == "sell" and o.order_type in (Order.OrderType.STOP, Order.OrderType.STOP_LIMIT)]

    def _cancel_stops(self, asset):
        for o in self._stop_orders(asset):
            self.cancel_order(o)

    def _ensure_stop(self, asset, ref_price=None):
        """Idempotent: make sure exactly one resting stop protects the held risk position."""
        pct = self.parameters["stop_pct"]
        qty = self._held(asset)
        if not pct or qty <= 0 or self._stop_orders(asset):
            return
        pos = self.get_position(asset)
        entry = ref_price or (float(pos.avg_fill_price) if pos is not None and pos.avg_fill_price else self._price(asset))
        stop = round(float(entry) * (1 - pct), 2)
        crypto = asset.asset_type == Asset.AssetType.CRYPTO
        kwargs = dict(stop_price=stop, time_in_force="gtc", quote=self._quote(asset))
        if crypto:  # Alpaca has no plain stop for crypto: stop-limit a hair below
            kwargs.update(stop_limit_price=round(stop * 0.995, 2), order_type=Order.OrderType.STOP_LIMIT)
        else:
            kwargs.update(order_type=Order.OrderType.STOP)
        # crypto: ROUND DOWN to 6 decimals (rounding up asks to sell more than is held and Alpaca rejects it; found in the live test)
        self.submit_order(self.create_order(asset, math.floor(qty * 1e6) / 1e6 if crypto else int(qty), "sell", **kwargs))
        self.log_message(f"resting stop placed on {asset.symbol}: {stop}")

    # ---------------------------------------------------------------- main loop
    def on_trading_iteration(self):
        p, cfg = self.parameters, self._cfg()
        if KILL_FILE.exists():
            self.log_message("KILL file present: no trading")
            return
        st = self._load()
        wsum = sum(s["weight"] for s in cfg.values())
        capital = self.portfolio_value if self.is_backtesting else p["budget_total"] * wsum
        month = self.get_datetime().strftime("%Y-%m")
        rebalance = st["month"] != month

        # 1. signals and stop-out detection
        plan = {}
        for name, s in cfg.items():
            sig = self._signal(name, s["sma"])
            if sig is None:
                self.log_message(f"{name}: not enough history; skipping this iteration")
                return
            on, last, sma = sig
            ss = st["sleeves"].setdefault(name, {"signal": None, "holding": False, "stopped": False})
            held = self._held(self._asset(name)) > 0
            if ss["holding"] and not held and on:
                ss["stopped"], ss["holding"] = True, False
                self.log_message(f"{name}: position gone while trend still ON -> stop assumed triggered; waiting for trend reset")
            if not on:
                ss["stopped"] = False
            changed = ss["signal"] is not None and ss["signal"] != on
            self.log_message(f"{name}: close {last:.2f} vs SMA{s['sma']} {sma:.2f} -> {'ON' if on else 'OFF'}{' (CHANGED)' if changed else ''}{' [stopped]' if ss['stopped'] else ''}")
            ss["signal"] = bool(on)
            plan[name] = dict(on=on and not ss["stopped"], changed=changed)

        # 2. target dollars per symbol
        targets = {}
        for name, s in cfg.items():
            targets.setdefault(name, 0.0)
            if s["safe"]:
                targets.setdefault(s["safe"], 0.0)
            dollars = capital * s["weight"] / wsum
            if plan[name]["on"]:
                targets[name] += dollars
            elif s["safe"]:
                targets[s["safe"]] += dollars

        # 3. risk gate (deterministic; fail closed)
        if not set(targets) <= ALLOWED:
            self.log_message(f"RISK GATE: symbol not allowed {set(targets) - ALLOWED}; no trading")
            return
        if sum(targets.values()) > capital * 1.001:
            self.log_message("RISK GATE: targets exceed budget; no trading")
            return
        if not self.is_backtesting and p["dd_halt"]:
            managed = [pp for pp in self.get_positions() if pp.symbol in targets and pp.quantity]
            value = sum(float(pp.quantity) * float(self._price(pp.asset) or 0) for pp in managed)
            cost = sum(float(pp.quantity) * float(pp.avg_fill_price or 0) for pp in managed)
            nav = value + max(0.0, capital - cost)
            st["hwm"] = max(st["hwm"] or nav, nav)
            if nav < st["hwm"] * (1 - p["dd_halt"]):
                self.log_message(f"RISK GATE: drawdown halt (NAV {nav:,.0f} vs high-water {st['hwm']:,.0f}); no new risk buys")
                for name in cfg:
                    a = self._asset(name)
                    targets[name] = min(targets[name], self._held(a) * float(self._price(a) or 0))  # never increase risk while halted

        # 4. orders: sells first, then buys
        def cur_value(sym):
            a = self._asset(sym)
            return self._held(a) * float(self._price(a) or 0)

        for sym, tgt in sorted(targets.items(), key=lambda kv: kv[1] - cur_value(kv[0])):
            a = self._asset(sym)
            cur, price = cur_value(sym), self._price(a)
            delta = tgt - cur
            sig_changed = any(plan[n]["changed"] and (n == sym or cfg[n]["safe"] == sym) for n in plan)
            zero_gap = (cur == 0 and tgt > 0) or (tgt == 0 and cur > 0)
            if abs(delta) < p["min_trade_usd"] or not price or not (zero_gap or sig_changed or rebalance):
                continue
            crypto = a.asset_type == Asset.AssetType.CRYPTO
            if delta < 0:
                qty = self._held(a) if tgt == 0 else (round(-delta / price, 6) if crypto else int(-delta // price))
                if qty <= 0:
                    continue
                self._cancel_stops(a)
                order = self.create_order(a, qty, "sell", quote=self._quote(a))
            else:
                qty = round(delta / price, 6) if crypto else int(delta // price)
                if qty <= 0:
                    continue
                self._cancel_stops(a) if sym in cfg else None
                order = self.create_order(a, qty, "buy", quote=self._quote(a))
            self.submit_order(order)
            self.log_message(f"{'SELL' if delta < 0 else 'BUY'} {qty} {sym} (~${abs(delta):,.0f}; now ${cur:,.0f} -> target ${tgt:,.0f})")
            if not self.is_backtesting:
                self.wait_for_order_execution(order)

        for name in cfg:
            st["sleeves"][name]["holding"] = bool(targets[name] > 0)
        st["month"] = month
        self._save(st)
        if not self.is_backtesting:
            for name in cfg:
                self._ensure_stop(self._asset(name))

    def on_filled_order(self, position, order, price, quantity, multiplier):
        if order.side == "buy" and order.asset.symbol in self._cfg():
            self._ensure_stop(order.asset, ref_price=float(price))
        elif order.side == "sell" and order.order_type in (Order.OrderType.STOP, Order.OrderType.STOP_LIMIT):
            self.log_message(f"stop-loss filled on {order.asset.symbol} at {price}")


# --------------------------------------------------------------------------------------- entry points
def backtest(group, start, end, budget=50_000, stop=0.08, label=None):
    import pytz
    from lumibot.backtesting import AlpacaBacktesting
    from lumibot.entities import TradingFee

    ny = pytz.timezone("America/New_York")
    fee = 0.0025 if group == "btc" else 0.0005
    return TrendPortfolio.backtest(
        AlpacaBacktesting, backtesting_start=ny.localize(start), backtesting_end=ny.localize(end), benchmark_asset="SPY", budget=budget,
        name=label or f"LumiPortfolio_{group}", parameters={"group": group, "stop_pct": stop, "dd_halt": None, "use_state_file": False},
        buy_trading_fees=[TradingFee(percent_fee=fee)], sell_trading_fees=[TradingFee(percent_fee=fee)],
        timestep="day", market=GROUPS[group]["market"], config=None, warm_up_trading_days=260, auto_adjust=True,
        show_plot=False, show_tearsheet=False, show_indicators=False)


def live(group, budget=50_000):
    from lumibot.credentials import ALPACA_CONFIG
    from lumi_alpaca_compat import RobustAlpaca  # tolerates old bracket orders in the account history (see that file)

    assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: ALPACA_IS_PAPER is not true"
    TrendPortfolio(broker=RobustAlpaca(ALPACA_CONFIG), parameters={"group": group, "budget_total": budget}).run_live(run_once=True)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    mode, group = sys.argv[1], sys.argv[2]
    if mode == "backtest":
        backtest(group, datetime.fromisoformat(sys.argv[3]), datetime.fromisoformat(sys.argv[4]))
    else:
        live(group, float(sys.argv[3]) if len(sys.argv) > 3 else 50_000)
