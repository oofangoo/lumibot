"""One trend-following sleeve as a native LumiBot strategy (used to cross-check E005 and, later, to run live).

Rule: hold the risk asset while its last completed daily close is above its SMA(n); otherwise hold the safe asset
(T-bill ETF SHV) or cash. An optional resting GTC stop sits `stop_pct` below the fill price of every risk-asset buy.
After a stop-out the sleeve stays out until the trend turns off and on again (a fresh entry), as in research/portfolio_backtest.py.

Parameters (self.parameters):
  symbol      e.g. "SPY", "QQQ" or "BTC"
  kind        "stock" or "crypto"
  sma_days    200 for stocks, 50 for BTC in E005
  safe_symbol "SHV" or None (None = cash)
  stop_pct    0.08, or None for no stop
  max_capital None = use the whole account (compounding backtests); a dollar cap for live/paper use
"""
from datetime import datetime

import pytz
from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy


class TrendSleeve(Strategy):
    parameters = {"symbol": "SPY", "kind": "stock", "sma_days": 200, "safe_symbol": "SHV", "stop_pct": 0.08, "max_capital": None}

    # ---- helpers -------------------------------------------------------------------------------------------
    def _asset(self, symbol):
        if symbol is None:
            return None
        if self.parameters["kind"] == "crypto" and symbol == self.parameters["symbol"]:
            return Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
        return Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

    def _quote(self, asset):
        return Asset(symbol="USD", asset_type=Asset.AssetType.FOREX) if asset.asset_type == Asset.AssetType.CRYPTO else None

    def _qty(self, asset, dollars):
        price = self.get_last_price(asset, quote=self._quote(asset))
        if not price or price <= 0:
            return 0
        return round(dollars / price, 6) if asset.asset_type == Asset.AssetType.CRYPTO else int(dollars // price)

    def _held(self, asset):
        pos = self.get_position(asset)
        return float(pos.quantity) if pos is not None and pos.quantity else 0.0

    # ---- lifecycle -----------------------------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = "1D"
        if self.parameters["kind"] == "crypto":
            self.set_market("24/7")
        self.vars.stopped = False
        self.vars.pending_buy = None  # symbol to buy once the exit sell has filled

    def on_trading_iteration(self):
        p = self.parameters
        risk, safe = self._asset(p["symbol"]), self._asset(p["safe_symbol"])
        bars = self.get_historical_prices(risk, p["sma_days"] + 5, "day", quote=self._quote(risk))
        if bars is None or len(bars.df) < p["sma_days"]:
            self.log_message("not enough history yet")
            return
        close = bars.df["close"]
        trend_on = float(close.iloc[-1]) > float(close.tail(p["sma_days"]).mean())
        if not trend_on:
            self.vars.stopped = False  # a later 'on' is a fresh entry
        want = risk if (trend_on and not self.vars.stopped) else safe  # safe may be None -> cash

        have_risk, have_safe = self._held(risk) > 0, (safe is not None and self._held(safe) > 0)
        if (want == risk and have_risk) or (want == safe and want is not None and have_safe) or (want is None and not have_risk and not have_safe):
            return  # already positioned

        # exit whatever we should not hold; buy the target once the exit has filled (or right away if flat)
        exits = [a for a in (risk, safe) if a is not None and a != want and self._held(a) > 0]
        for a in exits:
            self.cancel_open_orders()  # cancels our resting stop before selling
            self.submit_order(self.create_order(a, self._held(a), "sell", quote=self._quote(a)))
        if exits:
            self.vars.pending_buy = want.symbol if want is not None else None
        elif want is not None:
            self._buy(want)

    def _buy(self, asset):
        cap = self.parameters["max_capital"]
        dollars = (self.portfolio_value if not cap else min(self.portfolio_value, cap)) * 0.98
        qty = self._qty(asset, dollars)
        if qty > 0:
            self.submit_order(self.create_order(asset, qty, "buy", quote=self._quote(asset)))

    def on_filled_order(self, position, order, price, quantity, multiplier):
        p = self.parameters
        sym = order.asset.symbol
        if order.side == "sell" and order.order_type == Order.OrderType.STOP:
            self.vars.stopped = True
            self.log_message(f"stop-loss filled on {sym} at {price}")
        elif order.side == "sell" and self.vars.pending_buy and self._held(order.asset) <= 0:
            tgt = self.vars.pending_buy
            self.vars.pending_buy = None
            self._buy(self._asset(tgt))
        elif order.side == "buy" and sym == p["symbol"] and p["stop_pct"]:
            stop = round(float(price) * (1 - p["stop_pct"]), 2)
            self.submit_order(self.create_order(order.asset, quantity, "sell", stop_price=stop, order_type=Order.OrderType.STOP,
                                                time_in_force="gtc", quote=self._quote(order.asset)))


def run_backtest(symbol, kind, sma_days, safe, stop_pct, start, end, budget, fee, label):
    from lumibot.backtesting import AlpacaBacktesting
    from lumibot.entities import TradingFee

    ny = pytz.timezone("America/New_York")
    return TrendSleeve.backtest(
        AlpacaBacktesting,
        backtesting_start=ny.localize(start),
        backtesting_end=ny.localize(end),
        benchmark_asset="SPY",
        budget=budget,
        name=label,
        parameters={"symbol": symbol, "kind": kind, "sma_days": sma_days, "safe_symbol": safe, "stop_pct": stop_pct, "max_capital": None},
        buy_trading_fees=[TradingFee(percent_fee=fee)],
        sell_trading_fees=[TradingFee(percent_fee=fee)],
        timestep="day",
        market="24/7" if kind == "crypto" else "NYSE",
        config=None,  # environment mode: ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_IS_PAPER from .env
        warm_up_trading_days=260,
        auto_adjust=True,
        show_plot=False,
        show_tearsheet=False,
        show_indicators=False,
    )


if __name__ == "__main__":
    import sys

    # usage: python lumi_trend_sleeve.py SPY stock 200 SHV 0.08 2020-01-01 2024-12-31
    sym, kind, n, safe, stop, a, b = sys.argv[1:8]
    run_backtest(sym, kind, int(n), None if safe == "None" else safe, None if stop == "None" else float(stop),
                 datetime.fromisoformat(a), datetime.fromisoformat(b), 20_000, 0.0025 if kind == "crypto" else 0.0005, f"Sleeve_{sym}")
