"""Bitkub spot example: log prices and account state, with an opt-in order smoke test.

Bitkub is a real-money exchange with no paper mode. By default this strategy **places no orders**: it logs the BTC/THB
quote, your cash and your positions. Set ``place_test_order`` to ``True`` to run a deliberate smoke test: the first
iteration places one *limit buy far below the market* (so it should not fill) worth about ``test_order_thb`` baht, and
the next iteration cancels it. That exercises placement, open-order tracking and cancel without market risk.

Requires ``BITKUB_API_KEY`` and ``BITKUB_API_SECRET`` (market data alone needs none; see ``BitkubData``).
"""

from lumibot.brokers import Bitkub
from lumibot.credentials import BITKUB_CONFIG
from lumibot.entities import Asset
from lumibot.strategies import Strategy


class BitkubSpotExample(Strategy):
    parameters = {
        "symbol": "BTC",
        "place_test_order": False,  # set True to run the place-then-cancel smoke test (a real, but non-filling, order)
        "test_order_thb": 20,  # approximate order value; Bitkub's minimum is about 10 THB
        "test_discount": 0.5,  # limit price = last price x (1 - discount), i.e. far below the market
    }

    def initialize(self):
        self.sleeptime = "1M"
        self.set_market("24/7")
        self.test_order = None

    def on_trading_iteration(self):
        asset = Asset(self.parameters["symbol"], Asset.AssetType.CRYPTO)
        quote = self.quote_asset  # THB (or USDT) from the broker configuration
        price = self.get_last_price(asset, quote=quote)
        self.log_message(f"{asset.symbol}/{quote.symbol} last price: {price}")
        holdings = [(p.asset.symbol, p.quantity) for p in self.get_positions()]
        self.log_message(f"Cash {self.get_cash()} {quote.symbol}; positions: {holdings}")

        if not self.parameters["place_test_order"] or price is None:
            return

        if self.test_order is None:
            limit_price = round(price * (1 - self.parameters["test_discount"]), 2)
            quantity = self.parameters["test_order_thb"] / limit_price
            self.test_order = self.create_order(
                asset, quantity, "buy", order_type="limit", limit_price=limit_price, quote=quote
            )
            self.submit_order(self.test_order)
            self.log_message(f"Placed test limit buy {quantity:.8f} {asset.symbol} @ {limit_price}", color="yellow")
        else:
            self.log_message(f"Cancelling test order {self.test_order.identifier} (status {self.test_order.status})")
            self.cancel_order(self.test_order)
            self.test_order = None


if __name__ == "__main__":
    broker = Bitkub(BITKUB_CONFIG)
    BitkubSpotExample(broker=broker).run_live()
