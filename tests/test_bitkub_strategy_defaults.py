"""A strategy on Bitkub must default to the broker's quote currency (THB/USDT), not USD (Bitkub has no USD markets)."""

from lumibot.brokers.bitkub import Bitkub
from lumibot.entities import Asset
from lumibot.strategies import Strategy

CONFIG = {"API_KEY": "test_key", "API_SECRET": "test_secret"}


class _Idle(Strategy):
    def on_trading_iteration(self):
        pass


class _ReadOnlyClient:
    """A live strategy reads account state when it starts, but constructing one must never place or cancel an order."""

    def get_balances(self):
        return [{"currency": "THB", "available": "100", "reserved": "0", "total": "100"}]

    def get_ticker(self, symbol=None):
        return []

    def get_symbols(self):
        return []

    def get_open_orders(self, symbol):
        return []

    def __getattr__(self, name):  # place_bid, place_ask, cancel_order, ...
        raise AssertionError(f"constructing a strategy must not call Bitkub's {name}")


def _strategy(config=CONFIG, **kwargs):
    broker = Bitkub(config, client=_ReadOnlyClient(), connect_stream=False)
    return _Idle(broker=broker, **kwargs), broker


def test_default_quote_asset_is_thb_not_usd():
    strategy, broker = _strategy()
    assert strategy.quote_asset == Asset("THB", Asset.AssetType.FOREX)
    assert strategy.quote_asset in broker.quote_assets


def test_default_quote_asset_follows_the_broker_configuration():
    strategy, _ = _strategy({**CONFIG, "QUOTE_ASSET": "USDT"})
    assert strategy.quote_asset == Asset("USDT", Asset.AssetType.CRYPTO)


def test_an_explicit_quote_asset_wins():
    explicit = Asset("USDT", Asset.AssetType.CRYPTO)
    strategy, _ = _strategy(quote_asset=explicit)
    assert strategy.quote_asset == explicit
