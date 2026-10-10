"""Live smoke tests against Bitkub's PUBLIC endpoints (no API key, no account, nothing can be traded).

Deselected by default (``-m "not apitest"``). Run with ``pytest tests/test_bitkub_live_public.py -m apitest``.
They assert invariants that must hold on real data rather than exact prices.
"""

import pandas as pd
import pytest

from lumibot.data_sources.bitkub_data import BitkubData
from lumibot.entities import Asset
from lumibot.tools.bitkub_helpers import BitkubClient

pytestmark = [pytest.mark.apitest, pytest.mark.public_http]  # public_http: needs no provider credentials
BTC = Asset("BTC", Asset.AssetType.CRYPTO)


@pytest.fixture(scope="module")
def data():
    return BitkubData()


def test_server_time_and_status_are_sane():
    client = BitkubClient()
    server_ms = client.get_server_time()
    assert (
        abs(server_ms - pd.Timestamp.now(tz="UTC").timestamp() * 1000) < 5 * 60 * 1000
    )  # within five minutes of local time
    assert {row["name"] for row in client.get_status()} >= {"Non-secure endpoints", "Secure endpoints"}


def test_symbols_describe_tradable_markets(data):
    rows = {row["symbol"]: row for row in data.get_symbols()}
    btc = rows["BTC_THB"]
    assert (
        btc["status"] == "active"
        and btc["source"] == "exchange"
        and int(btc["base_asset_scale"]) >= 1
        and float(btc["min_quote_size"]) > 0
    )


def test_last_price_and_quote_are_consistent(data):
    price, quote = data.get_last_price(BTC), data.get_quote(BTC)
    assert price and price > 0 and quote.bid and quote.ask and quote.bid <= quote.ask
    assert quote.bid * 0.9 <= price <= quote.ask * 1.1


def test_orderbook_is_ordered_and_not_crossed(data):
    book = data.get_orderbook(BTC, depth=5)
    assert book["bids"][0][0] < book["asks"][0][0]
    assert [p for p, _ in book["bids"]] == sorted((p for p, _ in book["bids"]), reverse=True)


@pytest.mark.parametrize("timestep, length", [("day", 30), ("hour", 48), ("15 minutes", 40)])
def test_history_is_clean_ordered_completed_and_not_padded(data, timestep, length):
    bars = data.get_historical_prices(BTC, length, timestep)
    frame = bars.df
    assert 0 < len(frame) <= length
    assert frame.index.is_monotonic_increasing and frame.index.is_unique  # duplicates resolved
    assert (frame[["open", "high", "low", "close"]] > 0).all().all()
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1) + 1e-6).all()
    interval = {"day": 86400, "hour": 3600, "15 minutes": 900}[timestep]
    assert frame.index[-1] + pd.Timedelta(seconds=interval) <= pd.Timestamp.now(tz="UTC")  # forming candle excluded


def test_unknown_market_returns_none_not_an_exception(data):
    assert data.get_last_price(Asset("NOSUCHCOIN", Asset.AssetType.CRYPTO)) is None
    assert data.get_historical_prices(Asset("NOSUCHCOIN", Asset.AssetType.CRYPTO), 5, "day") is None


# ------------------------------------------------------------------ real history, cache and a real backtest
class _CountingClient(BitkubClient):
    """A real public client that counts how many history requests actually reach the network."""

    history_calls = 0

    def get_tradingview_history(self, *args, **kwargs):
        type(self).history_calls += 1
        return super().get_tradingview_history(*args, **kwargs)


def test_history_download_is_clean_cached_and_extendable_on_real_data(tmp_path):
    from lumibot.tools.bitkub_history import BitkubHistory

    _CountingClient.history_calls = 0
    now = int(pd.Timestamp.now(tz="UTC").timestamp())
    start, end = now - 400 * 86400, now
    history = BitkubHistory(client=_CountingClient(), cache_dir=tmp_path)
    first = history.get("BTC_THB", "1D", start, end)
    assert 380 <= len(first) <= 401 and first.index.is_unique and first.index.is_monotonic_increasing
    assert (first[["open", "high", "low", "close"]] > 0).all().all()
    assert first.index[-1] + pd.Timedelta(days=1) <= pd.Timestamp.now(tz="UTC")  # nothing still forming
    downloads = _CountingClient.history_calls
    assert downloads >= 1

    again = BitkubHistory(client=_CountingClient(), cache_dir=tmp_path).get(
        "BTC_THB", "1D", start + 30 * 86400, end - 30 * 86400
    )
    assert _CountingClient.history_calls == downloads  # a fresh object read the cache; no network
    assert again.index.equals(first.index[(first.index >= again.index[0]) & (first.index <= again.index[-1])])


def test_a_real_backtest_runs_end_to_end_on_real_bitkub_candles(tmp_path):
    from datetime import datetime

    from lumibot.backtesting import BitkubBacktesting
    from lumibot.entities import TradingFee
    from lumibot.strategies import Strategy

    thb = Asset("THB", Asset.AssetType.FOREX)

    class Trend(Strategy):
        def initialize(self):
            self.sleeptime = "1D"
            self.set_market("24/7")
            self.fills = []

        def on_trading_iteration(self):
            bars = self.get_historical_prices(BTC, 20, "day", quote=thb)
            if bars is None or len(bars.df) < 20:
                return
            closes = bars.df["close"]
            held = self.get_position(BTC)
            if closes.iloc[-1] > closes.mean() and held is None:
                self.submit_order(
                    self.create_order(BTC, self.get_cash() * 0.99 / float(closes.iloc[-1]), "buy", quote=thb)
                )
            elif closes.iloc[-1] <= closes.mean() and held is not None:
                self.submit_order(self.create_order(BTC, held.quantity, "sell", quote=thb))

        def on_filled_order(self, position, order, price, quantity, multiplier):
            self.fills.append((self.get_datetime(), str(order.side), float(price)))

    fees = [TradingFee(percent_fee=0.0025)]
    _, strategy = Trend.run_backtest(
        BitkubBacktesting,
        backtesting_start=datetime(2023, 1, 1),
        backtesting_end=datetime(2023, 7, 1),
        cache_dir=tmp_path,
        quote_asset=thb,
        benchmark_asset=(BTC, thb),
        budget=100_000,
        buy_trading_fees=fees,
        sell_trading_fees=fees,
        analyze_backtest=False,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        save_logfile=False,
        show_progress_bar=False,
        quiet_logs=True,
        show_indicators=False,
        save_stats_file=False,
    )
    assert len(strategy.fills) >= 2  # a 20-day trend rule on half a year of BTC trades more than once
    assert 20_000 < strategy.portfolio_value < 500_000  # finite and plausible, not a data accident
    assert all(side in ("buy", "sell") and price > 0 for _, side, price in strategy.fills)
