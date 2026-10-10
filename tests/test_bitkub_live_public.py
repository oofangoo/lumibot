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
