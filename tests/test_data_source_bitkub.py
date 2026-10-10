"""Unit tests for BitkubData (no network, no credentials). Fixtures are trimmed recordings of the live public API."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lumibot.data_sources.bitkub_data import BitkubData
from lumibot.entities import Asset
from lumibot.tools.bitkub_helpers import BitkubAPIError

FIXTURES = Path(__file__).parent / "fixtures" / "bitkub"
BTC = Asset("BTC", Asset.AssetType.CRYPTO)
NO_DATA = {"c": [], "o": [], "h": [], "l": [], "v": [], "t": [], "s": "no_data"}


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClient:
    """Stands in for BitkubClient. ``history`` is a list of payloads (or exceptions) consumed per request."""

    def __init__(self, ticker=None, history=None, depth=None):
        self.ticker = ticker
        self.history = list(history or [])
        self.depth = depth
        self.history_calls = []
        self.ticker_calls = []

    def get_ticker(self, symbol=None):
        self.ticker_calls.append(symbol)
        if isinstance(self.ticker, Exception):
            raise self.ticker
        return self.ticker

    def get_tradingview_history(self, symbol, resolution, from_ts, to_ts):
        self.history_calls.append((symbol, resolution, from_ts, to_ts))
        item = self.history.pop(0) if len(self.history) > 1 else self.history[0]
        if isinstance(item, Exception):
            raise item
        return item

    def get_depth(self, symbol, limit):
        return self.depth


def make(client, now_ts):
    source = BitkubData(client=client)
    source.get_datetime = lambda adjust_for_delay=False: pd.Timestamp(now_ts, unit="s", tz="UTC")
    return source


# ------------------------------------------------------------------ construction, symbols, timesteps
def test_constructs_without_credentials_or_network():
    source = BitkubData()
    assert source.name == "bitkub" and source.SOURCE == "BITKUB"
    assert not source.client.has_credentials


def test_credentials_in_config_are_passed_to_the_shared_client():
    source = BitkubData({"API_KEY": "k", "API_SECRET": "s"})
    assert source.client.has_credentials


def test_symbol_resolution_defaults_to_thb_and_accepts_other_quotes():
    source = BitkubData(client=FakeClient())
    asset, quote = source._sanitize_base_and_quote_asset(BTC)
    assert source.symbol_for(asset, quote) == "BTC_THB"
    asset, quote = source._sanitize_base_and_quote_asset("eth", "USDT")
    assert source.symbol_for(asset, quote) == "ETH_USDT"
    asset, quote = source._sanitize_base_and_quote_asset((BTC, Asset("THB", Asset.AssetType.FOREX)))
    assert source.symbol_for(asset, quote) == "BTC_THB"


def test_non_crypto_assets_are_rejected():
    with pytest.raises(ValueError):
        BitkubData(client=FakeClient())._sanitize_base_and_quote_asset(Asset("SPY", Asset.AssetType.STOCK))


@pytest.mark.parametrize(
    "given, canonical, resolution",
    [
        ("minute", "minute", "1"),
        ("1m", "minute", "1"),
        ("5 minutes", "5 minutes", "5"),
        ("15m", "15 minutes", "15"),
        ("hour", "hour", "60"),
        ("1h", "hour", "60"),
        ("4 hours", "4 hours", "240"),
        ("day", "day", "1D"),
        ("1D", "day", "1D"),
        ("1 day", "day", "1D"),
    ],
)
def test_timestep_mapping(given, canonical, resolution):
    source = BitkubData(client=FakeClient())
    assert source.get_timestep_from_string(given) == canonical
    assert source._parse_source_timestep(given) == resolution
    assert source.supports_native_timestep(given)


@pytest.mark.parametrize("unsupported", ["30 minutes", "2 hours", "week", "3m"])
def test_unsupported_timesteps_raise_instead_of_substituting(unsupported):
    source = BitkubData(client=FakeClient(history=[NO_DATA]))
    assert not source.supports_native_timestep(unsupported)
    with pytest.raises(ValueError):
        source.get_historical_prices(BTC, 10, unsupported)
    assert source.client.history_calls == []


# ------------------------------------------------------------------ last price / quote
def test_last_price_and_quote_come_from_the_ticker():
    client = FakeClient(ticker=fixture("ticker_btc_thb.json"))
    source = BitkubData(client=client)
    assert source.get_last_price(BTC) == pytest.approx(2778954.76)
    assert client.ticker_calls == ["BTC_THB"]
    quote = source.get_quote(BTC)
    assert quote.price == pytest.approx(2778954.76)
    assert quote.bid == pytest.approx(2778954.75) and quote.ask == pytest.approx(2778954.76)
    assert quote.percent_change == pytest.approx(0.28)
    assert quote.asset == BTC


@pytest.mark.parametrize("ticker", [[], BitkubAPIError("bad", code=11), [{"symbol": "ETH_THB", "last": "1"}], "oops"])
def test_missing_ticker_gives_none_and_an_empty_quote(ticker):
    source = BitkubData(client=FakeClient(ticker=ticker))
    assert source.get_last_price(BTC) is None
    quote = source.get_quote(BTC)
    assert quote.price is None and quote.bid is None and quote.ask is None


def test_zero_price_is_not_a_price():
    source = BitkubData(
        client=FakeClient(ticker=[{"symbol": "BTC_THB", "last": "0", "highest_bid": "0", "lowest_ask": "0"}])
    )
    assert source.get_last_price(BTC) is None


def test_get_last_prices_uses_one_request_for_the_batch():
    rows = [{"symbol": "BTC_THB", "last": "100.5"}, {"symbol": "ETH_THB", "last": "5"}]
    client = FakeClient(ticker=rows)
    prices = BitkubData(client=client).get_last_prices(
        [BTC, Asset("ETH", Asset.AssetType.CRYPTO), Asset("NOPE", Asset.AssetType.CRYPTO)]
    )
    assert client.ticker_calls == [None]
    assert prices[BTC] == 100.5 and prices[Asset("ETH", Asset.AssetType.CRYPTO)] == 5.0
    assert prices[Asset("NOPE", Asset.AssetType.CRYPTO)] is None


def test_orderbook_snapshot_is_best_first_floats():
    depth = fixture("depth_btc_thb.json")["result"]
    book = BitkubData(client=FakeClient(depth=depth)).get_orderbook(BTC, depth=2)
    assert book["asks"][0] == (2778954.76, 0.00181478) and book["bids"][0][0] == 2778954.75
    assert book["asks"][0][0] > book["bids"][0][0]


def test_get_chains_is_empty():
    assert BitkubData(client=FakeClient()).get_chains(BTC)["Chains"] == {}


# ------------------------------------------------------------------ historical prices
HOURLY = fixture("tradingview_60.json")  # 7 hourly candles, 1791590400 .. 1791612000
LAST_HOURLY_T = 1791612000


def test_history_returns_completed_bars_only():
    source = make(FakeClient(history=[HOURLY]), now_ts=LAST_HOURLY_T + 1800)  # the last candle is still forming
    bars = source.get_historical_prices(BTC, 6, "hour")
    assert len(bars.df) == 6 and bars.source == "BITKUB"
    assert bars.df.index[-1] == pd.Timestamp(LAST_HOURLY_T - 3600, unit="s", tz="UTC")
    assert list(bars.df.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert bars.df["close"].iloc[0] == pytest.approx(2771416.98)


def test_history_includes_a_candle_once_its_hour_has_elapsed():
    source = make(FakeClient(history=[HOURLY]), now_ts=LAST_HOURLY_T + 3600)
    assert len(source.get_historical_prices(BTC, 7, "hour").df) == 7


def test_history_requests_the_right_symbol_resolution_and_window():
    client = FakeClient(history=[HOURLY])
    make(client, now_ts=LAST_HOURLY_T + 3600).get_historical_prices(
        Asset("BTC", Asset.AssetType.CRYPTO), 5, "hour", quote="USDT"
    )
    symbol, resolution, from_ts, to_ts = client.history_calls[0]
    assert (symbol, resolution) == ("BTC_USDT", "60")
    assert to_ts == LAST_HOURLY_T + 3600 and from_ts == to_ts - (5 + 2) * 3600


def test_history_timeshift_in_bars_moves_the_window_back():
    client = FakeClient(history=[HOURLY])
    make(client, now_ts=LAST_HOURLY_T + 3600).get_historical_prices(BTC, 3, "hour", timeshift=2)
    assert client.history_calls[0][3] == LAST_HOURLY_T + 3600 - 2 * 3600


def test_history_tail_is_limited_to_the_requested_length():
    bars = make(FakeClient(history=[HOURLY]), now_ts=LAST_HOURLY_T + 3600).get_historical_prices(BTC, 3, "hour")
    assert len(bars.df) == 3
    assert bars.df.index[-1] == pd.Timestamp(LAST_HOURLY_T, unit="s", tz="UTC")


@pytest.mark.parametrize("complete_row_first", [False, True], ids=["complete-row-last", "complete-row-first"])
def test_conflicting_duplicate_candles_keep_the_most_complete_version(complete_row_first):
    """The exchange returns two rows for 2025-04-25 (UTC): v=48.9/close 3,137,364 and v=101.6/close 3,195,139.

    The larger version is the complete day: its close matches the next candle's open (3,195,842), the other does not.
    The live API returns the two rows in either order (observed both), so the choice must not depend on row order.
    """
    payload = fixture("tradingview_1d_duplicate_candle.json")
    assert len(payload["t"]) == 5 and payload["t"].count(1745539200) == 2  # the fixture really contains the conflict
    i, j = [k for k, t in enumerate(payload["t"]) if t == 1745539200]
    if (payload["v"][i] > payload["v"][j]) != complete_row_first:  # put the complete row where this case needs it
        for key in "tohlcv":
            payload[key][i], payload[key][j] = payload[key][j], payload[key][i]
    assert (payload["v"][i] > payload["v"][j]) == complete_row_first
    source = make(FakeClient(history=[payload]), now_ts=1745712000 + 86400)
    bars = source.get_historical_prices(BTC, 4, "day")
    assert len(bars.df) == 4 and bars.df.index.is_unique
    day = bars.df.loc[pd.Timestamp(1745539200, unit="s", tz="UTC")]
    assert day["volume"] == pytest.approx(101.56875227, rel=1e-6) and day["close"] == 3195139
    next_open = bars.df.loc[pd.Timestamp(1745625600, unit="s", tz="UTC"), "open"]
    assert abs(day["close"] - next_open) / next_open < 0.001


def test_exact_duplicate_rows_collapse_to_one_candle():
    payload = {
        "t": [100, 100, 160],
        "o": [1, 1, 2],
        "h": [2, 2, 3],
        "l": [1, 1, 2],
        "c": [2, 2, 3],
        "v": [5, 5, 4],
        "s": "ok",
    }
    bars = make(FakeClient(history=[payload]), now_ts=10_000).get_historical_prices(BTC, 2, "minute")
    assert list(bars.df["volume"]) == [5.0, 4.0]


def test_sparse_minutes_are_never_filled_in():
    payload = {
        "t": [60, 120, 300],
        "o": [1, 2, 3],
        "h": [1, 2, 3],
        "l": [1, 2, 3],
        "c": [1, 2, 3],
        "v": [1, 1, 1],
        "s": "ok",
    }
    bars = make(FakeClient(history=[payload]), now_ts=10_000).get_historical_prices(BTC, 3, "minute")
    assert len(bars.df) == 3  # exactly the candles that traded, with the 120 -> 300 gap left as a gap
    assert (bars.df.index[2] - bars.df.index[1]) == pd.Timedelta(seconds=180)


def test_sparse_market_widens_the_window_until_enough_candles_exist():
    thin = {"t": [60, 120], "o": [1, 1], "h": [1, 1], "l": [1, 1], "c": [1, 1], "v": [1, 1], "s": "ok"}
    rich = {
        "t": [60 * i for i in range(1, 11)],
        "o": [1] * 10,
        "h": [1] * 10,
        "l": [1] * 10,
        "c": [1] * 10,
        "v": [1] * 10,
        "s": "ok",
    }
    client = FakeClient(history=[thin, rich])
    bars = make(client, now_ts=10_000).get_historical_prices(BTC, 8, "minute")
    assert len(client.history_calls) == 2 and len(bars.df) == 8
    first_window = client.history_calls[0][3] - client.history_calls[0][2]
    second_window = client.history_calls[1][3] - client.history_calls[1][2]
    assert second_window == 4 * first_window


def test_short_history_returns_what_exists_rather_than_padding(caplog):
    short = {"t": [60, 120], "o": [1, 2], "h": [1, 2], "l": [1, 2], "c": [1, 2], "v": [1, 1], "s": "ok"}
    client = FakeClient(history=[short])
    bars = make(client, now_ts=10_000).get_historical_prices(BTC, 50, "minute")
    assert len(bars.df) == 2  # never padded up to 50
    assert len(client.history_calls) == len(BitkubData.LOOKBACK_MULTIPLIERS)  # tried every window before giving up


def test_no_data_returns_none_not_an_invented_bar():
    assert make(FakeClient(history=[NO_DATA]), now_ts=10_000).get_historical_prices(BTC, 5, "day") is None


def test_api_errors_return_none():
    source = make(FakeClient(history=[BitkubAPIError("Symbol not found", code="7002")]), now_ts=10_000)
    assert source.get_historical_prices(Asset("NOPE", Asset.AssetType.CRYPTO), 5, "day") is None


def test_corrupt_rows_are_dropped_not_repaired():
    payload = {
        "t": [60, 120, 180],
        "o": [1, 0, 3],
        "h": [1, 0, 3],
        "l": [1, 0, 3],
        "c": [1, 0, 3],
        "v": [1, 1, 1],
        "s": "ok",
    }
    bars = make(FakeClient(history=[payload]), now_ts=10_000).get_historical_prices(BTC, 3, "minute")
    assert len(bars.df) == 2 and (bars.df[["open", "high", "low", "close"]] > 0).all().all()
