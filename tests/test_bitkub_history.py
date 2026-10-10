"""BitkubHistory: chunked, cached download of completed candles. A scripted fake exchange stands in for the network."""

import json

import pandas as pd
import pytest

from lumibot.tools.bitkub_helpers import BitkubAPIError, BitkubTransportError
from lumibot.tools.bitkub_history import BitkubHistory

DAY = 86400
T0 = 1_640_995_200  # 2022-01-01 00:00 UTC (a candle boundary for every resolution)


class FakeExchange:
    """Holds candles by start time and answers /tradingview/history like Bitkub (inclusive from/to)."""

    def __init__(self, candles):
        self.candles = candles  # list of dict(t, o, h, l, c, v); duplicates allowed
        self.calls = []
        self.fail_on_call = None

    def get_tradingview_history(self, symbol, resolution, from_ts, to_ts):
        self.calls.append((symbol, resolution, from_ts, to_ts))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise BitkubTransportError("timeout")
        rows = [c for c in self.candles if from_ts <= c["t"] <= to_ts]
        if not rows:
            return {"c": [], "o": [], "h": [], "l": [], "v": [], "t": [], "s": "no_data"}
        return {
            "t": [r["t"] for r in rows],
            "o": [r["o"] for r in rows],
            "h": [r["h"] for r in rows],
            "l": [r["l"] for r in rows],
            "c": [r["c"] for r in rows],
            "v": [r["v"] for r in rows],
            "s": "ok",
        }


def daily(n, start=T0, volume=1.0):
    return [
        {"t": start + i * DAY, "o": 100.0 + i, "h": 101.0 + i, "l": 99.0 + i, "c": 100.5 + i, "v": volume}
        for i in range(n)
    ]


def history(exchange, tmp_path, now, max_bars=None):
    h = BitkubHistory(client=exchange, cache_dir=tmp_path, clock=lambda: now)
    if max_bars:
        h.MAX_BARS_PER_REQUEST = max_bars
    return h


# ------------------------------------------------------------------ basic download
def test_returns_completed_candles_indexed_by_utc_start(tmp_path):
    exchange = FakeExchange(daily(10))
    h = history(exchange, tmp_path, now=T0 + 20 * DAY)
    frame = h.get("btc_thb", "1D", T0, T0 + 10 * DAY)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"] and len(frame) == 10
    assert str(frame.index.tz) == "UTC" and frame.index[0] == pd.Timestamp(T0, unit="s", tz="UTC")
    assert frame["open"].iloc[3] == 103.0
    assert exchange.calls[0][0] == "BTC_THB"  # symbols are normalised to the exchange's upper case


def test_forming_candle_is_never_returned_or_stored(tmp_path):
    exchange = FakeExchange(daily(10))
    now = T0 + 5 * DAY + 3600  # day 5's candle is only an hour old
    h = history(exchange, tmp_path, now=now)
    frame = h.get("BTC_THB", "1D", T0, T0 + 99 * DAY)
    assert len(frame) == 5 and frame.index[-1] == pd.Timestamp(T0 + 4 * DAY, unit="s", tz="UTC")
    stored = pd.read_parquet(tmp_path / "BTC_THB_1D.parquet")
    assert stored["t"].max() == T0 + 4 * DAY  # the cache holds finished candles only


class SloppyExchange(FakeExchange):
    """Ignores the `to` bound and also returns later candles, including the forming one (a defensive-guard probe)."""

    def get_tradingview_history(self, symbol, resolution, from_ts, to_ts):
        return super().get_tradingview_history(symbol, resolution, from_ts, to_ts + 100 * DAY)


def test_the_forming_candle_stays_out_even_if_the_exchange_returns_it(tmp_path):
    now = T0 + 5 * DAY + 3600
    exchange = SloppyExchange(daily(10))  # candles for days 5..9 exist upstream; only days 0..4 are finished
    frame = history(exchange, tmp_path, now=now).get("BTC_THB", "1D", T0, T0 + 99 * DAY)
    assert frame.index[-1] == pd.Timestamp(T0 + 4 * DAY, unit="s", tz="UTC")
    assert pd.read_parquet(tmp_path / "BTC_THB_1D.parquet")["t"].max() == T0 + 4 * DAY


def test_large_windows_are_split_into_bounded_requests(tmp_path):
    exchange = FakeExchange(daily(25))
    h = history(exchange, tmp_path, now=T0 + 100 * DAY, max_bars=10)
    frame = h.get("BTC_THB", "1D", T0, T0 + 25 * DAY)
    assert len(frame) == 25 and frame.index.is_unique
    assert len(exchange.calls) == 3  # 10 + 10 + 5 bars
    assert [c[3] - c[2] + 1 for c in exchange.calls] == [
        10 * DAY,
        10 * DAY,
        5 * DAY,
    ]  # inclusive `to` stays inside its window


def test_conflicting_duplicates_keep_the_most_complete_candle(tmp_path):
    candles = daily(3)
    candles.insert(2, dict(candles[1], v=0.01, c=99.9))  # a thin stale version of day 1, delivered second
    exchange = FakeExchange(candles)
    frame = history(exchange, tmp_path, now=T0 + 10 * DAY).get("BTC_THB", "1D", T0, T0 + 3 * DAY)
    assert len(frame) == 3 and frame["volume"].iloc[1] == 1.0 and frame["close"].iloc[1] == 101.5


def test_gaps_are_never_filled(tmp_path):
    sparse = [c for c in daily(10) if (c["t"] - T0) // DAY not in (3, 4, 5)]
    frame = history(FakeExchange(sparse), tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    assert len(frame) == 7 and pd.Timedelta(frame.index[3] - frame.index[2]) == pd.Timedelta(days=4)


# ------------------------------------------------------------------ cache
def test_second_request_is_served_from_the_cache_without_network(tmp_path):
    exchange = FakeExchange(daily(10))
    first = history(exchange, tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    calls_after_first = len(exchange.calls)
    again = history(exchange, tmp_path, now=T0 + 20 * DAY).get(
        "BTC_THB", "1D", T0 + 2 * DAY, T0 + 8 * DAY
    )  # fresh object
    assert len(exchange.calls) == calls_after_first
    assert list(again.index) == list(first.index[2:8])


def test_cache_extends_forward_and_refetches_the_newest_candles(tmp_path):
    exchange = FakeExchange(daily(10))
    history(exchange, tmp_path, now=T0 + 6 * DAY).get("BTC_THB", "1D", T0, T0 + 6 * DAY)  # days 0..5 cached
    exchange.calls.clear()
    frame = history(exchange, tmp_path, now=T0 + 10 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    assert len(frame) == 10
    assert len(exchange.calls) == 1
    assert (
        exchange.calls[0][2] == T0 + 6 * DAY - 2 * DAY
    )  # re-requested from two candles before the old edge, not from T0


def test_cache_extends_backward(tmp_path):
    exchange = FakeExchange(daily(10))
    history(exchange, tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0 + 5 * DAY, T0 + 10 * DAY)
    exchange.calls.clear()
    frame = history(exchange, tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    assert len(frame) == 10 and len(exchange.calls) == 1 and exchange.calls[0][3] == T0 + 5 * DAY - 1


def test_a_late_correction_to_a_recent_candle_is_picked_up_on_extension(tmp_path):
    exchange = FakeExchange(daily(10))
    history(exchange, tmp_path, now=T0 + 6 * DAY).get("BTC_THB", "1D", T0, T0 + 6 * DAY)
    exchange.candles[5] = dict(exchange.candles[5], c=777.0, v=50.0)  # day 5 was corrected upstream after it was cached
    frame = history(exchange, tmp_path, now=T0 + 10 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    assert frame["close"].iloc[5] == 777.0


def test_a_failed_download_records_nothing_as_covered(tmp_path):
    exchange = FakeExchange(daily(25))
    exchange.fail_on_call = 2  # the second chunk fails
    h = history(exchange, tmp_path, now=T0 + 100 * DAY, max_bars=10)
    with pytest.raises(BitkubAPIError):
        h.get("BTC_THB", "1D", T0, T0 + 25 * DAY)
    assert not list(tmp_path.glob("*"))  # no data and, crucially, no coverage record that would hide the hole later
    exchange.fail_on_call = None
    assert len(h.get("BTC_THB", "1D", T0, T0 + 25 * DAY)) == 25  # a retry downloads everything again


def test_empty_answers_are_remembered_as_fetched_not_refetched_forever(tmp_path):
    exchange = FakeExchange([])  # a market with no candles in the window (e.g. before it was listed)
    h = history(exchange, tmp_path, now=T0 + 20 * DAY)
    assert h.get("BTC_THB", "1D", T0, T0 + 10 * DAY).empty
    calls = len(exchange.calls)
    assert h.get("BTC_THB", "1D", T0, T0 + 10 * DAY).empty and len(exchange.calls) == calls


def test_a_damaged_cache_is_discarded_and_rebuilt(tmp_path):
    exchange = FakeExchange(daily(10))
    history(exchange, tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    (tmp_path / "BTC_THB_1D.parquet").write_bytes(b"not a parquet file")
    exchange.calls.clear()
    frame = history(exchange, tmp_path, now=T0 + 20 * DAY).get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    assert len(frame) == 10 and exchange.calls  # re-downloaded rather than trusted


def test_resolutions_are_cached_separately(tmp_path):
    exchange = FakeExchange(daily(10))
    h = history(exchange, tmp_path, now=T0 + 20 * DAY)
    h.get("BTC_THB", "1D", T0, T0 + 10 * DAY)
    h.get("BTC_THB", "60", T0, T0 + 2 * DAY)
    assert {p.name for p in tmp_path.glob("*.parquet")} == {"BTC_THB_1D.parquet", "BTC_THB_60.parquet"}
    assert json.loads((tmp_path / "BTC_THB_60.json").read_text())["to"] == T0 + 2 * DAY


def test_invalid_resolution_is_refused_before_any_request(tmp_path):
    exchange = FakeExchange([])
    with pytest.raises(ValueError):
        history(exchange, tmp_path, now=T0 + DAY).get("BTC_THB", "30", T0, T0 + DAY)
    assert exchange.calls == []


def test_empty_and_mixed_windows_do_not_trigger_pandas_concat_warnings(tmp_path):
    """A window with no trades next to windows with candles must merge cleanly (no pandas concat warnings)."""
    import warnings

    candles = [
        c for c in daily(30) if not (10 <= (c["t"] - T0) // DAY < 20)
    ]  # a 10-day hole in the middle of the history
    exchange = FakeExchange(candles)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        frame = history(exchange, tmp_path, now=T0 + 100 * DAY, max_bars=10).get("BTC_THB", "1D", T0, T0 + 30 * DAY)
        assert len(frame) == 20  # the empty middle chunk contributed nothing, and nothing was invented for it
        assert (
            history(FakeExchange([]), tmp_path / "other", now=T0 + 100 * DAY)
            .get("BTC_THB", "1D", T0, T0 + 5 * DAY)
            .empty
        )
