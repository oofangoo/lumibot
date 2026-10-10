"""Unit tests for the Bitkub REST client (no network, no credentials).

The signing vectors were computed independently with ``openssl dgst -sha256 -hmac`` so the tests prove the
implementation matches the algorithm in Bitkub's official docs rather than just agreeing with itself. Fixtures under
tests/fixtures/bitkub are trimmed recordings of the live public API (2026-10-10).
"""

import json
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumibot.tools import bitkub_helpers as bk
from lumibot.tools.bitkub_helpers import (
    BitkubAPIError,
    BitkubClient,
    BitkubRateLimitError,
    BitkubTransportError,
    build_json_body,
    format_decimal,
    format_symbol,
    parse_symbol,
    sign_request,
)

FIXTURES = Path(__file__).parent / "fixtures" / "bitkub"
SECRET = "test_secret"


def fixture_text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=None):
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)


class FakeSession:
    """Records every request; ``script`` is a list of responses/exceptions consumed in order, or a callable."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def request(self, method, url, headers=None, data=None, timeout=None):
        call = SimpleNamespace(method=method, url=url, headers=headers or {}, data=data, timeout=timeout)
        self.calls.append(call)
        item = self.script(call) if callable(self.script) else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def paths(self):
        return [c.url.split("bitkub.com", 1)[1] for c in self.calls]


def make_client(script, **kwargs):
    client = BitkubClient("test_key", SECRET, session=FakeSession(script), **kwargs)
    client._time_synced_at = time.monotonic()  # skip the servertime round trip unless a test wants it
    client._timestamp_ms = lambda: 1699381086593
    return client


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Retry/backoff tests must not actually wait."""
    monkeypatch.setattr(bk.time, "sleep", lambda s: None)


# ------------------------------------------------------------------ signing, formatting
def test_signature_matches_independent_openssl_vectors():
    post_body = '{"sym":"btc_thb","amt":1000,"rat":10,"typ":"limit"}'
    assert sign_request(SECRET, 1699376552354, "POST", "/api/v3/market/place-bid", post_body) == (
        "68c10c4d62d54512f82c2d8077a2cc9f87a7e6e8a6de53b8dd12089bfbfc08ca"
    )
    # The query string is part of the signed path for GET requests.
    assert sign_request(SECRET, 1699381086593, "GET", "/api/v3/market/my-order-history?sym=BTC_THB") == (
        "81f291f11942805a0033863dd0d03d1933b330e7201cfd11c33391548bc01ecf"
    )
    assert sign_request(SECRET, 1699381086593, "get", "/api/v4/wallet/balances") == (
        "c957474c2aed1df6343c963036b4f6769d9d59da0109bfdf8f03bddf7869ab0d"
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        (1000, "1000"),
        (Decimal("1000.00"), "1000"),
        (1000.0, "1000"),
        ("0.10000000", "0.1"),
        (Decimal("1E+3"), "1000"),
        (Decimal("0.00000001"), "0.00000001"),
        (Decimal("0E-8"), "0"),
        (Decimal("2778954.76"), "2778954.76"),
    ],
)
def test_format_decimal_has_no_trailing_zeros_or_exponent(value, expected):
    assert format_decimal(value) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc", None])
def test_format_decimal_rejects_non_numbers(bad):
    with pytest.raises(ValueError):
        format_decimal(bad)


def test_json_body_matches_the_docs_example_and_omits_none():
    body = build_json_body(
        {
            "sym": "btc_thb",
            "amt": Decimal("1000.00"),
            "rat": Decimal("10"),
            "typ": "limit",
            "client_id": None,
            "post_only": None,
        }
    )
    assert body == '{"sym":"btc_thb","amt":1000,"rat":10,"typ":"limit"}'
    with_extras = build_json_body({"sym": "btc_thb", "amt": Decimal("0.1"), "client_id": 'a"b', "post_only": True})
    assert with_extras == '{"sym":"btc_thb","amt":0.1,"client_id":"a\\"b","post_only":true}'


def test_symbol_helpers():
    assert format_symbol("btc", "thb") == "BTC_THB"
    assert parse_symbol("btc_thb") == ("BTC", "THB")
    assert parse_symbol("BTC-THB") == ("BTC", "THB")
    assert parse_symbol("BTC/THB") == ("BTC", "THB")
    for bad in ("BTCTHB", "A_B_C", "_THB", ""):
        with pytest.raises(ValueError):
            parse_symbol(bad)


def test_repr_never_leaks_credentials():
    client = BitkubClient("my_key_value", "my_secret_value")
    assert "my_secret_value" not in repr(client) and "my_key_value" not in repr(client)
    assert "authenticated=True" in repr(client)


# ------------------------------------------------------------------ request construction
def test_public_calls_send_no_credentials_and_exact_query():
    session = FakeSession([FakeResponse(text=fixture_text("ticker_btc_thb.json"))])
    client = BitkubClient(session=session)
    rows = client.get_ticker("btc_thb")
    call = session.calls[0]
    assert call.url == "https://api.bitkub.com/api/v3/market/ticker?sym=btc_thb"
    assert not any(h.startswith("X-BTK") for h in call.headers)
    assert rows[0]["symbol"] == "BTC_THB"


def test_get_signature_covers_path_and_query_string():
    client = make_client([FakeResponse({"error": 0, "result": []})])
    client.get_open_orders("BTC_THB")
    call = client._session.calls[0]
    assert call.url.endswith("/api/v3/market/my-open-orders?sym=btc_thb")
    assert call.headers["X-BTK-APIKEY"] == "test_key"
    assert call.headers["X-BTK-TIMESTAMP"] == "1699381086593"
    expected = sign_request(SECRET, 1699381086593, "GET", "/api/v3/market/my-open-orders?sym=btc_thb")
    assert call.headers["X-BTK-SIGN"] == expected
    assert call.data is None


def test_post_signature_covers_the_exact_body_that_is_sent():
    client = make_client([FakeResponse({"error": 0, "result": {"id": "1"}})])
    client.place_bid("BTC_THB", amount=Decimal("1000.00"), rate=Decimal("10"), order_type="limit", client_id="abc")
    call = client._session.calls[0]
    sent = call.data.decode("utf-8")
    assert sent == '{"sym":"btc_thb","amt":1000,"rat":10,"typ":"limit","client_id":"abc"}'
    assert call.headers["X-BTK-SIGN"] == sign_request(SECRET, 1699381086593, "POST", "/api/v3/market/place-bid", sent)
    assert call.headers["Content-Type"] == "application/json"


def test_market_order_forces_rate_zero_and_validates_inputs():
    client = make_client([FakeResponse({"error": 0, "result": {"id": "2"}}), FakeResponse({"error": 0, "result": {}})])
    client.place_ask("BTC_THB", amount="0.001", rate=Decimal("999"), order_type="market")
    assert client._session.calls[0].data.decode() == '{"sym":"btc_thb","amt":0.001,"rat":0,"typ":"market"}'
    for kwargs in (
        {"amount": 0, "rate": 10},  # zero amount
        {"amount": 1, "rate": 0},  # limit order without a rate
        {"amount": 1, "rate": 10, "order_type": "stop"},  # unsupported type
    ):
        with pytest.raises(ValueError):
            client.place_bid("BTC_THB", **kwargs)
    assert len(client._session.calls) == 1  # validation failures never reach the network


def test_cancel_order_body_and_side_validation():
    client = make_client([FakeResponse({"error": 0})])
    client.cancel_order("BTC_THB", "123456", "BUY")
    assert client._session.calls[0].data.decode() == '{"sym":"btc_thb","id":"123456","sd":"buy"}'
    with pytest.raises(ValueError):
        client.cancel_order("BTC_THB", "1", "hold")


def test_missing_credentials_fail_before_any_request():
    session = FakeSession([])
    client = BitkubClient(session=session)
    for call in (
        client.get_balances,
        lambda: client.get_open_orders("btc_thb"),
        lambda: client.place_bid("btc_thb", 10, 1),
    ):
        with pytest.raises(BitkubAPIError):
            call()
    assert session.calls == []


def test_tradingview_history_params_validation_and_no_data_passthrough():
    client = make_client(
        [
            FakeResponse(text=fixture_text("tradingview_60.json")),
            FakeResponse({"c": [], "o": [], "h": [], "l": [], "v": [], "t": [], "s": "no_data"}),
        ]
    )
    data = client.get_tradingview_history("BTC_THB", "60", 1791590400, 1791612000)
    assert client._session.calls[0].url.endswith(
        "/tradingview/history?symbol=BTC_THB&resolution=60&from=1791590400&to=1791612000"
    )
    assert data["s"] == "ok" and len(data["t"]) == len(data["c"]) == len(data["o"]) == 7
    empty = client.get_tradingview_history("BTC_THB", "1D", 1, 2)
    assert empty["s"] == "no_data" and empty["t"] == []  # absence is reported, never filled in
    with pytest.raises(ValueError):
        client.get_tradingview_history("BTC_THB", "30", 1, 2)


# ------------------------------------------------------------------ response / error handling
def test_decimal_precision_is_preserved():
    client = make_client([FakeResponse(text=fixture_text("depth_btc_thb.json"))])
    depth = client.get_depth("BTC_THB", 2)
    assert depth["asks"][0][0] == Decimal("2778954.76") and depth["bids"][0][1] == Decimal("0.03896274")
    assert client._session.calls[0].url.endswith("/api/v3/market/depth?sym=BTC_THB&lmt=2")


def test_bare_payloads_pass_through():
    client = make_client([FakeResponse(text="1791615363890"), FakeResponse(text=fixture_text("ticker_btc_thb.json"))])
    assert client.get_server_time() == 1791615363890
    assert isinstance(client.get_ticker(), list)


def test_v3_error_inside_http_200_raises_with_code_and_message():
    client = make_client([FakeResponse({"error": 11})])
    with pytest.raises(BitkubAPIError) as caught:
        client.get_ticker("thb_btc")
    assert caught.value.code == 11 and "Invalid symbol" in str(caught.value)


def test_http_400_body_is_parsed_not_just_the_status():
    client = make_client([FakeResponse({"error": 2, "result": None}, status_code=400)])
    with pytest.raises(BitkubAPIError) as caught:
        client.get_open_orders("btc_thb")
    assert caught.value.code == 2 and caught.value.http_status == 400


def test_v4_error_envelope():
    client = make_client([FakeResponse({"code": "V1007-CW", "message": "bad input"})])
    with pytest.raises(BitkubAPIError) as caught:
        client.get_balances()
    assert caught.value.code == "V1007-CW"


def test_v4_success_envelope_returns_data():
    rows = [{"currency": "THB", "available": "10", "reserved": "0", "total": "10"}]
    client = make_client([FakeResponse({"code": "0", "message": "success", "data": rows})])
    assert client.get_balances() == rows


def test_non_json_response_is_an_api_error():
    client = make_client([FakeResponse(text="<html>gateway</html>")])
    with pytest.raises(BitkubAPIError):
        client.get_symbols()


def test_http_429_blocks_following_requests_for_the_documented_window():
    client = make_client([FakeResponse(text="", status_code=429), FakeResponse({"error": 0, "result": []})])
    with pytest.raises(BitkubRateLimitError):
        client.get_symbols()
    with pytest.raises(BitkubRateLimitError):  # refused locally, nothing is sent while blocked
        client.get_symbols()
    assert len(client._session.calls) == 1


# ------------------------------------------------------------------ retry policy (money safety)
def test_get_is_retried_on_5xx_and_transport_errors():
    client = make_client(
        [ConnectionError("reset"), FakeResponse(text="", status_code=502), FakeResponse({"error": 0, "result": []})]
    )
    assert client.get_symbols() == []
    assert len(client._session.calls) == 3


def test_get_gives_up_after_max_retries():
    client = make_client([FakeResponse(text="", status_code=503)] * 3, max_retries=2)
    with pytest.raises(BitkubTransportError):
        client.get_symbols()
    assert len(client._session.calls) == 3


@pytest.mark.parametrize("failure", [FakeResponse(text="", status_code=502), TimeoutError("timed out")])
def test_post_is_never_retried_because_the_order_state_is_unknown(failure):
    client = make_client([failure, FakeResponse({"error": 0, "result": {"id": "should-not-happen"}})])
    with pytest.raises(BitkubTransportError) as caught:
        client.place_bid("BTC_THB", amount=100, rate=0, order_type="market")
    assert "may not have been processed" in str(caught.value) or caught.value.http_status == 502
    assert len(client._session.calls) == 1


def test_invalid_timestamp_resyncs_the_clock_once_and_retries_even_for_post():
    def script(call):
        if call.url.endswith("/api/v3/servertime"):
            return FakeResponse(text="1791615363890")
        responses = script.queue
        return responses.pop(0)

    script.queue = [FakeResponse({"error": 8}), FakeResponse({"error": 0, "result": {"id": "9"}})]
    client = BitkubClient("test_key", SECRET, session=FakeSession(script))
    client._time_synced_at = time.monotonic()
    result = client.place_bid("BTC_THB", amount=100, rate=0, order_type="market")
    assert result == {"id": "9"}
    assert client._session.paths.count("/api/v3/market/place-bid") == 2
    assert client._session.paths.count("/api/v3/servertime") == 1


def test_persistent_invalid_timestamp_is_not_retried_forever():
    def script(call):
        return (
            FakeResponse(text="1791615363890")
            if call.url.endswith("/api/v3/servertime")
            else FakeResponse({"error": 8})
        )

    client = BitkubClient("test_key", SECRET, session=FakeSession(script))
    client._time_synced_at = time.monotonic()
    with pytest.raises(BitkubAPIError) as caught:
        client.get_balances()
    assert caught.value.code == 8
    assert client._session.paths.count("/api/v4/wallet/balances") == 2  # original + exactly one retry


def test_time_offset_tracks_the_server_clock(monkeypatch):
    monkeypatch.setattr(bk.time, "time", lambda: 1000.0)  # local clock: 1,000,000 ms
    client = BitkubClient("k", "s", session=FakeSession([FakeResponse(text="1000500")]))
    assert client.sync_time() == 1000500
    assert client._time_offset_ms == 500
    assert client._timestamp_ms() == 1_000_500


# ------------------------------------------------------------------ rate limiter
def test_rate_limits_are_looked_up_per_endpoint_family():
    limiter = bk._RateLimiter()
    assert limiter._limit_for("/api/v3/market/depth") == 10
    assert limiter._limit_for("/api/v3/user/limits") == 20
    assert limiter._limit_for("/api/v3/market/ticker") == bk._DEFAULT_RATE_LIMIT_PER_SECOND


def test_rate_limiter_delays_instead_of_exceeding_the_window(monkeypatch):
    clock = {"now": 100.0}
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(bk.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(bk.time, "sleep", fake_sleep)
    limiter = bk._RateLimiter()
    for _ in range(8):  # depth is 10/s; the limiter keeps an 80% margin => 8 calls per second
        limiter.wait("/api/v3/market/depth")
    assert slept == []
    limiter.wait("/api/v3/market/depth")  # the 9th call inside the same second must wait for the window to roll
    assert slept and slept[0] > 0
