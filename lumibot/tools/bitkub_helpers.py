"""Bitkub REST client (public market data, v3 trading, v4 wallet).

Bitkub is a Thai spot crypto exchange (THB and USDT quote markets). This module owns *all* provider-specific request
construction, response parsing and error normalisation; the broker and data-source adapters stay provider-generic.

API facts this client encodes (verified against the official docs and the live public endpoints, 2026-10-10):

* Private requests carry ``X-BTK-APIKEY``, ``X-BTK-TIMESTAMP`` (milliseconds) and ``X-BTK-SIGN``, where the signature is
  ``hex(HMAC_SHA256(secret, timestamp + METHOD + path [+ "?" + query] + body))``. For GET requests the *query string is
  part of the signed path*; for POST requests the signature covers the exact JSON body string that is sent.
* Numbers in order bodies must have **no trailing zeros** (``1000`` not ``1000.00``), so the JSON body is built by hand
  from :class:`~decimal.Decimal` values instead of ``json.dumps`` (which would emit ``1000.0``).
* v3 endpoints answer ``{"error": <int>, "result": ...}`` (the ticker returns a bare list and servertime a bare
  integer);
  v4 endpoints answer ``{"code": "<str>", "message": ..., "data": ...}``. Failures can arrive with HTTP 200 (for example
  ``{"error": 11}`` for an invalid symbol) **or** HTTP 400, so the body is always parsed, never the status alone.
* Symbols are ``BASE_QUOTE`` (``BTC_THB``), case-insensitive; the legacy ``THB_BTC`` order is rejected (error 11).
* Trade timestamps from ``/api/v3/market/trades`` are in *seconds* although the docs example shows milliseconds.
* Exceeding a rate limit blocks the caller for 30 seconds (HTTP 429).

Safety properties: POST requests (place/cancel) are never blindly retried, because an order may already exist after a
timeout; the secret is never logged or included in ``repr``; and no request is signed with a key that was not supplied.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from collections import deque
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from lumibot._lazy_imports import LazyLogger

logger = LazyLogger(__name__)

BASE_URL = "https://api.bitkub.com"

# Documented v3 error codes (rest-v3.md "Error Codes").
ERROR_MESSAGES = {
    1: "Invalid JSON payload",
    2: "Missing X-BTK-APIKEY",
    3: "Invalid API key",
    4: "API pending for activation",
    5: "IP not allowed",
    6: "Missing / invalid signature",
    7: "Missing timestamp",
    8: "Invalid timestamp",
    9: "Invalid user",
    10: "Invalid parameter",
    11: "Invalid symbol",
    12: "Invalid amount",
    13: "Invalid rate",
    14: "Improper rate",
    15: "Amount too low",
    16: "Failed to get balance",
    17: "Wallet is empty",
    18: "Insufficient balance",
    19: "Failed to insert order into db",
    20: "Failed to deduct balance",
    21: "Invalid order for cancellation (unable to find order id or symbol)",
    22: "Invalid side",
    23: "Failed to update order status",
    24: "Invalid order for lookup / invalid KYC level",
    25: "KYC level 1 is required to proceed",
    30: "Limit exceeds",
    52: "Invalid permission",
    55: "Cancel only mode",
    56: "User has been suspended from purchasing",
    57: "User has been suspended from selling",
    58: "User bank is not verified",
    61: "This endpoint does not support broker coins (symbol source = broker)",
    90: "Server error (contact Bitkub support)",
}
ERROR_INVALID_TIMESTAMP = 8

# Documented per-endpoint limits (requests per second); the limiter keeps a safety margin below them.
_RATE_LIMITS_PER_SECOND = (
    ("/api/v3/market/depth", 10),
    ("/api/v3/user/", 20),
    ("/api/v4/", 20),
    ("/api/v3/servertime", 100),
)
_DEFAULT_RATE_LIMIT_PER_SECOND = 50
_RATE_LIMIT_BLOCK_SECONDS = 30.0  # documented block after exceeding a limit

RESOLUTIONS = ("1", "5", "15", "60", "240", "1D")


class BitkubAPIError(Exception):
    """The exchange rejected a request (or answered with something unusable)."""

    def __init__(self, message, code=None, endpoint=None, http_status=None):
        super().__init__(message)
        self.code = code
        self.endpoint = endpoint
        self.http_status = http_status


class BitkubRateLimitError(BitkubAPIError):
    """HTTP 429, or a request refused locally while the documented 30 second block is in force."""


class BitkubTransportError(BitkubAPIError):
    """The request may not have reached the exchange or its answer was lost (timeout, connection reset, HTTP 5xx).

    For POST requests this means the order/cancel state is UNKNOWN and must be reconciled before it is retried.
    """


def format_decimal(value) -> str:
    """Render a number the way Bitkub requires: plain notation, no exponent, no trailing zeros (``1000``, ``0.1``)."""
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite number: {value!r}")
    text = format(number.normalize(), "f")
    return "0" if text in ("-0", "") else text


def build_json_body(fields: dict) -> str:
    """Serialise an order body compactly with Bitkub-compatible numbers. ``None`` values are omitted.

    Keys keep their insertion order, which is also the order that is signed and sent, so the signature always matches
    the bytes on the wire.
    """
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float, Decimal)):
            rendered = format_decimal(value)
        else:
            rendered = json.dumps(str(value))
        parts.append(f"{json.dumps(str(key))}:{rendered}")
    return "{" + ",".join(parts) + "}"


def sign_request(secret: str, timestamp_ms: int, method: str, path: str, body: str = "") -> str:
    """Return the ``X-BTK-SIGN`` header value: hex HMAC-SHA256 over ``timestamp + METHOD + path + body``."""
    payload = f"{int(timestamp_ms)}{method.upper()}{path}{body}"
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def format_symbol(base: str, quote: str) -> str:
    """``("btc", "thb")`` -> ``"BTC_THB"``."""
    return f"{str(base).strip().upper()}_{str(quote).strip().upper()}"


def parse_symbol(symbol: str) -> tuple:
    """``"btc_thb"`` / ``"BTC-THB"`` / ``"BTC/THB"`` -> ``("BTC", "THB")``. Raises ValueError for anything else."""
    text = str(symbol).strip().upper().replace("-", "_").replace("/", "_")
    pieces = text.split("_")
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"Bitkub symbols look like BTC_THB, got {symbol!r}")
    return pieces[0], pieces[1]


class _RateLimiter:
    """Small sliding-window limiter, one window per endpoint family, shared by all threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._calls = {}

    @staticmethod
    def _limit_for(path: str) -> int:
        for prefix, per_second in _RATE_LIMITS_PER_SECOND:
            if path.startswith(prefix):
                return per_second
        return _DEFAULT_RATE_LIMIT_PER_SECOND

    def wait(self, path: str) -> None:
        limit = max(1, int(self._limit_for(path) * 0.8))
        key = next((prefix for prefix, _ in _RATE_LIMITS_PER_SECOND if path.startswith(prefix)), "default")
        while True:
            with self._lock:
                window = self._calls.setdefault(key, deque())
                now = time.monotonic()
                while window and now - window[0] >= 1.0:
                    window.popleft()
                if len(window) < limit:
                    window.append(now)
                    return
                delay = 1.0 - (now - window[0])
            time.sleep(max(delay, 0.001))


class BitkubClient:
    """Synchronous Bitkub REST client.

    Public (market data) calls need no credentials. Private calls require ``api_key`` and ``api_secret`` and raise
    ``BitkubAPIError`` immediately if they are missing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 2,
        time_sync_interval: float = 30.0,
        session=None,
    ):
        self._api_key = api_key or None
        self._api_secret = api_secret or None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.time_sync_interval = time_sync_interval
        self._session = session
        self._limiter = _RateLimiter()
        self._time_offset_ms = 0
        self._time_synced_at = None
        self._blocked_until = 0.0

    def __repr__(self) -> str:  # never reveal credentials
        return f"BitkubClient(base_url={self.base_url!r}, authenticated={self.has_credentials})"

    @property
    def has_credentials(self) -> bool:
        return bool(self._api_key and self._api_secret)

    # ------------------------------------------------------------------ transport
    def _http(self):
        if self._session is None:
            import requests  # deferred: importing the broker/data source must stay cheap

            self._session = requests.Session()
        return self._session

    def _timestamp_ms(self) -> int:
        """Server-aligned millisecond timestamp. The offset is refreshed from /api/v3/servertime periodically."""
        stale = self._time_synced_at is None or (time.monotonic() - self._time_synced_at) > self.time_sync_interval
        if stale:
            self.sync_time()
        return int(time.time() * 1000) + self._time_offset_ms

    def sync_time(self) -> int:
        """Fetch the exchange clock and remember its offset from the local clock. Returns the server time (ms)."""
        local_before = int(time.time() * 1000)
        server_ms = self.get_server_time()
        local_after = int(time.time() * 1000)
        self._time_offset_ms = server_ms - (local_before + local_after) // 2
        self._time_synced_at = time.monotonic()
        return server_ms

    @staticmethod
    def _parse(text: str):
        try:
            return json.loads(text, parse_float=Decimal)
        except ValueError as exc:
            raise BitkubAPIError(f"Bitkub returned a non-JSON response: {text[:120]!r}") from exc

    def _request(
        self, method: str, path: str, params: dict | None = None, body: dict | None = None, auth: bool = False
    ):
        method = method.upper()
        if auth and not self.has_credentials:
            raise BitkubAPIError("Bitkub API credentials are required for this call", endpoint=path)
        retryable = method == "GET"  # a repeated POST could create a second order
        attempts = (self.max_retries if retryable else 0) + 1
        attempt = 0
        resynced = False

        while True:
            attempt += 1
            if time.monotonic() < self._blocked_until:
                raise BitkubRateLimitError(
                    "Bitkub rate limit block in force (30 s); not sending request",
                    code=429,
                    endpoint=path,
                    http_status=429,
                )
            self._limiter.wait(path)

            query = urlencode(params) if params else ""
            signed_path = f"{path}?{query}" if query else path
            body_text = build_json_body(body) if body is not None else ""
            headers = {"Accept": "application/json", "Content-Type": "application/json"}
            if auth:
                timestamp = self._timestamp_ms()
                headers["X-BTK-APIKEY"] = self._api_key
                headers["X-BTK-TIMESTAMP"] = str(timestamp)
                headers["X-BTK-SIGN"] = sign_request(self._api_secret, timestamp, method, signed_path, body_text)

            try:
                response = self._http().request(
                    method,
                    f"{self.base_url}{signed_path}",
                    headers=headers,
                    data=body_text.encode("utf-8") if body is not None else None,
                    timeout=self.timeout,
                )
            except Exception as exc:  # requests exceptions are imported lazily; any transport failure lands here
                if attempt < attempts:
                    time.sleep(min(2.0 ** (attempt - 1), 4.0))
                    continue
                raise BitkubTransportError(
                    f"{method} {path} failed in transit ({type(exc).__name__}); it may not have been processed",
                    endpoint=path,
                ) from exc

            status = response.status_code
            if status == 429:
                self._blocked_until = time.monotonic() + _RATE_LIMIT_BLOCK_SECONDS
                raise BitkubRateLimitError(
                    "Bitkub rate limit exceeded (HTTP 429)", code=429, endpoint=path, http_status=status
                )
            if status >= 500:
                if attempt < attempts:
                    time.sleep(min(2.0 ** (attempt - 1), 4.0))
                    continue
                raise BitkubTransportError(f"{method} {path} returned HTTP {status}", endpoint=path, http_status=status)

            payload = self._parse(response.text)
            try:
                return self._unwrap(payload, path, status)
            except BitkubAPIError as exc:
                # The exchange rejects a bad timestamp *before* touching the order book, so one resync + retry is safe
                # even for POST. Anything else is returned to the caller untouched.
                if auth and exc.code == ERROR_INVALID_TIMESTAMP and not resynced:
                    resynced = True
                    self.sync_time()
                    continue
                raise

    @staticmethod
    def _unwrap(payload, path: str, status: int):
        """Return the useful part of a response or raise BitkubAPIError. Handles the v3 and v4 envelopes."""
        if isinstance(payload, dict) and "error" in payload:  # v3 envelope
            code = payload["error"]
            if code not in (0, None):
                message = ERROR_MESSAGES.get(int(code), "Unknown error") if isinstance(code, int) else str(code)
                raise BitkubAPIError(
                    f"Bitkub error {code}: {message} ({path})", code=code, endpoint=path, http_status=status
                )
            return payload.get("result")
        if (
            isinstance(payload, dict) and "code" in payload and ("data" in payload or "message" in payload)
        ):  # v4 envelope
            if str(payload["code"]) != "0":
                raise BitkubAPIError(
                    f"Bitkub error {payload['code']}: {payload.get('message')} ({path})",
                    code=str(payload["code"]),
                    endpoint=path,
                    http_status=status,
                )
            return payload.get("data")
        if status >= 400:
            raise BitkubAPIError(f"Bitkub HTTP {status} for {path}", endpoint=path, http_status=status)
        return payload  # bare payloads: servertime (int), ticker (list), status (list)

    # ------------------------------------------------------------------ public market data
    def get_server_time(self) -> int:
        """Exchange clock in milliseconds."""
        value = self._request("GET", "/api/v3/servertime")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise BitkubAPIError(f"Unexpected servertime payload: {value!r}", endpoint="/api/v3/servertime") from exc

    def get_status(self) -> list:
        """Endpoint health: ``[{"name": ..., "status": "ok", "message": ...}, ...]``."""
        return self._request("GET", "/api/status")

    def get_symbols(self) -> list:
        """All trading symbols with precision, minimum size, status and ``source`` ("exchange" or "broker")."""
        return self._request("GET", "/api/v3/market/symbols")

    def get_ticker(self, symbol: str | None = None) -> list:
        """Ticker rows (``last``, ``highest_bid``, ``lowest_ask``, 24h stats); all symbols if ``symbol`` is omitted."""
        return self._request("GET", "/api/v3/market/ticker", params={"sym": symbol} if symbol else None)

    def get_depth(self, symbol: str, limit: int = 10) -> dict:
        """Order-book snapshot: ``{"asks": [[price, size], ...], "bids": [[price, size], ...]}``."""
        return self._request("GET", "/api/v3/market/depth", params={"sym": symbol, "lmt": int(limit)})

    def get_bids(self, symbol: str, limit: int | None = None) -> list:
        params = {"sym": symbol}
        if limit:
            params["lmt"] = int(limit)
        return self._request("GET", "/api/v3/market/bids", params=params)

    def get_asks(self, symbol: str, limit: int | None = None) -> list:
        params = {"sym": symbol}
        if limit:
            params["lmt"] = int(limit)
        return self._request("GET", "/api/v3/market/asks", params=params)

    def get_trades(self, symbol: str, limit: int = 10) -> list:
        """Recent trades as ``[timestamp_seconds, price, size, "BUY"|"SELL"]`` rows."""
        return self._request("GET", "/api/v3/market/trades", params={"sym": symbol, "lmt": int(limit)})

    def get_tradingview_history(self, symbol: str, resolution: str, from_ts: int, to_ts: int) -> dict:
        """OHLCV candles: ``{"t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...], "s": "ok"}``.

        ``resolution`` is one of ``1, 5, 15, 60, 240, 1D``; ``from_ts``/``to_ts`` are Unix **seconds**. A range with
        no candles answers ``{"s": "no_data"}`` (with empty arrays), which is returned as-is and never filled in.
        """
        if str(resolution) not in RESOLUTIONS:
            raise ValueError(f"Bitkub resolution must be one of {RESOLUTIONS}, got {resolution!r}")
        params = {"symbol": symbol, "resolution": str(resolution), "from": int(from_ts), "to": int(to_ts)}
        return self._request("GET", "/tradingview/history", params=params)

    # ------------------------------------------------------------------ private: account
    def get_balances(self) -> list:
        """Funding-wallet balances: ``[{"currency": "BTC", "available": "...", "reserved": "...", "total": "..."}]``."""
        return self._request("GET", "/api/v4/wallet/balances", auth=True)

    # ------------------------------------------------------------------ private: trading
    def _place(self, path: str, symbol, amount, rate, order_type, client_id, post_only) -> dict:
        order_type = str(order_type).lower()
        if order_type not in ("limit", "market"):
            raise ValueError("Bitkub supports only limit and market orders")
        rate = Decimal(str(rate)) if rate is not None else Decimal(0)
        if order_type == "market":
            rate = Decimal(0)  # documented: rat = 0 means market
        elif rate <= 0:
            raise ValueError("a limit order needs a positive rate")
        amount = Decimal(str(amount))
        if amount <= 0:
            raise ValueError("order amount must be positive")
        body = {
            "sym": str(symbol).lower(),
            "amt": amount,
            "rat": rate,
            "typ": order_type,
            "client_id": client_id,
            "post_only": post_only,
        }
        return self._request("POST", path, body=body, auth=True)

    def place_bid(
        self,
        symbol: str,
        amount,
        rate=0,
        order_type: str = "limit",
        client_id: str | None = None,
        post_only: bool | None = None,
    ) -> dict:
        """Buy. ``amount`` is the **quote currency to spend** (THB/USDT), not a coin quantity. ``rate`` 0 = market."""
        return self._place("/api/v3/market/place-bid", symbol, amount, rate, order_type, client_id, post_only)

    def place_ask(
        self,
        symbol: str,
        amount,
        rate=0,
        order_type: str = "limit",
        client_id: str | None = None,
        post_only: bool | None = None,
    ) -> dict:
        """Sell. ``amount`` is the **coin quantity to sell**. ``rate`` 0 = market."""
        return self._place("/api/v3/market/place-ask", symbol, amount, rate, order_type, client_id, post_only)

    def cancel_order(self, symbol: str, order_id: str, side: str) -> None:
        side = str(side).lower()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        self._request(
            "POST",
            "/api/v3/market/cancel-order",
            body={"sym": str(symbol).lower(), "id": str(order_id), "sd": side},
            auth=True,
        )

    def get_open_orders(self, symbol: str) -> list:
        """Open orders for ONE symbol (the endpoint requires it)."""
        return self._request("GET", "/api/v3/market/my-open-orders", params={"sym": str(symbol).lower()}, auth=True)

    def get_order_info(self, symbol: str, order_id: str, side: str) -> dict:
        return self._request(
            "GET",
            "/api/v3/market/order-info",
            params={"sym": str(symbol).lower(), "id": str(order_id), "sd": str(side).lower()},
            auth=True,
        )

    def get_order_history(
        self, symbol: str, limit: int = 10, page: int | None = None, start: int | None = None, end: int | None = None
    ) -> list:
        params = {"sym": str(symbol).upper(), "lmt": int(limit)}
        if page is not None:
            params["p"] = int(page)
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        return self._request("GET", "/api/v3/market/my-order-history", params=params, auth=True)
