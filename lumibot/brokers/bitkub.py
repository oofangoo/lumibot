"""Bitkub broker: spot crypto trading on the Thai exchange (THB and USDT markets).

Supported: market and limit orders on ``source == "exchange"`` markets, cancel, balances/positions, order tracking by
polling. Not supported (rejected loudly instead of being approximated): stop / stop-limit / trailing / smart-limit
orders, bracket/OCO/OTO/multi-leg orders, order modification, shorting and margin. Bitkub has no broker-side stop
orders, so protective stops must be handled by the strategy.

Design notes (see docs/BITKUB_BROKER.md for the full list of API quirks):

* ``amt`` means *quote currency to spend* on a buy and *coin quantity* on a sell, so a LumiBot buy of ``quantity`` coins
  is converted to a THB amount from the limit price (limit orders) or the live best ask (market orders). A market buy
  therefore receives *approximately* ``quantity`` coins (less the fee and any book-walking).
* Fills are dispatched as true increments over what LumiBot has already recorded for the order
  (``Order.add_transaction`` is additive), and an order that disappears from the open list is looked up with
  ``order-info`` rather than assumed cancelled, because filled market orders disappear too.
* A timed-out order submission is reconciled by ``client_id`` against open orders and recent history before it is
  reported as failed; it is never blindly re-sent.
"""

from __future__ import annotations

import os
import time
import uuid
from decimal import ROUND_DOWN, Decimal, InvalidOperation

from lumibot._lazy_imports import LazyLogger, LazyModule, lazy_class

from .broker import Broker, LumibotBrokerAPIError

logger = LazyLogger(__name__)
TYPE_CHECKING = False
pd = LazyModule("pandas")
Asset = lazy_class("lumibot.entities", "Asset")
Order = lazy_class("lumibot.entities", "Order")

if TYPE_CHECKING:
    from typing import Any

    from lumibot.entities import Position


def _position_class():
    from lumibot.entities import Position

    return Position


def _format_exc():
    import traceback

    return traceback.format_exc()


def _client_class():
    from lumibot.tools.bitkub_helpers import BitkubClient

    return BitkubClient


def _data_source_class():
    from lumibot.data_sources.bitkub_data import BitkubData

    return BitkubData


def _decimal(value, default=Decimal(0)) -> Decimal:
    if value is None or value == "":
        return default
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation:
        return default
    return number if number.is_finite() else default


class Bitkub(Broker):
    """Spot crypto broker for https://www.bitkub.com (REST polling; no WebSocket)."""

    ASSET_TYPE_MAP = dict(stock=[], option=[], future=[], crypto_future=[], forex=[], crypto=["crypto"])

    POLL_EVENT = "poll"
    DEFAULT_POLL_INTERVAL = 5  # seconds between polling cycles
    SYMBOL_RULES_TTL = 300  # seconds a /symbols snapshot is trusted
    DEFAULT_QUOTE_SYMBOL = "THB"
    UNKNOWN_SUBMIT_HISTORY_DEPTH = 20

    def __init__(
        self,
        config=None,
        max_workers: int = 1,
        chunk_size: int = 100,
        connect_stream: bool = True,
        poll_interval: float | None = None,
        data_source=None,
        client=None,
    ):
        if isinstance(config, dict):
            options = dict(config)
        else:
            options = {
                key: getattr(config, key, None)
                for key in ("API_KEY", "API_SECRET", "QUOTE_ASSET", "BASE_URL", "SYMBOLS")
            }
        api_key, api_secret = options.get("API_KEY"), options.get("API_SECRET")
        if client is None and not (api_key and api_secret):
            raise ValueError("API_KEY and API_SECRET must be provided in config")

        quote_symbol = str(
            options.get("QUOTE_ASSET") or os.environ.get("BITKUB_QUOTE_ASSET") or self.DEFAULT_QUOTE_SYMBOL
        ).upper()
        self._quote_symbol = quote_symbol
        self._quote_asset = None  # built lazily so importing this module stays cheap
        self._extra_symbols = {s.strip().upper() for s in str(options.get("SYMBOLS") or "").split(",") if s.strip()}

        if client is None:
            extra = {"base_url": options["BASE_URL"]} if options.get("BASE_URL") else {}
            client = _client_class()(api_key, api_secret, **extra)
        self.api = client
        self._symbol_rules: dict[str, dict] | None = None
        self._symbol_rules_at = 0.0
        self._order_registry: dict[str, tuple[str, str]] = {}  # order id -> (symbol, "buy" | "sell")
        self._watched_symbols: set[str] = set(self._extra_symbols)
        self._held_symbols: set[str] = set()

        if data_source is None:
            data_source = _data_source_class()(options, max_workers=max_workers, chunk_size=chunk_size, client=self.api)

        self.poll_interval = poll_interval or self.DEFAULT_POLL_INTERVAL
        super().__init__(
            name="bitkub",
            connect_stream=connect_stream,
            data_source=data_source,
            config=config,
            max_workers=max_workers,
        )
        self.market = "24/7"  # Broker.__init__ resets this from env/default; Bitkub trades around the clock

    # ------------------------------------------------------------------ clock and quote asset
    def get_timestamp(self):
        return time.time()

    def is_market_open(self):
        return True

    def get_time_to_open(self):
        return 0

    def get_time_to_close(self):
        return float("inf")

    def get_quote_asset(self):
        if self._quote_asset is None:
            kind = Asset.AssetType.FOREX if self._quote_symbol == "THB" else Asset.AssetType.CRYPTO
            self._quote_asset = Asset(self._quote_symbol, kind)
        if self._quote_asset not in self.quote_assets:
            self.quote_assets.add(self._quote_asset)
        return self._quote_asset

    def _quote_symbol_for(self, quote_asset) -> str:
        """The strategy's quote asset when Bitkub has such markets (THB/USDT); otherwise the broker default."""
        symbol = str(getattr(quote_asset, "symbol", "") or "").upper()
        return symbol if symbol in ("THB", "USDT") else self._quote_symbol

    def _symbol_for_order(self, order) -> str:
        quote = getattr(order, "quote", None)
        quote_symbol = self._quote_symbol_for(quote) if quote is not None else self._quote_symbol
        return f"{order.asset.symbol.upper()}_{quote_symbol}"

    # ------------------------------------------------------------------ market rules
    def _rules_for(self, symbol: str) -> dict:
        """Precision, minimum size, status and source for one market (cached; stale beats none for safety checks)."""
        now = time.monotonic()
        if self._symbol_rules is None or now - self._symbol_rules_at > self.SYMBOL_RULES_TTL:
            try:
                rows = self.api.get_symbols()
                self._symbol_rules = {str(row["symbol"]).upper(): row for row in rows}
                self._symbol_rules_at = now
            except Exception as exc:
                if self._symbol_rules is None:
                    raise LumibotBrokerAPIError(f"Cannot load Bitkub market rules: {exc}") from exc
                logger.warning("Bitkub market rules refresh failed (%s); using the previous snapshot", exc)
        rule = self._symbol_rules.get(symbol)
        if rule is None:
            raise LumibotBrokerAPIError(f"Bitkub has no market {symbol}")
        return rule

    @staticmethod
    def _round_down(value: Decimal, scale: int) -> Decimal:
        return value.quantize(Decimal(1).scaleb(-int(scale)), rounding=ROUND_DOWN)

    def _fresh_ticker(self, symbol: str) -> dict:
        rows = self.api.get_ticker(symbol)
        row = next((r for r in (rows or []) if str(r.get("symbol", "")).upper() == symbol), None)
        if row is None:
            raise LumibotBrokerAPIError(f"Bitkub has no ticker for {symbol}")
        return row

    # ------------------------------------------------------------------ account
    def _read_balances(self) -> list[dict]:
        try:
            rows = self.api.get_balances()
        except Exception as exc:
            raise LumibotBrokerAPIError(f"Unable to read Bitkub balances: {exc}") from exc
        if not isinstance(rows, list) or any(not isinstance(r, dict) or "currency" not in r for r in rows):
            raise LumibotBrokerAPIError("Bitkub balances response was not a list of {currency,...} rows")
        return rows

    def _get_balances_at_broker(self, quote_asset, strategy) -> tuple[float, float, float] | None:
        """Return ``(cash, positions_value, portfolio_value)`` in the quote currency.

        ``cash`` is the *total* quote balance (as for the CCXT broker), so THB reserved by open buy orders is included;
        a buy that exceeds the unreserved amount is rejected by the exchange (error 18) and surfaces as an ERROR order.
        Coins without a market in the quote currency cannot be valued and are left out of ``positions_value``.
        """
        quote_symbol = self._quote_symbol_for(quote_asset)
        cash = Decimal(0)
        coins = []
        for row in self._read_balances():
            currency, total = str(row["currency"]).upper(), _decimal(row.get("total"))
            if currency == quote_symbol:
                cash = total
            elif total > 0:
                coins.append((currency, total))

        positions_value = Decimal(0)
        if coins:
            try:
                tickers = {str(r.get("symbol", "")).upper(): r for r in (self.api.get_ticker() or [])}
            except Exception as exc:
                raise LumibotBrokerAPIError(f"Unable to price Bitkub holdings: {exc}") from exc
            unvalued = []
            for currency, total in coins:
                row = tickers.get(f"{currency}_{quote_symbol}")
                price = _decimal(row.get("last")) if row else Decimal(0)
                if price <= 0:
                    unvalued.append(currency)
                    continue
                positions_value += total * price
            if unvalued:
                logger.info(
                    "Bitkub holdings without a %s market (not valued): %s", quote_symbol, ", ".join(sorted(unvalued))
                )
        return float(cash), float(positions_value), float(cash + positions_value)

    def _pull_positions(self, strategy) -> list[Position]:
        """Complete snapshot of coin holdings (available + reserved), or raise.

        Raising on an unreadable response matters: the shared position sync prunes assets that are absent, so a partial
        or empty answer would make held coins look sold. The quote currency is cash, not a position.
        """
        Position = _position_class()
        strategy_name = self._strategy_name_from_input(strategy) or ""
        positions, held_symbols = [], set()
        for row in self._read_balances():
            currency, total = str(row["currency"]).upper(), _decimal(row.get("total"))
            if currency == self._quote_symbol or total <= 0:
                continue
            asset = Asset(currency, Asset.AssetType.CRYPTO)
            positions.append(Position(strategy_name, asset, total))
            held_symbols.add(f"{currency}_{self._quote_symbol}")
        self._held_symbols = held_symbols
        return positions

    def _pull_position(self, strategy, asset) -> Position | None:
        return next((p for p in self._pull_positions(strategy) if p.asset == asset), None)

    def get_historical_account_value(self, start_date=None, end_date=None, frequency=None) -> dict:
        """Bitkub exposes no account-value history."""
        logger.error("get_historical_account_value is not implemented for the Bitkub broker.")
        return {}

    # ------------------------------------------------------------------ order translation
    @staticmethod
    def _bitkub_side(order) -> str:
        side = str(order.side).lower()
        if side not in ("buy", "sell"):
            raise LumibotBrokerAPIError(f"Bitkub supports only buy and sell orders (spot), not {order.side!r}")
        return side

    def _validate_supported(self, order) -> None:
        if order.asset is None or order.asset.asset_type != Asset.AssetType.CRYPTO:
            raise LumibotBrokerAPIError("Bitkub trades spot crypto only; the order asset must be a crypto Asset")
        order_type = str(order.order_type).lower()
        if order_type not in ("market", "limit"):
            raise LumibotBrokerAPIError(
                f"Bitkub supports only market and limit orders, not {order_type!r}. "
                "It has no broker-side stop orders; manage protective stops in the strategy."
            )
        if str(getattr(order, "order_class", "simple")).lower() not in ("simple", "none"):
            raise LumibotBrokerAPIError("Bitkub does not support bracket, OCO, OTO or multi-leg orders")
        for field in (
            "take_profit_price",
            "stop_loss_price",
            "secondary_limit_price",
            "secondary_stop_price",
            "stop_price",
            "stop_limit_price",
            "trail_price",
            "trail_percent",
        ):
            if getattr(order, field, None) is not None:
                raise LumibotBrokerAPIError(f"Bitkub cannot honour {field}; remove it or use a different broker")

    def _prepare_order(self, order) -> dict:
        """Validate an order against Bitkub's rules and compute the exact request. Raises LumibotBrokerAPIError."""
        self._validate_supported(order)
        side = self._bitkub_side(order)
        symbol = self._symbol_for_order(order)
        rule = self._rules_for(symbol)
        if rule.get("status") != "active":
            raise LumibotBrokerAPIError(f"Bitkub market {symbol} is {rule.get('status')}, not active")
        if rule.get("source") != "exchange":
            raise LumibotBrokerAPIError(
                f"{symbol} is a broker-sourced market; Bitkub's trading API rejects those (error 61)"
            )
        if rule.get("freeze_buy" if side == "buy" else "freeze_sell"):
            raise LumibotBrokerAPIError(f"Bitkub has frozen {side} orders on {symbol}")

        coin_scale, quote_scale, price_scale = (
            int(rule[k]) for k in ("base_asset_scale", "quote_asset_scale", "price_scale")
        )
        minimum = _decimal(rule.get("min_quote_size"))
        quantity = _decimal(order.quantity)
        if quantity <= 0:
            raise LumibotBrokerAPIError(f"Order quantity must be positive, got {order.quantity}")
        is_limit = str(order.order_type).lower() == "limit"

        rate = Decimal(0)
        if is_limit:
            rate = self._round_down(_decimal(order.limit_price), price_scale)
            if rate <= 0:
                raise LumibotBrokerAPIError("A limit order needs a positive limit_price")

        if side == "sell":
            reference = rate if is_limit else _decimal(self._fresh_ticker(symbol).get("highest_bid"))
        else:
            reference = rate if is_limit else _decimal(self._fresh_ticker(symbol).get("lowest_ask"))
        if reference <= 0:
            raise LumibotBrokerAPIError(f"No usable {symbol} price to size a {side} order")

        if side == "sell":
            amount = self._round_down(quantity, coin_scale)  # coins to sell
            if amount <= 0:
                raise LumibotBrokerAPIError(f"{symbol} quantity {quantity} rounds to zero at {coin_scale} decimals")
            notional = amount * reference
            executable_quantity = amount
        else:
            amount = self._round_down(quantity * reference, quote_scale)  # THB (quote) to spend
            notional = amount
            executable_quantity = self._round_down(amount / reference, coin_scale)
        if notional < minimum:
            raise LumibotBrokerAPIError(f"{symbol} order value {notional} is below Bitkub's minimum of {minimum}")
        return {
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "rate": rate,
            "type": "limit" if is_limit else "market",
            "executable_quantity": executable_quantity,
            "client_id": self._new_client_id(),
        }

    @staticmethod
    def _new_client_id() -> str:
        return f"lmbot-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"

    def _fail_order(self, order, message: str, raw=None):
        error = LumibotBrokerAPIError(message)
        order.set_error(error)
        order.status = Order.OrderStatus.ERROR
        if raw is not None:
            order.update_raw(raw)
        self._safe_stream_dispatch(self.ERROR_ORDER, order=order, error_msg=message)
        return order

    def _safe_stream_dispatch(self, event, **kwargs):
        stream = getattr(self, "stream", None)
        if stream is not None and hasattr(stream, "dispatch"):
            stream.dispatch(event, **kwargs)

    def _register_order(self, identifier, symbol: str, side: str) -> None:
        self._order_registry[str(identifier)] = (symbol, side)
        self._watched_symbols.add(symbol)

    def _adopt_submitted(self, order, request: dict, result: dict):
        """Common success path once Bitkub has accepted (or we have proven it accepted) the order."""
        order.identifier = str(result.get("id"))
        order.quantity = request["executable_quantity"]  # fill tracking must use the executable size
        if request["type"] == "limit":
            order.limit_price = request["rate"]
        order.status = Order.OrderStatus.SUBMITTED
        order.update_raw(result)
        self._register_order(order.identifier, request["symbol"], request["side"])
        self._unprocessed_orders.append(order)
        self._process_trade_event(order, self.NEW_ORDER)
        return order

    def _submit_orders(self, orders, is_multileg=False, order_type=None, duration="day", price=None):
        if is_multileg:
            raise NotImplementedError("Bitkub does not support multi-leg orders.")
        return [self._submit_order(order) for order in orders]

    def _submit_order(self, order) -> Order:
        from lumibot.tools.bitkub_helpers import BitkubAPIError, BitkubTransportError

        try:
            request = self._prepare_order(order)
        except LumibotBrokerAPIError as exc:
            return self._fail_order(order, f"Bitkub order rejected before sending: {exc}")
        except Exception as exc:
            logger.error(_format_exc())
            return self._fail_order(order, f"Bitkub order could not be prepared: {exc}")

        place = self.api.place_bid if request["side"] == "buy" else self.api.place_ask
        try:
            result = place(
                request["symbol"], request["amount"], request["rate"], request["type"], client_id=request["client_id"]
            )
        except BitkubTransportError as exc:
            logger.warning(
                "Bitkub order submission was interrupted (%s); reconciling by client_id before reporting", exc
            )
            found = self._reconcile_unknown_submit(request)
            if found is not None:
                return self._adopt_submitted(order, request, found)
            return self._fail_order(
                order,
                f"Bitkub order state UNKNOWN after a transport failure ({exc}); it was not found by client_id "
                f"{request['client_id']}. Check the Bitkub app before retrying.",
            )
        except BitkubAPIError as exc:
            return self._fail_order(order, f"Bitkub rejected the order: {exc}")
        except Exception as exc:
            logger.error(_format_exc())
            return self._fail_order(order, f"Exception placing Bitkub order: {exc}")

        if not isinstance(result, dict) or not result.get("id"):
            return self._fail_order(
                order, f"Bitkub accepted the request but returned no order id: {result!r}", raw=result
            )
        return self._adopt_submitted(order, request, result)

    def _reconcile_unknown_submit(self, request: dict) -> dict | None:
        """After a lost response, look for the order by client_id in open orders, then in recent history."""
        try:
            for row in self.api.get_open_orders(request["symbol"]) or []:
                if row.get("client_id") == request["client_id"]:
                    return {"id": row.get("id"), "reconciled": "open_orders"}
            for row in self.api.get_order_history(request["symbol"], limit=self.UNKNOWN_SUBMIT_HISTORY_DEPTH) or []:
                if row.get("client_id") == request["client_id"]:
                    return {"id": row.get("order_id"), "reconciled": "order_history"}
        except Exception as exc:
            logger.warning("Could not reconcile the interrupted Bitkub order: %s", exc)
        return None

    def cancel_order(self, order) -> None:
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        if not order.identifier:
            logger.warning("Cannot cancel a Bitkub order that has no identifier")
            return
        context = self._order_context(order)
        if context is None:
            logger.error("Cannot cancel Bitkub order %s: its market and side are unknown", order.identifier)
            return
        symbol, side = context
        try:
            self.api.cancel_order(symbol, order.identifier, side)
        except BitkubAPIError as exc:
            # A failed cancel must NOT mark a possibly-live order as errored. Ask what really happened (it may
            # already be filled or cancelled) and let polling settle anything still unclear.
            logger.error("Failed to cancel Bitkub order %s: %s; checking its real state", order.identifier, exc)
            self._reconcile_order(order, direct=True)
            return
        # Book any fill that happened before the cancel took effect, then close the order.
        info = self._pull_broker_order(order.identifier)
        parsed = self._parse_broker_order(info, order.strategy) if info else None
        if parsed is not None and _decimal(parsed.filled_quantity) > self._recorded_fills(order)[0]:
            self._dispatch_fill_delta(order, parsed, direct=True)
        self._process_trade_event(order, self.CANCELED_ORDER)

    def _modify_order(self, order, limit_price=None, stop_price=None):
        raise LumibotBrokerAPIError("Bitkub cannot modify orders; cancel the order and submit a new one")

    def _order_context(self, order_or_id):
        identifier = str(getattr(order_or_id, "identifier", order_or_id))
        if identifier in self._order_registry:
            return self._order_registry[identifier]
        for order in self.get_all_orders():
            if str(order.identifier) == identifier and order.asset is not None:
                try:
                    return self._symbol_for_order(order), self._bitkub_side(order)
                except LumibotBrokerAPIError:
                    return None
        return None

    # ------------------------------------------------------------------ order queries
    def _symbols_to_poll(self) -> set[str]:
        symbols = set(self._watched_symbols) | set(self._held_symbols)
        for order in self.get_tracked_orders():
            if order.is_active() and order.asset is not None:
                symbols.add(self._symbol_for_order(order))
        return symbols

    def _pull_broker_all_orders(self) -> list[dict]:
        """Open orders for every market we know about. Raises if ANY market cannot be read.

        Bitkub lists open orders per symbol only. A partial answer would make an unread market's orders look vanished,
        so the caller must skip the cycle instead of acting on an incomplete list.
        """
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        rows = []
        for symbol in sorted(self._symbols_to_poll()):
            try:
                result = self.api.get_open_orders(symbol)
            except BitkubAPIError as exc:
                if exc.code == 11:  # a market that does not exist cannot hold orders
                    continue
                raise LumibotBrokerAPIError(f"Unable to read open Bitkub orders for {symbol}: {exc}") from exc
            except Exception as exc:
                raise LumibotBrokerAPIError(f"Unable to read open Bitkub orders for {symbol}: {exc}") from exc
            for row in result or []:
                row = dict(row)
                row["_symbol"] = symbol
                rows.append(row)
        return rows

    def _pull_broker_order(self, identifier: str) -> dict | None:
        """Authoritative single-order lookup (needs the market and side, which we remember per order id)."""
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        context = self._order_context(identifier)
        if context is None:
            logger.warning("Cannot look up Bitkub order %s: its market and side are unknown", identifier)
            return None
        symbol, side = context
        try:
            info = self.api.get_order_info(symbol, identifier, side)
        except BitkubAPIError as exc:
            logger.warning("Bitkub order-info for %s failed: %s", identifier, exc)
            return None
        if not isinstance(info, dict):
            return None
        info = dict(info)
        info.setdefault("id", str(identifier))
        info["_symbol"], info["_side"] = symbol, side
        return info

    @staticmethod
    def _fills_from_info(info: dict, side: str) -> tuple[Decimal, Decimal]:
        """Cumulative ``(quantity, value)`` filled, derived from the per-fill ``history`` of order-info.

        Buy fills are in quote currency (THB): coins received = (amount - fee) / rate. Sell fills are in coins.
        The history semantics come from the official docs and are the one part of this broker that needs a live
        confirmation with a small order; positions themselves always come from wallet balances, never from
        this arithmetic.
        """
        quantity = value = Decimal(0)
        for fill in info.get("history") or []:
            rate, amount, fee = _decimal(fill.get("rate")), _decimal(fill.get("amount")), _decimal(fill.get("fee"))
            if rate <= 0 or amount <= 0:
                continue
            coins = (amount - fee) / rate if side == "buy" else amount
            if coins > 0:
                quantity += coins
                value += coins * rate
        return quantity, value

    def _parse_broker_order(self, response: dict, strategy_name: str, strategy_object: Any = None) -> Order | None:
        """Convert an open-orders row or an order-info payload (both annotated with ``_symbol``) to an Order."""
        if not response:
            return None
        try:
            base, quote_symbol = str(response["_symbol"]).upper().split("_")
            side = str(response.get("side") or response.get("_side") or "").lower()
            if side not in ("buy", "sell"):
                raise ValueError(f"unknown side {side!r}")
            rate = _decimal(response.get("rate"))
            amount = _decimal(response.get("amount"))
            is_info = "status" in response
            if side == "sell":
                quantity = amount
            elif rate > 0:
                quantity = amount / rate
            else:
                quantity = (
                    _decimal(response.get("receive")) if not is_info else self._fills_from_info(response, side)[0]
                )
            quote_kind = Asset.AssetType.FOREX if quote_symbol == "THB" else Asset.AssetType.CRYPTO
            order_type = str(response.get("type") or ("limit" if rate > 0 else "market")).lower()
            filled_quantity, filled_value = (
                self._fills_from_info(response, side) if is_info else (Decimal(0), Decimal(0))
            )

            if is_info:
                status_text = str(response.get("status", "")).lower()
                if status_text == "filled":
                    status = Order.OrderStatus.FILLED
                elif status_text in ("cancelled", "canceled"):
                    status = Order.OrderStatus.CANCELED
                elif response.get("partial_filled") or filled_quantity > 0:
                    status = Order.OrderStatus.PARTIALLY_FILLED
                else:
                    status = Order.OrderStatus.SUBMITTED
            else:
                status = Order.OrderStatus.SUBMITTED

            order = Order(
                strategy=strategy_name,
                asset=Asset(base, Asset.AssetType.CRYPTO),
                quantity=quantity,
                side=Order.OrderSide.BUY if side == "buy" else Order.OrderSide.SELL,
                order_type=Order.OrderType.LIMIT if order_type == "limit" else Order.OrderType.MARKET,
                limit_price=rate if rate > 0 else None,
                status=status,
                identifier=str(response.get("id") or response.get("order_id")),
                quote=Asset(quote_symbol, quote_kind),
            )
            order.filled_quantity = filled_quantity
            order.avg_fill_price = (filled_value / filled_quantity) if filled_quantity > 0 else None
            created = response.get("ts")
            if created:
                order.broker_create_date = pd.to_datetime(int(created), unit="ms", utc=True)
            order.update_raw(response)
            return order
        except Exception as exc:
            logger.error("Error parsing Bitkub order: %s", exc)
            logger.error(_format_exc())
            return None

    # ------------------------------------------------------------------ polling stream
    def _recorded_fills(self, order) -> tuple[Decimal, Decimal]:
        """Quantity and value LumiBot has already booked against an order (sum of its transactions)."""
        quantity = value = Decimal(0)
        for transaction in order.transactions:
            q, p = _decimal(transaction.quantity), _decimal(transaction.price)
            quantity += q
            value += q * p
        return quantity, value

    def _emit(self, event, direct: bool, order, **payload) -> None:
        """Deliver a trade event through the polling stream, or synchronously when the caller needs strict ordering."""
        if direct:
            self._process_trade_event(order, event, **payload)
        else:
            self._safe_stream_dispatch(event, order=order, **payload)

    def _dispatch_fill_delta(self, order, parsed, direct: bool = False) -> None:
        """Dispatch only the *new* fill since the last event: ``Order.add_transaction`` is additive."""
        cumulative_quantity = _decimal(parsed.filled_quantity)
        cumulative_value = cumulative_quantity * _decimal(parsed.avg_fill_price)
        booked_quantity, booked_value = self._recorded_fills(order)
        delta_quantity = cumulative_quantity - booked_quantity
        complete = parsed.status == Order.OrderStatus.FILLED
        if delta_quantity <= 0:
            if complete:  # the last partial event already carried everything
                last_price = (
                    float(order.transactions[-1].price)
                    if order.transactions
                    else float(_decimal(parsed.avg_fill_price))
                )
                self._emit(self.FILLED_ORDER, direct, order, price=last_price, filled_quantity=0.0)
            return
        price = (cumulative_value - booked_value) / delta_quantity
        if price <= 0:
            price = _decimal(parsed.avg_fill_price)
        if complete:
            order.quantity = cumulative_quantity  # a market buy's coin quantity is only known once it has filled
        event = self.FILLED_ORDER if complete else self.PARTIALLY_FILLED_ORDER
        self._emit(event, direct, order, price=float(price), filled_quantity=float(delta_quantity))

    def _reconcile_order(self, order, direct: bool = False) -> None:
        """Ask Bitkub (order-info) what really happened to a tracked active order and book the difference.

        The open-orders list cannot be trusted for this: a filled market order leaves it immediately and a
        partially filled limit order stays in it, so order-info is the single source of truth for status and fills.
        """
        info = self._pull_broker_order(order.identifier)
        if info is None:
            return
        parsed = self._parse_broker_order(info, order.strategy)
        if parsed is None:
            return
        if parsed.status in (Order.OrderStatus.FILLED, Order.OrderStatus.PARTIALLY_FILLED):
            self._dispatch_fill_delta(order, parsed, direct=direct)
        elif parsed.status == Order.OrderStatus.CANCELED:
            if _decimal(parsed.filled_quantity) > self._recorded_fills(order)[0]:
                self._dispatch_fill_delta(order, parsed, direct=direct)  # book what filled before the cancel
            self._emit(self.CANCELED_ORDER, direct, order)
        # SUBMITTED: still resting with no fills; nothing to do

    def do_polling(self):
        """Poll Bitkub: sync positions, publish new/changed open orders, and settle orders that left the open list."""
        try:
            self.sync_positions(None)
        except Exception as exc:
            logger.warning("Bitkub position sync failed this cycle: %s", exc)
        try:
            raw_orders = self._pull_broker_all_orders()
        except LumibotBrokerAPIError as exc:
            logger.warning("Skipping this Bitkub poll, open orders could not be read completely: %s", exc)
            return

        # 1. Discover orders LumiBot has not seen (placed in the Bitkub app or by another process).
        known_ids = {str(o.identifier) for o in self.get_all_orders()}
        for row in raw_orders:
            order = self._parse_broker_order(row, strategy_name=self._strategy_name)
            if order is None:
                continue
            self._register_order(order.identifier, row["_symbol"], str(row.get("side")).lower())
            if order.identifier not in known_ids:
                self._process_trade_event(order, self.NEW_ORDER)

        # 2. Settle every active order we track through order-info (status and fills), open or not.
        for order in self.get_tracked_orders():
            if order.is_active():
                self._reconcile_order(order)

    def _get_stream_object(self):
        from lumibot.trading_builtins import PollingStream

        return PollingStream(self.poll_interval)

    def _register_stream_events(self):
        broker = self

        @broker.stream.add_action(broker.POLL_EVENT)
        def on_trade_event_poll():
            broker.do_polling()

        @broker.stream.add_action(broker.NEW_ORDER)
        def on_trade_event_new(order):
            try:
                broker._process_trade_event(order, broker.NEW_ORDER)
                return True
            except Exception:
                logger.error(_format_exc())

        @broker.stream.add_action(broker.FILLED_ORDER)
        def on_trade_event_fill(order, price, filled_quantity):
            try:
                broker._process_trade_event(order, broker.FILLED_ORDER, price=price, filled_quantity=filled_quantity)
                return True
            except Exception:
                logger.error(_format_exc())

        @broker.stream.add_action(broker.PARTIALLY_FILLED_ORDER)
        def on_trade_event_partial(order, price, filled_quantity):
            try:
                broker._process_trade_event(
                    order, broker.PARTIALLY_FILLED_ORDER, price=price, filled_quantity=filled_quantity
                )
                return True
            except Exception:
                logger.error(_format_exc())

        @broker.stream.add_action(broker.CANCELED_ORDER)
        def on_trade_event_cancel(order):
            try:
                broker._process_trade_event(order, broker.CANCELED_ORDER)
            except Exception:
                logger.error(_format_exc())

        @broker.stream.add_action(broker.ERROR_ORDER)
        def on_trade_event_error(order, error_msg):
            logger.error("Bitkub order error %s | %s", order, error_msg)
            try:
                if order.is_active():
                    broker._process_trade_event(order, broker.ERROR_ORDER, error=LumibotBrokerAPIError(str(error_msg)))
                order.set_error(error_msg)
            except Exception:
                logger.error(_format_exc())

    def _run_stream(self):
        self._stream_established()
        try:
            self.stream._run()
        except Exception as exc:
            logger.error("Error running the Bitkub polling stream: %s", exc)
