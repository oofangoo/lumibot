"""Unit tests for the Bitkub broker. No network and no real money: the exchange is a scripted fake and the order
tracking is LumiBot's real machinery. Fixtures are trimmed recordings of the live public API (2026-10-10)."""

import copy
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lumibot.brokers.bitkub import Bitkub
from lumibot.brokers.broker import LumibotBrokerAPIError
from lumibot.entities import Asset, Order
from lumibot.tools.bitkub_helpers import BitkubAPIError, BitkubTransportError

FIXTURES = Path(__file__).parent / "fixtures" / "bitkub"
BTC = Asset("BTC", Asset.AssetType.CRYPTO)
ETH = Asset("ETH", Asset.AssetType.CRYPTO)
THB = Asset("THB", Asset.AssetType.FOREX)
CONFIG = {"API_KEY": "test_key", "API_SECRET": "test_secret"}


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClient:
    """Scripted stand-in for BitkubClient; records every call so tests can assert exactly what would be sent."""

    def __init__(self):
        self.calls = []
        self.symbols = load("symbols.json")["result"]
        self.ticker_rows = load("ticker_btc_thb.json")
        self.balances = [
            {"currency": "THB", "available": "600", "reserved": "400", "total": "1000"},
            {"currency": "BTC", "available": "0.4", "reserved": "0.1", "total": "0.5"},
        ]
        self.open_orders = {}  # symbol -> rows
        self.order_info = {}  # order id -> payload or exception
        self.history_rows = []
        self.place_result = {"id": "111"}
        self.place_error = None
        self.cancel_error = None
        self.balances_error = None
        self.open_orders_error = {}

    def _log(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def names(self):
        return [c[0] for c in self.calls]

    def get_symbols(self):
        return self.symbols

    def get_ticker(self, symbol=None):
        self._log("get_ticker", symbol)
        return self.ticker_rows

    def get_balances(self):
        if self.balances_error:
            raise self.balances_error
        return self.balances

    def _place(self, name, symbol, amount, rate, order_type, client_id=None, post_only=None):
        self._log(name, symbol, amount, rate, order_type, client_id=client_id)
        if self.place_error:
            raise self.place_error
        return self.place_result

    def place_bid(self, *args, **kwargs):
        return self._place("place_bid", *args, **kwargs)

    def place_ask(self, *args, **kwargs):
        return self._place("place_ask", *args, **kwargs)

    def cancel_order(self, symbol, order_id, side):
        self._log("cancel_order", symbol, order_id, side)
        if self.cancel_error:
            raise self.cancel_error

    def get_open_orders(self, symbol):
        self._log("get_open_orders", symbol)
        if symbol in self.open_orders_error:
            raise self.open_orders_error[symbol]
        return self.open_orders.get(symbol, [])

    def get_order_info(self, symbol, order_id, side):
        self._log("get_order_info", symbol, order_id, side)
        item = self.order_info.get(order_id)
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise BitkubAPIError("Invalid order for lookup", code=24)
        return copy.deepcopy(item)

    def get_order_history(self, symbol, limit=10, **kwargs):
        self._log("get_order_history", symbol, limit)
        return self.history_rows


class FakeStream:
    def __init__(self):
        self.events = []

    def dispatch(self, event, wait_until_complete=False, **payload):
        self.events.append((event, payload))

    def of(self, event):
        return [payload for name, payload in self.events if name == event]


def make_broker(client=None, **kwargs):
    client = client or FakeClient()
    broker = Bitkub(CONFIG, client=client, connect_stream=False, **kwargs)
    broker.stream = FakeStream()
    broker.sync_positions = MagicMock()
    subscriber = MagicMock()  # the running strategy; LumiBot routes order events to it
    subscriber.name = "strat"
    broker._subscribers = [subscriber]
    broker.subscriber = subscriber
    return broker, client


def make_order(side="buy", quantity="0.001", order_type=Order.OrderType.LIMIT, limit_price=2500000, asset=BTC, **extra):
    if order_type == Order.OrderType.MARKET:
        limit_price = None
    return Order(
        "strat", asset, Decimal(quantity), side, limit_price=limit_price, order_type=order_type, quote=THB, **extra
    )


def info(order_id="111", status="unfilled", rate=2500000, amount="2500", partial=False, fills=()):
    return {
        "id": order_id,
        "status": status,
        "rate": rate,
        "amount": amount,
        "partial_filled": partial,
        "history": [
            {"amount": a, "fee": f, "rate": r, "id": order_id, "timestamp": 1, "txn_id": "t"} for a, f, r in fills
        ],
    }


# ------------------------------------------------------------------ construction
def test_credentials_are_required_unless_a_client_is_injected():
    with pytest.raises(ValueError):
        Bitkub({"API_KEY": "only-key"}, connect_stream=False)
    broker, _ = make_broker()
    assert broker.name == "bitkub"


def test_construction_makes_no_network_calls_and_sets_24_7():
    broker, client = make_broker()
    assert client.calls == []
    assert broker.market == "24/7" and broker.is_market_open() and broker.get_time_to_close() == float("inf")


def test_quote_asset_defaults_to_thb_and_can_be_usdt():
    broker, _ = make_broker()
    assert broker.get_quote_asset() == THB and broker.get_quote_asset().asset_type == Asset.AssetType.FOREX
    usdt_broker = Bitkub({**CONFIG, "QUOTE_ASSET": "usdt"}, client=FakeClient(), connect_stream=False)
    assert (
        usdt_broker.get_quote_asset().symbol == "USDT"
        and usdt_broker.get_quote_asset().asset_type == Asset.AssetType.CRYPTO
    )


# ------------------------------------------------------------------ balances and positions
def test_balances_are_total_quote_cash_plus_valued_holdings():
    broker, _ = make_broker()
    cash, positions_value, portfolio = broker._get_balances_at_broker(THB, None)
    assert cash == 1000.0  # total, including the 400 reserved by open buy orders
    assert positions_value == pytest.approx(0.5 * 2778954.76)
    assert portfolio == pytest.approx(cash + positions_value)


def test_coins_without_a_quote_market_are_left_out_of_the_valuation():
    broker, client = make_broker()
    client.balances.append({"currency": "ZZZ", "available": "5", "reserved": "0", "total": "5"})
    _, positions_value, _ = broker._get_balances_at_broker(THB, None)
    assert positions_value == pytest.approx(0.5 * 2778954.76)


def test_positions_exclude_cash_and_dust_zero_and_include_reserved_coins():
    broker, client = make_broker()
    client.balances.append({"currency": "ETH", "available": "0", "reserved": "0", "total": "0"})
    positions = broker._pull_positions(None)
    assert [(p.asset.symbol, p.quantity) for p in positions] == [
        ("BTC", Decimal("0.5"))
    ]  # 0.4 free + 0.1 in an open sell
    assert positions[0].asset.asset_type == Asset.AssetType.CRYPTO


@pytest.mark.parametrize("failure", ["error", "garbage", "rows"])
def test_unreadable_balances_raise_instead_of_looking_like_a_flat_account(failure):
    broker, client = make_broker()
    if failure == "error":
        client.balances_error = BitkubAPIError("boom")
    elif failure == "garbage":
        client.balances = {"not": "a list"}
    else:
        client.balances = [{"no_currency_key": 1}]
    with pytest.raises(LumibotBrokerAPIError):
        broker._pull_positions(
            None
        )  # the position sync prunes absent assets, so a partial answer must never be returned
    with pytest.raises(LumibotBrokerAPIError):
        broker._get_balances_at_broker(THB, None)


# ------------------------------------------------------------------ order sizing: what exactly is sent
def test_limit_buy_converts_coin_quantity_to_a_thb_amount_and_rounds_conservatively():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000.456))
    name, args, kwargs = client.calls[-1]
    # rate rounds DOWN to the 0.01 tick; THB to spend = quantity x rate, rounded DOWN to the cent
    assert name == "place_bid" and args == ("BTC_THB", Decimal("2500.00"), Decimal("2500000.45"), "limit")
    assert kwargs["client_id"].startswith("lmbot-")
    assert order.identifier == "111" and order.status == Order.OrderStatus.NEW  # accepted and tracked as a live order
    assert order.limit_price == Decimal("2500000.45")
    assert order.quantity == Decimal("0.00099999")  # the executable quantity, not the strategy's unrounded request


def test_market_buy_spends_thb_computed_from_the_live_best_ask():
    broker, client = make_broker()
    broker.submit_order(make_order("buy", "0.0005", Order.OrderType.MARKET))
    name, args, _ = next(c for c in client.calls if c[0] == "place_bid")
    # 0.0005 x lowest_ask 2778954.76 = 1389.47738 THB -> 1389.47; rate 0 means market
    assert args == ("BTC_THB", Decimal("1389.47"), Decimal(0), "market")
    assert ("get_ticker", ("BTC_THB",), {}) in client.calls  # the ask was fetched fresh, not guessed


def test_market_sell_sends_the_coin_quantity_rounded_down_to_eight_decimals():
    broker, client = make_broker()
    broker.submit_order(make_order("sell", "0.00123456789", Order.OrderType.MARKET))
    name, args, _ = next(c for c in client.calls if c[0] == "place_ask")
    assert args == ("BTC_THB", Decimal("0.00123456"), Decimal(0), "market")


def test_limit_sell_sends_coins_and_the_rounded_rate():
    broker, client = make_broker()
    broker.submit_order(make_order("sell", "0.01", limit_price=3000000.999))
    name, args, _ = next(c for c in client.calls if c[0] == "place_ask")
    assert args == ("BTC_THB", Decimal("0.01"), Decimal("3000000.99"), "limit")


def test_submitted_order_is_tracked_and_remembers_its_market_and_side():
    broker, _ = make_broker()
    broker.submit_order(make_order("sell", "0.01", limit_price=3000000))
    assert broker._order_registry["111"] == ("BTC_THB", "sell")
    assert [(o.identifier, o.status) for o in broker.get_tracked_orders()] == [("111", "new")]


# ------------------------------------------------------------------ orders Bitkub cannot do are refused loudly
def _with(order, **attrs):
    for key, value in attrs.items():
        setattr(order, key, value)
    return order


def _stop(**kw):
    return make_order("sell", "0.01", Order.OrderType.STOP, limit_price=None, stop_price=2400000, **kw)


@pytest.mark.parametrize(
    "build, reason",
    [
        (lambda: _stop(), "market and limit"),
        (
            lambda: make_order("sell", "0.01", Order.OrderType.STOP_LIMIT, limit_price=2400000, stop_price=2450000),
            "market and limit",
        ),
        (
            lambda: make_order("sell", "0.01", Order.OrderType.TRAIL, limit_price=None, trail_percent=5),
            "market and limit",
        ),
        (
            lambda: _with(
                make_order("buy", "0.01", limit_price=2500000), take_profit_price=3000000, stop_loss_price=2000000
            ),
            "cannot honour",
        ),
        (
            lambda: _with(make_order("buy", "0.01", limit_price=2500000), order_class=Order.OrderClass.BRACKET),
            "bracket",
        ),
        (lambda: make_order("sell_short", "0.01", Order.OrderType.MARKET), "buy and sell"),
        (lambda: make_order("buy", "0", limit_price=2500000), "positive"),
        (lambda: make_order("buy", "0.000001", limit_price=2500000), "minimum"),  # 2.5 THB < 10 THB minimum
        (lambda: make_order("sell", "0.000000001", Order.OrderType.MARKET), "rounds to zero"),
        (lambda: make_order("buy", "1", asset=Asset("WAN", Asset.AssetType.CRYPTO), limit_price=10), "stopped"),
        (lambda: make_order("buy", "1", asset=Asset("NEXO", Asset.AssetType.CRYPTO), limit_price=10), "broker-sourced"),
        (lambda: make_order("buy", "1", asset=Asset("NOPE", Asset.AssetType.CRYPTO), limit_price=10), "no market"),
    ],
    ids=[
        "stop",
        "stop_limit",
        "trailing",
        "take-profit-stop-loss",
        "bracket",
        "short-side",
        "zero-qty",
        "below-minimum",
        "rounds-to-zero",
        "stopped-market",
        "broker-coin",
        "unknown-market",
    ],
)
def test_unsupported_or_invalid_orders_fail_before_any_trading_call(build, reason):
    broker, client = make_broker()
    order = broker.submit_order(build())
    assert order.status == Order.OrderStatus.ERROR
    assert (
        reason in str(order.error_message).lower().replace("-", " ").replace("_", " ")
        or reason in str(order.error_message).lower()
    )
    assert not any(name.startswith("place_") for name in client.names())  # nothing was sent to the exchange
    assert broker.stream.of(broker.ERROR_ORDER)  # and the strategy is told


def test_frozen_markets_are_refused():
    broker, client = make_broker()
    for row in client.symbols:
        if row["symbol"] == "BTC_THB":
            row["freeze_buy"] = True
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    assert order.status == Order.OrderStatus.ERROR and "frozen" in str(order.error_message)
    assert not any(n.startswith("place_") for n in client.names())


def test_stock_assets_are_refused():
    broker, client = make_broker()
    order = broker.submit_order(
        Order("strat", Asset("SPY", Asset.AssetType.STOCK), 1, "buy", order_type=Order.OrderType.MARKET)
    )
    assert order.status == Order.OrderStatus.ERROR and "crypto" in str(order.error_message)
    assert client.calls == []


def test_exchange_rejections_become_error_orders_with_the_exchange_message():
    broker, client = make_broker()
    client.place_error = BitkubAPIError("Bitkub error 18: Insufficient balance", code=18)
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    assert order.status == Order.OrderStatus.ERROR and "Insufficient balance" in str(order.error_message)
    assert order.identifier != "111" and broker._order_registry == {} and broker.get_tracked_orders() == []
    assert len([n for n in client.names() if n == "place_bid"]) == 1


def test_a_response_without_an_order_id_is_an_error_not_a_ghost_order():
    broker, client = make_broker()
    client.place_result = {"typ": "limit"}
    assert broker.submit_order(make_order("buy", "0.001", limit_price=2500000)).status == Order.OrderStatus.ERROR


# ------------------------------------------------------------------ lost responses are reconciled, never re-sent
def test_interrupted_submission_is_adopted_when_the_order_is_found_open_by_client_id():
    broker, client = make_broker()
    client.place_error = BitkubTransportError("timeout")

    def open_orders(symbol):
        client._log("get_open_orders", symbol)
        client_id = next(c for c in client.calls if c[0] == "place_bid")[2]["client_id"]
        return [{"id": "777", "client_id": client_id, "side": "buy"}]

    client.get_open_orders = open_orders
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    assert order.identifier == "777" and order.status == Order.OrderStatus.NEW
    assert len([n for n in client.names() if n == "place_bid"]) == 1  # never re-sent


def test_interrupted_market_order_is_adopted_from_history_when_it_already_filled():
    broker, client = make_broker()
    client.place_error = BitkubTransportError("timeout")
    original = client.get_order_history

    def history(symbol, limit=10, **kw):
        client_id = next(c for c in client.calls if c[0] == "place_bid")[2]["client_id"]
        client.history_rows = [{"order_id": "888", "client_id": client_id, "side": "buy"}]
        return original(symbol, limit)

    client.get_order_history = history
    order = broker.submit_order(make_order("buy", "0.0005", Order.OrderType.MARKET))
    assert order.identifier == "888"


def test_interrupted_submission_that_cannot_be_found_is_reported_as_unknown_not_retried():
    broker, client = make_broker()
    client.place_error = BitkubTransportError("timeout")
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    assert order.status == Order.OrderStatus.ERROR and "UNKNOWN" in str(order.error_message)
    assert len([n for n in client.names() if n == "place_bid"]) == 1


# ------------------------------------------------------------------ cancel
def test_cancel_uses_the_remembered_market_and_side_and_books_the_cancel():
    broker, client = make_broker()
    order = broker.submit_order(make_order("sell", "0.01", limit_price=3000000))
    client.order_info["111"] = info("111", "cancelled", amount="0.01")
    broker.cancel_order(order)
    assert ("cancel_order", ("BTC_THB", "111", "sell"), {}) in client.calls
    assert order.status == Order.OrderStatus.CANCELED


def test_cancel_books_a_fill_that_happened_before_the_cancel_took_effect():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "cancelled", partial=True, fills=[(1000, 2.5, 2500000)])
    broker.cancel_order(order)
    assert [(float(t.price), float(t.quantity)) for t in order.transactions] == [(2500000.0, pytest.approx(0.000399))]
    assert order.status == Order.OrderStatus.CANCELED


def test_failed_cancel_never_marks_a_live_order_as_errored():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.cancel_error = BitkubAPIError("Invalid order for cancellation", code=21)
    client.order_info["111"] = info("111", "unfilled")  # still resting: the cancel simply failed
    broker.cancel_order(order)
    assert order.status == Order.OrderStatus.NEW and order.is_active()
    assert broker.stream.events == []


def test_failed_cancel_of_an_order_that_actually_filled_books_the_fill():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.cancel_error = BitkubAPIError("Invalid order for cancellation", code=21)
    client.order_info["111"] = info("111", "filled", fills=[(2500, 6.25, 2500000)])
    broker.cancel_order(order)
    assert order.status == Order.OrderStatus.FILLED
    assert float(order.transactions[0].quantity) == pytest.approx(0.0009975)


def test_cancel_without_known_context_does_nothing_dangerous():
    broker, client = make_broker()
    ghost = make_order("buy", "0.001", limit_price=2500000)
    ghost.identifier = "999"
    ghost.asset = None  # nothing to derive a market from
    broker.cancel_order(ghost)
    assert "cancel_order" not in client.names()


def test_orders_cannot_be_modified():
    broker, _ = make_broker()
    with pytest.raises(LumibotBrokerAPIError):
        broker._modify_order(make_order(), limit_price=1)


# ------------------------------------------------------------------ parsing the exchange's order shapes
def open_row(**kw):
    row = {
        "id": "2",
        "side": "sell",
        "type": "limit",
        "rate": "15000",
        "fee": "35.01",
        "credit": "35.01",
        "amount": "0.93333334",
        "receive": "14000",
        "parent_id": "1",
        "super_id": "1",
        "client_id": "cid",
        "ts": 1702543272000,
        "_symbol": "BTC_THB",
    }
    row.update(kw)
    return row


def test_parse_open_sell_order_amount_is_coins():
    broker, _ = make_broker()
    order = broker._parse_broker_order(open_row(), "strat")
    assert (order.asset.symbol, order.quote.symbol, str(order.side)) == ("BTC", "THB", "sell")
    assert order.quantity == Decimal("0.93333334") and order.limit_price == Decimal("15000")
    assert order.status == Order.OrderStatus.SUBMITTED and order.identifier == "2"
    assert order.broker_create_date.year == 2023


def test_parse_open_buy_order_amount_is_thb_so_quantity_is_amount_over_rate():
    broker, _ = make_broker()
    order = broker._parse_broker_order(open_row(id="3", side="buy", rate="10", amount="100", receive="9.975"), "strat")
    assert str(order.side) == "buy" and order.quantity == Decimal("10")


def test_parse_order_info_status_mapping_and_fill_math():
    broker, _ = make_broker()
    filled = info("5", "filled", rate=291000, amount="4000", fills=[(98.14848, 0.25, 291000)])
    filled.update(_symbol="BTC_THB", _side="buy")
    order = broker._parse_broker_order(filled, "strat")
    assert order.status == Order.OrderStatus.FILLED
    assert (
        float(order.filled_quantity) == pytest.approx((98.14848 - 0.25) / 291000)
        and float(order.avg_fill_price) == 291000
    )

    def status_of(**kw):
        payload = info("6", **kw)
        payload.update(_symbol="BTC_THB", _side="sell")
        return broker._parse_broker_order(payload, "strat").status

    assert status_of(status="unfilled") == Order.OrderStatus.SUBMITTED
    assert status_of(status="unfilled", partial=True) == Order.OrderStatus.PARTIALLY_FILLED
    assert status_of(status="cancelled") == Order.OrderStatus.CANCELED


def test_fill_math_weights_the_average_price_across_fills():
    qty, value = Bitkub._fills_from_info(info(fills=[(1000, 0, 2000000), (3000, 0, 3000000)]), "buy")
    assert qty == Decimal("0.0005") + Decimal("0.001")
    assert value / qty == pytest.approx(Decimal("2666666.6666666666"), rel=Decimal("1e-9"))
    sell_qty, sell_value = Bitkub._fills_from_info(info(fills=[(0.5, 0, 100), (0.5, 0, 200)]), "sell")
    assert sell_qty == Decimal("1.0") and sell_value == Decimal("150")


def test_malformed_orders_parse_to_none_not_a_crash():
    broker, _ = make_broker()
    assert broker._parse_broker_order(None, "strat") is None
    assert broker._parse_broker_order({"_symbol": "garbage"}, "strat") is None
    assert broker._parse_broker_order(open_row(side="sideways"), "strat") is None


# ------------------------------------------------------------------ querying orders
def test_all_orders_cover_held_watched_and_tracked_markets_and_tag_each_row():
    broker, client = make_broker()
    broker._held_symbols = {"ETH_THB"}
    broker._watched_symbols = {"BTC_THB"}
    client.open_orders = {"BTC_THB": [{"id": "1", "side": "buy"}], "ETH_THB": [{"id": "2", "side": "sell"}]}
    rows = broker._pull_broker_all_orders()
    assert {(r["id"], r["_symbol"]) for r in rows} == {("1", "BTC_THB"), ("2", "ETH_THB")}


def test_all_orders_skip_nonexistent_markets_but_raise_when_any_real_market_is_unreadable():
    broker, client = make_broker()
    broker._watched_symbols = {"BTC_THB", "GONE_THB"}
    client.open_orders_error = {"GONE_THB": BitkubAPIError("Invalid symbol", code=11)}
    assert broker._pull_broker_all_orders() == []
    client.open_orders_error = {"BTC_THB": BitkubAPIError("Server error", code=90)}
    with pytest.raises(LumibotBrokerAPIError):  # a partial list would make unread orders look vanished
        broker._pull_broker_all_orders()


def test_single_order_lookup_needs_and_uses_the_remembered_market_and_side():
    broker, client = make_broker()
    broker.submit_order(make_order("sell", "0.01", limit_price=3000000))
    client.order_info["111"] = info("111", "unfilled", amount="0.01")
    result = broker._pull_broker_order("111")
    assert ("get_order_info", ("BTC_THB", "111", "sell"), {}) in client.calls
    assert result["_symbol"] == "BTC_THB" and result["_side"] == "sell"
    assert broker._pull_broker_order("unknown-id") is None


# ------------------------------------------------------------------ polling: the part that must never lie about state
def test_polling_books_a_filled_market_order_that_left_the_open_list():
    broker, client = make_broker()
    broker.submit_order(make_order("buy", "0.0005", Order.OrderType.MARKET))
    client.order_info["111"] = info(
        "111", "filled", rate=2778954, amount="1389.47", fills=[(1389.47, 3.47, 2778954.76)]
    )
    broker.do_polling()  # the open list is empty: a filled order must be settled, NOT reported cancelled
    fills = broker.stream.of(broker.FILLED_ORDER)
    assert len(fills) == 1 and not broker.stream.of(broker.CANCELED_ORDER)
    assert fills[0]["filled_quantity"] == pytest.approx((1389.47 - 3.47) / 2778954.76)
    assert fills[0]["price"] == pytest.approx(2778954.76)


def test_partial_then_final_fill_dispatch_increments_not_totals():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "unfilled", partial=True, fills=[(1000, 2.5, 2500000)])
    broker.do_polling()
    first = broker.stream.of(broker.PARTIALLY_FILLED_ORDER)[0]
    assert first["filled_quantity"] == pytest.approx(0.000399)
    broker._process_trade_event(
        order, broker.PARTIALLY_FILLED_ORDER, price=first["price"], filled_quantity=first["filled_quantity"]
    )

    client.order_info["111"] = info("111", "filled", fills=[(1000, 2.5, 2500000), (1500, 3.75, 2500000)])
    broker.do_polling()
    final = broker.stream.of(broker.FILLED_ORDER)[0]
    assert final["filled_quantity"] == pytest.approx(
        0.0005985
    )  # only the new part, so add_transaction never double counts
    assert order.quantity == Decimal("0.0009975")  # the order now carries what really executed


def test_polling_is_idempotent_for_unchanged_partial_fills():
    broker, client = make_broker()
    order = broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "unfilled", partial=True, fills=[(1000, 2.5, 2500000)])
    broker.do_polling()
    event = broker.stream.of(broker.PARTIALLY_FILLED_ORDER)[0]
    broker._process_trade_event(
        order, broker.PARTIALLY_FILLED_ORDER, price=event["price"], filled_quantity=event["filled_quantity"]
    )
    broker.stream.events.clear()
    broker.do_polling()
    assert broker.stream.events == []


def test_an_order_missing_from_a_lagging_open_list_is_left_alone_not_cancelled():
    broker, client = make_broker()
    broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "unfilled")  # open list is empty (lag) but the order is alive
    broker.do_polling()
    assert broker.stream.events == []


def test_a_cancelled_order_with_prior_fills_books_them_before_the_cancel():
    broker, client = make_broker()
    broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "cancelled", partial=True, fills=[(1000, 2.5, 2500000)])
    broker.do_polling()
    kinds = [name for name, _ in broker.stream.events]
    assert kinds == [broker.PARTIALLY_FILLED_ORDER, broker.CANCELED_ORDER]


def test_an_unreadable_open_list_skips_the_cycle_without_touching_any_order():
    broker, client = make_broker()
    broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.open_orders_error = {"BTC_THB": BitkubAPIError("Server error", code=90)}
    broker.do_polling()
    assert broker.stream.events == [] and "get_order_info" not in client.names()


def test_orders_placed_outside_lumibot_are_discovered():
    broker, client = make_broker()
    broker._held_symbols = {"BTC_THB"}
    client.open_orders = {
        "BTC_THB": [open_row(id="424242", side="buy", rate="2000000", amount="2000", client_id="from-app")]
    }
    broker._process_trade_event = MagicMock()
    broker.do_polling()
    (order, event), _ = broker._process_trade_event.call_args
    assert event == broker.NEW_ORDER and order.identifier == "424242" and order.quantity == Decimal("0.001")
    assert broker._order_registry["424242"] == ("BTC_THB", "buy")


def test_a_failing_position_sync_does_not_stop_order_polling():
    broker, client = make_broker()
    broker.sync_positions.side_effect = LumibotBrokerAPIError("balances unreadable")
    broker.submit_order(make_order("buy", "0.001", limit_price=2500000))
    client.order_info["111"] = info("111", "filled", fills=[(2500, 6.25, 2500000)])
    broker.do_polling()
    assert broker.stream.of(broker.FILLED_ORDER)


def test_stream_registers_every_event_the_polling_loop_dispatches():
    broker, _ = make_broker()
    registered = []

    class Recorder:
        def add_action(self, name):
            def decorator(fn):
                registered.append(name)
                return fn

            return decorator

    broker.stream = Recorder()
    broker._register_stream_events()
    assert set(registered) == {
        broker.POLL_EVENT,
        broker.NEW_ORDER,
        broker.FILLED_ORDER,
        broker.PARTIALLY_FILLED_ORDER,
        broker.CANCELED_ORDER,
        broker.ERROR_ORDER,
    }


def test_account_value_history_is_not_available():
    assert make_broker()[0].get_historical_account_value() == {}


# ------------------------------------------------------------------ packaging
def test_public_exports_and_lazy_helper_import():
    from lumibot.brokers import Bitkub as ExportedBroker
    from lumibot.data_sources import BitkubData

    assert ExportedBroker is Bitkub and BitkubData.SOURCE == "BITKUB"
    code = (
        "import sys; import lumibot.tools.bitkub_helpers as h; "
        "print('requests' in sys.modules, h.BitkubClient.__name__)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.stdout.split()[-2:] == ["False", "BitkubClient"], result.stderr  # `requests` loads only on first use
