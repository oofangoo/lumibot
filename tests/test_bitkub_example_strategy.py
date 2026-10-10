"""The Bitkub example must be safe by default: it logs, and only trades when the smoke-test parameter is switched on."""

from types import SimpleNamespace

from lumibot.entities import Asset
from lumibot.example_strategies.bitkub_spot_example import BitkubSpotExample

THB = Asset("THB", Asset.AssetType.FOREX)


def _fake_self(place_test_order):
    orders, cancels, logs = [], [], []
    stub = SimpleNamespace(
        parameters={**BitkubSpotExample.parameters, "place_test_order": place_test_order},
        quote_asset=THB,
        test_order=None,
        get_last_price=lambda asset, quote=None: 2_800_000.0,
        get_cash=lambda: 1000.0,
        get_positions=lambda: [],
        log_message=lambda message, **kwargs: logs.append(message),
        create_order=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs, identifier="o1", status="new"),
        submit_order=lambda order: orders.append(order),
        cancel_order=lambda order: cancels.append(order),
    )
    return stub, orders, cancels, logs


def test_example_places_no_orders_by_default():
    stub, orders, cancels, logs = _fake_self(place_test_order=False)
    for _ in range(3):
        BitkubSpotExample.on_trading_iteration(stub)
    assert orders == [] and cancels == [] and any("last price" in line for line in logs)


def test_smoke_test_places_one_far_below_market_limit_order_then_cancels_it():
    stub, orders, cancels, _ = _fake_self(place_test_order=True)
    BitkubSpotExample.on_trading_iteration(stub)
    assert len(orders) == 1
    order = orders[0]
    asset, quantity, side = order.args
    assert side == "buy" and order.kwargs["order_type"] == "limit" and order.kwargs["quote"] == THB
    assert order.kwargs["limit_price"] == 1_400_000.0  # half the last price: it should rest, not fill
    assert abs(quantity * order.kwargs["limit_price"] - 20) < 1e-6  # about 20 THB

    BitkubSpotExample.on_trading_iteration(stub)  # next iteration cancels instead of placing another
    assert len(orders) == 1 and cancels == [order] and stub.test_order is None
