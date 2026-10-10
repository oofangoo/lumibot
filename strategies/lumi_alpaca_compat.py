"""Compatibility wrapper around LumiBot's Alpaca broker (does not modify the vendored library).

Problem (found 2026-10-09): LumiBot's `Alpaca._parse_broker_order` raises
    ValueError: Order class is BRACKET but no child orders or secondary limit/stop prices have been provided
for any order in the account history whose order_class is 'bracket', because Alpaca's order-list response omits the legs unless
`nested=True` is requested. One old bracket order (IREN, 2026-10-08, placed outside this bot) made every LumiBot live run crash
inside sync_broker before any strategy code ran.

Fix: catch exactly that error, ACKNOWLEDGE the order (log it and remember it in `legacy_orders`), and re-parse it as a plain order so
its fills still appear in the history. Strategies never trade these; they only need to survive seeing them.
"""
from lumibot.brokers import Alpaca

try:
    from alpaca.trading.enums import OrderClass as _AlpacaOrderClass
except Exception:  # pragma: no cover
    _AlpacaOrderClass = None


class RobustAlpaca(Alpaca):
    legacy_orders: dict = {}

    def _parse_broker_order(self, response, strategy_name, strategy_object=None):
        try:
            return super()._parse_broker_order(response, strategy_name, strategy_object=strategy_object)
        except ValueError as e:
            if "BRACKET" not in str(e).upper() or _AlpacaOrderClass is None:
                raise
            oid = str(getattr(response, "id", ""))
            if oid not in RobustAlpaca.legacy_orders:
                RobustAlpaca.legacy_orders[oid] = {
                    "symbol": getattr(response, "symbol", None), "side": str(getattr(response, "side", "")),
                    "status": str(getattr(response, "status", "")), "created": str(getattr(response, "created_at", "")),
                }
                self.logger.warning(
                    f"Acknowledged legacy bracket order {oid[:8]} ({getattr(response, 'symbol', '?')}, "
                    f"{getattr(response, 'status', '?')}): legs not provided by the order list; treating it as a plain historical order."
                )
            simple = response.model_copy(update={"order_class": _AlpacaOrderClass.SIMPLE})
            return super()._parse_broker_order(simple, strategy_name, strategy_object=strategy_object)
