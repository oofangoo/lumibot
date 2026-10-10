"""Read-only: list recent orders (all statuses) with their order_class, to see what LumiBot's broker sync is choking on."""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: not paper"
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

c = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)
for o in c.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100, nested=True)):
    legs = len(o.legs) if getattr(o, "legs", None) else 0
    print(f"{str(o.created_at)[:19]} {o.symbol:8} {o.side.value:4} {o.order_type.value:10} class={getattr(o.order_class, 'value', o.order_class)!s:8} "
          f"legs={legs} status={o.status.value:10} qty={o.qty}")
