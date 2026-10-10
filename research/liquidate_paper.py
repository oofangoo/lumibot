"""PAPER ONLY: cancel open orders and close the three pre-existing positions the user asked to sell
(BTCUSD, IREN, SPY). Authorized by the user on 2026-10-09. Refuses to run unless ALPACA_IS_PAPER=true."""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: not paper"

from alpaca.trading.client import TradingClient

c = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)
for o in c.get_orders():
    print(f"open order: {o.symbol} {o.side} qty={o.qty} type={o.order_type} status={o.status}")
    c.cancel_order_by_id(o.id)
    print("  cancelled")
for sym in ("BTCUSD", "IREN", "SPY"):
    try:
        r = c.close_position(sym)
        print(f"close {sym}: submitted order {r.side} qty={r.qty} status={r.status}")
    except Exception as e:
        print(f"close {sym}: {type(e).__name__}: {e}")
