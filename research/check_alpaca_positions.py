"""Read-only: list positions and open orders in the Alpaca PAPER account. Never prints keys."""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: not paper"

from alpaca.trading.client import TradingClient

c = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"], paper=True)
for p in c.get_all_positions():
    print(f"{p.symbol:8} qty={p.qty:>10} mkt_value={float(p.market_value):>12,.2f} unreal_pl={float(p.unrealized_pl):>10,.2f}")
print("open orders:", len(c.get_orders()))
