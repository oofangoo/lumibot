"""Read-only: status of the discretionary MSFT position (position, resting orders, price vs 50/200-day averages). Never prints keys, sends no orders."""
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
assert os.environ.get("ALPACA_IS_PAPER", "").lower() == "true", "refusing: not paper"
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

key, secret = os.environ["ALPACA_API_KEY"], os.environ["ALPACA_API_SECRET"]
t = TradingClient(key, secret, paper=True)
equity = float(t.get_account().equity)

pos = [p for p in t.get_all_positions() if p.symbol == "MSFT"]
if pos:
    p = pos[0]
    mv = float(p.market_value)
    print(f"MSFT qty={p.qty} avg={float(p.avg_entry_price):.2f} now={float(p.current_price):.2f} value={mv:,.0f} ({mv / equity * 100:.1f}% of equity {equity:,.0f}) "
          f"unrealized={float(p.unrealized_pl):+,.0f} ({float(p.unrealized_plpc) * 100:+.1f}%)")
else:
    print("MSFT: no position")

for o in t.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=["MSFT"])):
    print(f"open order: {o.side.value} {o.order_type.value} qty={o.qty} trail%={o.trail_percent} stop={o.stop_price} hwm={o.hwm} tif={o.time_in_force.value} id={o.client_order_id}")

bars = StockHistoricalDataClient(key, secret).get_stock_bars(
    StockBarsRequest(symbol_or_symbols="MSFT", timeframe=TimeFrame.Day, start=datetime.now(timezone.utc) - timedelta(days=400), feed="iex")).df
close = bars["close"].droplevel(0)
last, s50, s200 = float(close.iloc[-1]), float(close.tail(50).mean()), float(close.tail(200).mean())
print(f"last close {last:.2f} ({str(close.index[-1])[:10]})  SMA50 {s50:.2f} ({(last / s50 - 1) * 100:+.1f}%)  SMA200 {s200:.2f} ({(last / s200 - 1) * 100:+.1f}%)  "
      f"1y high {float(close.tail(252).max()):.2f}  trend {'ON' if last > s200 else 'OFF'}")
