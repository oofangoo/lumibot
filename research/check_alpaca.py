"""Read-only check that the Alpaca keys in .env work and point at the PAPER account. Never prints keys."""
import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
key, secret = os.environ.get("ALPACA_API_KEY", ""), os.environ.get("ALPACA_API_SECRET", "")
paper = os.environ.get("ALPACA_IS_PAPER", "").lower() == "true"
print(f"key set: {bool(key)}  secret set: {bool(secret)}  ALPACA_IS_PAPER={paper}")
if not (key and secret):
    raise SystemExit("keys missing in .env")
if not paper:
    raise SystemExit("ALPACA_IS_PAPER is not true - refusing to connect")

from alpaca.trading.client import TradingClient

acct = TradingClient(key, secret, paper=True).get_account()
print(f"status={acct.status} equity={acct.equity} cash={acct.cash} buying_power={acct.buying_power} "
      f"pdt={acct.pattern_day_trader} trading_blocked={acct.trading_blocked}")
