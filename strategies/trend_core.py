"""Trend Core: hold a risk asset (default QQQ) while it trades above its 200-day average, else hold T-bills (SHV).

Why: in research/scan_daily.py (E002) this kind of rule cut drawdowns roughly in half versus buy-and-hold while
keeping most of the return. The idle-cash leg (SHV) earns T-bill interest instead of sitting at 0%.

Run:  .\\run.ps1 strategies\\trend_core.py
Backtest vs live is chosen by IS_BACKTESTING in .env (true = backtest).
"""
from datetime import datetime

from lumibot.strategies import Strategy


class TrendCore(Strategy):
    parameters = {"risk_symbol": "QQQ", "safe_symbol": "SHV", "ma_days": 200, "max_capital": None}

    def initialize(self):
        self.sleeptime = "1D"  # one decision per day
        self.last_target = None

    def on_trading_iteration(self):
        p = self.parameters
        bars = self.get_historical_prices(p["risk_symbol"], p["ma_days"] + 5, "day")
        if bars is None or len(bars.df) < p["ma_days"]:
            self.log_message("Not enough history yet; staying put.")
            return

        close = bars.df["close"]
        price, ma = float(close.iloc[-1]), float(close.tail(p["ma_days"]).mean())
        target = p["risk_symbol"] if price > ma else p["safe_symbol"]
        self.log_message(f"{p['risk_symbol']} {price:.2f} vs SMA{p['ma_days']} {ma:.2f} -> target {target}")

        # Only ever touch the two symbols this strategy owns; other positions in the account are not ours.
        managed = {p["risk_symbol"], p["safe_symbol"]}
        mine = [pos for pos in self.get_positions() if pos.symbol in managed and pos.quantity > 0]
        if {pos.symbol for pos in mine} == {target}:
            return  # already positioned

        for pos in mine:
            if pos.symbol != target:
                self.submit_order(self.create_order(pos.asset, pos.quantity, "sell"))

        # max_capital caps the money this strategy may use (live/paper); None = use all cash (backtests compound).
        capital = p.get("max_capital") or self.get_cash()
        last = self.get_last_price(target)
        qty = int((capital * 0.98) // last) if last else 0
        if qty > 0:
            self.submit_order(self.create_order(target, qty, "buy"))
        self.last_target = target


if __name__ == "__main__":
    from lumibot.credentials import IS_BACKTESTING

    if IS_BACKTESTING:
        from lumibot.backtesting import YahooDataBacktesting

        TrendCore.backtest(
            YahooDataBacktesting,
            backtesting_start=datetime(2018, 1, 1),
            backtesting_end=datetime(2026, 10, 1),
            benchmark_asset="SPY",
            budget=50000,
        )
    else:
        from lumibot.brokers import Alpaca
        from lumibot.credentials import ALPACA_CONFIG

        TrendCore(broker=Alpaca(ALPACA_CONFIG), parameters={**TrendCore.parameters, "max_capital": 50000}).run_live()
