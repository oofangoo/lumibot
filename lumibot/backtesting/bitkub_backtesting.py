from lumibot.data_sources.bitkub_backtesting_data import BitkubBacktestingData


class BitkubBacktesting(BitkubBacktestingData):
    """Backtest on real Bitkub candles: pass this class to ``Strategy.backtest()``."""
