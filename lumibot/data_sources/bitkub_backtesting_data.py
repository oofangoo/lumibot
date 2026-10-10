"""Backtesting data source for Bitkub spot markets (THB and USDT quotes).

Downloads Bitkub candles on first use (chunked, cached on disk; see :mod:`lumibot.tools.bitkub_history`) and serves
them to a backtest the way LumiBot's other crypto exchange source (CCXT) does:

* Strategies only ever see **completed** candles. Bitkub stamps a candle with its *start* time, so a candle is visible
  once ``start + bar_duration <= now``; the whole-day close of a daily candle is never available at its own start.
  (Feeding
  start-stamped candles to ``PandasData`` would let a strategy read the candle's close at the instant it opens.)
* Orders fill on the candle that **starts now**: market orders at its open, limit orders against its range. The
  backtesting broker requests that candle explicitly for this source (see ``_execution_bar_request``).
* Nothing is filled in, interpolated or carried forward. A market with no candles for the requested window returns
  ``None``, and candles that never traded stay absent (Bitkub only has candles for periods with trades).

Fees: Bitkub charges 0.25% per side on a standard account. Pass them to ``backtest()`` explicitly, for example
``buy_trading_fees=[TradingFee(percent_fee=0.0025)]`` and the same for ``sell_trading_fees``; verify your own tier.
"""

from __future__ import annotations

from datetime import timedelta

from lumibot._lazy_imports import LazyLogger, LazyModule, lazy_class
from lumibot.data_sources.data_source_backtesting import DataSourceBacktesting

logger = LazyLogger(__name__)
TYPE_CHECKING = False
pd = LazyModule("pandas")
Asset = lazy_class("lumibot.entities", "Asset")

if TYPE_CHECKING:
    from lumibot.entities import Bars


def _bars_class():
    from lumibot.entities import Bars

    return Bars


def _quote_class():
    from lumibot.entities import Quote

    return Quote


def _history_class():
    from lumibot.tools.bitkub_history import BitkubHistory

    return BitkubHistory


class BitkubBacktestingData(DataSourceBacktesting):
    # The backtesting broker keys its crypto execution model (current-candle fills, completed-candle history) on SOURCE.
    SOURCE = "BITKUB"
    MIN_TIMESTEP = "minute"
    DEFAULT_QUOTE_SYMBOL = "THB"
    # The first representation is the Bitkub resolution that `_parse_source_timestep(..., reverse=True)` returns.
    TIMESTEP_MAPPING = [
        {"timestep": "minute", "representations": ["1", "1m"]},
        {"timestep": "5 minutes", "representations": ["5", "5m"]},
        {"timestep": "15 minutes", "representations": ["15", "15m"]},
        {"timestep": "hour", "representations": ["60", "1h"]},
        {"timestep": "4 hours", "representations": ["240", "4h"]},
        {"timestep": "day", "representations": ["1D", "1d"]},
    ]

    def __init__(
        self,
        datetime_start=None,
        datetime_end=None,
        *args,
        cache_dir=None,
        client=None,
        history=None,
        prebuffer_bars: int = 300,
        **kwargs,
    ):
        """
        Parameters
        ----------
        cache_dir : str | Path, optional
            Where downloaded candles are cached (default: ``<LumiBot cache folder>/bitkub``).
        client : BitkubClient, optional
            Inject a client (tests, shared rate limiter). A keyless public client is created by default.
        history : BitkubHistory, optional
            Inject a complete history provider (tests).
        prebuffer_bars : int
            Candles downloaded *before* the backtest start so strategies have look-back data on day one.
        """
        super().__init__(datetime_start, datetime_end, *args, **kwargs)
        self.name = "bitkub"
        self._prebuffer_bars = int(prebuffer_bars)
        self._history = history if history is not None else _history_class()(client=client, cache_dir=cache_dir)
        self._data_store: dict = {}

    # ------------------------------------------------------------------ symbols and timesteps
    def _sanitize_base_and_quote_asset(self, asset, quote=None):
        """Accept an Asset, an ``(asset, quote)`` tuple, ``"BTC/THB"``, ``"BTC_THB"`` or ``"BTC"`` (quote: THB)."""
        if isinstance(asset, tuple):
            asset, quote = asset
        if isinstance(asset, str):
            parts = asset.replace("/", "_").replace("-", "_").split("_")
            if len(parts) == 2 and quote is None:
                asset, quote = parts
            elif len(parts) == 1:
                asset = parts[0]
            else:
                raise ValueError(f"Bitkub symbols look like BTC_THB or BTC/THB, got {asset!r}")
        if not isinstance(asset, Asset):
            asset = Asset(symbol=str(asset), asset_type=Asset.AssetType.CRYPTO)
        if asset.asset_type != Asset.AssetType.CRYPTO:
            raise ValueError(f"Bitkub only trades spot crypto; got {asset.asset_type} for {asset.symbol}")
        if quote is None:
            quote = Asset(symbol=self.DEFAULT_QUOTE_SYMBOL, asset_type=Asset.AssetType.FOREX)
        elif not isinstance(quote, Asset):
            quote = Asset(symbol=str(quote), asset_type=Asset.AssetType.FOREX)
        return asset, quote

    @staticmethod
    def symbol_for(asset, quote) -> str:
        return f"{asset.symbol.upper()}_{quote.symbol.upper()}"

    def _bar_duration(self, timestep: str) -> timedelta:
        duration, _ = self.convert_timestep_str_to_timedelta(timestep)
        return duration

    # ------------------------------------------------------------------ data loading
    def _frame_for(self, symbol: str, timestep: str):
        """All completed candles for the backtest window (plus look-back), downloaded once and kept for the run."""
        resolution = self._parse_source_timestep(timestep, reverse=True)
        key = (symbol, resolution)
        if key not in self._data_store:
            duration = self._bar_duration(timestep)
            start = self.datetime_start - duration * self._prebuffer_bars
            frame = self._history.get(symbol, resolution, int(start.timestamp()), int(self.datetime_end.timestamp()))
            if frame is None or frame.empty:
                logger.error(
                    "Bitkub returned no %s candles for %s in the backtest window; check the symbol", resolution, symbol
                )
                self._data_store[key] = None
            else:
                frame = frame.copy()
                frame["price_change"] = frame["close"].pct_change()
                frame["dividend_yield"] = 0.0  # spot crypto pays none
                frame["return"] = frame["dividend_yield"] + frame["price_change"]
                frame.index = frame.index.tz_convert(self.tzinfo)
                self._data_store[key] = frame
        return self._data_store[key]

    def _slice(self, frame, timestep: str, length, timeshift):
        """Candles visible now: the finished ones, optionally shifted (a negative shift adds the current one)."""
        duration = self._bar_duration(timestep)
        end = self.to_default_timezone(self.get_datetime())
        if timeshift:
            end = end - (timeshift * duration if isinstance(timeshift, int) else timeshift)
        visible = frame[frame.index + duration <= end]
        return visible if length is None else visible.tail(length)

    # ------------------------------------------------------------------ DataSource API
    def get_historical_prices(
        self, asset, length, timestep=None, timeshift=None, quote=None, exchange=None, include_after_hours=True
    ) -> Bars | None:
        base, quote = self._sanitize_base_and_quote_asset(asset, quote)
        timestep = timestep or self.get_timestep()
        frame = self._frame_for(self.symbol_for(base, quote), timestep)
        if frame is None:
            return None
        window = self._slice(frame, timestep, length, timeshift)
        if window.empty:
            return None
        return _bars_class()(window, self.SOURCE, base, quote=quote, raw=window)

    def get_historical_prices_between_dates(
        self,
        asset,
        timestep="minute",
        quote=None,
        exchange=None,
        include_after_hours=True,
        start_date=None,
        end_date=None,
    ) -> Bars | None:
        """Completed candles inside ``[start_date, end_date]`` (used for the benchmark curve)."""
        base, quote = self._sanitize_base_and_quote_asset(asset, quote)
        frame = self._frame_for(self.symbol_for(base, quote), timestep)
        if frame is None:
            return None
        duration = self._bar_duration(timestep)
        window = frame[frame.index + duration <= self.to_default_timezone(end_date)] if end_date is not None else frame
        if start_date is not None:
            window = window[window.index >= self.to_default_timezone(start_date)]
        if window.empty:
            return None
        return _bars_class()(window, self.SOURCE, base, quote=quote, raw=window)

    def get_last_price(self, asset, timestep=None, quote=None, exchange=None, **kwargs):
        """Close of the newest completed candle (the price at the start of the current one)."""
        bars = self.get_historical_prices(asset, 1, timestep=timestep, quote=quote)
        if bars is None or bars.df.empty:
            return None
        return float(bars.df["close"].iloc[-1])

    def get_quote(self, asset, quote=None, exchange=None):
        """Last completed close only: candles carry no bid/ask, and none is invented."""
        base, quote = self._sanitize_base_and_quote_asset(asset, quote)
        return _quote_class()(asset=base, price=self.get_last_price(base, quote=quote))

    def get_chains(self, asset, quote=None, exchange=None, strike_count=100):
        raise NotImplementedError("Bitkub lists spot crypto only, so there are no option chains.")

    def get_strikes(self, asset):
        raise NotImplementedError("Bitkub lists spot crypto only, so there are no option strikes.")
