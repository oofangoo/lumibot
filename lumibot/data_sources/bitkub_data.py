"""Bitkub market data (spot crypto, THB and USDT markets).

Needs no API key: every call here uses Bitkub's public endpoints. Credentials may still be passed in ``config`` so the
data source can share one client with :class:`~lumibot.brokers.bitkub.Bitkub`.

Behaviours worth knowing (all verified against the live API, 2026-10-10):

* Candles come from ``/tradingview/history`` and only exist for periods that had trades, so illiquid markets return
  *sparse* minute bars. Missing periods are never filled in (no fabricated or forward-filled data).
* The endpoint can return **conflicting duplicate rows for one timestamp** (the same candle with different volume,
  high and close), even for a single-candle query. The most complete version has the largest volume and its close
  matches the next candle's open, so the row with the largest volume is kept.
* The still-forming candle is excluded: ``get_historical_prices`` returns completed bars only, matching how
  backtests see data. Use ``get_last_price`` / ``get_quote`` for the live price.
* Native bar sizes are 1m, 5m, 15m, 1h, 4h and 1d; other sizes raise ``ValueError`` instead of silently substituting.
"""

from __future__ import annotations

from lumibot._lazy_imports import LazyLogger, LazyModule, LazyPytzTimezoneRef, lazy_class
from lumibot.data_sources.data_source import DataSource

logger = LazyLogger(__name__)
TYPE_CHECKING = False
pd = LazyModule("pandas")
pytz = LazyModule("pytz")
Asset = lazy_class("lumibot.entities", "Asset")

if TYPE_CHECKING:
    from lumibot.entities import Bars, Quote


def _get_bars_class():
    from lumibot.entities import Bars

    return Bars


def _get_quote_class():
    from lumibot.entities import Quote

    return Quote


def _get_client_class():
    from lumibot.tools.bitkub_helpers import BitkubClient

    return BitkubClient


def _get_assets_mapping_class():
    from lumibot.entities import AssetsMapping

    return AssetsMapping


class BitkubData(DataSource):
    SOURCE = "BITKUB"
    DEFAULT_TIMEZONE = "UTC"
    DEFAULT_PYTZ = LazyPytzTimezoneRef(DEFAULT_TIMEZONE)
    MIN_TIMESTEP = "minute"
    DEFAULT_QUOTE_SYMBOL = "THB"

    # Lumibot timestep -> Bitkub resolution, with the seconds each bar spans.
    TIMESTEP_MAPPING = [
        {"timestep": "minute", "representations": ["1", "1m", "1min", "1minute", "minute"]},
        {"timestep": "5 minutes", "representations": ["5", "5m", "5min", "5minutes", "5minute"]},
        {"timestep": "15 minutes", "representations": ["15", "15m", "15min", "15minutes", "15minute"]},
        {"timestep": "hour", "representations": ["60", "60m", "60min", "1h", "1hour", "hour"]},
        {"timestep": "4 hours", "representations": ["240", "240m", "4h", "4hour", "4hours"]},
        {"timestep": "day", "representations": ["1d", "d", "1day", "day"]},
    ]
    RESOLUTION_BY_TIMESTEP = {
        "minute": "1",
        "5 minutes": "5",
        "15 minutes": "15",
        "hour": "60",
        "4 hours": "240",
        "day": "1D",
    }
    SECONDS_BY_RESOLUTION = {"1": 60, "5": 300, "15": 900, "60": 3600, "240": 14400, "1D": 86400}
    LOOKBACK_MULTIPLIERS = (1, 4, 16, 64)  # widen the window for sparse markets until enough candles exist

    def __init__(self, config=None, max_workers: int = 1, chunk_size: int = 100, tzinfo=None, client=None, **kwargs):
        super().__init__(delay=0, tzinfo=tzinfo, max_workers=max_workers)
        self.name = "bitkub"
        self.chunk_size = chunk_size
        if isinstance(config, dict):
            api_key, api_secret = config.get("API_KEY"), config.get("API_SECRET")
            base_url = config.get("BASE_URL")
        else:
            api_key, api_secret = getattr(config, "API_KEY", None), getattr(config, "API_SECRET", None)
            base_url = getattr(config, "BASE_URL", None)
        if client is not None:
            self.client = client
        else:
            options = {"base_url": base_url} if base_url else {}
            self.client = _get_client_class()(api_key, api_secret, **options)

    # ------------------------------------------------------------------ symbol handling
    def _sanitize_base_and_quote_asset(self, base_asset, quote_asset=None):
        """Accept an Asset, a ``(asset, quote)`` tuple or plain strings; default the quote to THB."""
        if isinstance(base_asset, tuple):
            asset, quote = base_asset
        else:
            asset, quote = base_asset, quote_asset
        if not isinstance(asset, Asset):
            asset = Asset(symbol=str(asset), asset_type=Asset.AssetType.CRYPTO)
        if asset.asset_type not in (Asset.AssetType.CRYPTO,):
            raise ValueError(f"Bitkub only trades spot crypto; got {asset.asset_type} for {asset.symbol}")
        if quote is None:
            quote = Asset(symbol=self.DEFAULT_QUOTE_SYMBOL, asset_type=Asset.AssetType.FOREX)
        elif not isinstance(quote, Asset):
            quote = Asset(symbol=str(quote), asset_type=Asset.AssetType.FOREX)
        return asset, quote

    @staticmethod
    def symbol_for(asset, quote) -> str:
        """``BTC`` + ``THB`` -> ``BTC_THB``."""
        return f"{asset.symbol.upper()}_{quote.symbol.upper()}"

    # ------------------------------------------------------------------ timesteps
    def get_timestep_from_string(self, timestep: str) -> str:
        """Normalise a timestep ("1d", "5m", "hour", ...) to the canonical LumiBot name; raises if unsupported."""
        compact = str(timestep).lower().replace(" ", "").strip()
        for mapping in self.TIMESTEP_MAPPING:
            if compact == mapping["timestep"].replace(" ", "") or compact in mapping["representations"]:
                return mapping["timestep"]
        supported = ", ".join(m["timestep"] for m in self.TIMESTEP_MAPPING)
        raise ValueError(f"Bitkub does not serve {timestep!r} candles; supported timesteps: {supported}")

    def _parse_source_timestep(self, timestep: str) -> str:
        """Lumibot timestep -> Bitkub ``resolution`` ("1", "5", "15", "60", "240", "1D")."""
        return self.RESOLUTION_BY_TIMESTEP[self.get_timestep_from_string(timestep)]

    def supports_native_timestep(self, timestep: str) -> bool:
        try:
            self.get_timestep_from_string(timestep)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ prices and quotes
    def _ticker_row(self, symbol: str):
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        try:
            rows = self.client.get_ticker(symbol)
        except BitkubAPIError as exc:
            logger.warning("Bitkub ticker for %s unavailable: %s", symbol, exc)
            return None
        if not isinstance(rows, list):
            return None
        return next((row for row in rows if str(row.get("symbol", "")).upper() == symbol), None)

    @staticmethod
    def _positive_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def get_last_price(self, asset, quote=None, exchange=None, **kwargs):
        asset, quote = self._sanitize_base_and_quote_asset(asset, quote)
        row = self._ticker_row(self.symbol_for(asset, quote))
        return self._positive_float(row.get("last")) if row else None

    def get_last_prices(self, assets, quote=None, exchange=None):
        """One request for the whole batch (the ticker endpoint returns every market)."""
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        result = {}
        try:
            rows = {str(r.get("symbol", "")).upper(): r for r in (self.client.get_ticker() or [])}
        except BitkubAPIError as exc:
            logger.warning("Bitkub batch ticker unavailable: %s", exc)
            rows = {}
        for asset in assets:
            base, quote_asset = self._sanitize_base_and_quote_asset(asset, quote)
            row = rows.get(self.symbol_for(base, quote_asset))
            result[asset] = self._positive_float(row.get("last")) if row else None
        return _get_assets_mapping_class()(result)

    def get_quote(self, asset, quote=None, exchange=None) -> Quote:
        """Best bid/ask, last price and 24h change. Fields are ``None`` when the market has no ticker."""
        asset, quote = self._sanitize_base_and_quote_asset(asset, quote)
        row = self._ticker_row(self.symbol_for(asset, quote))
        Quote = _get_quote_class()
        if not row:
            return Quote(asset=asset)
        return Quote(
            asset=asset,
            price=self._positive_float(row.get("last")),
            bid=self._positive_float(row.get("highest_bid")),
            ask=self._positive_float(row.get("lowest_ask")),
            volume=float(row["base_volume"]) if row.get("base_volume") is not None else None,
            percent_change=float(row["percent_change"]) if row.get("percent_change") is not None else None,
            raw_data=dict(row),
        )

    def get_orderbook(self, asset, quote=None, depth: int = 10) -> dict:
        """Order-book snapshot as ``{"bids": [(price, size), ...], "asks": [...]}`` (best first)."""
        asset, quote = self._sanitize_base_and_quote_asset(asset, quote)
        book = self.client.get_depth(self.symbol_for(asset, quote), depth)
        return {side: [(float(p), float(s)) for p, s in book.get(side, [])] for side in ("bids", "asks")}

    def get_symbols(self) -> list:
        """Every market with precision, minimum size, ``status`` and ``source`` (only ``exchange`` markets trade)."""
        return self.client.get_symbols()

    # ------------------------------------------------------------------ candles
    def _history_frame(self, data: dict, interval: int, end_ts: int):
        """Turn a /tradingview/history payload into a clean, completed-bars-only OHLCV frame (possibly empty)."""
        columns = ["open", "high", "low", "close", "volume"]
        if not isinstance(data, dict) or data.get("s") != "ok" or not data.get("t"):
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC"))
        frame = pd.DataFrame(
            {
                "t": [int(t) for t in data["t"]],
                "open": [float(v) for v in data["o"]],
                "high": [float(v) for v in data["h"]],
                "low": [float(v) for v in data["l"]],
                "close": [float(v) for v in data["c"]],
                "volume": [float(v) for v in data["v"]],
            }
        )
        frame = frame[
            (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        ]  # drop corrupt rows; never invent replacements
        # Conflicting duplicates: keep the most complete version of each candle (largest volume; the last one on ties).
        before = len(frame)
        frame = frame.sort_values(["t", "volume"], kind="stable").drop_duplicates("t", keep="last")
        if len(frame) < before:
            logger.debug(
                "Bitkub returned %s conflicting duplicate candle row(s); kept the highest-volume version",
                before - len(frame),
            )
        frame = frame[frame["t"] + interval <= end_ts]  # completed candles only
        frame.index = pd.to_datetime(frame["t"], unit="s", utc=True)
        return frame[columns]

    def get_historical_prices(
        self,
        asset,
        length: int,
        timestep: str = "",
        timeshift=None,
        quote=None,
        exchange: str = None,
        include_after_hours: bool = True,
    ) -> Bars | None:
        from lumibot.tools.bitkub_helpers import BitkubAPIError

        asset, quote = self._sanitize_base_and_quote_asset(asset, quote)
        symbol = self.symbol_for(asset, quote)
        resolution = self._parse_source_timestep(timestep or self.get_timestep())
        interval = self.SECONDS_BY_RESOLUTION[resolution]

        end = pd.Timestamp(self.get_datetime())
        if timeshift is not None:
            end = end - (pd.Timedelta(seconds=int(timeshift) * interval) if isinstance(timeshift, int) else timeshift)
        end_ts = int(end.timestamp())

        frame = None
        for multiplier in self.LOOKBACK_MULTIPLIERS:
            start_ts = end_ts - (int(length) + 2) * interval * multiplier
            try:
                data = self.client.get_tradingview_history(symbol, resolution, start_ts, end_ts)
            except BitkubAPIError as exc:
                logger.warning("Bitkub history for %s unavailable: %s", symbol, exc)
                return None
            frame = self._history_frame(data, interval, end_ts)
            if len(frame) >= length:
                break

        if frame is None or frame.empty:
            logger.warning("Bitkub returned no %s candles for %s", resolution, symbol)
            return None
        if len(frame) < length:
            logger.warning(
                "Bitkub has only %s of the %s requested %s candles for %s (sparse or newly listed market)",
                len(frame),
                length,
                resolution,
                symbol,
            )
        frame = frame.tail(length)
        frame.index = frame.index.tz_convert(self.tzinfo)
        return self._parse_source_symbol_bars(frame, asset, quote=quote, length=length)

    def _parse_source_symbol_bars(self, df, asset, quote=None, length: int = None) -> Bars:
        return _get_bars_class()(df, self.SOURCE, asset, raw=df, quote=quote)

    def get_chains(self, asset, quote=None, exchange: str = None, strike_count: int = 100) -> dict:
        """Bitkub lists spot crypto only, so there are no option chains."""
        return {"Multiplier": 1, "Exchange": exchange or "", "Chains": {}}
