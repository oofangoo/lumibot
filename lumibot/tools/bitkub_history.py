"""Chunked, cached download of completed Bitkub candles for backtesting.

``BitkubHistory.get(symbol, resolution, start_ts, end_ts)`` returns every *completed* candle that started inside the
requested window, as a DataFrame indexed by candle start time (UTC) with ``open, high, low, close, volume`` columns.

Integrity rules (a backtest is only as honest as its data):

* Only completed candles are ever stored; the still-forming candle is never cached.
* The cache records which time range has been *fetched*, because an absent candle is ambiguous: Bitkub only has candles
  for periods that had trades. A range is marked as covered only after every request for it succeeded, so a failed or
  interrupted download can never be mistaken for a complete history.
* Nothing is filled in, interpolated or carried forward. Corrupt rows are dropped and the exchange's conflicting
  duplicate candles are resolved by :func:`lumibot.tools.bitkub_helpers.clean_candles`.
* Re-fetching the newest cached candles on every extension picks up any late correction of the last stored candles.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from lumibot._lazy_imports import LazyLogger, LazyModule

from .bitkub_helpers import (
    CANDLE_COLUMNS,
    RESOLUTIONS,
    SECONDS_BY_RESOLUTION,
    BitkubClient,
    candles_frame,
    clean_candles,
)

logger = LazyLogger(__name__)
pd = LazyModule("pandas")

OHLCV = ["open", "high", "low", "close", "volume"]


class BitkubHistory:
    """Download Bitkub candles in bounded requests and cache them on disk."""

    MAX_BARS_PER_REQUEST = 20_000  # keeps each response around a megabyte
    REFETCH_BARS = 2  # newest cached candles re-requested whenever the cache is extended forward

    def __init__(self, client=None, cache_dir=None, clock=time.time):
        self.client = client if client is not None else BitkubClient()
        if cache_dir is None:
            from lumibot.constants import LUMIBOT_CACHE_FOLDER

            cache_dir = Path(LUMIBOT_CACHE_FOLDER) / "bitkub"
        self.cache_dir = Path(cache_dir)
        self._clock = clock
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cache files
    def _paths(self, symbol: str, resolution: str):
        stem = f"{symbol.upper()}_{resolution}"
        return self.cache_dir / f"{stem}.parquet", self.cache_dir / f"{stem}.json"

    def _load_cache(self, symbol: str, resolution: str):
        """Cached candles and the ``(from, to)`` range already fetched, or ``(None, None)`` if absent or unreadable."""
        data_path, meta_path = self._paths(symbol, resolution)
        if not (data_path.exists() and meta_path.exists()):
            return None, None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = pd.read_parquet(data_path)
            coverage = (int(meta["from"]), int(meta["to"]))
            if list(frame.columns) != CANDLE_COLUMNS:
                raise ValueError(f"unexpected columns {list(frame.columns)}")
            return frame, coverage
        except Exception as exc:  # a damaged cache is discarded, never trusted
            logger.warning(
                "Ignoring unreadable Bitkub cache for %s %s (%s); it will be downloaded again", symbol, resolution, exc
            )
            return None, None

    def _save_cache(self, symbol: str, resolution: str, frame, coverage) -> None:
        """Write data first, then the coverage record, each atomically; coverage is the commit point."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data_path, meta_path = self._paths(symbol, resolution)
        tmp_data, tmp_meta = data_path.with_suffix(".parquet.tmp"), meta_path.with_suffix(".json.tmp")
        frame.to_parquet(tmp_data, index=False)
        os.replace(tmp_data, data_path)
        tmp_meta.write_text(json.dumps({"from": int(coverage[0]), "to": int(coverage[1])}), encoding="utf-8")
        os.replace(tmp_meta, meta_path)

    # ------------------------------------------------------------------ download
    def _download(self, symbol: str, resolution: str, start_ts: int, end_ts: int):
        """All candles that started in ``[start_ts, end_ts)``; raises on ANY failed request (never partial)."""
        interval = SECONDS_BY_RESOLUTION[resolution]
        window = self.MAX_BARS_PER_REQUEST * interval
        frames, cursor = [], start_ts
        while cursor < end_ts:
            window_end = min(cursor + window, end_ts)
            # `to` is inclusive; end_ts - 1 keeps a candle that starts at end_ts out of this request
            payload = self.client.get_tradingview_history(symbol, resolution, cursor, window_end - 1)
            frames.append(candles_frame(payload))
            cursor = window_end
        if not frames:
            return pd.DataFrame(columns=CANDLE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ public API
    def get(self, symbol: str, resolution: str, start_ts: int, end_ts: int):
        """Completed candles that start at/after ``start_ts`` and end by ``end_ts`` (UTC-indexed OHLCV frame)."""
        resolution = str(resolution)
        if resolution not in RESOLUTIONS:
            raise ValueError(f"Bitkub resolution must be one of {RESOLUTIONS}, got {resolution!r}")
        symbol = symbol.upper()
        interval = SECONDS_BY_RESOLUTION[resolution]
        last_completed_end = int(self._clock() // interval) * interval  # end of the newest finished candle
        want_start = (int(start_ts) // interval) * interval
        want_end = min(int(end_ts) // interval * interval, last_completed_end)

        with self._lock:
            cached, coverage = self._load_cache(symbol, resolution)
            if want_end > want_start:
                fetched = []
                if coverage is None:
                    fetched.append(self._download(symbol, resolution, want_start, want_end))
                    new_coverage = (want_start, want_end)
                else:
                    cov_from, cov_to = coverage
                    new_from, new_to = min(want_start, cov_from), max(want_end, cov_to)
                    if want_start < cov_from:
                        fetched.append(self._download(symbol, resolution, want_start, cov_from))
                    if want_end > cov_to:
                        resume = max(cov_to - self.REFETCH_BARS * interval, new_from)
                        fetched.append(self._download(symbol, resolution, resume, want_end))
                    new_coverage = (new_from, new_to)
                if fetched:
                    parts = ([cached] if cached is not None and len(cached) else []) + fetched
                    merged = (
                        clean_candles(pd.concat(parts, ignore_index=True))
                        if parts
                        else pd.DataFrame(columns=CANDLE_COLUMNS)
                    )
                    merged = merged[merged["t"] + interval <= last_completed_end]  # never store a forming candle
                    self._save_cache(symbol, resolution, merged, new_coverage)
                    cached = merged
            if cached is None or cached.empty:
                return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([], tz="UTC", name="datetime"))

            window = cached[(cached["t"] >= want_start) & (cached["t"] + interval <= want_end)].copy()
            window.index = pd.DatetimeIndex(pd.to_datetime(window["t"], unit="s", utc=True), name="datetime")
            return window[OHLCV]
