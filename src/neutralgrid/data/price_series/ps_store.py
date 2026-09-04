"""Dual-layer storage: in-memory ring buffer + append-only on-disk Parquet.

The ring buffer provides fast O(1) append for real-time queries.
Parquet files provide durable, reproducible audit trail of finalized candles.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from pathlib import Path
from typing import Deque

import pyarrow as pa
import pyarrow.parquet as pq

from neutralgrid.core.config import get_config
from neutralgrid.data.price_series.ps_types import Candle, PriceTick, SeriesKind

logger = logging.getLogger(__name__)

# Parquet schema for candle files
_CANDLE_SCHEMA = pa.schema([
    ("open_time_ms", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("close_time_ms", pa.int64()),
    ("is_final", pa.bool_()),
])

# Type alias for the composite key used to index ring buffers
_CandleKey = tuple[str, SeriesKind, str]  # (symbol, kind, interval)


class PriceStore:
    """In-memory ring buffer + on-disk Parquet store for price data.

    Parameters
    ----------
    ring_buffer_size:
        Maximum number of entries per key in the ring buffer.
    store_dir:
        Root directory for on-disk Parquet files.  Resolved relative to
        the project base directory.
    """

    def __init__(
        self,
        ring_buffer_size: int | None = None,
        store_dir: str | Path | None = None,
    ) -> None:
        cfg = get_config()
        ps = cfg.price_series
        self._buf_size = ring_buffer_size or ps.ring_buffer_size
        self._store_root = cfg.resolve_path(str(store_dir or ps.store_dir))
        self._store_root.mkdir(parents=True, exist_ok=True)

        # Ring buffers keyed by (symbol, kind, interval)
        self._candle_bufs: dict[_CandleKey, Deque[Candle]] = {}
        # Tick buffers keyed by symbol
        self._tick_bufs: dict[str, Deque[PriceTick]] = {}
        # Pending candles for disk flush, keyed by (symbol, kind, interval)
        self._pending_candles: dict[_CandleKey, list[Candle]] = {}

    # ------------------------------------------------------------------
    # Ring buffer helpers
    # ------------------------------------------------------------------

    def _candle_buf(self, key: _CandleKey) -> Deque[Candle]:
        if key not in self._candle_bufs:
            self._candle_bufs[key] = deque(maxlen=self._buf_size)
        return self._candle_bufs[key]

    def _tick_buf(self, symbol: str) -> Deque[PriceTick]:
        if symbol not in self._tick_bufs:
            self._tick_bufs[symbol] = deque(maxlen=self._buf_size)
        return self._tick_bufs[symbol]

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_candle(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        candle: Candle,
    ) -> None:
        """Append a candle to the ring buffer.

        If ``candle.is_final`` the candle is also queued for the next
        ``flush_to_disk`` call.
        """
        key: _CandleKey = (symbol.upper(), kind, interval)
        buf = self._candle_buf(key)

        # If the latest entry has the same open_time_ms, replace it (update)
        if buf and buf[-1].open_time_ms == candle.open_time_ms:
            buf.pop()
            buf.append(candle)
        else:
            buf.append(candle)

        if candle.is_final:
            self._pending_candles.setdefault(key, []).append(candle)

    def append_tick(self, symbol: str, tick: PriceTick) -> None:
        """Append a mark-price tick to the ring buffer (memory only)."""
        self._tick_buf(symbol.upper()).append(tick)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        limit: int = 500,
        closed_only: bool = True,
    ) -> list[Candle]:
        """Read the latest *limit* candles from the ring buffer."""
        key: _CandleKey = (symbol.upper(), kind, interval)
        buf = self._candle_bufs.get(key)
        if buf is None:
            return []
        candles = list(buf)
        if closed_only:
            candles = [c for c in candles if c.is_final]
        if limit and len(candles) > limit:
            candles = candles[-limit:]
        return candles

    def get_latest_tick(self, symbol: str) -> PriceTick | None:
        """Return the most recent mark-price tick, or ``None``."""
        buf = self._tick_bufs.get(symbol.upper())
        if buf:
            return buf[-1]
        return None

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def flush_to_disk(self) -> int:
        """Write all pending finalized candles to daily Parquet files.

        Returns the number of candles flushed.
        """
        total = 0
        for key, candles in self._pending_candles.items():
            if not candles:
                continue
            symbol, kind, interval = key
            # Group candles by UTC date
            by_date: dict[str, list[Candle]] = {}
            for c in candles:
                day = dt.datetime.fromtimestamp(
                    c.open_time_ms / 1000, tz=dt.timezone.utc,
                ).strftime("%Y-%m-%d")
                by_date.setdefault(day, []).append(c)

            for day, day_candles in by_date.items():
                self._write_day(symbol, kind, interval, day, day_candles)
                total += len(day_candles)

        self._pending_candles.clear()
        if total:
            logger.debug("Flushed %d candles to disk", total)
        return total

    def _write_day(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        day: str,
        candles: list[Candle],
    ) -> None:
        dir_path = self._store_root / symbol.upper() / kind.value / interval
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"{day}.parquet"

        # Build table from new candles
        new_table = pa.table(
            {
                "open_time_ms": [c.open_time_ms for c in candles],
                "open": [c.open for c in candles],
                "high": [c.high for c in candles],
                "low": [c.low for c in candles],
                "close": [c.close for c in candles],
                "volume": [c.volume for c in candles],
                "close_time_ms": [c.close_time_ms for c in candles],
                "is_final": [c.is_final for c in candles],
            },
            schema=_CANDLE_SCHEMA,
        )

        # Append to existing file if present
        if path.exists():
            existing = pq.read_table(path, schema=_CANDLE_SCHEMA)
            merged = pa.concat_tables([existing, new_table])
            # Dedupe by open_time_ms (keep last)
            ot = merged.column("open_time_ms").to_pylist()
            seen: dict[int, int] = {}
            for idx, t in enumerate(ot):
                seen[t] = idx
            keep = sorted(seen.values())
            merged = merged.take(keep)
        else:
            merged = new_table

        merged = merged.sort_by([("open_time_ms", "ascending")])

        pq.write_table(merged, path)

    # ------------------------------------------------------------------
    # Disk recovery
    # ------------------------------------------------------------------

    def load_from_disk(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Candle]:
        """Load candles from Parquet files into a list.

        Parameters
        ----------
        start_date, end_date:
            ISO date strings (``YYYY-MM-DD``).  If ``None``, all
            available files are loaded.
        """
        dir_path = self._store_root / symbol.upper() / kind.value / interval
        if not dir_path.exists():
            return []

        files = sorted(dir_path.glob("*.parquet"))
        if start_date:
            files = [f for f in files if f.stem >= start_date]
        if end_date:
            files = [f for f in files if f.stem <= end_date]

        candles: list[Candle] = []
        for fp in files:
            table = pq.read_table(fp, schema=_CANDLE_SCHEMA)
            for i in range(table.num_rows):
                candles.append(Candle(
                    open_time_ms=table.column("open_time_ms")[i].as_py(),
                    open=table.column("open")[i].as_py(),
                    high=table.column("high")[i].as_py(),
                    low=table.column("low")[i].as_py(),
                    close=table.column("close")[i].as_py(),
                    volume=table.column("volume")[i].as_py(),
                    close_time_ms=table.column("close_time_ms")[i].as_py(),
                    is_final=table.column("is_final")[i].as_py(),
                ))
        return candles
