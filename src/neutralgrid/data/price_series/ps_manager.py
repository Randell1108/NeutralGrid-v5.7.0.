"""Top-level orchestrator for the PriceSeries subsystem.

``PriceSeriesManager`` is the single entry-point that downstream code
(trigger logic, indicator computation) should use.  It coordinates the
store, WebSocket stream, and REST backfill behind a simple API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from neutralgrid.core.config import get_config
from neutralgrid.data.price_series.ps_quality import QualityReport, validate_candle_series
from neutralgrid.data.price_series.ps_rest_backfill import (
    backfill,
    initial_backfill,
    interval_to_ms,
)
from neutralgrid.data.price_series.ps_store import PriceStore
from neutralgrid.data.price_series.ps_types import Candle, SeriesKind
from neutralgrid.data.price_series.ps_ws_stream import PriceWebSocket

if TYPE_CHECKING:
    from neutralgrid.api.binance_client import BinanceClient

logger = logging.getLogger(__name__)


class PriceSeriesManager:
    """High-level price data orchestrator.

    Lifecycle::

        manager = PriceSeriesManager(client)
        await manager.start()
        await manager.ensure_symbol("BTCUSDT")
        candles = manager.get_candles("BTCUSDT", SeriesKind.LAST_KLINE, "1m")
        await manager.stop()
    """

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._cfg = get_config().price_series
        self._store = PriceStore()
        self._ws = PriceWebSocket(self._store)
        self._subscribed: set[str] = set()
        self._flush_task: asyncio.Task[None] | None = None
        self._backfill_tasks: set[asyncio.Task[int]] = set()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize store and start periodic flush task."""
        if self._started:
            return
        self._started = True
        # Register gap callback so the WS stream triggers backfill
        self._ws._on_gap = self._on_gap_detected
        # Start periodic disk flush (every 60 s)
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("PriceSeriesManager started")

    async def stop(self) -> None:
        """Flush store, close WS, clean up tasks."""
        self._started = False
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Cancel outstanding gap-backfill tasks
        for t in self._backfill_tasks:
            if not t.done():
                t.cancel()
        if self._backfill_tasks:
            await asyncio.gather(*self._backfill_tasks, return_exceptions=True)
            self._backfill_tasks.clear()
        await self._ws.stop()
        self._store.flush_to_disk()
        logger.info("PriceSeriesManager stopped")

    # ------------------------------------------------------------------
    # Symbol management
    # ------------------------------------------------------------------

    async def ensure_symbol(
        self,
        symbol: str,
        intervals: list[str] | None = None,
        need_mark: bool = True,
    ) -> None:
        """Ensure *symbol* is streaming and backfilled.

        Safe to call multiple times — duplicate subscriptions are skipped.
        """
        sym = symbol.upper()
        if intervals is None:
            intervals = ["1m"]

        if sym not in self._subscribed:
            # Initial REST backfill
            await initial_backfill(
                self._client,
                self._store,
                sym,
                intervals,
                need_mark=need_mark,
                n_bars=self._cfg.backfill_bars,
            )
            self._subscribed.add(sym)

        # Subscribe WS streams (idempotent via add_streams)
        if not self._ws._task or self._ws._task.done():
            await self._ws.start([sym], intervals, need_mark)
        else:
            await self._ws.add_streams([sym], intervals, need_mark)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_latest_mark(self, symbol: str) -> tuple[int, float] | None:
        """Return ``(timestamp_ms, mark_price)`` or ``None``."""
        tick = self._store.get_latest_tick(symbol.upper())
        if tick is not None:
            return (tick.ts_ms, tick.mark_price)
        return None

    def get_latest_last(self, symbol: str) -> tuple[int, float] | None:
        """Return ``(timestamp_ms, last_price)`` from the most recent kline."""
        candles = self._store.get_candles(
            symbol.upper(), SeriesKind.LAST_KLINE, "1m",
            limit=1, closed_only=False,
        )
        if candles:
            c = candles[-1]
            return (c.close_time_ms, c.close)
        return None

    def get_candles(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        closed_only: bool = True,
        limit: int = 500,
    ) -> list[Candle]:
        """Read candles from the in-memory ring buffer."""
        return self._store.get_candles(
            symbol.upper(), kind, interval,
            limit=limit, closed_only=closed_only,
        )

    def get_quality_report(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
    ) -> QualityReport:
        """Generate a quality report for the current candle buffer."""
        candles = self._store.get_candles(
            symbol.upper(), kind, interval, closed_only=True,
        )
        iv_ms = interval_to_ms(interval)
        return validate_candle_series(candles, iv_ms, self._cfg.gap_threshold_factor)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_gap_detected(
        self,
        symbol: str,
        interval: str,
        gap_start_ms: int,
        gap_end_ms: int,
    ) -> None:
        """Schedule backfill tasks when the WS stream detects a gap."""
        logger.info(
            "Scheduling gap backfill %s %s [%d -> %d]",
            symbol, interval, gap_start_ms, gap_end_ms,
        )
        # Backfill LAST_KLINE; also MARK_KLINE if the symbol was
        # subscribed with need_mark (mark data has the same gap).
        kinds = [SeriesKind.LAST_KLINE, SeriesKind.MARK_KLINE]
        for kind in kinds:
            task = asyncio.create_task(
                self._safe_backfill(symbol, kind, interval, gap_start_ms, gap_end_ms),
            )
            self._backfill_tasks.add(task)
            task.add_done_callback(self._backfill_tasks.discard)

    async def _safe_backfill(
        self,
        symbol: str,
        kind: SeriesKind,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> int:
        """Run backfill with exception logging so errors are never lost."""
        try:
            return await backfill(
                self._client, self._store, symbol, kind, interval,
                start_ms, end_ms,
            )
        except Exception:
            logger.exception(
                "Gap backfill failed %s %s %s", symbol, kind.value, interval,
            )
            return 0

    async def _periodic_flush(self) -> None:
        """Flush the store to disk every 60 seconds."""
        try:
            while self._started:
                await asyncio.sleep(60)
                self._store.flush_to_disk()
        except asyncio.CancelledError:
            pass
