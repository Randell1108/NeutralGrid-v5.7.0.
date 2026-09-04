"""WebSocket streaming for real-time mark-price ticks and kline updates.

Connects to the Binance Futures combined stream endpoint via ``aiohttp``
and writes every message into the ``PriceStore``.  Reconnects
automatically with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import aiohttp

from neutralgrid.core.config import get_config
from neutralgrid.core.exceptions import PriceSeriesError
from neutralgrid.data.price_series.ps_types import Candle, PriceTick, SeriesKind

if TYPE_CHECKING:
    from neutralgrid.data.price_series.ps_store import PriceStore

logger = logging.getLogger(__name__)

# Binance terminates WS connections after 24 h; refresh proactively.
_WS_MAX_LIFETIME_S = 23 * 3600  # 23 h to leave margin


class PriceWebSocket:
    """WebSocket client for Binance Futures combined streams.

    Subscribes to:
    - ``<symbol>@markPrice@1s``  — mark-price tick every 1 s
    - ``<symbol>@kline_<interval>`` — kline updates with ``is_final``
    """

    def __init__(self, store: PriceStore) -> None:
        cfg = get_config().price_series
        self._store = store
        self._base_url = cfg.ws_base_url
        self._reconnect_base = cfg.ws_reconnect_base_s
        self._reconnect_max = cfg.ws_reconnect_max_s
        self._ping_interval = cfg.ws_ping_interval_s
        self._gap_factor = cfg.gap_threshold_factor

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._running = False

        # Current subscriptions: set of stream names
        self._streams: set[str] = set()
        # Track last kline open_time per (symbol, interval) for gap detection
        self._last_kline_ot: dict[tuple[str, str], int] = {}
        # Callback for gap detection (set by manager)
        self._on_gap: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        symbols: list[str],
        intervals: list[str],
        need_mark: bool = True,
    ) -> None:
        """Build stream list, connect, and start the receive loop."""
        streams: list[str] = []
        for sym in symbols:
            s = sym.lower()
            for iv in intervals:
                streams.append(f"{s}@kline_{iv}")
            if need_mark:
                streams.append(f"{s}@markPrice@1s")
        self._streams = set(streams)

        if not self._streams:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("PriceWebSocket stopped")

    async def add_streams(
        self,
        symbols: list[str],
        intervals: list[str],
        need_mark: bool = True,
    ) -> None:
        """Subscribe to additional streams on an existing connection."""
        new: list[str] = []
        for sym in symbols:
            s = sym.lower()
            for iv in intervals:
                name = f"{s}@kline_{iv}"
                if name not in self._streams:
                    new.append(name)
                    self._streams.add(name)
            if need_mark:
                name = f"{s}@markPrice@1s"
                if name not in self._streams:
                    new.append(name)
                    self._streams.add(name)

        if new and self._ws and not self._ws.closed:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": new,
                "id": int(time.time()),
            }
            await self._ws.send_json(subscribe_msg)
            logger.info("Subscribed to %d new streams", len(new))

    # ------------------------------------------------------------------
    # Connection loop with reconnect
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Outer loop: connect → receive → reconnect on failure."""
        backoff = self._reconnect_base
        while self._running:
            try:
                await self._connect_and_receive()
            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                logger.exception(
                    "WS connection lost, reconnecting in %.1fs", backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max)
            else:
                # Clean exit (e.g. 24h refresh) — reset backoff
                backoff = self._reconnect_base

    async def _connect_and_receive(self) -> None:
        """Single connection lifecycle: connect → subscribe → read."""
        stream_list = "/".join(sorted(self._streams))
        url = f"{self._base_url}/stream?streams={stream_list}"

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(
                url, heartbeat=self._ping_interval,
            )
            logger.info("WS connected (%d streams)", len(self._streams))

            # Start ping keep-alive
            self._ping_task = asyncio.create_task(self._ping_loop())
            connect_time = time.monotonic()

            async for msg in self._ws:
                if not self._running:
                    return

                # 24 h refresh
                if time.monotonic() - connect_time > _WS_MAX_LIFETIME_S:
                    logger.info("24h WS refresh, reconnecting")
                    return  # will reconnect via outer loop

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_message(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    raise PriceSeriesError(
                        f"WS closed/error: {self._ws.close_code}"
                    )
        finally:
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                await self._session.close()

    async def _ping_loop(self) -> None:
        """Send pings at the configured interval to keep the connection alive."""
        try:
            while self._running and self._ws and not self._ws.closed:
                await asyncio.sleep(self._ping_interval)
                if self._ws and not self._ws.closed:
                    await self._ws.ping()
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, raw: str) -> None:
        """Route a single WS message to the correct handler."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed WS message: %.200s", raw)
            return

        # Combined stream envelope: {"stream": "...", "data": {...}}
        payload = data.get("data", data)
        event = payload.get("e")

        if event == "kline":
            self._handle_kline(payload)
        elif event == "markPriceUpdate":
            self._handle_mark_price(payload)
        # Ignore subscription confirmations / pongs

    def _handle_kline(self, payload: dict) -> None:
        """Process a kline WS event and write to the store."""
        k = payload.get("k", {})
        symbol = str(k.get("s", "")).upper()
        interval = str(k.get("i", ""))
        is_final = bool(k.get("x", False))

        candle = Candle(
            open_time_ms=int(k.get("t", 0)),
            open=float(k.get("o", 0)),
            high=float(k.get("h", 0)),
            low=float(k.get("l", 0)),
            close=float(k.get("c", 0)),
            volume=float(k.get("v", 0)),
            close_time_ms=int(k.get("T", 0)),
            is_final=is_final,
        )

        if (
            candle.open <= 0
            or candle.high <= 0
            or candle.low <= 0
            or candle.close <= 0
            or candle.high < max(candle.open, candle.close)
            or candle.low > min(candle.open, candle.close)
            or candle.volume < 0
        ):
            logger.warning(
                "Invalid kline for %s %s: O=%.8g H=%.8g L=%.8g C=%.8g V=%.8g",
                symbol, interval,
                candle.open, candle.high, candle.low, candle.close, candle.volume,
            )
            return

        self._store.append_candle(symbol, SeriesKind.LAST_KLINE, interval, candle)

        # Gap detection on final candles
        if is_final:
            gap_key = (symbol, interval)
            prev_ot = self._last_kline_ot.get(gap_key)
            if prev_ot is not None:
                from neutralgrid.data.price_series.ps_rest_backfill import (
                    interval_to_ms,
                )
                iv_ms = interval_to_ms(interval)
                expected = prev_ot + iv_ms
                if candle.open_time_ms - prev_ot > iv_ms * self._gap_factor:
                    logger.warning(
                        "Gap detected %s %s: expected %d, got %d",
                        symbol, interval, expected, candle.open_time_ms,
                    )
                    if self._on_gap is not None:
                        self._on_gap(symbol, interval, expected, candle.open_time_ms)
            self._last_kline_ot[gap_key] = candle.open_time_ms

    def _handle_mark_price(self, payload: dict) -> None:
        """Process a markPrice WS event and write to the store."""
        symbol = str(payload.get("s", "")).upper()
        mark_price = float(payload.get("p", 0))
        if mark_price <= 0:
            logger.warning("Invalid mark price for %s: %.8g", symbol, mark_price)
            return
        tick = PriceTick(
            ts_ms=int(payload.get("E", 0)),
            mark_price=mark_price,
            index_price=float(payload.get("i", 0)) if payload.get("i") else None,
        )
        self._store.append_tick(symbol, tick)
