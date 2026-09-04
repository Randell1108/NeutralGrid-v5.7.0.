"""REST-based gap repair and initial history backfill.

Uses the existing ``BinanceClient`` methods to fill gaps detected in the
candle stream.  Respects rate limits via the client's built-in tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from neutralgrid.data.price_series.ps_types import Candle, SeriesKind

if TYPE_CHECKING:
    from neutralgrid.api.binance_client import BinanceClient
    from neutralgrid.data.price_series.ps_store import PriceStore

logger = logging.getLogger(__name__)

# Binance klines endpoint returns at most 1500 bars per call
_MAX_BARS_PER_REQUEST = 1500

# Interval string → duration in milliseconds
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def interval_to_ms(interval: str) -> int:
    """Convert a Binance interval string to milliseconds."""
    ms = _INTERVAL_MS.get(interval)
    if ms is None:
        raise ValueError(f"Unsupported interval: {interval}")
    return ms


def _raw_kline_to_candle(raw: list, is_final: bool = True) -> Candle:
    """Convert a raw Binance kline list to a ``Candle``."""
    if len(raw) < 7:
        raise ValueError(f"Raw kline has {len(raw)} fields, expected >= 7")
    return Candle(
        open_time_ms=int(raw[0]),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        close_time_ms=int(raw[6]),
        is_final=is_final,
    )


async def backfill(
    client: BinanceClient,
    store: PriceStore,
    symbol: str,
    kind: SeriesKind,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> int:
    """Fetch candles from *start_ms* to *end_ms* and persist to *store*.

    Paginates transparently when the requested range exceeds a single
    API call.  Returns the number of candles written.
    """
    iv_ms = interval_to_ms(interval)
    total = 0
    cursor = start_ms

    while cursor < end_ms:
        page_limit = min(
            _MAX_BARS_PER_REQUEST,
            (end_ms - cursor) // iv_ms + 1,
        )
        if page_limit <= 0:
            break

        if kind == SeriesKind.MARK_KLINE:
            raw = await client.get_mark_price_klines(
                symbol, interval, limit=int(page_limit),
                start_time=cursor, end_time=end_ms,
            )
        else:
            raw = await client.get_klines(
                symbol, interval, limit=int(page_limit),
                start_time=cursor, end_time=end_ms,
            )

        if not raw:
            break

        for row in raw:
            candle = _raw_kline_to_candle(row, is_final=True)
            store.append_candle(symbol, kind, interval, candle)
            total += 1

        # Advance cursor past the last received bar
        last_open = int(raw[-1][0])
        next_cursor = last_open + iv_ms
        if next_cursor <= cursor:
            # Safety: avoid infinite loop on unexpected data
            break
        cursor = next_cursor

    logger.info(
        "Backfill %s %s %s: %d candles [%d → %d]",
        symbol, kind.value, interval, total, start_ms, end_ms,
    )
    return total


async def initial_backfill(
    client: BinanceClient,
    store: PriceStore,
    symbol: str,
    intervals: list[str],
    need_mark: bool = True,
    n_bars: int = 500,
) -> int:
    """Backfill initial history for a symbol across intervals.

    Always backfills ``LAST_KLINE``.  If ``need_mark`` is ``True``,
    also backfills ``MARK_KLINE`` for each interval.

    Returns total candle count across all interval/kind combos.
    """
    import time

    total = 0
    now_ms = int(time.time() * 1000)

    kinds = [SeriesKind.LAST_KLINE]
    if need_mark:
        kinds.append(SeriesKind.MARK_KLINE)

    for iv in intervals:
        iv_ms = interval_to_ms(iv)
        start_ms = now_ms - iv_ms * n_bars
        for kind in kinds:
            count = await backfill(
                client, store, symbol, kind, iv, start_ms, now_ms,
            )
            total += count

    return total
