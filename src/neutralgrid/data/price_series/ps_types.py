"""Canonical data structures for the PriceSeries subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SeriesKind(Enum):
    """Discriminator for the type of price series being stored."""

    LAST_KLINE = "last_kline"
    MARK_KLINE = "mark_kline"
    MARK_TICK = "mark_tick"


@dataclass(frozen=True, slots=True)
class PriceTick:
    """A single mark-price snapshot from the WebSocket stream."""

    ts_ms: int
    mark_price: float
    index_price: float | None = None


@dataclass(frozen=True, slots=True)
class Candle:
    """A single OHLCV bar (kline).

    ``is_final`` is ``True`` when the bar's time window has closed.  Only
    final candles should be used for indicator computation or storage.
    """

    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    is_final: bool
