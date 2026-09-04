"""Data-quality gates for PriceSeries candle and tick streams.

All functions are stateless and side-effect free so they are easy to
unit-test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from neutralgrid.data.price_series.ps_types import Candle, PriceTick


# ------------------------------------------------------------------
# Quality report
# ------------------------------------------------------------------

@dataclass
class QualityReport:
    """Summary of quality checks on a candle series."""

    is_monotonic: bool = True
    has_gaps: bool = False
    gap_ranges: list[tuple[int, int]] = field(default_factory=list)
    duplicate_count: int = 0


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def validate_candle_series(
    candles: Sequence[Candle],
    interval_ms: int,
    threshold_factor: float = 1.5,
) -> QualityReport:
    """Run all quality checks on *candles* and return a single report."""
    report = QualityReport()
    if len(candles) < 2:
        return report

    report.is_monotonic = check_candle_monotonic(candles)
    report.duplicate_count = _count_duplicates(candles)
    gaps = detect_gaps(candles, interval_ms, threshold_factor)
    report.gap_ranges = gaps
    report.has_gaps = len(gaps) > 0
    return report


def detect_gaps(
    candles: Sequence[Candle],
    interval_ms: int,
    threshold_factor: float = 1.5,
) -> list[tuple[int, int]]:
    """Return ``(gap_start_ms, gap_end_ms)`` for each detected gap.

    A gap is detected when the distance between consecutive
    ``open_time_ms`` values exceeds ``interval_ms * threshold_factor``.
    """
    gaps: list[tuple[int, int]] = []
    threshold = interval_ms * threshold_factor
    for i in range(1, len(candles)):
        delta = candles[i].open_time_ms - candles[i - 1].open_time_ms
        if delta > threshold:
            gaps.append((candles[i - 1].open_time_ms, candles[i].open_time_ms))
    return gaps


def dedupe_candles(candles: Sequence[Candle]) -> list[Candle]:
    """Remove duplicates by ``open_time_ms``, keeping the last occurrence."""
    seen: dict[int, Candle] = {}
    for c in candles:
        seen[c.open_time_ms] = c
    return sorted(seen.values(), key=lambda c: c.open_time_ms)


def filter_closed_only(candles: Sequence[Candle]) -> list[Candle]:
    """Drop candles where ``is_final`` is False."""
    return [c for c in candles if c.is_final]


def check_candle_monotonic(candles: Sequence[Candle]) -> bool:
    """Verify ``open_time_ms`` is strictly increasing."""
    for i in range(1, len(candles)):
        if candles[i].open_time_ms <= candles[i - 1].open_time_ms:
            return False
    return True


def check_tick_monotonic(ticks: Sequence[PriceTick]) -> bool:
    """Verify tick timestamps are non-decreasing."""
    for i in range(1, len(ticks)):
        if ticks[i].ts_ms < ticks[i - 1].ts_ms:
            return False
    return True


# ------------------------------------------------------------------
# Internal
# ------------------------------------------------------------------

def _count_duplicates(candles: Sequence[Candle]) -> int:
    seen: set[int] = set()
    dups = 0
    for c in candles:
        if c.open_time_ms in seen:
            dups += 1
        seen.add(c.open_time_ms)
    return dups
