"""Small deterministic helpers for non-overlapping monotonic schedulers."""

from __future__ import annotations

import math


def advance_nominal_start(
    scheduled_start: float,
    *,
    interval_seconds: float,
    completed_at: float,
) -> tuple[float, int]:
    """Advance to the first nominal start not already missed at completion.

    A cycle completing exactly on a nominal boundary may start the next cycle
    immediately.  A boundary strictly before completion is counted as skipped.
    """

    if not all(math.isfinite(value) for value in (scheduled_start, interval_seconds, completed_at)):
        raise ValueError("scheduler inputs must be finite")
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    next_start = scheduled_start + interval_seconds
    skipped = 0
    while next_start < completed_at:
        next_start += interval_seconds
        skipped += 1
    return next_start, skipped
