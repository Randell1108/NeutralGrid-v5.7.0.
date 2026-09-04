from __future__ import annotations

import pytest

from neutralgrid.live.monotonic_schedule import advance_nominal_start


def test_scheduler_keeps_next_boundary_when_cycle_finishes_early() -> None:
    assert advance_nominal_start(100.0, interval_seconds=180.0, completed_at=250.0) == (
        280.0,
        0,
    )


def test_scheduler_allows_exact_boundary_without_overlap() -> None:
    assert advance_nominal_start(100.0, interval_seconds=180.0, completed_at=280.0) == (
        280.0,
        0,
    )


def test_scheduler_skips_every_missed_nominal_start() -> None:
    assert advance_nominal_start(100.0, interval_seconds=180.0, completed_at=650.0) == (
        820.0,
        3,
    )


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan")])
def test_scheduler_rejects_invalid_interval(interval: float) -> None:
    with pytest.raises(ValueError):
        advance_nominal_start(1.0, interval_seconds=interval, completed_at=2.0)
