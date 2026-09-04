"""Shared grid-construction formulas — single source of truth for spacing and profit-per-grid.

Replaces three independent copies of the arithmetic formula:
  - F1: ``ExistingDataMapper.compute_profit_per_grid`` in ``training/data_generator.py``
  - F2: ``_compute_profit_per_grid_pct`` in ``scanner/empirical_profile_v20260302.py``
  - F3: ``GridCalculator.calculate_profit_per_grid`` in ``grid/calculator.py``

Both arithmetic and geometric branches are implemented per Binance documentation:
  - https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3
  - https://www.binance.com/en/support/faq/detail/688ff6ff08734848915de76a07b953dd

Two grid-count semantics are intentionally supported:

* ``legacy_line_count`` keeps the historical extractor convention where
  ``n`` is the number of visible grid lines and intervals are ``n - 1``.
* ``binance_displayed_intervals`` follows Binance's public Futures Grid FAQ,
  where the displayed grid count divides the range and therefore creates
  ``n + 1`` price levels.

See ``GRID_SYNCH.md`` (repo root) for the full design rationale and discrepancy audit.
"""
from __future__ import annotations

ARITHMETIC = "arithmetic"
GEOMETRIC = "geometric"
_VALID_MODES = (ARITHMETIC, GEOMETRIC)

LEGACY_LINE_COUNT = "legacy_line_count"
BINANCE_DISPLAYED_INTERVALS = "binance_displayed_intervals"
_VALID_GRID_COUNT_SEMANTICS = (LEGACY_LINE_COUNT, BINANCE_DISPLAYED_INTERVALS)


def grid_interval_count(n: int, grid_count_semantics: str = LEGACY_LINE_COUNT) -> int:
    """Return the number of price intervals implied by ``n``.

    The default is the historical project convention so older feature rows are
    not silently reinterpreted. Backtest code that wants Binance UI semantics
    must pass ``BINANCE_DISPLAYED_INTERVALS`` explicitly.
    """
    _validate_grid_count(n, grid_count_semantics)
    if grid_count_semantics == BINANCE_DISPLAYED_INTERVALS:
        return n
    return n - 1


def grid_level_count(n: int, grid_count_semantics: str = LEGACY_LINE_COUNT) -> int:
    """Return the number of grid price levels implied by ``n``."""
    intervals = grid_interval_count(n, grid_count_semantics)
    return intervals + 1


def grid_spacing_pct(
    low: float,
    high: float,
    n: int,
    mode: str,
    grid_count_semantics: str = LEGACY_LINE_COUNT,
) -> float:
    """Return the percent spacing between consecutive grid levels.

    Geometric spacing is constant ``(r - 1) * 100``. Arithmetic spacing is
    reported at the bottom of the range.
    """
    _validate_inputs(low, high, n, mode)
    intervals = grid_interval_count(n, grid_count_semantics)
    if mode == GEOMETRIC:
        r = (high / low) ** (1.0 / intervals)
        return (r - 1.0) * 100.0
    return (high - low) / low / intervals * 100.0


def profit_per_grid_pct(
    low: float,
    high: float,
    n: int,
    mode: str,
    c: float,
    grid_count_semantics: str = LEGACY_LINE_COUNT,
) -> float:
    """Return the average per-matched-grid-pair profit percent, net of trading fee ``c``.

    ``c`` is the per-side trading fee as a decimal (e.g. 0.001 = 0.1%).
    No default — caller must pass the fee constant explicitly.

    Geometric: for Binance semantics this returns Binance's documented
    ``(1-c) * r - 1 - c`` value. For legacy line-count semantics it preserves
    the historical ``(r - 1) - 2c`` approximation.
    Arithmetic: returns the average of ``min_profit`` and ``max_profit`` per Binance's
    arithmetic-grid formulas.
    """
    _validate_inputs(low, high, n, mode)
    intervals = grid_interval_count(n, grid_count_semantics)
    if c < 0:
        raise ValueError(f"taker fee c must be non-negative, got {c!r}")
    if mode == GEOMETRIC:
        r = (high / low) ** (1.0 / intervals)
        if grid_count_semantics == BINANCE_DISPLAYED_INTERVALS:
            return max(0.0, ((1.0 - c) * r) - 1.0 - c) * 100.0
        return max(0.0, (r - 1.0) - 2.0 * c) * 100.0
    d = (high - low) / intervals
    if d <= 0:
        return 0.0
    max_profit = (1.0 - c) * d / low - 2.0 * c
    min_profit = high * (1.0 - c) / (high - d) - 1.0 - c
    avg_profit = (min_profit + max_profit) / 2.0
    return max(0.0, avg_profit) * 100.0


def _validate_inputs(low: float, high: float, n: int, mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")
    _validate_grid_count(n, LEGACY_LINE_COUNT)
    if low <= 0:
        raise ValueError(f"low must be > 0, got {low}")
    if high <= low:
        raise ValueError(f"high ({high}) must be > low ({low})")


def _validate_grid_count(n: int, grid_count_semantics: str) -> None:
    if grid_count_semantics not in _VALID_GRID_COUNT_SEMANTICS:
        raise ValueError(
            "grid_count_semantics must be one of "
            f"{_VALID_GRID_COUNT_SEMANTICS!r}, got {grid_count_semantics!r}"
        )
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
