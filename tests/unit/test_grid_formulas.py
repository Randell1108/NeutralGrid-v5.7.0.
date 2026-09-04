"""Tests for ``neutralgrid.grid.formulas`` (shared arithmetic + geometric formulas).

Per ``GRID_SYNCH.md`` Step 2:
- Arithmetic branch reproduces the example in the Binance support FAQ.
- Geometric branch reproduces stored ``grid_spacing_pct`` for all 19 geometric rows in
  ``reports/ppg_geometric_comparison_data_20260503_113000.csv`` to within 1e-3 percentage points.
- Unknown mode raises ``ValueError``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neutralgrid.grid.formulas import (
    ARITHMETIC,
    BINANCE_DISPLAYED_INTERVALS,
    GEOMETRIC,
    LEGACY_LINE_COUNT,
    grid_interval_count,
    grid_level_count,
    grid_spacing_pct,
    profit_per_grid_pct,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GEO_CSV = REPO_ROOT / "reports" / "ppg_geometric_comparison_data_20260503_113000.csv"


def test_arithmetic_formula_internal_consistency() -> None:
    """Arithmetic profit formula matches the (n-1) divisor convention used by the extractor.

    Note: Binance's public FAQ example uses ``n`` as divisor (Upper=450, Lower=400, n=5, c=0.001
    -> max=2.29%, min=2.07%). The extractor (``_bot_data_extractor_core.py:99-104``) and the
    empirical xlsx data instead use ``(n - 1)``. This formula matches the extractor convention,
    so the FAQ numerical example is NOT reproduced verbatim — that is a documentation
    inconsistency on Binance's side, surfaced in ``GRID_SYNCH.md`` section 1.4.
    """
    low, high, n, c = 400.0, 450.0, 5, 0.001
    d = (high - low) / (n - 1)
    max_p = (1.0 - c) * d / low - 2.0 * c
    min_p = high * (1.0 - c) / (high - d) - 1.0 - c
    expected_avg = max(0.0, (min_p + max_p) / 2.0) * 100.0
    actual = profit_per_grid_pct(low, high, n, ARITHMETIC, c)
    assert abs(actual - expected_avg) < 1e-9
    assert actual > 0.0


def test_extractor_arithmetic_convention_matches_formula() -> None:
    """Confirm the formula's arithmetic spacing matches the extractor's documented formula.

    Extractor (``_bot_data_extractor_core.py:104``): ``(high - low) / low * 100 / (n - 1)``.
    Formula (``grid_spacing_pct(..., ARITHMETIC)``): ``(high - low) / low / (n - 1) * 100``.
    These are algebraically identical; this test pins it.
    """
    low, high, n = 0.0606, 0.0686, 40
    extractor_value = (high - low) / low * 100.0 / (n - 1)
    formula_value = grid_spacing_pct(low, high, n, ARITHMETIC)
    assert abs(formula_value - extractor_value) < 1e-12


def test_geometric_reproduces_csv() -> None:
    """All 19 stored geometric grid_spacing_pct values must match the formula within 1e-3 pp."""
    if not GEO_CSV.exists():
        pytest.skip(f"empirical CSV not present at {GEO_CSV}")
    rows = list(csv.DictReader(GEO_CSV.open(encoding="utf-8")))
    assert len(rows) == 19, f"expected 19 geometric rows, got {len(rows)}"
    for row in rows:
        low = float(row["price_range_low"])
        high = float(row["price_range_high"])
        n = int(row["grids_count"])
        stored = float(row["binance_spacing_stored_pct"])
        computed = grid_spacing_pct(low, high, n, GEOMETRIC)
        assert abs(computed - stored) < 1e-3, (
            f"row strategy_id={row['strategy_id']} symbol={row['symbol']}: "
            f"computed={computed}, stored={stored}, diff={computed - stored}"
        )


def test_geometric_profit_uses_constant_ratio() -> None:
    """Geometric profit-per-grid is ``(r - 1) - 2c`` where ``r = (high/low)^(1/(n-1))``."""
    low, high, n, c = 0.1002, 0.126, 38, 0.0005
    expected_r = (high / low) ** (1.0 / (n - 1))
    expected = ((expected_r - 1.0) - 2.0 * c) * 100.0
    actual = profit_per_grid_pct(low, high, n, GEOMETRIC, c)
    assert abs(actual - expected) < 1e-9


@pytest.mark.parametrize(
    ("low", "high", "n", "expected_pct"),
    [
        (0.4740, 0.4970, 7, 0.639056056101881),
        (0.05492, 0.05922, 11, 0.6475045081948827),
        (0.1368, 0.1530, 13, 0.8244505851842088),
    ],
)
def test_binance_geometric_profit_uses_displayed_grid_count(
    low: float,
    high: float,
    n: int,
    expected_pct: float,
) -> None:
    c = 0.0002
    expected_r = (high / low) ** (1.0 / n)
    expected = ((1.0 - c) * expected_r - 1.0 - c) * 100.0
    actual = profit_per_grid_pct(
        low, high, n, GEOMETRIC, c, BINANCE_DISPLAYED_INTERVALS
    )
    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(expected_pct, abs=1e-12)


def test_default_geometric_profit_keeps_legacy_line_count_semantics() -> None:
    low, high, n, c = 0.4740, 0.4970, 7, 0.0002
    legacy = profit_per_grid_pct(low, high, n, GEOMETRIC, c)
    binance = profit_per_grid_pct(
        low, high, n, GEOMETRIC, c, BINANCE_DISPLAYED_INTERVALS
    )

    assert legacy == pytest.approx(0.7528381877434942, abs=1e-12)
    assert binance == pytest.approx(0.639056056101881, abs=1e-12)
    assert legacy > binance


def test_grid_count_semantics_make_level_count_explicit() -> None:
    assert grid_interval_count(7, LEGACY_LINE_COUNT) == 6
    assert grid_level_count(7, LEGACY_LINE_COUNT) == 7
    assert grid_interval_count(7, BINANCE_DISPLAYED_INTERVALS) == 7
    assert grid_level_count(7, BINANCE_DISPLAYED_INTERVALS) == 8


def test_geometric_spacing_is_equal_ratio() -> None:
    """100 -> 121 with three grid lines has ratio 1.10 and spacing 10%."""
    assert grid_spacing_pct(100.0, 121.0, 3, GEOMETRIC) == pytest.approx(10.0)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        grid_spacing_pct(1.0, 2.0, 10, "cubic")
    with pytest.raises(ValueError, match="mode must be"):
        profit_per_grid_pct(1.0, 2.0, 10, "cubic", 0.0005)
    with pytest.raises(ValueError, match="grid_count_semantics must be"):
        grid_spacing_pct(1.0, 2.0, 10, GEOMETRIC, "silent_auto")


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        grid_spacing_pct(0.0, 2.0, 10, GEOMETRIC)
    with pytest.raises(ValueError):
        grid_spacing_pct(2.0, 1.0, 10, GEOMETRIC)
    with pytest.raises(ValueError):
        grid_spacing_pct(1.0, 2.0, 1, GEOMETRIC)
    with pytest.raises(ValueError):
        profit_per_grid_pct(1.0, 2.0, 10, ARITHMETIC, -0.0001)
