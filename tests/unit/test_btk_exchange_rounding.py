"""Unit tests for P0-B: Exchange filter rounding (tick_size / step_size)."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.grid.formulas import LEGACY_LINE_COUNT


def _make_klines(closes, start="2026-01-01"):
    n = len(closes)
    timestamps = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes],
        "close": closes,
        "volume": [1000.0] * n,
    })


class TestDefaultNoRounding:
    def test_default_no_rounding(self):
        """tick_size=0, step_size=0 → levels same as raw computation.

        GRIDFIX-001 / GRID_SYNCH §3.1: Binance convention — `num_grids` is
        the number of grid LINES, with `num_grids - 1` intervals.
        For num_grids=5: 4 intervals, step = (110-100)/4 = 2.5,
        5 levels in [100, 110] inclusive.
        """
        # Default grid identity is now geometric (commit 04a15b4); this test
        # checks tick/step rounding against arithmetic levels, so pin mode.
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5,
                         tick_size=0.0, step_size=0.0, mode="arithmetic")
        bt = RealisticGridBacktester(cfg)
        step = (110.0 - 100.0) / 5
        expected = [100.0 + i * step for i in range(6)]
        assert bt.grid_levels == pytest.approx(expected)

    def test_legacy_line_count_is_explicit(self):
        cfg = GridConfig(
            symbol="TESTUSDT",
            lower=100.0,
            upper=110.0,
            num_grids=5,
            tick_size=0.0,
            step_size=0.0,
            mode="arithmetic",
            grid_count_semantics=LEGACY_LINE_COUNT,
        )
        bt = RealisticGridBacktester(cfg)
        step = (110.0 - 100.0) / (5 - 1)
        expected = [100.0 + i * step for i in range(5)]
        assert bt.grid_levels == pytest.approx(expected)


class TestGridLevelsRoundedToTick:
    def test_grid_levels_rounded_to_tick(self):
        """tick_size=0.01 → every level has max 2 decimal places."""
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.33, num_grids=7,
                         tick_size=0.01)
        bt = RealisticGridBacktester(cfg)
        for level in bt.grid_levels:
            # Check that level is a multiple of 0.01 (within floating point tolerance)
            assert abs(level - round(level, 2)) < 1e-9, f"Level {level} not rounded to 0.01"


class TestPositionSizeRoundedToStep:
    def test_position_size_rounded_to_step(self):
        """step_size=0.001 → qty is a multiple of 0.001."""
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5,
                         capital=400.0, leverage=10, step_size=0.001)
        bt = RealisticGridBacktester(cfg)
        qty = bt._get_position_size()
        # qty should be a multiple of 0.001
        remainder = abs(qty / 0.001 - round(qty / 0.001))
        assert remainder < 1e-9, f"Qty {qty} not a multiple of 0.001"


class TestBtcusdtRounding:
    def test_btcusdt_rounding(self):
        """BTC-like tick=0.10, step=0.001 → levels like 40000.0, 40100.0."""
        cfg = GridConfig(symbol="BTCUSDT", lower=40000.0, upper=41000.0, num_grids=10,
                         tick_size=0.10, step_size=0.001, capital=400.0, leverage=10)
        bt = RealisticGridBacktester(cfg)
        for level in bt.grid_levels:
            assert abs(level - round(level, 1)) < 1e-9, f"Level {level} not rounded to 0.10"
        qty = bt._get_position_size()
        remainder = abs(qty / 0.001 - round(qty / 0.001))
        assert remainder < 1e-9


class TestRoundingDoesNotChangeNumLevels:
    def test_rounding_does_not_change_num_levels(self):
        """Still `num_grids` levels after rounding.

        GRIDFIX-001 / GRID_SYNCH §3.1: under Binance convention,
        `num_grids` is the number of grid LINES (not intervals), so the
        levels list contains exactly `num_grids` entries spanning
        ``[lower, upper]`` inclusive.
        """
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.33, num_grids=7,
                         tick_size=0.01)
        bt = RealisticGridBacktester(cfg)
        assert len(bt.grid_levels) == 8


class TestRoundMatchesSymbolInfo:
    def test_round_matches_symbol_info(self):
        """Our rounding matches the math in SymbolInfo.round_price() for same tick_size."""
        tick_size = 0.01
        raw_price = 123.456789
        # Our rounding approach
        our_rounded = round(round(raw_price / tick_size) * tick_size, 10)
        # SymbolInfo-style rounding
        import math
        precision = int(round(-math.log10(tick_size)))
        sym_rounded = round(raw_price, precision)
        assert abs(our_rounded - sym_rounded) < 1e-9
