"""
Unit tests for R7: 6h horizon defaults.

Validates that GridConfig defaults to a 6-hour holding horizon (360 bars
at 1-minute resolution).
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig


def test_default_max_holding_bars_is_360():
    """Default max_holding_bars should be 360 (6h at 1m resolution)."""
    cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5)
    assert cfg.max_holding_bars == 360


def test_default_horizon_is_6_hours():
    """360 bars / 60 minutes_per_hour = 6h."""
    cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5)
    assert cfg.max_holding_bars / 60 == 6.0
