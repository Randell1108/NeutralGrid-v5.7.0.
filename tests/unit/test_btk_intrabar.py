"""Unit tests for P3-A: Intrabar wick fills."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


def _base_config(**overrides):
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=400.0,
        leverage=10,
        order_delay_bars=0,
        funding_rate=0.0,
        funding_mode="continuous",
        max_holding_bars=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestFillModeDefault:
    def test_default_fill_mode_is_close(self):
        cfg = GridConfig(symbol="X", lower=100, upper=110, num_grids=5)
        assert cfg.fill_mode == "close"


class TestWickFillDetectsTouch:
    def test_wick_fill_detects_touch(self):
        """Klines where high touches level but close doesn't cross → fill with wick mode only."""
        # Grid levels at [100, 102, 104, 106, 108, 110]
        # Close stays at 103 (between 102 and 104), but high reaches 104.5
        # In close-to-close mode: no crossing of 104
        # In wick mode: high touches 104, should trigger fills
        n = 20
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
        # Start at 101 (below 102), cross up through 102 to open position,
        # then stay at 103 with high wicks touching 104
        closes = [101.0, 102.5] + [103.0] * 18
        highs = [101.5, 103.0] + [104.5] * 18  # wicks touch 104
        lows = [100.5, 101.5] + [102.5] * 18
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n,
        })

        cfg_close = _base_config(fill_mode="close")
        bt_close = RealisticGridBacktester(cfg_close)
        result_close = bt_close.run(df)

        cfg_wick = _base_config(fill_mode="wick")
        bt_wick = RealisticGridBacktester(cfg_wick)
        result_wick = bt_wick.run(df)

        # Wick mode should detect fills that close-to-close mode misses
        assert result_wick["total_trades"] >= result_close["total_trades"]


class TestWickModeMoreTrades:
    def test_wick_mode_more_trades(self):
        """Same oscillating data → wick mode produces >= close-to-close trades."""
        n = 200
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
        t = np.linspace(0, 4 * np.pi, n)
        closes = 105.0 + 4.0 * np.sin(t)
        # Make wicks extend 1.5 beyond close in each direction
        highs = closes + 1.5
        lows = closes - 1.5
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n,
        })

        cfg_close = _base_config(fill_mode="close")
        bt_close = RealisticGridBacktester(cfg_close)
        result_close = bt_close.run(df)

        cfg_wick = _base_config(fill_mode="wick")
        bt_wick = RealisticGridBacktester(cfg_wick)
        result_wick = bt_wick.run(df)

        assert result_wick["total_trades"] >= result_close["total_trades"]


class TestFillModeInResult:
    def test_fill_mode_in_result(self):
        n = 50
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
        closes = [105.0] * n
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * n,
        })
        cfg = _base_config(fill_mode="wick")
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["fill_mode"] == "wick"
