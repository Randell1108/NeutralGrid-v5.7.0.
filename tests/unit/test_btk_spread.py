"""Unit tests for P2-B: Bid-ask spread model."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


def _make_klines(n=200, base=105.0, amp=4.0):
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    t = np.linspace(0, 4 * np.pi, n)
    closes = base + amp * np.sin(t)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": [1000.0] * n,
    })


def _base_config(**overrides):
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=400.0,
        leverage=10,
        order_delay_bars=0,
        funding_rate=0.0001,
        funding_mode="continuous",
        max_holding_bars=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestSpreadDefaults:
    def test_default_spread_is_zero(self):
        cfg = GridConfig(symbol="X", lower=100, upper=110, num_grids=5)
        assert cfg.spread_bps == 0.0


class TestSpreadReducesPnl:
    def test_spread_reduces_pnl(self):
        """Same config with spread_bps=1.0 → lower net_pnl than spread_bps=0.0."""
        df = _make_klines()

        cfg_no_spread = _base_config(spread_bps=0.0)
        bt_no = RealisticGridBacktester(cfg_no_spread)
        result_no = bt_no.run(df)

        cfg_spread = _base_config(spread_bps=1.0)
        bt_spread = RealisticGridBacktester(cfg_spread)
        result_spread = bt_spread.run(df)

        # With spread, PnL should be lower (spread is a cost)
        assert result_spread["net_pnl"] < result_no["net_pnl"]


class TestSpreadAdditiveWithSlippage:
    def test_spread_is_additive_with_slippage(self):
        """spread + slippage → more cost than either alone."""
        df = _make_klines()

        cfg_spread_only = _base_config(spread_bps=1.0, slippage_bps=0.0)
        bt_spread = RealisticGridBacktester(cfg_spread_only)
        result_spread = bt_spread.run(df)

        cfg_slip_only = _base_config(spread_bps=0.0, slippage_bps=1.0)
        bt_slip = RealisticGridBacktester(cfg_slip_only)
        result_slip = bt_slip.run(df)

        cfg_both = _base_config(spread_bps=1.0, slippage_bps=1.0)
        bt_both = RealisticGridBacktester(cfg_both)
        result_both = bt_both.run(df)

        # Both should cost more than either alone
        assert result_both["net_pnl"] < result_spread["net_pnl"]
        assert result_both["net_pnl"] < result_slip["net_pnl"]


class TestSpreadInResult:
    def test_spread_bps_in_result(self):
        df = _make_klines(n=50)
        cfg = _base_config(spread_bps=2.5)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["spread_bps"] == 2.5
