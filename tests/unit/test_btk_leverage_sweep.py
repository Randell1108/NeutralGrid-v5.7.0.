"""Unit tests for P2-A: Leverage/margin sweep."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig
from backtest.btk_leverage_sweep import sweep_leverage, find_optimal_leverage


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
        maintenance_margin_rate=0.004,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestSweepLeverage:
    def test_sweep_returns_list(self):
        cfg = _base_config()
        df = _make_klines()
        results = sweep_leverage(cfg, df, leverage_range=range(1, 6))
        assert isinstance(results, list)
        assert len(results) > 0

    def test_sweep_result_count(self):
        cfg = _base_config()
        df = _make_klines()
        lev_range = range(1, 6)
        results = sweep_leverage(cfg, df, leverage_range=lev_range)
        assert len(results) == len(lev_range)

    def test_optimal_leverage_not_liquidated(self):
        cfg = _base_config()
        df = _make_klines()
        results = sweep_leverage(cfg, df, leverage_range=range(1, 6))
        optimal = find_optimal_leverage(results)
        assert not optimal.get("liquidated", False)

    def test_higher_leverage_increases_pnl_or_liquidates(self):
        """Monotonic relationship: more leverage = higher PnL or liquidation."""
        cfg = _base_config()
        df = _make_klines()
        results = sweep_leverage(cfg, df, leverage_range=range(1, 11))
        # Sort by leverage (original order)
        by_lev = sorted(results, key=lambda r: r["sweep_leverage"])
        # Check the non-liquidated ones have non-decreasing abs(net_pnl)
        # (This is approximate — leverage amplifies both gains and losses)
        non_liq = [r for r in by_lev if not r.get("liquidated", False)]
        if len(non_liq) >= 2:
            # Higher leverage should produce larger absolute PnL
            assert abs(non_liq[-1]["net_pnl"]) >= abs(non_liq[0]["net_pnl"]) - 0.01
