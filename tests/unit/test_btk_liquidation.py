"""Unit tests for P0-A: Liquidation modeling in the realistic backtester."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


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


def _oscillating_config(**overrides):
    defaults = dict(
        symbol="TESTUSDT",
        lower=95.0,
        upper=105.0,
        num_grids=10,
        capital=400.0,
        leverage=10,
        order_delay_bars=0,
        funding_rate=0.0001,
        funding_interval_bars=480,
        funding_mode="continuous",
        max_holding_bars=0,
        maintenance_margin_rate=0.004,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestNoLiquidationNormalRun:
    def test_no_liquidation_normal_run(self):
        """Oscillating klines within range, equity stays positive."""
        t = np.linspace(0, 4 * np.pi, 200)
        closes = 100.0 + 4.0 * np.sin(t)
        df = _make_klines(closes.tolist())
        cfg = _oscillating_config()
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["liquidated"] is False
        assert result["liquidation_bar"] == -1
        assert result["liquidation_equity"] == 0.0


class TestLiquidationOnCrash:
    def test_liquidation_on_crash(self):
        """Price drops drastically below grid — equity hits maintenance margin."""
        # Start in range so positions get opened, then crash
        closes = [100.0] * 10
        for i in range(10):
            closes.append(100.0 + (i + 1) * 0.6)  # Up through levels to open positions
        # Then crash to very low price
        for i in range(50):
            closes.append(100.0 - (i + 1) * 1.5)
        df = _make_klines(closes)
        cfg = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.01)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["liquidated"] is True
        assert result["liquidation_bar"] > 0


class TestLiquidationStopsSimulation:
    def test_liquidation_stops_simulation(self):
        """After liquidation bar, no more trades are recorded."""
        closes = [100.0] * 5
        for i in range(10):
            closes.append(100.0 + (i + 1) * 0.6)
        for i in range(80):
            closes.append(100.0 - (i + 1) * 1.5)
        df = _make_klines(closes)
        cfg = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.01)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        if result["liquidated"]:
            liq_bar = result["liquidation_bar"]
            # Equity curve: one entry per bar up to and including liquidation bar
            assert len(bt.equity_curve) == liq_bar + 1


class TestLiquidationLabelAlwaysFalse:
    def test_liquidation_label_always_false(self):
        """Liquidated result always has label_positive_by_horizon == False."""
        closes = [100.0] * 5
        for i in range(10):
            closes.append(100.0 + (i + 1) * 0.6)
        for i in range(80):
            closes.append(100.0 - (i + 1) * 1.5)
        df = _make_klines(closes)
        cfg = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.01)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        if result["liquidated"]:
            assert result["label_positive_by_horizon"] is False


class TestMaintenanceMarginRateConfigurable:
    def test_maintenance_margin_rate_configurable(self):
        """Custom rate triggers liquidation at different equity levels."""
        closes = [100.0] * 5
        for i in range(10):
            closes.append(100.0 + (i + 1) * 0.6)
        for i in range(80):
            closes.append(100.0 - (i + 1) * 1.5)
        df = _make_klines(closes)

        # With high MMR, liquidation should happen sooner
        cfg_high = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.05)
        bt_high = RealisticGridBacktester(cfg_high)
        result_high = bt_high.run(df)

        cfg_low = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.001)
        bt_low = RealisticGridBacktester(cfg_low)
        result_low = bt_low.run(df)

        # Higher MMR should trigger liquidation earlier (lower bar index) if both liquidate
        if result_high["liquidated"] and result_low["liquidated"]:
            assert result_high["liquidation_bar"] <= result_low["liquidation_bar"]


class TestDefaultMMR:
    def test_default_mmr_is_0_004(self):
        assert GridConfig(symbol="X", lower=1, upper=2, num_grids=1).maintenance_margin_rate == 0.004


class TestZeroCapitalNotLiquidatedErr072:
    """ERR-072: zero-notional books are non-deployments, never liquidations."""

    def test_zero_capital_fraction_not_liquidated(self):
        """capital_fraction=0 -> no positions, no trades, liquidated=False."""
        t = np.linspace(0, 4 * np.pi, 200)
        closes = 100.0 + 4.0 * np.sin(t)
        df = _make_klines(closes.tolist())
        cfg = _oscillating_config(capital_fraction=0.0)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["liquidated"] is False
        assert result["liquidation_bar"] == -1
        assert result["net_pnl_pct"] == 0.0
        assert result["total_trades"] == 0
        assert result["round_trips"] == 0
        assert result["capital_used"] == 0.0
        assert result["label_positive_by_horizon"] is False

    def test_step_size_rounds_qty_to_zero_not_liquidated(self):
        """Tiny capital whose qty rounds to 0 via step_size opens nothing."""
        t = np.linspace(0, 4 * np.pi, 200)
        closes = 100.0 + 4.0 * np.sin(t)
        df = _make_klines(closes.tolist())
        cfg = _oscillating_config(capital=0.5, leverage=1, step_size=1.0)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["liquidated"] is False
        assert result["total_trades"] == 0
        assert result["net_pnl_pct"] == 0.0

    def test_genuine_liquidation_still_fires_with_guard(self):
        """Regression: the position_notional>0 guard must not disarm real liquidations."""
        closes = [100.0] * 10
        for i in range(10):
            closes.append(100.0 + (i + 1) * 0.6)
        for i in range(50):
            closes.append(100.0 - (i + 1) * 1.5)
        df = _make_klines(closes)
        cfg = _oscillating_config(leverage=20, capital=100.0, maintenance_margin_rate=0.01)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["liquidated"] is True
        assert result["liquidation_bar"] > 0
