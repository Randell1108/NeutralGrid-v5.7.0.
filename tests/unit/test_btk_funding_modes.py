"""Tests for Phase 2: continuous vs snapshot funding accrual modes.

Verifies that:
- ``funding_mode="snapshot"`` charges only at exact 480-bar intervals (legacy).
- ``funding_mode="continuous"`` prorates funding per bar across open notional.
- Continuous funding >= snapshot funding for identical exposure profiles.
- Default ``funding_mode`` is ``"snapshot"`` (backward compatibility).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the top-level backtest package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_klines(n_bars: int, base_price: float, amplitude: float, lower: float, upper: float) -> pd.DataFrame:
    """Create synthetic 1-min klines that oscillate within a grid range.

    The price oscillates as a sine wave centered at *base_price* with the
    given *amplitude*, clamped to [lower, upper].  This ensures positions
    are opened and held throughout the test window so funding charges
    accumulate.
    """
    timestamps = pd.date_range("2026-01-01", periods=n_bars, freq="min")
    # Slow sine oscillation — period ~200 bars so the price crosses grid
    # levels multiple times, creating positions that stay open.
    raw = base_price + amplitude * np.sin(2 * np.pi * np.arange(n_bars) / 200)
    close = np.clip(raw, lower + amplitude * 0.1, upper - amplitude * 0.1)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": np.ones(n_bars) * 1000,
    })


def _make_flat_klines(n_bars: int, price: float) -> pd.DataFrame:
    """Create klines where price stays perfectly flat (no grid crossings)."""
    timestamps = pd.date_range("2026-01-01", periods=n_bars, freq="min")
    close = np.full(n_bars, price)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": np.ones(n_bars) * 1000,
    })


def _base_config(**overrides) -> GridConfig:
    """Return a GridConfig with sensible defaults for testing.

    GRIDFIX-001 / GRID_SYNCH §3.1: num_grids is grid LINES (Binance
    convention). 11 lines over [90, 110] yields step=2.0 and the
    integer-spaced level set [90, 92, ..., 110].
    """
    defaults = dict(
        symbol="TESTUSDT",
        lower=90.0,
        upper=110.0,
        num_grids=11,
        capital=400.0,
        leverage=10,
        maker_fee=0.0002,
        taker_fee=0.0005,
        funding_rate=0.0001,
        funding_interval_bars=480,
        slippage_bps=0.0,
        order_delay_bars=0,       # no delay so fills happen immediately
        max_holding_bars=99999,   # effectively disable stale-close
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSnapshotFunding:
    """Test 1: funding_mode='snapshot' charges only at exact interval ticks."""

    def test_snapshot_charges_at_interval(self):
        cfg = _base_config(funding_mode="snapshot")
        # 600 bars — one snapshot checkpoint at bar 480
        df = _make_klines(600, base_price=100.0, amplitude=5.0,
                          lower=cfg.lower, upper=cfg.upper)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)

        # Funding should have been charged (positions exist at bar 480).
        # With bidirectional positions, net funding can be positive or negative
        # depending on long/short balance at the checkpoint.
        assert result["funding_fees"] != 0, "Snapshot mode should charge funding at bar 480"
        assert result["funding_mode"] == "snapshot"

    def test_snapshot_no_charge_before_interval(self):
        """With only 479 bars, no snapshot checkpoint is reached."""
        cfg = _base_config(funding_mode="snapshot")
        df = _make_klines(479, base_price=100.0, amplitude=5.0,
                          lower=cfg.lower, upper=cfg.upper)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)

        assert result["funding_fees"] == 0.0, (
            "Snapshot mode should NOT charge funding before bar 480"
        )


class TestContinuousFunding:
    """Test 2: funding_mode='continuous' prorates per bar."""

    def test_continuous_charges_proportional(self):
        cfg = _base_config(funding_mode="continuous")
        df = _make_klines(600, base_price=100.0, amplitude=5.0,
                          lower=cfg.lower, upper=cfg.upper)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)

        assert result["funding_fees"] != 0, (
            "Continuous mode should accumulate funding every bar with positions "
            "(funding_fees is signed: net-short settles negative)"
        )
        assert result["funding_mode"] == "continuous"

    def test_continuous_zero_when_no_positions(self):
        """If price never crosses a grid level, no positions open, no funding."""
        cfg = _base_config(funding_mode="continuous")
        # Flat price at a grid level — close never crosses through a level
        df = _make_flat_klines(600, price=100.0)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)

        assert result["funding_fees"] == 0.0, (
            "Continuous mode should not charge funding when there are no positions"
        )


class TestContinuousVsSnapshot:
    """Test 3: continuous funding >= snapshot funding for same exposure."""

    def test_continuous_geq_snapshot_over_full_interval(self):
        """Over 600 bars (one 480-bar checkpoint), continuous should accumulate
        at least as much funding as snapshot because continuous charges every
        bar where positions exist, while snapshot charges only at bar 480.

        With steady positions, over 480 bars:
          continuous ≈ sum over bars of (notional * rate / 480)
          snapshot   = notional_at_480 * rate  (one charge)

        If notional is roughly constant, continuous total for the first 480
        bars ≈ notional * rate (same).  But continuous also charges bars
        481-599, so continuous > snapshot.
        """
        klines = _make_klines(600, base_price=100.0, amplitude=5.0,
                              lower=90.0, upper=110.0)

        cfg_snap = _base_config(funding_mode="snapshot")
        cfg_cont = _base_config(funding_mode="continuous")

        bt_snap = RealisticGridBacktester(cfg_snap)
        bt_cont = RealisticGridBacktester(cfg_cont)

        res_snap = bt_snap.run(klines.copy())
        res_cont = bt_cont.run(klines.copy())

        assert res_cont["funding_fees"] >= res_snap["funding_fees"], (
            f"Continuous ({res_cont['funding_fees']:.6f}) should be >= "
            f"snapshot ({res_snap['funding_fees']:.6f}) over 600 bars"
        )


class TestDefaultFundingMode:
    """Test 4: default funding_mode is 'snapshot' for backward compat."""

    def test_default_is_snapshot(self):
        cfg = GridConfig(
            symbol="TESTUSDT",
            lower=90.0,
            upper=110.0,
            num_grids=10,
        )
        assert cfg.funding_mode == "snapshot", (
            f"Default funding_mode should be 'snapshot', got '{cfg.funding_mode}'"
        )

    def test_result_includes_funding_mode(self):
        cfg = _base_config()  # uses default = "snapshot"
        df = _make_klines(100, base_price=100.0, amplitude=3.0,
                          lower=cfg.lower, upper=cfg.upper)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert "funding_mode" in result
        assert result["funding_mode"] == "snapshot"
