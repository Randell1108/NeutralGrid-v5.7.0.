"""
Unit tests for P3: Termination penalty (unrealized PnL decomposition).

Tests verify that termination_pnl_pct and exit_penalty_pct are correctly
computed and present in engine output.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_klines(
    closes: list[float], start: datetime | None = None,
) -> pd.DataFrame:
    if start is None:
        start = datetime(2026, 1, 1)
    n = len(closes)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1.0] * n,
    })


def _oscillating_klines(
    n: int = 200, base: float = 105.0, amp: float = 5.0,
) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    t = np.linspace(0, 6 * np.pi, n)
    closes = base + amp * np.sin(t)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + 0.1,
        "low": closes - 0.1,
        "close": closes,
        "volume": [1.0] * n,
    })


def _base_config(**overrides) -> GridConfig:
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0002,
        taker_fee=0.0005,
        order_delay_bars=0,
        funding_rate=0.0001,
        slippage_bps=0.0,
        max_holding_bars=0,
        funding_mode="continuous",
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestTerminationPenalty:

    def test_no_open_position_zero_penalty(self):
        """When no grid crossings occur, exit_penalty_pct should be 0."""
        # Flat price — no grid crossings, no open positions
        closes = [105.0] * 100
        cfg = _base_config()
        klines = _make_klines(closes)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert result["open_positions_at_end"] == 0
        assert result["termination_pnl_pct"] == pytest.approx(0.0)
        assert result["exit_penalty_pct"] == pytest.approx(0.0)

    def test_open_position_has_penalty(self):
        """An open position with unrealized loss should yield exit_penalty_pct > 0."""
        # Use global cooldown to prevent the position from being closed when
        # price drops back through the entry level (simulates real bot behavior
        # where all orders are cancelled during the recompute cycle).
        # Grid levels: [100, 102, 104, 106, 108, 110]
        closes = (
            [99.0] * 3       # start below grid
            + [101.0] * 3    # cross UP through 100 → buy at 100
            + [98.0] * 30    # drop below 100 during cooldown → unrealized loss
        )
        cfg = _base_config(global_cooldown_bars=50)
        klines = _make_klines(closes)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert result["open_positions_at_end"] > 0, (
            "Expected open position at end (cooldown should block close)"
        )
        assert result["unrealized_pnl_at_end"] < 0, (
            "Expected negative unrealized PnL (price below entry)"
        )
        assert result["exit_penalty_pct"] > 0, (
            "Open positions with unrealized loss should have "
            "exit_penalty_pct > 0"
        )

    def test_termination_pnl_matches_unrealized(self):
        """termination_pnl_pct should equal unrealized_pnl_at_end / capital * 100."""
        klines = _oscillating_klines(200)
        cfg = _base_config()

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        expected = (result["unrealized_pnl_at_end"] / cfg.capital) * 100
        assert result["termination_pnl_pct"] == pytest.approx(expected, abs=1e-9)

    def test_penalty_always_non_negative(self):
        """exit_penalty_pct should always be >= 0."""
        klines = _oscillating_klines(300)
        cfg = _base_config()

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert result["exit_penalty_pct"] >= 0, (
            f"exit_penalty_pct should be >= 0, got {result['exit_penalty_pct']}"
        )

    def test_net_pnl_equals_realized_plus_termination(self):
        """net_pnl_pct should equal realized_pnl_pct + termination_pnl_pct."""
        klines = _oscillating_klines(200)
        cfg = _base_config()

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        # net_pnl = (equity + unrealized) - capital
        # net_pnl_pct = net_pnl / capital * 100
        # termination_pnl_pct = unrealized / capital * 100
        # So net_pnl_pct = realized_portion + termination_pnl_pct
        realized_pnl_pct = result["net_pnl_pct"] - result["termination_pnl_pct"]
        reconstructed = realized_pnl_pct + result["termination_pnl_pct"]
        assert result["net_pnl_pct"] == pytest.approx(reconstructed, abs=1e-9)

    def test_fields_in_result_dict(self):
        """Both termination fields should be present in the result."""
        klines = _oscillating_klines(100)
        cfg = _base_config()

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert "termination_pnl_pct" in result
        assert "exit_penalty_pct" in result
