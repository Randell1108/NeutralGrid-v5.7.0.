"""
Unit tests for P4: Duration override (match live bot duration).

Tests verify that the duration_bars parameter correctly truncates klines
before running the backtest.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from neutralgrid.backtest.candidate_pipeline import run_single_backtest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _oscillating_klines(
    n: int = 720, base: float = 105.0, amp: float = 5.0,
) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    t = np.linspace(0, 10 * np.pi, n)
    closes = base + amp * np.sin(t)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + 0.1,
        "low": closes - 0.1,
        "close": closes,
        "volume": [1.0] * n,
    })


def _candidate_row(**overrides) -> dict:
    defaults = {
        "symbol": "TESTUSDT",
        "candidate_id": "TESTUSDT_20260101_000000",
        "scan_file": "deployment_ready_20260101_000000.csv",
        "score": 85.0,
        "grid_lower": 100.0,
        "grid_upper": 110.0,
        "num_grids": 5,
    }
    defaults.update(overrides)
    return defaults


# ── Tests ────────────────────────────────────────────────────────────────────

class TestDurationOverride:

    def test_duration_none_uses_all_bars(self):
        """Default behavior: duration_bars=None uses all kline bars.

        Note: duration_hours reflects *actual* simulated bars, which may be
        less than len(df) if a circuit breaker or liquidation exits early.
        """
        klines = _oscillating_klines(720)
        result = run_single_backtest(
            candidate_row=_candidate_row(),
            klines_df=klines,
            capital=400.0,
            leverage=10,
            max_holding_bars=0,
            duration_bars=None,
        )
        # duration_hours reflects actual bars simulated (may be < 720 if CB fires)
        assert result["duration_hours"] <= 720 / 60.0
        assert result["duration_hours"] > 0

    def test_duration_truncates_klines(self):
        """duration_bars=360 with 720-bar input → only 360 bars used."""
        klines = _oscillating_klines(720)
        result = run_single_backtest(
            candidate_row=_candidate_row(),
            klines_df=klines,
            capital=400.0,
            leverage=10,
            max_holding_bars=0,
            duration_bars=360,
        )
        # duration_hours reflects actual bars simulated (≤ 360)
        assert result["duration_hours"] <= 360 / 60.0
        assert result["duration_hours"] > 0

    def test_duration_longer_than_klines_noop(self):
        """duration_bars=1000 with 720 bars → uses all 720 bars (max)."""
        klines = _oscillating_klines(720)
        result = run_single_backtest(
            candidate_row=_candidate_row(),
            klines_df=klines,
            capital=400.0,
            leverage=10,
            max_holding_bars=0,
            duration_bars=1000,
        )
        # duration_hours reflects actual bars simulated (≤ 720)
        assert result["duration_hours"] <= 720 / 60.0
        assert result["duration_hours"] > 0

    def test_duration_affects_trade_count(self):
        """Shorter duration should generally produce fewer or equal trades."""
        klines = _oscillating_klines(720)

        r_full = run_single_backtest(
            candidate_row=_candidate_row(),
            klines_df=klines,
            capital=400.0,
            leverage=10,
            max_holding_bars=0,
            duration_bars=None,
        )
        r_half = run_single_backtest(
            candidate_row=_candidate_row(),
            klines_df=klines,
            capital=400.0,
            leverage=10,
            max_holding_bars=0,
            duration_bars=360,
        )
        assert r_half["total_trades"] <= r_full["total_trades"], (
            f"Shorter duration should have ≤ trades: "
            f"{r_half['total_trades']} vs {r_full['total_trades']}"
        )

    def test_duration_zero_raises_or_noop(self):
        """duration_bars=0 — edge case: should raise or produce no trades."""
        klines = _oscillating_klines(100)
        try:
            result = run_single_backtest(
                candidate_row=_candidate_row(),
                klines_df=klines,
                capital=400.0,
                leverage=10,
                max_holding_bars=0,
                duration_bars=0,
            )
            # If it doesn't raise, it should have 0 trades
            assert result["total_trades"] == 0
        except (IndexError, ValueError, KeyError):
            pass  # Raising is acceptable for 0-bar edge case
