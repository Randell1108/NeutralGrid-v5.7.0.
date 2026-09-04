"""
Unit tests for P0+P1: Global cooldown in the realistic backtester.

Tests verify that after ANY grid fill, ALL levels are blocked for N bars,
modeling the real bot cancel → recompute → replace cycle.
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
from backtest.btk_label_contract import TRAINING_ENGINE_DEFAULTS
from backtest.btk_unified_runner import build_training_config


# ── Helpers ──────────────────────────────────────────────────────────────────

def _oscillating_klines(
    n: int = 300, base: float = 105.0, amp: float = 5.0,
) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    t = np.linspace(0, 8 * np.pi, n)
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
    # GRIDFIX-001 / GRID_SYNCH §3.1: num_grids is grid LINES (Binance
    # convention). 6 lines over [100, 110] yields step=2.0 and the
    # integer-spaced level set [100, 102, 104, 106, 108, 110].
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=6,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0002,
        taker_fee=0.0005,
        order_delay_bars=0,
        funding_rate=0.0001,
        slippage_bps=0.0,
        max_holding_bars=0,
        funding_mode="continuous",
        close_fee_mode="taker",
        global_cooldown_bars=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestGlobalCooldown:

    def test_global_cooldown_zero_is_noop(self):
        """cooldown=0 should match legacy behavior (no blocking)."""
        klines = _oscillating_klines(200)

        cfg = _base_config(global_cooldown_bars=0)
        bt = RealisticGridBacktester(cfg)
        r = bt.run(klines)

        # Sanity: with no cooldown and no delay, there should be trades
        assert r["total_trades"] > 0

    def test_global_cooldown_blocks_all_levels(self):
        """With cooldown=5, trade count should drop vs cooldown=0."""
        klines = _oscillating_klines(300)

        cfg_no_cd = _base_config(global_cooldown_bars=0, order_delay_bars=0)
        bt_no_cd = RealisticGridBacktester(cfg_no_cd)
        r_no_cd = bt_no_cd.run(klines)

        cfg_cd = _base_config(global_cooldown_bars=5, order_delay_bars=0)
        bt_cd = RealisticGridBacktester(cfg_cd)
        r_cd = bt_cd.run(klines)

        assert r_cd["total_trades"] < r_no_cd["total_trades"], (
            f"Cooldown=5 produced {r_cd['total_trades']} trades, "
            f"but no-cooldown produced {r_no_cd['total_trades']}"
        )

    def test_global_cooldown_resumes_after_window(self):
        """Fills should resume after the cooldown window expires."""
        klines = _oscillating_klines(300)
        cfg = _base_config(global_cooldown_bars=5, order_delay_bars=0)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert result["round_trips"] > 0, (
            "No round trips with cooldown=5 — fills never resumed"
        )

    def test_global_cooldown_reduces_trade_count(self):
        """cooldown=20 should produce far fewer trades than cooldown=0."""
        klines = _oscillating_klines(500, base=105.0, amp=5.0)

        cfg_no_cd = _base_config(global_cooldown_bars=0, order_delay_bars=0)
        bt_no_cd = RealisticGridBacktester(cfg_no_cd)
        r_no_cd = bt_no_cd.run(klines)

        cfg_cd = _base_config(global_cooldown_bars=20, order_delay_bars=0)
        bt_cd = RealisticGridBacktester(cfg_cd)
        r_cd = bt_cd.run(klines)

        assert r_no_cd["total_trades"] > 0, "No trades without cooldown"
        assert r_cd["total_trades"] > 0, "No trades with cooldown"
        # Cooldown=20 should reduce trade count substantially
        assert r_cd["total_trades"] < r_no_cd["total_trades"] * 0.75, (
            f"Expected >25% reduction: {r_cd['total_trades']} vs "
            f"{r_no_cd['total_trades']}"
        )

    def test_global_cooldown_resets_on_each_fill(self):
        """After cooldown expires, a new fill should start a new window."""
        klines = _oscillating_klines(600, base=105.0, amp=5.0)
        cfg = _base_config(global_cooldown_bars=10, order_delay_bars=0)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        # Multiple fills should occur (cooldown resets for each)
        assert result["round_trips"] >= 2, (
            f"Expected ≥2 round trips with cooldown=10 over 600 bars, "
            f"got {result['round_trips']}"
        )

    def test_global_cooldown_in_result_dict(self):
        """global_cooldown_bars should be present in the engine output."""
        cfg = _base_config(global_cooldown_bars=15)
        klines = _oscillating_klines(100)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert "global_cooldown_bars" in result
        assert result["global_cooldown_bars"] == 15

    def test_global_cooldown_with_order_delay(self):
        """Global cooldown and order_delay_bars should coexist correctly."""
        klines = _oscillating_klines(500)

        cfg_delay = _base_config(order_delay_bars=3, global_cooldown_bars=0)
        bt_delay = RealisticGridBacktester(cfg_delay)
        r_delay = bt_delay.run(klines)

        cfg_both = _base_config(order_delay_bars=3, global_cooldown_bars=20)
        bt_both = RealisticGridBacktester(cfg_both)
        r_both = bt_both.run(klines)

        # Global cooldown should reduce trades further beyond order delay alone
        assert r_both["total_trades"] <= r_delay["total_trades"], (
            f"Adding cooldown should not increase trades: "
            f"{r_both['total_trades']} vs {r_delay['total_trades']}"
        )

    def test_training_defaults_include_global_cooldown(self):
        """TRAINING_ENGINE_DEFAULTS sets global_cooldown_bars=0 (commit 04a15b4:
        resting grid orders should not globally pause all levels after one fill;
        operator-confirmed intentional 2026-05-22)."""
        assert "global_cooldown_bars" in TRAINING_ENGINE_DEFAULTS
        assert TRAINING_ENGINE_DEFAULTS["global_cooldown_bars"] == 0

        # Verify build_training_config applies it
        cfg = build_training_config("TESTUSDT", 100.0, 110.0, 5)
        assert cfg.global_cooldown_bars == 0
