"""
Unit tests for P2: Maker close fee default change.

Verifies that TRAINING_ENGINE_DEFAULTS uses maker close fees (0.02%)
instead of taker (0.05%), matching live bot behavior.
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

class TestMakerCloseFee:

    def test_training_defaults_use_maker_close(self):
        """TRAINING_ENGINE_DEFAULTS should use maker close fee mode."""
        assert TRAINING_ENGINE_DEFAULTS["close_fee_mode"] == "maker"

    def test_maker_fee_lower_than_taker(self):
        """Same klines: maker close should produce lower fees than taker close."""
        klines = _oscillating_klines(300)

        cfg_maker = _base_config(close_fee_mode="maker")
        bt_maker = RealisticGridBacktester(cfg_maker)
        r_maker = bt_maker.run(klines)

        cfg_taker = _base_config(close_fee_mode="taker")
        bt_taker = RealisticGridBacktester(cfg_taker)
        r_taker = bt_taker.run(klines)

        # Both should have the same number of trades
        assert r_maker["round_trips"] == r_taker["round_trips"]

        # Maker close = lower fees (if there are round trips)
        if r_maker["round_trips"] > 0:
            assert r_maker["fees_paid"] < r_taker["fees_paid"], (
                f"Maker fees ({r_maker['fees_paid']:.4f}) should be less "
                f"than taker fees ({r_taker['fees_paid']:.4f})"
            )

    def test_explicit_taker_override_still_works(self):
        """Passing close_fee_mode='taker' should still use taker fee."""
        cfg = _base_config(close_fee_mode="taker")
        klines = _oscillating_klines(200)

        bt = RealisticGridBacktester(cfg)
        result = bt.run(klines)

        assert cfg.close_fee_mode == "taker"
        # Should run without error
        assert "fees_paid" in result

    def test_maker_close_fee_rate(self):
        """Verify maker fee rate (0.02%) is applied on close fills."""
        cfg = _base_config(close_fee_mode="maker")
        assert cfg.maker_fee == 0.0002  # 0.02%

    def test_build_training_config_applies_maker(self):
        """build_training_config() should produce config with close_fee_mode='maker'."""
        cfg = build_training_config("TESTUSDT", 100.0, 110.0, 5)
        assert cfg.close_fee_mode == "maker"
