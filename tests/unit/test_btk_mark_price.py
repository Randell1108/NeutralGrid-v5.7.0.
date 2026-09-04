"""Unit tests for P1-A: Mark-price klines config wiring."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


def _make_klines(n=50, base=105.0):
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    t = np.linspace(0, 2 * np.pi, n)
    closes = base + 4.0 * np.sin(t)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": [1000.0] * n,
    })


class TestMarkPriceConfig:
    def test_default_price_source_is_last(self):
        cfg = GridConfig(symbol="X", lower=100, upper=110, num_grids=5)
        assert cfg.price_source == "last"

    def test_price_source_in_result(self):
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5,
                         capital=400.0, leverage=10, order_delay_bars=0,
                         max_holding_bars=0, funding_mode="continuous")
        df = _make_klines()
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["price_source"] == "last"

    def test_mark_price_source_accepted(self):
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5,
                         price_source="mark")
        assert cfg.price_source == "mark"

    def test_margin_mode_default_is_isolated(self):
        cfg = GridConfig(symbol="X", lower=100, upper=110, num_grids=5)
        assert cfg.margin_mode == "isolated"

    def test_margin_mode_in_result(self):
        cfg = GridConfig(symbol="TESTUSDT", lower=100.0, upper=110.0, num_grids=5,
                         capital=400.0, leverage=10, order_delay_bars=0,
                         max_holding_bars=0, funding_mode="continuous")
        df = _make_klines()
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["margin_mode"] == "isolated"
