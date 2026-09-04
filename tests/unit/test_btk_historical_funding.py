"""Unit tests for P1-B: Historical funding rate series support."""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


def _make_klines(n=1000, base=105.0, amp=4.0):
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min")
    t = np.linspace(0, 8 * np.pi, n)
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
        funding_interval_bars=480,
        funding_mode="continuous",
        max_holding_bars=0,
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


class TestDefaultNoSeries:
    def test_default_no_series(self):
        """funding_rate_series=None → uses static rate (backward compat)."""
        cfg = _base_config()
        assert cfg.funding_rate_series is None
        df = _make_klines(n=500)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["funding_rate_series_len"] == 0


class TestSeriesOverridesStaticRate:
    def test_series_overrides_static_rate(self):
        """Series of [0.0001, 0.0005] → second 8h window uses higher rate → more funding."""
        df = _make_klines(n=1000)

        # Static rate baseline
        cfg_static = _base_config(funding_rate=0.0001)
        bt_static = RealisticGridBacktester(cfg_static)
        result_static = bt_static.run(df)

        # Series with higher rate in second window
        cfg_series = _base_config(funding_rate=0.0001, funding_rate_series=[0.0001, 0.0005])
        bt_series = RealisticGridBacktester(cfg_series)
        result_series = bt_series.run(df)

        # Series should charge more funding because second window rate is 5x higher
        # (funding_fees is signed; compare magnitudes — see ERR-056)
        assert abs(result_series["funding_fees"]) > abs(result_static["funding_fees"])


class TestSeriesShorterThanDataExtendsLast:
    def test_series_shorter_than_data_extends_last(self):
        """1-element series applied to 24h data → last rate used for all windows."""
        df = _make_klines(n=1000)
        cfg = _base_config(funding_rate_series=[0.0003])
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["funding_rate_series_len"] == 1
        # Should not error; funding should be charged (sign depends on net notional)
        assert result["funding_fees"] != 0


class TestSeriesLengthInResult:
    def test_series_length_in_result(self):
        """Result dict contains correct funding_rate_series_len."""
        cfg = _base_config(funding_rate_series=[0.0001, 0.0002, 0.0003])
        df = _make_klines(n=100)
        bt = RealisticGridBacktester(cfg)
        result = bt.run(df)
        assert result["funding_rate_series_len"] == 3
