"""Sizing consistency tests for capital_fraction, vol targeting, and Kelly sweep."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.core.config import get_config
from neutralgrid.live.candidate_deploy_linker import DeployLinker
from neutralgrid.scanner.empirical_profile_v20260302 import (
    DEFAULT_PROFILE,
    align_ev_with_profile_context,
    generalized_kelly_details,
)


def _oscillating_klines(n: int = 180, base: float = 105.0, amp: float = 4.0) -> pd.DataFrame:
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    t = np.linspace(0, 6 * np.pi, n)
    closes = base + amp * np.sin(t)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def test_backtester_applies_capital_fraction_and_denominator():
    df = _oscillating_klines()
    cfg = GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=1000.0,
        capital_fraction=0.4,
        leverage=10,
        order_delay_bars=0,
        max_holding_bars=0,
        funding_mode="continuous",
        close_fee_mode="maker",
    )
    result = RealisticGridBacktester(cfg).run(df)

    assert result["capital_used"] == 400.0
    assert result["net_pnl"] == result["final_equity"] - result["capital_used"]
    assert result["net_pnl_pct"] == (result["net_pnl"] / result["capital_used"]) * 100.0


def test_backtester_applies_volatility_target_scale():
    df = _oscillating_klines()
    cfg = GridConfig(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=1000.0,
        capital_fraction=1.0,
        volatility_target_pct=3.0,
        volatility_proxy_pct=6.0,
        leverage=10,
        order_delay_bars=0,
        max_holding_bars=0,
    )
    result = RealisticGridBacktester(cfg).run(df)
    assert result["volatility_scale_applied"] == 0.5
    assert result["capital_used"] == 500.0


def test_deploy_linker_enforces_margin_from_capital_fraction(tmp_path):
    from neutralgrid.core.candidate_id import make_candidate_id

    linker = DeployLinker(linkage_dir=tmp_path / "linkage")
    candidate_id = make_candidate_id(
        "BTCUSDT",
        "20260304_100000",
        grid_lower=100.0,
        grid_upper=110.0,
        num_grids=5,
        leverage=10,
    )
    row = linker.log_deployment_from_row(
        {
            "candidate_id": candidate_id,
            "capital_fraction": 0.25,
            "grid_lower": 100.0,
            "grid_upper": 110.0,
            "num_grids": 5,
            "leverage": 10,
        },
        strategy_id="ng_test_margin",
    )
    expected = float(get_config().grid.capital) * 0.25
    assert float(row["margin_usdt"]) == expected
    assert float(row["capital_fraction"]) == 0.25


def test_kelly_optimizer_runs_sweep():
    out = generalized_kelly_details(
        meta_prob=0.68,
        profile=DEFAULT_PROFILE,
        fractional_kelly=0.50,
        optimize_fractional=True,
        fractional_sweep_min=0.10,
        fractional_sweep_max=1.00,
        fractional_sweep_step=0.10,
        drawdown_tolerance_pct=8.0,
        volatility_target_pct=3.0,
        volatility_proxy_pct=2.0,
        min_samples=20,
    )
    assert out["fractional_mode"] == "optimized"
    assert 0.10 <= float(out["fractional_multiplier"]) <= 1.00
    assert float(out["sweep_evaluated_points"]) > 0


def test_ev_alignment_uses_shrinkage_under_low_samples():
    profile = DEFAULT_PROFILE.__class__(
        source="test",
        samples=5,
        win_rate=0.5,
        avg_win_pct=3.0,
        avg_loss_pct=12.0,
        payoff_ratio_b=0.25,
        p95_drawdown_pct=12.0,
        fill_rate_scale=1.0,
        ev_alignment_slope=2.0,
        ev_alignment_intercept=0.0,
        ev_alignment_r2=0.5,
        ev_alignment_samples=5,
        symbol_profiles={},
        regime_profiles={},
    )
    out = align_ev_with_profile_context(
        ev_raw_pct=1.0,
        profile=profile,
        min_samples=20,
    )
    assert str(out["scope"]).endswith("_shrunk")
    assert 1.0 < float(out["ev_aligned"]) < 2.0


def test_empirical_fit_analytic_ev_matches_live_ranker_basis():
    from neutralgrid.scanner.empirical_profile_v20260302 import _analytic_ev_pct
    from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig

    config = RankingConfig(use_empirical_alignment=False)
    ranker = PnLRanker(config)
    score = ranker.compute_score(
        profit_per_grid_pct=1.25,
        num_grids=64,
        survival_prob=0.74,
        trend_prob=0.28,
        funding_rate=0.0015,
        range_size_pct=3.1,
        leverage=17,
    )

    expected_ev_raw = _analytic_ev_pct(
        profit_per_grid_pct=1.25,
        num_grids=64,
        survival_prob=0.74,
        funding_rate=0.0015,
        horizon_hours=float(config.horizon_hours),
        leverage=17,
        sl_pct=float(config.sl_pct),
        baseline_fills_per_hour=float(config.baseline_fills_per_hour),
        fill_rate_scale=1.0,
    )

    assert score.ev_raw == pytest.approx(expected_ev_raw, rel=1e-12)


def test_profile_from_rows_excludes_duration_gt_7h():
    from neutralgrid.scanner.empirical_profile_v20260302 import _profile_from_rows

    fast_rows = pd.DataFrame(
        [
            {
                "net_pnl_pct": 2.0 + i,
                "round_trips": 10 + i,
                "duration_hours": 6.0,
                "num_grids": 16,
                "grid_lower": 100.0,
                "grid_upper": 110.0,
                "survival_prob": 0.72,
                "funding_rate": 0.0005,
                "leverage": 12,
            }
            for i in range(6)
        ]
    )
    slow_rows = pd.DataFrame(
        [
            {
                "net_pnl_pct": -20.0 - i,
                "round_trips": 2 + i,
                "duration_hours": 12.0,
                "num_grids": 4,
                "grid_lower": 90.0,
                "grid_upper": 130.0,
                "survival_prob": 0.35,
                "funding_rate": 0.0040,
                "leverage": 3,
            }
            for i in range(6)
        ]
    )
    combined = pd.concat([fast_rows, slow_rows], ignore_index=True)

    fast_profile = _profile_from_rows(fast_rows)
    combined_profile = _profile_from_rows(combined)

    assert combined_profile["samples"] == pytest.approx(fast_profile["samples"])
    assert combined_profile["fill_rate_scale"] == pytest.approx(fast_profile["fill_rate_scale"])
    assert combined_profile["ev_alignment_slope"] == pytest.approx(fast_profile["ev_alignment_slope"])
    assert combined_profile["ev_alignment_intercept"] == pytest.approx(fast_profile["ev_alignment_intercept"])
    assert combined_profile["win_rate"] == pytest.approx(1.0)
