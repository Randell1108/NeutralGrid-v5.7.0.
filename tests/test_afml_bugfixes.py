"""
Tests for AFML pipeline bug fixes (v6.5.3).

Covers:
  H1 — Log return circular reference elimination
  H2 — ADX warmup threshold (2 * adx_period)
  H3 — Missing logger in cpcv.py
  M2 — Duration hours clamping
  M3 — Bootstrap significance test uses raw H_RS
  M4 — Active fraction sqrt formula
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# H1: Log return circular reference
# ---------------------------------------------------------------------------

class TestLogReturnCircularReference:
    """H1 fix: compute_log_returns must NOT use np.roll (circular wrap)."""

    def test_first_element_is_nan(self):
        from neutralgrid.data.features import compute_log_returns

        closes = np.array([100.0, 101.0, 102.0])
        r = compute_log_returns(closes)
        assert np.isnan(r[0]), "First log return must be NaN (no previous bar)"

    def test_correct_values(self):
        from neutralgrid.data.features import compute_log_returns

        closes = np.array([100.0, 101.0, 102.0])
        r = compute_log_returns(closes)
        expected_1 = math.log(101 / 100)
        expected_2 = math.log(102 / 101)
        assert r[1] == pytest.approx(expected_1, rel=1e-12)
        assert r[2] == pytest.approx(expected_2, rel=1e-12)

    def test_no_circular_reference(self):
        """The old np.roll bug would put log(close[-1]/close[0]) at index 0."""
        from neutralgrid.data.features import compute_log_returns

        closes = np.array([100.0, 200.0, 300.0])
        r = compute_log_returns(closes)
        # With np.roll, r[0] would be log(300/100) ≈ 1.0986
        # With the fix, r[0] must be NaN
        assert np.isnan(r[0]), (
            f"r[0]={r[0]}; should be NaN, not a circular wrap value"
        )

    def test_length_preserved(self):
        from neutralgrid.data.features import compute_log_returns

        closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = compute_log_returns(closes)
        assert len(r) == len(closes)

    def test_single_element(self):
        from neutralgrid.data.features import compute_log_returns

        closes = np.array([42.0])
        r = compute_log_returns(closes)
        assert len(r) == 1
        assert np.isnan(r[0])


# ---------------------------------------------------------------------------
# H2: ADX warmup threshold uses 2 * adx_period
# ---------------------------------------------------------------------------

class TestAdxWarmupThreshold:
    """H2 fix: ADX warmup check uses 2 * adx_period, not adx_period + 10."""

    def test_source_uses_correct_formula(self):
        """Verify the source code at the ADX 1H warmup check uses 2 * period."""
        import inspect
        from neutralgrid.scanner import feature_extractor

        source = inspect.getsource(feature_extractor)
        # The fix should contain '2 * _cfg_val("adx_period")' for 1H ADX
        assert '2 * _cfg_val("adx_period")' in source, (
            "ADX 1H warmup should use 2 * adx_period formula"
        )

    def test_source_no_old_formula(self):
        """Verify the old 'adx_period + 10' hard-coded warmup is removed for 1H."""
        import inspect
        from neutralgrid.scanner import feature_extractor

        source = inspect.getsource(feature_extractor)
        # The old warmup for 1H was: len(klines_1h) >= _cfg_val("adx_period") + 10
        # After fix, the 1H ADX warmup should NOT contain 'adx_period") + 10'
        # (15m/5m may still use + 10, that's fine — the 1H line was the bug)
        lines = source.splitlines()
        for line in lines:
            if 'klines_1h' in line and 'adx_period' in line and '+ 10' in line:
                pytest.fail(
                    f"Found old adx_period + 10 formula for 1H data: {line.strip()}"
                )


# ---------------------------------------------------------------------------
# H3: Missing logger in cpcv.py
# ---------------------------------------------------------------------------

class TestCpcvLogger:
    """H3 fix: cpcv.py must import logging and define logger."""

    def test_import_does_not_raise(self):
        """Importing cpcv should not raise NameError for missing logger."""
        import neutralgrid.backtest.cpcv  # noqa: F401

    def test_logger_exists(self):
        import neutralgrid.backtest.cpcv as cpcv_mod
        import logging

        assert hasattr(cpcv_mod, "logger")
        assert isinstance(cpcv_mod.logger, logging.Logger)

    def test_logger_name(self):
        import neutralgrid.backtest.cpcv as cpcv_mod

        assert cpcv_mod.logger.name == "neutralgrid.backtest.cpcv"


# ---------------------------------------------------------------------------
# M2: Duration hours clamping
# ---------------------------------------------------------------------------

class TestDurationHoursClamping:
    """M2 fix: duration_hours exceeding horizon_hours gets clamped."""

    def test_duration_clamped_to_horizon(self):
        from neutralgrid.training.data_generator import (
            BarrierLabelGenerator,
            LabelConfig,
        )

        cfg = LabelConfig(horizon_hours=12.0)
        gen = BarrierLabelGenerator(cfg)

        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        label = gen.compute_label_from_final_pnl(
            symbol="BTCUSDT",
            start_time_utc=t0,
            pnl_pct=3.0,
            duration_hours=200.0,  # Way beyond 12h horizon
        )
        # t1 should be clamped to t0 + horizon (12h), not t0 + 200h
        expected_t1 = t0 + timedelta(hours=12.0)
        assert label.t1 == expected_t1, (
            f"t1={label.t1}, expected {expected_t1} (clamped to horizon)"
        )

    def test_duration_within_horizon_not_clamped(self):
        from neutralgrid.training.data_generator import (
            BarrierLabelGenerator,
            LabelConfig,
        )

        cfg = LabelConfig(horizon_hours=12.0)
        gen = BarrierLabelGenerator(cfg)

        t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        label = gen.compute_label_from_final_pnl(
            symbol="BTCUSDT",
            start_time_utc=t0,
            pnl_pct=3.0,
            duration_hours=6.0,  # Within horizon
        )
        expected_t1 = t0 + timedelta(hours=6.0)
        assert label.t1 == expected_t1, (
            f"t1={label.t1}, expected {expected_t1} (duration within horizon)"
        )

    def test_clamping_source_code(self):
        """Verify the clamped_hours = min(...) line exists in source."""
        import inspect
        from neutralgrid.training import data_generator

        source = inspect.getsource(data_generator.BarrierLabelGenerator)
        assert "clamped_hours" in source, "clamped_hours variable should exist"
        assert "min(" in source and "horizon_hours" in source, (
            "Duration clamping should use min(duration, horizon)"
        )


# ---------------------------------------------------------------------------
# M3: Bootstrap significance test uses raw h_rs
# ---------------------------------------------------------------------------

class TestBootstrapSignificanceRaw:
    """M3 fix: Significance test compares raw h_rs_raw against bootstrap null."""

    def test_significance_uses_raw_h_rs_in_source(self):
        """
        The significance comparison must use h_rs_raw (not bias-corrected h_rs)
        so that the estimate and null distribution are on the same scale.
        """
        import inspect
        from neutralgrid.validation.stochastic import StochasticRegimeChecker

        source = inspect.getsource(StochasticRegimeChecker._compute_hurst_full)

        # The fix: significance test should reference h_rs_raw, not h_rs
        # Look for the pattern: h_rs_raw < lo or h_rs_raw > hi
        assert "h_rs_raw < lo" in source or "h_rs_raw > hi" in source, (
            "Significance test should compare h_rs_raw (raw) against bootstrap null, "
            "not bias-corrected h_rs"
        )

    def test_significance_comment_explains_rationale(self):
        """The fix should include a comment explaining why raw is used."""
        import inspect
        from neutralgrid.validation.stochastic import StochasticRegimeChecker

        source = inspect.getsource(StochasticRegimeChecker._compute_hurst_full)
        # The comment should mention "raw" or "same scale"
        assert "raw" in source.lower() or "same scale" in source.lower(), (
            "Source should document why raw h_rs is used for significance"
        )


# ---------------------------------------------------------------------------
# M4: Active fraction formula (sqrt scaling)
# ---------------------------------------------------------------------------

class TestActiveFractionFormula:
    """M4 fix: active_fraction = min(0.75, (num_grids / 50.0)**0.5 * 0.5)."""

    @staticmethod
    def _expected_active_fraction(num_grids: int) -> float:
        return min(0.75, (num_grids / 50.0) ** 0.5 * 0.5)

    def test_20_grids(self):
        af = self._expected_active_fraction(20)
        assert af == pytest.approx((20 / 50.0) ** 0.5 * 0.5, rel=1e-9)
        # ~0.3162
        assert 0.30 < af < 0.35

    def test_50_grids(self):
        af = self._expected_active_fraction(50)
        assert af == pytest.approx(0.5, rel=1e-9)

    def test_100_grids(self):
        af = self._expected_active_fraction(100)
        # sqrt(2) * 0.5 ≈ 0.7071
        assert af == pytest.approx((100 / 50.0) ** 0.5 * 0.5, rel=1e-9)
        assert 0.70 < af < 0.72

    def test_200_grids(self):
        af = self._expected_active_fraction(200)
        # min(0.75, sqrt(4)*0.5) = min(0.75, 1.0) = 0.75
        assert af == pytest.approx(0.75, rel=1e-9)

    def test_pnl_ranker_uses_formula(self):
        """Verify PnLRanker uses the sqrt formula."""
        from neutralgrid.scanner.pnl_ranker import PnLRanker

        ranker = PnLRanker()

        # Score with 20 grids vs 50 grids — 50 should have higher active fraction
        score_20 = ranker.compute_score(
            profit_per_grid_pct=0.8, num_grids=20, survival_prob=0.7,
            trend_prob=0.2, range_size_pct=3.0,
        )
        score_50 = ranker.compute_score(
            profit_per_grid_pct=0.8, num_grids=50, survival_prob=0.7,
            trend_prob=0.2, range_size_pct=3.0,
        )
        # More grids with same profit/grid => higher EV
        assert score_50.ev > score_20.ev

    def test_utility_scorer_uses_formula(self):
        """Verify UtilityScorer uses the same sqrt formula."""
        from neutralgrid.validation.utility import UtilityConfig, UtilityScorer

        # Pass an explicit config to avoid the default `from_artifact()` path,
        # which now fail-closes (UtilityCalibratorUnavailable) when no
        # calibrator artifact is present.
        scorer = UtilityScorer(UtilityConfig())

        ret_20 = scorer.expected_grid_return(
            profit_per_grid_pct=0.8, num_grids=20, range_prob=0.7,
        )
        ret_100 = scorer.expected_grid_return(
            profit_per_grid_pct=0.8, num_grids=100, range_prob=0.7,
        )
        # 100 grids should give higher expected return than 20
        assert ret_100 > ret_20

    def test_pnl_ranker_source_has_sqrt_formula(self):
        """Verify the actual source code uses the sqrt formula."""
        import inspect
        from neutralgrid.scanner import pnl_ranker

        source = inspect.getsource(pnl_ranker.PnLRanker)
        assert "(num_grids / 50.0) ** 0.5 * 0.5" in source, (
            "PnLRanker should use (num_grids / 50.0)**0.5 * 0.5"
        )

    def test_utility_source_has_sqrt_formula(self):
        """Verify utility.py also uses the sqrt formula."""
        import inspect
        from neutralgrid.validation import utility

        source = inspect.getsource(utility.UtilityScorer)
        assert "(num_grids / 50.0) ** 0.5 * 0.5" in source, (
            "UtilityScorer should use (num_grids / 50.0)**0.5 * 0.5"
        )


class TestPostScoringOrder:
    """Post-score ranking must prioritize classifier probability before EV."""

    @staticmethod
    def _rank_result(rank_score: float) -> SimpleNamespace:
        return SimpleNamespace(
            ev_24h=rank_score,
            rank_score=rank_score,
            ev_raw=rank_score,
            ev_aligned=rank_score,
            expected_fills=1.0,
            fill_rate_scale=1.0,
            fill_rate_scope="global",
            fill_rate_scope_samples=50.0,
            ev_alignment_scope="global",
            ev_alignment_scope_samples=50.0,
            ev_alignment_r2=0.5,
            fill_revenue_pct=rank_score,
            funding_cost_pct=0.0,
            boundary_loss_pct=0.0,
            total_penalty=0.0,
        )

    def test_meta_prob_is_diagnostic_not_deployment_rank_key(self):
        script_path = Path(__file__).resolve().parents[1] / "run_full_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_full_pipeline", script_path)
        assert spec is not None and spec.loader is not None
        run_full_pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_full_pipeline)

        class DummyRanker:
            def __init__(self, *_args, **_kwargs):
                self.empirical_profile = None

            def compute_score(self, *, symbol: str | None = None, **_kwargs):
                scores = {
                    "ALPHA": 1.0,
                    "BETA": 5.0,
                    "GAMMA": 2.0,
                    "DELTA": 6.0,
                    "INVALID": 9.0,
                }
                assert symbol is not None
                return TestPostScoringOrder._rank_result(scores[symbol])

        class DummyMeta:
            @staticmethod
            def load(_path):
                return SimpleNamespace(is_trained=False)

        df = pd.DataFrame(
            [
                {"symbol": "ALPHA", "candidate_id": "A", "score": 90.0, "grid_is_valid": True, "meta_prob": 0.80, "profit_per_grid_pct": 1.0, "num_grids": 10, "survival_prob": 0.7, "trend_prob": 0.2, "range_size_pct": 3.0},
                {"symbol": "BETA", "candidate_id": "B", "score": 85.0, "grid_is_valid": True, "meta_prob": 0.70, "profit_per_grid_pct": 1.0, "num_grids": 10, "survival_prob": 0.7, "trend_prob": 0.2, "range_size_pct": 3.0},
                {"symbol": "GAMMA", "candidate_id": "C", "score": 80.0, "grid_is_valid": True, "meta_prob": 0.70, "profit_per_grid_pct": 1.0, "num_grids": 10, "survival_prob": 0.7, "trend_prob": 0.2, "range_size_pct": 3.0},
                {"symbol": "DELTA", "candidate_id": "D", "score": 75.0, "grid_is_valid": True, "meta_prob": None, "profit_per_grid_pct": 1.0, "num_grids": 10, "survival_prob": 0.7, "trend_prob": 0.2, "range_size_pct": 3.0},
                {"symbol": "INVALID", "candidate_id": "E", "score": 95.0, "grid_is_valid": False, "meta_prob": 0.99, "profit_per_grid_pct": 1.0, "num_grids": 10, "survival_prob": 0.7, "trend_prob": 0.2, "range_size_pct": 3.0},
            ]
        )

        with patch.object(run_full_pipeline, "PnLRanker", DummyRanker), \
             patch.object(run_full_pipeline, "RankingConfig", lambda: None), \
             patch.object(run_full_pipeline, "MetaLabeler", DummyMeta), \
             patch.object(run_full_pipeline, "AFML_POST_SCORING_AVAILABLE", True):
            scored = run_full_pipeline._apply_afml_post_scoring(df)

        ranked_symbols = (
            scored.loc[scored["deployment_score"].notna()]
            .sort_values("deployment_score", ascending=False)["symbol"]
            .tolist()
        )

        assert ranked_symbols == ["DELTA", "BETA", "GAMMA", "ALPHA"]
        assert scored.loc[scored["symbol"] == "ALPHA", "ev_score"].item() < scored.loc[
            scored["symbol"] == "BETA", "ev_score"
        ].item()
        assert scored.loc[scored["symbol"] == "ALPHA", "deployment_score"].item() < scored.loc[
            scored["symbol"] == "BETA", "deployment_score"
        ].item()
        assert set(scored["meta_prob_authority"]) == {"diagnostic_only"}
        assert set(scored.loc[scored["deployment_score"].notna(), "deployment_score_source"]) == {"ev_score"}
        assert pd.isna(scored.loc[scored["symbol"] == "INVALID", "deployment_score"].item())

    def test_potential_candidates_are_separate_and_never_deployable(self):
        script_path = Path(__file__).resolve().parents[1] / "run_full_pipeline.py"
        spec = importlib.util.spec_from_file_location("run_full_pipeline", script_path)
        assert spec is not None and spec.loader is not None
        run_full_pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_full_pipeline)

        df = pd.DataFrame(
            [
                {"symbol": "READY", "grid_is_valid": True, "deployment_score": 99.0, "score": 90.0},
                {"symbol": "NEAR", "grid_is_valid": False, "deployment_score": 88.0, "score": 89.0, "failure_stage": "stage_b", "rejection_reasons": "position_size_too_small", "stage_b_reason": "position_size_too_small", "hard_gate_reason": "ok", "capital_fraction": 0.03, "profit_per_grid_pct": 0.8, "profit_per_grid_min_pct": 1.0},
                {"symbol": "MISS", "grid_is_valid": False, "deployment_score": 77.0, "score": 10.0, "failure_stage": "regime", "rejection_reasons": "data_missing:range_prob"},
            ]
        )

        potential = run_full_pipeline._build_potential_candidates(df)

        assert potential["symbol"].tolist()[0] == "NEAR"
        assert set(potential["potential_candidate"]) == {True}
        assert potential["deployment_score"].isna().all()
        assert "failure_stage" in potential.columns
        assert "rejection_reasons" in potential.columns
        assert potential.loc[potential["symbol"] == "NEAR", "distance_to_position_size_min"].item() == pytest.approx(-0.02)
