"""
Unit tests for enrich_grid_params score threshold enforcement.

Tests call the actual production enrich_with_grid_params() function
with mocked external dependencies (BinanceClient, HMM model).
"""

import asyncio
import copy
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from neutralgrid.core.config import get_config
from neutralgrid.grid.spacing_profile import (
    build_winner_iqr_profile,
    score_candidates_against_winner_iqr,
)
from neutralgrid.scanner.enrich_grid_params import (
    _merge_kline_sets,
    _preserve_scan_inputs,
    enrich_with_grid_params,
    EnrichConfig,
)
from neutralgrid.training.scanner_integration import build_feature_snapshot


def _run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


def _make_candidates(symbols_scores: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a minimal candidate DataFrame like scan_top_symbols produces."""
    return pd.DataFrame({
        "symbol": [s for s, _ in symbols_scores],
        "score": [sc for _, sc in symbols_scores],
    })


def _mock_pattern_profile():
    """Create a minimal PatternProfile mock."""
    mock = MagicMock()
    mock.features = ["adx_1h", "adx_15m"]
    mock.means = {"adx_1h": 25.0, "adx_15m": 20.0}
    mock.stds = {"adx_1h": 10.0, "adx_15m": 8.0}
    mock.q10 = {"adx_1h": 15.0, "adx_15m": 12.0}
    mock.q90 = {"adx_1h": 40.0, "adx_15m": 35.0}
    return mock


def test_scan_inputs_preserved_before_refresh_overwrites_canonical_fields():
    df = pd.DataFrame(
        [{
            "symbol": "BTCUSDT",
            "range_prob": 0.21,
            "trend_prob": 0.34,
            "survival_prob": 0.56,
            "micro_osc_score": 0.78,
        }]
    )

    preserved = _preserve_scan_inputs(df)

    assert preserved.loc[0, "scan_range_prob"] == pytest.approx(0.21)
    assert preserved.loc[0, "scan_trend_prob"] == pytest.approx(0.34)
    assert preserved.loc[0, "scan_survival_prob"] == pytest.approx(0.56)
    assert preserved.loc[0, "scan_micro_osc_score"] == pytest.approx(0.78)
    assert preserved.loc[0, "range_prob"] == pytest.approx(0.21)


def test_kline_refresh_adds_missing_rows_and_prefers_refreshed_duplicates():
    cached = [
        [1000, "1", "2", "0.5", "1.5"],
        [2000, "2", "3", "1.5", "2.5"],
    ]
    refreshed = [
        [2000, "2", "4", "1.5", "3.5"],
        [3000, "3", "5", "2.5", "4.5"],
    ]

    merged = _merge_kline_sets(cached, refreshed)

    assert [row[0] for row in merged] == [1000, 2000, 3000]
    assert merged[1][2] == "4"


@pytest.fixture
def mock_client():
    """Create a mock BinanceClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_profile():
    return _mock_pattern_profile()


def test_below_threshold_marked_invalid(mock_client, mock_profile):
    """Rows below score threshold should be marked grid_is_valid=False with reason."""
    df = _make_candidates([
        ("ABOVE1", 60.0),
        ("ABOVE2", 55.0),
        ("BELOW1", 49.0),
        ("BELOW2", 45.0),
    ])
    cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
    )

    # Mock ensure_hmm_model to avoid network calls
    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        # Mock RegimeValidator and GridCalculator to return valid results for above-threshold
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls:
            mock_validator = MagicMock()
            mock_rv_cls.return_value = mock_validator

            # Make validator return invalid results (simplifies test — focus on threshold logic)
            mock_vres = MagicMock()
            mock_vres.is_valid = False
            mock_vres.tf_1h = MagicMock(is_valid=False, reason="test_invalid", checks={})
            mock_vres.tf_15m = None
            mock_vres.tf_5m = None
            mock_vres.range_prob = None
            mock_vres.trend_prob = None
            mock_vres.survival_prob = None
            mock_vres.hurst_exponent = None
            mock_vres.ou_halflife = None
            mock_vres.regime_utility = None
            mock_vres.data_quality_passed = True
            mock_vres.conditional_tail_risk_enhanced = {}
            mock_vres.tail_correction_applied = False
            mock_vres.volatility_tier_kurtosis_aware = None
            # Top-level HMM lineage fields (15m migration)
            mock_vres.regime_conf = None
            mock_vres.posterior_mode = None
            mock_vres.hmm_artifact_version = None
            mock_vres.hmm_pipeline_version = None
            mock_vres.hmm_calibration_provenance = None
            mock_vres.persistence_prob = None
            mock_vres.posteriors = None
            mock_vres.hmm_trained_at_utc = None
            mock_validator.validate.return_value = mock_vres

            mock_client.get_all_market_data = AsyncMock(return_value={})

            result = _run_async(enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            ))

    # Below-threshold rows should be invalid with threshold reason
    below = result[result["score"] < 50.0]
    for _, row in below.iterrows():
        assert not bool(row["grid_is_valid"])
        assert "score_below_threshold" in str(row["grid_reason"])
        assert row["failure_stage"] == "score_threshold"
        assert row["rejection_reasons"] == "score_below_threshold"


def test_below_threshold_params_cleared(mock_client, mock_profile):
    """Pre-existing grid params should be cleared for below-threshold rows."""
    df = _make_candidates([("SEIUSDT", 48.99)])
    # Simulate pre-populated grid params
    df["grid_lower"] = 0.118
    df["grid_upper"] = 0.122
    df["num_grids"] = 5
    df["grid_is_valid"] = True
    df["grid_reason"] = "ok"

    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator"):
            result = _run_async(enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            ))

    row = result.iloc[0]
    assert not row["grid_is_valid"]
    assert pd.isna(row["grid_lower"]) or row["grid_lower"] is None
    assert pd.isna(row["grid_upper"]) or row["grid_upper"] is None
    assert "score_below_threshold" in str(row["grid_reason"])
    assert row["failure_stage"] == "score_threshold"


@pytest.mark.parametrize(
    ("stage_b_value", "expected_valid"),
    [
        (None, False),
        (False, False),
        (True, True),
        (np.bool_(True), True),
    ],
)
def test_enrichment_initialization_is_warning_free_and_preserves_stage_b_booleans(
    mock_client,
    mock_profile,
    stage_b_value,
    expected_valid,
):
    """Batch column creation must not alter the below-threshold gate contract."""
    df = pd.DataFrame(
        {
            "symbol": ["LOWUSDT"],
            "score": [1.0],
            "grid_is_valid": [True],
            "grid_reason": ["ok"],
            "failure_stage": ["approved"],
            "stage_b_approved": [stage_b_value],
        }
    )
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        warnings.simplefilter("error", FutureWarning)
        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            )
        )

    row = result.iloc[0]
    assert bool(row["grid_is_valid"]) is expected_valid
    if expected_valid:
        assert row["failure_stage"] == "approved"
        assert row["grid_reason"] == "ok"
    else:
        assert row["failure_stage"] == "score_threshold"
        assert row["grid_reason"] == "score_below_threshold"


@pytest.mark.parametrize("invalid_stage_b_value", ["False", "unexpected"])
def test_stage_b_text_values_fail_closed(
    mock_client,
    mock_profile,
    invalid_stage_b_value,
):
    """Text must not become truthy and override the below-threshold gate."""
    df = pd.DataFrame(
        {
            "symbol": ["LOWUSDT"],
            "score": [1.0],
            "grid_is_valid": [True],
            "stage_b_approved": [invalid_stage_b_value],
        }
    )
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with pytest.raises(TypeError):
        _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            )
        )


def test_above_threshold_not_affected(mock_client, mock_profile):
    """Rows at or above threshold should NOT get threshold rejection reason."""
    df = _make_candidates([
        ("ABOVE1", 60.0),
        ("EXACTLY50", 50.0),
        ("BELOW1", 49.9),
    ])
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls:
            mock_validator = MagicMock()
            mock_rv_cls.return_value = mock_validator
            mock_vres = MagicMock()
            mock_vres.is_valid = False
            mock_vres.tf_1h = MagicMock(is_valid=False, reason="test_invalid", checks={})
            mock_vres.tf_15m = None
            mock_vres.tf_5m = None
            mock_vres.range_prob = None
            mock_vres.trend_prob = None
            mock_vres.survival_prob = None
            mock_vres.hurst_exponent = None
            mock_vres.ou_halflife = None
            mock_vres.regime_utility = None
            mock_vres.data_quality_passed = True
            mock_vres.conditional_tail_risk_enhanced = {}
            mock_vres.tail_correction_applied = False
            mock_vres.volatility_tier_kurtosis_aware = None
            # Top-level HMM lineage fields (15m migration)
            mock_vres.regime_conf = None
            mock_vres.posterior_mode = None
            mock_vres.hmm_artifact_version = None
            mock_vres.hmm_pipeline_version = None
            mock_vres.hmm_calibration_provenance = None
            mock_vres.persistence_prob = None
            mock_vres.posteriors = None
            mock_vres.hmm_trained_at_utc = None
            mock_validator.validate.return_value = mock_vres
            mock_client.get_all_market_data = AsyncMock(return_value={})

            result = _run_async(enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            ))

    above_rows = result[result["score"] >= 50.0]
    for _, row in above_rows.iterrows():
        reason = str(row.get("grid_reason", ""))
        assert "score_below_threshold" not in reason, \
            f"Above-threshold row {row['symbol']} has threshold reason: {reason}"

    below_row = result[result["score"] < 50.0].iloc[0]
    assert "score_below_threshold" in str(below_row["grid_reason"])
    assert below_row["failure_stage"] == "score_threshold"


def test_custom_threshold(mock_client, mock_profile):
    """Custom threshold values should work correctly."""
    df = _make_candidates([
        ("A", 70.0),
        ("B", 60.0),
        ("C", 50.0),
    ])
    cfg = EnrichConfig(score_threshold=65.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator"):
            mock_client.get_all_market_data = AsyncMock(return_value={})
            result = _run_async(enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            ))

    # With threshold=65, rows B (60) and C (50) should be below threshold
    below = result[result["score"] < 65.0]
    for _, row in below.iterrows():
        assert not bool(row["grid_is_valid"])
        assert "score_below_threshold" in str(row["grid_reason"])
        assert row["failure_stage"] == "score_threshold"


def test_invalid_regime_preserves_utility_score_alias(mock_client, mock_profile):
    df = _make_candidates([("UTILITYUSDT", 60.0)])
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls:
            mock_validator = MagicMock()
            mock_rv_cls.return_value = mock_validator
            mock_vres = MagicMock()
            mock_vres.is_valid = False
            mock_vres.tf_1h = MagicMock(is_valid=False, reason="test_invalid", checks={})
            mock_vres.tf_15m = None
            mock_vres.tf_5m = None
            mock_vres.range_prob = 0.62
            mock_vres.trend_prob = 0.38
            mock_vres.survival_prob = 0.74
            mock_vres.hurst_exponent = 0.47
            mock_vres.ou_halflife = 12.0
            mock_vres.regime_utility = 1.25
            mock_vres.data_quality_passed = True
            mock_vres.conditional_tail_risk_enhanced = {}
            mock_vres.tail_correction_applied = False
            mock_vres.volatility_tier_kurtosis_aware = None
            # Top-level HMM lineage fields (15m migration)
            mock_vres.regime_conf = None
            mock_vres.posterior_mode = None
            mock_vres.hmm_artifact_version = None
            mock_vres.hmm_pipeline_version = None
            mock_vres.hmm_calibration_provenance = None
            mock_vres.persistence_prob = None
            mock_vres.posteriors = None
            mock_vres.hmm_trained_at_utc = None
            mock_validator.validate.return_value = mock_vres
            mock_client.get_all_market_data = AsyncMock(return_value={})

            result = _run_async(enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            ))

    row = result.iloc[0]
    assert row["regime_utility"] == pytest.approx(1.25)
    assert row["utility_score"] == pytest.approx(1.25)


def _valid_vres(
    *,
    range_prob: float = 0.72,
    trend_prob: float = 0.18,
    survival_prob: float = 0.82,
    is_valid: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_valid=is_valid,
        tf_1h=SimpleNamespace(
            is_valid=True,
            reason=None,
            checks={"range_prob": range_prob, "trend_prob": trend_prob},
        ),
        tf_15m=None,
        tf_5m=None,
        range_prob=range_prob,
        trend_prob=trend_prob,
        survival_prob=survival_prob,
        hurst_exponent=0.46,
        ou_halflife=12.0,
        regime_utility=1.15,
        data_quality_passed=True,
        conditional_tail_risk={},
        conditional_tail_risk_enhanced={},
        tail_correction_applied=False,
        volatility_tier=None,
        volatility_tier_kurtosis_aware=None,
        current_price=100.0,
        range_high=102.0,
        range_low=98.0,
        atr_1m=0.2,
        # Top-level HMM lineage fields (15m migration)
        regime_conf=None,
        posterior_mode=None,
        hmm_artifact_version=None,
        hmm_pipeline_version=None,
        hmm_calibration_provenance=None,
        persistence_prob=None,
        posteriors=None,
        hmm_trained_at_utc=None,
    )


def _valid_grid(*, capital_fraction: float | None = 0.60, leverage: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        is_valid=True,
        reason="ok",
        grid_lower=99.0,
        grid_upper=101.0,
        num_grids=4,
        grid_spacing_pct=0.5,
        profit_per_grid_pct=1.2,
        profit_per_grid_min_pct=1.0,
        profit_per_grid_max_pct=1.4,
        leverage=leverage,
        stop_loss_pct=3.0,
        capital_fraction=capital_fraction,
        sizing_reason="legacy_grid_fraction",
    )


def test_enrich_final_payload_promotes_canonical_hmm_fields_and_preserves_scan_meta(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["CANONUSDT"],
            "score": [62.0],
            "range_prob": [0.61],
            "trend_prob": [0.91],
            "scan_meta_prob": [0.13],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres()
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert row["hmm_range_prob"] == pytest.approx(0.72)
    assert row["hmm_trend_prob"] == pytest.approx(0.18)
    assert row["range_prob"] == pytest.approx(0.72)
    assert row["trend_prob"] == pytest.approx(0.18)
    assert row["scan_meta_prob"] == pytest.approx(0.13)
    assert row["meta_prob"] == pytest.approx(0.70)
    assert row["meta_prob_source"] == "enrich"

    snapshot = build_feature_snapshot("CANONUSDT", cast(dict, row.to_dict()))
    assert snapshot.range_prob == pytest.approx(0.72)
    assert snapshot.trend_prob == pytest.approx(0.18)


def test_pre_reject_low_range_prob_is_explicit(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["LOWRPUSDT"],
            "score": [60.0],
            "range_prob": [0.19],
            "survival_prob": [0.80],
        }
    )
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=cfg,
            )
        )

    row = result.iloc[0]
    assert not row["grid_is_valid"]
    assert row["failure_stage"] == "pre_reject"
    assert row["grid_reason"] == "pre_reject:low_range_prob(0.190<0.20)"
    assert row["rejection_reasons"] == "pre_reject:low_range_prob(0.190<0.20)"


def test_stale_scan_low_range_defers_to_fresh_regime_validation(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["STALERPUSDT"],
            "score": [60.0],
            "range_prob": [0.19],
            "survival_prob": [0.80],
        }
    )
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)
    scan_cache = {
        "STALERPUSDT": {
            "cached_at_utc": datetime.now(timezone.utc) - timedelta(seconds=301),
        }
    }

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls:
            mock_validator = MagicMock()
            mock_rv_cls.return_value = mock_validator

            mock_vres = _valid_vres(range_prob=0.19, survival_prob=0.80, is_valid=False)
            mock_vres.tf_1h = SimpleNamespace(
                is_valid=False,
                reason="fresh_low_range",
                checks={"range_prob": 0.19},
            )
            mock_validator.validate.return_value = mock_vres
            mock_client.get_all_market_data = AsyncMock(return_value={})

            result = _run_async(
                enrich_with_grid_params(
                    df_candidates=df,
                    client=mock_client,
                    pattern_profile=mock_profile,
                    cfg=cfg,
                    scan_data_cache=scan_cache,
                )
            )

    row = result.iloc[0]
    assert bool(row["pre_reject_would_reject"]) is True
    assert row["pre_reject_reason"] == "pre_reject:low_range_prob(0.190<0.20)"
    assert bool(row["scan_cache_stale"]) is True
    assert float(row["scan_cache_age_seconds"]) >= 300.0
    assert row["failure_stage"] == "regime"
    assert row["grid_reason"] == "fresh_low_range"
    assert "low_range_prob(0.190)" in str(row["rejection_reasons"])


def test_range_prob_at_new_floor_is_not_pre_rejected(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["EDGEFLOORUSDT"],
            "score": [60.0],
            "range_prob": [0.20],
            "survival_prob": [0.80],
        }
    )
    cfg = EnrichConfig(score_threshold=50.0, max_symbols=5, concurrency=1)

    with patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock):
        with patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls:
            mock_validator = MagicMock()
            mock_rv_cls.return_value = mock_validator

            mock_vres = MagicMock()
            mock_vres.is_valid = False
            mock_vres.tf_1h = MagicMock(is_valid=False, reason="test_invalid", checks={})
            mock_vres.tf_15m = None
            mock_vres.tf_5m = None
            mock_vres.range_prob = 0.20
            mock_vres.trend_prob = None
            mock_vres.survival_prob = 0.80
            mock_vres.hurst_exponent = None
            mock_vres.ou_halflife = None
            mock_vres.regime_utility = None
            mock_vres.data_quality_passed = True
            mock_vres.conditional_tail_risk_enhanced = {}
            mock_vres.tail_correction_applied = False
            mock_vres.volatility_tier_kurtosis_aware = None
            # Top-level HMM lineage fields (15m migration)
            mock_vres.regime_conf = None
            mock_vres.posterior_mode = None
            mock_vres.hmm_artifact_version = None
            mock_vres.hmm_pipeline_version = None
            mock_vres.hmm_calibration_provenance = None
            mock_vres.persistence_prob = None
            mock_vres.posteriors = None
            mock_vres.hmm_trained_at_utc = None
            mock_validator.validate.return_value = mock_vres
            mock_client.get_all_market_data = AsyncMock(return_value={})

            result = _run_async(
                enrich_with_grid_params(
                    df_candidates=df,
                    client=mock_client,
                    pattern_profile=mock_profile,
                    cfg=cfg,
                )
            )

    row = result.iloc[0]
    assert row["failure_stage"] == "regime"
    assert "pre_reject:low_range_prob" not in str(row["grid_reason"])
    assert "low_range_prob(0.200)" not in str(row["rejection_reasons"])


def test_scan_low_survival_is_audit_only_until_fresh_validation(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["LOWSURVUSDT"],
            "score": [40.0],
            "range_prob": [0.60],
            "survival_prob": [0.30],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres(
            range_prob=0.62,
            survival_prob=0.82,
            is_valid=True,
        )
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=0.60,
            leverage=10,
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.promotion_status = "pass"
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert bool(row["below_threshold_tag"]) is True
    assert bool(row["pre_reject_would_reject"]) is True
    assert row["pre_reject_reason"] == "pre_reject:low_survival(0.300<0.40)"
    assert row["failure_stage"] == "approved"
    assert bool(row["grid_is_valid"]) is True
    assert row["stage_b_approved"] is True
    assert row["meta_prob_source"] == "enrich"


def test_no_mean_reversion_halflife_fails_closed_for_meta_feature(mock_client, mock_profile):
    """ERR-065: halflife=inf (no mean-reversion) must FAIL CLOSED — the meta
    feature stays absent so the probe is skipped (meta_prob_source='missing'),
    never imputed with the ou_halflife_max_bars cap (48 is MEDIAN behavior in
    the raw, uncapped training distribution)."""
    df = pd.DataFrame(
        {
            "symbol": ["INFHALFUSDT"],
            "score": [62.0],
            "range_prob": [0.90],
            "survival_prob": [0.85],
            "micro_osc_score": [0.90],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    def _assert_meta_features(features: dict[str, object]) -> list[str]:
        assert features["ou_halflife"] is None
        return ["ou_halflife"]

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_vres = _valid_vres(range_prob=0.01, trend_prob=0.99, survival_prob=0.85, is_valid=False)
        mock_vres.ou_halflife = None
        mock_vres.tf_1h = SimpleNamespace(
            is_valid=False,
            reason="stochastic_regime_fail(halflife=inf (no mean-reversion))",
            checks={},
        )
        mock_rv_cls.return_value.validate.return_value = mock_vres
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=0.60,
            leverage=10,
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.promotion_status = "pass"
        meta_labeler.get_missing_feature_names.side_effect = _assert_meta_features
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert pd.isna(row["ou_halflife"])
    assert pd.isna(row["ou_halflife_raw"])
    assert row["ou_halflife_feature_reason"] == "non_mean_reverting_fail_closed"
    ps_kwargs = mock_ps_cls.return_value.compute.call_args.kwargs
    assert ps_kwargs["range_prob"] == pytest.approx(0.85)
    assert ps_kwargs["trend_prob"] == pytest.approx(0.15)
    assert row["ps_archetype"] == "micro_osc_survival"
    # no authoritative meta_prob -> Kelly inactive -> PositionSizer owns the
    # volatility penalty (range-derived proxy, not the Kelly vol target)
    assert row["ps_volatility_source"] == "position_sizer"
    # fail-closed: probe skipped, meta_prob stays missing for this row
    assert pd.isna(row["meta_prob"])
    assert row["meta_prob_source"] == "missing"


def test_micro_osc_dynamic_profit_floor_uses_survival_archetype(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["MICROFLOORUSDT"],
            "score": [62.0],
            "range_prob": [0.03],
            "survival_prob": [0.90],
            "micro_osc_score": [0.90],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = True
    cfg_obj.micro_osc.enabled = True
    cfg_obj.micro_osc.min_score = 0.45
    cfg_obj.micro_osc.min_survival_prob = 0.60

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.50,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres(
            range_prob=0.03,
            trend_prob=0.97,
            survival_prob=0.90,
            is_valid=False,
        )
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=None,
            leverage=10,
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.promotion_status = "pass"
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    floor_kwargs = mock_ms_cls.return_value.compute_dynamic_profit_floor.call_args.kwargs
    assert floor_kwargs["trend_prob"] == pytest.approx(0.10)
    assert floor_kwargs["survival_prob"] == pytest.approx(0.90)
    assert row["micro_floor_archetype"] == "micro_osc_survival"
    assert row["micro_floor_trend_prob_input"] == pytest.approx(0.10)
    assert row["micro_floor_survival_prob_input"] == pytest.approx(0.90)
    assert row["edge_tier_chosen"] == "MICRO_OSC_DENSE"
    assert bool(row["grid_is_valid"]) is True
    assert row["failure_stage"] == "approved"


def test_runtime_capital_and_live_sizers_are_authoritative(mock_client, mock_profile):
    df = pd.DataFrame({"symbol": ["FLOWUSDT"], "score": [62.0]})
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )
    tos_calls: list[float] = []

    def _tos_compute(**kwargs):
        tos_calls.append(float(kwargs["position_size_usdt"]))
        return SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres()
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.side_effect = _tos_compute
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert row["failure_stage"] == "approved"
    assert row["hard_gate_passed"] is True
    assert row["stage_b_approved"] is True
    assert row["meta_prob"] == pytest.approx(0.70)
    assert row["meta_prob_authority"] == "diagnostic_only"
    assert row["capital_fraction"] == pytest.approx(0.5)
    mock_kelly.assert_not_called()
    assert tos_calls == [8000.0]


def test_promoted_meta_labeler_is_authoritative_and_drives_kelly(mock_client, mock_profile):
    """FASTWIN-01: a PROMOTED meta-labeler (promotion_status == "pass") is
    authoritative — deployment_meta_prob = meta_prob, so generalized Kelly sizing
    fires and the row is labeled "authoritative". The unpromoted counterpart
    (test_runtime_capital_and_live_sizers_are_authoritative) keeps it
    diagnostic-only with Kelly not called; this is the only difference.

    ERR-090: generalized Kelly is opt-in (default False) since 2026-07-12; this
    test enables it explicitly to exercise the promoted-authority path."""
    df = pd.DataFrame({"symbol": ["FLOWUSDT"], "score": [62.0]})
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False
    cfg_obj.position_sizing.enable_generalized_kelly = True

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres()
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.2,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": 0.4,
            "fractional_kelly": 0.2,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.5,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.promotion_status = "pass"  # FASTWIN-01: promoted -> authoritative
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.70

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert row["meta_prob"] == pytest.approx(0.70)
    assert row["meta_prob_authority"] == "authoritative"
    # Promoted -> deployment_meta_prob = meta_prob -> generalized Kelly fires.
    mock_kelly.assert_called()


def test_err090_default_kelly_disabled_positionsizer_owns_sizing(mock_client, mock_profile):
    """ERR-090: with the DEFAULT config (enable_generalized_kelly=False), a
    promoted/authoritative meta-labeler must NOT trigger generalized Kelly;
    sizing authority is the risk-budget PositionSizer (pre-2026-06-24
    behavior), so capital_fraction == ps fraction and the row is not zeroed
    into Stage B position_too_small by a negative Kelly edge."""
    df = pd.DataFrame({"symbol": ["FLOWUSDT"], "score": [62.0]})
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False
    # Default must already be False (ERR-090); assert rather than set.
    assert cfg_obj.position_sizing.enable_generalized_kelly is False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres()
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=75.0,
            grid_cross_frequency=1.2,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.promotion_status = "pass"  # promoted, still must NOT size via Kelly
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.44  # below old Kelly break-even ~0.63

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert row["meta_prob"] == pytest.approx(0.44)
    mock_kelly.assert_not_called()
    # PositionSizer owns sizing: fraction 0.5, NOT zeroed by negative Kelly.
    assert row["capital_fraction"] == pytest.approx(0.5)
    assert "gen_kelly_disabled" in str(row["sizing_reason"])
    # PositionSizer keeps its own volatility penalty (not the Kelly-owned target).
    assert row["ps_volatility_source"] == "position_sizer"


def test_viability_and_negative_kelly_are_non_terminal(mock_client, mock_profile):
    df = pd.DataFrame({"symbol": ["TAOUSDT"], "score": [63.0]})
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    def _approve(**kwargs):
        assert kwargs["position_size_fraction"] == pytest.approx(0.6)
        return SimpleNamespace(approved=True, reason="ok")

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.generalized_kelly_details") as mock_kelly:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres()
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (False, "insufficient_liquidity")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.6,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=70.0,
            grid_cross_frequency=1.0,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.side_effect = _approve
        mock_kelly.return_value = {
            "payoff_ratio_b": 1.0,
            "avg_win_pct": 2.0,
            "avg_loss_pct": 1.0,
            "raw_fraction": -0.2,
            "fractional_kelly": 0.0,
            "fractional_multiplier": 0.5,
            "fractional_mode": "fixed",
            "drawdown_scale": 1.0,
            "volatility_scale": 1.0,
            "profile_samples": 20,
            "profile_scope": "global",
            "profile_scope_samples": 20.0,
            "sweep_objective_growth": 0.0,
            "sweep_estimated_drawdown_pct": 0.0,
            "sweep_feasible_points": 1,
            "sweep_evaluated_points": 1,
            "final_fraction": 0.0,
        }
        meta_labeler = MagicMock()
        meta_labeler.is_trained = True
        meta_labeler.get_missing_feature_names.return_value = []
        meta_labeler.predict_proba.return_value = 0.40

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
                meta_labeler=meta_labeler,
            )
        )

    row = result.iloc[0]
    assert row["micro_viable"] is False
    assert row["micro_reason"] == "insufficient_liquidity"
    assert row["hard_gate_passed"] is True
    assert row["ps_fraction"] == pytest.approx(0.6)
    assert row["meta_prob"] == pytest.approx(0.40)
    assert row["meta_prob_authority"] == "diagnostic_only"
    assert row["capital_fraction"] == pytest.approx(0.6)
    assert row["stage_b_approved"] is True
    assert row["failure_stage"] == "approved"
    mock_kelly.assert_not_called()


def test_discovery_mode_preserves_stage_b_verdict_but_marks_geometry_valid(mock_client, mock_profile):
    df = pd.DataFrame({"symbol": ["BOOTUSDT"], "score": [22.0]})
    enrich_cfg = EnrichConfig(
        score_threshold=20.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
        discovery_mode=True,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres(is_valid=False)
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(capital_fraction=0.60, leverage=10)
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=False,
            reason="profit_per_grid_below_min(0.20%<0.30%)",
            rejection_codes=("profit_per_grid_below_min",),
            details={},
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=20.0,
            grid_cross_frequency=1.0,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(
            approved=False,
            reason="hard_gate_fail:profit_per_grid_below_min",
        )

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
            )
        )

    row = result.iloc[0]
    assert bool(row["discovery_mode"]) is True
    assert bool(row["grid_is_valid"]) is True
    assert row["failure_stage"] == "discovery"
    assert bool(row["regime_would_reject"]) is True
    assert row["regime_rejection_reasons"] == "regime_invalid"
    assert row["hard_gate_passed"] is False
    assert bool(row["hard_gate_would_reject"]) is True
    assert bool(row["stage_b_approved"]) is False
    assert str(row["grid_reason"]).startswith("discovery_mode:stage_b_rejected:")
    assert pd.notna(row["grid_lower"])
    assert pd.notna(row["grid_upper"])
    assert pd.notna(row["num_grids"])


def test_discovery_mode_fills_geometry_after_regime_reject(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["RANGEUSDT"],
            "score": [24.0],
            "range_prob": [0.22],
            "survival_prob": [0.80],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=20.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
        discovery_mode=True,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    vres = _valid_vres(range_prob=0.22, is_valid=False)
    vres.range_high = None
    vres.range_low = None
    vres.current_price = None
    vres.atr_1m = None

    kline_frame = pd.DataFrame(
        {
            "high": [101.0 + i * 0.01 for i in range(60)],
            "low": [99.0 + i * 0.01 for i in range(60)],
            "close": [100.0 + i * 0.01 for i in range(60)],
        }
    )
    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls:
        mock_client.get_all_market_data = AsyncMock(
            return_value={"klines": {"15m": kline_frame, "1m": kline_frame}}
        )
        mock_validator = mock_rv_cls.return_value
        mock_validator.validate.return_value = vres
        mock_validator.check_range_quality.return_value = SimpleNamespace(
            passed=True,
            reason=None,
            metrics={
                "range_high": 103.0,
                "range_low": 97.0,
                "current_price": 100.5,
            },
        )
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=0.60,
            leverage=10,
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=70.0,
            grid_cross_frequency=1.0,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
            )
        )

    grid_vres = mock_calc_cls.return_value.generate_params.call_args.args[0]
    assert grid_vres.is_valid is True
    assert grid_vres.range_high == pytest.approx(103.0)
    assert grid_vres.range_low == pytest.approx(97.0)
    assert grid_vres.current_price == pytest.approx(100.5)
    assert grid_vres.atr_1m is not None

    row = result.iloc[0]
    assert bool(row["discovery_geometry_filled"]) is True
    assert row["discovery_geometry_reason"] == "range_quality_after_regime_reject"
    assert bool(row["grid_is_valid"]) is True
    assert row["failure_stage"] == "approved"


def test_micro_osc_bypass_reuses_geometry_for_grid_generation(mock_client, mock_profile):
    df = pd.DataFrame(
        {
            "symbol": ["BYPASSUSDT"],
            "score": [24.0],
            "range_prob": [0.22],
            "survival_prob": [0.80],
            "micro_osc_score": [0.55],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=45.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False
    cfg_obj.micro_osc.enabled = True
    cfg_obj.micro_osc.min_score = 0.45
    cfg_obj.micro_osc.min_survival_prob = 0.60

    vres = _valid_vres(range_prob=0.22, survival_prob=0.80, is_valid=False)

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_validator = mock_rv_cls.return_value
        mock_validator.validate.return_value = vres
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=0.60,
            leverage=10,
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (0.50, "mid", {"base": 0.50})
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=True, reason="hard_gate_pass", rejection_codes=(), details={}
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=70.0,
            grid_cross_frequency=1.0,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(approved=True, reason="ok")

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
            )
        )

    grid_vres = mock_calc_cls.return_value.generate_params.call_args.args[0]
    assert grid_vres.is_valid is True
    assert grid_vres.range_high == pytest.approx(102.0)
    assert grid_vres.range_low == pytest.approx(98.0)
    assert grid_vres.current_price == pytest.approx(100.0)
    assert grid_vres.atr_1m == pytest.approx(0.2)

    row = result.iloc[0]
    assert bool(row["micro_osc_bypass"]) is True
    assert bool(row["regime_would_reject"]) is True
    assert bool(row["grid_is_valid"]) is True
    assert row["failure_stage"] == "approved"
    assert bool(row["discovery_geometry_filled"]) is False


def _winner_iqr_rows() -> pd.DataFrame:
    rows = []
    for i, range_size in enumerate(range(1, 13), start=1):
        grids_count = 8 + i * 2
        spacing_pct = 0.60 + i * 0.01
        # Construct price bounds consistent with the geometric identity so the
        # row reconstructs to mode == "geometric" via _infer_mode (which mirrors
        # _bot_data_extractor_core.py).
        n_intervals = grids_count - 1
        r = 1.0 + spacing_pct / 100.0
        price_low = 1.0
        price_high = price_low * (r ** n_intervals)
        rows.append(
            {
                "strategy_id": i,
                "symbol": f"WIN{i}USDT",
                "duration_hours": 6.0,
                "pnl_pct": 2.0,
                "grids_count": grids_count,
                "range_size_pct": float(range_size),
                "grid_spacing_pct": spacing_pct,
                "profit_per_grid_pct": 0.50 + i * 0.01,
                "price_range_low": price_low,
                "price_range_high": price_high,
            }
        )
    rows.append(
        {
            "strategy_id": 99,
            "symbol": "DROPUSDT",
            "duration_hours": 6.0,
            "pnl_pct": 2.0,
            "grids_count": None,
            "range_size_pct": 3.0,
            "grid_spacing_pct": 0.70,
            "profit_per_grid_pct": 0.60,
            "price_range_low": 1.0,
            "price_range_high": 1.07,
        }
    )
    rows.append(
        {
            "strategy_id": 100,
            "symbol": "LOSERUSDT",
            "duration_hours": 6.0,
            "pnl_pct": 0.5,
            "grids_count": 100,
            "range_size_pct": 4.0,
            "grid_spacing_pct": 0.10,
            "profit_per_grid_pct": 0.10,
            "price_range_low": 1.0,
            "price_range_high": 1.10,
        }
    )
    return pd.DataFrame(rows)


def test_winner_iqr_profile_rejects_duplicate_strategy_ids() -> None:
    winners = _winner_iqr_rows()
    winners.loc[1, "strategy_id"] = winners.loc[0, "strategy_id"]

    with pytest.raises(ValueError, match="Duplicate strategy_id"):
        build_winner_iqr_profile(winners, deciles=2)


def test_winner_iqr_scores_candidates_and_counts_dropped_geometry() -> None:
    profile = build_winner_iqr_profile(_winner_iqr_rows(), deciles=2, min_pool_n=8)

    candidates = pd.DataFrame(
        [
            {
                "symbol": "INSIDEUSDT",
                "hmm_winner_score_source": "calibrated",
                "range_size_pct": 3.0,
                "num_grids": 15,
                "grid_spacing_pct": 0.64,
                "profit_per_grid_pct": 0.54,
                "micro_round_trip_cost_pct": 0.05,
                "micro_min_profit_required_pct": 0.08,
            },
            {
                "symbol": "DROPCAND",
                "hmm_winner_score_source": "calibrated",
                "range_size_pct": None,
                "num_grids": 15,
                "grid_spacing_pct": 0.64,
                "profit_per_grid_pct": 0.54,
            },
            {
                "symbol": "UNCALUSDT",
                "hmm_winner_score_source": "raw",
                "range_size_pct": 3.0,
                "num_grids": 15,
                "grid_spacing_pct": 0.64,
                "profit_per_grid_pct": 0.54,
            },
        ]
    )

    scored, summary = score_candidates_against_winner_iqr(
        candidates,
        profile,
        min_grids=5,
        max_grids=175,
    )

    assert profile.winner_rows == 13
    assert profile.geometry_rows == 12
    assert profile.dropped_geometry_rows == 1
    assert summary["input_rows"] == 3
    assert summary["calibrated_rows"] == 2
    assert summary["geometry_rows"] == 1
    assert summary["dropped_geometry_rows"] == 1

    row = scored.iloc[0]
    assert row["symbol"] == "INSIDEUSDT"
    assert bool(row["num_grids_in_iqr"]) is True
    assert bool(row["grid_spacing_in_iqr"]) is True
    assert bool(row["profit_per_grid_in_iqr"]) is True
    assert bool(row["tuple_in_iqr"]) is True
    assert row["winner_iqr_binding_constraint"] == "winner_iqr_aligned"


def test_winner_iqr_binding_classification_is_single_surface() -> None:
    profile = build_winner_iqr_profile(_winner_iqr_rows(), deciles=2, min_pool_n=8)
    candidates = pd.DataFrame(
        [
            {
                "symbol": "COSTUSDT",
                "range_size_pct": 3.0,
                "num_grids": 15,
                "grid_spacing_pct": 0.10,
                "profit_per_grid_pct": 0.54,
                "micro_round_trip_cost_pct": 0.105,
                "micro_min_profit_required_pct": 0.08,
            },
            {
                "symbol": "CAPUSDT",
                "range_size_pct": 9.0,
                "num_grids": 175,
                "grid_spacing_pct": 0.70,
                "profit_per_grid_pct": 0.60,
                "micro_round_trip_cost_pct": 0.05,
                "micro_min_profit_required_pct": 0.08,
            },
            {
                "symbol": "SPACINGUSDT",
                "range_size_pct": 9.0,
                "num_grids": 5,
                "grid_spacing_pct": 1.40,
                "profit_per_grid_pct": 1.20,
                "micro_round_trip_cost_pct": 0.05,
                "micro_min_profit_required_pct": 0.08,
            },
        ]
    )

    scored, summary = score_candidates_against_winner_iqr(
        candidates,
        profile,
        min_grids=5,
        max_grids=175,
        require_calibrated=False,
    )

    by_symbol = scored.set_index("symbol")
    assert by_symbol.loc["COSTUSDT", "winner_iqr_binding_constraint"] == "round_trip_cost_floor"
    assert by_symbol.loc["CAPUSDT", "winner_iqr_binding_constraint"] == "max_grids_cap"
    assert by_symbol.loc["SPACINGUSDT", "winner_iqr_binding_constraint"] == "spacing_target_at_min_grids"
    assert summary["binding_counts"] == {
        "round_trip_cost_floor": 1,
        "max_grids_cap": 1,
        "spacing_target_at_min_grids": 1,
    }




def test_hard_gate_rejection_records_validated_range_size_pct(mock_client, mock_profile):
    """ERR-073: the hard-gate rejection return must record the VALIDATED
    grid-derived range_size_pct (the value the Stage 12 meta probe consumed),
    overriding the scan-time estimate, so the recorded meta_prob stays
    reproducible from the row's own feature columns."""
    df = pd.DataFrame(
        {
            "symbol": ["HGUSDT"],
            "score": [60.0],
            "range_prob": [0.72],
            "survival_prob": [0.82],
            # Scan-time BB-based estimate — must be OVERRIDDEN in the output.
            "range_size_pct": [4.5],
        }
    )
    enrich_cfg = EnrichConfig(
        score_threshold=50.0,
        max_symbols=5,
        concurrency=1,
        capital_base_usdt=800.0,
        discovery_mode=False,
    )
    cfg_obj = copy.deepcopy(get_config())
    cfg_obj.edge_tier.enable = False

    ms_costs = SimpleNamespace(
        round_trip_cost_pct=0.20,
        spread_cost_pct=0.05,
        funding_cost_pct=0.05,
        sufficient_liquidity=True,
        min_profit_required_pct=0.30,
        extreme_funding=False,
    )

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg_obj), \
         patch("neutralgrid.scanner.enrich_grid_params.ensure_hmm_model", new_callable=AsyncMock), \
         patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
         patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator") as mock_rv_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
         patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls:
        mock_client.get_all_market_data = AsyncMock(return_value={})
        mock_rv_cls.return_value.validate.return_value = _valid_vres(is_valid=True)
        mock_calc_cls.return_value.generate_params.return_value = _valid_grid(
            capital_fraction=0.60, leverage=10
        )
        mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
        mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (
            0.50, "mid", {"base": 0.50}
        )
        mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
        mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
            passed=False,
            reason="profit_per_grid_below_min(0.20%<0.30%)",
            rejection_codes=("profit_per_grid_below_min",),
            details={},
        )
        mock_ps_cls.return_value.compute.return_value = SimpleNamespace(
            fraction=0.5,
            regime_confidence_scale=1.0,
            survival_scale=1.0,
            microstructure_scale=1.0,
            volatility_scale=1.0,
            portfolio_heat_scale=1.0,
            sizing_reason="unit_test",
        )
        mock_tos_cls.return_value.compute.return_value = SimpleNamespace(
            tos=20.0,
            grid_cross_frequency=1.0,
            mean_reversion_strength=0.8,
            range_containment=0.9,
        )
        mock_sb_cls.return_value.approve.return_value = SimpleNamespace(
            approved=False,
            reason="unreached",
        )

        result = _run_async(
            enrich_with_grid_params(
                df_candidates=df,
                client=mock_client,
                pattern_profile=mock_profile,
                cfg=enrich_cfg,
            )
        )

    row = result.iloc[0]
    assert bool(row["grid_is_valid"]) is False
    assert row["failure_stage"] == "hard_gate"
    assert row["hard_gate_passed"] is False
    assert bool(row["hard_gate_would_reject"]) is True
    # The probe-consumed, grid-derived value ((101-99)/100*100 = 2.0) must
    # override the scan-time estimate (4.5) — parity with the Stage 18 return.
    assert row["range_size_pct"] == pytest.approx(2.0)
    # Lock the pre-existing rejection-dict contract for the other grid features.
    assert row["num_grids"] == 4
    assert row["grid_spacing_pct"] == pytest.approx(0.5)
    assert row["profit_per_grid_pct"] == pytest.approx(1.2)
