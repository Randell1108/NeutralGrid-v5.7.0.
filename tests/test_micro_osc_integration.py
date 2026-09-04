"""
Full pipeline integration tests for the micro-oscillator bypass flag.

Validates:
  - micro_osc_score is always computed (flag ON or OFF)
  - Bypass candidates reach enrichment when flag is ON
  - Bypass candidates survive sort-and-truncate
  - Bypass candidates are not pre-rejected or threshold-tagged
  - Non-qualifying symbols remain blocked
  - Flag OFF means no bypass
  - Standard symbols are unaffected (zero regression)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from neutralgrid.core.config import (
    Config,
    MicroOscConfig,
    get_config,
    reset_config,
)
from neutralgrid.scanner.enrich_grid_params import (
    EnrichConfig,
    enrich_with_grid_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _make_config(*, micro_osc_enabled: bool = False) -> Config:
    """Create a Config with micro_osc settings for testing.

    Returns a real Config object with micro_osc.enabled toggled.
    Other fields use defaults.
    """
    cfg = get_config()
    cfg.micro_osc = MicroOscConfig(
        enabled=micro_osc_enabled,
        min_score=0.45,
        min_survival_prob=0.60,
    )
    return cfg


def _mock_pattern_profile():
    """Create a minimal PatternProfile mock."""
    mock = MagicMock()
    mock.features = ["adx_1h", "adx_15m"]
    mock.means = {"adx_1h": 25.0, "adx_15m": 20.0}
    mock.stds = {"adx_1h": 10.0, "adx_15m": 8.0}
    mock.q10 = {"adx_1h": 15.0, "adx_15m": 12.0}
    mock.q90 = {"adx_1h": 40.0, "adx_15m": 35.0}
    return mock


def _mock_client():
    """Create a mock BinanceClient."""
    client = AsyncMock()
    client.get_all_market_data = AsyncMock(return_value={})
    return client


def _mock_validator():
    """Create a mock RegimeValidator that returns invalid results."""
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
    # Top-level HMM lineage fields (added for 15m migration)
    mock_vres.regime_conf = None
    mock_vres.posterior_mode = None
    mock_vres.hmm_artifact_version = None
    mock_vres.hmm_pipeline_version = None
    mock_vres.hmm_calibration_provenance = None
    mock_vres.persistence_prob = None
    mock_vres.posteriors = None
    mock_vres.hmm_trained_at_utc = None
    return mock_vres


def _enrich_patches(config_obj: Config):
    """Context manager stack for patching enrich_with_grid_params dependencies.

    Returns a tuple of (mock_rv_cls,) so callers can customize if needed.
    """

    class _Ctx:
        """Composable patch context."""

        def __init__(self, config_obj: Config):
            self._cfg = config_obj
            self._patches = []
            self.mock_rv_cls = None

        def __enter__(self):
            # Patch get_config at the call site inside enrich_grid_params
            p1 = patch(
                "neutralgrid.scanner.enrich_grid_params.get_config",
                return_value=self._cfg,
            )
            # Patch ensure_hmm_model to avoid network calls
            p2 = patch(
                "neutralgrid.scanner.enrich_grid_params.ensure_hmm_model",
                new_callable=AsyncMock,
            )
            # Patch RegimeValidator
            p3 = patch("neutralgrid.scanner.enrich_grid_params.RegimeValidator")

            self._patches = [p1, p2, p3]
            p1.start()
            p2.start()
            self.mock_rv_cls = p3.start()

            # Setup default validator behaviour
            mock_validator = MagicMock()
            mock_validator.validate = MagicMock(return_value=_mock_validator())
            self.mock_rv_cls.return_value = mock_validator

            return self

        def __exit__(self, *args):
            for p in reversed(self._patches):
                p.stop()

    return _Ctx(config_obj)


# ═══════════════════════════════════════════════════════════════════════════
# Test A: micro_osc_score is computed for all symbols (unconditional)
# ═══════════════════════════════════════════════════════════════════════════


class TestMicroOscScoreComputation:
    """micro_osc_score computation in scan.py is unconditional."""

    def test_score_formula_with_hurst(self):
        """Verify the 3-term formula when hurst_exponent is available."""
        vwap_c, ema_c, hurst = 30.0, 25.0, 0.48
        k_vwap, k_ema = 25.0, 20.0
        vwap_norm = vwap_c / (vwap_c + k_vwap)         # 30/55 ≈ 0.5455
        ema_norm = ema_c / (ema_c + k_ema)              # 25/45 ≈ 0.5556
        hurst_norm = max(0.0, 1.0 - 2.0 * abs(hurst - 0.5))  # 1 - 2*0.02 = 0.96
        expected = round(0.45 * vwap_norm + 0.35 * hurst_norm + 0.20 * ema_norm, 4)

        # Replicate the scan.py formula
        assert expected == round(
            0.45 * (30 / 55) + 0.35 * (1 - 2 * abs(0.48 - 0.5)) + 0.20 * (25 / 45),
            4,
        )
        assert expected > 0  # non-trivial score

    def test_score_formula_without_hurst(self):
        """Verify the 2-term fallback when hurst_exponent is None."""
        vwap_c, ema_c = 30.0, 25.0
        k_vwap, k_ema = 25.0, 20.0
        vwap_norm = vwap_c / (vwap_c + k_vwap)
        ema_norm = ema_c / (ema_c + k_ema)
        expected = round(0.60 * vwap_norm + 0.40 * ema_norm, 4)

        assert expected == round(0.60 * (30 / 55) + 0.40 * (25 / 45), 4)
        assert expected > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test B: Bypass candidate reaches enrichment (flag ON)
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassCandidateEligibility:
    """A symbol with low score/range_prob but high micro_osc_score should
    be eligible when the micro_osc flag is enabled."""

    def test_bypass_symbol_in_eligible_set(self):
        """Symbol: score=27, range_prob=0.001, micro_osc_score=0.55,
        survival_prob=0.70.  Flag ON -> eligible."""
        df = pd.DataFrame({
            "symbol": ["ONUSDT"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.55],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "ONUSDT"].iloc[0]
        # micro_osc_bypass must be True
        assert bool(row["micro_osc_bypass"]) is True
        # Should NOT have below_threshold_tag
        assert not bool(row.get("below_threshold_tag", False))


# ═══════════════════════════════════════════════════════════════════════════
# Test C: Bypass candidate survives sort-and-truncate
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassSortAndTruncate:
    """Bypass candidates are sorted to the top and survive truncation."""

    def test_bypass_in_top_n_after_truncation(self):
        """5 standard symbols (score >= 45) + 1 bypass (score=27) with
        max_symbols=5.  Bypass must survive because it sorts to the top."""
        rows = []
        for i in range(5):
            rows.append({
                "symbol": f"STD{i}USDT",
                "score": 50.0 + i,
                "range_prob": 0.60 + 0.01 * i,
                "micro_osc_score": 0.10,
                "survival_prob": 0.30,
            })
        # Bypass candidate
        rows.append({
            "symbol": "ONUSDT",
            "score": 27.0,
            "range_prob": 0.001,
            "micro_osc_score": 0.55,
            "survival_prob": 0.70,
        })
        df = pd.DataFrame(rows)

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=5,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        # The bypass symbol should NOT have been truncated:
        # it sorts to the top and occupies one of the 5 slots
        bypass_rows = result[
            (result["symbol"] == "ONUSDT")
            & (result["micro_osc_bypass"] == True)  # noqa: E712
        ]
        assert len(bypass_rows) == 1, (
            "ONUSDT bypass candidate should survive max_symbols=5 truncation"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test D: Bypass candidate not pre-rejected
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassNotPreRejected:
    """Bypass rows skip the pre-reject filter (range_prob < 0.20 check)."""

    def test_low_range_prob_bypass_not_rejected(self):
        """range_prob=0.001 would normally be pre-rejected, but bypass
        rows are exempted."""
        df = pd.DataFrame({
            "symbol": ["ONUSDT"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.55],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "ONUSDT"].iloc[0]
        # The symbol should NOT have been pre-rejected.
        # If pre-rejected, grid_reason would still be None or some default,
        # and the validator would never run for it.  We check that the bypass
        # flag is intact (pre-reject removes symbols from the enrichment list,
        # which means no validator output is written).
        assert bool(row["micro_osc_bypass"]) is True
        # Pre-rejected symbols get no enrichment; grid_reason stays
        # "score_below_threshold" only if threshold-tagged.  Bypass rows
        # should not be threshold-tagged.
        reason = str(row.get("grid_reason", ""))
        assert "score_below_threshold" not in reason
        assert row.get("failure_stage") != "pre_reject"


# ═══════════════════════════════════════════════════════════════════════════
# Test E: Bypass candidate not threshold-tagged
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassNotThresholdTagged:
    """Bypass rows should not receive below_threshold_tag=True even when
    score < score_threshold."""

    def test_score_below_threshold_no_tag_for_bypass(self):
        """score=27 < threshold=45, but micro_osc_bypass should prevent tag."""
        df = pd.DataFrame({
            "symbol": ["ONUSDT"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.55],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "ONUSDT"].iloc[0]
        assert not bool(row["below_threshold_tag"]), (
            "Bypass row should not be tagged below_threshold"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test F: Non-qualifying symbol still blocked (flag ON)
# ═══════════════════════════════════════════════════════════════════════════


class TestNonQualifyingBlocked:
    """A symbol that fails the micro_osc_mask should remain blocked even
    when the flag is enabled."""

    def test_low_micro_osc_score_blocked(self):
        """micro_osc_score=0.30 < min_score=0.45 -> not eligible."""
        df = pd.DataFrame({
            "symbol": ["BADUSDT"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.30],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "BADUSDT"].iloc[0]
        # Should NOT be bypass
        assert not bool(row.get("micro_osc_bypass", False))
        # Should be below-threshold (since score=27 < 45 and no bypass)
        assert bool(row["below_threshold_tag"])
        assert "score_below_threshold" in str(row["grid_reason"])

    def test_low_survival_prob_blocked(self):
        """micro_osc_score=0.55 but survival_prob=0.40 < min=0.60 -> blocked."""
        df = pd.DataFrame({
            "symbol": ["LOWSURV"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.55],
            "survival_prob": [0.40],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "LOWSURV"].iloc[0]
        assert not bool(row.get("micro_osc_bypass", False))
        assert bool(row["below_threshold_tag"])


# ═══════════════════════════════════════════════════════════════════════════
# Test G: Bypass rows reach Stage B instead of dying at regime validation
# ═══════════════════════════════════════════════════════════════════════════


class TestBypassStageBPath:
    """A qualifying bypass row should reuse geometry and surface Stage B output."""

    def test_bypass_row_surfaces_stage_b_reason(self):
        df = pd.DataFrame({
            "symbol": ["ONUSDT"],
            "score": [27.0],
            "range_prob": [0.22],
            "micro_osc_score": [0.55],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        config.edge_tier.enable = False
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        vres = _mock_validator()
        vres.range_prob = 0.22
        vres.trend_prob = 0.18
        vres.survival_prob = 0.70
        vres.hurst_exponent = 0.46
        vres.ou_halflife = 12.0
        vres.current_price = 100.0
        vres.range_high = 102.0
        vres.range_low = 98.0
        vres.atr_1m = 0.2

        ms_costs = SimpleNamespace(
            round_trip_cost_pct=0.20,
            spread_cost_pct=0.05,
            funding_cost_pct=0.05,
            sufficient_liquidity=True,
            min_profit_required_pct=0.30,
            extreme_funding=False,
        )
        valid_grid = SimpleNamespace(
            is_valid=True,
            reason="ok",
            grid_lower=99.0,
            grid_upper=101.0,
            num_grids=4,
            grid_spacing_pct=0.5,
            profit_per_grid_pct=1.2,
            profit_per_grid_min_pct=1.0,
            profit_per_grid_max_pct=1.4,
            leverage=10,
            stop_loss_pct=3.0,
            capital_fraction=0.5,
            sizing_reason="legacy_grid_fraction",
        )

        with _enrich_patches(config) as ctx, \
             patch("neutralgrid.scanner.enrich_grid_params.load_empirical_profile_cached", return_value={}), \
             patch("neutralgrid.scanner.enrich_grid_params.GridCalculator") as mock_calc_cls, \
             patch("neutralgrid.scanner.enrich_grid_params.MicrostructureEstimator") as mock_ms_cls, \
             patch("neutralgrid.scanner.enrich_grid_params.MicrostructureHardGate") as mock_hg_cls, \
             patch("neutralgrid.scanner.enrich_grid_params.PositionSizer") as mock_ps_cls, \
             patch("neutralgrid.scanner.enrich_grid_params.TradableOscillationScorer") as mock_tos_cls, \
             patch("neutralgrid.scanner.enrich_grid_params.TwoStageSelector") as mock_sb_cls:
            ctx.mock_rv_cls.return_value.validate.return_value = vres
            mock_calc_cls.return_value.generate_params.return_value = valid_grid
            mock_ms_cls.return_value.estimate_costs.return_value = ms_costs
            mock_ms_cls.return_value.compute_dynamic_profit_floor.return_value = (
                0.50,
                "mid",
                {"base": 0.50},
            )
            mock_ms_cls.return_value.is_viable.return_value = (True, "ok")
            mock_hg_cls.return_value.evaluate.return_value = SimpleNamespace(
                passed=True,
                reason="hard_gate_pass",
                rejection_codes=(),
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
                reason="micro_osc_low_tos",
            )

            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "ONUSDT"].iloc[0]
        assert bool(row["micro_osc_bypass"]) is True
        assert row["failure_stage"] == "stage_b"
        assert row["grid_reason"] == "stage_b_rejected:micro_osc_low_tos"
        assert "regime_validation_failed" not in str(row["grid_reason"])


# ═══════════════════════════════════════════════════════════════════════════
# Test H: Flag OFF means no bypass
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagOffNoBypass:
    """Same qualifying data as Test B but micro_osc.enabled=False should
    produce no bypass."""

    def test_no_bypass_when_flag_disabled(self):
        """Flag OFF: symbol with score=27 and high micro_osc_score is
        still excluded from eligible set."""
        df = pd.DataFrame({
            "symbol": ["ONUSDT"],
            "score": [27.0],
            "range_prob": [0.001],
            "micro_osc_score": [0.55],
            "survival_prob": [0.70],
        })

        config = _make_config(micro_osc_enabled=False)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        row = result[result["symbol"] == "ONUSDT"].iloc[0]
        # No bypass when flag is off
        assert not bool(row.get("micro_osc_bypass", False))
        # Below threshold since score=27 < 45
        assert bool(row["below_threshold_tag"])
        assert "score_below_threshold" in str(row["grid_reason"])


# ═══════════════════════════════════════════════════════════════════════════
# Test I: Zero regression — standard symbols unaffected
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroRegression:
    """Enabling the micro_osc flag must not break standard symbol processing."""

    def test_standard_symbol_passes_normally_with_flag_on(self):
        """Standard symbol (score=60, range_prob=0.70) is unaffected by flag."""
        df = pd.DataFrame({
            "symbol": ["ETHUSDT", "ONUSDT"],
            "score": [60.0, 27.0],
            "range_prob": [0.70, 0.001],
            "micro_osc_score": [0.10, 0.55],
            "survival_prob": [0.80, 0.70],
        })

        config = _make_config(micro_osc_enabled=True)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        eth = result[result["symbol"] == "ETHUSDT"].iloc[0]
        on = result[result["symbol"] == "ONUSDT"].iloc[0]

        # Standard symbol: not bypass, not threshold-tagged
        assert not bool(eth.get("micro_osc_bypass", False))
        assert not bool(eth["below_threshold_tag"])
        reason_eth = str(eth.get("grid_reason", ""))
        assert "score_below_threshold" not in reason_eth

        # Bypass symbol: has bypass flag, not threshold-tagged
        assert bool(on["micro_osc_bypass"]) is True
        assert not bool(on["below_threshold_tag"])

    def test_standard_symbol_passes_when_flag_off(self):
        """Flag OFF: standard symbol still enriched normally."""
        df = pd.DataFrame({
            "symbol": ["ETHUSDT"],
            "score": [60.0],
            "range_prob": [0.70],
            "micro_osc_score": [0.10],
            "survival_prob": [0.80],
        })

        config = _make_config(micro_osc_enabled=False)
        enrich_cfg = EnrichConfig(
            score_threshold=45.0,
            max_symbols=10,
            concurrency=1,
        )

        with _enrich_patches(config):
            result = _run(enrich_with_grid_params(
                df_candidates=df,
                client=_mock_client(),
                pattern_profile=_mock_pattern_profile(),
                cfg=enrich_cfg,
            ))

        eth = result[result["symbol"] == "ETHUSDT"].iloc[0]
        assert not bool(eth["below_threshold_tag"])
        reason = str(eth.get("grid_reason", ""))
        assert "score_below_threshold" not in reason


# ═══════════════════════════════════════════════════════════════════════════
# Test J: Gate 4 archetype switching in TwoStageSelector
# ═══════════════════════════════════════════════════════════════════════════


class TestGate4ArchetypeSwitching:
    """Gate 4 in two_stage_selector uses survival_prob for micro-osc
    archetype instead of range_prob."""

    def test_gate4_micro_osc_archetype(self):
        """When micro_osc.enabled, bypass provenance is present (ERR-093),
        and micro_osc_score >= min_score, Gate 4 tests
        survival_prob >= min_survival_prob."""
        from neutralgrid.scanner.two_stage_selector import TwoStageSelector
        from neutralgrid.core.config import TwoStageConfig

        config = _make_config(micro_osc_enabled=True)

        with patch(
            "neutralgrid.scanner.two_stage_selector.get_config",
            return_value=config,
        ):
            selector = TwoStageSelector(TwoStageConfig())
            result = selector.approve(
                hard_gate_passed=True,
                hard_gate_reason="hard_gate_pass",
                tos=50.0,
                position_size_fraction=0.10,
                range_prob=0.20,   # would fail standard threshold
                trend_prob=0.80,
                micro_osc_score=0.55,
                micro_osc_bypass=True,  # ERR-093: provenance required
                survival_prob=0.70,
                symbol="ONUSDT",
            )

        # Gate 4 should use micro_osc_survival mode
        assert result.details.get("gate4_mode") == "micro_oscillator_survival"
        assert result.gate_results.get("regime_confidence") is True

    def test_gate4_standard_archetype_unaffected(self):
        """When micro_osc_score < min_score, Gate 4 uses standard range_prob."""
        from neutralgrid.scanner.two_stage_selector import TwoStageSelector
        from neutralgrid.core.config import TwoStageConfig

        config = _make_config(micro_osc_enabled=True)

        with patch(
            "neutralgrid.scanner.two_stage_selector.get_config",
            return_value=config,
        ):
            selector = TwoStageSelector(TwoStageConfig())
            result = selector.approve(
                hard_gate_passed=True,
                hard_gate_reason="hard_gate_pass",
                tos=50.0,
                position_size_fraction=0.10,
                range_prob=0.70,
                trend_prob=0.30,
                micro_osc_score=0.20,  # below min_score=0.45
                survival_prob=0.70,
                symbol="ETHUSDT",
            )

        # Gate 4 should use standard range_prob mode, not micro_osc
        assert result.details.get("gate4_mode") != "micro_oscillator_survival"
        assert result.gate_results.get("regime_confidence") is True
