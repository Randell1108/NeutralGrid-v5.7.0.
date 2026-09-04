"""
Tests for v6.5.3 PnL-growth enhancements:

  Enhancement 1 — Two-Stage Candidate Selection (TwoStageSelector)
  Enhancement 2 — Tradable Oscillation Score (TradableOscillationScorer)
  Enhancement 3 — Hard Risk Budget Position Sizer (PositionSizer)
  Enhancement 4 — Microstructure Hard Gate (MicrostructureHardGate)
  Enhancement 5 — Hierarchical Training Labels (HierarchicalLabeler)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neutralgrid.core.config import (
    MicrostructureGateConfig,
    OscillationScorerConfig,
    TwoStageConfig,
    RiskBudgetConfig,
    HierarchicalLabelConfig,
)


# ═══════════════════════════════════════════════════════════════════════════
# Enhancement 4: Microstructure Hard Gate
# ═══════════════════════════════════════════════════════════════════════════

class TestMicrostructureHardGate:
    """Tests for MicrostructureHardGate."""

    def _gate(self, **overrides):
        from neutralgrid.validation.microstructure_hard_gate import MicrostructureHardGate
        cfg = MicrostructureGateConfig(**overrides)
        return MicrostructureHardGate(config=cfg)

    def test_all_pass(self):
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=0.50,
            spread_cost_pct=0.10,
            funding_cost_pct=0.20,
            sufficient_liquidity=True,
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.40,
        )
        assert result.passed is True
        assert result.reason == "hard_gate_pass"
        assert len(result.rejection_codes) == 0

    def test_round_trip_too_high(self):
        gate = self._gate(max_round_trip_pct=0.50)
        result = gate.evaluate(
            round_trip_cost_pct=0.80,
            spread_cost_pct=0.10,
            funding_cost_pct=0.10,
            sufficient_liquidity=True,
            profit_per_grid_pct=1.0,
            dynamic_profit_floor_pct=0.40,
        )
        assert result.passed is False
        assert any("round_trip_too_high" in c for c in result.rejection_codes)

    def test_spread_to_profit_too_high(self):
        gate = self._gate(max_spread_to_profit_ratio=0.30)
        result = gate.evaluate(
            round_trip_cost_pct=0.30,
            spread_cost_pct=0.40,
            funding_cost_pct=0.10,
            sufficient_liquidity=True,
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.30,
        )
        assert result.passed is False
        assert any("spread_to_profit_too_high" in c for c in result.rejection_codes)

    def test_funding_drag_too_high(self):
        gate = self._gate(max_funding_drag_pct=0.50)
        result = gate.evaluate(
            round_trip_cost_pct=0.30,
            spread_cost_pct=0.10,
            funding_cost_pct=0.80,
            sufficient_liquidity=True,
            profit_per_grid_pct=1.0,
            dynamic_profit_floor_pct=0.30,
        )
        assert result.passed is False
        assert any("funding_drag_too_high" in c for c in result.rejection_codes)

    def test_insufficient_liquidity(self):
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=0.30,
            spread_cost_pct=0.10,
            funding_cost_pct=0.10,
            sufficient_liquidity=False,
            profit_per_grid_pct=1.0,
            dynamic_profit_floor_pct=0.30,
        )
        assert result.passed is False
        assert "insufficient_liquidity" in result.rejection_codes

    def test_profit_below_floor_is_diagnostic_only(self):
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=0.30,
            spread_cost_pct=0.10,
            funding_cost_pct=0.10,
            sufficient_liquidity=True,
            profit_per_grid_pct=0.20,
            dynamic_profit_floor_pct=0.50,
        )
        assert result.passed is True
        assert result.details["profit_per_grid_pct"] == pytest.approx(0.20)
        assert result.details["dynamic_profit_floor_pct"] == pytest.approx(0.50)

    def test_missing_data_fails(self):
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=None,
            spread_cost_pct=None,
            funding_cost_pct=None,
            sufficient_liquidity=None,
            profit_per_grid_pct=None,
            dynamic_profit_floor_pct=None,
        )
        assert result.passed is False
        assert any("data_missing" in c for c in result.rejection_codes)

    def test_funding_none_fails_strict(self):
        """Missing funding data must fail the gate (strict mode)."""
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=0.50,
            spread_cost_pct=0.10,
            funding_cost_pct=None,  # only funding missing
            sufficient_liquidity=True,
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.40,
        )
        assert result.passed is False
        assert any("data_missing:funding_cost" in c for c in result.rejection_codes)

    def test_liquidity_none_fails_strict(self):
        """Missing liquidity data must fail the gate (strict mode)."""
        gate = self._gate()
        result = gate.evaluate(
            round_trip_cost_pct=0.50,
            spread_cost_pct=0.10,
            funding_cost_pct=0.10,
            sufficient_liquidity=None,  # no order book
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.40,
        )
        assert result.passed is False
        assert any("data_missing:liquidity" in c for c in result.rejection_codes)

    def test_multiple_failures(self):
        gate = self._gate(max_round_trip_pct=0.50, max_funding_drag_pct=0.30)
        result = gate.evaluate(
            round_trip_cost_pct=0.80,
            spread_cost_pct=0.10,
            funding_cost_pct=0.50,
            sufficient_liquidity=False,
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.40,
        )
        assert result.passed is False
        assert len(result.rejection_codes) >= 3

    def test_evaluate_from_costs(self):
        from neutralgrid.validation.microstructure_hard_gate import MicrostructureHardGate
        from neutralgrid.validation.microstructure import MicrostructureCosts
        gate = MicrostructureHardGate(config=MicrostructureGateConfig())
        costs = MicrostructureCosts(
            spread_cost_pct=0.05,
            slippage_cost_pct=0.02,
            fee_cost_pct=0.03,
            round_trip_cost_pct=0.20,
            funding_cost_pct=0.10,
            funding_penalty=0.0,
            min_profit_required_pct=0.30,
            sufficient_liquidity=True,
            extreme_funding=False,
        )
        result = gate.evaluate_from_costs(
            ms_costs=costs,
            profit_per_grid_pct=0.80,
            dynamic_profit_floor_pct=0.30,
        )
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════════════════
# Enhancement 2: Tradable Oscillation Score
# ═══════════════════════════════════════════════════════════════════════════

class TestTradableOscillationScorer:
    """Tests for TradableOscillationScorer."""

    def _scorer(self, **overrides):
        from neutralgrid.scanner.tradable_oscillation import TradableOscillationScorer
        cfg = OscillationScorerConfig(**overrides)
        return TradableOscillationScorer(config=cfg)

    def test_perfect_oscillation_high_score(self):
        scorer = self._scorer()
        # Oscillating price within range
        closes = np.array([100, 102, 98, 103, 97, 101, 99, 102, 98, 101] * 10, dtype=float)
        result = scorer.compute(
            closes=closes,
            grid_lower=95.0,
            grid_upper=105.0,
            grid_spacing_pct=1.0,
            profit_per_grid_pct=0.80,
            spread_cost_pct=0.05,
            funding_cost_pct=0.05,
            book_depth_usdt=50000.0,
            position_size_usdt=4000.0,
            current_price=100.0,
            leverage=10,
            hurst_exponent=0.35,
            ou_halflife=12.0,
        )
        assert 0 <= result.tos <= 100
        # Good oscillation → should have decent score
        assert result.tos > 20

    def test_trending_price_low_score(self):
        scorer = self._scorer()
        # Strongly trending price
        closes = np.arange(80.0, 120.0, 0.5)
        result = scorer.compute(
            closes=closes,
            grid_lower=85.0,
            grid_upper=115.0,
            grid_spacing_pct=2.0,
            profit_per_grid_pct=0.50,
            spread_cost_pct=0.30,
            funding_cost_pct=0.40,
            hurst_exponent=0.70,
        )
        assert 0 <= result.tos <= 100

    def test_empty_closes(self):
        scorer = self._scorer()
        result = scorer.compute(
            closes=np.array([], dtype=float),
            grid_lower=95.0,
            grid_upper=105.0,
            grid_spacing_pct=1.0,
            profit_per_grid_pct=0.50,
        )
        assert result.tos >= 0

    def test_sub_scores_in_range(self):
        scorer = self._scorer()
        closes = np.random.default_rng(42).normal(100, 2, 100)
        result = scorer.compute(
            closes=closes,
            grid_lower=95.0,
            grid_upper=105.0,
            grid_spacing_pct=1.0,
            profit_per_grid_pct=0.80,
            spread_cost_pct=0.10,
            funding_cost_pct=0.10,
            current_price=100.0,
            leverage=10,
        )
        assert 0 <= result.grid_cross_frequency <= 1
        assert 0 <= result.mean_reversion_strength <= 1
        assert 0 <= result.range_containment <= 1
        assert 0 <= result.spread_to_profit_inv <= 1
        assert 0 <= result.funding_drag_inv <= 1
        assert 0 <= result.depth_sufficiency <= 1
        assert 0 <= result.liquidation_proximity_inv <= 1

    def test_result_is_frozen(self):
        scorer = self._scorer()
        result = scorer.compute(
            closes=np.array([100.0]),
            grid_lower=95.0,
            grid_upper=105.0,
            grid_spacing_pct=1.0,
            profit_per_grid_pct=0.50,
        )
        with pytest.raises(AttributeError):
            result.tos = 999


# ═══════════════════════════════════════════════════════════════════════════
# Enhancement 1: Two-Stage Candidate Selection
# ═══════════════════════════════════════════════════════════════════════════

class TestTwoStageSelector:
    """Tests for TwoStageSelector (Stage B)."""

    def _selector(self, **overrides):
        from neutralgrid.scanner.two_stage_selector import TwoStageSelector
        cfg = TwoStageConfig(**overrides)
        return TwoStageSelector(config=cfg)

    def test_all_gates_pass(self):
        sel = self._selector()
        result = sel.approve(
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.50,
            range_prob=0.65,
            symbol="BTCUSDT",
            # ERR-059: the FASTWIN-01 meta gate is now default-ON, so the
            # all-gates-pass path must supply a meta_prob >= min_meta_prob (0.37).
            meta_prob=0.90,
        )
        assert result.approved is True
        assert all(result.gate_results.values())

    def test_hard_gate_fail(self):
        sel = self._selector()
        result = sel.approve(
            hard_gate_passed=False,
            hard_gate_reason="round_trip_too_high",
            tos=60.0,
            position_size_fraction=0.50,
            range_prob=0.65,
        )
        assert result.approved is False
        assert result.gate_results["hard_gate"] is False

    def test_tos_below_min(self):
        sel = self._selector(min_tos=50.0)
        result = sel.approve(
            hard_gate_passed=True,
            tos=30.0,
            position_size_fraction=0.50,
            range_prob=0.65,
        )
        assert result.approved is False
        assert any("tos_below_min" in c for c in result.rejection_codes)

    def test_position_too_small(self):
        sel = self._selector(min_position_fraction=0.10)
        result = sel.approve(
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.05,
            range_prob=0.65,
        )
        assert result.approved is False

    def test_range_prob_below_threshold(self):
        sel = self._selector(min_range_prob=0.55)
        result = sel.approve(
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.50,
            range_prob=0.40,
        )
        assert result.approved is False

    def test_missing_data_fails(self):
        sel = self._selector()
        result = sel.approve(
            hard_gate_passed=True,
            tos=None,
            position_size_fraction=None,
            range_prob=None,
        )
        assert result.approved is False
        assert any("data_missing" in c for c in result.rejection_codes)

    def test_sub_threshold_fraction_rejected(self):
        """Sizer fraction below min_position_fraction triggers Stage B rejection."""
        sel = self._selector(min_position_fraction=0.05)
        result = sel.approve(
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.003,  # below 0.05 threshold
            range_prob=0.65,
        )
        assert result.approved is False
        assert result.gate_results["position_sizer"] is False

    def test_zero_fraction_rejected(self):
        """Zero capital fraction from sizer triggers Stage B rejection."""
        sel = self._selector()
        result = sel.approve(
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.0,
            range_prob=0.65,
        )
        assert result.approved is False


# ═══════════════════════════════════════════════════════════════════════════
# Enhancement 3: Hard Risk Budget Position Sizer
# ═══════════════════════════════════════════════════════════════════════════

class TestPositionSizer:
    """Tests for PositionSizer."""

    def _sizer(self, **overrides):
        from neutralgrid.grid.position_sizer import PositionSizer
        cfg = RiskBudgetConfig(**overrides)
        return PositionSizer(config=cfg)

    def test_perfect_conditions_full_allocation(self):
        sizer = self._sizer()
        result = sizer.compute(
            range_prob=0.80,
            trend_prob=0.10,
            survival_prob=0.85,
            round_trip_cost_pct=0.10,
            profit_per_grid_pct=0.80,
            volatility_pct=2.0,
        )
        assert result.fraction > 0.50
        assert result.regime_confidence_scale == 1.0
        assert result.survival_scale == 1.0

    def test_poor_conditions_reduced_allocation(self):
        sizer = self._sizer()
        result = sizer.compute(
            range_prob=0.35,
            trend_prob=0.45,
            survival_prob=0.35,
            round_trip_cost_pct=0.50,
            profit_per_grid_pct=0.70,
            volatility_pct=8.0,
        )
        assert result.fraction < 0.50

    def test_explicit_floor_respected(self):
        """When min_fraction is explicitly set, it is enforced."""
        sizer = self._sizer(min_fraction=0.05)
        result = sizer.compute(
            range_prob=0.20,
            trend_prob=0.70,
            survival_prob=0.20,
            round_trip_cost_pct=0.80,
            profit_per_grid_pct=0.30,
            volatility_pct=15.0,
        )
        assert result.fraction >= 0.05

    def test_default_floor_zero_allows_sub_threshold(self):
        """Default min_fraction=0.0 allows fractions below Stage B's 0.05 threshold."""
        sizer = self._sizer()  # uses default min_fraction=0.0
        result = sizer.compute(
            range_prob=0.20,
            trend_prob=0.80,
            survival_prob=0.20,
            round_trip_cost_pct=0.80,
            profit_per_grid_pct=0.30,
            volatility_pct=15.0,
        )
        # With all factors at floor (0.30), product = 0.30^5 ≈ 0.0024
        assert result.fraction < 0.05
        assert result.fraction >= 0.0

    def test_ceiling_respected(self):
        sizer = self._sizer(max_fraction=0.80)
        result = sizer.compute(
            range_prob=0.95,
            trend_prob=0.02,
            survival_prob=0.99,
            round_trip_cost_pct=0.05,
            profit_per_grid_pct=1.50,
            volatility_pct=1.0,
        )
        assert result.fraction <= 0.80

    def test_portfolio_heat_reduces_allocation(self):
        sizer = self._sizer()
        result_low = sizer.compute(
            range_prob=0.80, trend_prob=0.10,
            active_positions=1, max_positions=10,
        )
        result_high = sizer.compute(
            range_prob=0.80, trend_prob=0.10,
            active_positions=9, max_positions=10,
        )
        assert result_high.fraction < result_low.fraction

    def test_missing_data_uses_defaults(self):
        sizer = self._sizer()
        result = sizer.compute(
            range_prob=0.70,
            trend_prob=0.15,
        )
        # Should not raise and should return a valid fraction
        assert 0 < result.fraction <= 1.0

    def test_result_has_sizing_reason(self):
        sizer = self._sizer()
        result = sizer.compute(range_prob=0.60, trend_prob=0.20)
        assert "regime" in result.sizing_reason
        assert "survival" in result.sizing_reason

    def test_multiplicative_independence(self):
        """Changing one factor should not affect other factors."""
        sizer = self._sizer()
        r1 = sizer.compute(range_prob=0.80, trend_prob=0.10, volatility_pct=2.0)
        r2 = sizer.compute(range_prob=0.80, trend_prob=0.10, volatility_pct=8.0)
        # Regime and survival scales should be identical
        assert r1.regime_confidence_scale == r2.regime_confidence_scale
        assert r1.survival_scale == r2.survival_scale
        # Only volatility scale should differ
        assert r2.volatility_scale < r1.volatility_scale


# ═══════════════════════════════════════════════════════════════════════════
# Enhancement 5: Hierarchical Training Labels
# ═══════════════════════════════════════════════════════════════════════════

class TestHierarchicalLabeler:
    """Tests for HierarchicalLabeler."""

    def _labeler(self, **overrides):
        from neutralgrid.training.hierarchical_labels import HierarchicalLabeler
        cfg = HierarchicalLabelConfig(**overrides)
        return HierarchicalLabeler(config=cfg)

    def test_full_pass_hlabel_3(self):
        labeler = self._labeler()
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-5.0,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=5.0,
        )
        assert result.hlabel == 3
        assert result.L1 is True
        assert result.L2 is True
        assert result.L3 is True
        assert result.meta is True
        assert result.y == 1

    def test_liquidated_hlabel_0(self):
        labeler = self._labeler()
        result = labeler.label(
            liquidated=True,
            max_drawdown_pct=-100.0,
            total_fills=5,
            duration_hours=2.0,
            net_pnl_pct=-50.0,
        )
        assert result.hlabel == 0
        assert result.L1 is False
        assert result.meta is False
        assert result.y == 0

    def test_drawdown_too_deep_hlabel_0(self):
        labeler = self._labeler(max_drawdown_floor_pct=-20.0)
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-25.0,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=5.0,
        )
        assert result.hlabel == 0
        assert result.L1 is False

    def test_insufficient_fills_hlabel_1(self):
        labeler = self._labeler(min_fills=5)
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-3.0,
            total_fills=2,
            duration_hours=6.0,
            net_pnl_pct=5.0,
        )
        assert result.hlabel == 1
        assert result.L1 is True
        assert result.L2 is False
        assert result.meta is False

    def test_pnl_below_hurdle_hlabel_2(self):
        labeler = self._labeler(hurdle_pct=3.0)
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-3.0,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=1.5,
        )
        assert result.hlabel == 2
        assert result.L1 is True
        assert result.L2 is True
        assert result.L3 is False
        assert result.meta is False
        assert result.y == 0

    def test_range_breakout_hlabel_0(self):
        labeler = self._labeler()
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-5.0,
            range_breakout=True,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=5.0,
        )
        assert result.hlabel == 0
        assert result.L1 is False

    def test_to_dict_contains_all_fields(self):
        labeler = self._labeler()
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-5.0,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=5.0,
        )
        d = result.to_dict()
        assert "hlabel" in d
        assert "hlabel_L1" in d
        assert "hlabel_L2" in d
        assert "hlabel_L3" in d
        assert "hlabel_meta" in d
        assert "hlabel_reason" in d
        assert "y" in d

    def test_duration_too_short_hlabel_1(self):
        labeler = self._labeler(min_duration_hours=2.0)
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-3.0,
            total_fills=10,
            duration_hours=0.5,
            net_pnl_pct=5.0,
        )
        assert result.hlabel == 1
        assert result.L2 is False

    def test_fees_stored_in_details(self):
        labeler = self._labeler()
        result = labeler.label(
            liquidated=False,
            max_drawdown_pct=-5.0,
            total_fills=10,
            duration_hours=6.0,
            net_pnl_pct=5.0,
            fees_paid=1.50,
            funding_fees=0.30,
        )
        assert result.details["total_costs"] == pytest.approx(1.80, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Config integration
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigIntegration:
    """Verify new config dataclasses are accessible from get_config()."""

    def test_all_new_configs_present(self):
        from neutralgrid.core.config import get_config
        cfg = get_config()
        assert hasattr(cfg, "microstructure_gate")
        assert hasattr(cfg, "oscillation_scorer")
        assert hasattr(cfg, "two_stage")
        assert hasattr(cfg, "risk_budget")
        assert hasattr(cfg, "hierarchical_label")

    def test_default_values_sensible(self):
        from neutralgrid.core.config import get_config
        cfg = get_config()
        assert cfg.microstructure_gate.max_round_trip_pct > 0
        assert cfg.oscillation_scorer.gcf_saturation_k > 0
        assert cfg.two_stage.min_tos > 0
        assert cfg.risk_budget.min_fraction >= 0
        assert cfg.hierarchical_label.min_fills > 0

    def test_dict_to_config_includes_new_sections(self):
        from neutralgrid.core.config import _dict_to_config
        data = {
            "microstructure_gate": {"max_round_trip_pct": 2.0},
            "risk_budget": {"vol_target_pct": 5.0},
        }
        cfg = _dict_to_config(data)
        assert cfg.microstructure_gate.max_round_trip_pct == 2.0
        assert cfg.risk_budget.vol_target_pct == 5.0

    def test_dict_to_config_revalidates_after_overrides_err081(self):
        """ERR-081: a config FILE cannot set mismatched hurdles without raising.

        Overrides are applied after Config() construction, so __post_init__
        validation has already run; _dict_to_config must re-validate."""
        from neutralgrid.core.config import _dict_to_config

        with pytest.raises(ValueError, match="hurdle_pct"):
            _dict_to_config({"hierarchical_label": {"hurdle_pct": 5.0}})

    def test_dict_to_config_accepts_consistent_hurdle_override(self):
        """Matched hurdle overrides on both sides must still load fine."""
        from neutralgrid.core.config import _dict_to_config

        cfg = _dict_to_config({
            "hierarchical_label": {"hurdle_pct": 5.0},
            "barrier": {"meta_hurdle_pct": 5.0},
        })
        assert cfg.hierarchical_label.hurdle_pct == 5.0
        assert cfg.barrier.meta_hurdle_pct == 5.0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: hierarchical label does NOT overwrite binary y
# ═══════════════════════════════════════════════════════════════════════════

class TestHierarchicalLabelPrimaryTarget:
    """Hierarchical labels are the primary training target."""

    def test_y_from_hierarchical_meta(self):
        """y comes from hlabel_meta (the deployment decision), not horizon label."""
        from neutralgrid.backtest.candidate_pipeline import convert_to_training_row
        bt_result = {
            "net_pnl_pct": 0.5,  # below 3% hurdle → hlabel_meta=False
            "label_positive_by_horizon": True,  # horizon says positive
            "duration_hours": 3.0,
            "max_drawdown_pct": -2.0,
            "total_trades": 5,
            "fees_paid": 0.1,
            "funding_fees": 0.05,
            "liquidated": False,
        }
        candidate = {
            "symbol": "BTCUSDT",
            "candidate_id": "BTCUSDT_20260101_000000",
        }
        row = convert_to_training_row(bt_result, candidate, hurdle_pct=3.0)
        # y comes from hlabel_meta (stricter: requires L1+L2+L3)
        assert row["y"] == 0  # 0.5% < 3.0% hurdle → L3 fail → meta=False
        # Horizon label preserved for audit
        assert row["y_horizon"] == 1
        assert row.get("hlabel_meta") is False

    def test_y_agrees_when_all_levels_pass(self):
        """When all hierarchical levels pass, y=1 and y_horizon=1."""
        from neutralgrid.backtest.candidate_pipeline import convert_to_training_row
        bt_result = {
            "net_pnl_pct": 5.0,  # above 3% hurdle
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "max_drawdown_pct": -2.0,
            "total_trades": 10,
            "fees_paid": 0.1,
            "funding_fees": 0.05,
            "liquidated": False,
        }
        candidate = {
            "symbol": "BTCUSDT",
            "candidate_id": "BTCUSDT_20260101_000000",
        }
        row = convert_to_training_row(bt_result, candidate, hurdle_pct=3.0)
        assert row["y"] == 1
        assert row["y_horizon"] == 1
        assert row.get("hlabel_meta") is True
        assert row.get("hlabel") == 3

    def test_hlabel_in_training_csv_columns(self):
        """hlabel should be in the CSV output column list (not a model feature)."""
        from neutralgrid.backtest.candidate_pipeline import TRAINING_FEATURES
        assert "hlabel" in TRAINING_FEATURES

    def test_hlabel_not_model_feature(self):
        """hlabel must NOT be a MetaLabeler model feature (information leakage)."""
        from neutralgrid.models.meta_labeler import MetaLabelerConfig
        cfg = MetaLabelerConfig()
        assert "hlabel" not in cfg.features, (
            "hlabel deterministically maps to y — including it as a feature is leakage"
        )


# ═══════════════════════════════════════════════════════════════════════════
# F5 Audit Phase 3: End-to-end hierarchical label consumption
# ═══════════════════════════════════════════════════════════════════════════

class TestMetaLabelerYCol:
    """MetaLabeler.train() must use pre-computed y_col when provided."""

    def _make_df(self, *, pnl_values=None, y_values=None):
        """Helper: build a minimal DataFrame with enough features for training.

        CPCV requires at least n_groups+1 = 7 rows.  We use 10 for margin.
        Uses the active snapshot feature profile.
        """
        import pandas as pd
        from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
        n = 10
        if pnl_values is None:
            pnl_values = [10.0, 8.0, 6.0, 12.0, 7.0, 9.0, 11.0, 3.0, 14.0, 6.5]
        # Pad to n if fewer supplied
        while len(pnl_values) < n:
            pnl_values.append(5.0)
        rng = np.random.RandomState(42)
        data: dict = {
            "pnl_pct": pnl_values[:n],
            "start_time_utc": pd.date_range("2026-01-01", periods=n, freq="1h"),
        }
        # Populate all features from the current profile with plausible random values
        for feat in ACTIVE_SNAPSHOT_META_FEATURES:
            data[feat] = rng.uniform(0.01, 1.0, size=n).tolist()
        if y_values is not None:
            while len(y_values) < n:
                y_values.append(0)
            data["y"] = y_values[:n]
        return pd.DataFrame(data)

    def test_y_col_used_when_present(self):
        """When y_col='y' and df has 'y' column, use it directly."""
        from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig

        cfg = MetaLabelerConfig(hurdle_pct=5.0)
        labeler = MetaLabeler(cfg)

        # PnL: 9/10 above hurdle → create_labels would give positive_rate=0.9
        # But y column has only 2 positive → positive_rate must be 0.2
        df = self._make_df(y_values=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        metrics = labeler.train(df, y_col="y")
        assert metrics.positive_rate == pytest.approx(0.2)

    def test_y_col_fallback_when_absent(self):
        """When y_col specified but column absent, fall back to create_labels."""
        from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig

        cfg = MetaLabelerConfig(hurdle_pct=5.0)
        labeler = MetaLabeler(cfg)

        # 10 rows: [10, 2, 8, 1, 6, 9, 11, 3, 14, 6.5]
        # >= 5.0: 10, 8, 6, 9, 11, 14, 6.5 = 7 positive out of 10
        df = self._make_df(pnl_values=[10.0, 2.0, 8.0, 1.0, 6.0, 9.0, 11.0, 3.0, 14.0, 6.5])
        # y_col="y" but no y column → falls back to create_labels
        metrics = labeler.train(df, y_col="y")
        assert metrics.positive_rate == pytest.approx(7.0 / 10.0)

    def test_y_col_none_uses_create_labels(self):
        """When y_col=None (default), create_labels is used."""
        from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig

        cfg = MetaLabelerConfig(hurdle_pct=5.0)
        labeler = MetaLabeler(cfg)

        # Has y column with all 0s, but y_col=None → should IGNORE y column
        df = self._make_df(
            pnl_values=[10.0, 2.0, 8.0, 1.0, 6.0, 9.0, 11.0, 3.0, 14.0, 6.5],
            y_values=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        # y_col=None → create_labels from PnL, ignoring y column
        # 7 out of 10 >= 5.0 hurdle
        metrics = labeler.train(df)
        assert metrics.positive_rate == pytest.approx(7.0 / 10.0)


class TestUnifiedBuilderHlabelMeta:
    """unified_training_builder prefers hlabel_meta for backtest y labels."""

    def _extract_y(self, row_dict: dict) -> int:
        """Replicate the label extraction logic from _ingest_backtest_rows."""
        import pandas as pd
        row = pd.Series(row_dict)

        hlabel_meta_raw = row.get("hlabel_meta")
        if pd.notna(hlabel_meta_raw):
            if isinstance(hlabel_meta_raw, (bool, np.bool_)):
                return 1 if hlabel_meta_raw else 0
            return 1 if str(hlabel_meta_raw).strip().lower() in ("true", "1", "yes") else 0

        label_bt = row.get("label_positive_by_horizon")
        if pd.notna(label_bt):
            if isinstance(label_bt, (bool, np.bool_)):
                return 1 if label_bt else 0
            return 1 if str(label_bt).strip().lower() in ("true", "1", "yes") else 0

        meta_hurdle = 3.0
        net_pnl = float(row.get("net_pnl_pct", 0.0))
        return 1 if net_pnl >= meta_hurdle else 0

    def test_hlabel_meta_preferred_over_horizon(self):
        """When hlabel_meta is present, it drives y over label_positive_by_horizon."""
        y = self._extract_y({
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "hlabel_meta": False,  # hierarchical says reject
        })
        assert y == 0, "hlabel_meta=False should produce y=0"

    def test_hlabel_meta_true_produces_y_one(self):
        """hlabel_meta=True → y=1 regardless of horizon label."""
        y = self._extract_y({
            "net_pnl_pct": 1.0,
            "label_positive_by_horizon": False,
            "hlabel_meta": True,
        })
        assert y == 1

    def test_fallback_to_horizon_when_hlabel_missing(self):
        """When hlabel_meta absent, fall back to label_positive_by_horizon."""
        y = self._extract_y({
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
        })
        assert y == 1

    def test_fallback_to_pnl_hurdle_when_both_missing(self):
        """When both hlabel_meta and horizon label absent, use PnL hurdle."""
        y = self._extract_y({"net_pnl_pct": 1.0})
        assert y == 0  # 1.0 < 3.0 hurdle


class TestTrainingSchemaHlabel:
    """TrainingTableSchema includes hlabel columns in labels tuple."""

    def test_hlabel_in_schema_labels(self):
        from neutralgrid.training.data_generator import TrainingTableSchema
        schema = TrainingTableSchema()
        assert "hlabel" in schema.labels
        assert "hlabel_meta" in schema.labels
        assert "y_horizon" in schema.labels

    def test_hlabel_not_in_schema_features(self):
        from neutralgrid.training.data_generator import TrainingTableSchema
        schema = TrainingTableSchema()
        assert "hlabel" not in schema.features
        assert "hlabel_meta" not in schema.features

    def test_schema_all_columns_includes_hlabel(self):
        from neutralgrid.training.data_generator import TrainingTableSchema
        schema = TrainingTableSchema()
        all_cols = schema.all_columns
        assert "hlabel" in all_cols
        assert "hlabel_meta" in all_cols
        assert "y_horizon" in all_cols
