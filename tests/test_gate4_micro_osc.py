"""
Tests for Gate 4 archetype-dependent behaviour in TwoStageSelector.

Gate 4 has two modes:
  1. Standard mode  — ``range_prob >= effective_min_range_prob``
  2. Micro-oscillator mode — when ``micro_osc.enabled`` AND
     ``micro_osc_bypass`` (scan-phase provenance, ERR-093) AND
     ``micro_osc_score >= min_score``, tests ``survival_prob >= min_survival_prob``

These tests verify:
  - Correct mode selection based on MicroOscConfig + bypass provenance + score
  - Pass/fail logic for each mode
  - Fallback from micro-osc to standard when score < min_score or no bypass
  - Boundary conditions at thresholds
  - ``all(gates.values())`` aggregation with archetype-dependent Gate 4
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from neutralgrid.core.config import MicroOscConfig, TwoStageConfig
from neutralgrid.scanner.two_stage_selector import StageBResult, TwoStageSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_selector(**two_stage_overrides) -> TwoStageSelector:
    """Build a TwoStageSelector with custom TwoStageConfig values."""
    # This suite isolates Gate 4 (micro_osc). The FASTWIN-01 meta gate (ERR-059,
    # now default-ON) is orthogonal and would otherwise fail-closed on the absent
    # meta_prob; disable it here unless a test overrides it explicitly.
    two_stage_overrides.setdefault("meta_gate_enabled", False)
    cfg = TwoStageConfig(**two_stage_overrides)
    return TwoStageSelector(config=cfg)


@dataclass
class _FakeGlobalConfig:
    """Minimal stand-in for the global Config returned by get_config().

    Only ``micro_osc`` is needed because Gate 4 accesses
    ``get_config().micro_osc`` directly (line 156 of two_stage_selector.py).
    """

    micro_osc: MicroOscConfig


def _approve_with_micro_osc(
    micro_osc_cfg: MicroOscConfig,
    *,
    two_stage_overrides: dict | None = None,
    **approve_kwargs,
) -> StageBResult:
    """Call ``approve()`` with a patched global config for micro_osc."""
    fake_global = _FakeGlobalConfig(micro_osc=micro_osc_cfg)
    sel = _make_selector(**(two_stage_overrides or {}))
    with patch(
        "neutralgrid.scanner.two_stage_selector.get_config",
        return_value=fake_global,
    ):
        return sel.approve(**approve_kwargs)


# Shared kwargs that make gates 1-3 pass unconditionally.
_GATES_123_PASS = dict(
    hard_gate_passed=True,
    tos=60.0,
    position_size_fraction=0.50,
)


# ---------------------------------------------------------------------------
# Test 1: Standard mode — Gate 4 passes with sufficient range_prob
# ---------------------------------------------------------------------------


class TestStandardModePass:
    """micro_osc disabled => standard mode, range_prob high => Gate 4 passes."""

    def test_approved_and_gate4_mode(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=False),
            **_GATES_123_PASS,
            range_prob=0.50,
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.gate_results["regime_confidence"] is True
        assert result.details["gate4_mode"] == "hmm_range_prob"

    def test_enabled_but_score_below_also_standard(self):
        """enabled=True but micro_osc_score < min_score => standard mode."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45),
            **_GATES_123_PASS,
            range_prob=0.50,
            micro_osc_score=0.20,
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.details["gate4_mode"] == "hmm_range_prob"

    def test_err093_high_score_without_bypass_stays_standard(self):
        """ERR-093: a raw micro_osc_score >= min_score WITHOUT bypass
        provenance must NOT flip Gate 4 into survival mode. Previously
        score>=0.45 alone routed ~94% of the universe into the
        survival>=0.60 test (median survival 0.208)."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.50,        # passes the standard path
            micro_osc_score=0.55,   # above min_score...
            micro_osc_bypass=False,  # ...but no provenance
            survival_prob=0.10,     # would fail survival mode hard
            symbol="BTCUSDT",
        )
        assert result.details["gate4_mode"] == "hmm_range_prob"
        assert result.gate_results["regime_confidence"] is True
        assert result.approved is True


# ---------------------------------------------------------------------------
# Test 2: Standard mode — Gate 4 fails with low range_prob
# ---------------------------------------------------------------------------


class TestStandardModeFail:
    """micro_osc disabled, range_prob too low => Gate 4 fails."""

    def test_rejected_regime_confidence_false(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=False),
            **_GATES_123_PASS,
            range_prob=0.01,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.gate_results["regime_confidence"] is False
        assert any(
            "range_prob_below_threshold" in c for c in result.rejection_codes
        )


# ---------------------------------------------------------------------------
# Test 3: Micro-osc mode — Gate 4 passes with sufficient survival_prob
# ---------------------------------------------------------------------------


class TestMicroOscModePass:
    """micro_osc enabled, score >= min_score, survival_prob high => pass."""

    def test_approved_with_micro_osc_gate4(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,       # would fail standard mode
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.70,
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.gate_results["regime_confidence"] is True
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        assert result.details["gate4_survival_prob"] == round(0.70, 4)
        assert result.details["gate4_micro_osc_score"] == round(0.55, 4)


# ---------------------------------------------------------------------------
# Test 4: Micro-osc mode — Gate 4 fails with low survival_prob
# ---------------------------------------------------------------------------


class TestMicroOscModeFail:
    """micro_osc enabled, score ok, but survival_prob too low => fail."""

    def test_rejected_survival_below_min(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.40,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.gate_results["regime_confidence"] is False
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        assert any(
            "micro_osc_survival_below_min" in c for c in result.rejection_codes
        )


# ---------------------------------------------------------------------------
# Test 4b (ERR-082): Micro-osc mode — MISSING survival_prob is a data gap
# ---------------------------------------------------------------------------


class TestMicroOscModeMissingSurvival:
    """micro_osc enabled, score ok, survival_prob ABSENT => fail-closed with
    data_missing:survival_prob, NOT micro_osc_survival_below_min (ERR-082:
    telemetry must distinguish a data gap from a measured zero)."""

    def test_missing_survival_prob_emits_data_missing(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=None,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.gate_results["regime_confidence"] is False
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        assert result.details["gate4_survival_prob"] is None
        assert "data_missing:survival_prob" in result.rejection_codes
        assert not any(
            "micro_osc_survival_below_min" in c for c in result.rejection_codes
        )

    def test_nan_survival_prob_emits_data_missing(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=float("nan"),
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert "data_missing:survival_prob" in result.rejection_codes

    def test_measured_zero_survival_prob_is_threshold_rejection(self):
        """A genuine measured 0.0 must remain a threshold rejection."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.0,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert any(
            "micro_osc_survival_below_min" in c for c in result.rejection_codes
        )
        assert "data_missing:survival_prob" not in result.rejection_codes


# ---------------------------------------------------------------------------
# Test 5: Flag enabled but score below threshold => falls back to standard
# ---------------------------------------------------------------------------


class TestFallbackToStandard:
    """enabled=True, score < min_score => standard mode used."""

    def test_fallback_passes_standard(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.50,
            micro_osc_score=0.30,   # below 0.45
            survival_prob=0.70,     # irrelevant — not used in standard mode
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.details["gate4_mode"] == "hmm_range_prob"
        assert result.gate_results["regime_confidence"] is True


# ---------------------------------------------------------------------------
# Test 6: Fallback to standard mode, standard also fails
# ---------------------------------------------------------------------------


class TestFallbackStandardFails:
    """enabled=True, score < min_score => standard, range_prob low => fail."""

    def test_rejected_standard_fallback(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.01,
            micro_osc_score=0.30,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.gate_results["regime_confidence"] is False
        assert result.details["gate4_mode"] == "hmm_range_prob"
        assert any(
            "range_prob_below_threshold" in c for c in result.rejection_codes
        )


# ---------------------------------------------------------------------------
# Test 7: all(gates.values()) aggregates correctly with micro-osc Gate 4
# ---------------------------------------------------------------------------


class TestAllGatesAggregation:
    """Gate 4 micro-osc passes, but another gate fails => rejected."""

    def test_gate4_pass_but_gate1_fails(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            hard_gate_passed=False,            # Gate 1 fails
            hard_gate_reason="round_trip_too_high",
            tos=60.0,
            position_size_fraction=0.50,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.70,
            symbol="BTCUSDT",
        )
        # Gate 4 micro-osc passes
        assert result.gate_results["regime_confidence"] is True
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        # But Gate 1 failed => overall rejected
        assert result.gate_results["hard_gate"] is False
        assert result.approved is False

    def test_gate4_pass_but_gate2_fails(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            hard_gate_passed=True,
            tos=10.0,                          # Gate 2 fails (min_tos=40)
            position_size_fraction=0.50,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.70,
            symbol="BTCUSDT",
        )
        assert result.gate_results["regime_confidence"] is True
        assert result.gate_results["tos"] is False
        assert result.approved is False

    def test_gate4_pass_but_gate3_fails(self):
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            hard_gate_passed=True,
            tos=60.0,
            position_size_fraction=0.0,        # Gate 3 fails
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.70,
            symbol="BTCUSDT",
        )
        assert result.gate_results["regime_confidence"] is True
        assert result.gate_results["position_sizer"] is False
        assert result.approved is False


# ---------------------------------------------------------------------------
# Test 8: Boundary conditions — at exact thresholds (>=)
# ---------------------------------------------------------------------------


class TestBoundaryExactThresholds:
    """Values exactly at threshold — ``>=`` must pass."""

    def test_micro_osc_score_exactly_at_min(self):
        """micro_osc_score == min_score (0.45) triggers micro-osc mode."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.45,   # exactly at min_score
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.60,     # exactly at min_survival_prob
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        assert result.gate_results["regime_confidence"] is True

    def test_standard_range_prob_exactly_at_min(self):
        """range_prob == min_range_prob (0.45 default) passes standard mode."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=False),
            two_stage_overrides={"min_range_prob": 0.45},
            **_GATES_123_PASS,
            range_prob=0.45,
            symbol="BTCUSDT",
        )
        assert result.approved is True
        assert result.details["gate4_mode"] == "hmm_range_prob"
        assert result.gate_results["regime_confidence"] is True


# ---------------------------------------------------------------------------
# Test 9: Boundary — just below thresholds
# ---------------------------------------------------------------------------


class TestBoundaryJustBelow:
    """Values epsilon below thresholds."""

    def test_score_just_below_falls_back_to_standard(self):
        """micro_osc_score = 0.4499 < 0.45 => falls back to standard mode."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.50,        # passes standard
            micro_osc_score=0.4499,
            survival_prob=0.70,
            symbol="BTCUSDT",
        )
        # Should fall back to standard mode and pass
        assert result.approved is True
        assert result.details["gate4_mode"] == "hmm_range_prob"

    def test_survival_prob_just_below_fails(self):
        """survival_prob = 0.5999 < 0.60 => micro-osc Gate 4 fails."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.001,
            micro_osc_score=0.55,
            micro_osc_bypass=True,  # ERR-093: micro mode requires bypass provenance
            survival_prob=0.5999,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.gate_results["regime_confidence"] is False
        assert result.details["gate4_mode"] == "micro_oscillator_survival"
        assert any(
            "micro_osc_survival_below_min" in c for c in result.rejection_codes
        )

    def test_score_just_below_and_standard_fails(self):
        """score < min_score => fallback to standard; range_prob low => fail."""
        result = _approve_with_micro_osc(
            MicroOscConfig(enabled=True, min_score=0.45, min_survival_prob=0.60),
            **_GATES_123_PASS,
            range_prob=0.01,
            micro_osc_score=0.4499,
            symbol="BTCUSDT",
        )
        assert result.approved is False
        assert result.details["gate4_mode"] == "hmm_range_prob"
        assert result.gate_results["regime_confidence"] is False
