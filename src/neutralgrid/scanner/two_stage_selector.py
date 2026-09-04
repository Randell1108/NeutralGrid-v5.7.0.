"""
Two-Stage Candidate Selection — Stage A (discovery) + Stage B (approval).

Stage A is the existing scan pipeline (scan.py): broad net, feature extraction,
similarity/EV scoring.  Stage A produces *candidates*.

Stage B is this module: strict deployment-approval gates that every candidate
must pass before receiving capital.

Stage B gates (all must pass):
  1. Microstructure hard gate  — from ``microstructure_hard_gate.py``
  2. Tradable Oscillation Score — TOS >= configured threshold
  3. Position sizer approval   — ``PositionSizer`` returns non-zero allocation
  4. Regime confidence         — range_prob above strict deployment threshold

Design:
  - Pure filter: does NOT modify candidate data, only annotates pass/fail.
  - Deterministic: identical inputs → identical verdict.
  - Each gate produces a structured reason on failure.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neutralgrid.core.config import get_config

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Result
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageBResult:
    """Immutable result of Stage B approval."""

    approved: bool
    gate_results: Dict[str, bool]    # gate_name → pass/fail
    rejection_codes: tuple[str, ...]
    details: Dict[str, Any]

    @property
    def reason(self) -> str:
        if self.approved:
            return "stage_b_approved"
        return "|".join(self.rejection_codes) if self.rejection_codes else "stage_b_rejected"


# ───────────────────────────────────────────────────────────────────────────
# Selector
# ───────────────────────────────────────────────────────────────────────────

class TwoStageSelector:
    """Stage B deployment-approval gate.

    Parameters
    ----------
    config : TwoStageConfig | None
        Thresholds.  Falls back to ``get_config().two_stage`` when *None*.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is None:
            from neutralgrid.core.config import get_config
            config = get_config().two_stage
        self._cfg = config

    def approve(
        self,
        *,
        # Gate 1: hard gate result
        hard_gate_passed: bool,
        hard_gate_reason: str = "",
        # Gate 2: TOS
        tos: Optional[float],
        # Gate 3: position sizer allocation
        position_size_fraction: Optional[float],
        # Gate 4: regime confidence
        range_prob: Optional[float],
        trend_prob: Optional[float] = None,
        # Gate 5 (optional): conformal meta-prob gate
        meta_prob: Optional[float] = None,
        conformal_quantile: Optional[Any] = None,
        # Extra context for details
        symbol: str = "",
        # Optional: HMM posteriors for entropy-adaptive thresholds
        posteriors: Optional[Any] = None,
        # Micro-oscillator archetype
        micro_osc_score: float = 0.0,
        survival_prob: Optional[float] = None,
        # ERR-093: bypass PROVENANCE (scan-phase micro_osc_score>=min AND scan
        # survival>=min) is required to enter Gate 4's survival mode; a raw
        # score alone no longer flips the mode.
        micro_osc_bypass: bool = False,
    ) -> StageBResult:
        """Evaluate all Stage B gates.

        Parameters
        ----------
        hard_gate_passed
            Result of ``MicrostructureHardGate.evaluate()``.
        hard_gate_reason
            Reason string from the hard gate result.
        tos
            Tradable Oscillation Score [0, 100].
        position_size_fraction
            Capital fraction from ``PositionSizer`` (0 = rejected).
        range_prob
            HMM posterior range probability.
        trend_prob
            HMM posterior trend probability (optional, for logging).
        symbol
            Symbol name for logging.

        Returns
        -------
        StageBResult
        """
        cfg = self._cfg
        gates: Dict[str, bool] = {}
        codes: List[str] = []
        details: Dict[str, Any] = {"symbol": symbol}

        # ── Gate 1: microstructure hard gate ──────────────────────────────
        gates["hard_gate"] = hard_gate_passed
        if not hard_gate_passed:
            codes.append(f"hard_gate_fail:{hard_gate_reason}")
        details["hard_gate_reason"] = hard_gate_reason

        # ── Gate 2: Tradable Oscillation Score ────────────────────────────
        if tos is not None:
            tos_pass = tos >= cfg.min_tos
            gates["tos"] = tos_pass
            if not tos_pass:
                codes.append(f"tos_below_min({tos:.2f}<{cfg.min_tos:.2f})")
            details["tos"] = round(tos, 2)
            details["min_tos"] = cfg.min_tos
        else:
            gates["tos"] = False
            codes.append("data_missing:tos")

        # ── Gate 3: position sizer approval ───────────────────────────────
        if position_size_fraction is not None:
            size_pass = position_size_fraction > cfg.min_position_fraction
            gates["position_sizer"] = size_pass
            if not size_pass:
                codes.append(
                    f"position_too_small({position_size_fraction:.4f}<={cfg.min_position_fraction:.4f})"
                )
            details["position_size_fraction"] = round(position_size_fraction, 4)
        else:
            gates["position_sizer"] = False
            codes.append("data_missing:position_size")

        # ── Gate 4: regime confidence (archetype-dependent) ─────────────────
        # ERR-093: micro-oscillator survival mode requires BYPASS PROVENANCE
        # (micro_osc_bypass, set at enrichment eligibility from scan-phase
        # score AND scan-phase survival), not just a raw micro_osc_score.
        # Previously score>=0.45 alone flipped 94% of the universe into the
        # survival>=0.60 test (post-grid survival median 0.208), leaving the
        # entropy-adaptive range_prob path designed for the standard archetype
        # effectively dead code. Gate 4 remains MANDATORY in both modes.
        _mosc = get_config().micro_osc
        if _mosc.enabled and micro_osc_bypass and micro_osc_score >= _mosc.min_score:
            # Micro-oscillator archetype: test MC containment via survival_prob
            details["gate4_mode"] = "micro_oscillator_survival"
            details["gate4_micro_osc_score"] = round(micro_osc_score, 4)
            if survival_prob is None or not math.isfinite(survival_prob):
                # ERR-082: a MISSING survival_prob is a data gap, not a measured
                # zero — fail-closed with a data_missing:* code so telemetry can
                # distinguish the two (safety-invariants.md, Fail-Closed Behavior).
                gates["regime_confidence"] = False
                details["gate4_survival_prob"] = None
                codes.append("data_missing:survival_prob")
            else:
                regime_pass = survival_prob >= _mosc.min_survival_prob
                gates["regime_confidence"] = regime_pass
                details["gate4_survival_prob"] = round(survival_prob, 4)
                if not regime_pass:
                    codes.append(
                        f"micro_osc_survival_below_min({survival_prob:.4f}<{_mosc.min_survival_prob:.4f})"
                    )
        elif range_prob is not None:
            # Standard archetype: entropy-adaptive HMM range_prob threshold
            effective_min_range_prob = cfg.min_range_prob
            if posteriors is not None:
                try:
                    from neutralgrid.scanner.entropy_adaptive_threshold_v20260311 import (
                        select_range_prob_threshold,
                    )
                    import numpy as _np
                    _post = _np.asarray(posteriors, dtype=float)
                    effective_min_range_prob, _tier = select_range_prob_threshold(_post)
                    details["entropy_tier"] = _tier
                    details["entropy_adaptive_threshold"] = round(effective_min_range_prob, 4)
                except (ImportError, Exception) as _e:
                    logger.warning("Entropy adaptive threshold failed, using fixed: %s", _e)

            regime_pass = range_prob >= effective_min_range_prob
            gates["regime_confidence"] = regime_pass
            details["gate4_mode"] = "hmm_range_prob"
            if not regime_pass:
                codes.append(
                    f"range_prob_below_threshold({range_prob:.4f}<{effective_min_range_prob:.4f})"
                )
            details["range_prob"] = round(range_prob, 4)
            details["min_range_prob"] = round(effective_min_range_prob, 4)
        else:
            gates["regime_confidence"] = False
            codes.append("data_missing:range_prob")

        # ── Gate 5 (optional): conformal meta-prob gate ────────────────────
        if conformal_quantile is not None and meta_prob is not None:
            try:
                from neutralgrid.calibration.conformal_risk_control_v20260311 import (
                    ConformalRiskController,
                )
                _ctrl = ConformalRiskController()
                _deploy, _cdetails = _ctrl.should_deploy(meta_prob, conformal_quantile)
                gates["conformal_meta"] = _deploy
                if not _deploy:
                    codes.append(
                        f"conformal_meta_fail(prob={meta_prob:.4f}"
                        f"<threshold={_cdetails.get('conformal_threshold', 0):.4f})"
                    )
                details["conformal_details"] = _cdetails
            except (ImportError, Exception) as _ce:
                logger.warning("Conformal gate FAILED: %s", _ce)
                codes.append(f"conformal_gate_error:{type(_ce).__name__}")
                gates["conformal_meta"] = False

        # ── Soft meta gate (FASTWIN-01; default OFF, opt-in via config) ─────
        # Only contributes a key to ``gates`` when enabled, so it affects the
        # verdict (``all(gates.values())`` below) only when explicitly turned on.
        # Fail-closed: a missing/NaN meta_prob while enabled yields
        # ``data_missing:meta`` (parallel to data_missing:tos), never a silent
        # skip. ``meta_prob`` here is the authority-gated deployment value, so a
        # non-promoted model presents as missing and is rejected when the gate
        # is on — the intended "meta influences deployment only when promoted".
        if cfg.meta_gate_enabled:
            if meta_prob is None or (
                isinstance(meta_prob, float) and math.isnan(meta_prob)
            ):
                gates["meta_gate"] = False
                codes.append("data_missing:meta")
            else:
                meta_value = float(meta_prob)
                meta_pass = meta_value >= float(cfg.min_meta_prob)
                gates["meta_gate"] = meta_pass
                if not meta_pass:
                    codes.append(
                        f"meta_prob_below_min(prob={meta_value:.4f}"
                        f"<min={float(cfg.min_meta_prob):.4f})"
                    )
                details["meta_prob"] = round(meta_value, 4)
                details["min_meta_prob"] = float(cfg.min_meta_prob)

        if trend_prob is not None:
            details["trend_prob"] = round(trend_prob, 4)

        approved = all(gates.values())

        result = StageBResult(
            approved=approved,
            gate_results=gates,
            rejection_codes=tuple(codes),
            details=details,
        )

        if not approved:
            logger.debug(
                "%s: Stage B REJECTED — %s", symbol, result.reason
            )
        else:
            logger.debug("%s: Stage B APPROVED", symbol)

        return result
