"""
Microstructure Hard Gate — binary pass/fail filter for grid viability.

Promotes the existing soft microstructure penalty to a mandatory hard gate
that runs in the post-grid deployment path. Any candidate that fails the hard gate
is immediately rejected with a structured reason code.

Hard-gate checks (all must pass):
  1. Spread-to-profit ratio:  spread_cost / profit_per_grid < threshold
  2. Round-trip cost ceiling:  round_trip_cost_pct < max_round_trip_pct
  3. Funding drag ceiling:     abs(funding_cost_pct) < max_funding_drag_pct
  4. Liquidity floor:          order-book depth >= min_depth_usdt

Design:
  - Deterministic: identical inputs always produce identical gate decisions.
  - No imputation: missing data → fail with ``data_missing`` reason.
  - All thresholds come from ``MicrostructureGateConfig`` (frozen dataclass
    in ``core/config.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Gate result
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MicrostructureGateResult:
    """Immutable result of the hard microstructure gate."""

    passed: bool
    rejection_codes: tuple[str, ...]
    details: Dict[str, Any]

    @property
    def reason(self) -> str:
        """Single-string summary suitable for CSV ``rejection_reasons`` column."""
        if self.passed:
            return "hard_gate_pass"
        return "|".join(self.rejection_codes) if self.rejection_codes else "hard_gate_fail"


# ───────────────────────────────────────────────────────────────────────────
# Hard gate
# ───────────────────────────────────────────────────────────────────────────

class MicrostructureHardGate:
    """Binary pass/fail microstructure gate.

    Parameters
    ----------
    config : MicrostructureGateConfig | None
        Gate thresholds.  Falls back to ``get_config().microstructure_gate``
        when *None*.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is None:
            from neutralgrid.core.config import get_config
            config = get_config().microstructure_gate
        self._cfg = config

    # ── public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        *,
        round_trip_cost_pct: Optional[float],
        spread_cost_pct: Optional[float],
        funding_cost_pct: Optional[float],
        sufficient_liquidity: Optional[bool],
        profit_per_grid_pct: Optional[float],
        dynamic_profit_floor_pct: Optional[float],
    ) -> MicrostructureGateResult:
        """Run all hard-gate checks.

        Parameters
        ----------
        round_trip_cost_pct
            Full round-trip cost as %, from ``MicrostructureCosts``.
        spread_cost_pct
            Half-spread cost as %, from ``MicrostructureCosts``.
        funding_cost_pct
            Funding cost over horizon as %, from ``MicrostructureCosts``.
        sufficient_liquidity
            Whether order-book depth met the minimum, from ``MicrostructureCosts``.
        profit_per_grid_pct
            Expected profit per grid fill as %.
        dynamic_profit_floor_pct
            Dynamic minimum profit floor from ``compute_dynamic_profit_floor()``.
            Recorded for telemetry only; grid generation owns floor enforcement.

        Returns
        -------
        MicrostructureGateResult
            Frozen result with ``passed``, ``rejection_codes``, ``details``.
        """
        codes: List[str] = []
        details: Dict[str, Any] = {}

        cfg = self._cfg

        # ── Check 1: round-trip cost ceiling ──────────────────────────────
        if round_trip_cost_pct is None:
            codes.append("data_missing:round_trip_cost")
        elif round_trip_cost_pct > cfg.max_round_trip_pct:
            codes.append(
                f"round_trip_too_high({round_trip_cost_pct:.4f}>{cfg.max_round_trip_pct:.4f})"
            )
        details["round_trip_cost_pct"] = round_trip_cost_pct
        details["max_round_trip_pct"] = cfg.max_round_trip_pct

        # ── Check 2: spread-to-profit ratio ───────────────────────────────
        if spread_cost_pct is not None and profit_per_grid_pct is not None:
            if profit_per_grid_pct > 0:
                ratio = spread_cost_pct / profit_per_grid_pct
            else:
                ratio = float("inf")
            details["spread_to_profit_ratio"] = round(ratio, 6)
            if ratio > cfg.max_spread_to_profit_ratio:
                codes.append(
                    f"spread_to_profit_too_high({ratio:.4f}>{cfg.max_spread_to_profit_ratio:.4f})"
                )
        elif spread_cost_pct is None:
            codes.append("data_missing:spread_cost")
        elif profit_per_grid_pct is None:
            # spread data available but no profit — cannot compute ratio
            codes.append("data_missing:profit_per_grid")

        # ── Check 3: funding drag ceiling ─────────────────────────────────
        if funding_cost_pct is None:
            codes.append("data_missing:funding_cost")
            details["funding_cost_pct"] = None
        else:
            details["funding_cost_pct"] = round(funding_cost_pct, 4)
            if abs(funding_cost_pct) > cfg.max_funding_drag_pct:
                codes.append(
                    f"funding_drag_too_high({abs(funding_cost_pct):.4f}>{cfg.max_funding_drag_pct:.4f})"
                )

        # ── Check 4: liquidity floor ──────────────────────────────────────
        if sufficient_liquidity is None:
            codes.append("data_missing:liquidity")
        elif not sufficient_liquidity:
            codes.append("insufficient_liquidity")
        details["sufficient_liquidity"] = sufficient_liquidity

        # ── Check 5: profit-floor viability ───────────────────────────────
        if profit_per_grid_pct is not None and dynamic_profit_floor_pct is not None:
            details["profit_per_grid_pct"] = round(profit_per_grid_pct, 4)
            details["dynamic_profit_floor_pct"] = round(dynamic_profit_floor_pct, 4)
        elif profit_per_grid_pct is None:
            # May not be available pre-grid; caller decides if that's OK
            details["profit_per_grid_pct"] = None

        passed = len(codes) == 0
        result = MicrostructureGateResult(
            passed=passed,
            rejection_codes=tuple(codes),
            details=details,
        )

        if not passed:
            logger.debug("Hard gate FAIL: %s", result.reason)

        return result

    def evaluate_from_costs(
        self,
        *,
        ms_costs: Any,
        profit_per_grid_pct: Optional[float] = None,
        dynamic_profit_floor_pct: Optional[float] = None,
    ) -> MicrostructureGateResult:
        """Convenience wrapper that unpacks a ``MicrostructureCosts`` object.

        Parameters
        ----------
        ms_costs
            ``MicrostructureCosts`` dataclass from ``MicrostructureEstimator``.
        profit_per_grid_pct
            Expected profit per grid (may be *None* pre-grid-generation).
        dynamic_profit_floor_pct
            Dynamic floor from ``compute_dynamic_profit_floor()``.
        """
        return self.evaluate(
            round_trip_cost_pct=getattr(ms_costs, "round_trip_cost_pct", None),
            spread_cost_pct=getattr(ms_costs, "spread_cost_pct", None),
            funding_cost_pct=getattr(ms_costs, "funding_cost_pct", None),
            sufficient_liquidity=getattr(ms_costs, "sufficient_liquidity", None),
            profit_per_grid_pct=profit_per_grid_pct,
            dynamic_profit_floor_pct=dynamic_profit_floor_pct,
        )
