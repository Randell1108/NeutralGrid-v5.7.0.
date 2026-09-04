"""
Hard Risk Budget Position Sizer — centralised, multiplicative capital allocation.

Replaces the scattered regime-aware sizing logic with a single deterministic
formula:

    final_size = regime_confidence_scale
                 × regime_confidence_scale
                 × survival_scale
                 × microstructure_scale
                 × volatility_scale
                 × portfolio_heat_scale

Each factor is independently computed and clamped to a configurable range,
ensuring the product is always between ``min_fraction`` and ``max_fraction``.

Design:
  - Multiplicative: factors are independent and composable.
  - No hidden fallbacks: every factor has explicit bounds.
  - Auditable: ``PositionSizeResult`` stores every intermediate factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Result
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PositionSizeResult:
    """Immutable result of position sizing."""

    fraction: float                     # final capital fraction [min, max]
    regime_confidence_scale: float      # [0.3, 1.0]
    survival_scale: float               # [0.3, 1.0]
    microstructure_scale: float         # [0.3, 1.0]
    volatility_scale: float             # [0.3, 1.0]
    portfolio_heat_scale: float         # [0.3, 1.0]
    sizing_reason: str
    details: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────
# Sizer
# ───────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """Centralized multiplicative position sizer.

    Parameters
    ----------
    config : RiskBudgetConfig | None
        Sizing bounds and factor parameters.  Falls back to
        ``get_config().risk_budget`` when *None*.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is None:
            from neutralgrid.core.config import get_config
            config = get_config().risk_budget
        self._cfg = config

    # ── factor computations ───────────────────────────────────────────────

    def _regime_confidence(
        self,
        range_prob: float,
        trend_prob: float,
    ) -> tuple[float, str]:
        """Scale based on HMM regime posterior.

        Full allocation when range_prob >= 0.70 and trend_prob <= 0.20.
        Linear taper otherwise.
        """
        cfg = self._cfg

        # Range contribution: linear from range_floor → 1.0 over [min_range, 0.70]
        if range_prob >= 0.70:
            r_scale = 1.0
        elif range_prob >= cfg.regime_range_floor and cfg.regime_range_floor < 0.70:
            denom = 0.70 - cfg.regime_range_floor
            r_scale = cfg.factor_floor + (1.0 - cfg.factor_floor) * (
                (range_prob - cfg.regime_range_floor) / denom
            )
        else:
            r_scale = cfg.factor_floor

        # Trend penalty: linear from 0% at trend_prob=0.20 to 80% at trend_prob=1.0
        trend_penalty = max(0.0, trend_prob - 0.20)
        scale = max(cfg.factor_floor, r_scale * (1.0 - trend_penalty))
        reason = f"range={range_prob:.3f},trend={trend_prob:.3f}"
        return (round(scale, 4), reason)

    def _survival(
        self,
        survival_prob: Optional[float],
    ) -> tuple[float, str]:
        """Scale based on stochastic survival probability.

        Full allocation when survival >= 0.70.  Linear from floor to 1.0
        over [0.40, 0.70].  Below 0.40 → floor.
        """
        cfg = self._cfg
        if survival_prob is None:
            return (cfg.default_survival_scale, "survival_unknown")

        if survival_prob >= 0.70:
            return (1.0, "survival_high")
        elif survival_prob >= 0.40:
            scale = cfg.factor_floor + (1.0 - cfg.factor_floor) * (
                (survival_prob - 0.40) / 0.30
            )
            return (round(scale, 4), f"survival_moderate({survival_prob:.3f})")
        else:
            return (cfg.factor_floor, f"survival_low({survival_prob:.3f})")

    def _microstructure(
        self,
        round_trip_cost_pct: Optional[float],
        profit_per_grid_pct: Optional[float],
    ) -> tuple[float, str]:
        """Scale inversely with microstructure drag.

        Full allocation when cost < 20% of profit.  Linear taper to floor
        when cost approaches 60% of profit.
        """
        cfg = self._cfg
        if round_trip_cost_pct is None or profit_per_grid_pct is None:
            return (cfg.default_micro_scale, "micro_unknown")

        if profit_per_grid_pct <= 0:
            return (cfg.factor_floor, "micro_no_profit")

        cost_ratio = round_trip_cost_pct / profit_per_grid_pct

        if cost_ratio <= 0.20:
            return (1.0, "micro_clean")
        elif cost_ratio <= 0.60:
            scale = 1.0 - (cost_ratio - 0.20) / 0.40 * (1.0 - cfg.factor_floor)
            return (round(max(cfg.factor_floor, scale), 4), f"micro_moderate({cost_ratio:.3f})")
        else:
            return (cfg.factor_floor, f"micro_heavy({cost_ratio:.3f})")

    def _volatility(
        self,
        volatility_pct: Optional[float],
    ) -> tuple[float, str]:
        """Scale inversely with realised volatility.

        Full allocation when vol <= target.  Linear taper when vol
        approaches 2× target.
        """
        cfg = self._cfg
        if volatility_pct is None:
            return (cfg.default_vol_scale, "vol_unknown")

        target = cfg.vol_target_pct
        if volatility_pct <= target:
            return (1.0, "vol_in_target")
        elif volatility_pct <= target * 2.0:
            excess = (volatility_pct - target) / target
            scale = 1.0 - excess * (1.0 - cfg.factor_floor)
            return (round(max(cfg.factor_floor, scale), 4), f"vol_elevated({volatility_pct:.2f})")
        else:
            return (cfg.factor_floor, f"vol_extreme({volatility_pct:.2f})")

    def _portfolio_heat(
        self,
        active_positions: int,
        max_positions: int,
    ) -> tuple[float, str]:
        """Scale inversely with portfolio heat (number of active positions).

        Full allocation when active < 50% of max.  Linear taper to floor
        as portfolio fills up.
        """
        cfg = self._cfg
        if max_positions <= 0:
            return (1.0, "heat_unlimited")

        utilisation = active_positions / max_positions
        if utilisation <= 0.50:
            return (1.0, "heat_low")
        elif utilisation <= 0.90:
            scale = 1.0 - (utilisation - 0.50) / 0.40 * (1.0 - cfg.factor_floor)
            return (round(max(cfg.factor_floor, scale), 4), f"heat_moderate({utilisation:.2f})")
        else:
            return (cfg.factor_floor, f"heat_high({utilisation:.2f})")

    # ── main API ──────────────────────────────────────────────────────────

    def compute(
        self,
        *,
        range_prob: float,
        trend_prob: float,
        survival_prob: Optional[float] = None,
        round_trip_cost_pct: Optional[float] = None,
        profit_per_grid_pct: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        active_positions: int = 0,
        max_positions: int = 10,
    ) -> PositionSizeResult:
        """Compute the final position size fraction.

        Parameters
        ----------
        range_prob, trend_prob
            HMM posterior probabilities.
        survival_prob
            Stochastic survival probability [0, 1].
        round_trip_cost_pct
            Microstructure round-trip cost as %.
        profit_per_grid_pct
            Expected profit per grid fill as %.
        volatility_pct
            Realised volatility as %.
        active_positions
            Number of currently active grid positions.
        max_positions
            Maximum allowed concurrent positions.

        Returns
        -------
        PositionSizeResult
        """
        cfg = self._cfg
        reasons: list[str] = []

        rc_scale, rc_reason = self._regime_confidence(range_prob, trend_prob)
        reasons.append(f"regime({rc_reason})")

        sv_scale, sv_reason = self._survival(survival_prob)
        reasons.append(f"survival({sv_reason})")

        ms_scale, ms_reason = self._microstructure(
            round_trip_cost_pct, profit_per_grid_pct,
        )
        reasons.append(f"micro({ms_reason})")

        vol_scale, vol_reason = self._volatility(volatility_pct)
        reasons.append(f"vol({vol_reason})")

        ph_scale, ph_reason = self._portfolio_heat(active_positions, max_positions)
        reasons.append(f"heat({ph_reason})")

        # Multiplicative composition
        raw = rc_scale * sv_scale * ms_scale * vol_scale * ph_scale
        fraction = max(cfg.min_fraction, min(cfg.max_fraction, raw))
        fraction = round(fraction, 4)

        sizing_reason = "; ".join(reasons)

        return PositionSizeResult(
            fraction=fraction,
            regime_confidence_scale=rc_scale,
            survival_scale=sv_scale,
            microstructure_scale=ms_scale,
            volatility_scale=vol_scale,
            portfolio_heat_scale=ph_scale,
            sizing_reason=sizing_reason,
            details={
                "raw_product": round(raw, 6),
                "clamped_fraction": fraction,
            },
        )
