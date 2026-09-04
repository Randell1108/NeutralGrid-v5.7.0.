"""
Tradable Oscillation Score (TOS) — composite grid-suitability metric.

Replaces raw volatility as the primary signal for grid-trading viability.
Raw volatility is undirected: it cannot distinguish range-bound oscillation
(good for grids) from trending volatility (bad for grids).

TOS combines seven sub-signals into a single [0, 100] score:

  1. grid_cross_frequency   — how often price crosses grid levels per hour
  2. mean_reversion_strength — OU half-life / Hurst evidence of reversion
  3. range_containment       — fraction of bars inside the proposed range
  4. spread_to_profit_ratio  — how much spread eats into per-grid profit
  5. funding_drag            — normalised funding cost over horizon
  6. depth_sufficiency       — order-book depth relative to position size
  7. liquidation_proximity   — distance to liquidation as fraction of range

All sub-signals are normalised to [0, 1] before weighting.

Design:
  - Pure function: no side effects, no state.
  - Deterministic: identical inputs → identical output.
  - Graceful degradation: missing optional inputs → conservative defaults.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Result
# ───────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OscillationScoreResult:
    """Immutable result of the Tradable Oscillation Score computation."""

    tos: float                        # composite score [0, 100]
    grid_cross_frequency: float       # normalised [0, 1]
    mean_reversion_strength: float    # normalised [0, 1]
    range_containment: float          # normalised [0, 1]
    spread_to_profit_inv: float       # normalised [0, 1]  (1 − ratio / cap)
    funding_drag_inv: float           # normalised [0, 1]  (1 − drag / cap)
    depth_sufficiency: float          # normalised [0, 1]
    liquidation_proximity_inv: float  # normalised [0, 1]  (1 − prox / cap)
    sub_scores: Dict[str, float] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────────
# Scorer
# ───────────────────────────────────────────────────────────────────────────

class TradableOscillationScorer:
    """Compute the Tradable Oscillation Score (TOS) for a symbol.

    Parameters
    ----------
    config : OscillationScorerConfig | None
        Weights and caps.  Falls back to ``get_config().oscillation_scorer``
        when *None*.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is None:
            from neutralgrid.core.config import get_config
            config = get_config().oscillation_scorer
        self._cfg = config

    # ── sub-signal helpers ────────────────────────────────────────────────

    @staticmethod
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _grid_cross_frequency(
        self,
        closes: np.ndarray,
        grid_spacing_pct: float,
    ) -> float:
        """Estimate grid-level crossings per bar.

        Counts how many times price crosses any grid level in the array of
        close prices, normalised to [0, 1] via a saturating function.
        """
        if len(closes) < 2 or grid_spacing_pct <= 0:
            return 0.0

        mid = float(np.median(closes))
        if mid <= 0:
            return 0.0

        spacing = mid * grid_spacing_pct / 100.0
        if spacing <= 0:
            return 0.0

        # Quantise to grid levels and count crossings
        levels = np.round((closes - mid) / spacing)
        crossings = int(np.sum(np.abs(np.diff(levels)) >= 1))
        rate = crossings / len(closes)

        # Saturating normalisation: rate / (rate + k)
        k = self._cfg.gcf_saturation_k
        return self._clamp01(rate / (rate + k))

    def _mean_reversion_strength(
        self,
        hurst_exponent: Optional[float],
        ou_halflife: Optional[float],
    ) -> float:
        """Score mean-reversion evidence from Hurst + OU half-life.

        Hurst < 0.5 → mean-reverting.  OU half-life in [4, 48] bars → tradable.
        """
        score = 0.0
        n = 0

        if hurst_exponent is not None and math.isfinite(hurst_exponent):
            # Linearly map: hurst 0.3 → 1.0, hurst 0.5 → 0.0, hurst 0.7 → 0.0
            h_score = self._clamp01((0.5 - hurst_exponent) / 0.2)
            score += h_score
            n += 1

        if ou_halflife is not None and math.isfinite(ou_halflife):
            # Sweet-spot: half-life in [ou_halflife_min_bars, ou_halflife_max_bars]
            # (canonical 4..48). Peak at ~12 bars (empirical).
            from neutralgrid.core.config import get_config
            _stoch = get_config().stochastic
            _lo = float(_stoch.ou_halflife_min_bars)
            _hi = float(_stoch.ou_halflife_max_bars)
            _peak = 12.0
            if _lo <= ou_halflife <= _hi:
                if ou_halflife <= _peak:
                    ou_score = (ou_halflife - _lo) / max(_peak - _lo, 1e-9)
                else:
                    ou_score = (_hi - ou_halflife) / max(_hi - _peak, 1e-9)
                ou_score = self._clamp01(ou_score)
            else:
                ou_score = 0.0
            score += ou_score
            n += 1

        return self._clamp01(score / n) if n > 0 else 0.0

    def _range_containment(
        self,
        closes: np.ndarray,
        grid_lower: float,
        grid_upper: float,
    ) -> float:
        """Fraction of bars within the proposed grid range."""
        if len(closes) == 0 or grid_lower >= grid_upper:
            return 0.0
        inside = np.sum((closes >= grid_lower) & (closes <= grid_upper))
        return self._clamp01(float(inside) / len(closes))

    def _spread_to_profit_inv(
        self,
        spread_cost_pct: float,
        profit_per_grid_pct: float,
    ) -> float:
        """Inverted spread-to-profit ratio, capped at configured maximum."""
        if profit_per_grid_pct <= 0:
            return 0.0
        ratio = spread_cost_pct / profit_per_grid_pct
        cap = self._cfg.spr_cap
        return self._clamp01(1.0 - ratio / cap)

    def _funding_drag_inv(
        self,
        funding_cost_pct: float,
    ) -> float:
        """Inverted funding drag, normalised by cap."""
        cap = self._cfg.funding_drag_cap_pct
        return self._clamp01(1.0 - abs(funding_cost_pct) / cap)

    def _depth_sufficiency(
        self,
        book_depth_usdt: Optional[float],
        position_size_usdt: float,
    ) -> float:
        """Order-book depth relative to position size."""
        if book_depth_usdt is None or book_depth_usdt <= 0:
            return 0.0
        if position_size_usdt <= 0:
            return 1.0
        ratio = book_depth_usdt / position_size_usdt
        # Saturating: depth / (depth + position)
        return self._clamp01(ratio / (ratio + 1.0))

    def _liquidation_proximity_inv(
        self,
        current_price: float,
        grid_lower: float,
        grid_upper: float,
        leverage: int,
    ) -> float:
        """Inverted liquidation proximity as fraction of grid range.

        Liquidation price ≈ entry × (1 − 1/leverage) for longs.
        We measure how far liquidation is from the grid boundary relative to
        the grid range width.
        """
        if leverage <= 0 or current_price <= 0:
            return 0.0
        range_width = grid_upper - grid_lower
        if range_width <= 0:
            return 0.0

        liq_distance = current_price / leverage
        proximity = liq_distance / range_width  # 0 = at boundary, 1 = far
        cap = self._cfg.liq_proximity_cap
        return self._clamp01(proximity / cap)

    # ── main scoring method ───────────────────────────────────────────────

    def compute(
        self,
        *,
        closes: np.ndarray,
        grid_lower: float,
        grid_upper: float,
        grid_spacing_pct: float,
        profit_per_grid_pct: float,
        spread_cost_pct: float = 0.0,
        funding_cost_pct: float = 0.0,
        book_depth_usdt: Optional[float] = None,
        position_size_usdt: float = 0.0,
        current_price: float = 0.0,
        leverage: int = 10,
        hurst_exponent: Optional[float] = None,
        ou_halflife: Optional[float] = None,
    ) -> OscillationScoreResult:
        """Compute the composite Tradable Oscillation Score.

        Returns
        -------
        OscillationScoreResult
            ``tos`` in [0, 100]; sub-scores each in [0, 1].
        """
        cfg = self._cfg

        gcf = self._grid_cross_frequency(closes, grid_spacing_pct)
        mrs = self._mean_reversion_strength(hurst_exponent, ou_halflife)
        rc = self._range_containment(closes, grid_lower, grid_upper)
        spr_inv = self._spread_to_profit_inv(spread_cost_pct, profit_per_grid_pct)
        fd_inv = self._funding_drag_inv(funding_cost_pct)
        ds = self._depth_sufficiency(book_depth_usdt, position_size_usdt)
        lp_inv = self._liquidation_proximity_inv(
            current_price, grid_lower, grid_upper, leverage,
        )

        # Weighted sum → [0, 1] → scale to [0, 100]
        w = cfg.weights
        raw = (
            w[0] * gcf
            + w[1] * mrs
            + w[2] * rc
            + w[3] * spr_inv
            + w[4] * fd_inv
            + w[5] * ds
            + w[6] * lp_inv
        )
        w_sum = sum(w)
        normalised = raw / w_sum if w_sum > 0 else 0.0
        tos = round(self._clamp01(normalised) * 100.0, 2)

        sub = {
            "gcf": round(gcf, 4),
            "mrs": round(mrs, 4),
            "rc": round(rc, 4),
            "spr_inv": round(spr_inv, 4),
            "fd_inv": round(fd_inv, 4),
            "ds": round(ds, 4),
            "lp_inv": round(lp_inv, 4),
        }

        return OscillationScoreResult(
            tos=tos,
            grid_cross_frequency=gcf,
            mean_reversion_strength=mrs,
            range_containment=rc,
            spread_to_profit_inv=spr_inv,
            funding_drag_inv=fd_inv,
            depth_sufficiency=ds,
            liquidation_proximity_inv=lp_inv,
            sub_scores=sub,
        )
