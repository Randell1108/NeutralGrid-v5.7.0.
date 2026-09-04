"""
Hierarchical Training Labels — 4-level labelling for richer training signal.

Replaces the binary (y=0/1) label with a hierarchical label that encodes
the *stage* at which a candidate succeeded or failed:

  L1  Geometry Survival   — did the grid hold together?
      Pass: price stayed within range; no liquidation.
      Fail: range breakout or liquidation.

  L2  Execution Viability  — were fills achievable?
      Pass: at least ``min_fills`` grid fills occurred.
      Fail: fills below threshold (dead grid).

  L3  Net Return Hurdle    — did it beat the cost hurdle?
      Pass: net_pnl_pct >= hurdle_pct (covers fees + costs).
      Fail: profitable-looking grid that doesn't clear costs.

  Meta  Deploy             — composite approval for deployment.
      Pass: L1 AND L2 AND L3 all passed.
      Fail: any level failed.

The hierarchical label is stored as:
  - ``hlabel``:       integer 0–3 (highest level passed)
  - ``hlabel_L1``:    bool
  - ``hlabel_L2``:    bool
  - ``hlabel_L3``:    bool
  - ``hlabel_meta``:  bool (= L1 & L2 & L3)
  - ``hlabel_reason``: human-readable summary

Design:
  - Pure function: no side effects.
  - Backward-compatible: ``y`` (binary label) is derived from ``hlabel_meta``.
  - All thresholds from ``HierarchicalLabelConfig``.
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
class HierarchicalLabelResult:
    """Immutable hierarchical label."""

    hlabel: int          # 0 = failed L1, 1 = passed L1 only, 2 = L1+L2, 3 = L1+L2+L3
    L1: bool             # geometry survival
    L2: bool             # execution viability
    L3: bool             # net return hurdle
    meta: bool           # deploy = L1 & L2 & L3
    reason: str          # human-readable summary
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> int:
        """Binary label compatible with existing training pipeline."""
        return 1 if self.meta else 0

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for merging into a training row."""
        return {
            "hlabel": self.hlabel,
            "hlabel_L1": self.L1,
            "hlabel_L2": self.L2,
            "hlabel_L3": self.L3,
            "hlabel_meta": self.meta,
            "hlabel_reason": self.reason,
            "y": self.y,
            **{f"hlabel_detail_{k}": v for k, v in self.details.items()},
        }


# ───────────────────────────────────────────────────────────────────────────
# Labeller
# ───────────────────────────────────────────────────────────────────────────

class HierarchicalLabeler:
    """Compute hierarchical training labels from backtest results.

    Parameters
    ----------
    config : HierarchicalLabelConfig | None
        Thresholds.  Falls back to ``get_config().hierarchical_label``
        when *None*.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is None:
            from neutralgrid.core.config import get_config
            config = get_config().hierarchical_label
        self._cfg = config

    def label(
        self,
        *,
        # L1 inputs
        liquidated: bool,
        max_drawdown_pct: float,
        range_breakout: Optional[bool] = None,
        # L2 inputs
        total_fills: int,
        duration_hours: float,
        # L3 inputs
        net_pnl_pct: float,
        fees_paid: float = 0.0,
        funding_fees: float = 0.0,
        # ── alignment-v1 inputs (optional, backward-compatible) ──
        realized_net_pnl_pct: Optional[float] = None,
        unrealized_fraction: Optional[float] = None,
    ) -> HierarchicalLabelResult:
        """Compute the hierarchical label.

        Parameters
        ----------
        liquidated
            Whether the position was liquidated.
        max_drawdown_pct
            Maximum drawdown as percentage (negative value).
        range_breakout
            Whether price broke out of the grid range (optional; inferred
            from drawdown if *None*).
        total_fills
            Number of grid fills that occurred.
        duration_hours
            Duration of the backtest in hours.
        net_pnl_pct
            Net PnL as percentage (after all costs), mark-to-market.
        fees_paid
            Total trading fees paid.
        funding_fees
            Total funding fees paid.
        realized_net_pnl_pct
            Realized-only net PnL as percentage (alignment-v1).
            When provided, L3 uses this instead of MTM ``net_pnl_pct``.
        unrealized_fraction
            ``|unrealized| / (|unrealized| + |realized|)``.
            When provided and exceeds ``max_unrealized_fraction``, L2 fails.

        Returns
        -------
        HierarchicalLabelResult
        """
        cfg = self._cfg
        details: Dict[str, Any] = {}

        # ── L1: Geometry Survival ─────────────────────────────────────────
        if liquidated:
            L1 = False
            l1_reason = "liquidated"
        elif max_drawdown_pct < cfg.max_drawdown_floor_pct:
            L1 = False
            l1_reason = f"drawdown_exceeded({max_drawdown_pct:.2f}<{cfg.max_drawdown_floor_pct:.2f})"
        elif range_breakout is True:
            L1 = False
            l1_reason = "range_breakout"
        else:
            L1 = True
            l1_reason = "geometry_held"
        details["l1_reason"] = l1_reason
        details["range_breakout_evaluated"] = range_breakout is not None

        # ── L2: Execution Viability ───────────────────────────────────────
        if not L1:
            L2 = False
            l2_reason = "l1_failed"
        elif total_fills < cfg.min_fills:
            L2 = False
            l2_reason = f"insufficient_fills({total_fills}<{cfg.min_fills})"
        elif duration_hours < cfg.min_duration_hours:
            L2 = False
            l2_reason = f"duration_too_short({duration_hours:.1f}<{cfg.min_duration_hours:.1f})"
        elif (
            unrealized_fraction is not None
            and hasattr(cfg, "max_unrealized_fraction")
            and unrealized_fraction > cfg.max_unrealized_fraction
        ):
            L2 = False
            l2_reason = (
                f"unrealized_too_high({unrealized_fraction:.2f}"
                f">{cfg.max_unrealized_fraction:.2f})"
            )
        else:
            L2 = True
            l2_reason = "execution_viable"
        details["l2_reason"] = l2_reason
        details["total_fills"] = total_fills
        if unrealized_fraction is not None:
            details["unrealized_fraction"] = round(unrealized_fraction, 4)

        # ── L3: Net Return Hurdle ─────────────────────────────────────────
        # Prefer realized PnL when available (alignment-v1); fall back to MTM.
        l3_pnl = realized_net_pnl_pct if realized_net_pnl_pct is not None else net_pnl_pct
        l3_source = "realized" if realized_net_pnl_pct is not None else "mtm"
        if not L2:
            L3 = False
            l3_reason = "l2_failed"
        elif l3_pnl < cfg.hurdle_pct:
            L3 = False
            l3_reason = f"pnl_below_hurdle({l3_pnl:.2f}<{cfg.hurdle_pct:.2f})"
        else:
            L3 = True
            l3_reason = "hurdle_cleared"
        details["l3_reason"] = l3_reason
        details["l3_pnl_source"] = l3_source
        details["l3_pnl_value"] = round(l3_pnl, 4)
        details["net_pnl_pct"] = round(net_pnl_pct, 4)
        details["total_costs"] = round(fees_paid + funding_fees, 4)

        # ── Meta: Deploy ──────────────────────────────────────────────────
        meta = L1 and L2 and L3

        # Compute hlabel level
        if not L1:
            hlabel = 0
        elif not L2:
            hlabel = 1
        elif not L3:
            hlabel = 2
        else:
            hlabel = 3

        reason_parts = [l1_reason]
        if L1:
            reason_parts.append(l2_reason)
        if L2:
            reason_parts.append(l3_reason)
        reason = " → ".join(reason_parts)

        return HierarchicalLabelResult(
            hlabel=hlabel,
            L1=L1,
            L2=L2,
            L3=L3,
            meta=meta,
            reason=reason,
            details=details,
        )
