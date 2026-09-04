"""
Percentile-based adaptive microstructure gates with EMA smoothing.

Replaces fixed thresholds (max_round_trip_pct=1.20, etc.) with data-driven
thresholds based on the 75th percentile of the current batch, smoothed via
EMA for stability:

    τ_t = α × p75(batch) + (1 − α) × τ_{t-1}

Safety floors prevent thresholds from dropping to unrealistic levels.
Cold-start uses existing fixed thresholds from MicrostructureGateConfig.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveMicroGateConfig:
    """Configuration for adaptive microstructure gates."""
    ema_alpha: float = 0.30  # EMA smoothing factor
    percentile: float = 75.0  # p75 threshold
    round_trip_floor: float = 0.50  # Absolute minimum
    spread_ratio_floor: float = 0.30
    funding_drag_floor: float = 0.30
    cache_path: str = "data/cache/micro_gate_ema.json"


@dataclass(frozen=True)
class AdaptiveGateThresholds:
    """Immutable adaptive gate thresholds for a given batch."""
    round_trip_pct: float
    spread_to_profit_ratio: float
    funding_drag_pct: float
    source: str  # "adaptive", "fixed_floor", or "cold_start"


class AdaptiveMicrostructureGate:
    """Percentile-based adaptive microstructure gate with EMA smoothing.

    Usage::

        gate = AdaptiveMicrostructureGate()
        thresholds = gate.compute_batch_thresholds(batch_metrics)
        passed, reasons = gate.evaluate(candidate_metrics, thresholds)
    """

    def __init__(self, config: Optional[AdaptiveMicroGateConfig] = None) -> None:
        self._cfg = config or AdaptiveMicroGateConfig()

    def _load_ema_state(self) -> Optional[Dict[str, float]]:
        """Load EMA state from cache file."""
        path = Path(self._cfg.cache_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data
        except Exception as e:
            logger.warning("Failed to load EMA state from %s: %s", path, e)
            return None

    def _save_ema_state(self, state: Dict[str, float]) -> None:
        """Save EMA state to cache file."""
        path = Path(self._cfg.cache_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning("Failed to save EMA state to %s: %s", path, e)

    def compute_batch_thresholds(
        self,
        batch_metrics: Dict[str, np.ndarray],
    ) -> AdaptiveGateThresholds:
        """Compute adaptive thresholds from a batch of microstructure metrics.

        Parameters
        ----------
        batch_metrics : dict
            Keys: "round_trip_pct", "spread_to_profit_ratio", "funding_drag_pct"
            Values: arrays of metric values from the current scan batch.

        Returns
        -------
        AdaptiveGateThresholds
        """
        cfg = self._cfg
        metric_keys = ["round_trip_pct", "spread_to_profit_ratio", "funding_drag_pct"]
        floors = {
            "round_trip_pct": cfg.round_trip_floor,
            "spread_to_profit_ratio": cfg.spread_ratio_floor,
            "funding_drag_pct": cfg.funding_drag_floor,
        }

        # Cold-start defaults (from existing MicrostructureGateConfig)
        cold_defaults = {
            "round_trip_pct": 1.20,
            "spread_to_profit_ratio": 0.60,
            "funding_drag_pct": 0.80,
        }

        # Load previous EMA
        prev = self._load_ema_state()

        # Compute p75 for each metric
        batch_p75: Dict[str, float] = {}
        for key in metric_keys:
            arr = batch_metrics.get(key)
            if arr is not None:
                vals = np.asarray(arr, dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) >= 3:
                    batch_p75[key] = float(np.percentile(vals, cfg.percentile))
                else:
                    batch_p75[key] = cold_defaults[key]
            else:
                batch_p75[key] = cold_defaults[key]

        # Apply EMA
        new_state: Dict[str, float] = {}
        source = "adaptive"

        if prev is None:
            # Cold start: use batch p75 directly (or cold defaults)
            for key in metric_keys:
                new_state[key] = batch_p75[key]
            source = "cold_start"
            logger.info("AdaptiveMicroGate: cold start — using batch p75 directly")
        else:
            for key in metric_keys:
                prev_val = prev.get(key, cold_defaults[key])
                new_val = cfg.ema_alpha * batch_p75[key] + (1.0 - cfg.ema_alpha) * prev_val
                new_state[key] = new_val

        # Enforce floors
        for key in metric_keys:
            if new_state[key] < floors[key]:
                logger.debug(
                    "AdaptiveMicroGate: %s=%.4f < floor %.4f — clamped",
                    key, new_state[key], floors[key],
                )
                new_state[key] = floors[key]
                if source in ("adaptive", "cold_start"):
                    source = "fixed_floor"

        # Save updated EMA
        self._save_ema_state(new_state)

        thresholds = AdaptiveGateThresholds(
            round_trip_pct=float(new_state["round_trip_pct"]),
            spread_to_profit_ratio=float(new_state["spread_to_profit_ratio"]),
            funding_drag_pct=float(new_state["funding_drag_pct"]),
            source=source,
        )

        logger.info(
            "AdaptiveMicroGate: thresholds round_trip=%.4f, spread_ratio=%.4f, "
            "funding_drag=%.4f (source=%s)",
            thresholds.round_trip_pct,
            thresholds.spread_to_profit_ratio,
            thresholds.funding_drag_pct,
            source,
        )

        return thresholds

    def evaluate(
        self,
        candidate_metrics: Dict[str, Optional[float]],
        thresholds: AdaptiveGateThresholds,
    ) -> tuple[bool, List[str]]:
        """Evaluate a single candidate against adaptive thresholds.

        Parameters
        ----------
        candidate_metrics : dict
            Keys: "round_trip_pct", "spread_to_profit_ratio", "funding_drag_pct"
            Values: metric values for this candidate (None entries are skipped).
        thresholds : AdaptiveGateThresholds
            Current adaptive thresholds.

        Returns
        -------
        (passed, rejection_reasons)
        """
        reasons: List[str] = []

        rt = candidate_metrics.get("round_trip_pct")
        if rt is not None and rt > thresholds.round_trip_pct:
            reasons.append(
                f"round_trip_pct({rt:.4f})>{thresholds.round_trip_pct:.4f}"
            )

        sr = candidate_metrics.get("spread_to_profit_ratio")
        if sr is not None and sr > thresholds.spread_to_profit_ratio:
            reasons.append(
                f"spread_ratio({sr:.4f})>{thresholds.spread_to_profit_ratio:.4f}"
            )

        fd = candidate_metrics.get("funding_drag_pct")
        if fd is not None and fd > thresholds.funding_drag_pct:
            reasons.append(
                f"funding_drag({fd:.4f})>{thresholds.funding_drag_pct:.4f}"
            )

        passed = len(reasons) == 0
        return passed, reasons
