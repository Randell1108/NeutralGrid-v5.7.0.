"""
Phase 1 meta-labeler recalibration configuration and validation.

Provides recommended Phase 1 overrides that reduce GBM complexity to match
the available sample size (~190 rows), switch from isotonic to sigmoid
(Platt scaling) calibration, and enable time-decay weighting with N_eff guard.

Key changes from default MetaLabelerConfig:
  n_estimators:  100 → 30    (reduce leaf-node capacity)
  max_depth:       5 → 3     (shallower trees)
  min_samples_leaf: 5 → 15   (prevent micro-splits)
  learning_rate: 0.10 → 0.05 (smaller steps)
  calibration_method: isotonic → sigmoid  (2-param Platt; stable at N < 200)
  cv_folds:        6 → 4     (larger train folds for small data)
  use_time_decay:   False → True
  time_decay_halflife_days: 30.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Phase1MetaLabelerOverrides:
    """Frozen dataclass with Phase 1 recalibrated meta-labeler settings."""
    n_estimators: int = 30
    max_depth: int = 3
    min_samples_leaf: int = 15
    learning_rate: float = 0.05
    calibration_method: str = "sigmoid"
    cv_folds: int = 4
    use_time_decay: bool = True
    time_decay_halflife_days: float = 30.0


def build_phase1_meta_config() -> Dict[str, Any]:
    """Return a dict of Phase 1 overrides suitable for MetaLabelerConfig construction.

    Usage::

        from neutralgrid.calibration.phase1_config_v20260311 import build_phase1_meta_config
        overrides = build_phase1_meta_config()
        # Merge with existing MetaLabelerConfig fields
        config = MetaLabelerConfig(**{**defaults, **overrides})

    Returns
    -------
    dict
        Field-name → value mapping matching MetaLabelerConfig attributes.
    """
    p1 = Phase1MetaLabelerOverrides()
    return {
        "n_estimators": p1.n_estimators,
        "max_depth": p1.max_depth,
        "min_samples_leaf": p1.min_samples_leaf,
        "learning_rate": p1.learning_rate,
        "calibration_method": p1.calibration_method,
        "cv_folds": p1.cv_folds,
        "use_time_decay": p1.use_time_decay,
        "time_decay_halflife_days": p1.time_decay_halflife_days,
    }


def validate_phase1_auc_floor(
    auc_cv: float,
    floor: float = 0.56,
) -> tuple[bool, str]:
    """Validate that the Phase 1 reduced-complexity model meets AUC floor.

    The floor is derived from the null hypothesis test:
        AUC_null = 0.50
        σ(AUC) ≈ 1 / √(4 × N) for N ≈ 190 → σ ≈ 0.036
        Floor = AUC_null + 1σ ≈ 0.536, rounded up to 0.56 for margin.

    If the CV AUC falls below this floor, the reduced model is worse than
    chance at the 1σ level and the caller should fall back to default config.

    Parameters
    ----------
    auc_cv : float
        Cross-validated AUC from the Phase 1 model.
    floor : float
        Minimum acceptable AUC.  Default 0.56.

    Returns
    -------
    (passes, reason) : tuple[bool, str]
    """
    import math

    if not math.isfinite(auc_cv):
        return False, f"non_finite_auc: {auc_cv}"

    passes = auc_cv >= floor
    if passes:
        margin = auc_cv - floor
        reason = (
            f"AUC={auc_cv:.4f} ≥ {floor:.4f} (margin=+{margin:.4f}) — "
            "Phase 1 model passes floor check"
        )
    else:
        deficit = floor - auc_cv
        reason = (
            f"AUC={auc_cv:.4f} < {floor:.4f} (deficit={deficit:.4f}) — "
            "Phase 1 reduced model fails; recommend fallback to default config"
        )

    logger.info("Phase1 AUC floor check: %s", reason)
    return passes, reason
