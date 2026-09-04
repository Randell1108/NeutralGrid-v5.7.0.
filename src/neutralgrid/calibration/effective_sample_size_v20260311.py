"""
Kish's Effective Sample Size for time-decay weight quality assurance.

When time-decay weighting is applied during meta-labeler training, the
effective sample size (N_eff) can fall well below the raw count, making
the model under-determined.  This module provides:

  - compute_kish_effective_n(): the ESS formula
  - should_disable_time_decay(): guard that disables decay when N_eff < 30
  - EffectiveSampleReport: frozen result for logging/diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectiveSampleReport:
    """Immutable report of effective sample size analysis."""
    raw_n: int
    effective_n: float
    ratio: float           # effective_n / raw_n ∈ [0, 1]
    time_decay_disabled: bool
    reason: str


def compute_kish_effective_n(weights: np.ndarray) -> float:
    """Compute Kish's effective sample size.

    Formula:  N_eff = (Σ w_i)² / Σ (w_i²)

    Parameters
    ----------
    weights : array-like
        Non-negative sample weights.

    Returns
    -------
    float
        Effective sample size.  Returns 0.0 for degenerate inputs.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)

    # Edge cases
    if len(w) == 0:
        return 0.0

    # Remove non-finite
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return 0.0

    # Negative weights are not meaningful for ESS — clamp to 0
    w = np.clip(w, 0.0, None)

    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(w ** 2))

    if sum_w2 < 1e-30:
        return 0.0

    n_eff = (sum_w ** 2) / sum_w2
    return float(n_eff)


def should_disable_time_decay(
    weights: np.ndarray,
    min_effective_n: int = 30,
) -> tuple[bool, float]:
    """Check whether time-decay should be disabled due to low N_eff.

    Parameters
    ----------
    weights : array-like
        Time-decay weights (before combining with uniqueness weights).
    min_effective_n : int
        Minimum acceptable N_eff.  Default 30 matches the
        ``oos_calibration_min_samples`` floor in MetaLabelerConfig.

    Returns
    -------
    (should_disable, n_eff) : tuple[bool, float]
    """
    n_eff = compute_kish_effective_n(weights)
    disable = n_eff < float(min_effective_n)

    if disable:
        logger.warning(
            "Time-decay N_eff=%.1f < %d minimum — disabling time-decay weighting "
            "to preserve model stability",
            n_eff, min_effective_n,
        )
    else:
        logger.info(
            "Time-decay N_eff=%.1f ≥ %d — time-decay weighting is safe",
            n_eff, min_effective_n,
        )

    return disable, float(n_eff)


def compute_effective_sample_report(
    weights: np.ndarray,
    min_effective_n: int = 30,
) -> EffectiveSampleReport:
    """Generate full diagnostic report for sample weight quality.

    Parameters
    ----------
    weights : array-like
        Sample weights (time-decay or combined).
    min_effective_n : int
        Minimum acceptable N_eff threshold.

    Returns
    -------
    EffectiveSampleReport
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    raw_n = len(w)
    n_eff = compute_kish_effective_n(w)
    ratio = float(n_eff / raw_n) if raw_n > 0 else 0.0
    disable = n_eff < float(min_effective_n)

    if raw_n == 0:
        reason = "no_samples"
    elif disable:
        reason = (
            f"N_eff={n_eff:.1f} < {min_effective_n} — time-decay too aggressive "
            f"(ratio={ratio:.3f}), disabled to avoid under-determined model"
        )
    else:
        reason = (
            f"N_eff={n_eff:.1f} ≥ {min_effective_n} — acceptable "
            f"(ratio={ratio:.3f})"
        )

    return EffectiveSampleReport(
        raw_n=raw_n,
        effective_n=float(n_eff),
        ratio=float(ratio),
        time_decay_disabled=disable,
        reason=reason,
    )
