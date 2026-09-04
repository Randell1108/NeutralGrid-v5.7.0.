"""
Entropy-adaptive regime probability thresholds.

Replaces the fixed ``range_prob_min`` threshold with entropy-conditional
tiers that tighten when the HMM is confident and relax when uncertain.

Entropy tiers for a 4-state HMM (max entropy = ln(4) ≈ 1.386):
  H < 0.5   → high-confidence tier  → range_prob_min = 0.35 (tighter)
  0.5 ≤ H < 1.0 → moderate tier     → range_prob_min = 0.25 (default)
  H ≥ 1.0   → low-confidence tier   → range_prob_min = 0.20 (lenient)

This improves the 90% funnel rejection rate by not filtering out
candidates when the HMM itself is uncertain (high-entropy posteriors).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropyThresholdConfig:
    """Configuration for entropy-adaptive regime thresholds."""
    high_confidence_threshold: float = 0.35  # H < 0.5 → tighter gate
    moderate_confidence_threshold: float = 0.25  # 0.5 ≤ H < 1.0
    low_confidence_threshold: float = 0.20  # H ≥ 1.0 → lenient
    entropy_bins: Tuple[float, float] = (0.5, 1.0)  # Bin edges
    max_entropy: float = 1.3863  # ln(4) for 4-state HMM


def compute_posterior_entropy(posteriors: np.ndarray) -> float:
    """Compute Shannon entropy of a posterior distribution.

    H = -Σ p_k · log(p_k)   with convention 0·log(0) = 0

    Parameters
    ----------
    posteriors : array of shape (K,)
        Probability distribution summing to ~1.0.

    Returns
    -------
    float
        Shannon entropy in nats.
    """
    p = np.asarray(posteriors, dtype=float).reshape(-1)

    # Handle degenerate cases
    if len(p) == 0:
        return 0.0

    # Remove non-finite
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return 0.0

    # Normalize to ensure valid distribution
    total = float(np.sum(p))
    if total <= 0:
        return 0.0
    p = p / total

    # Shannon entropy: 0 * log(0) = 0 by convention
    entropy = 0.0
    for pk in p:
        if pk > 0:
            entropy -= float(pk * math.log(pk))

    return float(entropy)


def select_range_prob_threshold(
    posteriors: np.ndarray,
    config: Optional[EntropyThresholdConfig] = None,
) -> tuple[float, str]:
    """Select the range_prob_min threshold based on posterior entropy.

    Parameters
    ----------
    posteriors : array of shape (K,)
        Current HMM posterior distribution.
    config : EntropyThresholdConfig | None
        Threshold configuration. Uses defaults if None.

    Returns
    -------
    (threshold, tier_name)
        threshold: the range_prob_min to use for this candidate
        tier_name: "high_confidence", "moderate", or "low_confidence"
    """
    cfg = config or EntropyThresholdConfig()
    H = compute_posterior_entropy(posteriors)
    lo, hi = cfg.entropy_bins

    if H < lo:
        threshold = cfg.high_confidence_threshold
        tier = "high_confidence"
    elif H < hi:
        threshold = cfg.moderate_confidence_threshold
        tier = "moderate"
    else:
        threshold = cfg.low_confidence_threshold
        tier = "low_confidence"

    logger.debug(
        "Entropy-adaptive threshold: H=%.4f → tier=%s → range_prob_min=%.3f",
        H, tier, threshold,
    )
    return float(threshold), tier


def compute_confidence_weighted_priority(
    scanner_score: float,
    posteriors: np.ndarray,
    config: Optional[EntropyThresholdConfig] = None,
) -> float:
    """Compute confidence-weighted priority score for candidate sorting.

    priority = scanner_score × (1 − H_normalized)

    where H_normalized = H / max_entropy ∈ [0, 1].

    High-confidence candidates (low H) get boosted; uncertain ones are
    deprioritized when the deployment budget is reached.

    Parameters
    ----------
    scanner_score : float
        Base scanner score for the candidate.
    posteriors : array of shape (K,)
        HMM posterior distribution.
    config : EntropyThresholdConfig | None
        Configuration with max_entropy.

    Returns
    -------
    float
        Adjusted priority score.
    """
    cfg = config or EntropyThresholdConfig()
    H = compute_posterior_entropy(posteriors)
    h_norm = min(H / cfg.max_entropy, 1.0) if cfg.max_entropy > 0 else 0.0
    priority = float(scanner_score) * (1.0 - h_norm)

    return float(priority)
