"""
Mutual-Information weighted scanner scoring with Bayesian shrinkage.

Replaces the fixed scoring weights (similarity=0.30, profile_proba=0.35,
ev_score=0.20, meta_prob=0.25) with data-driven weights estimated from
MI(signal, target) with shrinkage toward uniform:

    w_final = (1 − α) · w_MI + α · w_uniform

where α = 0.30 (matching existing retrain_scanner.py shrinkage parameter).

Falls back to uniform weights when max(MI) < MI_null threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MIWeightConfig:
    """Configuration for MI-weighted scoring."""
    shrinkage: float = 0.30  # Bayesian shrinkage toward uniform
    mi_null_threshold: float = 0.015  # 1/(2·48·ln(2)) for N=48
    uniform_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    signal_names: Tuple[str, ...] = (
        "similarity_score",
        "profile_proba",
        "ev_score",
        "meta_prob",
    )


def estimate_mutual_information(
    feature: np.ndarray,
    target: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """Estimate mutual information between a continuous feature and discrete target.

    Parameters
    ----------
    feature : array of shape (N,)
        Continuous signal values.
    target : array of shape (N,)
        Binary target labels.
    n_neighbors : int
        Number of neighbors for MI estimation.

    Returns
    -------
    float
        Estimated MI in nats. Returns 0.0 if estimation fails.
    """
    f = np.asarray(feature, dtype=float).reshape(-1)
    t = np.asarray(target, dtype=int).reshape(-1)

    # Remove NaN pairs
    valid = np.isfinite(f)
    f, t = f[valid], t[valid]

    if len(f) < max(n_neighbors + 1, 5):
        logger.debug(
            "MI estimation: too few samples (%d) — returning 0.0", len(f)
        )
        return 0.0

    # Check variance
    if np.std(f) < 1e-10:
        return 0.0

    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(
            f.reshape(-1, 1),
            t,
            discrete_features="auto",
            n_neighbors=n_neighbors,
            random_state=42,
        )
        return float(max(mi[0], 0.0))
    except ImportError:
        logger.warning("sklearn not available for MI estimation — returning 0.0")
        return 0.0
    except Exception as e:
        logger.warning("MI estimation failed: %s — returning 0.0", e)
        return 0.0


def compute_mi_weights(
    signals: Dict[str, np.ndarray],
    target: np.ndarray,
    config: Optional[MIWeightConfig] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Compute MI-weighted scoring weights with Bayesian shrinkage.

    Parameters
    ----------
    signals : dict
        Mapping signal_name → array of signal values.
    target : array of shape (N,)
        Binary target labels.
    config : MIWeightConfig | None
        Configuration. Uses defaults if None.

    Returns
    -------
    (final_weights, raw_mi_values) : tuple
        final_weights: signal_name → final weight (normalized, sum to 1.0)
        raw_mi_values: signal_name → raw MI estimate
    """
    cfg = config or MIWeightConfig()
    t = np.asarray(target, dtype=int).reshape(-1)

    # Compute MI for each signal
    raw_mi: Dict[str, float] = {}
    for name in cfg.signal_names:
        if name in signals:
            mi = estimate_mutual_information(signals[name], t)
            raw_mi[name] = mi
        else:
            raw_mi[name] = 0.0
            logger.debug("MI weights: signal '%s' not in signals dict", name)

    # Check if any signal has meaningful MI
    max_mi = max(raw_mi.values()) if raw_mi else 0.0
    n_signals = len(cfg.signal_names)

    if max_mi < cfg.mi_null_threshold:
        logger.info(
            "MI weights: max(MI)=%.6f < null threshold %.6f — using uniform",
            max_mi, cfg.mi_null_threshold,
        )
        uniform = {
            name: 1.0 / n_signals
            for name in cfg.signal_names
        }
        return uniform, raw_mi

    # Normalize MI to get raw data-driven weights
    total_mi = sum(raw_mi.values())
    if total_mi < 1e-12:
        w_mi = {name: 1.0 / n_signals for name in cfg.signal_names}
    else:
        w_mi = {name: raw_mi[name] / total_mi for name in cfg.signal_names}

    # Apply Bayesian shrinkage: w_final = (1-α)·w_MI + α·w_uniform
    alpha = cfg.shrinkage
    w_uniform = 1.0 / n_signals
    final_weights = {
        name: (1.0 - alpha) * w_mi[name] + alpha * w_uniform
        for name in cfg.signal_names
    }

    # Normalize to sum to 1.0 (should already be close, but ensure)
    total = sum(final_weights.values())
    if total > 0:
        final_weights = {name: w / total for name, w in final_weights.items()}

    logger.info(
        "MI weights: raw_MI=%s, final=%s (shrinkage=%.2f)",
        {k: f"{v:.4f}" for k, v in raw_mi.items()},
        {k: f"{v:.4f}" for k, v in final_weights.items()},
        alpha,
    )

    return final_weights, raw_mi


_EXPECTED_RUNTIME_SIGNALS: frozenset[str] = frozenset({
    "similarity_score",
    "profile_proba",
    "ev_score",
    "meta_prob",
})


def default_runtime_mi_weights() -> Dict[str, float]:
    """Return the canonical uniform scan-time MI weights."""
    cfg = MIWeightConfig()
    return {
        name: float(weight)
        for name, weight in zip(cfg.signal_names, cfg.uniform_weights)
    }


def validate_mi_weights(
    weights: Dict[str, float],
    expected_signals: frozenset[str] | None = None,
) -> Dict[str, float] | None:
    """Validate MI weights for runtime signal compatibility.

    Checks that the signal names in *weights* are a subset of the expected
    runtime signals and that no unexpected names are present.  Returns the
    validated weights dict if valid, or ``None`` (uniform fallback) if
    validation fails.

    Parameters
    ----------
    weights : dict
        Signal name -> weight mapping loaded from the artifact.
    expected_signals : frozenset[str] | None
        Canonical set of runtime signal names.  Defaults to
        ``_EXPECTED_RUNTIME_SIGNALS``.

    Returns
    -------
    dict[str, float] | None
        The original *weights* if valid, ``None`` otherwise.
    """
    if expected_signals is None:
        expected_signals = _EXPECTED_RUNTIME_SIGNALS

    if not weights:
        logger.warning("MI weights validation: empty weights dict — using uniform")
        return None

    weight_names = set(weights.keys())
    unexpected = weight_names - expected_signals
    if unexpected:
        logger.warning(
            "MI weights validation: unexpected signal names %s "
            "(expected subset of %s) — using uniform weights",
            sorted(unexpected),
            sorted(expected_signals),
        )
        return None

    missing = expected_signals - weight_names
    if missing:
        logger.info(
            "MI weights validation: missing signals %s — "
            "present signals will retain their configured weights",
            sorted(missing),
        )

    # Weights must be finite, non-negative, and sum to a positive value
    if any(not np.isfinite(v) for v in weights.values()):
        logger.warning(
            "MI weights validation: non-finite weight detected — using uniform"
        )
        return None

    if any(v < 0 for v in weights.values()):
        logger.warning(
            "MI weights validation: negative weight detected — using uniform"
        )
        return None

    total = sum(weights.values())
    if total <= 0:
        logger.warning(
            "MI weights validation: total weight <= 0 — using uniform"
        )
        return None

    return weights


def compute_mi_weighted_score(
    features: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> float:
    """Compute MI-weighted composite score.

    Parameters
    ----------
    features : dict
        signal_name → signal value for this candidate. None for missing signals.
    weights : dict
        signal_name → weight (from compute_mi_weights).

    Returns
    -------
    float
        Weighted average over finite, available features. Missing/None features
        are excluded from both numerator and denominator. Returns 0.0 when no
        weighted feature is available.
    """
    score = 0.0
    available_weight = 0.0

    for name, w in weights.items():
        if not np.isfinite(w) or float(w) <= 0.0:
            continue
        val = features.get(name)
        if val is not None and np.isfinite(val):
            score += float(val) * w
            available_weight += float(w)

    if available_weight <= 0.0:
        return 0.0

    return float(score / available_weight)
