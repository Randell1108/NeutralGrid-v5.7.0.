"""Feature multicollinearity diagnostics (AFML Ch 8).

Computes VIF and pairwise correlations to flag collinear features.
Does NOT remove features automatically -- logs informational diagnostics
for human review.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VIF_THRESHOLD = 5.0
CORR_THRESHOLD = 0.7


def _compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute Variance Inflation Factor for each column."""
    n_features = X.shape[1]
    vifs = np.zeros(n_features)
    for i in range(n_features):
        mask = np.ones(n_features, dtype=bool)
        mask[i] = False
        y_i = X[:, i]
        X_rest = X[:, mask]
        # Add intercept
        ones = np.ones((X_rest.shape[0], 1))
        X_aug = np.hstack([ones, X_rest])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_i, rcond=None)
            y_hat = X_aug @ coeffs
            ss_res = np.sum((y_i - y_hat) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[i] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf
        except Exception:
            vifs[i] = np.nan
    return vifs


def compute_feature_diagnostics(
    X: pd.DataFrame,
    vif_threshold: float = VIF_THRESHOLD,
    corr_threshold: float = CORR_THRESHOLD,
) -> Dict:
    """Compute multicollinearity diagnostics for a feature matrix.

    Args:
        X: Feature DataFrame (samples x features), NaN rows dropped internally.
        vif_threshold: Flag features with VIF above this (default 5.0).
        corr_threshold: Flag pairs with |r| above this (default 0.7).

    Returns:
        Dict with 'vif', 'high_vif', 'high_corr_pairs', 'correlation_matrix'.
    """
    X_clean = X.dropna()
    if X_clean.empty or X_clean.shape[1] < 2:
        logger.info("Insufficient data for multicollinearity diagnostics")
        return {"vif": {}, "high_vif": [], "high_corr_pairs": [], "correlation_matrix": {}}

    feature_names = list(X_clean.columns)
    values = np.asarray(X_clean).astype(float)

    # VIF
    vifs = _compute_vif(values)
    vif_dict = {name: round(float(v), 2) for name, v in zip(feature_names, vifs)}
    high_vif = [(name, vif_dict[name]) for name in feature_names if vif_dict[name] > vif_threshold]

    # Pairwise correlations
    corr_matrix = np.corrcoef(values, rowvar=False)
    high_corr_pairs: List[Tuple[str, str, float]] = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix[i, j]
            if abs(r) > corr_threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], round(float(r), 3)))

    # Log results
    logger.info("Feature multicollinearity diagnostics (%d features, %d samples):", len(feature_names), len(X_clean))
    if high_vif:
        for name, v in high_vif:
            logger.info("  HIGH VIF: %s = %.1f (threshold %.1f)", name, v, vif_threshold)
    else:
        logger.info("  VIF: all features below %.1f", vif_threshold)

    if high_corr_pairs:
        for f1, f2, r in high_corr_pairs:
            logger.info("  HIGH CORR: %s <-> %s  r=%.3f (threshold %.1f)", f1, f2, r, corr_threshold)
    else:
        logger.info("  Correlations: all pairs below |%.1f|", corr_threshold)

    return {
        "vif": vif_dict,
        "high_vif": high_vif,
        "high_corr_pairs": high_corr_pairs,
        "correlation_matrix": {
            name: {feature_names[j]: round(float(corr_matrix[i, j]), 3) for j in range(n)}
            for i, name in enumerate(feature_names)
        },
    }
