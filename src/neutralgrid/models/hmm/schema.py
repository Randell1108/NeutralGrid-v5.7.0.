"""
HMM feature schema - single source of truth for feature definitions.

This module defines the AUTHORITATIVE feature schema for HMM regime detection.
Both training and inference MUST use this schema to prevent mismatch.

Design principles:
- Single source of truth for feature names and order
- Validation functions to catch schema violations
- Documentation of each feature
"""

from typing import Any, Optional, Tuple, List
import pandas as pd
import numpy as np

from neutralgrid.core.config import get_config

# =============================================================================
# AUTHORITATIVE FEATURE SCHEMA
# =============================================================================

# Base features (always present)
_BASE_FEATURE_NAMES: Tuple[str, ...] = (
    "r_t",        # Log return over last bar
    "vol_t",      # Rolling volatility (std of log returns)
    "trend_t",    # EMA slope proxy (normalized regression slope)
    "adx_t",      # ADX trend strength indicator
    "bbwidth_t",  # Bollinger bandwidth (volatility measure)
)

FEATURE_NAMES: Tuple[str, ...] = _BASE_FEATURE_NAMES


FEATURE_DESCRIPTIONS = {
    "r_t": "Log return: ln(close[t] / close[t-1])",
    "vol_t": "Rolling volatility: rolling std of log returns over window",
    "trend_t": "EMA slope: normalized linear regression slope of EMA",
    "adx_t": "ADX: Average Directional Index (trend strength)",
    "bbwidth_t": "Bollinger bandwidth: (upper_band - lower_band) / middle_band",
}

# Required columns in input DataFrame for feature computation
REQUIRED_KLINE_COLUMNS = ["open", "high", "low", "close", "volume"]

# Timeframe used for HMM features
TIMEFRAME = "15m"


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_feature_matrix(X: np.ndarray, feature_names: List[str]) -> None:
    """
    Validate that feature matrix matches expected schema.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        feature_names: List of feature names in same order as X columns

    Raises:
        ValueError: If feature matrix doesn't match schema
    """
    if X.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {X.shape}")

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature matrix has {X.shape[1]} features, "
            f"expected {len(FEATURE_NAMES)}: {FEATURE_NAMES}"
        )

    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(
            f"Feature names mismatch!\n"
            f"  Expected: {FEATURE_NAMES}\n"
            f"  Got:      {tuple(feature_names)}"
        )


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that DataFrame has required columns for feature computation.

    Args:
        df: Kline DataFrame

    Raises:
        ValueError: If DataFrame doesn't have required columns
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")

    missing_cols = set(REQUIRED_KLINE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}\n"
            f"Required: {REQUIRED_KLINE_COLUMNS}\n"
            f"Found: {list(df.columns)}"
        )


def validate_artifact_features(artifact_features: List[str]) -> None:
    """
    Validate that artifact features match current schema.

    This is critical for preventing training/inference mismatch.

    Args:
        artifact_features: Feature list from loaded artifact metadata

    Raises:
        ValueError: If artifact features don't match current schema
    """
    if tuple(artifact_features) != FEATURE_NAMES:
        raise ValueError(
            f"FATAL: Artifact feature schema mismatch!\n"
            f"  Artifact features: {artifact_features}\n"
            f"  Current schema:    {list(FEATURE_NAMES)}\n"
            f"\n"
            f"This means the loaded model was trained on different features\n"
            f"than the current code expects. This will cause silent inference errors!\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Retrain the model with current feature schema\n"
            f"  2. Update the code to match artifact feature schema\n"
            f"  3. Use artifact versioning to load compatible model"
        )


def get_feature_schema() -> dict[str, Any]:
    """
    Get complete feature schema as a dictionary for artifact metadata.

    Returns:
        Dictionary with feature schema information
    """
    feature_schema = get_config().hmm.feature_schema.lower()
    return {
        "features": list(FEATURE_NAMES),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "timeframe": TIMEFRAME,
        "required_kline_columns": REQUIRED_KLINE_COLUMNS,
        "schema_version": "1.0.0",
        "feature_schema": feature_schema,
    }


def validate_artifact_schema_version(artifact_metadata: dict) -> None:
    """Validate that artifact feature schema matches current config.

    When loading a trained model artifact, this checks that the feature
    transform version (v1 vs v2) matches what the current config expects.
    A mismatch would cause silent inference errors because the model was
    trained on differently-transformed features.

    Args:
        artifact_metadata: Metadata dict from the loaded artifact.

    Raises:
        ValueError: If artifact feature schema doesn't match current config.
    """
    current_schema = get_config().hmm.feature_schema.lower()
    artifact_schema = artifact_metadata.get("feature_schema", "v1")
    if artifact_schema != current_schema:
        raise ValueError(
            f"FATAL: Feature schema mismatch!\n"
            f"  Artifact feature_schema: {artifact_schema}\n"
            f"  Current HMM_FEATURE_SCHEMA: {current_schema}\n"
            f"\n"
            f"The loaded model was trained with {artifact_schema} features but\n"
            f"the current config expects {current_schema}. This will cause\n"
            f"silent inference errors.\n"
            f"\n"
            f"Solutions:\n"
            f"  1. Retrain the model with HMM_FEATURE_SCHEMA={current_schema}\n"
            f"  2. Set HMM_FEATURE_SCHEMA={artifact_schema} in config.py"
        )


def _stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute stationary distribution from a transition matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, idx])
    # Eigenvector sign is arbitrary; flip if sum is negative
    if pi.sum() < 0:
        pi = -pi
    total = pi.sum()
    if total > 0:
        pi = pi / total
    return pi


def infer_state_mapping(
    state_means_unscaled: np.ndarray,
    transmat: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """
    Infer which HMM states correspond to TREND vs RANGE regimes.

    Uses trendiness heuristic: ADX + |trend| + BB width.
    Higher values -> more trendy.

    When ``transmat`` is supplied and there are 3+ states, the stationary
    distribution is computed and states with < 1% occupation are excluded
    from range candidacy.  This prevents labelling a "dead" outlier cluster
    as the range state.

    CANONICAL IMPLEMENTATION - All other files must import from here.

    Args:
        state_means_unscaled: Unscaled state means (n_states, n_features)
        transmat: Optional transition matrix (n_states, n_states)

    Returns:
        Tuple of (trend_state, range_state)

    Edge case: If input is malformed (ndim != 2 or shape[0] < 2),
               returns (0, 1) as default distinct states.
    """
    if state_means_unscaled.ndim != 2 or state_means_unscaled.shape[0] < 2:
        return 0, 1  # CRITICAL: Must return distinct states

    n_states = state_means_unscaled.shape[0]

    idx_trend = FEATURE_NAMES.index("trend_t")
    idx_adx = FEATURE_NAMES.index("adx_t")
    idx_bbw = FEATURE_NAMES.index("bbwidth_t")

    adx = state_means_unscaled[:, idx_adx]
    tr = np.abs(state_means_unscaled[:, idx_trend])
    bbw = state_means_unscaled[:, idx_bbw]

    def z(x: np.ndarray) -> np.ndarray:
        s = float(np.nanstd(x))
        if not np.isfinite(s) or s <= 0:
            return x * 0.0
        return (x - float(np.nanmean(x))) / s

    trendiness = z(adx) + z(tr) + z(bbw)

    trend_state = int(np.nanargmax(trendiness))

    # For range state: among non-trend states, pick the one with lowest
    # trendiness that is NOT a dead state (occupation >= 1%).
    if transmat is not None and n_states >= 3:
        pi = _stationary_distribution(transmat)
        candidates = [
            i for i in range(n_states)
            if i != trend_state and pi[i] >= 0.01
        ]
        if candidates:
            range_state = min(candidates, key=lambda i: trendiness[i])
        else:
            # All non-trend states are dead — fall back to argmin
            range_state = int(np.nanargmin(trendiness))
    else:
        range_state = int(np.nanargmin(trendiness))

    return trend_state, range_state


def infer_state_mapping_directional(
    state_means_unscaled: np.ndarray,
) -> dict[str, Any]:
    """
    Direction-aware state mapping: separate bull-trend, bear-trend, and range.

    Uses signed trend_t (no abs) to distinguish bullish from bearish trends.

    Args:
        state_means_unscaled: Unscaled state means (n_states, n_features)

    Returns:
        Dictionary with keys:
        - trend_state: int (highest trendiness — legacy compat)
        - range_state: int (lowest trendiness)
        - bull_trend_state: int | None (positive trend_t among trend states)
        - bear_trend_state: int | None (negative trend_t among trend states)
        - direction_aware: True
    """
    # Fall back for bad input
    if state_means_unscaled.ndim != 2 or state_means_unscaled.shape[0] < 2:
        return {
            "trend_state": 0,
            "range_state": 1,
            "bull_trend_state": None,
            "bear_trend_state": None,
            "direction_aware": True,
        }

    idx_trend = FEATURE_NAMES.index("trend_t")
    idx_adx = FEATURE_NAMES.index("adx_t")
    idx_bbw = FEATURE_NAMES.index("bbwidth_t")

    adx = state_means_unscaled[:, idx_adx]
    tr_signed = state_means_unscaled[:, idx_trend]  # signed — NO abs()
    bbw = state_means_unscaled[:, idx_bbw]

    def z(x: np.ndarray) -> np.ndarray:
        s = float(np.nanstd(x))
        if not np.isfinite(s) or s <= 0:
            return x * 0.0
        return (x - float(np.nanmean(x))) / s

    # Trendiness still uses abs for ranking, to find the trend vs range split
    trendiness = z(adx) + z(np.abs(tr_signed)) + z(bbw)

    range_state = int(np.nanargmin(trendiness))

    # Among non-range states, classify by sign of trend_t
    non_range = [i for i in range(state_means_unscaled.shape[0]) if i != range_state]

    bull_trend_state = None
    bear_trend_state = None

    if len(non_range) == 1:
        # Only 2 states total — single trend state, classify by sign
        ts = non_range[0]
        if tr_signed[ts] >= 0:
            bull_trend_state = ts
        else:
            bear_trend_state = ts
    else:
        # 3+ states — find the most bullish and most bearish among trending states
        trend_vals = [(i, tr_signed[i]) for i in non_range]
        trend_vals.sort(key=lambda x: x[1])
        bear_trend_state = trend_vals[0][0]   # most negative trend_t
        bull_trend_state = trend_vals[-1][0]   # most positive trend_t

    # Legacy compat: trend_state = highest overall trendiness
    trend_state = int(np.nanargmax(trendiness))

    return {
        "trend_state": trend_state,
        "range_state": range_state,
        "bull_trend_state": bull_trend_state,
        "bear_trend_state": bear_trend_state,
        "direction_aware": True,
    }


def compute_state_groups(
    state_means_unscaled: np.ndarray,
) -> dict[str, Any]:
    """Partition HMM states into range-like and trend-like groups by trendiness.

    With >2 states, single-state range_prob/trend_prob can miss probability
    mass in "intermediate" states. This function splits ALL states into two
    groups at the median trendiness so aggregated probabilities always sum
    to 1.0.

    Args:
        state_means_unscaled: Unscaled state means (n_states, n_features)

    Returns:
        Dictionary with:
        - range_states: list of state indices with below-median trendiness
        - trend_states: list of state indices with above-median trendiness
        - trendiness: per-state trendiness scores
    """
    n = state_means_unscaled.shape[0]
    if state_means_unscaled.ndim != 2 or n < 2:
        return {"range_states": [0], "trend_states": [1] if n > 1 else [], "trendiness": []}

    idx_trend = FEATURE_NAMES.index("trend_t")
    idx_adx = FEATURE_NAMES.index("adx_t")
    idx_bbw = FEATURE_NAMES.index("bbwidth_t")

    adx = state_means_unscaled[:, idx_adx]
    tr = np.abs(state_means_unscaled[:, idx_trend])
    bbw = state_means_unscaled[:, idx_bbw]

    def z(x: np.ndarray) -> np.ndarray:
        s = float(np.nanstd(x))
        if not np.isfinite(s) or s <= 0:
            return x * 0.0
        return (x - float(np.nanmean(x))) / s

    trendiness = z(adx) + z(tr) + z(bbw)
    median_t = float(np.median(trendiness))

    range_states = sorted(int(i) for i in range(n) if trendiness[i] <= median_t)
    trend_states = sorted(int(i) for i in range(n) if trendiness[i] > median_t)

    # Ensure at least one state per group
    if not range_states:
        range_states = [int(np.argmin(trendiness))]
        trend_states = [i for i in range(n) if i not in range_states]
    if not trend_states:
        trend_states = [int(np.argmax(trendiness))]
        range_states = [i for i in range(n) if i not in trend_states]

    return {
        "range_states": range_states,
        "trend_states": trend_states,
        "trendiness": trendiness.tolist(),
    }
