"""
Single source of truth for feature computation.

This module provides the AUTHORITATIVE feature extraction functions used by
both training and inference. This ensures consistency and prevents
training/inference mismatch.

Design principles:
- One function per feature (clear, testable, documented)
- No duplication across training/inference code
- Type hints and validation
- Consistent with feature schema from neutralgrid.models.hmm.schema
"""

from typing import Any, Tuple
import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.indicators.technical import (
    calc_adx,
    calc_bollinger_bands,
    calc_ema,
    calc_ema_slope,
)
from neutralgrid.models.hmm.schema import (
    FEATURE_NAMES,
    validate_dataframe,
    validate_feature_matrix,
)


def compute_vol_ratio(vol: np.ndarray, long_window: int = 100) -> np.ndarray:
    """vol_t / rolling_mean(vol_t, long_window) -- stationary ratio ~1.0.

    Normalizes rolling volatility by its own long-run mean so the feature
    is approximately stationary across different market regimes (AFML Ch 5).

    Args:
        vol: Raw rolling volatility array.
        long_window: Long-run rolling mean window.

    Returns:
        Array of volatility ratios (NaN where rolling mean is unavailable).
    """
    rolling_mean = np.asarray(pd.Series(vol).rolling(long_window, min_periods=long_window).mean())
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = vol / rolling_mean
    ratio[np.isnan(rolling_mean) | (rolling_mean == 0)] = np.nan
    return ratio


def compute_trend_normalized(trend: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """trend_t / vol_t -- volatility-normalized trend, dimensionless.

    Divides the EMA slope proxy by rolling volatility so the trend signal
    is scale-free across different volatility regimes (AFML Ch 8).

    Args:
        trend: Raw EMA slope proxy array.
        vol: Raw rolling volatility array.

    Returns:
        Array of normalized trend values (NaN where vol is zero/NaN).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        normed = trend / vol
    normed[np.isnan(vol) | (vol == 0)] = np.nan
    return normed


def compute_log_returns(closes: np.ndarray) -> np.ndarray:
    """
    Compute log returns: ln(close[t] / close[t-1]).

    Args:
        closes: Array of close prices

    Returns:
        Array of log returns (first element is NaN)
    """
    r = np.empty_like(closes, dtype=np.float64)
    r[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = np.log(closes[1:] / closes[:-1])
    return r


def compute_rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling volatility as standard deviation of returns.

    Uses explicit min_periods=window to ensure NaN during warmup (AFML compliance).

    Args:
        returns: Array of returns (typically log returns)
        window: Rolling window size

    Returns:
        Array of rolling volatility (first 'window' elements are NaN)
    """
    return np.asarray(pd.Series(returns).rolling(window=window, min_periods=window).std())


def compute_ema_slope_proxy(closes: np.ndarray, ema_period: int, lookback: int = 5) -> np.ndarray:
    """
    Compute EMA slope proxy via normalized linear regression.

    Args:
        closes: Array of close prices
        ema_period: EMA period for smoothing
        lookback: Lookback for slope calculation

    Returns:
        Array of normalized EMA slopes
    """
    ema = calc_ema(closes, ema_period)
    slope = calc_ema_slope(ema, lookback=lookback)
    return slope


def compute_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """
    Compute Average Directional Index (ADX) trend strength indicator.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ADX period

    Returns:
        Array of ADX values
    """
    adx, _, _ = calc_adx(highs, lows, closes, period)
    return adx


def compute_bollinger_bandwidth(closes: np.ndarray, period: int, std_dev: float) -> np.ndarray:
    """
    Compute Bollinger bandwidth: (upper_band - lower_band) / middle_band.

    Args:
        closes: Array of close prices
        period: Bollinger period
        std_dev: Number of standard deviations

    Returns:
        Array of Bollinger bandwidth values
    """
    _, _, _, bbwidth = calc_bollinger_bands(closes, period, std_dev)
    return bbwidth


def compute_hmm_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute HMM feature matrix from OHLCV klines.

    This is the SINGLE SOURCE OF TRUTH for HMM feature computation.
    Both training and inference MUST use this function.

    Args:
        df: DataFrame with kline data (columns: open, high, low, close, volume)

    Returns:
        Tuple of:
        - X: Feature matrix of shape (n_samples, n_features) with columns in order:
             ["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"]
        - valid_mask: Boolean mask indicating which rows are valid after indicator warmup
                     (same length as df)

    Raises:
        ValueError: If DataFrame doesn't have required columns

    Example:
        >>> X, valid = compute_hmm_features(df)
        >>> X_valid = X[valid]  # Get only valid rows
        >>> # Now X_valid can be used for training or inference
    """
    if df is None or df.empty:
        return np.empty((0, len(FEATURE_NAMES))), np.zeros(0, dtype=bool)

    # Validate input
    validate_dataframe(df)

    # Enforce temporal ordering for rolling indicators
    df = df.sort_values("open_time").reset_index(drop=True)

    # Extract price arrays
    closes = np.asarray(pd.to_numeric(df["close"], errors="coerce"), dtype=float)
    highs = np.asarray(pd.to_numeric(df["high"], errors="coerce"), dtype=float)
    lows = np.asarray(pd.to_numeric(df["low"], errors="coerce"), dtype=float)

    # Get parameters from config
    _cfg = get_config()
    vol_window = _cfg.hmm.vol_window
    ema_period = _cfg.hmm.ema_period
    adx_period = _cfg.indicators.adx_period
    bb_period = _cfg.indicators.bb_period
    bb_std_dev = _cfg.indicators.bb_std_dev

    # Compute features (in schema order)
    r_t = compute_log_returns(closes)
    vol_t = compute_rolling_volatility(r_t, vol_window)
    trend_t = compute_ema_slope_proxy(closes, ema_period, lookback=5)
    adx_t = compute_adx(highs, lows, closes, adx_period)
    bbwidth_t = compute_bollinger_bandwidth(closes, bb_period, bb_std_dev)

    # Apply stationary transforms when feature schema v2 is active
    feature_schema = _cfg.hmm.feature_schema.lower()
    if feature_schema == "v2":
        vol_t = compute_vol_ratio(vol_t)
        trend_t = compute_trend_normalized(trend_t, compute_rolling_volatility(r_t, vol_window))

    # Stack base features in schema order
    columns = [r_t, vol_t, trend_t, adx_t, bbwidth_t]

    X = np.column_stack(columns).astype(float)

    # Valid rows: all finite (indicators have warmed up)
    valid_mask = np.asarray(np.isfinite(X).all(axis=1))

    # Enforce explicit warmup period so that "valid" rows begin strictly after
    # the maximum indicator lookback. This matches the unit-test expectation
    # that valid data starts after index 20 when default windows are used.
    feature_schema = _cfg.hmm.feature_schema.lower()
    vol_ratio_lookback = 100 if feature_schema == "v2" else 0
    warmup_n = max(
        vol_window + vol_ratio_lookback,
        ema_period + 5,
        2 * adx_period,
        bb_period,
    ) + 1
    if warmup_n > 0:
        valid_mask[: min(warmup_n, len(valid_mask))] = False

    # Validate feature matrix
    validate_feature_matrix(X, list(FEATURE_NAMES))

    return X, valid_mask


def compute_hmm_features_dict(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute HMM features and return as a dictionary with feature names.

    This is useful for debugging and inspection.

    Args:
        df: DataFrame with OHLCV kline data

    Returns:
        Dictionary with keys:
        - features: Dict mapping feature name to array
        - valid_mask: Boolean mask for valid rows
        - feature_names: List of feature names in order

    Example:
        >>> result = compute_hmm_features_dict(df)
        >>> print(result['features']['adx_t'][-10:])  # Last 10 ADX values
    """
    X, valid_mask = compute_hmm_features(df)

    features_dict = {}
    for i, name in enumerate(FEATURE_NAMES):
        features_dict[name] = X[:, i]

    return {
        "features": features_dict,
        "valid_mask": valid_mask,
        "feature_names": list(FEATURE_NAMES),
    }
