"""
Regime uncertainty analytics for HMM training/inference.

This module computes regime-conditional risk diagnostics from historical
returns and state assignments, including:
- Empirical tail risk (VaR/CVaR)
- Parametric Gaussian tail risk (for empirical-vs-parametric comparison)
- Higher moments (skewness, excess kurtosis)
- Tail-event probabilities conditioned by regime
- Volatility tier mapping (low/medium/high)

Design goals:
- Deterministic and serializable outputs for artifact metadata
- No external data assumptions (optional exogenous signal integration only)
- Strictly additive: does not alter core HMM training/inference contracts
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, norm, skew


_DEFAULT_TAIL_LEVELS: tuple[float, float] = (0.95, 0.99)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert value to finite float, or default when not possible."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _nan_safe_mean(values: np.ndarray) -> float:
    """Nan-safe mean as float."""
    if values.size == 0:
        return 0.0
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return 0.0
    return float(np.nanmean(v))


def _nan_safe_std(values: np.ndarray) -> float:
    """Nan-safe sample std (ddof=1) as float."""
    if values.size < 2:
        return 0.0
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return 0.0
    std = float(np.nanstd(v, ddof=1))
    return std if np.isfinite(std) else 0.0


def _compute_var_cvar_empirical(returns: np.ndarray, confidence: float) -> tuple[float, float]:
    """Empirical lower-tail VaR/CVaR for returns.

    Args:
        returns: 1D return array.
        confidence: Confidence level (e.g., 0.95, 0.99).

    Returns:
        (var, cvar), where var is the lower-tail quantile and cvar is the
        conditional mean of returns below that quantile.
    """
    if returns.size == 0:
        return 0.0, 0.0
    alpha = float(1.0 - confidence)
    if alpha <= 0 or alpha >= 1:
        return 0.0, 0.0

    q = float(np.nanquantile(returns, alpha))
    tail = returns[returns <= q]
    if tail.size == 0:
        return q, q
    return q, float(np.nanmean(tail))


def _compute_var_cvar_gaussian(returns: np.ndarray, confidence: float) -> tuple[float, float]:
    """Parametric Gaussian lower-tail VaR/CVaR for returns."""
    mu = _nan_safe_mean(returns)
    sigma = _nan_safe_std(returns)
    if sigma <= 0:
        return mu, mu

    alpha = float(1.0 - confidence)
    if alpha <= 0 or alpha >= 1:
        return mu, mu

    z = float(norm.ppf(alpha))
    var = mu + sigma * z
    # For lower tail of normal: E[X | X <= VaR_alpha] = mu - sigma * phi(z)/alpha
    cvar = mu - sigma * (float(norm.pdf(z)) / alpha)
    return float(var), float(cvar)


def _compute_higher_moments(returns: np.ndarray) -> dict[str, float]:
    """Compute mean/std/skew/excess kurtosis with robust finite handling."""
    if returns.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "excess_kurtosis": 0.0,
        }

    mu = _nan_safe_mean(returns)
    sigma = _nan_safe_std(returns)

    if returns.size >= 3:
        sk = _safe_float(skew(returns, bias=False, nan_policy="omit"), default=0.0)
    else:
        sk = 0.0

    # Excess kurtosis (Fisher definition): normal distribution -> 0.0.
    if returns.size >= 4:
        ex_kurt = _safe_float(
            kurtosis(returns, fisher=True, bias=False, nan_policy="omit"),
            default=0.0,
        )
    else:
        ex_kurt = 0.0

    return {
        "mean": mu,
        "std": sigma,
        "skew": sk,
        "excess_kurtosis": ex_kurt,
    }


def _compute_tail_event_probabilities(
    returns: np.ndarray,
    *,
    global_p05: float,
    global_p01: float,
    global_loss_2sigma: float,
) -> dict[str, float]:
    """Compute regime-conditional tail event probabilities."""
    if returns.size == 0:
        return {
            "p_loss_le_global_p05": 0.0,
            "p_loss_le_global_p01": 0.0,
            "p_loss_le_global_2sigma": 0.0,
        }

    return {
        "p_loss_le_global_p05": float(np.mean(returns <= global_p05)),
        "p_loss_le_global_p01": float(np.mean(returns <= global_p01)),
        "p_loss_le_global_2sigma": float(np.mean(returns <= global_loss_2sigma)),
    }


def _classify_volatility_tiers(state_vol: Mapping[int, float]) -> dict[str, Any]:
    """Assign each state into low/medium/high volatility tiers."""
    if not state_vol:
        return {
            "state_tiers": {},
            "groups": {"low": [], "medium": [], "high": []},
            "boundaries": {"low_high_cut": 0.0, "high_low_cut": 0.0},
        }

    states = sorted(state_vol.keys())
    vols = np.asarray([_safe_float(state_vol[s], 0.0) for s in states], dtype=float)

    if len(states) == 1:
        tiers = {states[0]: "medium"}
        return {
            "state_tiers": tiers,
            "groups": {"low": [], "medium": [states[0]], "high": []},
            "boundaries": {"q33": float(vols[0]), "q66": float(vols[0])},
        }

    q33 = float(np.nanquantile(vols, 0.33))
    q66 = float(np.nanquantile(vols, 0.66))

    tiers: dict[int, str] = {}
    groups: dict[str, list[int]] = {"low": [], "medium": [], "high": []}

    for s, v in zip(states, vols):
        if v <= q33:
            t = "low"
        elif v >= q66:
            t = "high"
        else:
            t = "medium"
        tiers[s] = t
        groups[t].append(int(s))

    return {
        "state_tiers": tiers,
        "groups": groups,
        "boundaries": {"q33": q33, "q66": q66},
    }


def _compute_var_cvar_cornish_fisher(
    returns: np.ndarray,
    confidence: float,
    skewness_val: float,
    excess_kurtosis_val: float,
) -> tuple[float, float]:
    """Cornish-Fisher adjusted lower-tail VaR/CVaR.

    Uses the Cornish-Fisher expansion to adjust Gaussian quantiles
    for non-normality (skewness and excess kurtosis).  This is more
    conservative than pure Gaussian for leptokurtic distributions.

    Reference: Hull, "Risk Management and Financial Institutions".

    Args:
        returns: 1D return array (for mean/std estimation).
        confidence: Confidence level (e.g., 0.95, 0.99).
        skewness_val: Sample skewness of the return distribution.
        excess_kurtosis_val: Sample excess kurtosis (Fisher definition).

    Returns:
        (var, cvar) tuple.  Falls back to Gaussian when CF is non-finite.
    """
    mu = _nan_safe_mean(returns)
    sigma = _nan_safe_std(returns)
    if sigma <= 0:
        return mu, mu

    alpha = float(1.0 - confidence)
    if alpha <= 0 or alpha >= 1:
        return mu, mu

    z = float(norm.ppf(alpha))
    S = float(skewness_val)
    K = float(excess_kurtosis_val)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z ** 2 - 1.0) * S / 6.0
        + (z ** 3 - 3.0 * z) * K / 24.0
        - (2.0 * z ** 3 - 5.0 * z) * S ** 2 / 36.0
    )

    # Numerical stability: clamp to [-6, 6]
    z_cf = max(-6.0, min(6.0, z_cf))
    if not np.isfinite(z_cf):
        # Fall back to Gaussian
        return _compute_var_cvar_gaussian(returns, confidence)

    var_cf = mu + sigma * z_cf

    # CVaR approximation: use empirical tail below CF-VaR
    tail = returns[returns <= var_cf]
    if tail.size > 0:
        cvar_cf = float(np.nanmean(tail))
    else:
        cvar_cf = var_cf

    return float(var_cf), float(cvar_cf)


def _compute_gaussian_divergence(returns: np.ndarray) -> dict[str, float]:
    """Quantify empirical-vs-Gaussian divergence for a return series.

    Uses the Jarque-Bera test statistic (combines skewness + kurtosis)
    and a tail weight ratio (empirical vs Gaussian mass beyond 2-sigma).

    Returns:
        Dict with jarque_bera_stat, jarque_bera_pvalue, tail_weight_ratio.
    """
    if returns.size < 4:
        return {
            "jarque_bera_stat": 0.0,
            "jarque_bera_pvalue": 1.0,
            "tail_weight_ratio": 1.0,
        }

    finite = returns[np.isfinite(returns)]
    if finite.size < 4:
        return {
            "jarque_bera_stat": 0.0,
            "jarque_bera_pvalue": 1.0,
            "tail_weight_ratio": 1.0,
        }

    jb_stat, jb_pvalue = jarque_bera(finite)
    jb_stat = _safe_float(jb_stat, 0.0)
    jb_pvalue = _safe_float(jb_pvalue, 1.0)

    # Tail weight ratio: empirical mass beyond 2-sigma vs Gaussian expectation
    mu = _nan_safe_mean(finite)
    sigma = _nan_safe_std(finite)
    if sigma > 0:
        empirical_tail = float(np.mean(np.abs(finite - mu) > 2.0 * sigma))
        gaussian_tail = 2.0 * float(norm.sf(2.0))  # ~0.0456
        tail_weight_ratio = empirical_tail / gaussian_tail if gaussian_tail > 0 else 1.0
    else:
        tail_weight_ratio = 1.0

    return {
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "tail_weight_ratio": _safe_float(tail_weight_ratio, 1.0),
    }


def _compute_kurtosis_adjusted_vol(std: float, excess_kurtosis_val: float) -> float:
    """Compute kurtosis-adjusted volatility.

    States with higher excess kurtosis have fatter tails and thus higher
    effective risk, even if their standard deviation is similar.

    Formula: vol_adj = std * sqrt(1 + max(excess_kurtosis, 0) / 4)

    Derivation: from the variance of the sample variance estimator for a
    distribution with excess kurtosis kappa.
    """
    ek = max(float(excess_kurtosis_val), 0.0)
    adjustment = np.sqrt(1.0 + ek / 4.0)
    return float(std) * float(adjustment)


def _classify_volatility_tiers_kurtosis_aware(
    state_vol: Mapping[int, float],
    state_kurtosis: Mapping[int, float],
) -> dict[str, Any]:
    """Kurtosis-aware volatility tier classification.

    Uses kurtosis-adjusted volatility (via ``_compute_kurtosis_adjusted_vol``)
    instead of raw std for tier boundary computation.
    """
    if not state_vol:
        return {
            "state_tiers": {},
            "groups": {"low": [], "medium": [], "high": []},
            "boundaries": {"q33": 0.0, "q66": 0.0},
        }

    states = sorted(state_vol.keys())
    adj_vols = np.asarray(
        [
            _compute_kurtosis_adjusted_vol(
                _safe_float(state_vol[s], 0.0),
                _safe_float(state_kurtosis.get(s, 0.0), 0.0),
            )
            for s in states
        ],
        dtype=float,
    )

    if len(states) == 1:
        return {
            "state_tiers": {states[0]: "medium"},
            "groups": {"low": [], "medium": [states[0]], "high": []},
            "boundaries": {"q33": float(adj_vols[0]), "q66": float(adj_vols[0])},
        }

    q33 = float(np.nanquantile(adj_vols, 0.33))
    q66 = float(np.nanquantile(adj_vols, 0.66))

    tiers_out: dict[int, str] = {}
    groups_out: dict[str, list[int]] = {"low": [], "medium": [], "high": []}

    for s, v in zip(states, adj_vols):
        if v <= q33:
            t = "low"
        elif v >= q66:
            t = "high"
        else:
            t = "medium"
        tiers_out[s] = t
        groups_out[t].append(int(s))

    return {
        "state_tiers": tiers_out,
        "groups": groups_out,
        "boundaries": {"q33": q33, "q66": q66},
    }


def compute_tail_adjustment_weights(
    observation: np.ndarray,
    state_params: list[dict[str, float]],
    *,
    return_feature_idx: int = 0,
    threshold_sigma: float = 2.0,
    max_weight: float = 3.0,
) -> np.ndarray:
    """Compute multiplicative correction factors for Gaussian posteriors.

    For each state, if the current return observation falls in the tail
    (beyond ``threshold_sigma``), apply a correction based on the state's
    empirical tail probability vs Gaussian tail probability.  This upweights
    states whose empirical distributions better explain extreme returns.

    Args:
        observation: Current feature vector (1D, n_features).
        state_params: List of per-state dicts with keys:
            mean, std, excess_kurtosis,
            p_empirical_below_2sigma, p_gaussian_below_2sigma.
        return_feature_idx: Index of the return feature (r_t) in observation.
        threshold_sigma: Sigma threshold for tail activation.
        max_weight: Maximum correction factor (clamped symmetrically).

    Returns:
        Array of multiplicative weights (n_states,), default 1.0.
    """
    n_states = len(state_params)
    weights = np.ones(n_states, dtype=float)

    if observation.ndim == 0 or observation.size == 0:
        return weights

    r = float(observation[return_feature_idx]) if return_feature_idx < observation.size else 0.0
    min_weight = 1.0 / max(max_weight, 1.0)

    for i, params in enumerate(state_params):
        mu = _safe_float(params.get("mean", 0.0))
        sigma = _safe_float(params.get("std", 0.0))
        if sigma <= 0:
            continue

        z = (r - mu) / sigma
        if abs(z) <= threshold_sigma:
            continue

        # Gaussian tail probability at this z
        p_gauss = float(norm.sf(abs(z)))
        if p_gauss <= 0:
            continue

        # Empirical tail probability (stored from training)
        p_emp = _safe_float(params.get("p_empirical_below_2sigma", p_gauss))
        if p_emp <= 0:
            p_emp = p_gauss

        # Correction: ratio of empirical-to-Gaussian tail mass
        ratio = p_emp / p_gauss
        ratio = max(min_weight, min(max_weight, ratio))
        weights[i] = ratio

    return weights


def _generate_synthetic_exogenous_signals(
    returns: np.ndarray,
    vol: np.ndarray,
) -> pd.DataFrame:
    """Generate synthetic exogenous risk signals from market data.

    When actual news/event/macro/political signals are unavailable,
    derive data-based proxy signals:
    - volatility_shock: |vol_t / rolling_mean(vol) - 1|
    - return_shock: rolling excess kurtosis over 12-bar window
    - momentum_divergence: |short_ema - long_ema| / vol

    Args:
        returns: 1D array of log returns.
        vol: 1D array of rolling volatility (same length as returns).

    Returns:
        DataFrame with synthetic signal columns.
    """
    n = len(returns)
    out = pd.DataFrame(index=np.arange(n))

    # Volatility shock: deviation from rolling mean
    vol_s = pd.Series(vol, dtype=float)
    vol_mean = vol_s.rolling(100, min_periods=20).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["volatility_shock"] = np.abs(vol_s / vol_mean - 1.0)

    # Return shock: rolling excess kurtosis over 12-bar window
    ret_s = pd.Series(returns, dtype=float)
    out["return_shock"] = ret_s.rolling(12, min_periods=4).apply(
        lambda x: _safe_float(kurtosis(x, fisher=True, bias=False, nan_policy="omit"), 0.0),
        raw=True,
    )

    # Momentum divergence: |short_ema - long_ema| / vol
    ema_short = ret_s.ewm(span=5, min_periods=3).mean()
    ema_long = ret_s.ewm(span=20, min_periods=10).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["momentum_divergence"] = np.abs(ema_short - ema_long) / vol_s

    # Replace inf/nan edges
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


def _clean_exogenous_signals(
    exogenous_signals: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Sanitize optional exogenous uncertainty signals."""
    if exogenous_signals is None or exogenous_signals.empty:
        return None, []

    df = exogenous_signals.copy()
    valid_cols: list[str] = []
    for col in df.columns:
        s = cast(pd.Series, pd.to_numeric(df[col], errors="coerce"))
        if s.notna().any():
            df[col] = s
            valid_cols.append(str(col))

    if not valid_cols:
        return None, []

    df = cast(pd.DataFrame, df[valid_cols].replace([np.inf, -np.inf], np.nan))
    return df, valid_cols


def _compute_exogenous_framework(
    state_assignments: np.ndarray,
    exogenous_signals: pd.DataFrame | None,
    *,
    feature_matrix_unscaled: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute optional exogenous uncertainty summary by regime.

    When actual exogenous signals are unavailable (all NaN), generates
    synthetic data-derived proxy signals from the feature matrix as fallback.
    """
    clean_df, cols = _clean_exogenous_signals(exogenous_signals)
    if clean_df is None and feature_matrix_unscaled is not None and feature_names is not None:
        # Generate synthetic fallback signals from market data
        feat_list = list(feature_names)
        if "r_t" in feat_list and "vol_t" in feat_list:
            idx_r = feat_list.index("r_t")
            idx_v = feat_list.index("vol_t")
            returns = feature_matrix_unscaled[:, idx_r]
            vol = feature_matrix_unscaled[:, idx_v]
            synthetic = _generate_synthetic_exogenous_signals(returns, vol)
            clean_df, cols = _clean_exogenous_signals(synthetic)
            if clean_df is not None:
                # Tag as synthetic for transparency
                framework_desc = (
                    "Synthetic exogenous signals derived from market data "
                    "(volatility_shock, return_shock, momentum_divergence). "
                    "Robust z-score aggregation (median/MAD)."
                )
            else:
                return {
                    "enabled": False,
                    "features_used": [],
                    "framework": (
                        "Synthetic exogenous signal generation attempted but "
                        "insufficient data available."
                    ),
                    "regime_scores": {},
                }
        else:
            return {
                "enabled": False,
                "features_used": [],
                "framework": (
                    "Optional exogenous risk channels are supported but no usable "
                    "signals were provided at training time."
                ),
                "regime_scores": {},
            }
    elif clean_df is None:
        return {
            "enabled": False,
            "features_used": [],
            "framework": (
                "Optional exogenous risk channels are supported but no usable "
                "signals were provided at training time."
            ),
            "regime_scores": {},
        }
    else:
        framework_desc = (
            "Exogenous uncertainty score uses robust absolute z-scores across "
            "available signals (news/event/macro/political style channels)."
        )

    # Robust z-score using median/MAD to avoid outlier domination.
    zdf = cast(pd.DataFrame, clean_df.copy())
    for col in cols:
        s = cast(pd.Series, zdf[col])
        med = float(np.nanmedian(s))
        mad = float(np.nanmedian(np.abs(s - med)))
        scale = 1.4826 * mad if mad > 0 else float(np.nanstd(s, ddof=1) or 1.0)
        if scale <= 0:
            scale = 1.0
        zdf[col] = (s - med) / scale

    shock_series = cast(pd.Series, zdf.abs().mean(axis=1, skipna=True))
    shock = shock_series.to_numpy(dtype=float)
    if shock.shape[0] != state_assignments.shape[0]:
        # Alignment guard: do not emit misleading exogenous metrics.
        return {
            "enabled": False,
            "features_used": cols,
            "framework": "Exogenous signals found but length mismatch prevented alignment.",
            "regime_scores": {},
        }

    regime_scores: dict[str, Any] = {}
    for s in sorted(int(x) for x in np.unique(state_assignments)):
        mask = state_assignments == s
        vals = shock[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            regime_scores[f"state_{s}"] = {"mean_abs_z": 0.0, "p95_abs_z": 0.0}
        else:
            regime_scores[f"state_{s}"] = {
                "mean_abs_z": float(np.nanmean(vals)),
                "p95_abs_z": float(np.nanquantile(vals, 0.95)),
            }

    return {
        "enabled": True,
        "features_used": cols,
        "framework": framework_desc,
        "regime_scores": regime_scores,
    }


def compute_regime_uncertainty_profile(
    *,
    feature_matrix_unscaled: np.ndarray,
    state_assignments: np.ndarray,
    state_means_unscaled: np.ndarray,
    transmat: np.ndarray | None,
    feature_names: Sequence[str],
    exogenous_signals: pd.DataFrame | None = None,
    tail_levels: Sequence[float] = _DEFAULT_TAIL_LEVELS,
    min_regime_samples: int = 25,
) -> dict[str, Any]:
    """Compute a serializable uncertainty profile for a trained HMM.

    Args:
        feature_matrix_unscaled: Unscaled training feature matrix (N, F).
        state_assignments: Most-likely state per row (N,).
        state_means_unscaled: Unscaled state means (K, F).
        transmat: Transition matrix (K, K), optional.
        feature_names: Feature names aligned with matrix columns.
        exogenous_signals: Optional per-row exogenous signals aligned to N.
        tail_levels: Confidence levels for VaR/CVaR calculations.
        min_regime_samples: Minimum rows before full regime metrics are trusted.
    """
    X = np.asarray(feature_matrix_unscaled, dtype=float)
    states = np.asarray(state_assignments, dtype=int)
    if X.ndim != 2 or states.ndim != 1 or X.shape[0] != states.shape[0]:
        raise ValueError("feature_matrix_unscaled/state_assignments shape mismatch")

    feats = list(feature_names)
    if "r_t" not in feats:
        raise ValueError("feature_names must include 'r_t' for return-based risk metrics")

    idx_r = feats.index("r_t")
    returns_all = X[:, idx_r]
    returns_all = returns_all[np.isfinite(returns_all)]
    if returns_all.size == 0:
        returns_all = np.array([0.0], dtype=float)

    global_mu = _nan_safe_mean(returns_all)
    global_sigma = _nan_safe_std(returns_all)
    global_p05 = float(np.nanquantile(returns_all, 0.05))
    global_p01 = float(np.nanquantile(returns_all, 0.01))
    global_loss_2sigma = float(global_mu - 2.0 * global_sigma)

    # Per-state realized volatility proxy from returns.
    unique_states = sorted(int(s) for s in np.unique(states))
    state_vol: dict[int, float] = {}
    for s in unique_states:
        ret_s = X[states == s, idx_r]
        ret_s = ret_s[np.isfinite(ret_s)]
        state_vol[s] = _nan_safe_std(ret_s)

    tiers = _classify_volatility_tiers(state_vol)
    state_tiers: dict[int, str] = tiers["state_tiers"]

    levels = sorted(float(l) for l in tail_levels if 0.5 < float(l) < 1.0)
    if not levels:
        levels = list(_DEFAULT_TAIL_LEVELS)

    regimes: dict[str, Any] = {}
    n_total = float(X.shape[0])
    for s in unique_states:
        mask = states == s
        ret = X[mask, idx_r]
        ret = ret[np.isfinite(ret)]
        n_s = int(ret.size)
        weight = float(n_s / n_total) if n_total > 0 else 0.0

        moments = _compute_higher_moments(ret)
        empirical: dict[str, float] = {}
        gaussian: dict[str, float] = {}
        for level in levels:
            e_var, e_cvar = _compute_var_cvar_empirical(ret, level)
            g_var, g_cvar = _compute_var_cvar_gaussian(ret, level)
            key = str(int(level * 100))
            empirical[f"var_{key}"] = e_var
            empirical[f"cvar_{key}"] = e_cvar
            gaussian[f"var_{key}"] = g_var
            gaussian[f"cvar_{key}"] = g_cvar

        tail_probs = _compute_tail_event_probabilities(
            ret,
            global_p05=global_p05,
            global_p01=global_p01,
            global_loss_2sigma=global_loss_2sigma,
        )

        # Cornish-Fisher adjusted VaR/CVaR
        cornish_fisher: dict[str, float] = {}
        for level in levels:
            cf_var, cf_cvar = _compute_var_cvar_cornish_fisher(
                ret, level, moments["skew"], moments["excess_kurtosis"],
            )
            key = str(int(level * 100))
            cornish_fisher[f"var_{key}"] = cf_var
            cornish_fisher[f"cvar_{key}"] = cf_cvar

        # Gaussian divergence
        gauss_div = _compute_gaussian_divergence(ret)

        # Kurtosis-adjusted volatility
        kurt_adj_vol = _compute_kurtosis_adjusted_vol(
            moments["std"], moments["excess_kurtosis"],
        )

        # Empirical tail probability below 2-sigma (for inference-time correction)
        mu_s = moments["mean"]
        sigma_s = moments["std"]
        if sigma_s > 0 and ret.size > 0:
            p_emp_2sigma = float(np.mean(ret <= (mu_s - 2.0 * sigma_s)))
        else:
            p_emp_2sigma = 0.0
        p_gauss_2sigma = float(norm.cdf(-2.0))

        regimes[f"state_{s}"] = {
            "sample_count": n_s,
            "sample_weight": weight,
            "sufficient_samples": bool(n_s >= int(min_regime_samples)),
            "volatility_tier": state_tiers.get(s, "medium"),
            "moments": moments,
            "tail_risk": {
                "empirical": empirical,
                "gaussian": gaussian,
                "cornish_fisher": cornish_fisher,
                "tail_event_probabilities": tail_probs,
            },
            "gaussian_divergence": gauss_div,
            "kurtosis_adjusted_vol": kurt_adj_vol,
            "tail_adjustment_params": {
                "mean": mu_s,
                "std": sigma_s,
                "skew": moments["skew"],
                "excess_kurtosis": moments["excess_kurtosis"],
                "p_empirical_below_2sigma": p_emp_2sigma,
                "p_gaussian_below_2sigma": p_gauss_2sigma,
            },
            # State mean vector retained for interpretability of regime identity.
            "state_mean_unscaled": (
                [float(x) for x in state_means_unscaled[s].tolist()]
                if 0 <= s < state_means_unscaled.shape[0]
                else []
            ),
        }

    # Kurtosis-aware volatility tier classification
    state_kurtosis: dict[int, float] = {}
    for s in unique_states:
        rm = regimes.get(f"state_{s}", {})
        mom = rm.get("moments", {})
        state_kurtosis[s] = _safe_float(mom.get("excess_kurtosis", 0.0))

    tiers_kurtosis = _classify_volatility_tiers_kurtosis_aware(state_vol, state_kurtosis)

    exogenous = _compute_exogenous_framework(
        states,
        exogenous_signals,
        feature_matrix_unscaled=X,
        feature_names=feature_names,
    )

    out = {
        "method": "empirical_vs_gaussian_tail_risk",
        "n_samples": int(X.shape[0]),
        "feature": "r_t",
        "tail_levels": [float(x) for x in levels],
        "global_reference": {
            "mean": global_mu,
            "std": global_sigma,
            "p05": global_p05,
            "p01": global_p01,
            "loss_2sigma": global_loss_2sigma,
        },
        "volatility_tiers": {
            "state_tiers": {f"state_{k}": v for k, v in state_tiers.items()},
            "groups": tiers["groups"],
            "boundaries": tiers["boundaries"],
        },
        "regimes": regimes,
        "volatility_tiers_kurtosis_aware": {
            "state_tiers": {f"state_{k}": v for k, v in tiers_kurtosis["state_tiers"].items()},
            "groups": tiers_kurtosis["groups"],
            "boundaries": tiers_kurtosis["boundaries"],
        },
        "exogenous_uncertainty": exogenous,
    }

    if transmat is not None:
        out["transition_matrix"] = np.asarray(transmat, dtype=float).tolist()

    return out


def compute_weighted_conditional_tail_risk(
    posterior: np.ndarray,
    regime_uncertainty: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compute posterior-weighted tail risk from stored regime uncertainty profile."""
    p = np.asarray(posterior, dtype=float)
    if p.ndim != 1 or p.size == 0:
        return {}
    total = float(np.nansum(p))
    if total <= 0:
        return {}
    p = p / total

    if not regime_uncertainty:
        return {}

    regimes = regime_uncertainty.get("regimes", {})
    if not isinstance(regimes, Mapping):
        return {}

    out: dict[str, float] = {}
    probs: dict[str, float] = {"p_loss_le_global_p05": 0.0, "p_loss_le_global_p01": 0.0}

    # Weighted empirical CVaR/Var at supported levels.
    for state_idx in range(p.size):
        rk = f"state_{state_idx}"
        rmeta = regimes.get(rk)
        if not isinstance(rmeta, Mapping):
            continue

        tail = rmeta.get("tail_risk", {})
        emp = tail.get("empirical", {}) if isinstance(tail, Mapping) else {}
        tail_probs = (
            tail.get("tail_event_probabilities", {})
            if isinstance(tail, Mapping)
            else {}
        )
        w = float(p[state_idx])

        for key in ("var_95", "cvar_95", "var_99", "cvar_99"):
            if key in emp:
                out[key] = out.get(key, 0.0) + w * _safe_float(emp.get(key))

        for key in probs:
            if key in tail_probs:
                probs[key] += w * _safe_float(tail_probs.get(key))

    out.update(probs)
    return {k: float(v) for k, v in out.items()}


def compute_weighted_conditional_tail_risk_enhanced(
    posterior: np.ndarray,
    regime_uncertainty: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Enhanced posterior-weighted tail risk using Cornish-Fisher VaR.

    Extends the base function with:
    - Cornish-Fisher adjusted VaR/CVaR (accounting for skew/kurtosis)
    - Posterior-weighted excess kurtosis and skewness
    - Gaussian divergence score (how non-normal is the current regime mix)
    """
    p = np.asarray(posterior, dtype=float)
    if p.ndim != 1 or p.size == 0:
        return {}
    total = float(np.nansum(p))
    if total <= 0:
        return {}
    p = p / total

    if not regime_uncertainty:
        return {}

    regimes = regime_uncertainty.get("regimes", {})
    if not isinstance(regimes, Mapping):
        return {}

    cf_metrics: dict[str, float] = {}
    weighted_kurtosis = 0.0
    weighted_skewness = 0.0
    weighted_gauss_div = 0.0

    for state_idx in range(p.size):
        rk = f"state_{state_idx}"
        rmeta = regimes.get(rk)
        if not isinstance(rmeta, Mapping):
            continue

        w = float(p[state_idx])

        # Cornish-Fisher VaR/CVaR
        tail = rmeta.get("tail_risk", {})
        cf = tail.get("cornish_fisher", {}) if isinstance(tail, Mapping) else {}
        for key in ("var_95", "cvar_95", "var_99", "cvar_99"):
            cf_key = f"cf_{key}"
            if key in cf:
                cf_metrics[cf_key] = cf_metrics.get(cf_key, 0.0) + w * _safe_float(cf.get(key))

        # Weighted moments
        moments = rmeta.get("moments", {})
        if isinstance(moments, Mapping):
            weighted_kurtosis += w * _safe_float(moments.get("excess_kurtosis", 0.0))
            weighted_skewness += w * _safe_float(moments.get("skew", 0.0))

        # Gaussian divergence
        gauss_div = rmeta.get("gaussian_divergence", {})
        if isinstance(gauss_div, Mapping):
            weighted_gauss_div += w * _safe_float(gauss_div.get("tail_weight_ratio", 1.0))

    cf_metrics["weighted_excess_kurtosis"] = weighted_kurtosis
    cf_metrics["weighted_skewness"] = weighted_skewness
    cf_metrics["weighted_gaussian_divergence"] = weighted_gauss_div

    return {k: float(v) for k, v in cf_metrics.items()}
