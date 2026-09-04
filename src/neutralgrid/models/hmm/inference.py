"""
HMM inference module - load trained model and predict regime.

This module handles ONLY inference (prediction). Training is in train.py.

Design principles:
- Load model artifact with metadata validation
- Use schema validation to prevent training/inference mismatch
- Use single source of truth for feature extraction
- Fast inference for production use
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Optional, Mapping
import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.models.artifacts import load_artifact, resolve_hmm_artifact_dir
from neutralgrid.models.hmm.schema import (
    validate_artifact_features,
    compute_state_groups,
    infer_state_mapping,
    infer_state_mapping_directional,
)
from neutralgrid.models.hmm.uncertainty import (
    compute_tail_adjustment_weights,
    compute_weighted_conditional_tail_risk,
    compute_weighted_conditional_tail_risk_enhanced,
    _safe_float,
)
from neutralgrid.data.features import compute_hmm_features
from neutralgrid.models.alerts import (
    alert_model_not_found,
    alert_model_load_failed,
    alert_inference_failed,
)
from neutralgrid.core.exceptions import ModelLoadError, InferenceError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HMMInferenceResult:
    """Result of HMM regime inference."""

    # Regime probabilities (smoothed if smooth_k > 1)
    trend_prob: float
    range_prob: float

    # Raw (unsmoothed) probabilities for diagnostics
    trend_prob_raw: float
    range_prob_raw: float

    # State information
    trend_state: int
    range_state: int
    posteriors: list  # Full posterior distribution over all states

    # Transition matrix and persistence
    transition_matrix: list   # Full transition matrix (n_states x n_states)
    persistence_prob: float   # Self-transition probability for current dominant state

    # Model metadata
    trained_at_utc: str
    artifact_version: str
    pipeline_version: str
    state_means: list  # Unscaled state means for interpretability

    # Aggregated probabilities (summing range-like / trend-like state groups)
    # These are the primary metrics for dominance gating with >2 states.
    range_prob_agg: float = 0.0
    trend_prob_agg: float = 0.0

    # Fields with defaults must come last
    # Direction-aware regime info (None if HMM_DIRECTION_AWARE=False)
    bull_trend_state: Optional[int] = None
    bear_trend_state: Optional[int] = None
    trend_direction: str = "unknown"  # "bull", "bear", or "unknown"
    smooth_k: int = 1  # Smoothing window used
    posterior_mode: str = "ema"
    dominant_volatility_tier: str = "unknown"
    volatility_tier_probs: dict[str, float] = field(default_factory=dict)
    conditional_tail_risk: dict[str, Any] = field(default_factory=dict)

    # Non-stationary tail correction diagnostics
    tail_correction_applied: bool = False
    tail_correction_weights: list = field(default_factory=list)
    conditional_tail_risk_enhanced: dict[str, Any] = field(default_factory=dict)
    dominant_volatility_tier_kurtosis_aware: str = "unknown"
    volatility_tier_probs_kurtosis_aware: dict[str, float] = field(default_factory=dict)
    calibration_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "trend_prob": float(self.trend_prob),
            "range_prob": float(self.range_prob),
            "trend_prob_raw": float(self.trend_prob_raw),
            "range_prob_raw": float(self.range_prob_raw),
            "trend_state": int(self.trend_state),
            "range_state": int(self.range_state),
            "posteriors": [float(p) for p in self.posteriors],
            "transition_matrix": [
                [float(x) for x in row] for row in self.transition_matrix
            ],
            "persistence_prob": float(self.persistence_prob),
            "trained_at_utc": self.trained_at_utc,
            "artifact_version": self.artifact_version,
            "pipeline_version": self.pipeline_version,
            "state_means": [[float(x) for x in row] for row in self.state_means],
            "smooth_k": self.smooth_k,
            "range_prob_agg": float(self.range_prob_agg),
            "trend_prob_agg": float(self.trend_prob_agg),
            "bull_trend_state": self.bull_trend_state,
            "bear_trend_state": self.bear_trend_state,
            "trend_direction": self.trend_direction,
            "posterior_mode": self.posterior_mode,
            "dominant_volatility_tier": self.dominant_volatility_tier,
            "volatility_tier_probs": {
                str(k): float(v) for k, v in self.volatility_tier_probs.items()
            },
            "conditional_tail_risk": {
                str(k): float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in self.conditional_tail_risk.items()
            },
            "tail_correction_applied": self.tail_correction_applied,
            "tail_correction_weights": [float(w) for w in self.tail_correction_weights],
            "conditional_tail_risk_enhanced": {
                str(k): float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in self.conditional_tail_risk_enhanced.items()
            },
            "dominant_volatility_tier_kurtosis_aware": self.dominant_volatility_tier_kurtosis_aware,
            "volatility_tier_probs_kurtosis_aware": {
                str(k): float(v) for k, v in self.volatility_tier_probs_kurtosis_aware.items()
            },
            "calibration_provenance": {
                str(k): v for k, v in self.calibration_provenance.items()
            },
        }


def _ema_smooth_posteriors(posteriors: np.ndarray, k: int) -> np.ndarray:
    """Apply EMA smoothing over the last k rows of a posterior matrix.

    Args:
        posteriors: Array of shape (n_samples, n_states)
        k: Window size for EMA (1 = no smoothing)

    Returns:
        Smoothed posterior for the last time-step (1D array, sums to ~1).
    """
    if k <= 1 or posteriors.shape[0] <= 1:
        return posteriors[-1]

    # Use at most the last k posteriors
    window = posteriors[-k:]

    # Allow explicit alpha override from config
    _cfg = get_config()
    alpha_override = _cfg.hmm.smooth_alpha
    if alpha_override is not None:
        alpha = float(alpha_override)
    else:
        alpha = 2.0 / (len(window) + 1)

    smoothed = window[0].copy()
    for row in window[1:]:
        smoothed = alpha * row + (1.0 - alpha) * smoothed

    # Re-normalise so probabilities sum to 1
    total = smoothed.sum()
    if total > 1e-10:
        smoothed /= total
    return smoothed


def _estimate_local_transition_matrix(
    posteriors: np.ndarray,
    *,
    n_states: int,
    window: int,
) -> np.ndarray:
    """Estimate a local transition matrix from recent decoded states."""
    if posteriors.shape[0] < 3:
        return np.eye(n_states, dtype=float)

    w = max(3, min(int(window), int(posteriors.shape[0])))
    states = np.argmax(posteriors[-w:], axis=1)

    # Add a tiny pseudocount for numerical stability.
    counts = np.full((n_states, n_states), 1e-6, dtype=float)
    for i in range(len(states) - 1):
        counts[int(states[i]), int(states[i + 1])] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return counts / row_sums


def _apply_adaptive_transitions(
    *,
    base_posterior: np.ndarray,
    posteriors: np.ndarray,
    transmat: np.ndarray,
    window: int,
    weight: float,
 ) -> tuple[np.ndarray, str, np.ndarray]:
    """Project posterior through blended global+local transitions.

    This adds non-stationary adaptation without replacing the trained HMM:
    - Global transition matrix: long-run structure from training
    - Local transition matrix: recent state dynamics over inference window
    - Weighted blend controls adaptability vs stability
    """
    if weight <= 0 or posteriors.shape[0] < 3:
        base = np.asarray(transmat, dtype=float)
        return base_posterior, "ema", base

    n_states = int(transmat.shape[0])
    local = _estimate_local_transition_matrix(
        posteriors,
        n_states=n_states,
        window=window,
    )
    w = min(1.0, max(0.0, float(weight)))
    blended = (1.0 - w) * np.asarray(transmat, dtype=float) + w * local

    projected = np.asarray(base_posterior, dtype=float) @ blended
    total = float(np.sum(projected))
    if total > 1e-10:
        projected = projected / total
    else:
        projected = np.asarray(base_posterior, dtype=float)

    return projected, "ema+adaptive_transitions", blended


def _load_runtime_temperature_scaler(artifact_dir: Path) -> tuple[Any | None, dict[str, Any]]:
    """Load a runtime-safe temperature scaler from the predictor artifact."""
    provenance: dict[str, Any] = {
        "artifact_dir": str(artifact_dir),
        "scaler_path": str(Path(artifact_dir) / "temperature_scaler.json"),
        "applied": False,
        "temperature": 1.0,
        "fitted": False,
        "runtime_validated": False,
        "status": "unavailable",
    }

    if not getattr(get_config(), "temperature_scaling_enabled", True):
        provenance["status"] = "disabled_by_config"
        return None, provenance

    ts_path = Path(artifact_dir) / "temperature_scaler.json"
    if not ts_path.exists():
        provenance["status"] = "missing"
        return None, provenance

    try:
        from neutralgrid.calibration.temperature_scaling_v20260311 import TemperatureScaler

        scaler = TemperatureScaler.load(ts_path)
    except Exception as exc:
        provenance["status"] = f"load_failed:{type(exc).__name__}"
        return None, provenance

    metadata = scaler.metadata
    provenance.update(metadata)
    provenance["temperature"] = float(scaler.temperature)
    provenance["fitted"] = bool(scaler.fitted)
    provenance["runtime_validated"] = bool(scaler.runtime_validated)

    if not scaler.runtime_validated:
        provenance["status"] = str(metadata.get("status") or "legacy_unvalidated")
        return None, provenance

    if not scaler.fitted:
        provenance["status"] = str(metadata.get("status") or "not_fitted")
        return None, provenance

    provenance["status"] = str(metadata.get("status") or "applied")
    return scaler, provenance


def _extract_state_volatility_tiers(
    regime_uncertainty: Mapping[str, Any] | None,
) -> dict[int, str]:
    """Extract state->volatility-tier mapping from saved uncertainty profile."""
    if not regime_uncertainty:
        return {}

    regimes = regime_uncertainty.get("regimes", {})
    if not isinstance(regimes, Mapping):
        return {}

    out: dict[int, str] = {}
    for key, val in regimes.items():
        if not isinstance(key, str) or not key.startswith("state_"):
            continue
        if not isinstance(val, Mapping):
            continue
        try:
            idx = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        tier = val.get("volatility_tier")
        if isinstance(tier, str) and tier in {"low", "medium", "high"}:
            out[idx] = tier
    return out


def _compute_volatility_tier_probs(
    posterior: np.ndarray,
    state_tiers: Mapping[int, str],
) -> tuple[dict[str, float], str]:
    """Aggregate posterior mass into low/medium/high volatility tiers."""
    tier_probs = {"low": 0.0, "medium": 0.0, "high": 0.0}
    if posterior.ndim != 1 or posterior.size == 0:
        return tier_probs, "unknown"

    for i, p in enumerate(posterior):
        tier = state_tiers.get(i)
        if tier in tier_probs:
            tier_probs[tier] += float(p)

    # If no mapping exists, keep unknown tier.
    if max(tier_probs.values()) <= 0:
        return tier_probs, "unknown"

    dominant = max(tier_probs, key=lambda k: tier_probs[k])
    return tier_probs, dominant


def _extract_state_volatility_tiers_kurtosis_aware(
    regime_uncertainty: Mapping[str, Any] | None,
) -> dict[int, str]:
    """Extract kurtosis-aware state->tier mapping from uncertainty profile."""
    if not regime_uncertainty:
        return {}

    tiers_ka = regime_uncertainty.get("volatility_tiers_kurtosis_aware", {})
    if not isinstance(tiers_ka, Mapping):
        return {}

    state_tiers_raw = tiers_ka.get("state_tiers", {})
    if not isinstance(state_tiers_raw, Mapping):
        return {}

    out: dict[int, str] = {}
    for key, val in state_tiers_raw.items():
        # Keys may be "state_0", "state_1", etc. or int
        if isinstance(key, str) and key.startswith("state_"):
            try:
                idx = int(key.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
        elif isinstance(key, int):
            idx = key
        else:
            continue
        if isinstance(val, str) and val in {"low", "medium", "high"}:
            out[idx] = val
    return out


def _apply_tail_correction(
    posteriors: np.ndarray,
    observation: np.ndarray,
    regime_uncertainty: Mapping[str, Any] | None,
    *,
    enabled: bool = True,
    threshold_sigma: float = 2.0,
    max_weight: float = 3.0,
) -> tuple[np.ndarray, bool, list[float]]:
    """Apply tail-adjusted correction to the last posterior row.

    Uses per-state empirical distribution parameters (stored in the artifact's
    regime_uncertainty profile) to correct Gaussian posteriors for heavy tails.

    Args:
        posteriors: Full posterior matrix (n_samples, n_states).
        observation: Current (unscaled) feature vector.
        regime_uncertainty: Stored regime uncertainty profile from artifact.
        enabled: Whether tail correction is active (config kill-switch).
        threshold_sigma: Sigma threshold for tail activation.
        max_weight: Maximum correction factor.

    Returns:
        Tuple of (corrected_last_posterior, correction_applied, weights_list).
    """
    last = posteriors[-1].copy()

    if not enabled or regime_uncertainty is None:
        return last, False, []

    regimes = regime_uncertainty.get("regimes", {})
    if not isinstance(regimes, Mapping):
        return last, False, []

    # Extract tail_adjustment_params for each state
    n_states = posteriors.shape[1]
    state_params: list[dict[str, float]] = []
    has_params = False

    for i in range(n_states):
        rk = f"state_{i}"
        rmeta = regimes.get(rk, {})
        tap = rmeta.get("tail_adjustment_params", {}) if isinstance(rmeta, Mapping) else {}
        if isinstance(tap, Mapping) and "mean" in tap and "std" in tap:
            state_params.append({
                "mean": _safe_float(tap.get("mean", 0.0)),
                "std": _safe_float(tap.get("std", 0.0)),
                "skew": _safe_float(tap.get("skew", 0.0)),
                "excess_kurtosis": _safe_float(tap.get("excess_kurtosis", 0.0)),
                "p_empirical_below_2sigma": _safe_float(tap.get("p_empirical_below_2sigma", 0.0)),
                "p_gaussian_below_2sigma": _safe_float(tap.get("p_gaussian_below_2sigma", 0.0)),
            })
            has_params = True
        else:
            state_params.append({"mean": 0.0, "std": 0.0})

    if not has_params:
        return last, False, []

    weights = compute_tail_adjustment_weights(
        observation,
        state_params,
        return_feature_idx=0,
        threshold_sigma=threshold_sigma,
        max_weight=max_weight,
    )

    # If all weights are 1.0, no correction needed
    if np.allclose(weights, 1.0):
        return last, False, weights.tolist()

    corrected = last * weights
    total = float(np.sum(corrected))
    if total > 1e-10:
        corrected = corrected / total
    else:
        corrected = last

    return corrected, True, weights.tolist()


class HMMRegimePredictor:
    """
    HMM regime predictor for production inference.

    Loads trained HMM model artifact and provides fast inference.
    Validates feature schema to prevent training/inference mismatch.
    """

    def __init__(self, artifact_dir: Optional[Path] = None):
        """
        Initialize predictor by loading artifact.

        Args:
            artifact_dir: Path to artifact directory. If None, uses default from config.

        Raises:
            FileNotFoundError: If artifact not found
            ValueError: If feature schema mismatch
        """
        if artifact_dir is None:
            artifact_dir = resolve_hmm_artifact_dir()
        self._artifact_dir = Path(artifact_dir)

        # Load artifact with validation
        artifact = load_artifact(self._artifact_dir, validate_schema=True)

        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.metadata = artifact["metadata"]
        self.state_means_unscaled = artifact.get("state_means_unscaled")

        # Validate feature schema
        if self.metadata.features is not None:
            validate_artifact_features(self.metadata.features)

        # Validate feature_schema version (v1 vs v2) to prevent silent mismatch
        from neutralgrid.models.hmm.schema import validate_artifact_schema_version
        # Build a pseudo-metadata dict for schema version validation.
        # Check artifact metadata, feature_schema.json, or fall back to "v1".
        _schema_meta = {}
        if hasattr(self.metadata, "feature_schema") and self.metadata.feature_schema:
            _schema_meta["feature_schema"] = self.metadata.feature_schema
        elif artifact.get("feature_schema") and isinstance(artifact["feature_schema"], dict):
            _schema_meta["feature_schema"] = artifact["feature_schema"].get("feature_schema", "v1")
        else:
            _schema_meta["feature_schema"] = "v1"
        validate_artifact_schema_version(_schema_meta)

        # Extract state mapping
        if self.metadata.state_mapping is not None:
            self.trend_state = self.metadata.state_mapping.get("trend", 0)
            self.range_state = self.metadata.state_mapping.get("range", 0)
        else:
            # Legacy: try to infer from state_means_unscaled
            if self.state_means_unscaled is not None:
                self.trend_state, self.range_state = infer_state_mapping(
                    self.state_means_unscaled,
                    transmat=getattr(self.model, "transmat_", None),
                )
            else:
                # Fallback
                self.trend_state = 0
                self.range_state = 1

        # Direction-aware state mapping (bull/bear trend separation)
        self.bull_trend_state = None
        self.bear_trend_state = None
        _init_cfg = get_config()
        if _init_cfg.hmm.direction_aware and self.state_means_unscaled is not None:
            dir_map = infer_state_mapping_directional(self.state_means_unscaled)
            self.bull_trend_state = dir_map["bull_trend_state"]
            self.bear_trend_state = dir_map["bear_trend_state"]

        # Compute range-like / trend-like state groups for aggregated probs
        if self.state_means_unscaled is not None:
            groups = compute_state_groups(self.state_means_unscaled)
            self.range_states = groups["range_states"]
            self.trend_states = groups["trend_states"]
        else:
            self.range_states = [self.range_state]
            self.trend_states = [self.trend_state]

        eval_metrics = getattr(self.metadata, "eval_metrics", None)
        self.regime_uncertainty = None
        if isinstance(eval_metrics, dict):
            ru = eval_metrics.get("regime_uncertainty")
            if isinstance(ru, dict):
                self.regime_uncertainty = ru
        self.state_volatility_tiers = _extract_state_volatility_tiers(self.regime_uncertainty)
        self.state_volatility_tiers_kurtosis_aware = _extract_state_volatility_tiers_kurtosis_aware(
            self.regime_uncertainty
        )

    def predict(self, df: pd.DataFrame, infer_limit: Optional[int] = None) -> HMMInferenceResult:
        """
        Predict regime for a symbol.

        Args:
            df: DataFrame with OHLCV kline data
            infer_limit: Optional limit on number of bars to use for inference

        Returns:
            HMMInferenceResult with regime probabilities and metadata

        Raises:
            ValueError: If insufficient data for inference
        """
        # Compute features using single source of truth
        X, valid = compute_hmm_features(df)

        if len(X) == 0:
            raise ValueError("No kline data for HMM inference")

        # Filter to valid rows
        X_valid = X[valid]

        # Apply inference limit if specified
        if infer_limit is not None and infer_limit > 0 and X_valid.shape[0] > infer_limit:
            X_valid = X_valid[-infer_limit:]

        if X_valid.shape[0] < 60:
            raise ValueError(
                f"Insufficient kline history for HMM inference: {X_valid.shape[0]} bars "
                f"(need at least 60)"
            )

        # Scale features
        X_scaled = self.scaler.transform(X_valid)

        # Predict posterior probabilities
        posteriors = self.model.predict_proba(X_scaled)

        # Temperature scaling (Phase 2 calibration) — applied before EMA smoothing
        # Loads pre-fitted TemperatureScaler artifact if available.
        temperature_scaler, calibration_provenance = _load_runtime_temperature_scaler(
            self._artifact_dir
        )
        if temperature_scaler is not None:
            posteriors = temperature_scaler.transform(posteriors)
            calibration_provenance["applied"] = True
            logger.debug(
                "Temperature scaling applied from %s (T=%.4f)",
                self._artifact_dir,
                temperature_scaler.temperature,
            )

        # Tail-adjusted posterior correction (non-stationary enhancement)
        _cfg = get_config()
        tail_corrected, tail_applied, tail_weights = _apply_tail_correction(
            posteriors,
            observation=X_valid[-1],  # unscaled observation for tail comparison
            regime_uncertainty=self.regime_uncertainty,
            enabled=bool(_cfg.hmm.tail_correction_enabled),
            threshold_sigma=float(_cfg.hmm.tail_correction_threshold_sigma),
            max_weight=float(_cfg.hmm.tail_correction_max_weight),
        )

        # Replace last row with corrected values for downstream smoothing
        if tail_applied:
            posteriors_adjusted = posteriors.copy()
            posteriors_adjusted[-1] = tail_corrected
        else:
            posteriors_adjusted = posteriors

        # Raw last-bar posterior (unsmoothed, after tail correction)
        last_posterior_raw = posteriors_adjusted[-1]
        trend_prob_raw = float(last_posterior_raw[self.trend_state])
        range_prob_raw = float(last_posterior_raw[self.range_state])

        # Smoothed posterior (EMA over last k bars)
        smooth_k = int(_cfg.hmm.smooth_k)
        smoothed_posterior = _ema_smooth_posteriors(posteriors_adjusted, smooth_k)

        # Transition matrix and persistence probability
        transmat = self.model.transmat_
        adaptive_enabled = bool(_cfg.hmm.adaptive_transitions)
        adaptive_window = int(_cfg.hmm.adaptive_transition_window)
        adaptive_weight = float(_cfg.hmm.adaptive_transition_weight)
        final_posterior, posterior_mode, effective_transmat = _apply_adaptive_transitions(
            base_posterior=smoothed_posterior,
            posteriors=posteriors,
            transmat=transmat,
            window=adaptive_window,
            weight=adaptive_weight if adaptive_enabled else 0.0,
        )

        n_states = int(transmat.shape[0])
        if self.trend_state < 0 or self.trend_state >= n_states:
            raise ValueError(
                f"trend_state {self.trend_state} out of bounds for "
                f"{n_states}-state HMM"
            )
        if self.range_state < 0 or self.range_state >= n_states:
            raise ValueError(
                f"range_state {self.range_state} out of bounds for "
                f"{n_states}-state HMM"
            )
        trend_prob = float(final_posterior[self.trend_state])
        range_prob = float(final_posterior[self.range_state])
        dominant_state = int(np.argmax(final_posterior))
        if dominant_state < 0 or dominant_state >= n_states:
            raise ValueError(
                f"dominant_state {dominant_state} out of bounds for "
                f"{n_states}-state HMM"
            )
        persistence_prob = float(effective_transmat[dominant_state, dominant_state])

        # Aggregated probabilities: sum posteriors across range-like / trend-like groups
        range_prob_agg = float(sum(final_posterior[s] for s in self.range_states))
        trend_prob_agg = float(sum(final_posterior[s] for s in self.trend_states))

        # Direction-aware trend classification
        if self.bull_trend_state is not None and self.bear_trend_state is not None:
            bull_p = float(final_posterior[self.bull_trend_state])
            bear_p = float(final_posterior[self.bear_trend_state])
            trend_direction = "bull" if bull_p >= bear_p else "bear"
        elif self.bull_trend_state is not None:
            trend_direction = "bull"
        elif self.bear_trend_state is not None:
            trend_direction = "bear"
        else:
            trend_direction = "unknown"

        volatility_tier_probs, dominant_volatility_tier = _compute_volatility_tier_probs(
            final_posterior, self.state_volatility_tiers
        )
        conditional_tail_risk = compute_weighted_conditional_tail_risk(
            final_posterior,
            self.regime_uncertainty,
        )

        # Enhanced metrics (Cornish-Fisher, weighted kurtosis, Gaussian divergence)
        conditional_tail_risk_enhanced = compute_weighted_conditional_tail_risk_enhanced(
            final_posterior,
            self.regime_uncertainty,
        )

        # Kurtosis-aware volatility tiers
        vol_tier_probs_ka, dom_vol_tier_ka = _compute_volatility_tier_probs(
            final_posterior, self.state_volatility_tiers_kurtosis_aware
        )

        return HMMInferenceResult(
            trend_prob=trend_prob,
            range_prob=range_prob,
            trend_prob_raw=trend_prob_raw,
            range_prob_raw=range_prob_raw,
            trend_state=self.trend_state,
            range_state=self.range_state,
            posteriors=final_posterior.astype(float).tolist(),
            transition_matrix=effective_transmat.astype(float).tolist(),
            persistence_prob=persistence_prob,
            trained_at_utc=self.metadata.trained_at_utc,
            artifact_version=self.metadata.artifact_version,
            pipeline_version=str(getattr(self.metadata, "pipeline_version", "") or ""),
            range_prob_agg=range_prob_agg,
            trend_prob_agg=trend_prob_agg,
            bull_trend_state=self.bull_trend_state,
            bear_trend_state=self.bear_trend_state,
            trend_direction=trend_direction,
            state_means=(
                self.state_means_unscaled.astype(float).tolist()
                if self.state_means_unscaled is not None
                else []
            ),
            smooth_k=smooth_k,
            posterior_mode=posterior_mode,
            dominant_volatility_tier=dominant_volatility_tier,
            volatility_tier_probs=volatility_tier_probs,
            conditional_tail_risk=conditional_tail_risk,
            tail_correction_applied=tail_applied,
            tail_correction_weights=tail_weights,
            conditional_tail_risk_enhanced=conditional_tail_risk_enhanced,
            dominant_volatility_tier_kurtosis_aware=dom_vol_tier_ka,
            volatility_tier_probs_kurtosis_aware=vol_tier_probs_ka,
            calibration_provenance=calibration_provenance,
        )


def load_predictor(artifact_dir: Optional[Path] = None) -> HMMRegimePredictor:
    """
    Load HMM regime predictor.

    Args:
        artifact_dir: Path to artifact directory. If None, uses default from config.

    Returns:
        HMMRegimePredictor instance.

    Raises:
        ModelLoadError: If the artifact is missing, corrupt, or cannot be loaded.

    Example:
        >>> predictor = load_predictor()
        >>> result = predictor.predict(df)
        >>> print(f"Range prob: {result.range_prob:.2f}")
    """
    # Resolve artifact_dir for alerting context
    resolved_dir = artifact_dir if artifact_dir is not None else resolve_hmm_artifact_dir()

    try:
        return HMMRegimePredictor(artifact_dir)
    except FileNotFoundError as e:
        # AFML: Alert on model not found (Station 6 compliance)
        alert_model_not_found(str(resolved_dir), model_type="HMM")
        raise ModelLoadError(
            artifact_path=str(resolved_dir),
            message=f"Artifact not found: {e}",
        ) from e
    except Exception as e:
        # AFML: Alert on model load failure (Station 6 compliance)
        alert_model_load_failed(str(resolved_dir), e, model_type="HMM")
        raise ModelLoadError(
            artifact_path=str(resolved_dir),
            message=f"{type(e).__name__}: {e}",
        ) from e


def predict_regime(
    df: pd.DataFrame,
    infer_limit: Optional[int] = None,
    symbol: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convenience function for regime prediction.

    Args:
        df: DataFrame with OHLCV kline data
        infer_limit: Optional limit on number of bars to use
        symbol: Optional symbol name for alerting context

    Returns:
        Dictionary with regime information.

    Raises:
        ModelLoadError: If the HMM artifact cannot be loaded.
        InferenceError: If model prediction fails at runtime.

    Example:
        >>> result = predict_regime(df, symbol="BTCUSDT")
        >>> print(f"Range prob: {result['range_prob']:.2f}")
    """
    predictor = load_predictor()  # raises ModelLoadError on failure

    try:
        result = predictor.predict(df, infer_limit)
        return result.to_dict()
    except Exception as e:
        # AFML: Alert on inference failure (Station 6 compliance)
        alert_inference_failed(symbol or "unknown", e, model_type="HMM")
        raise InferenceError(
            symbol=symbol or "unknown",
            model_type="HMM",
            message=f"{type(e).__name__}: {e}",
        ) from e
