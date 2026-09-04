"""
Temperature scaling for HMM posterior calibration.

Fits a single temperature parameter T and applies softmax(log(p)/T)
to calibrate overconfident or underconfident HMM posteriors. Designed to
be applied AFTER predict_proba() and BEFORE EMA smoothing in the HMM
inference pipeline.

Mathematical formulation:
    p_cal_k = p_k^(1/T) / Σ_j p_j^(1/T)
    equivalently: softmax(log(p) / T)

T > 1 → softens overconfident posteriors (entropy increases)
T < 1 → sharpens underconfident posteriors (entropy decreases)
T = 1 → no change (identity)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

_CLIP_EPS = 1e-10  # Floor for log-domain operations


@dataclass(frozen=True)
class TemperatureScalingResult:
    """Immutable result of temperature scaling fit."""
    temperature: float
    nll_before: float
    nll_after: float
    n_samples: int
    converged: bool


class TemperatureScaler:
    """Single-parameter temperature scaling for posterior calibration.

    Usage::

        scaler = TemperatureScaler()
        result = scaler.fit(posteriors, labels)
        calibrated = scaler.transform(posteriors)
    """

    def __init__(self) -> None:
        self._temperature: float = 1.0
        self._fitted: bool = False
        self._metadata: Dict[str, Any] = {}

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    @property
    def runtime_validated(self) -> bool:
        return bool(self._metadata.get("runtime_validated", False))

    def set_metadata(self, **metadata: Any) -> None:
        self._metadata.update(metadata)

    def fit(
        self,
        posteriors: np.ndarray,
        labels: np.ndarray,
    ) -> TemperatureScalingResult:
        """Fit the temperature parameter by minimizing NLL.

        Parameters
        ----------
        posteriors : array of shape (N, K)
            HMM posterior probabilities over K states.
        labels : array of shape (N,)
            Integer state labels (typically argmax of ground truth).

        Returns
        -------
        TemperatureScalingResult
        """
        from scipy.optimize import minimize_scalar

        P = np.asarray(posteriors, dtype=float)
        y = np.asarray(labels, dtype=int).reshape(-1)

        if P.ndim != 2:
            raise ValueError(f"posteriors must be 2D, got shape {P.shape}")
        if P.shape[0] != len(y):
            raise ValueError(
                f"posteriors rows ({P.shape[0]}) != labels length ({len(y)})"
            )

        n, k = P.shape

        # Validate label range
        if np.any(y < 0) or np.any(y >= k):
            raise ValueError(
                f"labels must be in [0, {k}), got range [{int(np.min(y))}, {int(np.max(y))}]"
            )

        # Edge cases
        if n < 2:
            logger.warning("TemperatureScaler: too few samples (%d), using T=1.0", n)
            return TemperatureScalingResult(
                temperature=1.0, nll_before=0.0, nll_after=0.0,
                n_samples=n, converged=False,
            )

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning(
                "TemperatureScaler: single class in labels, using T=1.0"
            )
            return TemperatureScalingResult(
                temperature=1.0, nll_before=0.0, nll_after=0.0,
                n_samples=n, converged=False,
            )

        # Clip posteriors for numerical stability
        P_safe = np.clip(P, _CLIP_EPS, None)
        log_P = np.log(P_safe)

        def nll(T: float) -> float:
            """Negative log-likelihood under temperature T."""
            scaled = log_P / T
            # Log-sum-exp for numerical stability
            max_scaled = np.max(scaled, axis=1, keepdims=True)
            log_norm = max_scaled.squeeze(-1) + np.log(
                np.sum(np.exp(scaled - max_scaled), axis=1)
            )
            # NLL = -mean(scaled[i, y[i]] - log_norm[i])
            sample_nll = -(scaled[np.arange(n), y] - log_norm)
            return float(np.mean(sample_nll))

        nll_before = nll(1.0)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        converged = bool(result.success) if hasattr(result, "success") else True
        T_opt = float(result.x)

        # Sanity: if optimization failed badly, keep T=1
        if not np.isfinite(T_opt) or T_opt <= 0:
            T_opt = 1.0
            converged = False

        nll_after = nll(T_opt)

        self._temperature = T_opt
        self._fitted = True

        logger.info(
            "TemperatureScaler fit: T=%.4f, NLL %.4f → %.4f (N=%d, K=%d, converged=%s)",
            T_opt, nll_before, nll_after, n, k, converged,
        )

        return TemperatureScalingResult(
            temperature=float(T_opt),
            nll_before=float(nll_before),
            nll_after=float(nll_after),
            n_samples=n,
            converged=converged,
        )

    def transform(self, posteriors: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to posteriors.

        Parameters
        ----------
        posteriors : array of shape (N, K) or (K,)
            Raw posterior probabilities.

        Returns
        -------
        array of same shape
            Calibrated posteriors summing to 1 per row.
        """
        P = np.asarray(posteriors, dtype=float)
        single = P.ndim == 1
        if single:
            P = P.reshape(1, -1)

        T = self._temperature
        if not np.isfinite(T) or T <= 0:
            logger.warning("Invalid temperature T=%.6f in transform(), using T=1.0", T)
            T = 1.0

        P_safe = np.clip(P, _CLIP_EPS, None)
        log_P = np.log(P_safe)
        scaled = log_P / T

        # Softmax (numerically stable)
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - max_scaled)
        calibrated = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

        if single:
            calibrated = calibrated.squeeze(0)

        return calibrated

    def fit_transform(
        self,
        posteriors: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, TemperatureScalingResult]:
        """Fit and transform in one call.

        Returns
        -------
        (calibrated_posteriors, result)
        """
        result = self.fit(posteriors, labels)
        calibrated = self.transform(posteriors)
        return calibrated, result

    def save(self, path: Path) -> None:
        """Save temperature parameter and metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self._temperature,
            "fitted": self._fitted,
            "module": "temperature_scaling_v20260311",
        }
        data.update(self._metadata)
        path.write_text(json.dumps(data, indent=2))
        logger.info("TemperatureScaler saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> TemperatureScaler:
        """Load from JSON artifact.

        Returns
        -------
        TemperatureScaler
            Scaler with restored temperature parameter.

        Raises
        ------
        FileNotFoundError
            If the artifact file does not exist.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        scaler = cls()
        T_loaded = float(data["temperature"])
        if not np.isfinite(T_loaded) or T_loaded <= 0:
            logger.warning(
                "TemperatureScaler loaded invalid T=%.4f from %s — using T=1.0",
                T_loaded, path,
            )
            T_loaded = 1.0
        scaler._temperature = T_loaded
        scaler._fitted = bool(data.get("fitted", True))
        scaler._metadata = {
            k: v
            for k, v in data.items()
            if k not in {"temperature", "fitted", "module"}
        }
        logger.info(
            "TemperatureScaler loaded from %s (T=%.4f)", path, scaler._temperature
        )
        return scaler
