"""
Beta calibration for binary meta-labeler probabilities.

Handles asymmetric miscalibration from GBM outputs using a 3-parameter
logistic model on the beta-transformed probability space.

Model:
    logit(P_cal) = c + a·log(s) − b·log(1−s)

where s is the raw probability from the GBM. This is fit as a logistic
regression on features [log(s), −log(1−s)] with an intercept c.

When to use: when ReliabilityDiagnostic reports max_deviation > 0.05,
indicating asymmetric miscalibration that sigmoid (Platt) scaling cannot fix.

Compatible with ``_PostProbabilityCalibrator._apply()`` via the
``predict_proba(X)`` interface (returns (N, 2) array).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

_CLIP_EPS = 1e-6


@dataclass(frozen=True)
class BetaCalibrationResult:
    """Immutable result of beta calibration fit."""
    a: float
    b: float
    c: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float
    n_samples: int
    converged: bool


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Mirrors the formula in ``meta_labeler.py`` for consistency.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(yt) == 0:
        return 0.0
    w = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None
    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    idx = np.digitize(yp, bins, right=True) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    ece = 0.0
    total_weight = float(np.sum(w)) if w is not None else float(len(yt))
    for b in range(len(bins) - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        if w is not None:
            bin_weight = float(np.sum(w[mask]))
            frac = bin_weight / total_weight
            conf = float(np.average(yp[mask], weights=w[mask]))
            acc = float(np.average(yt[mask], weights=w[mask]))
        else:
            frac = float(np.sum(mask)) / total_weight
            conf = float(np.mean(yp[mask]))
            acc = float(np.mean(yt[mask]))
        ece += abs(acc - conf) * frac
    return float(ece)


def _brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute Brier score = mean((y - p)^2)."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(yt) == 0:
        return 0.0
    sq = (yt - yp) ** 2
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        return float(np.average(sq, weights=w))
    return float(np.mean(sq))


class BetaCalibrator:
    """3-parameter beta calibration for binary probabilities.

    Usage::

        cal = BetaCalibrator()
        result = cal.fit(y_true, y_prob)
        calibrated = cal.predict(y_prob_test)
    """

    def __init__(self) -> None:
        self._lr: Any = None  # Fitted LogisticRegression
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> BetaCalibrationResult:
        """Fit the beta calibration model.

        Parameters
        ----------
        y_true : array of shape (N,)
            Binary ground-truth labels (0 or 1).
        y_prob : array of shape (N,)
            Predicted probabilities from the base model.
        sample_weight : array of shape (N,) or None
            Per-sample weights for fitting and metric computation.

        Returns
        -------
        BetaCalibrationResult
        """
        from sklearn.linear_model import LogisticRegression

        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)

        if len(yt) != len(yp):
            raise ValueError(
                f"y_true length ({len(yt)}) != y_prob length ({len(yp)})"
            )

        sw = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None

        # Pre-calibration metrics
        ece_before = _expected_calibration_error(yt, yp, sample_weight=sw)
        brier_before = _brier_score(yt, yp, sample_weight=sw)

        # Clip for log stability
        s = np.clip(yp, _CLIP_EPS, 1.0 - _CLIP_EPS)

        # Build beta calibration features: [log(s), -log(1-s)]
        X_cal = np.column_stack([np.log(s), -np.log(1.0 - s)])

        # Fit logistic regression (intercept = c, coefs = [a, b])
        lr = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )

        converged = True
        try:
            lr.fit(X_cal, yt.astype(int), sample_weight=sw)
        except Exception as e:
            logger.warning("BetaCalibrator fit failed: %s — using identity", e)
            converged = False
            # Fallback: create a dummy model
            lr = None

        self._lr = lr
        self._fitted = lr is not None

        # Extract parameters
        if lr is not None:
            coef = lr.coef_.flatten()
            a = float(coef[0])
            b = float(coef[1])
            c = float(np.asarray(lr.intercept_).flat[0])
        else:
            a, b, c = 1.0, 1.0, 0.0

        # Post-calibration metrics
        yp_cal = self.predict(yp)
        ece_after = _expected_calibration_error(yt, yp_cal, sample_weight=sw)
        brier_after = _brier_score(yt, yp_cal, sample_weight=sw)

        logger.info(
            "BetaCalibrator fit: a=%.4f, b=%.4f, c=%.4f — "
            "ECE %.4f→%.4f, Brier %.4f→%.4f (N=%d)",
            a, b, c, ece_before, ece_after, brier_before, brier_after, len(yt),
        )

        return BetaCalibrationResult(
            a=float(a),
            b=float(b),
            c=float(c),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            brier_before=float(brier_before),
            brier_after=float(brier_after),
            n_samples=len(yt),
            converged=converged,
        )

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform raw probabilities through beta calibration.

        Parameters
        ----------
        y_prob : array of shape (N,)
            Raw probabilities.

        Returns
        -------
        array of shape (N,)
            Calibrated probabilities clipped to [0, 1].
        """
        yp = np.asarray(y_prob, dtype=float).reshape(-1)

        if self._lr is None:
            return np.clip(yp, 0.0, 1.0)

        s = np.clip(yp, _CLIP_EPS, 1.0 - _CLIP_EPS)
        X_cal = np.column_stack([np.log(s), -np.log(1.0 - s)])
        calibrated = self._lr.predict_proba(X_cal)[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """sklearn-compatible interface for _PostProbabilityCalibrator.

        Parameters
        ----------
        X : array of shape (N, 1)
            Raw probabilities as column vector.

        Returns
        -------
        array of shape (N, 2)
            [1 - p_cal, p_cal] for each sample.
        """
        x = np.asarray(X, dtype=float)
        if x.ndim == 2:
            x = x[:, 0]  # Extract from column
        calibrated = self.predict(x)
        return np.column_stack([1.0 - calibrated, calibrated])
