"""
Split conformal prediction for deployment risk control.

Replaces fixed Stage B thresholds with data-driven quantiles providing
finite-sample coverage guarantees:

    P(failure) <= alpha + 1/(n_cal + 1)

This module provides:
  - ConformalRiskController: fits nonconformity quantiles from holdout data
  - ConformalQuantile: frozen result with coverage guarantee
  - should_deploy(): checks if a candidate exceeds the conformal threshold

Integration point: replaces fixed ``min_range_prob`` (or ``min_tos``,
``meta_prob``) in ``TwoStageSelector.approve()`` with conformal q-hat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformalConfig:
    """Configuration for conformal risk control."""
    target_alpha: float = 0.20  # Target miscoverage rate
    min_calibration_samples: int = 20  # Minimum holdout samples
    use_time_decay_weights: bool = True
    time_decay_halflife_days: float = 30.0
    artifact_path: str = "data/cache/conformal_quantile.json"
    max_artifact_age_days: float = 30.0  # Warn if artifact is older than this


@dataclass(frozen=True)
class ConformalQuantile:
    """Immutable result of conformal quantile computation."""
    quantile_value: float  # The q-hat threshold
    alpha: float  # Target miscoverage
    n_calibration: int
    coverage_guarantee: float  # = 1 - alpha - 1/(n_cal + 1)
    computed_at_utc: str
    gate_type: str  # "meta_prob", "range_prob", "tos"


class ConformalRiskController:
    """Split conformal prediction for deployment threshold calibration.

    Usage::

        ctrl = ConformalRiskController()
        quantile = ctrl.fit(holdout_labels, holdout_probs, gate_type="meta_prob")
        deploy, details = ctrl.should_deploy(candidate_prob, quantile)
    """

    def __init__(self, config: Optional[ConformalConfig] = None) -> None:
        self._cfg = config or ConformalConfig()

    def compute_nonconformity_scores(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> np.ndarray:
        """Compute nonconformity scores from holdout data.

        For binary classification with positive = "should deploy":
            s_i = 1 - p_hat(y_i | x_i)

        For a successful candidate (y=1): score = 1 - p (low score = well predicted)
        For a failed candidate (y=0): score = p (high score = surprisng failure)

        Parameters
        ----------
        y_true : array of shape (N,)
            Binary labels (1 = success, 0 = failure).
        y_prob : array of shape (N,)
            Predicted probabilities for success (positive class).

        Returns
        -------
        array of shape (N,)
            Nonconformity scores.
        """
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)

        if len(yt) != len(yp):
            raise ValueError(
                f"y_true length ({len(yt)}) != y_prob length ({len(yp)})"
            )

        # s_i = 1 - p for y=1, s_i = p for y=0
        scores = np.where(yt > 0.5, 1.0 - yp, yp)
        return scores

    def fit(
        self,
        holdout_labels: np.ndarray,
        holdout_probs: np.ndarray,
        gate_type: str = "meta_prob",
    ) -> ConformalQuantile:
        """Fit conformal quantile from holdout data.

        Parameters
        ----------
        holdout_labels : array of shape (N,)
            Binary labels from holdout set.
        holdout_probs : array of shape (N,)
            Predicted probabilities from holdout set.
        gate_type : str
            Which gate this quantile replaces.

        Returns
        -------
        ConformalQuantile

        Raises
        ------
        ValueError
            If holdout has fewer than min_calibration_samples.
        """
        yt = np.asarray(holdout_labels, dtype=float).reshape(-1)
        yp = np.asarray(holdout_probs, dtype=float).reshape(-1)

        if len(yt) != len(yp):
            raise ValueError(
                f"holdout_labels length ({len(yt)}) != holdout_probs length ({len(yp)})"
            )

        # Remove NaN/Inf
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]

        n = len(yt)
        if n < self._cfg.min_calibration_samples:
            raise ValueError(
                f"Conformal calibration requires at least "
                f"{self._cfg.min_calibration_samples} samples, got {n}"
            )

        # Compute nonconformity scores
        scores = self.compute_nonconformity_scores(yt, yp)

        # Compute quantile: q-hat = quantile at level ceil((1-alpha)(n+1))/n
        alpha = self._cfg.target_alpha
        quantile_level = np.ceil((1.0 - alpha) * (n + 1)) / n

        coverage = 1.0 - alpha - 1.0 / (n + 1)

        # Guard: if n too small for alpha, coverage guarantee is non-positive
        if coverage <= 0.0:
            logger.warning(
                "Conformal coverage guarantee non-positive (%.4f) for n=%d, "
                "alpha=%.2f — n must be >= ceil(1/alpha)-1=%d",
                coverage, n, alpha, int(np.ceil(1.0 / alpha)) - 1,
            )

        if quantile_level > 1.0:
            # Insufficient samples: use max score (most conservative)
            q_hat = float(np.max(scores))
            logger.warning(
                "Conformal quantile_level=%.4f > 1.0 (n=%d, alpha=%.2f) — "
                "using max nonconformity score as threshold",
                quantile_level, n, alpha,
            )
        else:
            # Use method="higher" for finite-sample coverage guarantee
            q_hat = float(np.quantile(scores, quantile_level, method="higher"))

        now_utc = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Conformal fit (gate=%s): q_hat=%.4f, alpha=%.2f, n=%d, "
            "coverage_guarantee=%.4f",
            gate_type, q_hat, alpha, n, coverage,
        )

        return ConformalQuantile(
            quantile_value=float(q_hat),
            alpha=float(alpha),
            n_calibration=n,
            coverage_guarantee=float(coverage),
            computed_at_utc=now_utc,
            gate_type=gate_type,
        )

    def fit_weighted(
        self,
        holdout_labels: np.ndarray,
        holdout_probs: np.ndarray,
        holdout_timestamps: np.ndarray,
        gate_type: str = "meta_prob",
    ) -> ConformalQuantile:
        """Fit time-decay weighted conformal quantile.

        Parameters
        ----------
        holdout_labels : array of shape (N,)
        holdout_probs : array of shape (N,)
        holdout_timestamps : array of shape (N,)
            UTC timestamps (as datetime64 or float epoch seconds).
        gate_type : str
            Which gate this quantile replaces.

        Returns
        -------
        ConformalQuantile
        """
        import pandas as pd

        yt = np.asarray(holdout_labels, dtype=float).reshape(-1)
        yp = np.asarray(holdout_probs, dtype=float).reshape(-1)

        if len(yt) != len(yp):
            raise ValueError(
                f"holdout_labels length ({len(yt)}) != holdout_probs length ({len(yp)})"
            )

        # Remove NaN/Inf
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]
        ts_valid = np.asarray(holdout_timestamps)[valid]

        n = len(yt)
        if n < self._cfg.min_calibration_samples:
            raise ValueError(
                f"Conformal calibration requires at least "
                f"{self._cfg.min_calibration_samples} samples, got {n}"
            )

        # Compute nonconformity scores
        scores = self.compute_nonconformity_scores(yt, yp)

        # Compute time-decay weights
        try:
            ts_series = pd.to_datetime(ts_valid, utc=True, errors="coerce")
            latest = ts_series.max()
            delta = latest - ts_series
            age_days = pd.Series(delta).dt.total_seconds() / 86400.0
            halflife = self._cfg.time_decay_halflife_days
            weights = np.exp(-np.log(2) * np.asarray(age_days, dtype=float) / halflife)
        except Exception as e:
            logger.warning(
                "Time-decay weight computation failed: %s — using uniform", e
            )
            weights = np.ones(n)

        # Guard: replace NaN/Inf weights with zero, fall back to uniform if all zero
        weights = np.where(np.isfinite(weights), weights, 0.0)
        w_sum = weights.sum()
        if w_sum < 1e-30:
            logger.warning("All time-decay weights are zero/NaN — using uniform")
            weights = np.ones(n)
            w_sum = float(n)

        # Normalize weights
        weights = weights / w_sum

        # Weighted quantile (consistent with unweighted: ceil for coverage guarantee)
        alpha = self._cfg.target_alpha
        quantile_level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)

        # Sort by score and compute weighted CDF
        order = np.argsort(scores)
        sorted_scores = scores[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)

        # Find score where cumulative weight exceeds quantile_level
        idx = np.searchsorted(cumulative, quantile_level)
        idx = min(idx, len(sorted_scores) - 1)
        q_hat = float(sorted_scores[idx])

        coverage = 1.0 - alpha - 1.0 / (n + 1)
        now_utc = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Conformal fit_weighted (gate=%s): q_hat=%.4f, alpha=%.2f, n=%d, "
            "coverage_guarantee=%.4f",
            gate_type, q_hat, alpha, n, coverage,
        )

        return ConformalQuantile(
            quantile_value=float(q_hat),
            alpha=float(alpha),
            n_calibration=n,
            coverage_guarantee=float(coverage),
            computed_at_utc=now_utc,
            gate_type=gate_type,
        )

    def should_deploy(
        self,
        candidate_prob: float,
        quantile: ConformalQuantile,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if a candidate exceeds the conformal threshold.

        Parameters
        ----------
        candidate_prob : float
            Predicted probability for the candidate.
        quantile : ConformalQuantile
            Fitted conformal quantile.

        Returns
        -------
        (deploy, details)
            deploy: True if candidate meets the conformal threshold
            details: diagnostic dict with margin and coverage info
        """
        threshold = 1.0 - quantile.quantile_value
        deploy = candidate_prob >= threshold
        margin = candidate_prob - threshold

        details = {
            "candidate_prob": float(candidate_prob),
            "conformal_threshold": float(threshold),
            "quantile_value": float(quantile.quantile_value),
            "margin": float(margin),
            "coverage_guarantee": float(quantile.coverage_guarantee),
            "gate_type": quantile.gate_type,
            "n_calibration": quantile.n_calibration,
        }

        return deploy, details

    def save(
        self,
        quantile: ConformalQuantile,
        path: Optional[Path] = None,
    ) -> None:
        """Save conformal quantile as JSON artifact."""
        p = Path(path) if path else Path(self._cfg.artifact_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(quantile)
        p.write_text(json.dumps(data, indent=2))
        logger.info("ConformalQuantile saved to %s", p)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        config: Optional[ConformalConfig] = None,
        expected_gate_type: Optional[str] = None,
    ) -> Optional[ConformalQuantile]:
        """Load conformal quantile from JSON artifact with compatibility validation.

        Validates that the loaded artifact matches the expected gate type and
        alpha, that it has sufficient calibration samples, and that it is not
        stale.  Returns None (fail-closed) on any validation failure.

        Parameters
        ----------
        path : Path | None
            Override path to the artifact JSON.
        config : ConformalConfig | None
            Configuration to validate against.
        expected_gate_type : str | None
            If provided, the artifact's ``gate_type`` must match.

        Returns
        -------
        ConformalQuantile | None
        """
        cfg = config or ConformalConfig()
        p = Path(path) if path else Path(cfg.artifact_path)
        if not p.exists():
            logger.debug("ConformalQuantile not found at %s", p)
            return None
        try:
            data = json.loads(p.read_text())
            quantile = ConformalQuantile(**data)
        except Exception as e:
            logger.warning("Failed to load ConformalQuantile from %s: %s", p, e)
            return None

        # --- compatibility validation ----------------------------------
        # 1. gate_type must match expected if specified
        if expected_gate_type is not None and quantile.gate_type != expected_gate_type:
            logger.warning(
                "Conformal artifact gate_type mismatch: expected '%s', "
                "got '%s' — ignoring artifact (fail-closed)",
                expected_gate_type,
                quantile.gate_type,
            )
            return None

        # 2. alpha must match the configured target_alpha
        if abs(quantile.alpha - cfg.target_alpha) > 1e-6:
            logger.warning(
                "Conformal artifact alpha mismatch: expected %.4f, "
                "got %.4f — ignoring artifact (fail-closed)",
                cfg.target_alpha,
                quantile.alpha,
            )
            return None

        # 3. minimum calibration samples
        if quantile.n_calibration < cfg.min_calibration_samples:
            logger.warning(
                "Conformal artifact has too few calibration samples: "
                "%d < %d — ignoring artifact (fail-closed)",
                quantile.n_calibration,
                cfg.min_calibration_samples,
            )
            return None

        # 4. staleness check
        try:
            computed = datetime.fromisoformat(quantile.computed_at_utc)
            now = datetime.now(timezone.utc)
            age_days = (now - computed).total_seconds() / 86400.0
            if age_days > cfg.max_artifact_age_days:
                logger.warning(
                    "Conformal artifact is stale: %.1f days old "
                    "(max_artifact_age_days=%.1f) — consider recalibrating",
                    age_days,
                    cfg.max_artifact_age_days,
                )
                # Staleness is a warning, not a hard rejection — the artifact
                # is still structurally valid, just potentially out-of-date.
        except Exception as ts_err:
            logger.warning(
                "Could not parse conformal artifact timestamp: %s", ts_err
            )

        return quantile

    def compute_staleness_days(self, quantile: ConformalQuantile) -> float:
        """Compute how many days old the conformal quantile is."""
        try:
            computed = datetime.fromisoformat(quantile.computed_at_utc)
            now = datetime.now(timezone.utc)
            return float((now - computed).total_seconds() / 86400.0)
        except Exception:
            return float("inf")
