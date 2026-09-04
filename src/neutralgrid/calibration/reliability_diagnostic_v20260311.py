"""
Reliability diagnostics for probability calibration assessment.

Generates reliability diagrams and computes diagnostic metrics to determine
whether sigmoid (Platt scaling) calibration is sufficient or if asymmetric
beta calibration is needed.

Diagnostic trigger: if max_deviation > 0.05 → recommend beta calibration upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReliabilityReport:
    """Immutable result of reliability diagnostic computation."""
    n_bins: int
    bin_edges: List[float]
    bin_counts: List[int]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    ece: float                    # Expected Calibration Error
    mce: float                    # Maximum Calibration Error
    max_deviation: float          # Maximum absolute deviation at any decile
    needs_beta_upgrade: bool      # True if max_deviation > 0.05
    diagnostic_reason: str        # Human-readable reason for the recommendation


class ReliabilityDiagnostic:
    """Compute reliability diagrams and calibration quality metrics.

    Usage:
        diag = ReliabilityDiagnostic()
        report = diag.compute(y_true, y_prob)
        if report.needs_beta_upgrade:
            # Switch to BetaCalibrator
    """

    def compute(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        sample_weight: Optional[np.ndarray] = None,
    ) -> ReliabilityReport:
        """Compute reliability report for binary calibration assessment.

        Parameters
        ----------
        y_true : array of shape (N,)
            Binary ground-truth labels (0 or 1).
        y_prob : array of shape (N,)
            Predicted probabilities for the positive class.
        n_bins : int
            Number of equal-width bins for the reliability diagram.
        sample_weight : array of shape (N,) or None
            Per-sample weights for accuracy/confidence computation.

        Returns
        -------
        ReliabilityReport
        """
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_prob, dtype=float).reshape(-1)
        w = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None

        # ── Edge cases ───────────────────────────────────────────────
        if len(yt) == 0 or len(yp) == 0:
            return self._empty_report(n_bins, "empty_input")

        # Remove NaN/Inf
        valid = np.isfinite(yt) & np.isfinite(yp)
        if w is not None:
            valid = valid & np.isfinite(w)
        if not valid.any():
            return self._empty_report(n_bins, "all_nan_or_inf")
        yt, yp = yt[valid], yp[valid]
        if w is not None:
            w = w[valid]

        # Single-class check
        unique_labels = np.unique(yt)
        if len(unique_labels) < 2:
            return self._empty_report(
                n_bins, f"single_class_only_{int(unique_labels[0])}"
            )

        # ── Bin computation ──────────────────────────────────────────
        n_bins = max(2, int(n_bins))
        bin_edges_arr = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(yp, bin_edges_arr, right=True) - 1
        idx = np.clip(idx, 0, n_bins - 1)

        bin_counts: List[int] = []
        bin_accuracies: List[float] = []
        bin_confidences: List[float] = []
        total_weight = float(np.sum(w)) if w is not None else float(len(yt))
        ece = 0.0
        mce = 0.0

        for b in range(n_bins):
            mask = idx == b
            count = int(np.sum(mask))
            bin_counts.append(count)
            if count == 0:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                continue
            if w is not None:
                acc = float(np.average(yt[mask], weights=w[mask]))
                conf = float(np.average(yp[mask], weights=w[mask]))
                frac = float(np.sum(w[mask])) / total_weight
            else:
                acc = float(np.mean(yt[mask]))
                conf = float(np.mean(yp[mask]))
                frac = count / total_weight
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            gap = abs(acc - conf)
            ece += gap * frac
            mce = max(mce, gap)

        # max_deviation: maximum |acc - conf| over populated bins
        deviations = [
            abs(a - c)
            for a, c, cnt in zip(bin_accuracies, bin_confidences, bin_counts)
            if cnt > 0
        ]
        max_dev = float(max(deviations)) if deviations else 0.0

        # Diagnostic decision
        needs_beta = max_dev > 0.05
        if needs_beta:
            reason = (
                f"max_deviation={max_dev:.4f} > 0.05 threshold — "
                "asymmetric miscalibration detected, beta calibration recommended"
            )
        else:
            reason = (
                f"max_deviation={max_dev:.4f} ≤ 0.05 — "
                "sigmoid calibration sufficient"
            )

        logger.info(
            "ReliabilityDiagnostic: ECE=%.4f, MCE=%.4f, max_dev=%.4f, "
            "needs_beta=%s (N=%d)",
            ece, mce, max_dev, needs_beta, len(yt),
        )

        return ReliabilityReport(
            n_bins=n_bins,
            bin_edges=[float(e) for e in bin_edges_arr],
            bin_counts=bin_counts,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            ece=float(ece),
            mce=float(mce),
            max_deviation=float(max_dev),
            needs_beta_upgrade=needs_beta,
            diagnostic_reason=reason,
        )

    def compute_multiclass(
        self,
        posteriors: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> ReliabilityReport:
        """Compute reliability report for multiclass (HMM) posterior calibration.

        Computes one-vs-rest reliability per class, then aggregates via
        sample-weighted average.

        Parameters
        ----------
        posteriors : array of shape (N, K)
            Posterior probabilities over K classes.
        labels : array of shape (N,)
            Integer ground-truth class labels.
        n_bins : int
            Number of bins per class.

        Returns
        -------
        ReliabilityReport
            Aggregated report across all classes.
        """
        posteriors = np.asarray(posteriors, dtype=float)
        labels = np.asarray(labels, dtype=int).reshape(-1)

        if posteriors.ndim != 2 or len(labels) == 0:
            return self._empty_report(n_bins, "invalid_multiclass_input")

        n_samples, n_classes = posteriors.shape
        if n_samples != len(labels):
            return self._empty_report(n_bins, "shape_mismatch")

        # One-vs-rest per class
        class_eces: List[float] = []
        class_mces: List[float] = []
        class_devs: List[float] = []
        class_counts: List[int] = []

        for k in range(n_classes):
            y_binary = (labels == k).astype(float)
            p_class = posteriors[:, k]
            n_k = int(np.sum(labels == k))
            # Skip classes with too few samples (avoids diluting metrics)
            if n_k < 2 or (n_samples - n_k) < 2:
                continue
            report_k = self.compute(y_binary, p_class, n_bins=n_bins)
            class_eces.append(report_k.ece)
            class_mces.append(report_k.mce)
            class_devs.append(report_k.max_deviation)
            class_counts.append(n_k)

        total = sum(class_counts)
        if total == 0:
            return self._empty_report(n_bins, "no_samples_per_class")

        # Weighted-average ECE
        weights = [c / total for c in class_counts]
        agg_ece = float(sum(e * w for e, w in zip(class_eces, weights)))
        agg_mce = float(max(class_mces)) if class_mces else 0.0
        agg_dev = float(max(class_devs)) if class_devs else 0.0

        needs_beta = agg_dev > 0.05
        reason = (
            f"multiclass agg: ECE={agg_ece:.4f}, MCE={agg_mce:.4f}, "
            f"max_dev={agg_dev:.4f}, beta={'recommended' if needs_beta else 'not needed'}"
        )

        # Return aggregated report (bin-level details from last class for reference)
        bin_edges_arr = np.linspace(0.0, 1.0, n_bins + 1)
        return ReliabilityReport(
            n_bins=n_bins,
            bin_edges=[float(e) for e in bin_edges_arr],
            bin_counts=[total] * n_bins,  # Aggregated
            bin_accuracies=[0.0] * n_bins,
            bin_confidences=[0.0] * n_bins,
            ece=agg_ece,
            mce=agg_mce,
            max_deviation=agg_dev,
            needs_beta_upgrade=needs_beta,
            diagnostic_reason=reason,
        )

    @staticmethod
    def _empty_report(n_bins: int, reason: str) -> ReliabilityReport:
        """Return a safe empty report for edge cases."""
        bin_edges = [float(e) for e in np.linspace(0.0, 1.0, n_bins + 1)]
        return ReliabilityReport(
            n_bins=n_bins,
            bin_edges=bin_edges,
            bin_counts=[0] * n_bins,
            bin_accuracies=[0.0] * n_bins,
            bin_confidences=[0.0] * n_bins,
            ece=0.0,
            mce=0.0,
            max_deviation=0.0,
            needs_beta_upgrade=False,
            diagnostic_reason=f"empty_report:{reason}",
        )
