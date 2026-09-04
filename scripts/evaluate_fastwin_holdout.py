"""Evaluate incumbent and challenger meta-labelers on a frozen FASTWIN holdout.

This script is read-only with respect to production artifacts.  It verifies the
frozen cohort/model hashes, derives the exact endogenous target, writes paired
predictions, and applies a fail-closed promotion rule whose primary statistic
is the scan-file-clustered paired AUC delta confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from neutralgrid.models.meta_labeler import MetaLabeler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1]
            if index < bins - 1
            else probability <= edges[index + 1]
        )
        if bool(mask.any()):
            result += float(mask.mean()) * abs(
                float(y[mask].mean()) - float(probability[mask].mean())
            )
    return float(result)


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece_10_equal_width": _ece(y, probability),
        "accuracy_at_0_5": float(accuracy_score(y, predicted)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, predicted)),
        "precision_at_0_5": float(
            precision_score(y, predicted, zero_division=cast(Any, 0))
        ),
        "recall_at_0_5": float(
            recall_score(y, predicted, zero_division=cast(Any, 0))
        ),
        "probability_min": float(probability.min()),
        "probability_mean": float(probability.mean()),
        "probability_max": float(probability.max()),
    }


def _quantiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _paired_iid_bootstrap(
    y: np.ndarray,
    incumbent: np.ndarray,
    challenger: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    incumbent_aucs: list[float] = []
    challenger_aucs: list[float] = []
    deltas: list[float] = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        labels = y[sample]
        if np.unique(labels).size != 2:
            continue
        incumbent_auc = float(roc_auc_score(labels, incumbent[sample]))
        challenger_auc = float(roc_auc_score(labels, challenger[sample]))
        incumbent_aucs.append(incumbent_auc)
        challenger_aucs.append(challenger_auc)
        deltas.append(challenger_auc - incumbent_auc)
    return {
        "requested_iterations": iterations,
        "valid_iterations": len(deltas),
        "incumbent_auc_ci_95": _quantiles(incumbent_aucs),
        "challenger_auc_ci_95": _quantiles(challenger_aucs),
        "delta_auc_ci_95": _quantiles(deltas),
    }


def _paired_cluster_bootstrap(
    y: np.ndarray,
    incumbent: np.ndarray,
    challenger: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    unique_clusters = np.unique(clusters)
    cluster_indices = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    deltas: list[float] = []
    incumbent_aucs: list[float] = []
    challenger_aucs: list[float] = []
    for _ in range(iterations):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sample = np.concatenate([cluster_indices[cluster] for cluster in sampled_clusters])
        labels = y[sample]
        if np.unique(labels).size != 2:
            continue
        incumbent_auc = float(roc_auc_score(labels, incumbent[sample]))
        challenger_auc = float(roc_auc_score(labels, challenger[sample]))
        incumbent_aucs.append(incumbent_auc)
        challenger_aucs.append(challenger_auc)
        deltas.append(challenger_auc - incumbent_auc)
    return {
        "cluster_key": "scan_file",
        "cluster_count": len(unique_clusters),
        "requested_iterations": iterations,
        "valid_iterations": len(deltas),
        "incumbent_auc_ci_95": _quantiles(incumbent_aucs),
        "challenger_auc_ci_95": _quantiles(challenger_aucs),
        "delta_auc_ci_95": _quantiles(deltas),
    }


def evaluate(
    *,
    holdout_dir: Path,
    incumbent_path: Path,
    challenger_path: Path,
    bootstrap_iterations: int,
    random_state: int,
) -> dict[str, Any]:
    manifest_path = holdout_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_summary = json.loads((holdout_dir / "run_summary.json").read_text(encoding="utf-8"))
    if run_summary.get("complete") is not True or int(run_summary.get("errors", -1)) != 0:
        raise ValueError("holdout run is incomplete or contains errors")
    incumbent_records = manifest.get("incumbent_artifacts", [])
    expected_incumbent_hash = next(
        (
            record["sha256"]
            for record in incumbent_records
            if Path(record["path"]).name == incumbent_path.name
        ),
        None,
    )
    if expected_incumbent_hash is None or _sha256(incumbent_path) != expected_incumbent_hash:
        raise ValueError("incumbent artifact hash differs from the frozen manifest")

    training = pd.read_csv(holdout_dir / "training_data.csv", low_memory=False)
    cohort = pd.read_csv(holdout_dir / "cohort.csv", low_memory=False)
    training_ids = cast(pd.Series, training["candidate_id"]).astype(str)
    cohort_ids = cast(pd.Series, cohort["candidate_id"]).astype(str)
    if len(training) != len(cohort) or set(training_ids) != set(cohort_ids):
        raise ValueError("materialized training rows do not exactly match frozen candidate IDs")
    if bool(training_ids.duplicated().any()):
        raise ValueError("materialized holdout contains duplicate candidate IDs")
    scan_file_by_id = dict(zip(cohort_ids, cast(pd.Series, cohort["scan_file"]).astype(str)))
    scan_files = np.asarray([scan_file_by_id[candidate_id] for candidate_id in training_ids])

    time_to_target = cast(
        pd.Series,
        pd.to_numeric(training["time_to_target_hours"], errors="coerce"),
    )
    y = np.asarray(time_to_target.le(7.0).fillna(False).astype(int), dtype=int)
    if np.unique(y).size != 2:
        raise ValueError("holdout target is single-class")
    incumbent_model = MetaLabeler.load(incumbent_path)
    challenger_model = MetaLabeler.load(challenger_path)
    incumbent_probability = incumbent_model.predict_proba_batch(training)
    challenger_probability = challenger_model.predict_proba_batch(training)
    if not (
        np.isfinite(incumbent_probability).all()
        and np.isfinite(challenger_probability).all()
    ):
        raise ValueError("non-finite model probability")

    incumbent_metrics = _metrics(y, incumbent_probability)
    challenger_metrics = _metrics(y, challenger_probability)
    delta_auc = challenger_metrics["roc_auc"] - incumbent_metrics["roc_auc"]
    iid = _paired_iid_bootstrap(
        y,
        incumbent_probability,
        challenger_probability,
        iterations=bootstrap_iterations,
        rng=np.random.default_rng(random_state),
    )
    clustered = _paired_cluster_bootstrap(
        y,
        incumbent_probability,
        challenger_probability,
        scan_files,
        iterations=bootstrap_iterations,
        rng=np.random.default_rng(random_state + 1),
    )
    cluster_delta_ci = clustered["delta_auc_ci_95"]
    reasons: list[str] = []
    if cluster_delta_ci is None:
        reasons.append("clustered_delta_auc_ci_unavailable")
    elif float(cluster_delta_ci[0]) <= 0.0:
        reasons.append("clustered_delta_auc_ci_includes_zero")
    if challenger_metrics["brier"] > incumbent_metrics["brier"] + 0.005:
        reasons.append("brier_regression_gt_0_005")
    if challenger_metrics["ece_10_equal_width"] > incumbent_metrics["ece_10_equal_width"] + 0.01:
        reasons.append("ece_regression_gt_0_01")

    predictions = pd.DataFrame(
        {
            "candidate_id": training_ids,
            "symbol": training["symbol"],
            "scan_file": scan_files,
            "time_to_target_hours": time_to_target,
            "fast_winner_target": y,
            "incumbent_probability": incumbent_probability,
            "challenger_probability": challenger_probability,
        }
    )
    predictions_path = holdout_dir / "paired_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    report: dict[str, Any] = {
        "target_contract": manifest["target_contract"],
        "holdout": {
            "rows": len(training),
            "candidate_ids_unique": int(training_ids.nunique()),
            "symbols": int(cast(pd.Series, training["symbol"]).nunique()),
            "scan_file_clusters": int(len(np.unique(scan_files))),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "positive_rate": float(y.mean()),
            "time_to_target_finite": int(time_to_target.notna().sum()),
            "time_to_target_gt_7h": int(time_to_target.gt(7.0).sum()),
        },
        "artifacts": {
            "incumbent_path": str(incumbent_path.resolve()),
            "incumbent_sha256": _sha256(incumbent_path),
            "challenger_path": str(challenger_path.resolve()),
            "challenger_sha256": _sha256(challenger_path),
            "predictions_path": str(predictions_path.resolve()),
            "predictions_sha256": _sha256(predictions_path),
        },
        "incumbent": incumbent_metrics,
        "challenger": challenger_metrics,
        "paired_delta_auc": float(delta_auc),
        "iid_bootstrap": iid,
        "scan_file_cluster_bootstrap": clustered,
        "promotion": {
            "promote": not reasons,
            "reasons": reasons,
            "primary_rule": "scan_file_clustered_paired_delta_auc_ci_low_gt_0",
            "brier_guard": "challenger_brier <= incumbent_brier + 0.005",
            "ece_guard": "challenger_ece <= incumbent_ece + 0.01",
        },
    }
    _atomic_write_json(holdout_dir / "evaluation.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, default=Path("models/meta_labeler.pkl"))
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=20260801)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        holdout_dir=args.holdout_dir,
        incumbent_path=args.incumbent,
        challenger_path=args.challenger,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    print(json.dumps(report["promotion"], indent=2))


if __name__ == "__main__":
    main()
