"""Fit a quarantined diagnostic surrogate for historical ``meta_prob`` snapshots.

This is a probability-distillation tool, not a meta-labeler retraining path.  It
uses observed deployment ``meta_prob`` values as a continuous teacher target and
never creates profitability labels, updates ``models/``, or touches the active
artifact manifest.  The output is deliberately incompatible with the runtime
meta-labeler loader and is marked non-promotable in its manifest.

Example (the last recoverable champion lineage):

    python scripts/build_meta_prob_diagnostic_surrogate.py \
        --source-dir "D:\\Deployment files\\organized" \
        --hmm-version rolling_180d_20260822_203741
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Direct execution must resolve this repository's source, not another editable
# install left on the host.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.models.meta_labeler import META_FEATURE_PROFILES


_CANDIDATE_TS_RE = re.compile(r"_(\d{8}_\d{6})(?:_|$)")
_DEFAULT_PROFILE = "snapshot_v20260530_fastwin"
_DEFAULT_OUTPUT_ROOT = Path("artifacts/diagnostics/meta_prob_surrogate")


def parse_args() -> argparse.Namespace:
    """Return validated command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a diagnostic-only regression surrogate that imitates observed "
            "deployment meta_prob values for one HMM lineage."
        ),
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory recursively containing deployment_ready_*.csv snapshots.",
    )
    parser.add_argument(
        "--hmm-version",
        required=True,
        help="Exact HMM artifact version to isolate; mixed-lineage fitting is refused.",
    )
    parser.add_argument(
        "--feature-profile",
        default=_DEFAULT_PROFILE,
        choices=sorted(META_FEATURE_PROFILES),
        help=f"Feature profile to emulate (default: {_DEFAULT_PROFILE}).",
    )
    parser.add_argument(
        "--output-root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help=(
            "Diagnostic output root. A timestamped child is created here; output "
            "inside models/ or a production artifact directory is refused."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
        help="Fraction of latest distinct snapshot times reserved for OOT validation.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum scored, feature-complete rows required (default: 100).",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_timestamp(candidate_id: Any) -> Any:
    match = _CANDIDATE_TS_RE.search(str(candidate_id))
    if match is None:
        return pd.NaT
    return pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", utc=True, errors="coerce")


def _load_snapshots(source_dir: Path, required: Iterable[str]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    files = sorted(source_dir.rglob("deployment_ready_*.csv"))
    if not files:
        raise FileNotFoundError(f"No deployment_ready CSV files found below {source_dir}")

    frames: list[pd.DataFrame] = []
    fingerprints: list[dict[str, str]] = []
    required_set = set(required)
    for path in files:
        header = pd.read_csv(path, nrows=0)
        missing = required_set.difference(header.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame = pd.read_csv(path, usecols=lambda column: column in required_set, low_memory=False)
        frame["_source_file"] = str(path.resolve())
        frames.append(frame)
        fingerprints.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": str(len(frame)),
            }
        )

    return pd.concat(frames, ignore_index=True), fingerprints


def _deduplicate_candidates(frame: pd.DataFrame, compare_columns: list[str]) -> pd.DataFrame:
    """Remove exact repeated snapshots but reject contradictory duplicate IDs."""
    duplicated = frame.loc[frame.duplicated("candidate_id", keep=False)].copy()
    if not duplicated.empty:
        for candidate_id, group in duplicated.groupby("candidate_id", dropna=False):
            comparison = group[compare_columns].copy()
            for column in compare_columns:
                comparison[column] = comparison[column].map(
                    lambda value: "<NA>" if pd.isna(value) else str(value)
                )
            if len(comparison.drop_duplicates()) != 1:
                raise ValueError(
                    "Candidate ID is present with contradictory probability/feature values: "
                    f"{candidate_id}"
                )
    return frame.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)


def _assert_quarantined_output(root: Path) -> Path:
    resolved = root.resolve()
    project = Path.cwd().resolve()
    forbidden = (project / "models", project / "artifacts" / "hmm", project / "artifacts" / "utility")
    if any(resolved == path or path in resolved.parents for path in forbidden):
        raise ValueError(f"Diagnostic output must not be placed in a production path: {resolved}")
    if not any("diagnostic" in part.lower() for part in resolved.parts):
        raise ValueError("Diagnostic output root must contain a 'diagnostic' path component")
    return resolved


def _chronological_split(frame: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("--holdout-fraction must be greater than 0 and less than 0.5")
    snapshot_times = sorted(cast(pd.Series, frame["_snapshot_time"]).dropna().unique())
    if len(snapshot_times) < 2:
        raise ValueError("At least two distinct candidate snapshot times are required for OOT validation")
    holdout_groups = max(1, math.ceil(len(snapshot_times) * holdout_fraction))
    if holdout_groups >= len(snapshot_times):
        raise ValueError("Holdout would consume every snapshot time")
    holdout_start = snapshot_times[-holdout_groups]
    train = frame.loc[frame["_snapshot_time"] < holdout_start].copy()
    holdout = frame.loc[frame["_snapshot_time"] >= holdout_start].copy()
    if train.empty or holdout.empty:
        raise ValueError("Chronological split produced an empty train or holdout partition")
    return train, holdout


def _metric_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    correlation: float | None = None
    if float(np.std(y_true)) > 0.0 and float(np.std(y_pred)) > 0.0:
        correlation = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": r2,
        "spearman": correlation if correlation is not None and np.isfinite(correlation) else None,
        "max_abs_error": float(np.max(np.abs(y_true - y_pred))),
    }


def main() -> int:
    """Fit the surrogate and write a self-describing diagnostic run directory."""
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")
    if args.min_rows < 2:
        raise ValueError("--min-rows must be at least 2")

    features = list(META_FEATURE_PROFILES[args.feature_profile])
    required_columns = [
        "candidate_id",
        "meta_prob",
        "meta_prob_source",
        "meta_feature_profile",
        "hmm_artifact_version",
        *features,
    ]
    snapshots, source_fingerprints = _load_snapshots(source_dir, required_columns)
    raw_row_count = len(snapshots)
    snapshots = _deduplicate_candidates(snapshots, ["meta_prob", *features])
    deduplicated_row_count = len(snapshots)

    probabilities = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, snapshots["meta_prob"]), errors="coerce"),
    )
    filtered = snapshots.loc[
        snapshots["hmm_artifact_version"].astype("string").eq(args.hmm_version)
        & snapshots["meta_feature_profile"].astype("string").eq(args.feature_profile)
        & probabilities.notna()
    ].copy()
    filtered["_snapshot_time"] = filtered["candidate_id"].map(_candidate_timestamp)
    for feature in features:
        filtered[feature] = pd.to_numeric(cast(pd.Series, filtered[feature]), errors="coerce")
    filtered["meta_prob"] = pd.to_numeric(cast(pd.Series, filtered["meta_prob"]), errors="coerce")
    complete = filtered.dropna(subset=["_snapshot_time", "meta_prob", *features]).copy()
    complete = complete.sort_values(["_snapshot_time", "candidate_id"]).reset_index(drop=True)
    if len(complete) < args.min_rows:
        raise ValueError(
            f"Only {len(complete)} scored, feature-complete rows remain; minimum is {args.min_rows}"
        )
    if not bool(complete["hmm_artifact_version"].eq(args.hmm_version).all()):
        raise AssertionError("Mixed HMM lineage survived filtering")

    train, holdout = _chronological_split(complete, args.holdout_fraction)
    x_train = train[features]
    y_train = np.asarray(train["meta_prob"], dtype=float)
    x_holdout = holdout[features]
    y_holdout = np.asarray(holdout["meta_prob"], dtype=float)

    # Fixed, documented learner.  It is a regressor on teacher probabilities,
    # not a classifier and does not implement the runtime meta-labeler interface.
    estimator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=200,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=20260827,
                ),
            ),
        ]
    )
    estimator.fit(x_train, y_train)
    predicted = np.clip(np.asarray(estimator.predict(x_holdout), dtype=float), 0.0, 1.0)
    metrics = _metric_payload(y_holdout, predicted)
    baseline = np.full(len(y_holdout), float(np.mean(y_train)), dtype=float)
    baseline_metrics = _metric_payload(y_holdout, baseline)

    output_root = _assert_quarantined_output(Path(args.output_root))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"hmm_{args.hmm_version}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    artifact_path = run_dir / "diagnostic_meta_prob_surrogate.joblib"
    artifact = {
        "artifact_kind": "diagnostic_meta_prob_distillation_surrogate",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "runtime_meta_labeler_compatible": False,
        "target": "observed_meta_prob",
        "feature_profile": args.feature_profile,
        "hmm_artifact_version": args.hmm_version,
        "features": features,
        "estimator": estimator,
    }
    joblib.dump(artifact, artifact_path)

    prediction_frame = holdout[["candidate_id", "_snapshot_time", "meta_prob", "_source_file"]].copy()
    prediction_frame["observed_meta_prob"] = prediction_frame.pop("meta_prob")
    prediction_frame["surrogate_meta_prob"] = predicted
    prediction_frame["absolute_error"] = np.abs(y_holdout - predicted)
    prediction_frame["diagnostic_only"] = True
    prediction_frame.to_csv(run_dir / "oot_predictions.csv", index=False)

    manifest: dict[str, Any] = {
        "artifact_kind": "diagnostic_meta_prob_distillation_surrogate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "promotion_eligible": False,
        "not_a_champion_replacement": True,
        "runtime_meta_labeler_compatible": False,
        "synthetic_rows_generated": 0,
        "target_semantics": "observed deployment meta_prob as a soft teacher target",
        "explicit_exclusions": [
            "No fast-winner labels were created or inferred.",
            "No synthetic data are suitable for meta-labeler training or calibration.",
            "No active model, promotion decision, or artifact manifest was changed.",
        ],
        "source_dir": str(source_dir),
        "source_files": source_fingerprints,
        "raw_row_count": raw_row_count,
        "candidate_deduplicated_row_count": deduplicated_row_count,
        "selected_complete_row_count": len(complete),
        "hmm_artifact_version": args.hmm_version,
        "feature_profile": args.feature_profile,
        "features": features,
        "meta_prob_sources": sorted(complete["meta_prob_source"].dropna().astype(str).unique().tolist()),
        "chronological_split": {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "train_snapshot_times": sorted(train["_snapshot_time"].astype(str).unique().tolist()),
            "holdout_snapshot_times": sorted(holdout["_snapshot_time"].astype(str).unique().tolist()),
        },
        "learner": {
            "class": "HistGradientBoostingRegressor",
            "purpose": "continuous probability imitation only",
            "random_state": 20260827,
        },
        "oot_metrics": metrics,
        "constant_mean_baseline_oot_metrics": baseline_metrics,
        "artifact_file": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
        "oot_predictions_file": "oot_predictions.csv",
    }
    (run_dir / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "README.md").write_text(
        "# Diagnostic meta-probability surrogate\n\n"
        "This artifact is a non-promotable distillation surrogate. It imitates historical "
        "`meta_prob` outputs for one stated HMM lineage from deployment snapshots. It is "
        "not a recovered champion, does not contain a fast-winner outcome model, and must "
        "not be used for production sizing, ranking, calibration, promotion, or retraining.\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir.resolve()), "oot_metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
