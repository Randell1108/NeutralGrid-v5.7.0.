"""Canonical real-backtest experiment for the four-field FASTWIN profile.

The experiment is intentionally separated into freeze and evaluation phases.
Freeze joins authoritative canonical backtest outcomes to the exact historical
scanner candidate by ``candidate_id`` and retains only rows where all four
pre-outcome profile fields are finite. Development evaluation never opens the
frozen final holdout. Production profile artifacts are outside this module's
write scope.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import numpy as np
import pandas as pd

from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    SHADOW_REALISM_PROFILES,
    validate_realism_output_path,
)
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, PatternProfile
from neutralgrid.scanner.profile_model import (
    ProfileModel,
    load_profile_model,
    save_profile_model,
)
from neutralgrid.scanner.profile_model_walkforward import (
    AUC_FOLD_PASS_THRESHOLD,
    COVERAGE_FLOOR,
    FINITE_FOLD_COVERAGE_FLOOR,
    MEAN_AUC_FLOOR,
    MEAN_PASS_RATE_FLOOR,
    MIN_FINITE_FOLDS,
    POOLED_OOF_AUC_FLOOR,
    WalkForwardResult,
    _auc,
    _expected_calibration_error,
    _ks_statistic,
    _train_from_frame,
    _walkforward_integrity_error,
    promote_profile_version,
)


EXPERIMENT_SCHEMA_VERSION = 1
TARGET_NAME = "fast_winner_time_to_3pct_le_7h"
TARGET_HOURS = 7.0
OBSERVATION_HOURS = 24.0
REALISM_PROFILE = CANDIDATE_TIME_GEOMETRIC_PROFILE
REALISM_SHADOW_PROMOTION_BLOCKER = (
    "realism_profile_shadow_only_pending_event_complete_bot_disjoint_temporal_oos"
)
DEVELOPMENT_FRACTION = 0.80
MIN_SCAN_GROUPS_PER_SPLIT = 10
MIN_CLASS_ROWS = 30
N_DEVELOPMENT_FOLDS = 5
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260801
BRIER_REGRESSION_TOLERANCE = 0.005
ECE_REGRESSION_TOLERANCE = 0.01

_CANDIDATE_RE = re.compile(
    r"^(?P<symbol>.+)_(?P<date>\d{8})_(?P<time>\d{6})(?:_[^_]+)?$"
)
_CANONICAL_REQUIRED = (
    "candidate_id",
    "symbol",
    "start_time_utc",
    "backtest_timestamp",
    "time_to_target_hours",
    "target_reached",
    "source",
    "is_authoritative",
    "engine_version",
    "label_contract_version",
    "formula_version",
    "realism_profile",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _source_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _parse_candidate_identity(candidate_id: str) -> tuple[str, pd.Timestamp]:
    match = _CANDIDATE_RE.fullmatch(candidate_id)
    if match is None:
        raise ValueError(f"invalid candidate_id format: {candidate_id!r}")
    timestamp = pd.to_datetime(
        match.group("date") + match.group("time"),
        format="%Y%m%d%H%M%S",
        utc=True,
        errors="raise",
    )
    return match.group("symbol"), cast(pd.Timestamp, timestamp)


def _true_mask(series: pd.Series) -> pd.Series:
    return cast(pd.Series, series.astype(str).str.strip().str.lower().eq("true"))


def _load_canonical_outcomes(
    fastwin_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(fastwin_dir.glob("training_data_fastwin_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no training_data_fastwin_*.csv files found in {fastwin_dir}"
        )
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(set(_CANONICAL_REQUIRED) - set(frame.columns))
        if missing:
            raise ValueError(f"canonical source {path} missing columns: {missing}")
        frames.append(frame)
        records.append(_source_record(path, rows=len(frame)))
    combined = pd.concat(frames, ignore_index=True, sort=False)

    candidate_ids = (
        cast(pd.Series, combined["candidate_id"])
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if bool(candidate_ids.eq("").any()):
        raise ValueError("canonical outcomes contain blank candidate_id values")
    if bool(candidate_ids.duplicated().any()):
        duplicates = sorted(candidate_ids.loc[candidate_ids.duplicated(keep=False)].unique())
        raise ValueError(f"canonical outcomes contain duplicate candidate IDs: {duplicates[:5]}")
    combined["candidate_id"] = candidate_ids

    expected = {
        "source": "backtest",
        "engine_version": ENGINE_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "formula_version": FORMULA_VERSION,
        "realism_profile": REALISM_PROFILE,
    }
    for column, required in expected.items():
        values = cast(pd.Series, combined[column]).astype(str).str.strip()
        mismatches = int(values.ne(required).sum())
        if mismatches:
            raise ValueError(
                f"canonical {column} mismatch rows={mismatches}; expected={required!r}"
            )
    authoritative = _true_mask(cast(pd.Series, combined["is_authoritative"]))
    if not bool(authoritative.all()):
        raise ValueError(
            f"canonical non-authoritative rows={int((~authoritative).sum())}"
        )

    starts = pd.to_datetime(combined["start_time_utc"], utc=True, errors="coerce")
    backtests = pd.to_datetime(
        combined["backtest_timestamp"], utc=True, errors="coerce"
    )
    if bool(starts.isna().any()) or bool(backtests.isna().any()):
        raise ValueError("canonical timestamps contain invalid values")
    parsed_symbols: list[str] = []
    parsed_times: list[pd.Timestamp] = []
    for candidate_id in candidate_ids:
        symbol, timestamp = _parse_candidate_identity(candidate_id)
        parsed_symbols.append(symbol)
        parsed_times.append(timestamp)
    identity_times = pd.Series(parsed_times, index=combined.index)
    identity_symbols = pd.Series(parsed_symbols, index=combined.index)
    symbols = cast(pd.Series, combined["symbol"]).astype(str).str.strip()
    if not bool(symbols.eq(identity_symbols).all()):
        raise ValueError("canonical symbol disagrees with candidate_id")
    start_series = cast(pd.Series, starts)
    if not bool(start_series.eq(identity_times).all()):
        raise ValueError("canonical start_time_utc disagrees with candidate_id timestamp")
    backtest_series = cast(pd.Series, backtests)
    if not bool(backtest_series.gt(start_series).all()):
        raise ValueError("canonical backtest_timestamp does not follow candidate start")

    time_to_target = cast(
        pd.Series,
        pd.to_numeric(combined["time_to_target_hours"], errors="coerce"),
    )
    finite_time = pd.Series(
        np.isfinite(np.asarray(time_to_target, dtype=float)),
        index=combined.index,
    )
    target_reached = _true_mask(cast(pd.Series, combined["target_reached"]))
    if bool((target_reached & ~finite_time).any()):
        raise ValueError("target_reached rows contain non-finite time_to_target_hours")
    if bool((~target_reached & finite_time).any()):
        raise ValueError("target-not-reached rows contain finite time_to_target_hours")
    if bool(cast(pd.Series, time_to_target.lt(0.0)).fillna(False).any()):
        raise ValueError("canonical time_to_target_hours contains negative values")

    result = combined.copy()
    result["start_time_utc"] = start_series
    result["backtest_timestamp"] = backtest_series
    result["time_to_target_hours"] = time_to_target
    result["fastwin_label"] = (
        finite_time & cast(pd.Series, time_to_target.le(TARGET_HOURS)).fillna(False)
    ).astype(int)
    return result, records


def _load_matching_scanner_rows(
    results_dir: Path,
    candidate_ids: set[str],
    *,
    pattern: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    wanted = {"candidate_id", "symbol", *DEFAULT_FEATURES}
    for path in sorted(results_dir.glob(pattern)):
        header = pd.read_csv(path, nrows=0)
        if "candidate_id" not in header.columns:
            continue
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in wanted,
            low_memory=False,
        )
        ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
        matched = cast(pd.DataFrame, frame.loc[ids.isin(candidate_ids)].copy())
        if matched.empty:
            continue
        matched["candidate_id"] = ids.loc[matched.index]
        for feature in DEFAULT_FEATURES:
            if feature not in matched.columns:
                matched[feature] = np.nan
        matched["scanner_source_file"] = str(path.resolve())
        frames.append(matched)
        records.append(_source_record(path, rows=len(matched)))
    if not frames:
        raise ValueError(f"no scanner rows matched canonical IDs under {pattern}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_mask = cast(pd.Series, combined["candidate_id"]).duplicated(keep=False)
    if bool(duplicate_mask.any()):
        duplicates = sorted(
            cast(pd.Series, combined.loc[duplicate_mask, "candidate_id"])
            .astype(str)
            .unique()
        )
        raise ValueError(
            f"scanner sources contain duplicate canonical candidate IDs: {duplicates[:5]}"
        )
    return combined, records


def _potential_crosscheck(
    results_dir: Path,
    candidate_ids: set[str],
    deployments: pd.DataFrame,
) -> dict[str, Any]:
    potential, potential_records = _load_matching_scanner_rows(
        results_dir,
        candidate_ids,
        pattern="potential_candidates_*.csv",
    )
    joined = deployments.merge(
        potential,
        on="candidate_id",
        how="inner",
        suffixes=("_deployment", "_potential"),
        validate="one_to_one",
    )
    mismatches: dict[str, int] = {}
    maximum_absolute_difference: dict[str, float | None] = {}
    for feature in DEFAULT_FEATURES:
        deployment = cast(
            pd.Series,
            pd.to_numeric(joined[f"{feature}_deployment"], errors="coerce"),
        )
        original = cast(
            pd.Series,
            pd.to_numeric(joined[f"{feature}_potential"], errors="coerce"),
        )
        deployment_finite = np.isfinite(np.asarray(deployment, dtype=float))
        original_finite = np.isfinite(np.asarray(original, dtype=float))
        one_sided = deployment_finite ^ original_finite
        both = deployment_finite & original_finite
        difference = np.abs(
            np.asarray(deployment, dtype=float)[both]
            - np.asarray(original, dtype=float)[both]
        )
        mismatch_count = int(one_sided.sum()) + int((difference > 1e-12).sum())
        mismatches[feature] = mismatch_count
        maximum_absolute_difference[feature] = (
            float(np.max(difference)) if difference.size else None
        )
    if any(mismatches.values()):
        raise ValueError(
            f"potential/deployment profile feature mismatch: {mismatches}"
        )
    return {
        "matched_candidate_ids": len(joined),
        "mismatch_rows_by_feature": mismatches,
        "maximum_absolute_difference": maximum_absolute_difference,
        "source_records": potential_records,
    }


def _incumbent_training_ids(workbook_path: Path) -> set[str]:
    frame = pd.read_excel(
        workbook_path,
        sheet_name="General",
        usecols=lambda column: column == "candidate_id",
    )
    if "candidate_id" not in frame.columns:
        raise ValueError(f"incumbent workbook missing candidate_id: {workbook_path}")
    values = (
        cast(pd.Series, frame["candidate_id"])
        .dropna()
        .astype(str)
        .str.strip()
    )
    return {value for value in values if value and value.lower() != "nan"}


def freeze_experiment(
    *,
    output_dir: Path,
    fastwin_dir: Path,
    results_dir: Path,
    incumbent_model_path: Path,
    incumbent_pattern_path: Path,
    incumbent_workbook_path: Path,
    code_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Freeze a canonical development/final-holdout split without scoring it."""
    validate_realism_output_path(REALISM_PROFILE, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"experiment directory already exists: {output_dir}")
    outcomes, canonical_records = _load_canonical_outcomes(fastwin_dir)
    canonical_ids = set(cast(pd.Series, outcomes["candidate_id"]).astype(str))
    deployments, deployment_records = _load_matching_scanner_rows(
        results_dir,
        canonical_ids,
        pattern="deployment_ready_*.csv",
    )
    if len(deployments) != len(outcomes):
        found = set(cast(pd.Series, deployments["candidate_id"]).astype(str))
        missing = sorted(canonical_ids - found)
        raise ValueError(
            f"deployment exact-ID coverage {len(found)}/{len(canonical_ids)}; "
            f"missing={missing[:5]}"
        )
    crosscheck = _potential_crosscheck(results_dir, canonical_ids, deployments)

    joined = outcomes.merge(
        deployments,
        on="candidate_id",
        how="inner",
        suffixes=("_outcome", "_scanner"),
        validate="one_to_one",
    )
    outcome_symbols = cast(pd.Series, joined["symbol_outcome"]).astype(str)
    scanner_symbols = cast(pd.Series, joined["symbol_scanner"]).astype(str)
    if not bool(outcome_symbols.eq(scanner_symbols).all()):
        raise ValueError("scanner/outcome symbol mismatch after exact candidate join")

    feature_complete = pd.Series(True, index=joined.index, dtype=bool)
    for feature in DEFAULT_FEATURES:
        numeric = cast(pd.Series, pd.to_numeric(joined[feature], errors="coerce"))
        joined[feature] = numeric
        feature_complete &= pd.Series(
            np.isfinite(np.asarray(numeric, dtype=float)),
            index=joined.index,
        )
    recovered = cast(pd.DataFrame, joined.loc[feature_complete].copy())
    if recovered.empty:
        raise ValueError("exact-ID recovery produced zero feature-complete rows")

    incumbent_model = load_profile_model(incumbent_model_path)
    incumbent_pattern = PatternProfile.load_json(incumbent_pattern_path)
    if incumbent_pattern is None:
        raise ValueError("incumbent pattern profile is invalid")
    if incumbent_model.features != list(DEFAULT_FEATURES):
        raise ValueError("incumbent model features differ from active profile contract")
    if incumbent_pattern.features != list(DEFAULT_FEATURES):
        raise ValueError("incumbent pattern features differ from active profile contract")
    incumbent_ids = _incumbent_training_ids(incumbent_workbook_path)
    overlap = set(cast(pd.Series, recovered["candidate_id"]).astype(str)) & incumbent_ids
    eligible = cast(
        pd.DataFrame,
        recovered.loc[
            ~cast(pd.Series, recovered["candidate_id"]).astype(str).isin(overlap)
        ].copy(),
    )
    eligible["symbol"] = cast(pd.Series, eligible["symbol_outcome"]).astype(str)
    eligible = cast(
        pd.DataFrame,
        eligible.sort_values(["start_time_utc", "candidate_id"], kind="stable")
        .reset_index(drop=True),
    )
    scan_groups = sorted(
        cast(pd.Series, eligible["start_time_utc"]).drop_duplicates().tolist()
    )
    if len(scan_groups) < 2 * MIN_SCAN_GROUPS_PER_SPLIT:
        raise ValueError(
            f"insufficient unique scan groups for split: {len(scan_groups)}"
        )
    development_group_count = int(np.floor(len(scan_groups) * DEVELOPMENT_FRACTION))
    development_group_count = min(
        len(scan_groups) - MIN_SCAN_GROUPS_PER_SPLIT,
        max(MIN_SCAN_GROUPS_PER_SPLIT, development_group_count),
    )
    development_groups = set(scan_groups[:development_group_count])
    development = cast(
        pd.DataFrame,
        eligible.loc[
            cast(pd.Series, eligible["start_time_utc"]).isin(development_groups)
        ].copy(),
    )
    holdout = cast(
        pd.DataFrame,
        eligible.loc[
            ~cast(pd.Series, eligible["start_time_utc"]).isin(development_groups)
        ].copy(),
    )
    if development.empty or holdout.empty:
        raise ValueError("frozen development/holdout split is empty")
    development_end = cast(pd.Timestamp, development["start_time_utc"].max())
    holdout_start = cast(pd.Timestamp, holdout["start_time_utc"].min())
    if holdout_start <= development_end:
        raise ValueError("holdout does not begin strictly after development")

    output_columns = [
        "candidate_id",
        "symbol",
        "start_time_utc",
        "backtest_timestamp",
        "time_to_target_hours",
        "target_reached",
        "fastwin_label",
        "scanner_source_file",
        *DEFAULT_FEATURES,
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    development_path = output_dir / "development.csv"
    holdout_path = output_dir / "holdout.csv"
    _atomic_csv(development_path, cast(pd.DataFrame, development[output_columns]))
    _atomic_csv(holdout_path, cast(pd.DataFrame, holdout[output_columns]))

    code_records = [_source_record(path) for path in code_paths]
    manifest: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_contract": {
            "name": TARGET_NAME,
            "definition": "finite time_to_target_hours <= 7.0; otherwise negative",
            "observation_hours": OBSERVATION_HOURS,
            "source": "authoritative canonical backtest rows",
        },
        "feature_contract": {
            "features": list(DEFAULT_FEATURES),
            "source": "exact candidate_id deployment join",
            "imputation_allowed": False,
            "recovered_complete_rows": len(recovered),
            "historical_rows_without_complete_features": len(joined) - len(recovered),
            "potential_deployment_crosscheck": crosscheck,
        },
        "identity_contract": {
            "canonical_rows": len(outcomes),
            "canonical_unique_candidate_ids": len(canonical_ids),
            "deployment_exact_matches": len(deployments),
            "duplicate_matches": 0,
            "incumbent_training_candidate_ids": len(incumbent_ids),
            "excluded_incumbent_overlap_ids": len(overlap),
        },
        "split": {
            "method": "chronological_unique_scan_groups_80_20",
            "development_fraction": DEVELOPMENT_FRACTION,
            "unique_scan_groups": len(scan_groups),
            "development_scan_groups": development_group_count,
            "holdout_scan_groups": len(scan_groups) - development_group_count,
            "development_rows": len(development),
            "holdout_rows": len(holdout),
            "development_end_utc": development_end.isoformat(),
            "holdout_start_utc": holdout_start.isoformat(),
            "purge_hours": OBSERVATION_HOURS,
        },
        "frozen_files": {
            "development": _source_record(development_path, rows=len(development)),
            "holdout": _source_record(holdout_path, rows=len(holdout)),
        },
        "canonical_sources": canonical_records,
        "deployment_sources": deployment_records,
        "incumbent_sources": {
            "model": _source_record(incumbent_model_path),
            "pattern": _source_record(incumbent_pattern_path),
            "training_workbook": _source_record(incumbent_workbook_path),
        },
        "code_sources": code_records,
        "holdout_status": "frozen_unopened_by_evaluator",
        "production_artifacts_modified": False,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    if not path.exists():
        raise ValueError(f"frozen source missing: {path}")
    expected = str(record["sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"frozen source hash mismatch: {path}")
    return path


def _metrics(labels: np.ndarray, scores: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": _auc(scores, labels),
        "ks": _ks_statistic(scores[labels == 1], scores[labels == 0]),
        "brier": float(np.mean(np.square(probabilities - labels))),
        "ece_10_equal_width": _expected_calibration_error(probabilities, labels),
        "accuracy_at_0_5": float(np.mean((probabilities >= 0.5) == labels)),
    }


def _paired_cluster_auc_interval(
    *,
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    incumbent_scores: np.ndarray,
    clusters: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    unique_clusters = np.unique(clusters)
    indices = {
        cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters
    }
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sample = np.concatenate([indices[cluster] for cluster in sampled])
        sample_labels = labels[sample]
        if np.unique(sample_labels).size != 2:
            continue
        delta = _auc(candidate_scores[sample], sample_labels) - _auc(
            incumbent_scores[sample], sample_labels
        )
        if np.isfinite(delta):
            deltas.append(float(delta))
    return {
        "cluster_key": "start_time_utc",
        "cluster_count": len(unique_clusters),
        "requested_replicates": replicates,
        "valid_replicates": len(deltas),
        "delta_auc": float(
            _auc(candidate_scores, labels) - _auc(incumbent_scores, labels)
        ),
        "delta_auc_ci_95": (
            [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
            if deltas
            else None
        ),
    }


def walkforward_exact_fastwin(
    frame: pd.DataFrame,
    *,
    source_sha256: str,
    n_folds: int = N_DEVELOPMENT_FOLDS,
    purge_hours: float = OBSERVATION_HOURS,
    shrinkage: float = 0.30,
) -> WalkForwardResult:
    """Run expanding folds without splitting a scan batch across folds."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if purge_hours < OBSERVATION_HOURS:
        raise ValueError("purge_hours must cover the full 24h outcome window")
    required = {"candidate_id", "start_time_utc", "fastwin_label", *DEFAULT_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"development frame missing columns: {missing}")
    data = frame.copy()
    data["start_time_utc"] = pd.to_datetime(
        data["start_time_utc"], utc=True, errors="coerce"
    )
    if bool(cast(pd.Series, data["start_time_utc"]).isna().any()):
        raise ValueError("development frame contains invalid start_time_utc")
    if bool(cast(pd.Series, data["candidate_id"]).astype(str).duplicated().any()):
        raise ValueError("development frame contains duplicate candidate_id")
    labels = cast(
        pd.Series,
        pd.to_numeric(data["fastwin_label"], errors="coerce"),
    )
    if not set(labels.dropna().astype(int).unique()).issubset({0, 1}):
        raise ValueError("development labels are not binary")
    data["_is_winner"] = labels.astype(int)
    for feature in DEFAULT_FEATURES:
        numeric = cast(pd.Series, pd.to_numeric(data[feature], errors="coerce"))
        if not bool(np.isfinite(np.asarray(numeric, dtype=float)).all()):
            raise ValueError(f"development feature is non-finite: {feature}")
        data[feature] = numeric
    data = cast(
        pd.DataFrame,
        data.sort_values(["start_time_utc", "candidate_id"], kind="stable")
        .reset_index(drop=True),
    )
    groups = sorted(cast(pd.Series, data["start_time_utc"]).drop_duplicates().tolist())
    purge = pd.Timedelta(hours=float(purge_hours))
    first_test_group: int | None = None
    for group_index, test_start in enumerate(groups):
        train = cast(
            pd.DataFrame,
            data.loc[cast(pd.Series, data["start_time_utc"]) + purge < test_start],
        )
        winners = int(cast(pd.Series, train["_is_winner"]).sum())
        if winners >= MIN_CLASS_ROWS and len(train) - winners >= MIN_CLASS_ROWS:
            first_test_group = group_index
            break
    if first_test_group is None:
        raise ValueError("no purge-safe development origin reaches the class floor")
    remaining_groups = groups[first_test_group:]
    if len(remaining_groups) < n_folds:
        raise ValueError("insufficient future scan groups for requested folds")
    fold_groups = [list(chunk) for chunk in np.array_split(remaining_groups, n_folds)]

    fold_auc: list[float] = []
    fold_ks: list[float] = []
    fold_train_rows: list[int] = []
    fold_test_rows: list[int] = []
    fold_train_winners: list[int] = []
    fold_test_winners: list[int] = []
    fold_test_start_utc: list[str] = []
    fold_test_end_utc: list[str] = []
    oof_ids: list[str] = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    oof_probabilities: list[float] = []
    for test_groups in fold_groups:
        test_start = cast(pd.Timestamp, min(test_groups))
        train = cast(
            pd.DataFrame,
            data.loc[cast(pd.Series, data["start_time_utc"]) + purge < test_start].copy(),
        )
        test = cast(
            pd.DataFrame,
            data.loc[cast(pd.Series, data["start_time_utc"]).isin(test_groups)].copy(),
        )
        train_winners = int(cast(pd.Series, train["_is_winner"]).sum())
        test_winners = int(cast(pd.Series, test["_is_winner"]).sum())
        fold_train_rows.append(len(train))
        fold_test_rows.append(len(test))
        fold_train_winners.append(train_winners)
        fold_test_winners.append(test_winners)
        fold_test_start_utc.append(test_start.isoformat())
        fold_test_end_utc.append(
            cast(pd.Timestamp, test["start_time_utc"].max()).isoformat()
        )
        model = _train_from_frame(
            train,
            list(DEFAULT_FEATURES),
            shrinkage=shrinkage,
            max_duration_hours=TARGET_HOURS,
        )
        if model is None:
            fold_auc.append(float("nan"))
            fold_ks.append(float("nan"))
            continue
        scores: list[float] = []
        fold_labels: list[int] = []
        for _, row in test.iterrows():
            raw = {feature: row[feature] for feature in DEFAULT_FEATURES}
            score = model.llr(raw)
            probability = model.proba(raw)
            if score is None or probability is None:
                raise ValueError("profile model could not score a feature-complete row")
            label = int(row["_is_winner"])
            candidate_id = str(row["candidate_id"])
            scores.append(float(score))
            fold_labels.append(label)
            oof_ids.append(candidate_id)
            oof_labels.append(label)
            oof_scores.append(float(score))
            oof_probabilities.append(float(probability))
        score_array = np.asarray(scores, dtype=float)
        label_array = np.asarray(fold_labels, dtype=int)
        fold_auc.append(_auc(score_array, label_array))
        fold_ks.append(
            _ks_statistic(
                score_array[label_array == 1], score_array[label_array == 0]
            )
        )

    finite_auc = np.asarray([value for value in fold_auc if np.isfinite(value)])
    finite_ks = np.asarray([value for value in fold_ks if np.isfinite(value)])
    score_array = np.asarray(oof_scores, dtype=float)
    label_array = np.asarray(oof_labels, dtype=int)
    probability_array = np.asarray(oof_probabilities, dtype=float)
    return WalkForwardResult(
        n_folds=n_folds,
        fold_auc=fold_auc,
        fold_ks=fold_ks,
        mean_auc=float(np.mean(finite_auc)) if finite_auc.size else float("nan"),
        mean_ks=float(np.mean(finite_ks)) if finite_ks.size else float("nan"),
        mean_pass_rate=(
            float(np.mean(finite_auc >= AUC_FOLD_PASS_THRESHOLD))
            if finite_auc.size
            else 0.0
        ),
        purge_hours=purge_hours,
        requested_features=list(DEFAULT_FEATURES),
        admitted_features=list(DEFAULT_FEATURES),
        feature_coverage=1.0,
        fold_train_rows=fold_train_rows,
        fold_test_rows=fold_test_rows,
        fold_train_winners=fold_train_winners,
        fold_test_winners=fold_test_winners,
        fold_pnl_thresholds=[float("nan")] * n_folds,
        fold_test_start_utc=fold_test_start_utc,
        fold_test_end_utc=fold_test_end_utc,
        oof_strategy_ids=oof_ids,
        oof_labels=oof_labels,
        oof_scores=oof_scores,
        oof_probabilities=oof_probabilities,
        pooled_oof_auc=_auc(score_array, label_array),
        pooled_oof_ks=_ks_statistic(
            score_array[label_array == 1], score_array[label_array == 0]
        ),
        pooled_oof_brier=float(np.mean(np.square(probability_array - label_array))),
        pooled_oof_ece=_expected_calibration_error(probability_array, label_array),
        source_sha256=source_sha256,
        labeled_rows=len(data),
        duplicate_strategy_ids=0,
        holdout_start_after_utc=None,
    )


def _build_pattern(frame: pd.DataFrame, selection_summary: dict[str, Any]) -> PatternProfile:
    winners = cast(pd.DataFrame, frame.loc[cast(pd.Series, frame["_is_winner"]).eq(1)])
    if len(winners) < MIN_CLASS_ROWS:
        raise ValueError("insufficient exact FASTWIN positives for pattern profile")
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    q10: dict[str, float] = {}
    q90: dict[str, float] = {}
    for feature in DEFAULT_FEATURES:
        values = cast(pd.Series, pd.to_numeric(winners[feature], errors="coerce"))
        if not bool(np.isfinite(np.asarray(values, dtype=float)).all()):
            raise ValueError(f"winner pattern feature is non-finite: {feature}")
        means[feature] = float(values.mean())
        standard_deviation = float(values.std(ddof=0))
        stds[feature] = standard_deviation if standard_deviation > 0.0 else 1.0
        q10[feature] = float(values.quantile(0.10))
        q90[feature] = float(values.quantile(0.90))
    return PatternProfile(
        features=list(DEFAULT_FEATURES),
        means=means,
        stds=stds,
        q10=q10,
        q90=q90,
        selection_summary=selection_summary,
    )


def evaluate_development(experiment_dir: Path) -> dict[str, Any]:
    """Evaluate only development rows; never open the frozen holdout CSV."""
    validate_realism_output_path(REALISM_PROFILE, experiment_dir)
    report_path = experiment_dir / "development_evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"development was already evaluated: {report_path}")
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical profile experiment schema")
    for section in ("canonical_sources", "deployment_sources", "code_sources"):
        for record in manifest.get(section, []):
            _verify_record(record)
    for record in manifest["feature_contract"]["potential_deployment_crosscheck"].get(
        "source_records", []
    ):
        _verify_record(record)
    incumbent_sources = manifest["incumbent_sources"]
    incumbent_model_path = _verify_record(incumbent_sources["model"])
    _verify_record(incumbent_sources["pattern"])
    _verify_record(incumbent_sources["training_workbook"])
    development_record = manifest["frozen_files"]["development"]
    development_path = _verify_record(development_record)
    # Deliberately do not read or hash the holdout path here.
    development = pd.read_csv(development_path, low_memory=False)
    source_sha256 = str(development_record["sha256"])
    wf = walkforward_exact_fastwin(
        development,
        source_sha256=source_sha256,
    )
    integrity_error = _walkforward_integrity_error(wf)

    indexed = development.set_index("candidate_id", drop=False)
    oof = cast(pd.DataFrame, indexed.loc[wf.oof_strategy_ids].copy())
    incumbent = load_profile_model(incumbent_model_path)
    incumbent_scores: list[float] = []
    incumbent_probabilities: list[float] = []
    for _, row in oof.iterrows():
        raw = {feature: row[feature] for feature in DEFAULT_FEATURES}
        score = incumbent.llr(raw)
        probability = incumbent.proba(raw)
        if score is None or probability is None:
            raise ValueError("incumbent could not score feature-complete OOF row")
        incumbent_scores.append(float(score))
        incumbent_probabilities.append(float(probability))
    labels = np.asarray(wf.oof_labels, dtype=int)
    candidate_scores = np.asarray(wf.oof_scores, dtype=float)
    candidate_probabilities = np.asarray(wf.oof_probabilities, dtype=float)
    incumbent_score_array = np.asarray(incumbent_scores, dtype=float)
    incumbent_probability_array = np.asarray(incumbent_probabilities, dtype=float)
    clusters = np.asarray(
        pd.to_datetime(oof["start_time_utc"], utc=True, errors="raise")
        .astype(str)
        .tolist()
    )
    candidate_metrics = _metrics(labels, candidate_scores, candidate_probabilities)
    incumbent_metrics = _metrics(
        labels, incumbent_score_array, incumbent_probability_array
    )
    paired = _paired_cluster_auc_interval(
        labels=labels,
        candidate_scores=candidate_scores,
        incumbent_scores=incumbent_score_array,
        clusters=clusters,
    )

    finite_fold_count = int(sum(np.isfinite(value) for value in wf.fold_auc))
    finite_fold_coverage = finite_fold_count / len(wf.fold_auc) if wf.fold_auc else 0.0
    reasons: list[str] = []
    if integrity_error is not None:
        reasons.append(f"walkforward_integrity={integrity_error}")
    if finite_fold_count < MIN_FINITE_FOLDS:
        reasons.append("finite_fold_count_below_floor")
    if finite_fold_coverage < FINITE_FOLD_COVERAGE_FLOOR:
        reasons.append("finite_fold_coverage_below_floor")
    if not np.isfinite(wf.mean_auc) or wf.mean_auc < MEAN_AUC_FLOOR:
        reasons.append("mean_auc_below_floor")
    if not np.isfinite(wf.pooled_oof_auc) or wf.pooled_oof_auc < POOLED_OOF_AUC_FLOOR:
        reasons.append("pooled_oof_auc_below_floor")
    if wf.mean_pass_rate < MEAN_PASS_RATE_FLOOR:
        reasons.append("mean_pass_rate_below_floor")
    if wf.feature_coverage < COVERAGE_FLOOR:
        reasons.append("feature_coverage_below_floor")
    paired_interval = paired["delta_auc_ci_95"]
    if paired_interval is None or float(paired_interval[0]) <= 0.0:
        reasons.append("clustered_paired_auc_delta_ci_includes_zero")
    if candidate_metrics["brier"] > incumbent_metrics["brier"] + BRIER_REGRESSION_TOLERANCE:
        reasons.append("brier_regression_gt_0_005")
    if (
        candidate_metrics["ece_10_equal_width"]
        > incumbent_metrics["ece_10_equal_width"] + ECE_REGRESSION_TOLERANCE
    ):
        reasons.append("ece_regression_gt_0_01")
    holdout_start = cast(
        pd.Timestamp,
        pd.to_datetime(
            manifest["split"]["holdout_start_utc"], utc=True, errors="raise"
        ),
    )
    data = development.copy()
    data["start_time_utc"] = pd.to_datetime(data["start_time_utc"], utc=True, errors="raise")
    data["_is_winner"] = cast(
        pd.Series,
        pd.to_numeric(data["fastwin_label"], errors="raise"),
    ).astype(int)
    final_train = cast(
        pd.DataFrame,
        data.loc[
            cast(pd.Series, data["start_time_utc"])
            + pd.Timedelta(hours=OBSERVATION_HOURS)
            < holdout_start
        ].copy(),
    )
    final_winners = int(cast(pd.Series, final_train["_is_winner"]).sum())
    if final_winners < MIN_CLASS_ROWS or len(final_train) - final_winners < MIN_CLASS_ROWS:
        raise ValueError("purged final development fit does not meet class floors")
    candidate_model = _train_from_frame(
        final_train,
        list(DEFAULT_FEATURES),
        shrinkage=0.30,
        max_duration_hours=TARGET_HOURS,
    )
    if candidate_model is None:
        raise ValueError("canonical final development model fit failed")
    selection_summary = {
        "label_name": TARGET_NAME,
        "label_definition": {
            "time_to_target_rule": "finite time_to_target_hours <= 7.0",
            "time_to_target_claimed": True,
            "source": "authoritative canonical realistic backtest",
        },
        "features": list(DEFAULT_FEATURES),
        "fit_rows": len(final_train),
        "winners_count": final_winners,
        "losers_count": len(final_train) - final_winners,
        "purge_hours": OBSERVATION_HOURS,
        "shrinkage": 0.30,
        "development_sha256": source_sha256,
        "holdout_evaluated": False,
    }
    candidate_model = replace(candidate_model, selection_summary=selection_summary)
    pattern = _build_pattern(final_train, selection_summary)
    candidate_model_path = experiment_dir / "candidate_profile_model.json"
    candidate_pattern_path = experiment_dir / "candidate_pattern_profile.json"
    save_profile_model(candidate_model, candidate_model_path)
    pattern.save_json(candidate_pattern_path)

    report: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_source_sha256": source_sha256,
        "candidate_walkforward": asdict(wf),
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "paired_cluster_bootstrap": paired,
        "gates": {
            "finite_fold_count": finite_fold_count,
            "finite_fold_count_floor": MIN_FINITE_FOLDS,
            "finite_fold_coverage": finite_fold_coverage,
            "finite_fold_coverage_floor": FINITE_FOLD_COVERAGE_FLOOR,
            "mean_auc_floor": MEAN_AUC_FLOOR,
            "pooled_oof_auc_floor": POOLED_OOF_AUC_FLOOR,
            "mean_pass_rate_floor": MEAN_PASS_RATE_FLOOR,
            "feature_coverage_floor": COVERAGE_FLOOR,
            "clustered_paired_auc_delta_ci_low_floor_exclusive": 0.0,
            "brier_regression_tolerance": BRIER_REGRESSION_TOLERANCE,
            "ece_regression_tolerance": ECE_REGRESSION_TOLERANCE,
        },
        "decision": {
            "holdout_eligible": not reasons,
            "promotion_attempted": False,
            "production_artifacts_modified": False,
            "reasons": reasons,
        },
        "candidate_artifacts": {
            "model": _source_record(candidate_model_path),
            "pattern": _source_record(candidate_pattern_path),
        },
        "final_fit": {
            "rows": len(final_train),
            "winners": final_winners,
            "losers": len(final_train) - final_winners,
            "latest_start_utc": cast(
                pd.Timestamp, final_train["start_time_utc"].max()
            ).isoformat(),
            "holdout_start_utc": cast(pd.Timestamp, holdout_start).isoformat(),
        },
        "holdout_status": (
            "eligible_but_not_evaluated"
            if not reasons
            else "unopened_development_gate_failed"
        ),
    }
    _atomic_json(report_path, report)
    return report


def _fixed_model_holdout_result(
    frame: pd.DataFrame,
    *,
    model: ProfileModel,
    source_sha256: str,
    holdout_start_after_utc: str,
    train_rows: int,
    train_winners: int,
    n_folds: int = N_DEVELOPMENT_FOLDS,
) -> WalkForwardResult:
    """Score a frozen model on chronological, scan-group-preserving folds."""
    data = frame.copy()
    data["start_time_utc"] = pd.to_datetime(
        data["start_time_utc"], utc=True, errors="coerce"
    )
    if bool(cast(pd.Series, data["start_time_utc"]).isna().any()):
        raise ValueError("holdout contains invalid start_time_utc")
    ids = cast(pd.Series, data["candidate_id"]).astype(str)
    if bool(ids.duplicated().any()):
        raise ValueError("holdout contains duplicate candidate IDs")
    labels = cast(
        pd.Series,
        pd.to_numeric(data["fastwin_label"], errors="coerce"),
    )
    if bool(labels.isna().any()) or not set(labels.astype(int).unique()).issubset(
        {0, 1}
    ):
        raise ValueError("holdout labels are not complete binary values")
    data["_is_winner"] = labels.astype(int)
    for feature in DEFAULT_FEATURES:
        numeric = cast(pd.Series, pd.to_numeric(data[feature], errors="coerce"))
        if not bool(np.isfinite(np.asarray(numeric, dtype=float)).all()):
            raise ValueError(f"holdout feature is non-finite: {feature}")
        data[feature] = numeric
    data = cast(
        pd.DataFrame,
        data.sort_values(["start_time_utc", "candidate_id"], kind="stable")
        .reset_index(drop=True),
    )
    groups = sorted(cast(pd.Series, data["start_time_utc"]).drop_duplicates().tolist())
    if len(groups) < n_folds:
        raise ValueError("insufficient holdout scan groups for fold metrics")
    fold_groups = [list(chunk) for chunk in np.array_split(groups, n_folds)]

    fold_auc: list[float] = []
    fold_ks: list[float] = []
    fold_test_rows: list[int] = []
    fold_test_winners: list[int] = []
    fold_test_start_utc: list[str] = []
    fold_test_end_utc: list[str] = []
    oof_ids: list[str] = []
    oof_labels: list[int] = []
    oof_scores: list[float] = []
    oof_probabilities: list[float] = []
    for test_groups in fold_groups:
        test = cast(
            pd.DataFrame,
            data.loc[cast(pd.Series, data["start_time_utc"]).isin(test_groups)].copy(),
        )
        fold_scores: list[float] = []
        fold_labels: list[int] = []
        for _, row in test.iterrows():
            raw = {feature: row[feature] for feature in DEFAULT_FEATURES}
            score = model.llr(raw)
            probability = model.proba(raw)
            if score is None or probability is None:
                raise ValueError("fixed profile model could not score holdout row")
            label = int(row["_is_winner"])
            candidate_id = str(row["candidate_id"])
            fold_scores.append(float(score))
            fold_labels.append(label)
            oof_ids.append(candidate_id)
            oof_labels.append(label)
            oof_scores.append(float(score))
            oof_probabilities.append(float(probability))
        score_array = np.asarray(fold_scores, dtype=float)
        label_array = np.asarray(fold_labels, dtype=int)
        fold_auc.append(_auc(score_array, label_array))
        fold_ks.append(
            _ks_statistic(
                score_array[label_array == 1], score_array[label_array == 0]
            )
        )
        fold_test_rows.append(len(test))
        fold_test_winners.append(int(label_array.sum()))
        fold_test_start_utc.append(
            cast(pd.Timestamp, test["start_time_utc"].min()).isoformat()
        )
        fold_test_end_utc.append(
            cast(pd.Timestamp, test["start_time_utc"].max()).isoformat()
        )

    finite_auc = np.asarray([value for value in fold_auc if np.isfinite(value)])
    finite_ks = np.asarray([value for value in fold_ks if np.isfinite(value)])
    score_array = np.asarray(oof_scores, dtype=float)
    label_array = np.asarray(oof_labels, dtype=int)
    probability_array = np.asarray(oof_probabilities, dtype=float)
    return WalkForwardResult(
        n_folds=n_folds,
        fold_auc=fold_auc,
        fold_ks=fold_ks,
        mean_auc=float(np.mean(finite_auc)) if finite_auc.size else float("nan"),
        mean_ks=float(np.mean(finite_ks)) if finite_ks.size else float("nan"),
        mean_pass_rate=(
            float(np.mean(finite_auc >= AUC_FOLD_PASS_THRESHOLD))
            if finite_auc.size
            else 0.0
        ),
        purge_hours=OBSERVATION_HOURS,
        requested_features=list(DEFAULT_FEATURES),
        admitted_features=list(DEFAULT_FEATURES),
        feature_coverage=1.0,
        fold_train_rows=[train_rows] * n_folds,
        fold_test_rows=fold_test_rows,
        fold_train_winners=[train_winners] * n_folds,
        fold_test_winners=fold_test_winners,
        fold_pnl_thresholds=[float("nan")] * n_folds,
        fold_test_start_utc=fold_test_start_utc,
        fold_test_end_utc=fold_test_end_utc,
        oof_strategy_ids=oof_ids,
        oof_labels=oof_labels,
        oof_scores=oof_scores,
        oof_probabilities=oof_probabilities,
        pooled_oof_auc=_auc(score_array, label_array),
        pooled_oof_ks=_ks_statistic(
            score_array[label_array == 1], score_array[label_array == 0]
        ),
        pooled_oof_brier=float(np.mean(np.square(probability_array - label_array))),
        pooled_oof_ece=_expected_calibration_error(probability_array, label_array),
        source_sha256=source_sha256,
        labeled_rows=len(data),
        duplicate_strategy_ids=0,
        holdout_start_after_utc=holdout_start_after_utc,
    )


def evaluate_holdout(experiment_dir: Path) -> dict[str, Any]:
    """Consume the frozen final holdout once after the development gate passes."""
    validate_realism_output_path(REALISM_PROFILE, experiment_dir)
    report_path = experiment_dir / "holdout_evaluation.json"
    predictions_path = experiment_dir / "holdout_predictions.csv"
    if report_path.exists() or predictions_path.exists():
        raise FileExistsError("canonical FASTWIN holdout was already evaluated")
    manifest_path = experiment_dir / "manifest.json"
    development_report_path = experiment_dir / "development_evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_report = json.loads(
        development_report_path.read_text(encoding="utf-8")
    )
    decision = development_report.get("decision", {})
    if decision.get("holdout_eligible") is not True or decision.get("reasons"):
        raise ValueError("development gate did not authorize holdout evaluation")
    for section in ("canonical_sources", "deployment_sources", "code_sources"):
        for record in manifest.get(section, []):
            _verify_record(record)
    for record in manifest["feature_contract"]["potential_deployment_crosscheck"].get(
        "source_records", []
    ):
        _verify_record(record)
    incumbent_sources = manifest["incumbent_sources"]
    incumbent_model_path = _verify_record(incumbent_sources["model"])
    _verify_record(incumbent_sources["pattern"])
    _verify_record(incumbent_sources["training_workbook"])
    candidate_sources = development_report["candidate_artifacts"]
    candidate_model_path = _verify_record(candidate_sources["model"])
    _verify_record(candidate_sources["pattern"])
    holdout_record = manifest["frozen_files"]["holdout"]
    holdout_path = _verify_record(holdout_record)
    holdout = pd.read_csv(holdout_path, low_memory=False)

    candidate_model = load_profile_model(candidate_model_path)
    incumbent_model = load_profile_model(incumbent_model_path)
    final_fit = development_report["final_fit"]
    cutoff = str(manifest["split"]["development_end_utc"])
    candidate_result = _fixed_model_holdout_result(
        holdout,
        model=candidate_model,
        source_sha256=str(holdout_record["sha256"]),
        holdout_start_after_utc=cutoff,
        train_rows=int(final_fit["rows"]),
        train_winners=int(final_fit["winners"]),
    )
    incumbent_result = _fixed_model_holdout_result(
        holdout,
        model=incumbent_model,
        source_sha256=str(holdout_record["sha256"]),
        holdout_start_after_utc=cutoff,
        train_rows=int(manifest["identity_contract"]["incumbent_training_candidate_ids"]),
        train_winners=int(
            (incumbent_model.selection_summary or {}).get("winners_count", 0)
        ),
    )
    candidate_integrity = _walkforward_integrity_error(candidate_result)
    incumbent_integrity = _walkforward_integrity_error(incumbent_result)
    if candidate_result.oof_strategy_ids != incumbent_result.oof_strategy_ids:
        raise ValueError("candidate/incumbent holdout ID order differs")
    if candidate_result.oof_labels != incumbent_result.oof_labels:
        raise ValueError("candidate/incumbent holdout labels differ")

    labels = np.asarray(candidate_result.oof_labels, dtype=int)
    candidate_scores = np.asarray(candidate_result.oof_scores, dtype=float)
    candidate_probabilities = np.asarray(
        candidate_result.oof_probabilities, dtype=float
    )
    incumbent_scores = np.asarray(incumbent_result.oof_scores, dtype=float)
    incumbent_probabilities = np.asarray(
        incumbent_result.oof_probabilities, dtype=float
    )
    indexed = holdout.set_index("candidate_id", drop=False)
    ordered = cast(
        pd.DataFrame,
        indexed.loc[candidate_result.oof_strategy_ids].copy(),
    )
    clusters = np.asarray(
        pd.to_datetime(ordered["start_time_utc"], utc=True, errors="raise")
        .astype(str)
        .tolist()
    )
    candidate_metrics = _metrics(labels, candidate_scores, candidate_probabilities)
    incumbent_metrics = _metrics(labels, incumbent_scores, incumbent_probabilities)
    paired = _paired_cluster_auc_interval(
        labels=labels,
        candidate_scores=candidate_scores,
        incumbent_scores=incumbent_scores,
        clusters=clusters,
    )

    finite_fold_count = int(
        sum(np.isfinite(value) for value in candidate_result.fold_auc)
    )
    finite_fold_coverage = (
        finite_fold_count / len(candidate_result.fold_auc)
        if candidate_result.fold_auc
        else 0.0
    )
    reasons: list[str] = []
    if candidate_integrity is not None:
        reasons.append(f"candidate_integrity={candidate_integrity}")
    if incumbent_integrity is not None:
        reasons.append(f"incumbent_integrity={incumbent_integrity}")
    if finite_fold_count < MIN_FINITE_FOLDS:
        reasons.append("finite_fold_count_below_floor")
    if finite_fold_coverage < FINITE_FOLD_COVERAGE_FLOOR:
        reasons.append("finite_fold_coverage_below_floor")
    if candidate_result.mean_auc < MEAN_AUC_FLOOR:
        reasons.append("mean_auc_below_floor")
    if candidate_result.pooled_oof_auc < POOLED_OOF_AUC_FLOOR:
        reasons.append("pooled_oof_auc_below_floor")
    if candidate_result.mean_pass_rate < MEAN_PASS_RATE_FLOOR:
        reasons.append("mean_pass_rate_below_floor")
    paired_interval = paired["delta_auc_ci_95"]
    if paired_interval is None or float(paired_interval[0]) <= 0.0:
        reasons.append("clustered_paired_auc_delta_ci_includes_zero")
    if candidate_metrics["brier"] > incumbent_metrics["brier"] + BRIER_REGRESSION_TOLERANCE:
        reasons.append("brier_regression_gt_0_005")
    if (
        candidate_metrics["ece_10_equal_width"]
        > incumbent_metrics["ece_10_equal_width"] + ECE_REGRESSION_TOLERANCE
    ):
        reasons.append("ece_regression_gt_0_01")
    if REALISM_PROFILE in SHADOW_REALISM_PROFILES:
        reasons.append(REALISM_SHADOW_PROMOTION_BLOCKER)

    predictions = pd.DataFrame(
        {
            "candidate_id": candidate_result.oof_strategy_ids,
            "symbol": ordered["symbol"].astype(str).tolist(),
            "start_time_utc": ordered["start_time_utc"].astype(str).tolist(),
            "fastwin_label": labels,
            "candidate_score": candidate_scores,
            "candidate_probability": candidate_probabilities,
            "incumbent_score": incumbent_scores,
            "incumbent_probability": incumbent_probabilities,
        }
    )
    _atomic_csv(predictions_path, predictions)
    report: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
        "development_evaluation_sha256": sha256_file(development_report_path),
        "holdout_sha256": str(holdout_record["sha256"]),
        "predictions": _source_record(predictions_path, rows=len(predictions)),
        "candidate_result": asdict(candidate_result),
        "incumbent_result": asdict(incumbent_result),
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "paired_cluster_bootstrap": paired,
        "gates": {
            "finite_fold_count": finite_fold_count,
            "finite_fold_count_floor": MIN_FINITE_FOLDS,
            "finite_fold_coverage": finite_fold_coverage,
            "finite_fold_coverage_floor": FINITE_FOLD_COVERAGE_FLOOR,
            "mean_auc_floor": MEAN_AUC_FLOOR,
            "pooled_oof_auc_floor": POOLED_OOF_AUC_FLOOR,
            "mean_pass_rate_floor": MEAN_PASS_RATE_FLOOR,
            "clustered_paired_auc_delta_ci_low_floor_exclusive": 0.0,
            "brier_regression_tolerance": BRIER_REGRESSION_TOLERANCE,
            "ece_regression_tolerance": ECE_REGRESSION_TOLERANCE,
        },
        "decision": {
            "promotion_eligible": not reasons,
            "promotion_attempted": False,
            "production_artifacts_modified": False,
            "reasons": reasons,
        },
        "candidate_artifacts": candidate_sources,
        "holdout_status": "consumed_once",
    }
    _atomic_json(report_path, report)
    return report


def _walkforward_from_json(payload: Mapping[str, Any]) -> WalkForwardResult:
    values = dict(payload)
    values["fold_pnl_thresholds"] = [
        float("nan") if value is None else float(value)
        for value in values.get("fold_pnl_thresholds", [])
    ]
    return WalkForwardResult(**values)


def promote_from_holdout(
    experiment_dir: Path,
    *,
    profile_dir: Path,
) -> dict[str, Any]:
    """Promote only the exact artifacts that passed the single-use holdout."""
    validate_realism_output_path(REALISM_PROFILE, experiment_dir)
    result_path = experiment_dir / "promotion_result.json"
    if result_path.exists():
        raise FileExistsError(f"promotion was already attempted: {result_path}")
    evaluation_path = experiment_dir / "holdout_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    decision = evaluation.get("decision", {})
    if decision.get("promotion_eligible") is not True or decision.get("reasons"):
        raise ValueError("holdout gate did not authorize promotion")
    paired_interval = evaluation["paired_cluster_bootstrap"]["delta_auc_ci_95"]
    if paired_interval is None or float(paired_interval[0]) <= 0.0:
        raise ValueError("clustered paired holdout interval does not authorize promotion")
    candidate_record = evaluation["candidate_artifacts"]["model"]
    pattern_record = evaluation["candidate_artifacts"]["pattern"]
    candidate_path = _verify_record(candidate_record)
    pattern_path = _verify_record(pattern_record)
    model = load_profile_model(candidate_path)
    pattern = PatternProfile.load_json(pattern_path)
    if pattern is None:
        raise ValueError("passing candidate pattern artifact is invalid")
    if (model.selection_summary or {}).get("label_name") != TARGET_NAME:
        raise ValueError("passing candidate does not declare the exact FASTWIN target")
    candidate_result = _walkforward_from_json(evaluation["candidate_result"])
    incumbent_result = _walkforward_from_json(evaluation["incumbent_result"])
    promotion_decision = promote_profile_version(
        model,
        requested_features=list(DEFAULT_FEATURES),
        wf_result=candidate_result,
        incumbent_wf_result=incumbent_result,
        candidate_pattern_profile=pattern,
        profile_dir=profile_dir,
        trial_hyperparameters={
            "canonical_holdout_evaluation": str(evaluation_path.resolve()),
            "canonical_holdout_evaluation_sha256": sha256_file(evaluation_path),
            "clustered_auc_delta": evaluation["paired_cluster_bootstrap"]["delta_auc"],
            "clustered_auc_delta_ci_95": paired_interval,
            "target_name": TARGET_NAME,
        },
    )
    result: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_evaluation": _source_record(evaluation_path),
        "decision": asdict(promotion_decision),
        "production_profile_dir": str(profile_dir.resolve()),
    }
    if promotion_decision.promoted:
        result["current"] = _source_record(profile_dir / "current.json")
    _atomic_json(result_path, result)
    return result
