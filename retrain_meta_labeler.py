#!/usr/bin/env python3
"""
Train the active bootstrap MetaLabeler.

AFML M1: CLI script to train the meta-labeler binary classifier on the unified
snapshot/backtest backbone.

Usage:
    python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --output models/meta_labeler.pkl
    python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --analyze-only

Current active default (FASTWIN v2, 2026-06-08):
    - feature profile: snapshot_v20260530_fastwin
    - selected feature count: 20  (ex-ante ev_score ... funding_rate)
    - label: ENDOGENOUS time-to-target (fast winner == time_to_target_hours <= 7h)
    - data pool: an explicitly supplied finalized fresh full-pool directory
    - training backbone: unified builder -> build_meta_labeler_pool

The active FASTWIN retrainer never selects a static or historical pool by
default. The source must be a newly finalized full-pool backtest run. Historical
exact-replay pools remain available only through --allow-historical-replay for
explicit diagnostics. The reference --input workbook is kept for compatibility
checks and analyze-only reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import pickle
import sys
from typing import Any, Dict, List, Sequence, cast
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Ensure local `src/` package is used when running this script directly.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.models.meta_labeler import (
    MetaLabeler,
    PROMOTION_EVALUATION_CONTRACT,
    MetaLabelerConfig,
    META_FEATURE_PROFILES,
    _FEATURE_MEDIAN_DEFAULTS,
    ACTIVE_META_TARGET_CONTRACT,
    ACTIVE_META_TARGET_DURATION_COLUMN,
    ACTIVE_META_TARGET_DURATION_MAX_HOURS,
    ACTIVE_META_TARGET_LABEL_COLUMN,
    ACTIVE_META_TARGET_PNL_THRESHOLD_PCT,
    resolve_feature_profile,
)
from neutralgrid.training.data_generator import (
    LabelConfig,
    get_missing_features_report,
    DEFAULT_SCHEMA,
    HMM_FEATURE_SEMANTICS_VERSION,
)
from neutralgrid.training.trial_tracker import TrialTracker, TrialRecord
from neutralgrid.training.unified_training_builder import (
    META_POOL_MAX_ROWS_PER_SYMBOL,
)
from neutralgrid.training.meta_pool_contract import (
    load_pool_manifest,
    validate_retrain_pool_source,
)
from neutralgrid.core.config import get_config
from neutralgrid.validation.utility import (
    DEFAULT_UTILITY_ARTIFACT_PATH,
    UtilityCalibratorUnavailable,
    UtilityConfig as ValidationUtilityConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_meta_labeler")

ACTIVE_BOOTSTRAP_META_FEATURE_PROFILE = "snapshot_v20260421_bootstrap"
# FASTWIN-01: active profile is the ex-ante fast-winner set. Unlike the exact
# bootstrap profile it permits median imputation, so it is NOT subject to the
# bootstrap exactness/non-imputed guard in enforce_requested_profile_mode.
ACTIVE_META_FEATURE_PROFILE = "snapshot_v20260530_fastwin"
FAST_TARGET_DURATION_COLUMN = ACTIVE_META_TARGET_DURATION_COLUMN
FAST_TARGET_DURATION_MAX_HOURS = ACTIVE_META_TARGET_DURATION_MAX_HOURS
FAST_TARGET_LABEL_COLUMN = ACTIVE_META_TARGET_LABEL_COLUMN
FAST_TARGET_PNL_THRESHOLD_PCT = ACTIVE_META_TARGET_PNL_THRESHOLD_PCT


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train MetaLabeler on historical bot performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/new_expired_bots.xlsx",
        help=(
            "Reference workbook path used for compatibility checks/analyze-only "
            "reporting. Active training features come from the unified "
            "snapshot/backtest backbone (default: data/new_expired_bots.xlsx)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/meta_labeler.pkl",
        help="Output file path for trained model (default: models/meta_labeler.pkl)",
    )
    parser.add_argument(
        "--include-live-outcomes",
        action="store_true",
        help=(
            "PIPELINE_FIX C/D: union REAL bot outcomes (live + expired, via "
            "LiveOutcomeIngestor) into the fast-winner meta pool. The union is "
            "accepted only when live rows carry the active path-based "
            "time_to_target_hours label; otherwise retraining fails closed. "
            "Scoped to the meta pool only; off by default."
        ),
    )
    parser.add_argument(
        "--hurdle-pct",
        type=float,
        default=3.0,
        help="PnL hurdle for success label in %% (default: 3.0 = 3%%)",
    )
    parser.add_argument(
        "--sl-pct",
        type=float,
        default=-12.0,
        help="Stop loss threshold for failure in %% (default: -12.0 = -12%%)",
    )
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=6.0,
        help="Prediction horizon in hours (default: 6.0)",
    )
    parser.add_argument(
        "--pnl-col",
        type=str,
        default="pnl_pct",
        help="Column name for PnL percentage (default: pnl_pct)",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="start_time_utc",
        help="Column name for timestamp (default: start_time_utc)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data without training",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze data and report feature availability (no training)",
    )
    parser.add_argument(
        "--export-training-data",
        type=str,
        default=None,
        help="Export prepared training data to CSV (for inspection)",
    )
    # Legacy no-op flag kept only to avoid breaking old wrappers while the
    # snapshot-only path remains the sole supported training backbone.
    parser.add_argument("--unified-table", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--training-backbone",
        type=str,
        choices=["unified"],
        default="unified",
        help="Training data backbone (only 'unified' snapshot path is supported).",
    )
    parser.add_argument(
        "--feature-profile",
        type=str,
        choices=sorted(META_FEATURE_PROFILES),
        default=ACTIVE_META_FEATURE_PROFILE,
        help=(
            "Feature profile to train on "
            f"(default: {ACTIVE_META_FEATURE_PROFILE})"
        ),
    )
    parser.add_argument(
        "--estimator",
        type=str,
        choices=["profile", "logistic", "gbm", "vote_logit_hgb"],
        default="profile",
        help=(
            "Base estimator override. 'profile' (default) keeps the historical "
            "profile-derived choice (logistic for the active FASTWIN profile, "
            "gbm otherwise). 'vote_logit_hgb' is the FASTWIN-02 soft-vote of "
            "the FASTWIN-01 logistic and a strongly-regularized "
            "HistGradientBoosting (challenger-validated 2026-07-04: promotion-"
            "OOF AUC 0.807 vs 0.769, paired dAUC CI-low > 0 on 5 fold seeds "
            "and under time-blocked purged folds)."
        ),
    )
    parser.add_argument(
        "--phase1",
        action="store_true",
        default=False,
        help="Apply Phase 1 calibration config overrides (v20260311)",
    )
    parser.add_argument(
        "--allow-imputation",
        action="store_true",
        default=False,
        help=(
            "Fill null selected feature values with explicit feature defaults "
            "before enforcing the retrain contract. Legacy comparison path "
            "only; the active bootstrap profile must remain non-imputed."
        ),
    )
    # Legacy CLI args (no longer used, kept for backward-compat):
    parser.add_argument("--linkage-dir", type=str, default="data/linkage", help=argparse.SUPPRESS)
    parser.add_argument("--scanner-results-dir", type=str, default="results", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backtest-results-dir",
        type=str,
        default="",
        help=(
            "Required finalized pool directory for active FASTWIN retraining. "
            "It must be produced by a fresh unbounded backtest, active-HMM "
            "backfill, and authoritative finalizer; static/shadow/replay-only "
            "directories are rejected by default."
        ),
    )
    parser.add_argument(
        "--allow-historical-replay",
        action="store_true",
        help=(
            "Explicit diagnostic opt-in for a historical exact-replay pool. "
            "Never use this mode for a promoted retrain."
        ),
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="data/training_snapshots",
        help="Directory containing snapshot parquet files (default: data/training_snapshots)",
    )
    parser.add_argument(
        "--max-rows-per-symbol",
        type=_nonnegative_int,
        default=META_POOL_MAX_ROWS_PER_SYMBOL,
        help=(
            "Maximum admitted meta-pool rows per symbol after deterministic "
            "deduplication (default: %(default)s; 0 disables the cap)."
        ),
    )
    parser.add_argument("--min-bot-date", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-tolerance-hours", type=float, default=6.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def enforce_requested_profile_mode(
    *,
    feature_profile: str,
    allow_imputation: bool,
) -> None:
    """Block options that would weaken the active bootstrap contract."""
    if feature_profile == ACTIVE_BOOTSTRAP_META_FEATURE_PROFILE and allow_imputation:
        raise ValueError(
            "The active bootstrap profile "
            f"'{ACTIVE_BOOTSTRAP_META_FEATURE_PROFILE}' must remain exact and "
            "non-imputed. Remove --allow-imputation or use an explicit legacy "
            "comparison profile."
        )


def analyze_and_report(input_path: Path) -> dict:
    """Analyze data and report feature availability."""
    report = get_missing_features_report(input_path)

    logger.info("-" * 70)
    logger.info("FEATURE AVAILABILITY REPORT")
    logger.info("-" * 70)
    logger.info(f"Total samples: {report['total_samples']}")
    logger.info("")

    # Available features
    total_expected = len(report['available_features']) + len(report['missing_features'])
    logger.info(f"AVAILABLE FEATURES ({len(report['available_features'])}/{total_expected}):")
    for feat in report['available_features']:
        logger.info(f"  [OK] {feat}")

    # Partial features
    if report['partial_features']:
        logger.info("")
        logger.info(f"PARTIAL FEATURES ({len(report['partial_features'])}):")
        for item in report['partial_features']:
            logger.info(f"  [~] {item['feature']}: {item['coverage_pct']:.1f}% coverage")

    # Missing features
    if report['missing_features']:
        logger.info("")
        logger.info(f"MISSING FEATURES ({len(report['missing_features'])}/{total_expected}):")
        for feat in report['missing_features']:
            logger.info(f"  [X] {feat}")

    # Positive label rate
    if 'positive_label_rate' in report:
        logger.info("")
        logger.info(f"Positive label rate: {report['positive_label_rate']:.1f}%")

    logger.info("-" * 70)

    return report



def print_data_quality_warnings(metadata: dict) -> None:
    """Print warnings about data quality issues."""
    availability = metadata.get("feature_availability", {})

    # Check for critical missing features
    critical_missing = []
    if "range_prob" in availability.get("missing_features", []):
        critical_missing.append("range_prob (HMM regime probability)")
    if "trend_prob" in availability.get("missing_features", []):
        critical_missing.append("trend_prob (HMM trend probability)")
    if "survival_prob" in availability.get("missing_features", []):
        critical_missing.append("survival_prob (stochastic survival)")

    if critical_missing:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("DATA QUALITY WARNING: Critical features are missing!")
        logger.warning("=" * 70)
        for feat in critical_missing:
            logger.warning(f"  - {feat}")
        logger.warning("")
        logger.warning("These features require historical OHLCV data at decision time.")
        logger.warning("The model will be trained on available features only.")
        logger.warning("For better performance, consider:")
        logger.warning("  1. Logging complete feature snapshots at scan time (see FeatureSnapshotLogger)")
        logger.warning("  2. Rebuilding incomplete snapshot rows from current scan-time sources")
        logger.warning("=" * 70)
        logger.warning("")


def _verification_log_path(output_path: Path) -> Path:
    """Return the simple JSON verification log path for a retrain run."""
    return output_path.parent / "meta_labeler_verification.json"


def inspect_deployed_artifact_health(
    output_path: Path,
    expected_features: List[str],
) -> Dict[str, Any]:
    """Inspect the currently deployed artifact directory and legacy pickle.

    The live loader prefers ``output_path.parent / "meta_labeler"`` over the
    legacy pickle. A deploy is therefore only healthy when both the artifact
    directory and the pickle exist and both advertise the exact expected
    feature list in the same order.
    """
    artifact_dir = output_path.parent / "meta_labeler"
    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.joblib"

    artifact_features: List[str] = []
    pickle_features: List[str] = []
    artifact_error: str | None = None
    pickle_error: str | None = None

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            artifact_features = list(metadata.get("features") or [])
        except Exception as exc:
            artifact_error = str(exc)
    else:
        artifact_error = "missing_metadata.json"

    if output_path.exists():
        try:
            with open(output_path, "rb") as fh:
                state = pickle.load(fh)
            pickle_features = list(state.get("feature_names") or [])
        except Exception as exc:
            pickle_error = str(exc)
    else:
        pickle_error = "missing_pickle"

    artifact_matches_expected = (
        model_path.exists()
        and artifact_features == expected_features
    )
    pickle_matches_expected = pickle_features == expected_features

    return {
        "output_path": str(output_path),
        "artifact_dir": str(artifact_dir),
        "artifact_dir_exists": artifact_dir.exists(),
        "artifact_model_exists": model_path.exists(),
        "artifact_metadata_exists": metadata_path.exists(),
        "artifact_feature_count": len(artifact_features),
        "artifact_features": artifact_features,
        "artifact_matches_expected": artifact_matches_expected,
        "artifact_error": artifact_error,
        "pickle_exists": output_path.exists(),
        "pickle_feature_count": len(pickle_features),
        "pickle_features": pickle_features,
        "pickle_matches_expected": pickle_matches_expected,
        "pickle_error": pickle_error,
        "expected_feature_count": len(expected_features),
        "expected_features": list(expected_features),
        "is_stale": not (artifact_matches_expected and pickle_matches_expected),
    }


def evaluate_champion_challenger_gate(
    output_path: Path,
    expected_features: List[str],
    candidate_metrics: Any,
) -> Dict[str, Any]:
    """Fail closed before a retrain can replace the deployed champion.

    ERR-088: the model's absolute promotion gate is necessary but was not
    sufficient because the retrain entry point saved every trained model,
    including a candidate with ``promotion_status='fail'`` or a candidate that
    regressed relative to the deployed champion.  This gate compares the
    governed stored OOF summaries produced by the identical retrain path.

    The comparison is intentionally labelled ``stored_summary_nonpaired``.  It
    prevents an accidental overwrite by an obviously worse routine retrain; it
    does not claim the statistical strength of the separate paired temporal
    challenger protocol used for a new model recipe.
    """
    metadata_path = output_path.parent / "meta_labeler" / "metadata.json"
    candidate = {
        "promotion_status": getattr(candidate_metrics, "promotion_status", None),
        "oof_auc": getattr(candidate_metrics, "oof_auc", None),
        "oof_auc_ci_low": getattr(candidate_metrics, "oof_auc_ci_low", None),
        "oof_auc_ci_high": getattr(candidate_metrics, "oof_auc_ci_high", None),
        "oof_ece": getattr(candidate_metrics, "oof_ece", None),
        "oof_deployable_auc": getattr(
            candidate_metrics, "oof_deployable_auc", None
        ),
        "oof_deployable_ece": getattr(
            candidate_metrics, "oof_deployable_ece", None
        ),
        "train_samples": getattr(candidate_metrics, "train_samples", None),
        "promotion_evaluation_contract": PROMOTION_EVALUATION_CONTRACT,
        "hmm_artifact_version": _get_active_hmm_artifact_version(),
    }
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_kind": "stored_summary_nonpaired",
        "output_path": str(output_path),
        "champion_metadata_path": str(metadata_path),
        "candidate": candidate,
        "champion": None,
        "status": "fail",
        "reasons": [],
    }
    reasons = cast(List[str], report["reasons"])

    if candidate["promotion_status"] != "pass":
        reasons.append(
            "candidate_absolute_promotion_gate_not_pass"
            f"(status={candidate['promotion_status']!r})"
        )

    # A genuinely new deployment has no champion to regress against.  The
    # absolute gate above remains mandatory.  A partial/corrupt deployment is
    # not treated as new: if either side exists without metadata, fail closed.
    artifact_dir = metadata_path.parent
    deployed_state_exists = output_path.exists() or artifact_dir.exists()
    if not metadata_path.exists():
        if deployed_state_exists:
            reasons.append("champion_metadata_missing_for_existing_deployment")
        report["comparison_scope"] = "initial_deployment"
        report["status"] = "pass" if not reasons else "fail"
        return report

    try:
        champion_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"champion_metadata_unreadable:{type(exc).__name__}")
        return report

    champion_eval = champion_metadata.get("eval_metrics")
    if not isinstance(champion_eval, dict):
        reasons.append("champion_eval_metrics_missing")
        return report
    champion = {
        "artifact_version": champion_metadata.get("artifact_version"),
        "features": champion_metadata.get("features"),
        "target_contract": champion_eval.get("target_contract"),
        "promotion_status": champion_eval.get("promotion_status"),
        "oof_auc": champion_eval.get("oof_auc"),
        "oof_auc_ci_low": champion_eval.get("oof_auc_ci_low"),
        "oof_auc_ci_high": champion_eval.get("oof_auc_ci_high"),
        "oof_ece": champion_eval.get("oof_ece"),
        "oof_deployable_auc": champion_eval.get("oof_deployable_auc"),
        "oof_deployable_ece": champion_eval.get("oof_deployable_ece"),
        "total_samples": champion_metadata.get("total_samples"),
        "promotion_evaluation_contract": champion_eval.get(
            "promotion_evaluation_contract"
        ),
        "hmm_artifact_version": (
            champion_metadata.get("lineage", {}).get("hmm_artifact_version")
            if isinstance(champion_metadata.get("lineage"), dict)
            else None
        ),
    }
    report["champion"] = champion
    report["comparison_scope"] = "existing_champion"

    incomparable_reasons: List[str] = []
    if (
        champion["promotion_evaluation_contract"]
        != candidate["promotion_evaluation_contract"]
    ):
        incomparable_reasons.append(
            "promotion_evaluation_contract_mismatch"
            f"(champion={champion['promotion_evaluation_contract']!r},"
            f"candidate={candidate['promotion_evaluation_contract']!r})"
        )
    if champion["hmm_artifact_version"] != candidate["hmm_artifact_version"]:
        incomparable_reasons.append(
            "hmm_lineage_mismatch"
            f"(champion={champion['hmm_artifact_version']!r},"
            f"candidate={candidate['hmm_artifact_version']!r})"
        )
    if incomparable_reasons:
        report["comparison_scope"] = "incomparable_champion_absolute_gate_only"
        report["comparison_skipped_reasons"] = incomparable_reasons
        report["status"] = "pass" if not reasons else "fail"
        return report

    if champion["promotion_status"] != "pass":
        reasons.append(
            "champion_promotion_status_not_pass"
            f"(status={champion['promotion_status']!r})"
        )
    if champion["target_contract"] != ACTIVE_META_TARGET_CONTRACT:
        reasons.append(
            "target_contract_mismatch"
            f"(champion={champion['target_contract']!r},"
            f"candidate={ACTIVE_META_TARGET_CONTRACT!r})"
        )
    if list(champion["features"] or []) != list(expected_features):
        reasons.append("feature_contract_mismatch")

    tolerance = 1e-12
    comparisons = (
        ("oof_auc", "gte"),
        ("oof_auc_ci_low", "gte"),
        ("oof_ece", "lte"),
    )
    for metric_name, direction in comparisons:
        candidate_value = candidate.get(metric_name)
        champion_value = champion.get(metric_name)
        if not isinstance(candidate_value, (int, float)) or not math.isfinite(
            float(candidate_value)
        ):
            reasons.append(f"candidate_{metric_name}_unavailable")
            continue
        if not isinstance(champion_value, (int, float)) or not math.isfinite(
            float(champion_value)
        ):
            reasons.append(f"champion_{metric_name}_unavailable")
            continue
        candidate_float = float(candidate_value)
        champion_float = float(champion_value)
        regressed = (
            candidate_float + tolerance < champion_float
            if direction == "gte"
            else candidate_float - tolerance > champion_float
        )
        if regressed:
            reasons.append(
                f"candidate_{metric_name}_regressed"
                f"(candidate={candidate_float:.12g},champion={champion_float:.12g})"
            )

    # Deployable-stratum diagnostics did not exist on pre-METAGOV champions.
    # Once a champion records them, however, a challenger must provide and
    # preserve them.  This keeps the migration backward compatible without
    # allowing the newly governed evidence to disappear on later retrains.
    deployable_comparisons = (
        ("oof_deployable_auc", "gte"),
        ("oof_deployable_ece", "lte"),
    )
    for metric_name, direction in deployable_comparisons:
        champion_value = champion.get(metric_name)
        if not isinstance(champion_value, (int, float)) or not math.isfinite(
            float(champion_value)
        ):
            continue
        candidate_value = candidate.get(metric_name)
        if not isinstance(candidate_value, (int, float)) or not math.isfinite(
            float(candidate_value)
        ):
            reasons.append(f"candidate_{metric_name}_unavailable")
            continue
        candidate_float = float(candidate_value)
        champion_float = float(champion_value)
        regressed = (
            candidate_float + tolerance < champion_float
            if direction == "gte"
            else candidate_float - tolerance > champion_float
        )
        if regressed:
            reasons.append(
                f"candidate_{metric_name}_regressed"
                f"(candidate={candidate_float:.12g},champion={champion_float:.12g})"
            )

    report["status"] = "pass" if not reasons else "fail"
    return report


def write_champion_challenger_decision(
    report: Dict[str, Any], output_path: Path
) -> Path:
    """Atomically persist the latest pre-save promotion decision."""
    decision_path = output_path.parent / "meta_labeler_promotion_decision.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = decision_path.with_name(f"{decision_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        temporary_path.replace(decision_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return decision_path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_range_utc(df: pd.DataFrame, column: str) -> Dict[str, str | None]:
    if column not in df.columns:
        return {"min": None, "max": None}
    values = pd.to_datetime(df[column], utc=True, errors="coerce", format="mixed")
    finite = cast(pd.Series, values).dropna()
    if finite.empty:
        return {"min": None, "max": None}
    return {
        "min": cast(pd.Timestamp, finite.min()).isoformat(),
        "max": cast(pd.Timestamp, finite.max()).isoformat(),
    }


def build_training_source_manifest(
    backtest_results_dir: Path,
    training_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Describe and hash the exact file set eligible for the unified builder."""
    patterns = (
        "training_data_*.csv",
        "training_rows_*.csv",
        "backtest_training_*.csv",
        "backtest_results_*.csv",
    )
    source_paths = sorted(
        {path for pattern in patterns for path in backtest_results_dir.glob(pattern)}
    )
    source_files: List[Dict[str, Any]] = []
    for path in source_paths:
        stat = path.stat()
        source_files.append(
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "modified_at_utc": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(),
                "sha256": _sha256_path(path),
            }
        )

    candidate_ids = pd.Series(dtype="string")
    if "candidate_id" in training_df.columns:
        candidate_ids = (
            cast(pd.Series, training_df["candidate_id"])
            .astype("string")
            .str.strip()
        )
        candidate_ids = candidate_ids[
            candidate_ids.notna()
            & candidate_ids.ne("")
            & candidate_ids.str.lower().ne("nan")
        ]

    try:
        pool_manifest = load_pool_manifest(backtest_results_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        pool_manifest = None

    return {
        "backtest_results_dir": str(backtest_results_dir),
        "source_file_count": len(source_files),
        "source_files": source_files,
        "source_training_rows": int(len(training_df)),
        "unique_candidate_ids": int(candidate_ids.nunique()),
        "duplicate_candidate_id_rows": int(
            candidate_ids.duplicated(keep=False).sum()
        ),
        "market_event_time_utc": _timestamp_range_utc(
            training_df, "start_time_utc"
        ),
        "backtest_write_time_utc": _timestamp_range_utc(
            training_df, "backtest_timestamp"
        ),
        "pool_generation_mode": (
            pool_manifest.get("generation_mode") if pool_manifest else None
        ),
        "pool_manifest_path": (
            pool_manifest.get("_manifest_path") if pool_manifest else None
        ),
    }


HMM_DERIVED_META_FEATURES = frozenset(
    {
        "ev_score",
        "hmm_tail_cvar_95",
        "persistence_prob",
        "range_prob",
        "regime_conf",
        "trend_prob",
    }
)


def audit_training_hmm_lineage(
    training_df: pd.DataFrame,
    *,
    selected_features: Sequence[str],
    active_hmm_artifact_version: str | None,
    active_hmm_trained_at_utc: str | None = None,
) -> Dict[str, Any]:
    """Fail closed unless every HMM-derived row has direct active provenance."""
    dependent_features = sorted(
        set(selected_features).intersection(HMM_DERIVED_META_FEATURES)
    )
    report: Dict[str, Any] = {
        "required": bool(dependent_features),
        "dependent_selected_features": dependent_features,
        "active_hmm_artifact_version": active_hmm_artifact_version,
        "training_row_count": int(len(training_df)),
        "lineage_column_present": "hmm_artifact_version" in training_df.columns,
        "missing_lineage_count": 0,
        "mismatched_lineage_count": 0,
        "observed_versions": {},
        "active_hmm_trained_at_utc": active_hmm_trained_at_utc,
        "observed_hmm_trained_at_utc": {},
        "passes": False,
        "reasons": [],
    }
    reasons = cast(List[str], report["reasons"])
    if not dependent_features:
        report["passes"] = True
        return report
    if active_hmm_artifact_version is None:
        reasons.append("active_hmm_artifact_version_unavailable")
    if "hmm_artifact_version" not in training_df.columns:
        report["missing_lineage_count"] = int(len(training_df))
        reasons.append("hmm_artifact_version_column_missing")
        return report

    versions = (
        cast(pd.Series, training_df["hmm_artifact_version"])
        .astype("string")
        .str.strip()
    )
    missing = (
        versions.isna()
        | versions.eq("")
        | versions.str.lower().isin({"nan", "none", "nat"})
    )
    report["missing_lineage_count"] = int(missing.sum())
    observed = cast(pd.Series, versions.loc[~missing]).value_counts(dropna=False)
    report["observed_versions"] = {
        str(version): int(count) for version, count in observed.items()
    }
    if bool(missing.any()):
        reasons.append("hmm_artifact_version_missing_on_training_rows")
    if active_hmm_artifact_version is not None:
        mismatched = cast(pd.Series, versions.loc[~missing]).ne(
            active_hmm_artifact_version
        )
        report["mismatched_lineage_count"] = int(mismatched.sum())
        if bool(mismatched.any()):
            reasons.append("training_hmm_lineage_does_not_match_active_hmm")

    required_text_columns = (
        "hmm_trained_at_utc",
        "hmm_feature_semantics_version",
        "hmm_feature_source",
        "feature_cutoff_utc",
    )
    for column in required_text_columns:
        if column not in training_df.columns:
            reasons.append(f"{column}_column_missing")

    if "hmm_trained_at_utc" in training_df.columns:
        trained_raw = (
            cast(pd.Series, training_df["hmm_trained_at_utc"])
            .astype("string")
            .str.strip()
        )
        trained_missing = (
            trained_raw.isna()
            | trained_raw.eq("")
            | trained_raw.str.lower().isin({"nan", "none", "nat"})
        )
        if bool(trained_missing.any()):
            reasons.append("hmm_trained_at_utc_missing_on_training_rows")
        trained_parsed = pd.to_datetime(
            trained_raw, utc=True, errors="coerce", format="mixed"
        )
        trained_invalid = (~trained_missing) & trained_parsed.isna()
        if bool(trained_invalid.any()):
            reasons.append("hmm_trained_at_utc_invalid_on_training_rows")
        observed_trained = cast(
            pd.Series, trained_parsed.loc[trained_parsed.notna()]
        ).value_counts()
        report["observed_hmm_trained_at_utc"] = {
            cast(pd.Timestamp, timestamp).isoformat(): int(count)
            for timestamp, count in observed_trained.items()
        }
        expected_trained_series = pd.to_datetime(
            pd.Series([active_hmm_trained_at_utc]),
            utc=True,
            errors="coerce",
        )
        expected_trained = expected_trained_series.iloc[0]
        if active_hmm_trained_at_utc is None or pd.isna(expected_trained):
            reasons.append("active_hmm_trained_at_utc_unavailable")
        else:
            trained_mismatch = trained_parsed.notna() & trained_parsed.ne(
                expected_trained
            )
            if bool(trained_mismatch.any()):
                reasons.append("training_hmm_trained_at_does_not_match_active_hmm")

    if "hmm_feature_semantics_version" in training_df.columns:
        semantics = (
            cast(pd.Series, training_df["hmm_feature_semantics_version"])
            .astype("string")
            .str.strip()
        )
        if not bool(semantics.eq(HMM_FEATURE_SEMANTICS_VERSION).all()):
            reasons.append("training_hmm_feature_semantics_version_invalid")

    if "hmm_feature_source" in training_df.columns:
        feature_source = (
            cast(pd.Series, training_df["hmm_feature_source"])
            .astype("string")
            .str.strip()
        )
        source_missing = (
            feature_source.isna()
            | feature_source.eq("")
            | feature_source.str.lower().isin({"nan", "none", "nat"})
        )
        if bool(source_missing.any()):
            reasons.append("hmm_feature_source_missing_on_training_rows")

    if "feature_cutoff_utc" in training_df.columns:
        cutoff = pd.to_datetime(
            training_df["feature_cutoff_utc"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        if bool(cutoff.isna().any()):
            reasons.append("feature_cutoff_utc_invalid_on_training_rows")
        if "start_time_utc" not in training_df.columns:
            reasons.append("start_time_utc_column_missing_for_causality_check")
        else:
            event_start = pd.to_datetime(
                training_df["start_time_utc"],
                utc=True,
                errors="coerce",
                format="mixed",
            )
            if bool(event_start.isna().any()):
                reasons.append("start_time_utc_invalid_on_training_rows")
            noncausal = cutoff.notna() & event_start.notna() & cutoff.gt(event_start)
            if bool(noncausal.any()):
                reasons.append("feature_cutoff_utc_after_event_start")

    probability_invalid = False
    probability_out_of_range = False
    for column in ("range_prob", "trend_prob", "persistence_prob"):
        if column not in training_df.columns:
            probability_invalid = True
            continue
        numeric = cast(
            pd.Series,
            pd.to_numeric(cast(pd.Series, training_df[column]), errors="coerce"),
        )
        finite = pd.Series(
            np.isfinite(np.asarray(numeric, dtype=float)),
            index=training_df.index,
            dtype=bool,
        )
        if not bool(finite.all()):
            probability_invalid = True
        in_range = finite & numeric.between(0.0, 1.0, inclusive="both")
        if bool((finite & ~in_range).any()):
            probability_out_of_range = True
    if probability_invalid:
        reasons.append("training_hmm_probability_missing_or_nonfinite")
    if probability_out_of_range:
        reasons.append("training_hmm_probability_out_of_range")

    report["reasons"] = list(dict.fromkeys(reasons))
    report["passes"] = not report["reasons"]
    return report


def build_feature_verification_report(
    training_df: pd.DataFrame,
    *,
    feature_profile: str,
    selected_features: List[str],
    output_path: Path,
    artifact_health_before: Dict[str, Any],
    artifact_health_after: Dict[str, Any] | None = None,
    source_training_df: pd.DataFrame | None = None,
    training_target_summary: Dict[str, Any] | None = None,
    modelable_feature_filter: Dict[str, Any] | None = None,
    training_source_manifest: Dict[str, Any] | None = None,
    training_hmm_lineage: Dict[str, Any] | None = None,
    allow_imputation: bool = False,
    imputed_features: Dict[str, Dict[str, float | int]] | None = None,
) -> Dict[str, Any]:
    """Create a simple verification report for the active feature contract."""
    expected_features = list(META_FEATURE_PROFILES[feature_profile])
    source_df = source_training_df if source_training_df is not None else training_df
    # Per UTILFIX-01, from_artifact() raises when current.json is missing.
    # The verification report records the actual provenance: a calibrated
    # artifact, or "calibrator_unavailable" with None coefficients. There is
    # no longer a "pinned_v0_fallback" path at runtime.
    utility_config: ValidationUtilityConfig | None
    try:
        utility_config = ValidationUtilityConfig.from_artifact()
        utility_source = "current_artifact"
    except UtilityCalibratorUnavailable:
        utility_config = None
        utility_source = "calibrator_unavailable"
    active_hmm_artifact_version = _get_active_hmm_artifact_version()
    artifact_lineage_before = _inspect_meta_artifact_lineage(
        output_path,
        active_hmm_artifact_version,
    )
    artifact_lineage_after = (
        _inspect_meta_artifact_lineage(output_path, active_hmm_artifact_version)
        if artifact_health_after is not None
        else None
    )
    artifact_loadability_before = _inspect_meta_artifact_loadability(output_path)
    artifact_loadability_after = (
        _inspect_meta_artifact_loadability(output_path)
        if artifact_health_after is not None
        else None
    )
    fast_target_contract = _summarize_fast_target_population(source_df)
    if training_target_summary is None:
        training_target_summary = _summarize_fast_target_population(training_df)
    feature_fill: List[Dict[str, Any]] = []
    incomplete_features: List[Dict[str, Any]] = []

    for feat in expected_features:
        non_null = int(cast(pd.Series, training_df[feat]).notna().sum()) if feat in training_df.columns else 0
        null_count = int(len(training_df) - non_null)
        item = {
            "feature": feat,
            "non_null_count": non_null,
            "null_count": null_count,
        }
        feature_fill.append(item)
        if null_count > 0:
            incomplete_features.append(item)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_profile": feature_profile,
        "expected_feature_count": len(expected_features),
        "selected_feature_count": len(selected_features),
        "selected_features": list(selected_features),
        "source_training_row_count": int(len(source_df)),
        "training_row_count": int(len(training_df)),
        "training_source_manifest": training_source_manifest or {},
        "training_hmm_lineage": training_hmm_lineage or {},
        "allow_imputation": bool(allow_imputation),
        "imputed_features": imputed_features or {},
        "fast_target_contract": fast_target_contract,
        "active_training_target": training_target_summary,
        "modelable_feature_filter": modelable_feature_filter or {
            "input_rows": int(len(training_df)),
            "modelable_rows": int(len(training_df)),
            "excluded_unmodelable_count": 0,
            "excluded_unmodelable_rows": [],
        },
        "meta_target_contract": {
            "contract_name": ACTIVE_META_TARGET_CONTRACT,
            "target_column": FAST_TARGET_LABEL_COLUMN,
            "duration_column": FAST_TARGET_DURATION_COLUMN,
            "duration_hours_max": FAST_TARGET_DURATION_MAX_HOURS,
        },
        "utility_provenance": {
            "source": utility_source,
            "artifact_path": str(DEFAULT_UTILITY_ARTIFACT_PATH),
            "lambda_risk": (
                float(utility_config.lambda_risk) if utility_config is not None else None
            ),
            "kappa_trend": (
                float(utility_config.kappa_trend) if utility_config is not None else None
            ),
            "horizon_hours": (
                float(utility_config.horizon_hours) if utility_config is not None else None
            ),
        },
        "is_selected_feature_count_exact": len(selected_features) == len(expected_features),
        "is_value_complete": len(incomplete_features) == 0,
        "incomplete_features": incomplete_features,
        "feature_fill": feature_fill,
        "active_hmm_artifact_version": active_hmm_artifact_version,
        "artifact_lineage_before": artifact_lineage_before,
        "artifact_lineage_after": artifact_lineage_after,
        "artifact_loadability_before": artifact_loadability_before,
        "artifact_loadability_after": artifact_loadability_after,
        "artifact_health_before": artifact_health_before,
        "artifact_health_after": artifact_health_after,
        "verification_log_path": str(_verification_log_path(output_path)),
    }


def _infer_effective_pnl_col(training_df: pd.DataFrame) -> str | None:
    for candidate in ("net_pnl_pct", "pnl_pct"):
        if candidate in training_df.columns and bool(cast(pd.Series, training_df[candidate]).notna().any()):
            return candidate
    return None


def _compute_fast_target_population(
    training_df: pd.DataFrame,
    *,
    pnl_col: str | None = None,
) -> tuple[Dict[str, Any], pd.Series | None, pd.Series | None]:
    effective_pnl_col = pnl_col or _infer_effective_pnl_col(training_df)
    summary: Dict[str, Any] = {
        "duration_column": (
            FAST_TARGET_DURATION_COLUMN
            if FAST_TARGET_DURATION_COLUMN in training_df.columns
            else None
        ),
        "target_contract": ACTIVE_META_TARGET_CONTRACT,
        "duration_hours_max": FAST_TARGET_DURATION_MAX_HOURS,
        "pnl_column": effective_pnl_col,
        "target_column": FAST_TARGET_LABEL_COLUMN,
        "eligible_rows": 0,
        "positive_count": 0,
        "negative_count": 0,
        "negative_rows_duration_gt_7h_count": 0,
        "excluded_rows_missing_duration_count": 0,
        "excluded_rows_missing_pnl_count": 0,
        "builder_y_compared_count": 0,
        "builder_y_mismatch_count": None,
        "label_basis": None,
    }

    if FAST_TARGET_DURATION_COLUMN not in training_df.columns or effective_pnl_col is None:
        return summary, None, None

    duration = cast(
        pd.Series,
        pd.to_numeric(training_df[FAST_TARGET_DURATION_COLUMN], errors="coerce"),
    )
    pnl = cast(pd.Series, pd.to_numeric(training_df[effective_pnl_col], errors="coerce"))
    duration_missing_mask = duration.isna()
    pnl_missing_mask = pnl.isna()
    # PIPELINE_FIX v2: prefer the ENDOGENOUS time-to-target (first bar the net MTM
    # PnL curve reaches +3%, from the backtest engine) as the "fast" measure when it
    # is present. It varies across the 7h boundary (a long observation window can
    # observe fast/slow/never), so the fast-winner label is non-degenerate and
    # learnable (verified symbol-held-out AUC ~0.73). Legacy rows that predate the
    # time-to-target instrumentation fall back to the B.1 duration-window AND-label
    # (duration_hours <= 7h AND pnl >= 3), which is a no-op on the constant-6h pool.
    _t2t_raw = (
        pd.to_numeric(training_df["time_to_target_hours"], errors="coerce")
        if "time_to_target_hours" in training_df.columns
        else None
    )
    use_endogenous = _t2t_raw is not None and bool(cast(pd.Series, _t2t_raw).notna().any())
    if use_endogenous:
        t2t = cast(pd.Series, _t2t_raw)
        # A row is labelable iff the backtest produced an outcome (pnl present).
        # fast == reached the +3% target within the cap (t2t <= 7h); NaN t2t means
        # the target was never reached -> NOT fast (a valid negative).
        labelable_mask = pnl.notna()
        is_fast = cast(pd.Series, t2t.le(FAST_TARGET_DURATION_MAX_HOURS)).fillna(False)
        fast_target = is_fast.astype(int)
        positive_mask = labelable_mask & is_fast
        negative_mask = labelable_mask & ~is_fast
        # "slow" negative = target reached but AFTER the cap.
        slow_negative_mask = labelable_mask & cast(pd.Series, t2t.gt(FAST_TARGET_DURATION_MAX_HOURS)).fillna(False)
        summary["label_basis"] = "endogenous_time_to_target"
    else:
        # Legacy AND-label (B.1): duration window length <= 7h AND pnl >= 3.
        labelable_mask = duration.notna() & pnl.notna()
        is_fast = duration.le(FAST_TARGET_DURATION_MAX_HOURS)
        is_profit = pnl.ge(FAST_TARGET_PNL_THRESHOLD_PCT)
        fast_target = (is_fast & is_profit).astype(int)
        positive_mask = labelable_mask & is_fast & is_profit
        negative_mask = labelable_mask & ~(is_fast & is_profit)
        slow_negative_mask = labelable_mask & duration.gt(FAST_TARGET_DURATION_MAX_HOURS)
        summary["label_basis"] = "duration_window_and_pnl"

    summary["eligible_rows"] = int(labelable_mask.sum())
    summary["positive_count"] = int(positive_mask.sum())
    summary["negative_count"] = int(negative_mask.sum())
    # Renamed from excluded_rows_duration_gt_7h_count: >7h rows are AUDITED
    # NEGATIVES now, not exclusions.
    summary["negative_rows_duration_gt_7h_count"] = int(slow_negative_mask.sum())
    summary["excluded_rows_missing_duration_count"] = int(duration_missing_mask.sum())
    summary["excluded_rows_missing_pnl_count"] = int((pnl_missing_mask & duration.notna()).sum())

    if "y" in training_df.columns:
        builder_y = cast(pd.Series, pd.to_numeric(training_df["y"], errors="coerce"))
        compare_mask = labelable_mask & builder_y.notna()
        builder_y_int = builder_y.fillna(0).astype(int)
        mismatch_mask = compare_mask & builder_y_int.ne(fast_target)
        summary["builder_y_compared_count"] = int(compare_mask.sum())
        summary["builder_y_mismatch_count"] = int(mismatch_mask.sum())

    return summary, labelable_mask, fast_target


def _summarize_fast_target_population(
    training_df: pd.DataFrame,
    *,
    pnl_col: str | None = None,
) -> Dict[str, Any]:
    summary, _, _ = _compute_fast_target_population(training_df, pnl_col=pnl_col)
    return summary


def prepare_fast_target_training_frame(
    training_df: pd.DataFrame,
    *,
    pnl_col: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    summary, labelable_mask, fast_target = _compute_fast_target_population(
        training_df,
        pnl_col=pnl_col,
    )
    if "source" in training_df.columns:
        source = cast(pd.Series, training_df["source"]).astype("string").str.strip().str.lower()
        live_mask = source.eq("live")
        if "time_to_target_hours" in training_df.columns:
            time_to_target = cast(
                pd.Series,
                pd.to_numeric(training_df["time_to_target_hours"], errors="coerce"),
            )
        else:
            time_to_target = pd.Series(float("nan"), index=training_df.index, dtype=float)
        incompatible_live_mask = live_mask & time_to_target.isna()
        incompatible_live_count = int(incompatible_live_mask.sum())
        if incompatible_live_count:
            raise ValueError(
                "Fast-target retraining refuses "
                f"{incompatible_live_count} live outcome row(s) missing "
                "'time_to_target_hours' while the active target contract requires the "
                "endogenous time-to-target label. Treating those nulls as 'never reached' would "
                "silently label the live rows as negatives. Capture the same path-based "
                "target for live outcomes or retrain without --include-live-outcomes."
            )
    if FAST_TARGET_DURATION_COLUMN not in training_df.columns:
        raise ValueError(
            f"Fast-target retraining requires '{FAST_TARGET_DURATION_COLUMN}' in the unified training table."
        )
    if pnl_col not in training_df.columns:
        raise ValueError(
            f"Fast-target retraining requires pnl column '{pnl_col}' in the unified training table."
        )
    if labelable_mask is None or fast_target is None or summary["eligible_rows"] <= 0:
        raise ValueError(
            "Fast-target retraining found no labelable rows with non-null "
            f"'{FAST_TARGET_DURATION_COLUMN}' and '{pnl_col}'."
        )

    # Train on ALL labelable rows (slow/never bots kept as negatives), NOT a
    # <=7h filter. The fast-winner target column carries the active label, whose
    # basis is reported in summary["label_basis"]: the ENDOGENOUS time-to-target
    # (time_to_target_hours <= 7h) when that column is present, else the legacy
    # duration-window AND-label (duration <= 7h AND pnl >= 3). See
    # _compute_fast_target_population.
    filtered_df = cast(pd.DataFrame, training_df.loc[labelable_mask].copy())
    filtered_df[FAST_TARGET_LABEL_COLUMN] = cast(pd.Series, fast_target.loc[labelable_mask]).astype(int)

    if summary["positive_count"] <= 0 or summary["negative_count"] <= 0:
        raise ValueError(
            "Fast-target retraining requires both classes (fast-winner positives and "
            f"negatives). Found positives={summary['positive_count']}, "
            f"negatives={summary['negative_count']}."
        )

    active_summary = dict(summary)
    active_summary["trained_row_count"] = int(len(filtered_df))
    return filtered_df, active_summary


def _get_active_hmm_artifact_version() -> str | None:
    try:
        from neutralgrid.models.artifacts import get_active_hmm_version

        active_version = get_active_hmm_version()
        return str(active_version) if active_version is not None else None
    except Exception:
        return None


def _get_active_hmm_trained_at_utc(
    active_hmm_artifact_version: str | None,
) -> str | None:
    """Resolve the exact trained timestamp from the active HMM metadata."""
    if not active_hmm_artifact_version:
        return None
    try:
        root = Path(__file__).resolve().parent
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        hmm_manifest = manifest.get("hmm")
        if not isinstance(hmm_manifest, dict):
            return None
        if str(hmm_manifest.get("active_version", "")) != active_hmm_artifact_version:
            return None
        artifact_dir_value = hmm_manifest.get("artifact_dir")
        if not artifact_dir_value:
            return None
        metadata = json.loads(
            (root / str(artifact_dir_value) / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        trained_at = metadata.get("trained_at_utc")
        return str(trained_at).strip() if trained_at else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _inspect_meta_artifact_lineage(
    output_path: Path,
    active_hmm_artifact_version: str | None,
) -> Dict[str, Any]:
    metadata_path = output_path.parent / "meta_labeler" / "metadata.json"
    summary: Dict[str, Any] = {
        "metadata_path": str(metadata_path),
        "active_hmm_artifact_version": active_hmm_artifact_version,
        "artifact_hmm_artifact_version": None,
        "lineage_matches_active_hmm": False,
        "error": None,
    }
    if not metadata_path.exists():
        summary["error"] = "metadata_missing"
        return summary

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        lineage = metadata.get("lineage")
        lineage = lineage if isinstance(lineage, dict) else {}
        artifact_hmm_artifact_version = lineage.get("hmm_artifact_version")
        if artifact_hmm_artifact_version is not None:
            summary["artifact_hmm_artifact_version"] = str(artifact_hmm_artifact_version)
        summary["lineage_matches_active_hmm"] = bool(
            active_hmm_artifact_version is not None
            and artifact_hmm_artifact_version is not None
            and str(artifact_hmm_artifact_version) == active_hmm_artifact_version
        )
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _inspect_meta_artifact_loadability(output_path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "output_path": str(output_path),
        "can_load": False,
        "error": None,
    }
    if not output_path.exists():
        summary["error"] = "model_missing"
        return summary

    try:
        MetaLabeler.load(output_path)
        summary["can_load"] = True
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def write_feature_verification_report(
    report: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Atomically persist the retrain verification report as simple JSON."""
    verification_path = _verification_log_path(output_path)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = verification_path.with_name(f"{verification_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        temporary_path.replace(verification_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    logger.info("Feature verification log written to: %s", verification_path)
    return verification_path


def enforce_feature_contract(
    training_df: pd.DataFrame,
    *,
    feature_profile: str,
    selected_features: List[str],
) -> None:
    """Fail fast when the requested feature profile silently degrades."""
    expected_features = list(META_FEATURE_PROFILES[feature_profile])

    if selected_features != expected_features:
        missing = [feat for feat in expected_features if feat not in selected_features]
        unexpected = [feat for feat in selected_features if feat not in expected_features]
        raise ValueError(
            "Feature selection contract BLOCKED retrain: "
            f"profile '{feature_profile}' expects {len(expected_features)} feature(s), "
            f"but resolve_feature_profile selected {len(selected_features)}. "
            f"Missing from selected set: {missing}. Unexpected extras: {unexpected}."
        )

    incomplete: List[str] = []
    for feat in expected_features:
        null_count = int(cast(pd.Series, training_df[feat]).isna().sum()) if feat in training_df.columns else int(len(training_df))
        if null_count > 0:
            incomplete.append(f"{feat} ({null_count}/{len(training_df)} null)")

    if incomplete:
        raise ValueError(
            "Feature value completeness gate BLOCKED retrain: "
            f"{len(incomplete)} selected feature(s) are not fully populated. "
            f"Blocking features: {', '.join(incomplete)}."
        )


def impute_selected_feature_nulls(
    training_df: pd.DataFrame,
    *,
    selected_features: List[str],
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float | int]]]:
    """Fill null selected features with fixed model-contract defaults."""
    out = training_df.copy()
    imputed: Dict[str, Dict[str, float | int]] = {}

    for feat in selected_features:
        if feat not in out.columns:
            continue
        values = cast(pd.Series, out[feat])
        null_count = int(values.isna().sum())
        if null_count <= 0:
            continue
        default = float(_FEATURE_MEDIAN_DEFAULTS.get(feat, 0.0))
        out[feat] = values.fillna(default)
        imputed[feat] = {"count": null_count, "default": default}

    return out, imputed


def filter_modelable_selected_feature_rows(
    training_df: pd.DataFrame,
    *,
    selected_features: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Exclude rows whose selected model feature vector is incomplete.

    The active bootstrap profile is non-imputed. A row with a missing selected
    feature cannot enter the model X-matrix without either weakening the active
    feature contract or silently introducing an assumed value. Such rows remain
    valid backtest outcomes, but are classified as unmodelable for training.
    """
    out = training_df.copy()
    input_rows = int(len(out))
    excluded_rows: List[Dict[str, Any]] = []

    if input_rows == 0:
        return out, {
            "input_rows": 0,
            "modelable_rows": 0,
            "excluded_unmodelable_count": 0,
            "excluded_unmodelable_rows": [],
        }

    feature_matrix = pd.DataFrame(index=out.index)
    for feat in selected_features:
        if feat in out.columns:
            feature_matrix[feat] = pd.to_numeric(out[feat], errors="coerce")
        else:
            feature_matrix[feat] = pd.NA

    missing_mask = cast(pd.Series, feature_matrix.isna().any(axis=1))
    if bool(missing_mask.any()):
        for idx in out.index[missing_mask]:
            missing_features = [
                str(feat)
                for feat in selected_features
                if bool(pd.isna(feature_matrix.at[idx, feat]))
            ]
            row = out.loc[idx]
            excluded_rows.append({
                "candidate_id": str(row.get("candidate_id", "")),
                "symbol": str(row.get("symbol", "")),
                "classification": "excluded_unmodelable",
                "reason": "missing_selected_feature",
                "missing_features": missing_features,
            })

        out = cast(pd.DataFrame, out.loc[~missing_mask].reset_index(drop=True))

    return out, {
        "input_rows": input_rows,
        "modelable_rows": int(len(out)),
        "excluded_unmodelable_count": int(len(excluded_rows)),
        "excluded_unmodelable_rows": excluded_rows,
    }


def _build_unified_training_data(args) -> pd.DataFrame:
    """Build training data via the snapshot-only unified training builder."""
    from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder

    if not str(args.backtest_results_dir).strip():
        logger.error(
            "Active FASTWIN retraining requires --backtest-results-dir pointing "
            "to a finalized fresh full-pool run"
        )
        raise SystemExit(1)

    backtest_results_dir = Path(args.backtest_results_dir)
    if getattr(args, "feature_profile", None) == ACTIVE_META_FEATURE_PROFILE:
        try:
            validate_retrain_pool_source(
                backtest_results_dir,
                allow_historical_replay=bool(
                    getattr(args, "allow_historical_replay", False)
                ),
            )
        except ValueError as exc:
            logger.error("FASTWIN pool contract rejected: %s", exc)
            raise SystemExit(1) from exc
        pool_manifest = load_pool_manifest(backtest_results_dir)
        missing_active_features = pool_manifest.get("missing_active_features", {})
        if missing_active_features and not bool(getattr(args, "allow_imputation", False)):
            logger.error(
                "FASTWIN pool requires explicit --allow-imputation for missing "
                "active features: %s",
                missing_active_features,
            )
            raise SystemExit(1)

    label_cfg = LabelConfig(
        meta_hurdle_pct=args.hurdle_pct,
        sl_pct=args.sl_pct,
        horizon_hours=args.horizon_hours,
    )

    builder = UnifiedTrainingBuilder(
        backtest_results_dir=backtest_results_dir,
        snapshot_dir=Path(args.snapshot_dir),
        label_config=label_cfg,
    )

    # FASTWIN-01: the fast-winner profile sources the authoritative GEOMETRIC
    # backtest pool directly (ex-ante features + outcomes, geometric authority
    # gate kept). The snapshot inner-join in build() collapses to ~60 rows; the
    # geometric pool yields ~256 while preserving every leakage/authority guard.
    if getattr(args, "feature_profile", None) == ACTIVE_META_FEATURE_PROFILE:
        _hybrid = bool(getattr(args, "include_live_outcomes", False))
        logger.info(
            "FASTWIN: sourcing finalized authoritative pool (build_meta_labeler_pool, "
            "include_live_outcomes=%s, max_rows_per_symbol=%d)",
            _hybrid,
            int(args.max_rows_per_symbol),
        )
        training_df = builder.build_meta_labeler_pool(
            include_live_outcomes=_hybrid,
            max_rows_per_symbol=int(args.max_rows_per_symbol),
        )
    else:
        training_df = builder.build()

    if training_df.empty:
        logger.error("Unified training builder returned empty table")
        sys.exit(1)

    # Report provenance breakdown
    logger.info("")
    logger.info("Unified training table provenance:")
    if "source" not in training_df.columns:
        logger.warning("No 'source' column in training table — skipping provenance breakdown")
        return training_df
    for src, count in training_df["source"].value_counts().items():
        weight = training_df.loc[training_df["source"] == src, "sample_weight_override"].mean()
        logger.info(f"  {src}: {count} rows (mean weight: {weight:.2f})")

    return training_df


    # _maybe_run_backfill and _should_try_unified_backbone removed in
    # snapshot-only architecture redesign (2026-04-07).


def main():
    """Main training function."""
    args = parse_args()
    try:
        enforce_requested_profile_mode(
            feature_profile=args.feature_profile,
            allow_imputation=bool(args.allow_imputation),
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Meta-Labeler Training Script (AFML M1)")
    logger.info("=" * 70)

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading training data from: {input_path}")

    # Handle analyze-only mode
    if args.analyze_only:
        analyze_and_report(input_path)
        return

    # ── Snapshot-only unified path ────────────────────────────────────
    logger.info("Using snapshot-only unified training table builder")
    training_df = _build_unified_training_data(args)
    training_source_manifest = build_training_source_manifest(
        Path(args.backtest_results_dir), training_df
    )
    training_source_manifest["meta_pool_max_rows_per_symbol"] = int(
        args.max_rows_per_symbol
    )
    metadata = {
        "feature_availability": {
            "available_features": [
                f for f in training_df.columns
                if bool(cast(pd.Series, training_df[f]).notna().any())
            ],
            "missing_features": [
                f for f in DEFAULT_SCHEMA.features
                if f not in training_df.columns or not bool(cast(pd.Series, training_df[f]).notna().any())
            ],
        },
        "validation_issues": [],
    }

    # Report on feature availability
    availability = metadata.get("feature_availability", {})
    available_features = availability.get("available_features", [])
    missing_features = availability.get("missing_features", [])

    logger.info("")
    logger.info(f"Feature availability: {len(available_features)}/{len(available_features) + len(missing_features)} available")
    if missing_features:
        logger.info(f"  Missing: {', '.join(missing_features[:5])}{'...' if len(missing_features) > 5 else ''}")

    # Print data quality warnings
    print_data_quality_warnings(metadata)

    # Validate data
    logger.info("Validating prepared data...")
    validation_issues = metadata.get("validation_issues", [])
    if validation_issues:
        for issue in validation_issues:
            logger.warning(f"  {issue}")

    # C4 fix: validate that --hurdle-pct matches BarrierConfig.meta_hurdle_pct
    cfg = get_config()
    if abs(args.hurdle_pct - cfg.barrier.meta_hurdle_pct) > 1e-6:
        raise ValueError(
            f"--hurdle-pct ({args.hurdle_pct}) != "
            f"BarrierConfig.meta_hurdle_pct ({cfg.barrier.meta_hurdle_pct}). "
            f"These must match to prevent label/deployment divergence. "
            f"Update .env or config if the hurdle has changed."
        )

    # Configure meta-labeler
    # Note: hurdle_pct and sl_pct in MetaLabelerConfig are in percentage units (3.0 = 3%)
    # while LabelConfig uses percentages (3.0 = 3%)
    logger.info("-" * 70)
    logger.info("Training configuration:")
    logger.info(f"  Hurdle: {args.hurdle_pct:.1f}%")
    logger.info(f"  Stop loss: {args.sl_pct:.1f}%")
    logger.info(f"  Horizon: {args.horizon_hours}h")
    logger.info(f"  CV folds: {args.cv_folds}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  Feature profile: {args.feature_profile}")
    logger.info("-" * 70)

    effective_pnl_col = args.pnl_col
    if (
        effective_pnl_col == "pnl_pct"
        and "net_pnl_pct" in training_df.columns
        and bool(cast(pd.Series, training_df["net_pnl_pct"]).notna().any())
    ):
        effective_pnl_col = "net_pnl_pct"
        logger.info(
            "Using net_pnl_pct for labels to align with post-cost profitability objective"
        )
    elif effective_pnl_col not in training_df.columns:
        if "net_pnl_pct" in training_df.columns and bool(cast(pd.Series, training_df["net_pnl_pct"]).notna().any()):
            effective_pnl_col = "net_pnl_pct"
            logger.warning(
                "Requested pnl column '%s' missing. Falling back to net_pnl_pct.",
                args.pnl_col,
            )
        elif "pnl_pct" in training_df.columns:
            effective_pnl_col = "pnl_pct"
            logger.warning(
                "Requested pnl column '%s' missing. Falling back to pnl_pct.",
                args.pnl_col,
            )
        else:
            logger.error(
                "No usable PnL column found (tried '%s', 'net_pnl_pct', 'pnl_pct'). "
                "Available columns: %s",
                args.pnl_col,
                list(training_df.columns),
            )
            sys.exit(1)

    if effective_pnl_col not in training_df.columns:
        logger.error("PnL column '%s' not in training data", effective_pnl_col)
        sys.exit(1)

    selected_features = resolve_feature_profile(
        args.feature_profile,
        available_columns=list(training_df.columns),
    )
    if not selected_features:
        logger.error(
            "Feature profile '%s' resolved to zero available features. Available columns: %s",
            args.feature_profile,
            list(training_df.columns),
        )
        sys.exit(1)
    logger.info(
        "Selected %d feature(s): %s",
        len(selected_features),
        ", ".join(selected_features),
    )

    try:
        fast_training_df, training_target_summary = prepare_fast_target_training_frame(
            training_df,
            pnl_col=effective_pnl_col,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    logger.info(
        "Fast-target training universe (label_basis=%s): %d/%d labelable rows "
        "(positive=%d, negative=%d, of which slow_negatives_gt_%.1fh=%d)",
        training_target_summary.get("label_basis"),
        training_target_summary["trained_row_count"],
        len(training_df),
        training_target_summary["positive_count"],
        training_target_summary["negative_count"],
        FAST_TARGET_DURATION_MAX_HOURS,
        training_target_summary["negative_rows_duration_gt_7h_count"],
    )

    output_path = Path(args.output)
    expected_features = list(META_FEATURE_PROFILES[args.feature_profile])
    artifact_health_before = inspect_deployed_artifact_health(output_path, expected_features)
    if artifact_health_before["is_stale"]:
        logger.warning(
            "Deployed artifact health check: STALE "
            "(artifact_features=%d/%d, pickle_features=%d/%d, artifact_dir=%s)",
            artifact_health_before["artifact_feature_count"],
            artifact_health_before["expected_feature_count"],
            artifact_health_before["pickle_feature_count"],
            artifact_health_before["expected_feature_count"],
            artifact_health_before["artifact_dir"],
        )
    else:
        logger.info(
            "Deployed artifact health check: healthy (%d/%d features in both artifact directory and pickle)",
            artifact_health_before["artifact_feature_count"],
            artifact_health_before["expected_feature_count"],
        )

    imputation_report: Dict[str, Dict[str, float | int]] = {}
    if args.allow_imputation:
        fast_training_df, imputation_report = impute_selected_feature_nulls(
            fast_training_df,
            selected_features=selected_features,
        )
        if imputation_report:
            total_imputed = sum(int(item["count"]) for item in imputation_report.values())
            logger.warning(
                "Feature imputation enabled: filled %d null selected feature value(s) "
                "with fixed contract defaults",
                total_imputed,
            )
            for feat, item in imputation_report.items():
                logger.warning(
                    "  Imputed %d null(s) in '%s' with %.4f",
                    int(item["count"]),
                    feat,
                    float(item["default"]),
                )
        else:
            logger.info("Feature imputation enabled: no selected feature nulls found")

    fast_training_df, modelable_feature_filter = filter_modelable_selected_feature_rows(
        fast_training_df,
        selected_features=selected_features,
    )
    if int(modelable_feature_filter["excluded_unmodelable_count"]) > 0:
        logger.warning(
            "Excluded %d unmodelable row(s) with missing selected feature values before training",
            int(modelable_feature_filter["excluded_unmodelable_count"]),
        )
        for item in modelable_feature_filter["excluded_unmodelable_rows"]:
            logger.warning(
                "  %s excluded_unmodelable: missing %s",
                item.get("candidate_id", ""),
                ", ".join(item.get("missing_features", [])),
            )
    if fast_training_df.empty:
        logger.error("No modelable rows remain after selected-feature completeness filtering")
        sys.exit(1)
    training_target_summary = _summarize_fast_target_population(
        fast_training_df,
        pnl_col=effective_pnl_col,
    )
    active_hmm_artifact_version = _get_active_hmm_artifact_version()
    training_hmm_lineage = audit_training_hmm_lineage(
        fast_training_df,
        selected_features=selected_features,
        active_hmm_artifact_version=active_hmm_artifact_version,
        active_hmm_trained_at_utc=_get_active_hmm_trained_at_utc(
            active_hmm_artifact_version
        ),
    )

    pretrain_report = build_feature_verification_report(
        fast_training_df,
        feature_profile=args.feature_profile,
        selected_features=selected_features,
        output_path=output_path,
        artifact_health_before=artifact_health_before,
        source_training_df=training_df,
        training_target_summary=training_target_summary,
        modelable_feature_filter=modelable_feature_filter,
        training_source_manifest=training_source_manifest,
        training_hmm_lineage=training_hmm_lineage,
        allow_imputation=bool(args.allow_imputation),
        imputed_features=imputation_report,
    )
    write_feature_verification_report(pretrain_report, output_path)
    if not bool(training_hmm_lineage["passes"]):
        logger.error(
            "Training HMM lineage gate failed: %s",
            ", ".join(cast(List[str], training_hmm_lineage["reasons"])),
        )
        sys.exit(1)
    try:
        enforce_feature_contract(
            fast_training_df,
            feature_profile=args.feature_profile,
            selected_features=selected_features,
        )
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if args.export_training_data:
        export_path = Path(args.export_training_data)
        fast_training_df.to_csv(export_path, index=False)
        logger.info(f"Active fast-target training data exported to: {export_path}")

    if args.dry_run:
        logger.info("")
        logger.info("Dry run complete - feature contract is valid")
        analyze_and_report(input_path)
        return

    # FASTWIN-02 (2026-07-04): the active profile's verified-best estimator is
    # the logistic+HGB soft vote (promotion-OOF AUC 0.808 vs 0.770 logistic;
    # paired dAUC CI-low > 0 on 5 fold seeds AND under time-blocked purged
    # folds; gate ECE 0.089). "profile" derives it so routine retrains
    # reproduce the promoted architecture; --estimator logistic reproduces
    # FASTWIN-01 exactly.
    effective_estimator_type = (
        (
            "vote_logit_hgb"
            if args.feature_profile == ACTIVE_META_FEATURE_PROFILE
            else "gbm"
        )
        if args.estimator == "profile"
        else args.estimator
    )
    config = MetaLabelerConfig(
        hurdle_pct=args.hurdle_pct,
        sl_pct=args.sl_pct,
        horizon_hours=args.horizon_hours,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        features=selected_features,
        estimator_type=effective_estimator_type,
    )

    if args.phase1:
        try:
            from neutralgrid.calibration.phase1_config_v20260311 import (
                build_phase1_meta_config,
                validate_phase1_auc_floor,
            )
            p1_overrides = build_phase1_meta_config()
            logger.info("Phase 1 overrides applied: %s", p1_overrides)
            config = MetaLabelerConfig(
                hurdle_pct=args.hurdle_pct,
                sl_pct=args.sl_pct,
                horizon_hours=args.horizon_hours,
                cv_folds=p1_overrides["cv_folds"],
                random_state=args.random_state,
                n_estimators=p1_overrides["n_estimators"],
                max_depth=p1_overrides["max_depth"],
                min_samples_leaf=p1_overrides["min_samples_leaf"],
                learning_rate=p1_overrides["learning_rate"],
                calibration_method=p1_overrides["calibration_method"],
                use_time_decay=p1_overrides["use_time_decay"],
                time_decay_halflife_days=p1_overrides["time_decay_halflife_days"],
                features=selected_features,
                estimator_type=effective_estimator_type,
            )
        except ImportError:
            logger.warning("Phase 1 calibration module not available, using default config")

    labeler = MetaLabeler(config)

    # Train
    # Note: feature diagnostics (AFML Ch8) now run inside MetaLabeler.train().
    # The active training target is the explicit fast-winner contract built in
    # prepare_fast_target_training_frame(), not the builder's legacy y column.
    logger.info("Training meta-labeler...")
    assert isinstance(fast_training_df, pd.DataFrame), "fast_training_df must be a DataFrame"
    try:
        metrics = labeler.train(
            fast_training_df,
            timestamp_col=args.timestamp_col,
            pnl_col=effective_pnl_col,
            y_col=FAST_TARGET_LABEL_COLUMN,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    if args.phase1:
        try:
            from neutralgrid.calibration.phase1_config_v20260311 import validate_phase1_auc_floor
            p1_passes, p1_reason = validate_phase1_auc_floor(metrics.auc_cv)
            if not p1_passes:
                logger.warning("Phase 1 AUC floor FAILED: %s", p1_reason)
                logger.warning("Consider re-running without --phase1 for default complexity")
        except ImportError:
            pass

    # Log trial for AFML deflated Sharpe calculation
    logger.info("")
    logger.info("Logging trial to trial tracker...")
    try:
        tracker = TrialTracker()

        trial = TrialRecord(
            trial_id=f"meta_labeler_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            model_type='meta_labeler',
            hyperparameters={
                'hurdle_pct': args.hurdle_pct,
                'sl_pct': args.sl_pct,
                'horizon_hours': args.horizon_hours,
                'cv_folds': args.cv_folds,
                'random_state': args.random_state,
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
            },
            cv_score=metrics.auc_cv,
            feature_set=list(metrics.feature_importance.keys()),
            notes=f"Training samples: {metrics.train_samples}, Positive rate: {metrics.positive_rate:.1%}"
        )

        tracker.log_trial(trial)

        # Report trial history
        n_trials = tracker.get_trial_count(model_type='meta_labeler')
        logger.info(f"  Total trials logged: {n_trials}")

        if n_trials > 1:
            cv_scores = tracker.get_cv_scores('meta_labeler')
            if cv_scores:
                logger.info(f"  Mean CV AUC (all trials): {sum(cv_scores)/len(cv_scores):.4f}")
                logger.info(f"  Best CV AUC (all trials): {max(cv_scores):.4f}")

    except Exception as e:
        logger.warning(f"Failed to log trial: {e}")

    # Report metrics
    logger.info("-" * 70)
    logger.info("Training Results:")
    logger.info(f"  AUC (CV): {metrics.auc_cv:.3f}")
    logger.info(f"  Precision@5: {metrics.precision_at_5:.3f}")
    logger.info(f"  Training samples: {metrics.train_samples}")
    logger.info(f"  Positive rate: {metrics.positive_rate:.1%}")

    # Feature importance
    logger.info("")
    logger.info("Feature Importance (top 10):")
    # Handle both old format (float) and new AFML format (dict with mdi/permutation)
    def _get_importance_value(imp):
        if isinstance(imp, dict):
            return imp.get("mdi", 0.0)
        return float(imp)

    importance_sorted = sorted(
        metrics.feature_importance.items(),
        key=lambda x: _get_importance_value(x[1]),
        reverse=True
    )
    for feat, imp in importance_sorted[:10]:
        if isinstance(imp, dict):
            logger.info(f"  {feat:25s}: MDI={imp.get('mdi', 0):.4f}  Perm={imp.get('permutation', 0):.4f}")
        else:
            logger.info(f"  {feat:25s}: {imp:.4f}")

    zero_mdi_features = [
        f for f, imp in importance_sorted if _get_importance_value(imp) == 0
    ]
    if zero_mdi_features:
        logger.info("")
        logger.info(
            "Note: MDI is unavailable/zero for %d features; OOS permutation "
            "importance is reported separately",
            len(zero_mdi_features),
        )

    # ERR-088: a trained candidate is not automatically a deployable champion.
    # Evaluate and persist the fail-closed decision before creating a backup or
    # writing either the artifact directory or the legacy pickle.
    promotion_decision = evaluate_champion_challenger_gate(
        output_path,
        expected_features,
        metrics,
    )
    decision_path = write_champion_challenger_decision(
        promotion_decision, output_path
    )
    if promotion_decision["status"] != "pass":
        logger.error(
            "Champion/challenger promotion gate FAILED; deployed artifact is "
            "unchanged. reasons=%s decision=%s",
            promotion_decision["reasons"],
            decision_path,
        )
        sys.exit(1)
    logger.info(
        "Champion/challenger promotion gate passed (%s); decision=%s",
        promotion_decision.get("comparison_scope"),
        decision_path,
    )

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # C6 fix: timestamped backup before overwriting existing model
    if output_path.exists():
        import shutil
        _backup_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(
            f"{output_path.stem}_backup_{_backup_ts}{output_path.suffix}"
        )
        shutil.copy2(output_path, backup_path)
        logger.info("Backed up previous model to %s", backup_path)

    logger.info("-" * 70)
    try:
        labeler.save(output_path)
        logger.info(f"Model saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)

    artifact_health_after = inspect_deployed_artifact_health(output_path, expected_features)
    posttrain_report = build_feature_verification_report(
        fast_training_df,
        feature_profile=args.feature_profile,
        selected_features=selected_features,
        output_path=output_path,
        artifact_health_before=artifact_health_before,
        artifact_health_after=artifact_health_after,
        source_training_df=training_df,
        training_target_summary=training_target_summary,
        modelable_feature_filter=modelable_feature_filter,
        training_source_manifest=training_source_manifest,
        training_hmm_lineage=training_hmm_lineage,
        allow_imputation=bool(args.allow_imputation),
        imputed_features=imputation_report,
    )
    write_feature_verification_report(posttrain_report, output_path)
    if artifact_health_after["is_stale"]:
        logger.error(
            "Post-save artifact verification FAILED: the deployed model is not healthy until "
            "both %s and %s exist and both advertise the exact %d-feature profile.",
            artifact_health_after["artifact_dir"],
            output_path,
            artifact_health_after["expected_feature_count"],
        )
        sys.exit(1)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Samples:           {metrics.train_samples}")
    logger.info(
        "  Model input features: %d/%d",
        len(metrics.feature_importance),
        len(selected_features),
    )
    logger.info(f"  Positive rate:     {metrics.positive_rate:.1%}")
    logger.info(f"  CV AUC:            {metrics.auc_cv:.3f}")
    logger.info(f"  Precision@5:       {metrics.precision_at_5:.3f}")
    logger.info(f"  Model saved to:    {output_path}")
    logger.info("=" * 70)
    logger.info("Training complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
