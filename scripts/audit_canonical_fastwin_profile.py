"""Audit whether canonical FASTWIN backtest rows can support a profile refit.

This tool is deliberately read-only with respect to production artifacts. It
accepts only authoritative canonical backtest rows, derives the fixed FASTWIN
target from ``time_to_target_hours <= 7``, and reports feature/data readiness.
It never trains a model and never writes ``data/profile/current.json``.

``t1_is_synthetic`` is not a row-origin field. In the canonical backtest
contract it records how the triple-barrier vertical horizon was constructed;
row origin is governed by ``source == 'backtest'`` plus the authoritative and
engine/label/formula/realism lineage columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    validate_realism_output_path,
)
from neutralgrid.core.constants import (
    ENGINE_VERSION,
    FORMULA_VERSION,
    LABEL_CONTRACT_VERSION,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES


AUDIT_SCHEMA_VERSION = 1
FASTWIN_TARGET_NAME = "fast_winner_time_to_3pct_le_7h"
FASTWIN_TARGET_HOURS = 7.0
REALISM_PROFILE = CANDIDATE_TIME_GEOMETRIC_PROFILE
MIN_CLASS_ROWS_FOR_PROFILE_FIT = 30

_BASE_REQUIRED_COLUMNS = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    validate_realism_output_path(REALISM_PROFILE, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _true_mask(series: pd.Series) -> pd.Series:
    return cast(pd.Series, series.astype(str).str.strip().str.lower().eq("true"))


def _contract_mismatch_count(
    frame: pd.DataFrame,
    column: str,
    expected: str,
) -> int:
    if column not in frame.columns:
        return len(frame)
    actual = cast(pd.Series, frame[column]).astype(str).str.strip()
    return int(actual.ne(expected).sum())


def audit_canonical_fastwin_profile(data_dir: Path) -> dict[str, Any]:
    """Return a fail-closed readiness report for canonical FASTWIN rows."""
    source_paths = sorted(data_dir.glob("training_data_fastwin_*.csv"))
    if not source_paths:
        raise FileNotFoundError(
            f"no training_data_fastwin_*.csv files found in {data_dir}"
        )

    frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    missing_columns_by_file: dict[str, list[str]] = {}
    required_columns = [*_BASE_REQUIRED_COLUMNS, *DEFAULT_FEATURES]
    for path in source_paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            missing_columns_by_file[path.name] = missing
        frames.append(frame)
        source_records.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "rows": len(frame),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    row_count = len(combined)
    blockers: list[str] = []
    if missing_columns_by_file:
        blockers.append("required_columns_missing")

    candidate_ids = (
        cast(pd.Series, combined["candidate_id"])
        .fillna("")
        .astype(str)
        .str.strip()
        if "candidate_id" in combined.columns
        else pd.Series("", index=combined.index, dtype=str)
    )
    missing_candidate_ids = int(candidate_ids.eq("").sum())
    duplicate_candidate_ids = int(
        candidate_ids.loc[candidate_ids.ne("")].duplicated(keep=False).sum()
    )
    if missing_candidate_ids:
        blockers.append(f"missing_candidate_ids={missing_candidate_ids}")
    if duplicate_candidate_ids:
        blockers.append(f"duplicate_candidate_id_rows={duplicate_candidate_ids}")

    expected_contract = {
        "source": "backtest",
        "engine_version": ENGINE_VERSION,
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "formula_version": FORMULA_VERSION,
        "realism_profile": REALISM_PROFILE,
    }
    contract_mismatches = {
        column: _contract_mismatch_count(combined, column, expected)
        for column, expected in expected_contract.items()
    }
    for column, count in contract_mismatches.items():
        if count:
            blockers.append(f"{column}_mismatch_rows={count}")

    authoritative = (
        _true_mask(cast(pd.Series, combined["is_authoritative"]))
        if "is_authoritative" in combined.columns
        else pd.Series(False, index=combined.index, dtype=bool)
    )
    non_authoritative_rows = int((~authoritative).sum())
    if non_authoritative_rows:
        blockers.append(f"non_authoritative_rows={non_authoritative_rows}")

    invalid_timestamp_counts: dict[str, int] = {}
    for column in ("start_time_utc", "backtest_timestamp"):
        if column not in combined.columns:
            invalid_timestamp_counts[column] = row_count
            continue
        parsed = pd.to_datetime(combined[column], errors="coerce", utc=True)
        invalid_timestamp_counts[column] = int(parsed.isna().sum())
    for column, count in invalid_timestamp_counts.items():
        if count:
            blockers.append(f"invalid_{column}_rows={count}")

    time_to_target = (
        cast(
            pd.Series,
            pd.to_numeric(combined["time_to_target_hours"], errors="coerce"),
        )
        if "time_to_target_hours" in combined.columns
        else pd.Series(np.nan, index=combined.index, dtype=float)
    )
    target_reached = (
        _true_mask(cast(pd.Series, combined["target_reached"]))
        if "target_reached" in combined.columns
        else pd.Series(False, index=combined.index, dtype=bool)
    )
    finite_time = pd.Series(
        np.isfinite(np.asarray(time_to_target, dtype=float)),
        index=combined.index,
        dtype=bool,
    )
    negative_time = cast(pd.Series, time_to_target.lt(0.0)).fillna(False)
    target_reached_missing_time = int((target_reached & ~finite_time).sum())
    target_not_reached_with_time = int((~target_reached & finite_time).sum())
    negative_time_rows = int(negative_time.sum())
    if target_reached_missing_time:
        blockers.append(
            f"target_reached_missing_time_rows={target_reached_missing_time}"
        )
    if target_not_reached_with_time:
        blockers.append(
            f"target_not_reached_with_time_rows={target_not_reached_with_time}"
        )
    if negative_time_rows:
        blockers.append(f"negative_time_to_target_rows={negative_time_rows}")

    exact_target = (
        cast(pd.Series, time_to_target.le(FASTWIN_TARGET_HOURS)).fillna(False)
        & finite_time
        & ~negative_time
    )
    positive_rows = int(exact_target.sum())
    negative_rows = int((~exact_target).sum())

    feature_finite_rows: dict[str, int] = {}
    feature_complete = pd.Series(True, index=combined.index, dtype=bool)
    for feature in DEFAULT_FEATURES:
        if feature not in combined.columns:
            feature_finite_rows[feature] = 0
            feature_complete &= False
            continue
        values = pd.to_numeric(combined[feature], errors="coerce")
        finite = pd.Series(np.isfinite(np.asarray(values, dtype=float)), index=combined.index)
        feature_finite_rows[feature] = int(finite.sum())
        feature_complete &= finite
    complete_rows = int(feature_complete.sum())
    complete_positive_rows = int((feature_complete & exact_target).sum())
    complete_negative_rows = int((feature_complete & ~exact_target).sum())
    if complete_rows != row_count:
        blockers.append(
            f"profile_feature_incomplete_rows={row_count - complete_rows}"
        )
    if complete_positive_rows < MIN_CLASS_ROWS_FOR_PROFILE_FIT:
        blockers.append(
            "feature_complete_positive_rows="
            f"{complete_positive_rows} < floor={MIN_CLASS_ROWS_FOR_PROFILE_FIT}"
        )
    if complete_negative_rows < MIN_CLASS_ROWS_FOR_PROFILE_FIT:
        blockers.append(
            "feature_complete_negative_rows="
            f"{complete_negative_rows} < floor={MIN_CLASS_ROWS_FOR_PROFILE_FIT}"
        )

    fit_viable = not blockers
    # Training/development rows are never their own untouched promotion test.
    promotion_blockers = list(blockers)
    promotion_blockers.append("fresh_unused_temporal_holdout_required")

    t1_synthetic_counts: dict[str, int] | None = None
    if "t1_is_synthetic" in combined.columns:
        normalized = (
            cast(pd.Series, combined["t1_is_synthetic"])
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        t1_synthetic_counts = {
            str(key): int(value)
            for key, value in normalized.value_counts(dropna=False).items()
        }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": source_records,
        "row_origin_contract": {
            "accepted_source": "backtest",
            "authoritative_required": True,
            "synthetic_training_rows_allowed": False,
            "t1_is_synthetic_semantics": (
                "triple_barrier_vertical_horizon_metadata_not_row_origin"
            ),
            "t1_is_synthetic_value_counts": t1_synthetic_counts,
        },
        "target_contract": {
            "name": FASTWIN_TARGET_NAME,
            "definition": "finite time_to_target_hours <= 7.0; otherwise negative",
            "target_hours": FASTWIN_TARGET_HOURS,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "finite_time_to_target_rows": int(time_to_target.notna().sum()),
            "negative_time_to_target_rows": negative_time_rows,
            "target_reached_missing_time_rows": target_reached_missing_time,
            "target_not_reached_with_time_rows": target_not_reached_with_time,
        },
        "lineage": {
            "expected": expected_contract,
            "mismatch_rows": contract_mismatches,
            "non_authoritative_rows": non_authoritative_rows,
            "invalid_timestamp_rows": invalid_timestamp_counts,
        },
        "identity": {
            "rows": row_count,
            "unique_candidate_ids": int(candidate_ids.loc[candidate_ids.ne("")].nunique()),
            "missing_candidate_ids": missing_candidate_ids,
            "duplicate_candidate_id_rows": duplicate_candidate_ids,
        },
        "feature_contract": {
            "requested_features": list(DEFAULT_FEATURES),
            "missing_columns_by_file": missing_columns_by_file,
            "finite_rows_by_feature": feature_finite_rows,
            "feature_complete_rows": complete_rows,
            "feature_complete_positive_rows": complete_positive_rows,
            "feature_complete_negative_rows": complete_negative_rows,
            "minimum_rows_per_class": MIN_CLASS_ROWS_FOR_PROFILE_FIT,
            "imputation_allowed_for_promotion_fit": False,
        },
        "fit_assessment": {
            "viable": fit_viable,
            "blockers": blockers,
        },
        "promotion_assessment": {
            "promote": False,
            "reason": "canonical_training_rows_are_not_promotion_sufficient",
            "blockers": promotion_blockers,
            "production_artifacts_modified": False,
        },
        "accuracy_assessment": {
            "status": "not_measurable" if not fit_viable else "not_evaluated",
            "reason": (
                "profile features are unavailable on canonical rows"
                if complete_rows == 0
                else "a fresh unused temporal holdout is required"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/fastwin_dataset"),
        help="Directory containing training_data_fastwin_*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Audit JSON path (default: timestamped path under outputs/audits).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = Path("outputs/audits") / (
            f"canonical_fastwin_profile_readiness_{stamp}.json"
        )
    report = audit_canonical_fastwin_profile(args.data_dir)
    _atomic_write_json(output, report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
