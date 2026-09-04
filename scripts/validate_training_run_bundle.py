#!/usr/bin/env python3
"""Validate a run-scoped full-pipeline training bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, cast

import pandas as pd


def _candidate_ids(df: pd.DataFrame) -> pd.Series:
    if df.empty or "candidate_id" not in df.columns:
        return pd.Series(dtype=str)
    return (
        cast(pd.Series, df["candidate_id"])
        .fillna("")
        .astype(str)
        .str.strip()
    )


def _duplicate_ids(ids: pd.Series) -> list[str]:
    nonblank = ids.loc[ids != ""]
    if nonblank.empty:
        return []
    return sorted(cast(pd.Series, nonblank.loc[nonblank.duplicated(keep=False)]).unique().tolist())


def _read_csv_frames(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def _read_parquet_frames(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return cast(pd.DataFrame, pd.concat(frames, ignore_index=True))


def _sorted_nonblank_set(ids: pd.Series) -> set[str]:
    return set(cast(pd.Series, ids.loc[ids != ""]).astype(str).tolist())


def validate_run_bundle(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    snapshot_dir = run_dir / "snapshots"
    backtest_dir = run_dir / "backtest_candidates"

    snapshot_paths = sorted(snapshot_dir.glob("snapshots_*.parquet"))
    training_paths = sorted(backtest_dir.glob("training_data_*.csv"))
    result_paths = sorted(backtest_dir.glob("backtest_results_*.csv"))

    snapshot_df = _read_parquet_frames(snapshot_paths)
    training_df = _read_csv_frames(training_paths)
    result_df = _read_csv_frames(result_paths)

    snapshot_ids = _candidate_ids(snapshot_df)
    training_ids = _candidate_ids(training_df)
    result_ids = _candidate_ids(result_df)

    snapshot_id_set = _sorted_nonblank_set(snapshot_ids)
    backtest_id_set = _sorted_nonblank_set(training_ids) | _sorted_nonblank_set(result_ids)

    matched = sorted(backtest_id_set & snapshot_id_set)
    missing = sorted(backtest_id_set - snapshot_id_set)
    snapshot_without_backtest = sorted(snapshot_id_set - backtest_id_set)
    blank_backtest_count = int((training_ids == "").sum()) + int((result_ids == "").sum())

    duplicate_backtest = sorted(set(_duplicate_ids(training_ids)) | set(_duplicate_ids(result_ids)))
    duplicate_snapshot = _duplicate_ids(snapshot_ids)

    match_rate = float(len(matched) / len(backtest_id_set)) if backtest_id_set else 0.0
    errors: list[str] = []
    if not snapshot_paths:
        errors.append("missing_snapshot_files")
    if not training_paths and not result_paths:
        errors.append("missing_backtest_files")
    if blank_backtest_count:
        errors.append("blank_backtest_candidate_id")
    if missing:
        errors.append("backtested_missing_snapshot")
    if backtest_id_set and match_rate != 1.0:
        errors.append("backtest_snapshot_match_rate_below_1")
    if not backtest_id_set:
        errors.append("no_backtest_candidate_ids")

    passed = not errors
    return {
        "run_dir": str(run_dir),
        "passed": passed,
        "errors": errors,
        "metrics": {
            "snapshot_rows": int(len(snapshot_df)),
            "training_rows": int(len(training_df)),
            "backtest_result_rows": int(len(result_df)),
            "unique_snapshot_candidate_ids": int(len(snapshot_id_set)),
            "unique_backtest_candidate_ids": int(len(backtest_id_set)),
            "backtest_snapshot_match_rate": match_rate,
            "backtested_missing_snapshot_count": int(len(missing)),
            "blank_backtest_candidate_id_count": blank_backtest_count,
            "duplicate_backtest_candidate_id_count": int(len(duplicate_backtest)),
            "duplicate_snapshot_candidate_id_count": int(len(duplicate_snapshot)),
            "snapshot_without_backtest_count": int(len(snapshot_without_backtest)),
        },
        "classifications": {
            "backtested_snapshot_matched": matched,
            "backtested_missing_snapshot": missing,
            "snapshot_without_backtest": snapshot_without_backtest,
            "duplicate_backtest_candidate_id": duplicate_backtest,
            "duplicate_snapshot_candidate_id": duplicate_snapshot,
            "blank_candidate_id": {
                "backtest_count": blank_backtest_count,
                "snapshot_count": int((snapshot_ids == "").sum()),
            },
        },
        "sources": {
            "snapshot_files": [str(path) for path in snapshot_paths],
            "training_data_files": [str(path) for path in training_paths],
            "backtest_result_files": [str(path) for path in result_paths],
        },
    }


def write_validation(run_dir: Path, payload: dict[str, Any]) -> Path:
    validation_dir = Path(run_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    output_path = validation_dir / "run_bundle_validation.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that a run-scoped training bundle has exact snapshot/backtest candidate_id coverage.",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = validate_run_bundle(Path(args.run_dir))
    output_path = write_validation(Path(args.run_dir), payload)
    print(json.dumps({
        "passed": payload["passed"],
        "output_path": str(output_path),
        "metrics": payload["metrics"],
        "errors": payload["errors"],
    }, indent=2, sort_keys=True))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
