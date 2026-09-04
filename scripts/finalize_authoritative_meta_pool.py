#!/usr/bin/env python3
"""Finalize a replayed backtest CSV into a governed meta-labeler pool.

The finalizer never infers or stamps HMM values. It admits only rows whose
replay output already proves the exact active artifact, trained timestamp,
feature semantics, causal cutoff, finite probabilities, complete model feature
vector, current authority contract, unique candidate identity, and unchanged
outcome evidence. The output directory is published through a same-volume
atomic rename and must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast
import uuid

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.training.data_generator import HMM_FEATURE_SEMANTICS_VERSION
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder
from neutralgrid.training.meta_pool_contract import (
    AUTHORITATIVE_POOL_MANIFEST,
    HISTORICAL_EXACT_REPLAY_GENERATION_MODE,
    POOL_CONTRACT_VERSION,
)


OUTCOME_COLUMNS = (
    "pnl_pct",
    "y",
    "sl_hit",
    "barrier_touched",
    "duration_hours",
    "time_to_target_hours",
    "target_reached",
    "horizon_censored",
    "realized_net_pnl",
    "realized_net_pnl_pct",
    "unrealized_fraction",
    "hlabel",
    "hlabel_meta",
)
NUMERIC_OUTCOME_COLUMNS = frozenset(
    {
        "pnl_pct",
        "duration_hours",
        "time_to_target_hours",
        "realized_net_pnl",
        "realized_net_pnl_pct",
        "unrealized_fraction",
    }
)
_CSV_FLOAT_RTOL = 8.0 * np.finfo(np.float64).eps
_CSV_FLOAT_ATOL = 8.0 * np.finfo(np.float64).eps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = cast(pd.Series, frame[column]).value_counts(dropna=False)
    return {str(value): int(count) for value, count in counts.items()}


def _load_active_contract(expected_version: str) -> dict[str, str]:
    manifest_path = _ROOT / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hmm_manifest = manifest.get("hmm")
    if not isinstance(hmm_manifest, dict):
        raise ValueError("artifact_manifest.json has no hmm object")
    active_version = str(hmm_manifest.get("active_version", "")).strip()
    if not active_version or active_version != expected_version:
        raise ValueError(
            "Requested HMM is not active: "
            f"requested={expected_version!r}, active={active_version!r}"
        )
    artifact_dir = hmm_manifest.get("artifact_dir")
    if not artifact_dir:
        raise ValueError("Active HMM manifest has no artifact_dir")
    metadata_path = _ROOT / str(artifact_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("artifact_version", "")).strip() != active_version:
        raise ValueError("Active HMM metadata artifact_version disagrees with manifest")
    trained_at = str(metadata.get("trained_at_utc", "")).strip()
    pipeline_version = str(metadata.get("pipeline_version", "")).strip()
    if not trained_at or not pipeline_version:
        raise ValueError("Active HMM metadata lacks trained_at_utc or pipeline_version")
    return {
        "artifact_version": active_version,
        "trained_at_utc": trained_at,
        "pipeline_version": pipeline_version,
        "feature_semantics_version": HMM_FEATURE_SEMANTICS_VERSION,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def _strict_true(series: pd.Series) -> pd.Series:
    normalized = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() == "true"
    )
    return cast(pd.Series, normalized.astype(bool))


def _parse_utc_boundary(value: str, *, field: str) -> pd.Timestamp:
    parsed = pd.to_datetime(
        pd.Series([value]), utc=True, errors="coerce", format="mixed"
    ).iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"{field} is not a valid UTC date: {value!r}")
    return cast(pd.Timestamp, parsed)


def _assert_outcomes_unchanged(source: pd.DataFrame, replay: pd.DataFrame) -> None:
    if len(source) != len(replay):
        raise ValueError(
            f"Replay row count changed: source={len(source)}, replay={len(replay)}"
        )
    for frame_name, frame in (("source", source), ("replay", replay)):
        if "candidate_id" not in frame.columns:
            raise ValueError(f"{frame_name} is missing candidate_id")
        ids = cast(pd.Series, frame["candidate_id"]).fillna("").astype(str).str.strip()
        if bool(ids.eq("").any()) or bool(ids.duplicated(keep=False).any()):
            raise ValueError(f"{frame_name} candidate_id values are blank or duplicated")

    source_by_id = source.set_index("candidate_id", drop=False).sort_index()
    replay_by_id = replay.set_index("candidate_id", drop=False).sort_index()
    if source_by_id.index.tolist() != replay_by_id.index.tolist():
        raise ValueError("Replay candidate_id population differs from source")
    for column in OUTCOME_COLUMNS:
        if column not in source_by_id.columns or column not in replay_by_id.columns:
            continue
        left = cast(pd.Series, source_by_id[column])
        right = cast(pd.Series, replay_by_id[column])
        if column in NUMERIC_OUTCOME_COLUMNS:
            left_numeric = pd.to_numeric(left, errors="coerce")
            right_numeric = pd.to_numeric(right, errors="coerce")
            equal = pd.Series(
                np.isclose(
                    np.asarray(left_numeric, dtype=float),
                    np.asarray(right_numeric, dtype=float),
                    rtol=_CSV_FLOAT_RTOL,
                    atol=_CSV_FLOAT_ATOL,
                    equal_nan=True,
                ),
                index=left.index,
            )
        else:
            equal = left.eq(right) | (left.isna() & right.isna())
        if not bool(equal.all()):
            raise ValueError(
                f"Replay modified outcome column {column!r} on {int((~equal).sum())} row(s)"
            )


def finalize_pool(
    *,
    source_path: Path,
    replay_path: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    active_hmm_artifact_version: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output directory: {output_dir}")
    source = pd.read_csv(source_path, low_memory=False)
    replay = pd.read_csv(replay_path, low_memory=False)
    _assert_outcomes_unchanged(source, replay)
    contract = _load_active_contract(active_hmm_artifact_version)

    start = cast(
        pd.Timestamp,
        _parse_utc_boundary(start_date, field="start-date").normalize(),
    )
    end = cast(
        pd.Timestamp,
        _parse_utc_boundary(end_date, field="end-date").normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1),
    )
    if end < start:
        raise ValueError("end-date precedes start-date")

    replay = replay.reset_index(drop=True)
    reasons: list[list[str]] = [[] for _ in range(len(replay))]

    def mark(mask: pd.Series, reason: str) -> None:
        positions = np.flatnonzero(np.asarray(mask.fillna(False), dtype=bool))
        for position in positions:
            reasons[int(position)].append(reason)

    candidate_ids = (
        cast(pd.Series, replay["candidate_id"]).fillna("").astype(str).str.strip()
    )
    parts = candidate_ids.str.extract(
        r"^([^_]+)_(\d{8})_(\d{6})_([0-9a-fA-F]{8})$"
    )
    scan_time = pd.to_datetime(
        parts[1].fillna("") + parts[2].fillna(""),
        format="%Y%m%d%H%M%S",
        utc=True,
        errors="coerce",
    )
    mark(scan_time.isna(), "candidate_id_invalid")
    mark(scan_time.notna() & ((scan_time < start) | (scan_time > end)), "outside_date_window")
    mark(candidate_ids.duplicated(keep=False), "candidate_id_duplicate")
    if "symbol" not in replay.columns:
        mark(pd.Series(True, index=replay.index), "symbol_column_missing")
    else:
        symbols = cast(pd.Series, replay["symbol"]).fillna("").astype(str).str.upper()
        mark(parts[0].fillna("").str.upper().ne(symbols), "candidate_id_symbol_mismatch")

    gated = UnifiedTrainingBuilder()._apply_ingestion_gate(replay)
    if "version_gated" not in gated.columns:
        mark(pd.Series(True, index=replay.index), "authority_gate_missing")
    else:
        mark(cast(pd.Series, gated["version_gated"]).astype(bool), "authority_contract_failed")
    if "is_authoritative" not in replay.columns:
        mark(pd.Series(True, index=replay.index), "is_authoritative_missing")
    else:
        mark(~_strict_true(cast(pd.Series, replay["is_authoritative"])), "is_authoritative_not_true")

    expected_text = {
        "hmm_artifact_version": contract["artifact_version"],
        "hmm_trained_at_utc": contract["trained_at_utc"],
        "hmm_pipeline_version": contract["pipeline_version"],
        "hmm_feature_semantics_version": contract["feature_semantics_version"],
        "hmm_feature_source": "pinned_artifact_replay",
        "hmm_replay_scope": "hmm_lineage_only",
    }
    for column, expected in expected_text.items():
        if column not in replay.columns:
            mark(pd.Series(True, index=replay.index), f"{column}_missing")
            continue
        values = cast(pd.Series, replay[column]).fillna("").astype(str).str.strip()
        if column == "hmm_trained_at_utc":
            parsed = pd.to_datetime(values, utc=True, errors="coerce", format="mixed")
            expected_parsed = pd.Timestamp(expected)
            mark(parsed.isna() | parsed.ne(expected_parsed), f"{column}_mismatch")
        else:
            mark(values.ne(expected), f"{column}_mismatch")

    if "feature_cutoff_utc" not in replay.columns or "start_time_utc" not in replay.columns:
        mark(pd.Series(True, index=replay.index), "causal_timestamp_missing")
    else:
        cutoff = pd.to_datetime(
            replay["feature_cutoff_utc"], utc=True, errors="coerce", format="mixed"
        )
        event_start = pd.to_datetime(
            replay["start_time_utc"], utc=True, errors="coerce", format="mixed"
        )
        mark(cutoff.isna() | event_start.isna(), "causal_timestamp_invalid")
        mark(cutoff.notna() & event_start.notna() & cutoff.gt(event_start), "feature_cutoff_after_event")
        mark(
            cutoff.notna() & scan_time.notna() & cutoff.ne(scan_time),
            "feature_cutoff_not_candidate_scan_time",
        )

    for column in ("range_prob", "trend_prob", "persistence_prob"):
        if column not in replay.columns:
            mark(pd.Series(True, index=replay.index), f"{column}_missing")
            continue
        numeric = cast(pd.Series, pd.to_numeric(replay[column], errors="coerce"))
        finite = pd.Series(
            np.isfinite(np.asarray(numeric, dtype=float)), index=replay.index
        )
        mark(~finite, f"{column}_nonfinite")
        mark(finite & ~numeric.between(0.0, 1.0, inclusive="both"), f"{column}_out_of_range")

    feature_frame = pd.DataFrame(index=replay.index)
    for feature in ACTIVE_SNAPSHOT_META_FEATURES:
        if feature in replay.columns:
            feature_frame[feature] = pd.to_numeric(replay[feature], errors="coerce")
        else:
            feature_frame[feature] = np.nan
    feature_finite = pd.DataFrame(
        np.isfinite(np.asarray(feature_frame, dtype=float)),
        index=replay.index,
        columns=feature_frame.columns,
    )
    mark(~cast(pd.Series, feature_finite.all(axis=1)), "selected_feature_incomplete")

    eligible_mask = pd.Series([not row_reasons for row_reasons in reasons], dtype=bool)
    eligible = cast(pd.DataFrame, replay.loc[eligible_mask].copy())
    eligible["pool_contract_version"] = POOL_CONTRACT_VERSION
    excluded = pd.DataFrame(
        {
            "candidate_id": candidate_ids.loc[~eligible_mask].tolist(),
            "symbol": cast(pd.Series, replay.get("symbol", pd.Series("", index=replay.index))).loc[
                ~eligible_mask
            ].tolist(),
            "exclusion_reasons": [
                ";".join(reasons[position])
                for position in np.flatnonzero(np.asarray(~eligible_mask, dtype=bool))
            ],
        }
    )
    reason_counts: dict[str, int] = {}
    for row_reasons in reasons:
        for reason in row_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if eligible.empty:
        raise ValueError(
            "Authoritative pool contains zero eligible rows; refusing publication"
        )

    final_name = f"training_data_{end.strftime('%Y%m%d')}.csv"
    manifest_name = f"authoritative_pool_manifest_{end.strftime('%Y%m%d')}.json"
    excluded_name = f"excluded_rows_{end.strftime('%Y%m%d')}.csv"
    temp_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        pool_path = temp_dir / final_name
        excluded_path = temp_dir / excluded_name
        eligible.to_csv(pool_path, index=False)
        excluded.to_csv(excluded_path, index=False)
        manifest: dict[str, Any] = {
            "pool_contract_version": POOL_CONTRACT_VERSION,
            "generation_mode": HISTORICAL_EXACT_REPLAY_GENERATION_MODE,
            "full_pool": False,
            "feature_cutoff_contract": "candidate_id_scan_time",
            "output_dir": str(output_dir.resolve()),
            "source_path": str(source_path.resolve()),
            "source_sha256": _sha256(source_path),
            "replay_path": str(replay_path.resolve()),
            "replay_sha256": _sha256(replay_path),
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "input_rows": int(len(replay)),
            "eligible_rows": int(len(eligible)),
            "excluded_rows": int(len(excluded)),
            "unique_eligible_candidate_ids": int(eligible["candidate_id"].nunique()),
            "eligible_symbols": int(eligible["symbol"].nunique()),
            "eligible_label_counts": _value_counts(eligible, "y"),
            "eligible_valid_indicator_counts": _value_counts(
                eligible, "valid_indicator"
            ),
            "eligible_sample_weight_override_counts": _value_counts(
                eligible, "sample_weight_override"
            ),
            "eligible_hmm_artifact_version_counts": _value_counts(
                eligible, "hmm_artifact_version"
            ),
            "eligible_hmm_trained_at_utc_counts": _value_counts(
                eligible, "hmm_trained_at_utc"
            ),
            "eligible_hmm_feature_source_counts": _value_counts(
                eligible, "hmm_feature_source"
            ),
            "eligible_candidate_scan_time_utc": {
                "min": cast(
                    pd.Timestamp, scan_time.loc[eligible_mask].min()
                ).isoformat(),
                "max": cast(
                    pd.Timestamp, scan_time.loc[eligible_mask].max()
                ).isoformat(),
            },
            "eligible_event_start_time_utc": {
                "min": cast(
                    pd.Timestamp,
                    pd.to_datetime(
                        eligible["start_time_utc"],
                        utc=True,
                        errors="coerce",
                        format="mixed",
                    ).min(),
                ).isoformat(),
                "max": cast(
                    pd.Timestamp,
                    pd.to_datetime(
                        eligible["start_time_utc"],
                        utc=True,
                        errors="coerce",
                        format="mixed",
                    ).max(),
                ).isoformat(),
            },
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "active_hmm": contract,
            "selected_features": list(ACTIVE_SNAPSHOT_META_FEATURES),
            "pool_file": final_name,
            "pool_sha256": _sha256(pool_path),
            "excluded_file": excluded_name,
            "excluded_sha256": _sha256(excluded_path),
        }
        manifest_path = temp_dir / manifest_name
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (temp_dir / AUTHORITATIVE_POOL_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for path in (pool_path, excluded_path, manifest_path):
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, output_dir)
        return manifest
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--active-hmm-artifact-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = finalize_pool(
        source_path=args.source,
        replay_path=args.replay,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        active_hmm_artifact_version=args.active_hmm_artifact_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
