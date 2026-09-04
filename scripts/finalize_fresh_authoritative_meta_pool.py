#!/usr/bin/env python3
"""Finalize a fresh full-pool backtest after active-HMM backfill."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neutralgrid.backtest.realism_governance import LEGACY_REALISM_PROFILE
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.training.data_generator import HMM_FEATURE_SEMANTICS_VERSION
from neutralgrid.training.meta_pool_contract import (
    AUTHORITATIVE_POOL_MANIFEST,
    FRESH_FULL_POOL_GENERATION_MODE,
    POOL_CONTRACT_VERSION,
)
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder
try:
    from scripts.finalize_authoritative_meta_pool import _load_active_contract, _sha256
except ModuleNotFoundError:
    from finalize_authoritative_meta_pool import _load_active_contract, _sha256


def _strict_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1"}


def _parse_scan_time(candidate_id: pd.Series) -> pd.Series:
    parts = candidate_id.astype("string").str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    return pd.to_datetime(
        parts[0].fillna("") + parts[1].fillna(""),
        format="%Y%m%d%H%M%S",
        utc=True,
        errors="coerce",
    )


def finalize_fresh_pool(
    *,
    source_path: Path,
    run_manifest_path: Path,
    output_dir: Path,
    start_date: str,
    end_date: str,
    active_hmm_artifact_version: str,
    allow_active_feature_imputation: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace existing output directory: {output_dir}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    required_run = {
        "generation_mode": FRESH_FULL_POOL_GENERATION_MODE,
        "full_pool": True,
        "realism_profile": LEGACY_REALISM_PROFILE,
        "max_candidates": None,
        "hours": 7,
        "min_bars": 420,
    }
    for key, expected in required_run.items():
        if run_manifest.get(key) != expected:
            raise ValueError(
                f"Fresh full-pool run manifest mismatch for {key}: "
                f"expected={expected!r}, observed={run_manifest.get(key)!r}"
            )

    frame = pd.read_csv(source_path, low_memory=False)
    if frame.empty:
        raise ValueError("Fresh full-pool source is empty")
    expected_rows = run_manifest.get("successful_rows")
    if expected_rows is not None and int(expected_rows) != len(frame):
        raise ValueError(
            "Fresh full-pool source row count does not match its run manifest: "
            f"manifest={expected_rows}, source={len(frame)}"
        )
    if "candidate_id" not in frame.columns:
        raise ValueError("Fresh full-pool source lacks candidate_id")
    candidate_id = frame["candidate_id"].astype("string").str.strip()
    if bool(candidate_id.eq("").any()) or bool(candidate_id.duplicated(keep=False).any()):
        raise ValueError("Fresh full-pool source has blank or duplicate candidate_id values")

    scan_time = _parse_scan_time(candidate_id)
    start_timestamp = pd.Timestamp(start_date, tz="UTC")
    end_timestamp = pd.Timestamp(end_date, tz="UTC")
    if not isinstance(start_timestamp, pd.Timestamp) or not isinstance(
        end_timestamp, pd.Timestamp
    ):
        raise ValueError("Fresh full-pool window dates must be valid timestamps")
    start = start_timestamp.normalize()
    end_of_day = (
        end_timestamp.normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(microseconds=1)
    )
    if not isinstance(end_of_day, pd.Timestamp):
        raise ValueError("Fresh full-pool end date did not resolve to a timestamp")
    end = end_of_day
    if bool(scan_time.isna().any()) or bool((scan_time < start).fillna(True).any()) or bool((scan_time > end).fillna(True).any()):
        raise ValueError("Fresh full-pool source contains candidate IDs outside the requested window")

    contract = _load_active_contract(active_hmm_artifact_version)
    gated = UnifiedTrainingBuilder()._apply_ingestion_gate(frame)
    if "version_gated" not in gated.columns or bool(gated["version_gated"].astype(bool).any()):
        raise ValueError("Fresh full-pool source failed the current authority/version gate")
    if "is_authoritative" not in gated.columns or not bool(gated["is_authoritative"].map(_strict_true).all()):
        raise ValueError("Fresh full-pool source is not fully authoritative")
    if "realism_profile" not in gated.columns or not bool(gated["realism_profile"].astype(str).str.strip().str.lower().eq(LEGACY_REALISM_PROFILE).all()):
        raise ValueError("Fresh full-pool source is not the canonical legacy-authority profile")

    expected_hmm = {
        "hmm_artifact_version": active_hmm_artifact_version,
        "hmm_feature_source": "pinned_artifact_replay",
        "hmm_replay_scope": "hmm_lineage_only",
        "hmm_feature_semantics_version": HMM_FEATURE_SEMANTICS_VERSION,
    }
    for column, expected in expected_hmm.items():
        if column not in gated.columns or not bool(gated[column].astype(str).str.strip().eq(str(expected)).all()):
            raise ValueError(f"Fresh full-pool source failed active-HMM contract: {column}")
    missing_active_features: dict[str, int] = {}
    for feature in ACTIVE_SNAPSHOT_META_FEATURES:
        if feature not in gated.columns:
            missing_active_features[feature] = len(gated)
            continue
        numeric = pd.to_numeric(gated[feature], errors="coerce")
        missing_count = int((~np.isfinite(np.asarray(numeric, dtype=float))).sum())
        if missing_count:
            missing_active_features[feature] = missing_count
    if missing_active_features and not allow_active_feature_imputation:
        details = ", ".join(
            f"{feature}={count}" for feature, count in missing_active_features.items()
        )
        raise ValueError(
            "Fresh full-pool source has incomplete active features: "
            f"{details}. Re-run with explicit --allow-active-feature-imputation "
            "only when retraining will also pass --allow-imputation."
        )

    eligible = gated.copy()
    eligible["pool_contract_version"] = POOL_CONTRACT_VERSION
    eligible["generation_mode"] = FRESH_FULL_POOL_GENERATION_MODE
    final_name = f"training_data_{end.strftime('%Y%m%d')}.csv"
    temp_dir = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
        pool_path = temp_dir / final_name
        eligible.to_csv(pool_path, index=False)
        manifest = {
            "pool_contract_version": POOL_CONTRACT_VERSION,
            "generation_mode": FRESH_FULL_POOL_GENERATION_MODE,
            "full_pool": True,
            "source_path": str(source_path.resolve()),
            "source_sha256": _sha256(source_path),
            "run_manifest_path": str(run_manifest_path.resolve()),
            "run_manifest_sha256": _sha256(run_manifest_path),
            "active_hmm": contract,
            "feature_cutoff_contract": "candidate_id_scan_time",
            "window_start_utc": start.isoformat(),
            "window_end_utc": end.isoformat(),
            "input_rows": int(len(frame)),
            "eligible_rows": int(len(eligible)),
            "unique_eligible_candidate_ids": int(eligible["candidate_id"].nunique()),
            "selected_features": list(ACTIVE_SNAPSHOT_META_FEATURES),
            "allow_active_feature_imputation": bool(allow_active_feature_imputation),
            "missing_active_features": missing_active_features,
            "pool_file": final_name,
            "pool_sha256": _sha256(pool_path),
        }
        (temp_dir / AUTHORITATIVE_POOL_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_dir, output_dir)
        return manifest
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--active-hmm-artifact-version", required=True)
    parser.add_argument("--allow-active-feature-imputation", action="store_true")
    args = parser.parse_args()
    finalize_fresh_pool(
        source_path=args.source,
        run_manifest_path=args.run_manifest,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        active_hmm_artifact_version=args.active_hmm_artifact_version,
        allow_active_feature_imputation=args.allow_active_feature_imputation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
