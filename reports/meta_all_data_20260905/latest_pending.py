"""Audit late-arriving snapshots without inventing unfinished FASTWIN labels."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import cast

import joblib
import numpy as np
import pandas as pd

import experiment as ex
from inventory import sha
from neutralgrid.models.meta_labeler import normalize_inference_feature_frame


def main():
    source = ex.ROOT / "artifacts/pipeline_runs/default_250_20260906_232957/deployment_ready_20260906_233011.csv"
    manifest_path = source.parent / "validation/pipeline_run_manifest.json"
    frame = normalize_inference_feature_frame(pd.read_csv(source, low_memory=False))
    manifest = json.loads(manifest_path.read_text())
    saved_at = pd.Timestamp(datetime.fromtimestamp(source.stat().st_mtime, timezone.utc))
    generated_at = pd.Timestamp(manifest["generated_at_utc"])
    available = max(saved_at, generated_at)
    mature = available + pd.Timedelta(hours=7, minutes=2)
    now = pd.Timestamp.now(tz="UTC")
    assert now < mature, "New outcome windows have matured; replay them before reporting"
    assert not frame.candidate_id.duplicated().any()
    known = pd.read_csv(ex.OUT / "research_training.csv", usecols=lambda c: c == "candidate_id")
    assert not frame.candidate_id.isin(known.candidate_id).any()
    saved = joblib.load(ex.OUT / "best_research_candidate.joblib")
    for f in saved["features"]:
        if f not in frame:
            frame[f] = np.nan
        frame[f] = cast(pd.Series, pd.to_numeric(frame[f], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    raw = saved["model"].predict_proba(frame[saved["features"]].to_numpy(float))[:, 1]
    raw = np.clip(raw, 1e-6, 1-1e-6)
    probability = saved["calibrator"].predict_proba(np.log(raw / (1-raw)).reshape(-1, 1))[:, 1]
    geometry = frame[["grid_lower", "grid_upper", "num_grids"]].apply(pd.to_numeric, errors="coerce")
    valid = geometry.notna().all(axis=1) & geometry.grid_lower.gt(0) & geometry.grid_upper.gt(geometry.grid_lower) & geometry.num_grids.ge(2)
    ledger = frame[["candidate_id", "symbol"]].copy()
    ledger["recorded_geometry_valid"] = valid
    ledger["fast_winner_target"] = np.nan
    ledger["disposition"] = np.where(valid, "pending_full_7h_outcome", "missing_recorded_grid_geometry")
    ledger["available_utc"] = available
    ledger["earliest_complete_window_utc"] = mature
    ledger["research_probability"] = probability
    ledger["missing_selected_features"] = frame[saved["features"]].isna().sum(axis=1)
    ledger["production_eligible"] = False
    ledger.to_csv(ex.OUT / "latest_september6_snapshot_audit.csv", index=False)
    result = {"audited_utc": now, "source": str(source), "sha256": sha(source), "rows": len(frame),
        "scan_timestamp_utc": "2026-09-06T23:30:11+00:00", "recorded_available_utc": available,
        "earliest_complete_window_utc": mature, "valid_geometry_pending_rows": int(valid.sum()),
        "missing_geometry_rows": int((~valid).sum()), "new_usable_fastwin_labels": 0, "rows_scored": len(frame),
        "label_policy": "No partial-window result is treated as a loss. These scores do not alter the frozen selection or its independent holdout.",
        "source_inventory_update": "1206 files, 261 unique snapshot sources, 25655 unique persisted snapshot IDs after adding this 250-row file"}
    ex.write("latest_september6_snapshot_audit.json", result)
    text = (ex.OUT / "README.md").read_text()
    text = text.replace("\nCompleted ", "\nStatistical study completed ")
    old = "Source inventory was checked on September 6, 2026 (America/Lima). The latest available saved candidate scan is September 5 at 16:55:05 UTC. No September 6 candidate snapshot was present in the audited roots."
    new = "Source inventory was checked again on September 6, 2026 (America/Lima). The final check found a newly saved September 6 23:30:11 UTC snapshot: all 250 rows were audited and scored. Its 111 candidates with recorded geometry have unfinished seven-hour outcome windows; 139 lack geometry. They provide zero additional usable FASTWIN labels as of this report. The latest fully labelled scan used for testing remains September 5 at 16:55:05 UTC."
    text = text.replace(old, new)
    text = text.replace("The 1,205-file inventory contains exact duplicates and 260 unique snapshot files.", "The frozen 1,205-file inventory contained exact duplicates and 260 unique snapshot files. The final late-arrival audit increases that to 1,206 files and 261 unique snapshot sources.")
    text = text.replace("The evidence matrix, all model comparison results", f"Under the conservative availability rule, the late September 6 snapshot's seven-hour replay window finishes at {mature} (UTC). Its row-level scores and exclusions are in `latest_september6_snapshot_audit.csv`; these are diagnostic scores, not deployment instructions.\n\nThe evidence matrix, all model comparison results")
    (ex.OUT / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(ex.clean(result), default=str), flush=True)


if __name__ == "__main__":
    main()
