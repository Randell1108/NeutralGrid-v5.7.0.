"""Build a deduplicated, provenance-preserving research matrix."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

import experiment as ex
import retrain_meta_labeler as rml
from neutralgrid.backtest.candidate_pipeline import _parse_scan_timestamp
from inventory import sha


def main():
    out, root = ex.OUT, ex.ROOT
    a = pd.read_csv(out / "existing_pool_active_hmm.csv", low_memory=False)
    b = pd.read_csv(out / "new_replay_active_hmm.csv", low_memory=False)
    assert not set(a.candidate_id) & set(b.candidate_id)
    data = pd.concat([a, b], ignore_index=True)
    assert not data.candidate_id.duplicated().any()
    active = json.loads((root / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    assert set(data.hmm_artifact_version) == {active}
    assert np.isfinite(data[["range_prob", "trend_prob", "persistence_prob"]].to_numpy(float)).all()
    # Preserve the current-HMM features, but use the new availability-anchored
    # outcome for the independent latest cohort. Never duplicate both labels.
    availability = pd.read_csv(out / "availability/new_replay_training.csv", low_memory=False)
    outcome_fields = ["start_time_utc", "t1", "duration_hours", "time_to_target_hours", "target_reached", "horizon_censored", "pnl_pct", "y", "sl_hit"]
    data = data.set_index("candidate_id")
    aligned = availability.set_index("candidate_id")
    for col in outcome_fields:
        data.loc[aligned.index, col] = aligned[col]
    data.loc[aligned.index, "study_source_pool"] = "latest_after_artifact_availability"
    data = data.reset_index()
    data["net_pnl_pct"] = pd.to_numeric(data.pnl_pct, errors="coerce")
    data["scan_time"] = pd.to_datetime(data.candidate_id.map(lambda s: _parse_scan_timestamp(str(s))), utc=True)
    data["event_end"] = pd.concat([pd.to_datetime(data.t1, utc=True, format="mixed"), pd.to_datetime(data.start_time_utc, utc=True, format="mixed") + pd.Timedelta(hours=7)], axis=1).max(axis=1)
    data["legacy_timestamp_uncertainty"] = data.scan_time < pd.Timestamp("2026-04-01", tz="UTC")
    # Older producer code uses datetime.now() without an offset. Retain these
    # counterfactual outcomes for an explicitly labelled sensitivity, not the
    # main selection population.
    assert data.max_holding_bars.eq(420).all()
    assert data.realism_profile.eq("legacy").all()
    assert data.horizon_censored.eq(False).all()
    assert data.time_to_target_hours.notna().eq(data.target_reached.eq(True)).all()
    frame, target_summary = rml.prepare_fast_target_training_frame(data, pnl_col="net_pnl_pct")
    assert len(frame) == len(data)
    assert frame.fast_winner_target.eq(frame.time_to_target_hours.le(7).astype(int)).all()
    assert frame.event_end.notna().all() and frame.scan_time.notna().all()
    coverage = []
    for f in ex.ALL:
        if f not in frame:
            frame[f] = np.nan
        frame[f] = cast(pd.Series, pd.to_numeric(frame[f], errors="coerce")).replace([np.inf, -np.inf], np.nan)
        for source, group in frame.groupby("study_source_pool"):
            coverage.append({"feature": f, "source": source, "rows": len(group), "finite_rows": int(group[f].notna().sum())})
    keep = ["candidate_id", "symbol", "scan_time", "event_end", "start_time_utc", "study_source_pool", "mode",
        "legacy_timestamp_uncertainty", "fast_winner_target", "time_to_target_hours", "target_reached", "horizon_censored",
        "net_pnl_pct", "duration_hours", "hmm_artifact_version", "hmm_trained_at_utc", "hmm_feature_semantics_version",
        "range_prob", "trend_prob", "persistence_prob"] + ex.ALL
    cast(pd.DataFrame, frame[keep]).sort_values(["scan_time", "candidate_id"]).to_csv(out / "research_training.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "feature_coverage.csv", index=False)
    inv = json.loads((out / "source_inventory.json").read_text())
    source_ledger = []
    ids = set(frame.candidate_id)
    for source in inv["files"]:
        if "duplicate_of" in source:
            continue
        path = Path(source["path"])
        disposition = "audited source"
        if source.get("kind") == "snapshot" and "candidate_id" not in source.get("column_names", []):
            disposition = "no persisted candidate identity; old filename producer has no timezone offset; cannot reconstruct a verified event time"
        elif source.get("kind") == "snapshot":
            disposition = "row-level accounting in snapshot_row_dispositions.csv"
        elif source.get("kind") == "workbook":
            disposition = "all historical workbook identities compared; current 366 and additional 27 archived identities audited separately"
        elif "index" in path.name:
            disposition = "identity-only index; no features or FASTWIN path labels"
        elif "shadow_approved" in str(path):
            disposition = "selection-filtered six-hour diagnostic; full seven-hour replay preferred for matching IDs"
        elif not source.get("has_fastwin_path_target"):
            disposition = "no endogenous FASTWIN path target; use recovered snapshots where available"
        matched = None
        if source.get("kind") == "outcome_or_audit" and "candidate_id" in source.get("column_names", []):
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, usecols=lambda c: c == "candidate_id")
            matched = int(df.candidate_id.isin(ids).sum())
        source_ledger.append({"path": str(path), "sha256": source["sha256"], "rows": source.get("rows"), "disposition": disposition, "matched_research_ids": matched})
    pd.DataFrame(source_ledger).to_csv(out / "all_source_dispositions.csv", index=False)
    summary = {"assembled_utc": datetime.now(timezone.utc).isoformat(), "existing_rows": len(a), "new_replay_rows": len(b),
        "total_unique_rows": len(frame), "availability_anchored_holdout_rows": len(availability),
        "primary_timestamp_eligible_rows": int((~frame.legacy_timestamp_uncertainty).sum()),
        "legacy_timestamp_sensitivity_rows": int(frame.legacy_timestamp_uncertainty.sum()),
        "source_counts": frame.study_source_pool.value_counts().to_dict(), "target_summary": target_summary,
        "active_hmm": active, "uniform_lineage_and_finite_probabilities": True,
        "period_start": frame.scan_time.min(), "period_end": frame.scan_time.max(),
        "original_source_hashes": {str(p): sha(p) for p in [root / "data/new_expired_bots.xlsx", root / "artifact_manifest.json", root / "src/neutralgrid/models/meta_labeler.py"]},
        "retrospective_hmm_rows": int((pd.to_datetime(frame.hmm_trained_at_utc, utc=True, format="mixed") > frame.scan_time).sum()),
        "lineage_limit": "Uniform active-HMM lineage does not establish point-in-time HMM training. Ranked profiles omit EV; fixed EV profiles are conditional diagnostics.",
        "timing_limit": "Historical development follows existing scan-anchored research contracts. Latest holdout starts only after saved output availability. Legacy February-March timestamps are excluded from main selection and retained for sensitivity."}
    ex.write("assembled_data_audit.json", summary)
    print(json.dumps(ex.clean(summary), default=str), flush=True)


if __name__ == "__main__":
    main()
