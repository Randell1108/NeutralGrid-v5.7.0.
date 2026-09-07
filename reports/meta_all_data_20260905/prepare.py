"""Freeze all accessible snapshots, old pool identities, and workbook rows."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["NEUTRALGRID_BASE_DIR"] = str(ROOT)
from neutralgrid.models.meta_labeler import normalize_inference_feature_frame
from neutralgrid.backtest.candidate_pipeline import _parse_scan_timestamp


def number(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def main():
    inventory = json.loads((OUT / "source_inventory.json").read_text())
    sources = [r for r in inventory["files"] if "duplicate_of" not in r and r.get("kind") == "snapshot"]
    # CSV representations precede XLSX; row order never depends on an outcome.
    sources.sort(key=lambda r: (Path(r["path"]).suffix != ".csv", r["path"]))
    frames = []
    for source in sources:
        path = Path(source["path"])
        df = pd.read_csv(path, low_memory=False) if path.suffix == ".csv" else pd.read_excel(path)
        if "candidate_id" not in df:
            # Old snapshots without a persisted ID are counted in inventory,
            # not assigned a made-up identity for the authoritative pool.
            continue
        df = normalize_inference_feature_frame(df)
        df["study_snapshot_path"] = str(path)
        df["study_snapshot_sha256"] = source["sha256"]
        df["study_snapshot_mtime"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    ids = combined.candidate_id.astype("string")
    combined = combined.loc[ids.notna() & ids.ne("")].copy()
    duplicate_count = int(combined.candidate_id.duplicated().sum())
    # Check conflicting grid geometry explicitly before resolving copies.
    conflicts = []
    duplicates = combined.loc[combined.candidate_id.duplicated(keep=False)]
    for cid, group in duplicates.groupby("candidate_id", sort=False):
        for col in ["grid_lower", "grid_upper", "num_grids"]:
            v = cast(pd.Series, pd.to_numeric(group[col], errors="coerce")).dropna() if col in group else pd.Series(dtype=float)
            if len(v) and not np.allclose(v.to_numpy(float), float(v.iloc[0]), rtol=1e-8, atol=1e-10):
                conflicts.append(str(cid))
                break
    combined = combined.drop_duplicates("candidate_id", keep="first").copy()
    combined["scan_time"] = pd.to_datetime(combined.candidate_id.map(lambda c: _parse_scan_timestamp(str(c))), utc=True)
    dispositions = []
    existing = pd.read_csv(OUT / "_tmp/existing_full_pool_union.csv", usecols=lambda c: c == "candidate_id")
    existing_ids = set(existing.candidate_id)
    queue = []
    now = pd.Timestamp.now(tz="UTC")
    for row in combined.to_dict("records"):
        cid = str(row["candidate_id"])
        reason = "new_outcome_required"
        numeric = [number(row.get(c)) for c in ["grid_lower", "grid_upper", "num_grids"]]
        if cid in existing_ids:
            reason = "existing_full_7h_outcome"
        elif cid in conflicts:
            reason = "conflicting_geometry_across_saved_snapshots"
        elif pd.isna(row["scan_time"]):
            reason = "invalid_persisted_candidate_timestamp"
        elif not all(pd.notna(v) and np.isfinite(v) for v in numeric):
            reason = "missing_recorded_grid_geometry"
        elif not (numeric[0] > 0 and numeric[1] > numeric[0] and numeric[2] >= 2):
            reason = "invalid_recorded_grid_geometry"
        elif row["scan_time"] > now:
            reason = "future_candidate_timestamp"
        if reason == "new_outcome_required":
            # New diagnostics explicitly anchor the outcome to the persisted
            # candidate scan timestamp. They are historical research replays.
            row["candidate_available_ts_utc"] = row["scan_time"].isoformat()
            row["candidate_available_source"] = "persisted_candidate_id_research_replay"
            row["study_source_pool"] = "new_snapshot_replay"
            queue.append(row)
        dispositions.append({"candidate_id": cid, "scan_time": row["scan_time"], "source_path": row["study_snapshot_path"], "disposition": reason})
    combined.to_pickle(OUT / "_tmp/all_unique_snapshots.pkl")
    pd.DataFrame(queue).to_pickle(OUT / "_tmp/snapshot_replay_queue.pkl")
    pd.DataFrame(dispositions).to_csv(OUT / "snapshot_row_dispositions.csv", index=False)
    workbook = pd.read_excel(ROOT / "data/new_expired_bots.xlsx")
    refreshed = pd.read_excel(ROOT / "data/new_expired_bots_backfilled_full_v2_20260904_hmm_rolling_180d_20260903_153527.xlsx")
    records = []
    snapshot_map = combined.set_index("candidate_id")
    for idx, (_, row) in enumerate(workbook.iterrows()):
        cid = str(row.get("candidate_id", ""))
        dur = number(row.get("duration_hours"))
        pnl = number(row.get("pnl_pct"))
        early_positive = pd.notna(dur) and 0 < dur <= 7 and pd.notna(pnl) and pnl >= 3
        records.append({"workbook_row": int(idx) + 2, "strategy_id": str(row.strategy_id), "candidate_id": cid,
            "symbol": row.symbol, "mode": row.get("mode"), "start_time_utc": row.start_time_utc,
            "duration_hours": dur, "recorded_pnl_pct": pnl,
            "exact_candidate_snapshot_found": cid in snapshot_map.index,
            "existing_7h_candidate_backtest_found": cid in existing_ids,
            "fastwin_actual_label": 1 if early_positive else None,
            "actual_label_evidence": "recorded endpoint at or above +3% within 7h; crossing time bounded, not known exactly" if early_positive else "path_time_evidence_required",
            "not_a_backtest_label": True})
    pd.DataFrame(records).to_csv(OUT / "workbook_row_dispositions.csv", index=False)
    # Preserve each workbook row for the real-outcome diagnostic; do not
    # silently replace an observed bot outcome with its simulated candidate.
    refreshed.to_pickle(OUT / "_tmp/workbook_features.pkl")
    workbook.to_pickle(OUT / "_tmp/workbook_original.pkl")
    summary = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "snapshot_sources": len(sources),
        "duplicate_snapshot_rows": duplicate_count, "unique_snapshot_candidate_ids": len(combined),
        "geometry_conflict_ids": len(set(conflicts)), "dispositions": pd.Series([r["disposition"] for r in dispositions]).value_counts().to_dict(),
        "replay_queue_rows": len(queue), "workbook_rows": len(records),
        "workbook_exact_snapshot_matches": sum(r["exact_candidate_snapshot_found"] for r in records),
        "workbook_existing_candidate_outcomes": sum(r["existing_7h_candidate_backtest_found"] for r in records),
        "workbook_provable_early_positive_labels": sum(r["fastwin_actual_label"] == 1 for r in records),
        "old_pool_index_rows": 6048,
        "old_pool_index_exact_snapshot_matches": int(pd.read_parquet(ROOT / "outputs/audits/selection_experiments2_20260708/training_pool_index.parquet").candidate_id.isin(set(combined.candidate_id)).sum())}
    (OUT / "prepared_sources.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
