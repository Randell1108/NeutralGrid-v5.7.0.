"""Prepare the 57 newly added bots for a full causal feature replay."""
from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES, normalize_inference_feature_frame
from scripts.backfill_training_features import BASELINE_BACKFILL_COLUMNS, LIVE_PLUS_BACKFILL_COLUMNS_V20260312, HMM_LINEAGE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "data/new_expired_bots.xlsx"
    backup = OUT / "backups/workbook_before_full_57.xlsx"
    assert not backup.exists()
    shutil.copy2(source, backup)
    wanted = set(json.loads((OUT / "workbook_intake_audit.json").read_text())["new_strategy_ids"])
    general = pd.read_excel(source, sheet_name="General")
    frame = general.loc[general.strategy_id.astype(str).isin(wanted)].copy().reset_index(drop=True)
    assert len(frame) == len(wanted) == 57 and frame.strategy_id.is_unique
    # Clear computed fields so an unsuccessful replay cannot masquerade as a
    # freshly reconstructed value. Preserve only recorded bot grid settings.
    for column in set(BASELINE_BACKFILL_COLUMNS + LIVE_PLUS_BACKFILL_COLUMNS_V20260312 + HMM_LINEAGE_COLUMNS + list(ACTIVE_SNAPSHOT_META_FEATURES)) - {"grid_spacing_pct", "num_grids"}:
        frame[column] = None if column in HMM_LINEAGE_COLUMNS or column == "ev_contract_fingerprint" else np.nan
    frame["num_grids"] = frame.grids_count
    evidence = []
    snapshot_frames = []
    wanted_candidates = set(frame.candidate_id.astype(str))
    for path in sorted((OUT / "_tmp/scanners_validated").glob("*.csv")):
        raw = pd.read_csv(path, low_memory=False)
        if "candidate_id" not in raw:
            continue
        selected = raw.loc[raw.candidate_id.astype(str).isin(wanted_candidates)]
        if selected.empty:
            continue
        selected = normalize_inference_feature_frame(selected).copy()
        selected["_source_path"] = str(path)
        selected["_source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected["_available"] = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        snapshot_frames.append(selected)
    snapshots = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    retained_fields = ["quote_volume_24h", "open_interest", "micro_round_trip_cost_pct", "primary_pipeline_score"]
    for index, bot in frame.iterrows():
        ident = str(bot.candidate_id)
        parts = ident.split("_")
        scan = pd.to_datetime(parts[1]+parts[2], format="%Y%m%d%H%M%S", utc=True)
        start = pd.to_datetime(bot.start_time_utc, utc=True)
        matches = snapshots.loc[snapshots.candidate_id.astype(str).eq(ident)].copy() if not snapshots.empty else pd.DataFrame()
        if len(matches):
            matches = matches.loc[matches._available.le(start) & (scan <= start)]
        record = {"strategy_id": str(bot.strategy_id), "candidate_id": ident, "bot_start_utc": str(start), "scan_utc": str(scan), "eligible_snapshot_rows": len(matches), "fields": {}}
        for field in retained_fields:
            values = pd.to_numeric(matches[field], errors="coerce") if field in matches else pd.Series(dtype=float)
            values = values.loc[np.isfinite(values)]
            if len(values) and np.isclose(values, values.iloc[0], rtol=1e-12, atol=1e-14).all():
                frame.at[index, field] = float(values.iloc[0])
                row = matches.loc[values.index[0]]
                record["fields"][field] = {"value": float(values.iloc[0]), "source": row._source_path, "sha256": row._source_sha256, "recorded_availability_utc": str(row._available), "method": "Exact candidate snapshot available before bot start; not a fabricated historical ticker/order book"}
            else:
                record["fields"][field] = {"status": "missing_or_conflicting_verified_snapshot", "finite_observations": len(values)}
        evidence.append(record)
    frame.to_excel(OUT / "full_57_input.xlsx", index=False)
    (OUT / "full_57_snapshot_provenance.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({"rows": len(frame), "verified_snapshot_coverage": {f:int(frame[f].notna().sum()) for f in retained_fields}}, indent=2))


if __name__ == "__main__":
    main()
