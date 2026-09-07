"""Audit actual bot evidence; retain unknown outcomes without inventing labels."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES, normalize_inference_feature_frame
from inventory import sha


def utc(value: Any) -> Any:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return pd.Timestamp("1899-12-30", tz="UTC") + pd.Timedelta(days=float(value))
    return pd.to_datetime(value, utc=True, errors="coerce")


def main():
    original = cast(pd.DataFrame, pd.read_pickle(OUT / "_tmp/workbook_original.pkl"))
    features = pd.read_pickle(OUT / "_tmp/workbook_features.pkl").set_index("strategy_id")
    snapshots = pd.read_pickle(OUT / "_tmp/all_unique_snapshots.pkl").set_index("candidate_id")
    evidence = []
    for row in original.to_dict("records"):
        start, end = utc(row["start_time_utc"]), utc(row["end_time_utc"])
        elapsed = (end - start).total_seconds() / 3600 if pd.notna(start) and pd.notna(end) else np.nan
        pnl = float(cast(pd.Series, pd.to_numeric(pd.Series([row["pnl_pct"]]), errors="coerce")).iloc[0])
        # Require both recorded duration and exact timestamp endpoint to be
        # inside the target window; rounding alone cannot establish a label.
        yes = 0 < elapsed <= 7 and 0 < row["duration_hours"] <= 7 and pd.notna(pnl) and pnl >= 3
        evidence.append({"strategy_id": row["strategy_id"], "candidate_id": row.get("candidate_id"),
            "symbol": row["symbol"], "mode": row["mode"], "scan_time": start, "event_end": start + pd.Timedelta(hours=7),
            "actual_end_time": end, "elapsed_hours": elapsed, "recorded_duration_hours": row["duration_hours"],
            "recorded_pnl_pct": pnl, "fast_winner_target": 1. if yes else np.nan,
            "label_evidence": "net terminal profit >=3% within exact seven-hour window" if yes else "unknown: full timed net-MTM path not established",
            "label_source": str(ROOT / "data/new_expired_bots.xlsx"), "time_to_target_hours": np.nan})
    data = pd.DataFrame(evidence)
    # These two archived files explicitly record hourly observed net PnL.
    # A threshold-crossing observation proves a positive; absent observations
    # cannot prove a negative or the exact first crossing time.
    backup = Path("D:/Backup/Grid Bots/NEUTRAL grid bot v6.5.1/Live/02-03")
    curves = [(409845187, backup / "TRUMPUSDT/final_pnl_curve.csv"),
              (409845158, backup / "final_pnl_curve_xrp.csv")]
    curve_records = []
    for sid, path in curves:
        curve = pd.read_csv(path)
        margin = float(original.set_index("strategy_id").loc[sid, "invested_margin_usdt"])
        crossed = curve.loc[(curve.Hour > 0) & (curve.Hour <= 7) & (curve.PnL_USDT >= .03 * margin)]
        if len(crossed):
            data.loc[data.strategy_id == sid, "fast_winner_target"] = 1.
            data.loc[data.strategy_id == sid, "label_evidence"] = "observed hourly net PnL >=3% inside seven hours; exact first crossing unknown"
            data.loc[data.strategy_id == sid, "label_source"] = str(path)
        curve_records.append({"strategy_id": sid, "source": str(path), "sha256": sha(path),
            "positive_proven": len(crossed) > 0, "first_observed_crossing_upper_bound_hours": float(crossed.Hour.min()) if len(crossed) else None})
    # Stored workbook measurements and recorded grid geometry are diagnostic
    # inputs; their historical availability is not independently certified.
    # New retrospective backfills are not substituted as original observations.
    technical = ["adx_1h", "adx_15m", "adx_5m", "rsi_15m", "ema_slope_1h", "ema_crosses_5m", "vwap_crosses_5m", "range_size_pct", "bb_width", "grid_spacing_pct"]
    provenance = []
    for f in ACTIVE_SNAPSHOT_META_FEATURES:
        data[f] = np.nan
    for i, row in enumerate(original.to_dict("records")):
        for f in technical:
            data.loc[i, f] = pd.to_numeric(row.get(f), errors="coerce")
        data.loc[i, "num_grids"] = row["grids_count"]
        cid = row.get("candidate_id")
        found = isinstance(cid, str) and cid in snapshots.index
        if found:
            snap = normalize_inference_feature_frame(snapshots.loc[[cid]]).iloc[0]
            for f in ACTIVE_SNAPSHOT_META_FEATURES:
                if f != "ev_score" and pd.isna(data.loc[i, f]):
                    data.loc[i, f] = pd.to_numeric(snap.get(f), errors="coerce")
        provenance.append({"strategy_id": row["strategy_id"], "exact_snapshot_match": found,
            "observed_feature_count": int(data.loc[i, list(ACTIVE_SNAPSHOT_META_FEATURES)].notna().sum()),
            "backfill_hmm_version": str(features.loc[row["strategy_id"], "hmm_artifact_version"]),
            "retrospective_backfill_used_as_original_observation": False})
    data["study_source_pool"] = "actual_bot_positive_bound"
    data.to_csv(OUT / "workbook_actual_evidence.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT / "workbook_feature_provenance.csv", index=False)
    # Account for auxiliary historical sources omitted by the first filename
    # filter. Their terminal/legacy labels are not relabelled as FASTWIN.
    old_sources = []
    paths = []
    for project in Path("D:/Backup/Grid Bots").iterdir():
        base = project / "meta-labeling"
        if base.exists():
            paths.extend(base.rglob("*.csv"))
    old = Path("D:/Backup/Christian/Crypto/Antigravity - NEUTRAL grid v1/data")
    if old.exists():
        paths.extend(p for p in old.iterdir() if "expired" in p.name and p.suffix in {".csv", ".xlsx"})
    seen = set()
    for path in paths:
        digest = sha(path)
        if digest in seen:
            continue
        seen.add(digest)
        frame = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path, low_memory=False)
        idcol = next((c for c in ["strategy_id", "strategy_number"] if c in frame), None)
        old_sources.append({"path": str(path), "sha256": digest, "rows": len(frame), "columns": list(frame.columns),
            "has_time_to_target": "time_to_target_hours" in frame,
            "matched_workbook_ids": int(frame[idcol].astype(str).isin(original.strategy_id.astype(str)).sum()) if idcol else None,
            "disposition": "legacy terminal-label records; not an independently verifiable FASTWIN path"})
    summary = {"audited_utc": datetime.now(timezone.utc).isoformat(), "rows": len(data),
        "known_positive_labels": int(data.fast_winner_target.eq(1).sum()), "known_negative_labels": 0,
        "unknown_labels": int(data.fast_winner_target.isna().sum()),
        "grid_modes": original["mode"].value_counts().to_dict(), "curves": curve_records,
        "original_sha256": sha(ROOT / "data/new_expired_bots.xlsx"), "additional_legacy_sources": old_sources,
        "policy": "Unknown outcomes remain unknown; known positives may be tested only as explicitly biased auxiliary training data. All 366 rows are retained for scoring and coverage."}
    (OUT / "workbook_audit.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "additional_legacy_sources"}, default=str), flush=True)


if __name__ == "__main__":
    main()
