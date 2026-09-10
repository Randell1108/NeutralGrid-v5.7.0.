"""Freeze verified scanner inputs and a test protocol without reading new outcomes."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from neutralgrid.backtest.candidate_pipeline import resolve_backtest_start_timestamp
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "reports/meta_canonical_refit_20260907"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protected = [ROOT / "artifact_manifest.json", ROOT / "data/new_expired_bots.xlsx", ROOT / "models/meta_labeler.pkl"]
    protected += list((ROOT / "data/profile").glob("*"))
    protected += list((ROOT / "models/meta_labeler").glob("*"))
    (OUT / "protected_files_before.json").write_text(json.dumps([
        {"path": str(p), "sha256": sha(p)} for p in protected if p.is_file()
    ], indent=2) + "\n")
    inventory = json.loads((PREVIOUS / "source_inventory.json").read_text())
    lookup = {Path(r["staged_path"]).name: r for r in inventory["files"] if r.get("staged_path")}
    lookup.update({Path(r["staged_path"]).name: {"source": r["source"], "sha256": r["source_sha256"]}
                   for r in json.loads((PREVIOUS / "excel_conversion_manifest.json").read_text())})
    audit = pd.read_parquet(PREVIOUS / "loaded_snapshot_inventory.parquet")
    required = [*DEFAULT_FEATURES, "grid_lower", "grid_upper", "num_grids"]
    finite = np.isfinite(audit[required].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    proposed = audit.loc[finite].copy()
    payloads, rejected, sources = [], [], []
    for filename, group in proposed.groupby("scan_file", sort=True):
        record = lookup[str(filename)]
        source = Path(record["source"])
        assert sha(source) == record["sha256"], f"Source changed: {source}"
        original = pd.read_excel(source) if source.suffix.lower() == ".xlsx" else pd.read_csv(source, low_memory=False)
        if "candidate_id" not in original:
            for cid in group.candidate_id:
                rejected.append({"candidate_id": cid, "reason": "original_source_lacks_candidate_id"})
            continue
        original["candidate_id"] = original.candidate_id.astype(str)
        sources.append({"path": str(source), "sha256": sha(source)})
        for _, audit_row in group.iterrows():
            cid = str(audit_row.candidate_id)
            matches = original.loc[original.candidate_id.eq(cid)]
            if len(matches) != 1:
                rejected.append({"candidate_id": cid, "reason": "nonunique_original_source_match"})
                continue
            row = matches.iloc[0].to_dict()
            if "\ufffd" in cid or "\ufffd" in str(row.get("symbol")):
                rejected.append({"candidate_id": cid, "reason": "corrupted_symbol_text"})
                continue
            parts = pd.Series([cid]).str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
            scan = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True, errors="coerce").iloc[0]
            if pd.isna(scan):
                rejected.append({"candidate_id": cid, "reason": "unparseable_scan_identity"})
                continue
            row["candidate_available_ts_utc"] = str(audit_row.candidate_available_ts_utc)
            row["scan_file"] = str(filename)
            row["scan_timestamp"] = scan.isoformat()
            row["original_source_path"] = str(source)
            row["original_source_sha256"] = record["sha256"]
            start = pd.Timestamp(resolve_backtest_start_timestamp(row))
            if start < scan:
                rejected.append({"candidate_id": cid, "reason": "source_available_before_scan"})
                continue
            if start + pd.Timedelta(hours=7) > pd.Timestamp.now(tz="UTC"):
                rejected.append({"candidate_id": cid, "reason": "outcome_window_not_mature"})
                continue
            for col in required:
                assert np.isclose(float(row[col]), float(audit_row[col]), rtol=1e-10, atol=1e-12), (cid, col)
            raw_mode = row.get("mode")
            if pd.isna(raw_mode) or raw_mode is None:
                raw_mode = row.get("backtest_mode")
            if raw_mode is not None and not pd.isna(raw_mode) and str(raw_mode).lower() != "geometric":
                rejected.append({"candidate_id": cid, "reason": "non_geometric_recorded_mode"})
                continue
            if not (0 < float(row["grid_lower"]) < float(row["grid_upper"]) and float(row["num_grids"]) >= 2):
                rejected.append({"candidate_id": cid, "reason": "invalid_recorded_geometry"})
                continue
            row["start_time_utc"] = start.isoformat()
            payloads.append(row)
    rows = pd.DataFrame(payloads)
    assert rows.candidate_id.is_unique
    starts = pd.to_datetime(rows.start_time_utc, utc=True, format="ISO8601")
    groups = sorted(starts.unique())
    boundary = pd.Timestamp(groups[int(len(groups) * .8)])
    split = np.where(starts >= boundary, "holdout", np.where(starts + pd.Timedelta(hours=24) < boundary, "development", "boundary_purge"))
    rows["split"] = split
    rows = rows.sort_values(["start_time_utc", "candidate_id"])
    with (OUT / "frozen_candidate_payloads.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
    columns = ["candidate_id", "symbol", "start_time_utc", "scan_timestamp", "split", "original_source_path", "original_source_sha256", *DEFAULT_FEATURES]
    rows[columns].to_csv(OUT / "frozen_feature_cohort.csv", index=False)
    pd.DataFrame(rejected).to_csv(OUT / "input_rejections.csv", index=False)
    # Only candidate identities are read here. Outcome values remain outside freeze.
    previous_ids = set(pd.read_csv(PREVIOUS / "fresh_all_outcomes/training_data_20260907.csv", usecols=["candidate_id"]).candidate_id.astype(str))
    new_ids = sorted(set(rows.candidate_id) - previous_ids)
    (OUT / "additional_candidate_ids.json").write_text(json.dumps(new_ids, indent=2) + "\n")
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "isolated_profile_training",
        "canonical_workbook_label": "completed duration <7h; profit_factor >=1.5 and training-only PnL quantile .68",
        "expanded_shadow_label": "time_to_3pct_target <=7h", "backtest_realism_profile": "legacy",
        "backtest_hours": 7, "capital": 400, "leverage": 10, "max_concurrency": 4,
        "feature_contract": list(DEFAULT_FEATURES), "features_source": "original recorded scanner rows, source hashes checked",
        "selected_rows": len(rows), "proposed_complete_geometry_rows": len(proposed), "rejected_rows": len(rejected),
        "rows_by_split": rows.split.value_counts().to_dict(), "holdout_start_utc": boundary.isoformat(),
        "split_unit": "recorded outcome-start timestamp group", "purge_hours": 24,
        "outer_folds": 5, "inner_folds": 4, "initial_train_group_fraction": .5,
        "gaussian_shrinkage_grid": [0.0, 0.1, 0.3, 0.6, 0.9], "logistic_c_grid": [0.01, 0.1, 1.0, 10.0],
        "selection": "nested development mean AUC, then lowest Brier as tie-break; choose one recipe before opening holdout",
        "holdout_evaluation": "one selected frozen recipe versus preregistered Gaussian shrinkage .30 and train-prevalence baseline; no tuning after disclosure",
        "ablation": "development-only Gaussian excluding funding is diagnostic and cannot satisfy four-feature promotion coverage",
        "promotion_policy": "unchanged existing gates; expanded target remains shadow without canonical target migration and prospective evidence",
        "reuse": {"existing_fresh_rows": len(rows) - len(new_ids), "new_candidates_to_backtest": len(new_ids), "previous_manifest": str(PREVIOUS / "fresh_all_outcomes/backtest_run_manifest.json")},
        "holdout_disclosure_caveat": "Some seven-hour outcomes were previously analyzed for the meta-labeler; this is a frozen profile comparison, not wholly unseen prospective evidence.",
        "sources": sources,
    }
    (OUT / "preregistration.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: v for k, v in protocol.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
