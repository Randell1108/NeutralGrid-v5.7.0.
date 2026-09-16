"""Verify the replacement full pool and recovery of incomplete outcome windows."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from neutralgrid.core.constants import ENGINE_VERSION, FORMULA_VERSION, LABEL_CONTRACT_VERSION

OUT = Path(__file__).resolve().parent


def main() -> None:
    manifest = json.loads((OUT / "fresh_production_pool/backtest_run_manifest.json").read_text())
    path = Path(manifest["training_file"])
    frame = pd.read_csv(path, low_memory=False)
    assert frame.candidate_id.is_unique
    assert len(frame) == manifest["successful_rows"] == manifest["union_count_after_window"]
    assert manifest["full_pool"] and manifest["max_candidates"] is None
    assert manifest["generation_mode"] == "fresh_full_pool" and manifest["hours"] == 7 and manifest["min_bars"] == 420
    for column, expected in {"engine_version": ENGINE_VERSION, "formula_version": FORMULA_VERSION, "label_contract_version": LABEL_CONTRACT_VERSION, "realism_profile": "legacy"}.items():
        assert frame[column].eq(expected).all(), column
    parts = frame.candidate_id.str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scan = pd.to_datetime(parts[0]+parts[1], format="%Y%m%d%H%M%S", utc=True)
    start = pd.to_datetime(frame.start_time_utc, utc=True, format="mixed")
    assert scan.le(start).all()
    skipped = json.loads((OUT / "initial_incomplete_windows.json").read_text())["skipped_rows"]
    assert {r["candidate_id"] for r in skipped} <= set(frame.candidate_id)
    initial = pd.read_csv(OUT / "initial_incomplete_pool/training_data_20260915.csv", low_memory=False).set_index("candidate_id")
    repeated = frame.set_index("candidate_id").loc[initial.index]
    old_y = pd.to_numeric(initial.time_to_target_hours, errors="coerce").le(7)
    new_y = pd.to_numeric(repeated.time_to_target_hours, errors="coerce").le(7)
    outcome_changes = int((~np.isclose(pd.to_numeric(initial.pnl_pct), pd.to_numeric(repeated.pnl_pct), rtol=1e-10, atol=1e-12, equal_nan=True)).sum())
    result = {"status": "PASS", "selected_rows": manifest["union_count_after_window"], "successful_rows": len(frame), "recovered_incomplete_windows": len(skipped), "all_ids_unique": True, "all_scan_times_no_later_than_outcome_start": True, "earliest_scan_utc": str(scan.min()), "latest_scan_utc": str(scan.max()), "latest_outcome_start_utc": str(start.max()), "fastwin_positive_rows": int(pd.to_numeric(frame.time_to_target_hours, errors="coerce").le(7).sum()), "repeated_initial_rows": len(initial), "repeated_label_changes": int(old_y.ne(new_y).sum()), "repeated_net_pnl_changes": outcome_changes, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / "production_raw_pool_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
