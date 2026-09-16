"""Validate pinned features, then atomically update only workbook HMM fields."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from pandas.testing import assert_frame_equal
from neutralgrid.calibration.utility_calibrator import REQUIRED_FEATURE_COLS, OPTIONAL_FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HMM_COLUMNS = ["range_prob", "trend_prob", "persistence_prob", "hmm_trained_at_utc", "hmm_artifact_version", "hmm_pipeline_version", "hmm_feature_semantics_version", "hmm_feature_source", "hmm_replay_scope", "hmm_feature_cutoff_utc", "feature_cutoff_utc", "hmm_calibration_status", "utility_score", "ev_score", "regime_conf", "hmm_tail_cvar_95"]


def key(value) -> str:
    return str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value) and float(value).is_integer() else str(value)


def main() -> None:
    path = OUT / "utility_backfilled_entry_time.xlsx"
    flat = pd.read_excel(path)
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    columns = ["range_prob", "trend_prob", "persistence_prob"]
    probabilities = flat[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid = np.isfinite(probabilities).all(axis=1) & (probabilities >= 0).all(axis=1) & (probabilities <= 1).all(axis=1)
    valid &= flat.hmm_artifact_version.eq(version).to_numpy()
    valid &= flat.hmm_feature_source.eq("pinned_artifact_replay").to_numpy()
    cutoff = pd.to_datetime(flat.hmm_feature_cutoff_utc, utc=True, format="mixed")
    start = pd.to_datetime(flat.start_time_utc, utc=True, format="mixed")
    valid &= cutoff.eq(start).to_numpy()
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "active_hmm": version, "rows": len(flat), "valid_rows": int(valid.sum()), "invalid_rows": flat.loc[~valid, ["strategy_id", "symbol", "hmm_feature_source", *columns]].to_dict("records"), "versions": flat.hmm_artifact_version.value_counts(dropna=False).to_dict()}
    (OUT / "utility_lineage_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    assert valid.all(), "Incomplete replay: do not publish or claim all-row lineage PASS"
    assert flat.strategy_id.is_unique
    source = ROOT / "data/new_expired_bots.xlsx"
    backup = OUT / "backups/new_expired_bots_before.xlsx"
    assert source.read_bytes() == backup.read_bytes(), "Canonical workbook changed during this task"
    before = pd.read_excel(source, sheet_name=None)
    model_rows = {key(row["strategy_id"]): row for row in flat.to_dict("records")}
    wb = openpyxl.load_workbook(source)
    changed_columns = {}
    for sheet_name in ("General", "Meta Features"):
        ws = wb[sheet_name]
        headers = {cell.value: cell.column for cell in ws[1] if cell.value is not None}
        if sheet_name == "Meta Features":
            existing_ids = {key(ws.cell(n, headers["strategy_id"]).value) for n in range(2, ws.max_row + 1)}
            additional_columns = ["symbol", "start_time_utc", "candidate_id", *REQUIRED_FEATURE_COLS, *OPTIONAL_FEATURE_COLS]
            for column in additional_columns:
                if column not in headers:
                    headers[column] = ws.max_column + 1
                    ws.cell(1, headers[column], column)
            for ident, record in model_rows.items():
                if ident in existing_ids:
                    continue
                assert sorted(headers.values()) == list(range(1, len(headers) + 1)), "Unexpected blank header gap"
                values = []
                for column, number in sorted(headers.items(), key=lambda pair: pair[1]):
                    value = record.get(column)
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, np.generic):
                        value = value.item()
                    values.append(value)
                ws.append(values)
        selected = [c for c in HMM_COLUMNS if c in flat]
        if sheet_name == "Meta Features":
            selected += ["backfill_status"]
        for column in selected:
            if column not in headers:
                headers[column] = ws.max_column + 1
                ws.cell(1, headers[column], column)
        for row_number in range(2, ws.max_row + 1):
            ident = key(ws.cell(row_number, headers["strategy_id"]).value)
            assert ident in model_rows, (sheet_name, ident)
            record = model_rows[ident]
            for column in selected:
                value = "complete" if column == "backfill_status" else record[column]
                if pd.isna(value):
                    value = None
                elif isinstance(value, np.generic):
                    value = value.item()
                ws.cell(row_number, headers[column]).value = value
        changed_columns[sheet_name] = selected
    candidate = OUT / "canonical_workbook_candidate.xlsx"
    wb.save(candidate)
    after = pd.read_excel(candidate, sheet_name=None)
    assert list(before) == list(after)
    for sheet_name, original in before.items():
        untouched = [c for c in original if c not in changed_columns.get(sheet_name, [])]
        assert_frame_equal(original[untouched], after[sheet_name].iloc[:len(original)][untouched], check_dtype=False, check_exact=True)
    assert len(after["General"]) == len(flat)
    assert len(after["Meta Features"]) == len(flat)
    candidate.replace(source)
    report.update(status="PASS", workbook_updated=True, changed_columns=changed_columns, appended_meta_rows=len(after["Meta Features"])-len(before["Meta Features"]), new_meta_feature_provenance="Recorded General fields plus canonical grid geometry derivation; optional unavailable features remain missing", outcome_and_unrelated_cells_preserved=True, canonical_sha256_after=hashlib.sha256(source.read_bytes()).hexdigest(), canonical_sha256_before=hashlib.sha256(backup.read_bytes()).hexdigest())
    (OUT / "utility_lineage_audit.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
