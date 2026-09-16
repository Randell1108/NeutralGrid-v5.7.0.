"""Preserve canonical General outcomes and Meta Features for a pinned replay."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from neutralgrid.calibration.utility_calibrator import REQUIRED_FEATURE_COLS, OPTIONAL_FEATURE_COLS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "data/new_expired_bots.xlsx"
    backup = OUT / "backups"
    backup.mkdir(exist_ok=False)
    shutil.copy2(source, backup / "new_expired_bots_before.xlsx")
    general = pd.read_excel(source, sheet_name="General")
    meta = pd.read_excel(source, sheet_name="Meta Features")
    assert general.strategy_id.is_unique and meta.strategy_id.is_unique
    assert set(general.strategy_id) == set(meta.strategy_id)
    meta = meta.set_index("strategy_id").reindex(general.strategy_id).reset_index()
    flat = general.copy()
    # The utility loader's contract explicitly takes these features from Meta
    # Features. General remains authoritative for actual outcome and start time.
    columns = list(REQUIRED_FEATURE_COLS) + list(OPTIONAL_FEATURE_COLS)
    columns += [c for c in meta if c.startswith("hmm_") or c in {"persistence_prob", "feature_cutoff_utc", "ev_score", "regime_conf"}]
    for column in dict.fromkeys(columns):
        if column in meta:
            flat[column] = meta[column].to_numpy()
    flat = flat.drop(columns=["backfill_status"], errors="ignore")
    destination = OUT / "utility_input.xlsx"
    flat.to_excel(destination, index=False)
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": len(flat), "outcomes_and_cutoff": "General", "utility_features": "Meta Features", "feature_columns_copied": sorted(set(columns) & set(meta)), "backfill_status": "Derived from the result of the forthcoming pinned HMM replay", "destination": str(destination)}
    (OUT / "utility_input_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
