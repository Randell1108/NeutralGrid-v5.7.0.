"""Inventory all accessible project data sources without changing originals."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["NEUTRALGRID_BASE_DIR"] = str(ROOT)
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES, normalize_inference_feature_frame


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    roots = [ROOT / n for n in ["data", "artifacts", "outputs"]]
    roots += [Path("D:/Neutral Grids") / n for n in ["data", "artifacts", "outputs"]]
    roots += [Path("D:/Deployment files/organized")]
    backup = Path("D:/Backup/Grid Bots")
    for project in sorted(backup.iterdir()):
        if project.is_dir():
            roots.extend(project / n for n in ["data", "results", "Live", "Expired", "expired"] if (project / n).exists())
    old = Path("D:/Backup/Christian/Crypto/Antigravity - NEUTRAL grid v1")
    roots.extend(old / n for n in ["data", "results"] if (old / n).exists())
    records, seen = [], {}
    paths = []
    for base in roots:
        if not base.exists():
            continue
        for directory, subdirs, files in os.walk(base):
            subdirs[:] = [n for n in subdirs if n not in {".git", ".venv", "node_modules", "__pycache__", "klines", "cache", "training_sets", "binance_vision"}]
            for name in files:
                path = Path(directory) / name
                lower = name.lower()
                if path.suffix.lower() not in {".csv", ".parquet", ".xlsx"}:
                    continue
                if lower.startswith("deployment_ready_") or any(k in lower for k in ["training", "backtest_results", "pool", "cohort", "new_expired", "pnl_curve", "final_pnl"]):
                    paths.append(path)
    for path in sorted(set(paths)):
        rec = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        if rec["sha256"] in seen:
            rec["duplicate_of"] = seen[rec["sha256"]]
            records.append(rec)
            continue
        seen[rec["sha256"]] = str(path)
        try:
            frame = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
            rec["rows"], rec["columns"] = frame.shape
            rec["column_names"] = list(frame.columns)
            rec["kind"] = "snapshot" if path.name.startswith("deployment_ready_") else "workbook" if "new_expired" in path.name else "outcome_or_audit"
            rec["has_fastwin_path_target"] = "time_to_target_hours" in frame
            if "time_to_target_hours" in frame:
                rec["finite_target_time"] = int(cast(pd.Series, pd.to_numeric(frame.time_to_target_hours, errors="coerce")).notna().sum())
            if "candidate_id" in frame:
                rec["unique_candidate_ids"] = int(frame.candidate_id.nunique())
            for c in ["hmm_artifact_version", "realism_profile", "mode", "label_contract_version", "formula_version"]:
                if c in frame:
                    rec[c] = {str(k): int(v) for k, v in frame[c].value_counts(dropna=False).items()}
            normalized = normalize_inference_feature_frame(frame)
            rec["active_feature_columns"] = sum(f in normalized for f in ACTIVE_SNAPSHOT_META_FEATURES)
        except Exception as exc:
            rec["read_error"] = str(exc)
        records.append(rec)
    result = {"inventory_utc": datetime.now(timezone.utc).isoformat(), "roots": [str(r) for r in roots], "files": records}
    (OUT / "source_inventory.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    brief = [{k: r.get(k) for k in ["path", "rows", "kind", "has_fastwin_path_target", "active_feature_columns"]} for r in records if "duplicate_of" not in r]
    print(json.dumps({"files": len(records), "unique_files": len(seen), "sources": brief}, indent=2), flush=True)


if __name__ == "__main__":
    main()
