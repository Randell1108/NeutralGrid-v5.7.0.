"""Stage original scanner snapshots for a new canonical backtest run."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    previous = ROOT / "reports/meta_canonical_refit_20260907"
    roots = json.loads((previous / "source_inventory.json").read_text())["roots"]
    target = OUT / "_tmp/scanners"
    target.mkdir(parents=True, exist_ok=False)
    (OUT / "_tmp/runtime").mkdir()
    sources, errors = [], []
    for root in roots:
        if not Path(root).exists():
            continue
        for directory, _, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
            for name in files:
                if name.lower().startswith("deployment_ready") and Path(name).suffix.lower() in {".csv", ".xlsx"}:
                    sources.append(Path(directory) / name)
    assert not errors, errors
    seen, records = {}, []
    for index, source in enumerate(sorted(set(sources))):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        record = {"source": str(source), "sha256": digest, "mtime_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat()}
        if digest in seen:
            record.update(disposition="identical_bytes_duplicate", duplicate_of=seen[digest])
            records.append(record)
            continue
        seen[digest] = str(source)
        stamp = re.search(r"(\d{8}_\d{6})", source.stem)
        if stamp is None:
            probe = pd.read_excel(source) if source.suffix.lower() == ".xlsx" else pd.read_csv(source)
            if "candidate_id" not in probe or not probe.candidate_id.astype(str).str.match(r"^[^_]+_\d{8}_\d{6}(?:_|$)").any():
                record["disposition"] = "undated_export_excluded; no verified candidate timestamp"
                records.append(record)
                continue
        suffix = stamp.group(1) if stamp else "recorded_ids"
        destination = target / f"deployment_ready_{index:05d}_{suffix}.csv"
        if source.suffix.lower() == ".csv":
            shutil.copy2(source, destination)
            record["disposition"] = "original_bytes_and_mtime"
        else:
            frame = pd.read_excel(source)
            if "candidate_id" not in frame or "symbol" not in frame:
                record["disposition"] = "xlsx_missing_identity_columns"
                records.append(record)
                continue
            frame.to_csv(destination, index=False)
            os.utime(destination, (source.stat().st_atime, source.stat().st_mtime))
            record.update(disposition="xlsx_converted_with_original_mtime", rows=len(frame))
        record.update(staged_path=str(destination), staged_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        records.append(record)
    manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "roots": roots, "files": records, "scan_errors": errors}
    (OUT / "source_inventory.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source_files": len(records), "staged_files": sum("staged_path" in r for r in records), "dispositions": pd.Series([r["disposition"] for r in records]).value_counts().to_dict()}), flush=True)


if __name__ == "__main__":
    main()
