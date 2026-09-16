"""Exclude inconsistent recorded timestamps; never synthesize replacement times."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent


def main() -> None:
    destination = OUT / "_tmp/scanners_validated"
    destination.mkdir(exist_ok=False)
    ledger = []
    exclusions = []
    for path in sorted((OUT / "_tmp/scanners").glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if "candidate_id" not in frame:
            ledger.append({"source": str(path), "reason": "missing_recorded_candidate_id", "rows": len(frame)})
            continue
        parts = frame.candidate_id.astype(str).str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
        scan = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True, errors="coerce")
        available = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
        # Canonical backtests prioritize an explicit recorded backtest start,
        # otherwise use the loader's original-file mtime. Check both boundaries.
        start = pd.Series(available, index=frame.index)
        if "backtest_start_ts_utc" in frame:
            explicit = pd.to_datetime(frame.backtest_start_ts_utc, utc=True, format="mixed", errors="coerce")
            start = explicit.fillna(available)
        valid = scan.notna() & scan.le(available) & scan.le(start)
        for index in frame.index[~valid]:
            exclusions.append({"source": str(path), "candidate_id": str(frame.at[index,"candidate_id"]), "scan_utc": str(scan.at[index]), "recorded_file_mtime_utc": str(available), "recorded_backtest_start_utc": str(start.at[index]), "reason": "invalid_scan_identity" if pd.isna(scan.at[index]) else "availability_or_start_before_scan"})
        record = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "input_rows": len(frame), "retained_rows": int(valid.sum()), "excluded_rows": int((~valid).sum()), "mtime_utc": str(available)}
        if valid.any():
            target = destination / path.name
            frame.loc[valid].to_csv(target,index=False)
            os.utime(target, (path.stat().st_atime,path.stat().st_mtime))
            record.update(destination=str(target), destination_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        ledger.append(record)
    pd.DataFrame(exclusions).to_csv(OUT / "timestamp_excluded_source_rows.csv",index=False)
    report = {"created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "policy": "Retain original first-inventory source identity and mtime. Require recorded scan <= recorded file availability and outcome start. Conflicting archive/recovery copies are not assigned a guessed replacement timestamp.", "original_files": len(ledger), "staged_files": sum("destination" in r for r in ledger), "excluded_source_rows": len(exclusions), "ledger": ledger}
    (OUT / "causal_staging_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="ledger"},indent=2),flush=True)


if __name__ == "__main__":
    main()
