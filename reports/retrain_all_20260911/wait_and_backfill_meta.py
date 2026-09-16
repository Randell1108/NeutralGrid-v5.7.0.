"""Replay fresh backtest rows against the completed canonical HMM selection."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    path = OUT / "fresh_full_pool/backtest_run_manifest.json"
    while not path.exists() or not (OUT / "active_hmm_validation.json").exists() or not (OUT / "utility_backfilled.xlsx").exists():
        time.sleep(5)
    manifest = json.loads(path.read_text())
    assert manifest["generation_mode"] == "fresh_full_pool" and manifest["full_pool"]
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    source = Path(manifest["training_file"])
    args = [sys.executable, str(ROOT / "scripts/backfill_training_features.py"), "--input", str(source), "--output", str(OUT / "fresh_meta_backfilled.csv"), "--default-artifact-version", version, "--hmm-only", "--feature-cutoff-source", "candidate_id_scan_time", "--replay-scope", "hmm_lineage_only", "--require-fresh-output", "--max-concurrency", "4"]
    (OUT / "meta_backfill_command.json").write_text(json.dumps(args, indent=2) + "\n")
    with (OUT / "meta_backfill.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(args, cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.returncode
    print("Fresh meta pool replay completed; inspect lineage and finalize before fitting.", flush=True)


if __name__ == "__main__":
    main()
