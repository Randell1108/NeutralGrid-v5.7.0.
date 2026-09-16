"""Start the staged workbook replay only after the canonical HMM run finishes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    original = json.loads((OUT / "artifact_manifest_before.json").read_text())["hmm"]["active_version"]
    while True:
        log = (OUT / "hmm_retrain.log").read_text(encoding="utf-8", errors="replace")
        if "Done. Run 'python run_full_pipeline.py'" in log:
            break
        if any(token in log for token in ("Canonical pipeline failed:", "Fatal error:", "Training interrupted by user")):
            raise RuntimeError("HMM retrain failed or was interrupted; dependent replay not started")
        time.sleep(5)
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())
    version = manifest["hmm"]["active_version"]
    artifact = ROOT / manifest["hmm"]["artifact_dir"]
    metadata = json.loads((artifact / "metadata.json").read_text())
    temperature = json.loads((artifact / "temperature_scaler.json").read_text())
    assert metadata["eval_metrics"]["mean_pass_rate"] >= .5
    audit = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "prior_hmm": original, "active_hmm": version, "rotated": version != original, "artifact_dir": str(artifact), "eval_metrics": metadata["eval_metrics"], "temperature_scaler": temperature}
    (OUT / "active_hmm_validation.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    assert temperature["temperature"] == 1.0 and temperature["runtime_validated"] is False
    args = [sys.executable, str(ROOT / "scripts/backfill_training_features.py"), "--input", str(OUT / "utility_input.xlsx"), "--output", str(OUT / "utility_backfilled.xlsx"), "--default-artifact-version", version, "--hmm-only", "--feature-cutoff-source", "start_time_utc", "--replay-scope", "hmm_lineage_only", "--require-fresh-output", "--max-concurrency", "1"]
    (OUT / "utility_backfill_command.json").write_text(json.dumps(args, indent=2) + "\n")
    with (OUT / "utility_backfill.log").open("w", encoding="utf-8") as handle:
        result = subprocess.run(args, cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.returncode
    print("Utility workbook replay completed; inspect lineage before fitting.", flush=True)


if __name__ == "__main__":
    main()
