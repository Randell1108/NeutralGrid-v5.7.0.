"""Run canonical finalization and refit only after the fresh replay audit passes."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def run(args: list[str], name: str) -> None:
    with (OUT / name).open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (name, result.returncode)


def main() -> None:
    run([str(OUT / "audit_fresh_pool.py")], "fresh_pool_audit.log")
    run([str(OUT / "audit_meta_pool.py")], "meta_lineage_audit.log")
    audit = json.loads((OUT / "meta_lineage_audit.json").read_text())
    assert audit["status"] == "PASS"
    missing = {k:v for k,v in audit["feature_nonfinite_counts"].items() if v}
    assert set(missing) <= {"ou_halflife"}, missing
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    args = ["scripts/finalize_fresh_authoritative_meta_pool.py", "--source", str(OUT / "fresh_meta_validated_backfilled.csv"), "--run-manifest", str(OUT / "fresh_production_pool/backtest_run_manifest.json"), "--output-dir", str(OUT / "finalized_fresh_pool"), "--start-date", "2026-02-18", "--end-date", "2026-09-15", "--active-hmm-artifact-version", version]
    if missing:
        args.append("--allow-active-feature-imputation")
    run(args, "meta_finalization.log")
    args = ["retrain_meta_labeler.py", "--input", str(ROOT / "data/new_expired_bots.xlsx"), "--backtest-results-dir", str(OUT / "finalized_fresh_pool"), "--max-rows-per-symbol", "0", "--estimator", "vote_logit_hgb", "--export-training-data", str(OUT / "prepared_training.csv")]
    if missing:
        args.append("--allow-imputation")
    (OUT / "meta_retrain_command.json").write_text(json.dumps(args, indent=2)+"\n")
    run(args, "meta_retrain.log")
    print("Canonical refit finished; validate promotion result and runtime separately.", flush=True)


if __name__ == "__main__":
    main()
