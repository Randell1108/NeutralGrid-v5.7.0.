"""Audit and union fresh canonical outcomes, including matured-window retries.

This report-local helper does not generate outcomes or alter any model gate.
Original canonical manifests and files remain intact and hash-addressed.
"""
from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-fetch-recovery", action="store_true")
    args = parser.parse_args()
    selected = pd.read_csv(ROOT / "selected_candidates.csv")
    allowed_ids = set(selected.candidate_id.astype(str))
    retry = json.loads((ROOT / "mature_window_retry.json").read_text())
    retry_ids = set(retry["now_mature_retry_ids"])
    contracts = {
        "generation_mode": "fresh_full_pool", "full_pool": True,
        "realism_profile": "legacy", "max_candidates": None,
        "hours": 7, "min_bars": 420, "leverage": 10,
        "pool_start_date": "2026-02-18", "pool_end_date": "2026-09-07",
    }
    frames, result_frames, components = [], [], []
    seen: set[str] = set()
    main_manifest = None
    names = ["fresh_backtest", "fresh_retry"]
    recovery_ids: set[str] = set()
    if args.include_fetch_recovery:
        recovery_ids.add(json.loads((ROOT / "fetch_recovery_input.json").read_text())["candidate_id"])
        names.append("fresh_fetch_recovery")
    for name in names:
        manifest_path = ROOT / name / "backtest_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in contracts.items():
            assert manifest.get(key) == expected, (name, key, manifest.get(key), expected)
        training_path = Path(manifest["training_file"])
        results_path = Path(manifest["results_file"])
        frame = pd.read_csv(training_path, low_memory=False)
        results = pd.read_csv(results_path, low_memory=False)
        ids = set(frame.candidate_id.astype(str))
        assert len(ids) == len(frame) == manifest["successful_rows"]
        assert set(results.candidate_id.astype(str)) == ids
        assert ids <= allowed_ids, sorted(ids - allowed_ids)
        assert not ids & seen, "Retry outcomes must replace only previously absent outcomes"
        if name == "fresh_retry":
            assert ids <= retry_ids
        elif name == "fresh_fetch_recovery":
            assert ids <= recovery_ids
        elif name == "fresh_backtest":
            main_manifest = manifest
        seen |= ids
        frames.append(frame)
        result_frames.append(results)
        components.append({
            "run": name, "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "training_path": str(training_path), "training_sha256": sha256(training_path),
            "results_path": str(results_path), "results_sha256": sha256(results_path),
            "successful_rows": len(frame), "generated_at_utc": manifest["generated_at_utc"],
        })
    assert main_manifest is not None
    output = ROOT / ("fresh_all_outcomes" if args.include_fetch_recovery else "fresh_combined")
    output.mkdir(exist_ok=False)
    training_path = output / "training_data_20260907.csv"
    results_path = output / "backtest_results_20260907.csv"
    pd.concat(frames, ignore_index=True).to_csv(training_path, index=False)
    pd.concat(result_frames, ignore_index=True).to_csv(results_path, index=False)
    manifest = dict(main_manifest)
    manifest.update({
        "aggregation_method": "disjoint_union_of_fresh_canonical_runs",
        "aggregation_reason": "Add previously absent outcomes from matured-window retries and, if supplied, transient fetch recovery",
        "component_runs": components,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_file": str(training_path), "results_file": str(results_path),
        "successful_rows": len(seen), "selected_unique_candidates": len(allowed_ids),
        "retry_selected_candidates": len(retry_ids),
        "fetch_recovery_selected_candidates": len(recovery_ids),
        "training_sha256": sha256(training_path), "results_sha256": sha256(results_path),
        "historical_outcomes_reused": False,
        "canonical_runner_source_sha256": sha256(ROOT.parents[1] / "backtest_candidates.py"),
    })
    (output / "backtest_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    ledger = selected.copy()
    ledger["fresh_outcome_available"] = ledger.candidate_id.astype(str).isin(seen)
    ledger["mature_window_retry"] = ledger.candidate_id.astype(str).isin(retry_ids)
    ledger["fetch_recovery_retry"] = ledger.candidate_id.astype(str).isin(recovery_ids)
    ledger.to_csv(ROOT / "fresh_outcome_dispositions.csv", index=False)
    print(json.dumps({"successful_rows": len(seen), "selected_rows": len(allowed_ids), "components": components}, indent=2))


if __name__ == "__main__":
    main()
