"""Independent persisted-output and source-integrity checks; no model tuning."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from study import OUT, ROOT, FEATURES, load_frame, score, sha, write_json, _build_pattern
from neutralgrid.scanner.profile_model import load_profile_model, save_profile_model
from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.backtest.candidate_pipeline import resolve_backtest_start_timestamp


def main() -> None:
    previous = ROOT / "reports/meta_canonical_refit_20260907"
    original = json.loads((previous / "source_inventory.json").read_text())
    undated = json.loads((previous / "undated_export_audit.json").read_text())
    known = {str(Path(r["source"]).resolve()).lower(): r["sha256"] for r in original["files"]}
    known.update({str(Path(r["path"]).resolve()).lower(): r["sha256"] for r in undated})
    found, errors = set(), []
    def error(exc):
        errors.append(str(exc))
    for root in original["roots"]:
        if not Path(root).exists():
            continue
        for directory, _, files in os.walk(root, onerror=error):
            for name in files:
                if name.lower().startswith("deployment_ready") and Path(name).suffix.lower() in {".csv", ".xlsx"}:
                    found.add(str((Path(directory) / name).resolve()).lower())
    changed = [p for p, expected in known.items() if not Path(p).exists() or sha(Path(p)) != expected]
    refresh = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "known_paths": len(known), "current_scanner_files": len(found), "additional_paths": sorted(found - set(known)), "changed_or_missing_sources": changed, "scan_errors": errors, "roots": original["roots"]}
    extra_audits = []
    for path in refresh["additional_paths"]:
        p = Path(path)
        frame = pd.read_excel(p) if p.suffix.lower() == ".xlsx" else pd.read_csv(p)
        available = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
        starts = []
        for row in frame.to_dict("records"):
            row["candidate_available_ts_utc"] = available.isoformat()
            starts.append(pd.Timestamp(resolve_backtest_start_timestamp(row)))
        mature = [t + pd.Timedelta(hours=7) <= pd.Timestamp.now(tz="UTC") for t in starts]
        assert not any(mature), "New mature candidates require a separate frozen supplement"
        extra_audits.append({"path": path, "sha256": sha(p), "rows": len(frame), "mature_rows": sum(mature), "earliest_maturity_utc": (min(starts) + pd.Timedelta(hours=7)).isoformat(), "latest_maturity_utc": (max(starts) + pd.Timedelta(hours=7)).isoformat(), "disposition": "excluded: seven-hour outcomes are not yet observable"})
    refresh["additional_file_audits"] = extra_audits
    write_json(OUT / "inventory_refresh.json", refresh)
    assert not changed and not errors, refresh

    protected = json.loads((OUT / "protected_files_before.json").read_text())
    for entry in protected:
        assert sha(Path(entry["path"])) == entry["sha256"], entry["path"]
    assert not (ROOT / "data/profile/current.json").exists()
    assert not (ROOT / "data/profile/profile_model.json").exists()
    protocol = json.loads((OUT / "preregistration.json").read_text())
    for entry in protocol["sources"]:
        assert sha(Path(entry["path"])) == entry["sha256"]
    full = load_frame("complete_profile_pool.csv")
    dev, hold, purge = [load_frame(n + ".csv") for n in ("development", "holdout", "boundary_purge")]
    assert len(full) == len(dev) + len(hold) + len(purge) == 6832
    assert set(full.candidate_id) == set(dev.candidate_id) | set(hold.candidate_id) | set(purge.candidate_id)
    assert dev.start_time_utc.max() + pd.Timedelta(hours=24) < hold.start_time_utc.min()
    assert not set(dev.candidate_id) & set(hold.candidate_id)
    assert np.isfinite(full[FEATURES].to_numpy()).all()
    assert len(FEATURES) == 4 and all(f not in {"_is_winner", "fastwin_label", "hlabel"} for f in FEATURES)
    selection = json.loads((OUT / "selection.json").read_text())
    evaluation = json.loads((OUT / "holdout_evaluation.json").read_text())
    assert selection == evaluation["selection"]
    for file, key in [("frozen_shadow/profile_model.json", "model_sha256"), ("frozen_shadow/pattern_profile.json", "pattern_sha256"), ("frozen_shadow/baseline_profile_model.json", "baseline_sha256"), ("holdout.csv", "holdout_sha256")]:
        assert sha(OUT / file) == selection[key]
    frozen_model = load_profile_model(OUT / "frozen_shadow/profile_model.json")
    raw, probability = score(frozen_model, hold)
    stored = pd.read_csv(OUT / "holdout_predictions.csv")
    assert stored.candidate_id.tolist() == hold.candidate_id.tolist()
    assert np.allclose(probability, stored.selected_probability, rtol=1e-12, atol=1e-12)
    assert np.isclose(roc_auc_score(hold.fastwin_label, raw), evaluation["selected"]["auc"])
    # Repair copied development-only metadata on the full-data refit. Coefficients
    # and all frozen validation artifacts remain unchanged; no fitting occurs.
    full_dir = OUT / "full_refit_shadow"
    model = load_profile_model(full_dir / "profile_model.json")
    raw_before, proba_before = score(model, full)
    summary = dict(model.selection_summary or {}, label_counts=full.fastwin_label.value_counts().to_dict(), source_sha256=sha(OUT / "complete_profile_pool.csv"))
    corrected = dataclasses.replace(model, selection_summary=summary)
    save_profile_model(corrected, full_dir / "profile_model.json")
    _build_pattern(full, summary).save_json(full_dir / "pattern_profile.json")
    manifest = json.loads((full_dir / "manifest.json").read_text())
    manifest.update(model_sha256=sha(full_dir / "profile_model.json"), pattern_sha256=sha(full_dir / "pattern_profile.json"), label_counts=summary["label_counts"], source_sha256=summary["source_sha256"])
    write_json(full_dir / "manifest.json", manifest)
    reloaded = load_profile_model(full_dir / "profile_model.json")
    raw_after, proba_after = score(reloaded, full)
    assert np.array_equal(raw_before, raw_after) and np.array_equal(proba_before, proba_after)
    assert reloaded.selection_summary["training_rows"] == len(full)
    assert sum(int(n) for n in reloaded.selection_summary["label_counts"].values()) == len(full)
    assert np.isclose(reloaded.prior_winner, full.fastwin_label.mean())
    pattern = PatternProfile.load_json(full_dir / "pattern_profile.json")
    assert pattern.features == reloaded.features == FEATURES
    dev_results = json.loads((OUT / "development_evaluation.json").read_text())
    for family in dev_results["families"]:
        oof = pd.read_csv(OUT / (family["family"] + "_development_oof.csv"))
        assert oof.candidate_id.is_unique and not set(oof.candidate_id) & set(hold.candidate_id)
        assert np.isclose(roc_auc_score(oof.fastwin_label, oof.score), family["pooled"]["auc"])
        for fold in family["folds"]:
            assert pd.Timestamp(fold["training_end"]) + pd.Timedelta(hours=24) < pd.Timestamp(fold["test_start"])
    audit = {"validated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "protected_files_unchanged": protected, "source_files_rehashed": len(protocol["sources"]), "full_rows": len(full), "label_counts": full.fastwin_label.value_counts().to_dict(), "start_time_min_utc": full.start_time_utc.min().isoformat(), "start_time_max_utc": full.start_time_utc.max().isoformat(), "unique_start_groups": int(full.start_time_utc.nunique()), "symbols": int(full.symbol.nunique()), "features": FEATURES, "complete_feature_rows": len(full), "runtime_predictions_verified": True, "frozen_holdout_hashes_unchanged": True, "chronological_purge_and_candidate_id_disjointness": True, "metadata_repair": "Full-refit label counts and source hash corrected from copied development metadata; coefficients and predictions unchanged.", "deployment_status": "shadow_only; active profile pointer absent and bootstrap unchanged"}
    write_json(OUT / "artifact_validation.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
