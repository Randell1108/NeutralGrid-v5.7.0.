"""Refit the existing four-feature shadow recipe with audited historical reuse."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES
from neutralgrid.scanner.profile_model import load_profile_model, save_profile_model
from neutralgrid.scanner.profile_model_walkforward import _train_from_frame, _expected_calibration_error
from neutralgrid.scanner.canonical_fastwin_profile import _build_pattern
from neutralgrid.core.constants import ENGINE_VERSION, FORMULA_VERSION, LABEL_CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "reports/retrain_all_20260911/expanded_profile_shadow"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    destination = OUT / "expanded_profile_shadow"
    destination.mkdir(exist_ok=False)
    previous_manifest = json.loads((OLD / "manifest.json").read_text())
    assert sha(OLD / "training_pool.csv") == previous_manifest["training"]["source_sha256"]
    assert sha(OLD / "profile_model.json") == previous_manifest["model_sha256"]
    prior_audit = json.loads((OLD.parent / "profile_pool_final_audit.json").read_text())
    assert prior_audit["status"] == "PASS"
    previous = pd.read_csv(OLD / "training_pool.csv")
    assert previous.candidate_id.is_unique
    previous_all_ids = set(previous.candidate_id)
    source_manifest = json.loads((OUT / "fresh_production_pool/backtest_run_manifest.json").read_text())
    fresh_path = Path(source_manifest["training_file"])
    fresh = pd.read_csv(fresh_path, low_memory=False)
    for column, expected in {"engine_version": ENGINE_VERSION, "formula_version": FORMULA_VERSION, "label_contract_version": LABEL_CONTRACT_VERSION, "realism_profile": "legacy"}.items():
        assert fresh[column].astype(str).eq(expected).all(), column
    assert fresh.candidate_id.is_unique
    features = list(DEFAULT_FEATURES)
    complete = np.isfinite(fresh[features].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    fresh.loc[~complete].to_csv(destination / "excluded_fresh_rows.csv", index=False)
    fresh = fresh.loc[complete].copy()
    target_time = pd.to_numeric(fresh.time_to_target_hours, errors="coerce")
    fresh["fastwin_label"] = (np.isfinite(target_time) & target_time.le(7)).astype(int)
    fresh["_is_winner"] = fresh.fastwin_label
    unseen = fresh.loc[~fresh.candidate_id.isin(previous_all_ids)].copy()
    frozen = load_profile_model(OLD / "profile_model.json")
    # This recorded completion receipt proves the frozen artifact existed by
    # this time; use it rather than infer a date from a directory name.
    frozen_at = pd.Timestamp(json.loads((OLD.parent / "summary.json").read_text())["completed_at_utc"])
    evaluations = {}
    if len(unseen):
        unseen["probability"] = [frozen.proba(row) for row in unseen.to_dict("records")]
        parts = unseen.candidate_id.str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
        scan = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True)
        start = pd.to_datetime(unseen.start_time_utc, utc=True, format="mixed")
        unseen["after_frozen_artifact"] = scan.gt(frozen_at) & start.gt(frozen_at)
        unseen[["candidate_id", "start_time_utc", "fastwin_label", "probability", "after_frozen_artifact"]].to_csv(destination / "prior_model_new_id_predictions.csv", index=False)
        for name, frame in {"new_ids": unseen, "new_ids_after_frozen_artifact": unseen.loc[unseen.after_frozen_artifact]}.items():
            labels = frame.fastwin_label.to_numpy()
            probabilities = frame.probability.to_numpy()
            evaluations[name] = {"rows": len(frame), "positives": int(labels.sum()), "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None, "brier": float(np.mean((probabilities-labels)**2)) if len(frame) else None, "ece": float(_expected_calibration_error(probabilities, labels)) if len(frame) else None}
    common = ["candidate_id", "symbol", "start_time_utc", "fastwin_label", "_is_winner", *features]
    retained = previous.loc[~previous.candidate_id.isin(fresh.candidate_id), common]
    pool = pd.concat([fresh[common], retained], ignore_index=True)
    assert pool.candidate_id.is_unique
    pool["start_time_utc"] = pd.to_datetime(pool.start_time_utc, utc=True, format="mixed")
    parts = pool.candidate_id.str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scans = pd.to_datetime(parts[0]+parts[1], format="%Y%m%d%H%M%S", utc=True)
    assert scans.le(pool.start_time_utc).all()
    assert np.isfinite(pool[features].to_numpy(dtype=float)).all()
    assert pool.fastwin_label.isin([0, 1]).all()
    pool = pool.sort_values(["start_time_utc", "candidate_id"]).reset_index(drop=True)
    pool.to_csv(destination / "training_pool.csv", index=False)
    model = _train_from_frame(pool, features, shrinkage=.9, max_duration_hours=7)
    assert model is not None
    summary = {"artifact_role": "shadow_full_refit", "target_contract": "fast_winner_time_to_3pct_le_7h", "training_rows": len(pool), "label_counts": pool.fastwin_label.value_counts().to_dict(), "family": "gaussian_lda_v1", "shrinkage": .9, "source_sha256": sha(destination / "training_pool.csv"), "fresh_outcome_rows": len(fresh), "retained_previous_outcome_rows": len(retained), "new_candidate_ids": len(unseen), "promotion_blockers": ["Target differs from canonical completed-bot profile", "No verified production incumbent comparison or bot-disjoint, event-complete promotion evidence"], "hmm_features_used": False}
    model = dataclasses.replace(model, selection_summary=summary)
    save_profile_model(model, destination / "profile_model.json")
    _build_pattern(pool, summary).save_json(destination / "pattern_profile.json")
    report = {"status": "shadow_only", "training": summary, "historical_source_policy": "Reuse previously audited September 11 pool, including its verified fresh replacements for timestamp-conflicted older outcomes; no raw conflicted historical pool is reintroduced", "prior_pool_audit": prior_audit, "prior_frozen_artifact_available_by": str(frozen_at), "prior_frozen_model_evaluation": evaluations, "evaluation_scope": "Prior model only; candidate-ID temporal separation does not prove bot-disjoint independence", "newly_refitted_model_has_independent_holdout": False, "model_sha256": sha(destination / "profile_model.json"), "pattern_sha256": sha(destination / "pattern_profile.json"), "source_fresh_sha256": sha(fresh_path), "source_previous_sha256": sha(OLD / "training_pool.csv")}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "prior_pool_audit"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
