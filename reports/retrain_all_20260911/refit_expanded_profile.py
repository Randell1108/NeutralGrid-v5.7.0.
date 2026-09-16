"""Evaluate the frozen prior shadow on new IDs, then refit its unchanged recipe."""
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
OLD = ROOT / "reports/profile_training_20260908"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    destination = OUT / "expanded_profile_shadow"
    destination.mkdir(exist_ok=False)
    previous = pd.read_csv(OLD / "complete_profile_pool.csv")
    previous_all_ids = set(previous.candidate_id)
    conflicts = set(pd.read_csv(OUT / "timestamp_excluded_source_rows.csv", usecols=["candidate_id"]).candidate_id.astype(str))
    quarantined_previous = previous.loc[previous.candidate_id.isin(conflicts)].copy()
    quarantined_previous.to_csv(destination / "quarantined_historical_rows.csv", index=False)
    previous = previous.loc[~previous.candidate_id.isin(conflicts)].copy()
    source_manifest = json.loads((OUT / "fresh_validated_pool/backtest_run_manifest.json").read_text())
    fresh_path = Path(source_manifest["training_file"])
    fresh = pd.read_csv(fresh_path, low_memory=False)
    for column, expected in {"engine_version": ENGINE_VERSION, "formula_version": FORMULA_VERSION, "label_contract_version": LABEL_CONTRACT_VERSION, "realism_profile": "legacy"}.items():
        assert fresh[column].astype(str).eq(expected).all(), column
    assert fresh.candidate_id.is_unique
    parts = fresh.candidate_id.str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scan = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True)
    start = pd.to_datetime(fresh.start_time_utc, utc=True, format="mixed")
    assert scan.le(start).all(), "Fresh profile outcomes start before the recorded scan"
    features = list(DEFAULT_FEATURES)
    complete = np.isfinite(fresh[features].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    fresh = fresh.loc[complete].copy()
    t2t = pd.to_numeric(fresh.time_to_target_hours, errors="coerce")
    fresh["fastwin_label"] = (np.isfinite(t2t) & t2t.le(7)).astype(int)
    fresh["_is_winner"] = fresh.fastwin_label
    common = ["candidate_id", "symbol", "start_time_utc", "fastwin_label", "_is_winner", *features]
    unseen = fresh.loc[~fresh.candidate_id.isin(previous_all_ids)].copy()
    old_model_path = OLD / "full_refit_shadow/profile_model.json"
    old_manifest = json.loads((OLD / "full_refit_shadow/manifest.json").read_text())
    assert sha(old_model_path) == old_manifest["model_sha256"]
    frozen = load_profile_model(old_model_path)
    metrics = None
    if len(unseen):
        probabilities = np.array([frozen.proba(row) for row in unseen.to_dict("records")])
        labels = unseen.fastwin_label.to_numpy()
        unseen[["candidate_id", "start_time_utc", "fastwin_label"]].assign(probability=probabilities).to_csv(destination / "prior_model_new_id_predictions.csv", index=False)
        metrics = {"rows": len(unseen), "positives": int(labels.sum()), "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None, "brier": float(np.mean((probabilities-labels)**2)), "ece": float(_expected_calibration_error(probabilities, labels)), "scope": "Prior frozen September 8 full-refit model on IDs absent from its training pool; no claim of global prior non-disclosure or bot-disjoint independence"}
    pool = pd.concat([fresh[common], previous.loc[~previous.candidate_id.isin(fresh.candidate_id), common]], ignore_index=True)
    assert pool.candidate_id.is_unique
    pool["start_time_utc"] = pd.to_datetime(pool.start_time_utc, utc=True, format="mixed")
    pool = pool.sort_values(["start_time_utc", "candidate_id"]).reset_index(drop=True)
    pool.to_csv(destination / "training_pool.csv", index=False)
    model = _train_from_frame(pool, features, shrinkage=.9, max_duration_hours=7)
    assert model is not None
    summary = {"artifact_role": "shadow_full_refit", "target_contract": "fast_winner_time_to_3pct_le_7h", "training_rows": len(pool), "label_counts": pool.fastwin_label.value_counts().to_dict(), "family": "gaussian_lda_v1", "shrinkage": .9, "recipe_source": str(OLD / "selection.json"), "source_sha256": sha(destination / "training_pool.csv"), "fresh_outcome_rows": len(fresh), "retained_previous_outcome_rows": int((~previous.candidate_id.isin(fresh.candidate_id)).sum()), "new_candidate_ids": len(unseen), "promotion_blockers": ["Target differs from canonical completed-bot profile", "No verified production incumbent comparison or bot-disjoint promotion evidence"], "hmm_features_used": False}
    model = dataclasses.replace(model, selection_summary=summary)
    save_profile_model(model, destination / "profile_model.json")
    _build_pattern(pool, summary).save_json(destination / "pattern_profile.json")
    report = {"status": "shadow_only", "training": summary, "quarantined_historical_rows": len(quarantined_previous), "prior_frozen_model_evaluation": metrics, "prior_model_limitation": "Previously frozen model included historical IDs now subject to timestamp conflict; its score is diagnostic only", "newly_refitted_model_has_independent_holdout": False, "model_sha256": sha(destination / "profile_model.json"), "pattern_sha256": sha(destination / "pattern_profile.json"), "source_fresh_sha256": sha(fresh_path), "source_previous_sha256": sha(OLD / "complete_profile_pool.csv")}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
