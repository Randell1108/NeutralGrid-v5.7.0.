"""Validate results, uncertainty, and isolated native MetaLabeler fits."""
from __future__ import annotations

import dataclasses
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

import study as s
from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    logging.basicConfig(level=logging.INFO)
    audit = json.loads((s.OUT / "audit.json").read_text())
    assert s.digest(s.SOURCE) == audit["source_sha256"]
    assert s.digest(s.ROOT / "src/neutralgrid/models/meta_labeler.py") == audit["source_code_sha256"]
    data = pd.read_csv(s.TEMP / "complete_training.csv", low_memory=False)
    prediction = pd.read_csv(s.OUT / "holdout_predictions.csv")
    assert not data.candidate_id.duplicated().any()
    assert not prediction.candidate_id.duplicated().any()
    assert set(prediction.candidate_id).issubset(set(data.candidate_id))
    assert np.array_equal(data.set_index("candidate_id").loc[prediction.candidate_id, "fast_winner_target"].to_numpy(), prediction.fast_winner_target.to_numpy())
    y = prediction.fast_winner_target.to_numpy(int)
    scan_codes, scan_values = pd.factorize(prediction.scan_time)
    symbol_codes, symbol_values = pd.factorize(prediction.symbol)
    uncertainty = []
    for seed in [42, 7, 123]:
        selected = prediction[f"selected_{seed}"].to_numpy(float)
        baseline = prediction[f"baseline_{seed}"].to_numpy(float)
        rng = np.random.default_rng(20260905)
        selected_aucs, baseline_aucs, deltas = [], [], []
        for _ in range(2000):
            scan_weight = np.bincount(rng.integers(len(scan_values), size=len(scan_values)), minlength=len(scan_values))
            symbol_weight = np.bincount(rng.integers(len(symbol_values), size=len(symbol_values)), minlength=len(symbol_values))
            weight = scan_weight[scan_codes] * symbol_weight[symbol_codes]
            if len(np.unique(y[weight > 0])) < 2:
                continue
            a = float(roc_auc_score(y, selected, sample_weight=weight))
            b = float(roc_auc_score(y, baseline, sample_weight=weight))
            selected_aucs.append(a)
            baseline_aucs.append(b)
            deltas.append(a-b)
        uncertainty.append({"seed": seed,
            "method": "independently resample scan groups and symbols; multiplicative observation weights",
            "scan_groups": len(scan_values), "symbols": len(symbol_values), "replicates": len(deltas),
            "selected_auc_ci": np.quantile(selected_aucs, [.025, .975]).tolist(),
            "baseline_auc_ci": np.quantile(baseline_aucs, [.025, .975]).tolist(),
            "paired_delta_ci": np.quantile(deltas, [.025, .975]).tolist()})
    s.write("two_way_uncertainty.json", uncertainty)
    native_results = []
    for role, features in [("baseline", s.FEATURES), ("selected", [f for f in s.FEATURES if f != "ou_halflife"])]:
        assert np.isfinite(data[features].to_numpy(float)).all()
        labeler = MetaLabeler(MetaLabelerConfig(features=features, estimator_type="vote_logit_hgb", cv_folds=5, random_state=42))
        result = labeler.train(data.copy(), timestamp_col="start_time_utc", pnl_col="net_pnl_pct", y_col="fast_winner_target")
        native_results.append({"role": role, "features": features,
            "metrics": finite_json(dataclasses.asdict(result)),
            "promotion_eligible": False,
            "reason": "Native training diagnostic on complete rows; strict full-pool admission failed; chronological development ECE failed; no model publication."})
        s.write("native_training.json", native_results)
        if role == "selected":
            path = s.OUT / "selected_candidate_diagnostic.joblib"
            joblib.dump({"artifact_kind": "isolated_meta_feature_study_candidate", "diagnostic_only": True,
                "promotion_eligible": False, "runtime_meta_labeler_compatible": False,
                "features": features, "labeler": labeler,
                "training_source_sha256": audit["source_sha256"],
                "hmm_artifact_version": audit["active_hmm"]}, path)
            loaded = joblib.load(path)
            for row in data[features].iloc[:10].to_dict("records"):
                assert labeler.predict_proba(row) == loaded["labeler"].predict_proba(row)
            s.write("candidate_manifest.json", {"artifact": str(path), "sha256": s.digest(path),
                "feature_count": len(features), "features": features, "diagnostic_only": True,
                "promotion_eligible": False, "runtime_meta_labeler_compatible": False,
                "roundtrip_predictions_verified": 10, "train_rows": len(data),
                "note": "Full-data research fit; held-out study metrics evaluate earlier past-only fits, not independent testing of this full-data object."})
    assert s.digest(s.SOURCE) == audit["source_sha256"]
    assert s.digest(s.ROOT / "src/neutralgrid/models/meta_labeler.py") == audit["source_code_sha256"]
    print("VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
