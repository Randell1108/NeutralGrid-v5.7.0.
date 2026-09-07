"""Isolated, reproducible feature-count study; never writes production models.

Run with the repository virtualenv and NEUTRALGRID_BASE_DIR set to this checkout.
The plan is written before any development fits. Holdout results never select a model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ["NEUTRALGRID_BASE_DIR"] = str(ROOT)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import retrain_meta_labeler as rml
from neutralgrid.models.meta_labeler import (
    ACTIVE_SNAPSHOT_META_FEATURES, MetaLabeler, MetaLabelerConfig,
    PROMOTION_EVALUATION_CONTRACT, _is_label_column,
)
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder
from scripts.finalize_fresh_authoritative_meta_pool import finalize_fresh_pool

OUT = Path(__file__).resolve().parent
TEMP = OUT / "_tmp"
SOURCE_DIR = ROOT / "artifacts/candidate_refits/meta_full_pool_20260903_204200"
SOURCE = SOURCE_DIR / "training_data_20260904_hmm_rolling_180d_20260903_153527.csv"
FEATURES = list(ACTIVE_SNAPSHOT_META_FEATURES)
PURGE = pd.Timedelta(hours=12)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimator(kind, seed):
    lr = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=seed)
    if kind == "logistic":
        return lr
    hgb = HistGradientBoostingClassifier(
        learning_rate=.03, max_leaf_nodes=10, max_iter=350,
        min_samples_leaf=60, l2_regularization=3.0, max_features=.8,
        early_stopping=cast(Any, False), random_state=seed,
    )
    return VotingClassifier([("logit", lr), ("hgb", hgb)], voting="soft", weights=[1., 1.])


def rank_features(train):
    y = train.fast_winner_target.to_numpy(int)
    scores = {f: abs(float(roc_auc_score(y, train[f])) - .5) for f in FEATURES}
    return sorted(FEATURES, key=lambda f: (-scores[f], FEATURES.index(f)))


def chosen_features(spec, train):
    if spec["set"] == "top_k":
        return rank_features(train)[:spec["k"]]
    if spec["set"] == "no_ev":
        return [f for f in FEATURES if f != "ev_score"]
    if spec["set"] == "no_ou":
        return [f for f in FEATURES if f != "ou_halflife"]
    return FEATURES.copy()


def partition(past, test_start):
    eligible = past.loc[past.event_end < test_start - PURGE].copy()
    times = sorted(eligible.scan_time.unique())
    if len(times) < 2:
        raise ValueError("Too few independent training/calibration scan groups")
    cal_start = times[-1]
    cal = eligible.loc[eligible.scan_time == cal_start].copy()
    fit = eligible.loc[eligible.event_end < cal_start - PURGE].copy()
    for name, part in [("fit", fit), ("calibration", cal)]:
        if len(part) < 30 or part.fast_winner_target.nunique() != 2:
            raise ValueError(f"Insufficient {name} support: {len(part)}")
    assert fit.event_end.max() < cal.scan_time.min() - PURGE
    assert cal.event_end.max() < test_start - PURGE
    assert not set(fit.candidate_id) & set(cal.candidate_id)
    return fit, cal


def fit_predict(spec, past, test, seed=42):
    fit, cal = partition(past, test.scan_time.min())
    features = chosen_features(spec, fit)
    assert not any(_is_label_column(f) for f in features)
    model = make_pipeline(StandardScaler(), estimator(spec["estimator"], seed))
    model.fit(fit[features].to_numpy(float), fit.fast_winner_target.to_numpy(int))
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(cal[features].to_numpy(float), cal.fast_winner_target.to_numpy(int))
    pred = calibrated.predict_proba(test[features].to_numpy(float))[:, 1]
    return pred, features, {"fit_rows": len(fit), "calibration_rows": len(cal), "test_rows": len(test),
                            "fit_end": str(fit.event_end.max()), "cal_start": str(cal.scan_time.min()),
                            "cal_end": str(cal.event_end.max()), "test_start": str(test.scan_time.min())}


def metrics(y, p):
    return {"n": len(y), "positives": int(np.sum(y)), "auc": float(roc_auc_score(y, p)),
            "ece": float(MetaLabeler._expected_calibration_error(y, p)),
            "brier": float(brier_score_loss(y, p))}


def cluster_ci(frame, probability, other=None, n_boot=2000):
    groups = frame.scan_time.astype(str).to_numpy()
    unique = np.unique(groups)
    y = frame.fast_winner_target.to_numpy(int)
    rng = np.random.default_rng(20260905)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        v = roc_auc_score(y[idx], probability[idx])
        if other is not None:
            v -= roc_auc_score(y[idx], other[idx])
        vals.append(v)
    return {"low": float(np.quantile(vals, .025)), "high": float(np.quantile(vals, .975)),
            "scan_groups": len(unique), "bootstrap_replicates": len(vals)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    TEMP.mkdir(exist_ok=False)
    (TEMP / "input").mkdir()
    raw = pd.read_csv(SOURCE, low_memory=False)
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())
    version = manifest["hmm"]["active_version"]
    hmm_meta = json.loads((ROOT / "artifacts/hmm" / version / "metadata.json").read_text())
    audit = {"source": str(SOURCE), "source_sha256": digest(SOURCE), "rows": len(raw),
             "active_hmm": version, "features": FEATURES,
             "finite_feature_rows": {f: int(np.isfinite(np.asarray(pd.to_numeric(raw[f], errors="coerce"), dtype=float)).sum()) for f in FEATURES},
             "lineage": rml.audit_training_hmm_lineage(raw, selected_features=FEATURES,
                 active_hmm_artifact_version=version, active_hmm_trained_at_utc=hmm_meta["trained_at_utc"]),
             "finite_regime_rows": {f: int(np.isfinite(np.asarray(pd.to_numeric(raw[f], errors="coerce"), dtype=float)).sum())
                                    for f in ["range_prob", "trend_prob", "persistence_prob"]},
             "source_code_sha256": digest(ROOT / "src/neutralgrid/models/meta_labeler.py"),
             "declared_promotion_contract": PROMOTION_EVALUATION_CONTRACT,
             "canonical_model_present": (ROOT / "models/meta_labeler.pkl").exists()}
    assert audit["lineage"]["passes"]
    try:
        finalize_fresh_pool(source_path=SOURCE, run_manifest_path=SOURCE_DIR / "backtest/backtest_run_manifest.json",
            output_dir=TEMP / "strict_finalization", start_date="2026-06-08", end_date="2026-09-04",
            active_hmm_artifact_version=version)
        audit["strict_finalizer"] = "PASS"
    except ValueError as exc:
        audit["strict_finalizer"] = str(exc)
    shutil.copy2(SOURCE, TEMP / "input/training_data_20260904.csv")
    builder = UnifiedTrainingBuilder(backtest_results_dir=TEMP / "input", snapshot_dir=TEMP / "snapshots")
    pool = builder.build_meta_labeler_pool(max_rows_per_symbol=30)
    frame, summary = rml.prepare_fast_target_training_frame(pool, pnl_col="net_pnl_pct")
    audit["builder_rows"] = len(pool)
    audit["target_summary"] = summary
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    audit["excluded_incomplete_candidate_ids"] = frame.loc[~finite, "candidate_id"].tolist()
    data = frame.loc[finite].copy()
    data[FEATURES] = numeric.loc[finite]
    data["scan_time"] = pd.to_datetime(data.feature_cutoff_utc, utc=True)
    data["event_start"] = pd.to_datetime(data.start_time_utc, utc=True)
    # The saved t1 is six hours on this seven-hour label pool. Use the later
    # observed end or full target horizon so diagnostics cannot under-purge.
    data["event_end"] = pd.concat([pd.to_datetime(data.t1, utc=True),
        data.event_start + pd.Timedelta(hours=7)], axis=1).max(axis=1)
    audit["t1_shorter_than_7h_rows"] = int((pd.to_datetime(data.t1, utc=True) < data.event_start + pd.Timedelta(hours=7)).sum())
    assert not data.candidate_id.duplicated().any()
    data = data.sort_values(["scan_time", "candidate_id"]).reset_index(drop=True)
    scans = sorted(data.scan_time.unique())
    dev = data.loc[data.scan_time < scans[-3]].copy()
    holdout = data.loc[data.scan_time >= scans[-3]].copy()
    audit["complete_rows"] = len(data)
    audit["scan_groups"] = len(scans)
    audit["data_start"] = str(data.scan_time.min())
    audit["data_end"] = str(data.scan_time.max())
    audit["hmm_trained_after_event_rows"] = int((pd.to_datetime(data.hmm_trained_at_utc, utc=True) > data.scan_time).sum())
    write("audit.json", audit)
    data.to_csv(TEMP / "complete_training.csv", index=False)
    specs = []
    for kind in ["logistic", "vote_logit_hgb"]:
        specs.extend({"id": f"{kind}_top{k}", "estimator": kind, "set": "top_k", "k": k} for k in range(3, 20))
        specs.extend({"id": f"{kind}_{s}", "estimator": kind, "set": s, "k": 20 if s == "full" else 19}
                     for s in ["full", "no_ev", "no_ou"])
    plan = {"candidate_count": len(specs), "candidates": specs,
            "development_scan_groups": [str(s) for s in scans[:-3]],
            "holdout_scan_groups": [str(s) for s in scans[-3:]],
            "development_rows": len(dev), "holdout_rows": len(holdout),
            "folds": "Last six development scan groups, evaluated in three consecutive two-scan blocks; strictly past fitting, separate last eligible scan for sigmoid calibration.",
            "purge": "12h between full 7h label endpoint and next calibration/evaluation start",
            "imputation": "none; 13 incomplete rows excluded uniformly for paired comparisons",
            "feature_ranking": "absolute univariate AUC distance from 0.5, fit partition only, ties in profile order",
            "selection": "maximum mean development-fold AUC among pooled ECE<=0.10; if none, maximum mean AUC. Ties: fewer features then ID.",
            "holdout": "Evaluate frozen winner and 20-feature vote baseline, seeds 42/7/123; no selection or tuning after results.",
            "uncertainty": "2000 scan-group bootstrap replicates; only three held-out scan groups, limited regime coverage",
            "production_status": "diagnostics only; no production files or gate thresholds modified"}
    write("plan.json", plan)
    dev_scans = scans[:-3]
    blocks = [dev_scans[-6:-4], dev_scans[-4:-2], dev_scans[-2:]]
    results = []
    for spec in specs:
        predictions, ys, fold_reports = [], [], []
        for block in blocks:
            test = dev.loc[dev.scan_time.isin(block)]
            past = dev.loc[dev.scan_time < min(block)]
            p, features, support = fit_predict(spec, past, test)
            y = test.fast_winner_target.to_numpy(int)
            fold_reports.append({**metrics(y, p), **support, "features": features})
            predictions.append(p)
            ys.append(y)
        result = {**spec, "pooled": metrics(np.concatenate(ys), np.concatenate(predictions)),
                  "mean_fold_auc": float(np.mean([f["auc"] for f in fold_reports])), "folds": fold_reports}
        results.append(result)
        write("development.json", results)
        print(json.dumps({"trial": spec["id"], "mean_auc": result["mean_fold_auc"], **result["pooled"]}), flush=True)
    eligible = [r for r in results if r["pooled"]["ece"] <= .10] or results
    best = sorted(eligible, key=lambda r: (-r["mean_fold_auc"], r["k"], r["id"]))[0]
    winner = next(s for s in specs if s["id"] == best["id"])
    baseline = next(s for s in specs if s["id"] == "vote_logit_hgb_full")
    write("frozen_selection.json", {"winner": winner, "development": best, "holdout_not_yet_evaluated": True})
    final = []
    prediction_table = holdout[["candidate_id", "symbol", "scan_time", "fast_winner_target", "capital_fraction"]].copy()
    for seed in [42, 7, 123]:
        by_kind = {}
        for label, spec in [("selected", winner), ("baseline", baseline)]:
            p, features, support = fit_predict(spec, dev, holdout, seed)
            by_kind[label] = p
            prediction_table[f"{label}_{seed}"] = p
            result = {"role": label, "spec": spec, "seed": seed, "features": features,
                      **metrics(holdout.fast_winner_target.to_numpy(int), p), **support,
                      "cluster_auc_ci": cluster_ci(holdout, p),
                      "per_scan": []}
            for scan in scans[-3:]:
                mask = (holdout.scan_time == scan).to_numpy()
                result["per_scan"].append({"scan": str(scan), **metrics(holdout.fast_winner_target.to_numpy(int)[mask], p[mask])})
            final.append(result)
        final.append({"seed": seed, "paired_selected_minus_baseline_auc_ci": cluster_ci(holdout, by_kind["selected"], by_kind["baseline"])})
    write("holdout.json", final)
    prediction_table.to_csv(OUT / "holdout_predictions.csv", index=False)
    # Diagnostic replication of the current runtime gate, independently of the
    # chronological search. No source-admission or save operation is bypassed.
    gate_results = []
    for spec in [baseline, winner]:
        fit, _ = partition(dev, holdout.scan_time.min())
        features = chosen_features(spec, fit)
        config = MetaLabelerConfig(features=features, estimator_type=spec["estimator"], cv_folds=5)
        labeler = MetaLabeler(config)
        values = labeler._evaluate_promotion_oof(data[features].to_numpy(float),
            data.fast_winner_target.to_numpy(int), data.symbol.to_numpy(),
            data.event_start.to_numpy(), data.event_end.to_numpy(),
            estimator_factory=lambda: estimator(spec["estimator"], 42))
        auc, low, high, ece = values
        status, reasons = labeler._evaluate_promotion_gate(oof_auc_ci_low=low, oof_auc_ci_high=high,
            n_pos=int(data.fast_winner_target.sum()), oof_ece=ece)
        gate_results.append({"spec": spec, "features": features, "auc": auc, "ci_low": low,
            "ci_high": high, "ece": ece, "status": status, "reasons": reasons,
            "qualification": "exploratory runtime-gate replication; all labels reused, not independent selection evidence"})
        write("runtime_gate.json", gate_results)
    print("STUDY_COMPLETE", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
