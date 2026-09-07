"""Isolated all-data chronological FASTWIN experiment; never publishes models."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["NEUTRALGRID_BASE_DIR"] = str(ROOT)
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES, MetaLabeler, _is_label_column

ACTIVE = list(ACTIVE_SNAPSHOT_META_FEATURES)
EXTRA = ["long_short_ratio", "funding_rate_zscore", "open_interest_change_pct", "bb_width_ratio_1h_15m",
    "top_account_lsr_log", "top_position_lsr_log", "top_position_vs_account_delta", "global_lsr_log", "taker_imbalance", "taker_buy_sell_log", "basis_pct",
    "spread_pct", "top20_bid_depth_usdt", "top20_ask_depth_usdt", "top20_depth_min_usdt", "book_imbalance_top20", "oi_notional", "oi_to_volume",
    "parkinson_vol_ratio_4h_24h_pre", "variance_ratio_1m_15m_pre_2h", "funding_carry_expected_next_7h", "liquidity_stability_z_1h"]
ALL = ACTIVE + EXTRA
INDEPENDENT = [f for f in ALL if f != "ev_score"]
PURGE = pd.Timedelta(hours=12)
HOLDOUT = pd.Timestamp("2026-09-04", tz="UTC")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write(name, value):
    (OUT / name).write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False), encoding="utf-8")


def load_frame(name):
    df = pd.read_csv(OUT / name, low_memory=False)
    for c in ["scan_time", "event_end"]:
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
    for f in ALL:
        if f not in df:
            df[f] = np.nan
        df[f] = cast(pd.Series, pd.to_numeric(df[f], errors="coerce")).replace([np.inf, -np.inf], np.nan)
    return df


def read_development_prediction(trial_id):
    csv_path = OUT / "development_predictions" / (trial_id + ".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.read_parquet(OUT / "development_predictions.parquet", filters=[("trial", "==", trial_id)])


def metrics(y, p):
    return {"rows": len(y), "positives": int(np.sum(y)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "ece": float(MetaLabeler._expected_calibration_error(y, p)),
        "brier": float(brier_score_loss(y, p))}


def partition(past, test_start):
    available = past.loc[past.event_end < test_start - PURGE].copy()
    scans = sorted(available.scan_time.unique())
    ncal = min(8, max(3, len(scans) // 5))
    if len(scans) <= ncal + 2:
        raise ValueError("Insufficient independent past scans")
    cal_start = scans[-ncal]
    cal = available.loc[available.scan_time >= cal_start].copy()
    fit = available.loc[available.event_end < cal_start - PURGE].copy()
    assert len(fit) >= 100 and len(cal) >= 50
    assert fit.fast_winner_target.nunique() == cal.fast_winner_target.nunique() == 2
    assert fit.event_end.max() < cal.scan_time.min() - PURGE
    assert cal.event_end.max() < test_start - PURGE
    return fit, cal


def pick(spec, fit):
    if spec["profile"] != "rank":
        return spec["features"]
    scores = {}
    for f in INDEPENDENT:
        good = fit[f].notna()
        if good.sum() < 50 or fit.loc[good, "fast_winner_target"].nunique() != 2:
            scores[f] = -1.
        else:
            scores[f] = abs(float(roc_auc_score(fit.loc[good, "fast_winner_target"], fit.loc[good, f])) - .5)
    return sorted(INDEPENDENT, key=lambda f: (-scores[f], INDEPENDENT.index(f)))[:spec["k"]]


def base_model(spec, seed):
    lr = LogisticRegression(C=.1, solver="liblinear", max_iter=1000, random_state=seed)
    est = lr
    if spec["estimator"] == "vote":
        hgb = HistGradientBoostingClassifier(learning_rate=.04, max_leaf_nodes=10,
            max_iter=180, min_samples_leaf=60, l2_regularization=3., early_stopping=cast(Any, False), random_state=seed)
        est = VotingClassifier([("logit", lr), ("hgb", hgb)], voting="soft")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), est)


def fit_model(spec, past, test_start, live, seed=42):
    if not spec.get("include_legacy_timing"):
        past = past.loc[~past.legacy_timestamp_uncertainty].copy()
    fit, cal = partition(past, test_start)
    if spec.get("geometric_only"):
        fit = fit.loc[fit["mode"].eq("geometric")].copy()
    features = pick(spec, fit)
    assert not any(_is_label_column(f) for f in features)
    aux = live.iloc[:0]
    if spec["live"]:
        aux = live.loc[live.fast_winner_target.notna() & (live.event_end < cal.scan_time.min() - PURGE)].copy()
        # Do not let an actual-bot twin cross the fit/calibration boundary.
        aux = aux.loc[~aux.candidate_id.isin(cal.candidate_id)]
    train = pd.concat([fit, aux], ignore_index=True)
    # Missing values remain missing in the evidence file. Medians learned only
    # on this training partition are statistical model preprocessing.
    model = base_model(spec, seed)
    model.fit(train[features].to_numpy(float), train.fast_winner_target.to_numpy(int))
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(cal[features].to_numpy(float), cal.fast_winner_target.to_numpy(int))
    support = {"fit_rows": len(fit), "auxiliary_live_rows": len(aux), "calibration_rows": len(cal),
        "fit_end": fit.event_end.max(), "calibration_start": cal.scan_time.min(),
        "calibration_end": cal.event_end.max(), "test_start": test_start,
        "fit_scan_groups": fit.scan_time.nunique(), "calibration_scan_groups": cal.scan_time.nunique(),
        "fit_rows_with_missing_selected_features": int(fit[features].isna().any(axis=1).sum()), "features": features}
    return calibrated, features, support


def trial(spec, data, live, blocks):
    with threadpool_limits(limits=1):
        reports, predictions = [], []
        for block in blocks:
            test = data.loc[data.scan_time.isin(block)].copy()
            past = data.loc[data.scan_time < min(block)].copy()
            model, features, support = fit_model(spec, past, test.scan_time.min(), live)
            p = model.predict_proba(test[features].to_numpy(float))[:, 1]
            reports.append({**metrics(test.fast_winner_target.to_numpy(int), p), **support})
            pred = test[["candidate_id", "symbol", "scan_time", "fast_winner_target"]].copy()
            pred["probability"] = p
            assert model.estimator is not None
            pred["raw_probability"] = model.estimator.predict_proba(test[features].to_numpy(float))[:, 1]
            predictions.append(pred)
        pred = pd.concat(predictions, ignore_index=True)
        pred.to_csv(OUT / "development_predictions" / (spec["id"] + ".csv"), index=False)
        return {**spec, "folds": reports, "mean_fold_auc": float(np.mean([r["auc"] for r in reports])),
            "pooled": metrics(pred.fast_winner_target.to_numpy(int), pred.probability.to_numpy(float))}


def interval(frame, pred, other=None):
    scans, scan_names = pd.factorize(frame.scan_time)
    symbols, symbol_names = pd.factorize(frame.symbol)
    y = frame.fast_winner_target.to_numpy(int)
    rng = np.random.default_rng(20260906)
    vals = []
    for _ in range(2000):
        sw = np.bincount(rng.integers(len(scan_names), size=len(scan_names)), minlength=len(scan_names))
        yw = np.bincount(rng.integers(len(symbol_names), size=len(symbol_names)), minlength=len(symbol_names))
        weight = sw[scans] * yw[symbols]
        if len(np.unique(y[weight > 0])) != 2:
            continue
        value = roc_auc_score(y, pred, sample_weight=weight)
        if other is not None:
            value -= roc_auc_score(y, other, sample_weight=weight)
        vals.append(value)
    return {"low": float(np.quantile(vals, .025)), "high": float(np.quantile(vals, .975)),
        "scan_groups": len(scan_names), "symbols": len(symbol_names), "replicates": len(vals)}


def main():
    data = load_frame("research_training.csv").sort_values(["scan_time", "candidate_id"]).reset_index(drop=True)
    live = load_frame("workbook_actual_evidence.csv")
    live["is_current_workbook"] = True
    archived = load_frame("archived_bot_evidence.csv")
    archived["is_current_workbook"] = False
    live = pd.concat([live, archived], ignore_index=True)
    assert not data.candidate_id.duplicated().any()
    dev, holdout = data.loc[data.scan_time < HOLDOUT].copy(), data.loc[data.scan_time >= HOLDOUT].copy()
    scans = sorted(dev.scan_time.unique())
    blocks = [list(b) for b in np.array_split(np.array(scans[-30:], dtype=object), 5)]
    specs = []
    for kind in ["logistic", "vote"]:
        for k in range(3, 42):
            for aux in [False, True]:
                specs.append({"id": f"{kind}_top{k}_live{int(aux)}", "estimator": kind, "profile": "rank", "k": k, "live": aux})
        for name, features in [("active20", ACTIVE), ("no_ev19", [f for f in ACTIVE if f != "ev_score"]),
            ("prior_no_ou19", [f for f in ACTIVE if f != "ou_halflife"]), ("all42", ALL)]:
            specs.append({"id": f"{kind}_{name}", "estimator": kind, "profile": "fixed", "features": features, "k": len(features), "live": False})
        specs.append({"id": f"{kind}_legacy_timing_sensitivity", "estimator": kind, "profile": "fixed", "features": [f for f in ACTIVE if f != "ev_score"], "k": 19, "live": False, "include_legacy_timing": True})
    plan = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "trials": specs, "trial_count": len(specs),
        "development_rows": len(dev), "holdout_rows": len(holdout), "total_rows": len(data),
        "holdout_start": HOLDOUT, "holdout_scans": sorted(holdout.scan_time.unique()), "development_blocks": blocks,
        "row_policy": "All validated unique labelable simulated rows; no 30-row symbol cap; no global complete-case deletion. Strict temporal eligibility applies separately in each fold.",
        "label": "Canonical FASTWIN: seven-hour endogenous net-MTM +3% threshold; actual bots use only independently provable positive bounds in explicit auxiliary tests.",
        "feature_selection": "Univariate absolute AUC from fit rows only; EV excluded from ranked family because active HMM was fitted after historical events.",
        "preprocessing": "Training-only median imputation, no missing indicators or invented observations. Actual-bot measurement timing remains diagnostic, not production-certified.",
        "validation": "Five chronological six-scan blocks; 12h purge after full seven-hour label end at fit/calibration/test boundaries; calibration uses 3-8 recent past scan groups.",
        "selection": "Among profiles without EV and without ambiguous legacy timing, highest mean fold AUC with pooled ECE<=0.10. If none pass, lowest ECE then highest mean AUC. Ties fewer features, no live augmentation, ID.",
        "full_refit": "Fit the frozen algorithm on every primary timestamp-eligible row, plus known live positives if selected. Fit final sigmoid on genuinely past-only raw OOF probabilities from development and the now-consumed holdout. Full refit is diagnostic and has no new independent score.",
        "holdout_policy": "Freeze selected spec before evaluating September 4-5. Compare selected, active20 and prior19, seeds42/7/123. No retuning after holdout.",
        "promotion_policy": "Historical research source and timing are diagnostic; numerical thresholds alone cannot override production contracts.",
        "data_sha256": hashlib.sha256((OUT / "research_training.csv").read_bytes()).hexdigest()}
    if (OUT / "experiment_plan.json").exists():
        old = json.loads((OUT / "experiment_plan.json").read_text())
        assert old["data_sha256"] == plan["data_sha256"], "Frozen study data changed"
    else:
        write("experiment_plan.json", plan)
    (OUT / "development_predictions").mkdir(exist_ok=True)
    results = json.loads((OUT / "development_results.json").read_text()) if (OUT / "development_results.json").exists() else []
    done = {r["id"] for r in results}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(trial, spec, dev, live, blocks): spec["id"] for spec in specs if spec["id"] not in done}
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            write("development_results.json", results)
            print(json.dumps({"completed": len(results), "trial": result["id"], "mean_auc": result["mean_fold_auc"], "ece": result["pooled"]["ece"]}), flush=True)
    selectable = [r for r in results if not r.get("include_legacy_timing") and "ev_score" not in r.get("features", [])]
    eligible = [r for r in selectable if r["pooled"]["ece"] <= .1]
    if eligible:
        best = sorted(eligible, key=lambda r: (-r["mean_fold_auc"], r["k"], r["live"], r["id"]))[0]
    else:
        best = sorted(selectable, key=lambda r: (r["pooled"]["ece"], -r["mean_fold_auc"], r["k"], r["id"]))[0]
    selected = next(s for s in specs if s["id"] == best["id"])
    baseline = next(s for s in specs if s["id"] == "vote_active20")
    prior = next(s for s in specs if s["id"] == "vote_prior_no_ou19")
    write("frozen_selection.json", {"spec": selected, "development": best, "holdout_not_yet_evaluated": True})
    table = holdout[["candidate_id", "symbol", "scan_time", "fast_winner_target", "study_source_pool"]].copy()
    holdout_results = []
    with threadpool_limits(limits=1):
        for seed in [42, 7, 123]:
            for role, spec in [("selected", selected), ("baseline", baseline), ("prior", prior)]:
                model, features, support = fit_model(spec, dev, holdout.scan_time.min(), live, seed)
                p = model.predict_proba(holdout[features].to_numpy(float))[:, 1]
                table[f"{role}_{seed}"] = p
                assert model.estimator is not None
                table[f"{role}_raw_{seed}"] = model.estimator.predict_proba(holdout[features].to_numpy(float))[:, 1]
                holdout_results.append({"role": role, "spec": spec, "seed": seed, **support,
                    **metrics(holdout.fast_winner_target.to_numpy(int), p), "auc_ci": interval(holdout, p)})
        table.to_csv(OUT / "holdout_predictions.csv", index=False)
        write("holdout_results.json", holdout_results)
        write("paired_holdout_comparison.json", [{"seed": seed, "auc_delta_ci": interval(holdout,
            table[f"selected_{seed}"].to_numpy(float), table[f"baseline_{seed}"].to_numpy(float))} for seed in [42, 7, 123]])
        primary = data.loc[~data.legacy_timestamp_uncertainty].copy()
        features = pick(selected, primary)
        auxiliary = live.loc[live.fast_winner_target.notna()].copy() if selected["live"] else live.iloc[:0]
        complete_fit = pd.concat([primary, auxiliary], ignore_index=True)
        model = base_model(selected, 42)
        model.fit(complete_fit[features].to_numpy(float), complete_fit.fast_winner_target.to_numpy(int))
        oof = read_development_prediction(selected["id"])
        raw_oof = np.concatenate([oof.raw_probability.to_numpy(float), table.selected_raw_42.to_numpy(float)])
        y_oof = np.concatenate([oof.fast_winner_target.to_numpy(int), table.fast_winner_target.to_numpy(int)])
        def logit(p):
            p = np.clip(p, 1e-6, 1-1e-6)
            return np.log(p / (1-p)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1., solver="lbfgs").fit(logit(raw_oof), y_oof)
        support = {"base_fit_rows": len(primary), "auxiliary_live_rows": len(auxiliary), "total_base_fit_rows": len(complete_fit),
            "past_only_oof_calibration_rows": len(y_oof), "full_data_model_has_independent_holdout": False}
        joblib.dump({"diagnostic_only": True, "promotion_eligible": False, "runtime_meta_labeler_compatible": False,
            "model": model, "calibrator": calibrator, "calibration_input": "logit of clipped raw model probability",
            "hmm_artifact_version": str(data.hmm_artifact_version.iloc[0]), "target_contract": "FASTWIN endogenous net-MTM +3% within seven hours",
            "features": features, "spec": selected, "data_sha256": plan["data_sha256"]}, OUT / "best_research_candidate.joblib")
        scoring = live[["strategy_id", "candidate_id", "symbol", "mode", "fast_winner_target", "label_evidence"]].copy()
        scoring["research_probability"] = calibrator.predict_proba(logit(model.predict_proba(live[features].to_numpy(float))[:, 1]))[:, 1]
        scoring["missing_selected_features"] = live[features].isna().sum(axis=1)
        scoring["independent_validation"] = False
        scoring.loc[live.is_current_workbook].to_csv(OUT / "workbook_all_366_scores.csv", index=False)
        scoring.loc[~live.is_current_workbook].to_csv(OUT / "archived_all_bot_scores.csv", index=False)
        write("candidate_manifest.json", {"spec": selected, "features": features, "feature_count": len(features),
            "support": support, "diagnostic_only": True, "promotion_eligible": False,
            "workbook_scored_rows": int(live.is_current_workbook.sum()), "archived_scored_rows": int((~live.is_current_workbook).sum()),
            "warning": "Workbook scores are descriptive, with imputation and possible training overlap; not an accuracy estimate."})
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
