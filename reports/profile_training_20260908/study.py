"""Isolated, pre-registered profile study: assemble, develop, then disclose holdout."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from neutralgrid.core.constants import ENGINE_VERSION, FORMULA_VERSION, LABEL_CONTRACT_VERSION
from neutralgrid.scanner.canonical_fastwin_profile import _build_pattern
from neutralgrid.scanner.fastwin_logistic_profile import _fit_model, _temporal_splits
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, PatternProfile
from neutralgrid.scanner.profile_model import ProfileModel, load_profile_model, save_profile_model
from neutralgrid.scanner.profile_model_walkforward import _train_from_frame, _expected_calibration_error

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "reports/meta_canonical_refit_20260907"
FEATURES = list(DEFAULT_FEATURES)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def load_frame(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name, low_memory=False)
    frame["start_time_utc"] = pd.to_datetime(frame.start_time_utc, utc=True, format="ISO8601")
    assert frame.candidate_id.is_unique
    assert np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all()
    assert frame.fastwin_label.isin([0, 1]).all()
    return frame.sort_values(["start_time_utc", "candidate_id"]).reset_index(drop=True)


def assemble() -> None:
    cohort = pd.read_csv(OUT / "frozen_feature_cohort.csv")
    old_path = PREVIOUS / "fresh_all_outcomes/training_data_20260907.csv"
    old = pd.read_csv(old_path, low_memory=False)
    extra_path = OUT / "additional_backtest_results.jsonl"
    extra = pd.DataFrame([json.loads(line) for line in extra_path.read_text().splitlines()])
    cols = ["candidate_id", "symbol", "start_time_utc", "time_to_target_hours", "target_reached", "source", "is_authoritative", "engine_version", "label_contract_version", "formula_version", "realism_profile", *FEATURES]
    merged = pd.concat([old[cols], extra[cols]], ignore_index=True)
    merged = merged.loc[merged.candidate_id.isin(cohort.candidate_id)]
    assert merged.candidate_id.is_unique
    for key, value in {"source": "backtest", "engine_version": ENGINE_VERSION, "formula_version": FORMULA_VERSION, "label_contract_version": LABEL_CONTRACT_VERSION, "realism_profile": "legacy"}.items():
        assert merged[key].astype(str).eq(value).all(), key
    assert merged.is_authoritative.astype(str).str.lower().isin(["true", "1"]).all()
    joined = cohort.merge(merged, on="candidate_id", suffixes=("", "_outcome"), validate="one_to_one")
    assert joined.symbol.eq(joined.symbol_outcome).all()
    assert pd.to_datetime(joined.start_time_utc, utc=True, format="ISO8601").eq(pd.to_datetime(joined.start_time_utc_outcome, utc=True, format="ISO8601")).all()
    for feature in FEATURES:
        assert np.allclose(joined[feature], joined[feature + "_outcome"], rtol=1e-10, atol=1e-12), feature
    t2t = pd.to_numeric(joined.time_to_target_hours, errors="coerce")
    reached = joined.target_reached.astype(str).str.lower().isin(["true", "1"])
    assert reached.eq(np.isfinite(t2t)).all()
    joined["fastwin_label"] = (np.isfinite(t2t) & t2t.le(7)).astype(int)
    joined["_is_winner"] = joined.fastwin_label
    kept = ["candidate_id", "symbol", "start_time_utc", "scan_timestamp", "split", "fastwin_label", "_is_winner", "original_source_path", "original_source_sha256", *FEATURES]
    joined = joined[kept].sort_values(["start_time_utc", "candidate_id"])
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(joined), "old_source": {"path": str(old_path), "sha256": sha(old_path)}, "new_source": {"path": str(extra_path), "sha256": sha(extra_path)}, "excluded_failed_ids": sorted(set(cohort.candidate_id) - set(joined.candidate_id)), "splits": {}}
    for split in ("development", "holdout", "boundary_purge"):
        frame = joined.loc[joined.split.eq(split)]
        path = OUT / f"{split}.csv"
        assert not path.exists()
        frame.to_csv(path, index=False)
        manifest["splits"][split] = {"path": str(path), "rows": len(frame), "sha256": sha(path)}
    joined.to_csv(OUT / "complete_profile_pool.csv", index=False)
    write_json(OUT / "assembled_manifest.json", manifest)
    print(json.dumps({"rows": len(joined), "splits": {k: v["rows"] for k, v in manifest["splits"].items()}, "failed_rows": len(manifest["excluded_failed_ids"])}), flush=True)


def score(model: ProfileModel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feats = model.features
    assert model.feature_mean is not None and model.feature_std is not None
    x = frame[feats].to_numpy(dtype=float)
    x = (x - np.array([model.feature_mean[f] for f in feats])) / np.array([model.feature_std[f] for f in feats])
    if model.model_family == "robust_logistic_v1":
        assert model.linear_coef is not None and model.linear_intercept is not None
        logits = x @ np.array([model.linear_coef[f] for f in feats]) + model.linear_intercept
    else:
        winner = np.array([model.winner_mu[f] for f in feats])
        loser = np.array([model.loser_mu[f] for f in feats])
        inv = np.asarray(model.inv_cov)
        prior = min(.999, max(.001, model.prior_winner))
        logits = x @ inv @ (winner - loser) - .5 * (winner @ inv @ winner - loser @ inv @ loser) + np.log(prior / (1 - prior))
    probabilities = expit(np.clip(logits, -50, 50))
    assert np.isfinite(logits).all() and np.isfinite(probabilities).all()
    # Verify vectorized calculations against the actual runtime API on every call.
    for idx in np.linspace(0, len(frame) - 1, min(5, len(frame)), dtype=int):
        row = frame.iloc[idx].to_dict()
        assert np.isclose(logits[idx], model.llr(row), rtol=1e-9, atol=1e-9)
        assert np.isclose(probabilities[idx], model.proba(row), rtol=1e-9, atol=1e-9)
    return logits, probabilities


def metrics(y: np.ndarray, scores: np.ndarray, probabilities: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, scores)) if np.unique(y).size == 2 else None, "brier": float(np.mean((probabilities - y) ** 2)), "ece": float(_expected_calibration_error(probabilities, y)), "rows": len(y), "positives": int(y.sum())}


def fit(frame: pd.DataFrame, family: str, parameter: float, features: list[str] | None = None) -> ProfileModel:
    if family == "logistic":
        return _fit_model(frame, c_value=parameter)
    model = _train_from_frame(frame, features or FEATURES, shrinkage=parameter, max_duration_hours=7)
    if model is None:
        raise ValueError("Insufficient classes for Gaussian fit")
    return model


def splits(frame: pd.DataFrame, count: int) -> list:
    output = _temporal_splits(frame, n_folds=count)
    for train, test in output:
        assert train.start_time_utc.max() + pd.Timedelta(hours=24) < test.start_time_utc.min()
        assert not set(train.candidate_id) & set(test.candidate_id)
        assert train.fastwin_label.value_counts().min() >= 30 and train.fastwin_label.nunique() == 2
    return output


def evaluate_grid(frame: pd.DataFrame, family: str, grid: list[float], count: int) -> tuple[float, list[dict]]:
    folded = splits(frame, count)
    reports = []
    for parameter in grid:
        results = []
        for train, test in folded:
            model = fit(train, family, parameter)
            raw, probabilities = score(model, test)
            results.append(metrics(test.fastwin_label.to_numpy(dtype=int), raw, probabilities))
        aucs = [r["auc"] for r in results if r["auc"] is not None]
        reports.append({"family": family, "parameter": parameter, "mean_auc": float(np.mean(aucs)) if aucs else -1.0, "mean_brier": float(np.mean([r["brier"] for r in results])), "folds": results})
    best = sorted(reports, key=lambda x: (-x["mean_auc"], x["mean_brier"], x["parameter"]))[0]
    return best["parameter"], reports


def development() -> None:
    assert not (OUT / "selection.json").exists()
    protocol = json.loads((OUT / "preregistration.json").read_text())
    manifest = json.loads((OUT / "assembled_manifest.json").read_text())
    assert sha(OUT / "development.csv") == manifest["splits"]["development"]["sha256"]
    # No holdout file is opened in this phase.
    frame = load_frame("development.csv")
    grids = {"gaussian": protocol["gaussian_shrinkage_grid"], "logistic": protocol["logistic_c_grid"]}
    outer = splits(frame, 5)
    families = []
    for family, grid in grids.items():
        folds, predictions = [], []
        for number, (train, test) in enumerate(outer):
            parameter, inner = evaluate_grid(train, family, grid, 4)
            model = fit(train, family, parameter)
            raw, probability = score(model, test)
            result = metrics(test.fastwin_label.to_numpy(dtype=int), raw, probability)
            folds.append({"fold": number, "parameter": parameter, "training_rows": len(train), "training_end": train.start_time_utc.max().isoformat(), "test_start": test.start_time_utc.min().isoformat(), "test_end": test.start_time_utc.max().isoformat(), "result": result, "inner_grid": inner})
            scored = test[["candidate_id", "start_time_utc", "fastwin_label"]].copy()
            scored["score"], scored["probability"], scored["fold"] = raw, probability, number
            predictions.append(scored)
            print(json.dumps({"family": family, "fold": number, "selected_parameter": parameter, **result}), flush=True)
        pooled = pd.concat(predictions, ignore_index=True)
        assert pooled.candidate_id.is_unique
        pooled.to_csv(OUT / f"{family}_development_oof.csv", index=False)
        aucs = [f["result"]["auc"] for f in folds if f["result"]["auc"] is not None]
        final_parameter, final_grid = evaluate_grid(frame, family, grid, 5)
        families.append({"family": family, "final_parameter": final_parameter, "mean_auc": float(np.mean(aucs)), "finite_folds": len(aucs), "pass_rate": float(np.mean(np.asarray(aucs) >= .55)), "pooled": metrics(pooled.fastwin_label.to_numpy(dtype=int), pooled.score.to_numpy(), pooled.probability.to_numpy()), "folds": folds, "final_development_grid": final_grid})
    best = sorted(families, key=lambda x: (-x["mean_auc"], x["pooled"]["brier"]))[0]
    selected = fit(frame, best["family"], best["final_parameter"])
    summary = {"artifact_role": "shadow", "target_contract": "fast_winner_time_to_3pct_le_7h", "features": FEATURES, "training_rows": len(frame), "label_counts": frame.fastwin_label.value_counts().to_dict(), "family": best["family"], "parameter": best["final_parameter"], "source_sha256": sha(OUT / "development.csv"), "promotion_blockers": ["Target differs from canonical completed-bot profile", "No wholly unseen prospective bot-disjoint validation"]}
    selected = dataclasses.replace(selected, selection_summary=summary)
    destination = OUT / "frozen_shadow"
    destination.mkdir(exist_ok=False)
    save_profile_model(selected, destination / "profile_model.json")
    _build_pattern(frame, summary).save_json(destination / "pattern_profile.json")
    baseline = fit(frame, "gaussian", .3)
    save_profile_model(baseline, destination / "baseline_profile_model.json")
    ablation = []
    for train, test in outer:
        model = fit(train, "gaussian", .3, [f for f in FEATURES if f != "funding_carry_expected_next_7h"])
        raw, probability = score(model, test)
        ablation.append(metrics(test.fastwin_label.to_numpy(dtype=int), raw, probability))
    selection = {"selected_at_utc": datetime.now(timezone.utc).isoformat(), "family": best["family"], "parameter": best["final_parameter"], "training_rows": len(frame), "training_positive_rate": float(frame.fastwin_label.mean()), "model_sha256": sha(destination / "profile_model.json"), "pattern_sha256": sha(destination / "pattern_profile.json"), "baseline_sha256": sha(destination / "baseline_profile_model.json"), "holdout_sha256": manifest["splits"]["holdout"]["sha256"], "nested_result": {k: v for k, v in best.items() if k not in {"folds", "final_development_grid"}}, "holdout_opened": False}
    write_json(OUT / "development_evaluation.json", {"families": families, "funding_excluded_ablation": ablation, "ablation_role": "diagnostic_only; three features fail promotion coverage"})
    write_json(OUT / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def holdout() -> None:
    assert not (OUT / "holdout_evaluation.json").exists(), "Holdout already disclosed"
    selection = json.loads((OUT / "selection.json").read_text())
    destination = OUT / "frozen_shadow"
    for path, key in [(destination / "profile_model.json", "model_sha256"), (destination / "baseline_profile_model.json", "baseline_sha256"), (OUT / "holdout.csv", "holdout_sha256")]:
        assert sha(path) == selection[key]
    test = load_frame("holdout.csv")
    model = load_profile_model(destination / "profile_model.json")
    baseline = load_profile_model(destination / "baseline_profile_model.json")
    raw, probability = score(model, test)
    base_raw, base_probability = score(baseline, test)
    y = test.fastwin_label.to_numpy(dtype=int)
    prevalence = np.full(len(y), selection["training_positive_rate"])
    clusters = test.start_time_utc.astype(str).to_numpy()
    groups = np.unique(clusters)
    group_indices = [np.flatnonzero(clusters == group) for group in groups]
    rng = np.random.default_rng(20260801)
    intervals = []
    for _ in range(5000):
        sample = np.concatenate([group_indices[i] for i in rng.integers(0, len(groups), len(groups))])
        if np.unique(y[sample]).size != 2:
            continue
        auc = roc_auc_score(y[sample], raw[sample])
        delta = auc - roc_auc_score(y[sample], base_raw[sample])
        intervals.append([auc, delta])
    ci = np.quantile(np.asarray(intervals), [.025, .975], axis=0)
    scored = test[["candidate_id", "start_time_utc", "fastwin_label"]].copy()
    scored["selected_probability"], scored["baseline_probability"] = probability, base_probability
    scored.to_csv(OUT / "holdout_predictions.csv", index=False)
    report = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "selection": selection, "selected": metrics(y, raw, probability), "gaussian_0_3_baseline": metrics(y, base_raw, base_probability), "train_prevalence_baseline": metrics(y, prevalence, prevalence), "cluster_bootstrap": {"cluster_unit": "recorded outcome-start timestamp", "clusters": len(groups), "replicates": len(intervals), "auc_ci95": ci[:, 0].tolist(), "paired_auc_delta_ci95": ci[:, 1].tolist()}, "promotion_status": "shadow_only", "production_blockers": ["Expanded FASTWIN target differs from canonical completed-bot profile labels", "Holdout includes prior meta-labeler outcome exposure and is not a wholly unseen prospective bot cohort"], "no_post_holdout_tuning": True}
    write_json(OUT / "holdout_evaluation.json", report)
    full = load_frame("complete_profile_pool.csv")
    full_model = fit(full, selection["family"], selection["parameter"])
    full_summary = dict(model.selection_summary or {}, training_rows=len(full), label_counts=full.fastwin_label.value_counts().to_dict(), source_sha256=sha(OUT / "complete_profile_pool.csv"), artifact_role="shadow_full_refit", validation_artifact="frozen_shadow/profile_model.json", refit_includes_disclosed_holdout=True)
    full_model = dataclasses.replace(full_model, selection_summary=full_summary)
    full_dir = OUT / "full_refit_shadow"
    full_dir.mkdir(exist_ok=False)
    save_profile_model(full_model, full_dir / "profile_model.json")
    _build_pattern(full, full_summary).save_json(full_dir / "pattern_profile.json")
    write_json(full_dir / "manifest.json", {"status": "shadow_only", "training_rows": len(full), "features": FEATURES, "family": selection["family"], "parameter": selection["parameter"], "model_sha256": sha(full_dir / "profile_model.json"), "pattern_sha256": sha(full_dir / "pattern_profile.json"), "evaluation": str(OUT / "holdout_evaluation.json"), "validation_scope": "Holdout metrics belong to the frozen development-trained model; the full refit includes holdout rows."})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["assemble", "development", "holdout"])
    args = parser.parse_args()
    {"assemble": assemble, "development": development, "holdout": holdout}[args.phase]()
