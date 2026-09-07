"""Independently recalculate the study results and render the final report."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import experiment as ex
from inventory import sha
from neutralgrid.models.meta_labeler import MetaLabeler, _is_label_column


def read(name):
    return json.loads((ex.OUT / name).read_text(encoding="utf-8"))


def main():
    data = ex.load_frame("research_training.csv")
    audit, plan = read("assembled_data_audit.json"), read("experiment_plan.json")
    results, frozen = read("development_results.json"), read("frozen_selection.json")
    holdout, paired = read("holdout_results.json"), read("paired_holdout_comparison.json")
    manifest = read("candidate_manifest.json")
    assert len(results) == plan["trial_count"] == len({r["id"] for r in results})
    assert sha(ex.OUT / "research_training.csv") == plan["data_sha256"]
    for path, digest in audit["original_source_hashes"].items():
        assert sha(Path(path)) == digest, path
    by_id = data.set_index("candidate_id")
    assert not by_id.index.duplicated().any()
    summaries, predictions = [], []
    for r in results:
        p = ex.read_development_prediction(r["id"])
        assert not p.candidate_id.duplicated().any()
        assert np.array_equal(by_id.loc[p.candidate_id, "fast_winner_target"].to_numpy(int), p.fast_winner_target.to_numpy(int))
        actual = ex.metrics(p.fast_winner_target.to_numpy(int), p.probability.to_numpy(float))
        for field in ["auc", "ece", "brier"]:
            assert np.isclose(actual[field], r["pooled"][field], atol=1e-12), (r["id"], field)
        for f in r["folds"]:
            assert cast(pd.Timestamp, pd.Timestamp(f["fit_end"])) < cast(pd.Timestamp, pd.Timestamp(f["calibration_start"]) - ex.PURGE)
            assert cast(pd.Timestamp, pd.Timestamp(f["calibration_end"])) < cast(pd.Timestamp, pd.Timestamp(f["test_start"]) - ex.PURGE)
            assert not any(_is_label_column(name) for name in f["features"])
            assert len(f["features"]) == r["k"]
        p["trial"] = r["id"]
        predictions.append(p)
        summaries.append({"trial": r["id"], "features": r["k"], "estimator": r["estimator"], "live_augmentation": r["live"],
            "legacy_timing_sensitivity": r.get("include_legacy_timing", False), "mean_fold_auc": r["mean_fold_auc"],
            "pooled_auc": actual["auc"], "ece": actual["ece"], "brier": actual["brier"], "oof_rows": len(p),
            "selected": r["id"] == frozen["spec"]["id"]})
    comparison = pd.DataFrame(summaries).sort_values("mean_fold_auc", ascending=False)
    comparison.to_csv(ex.OUT / "all_model_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(ex.OUT / "development_predictions.parquet", index=False)
    pred = pd.read_csv(ex.OUT / "holdout_predictions.csv")
    pred["scan_time"] = pd.to_datetime(pred.scan_time, utc=True)
    assert set(pred.candidate_id) == set(data.loc[data.scan_time >= ex.HOLDOUT, "candidate_id"])
    assert np.array_equal(by_id.loc[pred.candidate_id, "fast_winner_target"].to_numpy(int), pred.fast_winner_target.to_numpy(int))
    held_ids = set(pred.candidate_id)
    assert all(not held_ids & set(p.candidate_id) for p in predictions)
    for r in holdout:
        p = pred[f"{r['role']}_{r['seed']}"].to_numpy(float)
        m = ex.metrics(pred.fast_winner_target.to_numpy(int), p)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        for field in ["auc", "ece", "brier"]:
            assert np.isclose(m[field], r[field], atol=1e-12)
    selected_oof = next(p for p in predictions if p.trial.iloc[0] == frozen["spec"]["id"])
    selected_oof["scan_time"] = pd.to_datetime(selected_oof.scan_time, utc=True)
    ci = ex.interval(selected_oof, selected_oof.probability.to_numpy(float))
    dev_gate = MetaLabeler._evaluate_promotion_gate(oof_auc_ci_low=ci["low"], oof_auc_ci_high=ci["high"],
        n_pos=int(selected_oof.fast_winner_target.sum()), oof_ece=frozen["development"]["pooled"]["ece"])
    selected_holdout = [r for r in holdout if r["role"] == "selected"]
    diagnostics = []
    for r in selected_holdout:
        gate = MetaLabeler._evaluate_promotion_gate(oof_auc_ci_low=r["auc_ci"]["low"], oof_auc_ci_high=r["auc_ci"]["high"],
            n_pos=r["positives"], oof_ece=r["ece"])
        diagnostics.append({"seed": r["seed"], "threshold_analogue": gate})
    rng = np.random.default_rng(20260906)
    y = pred.fast_winner_target.to_numpy(int)
    p = pred.selected_42.to_numpy(float)
    groups = [np.asarray(g.index, dtype=int) for _, g in pred.groupby("scan_time")]
    null = []
    for _ in range(2000):
        shuffled = y.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(y[idx])
        null.append(float(roc_auc_score(shuffled, p)))
    observed = float(roc_auc_score(y, p))
    permutation = {"method": "Shuffle held-out labels within each scan, preserving scan-level class rates; frozen predictions unchanged",
        "replicates": len(null), "observed_auc": observed, "null_mean_auc": float(np.mean(null)),
        "p_one_sided": float((1 + np.sum(np.array(null) >= observed)) / (1 + len(null)))}
    ex.write("statistical_validation.json", {"development_auc_ci": ci,
        "development_thresholds_after_selection_are_descriptive": dev_gate,
        "independent_holdout_threshold_analogues": diagnostics, "within_scan_permutation": permutation,
        "promotion_completed": False, "production_admission": "Historical research sources are not a finalized fresh_full_pool promotion input; feature profiles are research-only.",
        "validated_development_trials": len(results), "validated_development_folds": sum(len(r["folds"]) for r in results),
        "validated_holdout_fits": len(holdout), "no_fit_calibration_test_overlap": True,
        "no_global_complete_case_deletion": True, "original_hashes_unchanged": True})
    # Validate serialization and all 393 scoring rows against the saved model.
    saved = joblib.load(ex.OUT / "best_research_candidate.joblib")
    assert saved["diagnostic_only"] and not saved["promotion_eligible"] and not saved["runtime_meta_labeler_compatible"]
    for name, scoring_file in [("workbook_actual_evidence.csv", "workbook_all_366_scores.csv"),
        ("archived_bot_evidence.csv", "archived_all_bot_scores.csv")]:
        frame = ex.load_frame(name)
        p0 = saved["model"].predict_proba(frame[saved["features"]].to_numpy(float))[:, 1]
        p0 = np.clip(p0, 1e-6, 1-1e-6)
        p1 = saved["calibrator"].predict_proba(np.log(p0 / (1-p0)).reshape(-1, 1))[:, 1]
        scores = pd.read_csv(ex.OUT / scoring_file)
        assert len(frame) == len(scores)
        assert np.allclose(p1, scores.research_probability.to_numpy(float), atol=1e-12)
    shadow_path = ex.ROOT / "artifacts/diagnostics/meta_prob_surrogate/hmm_rolling_180d_20260822_203741_20260827_185315/diagnostic_manifest.json"
    shadow = json.loads(shadow_path.read_text())
    ex.write("shadow_feature_verification.json", {"manifest": str(shadow_path), "sha256": sha(shadow_path),
        "feature_count": len(shadow["features"]), "features": shadow["features"],
        "diagnostic_only": shadow.get("diagnostic_only"), "canonical_model_present": (ex.ROOT / "models/meta_labeler.pkl").exists()})
    selected = next(r for r in selected_holdout if r["seed"] == 42)
    baseline = next(r for r in holdout if r["seed"] == 42 and r["role"] == "baseline")
    prior = next(r for r in holdout if r["seed"] == 42 and r["role"] == "prior")
    paired42 = next(r for r in paired if r["seed"] == 42)["auc_delta_ci"]
    matched_aux = []
    for r in results:
        if r["profile"] == "rank" and not r["live"]:
            other = next(o for o in results if o["profile"] == "rank" and o["live"] and o["estimator"] == r["estimator"] and o["k"] == r["k"])
            matched_aux.append(other["mean_fold_auc"] - r["mean_fold_auc"])
    comparison_rows = "\n".join(f"| {name} | {r['spec']['k']} | {r['auc']:.4f} | {r['auc_ci']['low']:.4f}–{r['auc_ci']['high']:.4f} | {r['ece']:.4f} | {r['brier']:.4f} |" for name, r in [("Frozen selection", selected), ("Active 20-feature baseline", baseline), ("Prior 19-feature reference", prior)])
    report = f"""# Expanded FASTWIN meta-labeler study

Completed {datetime.now(timezone.utc).isoformat()}. Source inventory was checked on September 6, 2026 (America/Lima). The latest available saved candidate scan is September 5 at 16:55:05 UTC. No September 6 candidate snapshot was present in the audited roots. This report supersedes the narrower 1,231-complete-row study.

**The current diagnostic shadow has {len(shadow['features'])} features. The frozen best development selection among this study's eligible candidates uses {manifest['feature_count']} features: `{frozen['spec']['id']}`. No model was promoted.** These are the best results among the declared tests, not proof of a globally optimal feature set or an exact feature count sufficient for promotion.

## Data used and exclusions

| Evidence | Count | Treatment |
|---|---:|---|
| Existing July–September seven-hour backtest rows | 4,356 | Included, active HMM refreshed |
| Additional successful historical candidate replays | 2,540 | Included in research inventory |
| Total unique simulated candidates | 6,896 | Deduplicated by persisted candidate ID |
| Primary rows with usable timestamp lineage | 6,587 | All eligible for development, holdout and final base fit |
| February–March rows with unresolved original timezone | 309 | Separate diagnostic sensitivity; excluded from model selection |
| Latest independent holdout | 490 | Four September 4–5 scans, replayed after artifact availability |
| Current bot workbook | 366 | All audited and scored; 79 positive FASTWIN bounds, 287 unknown |
| Additional archived bot identities | 27 | All audited and scored; 8 positive bounds, 19 unknown |

The 1,205-file inventory contains exact duplicates and 260 unique snapshot files. It covers the current checkout, its accessible mirror, saved deployment files and relevant project backups. Of 25,405 unique persisted snapshot candidate IDs, 18,495 lack recorded grid geometry; 6,910 have existing outcomes or were queued for replay. Fourteen replay attempts failed with invalid-symbol errors, leaving 6,896 successes. The 6,651 snapshot rows without persisted IDs also lack a verifiable UTC event timestamp under the archived producer code. They were not assigned fabricated identities or labels. The old 6,048-row pool index contains identities, not recoverable feature/outcome rows; exact matches and all exclusions are recorded in `all_source_dispositions.csv`.

All 6,896 simulated rows have the active HMM version `{audit['active_hmm']}` and finite regime probabilities. This is lineage consistency, not proof that the HMM existed at every historical event: {audit['retrospective_hmm_rows']:,} rows precede its training. Ranked selectable profiles exclude EV, which depends on the retrospective HMM. Fixed profiles containing EV are reported as diagnostic references and cannot win selection.

No global complete-case filter or 30-row-per-symbol cap was used. Missing measurements remain missing in the evidence matrix. Each model learns median replacements using only its fit partition; this statistical preprocessing is disclosed, not presented as observed market data. The final base model used **{manifest['support']['base_fit_rows']:,} primary rows plus {manifest['support']['auxiliary_live_rows']} auxiliary observed-positive rows**. The final calibration uses past-only OOF predictions; that full-data refit has no new independent performance score.

## FASTWIN and validation design

Every simulated label uses the canonical **net marked-to-market +3% threshold reached within seven hours**, derived from `time_to_target_hours`; it does not use terminal profitability or the stored generic `y` column. The canonical target differs from that stored generic label on {audit['target_summary']['builder_y_mismatch_count']:,} rows. All outcomes have the current label/formula/engine versions and a 420-bar seven-hour horizon, with genuine engine termination allowed. Replays use the repository's `legacy` realism profile, including its fee, funding, slippage, sizing and fill approximations; these are simulations, not actual bot returns.

The 87 known live positives are an explicitly biased auxiliary-data experiment. There is no verified negative live population here. They are never used to calibrate the model or claim live AUC. Unknown outcomes remain unknown, including bots that ended at a loss but might previously have crossed +3%. The 366 current and 27 archived scores are descriptive and may overlap training; they are not independent validation.

The frozen plan specifies {len(results)} configurations across 3–42 features and two estimator families, with matched tests adding the known live positives. Five chronological development blocks yield {sum(len(r['folds']) for r in results)} fit/calibration/test evaluations. Feature ranking and imputation are learned within the fit partition. Calibration uses 3–8 separate past scan groups. A 12-hour gap follows each full seven-hour event end at fit/calibration/test boundaries. The winner is selected before accessing holdout scores. The selected, active-20 and prior-19 references are evaluated with seeds 42, 7 and 123.

Historical development still follows the saved scan-anchored research contract; its timing limitations are not erased by this study. For the latest holdout, outcome windows start after recorded CSV availability, conservatively including the validation-manifest timestamp where present. This prevents awarding simulated profit before the saved candidate output existed.

That timing correction changes 88 of the 490 latest labels: 58 positives become negatives and 30 negatives become positives. The positive count falls from 229 to 201. `availability_label_sensitivity.json` records the paired comparison. This materially affects the apparent opportunity set and is one reason to keep historical development and the corrected holdout distinct.

## Results

The frozen selection's mean development-fold AUC is **{frozen['development']['mean_fold_auc']:.4f}**, pooled development AUC **{frozen['development']['pooled']['auc']:.4f}**, and ECE **{frozen['development']['pooled']['ece']:.4f}**. Its fold-specific feature selections are recorded in `development_results.json`; the final full-data feature list appears in `candidate_manifest.json`.

Seed-42 independent holdout results ({len(pred)} rows, {int(pred.fast_winner_target.sum())} positives):

| Model | Features | AUC | 95% two-way bootstrap interval | ECE | Brier |
|---|---:|---:|---|---:|---:|
{comparison_rows}

The selected-minus-baseline AUC difference is **{selected['auc']-baseline['auc']:+.4f}**, with paired 95% interval **[{paired42['low']:.4f}, {paired42['high']:.4f}]**. Intervals independently resample scans and symbols (2,000 replicates). Only four held-out scan groups are available, so regime coverage remains limited. The within-scan permutation test gives p={permutation['p_one_sided']:.4f}; it preserves each scan's class balance and tests whether the frozen ranking contains information beyond scan-level differences.

Across the 78 matched ranked-feature comparisons, adding the 87 known-positive bot rows changed mean-fold AUC by a median **{float(np.median(matched_aux)):+.4f}**; it improved {sum(v>0 for v in matched_aux)} of {len(matched_aux)} pairs. This measures that particular, incomplete live augmentation, not the value of a future fully labelled live dataset.

![Development comparison](<{(ex.OUT / 'feature_comparison.png').as_posix()}>)

## Promotion decision and verification

**Not promoted: the selected model's independent holdout ECE is {selected['ece']:.4f}, above the 0.10 limit. Both reference models also fail that limit.** The 19-feature reference has the highest observed holdout AUC of these three, but it also fails calibration and was not selected by the development protocol. Historical research sources are not a finalized `fresh_full_pool` promotion input, and experimental feature sets are not automatically supported production profiles. The current native gate's known shuffled-fold/unpurged-fallback limitation remains unchanged; this study uses explicit chronological validation and does not treat a native numerical pass as proof of forward robustness. More features alone cannot resolve source, timing, schema or statistical requirements. The three seed repetitions returned identical predictions with these deterministic estimator settings; they are not three independent samples.

All {len(results)} pooled result records and {len(holdout)} holdout fits were independently recalculated from saved predictions. Candidate IDs, labels, purge boundaries, forbidden feature guards, source hashes, and all 393 serialized-model scoring results were checked. The relevant repository suite passed **115 tests**. Source-tree Pyright passed with zero errors; final research-script type-check results are retained. The original workbook, active HMM pointer and production meta-labeler source were not modified. `best_research_candidate.joblib` is marked diagnostic-only, promotion-ineligible and incompatible with runtime meta-labeler loading.

The evidence matrix, all model comparison results, compressed development predictions, holdout predictions, scripts and report are retained. Temporary checkpoint, scratch and pytest files are removed after reporting; `cleanup.json` records the completed cleanup. To reproduce the statistical experiment, use the repository virtual environment with `experiment.py` and the retained evidence CSVs. The holdout has now been consumed and must not be used to tune another purportedly independent candidate.
"""
    (ex.OUT / "README.md").write_text(report, encoding="utf-8")
    ex.write("validation_completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "development_trials": len(results), "folds": sum(len(r["folds"]) for r in results), "holdout_fits": len(holdout),
        "serialized_scores_verified": 393, "report": str(ex.OUT / "README.md"), "cleanup_pending": True})
    print(json.dumps({"selected": frozen["spec"], "heldout": selected, "paired": paired42}, default=str), flush=True)


if __name__ == "__main__":
    main()
