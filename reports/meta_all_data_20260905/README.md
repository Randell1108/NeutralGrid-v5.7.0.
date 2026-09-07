# Expanded FASTWIN meta-labeler study

September 7 follow-up: [calibration diagnosis](calibration_addendum_20260907.md) finds that the saved **raw 24-feature probabilities have ECE 0.09766**. The three failing results below use the fixed sigmoid calibration step. This post-hoc finding expands the comparison's scope; it does not establish a new promotion pass.

Statistical study completed 2026-09-07T03:04:34.831762+00:00. Source inventory was checked again on September 6, 2026 (America/Lima). The final check found a newly saved September 6 23:30:11 UTC snapshot: all 250 rows were audited and scored. Its 111 candidates with recorded geometry have unfinished seven-hour outcome windows; 139 lack geometry. They provide zero additional usable FASTWIN labels as of this report. The latest fully labelled scan used for testing remains September 5 at 16:55:05 UTC. This report supersedes the narrower 1,231-complete-row study.

**The current diagnostic shadow has 20 features. The frozen best development selection among this study's eligible candidates uses 24 features: `vote_top24_live0`. No model was promoted.** These are the best results among the declared tests, not proof of a globally optimal feature set or an exact feature count sufficient for promotion.

## Data used and exclusions

| Evidence | Count | Treatment |
|---|---:|---|
| Existing Julyâ€“September seven-hour backtest rows | 4,356 | Included, active HMM refreshed |
| Additional successful historical candidate replays | 2,540 | Included in research inventory |
| Total unique simulated candidates | 6,896 | Deduplicated by persisted candidate ID |
| Primary rows with usable timestamp lineage | 6,587 | All eligible for development, holdout and final base fit |
| Februaryâ€“March rows with unresolved original timezone | 309 | Separate diagnostic sensitivity; excluded from model selection |
| Latest independent holdout | 490 | Four September 4â€“5 scans, replayed after artifact availability |
| Current bot workbook | 366 | All audited and scored; 79 positive FASTWIN bounds, 287 unknown |
| Additional archived bot identities | 27 | All audited and scored; 8 positive bounds, 19 unknown |
| Late September 6 snapshot | 250 | All audited and scored; 111 pending windows, 139 missing geometry |

The frozen 1,205-file inventory contained exact duplicates and 260 unique snapshot files. The final late-arrival audit increases that to 1,206 files and 261 unique snapshot sources. It covers the current checkout, its accessible mirror, saved deployment files and relevant project backups. Of 25,405 unique persisted snapshot candidate IDs, 18,495 lack recorded grid geometry; 6,910 have existing outcomes or were queued for replay. Fourteen replay attempts failed with invalid-symbol errors, leaving 6,896 successes. The 6,651 snapshot rows without persisted IDs also lack a verifiable UTC event timestamp under the archived producer code. They were not assigned fabricated identities or labels. The old 6,048-row pool index contains identities, not recoverable feature/outcome rows; exact matches and all exclusions are recorded in `all_source_dispositions.csv`.

All 6,896 simulated rows have the active HMM version `rolling_180d_20260903_153527` and finite regime probabilities. This is lineage consistency, not proof that the HMM existed at every historical event: 6,314 rows precede its training. Ranked selectable profiles exclude EV, which depends on the retrospective HMM. Fixed profiles containing EV are reported as diagnostic references and cannot win selection.

No global complete-case filter or 30-row-per-symbol cap was used. Missing measurements remain missing in the evidence matrix. Each model learns median replacements using only its fit partition; this statistical preprocessing is disclosed, not presented as observed market data. The final base model used **6,587 primary rows plus 0 auxiliary observed-positive rows**. The final calibration uses past-only OOF predictions; that full-data refit has no new independent performance score.

## FASTWIN and validation design

Every simulated label uses the canonical **net marked-to-market +3% threshold reached within seven hours**, derived from `time_to_target_hours`; it does not use terminal profitability or the stored generic `y` column. The canonical target differs from that stored generic label on 1,088 rows. All outcomes have the current label/formula/engine versions and a 420-bar seven-hour horizon, with genuine engine termination allowed. Replays use the repository's `legacy` realism profile, including its fee, funding, slippage, sizing and fill approximations; these are simulations, not actual bot returns.

The 87 known live positives are an explicitly biased auxiliary-data experiment. There is no verified negative live population here. They are never used to calibrate the model or claim live AUC. Unknown outcomes remain unknown, including bots that ended at a loss but might previously have crossed +3%. The 366 current and 27 archived scores are descriptive and may overlap training; they are not independent validation.

The frozen plan specifies 166 configurations across 3â€“42 features and two estimator families, with matched tests adding the known live positives. Five chronological development blocks yield 830 fit/calibration/test evaluations. Feature ranking and imputation are learned within the fit partition. Calibration uses 3â€“8 separate past scan groups. A 12-hour gap follows each full seven-hour event end at fit/calibration/test boundaries. The winner is selected before accessing holdout scores. The selected, active-20 and prior-19 references are evaluated with seeds 42, 7 and 123.

Historical development still follows the saved scan-anchored research contract; its timing limitations are not erased by this study. For the latest holdout, outcome windows start after recorded CSV availability, conservatively including the validation-manifest timestamp where present. This prevents awarding simulated profit before the saved candidate output existed.

That timing correction changes 88 of the 490 latest labels: 58 positives become negatives and 30 negatives become positives. The positive count falls from 229 to 201. `availability_label_sensitivity.json` records the paired comparison. This materially affects the apparent opportunity set and is one reason to keep historical development and the corrected holdout distinct.

## Results

The frozen selection's mean development-fold AUC is **0.6292**, pooled development AUC **0.6021**, and ECE **0.0592**. Its fold-specific feature selections are recorded in `development_results.json`; the final full-data feature list appears in `candidate_manifest.json`.

Seed-42 independent holdout results (490 rows, 201 positives):

| Model | Features | AUC | 95% two-way bootstrap interval | ECE | Brier |
|---|---:|---:|---|---:|---:|
| Frozen selection | 24 | 0.6999 | 0.5833â€“0.8034 | 0.1559 | 0.2335 |
| Active 20-feature baseline | 20 | 0.7155 | 0.5649â€“0.8292 | 0.1407 | 0.2288 |
| Prior 19-feature reference | 19 | 0.7173 | 0.5744â€“0.8283 | 0.1378 | 0.2279 |

The selected-minus-baseline AUC difference is **-0.0155**, with paired 95% interval **[-0.0589, 0.0467]**. Intervals independently resample scans and symbols (2,000 replicates). Only four held-out scan groups are available, so regime coverage remains limited. The within-scan permutation test gives p=0.0005; it preserves each scan's class balance and tests whether the frozen ranking contains information beyond scan-level differences.

Across the 78 matched ranked-feature comparisons, adding the 87 known-positive bot rows changed mean-fold AUC by a median **-0.0158**; it improved 0 of 78 pairs. This measures that particular, incomplete live augmentation, not the value of a future fully labelled live dataset.

![Development comparison](<D:/Neutral Grid v5.7.0/reports/meta_all_data_20260905/feature_comparison.png>)

## Promotion decision and verification

**Not promoted: the selected model's independent holdout ECE is 0.1559, above the 0.10 limit. Both reference models also fail that limit.** The 19-feature reference has the highest observed holdout AUC of these three, but it also fails calibration and was not selected by the development protocol. Historical research sources are not a finalized `fresh_full_pool` promotion input, and experimental feature sets are not automatically supported production profiles. The current native gate's known shuffled-fold/unpurged-fallback limitation remains unchanged; this study uses explicit chronological validation and does not treat a native numerical pass as proof of forward robustness. More features alone cannot resolve source, timing, schema or statistical requirements. The three seed repetitions returned identical predictions with these deterministic estimator settings; they are not three independent samples.

All 166 pooled result records and 9 holdout fits were independently recalculated from saved predictions. Candidate IDs, labels, purge boundaries, forbidden feature guards, source hashes, and all 393 serialized-model scoring results were checked. The relevant repository suite passed **115 tests**. Source-tree Pyright passed with zero errors; final research-script type-check results are retained. The original workbook, active HMM pointer and production meta-labeler source were not modified. `best_research_candidate.joblib` is marked diagnostic-only, promotion-ineligible and incompatible with runtime meta-labeler loading.

Under the conservative availability rule, the late September 6 snapshot's seven-hour replay window finishes at 2026-09-07 09:50:56.693464+00:00 (UTC), or 04:50:56 in Lima. Its row-level scores and exclusions are in `latest_september6_snapshot_audit.csv`; these are diagnostic scores, not deployment instructions.

The evidence matrix, all model comparison results, compressed development predictions, holdout predictions, scripts and report are retained. Temporary checkpoint, scratch and pytest files are removed after reporting; `cleanup.json` records the completed cleanup. To reproduce the statistical experiment, use the repository virtual environment with `experiment.py` and the retained evidence CSVs. The holdout has now been consumed and must not be used to tune another purportedly independent candidate.
