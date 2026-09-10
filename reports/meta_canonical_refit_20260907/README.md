# Canonical meta-labeler refit completed — September 8, 2026

Artifact `20260908_143214` was trained, passed the existing promotion gate and was saved to `models/meta_labeler.pkl` and `models/meta_labeler/`. It uses the canonical **20-feature FASTWIN profile**, the logistic-regression/HistGradientBoosting voting ensemble, and **sigmoid_oos calibration**. All **5,369 eligible labeled rows** entered the fit; the final fitted scaler independently records 5,369 samples.

This completes the requested canonical refit. It does not establish that this recipe is the best possible model, or that it beats the previous champion in a paired test. The canonical champion pickle and metadata were absent before this run, so the script recorded `comparison_scope=initial_deployment` and applied its absolute promotion gate.

| Measurement | Observed result | Existing requirement |
|---|---:|---:|
| Promotion OOF AUC | 0.686774 | Lower confidence bound > 0.50 |
| Promotion AUC 95% bootstrap interval | 0.672276–0.699758 | Lower bound > 0.50 |
| Promotion OOF calibration error (ECE) | **0.013506** | **<= 0.10** |
| Positive training labels | 2,835 | >= 70 |
| Negative training labels | 2,534 | Both classes present |
| Separate training CV AUC | **0.553453** | Reported separately from promotion AUC |
| Precision@5 | 0.533333 | Diagnostic |
| Canonical promotion decision | **PASS** | All gate conditions satisfied |

The promotion evaluation warned that **time purging collapsed in all five folds**; the folds trained without that purge while retaining symbol grouping. This is the existing ERR-066 behavior. Its manifest string contains `chronological_contiguous_group_purged_weighted_kfold_v2`, but that string must not be read as proof of chronological purging. The promotion AUC can be optimistic; the separate CV AUC of 0.553453 is material context. No feature, split, calibration or promotion implementation was changed for this refit.

## Data coverage and exclusions

The final inventory check completed at **2026-09-08 14:31:17 UTC** (09:31 Lima). It found **no additional exports and no changed or missing original sources**. The latest available scanner snapshot remained **2026-09-07 14:34:01 UTC**. Thus this is all compatible data available at the check, not a claim that September 8 scanner observations existed.

- Audited 1,074 scanner-export paths across the workspace, project mirror, deployment archive and recorded backups. The original 1,065 paths were unchanged; nine additional paths were identical copies of one undated 100-row export with no candidate IDs and no recorded grid geometry. That export contributes zero eligible rows.
- The canonical input loader received 211 distinct original CSVs and 21 CSV conversions from audited Excel snapshots. The Excel audit found 289 additional candidate IDs, of which 84 met canonical selection. Source timestamps were preserved, and the conversion manifest records source and output hashes.
- The resulting inventory contained 32,556 rows. The disposition ledger records 9,362 below/missing the score floor, 17,520 lacking required grid geometry, 301 otherwise eligible rows with unparseable IDs, and **5,373 selected candidates**. These categories are sequential and sum to the inventory size.
- Canonical fresh backtests generated **5,369 unique outcomes across 472 symbols**. Ninety-one initially incomplete windows succeeded after maturation; one HOLOUSDT fetch failure succeeded on retry. Four rows containing corrupted symbol text were rejected by Binance and remain excluded. Their original IDs and exact errors are retained; no replacement symbols were guessed.
- Coverage by scan month: February 236; March 73; July 855; August 3,058; September 1,147. Earliest scan: February 18, 2026 at 19:24:30 UTC. The absence of April–June rows reflects compatible snapshot availability, not an assertion that old outcome tables did not exist.

The canonical workbook `data/new_expired_bots.xlsx` has **366 General-sheet rows**, including **214 nonempty, distinct candidate IDs**. **99 workbook rows** have compatible original scanner snapshots; all 99 received fresh backtest outcomes. The script's workbook route added 23 candidates beyond standard selection. The workbook contains no observed `time_to_target_hours` values, so the remaining actual bot outcomes were not converted into invented FASTWIN path labels. The workbook was used in the canonical reference/selection role and remains byte-for-byte unchanged.

The older **6,048-ID historical pool index** yielded **209 IDs with compatible original snapshots**; all 209 received new canonical outcomes. Historical outcome values were not substituted for fresh backtests. The row-level workbook, historical-pool and scanner ledgers make these inclusions and exclusions explicit.

## Training and calibration

Every component backtest used the unchanged `backtest_candidates.py --fastwin-full-pool` path: canonical legacy-authority geometric simulation, seven hours, 420 one-minute bars, no candidate cap, 10x leverage and the original recorded grid geometry. Outcome windows started at recorded scanner-output availability. Because the canonical model was absent, the runner applied its documented bootstrap selection behavior; the missing deployment linkage log also triggered its documented skip of linkage exclusion. Both conditions are recorded in the run log.

The three original fresh-run manifests and files are retained. Their **disjoint union** is explicitly identified as an aggregation with component paths, hashes and counts; no original manifest was rewritten. The full run, matured-window retry and fetch recovery all use identical outcome contracts. The canonical finalizer accepted the aggregated source after backfill and validation.

Backfill used `rolling_180d_20260903_153527`, `--hmm-only`, `--feature-cutoff-source candidate_id_scan_time`, `--replay-scope hmm_lineage_only` and fresh output paths. All 5,369 rows have the active HMM lineage, finite regime probabilities, matching artifact training timestamps and feature cutoffs equal to their scan timestamps. All 19 independent active features and outcome fields were preserved within recorded floating-point serialization tolerance. The HMM-dependent EV feature was refreshed. The active HMM was trained after most historical scans: this is the required pinned-artifact replay, not a simulation of which HMM would have been available at every historical scan.

The finalized input had **144 missing OU half-life values**. The explicit finalizer/retrainer imputation options retained those rows using the script's fixed `ou_halflife=24.0` default. These are imputations, not measured values. All 20 final model input columns are finite; no rows were removed by the modelable-feature filter. The per-symbol training cap was explicitly disabled with `--max-rows-per-symbol 0`.

The saved estimator was inspected directly: equal-weight soft voting between balanced logistic regression and HistGradientBoosting (350 iterations, 10 leaves, learning rate 0.03, minimum leaf size 60, L2=3, max_features=0.8), with imputation and standardization. The final wrapper applies a logistic sigmoid calibrator to the raw ensemble probability. It does not apply the temporary temporal-holdout calibrator before the OOS calibrator.

On the **10,578 pooled CV prediction records** used by the calibration routine, raw ECE was **0.133206**, calibrated ECE **0.011921**, and Brier score improved from **0.271341 to 0.246739**. These prediction records include repeated observations across CV folds; they are not 10,578 distinct training rows. Sigmoid calibration was accepted; no beta upgrade was applied. The calibration-fit ECE is distinct from the promotion OOF ECE of **0.013506**, and neither is a new untouched prospective test.

Exact final training command:

```powershell
.venv/Scripts/python.exe retrain_meta_labeler.py --input data/new_expired_bots.xlsx --backtest-results-dir reports/meta_canonical_refit_20260907/finalized_fresh_pool --max-rows-per-symbol 0 --allow-imputation --export-training-data reports/meta_canonical_refit_20260907/prepared_training.csv
```

The canonical metadata leaves `code_commit`, training date ranges and symbol lists null. This report, the pool manifest and feature-verification record supply the coverage, source hashes and execution evidence. Workspace HEAD was `4620a938724293340fec9c0b8c89a9070ebb62ff`; pre-existing working-tree edits were preserved. The meta-labeler implementation and retrain entry point have unchanged before/after hashes.

## Validation and remaining limitations

- **19 artifact checks passed**, including exact feature order, all 5,369 candidate IDs retained, target recomputation, all-row scaler fitting, active HMM compatibility, promotion status, artifact health and trial logging.
- Reloaded artifact and legacy pickle probabilities match exactly on all 5,369 prepared rows; all are finite and within [0,1]. Their range is 0.426940–0.610588. This is a load/inference smoke test on training rows, not out-of-sample performance.
- Post-refit tests: **96 passed**, including the dated FASTWIN retrain contract, calibration gate, fresh-pool source contract, finalizer and AFML compliance tests. Preflight: 87 passed. Pyright: **0 errors, 0 warnings**.
- The earlier full-suite run had 1,965 passed, 11 failed and 2 skipped. Ten Windows path-length failures passed on a shorter temporary path. One pre-existing utility test remains failing: `test_canonical_workbook_exposes_governed_utility_pool`, because the workbook's utility-eligible rows carry August 27 HMM lineage while September 3 is active. The full suite is therefore **not fully green**. This refit does not repair or promote the utility calibrator.
- The canonical workbook, active HMM manifest and runtime conformal-cache state are unchanged. No production source code was modified. Trial `meta_labeler_20260908_143214` was logged.
- The prior research study's held-out calibration numbers and the previous champion's stored metrics use different data/evaluation contexts. This result does not demonstrate paired superiority over those results, profitability, or an untouched prospective holdout.

Evidence in this report directory: `artifact_validation.json`, `model_metadata.json`, `promotion_decision.json`, `model_verification.json`, `final_lineage_audit.json`, `fresh_pool_audit.json`, `final_inventory_refresh.json`, `finalized_fresh_pool/authoritative_pool_manifest.json`, source/conversion inventories, row disposition ledgers, all component run manifests and execution/test logs. The completed training datasets and these audit records are retained. Temporary scanner copies, isolated runtime files and test directories are removed separately and recorded in `cleanup.json`.