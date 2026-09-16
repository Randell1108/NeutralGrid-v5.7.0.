Retraining and validation — September 11, 2026

The HMM and meta-labeler were retrained and promoted by the existing canonical gates. The utility calibrator and both profile models were fitted but remain unpromoted. The complete pipeline still lacks a promoted profile model and utility calibrator. No deployment was run, and production source code and gate thresholds were not changed.

| Component | Result | Data and measured statistics |
|---|---|---|
| HMM `rolling_180d_20260911_131128` | Promoted | 5 features, 4 states, 50 symbols, 858,000 feature observations; 180-day training window ending September 10 at 23:45 UTC |
| Meta-labeler `20260911_145206` | Promoted | 20 features; all 840 eligible distinct rows; 417 FASTWIN positives; OOF AUC **0.641694**, reported interval **[0.604133, 0.679607]**; OOF ECE **0.031889** |
| Utility `utility_20260911_140229_939458` | Candidate only | 222 eligible bots; 166 fit / 56 holdout; holdout AUC **0.571615**; G0–G6 pass, G7 fails |
| Canonical profile + pattern pair | Shadow | 4 features, 195 labeled bots: 62 winners / 133 losers; zero fresh finite evaluation folds, versus three required |
| Expanded FASTWIN profile + pattern pair | Shadow | 4 features, **4,109** distinct candidates: 2,009 positives / 2,100 negatives; 3,710 fresh complete rows plus 399 retained historical rows |

**The gate passes do not establish an improvement in predictive performance.** The HMM's configured soft-mode walk-forward pass rate is always one by implementation. A separate comparison on 2,800 post-cutoff bars found mean new-minus-old log density of **-0.061118 per bar**, with improvement for 21 of 50 symbols. The meta-labeler's auxiliary CPCV AUC is **0.493549**, and all five promotion-OOF folds used the existing fallback without time purging. Its comparison with the previous champion was explicitly skipped as incomparable because the HMM lineage changed and the previous metadata lacks the evaluation-contract tag. The new metadata also lacks that tag under the existing save implementation. These are material validation limitations, not evidence of a better champion.

The meta-labeler retains the previous voting architecture (LogisticRegression plus HistGradientBoosting) and **sigmoid OOS calibration**; the serialized configuration comparison found no changes. Calibration diagnostic ECE is 0.010236, distinct from the promotion-OOF ECE above. The 1,749 auxiliary validation predictions came from **583 distinct candidates**, each appearing three times across six valid CPCV folds. They are not 1,749 independent rows. One missing `ou_halflife` used the existing explicit default **24.0**. The FASTWIN target remains reaching +3% within seven hours, rather than terminal PnL at seven hours.

All 366 canonical-workbook rows received verified active-HMM features at their actual bot-start cutoff. General and Meta Features were updated; exact comparisons confirmed preservation of outcomes and unrelated data. Utility fitting excluded 143 bots over seven hours and one missing PnL. Its selected risk coefficient, `lambda_risk=0.05`, sits on the search boundary and fails G7; `current.json` was not created. A rollback copy of the original workbook is retained.

The source audit covered 1,081 export paths across 37 recorded roots, including historical pools. The first broad backtest yielded 6,042 outcomes, but **2,818** had outcome starts earlier than their recorded candidate scan times. Those results were not admitted as a promotion pool. Source timestamps were not guessed or shifted. Conflicting source rows were excluded, then the backtests were rerun. An additional isolation audit detected a relative-path lookup that enabled bootstrap eligibility in an empty working directory. The corrected production context used exact model copies and matched the project-root dry run. It selected **840 candidates without a row cap**, with the standard grid-validity gate active; all 840 backtests succeeded and passed final HMM-lineage checks.

The broader timestamp-valid bootstrap run was used only for shadow research: 4,019 successful outcomes from 4,020 attempts; one invalid symbol was skipped. Of those, 309 lacked the four profile features. Historical reuse quarantined 4,220 rows intersecting the full timestamp-conflict ledger. The final expanded profile has no independent holdout for its newly fitted model. The previous frozen profile scored AUC 0.500082 on 673 new-to-its-training-pool IDs; its own training data includes IDs now subject to timestamp conflicts, so this remains a diagnostic. The expanded target also differs from the canonical completed-bot target. Neither profile was installed as the production current model.

The latest admitted candidate scan is September 10 at 22:12:09 UTC; the latest outcome start is 22:53:22 UTC, with its seven-hour window completed on September 11. Historical labels and market data were fetched/replayed during this task. This is not a claim that every possible market observation through the end of September 11 exists in the training set.

HMM temperature scaling remains the documented identity artifact (T=1, fitted=false). Conformal calibration was not activated: independent calibration outcomes attributable to the new model are unavailable for the production cohort. The code requires at least 20 such calibration samples. MI artifact generation remains disabled by existing code. No active CPCV threshold artifact existed to rotate. The missing deployment-linkage log also prevents a claim that all previously deployed bots have been independently excluded.

Verification includes the complete regression suite, 155 post-refit contract tests, Pyright, exact feature-schema checks, full-row HMM lineage/cutoff audits, and artifact integrity checks. The actual meta-labeler artifact and legacy pickle produce identical probabilities on all 840 rows; single-row and batch inference agree. Both shadow pairs load with finite, positive-definite inverse covariance matrices. The final regression and cleanup receipts below record the terminal results.

Terminal verification: **2,017 passed, 2 skipped** in the final full suite (203.75 seconds, exit 0); **155 passed** in the post-refit contract suite; final Pyright reported **0 errors, 0 warnings, 0 information messages** (exit 0).

Primary evidence and artifacts:

- [Detailed audit findings](audit_notes.md)
- [Runtime artifact validation and hashes](runtime_artifact_validation.json)
- [Meta-labeler metadata](meta_metadata_final.json), [promotion decision](meta_promotion_final.json), [lineage audit](meta_lineage_audit.json), [calibration occurrence audit](calibration_occurrence_audit.json)
- [Production-context parity](production_context_parity.json) and [finalized training pool manifest](finalized_fresh_pool/authoritative_pool_manifest.json)
- [HMM post-training likelihood audit](hmm_post_training_audit.json)
- [Utility gate validation](utility_validation.json) and [workbook preservation audit](utility_lineage_audit.json)
- [Canonical profile report](profile_candidate/report.json) and [model](profile_candidate/profile_model.json)
- [Expanded profile manifest](expanded_profile_shadow/manifest.json), [model](expanded_profile_shadow/profile_model.json), and [pool audit](profile_pool_final_audit.json)
- [Final regression log](full_tests_final.log), [post-fit contracts](postfit_contract_tests.log), [Pyright](pyright_final.log), and [cleanup receipt](cleanup.json)

Training/source datasets, final model artifacts, rollback material and audit evidence are retained. Disposable staging copies, isolated runtime caches, superseded temporary artifacts and test directories are removed after validation; `cleanup.json` records the checked paths.
