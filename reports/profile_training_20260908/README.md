# Profile training study — 2026-09-08

A four-feature profile model and its matching pattern artifact were trained and saved as **shadow-only**. No tested recipe met the existing promotion requirements. Production source code, the active meta-labeler, the HMM manifest, the canonical workbook and the production profile bootstrap were not changed.

The selected expanded recipe is `gaussian_lda_v1`, covariance shrinkage **0.9**. This is the best of the recipes specified in `preregistration.json` under its development selection rule; it is not a claim of the best possible model.

## Deliverables

- `full_refit_shadow/profile_model.json`: final shadow model fitted on **6,832 rows**, **3,567 winners / 3,265 losers**.
- `full_refit_shadow/pattern_profile.json`: matching four-feature pattern profile.
- `full_refit_shadow/manifest.json`: role, training counts, dataset/model/pattern hashes and validation scope.
- `frozen_shadow/profile_model.json`: the development-only model whose holdout results are reported below.
- `workbook_baseline/profile_model_shrinkage_0.9.json`: best mean-AUC model among the five tested canonical workbook recipes, also diagnostic/shadow-only. Its paired pattern is in the same directory.

The full refit includes the disclosed holdout and the 137 boundary-purge rows. Therefore, the holdout statistics validate the **frozen 5,084-row development fit**, not an independently tested 6,832-row refit. Neither artifact is activated through `data/profile/current.json`.

## Data and coverage

The study reused the previous source audit and reread the original scanner CSV/XLSX rows, checking their hashes and recorded feature and grid values. The final September 8 refresh searched the 37 listed current, mirrored and backup roots in `inventory_refresh.json`: 1,075 scanner files versus 1,074 previously known files. No known source changed or disappeared, and no directory scan errors were recorded. Eighty-two original files supplied the selected feature cohort and were rehashed during final validation.

| Stage | Rows |
|---|---:|
| Previously inventoried candidate rows | 32,556 |
| All four profile fields finite | 17,263 |
| Also complete recorded grid geometry | 6,846 |
| Rejected identity/timing/geometry records | 13 |
| Frozen eligible candidate cohort | 6,833 |
| Existing compatible fresh legacy backtest outcomes reused | 5,060 |
| Additional backtests requested | 1,773 |
| Additional backtests successful | 1,772 |
| Final expanded model pool | **6,832** |
| Development / reserved holdout / boundary purge | **5,084 / 1,611 / 137** |

The pool spans recorded outcome starts from **2026-07-20 23:01:36 UTC through 2026-09-07 15:20:36 UTC**, 69 distinct start groups and 480 symbols. Every expanded row has four finite, recorded features; no missing profile feature was fabricated or imputed to admit a row. Earlier rows without the complete entry-time feature and geometry record could not be treated as equivalent observations.

The single failed candidate, `AERGOUSDT_20260720_123348_dd64e7ad`, failed all three REST attempts with Binance `-1121 Invalid symbol`. It is excluded, with the exact error retained in `additional_backtest_manifest.json`.

The new September 8 export contains **250 rows**, all still immature when checked at approximately 18:06 UTC. Their seven-hour windows become observable at **22:13:05 UTC / 17:13:05 Lima**. They were not assigned outcomes or included in training. This result uses the eligible data available at the stated audit time; it does not claim to include future outcomes from today's scan.

New outcomes used the canonical `run_single_backtest -> build_training_config -> run_backtest` path, `legacy` realism, recorded geometric grid parameters, $400 capital, 10x leverage, seven hours and at least 420 one-minute bars. Four concurrent requests shared the Binance client. Raw results retain engine, formula and label-contract versions, authority flags and source hashes. The assembled dataset checks those contracts and unique candidate identities before admission.

## The canonical workbook is a separate target

All **366** rows in `data/new_expired_bots.xlsx` were considered by the canonical training code. Its bounded universe contains **223** rows with `0 <= duration_hours < 7`; **195** also have the required PnL and profit-factor labels. At the scanner CLI's **0.68** PnL quantile and profit-factor floor **1.5**, the full training subset has **62 winners / 133 losers**, with a PnL threshold of **3.4852**. Walk-forward thresholds are computed separately from each training fold.

The candidate backtester does not supply the per-trade profit factor required by this completed-bot label. Consequently, the additional FASTWIN rows were not appended to the workbook or passed off as completed-bot observations. The expanded shadow target is explicitly **time to +3% <= 7 hours**, not the canonical completed-bot profile label.

The original workbook has substantial feature missingness; canonical core training applies its existing within-training preprocessing. Further, entry-time availability of its stored funding feature was not established. The canonical backfill at `retrain_scanner.py:414` calls `realized_funding_carry_proxy_next_hours`; `src/neutralgrid/data/funding_rate.py:68` selects a funding settlement after entry. That routine cannot prove a historical *expected* funding input was known at entry. No such backfill was performed, and no claim is made that every existing workbook funding value came from that routine. The expanded study uses the recorded scanner fields instead.

## Tests and results

The five canonical Gaussian shrinkages were 0, 0.1, 0.3, 0.6 and 0.9. The expanded study compared the same five shrinkages with logistic C values 0.01, 0.1, 1 and 10 using five outer chronological folds, four inner folds, timestamp grouping and a **24-hour purge**. Model family and final parameter were chosen using development data before holdout disclosure. Three-feature funding-exclusion ablations were development-only diagnostics. No recipe was tuned after seeing holdout results.

| Model / target | Mean walk-forward AUC | Pooled OOF AUC | Passing finite folds |
|---|---:|---:|---:|
| Canonical workbook Gaussian, shrinkage 0 | 0.4278 | 0.4055 | 1/5 |
| Canonical workbook Gaussian, shrinkage 0.1 | 0.4278 | 0.4063 | 1/5 |
| Canonical workbook Gaussian, shrinkage 0.3 | 0.4424 | 0.4107 | 1/5 |
| Canonical workbook Gaussian, shrinkage 0.6 | 0.4403 | 0.4129 | 1/5 |
| Canonical workbook Gaussian, shrinkage 0.9 | **0.4515** | **0.4256** | **1/5** |
| Expanded Gaussian, nested parameter selection | **0.5284** | **0.5374** | **2/5** |
| Expanded logistic, nested parameter selection | 0.5193 | 0.5339 | 1/5 |

The Gaussian's final development-selected shrinkage was 0.9; logistic's was C=0.01. The Gaussian's pooled development ECE was 0.06730 and Brier score 0.25069. Removing funding did not establish promotion-grade discrimination and retains only 3/4 requested features, below the 90% feature-coverage rule.

The single frozen holdout contains **1,611 rows, 855 positives and 14 start groups**:

| Frozen model | AUC | Brier score | ECE |
|---|---:|---:|---:|
| Selected Gaussian, shrinkage 0.9 | **0.55149** | **0.24783** | **0.03618** |
| Fixed Gaussian baseline, shrinkage 0.3 | 0.54563 | 0.24796 | 0.02315 |
| Constant development prevalence | 0.50000 | 0.24914 | 0.00909 |

Five thousand timestamp-cluster bootstrap replicates produced selected-model AUC 95% interval **[0.48660, 0.61671]**, and paired AUC-improvement interval **[-0.00084, 0.01434]** versus the fixed Gaussian baseline. The evidence does not establish positive improvement. That baseline is not a verified production incumbent.

Some outcomes had already been examined during the previous meta-labeler study. This holdout was reserved for the profile recipe comparison, but is **not wholly unseen prospective evidence**. Timestamp grouping and purging do not prove independence between related symbols, overlapping bots or market regimes. The report does not infer economic profitability from classification metrics.

## Why promotion was rejected

The existing constants in `src/neutralgrid/scanner/profile_model_walkforward.py:54` require mean AUC >=0.55, pooled OOF AUC >=0.55, passing finite-fold fraction >=0.50, at least three finite folds, finite-fold coverage >=0.60 and feature coverage >=0.90. The canonical and expanded candidates fail the discrimination/pass-rate requirements. A calibration error below 0.10 does not override these profile gates.

Promotion also requires valid, fresh provenance, paired model/pattern artifacts and, where an incumbent exists, an aligned paired comparison with positive 95% lower confidence bound. The current directory has a bootstrap pattern and manifest but no profile-model/current pointer pair; the scanner CLI rejects that incomplete state. The isolated study did not forge an incumbent comparison or bypass that check. The changed FASTWIN target, previously exposed outcomes and unverified workbook funding availability are additional reasons to retain shadow status. Existing ERR-095 and ERR-096 describe the associated discrimination and freshness constraints.

A future promotion attempt needs genuinely new, mature authoritative observations with the four entry-time fields, an explicit compatible target contract, fresh purged evaluation evidence, the unchanged statistical gates, and a governed valid initial/paired artifact state. Generating more trials on this disclosed holdout would not supply that evidence.

## Validation, corrections and cleanup

- **80 focused tests passed** in separate test processes/output paths, including scanner CLI, provenance, promotion, profile fitting, logistic runtime and FASTWIN study contracts (`preflight_tests.log`). The existing project virtual environment was used; this was filesystem/process isolation, not a separate container or dependency installation.
- Configured project **Pyright: 0 errors, 0 warnings** (`pyright.log`). No production Python source was changed; this configured check covers the project's source scope, not every report-local script.
- `artifact_validation.json` records source and protected-artifact hash checks, disjoint IDs, 24-hour purge checks, finite features, saved runtime scoring equivalence, recomputed OOF/holdout AUC and full-refit class-count validation.
- An initial diagnostic used the core library's default quantile **0.75**, whereas the CLI uses 0.68. Those five trials and the funding ablation are preserved in `workbook_library_defaults_diagnostic`. Their best mean AUC was 0.67063 over three finite folds with 2/3 passing; they do not implement the canonical 0.68 target. The corrected 0.68 study above was rerun explicitly. No favorable default was substituted for the CLI contract.
- During implementation, an initial frozen-dataclass assignment was corrected to `dataclasses.replace`. Final validation also corrected copied development class-count/source-hash metadata on the full refit; its coefficients and predictions were unchanged. The frozen validation artifacts and selection hashes were unchanged.
- The final inventory assertion correctly detected today's new export; it was audited and excluded for immaturity before validation passed.
- Owned temporary paths `.pft8` and `reports/profile_training_20260908/_tmp` are removed after testing. See `cleanup.json` for verified completion. Trained artifacts, source/outcome evidence, logs and reproducible study scripts are retained as deliverables, not temporary files.

Reproduction entrypoints are `freeze_inputs.py`, `backtest_additional.py`, `workbook_baseline.py`, and `study.py assemble/development/holdout`, followed by `validate_study.py`. They intentionally refuse to overwrite frozen split or holdout outputs. Reproduction belongs in a new isolated destination; do not delete this evidence to rerun a disclosed holdout.
