# August 22 champion: Git calibration audit

Reviewed September 7, 2026. Repository: `Randell1108/neutral-grid-bot-v6.5.8-clean`. This audit reads historical records and current code; it does not retrain or promote a model.

**The archived canonical training pipeline uses sigmoid calibration, and that code is still present unchanged locally. The final calibrator retained by the exact champion `20260822_230654` remains unverified.** Its surviving summary omits calibration method, and the code permits sigmoid to be replaced by beta. The current August 27 recovery shadow explicitly records `sigmoid_oos`.

## Repository search and provenance

The default branch, `fastwin-v2-regrow-err059-v658`, ends at August 18 commit `60ab8998887f36d60c8a41443abe016a49507eb0`. Its code search therefore misses the later champion. I inspected all five branch tips and their complete, non-truncated recursive trees. The relevant branch is `codex/latest-worktree-20260820`, ending at August 24 commit `55be3a5e1af8bc274149fdfe2d683533df921b60`.

The champion reference appears in [the promotion decision at that immutable commit](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/models/meta_labeler_promotion_decision.json#L19). That file entered its present form with August 24 commit `b0f7d3ed091be11d0009932a0374d9ea0eec1113`; the decision itself was generated August 23 at 03:00:10 UTC. The previous committed decision and verification describe August 20 training, not the August 22 champion's own training run.

The [repository ignore rules](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/.gitignore#L1) exclude `models/meta_labeler/`, model binaries and logs. The dedicated metadata path has no commits on either the default branch or the August 24 branch. None of the five inspected branch tips contains it. An exact-version commit-message search also returned no matches. This search cannot exclude unreachable commits or external backups.

Three tracked experimental pickles under `temp_runs/` were inspected as bytes in memory, without loading or executing their serialized objects. They contain 4,957/5,143/5,143 training samples, have different OOF ECE values, and contain no exact champion version string. The baseline contains `sigmoid_oos`; the other two contain both beta and sigmoid method strings. They are not evidence of the requested 1,386-row champion's final method. No downloaded pickle files were created.

## What the exact champion record establishes

| Recorded property | Value |
|---|---|
| Artifact version | `20260822_230654` |
| Feature count | 20 |
| Total training samples | 1,386 |
| Target contract | `fast_winner_time_to_3pct_le_7h` |
| HMM lineage | `rolling_180d_20260822_203741` |
| OOF AUC | 0.6173465131798465 |
| OOF AUC confidence interval | [0.5881841344920049, 0.6456244703278164] |
| OOF ECE | 0.02288292063763023 |
| Stored promotion status | pass |
| Promotion evaluation contract | null |
| Final calibration method / exact fit configuration | Not included in surviving summary |

The separate 5,849-row candidate in that decision has ECE 0.08151344484947146. Its training-source manifest must not be assigned to the 1,386-row incumbent. The decision explicitly says the comparison is nonpaired and the evaluation contracts are incomparable. The old champion's ECE passed its recorded 0.10 gate; this is not September holdout evidence.

## Training method supported by the archived code

These are verified properties of the archived canonical implementation, not proof that the missing champion binary was produced with every default unchanged:

1. [The FASTWIN profile defaults to `vote_logit_hgb`](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/retrain_meta_labeler.py#L1853): an equal-weight soft vote of L2 logistic regression and regularized histogram gradient boosting. Explicit estimator overrides are supported. The CLI trains against the explicit FASTWIN target column.
2. [Estimator construction and preprocessing](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L1384): logistic uses C=1, balanced classes and liblinear; HGB defaults to learning rate 0.03, 10 leaves, 350 iterations, minimum leaf size 60, L2=3 and max_features=0.8. Mean imputation and standardization are fit within the training folds. The training path supports concurrency, time-decay and class weights.
3. [Calibration defaults](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L538): enabled, method `sigmoid`, temporal calibration fraction 20%. CPCV produces pooled raw out-of-sample probabilities.
4. [Final pooled OOS sigmoid calibration](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L1854): a one-input logistic regression maps clipped raw predicted probability to observed frequency. The input is raw probability, not its logit. The candidate must pass the ECE/Brier acceptance check to be retained as `sigmoid_oos`.
5. [Conditional beta replacement](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L1948): if the accepted calibration still has sufficient residual error, beta is fitted and can replace it only after convergence and strictly lower measured ECE. Therefore `calibration_method="sigmoid"` in configuration does not guarantee the final artifact says `sigmoid_oos`.
6. [Inference wrapper](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L2047): an accepted OOS calibrator wraps the full-data raw base model. It does not apply OOS calibration on top of the temporal calibrator.

The code's separate promotion OOF evaluator explicitly uses `CalibratedClassifierCV(method="sigmoid", cv=3)` at lines 1031-1033. That makes sigmoid part of the archived promotion-evaluation recipe; it does not identify the final saved OOS mapping. Its shuffled/grouped folds and purge fallback also differ from the later study's chronological evaluation. The exact champion has no stored evaluation-contract identifier, so an identical historical fold execution cannot be certified.

The [metadata writer](https://github.com/Randell1108/neutral-grid-bot-v6.5.8-clean/blob/55be3a5e1af8bc274149fdfe2d683533df921b60/src/neutralgrid/models/meta_labeler.py#L2418) does record the final calibration method, raw/calibrated ECE and Brier, and OOS acceptance. The pickle state also stores `oos_calibration_method` and `oos_calibration_report`. The promotion summary copies only a subset of those fields, omitting the method. Recovering either exact artifact would resolve the remaining question.

## Is this still used now?

| Current object | Verified calibration status |
|---|---|
| Canonical training implementation | Same sigmoid-first and conditional-beta implementation as the archived branch |
| August 27 recovery shadow `20260827_204845`, 20 features | Metadata explicitly says `is_calibrated=true`, `calibration_method=sigmoid_oos`, `calibration_oos_accepted=true` |
| September study, three holdout models | Used temporal sigmoid calibration through `CalibratedClassifierCV` |
| September 24-feature full-data research artifact | Uses logistic calibration on the logit of raw probabilities; same sigmoid family, different input transformation and evaluation design |
| Canonical production artifact in this workspace | `models/meta_labeler.pkl` and `models/meta_labeler/metadata.json` are absent; no claim of an active production champion's method is possible |

The recovery shadow's calibration-fit ECE is 0.003881634767576937 and its separate promotion OOF ECE is 0.03532796255188195. These are different evaluations; neither establishes September performance. The 24-feature research artifact remains diagnostic and is not a promoted replacement.

## Validation and cleanup

Local `git hash-object` values equal the remote Git blob SHAs for all three checked files:

| File | Matching Git blob SHA |
|---|---|
| `src/neutralgrid/models/meta_labeler.py` | `0887095dd79e8fdc67970a4284c073be7060cff4` |
| `retrain_meta_labeler.py` | `8bbfb156825bd62cabe65222a1cfc64f11e662d9` |
| `models/meta_labeler_promotion_decision.json` | `0ae595543b6db9576641a47e1c9395c4aebe441d` |

Ran the existing calibration regression suite with bytecode and pytest cache disabled: `python -m pytest tests/unit/test_meta_labeler_calibration_gate.py -q -p no:cacheprovider`: **4 passed in 25.26 seconds**. These validate current calibration-gate behavior on synthetic regression fixtures; they do not reproduce the missing historical champion.

No production code, model or GitHub content was changed. No temporary downloads or clone directories were created. Only this report and its [machine-readable evidence](champion_git_calibration_evidence_20260907.json) were added.
