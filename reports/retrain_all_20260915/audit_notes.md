Retraining audit - September 15, 2026

Final disposition: see REPORT.md and final_review.json. The earlier intake notes
below describe intermediate states. Subsequently, all 57 additional workbook
rows received all 20 finite selected features, before meta fitting. Utility was
rerun with complete optional features; the final candidate is
utility_20260915_220851_633469, holdout AUC 0.652174, G7 rejected. All 1,211 fresh
backtests completed after a full rerun; the 23 initially incomplete windows were
recovered without changes to the other 1,188 labels or PnL. The promoted meta
artifact is 20260915_220809 with beta_oos calibration, AUC 0.663243 and ECE
0.028080. Its auxiliary calibration observations comprise 841 distinct candidates
repeated three times. Full regression: 2,027 passed, two skipped. Report-local
audit failures involving PnL column naming, mixed timestamp parsing and Unicode
decoding were corrected and verified; original failure logs are retained.

This run uses the canonical training and promotion implementations without changing production source code, feature lists, or gate thresholds. Pre-existing workspace changes are recorded in `git_status_before.txt` and `baseline_hashes.json`. The prior workbook and meta-labeler are retained under `backups/`.

Data intake:

- The canonical General sheet contains 423 bots; Meta Features initially contains 366. The additional 57 bots have recorded grid counts, bounds, modes and all four profile features. Their `num_grids` is taken from recorded `grids_count`; `profit_per_grid_pct` is calculated by the canonical `TrainingDataBackfiller` geometry method. Optional missing features remain missing. The original equal-sheet-ID assumption failed explicitly and was corrected in report-local preparation code before any fitting or workbook publication.
- The source inventory contains 1,170 export paths across the previously recorded search roots. There are 276 unique staged exports, 892 byte-identical duplicates, and two undated exports without usable candidate identities. Timestamp checks retained 120 files and excluded 18,994 source rows. These source-row counts include duplicate candidates; they are not distinct training-row counts.
- Production selection and isolated-runtime selection agree exactly on the normalized selection log and 1,211 candidates. The isolated runtime contains exact copies of the current meta-labeler to preserve the existing bootstrap-selection behavior. Fresh outcomes use the canonical legacy FASTWIN full-pool contract, not the historical replay path.

Canonical profile:

- The unchanged four-feature recipe trained on 246 labeled bots (79 winners, 167 losers) from 274 duration-eligible bots.
- Walk-forward evaluation begins after the previously disclosed August 14 endpoint, with seven-hour purging. Five folds contain 51 test rows; one fold has only one class, leaving four finite fold AUCs. Mean finite-fold AUC is 0.633681; pooled OOF AUC is 0.503861, Brier 0.220201 and ECE 0.130153.
- The existing gate rejects promotion because pooled OOF AUC is below 0.55. The candidate remains a shadow pair. Historical funding-feature availability is not independently verified, and no realized-future-funding proxy was substituted during this run.

Known validation limits retained for final review:

- HMM soft-mode walk-forward checks return unconditional passes. Report this separately from the supplemental old/new post-training likelihood diagnostic; it is not predictive accuracy.
- Meta-labeler calibration and promotion diagnostics use different prediction sets. Record distinct candidates versus repeated fold occurrences and disclose any fallback without temporal purging.
- HMM replay at historical cutoffs uses the newly trained frozen model. It does not prove that model existed at those historical times.
- Expanded-profile FASTWIN labels differ from the canonical completed-bot profile target. Candidate-ID temporal separation does not prove bot-disjoint, event-complete promotion evidence.
- The deployment-linkage log is absent; no universal deployed-bot exclusion claim is made.

Interim verification: dependency checks passed; 115 targeted contract, leakage, backfill and lineage tests passed. All 20 selected features are present in the relevant producer/consumer schemas. Candidate output containers legitimately include outcome columns; leakage checks apply to selected model inputs, with negative cases covered by the tests.

The initial Pyright invocation did not resolve the project virtual environment and reported 222 errors and three warnings, primarily missing imports. Re-running with `--pythonpath D:/Neutral Grid v5.7.0/.venv/Scripts/python.exe` passed with zero errors and warnings. No source changes were needed; both logs are retained.
