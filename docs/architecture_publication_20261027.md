# Architecture publication review - 2026-10-27

The operator authorized review, validation, commits in coherent dependency order and a push of architectural changes, including tests and documentation. Model-state JSON, binaries, workbooks, backups and temporary files are excluded. The starting main commit and freshly fetched origin/main were both `0a38fe08dfd2fcb8eb1fe473bf2094e2eebddefc`.

## Reviewed scope and commit order

These are complete current changes, grouped by dependency. Existing dated entries document earlier sessions; this publication does not reconstruct intermediate implementations or backdate commits.

1. Training contracts and defaults: synchronize -10% label/configuration/EV defaults; preserve reported final net PnL; validate canonical General/Meta identity and source-bound features; retain separate execution identities and recorded event ends; normalize timestamps; add explicit final-net-PnL targets and target-migration rejection; and expose default-off hash-validated canonical simulation admission. The simulation reader, ingestor, builder and retrainer remain in one commit with six regression test files.
2. Native evidence and capital preservation: bounded authenticated capture, account-scope comparison, account-only income evidence, reviewed exact-order ownership, source-bound facts, forecast envelopes, native SAR, actual receipt/per-bot timestamps, immutable terminal outcomes, and the per-bot invested-principal shield. Keep all mutually dependent modules, operating contracts and fourteen regression test files together. The CLIs are durable producer/adapter entry points, not one-off audit scripts.
3. Publication and operating records: accumulated changelog, ERR-121 timestamp blocker, this validation record and the volatility guide's existing supervisor-ownership/historical-symbol workflow.

AST inspection found no imports of new modules belonging to later commits. The selected footprint is 61 distinct files: 32 production/entry-point Python files, 20 integrated regression test files and nine documents. Fifty-nine previously reviewed files matched their prior review hashes. The additional existing volatility-guide diff was checked against current CLI parser options and the tracked P0 owner procedure; the remaining new document is this report.

### Exact commit paths

#### Commit 1: feat(training): govern canonical executions and explicit target contracts

- `retrain_meta_labeler.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/core/config.py`
- `src/neutralgrid/core/constants.py`
- `src/neutralgrid/models/barrier_config.py`
- `src/neutralgrid/models/meta_labeler.py`
- `src/neutralgrid/models/triple_barrier.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- `src/neutralgrid/training/canonical_simulation_labels.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/training/live_outcome_ingestor.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `src/neutralgrid/validation/utility.py`
- `tests/unit/test_canonical_live_features.py`
- `tests/unit/test_canonical_pnl_accounting.py`
- `tests/unit/test_canonical_simulation_labels.py`
- `tests/unit/test_final_net_pnl_target.py`
- `tests/unit/test_meta_labeler_retrain_contract_v20260530.py`
- `tests/unit/test_training_stop_loss_defaults.py`

#### Commit 2: feat(evidence): bind native observations and enforce invested capital floor

- `LIVE_DECISION.md`
- `STATE_ACTION_REWARD_CONTRACT.md`
- `docs/authenticated_telemetry_evidence.md`
- `docs/exact_bot_evidence_ledger.md`
- `docs/terminal_bot_outcomes.md`
- `scripts/capture_account_income_evidence.py`
- `scripts/capture_authenticated_bot_evidence.py`
- `scripts/capture_live_evidence_cohort.py`
- `scripts/capture_native_sar.py`
- `scripts/capture_terminal_bot_outcomes.py`
- `scripts/execute_live_telemetry_modifications.py`
- `scripts/probe_telemetry_account_scope.py`
- `scripts/run_live_telemetry_controller.py`
- `src/neutralgrid/live/decision/account_income.py`
- `src/neutralgrid/live/decision/action_shadow_io.py`
- `src/neutralgrid/live/decision/authenticated_evidence.py`
- `src/neutralgrid/live/decision/capital_policy.py`
- `src/neutralgrid/live/decision/capital_preservation.py`
- `src/neutralgrid/live/decision/exact_bot_ledger.py`
- `src/neutralgrid/live/decision/forecast_contract.py`
- `src/neutralgrid/live/decision/native_sar.py`
- `src/neutralgrid/live/decision/renderer.py`
- `src/neutralgrid/live/decision/state_action_reward.py`
- `src/neutralgrid/live/decision/terminal_outcomes.py`
- `tests/unit/test_account_income.py`
- `tests/unit/test_action_shadow_integration.py`
- `tests/unit/test_authenticated_evidence.py`
- `tests/unit/test_capital_preservation.py`
- `tests/unit/test_capture_authenticated_bot_evidence.py`
- `tests/unit/test_decision_renderer.py`
- `tests/unit/test_execute_live_telemetry_modifications.py`
- `tests/unit/test_forecast_contract.py`
- `tests/unit/test_live_evidence_cohort.py`
- `tests/unit/test_live_telemetry_controller.py`
- `tests/unit/test_native_sar.py`
- `tests/unit/test_probe_telemetry_account_scope.py`
- `tests/unit/test_state_action_reward.py`
- `tests/unit/test_terminal_outcomes.py`

#### Commit 3: docs: record architecture validation and collector ownership workflow

- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `docs/architecture_publication_20261027.md`
- `docs/live_volatility_forecast_v1.md`

## Validation

Fresh full existing Python suite: **2,871 passed, two skipped, zero failures**
in 315.06 seconds. The Windows path-length regression also passed in this full
run using the shorter temporary path.

- Configured source Pyright, using the repository interpreter: zero errors, warnings and informational diagnostics.
- Separate Pyright over the nine changed/new operational Python entry points: zero errors, warnings and informational diagnostics.
- The prior schema/leakage review checked all 20 active features across candidate output, FeatureSnapshot and unified builder, equal FeatureSnapshot/to_dict keys, both shared label-guard sites, and no label in active feature profiles or inference aliases. Those source bytes are unchanged; their existing regression/contract coverage is included in the fresh suite.
- Scoped staged path and whitespace checks pass for each commit. Every path is explicitly enumerated; broad staging and ignored-file force-adds are not used.
- The guide's target-csv/audit-root, explicit symbol, contract, backfill-manifest, run-id and history-extension options match the current parsers. Historical acquisition remains separate from fresh roster claims and current Live advice; stopped collection is not restarted by this publication.
- A concurrent guide clarification was rereviewed before staging: run_live_volatility_loop.py can persist explicit unavailable records despite a completed cycle, so activation requires reviewing fresh eligible available records for the intended pairs. No source or test byte changed with that clarification.

Commands:

```text
.venv/Scripts/python.exe -B -m pytest tests/ -q -p no:cacheprovider --tb=short --basetemp=.tmp_pytest/p27
.venv/Scripts/pyright.exe --pythonpath .venv/Scripts/python.exe
```

The operational Pyright check explicitly supplied retrain_meta_labeler.py and the eight changed/new scripts. Logs remain local and ignored:

- `outputs/architecture_publish_20261027_pytest.log`
- `outputs/architecture_publish_20261027_pyright.log`
- `outputs/architecture_publish_20261027_cli_pyright.log`

The earlier review's long temporary path produced one fixture padding assertion. This publication uses a shorter temporary path; it does not change or weaken that test.

## Preserved boundaries and limitations

- Active runtime feature lists and both leakage guards remain intact. Final-net-PnL training is an explicit different target; it does not reinterpret frozen FASTWIN artifacts. A target mismatch blocks promotion.
- Canonical simulation labels remain explicitly modeled. Their workbook evidence JSON and five hash-bound source files are durable local training dependencies, not disposable scratch. They are not included in these source commits.
- The per-bot capital floor is a pre-state admission condition, not a filter removing later negative rewards, a guarantee of future equity, or execution permission. Account-wide observations do not establish exact-bot ownership/equity.
- The historical promotion evaluator retains its logged unpurged fallback and partial-coverage rule. The historical v2 identifier alone does not establish strict temporal independence. No model was fitted, evaluated for new performance, promoted or enabled by this publication.
- The pending defaults are -10%, while the recorded October 26 fit explicitly used -12% to reproduce its frozen configuration. Neither change enables a live execution stop.
- ERR-121 concerns four unresolved ARIA/CROSS/GIGGLE/GUA lifecycle intervals. The separate October 26 workbook report records four NEAR/FOLKS/ONDO/ROBO replay cutoff failures. No timestamp or candidate identity was repaired by this publication.
- The utility runtime pointer was absent in the review environment; existing utility/profile promotion rejections and conformal nonactivation remain separate. Passing source validation does not establish whole-pipeline deployment readiness.
- Existing changelog training/live-run/workbook claims are retained as historical session records. They were not rerun as part of publication.

## Explicit publication exclusions

- `artifact_manifest.json`
- `models/meta_labeler_promotion_decision.json`
- `models/meta_labeler_verification.json`
- All model binaries and model artifact directories, workbooks, backups, raw market/private data, run logs, generated metadata and retained audit evidence.
- The tracked temporary helper `tmp_scan_ports.py`.
- `.pt_c84e/`, `.rt21/`, `.rt21r2/`, `.rt21r2c/`, `t/`, `tests/.funding_tmp21a/`, ignored temporary directories and concurrently appearing isolated experiment workspaces.
- Historical `docs/operational_prompt_10minute_snapshots_and_lifecycle_20261019.txt`, `docs/validation_10minute_snapshot_prompt_20261019.txt` and `docs/validation_operational_prompt_standalone_20261019.txt`. The tracked standalone guide is their execution replacement.

Checksums of the three excluded JSON files, tracked temporary helper, historical documents, current HMM/meta artifacts, canonical workbook and original repair backup are recorded locally for before/after comparison. The working tree is expected to remain dirty because these exclusions are preserved.

## Publication verification

The GitHub connection publishes only the reviewed text blobs and constructs each tree from the previous reviewed base. Every remote tree must equal its local reviewed tree before main is advanced using the expected original head and without force. Completion requires fetching the published head, verifying exact changed paths and trees, and aligning the local branch only when its tree is identical. No collector, scanner, schedule, exchange action or training command is started.
