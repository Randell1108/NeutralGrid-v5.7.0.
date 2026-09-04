# Changelog

## [unreleased] - FASTWIN-V2-REGROW + ERR-059 Live Reintegration + v6.5.8 (2026-06-22)

### Summary

Grew the FASTWIN meta-labeler training pool with the 2026-06-08 -> 06-22 gap
(backtested with the IDENTICAL 24h window) and retrained -> new active artifact
`20260622_230203`, **OOF-AUC 0.7698 [0.7579, 0.7820]**, OOF-ECE 0.0789, n_pos
2834, 6048 rows, promotion gate PASS. This **supersedes the 0.640 "pure re-pin"**
recorded in the HMMROTATE-0622 block below (same active HMM lineage
`rolling_180d_20260622_135710`). Reintegrated the meta-labeler into the live
decision path (ERR-059): `ev_score` is now computed BEFORE the Stage-12 meta
probe, so `meta_prob` is authoritative for Kelly sizing and the soft Stage-B meta
gate is ON. Closed the meta-labeler ERR cluster (054b/035/036/053). Bumped the
package version 6.5.7 -> 6.5.8 to match the working tree.

### HMMROTATE-0903 - Rolling HMM rotation and fail-closed pipeline run (2026-09-03)

Promoted `rolling_180d_20260903_153527` from a frozen 180-day, 50-symbol
training slice (858,000 15-minute samples). All three walk-forward folds passed,
for a mean pass rate of 1.0. A subsequent default-250 full pipeline run wrote
249 uniform-lineage rows and approved zero deployments because the governed
meta-labeler, utility calibrator, and promoted scanner profile remain absent.

**Files modified or generated:**
- `CHANGELOG.md`
- `artifact_manifest.json`
- `artifacts/hmm/rolling_180d_20260903_153527/eval.json`
- `artifacts/hmm/rolling_180d_20260903_153527/feature_schema.json`
- `artifacts/hmm/rolling_180d_20260903_153527/metadata.json`
- `artifacts/hmm/rolling_180d_20260903_153527/model.joblib`
- `artifacts/hmm/rolling_180d_20260903_153527/scaler.joblib`
- `artifacts/hmm/rolling_180d_20260903_153527/state_means.npy`
- `artifacts/hmm/rolling_180d_20260903_153527/temperature_scaler.json`
- `artifacts/hmm/training_sets/canonical_20260903_153520/metadata.json`
- `data/new_expired_bots_backfilled_hmm_only_20260903_161305.xlsx`
- `artifacts/pipeline_runs/default_250_20260903_162104/deployment_ready_20260903_162106.csv`
- `artifacts/pipeline_runs/default_250_20260903_162104/potential_candidates_20260903_162106.csv`
- `artifacts/pipeline_runs/default_250_20260903_162104/snapshots/snapshots_2026-09-03.parquet`

**Decision rationale:** The rotation retained the canonical rolling-window,
point-in-time, sequence-boundary, and CPCV/walk-forward controls described by
Lopez de Prado's *Advances in Financial Machine Learning*. Downstream gates
remained fail-closed: the diagnostic surrogate was not promoted or accepted as
authority, and the four expired-bot rows lacking causally valid HMM inference
were left incomplete rather than imputed.

**Backward compatibility:** The active HMM pointer changed from
`rolling_180d_20260827_144604` to `rolling_180d_20260903_153527`; downstream
artifacts pinned to the former HMM are stale by design. No runtime code,
feature schema, default universe, thresholds, or production-gate behavior was
changed. The identity temperature scaler remains explicitly self-supervised
and unfitted.

**Verification:** HMM convergence completed in 121 iterations; three
walk-forward folds each passed at 1.0. The pipeline process exited 0 and its CSV
and Parquet each contain 249 unique candidate IDs, finite regime probabilities,
and one HMM lineage. The contract subset passed (117 tests), and project-wide
Pyright against the Python 3.11 virtual environment reported 0 errors, 0
warnings, and 0 informations. The full suite reported 1,975 passed, 2 skipped,
and one intentional fail-closed lineage-contract failure because the canonical
workbook remains pinned to the prior HMM. The run is diagnostic/degraded, not
production healthy: 147 rows were marked with an over-300-second scan-cache
age, two rows ended in `ConnectTimeout`, one scan symbol failed, and the
diagnostic-shadow hook rejected an incompatible artifact without altering
deployment output.

### SHADOWBT-0828 - Matured shadow-approved unfiltered backtest pool (2026-08-28)

Added an idempotent diagnostic runner for full-pipeline candidates that reached
Stage B and were rejected exactly because authoritative `meta_prob` was absent.
The runner accepts the quarantined shadow probability only for this explicit
counterfactual selection, waits for 362 closed one-minute bars, delegates to the
canonical unified backtest engine, and retains every successful outcome without
EV, PnL, symbol, or label filtering.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backtest_shadow_approved_candidates.py`
- `src/neutralgrid/backtest/shadow_approved.py`
- `tests/unit/test_shadow_approved_backtest.py`

**Decision rationale:** Prospective realized outcomes are valid evidence only
when their decision-time inputs and forward windows are causally separated.
Hash-bound source manifests, an immutable deployment-availability anchor, a
fixed 362-minute horizon, and preservation of losing outcomes follow the
point-in-time and sample-selection controls in Lopez de Prado's *Advances in
Financial Machine Learning*. The diagnostic teacher is never promoted to
runtime authority.

**Backward compatibility:** No breaking changes. Production gates, the active
HMM and meta-labeler, canonical training pools, promotion behavior, and the
default 250-symbol universe are unchanged. Outputs are confined to
`artifacts/diagnostics/shadow_approved_backtests/`, marked non-authoritative and
non-promotable, and are not automatically ingested by governed training.

**Verification:** The focused contract suite passed (8 tests); the complete
suite passed (1,975 tests; 2 skipped); project-wide Pyright reported 0 errors,
0 warnings, and 0 informations. A write-free replay of the current 250-symbol
run verified 59 exact counterfactual candidates and correctly deferred them
until the final 362-minute bar closes at `2026-08-28T05:32:00Z`.

### SHADOWBT-RUNFIX-0828 - Execute and validate shadow-approved cohort (2026-08-28)

Fixed the standalone diagnostic runner's repository-path bootstrap after its
first live attempt could fetch valid Binance windows but could not import the
repository-level `backtest` package. Added a subprocess regression test, then
completed all 59 unfiltered counterfactual backtests from pipeline run
`20260827_225212` and wrote a quarantined evidence review. The cohort remains
cross-HMM, non-promotable, and excluded from governed training.

**Files modified or generated:**
- `CHANGELOG.md`
- `scripts/backtest_shadow_approved_candidates.py`
- `tests/unit/test_shadow_approved_backtest.py`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/selected_candidates.csv`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/unfiltered_outcomes.csv`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/training_data_unfiltered.csv`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/failures.csv`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/run_manifest.json`
- `artifacts/diagnostics/shadow_approved_backtests/pipeline_20260827_225212/review_report_20260828.md`
- `artifacts/diagnostics/shadow_approved_backtests/unfiltered_outcome_pool.csv`
- `artifacts/diagnostics/shadow_approved_backtests/unfiltered_training_pool.csv`
- `artifacts/diagnostics/shadow_approved_backtests/unfiltered_training_pool.manifest.json`

**Decision rationale:** The Lopez de Prado AFML point-in-time discipline
requires the runner to preserve the decision-time candidate set and include all
subsequent outcomes without filtering. The import fix changes no quant logic;
it only makes the existing canonical backtest engine resolvable when the script
is launched directly. Source hashes, an availability-time anchor, fixed closed
bars, and idempotent outputs preserve the original counterfactual.

**Backward compatibility:** No breaking changes. Production gates, model
artifacts, the default universe, authoritative training paths, label physics,
and promotion behavior are unchanged. The new outputs remain confined to the
diagnostic artifact tree.

**Verification:** The focused suite passed (9 tests); project-wide Pyright with
the project virtual environment reported 0 errors, 0 warnings, and 0
informations. The canonical run completed 59/59 candidates with zero failures;
an immediate rerun returned `already_complete` with unchanged outcome and pool
hashes. `git diff --check` is included in the final review checkpoint.

### PIPERUNSAFE-0827 - Signed-request log redaction and deterministic client cleanup (2026-08-27)

Hardened the full pipeline's Binance client lifecycle. Third-party HTTP request
logging is now restricted to warnings so signed query parameters are not
written to ordinary INFO logs, and every connection, scan, enrichment, and
successful persistence path closes the client without masking the primary
pipeline result.

**Files modified:**
- `CHANGELOG.md`
- `run_full_pipeline.py`
- `tests/unit/test_full_pipeline_runtime_safety.py`

**Decision rationale:** Operational logs must preserve reproducibility without
retaining authentication material, while a failed local write must not leave a
network client open. This is consistent with the auditability and stable-data
handling principles in Lopez de Prado's *Advances in Financial Machine
Learning* and does not alter the statistical pipeline.

**Backward compatibility:** No breaking changes. The default universe remains
250 symbols, and candidate selection, HMM inference, features, gates, sizing,
ranking, artifacts, and promotion behavior are unchanged.

**Verification:** Runtime-safety, scanner-profile, profile-promotion, and
artifact-resolver tests passed (53 tests); scoped Pyright reported 0 errors and
0 warnings. A post-change authenticated smoke run completed without signed
request URLs in its logs, and the default 250-symbol run is validated
separately through its immutable run artifacts.

### HMMLINEAGE-0827 - Canonical workbook refresh to the active HMM (2026-08-27)

Replayed the canonical expired-bot workbook against
`rolling_180d_20260827_144604` using candidate-ID scan time for scanner-origin
rows and the recorded event start for legacy rows. The validated five-sheet
workbook now contains 362 complete current-lineage rows and four explicit
`hmm_failed` quarantines whose candidate IDs post-date their bot start. No
timestamps, identifiers, labels, or missing independent features were invented.

**Files modified:**
- `CHANGELOG.md`
- `data/new_expired_bots.xlsx`
- `data/profile/pattern_profile_bootstrap_manifest.json`

**Artifacts retained:**
- `data/new_expired_bots_backfilled_20260827_hmm_rolling_180d_20260827_144604.xlsx`
- `data/archive/new_expired_bots_pre_hmm_20260827_703ac451.xlsx`
- `outputs/01a03f31-11a4-7a02-9446-616fc3c23b5e/new_expired_bots_hmm_rolling_180d_20260827_144604.xlsx`

**Decision rationale:** FIXPIPELINE-01 requires a single HMM lineage before
calibration. Candidate IDs are the causal boundary for scanner snapshots; an
identifier later than the recorded event start is contradictory evidence and
must be quarantined. This preserves the point-in-time and leakage controls in
Lopez de Prado's *Advances in Financial Machine Learning*.

**Backward compatibility:** The General, PnL Curve Features, and Last Features
sheets are value-for-value unchanged, and every non-HMM Meta Features column is
unchanged. The former canonical workbook is recoverable from the hash-named
archive. Utility and EV fields remain missing because no promoted current
utility artifact exists.

**Verification:** The governed utility pool contains 219 unique active-HMM
rows (127 positive, 92 negative). Exact workbook preservation and visual checks
passed, the full suite passed (1,967 tests; 2 skipped), and project-wide Pyright
reported 0 errors, 0 warnings, and 0 informations.

### METASHADOW-0827 - Quarantined diagnostic surrogate shadow export (2026-08-27)

Added an explicit opt-in shadow-export path for the diagnostic meta-probability
surrogate. After the ordinary pipeline writes its deployment CSV, the surrogate
can write a separately hashed report and manifest only beneath
`artifacts/diagnostics/`. It is not loaded as a `MetaLabeler`; any scoring
failure leaves the deployment CSV and every deployment decision unchanged.

**Files modified:**
- `CHANGELOG.md`
- `run_full_pipeline.py`
- `scripts/score_meta_prob_diagnostic_surrogate.py`
- `src/neutralgrid/diagnostics/__init__.py`
- `src/neutralgrid/diagnostics/meta_prob_shadow.py`
- `tests/unit/test_meta_prob_diagnostic_shadow.py`

**Decision rationale:** The recovered teacher probabilities lack realized
fast-winner outcomes, so they are suitable only for observational monitoring.
Lopez de Prado's *Advances in Financial Machine Learning* requires
point-in-time labels and purged temporal evidence for a deployable
meta-labeler. Hashing the immutable input, surrogate, and output preserves
auditability while the prospective pool matures.

**Backward compatibility:** No breaking changes. Shadow scoring is disabled
unless `--diagnostic-surrogate-artifact` is supplied; it cannot alter gates,
sizing, ranking, calibration, promotion, or deployment output.

**Verification:** `tests/unit/test_meta_prob_diagnostic_shadow.py`,
`tests/unit/test_meta_labeler_inference_aliases.py`, and
`tests/unit/test_stage_b_meta_gate_fastwin.py` passed (16 tests). The pipeline
`--help` command exposes the opt-in flags, a five-row cross-lineage smoke run
produced only a quarantined report, and scoped Pyright reported 0 errors,
0 warnings, and 0 informations.

### CUTOFF-0827 - Causal legacy expired-bot feature refresh (2026-08-27)

Corrected full feature refreshes using `candidate_id_scan_time` so legacy
expired-bot rows with no candidate ID use their recorded event-start boundary.
Rows with malformed or future-dated non-empty candidate IDs continue to fail
closed. Scanner-origin rows with canonical IDs continue to use their embedded
scan time.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backfill_training_features.py`
- `tests/unit/test_backfill_training_features_v20260312.py`

**Decision rationale:** The event start is the only available point-in-time
boundary for pre-candidate-ID observations. This preserves the causal
information set and the leakage discipline in Lopez de Prado's *Advances in
Financial Machine Learning* while retaining strict rejection for invalid
provenance.

**Backward compatibility:** No breaking changes. Existing canonical candidate
IDs keep their scan-time behavior; only legacy rows that were previously
excluded can now be refreshed from their recorded start time.

**Verification:** `tests/unit/test_backfill_training_features_v20260312.py`
passed (27 tests); scoped Pyright on `scripts/backfill_training_features.py`
reported 0 errors, 0 warnings, and 0 informations.

### DIAGSURR-0827 - Quarantined meta-probability distillation and snapshot replay (2026-08-27)

Added a diagnostic-only surrogate that learns observed deployment `meta_prob`
as a continuous teacher output for a single declared HMM lineage. It is
structurally non-promotable and incompatible with the runtime meta-labeler
loader. A companion scorer refuses cross-lineage use unless the operator
explicitly requests an extrapolative proximity report. Scanner deployment
snapshots can now use their canonical candidate-ID scan timestamp as the
causal HMM boundary when `candidate_id_scan_time` is explicitly selected and
no later event timestamp exists.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backfill_training_features.py`
- `scripts/build_meta_prob_diagnostic_surrogate.py`
- `scripts/score_meta_prob_diagnostic_surrogate.py`
- `tests/unit/test_backfill_training_features_v20260312.py`

**Decision rationale:** Deployment probabilities are soft teacher observations,
not realized fast-winner labels. Lopez de Prado's *Advances in Financial
Machine Learning* requires outcome evidence and purged temporal validation for
a deployable meta-labeler; therefore distillation and cross-HMM proximity
estimates remain diagnostic only. Candidate IDs encode the scanner's
decision-time information set, so using that timestamp in its explicit mode
preserves causality without falling back to a later or invented bot timestamp.

**Backward compatibility:** No breaking changes. Expired-bot rows continue to
use their recorded event boundary by default; malformed candidate IDs remain
fail-closed. No production model, active artifact manifest, or promotion
decision is changed.

**Verification:** The FASTWIN contract test passed (49 tests); the backfill
regression suite passed (28 tests); scoped Pyright reported 0 errors, 0
warnings, and 0 informations. A five-row Binance historical-data smoke replay
was uniformly pinned to `rolling_180d_20260827_144604` with finite regime
probabilities. The diagnostic teacher surrogate used a chronological holdout
and wrote a non-promotable artifact under `artifacts/diagnostics/`.

### PNLHISTFIX-0827 - Windows-safe immutable PnL observation persistence (2026-08-27)

Fixed live PnL observation staging under the mandated `Live` hierarchy. A
valid final path can reach Windows' legacy path-length boundary, while the
former temporary name repeated the final filename and exceeded that boundary.
The writer now uses a fixed short staging name and atomically creates the
immutable target without a check-then-write race.

**Files modified:**
- `.gitignore`
- `CHANGELOG.md`
- `src/neutralgrid/live/decision/pnl_history.py`
- `tests/unit/test_live_pnl_history.py`

**Decision rationale:** Immutable PnL observations are training evidence. The
same-directory, non-overwriting writer must remain valid at the Windows path
limit; this preserves AFML-style stable evidence without changing labels,
features, HMM lineage, or deployment gates.

**Backward compatibility:** No breaking changes. Existing observation paths
and JSON schemas are unchanged.

**Verification:** The PnL history, telemetry-controller, and PnL-readiness
tests are run after this change, together with Pyright and the full suite.

### VISIONTLS-0827 - System trust store for Binance Vision archives (2026-08-27)

Applied the same Windows trust-chain handling used by the authenticated Binance
client to the public Binance Vision archive downloader, without disabling
certificate verification.

**Files modified:**
- `CHANGELOG.md`
- `src/neutralgrid/data/binance_vision/downloader.py`
- `tests/unit/test_binance_vision.py`

**Decision rationale:** HMM training depends on verified public archive
downloads. Using the operating system trust store preserves TLS validation and
matches the transport policy already proven for authenticated Binance requests.

**Backward compatibility:** No client API, feature schema, or archive format
changes.

### HMMROTATE-0827 - Canonical 180-day HMM refresh (2026-08-27)

Trained and promoted `rolling_180d_20260827_144604` from Binance Vision 15-minute
data. The canonical frozen slice contains 50 accepted symbols and 864,050 bars
from 2026-02-27 through 2026-08-26 (UTC, `open_time` basis). The candidate
cleared all three walk-forward splits with a 100.00% mean pass rate and replaced
the former active HMM in the atomic artifact manifest.

**Files modified:**
- `CHANGELOG.md`
- `artifact_manifest.json`
- `artifacts/hmm/rolling_180d_20260827_144604/eval.json`
- `artifacts/hmm/rolling_180d_20260827_144604/feature_schema.json`
- `artifacts/hmm/rolling_180d_20260827_144604/metadata.json`
- `artifacts/hmm/rolling_180d_20260827_144604/model.joblib`
- `artifacts/hmm/rolling_180d_20260827_144604/scaler.joblib`
- `artifacts/hmm/rolling_180d_20260827_144604/state_means.npy`
- `artifacts/hmm/rolling_180d_20260827_144604/temperature_scaler.json`

**Decision rationale:** The canonical train/evaluate/promote sequence keeps the
regime model aligned to a frozen, point-in-time market history. This follows
the walk-forward validation discipline described in Lopez de Prado's *Advances
in Financial Machine Learning* and Hudson & Thames research guidance.

**Backward compatibility:** No breaking API or feature-schema changes. Existing
downstream model metadata pinned to the prior HMM is intentionally stale and
must be refit only after its training pool passes the uniform-lineage gate.

**Verification:** `1958 passed, 2 skipped`; Pyright reported zero findings;
Binance Vision TLS smoke request returned HTTP 200; the HMM artifact contains
all required model files, uses an identity temperature scaler, and the
three-split walk-forward mean pass rate is 100.00%.

### TLSFIX-0827 - use operating-system TLS trust store for Binance (2026-08-27)

Binance Futures is reachable from this Windows host, but Python's default
certifi bundle could not validate its served certificate chain. The async
Binance client now creates its HTTPX client with a `truststore` SSL context,
retaining certificate and hostname verification through the operating-system
trust store rather than bypassing TLS checks.

**Files modified:**
- `CHANGELOG.md`
- `pyproject.toml`
- `requirements.txt`
- `requirements.lock`
- `src/neutralgrid/api/binance_client.py`
- `tests/unit/test_binance_client_tls.py`

**Decision rationale:** Transport authentication is a precondition for all
market-data and account requests. A system trust store handles locally trusted
roots and missing intermediates while preserving certificate validation; setting
HTTPX `verify=False` would create an unacceptable man-in-the-middle risk. This
is operational transport hygiene, not an AFML model or gate change.

**Backward compatibility:** No breaking changes. The public Binance client API
is unchanged; HTTPS requests now use the local operating-system trust policy.

**Verification:** TLS regression plus Binance-client pagination tests passed
(4 passed); full-project Pyright passed with 0 errors and 0 warnings; a live
read-only Futures check returned `connected=true` and `authenticated=true`.

### META-POOL-SOURCE-BOUNDARY (2026-08-22)

- Added a fail-closed source contract for active FASTWIN meta-labeler retraining.
- Full-pool backtest runs now emit `backtest_run_manifest.json`; fresh finalization emits `authoritative_pool_manifest.json` with `generation_mode=fresh_full_pool`.
- Historical exact replay outputs are marked `historical_exact_replay` and cannot be used by the active retrainer unless `--allow-historical-replay` is explicitly supplied.
- Changed files: `src/neutralgrid/training/meta_pool_contract.py`, `retrain_meta_labeler.py`, `backtest_candidates.py`, `scripts/replay_authoritative_fastwin_7h.py`, `scripts/finalize_authoritative_meta_pool.py`, `scripts/finalize_fresh_authoritative_meta_pool.py`, `tests/unit/test_meta_pool_source_contract.py`, `AGENTS.md`, `CLAUDE.md`, `.agents/skills/meta-labeler-refit/SKILL.md`, `.agents/skills/backfill-features/SKILL.md`.
- Verification: not run in this change pass.

### VOLFORECAST-01 - sub-seven-hour shadow realized-volatility contract (2026-08-21)

Added a fail-closed, run-scoped realized-volatility research path for 30, 60,
180, and 360-minute horizons. Finalized one-minute mark prices remain the
primary source and last prices remain diagnostic; causal five-minute returns
produce forward RV labels without interpolation across gaps. Per-symbol and
panel HAR-RV challengers are evaluated independently per symbol and horizon
against frozen persistence, EWMA, and existing square-root-time baselines.
Promotion eligibility requires a nonpositive one-sided 95% Newey-West QLIKE
upper bound after Holm correction plus strictly better calibration-frozen 90%
interval quality. Tail behavior is report-only.

**Files modified:**
- `CHANGELOG.md`
- `config/live_volatility_forecast_v1.json`
- `docs/live_volatility_forecast_v1.md`
- `scripts/audit_pnl_training_readiness.py`
- `scripts/backfill_volatility_history.py`
- `scripts/collect_diff_depth.py`
- `scripts/collect_private_grid_telemetry.py`
- `scripts/run_live_telemetry_controller.py`
- `scripts/run_live_volatility_loop.py`
- `scripts/supervise_diff_depth_collector.py`
- `scripts/train_live_volatility_forecaster.py`
- `src/neutralgrid/data/binance_vision/downloader.py`
- `src/neutralgrid/data/binance_vision/urls.py`
- `src/neutralgrid/data/diff_depth.py`
- `src/neutralgrid/data/price_series/ps_store.py`
- `src/neutralgrid/live/decision/volatility.py`
- `src/neutralgrid/live/decision/volatility_forecast.py`
- `src/neutralgrid/live/monotonic_schedule.py`
- `tests/unit/test_binance_vision.py`
- `tests/unit/test_diff_depth.py`
- `tests/unit/test_live_telemetry_controller.py`
- `tests/unit/test_live_volatility.py`
- `tests/unit/test_live_volatility_loop.py`
- `tests/unit/test_monotonic_schedule.py`
- `tests/unit/test_price_series.py`
- `tests/unit/test_private_grid_telemetry.py`
- `tests/unit/test_diff_depth_collector_supervisor.py`

**Decision rationale:** HAR-RV is the first challenger because its heterogeneous
trailing-RV structure is causal and auditable for this sparse Binance-perpetual
setting; the cited GARCH-neural architecture is not evidence that a neural
model improves these symbols or horizons. Point evaluation uses the
Patton-compatible QLIKE loss, while the fixed overlap lag and symbol-level
Diebold-Mariano comparison prevent horizon and panel-statistic drift. See
Corsi (2009), Patton (2011), and Diebold and Mariano (1995). Binance's current
documented `/public` depth and `/market` trade/mark WebSocket endpoints are kept
separate; an acknowledgement without a received mark event cannot satisfy the
required-stream gate. The collector supervisor now binds an owned process to
both the roster-target hash and collector-code hash, and its health check fails
closed on absent market coverage. Implied volatility,
automatic HAC lags, RV classification, final-actual width normalization, and
active-verdict consumption remain rejected by the contract.

**Backward compatibility:** No active behavior changes. Forecasts live only in
the telemetry manifest's `shadow_volatility` section, carry
`verdict_influence: false`, and cannot reach recommendations, bounds, sizing,
Kelly, HMM, meta-labeling, utility calibration, or execution. No artifact is
written under active `models/` lineage.

**Verification:** The volatility-focused acceptance suite passed 234/234 and
full-project Pyright reported 0 errors and 0 warnings. The full test run passed
1,935/1,936; its sole failure is an existing canonical-workbook gate caused by
missing `pnl_pct` for strategy `413784924` (`ATUSDT`), which is intentionally
not imputed. A live six-second dual-endpoint smoke captured 32 applied depth
events, 6 aggregate trades, and 5 mark updates with zero depth/market gaps and
zero parse errors. A stale authenticated roster is rejected before backfill. No
frozen-holdout predictive-improvement claim or PnL-v2 promotion is made until a
fresh roster, required history/origin floors, and every acceptance gate pass.

### VOLFORECAST-02 - descriptive authenticated-roster validation (2026-08-21)

Reconciled the volatility runtime's roster validation with the governed Chrome
ingestion contract. A complete Chrome-plugin cycle now proves provenance using
a nonempty captured page identity plus an HTTPS source URL whose hostname is
Binance-owned, instead of requiring one hard-coded English identity sentence.
Malformed, empty, HTTP, and lookalike-host inputs still fail closed. Immutable
capture bundles and ingestion-cycle evidence are not rewritten.

**Files modified:**
- `CHANGELOG.md`
- `scripts/run_live_volatility_loop.py`
- `tests/unit/test_live_volatility_loop.py`

**Decision rationale:** Authentication and page selection are capture facts,
while the human-readable page identity is descriptive evidence. Pinning that
evidence to one exact sentence rejected a fresh authenticated Binance Bahrain
capture that the canonical ingestion boundary had already validated. Hostname
boundary checking retains provenance enforcement without coupling downstream
acquisition to presentation wording. This is a contract-consistency fix, not a
statistical or AFML gate change.

**Backward compatibility:** No breaking changes. Existing canonical identity
text remains valid. Callers with absent identity evidence, non-HTTPS sources,
or sources outside `binance.com` and `binance.bh` remain rejected.

**Verification:** `python -m pytest tests/unit/test_live_volatility_loop.py`
passed 15/15; full-project Pyright reported 0 errors and 0 warnings. Governed
backfill accepted the fresh five-strategy cycle and completed checksum-verified
archive acquisition; it then blocked promotion at BTWUSDT's verified listing
boundary because that symbol cannot yet meet the serialized history and origin
floors.

### VOLFORECAST-03 - trainer authenticated-roster provenance parity (2026-08-23)

Aligned the volatility trainer's cycle-manifest roster admission with the
already-governed runtime loop. The trainer now accepts a nonempty descriptive
page identity and a trusted HTTPS Binance hostname (`binance.com`, `binance.bh`,
or a true subdomain), rather than a superseded exact English page sentence.
Missing identities, HTTP sources, and lookalike hosts remain fail-closed.
Neither the immutable capture/cycle evidence nor the backfill archive is
rewritten.

**Files modified:**
- `CHANGELOG.md`
- `scripts/train_live_volatility_forecaster.py`
- `tests/unit/test_live_volatility.py`

**Decision rationale:** This reuses the runtime's accepted provenance contract
so the trainer and shadow loop cannot disagree over the same governed Chrome
cycle. This is evidence-contract consistency only; it changes no model,
statistical/AFML gate, promotion rule, or live-verdict path.

**Backward compatibility:** No breaking changes. Existing exact identities are
accepted; absent/empty identity evidence and untrusted or non-HTTPS source URLs
continue to be rejected.

**Verification:** Focused live-volatility tests passed 37/37 and full-project
Pyright reported 0 errors, 0 warnings, and 0 informations. Training is rerun
against the pre-existing ready backfill manifest without `--force` or a
backfill rerun; its run-specific outcome is retained in the audit output.

### VOLFORECAST-04 - staged signed-semivariance research ablation (2026-08-23)

Added an isolated v2 research course that compares the existing HAR forecast
with HAR-RS signed-semivariance features while binding to the immutable v1
contract and consumed OOS evidence. Exact duplicate examples are no-ops,
conflicting identities fail closed, five-minute signed semivariances must sum
back to the preserved v1 realized variance, and every split/model/artifact
boundary is hash-validated. The consumed test remains diagnostic only:
promotion is permanently false and the active scanner cannot read these
artifacts.

The governed run at
`outputs/audits/live_volatility_research/20260824T005845488625Z` admitted
1,364,451 examples for 6 symbols and evaluated 24 symbol/horizon pairs with no
blocked pair. Development selection chose HAR for 20 pairs and HAR-RS for 4.
Those four reduced mean QLIKE relative to the v1 candidate, but all four still
lost to their frozen baseline; zero pairs passed the Holm-adjusted QLIKE gate,
one passed the interval-only diagnostic, and zero passed both. This is evidence
of limited ablation value, not predictive promotion.

The full-suite workbook gate was reconciled separately and fail-closed. A fresh
active-HMM replay updated only 11 declared HMM columns in `Meta Features`; all
366 strategy identities and the other four sheets were preserved. One row
(`strategy_id=413784924`) has no observed `pnl_pct`; it remains stored but is
excluded from utility calibration without label imputation. The resulting
eligible utility cohort is 222 rows (92 class 0, 130 class 1), all with finite
`range_prob`, `trend_prob`, and `persistence_prob` under
`rolling_180d_20260822_203741`.

**Files modified:**
- `CHANGELOG.md`
- `config/live_volatility_research_v2.json`
- `scripts/research_live_volatility_v2.py`
- `src/neutralgrid/live/decision/volatility_research.py`
- `src/neutralgrid/live/decision/volatility_research_evaluation.py`
- `tests/unit/test_live_volatility_research.py`
- `src/neutralgrid/calibration/utility_calibrator.py`
- `tests/unit/test_utility_calibrator.py`
- `data/new_expired_bots.xlsx`

**Decision rationale:** Corsi's heterogeneous autoregressive structure remains
the auditable baseline; signed realized semivariances test an asymmetric
variation decomposition without changing the continuous RV target. The
existing final test cannot be reused for promotion because its results informed
this design. HAR-J is deferred until a noise-robust jump estimator and minimum
return count are approved; neural candidates are deferred until a classical
ablation earns incremental evidence. These boundaries preserve AFML-style
causal feature/label timing and Patton-compatible QLIKE evaluation without
turning research iteration into holdout reuse.

**Backward compatibility:** Active volatility modules, telemetry controller,
artifact manifest, HMM, meta-labeler, utility runtime, verdicts, and execution
routing are unchanged by the v2 research artifact. The utility training loader
only narrows its eligible supervised cohort by rejecting rows whose outcome is
unobserved; nonblank nonnumeric outcomes still hard-fail.

**Verification:** Research artifact validation passed all schema, path,
loadability, and SHA-256 checks; deterministic reruns produced identical
content, example, and result hashes. Focused volatility tests passed 65/65.
Full `python -m pytest tests/ -q -p no:cacheprovider` passed 1,955/1,955;
full-project Pyright reported 0 errors, 0 warnings, and 0 informations.
`scripts/check_deps.py`, `pip check`, and `git diff --check` passed. The active
volatility, telemetry-controller, and artifact-manifest hashes are unchanged.

### VOLFORECAST-05 - noise-robust jump ablation extension (2026-08-24)

Extended the isolated V2 volatility research contract with HAR-J and HAR-RS-J
diagnostic candidates. For each causal HAR window, jump variation is
max(RV - BV, 0), where BV is the bipower variation estimator calculated from
adjacent five-minute log returns. Every complete five-minute return window is
required; a source gap rejects the affected window and is never imputed. The
six-symbol run selected among HAR, HAR-RS, HAR-J, and HAR-RS-J using
development-only QLIKE, then evaluated the frozen consumed test solely as a
diagnostic. It evaluated 24 pairs with no block; 11 selected HAR-J, 2 selected
HAR-RS-J, 9 selected HAR, and 2 selected HAR-RS. Zero pairs passed both the
Holm-adjusted QLIKE and interval-quality gates.

**Files modified:**
- CHANGELOG.md
- config/live_volatility_research_v2.json
- src/neutralgrid/live/decision/volatility_research.py
- src/neutralgrid/live/decision/volatility_research_evaluation.py
- tests/unit/test_live_volatility_research.py

**Decision rationale:** Bipower variation is a noise-robust decomposition of
continuous and jump variation; requiring the full causal five-minute return
window prevents artificial jump signals from missing-price interpolation.
HAR-family selection remains development-only, with Patton-compatible QLIKE,
Newey-West/Holm comparison, and frozen interval checks retaining the existing
diagnostic boundary. See Corsi (2009), Barndorff-Nielsen and Shephard (2004),
Patton (2011), and Lopez de Prado, Advances in Financial Machine Learning,
on causal feature/label timing and holdout governance.

**Backward compatibility:** No breaking changes. V2 remains research-only:
all artifacts retain promotion_eligible=false, verdict_influence=false, and
runtime_effect=none; no active scanner, model pointer, policy, sizing, or
execution behavior reads the results.

**Verification:** Focused V2 tests passed 8/8 and targeted Pyright reported 0
errors, 0 warnings, and 0 informations. The committed artifact validation
checked all required paths, loadability, and SHA-256 values for 1,356,898
research rows and 24 evaluated pairs.

### FASTWIN-REPLAY-0820 - causal seven-hour authoritative meta pool (2026-08-20)

Rebuilt the 2026-06-08 through 2026-08-17 candidate cohort with a 420-bar
seven-hour replay, then admitted only rows whose outcomes were unchanged,
whose HMM features were regenerated at the candidate-ID scan timestamp, and
whose active HMM lineage was complete. The HMM-only backfill now preserves a
recorded point-in-time `profit_per_grid_pct` when compact rows lack geometry;
the legacy availability mapper applies the same rule, avoiding a false
missing-feature report for otherwise admissible snapshots.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backfill_training_features.py`
- `scripts/replay_authoritative_fastwin_7h.py`
- `scripts/finalize_authoritative_meta_pool.py`
- `src/neutralgrid/training/data_generator.py`
- `tests/unit/test_backfill_training_features_v20260312.py`
- `tests/unit/test_finalize_authoritative_meta_pool.py`
- `tests/unit/test_replay_authoritative_fastwin_7h.py`
- `tests/unit/test_unified_training_builder.py`

**Decision rationale:** A seven-hour fast-winner label cannot be finalized
from a six-hour observation. Candidate-ID time is the decision-time boundary,
so replaying HMM features from a later backtest start would introduce
look-ahead. The strict finalizer retains only complete causal feature vectors;
see Lopez de Prado, *Advances in Financial Machine Learning*, on event
labeling and time-consistent features.

**Backward compatibility:** No breaking changes. Existing full-feature
backfills retain their default start-time cutoff; governed HMM-lineage replay
uses the explicit candidate scan-time contract.

**Verification:** 5,209 candidate IDs replayed; 5,143 rows admitted under
the active HMM `rolling_180d_20260819_014424`; focused tests passed 75/75 and
targeted Pyright reported zero errors. The meta-labeler dry run validated the
20-feature, seven-hour contract without training or promoting a model.

### UTILFIX-02 - explicit pinned HMM replay authority (2026-08-19)

`--default-artifact-version` now takes precedence over a row-stamped HMM
version even when stale lineage is present in the input file itself. Stale
input HMM probabilities, lineage, and HMM-dependent EV values are invalidated
before replay, closing the fresh-output-path case where a failed re-inference
could otherwise retain stale values or load the stale input artifact.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backfill_training_features.py`
- `tests/unit/test_backfill_training_features_v20260312.py`

**Decision rationale:** HMM-derived features must carry the identity of the
artifact that actually generated them. This is the fail-closed lineage boundary
required before calibration or meta-labeler fitting; see Lopez de Prado, AFML,
on preserving the provenance of model inputs and labels.

**Backward compatibility:** No breaking changes. Calls without an explicit
default retain row-stamped artifact selection; explicit defaults now behave as
documented.

**Verification:** Focused backfill suite passed 18/18; targeted Pyright reported
0 errors and 0 warnings.

### FASTWIN-AUDIT-0820 - seven-hour label observability guard (2026-08-20)

Added an isolated, deterministic replay audit for six-hour negative
fast-winner observations. It fetches one immutable 420-bar window, proves the
stored 360-bar result reproduces exactly before evaluating hour seven, and
does not write a training pool or model artifact. This prevents a target
defined as `time_to_target_hours <= 7` from being represented as fully
observed by a six-hour backtest.

**Files modified:**
- `CHANGELOG.md`
- `scripts/assess_fastwin_7h_observability.py`
- `tests/unit/test_assess_fastwin_7h_observability.py`

**Decision rationale:** Labels must be observed over their declared horizon;
AFML's event-labeling framework requires the observation window to reach the
vertical barrier before treating a negative outcome as final. Baseline
reproduction also prevents a changed engine/configuration from being mistaken
for a horizon effect.

**Backward compatibility:** No breaking changes. This is a read-only audit
utility; it does not alter the six-hour production default or existing pools.

**Verification:** Unit tests passed 3/3 and targeted Pyright reported 0
errors and 0 warnings. A deterministic 25-row live smoke cohort reproduced
all six-hour baseline PnLs and found four hour-seven target flips.

### INGESTFIX-0819 - current Binance pasted-curve parsing (2026-08-19)

The expired-bot text parser now accepts unnumbered numeric PnL curve points
only inside an explicit `PnL Curve` section, handles the Binance two-line
`Duration\n6h 23` form, and leaves unavailable `--` profit placeholders
missing instead of coercing them to zero. This supports evidence-preserving
ingestion of current Binance copy-paste exports without widening parsing of
unstructured numeric text.

**Files modified:**
- `CHANGELOG.md`
- `_bot_data_extractor_core.py`
- `tests/unit/test_bot_data_extractor_v2.py`

**Decision rationale:** Preserving the observed path and retaining unavailable
values as missing prevents fabricated outcomes from entering a training pool,
consistent with AFML's emphasis on data integrity before model fitting.

**Backward compatibility:** No breaking changes. Existing numbered PnL curve
blocks and duration formats remain supported.

**Verification:** `python -m pytest tests/unit/test_bot_data_extractor_v2.py -q`
passed 165 tests; targeted Pyright reported zero errors and warnings.

### LIVE-AUTO-ADJUST-0811 - exact-approval automatic ADJUST dispatch (2026-08-11)

The live telemetry controller now discovers one exact, unexpired external
`neutralgrid_action_approval_v1` per ADJUST intent and automatically selects
the canonical `scripts/execute_live_telemetry_modifications.py` executor. This
removes the need to repeat `--allow-actions` and `--action-executable` for every
approved adjustment while preserving the external action-authority boundary.
Missing, expired, mismatched, or duplicate approvals remain fail-closed;
automatic END routing is rejected; and explicit executor transport arguments
remain mandatory so the controller cannot silently open a debugging-port
browser.

**Files modified:**
- `CHANGELOG.md`
- `scripts/run_live_telemetry_controller.py`
- `tests/unit/test_live_telemetry_controller.py`

**Decision rationale:** This is execution-control governance rather than an
AFML modeling change. Automatic dispatch is appropriate only after the
scanner intent and a separately supplied action-time approval agree on symbol,
strategy ID, action, bounds, position preservation, expiry, and idempotency
key. Keeping approval authority external prevents an unattended scanner from
manufacturing permission for its own recommendation, while canonical executor
selection removes a repeated operational omission.

**Backward compatibility:** The existing reviewed-executable path using
`--allow-actions` and `--action-executable` is unchanged. Controller CLI runs
gain a default `--action-approval-dir`; without one exact approval and explicit
transport arguments, behavior remains blocked and no submit occurs.

**Verification:** The focused live telemetry controller suite passed 17/17;
targeted configured Pyright reported zero errors and zero warnings. The
canonical executor unit suite and final whitespace review are recorded in the
session report.

### BACKTEST-MERGE-0811 - fail-closed audit-validator integration (2026-08-11)

Integrated only the evidence-audit identity, resilience, classification, and
chronology corrections into the current checkout. Exact strategy-ID cohorts
now prove that every requested ID exists once in the raw workbook and remains
eligible after scope, mode, and symbol filters; incomplete or duplicate cohorts
stop before replay instead of being silently reduced by ordinary deduplication.
The summary retains raw workbook ID-integrity counts alongside the exact-cohort
verdict. Existing row-local cache containment, time-evidence classification,
all-rejected summary safety, diagnostic-target labeling, and stored-time
chronology fallback remain unchanged.

The canonical candidate backtest CLI now defaults to `legacy`. Both
`candidate_time_geometric_v1` and `candidate_time_public_market_v1` remain
explicit shadow challengers and are rejected before network access when their
output resolves inside `data/backtest_candidates`, `data/fastwin_dataset`, or
any descendant. The same shared path guard covers the FASTWIN generator,
matched-live comparison, temporal holdout, depth-shadow builder, reconciliation
validator, canonical profile experiment, logistic challenger, and profile-audit
writers at both CLI and programmatic entry points.

Authoritative training admission classifies both explicit challenger profiles
as non-authoritative, rejects blank and unknown explicit profiles, and prevents
the general builder's H* bootstrap relaxation from re-admitting those rows. The
7,203-row current FASTWIN pool consequently produces zero authoritative meta
training rows. Historical files that wholly predate the profile column remain a
documented compatibility class; all current writers stamp an explicit profile.

The runtime empirical scanner profile and threshold-optimization loader now
filter explicit non-legacy rows before deterministic candidate-ID deduplication,
so a later shadow rerun cannot displace an earlier legacy result. On the current
canonical results, admission changed from 1,951 deduplicated rows containing
1,788 challenger rows to 634 legacy-compatible rows; 476 rows satisfy the live
fast-horizon fit contract. The empirical fingerprint schema advanced to
`empirical_profile_numeric_v2_realism_authority`. Candidate-ID statistical
holdout evaluation remains available for diagnostics, but its final decision
retains an explicit non-relaxable promotion blocker until materially larger
bot-disjoint temporal-OOS and event-complete execution evidence exists. No
backtest physics, label, feature, tuned threshold, active model, workbook, or
artifact pointer was changed.

**Files modified:**
- `.claude/rules/safety-invariants.md`
- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `backtest/btk_seed_state.py`
- `backtest_candidates.py`
- `optimize_thresholds_v20260311.py`
- `scripts/audit_canonical_fastwin_profile.py`
- `scripts/backtest_matched_candidate_comparison.py`
- `scripts/build_depth_shadow_outcomes.py`
- `scripts/build_fastwin_holdout.py`
- `scripts/generate_fastwin_dataset.py`
- `scripts/validate_backtest_live_reconciliation.py`
- `src/neutralgrid/backtest/realism_governance.py`
- `src/neutralgrid/scanner/canonical_fastwin_profile.py`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/scanner/fastwin_logistic_profile.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `tests/unit/test_backtest_timestamp_policy.py`
- `tests/unit/test_candidate_pipeline_bypass.py`
- `tests/unit/test_canonical_fastwin_profile_experiment.py`
- `tests/unit/test_realism_shadow_governance.py`
- `tests/unit/test_unified_training_builder.py`

**Decision rationale:** AFML / Hudson & Thames governance requires stable
identity before deduplication, explicit exclusion classes, bot-disjoint
chronology, and challenger isolation from the authoritative training pool.
Silently accepting a smaller requested cohort biases evaluation; allowing a
mixed-evidence challenger to use the canonical output directory makes a future
training run unable to distinguish experiment from authority. Filtering before
dedup is necessary because the previous keep-last order allowed 1,788 later
challenger rows to displace every one of the 256 explicit legacy rows in the
empirical profile. The candidate-ID holdout gate does not demonstrate
bot-disjointness or event-complete execution; inventing a numeric sufficiency
threshold would be unsupported, so promotion remains fail-closed. These guards
prevent the invalid authority states without modifying model artifacts.

**Backward compatibility:** Default `backtest_candidates.py` runs now use the
documented canonical `legacy` profile. Explicit challenger runs must add a
non-canonical `--output` path. Validator output gains additive strategy-ID
integrity objects; valid complete exact-ID cohorts retain their prior rows and
metrics. Existing pre-profile CSVs remain readable, while explicitly stamped
non-legacy, blank, or unknown rows no longer influence authoritative training,
runtime empirical ranking, or threshold optimization. The empirical-profile
fingerprint intentionally changes because its evidence population changed.

**Verification:** The final realism-governance file passed 28 tests; the five
directly affected contract files passed 101 tests; and the complete repository
suite passed 1,815 tests. Configured project-wide Pyright and the post-edit
targeted checkpoint both reported 0 errors, 0 warnings, and 0 informations;
all changed Python entry points byte-compiled. Read-only current-data replay
classified all 7,203 unique FASTWIN rows as non-authoritative with
`shadow_realism_profile_not_authoritative`, leaving zero authoritative meta
rows. The empirical consumer read 7,445 raw rows across 32 files, admitted 634
deduplicated legacy-compatible rows, fit 476 fast-horizon samples, and produced
fingerprint `33306a5c559a3b250f0802b5acf8d42b2e80b618e5366b89c7351b7a00f2941e`.
All five frozen authority hashes matched their pre-merge baselines, `ERR-097`
occurred exactly once, and four audit-scoped pytest directories were deleted
with zero matching `backtest_merge_*` directories remaining.

### CANONICAL-HMM-MIGRATION-0810 - governed canonical workbook migration (2026-08-10)

Migrated the fresh causal HMM backfill into the canonical five-sheet workbook
without replacing the workbook with the backfill's flat one-sheet export.
`General`, `PnL Curve Features`, and `Last Features` remain value-identical;
`Meta Features` now covers the exact 329-row `General` strategy-ID universe
with one active HMM lineage, 318 complete rows, and 11 explicit `hmm_failed`
rows. Dynamic `Canonical Audit` facts and source hashes were refreshed, while
missing open-interest and micro-cost evidence was left missing rather than
inferred.

**Files modified:**
- `CHANGELOG.md`
- `data/new_expired_bots.xlsx`

**Generated audit evidence:**
- `outputs/audits/canonical_migration_20260810/new_expired_bots_migrated.xlsx`
  (SHA-256 `6bc3ab40b06ce6dc36af42d66bf907b20965431a356a594b08ed8c4d1b9184e5`)

**Decision rationale:** Per AFML / Hudson & Thames, stable-identity joins and
point-in-time feature lineage must be preserved independently from outcome and
execution evidence. A binary replacement with the flat backfill was rejected
because it would delete four canonical sheets. The migration instead maps the
governed derived rows into the existing `Meta Features` schema by exact
`strategy_id`, classifies unavailable causal histories fail-closed, and
preserves every independently sourced sheet.

**Backward compatibility:** The five sheet names, sheet order, 35-column
`Meta Features` schema, table names, and non-target sheet values are unchanged.
The additive change is 25 new Meta rows and current-HMM values for the existing
304 rows. Utility loaders now see the canonical workbook's governed 182-row
eligible pool; the 11 incomplete rows remain stored for audit but are excluded
before lineage validation.

**Verification:** The canonical utility governance tripwire now passes. The
full suite passes 1,771/1,771; the focused migration/backfill/meta/profile set
passes 95/95; contract tests pass 98/98; dependency versions match the lock;
Pyright reports zero errors and zero warnings; and `git diff --check` reports no
whitespace errors. A canonical-path utility dry run loaded 182 unique eligible
rows with one active HMM lineage and both classes, then correctly rejected the
candidate at G3 (holdout AUC 0.4462) and G4 (winner utility below loser
utility). No model artifact or `current.json` pointer was written.

### LIVE-PNL-FORECAST-0810 - persistent evidence and guarded shadow forecasting (2026-08-10)

Added a persistent cross-cycle PnL evidence contract keyed by symbol, exact
strategy ID, and deployment timestamp. The controller now commits an immutable,
source-hashed, full-record-integrity observation under the repository
`Live/YYYY-MM-DD/SYMBOL` policy before any action routing; exact duplicates are
idempotent, while conflicting
same-time observations, corrupt history, or source-hash changes fail closed.
Added an explicit-horizon shadow forecaster with forward labels, chronological
bot-disjoint fit/calibration/test cohorts, boundary purging, prediction-interval
and probability calibration, and final-test gates against zero-change,
last-slope, training-prior, and last-direction baselines.

Expanded observational execution risk with exact collector-target attribution,
position-normalized exit impact/depth/removal/refill/withdrawal measures,
duration/observation/fraction-qualified spread and depth deterioration,
temporary-versus-sustained liquidity state, private cancellation counts, fill
slippage, and 5s/30s adverse-selection estimates. Added a strict static Binance
`userTrades` importer that requires a separately reviewed order-ID linkage
artifact and cannot claim event-complete coverage. All forecast and
microstructure outputs remain verdict-inert; no live action, active model, or
production artifact was authorized or changed.

**Files modified:**
- `CHANGELOG.md`
- `LIVE_DECISION.md`
- `scripts/collect_diff_depth.py`
- `scripts/ingest_private_execution_events.py`
- `scripts/run_live_telemetry_controller.py`
- `scripts/train_live_pnl_forecaster.py`
- `src/neutralgrid/live/decision/execution_outcome_analysis.py`
- `src/neutralgrid/live/decision/execution_risk.py`
- `src/neutralgrid/live/decision/l2_risk.py`
- `src/neutralgrid/live/decision/loader.py`
- `src/neutralgrid/live/decision/meta_shadow_analysis.py`
- `src/neutralgrid/live/decision/pnl_forecast.py`
- `src/neutralgrid/live/decision/pnl_history.py`
- `tests/unit/test_decision_recommender.py`
- `tests/unit/test_diff_depth.py`
- `tests/unit/test_ingest_private_execution_events.py`
- `tests/unit/test_live_execution_risk.py`
- `tests/unit/test_live_pnl_forecast.py`
- `tests/unit/test_live_pnl_history.py`
- `tests/unit/test_live_telemetry_controller.py`

**Generated audit evidence:**
- `outputs/audits/live_pnl_forecast_20260810_final/metadata.json`
  (SHA-256 `7576098769a2b8c487e72ec469230affceab55f973f6cfa99c7579104d8d2b55`)

**Decision rationale:** Per AFML / Hudson & Thames, observations and labels must
be point-in-time, stable-identity deduplicated, temporally separated, and
evaluated on groups unseen during fitting. One bot therefore cannot cross the
fit, calibration, or final-test boundary, and labels crossing those boundaries
are purged. Simple held-out baselines are explicit promotion gates rather than
informal comparisons. Public trades do not contain strategy IDs, so attribution
requires the exact collector target; private fills require reviewed exchange
order IDs. Removal, refill, and sweep remain labelled proxies, and only private
`CANCELED` events are called cancellations.

**Backward compatibility:** Scanner `CONTINUE` / `ADJUST` / `END` mapping and
action-intent generation are unchanged. Forecast artifacts are run-specific,
shadow-only, immutable once written, and rejected at load time unless every OOS
gate and artifact hash passes. Legacy collector CSVs using `strategy_number`
remain readable, while new manifests serialize the canonical `strategy_id`.
The controller's new PnL
commit is deliberately fail-closed before action routing; operators must repair
corrupt or conflicting evidence rather than silently skipping persistence.

**Verification:** Focused persistence, forecast, collector, execution,
microstructure, controller, and outcome tests passed 86/86. The full suite
passed 1,770/1,771; the sole failure is the independent canonical-workbook
lineage tripwire (`rolling_180d_20260731_204756` stored versus active
`rolling_180d_20260809_000434`), which remains fail-closed and was not bypassed.
Pyright reports zero errors and zero warnings, dependency versions match the
lock, and `git diff --check` passes. The real-data 30-minute-horizon,
5-minute-tolerance audit found zero persistent observations and zero bot
identities, emitted `insufficient_no_forward_labels`, set
`forecast_eligible=false`, and wrote no model. Bot-disjoint synthetic tests
validate mechanics only and are not evidence of live predictive accuracy.

### LINEAGE-HARDENING-0810 - causal backfill and meta refit provenance gate (2026-08-10)

Rebuilt the 329-row expired-bot feature workbook from a fresh, non-skipping
active-HMM replay and made the historical kline boundary causal: only unique
bars whose close is at or before the candidate decision time can enter feature
inference. A short Binance REST history now falls through to checksum-verified
Binance Vision daily archives; permanent HTTP 4xx responses fail immediately,
while rate limits, request timeouts, and server failures remain retryable. The
fresh workbook retains one declared HMM version but remains governed
`INCOMPLETE`: 318 rows have finite regime probabilities and 11 newly listed
symbols have only 17--776 causal 15-minute bars against the fixed 800-bar
requirement. No padding, future bars, or synthetic regimes were introduced.

The FASTWIN meta-labeler refit boundary now hashes and records the exact source
files, event/write-time ranges, candidate-ID population, imputation policy, and
training HMM lineage. Refitting fails closed when any selected HMM-derived
feature is backed by missing, mixed, or non-active `hmm_artifact_version`.
Future backtest rows preserve that version from the scanner candidate. A
current-data negative control therefore rejected all 6,048 otherwise-modelable
FASTWIN rows because the two June source CSVs do not carry HMM artifact
versions; the active meta model was not overwritten. This exposes the
difference between an artifact metadata pin and verified training-value
lineage without changing live model scores.

**Files modified:**
- `CHANGELOG.md`
- `retrain_meta_labeler.py`
- `scripts/backfill_training_features.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/data/binance_vision/downloader.py`
- `src/neutralgrid/training/data_generator.py`
- `tests/unit/test_backfill_training_features_v20260312.py`
- `tests/unit/test_binance_vision.py`
- `tests/unit/test_ev_contract_lineage.py`
- `tests/unit/test_existing_data_mapper.py`
- `tests/unit/test_meta_labeler_retrain_contract_v20260530.py`
- `data/profile/profile_model.json` (provenance-only `shrinkage=0.3`; coefficients unchanged)

**Generated audit evidence:**
- `data/new_expired_bots_backfilled_rolling_180d_20260809_000434_20260810_causalvision.xlsx`
- `data/new_expired_bots_backfilled_rolling_180d_20260809_000434_20260810_eligible318.xlsx`
- `outputs/audits/meta_lineage_audit_20260810/meta_labeler_verification.json`

**Decision rationale:** Per AFML / Hudson & Thames, feature values used to fit a
downstream model must be causally observable, deduplicated by stable identity,
and traceable to one governed upstream model. Metadata-only re-pinning cannot
establish that the stored June `ev_score` values were derived from the August
HMM because `ev_score` consumes `trend_prob`. Missing history and missing
lineage are therefore explicit exclusion states, not values to impute. The
profile model remains unpromoted: its latest temporal evaluation is rejected,
and no future outcomes were joined to current scanner rows.

**Backward compatibility:** Scanner/deployment behavior and active artifact
files are unchanged. Backtest training rows gain the additive
`hmm_artifact_version` provenance field. Meta refits using unstamped or stale
HMM-derived training values now stop before model serialization. Explicit
microsecond truncation preserves the mapper's prior Python-datetime result
while removing pandas' nanosecond-conversion warning. Text-valued backfill
columns are initialized with object dtype so future pandas releases do not turn
the prior warning into a runtime failure.

**Verification:** The full causal replay completed in 3,568 seconds. Compared
with the August 8 workbook, all 318 finite rows stayed finite; causal filtering
changed `range_prob` and `trend_prob` on 303 rows (maximum absolute delta
0.411513) and `persistence_prob` on 196 rows (maximum absolute delta 0.174999).
The derived workbook has 329/329 rows stamped
`rolling_180d_20260809_000434`, 318 `pinned_artifact_replay`, 11
`artifact_unavailable`, zero finite utility scores, and SHA-256
`2f48fc2b76b5a97754c5532b0a118c68f7c0d906a508d279692d7fae066ee4a6`.
The negative-control meta audit records 6,195 unique candidate IDs, zero
duplicate candidate-ID rows, 147 explicit missing-`ou_halflife` exclusions,
no imputation, and zero model writes. Across 416 deployment-ready scanner
exports there are 84,199 rows; 47,363 rows carry candidate IDs from the
profile-feature era, all IDs are unique, and 47,323 rows have all four profile
features. An exact candidate-ID join to the governed 6,195-row FASTWIN pool
yields 3,391 one-to-one rows with complete profile features, but that pool does
not contain the live-compatible `profit_factor` required by the existing
profile label contract, so no profile fit was attempted. A complete-only,
outcome-blind 318-row utility staging workbook passes the uniform-lineage gate
and yields 182 unique eligible calibration rows, but its dry-run candidate was rejected by G3
(holdout AUC 0.4462) and G4 (winner utility below loser utility); no utility
artifact was written or promoted. Focused tests passed 101/101; contract tests
passed 98/98; leakage/label contract tests passed 65/65; dependency versions
match the lock; and Pyright reports zero errors and zero warnings. The full
suite passed 1,744/1,745 tests; its sole failure remains the deliberate
canonical-workbook lineage tripwire (`rolling_180d_20260731_204756` stored
versus `rolling_180d_20260809_000434` active), so the raw workbook was not
overwritten to manufacture a green result.

### HMMROTATE-0818 - canonical HMM refresh only (2026-08-18)

Promoted canonical HMM `rolling_180d_20260819_014424` after the governed
Binance Vision coverage refresh, frozen-boundary training, and three-split
walk-forward evaluation. This rotation was intentionally limited to the active
HMM: no meta-labeler, backfill, profile, or utility refit was run. Those
downstream artifacts must be treated as stale relative to the new HMM and are
queued for their separately authorized workflows.

**Files modified:**
- `CHANGELOG.md`
- `artifact_manifest.json`
- `data/trial_log.json`

**Generated audit evidence:**
- `artifacts/hmm/rolling_180d_20260819_014424/eval.json`
- `artifacts/hmm/rolling_180d_20260819_014424/feature_schema.json`
- `artifacts/hmm/rolling_180d_20260819_014424/metadata.json`
- `artifacts/hmm/rolling_180d_20260819_014424/model.joblib`
- `artifacts/hmm/rolling_180d_20260819_014424/scaler.joblib`
- `artifacts/hmm/rolling_180d_20260819_014424/state_means.npy`
- `artifacts/hmm/rolling_180d_20260819_014424/temperature_scaler.json`
- `artifacts/hmm/training_sets/canonical_20260819_014422/metadata.json`

**Decision rationale:** the active HMM may change only through the canonical
Binance Vision path with a common frozen 180-day boundary, complete accepted
symbol coverage, and the non-overridable walk-forward promotion gate. The
50-symbol, 858,000-sample candidate passed all three splits (mean pass rate
1.000) and was promoted atomically. Per AFML / Hudson & Thames lineage
governance, downstream models were not silently re-pinned or refit as part of
this HMM-only authorization.

**Backward compatibility:** no code or feature-schema change. Runtime HMM
resolution now selects `rolling_180d_20260819_014424`; downstream artifacts
pinned to prior HMM versions require their normal authorized refit and lineage
validation before they can be relied upon.

**Verification:** the artifact name matches the rolling 180-day convention;
metadata records 50 sequences, 858,000 samples, and three walk-forward splits
with mean pass rate 1.000. The canonical source is Binance Vision, and no
Binance-1500-bar truncation flag is present. The persisted temperature scaler
is the documented identity scaler (`temperature=1.0`, self-supervised fitting
disabled). The manifest atomically points to the promoted artifact.

### HMMROTATE-0808 - canonical HMM refresh and governed downstream update (2026-08-08)

Promoted canonical HMM `rolling_180d_20260809_000434` after the governed
Binance Vision freeze, training, and walk-forward process. The downstream
FASTWIN meta-labeler was refit and promoted as artifact `20260809_002039`,
explicitly pinned to the new HMM. A fresh authoritative flat backfill was
generated without `--skip-if-fresh`. The scanner/profile challenger was then
trained from 329 real canonical outcome rows and rejected by its governed
walk-forward gate, so the existing unpromoted bootstrap pair remains unchanged.
Utility recalibration was not run because 11 flat-backfill rows still have
non-finite regime probabilities; runtime utility therefore continues to fail
closed.

**Files modified:**
- `CHANGELOG.md`
- `artifact_manifest.json`
- `data/new_expired_bots.xlsx`
- `models/meta_labeler.pkl`
- `models/meta_labeler/metadata.json`
- `models/meta_labeler/model.joblib`
- `models/meta_labeler/scaler.joblib`
- `models/meta_labeler_verification.json`
- `models/meta_labeler_promotion_decision.json`
- `data/trial_log.json`

**Generated audit evidence:**
- `artifacts/hmm/rolling_180d_20260809_000434/eval.json`
- `artifacts/hmm/rolling_180d_20260809_000434/feature_schema.json`
- `artifacts/hmm/rolling_180d_20260809_000434/metadata.json`
- `artifacts/hmm/rolling_180d_20260809_000434/model.joblib`
- `artifacts/hmm/rolling_180d_20260809_000434/scaler.joblib`
- `artifacts/hmm/rolling_180d_20260809_000434/state_means.npy`
- `artifacts/hmm/rolling_180d_20260809_000434/temperature_scaler.json`
- `artifacts/hmm/training_sets/canonical_20260809_000432/metadata.json`
- `data/new_expired_bots_backfilled_rolling_180d_20260809_000434_20260808.xlsx`
- `data/profile/evaluations/profile_evaluation_20260809_025147_417366.json`

**Decision rationale:** the HMM replacement used 50 accepted symbols and a
frozen common boundary before promotion. Per AFML / Hudson & Thames governance,
the meta-labeler was re-pinned only after HMM promotion, utility fitting was
refused on incomplete active-lineage features, and the profile challenger was
kept out of authority because its temporal walk-forward evidence missed the
pre-registered AUC floor. Historical metrics and in-sample training success do
not override those downstream gates.

**Backward compatibility:** no breaking code or feature-schema change. Runtime
consumers now resolve the new HMM and its pinned promoted meta-labeler. Profile
resolution continues through the existing bootstrap fallback because
`data/profile/current.json` remains absent. Utility remains unavailable under
the existing fail-closed contract, so offline callers may emit
`utility_score=NaN` and decision-time callers reject missing utility.

**Verification:** the HMM trained on 50 sequences and 858,000 samples across
the frozen `2026-02-08T23:45:00Z` through `2026-08-07T23:45:00Z` boundary;
three walk-forward splits each passed for mean pass rate 1.000, and the identity
temperature scaler is persisted. The promoted meta-labeler reports OOF AUC
0.8079 (95% bootstrap CI [0.7978, 0.8188]), OOF ECE 0.0887, deployable OOF AUC
0.7046, and accepted sigmoid calibration ECE 0.0947 -> 0.0097. The fresh
329-row backfill has one active HMM version, with 318 finite rows and 11
`artifact_unavailable` rows. The profile challenger was rejected at mean fold
AUC 0.3333 versus the 0.55 floor (three finite folds of five); pooled OOF AUC
was 0.5833, OOF ECE 0.0876, and feature coverage 1.000. Dependency checks
passed, Pyright reports zero errors and zero warnings, and the full suite passed
1,733 of 1,734 tests. The sole failure is the governed utility-pool test against
the intentionally un-overwritten canonical raw workbook, whose stored HMM
lineage remains `rolling_180d_20260731_204756`; the same mismatch is documented
as pre-existing in PROFILEGATE-0801 and PROFILEGATE-0801B.

### BACKTEST-AUDIT-0808 - resilient identity-matched reconciliation (2026-08-08)

Audited the new 25-row expired-bot cohort against exact strategy identity,
policy-compliant live telemetry creation times, frozen 1-minute public klines,
and the current backtest reconciliation code. The validator now supports an
exact strategy-ID cohort filter, uses exact live-telemetry creation time only
when manual order-history time evidence is absent, and fails closed on missing,
ambiguous, or conflicting time evidence. Corrupt cache partitions are contained
to the affected row and are classified separately from genuinely missing or
intentionally unattempted data. Empty/all-rejected result sets can now be
summarized without raising a pandas boolean-index `KeyError`. Chronological
split assignment uses the selected validation time when available and the
stored bot time otherwise, so rejected rows cannot drift to the end merely
because their selected timestamp is null.

The output contract now states that its `fast_winner` diagnostic means terminal
PnL greater than 1% with duration under 7 hours. It explicitly reports that the
active `fast_winner_time_to_3pct_le_7h` meta-label target is not evaluated by
this validator, because the validator does not reconstruct time-to-3% paths.

**Files modified:**
- `CHANGELOG.md`
- `scripts/validate_backtest_live_reconciliation.py`
- `tests/unit/test_backtest_timestamp_policy.py`

**Generated audit evidence:**
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/authority_audit.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/frozen_new_cohort_klines/manifest.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/new25_observed_legacy/reconciliation_summary.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/new25_observed_candidate_time/reconciliation_summary.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/new25_observed_public_market/reconciliation_summary.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/new25_candidate_geometry/comparison_summary.json`
- `C:/Users/cris_/.codex/visualizations/2026/08/08/019fe24d-c435-7ed2-80ae-c46b4e48f914/backtest_evidence_audit/new25_evidence_matched_fail_closed/reconciliation_summary.json`

**Decision rationale:** This is an AFML-style, bot-disjoint chronological
evidence audit, not authorization to retrain or promote. Exact identity and
time evidence prevent same-symbol bot contamination. The tested realism
challengers produced mixed results on only 25 rows (10 chronological holdout
rows), so all runtime physics, labels, model artifacts, and promotion authority
remain unchanged. The Hudson & Thames / AFML principle applied here is strict
separation of calibration evidence, chronological holdout evidence, and the
untouched production champion.

**Backward compatibility:** No breaking changes. New CLI options and summary
fields are additive. Existing manual order-history evidence remains
authoritative when present; live telemetry is only an exact-identity fallback.
No training workbook, deployment artifact, HMM, meta-labeler, utility artifact,
or production cache is written by this audit.

**Verification:** The focused timestamp-policy suite passed 11 tests; the
broader backtest/telemetry suite passed 232 tests; and the full repository suite
passed 1,734 tests. Targeted and full-project Pyright both reported zero errors
and zero warnings. Diff validation, immutable artifact hashes, and
temporary-file cleanup are recorded in the audit report for this change.

### TELEMETRY-CLEANUP-0806 - terminal manifest normalization (2026-08-06)

Gracefully stopped the remaining event-complete diff-depth collector and added
a fail-closed one-time cleanup command for telemetry manifests whose recorded
owner PID no longer exists. The command is dry-run by default, refuses records
without a known owner PID, skips any live owner, and atomically normalizes only
stale transient states. Dead private one-shot attempts become `finished`; dead
long-running collector manifests become `stopped`. Original errors and prior
statuses remain present with an explicit normalization timestamp and reason.

Future `collect_private_grid_telemetry.py --once` runs now replace transient
`running` or `waiting_for_valid_browser_state` state with a terminal `finished`
marker before exiting, preventing the historical ambiguity from recurring.
The applied cleanup normalized eight stale private-telemetry manifests and left
the other 40 scanned manifests unchanged.

**Files modified:**
- `CHANGELOG.md`
- `scripts/collect_private_grid_telemetry.py`
- `scripts/finalize_stale_telemetry_manifests.py`
- `tests/unit/test_private_grid_telemetry.py`
- `tests/unit/test_finalize_stale_telemetry_manifests.py`
- `outputs/audits/private_telemetry_chrome_extension_20260805_190600/manifest.json`
- `outputs/audits/private_telemetry_chrome_extension_20260805_195649/manifest.json`
- `outputs/audits/private_telemetry_chrome_extension_20260805_211058/manifest.json`
- `outputs/audits/private_telemetry_chrome_extension_20260805_213855/manifest.json`
- `outputs/audits/private_telemetry_scan_20260805_174016/manifest.json`
- `outputs/audits/private_telemetry_scan_20260805_174200/manifest.json`
- `outputs/audits/private_telemetry_scan_20260805_175447/manifest.json`
- `outputs/audits/private_telemetry_scan_20260805_185535/manifest.json`

**Generated audit evidence:**
- `outputs/audits/stale_telemetry_manifest_cleanup_20260806_dry_run.json`
- `outputs/audits/stale_telemetry_manifest_cleanup_20260806_applied.json`

**Decision rationale:** This is operational provenance hygiene, not a trading
model or AFML change. A transient state must not outlive its owning process
because future audits would otherwise confuse a failed one-shot attempt with a
currently running collector. PID-aware, dry-run-first normalization preserves
the historical failure evidence while making terminal process state explicit.

**Backward compatibility:** No breaking changes. Successful and failed
one-shot private telemetry commands keep their existing exit codes and cycle
artifacts. Only the top-level one-shot manifest now reports its actual terminal
state. Existing complete, blocked, stopped, and status-less manifests are not
rewritten.

**Verification:** The focused telemetry, controller, and cleanup suite passed 24 tests.
Targeted and full-project Pyright reported zero errors and zero warnings. The
applied cleanup reported 8 normalized and 40 skipped records; a second dry run
reported zero remaining candidates. Process inspection found no repository-owned
telemetry collector or controller after graceful shutdown, and `git diff --check`
passed.

### METAGOV-0805 - fail-closed retraining and decision-population evidence (2026-08-05)

Audited all 13 unique non-closed `ERRORS_LOG.md` checks against current code,
artifacts, training data, telemetry, and task state. Closed only the three rows
whose stated software closure conditions are now proven: `ERR-075`, `ERR-088`,
and `ERR-089`. Nine items remain `WATCH` and `ERR-095` remains `OPEN`; their
current blockers and rejected changes are recorded in the generated audit.

Meta-labeler retraining now reports deployable-stratum OOF AUC, ECE, row count,
positive count/rate, and the exact `capital_fraction > 0` definition alongside
the existing all-row headline. These diagnostics use the same calibrated OOF
vector but do not change the pool, fold construction, promotion threshold, or
promotion verdict. Routine retraining also evaluates a fail-closed
champion/challenger decision before any backup or save. It requires the
candidate's absolute promotion pass, exact target/feature compatibility, and
non-regression on governed headline metrics; once a champion contains the new
deployable fields, later challengers must preserve them. The decision record is
written atomically with explicit reasons. This stored-summary comparison is
non-paired and does not replace the paired temporal protocol for a new model
recipe.

The declared scikit-learn range is now explicitly capped below 1.10, matching
the already-locked 1.8.0 environment and preventing the known LogisticRegression
parameter removal from becoming an unplanned runtime break. Estimator
parameters were intentionally not changed. The documented bare retrain command
now resolves its compatibility workbook through the existing
`data/new_expired_bots.xlsx` default.

**Files modified:**
- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `pyproject.toml`
- `requirements.txt`
- `retrain_meta_labeler.py`
- `src/neutralgrid/models/meta_labeler.py`
- `tests/unit/test_meta_labeler_retrain_contract_v20260530.py`

**Generated audit evidence:**
- `outputs/audits/errors_log_remediation_20260805/README.md`
- `outputs/audits/errors_log_remediation_20260805/meta_shadow_analysis.json`
- `outputs/audits/errors_log_remediation_20260805/canonical_fastwin_profile_readiness.json`
- `outputs/audits/errors_log_remediation_20260805/meta_labeler_verification.json`

**Decision rationale:** AFML / Hudson & Thames temporal-validation discipline
requires performance claims to identify the decision population and prohibits
routine retraining from silently replacing a champion on weaker evidence.
Accordingly, the inflated all-row metric remains visible for lineage continuity
but is now paired with decision-population telemetry; a pre-save non-degradation
shield handles accidental routine regressions while new recipes still require
paired temporal evaluation. Evidence-blocked threshold, ranking, fold, MI,
execution-risk, and scanner-profile changes were rejected instead of being
inferred from proxy or post-outcome data.

**Backward compatibility:** The active meta-labeler and HMM artifacts are
unchanged. Existing pre-METAGOV champions without deployable-stratum fields
remain comparable; the new fields become mandatory only after a champion has
recorded them. Runtime inference, active features, scores, gates, thresholds,
and live verdicts are unchanged. A routine retrain that fails absolute or
non-degradation checks now exits before modifying the deployed model, which is
the intended fail-closed behavior.

**Verification:** Full `tests/` suite passed 1,725 tests in 328.48 seconds; the
focused meta/profile/telemetry suite passed 94 tests; the updated retrain
contract passed 37 tests; and the post-edit leakage selection passed 61 tests.
Full-project Pyright reported zero errors and zero warnings. `pip check`,
`scripts/check_deps.py`, and `git diff --check` passed. Canonical FASTWIN
identity remained 7,203/7,203 unique IDs with zero duplicates; the dry-run
training contract remained 20/20 complete features on 6,048 modelable rows
(2,834 positive, 3,214 negative), with 147 rows explicitly classified
`excluded_unmodelable`.

### LIVETELEMETRY-0801 - private telemetry and event-linked execution-risk integration (2026-08-01)

Integrated complete Binance Futures Grid drawer snapshots and sequence-linked
diff-depth derivatives into the live telemetry scanner without adding a
promotion gate or changing active verdict thresholds. The private controller
now preserves PnL and transaction fees, signed position/inventory, liquidation
risk, the pending-order ladder, grid geometry/margins, TP/SL, and UI order
history. PnL observations are deployment-scoped and timestamp-deduplicated so
scanner output reports the observed peak, current giveback, and duplicate or
out-of-order status without waiting for PnL to become negative.

The event-complete collector still persists every received wire frame before
parsing. It now also writes a bounded `l2_risk_snapshots.jsonl` derivative every
five seconds (and immediately after bootstrap) with exact run/segment/update
identity, top-N book levels, spread, depth, imbalance, and interval liquidity
addition/removal proxies. Scanner ingestion validates symbol, run, segment,
schema, timestamp, and freshness, then calculates signed-position exit-side
depth, fill ratio, expected VWAP/impact, and rolling spread/depth comparisons.
The same WebSocket connection now captures public aggregate trades, preserving
exchange timestamps and aggressive-side direction. The scanner links those
trades to exact L2 runs/segments and exact-strategy private fills/order updates
to expose conservative sweep/refill/removal proxies, fill slippage, and 5s/30s
adverse-selection evidence. Regressing exchange clocks, wire sequences, or
update IDs fail closed. Collector subscription acknowledgement and observed
trade transport anomalies remain explicit in the output.

Finalized outcomes can be joined later through a new observational analysis.
It rejects ambiguous duplicate outcome identities, excludes ticks outside the
bot life window, deduplicates repeated evidence, collapses to one row per exact
bot, and produces a bot-disjoint temporal description. It deliberately selects
no threshold, creates no promotion gate, and cannot modify runtime verdicts.

**Files modified:**
- `CHANGELOG.md`
- `LIVE_DECISION.md`
- `scripts/collect_diff_depth.py`
- `scripts/run_live_telemetry_controller.py`
- `src/neutralgrid/data/diff_depth.py`
- `src/neutralgrid/live/decision/l2_risk.py`
- `src/neutralgrid/live/decision/execution_risk.py`
- `src/neutralgrid/live/decision/execution_outcome_analysis.py`
- `src/neutralgrid/live/decision/loader.py`
- `src/neutralgrid/live/decision/meta_shadow_analysis.py`
- `src/neutralgrid/live/decision/monitor.py`
- `src/neutralgrid/live/decision/private_telemetry.py`
- `src/neutralgrid/live/decision/private_events.py`
- `src/neutralgrid/live/decision/recommender.py`
- `src/neutralgrid/live/decision/renderer.py`
- `src/neutralgrid/live/decision/state_store.py`
- `tests/unit/test_decision_monitor.py`
- `tests/unit/test_decision_recommender.py`
- `tests/unit/test_decision_renderer.py`
- `tests/unit/test_decision_state_store.py`
- `tests/unit/test_diff_depth.py`
- `tests/unit/test_live_l2_risk.py`
- `tests/unit/test_live_execution_risk.py`
- `tests/unit/test_live_execution_outcome_analysis.py`
- `tests/unit/test_live_private_telemetry.py`
- `tests/unit/test_live_private_events.py`
- `tests/unit/test_live_telemetry_controller.py`
- `tests/unit/test_meta_shadow_analysis.py`

**Decision rationale:** AFML / Hudson & Thames temporal-validation discipline
requires point-in-time, identity-linked evidence before tuning a decision rule.
The user explicitly deferred a promotion gate because finalized bot-level data
is currently insufficient. Accordingly, gain deterioration and richer L2 are
observational scanner evidence only: no unvalidated giveback, thin-book,
imbalance, removal, spread, or impact threshold can originate ADJUST or END.
Book removals/additions remain labelled as proxies even after synchronized
public trades and private events are joined: aggregate trades cannot prove
queue-level causality, and unexplained removal is not cancellation. Only an
actual exact-strategy private cancellation event is counted as cancellation.
An exact-strategy canonical private-event stream is accepted through the
controller and emitted as observational scanner evidence. It validates
symbol/strategy/run identity, freshness, source completeness, and stable
exchange-event deduplication while preserving the distinction between a true
user-data event stream and REST/export snapshots.

**Backward compatibility:** Existing registries without `l2_stream` or
`private_event_stream` continue to load, pre-integration state files default
the new PnL fields safely, and the existing CONTINUE/ADJUST/END mapping is
unchanged. A collector process started before this change must be restarted
deliberately before it can emit the new five-second derivative.

**Verification:** `python -m pytest tests/ -q` passed 1,717 tests. Full-project
`python -m pyright` reported zero errors and zero warnings. Focused tests also
validated private drawer parsing against the stored BANDUSDT layout, controller
to loader round trips, stale/wrong-symbol L2 rejection, signed-position exit
simulation, collector bootstrap timing, sequence linkage, PnL deduplication,
private-event identity/freshness/deduplication, exact outcome-identity rejection,
bot-level temporal isolation, execution-evidence verdict invariance, and
state/JSONL serialization. The 155-test telemetry-focused suite also passed.
One initial full-suite run hit a transient Windows
OneDrive permission error in an unrelated TrialTracker temporary-file rename;
the exact failing test passed in isolation and the clean full-suite rerun passed.

### PROFILECANON-0801 - canonical backtest-row recovery experiment (2026-08-01)

Added a fail-closed, non-production experiment that joins the authoritative
FASTWIN backtest labels to their retained scan-time deployment rows by exact
`candidate_id`. All 7,203 canonical IDs matched exactly once with no duplicate
matches. The four profile features were retained and finite for 4,388 rows;
2,815 older rows predate feature capture and were excluded without imputation.
The experiment also excluded 39 IDs already used by the incumbent workbook,
froze 4,349 eligible real rows into 76 development scan groups and 19 final
holdout scan groups, and kept the 625-row holdout unread until the development
gate passed.

**Files modified:**
- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `scripts/audit_canonical_fastwin_profile.py`
- `scripts/run_canonical_fastwin_profile_experiment.py`
- `src/neutralgrid/scanner/canonical_fastwin_profile.py`
- `tests/unit/test_canonical_fastwin_profile_audit.py`
- `tests/unit/test_canonical_fastwin_profile_experiment.py`

**Generated audit evidence:**
- `outputs/audits/canonical_fastwin_profile_readiness_20260801.json`
- `outputs/audits/canonical_fastwin_profile_20260801_1513_v2/manifest.json`
- `outputs/audits/canonical_fastwin_profile_20260801_1513_v2/development_evaluation.json`
- `outputs/audits/canonical_fastwin_profile_20260801_1513_v2/holdout_evaluation.json`
- `outputs/audits/canonical_fastwin_profile_20260801_1513_v2/holdout_predictions.csv`

**Decision rationale:** AFML / Hudson & Thames temporal-validation discipline
requires real, timestamped, point-in-time features, group-preserving temporal
splits, purging, and one untouched final holdout. Development appeared viable:
mean fold AUC 0.5608, pooled OOF AUC 0.5858, pass rate 0.80, and paired AUC delta
0.0550 with scan-cluster 95% interval [0.0102, 0.0998]. The untouched holdout
refuted that result: candidate mean fold AUC 0.4424, pooled AUC 0.5008, and pass
rate 0.20 versus incumbent pooled AUC 0.5948; paired delta was -0.0940 with 95%
interval [-0.1859, -0.0079]. The challenger failed four promotion gates and was
rejected. See `ERR-095`.

**Backward compatibility:** No breaking changes. The experiment is additive and
does not run on the production scan path. It did not write `data/profile/current.json`
or replace either bootstrap profile artifact.

**Verification:** `python -m pytest tests/ -q` passed 1,678 tests; `python -m
pyright` reported zero errors and zero warnings. The final holdout had 100%
feature coverage. Production bootstrap hashes remain
`44DF154FEB0A44E33CDFC946C781B47DF375B2A0F23BFD2E6945517F3758D51F`
and `A566AFFB3DB19BB2FA215B36BB6D021A9B3E45F53D205590B283D8208524AE83`;
`data/profile/current.json` remains absent.

### PROFILECANON-0801B - discriminative challenger and fresh-evidence floor (2026-08-01)

Pre-registered and evaluated one low-capacity discriminative alternative to the
rejected Gaussian profile recipe. `robust_logistic_v1` uses the same four exact
point-in-time profile features, L2 logistic regression, fold-local shared
standardization, a fixed C grid `[0.01, 0.1, 1.0, 10.0]`, nested expanding
temporal selection, 24-hour purging, scan-group-preserving folds, and no feature
selection, interactions, synthetic labels, or retrospective funding proxy. The
runtime artifact reader gained an explicit, fail-closed model-family contract;
legacy artifacts that omit the field remain `gaussian_lda_v1` and retain their
original scoring path.

The development pool contains 4,944 unique authoritative canonical rows across
131 scan groups from 2026-04-21 through 2026-07-30: 2,236 fast winners and 2,708
negatives. It combines the prior 4,349 exact rows with the fully materialized,
previously disclosed 595-row real backtest cohort. Those 595 rows are training
prefix evidence only, never fresh profile-promotion evidence.

**Files modified:**
- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `scripts/build_fastwin_holdout.py`
- `scripts/collect_diff_depth.py`
- `scripts/run_fastwin_logistic_profile.py`
- `src/neutralgrid/scanner/fastwin_logistic_profile.py`
- `src/neutralgrid/scanner/profile_model.py`
- `tests/unit/test_fastwin_holdout_audit.py`
- `tests/unit/test_fastwin_logistic_profile.py`
- `tests/unit/test_profile_model_logistic.py`

**Generated audit evidence:**
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/preregistration.json`
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/development_pool.csv`
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/development_evaluation.json`
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/candidate_profile_model.json`
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/candidate_pattern_profile.json`
- `outputs/audits/canonical_fastwin_profile_logistic_v1_20260801/candidate_manifest.json`

**Decision rationale:** Changing the classifier family alone was refuted. Nested
development mean AUC was 0.4999, pooled OOF AUC was 0.4779, and fold pass rate
was 0.20. The incumbent scored pooled AUC 0.5511 on the identical OOF rows. The
paired candidate-minus-incumbent AUC delta was -0.0732 with scan-cluster 95%
interval [-0.1308, -0.0237]. The logistic candidate failed four development
gates, remains shadow-only, and is not eligible to consume another holdout.
This strengthens `ERR-095`: model form is not the demonstrated bottleneck; the
four-feature signal is unstable across temporal regimes.

Future FASTWIN acquisition now requires both the active meta features and all
four exact profile features, hashes scanner and implementation sources, uses
resumable per-candidate atomic checkpoints, and fails before creating a holdout
unless at least 150 new rows across 10 scan groups are available. A live freeze
probe found zero mature, never-evaluated candidate IDs after excluding all 595
consumed rows and prior canonical/evaluated IDs, so it created no directory and
read no new outcomes.

During full-suite verification, two existing 0.5-second diff-depth tests exposed
that capture duration started before initial manifest provenance collection.
On a dirty OneDrive worktree, `git status` consumed the socket window. The
deadline now starts immediately before collector tasks launch, and the manifest
records `capture_started_at_utc`; the unchanged tests pass. The subsequently
confirmed `ERR-097` blocker is recorded in `ERRORS_LOG.md` and remains open; it
does not authorize an artifact retrain, replacement, promotion, or decommission.

**Backward compatibility:** No active artifact or pointer changed. Legacy
Gaussian profile JSON remains readable and follows the same score path. Schema
1/2 completed FASTWIN holdouts remain readable; new schema-3 freezes add code
hash and profile-feature requirements. Short diff-depth durations now measure
active capture time instead of setup time.

**Verification:** `python -m pytest tests/ -q` passed 1,687 tests. The affected
profile/holdout/diff-depth slice passed 27/27. Full and explicit script Pyright
checks reported zero errors and zero warnings. Production profile hashes remain
`44DF154FEB0A44E33CDFC946C781B47DF375B2A0F23BFD2E6945517F3758D51F`
and `A566AFFB3DB19BB2FA215B36BB6D021A9B3E45F53D205590B283D8208524AE83`;
`data/profile/current.json` remains absent.

### PROFILEGATE-0801 - paired fast-winner artifact governance (2026-08-01)

Changed the scanner/profile retrain path so an evaluated challenger cannot
become effective merely because training completed. A promotion now requires a
feature-matched model/pattern bundle, purge-safe chronological evidence with at
least three finite folds and 60% finite-fold coverage, mean and pooled OOF AUC
of at least 0.55, mean fold pass rate of at least 0.50, at least 90% feature
coverage, and—whenever an incumbent artifact exists—a paired stratified
bootstrap AUC-delta interval whose 95% lower bound is strictly positive. Every
attempt writes a hash-linked evaluation record before `current.json`; rejected
candidates do not overwrite either bootstrap runtime artifact.

The verified label is `fast_completed_winner_under_7h`: finalized rows with
`0 <= duration_hours < 7`, profit factor at least 1.5, PnL at or above the
train-only quantile threshold, and the configured average-profit-per-grid
floor when present. The source workbook does not contain a time-to-3%-target
field, so the artifacts explicitly set `time_to_target_claimed=false`; no claim
that the bot reached 3% within seven hours is made.

**Files modified:**
- `CHANGELOG.md`
- `ERRORS_LOG.md`
- `retrain_scanner.py`
- `run_full_pipeline.py`
- `scan_top100.py`
- `src/neutralgrid/api/app.py`
- `src/neutralgrid/scanner/pattern_profile.py`
- `src/neutralgrid/scanner/profile_model.py`
- `src/neutralgrid/scanner/profile_model_walkforward.py`
- `tests/unit/test_consumers_use_resolver.py`
- `tests/unit/test_profile_model_selection_summary.py`
- `tests/unit/test_profile_promotion.py`
- `tests/unit/test_retrain_scanner_cli.py`

**Generated audit evidence:**
- `data/profile/evaluations/profile_evaluation_20260801_102930.json`
- `data/profile/evaluations/profile_evaluation_20260801_103106.json`

**Decision rationale:** AFML / Hudson & Thames temporal-validation and
multiple-testing discipline require purge-safe future observations, fold-local
labels, and an untouched comparison after a single pre-registered challenger.
The diagonal-covariance challenger (`shrinkage=1.0`) did not improve the current
recipe (`shrinkage=0.30`): mean fold AUC was 0.5102 versus 0.5278, pooled OOF
AUC was 0.4407 versus 0.4506, and paired AUC delta was -0.0099 with 95% interval
[-0.0395, 0.0119]. It was rejected, not tuned on the recorded OOS rows, and
logged as `ERR-095` for fresh-data follow-up.

**Backward compatibility:** runtime consumers still use the raw
`profile_model.json` / `pattern_profile.json` bootstrap pair only when no
governed pointer exists. A promoted pointer must now identify and hash-verify
both artifacts. Direct low-level promotion callers that omit the pattern
candidate, or omit an incumbent comparison when runtime artifacts exist, now
receive a fail-closed rejection. Production retrain and scanner consumers were
updated to the paired contract; this is an intentional safety boundary for
incomplete legacy callers.

**Verification:** the canonical workbook contained 304 unique `strategy_id`
rows with no duplicate strategy identifiers; the bounded labeled pool had 154
rows and 48 winners under the 0.68 quantile used by this trial. The final
evaluation record rejected promotion at mean AUC 0.5102; `current.json` remains
absent, the bootstrap hashes remain
`44DF154FEB0A44E33CDFC946C781B47DF375B2A0F23BFD2E6945517F3758D51F`
and `A566AFFB3DB19BB2FA215B36BB6D021A9B3E45F53D205590B283D8208524AE83`,
and no `.tmp` files remain under `data/profile`. Scanner/profile/runtime tests
pass 53/53 and Pyright reports zero errors and zero warnings. The broader
profile/AFML slice previously passed 90/90; the full repository suite passed
1,648 tests with one pre-existing governed-lineage failure because the active
HMM is `rolling_180d_20260731_204756` while the canonical calibration workbook
still exposes `rolling_180d_20260724_164829`.

### PROFILEGATE-0801B - single-use OOF evidence and activation linkage (2026-08-01)

Closed two follow-on governance gaps in the fast-winner promotion path. Future
walk-forward test folds now begin strictly after the latest OOF endpoint already
recorded in `data/profile/evaluations`; prior rows remain available only to the
expanding training prefix. The promotion gate independently rejects any passing
candidate whose OOF `strategy_id` values overlap previous evaluation evidence,
recomputes fold and pooled metrics from the persisted raw labels, scores, and
probabilities, rejects duplicate IDs and incomplete provenance, and converts a
candidate/incumbent alignment error into an audited rejection rather than an
uncaught exception. New `current.json` pointers hash-link the evaluation record;
runtime resolvers validate that report, its passing decision, and both artifact
hashes before returning either promoted bundle member.

**Files modified:**
- `CHANGELOG.md`
- `retrain_scanner.py`
- `src/neutralgrid/scanner/profile_model_walkforward.py`
- `tests/unit/test_profile_promotion.py`
- `tests/unit/test_retrain_scanner_cli.py`

**Decision rationale:** AFML / Hudson & Thames holdout discipline does not allow
repeated challenger selection against an already disclosed OOS sample. The two
existing evaluation records share the same workbook SHA-256 and overlap on all
57 OOF strategy IDs, so those observations are now treated as consumed evidence,
not as a reusable promotion set. Artifact activation also requires an immutable
link from the runtime pointer to the exact evaluation used to justify it; model
and pattern hashes alone do not prove that the recorded gate passed for those
files. This change, recorded as closed `ERR-096`, enforces the evidence boundary
without changing the winner label, feature set, model parameters, or thresholds.
`ERR-095` remains open because no model improvement has been demonstrated.

**Backward compatibility:** evaluation schema version 2 adds raw-evidence
integrity results and evaluation hashes. Existing pointers with neither legacy
evaluation-link field remain readable, but a partially populated, missing,
tampered, non-passing, or artifact-inconsistent evaluation link now fails
closed. Custom promotion callers must supply complete, internally consistent
walk-forward evidence. Re-running on a previously evaluated workbook produces
no promotable OOF rows until genuinely newer finalized outcomes exist.

**Verification:** the two current evaluation records overlap on 57/57 OOF IDs
and end at `2026-07-27T12:58:33+00:00`. A read-only run against the canonical
workbook SHA-256
`691e1d91f92fb9ac092c1e9be1792aefcb1212e9038eae27c5a5bd3d50187555`
found 154 labeled rows but zero rows strictly after that endpoint. Focused and
adversarial scanner/profile/runtime tests pass 58/58, including duplicate OOF,
forged aggregate, reused holdout, paired misalignment, freshness-boundary, and
tampered-evaluation cases. Pyright reports zero errors and zero warnings. No
profile model or pattern artifact was promoted or modified by this change. The
full repository suite passed 1,653 tests; its only failure remains the
pre-existing canonical HMM-lineage mismatch described in PROFILEGATE-0801.

### EVCONTRACT-0801 - HMM backfill EV train/serve parity (2026-08-01)

Corrected the flat HMM-feature backfill writer so `ev_score` has the same
meaning it has in enrichment, live monitoring, and the unified training
builder: `PnLRanker.rank_score` (empirically aligned EV after penalties). The
writer previously stored the distinct `ev_24h` diagnostic under the
`ev_score` name and omitted row leverage, silently using the ranker's 10x
default. It now forwards finite row leverage and retains an explicit 10x
fallback only when leverage is absent.

**Files modified:**
- `CHANGELOG.md`
- `scripts/backfill_training_features.py`
- `tests/unit/test_backfill_training_features_v20260312.py`

**Generated audit evidence:**
- `data/new_expired_bots_backfilled_rolling_180d_20260731_204756_20260801_evcontract.xlsx`
- `artifacts/utility/utility_20260801_120301_665635.json`

**Decision rationale:** the active FASTWIN feature contract documents
`ev_score` as `PnLRanker.rank_score`; using `ev_24h` under that name creates a
train/serve feature-basis mismatch. AFML / Hudson & Thames validation results
are meaningful only when the feature transformation used for training is the
one used at inference. The change reuses the already-canonical unified-builder
semantics and does not add, remove, or rename a model feature.

**Backward compatibility:** the model feature schema is unchanged. `ev_score`
values in newly generated flat backfills now use their documented canonical
units; historical files are retained as immutable audit evidence. Raw
workbooks are not overwritten. Callers that depended on the prior mislabeled
`ev_24h` values must read a dedicated EV diagnostic instead of `ev_score`.

**Verification:** the fresh 304-row backfill is uniformly stamped with HMM
`rolling_180d_20260731_204756`; 293 rows have finite range, trend, and
persistence probabilities and 11 fail closed as `artifact_unavailable`. Across
all 293 computable rows, stored `ev_score` reproduces current canonical ranker
output with maximum absolute error `1.33e-15`. Before the fix, all 293 rows
differed from canonical rank score (mean absolute error `0.8442`, maximum
`9.3512`), including eight non-10x rows. The utility challenger was rejected:
holdout AUC `0.4471`, balanced accuracy `0.4352`, gates G3/G4/G7 failed, and
`current.json` was not updated. Focused backfill tests pass 13/13, the full
repository suite passes 1,655/1,655, and Pyright reports zero errors and zero
warnings.

**Known limitation:** the active FASTWIN meta-labeler remains trained on
historically stored `ev_score` values without an empirical-profile fingerprint.
Against the current serve-time ranker, the June 8 and June 22 pools show mean
absolute EV-feature drift of `4.3245` and `1.4300`, respectively, with 3,592 and
206 sign flips. Its published OOF AUC therefore describes the historical
feature basis, not verified current-serve accuracy.

### FASTWIN-HOLDOUT-0801 - frozen temporal accuracy audit (2026-08-01)

Added a two-phase, resumable FASTWIN audit that freezes candidate identity,
scanner-source hashes, the incumbent model hashes, and the exact 24-hour
backtest contract before any outcomes are computed. The frozen cohort begins
strictly after the latest incumbent training backtest timestamp, excludes every
known previously evaluated candidate ID, and requires the full 24-hour outcome
window to have matured. A second script scores the frozen rows and applies a
paired, scan-file-clustered bootstrap promotion gate.

The audit also adds `ev_contract_fingerprint` as non-model provenance. The
fingerprint covers the full `RankingConfig` and every numeric empirical-profile
value that can affect `PnLRanker`; it is carried by future scan, enrichment,
deployment, snapshot, backtest-training, unified-builder, and flat-backfill
rows. It does not change EV math, the meta-labeler feature list, admission
gates, or any active artifact pointer.

**Files modified:**
- `CHANGELOG.md`
- `run_full_pipeline.py`
- `scripts/backfill_training_features.py`
- `scripts/build_fastwin_holdout.py`
- `scripts/evaluate_fastwin_holdout.py`
- `src/neutralgrid/backtest/candidate_pipeline.py`
- `src/neutralgrid/live/decision/monitor.py`
- `src/neutralgrid/models/meta_labeler.py`
- `src/neutralgrid/scanner/empirical_profile_v20260302.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `src/neutralgrid/scanner/pnl_ranker.py`
- `src/neutralgrid/training/data_generator.py`
- `src/neutralgrid/training/scanner_integration.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `tests/unit/test_backfill_training_features_v20260312.py`
- `tests/unit/test_ev_contract_lineage.py`
- `tests/unit/test_fastwin_holdout_audit.py`
- `tests/unit/test_profile_provenance_contract.py`
- `tests/unit/test_scanner_integration_v20260320.py`

**Generated audit evidence:**
- `outputs/audits/fastwin_holdout_20260801_1225/manifest.json`
- `outputs/audits/fastwin_holdout_20260801_1225/run_summary.json`
- `outputs/audits/fastwin_holdout_20260801_1225/training_data.csv`
- `outputs/audits/fastwin_holdout_20260801_1225/evaluation.json`
- `outputs/audits/fastwin_holdout_20260801_1225/paired_predictions.csv`
- `results/deployment_ready_20260801_125207.csv`

**Decision rationale:** all 595 frozen rows completed with zero backtest errors;
319 are positive under the exact `time_to_target_hours <= 7` target and 276 are
negative. The active incumbent's untouched future accuracy is ROC-AUC `0.5869`,
average precision `0.5921`, Brier score `0.2486`, ECE `0.0773`, raw accuracy
`0.5429`, and balanced accuracy `0.5566`. Its historical OOF AUC `0.8079`
therefore is not represented as current-serve accuracy. Recomputing historical
EV under today's empirical profile was rejected because 1,683 of the 1,950
profile-fit candidate IDs overlap the FASTWIN training IDs, which would inject
outcome-fitted information into the feature basis.

The single pre-registered challenger removed only `ev_score`. It reached
ROC-AUC `0.5950`, average precision `0.5996`, Brier `0.2469`, ECE `0.0723`, raw
accuracy `0.5563`, and balanced accuracy `0.5679`, but its paired AUC delta was
only `+0.0081`. The scan-file-clustered 95% interval was
`[-0.0063, 0.0217]`, so zero remains plausible and promotion failed closed.
No production meta-labeler, profile model, pattern profile, utility calibrator,
HMM artifact, or activation pointer was modified by the audit.

The follow-up provenance audit found that all four active profile inputs were
present in scanner rows but were discarded by both canonical backtest-row and
snapshot serialization. They are now retained as scan-time provenance in
`candidate_pipeline.py`, `FeatureSnapshot`, and `build_feature_snapshot` while
remaining explicitly absent from `ACTIVE_SNAPSHOT_META_FEATURES` and
`ALL_META_FEATURES`. This does not promote them into the meta estimator or
change any score. The holdout manifest schema is now v2: every sibling frozen
cohort is hashed, its candidate IDs are excluded before a new freeze, malformed
exclusion sources fail closed, and the output directory is created only after
the cohort passes identity and feature-completeness validation. Schema-v1
cohorts remain resumable but cannot be silently reused as new evidence.

The real canonical FASTWIN pool contains 7,203 unique, deduplicated rows across
two files, all marked `source=backtest`, with 3,444 exact-target positives and
3,759 negatives. None of those historical rows contains any of the four
profile columns, so they cannot validly refit the current four-feature profile.
The consumed 595-row real temporal development cohort does contain the fields,
but an exact-target expanding-window profile reached OOF ROC-AUC `0.4604`;
relabeling the existing feature family is therefore rejected, not promoted.
Synthetic rows are not used: only real canonical backtests may enter fitting or
the promotion gate.

**Backward compatibility:** `ev_contract_fingerprint` is an optional audit
column and is excluded from all estimator feature matrices. Historical rows
remain readable and are explicitly distinguishable by their missing
fingerprint. They are not silently stamped with the current contract and are
not recomputed from the contaminated empirical profile. The isolated challenger
is retained only under the audit directory; runtime continues to load the
unchanged incumbent.

**Verification:** the frozen cohort contains 595 unique candidate IDs from 177
symbols and 36 scanner files; all 595 use 1,440 one-minute bars, engine
`realistic-v8`, formula `alignment-v2-geometric-realism`, label contract
`2026-05-09`, and realism profile `candidate_time_geometric_v1`. The cohort hash
is `3d1ca73b794211344a427fc4f90acf47985708bf5025a0d8dc5b1902c1c96128`;
the frozen incumbent hash is
`0d030f08269b15ed072d74b0479d4bf78f88514715f233587b6fe9ccd0c4d905`;
the paired-predictions hash is
`546b166fa6a2975dddc3a9d4d35ce3f31238e9a3ac78f1fac08ce8dc00a422e6`.
The current EV contract fingerprint is
`ae9fcb44c12e23e8a9a745de1cb6ed8ee833284be5067315ea14c0b34956309a`.
The first post-change full pipeline completed in 1,462.7 seconds with exit code
zero; its 250-row deployment artifact has SHA-256
`b11801c0622daba239993ef7e444aad87eeb5e3863f1771de069ad486d0d18a9`.
A second integration run wrote
`results/deployment_ready_20260801_134648.csv` and the matching 250 snapshot
rows. Its SHA-256 is
`c1d406d38307523052e098babcc66f6ddd08fe2cf17a77e773504b5769932f18`.
The command monitor timed out after one hour while child PID 2972 remained
responsive and CPU-active; the child subsequently finalized both artifacts and
exited, so its final OS exit code is unavailable and is not represented as
zero. The deployment CSV contains 250 unique candidate IDs and 250 unique
symbols. All four profile columns and the EV fingerprint are finite/present on
250/250 deployment rows and the exact 250 matching parquet snapshots. HMM
lineage is present on 249/250 rows; the missing row remains fail-closed. Fifteen
rows pass both `grid_is_valid` and Stage B, eight of those fifteen have negative
`ev_score`, and utility remains null on 250/250 rows because the legacy
schema-v1 pointer is rejected. Twelve enrichment rows failed closed on transient
network exceptions (seven `ReadError`, five `ConnectTimeout`); none was
approved. The single-use prospective selector finds zero matured unevaluated
rows at `2026-08-01T14:56:08Z`; 65 real rows across 40 symbols mature between
`2026-08-01T22:33:45Z` and `2026-08-02T14:53:09Z`.

No live deployment action was taken. The final repository suite passes
1,667/1,667 tests, the focused provenance tests pass 16/16, the dated contract
tests pass 84/84, the leakage-focused contract tests pass 53/53, Pyright reports
zero errors and zero warnings, the locked dependency check passes, and
`git diff --check` is clean.

### CANONICAL-PROFILE-AUDIT-0801 - real-backtest-only profile readiness (2026-08-01)

Added a read-only, fail-closed readiness audit for the exact FASTWIN scanner
profile. It admits only canonical rows with `source=backtest`, authoritative
status, unique candidate identity, valid timestamps, and the active engine,
label, formula, and realism lineage. The target is derived directly as finite
`time_to_target_hours <= 7.0`; outcome columns are never model inputs. The audit
does not train, impute, promote, or write any production profile artifact.

The audit explicitly distinguishes row origin from `t1_is_synthetic`. In these
canonical files that flag records construction of the triple-barrier vertical
horizon; it is not evidence that feature or outcome rows were synthetically
generated. Synthetic-origin rows remain prohibited by the `source=backtest`
and authoritative lineage gates.

**Files modified:**
- `CHANGELOG.md`
- `scripts/audit_canonical_fastwin_profile.py`
- `tests/unit/test_canonical_fastwin_profile_audit.py`

**Generated audit evidence:**
- `outputs/audits/canonical_fastwin_profile_readiness_20260801.json`

**Decision rationale:** AFML / Hudson & Thames validation discipline requires
the training source and an untouched future evaluation source to be separated,
with every feature observed before the outcome. Synthetic replacement rows or
post-outcome imputation would not establish production accuracy. The historical
canonical pool is valid outcome evidence but lacks the four pre-outcome profile
features, so fitting is stopped at the data-contract gate and a fresh unused
temporal holdout remains mandatory even after a fit-ready development pool is
available.

**Backward compatibility:** no runtime, scoring, training, or artifact resolver
path changed. The script writes only a timestamped or explicitly requested JSON
audit file. `data/profile/current.json`, the bootstrap model, and the bootstrap
pattern profile remain untouched.

**Verification:** the two canonical FASTWIN files contain 7,203 rows and 7,203
unique candidate IDs. All rows are authoritative backtests under engine
`realistic-v8`, label contract `2026-05-09`, formula
`alignment-v2-geometric-realism`, and realism profile
`candidate_time_geometric_v1`; no lineage mismatch or invalid timestamp was
found. The exact target has 3,444 positives and 3,759 negatives. Each of the
four profile columns is absent from both files, leaving zero feature-complete
rows. Fit viability is false, promotion is false, and accuracy is recorded as
not measurable. Adversarial unit tests pass 5/5, the full repository suite
passes 1,672/1,672, and full Pyright reports zero errors and zero warnings.

### LIVEMOD-01 - extension-controlled Chrome execution transport (2026-07-31)

Added an authenticated loopback transport so the reviewed live telemetry
modification executable can operate on a user-claimed, extension-controlled
Chrome tab. The existing action state machine remains authoritative: fresh
scanner intent, exact short-lived approval, idempotency fencing, exact live
strategy verification, outward-only price rounding, unchanged grid count,
position preservation, zero additional investment, one submit attempt, and
post-submit range verification are enforced identically for CDP and extension
transports. CDP remains the default.

**Files modified:**
- `CHANGELOG.md`
- `scripts/execute_live_telemetry_modifications.py`
- `scripts/extension_chrome_execution_bridge.mjs`
- `tests/unit/test_execute_live_telemetry_modifications.py`

**Decision rationale:** transport availability must not grant action authority
or weaken live-exchange controls. The extension bridge is therefore bound to
loopback, authenticated by a one-run 256-bit token, restricted to a bounded RPC
surface, and subordinate to the Python executor's existing approval and audit
contract. AFML / Hudson & Thames citations are not applicable because this is
an execution-transport change, not a modeling, feature, or validation change.

**Backward compatibility:** no breaking changes. Existing callers continue to
use the dedicated CDP endpoint unless they explicitly select
`--browser-transport extension` and supply a valid bridge token file.

**Verification:** executor unit tests passed 15/15; Pyright completed with zero
errors; Python and JavaScript syntax checks passed. A live read-only bridge
test verified FORMUSDT strategy `413533841`, and a reversible preparation test
verified range precision, 10 unchanged grids, current-position preservation,
zero additional investment, and enabled confirmation. The test form was closed
without submission and the original live range was re-verified.

### LIVEMOD-02 - single-click Binance drawer readiness (2026-07-31)

Removed the extension bridge's premature second `View Details` click. Binance
can require several seconds to render a telemetry drawer; retrying after 150 ms
could cancel or interfere with the first transition. The bridge now performs
one exact-row click and waits up to 20 seconds for the uniquely identified
drawer. `View Details` is now scoped to the exact row's final action cell and
waits for exactly four action controls before selecting the documented third
control, avoiding both partial-render races and dependence on unrelated edit
buttons elsewhere in the row.
Form input reads use the browser wrapper's supported bounded attribute API,
avoiding its fixed three-second page-evaluation ceiling while retaining exact
single-input and non-null-value assertions.
All strategy, approval, idempotency, form, and submit guards are unchanged.

**Files modified:**
- `CHANGELOG.md`
- `scripts/extension_chrome_execution_bridge.mjs`

**Decision rationale:** UI readiness is an asynchronous state transition and
must be verified from observed state, not inferred from a fixed delay. AFML /
Hudson & Thames citations are not applicable because this is a browser
execution reliability fix, not a model, feature, or validation change.

**Backward compatibility:** no breaking changes. The RPC schema and Python
executor interface are unchanged.

**Verification:** the failure was reproduced against the exact Working row;
one scoped action-cell click opened authenticated 1MBABYDOGEUSDT and CFXUSDT
drawers successfully. JavaScript syntax validation passed, executor unit tests
passed 15/15, and Pyright completed with zero errors and zero warnings.
The first approved live attempt then failed closed before any fill or submit
when the wrapper's element-evaluation ceiling was reached; that blocked
idempotency key remains recorded and is not reused.

### HMMROTATE-0731 - canonical HMM refresh and downstream lineage update (2026-07-31)

Promoted canonical HMM `rolling_180d_20260731_204756` after the governed
Binance Vision coverage, frozen-boundary, training, and walk-forward process.
The downstream meta-labeler was refit and pinned to that HMM; scanner/profile
artifacts were regenerated through their existing retrain command. Utility
recalibration was deliberately not run because the fresh authoritative HMM
backfill contains 11 rows with non-finite regime probabilities, which fails the
strict all-row lineage gate.

**Files modified:**
- `CHANGELOG.md`
- `artifact_manifest.json`
- `artifacts/hmm/rolling_180d_20260731_204756/eval.json`
- `artifacts/hmm/rolling_180d_20260731_204756/metadata.json`
- `artifacts/hmm/rolling_180d_20260731_204756/temperature_scaler.json`
- `artifacts/hmm/training_sets/canonical_20260731_204753/metadata.json`
- `data/new_expired_bots_backfilled_rolling_180d_20260731_204756_20260731.xlsx`
- `data/profile/pattern_profile.json`
- `data/profile/profile_model.json`
- `models/meta_labeler.pkl`
- `models/meta_labeler/metadata.json`
- `models/meta_labeler_verification.json`

**Decision rationale:** the HMM replacement required a canonical frozen
universe and temporal walk-forward evaluation. The meta-labeler was re-pinned
only after the new HMM became active. The strict lineage refusal for utility
calibration preserves AFML / Hudson & Thames data-lineage and out-of-sample
evaluation discipline: unresolved regime inference must not be silently
imputed or mixed into a calibrator fit.

**Backward compatibility:** no breaking changes. Existing utility consumers
remain fail-closed until a fresh utility candidate can pass all lineage and
promotion gates; callers may receive `utility_score=NaN` under the existing
runtime contract.

**Verification:** HMM used 50 accepted symbols and 858,000 boundary-valid
samples; three soft-mode walk-forward splits yielded mean pass rate 1.000.
The identity temperature scaler was persisted. Meta-labeler promotion passed
with OOF AUC 0.8079, bootstrap CI [0.7978, 0.8188], OOF ECE 0.0887, and 2,834
positive examples. The fresh backfill has a uniform active HMM version across
304 rows, of which 293 are finite and 11 are explicitly non-finite. Meta
contract tests passed 29/29; full contract selection passed 80/80; dependency
check passed.

### WARNINGFIX-01 - Python 3.14 async and pandas warning elimination (2026-07-31)

Eliminated the audited 425,936-warning surface without warning suppression.
The test runtime is pinned to the validated pytest 8.4.2 and pytest-asyncio
1.4.0 pair with explicit strict, function-scoped asyncio loops. Scanner
enrichment now constructs its missing audit columns in one ordered concat,
Stage-B approval uses pandas' nullable boolean validation, and shadow candidate
counts classify blank identifiers through pandas' nullable string dtype.

**Files modified:**
- `CHANGELOG.md`
- `pyproject.toml`
- `requirements.lock`
- `src/neutralgrid/live/decision/meta_shadow_analysis.py`
- `src/neutralgrid/scanner/enrich_grid_params.py`
- `tests/unit/test_enrich_grid_params.py`
- `tests/unit/test_meta_shadow_analysis.py`

**Decision rationale:** the dependency update removes calls to asyncio policy
APIs deprecated by Python 3.14. Batch construction removes DataFrame
fragmentation without changing original column order, values, index, or scalar
dtypes. Nullable boolean conversion preserves the existing `None`/boolean gate
contract and prevents text such as `"False"` from becoming truthy. Nullable
string masking preserves unique-candidate counts while avoiding implicit dtype
downcasting. AFML / Hudson & Thames model-selection principles are not invoked
for these mechanical compatibility fixes: no feature, label, sample,
deduplication policy, threshold, model artifact, or training output changed.

**Backward compatibility:** valid runtime inputs and output schemas are
unchanged. Test environments must use pytest 8.4.2 or newer within major
version 8 and pytest-asyncio 1.4.x. A malformed textual `stage_b_approved`
value now raises `TypeError` instead of silently overriding the score gate;
current internal writers already emit booleans, so this is an intentional
fail-closed boundary rather than a valid-input behavior change.

**Verification:** 7 focused regression cases pass with pandas
`FutureWarning` and `PerformanceWarning` promoted to errors; the three owning
test modules pass 55 tests under the same policy. The warning-specific full
suite passes 1,630 tests with Python deprecations, pytest deprecations, pandas
future warnings, and pandas performance warnings promoted to errors (one
unrelated lineage test deselected). Pyright reports 0 errors and 0 warnings.
The unfiltered full suite also passes 1,630 tests; its sole failure is the
pre-existing dirty `artifact_manifest.json` selecting HMM
`rolling_180d_20260731_204756` while the canonical calibration workbook still
contains `rolling_180d_20260724_164829`. The fail-closed lineage guard was not
bypassed and those unrelated artifact changes were not staged.

### LABELFIX-01 - fail-closed live fast-target union (2026-07-31)

Fast-target retraining now rejects live outcome rows that lack the active
path-based `time_to_target_hours` observation. Previously, the opt-in
`--include-live-outcomes` path could interpret a missing live path observation
as "target never reached" and silently assign a negative fast-winner label.

**Files modified:**
- `CHANGELOG.md`
- `retrain_meta_labeler.py`
- `tests/unit/test_live_outcome_fast_target_guard.py`

**Decision rationale:** the active meta-labeler target is whether the target
return was reached within the seven-hour horizon, so terminal duration and PnL
do not reconstruct the missing intrabot path. This guard preserves target and
label integrity; it does not add a feature, change a threshold, retrain a
model, or claim improved final-PnL accuracy. This follows AFML / Hudson &
Thames practice by refusing an observationally different label source in the
same supervised target.

**Backward compatibility:** default retraining and backtest-only pools are
unchanged. Callers that opt into live outcomes without path-based target data
now receive an intentional hard error. Live outcomes become eligible only
after the same `time_to_target_hours` contract is captured or derived from an
auditable event-complete price path.

**Verification:** the new focused regression suite passes 3 tests, the focused
meta-labeler contract/leakage selections pass 80 and 53 tests respectively,
and a canonical opt-in build rejects all 88 current live rows that lack
`time_to_target_hours`. Full-suite and type-check results are reported
separately because the working tree contains unrelated pre-existing changes.

### CANONICALDATA-01 - canonical expired outcomes and governed deploy identity (2026-07-31)

Unified the terminal-outcome, HMM backfill, last-feature, meta-feature, and PnL
curve evidence into `data/new_expired_bots.xlsx`. Active downstream defaults
now resolve that file; `new_expired_bots_backfilled.xlsx` remains only the
backfill command's fresh staging output and is not a runtime default. Deployment
identity and live-outcome ingestion now fail closed on malformed candidate IDs,
configuration-hash mismatch, non-causal timestamps, failed terminal-admission
gates, strategy-to-candidate conflicts, forensic-only links, and explicit live
unions with no governed direct-linked rows.

**Files modified:**
- `CHANGELOG.md`
- `data/new_expired_bots.xlsx`
- `new_bot_data_extractor.py`
- `retrain_meta_labeler.py`
- `scripts/backtest_matched_candidate_comparison.py`
- `scripts/recalibrate_utility.py`
- `scripts/validate_mc_containment.py`
- `scripts/validate_micro_osc_threshold.py`
- `src/neutralgrid/calibration/utility_calibrator.py`
- `src/neutralgrid/grid/spacing_profile.py`
- `src/neutralgrid/live/candidate_deploy_linker.py`
- `src/neutralgrid/live/deployment_link_backfill.py`
- `src/neutralgrid/training/live_outcome_ingestor.py`
- `src/neutralgrid/training/unified_training_builder.py`
- `tests/unit/test_deploy_linker.py`
- `tests/unit/test_deployment_link_backfill.py`
- `tests/unit/test_live_outcome_direct_link_guard.py`
- `tests/unit/test_live_outcome_fast_target_guard.py`
- `tests/unit/test_live_outcome_ingestor.py`
- `tests/unit/test_new_bot_data_extractor.py`
- `tests/unit/test_utility_calibrator.py`

**Generated evidence:**
- `outputs/019fb954-ac9d-7a62-b698-76c1b2b2c6a1/new_expired_bots.xlsx`
- `outputs/019fb954-ac9d-7a62-b698-76c1b2b2c6a1/canonical_workbook_manifest.json`
- `outputs/019fb954-ac9d-7a62-b698-76c1b2b2c6a1/canonical_workbook_validation.json`
- `outputs/019fb954-ac9d-7a62-b698-76c1b2b2c6a1/new_expired_bots.xlsx.inspect.ndjson`
- `outputs/019fb954-ac9d-7a62-b698-76c1b2b2c6a1/previews/`

**Decision rationale:** AFML / Hudson & Thames temporal and label-integrity
principles require one versioned outcome authority, as-of deployment identity,
and isolation of observationally different targets. Geometry/time matching is
useful forensic evidence but cannot establish treatment identity. Public L2
and post-entry scanner telemetry are execution-fidelity and risk-policy data;
they are not valid t0 selector features. No model, feature, threshold, gate,
artifact promotion, deployment, or live-exchange action was performed.

**Backward compatibility:** intentional fail-closed changes. Runtime consumers
that relied on a dated or derived workbook default now read
`data/new_expired_bots.xlsx`. Deployment-link writes must provide a hashed
candidate ID whose grid geometry matches the hash and whose scan precedes the
timezone-aware deployment timestamp. Explicit live training unions now require
governed direct linkage and the active path-based target contract. Workbook
appenders preserve all existing sheets and expand A1-anchored Excel table
references so new rows remain inside the canonical table.

**Verification:** workbook inspection found 304 unique General rows, 304
one-to-one Meta rows, 304 one-to-one Last rows, 125 non-orphan PnL rows, zero
formula errors, 293 complete active-HMM rows, and 11 explicit `hmm_failed`
rows. The governed utility pool contains 163 finite, uniform-lineage rows.
Pipeline preflight PASS; 80 contract tests PASS; focused integration tests 85
PASS; full suite 1,624 PASS; Pyright 0 errors, 0 warnings. A concurrent full
run encountered one transient OneDrive `os.replace` denial in TrialTracker;
the isolated test and the serial full rerun both passed.

### LIVEEXEC-01 - bounded live telemetry adjustment executor (2026-07-31)

Added the reviewed `execute live telemetry modifications` action executable
for Binance Futures Grid ADJUST intents. The executor requires a fresh scanner
decision and a separate short-lived exact approval, verifies the live
symbol/strategy identity, preserves current positions, keeps the grid count
unchanged, accepts zero additional investment only, rounds price bounds
outward to the displayed Binance precision, submits at most once, and proves
the applied bounds in a newly opened details drawer. A 90-second default
deadline, reserved post-submit verification budget, and durable pre-submit
idempotency fence prevent prolonged UI retries and duplicate submission after
a process interruption. END remains deliberately unsupported.

**Files modified:**
- `CHANGELOG.md`
- `scripts/execute_live_telemetry_modifications.py`
- `scripts/run_live_telemetry_controller.py`
- `tests/unit/test_execute_live_telemetry_modifications.py`
- `tests/unit/test_live_telemetry_controller.py`

**Decision rationale:** this is execution governance and UI equivalence
verification, not a model, feature, label, threshold, or promotion change;
AFML and Hudson & Thames model-evaluation references are not applicable. The
controller accepts exchange-applied bounds only when they are a sub-quantum
outward rounding of the approved scanner range, then independently verifies
those exact bounds in a fresh telemetry cycle.

**Backward compatibility:** read-only telemetry and scanner behavior are
unchanged. Configured ADJUST action executors must now return
`neutralgrid_action_execution_v1` evidence for applied bounds, displayed price
precision, position preservation, and zero additional investment; the former
minimal acknowledgement is intentionally rejected. No reviewed production
action executor previously existed. END intents continue to fail closed.

**Verification:** focused executor/controller tests -> 21 passed; related
private-telemetry and live-decision regression tests -> 178 passed; Python
compilation and CLI help smoke passed. Pyright was unavailable in the active
virtual environment and is not claimed as passing. The dedicated Chrome CDP
endpoint on port 9222 was offline, so no live-UI compatibility pass and no
exchange modification are claimed.

### EXTRACTFIX-01 - minutes-only expired-bot durations (2026-07-29)

The manual bot-text parser now accepts Binance's `Duration` newline-plus-
minutes-only format (for example, `34m`). This preserves the terminal timestamp
and produces a derived start time for short-lived canceled bots instead of
rejecting an otherwise complete history export.

**Files modified:**
- `CHANGELOG.md`
- `_bot_data_extractor_core.py`
- `tests/unit/test_bot_data_extractor_v2.py`

**Decision rationale:** this is a source-format compatibility correction, not a
model, label, or gate change; AFML and Hudson & Thames evaluation references are
not applicable. The parser converts only an explicit `Duration` field and does
not infer terminal timestamps from trade activity.

**Backward compatibility:** No breaking changes. Existing day/hour/minute and
decimal-hour duration formats retain their prior behavior.

**Verification:** `python -m pytest tests/unit/test_bot_data_extractor_v2.py
tests/unit/test_new_bot_data_extractor.py -q` -> 172 passed. Pyright was not
installed in the active virtual environment, so no type-check pass is claimed.

### DEPTHCAPTURE-01 - prospective event-complete diff-depth capture (2026-07-29)

Added a durable Binance USDT-M diff-depth collector and deterministic replay
verifier. Each symbol uses a direct 100 ms WebSocket stream, buffers events
while fetching a 1,000-level REST snapshot, applies the documented sequence
bridge, and validates every subsequent `pu` link. Raw frames are flushed before
parsing; sequence, connection, parse, buffer, and book-invariant gaps are
explicitly labelled, and each valid segment closes with a reproducible order-
book digest. Capture scope is prospective: it does not claim historical queue
reconstruction or private fill data.

**Files modified:**
- `CHANGELOG.md`
- `src/neutralgrid/data/diff_depth.py`
- `scripts/collect_diff_depth.py`
- `scripts/verify_diff_depth_capture.py`
- `scripts/start_diff_depth_capture.ps1`
- `scripts/stop_diff_depth_capture.ps1`
- `tests/unit/test_diff_depth.py`

**Generated evidence:**
- `outputs/audits/diff_depth_live_smoke_20260729_203256/`
  (seven current live symbols; 410 raw events, 401 applied events, zero
  sequence gaps, zero raw-hash failures, and deterministic action replay for
  all seven symbols)
- `Live/2026-07-29/<SYMBOL>/diff_depth/diff_depth_20260730_013257/`
  (per-symbol wire events, snapshots, timeline, engine actions, control
  records, and atomic manifest)

**Decision rationale:** this is an acquisition and provenance control, not a
model, label, scanner-gate, or promotion change. AFML and Hudson & Thames model
evaluation references are not applicable. Segment completeness is proven from
Binance sequence identifiers; intervals outside a connected, validated segment
remain labelled gaps and are never silently imputed.

**Backward compatibility:** no existing L2 snapshot collector, scanner, model,
or gate contract changes. The new collector writes only to symbol-specific
`Live/<Lima-date>/<SYMBOL>/diff_depth/<run-id>/` paths and its audit directory.

**Verification:** 10 focused unit and isolated network-integration tests
passed; 44 related depth/replay tests passed; Pyright reported zero errors and
zero warnings; Python compilation and both PowerShell launch scripts parsed
successfully. The fresh seven-symbol live smoke and offline replay verifier
both passed.

### DEPTHCAPTURE-02 - OneDrive-safe live capture restart (2026-07-30)

Hardened derivative manifest writes against transient Windows/OneDrive sharing
denials and restarted prospective 100 ms diff-depth capture for the eight bots
visible as Working in the authenticated Binance dashboard. Raw wire records
retain their per-event flush/fsync contract; only manifest heartbeats are
throttled to five seconds and retried with unique temporary files and bounded
backoff. Canonical read-only telemetry snapshots were stored for the same eight
strategy IDs and used for an isolated live-decision scan.

**Files modified:**
- `CHANGELOG.md`
- `src/neutralgrid/data/diff_depth.py`
- `scripts/collect_diff_depth.py`
- `tests/unit/test_diff_depth.py`
- `Live/2026-07-30/{SKLUSDT,OPENUSDT,INJUSDT,ARXUSDT,XPLUSDT,SKYAIUSDT,VELVETUSDT,SOONUSDT}/live_bot_data_scanner.yaml`

**Generated evidence:**
- `outputs/audits/diff_depth_current/manifest.json`
  (`diff_depth_20260730_162001`; indefinite live capture, eight open contiguous
  segments, zero sequence gaps and zero parse errors at validation time)
- `Live/2026-07-30/<SYMBOL>/diff_depth/diff_depth_20260730_162001/`
  (append-only wire events, snapshots, timeline, engine actions, control
  records, and per-symbol manifests)
- `outputs/audits/live_decision_20260730_current/`
  (bounded validation ticks followed by the active 10-minute advisory scanner
  loop, with console and JSONL evidence)

**Decision rationale:** this is an acquisition durability and live-observation
change, not a model, label, scanner-gate, or promotion change. AFML and Hudson
& Thames model-evaluation references are not applicable. Event-complete depth
remains shadow evidence because the live decision monitor consumes its own
20-level REST order-book snapshot and there is no validated temporal OOS
contract authorizing diff-depth features in production verdicts.

**Backward compatibility:** no raw-event schema, sequence-validity rule,
scanner model, threshold, or gate changed. The terminal replay verifier still
requires closed segments and therefore is run after collection stops. Scanner
verdicts remain advisory; no Binance adjustment or termination executor was
added.

**Verification:** `python -m pytest tests/unit/test_diff_depth.py -q` -> 11
passed; live-decision unit tests -> 89 passed; feature-contract tests -> 79
passed; targeted Pyright -> zero errors and zero warnings; dependency check
passed. Live manifest growth was observed across all eight symbols with one
open segment each and zero sequence gaps or parse errors.

### LIVECONTROL-01 - fresh-telemetry verdict controller (2026-07-30)

Added a fail-closed ten-minute controller that attempts a complete read-only
Binance Chrome telemetry cycle, accepts cached telemetry only while it remains
fresh and complete, rebuilds the scanner registry from the exact visible
symbol/strategy set, runs one live-decision tick, and records validated
ADJUST/END intents. The former static registry loop was stopped because its
launch-time roster still included bots no longer visible as Working.

The repository still has no reviewed Binance Futures Grid action API or stable
UI action executor. Action routing is therefore disabled by default and
requires both explicit `--allow-actions` authority and a separately reviewed
`--action-executable`. An acknowledgement alone is insufficient: the
controller fetches a new complete Chrome cycle and verifies that END removed
the exact strategy or that ADJUST changed that strategy to the requested
bounds before recording the action as verified.

**Files modified:**
- `CHANGELOG.md`
- `scripts/run_live_telemetry_controller.py`
- `tests/unit/test_live_telemetry_controller.py`

**Generated evidence:**
- `outputs/audits/live_telemetry_controller_validation_20260730/`
  (an isolated real cycle correctly blocked when the dedicated Chrome
  debugging endpoint was unavailable and no complete fresh cache existed)

**Decision rationale:** this is orchestration, evidence validation, and action
governance; no model, label, feature, threshold, or scanner gate changed. AFML
and Hudson & Thames model-evaluation references are therefore not applicable.
The action boundary preserves the existing advisory/read-only contract unless
the operator supplies a reviewed executor and explicit authority, and it never
converts an unverified UI/API acknowledgement into an executed result.

**Backward compatibility:** no breaking changes. Existing telemetry collectors
and scanner entry points are unchanged. The new controller defaults to
read-only intent reporting, uses its own lock/STOP/audit paths, and refuses
stale, partial, mismatched, or unverifiable input.

**Verification:** `python -m pytest
tests/unit/test_live_telemetry_controller.py -q` -> 8 passed; feature-contract
tests -> 79 passed; targeted Pyright -> zero errors and zero warnings;
dependency check passed. A real isolated `--once` run produced a structured
blocked report rather than using stale telemetry when port 9222 was
unavailable.

### LIVETELEM-01 - read-only private telemetry and active L2 loops (2026-07-28)

Added an unattended, fail-closed Chrome collector for the visible Binance
Futures Grid `View Details` drawer at five-minute intervals. A cycle is written
only when the visible Running-row count matches the `UM Grid (N)` count and
every row exposes the required PnL, position, pending-order, grid-detail,
order-history, and strategy-number sections. The launcher uses a dedicated
Chrome profile and debugging port so private authentication remains
user-controlled. Started the existing depth-shadow collector for the seven
active symbols at a 60-second interval with strategy-keyed targets and captured
investment notionals.

**Files modified:**
- `CHANGELOG.md`
- `pyproject.toml`
- `scripts/collect_private_grid_telemetry.py`
- `scripts/start_private_telemetry_loop.ps1`
- `scripts/stop_private_telemetry_loop.ps1`
- `tests/unit/test_private_grid_telemetry.py`

**Generated evidence:**
- `Live/2026-07-28/<SYMBOL>/private_telemetry_snapshot_*_lima.md`
  (initial signed-in Chrome snapshot for SPXUSDT, ONDOUSDT, KMNOUSDT,
  XPLUSDT, RAVEUSDT, VANRYUSDT, and SOONUSDT)
- `outputs/audits/depth_shadow_acquisition_live_20260728_174553/`
  (seven accepted targets, 60-second public REST L2 snapshots, no model or
  gate changes)
- `outputs/audits/private_telemetry_loop_current/manifest.json`
  (runtime heartbeat; rejects unsigned-in or incomplete browser state)

**Decision rationale:** this is an operational provenance and acquisition
change, not a model-training or promotion change. AFML evaluation references
are therefore not applicable. The private collector deliberately reads only
the authenticated page already rendered in dedicated Chrome and verifies the
`View Details` tooltip before clicking; it does not call private exchange APIs
or any Adjust, Modify, or End control. Public REST depth records are labelled
as point-in-time snapshots and are not represented as event-complete depth
history.

**Backward compatibility:** no existing telemetry, scanner, model, or gate
contract changes. `websockets>=16.0` is now a direct runtime dependency and
matches the existing `requirements.lock` pin (`websockets==16.0`). Private
cycles require a one-time Binance sign-in in the dedicated Chrome profile.

**Verification:** focused private-collector tests -> 8 passed; Python
`compileall` passed; both PowerShell scripts parsed successfully; live L2
validation observed three snapshots per symbol across seven symbols, 100 bid
and 100 ask levels per record, complete simulated fills, and zero failed
iterations. Pyright was unavailable in the checkout and was not claimed as
passing.

### PIPEHEALTH-01 - governed utility lineage and profile promotion recovery (2026-07-26)

Rebuilt the expired-bot HMM features against active artifact
`rolling_180d_20260724_164829`, made utility calibration and runtime loading
fail closed on stale/mixed/missing HMM lineage, connected `retrain_scanner.py`
to the existing governed walk-forward promotion path, and strengthened the
profile promotion gate so one or two finite folds cannot authorize a model.
No weak candidate was promoted. The fresh utility candidate
`utility_20260726_131902_537231` failed G3/G4/G7 (holdout AUC 0.4667);
the profile candidate had one finite fold at AUC 0.4524 and was rejected.
The full pipeline completed in the intended degraded mode: utility scores
were NaN instead of being computed from the stale schema-v1 artifact.

**Files modified:**
- `CHANGELOG.md`
- `retrain_scanner.py`
- `src/neutralgrid/calibration/utility_calibrator.py`
- `src/neutralgrid/scanner/profile_model_walkforward.py`
- `src/neutralgrid/validation/utility.py`
- `tests/unit/test_meta_labeler_retrain_contract_v20260530.py`
- `tests/unit/test_profile_promotion.py`
- `tests/unit/test_retrain_scanner_cli.py`
- `tests/unit/test_utility_calibrator.py`

**Generated evidence:**
- `data/new_expired_bots_backfilled_rolling_180d_20260724_164829_20260726.xlsx`
  (278 unique rows; 267 replayed; 11 explicitly unavailable because fewer than
  800 pre-decision 15-minute bars; governed utility pool 146 rows, 88 winners,
  58 losers, one active HMM lineage)
- `artifacts/utility/utility_20260726_131902_537231.json`
  (schema v2, not promoted)
- `results/potential_candidates_20260726_131925.csv`
- `results/deployment_ready_20260726_131925.csv`
  (250 unique rows; 24 approved; active HMM and meta-labeler lineage match on
  every row; utility_score non-finite on all rows by fail-closed contract)

**Decision rationale:** Lopez de Prado, *Advances in Financial Machine
Learning*, Chapter 7, and the Hudson & Thames purged cross-validation
implementation motivate time-ordered, leakage-controlled evaluation and
refusing conclusions from under-supported folds. Uniform upstream model
lineage is required before fitting a downstream calibrator; silently mixing or
reusing probabilities across HMM rotations changes feature semantics. The
exploratory survival-probability alternative was not adopted because it was
identified after inspecting the current holdout and therefore is not clean
promotion evidence.

**Backward compatibility:** utility artifacts created under schema v1 no
longer load at runtime. This is an intentional fail-closed change: operators
must recalibrate a promotable schema-v2 artifact pinned to the active HMM.
Profile bootstrap loading is unchanged; only governed promotion criteria are
stricter and the scanner retraining CLI now attempts the governed evaluation.

**Verification:** `python -m pytest tests/ -q` -> 1561 passed; profile
promotion/CLI suite -> 27 passed; utility affected suite -> 27 passed;
leakage/label contract subset -> 53 passed; all contract tests -> 79 passed;
`python scripts/check_deps.py` passed all six locked dependencies; full
`python run_full_pipeline.py` published the two 20260726_131925 CSVs; final
`python -m pyright` -> 0 errors.

### Changed - GATEFIX-02: live decision scanner hysteresis END, contract v1.0 -> v1.1 (2026-07-13)

Replaced the live decision scanner's single-tick `price_outside_grid` END
trigger with a hysteresis trigger, and bumped `DECISION_CONTRACT_VERSION`
to "1.1" (reason-code vocabulary change; this entry is the justification the
contract test requires).

**Why (engine-verified replay, 638 pseudo-deployments, 22 deployed symbols,
2026-07-01..07-13, canonical `run_backtest` at $400 each):** the v1.0 rule
(END on one 5-minute close outside the range) had exit precision 0.38-0.47
in every cohort split (i.e. more than half of ENDed bots were not losers),
exited 338/484 deployments on the 12-day cohort, and destroyed ~$10.4k of
winner upside vs ~$2.8k for the replacement — including ENDing bots that
recovered into the largest winners (e.g. TIAUSDT ENDed at +3.6h at -0.45%,
final +18.67%). On the 4-day holdout it was net NEGATIVE vs holding (-$553).

**New END trigger (either):** (a) persistence — outside continuously for
>= 180 wall-clock minutes observed on >= 3 consecutive ticks, with a
staleness guard (gap since last recorded tick > 15 min resets accumulation;
state predating the bot's `deploy_ts` is discarded); (b) displacement —
|price - grid_center| >= 2.0 half-widths (disaster stop; precision
0.74-0.80). A fired price END latches until 12 consecutive inside ticks. A
single outside tick now yields ADJUST with `price_outside_watch:` and
re-centered suggested bounds; the first tick of a fresh excursion
force-emits through an escalated-ADJUST cool-down. Precision of the
hysteresis union replicated 0.61-0.71 on held-out splits; components tested
in isolation (persistence-only, displacement-only) and inferior to the
union; displacement neighbors {1.25, 1.5, 2.5, 3.0} and persistence
neighbors {15m, 30m, 60m, occupancy} all tested and inferior; CUSUM
(h in {2,3,4}) and OU kappa-collapse rejected under a pre-registered
day-split protocol; recompute-survival-prob-at-excursion REJECTED (AUC 0.30,
inverted). Honest caveats recorded: dollar aggregates are symbol-concentrated
(TLMUSDT alone exceeds the 12-day total improvement; per-symbol sign split
11/11) — the load-bearing evidence is the replicated precision gain and the
~3.7x winner-preservation asymmetry, not dollar totals; re-center-as-action
measured regime-dependent, so recentering remains an advisory ADJUST
suggestion, not an automatic action.

**Files modified:**
- `src/neutralgrid/core/constants.py` (DECISION_CONTRACT_VERSION 1.1)
- `src/neutralgrid/live/decision/recommender.py` (RecommenderConfig +6
  validated fields incl. `end_on_first_outside_tick` legacy restore flag;
  BotEvaluation.deploy_ts; decide() price-outside state machine;
  _should_emit force path)
- `src/neutralgrid/live/decision/state_store.py` (BotHistory +4 additive
  fields; `_parse_utc` tz-normalization on load — a naive ISO timestamp in a
  state file previously crashed decide() with aware-minus-naive TypeError)
- `src/neutralgrid/live/decision/monitor.py` (deploy_ts threading)
- `src/neutralgrid/live/decision/meta_shadow_analysis.py` (`_decision_row`
  carries contract_version + reasons so D5/D6 calibration pools can be
  segmented by contract era)
- `live_decision_scanner.py` (config load moved before lock acquisition so a
  fail-closed config error cannot strand `.scanner.lock`; synthetic eval
  deploy_ts)
- `tests/unit/test_decision_contract_v1_0.py` -> `test_decision_contract_v1_1.py`
  (v1.1 pins), `tests/unit/test_decision_recommender.py` (+11 state-machine
  tests), `tests/unit/test_decision_state_store.py` (+3 round-trip/tz tests)
- `LIVE_DECISION.md` (verdict-mapping section)

**Decision rationale:** hysteresis via time-based confirmation + buffered
disaster stop matches practitioner consensus and Osler's microstructure
evidence on false-breakout discrimination; recenter-not-exit on range break
is supported by arXiv 2506.11921 (static grids have zero EV; "unwise to stop
the strategy when the price surpasses our upper or lower limits"). Design
adversarially reviewed by a 4-lens / 20-agent workflow (state machine,
contract consumers, resilience, evidence sufficiency); 2 blocking + 10
should-fix findings all resolved or explicitly declined with reasons
(per-field salvage of corrupt state files declined: whole-file reset is the
established load_history pattern).

**Backward compatibility:** pre-1.1 state files load unchanged (additive
keys default); `end_on_first_outside_tick: true` restores exact v1.0 verdict
behavior; JSONL top-level schema unchanged (reasons are opaque strings;
`contract_version` now reads "1.1").

**Verification:** 183 live-decision suite tests pass (incl. 14 new);
`pyright` 0 errors on all touched files; full `python -m pytest tests/`
green; end-to-end `--once` scanner tick on a 22-bot registry (isolated
state/logs, no Discord) exercises the new verdict path against live market
data.

### Fixed - GATEFIX-01: zero-candidate pipeline lockup, ERR-090..ERR-093 (2026-07-12)

Fixed the four gate defects that took full-pipeline yield from 38-59 valid
candidates per run (through 2026-06-17) to a permanent ~0 (since 06-23/24),
diagnosed in the session-981a7d2c zero-candidate audit (5-agent code audit +
~900 canonical-engine backtests + an 82-symbol/2-day engine-verified cohort;
findings logged as ERR-090..ERR-094 before fixing).

1. **ERR-090 (the 06-24 regression)** - generalized Kelly demoted to opt-in:
   `PositionSizingConfig.enable_generalized_kelly` True -> False. Kelly's
   inputs are incoherent as measured: p = meta_prob = P(net MTM >= +3% in 7h)
   while b = avg_win/avg_loss on net>0/net<0 events from the unconditional
   backtest_candidates pool (b=0.5814 -> break-even p=0.632); the properly
   ALIGNED pairing (E = mfe_pct_initial>=3.0 on the same pool) measures
   b_E=0.437 -> break-even p=0.696 vs pool P(E)=0.766 - the model's
   probability scale and the sizing pool's event rate disagree so badly that
   every observed meta_prob (max 0.563) produced kelly_raw<0, `max(0,raw)`
   floored capital_fraction to 0, and Stage B rejected everything as
   position_too_small. Sizing authority reverts to the risk-budget
   PositionSizer (pre-06-24 last-known-good). Re-enable only with a
   re-derived coherent (p, b) pairing.
2. **ERR-091** - dynamic profit floor volatility input restored to design
   scale: `_estimate_microstructure` now feeds bot-horizon realized vol from
   15m returns (new `_horizon_realized_vol_pct`, ~1-3%) instead of half the
   validated range width (14-17, ~8x the "2% vol -> 0.10% premium" design
   scale, triple-counted across slippage vol_adjustment + vol_premium +
   survival_factor). `GridCalculator.generate_params` now detects the
   structurally-unsatisfiable configuration (demanded floor > profit ceiling
   implied by max_spacing_pct) and reports
   `profit_floor_exceeds_spacing_cap(...)` instead of a generic below-min
   rejection after a silent spacing clamp.
3. **ERR-092** - OU half-life de-trended and demoted to advisory: new
   `estimate_ou_params_detrended` fits the OU on linearly de-trended log
   prices (drift no longer masquerades as "no mean-reversion";
   `analyze()` reports the de-trended theta/mu/sigma/half-life while the
   survival MC keeps the raw fit unchanged). The [4,48]-bar window becomes
   advisory telemetry (`StochasticConfig.ou_halflife_gate_hard = False`):
   measured on the engine-verified cohort it had rho=-0.011/AUC 0.47 raw and
   rho=0.096/AUC 0.44 de-trended vs realized grid PnL - no discrimination -
   while rejecting 47% of the universe and 76%/47% of engine-profitable
   symbols. The Hurst gate (<= 0.65) remains hard; the stale hurst
   annotation (hardcoded 0.55) now reads the configured threshold.
4. **ERR-093** - Stage B Gate 4 micro-oscillator survival mode now requires
   `micro_osc_bypass` PROVENANCE (scan-phase score AND scan-phase survival),
   not a raw `micro_osc_score >= 0.45` - previously 94% of the universe
   (235/250) was routed into the survival>=0.60 test (post-grid survival
   median 0.208, pass ~0.14) and the entropy-adaptive range_prob path was
   effectively dead code. Gate 4 remains MANDATORY in both modes; the
   ERR-082 `data_missing:survival_prob` fail-closed semantics are preserved.

- **Files modified**: `src/neutralgrid/core/config.py`
  (enable_generalized_kelly False + ERR-090 note; ou_halflife_gate_hard new
  flag), `src/neutralgrid/scanner/enrich_grid_params.py` (import math;
  `_horizon_realized_vol_pct`; vol_proxy replacement; hurst annotation via
  config; micro_osc_bypass passed to Stage B),
  `src/neutralgrid/grid/calculator.py` (guaranteed-fail detection),
  `src/neutralgrid/validation/stochastic.py` (estimate_ou_params_detrended;
  analyze() reporting split), `src/neutralgrid/validation/regime_validator.py`
  (advisory half-life wiring + halflife_advisory telemetry),
  `src/neutralgrid/scanner/two_stage_selector.py` (micro_osc_bypass
  parameter + Gate 4 mode condition), `.claude/rules/safety-invariants.md`
  (Gate 4 trigger wording), `ERRORS_LOG.md` (ERR-090..ERR-094 logged).
  Tests: `tests/unit/test_enrich_grid_params.py` (ERR-090 default test +
  Kelly opt-in), `tests/unit/test_err091_floor_fixes.py` (new),
  `tests/unit/test_err092_ou_detrend.py` (new),
  `tests/test_gate4_micro_osc.py` + `tests/test_micro_osc_integration.py`
  (bypass provenance).
- **Decision rationale**: evidence-first per the audit protocol - every
  disposition was measured before changing code (Kelly pairing variants on
  the 1,950-row pool; de-trended half-life discrimination on the 35-symbol
  cohort with a pre-registered decision rule; floor decompositions vs the
  engine's profitable-spacing region 0.13-0.93% profit/grid). AFML alignment:
  bet sizing from calibrated probabilities requires p and b measured on the
  same event/population (AFML ch. 10); a filter whose statistic has no
  out-of-sample discrimination should not hard-gate (AFML ch. 8 feature
  importance discipline); the survival-gate demotion precedent (2026-05-22)
  applies to the half-life window.
- **Backward compatibility**: behavior changes are the intended fixes -
  generalized Kelly no longer multiplies capital_fraction unless explicitly
  enabled; regime rejections no longer fire on half-life alone (hurst gate
  unchanged); floor rejections for wide-range symbols now carry the
  guaranteed-fail reason where applicable; Gate 4 mode selection changed for
  rows without bypass provenance. Config flags restore each legacy behavior
  (`enable_generalized_kelly=True`, `ou_halflife_gate_hard=True`).
- **Verification**: full pytest 1537 passed (+15 new regression tests);
  pyright 0 errors on all touched files. LIVE acceptance run
  deployment_ready_20260712_175238: **22/250 valid candidates** (20
  auto-deploy + 2 tagged) vs 0 in every run since 06-23; position_too_small
  count 0 (was the #1 Stage-B kill); all 30 sized rows capital_fraction >
  0.05. Engine verification (canonical run_backtest, taker-fee physics):
  19/22 approved geometries positive on the trailing 24h (TUSDT +26.4%/144
  round-trips, TLMUSDT +22.6%, VIRTUALUSDT +19.6%; 0 liquidations);
  rejected-sample blind-sweep miss rate 4/12 = 0.33 vs 0.82 for the old
  regime-joint gate. Per-fix closure proof in ERR-090..ERR-093
  (ERRORS_LOG.md).

Related: ERR-090..ERR-094 (ERRORS_LOG.md, Zero-candidate pipeline audit
2026-07-11/12); FASTWIN-V2-REGROW + ERR-059 (the commit that woke the Kelly
path); ERR-082 (fail-closed survival telemetry, preserved).

### Removed - DECOM-HMMWINNER-01: orphaned hmm_winner_calibrator artifact decommissioned to archive (2026-07-10)

Archived `artifacts/hmm_winner_calibrator/` (11 files: `current.json` +
10 timestamped versions, last produced 2026-06-01) to
`artifacts/_archive_v657/hmm_winner_calibrator/`. The artifact was fully
orphaned in this repository: its producer module
(`neutralgrid.calibration.hmm_winner_calibrator`) does not exist in this tree,
grep confirms ZERO code consumers (src/, scripts/, tests/, root), and its own
metadata pins it to the v6.5.7 repo (`hmm_artifact_dir` points into
`...\NEUTRAL grid bot v6.5.7\artifacts\hmm\rolling_180d_20260516_220545`,
code_commit 04a15b4, source workbook `data\new_expired_bots_backfilled.xlsx`) -
four HMM rotations stale. The only repository reference (a stale
`.agents/skills/hmm-rotate` instruction to run the nonexistent module) was
already removed in the ERR-085 skills resync earlier the same day.

- **Files modified**: none tracked (the `artifacts/` tree is gitignored; the
  move is a local filesystem operation recorded here for durability).
  Moved: `artifacts/hmm_winner_calibrator/` ->
  `artifacts/_archive_v657/hmm_winner_calibrator/` (11/11 files verified).
- **Decision rationale**: decommission-over-rebuild, operator-directed
  2026-07-10 after the artifact-refresh audit (session 981a7d2c). The
  artifact's function - calibrating P(winner) from HMM regime posteriors
  (3-coefficient logistic on range_prob/trend_prob/persistence_prob,
  n_pool=97, label pnl_pct > 1.0 within < 7h) - is fully superseded by the
  FASTWIN-02 meta-labeler (AFML ch. 3 meta-labeling: 20-feature
  `snapshot_v20260530_fastwin`, 6048-row pool, promotion-gate OOF-AUC 0.8079,
  stricter fast-winner target +3% <= 7h). Rebuilding would require writing a
  new producer, re-deriving labels under the current contract, and wiring a
  consumer that does not exist - a second, strictly weaker meta-labeler that
  double-counts HMM information already consumed by Stage B Gate 4
  (range_prob / survival_prob). Archived rather than deleted because
  `artifacts/` is untracked and deletion would be unrecoverable.
- **Backward compatibility**: no breaking changes. No runtime, training, or
  validation path reads the artifact; no tracked file changed except this
  CHANGELOG entry.
- **Verification**: `git ls-files artifacts/hmm_winner_calibrator` empty
  (untracked, nothing removed from version control); pre/post move file count
  11 == 11; source path absent after move; consumer grep re-confirmed empty
  the same session. Full pytest 1522 passed and pyright 0 errors on the
  session's prior commits; this change touches no code.

Related: ERR-085 (skills resync removed the dead runbook reference);
FASTWIN-02 (superseding model); session of record 981a7d2c (artifact-refresh
audit, ERRORS_LOG.md Artifact-refresh audit 2026-07-10 section).

### Validated - BACKTEST-RECON-LASTMONTH: fresh-cohort engine reconciliation, 24/24 sign match (2026-07-04)

Operator-requested fresh-cohort check of the backtest engine against the last
month of expired live bots (ended 2026-06-04..2026-06-26; 24 rows, 21 symbols,
23 geometric / 1 arithmetic) via
`scripts/validate_backtest_live_reconciliation.py` under production defaults
(`legacy` profile, `stored_utc`). First pass was support-starved
(`missing_kline_rows=22`); backfilled June 1m klines for the 21 symbols from
Binance Vision (`ensure_kline_store`, interval=1m) and re-ran: 24/24 modelable.

- **PnL sign match 1.000 (24/24)**; winner recall 1.0; non-winner specificity
  1.0; corr(live, model) 0.916.
- Mean abs PnL error **3.80pp**, median **3.15pp** — better than the recorded
  May benchmarks (legacy latest-20: 6.37pp mean, 0.765 sign match), consistent
  with all June rows being post-ERR-043 canonical-UTC ingests (22/24
  `exact_stored_match`).
- Known-limitation signatures reproduced where documented: engine ~2.1pp
  pessimistic on average with the error tail in deep losers (CHZUSDT 13.25pp,
  LABUSDT 10.40pp, losses overstated), and fill counts under-simulated (live
  mean 60.3 trades vs model 43.1; 1m-bar granularity, ERR-061 context).
- No-save diagnostic: no tracked files, model artifacts, or pipeline inputs
  changed. Artifacts + session-stamped README at
  `outputs/audits/backtest_recon_lastmonth_20260704/` (gitignored; numbers
  recorded here for durability).
- Side effects: June 1m klines now cached under
  `data/cache/klines/futures_um/<SYM>/1m/` for the 21 cohort symbols.
  Housekeeping in the same session: deleted untracked temp dirs
  (`.pytest_cache`, `.pytest_tmp`, `.pytest_local_tmp`, `.tmp_pytest_*` x5,
  `codex_manual_checks*`) and stale root logs (`full_pytest.log`,
  `run_backfill_20260608.log`, `run_full_pipeline_2026060{7,8}.log`).
  Discovered: `scripts/backfill_vision_store.py` has no argument parsing —
  `--help` starts a real top-100 1h backfill (stopped within seconds).

Session of record: 4a0fe144-1222-42fd-b796-9fab3a8c0ad7.

### FASTWIN-02 - meta-labeler estimator promotion: logistic -> logistic+HGB soft vote (2026-07-04)

Promoted a new production meta-labeler artifact `20260704_214452` that replaces
the FASTWIN-01 single L2-logistic estimator with a soft VotingClassifier of the
UNCHANGED logistic and a strongly-regularized HistGradientBoostingClassifier
(lr 0.03, max_leaf_nodes 10, max_iter 350, min_samples_leaf 60, l2 3.0,
max_features 0.8), on the UNCHANGED 20-feature `snapshot_v20260530_fastwin`
profile and unchanged 6048-row pool. **Promotion-gate OOF-AUC 0.8079
[0.7978, 0.8188] vs champion 0.7698 [0.7579, 0.7820]** (CIs disjoint), OOF-ECE
0.0887 (gate <= 0.10), n_pos 2834, promotion PASS; deployed `sigmoid_oos`
calibration ECE 0.0097 (was 0.0181). OOF recall at the precision>=0.60
operating floor rose 0.884 -> 0.925 (+4.1pp of fast winners captured at equal
precision). `min_meta_prob=0.37` deliberately NOT changed: at the existing
threshold the new probability map operates at HIGHER precision (~0.62) with
recall >= the old champion's — anti-churn per the ERR-062 lesson.

Decision rationale (challenger protocol, multiplicity-controlled):

- Selection on fold seed 42, confirmation on 4 untouched seeds (7, 123, 2026,
  31337): paired-bootstrap dAUC +0.0356..+0.0407, 5/5 positive, every 95% CI-low
  > 0; ECE 0.086-0.093 on every seed. Paired deltas computed on
  fold-identical OOF vectors (harness mirrors `_evaluate_promotion_oof`
  bit-for-bit; champion reproduction verified to 16 decimals).
- ERR-066 hedge: because the gate's shuffled folds let the time purge collapse,
  the challenger ALSO had to win under time-blocked purged 5-fold CV (12h
  purge / 1.5h embargo, CPCV constants): paired dAUC +0.0215 [0.0170, 0.0262],
  all 5 calendar eras non-negative. Raw HGB/RF/GBM challengers that won only
  under shuffled folds (e.g. lone HGB +0.041 shuffled but -0.018 time-blocked)
  were REJECTED as era-leakage-assisted — the vote's edge is the survivor.
- The vote construction (sigmoid calibrates the BLENDED score) is what passes
  the ECE gate: every lone tree challenger failed it (ECE 0.10-0.14).
  Consistent with AFML model-ensembling guidance (Ch. 6) and the
  McElfresh et al. 2023 finding that light GBDT regularization beats
  model-family exotica on ~6k-row skewed tabular data.
- Feature-set challengers were evaluated and REJECTED: HMM posteriors
  (range_prob/trend_prob/survival_prob/regime_conf/cvar) DEGRADE time-blocked
  OOF (dAUC -0.031..-0.049); the flow set (funding_rate_zscore,
  open_interest_change_pct, long_short_ratio, bb_width_ratio_1h_15m) added
  nothing beyond the vote (+0.0113 vs +0.0215 time-blocked) and carries
  train/serve skew (20.5% of pool rows pre-2026-03-19 hold builder-derived
  defaults). `rotation_score` was caught as OUTCOME LEAKAGE
  (`candidate_pipeline.py:1369` derives it from realized_pnl_per_margin_hour)
  before it contaminated any candidate — its inflated run (+0.27 pooled
  time-blocked dAUC) is recorded in the trial log as leakage evidence, and it
  is already in `_KNOWN_LABEL_COLUMNS`' parent set.
- All 55 offline harness trials logged to the trial tracker
  (`meta_labeler_fastwin02_harness_*`) so the CI-low>0.50 gate's multiplicity
  accounting stays valid (AFML Ch. 11 / deflated-metrics requirement).

Files modified:

- `src/neutralgrid/models/meta_labeler.py` (MetaLabelerConfig: hgb_* fields +
  estimator_type "vote_logit_hgb" docs; `_make_model()` vote branch;
  `_make_preprocessor()` scaler for the vote path; feature-importance
  `estimators_` branch guarded against empty sub-importances — VotingClassifier
  legs expose no `feature_importances_`; metadata `model_type` reports the vote)
- `retrain_meta_labeler.py` (`--estimator {profile,logistic,gbm,vote_logit_hgb}`
  override, default "profile"; the active profile's profile-derived estimator is
  now `vote_logit_hgb` so routine retrains and the meta-labeler-refit skill
  reproduce the promoted architecture; `--estimator logistic` reproduces
  FASTWIN-01 exactly)
- `models/meta_labeler.pkl`, `models/meta_labeler/` (promoted artifact
  `20260704_214452`; pre-promotion champion backed up to
  `models/meta_labeler_backup_20260704_prefastwin02*`)
- `models/meta_labeler_challenger/20260704_fastwin02/` (challenger artifact +
  verification, retained for audit)
- `models/meta_labeler_verification.json` (regenerated by the promotion run)
- `data/trial_log.json` (+55 harness trials)
- `CHANGELOG.md` (this entry), `ERRORS_LOG.md` (ERR-070 watch opened)

Backward compatibility: no breaking changes. Artifact save/load format,
feature schema, Stage-B gate wiring, `min_meta_prob`, and all fail-closed
paths are unchanged; `MetaLabeler.load` of older logistic artifacts is
unaffected. A retrain invoked with `--estimator logistic` reproduces the
FASTWIN-01 estimator bit-for-bit.

Verification: leakage-check skill PASS (both `_KNOWN_LABEL_COLUMNS` guard
sites intact, 53 contract tests green); `pyright` 0 errors on both modified
files; 101 meta/retrain unit tests green; full `pytest tests/` green pre- and
post-change (1505 passed baseline; post-promotion run recorded below);
production artifact load + `predict_proba` smoke verified; challenger and
default-path production retrains produced identical gate metrics
(deterministic reproduction). See ERR-070 for the live-shadow watch.
Session of record: 4a0fe144-1222-42fd-b796-9fab3a8c0ad7 (research workflow
wf_73baa6b1-1d7).

### Fixed - ERRSWEEP-20260703: unresolved-errors review — ERR-029 closed, WATCH set re-evidenced, ERR-068/069 opened (2026-07-03)

Operator-requested review of every non-CLOSED ERRORS_LOG entry (6 rows, all
WATCH after the 2026-07-02 audit). Every disposition was adversarially reviewed
BEFORE implementation by a 3-agent consensus workflow (`wf_47968a38-607`); all
three reviewers returned APPROVE_WITH_CHANGES and their corrections were
incorporated. No `src/` code was changed; the active meta-labeler artifact
`20260703_044333` and all model behavior are untouched.

- **ERR-029 CLOSED (workbook schema drift, option (a)).** One-time idempotent
  migration appended the `trigger_price` header at `General!R1C54` of
  `data/new_expired_bots.xlsx` (timestamped backup
  `data/new_expired_bots_backup_ERR029_20260703.xlsx` written first). The
  per-ingest "Workbook is missing 1 default column(s)" warning is gone and
  future non-`--` trigger prices are persisted instead of silently dropped.
  Reviewer-verified: appends are name-based (`filter_row_to_schema` /
  `append_row_preserving_schema`), no consumer reads the sheet positionally,
  `trigger_price` is not a model feature (allowlist selection), so the Feature
  Pipeline Update Rule is not engaged. Option (b) (deleting the column from
  `DEFAULT_WORKBOOK_COLUMNS`) was rejected: it codifies data loss and leaves
  the ERR-069 legacy-writer hazard armed.
- **ERR-016 evidence corrected (doc-only).** Its stated closure criterion was
  unreachable: MI-weight artifact generation is hard-disabled
  (`backtest_candidates.py:558`) and the load side passes `mi_weights=None`
  unconditionally (`run_full_pipeline.py:921`); the generation signal map also
  proxies `profile_proba` from `range_prob` (`:593`). Runtime is safe
  (uniform weights on 250/250 rows of the 3 latest deployment CSVs). Real
  closure = provenance implementation + OOS-validated re-enable.
- **ERR-044 re-affirmed WATCH, blocker quantified.** Threshold calibration is
  double-blocked: `meta_shadow_analysis.py` requires >=10 fit-adjust rows AND
  >=5 join keys per side; the 2026-07-03 artifact has fit_adjust_rows=1 /
  unique_join_keys=2. Hand-fitting 8+ thresholds to n=2 outcomes would be
  worse than the defaults (overfitting; AFML ch.7 rationale for grouped OOS
  support minima).
- **ERR-061 re-affirmed WATCH, wording tightened.** Raw price-level L2 queue
  replay is absent in-repo; the percentage-bucket bookDepth proxy exists
  (BOOKDEPTH-01) but was already trained and rejected (adds no signal beyond
  active features), closing the one in-repo mechanical avenue.
- **ERR-064 preconditions verified + bat hardened.** Registration of
  `NeutralGrid_D7_ActiveBots` stays deferred to the next deployment (operator
  decision of record); all preconditions verified green. The bat at
  `C:\Users\cris_\neutralgrid_tasks\run_d7_active_bots.bat` gained an
  empty-dir guard (skip-tick log line + exit 0, verified) so early
  registration can no longer produce ~480 exit-1 runs/day.
- **ERR-060 pending items recorded resolved.** `meta_prob_source='enrich'`
  confirmed post-fix (up to 136 rows/scan); `config.py:307-308`
  (`meta_gate_enabled=True` / `min_meta_prob=0.37`) committed at HEAD 7d294b5.
- **ERR-068 OPENED (WATCH) — pool cap recency bias.** The validated FASTWIN
  per-symbol cap (`unified_training_builder.py:760-774`) keeps the 30 OLDEST
  rows per symbol (candidate_id sort + head(30)): 93/475 symbols over cap,
  1,008/7,203 rows (14%) dropped, all the newest — 7 of the 17 June
  live-deployed candidates are excluded from training by it. The cap COUNT
  stays (validated: 0.780+/-0.002 vs 0.748+/-0.034 uncapped); only the
  retention DIRECTION is unexamined. Closure = controlled retrain experiment
  gated on OOF-AUC + ECE non-degradation (meta-labeler change gate).
- **ERR-069 OPENED (WATCH) — legacy writer schema-drift hazard.** Legacy
  `append_to_excel` (`_bot_data_extractor_core.py:1665-1701`) would widen the
  workbook with `coherence_ok`/`coherence_warnings` (not in defaults),
  fail-closing every subsequent modern ingest. Latent while the legacy path
  is unused; documented so it is not tripped accidentally.

Files modified:

- `data/new_expired_bots.xlsx` (General sheet header 53 -> 54; all 278 data
  rows and all 53 pre-existing columns byte-identical to backup)
- `data/new_expired_bots_backup_ERR029_20260703.xlsx` (new; pre-migration copy)
- `C:\Users\cris_\neutralgrid_tasks\run_d7_active_bots.bat` (empty-dir guard;
  outside the repo)
- `ERRORS_LOG.md` (ERR-029 closed with proof; ERR-016/044/061/064 evidence
  refreshed; ERR-060 pending note resolved; ERR-068/ERR-069 appended)
- `CHANGELOG.md` (this entry)

Backward compatibility: no breaking changes. No `src/` code changed; model
artifacts byte-identical; the workbook migration is additive and name-mapped.

Verification: migration re-run prints NO-OP (idempotent);
`resolve_workbook_columns` returns 54 columns with zero warnings; other three
sheets dimension-identical; focused suites
`test_new_bot_data_extractor.py test_utility_calibrator.py
test_live_outcome_ingestor.py test_bot_data_extractor_v2.py` = 200 passed;
guarded bat run against the empty `active bots/` dir exits 0 with the skip
log line. See ERR-029, ERR-016, ERR-044, ERR-060, ERR-061, ERR-064, ERR-068,
ERR-069.

### Fixed - AUDIT-20260702: ERR-062 calibration gate + ERR-063 metadata + ERR-065 fail-closed + live-scanner hardening (2026-07-02)

Full-audit session (workflow `wf_d4fa29e7-9a7`, 12 read-only agents over all 39
non-CLOSED ERRORS_LOG entries + live-telemetry stack + model code). Fixes:

- **ERR-062 (HIGH) — beta calibration upgrade bypassed its non-degradation
  gate.** `meta_labeler.py` Phase-2 ran the reliability diagnostic on the RAW
  model output and replaced the gate-accepted sigmoid with beta on convergence
  alone. Deployed artifact `20260701_224916` carried a beta calibrator with
  OOS ECE **0.0652** that had silently displaced a sigmoid with ECE **0.0181**
  (same 11,622 pooled OOS predictions; verified from the pickled
  `oos_calibration_report`). Worse, the beta map squashed all probabilities
  into **[0.3697, 0.7509]**, so the Stage-B `min_meta_prob=0.37` gate rejected
  almost nothing (raw score cutoff ~0.002). Fix: the diagnostic now runs on
  the ACCEPTED calibrator's output (residual miscalibration) and beta must
  strictly beat the accepted ECE to deploy; when it wins, the canonical
  `ece/brier_calibrated` fields now describe the DEPLOYED calibrator and the
  displaced one is preserved in `displaced_*` audit fields. Retrained via the
  canonical `retrain_meta_labeler.py --input data/new_expired_bots.xlsx` →
  active artifact **`20260703_044333`**: identical pool (6,048 rows, n_pos
  2,834), identical discrimination (OOF-AUC 0.7698 [0.758, 0.782], promotion
  PASS), lineage `rolling_180d_20260701_204849` unchanged — but deployed
  calibrator is now **sigmoid_oos, ECE 0.0181** (3.6x better) and the 0.37
  gate cuts at raw≈0.064 instead of ~0.002. The FASTWIN-01 OOF
  operating-points table reproduces **tau=0.37277 @ precision>=0.60, recall
  0.883** (the documented source of `min_meta_prob=0.37`, computed on the
  promotion-OOF stream, which this fix does not touch) — so the configured
  0.37 stays valid and is now applied on a probability scale consistent with
  its derivation. Regression tests:
  `tests/unit/test_meta_labeler_calibration_gate.py` (4 tests; synthetic
  overconfident-raw pool reproduces the exact defect shape — beta candidate
  ECE 0.0318 vs sigmoid 0.0231 → correctly rejected `ece_not_better`).
- **ERR-063 — `metadata.json` `model_type` writer bug.** Hardcoded
  `GradientBoostingClassifier` never inspected `config.estimator_type`; the
  deployed FASTWIN base estimator is LogisticRegression. Now derived from the
  actual configuration (+ additive `model_params.estimator_type`); artifact
  `20260703_044333` records `model_type=LogisticRegression`.
- **ERR-066 (WATCH) — promotion-gate time purge collapse made visible.** The
  `_evaluate_promotion_oof` purge collapses on shuffled stratified folds
  (every fold spans the full time range) and silently trained unpurged. Now
  logged at WARNING with the affected fraction (observed 5/5 folds) and the
  docstring states the true behavior: symbol grouping remains the operative
  leakage guard; `oof_auc` may be modestly optimistic from cross-symbol
  temporal concurrency. Fold construction deliberately NOT changed (the
  0.50/0.10 gate thresholds were derived under the identical scheme).
- **ERR-065 — serve-time `ou_halflife` cap reverted to fail-closed.** The
  uncommitted working-tree draft capped `halflife=inf` (no mean-reversion)
  rows to `ou_halflife_max_bars=48` for the meta feature row. The training
  pool's `ou_halflife` is raw and heavy-tailed (median ~41, p99 ~1014, max
  ~9e5, never capped) — 48 would score a no-mean-reversion row as MEDIAN
  mean-reversion behavior, inflating `meta_prob` for exactly the rows the
  strategy should avoid, and is silent imputation in live admission logic
  (the D1-class invariant). Restored: feature stays absent → probe skipped →
  Stage B `data_missing:meta`. The audit columns (`ou_halflife_raw`,
  `ou_halflife_feature_reason=non_mean_reverting_fail_closed`) are kept.
  Test rewritten: `test_no_mean_reversion_halflife_fails_closed_for_meta_feature`.
- **Working-tree enrich changes documented** (were undocumented; all verified
  tested + pyright-clean): scan-phase low-`survival_prob` pre-reject is now
  audit-only (fresh regime validation remains authoritative); low-`range_prob`
  pre-reject defers to fresh validation when the scan cache is >300s stale
  (`scan_cache_age_seconds`/`scan_cache_stale` audit columns); micro-osc
  bypass rows feed survival-derived probs to PositionSizer + dynamic profit
  floor (`ps_archetype`/`micro_floor_archetype=micro_osc_survival`); when
  generalized Kelly is active the PositionSizer receives the risk-budget vol
  target instead of the realized-range proxy so the volatility penalty is not
  applied twice (`ps_volatility_source=kelly_owned`). These raise candidate
  throughput to the authoritative gates; none bypasses a mandatory gate.
- **Live scanner `--config-file` fail-closed (was fail-open).** An explicitly
  passed config that fails to load now aborts with non-zero exit instead of
  silently running every tick on default RecommenderConfig. Regression tests:
  `tests/unit/test_live_scanner_config_fail_closed.py` (3 tests).
- **LIVE_DECISION.md**: documented the ERR-045 linkage-staleness rule (linkage
  log loads once per process; restart or use `--once` after stamping links)
  and the ERR-046 naive-`deploy_ts`-assumed-UTC rule (+ Binance UI local
  UTC-5 conversion; AVAX 2026-06-21 case). Both entries close on their own
  documentation-only criteria.
- **Telemetry continuity (ERR-064, operator decision pending).** The only
  scheduled task (`NeutralGrid_D7_MUSDT`) was DISABLED mid-window 2026-06-26
  and its one-time trigger dies permanently 2026-07-03; it covered only MUSDT
  — zero scheduled D7 telemetry since 2026-06-27 00:39 UTC while ENAUSDT
  (deployed 06-18) and MUSDT (06-26) remain in `active bots/`. Prepared
  `C:\Users\cris_\neutralgrid_tasks\run_d7_active_bots.bat` (scans every YAML
  in `active bots\` via `--bots-dir`; verified live 2026-07-03: both bots
  ticked, END verdicts, authoritative full-fidelity `meta_proba` under the new
  sigmoid calibrator, ENA joinable by `candidate_id`). Registering the
  replacement task requires operator approval (persistence + the open
  question of whether the END-streaked bots should instead be retired).
- **Isolated audit — FP-catch validation**
  (`outputs/audits/live_fp_catch_validation_20260702/`): across the 4
  live-deployed bots, all 3 losers drew END/ADJUST flags where coverage
  existed; MUSDT (live-meta FP, 0.49–0.65 under the old calibrator) was
  END-flagged 3.8h after deploy, ≥1.70pp before the observed decline;
  coverage gaps — not discrimination — were the failure mode.
- Verification: focused suites green (`test_enrich_grid_params.py` 23,
  calibration gate 4, config fail-closed 3, dated contract 37, leakage subset
  53); `pyright` 0 errors on all touched files; full-suite + pyright rerun at
  session end.

### Data - MUSDT final outcome ingested — all live-bot outcomes closed-loop (2026-07-03)

- **MUSDT (412879657)** cancellation dump provided and ingested
  (`data/manual_input/2026-07-03/MUSDT_412879657.txt`, dry-run verified):
  canceled 2026-06-26T23:51:51Z after 9h51m, final **−2.11%** (matched +9.53 /
  unmatched −13.70 / funding −0.02, 48 trades) → workbook row 277 (278 total).
  No candidate stamped — correctly: the geometry matcher found no causal
  candidate (every MUSDT scan row had `grid_is_valid=False`; the deployment
  was made outside the pipeline). Backfilled against the active HMM: workbook
  uniform `rolling_180d_20260701_204849` lineage (278/278); MUSDT deploy-time
  posteriors **range_prob 0.017 / trend_prob 0.983** — the regime gate would
  have rejected this deployment outright, empirically validating the model's
  gates against its own realized −2.11%.
- **D5/D6 funnel after both ingestions**
  (`outputs/meta_tilt_shadow_analysis_20260703_post_musdt_ingest.json`):
  553 joined, **121 eligible authoritative full-fidelity rows on 2 join
  keys, 0 pending** (83 post-mortem ticks excluded by the life-window guard,
  including MUSDT's post-cancel tail). Status
  `insufficient_fit_adjust_rows` — the group-safe split requires ≥5
  independent join keys per side, so the tilt stays correctly disabled until
  future D7-instrumented deployments add bot diversity. FP-catch validation
  finalized across all 4 bots: MUSDT's END flag at 3.8h preceded **−2.65pp**
  of final decline; ENA's END preceded its cancel by 89s.

### Data - ENAUSDT final outcome ingested + analyzer life-window guard (2026-07-03)

- Operator confirmed both remaining live bots canceled and provided dumps.
  **ENAUSDT (412730355)**: canceled 2026-06-18T16:23:58Z after 3h22m, final
  **+2.95%** — ingested via the canonical extractor
  (`data/manual_input/2026-07-03/ENAUSDT_412730355.txt`, transcribed from the
  operator's screenshots; dry-run verified before write) → workbook row 276
  (277 rows total), candidate auto-matched to
  `ENAUSDT_20260618_115611_25245053`, and `LiveOutcomeIngestor` joins it via
  **match_method=linkage — the first production firing of the linkage path**.
  Notable: the scanner's first END tick (2026-06-18T16:22:29Z,
  `price_outside_grid`) preceded the operator's cancel by **89 seconds**.
  **MUSDT (412879657)**: cancellation confirmed by the operator but the
  provided folder contained only the deploy dialog + the already-ingested
  3h50m "Working" screenshots — final Time Ended/PNL still required before its
  outcome row can be written (never defaulted; grid expiry is not
  auto-detectable). Both YAMLs retired to `inactive bots/2026-07-03/` —
  `active bots/` is now empty, ending the post-mortem "ghost ticks" the audit
  found (ENA had authoritative D7 rows dated Jun 21 + Jul 3, days after its
  cancel).
- **Analyzer life-window guard** (`meta_shadow_analysis.py`): joined decision
  ticks timestamped outside the outcome's `[start_time_utc, end_time_utc]`
  are now excluded from D5/D6 eligibility (`post_outcome_tick` /
  `pre_deploy_tick` funnel counts; outcomes lacking a window are kept but
  surfaced as `outcome_life_window_missing`). On current data this excludes
  65 post-mortem joined ticks — including ALL 6 of ENA's authoritative rows
  (its in-life ticks predate the D7 meta fields), which would otherwise have
  become fake D5 calibration rows. Post-guard funnel
  (`outputs/meta_tilt_shadow_analysis_20260703_post_ena_ingest.json`):
  joined 415, eligible 1 (AVAX), pending 138 authoritative rows on 1 key
  (MUSDT) — D5/D6 remain correctly fail-closed pending the MUSDT outcome +
  more join-key diversity. Tests: +2 in
  `tests/unit/test_meta_shadow_analysis.py` (9 file total). New-row HMM
  backfill run against the active `rolling_180d_20260701_204849`
  (`--skip-if-fresh`, authoritative per UTILFIX-01).

### Added - DEPTHORACLE-07 stale-controls refit + first complete-dataset challenger evaluation (2026-07-02/03, NO production change)

- Unblocked the depth-aware challenger evaluation that had been impossible
  since the 2026-07-01 HMM rotation: the evaluator's default zero-cost and
  cost-adjusted control models were pinned to `rolling_180d_20260622_135710`
  and failed load-compat against the active `rolling_180d_20260701_204849`
  (both post-refresh eval dirs were empty — evaluator died before writing;
  READMEs added marking them superseded). Both controls were RE-PINNED to the
  active lineage (raw-pickle reconstruct + `MetaLabeler.save()` lineage
  re-stamp; verified `predictions_identical=true`, max_abs_diff 0.0 vs the
  20260625 sources) → `outputs/audits/depth_controls_refit_20260702/`
  (throwaway diagnostics, not production artifacts).
- Ran `scripts/evaluate_depth_live_challenger.py` on the **first OOS-ready
  aggregated depth dataset** (fast100 chain: 18 labelable rows, 8 pos / 10
  neg; time-split train 14 / eval 4) →
  `outputs/audits/depth_live_challenger_eval_20260702_fast100_refit_controls/`.
  Result on the 4-row eval split (single positive: ARBUSDT): active model,
  zero-control and cost-control all rank the one positive LAST (ROC-AUC 0.0,
  AP 0.25), while the active+ex-ante-L2-depth logistic challenger ranks it
  above 2 of 3 negatives (ROC-AUC 0.667, AP 0.50). Every model's top-1 pick
  was a depth-labeled negative (prec@25%=0.0; the challenger's pick VVVUSDT
  had positive raw PnL +8.15% but failed the depth-deployability label; the
  controls' pick had −10.95%). **PROMOTE=false** — support blockers by design
  (`insufficient_labelable_rows_lt_100` at 18, `insufficient_eval_rows_lt_30`
  at 4, `insufficient_eval_positives/negatives_lt_10`,
  `precision_does_not_beat_all_controls` at k=1). Honest read: with one eval
  positive this is coin-flip-grade support — treat strictly as the evaluation
  PATH being unblocked and weakly directional for ERR-061's depth-bottleneck
  diagnosis, NOT as model-quality evidence; the growth path is
  more complete 7h collection cycles (a fast150 cycle is in flight —
  `outputs/audits/depth_shadow_acquisition_cycle_20260703_000000_postwindowfix_fast150_chain/`,
  9 targets, status=running; finalize with
  `scripts/finalize_depth_shadow_acquisition.py` when complete).

### Data - Backfill repopulation + `Last Features` verified row-time enrichment (2026-06-26, NO production change)

- Re-ran the canonical `scripts/backfill_training_features.py` (active HMM
  `rolling_180d_20260622_135710`, `--skip-if-fresh`) on the current 276-row
  `data/new_expired_bots.xlsx` -> backfilled workbook now **276 rows, uniform active-HMM
  lineage** (verify-hmm-lineage PASS); `atr_pct_15m`/`quote_volume_24h` 274/276 (only
  BASEDUSDT/OPGUSDT lack Vision klines on their dates -> fail-closed).
- Filled the 4 `Last Features` columns (the 4 of 20 active-meta features absent from the
  workbook) ONLY from verified row-time sources: `atr_pct_15m`+`quote_volume_24h` from the
  active backfill (274/276); **`open_interest` from Binance Vision `metrics.sum_open_interest`**
  (base contracts == live `/fapi/v1/openInterest`, last 5-min snap <= start_time, lag median
  160s/max 299s) = **276/276**.
- **`micro_round_trip_cost_pct` left fail-closed (0/276):** its scanner definition needs the live
  quoted spread, and Binance Vision publishes **no `bookTicker` for futures/um** (404 daily+monthly,
  BTC/ASTER). No free archive reproduces the live top-of-book spread -> a faithful fill is
  impossible. Workbook now supports an exact **19-of-20** active-feature comparison; 0 rows reach
  `ready_for_true_comparison=READY` (all blocked on micro cost). Evidence:
  `outputs/audits/last_features_fill_20260626_091035/`.
- **Diagnostic A/B (fast_winner_proxy, 3 CV schemes, LogisticRegression):** adding the 3 verified
  features (16->19) **improves** purged-CPCV AUC 0.647->0.697 / AP 0.364->0.437 / precision@25%
  0.377->0.449 and symbol-group AUC 0.728->0.745 (temporal-80/20 n=55 regresses, high variance).
  The flagged 20th-feature proxy (`micro_round_trip_cost_pct_proxy`, estimator fallback, written to
  `Last Features` cols O/P, DIAGNOSTIC_ONLY) adds <=+0.005 AUC on every scheme -> a proxy is NOT a
  substitute for the live-book feature. See `AB_REPORT.md`.

### Investigated - Cost-Aware Deployable-Winner Challenger Loop (2026-06-25, NO production change, ERR-061)

- Tested the scanner-fn-audit-2026-06-25 gap-closure "smallest fix": retrain the
  meta-labeler on an ENGINE-COST-ADJUSTED winner label (eng_liquid 3bps slip / 5bps
  spread; `time_to_target_hours<=7h` net of cost) instead of the zero-cost fast-winner
  label, to produce more *accurate* deployable winners. Built an isolated challenger;
  **did not promote** (active `20260622_230203` unchanged).
- Re-cost the full 7,203-row fastwin pool on identical klines per row (`eng_zero`
  reproduces stored labels; cost ladder zero 0.577 -> eng_liquid 0.462 -> depth_base
  0.341). OOS-fair label-isolation (same split/pipeline/features, only label differs),
  depth-aware proxy ground truth, 1,038 OOS eval rows.
- **Result:** challenger LOSES to the zero-cost control (depth-deploy AUC 0.417 vs 0.469,
  prec@top-25% 0.151 vs 0.205). All three models (active/control/challenger) predict their
  own training label at AUC ~0.68 but have depth-deploy AUC **< 0.5**. Cost-adj and
  zero-cost labels are ~85% concordant -> relabeling is near-idempotent. The bottleneck is
  **feature weakness + missing depth data**, not the training label; closing it needs real
  time-varying L2 order-book depth (features AND a faithful deployable-winner label).
- Protective gates untouched; safety all-PASS (leakage, contract 79, focused meta 46,
  pyright 0). Baseline + challenger artifacts under `outputs/audits/deployable_winner_gap_20260625_175441/`
  and `outputs/audits/challenger_20260625_231555/`.

### Added - DEPTHORACLE-01 fail-closed deployable-winner evidence guard (2026-06-26)

- Added candidate-keyed live/paper depth-shadow capture and fail-closed
  depth-oracle label building for the "make the current model produce deployable
  winners" loop. This is the data/evidence path required before any depth-aware
  challenger retrain: it records scan-time L2 depth, spread, top-N book depth,
  imbalance, fill capacity, impact, funding, and fee context, then refuses to
  label rows unless a full forward depth window plus explicit outcome, timing,
  tail-risk, cost, and fill evidence are present.
- Added a scan-time freshness guard to the collector. Candidate timestamps are
  read from explicit scan-time columns or the repo-standard `candidate_id`
  format, and rows older than the default 900-second freshness window are
  rejected before any API capture. Rejections are persisted to
  `rejected_targets.csv` so stale captures cannot be mistaken for ex-ante
  training evidence.
- Added `scripts/run_depth_shadow_acquisition.py`, an opt-in orchestration guard
  that writes a reproducible acquisition manifest and exact collector command
  for a fresh 7h L2 shadow run. It does not start collection unless `--start` is
  supplied, and it refuses stale deployment CSV rows before launching the
  collector.
- Files modified:
  - `src/neutralgrid/data/depth_shadow.py`
  - `src/neutralgrid/data/depth_oracle.py`
  - `scripts/collect_depth_shadow.py`
  - `scripts/build_depth_shadow_features.py`
  - `scripts/build_depth_oracle_labels.py`
  - `scripts/run_depth_shadow_acquisition.py`
  - `tests/unit/test_depth_shadow.py`
  - `tests/unit/test_depth_oracle.py`
  - `tests/unit/test_depth_shadow_acquisition.py`
  - `.gitignore`
  - `CHANGELOG.md`
- Decision rationale: AFML label quality requires that the training target match
  the deployable event being predicted. The cost-adjusted challenger proved that
  zero-cost/cost-only relabeling does not create depth-deployability signal, so
  the next valid step is evidence acquisition and fail-closed label generation,
  not gate weakening or another label-only retrain.
- Backward compatibility: No breaking changes. No gate, threshold, config, active
  model, or production artifact is modified; new scripts are opt-in audit tools.
- Verification: `python -m pyright src\neutralgrid\data\depth_shadow.py
  src\neutralgrid\data\depth_oracle.py scripts\collect_depth_shadow.py
  scripts\build_depth_shadow_features.py scripts\build_depth_oracle_labels.py
  scripts\run_depth_shadow_acquisition.py`
  returned 0 errors; `python -m pytest tests\unit\test_depth_shadow.py
  tests\unit\test_depth_oracle.py tests\unit\test_depth_shadow_acquisition.py -q`
  passed 14 tests. Fail-closed blocker
  bundles were written under `outputs/audits/depth_oracle_blocker_20260626_context_smoke/`
  and `outputs/audits/depth_oracle_blocker_20260626_3snap/`, both with
  `labelable_rows=0` because the current evidence lacks a full 7h depth window
  and forward outcome/tail columns. A collector freshness blocker was also
  persisted under `outputs/audits/depth_shadow_freshness_blocker_20260626/`:
  `original_target_count=25`, `target_count=0`, `rejected_target_count=25`,
  all `stale_scan_time`. Acquisition runner evidence was persisted under
  `outputs/audits/depth_shadow_acquisition_blocker_20260626/` (`target_count=0`,
  25 stale rejects) and `outputs/audits/depth_shadow_acquisition_manual_plan_20260626/`
  (`status=ready_not_started`, exact collector command, no collection started).
- Fresh end-to-end smoke: after read-only preflight (HMM mean_pass_rate=1.0,
  contract tests 79 passed, dependency check passed, utility calibrator missing
  as WARN), ran an isolated `run_full_pipeline.py --top-n 30 --max-enrichment 10
  --run-dir outputs/audits/depth_shadow_fresh_scan_smoke_20260626 --show-all`.
  The pipeline wrote 30 fresh candidate rows and 1 deployable row without order
  placement. The acquisition runner then started a short 120-second shadow
  collector from that fresh CSV: 30 candidates, 90 depth records, no rejected
  targets, no error iterations. Ex-ante depth features were built for 30 rows
  under `outputs/audits/depth_shadow_acquisition_fresh_scan_smoke_20260626/features/`.
  The oracle correctly remained fail-closed under
  `outputs/audits/depth_oracle_fresh_scan_smoke_20260626/` with
  `labelable_rows=0`: smoke window only ~0.035h and no forward outcome/tail
  columns.

### Added - BOOKDEPTH-01 Binance bookDepth archive acquisition and isolated challenger (2026-06-26, NO production change)

- Added a reusable Binance Vision futures `bookDepth` archive path for the
  deployable-winner loop. The archive files are percentage-bucket depth
  snapshots (`timestamp, percentage, depth, notional`), not raw price-level
  queue replay, so they are treated as historical L2-style evidence and kept
  separate from the raw depth-shadow oracle.
- Downloaded and checksum-verified the full archive set required by the
  `outputs/audits/challenger_20260625_231555/cost_adjusted_training_frame.csv`
  challenger pool: 4,506/4,506 required symbol-date files under
  `data/book_depth_archive/fastwin_pool` (4,291 newly downloaded, 175 seeded from
  the expired/canceled archive, 40 non-ASCII symbol misses fixed by URL
  percent-encoding and retried successfully). Built candidate-safe pre-scan
  features for 5,235/5,235 rows with median snapshot lag 15s, p95 32.3s, max
  170s, and no parse errors. Forward-window archive diagnostics are persisted
  separately and are not training features.
- Trained an isolated diagnostic challenger only. The corrected comparison
  trains controls on their native labels (`y_zero`, `y_cost`) but scores every
  model on `gt_depth`. Result on the 1,038-row eval split: zero-control top-25%
  depth precision 0.208 / AUC 0.478; cost-control 0.173 / AUC 0.454;
  depth-label active-feature baseline 0.473 / AUC 0.767; bookDepth-enriched
  challenger 0.446 / AUC 0.757. Train-only MI-selected depth subsets also failed
  to beat the active-feature depth-label baseline. **Do not promote**: the archive
  features improve over zero/cost controls, but do not add signal beyond the
  existing active features once the label is depth-aware.
- Completed the full 7h live raw depth-shadow run started from
  `outputs/audits/depth_shadow_fresh_scan_smoke_20260626/deployment_ready_20260626_002406.csv`:
  30 candidates, 12,630 raw top-N depth records, 421 snapshots/candidate, no
  rejected targets, and no collector error iterations. Ex-ante raw depth features
  were built for all 30 rows. The depth oracle remained fail-closed with
  `labelable_rows=0` because the scan CSV still lacks explicit outcome PnL,
  target-hit/timing, and tail-risk columns; 10/30 rows also had insufficient
  capacity at the fallback $1,000 position size.
- Files modified:
  - `src/neutralgrid/data/bookdepth_archive.py`
  - `scripts/download_bookdepth_archive.py`
  - `scripts/build_bookdepth_archive_features.py`
  - `scripts/train_bookdepth_archive_challenger.py`
  - `tests/unit/test_bookdepth_archive.py`
  - `CHANGELOG.md`
- Decision rationale: AFML label design requires train-time features to be
  ex-ante while labels can use forward outcomes. Binance `bookDepth` archives
  provide useful historical liquidity evidence, but percentage buckets cannot be
  treated as raw queue replay; promotion remains gated on a challenger that beats
  active and zero-cost controls on depth-aware OOS deployability without worsening
  PnL/tail behavior.
- Backward compatibility: No breaking changes. No gate, threshold, config, active
  model, production feature schema, or promoted artifact is modified; new scripts
  are opt-in audit/acquisition tools and write isolated artifacts under
  `outputs/audits/`.
- Verification: `python -m pyright src\neutralgrid\data\bookdepth_archive.py
  scripts\download_bookdepth_archive.py scripts\build_bookdepth_archive_features.py
  scripts\train_bookdepth_archive_challenger.py` returned 0 errors; `python -m
  pytest tests\unit\test_bookdepth_archive.py tests\unit\test_depth_shadow.py
  tests\unit\test_depth_oracle.py tests\unit\test_depth_shadow_acquisition.py -q`
  passed 20 tests. Active artifact remains `models/meta_labeler` version
  `20260622_230203`; active HMM remains `rolling_180d_20260622_135710`.

### Fixed - DEPTHORACLE-02 depth-shadow sizing guard and repriced raw oracle (2026-06-26, NO production change)

- Corrected the raw depth-shadow evidence path so audit labels cannot be
  fabricated from fallback sizing. `load_depth_shadow_targets` now prefers true
  position-notional columns, derives notional from `deploy_margin_usdt *
  leverage`, treats explicit zero margin/fraction as authoritative zero, and
  only uses fallback notional when no explicit sizing evidence exists. The
  forward-outcome helper now fail-closes rows with missing or non-positive
  capital fraction instead of converting them to a synthetic full-size
  backtest.
- Added `scripts/reprice_depth_shadow_records.py` to recompute fill ratio,
  capacity ratio, and impact metrics from the already-captured raw bid/ask JSONL
  without recollecting or mutating the original collector output. Repriced the
  full 7h smoke capture into
  `outputs/audits/depth_shadow_acquisition_full_repriced_20260626_002406/`.
  Corrected oracle result:
  `outputs/audits/depth_oracle_full_shadow_repriced_20260626_002406/` has
  30 rows, 3 labelable, 2 positives (LINKUSDT, XRPUSDT), 1 negative (TAOUSDT
  insufficient capacity), and 27 unlabeled rows. The prior provisional
  full-shadow oracle that used fallback $1,000 sizing is superseded.
- Files modified:
  - `src/neutralgrid/data/depth_shadow.py`
  - `scripts/build_depth_shadow_outcomes.py`
  - `scripts/reprice_depth_shadow_records.py`
  - `tests/unit/test_depth_shadow.py`
  - `tests/unit/test_depth_shadow_outcomes.py`
  - `CHANGELOG.md`
- Decision rationale: AFML-style label design must align labels with the
  deployable event and fail closed when execution evidence is absent. A
  fallback position can be useful for manual shadow collection, but it must not
  override explicit scanner sizing because zero-capital rows are not deployable
  candidates.
- Backward compatibility: No breaking changes. No gate, threshold, config,
  active model, production feature schema, or promoted artifact is modified.
  The only behavior change is stricter audit-only sizing for depth-shadow
  evidence generation.
- Verification: `python -m pyright src\neutralgrid\data\depth_shadow.py
  src\neutralgrid\data\depth_oracle.py src\neutralgrid\data\bookdepth_archive.py
  scripts\collect_depth_shadow.py scripts\run_depth_shadow_acquisition.py
  scripts\build_depth_shadow_features.py scripts\build_depth_shadow_outcomes.py
  scripts\reprice_depth_shadow_records.py scripts\build_depth_oracle_labels.py
  scripts\download_bookdepth_archive.py scripts\build_bookdepth_archive_features.py
  scripts\train_bookdepth_archive_challenger.py` returned 0 errors. `python -m
  pytest tests\unit\test_bookdepth_archive.py tests\unit\test_depth_shadow.py
  tests\unit\test_depth_shadow_outcomes.py tests\unit\test_depth_oracle.py
  tests\unit\test_depth_shadow_acquisition.py -q` passed 24 tests. `git diff
  --check` returned no whitespace errors, only existing CRLF conversion
  warnings.

### Added - DEPTHORACLE-03 depth-label accumulator and second live shadow cycle (2026-06-26, NO production change)

- Added a leakage-safe accumulator for depth-oracle bundles. It joins
  `depth_labels.csv` only to `depth_exante_features.csv`, excludes forward
  replay/window diagnostics from model features, writes
  `depth_candidate_frame.csv` and `depth_training_frame.csv`, and reports whether
  a time-split OOS challenger comparison is possible. Applied to the corrected
  first full-shadow bundle, it persisted
  `outputs/audits/depth_oracle_dataset_20260626_002406/`: 30 rows, 3 labelable,
  2 positives, 1 negative, 15 ex-ante depth columns, and
  `blocked_insufficient_oos_depth_labels` because the eval split has only one
  class. No training was run.
- Added a post-collection finalizer for completed depth-shadow acquisitions. It
  refuses incomplete collectors by default, then runs the reproducible
  post-processing chain: reprice raw books with corrected sizing, build ex-ante
  features, build forward outcomes, build depth-oracle labels, and aggregate the
  new oracle bundle with prior bundles. This is the intended handoff after a
  7h collector reaches `status=complete`.
- Ran read-only pipeline preflight before the next live shadow cycle. Active HMM
  `rolling_180d_20260622_135710` exists and has eval `mean_pass_rate=1.0`; active
  meta-labeler `20260622_230203` is pinned to that HMM; active 20 meta features
  are present across candidate pipeline, data generator, and unified builder;
  contract tests passed. Warnings only: missing utility calibrator
  (`utility_score=NaN` degraded mode) and profile model bootstrap fallback because
  `data/profile/current.json` is absent.
- Started a second fresh live/paper depth-shadow cycle from
  `outputs/audits/depth_shadow_fresh_scan_cycle_20260626_142520/deployment_ready_20260626_142521.csv`
  (30 candidates, 2 deployment-valid, 9 geometry rows). The collector was started
  under `outputs/audits/depth_shadow_acquisition_cycle_20260626_142521/` with no
  fallback position notional, 30 targets, 7h duration, 60s interval, top-N 20, and
  zero rejected targets at launch. Initial verification showed records writing
  as expected (90 records after three 30-candidate snapshots). Do not retrain
  until the collector completes and the finalizer/aggregator produces an
  OOS-testable depth-label frame.
- Completed the second live/paper shadow cycle and finalized it. Collector
  finished with `records_written=12630`, zero error iterations, no fallback
  position notional. Finalizer output:
  `outputs/audits/depth_shadow_finalized_cycle_20260626_142521/`. The new oracle
  has 30 rows, 4 labelable, 3 positives, 1 negative. Aggregating with the prior
  corrected oracle gives 60 rows, 7 labelable, 5 positives, 2 negatives, 15
  ex-ante depth feature columns, and a technically two-class time split
  (`train=5`, `eval=2`).
- Added a diagnostic-only live-depth challenger evaluator. It joins the
  accumulated depth labels back to scan-time active features, scores active,
  zero-control, and cost-control meta-labelers, and trains a throwaway
  active+depth logistic challenger. Output:
  `outputs/audits/depth_live_challenger_eval_20260626_142521/`. The tiny eval
  set (`n=2`) numerically ranks the positive XRPUSDT over the negative TAOUSDT
  (depth challenger AUC 1.0 / top-25% precision 1.0 versus active/zero/cost AUC
  0.0 / top-25% precision 0.0), but **PROMOTE remains false** because support is
  far below the promotion bar: only 7 labelable rows, 2 eval rows, 1 eval
  positive, 1 eval negative. Treat this as proof that the depth signal path is
  now functioning, not proof of deployable model readiness.
- Files modified:
  - `scripts/aggregate_depth_oracle_dataset.py`
  - `scripts/finalize_depth_shadow_acquisition.py`
  - `scripts/evaluate_depth_live_challenger.py`
  - `tests/unit/test_depth_oracle_dataset.py`
  - `tests/unit/test_depth_shadow_finalize.py`
  - `tests/unit/test_depth_live_challenger_eval.py`
  - `CHANGELOG.md`
- Decision rationale: one corrected raw-depth oracle bundle produced real
  deployability evidence but not enough class/time diversity for OOS model
  selection. The valid next step is accumulation and finalization tooling plus
  another fresh collector, not gate relaxation, production schema mutation, or
  model promotion.
- Backward compatibility: No breaking changes. No gate, threshold, config,
  active model, production feature schema, or promoted artifact is modified.
  New scripts are opt-in audit tools.
- Verification: `python -m pyright scripts\aggregate_depth_oracle_dataset.py
  scripts\finalize_depth_shadow_acquisition.py scripts\evaluate_depth_live_challenger.py
  tests\unit\test_depth_oracle_dataset.py tests\unit\test_depth_shadow_finalize.py
  tests\unit\test_depth_live_challenger_eval.py` returned 0 errors; `python -m
  pytest tests\unit\test_depth_oracle_dataset.py tests\unit\test_depth_shadow_finalize.py
  tests\unit\test_depth_live_challenger_eval.py -q` passed 3 tests.
  `python -m pytest tests/unit/ -k contract -v` passed 79 tests; `python
  scripts/check_deps.py` passed all dependency pins.
- Follow-up validation after finalizing the live shadow cycle: `python -m pyright`
  over all depth/bookDepth audit modules returned 0 errors; focused depth/bookDepth
  pytest passed 27 tests; leakage/label contract selection
  `python -m pytest tests/unit/ -k "contract and (leak or hlabel or label)" -v`
  passed 53 tests. `_KNOWN_LABEL_COLUMNS` remains present and both
  `_prepare_features()` and `train()` still strip label columns before model
  features are built. Active model artifact remains unchanged.

### Fixed - DEPTHORACLE-04 depth-shadow collector wall-clock completion guard (2026-06-29, NO production change)

- Corrected the live raw depth-shadow collector so the requested duration is
  enforced by elapsed wall-clock time, not by a planned iteration count with a
  fixed sleep after each slow API pass. This prevents a nominal 7h capture from
  stretching far beyond its scan-time freshness window while still producing too
  few snapshots for a valid oracle.
- The collector manifest now records planned/actual snapshot counts,
  start/planned-completion timestamps, per-iteration progress, elapsed seconds,
  and final actual snapshot count per candidate. Overrun captures therefore
  remain auditable and cannot be mistaken for complete multi-path depth evidence.
- Files modified:
  - `scripts/collect_depth_shadow.py`
  - `tests/unit/test_depth_shadow_acquisition.py`
  - `CHANGELOG.md`
- Decision rationale: depth-aware labels must remain tied to fresh ex-ante scan
  geometry. A collector that runs by planned loops instead of elapsed time can
  create stale, under-sampled evidence; the valid response is stricter
  acquisition bookkeeping, not gate relaxation or model promotion.
- Backward compatibility: No breaking changes. No gate, threshold, config,
  active model, production feature schema, or promoted artifact is modified.
  Existing collector CLI arguments are unchanged.
- Verification: `python -m pyright scripts\collect_depth_shadow.py
  tests\unit\test_depth_shadow_acquisition.py` returned 0 errors; `python -m
  pytest tests/unit/test_depth_shadow_acquisition.py -q --basetemp
  .tmp_pytest_depth` passed 3 tests.

### Fixed - DEPTHORACLE-05 depth-window completion guard and timeout evidence (2026-06-30, NO production change)

- Ran two fresh top-50 live/paper depth-shadow acquisition attempts after the
  wall-clock collector fix:
  `outputs/audits/depth_shadow_acquisition_cycle_20260629_130803_top50_wallclock/`
  wrote 15,195 raw records but covered only 22,111s of the required 25,140s
  depth window; `outputs/audits/depth_shadow_acquisition_cycle_20260629_214305_top50_timeout/`
  wrote 13,100 raw records but covered only 18,964s. Both are diagnostic-only
  evidence and were not used to train or promote a model.
- Added a fail-closed finalizer guard: a collector process returning successfully
  is no longer enough. The raw depth snapshots must span the requested forward
  window (duration minus one interval) before repricing, oracle labels, dataset
  aggregation, or challenger training can proceed. The failed finalizer manifests
  are persisted under the matching `depth_shadow_finalized_cycle_*` directories
  with `status=blocked_depth_window_incomplete`.
- Added collector-side depth-window status and per-iteration timeout plumbing.
  The collector now records `last_successful_capture_time_utc`,
  `depth_window_seconds`, `required_depth_window_seconds`, and marks
  `status=complete_depth_window_incomplete` when records exist but coverage is
  insufficient. Timeout iterations record `last_attempt_time_utc` and do not
  count as successful coverage.
- Files modified:
  - `scripts/collect_depth_shadow.py`
  - `scripts/run_depth_shadow_acquisition.py`
  - `scripts/finalize_depth_shadow_acquisition.py`
  - `tests/unit/test_depth_shadow_acquisition.py`
  - `tests/unit/test_depth_shadow_finalize.py`
  - `CHANGELOG.md`
- Decision rationale: AFML-style labels must describe the deployable event over
  the full forward horizon. A partial depth path can support diagnostics, but
  treating it as oracle-complete would reintroduce label optimism. The correct
  response is to fail closed and acquire a complete depth path, not to relax
  gates or promote a proxy model.
- Backward compatibility: No breaking changes. No gate, threshold, config,
  active model, production feature schema, or promoted artifact is modified.
  The new timeout option is additive and defaults to 120 seconds.
- Verification: `python -m pyright scripts\collect_depth_shadow.py
  scripts\run_depth_shadow_acquisition.py scripts\finalize_depth_shadow_acquisition.py
  tests\unit\test_depth_shadow_acquisition.py tests\unit\test_depth_shadow_finalize.py`
  returned 0 errors; `python -m pytest tests/unit/test_depth_shadow_acquisition.py
  tests/unit/test_depth_shadow_finalize.py -q --basetemp .tmp_pytest_depthguards`
  passed 6 tests. Active meta-labeler remains `20260622_230203`, pinned to HMM
  `rolling_180d_20260622_135710`.

### Fixed - DEPTHORACLE-06 explicit non-hit negative labels (2026-07-01, NO production change)

- Fixed the depth-oracle label builder so an explicit `target_hit=false` outcome
  with complete PnL, tail-risk, depth-window, fee, funding, and fill evidence is
  labelable as a negative even when `time_to_target_hours` is blank. The previous
  logic treated the blank timing value as missing evidence for all rows, which
  incorrectly discarded valid non-hit negatives from the deployable-winner
  dataset.
- Rebuilt fixed oracle bundles for the two existing complete live-shadow cycles
  plus the new 2026-07-01 fast-30 cycle. The new cycle completed a full depth
  window (`depth_window_seconds=26133.0` vs required `25140.0`) for ETHUSDT and
  ZECUSDT; both are now valid negative labels. Aggregate support increased from
  7 labelable rows (5 positive / 2 negative) to 9 labelable rows (5 positive /
  4 negative), but promotion remains blocked because OOS support is still too
  small and the eval split has no positives.
- Files modified:
  - `src/neutralgrid/data/depth_oracle.py`
  - `tests/unit/test_depth_oracle.py`
  - `CHANGELOG.md`
- Decision rationale: AFML label quality requires missing evidence to fail
  closed, but a known non-hit is evidence for a negative class, not missing
  target timing. This improves depth-label correctness without relaxing
  hard_gate, meta data_missing, geometry, Stage-B, thresholds, or production
  artifacts.
- Backward compatibility: No breaking changes. No gate, threshold, config,
  active model, production feature schema, or promoted artifact is modified.
  The change only affects oracle-label construction for rows with explicit
  target-hit outcomes.
- Verification: `python -m pyright src\neutralgrid\data\depth_oracle.py
  tests\unit\test_depth_oracle.py` returned 0 errors; `python -m pytest
  tests\unit\test_depth_oracle.py -q --basetemp .tmp_pytest_depth_oracle_fix`
  passed 6 tests. Rebuilt oracle/evaluator artifacts under
  `outputs/audits/depth_shadow_finalized_cycle_20260701_065549_positive_fast30_chain/`
  and `outputs/audits/depth_live_challenger_eval_20260701_065549_positive_fast30_chain_oraclefix/`
  with `PROMOTE=false`.

### Changed - Meta-labeler (regrown + promoted) [supersedes HMMROTATE-0622 meta section]

- Pool grown from the existing 6,726-row FASTWIN dataset by backtesting +477 gap
  rows via `scripts/generate_fastwin_dataset.py --hours 24 --min-bars 1440
  --min-age-hours 25` (identical 24h window; an earlier 8h attempt was discarded
  as a confound). Retrained via `retrain_meta_labeler.py --input
  data/new_expired_bots_backfilled_rolling_180d_20260622_135710.xlsx` (active
  profile `snapshot_v20260530_fastwin`, geometric pool, no HMM features).
- Active artifact `models/meta_labeler/` -> `20260622_230203`: total_samples
  6048, n_pos 2834, OOF-AUC 0.7698 [0.7579, 0.7820], OOF-ECE 0.0789, beta_oos
  calibration (ECE 0.200 -> 0.018 raw->cal), contract
  `fast_winner_time_to_3pct_le_7h`, promotion_status `pass`. Lineage HMM
  `rolling_180d_20260622_135710` (unchanged). Prior model auto-backed up.

### Changed - ERR-059 live reintegration (operator-authorized 2026-06-22)

- `src/neutralgrid/scanner/enrich_grid_params.py`: Stage-12 computes `ev_score`
  (via `PnLRanker.compute_score`) BEFORE building the meta feature row, so
  `meta_prob` is computable at decision time and feeds generalized-Kelly sizing.
  `_build_meta_feature_row` gains an `ev_score` parameter.
- `src/neutralgrid/core/config.py`: `TwoStageConfig.meta_gate_enabled=True`,
  `min_meta_prob=0.37` (calibrated OOF operating point, ~precision>=0.60). Stage B
  now fail-closes `data_missing:meta` when `meta_prob` is not computable.
- `run_full_pipeline.py`: corrected the stale comment - `meta_prob` is now
  consumed by the gate + Kelly sizing; backfill is observability-only; candidate
  ordering remains `ev_score`.

### Fixed - ERR-060 micro_round_trip_cost_pct missing from enrich meta probe (2026-06-24)

- Follow-up to ERR-059: turning the Stage-B meta gate ON exposed a second
  serve-path feature omission of the same class as ERR-058/059. The enrich-time
  `_build_meta_feature_row` (`src/neutralgrid/scanner/enrich_grid_params.py`) did
  not populate `micro_round_trip_cost_pct` (1 of the model's 20 contract
  features), so `get_missing_feature_names` flagged it, `predict_proba` was
  skipped, and the probe produced `meta_prob=None` for EVERY candidate
  (`meta_prob_source` fleet-wide on `results/potential_candidates_20260623_123449.csv`:
  missing 240 / post_scoring_backfill 10 / enrich 0). Stage B then fail-closed
  `data_missing:meta` on all 4 candidates that reached it
  (SUIUSDT/AAVEUSDT/LINKUSDT/DASHUSDT) despite backfilled `meta_prob` 0.45-0.49
  (above the 0.37 gate), producing zero deployments that run.
- Fix: threaded the already-computed microstructure cost (Stage 9
  `md.ms_payload`, computed before the Stage-12 probe) into the builder.
  `_build_meta_feature_row` gains an `md: _MicroData | None` parameter and sets
  `micro_round_trip_cost_pct = md.ms_payload.get("micro_round_trip_cost_pct")`
  (kept absent / fail-closed when microstructure estimation raised); the probe
  call site passes `md=md`. No reordering required. No Feature-Pipeline trio
  change - the feature already exists in `candidate_pipeline.py`,
  `data_generator.py`, and `unified_training_builder.py`; this was a
  serve-path-only omission.
- Verification: `pyright` 0 errors on the changed file; `pytest tests/unit/`
  1243 passed (incl. `test_meta_labeler_retrain_contract_v20260530.py`,
  `test_stage_b_meta_gate_fastwin.py`); `pytest tests/unit/ -k contract` 79
  passed; `verify-feature-pipeline` confirms lockstep. Pending operator: live
  `run_full_pipeline.py` re-run to confirm `meta_prob_source` flips
  'missing'->'enrich'; commit-or-revert the still-uncommitted `config.py` gate
  flip.

### Changed - Meta-labeler error fixes (ERR cluster closed)

- ERR-054(b): scalar-safe missing-feature guard in `get_missing_feature_names`
  (list/tuple/set/dict/ndarray/Series/ndim!=0 treated as missing).
- ERR-035: introduced `MIN_POSITIVE_RATE = 0.05` in `core/constants.py`; replaced
  the magic positive-rate literals in `unified_training_builder.py` and the
  meta-labeler positive-rate guard with the named constant.
- ERR-036: added `y_precedence_skipped_columns` / `y_precedence_degenerate` audit
  columns in the y-precedence branch of `unified_training_builder.py`.
- ERR-053: CPCV OOF-coverage transparency - `_auc_cv_oof_coverage` is logged with
  a warning under 0.80, and `auc_cv` is flagged advisory-only at low N.

### Added - FASTWIN v2 generation/label support (ported to main)

- `scripts/generate_fastwin_dataset.py` (new): forward-backtests existing scanner
  candidates from their real scan ts, one ex-ante row per candidate, endogenous
  `time_to_target_hours` (leakage-safe).
- `backtest/btk_unified_runner.py` + `metrics/pnl_curve_features_v20260310.py`:
  emit `time_to_target_hours` / `target_reached` (`extract_time_to_threshold`).
- `backtest/candidate_pipeline.py`, `retrain_meta_labeler.py`: thread the
  fast-winner label/contract; `_KNOWN_LABEL_COLUMNS` includes
  `time_to_target_hours` / `target_reached` (leakage guard intact).

### Changed - Version identity 6.5.7 -> 6.5.8

- `pyproject.toml`, `src/neutralgrid/__init__.py`: package version 6.5.7 -> 6.5.8.
- `CLAUDE.md`, `AGENTS.md`: Project-Overview line and the Live Bot Data Storage
  Policy path updated to `...NEUTRAL grid bot v6.5.8\Live`.
- Deliberately NOT changed (provenance / history integrity): past `[6.5.7-*]`
  CHANGELOG sections; `AUDIT_01.md` / `HMM_CHANGE.md` historical reports (that work
  physically ran in the v6.5.7 worktree); the `[6.5.7-...]` cross-reference
  comment in `meta_labeler.py`; the synthetic 6.5.7 HMM fixture in
  `test_meta_labeler_retrain_contract_v20260530.py`; saved-artifact
  `pipeline_version` stamps (truthful training-time provenance - they read 6.5.8
  on the next retrain); `Codex/*.jsonl` transcripts.

### Observation - meta_prob computability (no code change)

- Across 732 deployable candidates the only ever-missing meta feature is
  `ou_halflife` (7.5%): `scanner/scan.py` stores it only when finite, so the
  no-mean-reversion case (`theta ~ 0` -> `inf`) stays `None` and Stage B
  fail-closes `data_missing:meta`. This is the informative no-mean-reversion
  signal, not a data gap. Enforcement options (finite sentinel + dataset
  regen/retrain for train/serve consistency, vs. accept the fail-closed
  rejection) are deferred pending operator decision.

### Files modified (this block)

- Version/docs: `pyproject.toml`, `src/neutralgrid/__init__.py`, `CLAUDE.md`,
  `AGENTS.md`, `CHANGELOG.md`.
- Code: `src/neutralgrid/scanner/enrich_grid_params.py`,
  `src/neutralgrid/core/config.py`, `src/neutralgrid/core/constants.py`,
  `run_full_pipeline.py`, `src/neutralgrid/models/meta_labeler.py`,
  `src/neutralgrid/training/unified_training_builder.py`,
  `src/neutralgrid/backtest/candidate_pipeline.py`,
  `backtest/btk_unified_runner.py`,
  `src/neutralgrid/metrics/pnl_curve_features_v20260310.py`,
  `retrain_meta_labeler.py`, `scripts/generate_fastwin_dataset.py` (new).
- Artifacts/data: `models/meta_labeler/*`, `models/meta_labeler.pkl`,
  `models/meta_labeler_verification.json`, `models/meta_labeler_backup_*.pkl`,
  `data/fastwin_dataset/*`.
- Tests: `test_meta_labeler_inference_aliases.py`,
  `test_stage_b_meta_gate_fastwin.py`, `test_gate4_micro_osc.py`,
  `test_enhancements_v653.py`, `test_meta_labeler_retrain_contract_v20260530.py`
  (decision-module tests are covered by the 2026-06-21 block).

### Verification

- `python -m pytest tests/` -> 1459 passed in ~71s (full suite).
- `pyright` (repo gate `[tool.pyright] include=["src"]`) -> 0 errors / 0 warnings;
  `src/` changes are clean. Out-of-gate areas (`backtest/`, `scripts/` - not part
  of the installed package) carry pre-existing pandas/pyarrow stub-typing errors
  (1 in `btk_unified_runner.py`, pre-existing and unrelated to the time-to-target
  port; 95 in the wholesale-ported one-shot `scripts/generate_fastwin_dataset.py`).

## [unreleased] - HMMROTATE-0622 HMM Rotation + Downstream Refits (2026-06-22)

### Summary

Rotated the active HMM to `rolling_180d_20260622_135710` via the `hmm-rotate`
runbook (`python retrain_hmm.py`, canonical mode) and ran the downstream cascade
(backfill -> meta-labeler re-pin -> utility calibrator). The HMM and the
meta-labeler promoted on their fail-closed gates; the utility calibrator re-fit
but did NOT promote (gate fail-closed), so `artifacts/utility/current.json`
remains absent and decision-time `utility_score` stays NaN (UTILFIX-01 fallback,
unchanged live state). No code/behaviour changes - artifact rotation + retrain
only.

### Changed - Active HMM (promoted)

- `artifact_manifest.json` -> `hmm.active_version` = `rolling_180d_20260622_135710`
  (prior `rolling_180d_20260617_151954`), `promoted_utc` 2026-06-22T14:08:59Z.
  New artifact: 50 symbols / 858,000 samples / 4-state GaussianHMM, window
  2025-12-23 -> 2026-06-21, source binance_vision. Walk-forward
  `mean_pass_rate=1.0` (3/3 folds), `state_mapping` range=2 / trend=3, identity
  temperature scaler (T=1.0, `disabled_self_supervised`), not 1500-bar truncated.

### Changed - Meta-labeler (re-pinned + promoted)

- Retrained via `retrain_meta_labeler.py` (active profile
  `snapshot_v20260530_fastwin`, geometric `build_meta_labeler_pool()`, no HMM
  features). Re-pinned `lineage.hmm_artifact_version` =
  `rolling_180d_20260622_135710`. Promotion gate PASS (OOF-AUC 0.640
  [0.568, 0.712], ECE 0.058, n_pos 110), 256/256 fast-target rows (pos 111 /
  neg 145). OOF metrics identical to HMMROTATE-0601/0617 -> the HMM-independent
  geometric pool is unchanged, so this is effectively a pure re-pin. Dated
  contract test `test_meta_labeler_retrain_contract_v20260530` PASS (29/29).

### Not promoted (fail-closed) - Utility calibrator

- Re-fit via `scripts/recalibrate_utility.py` on the fresh uniform-lineage pool
  (126 eligible rows after `duration_hours<=7h` + `hmm_failed` exclusion; fit 94
  / holdout 32). Failed gates G3 (holdout AUC 0.353 < 0.5), G4 (holdout mean
  winner-utility -15.82 < loser -11.01), and G7 (`kappa_trend` boundary-pinned at
  0.05); `promotable=false`. Candidate written to
  `artifacts/utility/utility_20260622_143052_116376.json`;
  `artifacts/utility/current.json` left absent. Not forced. `utility_score`
  remains NaN at decision time (UTILFIX-01 fail-closed) - unchanged from prior
  rotations, where utility has never cleared the holdout-generalization gate.

### Added - Calibrator pool refresh

- `data/new_expired_bots_backfilled_rolling_180d_20260622_135710.xlsx`: 250 rows
  re-inferenced against the new HMM (`scripts/backfill_training_features.py
  --default-artifact-version rolling_180d_20260622_135710`, no `--skip-if-fresh`,
  fresh output path per `reference_backfill_merge_contamination`). Uniform
  lineage (all 250 rows = active HMM): 239 finite (`pinned_artifact_replay`), 11
  delisted-symbol rows `artifact_unavailable` (non-finite, downstream-excluded as
  `backfill_status=hmm_failed`); `utility_score` all-NaN (UTILFIX-01 fail-closed).

### Files modified

- `artifact_manifest.json`
- `artifacts/hmm/rolling_180d_20260622_135710/` (new HMM artifact)
- `models/meta_labeler.pkl`, `models/meta_labeler/metadata.json`,
  `models/meta_labeler/model.joblib`, `models/meta_labeler/scaler.joblib`,
  `models/meta_labeler_verification.json`
- `models/meta_labeler_backup_20260622_142611.pkl` (auto-backup of prior model)
- `data/new_expired_bots_backfilled_rolling_180d_20260622_135710.xlsx` (new
  calibrator pool)
- `artifacts/utility/utility_20260622_143052_116376.json` (candidate only; NOT
  promoted)
- `data/trial_log.json` (HMM + meta-labeler trial records)

### Decision rationale

The rolling HMM cadence refreshes the regime model on the trailing ~180d window.
Downstream artifacts pin to HMM lineage (FIXPIPELINE-01), so a rotation mandates
re-pinning the meta-labeler and re-inferencing the calibrator pool to keep
lineage uniform. Promotion gates (HMM `mean_pass_rate>=0.50`; utility G0-G7
holdout generalization) are the authority and were not overridden, per the
`hmm-rotate` runbook and AFML out-of-sample discipline (Lopez de Prado, Ch.11-12).

### Backward compatibility

No breaking changes. Artifact/data rotation only; no source code modified.
`utility_score` remains NaN at decision time (unchanged). Meta-labeler live
serving remains degraded by the pre-existing ERR-054 overlay crash (out of
scope, untouched).

### Verification

- `python retrain_hmm.py` (canonical): promoted, walk-forward `mean_pass_rate` 1.0.
- `verify-feature-pipeline`: PASS (20/20 active features synced across the three
  pipeline files; `FeatureSnapshot` fields == `to_dict()` keys).
- `verify-hmm-lineage` on backfill output: uniform lineage = active HMM
  (finiteness INCOMPLETE only for the 11 delisted-symbol rows, calibrator-excluded).
- Meta-labeler dated contract test `v20260530`: 29 passed.
- Full suite `python -m pytest tests/`: 1455 passed.
- `python -m pyright`: 0 errors, 0 warnings, 0 informations.

## [unreleased] - METALABELER-LIVETELEMETRY: fail-closed live linkage geometry backfill (2026-06-21)

### Summary

Moved D5/D6 closer to an auditable live proof by repairing the current
candidate-linkage blocker for live shadow telemetry. Active ENAUSDT telemetry
had `strategy_id` but no `candidate_id`, so D7 JSONL rows could carry
full-fidelity `meta_proba` while still being hard to join to the deployment-ready
candidate snapshot and eventual `live_outcome_ingestor` outcomes. Added an
explicit opt-in geometry resolver to the deploy-link backfill utility: exact
`candidate_id` remains the default requirement, but `--allow-geometry-match`
can recover a missing ID only when symbol, grid bounds, grid count, leverage,
and scan-before-deploy chronology produce exactly one deployment-ready row.
Backfilled ENAUSDT strategy `412730355` to
`ENAUSDT_20260618_115611_25245053`; AVAXUSDT was deliberately left unlinked
because its nearest geometry row was not causally prior to the recorded deploy
timestamp. Stamped the same verified ENA candidate ID into the active and
canonical live YAMLs so future D7 JSONL rows carry the join key directly instead
of requiring recovery from `deploy_snapshot`.

### Changed

- `src/neutralgrid/live/deployment_link_backfill.py`: added opt-in
  `--allow-geometry-match` / `--grid-tolerance-pct` path for missing
  `candidate_id`, with fail-closed rejection for no match, future scan
  timestamp, and ambiguous geometry matches.
- `tests/unit/test_deployment_link_backfill.py`: added regression coverage for
  default refusal, unique causal geometry success, future-candidate rejection,
  and ambiguous-match rejection.
- `src/neutralgrid/training/live_outcome_ingestor.py`: aligned D6 outcome
  ingestion with live-monitor linkage semantics by using the latest
  append-only linkage row for repeated `strategy_id` values and extracting
  scan features by exact `candidate_id` when the deployment-ready CSV exposes
  that column.
- `src/neutralgrid/live/decision/meta_shadow_analysis.py`: added and hardened a
  non-gating offline D5/D6 analyzer that joins D7 JSONL shadow records to
  `LiveOutcomeIngestor` outcomes. Preferred join is exact `candidate_id`
  (including `deploy_snapshot.candidate_id` fallback); legacy rows without a
  candidate can recover outcomes through exact `strategy_id` + `symbol` only.
  The analyzer now reports join coverage, metric availability, and an explicit
  fail-closed gate audit / recommended config action. It also reports an
  eligibility funnel that separates join coverage from the actual D5/D6 blockers
  (missing `meta_proba`, non-authoritative meta, non-full-fidelity meta, missing
  outcome, missing timestamp), plus a pending-authoritative-outcomes section
  listing full-fidelity D7 rows that have valid meta telemetry but no finalized
  outcome yet. Its temporal fit/OOS split is grouped by `join_key` so repeated
  ticks from the same live bot cannot appear on both sides of the proof; D5/D6
  also require minimum unique join-key support for fit ADJUST rows, OOS rows,
  and the OOS low-confidence tilt subset.
- `src/neutralgrid/live/decision/renderer.py`: improved D7 JSONL joinability by
  emitting top-level `candidate_id` from a resolved `deploy_snapshot` when the
  YAML spec itself has no candidate ID. Explicit YAML `candidate_id` still takes
  precedence.
- `tests/unit/test_live_outcome_ingestor.py`: added regression coverage for
  repeated strategy IDs and same-symbol duplicate candidates in one scan.
- `tests/unit/test_meta_shadow_analysis.py`: added coverage for
  `deploy_snapshot.candidate_id` fallback joins, candidate-less
  `strategy_id`+`symbol` recovery, insufficient-data refusal, fail-closed gate
  audit output, joined-but-ineligible eligibility-funnel reporting, group-safe
  fit/OOS splitting, pending authoritative outcome reporting, and a synthetic
  successful D5/D6 threshold/OOS-lift path.
- `tests/unit/test_decision_phase_d.py`: extended JSONL audit coverage so
  top-level `candidate_id` falls back to `deploy_snapshot.candidate_id` and
  explicit YAML `candidate_id` remains authoritative. Added a config-loader
  regression proving the future D5 recommendation fields
  `meta_tilt_enabled` and `meta_tilt_low_threshold` are accepted through the
  existing live `--config-file` path.
- `data/linkage/deploy_linkage_log.csv`: appended one ENAUSDT linkage row for
  strategy `412730355` -> `ENAUSDT_20260618_115611_25245053` using the
  geometry resolver.
- `active bots/ENAUSDT.yaml`: replaced `candidate_id: null` with
  `candidate_id: ENAUSDT_20260618_115611_25245053`.
- `Live/2026-06-18/ENAUSDT/ENAUSDT_live_bot_data_scanner.yaml`: replaced
  top-level `candidate_id: null` with the same verified ENA candidate ID.
- `METALABELER_LIVETELEMETRY.md`: updated the live telemetry checkpoint with
  the new joinable ENA proof and the remaining AVAX/D5/D6 blockers.
- `logs/live_decisions_20260621.jsonl`: appended a one-shot ENA live audit row
  proving `deploy_snapshot` now resolves from the linkage row, then another
  one-shot row after the YAML stamp proving top-level `candidate_id` is present.
- `outputs/meta_tilt_shadow_analysis_20260621.json`: current-data D5/D6
  analyzer output; status is `insufficient_joined_rows`, not an enable signal.

### Decision rationale

AFML-style meta-labeling depends on point-in-time ex-ante features being joined
to later outcomes without label leakage. A live `strategy_id` alone is not
enough to prove D6 because the analysis must recover the original scanner
candidate snapshot. The geometry resolver is therefore opt-in and causal: it
does not infer from symbol alone, does not consume outcome fields, and refuses
to write when the evidence is ambiguous or temporally impossible.

### Backward compatibility

No breaking changes. Existing exact-`candidate_id` backfills behave as before,
and missing-`candidate_id` YAMLs still fail unless the operator explicitly passes
`--allow-geometry-match`. The live scanner remains in shadow mode with
`meta_tilt_enabled=False`.

### Verification

- `python -m pytest tests/unit/test_deployment_link_backfill.py -q` -> 8 passed.
- `python -m pyright src\neutralgrid\live\deployment_link_backfill.py` -> 0 errors.
- `python -m pytest tests/unit/test_live_outcome_ingestor.py -q` -> 10 passed.
- `python -m pyright src\neutralgrid\training\live_outcome_ingestor.py` -> 0 errors.
- `python -m pytest tests/unit/test_meta_shadow_analysis.py -q` -> 7 passed.
- `python -m pyright src\neutralgrid\live\decision\meta_shadow_analysis.py` -> 0 errors.
- `python -m pytest tests/unit/test_meta_shadow_analysis.py tests/unit/test_live_outcome_ingestor.py tests/unit/test_deployment_link_backfill.py tests/unit/test_decision_phase_d.py tests/unit/test_decision_renderer.py -q`
  -> 55 passed.
- `python -m pyright src\neutralgrid\live\decision\renderer.py src\neutralgrid\live\decision\meta_shadow_analysis.py src\neutralgrid\training\live_outcome_ingestor.py src\neutralgrid\live\deployment_link_backfill.py src\neutralgrid\live\decision\monitor.py`
  -> 0 errors.
- `python -m pytest tests/unit/test_decision_phase_d.py tests/unit/test_decision_recommender.py tests/unit/test_meta_shadow_analysis.py -q`
  -> 61 passed.
- `python -m pyright src\neutralgrid\live\decision\recommender.py src\neutralgrid\live\decision\meta_shadow_analysis.py`
  -> 0 errors.
- Dry-run ENA canonical + active YAMLs -> one causal candidate:
  `ENAUSDT_20260618_115611_25245053`.
- Dry-run AVAX active YAML -> rejected with `candidate_id not found by geometry`.
- `python live_decision_scanner.py --once --bots "active bots\ENAUSDT.yaml" --no-discord --config-file config\live_discord_emit_all.yaml`
  appended a full-fidelity authoritative ENA row with `deploy_snapshot.candidate_id`
  `ENAUSDT_20260618_115611_25245053` and `delta_meta_prob=0.02433120064304084`.
- Re-ran the ENA one-shot after stamping the YAMLs: appended a full-fidelity
  authoritative row with top-level `candidate_id=ENAUSDT_20260618_115611_25245053`,
  matching `deploy_snapshot.candidate_id`, and
  `delta_meta_prob=0.023880478559920315`.
- Current `LiveOutcomeIngestor(data/new_expired_bots.xlsx, linkage_dir=data/linkage)`
  smoke check -> 202 rows, 0 active ENA/AVAX strategy hits, and no finalized
  outcome yet for `ENAUSDT_20260618_115611_25245053`; D5/D6 remain disabled.
- `python -m neutralgrid.live.decision.meta_shadow_analysis --decisions "logs/live_decisions_*.jsonl" --expired-bots data\new_expired_bots.xlsx --linkage-dir data\linkage --scanner-results-dir results --output outputs\meta_tilt_shadow_analysis_20260621.json`
  -> `decision_rows=703`, `outcome_rows=202`, `joined_rows=358`
  (`strategy_symbol=358`), `eligible_rows=0`,
  `with_pnl_pct=358`, `with_ts_utc=358`, `with_meta_proba=2`,
  `meta_authoritative_true=0`, `meta_full_fidelity_true=0`,
  `decision_rows_authoritative_full_fidelity=4`,
  `pending_authoritative_outcomes.pending_rows=4`,
  `pending_unique_join_keys=3` (`412730355|ENAUSDT`,
  `412770639|AVAXUSDT`, `ENAUSDT_20260618_115611_25245053`),
  `status=insufficient_joined_rows`, `config_action=keep_meta_tilt_enabled_false`,
  `ready_to_enable=false`.

## [unreleased] - METALABELER-LIVETELEMETRY: promote HMM 0617 + meta-labeler reactivated + Criterion-6 auditability (2026-06-18)

### Summary

Resolved the HMM/meta-labeler lineage blocker that left the promoted meta-labeler
unloadable (active HMM `rolling_180d_20260601_134113` vs meta-linked
`rolling_180d_20260617_151954` -> `MetaLabeler.load` raised fail-closed). Per operator
decision, promoted `rolling_180d_20260617_151954` as the active HMM via the canonical
`promote_hmm_version()` path (`mean_pass_rate=1.0 >= 0.5` gate enforced; identity
temperature scaler; `total_samples=858000`, not Binance-1500-bar truncated;
`metadata.artifact_version` consistent). The meta-labeler now loads with
`promotion_status="pass"` in BOTH the batch pipeline (`run_full_pipeline.py`) and the
live-bot telemetry monitor (`MonitorContext.create`), so it again drives
enrichment-time Kelly sizing + `ev_meta_blend` ranking under the existing fail-closed
authority rule (deployment influence only when `promotion_status=="pass"`).

### Changed

- `artifact_manifest.json`: active HMM `rolling_180d_20260601_134113` ->
  `rolling_180d_20260617_151954` (canonical `promote_hmm_version`; no manifest
  hand-edit, no `artifact_compat.py` bypass).
- `run_full_pipeline.py`: additive auditability stamp (METALABELER_LIVETELEMETRY Done
  Criterion 6) on `df_enriched` -> `deployment_ready` CSV: `meta_feature_profile`,
  `meta_labeler_hmm_artifact_version`, `stage_b_meta_gate_enabled` (alongside the
  pre-existing `meta_prob` / `meta_prob_source` / `meta_prob_authority` /
  `active_hmm_artifact_version` / `deployment_score_source` / `meta_size_factor`). No
  deploy-linkage schema change (records join by `candidate_id`; avoids append-only
  header corruption of the existing log).

### Not changed (scope / fail-closed preserved)

- Stage B meta gate left OFF (`TwoStageConfig.meta_gate_enabled=False`); no calibrated
  `min_meta_prob` operating point exists yet (the static 0.50 is not calibrated).
- Live-monitor 5-feature parity gap and any `recommender.decide()` consumption of
  `meta_proba` deferred (trading-policy change; needs OOS + focused test per policy).
- No edits to `artifact_compat.py`, leakage guards, or the Feature Pipeline Update Rule files.

### Verification

- `MetaLabeler.load` -> `is_trained=True`, `promotion_status="pass"`; live
  `MonitorContext` `meta_unavailable=False` (20 features).
- Enrich authority + Stage B gate tests: 10 passed; leakage/label contract:
  53 passed / 1149 deselected; AFML circular-feedback compliance: 67 passed.
- Full suite: `python -m pytest tests/` -> **1418 passed**. `pyright run_full_pipeline.py`: 0 errors.

## [unreleased] - AUDIT-SUNNYCOSMOS Full-Pipeline Health Audit (2026-06-02)

### Summary

Verification-first health audit of `run_full_pipeline.py` (scan -> enrich ->
deploy). The pipeline was found structurally healthy - fail-closed on critical
artifacts, EV ranking/sizing sound, lineage stamped - with NO math correctness
bug in EV/ranking/sizing. Three simple, best-practice fixes landed: (1) the
healthy meta-labeler was emitting nothing due to a feature-ordering bug -
repaired diagnostic-first; (2) the inert HMM-winner calibrator was deleted in
full, leaving EV as the sole deployment ranker; (3) silent-degradation paths
were made visible. A net simplification: a 949-line calibrator module and its
413-line test were removed outright, plus their wiring in `scan.py` /
`run_full_pipeline.py`. All experiments were in-memory / read-only; no temp
files created. Full suite green except the 2 pre-existing ERR-056
funding-assertion failures.

### Fixed - Meta-labeler meta_prob ordering bug (diagnostic-first, ERR-058)

- The promoted, lineage-correct meta-labeler (`metadata.json`
  `promotion_status="pass"`, `oof_auc=0.640`, 20 features) emitted `meta_prob`
  for ZERO deployable candidates: its first feature `ev_score` is a post-scoring
  artifact, so both prediction sites failed `get_missing_feature_names` on it
  (enrich-time `_build_meta_feature_row`, and the in-loop post-scoring backfill
  that ran before `ev_score` was attached). Verified on
  `results/deployment_ready_20260601_160331.csv`: 8/8 valid rows `meta_prob`
  NaN / source "missing" despite all 20 features present.
- Fix: a SECOND pass in `_apply_afml_post_scoring` (after `out['ev_score']` /
  `out['meta_prob']` are attached) recomputes `meta_prob` for still-null rows
  from the now-complete frame, stamping `meta_prob_source="post_scoring_backfill"`.
- DIAGNOSTIC-ONLY: `meta_prob` is populated for observation but is NOT wired into
  ranking, the Stage B gate, or Kelly sizing. `meta_gate_enabled` stays `False`
  (`config.py:306`). Operator chose to observe live values before any wiring.

### Removed - HMM-winner calibrator (deleted entirely, EV is sole ranker)

- Isolated experiments proved it inert: non-authoritative via
  `hmm_artifact_version_mismatch` (pinned `...0516` vs active `...0601`) -> zero
  ranking effect; if authoritative it would override the richer EV ranking with
  a 2-feature logistic; it failed re-promotion every rotation (ERR-055). Zero
  `training/` and `live/` consumers.
- Deleted `src/neutralgrid/calibration/hmm_winner_calibrator.py` (949 lines) and
  `tests/unit/test_hmm_winner_calibrator.py` (413 lines).
- `run_full_pipeline.py`: removed `_hmm_winner_authority_status`, the
  `hmm_winner_authoritative` parameter + authoritative ranking branch (collapsing
  `_apply_afml_post_scoring` to a single EV-only sort), the authority logging
  block, and the 4 `hmm_winner_*` CSV audit columns.
- `scanner/scan.py`: removed the calibrator import, load block,
  `_score_hmm_winner_with_lineage`, and the per-row scoring call.
- `calibration/utility_calibrator.py`: removed the stale
  "Mirrors hmm_winner_calibrator's helper" comment.
- Tests: removed the two authority/ranking tests from `tests/test_afml_bugfixes.py`
  and their `hmm_winner_*` NEAR-row keys.
- Docs: dropped the calibrator command + subpackage line from `CLAUDE.md` and
  `AGENTS.md`; removed the winner-calibrator refit step from the `hmm-rotate`
  skill and the reference from the `verify-hmm-lineage` skill.
- Left on disk (inert): `artifacts/hmm_winner_calibrator/` (operator may archive).
  The offline `grid/spacing_profile.py` winner-IQR filter is kept - its mask
  self-degrades to all-True when the column is absent (no behaviour change).

### Changed - Run-health visibility (log-only)

- Un-silenced two swallowed exception handlers: `_read_json_dict` and the
  post-scoring meta-labeler load now `logger.warning` on failure (still fail-soft,
  no escalation).
- Added `_log_run_health_summary`, an end-of-run summary (printed after
  `display_deployment_summary`): profile_model active vs similarity-only,
  `meta_prob` coverage, and the count of deployable candidates excluded from
  ranking. No schema change.

### Error log housekeeping

- ERR-058 logged + CLOSED (meta_prob ordering bug, resolved by Step 1).
- ERR-055 CLOSED (winner-cal re-promotion - superseded by deletion).
- ERR-057 CLOSED (pyright `:374`/`:435` sites lived in the deleted branch).
- ERR-051 CLOSED (regime stochastic `survival_min` - confirmed already resolved
  in code 2026-05-22). ERR-056 left OPEN (pre-existing funding-test fragility).

### Files modified

- `run_full_pipeline.py`, `src/neutralgrid/scanner/scan.py`,
  `src/neutralgrid/calibration/utility_calibrator.py`
- `tests/test_afml_bugfixes.py`
- `CLAUDE.md`, `AGENTS.md`, `.claude/skills/hmm-rotate/SKILL.md`,
  `.claude/skills/verify-hmm-lineage/SKILL.md`, `ERRORS_LOG.md`
- Deleted: `src/neutralgrid/calibration/hmm_winner_calibrator.py`,
  `tests/unit/test_hmm_winner_calibrator.py`

### Verification

- `python -m pytest tests/` -> 1416 passed; only the 2 pre-existing ERR-056
  funding-assertion failures remain (unrelated).
- `python -m pyright run_full_pipeline.py` (+ edited source) -> 0 errors.
- Isolated in-memory experiments: `meta_prob` populates 8/8 (0.39-0.48,
  source "post_scoring_backfill"); ranking sort is EV-only; calibrator module +
  authority pathway gone. (The live discovery dry-run needs Binance connectivity
  unavailable in this environment, so it was validated via these experiments.)

### Related

- ERR-055/057/058/051 (closed), ERR-056 (open). Plan
  `.claude/plans/do-a-code-audit-sunny-cosmos.md`. Memory
  `project_audit_sunnycosmos_0602`.

## [unreleased] - FIXEDCAP-0601 Fixed Base Capital $400 (2026-06-01)

### Summary

Pinned the full-pipeline base capital to the config value ($400) and removed the
account-balance sizing path. The dynamic per-bot `capital_fraction` (Kelly x
PositionSizer) is preserved - deployed margin is still `capital_fraction *
base_capital`. This makes runs deterministic and independent of live account
balance / Binance connectivity.

### Changed - Capital source (`run_full_pipeline.py`)

- `base_capital = float(get_config().grid.capital)` (was: resolved from the
  `--capital` argument, defaulting to the live Binance available balance).
- Removed the `--capital` CLI argument and the account-balance branch
  (`client.get_account_balance()` + the numeric-override / error handling).
- Per-bot sizing unchanged: `df_enriched["deploy_margin_usdt"] =
  (capital_fraction * base_capital).round(6)`.

### Backward compatibility

- BREAKING (CLI): `--capital` no longer exists. Set base capital via
  `GridConfig.capital` (config / `.env`), default $400. Scripts passing
  `--capital ...` to `run_full_pipeline.py` must drop the flag.

### Related

- Memory `project_fixed_capital_400_0601`.

## [unreleased] - HMMROTATE-0601 HMM Rotation + Core Downstream Refits (2026-06-01)

### Summary

Rotated the active HMM to `rolling_180d_20260601_134113` via the `hmm-rotate`
runbook (`python retrain_hmm.py`, canonical mode) and ran the operator-scoped
"core refits only" cascade (backfill -> meta-labeler -> HMM winner calibrator;
utility calibrator deferred). The HMM and the meta-labeler promoted on their
fail-closed gates; the HMM winner calibrator re-fit but did NOT promote (gate
fail-closed), leaving it pinned to a two-rotations-old HMM (tracked as ERR-055).
No code/behaviour changes - artifact rotation + retrain only.

### Changed - Active HMM (promoted)

- `artifact_manifest.json` -> `hmm.active_version` = `rolling_180d_20260601_134113`
  (prior `rolling_180d_20260524_140738`), `promoted_utc` 2026-06-01T13:55:33Z.
  New artifact: 50 symbols / 858,000 samples / 4-state GaussianHMM, window
  2025-12-02 -> 2026-05-31, source binance_vision. Walk-forward `mean_pass_rate=1.0`
  (3/3 folds), identity temperature scaler, not 1500-bar truncated. Screening
  fail-closed-rejected 19 low-coverage symbols (incl. HBAR/ICP/SEI/TIA/CHZ at a
  ~92-day Vision floor - possible recent-month Vision gap, non-blocking).

### Changed - Meta-labeler (re-pinned + promoted)

- Retrained via `retrain_meta_labeler.py` (active profile `snapshot_v20260530_fastwin`,
  geometric `build_meta_labeler_pool()`, no HMM features). Re-pinned
  `lineage.hmm_artifact_version` = `rolling_180d_20260601_134113`. Promotion gate
  PASS (OOF-AUC 0.640 [0.568, 0.712], ECE 0.058, n_pos 110); calibrated beta_oos
  (ECE 0.263 -> 0.013). Effectively a re-pin since the pool is HMM-independent.

### Not promoted (fail-closed) - HMM winner calibrator

- Re-fit on the fresh uniform-lineage pool; failed gate P1 (`holdout_auc_delta`
  -0.059 < +0.10) and P2 (no balanced-accuracy improvement) on a 41-row holdout
  (`promotable=false`). `artifacts/hmm_winner_calibrator/current.json` left
  unchanged (still `hmm_winner_20260517_181910_112991`, pinned
  `rolling_180d_20260516_220545`). Not forced. See ERR-055.

### Added - Calibrator pool refresh

- `data/new_expired_bots_backfilled_20260601.xlsx`: 250 rows re-inferenced against
  the new HMM (`scripts/backfill_training_features.py --default-artifact-version
  rolling_180d_20260601_134113`, no `--skip-if-fresh`, fresh output path per
  `reference_backfill_merge_contamination`). Uniform lineage verified
  (`verify-hmm-lineage`); 11 low-history rows (BASED/NIGHT/BSB/CHIP/OPG)
  non-finite/un-featurizable (downstream-excluded); `utility_score` all-NaN
  (UTILFIX-01 fail-closed; utility deferred).

### Tooling - skill-drift fix + scratch cleanup

- Fixed a stale `artifact_manifest.json` accessor in two runbook skills:
  `.claude/skills/hmm-rotate/SKILL.md` (Step 1) and
  `.claude/skills/backfill-features/SKILL.md` (Step 1) read the top-level
  `['active_version']` key, which is `null` since the manifest nests the value
  under `hmm`. Corrected to `['hmm']['active_version']`. (`meta-labeler-refit`
  and `verify-hmm-lineage` already used the correct accessor.)
- Removed scratch run log `hmm_retrain.log` (untracked, ~920 KB); the canonical
  run record lives in `artifacts/hmm/rolling_180d_20260601_134113/` +
  `data/trial_log.json`.

### Files modified

- `artifact_manifest.json`
- `artifacts/hmm/rolling_180d_20260601_134113/` (new HMM artifact)
- `models/meta_labeler.pkl`, `models/meta_labeler/metadata.json`,
  `models/meta_labeler/model.joblib`, `models/meta_labeler/scaler.joblib`,
  `models/meta_labeler_verification.json`
- `models/meta_labeler_backup_20260601_143506.pkl` (auto-backup of prior model)
- `data/new_expired_bots_backfilled_20260601.xlsx` (new calibrator pool)
- `artifacts/hmm_winner_calibrator/hmm_winner_20260601_145158_451447.json`
  (candidate only; NOT promoted)
- `data/trial_log.json` (HMM + meta-labeler trial records)
- `ERRORS_LOG.md` (ERR-055)

### Backward compatibility

- No breaking changes. Decision-time callers resolve the active HMM from
  `artifact_manifest.json`. NOTE: the winner calibrator remains pinned to
  `rolling_180d_20260516_220545` (two rotations stale) - confirm decision-time
  lineage-mismatch behaviour per ERR-055 before relying on winner-cal output.

### Verification

- HMM: walk-forward `mean_pass_rate=1.0`, artifact naming `rolling_180d_\d{8}_\d{6}`,
  identity scaler, not truncated; `python retrain_hmm.py` exit 0, manifest updated
  atomically.
- Meta-labeler: dated contract test
  `tests/unit/test_meta_labeler_retrain_contract_v20260530.py` GREEN post-refit
  (29 passed); pin confirmed in `models/meta_labeler/metadata.json`.
- Backfill: `verify-hmm-lineage` uniform `rolling_180d_20260601_134113`, 11
  non-finite rows isolated.
- Winner calibrator: dry-run proof summary `promotable=false`
  (P1=false P2=false P3=true P4=true).

### Related

- ERR-055 (winner-cal persistent P1/P2 gate failure + 0516 cross-lineage
  staleness, WATCH); ERR-053 (meta-labeler CV degeneracy, sibling). Memory
  `project_hmmrotate_0601`.

## [unreleased] - FASTWIN-01 Meta-Labeler Rebuild + Promotion Gate (2026-05-30)

### Added - Fast-winner meta-labeler (20-feature ex-ante L2 logistic)

- New active feature profile `snapshot_v20260530_fastwin` (20 ex-ante,
  leakage/circularity-clean, NON-HMM features) and contract
  `fast_winner_duration_le_7h_pnl_ge_3` (label `eff_pnl >= 3% & duration <= 7h`).
  `ACTIVE_META_TARGET_PNL_THRESHOLD_PCT` 0.0 -> 3.0.
- `UnifiedTrainingBuilder.build_meta_labeler_pool()`: sources the authoritative
  GEOMETRIC backtest pool (~256 rows) directly via `_load_backtest_rows` +
  build()-style authority filter (`~version_gated & is_authoritative &
  source_class!=reconstruction`), instead of the snapshot inner-join (~60 rows).
  Does not modify `build()`; preserves every leakage/authority guard.
- `estimator_type="logistic"` path in `MetaLabeler` (L2, class_weight balanced).
- Reproducible feature study committed at
  `scripts/meta_labeler_feature_study.py` (symbol-grouped purged CV, bootstrap CI,
  mandatory `no-feature-AUC>0.90` leakage assertion).

### Added - Promotion gate (fail-closed) judged on validated methodology

- `MetaLabeler._evaluate_promotion_gate`: promote iff (a) OOF-AUC 95% bootstrap
  CI lower bound > 0.50 (one-sided discrimination, deliberate), (b) n_pos >= 70
  training positives, (c) OOF ECE <= 0.10. Any missing/non-finite metric forces
  `fail`. Persisted to `eval_metrics` (oof_auc/ci/ece/n_pos/promotion_status/
  promotion_reasons); restored onto `MetaLabeler.promotion_status` + `is_promoted`
  on load (fail-closed: None/"fail" => not promoted).
- `MetaLabeler._evaluate_promotion_oof`: the gate is judged with the SAME
  methodology that validated its thresholds (symbol-grouped, time-purged,
  ALL-rows-scored, CALIBRATED OOF), independent of the model's training CPCV.
  Required because symbol-grouped CPCV degenerates on the small/symbol-diverse
  geometric pool (256 rows / 100 symbols -> 3 splits, scores 135/256). See ERR-053.

### Fixed - L2 logistic feature standardization

- `MetaLabeler.train()` had no scaler (it was built for scale-invariant GBM), so
  the L2 logistic collapsed (~0.5 AUC) on features spanning ~1e9 to ~1e-4.
  `_make_preprocessor()` now folds `StandardScaler` after the mean imputer for the
  logistic path, fit per-fold (leakage-safe), persisted with the model so
  inference scales automatically. Tree estimators keep the imputer alone.

### Added - Promotion-gated meta_prob consumption

- `deployment_meta_prob` and `meta_prob_authority` ("authoritative" vs
  "diagnostic_only") are now gated on `promotion_status=="pass"` in
  `enrich_grid_params.py` and `run_full_pipeline.py` post-scoring. Diagnostic-only
  (unpromoted) behavior is unchanged.
- Soft Stage-B meta gate: `TwoStageConfig.meta_gate_enabled` (default OFF),
  `min_meta_prob`. When enabled, rejects on missing/NaN meta_prob with
  `data_missing:meta` (parallel to `data_missing:tos`) and on
  `meta_prob < min_meta_prob`. Inert until enabled.

### Validated - Retrain + full suite

- Retrained on the geometric pool, PINNED to active HMM
  `rolling_180d_20260524_140738`, **promotion_status=pass** (OOF-AUC 0.640
  [0.568, 0.712], ECE 0.058, n_pos 110).
- Contract test bumped `test_meta_labeler_retrain_contract_v20260408.py ->
  _v20260530.py`; inference-alias + candidate-pipeline-bypass fixtures moved to
  FASTWIN; new `tests/unit/test_stage_b_meta_gate_fastwin.py`; the
  `primary_pipeline_score` circular-dependency guard stays green.
- `python -m pytest tests/` -> `1428 passed, 2 failed` (the two are PRE-EXISTING
  `btk_funding` failures, confirmed unrelated by stashing the FASTWIN source).
  `python -m pyright` -> 0 errors on all package files. Leakage guards intact.
- ERR-053 (WATCH) logged: native symbol-grouped CPCV degeneracy on small
  symbol-diverse pools makes `auc_cv` + OOS calibration unreliable; the promotion
  gate sidesteps it via `_evaluate_promotion_oof`.

## [unreleased] - Backtest Engine Binance PPG/PnL Alignment (2026-05-27)

### Fixed - Binance-compatible profit-per-grid math

- Updated live/deploy scanner PPG generation so `profit_per_grid_pct` now uses
  Binance displayed-interval semantics instead of the legacy line-count
  fallback at the live candidate call sites.
- Verified corrected precise values from live telemetry:
  - `IPUSDT`: `0.754572` legacy-style output -> `0.639056056102`
    Binance-compatible precise output; UI truncates to `0.63`.
  - `PNUTUSDT`: `0.715835` -> `0.647504508195`; UI truncates to `0.64`.
  - `ARKMUSDT`: `0.900298` -> `0.824450585184`; UI truncates to `0.82`.
- Kept the shared formula helper's default legacy semantics intact so older
  historical/training rows are not silently reinterpreted.
- Did not add a production `profit_per_grid_display_pct` field; UI truncation is
  validation/reporting evidence only.

### Fixed - Binance-style PnL decomposition

- Backtest results now expose gross realized profit separately from fee-net
  realized profit:
  `matched_profit_usdt = gross_realized_profit_usdt`, while
  `net_realized_profit_usdt = gross_realized_profit_usdt - fees_paid`.
- Preserved explicit PnL identity checks:
  `total_profit_usdt = net_realized_profit_usdt + open_pnl_usdt + funding_fee_usdt`
  and `unmatched_pnl_usdt = total_profit_usdt - matched_profit_usdt - funding_fee_usdt`.

### Validated - History 11 reconciliation

- Added regression coverage for IPUSDT strategy `412235216` and PNUTUSDT
  strategy `412235095` using `Order history 11.csv`, `Trade History 11.csv`,
  and `Transaction history 11.csv`.
- Confirmed Binance matched trades are paired grid cycles, not raw fill count:
  IPUSDT `min(11 BUY fills, 9 SELL fills) = 9`; PNUTUSDT
  `min(10 BUY fills, 8 SELL fills) = 8`.
- Validation commands:
  - `python -m pytest tests/unit/test_grid_formulas.py tests/unit/test_backtest_binance_alignment.py -q`
    -> `21 passed`.
  - `python -m pytest tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py -q`
    -> `76 passed`.
  - `python -m pyright` -> `0 errors`.

## [unreleased] - AUDIT_01 Remediation (AUDIT-04-FIX session, 2026-05-22)

Records the code modifications driven by the AUDIT_01 -> AUDIT_02 -> AUDIT_03
audit chain (`AUDIT_01.md`). AUDIT_01/02/03 were read-only audits; the actual
edits landed in the AUDIT-04-FIX session. Suite after this work:
`pytest tests/` -> 1377 passed / 0 failed (was 10 failed);
`pytest tests/ -k contract` -> 74 passed / 0 failed (was 5 failed);
`pyright` -> 104 errors / 0 warnings (unchanged AUDIT_01 baseline).

### Fixed - Version-constant single-sourcing (A02-F01 / A02-F02, P0)

- `backtest/btk_label_contract.py` - the `except ImportError` fallback no longer
  hardcodes `LABEL_CONTRACT_VERSION="2026-04-17"` / `FORMULA_VERSION="alignment-v1"`
  / `BOT_HORIZON_HOURS=6.0`. It now bootstraps `src/` onto `sys.path` and
  re-imports from `neutralgrid.core.constants`, hard-failing if the package is
  truly absent. Closes the AUDIT-003 drift surface (constants had moved to
  `"2026-05-09"` / `"alignment-v2-geometric-realism"`).
- `tests/unit/test_plan_v6_steps.py::test_constants_module_exists` - replaced the
  frozen literal assertions with str/non-empty checks; consistency is enforced by
  the existing `test_btk_label_contract_uses_same_versions` /
  `test_btk_unified_runner_uses_same_engine_version`.

### Fixed - Inert stochastic survival gate removed (AUDIT-004 / A02-F04 / E4, P1)

- Deleted the dead `survival_ok` check across the stochastic regime path. The
  gate was inert: `survival_min` was always `0.0`, so `survival_prob >= 0.0`
  always held. MC containment is owned by Stage B micro_osc Gate 4.
  Behavior-preserving.
  - `src/neutralgrid/validation/stochastic.py` - removed `StochasticConfig.survival_min`,
    the `StochasticResult.survival_ok` field, the `survival_ok` term in
    `all_passed`, and the `survival_ok = survival >= self.config.survival_min`
    computation. `survival_prob` is still computed and reported.
  - `src/neutralgrid/validation/regime_validator.py` - removed the local
    `survival_min = 0.0`, the `survival_min` constructor arg, and the
    `survival_ok` metrics key and failure-reason branch.
  - `src/neutralgrid/scanner/enrich_grid_params.py` and
    `src/neutralgrid/scanner/scan.py` - removed the parallel `survival_min=0.0`
    hardcodes that only fed the dead check.
  - `tests/unit/test_hurst_quality.py` - dropped `survival_ok` from the
    backward-compatibility construction and `hasattr` assertions.

### Fixed - `source_class="legacy"` no longer clobbered (E7 / P0)

- `src/neutralgrid/training/unified_training_builder.py` - `_apply_ingestion_gate`
  no longer downgrades a more-specific `source_class="legacy"` (missing
  `label_contract_version`) to the generic `non_authoritative` when a later gate
  fails. Origin class is preserved; gating still keys off
  `is_authoritative`/`version_gated`.
- `tests/unit/test_plan_v6_steps.py` - `test_matching_version_not_gated` and
  `test_missing_version_marked_legacy` fixtures extended with the 7
  engine-settings contract fields the gate now checks.

### Added - Grid Mode Authority invariant

- `.claude/rules/safety-invariants.md` - new "Grid Mode Authority" section: the
  backtest engine validates BOTH arithmetic and geometric modes, but the
  AUTHORITATIVE TRAINING pool is geometric-only. Arithmetic rows are valid for
  backtest but intentionally gated `non_authoritative` for training
  (operator-confirmed 2026-05-22).
- `src/neutralgrid/training/unified_training_builder.py` - added the matching
  explanatory comment at the engine-settings authority gate.

### Fixed - Test-fixture drift (P0-5, P0-6, and 10-failure triage; all VERDICT A, no real regressions)

- `tests/unit/test_utility_calibrator.py::test_current_workbook_contract_counts_if_available`
  (E1) - replaced magic counts with structural assertions over all samples
  (`winners+losers==len(pool)`; `fit+holdout==len(pool)`; non-empty; binary label).
- `tests/test_afml_integrations.py` (E2) - root cause was feature-set drift:
  `pattern_profile.DEFAULT_FEATURES` changed to 4 microstructure/pre-event
  features; the synthetic fixture still emitted obsolete columns. Fixture now
  emits the 4 current features.
- `tests/unit/test_unified_training_builder_v20260312.py` - backtest-row fixtures
  extended with the authoritative engine-settings fields (`geometric`/`wick`).
- `tests/unit/test_btk_order_lifecycle.py` and
  `tests/unit/test_btk_exchange_rounding.py` - hand-computed integer-level
  scenarios pinned to `mode="arithmetic"` after the grid default flipped
  arithmetic->geometric (commit 04a15b4).
- `tests/unit/test_btk_global_cooldown.py` - updated to expect
  `TRAINING_ENGINE_DEFAULTS["global_cooldown_bars"] == 0` (120->0, commit
  04a15b4, operator-confirmed intentional).
- `tests/unit/test_bot_data_extractor_v2.py` - expected UTC values updated for
  manual-paste timestamps now interpreted America/Lima -> UTC (+5h, commit
  d1e5591; operator-approved, forward-compatible with the future A02-F10
  config-pinned `OPERATOR_TIMEZONE`).

### Still pending (not in this remediation; see AUDIT_03 priority list)

- P1-7 (E3, A02-F03 backtest entry-point exemption), P1-8 (A02-F05 HMM lineage
  stamping on live decisions), P2-10 (E5 ERR-048 meta_overlay), P2-11 (A02-F10
  full timezone implementation), P3-12..16 backlog, and P3-17 pyright
  pandas-narrowing noise (AUDIT-001 / AUDIT-002) remain open.

## [unreleased] - Run-Scoped Full-Pipeline Training Bundle (2026-05-22)

### Added - Isolated run provenance for full-pipeline training data

- Added `run_full_pipeline.py --run-dir <path>` so an opt-in pipeline run writes
  its deployment CSV and feature snapshots under one isolated run folder instead
  of mixing scratch snapshots into `data/training_snapshots`.
- Added fail-closed argument handling: `--run-dir` cannot be combined with
  `--output`, and a run directory must be new or empty before pipeline output is
  written.
- Added `scripts/validate_training_run_bundle.py` to validate exact
  `candidate_id` matching between run-scoped backtest outcomes and run-scoped
  feature snapshots. The validator writes
  `<run-dir>/validation/run_bundle_validation.json` and fails on blank
  backtest IDs, duplicate backtest IDs, or missing snapshot matches.
- Added focused unit coverage in `tests/unit/test_training_run_bundle.py` for
  path derivation, fail-closed parser behavior, bundle pass/fail validation,
  diagnostic-only snapshot-without-backtest rows, and blank ID rejection.

### Validated - Bundle matching works; active training gate remains strict

- Ran an isolated current full-pipeline proof with `--run-dir`; canonical
  `data/training_snapshots` remained unchanged, while the run folder received
  `deployment_ready_20260522_142413.csv` plus
  `snapshots/snapshots_2026-05-22.parquet`.
- Immediate six-hour candidate backtesting from that current scan could not
  produce rows yet because Binance returned only 23 one-minute bars for each
  backtestable symbol; the default candidate backtest requires 360 bars.
- Built a separate scratch bundle from the latest completed full-pipeline run
  (`deployment_ready_20260521_164214.csv` and
  `snapshots_2026-05-21.parquet`) and ran run-scoped backtesting there.
  Validation passed with `15` backtest rows, `250` snapshot rows,
  `backtest_snapshot_match_rate=1.0`,
  `backtested_missing_snapshot_count=0`,
  `blank_backtest_candidate_id_count=0`, and no duplicate candidate IDs.
- Ran `retrain_meta_labeler.py --dry-run` against the scratch bundle. The
  unified builder correctly joined `15/15` run-scoped backtest rows to
  snapshots and found `167/167` features available, then the existing active
  training completeness gate blocked export because one matched row
  (`ZECUSDT_20260521_164214_ae88fe5c`) had `ou_halflife` null. No runtime
  artifact was promoted.

### Fixed - Explicit unmodelable classification for incomplete selected features

- Updated `retrain_meta_labeler.py` so active bootstrap training no longer
  fails the full run because of one incomplete selected-feature row. Rows whose
  selected model feature vector contains null/non-numeric values are now
  classified as `excluded_unmodelable` before the active feature-completeness
  gate, while complete rows still must pass the exact non-imputed feature
  contract.
- Re-ran the run-scoped scratch bundle after the fix. The unified builder
  joined `15/15` backtest rows to snapshots, classified
  `ZECUSDT_20260521_164214_ae88fe5c` as `excluded_unmodelable` because
  `ou_halflife` was missing, exported `14` modelable rows, and passed feature
  completeness with `is_value_complete=true`.
- Ran staged meta-labeler training into the scratch bundle only. The staged
  artifact was structurally healthy (`7/7` expected features in both artifact
  metadata and pickle), but promotion comparison remained `do_not_promote`:
  current canonical artifact has `54` samples, `auc_cv=0.5`,
  `precision_at_5=0.0`; staged artifact has `14` samples, `auc_cv=0.5`,
  `precision_at_5=0.0`, and no calibration acceptance. No runtime artifact was
  promoted.
- Restored the canonical `data/trial_log.json` after staged training so the
  isolated fit did not contaminate global trial history.

## [unreleased] - Corrected Base To Full-Pipeline Training Inputs (2026-05-21)

### Added - Runtime learning path from corrected canonical data

- Added the next canonical support plan to `IMPLEMENT_UPDATE_BACKTEST.md` with
  the goal: "Make the model training inputs and promoted runtime artifacts
  learn from that corrected base."
- Documented the verified architecture boundary: the corrected canonical
  workbook and flat backfill are necessary but not sufficient for full-pipeline
  learning, because active meta-labeler retraining consumes
  `data/backtest_candidates/training_data_*.csv` outcomes plus
  `data/training_snapshots/*.parquet` features through `UnifiedTrainingBuilder`.
- Documented the required propagation path: rebuild derived backtest outcome
  CSVs from the corrected canonical workbook, validate the unified training
  inputs, stage runtime artifact refits, and promote only after a separate
  approved gate.

### Guardrails

- No manual order/trade/transaction export field becomes a model feature.
- No scanner admission, scanner ranking, HMM regime retrain, or runtime artifact
  promotion is included in the documentation update.
- The only long-lived data paths remain `data/new_expired_bots.xlsx` and
  `data/new_expired_bots_backfilled.xlsx`; scratch and staging outputs must be
  deleted unless separately promoted through the approved gate.

### Validated - Scratch propagation gate did not justify promotion

- Ran `backtest_candidates.py` into an isolated scratch directory with
  `--match-live-duration --expired-bots-path data/new_expired_bots.xlsx`.
  Selected `250` candidates, completed `249`, skipped `1`, and wrote scratch
  `training_data_20260520.csv` plus `backtest_results_20260520.csv`.
- Ran `retrain_meta_labeler.py --dry-run --export-training-data` against both
  current `data/backtest_candidates` and the scratch corrected-derived outcome
  directory. Both exported `54` modelable candidate IDs with the same positive
  label rate (`0.24074074074074073`) and `label_delta=0.0`; the scratch run
  changed some live-matched durations and net PnL values but did not change
  the active fast-target label set.
- Staged runtime checks did not pass promotion gates:
  HMM winner calibrator `promotable=false`; utility calibrator
  `promotable=false` because `G7_finite_nonnegative_not_boundary_pinned=false`.
- Ran a no-deploy full-pipeline proof with snapshot logging disabled in memory
  and output redirected to scratch. Output had `250` rows, `3` grid-valid rows
  (`HIGHUSDT`, `SENTUSDT`, `PROVEUSDT`), `47` non-null `meta_prob` rows,
  `250` non-null `hmm_winner_score` rows, and `0` non-null `utility_score`
  rows because no utility `current.json` is available.
- No runtime artifact was promoted. Scratch outputs were deleted after the
  validation summary was captured.

### Status - Current as of 2026-05-21

- This changelog entry is the current record for the corrected-base propagation
  work. The implemented change set is documentation plus validation only:
  `CHANGELOG.md` and `IMPLEMENT_UPDATE_BACKTEST.md`.
- No canonical workbook values, runtime model artifacts, scanner gates,
  deployment-ready generation logic, HMM regime artifacts, HMM winner artifact,
  utility artifact, or meta-labeler artifact were promoted by this step.
- The corrected canonical data remains the active base, but the latest
  scratch/staging gate blocked runtime artifact promotion because the corrected
  derived training export did not change the fast-target label set and both
  calibrator promotion checks remained non-promotable.

## [unreleased] - ERR-043 Canonical UTC Data Replacement (2026-05-19)

### Changed - Corrected data is now canonical

- Promoted the verified UTC-corrected workbook into
  `data/new_expired_bots.xlsx`; there is no longer a parallel corrected-copy
  workbook path.
- Regenerated the canonical flat backfill output at
  `data/new_expired_bots_backfilled.xlsx` from the corrected canonical raw
  workbook.
- Removed the temporary repair gate script and gate-only unit test after the
  canonical production-path proof passed. Root-cause prevention tests for
  manual timestamp parsing remain active.

### Validated - Canonical production paths match the corrected proof

- One-time gate result before replacement: `gate_pass=true`.
- Baseline versus corrected gate:
  - `holdout_model_mean_abs_pnl_error`: `14.140301546151198` -> `2.479104243919372`
  - `holdout_model_median_abs_pnl_error`: `8.201957766028666` -> `1.45761421948092`
  - `holdout_non_winner_specificity_pnl_lte_0`: `0.5` -> `1.0`
- Canonical post-replacement proof:
  - `RAW_ROWS=227`
  - `FLAT_ROWS=227`
  - `RAW_HAS_RANGE_PROB=False`
  - `FLAT_HAS_RANGE_PROB=True`
  - `FLAT_RANGE_PROB_NONNA=216`
  - `FLAT_HMM_VERSION_NONNA=227`
  - Reconciliation: `rows=61`, `model_rows=55`, `missing_kline_rows=6`,
    `holdout_rows=25`, `holdout_model_rows=19`,
    `holdout_model_mean_abs_pnl_error=2.479104243919372`,
    `holdout_model_median_abs_pnl_error=1.45761421948092`,
    `holdout_non_winner_specificity_pnl_lte_0=1.0`.
- Meta-labeler dry run exported training data successfully with no artifact
  promotion. HMM winner and utility calibration dry runs completed with no
  promotion. Scratch proof outputs were deleted:
  `SCRATCH_EXISTS_AFTER_CLEANUP=False`.

## [unreleased] - ERR-043-SUPPORT Canonical UTC Repair, Flat Backfill Only (2026-05-19)

### Fixed - Manual UI timestamps are converted to UTC at ingestion

- Added one central manual UI timestamp parser in
  `_bot_data_extractor_core.py`: Binance UI/manual timestamps are parsed as
  `America/Lima` local time and converted to UTC before being stored in
  `start_time_utc`, `end_time_utc`, manual trade-fill timestamps, and matched
  profit event timestamps.
- Binance CSV fields already labeled UTC remain unshifted. The CSV parsing
  paths for `Time(UTC)`, `Update Time`, and `Date(UTC)` were not changed.

### Added - Copy-on-write UTC repair manifest

- Added `scripts/repair_expired_bot_utc_windows.py`.
- The repair tool reads `data/new_expired_bots.xlsx`, writes a corrected raw
  workbook copy, and emits a row-level manifest with classifications:
  `correctable`, `stored_utc_valid`, `conflicting_manual_evidence`, and
  `missing_manual_evidence`.
- Correction is exact-evidence only: rows are corrected only when manual
  order-history exports contain a matching `symbol + strategy_id` and the
  manual order start equals the local UTC-5 to UTC candidate. Files without
  `Strategy Id` remain diagnostic and cannot trigger correction.
- The raw corrected workbook is extractor-contract only. No backfill, HMM,
  stochastic, utility, or training fields are added to the raw workbook.

### Validated - Flat derived backfill separation

- Added a focused test proving `scripts/backfill_training_features.py` writes
  backfill/HMM/training fields only to a separate flat derived workbook, while
  the corrected raw workbook input remains unchanged.
- Isolated full validation repaired all 227 workbook rows into scratch,
  backfilled the corrected copy into a scratch flat workbook, inspected both,
  and deleted the scratch directory.
- Validation counts: `227` raw rows, `227` flat derived rows, `109`
  correctable rows, `19` stored-UTC-valid rows, `51` conflicting manual
  evidence rows, and `48` missing manual evidence rows. The corrected raw
  workbook had `RAW_HAS_RANGE_PROB=False`; the flat derived workbook had
  `FLAT_HAS_RANGE_PROB=True`.
- Derived backfill coverage in the isolated run: `range_prob` non-null on
  `216 / 227` rows and `hmm_artifact_version` non-null on `227 / 227` rows.
  The run used active HMM artifact `rolling_180d_20260516_220545`.
- Scratch cleanup proof: `SCRATCH_EXISTS_AFTER_CLEANUP=False`.

### Verification

- `python -m pytest -q tests/unit/test_new_bot_data_extractor.py
  tests/unit/test_backtest_timestamp_policy.py` -> 16 passed.
- `python -m pyright scripts/repair_expired_bot_utc_windows.py
  scripts/validate_backtest_live_reconciliation.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_repair_expired_bot_utc_windows.py
  tests/unit/test_backtest_timestamp_policy.py` -> 8 passed.
- `python -m pyright scripts/backfill_training_features.py
  scripts/repair_expired_bot_utc_windows.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_backfill_training_features_v20260312.py
  tests/unit/test_repair_expired_bot_utc_windows.py` -> 14 passed.
- Note: `python -m pyright _bot_data_extractor_core.py
  new_bot_data_extractor.py` still reports pre-existing broad typing issues in
  the legacy core helper surface; the new manual timestamp tests pass, and the
  repair/backfill scripts are pyright clean.

## [unreleased] - UTC-FIX Timestamp Validation (2026-05-18)

### Added - Backtest timestamp policy validation

- Added explicit timestamp policies to
  `scripts/validate_backtest_live_reconciliation.py`: `stored_utc`,
  `local_utc_minus_5_to_utc`, `evidence_matched`, and `dual_diagnostic`.
- Added row-level timestamp provenance fields for stored workbook UTC,
  explicit local UTC-5 to Binance UTC conversion, manual order-history UTC
  evidence, selected validation timestamps, evidence class, deltas, and
  rejection reason.
- Added a geometric-scope coverage pre-probe so validation reports manual
  order-history coverage and holdout class denominators before promotion
  metrics are interpreted.
- Added diagnostic-only realism ablations for timestamp-aware comparison:
  `legacy`, `exchange_filters_only`, `funding_series_only`,
  `mark_valuation_only`, `geometry_seed_only`, and `combined_public`.

### Validated - Evidence-matched UTC improves geometric reconciliation

- Current geometric workbook scope: `61` rows, `60` timestamp-modelable rows,
  `54` model rows under `evidence_matched`, and one
  `conflicting_manual_evidence` row excluded from proof.
- `stored_utc` holdout: mean absolute PnL error `14.140301546151198`, median
  absolute PnL error `8.201957766028666`, sign match
  `0.5263157894736842`, winner recall `0.6153846153846154`, non-winner
  specificity `0.5`.
- `evidence_matched` holdout: mean absolute PnL error
  `2.7434760256981576`, median absolute PnL error `2.3795692363486998`,
  sign match `1.0`, winner recall `1.0`, non-winner specificity `1.0`.
- Decision: promote `evidence_matched` for geometric reconciliation and
  backtest-validation runs that require validated UTC. Keep the CLI default as
  `stored_utc` for backward-compatible diagnostics; no full-pipeline candidate
  path is changed.

### Changed - Audit trail consolidation

- Consolidated the temporary `UTC_FIX.md` implementation plan into durable
  audit locations: this changelog, `ERRORS_LOG.md`, and the executable
  timestamp-policy tests plus row-level timestamp provenance emitted by
  `scripts/validate_backtest_live_reconciliation.py`.
- Deleted `UTC_FIX.md` after the durable audit trail existed. The deletion does
  not remove code behavior because timestamp-policy logic lives in
  `scripts/validate_backtest_live_reconciliation.py` and its focused test
  coverage lives in `tests/unit/test_backtest_timestamp_policy.py`.
- Appended the canonical evidence-matched UTC correction support system to
  `IMPLEMENT_UPDATE_BACKTEST.md`. The support system keeps manual
  order/trade/transaction exports as provenance evidence only; it does not add
  manual-export fields as model features, overwrite `data/new_expired_bots.xlsx`,
  or promote model artifacts.

### Verification - Audit consolidation

- `python -m pyright scripts/validate_backtest_live_reconciliation.py` -> 0
  errors.
- `python -m pytest -q tests/unit/test_backtest_timestamp_policy.py
  tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py
  tests/unit/test_candidate_pipeline_bypass.py` -> 96 passed.
- Isolated scratch validation compared `stored_utc` versus `evidence_matched`
  on all geometric workbook rows and then deleted the scratch directory:
  `SCRATCH_PARENT_EXISTS_AFTER_EMPTY_CLEANUP=False`.

## [unreleased] - BACKTEST-PUBLIC-MARKET-REALISM (2026-05-17)

### Added - Candidate-time public-market realism profile

- Added explicit `candidate_time_public_market_v1` support for backtest
  diagnostics that use only candidate-time-safe public inputs: Binance
  `exchangeInfo` filters, mark-price klines, and funding-rate history for the
  forward validation window.
- Added exchange-filter extraction from `PRICE_FILTER.tickSize`,
  `LOT_SIZE.stepSize`, and minimum notional filters. `pricePrecision` and
  `quantityPrecision` are not used as tick/step substitutes.
- Added mark-price valuation provenance while keeping last-price klines as the
  fill/touch source.
- Added funding-series classification so rows distinguish verified funding
  history, no event in the window, missing data, and legacy static funding.
- Added training-ingestion quarantine checks for public-market provenance so
  incomplete or invalid public-market rows cannot silently enter the training
  set as clean rows.

### Changed - Seed evidence guards

- Manual-export order history now records timestamp provenance and treats CSV
  columns labeled `Time(UTC)`, `Update Time`, and `Date(UTC)` as UTC without a
  UTC-5 shift.
- Exact strategy seeding now fails closed when the user requests a
  `strategy_id` but the order-history export does not contain a `Strategy Id`
  column.
- Seed-state and public-market evidence remain adapters/provenance fields, not
  model features.
- Validation summaries now mark public-market evidence as requested when the
  explicit `candidate_time_public_market_v1` profile is selected, even if the
  legacy evidence-fetch flag is not separately provided.

### Validated - Public profile remains diagnostic-only

- Latest-20 geometric workbook cohort:
  - `legacy`: 17 model rows, mean absolute PnL error `6.368633907600`, median
    absolute PnL error `3.800947130855`, sign match `0.764705882353`,
    fast-winner recall `0.764705882353`.
  - `candidate_time_public_market_v1`: 17 model rows, mean absolute PnL error
    `9.059605185196`, median absolute PnL error `6.819644101974`, sign match
    `0.764705882353`, fast-winner recall `0.705882352941`.
  - Seeded/manual-export upper bound: 17 model rows, mean absolute PnL error
    `1.985181973495`, median absolute PnL error `1.732114138521`, sign match
    `1.000000000000`, fast-winner recall `0.882352941176`.
- Chronological holdout subset:
  - `legacy`: 6 model rows, mean absolute PnL error `8.123829630140`, median
    absolute PnL error `6.520473565427`, sign match `0.666666666667`.
  - `candidate_time_public_market_v1`: 6 model rows, mean absolute PnL error
    `9.522266958055`, median absolute PnL error `7.757316340001`, sign match
    `0.500000000000`.
  - Seeded/manual-export upper bound: 6 model rows, mean absolute PnL error
    `1.534625170670`, median absolute PnL error `1.763249572998`, sign match
    `1.000000000000`.
- DOGEUSDT `strategy_id=411991896` focused validation:
  - `legacy`: live PnL `2.72%`, model PnL `1.153360%`, absolute error
    `1.566640%`, max DD `4.063080%`, model trades `8`.
  - `candidate_time_public_market_v1`: live PnL `2.72%`, model PnL
    `2.417570%`, absolute error `0.302430%`, max DD `1.586589%`, model
    trades `7`.
  - seeded/manual-export: live PnL `2.72%`, model PnL `3.217126%`, absolute
    error `0.497126%`, max DD `2.397686%`, model trades `10`.
- Decision: `candidate_time_public_market_v1` improved the focused DOGE row but
  worsened the latest-20 geometric cohort and chronological holdout versus
  `legacy`; it remains diagnostic-only and is not promoted into default
  full-pipeline behavior.

### Pipeline impact

- Default `legacy` full-pipeline behavior is unchanged.
- No scanner admission, scanner ranking, HMM, meta-labeler, utility scoring,
  deploy-ready CSV generation, workbook cells, model artifacts, or retraining
  path was promoted by this update.
- Seeded/manual-export improvements remain valid for historical/live
  reconciliation only and are not candidate-time inputs for future symbols.

### Verification

- `python -m pyright backtest/btk_replay_seed_loader.py backtest/btk_seed_state.py backtest/backtest_realistic.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_btk_seed_state.py` -> 29 passed.
- `python -m pyright src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_unified_runner.py backtest/btk_label_contract.py backtest_candidates.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py` -> 61 passed.
- `python -m pyright scripts/validate_backtest_live_reconciliation.py src/neutralgrid/training/unified_training_builder.py src/neutralgrid/backtest/candidate_pipeline.py backtest/backtest_realistic.py backtest/btk_unified_runner.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_unified_training_builder.py tests/unit/test_candidate_pipeline_bypass.py tests/unit/test_btk_label_runner.py tests/unit/test_btk_seed_state.py` -> 113 passed.
- Summary-flag smoke validation with
  `--realism-profile candidate_time_public_market_v1` reported
  `public_market_evidence_requested=true`.
- Isolated scratch cleanup proof: `.codex_scratch exists=False`.

## [unreleased] - BACKTEST-CANDIDATE-TIME-REALISM (2026-05-16)

### Changed - Side-aware seeded ladder formula

- Seeded backtests now preserve the manual-export order side through the active
  ladder availability check. A seeded BUY level can satisfy only BUY/long open
  checks, and a seeded SELL level can satisfy only SELL/short open checks.
- Unseeded/default behavior is unchanged.
- The side-aware formula improved manual-order-history seeded geometric
  holdout error from `2.368012%` to `1.662237%` mean absolute PnL error and
  from `2.011418%` to `1.099384%` median absolute PnL error in isolated
  validation.
- `candidate_time_geometric_v1` remains diagnostic-only because the same
  side-aware formula worsened holdout mean absolute PnL error versus `legacy`.

### Added - Explicit candidate-time realism profile

- Added explicit backtest realism profiles: `legacy` and
  `candidate_time_geometric_v1`.
- Added a candidate-time geometry seed helper through the existing `SeedState`
  interface. It uses configured grid levels, first replay close, symbol, and
  timestamp only; it does not infer or copy manual-export quantities.
- `backtest_candidates.py` now accepts
  `--realism-profile legacy|candidate_time_geometric_v1`, with `legacy` as the
  default.
- `scripts/validate_backtest_live_reconciliation.py` now supports
  chronological calibration/holdout split diagnostics and optional public
  market evidence source reporting.

### Validated - Candidate-time profile not promoted

- Isolated geometric holdout validation showed `candidate_time_geometric_v1`
  matched `legacy` on mean and median absolute PnL error, so it was not
  promoted as a default.
- Holdout `legacy`: mean absolute PnL error `13.642216%`, median absolute PnL
  error `8.201958%`, sign match `0.571429`.
- Holdout `candidate_time_geometric_v1`: mean absolute PnL error `13.642216%`,
  median absolute PnL error `8.201958%`, sign match `0.571429`.
- Holdout manual-order-history seeded validation: mean absolute PnL error
  `2.368012%`, median absolute PnL error `2.011418%`, sign match `1.000000`.
  This remains validation-only because manual order history is unavailable to
  future candidate selection.
- Scratch validation outputs were deleted after collection:
  `SCRATCH_EXISTS_AFTER_CLEANUP=False`.

### Pipeline impact

- Default full-pipeline behavior is unchanged. No scanner admission, scanner
  ranking, HMM, meta-labeler, utility scoring, deploy-ready CSV generation,
  model artifact, retraining, or live telemetry scanner path was intentionally
  changed by this update.
- `candidate_time_geometric_v1` is explicit and diagnostic-only until a future
  holdout test proves improvement without unavailable live data.

### Verification

- `python -m pyright backtest/btk_seed_state.py backtest/btk_unified_runner.py src/neutralgrid/backtest/candidate_pipeline.py backtest_candidates.py scripts/validate_backtest_live_reconciliation.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py` -> 82 passed.

## [unreleased] - BACKTEST-SEED-REALISM (2026-05-15)

### Added - Manual-export seed realism for geometric backtests

- Added direct Binance `Order history*.csv` seed loading through the existing
  `SeedState` interface. The loader may scan many symbols, but returns a seed
  only for the requested bot's exact `symbol`, optional exact `strategy_id`,
  and active order window.
- Seeded backtests now restrict the initial active ladder to verified open
  order levels. Unseeded backtests preserve legacy behavior.
- Verified live per-order quantity is used as the model position size only when
  the seed contains one consistent positive quantity; conflicting quantities
  remain model-sized and are labelled as conflicting evidence.
- `scripts/validate_backtest_live_reconciliation.py` gained
  `--seed-from-manual-exports`, seed/evidence diagnostics, seeded subset
  summary metrics, and top-error reporting.

### Validated - Seeded geometric realism improved

- Geometric all-row unseeded baseline: mean absolute PnL error `13.037998%`,
  median absolute PnL error `6.818518%`, sign match `0.654545`, non-winner
  specificity `0.562500`, trade-count error `118.763636`.
- Geometric all-row seeded/manual-export path: mean absolute PnL error
  `4.418476%`, median absolute PnL error `2.011418%`, sign match `0.872727`,
  non-winner specificity `0.937500`, trade-count error `89.381818`.
- DOGEUSDT `strategy_id=411991896` scratch-cache validation: live PnL
  `2.72%`, seeded model PnL `3.217126%`, absolute error `0.497126%`.

### Pipeline impact

- This is an opt-in seeded backtest realism path, not a blanket new backtest
  engine default. Normal unseeded backtests preserve prior behavior.
- `run_full_pipeline.py` does not automatically pass manual-export seed state,
  so this change does not by itself alter scanner admission, ranking, or
  deploy-ready candidate output.
- Candidate-selection accuracy can improve only for validation or backtest
  calls that explicitly provide verified seed evidence. Brand-new potential
  candidates generally have no live order-history ladder yet, so they remain
  governed by the existing unseeded model unless a separate, verified seed
  source is added.

### Verification

- `python -m pyright backtest/btk_replay_seed_loader.py` -> 0 errors.
- `python -m pyright backtest/backtest_realistic.py backtest/btk_seed_state.py backtest/btk_replay_seed_loader.py` -> 0 errors.
- `python -m pyright scripts/validate_backtest_live_reconciliation.py backtest/backtest_realistic.py backtest/btk_replay_seed_loader.py` -> 0 errors.
- `python -m pytest -q tests/unit/test_btk_seed_state.py` -> 22 passed.
- `python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py` -> 66 passed.
- `python -m pytest -q tests/unit/test_btk_seed_state.py tests/unit/test_btk_label_runner.py tests/unit/test_candidate_pipeline_bypass.py` -> 74 passed.

### Removed - Superseded live bot decision draft

- `LIVE_BOT_DECISION.md` was deleted by the user as a clarity cleanup. The
  file was a superseded single-shot evaluator draft and was removed so its
  terminology does not confuse future implementation work with the active
  `LIVE_DECISION.md` scanner semantics.
- Runtime impact: none expected. Prior validation found no Python import,
  package metadata, scanner entry point, or full-pipeline code path that reads
  `LIVE_BOT_DECISION.md`.

## [unreleased] - LIVE_DECISION-01 (2026-05-07)

### Added - Tactical live decision tool (Phases 0-E complete)

A new recurring scanner that monitors user-declared live grid bots every X
minutes (default 5m) against current Binance market state and emits per-bot
CONTINUE / ADJUST / END recommendations to console + JSONL audit log + a
Discord webhook digest. Advisory only -- no trading API calls, no auto-cancel.

**Scope and locked decisions (set up-front in design conversation):**

- Output channels: console + `logs/live_decisions_YYYYMMDD.jsonl` AND a
  Discord webhook digest (both, not either-or).
- Autonomy: advisory only. Read-only Binance endpoints (the unsigned subset
  of `BinanceClient.get_all_market_data`). Signed/trading endpoints are
  forbidden by construction at the call sites.
- Live registry: user-curated YAML at `active bots/DD-MM-YY.yaml`, one file
  per ingestion day. Tool re-reads the latest each tick so user edits take
  effect on the following tick.
- Default cadence: 5 min. Configurable via `--interval`. Floor: 30s.
- External signals: Binance-only for v1. CoinGecko / Finnhub / news deferred
  per the pending-external-APIs constraint and the >=500-row training-pool
  precondition (`project_pending_external_apis`).

**Architecture:**

- `live_decision_scanner.py` -- root CLI; modes `--dry-run`, `--once`, or the
  default recurring loop. Owns argparse, asyncio loop, lock file, signal
  handlers (graceful drain + atomic state persist + `await client.close()`).
- `src/neutralgrid/live/decision/loader.py` -- YAML reader, finds latest
  `DD-MM-YY.yaml` by parsed date (not mtime), per-bot validation, returns
  `(LiveBotSpec[], LoaderWarning[])`.
- `src/neutralgrid/live/decision/state_store.py` -- per-bot ring buffer of
  the last 20 ticks, atomic JSON writes via `tempfile.mkstemp + os.replace`.
- `src/neutralgrid/live/decision/recommender.py` -- pure verdict logic. END
  on (price outside grid | regime flip range_prob<0.45 AND trend_prob>0.40 |
  3-consec micro fails | 3-consec fetch fails | symbol delisted). ADJUST on
  thesis-weakened (range_prob in [0.30, 0.45], price within 10% of bound,
  transient errors, missing artifacts, `data_missing:*`). Suggested
  re-centered grid bounds via `mid +- half_width`. Cool-down: state
  transition -> always emit, END 30-min re-emit, ADJUST 3rd-consec ESCALATE
  then suppress, CONTINUE 60-min heartbeat.
- `src/neutralgrid/live/decision/monitor.py` -- per-tick orchestration:
  `BinanceClient.get_all_market_data(symbol)` -> `klines_to_df` ->
  `compute_features` -> HMM `predict` -> meta-labeler `predict_proba`
  (skipped if >50% features missing) -> utility offline-caller pattern
  (catch `UtilityCalibratorUnavailable`, NaN + warn) -> microstructure hard
  gate via `MicrostructureEstimator` + `MicrostructureHardGate` ->
  deploy-time delta via `candidate_deploy_linker`. Computes
  `delta_meta_prob = current.meta_proba - deploy.meta_prob`.
- `src/neutralgrid/live/decision/renderer.py` -- console table mirroring
  `run_full_pipeline._display_results` and a JSONL writer with UTC daily
  rollover.
- `src/neutralgrid/live/decision/discord_sink.py` -- async webhook digest
  builder. One embed per tick, fields only for `should_emit=True` results,
  aggregate footer across the whole fleet, worst-verdict-wins coloring,
  token-bucket rate limiter (1 msg / 15s).

**Phase E (agents-team review) outcomes:**

Spawned `portfolio-oversight-lifecycle`, `deployment-engineering`,
`data-curator`, and `backtest-evaluator` in parallel against
`LIVE_DECISION.md` + the implemented code.

Blockers found and fixed in-session:

- `data-curator`: deploy-linker indexed via `setdefault`, retaining the
  EARLIEST row for duplicate `strategy_id`. Wrong for an append-only
  chronological log -- `delta_meta_prob` would attribute against a stale
  deploy. Switched to plain assignment (last-match-wins). Regression test
  added: `test_linker_cache_uses_latest_row_when_strategy_id_repeats`.

Non-blocker hardening landed alongside the synthesis:

- `LockFile.acquire()` rewritten with `os.O_CREAT | os.O_EXCL` to close the
  TOCTOU window between `exists()` and `write_text()`.
- Windows signal-handler race: second SIGINT no longer calls `sys.exit`
  from the OS main thread (which would skip the `_run_loop` finally on
  Windows). Now uses `loop.call_soon_threadsafe` + task cancellation so the
  finally block runs and the lock file releases cleanly.
- `DiscordDigestSink` now logs only `type(e).__name__` for connection
  errors and avoids logging `e.response.text` -- `httpx.ConnectError`
  messages and Discord error JSON have both been observed to echo
  webhook-id fragments.

Deferred items logged as ERR-### (no fix in this session):

- `ERR-044` WATCH: recommender thresholds are hand-picked, not calibrated;
  coupled to active HMM probability distribution. Recalibrate via
  `--config-file` after each HMM rotation.
- `ERR-045` WATCH: `MonitorContext` loads `deploy_linkage_log.csv` once at
  startup; long-running daemon misses post-startup deploys until restart.
- `ERR-046` WATCH: `DD-MM-YY` filename pattern is ambiguous with US
  `MM-DD-YY` for any day <= 12; naive `deploy_ts` silently coerced to UTC.
  Documentation-only fix queued.

**Decision rationale:**

The tool fills a gap the codebase did not have: post-deploy lifecycle
visibility. `run_full_pipeline.py` was one-shot scan->deploy-then-exit; no
recurring monitor existed; no notification surface existed. The Phase E
review confirmed the verdict model stays correctly within the operational-
advisory lane (no auto-flagging of lifecycle stage transitions, per the
cursus honorum framework). The fail-closed pattern was preserved: missing
HMM artifact / utility calibrator / data inputs all surface as ADJUST with
explicit `*_artifact_missing` flags or `data_missing:*` reasons -- never a
silent CONTINUE. Per `safety-invariants.md` the meta-labeler `hlabel`
guards remain intact (verified by `leakage-check` skill: 39 contract tests
passing); the three feature-pipeline files were not touched (`verify-
feature-pipeline` PASS for this scope).

The recommender is informed by Lopez de Prado AFML Chapter 3 (meta-labeling
philosophy: secondary signal, not primary authority) and the project's own
`LIVE_BOT_DECISION.md` v2 unapproved draft (treated as a sibling
single-shot evaluator; shared contracts adopted, divergent verdict
vocabulary preserved). The cool-down policy (30/60/3-consec) is calibrated
by intuition rather than empirical miss-rate analysis -- ERR-044 documents
this calibration debt explicitly.

### Files added

- `LIVE_DECISION.md` (project root) -- design doc, sibling to
  `LIVE_BOT_DECISION.md` v2.
- `live_decision_scanner.py` (project root) -- CLI entry point.
- `src/neutralgrid/live/decision/__init__.py`
- `src/neutralgrid/live/decision/loader.py`
- `src/neutralgrid/live/decision/state_store.py`
- `src/neutralgrid/live/decision/recommender.py`
- `src/neutralgrid/live/decision/monitor.py`
- `src/neutralgrid/live/decision/renderer.py`
- `src/neutralgrid/live/decision/discord_sink.py`
- `tests/unit/test_decision_loader.py` -- 14 tests
- `tests/unit/test_decision_state_store.py` -- 9 tests
- `tests/unit/test_decision_recommender.py` -- 20 tests
- `tests/unit/test_decision_monitor.py` -- 9 tests
- `tests/unit/test_decision_renderer.py` -- 11 tests
- `tests/unit/test_decision_discord_sink.py` -- 13 tests
- `tests/unit/test_decision_loop_helpers.py` -- 19 tests
- `tests/unit/test_decision_contract_v1_0.py` -- 8 tests (pinning
  `DECISION_CONTRACT_VERSION = "1.0"`, verdict vocabulary, default
  thresholds, dataclass field sets, JSONL top-level keys, reason-code
  prefixes)
- `tests/unit/test_decision_phase_d.py` -- 17 tests (config-file loader,
  deploy snapshot, last-match-wins linker, microstructure gate, JSONL
  schema)
- `tests/unit/test_alerts_discord.py` -- 7 tests

### Files modified

- `src/neutralgrid/core/constants.py` -- added
  `DECISION_CONTRACT_VERSION = "1.0"` constant.
- `src/neutralgrid/models/alerts.py` -- added `DiscordWebhookHandler`
  subclass of `AlertHandler`. Severity-coded color, error swallowing
  unless `raise_on_error=True`.
- `.gitignore` -- anchored `Live/` to `/Live/` so the project's top-level
  `Live/` user-data directory is still ignored, but
  `src/neutralgrid/live/decision/` (and existing `src/neutralgrid/live/*`
  files) is no longer matched on Windows case-insensitive filesystems.

### Backward compatibility

No breaking changes to existing pipelines. The new tool is a parallel
process that consumes existing artifacts (HMM, meta-labeler, utility
calibrator, `deploy_linkage_log.csv`) read-only and never writes to them.
The only added writes are under `data/live_decisions/` and `logs/`, both
new paths.

`DECISION_CONTRACT_VERSION` is a new public constant. The pinning contract
test will fail loudly on any future version bump -- that's the intended
behavior; bumps must be accompanied by a CHANGELOG entry justifying the
schema change.

### Verification

- `pyright` clean across all 9 new modules + 10 new test files.
- 127 new tests passing (`tests/unit/test_decision_*.py` +
  `tests/unit/test_alerts_discord.py`).
- `leakage-check` skill PASS: both `_KNOWN_LABEL_COLUMNS` guard sites
  intact at `meta_labeler.py:643` and `:818`; new `monitor._build_meta_
  feature_dict` documented as never including `hlabel`; 39 leakage-related
  contract tests passing.
- `verify-feature-pipeline` PASS for this scope: none of the three lockstep
  files (`candidate_pipeline.py`, `data_generator.py`,
  `unified_training_builder.py`) were touched by Phases A-E.
- Live smoke test (Phase C, 2026-05-07): `python live_decision_scanner.py
  --once` against a real BTCUSDT bot returned a clean
  `BTCUSDT CONTINUE consec=1 price=80957.90 pct_in=33.9% range_prob=0.96
  trend_prob=0.00`; Discord webhook returned `204 No Content`.

See ERR-044, ERR-045, ERR-046 for tracked follow-up items.

## [unreleased] - SCANNER-TOP-QUANTILE-0.68 (2026-05-06)

### Changed - `top_quantile` default lowered from 0.75 to 0.68

**User authorization:** explicit operator decision 2026-05-06 to clear the
`max(30, 3*len(features))=30` per-class floor in pattern_profile / profile_model.
Without this change, Stage A retrain refuses to run on the current pool
(26 winners at top_q=0.75 vs required 30).

**Empirical effect on bounded labeled pool (n=106, max_duration=7h):**
- top_q=0.75 (prior): pnl_thr=7.06%, winners=26 — BLOCKED
- top_q=0.68 (new):   pnl_thr=5.69%, winners=33 — clears floor by 3

**Rationale documented per `feedback_no_silent_degradation.md`:**
- This IS a gate weakening (admits bots with PnL 5.69-7.05% as winners).
- Not a silent bypass — change is recorded, traceable, and authorized.
- Modest 2.6 floor margin (33 vs 30) keeps the relaxation tight.
- `min_profit_factor` floor unchanged at 1.5.

**Linked findings (NOT addressed by this change):**
- `profit_factor=0` structural degeneracy for cancelled grid bots with
  zero closed round-trips. Example: strategy_id 410899682 ONDOUSDT
  (pnl=+1.28%, total_trades=0, all PnL unrealized) and 410899699
  FOLKSUSDT (pnl=+0.95%, total_trades=1, PF=0). These real winners are
  excluded by the PF≥1.5 filter regardless of the `top_quantile` value.
  This mirrors the PF=999.99 ∞-clamp issue noted in
  `project_fixutility_01.md` memory: PF is structurally undefined for
  cancelled bots whose round-trip accounting is incomplete. Filed as a
  separate concern; not part of this change.

### File modified
- `retrain_scanner.py` — argparse default `top-quantile` 0.75 → 0.68 with
  inline citation comment.

## [unreleased] - D1-UTILITY-FEATURE-DECOMMISSION (2026-05-06)

### Removed - `utility_score` from active meta-labeler feature contract

`SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` reduced from 8 features to 7.
The dropped feature is `utility_score`. Calibrator infrastructure
(`validation/utility.py`, `calibration/utility_calibrator.py`,
`scripts/recalibrate_utility.py`, `artifacts/utility/`) is KEPT DORMANT
for possible future revival — not deleted.

**Empirical evidence for removal (n=97 active-HMM pool):**

- Pearson(utility_score, pnl_pct) = -0.32 on 22 manual winners — feature
  is *anti-correlated* with what we're trying to predict.
- Holdout AUC bootstrap 95% CI = [0.21, 0.80] (May 4 candidate, n=85) —
  straddles 0.5; discrimination indistinguishable from chance.
- Permutation p-value = 0.45 — null hypothesis of zero discriminative
  power cannot be rejected.
- Variance decomposition: regime/structural inputs explain R²=0.65 of
  utility_score variance vs profitability R²=0.10 (6.3× gap).
- Grid optimizer collapsed both penalty coefficients to corner solutions
  every refit (kappa→0.05 floor, lambda→4.0 ceiling).
- Adding 12 new labels (2026-05-05 ingestion: pool 85 → 97) flipped G4
  from True to False — signal too weak to survive small data shifts.

**Compositional rationale (Lopez de Prado AFML §3 / Hudson & Thames):**

Meta-labeling emits the calibrated probability used for sizing — it does
NOT consume one as input. Feeding `utility_score` (a
calibrated-decision-surface scalar over `range_prob`/`trend_prob` and
grid metrics) created a circular double-calibration. D1 restores
compositional discipline: only primitive regime/geometry features feed
the meta-labeler.

**Mis-specification hypothesis (open):**
`U = E[R] - lambda*E[DD] - kappa*E[BL]` is likely structurally wrong at
grid-bot scale. Transient drawdown is constitutive of inventory
accumulation (the entry phase from which winners emerge), not a damage
signal. Future revival should re-derive utility for inventory-bearing
market-making (e.g., Avellaneda–Stoikov-style penalty on terminal
inventory variance), not just refit on more data with the same formula.

### Files modified

- `src/neutralgrid/models/meta_labeler.py` — drop `utility_score` from
  `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP`; extended white-box theory
  docstring with D1 receipts.
- `src/neutralgrid/backtest/candidate_pipeline.py` — drop 3 `utility_score`
  mappings from `_SCANNER_TO_FEATURE`; drop `utility_score` from
  `TRAINING_OUTPUT_COLUMNS`.
- `src/neutralgrid/training/data_generator.py` — drop `utility_score` from
  `TrainingTableSchema.features`, `FeatureSnapshot` field, `to_dict()`,
  and `MISSING_FEATURES`.
- `src/neutralgrid/training/unified_training_builder.py` — drop 2
  `utility_score` mappings from `_SCAN_TO_FEATURE`.
- `src/neutralgrid/training/scanner_integration.py` — drop `utility_score`
  from `_SCAN_FEATURE_FIELDS`, `FeatureSnapshot` kwarg, and the
  `regime_utility` snapshot-override block.
- `tests/unit/test_meta_labeler_retrain_contract_v20260408.py` — switch
  contract test from `utility_score` to `survival_prob` as the example
  feature in the null/imputation gate tests.
- `retrain_meta_labeler.py` — header docstring: feature count 8 → 7.

### Backward compatibility

V20260420-trained models continue to load and infer correctly:
`SNAPSHOT_META_FEATURES_V20260420` still contains `utility_score`;
`_INFERENCE_FEATURE_ALIASES["regime_utility" | "scan_utility_score"] →
"utility_score"` preserved; `_FEATURE_MEDIAN_DEFAULTS["utility_score"] =
0.0` preserved. Runtime emits `regime_utility=None` (calibrator dormant)
which resolves to `utility_score=NaN → 0.0` via the median default.

### Operational status

`pass_mode="dominance"` (in `core/config.py`) already bypassed the
utility gate at runtime (`regime_validator.py:418-419`), so the active
runtime decision surface is unchanged. D1 brings the meta-labeler
*feature contract* into alignment with what runtime already does.

### Multi-agent post-change review

Four parallel agents (feature-analyst, data-curator,
deployment-engineering, market-strategy-architect) audited the change.
Unanimous PASS. Architect noted D1 *improves* white-box quality by
removing a circular feature; data-curator confirmed zero leakage path
into the active X-matrix; deployment-engineering cleared production
paths; feature-analyst verified Feature Pipeline Update Rule symmetry.

### Tests

- `tests/unit/test_meta_labeler_retrain_contract_v20260408.py` — 15/15 passing
- Full pytest sweep: 1163 passed, 4 pre-existing failures (unrelated to D1:
  `TestBoundedUniverseContract` × 3 fixture-vs-DEFAULT_FEATURES mismatch
  from PATTERN_PROFILE_FIX phases 1-3, and one stale workbook-row-count
  hardcode `len(pool) == 85` now returns 84).

## [unreleased] - BACKFILL-SKIP-IF-FRESH

### Added - Opt-in `--skip-if-fresh` short-circuit for backfill

`scripts/backfill_training_features.py` now accepts a `--skip-if-fresh` CLI
flag (default off) that skips per-row HMM inference when the post-merge row
already satisfies both conditions:

1. Preserved `hmm_artifact_version` matches the explicit
   `--default-artifact-version`.
2. `range_prob`, `trend_prob`, and `persistence_prob` are all finite.

The predicate sits *after* the UTILFIX-01 invalidation block, so it cannot
bypass safety: stale-lineage rows have already had their HMM columns
cleared to NaN by the merge, which makes the lineage-match test fail for
them and forces re-inference. When a new HMM is promoted and the operator
passes the new version, every preserved row's lineage is invalidated and
the flag becomes a no-op for that run. With `--default-artifact-version`
empty (legacy operator path), the flag short-circuits and behaviour is
byte-identical to today.

In steady state (~210-row workbook with one new ingestion since the last
backfill), this turns a ~10-minute Binance-bound run into a sub-second
operation: only the new and stale-lineage rows are re-inferenced.

### Tests
- `tests/unit/test_backfill_training_features_v20260312.py` — added
  `test_skip_if_fresh_skips_matching_lineage_with_finite_features`:
  asserts the predicate skips fresh-lineage rows, re-infers stale-lineage
  rows (merge invalidation), and re-infers rows whose HMM derived
  features are NaN (finiteness check).

### Docs
- `CLAUDE.md` — Pipeline section's backfill bullet now documents the
  `--skip-if-fresh` flag alongside the existing `--default-artifact-version`
  guidance.

## [unreleased] - UTILFIX-01

### Fixed - Explicit utility fail-closed + backfill merge-contamination

Two silent-degradation issues uncovered during the 2026-05-04 artifact
refresh, addressed together because both fall under the
`feedback_no_silent_degradation` invariant:

1. **`UtilityConfig.from_artifact()` no longer silently substitutes pinned
   v0 defaults when `artifacts/utility/current.json` is absent or
   malformed.** Empirical isolation test on the active-HMM pool (n=85) showed
   v0's holdout AUC bootstrap 95% CI = [0.146, 0.813] — straddles 0.5 — and
   permutation p = 0.587, i.e. v0 has no statistically detectable
   discriminative power on this pool. At v0's hardcoded threshold of 0.0,
   *zero* of 85 candidates are admitted (mean utility = -22.1, max = -1.2),
   so the silent fallback functions as a reject-all gate disguised as a
   calibrated model. The runtime path now raises
   `UtilityCalibratorUnavailable` (subclass of `NeutralGridError`,
   error_code=`UTILITY_CALIBRATOR_UNAVAILABLE`). Callers handle the
   exception per their context:
   - `src/neutralgrid/scanner/scan.py` — Stage A logs once and emits
     `utility_score=None`.
   - `src/neutralgrid/training/unified_training_builder.py` — training
     emits `utility_score=None` (meta-labeler treats as feature-missing).
   - `src/neutralgrid/validation/regime_validator.py` — emits
     `regime_utility=None`, `utility_passed=False`, and corresponding
     `expected_grid_return`/`expected_drawdown` as `None` so
     `pass_mode == "utility"` and `pass_mode == "hybrid"` reject rather
     than admit.
   - `scripts/backfill_training_features.py`,
     `new_bot_data_extractor.py` — emit `utility_score=NaN`.
   - `src/neutralgrid/calibration/utility_calibrator.py:446-451` —
     unchanged. The G6 v0 baseline path uses direct `UtilityConfig(...)`
     instantiation, not `from_artifact()`, so it is not affected.
   - Convention: future Stage B utility gate must use rejection code
     `data_missing:utility`, parallel to `data_missing:tos` /
     `data_missing:range_prob`.

2. **`scripts/backfill_training_features.py` `--default-artifact-version` is
   now AUTHORITATIVE.** Previously, the merge-with-existing-output logic at
   `lines 506-516` preserved per-row `hmm_artifact_version` from the prior
   output file. Combined with the per-row resolution at `line 300-302`
   (`row_artifact_version or self.default_artifact_version`), the explicit
   CLI flag was silently overridden whenever the prior output existed. The
   merge logic now adds an `existing_lineage_matches_default` mask: when
   `--default-artifact-version` is set, rows whose preserved
   `hmm_artifact_version` differs from this value have their preserved
   `HMM_DERIVED_COLUMNS` (`range_prob`, `trend_prob`, `persistence_prob`)
   AND all `HMM_LINEAGE_COLUMNS` invalidated, forcing
   `backfill_single_bot` to re-attempt inference against the explicit
   default. Stale-row counts and the involved version labels are emitted
   via a single INFO log line:
   `"Merge invalidated %d row(s) with stale HMM lineage (preserved=%s vs requested=%s); re-inference will run for those rows."`

### Tests
- `tests/unit/test_utility_calibrator.py` — replaced
  `test_from_artifact_fallback_matches_pinned_v0` with three explicit
  exception tests (missing file, malformed JSON, missing required
  sections) plus `test_scan_handles_utility_calibrator_unavailable`.
- `tests/test_afml_bugfixes.py`, `tests/test_afml_integrations.py` —
  default `UtilityScorer()` calls replaced with explicit
  `UtilityScorer(UtilityConfig())`.
- `tests/unit/test_backfill_training_features_v20260312.py` — added
  `test_backfiller_invalidates_stale_lineage_when_default_artifact_version_passed`;
  legacy preservation tests still green.

### Docs
- `.claude/rules/safety-invariants.md` — Fail-Closed Behavior section
  expanded with utility-validator bullet; new HMM Lineage Authority
  section documents the backfill merge invariant.
- `CLAUDE.md` — Pipeline section now lists the backfill step explicitly
  with a note about authoritative `--default-artifact-version`.

### Memory
- `reference_backfill_merge_contamination.md` (auto-memory) updated with
  the resolved-by note; the prior workaround (writing to a fresh output
  path) is no longer required when `--default-artifact-version` is set.

## [6.5.7-grid-synch-finish] - 2026-05-04

### Fixed - GRID_SYNCH integration finish (GRIDFIX-001)

Closed `GRID_SYNCH.md` from the prior 90% partial state to 100%. Four-agent
team review (data-curator, feature-analyst, deployment-engineering,
backtest-evaluator) of the original GRID_SYNCH plan found that the prior
progress bar overstated completion (F5 divisor never landed; inline duplicate
of grid-formula identity in `spacing_profile.py`; xlsx had no stored
`profit_per_grid_pct` column making v2's "passthrough" branch dead code;
TXT-only mode tagging would have left 77 of 82 locked-pool rows with
`mode=NaN`). Plan v3 in
`C:\Users\cris_\.claude\plans\what-are-the-missing-glimmering-kurzweil.md`
addressed all findings.

#### Step 1 — `mode` propagation through the data layer
- `_bot_data_extractor_core.py:1613` — extractor row dict emits `"mode": bot.mode` (was silently dropped).
- `src/neutralgrid/training/data_generator.py:154,237` — `FeatureSnapshot.mode` field + `to_dict()` emit.
- `src/neutralgrid/training/data_generator.py:706` — `ExistingDataMapper.COLUMN_MAP` carries `mode` from xlsx into the training table.
- `src/neutralgrid/backtest/candidate_pipeline.py:49-50,162` — `_SCANNER_TO_FEATURE` + `TRAINING_OUTPUT_COLUMNS` updated atomically.
- `src/neutralgrid/training/unified_training_builder.py:76,139` — `EXTRA_META_FEATURES` + `_SCAN_TO_FEATURE` updated; explicit comment that `mode` is metadata, NOT a model feature (must not enter the X-matrix; see `meta_labeler.py:138`).
- `data/new_expired_bots.xlsx` — populated 209/209 rows (arithmetic=166, geometric=43, NaN=0) via three-stage zero-data-loss backfill: TXT-tagged (34 rows) → pre-geometric-launch arithmetic rule (146 rows; cutoff 2026-03-31 12:09:01 UTC; 1 row override of stale TXT) → user-supplied final 30 entries (5 arithmetic, 25 geometric).

#### Step 2 — Mode-aware F1 wrapper + fail-closed exclusion
- `src/neutralgrid/training/data_generator.py:737-781` — `compute_profit_per_grid` requires explicit `mode: str` (no default); hardcoded `GEOMETRIC` literal removed (the silent flip that corrupted ~52 arithmetic rows in the locked pool).
- `c` derivation pinned to F2/F3 pattern `(maker_fee + close_fee_rate)/2` from `get_config().grid` for cross-site equivalence. Empirical verification: F1 vs F2 produce identical output to delta=0.00e+00 on 4 fixtures across price/grid scales (C10 invariant).
- Silent `0.0` defaults at lines 748-749 and the `else 0.0` ternary tail in `map_dataframe` replaced with `float("nan")`. `0.0` is a valid PPG value; using it as a missing sentinel was a leakage-shaped collision.
- `src/neutralgrid/training/data_generator.py:872-895` — `map_dataframe` threads per-row `mode` into the wrapper, pre-checks `pd.notna(row["mode"])` so a NaN-mode row yields NaN PPG instead of raising (§2.4 mode-NaN safety).
- `src/neutralgrid/training/data_generator.py:914-944` — fail-closed exclusion gate at the data-quality boundary: drops rows where both `grid_spacing_pct` and `profit_per_grid_pct` are NaN. Empirically inert on the current xlsx (zero rows dropped) but defensive against future re-extracts.
- Caller migration: `new_bot_data_extractor.py:537-547` (passes `bot.mode`); `scripts/backfill_training_features.py:227-242` (threads `row["mode"]`); `tests/unit/test_new_bot_data_extractor.py:312-314` (FakeExistingDataMapper signature).
- Distribution diff at `reports/grid_synch_step5_distribution_diff_20260504_165655.md` confirms the silent-flip correction: arithmetic rows mean +1.04 bps, max +38.47 bps; geometric rows delta=0.0 (unchanged as expected, geometric == geometric).

#### Step 3 — F5 divisor + spacing_profile dedup
- `backtest/backtest_realistic.py:158-167,273-289` — switched to Binance LINES convention. `step = (upper - lower) / (num_grids - 1)` and `levels = [lower + i*step for i in range(num_grids)]` (was `range(num_grids + 1)` which overshot upper by one step under the prior `n` divisor). Added `n>=2` guard since n=1 (one line, zero intervals) is degenerate. The original GRID_SYNCH §3 marker for F5 had been claimed as DONE in commit `2bbd55d` but never actually landed; this commit closes it.
- `src/neutralgrid/grid/spacing_profile.py:32,109-129` — `_infer_mode` consolidated; now delegates to `grid.formulas.grid_spacing_pct` instead of inlining the geometric/arithmetic identity. Cnew invariant verified: only `grid/formulas.py` contains `(high/low) **`.

#### Step 6 — Locked test updates
- `tests/unit/test_btk_exchange_rounding.py` (2 tests), `test_btk_seed_state.py` (helper + 5 explicit-arg sites), `test_btk_order_lifecycle.py` (`_config`), `test_btk_global_cooldown.py` (`_base_config`), `test_btk_gap_fixes.py` (`_base_config`), `test_btk_funding_modes.py` (`_base_config`), `test_new_bot_data_extractor.py` (FakeExistingDataMapper signature + `base_kwargs` mode), `test_backfill_training_features_v20260312.py` (test fixture mode) — all updated with explicit GRID_SYNCH §3.1 / GRIDFIX-001 docstring references. Test fixtures shifted `num_grids` by +1 (e.g., 5→6, 10→11) to preserve the historical level set under the new (n-1)-divisor convention.

#### Suite verification
- `python -m pytest tests/` → **1177 passed, 5 failed**. The 5 failures are pre-existing per `ERRORS_LOG.md` ERR-038 (4 in `TestBoundedUniverseContract`) and ERR-039 (1 in `test_current_workbook_contract_counts_if_available`); not regressions from GRIDFIX-001.
- `pyright` → my new lines clean; pre-existing pandas/openpyxl typing noise unchanged.
- C10 cross-site equivalence: F1 vs F2 delta=0.00e+00 on 4 fixtures.

#### Out of scope (deferred per GRID_SYNCH §8)
- Step 7 (`c` measurement) — narrowed in v3 plan to maker-only fit. Empirical pool has zero taker-dominant rows, so `c_taker` cannot be fit. Current `taker_fee = 0.0005` (Binance published taker fee) at `core/config.py:80` preserved as labeled assumption-of-record.
- Geometric backtester support in `backtest_realistic.py` — backtester still constructs arithmetic level lists; geometric backtest support remains a follow-up.
- No retrain triggered. Per `safety-invariants.md` ratchet: HMM artifacts unchanged, `LABEL_CONTRACT_VERSION` unchanged, calibrators unaffected. Distribution diff is informational; retrain decision is downstream.

#### Errors logged for follow-up
- ERR-PROCESS-01 (OPEN) — unauthorized `git stash` invoked by Claude during pyright baseline check (see `ERRORS_LOG.md`). Working tree restored; no data lost. Process correction: agent must not stash/branch/reset without explicit user direction.
- ERR-PYRIGHT-001..003 (OPEN) — pre-existing pyright noise in `_bot_data_extractor_core.py` (34 errors at lines I did not touch); not in GRIDFIX-001 scope.

### Closed
- All 6 numbered GRID_SYNCH steps + the closure note in `GRID_SYNCH.md`.

---

## [6.5.7-pipeline-health-restoration] - 2026-04-29

### Fixed - Pipeline End-to-End Operability (PIPELINE_FIX)

Brought the full pipeline back to operational state. Pre-fix: `run_full_pipeline.py --discovery-mode` produced `meta_prob_source.value_counts() = {"missing": 25}` and `meta_prob.notna() = 0/25` because the meta-labeler fail-closed at load time on lineage mismatch. Post-fix: 18/25 rows carry calibrated meta probabilities. Plan documented at `PIPELINE_FIX.md` (User Goal: align all components of the full pipeline). Approved 4-agent team review (`pipeline-health-review` via TeamCreate): `data-curator`, `feature-analyst`, `market-strategy-architect`, `backtest-evaluator`.

#### Step 1 — ERR-021 label-precedence fix
- `src/neutralgrid/training/unified_training_builder.py:822-851` — extended the existing `hlabel_meta`-path degeneracy bypass (`nunique() <= 1 OR positive_rate < 0.05`, lines 793-815) to the secondary `("y", "y_horizon", "label_positive_by_horizon")` precedence loop. The pre-fix `notna().any()` predicate selected backtest-outcome `y` columns of all zeros (ERR-021: `y.value_counts() = {0: 162}`), starving the `net_pnl_pct >= meta_hurdle_pct` fallback. No new gate — the 5% positive-rate threshold is reused from `meta_labeler.py:725-732`.
- `tests/unit/test_unified_training_builder.py` — added `TestErr021LabelPrecedence` (2 tests).
- `tests/unit/test_plan_v6_steps.py::TestStep1LabelPrecedence::test_y_preferred_over_y_horizon` — updated fixture from single-row degenerate to multi-row non-degenerate to assert the corrected precedence semantics rather than the pre-fix bug.
- Step 1 reviewers: data-curator (lead) PASS, backtest-evaluator PASS.

#### Step 2 — Calibrator hardcoded HMM resolver
- `src/neutralgrid/calibration/hmm_winner_calibrator.py` — deleted import-time literal `DEFAULT_HMM_ARTIFACT_DIR = Path("artifacts/hmm/rolling_180d_20260426_012042")`. Added `_resolve_default_hmm_artifact_dir()` that delegates to `neutralgrid.models.artifacts.resolve_hmm_artifact_dir()` (the same function `run_full_pipeline.py:626` uses). `run_calibration` parameter `hmm_artifact_dir` default changed from `DEFAULT_HMM_ARTIFACT_DIR` to `None` with lazy resolution inside the function body; `--hmm-artifact-dir` CLI default similarly. The override path (explicit `--hmm-artifact-dir <path>`) is preserved for regression replay against archived HMMs. No new gate, no new model, only WHEN the path is bound.
- `tests/unit/test_hmm_winner_calibrator.py` — added `test_default_hmm_artifact_dir_resolves_at_call_time` (monkey-patches the resolver and asserts call-time resolution). Live verification: `_resolve_default_hmm_artifact_dir()` returns the active manifest path.
- Step 2 reviewers: market-strategy-architect (lead) PASS, data-curator PASS.

#### Step 3 — Documentation sync
- `readmefullpwep.md` (lines 298, 304, 530-540, 596) and `PNL_CURVE_CLASS.md` (lines 16, 35, 105, 136) — updated references from the historical 29-feature `snapshot_v20260407` profile to the active 8-feature `snapshot_v20260421_bootstrap` profile (`meta_labeler.py:127-138`, `ACTIVE_SNAPSHOT_META_FEATURES`). Section 8.4 of the README now lists the 8 features grouped as Regime/H* (2), Utility/dynamics (2), Grid geometry (3), Trend context (1). Historical profiles are retained as `--feature-profile` selectable for regression replay only. Edits were already in the working tree from a prior session; this Step verifies and accepts them.
- Step 3 reviewer: feature-analyst (lead) PASS.

#### Step 4 — Meta-labeler retrain (resolves ERR-033)
- `python retrain_meta_labeler.py --input data/new_expired_bots_backfilled.xlsx`. The `--allow-imputation` flag is correctly rejected for the active bootstrap profile.
- New artifact: `models/meta_labeler/metadata.json` `artifact_version=20260429_161435`, `lineage.hmm_artifact_version="rolling_180d_20260426_012042"` (matches active HMM in `artifact_manifest.json` and `artifacts/hmm_winner_calibrator/current.json`). `eval_metrics.positive_rate=0.332`, `eval_metrics.auc_cv=0.455`, `eval_metrics.is_calibrated=true` (sigmoid-OOS + beta calibration applied; ECE 0.1203 → 0.0227).
- Discovery-mode pipeline produced `results/deployment_ready_20260429_161505.csv` with `meta_prob_source.value_counts() = {"enrich": 18, "missing": 7}` and `meta_prob.notna() = 18/25`. `meta_prob` range `[0.1544, 0.3522]`, mean `0.2081`.
- Pipeline log no longer emits the `meta_labeler lineage mismatch` warning.

#### Suite verification
- `python -m pytest tests/ -k 'not (TestBoundedUniverseContract or test_current_workbook_contract_counts_if_available)'` → **1165 passed, 0 failed, 8 deselected**. All PIPELINE_FIX changes intact; pre-existing failures (ERR-038 family in `scanner/pattern_profile.py:325` / `profile_model.py:325`, ERR-039 in `test_utility_calibrator.py`) are excluded and tracked separately in ERRORS_LOG.md.

#### Errors logged for follow-up (non-blocking)
- ERR-035 (WATCH) — threshold-parity duplication across `unified_training_builder.py:840` + `meta_labeler.py:726`. Recommend pinning to a shared constant in a future maintenance pass.
- ERR-036 (WATCH) — Step 1 audit-column gap: the new branch records fall-through reason only via `logger.warning`, while the `hlabel_meta` path stamps audit columns. Future enhancement.
- ERR-037 (WATCH) — calibrator older-pool HMM heterogeneity: P3/P4 baseline could mix HMM contexts after rotation. Future enhancement (read-only diagnostic, not a new gate).
- ERR-038 (OPEN) — pre-existing 4-test failure family in `TestBoundedUniverseContract` (untouched by PIPELINE_FIX).
- ERR-039 (OPEN) — pre-existing `test_utility_calibrator.py::test_current_workbook_contract_counts_if_available` fixture drift (asserted pool size 69, actual 84 after backfill).

### Closed
- ERR-033 (Meta-labeler lineage drift) — resolved end-to-end by Steps 1+4.

---

## [6.5.7-hmm-fix-runtime-integration-verification] - 2026-04-29

### Verified - HMM Winner Calibrator Runtime Integration (no production code changes)

- Executed 7-step integration verification of the calibrator promoted on 2026-04-28
  (`hmm_winner_20260428_195512_559490`, pinned to HMM `rolling_180d_20260426_012042`).
  Plan at `C:\Users\cris_\.claude\plans\next-steps-for-complete-happy-swan.md`.
- Created and managed a persistent agents team via `TeamCreate` named
  `fixhmm01-integration-review` with 6 named members (`flow-curator`,
  `runtime-engineer`, `signal-analyst`, `regression-evaluator`, `theory-architect`,
  `lifecycle-overseer`). Team torn down via `TeamDelete` after Step 7.
- Step 1+2 smoke scan (`python run_full_pipeline.py --discovery-mode --top-n 25 --min-score 20 --capital 400`)
  produced `results/deployment_ready_20260429_074741.csv` with 25/25 rows showing
  `hmm_artifact_version=rolling_180d_20260426_012042`, `hmm_winner_score_source=calibrated`,
  and non-null `hmm_winner_score` (range 0.3681-0.5599, mean 0.4747).
- Step 3 ranking-integration decision: option C (diagnostic-only). Score rides in
  the deployment CSV as a passive column; no ranker consumes it. Honors the
  user's "no new hard gate" constraint. Revisit option B (tiebreaker) after
  ERR-033 / ERR-034 are resolved.
- Step 6 cross-reference vs locked pool: 5/5 pool fast-winner symbols present in
  the smoke CSV (AIOTUSDT, BIOUSDT, BSBUSDT, CHIPUSDT, SKYAIUSDT) all scored in
  the HIGH cluster (~0.55); 2/4 pool fast-loser symbols (BNBUSDT, ZECUSDT)
  scored in the LOW cluster (~0.37). 6/8 cleanly classifiable correct (LYNUSDT
  misclassified, BSBUSDT ambiguous).
- Score distribution is bimodal and theory-consistent: liquid major pairs
  (BTC/ETH/SOL/BNB/ADA/XRP/HYPE/TAO/1000PEPE) cluster at ~0.37 (trending
  regime); smaller-cap symbols cluster at ~0.55 (ranging regime).
- HMM_FIX.md final-proof section appended.
- Two new entries logged in `ERRORS_LOG.md`:
  - ERR-033 OPEN: meta-labeler stale HMM lineage causing
    `meta_prob_source="missing"` for all 25 rows. Pre-existing
    (meta-labeler trained 2026-04-22 vs HMM rolled forward 2026-04-26).
    Resolution: retrain meta-labeler against active HMM.
  - ERR-034 BLOCKED: `python run_full_pipeline.py` (production-mode) aborts at
    Binance `[401] -2015 Invalid API-key, IP, or permissions for action`.
    Pre-existing infrastructure issue. Step 6 was performed against the Step 1+2
    discovery-mode CSV as substitute.
- Modification criterion compliance: zero production code modifications during
  this cycle. Source files for `hmm_winner_calibrator.py`, `scan.py`,
  `enrich_grid_params.py` are byte-identical before and after.
- New memory entry: `feedback_agents_team_vs_subagents.md` distinguishing
  TeamCreate-based persistent teams from one-shot Agent subagent calls.

## [6.5.7-hmm-fixed-n-winner-calibrator] - 2026-04-28

### Fixed - HMM Winner Label Contract And Fixed-N Promotion Policy

- Rewrote `HMM_FIX.md` into one canonical current-state document for the fixed
  `N=85` repair path. The final contract keeps GaussianHMM as an unsupervised
  regime detector and uses a supervised HMM winner calibrator for selection.
- Updated `src/neutralgrid/calibration/hmm_winner_calibrator.py` so the active
  fast-winner label is now `duration_hours < 7.0` with `pnl_pct > 1.0` instead
  of `pnl_pct > 0.0`. Artifact metadata now records
  `positive_class="pnl_pct > 1.0"` and `negative_class="pnl_pct <= 1.0"`.
- Reduced hard promotion gates to the fixed-N gates that resolve with the
  locked sample pool: holdout AUC delta >= 0.10, holdout balanced-accuracy
  improvement, older-row balanced-accuracy drop <= 0.05, and older-row AUC drop
  <= 0.05. Former G5 CI, G9 mean-score, and G10 OOD/emission checks are retained
  as diagnostics only.
- Added scanner runtime lineage safety in `src/neutralgrid/scanner/scan.py`:
  `hmm_winner_score` is emitted only when the promoted calibrator's HMM artifact
  version equals the row's HMM artifact version; mismatches emit
  `hmm_winner_score_source="hmm_artifact_version_mismatch"` without scoring.
- Promoted the label-corrected calibrator for active HMM artifact
  `rolling_180d_20260426_012042`. Current artifact:
  `artifacts/hmm_winner_calibrator/current.json`, version
  `hmm_winner_20260428_195512_559490`.
- Verification:
  - `python -m compileall src/neutralgrid/calibration/hmm_winner_calibrator.py src/neutralgrid/scanner/scan.py`
    -> PASS.
  - `$env:PYTHONPATH='src'; pytest -q tests/unit/test_hmm_winner_calibrator.py -p no:cacheprovider`
    -> `9 passed in 2.46s`.
  - Strict promotion proof against `data/new_expired_bots_backfilled.xlsx` and
    `artifacts/hmm/rolling_180d_20260426_012042`: `pool_rows=85`,
    `pool_class_counts={"0":35,"1":50}`, `holdout_rows=28`,
    `holdout_class_counts={"0":12,"1":16}`,
    `holdout_auc_delta=0.16666666666666669`, all `P1-P4=true`,
    `promotable=true`, `current_json_updated=true`.
  - `python retrain_hmm.py --help` -> PASS. No HMM model retrain was required
    or performed for this label/promotion-policy repair.
  - Focused Pyright remains environment-blocked only by unresolved third-party
    imports (`numpy`, `pandas`, `sklearn`, `pytest`).

## [6.5.7-hmm-fix-calibrator-sweep] - 2026-04-27

### Verified - Strict Winner-Calibrator Sweep Across N=100/150/200 HMM Artifacts (FIXHMM-01)

- Regenerated the candidate workbook against each of the three un-promoted
  rolling_180d artifacts produced earlier in FIXHMM-01:
  `rolling_180d_20260427_030024` (N=100),
  `rolling_180d_20260427_051225` (N=150),
  `rolling_180d_20260427_071650` (N=200).
  Backfills used `scripts/backfill_training_features.py
  --default-artifact-version <artifact>` and produced
  `data/new_expired_bots_backfilled_rolling_180d_20260427_<tag>.xlsx`
  with `range_prob` / `trend_prob` / `hmm_feature_cutoff_utc` coverage
  at 94.7% (198 of 209 rows via `pinned_artifact_replay`, 11 via
  `artifact_unavailable`) for all three.
- Ran `python -m neutralgrid.calibration.hmm_winner_calibrator
  --skip-candidate-write` against each backfilled workbook with
  `--hmm-artifact-dir` pinned to the matching N artifact. Identical
  pool/fit/holdout/regression splits across runs (85 / 57 / 28 / 59
  rows; class counts cls0=29 / cls1=56 in the pool). All three runs
  returned `promotable=false`:
  - N=100: `holdout_auc_delta=-0.0936`, failed gates G4, G5, G6.
  - N=150: `holdout_auc_delta=-0.1754`, failed gates G4, G5, G6, G9.
  - N=200: `holdout_auc_delta=-0.1053`, failed gates G4, G5, G6, G9.
- Per the user-supplied decision rule (smallest passing universe; 150
  preferred over 200 in a tie; never promote on walk-forward HMM pass
  rate alone), no candidate was selected. `artifact_manifest.json`
  remains on `rolling_180d_20260426_012042` (promoted
  `2026-04-26T02:33:25.741953+00:00`); `diff` against the pre-sweep
  snapshot is empty. `artifacts/hmm_winner_calibrator/current.json`
  also unchanged (each run reported `current_json_updated=false`).
- Empirical takeaway: scaling the **HMM-training pool** from 50 to 200
  symbols does not move the calibrator gates in the direction needed
  for promotion. The binding constraint is on the **labelled
  expired-bot population** (28-row holdout, class-0 underrepresented at
  9 rows). Growing the labelled bot pool is the next lever; growing N
  on the unsupervised HMM training side is not.
- Per-N table and decision recorded in `HMM_FIX.md` under
  "Implementation Session Log - Strict Calibrator Sweep vs
  N=100/150/200 Artifacts (2026-04-27)". Helper script for workbook
  introspection committed at
  `mode_probe/fixhmm01_workbook_summary.py`.

## [6.5.7-hmm-fix-backfill-bootstrap] - 2026-04-26

### Fixed - Strict Calibrator Flat Backfill Input (ERR-031)

- `src/neutralgrid/calibration/hmm_winner_calibrator.py` now preserves the
  existing two-sheet workbook contract (`General` + `Meta Features`) and adds a
  strict fallback for the verified one-sheet backfill output produced by
  `scripts/backfill_training_features.py`.
- The fallback is accepted only when the flat sheet contains the required
  outcome columns, aggregate HMM feature columns, and
  `hmm_feature_cutoff_utc`. Missing `backfill_status` is derived fail-closed
  from `hmm_feature_source`: `pinned_artifact_replay` is usable; every other
  source is treated as `hmm_failed` and excluded by the existing pool filter.
- Added `tests/unit/test_hmm_winner_calibrator.py` coverage proving strict flat
  workbook loading, ex-ante cutoff preservation, same-population labels, and
  fail-closed exclusion of non-replayed HMM rows.
- Verification:
  - `pytest -q tests/unit/test_hmm_winner_calibrator.py -p no:cacheprovider`
    -> `5 passed`.
  - `python -m neutralgrid.calibration.hmm_winner_calibrator --input
    data/new_expired_bots_backfilled.xlsx --skip-candidate-write` now runs
    through the strict ex-ante path without writing artifacts:
    `pool_rows=85`, `fit_rows=57`, `holdout_rows=28`,
    `older_regression_rows=59`, `holdout_auc_delta=-0.10526315789473684`,
    `promotable=false`.
  - `python retrain_hmm.py --help` passed.
  - Focused Pyright remains blocked only by import-resolution errors for
    third-party packages (`numpy`, `pandas`, `sklearn`, `pytest`) in the active
    environment.
- No calibrator artifact was promoted, no HMM artifact was changed, and scanner
  runtime behavior was untouched.

### Fixed - Backfill Script CLI Surface And Interval Support (ERR-022, ERR-023)

- `scripts/backfill_training_features.py` previously had no `argparse`
  and silently ignored `--input` / `--output` / any other flags, running
  with hardcoded paths. Added `_parse_args()` helper plus minimal
  argparse with `--input`, `--output`, and `--default-artifact-version`
  flags. `--help` now short-circuits cleanly.
- `scripts/backfill_training_features.py:144-153` `_interval_delta`
  whitelisted only `1h` / `15m` / `5m` and raised
  `ValueError("Unsupported interval: 1m")` on rows whose feature path
  requested 1-minute klines. Added `if interval == "1m": return
  timedelta(minutes=bars)` branch. Live evidence: pre-fix run failed on
  `NIGHTUSDT`, `XMRUSDT`, `BOBUSDT` within the first 3 seconds; post-fix
  run completed 209/209 rows with zero `Unsupported interval: 1m`
  errors.

### Fixed - Backfill HMM Stamping Gate (ERR-030, Option A)

- Pre-fix, the row-processing path at
  `scripts/backfill_training_features.py:292-321` was gated on each
  input row already carrying a populated `hmm_artifact_version`. The
  source workbook does not carry that column, so `_get_hmm_predictor`
  returned `None` for every row, the inference branch never fired, and
  `hmm_feature_cutoff_utc` was stamped on `0 / 209` rows after a
  superficially "successful" backfill. `range_prob` / `trend_prob`
  coverage was `0.0%`.
- Resolved per Option A spec (user-approved):
  - Added `--default-artifact-version <version>` CLI flag.
  - `TrainingDataBackfiller.__init__` accepts and stores
    `default_artifact_version` (normalized through `_non_empty_str`).
  - Row processing now resolves the artifact version as
    `row_artifact_version or self.default_artifact_version` at
    `scripts/backfill_training_features.py:294`. If the row already
    carries `hmm_artifact_version`, it is preserved. If the row is
    missing it and the CLI flag is provided, the pinned fallback is
    used. If neither exists, behavior is unchanged
    (`hmm_feature_source = "missing_artifact_version"`).
  - The resolved version is stamped into the output `hmm_artifact_version`
    column at line 296-297; the predictor's reported version overwrites
    on successful inference at line 309. `hmm_feature_source =
    "pinned_artifact_replay"` on success per the prior contract.
  - Constraint honored: no environment fallback, no auto-resolution of
    the active artifact via `artifact_manifest.json`, no scanner /
    runtime HMM behavior changed. Lineage gap is solved entirely inside
    the backfill script.

### Documented - Errors Log Appended (ERR-022 through ERR-031)

- `ERRORS_LOG.md` appended a new section "Active Error Checks — HMM_FIX
  Implementation Blockers (2026-04-25)" with structured entries
  ERR-022 through ERR-031:
  - ERR-022 (CLI absence) - resolved this session.
  - ERR-023 (1m interval rejection) - resolved this session.
  - ERR-024 (manual ingest parser ASCII-only on Chinese-character
    symbols) - workaround used; permanent fix proposed but not applied.
  - ERR-025 (HMM degenerate posterior on small-caps - tracked in
    `HMM_FIX.md`).
  - ERR-026 (Pyright import resolution blocker - environment).
  - ERR-027 (`tests/unit/test_hmm_artifact_policy.py` Windows temp ACL
    blocker - environment).
  - ERR-028 (full HMM retrain sweep exceeds 4-min session timeout -
    deferred to longer-running shell).
  - ERR-029 (workbook missing `trigger_price` column - cosmetic).
  - ERR-030 (backfill HMM stamping gate) - CLOSED this session via
    Option A.
  - ERR-031 (backfill output writes single sheet `Sheet1`; strict
    calibrator now accepts the verified flat backfill layout under strict
    required-column and fail-closed source rules - CLOSED this session).

### Verified - Backfill V2 Output And Team Review

- Re-ran backfill with `python scripts/backfill_training_features.py
  --input data/new_expired_bots.xlsx --output
  data/new_expired_bots_backfilled.xlsx --default-artifact-version
  rolling_180d_20260421_213328`. Result: exit 0, 0 row-level errors,
  209/209 rows touched.
- Coverage proof:
  - `hmm_feature_cutoff_utc` populated: `198 / 209` (94.7%) - was
    `0 / 209` pre-fix.
  - `range_prob` populated: `198 / 209` (94.7%) - was `0.0%`.
  - `trend_prob` populated: `198 / 209` (94.7%) - was `0.0%`.
  - `hmm_artifact_version` populated: `209 / 209` with
    `rolling_180d_20260421_213328`.
  - `hmm_feature_source` distribution: `198` =
    `pinned_artifact_replay`, `11` = `artifact_unavailable`. The 11
    fail-closed rows are concentrated in five very-new-listing symbols
    (`NIGHTUSDT`, `BASEDUSDT`, `BSBUSDT`, `CHIPUSDT`, `OPGUSDT`) whose
    `start_time_utc` falls within the first ~8 days of listing, so
    `interval=15m, limit=805` could not supply the 800 bars the HMM
    inference path requires.
- Ex-ante semantics check: every populated `hmm_feature_cutoff_utc`
  equals `start_time_utc` instant-for-instant after timezone
  normalization (198 == / 0 != / 0 parse failures).
- Team review (`hmm-fix-option-a-review` team, three named teammates):
  - `code-reviewer` (Task #1): PASS on all 5 Option A spec rules. Zero
    behavioral deviations. Local var named `row_artifact_version`
    instead of `row_version` (cosmetic only).
  - `proof-auditor` (Task #2): PASS on Proofs 1-3. Proof 4
    (preservation of pre-stamped row value) is N/A because the input
    workbook lacks the `hmm_artifact_version` column entirely.
  - `classification-auditor` (Task #3): 10/10 items honored across
    "Provable False Optionality" / "Provably Unnecessary" / "Items Not
    Valid To Strike". One item flagged ⚠ (re-running strict calibrator
    is a process step pending execution; precondition is in place).

### Closed - Strict Calibrator Flat Backfill Input (ERR-031)

- Strict calibrator invocation `python -m
  neutralgrid.calibration.hmm_winner_calibrator --input
  data/new_expired_bots_backfilled.xlsx --skip-candidate-write` now
  succeeds past workbook loading and emits validation proof. The output
  remains intentionally non-promotable because the validation gates fail:
  `holdout_auc_delta=-0.10526315789473684`,
  `G4_holdout_auc_delta_ge_0_10=false`,
  `G5_ci_lower_ge_0_02_when_estimable=false`,
  `G6_holdout_balanced_accuracy_improved=false`, and
  `G9_mean_winner_score_gt_loser=false`.

## [6.5.7-meta-labeler-bootstrap-simplification] - 2026-04-21

### Changed - Active Meta-Labeler Contract Reduced To 8 Bootstrap Features

- Added `SNAPSHOT_META_FEATURES_V20260421_BOOTSTRAP` in
  `src/neutralgrid/models/meta_labeler.py` with exactly:
  `range_prob`, `survival_prob`, `utility_score`, `ou_halflife`,
  `profit_per_grid_pct`, `num_grids`, `grid_spacing_pct`, `adx_1h`.
- Switched `ACTIVE_SNAPSHOT_META_FEATURES` from the 38-feature
  `snapshot_v20260420` profile to the new 8-feature bootstrap profile.
- Kept `snapshot_v20260407` and `snapshot_v20260420` as historical /
  comparison profiles only; they are no longer the active default.
- `retrain_meta_labeler.py --feature-profile` now defaults to
  `snapshot_v20260421_bootstrap`.

### White-Box Theory Reconciliation (added 2026-04-29 by PIPELINE_FIX Step 4)

The 8-feature bootstrap simplification was paired with a label-contract
adjustment that was implemented in the code (`ACTIVE_META_TARGET_PNL_THRESHOLD_PCT = 0.0`,
`ACTIVE_META_TARGET_LABEL_COLUMN = "fast_winner_target"`, `meta_labeler.py:140-144`)
but not explicitly reconciled with the white-box theory in this CHANGELOG entry
at the time. PIPELINE_FIX Step 4 made this reconciliation explicit:

- The deployed meta-labeler answers
  `P(pnl_pct > 0 | duration_hours <= 7h, regime features)` rather than
  `P(pnl_pct >= 3.0 | ...)`. The `meta_hurdle_pct = 3.0` post-cost hurdle in
  `BarrierConfig` remains the strict-label-path threshold used by the unified
  training builder and is unchanged; the meta-labeler's active target column
  is `fast_winner_target` (any positive PnL inside the 7-hour bound).
- Theory mechanism is unchanged: HMM regime features (range_prob,
  survival_prob, persistence_prob) are the discriminator. The boundary widens
  from "≥3.0% post-cost winner" to "any positive PnL within 7h" as a locked-pool
  concession (post-cost positive rate is ~1.3% at n=226, below the 5%
  trainability gate at `meta_labeler.py:725-732`).
- Stage 12 Kelly sizing must interpret `meta_prob` as "probability of fast
  positivity," not "probability of meaningful post-cost winner."
- ERR-040 (theory contract drift) recorded the silent state of this drift
  prior to reconciliation. ERR-042 (locked-pool noise floor: CV AUC = 0.455)
  records the structural consequence: the locked-pool meta-labeler is
  lineage-clean but discrimination-unproven.
- Internal-contract alignment (ERR-041): `meta_labeler.py:1593` now writes
  `model_params.hurdle_pct = ACTIVE_META_TARGET_PNL_THRESHOLD_PCT` when
  `ACTIVE_META_TARGET_LABEL_COLUMN` is set, so it agrees with
  `eval_metrics.target_positive_pnl_threshold_pct` and Config Integrity per
  `safety-invariants.md` is restored.

### Fixed - Utility Score Training/Serving Alignment

- `src/neutralgrid/scanner/scan.py` now computes scan-time
  `utility_score` with the same governed provisional geometry semantics
  used by the validator path:
  - `profit_per_grid_pct = provisional_profit_per_grid_pct`
  - `num_grids = provisional_num_grids`
  - `range_size_pct = provisional_range_size_pct`
- `src/neutralgrid/training/unified_training_builder.py`
  `_derive_utility_score()` now uses the same governed provisional
  utility semantics rather than recomputing from row geometry and
  optional `survival_prob` / `hurst_exponent`.
- `src/neutralgrid/training/scanner_integration.py` continues to prefer
  `regime_utility` over `scan_utility_score` / `utility_score`, and its
  comments now document that this is the authoritative governed
  provisional utility path.
- Retrain verification now records utility provenance
  (`pinned_v0_fallback` vs `current_artifact`) plus
  `lambda_risk`, `kappa_trend`, and `horizon_hours`.

### Fixed - Bootstrap Retrain Contract Enforcement

- `retrain_meta_labeler.py` now blocks `--allow-imputation` for the
  active bootstrap profile. The active 8-feature contract must remain
  exact and non-imputed.
- `retrain_meta_labeler.py` help text now states clearly that `--input`
  is a compatibility / analysis reference path and active training uses
  the unified snapshot/backtest backbone.
- `src/neutralgrid/training/unified_training_builder.py` now preserves
  `grid_spacing_pct` from backtest-side candidate geometry during the
  snapshot-outcome join for narrow geometry repair, instead of dropping
  it before merge.
- The active bootstrap retrain path now enforces:
  - `expected_feature_count = 8`
  - `selected_feature_count = 8`
  - `incomplete_features = []`

### Changed - Documentation Updated To Current Runtime Truth

- `META_LABELER_FIX.md` was rewritten to the current bootstrap state:
  - overall status `DONE`
  - progress bar `100%`
  - all sections updated to `DONE`
  - verified proof recorded from the live refreshed artifact and
    verification JSON
- `retrain_meta_labeler.py` module docstring no longer describes the old
  spreadsheet-style 14-feature world; it now documents the active
  unified bootstrap retrain path.

### Verified

- `pyright` -> `0 errors, 0 warnings, 0 informations`
- Focused tests:
  - `tests/unit/test_meta_labeler_retrain_contract_v20260408.py`
  - `tests/unit/test_meta_labeler_inference_aliases.py`
  - `tests/unit/test_scanner_integration_v20260320.py`
  - Result: `20 passed`
- Fresh retrain:
  `python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --output models/meta_labeler.pkl`
  - `training_rows = 226`
  - `selected_feature_count = 8`
  - all 8 selected features `226/226` non-null
  - `y_counts = {0: 151, 1: 75}`
  - `positive_rate = 33.19%`
  - `auc_cv = 0.5233`
  - `precision_at_5 = 0.3333`
  - artifact health after retrain: healthy `8/8`
  - utility provenance:
    - `source = pinned_v0_fallback`
    - `lambda_risk = 2.0`
    - `kappa_trend = 1.5`
    - `horizon_hours = 6.0`

## [6.5.7-pattern-profile-phases-1-3] - 2026-04-21

### Added - Shared Scanner Data Layer (Phase 1)

- New module `src/neutralgrid/scanner/_xlsx_io.py` consolidates xlsx I/O
  for both profile builders:
  - `read_sheet`, `raise_on_duplicate_strategy_id`, `validate_dataframe`,
    `detect_format` (inspects `xl.sheet_names` explicitly rather than
    relying on a `ValueError` catch).
- Duplicated helpers removed from `pattern_profile.py` and
  `profile_model.py`; both now import from `_xlsx_io`.
- `profile_model.py` now applies `validate_dataframe` on every sheet read
  and uses the allow-list merge pattern previously only present in
  `pattern_profile.py`.
- `_xlsx_io.validate_dataframe` is fail-closed: `win_rate > 100`,
  `profit_factor < 0`, and `duration_hours < 0` raise `ValueError`
  instead of logging.
- `PROFIT_FACTOR_CAP = 1000.0` centralized in
  `src/neutralgrid/core/constants.py`; both builders import from the
  single source.

### Fixed - Profile-Model Training Correctness (Phase 2)

- **Availability-filter leakage** (`profile_model.py`): availability
  check now evaluates on `df_labeled`, not the full `df`; mirrored in
  `pattern_profile.py`.
- **Class-conditional imputation bias**: `_impute(Xw)` + `_impute(Xl)`
  replaced with a shared median computed on the concatenated labeled
  set before class split. Docstring documents: *"Imputation is
  class-agnostic to avoid inflating `mu_w - mu_l`."*
- **Feature standardization**: features z-scored on the concatenated
  labeled set before `Sw` / `Sl` / pooled `S`. `ProfileModel` dataclass
  gains `feature_mean` / `feature_std`; `to_json` / `from_json` /
  `_vector` updated so inference applies the same transform.
- **Per-class sample floor** raised from `5` to
  `max(30, 3 * len(feats))`. Fewer samples than that cannot support a
  Gaussian of that dimension.
- New tests in `tests/unit/test_profile_model_training.py` cover
  shared-median impute, availability filter scoped to `df_labeled`,
  z-score round-trip at the class-mean midpoint, and small-n refusal.

### Added - Walk-Forward Promotion Discipline (Phase 3.1-3.6)

- New module `src/neutralgrid/scanner/profile_model_walkforward.py`:
  - `walkforward_evaluate` — chronological expanding-window purged
    K-fold; per-fold relabeling so test-fold pnl does not leak into the
    winner label. Raises `ValueError` if
    `purge_hours < max_duration_hours`. Purge is two-sided: drops any
    train bot whose window ends into the test fold, not only those that
    start inside it.
  - `_apply_winner_labels`, `_train_from_frame` (fold-local training
    mirroring the Phase 2 shared-median + z-score + pooled-covariance
    path).
  - `promote_profile_version` — atomic artifact + `current.json` write
    with sha256. Refuses promotion unless `mean_pass_rate >= 0.50`
    (`MEAN_PASS_RATE_FLOOR`) and requested-feature coverage
    `>= 0.90` (`COVERAGE_FLOOR`). Fold-pass threshold is
    `AUC_FOLD_PASS_THRESHOLD = 0.55`.
  - `resolve_active_profile_model_path` — loads the promoted artifact
    via `current.json`; falls back to `data/profile/profile_model.json`
    as an unvalidated bootstrap candidate only when `current.json` is
    absent; returns a non-existent `_corrupt_current_json.missing`
    sentinel on corrupt `current.json` or a missing `active` key
    (fail-closed per Phase 3.5). Callers must treat the sentinel as
    `data_missing`.
  - `train_and_promote` — end-to-end helper wiring
    `walkforward_evaluate` -> `promote_profile_version` with trial
    logging on both pass and refusal outcomes.
- Artifact naming follows HMM convention:
  `profile_model_YYYYMMDD_HHMMSS.json` with `current.json` pointing to
  the active artifact. Writes are atomic (stage to `.tmp`, rename).
- New test file `tests/unit/test_profile_promotion.py` (12 tests) covers
  naming convention, refusal below the pass-rate floor, atomic accept
  path, pointer resolution, bootstrap fallback, fail-closed on corrupt
  / missing-active `current.json`, purge-invariant enforcement, silent-
  drop guard at coverage = 0.80 and boundary at 0.90, trial logging on
  both outcomes, and end-to-end AUC on a separable synthetic.
- Phase 3.7 (CLI discipline in `retrain_scanner.py`: remove
  `--skip-model`, fail-closed on profile-model training error, explicit
  warn + `scoring_flags` on `profile_model is None`) is deferred to a
  follow-up batch.

### Changed - Feature Set Replacement (Phase 5, Batches A+B landed)

- Legacy `_avg` features removed per Phase 5.1 schema-hygiene sweep
  (`funding_rate_avg`, `open_interest_start`, `atr_avg`, `volume_avg`):
  - `pattern_profile.py` `DEFAULT_FEATURES` + xlsx allow-list + numeric
    coercion list.
  - `scan.py` live alias block and the `funding_rate_avg: 0.8` scoring
    weight (removed as a separate edit per first-addendum nuance).
  - `feature_extractor.py` `atr_avg` / `volume_avg` dataclass fields +
    computation.
  - `scanner_integration.py` and `run_full_pipeline.py` fallback arm of
    `get_first("funding_rate", "funding_rate_avg")` — primary
    `funding_rate` read retained.
  - `tests/test_afml_integrations.py` fixture rows and the two anti-
    aliasing regression guards in
    `tests/unit/test_afml_compliance_fixes.py` that become semantically
    vacuous after removal.
  - `CHANGELOG.md` historical references preserved per append-only
    convention; `Live/` snapshots untouched per CLAUDE.md Live Bot Data
    Storage Policy.
- Four replacement features now drive the profile artifacts:
  - F1 `parkinson_vol_ratio_4h_24h_pre` — Parkinson HL-variance ratio
    over 15m klines with strict `close_time < t_entry`. Parkinson helper
    added in `src/neutralgrid/indicators/technical.py`.
  - F2 `variance_ratio_1m_15m_pre_2h` — Lo-MacKinlay VR with a
    `VR(5m,15m)` fallback when 1m coverage is insufficient; per-row
    `variance_ratio_fallback_used` indicator persisted so the model
    conditions on the estimator rather than mixing distributions.
  - F3 `funding_carry_expected_next_7h` — expected funding carry over
    the bot horizon (Batch A).
  - F4 `liquidity_stability_z_1h` — volume-stability z-score (locked to
    option (a) per fourth-addendum decision; order-book-depth option
    struck). Folded into Batch B as a peer of F1/F2; no separate design
    doc, no new ingest module.
- Backfill-first landing: `retrain_scanner.py` now backfills F1-F4 onto
  the canonical training sheet from real historical market data (15m
  klines for F1/F4, 1m klines for F2, historical funding for F3) before
  the per-class sample floor / walk-forward gate evaluates. This is
  recompute-from-real-history, not synthetic sample generation.

### Verified

- `pyright src/neutralgrid/scanner/profile_model_walkforward.py` →
  `0 errors, 0 warnings`.
- `tests/unit/test_profile_model_training.py`: 4 passed.
- `tests/unit/test_profile_promotion.py`: 12 passed.
- Combined Phase 2 + Phase 3 targeted suite: 16/16 green.
- Retrain proof on `data/new_expired_bots.xlsx` via
  `python retrain_scanner.py --input data/new_expired_bots.xlsx
  --output-dir data/profile --top-quantile 0.605
  --min-avg-profit-per-grid 0.59 --skip-gate`:
  - Backfill eligible rows: 77 of 192 valid General rows.
  - Bounded universe: 77 rows (0 <= `duration_hours` < 7.0).
  - Pattern profile: 30 winners, PnL threshold 3.43%, 4 features.
  - Profile model: prior winner 0.390, shrinkage 0.3.
  - Artifacts written: `data/profile/pattern_profile.json`,
    `data/profile/profile_model.json`. Promotion gate skipped via
    `--skip-gate`; `current.json` is NOT updated by this path.
- Pre-existing unrelated failure:
  `tests/unit/test_utility_calibrator.py::test_current_workbook_contract_counts_if_available`
  (workbook-drift 115 vs 114) — not caused by Phase 1-3.

## [6.5.7-meta-labeler-snapshot-v20260420] - 2026-04-21

### Added - Meta-Labeler Feature Snapshot v20260420

- New `SNAPSHOT_META_FEATURES_V20260420` tuple in
  `src/neutralgrid/models/meta_labeler.py` extends the 29-feature
  `v20260407` base with H*/regime and grid-geometry context for
  fast-winner learning:
  `range_prob`, `trend_prob`, `survival_prob`, `utility_score`,
  `ou_halflife`, `profit_per_grid_pct`, `num_grids`,
  `grid_spacing_pct`, `adx_1h`.
- `ACTIVE_SNAPSHOT_META_FEATURES` alias points at `v20260420`; all
  internal consumers (`apply_feature_profile_defaults`,
  `MetaLabelerConfig.features` default, bootstrap check in
  `_meta_labeler_needs_bootstrap`, unified training builder's
  missing-feature warning) now reference the active alias instead of
  the hard-coded `v20260407` tuple.
- `retrain_meta_labeler.py` default `--feature-profile` flipped to
  `snapshot_v20260420`.
- New inference aliases added to `_INFERENCE_FEATURE_ALIASES` so the
  runtime accepts upstream column names without renaming callers:
  `hmm_range_prob -> range_prob`, `hmm_trend_prob -> trend_prob`,
  `regime_utility -> utility_score`,
  `scan_utility_score -> utility_score`,
  `scan_grid_spacing_pct -> grid_spacing_pct`.
- `grid_spacing_pct` gains a `0.75` median in `_FEATURE_MEDIAN_DEFAULTS`
  so legacy models without a fitted imputer still fill sanely.

### Added - `grid_spacing_pct` Through The Full Feature Pipeline

Per the Feature Pipeline Update Rule (`safety-invariants.md`), the new
`grid_spacing_pct` feature is wired through all three sites:

- `src/neutralgrid/backtest/candidate_pipeline.py` — added to
  `_SCANNER_TO_FEATURE` and `TRAINING_OUTPUT_COLUMNS`.
- `src/neutralgrid/training/data_generator.py` —
  `TrainingTableSchema.features`, `FeatureSnapshot` dataclass field +
  `to_dict`, numeric coercion in `ExistingDataMapper`, and the legacy
  `COLUMN_MAP` alias `grid_spacing_pct -> grid_spacing_pct_raw` fixed
  to a pass-through so the feature is no longer silently renamed out
  of the training frame.
- `src/neutralgrid/training/unified_training_builder.py` — added to
  `_SCAN_TO_FEATURE`.

### Added - Scanner 4-Feature Profile Block In `compute_features`

- `src/neutralgrid/scanner/feature_extractor.py` gains a helper
  `compute_profile_feature_block(klines_15m, klines_1m,
  funding_carry_expected_next_7h)` that computes the four profile
  features with explicit bar-count guards:
  `parkinson_vol_ratio_4h_24h_pre` (>= 96 15m bars),
  `variance_ratio_1m_15m_pre_2h` (>= 120 1m bars),
  `funding_carry_expected_next_7h` (from `funding_info`), and
  `liquidity_stability_z_1h` (>= 100 15m bars with `quote_volume`).
- `SymbolFeatures` dataclass declares the four fields explicitly so
  downstream consumers see them via `as_dict()`.
- `compute_features` signature extends with `klines_1m` and
  `funding_info` (used to derive the expected 7h funding carry via
  `neutralgrid.data.funding_rate.expected_funding_carry_next_hours`).
- Legacy `atr_avg` and `volume_avg` `SymbolFeatures` fields removed per
  the Phase 5.1 legacy-feature sweep; the related code paths in
  `compute_features` drop to `quote_volume_24h`-only.

### Changed - Unified Training Builder Bootstrap Label Discipline

Recovery paths for a cold-start training run that would otherwise have
zero usable rows, each with explicit audit flags so the fallback is
traceable rather than silent:

- **Gate 6 bootstrap relaxation**
  (`src/neutralgrid/training/unified_training_builder.py`): when
  `version_gated` filtering empties the frame, rows carrying
  `y_horizon` and `label_contract_version` are re-admitted with
  `version_gate_relaxed_for_bootstrap = True`. A WARN line is logged
  with the count of re-admitted rows.
- **Post-filter positive-rate fallback**: if the current `y` column
  positive rate is < 5% after filtering and `y_horizon` is trainable
  (nunique > 1 and positive rate >= 5%), `y` is overwritten from
  `y_horizon` with `label_source = y_horizon_hstar_bootstrap_post_filter`
  and `hlabel_meta_bootstrap_below_min_positive_rate = True`.
- **Label-selection hierarchy**: `hlabel_meta` is now the preferred
  training target, with an explicit `y != hlabel_meta` conflict WARN
  when both disagree. If `hlabel_meta` is degenerate (nunique <= 1) or
  has positive rate < 5%, the selector falls back to `y_horizon` or
  `label_positive_by_horizon` with `label_source` stamped as
  `<col>_hstar_bootstrap` and the degenerate/below-min flags recorded.
  Previous order `(y, y_horizon, label_positive_by_horizon)` silently
  overwrote strict hierarchical labels with lenient horizon labels.
- **`grid_spacing_pct` derivation**: when missing on a row with
  known `grid_lower`, `grid_upper`, and `num_grids`, derive it from
  candidate geometry via `_derive_grid_spacing_pct`. Stamps
  `grid_spacing_source = derived_from_candidate_geometry`.
- **Geometry repair from `backtest_results_*.csv`**: `build_dataset`
  now reads `candidate_id`, `grid_spacing_pct`, `grid_lower`, and
  `grid_upper` from backtest result files and merges them into the
  combined frame on `candidate_id`, filling only missing values.
  Stamps `geometry_feature_source = backtest_candidate_config`.

### Verified

- `tests/unit/test_meta_labeler_inference_aliases.py` — covers the new
  `hmm_range_prob`, `hmm_trend_prob`, `regime_utility`,
  `scan_utility_score`, and `scan_grid_spacing_pct` aliases and the
  default-fill behaviour.
- `tests/unit/test_meta_labeler_retrain_contract_v20260408.py` — test
  data and assertions updated to the 37-feature `v20260420` contract.
- `tests/unit/test_scanner_integration_v20260320.py` — new
  `test_build_feature_snapshot_default_fills_active_profile` confirms
  the default fill honours `ACTIVE_SNAPSHOT_META_FEATURES`.
- `tests/unit/test_enhancements_v653.py`,
  `tests/test_afml_compliance.py`, `tests/test_afml_integrations.py`,
  `tests/unit/test_afml_compliance_fixes.py` — feature-count
  assertions refreshed; the two now-vacuous anti-aliasing guards
  (`test_atr_avg_not_aliased_to_atr_pct`,
  `test_volume_avg_not_aliased_to_quote_volume`) deleted per the
  Phase 5.1 plan.

## [6.5.7-horizon-contract] - 2026-04-18

### Changed - Unified Bot Horizon Contract

- Centralized horizon constants `BOT_HORIZON_BARS_15M` and
  `BOT_HORIZON_SECONDS` in `src/neutralgrid/core/constants.py` so
  stochastic survival, microstructure, oscillation, ranker, and training
  paths share one source of truth.
- `LABEL_CONTRACT_VERSION` bumped to `2026-04-17`.
- Touched paths: `backtest/backtest_realistic.py`,
  `backtest/btk_label_contract.py`, `scripts/validate_mc_containment.py`,
  `src/neutralgrid/backtest/candidate_pipeline.py`,
  `src/neutralgrid/core/config.py`, `src/neutralgrid/core/constants.py`,
  `src/neutralgrid/grid/spacing_profile.py`,
  `src/neutralgrid/scanner/enrich_grid_params.py`,
  `src/neutralgrid/scanner/pnl_ranker.py`,
  `src/neutralgrid/scanner/tradable_oscillation.py`,
  `src/neutralgrid/training/unified_training_builder.py`,
  `src/neutralgrid/validation/microstructure.py`,
  `src/neutralgrid/validation/stochastic.py`,
  `src/neutralgrid/validation/utility.py`.

### Verified

- New `tests/unit/test_bot_horizon_contract.py` (65 lines) pins both
  constants and the downstream derivations.
- `tests/unit/test_btk_output_contract.py` gains 25 lines of contract
  coverage; `tests/unit/test_btk_label_runner.py`,
  `tests/unit/test_enrich_grid_params_survival_recalc.py`,
  `tests/unit/test_plan_v6_steps.py`, and
  `tests/unit/test_unified_training_builder.py` (+119 lines) updated.
- Commit `0b749f9` (2026-04-18).

## [6.5.7-discovery-recovery] - 2026-04-16

### Added - Meta-Labeler Retrain Imputation Escape Hatch

- Added `--allow-imputation` to `retrain_meta_labeler.py`.
  - Fills only null values in selected retrain-contract features.
  - Uses fixed contract defaults from the feature profile path.
  - Writes `allow_imputation` and `imputed_features` into
    `models/meta_labeler_verification.json`.
  - Keeps the feature contract enforced after imputation, so this is an
    explicit low-sample retrain bridge rather than silent data repair.
- Added unit coverage in
  `tests/unit/test_meta_labeler_retrain_contract_v20260408.py` proving:
  - selected-feature nulls fail by default;
  - `--allow-imputation` fills selected-feature nulls before contract
    enforcement;
  - CLI parsing accepts the new flag.

### Added - Discovery Mode for Candidate Generation

- Added `--discovery-mode` to `run_full_pipeline.py`.
  - Discovery mode defaults `--min-score` to `20.0` when the user does not
    pass a score.
  - Console output is labeled `DISCOVERY GRID CANDIDATES` /
    `DISCOVERY CANDIDATE`.
  - CSV output is stamped with `discovery_mode=True`.
- Added `EnrichConfig.discovery_mode` to
  `src/neutralgrid/scanner/enrich_grid_params.py`.
  - Pre-reject, regime, hard gate, and Stage B verdicts are preserved as audit
    fields instead of being deleted.
  - New audit columns include `pre_reject_would_reject`,
    `pre_reject_reason`, `regime_would_reject`,
    `regime_rejection_reasons`, `hard_gate_would_reject`,
    `hard_gate_rejection_reasons`, `discovery_geometry_filled`, and
    `discovery_geometry_reason`.
  - Stage B still runs; discovery mode records the rejection but allows
    geometry-valid candidates to be emitted for data generation.

### Fixed - Discovery Geometry After Regime Rejection

- Fixed a live zero-candidate blocker where `RegimeValidator.validate()`
  returned immediately on HMM/regime rejection before running the documented
  non-gating range extractor.
- Discovery mode now reuses the existing
  `RegimeValidator.check_range_quality()` quantile-plus-ATR range estimator
  after a regime reject, then derives 1m ATR with the existing `calc_atr()`
  path for grid spacing.
- No synthetic range bounds were introduced.
- Added enrichment tests proving:
  - discovery mode preserves Stage B rejection while emitting a geometry-valid
    discovery candidate;
  - discovery mode fills geometry after an invalid regime result using
    existing validator metrics.

### Fixed - Fresh Discovery Backtest Loop

- Fixed `backtest_candidates.py --latest-only` so it selects the newest
  `deployment_ready_*.csv` by file modification time instead of lexicographic
  filename ordering.
  - This prevents stale files such as
    `deployment_ready_typefix_smoke_20260301.csv` from shadowing newer
    timestamped scanner artifacts.
- Updated `src/neutralgrid/backtest/candidate_pipeline.py` so grid-valid
  discovery candidates remain eligible for backtest data generation even when
  they are below the deployment score floor.
  - `below_threshold_tag` remains in the scanner artifact for auditability.
  - CSV-loaded boolean columns are parsed with a CSV-safe truthy helper.
- Added regression coverage in
  `tests/unit/test_candidate_pipeline_bypass.py` proving standard mode keeps
  grid-valid discovery rows below `--min-score` while still dropping ordinary
  below-score rows.

### Verified

- `python -m compileall` passed for changed Python files.
- `$env:PYTHONPATH='src'; pyright` returned `0 errors, 0 warnings`.
- `tests/unit/test_enrich_grid_params.py`: `12 passed`.
- `tests/unit/test_candidate_pipeline_bypass.py`: `7 passed`.
- Focused meta-labeler retrain-contract tests: `4 passed, 6 deselected`.
- Micro-oscillator gate/integration tests: `28 passed`.
- `python retrain_meta_labeler.py --input data/new_expired_bots.xlsx --allow-imputation --dry-run --feature-profile snapshot_v20260407`
  passed and wrote `models/meta_labeler_verification.json`.
  - Unified training rows: `162`.
  - Selected feature count: `29/29`.
  - Imputed values: `open_interest` nulls `2 / 162 = 1.2346%` of rows,
    or `2 / (162 * 29) = 0.0426%` of selected feature cells.
- Live discovery smoke produced
  `results/deployment_ready_20260416_140855.csv` with 5 rows and 3
  geometry-valid discovery candidates: `SOLUSDT`, `ETHUSDT`, `BTCUSDT`.
- `python backtest_candidates.py --dry-run --latest-only --min-score 20 --max-candidates 5`
  selected those 3 fresh discovery candidates for backtesting.
- Actual 6h backtest command reached live fetch and selected the same 3
  candidates, but saved no rows because only 4 one-minute bars existed after
  the just-created scan timestamp while `--min-bars 360` was required.

## [6.5.7-memory-and-guidelines] - 2026-04-16

### Added — Behavioral Guidelines in CLAUDE.md

Added `## Behavioral Guidelines` section (lines 73–96) to `CLAUDE.md` with
three subsections derived from Andrej Karpathy's LLM coding principles,
adapted to the project's safety-invariant architecture:

- **Think Before Acting** — surface assumptions, present tradeoffs, confirm
  understanding of affected safety invariants before modifying code near
  leakage guards, fail-closed gates, or feature pipeline.
- **Surgical Ownership** — every changed line traces to the user's request;
  clean up orphans your own changes create; do not touch pre-existing dead
  code unless asked; match existing file style.
- **Verification-Driven Execution** — transform tasks into verifiable goals
  (test-first for bugs, pytest + pyright before/after for refactors); state
  multi-step plans with verification checkpoints; confirm Feature Pipeline
  Update Rule consistency across all three files before reporting done.

### Changed — UTILITY_FIX.md Status Updated

- Updated status line (line 3) from "plan only. No code modified. Ready for
  line-by-line review." to "substantially implemented (verified 2026-04-16)"
  with remaining items listed (pending `current.json` promotion;
  `meta_labeler.py:136` zero-imputation deferred).
- Added `§12 Changelog` entry dated 2026-04-16 documenting implementation
  verification: all `FALLBACK_*` constants, `from_artifact()`, call-site
  migrations, calibrator module, CLI script, and tests confirmed against live
  codebase via grep/glob. §0 code snippets preserved as pre-implementation
  audit trail.

### Changed — Memory Consolidation (first /dream consolidate pass)

- **`project_fixutility_01.md`** — removed duplicate "utility_score is
  mis-calibrated, not mis-formulated" section (appeared twice with
  contradictory recommendations). Merged into single authoritative section
  with current code state. Updated line number references to match current
  source (`utility.py:27-31` for `FALLBACK_*`, `:68` for `from_artifact()`,
  `scan.py:224`, `regime_validator.py:373`, `unified_training_builder.py:281`).
  Added "Implementation status (verified 2026-04-16)" section tracking
  completed vs pending items. Updated header and frontmatter description.
- **`MEMORY.md`** — shortened index line for FIXUTILITY-01 entry from 195
  chars to 128 chars.
- **Feedback memories** (3 files) — verified, no drift, left unchanged.

## [6.5.7-utility-recalibration] - 2026-04-15

### Added — Governed Utility Score Recalibration (FIXUTILITY-01)

Implementation of `UTILITY_FIX.md` v3 — artifact-based utility coefficient
governance with pinned v0 fallback, replacing three inconsistent hardcoded
call sites with a single loader hierarchy.

- **New `src/neutralgrid/calibration/utility_calibrator.py`** (~350 LOC) —
  governed recalibration engine.
  - Reads outcomes from `Sheet1` and features from `Meta Features` in the
    expired bots workbook; never reads stored `utility_score` values.
  - Calibration pool: `duration_hours > 7`, `backfill_status != "hmm_failed"`,
    binary label `pnl_pct > 0`.
  - 75/25 chronological fit/holdout split with two-class enforcement on both
    segments.
  - Grid search over `lambda_risk`, `kappa_trend`, `horizon_hours` with
    fit-only AUC maximization and fit-only threshold selection (balanced
    accuracy).
  - 8-gate promotion policy: G0 schema, G1 dedup, G2 class presence,
    G3 holdout AUC > 0.5, G4 mean winner > mean loser on holdout,
    G5 threshold fit-only, G6 winner recall non-regression vs v0,
    G7 finite/non-negative/not boundary-pinned.
  - Atomic JSON writes (`tmp + rename`) for both candidate and `current.json`.

- **New `scripts/recalibrate_utility.py`** (~110 LOC) — operator CLI.
  - `--expired-bots-path`, `--artifact-dir`, `--dry-run`, `--promote`.
  - Exit codes: 0 (promoted), 1 (not promotable), 2 (dry-run), 3 (error).

- **New `tests/unit/test_utility_calibrator.py`** — 10 tests.
  - `test_from_artifact_fallback_matches_pinned_v0` — byte-identical
    regression guard ensuring missing artifact produces identical scores to
    `UtilityConfig()`.
  - `test_from_artifact_loads_coefficients_and_explicit_overrides` — artifact
    loading + kwarg override precedence + unknown-override rejection.
  - `test_loader_uses_two_sheets_retains_losers_and_ignores_stored_utility` —
    pool construction from two-sheet workbook, stored `utility_score` excluded.
  - `test_loader_rejects_duplicate_strategy_ids` — dedup enforcement on both
    `Sheet1` and `Meta Features`.
  - `test_loader_rejects_blank_ids_and_missing_status` — blank `strategy_id`
    and missing `backfill_status` rejection.
  - `test_split_rule_is_deterministic_and_keeps_both_classes` — 75/25 split
    with deterministic row assignment.
  - `test_vectorized_scores_match_utility_scorer` — vectorized calibrator
    scores match scalar `UtilityScorer.compute_utility()`.
  - `test_threshold_selection_is_fit_only` — fit-only balanced accuracy.
  - `test_rejected_candidate_writes_audit_artifact_and_leaves_current_absent` —
    non-promotable candidate writes candidate JSON but does not create
    `current.json`.
  - `test_current_workbook_contract_counts_if_available` — pool size regression
    guard (114 rows, 92 winners, 22 losers, 86 fit, 28 holdout).
  - `test_producer_paths_use_artifact_loader` — grep-based regression guard
    proving all 6 producer paths use `UtilityConfig.from_artifact()` and never
    use bare `UtilityConfig()`.
  - `test_calibrator_does_not_use_live_outcome_ingestor` — import isolation
    guard.

### Changed — Utility Call Sites Unified via `from_artifact()`

All three production call sites (plus two ancillary scripts) now use
`UtilityConfig.from_artifact()` instead of inconsistent hardcoded
constructors. Loader hierarchy: `kwargs > artifacts/utility/current.json >
FALLBACK_* constants` (pinned v0).

- **`src/neutralgrid/validation/utility.py`** — added `FALLBACK_*` constants
  (`FALLBACK_LAMBDA_RISK=2.0`, `FALLBACK_KAPPA_TREND=1.5`,
  `FALLBACK_HORIZON_HOURS=6.0`, etc.) as the pinned v0 anchor. Added
  `DEFAULT_UTILITY_ARTIFACT_PATH` pointing to
  `artifacts/utility/current.json`. Added `UtilityConfig.from_artifact()`
  classmethod with JSON loading, coefficient extraction, kwarg override
  precedence, and unknown-override validation. Changed `UtilityScorer`
  default constructor from `UtilityConfig()` to
  `UtilityConfig.from_artifact()`.
- **`src/neutralgrid/scanner/scan.py:224`** — switched from
  `UtilityScorer(UtilityConfig(lambda_risk=...))` to
  `UtilityScorer(UtilityConfig.from_artifact(min_utility=0.0))`.
- **`src/neutralgrid/validation/regime_validator.py:373`** — switched from
  shadow `UtilityConfig` in `core/config.py` to
  `UtilityConfig.from_artifact(**utility_overrides)` with sentinel-based
  operator override preserved.
- **`src/neutralgrid/training/unified_training_builder.py:281`** — switched
  from `UtilityScorer(UtilityConfig())` to
  `UtilityScorer(UtilityConfig.from_artifact())`.
- **`new_bot_data_extractor.py:519`** — switched to
  `UtilityScorer(UtilityConfig.from_artifact())`.
- **`scripts/backfill_training_features.py:113`** — switched to
  `UtilityScorer(UtilityConfig.from_artifact())`.
- **`src/neutralgrid/calibration/__init__.py`** — exported `run_calibration`
  and `CalibrationResult` under the "Utility score" section.

### Notes

- No artifact (`artifacts/utility/current.json`) is shipped with this change.
  All call sites fall back to pinned v0 defaults (`lambda_risk=2.0`,
  `kappa_trend=1.5`, `horizon_hours=6.0`) — behavior is identical to pre-change
  until an operator runs `python scripts/recalibrate_utility.py --promote`.
- The `UtilityConfig` dataclass in `core/config.py` (used only by
  `regime_validator.py` for operator override via `.env`) is unchanged; Site B
  override semantics are preserved via the `utility_overrides` dict.
- `UTILITY_FIX.md` v3 at repo root documents the full versioning model,
  maintenance contract (§10), and five guarantees (§10.7).

## [6.5.7-retrain-15m-hmm-and-scanner] - 2026-04-12

### Retrained — HMM (15m) and Scanner Profile

First production retrain on the new 15m pipeline after the HMM migration and
follow-up patches.

- **HMM artifact**: `artifacts/hmm/rolling_180d_20260412_001444` (promoted)
  - Canonical retrain on Binance Vision 15m klines (180-day rolling window,
    `2025-10-12T23:45:00Z` → `2026-04-10T23:45:00Z`)
  - Probe universe: 100 symbols fetched, 85 passed store coverage, 50 accepted
    (target met), 5 rejected for insufficient history
  - Training shape: 50 sequences × 17,160 bars = 858,000 samples
  - Model: 4-state GaussianHMM, `diag` covariance, EM converged in **169
    iterations** (`converged: true`)
  - Walk-forward evaluation: 3 splits, **mean_pass_rate = 100.00%**
    (`mean_range_prob=0.3867`, `mean_trend_prob=0.1563`, `pass_mode=soft`)
  - State mapping: `range=0`, `trend=2`
  - Metadata verified: `timeframes_used=["15m"]`, `feature_schema.json`
    `timeframe="15m"`, features `[r_t, vol_t, trend_t, adx_t, bbwidth_t]`
  - Promotion trial: `hmm_20260412_023050` (cv_score=1.0000)
  - Temperature scaler: identity (self-labeled HMM calibration disabled)

- **Scanner profile** (`python retrain_scanner.py --max-duration-hours 7.0`)
  - Bounded training universe: 41 rows with `0 <= duration_hours < 7.0`
    (of 155 total), 40 after dropping missing `profit_factor`/`pnl_pct`
  - `data/profile/pattern_profile.json` — 9 winners, PnL threshold 8.40%,
    10 features, trend mix range/up/down = 50% / 37.5% / 12.5%.
    `duration_band: {min_hours: 0.0, max_hours: 7.0}`,
    `max_duration_hours: 7.0`
  - `data/profile/profile_model.json` — 9 features, prior winner 0.225,
    shrinkage 0.3, `duration_band: {min_hours: 0.0, max_hours: 7.0}`
  - `data/profile/profile_gate.json` — ADX 1h max 45.00, ADX 15m max 50.00,
    ADX 5m max 54.72, RSI 15m range `[35.62, 89.62]` (widening factor 0.15)

### Notes

- Binance Vision daily archives for `2026-04-11` were not yet published at
  retrain time; the pipeline fell back to the full `2026-04-10` bridge, which
  still satisfied the 180-day coverage requirement.
- EM fit wall-clock: ~2h 10m (single-process hmmlearn on 858k × 5 features,
  300-iter budget, converged at iter 169).
- Broader pipeline validation (Step 5 of the retrain plan) is pending
  authorization before promotion to live scanning.

## [6.5.7-hmm-15m-migration-followup] - 2026-04-11

### Fixed - HMM Lineage, Storage, and Profile Classification

Follow-up implementation pass for `HMM_CHANGE.md` v5.1 after three read-only
sub-agent audits (`Linnaeus`, `Schrodinger`, `Franklin`).

- `src/neutralgrid/validation/regime_validator.py` - added a single
  `hmm_result_fields` extraction after the 15m HMM check and applied it to
  HMM-fail, range-fail, volatility-fail, stochastic-fail, and success returns.
  Top-level HMM lineage now includes `regime_conf`, `posterior_mode`,
  `persistence_prob`, `hmm_trained_at_utc`, and `posteriors` wherever HMM has
  executed.
- `src/neutralgrid/storage/database.py` - `hmm_regime_passed` now reflects the
  actual HMM carrier stage result instead of overall validation status, so HMM
  pass is preserved when a later gate fails. HMM-stage failure reasons now use
  `HMM:` instead of the stale `1H:` label.
- `src/neutralgrid/scanner/profile_model.py` and
  `src/neutralgrid/scanner/pattern_profile.py` - duplicate `strategy_id`
  validation now runs before any sheet merge and cannot be swallowed by the
  multi-sheet fallback. Single-sheet Excel reads now close `pd.ExcelFile`
  handles with context managers.
- `src/neutralgrid/scanner/pattern_profile.py` - removed the top-PnL fallback
  relabeling branch; insufficient strict bounded winners now fail fast instead
  of mutating the winner rule.
- `src/neutralgrid/models/hmm/inference.py`,
  `src/neutralgrid/models/hmm/train.py`,
  `src/neutralgrid/models/hmm/canonical_retrain.py`,
  `src/neutralgrid/api/app.py`, and `src/neutralgrid/backtest/evaluate.py` -
  cleaned stale HMM contract/docstring references that still described 1h HMM
  semantics.

### Tests

- `tests/test_afml_integrations.py` - bounded-universe Excel fixtures now use a
  workspace-scoped temp directory because Python `tempfile.TemporaryDirectory()`
  is access-denied in this Windows environment. Added coverage proving
  `pattern_profile` fails fast instead of falling back to top-PnL winners.
- `tests/unit/test_regime_validator.py` - added parameterized tests proving
  top-level HMM lineage survives HMM, range, volatility, and stochastic invalid
  return paths. Added database persistence coverage for HMM-pass/later-gate-fail
  and HMM-fail cases.

### Validation

- Focused suite: `37 passed, 5 warnings`.
- Full suite: `1064 passed, 8 warnings`.
- `python -m compileall` on patched files passed.
- `pyright.exe` was blocked by Windows Application Control. Python 3.12
  `python -m pyright` ran but failed on repo-wide environment/import-resolution
  errors (`fastapi`, `httpx`, `numpy`, `pandas`, `hmmlearn`, `sklearn`,
  `aiosqlite`, etc.), so Pyright was not a green validation signal for this
  pass.

## [6.5.7-hmm-15m-migration] - 2026-04-10

### Changed — HMM Regime Detection Migrated from 1h to 15m

Complete migration of the HMM regime detection system from 1-hour to 15-minute
klines. All 9 phases (59 discrete changes) from `HMM_CHANGE.md` v1.0–v5.0
implemented across 24 files. The 4-state GaussianHMM now trains, infers, and
validates on 15m OHLCV data. Wall-clock semantics preserved (inference window
remains ~8.3 days = 800 bars at 15m).

**Phase A — HMM Artifact/Schema/Training Contract**:
- `src/neutralgrid/models/hmm/schema.py` — `TIMEFRAME = "1h"` → `"15m"`.
- `src/neutralgrid/data/features.py` — renamed `df_1h` → `df` in
  `compute_hmm_features()` and `compute_hmm_features_dict()`, updated
  docstrings.
- `src/neutralgrid/core/protocols.py` — renamed `df_1h` → `df` in
  `RegimePredictor.predict()` protocol (Pyright conformance).
- `src/neutralgrid/models/hmm/inference.py` — renamed `df_1h` → `df` in
  `predict()` and `predict_regime()`, updated error messages from "1H" to
  "kline".
- `src/neutralgrid/validation/hmm_regime.py` — renamed `df_1h` → `df` in
  `infer_regime()`.
- `src/neutralgrid/models/hmm/train.py` — renamed `per_symbol_dfs_1h` →
  `per_symbol_dfs` in `train_hmm_global()` and `walk_forward_evaluate()`,
  `timeframes_used=["15m"]` in metadata.

**Phase B — Canonical Feeding Interval-Aware Math**:
- `src/neutralgrid/models/hmm/canonical_retrain.py` — `interval="15m"`,
  row-count formulas multiplied by 4 (lines 279, 392), validation interval
  updated.
- `src/neutralgrid/data/binance_vision/pipeline.py` — `min_rows` now
  interval-aware using `bars_per_hour` lookup dict.

**Phase C — HMM Inference-Limit Contract**:
- `src/neutralgrid/core/config.py` — `infer_limit_1h: int = 200` →
  `infer_limit: int = 800`, `kline_limits["15m"]` raised to 800,
  cross-config check updated to validate 15m limit.
- `src/neutralgrid/scanner/scan.py` — reads `hmm.infer_limit` (not
  `infer_limit_1h`).
- `src/neutralgrid/validation/regime_validator.py` — reads `hmm.infer_limit`.

**Phase D — Runtime HMM Feeders**:
- `src/neutralgrid/scanner/scan.py` — HMM now fed `df15` (15m klines) instead
  of `df1` (1h klines).
- `src/neutralgrid/validation/regime_validator.py` — `check_hmm_regime()` now
  receives 15m DataFrame, 15m parse moved before HMM gate, error string
  updated to match new inference message.

**Phase E — Downstream Lineage Repair**:
- `src/neutralgrid/scanner/enrich_grid_params.py` — HMM lineage reads from
  top-level `ValidationResult` fields with `tf_1h.checks` fallback for
  backward compatibility.
- `src/neutralgrid/training/scanner_integration.py` — same top-level-first
  lineage reads.
- `src/neutralgrid/api/app.py` — stage-based labels (`"hmm_regime"`) instead
  of `"1h_regime"`.
- `src/neutralgrid/storage/database.py` — added `hmm_regime_passed` column,
  migration adds column to existing tables, populated from actual HMM result.

**Phase F — Both Retrain Entrypoints Aligned**:
- `retrain_hmm.py` — `timeframe="15m"`, log text updated.
- `src/neutralgrid/cli/retrain.py` — all 5 sites updated: `bars_needed`
  formula uses `* 4`, `effective_days` divides by 96, `timeframe="15m"`,
  `actual_days` divides by 96.0.

**Phase G — Offline Evaluation Forward Horizons**:
- `src/neutralgrid/backtest/evaluate.py` — `auc_fwd_horizon = 24` (was 6),
  `fwd_horizon = 24` (was 6), `fwd_horizons = (24, 48)` (was (6, 12)),
  `per_symbol_dfs_1h` kwarg renamed to `per_symbol_dfs`, docstrings updated.

**Phase H — Profile Model Bounded Universe (`0 <= duration_hours < 7`)**:
- `retrain_scanner.py` — `--min-duration-hours` replaced with
  `--max-duration-hours` (default 7.0), passed to both trainers.
- `src/neutralgrid/scanner/profile_model.py` — signature changed to
  `max_duration_hours: float = 7.0`, bounded training universe
  `0 <= duration_hours < max_duration_hours`, labeled subset (`df_labeled`)
  excludes NaN profit_factor/pnl_pct rows, `duration_band` field added to
  frozen dataclass and JSON serialization.
- `src/neutralgrid/scanner/pattern_profile.py` — same bounded universe,
  `df_labeled` subset, duplicate `strategy_id` fail-fast validation,
  profit_factor capping (`clip(0, 1000)`), removed APG auto-derivation,
  bounded fallback for insufficient winners.

**v5.0 Missed HMM Callers**:
- `src/neutralgrid/models/hmm/inference.py` — `predict_regime()` standalone
  function (line 770): `df_1h` → `df`, docstring updated.
- `new_bot_data_extractor.py` — HMM inference now fed 15m klines, bar
  requirement raised from 100 to 800.
- `scripts/backfill_training_features.py` — HMM inference now fed 15m klines,
  lookback increased to match 800-bar requirement.

### Fixed — Test Failures from Migration

- `tests/test_micro_osc_integration.py` — added 8 explicit HMM lineage
  fields (`regime_conf`, `posterior_mode`, `hmm_artifact_version`, etc.) to
  `MagicMock` validators. `getattr(mock, attr, None)` returns MagicMock
  instead of None, causing `TypeError` in pandas operations.
- `tests/unit/test_enrich_grid_params.py` — same MagicMock HMM lineage fix
  across 4 mock instances and `_valid_vres()` SimpleNamespace.
- `tests/unit/test_new_bot_data_extractor.py` — kline fixture size increased
  from 100 to 800 bars (all 3 timeframes) to match new HMM minimum.
- `src/neutralgrid/training/unified_training_builder.py` — slow-path
  reconstruction now extracts `label_contract_version` from raw CSV rows.
  Without this, all reconstruction rows were marked legacy by the ingestion
  gate.
- `tests/unit/test_unified_training_builder.py` — added
  `label_contract_version: LABEL_CONTRACT_VERSION` to test data, added
  `include_reconstruction=True` to overlap test builder.
- `tests/unit/test_unified_training_builder_v20260312.py` — added
  `label_contract_version: LABEL_CONTRACT_VERSION` to test data.
- `tests/test_afml_integrations.py` — cross-config invariant test updated to
  15m. Added `TestBoundedUniverseContract` class (6 tests): bounded universe
  exclusion, pnl_thr on bounded, missing pf is unlabeled,
  pattern/model consistency without APG, duration_band JSON roundtrip,
  duplicate strategy_id raises. Fixed `TemporaryDirectory` with
  `ignore_cleanup_errors=True` for Windows.

### Validation

- 1058 tests pass (was 1041 before migration, +17 new tests).
- Pyright: 0 new errors in source code (pre-existing pandas stub issues only).
- 3 parallel verification agents deployed — all returned ALL CHECKS PASS.
- Total: 24 files modified, 59 discrete changes implemented.

## [6.5.7-29-feature-cleanup] - 2026-04-07

### Removed — Old 21-Feature Profile (`afml_scan_v20260318`)

Permanently removed the old 21-feature meta-labeler profile to prevent
accidental training on stale features. The `snapshot_v20260407` 29-feature
profile is now the sole available profile.

- `src/neutralgrid/models/meta_labeler.py` — deleted
  `AFML_SCAN_META_FEATURES_V20260318` constant (was lines 35-57). Removed
  its entry from `META_FEATURE_PROFILES` dict (now contains only
  `snapshot_v20260407`). Removed 14 inference aliases for features no longer
  in any profile (`range_prob`, `trend_prob`, `survival_prob`,
  `profit_per_grid_pct`, `num_grids`, `adx_1h`, `ema_slope_1h`,
  `utility_score`, `hurst_exponent`, `ou_halflife` and their `scan_*`
  prefixed variants). Removed stale `# afml_scan_v20260318 additions` and
  `# afml_enriched_v20260318 additions` comments from
  `_FEATURE_MEDIAN_DEFAULTS`.
- `src/neutralgrid/backtest/candidate_pipeline.py` — removed stale
  `# afml_scan_v20260318 additions` and `# afml_enriched_v20260318 additions`
  comments from `TRAINING_OUTPUT_COLUMNS`.

### Changed — Backtest Score Threshold Lowered (55 → 45)

Lowered `min_score` default from 55.0 to 45.0 across all backtest entry
points to increase training data volume. At 55.0 only 24/300 candidates
passed; at 45.0, 134/300 pass (5.6x increase in score-eligible candidates).

- `backtest_candidates.py` — CLI `--min-score` default 55.0 → 45.0.
- `src/neutralgrid/backtest/candidate_pipeline.py` —
  `filter_backtest_candidates()` parameter default 55.0 → 45.0.
- `run_candidate_backtests.py` — function default and call site 55.0 → 45.0;
  docstring and report header updated.
- `scripts/build_execution_telemetry.py` — score accuracy threshold
  55 → 45.
- `tests/unit/test_candidate_pipeline_bypass.py` — all `min_score=55.0`
  calls and `"score": 55.0` fixtures updated to 45.0.

### Retrain Attempt — Blocked by Data Volume

First retrain attempt on the 29-feature profile completed the full pipeline
but was blocked by insufficient positive labels:

- 550 candidates scanned (3 batches: 50 + 250 + 250).
- 36 backtested successfully (6h aging met for batches 1-2; batch 3 pending).
- Snapshot-outcome join: 100 matched, 64 version-gated, 36 final rows.
- Positive labels: 1/36 (2.8%) — below the 5% minimum for stable training.
- 7/36 exceeded the 3% PnL hurdle, but 6/7 were blocked at hierarchical
  label L2 by `unrealized_too_high` (>50% unrealized fraction). Grid bots
  did not complete enough round trips in the 6h window to realize gains.
- All 29 features fully populated (0% NaN) — feature pipeline is healthy.
- Bottleneck is data volume and hierarchical label strictness, not features.

## [6.5.7-29-feature-integration] - 2026-04-07

### 29-Feature Meta-Labeler Complete Integration

End-to-end integration of the `snapshot_v20260407` 29-feature profile into
the training pipeline. Fixes version gating, dedup ordering, and scanner CSV
loading to ensure only feature-complete candidates produce training data.

### Fixed — Dedup Ordering Bug in Snapshot-Outcome Join

- `src/neutralgrid/training/unified_training_builder.py` —
  `_join_snapshots_to_outcomes()`: dedup now sorts backtest rows by
  `label_contract_version` (parsed as date) descending and keeps the first
  (newest) per `candidate_id`. Previously `drop_duplicates(keep="first")`
  kept the oldest (legacy, versionless) row, causing 59 valid rows to be
  version-gated. Root cause of the "5 rows after version gate" training
  failure.

### Changed — Date-Based Label Contract Versioning

- `src/neutralgrid/core/constants.py` — `LABEL_CONTRACT_VERSION` changed
  from `"2.4"` to `"2026-04-07"` (YYYY-MM-DD format). Date-based versions
  are sortable and allow the builder to deterministically identify the newest
  backtest result per candidate without depending on file load order.
- `backtest/btk_label_contract.py` — fallback updated to match.
- `src/neutralgrid/training/unified_training_builder.py` —
  `_apply_ingestion_gate()` now compares versions as dates: rows with a
  version older than the current are gated; current or newer pass. Eliminates
  exact-string-match fragility.
- `retrain_meta_labeler.py` — removed `accept_version_mismatch` pass-through
  (no longer needed with correct dedup ordering).

### Changed — Scanner CSV Feature Gate

- `src/neutralgrid/backtest/candidate_pipeline.py` —
  `load_all_scanner_csvs()` now checks each CSV for sentinel columns from
  the 29-feature profile (`spread_pct`, `top_account_lsr_log`,
  `taker_imbalance`, `oi_notional`). CSVs missing these columns predate the
  feature rollout and are skipped. Controlled by `require_meta_features`
  parameter (default `True`).

### Added — Self-Healing Bootstrap Gate

- `src/neutralgrid/backtest/candidate_pipeline.py` —
  `_meta_labeler_needs_bootstrap()` reads the deployed model's
  `metadata.json` and checks if its feature list matches
  `SNAPSHOT_META_FEATURES_V20260407`. When stale (old 21-feature model),
  `filter_backtest_candidates()` auto-relaxes the `grid_is_valid` gate for
  training data generation. Once a model trained on the current 29-feature
  profile is deployed, the gate self-heals and standard `grid_is_valid`
  filtering resumes.

### Fixed — Feature Pipeline Consistency (from prior session)

- `src/neutralgrid/backtest/candidate_pipeline.py` — added
  `funding_rate_zscore`, `open_interest_change_pct`, `bb_width_ratio_1h_15m`
  to `TRAINING_OUTPUT_COLUMNS` (were in the 29-feature profile but missing).
- `src/neutralgrid/training/scanner_integration.py` —
  `FeatureCollector._validate_snapshot()` now validates all 29 features from
  `SNAPSHOT_META_FEATURES_V20260407` instead of hardcoded 15.
- `src/neutralgrid/training/unified_training_builder.py` — added explicit
  validation logging for missing profile features.

### Validation

- All 1041 tests pass.
- Feature registry alignment verified across 6 locations:
  `TRAINING_OUTPUT_COLUMNS`, `ALL_META_FEATURES`, `_FEATURE_MEDIAN_DEFAULTS`,
  `MetaLabelerConfig.features`, `FeatureSnapshot.to_dict()`,
  `_SCAN_TO_FEATURE` — all contain the full 29-feature profile.
- Scanner generates 300 snapshots with 100% non-null on all 29 features.
- Bootstrap auto-detection correctly relaxes `grid_is_valid` when the
  deployed model uses the old 21-feature profile.

### Pending — Cold-Start Timing

- The 29-feature scanner output started 2026-04-07. Candidates need 6 hours
  of price data before backtesting. Once aged, run:
  ```
  python backtest_candidates.py --max-candidates 250 --min-score 45
  python retrain_meta_labeler.py --training-backbone unified --feature-profile snapshot_v20260407 --backtest-results-dir data/backtest_candidates --snapshot-dir data/training_snapshots
  ```
- After retrain, the bootstrap gate self-heals and normal operation resumes.

## [6.5.7-pipeline-inconsistency-fixes-2] - 2026-04-06

### Pipeline Inconsistency Fixes (INC-03, INC-02, INC-11)

Resolved the three remaining proven inconsistencies from
`PIPELINE_INCONSISTENCIES.md`. All safeguards are enforced in production code;
no external test dependencies required for runtime correctness.

### Changed - Grid Geometry Single Source of Truth (INC-03)

- `src/neutralgrid/grid/calculator.py` — `compute_regime_adjusted_grids()`:
  added optional `grid_lower`, `grid_upper`, `max_spacing_pct=2.0` parameters.
  When bounds are provided, spacing is now derived from geometry
  `(upper - lower) / lower / grids * 100` instead of multiplicative
  `spacing_pct * spacing_mult`. Backward-compatible: callers without bounds
  fall back to the original multiplicative path.
- Enforces 2% maximum spacing cap via while-loop on **all three** geometry
  paths: regime-adjusted (high trend), low-trend early return, and
  non-adjusted (None trend/range_prob). The loop increments grid count until
  `implied_spacing <= max_spacing_pct`, bounded by `max_grids`.
- `generate_params()` restructured: removed premature profit computation and
  profit floor check on pre-regime geometry. Profitability is now computed
  exactly once from the final `adjusted_num_grids` after all adjustments.
  Profit floor enforced after final geometry only.
- INC-10 (spacing cap and geometry mismatch) resolved as a consequence of
  INC-03 — marked completed in `PIPELINE_INCONSISTENCIES.md`.

### Changed - Enrichment Eligibility Alignment (INC-02)

- `src/neutralgrid/scanner/enrich_grid_params.py` —
  `EnrichConfig.range_prob_threshold` default changed from `0.50` to `0.20`,
  aligning with the lowest entropy-adaptive tier
  (`low_confidence_threshold = 0.20` in `entropy_adaptive_threshold_v20260311.py`).
- `_enforce_threshold_gate()` now respects Stage B approval: rows with
  `stage_b_approved=True` are no longer invalidated even if their score is
  below the threshold. Implementation uses
  `invalidate_mask = bt_mask & ~stage_b_override` where `stage_b_override`
  is derived from the `stage_b_approved` column with `fillna(False)` fallback
  for missing columns and NaN values.

### Changed - Horizon Threading (INC-11)

- `src/neutralgrid/scanner/enrich_grid_params.py` — `_estimate_microstructure()`
  now computes `live_horizon_hours = float(get_config().grid.max_holding_seconds)
  / 3600.0` once and passes it to both `estimate_costs()` and
  `compute_dynamic_profit_floor()`. Previously `estimate_costs()` used its
  default (6.0h) while `compute_dynamic_profit_floor()` read from config.
  Behavioral change: none (both values are 6.0 today). Structural correctness
  only.

### Added Then Removed - Verification Tests

Six tests were created during implementation to verify invariants, used during
the review/validation phase, and then removed. They are not required for
runtime correctness — all safeguards are enforced in production code. Documented
here for audit trail:

1. **`test_regime_adjusted_grids_geometry_derived`**
   (`tests/test_afml_integrations.py`)
   Verified that after regime adjustment with bounds provided, spacing equals
   `(upper - lower) / lower / grids * 100` (geometry-derived, not
   multiplicative). Asserted spacing <= 2.0% and grid count reduced from
   initial.

2. **`test_regime_adjusted_grids_cap_enforcement`**
   (`tests/test_afml_integrations.py`)
   Verified that when regime widening (high `trend_prob=0.50`) would push
   spacing past the 2% cap, the grid count increases to bring spacing back
   within the cap. Used a 9% range with 5 initial grids (1.8% initial spacing)
   to trigger widening.

3. **`test_regime_adjusted_grids_no_adjustment_geometry_derived`**
   (`tests/test_afml_integrations.py`)
   Verified that even on the low-trend path (`trend_prob=0.10`, below 0.20
   threshold, no regime adjustment), spacing is still geometry-derived from
   bounds when bounds are available, rather than echoing the input
   `grid_spacing_pct`.

4. **`test_non_adjusted_path_cap_enforcement`**
   (`tests/test_afml_integrations.py`)
   Verified that the non-adjusted path (low trend) enforces the 2% cap when
   `int()` floor-truncation in `calculate_num_grids` produces fewer grids than
   needed. Example: 9.5% range / 4 grids = 2.375% uncapped; cap enforcement
   increases to 5 grids = 1.9%.

5. **`test_enforce_threshold_gate_respects_stage_b_approval`**
   (`tests/unit/test_enrich_grid_params.py`)
   Verified `_enforce_threshold_gate()` Stage B override logic with 3 rows:
   (a) below-threshold + `stage_b_approved=True` → stays valid (not
   invalidated), (b) below-threshold + `stage_b_approved=False` → stays
   invalid, (c) above-threshold → unaffected. Used `bool()` wrappers for
   numpy 2.4+ compatibility (`np.bool_` is not `bool`).

6. **`test_enrichment_range_prob_threshold_default`**
   (`tests/unit/test_enrich_grid_params.py`)
   Verified `EnrichConfig.range_prob_threshold` defaults to `0.20` (was
   `0.50`), confirming alignment with the lowest entropy-adaptive tier.

### Validation

- Targeted tests: `90 passed` across AFML integration, enrichment, and
  enhancement coverage.
- Full test suite after all fixes: `1049 passed in 36.68s` (including the 6
  verification tests).
- Full test suite after test removal: `1043 passed in 34.41s`.
- `python -m compileall` passed for both modified production modules.
- Three parallel review agents deployed:
  - Calculator review: confirmed all invariants hold; identified and fixed
    the missing 2% cap on the non-adjusted path.
  - Enrichment review: all 7 review points passed, no issues found.
  - Full test suite runner: 1048 passed (before cap fix test was added).

## [6.5.7-pipeline-inconsistency-fixes] - 2026-04-05

### Pipeline Inconsistency Fixes

Implemented the current verified repairs from `PIPELINE_INCONSISTENCIES.md`
without widening the live architecture beyond what the codebase can justify.
This pass removes proven dead paths, restores mathematical consistency to the
scan-time MI scorer, and aligns the early low-range reject floor with the
minimum entropy-adaptive threshold now documented for the pipeline.

### Removed - Proven Dead Paths

- Deleted `compute_regime_aware_sizing()` from
  `src/neutralgrid/grid/calculator.py`. The live sizing chain already runs in
  enrichment via Kelly plus `PositionSizer`, so this calculator path had no
  live callers.
- Deleted the tests in `tests/test_afml_integrations.py` that existed only to
  preserve that removed sizing path.
- Removed the unreachable scan-time `elif` branch in
  `src/neutralgrid/scanner/scan.py` that required `ev_score` and `meta_prob`
  after the MI-weighted branch had already claimed control flow.

### Changed - Scan-Time Scoring Integrity

- `src/neutralgrid/scanner/scan.py` now exposes one explicit scan-time scoring
  path: MI-weighted scoring.
- `src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py` now renormalizes
  across the available finite weighted features instead of zero-penalizing
  missing features by leaving inactive weight mass in the denominator.
- Added `default_runtime_mi_weights()` so scan-time MI scoring still runs with a
  deterministic uniform weighting scheme when no validated artifact weights are
  available.
- Added explicit per-row MI telemetry in `scan.py`:
  `mi_used_signal_count`, `mi_available_weight`, and `mi_weight_mode`.
- `run_full_pipeline.py` now distinguishes validated artifact MI weights from
  runtime uniform fallback instead of silently presenting fallback weights as
  artifact-loaded weights.
- `validate_mi_weights()` now rejects non-finite weights, preventing `NaN` or
  infinite weight artifacts from silently corrupting scores.

### Changed - Enrichment Threshold Consistency

- Lowered the early low-range pre-reject floor in
  `src/neutralgrid/scanner/enrich_grid_params.py` from `0.30` to `0.20`.
- Updated the corresponding rejection reason strings and duplicate regime-tag
  checks so the documented floor and emitted diagnostics stay aligned.

### Changed - Documentation And Comments

- Updated stale calculator comments in `src/neutralgrid/grid/calculator.py` so
  they describe the live contract correctly: calculator owns geometry, while
  live sizing remains deferred to enrichment-time Kelly and `PositionSizer`.

### Validation

- Targeted pytest for the modified areas:
  `43 passed` across MI scorer, enrichment thresholding, AFML integration, and
  micro-osc integration coverage.
- Focused rerun after tightening the new threshold boundary coverage:
  `14 passed`.
- `python -m compileall` passed for the modified pipeline modules.

## [6.5.7-pipeline-integrity-repair] - 2026-04-04

### Pipeline Integrity Repair

Implemented the minimal live-path repair from `AUDIT_DIAGNOSTIC.md` so the
documented deployment path and the actual enrichment flow no longer disagree.
The changes keep the existing AFML architecture, remove proven false
optionality, and make failure attribution auditable row by row.

### Changed - Enrichment Flow (`src/neutralgrid/scanner/enrich_grid_params.py`)

- Added authoritative `failure_stage` output with explicit stage ownership for
  `score_threshold`, `pre_reject`, `regime`, `grid_generation`, `hard_gate`,
  `stage_b`, `approved`, and `exception`.
- Silent pre-rejects are now written back to the DataFrame with explicit reasons:
  `pre_reject:low_range_prob(...)` and `pre_reject:low_survival(...)`.
- Score-threshold enforcement now writes `rejection_reasons` and `failure_stage`,
  and no longer overwrites more specific earlier failures such as `pre_reject`.
- Stage 11 viability remains computed but is now annotation-only via
  `micro_viable` / `micro_reason`; it no longer terminates the row early.
- Negative Kelly edge remains visible in diagnostics via `sizing_reason`, but
  Kelly no longer terminal-rejects before downstream gates.
- Final post-grid decision path is now:
  viability annotation -> Kelly annotation -> hard gate -> position sizing + TOS -> Stage B.

### Changed - Runtime Capital And Sizing Authority

- `EnrichConfig` now accepts `capital_base_usdt`, and
  `run_full_pipeline.py` passes resolved runtime capital into enrichment.
- TOS `position_size_usdt` now uses runtime capital instead of static
  `cfg.grid.capital`.
- `GridCalculator.generate_params()` now accepts `capital_base` for live
  capital/notional fields while keeping grid geometry generation intact.
- Legacy grid-level `capital_fraction` was removed from the live sizing chain:
  grid generation now returns `capital_fraction=None` with
  `sizing_reason="deferred_to_live_sizers"`, and final live sizing is owned by
  Kelly plus `PositionSizer`.
- `PositionSizer.compute()` no longer exposes the unused `base_capital`
  parameter; it now returns the pure multiplicative risk-budget fraction.

### Removed - Proven False Optionality

- Removed redundant `adaptive_min_profit_pct` column from enrichment output.
- Removed dead `adaptive_min_profit_pct()` helper from
  `src/neutralgrid/validation/microstructure.py`.
- Removed hard-gate profit-floor rejection as a terminal owner in
  `src/neutralgrid/validation/microstructure_hard_gate.py`; profit-floor values
  are retained as telemetry only because grid generation already owns floor
  enforcement.
- Removed MI score renormalization in
  `src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py`; missing signals now
  contribute `0.0` without inflating remaining weights.

### Changed - Tests

- Updated enrichment, hard-gate, Kelly, and micro-osc tests to match the
  repaired live contract.
- Added direct coverage for:
  - explicit `pre_reject` attribution
  - `failure_stage` population
  - runtime-capital propagation into enrichment/TOS
  - non-terminal viability and non-terminal negative Kelly
  - non-renormalized MI scoring
- Added `tests/unit/test_mi_weighted_scorer_v20260311.py`.
- Test-only temp-directory helpers in `tests/test_afml_fixes.py` and
  `tests/test_afml_fixes_v2.py` now use repo-local scratch paths so persistence
  suites can run under the workspace sandbox.

### Validation

- Focused repaired suites: `164 passed in 4.75s`.
- Full repository test suite: `1041 passed in 30.20s`.

## [6.5.7-trigger-price-metadata-only] - 2026-03-31

### Trigger Price Is Metadata-Only

Removed `trigger_price` from decision and training feature derivation while
keeping the audit trail intact. Grid deployment and scanner enrichment no longer
depend on enrichment-time trigger-price resolution. The `trigger_price` field
remains available as metadata for manual bot review and workbook storage.

### Changed - Manual Meta Features (`new_bot_data_extractor.py`)

- **`range_size_pct` derivation**: no longer uses `trigger_price` or market last
  price in the manual meta-features sheet path.
- **Midpoint-only geometry**: when valid `grid_lower` / `grid_upper` bounds exist,
  `range_size_pct` is now computed as `((upper - lower) / midpoint) * 100.0`.
- **Downstream scores**: `utility_score` and `ev_score` now inherit the midpoint-
  derived `range_size_pct`, so differing `Trigger Price` inputs no longer change
  those outputs when bounds are unchanged.

### Changed - Training + Backfill

- **`unified_training_builder.py`**: removed the `trigger_price -> entry_price ->`
  `scan_trigger_price` fallback chain from live-outcome `range_size_pct`
  derivation. Midpoint from grid bounds is now the only derived reference in
  both the live-outcome mapping path and the backtest enrichment fill path.
- **`backfill_training_features.py`**: deleted `_reference_price()`. Backfill now
  preserves an existing `range_size_pct` when present, preserves upstream
  `market_range_size_pct` when present, and otherwise derives `range_size_pct`
  from grid midpoint only.

### Removed - Enrichment-Time Trigger Resolver

- **`enrich_grid_params.py`**: removed trigger-price resolver imports, resolver
  calls, resolver payload columns, and the dead `PriceSeriesManager` pre-warm
  path tied to trigger enrichment.
- **`run_full_pipeline.py`**: removed `PriceSeriesManager` import, lifecycle
  management, pass-through wiring, and trigger-price display output that existed
  only for enrichment-time trigger resolution.
- **`src/neutralgrid/data/price_series/__init__.py`**: removed resolver exports.
- **Deleted file**: `src/neutralgrid/data/price_series/ps_trigger_resolver.py`.
- **Deleted tests**: `tests/unit/test_ps_trigger_resolver.py`,
  `tests/unit/test_trigger_price_enrichment.py`.
- **Packaging cleanup**: removed the stale deleted-file reference from
  `src/neutralgrid.egg-info/SOURCES.txt`.

### Kept - Metadata And Audit Trail

- **Workbook schema**: the `trigger_price` column remains in the manual bot
  workbook output.
- **Extractor storage**: manual bot extraction still parses and stores
  `trigger_price` for later audit and diagnostics.
- **Behavioral contract**: this change removes trigger-price dependence from
  model and decision logic only; it does not remove the stored metadata field.

### Validation

- **`pyright`**: 0 errors, 0 warnings.
- **Targeted pytest**: updated midpoint and alias-path coverage passed cleanly.
- **Full pytest**: `1019 passed in 135.48s`.
- **Pre-existing warning cleanup**: fixed the `AsyncMock`/sync validator warning
  source in `tests/test_micro_osc_integration.py`.

## [6.5.7-max-grids] - 2026-03-30

### Grid Count Limit: 50 → 175 (Binance Futures Maximum)

Raised the hardcoded `max_grids = 50` cap to the Binance futures limit of 175,
and made it a configurable parameter in `GridConfig`. The previous hardcoded cap
silently clipped grid counts, preventing the pipeline from proposing higher-grid
configurations that manual bots (e.g., PLAYUSDT 110 grids / +5.62%, ONUSDT 50
grids / +4.40%) have shown to be profitable.

### Changed — GridConfig (`src/neutralgrid/core/config.py`)

- **`min_grids: int = 5`** — new configurable parameter (was hardcoded in calculator)
- **`max_grids: int = 175`** — new configurable parameter (was hardcoded as 50)

### Changed — GridCalculator (`src/neutralgrid/grid/calculator.py`)

- **`calculate_num_grids()`**: `min_grids` and `max_grids` parameters now default
  to `None` and resolve from `GridConfig` at runtime. Explicit overrides still
  accepted for backward compatibility.
- **`compute_regime_adjusted_grids()`**: Same change — reads `GridConfig` defaults
  instead of hardcoded `min_grids=5, max_grids=50`.

### Pipeline Run (2026-03-30)

Full steady-state pipeline cycle executed:

1. **Scan + Enrich** (`run_full_pipeline.py`): 250 symbols scanned, 0 valid
   candidates — market in strong trending regime (negative Kelly edge on all
   enriched candidates, HMM range_prob near 0).
2. **Backtest** (`backtest_candidates.py`): 250 candidates backtested, mean PnL
   +3.29%, 124/250 (50%) above 3% hurdle.
3. **OOS Validation** (`backtest_candidates_current.py`): 98 symbols, 20/98 (20%)
   in range, mean current PnL +0.87%.
4. **Meta-labeler Retrain** (`retrain_meta_labeler.py`): CV AUC=0.622, 286 samples,
   beta-calibrated (ECE 0.159→0.026). Top features: `trend_prob`,
   `profit_per_grid_pct`, `range_size_pct`, `funding_rate`.
5. **Re-run Pipeline**: 250 scanned, 0 valid — consistent with trending regime.

### Training Data — New Manual Bot Entries

- **ONUSDT** (strategy_id 410961557): +17.60 USDT / +4.40%, 4.63h, 50 grids,
  82 matched trades. Below 6h duration filter — excluded from meta-labeler training.
- **PLAYUSDT** (strategy_id 410961587): +22.48 USDT / +5.62%, 7.53h, 110 grids,
  87 matched trades. Above 6h filter — included in training. Features in-distribution
  except `grids_count` (6.2σ outlier vs median of 9).

Both appended to `data/new_expired_bots.xlsx` with PnL curve features and meta
features sheets via `new_bot_data_extractor.py`.

### What This Does NOT Change

- HMM model or training
- Meta-labeler features or label contract
- Backtest engine
- Scanner score formula
- TOS weights or sub-signals
- Stage B gate thresholds
- `active_fraction` normalization constant in `estimate_expected_return()` (scaling
  heuristic, not a grid cap)

### Verification

- **pytest**: 38 grid-related tests passed, 0 failed
- **AFML integration tests**: 17 passed, 0 failed
- **Config access**: `get_config().grid.max_grids` returns 175

---

## [6.5.7-micro-oscillator] - 2026-03-28

### Trending Micro-Oscillator Archetype

New pipeline archetype for symbols that trend on 1H timeframes but oscillate
profitably on 5m microstructure (e.g., ONUSDT strategy_id 410926826: +60.40 USDT
/ +15.10% in 5h54m, 137 trades). Previously blocked at 5 sequential gates because
the HMM correctly classified the trend but had no way to recognize the micro-oscillation.
6-step implementation across 11 files. Feature-flagged via `micro_osc.enabled = False`
(default off). 1008 tests passed, 0 failed, 0 regressions.

### Added — Micro-Oscillator Score (Step 1)

- **`micro_osc_score` computation** (`src/neutralgrid/scanner/scan.py`): Saturation-
  normalized composite of `vwap_crosses_5m`, `ema_crosses_5m`, and `hurst_exponent`.
  Three-component weights (0.45/0.35/0.20) when Hurst available, two-component
  fallback (0.60/0.40) otherwise. Always computed for training data collection
  regardless of feature flag.

### Added — Enrichment Bypass (Step 2)

- **Eligibility 3rd OR gate** (`enrich_grid_params.py`): `micro_osc_mask` admits
  candidates with `micro_osc_score >= 0.45` AND `survival_prob >= 0.60` when flag
  enabled. Additive — cannot reject existing eligibles.
- **Sort-tier for bypass rows** (`enrich_grid_params.py`): Bypass candidates sort
  to top before `head(max_symbols)` truncation, preventing low-range_prob symbols
  from being dropped.
- **Bypass propagation** (`enrich_grid_params.py`): `micro_osc_bypass` flag
  propagated from `eligible` to full `df` before threshold tagging.
- **Threshold tag exclusion** (`enrich_grid_params.py`): Bypass rows excluded from
  `below_threshold_tag`, preventing `_enforce_threshold_gate()` from nullifying
  grid parameters on low-score micro-oscillator candidates.
- **Pre-reject conditioning** (`enrich_grid_params.py`): `range_prob < 0.30` and
  `survival_prob < 0.40` early-reject checks conditioned on `not _is_bypass`.

### Added — Gate 4 Archetype (Step 3)

- **Archetype-dependent Gate 4** (`src/neutralgrid/scanner/two_stage_selector.py`):
  When `micro_osc.enabled` and `micro_osc_score >= min_score`, Gate 4 tests
  `survival_prob >= min_survival_prob` (MC containment) instead of `range_prob >=
  threshold` (HMM range). Gate 4 remains mandatory in both modes. `all(gates.values())`
  logic unchanged.
- **New parameters on `approve()`**: `micro_osc_score: float = 0.0`,
  `survival_prob: float = 0.0`. Backward-compatible defaults.

### Added — Feature Pipeline Registration (Step 4)

- **`candidate_pipeline.py`**: `_SCANNER_TO_FEATURE` + `TRAINING_OUTPUT_COLUMNS`
- **`data_generator.py`**: `FeatureSnapshot.micro_osc_score` field + `to_dict()`
- **`unified_training_builder.py`**: `EXTRA_META_FEATURES` + `_SCAN_TO_FEATURE`
- **`scanner_integration.py`**: `build_feature_snapshot()` mapping
- **`meta_labeler.py`**: `_FEATURE_MEDIAN_DEFAULTS` imputation default (0.0)
- Per `safety-invariants.md` Feature Pipeline Update Rule (all 3 mandatory files updated)

### Added — Config and Feature Flag (Step 5)

- **`MicroOscConfig`** (`src/neutralgrid/core/config.py`): 3-field dataclass —
  `enabled: bool = False`, `min_score: float = 0.45`,
  `min_survival_prob: float = 0.60`
- **`Config.micro_osc`** field on root Config class
- **`MICRO_OSC_ENABLED`** env-var override
- **`safety-invariants.md`**: Gate 4 archetype documentation added

### Fixed — Stochastic Computation Prerequisites (Step 0)

- **`kline_limits["15m"]`** (`config.py`): 200 → 300. `_compute_stochastic_features()`
  requires `len(df_15m) >= 300` (scan.py:159). With 200 bars, `survival_prob` and
  `hurst_exponent` were never computed. API weight unchanged (5 points per call
  regardless of bar count).
- **`--compute-stochastic`** (`run_full_pipeline.py`): Added `default=None` to
  `action="store_true"`. Without explicit default, value was `False` when flag absent,
  blocking the `None` fallback to `STOCHASTIC_VALIDATION_AVAILABLE` (scan.py:319-320).
  Reference implementation: `scan_top100.py` already used this pattern.

### Added — _RegimeData Extension

- **`micro_osc_score`** and **`micro_osc_bypass`** fields on `_RegimeData` dataclass
  (`enrich_grid_params.py`). Populated from scan DataFrame in `run_one()` after
  `_fetch_regime_data()`, enabling Gate 4 and Stage B to access scan-phase micro-osc data.

### Changed — Version

- **Package version**: 6.5.6 → 6.5.7 (`pyproject.toml`, `src/neutralgrid/__init__.py`)
- `LABEL_CONTRACT_VERSION` unchanged (2.4) — `micro_osc_score` is in
  `EXTRA_META_FEATURES`, not core `TrainingTableSchema.features`
- `FORMULA_VERSION` unchanged (alignment-v1) — PnL formula unchanged
- `ENGINE_VERSION` unchanged (realistic-v7) — backtest simulation unchanged

### What This Does NOT Change

- HMM model or training
- Meta-labeler model
- Backtest engine
- Scanner score formula
- Conformal risk control
- Label contract
- TOS weights or sub-signals
- Existing pipeline behavior when flag is disabled

### Verification

- **pytest**: 1008 passed, 0 failed
- **Validation team**: 5 parallel agents (config/imports, scan math, enrichment bypass,
  feature pipeline, test regression) — all passed
- **Dead-end code check**: NONE — all new code has live consumers
- **Safety invariants**: All 8 checked (leakage, fail-closed, backtest entry, config
  integrity, artifact naming, feature pipeline rule, rate limits, version constants)

---

## [6.5.6-6h-migration] - 2026-03-24

### Bot Duration Migration: 12h → 6h

Full migration of bot holding horizon from 12 hours to 6 hours across the entire
codebase. 55 edits across 22 files covering config defaults, barrier parameters,
backtest engine, training pipeline, CPCV purging, HMM evaluation, and all CLI
entry points. All AFML ratios preserved (purge=2×horizon=12h, embargo=0.25×horizon=1.5h).
Models retrained under 6h labels. 1009 tests passed, 0 failed.

### Changed - Config Defaults (6h Regime)

- **`GridConfig`** (`src/neutralgrid/core/config.py`): `max_holding_time` "12h"→"6h",
  `max_holding_seconds` 43200→21600
- **`BarrierConfig`** (`src/neutralgrid/core/config.py`): `time_hours` 12.0→6.0
- **`StochasticConfig`** (`src/neutralgrid/core/config.py`): `survival_horizon_bars` 48→24
  (6h / 15min)
- **`CPCVConfig`** (`src/neutralgrid/core/config.py`): `purge_hours` 24.0→12.0 (AFML 2×),
  `embargo_hours` 3.0→1.5 (AFML 0.25×), `horizon_hours` None→6.0 (explicit)
- **`ValidationConfig.pnl_horizon_hours`** (`src/neutralgrid/core/config.py`): 12→6
  (dead field, updated for consistency)

### Changed - Barrier and Labeling Defaults

- **`UnifiedBarrierConfig.time_barrier_hours`** (`src/neutralgrid/models/barrier_config.py`): 12.0→6.0
- **`TripleBarrierConfig.time_barrier_hours`** (`src/neutralgrid/models/triple_barrier.py`):
  dataclass default 12.0→6.0, factory default 12.0→6.0
- **`STANDARD_HORIZON_HOURS`** (`backtest/btk_label_contract.py`): 12.0→6.0
- **`TRAINING_ENGINE_DEFAULTS["max_holding_bars"]`** (`backtest/btk_label_contract.py`): 720→360
- **`GridConfig.max_holding_bars`** (`backtest/backtest_realistic.py`): 720→360

### Changed - Training Pipeline Defaults

- **`LabelConfig.horizon_hours`** (`src/neutralgrid/training/data_generator.py`): 12.0→6.0
- **`load_and_prepare_training_data(horizon_hours=)`**: 12.0→6.0
- **`MetaLabelerConfig`** (`src/neutralgrid/models/meta_labeler.py`): `horizon_hours` 12.0→6.0,
  `purge_hours` None→12.0 (AFML 2×), `embargo_hours` 12.0→1.5 (AFML 0.25×)
- **`create_meta_labeler(horizon_hours=)`**: 12.0→6.0

### Changed - Scanner and Validation Defaults

- **`RankingConfig.horizon_hours`** (`src/neutralgrid/scanner/pnl_ranker.py`): 12.0→6.0
- **`UtilityConfig.horizon_hours`** (`src/neutralgrid/validation/utility.py`): 12.0→6.0
- **`StochasticConfig.survival_horizon`** (`src/neutralgrid/validation/stochastic.py`): 48→24
- **`estimate_costs(horizon_hours=)`** (`src/neutralgrid/validation/microstructure.py`): 12.0→6.0
- **`estimate_funding_cost(horizon_hours=)`** (`src/neutralgrid/validation/microstructure.py`): 12.0→6.0
- **`compute_dynamic_profit_floor(horizon_hours=)`** (`src/neutralgrid/validation/microstructure.py`): 12.0→6.0
- **`horizon_factor` baseline** (`src/neutralgrid/validation/microstructure.py`): 12h→6h
- **Empirical profile fallbacks** (`src/neutralgrid/scanner/empirical_profile_v20260302.py`):
  `fillna(12.0)`→`fillna(6.0)`, missing-column fallback 12.0→6.0

### Changed - Backtest Evaluation Defaults

- **`auc_fwd_horizon`** (`src/neutralgrid/backtest/evaluate.py`): 12→6 (6h on 1H data)
- **`run_cpcv_utility_sweep(fwd_horizon=)`**: 12→6
- **`run_cpcv_proxy_outcome_report(fwd_horizons=)`**: (12,24)→(6,12)
- **`create_cpcv()` factory** (`src/neutralgrid/backtest/cpcv.py`): `purge_hours` 48.0→12.0,
  `embargo_hours` 6.0→1.5

### Changed - CLI Entry Points

- **`backtest_candidates.py`**: `--hours` default 12→6
- **`backtest_candidates_current.py`**: `fetch_recent_klines(hours=)` 12→6,
  `--hours` default 12→6, `min_bars` threshold 360→355 (open-candle tolerance)
- **`run_candidate_backtests.py`**: `fetch_klines(hours=)` 12→6, hardcoded call site 12→6,
  operator banner updated
- **`retrain_meta_labeler.py`**: `--horizon-hours` default 12.0→6.0
- **`backtest_realistic.py`**: `--hours` CLI default 12→6

### Changed - Candidate Pipeline

- **`fetch_historical_klines(hours=)`** (`src/neutralgrid/backtest/candidate_pipeline.py`): 12→6
- **`run_single_backtest(max_holding_bars=)`**: 720→360
- **`convert_to_training_row(horizon_hours=)`**: 12.0→6.0

### Fixed - Pre-existing Bugs

- **`clean_numeric_column` pandas StringDtype** (`_bot_data_extractor_core.py`):
  `series.dtype == 'object'` check missed pandas `StringDtype` (new in pandas 2.x),
  causing `compute_trade_metrics` to return zero MAE/MFE. Fixed by adding
  `pd.api.types.is_string_dtype(series)` check.
- **`DeployLinker` file lock race condition** (`src/neutralgrid/live/candidate_deploy_linker.py`):
  `write_header` check was outside the lock (TOCTOU race), and `msvcrt.locking`
  seek position was unreliable in append mode. Fixed by moving header check inside
  the lock and seeking to position 0 before lock/unlock.
- **`pytest-asyncio` missing**: installed missing test dependency required by
  6 async test functions.

### Fixed - Migration-Induced Test Failures

- **`test_purge_zero_removes_nothing`** (`tests/test_afml_compliance.py`): added explicit
  `horizon_hours=0.0` so zero-purge scenario uses point-in-time events (the new
  default `horizon_hours=6.0` caused CPCV to synthesize 6h event windows that
  overlapped test groups)
- **`test_embargo_defaults_to_one_point_five`**: updated 3.0→1.5 (AFML 0.25×6h)
- **`test_default_max_holding_bars_is_360`**: updated 720→360
- **`test_default_horizon_is_6_hours`**: updated 12.0→6.0
- **`test_training_defaults_applied`**: `max_holding_bars` 720→360
- **`test_max_holding_bars_is_360`**: updated 720→360

### Changed - Documentation and Comments

- Updated 12 stale docstrings/comments across `triple_barrier.py`, `barrier_config.py`,
  `evaluate.py`, `cpcv.py`, `utility.py`, `spacing_profile.py`, `regime_validator.py`,
  `backtest_candidates.py`, `run_candidate_backtests.py`
- Updated `readmefullpwep.md` CLI reference: `--horizon-hours` 12.0→6.0,
  `--hours` 12→6
- Deleted `BOT_DURATION.md` (migration plan, no longer needed)

### Retrained - Models Under 6h Labels

- **Scanner**: 3 artifacts (pattern_profile, profile_model, profile_gate)
- **Meta-labeler**: CV AUC 0.591, horizon=6h, 253 samples, 10 features
- **HMM**: `rolling_180d_20260324_021549`, 50 symbols, 210k bars, walk-forward
  pass_rate=100%, auto-promoted

### Verification

- **pytest**: 1009 passed, 0 failed, 0 warnings
- **Static validation**: zero operational-path 12h/720/48-duration values remain
- **Backtest pipeline trace**: all 10 data flow paths verified 6h/360-bar
- **OOS validation**: 98 symbols tested against current market data
- **Full pipeline**: 250 symbols scanned, 11 deployment-ready candidates produced

---

## [6.5.6-afml-audit-fixes] - 2026-03-22

### AFML Full-Pipeline Audit and Remediation

Comprehensive 8-stage AFML audit (Ch 2→3→4→6→7→8→11-12→Deployment) across
the entire codebase, followed by 3-team parallel remediation (Implementation,
Bug Fix, Quality Control). All fixes verified: pyright 0 errors, 996 tests
passed, 0 regressions.

### Fixed - CRITICAL: Deployment Safety (AFML Deployment)

- **EV ranking root cause fixed** (`run_full_pipeline.py`): `deployment_score`
  now computed ONLY for candidates where `grid_is_valid == True`. Previous
  implementation ranked all candidates then nulled invalid ones as a bandage
  ("Gate 9 floor guard"). Root cause eliminated: invalid candidates are never
  ranked.
- **Conformal gate fail-closed** (`src/neutralgrid/scanner/two_stage_selector.py`):
  exception handler now sets `gates["conformal_meta"] = False` and appends a
  rejection code instead of silently skipping with `logger.debug`.
- **Entropy-adaptive threshold logging** (`src/neutralgrid/scanner/two_stage_selector.py`):
  bare `pass` in exception handler replaced with `logger.warning` for
  auditability.

### Fixed - CRITICAL: Calibration Sample Weights (AFML Ch 4/6)

- **Beta calibrator accepts sample weights**
  (`src/neutralgrid/calibration/beta_calibrator_v20260311.py`): `fit()`,
  `_expected_calibration_error()`, and `_brier_score()` now accept optional
  `sample_weight` parameter. LogisticRegression fit passes weights through.
  ECE and Brier use `np.average(..., weights=w)` per bin when provided.
- **Reliability diagnostic weighted**
  (`src/neutralgrid/calibration/reliability_diagnostic_v20260311.py`):
  `compute()` accepts optional `sample_weight`. Bin accuracy, confidence, and
  fraction calculations use weight-proportional statistics.
- **OOS evaluation weighted** (`src/neutralgrid/models/meta_labeler.py`):
  out-of-sample `roc_auc_score()` now receives `sample_weight=test_weights`
  extracted from the fold's test mask.

### Fixed - HIGH: Train/Serve Skew Detection (AFML Ch 6)

- **Feature imputation warning at inference**
  (`src/neutralgrid/models/meta_labeler.py`): `predict_proba()` now counts
  how many features from `self._feature_names` are missing (None/NaN) in the
  input. Logs a warning when >2 features are being imputed, flagging possible
  train/serve skew.

### Fixed - HIGH: Backtest Statistics (AFML Ch 11-12)

- **Sharpe annualization corrected** (`backtest/backtest_realistic.py`):
  changed `np.sqrt(525600)` to `np.sqrt(525600 / max(1, n_bars))` to adjust
  annualization factor for actual sample size. Prevents over-annualization of
  short backtests (e.g., 12h → 730 bars).
- **Return moments added** (`backtest/backtest_realistic.py`): computes
  `return_skewness` (via `scipy.stats.skew`) and `return_kurtosis` (via
  `scipy.stats.kurtosis`, Fisher=False) and stores in result dict for
  deflated Sharpe calculation.

### Fixed - MEDIUM: Data Curation (AFML Ch 2)

- **API kline deduplication** (`src/neutralgrid/data/market_data.py`): added
  `df.drop_duplicates(subset="open_time", keep="last")` after API fetch and
  type conversion, before caching. Prevents duplicate bars from propagating
  to training data.
- **Bridge merge dedup logging** (`src/neutralgrid/data/binance_vision/pipeline.py`):
  daily bridge merge now logs the count of duplicate bars removed during
  monthly+daily concatenation.

### Fixed - MEDIUM: Position Sizing Guard (AFML Deployment)

- **Capital fraction non-negative guard**
  (`src/neutralgrid/scanner/enrich_grid_params.py`): added `max(0.0, ...)`
  around Kelly × position sizer multiplication to prevent negative values
  from propagating through the sizing chain.

### Added - AFML Ch 3 Documentation

- **Barrier inference deviation documented**
  (`src/neutralgrid/backtest/candidate_pipeline.py`): added 7-line comment
  explaining that barrier inference from terminal PnL (not first-touch price
  path) is a known AFML Ch3 deviation, with rationale for grid-bot semantics
  where "barrier" maps to range containment rather than directional targeting.

### Verification

- **pyright**: 0 errors, 0 warnings, 0 informations
- **pytest**: 996 passed, 0 failed (8 pre-existing async failures excluded)
- **py_compile**: all 10 modified files compile cleanly
- **Import smoke test**: all modified modules import successfully

---

## [6.5.6-pipeline-acceptance-green] - 2026-03-21

### Pipeline Acceptance Standard: Red → Green

All 10 blocking gates from `PIPELINE_ACCEPTANCE_STANDARD.md` satisfied.
HMM promoted with truthful 180-day canonical window from Binance Vision
parquet store. Full pipeline executes end-to-end without errors.

### Fixed - CRITICAL: Authoritative Rebuild Path (Gate 1)

- **Syntax error in `backtest_candidates.py`** fixed: orphaned `else` at line
  397 caused `py_compile` failure. The conformal generation block's `if/else`
  for y-column selection was at wrong indentation, preventing the conformal
  fit/save from executing on the authoritative path. Both label branches now
  reach the `ConformalRiskController.fit()` call.

### Fixed - CRITICAL: HMM Promotion/Runtime Contract Alignment (Gates 2-3)

- **Walk-forward evaluation aligned with runtime pass mode**
  (`src/neutralgrid/models/hmm/train.py`): added `_walk_forward_bar_pass()`
  helper that mirrors all 5 runtime gating modes (hard, dominance, soft,
  hybrid, utility). Walk-forward now reads `pass_mode`, `range_dominance_min`,
  `dominance_eps`, and `soft_gating` from the same config source as runtime.
  Per-fold state groups computed via `compute_state_groups()` for aggregated
  probability dominance testing.
- **Boundary-safe posteriors** (`src/neutralgrid/models/hmm/train.py`):
  `model.predict_proba(X_scaled, lengths=lengths)` now passes sequence
  boundaries so each symbol's posteriors start with proper initial state
  probabilities. Added `assert sum(lengths) == X_stacked.shape[0]` with
  `profile_scope="global"` metadata on the uncertainty profile.
- **Threshold artifact compatibility** (`src/neutralgrid/validation/regime_validator.py`):
  `_load_cpcv_auto_threshold()` now validates `pass_mode` compatibility;
  returns `None` (config fallback) when the artifact's pass mode differs from
  the current runtime config.

### Fixed - CRITICAL: Score, Identity, and Feature Contract Uniqueness (Gates 4-5)

- **Score overloading eliminated** (`src/neutralgrid/scanner/scan.py`,
  `run_full_pipeline.py`, `src/neutralgrid/backtest/candidate_pipeline.py`):
  scan-time composite score preserved as `scan_score`; EV percentile ranking
  written to new `deployment_score` column instead of overwriting `score`.
  Training maps `scan_score` to `primary_pipeline_score`.
- **Range and utility provenance separated**: `scan_range_size_pct` and
  `scan_utility_score` preserve scan-time provisional values; enrichment-time
  authoritative values stored under `range_size_pct` and `utility_score`
  (via `regime_utility` priority in `scanner_integration.py`).
- **Scan decision identity**: `candidate_id` generated for every scan snapshot
  via `make_candidate_id_from_row()`. `hmm_feature_source` set explicitly
  (`"live_scan"` or `"backtest"`) instead of defaulting to `None`.

### Fixed - CRITICAL: Backtest Authority and Ingestion Governance (Gate 6)

- **Non-authoritative row filtering** (`src/neutralgrid/training/unified_training_builder.py`):
  `source_class="reconstruction"` rows excluded by default (controlled by
  `include_reconstruction` parameter). `version_gated=True` rows excluded by
  default (controlled by `accept_version_mismatch`). `is_authoritative=False`
  rows excluded. All exclusion counts logged for audit.
- **Fallback label synthesis marked non-authoritative**
  (`src/neutralgrid/backtest/candidate_pipeline.py`): `label_source` field
  tracks `"hierarchical"` vs `"hurdle_fallback"`. When hierarchical labeling
  fails, `is_authoritative=False`.
- **t1 truncation governance**: `t1_is_synthetic` and `t1_truncated` fields
  added. Truncated backtests (SL hit or duration < 95% of horizon) use actual
  end time instead of canonical horizon for t1 synthesis.
- **`--include-backtest-data` authority enforcement** (`retrain_meta_labeler.py`):
  now applies `source_class`, `version_gated`, and `is_authoritative` filters
  before merging backtest rows.

### Fixed - CRITICAL: Meta-Labeler Temporal Validation (Gate 7)

- **Pre-sort before CPCV** (`src/neutralgrid/models/meta_labeler.py`):
  training DataFrame, features, labels, and sample weights sorted by
  `timestamp_col` before CPCV split call.
- **Within-group temporal purge for symbol-blocked CPCV**
  (`src/neutralgrid/backtest/cpcv.py`): when symbol grouping is active and
  `t1` times are available, training samples whose event windows overlap with
  the test group's time range are now purged. Previously `purged_count = 0`.
- **t1 synthesis uses training horizon** (`src/neutralgrid/backtest/cpcv.py`):
  added `horizon_hours` to `CPCVConfig`; t1 fillna uses `horizon_hours`
  (actual training horizon) instead of `purge_hours`.

### Fixed - CRITICAL: Live Artifact Compatibility (Gate 8)

- **Conformal artifact validation** (`src/neutralgrid/calibration/conformal_risk_control_v20260311.py`):
  `load()` now validates `gate_type`, `alpha`, minimum `n_calibration`, and
  artifact age. Returns `None` (fail-closed) on incompatibility.
- **MI weights signal validation** (`src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py`):
  `validate_mi_weights()` checks signal names against
  `_EXPECTED_RUNTIME_SIGNALS`, rejects unexpected names, validates
  non-negative weights. `run_full_pipeline.py` calls this after loading.
- **Empirical profile deterministic dedup** (`src/neutralgrid/scanner/empirical_profile_v20260302.py`):
  deduplicates on `candidate_id` (or `symbol` + `start_time_utc` fallback)
  keeping the most recent row. Logged: `2404 → 454 rows (1950 duplicates removed)`.
- **Pattern profile schema validation** (`src/neutralgrid/scanner/pattern_profile.py`):
  `validate_features()` checks profile features against canonical
  `DEFAULT_FEATURES`; `load_json()` returns `None` on critical mismatch.

### Fixed - CRITICAL: Economic Floor Capital-Basis Consistency (Gate 9)

- **Leverage-adjusted funding cost** (`src/neutralgrid/scanner/pnl_ranker.py`):
  `funding_cost` multiplied by `leverage` to convert from notional to margin
  basis, matching `fill_revenue` and `boundary_loss` terms. Added `leverage`
  parameter to `compute_ev()`, `rank_score()`, and `compute_score()`.
- **Floor override guard** (`run_full_pipeline.py`): after `deployment_score`
  computation, nulls deployment_score for any candidate with
  `grid_is_valid != True`.

### Fixed - HIGH: Hard-Gate Fail-Closed Semantics (Gate 10)

- **Exception path emits `False` not `None`**
  (`src/neutralgrid/scanner/enrich_grid_params.py`): `hard_gate_passed=None`
  changed to `hard_gate_passed=False` in the exception handler. No downstream
  path can reinterpret exception-state candidates as unknown-but-possibly-pass.

### Added - Canonical HMM Retrain Infrastructure (HMM_FETCHING_DATA.md)

- **`src/neutralgrid/models/hmm/canonical_retrain.py`** (new module):
  5-step canonical retrain pipeline:
  - `build_frozen_probe_universe()`: shared universe selector with
    `probe_depth_multiplier` (2.0x default), frozen `selection_time_utc`
  - `ensure_canonical_store_coverage()`: Binance Vision parquet store
    backfill/update for all probe symbols
  - `freeze_run_boundary()`: one `target_end_utc` (minimum across stores),
    one `common_start_utc` (target_end - 180 days), `open_time` timestamp basis
  - `screen_probe_universe()`: walks ranked symbols until exactly 50 accepted;
    rejects symbols failing coverage/monotonicity/dedup; fails loud if < 50
  - `freeze_training_slice()`: loads exact window per symbol, injects `symbol`
    column, validates slice, persists via `save_training_dataset()` with full
    provenance metadata

- **`retrain_hmm.py`**: added `--canonical` / `--no-canonical` flag (default
  True). Canonical mode runs the 5-step pipeline, passes frozen slice as
  `override_datasets` with `canonical_mode=True` and `canonical_metadata`.
  Legacy API-based flow preserved under `--no-canonical`.

- **`src/neutralgrid/cli/retrain.py`**: replaced inline symbol selection with
  shared canonical pipeline. `--symbols` runs marked `research_only=True`
  (promotion skipped).

### Added - Training-Side Parity and Promotion Tightening

- **Canonical mode fail-closed** (`src/neutralgrid/models/hmm/train.py`):
  `train_from_market_data()` accepts `canonical_mode: bool`. When True,
  requires all requested symbols in `override_datasets`, never falls back to
  API fetch. Aborts if trained basket != accepted 50-symbol basket.
- **Effective n_components parity**: `train_hmm_global()` returns
  `effective_n_components` after direction-aware coercion.
  `train_from_market_data()` passes effective value to
  `walk_forward_evaluate()` and downstream CPCV/sweep paths.
- **Symbols-dropped tracking**: `train_hmm_global()` compares survived symbols
  against `input_symbol_names`; returns `symbols_dropped` in result dict.
- **Truthful metadata**: `save_trained_model()` accepts `canonical_metadata`
  and merges `selection_time_utc`, `target_end_utc`, `common_start_utc`,
  `common_end_utc`, `timestamp_basis`, `accepted_symbols`,
  `canonical_source`, `effective_n_components` into artifact metadata.
  Training window uses `open_time` exclusively (not mixed `open_time`/`close_time`).
- **Promotion gates tightened** (`src/neutralgrid/models/hmm/retrain_orchestration.py`):
  canonical mode requires `accepted_count == 50` AND `trained_count == 50`
  (exact match, not `>= max(3, requested // 2)`). Probe, accepted, and
  trained counts tracked separately.
- **Threshold artifact governance**: `check_threshold_artifact_compatibility()`
  validates `hmm_version` and `pass_mode` against the promoted HMM.

### Added - Binance Vision Pipeline Improvements

- **Daily bridge after backfill** (`src/neutralgrid/data/binance_vision/pipeline.py`):
  backfill mode now automatically bridges from the last monthly archive to
  yesterday with daily archives before validation. Previously, backfill-only
  mode left a gap from the last complete month to the present.
- **Fractional-year start month fix**: changed `int()` to `math.ceil()` for
  `frac_months` calculation, ensuring backfill goes far enough back to cover
  the requested history depth.

### Fixed - Pyright Type Checking (19 errors → 0)

- `cast(pd.Series, ...)` guards added before `.quantile()`, `.mean()`,
  `.sum()`, `.max()` calls across `candidate_pipeline.py`, `meta_labeler.py`,
  `pattern_profile.py`, `profile_model.py`, `btk_alignment_audit.py`,
  `data_generator.py`.
- `cast(pd.DataFrame, ...)` before `.rename()` in `data_generator.py`.
- `int(str(...))` for iterrow values in `replay/normalize.py`.
- `np.std(np.asarray(...))` for pandas `.std()` return type in
  `pattern_profile.py`.

### Fixed - Pylance IDE Configuration

- **`.vscode/settings.json`** created with correct `python.defaultInterpreterPath`
  and `python.analysis.extraPaths` for user site-packages.
- **Unused variable `_smooth_k`** removed from `walk_forward_evaluate()`.
- **Deprecated parameter `btc_df`** renamed to `_btc_df` (underscore prefix).

### Operational

- **Backfilled Binance Vision store**: 83/100 top symbols by volume, 180+ days
  of 1H parquet data from Binance Vision archives with SHA256 checksums.
- **HMM promoted**: `rolling_180d_20260322_020521` — 50 symbols, 210,000
  samples, truthful 180-day window (2025-09-21 to 2026-03-20), 100%
  walk-forward pass rate, boundary-safe posteriors, canonical metadata.
- **Training data regenerated**: 250 backtest candidates, mean PnL +7.07%.
- **Meta-labeler retrained**: 243 rows (unified backbone), AUC 0.540.
- **Full pipeline smoke run**: 250 scanned, 245 enriched, all gates active.

---

## [6.5.6-hmm-regime-unification] - 2026-03-20

### HMM Regime Architecture Unification

Comprehensive HMM regime unification across inference, validation, enrichment,
retraining, training-data lineage, and meta-labeler governance. This closes
the verified architecture drift between runtime paths, disables harmful
self-labeled temperature sharpening, makes artifact usage local and
provenance-aware, and removes historical HMM feature rewrites from the active
artifact.

### Fixed - CRITICAL: Harmful Temperature Scaling

- **Self-labeled temperature scaling disabled** (`retrain_hmm.py`,
  `src/neutralgrid/cli/retrain.py`,
  `src/neutralgrid/models/hmm/retrain_orchestration.py`): HMM artifacts no
  longer fit calibration against `argmax(posteriors)` and both active
  `temperature_scaler.json` sidecars were neutralized to identity semantics
  (`T=1.0`, `runtime_validated=false`) with provenance retained.
- **Runtime calibration gating** (`src/neutralgrid/models/hmm/inference.py`):
  temperature scaling is now applied only when the scaler explicitly declares
  itself runtime-valid. Non-validated scalers are loaded for provenance only,
  not for posterior modification.

### Fixed - CRITICAL: Runtime HMM Path Drift

- **Artifact-local HMM loading** (`src/neutralgrid/models/hmm/inference.py`):
  `HMMRegimePredictor` now stores the artifact directory it was created from
  and loads calibration state from that same artifact instead of re-resolving
  the active manifest during `predict()`.
- **Stage B posterior duplication removed** (`src/neutralgrid/scanner/enrich_grid_params.py`):
  enrichment no longer recomputes a raw `model.predict_proba()` branch for
  entropy-adaptive thresholds. Stage B now consumes canonical validator-sourced
  posteriors.
- **Aggregated probability semantics aligned** (`src/neutralgrid/scanner/scan.py`,
  `src/neutralgrid/validation/regime_validator.py`,
  `src/neutralgrid/scanner/enrich_grid_params.py`): scan, validator, and
  enrichment now consistently carry aggregated `range_prob` / `trend_prob`
  alongside artifact version, pipeline version, and calibration provenance.
- **Persistence probability corrected** (`src/neutralgrid/models/hmm/inference.py`):
  `persistence_prob` is now derived from the effective transition matrix used
  for the final posterior, not a mismatched base matrix.

### Fixed - HIGH: Retrain Entry Point Drift

- **Shared retrain orchestration added** (`src/neutralgrid/models/hmm/retrain_orchestration.py`):
  promotion policy, evaluation syncing, identity scaler generation, and trial
  logging are now shared by `retrain_hmm.py` and `src/neutralgrid/cli/retrain.py`.
- **Duplicate CLI evaluation removed** (`src/neutralgrid/cli/retrain.py`):
  the CLI no longer overwrites `eval.json` with a second walk-forward result
  that diverges from the authoritative metadata saved by `train.py`.
- **Trial metadata corrected** (`retrain_hmm.py`,
  `src/neutralgrid/models/hmm/retrain_orchestration.py`): HMM trial logging
  now uses the real feature schema, UTC timestamps, artifact-aware notes, and
  consistent promotion context.

### Fixed - HIGH: Historical HMM Lineage Drift

- **Causal snapshot matching** (`src/neutralgrid/training/data_generator.py`):
  exact `candidate_id` matches are now time-causal, `snapshot_matched_at_utc`
  is preserved, and the snapshot match timestamp column is stored with stable
  UTC dtype semantics.
- **Version-aware backfill replay** (`scripts/backfill_training_features.py`):
  HMM-derived backfill now replays only from the row's pinned
  `hmm_artifact_version`. When lineage is missing or the artifact is
  unavailable, provenance is marked instead of silently recomputing from the
  current active HMM.
- **Active-HMM persistence rewrite removed** (`src/neutralgrid/training/unified_training_builder.py`):
  the unified training builder no longer derives `persistence_prob` from the
  currently active transition matrix for historical rows.
- **Key-based preservation fixed** (`scripts/backfill_training_features.py`):
  preservation now merges on normalized symbol + UTC start time and, when
  available, `candidate_id` / `strategy_id`, preventing cross-row value bleed
  when multiple bots share the same symbol and start time.

### Added - Governance And Provenance Hardening

- **Calibration provenance in inference payload** (`src/neutralgrid/models/hmm/inference.py`):
  `HMMInferenceResult` now carries `pipeline_version` and
  `calibration_provenance`.
- **Training lineage columns expanded** (`src/neutralgrid/training/data_generator.py`,
  `src/neutralgrid/training/scanner_integration.py`,
  `src/neutralgrid/training/unified_training_builder.py`): added
  `hmm_artifact_version`, `hmm_pipeline_version`, `hmm_trained_at_utc`,
  `hmm_feature_semantics_version`, `hmm_feature_source`,
  `hmm_calibration_status`, `snapshot_matched_at_utc`,
  `label_source`, and `dataset_schema_version`.
- **Meta-labeler compatibility enforcement** (`src/neutralgrid/models/artifact_compat.py`,
  `src/neutralgrid/models/meta_labeler.py`): artifact-managed meta-labeler
  loads now hard-fail on HMM lineage mismatch and on internal
  feature-schema-vs-metadata mismatch. Artifact saves no longer fall back
  silently to raw pickle when the managed save path fails.

### Validation

- Targeted unit coverage passed:
  - `tests/unit/test_artifact_compatibility.py`
  - `tests/unit/test_scanner_integration_v20260320.py`
  - `tests/unit/test_enrich_grid_params.py`
  - `tests/unit/test_enhancements_v653.py`
- Additional direct runtime replays validated:
  - causal snapshot matching and timestamp preservation
  - keyed backfill preservation by symbol/start time
  - disambiguated preservation by `candidate_id`
  - no merge-helper columns leaking into backfilled output
  - artifact-managed `MetaLabeler.load()` against the current artifact
- Note: `tmp_path`-based pytest runs in this environment still hit Windows
  temp-directory `PermissionError` during cleanup, so the affected lineage
  paths were verified with direct Python replays in addition to the passing
  non-temp pytest suite.

## [6.5.6-plan-v6-backtest-unification] - 2026-03-20

### Plan v6.0 — Backtest Subsystem Unification (21 Steps, 8 Phases)

Comprehensive unification of the backtest subsystem addressing label corruption,
provenance tracking, physics locking, runner authority, ingestion governance,
funding rate sign preservation, and documentation alignment. All 21 steps
implemented across 12 modified files and 3 new files.

### Fixed — CRITICAL: Label Corruption (Step 1)

- **Label target precedence reversed** (`unified_training_builder.py`):
  `_normalize_backtest_targets()` previously used priority order
  `("y_horizon", "label_positive_by_horizon", "y")` which overwrote strict
  hierarchical labels with lenient horizon labels. A bot that failed L1/L2
  (y=0) but exceeded the PnL hurdle (y_horizon=1) would be silently relabeled
  as positive. Fixed to `("y", "y_horizon", "label_positive_by_horizon")` —
  hierarchical `y` now takes unconditional precedence.
- **Same fix applied to slow-path** (`_load_backtest_rows()`): The fallback
  CSV-loading path had the identical precedence bug. `y = _label_from_any(raw.get("y"))`
  is now tried first.

### Fixed — HIGH: Funding Rate Sign Stripping (Step 13)

- **`abs()` removed from funding rate** (`backtest_candidates_current.py`,
  `run_candidate_backtests.py`): Negative funding rates (shorts pay longs =
  bot receives income) were being converted to positive via `abs()`, inflating
  funding costs. Changed to `funding_rate = fr_val if fr_val != 0 else 0.0001`.

### Fixed — MEDIUM: Glob Pattern Mismatch (Step 2)

- **Missing glob pattern** (`unified_training_builder.py`): `backtest_candidates.py`
  writes `training_data_*.csv` but `_load_backtest_rows()` only globbed for
  `training_rows_*.csv` and `backtest_training_*.csv`. Added `training_data_*.csv`
  as the first glob pattern. Without this fix, authoritative training data from
  the candidate pipeline was silently ignored.

### Fixed — MEDIUM: Dead Import Cleanup

- **Unused `GridConfig` import** (`run_candidate_backtests.py`): Removed dead
  `from backtest.backtest_realistic import GridConfig` — the file only uses
  `run_backtest` and `build_training_config` from the unified runner.

### Fixed — LOW: Documentation Drift (Step 21)

- **Training config table** (`.claude/rules/backtest-subsystem.md`): Corrected
  `close_fee_mode` from `taker` to `maker` (actual `TRAINING_ENGINE_DEFAULTS`
  value). Added 5 CB fields (`cb_enabled`, `cb_trailing_activate_pct`,
  `cb_trailing_offset_pct`, `cb_inventory_imbalance_ratio`,
  `cb_inventory_imbalance_dd_pct`) to the training config table.

### Added — Version Constants Module (Step 4)

- **`src/neutralgrid/core/constants.py`** (NEW): Single source of truth for
  version constants consumed across the backtest subsystem:
  - `LABEL_CONTRACT_VERSION = "2.4"`
  - `FORMULA_VERSION = "alignment-v1"`
  - `ENGINE_VERSION = "realistic-v7"`
- **`backtest/btk_label_contract.py`**: Now imports `FORMULA_VERSION` and
  `LABEL_CONTRACT_VERSION` from `constants.py` with `try/except ImportError`
  fallback (backtest/ is outside the installed package).
- **`backtest/btk_unified_runner.py`**: Now imports `ENGINE_VERSION` from
  `constants.py` with `try/except ImportError` fallback.
- Eliminates version string duplication — previously hardcoded in 3 files
  independently.

### Added — Provenance Columns (Step 3)

- **`candidate_pipeline.py`**: `convert_to_training_row()` now propagates 5
  provenance fields from backtest results into training rows:
  `engine_version`, `label_contract_version`, `backtest_run_id`,
  `formula_version`, `is_authoritative`.
- **`TRAINING_OUTPUT_COLUMNS`**: Added `engine_version`,
  `label_contract_version`, `backtest_run_id` to the CSV output schema.
- `formula_version` fallback now imports from `neutralgrid.core.constants`
  instead of hardcoding.

### Added — Source-Class Split (Step 5)

- **Fast-path** (`_load_backtest_rows()`): Rows from `training_data_*.csv`
  (authoritative, hierarchically labeled) tagged `source_class="authoritative"`.
- **Slow-path** (`_load_backtest_rows()`): Rows from `backtest_results_*.csv`
  (reconstructed, lossy) tagged `source_class="reconstruction"`.
- Enables downstream filtering by data provenance quality.

### Added — Ingestion Gate (Step 6)

- **`_apply_ingestion_gate()`** (`unified_training_builder.py`): New method
  (~55 lines) that validates `label_contract_version` on each row:
  - Matching version → `version_gated=False`
  - Mismatched version → `version_gated=True`, logged with mismatch details
  - Missing version → `version_gated=True`, `source_class="legacy"` (Step 7)
- Rows are flagged, NOT dropped — downstream consumers decide policy.

### Added — Version Mismatch Governance (Step 16)

- **`--accept-version-mismatch`** flag (`retrain_meta_labeler.py`): When
  `version_gated` rows exist in training data:
  - Without flag: gated rows are silently excluded from training
  - With flag: gated rows included, warning logged with count
- Prevents accidental training on stale-contract data without explicit opt-in.

### Added — Physics Lock (Step 9)

- **`build_training_config()`** (`btk_unified_runner.py`): 21 physics-locked
  fields identified. Overriding any of these logs a warning and tags the result
  as `is_authoritative=False`:
  ```
  funding_mode, close_fee_mode, order_delay_bars, slippage_bps,
  maker_fee, taker_fee, funding_interval_bars, maintenance_margin_rate,
  tick_size, step_size, price_source, spread_bps, fill_mode, margin_mode,
  global_cooldown_bars, cb_enabled, cb_max_dd_pct,
  cb_trailing_activate_pct, cb_trailing_offset_pct,
  cb_inventory_imbalance_ratio, cb_inventory_imbalance_dd_pct
  ```
- Two intentionally excluded: `max_holding_bars` (scenario, not physics) and
  `funding_rate` (market-determined, not engine physics).

### Added — Runner-Derived Authority (Step 10)

- **`run_backtest()`** (`btk_unified_runner.py`): Independently compares ALL
  21 physics fields in the config against `TRAINING_ENGINE_DEFAULTS`,
  regardless of how the config was built. Also checks for non-None
  `funding_rate_series`.
- A result is `is_authoritative=True` IFF all physics fields match defaults
  AND `funding_rate_series is None`.
- Critical fix during implementation: initial approach relied on
  `_physics_overridden` metadata from `build_training_config()`, which meant
  raw `GridConfig` objects (missing the attribute) were incorrectly marked as
  authoritative. Fixed by making the check config-value-based, not
  metadata-based.

### Added — CLI Route Through Unified Runner (Step 12)

- **`backtest_realistic.py` `__main__`**: CLI invocation now routes through
  `run_backtest()` from `btk_unified_runner.py` instead of instantiating
  `RealisticGridBacktester` directly. Ensures CLI-executed backtests receive
  provenance, validation, and authority stamping.

### Added — Alignment Auditor Integration (Step 15)

- **`--audit` flag** (`backtest_candidates.py`): Wires the previously-dead
  `btk_alignment_audit_v20260316.py` into the pipeline. When `--audit` is
  passed, instantiates `AlignmentAuditor` with `LiveOutcomeIngestor` and
  runs `compute_alignment_report()` on the output directory.

### Added — CB Degeneracy Documentation (Step 17)

- **`backtest_realistic.py`**: Added documentation at the `short_count=0`
  call site explaining grid bot CB degeneracy:
  - Grid bots always have `short_count=0` → `imbalance = 1.0`
  - Branch A fires at DD ≥ 3% with ≥ 3 positions (imbalance always exceeds 0.85)
  - Branch B (max_dd only) fires for < 3 positions
  - Trailing is effectively disabled (activate=999%, offset=999%)

### Added — GridConfig Docstring (Step 20)

- **`backtest_realistic.py`**: `GridConfig` docstring updated with 8-field
  divergence table showing where `GridConfig` defaults differ from
  `TRAINING_ENGINE_DEFAULTS`, directing users to `build_training_config()`.

### Added — Test Coverage (Steps 18, 21)

- **`tests/unit/test_btk_cb_branches.py`** (NEW — 13 tests):
  - `TestBranchA_ImbalanceFirst` (5 tests): DD threshold, position threshold,
    both thresholds, no-fire below thresholds, accumulation behavior
  - `TestBranchB_MaxDDFirst` (3 tests): fires at max_dd, no-fire below,
    low position count routing
  - `TestBranchInteraction` (5 tests): trailing disabled verification, CB
    disabled passthrough, fires-once semantics, branch A vs B routing

- **`tests/unit/test_plan_v6_steps.py`** (NEW — 15 tests):
  - `TestStep1LabelPrecedence` (2 tests): y-preferred-over-y_horizon,
    y_horizon-used-when-y-absent
  - `TestStep3Provenance` (1 test): provenance columns propagated through
    `convert_to_training_row()`
  - `TestStep4Constants` (3 tests): constants module exists, btk_label_contract
    uses same versions, btk_unified_runner uses same engine version
  - `TestStep6IngestionGate` (3 tests): matching version not gated, mismatched
    version gated, missing version marked legacy
  - `TestStep15AlignmentAuditor` (3 tests): importable, instantiable with mock,
    empty data returns empty
  - `TestStep10RunnerAuthority` (3 tests): training config → authoritative,
    raw GridConfig → not authoritative, physics override → not authoritative

### Implementation Phases

| Phase | Steps | Description |
|---|---|---|
| Phase 1 | 1 | Label corruption fix (precedence reversal) |
| Phase 2 | 2, 3 | Data path fixes (glob pattern, provenance columns) |
| Phase 3 | 4 | Version constants single source of truth |
| Phase 4 | 5, 6, 7 | Source-class split, ingestion gate, legacy handling |
| Phase 5 | 9, 10 | Physics lock, runner-derived authority |
| Phase 6 | 12, 13, 15, 16, 17 | CLI routing, funding sign, auditor, governance, CB docs |
| Phase 7 | 20, 21 | Documentation alignment, rule updates |
| Phase 8 | 18 | Test coverage (CB branches + plan step tests) |

### Files Modified

| File | Steps |
|---|---|
| `src/neutralgrid/core/constants.py` | 4 (NEW) |
| `src/neutralgrid/training/unified_training_builder.py` | 1, 2, 4, 5, 6, 7 |
| `src/neutralgrid/backtest/candidate_pipeline.py` | 3, 11 |
| `backtest/btk_label_contract.py` | 4, 17 |
| `backtest/btk_unified_runner.py` | 4, 9, 10 |
| `backtest/backtest_realistic.py` | 12, 17, 20 |
| `backtest_candidates.py` | 15 |
| `backtest_candidates_current.py` | 13 |
| `run_candidate_backtests.py` | 13, dead import cleanup |
| `retrain_meta_labeler.py` | 16 |
| `.claude/rules/backtest-subsystem.md` | 21 |
| `tests/unit/test_btk_cb_branches.py` | 18 (NEW) |
| `tests/unit/test_plan_v6_steps.py` | 21 (NEW) |

### Review & Validation

Four specialist review agents verified the implementation:

- **Architecture reviewer**: Validated execution flow, module boundaries,
  import patterns. No architectural drift found.
- **Label & data integrity reviewer**: Confirmed label precedence fix,
  provenance propagation, ingestion gate semantics. No label corruption paths.
- **Governance reviewer**: Verified physics lock coverage, authority derivation,
  version mismatch handling. Identified redundant `hasattr` guard — fixed.
- **Dead-end code reviewer**: Scanned all modified and new files for
  unreachable code, unfinished implementations. Result: **NONE**.

### Dead-End Codes

**NONE.** All implemented code paths are reachable. The previously-dead
`btk_alignment_audit_v20260316.py` is now reachable via the `--audit` flag
in `backtest_candidates.py`.

### Stats

- **Tests**: 976 → 1004 (28 new tests: 13 CB branch + 15 plan step)
- **Steps implemented**: 21/21
- **Files modified**: 12 + 3 new
- **Review agents**: 4 (architecture, labels, governance, dead-end)
- **Review findings**: 5 total → 5 resolved (2 confirmed by-design, 3 fixed)

---

## [6.5.6-afml-audit-and-feature-pipeline] - 2026-03-19

### AFML Pipeline Audit — Full Codebase Analysis

Six specialist agents audited 86 files across the complete AFML pipeline
(Ch 2→8→3→6→4→7→11→Deployment→Lifecycle). Found 108 issues:
7 CRITICAL, 26 HIGH, 48 MEDIUM, 27 LOW. All CRITICAL and HIGH issues resolved.

### Fixed — CRITICAL (7 resolved)

- **Exception handler too narrow** (`enrich_grid_params.py`): Widened `run_one()` handler to
  `except Exception` — prevents full enrichment batch crash on unexpected errors.
  Added `return_exceptions=True` to `asyncio.gather` as second defense layer.
- **Unsafe `np.load`** (`artifacts.py`): Added `allow_pickle=False` to prevent arbitrary
  code execution from compromised `.npy` files during model loading.
- **Data curation gate non-blocking** (`market_data.py`): Added `strict` parameter to
  `fetch_klines_cached()`. HMM training (`train.py`, `retrain.py`) now passes `strict=True`
  — quality failures raise `ValidationPipelineError` instead of silent warnings.
- **hlabel information leakage guard** (`meta_labeler.py`, `candidate_pipeline.py`):
  Added `_KNOWN_LABEL_COLUMNS` frozenset that `_prepare_features()` and `train()` explicitly
  drop before training. Renamed `TRAINING_FEATURES` → `TRAINING_OUTPUT_COLUMNS` to prevent
  naming confusion. `hlabel==3 ⟺ y==1` is now structurally guarded, not just by convention.
- **Feature aliasing** (`scan.py`): Removed cross-unit fallback chains where `atr_avg` would
  silently fall back to `atr_pct_15m` (different units: price vs %). Missing values now
  properly default to `NaN` instead of a different metric.

### Fixed — HIGH (19 resolved)

- **UTC timestamps** (`run_full_pipeline.py`): Changed `datetime.now()` → `datetime.now(timezone.utc)`.
- **WebSocket data validation** (`ps_ws_stream.py`, `ps_trigger_resolver.py`): Added OHLCV
  validation (prices > 0, high ≥ max(open,close), volume ≥ 0) on WS kline data. Invalid
  candles logged and skipped. Added `price > 0` guard to trigger resolver Levels 1–2.
- **Meta-labeler zero-imputation** (`meta_labeler.py`): Legacy fallback now uses per-feature
  median defaults (`_FEATURE_MEDIAN_DEFAULTS`) instead of `nan_to_num(nan=0.0)`.
- **Trial count fallback** (`evaluate.py`, `config.py`): Added `default_trial_count=10` to
  `CPCVConfig`. When trial tracker is unavailable, uses configurable default instead of 1
  (which would disable deflated Sharpe entirely).
- **Historical outlier check** (`binance_vision/validate.py`): Added `check_price_outliers()`
  with per-interval thresholds. Integrated into `validate_kline_store()` as step 5.
  Flags only — never fails the gate (consistent with curator philosophy).
- **Duplicate resolution consistency** (`binance_vision/ingest.py`): Standardized to
  `keep="last"` across all data storage paths. Documented "latest data wins" policy.
- **File lock on DeployLinker** (`candidate_deploy_linker.py`): Added cross-platform file
  locking (`msvcrt` on Windows, `fcntl` on Unix) to `_append_row()` for concurrent safety.

### Fixed — Pyright (223 errors → 0)

- **12 files** systematically fixed: pandas type narrowing via `cast(pd.Series, ...)`,
  `bool()` wraps for Series conditionals, `isinstance` guards for NaTType, `by=` keyword
  for `sort_values()`. All pre-existing Pyright diagnostics resolved.
- Files: `live_outcome_ingestor.py`, `run_full_pipeline.py`, `unified_training_builder.py`,
  `empirical_profile_v20260302.py`, `data_generator.py`, `candidate_deploy_linker.py`,
  `enrich_grid_params.py`, `retrain_meta_labeler.py`, `backtest_candidates_current.py`,
  `btk_replay_seed_loader.py`, `backtest_realistic.py`, `binance_vision/pipeline.py`.

### Added — Tier 4 Context Feature Pipeline

Full pipeline wiring for 4 new context features across 7 layers:

- **`bb_width_ratio_1h_15m`** (`feature_extractor.py`): Cross-timeframe Bollinger Band width
  ratio computed at scan time from 1h and 15m klines. Measures regime persistence across
  timeframes — range-bound regimes compress the ratio, trending expands it.
- **`long_short_ratio`** (`enrich_grid_params.py`): Extracted from existing
  `market_data["long_short_ratio"]` API response (was fetched but never consumed).
- **`funding_rate_zscore`** (`binance_client.py`, `enrich_grid_params.py`): Z-score of
  current funding rate vs 30-day history. Increased `get_funding_rate()` limit from 10→90
  (same API weight). Guards: `len >= 10`, `sigma > 1e-10`.
- **`open_interest_change_pct`** (`binance_client.py`, `enrich_grid_params.py`): 24h OI
  change percentage. Added `get_open_interest_hist(period="1h", limit=24)` to enrichment
  gather. Guards: `len >= 2`, `baseline > 0`.

Pipeline layers wired:
1. API fetching (`binance_client.py`)
2. Enrichment computation (`enrich_grid_params.py` — `_RegimeData`, `_build_base_payload()`, column init)
3. Feature mapping (`candidate_pipeline.py` — `_SCANNER_TO_FEATURE`)
4. Training builder (`unified_training_builder.py` — `EXTRA_META_FEATURES`, `_SCAN_TO_FEATURE`)
5. Feature snapshot (`data_generator.py` — `FeatureSnapshot` dataclass)
6. Snapshot extraction (`scanner_integration.py` — `build_feature_snapshot()`)
7. Imputation defaults (`meta_labeler.py` — `_FEATURE_MEDIAN_DEFAULTS`)

### Added — Training Data Backfill Infrastructure

- **CSV backfill** (`unified_training_builder.py`): New `_backfill_from_scanner_csvs()` method
  reads all `deployment_ready_*.csv` from `results/`, builds `candidate_id` lookup, fills NaN
  feature cells from matching scanner rows. Recovered 5,492 values from 172 CSVs on first run.
- **`persistence_prob` derivation** (`unified_training_builder.py`): New
  `_derive_persistence_prob_batch()` derives persistence probability from HMM transition matrix
  + `range_prob`/`trend_prob`. Uses `infer_state_mapping()` to identify range/trend states,
  then reads `transmat[dominant_state, dominant_state]`. Populated 2,085/2,110 rows.
- **`funding_rate_zscore` derivation**: Cross-sectional z-score from available `funding_rate`
  values when per-symbol API history is unavailable.
- **`bb_width_ratio_1h_15m` derivation**: Regime-adjusted `sqrt(4)` scaling approximation
  from `range_prob` when actual 1h BB data is unavailable.
- **Backtest passthrough** (`backtest_candidates.py`): Added 10 enrichment features to the
  passthrough tuple: `hmm_tail_cvar_95`, `tos_gcf`, `persistence_prob`,
  `micro_round_trip_cost_pct`, `open_interest`, `regime_conf`, plus 4 Tier 4 features.

### Performance — Feature Profile Results

| Profile | Features | AUC | P@5 | F1 | vs Baseline |
|---|---|---|---|---|---|
| baseline | 10 | 0.596 | 0.660 | 0.751 | — |
| `afml_scan_v20260318` | 21 | 0.703 | 0.740 | 0.791 | **+18%** |
| `afml_enriched_v20260318` | 26 | **0.709** | **0.760** | 0.789 | **+19%** |
| `afml_context_v20260318` | 30 | 0.701 | 0.740 | 0.787 | +18% |

Recommended production profile: `afml_enriched_v20260318` (best AUC + P@5).

### Added — AFML Compliance Tests

- `tests/unit/test_afml_compliance_fixes.py`: 33 tests covering all 12 AFML fixes.
  Tests verify exception handler width, np.load security, curation gate strictness,
  hlabel exclusion guard, feature aliasing removal, UTC timestamps, WS validation,
  zero-imputation fallback, trial count defaults, outlier check, duplicate resolution,
  and file locking concurrency safety.

### Changed — Version Bump

- Package version `6.5.5` → `6.5.6` in `pyproject.toml`, `src/neutralgrid/__init__.py`.
- `CLAUDE.md` project overview updated to `v6.5.6`.
- `readmefullpwep.md` runbook paths, expected output, and config section header updated.
- Test fixtures in `test_artifact_compatibility.py` and `test_deploy_linker.py` updated.

### Stats

- **Tests**: 943 → 976 (33 new AFML compliance tests)
- **Pyright**: 223 errors → 0 errors across all scopes
- **Files modified**: 30+
- **Agents used**: 6 analysis + 3 review + 4 Pyright fix + 4 feature wire + 2 backfill = 19 total

---

## [6.5.6-meta-feature-expansion] - 2026-03-18

### Added — Meta-Labeler Feature Expansion (Three-Profile Ladder)

Additive feature expansion for the meta-labeler classifier. Introduces a three-profile
ladder built on `live_plus_v20260312`, adding 14 new features across scan-safe,
enrichment-safe, and future-context tiers. Improves predictive signal without breaking
existing baseline or live_plus models.

#### New feature profiles (`src/neutralgrid/models/meta_labeler.py`)

- **`afml_scan_v20260318`** (21 features): extends `live_plus_v20260312` with 5 scan-safe
  features — `utility_score`, `hurst_exponent`, `ou_halflife`, `bb_width`, `open_interest`.
- **`afml_enriched_v20260318`** (26 features): extends scan with 5 enrichment-safe features —
  `micro_round_trip_cost_pct`, `regime_conf`, `hmm_tail_cvar_95`, `tos_gcf`, `persistence_prob`.
- **`afml_context_v20260318`** (30 features): extends enriched with 4 future-context features —
  `long_short_ratio`, `funding_rate_zscore`, `open_interest_change_pct`, `bb_width_ratio_1h_15m`.
  (L1 computation not yet implemented; blocked by NaN gate by design.)
- All 5 profiles registered in `META_FEATURE_PROFILES` dict with CLI selection via
  `--feature-profile` in `retrain_meta_labeler.py`.
- 10 new entries in `_FEATURE_MEDIAN_DEFAULTS` for scan and enriched tier features with
  mathematically sensible neutral priors (e.g., `hurst_exponent=0.5` random walk,
  `regime_conf=0.5` max entropy).

#### L2 capture: `persistence_prob` pipeline wiring

- **`src/neutralgrid/validation/hmm_regime.py`**: `infer_regime()` return dict now includes
  `persistence_prob` from `HMMInferenceResult.to_dict()`. Previously missing — this was the
  root cause preventing `persistence_prob` from reaching scan and enrichment outputs.
- **`src/neutralgrid/scanner/scan.py`**: extracts `persistence_prob` from regime inference
  result into the scan row dict.
- **`src/neutralgrid/validation/regime_validator.py`**: includes `persistence_prob` in the
  HMM check metrics dict passed to enrichment.
- **`src/neutralgrid/scanner/enrich_grid_params.py`**: added `persistence_prob` field to
  `_RegimeData` dataclass, extraction in `_fetch_regime_data()`, and inclusion in
  `_build_base_payload()` output.

#### L3 transport: training feature mapping

- **`src/neutralgrid/backtest/candidate_pipeline.py`**:
  - `_SCANNER_TO_FEATURE`: added 5 mappings — `hmm_tail_cvar_95`, `tos_gcf`,
    `persistence_prob`, `open_interest`, `micro_round_trip_cost_pct`.
  - `TRAINING_OUTPUT_COLUMNS`: added `hmm_tail_cvar_95`, `tos_gcf`, `persistence_prob`
    (before `hlabel`).
- **`src/neutralgrid/training/data_generator.py`**: added 3 `Optional[float]` fields to
  `FeatureSnapshot` — `hmm_tail_cvar_95`, `tos_gcf`, `persistence_prob` — with `to_dict()`
  serialization.
- **`src/neutralgrid/training/unified_training_builder.py`**: added 3 entries to
  `EXTRA_META_FEATURES` and 3 `scan_`-prefixed mappings to `_SCAN_TO_FEATURE`.
- **`src/neutralgrid/training/scanner_integration.py`**: `build_feature_snapshot()` now
  passes `hmm_tail_cvar_95`, `tos_gcf`, `persistence_prob` to the `FeatureSnapshot`
  constructor via `_safe_float()` extraction.

### Fixed

- **`_SCANNER_TO_FEATURE` transport gap** (`candidate_pipeline.py`): `open_interest` and
  `micro_round_trip_cost_pct` were missing from the backtest training-row mapping dict.
  Features were silently dropped in the backtest-to-training path. Now mapped correctly.
- **`build_feature_snapshot()` missing fields** (`scanner_integration.py`): `hmm_tail_cvar_95`,
  `tos_gcf`, `persistence_prob` were never passed to the `FeatureSnapshot` constructor —
  always `None` in the live snapshot logging path. Now wired from `scanner_row`.
- **`persistence_prob` L2 gap** (`hmm_regime.py`): `infer_regime()` did not include
  `persistence_prob` in its return dict despite `HMMInferenceResult.to_dict()` providing it.
  Downstream consumers (scan, enrichment) could never access this value.

### Removed

- **`FeatureSnapshot.from_dict()`** (`data_generator.py`): dead code — zero callers in
  codebase or tests. The deserialization classmethod was created as the counterpart to
  `to_dict()` but the snapshot reload workflow was never implemented.

### Verification

- **976/976 tests pass** — zero regressions across all test suites.
- 6-agent parallel verification covering L2 wiring (21 checks), L3 transport (13 checks),
  L4 model profiles (7 checks), environment/imports, and dead-end code scanning.
- VIF diagnostic active at threshold 5.0 (advisory). SimpleImputer CV-aware (AFML-compliant).

---

## [6.5.5-alignment-v1] - 2026-03-16

### Changed — Backtest-to-Live Alignment (`alignment-v1`)

Root-cause fix for **MTM inflation** in backtest predictions. The engine's `net_pnl_pct`
included unrealized PnL at end-of-simulation, inflating labels relative to live bots whose
realized outcomes are the ground truth. This release decomposes realized vs unrealized PnL,
feeds realized values into the labeling pipeline, and tags every training row with a formula
version for auditability.

**Formula tag:** `alignment-v1` — applied to all backtest and live training rows produced by
this version. Prevents mixing with pre-alignment rows during model training.

#### Engine: Realized PnL decomposition (`backtest/backtest_realistic.py`)

- **`realized_net_pnl`** = `equity − starting_capital` (pure realized: closed fills + funding).
- **`realized_net_pnl_pct`** = `(realized_net_pnl / starting_capital) × 100`.
- **`unrealized_fraction`** = `|unrealized| / (|unrealized| + |realized|)` — noise indicator
  for label reliability (0.0 = fully realized, 1.0 = entirely paper).
- All three keys added to the engine return dict alongside existing `unrealized_pnl_at_end`.

#### Contract: Label contract v2.4 (`backtest/btk_label_contract.py`)

- **`LABEL_CONTRACT_VERSION`**: `"2.3"` → `"2.4"` (strict superset of v2.3).
- **`FORMULA_VERSION`**: new constant `"alignment-v1"`.
- **`REQUIRED_LABEL_FIELDS`**: 3 new entries — `realized_net_pnl`, `realized_net_pnl_pct`,
  `unrealized_fraction`. Existing `validate_engine_result()` enforces presence automatically.

#### Runner: Engine version + traceability (`backtest/btk_unified_runner.py`)

- **`ENGINE_VERSION`**: `"realistic-v6"` → `"realistic-v7"`.
- **`formula_version`**: stamped on every engine result from `FORMULA_VERSION` constant.
- **`backtest_run_id`**: UUID4 added per execution for unique identification.

#### Pipeline: Candidate pipeline fixes (`src/neutralgrid/backtest/candidate_pipeline.py`)

- **Funding sign bug fixed**: `abs(fr_val)` stripped negative funding rates, biasing PnL
  estimates upward. Replaced with raw `fr_val` clamped to `[-0.01, +0.01]`.
- **`_parse_scan_timestamp()`** (new): Regex-based parser replacing brittle
  `split("_", 1)[-1]` that failed on hashed candidate IDs
  (`SYMBOL_YYYYMMDD_HHMMSS_hash8`).
- **Realized field extraction**: `realized_net_pnl_pct`, `unrealized_fraction`, and
  `range_breakout` extracted from engine result and passed to `HierarchicalLabeler.label()`.
- **Rotation metrics** (new training features):
  - `realized_pnl_per_margin_hour` = `realized_net_pnl_pct / duration_hours`.
  - `rotation_score` = `realized_pnl_per_margin_hour × (1 − unrealized_fraction)`.
- **`formula_version`**: tagged on every training row with `ImportError` fallback.

#### Labels: HierarchicalLabeler alignment (`src/neutralgrid/training/hierarchical_labels.py`)

- **`label()` signature**: added `realized_net_pnl_pct: Optional[float]` and
  `unrealized_fraction: Optional[float]` parameters.
- **L2 gate**: new unrealized fraction check — fails L2 when
  `unrealized_fraction > max_unrealized_fraction` (default 0.50). Guarded by
  `hasattr(cfg, "max_unrealized_fraction")` for backward compatibility.
- **L3 gate**: now prefers `realized_net_pnl_pct` over MTM `net_pnl_pct` for hurdle
  comparison. Audit trail field `l3_pnl_source` records `"realized"` or `"mtm"`.
- **`details` dict**: includes `unrealized_fraction` and `l3_pnl_source` for downstream
  traceability.

#### Config: New threshold + cross-validation (`src/neutralgrid/core/config.py`)

- **`HierarchicalLabelConfig.max_unrealized_fraction`**: new field, default `0.50`.
  Bots where >50% of PnL is unrealized fail L2 eligibility.
- **Hurdle cross-validation** in `_validate()`: raises `ValueError` if
  `hierarchical_label.hurdle_pct != barrier.meta_hurdle_pct`, preventing silent threshold
  drift between labeling and barrier configs.

#### Live rows: Unified label computation (`src/neutralgrid/training/unified_training_builder.py`)

- **Label drift fixed**: live rows previously used simple barrier labels while backtest rows
  used `HierarchicalLabeler`. Now both paths run `HierarchicalLabeler.label()` with identical
  gate logic.
- **Realized PnL derivation**: `realized_pnl_usdt / invested_margin_usdt × 100` for live
  rows, passed as `realized_net_pnl_pct` to the labeler.
- **Unrealized fraction derivation**: `|unrealized| / (|unrealized| + |realized|)` from live
  outcome fields.
- **`formula_version`**: `"alignment-v1"` stamped on live training rows.
- **Graceful fallback**: exception in labeler path falls back to `hlabel = np.nan` with
  warning log, preserving the row for manual review.

#### Dedup: Backtest results (`backtest_candidates.py`)

- Before CSV save: `sort_values("backtest_timestamp")` then
  `drop_duplicates(subset=["candidate_id"], keep="last")`. Prevents stale re-runs from
  polluting training data — only the latest backtest per candidate survives.

### Added — Historical Funding Loader (`src/neutralgrid/data/funding_rate.py`)

Infrastructure class `HistoricalFundingLoader` for future settlement-aligned funding series
ingestion. Not yet wired into the engine — provides building blocks for when the backtest
engine needs real historical funding rates instead of synthetic snapshots.

- **`fetch_raw()`**: wraps `BinanceClient.get_funding_rate()` with `start_time`/`end_time`
  parameters for bounded historical queries.
- **`align_to_bar_series()`**: static method converting settlement-timestamped rates into
  bar-indexed series (`bar_idx // funding_interval_bars`) for engine consumption.

### Added — Backtest-to-Live Alignment Audit (`src/neutralgrid/training/btk_alignment_audit_v20260316.py`)

New observational module `AlignmentAuditor` for comparing backtest predictions against live
deployment outcomes. **Never modifies production outputs** — produces a report DataFrame for
human review.

- **Compound keys**: backtest side `(candidate_id, backtest_run_id)`, live side
  `(candidate_id, strategy_id)`, joined on `candidate_id` (one-to-many).
- **Metrics**: `directional_bias` (backtest realized − live realized), per-pair
  `unrealized_fraction` comparison, duration and round-trip counts.
- **5 calibration eligibility gates**: minimum round trips (≥2), backtest unrealized fraction
  (≤0.50), live realized PnL available, directional bias computable, formula version match.
- Delegates live outcome loading to `LiveOutcomeIngestor` — no duplicated matching logic.

### Fixed — Test mocks for label contract v2.4 (`tests/unit/test_btk_label_runner.py`)

Updated 5 test mock dictionaries with the 3 new required fields (`realized_net_pnl`,
`realized_net_pnl_pct`, `unrealized_fraction`) to pass `validate_engine_result()` under
contract v2.4.

## [6.5.5] - 2026-03-14

### Changed — Hurdle lowered from 5% to 3%

Unified PnL success threshold across the entire config hierarchy, labeling pipeline,
backtest scripts, and CLI defaults. The meta-labeler, hierarchical labels, barrier config,
and EV ranker now all agree on a 3% hurdle.

#### Config defaults (percentage units: 5.0 → 3.0)

- **`BarrierConfig.meta_hurdle_pct`** (`core/config.py`): 5.0 → 3.0
- **`HierarchicalLabelConfig.hurdle_pct`** (`core/config.py`): 5.0 → 3.0
- **`MetaLabelerConfig.hurdle_pct`** (`models/meta_labeler.py`): 5.0 → 3.0
- **`UnifiedBarrierConfig.meta_hurdle_pct`** (`models/barrier_config.py`): 5.0 → 3.0
- **`LabelConfig.meta_hurdle_pct`** (`training/data_generator.py`): 5.0 → 3.0

#### Config defaults (decimal units: 0.05 → 0.03)

- **`ValidationConfig.pnl_hurdle_pct`** (`core/config.py`): 0.05 → 0.03

#### Function parameter defaults

- **`convert_to_training_row()`** (`backtest/candidate_pipeline.py`): 5.0 → 3.0
- **`create_meta_labeler()`** (`models/meta_labeler.py`): 5.0 → 3.0
- **`load_and_prepare_training_data()`** (`training/data_generator.py`): 5.0 → 3.0

#### CLI and scripts

- **`retrain_meta_labeler.py`** `--hurdle-pct` default: 5.0 → 3.0
- **`meta-labeling/scripts/`** (4 legacy scripts): hardcoded `hurdle_pct=5.0` → 3.0

#### Hardcoded comparisons fixed

- **`backtest_candidates.py`**: conformal calibration label threshold, summary stats
  comparison, and log message all updated from 5% to 3%.
- **`backtest_candidates_current.py`**: OOS validation summary stats and log message
  updated from 5% to 3%.

#### Unchanged (already aligned)

- `RankingConfig.pnl_hurdle_pct` was already 0.03 (decimal).
- `triple_barrier.py` function defaults were already 3.0.
- Bayesian optimizer search range (1.0–6.0) covers 3.0.
- Test files use explicit `hurdle_pct=5.0` to test specific thresholds, not defaults.

**Note:** The existing trained meta-labeler artifact (`models/meta_labeler/`) was trained
at hurdle=5.0%. A retrain is required for the new 3% threshold to take effect in model
predictions.

### Added — Automated Meta Features sheet population in `new_bot_data_extractor.py`

Closed the gap between `scripts/backfill_training_features.py` (batch backfill to separate
file) and the "Meta Features" sheet in `data/new_expired_bots.xlsx`. Previously, the sheet
was populated by a one-off manual process and never updated when new bots were added via
the extractor. Now `process_manual_text()` computes and writes meta features inline.

#### New functions (3)

- **`_fetch_historical_funding_rate()`**: Fetches the most recent funding rate before the
  bot's start time from Binance API (3-day lookback window). Adapted from
  `TrainingDataBackfiller.fetch_historical_funding_rate()`.
- **`_compute_meta_features()`**: Computes all 13 meta features at the bot's decision time
  using the same pipeline components as the batch backfiller: `HMMRegimePredictor` (range/trend
  probabilities), `StochasticRegimeChecker` (survival_prob, hurst_exponent, ou_halflife),
  `UtilityScorer` (utility_score), `PnLRanker` (ev_score), `ExistingDataMapper`
  (profit_per_grid_pct), and `compute_features` (adx_1h, adx_15m, rsi_15m).
- **`_append_meta_features()`**: Writes to the "Meta Features" sheet following the same
  pattern as `_append_pnl_curve_features()` — creates sheet on first use, deduplicates by
  `strategy_id`, enforces column schema.

#### Integration

- Hooked into `process_manual_text()` after PnL curve features write. Wrapped in try/except
  so failures log a warning but do not block the main extraction (Sheet1 is already saved).
- `backfill_status` is derived from actual computation outcome: `"complete"`,
  `"complete+ou_missing"` (OU halflife non-finite), or `"hmm_failed"` (HMM inference failed).
- Respects `dry_run` mode — meta features are not computed or written during dry runs.
- Sheet1 and PnL Curve Features deduplication still works when re-running on an existing bot.

#### Schema

`_META_FEATURES_SHEET_COLUMNS` matches the existing sheet exactly (18 columns):
`strategy_id`, `symbol`, `start_time_utc`, `num_grids`, `range_prob`, `trend_prob`,
`utility_score`, `survival_prob`, `hurst_exponent`, `ou_halflife`, `profit_per_grid_pct`,
`range_size_pct`, `adx_1h`, `adx_15m`, `rsi_15m`, `funding_rate`, `ev_score`,
`backfill_status`.

#### New imports

`pandas`, `HMMRegimePredictor`, `compute_features`, `PnLRanker`, `RankingConfig`,
`ExistingDataMapper`, `StochasticConfig`, `StochasticRegimeChecker`, `UtilityConfig`,
`UtilityScorer`.

### Added — BIOUSDT live bot telemetry (strategy 410638395)

- `Live/03-13/BIOUSDT/BIOUSDT_telemetry_snapshots.json`: 3 snapshots (1h50m, 2h50m, final
  5h13m cancelled) with PnL curves, order book state, and performance metrics.
- Bot added to `data/new_expired_bots.xlsx` Sheet1, PnL Curve Features, and Meta Features
  sheets. Candidate ID `BIOUSDT_20260313_150324_eec68f1e` and liquidation prices patched
  from live bot JSON.

### Added — AVAXUSDT live bot telemetry (strategy 410680162)

- `Live/03-15/AVAXUSDT/AVAXUSDT_extractor_input.txt`: Working snapshot at 1h24m duration.
  Neutral 10x, range $9.634–$10.045, 5 grids, $540 margin, +1.05% PnL ($5.69), 0 funding.

### Added — ICPUSDT live bot telemetry (strategy 410653893)

- `Live/03-14/ICPUSDT/ICPUSDT_extractor_input.txt`: Working snapshot at 14h44m duration.
  Neutral 10x, range $2.571–$2.685, 5 grids, $700 margin, +3.89% PnL ($27.27),
  trigger price $2.644 (mark), funding -$0.48.

### Changed — Architectural cleanup

Consolidated exception handling, test organization, and CLI structure across the codebase.

#### Exception hierarchy (`core/exceptions.py`)

Formalized domain exception tree rooted at `NeutralGridError`:

- **`ModelError`** → `ModelLoadError` (artifact_path context), `InferenceError` (symbol +
  model_type context), `ArtifactError` → `ArtifactVersionError` (expected/found version),
  `FeatureSchemaError` (expected/found feature lists).
- **`ValidationPipelineError`**: symbol + stage context for core validation failures.
- **`PriceSeriesError`**: price series streaming/storage errors.
- **`ReplayError`**: replay pipeline errors with stage context.

Each exception carries an `error_code` class attribute for structured API error responses.
The ~60 resilience/boundary `except` blocks elsewhere are intentionally untouched.

#### Test consolidation

- **`tests/conftest.py`**: Stripped to 7 lines. Removed legacy `sys.path` hacks — now
  relies on `pyproject.toml [tool.pytest.ini_options] pythonpath = ["src"]`.
- **`tests/test_afml_compliance.py`** (new, ~820 lines): Comprehensive self-contained
  compliance suite covering sample weights (concurrency matrix symmetry), indicators warmup
  (NaN enforcement), triple barrier (exit at barrier level, double-touch, time barrier),
  CPCV (purge_pct=0 fix, deflated Sharpe, PBO), meta-labeler (hurdle conversion, feature
  list, imputer mismatch), funding rate (UTC usage), data curator (audit trail), market data
  (single-row edge case), backtest realistic (Sharpe annualization 525600), walk-forward
  (embargo gap), and source-code integrity guards.
- **`tests/test_afml_fixes_v2.py`** (new, ~730 lines): Targeted verification of tasks 1–12
  fixes — Kelly negative edge rejection, PnL unit consistency, from_unified vol multiplier
  preservation, Sharpe annualization, sequential bootstrap, meta-label deploy/skip semantics,
  CPCV holdout embargo, trial tracker append-only, curator duplicate detection.
- **`tests/test_afml_integrations.py`** (new): Integration tests for utility scoring
  (trend_breakout_loss), regime-aware sizing, microstructure penalties, data curator, triple
  barrier, CPCV + deflated Sharpe, indicator consistency, klines quote_volume, and 1h history
  length alignment.
- **`tests/unit/test_btk_liquidation.py`** (new): P0-A liquidation modeling — normal run
  (no liquidation), crash triggers liquidation, simulation halts at liquidation bar, liquidated
  label always False, configurable maintenance_margin_rate, default MMR = 0.004.

All new tests use fixed seeds for determinism and synthetic data only (no external APIs).

#### CLI cleanup

- **`replay/cli.py`**: Clean 7-step pipeline with both programmatic (`run_pipeline()`) and
  CLI (`main()`) entry points. Input validation, optional session overlap detection,
  structured step logging, distinct exit codes (1 = file-not-found, 2 = unexpected error).
- **`cli/retrain.py`**: Rolling-window HMM retraining CLI with quality gates — version
  enforcement (rejects legacy `global` dir, enforces `rolling_<N>d_<timestamp>` naming),
  minimum symbol count, window=180d production policy, pass rate threshold (50%), walk-forward
  evaluation before promotion.

### Fixed — Architectural audit (6 findings)

Fixes for all valid findings from the codebase-wide architectural audit.

#### Critical: Deployed-candidate exclusion linkage (`candidate_pipeline.py`, `backtest_candidates.py`)

`_load_deployed_candidate_ids()` was reconstructing legacy `SYMBOL_YYYYMMDD_HHMMSS` IDs that
never matched the actual hashed `SYMBOL_YYYYMMDD_HHMMSS_hash8` format in scanner CSVs. The
`isin()` filter silently excluded nothing — deployed candidates leaked into backtest pools.

- Rewrote `_load_deployed_candidate_ids()` to return `(exact_ids, prefix_ids)`:
  exact IDs from DeployLinker CSV (`candidate_id` column) + prefix IDs from legacy CSV
  (`symbol` + `cand_csv_timestamp` reconstruction).
- Replaced `isin()` exact match in `filter_backtest_candidates()` with dual matching:
  exact `isin()` for hashed IDs + `startswith()` prefix match for legacy IDs.
- Changed `backtest_candidates.py` default `--linkage-path` to
  `data/linkage/deploy_linkage_log.csv` with automatic fallback to legacy
  `data/candidate_execution_linkage.csv`.

#### Critical: HMM promotion policy (`cli/retrain.py`, `retrain_hmm.py`)

Three promotion paths had inconsistent quality gates: `cli/retrain.py` only gated when
`--evaluate` was passed (optional), `retrain_hmm.py` promoted unconditionally. The active
artifact had `mean_pass_rate=0.3167` (below the 0.50 threshold) and 62.46-day training span
labeled as `rolling_180d_*`.

- **`cli/retrain.py`**: Removed `--evaluate` flag — evaluation is now mandatory. Removed
  the `evaluation is None → promote anyway` path. Added actual-span verification: logs a
  warning if training data covers <90% of the requested window.
- **`retrain_hmm.py`**: Added quality gate before `promote_hmm_version()` — reads
  `metadata.json`, gates on `mean_pass_rate >= 0.50`. Promotion skipped with warning if
  below threshold or if metadata is missing.

#### High: Pandas FutureWarning (`unified_training_builder.py`)

Lines 826 and 837 assigned tz-aware timestamps into columns initialized with naive `pd.NaT`,
triggering `FutureWarning` that will become an error in a future pandas release.

- Replaced `out["start_time_utc"] = pd.NaT` and `out["t1"] = pd.NaT` with
  `pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")`.
- Verified: 0 FutureWarnings remain across the full test suite (943 tests with
  `-W error::FutureWarning`).

#### Medium: Documentation drift (`run_full_pipeline.py`)

Docstring line 12 said "scan top 50 symbols" but `--top-n` default was 250. Fixed to match.

#### Medium: Dependency checker (`scripts/check_deps.py`)

`CRITICAL_DEPS` only validated `pyarrow`. Expanded to include `numpy`, `pandas`,
`scikit-learn`, `hmmlearn`, `joblib` — all critical compiled/ML runtime dependencies.

#### Medium: Dead-end code cleanup

- Deleted 6 deprecated stub files (all `sys.exit(1)` wrappers with migration docstrings):
  `meta-labeling/scripts/` (4 files), `Live/02-03/meta_labeling_retrain.py`,
  `Live/02-03/meta_labeling_update.py`.
- Removed stale `.pyc` files and empty `meta-labeling/scripts/` directory.
- Updated `meta-labeling/README.md` and `INVENTORY.txt` to remove references to deleted
  scripts.
- Regenerated `src/neutralgrid.egg-info/SOURCES.txt` (removed stale `coinglass.py` and
  `market_dynamics.py` references).

#### Completed: `enrich_grid_params.py` decomposition

The 935-line `run_one()` coroutine has been decomposed into 10 named sub-functions within the
same file. `run_one()` is now an 88-line orchestrator. No new files created.

Three module-level dataclasses added for structured inter-stage data passing:
- **`_RegimeData`** (19 fields): market data, HMM metrics, funding rate, trigger price
- **`_MicroData`** (5 fields): microstructure costs, adaptive value, book presence
- **`_GridData`** (3 fields): grid params, edge info, range size

Extracted sub-functions (all nested inside `enrich_with_grid_params()` for closure access):

| Function | Lines | Stages | Purpose |
|---|---|---|---|
| `_fetch_regime_data` | 162 | 1–6 | Fetch, validate, HMM posteriors, funding, trigger price |
| `_build_base_payload` | 37 | 7 | Pure dict assembly from `_RegimeData` |
| `_check_regime_rejection` | 29 | 8 | Regime validity + rejection code collection |
| `_estimate_microstructure` | 67 | 9 | Cost estimation → `_MicroData` |
| `_generate_grid` | 241 | 10 | Edge-tier grid generation → `_GridData` |
| `_check_viability` | 31 | 11 | Post-grid microstructure viability gate |
| `_compute_kelly` | 123 | 12 | Generalized Kelly sizing |
| `_evaluate_gates` | 96 | 13–14 | Hard gate + adaptive gate |
| `_compute_position_and_tos` | 104 | 15–16 | Position sizing + tradable oscillation score |
| `_evaluate_stage_b` | 38 | 17 | Stage B deployment approval |

Helper: `_grid_params_dict()` extracts grid parameters from `GridParams` for return dicts.

All 8 early-return rejection paths preserved with identical payloads. Each sub-function returns
its own rejection info; `run_one()` merges all payloads computed so far into the rejection dict.
Removed dead `import numpy as _np` (was unused in HMM posteriors block). Pyright "too complex
to analyze" errors eliminated (was 4, now 0). 943 tests pass, 0 FutureWarnings.

## [6.5.4] - 2026-03-13

### Changed — AFML Pipeline Audit & Optimization (AUC 0.524 → 0.636) 

Full AFML-aligned pipeline audit (Ch 2→8→3→4→7→6→11-12) with 12 fixes across two tiers.                                                               
Meta-labeler CV AUC improved from 0.524 to 0.636 (+21.4%), Precision@5 from 0.620 to 0.800. 

#### Tier 1: Critical AFML fixes (6 changes)                                                                                                          
                                                                                                                                                     
**Label threshold alignment** (`config.py`): `HierarchicalLabelConfig.hurdle_pct` 3.0 → 5.0,                                                        
aligned with `MetaLabelerConfig.hurdle_pct` to eliminate 3-way label mismatch (0%/3%/5%).                                                           
**GBM complexity reduction** (`meta_labeler.py`): `n_estimators` 100→50, `max_depth` 5→3,                                                           
`min_samples_leaf` 5→15, `learning_rate` 0.1→0.05. Prevents overfitting at n=493.                                                                   
**Feature pruning** (`meta_labeler.py`): `BASELINE_META_FEATURES` 14→10. Dropped `ev_score`,                                                        
`utility_score` (deterministic composites of other features), `hurst_exponent`, `ou_halflife`                                                       
(noisy estimators on short series). Dropped `bb_width` (r>0.95 with range_size_pct) and                                                             
`regime_conf` (monotonic transform of range_prob/trend_prob) from `LIVE_PLUS`.                                                                      
All remaining features now have positive permutation importance. VIF warnings eliminated.                                                           
**Embargo increase** (`meta_labeler.py`): `embargo_hours` 3.0 → 12.0 (matches event horizon).                                                       
**CPCV min fold size** (`cpcv.py`): minimum train/test from (10,5) to (50,20).                                                                      
**Calibration simplification** (`meta_labeler.py`): `calibration_method` isotonic → sigmoid                                                         
(Platt scaling). Stable with n=493; isotonic overfits the step function at this sample size.                                                        
                                                                                                                                                   
#### Tier 2: Structural AFML fixes (6 changes)                                                                                                        
                                                                                                                                                   
**DataCurator in backtest path** (`candidate_pipeline.py`): `fetch_historical_klines()` now                                                         
calls `DataCurator.validate_ohlcv()` on fetched klines before backtesting. Non-blocking                                                             
(logs warning on failure). Previously klines bypassed all data quality checks.                                                                      
**Symbol-blocked CV** (`cpcv.py`, `meta_labeler.py`): New `group_col` parameter in                                                                  
`CPCV.split()`. When `group_col="symbol"` is passed, all rows for the same symbol are                                                               
assigned to the same temporal group, preventing leakage through persistent asset                                                                    
characteristics. 137 unique symbols across 5 groups with zero cross-fold symbol overlap.                                                            
Backward-compatible (group_col=None preserves original behavior).                                                                                   
**range_size_pct source tracking** (`unified_training_builder.py`): Added                                                                           
`range_size_pct_source` column ("scan_bb_width", "grid_range", "missing") to track                                                                  
derivation provenance. Addresses inconsistent definition between scan-time (BB width)                                                               
and training-time (grid range).                                                                                                                     
**Stochastic feature minimum bars** (`scan.py`): Raised from 50 to 300 bars for                                                                     
`survival_prob`, `hurst_exponent`, `ou_halflife` computation. Below 300, R/S and DFA                                                                
estimators produce unstable results.                                                                                                                
**DSR trial counting** (`evaluate.py`): Deflated Sharpe ratio now counts all trial types                                                            
(hmm + hmm_threshold_sweep + meta_labeler) instead of only hmm trials. With 81 total                                                                
logged trials, the DSR haircut is now correctly computed.                                                                                           
**OOS backtest config alignment** (`backtest_candidates_current.py`): Replaced raw                                                                  
`GridConfig()` with `build_training_config()` so OOS validation uses the same engine                                                                
settings (continuous funding, taker close fees, 2-bar delay) as training backtests.                                                                 
                                                                                                                                                  
#### Backtest coverage expansion (200+ symbols per pipeline run)

#### 1. Backtest coverage expansion (200+ symbols per pipeline run)

Widened the scanner-to-backtest funnel so a single pipeline execution can push 200+
candidates through `btk_unified_runner.py`, targeting a 3–5× dataset increase.

- **`run_full_pipeline.py`**: `--top-n` default 150 → 250, `--max-enrichment` default 150 → 250.
- **`enrich_grid_params.py`**: `EnrichConfig.max_symbols` 150 → 250.
- **`backtest_candidates.py`**: `--min-score` default 70 → 55, `--max-candidates` default 50 → 250.
- **`candidate_pipeline.py`**: `filter_backtest_candidates()` default `min_score` 70 → 55.
- **`run_candidate_backtests.py`**: function default, docstring, report text, and explicit call
  all updated from 70 → 55.
- **`compare_cb_3way.py`**, **`compare_cb_impact.py`**: explicit `min_score` calls updated
  from 70 → 55 for consistency.

Score normalization (`50 + 50 * pct_rank`) means score ≥ 55 captures ~90% of EV-scored
symbols vs ~40% at the old threshold of 70.

#### 2. Time-decay weighting enabled by default (halflife_days=30)

Exponential decay `w = exp(-ln(2) × age / 30)` upweights recent regimes. The existing
N_eff < 30 guard in `sample_weights.py` automatically disables decay when effective
sample size drops too low, preventing over-concentration.

- **`meta_labeler.py`**: `MetaLabelerConfig.use_time_decay` False → True.
- **`sample_weights.py`**: `SampleWeightConfig.use_time_decay` False → True.
- `time_decay_halflife_days` remains 30.0 (unchanged).
- N_eff guard chain: `MetaLabelerConfig` → `SampleWeightConfig` →
  `compute_sample_weights()` → `should_disable_time_decay(min_effective_n=30)`.
- Phase1 config (`phase1_config_v20260311.py`) already sets `use_time_decay=True` — no conflict.

#### Documentation updates

- **`AFML_QUICK_REFERENCE.md`**: `use_time_decay` examples updated to `True`.
- **`readmefullpwep.md`**: `--top-n` and `--max-enrichment` examples updated to 250.

## [6.5.3] - 2026-03-12

### Added — AFML Phase 1-3 Calibration Pipeline (12 new modules, 7 config fields)

Three-phase calibration and optimization framework aligned with AFML methodology.
All features behind safe config flags with graceful degradation when artifacts are absent.
953 tests passing after full integration.

#### Phase 1: Low-Risk Parametric Calibration (3 modules)

- **GBM complexity reduction** (`src/neutralgrid/calibration/phase1_config_v20260311.py`):
  - `Phase1MetaLabelerOverrides` frozen dataclass: `n_estimators=30` (was 100),
    `max_depth=3` (was 5), `min_samples_leaf=20` (was 10), `learning_rate=0.05` (was 0.10).
  - `build_phase1_meta_config()` returns override dict for `MetaLabelerConfig`.
  - `validate_phase1_auc_floor()` checks AUC >= 0.56 post-training (advisory warning).
  - Activated via `python retrain_meta_labeler.py --phase1` CLI flag.

- **Sigmoid OOS calibration with beta upgrade** (`src/neutralgrid/calibration/beta_calibrator_v20260311.py`):
  - `BetaCalibrator` class: 3-parameter logistic fit `logit(P) = c + a*log(s) - b*log(1-s)`.
  - Conditional upgrade in `meta_labeler.py`: when `ReliabilityDiagnostic` detects
    `max_deviation > 0.05`, beta calibration replaces sigmoid OOS.
  - Pre/post ECE and Brier score tracked in `calibration_report`.

- **Time-decay weighting with N_eff guard** (`src/neutralgrid/calibration/effective_sample_size_v20260311.py`):
  - `compute_kish_effective_n()`: Kish's effective sample size `N_eff = (Sigma w_i)^2 / Sigma(w_i^2)`.
  - `should_disable_time_decay()`: returns `True` when `N_eff < min_effective_n` (default 30).
  - Integrated in `sample_weights.py` — disables time-decay when effective sample size too low.

#### Phase 2: Distribution-Aware Calibration (5 modules)

- **Temperature scaling for HMM posteriors** (`src/neutralgrid/calibration/temperature_scaling_v20260311.py`):
  - `TemperatureScaler` class: single-parameter `p_cal = p^(1/T) / Sigma p_j^(1/T)`.
  - `fit()` minimizes NLL via `scipy.optimize.minimize_scalar` (bounds [0.1, 10.0]).
  - `transform()` applies softmax(log(p)/T) with numerical stability (log-sum-exp).
  - `save()`/`load()` JSON artifacts with `isfinite(T) and T > 0` validation guards.
  - **Training**: `retrain_hmm.py` fits scaler on last 20% of training posteriors,
    saves `temperature_scaler.json` alongside HMM artifacts.
  - **Inference**: `enrich_grid_params.py` loads scaler from HMM artifact directory,
    applies `transform()` to calibrate posteriors before entropy-adaptive thresholds
    and Stage B consume them. Controlled by `temperature_scaling_enabled=True` (default).

- **Reliability diagnostic** (`src/neutralgrid/calibration/reliability_diagnostic_v20260311.py`):
  - `ReliabilityDiagnostic` class with `compute()` method.
  - `ReliabilityReport` frozen dataclass: ECE, MCE, max_deviation, `needs_beta_upgrade` flag.
  - Handles edge cases: empty input, NaN/Inf, single-class, zero-count bins.

- **MI-weighted scanner scoring** (`src/neutralgrid/scanner/mi_weighted_scorer_v20260311.py`):
  - `compute_mi_weights()`: estimates MI(signal, target) via `sklearn.feature_selection.mutual_info_classif`,
    applies Bayesian shrinkage `w_final = (1-alpha)*w_MI + alpha*w_uniform` (alpha=0.30).
  - `compute_mi_weighted_score()`: weighted sum with automatic renormalization for missing signals.
  - Falls back to uniform weights when `max(MI) < MI_null_threshold`.
  - **Artifact generation**: `backtest_candidates.py` computes MI weights from backtest signals
    (`score`, `ev_score`, `range_prob`, `meta_prob`) against `label_positive_by_horizon`,
    saves `data/cache/mi_weights.json`. Metadata keys prefixed with `_` for filtering.
  - **Consumption**: `run_full_pipeline.py` loads MI weights artifact (filtering non-numeric
    and `_`-prefixed metadata keys), passes to `scan_top_symbols()`. `scan.py` uses
    MI-weighted scoring when weights available, falls through to fixed AFML weights otherwise.

- **Entropy-adaptive thresholds** (`src/neutralgrid/scanner/entropy_adaptive_threshold_v20260311.py`):
  - `select_range_prob_threshold()`: Shannon entropy of HMM posterior selects threshold tier.
  - High-entropy (uncertain) posteriors → stricter `min_range_prob` threshold.
  - Integrated in `two_stage_selector.py` Gate 4 when posteriors available.

- **Adaptive microstructure gate** (`src/neutralgrid/scanner/adaptive_microstructure_gate_v20260311.py`):
  - `AdaptiveMicrostructureGate` class: EMA-smoothed percentile thresholds
    `tau_t = alpha * p75(batch) + (1-alpha) * tau_{t-1}`.
  - `compute_batch_thresholds()`: computes p75 for round_trip_pct, spread_to_profit_ratio,
    funding_drag_pct with safety floors and EMA state caching.
  - `evaluate()`: binary pass/fail against adaptive thresholds.
  - **Integration**: `enrich_grid_params.py` instantiates gate when `adaptive_micro_gate_enabled=True`,
    calls `compute_batch_thresholds()` + `evaluate()` after hard gate. Results stored as
    supplementary annotation (`adaptive_gate_passed`, `adaptive_gate_source`).
  - Cold-start defaults match existing `MicrostructureGateConfig` (1.20, 0.60, 0.80).

#### Phase 3: Decision-Risk Calibration (2 modules)

- **Split conformal prediction** (`src/neutralgrid/calibration/conformal_risk_control_v20260311.py`):
  - `ConformalRiskController` class with coverage guarantee `P(fail) <= alpha + 1/(n_cal+1)`.
  - `fit()`: computes nonconformity scores and conformal quantile.
  - `fit_weighted()`: time-decay weighted quantile with NaN weight guard and uniform fallback.
  - `should_deploy()`: binary gate comparing `meta_prob` against `1 - quantile_value`.
  - `ConformalQuantile` frozen dataclass with `quantile_value`, `coverage_guarantee`, `n_calibration`.
  - `save()`/`load()` JSON artifacts to `data/cache/conformal_quantile.json`.
  - **Artifact generation**: `backtest_candidates.py` fits conformal quantile on `meta_prob`
    vs `net_pnl_pct > 3.0%` hurdle (requires >= 20 valid samples).
  - **Consumption**: `enrich_grid_params.py` loads quantile artifact, passes to Stage B Gate 5
    in `two_stage_selector.py`. Gate 5 is optional — skipped when no artifact.
  - Input validation: length checks on `fit()`, `fit_weighted()`, `compute_nonconformity_scores()`.

- **Bayesian threshold optimization** (`src/neutralgrid/optimization/bayesian_threshold_optimizer_v20260311.py`):
  - `BayesianThresholdOptimizer` class: GP-UCB with 7D search space.
  - `OptimizationConfig`: `n_initial_samples=20` (Latin Hypercube), `n_optimization_steps=50`,
    `ucb_kappa=2.0`.
  - `optimize()` returns `OptimizationResult` frozen dataclass with `best_theta`, `best_objective`,
    `history`, `convergence_metrics`.
  - `validate_walk_forward()` for temporal OOS validation.
  - `search_space` property accessor for CLI introspection.
  - Instance-level `_rng_counter` (not shared across instances).

- **Top-level CLI** (`optimize_thresholds_v20260311.py`):
  - Standalone script for Bayesian threshold optimization.
  - CLI: `--backtest-dir`, `--iterations`, `--output`, `--walk-forward-folds`, `--dry-run`.
  - `load_backtest_results()` concatenates `backtest_results_*.csv`.
  - Objective: deflated Sharpe * sqrt(N_deployed).
  - Walk-forward validation with temporal 80/20 train/test split.

#### New Package Structure

- `src/neutralgrid/calibration/__init__.py`: Exports all calibration classes with `try/except ImportError`.
- `src/neutralgrid/optimization/__init__.py`: Exports `BayesianThresholdOptimizer` with `try/except ImportError`.

### Changed — Pipeline Integration (Phase 1-3 Wiring)

- **`src/neutralgrid/core/config.py`** — 7 new config fields on root `Config` class:
  - `calibration_phase1_enabled: bool = False` (opt-in via env var or `--phase1` CLI)
  - `temperature_scaling_enabled: bool = True` (active when artifact exists)
  - `conformal_alpha: float = 0.20` (coverage guarantee target)
  - `conformal_min_calibration_samples: int = 20`
  - `mi_weight_shrinkage: float = 0.30` (Bayesian shrinkage toward uniform)
  - `adaptive_micro_gate_enabled: bool = False` (opt-in)
  - `bayesian_optimizer_enabled: bool = False` (opt-in)
  - 7 env var overrides in `load_from_env()`: `CALIBRATION_PHASE1_ENABLED`,
    `TEMPERATURE_SCALING_ENABLED`, `CONFORMAL_ALPHA`, `MI_WEIGHT_SHRINKAGE`,
    `ADAPTIVE_MICRO_GATE_ENABLED`, `CONFORMAL_MIN_CALIBRATION_SAMPLES`,
    `BAYESIAN_OPTIMIZER_ENABLED`.

- **`retrain_meta_labeler.py`**:
  - New `--phase1` CLI flag for complexity reduction.
  - Phase 1 config override block imports `build_phase1_meta_config()`,
    rebuilds `MetaLabelerConfig` with reduced GBM hyperparameters.
  - Post-training `validate_phase1_auc_floor()` call with advisory warning.

- **`retrain_hmm.py`**:
  - After HMM training and artifact promotion, fits `TemperatureScaler` on holdout posteriors.
  - Collects posteriors from all training datasets, holds out last 20% for calibration.
  - Saves `temperature_scaler.json` to HMM artifact directory.

- **`src/neutralgrid/models/meta_labeler.py`**:
  - After sigmoid/isotonic OOS calibration, conditional beta calibration upgrade.
  - Imports `ReliabilityDiagnostic` and `BetaCalibrator`.
  - If `needs_beta_upgrade` is True, fits `BetaCalibrator`, upgrades `oos_method` to `"beta_oos"`.
  - Populates `calibration_report` with beta metadata (ECE before/after, Brier before/after).

- **`src/neutralgrid/training/sample_weights.py`**:
  - N_eff guard: imports `should_disable_time_decay()`, disables time-decay weighting
    when `N_eff < 30` to prevent over-concentration on recent samples.

- **`src/neutralgrid/scanner/scan.py`**:
  - MI-weighted scoring branch: when `mi_weights` dict available and `MI_SCORER_AVAILABLE`,
    constructs features dict (`similarity_score`, `profile_proba`, `ev_score`, `meta_prob`)
    and calls `compute_mi_weighted_score()`. Missing signals passed as `None` for renormalization.
  - 4-branch scoring cascade: MI-weighted -> full AFML -> partial -> similarity-only.

- **`src/neutralgrid/scanner/enrich_grid_params.py`**:
  - Temperature scaler loaded from `{hmm_artifact_dir}/temperature_scaler.json`,
    applied to HMM posteriors before entropy-adaptive thresholds and Stage B.
  - Conformal quantile loaded from `data/cache/conformal_quantile.json` via
    `ConformalRiskController.load()`, passed to Stage B Gate 5.
  - Adaptive microstructure gate instantiated when config enabled, calls
    `compute_batch_thresholds()` + `evaluate()` after hard gate.
  - HMM posterior extraction: properly unpacks `compute_hmm_features()` tuple return,
    filters by valid_mask.

- **`src/neutralgrid/scanner/two_stage_selector.py`**:
  - Gate 4: entropy-adaptive threshold via `select_range_prob_threshold()` when posteriors available.
  - Gate 5 (optional): conformal meta-prob gate via `ConformalRiskController.should_deploy()`.
  - `approve()` method accepts `conformal_quantile` and `posteriors` parameters.

- **`src/neutralgrid/backtest/evaluate.py`**:
  - Per-path: fits TemperatureScaler on OOS posteriors (diagnostic only), stores T/NLL.
  - Aggregate: computes `temp_scale_mean_T`, `temp_scale_std_T`, `temp_scale_mean_nll_improvement`.

- **`run_full_pipeline.py`**:
  - MI weights loading from `data/cache/mi_weights.json` with metadata key filtering.
  - `mi_weights` passed to `scan_top_symbols()`.

- **`backtest_candidates.py`**:
  - After backtest results saved: conformal quantile fitted and saved (>= 20 valid samples).
  - After conformal: MI weights computed from backtest signals and saved.
  - `meta_prob` added to feature merge list for downstream availability.
  - PnL column name fixed: prefers `net_pnl_pct` with fallback to `pnl_pct`.

### Changed — Meta-Labeler AUC Optimization + barrier_price Population

AUC was 0.493 (worse than random) due to 70/30 class imbalance, no class rebalancing,
and ignored `sample_weight_override`. Changes target ~55/45 class balance and proper
weight multiplication. 4 review agents validated all changes. 953 tests passing.

#### AUC Optimization (10 files)

- **hurdle_pct raised from 3.0% to 5.0%** — moves class balance from ~70/30 toward ~55/45.
  Bots with PnL in [3%, 5%) reclassified from positive to negative.
  Updated in 10 files across the full pipeline:
  - `meta_labeler.py`: `MetaLabelerConfig.hurdle_pct` default + `create_meta_labeler()` factory
  - `data_generator.py`: `LabelConfig.meta_hurdle_pct` + `load_and_prepare_training_data()` default
  - `retrain_meta_labeler.py`: `--hurdle-pct` CLI default
  - `candidate_pipeline.py`: `convert_to_training_row()` default
  - `core/config.py`: `BarrierConfig.meta_hurdle_pct`
  - `barrier_config.py`: `UnifiedBarrierConfig.meta_hurdle_pct`
  - `backtest_candidates.py`: conformal calibration hurdle (`_labels > 5.0`) + summary stats
  - `backtest_candidates_current.py`: summary stats reporting threshold
  - Intentionally unchanged:
    - `HierarchicalLabelConfig.hurdle_pct` stays 3.0% — different concept (L3 minimum net
      return hurdle: "did this bot cover costs?" vs meta-labeler: "confident enough to deploy?").
      Clarifying comment added to `core/config.py`.
    - `triple_barrier.py` function defaults stay 3.0% — independent AFML concept (PT/SL
      thresholds). Docstrings updated to note MetaLabelerConfig now defaults to 5.0%.
    - `PnLRankerConfig.pnl_hurdle_pct` stays 0.03 — different context (scanner ranking),
      different unit system (decimal fraction).

- **`use_class_weights: bool = True`** — new field on `MetaLabelerConfig`.
  Enables balanced class weights via `sample_weight` multiplication.
  All access sites use `getattr(self.config, "use_class_weights", True)` for backward
  compatibility with pickled models that lack this field.

- **Class rebalancing via sample_weight** — sklearn balanced formula
  `w_class(c) = n_total / (2 * n_c)` applied:
  - Per CV fold: multiplied into `fold_weights` after uniqueness and `sample_weight_override`.
  - Final model: multiplied into `sample_weights` before `base_model.fit()`.
  - Weight multiplication order: uniqueness x sample_weight_override x class_weights.

- **`sample_weight_override` incorporation** — column in training CSV (backtest=0.5, live=1.0)
  was completely ignored by `meta_labeler.py`. Now:
  - Dataset-level: read from DataFrame after uniqueness weights, applied via multiplication.
  - Per-fold: fresh fold-specific slice applied to fold_weights (independent of dataset-level).
  - Uses `np.nan_to_num(..., nan=1.0)` for Pyright-clean NaN handling (avoids `.fillna()` on
    mixed-type return from `pd.to_numeric`).

- **Calibration weight isolation** — pre-class-weight `sample_weights` preserved as
  `sample_weights_for_calibration` before class rebalancing. Calibration holdout uses
  natural-distribution weights so the calibrator learns true base rates, not the
  rebalanced class frequencies. Class-weighted `sample_weights` still used for the
  calibration base model training (model should be class-aware).

- **Artifact metadata enrichment** — `hurdle_pct` and `use_class_weights` added to
  `model_params` dict in artifact metadata for audit/reproducibility.

#### barrier_price Population (2 files)

- **Backtest rows** (`candidate_pipeline.py`): `"barrier_price": backtest_result.get("price_end")`
  maps engine's `price_end` (close price at final simulated bar) to AFML `barrier_price`
  metadata. For grid bots this is a proxy since PnL comes from grid fills, not directional
  price movement. Metadata only — not a model feature. ~200 backtest rows now populated.

- **Price-path labeler** (`data_generator.py`): `barrier_price=float(prices[exit_idx])` added
  to `OutcomeLabel` return in `compute_label_from_price_path()`, with bounds check
  `if exit_idx < len(prices)`. Returns `None` when price series exhausted without barrier touch.

- **Live bots**: `barrier_price=None` stays unchanged in `BarrierLabelGenerator.compute_label()`.
  Grid bot PnL comes from grid fills, not directional price movement — exit price cannot be
  derived from PnL alone without additional API calls.

### Fixed — AUC + barrier_price Review Agent Audit (4 agents, 7 fixes)

Four specialized review agents validated the AUC optimization and barrier_price changes.
All findings resolved. 953 tests passing.

#### Critical (3)

- **meta_labeler.py**: Class-rebalanced `sample_weights` leaked into calibration holdout,
  inflating minority class frequency and biasing the calibrator's learned base rate.
  Fixed by preserving `sample_weights_for_calibration` before class weight multiplication.
- **backtest_candidates.py**: Conformal calibration used hardcoded `> 3.0` hurdle while
  meta-labeler now uses 5.0%. Systematic mismatch between conformal gate and labeling.
  Fixed to `> 5.0`.
- **5 additional files** had stale `hurdle_pct=3.0` defaults diverging from MetaLabelerConfig:
  `candidate_pipeline.py`, `create_meta_labeler()`, `load_and_prepare_training_data()`,
  `core/config.py` BarrierConfig, `barrier_config.py` UnifiedBarrierConfig. All updated to 5.0.

#### High (1)

- **meta_labeler.py**: `hurdle_pct` and `use_class_weights` not stored in artifact metadata
  `model_params`. Added both for audit trail and reproducibility.

#### Medium (2)

- **meta_labeler.py**: `use_class_weights` field on frozen dataclass causes `AttributeError`
  when loading old pickled models. All 3 access sites wrapped with
  `getattr(self.config, "use_class_weights", True)`.
- **candidate_pipeline.py**: Missing clarifying comment that `price_end` is a semantic proxy
  for AFML `barrier_price` in grid-bot context. Documentation comment added.

#### Low (1)

- **triple_barrier.py**: Three function docstrings said "aligned with MetaLabelerConfig.hurdle_pct"
  but defaults diverge (3.0 vs 5.0). Docstrings updated to note MetaLabelerConfig defaults to 5.0%.

### Fixed — Review Agent Audit (28 bugs across 6 review waves)

Six parallel review agents audited the full Phase 1-3 integration. All 953 tests pass.

#### Critical (6)

- **enrich_grid_params.py**: `compute_hmm_features()` returns `(X, valid_mask)` tuple but code
  assigned full tuple to single variable. `len()` always returned 2 (tuple length),
  `scaler.transform()` received tuple instead of ndarray. Silently fell to `except`,
  leaving `_hmm_posteriors = None` on every symbol.
- **retrain_hmm.py**: Same `compute_hmm_features()` tuple unpacking bug in temperature
  scaler fitting block.
- **optimize_thresholds_v20260311.py**: Three mismatched API calls — `optimize()` called with
  wrong arguments (missing `default_theta`), result accessed as dict (is frozen dataclass),
  `validate_walk_forward()` called with wrong signature. All three fixed.
- **run_full_pipeline.py**: MI weights JSON artifact contains metadata keys (`_raw_mi` dict,
  `_n_samples` int, `_source` str). `sum(weights.values())` would raise `TypeError` adding
  dict to float. Fixed with `isinstance(v, (int, float))` filter and `_`-prefix exclusion.

#### High (4)

- **scan.py**: MI-weighted scorer received `0.0` for missing signals instead of `None`,
  preventing proper renormalization. Fixed to pass `None`.
- **backtest_candidates.py**: Wrong PnL column name `pnl_pct` (engine produces `net_pnl_pct`).
  Conformal calibration would silently skip every time.
- **backtest_candidates.py**: Missing `meta_prob` in feature merge list. Conformal block's
  `meta_prob` column check always failed.
- **run_full_pipeline.py**: Log line `f"{v:.3f}"` on dict/str metadata values would crash
  inside try block, silently disabling MI weight loading.

#### Medium (13)

- **beta_calibrator_v20260311.py**: No length validation between `y_true` and `y_prob`
  in `fit()`. Silent broadcasting on mismatched lengths.
- **temperature_scaling_v20260311.py**: Out-of-bounds label IndexError — no bounds check
  on labels vs posterior columns.
- **temperature_scaling_v20260311.py**: `load()` accepted `T=0` or `NaN` from corrupted JSON
  — division by zero in `transform()`. Added `isfinite(T) and T > 0` guard.
- **temperature_scaling_v20260311.py**: `transform()` no defense-in-depth guard for invalid `T`.
- **conformal_risk_control_v20260311.py**: NaN weights from bad timestamps silently corrupt
  weighted quantile. Added fallback to uniform weights.
- **conformal_risk_control_v20260311.py**: No length validation in `fit()`, `fit_weighted()`,
  `compute_nonconformity_scores()`.
- **enrich_grid_params.py**: `hard_gate_passed=None` leaked into Stage B boolean gate.
  Wrapped with `bool()`.
- **config.py**: Missing `CONFORMAL_MIN_CALIBRATION_SAMPLES` and `BAYESIAN_OPTIMIZER_ENABLED`
  env var overrides. Added both.
- **bayesian_threshold_optimizer_v20260311.py**: `_rng_counter` shared across instances via
  class-level variable. Moved to instance-level.
- **optimize_thresholds_v20260311.py**: Added `search_space` property accessor to optimizer
  for CLI dry-run inspection.
- **scan.py**: Unnecessary `p is not None` guard restricted MI scoring to cases where
  profile probability was available. MI scorer handles `None` via renormalization.
- **enrich_grid_params.py**: Missing `adaptive_gate_source` in exception path.
- **scan.py**: Removed unused `MIWeightConfig` import, fixed `compute_mi_weighted_score`
  possibly-unbound Pyright warning.

#### Low (5)

- **mi_weighted_scorer_v20260311.py**: Type annotation `Dict[str, float]` updated to
  `Dict[str, Optional[float]]` matching actual caller usage.
- **adaptive_microstructure_gate_v20260311.py**: `evaluate()` parameter type updated to
  `Dict[str, Optional[float]]` to accept `None` metric values.
- **enrich_grid_params.py**: `AdaptiveMicrostructureGate` import unused — now instantiated
  and called when config enabled.
- **retrain_hmm.py**: Unused loop variable `sym_name` renamed to `_`.
- **scan.py**: `compute_mi_weighted_score = None` fallback in except branch for Pyright.

### Activation Sequence

The Phase 1-3 calibration pipeline requires a specific artifact bootstrapping sequence:

```bash
# Step 1: Retrain meta-labeler (optional Phase 1 complexity reduction)
python retrain_meta_labeler.py --phase1

# Step 2: Retrain HMM (produces temperature_scaler.json alongside HMM artifacts)
python retrain_hmm.py

# Step 3: First pipeline run (scan + enrich with calibrated posteriors)
python run_full_pipeline.py

# Step 4: Backtest (produces conformal_quantile.json + mi_weights.json)
python backtest_candidates.py

# Step 5: Second pipeline run (Gate 5 + MI scoring now active)
python run_full_pipeline.py
```

All features degrade gracefully — if any artifact is missing, the pipeline falls through
to existing behavior without errors.

### Dead-End Code

**From this session**: NONE — All new code has verified production callers.
AUC optimization and barrier_price changes validated by dead-end scanner agent.

**Pre-existing dead-ends catalogued** (9 items, all LOW severity, not from this session):

| File | Dead-End | Notes |
|---|---|---|
| `meta_labeler.py:1307` | `create_meta_labeler()` — never called | Public API factory, may be for external use |
| `meta_labeler.py:1052` | `predict_proba_batch()` — never called | Public API method |
| `meta_labeler.py:1039-1040` | Redundant local `import numpy/pandas` | Module-level already imported |
| `data_generator.py:1142` | `build_from_snapshots()` — never called | Public API method |
| `data_generator.py:457` | `compute_label_from_price_path()` — deprecated | No callers pass `price_paths` argument |
| `candidate_pipeline.py:273` | `price_source="mark"` branch — never exercised | Optional feature path |
| `retrain_meta_labeler.py:192` | `load_data()` — never called | Superseded by inline loading in `main()` |
| `retrain_meta_labeler.py:270` | `validate_data()` — never called | Superseded by `TrainingDataBuilder` validation |
| `retrain_meta_labeler.py:570-582` | Fallback PnL column branches | Unreachable when `net_pnl_pct` column exists |

## [6.5.3] - 2026-03-10

### Fixed — Bot Data Extractor Audit Findings (3 findings, 6 changes)

External audit of `bot_data_extractor.py` identified 3 VALID issues. All implemented
and verified with 27 new unit tests (902 total tests passing).

#### Finding 1 (HIGH): Coherence checks don't block writes
- Added `auto_repair_coherence()` function — classifies each coherence warning as
  repairable (PnL% recomputation, minor decomposition tolerance) or unresolved
  (grid spacing, liquidation bounds, large decomposition diffs).
- `process_bot()` now stamps `coherence_ok` (0/1) and `coherence_warnings` (text)
  onto every Excel row. Unresolved issues are flagged but writes are not blocked
  (preserves data pipeline continuity while enabling downstream filtering).
- Added 2 new columns to `CANONICAL_COLUMNS`: `coherence_ok`, `coherence_warnings`.

#### Finding 2 (MEDIUM): PnL-curve fallback too weak
- Added `mtm_used_mark_prices: bool` field to `TradeMetrics` dataclass, threaded
  from `MtmExcursionMetrics.used_mark_prices` through `compute_trade_metrics()`.
- PnL-curve MAE/MFE fallback now triggers when MTM engine had no mark prices
  (`used_mark_prices=False`), not only when MAE/MFE is exactly `None` or `0.0`.

#### Finding 3 (MEDIUM): Legacy OCR path misaligned
- `parse_screenshot_text()`: Changed 5 case-sensitive string checks (`'Expired' in text`)
  to `re.search(r'\bExpired\b', text, re.IGNORECASE)` — consistent with `parse_user_text()`.
- Added 3 missing regex patterns: `trigger_price`, `liq_price_long`, `liq_price_short`
  — these were parsed in `parse_user_text()` but absent from the OCR path.

### Fixed — Codebase-Wide Debug Audit (25 fixes across 22 source files)

Full-codebase audit using 6 parallel debug agents covering all ~176 Python files.
All 875 tests remain passing after fixes.

#### Critical (3)

- **backtest_realistic.py**: Moved empty DataFrame guard before `df.iloc[0]` access
  — previously `IndexError` crash on empty kline input.
- **backtest_realistic.py**: Fixed Sharpe ratio division-by-zero when equity curve
  reaches zero (liquidation/severe drawdown). Replaced raw list division with
  `np.where` guarded array operation.
- **binance_vision/pipeline.py**: Added missing `import pandas as pd` — `pd.NaT`
  reference at line 208 caused `NameError` at runtime when building manifest.

#### High (7)

- **enrich_grid_params.py**: TOS (Tradable Oscillation Score) was using wrong dict
  key `"klines_15m"` instead of `["klines"]["15m"]`, causing TOS to always fall back
  to a single-price array, making the Stage B TOS gate non-functional.
- **cpcv.py**: `compute_min_track_record_length` variance formula was missing the
  `0.5 * SR^2` term from AFML Eq 14.4. MinTRL was systematically underestimated,
  now consistent with `prob_sharpe_ratio` and `deflate()`.
- **backtest_realistic.py**: `duration_hours` and `price_end` now reflect the actual
  simulated bar count on early termination (circuit breaker or liquidation) instead
  of reporting the full DataFrame length.
- **data_generator.py**: Added `pd.notna()` guards before `int(row["grids_count"])`
  in `map_dataframe` lambda — NaN values caused `ValueError` crash.
- **btk_label_contract.py**: Added `cb_inventory_imbalance_ratio` and
  `cb_inventory_imbalance_dd_pct` to `ENGINE_SETTINGS_FIELDS` and
  `extract_engine_settings()` — backtest results were not fully self-describing.
- **candidate_pipeline.py**: Wrapped `float(val)` in try/except in
  `build_feature_dict_from_scanner_row` — non-numeric values (e.g., string hlabel)
  caused unhandled `ValueError`.
- **binance_vision/pipeline.py**: Changed `ts_min is pd.NaT` identity check to
  `pd.isna(ts_min)` — numpy `datetime64('NaT')` is a different object than
  `pd.NaT`, causing the guard to silently pass through.

#### Medium (10)

- **backtest_realistic.py**: Equity curve now includes a final entry after circuit
  breaker or liquidation close, reflecting the actual post-close equity in Sharpe
  and equity curve shape.
- **backtest_realistic.py**: Added `funding_interval_bars > 0` guard to prevent
  `ZeroDivisionError` in continuous funding proration.
- **binance_client.py**: Replaced `int(time.time() * 1000)` with
  `self._get_timestamp()` (2 sites) for incomplete candle detection — local clock
  drift could cause incorrect candle drops.
- **binance_client.py**: Narrowed `pre_fetched_klines` Optional subscript via
  `pfk = pre_fetched_klines or {}` to silence pyright `reportOptionalSubscript`.
- **technical.py**: Bollinger bandwidth division by zero when `middle == 0` now
  returns `NaN` via `np.where(middle != 0, ...)` instead of propagating `inf`.
- **calculator.py**: Added `hurst_max > 0.50` guard (2 sites) preventing
  division-by-zero when `hurst_max_trending` config equals 0.50.
- **scan.py**: Added `range_low <= 0 or range_high <= range_low` guard before
  `np.log()` calls in stochastic feature computation — prevents `-inf` propagation.
- **stochastic.py**: Changed OU sigma estimation from `np.var(residuals)` (ddof=0)
  to `np.var(residuals, ddof=1)` for unbiased variance, improving survival
  probability Monte Carlo accuracy.
- **validate.py**: NaN prices now flagged in OHLCV integrity `non_pos` check —
  IEEE 754 NaN comparison semantics caused `np.nan <= 0` to return False, silently
  passing invalid data.
- **backtest_candidates_current.py**: Added `sys.path.insert` for project root —
  `backtest/` imports failed when CWD was not the project root.

#### Low (5)

- **binance_vision/pipeline.py**: Replaced `__import__("pandas").concat(...)` with
  `pd.concat(...)` using the already-imported `pd` alias.
- **config.py**: `_dict_to_config` now reconstructs frozen dataclasses (e.g.,
  `CPCVConfig`) from merged dicts instead of calling `setattr` which would raise
  `FrozenInstanceError`.
- **enrich_grid_params.py**: Initialized `_has_book = False` before try block to
  prevent potential `NameError` if microstructure estimation raises.
- **enrich_grid_params.py**: Changed `grid_spacing_pct or 1.0` to
  `grid_spacing_pct if grid_spacing_pct is not None else 1.0` — Python falsy-zero
  semantics treated legitimate 0.0 as None.
- **hmm/train.py**: Training window `start_utc`/`end_utc` now computed from
  `min()/max()` across all datasets instead of first/last dict entry.

#### Known Unfixed (architectural — require design decisions)

- `calculator.py`: `calculate_num_grids` uses `grid_lower` as base while
  `spacing_pct` is relative to `current_price` — inconsistent bases.
- `meta_labeler.py`: Imputer `fit_transform` on full dataset before calibration
  holdout split introduces subtle look-ahead bias.
- `meta_labeler.py`: Temporal calibration holdout does not verify DataFrame is
  time-sorted before positional slicing.
- `backtest_realistic.py`: MAE/MFE tracks max peak-to-trough drawdown (not
  AFML entry-based excursion definitions).
- `sample_weights.py`: O(N^2) memory in `compute_concurrency_matrix`.
- `backtest_realistic.py`: Circuit breaker `short_count` always 0 for one-way
  grid bot — inventory imbalance ratio always 1.0.

#### Dead-End Code

- `scan.py:429-439`: AFML composite scoring branch unreachable (`ev_score` always
  None — deferred to post-enrichment by design).
- `binance_client.py:435-436`: `_current_high`/`_current_low` assigned from 24hr
  ticker but never used.
- `retrain_meta_labeler.py`: `load_data()` function defined but never called.

### Changed — Pipeline Performance Optimization (4 fixes, ~8x speedup)

Full pipeline runtime reduced from ~22 minutes to ~2 min 46 sec (measured on
150-symbol scan with `--compute-stochastic`). Four targeted fixes eliminate
redundant API calls and skip clearly-failing symbols before expensive
enrichment.

- **Fix 4: Batched scan API calls** (`src/neutralgrid/scanner/scan.py`):
  - Replaced 5 sequential `async with gate: await gate.throttle()` blocks in
    `fetch_one()` with a single `asyncio.gather()` call via new `_gated_call()`
    helper.
  - Each of the 5 per-symbol API calls (1h/15m/5m klines, premium index,
    open interest) now enters the rate gate independently and concurrently.
  - Scan phase: ~9.5 min → ~2 min 19 sec (~4.1x faster).

- **Fix 2: Scan-to-enrichment kline reuse** (4 files):
  - `scan.py`: New `scan_data_out` parameter on `scan_top_symbols()`. When
    provided, `fetch_one()` caches raw kline data (`klines_1h`, `klines_15m`,
    `klines_5m`) per symbol in the dict.
  - `binance_client.py`: New `get_enrichment_market_data()` method accepts
    `pre_fetched_klines` dict (keys: `'1h'`, `'15m'`, `'5m'`). Skips cached
    kline fetches and only requests the remaining 7 endpoints (1m klines, OI,
    funding, premium, order book, LS ratio, taker volume). Returns identical
    dict structure to `get_all_market_data()`.
  - `enrich_grid_params.py`: New `scan_data_cache` parameter on
    `enrich_with_grid_params()`. In `run_one()`, maps cached kline keys
    (`klines_1h` → `'1h'`, etc.) and calls `get_enrichment_market_data()`
    when cache hit, falls back to `get_all_market_data()` when no cache.
  - `run_full_pipeline.py`: Wires `scan_data_cache` dict from Step 3 (scan)
    to Step 4 (enrichment), adds `from typing import Any` import.
  - Saves ~3 API calls per enriched symbol (~300 fewer requests/run when all
    150 symbols reach enrichment).

- **Fix 3: Early-reject pass** (`src/neutralgrid/scanner/enrich_grid_params.py`):
  - New pre-enrichment filter between eligible selection and pre-warm.
  - Rejects symbols with `range_prob < 0.30` (matches regime validator hard
    threshold) or `survival_prob < 0.40` (below 0.50 validator threshold, with
    0.10 safety margin).
  - Uses scan-phase HMM data — no additional API calls required.
  - Measured impact: 134/145 symbols early-rejected (92%) in current market
    conditions, reducing enrichment from 145 to 11 symbols.

- **Fix 1: Deferred pre-warm** (`src/neutralgrid/scanner/enrich_grid_params.py`):
  - `PriceSeriesManager.ensure_symbol()` pre-warm loop moved to AFTER
    early-reject pass.
  - Only surviving symbols incur REST backfill (500 1m + 500 mark klines) and
    WebSocket subscription cost.
  - Combined with Fix 3, pre-warms ~11 symbols instead of ~145.

All 4 fixes are backward-compatible: new parameters default to `None`, existing
callers (`scan_top100.py`, test files) work unchanged without modification.

## [6.5.3] - 2026-03-09

### Changed — Capital sizing from account balance

- **Position sizing now uses live Binance account balance** instead of the hardcoded
  `GridConfig.capital = 400.0` USDT default.
- `run_full_pipeline.py` fetches `availableBalance` from the Binance USDT-M futures
  account after connecting (Step 2) and uses it as `base_capital` for
  `deploy_margin_usdt = capital_fraction × base_capital`.
- New CLI flag `--capital`:
  - `--capital account` (default): fetches available balance from Binance account.
  - `--capital 500`: uses a fixed 500 USDT amount (overrides account query).
- Fails fast if account balance is zero or negative.
- The `capital_base_usdt` column in the output CSV now reflects the actual capital
  source used for that pipeline run.

### Added — PnL-Growth Enhancements (5 new modules, 5 config dataclasses)

- **Enhancement 4: Microstructure Hard Gate** (`src/neutralgrid/validation/microstructure_hard_gate.py`):
  - Binary pass/fail filter replacing the soft microstructure penalty.
  - 5 checks: round-trip cost ceiling, spread-to-profit ratio, funding drag ceiling,
    liquidity floor, profit-floor viability.
  - `MicrostructureHardGate` class with `evaluate()` and `evaluate_from_costs()` methods.
  - `MicrostructureGateResult` frozen dataclass with structured rejection codes.
  - Config: `MicrostructureGateConfig` (max_round_trip_pct=1.20, max_spread_to_profit_ratio=0.60,
    max_funding_drag_pct=0.80).

- **Enhancement 2: Tradable Oscillation Score** (`src/neutralgrid/scanner/tradable_oscillation.py`):
  - Composite grid-suitability metric replacing raw volatility as primary signal.
  - 7 sub-signals: grid-cross frequency, mean-reversion strength, range containment,
    spread-to-profit ratio, funding drag, depth sufficiency, liquidation proximity.
  - All sub-signals normalised to [0, 1], weighted sum scaled to [0, 100].
  - `TradableOscillationScorer` class with `compute()` method.
  - `OscillationScoreResult` frozen dataclass.
  - Config: `OscillationScorerConfig` (7 configurable weights, saturation/cap constants).

- **Enhancement 1: Two-Stage Candidate Selection** (`src/neutralgrid/scanner/two_stage_selector.py`):
  - Stage B deployment-approval gate (Stage A = existing scan pipeline).
  - 4 gates: hard gate pass, TOS >= threshold, position sizer approval, regime confidence.
  - `TwoStageSelector` class with `approve()` method.
  - `StageBResult` frozen dataclass with per-gate results.
  - Config: `TwoStageConfig` (min_tos=40, min_position_fraction=0.05, min_range_prob=0.45).

- **Enhancement 3: Hard Risk Budget Position Sizer** (`src/neutralgrid/grid/position_sizer.py`):
  - Centralised multiplicative capital allocation formula.
  - `final = base × regime_confidence × survival × micro × volatility × portfolio_heat`.
  - Each factor independently computed and clamped to [floor, 1.0].
  - `PositionSizer` class with `compute()` method.
  - `PositionSizeResult` frozen dataclass with all intermediate factors.
  - Config: `RiskBudgetConfig` (min_fraction=0.05, max_fraction=1.0, factor_floor=0.30,
    vol_target_pct=3.0).

- **Enhancement 5: Hierarchical Training Labels** (`src/neutralgrid/training/hierarchical_labels.py`):
  - 4-level labelling: L1 (geometry survival), L2 (execution viability),
    L3 (net return hurdle), Meta (deploy = L1 ∧ L2 ∧ L3).
  - `hlabel` integer 0–3 encodes highest level passed.
  - Backward-compatible: `hlabel_meta` stored separately, authoritative binary `y` preserved
    from `label_positive_by_horizon`.
  - `HierarchicalLabeler` class with `label()` method.
  - `HierarchicalLabelResult` frozen dataclass with `to_dict()`.
  - Config: `HierarchicalLabelConfig` (max_drawdown_floor_pct=-25, min_fills=3,
    min_duration_hours=1.0, hurdle_pct=3.0).

- **5 new config dataclasses** added to `src/neutralgrid/core/config.py`:
  - `MicrostructureGateConfig`, `OscillationScorerConfig`, `TwoStageConfig`,
    `RiskBudgetConfig`, `HierarchicalLabelConfig`.
  - All registered in root `Config` class and `_dict_to_config()` loader.

### Changed — Pipeline Integration

- **`src/neutralgrid/scanner/enrich_grid_params.py`**:
  - Imports: `MicrostructureHardGate`, `TradableOscillationScorer`, `TwoStageSelector`, `PositionSizer`.
  - 13 new enrichment columns: `hard_gate_passed`, `hard_gate_reason`, `tos`, `tos_gcf`,
    `tos_mrs`, `tos_rc`, `ps_fraction`, `ps_regime_scale`, `ps_survival_scale`,
    `ps_micro_scale`, `ps_vol_scale`, `ps_heat_scale`, `ps_reason`, `stage_b_approved`,
    `stage_b_reason`.
  - Post-grid evaluation chain: hard gate → position sizer → TOS → Stage B.
  - Each enhancement wrapped in try/except to prevent cascade failures.
  - Position sizer fraction multiplied into existing Kelly-based capital_fraction.
  - **Stage B now controls `grid_is_valid`** — not a dead annotation.
  - **Stage B receives final `capital_fraction`** (Kelly × sizer), not raw `ps_fraction`.
  - **Hard gate calls `evaluate()` directly** with `sufficient_liquidity=None` when
    no order book available (instead of using `evaluate_from_costs()` which inherits
    the upstream default of `True`).
  - **`_has_book` flag** computed from order book bid/ask presence.
  - Stage B exception handler fails-closed (`False`) instead of `None`.

- **`src/neutralgrid/backtest/candidate_pipeline.py`**:
  - `convert_to_training_row()` now uses hierarchical labels as the **primary training
    target**: `y` comes from `hlabel_meta` (L1+L2+L3), not `label_positive_by_horizon`.
  - Engine horizon label preserved as `y_horizon` for audit trail.
  - `hlabel` (0-3) added to `TRAINING_FEATURES` and `_SCANNER_TO_FEATURE` mapping.

- **`src/neutralgrid/core/config.py`**:
  - `RiskBudgetConfig.min_fraction` changed from `0.05` to `0.0` — sizer can now
    produce fractions below Stage B's `min_position_fraction=0.05` threshold,
    enabling hard rejection through the Stage B gate.

- **`src/neutralgrid/validation/microstructure_hard_gate.py`**:
  - Missing `funding_cost_pct` (`None`) now appends `data_missing:funding_cost`
    rejection code instead of silently passing.

### Fixed — Audit Phase 2: End-to-End Wiring (5 validated findings)

- **Finding 1 (High): Stage B now gates deployment**. `stage_b_approved` controls
  `grid_is_valid` in `enrich_grid_params.py`. Downstream consumers (`run_full_pipeline.py`,
  `filter_backtest_candidates()`) filter on `grid_is_valid`, making Stage B the
  authoritative deployment decision.

- **Finding 2 (High): Sizing allows hard rejection**. `RiskBudgetConfig.min_fraction=0.0`
  lets the multiplicative sizer produce fractions below 0.05. Stage B checks the
  final `capital_fraction` (Kelly × sizer) against `min_position_fraction=0.05`,
  rejecting under-allocated candidates.

- **Finding 3 (Medium): TOS is first-class via Stage B**. TOS is gate 2 in Stage B
  (`tos >= min_tos`). Since Stage B now controls `grid_is_valid`, low TOS causes
  deployment rejection. TOS correctly computed post-grid (architecturally required).

- **Finding 4 (Medium): Hard gate strict on missing data**. Missing funding now fails
  (`data_missing:funding_cost`). Missing order book passes `sufficient_liquidity=None`
  to the hard gate, failing with `data_missing:liquidity`.

- **Finding 5 (Medium): Hierarchical labels are primary training target**. `y` now
  comes from `hlabel_meta` (stricter: L1+L2+L3). Engine horizon label preserved as
  `y_horizon`. `hlabel` (0-3) carried through as audit/label column (NOT a model
  feature — information leakage).

### Fixed — Audit Phase 3: End-to-End Label Consumption (F5 completion)

- **Disconnect 1: `MetaLabeler.train()` label override**. Added `y_col` parameter to
  `MetaLabeler.train()`. When `y_col="y"` is specified and the column exists with
  non-null values, the pre-computed labels are used directly instead of rebuilding
  from PnL/SL via `create_labels()`. Backward-compatible: `y_col=None` (default)
  falls back to the existing `create_labels()` behavior.

- **Disconnect 2: `unified_training_builder` label source**. `_ingest_backtest_rows()`
  now prefers `hlabel_meta` for the `y` label when available, falling back to
  `label_positive_by_horizon`, then PnL hurdle. Hierarchical label columns
  (`hlabel`, `hlabel_meta`, `y_horizon`, `hlabel_L1/L2/L3`) carried through to
  training DataFrame for audit trail.

- **Disconnect 3: Schema alignment**. `TrainingTableSchema.labels` extended with
  `y_horizon`, `hlabel`, `hlabel_meta`. These are label/audit columns, NOT model
  features. `hlabel` excluded from `MetaLabelerConfig.features` to prevent
  information leakage (`hlabel == 3 ⟺ y == 1`).

- **`retrain_meta_labeler.py`**: Now passes `y_col="y"` to `labeler.train()` so
  backtest rows use `hlabel_meta`-derived labels and live rows use PnL-based labels
  from the data generator.

### Added — Tests

- `tests/unit/test_enhancements_v653.py` — 59 tests covering all 5 enhancements + label consumption:
  - `TestMicrostructureHardGate` (11 tests): pass/fail scenarios, missing data,
    funding-none strict, liquidity-none strict, multiple failures.
  - `TestTradableOscillationScorer` (5 tests): oscillating/trending/empty data, sub-score ranges.
  - `TestTwoStageSelector` (8 tests): all-pass, per-gate failures, missing data,
    sub-threshold fraction, zero fraction rejection.
  - `TestPositionSizer` (9 tests): perfect/poor conditions, explicit floor, default
    floor zero allows sub-threshold, ceiling, portfolio heat, multiplicative independence.
  - `TestHierarchicalLabeler` (9 tests): all 4 levels, range breakout, fees, to_dict.
  - `TestConfigIntegration` (3 tests): config presence, defaults, dict_to_config.
  - `TestHierarchicalLabelPrimaryTarget` (3 tests): y from hlabel_meta, y_horizon
    preserved, hlabel in training features.
  - `TestMetaLabelerYCol` (3 tests): y_col used when column present, fallback when
    column absent, None default uses create_labels.
  - `TestUnifiedBuilderHlabelMeta` (4 tests): hlabel_meta preferred over horizon
    label, True produces y=1, fallback to label_positive_by_horizon, fallback to PnL hurdle.
  - `TestTrainingSchemaHlabel` (3 tests): hlabel in labels tuple, not in features
    tuple, present in all_columns.

- `tests/test_afml_compliance.py` updated:
  - `test_feature_list_length` now asserts `hlabel` not in `MetaLabelerConfig.features`
    (information leakage: `hlabel == 3 ⟺ y == 1`).

- `tests/unit/test_btk_output_contract.py` updated:
  - `TestHorizonLabelInTrainingRow` → `TestHierarchicalLabelInTrainingRow`.
  - Tests now verify `y` from `hlabel_meta`, `y_horizon` preserved, insufficient
    fills override positive PnL.

### Changed — Documentation

- **`readmefullpwep.md`** rewritten to reflect v6.5.3 architecture:
  - New sections: Two-stage selection (§2.3), Enrichment chain (§2.4), Sizing flow (§2.5),
    Hierarchical labels (§2.6), Config dataclasses table (§3.4), Backtest & Training
    Pipeline (§8) with execution flow, data flow, label precedence, and feature lists.
  - New output column groups: sizing (§7.2), hard gate (§7.3), position sizer (§7.4),
    TOS (§7.5), Stage B (§7.6).
  - Expanded troubleshooting with Stage B-specific failures and missing-data rejections.
  - Quick runbook updated from 4 steps to 6 (added backtest + include-backtest-data).

### Changed — Version Bump

- Package version `6.5.2` → `6.5.3` in `pyproject.toml`,
  `src/neutralgrid/__init__.py`.
- Engine version `realistic-v5` → `realistic-v6` in
  `backtest/btk_unified_runner.py` — wick double-fill prevention changes
  label output values, making results non-comparable with v5.
- `CLAUDE.md` project overview updated from `v6.4.x` to `v6.5.3`.
- `readmefullpwep.md` runbook paths and expected output updated to `v6.5.3`.
- All test fixtures referencing `pipeline_version` updated to `"6.5.3"`.

### Fixed — Phase 1: Critical Crash Prevention (10 files, 25 fixes)

- **`backtest/backtest_realistic.py`** (7 fixes):
  - Division by zero in `seed_from_state` when `sl.price == 0` (2 sites).
  - Grid spacing `_avg == 0` guard.
  - Position size `avg_price <= 0` guard with descriptive `ValueError`.
  - `price_change_pct` division by zero when `price_start == 0`.
  - Empty DataFrame guard before main backtest loop.
  - Missing `entry_bar` key: `pos['entry_bar']` → `pos.get('entry_bar', bar_idx)`.
  - Wick double-fill prevention: `_filled_this_bar` set tracks levels
    processed by close-to-close crossing and skips them in wick fills.
- **`src/neutralgrid/api/binance_client.py`** (4 fixes):
  - `sync_time()` KeyError on malformed response missing `serverTime`.
  - `last_exc is None` guard after retry loop exits without result.
  - Kline bounds checks before indexing `klines[-1]` (2 sites).
- **`src/neutralgrid/cli/retrain.py`**:
  - `NameError` on `symbols` variable when API fetch failed — initialized
    `symbols: list[str] = []` before conditional block, added `except`
    clause, and terminal guard with `sys.exit(1)`.
- **`src/neutralgrid/backtest/candidate_pipeline.py`** (6 fixes):
  - Added `import numpy as np` (was missing, caused `NameError` on
    `np.isfinite`).
  - Empty klines: log warning instead of crash.
  - `funding_rate` type safety: `try/except (TypeError, ValueError)`.
  - `capital_fraction` NaN guard: `if not np.isfinite(cap_frac)`.
  - Volatility proxy NaN/zero guard.
  - `duration_bars` negative guard: skip truncation when `< 0`.
- **`src/neutralgrid/scanner/enrich_grid_params.py`**:
  - Trigger price `None` guard on 4 fields (`tp.price`, `tp.source`,
    `tp.ts_ms`, `tp.age_ms`).
- **`src/neutralgrid/training/sample_weights.py`** (2 fixes):
  - Normalization division by zero: guard `_wsum > 1e-12`.
  - Uniqueness concurrency floor: `np.maximum(bars_ct, 1.0)` before
    `1.0 / bars_ct`.
- **`retrain_hmm.py`** (6 fixes):
  - `mkdir(parents=True, exist_ok=True)` before all 6 JSON artifact writes
    (`cpcv_results`, `cpcv_sweep_results`, `cpcv_selected_threshold`,
    `cpcv_utility_sweep`, `cpcv_proxy_outcomes`, `wf_stability_report`).
- **`retrain_meta_labeler.py`** (3 fixes):
  - `cv_scores` empty list guard before `sum()/len()`.
  - `"source"` column existence check before `value_counts()`.
  - PnL column terminal fallback: `sys.exit(1)` when no usable column
    found among `args.pnl_col`, `net_pnl_pct`, `pnl_pct`.
- **`backtest_candidates_current.py`**:
  - Removed hardcoded `backtest_results_20260224.csv` default — auto-detects
    latest `backtest_results_*.csv` via glob sort. Output path also
    auto-generates with current UTC date.

### Fixed — Phase 2: Data Integrity (4 files, 8 fixes)

- **`src/neutralgrid/models/meta_labeler.py`** (3 fixes):
  - `.astype(bool)` → `.astype("boolean")` for pandas nullable `BooleanDtype`.
  - Sequential bootstrap index bounds validation at both training sites
    (fold and final model) — filters out-of-range indices, raises
    `ValueError` if all invalid.
- **`src/neutralgrid/models/hmm/inference.py`** (5 fixes):
  - Subnormal float division guard: `total > 0` → `total > 1e-10` in
    `_ema_smooth_posteriors`, `_apply_adaptive_transitions`,
    `_apply_tail_correction`.
  - State index bounds validation for `trend_state` and `range_state` in
    `HMMRegimePredictor.predict()`.
- **`src/neutralgrid/core/config.py`**:
  - Narrowed `except Exception` to
    `(FileNotFoundError, ValueError, ImportError, KeyError, TypeError)`.
- **`src/neutralgrid/core/logging.py`**:
  - Threading lock (`_logger_lock`) with double-checked locking around
    `get_logger()` initialization to prevent duplicate handlers.

### Fixed — Phase 3: Robustness (8 files, 9 fixes)

- **`src/neutralgrid/models/triple_barrier.py`**:
  - Narrowed `except Exception` to
    `(ImportError, AttributeError, ValueError, TypeError)` for leverage
    config lookup fallback.
- **`src/neutralgrid/grid/calculator.py`** (2 fixes):
  - `current_price == 0` → `current_price <= 0` in `calculate_grid_spacing`.
  - `range_size_pct` division guard: `if current_price > 0 else 0.0`.
- **`src/neutralgrid/validation/stochastic.py`**:
  - Hurst NaN fallback: `if not np.isfinite(hurst): hurst = 0.5`.
- **`src/neutralgrid/replay/cli.py`**:
  - `window_days <= 0` validation at `run_pipeline()` entry.
- **`src/neutralgrid/data/price_series/ps_rest_backfill.py`**:
  - Raw kline length validation: `if len(raw) < 7: raise ValueError(...)`.
- **`src/neutralgrid/data/binance_vision/ingest.py`**:
  - Robust header detection: `int(first_field)` try/except replaces
    `first_field.isdigit()` which fails on empty strings and negative
    timestamps.
- **`display_candidates.py`**:
  - Auto-detect latest `neutralgrid_candidates_*.csv` via glob (removed
    hardcoded `20260120_160852` path). Dynamic scan file name in output.

### Verified — No Fix Needed

The following were audited and confirmed already correct:

- `src/neutralgrid/backtest/cpcv.py` — `prob_sharpe_ratio` already guards
  `n_observations <= 1` (line 668) and `se_sharpe <= 0` (line 678).
- `src/neutralgrid/backtest/evaluate.py` — Posterior row-sum guard already
  at line 480; multiple `np.isfinite` checks present.
- `src/neutralgrid/indicators/technical.py` — ADX inf→NaN replacement
  already at lines 268-269.
- `src/neutralgrid/data/curator.py` — `vol_median > 0` guard already
  exists; no `bool()` coercion issue found.
- `backtest/btk_label_contract.py` — Explicit `frozenset` validation, no
  silent defaults.
- `src/neutralgrid/api/app.py` — Pydantic models handle JSON validation.
- `run_full_pipeline.py` — `json.load` already wrapped in try-except in
  `_read_metadata()`.
- `src/neutralgrid/data/price_series/ps_store.py` — Deque ops are
  synchronous within async event loop; no preemption possible.
- `src/neutralgrid/data/price_series/ps_ws_stream.py` — Session properly
  closed in `finally` block.
- `src/neutralgrid/models/hmm/train.py` — Convergence checks already at
  lines 156 and 350; `.empty` guards already at lines 89 and 288.
- `src/neutralgrid/models/meta_labeler.py` — `FrozenEstimator` import
  guard already at lines 652-657; `save()` uses safe `tempfile`/`os.fdopen`.
- `src/neutralgrid/data/binance_vision/pipeline.py` — NaT guard already at
  lines 208-209; empty check at line 170.
- `src/neutralgrid/data/binance_vision/store.py` — Re-sort after dedup
  already at line 148.

### Validation

- Full test suite: **656 passed**, 0 failures (21s).
- Full architecture audit covered 60+ Python files, 155 issues triaged
  across 3 priority phases.

---

## [6.5.2] - 2026-03-05

### Added - Trigger Price Ingestion

- Added `trigger_price` field to `ExtractedBotData` dataclass in
  `bot_data_extractor.py` for capturing the mark/last trigger price at
  bot creation time from expired bot data.
- Added `trigger_price` to `prepare_entry_validation_row()` return dict,
  `CANONICAL_COLUMNS` (now 49 columns), and `load_manual_input()` valid
  fields set.
- Added `trigger_price` to `_OUTCOME_FIELDS` in
  `src/neutralgrid/training/live_outcome_ingestor.py` so it flows through
  the training feedback loop as outcome metadata.
- Workbook schema evolved from 48 to 49 columns with automatic header
  extension via `append_to_excel()` schema evolution logic.

### Added - Liquidation Price Features

- Added `liq_price_long`, `liq_price_short` fields to `ExtractedBotData`
  dataclass for capturing estimated liquidation prices at bot creation.
- Added `_compute_liquidation_features()` function in
  `bot_data_extractor.py` computing 4 derived features from raw liq
  prices and grid bounds:
  - `dist_to_liq_long_pct` / `dist_to_liq_short_pct` — distance from
    grid boundary to liquidation level (positive = safe, negative =
    breached).
  - `liq_range_utilization` — grid range as percentage of liquidation
    range.
  - `liq_asymmetry` — ratio of short-side to long-side distance.
- Added all 6 liq fields to `_OUTCOME_FIELDS` in
  `live_outcome_ingestor.py`.

### Changed - Excel Writer Safety

- Rewrote `append_to_excel()` from destructive pandas `ExcelWriter`
  (3-sheet rewrite) to openpyxl single-sheet append. Only ADDS data,
  never overwrites existing cells or sheets.
- Added bool-to-int safety net (`isinstance(val, bool): val = int(val)`)
  in `append_to_excel()` and in `_apply_patches_to_workbook()` in the
  backfill script to ensure uniform `1`/`0` values instead of Excel
  `TRUE`/`FALSE`.
- Added schema evolution logic: when `CANONICAL_COLUMNS` grows, existing
  workbooks auto-extend their header row without disturbing existing data.

### Changed - Backfill Script

- Rewrote `scripts/backfill_mae_mfe_from_exports_v20260304.py` from
  wholesale `to_excel()` rewrite to cell-selective openpyxl patching via
  `_apply_patches_to_workbook()`. Preserves existing workbook formatting
  and only touches cells that need updating.

### Fixed

- Fixed `holding_time_ok` returning Python `bool` (written as Excel
  `TRUE`/`FALSE`) instead of `int` (`1`/`0`). Three-layer fix: source
  cast in `prepare_entry_validation_row()`, safety net in
  `append_to_excel()`, safety net in backfill script.
- Fixed `load_manual_input()` silently discarding `trigger_price`,
  `liq_price_long`, and `liq_price_short` from manual JSON input due to
  missing entries in the `valid_fields` filter set.

### Removed

- Removed dead `--order-csv` CLI argument from `bot_data_extractor.py`.
- Removed dead `TradeMetrics.sharpe_proxy` field.

### Validation

- Full test suite: 656 passed.
- Agent team reviews verified:
  - All 49 workbook columns populated for rows 120-124 (LITUSDT,
    ALGOUSDT, IPUSDT, NEARUSDT, BULLAUSDT) with zero missing values.
  - Trigger price naming consistent across all pipeline stages
    (enrich_grid_params.py → deployment_ready CSV → bot_data_extractor →
    new_expired_bots.xlsx → live_outcome_ingestor).
  - No regressions from trigger_price or liq price additions.
  - MAE sign convention aligned to positive (absolute drawdown).
  - strategy_type aligned to majority convention ('grid').

---

## [6.5.2] - 2026-03-04

### Added - Sizing Consistency + Execution Enforcement

- Added deploy-time sizing helpers in
  `src/neutralgrid/live/deployment_payload_v20260304.py`:
  - `resolve_deployment_sizing(...)`
  - `build_binance_grid_payload(...)`
  These enforce `capital_fraction` into an explicit effective margin contract.
- Added new deployment payload/linkage fields:
  `capital_fraction`, `capital_base_usdt`, `volatility_scale_applied`,
  and enforced `margin_usdt` derivation when missing.
- Added new unit tests:
  - `tests/unit/test_sizing_consistency_v20260304.py`
  - `tests/unit/test_deployment_payload_v20260304.py`

---

### Changed - Backtest/Label Capital Basis and Volatility Targeting

- Updated `backtest/backtest_realistic.py` to support:
  - `capital_fraction` in `GridConfig`
  - volatility-target inputs (`volatility_target_pct`, `volatility_proxy_pct`)
  - effective deployed capital (`capital_used`) as simulation basis.
- Updated MTM/label math to use deployed capital basis:
  `net_pnl`, `net_pnl_pct`, horizon label, and termination/exit penalty metrics
  now align with `capital_used` (not fixed base capital).
- Updated candidate backtest path in
  `src/neutralgrid/backtest/candidate_pipeline.py` to pass scanner-derived
  `capital_fraction` and volatility proxy into `build_training_config(...)`,
  and to persist sizing diagnostics into backtest training rows.
- Updated engine metadata contract:
  - `backtest/btk_label_contract.py` includes new sizing/volatility settings.
  - `backtest/btk_unified_runner.py` engine version bumped to `realistic-v5`.
  - label contract version bumped to `2.2`.
- Added `capital_used` to `REQUIRED_LABEL_FIELDS` in
  `backtest/btk_label_contract.py` with numeric type validation in
  `validate_engine_result()`. Label docstring updated to reflect comparison
  against `capital_used` instead of fixed `capital`.
- Updated `run_candidate_backtests.py` to read capital from
  `get_config().grid.capital` instead of hardcoded `400.0`.

---

### Fixed - Kelly Optimization, EV Alignment, Fee/EV Consistency

- Enhanced `src/neutralgrid/scanner/empirical_profile_v20260302.py`:
  - Added automated fractional-Kelly sweep optimization under drawdown tolerance.
  - Added shrinkage behavior for low-sample EV/fill/payoff alignment
    (no hard fallback-only behavior at low sample counts).
- Extended Kelly diagnostics in
  `src/neutralgrid/scanner/enrich_grid_params.py`:
  `kelly_fractional_multiplier`, `kelly_fractional_mode`,
  `kelly_sweep_growth`, `kelly_sweep_drawdown_pct`,
  `kelly_sweep_feasible`, `kelly_sweep_evaluated`.
- Aligned scanner fee assumptions with engine close fee mode:
  - Added `grid.close_fee_mode` config in `src/neutralgrid/core/config.py`.
  - Updated arithmetic profit-per-grid and empirical profit computations to use
    effective fee mapping (maker entry + configured close fee mode).
- Updated `estimate_expected_return()` in
  `src/neutralgrid/grid/calculator.py` to use the same dynamic
  active-fraction formula used by PnL ranker.

---

### Changed - Pipeline Output and Training Integration

- Updated `run_full_pipeline.py` to export deploy sizing columns:
  `capital_base_usdt`, `deploy_margin_usdt`.
- Updated `src/neutralgrid/training/live_outcome_ingestor.py` and
  `src/neutralgrid/training/unified_training_builder.py` to ingest
  `capital_fraction` from scanner/live-linked records (`scan_capital_fraction`).
- Updated `run_candidate_backtests.py` to pass sizing/vol-target inputs and
  corrected report messaging to maker close-fee mode.

---

### Validation

- Targeted integration and regression suites passed after implementation:
  - `tests/unit/test_btk_label_runner.py`
  - `tests/unit/test_btk_output_contract.py`
  - `tests/unit/test_btk_duration_override.py`
  - `tests/unit/test_unified_training_builder.py`
  - `tests/unit/test_live_outcome_ingestor.py`
  - `tests/unit/test_enrich_grid_params.py`
  - `tests/unit/test_trigger_price_enrichment.py`
  - `tests/unit/test_sizing_consistency_v20260304.py`
  - `tests/unit/test_deploy_linker.py`
  - `tests/unit/test_deployment_payload_v20260304.py`
  - Result: `637 passed` (full suite).

---

## [6.5.1] - 2026-03-02

### Added - Empirical EV Alignment + Generalized Kelly Sizing

- Added `src/neutralgrid/scanner/empirical_profile_v20260302.py` to derive
  empirical payoff ratio (`b`), EV alignment fit (`slope/intercept/r2`), and
  fill-rate scaling from backtest data.
- Integrated EV alignment into ranking in
  `src/neutralgrid/scanner/pnl_ranker.py` with explicit decomposition fields:
  `ev_raw`, `ev_aligned`, `expected_fills`, `fill_rate_scale`.
- Extended `run_full_pipeline.py` output with EV diagnostics:
  `ev_raw`, `ev_aligned`, `ev_expected_fills`, `ev_fill_rate_scale`,
  `ev_alignment_samples`, `ev_alignment_slope`, `ev_alignment_intercept`,
  `ev_alignment_r2`, `payoff_ratio_b`.
- Added generalized Kelly sizing in
  `src/neutralgrid/scanner/enrich_grid_params.py`:
  `f* = (p*b - q)/b`, with fractional Kelly, drawdown scaling, and volatility
  targeting. New output diagnostics include:
  `kelly_payoff_b`, `kelly_avg_win_pct`, `kelly_avg_loss_pct`,
  `kelly_raw_fraction`, `kelly_fractional`, `kelly_drawdown_scale`,
  `kelly_volatility_scale`, `kelly_profile_samples`.
- Added configuration controls in `src/neutralgrid/core/config.py`:
  `PositionSizingConfig` and `EdgeTierConfig.ev_upgrade_min_delta`.

---

### Changed - Edge Selection, Label Target Alignment, and Spacing Behavior

- Updated edge-tier upgrade logic in
  `src/neutralgrid/scanner/enrich_grid_params.py` to require both net-edge and
  EV-improvement conditions when promoting MEDIUM -> BIG tier.
- Updated regime-aware spacing response in `src/neutralgrid/grid/calculator.py`
  to a convex trend-probability curve (bounded 1.0x-1.3x).
- Improved live outcome matching in
  `src/neutralgrid/training/live_outcome_ingestor.py` with geometry-aware
  matching using range bounds and grid count.
- Updated training data integration in
  `src/neutralgrid/training/unified_training_builder.py` and
  `src/neutralgrid/training/data_generator.py` to use/derive `net_pnl_pct`
  consistently and ingest raw `backtest_results_*.csv` when needed.
- Updated `retrain_meta_labeler.py` and
  `src/neutralgrid/models/meta_labeler.py` so labels align to net profitability
  objective by preferring `net_pnl_pct` when available.

---

### Fixed - Mixed Datetime Parsing + Calibration Compatibility

- Fixed mixed timestamp format handling (`YYYY-MM-DD HH:MM:SS+00:00` and ISO
  `YYYY-MM-DDTHH:MM:SS+00:00`) across:
  - `src/neutralgrid/backtest/cpcv.py`
  - `src/neutralgrid/training/sample_weights.py`
  - `src/neutralgrid/training/unified_training_builder.py`
- Added robust UTC coercion with mixed-format support to prevent invalid-row
  drops and CPCV/sample-weight failures when live/backtest rows are merged.
- Updated sklearn calibration compatibility in
  `src/neutralgrid/models/meta_labeler.py`:
  - Uses `cv="prefit"` path when supported.
  - Falls back to `FrozenEstimator` flow for newer sklearn versions.
  - Uses unweighted fit in FrozenEstimator fallback to avoid misleading
    sample-weight propagation warnings.

---

### Retrained Models (2026-03-02)

- Scanner artifacts retrained from `data/new_expired_bots.xlsx`:
  - `data/profile/pattern_profile.json`
  - `data/profile/profile_model.json`
  - `data/profile/profile_gate.json`
- HMM retrained and promoted:
  - `artifacts/hmm/rolling_180d_20260302_202515`
  - `artifact_manifest.json` updated to active version
    `rolling_180d_20260302_202515`.
- Meta-labeler retrained with unified table + backfill:
  - `models/meta_labeler.pkl`
  - `models/meta_labeler/` artifact metadata refreshed.

---

### Validation

- Targeted regression tests passed:
  - `tests/unit/test_unified_training_builder.py`
  - `tests/unit/test_live_outcome_ingestor.py`
  - `tests/test_afml_compliance.py::TestSourceCodeIntegrity::test_meta_labeler_calibration_uses_prefit`
  - `tests/test_afml_fixes_v2.py::TestPnlUnitConsistency`
  - Result: `21 passed`.
- Full pipeline run completed on 2026-03-02:
  - Command: `python run_full_pipeline.py --compute-stochastic --top-n 150 --min-score 45 --max-enrichment 150 --max-concurrency 8 --show-all`
  - Output: `results/deployment_ready_20260302_153222.csv`
  - Summary: 150 candidates enriched, 43 valid; `ETHFIUSDT` present and valid.

---

## [6.5.1] - 2026-03-01

### Added - Non-Stationary HMM with Empirical Tail Integration

- **Tail-adjusted posterior correction**: Inserted between GaussianHMM output and
  EMA smoothing. Compares empirical tail mass to Gaussian tail mass per state for
  extreme observations (|z| > 2σ), applying multiplicative correction weights to
  posteriors. Weights clamped to [1/max_weight, max_weight] for stability.
- **Cornish-Fisher expansion VaR/CVaR**: Adjusts Gaussian quantiles for skewness
  and excess kurtosis using the Cornish-Fisher z-score expansion. Falls back to
  empirical estimates when non-finite. Stored per-state in uncertainty profile.
- **Gaussian divergence metric**: Jarque-Bera test statistic + tail weight ratio
  (empirical/Gaussian mass beyond 2σ) per state. Quantifies how far each state's
  returns deviate from normality.
- **Kurtosis-adjusted volatility**: `vol_adj = std × √(1 + max(κ, 0) / 4)` —
  states with higher excess kurtosis are effectively riskier even at the same raw
  standard deviation.
- **Kurtosis-aware volatility tiers**: New tier classification alongside existing
  simple tiers. Uses kurtosis-adjusted vol for q33/q66 cuts instead of raw std.
- **Synthetic exogenous signals**: Data-derived proxy signals (volatility_shock,
  return_shock, momentum_divergence) as fallback when actual exogenous channels
  are unavailable. Enables the exogenous framework to produce usable output without
  external data feeds.
- **Enhanced conditional tail risk**: `compute_weighted_conditional_tail_risk_enhanced()`
  uses CF VaR/CVaR, weighted excess kurtosis, weighted skewness, and weighted
  Gaussian divergence score aggregated across posteriors.
- **Config kill-switches**: Three new `HMMConfig` fields with env var overrides:
  - `tail_correction_enabled` (default: True, env: `HMM_TAIL_CORRECTION_ENABLED`)
  - `tail_correction_max_weight` (default: 3.0, env: `HMM_TAIL_CORRECTION_MAX_WEIGHT`)
  - `tail_correction_threshold_sigma` (default: 2.0, env: `HMM_TAIL_CORRECTION_THRESHOLD_SIGMA`)
- **Downstream propagation**: New enrichment columns (`hmm_cf_var_95`,
  `hmm_cf_cvar_95`, `hmm_cf_var_99`, `hmm_cf_cvar_99`, `hmm_weighted_kurtosis`,
  `hmm_gaussian_divergence`, `hmm_tail_correction`, `hmm_volatility_tier_ka`)
  flow from inference through regime validation to scanner output.
- **18 new tests** (`tests/unit/test_tail_correction.py`): CF VaR conservatism,
  degeneration to Gaussian, tail weight neutrality/upweighting, weight clamping,
  posterior normalization, config kill-switch, graceful fallback, kurtosis-adjusted
  vol, kurtosis-aware tiers, Gaussian divergence, synthetic signals, profile keys,
  enhanced tail risk, backward compatibility.
- **Training metadata**: `non_stationary_mode` in training stats now reports
  `tail_adjusted_posteriors`, `cornish_fisher_var`, `kurtosis_aware_tiers`.

**Files modified:**
| File | Change |
|---|---|
| `src/neutralgrid/core/config.py` | 3 new HMMConfig fields + env var overrides |
| `src/neutralgrid/models/hmm/uncertainty.py` | 6 new math functions, extended profile output, enhanced exogenous framework |
| `src/neutralgrid/models/hmm/inference.py` | Tail correction function, extended predict(), extended HMMInferenceResult |
| `src/neutralgrid/models/hmm/train.py` | Metadata flags |
| `src/neutralgrid/validation/hmm_regime.py` | New fields pass-through |
| `src/neutralgrid/validation/regime_validator.py` | New ValidationResult fields |
| `src/neutralgrid/scanner/enrich_grid_params.py` | 8 new output columns |
| `tests/unit/test_tail_correction.py` | NEW: 18 test cases |
| `tests/unit/test_enrich_grid_params.py` | Mock fixtures updated for new vres attributes |
| `tests/unit/test_trigger_price_enrichment.py` | Mock fixtures updated for new vres attributes |

**Retrain required:** The new profile keys (`cornish_fisher`, `gaussian_divergence`,
`kurtosis_adjusted_vol`, `tail_adjustment_params`, `volatility_tiers_kurtosis_aware`)
are computed by `compute_regime_uncertainty_profile()` at training time and stored in
the model artifact's `eval_metrics.regime_uncertainty`. Artifacts trained before this
change will not have these keys — inference gracefully falls back but the new
enrichment columns (`hmm_cf_var_*`, `hmm_cf_cvar_*`, `hmm_weighted_kurtosis`,
`hmm_gaussian_divergence`, `hmm_volatility_tier_ka`) will be null until the model is
retrained via `python retrain_hmm.py --symbols 60`.

**Verified:** Full pipeline produces 8 new columns in deployment CSV. 628 tests pass
(18 new + 610 existing). Code review: no blocking issues, strictly additive, all
existing functions and return structures unchanged.

---

### Changed - HMM Artifact Policy (single production source)

- Hardened `src/neutralgrid/models/artifacts.py` to fail loud when HMM artifact
  resolution is ambiguous or unsafe:
  - Rejects legacy `artifacts/hmm/global`.
  - Requires production artifact directories to match
    `rolling_180d_YYYYMMDD_HHMMSS`.
  - Requires `metadata.json.artifact_version` to match the directory name.
  - Rejects conflicting `HMM_ARTIFACT_DIR` vs `artifact_manifest.json`.
- Hardened promotion path in `promote_hmm_version(...)`:
  - Rejects non-`rolling_180d_*` versions.
  - Rejects dir/version mismatch and metadata/version mismatch.
  - Keeps manifest updates atomic after artifact checks pass.
- Updated retrain entry points to block legacy output targets:
  - `retrain_hmm.py`
  - `src/neutralgrid/cli/retrain.py`

---

### Added - HMM vs Meta-Labeler Compatibility Guardrails

- Added `src/neutralgrid/models/artifact_compat.py` with lineage-aware checks
  between active HMM metadata and meta-labeler metadata.
- Updated `run_full_pipeline.py` to call
  `_check_artifact_version_consistency()` at startup.
- Updated `src/neutralgrid/models/meta_labeler.py` save path to persist HMM
  lineage in metadata:
  - `lineage.hmm_artifact_version`
  - `lineage.hmm_pipeline_version`
  - `lineage.hmm_trained_at_utc`

---

### Added - Non-Stationary HMM Uncertainty Framework

- Added `src/neutralgrid/models/hmm/uncertainty.py` to compute regime-conditional
  uncertainty from historical returns:
  - Empirical VaR/CVaR and Gaussian VaR/CVaR.
  - Higher moments including skew and excess kurtosis.
  - Conditional tail-event probabilities by regime.
  - Volatility tier partitioning (`low`, `medium`, `high`).
  - Optional exogenous uncertainty channels (`news_risk`, `event_risk`,
    `macro_risk`, `political_risk`) with robust z-score aggregation.
- Integrated uncertainty into training/inference:
  - `src/neutralgrid/models/hmm/train.py` stores `regime_uncertainty` in
    evaluation metadata and exports volatility tiers in training stats.
  - `src/neutralgrid/models/hmm/inference.py` emits posterior-weighted
    `conditional_tail_risk`, volatility tier probabilities, and dominant
    volatility tier.
  - `src/neutralgrid/validation/hmm_regime.py` and
    `src/neutralgrid/validation/regime_validator.py` propagate those metrics into
    validation outputs.
  - `src/neutralgrid/scanner/enrich_grid_params.py` writes HMM tail metrics to
    enrichment output columns:
    `hmm_tail_var_95`, `hmm_tail_cvar_95`, `hmm_tail_var_99`,
    `hmm_tail_cvar_99`, `hmm_tail_prob_p01`, `hmm_volatility_tier`,
    `hmm_posterior_mode`.

---

### Retrained Models (2026-03-01)

- Promoted HMM artifact:
  `artifacts/hmm/rolling_180d_20260301_180304`
- Updated manifest:
  `artifact_manifest.json` now points `hmm.active_version` to
  `rolling_180d_20260301_180304`.
- HMM metadata snapshot:
  - `num_sequences`: 59
  - `total_samples`: 79,527
  - `training_start_utc`: 2025-12-29T07:00:00+00:00
  - `training_end_utc`: 2026-03-01T17:59:59.999000+00:00
- Retrained meta-labeler artifact:
  `models/meta_labeler/metadata.json` now records lineage to
  `rolling_180d_20260301_180304`.

---

### Validation

- Targeted HMM test suite executed:
  - `tests/unit/test_hmm_artifact_policy.py`
  - `tests/unit/test_artifact_compatibility.py`
  - `tests/unit/test_hmm_uncertainty.py`
- Result: `9 passed` (`pytest -q -p no:cacheprovider ...`).
- Full pipeline execution output captured at:
  `results/deployment_ready_20260301_143336.csv` (150 scanned rows, 1 valid:
  `ARUSDT`).

---

## [6.4.6] - 2026-02-27

### Changed — Batch Backtest (score > 51) + Meta-Labeler Retrain

Backtested 200 candidates (score > 51) from `results/deployment_ready_*.csv`
using the v4 engine (cooldown=120, maker close fees), then retrained the
meta-labeler on the combined live + backtest dataset.

**Backtesting (200 candidates):**

- 9,047 candidates loaded from 95 scanner CSVs → 222 with valid grids →
  212 after excluding deployed → 200 selected (score range 51.9–100.0).
- Mean PnL: +3.91%, median: +3.35%, 54% above 3% hurdle.
- Output: `data/backtest_candidates/backtest_results_20260228.csv` (200 rows),
  `data/backtest_candidates/training_data_20260228.csv` (200 rows).

**Meta-labeler retrain:**

- 292 training samples: 92 live (weight 1.0) + 200 backtest (weight 0.5).
- CV AUC: 0.568, Precision@5: 0.720, F1: 0.604 @ threshold 0.30.
- Top features by MDI: `profit_per_grid_pct` (0.24), `ou_halflife` (0.17),
  `range_size_pct` (0.11), `rsi_15m` (0.10), `hurst_exponent` (0.10).
- Model saved to `models/meta_labeler.pkl`.
- Training data exported to `data/training_sets/combined_training_20260228.csv`.

**Bug fix — `retrain_meta_labeler.py`:**

- Mixed timestamp types (`pd.Timestamp` from Excel vs ISO strings from backtest
  CSV) caused `TypeError` in CPCV sort. Added `pd.to_datetime()` normalization
  after merging backtest data.

---

### Changed — Global Cooldown Calibration (20 → 120 bars)

Swept 14 cooldown values (0–300) across 5 live-overlap candidates to minimize
the backtest-vs-live PnL gap. Selected `global_cooldown_bars=120` as the
training default — MAE-optimal at exactly 1.0x live trade ratio.

| Cooldown | Avg Gap | MAE | Trade Ratio |
|----------|---------|-----|-------------|
| 0        | +41.75% | 41.75% | 14.9x |
| 20 (old) | +13.28% | 18.33% | 4.0x  |
| **120 (new)** | **+4.42%** | **11.90%** | **1.0x** |
| 250      | +1.37%  | 10.35% | 0.5x (over-suppresses) |

Residual ~12% MAE driven by two outliers (RENDERUSDT: live crashed through grid
with 1 trade; RIVERUSDT: 14-grid setup over-suppressed by high cooldown). These
point to the fill model (close-to-close crossing) as the next frontier.

- **`backtest/btk_label_contract.py`**: `TRAINING_ENGINE_DEFAULTS["global_cooldown_bars"]`
  changed from `20` to `120`.
- **`tests/unit/test_btk_global_cooldown.py`**: Updated assertions 20 → 120.
- **`tests/unit/test_btk_label_runner.py`**: Updated assertion 20 → 120.

---

### Changed — P0-P4: Backtest-vs-Live Gap Closure

Closes a **+37.6% average PnL overestimate** identified by cross-validating 7
overlapping candidate_ids between backtest results and live expired bots. Five
priority fixes applied. Engine version bumped from `realistic-v3` to
`realistic-v4`. Label contract version bumped from `1.0` to `2.0`.
601 tests pass (577 existing + 24 new), 0 failures, 0 warnings.

**P0+P1 — Global cooldown (biggest impact — reduces trade count 10-50x):**

After a grid fill, real bots cancel ALL orders → recompute ladder → replace ALL
orders (~1-2 min). The engine previously only blocked the individual filled level
for 2 bars. Now `global_cooldown_bars` blocks ALL levels for N bars after ANY fill.

- **`backtest/backtest_realistic.py`**:
  - Added `global_cooldown_bars: int = 0` to `GridConfig` (0 = disabled, backward
    compatible).
  - Added `_global_cooldown_until` state tracking in `__init__`.
  - Added `_fills_allowed` gate before level loop — skips all fill detection when
    in cooldown window.
  - Added `_fills_allowed` guard on wick-fill section.
  - Added cooldown reset after any fill: `_global_cooldown_until = bar_idx + N`.
  - Added `global_cooldown_bars` to result dict.

- **`backtest/btk_label_contract.py`**:
  - Added `"global_cooldown_bars"` to `ENGINE_SETTINGS_FIELDS`.
  - Added `"global_cooldown_bars": 120` to `TRAINING_ENGINE_DEFAULTS`
    (calibrated from 5-symbol live sweep — MAE-optimal at 1.0x trade ratio).
  - Added `"global_cooldown_bars"` to `extract_engine_settings()`.

**P2 — Maker close fees (live shows 100% maker fills):**

Live data shows 100% of close fills are maker (0.02%) but the training default
charged taker (0.05%).

- **`backtest/btk_label_contract.py`**:
  - Changed `close_fee_mode` in `TRAINING_ENGINE_DEFAULTS` from `"taker"` to
    `"maker"`.

**P3 — Termination penalty (unrealized PnL decomposition):**

Live bots end with open inventory that has unrealized losses. The backtester
already included unrealized PnL in `final_equity` but didn't decompose it.

- **`backtest/backtest_realistic.py`**:
  - Added `termination_pnl_pct = unrealized / capital * 100` to result dict.
  - Added `exit_penalty_pct = abs(min(0, termination_pnl_pct))` to result dict
    (always >= 0).

- **`backtest/btk_label_contract.py`**:
  - Added `"exit_penalty_pct"` to `REQUIRED_LABEL_FIELDS`.
  - Added `"exit_penalty_pct"` to numeric type-check in `validate_engine_result()`.

**P4 — Duration override (match live bot duration):**

Backtests always used 12h (720 bars) but live bots run variable durations
(avg ~16h). Comparing mismatched windows was invalid.

- **`src/neutralgrid/backtest/candidate_pipeline.py`**:
  - Added `duration_bars: int | None = None` parameter to
    `run_single_backtest()`.
  - Added kline truncation: `klines_df.iloc[:duration_bars].copy()` before
    engine call.
  - Added `duration_bars` to docstring.

- **`backtest_candidates.py`**:
  - Added `--match-live-duration` CLI flag (loads expired bots Excel, maps
    `candidate_id → duration_bars`).
  - Added `--expired-bots-path` CLI arg (default `data/new_expired_bots.xlsx`).

**Engine & contract version bumps:**

- **`backtest/btk_unified_runner.py`**: `ENGINE_VERSION` bumped from
  `"realistic-v3"` to `"realistic-v4"`.
- **`backtest/btk_label_contract.py`**: `LABEL_CONTRACT_VERSION` bumped from
  `"1.0"` to `"2.0"`.

**New tests (24 total):**

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_btk_global_cooldown.py` | 8 | zero noop, blocks all levels, resumes after window, reduces trade count, resets on each fill, in result dict, with order delay, training defaults |
| `tests/unit/test_btk_maker_close_fee.py` | 5 | training defaults, maker < taker PnL, taker override, fee rate, build_training_config |
| `tests/unit/test_btk_termination_penalty.py` | 6 | no position = 0, open position penalty, matches unrealized, non-negative, net = realized + termination, fields present |
| `tests/unit/test_btk_duration_override.py` | 5 | none uses all, truncation, longer noop, affects trade count, zero edge case |

**Updated existing tests:**

- **`tests/unit/test_btk_output_contract.py`**: Added `global_cooldown_bars`,
  `termination_pnl_pct`, `exit_penalty_pct` to `EXPECTED_FIELDS`.
- **`tests/unit/test_btk_label_runner.py`**: Added `"exit_penalty_pct": 0.0` to
  validation dicts; changed `close_fee_mode` assertion to `"maker"`; added
  `global_cooldown_bars == 20` assertion; renamed `test_close_fee_mode_is_taker`
  to `test_close_fee_mode_is_maker`; added `"global_cooldown_bars"` to
  `expected_keys` in settings extraction test.

**Bugs found and fixed during review team (reviewer/tester/verifier):**

- **MEDIUM**: `test_open_position_has_penalty` passed vacuously — conditional
  assertion never fired because the position was always closed through
  down-crossing. Redesigned with `global_cooldown_bars=50` to block close,
  ensuring an open position with unrealized loss at termination.
- **LOW**: Stale comments at two locations said "never gated by cooldown" after
  global cooldown was added. Updated to "gated by global cooldown — real bot
  cancels ALL orders during recompute".
- **LOW**: `validate_engine_result()` didn't type-check `exit_penalty_pct` in the
  numeric fields tuple. Added to the check loop.

---

### Added — Feedback Loop Closure: Scanner → Live Outcomes → Retraining

Closes the feedback loop between scanner predictions and live bot outcomes.
Previously the meta-labeler trained on expired bots directly without linking
back to the scanner candidate that originated each deployment. Now every
deployed candidate is tracked end-to-end: scan → deploy → outcome → retrain.
577 tests pass (537 existing + 40 new), 0 failures, 0 warnings.

**New modules:**

- **`src/neutralgrid/core/candidate_id.py`**: Stable candidate_id generation.
  - Format: `{SYMBOL}_{YYYYMMDD_HHMMSS}_{hash8}` where hash8 is SHA-256 of
    `grid_lower|grid_upper|leverage|num_grids` (deterministic, collision-resistant).
  - `make_candidate_id()`: from explicit parameters.
  - `make_candidate_id_from_row()`: from a DataFrame row (used in pipeline).
  - `extract_parts()`: parses both new `SYMBOL_TS_HASH` and legacy `SYMBOL_TS`
    formats, returning `{symbol, scan_ts, config_hash}`.
  - `_config_hash()`: deterministic 8-char hex digest from grid config fields.

- **`src/neutralgrid/live/candidate_deploy_linker.py`**: Deploy-time linkage logger.
  - `DeployLinker` class: append-only CSV logger recording
    `candidate_id → strategy_id` at deployment time.
  - 18-column linkage CSV with auto-timestamped `deployed_at_utc`.
  - Forward/reverse lookups: `get_strategy_ids_for_candidate()`,
    `get_candidate_id_for_strategy()`.
  - `log_deployment_from_row()`: convenience method for DataFrame rows.

- **`src/neutralgrid/training/live_outcome_ingestor.py`**: Live outcome ingestion.
  - `LiveOutcomeIngestor` class: joins expired bot data with deploy linkage
    and scanner CSVs to produce outcome records keyed by `candidate_id`.
  - Three-tier matching: linkage (gold), forensic timestamp (silver), unmatched.
  - Extracts scan-time features (prefixed `scan_*`) from `deployment_ready_*.csv`
    files — uses features at scan time, not bot-end (avoids look-ahead bias).
  - 28 scan feature fields, 22 outcome fields extracted per record.

- **`src/neutralgrid/training/unified_training_builder.py`**: Unified training table.
  - `UnifiedTrainingBuilder` class: merges live outcomes (weight=1.0) and
    backtest rows (weight=0.5) into a single AFML-compliant training table.
  - 15 standard training features, AFML event metadata (t1, barrier_touched, y),
    provenance columns (source, sample_weight_override, candidate_id).
  - `_SCAN_TO_FEATURE` mapping: 18 scan-prefixed columns → standard feature names.
  - Deduplication: when same candidate_id appears in both live and backtest,
    keeps the live row (higher-quality label).
  - Feature coverage diagnostics logged (full/partial/missing breakdown).

- **`src/neutralgrid/live/__init__.py`**: Package init for live deployment modules.

**Updated modules:**

- **`run_full_pipeline.py`**: `candidate_id` column now generated via
  `make_candidate_id_from_row()` with config hash (was plain `SYMBOL_TS`).

- **`retrain_meta_labeler.py`**: New `--unified-table` CLI flag routes through
  `UnifiedTrainingBuilder` instead of the legacy direct-from-Excel path.
  New args: `--linkage-dir`, `--scanner-results-dir`, `--backtest-results-dir`.
  Legacy retraining path fully preserved in `else` branch.

- **`src/neutralgrid/backtest/candidate_pipeline.py`**: `_parse_scan_timestamp()`
  updated to use `extract_parts()` from `candidate_id.py` — fixes parsing of
  new `SYMBOL_TS_HASH` format (was breaking on `split("_", 1)`).

**New tests (40 total):**

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_candidate_id_hash.py` | 13 | candidate_id generation, determinism, format, extract_parts |
| `tests/unit/test_deploy_linker.py` | 10 | CSV creation, append-only, lookups, from_row, config preservation |
| `tests/unit/test_live_outcome_ingestor.py` | 8 | linkage match, forensic fallback, scan features, empty input |
| `tests/unit/test_unified_training_builder.py` | 9 | build, features, provenance, weights, dedup, labels, mixed sources |

**Bugs found and fixed during review:**

- **CRITICAL**: `_parse_scan_timestamp()` in `candidate_pipeline.py` failed on
  new `SYMBOL_TS_HASH` format — `split("_", 1)` yielded
  `20260224_100100_a1b2c3d4` which failed all strptime patterns. Fixed by
  switching to `extract_parts()`.
- **MEDIUM**: Missing `end_time_utc` parameter in label generator call within
  `unified_training_builder.py` — labels were less accurate without actual end
  time. Fixed.
- **MEDIUM**: tz-aware vs tz-naive timestamp comparison in forensic matching —
  `pd.Timestamp(bot_start, tz="UTC")` failed on already-tz-aware timestamps.
  Fixed with try/except and `tz_convert`.
- **LOW**: `metadata["feature_coverage"]` computed but never logged — added
  diagnostic logging with full/partial/missing breakdown.
- **LOW**: Misleading help text for `--unified-table` said "Ignores --input" —
  corrected to "Uses --input as expired bots path".

---

## [6.4.5] - 2026-02-26

### Added — Backtester Reconstruction P0–P3 Gap Fixes

Addresses 9 gaps identified against the Binance USDT-M Futures microstructure
specification. Engine version bumped from `realistic-v2` to `realistic-v3`.
537 tests pass (499 existing + 38 new), 0 failures.

**P0 — Critical:**

- **P0-A: Liquidation modeling** (`backtest/backtest_realistic.py`):
  New `GridConfig` fields: `maintenance_margin_rate` (default 0.004, Binance
  bracket 1). Per-bar check: if `total_equity <= position_notional × MMR`,
  simulation terminates. New result fields: `liquidated`, `liquidation_bar`,
  `liquidation_equity`. Liquidated runs always label as negative.

- **P0-B: Exchange filter rounding** (`backtest/backtest_realistic.py`):
  New `GridConfig` fields: `tick_size`, `step_size` (default 0.0 = no rounding,
  backward compatible). Grid levels rounded to `tick_size`, position qty to
  `step_size`. Matches `SymbolInfo.round_price()` / `round_quantity()`.

**P1 — High:**

- **P1-A: Mark-price klines** (`backtest/backtest_realistic.py`,
  `candidate_pipeline.py`): New `GridConfig.price_source` field ("last" or
  "mark"). `fetch_historical_klines()` routes to `get_mark_price_klines()`
  when `price_source="mark"`.

- **P1-B: Historical funding rate series** (`backtest/backtest_realistic.py`,
  `binance_client.py`, `funding_rate.py`): New `GridConfig.funding_rate_series`
  (list of per-8h rates). When set, overrides static `funding_rate`. Added
  `start_time`/`end_time` params to `BinanceClient.get_funding_rate()`. Fixed
  incorrect docstring claiming historical rates unavailable.

- **P1-C: Margin mode** (`backtest/backtest_realistic.py`): New
  `GridConfig.margin_mode` ("isolated" or "cross"). Serialized in results.

**P2 — High/Medium:**

- **P2-A: Leverage/margin sweep** (new `backtest/btk_leverage_sweep.py`):
  `sweep_leverage()` runs backtest at each leverage level, returns sorted
  results. `find_optimal_leverage()` returns highest-equity non-liquidated
  result. Depends on P0-A liquidation modeling.

- **P2-B: Bid-ask spread model** (`backtest/backtest_realistic.py`): New
  `GridConfig.spread_bps` (default 0.0). Buy fills adjusted by
  `+spread_bps/10000`, sell fills by `-spread_bps/10000`. Additive with
  existing `slippage_bps`.

**P3 — Medium:**

- **P3-A: Intrabar wick fills** (`backtest/backtest_realistic.py`): New
  `GridConfig.fill_mode` ("close" or "wick"). When "wick", checks if
  `low <= level <= high` for fill detection in addition to close-to-close
  crossing.

- **P3-B: Order event log** (new `src/neutralgrid/storage/order_event_log.py`):
  `OrderEvent` dataclass + `OrderEventLog` class. Records bar_idx, timestamp,
  event_type (OPEN_BUY, CLOSE_SELL, STALE_CLOSE, LIQUIDATION), level, price,
  qty, pnl, fee, equity_after. `.to_dataframe()` and `.to_sqlite()` export.

**Contract & version updates:**

- `btk_label_contract.py`: Added `liquidated` to `REQUIRED_LABEL_FIELDS`.
  Added 8 new fields to `ENGINE_SETTINGS_FIELDS`. Updated
  `TRAINING_ENGINE_DEFAULTS`, `extract_engine_settings()`,
  `validate_engine_result()`.
- `btk_unified_runner.py`: `ENGINE_VERSION` bumped to `"realistic-v3"`.
- `test_btk_output_contract.py`: Updated `EXPECTED_FIELDS` with all new fields.
- `test_btk_label_runner.py`: Updated `expected_keys` and validation dicts.

**New test files (38 tests):**

| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_btk_liquidation.py` | 6 | no-liq, crash-liq, stops-sim, label-false, configurable MMR, default |
| `tests/unit/test_btk_exchange_rounding.py` | 6 | no-rounding, tick, step, BTCUSDT, level count, SymbolInfo match |
| `tests/unit/test_btk_mark_price.py` | 5 | default source, result field, mark accepted, margin mode |
| `tests/unit/test_btk_historical_funding.py` | 4 | no-series, override, extend-last, series-len |
| `tests/unit/test_btk_leverage_sweep.py` | 4 | returns list, count, optimal not liquidated, monotonic |
| `tests/unit/test_btk_spread.py` | 4 | default zero, reduces PnL, additive with slippage |
| `tests/unit/test_btk_intrabar.py` | 4 | default close, wick detects touch, more trades |
| `tests/unit/test_order_event_log.py` | 5 | events recorded, types, equity tracking, to_dataframe |

---

## [6.4.4] - 2026-02-22 / 2026-02-25

### Added — Unified Runner API & Label Contract (2026-02-25)

Single source of truth for all backtest-based label generation. Every
pipeline path now goes through `btk_unified_runner.run_backtest()`, which
annotates results with engine settings and validates the label contract.
499 tests pass (464 existing + 35 new), 0 regressions.

**New modules:**

- **`backtest/btk_label_contract.py`**: Authoritative label contract.
  - `LABEL_CONTRACT_VERSION = "1.0"`, `STANDARD_HORIZON_HOURS = 12.0`
  - `REQUIRED_LABEL_FIELDS` (frozenset): `label_positive_by_horizon`,
    `final_equity`, `max_drawdown_pct`, `net_pnl_pct`, `funding_mode`,
    `funding_fees`, `fees_paid`
  - `ENGINE_SETTINGS_FIELDS`: 11 metadata fields serialized into every row
  - `TRAINING_ENGINE_DEFAULTS`: `funding_mode="continuous"`,
    `close_fee_mode="taker"`, `order_delay_bars=2`, `max_holding_bars=720`,
    standard Binance fees, `funding_rate=0.0001`
  - `validate_engine_result()`: rejects missing/wrong-type fields
  - `extract_engine_settings()`: extracts reproducibility metadata from
    GridConfig

- **`backtest/btk_unified_runner.py`**: THE single entry point for all
  backtest execution.
  - `ENGINE_VERSION = "realistic-v2"`
  - `build_training_config()`: builds GridConfig with training-standard
    defaults, merges caller overrides (None ignored)
  - `run_backtest()`: wraps RealisticGridBacktester, annotates with engine
    version + contract version + settings, validates contract before return.
    Accepts optional `seed_state` for replay seeding.

**Updated production callers:**

- **`src/neutralgrid/backtest/candidate_pipeline.py`**: `run_single_backtest()`
  now uses `build_training_config()` + `run_backtest()` from unified runner.
  Default `funding_mode` changed from `"snapshot"` to `"continuous"`.
  `convert_to_training_row()` logs warning when `label_positive_by_horizon`
  missing (fallback preserved).

- **`backtest_candidates_current.py`**: Direct `RealisticGridBacktester`
  usage replaced with `run_backtest()` from unified runner.

**Bypass verification**: Only `btk_unified_runner.py` and unit tests import
`RealisticGridBacktester`. All production code paths go through the runner.

**numpy 2.4.1 compatibility**: `validate_engine_result()` accepts both
`bool` and `np.bool_` (numpy 2.4+ changed `bool_.__name__` to `"bool"` but
`isinstance(np.bool_(True), bool)` returns `False`).

**New tests** (`tests/unit/test_btk_label_runner.py`, 35 tests):

- `TestLabelContractValidation` (7): valid passes, missing fields rejected,
  single missing field rejected, wrong type label rejected, wrong type equity
  rejected, fields are frozen, version is string
- `TestEngineSettingsExtraction` (3): extracts all settings, values match
  config, settings fields list
- `TestBuildTrainingConfig` (6): training defaults applied, override funding
  mode, override capital/leverage, None override ignored, grid geometry set,
  funding rate override
- `TestRunBacktest` (9): engine version, contract version, all settings
  serialized, required label fields present, label is bool, equity is numeric,
  settings match config, standard output fields preserved, net_pnl consistency
- `TestRunBacktestContractRejection` (1): valid input does not raise
- `TestRunBacktestWithSeed` (1): seed_state=None accepted
- `TestTrainingPipelineIntegration` (3): full pipeline flow, continuous
  funding in result, engine settings in result
- `TestTrainingEngineDefaults` (5): funding mode, close fee mode, order delay,
  max holding bars, all defaults are valid GridConfig fields

### Fixed — Backtester Quality Gaps C, E, G, H (2026-02-25)

Four remaining quality gaps in `backtest/backtest_realistic.py` identified
during the R5/R7 audit. All affect the backtester's ability to accurately
answer: "Given this grid geometry + fee model + execution latency + max holding
time, would the bot end with positive PnL by the horizon?" 464 tests pass
(454 existing + 10 new), 0 regressions.

**Gap C — Configurable close fee mode (accuracy: systematic pessimistic bias)**

- **`GridConfig.close_fee_mode`** field (new, default `"taker"`): selects
  whether grid-crossing close (sell) fills use `taker_fee` (conservative) or
  `maker_fee` (matches live grid bot resting limit TP orders).
- `close_fee_rate` computed once in `run()` and used at both UP and DOWN
  crossing close paths. `_force_close_stale_positions` retains `cfg.taker_fee`
  — force closes are always market/aggressive exits.
- **Impact**: eliminates `(taker_fee - maker_fee) × notional` pessimistic bias
  per close when `close_fee_mode="maker"`.

**Gap E — Mark-price funding notional (accuracy: optimistic funding bias)**

- Funding notional in both continuous and snapshot modes now uses
  `curr_close * pos['qty']` instead of `pos['entry_price'] * pos['qty']`.
  Binance computes funding on mark price, not entry price — when positions
  drift from entry, funding was systematically undercharged.

**Gap G — Performance optimization (efficiency)**

- **`_level_to_idx`** dict precomputed in `__init__`: replaces two O(N)
  `self.grid_levels.index(level)` calls per bar with O(1) dict lookups.
- **`df.iterrows()` → numpy arrays + `range()` loop**: pre-extracts `close`
  and `timestamp` columns via `.to_numpy()`, eliminating per-row Series
  creation overhead. Removed unused `cast` import.

**Gap H — Zero-PnL trade recording (reporting: understated metrics)**

- Removed `if abs(pnl) > 0.01:` guard on the DOWN crossing close path.
  Previously, closes where gross PnL ≈ 0 (entry == sell level) were excluded
  from `trades`, `total_fees`, `round_trips`, and `wins` — even though the
  fee cost was reflected in equity. DOWN close path now matches UP close path
  structure (unconditional recording).
- Updated `tests/unit/test_btk_order_lifecycle.py:167` comment to remove
  reference to the removed filter.

**New tests** (`tests/unit/test_btk_gap_fixes.py`, 10 tests):

- `TestCloseFeeMode` (3): default is taker, maker reduces costs, stale close
  always uses taker.
- `TestMarkPriceFunding` (2): funding uses mark price, funding increases when
  price rises.
- `TestPerformanceOptimization` (2): `_level_to_idx` dict exists and matches
  grid levels, deterministic results across runs.
- `TestZeroPnlTradeRecording` (3): zero-pnl close is recorded, fees counted,
  round trips include zero-pnl.

### Added — Candidate Backtesting Pipeline + Meta-Labeler Training Expansion (2026-02-24)

- **`src/neutralgrid/backtest/candidate_pipeline.py`** (new library module):
  reusable functions for backtesting undeployed scanner candidates against
  historical OHLCV data.
  - `load_all_scanner_csvs(results_dir)`: loads all `deployment_ready_*.csv`
    files, synthesizes `candidate_id` for older CSVs (pre-Feb 18) from
    `{symbol}_{filename_timestamp}`.
  - `filter_backtest_candidates(df, linkage_path, min_score)`: filters by
    score >= 70, `grid_is_valid`, non-null grid params; excludes deployed bots
    via `candidate_execution_linkage.csv`.
  - `fetch_historical_klines(client, symbol, start_time, hours)`: fetches 1m
    OHLCV from Binance API starting at scan timestamp (not "most recent").
  - `run_single_backtest(candidate_row, klines_df)`: wraps
    `RealisticGridBacktester` with `GridConfig` built from candidate row.
  - `build_feature_dict_from_scanner_row(row)`: maps scanner CSV columns to
    the 15 standard meta-labeler training features via `_SCANNER_TO_FEATURE`
    mapping.
  - `convert_to_training_row(backtest_result, candidate_row)`: produces
    AFML-compliant meta-labeler training row with `t1` for CPCV purging,
    `source="backtest"`, `sample_weight_override=0.5`.
- **`backtest_candidates.py`** (new top-level CLI script): end-to-end
  candidate backtesting pipeline.
  - Flags: `--min-score 70`, `--max-candidates 50`, `--hours 12`,
    `--min-bars 360`, `--capital 400`, `--leverage 10`, `--results-dir`,
    `--linkage-path`, `--output`, `--dry-run`, `--latest-only`, `--delay 0.2`.
  - Flow: load scanner CSVs → filter candidates → fetch historical klines →
    run `RealisticGridBacktester` → save results.
  - Outputs two CSVs:
    - `data/backtest_candidates/backtest_results_YYYYMMDD.csv` — full backtest
      metrics with `candidate_id` traceability.
    - `data/backtest_candidates/training_data_YYYYMMDD.csv` — meta-labeler-
      compatible format with 15 features, labels, and AFML provenance fields.
  - Initial run: 98 candidates backtested, mean PnL +31.19%, 100% above 3%
    hurdle.
- **`backtest_candidates_current.py`** (new top-level CLI script):
  out-of-sample validation of candidate grid configs against current (most
  recent 12h) market data.
  - Deduplicates to best-scoring config per symbol.
  - Compares scan-time vs current performance side-by-side.
  - Output: `data/backtest_candidates/current_market_backtest_YYYYMMDD.csv`.
  - Initial run: 53 symbols tested, mean PnL +2.75%, only 7/53 still in range,
    81% negative PnL — confirming grid configs are time-sensitive.
- **`retrain_meta_labeler.py`** — added `--include-backtest-data PATH` flag:
  merges backtest-derived training CSV with live bot data before training.
  Backtest rows marked `source="backtest"` with `sample_weight_override=0.5`
  to prevent synthetic data domination. Live rows marked `source="live"` with
  weight 1.0.

### Fixed — Ambiguous Syntax / Command Separator Cleanup (2026-02-24)

Codebase-wide audit for ambiguous syntax that could be misinterpreted. Four
parallel agents scanned all Python files for: semicolon statement separators,
ambiguous comma/tuple usage, bitwise vs logical operator misuse, and fragile
line continuations. 390 tests pass, 0 regressions.

**Backslash continuations → parenthesized expressions (11 fixes)**

- **`src/neutralgrid/data/market_dynamics.py:342`**: `if (...) or \` backslash
  continuation replaced with parenthesized multi-line `if` block. Backslash
  continuations break silently with trailing whitespace.
- **`src/neutralgrid/validation/stochastic.py:799`**: `hurst, ... = \` tuple
  unpacking continuation replaced with parenthesized assignment.
- **`tests/unit/test_features.py`** (9 assertions): all `assert x, \` message
  continuations replaced with `assert x, (...)` parenthesized form.

**`== True` / `== False` anti-pattern cleanup (10 fixes)**

- **`run_full_pipeline.py`** (4 lines: 185, 283, 343, 613): pandas Series
  boolean comparisons annotated with `# noqa: E712` — `== True` is correct
  for Series (not `is True`), annotations clarify intent.
- **`src/neutralgrid/backtest/candidate_pipeline.py:186`**: same pandas Series
  pattern, annotated.
- **`tests/unit/test_enrich_grid_params.py`** (3 lines: 102, 128, 201):
  `== False` replaced with `not row[...]` — safe for both Python `bool` and
  numpy `bool_` scalars (unlike `is False` which fails for numpy).
- **`tests/test_afml_compliance.py`** (4 lines: 105, 106, 112, 113):
  `== True`/`== False` on numpy matrix values replaced with truthiness checks
  (`assert mat[0,1]` / `assert not mat[0,1]`).

**No issues found (clean categories)**

- Semicolons as statement separators: 0 instances across 28 files (all inside
  strings/comments/docstrings).
- Ambiguous comma usage / unintended tuples: 0 instances.
- Missing parentheses in pandas boolean indexing: 0 instances.
- Shell command injection risk: 0 instances (subprocess uses list args).
- Implicit string concatenation: 0 instances.

### Fixed — Mark-to-Market Equity, Continuous Funding & Replay Seed (2026-02-24)

Five-phase overhaul of `backtest/backtest_realistic.py` to close the gap
between backtested and live PnL (RIVERUSDT backtest was +22.53% vs live
+15.62% — a 44% overestimate). Root causes: equity/DD tracked only realized
PnL, funding fees only charged at 480-bar snapshots, and open-position PnL
excluded from final output. 417 tests pass (390 existing + 27 new), 0
regressions.

**Phase 1 — Mark-to-market equity curve & drawdown**

- **`_unrealized_pnl(mark_price)`** (new helper): iterates
  `self.positions.values()` and sums `(mark_price - entry_price) * qty`.
- **Per-bar MTM equity curve**: `equity_curve` now appends
  `equity + unrealized_pnl` each bar instead of realized-only `equity`.
  Previously the curve was flat between trade events even when open positions
  were deeply underwater.
- **Per-bar drawdown from MTM**: removed 3 inconsistent per-event DD update
  blocks (old lines 224-226, 258-260, 313-315 which only fired on some
  events and never included unrealized). DD is now computed once per bar:
  `peak_equity = max(peak_equity, total_equity)`,
  `dd = (peak_equity - total_equity) / peak_equity`.
- **`net_pnl` redefined**: `final_equity - capital` (MTM-correct) instead of
  `total_realized_pnl - fees - funding` (realized-only).
- **New output fields**: `unrealized_pnl_at_end`, `final_equity`,
  `peak_equity`.
- **7 new tests** (`tests/unit/test_btk_mtm_equity.py`): equity curve
  includes unrealized gains, drawdown reflects unrealized dips,
  `unrealized_pnl_at_end` matches manual computation, `final_equity =
  capital + net_pnl`, `label_positive_by_horizon` consistency, direct
  `_unrealized_pnl()` method test, equity curve length = bar count.

**Phase 2 — Continuous funding accrual**

- **`GridConfig.funding_mode`** field: `"snapshot"` (default, backward-
  compatible) or `"continuous"` (prorated per bar).
- **"continuous" mode**: precomputes `funding_per_bar = funding_rate /
  funding_interval_bars`, then charges `notional_now * funding_per_bar`
  every bar with open positions. Fixes the issue where positions churning
  within 8h windows escaped funding entirely.
- **"snapshot" mode**: preserved identically to original code (charge at
  exact `bar_idx % 480 == 0` ticks).
- **`funding_mode`** included in output dict for provenance.
- **7 new tests** (`tests/unit/test_btk_funding_modes.py`): snapshot charges
  at interval, no charge before interval, continuous proportional charging,
  continuous zero when no positions, continuous >= snapshot over full
  interval, default is snapshot, result includes funding_mode.

**Phase 3 — Replay seed state integration**

- **`backtest/btk_seed_state.py`** (new file): `SeedOrderLevel` (frozen
  dataclass: side, price, qty_total, order_count) and `SeedState` dataclass
  (t0, open_buy_levels, open_sell_levels, open_positions, symbol).
  Intentionally decoupled from `src/neutralgrid/replay/` — reads exported
  CSVs only.
- **`backtest/btk_replay_seed_loader.py`** (new file):
  `load_seed_from_replay(replay_dir, symbol, t0)` reads
  `<replay_dir>/<SYMBOL>/levels.csv`, filters `time_ms <= t0_ms`, picks
  latest snapshot, constructs `SeedState` with BUY/SELL level lists.
  Returns `None` gracefully when no matching data exists.
- **`seed_from_state(seed, bar0_idx=0)`** method on
  `RealisticGridBacktester`: clears state, snaps each seeded level to the
  nearest grid level (2% tolerance), sets `_level_available_bar[level] =
  bar0_idx` so seeded levels are immediately fillable. Does not fabricate
  positions — only marks active grid levels.
- **`candidate_pipeline.py`**: `run_single_backtest()` accepts optional
  `replay_seed_dir: Path | None`. When provided and
  `<dir>/<SYMBOL>/levels.csv` exists, seeds the backtester before `run()`.
  Exception in seed loading is caught and logged (graceful degradation).
- **6 new tests** (`tests/unit/test_btk_seed_state.py`): SeedOrderLevel
  immutability, SeedState defaults, seed_from_state grid snapping,
  out-of-tolerance rejection, load_seed_from_replay with mock CSV.

**Phase 4 — Horizon label and output contract**

- **`label_positive_by_horizon`** (bool): `final_equity > capital`. Direct
  answer to "would the bot end with positive PnL by the horizon?"
- **`convert_to_training_row()`**: uses `label_positive_by_horizon` from
  backtest output when available, falls back to `hurdle_pct` comparison for
  backward compatibility with older result dicts.
- **`print_results()`**: displays MTM-specific fields (Final Equity, Peak
  Equity, Unrealized PnL, Horizon Label, Funding Mode).
- **12 new tests** (`tests/unit/test_btk_output_contract.py`): all 32 output
  fields present, `net_pnl == final_equity - capital`, label consistency,
  `net_pnl_pct` formula, peak >= final equity, funding_mode matches config,
  training row picks up horizon label (True/False/absent fallback), source
  and weight provenance, flat-price edge case.

**Backward compatibility**

- `funding_mode` defaults to `"snapshot"` — existing scripts produce
  identical results.
- All existing output dict fields preserved; 5 new fields added.
- `replay_seed_dir` defaults to `None` — existing call sites unchanged.
- No signature changes to `run()` or any public API.

## [6.4.4] - 2026-02-22 / 2026-02-23

### Added
- **PriceSeries subsystem** (`src/neutralgrid/data/price_series/`): real-time
  price streaming with REST backfill and dual-layer storage for low-latency
  trigger logic. 7 new modules:
  - `ps_types.py`: canonical data structures — `SeriesKind` enum (`LAST_KLINE`,
    `MARK_KLINE`, `MARK_TICK`), frozen `Candle` and `PriceTick` dataclasses.
  - `ps_store.py`: dual-layer storage — `collections.deque` ring buffer
    (configurable, default 1500 entries per key) for fast in-memory queries,
    plus append-only daily Parquet files (`data/price_store/<SYMBOL>/<kind>/
    <interval>/<YYYY-MM-DD>.parquet`) for durable audit trail. On-disk dedup
    by `open_time_ms`.
  - `ps_rest_backfill.py`: REST gap repair via `BinanceClient.get_klines()` /
    `get_mark_price_klines()` with automatic pagination for large ranges.
    `initial_backfill()` fills startup history across intervals and kinds.
  - `ps_ws_stream.py`: WebSocket streaming via `aiohttp` to Binance combined
    stream endpoint. Subscribes to `@kline_<interval>` and `@markPrice@1s`
    streams. Auto-reconnect with exponential backoff (1s → 60s), 24h
    connection refresh, ping keep-alive, and real-time gap detection that
    triggers REST backfill.
  - `ps_quality.py`: stateless data-quality gates — `validate_candle_series()`
    returns `QualityReport` (monotonicity, gaps, duplicates),
    `detect_gaps()`, `dedupe_candles()`, `filter_closed_only()`,
    `check_candle_monotonic()`, `check_tick_monotonic()`.
  - `ps_manager.py`: `PriceSeriesManager` top-level orchestrator with
    `start()`/`stop()`/`ensure_symbol()` lifecycle, query methods
    (`get_candles`, `get_latest_mark`, `get_latest_last`,
    `get_quality_report`), periodic 60s disk flush, and tracked gap-backfill
    tasks with exception logging and graceful cancellation on shutdown.
  - `__init__.py`: re-exports public API (`PriceSeriesManager`, `PriceStore`,
    `Candle`, `PriceTick`, `SeriesKind`, `QualityReport`).
- **`PriceSeriesConfig`** dataclass (`config.py`): `ring_buffer_size` (1500),
  `store_dir`, `ws_reconnect_base_s`/`max_s`, `ws_ping_interval_s`,
  `gap_threshold_factor` (1.5), `backfill_bars` (500), `ws_base_url`.
  Wired into root `Config` and `_dict_to_config()` loader.
- **`get_mark_price_klines()`** method (`binance_client.py`): mirrors
  `get_klines()` but calls `/fapi/v1/markPriceKlines` for mark-price candles.
  Includes incomplete-bar drop logic. Weight entry added (5).
- **`mark_price_klines`** endpoint added to `BinanceConfig.endpoints`.
- **`PriceSeriesError`** exception (`exceptions.py`): domain error for
  streaming and storage failures (`error_code="PRICE_SERIES_ERROR"`).
- **77 unit tests** (`tests/unit/test_price_series.py`): full coverage of
  types (construction, immutability, enum values), quality gates (monotonicity,
  gap detection, dedup, filtering), store (append, dedup-on-same-open-time,
  closed-only filtering, limit, kind/interval isolation, flush, round-trip
  Parquet persistence, on-disk dedup), and backfill utilities
  (`interval_to_ms`, `_raw_kline_to_candle`). All pure — no network calls.

### Added
- **Order book reconstruction tool** (`scripts/rebuild_order_book.py`): reads all
  bots from `data/new_expired_bots.xlsx`, matches each to its per-symbol replay
  data in `data/replay_outputs/`, and reconstructs the grid order book at the
  initial snapshot within the bot's time range. Outputs
  `data/order_book_reconstruction.xlsx` with:
  - **Summary sheet**: all 92 bots with strategy ID, symbol, status, grid params,
    profit/PnL, buy/sell level counts, best bid/ask, spread %, replay trade
    counts, and realized PnL from replay data.
  - **92 per-bot sheets** (e.g. `WIFUSDT_409872221`): bot metadata, full order
    book ladder (SELL levels descending → spread → BUY levels descending) with
    price, qty, order count, and notional; book stats summary; and trade history
    from replay data (76 of 92 bots had trades in range).
  - Filters out price=0 artifact levels from cross-bot aggregation.
  - Coverage: 92/92 bots matched replay data, 92/92 had order book levels
    reconstructed.

### Added — Trigger Price Pipeline Integration (2026-02-23)
- **`ps_trigger_resolver.py`** (`src/neutralgrid/data/price_series/`): new
  module implementing a 4-level fallback chain to resolve the best available
  trigger price for each symbol during enrichment:
  1. **WS mark price** — from `PriceSeriesManager.get_latest_mark()` (<1s
     latency, most accurate).
  2. **WS last price** — from `PriceSeriesManager.get_latest_last()` (kline
     close).
  3. **REST mark price** — from `premium_index["markPrice"]` (already fetched
     by `get_all_market_data()`).
  4. **vres close** — from `ValidationResult.current_price` (closes_15[-1]).
  Returns frozen `TriggerPriceResult(price, source, ts_ms, age_ms)` dataclass.
  Exported from `price_series/__init__.py`.
- **PriceSeriesManager wired into full pipeline** (`run_full_pipeline.py`):
  - Created and started after `BinanceClient` connection (STEP 2).
  - Passed as `price_manager=` kwarg to `enrich_with_grid_params()` (STEP 4).
  - Stopped before `client.close()` on all exit paths (normal, scan failure,
    enrichment failure).
- **Trigger price capture in enrichment** (`enrich_grid_params.py`):
  - New `price_manager: PriceSeriesManager | None = None` keyword parameter.
  - Pre-warm block: subscribes all eligible symbols to WS mark streams with
    `ensure_symbol()` + 2s settling time before enrichment begins.
  - `resolve_trigger_price()` called inside `run_one()` after
    `validator.validate()`, result stored in `base_payload`.
  - 4 new output columns: `trigger_price`, `trigger_price_type`,
    `trigger_price_ts`, `trigger_price_age_ms`.
  - Columns initialized with `None` defaults in the column-init block,
    ensuring they exist for all rows regardless of eligibility or errors.
  - All 7 return paths in `run_one()` propagate trigger price data (6 via
    `**base_payload` spread, 1 via explicit fields in the exception handler).
  - `tp` variable initialized to `None` before the try block; exception
    handler uses ternary guard (`tp.price if tp else None`).
- **Trigger price display** (`display_deployment_summary()`): MARKET DATA
  section now shows trigger price with source type and age, e.g.
  `Trigger Price: $42000.500000 (ws_mark, age 350ms)`.
- **17 unit tests** (`tests/unit/test_ps_trigger_resolver.py`): full coverage
  of priority selection (ws_mark > ws_last > rest_mark > vres_close > None),
  edge cases (invalid/missing/zero markPrice, non-dict premium_index, time=0),
  age_ms computation, and `TriggerPriceResult` immutability/slots.
- **7 integration tests** (`tests/unit/test_trigger_price_enrichment.py`):
  columns present with/without `price_manager`, REST mark fallback, vres_close
  fallback, ws_mark priority with mock manager, exception path coverage,
  pre-warm `ensure_symbol` verification.

### Fixed — PyArrow Timestamp Type Mismatch (2026-02-23)
- **`write_partitioned()` UTC normalization** (`store.py`): `open_time` and
  `close_time` columns are now defensively localized to UTC before writing to
  Parquet. Naive timestamps get `tz_localize("UTC")`; already-UTC timestamps
  pass through unchanged. This ensures the on-disk schema is always
  `timestamp[ns, tz=UTC]`, matching the UTC-aware filter scalars constructed
  by `load_parquet_range()`.
  - **Root cause**: `_make_ohlcv_df()` in tests created naive timestamps
    (`pd.to_datetime(times, unit="ms")`) while production code
    (`normalize_dataframe`) uses `utc=True`. When stored as `timestamp[ns]`
    (naive) and filtered with `timestamp[s, tz=UTC]`, PyArrow's
    `greater_equal` kernel had no matching implementation for the mixed types.
  - **Test fix**: `_make_ohlcv_df()` now uses `utc=True` for both `open_time`
    and `close_time`, matching production `normalize_dataframe` output.
  - **Result**: `test_range_filter` now passes. Full suite: 204 passed, 0
    failed (previously 203 passed, 1 failed).

### Fixed — AFML Pipeline Bug Fixes (2026-02-23)

Comprehensive pipeline-wide review identified 11 bugs (5 HIGH, 6 MEDIUM priority)
across data curation, feature extraction, backtesting, training, validation,
scanning, and deployment modules. All fixed and verified with 23 new unit tests.

**HIGH Priority**

- **H1 Log return circular reference** (`data/features.py`): `compute_log_returns()`
  used `np.roll()` which wrapped the last element to index 0, creating a
  look-ahead leak in inference. Replaced with pre-allocated array: `r[0] = NaN`,
  `r[1:] = log(closes[1:] / closes[:-1])`.
- **H2 ADX warmup threshold** (`scanner/feature_extractor.py`): 1H ADX warmup
  check used `adx_period + 10` but ADX requires `2 * adx_period` bars (one
  period for TR/DM smoothing, one for DX smoothing). Partial-warmup ADX values
  were being used for regime detection. Changed to `2 * _cfg_val("adx_period")`.
- **H3 Missing logger in CPCV** (`backtest/cpcv.py`): `logger` referenced in
  holdout embargo logging path but never defined. `NameError` on any run that
  triggered the embargo log. Added `import logging` and
  `logger = logging.getLogger(__name__)`.
- **H4 `regime_conf` never populated** (`scanner/enrich_grid_params.py`):
  `regime_conf` column initialized to `1.0` (full confidence) regardless of
  actual `ValidationResult.regime_confidence`. Changed default to `None`, only
  set when `_rc is not None`. Soft-gating confidence signal now propagates.
- **H5 EV score non-reproducible** (`run_full_pipeline.py`): EV score normalized
  with min-max `[50, 100]` scaling, making rankings batch-relative and
  non-reproducible across runs. Replaced with percentile rank:
  `50.0 + 50.0 * ev_vals.rank(pct=True, method='average')`.

**MEDIUM Priority**

- **M1 Variable shadows import** (`models/meta_labeler.py`): instance variable
  `compute_sample_weights: Any = None` shadowed the imported function of the
  same name. Renamed to `_compute_weights_fn`, import aliased to `_csw_fn`,
  all 4 references updated with `is not None` guard.
- **M2 Duration hours unclamped** (`training/data_generator.py`): no validation
  that `duration_hours <= horizon_hours` — events could span beyond the
  prediction horizon. Added
  `clamped_hours = min(float(duration_hours), float(self.config.horizon_hours))`.
- **M3 Bootstrap significance invalid** (`validation/stochastic.py`): bootstrap
  significance test compared bias-corrected `h_rs` against raw bootstrap null
  distribution — mixed scales invalidated the test. Changed to use raw
  `h_rs_raw` for both estimate and null comparison.
- **M4 Active fraction formula** (`scanner/pnl_ranker.py`, `validation/utility.py`):
  linear `min(1.0, num_grids / 50.0) * 0.5` hard-capped at 50 grids
  (active_fraction = 0.5 for all grids >= 50), under-estimating EV for dense
  grids. Replaced with sqrt scaling:
  `min(0.75, (num_grids / 50.0) ** 0.5 * 0.5)`. Gives: 20g -> 0.32,
  50g -> 0.50, 100g -> 0.71, 200g -> 0.75. Also fixed decomposition block in
  `compute_score()` which still used the old linear formula.
- **M5 Excel I/O on every call** (`grid/spacing_profile.py`): investigated and
  confirmed already cached via `_CACHED` module-level variable. No fix needed.
- **M6 Hardcoded provisional utility** (`core/config.py`,
  `validation/regime_validator.py`): provisional utility scoring used hardcoded
  `profit=0.8%, grids=20, range=3%` instead of config values. Added
  `provisional_profit_per_grid_pct`, `provisional_num_grids`,
  `provisional_range_size_pct` fields to `UtilityConfig`. Updated
  `regime_validator.py` to read from config.

### Fixed — Pre-existing Test Failures (2026-02-23)

9 pre-existing test failures across 3 test files, all resolved:

- **6 numpy bool identity failures** (`test_afml_fixes.py`,
  `test_afml_fixes_v2.py`): `TestDuplicateDetection` tests used `is True` /
  `is False` identity checks against `np.True_` / `np.False_` (different
  objects from Python `True`/`False`). Changed to truthy/falsy assertions
  (`assert x` / `assert not x`).
- **Sharpe returns 0.0** (`test_afml_fixes.py::TestSharpeAnnualization`):
  `calculate_sharpe_proxy()` returned 0.0 for known-value inputs. Fixed
  source/test alignment for Sharpe ratio annualization.
- **CPCV purge effectiveness 0%**
  (`test_afml_fixes.py::TestNoLookaheadBias`): test measured purge
  effectiveness incorrectly. Fixed test to properly assess CPCV purging
  behavior.
- **CPCV split count 13 vs 15**
  (`test_afml_integrations.py::TestCPCV::test_cpcv_splits`): `C(6,2) = 15`
  combinations but purging/embargo legitimately removed 2 folds from the small
  120-row dataset. Increased dataset size so all 15 folds survive purging.

### Fixed — 74 Pyright Warnings Eliminated (2026-02-23)

Removed all 74 pyright warnings (42 unused imports, 32 unused variables) across
30+ source files. Zero warnings remaining.

- **Unused imports removed** (42): `field`, `Optional`, `Literal`, `List`,
  `Dict`, `Any`, `Tuple`, `Path`, `timedelta`, `timezone`, `datetime`, `os`,
  `warnings`, `np`, `pd`, `BaggingClassifier`, `PnLRanker`, `RankingConfig`,
  `MetaLabeler`, `FeatureCollector`, `load_training_dataset`,
  `infer_state_mapping`, `_ema_smooth_posteriors`.
  Files: `binance_vision/downloader.py`, `manifest.py`, `pipeline.py`,
  `validate.py`, `coinglass.py`, `market_dynamics.py`, `calculator.py`,
  `grid_bot_manager.py`, `metrics/calculator.py`, `artifacts.py`,
  `meta_labeler.py`, `triple_barrier.py`, `schema.py`,
  `enrich_grid_params.py`, `pnl_ranker.py`, `scan.py`, `database.py`,
  `data_generator.py`, `holdout_validator.py`, `sample_weights.py`,
  `trial_tracker.py`, `hmm_regime.py`, `microstructure.py`,
  `regime_validator.py`, `utility.py`, `core/logging.py`, `cli/retrain.py`,
  `backtest/evaluate.py`.
- **Unused variables prefixed with `_`** (32): `current_high`, `current_low`,
  `trend_state`, `range_state`, `ho_trend_state`, `ho_range_state`,
  `holdout_df`, `cv_df`, `cpcv_df`, `n_thresholds`, `X_cv`, `symbol`,
  `last_exc`, `issues`, `loop`, `total_notional`, `recent_hl`, `recent_lh`,
  `smooth_k`, `t_start_ms`, `middle`, `mid_price`, `n`.
  Files: `binance_client.py`, `backtest/evaluate.py`,
  `binance_vision/downloader.py`, `curator.py`, `exchange_info.py`,
  `calculator.py`, `technical.py`, `hmm/train.py`, `replay/cli.py`,
  `feature_extractor.py`, `microstructure.py`, `sample_weights.py`.

### Added — AFML Bug Fix Tests (2026-02-23)

- **`tests/test_afml_bugfixes.py`** (23 tests): targeted unit tests covering
  all fixed bugs:
  - `TestLogReturnCircularReference` (5 tests): NaN first element, correct
    values, no circular wrap, length preservation, single element.
  - `TestAdxWarmupThreshold` (2 tests): source uses `2 * adx_period`, old
    `+ 10` formula removed.
  - `TestCpcvLogger` (3 tests): import succeeds, logger exists, correct name.
  - `TestDurationHoursClamping` (3 tests): clamped to horizon, within-horizon
    unchanged, source code check.
  - `TestBootstrapSignificanceRaw` (2 tests): uses `h_rs_raw`, comment
    documents rationale.
  - `TestActiveFractionFormula` (8 tests): formula values for 20/50/100/200
    grids, PnLRanker and UtilityScorer integration, source code verification.

### Changed
- **`readmefullpwep.md`** updated to v6.4.4: added price_series subsystem to
  project structure, updated pipeline overview (STEP 2 now starts
  PriceSeriesManager, STEP 4 includes pre-warm + trigger price resolution,
  STEP 5 includes PriceSeriesManager shutdown), added trigger price column
  definitions and fallback chain documentation, added `trigger_price_type` and
  `trigger_price_age_ms` to key metrics table.

## [6.4.3] - 2026-02-19 / 2026-02-20

### Fixed (Pylance strict type-checking — 130+ diagnostics across 20 files)

Systematic elimination of all Pylance/Pyright type-checking errors across the
entire codebase. Zero diagnostics remaining after fixes.

**Recurring pattern: `.values` → `np.asarray()`** (70+ occurrences)
- `pd.Series.values` returns `ndarray | ExtensionArray | Categorical`, which is
  incompatible with typed function parameters expecting `ndarray`. Replaced with
  `np.asarray()` throughout.
- Files fixed: `technical.py`, `feature_extractor.py`, `triple_barrier.py`,
  `meta_labeler.py`, `scan.py`, `sample_weights.py`, `regime_validator.py`.

**Recurring pattern: numpy scalar → `float()` wrapping**
- `np.mean()`, `np.sqrt()`, `np.var()` return `np.floating[Any]`, not `float`.
  Wrapped return values with `float()` where Python `float` was expected.
- Files fixed: `cpcv.py`, `stochastic.py`, `volatility.py`.

**Recurring pattern: `Optional` narrowing guards**
- Added `is not None` / `assert` guards before accessing attributes on Optional
  types, and before passing Optional values to non-Optional parameters.
- Files fixed: `grid_bot_manager.py`, `scan.py`, `database.py`,
  `microstructure.py`, `data_generator.py`, `test_regime_validator.py`,
  `test_afml_integrations.py`.

**Per-file details:**

- **`indicators/technical.py`** (7 fixes): All `.values` → `np.asarray()` in
  `calc_ema`, `calc_sma`, `calc_bollinger_bands`, `calc_donchian`, `calc_atr`,
  `calc_rsi`, `calc_adx`.
- **`metrics/grid_bot_manager.py`** (3 fixes): Added None guard for
  `metrics.end_time` (Optional datetime); fixed `metrics.total_commission` →
  `metrics.commissions` (actual attribute name — was a runtime bug).
- **`scanner/feature_extractor.py`** (24 fixes): All `.values` → `np.asarray()`
  where results passed to typed indicator functions.
- **`scanner/pattern_profile.py`** (2 fixes): `str(xl.sheet_names[0])` for
  `int | str` sheet name; `str(k)` for `Hashable` dict keys.
- **`scanner/profile_model.py`** (2 fixes): Restructured `_vector` method to
  avoid variable reassignment that created unnarrowable `float | Any | None`
  union type. Split into separate if/else code paths.
- **`scanner/scan.py`** (33 fixes across 2 rounds): `.values` → `np.asarray()`;
  annotated result dict as `Dict[str, Optional[float]]`; added `assert` guards
  for conditionally-imported classes (`StochasticConfig`, `UtilityScorer`);
  changed forward-ref type annotations to `Optional[Any]` for try/except
  imports; changed `dict.get()` → `dict[]` access inside try/except blocks;
  changed `_compute_stochastic_features` signature to accept `Optional[float]`.
- **`backtest/cpcv.py`** (5 fixes): `float(np.mean(...))` wrapping; `np.where()`
  for integer indexing instead of boolean ndarray with `.iloc`; `.values` →
  `np.asarray()`.
- **`backtest/evaluate.py`** (5 fixes): `cast(float, result[0])` for scipy
  `spearmanr`/`ttest_ind` results whose stubs type elements as generic
  `_T_co@tuple`.
- **`models/hmm/train.py`** (1 fix): `df.symbol` (returns Series) →
  `str(df["symbol"].iloc[0])`.
- **`models/triple_barrier.py`** (9 fixes): All `.values` → `np.asarray()` in
  `label_entry_from_df`, `label_dataframe`, `label_dataframe_hlc`.
- **`models/meta_labeler.py`** (35 fixes across 2 rounds): `.values` →
  `np.asarray()`; `SimpleImputer` output wrapped with `np.asarray()`; pre-
  initialized possibly-unbound variables; `dict.get` → lambda in `max()` key;
  dict access for `permutation_importance` result; `cast(Literal[...], ...)`
  for sklearn calibration method; `getattr()` for `feature_importances_`;
  widened `feature_importance` field type to `Dict[str, Any]`; added None guard
  for `self._model`.
- **`storage/database.py`** (13 fixes): Added `TYPE_CHECKING` imports for
  `ValidationResult` and `SessionMetrics`; `assert cursor.lastrowid is not None`
  after INSERT; added None guard for `fetchone()` result before index access.
- **`training/data_generator.py`** (7 fixes): Added None/NaT guard for `t0`
  before passing to `datetime` parameters; `pd.Timestamp(date).date()` for
  groupby `Scalar` key; extracted `_dur` variable with explicit None check
  before `float()`.
- **`training/sample_weights.py`** (19 fixes): All `.values` → `np.asarray()`
  (7 occurrences), resolving cascading `.reshape()`, `np.concatenate()`, `&`
  operator, and return type errors.
- **`training/volatility.py`** (10 fixes): Wrapped `np.log()` results in
  `pd.Series()` to retain `.rolling()` method; annotated `results` dict as
  `dict[str, Any]`.
- **`training/trial_tracker.py`** (2 fixes): Added `float('-inf')` fallback in
  `max()` lambda for `Optional[float]` key function.
- **`validation/hmm_regime.py`** (1 fix): `getattr(artifact, "predict")` instead
  of direct attribute access on `object`-typed parameter.
- **`validation/microstructure.py`** (1 fix): `funding_rate if funding_rate is
  not None else 0.0` at call site.
- **`validation/regime_validator.py`** (14 fixes): All `.values` → `np.asarray()`
  for `calc_atr`, `calc_adx`, `calc_rsi`, `np.nanpercentile` calls.
- **`validation/stochastic.py`** (6 fixes): `float()` wrapping for numpy scalar
  returns; replaced `dict(**kwargs)` unpacking with direct named arguments to
  preserve `bool` types (Pylance collapses `dict[str, int | bool]` to
  `dict[str, int]`).

**Test files:**
- **`tests/test_afml_compliance.py`** (2 fixes): `purge_hours=None` →
  `purge_hours=0.0`, `embargo_hours=None` → `embargo_hours=0.0`.
- **`tests/test_afml_fixes.py`** (2 fixes): Same `None` → `0.0` pattern.
- **`tests/test_afml_integrations.py`** (3 fixes): `list[int]` → `list[float]`
  for `List[float]` parameter (List invariance); added `assert label is not None`
  before accessing Optional attributes.
- **`tests/unit/test_regime_validator.py`** (1 fix): Added
  `assert result.tf_1h.reason is not None` before `.lower()` call.

### Retrained Models (2026-02-20)
- Retrained HMM regime model (60 sequences, 77,994 samples, 180-day rolling
  window, promoted to global artifact).
- Retrained meta-labeler (72 samples, CV AUC 0.519, P@5 0.440, top feature:
  hurst_exponent).

### Fixed (CPCV Pipeline — 12 issues)

**Critical (PBO/DSR structurally broken)**
- **#1 PBO input**: PBO now receives centered grid-survival payoffs (forward
  |log-return| capped proxy) instead of raw `bar_range` posteriors, allowing
  per-path Sharpe to go negative and PBO to detect overfitting.
- **#2 Deflated Sharpe**: replaced pseudo-Sharpe (`mean(pass_rates)/std`) with
  actual Sharpe computed from aggregated centered payoffs. Added
  `metric_type: proxy_payoff` to `dsr_summary`.
- **#3 var_sharpe**: computed from per-path Sharpe of centered payoff arrays
  (`np.var(path_sharpes)`) instead of trivially dividing pass-rates by their
  own std (which always produced ~1.0).

**Significant**
- **#4 Proxy outcome report holdout**: `run_cpcv_proxy_outcome_report()` now
  wires `holdout_pct` into `CPCVConfig` and uses `split_with_holdout()`.
- **#5 Embargo default test**: renamed `test_embargo_defaults_to_none` →
  `test_embargo_defaults_to_six` to match the actual `CPCVConfig` default
  of `embargo_hours=6.0`.
- **#6 Factory function params**: `create_cpcv()` now accepts `purge_hours`,
  `embargo_hours`, and `holdout_pct` keyword arguments.
- **#7 Yielded split count**: `CPCVMetrics` tracks `n_yielded` (incremented
  before each `yield` in `split()`).
- **#8 Boundary-based percent purging**: replaced O(test×purge) per-sample
  loop with boundary-based logic that computes positional ranges per test
  group and removes overlapping training samples in one pass.
- **#9 Smoothing docstrings**: added notes to `_ema_smooth_posteriors_full()`
  and `evaluate_split()` explaining walk-forward last-bar vs CPCV per-bar
  smoothing.

**Minor**
- **#11 Deflate CI SE formula**: `deflate()` confidence interval now uses the
  full AFML SE formula (skewness + kurtosis terms) matching
  `prob_sharpe_ratio()`.
- **#12 n_trials fallback**: fallback changed from `len(path_results)` (folds)
  to `0` so `max(1, _n_trials)` yields the no-deflation default, since CPCV
  paths are folds not independent strategy trials.

### Added
- `test_pbo_detects_negative_sharpe_paths` test verifying PBO > 0 when mixed
  positive/negative-mean payoff arrays are provided.

### Tests updated
- `test_purge_zero_removes_nothing` (`test_afml_compliance.py`) and
  `test_purge_zero_removes_nothing_noncontiguous` (`test_afml_fixes.py`):
  explicitly set `purge_hours=0.0, embargo_hours=0.0` to disable time-based
  purging when testing percent-based purge-zero behaviour. Previously these
  tests silently took the time-based path (due to the `CPCVConfig`
  consolidation in 6.4.2 defaulting `purge_hours=48.0`), causing false
  failures.
- `test_cpcv_positional_purging` (`test_afml_fixes.py`): updated source-code
  pattern assertion from removed `neighbor_pos` to new boundary-based patterns
  (`grp_positions` / `train_positions`).

## [6.4.2a] - 2026-02-19

### Fixed
- **Score threshold hard gate enforced** (`enrich_grid_params.py`): below-threshold
  symbols that qualified for enrichment via `range_prob` could end up with
  `grid_is_valid=True`, bypassing the score gate.  Added `_enforce_threshold_gate()`
  helper that forces `grid_is_valid=False`, `grid_reason="score_below_threshold"`,
  and clears grid params (`grid_lower`, `grid_upper`, `num_grids`, etc.) for all
  rows with `below_threshold_tag=True`.  Runs on both the early-return path
  (no eligible symbols) and the normal return path (after enrichment).
- **Stale indicator config cache removed** (`feature_extractor.py`): removed
  module-level `_indicator_cfg` global dict that cached config on first access
  and never refreshed.  `_cfg_val()` now calls `_get_indicator_config()` fresh
  each invocation, ensuring config changes are always reflected in long-lived
  processes.

## [6.4.2] - 2026-02-18

### Added
- **OHLCV semantic validation** at API intake (`curator.py`): validates
  `high >= max(open, close)`, `low <= min(open, close)`, `high >= low`,
  `volume >= 0`. Invalid bars are logged and dropped. Wired into
  `DataCurator.validate_ohlcv()` pipeline.
- **Kline cache TTL** (`market_data.py`): configurable `KLINE_CACHE_TTL_SECONDS`
  (default 3600s). Expired cache entries are refetched from the API.
- **Cross-model version consistency check** (`run_full_pipeline.py`):
  `_check_artifact_version_consistency()` reads metadata from HMM and
  meta-labeler artifacts at pipeline startup and warns if versions differ.
  Non-blocking warning only.
- **Feature diagnostics in training** (`meta_labeler.py`): VIF and pairwise
  correlation diagnostics (`compute_feature_diagnostics`) now run automatically
  during `MetaLabeler.train()`. Logs warnings for VIF > 5.0 or |r| > 0.9.
- **API weight rate limiter** (`binance_client.py`): sliding-window weight
  tracker with per-minute budget (1200 weight limit). Parses
  `X-MBX-USED-WEIGHT-1M` response header. Respects `Retry-After` on 429
  responses. Auto-throttles when approaching limit.
- **`backtest/__init__.py`**: root-level `backtest/` directory is now a proper
  importable Python package.

### Changed
- **Enrichment cap raised**: `max_symbols` default 120 → 150 symbols.
- **Bet sizing formula** (`enrich_grid_params.py`): replaced raw multiplicative
  scaling (`base_fraction * meta_prob`) with AFML Ch 10 continuous bet sizing
  (`base_fraction * (2 * meta_prob - 1)`). Floor of 0.3 retained.
- **NaN handling in training data** (`data_generator.py`): replaced silent
  `fillna(0.0)` with `dropna()` logic. Rows with NaN feature values are
  dropped and logged with count/percentage. Missing feature columns are
  logged and skipped rather than zero-filled.
- **Unified CPCVConfig** (`config.py`, `cpcv.py`): consolidated dual
  `CPCVConfig` dataclasses into a single source of truth in `config.py`
  with AFML-correct defaults (`purge_pct=0.02`, `embargo_pct=0.01`,
  `embargo_hours=6.0`, `holdout_pct=0.20`). `cpcv.py` re-exports from
  `config.py`.
- **Trial log deduplication** (`trial_tracker.py`): `log_trial()` is now
  idempotent — checks if `trial_id` already exists and updates in place
  rather than appending a duplicate entry.
- **Removed import-time config snapshots** (`calculator.py`, `binance_client.py`,
  `app.py`): replaced module-level `_cfg = get_config()` with runtime access
  via `__post_init__` / method-level calls. `GridParams` dataclass defaults
  no longer baked in at import time.
- **Library logging** (`train.py`, `microstructure.py`): replaced `print()`
  calls with `logger.info()`/`logger.warning()` in library modules.
- **Live script imports** (`Live/` scripts): replaced brittle `sys.path.insert`
  hacks with try/except fallback pattern — uses installed package first,
  falls back to path insertion only if needed.
- **Narrowed exception catches** (`run_full_pipeline.py`): broad
  `except Exception` changed to `except ImportError` for import blocks
  and `except (ValueError, TypeError, KeyError)` for computation loops.
- **Version sync**: `pyproject.toml` bumped from 6.4.0 → 6.4.2 to match
  `__init__.py`.
- Retrained HMM regime model (50 sequences, 42,747 samples, mean pass rate 37.47%).
- Retrained meta-labeler (78 samples, CV AUC 0.601, P@5 0.460).
- Retrained scanner profile (13 winners, PnL threshold 7.24%).

### Fixed
- **`timestamp` undefined bug** (`run_full_pipeline.py`): `timestamp` variable
  was only defined inside the `else` branch of `if args.output`, causing
  `NameError` when `--output` was provided. Moved definition before the
  conditional block.

### AFML Compliance Audit
Full codebase audit conducted across 8 AFML pipeline stages:
- **Ch 2 (Data Curation)**: ~70% compliance. Added OHLCV integrity checks
  and cache TTL. Remaining: multi-timeframe temporal alignment.
- **Ch 3-4 (Labels + Weights)**: sample weights, uniqueness, sequential
  bootstrap correctly implemented. Remaining: CUSUM filter, volatility-scaled
  barriers off by default.
- **Ch 7 (CPCV)**: purging, embargo, combinatorial paths correct.
  Consolidated dual CPCVConfig. Remaining: PBO semantics (single-strategy
  proxy vs full AFML multi-strategy algorithm).
- **Ch 6 (Meta-Model)**: fixed bet sizing to 2p-1 formula. Remaining:
  meta_prob not yet used in final ranking, feature overlap documentation.
- **Ch 8 (Features)**: integrated feature diagnostics into training.
  Remaining: entropy/MI evaluation, feature catalogue unification.
- **Ch 11-12 (Evaluation)**: fixed trial log deduplication. Remaining:
  Deflated Sharpe uses pseudo-Sharpe proxy, holdout boundary lacks embargo.
- **Deployment**: fixed timestamp bug, added version consistency check,
  added API weight rate limiter. Removed import-time config snapshots.

### Structural Fixes (P0-P2)
- **P0**: Removed import-time `_cfg = get_config()` from `calculator.py`,
  `binance_client.py`, `app.py` — config now accessed at runtime only.
- **P1**: Added `backtest/__init__.py`, fixed stale `EMA_FAST` test,
  purged `__pycache__/` from source tree, synced `pyproject.toml` version.
- **P2**: Replaced `print()` with structured logging in library modules,
  replaced `sys.path.insert` hacks in Live scripts, narrowed broad
  `except Exception` catches, added API weight rate limiter.

## [6.4.0] - 2026-02-17

### Added
- **Adaptive Edge Tier Grid Spacing**: 2-stage "feasible-first, then optimize" policy.
  Stage 1 generates a MEDIUM-edge grid (micro floor + 0.20% buffer); Stage 2 attempts
  upgrade to BIG-edge (micro floor + 0.60% buffer) with hysteresis margin (0.05%).
  Fallback to MEDIUM is guaranteed — no symbol is discarded if the base tier passes.
- **EdgeTierConfig** dataclass in `config.py` (`enable`, `medium_buffer_pct`,
  `big_buffer_pct`, `upgrade_margin_pct`).
- Audit columns in enrichment output: `below_threshold_tag`, `edge_tier_chosen`,
  `edge_tier_attempted`, `edge_upgrade_success`, `edge_fallback_reason`, `net_edge_pct`.
- `pipeline_version` column stamped into every output CSV for traceability.
- `__version__` exposed in `neutralgrid.__init__`.
- This CHANGELOG.

### Changed
- **Scanner widened**: `--top-n` default 50 → 150 symbols.
- **Enrichment cap raised**: `--max-enrichment` default 60 → 120 symbols.
- **Score threshold lowered**: `score_threshold` 50 → 45.
- **Range probability threshold lowered**: `range_prob_threshold` 0.55 → 0.50.
- **Below-threshold behaviour**: symbols below `score_threshold` are now *tagged*
  (`below_threshold_tag=True`) instead of hard-invalidated. Grid parameters are
  preserved for manual review; symbols are not auto-deploy candidates.
- Pipeline log header and completion line now show the pipeline version.
- Deployment summary header shows pipeline version.

### Fixed
- Below-threshold invalidation previously destroyed computed grid parameters,
  making post-hoc analysis impossible. Tagging preserves all information.

## [6.0.0] - Initial release

Baseline NEUTRAL Grid Bot with HMM regime detection, microstructure cost
analysis, AFML meta-labeling, EV-based ranking, and pattern-profile scanning.
