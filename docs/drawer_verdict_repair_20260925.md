# Drawer timestamp and required-evidence repair

Date: 2026-09-25 (America/Lima).

## Observed failures

The captured three-bot cycle at 19:17 Lima had current public collection, but
the scanner used 19:20:00.294617 as the evaluation time for every bot. PLUME
had L2 age 2.294617 seconds. BILL and APR reported heartbeat ages -7.7 and
-8.7 seconds and returned null L2 evidence. The existing reader correctly
rejected these ages against its five-second future tolerance. The caller had
supplied a time that preceded those later live reads.

Evidence: `outputs/audits/recurring_drawer_verdict/heartbeats/20260926_001724/evidence_validation.json`.
An earlier cycle independently recorded L2 unavailable at approximately 4919
seconds old, yet the controller still returned complete recommendations:
`outputs/audits/recurring_drawer_verdict/heartbeats/20260926_001128/l2_freshness.json`.
Both cycles were invalidated through `--capture-failure`, not replayed.

## Changes and reasons

| File and location | Change | Reason and verification |
| --- | --- | --- |
| `live_decision_scanner.py`, `_run_tick` | Obtain a UTC timestamp immediately before each bot evaluation and pass the same timestamp to its decision step. | A previous bot's network/model work must not make a newer heartbeat appear future-dated. The batch timestamp remains the scheduling/output-grouping time. A regression test advances the clock by 12 seconds after the first bot and uses the actual L2 manifest reader on the second bot. |
| `scripts/run_live_telemetry_controller.py`, `run_scanner_tick` and `validate_required_l2_evidence` | An opt-in flag requires validated L2 for every exact roster member before rows return to PnL persistence, forecasts and action routing. | Exit code zero proves process completion, not usable L2. Reject missing evaluation/reference/evidence, malformed diagnostics, L2-unavailable diagnostics, run mismatch, non-finite age, inconsistent timestamps and ages outside the reference's existing limits. Preserve scanner JSONL and process evidence on failure. |
| `scripts/run_drawer_verdict_cycle.py`, `controller_args` and `consume` | Always enable the required-L2 flag for recurring cycles, require a P0 manifest and require a successful controller validation marker. | Prevent accidental omission of required market evidence. Preserve at-most-once claims and the existing pre/post drawer freshness and hash checks. |
| `scripts/run_drawer_verdict_cycle.py`, `health` | Require the validation marker on saved complete receipts. | A pre-repair receipt can have `complete` status despite unavailable L2. Such a receipt must not become current after deployment of the fix. It requires a fresh capture. |
| `tests/unit/test_live_evidence_publication.py` | Add slow-first-bot and successful-process/invalid-evidence regressions, plus valid-warning and legacy-mode checks. | Before the implementation, nine tests failed for the observed defects and two compatibility controls passed. |
| `tests/unit/test_drawer_verdict_cycle.py` | Require the new proof in simulated successful reports and test absent manifest, absent proof and old-receipt rejection. | Verify the wrapper enables the gate and cannot publish an unvalidated result or replay a rejected claim. |
| `tests/unit/test_chrome_plugin_telemetry_ingest.py` | Supply a synthetic P0 manifest and mock only scanner process output, retaining actual controller validation. Add a null-L2 output case. | Full ingestion-to-receipt integration proves a successful process with missing required L2 publishes nothing, appends no PnL history and cannot be replayed. Valid evidence still commits once without dispatching any action. |
| `docs/recurring_drawer_verdict.md` | Document the gate, per-bot time semantics and receipt compatibility. | Operators can distinguish a batch timestamp from evaluation time and understand why old receipts are blocked. |

## Integrity boundaries

- L2 maximum age remains 15 seconds; its future tolerance remains five seconds.
- Every drawer retains the 900-second limit; scheduling remains every 600 seconds.
- `src/neutralgrid/live/decision/l2_risk.py` and the recommender are unchanged.
- No model artifact, feature definition, trading threshold or training pool is changed.
- Missing utility, profile and linkage information is not replaced with invented values.
- Private signed-event authentication remains a separate source limitation.
- Public L2 is required for this workflow; the legacy controller remains opt-in
  to the stricter publication behavior for backward compatibility.
- The guard validates L2 at each recorded evaluation time. The 15-minute health
  window is an advisory snapshot lifetime, not continuous L2 surveillance.
- A rejected scanner run can already have written scanner history. This repair
  blocks controller publication and downstream PnL appends/action routing; it
  does not introduce a transaction or reset historical counters. Rejected
  observations remain claimed and cannot be replayed.

## Verification evidence

Audit root: `outputs/audits/drawer_repair_20260925/`.

- `red_tests.log`: nine reproduced failures, two passing controls before repair.
- `focused_tests.log`: first post-repair run; all new regressions pass. An
  existing PnL hard-link test exceeded Windows path length in the long isolated
  directory. The full suite uses a shorter dedicated temporary directory.
- `pyright_changed.log`: zero errors and zero warnings with the checkout
  interpreter explicitly selected. A default-interpreter run had unresolved
  third-party imports; no dependencies were installed or suppressed.
- `full_tests.log`: 2,210 passed, two skipped, one old integration fixture
  failed because it omitted the newly required P0 manifest. The fixture was
  updated and expanded to exercise the real publication check.
- `integration_tests.log`: 65 passed after that fixture update.
- `full_tests_final.log`: **2,212 passed, two skipped, no failures** in 241.91
  seconds. This rerun includes the expanded ingestion/publication fixture.
- `pyright_source.log` and `pyright_final.log`: source-tree and changed-script
  type checking both report **zero errors and zero warnings**.
- `source_hashes_before.json`: pre-repair hashes for changed scripts and
  unchanged L2/recommender modules. Existing unrelated local changes were retained.

Live acceptance is recorded separately after tests; unit tests do not prove
browser availability, exchange connectivity or future scheduler permissions.

## Live acceptance result

On 2026-09-25, the authenticated Chrome helper captured a complete Working
roster and proved deployment identity before and after collection. The
resulting cycle was ingested and consumed exactly once; no historical or
incomplete cycle was replayed.

The controller and receipt recorded `required_evidence_validated=true`; every
evaluation had concrete L2 evidence and no L2-unavailable diagnostic. Existing
utility, profile, candidate-link, and private-event completeness limitations
remained explicit. The observation was `observational_only=true` and
`training_eligible=false`; no trading action or retraining occurred.

The raw captures and run-specific audit evidence remain local, under the
repository's ignored `Live/` and `outputs/` locations. Exact symbols, strategy
identifiers, timestamps, receipt IDs, and paths are intentionally excluded from
this document and from version control.
