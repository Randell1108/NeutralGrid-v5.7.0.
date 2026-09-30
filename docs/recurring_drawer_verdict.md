# Recurring Chrome drawer capture and advisory verdicts

Run this workflow after the P0 start-and-verify section at line 37 of
`outputs/audits/live_data_acquisition_prompt_20260910.md`.

## Authority and cadence

`live_decision_scanner.py` owns CONTINUE / ADJUST / END recommendations.
`scripts/run_live_telemetry_controller.py` joins the drawer cycle to runtime
evidence. `scripts/run_drawer_verdict_cycle.py` adds one-consumption receipts,
failure reporting, and a freshness-aware health view around that controller.
The Chrome extension owns browser acquisition; Python never obtains browser
credentials or falls back to the separate debugging profile.

Schedule one capture-and-consume attempt every **10 minutes**, matching the
controller's default 600-second interval. This workflow supersedes the older
P3 dedicated-profile drawer worker for this session. Keep the public/market
supervisor from P0 running. The maximum age remains the controller's default
**900 seconds**, checked against every individual capture before and after the
scanner. L2 retains its existing 15-second age gate. These are existing runtime
limits, not newly calibrated trading thresholds.

This is periodic observation, not an event-complete stream or a real-time
scheduler guarantee. The computer, desktop app, Chrome extension, authenticated
Binance page and network must remain available. Delays can make the last
observation stale. Read `--status` before describing a saved verdict as current.
See the official [scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app)
for local-host availability and tasks that return to an existing conversation.

## One scheduled run

1. Work in `D:\Neutral Grid v5.7.0`. Read the Chrome skill and use its documented
   browser-client interface through the persistent JavaScript tool. Reuse the
   existing Chrome binding when available. Otherwise initialize the runtime
   using the installed skill's absolute module path, select Chrome, and read
   its complete documentation. List and claim the existing authenticated
   Binance futures-grid tab. Do not use an in-app browser or a CDP fallback.
2. Inspect the current page. Require UM Grid and Running to be selected and
   prove the complete Working roster from the rendered table. The table may
   virtualize rows; the helper makes a bounded top-to-bottom sweep and requires
   overlapping views, the tab count, unique symbols, Working status, and stable
   deployment times. Do not assume the last roster still applies. Never click
   End, adjustment, order, leverage or margin controls.
3. Read `scripts/capture_recurring_drawers.mjs`, then import it through the
   persistent JavaScript tool and call it with the claimed tab:

   ```javascript
   const drawerCollector = await import('file:///D:/Neutral%20Grid%20v5.7.0/scripts/capture_recurring_drawers.mjs');
   const captured = await drawerCollector.captureDrawers(gridTab, {
     workspaceRoot: 'D:/Neutral Grid v5.7.0'
   });
   console.log(captured);
   ```

   Binding names are examples; reuse existing bindings or choose unused names.
   The helper verifies a fresh View Details tooltip before clicking its observed
   document icon. It closes only the observed drawer close icon. It checks the
   complete roster before and after, including deployment times, and saves raw
   text before parsing. For virtualized rows it scrolls through overlapping
   table views, rejects gaps or changed identities, and reveals each proven row
   before its drawer. If the UI changes, a complete sweep cannot be proved, an
   identity is ambiguous, or a page request times out, stop the attempt. Do not
   weaken the checks to make a cycle pass.
4. Ingest only the `bundlePath` returned by that successful call:

   ```powershell
   .\.venv\Scripts\python.exe scripts\ingest_chrome_plugin_telemetry_cycle.py --bundle-manifest '<RETURNED_BUNDLE_PATH>'
   ```

   Require exit code zero and a complete result. Use its exact `cycle_manifest`
   path for the next step. Do not select a historical manifest by filename.
5. Verify the P0 public collector's current manifest covers the newly observed
   roster. Missing active symbols need the existing operational P0
   roster-reconciliation procedure. Extra public-market symbols are not active
   bot authority: report them and consume only the proven Working roster.
   Strategy-attributed streams must match the current deployment identity.
   Do not silently remove the L2 input to
   work around a coverage or freshness failure. Do not restart healthy services
   on every tick. A failed signed private stream remains unavailable, never
   an observed zero or evidence of event completeness.
6. Run the read-only pipeline preflight before the first scanner use and after
   an artifact change. Preserve its failures/warnings. Do not retrain, rotate,
   backfill, calibrate or promote artifacts as part of this workflow. Then run:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_drawer_verdict_cycle.py --cycle-manifest '<RETURNED_CYCLE_MANIFEST>' --diff-depth-manifest outputs\audits\binance_websocket_services_current\public\manifest.json
   .\.venv\Scripts\python.exe scripts\run_drawer_verdict_cycle.py --status
   ```

   The wrapper always uses plugin-manifest, observational-only and single-tick
   modes, with Discord disabled by the existing controller. It exposes no
   action-execution or model-training switches. Do not call the scanner a second
   time directly or run an overlapping legacy controller against this state.
   It also requires the P0 L2 manifest and enables `--require-l2-evidence` in
   the controller. A zero scanner exit code is insufficient: every bot must
   have an exact-run L2 evaluation, valid timestamps and no L2-unavailable
   diagnostic before the controller publishes results. Utility and linkage
   warnings remain recorded without silently replacing missing artifacts.
7. On any browser, ingestion or consumer failure, record the failure explicitly:

   ```powershell
   .\.venv\Scripts\python.exe scripts\run_drawer_verdict_cycle.py --capture-failure 'Concise observed failure; reference its evidence path'
   ```

   Quote shell arguments safely; do not interpolate captured page text into a
   command. The failure command clears the current published verdicts. Retain
   raw evidence and the detailed error record. Do not replay an already claimed
   cycle after a timeout/interruption. Acquire a new cycle on the next scheduled
   attempt. Do not loop, increase polling frequency, or bypass a browser limit.
8. Compare the health result with the last reported state. Report a changed
   roster, verdict, materially changed reason, failure, stale condition or
   recovery. Remain quiet when nothing actionable changed. Include exact
   symbol/strategy, capture time, verdict, reasons, evidence path and unavailable
   sources when reporting. ADJUST and END remain recommendations for user review.

The P0 public diff-depth collector can pre-create an empty aggregate-trade file
while `collect_agg_trades=false`. The controller attaches its L2 derivatives
without that disabled trade file only when both run and symbol manifests agree
on the disabled flag and the file is empty. Conflicting flags or nonempty
disabled evidence block the cycle. Enabled/legacy aggregate-trade attachments
still require the exact strategy target. Market-route aggregate trades are not
automatically treated as strategy-owned fills.

## Storage, deduplication and isolation

- Raw capture: `Live/<Lima-cycle-date>/<SYMBOL>/drawer_captures/<unique-run>/drawer.txt`.
  The ingestion date is resolved once at capture start, including across midnight.
- Capture bundle/failure metadata: `outputs/runtime/chrome_plugin_capture/<run>/`.
  This directory contains metadata, not newly captured bot raw text.
- Validated snapshots and PnL observations: existing `Live` ingestion/history paths.
- Validated cycle manifests: `outputs/audits/chrome_plugin_telemetry/cycles/`.
- Consumer receipts, health, controller reports, registry and separate scanner
  history: `outputs/audits/recurring_drawer_verdict/`.

The observation key includes exact symbol, strategy ID, capture timestamp and
raw SHA-256, independent of manifest filename or ordering. An exact repeated
observation cannot advance scanner history again. Per-bot observation claims
also reject an old observation inserted into a different roster bundle; a new
symbol cannot make reused evidence count again. A later genuine capture is
a different observation even if its PnL is unchanged. Do not manually accelerate
the scheduled scanner cadence: consecutive-evaluation counters are meaningful
only with the existing evaluation schedule.

An OS-owned lock prevents overlapping consumers and releases after process
death. The receipt is claimed before evaluation. A crash leaves it incomplete;
that observation is not replayed. Atomic receipts prevent truncated success
files. A failure can still leave valid raw data, PnL observations, or partly
advanced scanner history from the underlying controller; this is an at-most-once
consumer, not a transaction spanning every controller file. The health receipt
is the publication authority, and blocks the result on any detected failure.

The separate state directory starts with no inherited decision counters. This
avoids mixing legacy controller runs into the new collector's history; the
existing persistence requirements rebuild from subsequent valid observations.
Live observations are classified `live_bot_telemetry`, `training_eligible=false`.
No training-pool admission or model fit is performed. That classification is
an evidence label, not a replacement for the training pipeline's own gates if
the data is deliberately used for a future research task.

`--status` returns no verdict for missing, corrupt, failed, future-dated or
older-than-900-second evidence. It marks cadence overdue after 600 seconds while
remaining current only within the unchanged 900-second age limit. Archive
receipts and controller reports remain historical; do not treat their saved
`complete` status as a current health check.

Receipts created before required-L2 validation was added have no validation
marker and cannot appear current. Acquire a new cycle; do not replay or edit
an old receipt. L2 age is validated relative to each bot's evaluation time.
The scanner now takes that time immediately before evaluating each bot, while
retaining the batch timestamp for scheduling and top-level output grouping.
This prevents normal heartbeat updates during an earlier bot's evaluation from
being misclassified as future observations for a later bot. The existing
15-second age and five-second future tolerance are unchanged. The health view
is an age-limited advisory snapshot, not continuous L2 validation between runs.

## Boundaries that require observation, not inference

The helper requires a complete Working roster in the rendered table or across
a bounded overlapping sweep of its own scroller. It fails on a partial or
changing virtualized roster rather than silently omitting bots. It has a
four-minute between-drawer deadline and bounded UI waits, but a
browser transport outage can exceed an individual UI timeout. Scheduler sleep,
browser authentication, website changes and exchange availability cannot be
proven away by local tests. Missing utility/profile artifacts and missing signed
event authority remain existing limitations; this integration does not fill
them with defaults or promote unapproved candidates.

To disable this workflow, pause its existing Codex heartbeat. Do not stop the
P0 market services unless separately requested. If the old dedicated-profile
drawer loop is still running, retire it through
`scripts/stop_private_telemetry_loop.ps1`, not by killing unrelated browser or
Python processes.
