# Live volatility forecast v1 runbook

This runbook operates only the shadow realized-volatility path defined by
`config/live_volatility_forecast_v1.json`. It does not change active scanner
verdicts, grid bounds, sizing, Kelly inputs, model lineage, or execution.

## 1. Commit a fresh authenticated Working roster

Use the Brave extension to export one complete private-telemetry bundle. The
bundle must include the roster fence and every exact strategy identity. Then
validate and commit it:

```powershell
python scripts/ingest_brave_plugin_telemetry_cycle.py `
  --bundle-manifest <brave-plugin-bundle-manifest.json> `
  --audit-dir outputs/audits/brave_plugin_ingest
```

Record the `cycle_manifest` and `collector_targets_csv` paths printed by the
command. Do not substitute a manually assembled symbol list when the intent is
to acquire the current authenticated Working roster. The cycle must be no more
than 900 seconds old when historical acquisition starts.

## 2. Start prospective public event collection

Follow the current P0 ownership procedure in
`docs/operational_prompt_standalone_20261019.txt`. Its single owner is
`scripts/supervise_binance_websocket_services.py`, with public diff depth,
market trades/marks/closed klines, and private account events in separate
services. Use the exact checksum-verified `collector_targets_csv` from step 1
to verify an existing owner:

```powershell
python scripts/verify_binance_websocket_services.py `
  --target-csv <collector_targets_csv> `
  --audit-root outputs/audits/binance_websocket_services_current
```

Do not start `supervise_diff_depth_collector.py` alongside the P0 owner.
Reconcile membership, process identity and any supported start/stop through the
current P0 procedure; do not force-kill processes or delete ownership evidence.
If collection is operator-stopped, leave it stopped. Public stream health does
not establish private event completeness. Read the per-service findings;
private connection failures remain explicit and are not an additional
volatility-model acceptance gate. Public event collection is
prospective and cannot reconstruct events from before startup. Raw mark ticks
also do not replace the finalized mark-price bars required by step 3.

## 3. Backfill finalized one-minute mark and last prices

Run the checksum-verified Binance Vision backfill using the exact manifest from
step 1:

```powershell
python scripts/backfill_volatility_history.py `
  --cycle-manifest <cycle_manifest> `
  --contract config/live_volatility_forecast_v1.json `
  --initial-days 90 `
  --extension-days 30 `
  --max-days 365
```

The command preserves one-minute observations in the canonical price store.
Five-minute sampling is applied only while constructing forward realized
variance. REST is limited to audited gaps and the archive-to-live tail. A
stale roster, checksum failure, conflicting duplicate, listing boundary, or
insufficient clean origins causes a nonzero exit and an explicit audit record.

The successful command writes the governed manifest at:

```text
outputs/audits/live_volatility_backfill/manifest.json
```

## 4. Train and evaluate a run-scoped shadow artifact

Only after step 3 succeeds, run:

```powershell
$runId = "rv_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python scripts/train_live_volatility_forecaster.py `
  --cycle-manifest <cycle_manifest> `
  --backfill-manifest outputs/audits/live_volatility_backfill/manifest.json `
  --contract config/live_volatility_forecast_v1.json `
  --run-id $runId
```

Training writes only to `outputs/audits/live_volatility/<run_id>/`. Exit code 2
means training was blocked or no `(symbol, horizon)` passed both the
frozen-holdout QLIKE gate and the mandatory interval-quality gate. Inspect the
error log and, when written, `oos_report.json`. Do not launch inference from
such a run.

## 5. Smoke-test and start the three-minute shadow loop

Confirm that the run metadata reports an eligible forecast, then perform one
fail-closed cycle:

```powershell
python scripts/run_live_volatility_loop.py `
  --once `
  --interval-seconds 180 `
  --contract config/live_volatility_forecast_v1.json `
  --artifact-dir outputs/audits/live_volatility/<run_id> `
  --roster-audit-root outputs/audits/brave_plugin_ingest
```

Inspect the cycle report and each forecast record. A completed cycle or zero
CLI exit status alone is not proof of forecast availability: per-bot failures
can be persisted as explicit `unavailable` records. Require an `available`,
fresh, eligible record for each symbol/horizon intended for activation, and
retain unavailable records for unsupported pairs. If this smoke validation
succeeds and live collection is authorized, start the unattended loop by
removing `--once`:

```powershell
python scripts/run_live_volatility_loop.py `
  --interval-seconds 180 `
  --contract config/live_volatility_forecast_v1.json `
  --artifact-dir outputs/audits/live_volatility/<run_id> `
  --roster-audit-root outputs/audits/brave_plugin_ingest
```

The plugin must continue committing fresh cycle manifests to the audit root.
Pinning `--cycle-manifest` is suitable for a one-cycle audit but will correctly
become unavailable once that roster exceeds its freshness limit. Each runtime
result is written under `Live/YYYY-MM-DD/<SYMBOL>/volatility/` and retains
`verdict_influence: false`.

## Authorized historical symbols while live collection is stopped

When the operator authorizes public historical acquisition for canceled symbols,
use the existing explicit-symbol mode. Record the exact historical identity
source and the operator's scope separately; do not fabricate or refresh a
Working-roster manifest. This route can acquire and evaluate price-series
evidence without starting public/private streams, browser capture or schedules.

```powershell
python scripts/backfill_volatility_history.py `
  --symbol <SYMBOL_1> --symbol <SYMBOL_2> `
  --contract config/live_volatility_forecast_v1.json `
  --initial-days 90 --extension-days 30 --max-days 365 `
  --audit-root outputs/audits/<historical_acquisition_run>/backfill

python scripts/train_live_volatility_forecaster.py `
  --symbol <SYMBOL_1> --symbol <SYMBOL_2> `
  --backfill-manifest outputs/audits/<historical_acquisition_run>/backfill/manifest.json `
  --contract config/live_volatility_forecast_v1.json `
  --run-id <fresh_shadow_run_id>
```

Use the training command only after acquisition succeeds and reports `ready`.
Both commands must use the same exact symbol set. All history, origin, split,
QLIKE and interval-quality gates remain unchanged. Preserve the complete
run-scoped model, metadata, examples, source manifest, split audit and OOS report.
An exit code 2 is a rejection/blocker, not permission to launch inference.

An offline inference check may use an explicitly recorded historical source
cutoff, but must be labeled as such and stored in the audit directory. It is
not a fresh live smoke cycle and must not create an active-roster claim or
write current bot advice into `Live/`. Private terminal status or final bot
PnL is not established by public price-series acquisition.

Keep live collection stopped until separately authorized. Live activation still
requires the fresh authenticated roster, eligible symbol/horizon coverage,
source freshness and smoke cycle in steps 1 and 5. A V1 artifact is supplied
to the existing telemetry consumer through its explicit
`--volatility-forecast-artifact-dir` option; it must not be registered as an
active HMM/meta-labeler artifact or allowed to affect verdicts or actions.

## Optional read-only private telemetry loop

A separately launched Brave with DevTools debugging can collect complete
visible grid-bot snapshots every 180 seconds:

```powershell
python scripts/collect_private_grid_telemetry.py `
  --debug-endpoint http://127.0.0.1:9222 `
  --interval-seconds 180
```

This collector is read-only and does not replace event-level private Order,
Trade, or Transaction History exports.
