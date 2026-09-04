# Live volatility forecast v1 runbook

This runbook operates only the shadow realized-volatility path defined by
`config/live_volatility_forecast_v1.json`. It does not change active scanner
verdicts, grid bounds, sizing, Kelly inputs, model lineage, or execution.

## 1. Commit a fresh authenticated Working roster

Use the Chrome extension to export one complete private-telemetry bundle. The
bundle must include the roster fence and every exact strategy identity. Then
validate and commit it:

```powershell
python scripts/ingest_chrome_plugin_telemetry_cycle.py `
  --bundle-manifest <chrome-plugin-bundle-manifest.json> `
  --audit-dir outputs/audits/chrome_plugin_ingest
```

Record the `cycle_manifest` and `collector_targets_csv` paths printed by the
command. Do not substitute a manually assembled symbol list when the intent is
to acquire the current authenticated Working roster. The cycle must be no more
than 900 seconds old when historical acquisition starts.

## 2. Start prospective public event collection

Start or reconcile the checkout-owned collector from the fresh cycle manifest:

```powershell
python scripts/supervise_diff_depth_collector.py `
  --cycle-manifest <cycle_manifest>
```

This records diff depth, aggregate trades, and one-second mark-price updates.
The supervisor binds ownership to both the checksum-verified roster target and
the current collector-script SHA-256. It replaces an older checkout-owned
collector when either hash changes and reports healthy only after every symbol
has a contiguous depth segment, acknowledged market subscriptions, and at least
one received mark-price event without sequence, coverage, or parse gaps.
The collector uses Binance's documented `/public` endpoint for depth and its
separate `/market` endpoint for aggregate trades and mark price; it requires an
actual mark event in addition to the subscription acknowledgement.
It is prospective: it cannot reconstruct public events from before startup and
does not represent private orders, trades, or transactions.

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
means that no `(symbol, horizon)` passed both the frozen-holdout QLIKE gate and
the mandatory interval-quality gate. Do not launch inference from such a run.

## 5. Smoke-test and start the three-minute shadow loop

Confirm that the run metadata reports an eligible forecast, then perform one
fail-closed cycle:

```powershell
python scripts/run_live_volatility_loop.py `
  --once `
  --interval-seconds 180 `
  --contract config/live_volatility_forecast_v1.json `
  --artifact-dir outputs/audits/live_volatility/<run_id> `
  --roster-audit-root outputs/audits/chrome_plugin_ingest
```

If the smoke cycle succeeds, start the unattended loop by removing `--once`:

```powershell
python scripts/run_live_volatility_loop.py `
  --interval-seconds 180 `
  --contract config/live_volatility_forecast_v1.json `
  --artifact-dir outputs/audits/live_volatility/<run_id> `
  --roster-audit-root outputs/audits/chrome_plugin_ingest
```

The plugin must continue committing fresh cycle manifests to the audit root.
Pinning `--cycle-manifest` is suitable for a one-cycle audit but will correctly
become unavailable once that roster exceeds its freshness limit. Each runtime
result is written under `Live/YYYY-MM-DD/<SYMBOL>/volatility/` and retains
`verdict_influence: false`.

## Optional read-only private telemetry loop

A separately launched Chrome with DevTools debugging can collect complete
visible grid-bot snapshots every 180 seconds:

```powershell
python scripts/collect_private_grid_telemetry.py `
  --debug-endpoint http://127.0.0.1:9222 `
  --interval-seconds 180
```

This collector is read-only and does not replace event-level private Order,
Trade, or Transaction History exports.
