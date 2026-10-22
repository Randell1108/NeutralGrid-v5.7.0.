# Brave browser migration review - 2026-10-16

The repository browser workflow now targets Brave. Source searches found no
remaining Chrome/Chromium references in `scripts/`, `src/`, `tests/`, or
`tmp_scan_ports.py`. Historical changelog, incident, audit and capture records
retain their original browser attribution.

## Verified browser observations

- The running Brave main process was PID 19356, with executable
  `C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe`.
- Its observed main-process command line contained no remote-debugging flag.
- The browser-tool inventory exposed only the in-app browser and MCP Apps;
  neither a Brave connection nor an authenticated Brave tab was exposed.
- No live Brave capture, authenticated Binance state, roster, or live ingestion
  is claimed. This migration did not start collectors or execute trading actions.

## Changes reviewed

- `scripts/start_private_telemetry_loop.ps1` uses the observed Brave executable
  and a separate `NeutralGrid\PrivateTelemetryBrave` profile. Before reusing a
  debugging endpoint, it requires loopback listeners owned by that executable.
  A different browser occupying the port blocks startup. Existing collectors
  cannot bypass this ownership check. Background launches use hidden windows.
- `scripts/extension_brave_execution_bridge.mjs` replaces the former Chrome
  bridge. Its local RPC provider identifier and the Python consumer now agree
  on `brave-extension`. The token, loopback, exact approval, identity and
  idempotency checks remain in place.
- `scripts/ingest_brave_plugin_telemetry_cycle.py` replaces the former Chrome
  ingester. Capture and ingestion agree on
  `neutralgrid_brave_plugin_capture_bundle_v1` and `brave_plugin`; staging and
  audit defaults use `brave_plugin_capture` and `brave_plugin_telemetry`.
- The controller, advisory consumer, volatility roster loaders, training
  loader, validation fingerprint paths and tests use the same Brave contract.
  Dedicated capture stamps `dedicated_brave_cdp` in both cycle and snapshot
  metadata. The controller rejects missing/unknown cycle sources and
  mismatched snapshot sources rather than treating them as dedicated captures.
- The P0 heading remains at line 37 of
  `outputs/audits/live_data_acquisition_prompt_20260910.md`. Its first instruction
  now requires a verified Brave connection and references the Brave ingester.
  The accompanying recurring-workflow and follow-on prompt use Brave.
- The recurring instructions require browser inventory evidence and documented
  APIs; they do not assume an installed Brave skill or a browser ID.
- `tmp_scan_ports.py` reports a discovered CDP endpoint as browser identity
  unverified. A DevTools version label alone is not Brave identity evidence.

## Validation results

| Check | Result |
| --- | --- |
| Baseline focused Python suite | 146 passed |
| Baseline and migrated drawer Node suite | 31 passed in each run |
| Baseline and migrated configured source Pyright | 0 errors, 0 warnings |
| Final focused Python suite | 150 passed |
| PowerShell endpoint tests | 5 checks passed; missing ownership check reproduced before implementation |
| Both browser JavaScript modules | `node --check` passed |
| Eight changed operational Python scripts | Pyright: 0 errors, 1 existing unused `sys` import warning in `validate_pnl_evidence_runtime.py` |
| Final full Python suite | 2,371 passed, 2 skipped, 1 failed |
| Source-reference and whitespace scans | No remaining Chrome/Chromium source references; `git diff --check` passed |

The final full-suite command was:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q --tb=short --basetemp .pt_brave_20261016_final
```

Its remaining failure is
`tests/unit/test_utility_calibrator.py::test_canonical_workbook_exposes_governed_utility_pool`.
The unchanged calibration code rejects the canonical workbook's mixed lineage:
`rolling_180d_20261003_024008` and `rolling_180d_20261011_025309`.
The workbook, calibration implementation, test and existing model/artifact
changes were not modified by this migration. The full suite is not green.

The first full run also failed the Windows long-path test during its setup
(`padding == -5`). That test passed in the final run with the shorter temporary
directory; its implementation and test code were unchanged.

Detailed logs:

- `outputs/audits/brave_migration_focused_20261016.log`
- `outputs/audits/brave_migration_pytest_20261016.log`
- `outputs/audits/brave_migration_pytest_final_20261016.log`

## Compatibility and operational limit

The former Chrome-named Python/JavaScript entry points were renamed, with no
compatibility wrappers. Update external commands to the new filenames. New
source identifiers, bundle schema and capture/audit defaults also changed.
Historical bundles and source-less cycles are not relabeled or admitted by
the migrated controller; obtain fresh Brave evidence.

The separate Brave debugging profile requires its own Binance authentication.
Authentication in the already-open normal profile does not establish that
authentication. The extension workflow remains blocked until the browser tool
actually exposes a connection verified as Brave and the required Binance tab.

## Requested read-only live validation

A subsequent connection check and one-shot capture attempt at 13:37 Lima on
2026-10-16 remained blocked: Brave was absent from the browser-tool inventory,
port 9222 had no listener, and the updated collector timed out fetching
`http://127.0.0.1:9222/json/list`. Its isolated manifest records zero completed
cycles and `finish_reason=one_shot_failed`. No live capture or ingestion is
claimed. See
`outputs/audits/brave_live_capture_connection_20261016/LIVE_CAPTURE_REPORT.md`
and the adjacent `manifest.json` for the observed state and exact failure.
