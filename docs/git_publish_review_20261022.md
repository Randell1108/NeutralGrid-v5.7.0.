# Architectural publication review - 2026-10-22

The operator requested review, validation, ordered commits and a push of the
architectural changes, excluding temporary files, unnecessary scripts and
isolated tests. The starting commit was
`a6ea01bb09719a7d2f96127c1c6e6a589a60eaad` on `main`. A fresh fetch confirmed
zero commits of divergence from `origin/main` before publication.

## Commit order and review scope

The working tree contains accumulated edits. Existing dated changelog entries
establish the documented work sequence; they do not establish when every line
was written. Commit complete current changes in dependency order without
backdating or reconstructing intermediate implementations:

1. Brave telemetry acquisition, renamed bridge/ingester, strict browser source
   checks, bounded drawer reads and capture, and endpoint ownership checks.
2. Immutable original premium-index observations, explicit causal replay,
   scanner provenance and extractor integration.
3. Exact deployment joins, duplicate/conflict handling, calibration cohort and
   chronological label checks, per-bot adapter timestamps and wrapper inputs.
4. Windows access-denied owner resolution using two complete native process
   inventories; uncertain owners remain unknown.
5. Source-checked research cohorts and immutable exact-bot private ledgers.
   These remain ineligible for training and live influence.
6. Operating documentation, accumulated changelog and this review record.

Review covered the production diffs, new evidence modules and CLIs, renamed
entry points, regression coverage and operating documentation. Necessary CLI
entry points and regression tests in the existing test suite accompany their
architecture. An additional operational-script type check found a missing
Optional narrowing in the Brave ingester. The review added an explicit
non-None guard and reused the already validated investment list. Valid scope
behavior and all numerical limits remain unchanged.
The active operating guide was updated to describe the wrapper's newly forwarded,
optional private-event input; evidence is not attached automatically.
Existing historical changelog claims are retained as prior session records;
model performance, historical live runs and workbook repairs were not rerun.

## Validation performed

- Full Python suite: **2,468 passed, 2 skipped, 1 failed** in 199.15 seconds.
  Command: `.venv/Scripts/python.exe -m pytest tests/ -q
  --basetemp=.tmp_pytest/git_publish_20261022 --tb=short`.
- The failure is
  `tests/unit/test_utility_calibrator.py::test_canonical_workbook_exposes_governed_utility_pool`.
  The canonical workbook contains HMM lineages
  `rolling_180d_20261003_024008`, `rolling_180d_20261011_025309` and
  `rolling_180d_20261015_134647`. The unchanged loader rejects this mixture.
  The same class of failure is documented in earlier review/changelog records.
  The suite is **not fully green**. No workbook or lineage gate was changed.
- Configured source Pyright with explicit repository interpreter:
  **0 errors, 0 warnings**. The initial invocation could not resolve dependencies
  through the default interpreter; the explicit-interpreter run succeeded.
- After the ingester guard repair, its **31 regressions passed**. The full-suite
  totals above precede this small guard repair.
- The complete source tree plus repaired ingester passes Pyright with zero
  errors and warnings. All changed operational Python entry points pass with
  **0 errors and 1 existing unused-import warning** in
  `scripts/validate_pnl_evidence_runtime.py`. That unrelated import is preserved.
- Drawer Node regression suite: **76 passed**, no failures or skips. Its initial
  sandbox run could not create a directory junction; the unrestricted rerun
  exercised the containment test successfully.
- PowerShell endpoint ownership and availability regression: **5 checks passed**.
- `node --check` passed for both browser JavaScript modules.
- Staged whitespace checks pass after removing an extra blank line at the end
  of the new evidence-cohort regression file.

Logs are local ignored outputs:
`outputs/git_publish_20261022_pytest.log`,
`outputs/git_publish_20261022_pyright.log` and
`outputs/git_publish_20261022_node.log`. Additional checks are retained in
`outputs/git_publish_20261022_script_pyright.log` and
`outputs/git_publish_20261022_ingest.log`.

## Deliberate exclusions

The following remain local and are not part of the architectural commits:

- Generated model activation/evaluation metadata: `artifact_manifest.json`,
  `models/meta_labeler_promotion_decision.json`,
  `models/meta_labeler_verification.json`. Model rotation is outside this
  architectural publication; ignored model binaries are also excluded.
- The already tracked temporary helper `tmp_scan_ports.py`.
- Generated test workspaces `.pt_c84e/`, `.rt21/`, `.rt21r2/`, `.rt21r2c/`,
  `t/` and `tests/.funding_tmp21a/`, plus ignored runtime outputs and live data.
- Superseded prompt/report files
  `docs/operational_prompt_10minute_snapshots_and_lifecycle_20261019.txt`,
  `docs/validation_10minute_snapshot_prompt_20261019.txt` and
  `docs/validation_operational_prompt_standalone_20261019.txt`.
  The active standalone operating guide is included because the recurring
  workflow explicitly references it.

The working tree is expected to remain dirty because these files are preserved.
Validation used the current local model/data environment; model binaries and
the canonical workbook are not supplied by these commits. No live collector,
scanner, exchange action, model training or promotion was started by this review.
The Git CLI push is blocked by missing local HTTPS credentials. The connected
GitHub app is authenticated as Randell1108 and reports push permission for the
same repository. Publication uses that connection to upload the reviewed Git
blobs and trees and create six commits in the same order. Server-created commit
metadata changes their commit hashes; every resulting tree must match its local
reviewed tree before main is advanced with an expected-head check. Local history
is aligned only after fetching and verifying the final remote tree. Publication
completion and the resulting remote commit are reported after that verification.
