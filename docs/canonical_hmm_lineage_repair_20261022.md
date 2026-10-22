# Canonical HMM lineage repair - 2026-10-22

This document records the completed local workbook repair. Only this document and
its changelog entry are published to Git. Supporting JSON, CSV, previews and test
logs referenced below remain local under
`reports/canonical_hmm_lineage_repair_20261022T191156Z/`; filenames without a
directory refer to that evidence directory. The workbook, backup, audit helpers
and temporary files are excluded from publication.

The canonical workbook has been repaired using 853 hash-verified October 21
staging rows and four fresh, independently reproduced HMM replays. All 857
strategies now carry `rolling_180d_20261021_161805` in both General and Meta
Features, with finite range, trend and persistence probabilities.

## Backup and publication

- Canonical: `D:/Neutral Grid v5.7.0/data/new_expired_bots.xlsx`.
- Original backup:
  `D:/Neutral Grid v5.7.0/data/backups/canonical_hmm_lineage_repair_20261022T191156Z/new_expired_bots.before.xlsx`.
- Original and backup SHA-256:
  `85f56530c246f5ccd7bf80819281611fd1dc34229aa175866d7fb677d6f03854`.
- Published SHA-256:
  `1acf2e7a74a210b84c83085ea800fe219791b910558852490ad636ac09f7e651`.
- Published at `2026-10-22T19:29:30.374902+00:00` after candidate validation.

The original five-sheet workbook was backed up before any edit. Publication used
an atomic replacement only after confirming that the canonical still matched the
backup hash. `publication.json` records the completed operation.

## What caused the four failures

The previous command combined `--hmm-only` with an explicit
`--feature-cutoff-source start_time_utc`. The existing backfiller's constructor
switches start-time mode to candidate scan time whenever `--hmm-only` is set.
These four candidate IDs contain scan times after their recorded bot starts,
so the causal cutoff guard rejected their staging replays.

| Strategy | Symbol | Recorded start UTC used for replay | Later candidate scan UTC |
|---|---|---|---|
| 410524706 | NEARUSDT | 2026-03-08 09:44:23 | 2026-03-09 01:32:43 |
| 410899699 | FOLKSUSDT | 2026-03-26 10:10:41 | 2026-03-26 14:57:43 |
| 410899682 | ONDOUSDT | 2026-03-26 10:09:46 | 2026-03-26 14:57:43 |
| 410899660 | ROBOUSDT | 2026-03-26 10:08:53 | 2026-03-26 14:57:43 |

These are cutoff-selection failures, rather than evidence that the stored starts
should be changed. Each recorded start agrees across General, Meta Features,
Last Features, the frozen October 21 input, and the September 19 recorded-start
audit. That audit is recorded-source corroboration, not an independent exchange
creation-time receipt. No new timestamp, timezone offset or candidate identity
was inferred. The existing candidate IDs remain unchanged; this repair does not
establish them as causal deployment linkages.

The four rows were replayed through the unchanged
`scripts/backfill_training_features.py` entry point with:

```text
--default-artifact-version rolling_180d_20261021_161805
--feature-cutoff-source start_time_utc
--replay-scope hmm_lineage_only
--require-fresh-output
```

The `--hmm-only` and `--skip-if-fresh` flags were omitted. The input and output
were separate CSV staging files. Each replay retained 800 completed 15-minute
candles. Independent inference from those retained candles reproduced all three
probabilities exactly, with maximum absolute difference zero. The latest candle
close preceded each recorded cutoff. `preparation.json`, `replay_command.json`,
`replay_windows.json` and `replay_validation.json` retain the evidence.

## Merge and preservation checks

- The October 21 staging workbook and its frozen input matched their recorded
  run-index SHA-256 values. All 857 strategy identities, symbols, stored starts
  and candidate IDs matched the canonical source.
- The preserved EV inputs matched the frozen source, so the 853 staged
  HMM-transitive EV results could be reused without substituting newer inputs.
- Only HMM-derived probabilities, lineage/cutoff metadata, EV and regime
  confidence were updated where their values differed. Independent features,
  labels, PnL, funding, fees, outcomes, timestamps and candidate IDs were preserved.
- Exactly **13,914 cells** changed: 6,957 in General and 6,957 in Meta Features.
- All **118,400 cell positions** were compared against the backup. Unplanned
  values, formulas, styles, comments and hyperlinks are unchanged.
- All **27,424 HMM-field values** in the published workbook exactly match the
  selected validated staging or four-row replay sources. No numerical tolerance
  was needed for that final source comparison.
- Only `xl/worksheets/sheet1.xml` and `xl/worksheets/sheet3.xml` changed. Every
  other package member remains byte-identical, including the other three sheets,
  tables, styles, shared strings, relationships and workbook metadata.
- Artifact Tool authored and inspected the cell updates. A package-preserving
  cell transplant retained the original workbook features and formatting.
  Before/after previews were reviewed. Existing clipped lineage headers were
  preserved without restyling. Temporary string-literal formulas used to prevent
  ISO timestamp coercion were converted to verified literal text; none were
  published as formulas.

| Sheet | Data rows after repair | Preservation |
|---|---:|---|
| General | 857 | Only planned HMM-related cells changed |
| PnL Curve Features | 659 | Byte-identical worksheet |
| Meta Features | 857 | Only planned HMM-related cells changed |
| Last Features | 857 | Byte-identical worksheet |
| Canonical Audit | 531 | Byte-identical worksheet |

## Runtime validation

The reopened canonical workbook passes the existing governed utility-pool loader.
It admits **703 rows**, all under the active HMM, with 415 positive outcomes and
288 negative outcomes. The prior staging pool admitted 700 rows because three
of the four failed replays were otherwise eligible; NEAR remains outside the
existing seven-hour duration limit. No eligibility or risk threshold was changed.

The full existing Python suite run after the workbook repair recorded
**2,469 passed, 2 skipped, zero failures**, in
174.64 seconds. The formerly failing
`test_canonical_workbook_exposes_governed_utility_pool` passed as part of that
suite. Command:

```text
.venv/Scripts/python.exe -m pytest tests/ -q --tb=short --basetemp=.tmp_pytest/hmm_repair_20261022
```

`full_pytest.log` retains the complete result. `canonical_pool_validation.json`,
`validation.json` and `full_source_comparison.json` retain the reopened loader,
package/cell preservation and exact source comparison checks.

## Boundaries

The active HMM, meta-labeler and all fingerprinted model artifacts are unchanged.
No model was retrained, calibrated, promoted or enabled, and no live collector,
scanner, deployment or exchange action was started. This repair establishes
workbook lineage consistency; the previously rejected utility/profile promotions
remain separate issues. Runtime fail-closed gates and feature schemas are intact.

Audit helpers, retained market windows, intermediate workbooks and logs remain
under the ignored evidence directory identified above. The workbook and backup
remain local data; neither is uploaded to Git.
