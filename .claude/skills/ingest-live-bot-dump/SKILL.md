---
name: ingest-live-bot-dump
description: Operationalise the CLAUDE.md "Live Bot Data Storage Policy". Use when the user provides newly-extracted live bot data (active bot telemetry). Creates Live\<YYYY-MM-DD>\<SYMBOL>\ in strict format, distinguishes live vs expired bot data so each routes to the right downstream path, and never touches data/manual_input/ for live data. Flags legacy MM-DD folders without auto-migrating.
---

# ingest-live-bot-dump

## Purpose
Live bot data and expired bot data have different ingestion classes. Mixing them corrupts both pipelines. This skill enforces the policy in `CLAUDE.md` ("Live Bot Data Storage Policy") and routes correctly. As of 2026-07-10 (ERR-086) the authoritative Live root is THIS repository's `Live\` folder.

## Decision tree

### Is this live or expired bot data?
- **Live** = active bot telemetry (bot is currently running). Goes to `Live\<YYYY-MM-DD>\<SYMBOL>\`.
- **Expired** = a closed/terminated bot's history dump. Goes through the extractor pipeline -> `data/manual_input/<YYYY-MM-DD>/` -> `data/new_expired_bots.xlsx`.

If you are unsure, ask before placing files. Misclassification is the failure mode this skill exists to prevent.

## Procedure (live data)
1. Confirm with the user: this is active-bot telemetry, not a closed-bot dump.
2. Determine ingestion date (today, in `YYYY-MM-DD` strict - not `MM-DD`).
3. Determine symbol (Binance USDT-M futures pair, e.g. `BTCUSDT`).
4. Create the directory:
   ```
   Live\<YYYY-MM-DD>\<SYMBOL>\
   ```
   Under the project root, NOT inside `data/` and NOT inside `src/`.
5. Place the user-provided files only inside that folder. Do not duplicate elsewhere in the repo.
6. If sibling folders exist using legacy `MM-DD` format, FLAG them in the report but do not auto-migrate. Migration policy is the user's call.
7. Do not run downstream training/backfill on live data - the live pipeline is `live_outcome_ingestor`, not the expired-bot extractor.
8. Log linkage caveat: if the bot was started before any candidate scan that could match it, candidate_id linkage may be a post-deployment retrospective match. Surface the timing; do not block.

## Procedure (expired data)
1. Confirm with the user: this is a closed-bot dump.
2. Place input files in `data/manual_input/<YYYY-MM-DD>/`.
3. Run the extractor (`_bot_data_extractor_core.py` or the project's current extractor) to produce updated `data/new_expired_bots.xlsx`.
4. Run `backfill-features` skill against the updated workbook with `--default-artifact-version <active_hmm>`.
5. Run `verify-hmm-lineage` against the backfill output. Verdict must be PASS before any calibrator or meta-labeler refit.

## Refuse / fail-closed
- Refuse to place live data anywhere outside `Live\<YYYY-MM-DD>\<SYMBOL>\`.
- Refuse to place expired data inside `Live\`.
- Refuse to use `MM-DD` date format. Always `YYYY-MM-DD`.
- Refuse to auto-migrate legacy folders without explicit user confirmation.

## Critical files / paths
- `Live\<YYYY-MM-DD>\<SYMBOL>\` (live)
- `data/manual_input/<YYYY-MM-DD>\` (expired, pre-extraction)
- `data/new_expired_bots.xlsx` (expired, post-extraction canonical pool)
- `src/neutralgrid/training/live_outcome_ingestor.py` (live downstream)

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 (dry-run vs the repointed CLAUDE.md Live policy, ERR-086; stranded memory citations removed per ERR-085) -->
