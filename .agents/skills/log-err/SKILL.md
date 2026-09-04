---
name: log-err
description: Append a new ERR-### entry to ERRORS_LOG.md following the project's existing table format. Auto-increments the ID from the highest existing entry to prevent duplicates across parallel sessions. Use whenever a new error check, blocker, or watch item is discovered that requires durable tracking. Status vocabulary is OPEN / BLOCKED / WATCH / CLOSED.
---

# log-err

## Purpose
ERRORS_LOG.md is the canonical incident tracker. Entries use a sequential `ERR-###` ID. Multiple sessions editing in parallel can produce duplicate IDs; this skill enforces a grep-then-increment discipline.

## Procedure
1. Read `ERRORS_LOG.md`.
2. Grep for `ERR-\d{3}` and find the highest numeric ID currently present (whether OPEN, BLOCKED, WATCH, or CLOSED).
3. Increment by 1 to determine the new ID. If the result collides with an in-flight ID in another open editor, the user must reconcile manually.
4. Confirm the new ID with the user before writing.
5. Append a new row to the Active Error Checks table with these fields:
   - **ID**: `ERR-###` (zero-padded to 3 digits).
   - **Status**: `OPEN` (default for new entries; `BLOCKED` if dependent on an external resolution; `WATCH` for monitoring without immediate action).
   - **Area**: short tag (HMM artifacts, config, validation, pipeline, calibrator, meta-labeler, scanner, deployment, data ingestion, etc.).
   - **Error Check**: one-sentence description of what is wrong or being watched.
   - **Evidence**: a file:line reference, command output, or artifact path that demonstrates the issue.
   - **Required Action**: the next concrete step.
   - **Verification**: how closure will be confirmed.
6. Do not edit any existing rows. Closure is a separate operation (status change + verification note).

## Style notes
- No emojis; plain ASCII only.
- Plain ASCII only.
- Match the existing table column order; do not invent new columns.
- Reference related ERR entries in-line if relevant (e.g. "supersedes ERR-029", "blocked by ERR-034").

## Refuse / fail-closed
- Refuse to write if the chosen ID already exists in the file. Re-grep and increment again.
- Refuse to write if the user has not confirmed the chosen ID (avoids racing).

## Verification
See `.claude/rules/skill-verification.md`. After append, re-grep for the new ID and confirm exactly one occurrence in the Active Error Checks table.

<!-- Verified: 2026-07-10 (exercised live: ERR-077..ERR-089 appended in this format; stranded memory citation removed per ERR-085) -->
