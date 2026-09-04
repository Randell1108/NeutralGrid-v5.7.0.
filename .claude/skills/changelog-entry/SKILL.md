---
name: changelog-entry
description: Append a CHANGELOG.md entry in the project's existing narrative style. Use after landing any user-facing change, fix session (e.g. UTILFIX-01, FIXPIPELINE-01), gate parameter change, or feature decommission. Includes the standard fields the project uses: explicit file paths modified, AFML / Hudson & Thames citation slot, decision rationale, and backward-compatibility note for breaking changes.
---

# changelog-entry

## Purpose
The CHANGELOG is the durable record of decisions and their rationale. Recent entries (UTILFIX-01, FIXPIPELINE-01, GRIDFIX-001, FIXUTILITY-01) follow a consistent shape; this skill stamps a new entry in that shape so future readers can scan the history without parsing ad-hoc prose.

## Procedure
1. Read `CHANGELOG.md` to confirm the latest entry's heading style and section ordering.
2. Append a new entry under `[unreleased]` (or the appropriate version section) with:
   - **Heading**: `### <SESSION-ID> - <short title>` where `<SESSION-ID>` follows the existing convention (e.g. `UTILFIX-03`, `GRIDFIX-002`).
   - **Date**: today's date in `YYYY-MM-DD`.
   - **Summary**: one paragraph stating what changed and why.
   - **Files modified**: bullet list with exact file paths (no globbing).
   - **Decision rationale**: brief justification, citing AFML chapter / Hudson & Thames where relevant.
   - **Backward compatibility**: state explicitly if any behaviour changed in a way callers must adapt to; otherwise write "No breaking changes".
   - **Verification**: how the change was tested (`pytest`, `pyright`, smoke run, contract test).
3. Cross-link any related ERR-### entries (`see ERR-042`).
4. Match the existing tone: factual, no emojis, plain ASCII.

## Style notes
- Cite Lopez de Prado AFML or Hudson & Thames where the decision is grounded in those references - recent entries do this.
- Document gate parameter changes (e.g. `top_quantile` 0.75 -> 0.68) with the empirical justification, not just the new value.
- Document feature decommissions with theory, evidence, and the mis-specification hypothesis.

## Refuse / fail-closed
- Refuse to omit the file-paths list. Reviewers rely on it.
- Refuse to write a vague entry ("misc fixes", "various improvements"). The CHANGELOG is the record; vagueness defeats it.

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 (dry-run vs CHANGELOG entry shape; ERR-085 fixed the FIXUTILITY-02 exemplar to FIXUTILITY-01 and removed a stranded memory citation) -->
