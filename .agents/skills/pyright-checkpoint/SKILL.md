---
name: pyright-checkpoint
description: Run pyright in the project's basic mode and surface the project-specific fixes documented in CLAUDE.md. Use after each code-modifying step in a multi-step implementation. Highlights the recurring patterns (cast(pd.Series, df[col]), np.asarray() over .values, isinstance numpy bool/numeric, tz-aware NaT, Optional guards) so violations get the right fix instead of a workaround.
---

# pyright-checkpoint

## Purpose
Pyright in basic mode catches most type issues, but a handful of project-specific patterns recur often enough that the standard error message is unhelpful. This skill runs the type check and translates each error into the canonical project fix.

## Procedure
1. Run pyright over the source tree:
   ```powershell
   pyright
   ```
   (Configured in `pyproject.toml`: basic mode, `pythonVersion=3.11`, `include=["src"]`.)
2. For each error, classify and recommend:

   | Error pattern | Project fix |
   |---|---|
   | `df["col"]` in a context expecting `pd.Series` | `cast(pd.Series, df["col"])` (import `cast` from `typing`). |
   | `Argument of type "ndarray" is not assignable...` from `.values` | Use `np.asarray(series)` instead of `series.values`. |
   | `np.bool_` failing `isinstance(x, bool)` | `isinstance(x, (bool, np.bool_))`. |
   | numpy scalar where `int`/`float` expected | `isinstance(x, (int, float, np.integer, np.floating))`; wrap returns with `float(...)`. |
   | tz-aware/naive mixing on a NaT initialiser | `pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")`. |
   | `Optional[T]` attribute access | Add `is not None` guard or assert. |
   | Series-as-bool ambiguity | `if bool(s.any()):` or `if not df.empty:`. |

3. Apply the recommended fix where it cleanly maps. Do not paper over with `# type: ignore` unless the user explicitly authorises it.
4. After fixes, re-run pyright. Report count: errors before / errors after.
5. If unrelated lint issues appear (`scripts/lint_no_pandas_values.py` enforces "no .values"), mention them but do not auto-fix outside the user's stated scope.

## Refuse / fail-closed
- Refuse to silence errors with `# type: ignore` unprompted.
- Refuse to declare PASS if errors remain.

## When to skip
- If the user is mid-edit on an unrelated file, do not run pyright over the whole tree just to satisfy this skill - narrow to the changed files.

## Verification
See `.claude/rules/skill-verification.md`.

<!-- Verified: 2026-07-10 (exercised live: pyright 0 errors on this session's edits; stranded memory citation removed per ERR-085) -->
