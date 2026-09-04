#!/usr/bin/env python3
"""Lint guard: ban bare .values on pandas objects.

Usage:
    python scripts/lint_no_pandas_values.py [--fix]

Scans src/neutralgrid/**/*.py for `.values` attribute access on pandas
objects. The pattern `.values` (without parentheses) on Series/DataFrame
returns ndarray | ExtensionArray, causing Pylance type errors.

Correct alternatives:
    np.asarray(series)        # always returns ndarray
    series.to_numpy()         # pandas-native, returns ndarray

Dict .values() calls (with parentheses) are excluded.
"""

import re
import sys
from pathlib import Path

# Pattern: .values NOT followed by ( — i.e., attribute access, not method call
PATTERN = re.compile(r'\.values(?!\s*\()\b')

# Known false positives
FALSE_POSITIVE_HINTS = [
    '.values()',       # dict method call
    '# ',             # in a comment
    'values:',        # type annotation
    '"values"',       # string key
    "'values'",       # string key
]

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "neutralgrid"


def scan_file(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line_text) with violations."""
    violations = []
    try:
        lines = filepath.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return violations

    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        # Skip comments
        if stripped.startswith("#"):
            continue
        # Skip if it's .values() with parens (dict method)
        if ".values()" in line:
            continue
        # Check for bare .values
        if PATTERN.search(line):
            # Extra false-positive check
            if any(hint in line for hint in FALSE_POSITIVE_HINTS):
                continue
            violations.append((i, line.rstrip()))

    return violations


def main() -> int:
    if not SRC_ROOT.exists():
        print(f"Source root not found: {SRC_ROOT}")
        return 1

    all_violations: dict[Path, list[tuple[int, str]]] = {}
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        violations = scan_file(py_file)
        if violations:
            all_violations[py_file] = violations

    if not all_violations:
        print("OK: No bare .values usage found in src/neutralgrid/")
        return 0

    print(f"FAIL: Found bare .values usage in {len(all_violations)} file(s):\n")
    total = 0
    for filepath, violations in all_violations.items():
        rel = filepath.relative_to(SRC_ROOT.parent.parent)
        for lineno, line in violations:
            print(f"  {rel}:{lineno}: {line.strip()}")
            total += 1

    print(f"\n{total} violation(s) found.")
    print("Fix: Replace `series.values` with `np.asarray(series)` or `series.to_numpy()`")
    return 1


if __name__ == "__main__":
    sys.exit(main())
