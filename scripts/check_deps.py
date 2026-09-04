"""
Pre-deploy dependency check.

Verifies that critical compiled/ML runtime dependencies are importable and
that their installed version matches the pinned version in requirements.lock
(if that file exists).

Usage:
    python scripts/check_deps.py          # exit 0 = OK, exit 1 = mismatch
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements.lock"

CRITICAL_DEPS = ["pyarrow", "numpy", "pandas", "scikit-learn", "hmmlearn", "joblib"]


def _lock_versions() -> dict[str, str]:
    """Parse requirements.lock and return {package: pinned_version}."""
    versions: dict[str, str] = {}
    if not LOCK_FILE.exists():
        return versions
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            versions[m.group(1).lower()] = m.group(2)
    return versions


def main() -> int:
    lock = _lock_versions()
    ok = True

    for dep in CRITICAL_DEPS:
        # Check importable
        try:
            installed = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            print(f"FAIL: {dep} is not installed")
            ok = False
            continue

        # Check version against lock
        pinned = lock.get(dep.lower())
        if pinned is not None and installed != pinned:
            print(
                f"FAIL: {dep} version mismatch: "
                f"installed={installed}, lock={pinned}"
            )
            ok = False
        else:
            label = f" (lock: {pinned})" if pinned else ""
            print(f"OK: {dep}=={installed}{label}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
