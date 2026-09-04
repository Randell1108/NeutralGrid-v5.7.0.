"""Tests for the loop-mode helpers in live_decision_scanner.py (Phase C).

Covers:
  * parse_interval() — string-to-seconds conversion + validation.
  * LockFile         — acquire/release/conflict semantics.
"""
from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path

import pytest

# live_decision_scanner.py is a top-level script; importing it via path
# requires adding the project root to sys.path. The conftest.py + pyproject
# pythonpath setting cover src/, but not the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from live_decision_scanner import LockFile, _load_specs_from_yaml_paths, parse_interval  # noqa: E402


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _bot_yaml(symbol: str, *, strategy_id: str = "1") -> str:
    return f"""
bots:
  - symbol: {symbol}
    strategy_id: "{strategy_id}"
    deploy_ts: 2026-05-01T00:00:00Z
    grid_lower: 1
    grid_upper: 2
    num_grids: 10
    leverage: 5
    capital_usdt: 20
"""


# -- parse_interval ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("5m", 300.0),
        ("300s", 300.0),
        ("1h", 3600.0),
        ("90", 90.0),
        ("1.5m", 90.0),
        ("500ms", 0.5),
        (" 5M ", 300.0),  # whitespace + case insensitive
    ],
)
def test_parse_interval_valid(raw: str, expected: float) -> None:
    assert parse_interval(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "  ", "abc", "5x", "-1m", "0s", "-30"])
def test_parse_interval_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_interval(raw)


# -- LockFile ----------------------------------------------------------------


def test_lock_acquire_creates_file(tmp_path: Path) -> None:
    lock = LockFile(tmp_path / ".scanner.lock")
    assert lock.acquire() is True
    assert (tmp_path / ".scanner.lock").is_file()
    content = (tmp_path / ".scanner.lock").read_text(encoding="utf-8")
    # Should contain pid + ISO timestamp on two lines
    parts = content.strip().splitlines()
    assert len(parts) >= 2
    assert parts[0].isdigit()


def test_lock_release_removes_file(tmp_path: Path) -> None:
    lock = LockFile(tmp_path / ".scanner.lock")
    lock.acquire()
    lock.release()
    assert not (tmp_path / ".scanner.lock").exists()


def test_lock_release_idempotent(tmp_path: Path) -> None:
    lock = LockFile(tmp_path / ".scanner.lock")
    lock.acquire()
    lock.release()
    # Second call must not raise
    lock.release()


def test_lock_conflict_refuses_to_acquire(tmp_path: Path) -> None:
    lock1 = LockFile(tmp_path / ".scanner.lock")
    assert lock1.acquire() is True

    lock2 = LockFile(tmp_path / ".scanner.lock")
    assert lock2.acquire() is False  # second instance refused

    # Original lock still works
    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_lock_creates_parent_dir_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / ".scanner.lock"
    assert not target.parent.exists()
    lock = LockFile(target)
    assert lock.acquire() is True
    assert target.is_file()
    lock.release()


def test_active_symbol_file_requires_filename_symbol_match(tmp_path: Path) -> None:
    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    valid = _write(tmp_path / "RENDERUSDT.yaml", _bot_yaml("RENDERUSDT"))
    mismatch = _write(tmp_path / "WRONGUSDT.yaml", _bot_yaml("ESPORTSUSDT"))

    specs, issues = _load_specs_from_yaml_paths(
        [valid, mismatch],
        now=now,
        enforce_symbol_filenames=True,
    )

    assert [spec.symbol for spec in specs] == ["RENDERUSDT"]
    assert len(issues) == 1
    assert "active_symbol_file_error" in issues[0]
    assert "WRONGUSDT" in issues[0]


def test_legacy_dated_registry_allows_multiple_bots(tmp_path: Path) -> None:
    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    path = _write(
        tmp_path / "06-05-26.yaml",
        """
bots:
  - symbol: RENDERUSDT
    strategy_id: "1"
    deploy_ts: 2026-05-01T00:00:00Z
    grid_lower: 1
    grid_upper: 2
    num_grids: 10
    leverage: 5
    capital_usdt: 20
  - symbol: ESPORTSUSDT
    strategy_id: "2"
    deploy_ts: 2026-05-01T00:00:00Z
    grid_lower: 1
    grid_upper: 2
    num_grids: 10
    leverage: 5
    capital_usdt: 20
""",
    )

    specs, issues = _load_specs_from_yaml_paths(
        [path],
        now=now,
        enforce_symbol_filenames=True,
    )

    assert [spec.symbol for spec in specs] == ["RENDERUSDT", "ESPORTSUSDT"]
    assert issues == []
