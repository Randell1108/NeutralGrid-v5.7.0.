"""Normalize stale live-telemetry manifests whose owner process has exited.

The command is dry-run by default.  ``--apply`` atomically replaces only
manifests that still advertise a transient status, contain a known owner PID,
and have no live owner process.  Private one-shot manifests normalize to
``finished``; long-running collector manifests normalize to ``stopped``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TRANSIENT_STATUSES = frozenset(
    {"active", "running", "starting", "stopping", "waiting_for_valid_browser_state"}
)
PID_FIELDS = ("pid", "collector_pid")
DIFF_DEPTH_SCHEMA = "binance_usdm_diff_depth_manifest_v1"


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _owner_pids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    values: list[int] = []
    for field in PID_FIELDS:
        value = payload.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _linked_symbol_owners(
    manifests: Sequence[Path],
    *,
    root: Path,
) -> dict[Path, tuple[int, str]]:
    """Resolve exact per-symbol ownership declared by aggregate manifests."""

    root_resolved = root.resolve()
    manifest_paths = {path.resolve() for path in manifests}
    candidates: dict[Path, set[tuple[int, str]]] = {}
    for aggregate_path in manifests:
        aggregate = _read_manifest_object(aggregate_path)
        if aggregate is None:
            continue
        if aggregate.get("schema_version") != DIFF_DEPTH_SCHEMA:
            continue
        status = str(aggregate.get("status", "")).strip().lower()
        if status not in TRANSIENT_STATUSES:
            continue
        collector_pid = aggregate.get("collector_pid")
        if (
            not isinstance(collector_pid, int)
            or isinstance(collector_pid, bool)
            or collector_pid <= 0
        ):
            continue
        run_id = aggregate.get("run_id")
        symbol_run_dirs = aggregate.get("symbol_run_dirs")
        if not isinstance(run_id, str) or not run_id or not isinstance(
            symbol_run_dirs, Mapping
        ):
            continue
        for symbol, run_dir in symbol_run_dirs.items():
            if not isinstance(symbol, str) or not symbol or not isinstance(run_dir, str):
                continue
            run_path = Path(run_dir)
            if not run_path.is_absolute():
                continue
            child_path = (run_path.resolve() / "manifest.json").resolve()
            try:
                child_path.relative_to(root_resolved)
            except ValueError:
                continue
            if child_path not in manifest_paths:
                continue
            child = _read_manifest_object(child_path)
            if child is None:
                continue
            if (
                child.get("schema_version") != DIFF_DEPTH_SCHEMA
                or child.get("run_id") != run_id
                or child.get("symbol") != symbol
            ):
                continue
            owner = (collector_pid, str(aggregate_path))
            candidates.setdefault(child_path, set()).add(owner)
    return {
        path: next(iter(owners))
        for path, owners in candidates.items()
        if len(owners) == 1
    }


def normalize_manifest(
    path: Path,
    *,
    now: datetime,
    apply: bool,
    pid_is_alive: Callable[[int], bool] = _pid_is_alive,
    linked_owner: tuple[int, str] | None = None,
) -> dict[str, Any]:
    """Return one audit record and optionally normalize the manifest in place."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "outcome": "skipped_invalid_json", "error": str(exc)}
    if not isinstance(payload, dict):
        return {"path": str(path), "outcome": "skipped_non_object"}

    status = str(payload.get("status", "")).strip().lower()
    if status not in TRANSIENT_STATUSES:
        return {
            "path": str(path),
            "outcome": "skipped_terminal_or_unknown",
            "status": status or None,
        }

    owner_pids = _owner_pids(payload)
    owner_source_manifest: str | None = None
    if not owner_pids and linked_owner is not None:
        owner_pids = (linked_owner[0],)
        owner_source_manifest = linked_owner[1]
    if not owner_pids:
        return {
            "path": str(path),
            "outcome": "skipped_missing_owner_pid",
            "status": status,
        }
    live_pids = tuple(pid for pid in owner_pids if pid_is_alive(pid))
    if live_pids:
        return {
            "path": str(path),
            "outcome": "skipped_owner_alive",
            "status": status,
            "owner_pids": owner_pids,
            "live_pids": live_pids,
        }

    normalized_at = _utc_iso(now)
    terminal_status = (
        "stopped"
        if "collector_pid" in payload or owner_source_manifest is not None
        else "finished"
    )
    timestamp_field = (
        "stopped_at_utc" if terminal_status == "stopped" else "finished_at_utc"
    )
    payload["status_before_cleanup"] = status
    payload["status"] = terminal_status
    payload[timestamp_field] = normalized_at
    payload["cleanup_normalized_at_utc"] = normalized_at
    payload["cleanup_reason"] = "stale_manifest_owner_process_not_running"
    if owner_source_manifest is not None:
        payload["cleanup_owner_source_manifest"] = owner_source_manifest
        payload["cleanup_owner_pids"] = list(owner_pids)
    if apply:
        _atomic_write_json(path, payload)
    record = {
        "path": str(path),
        "outcome": "normalized" if apply else "would_normalize",
        "previous_status": status,
        "status": terminal_status,
        "owner_pids": owner_pids,
    }
    if owner_source_manifest is not None:
        record["owner_source_manifest"] = owner_source_manifest
    return record


def run_cleanup(
    root: Path,
    *,
    now: datetime,
    apply: bool,
    pid_is_alive: Callable[[int], bool] = _pid_is_alive,
) -> dict[str, Any]:
    manifests = sorted(root.rglob("manifest.json")) if root.is_dir() else []
    linked_owners = _linked_symbol_owners(manifests, root=root)
    records = [
        normalize_manifest(
            path,
            now=now,
            apply=apply,
            pid_is_alive=pid_is_alive,
            linked_owner=linked_owners.get(path.resolve()),
        )
        for path in manifests
    ]
    counts = Counter(str(record["outcome"]) for record in records)
    return {
        "schema_version": "neutralgrid_stale_manifest_cleanup_v1",
        "mode": "apply" if apply else "dry_run",
        "root": str(root.resolve()),
        "scanned_manifest_count": len(manifests),
        "linked_symbol_owner_count": len(linked_owners),
        "outcome_counts": dict(sorted(counts.items())),
        "records": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs" / "audits",
        help="Audit tree containing manifest.json files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically rewrite eligible stale manifests; default is dry-run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional UTF-8 JSON path for the cleanup report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_cleanup(args.root, now=datetime.now(UTC), apply=bool(args.apply))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
