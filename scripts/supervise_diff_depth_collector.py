"""Reconcile one repository-owned diff-depth collector to an exact live roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_telemetry_controller import load_complete_cycle


UTC = timezone.utc
OWNER_SCHEMA = "neutralgrid_diff_depth_collector_owner_v1"


class CollectorSupervisorError(RuntimeError):
    """Fail-closed collector ownership, identity, or health error."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CollectorSupervisorError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CollectorSupervisorError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CollectorSupervisorError(f"{field} lacks timezone authority")
    return parsed.astimezone(UTC)


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CollectorSupervisorError(f"non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except CollectorSupervisorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectorSupervisorError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CollectorSupervisorError(f"{path}: JSON root must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CollectorSupervisorError(f"cannot hash collector script {path}: {exc}") from exc


def collector_replacement_reasons(
    owner: Mapping[str, Any],
    *,
    target_sha256: str,
    collector_script_sha256: str,
) -> list[str]:
    """Return deterministic reasons why an owned collector must be replaced."""

    reasons: list[str] = []
    if owner.get("target_sha256") != target_sha256:
        reasons.append("fresh roster target mismatch")
    if owner.get("collector_script_sha256") != collector_script_sha256:
        reasons.append("collector script hash mismatch")
    return reasons


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def windows_command_line(command: Sequence[str]) -> str:
    """Return the Windows command-line representation used for owner evidence."""

    return subprocess.list2cmdline(list(command))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_expected_targets(
    cycle_manifest: Path,
    *,
    live_root: Path,
) -> tuple[Path, str, list[tuple[str, str]], str]:
    cycle = load_complete_cycle(cycle_manifest, allowed_live_root=live_root)
    payload = _strict_json_object(cycle_manifest)
    target_path_value = payload.get("collector_targets_csv")
    target_hash = payload.get("collector_targets_sha256")
    run_id = str(payload.get("run_id", "")).strip()
    if not isinstance(target_path_value, str) or not target_path_value.strip():
        raise CollectorSupervisorError("cycle collector_targets_csv is missing")
    if not isinstance(target_hash, str) or len(target_hash) != 64:
        raise CollectorSupervisorError("cycle collector_targets_sha256 is invalid")
    target_path = Path(target_path_value).resolve()
    target_bytes = target_path.read_bytes()
    if hashlib.sha256(target_bytes).hexdigest() != target_hash:
        raise CollectorSupervisorError("collector target CSV hash mismatch")
    with target_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        identities = [
            (
                str(row.get("symbol", "")).strip().upper(),
                str(row.get("strategy_id", "")).strip(),
            )
            for row in reader
        ]
    expected = list(zip(cycle.symbols, cycle.strategy_ids))
    if identities != expected:
        raise CollectorSupervisorError("collector target CSV identity mismatch")
    return target_path, target_hash, expected, run_id


def _list_windows_processes() -> list[dict[str, Any]]:
    script = (
        "$items = Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -and $_.CommandLine -match "
        "'collect_diff_depth\\.py' } | "
        "Select-Object @{n='pid';e={$_.ProcessId}},"
        "@{n='parent_process_id';e={$_.ParentProcessId}},"
        "@{n='executable_path';e={$_.ExecutablePath}},"
        "@{n='command_line';e={$_.CommandLine}}; "
        "if ($items) { $items | ConvertTo-Json -Depth 3 -Compress } else { '[]' }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise CollectorSupervisorError(f"cannot enumerate processes: {detail}")
    try:
        decoded = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise CollectorSupervisorError("process enumeration returned invalid JSON") from exc
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list):
        raise CollectorSupervisorError("process enumeration returned non-list JSON")
    return [item for item in decoded if isinstance(item, dict)]


def owner_matches_process(
    owner: Mapping[str, Any],
    process: Mapping[str, Any],
    *,
    workspace_root: Path,
) -> bool:
    command = owner.get("command")
    if owner.get("schema_version") != OWNER_SCHEMA:
        return False
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return False
    if owner.get("workspace") != str(workspace_root.resolve()):
        return False
    if owner.get("pid") != process.get("pid"):
        return False
    return process.get("command_line") == windows_command_line(command)


def _repo_collector_processes(
    processes: Sequence[Mapping[str, Any]],
    *,
    workspace_root: Path,
) -> list[dict[str, Any]]:
    script_path = (workspace_root / "scripts" / "collect_diff_depth.py").resolve()
    script_text = str(script_path).lower()
    matches: list[dict[str, Any]] = []
    for process in processes:
        executable_name = Path(str(process.get("executable_path", ""))).name.lower()
        if executable_name not in {"python.exe", "pythonw.exe"}:
            continue
        command_line = str(process.get("command_line", ""))
        if script_text in command_line.lower():
            matches.append(dict(process))
    return matches


def owned_collector_process_chain(
    owner: Mapping[str, Any],
    processes: Sequence[Mapping[str, Any]],
    *,
    workspace_root: Path,
) -> list[dict[str, Any]]:
    """Return one exact owner process plus same-command descendants.

    On Windows, a virtual-environment ``python.exe`` launcher can remain as the
    parent of the interpreter process that writes the collector manifest. Both
    processes have the exact recorded command line. Treat that verified chain
    as one logical collector without accepting unrelated Python processes.
    """

    roots = [
        dict(process)
        for process in processes
        if owner_matches_process(owner, process, workspace_root=workspace_root)
    ]
    if len(roots) != 1:
        return []
    command_line = str(roots[0].get("command_line", ""))
    owned_pids = {int(roots[0]["pid"])}
    changed = True
    while changed:
        changed = False
        for process in processes:
            try:
                pid = int(process.get("pid", -1))
                parent_pid = int(process.get("parent_process_id", -1))
            except (TypeError, ValueError):
                continue
            if (
                pid not in owned_pids
                and parent_pid in owned_pids
                and process.get("command_line") == command_line
            ):
                owned_pids.add(pid)
                changed = True
    return [
        dict(process)
        for process in processes
        if int(process.get("pid", -1)) in owned_pids
    ]


def _counter(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CollectorSupervisorError(f"{field} must be a non-negative integer")
    return value


def validate_collector_health(
    run_manifest_path: Path,
    *,
    expected_identities: Sequence[tuple[str, str]],
    expected_pids: Sequence[int],
    now: datetime,
    max_age_seconds: float,
) -> dict[str, Any]:
    payload = _strict_json_object(run_manifest_path)
    if payload.get("status") not in {"running", "complete"}:
        raise CollectorSupervisorError("collector run is not running/complete")
    collector_pid = payload.get("collector_pid")
    if collector_pid not in set(expected_pids):
        raise CollectorSupervisorError("collector PID does not match owner")
    updated_at = _parse_utc(payload.get("updated_at_utc"), field="updated_at_utc")
    age = (now.astimezone(UTC) - updated_at).total_seconds()
    if age < -5 or age > max_age_seconds:
        raise CollectorSupervisorError(f"collector manifest is stale/future: age={age:.1f}s")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise CollectorSupervisorError("collector targets are missing")
    actual_identities = [
        (
            str(item.get("symbol", "")).strip().upper(),
            str(item.get("strategy_id", "")).strip(),
        )
        for item in targets
        if isinstance(item, Mapping)
    ]
    if actual_identities != list(expected_identities):
        raise CollectorSupervisorError("collector target mismatch")
    run_dirs = payload.get("symbol_run_dirs")
    if not isinstance(run_dirs, Mapping) or set(run_dirs) != {
        symbol for symbol, _strategy in expected_identities
    }:
        raise CollectorSupervisorError("collector symbol run directories mismatch")
    total_keys = (
        "wire_events",
        "snapshots",
        "events_applied",
        "sequence_gaps",
        "parse_errors",
        "public_agg_trades",
        "public_mark_price_updates",
        "market_connections",
        "market_coverage_gaps",
        "market_parse_errors",
    )
    totals = {key: 0 for key in total_keys}
    per_symbol: dict[str, Any] = {}
    for symbol, strategy_id in expected_identities:
        run_dir = Path(str(run_dirs[symbol])).resolve()
        symbol_manifest = _strict_json_object(run_dir / "manifest.json")
        target = symbol_manifest.get("target")
        if not isinstance(target, Mapping) or (
            str(target.get("symbol", "")).upper(),
            str(target.get("strategy_id", "")),
        ) != (symbol, strategy_id):
            raise CollectorSupervisorError(f"{symbol}: symbol-manifest target mismatch")
        symbol_updated = _parse_utc(
            symbol_manifest.get("updated_at_utc"),
            field=f"{symbol}.updated_at_utc",
        )
        symbol_age = (now.astimezone(UTC) - symbol_updated).total_seconds()
        if symbol_age < -5 or symbol_age > max_age_seconds:
            raise CollectorSupervisorError(f"{symbol}: symbol manifest is stale/future")
        if symbol_manifest.get("last_error") not in {None, ""}:
            raise CollectorSupervisorError(f"{symbol}: last_error is not empty")
        if symbol_manifest.get("market_connection_error") not in {None, ""}:
            raise CollectorSupervisorError(
                f"{symbol}: market_connection_error is not empty"
            )
        if symbol_manifest.get("collect_agg_trades") is not True:
            raise CollectorSupervisorError(
                f"{symbol}: aggregate-trade collection is not enabled"
            )
        if symbol_manifest.get("collect_mark_price_updates") is not True:
            raise CollectorSupervisorError(
                f"{symbol}: mark-price collection is not enabled"
            )
        if symbol_manifest.get("trade_subscription_acknowledged") is not True:
            raise CollectorSupervisorError(
                f"{symbol}: aggregate-trade subscription is not acknowledged"
            )
        if symbol_manifest.get("mark_price_subscription_acknowledged") is not True:
            raise CollectorSupervisorError(
                f"{symbol}: mark-price subscription is not acknowledged"
            )
        counters = symbol_manifest.get("counters")
        if not isinstance(counters, Mapping):
            raise CollectorSupervisorError(f"{symbol}: counters are missing")
        normalized = {
            key: _counter(counters.get(key), field=f"{symbol}.{key}")
            for key in total_keys
        }
        if normalized["sequence_gaps"]:
            raise CollectorSupervisorError(f"{symbol}: sequence gaps are nonzero")
        if normalized["parse_errors"]:
            raise CollectorSupervisorError(f"{symbol}: parse errors are nonzero")
        if normalized["market_coverage_gaps"]:
            raise CollectorSupervisorError(
                f"{symbol}: market-stream coverage gaps are nonzero"
            )
        if normalized["market_parse_errors"]:
            raise CollectorSupervisorError(
                f"{symbol}: market-stream parse errors are nonzero"
            )
        if normalized["events_applied"] <= 0 or normalized["snapshots"] <= 0:
            raise CollectorSupervisorError(f"{symbol}: no valid sequence segment")
        if normalized["market_connections"] <= 0:
            raise CollectorSupervisorError(f"{symbol}: no market-stream connection")
        if normalized["public_mark_price_updates"] <= 0:
            raise CollectorSupervisorError(f"{symbol}: no mark-price event observed")
        for key, value in normalized.items():
            totals[key] += value
        per_symbol[symbol] = {
            "manifest": str(run_dir / "manifest.json"),
            "updated_at_utc": symbol_updated.isoformat(),
            "counters": normalized,
            "last_error": None,
        }
    return {
        "healthy": True,
        "collector_pid": collector_pid,
        "manifest": str(run_manifest_path.resolve()),
        "updated_at_utc": updated_at.isoformat(),
        "totals": totals,
        "per_symbol": per_symbol,
        "limitations": [
            "Collection is prospective from collector startup.",
            "Public depth and aggregate trades are not private fills or queue position.",
        ],
    }


def _process_exists(pid: int) -> bool:
    return any(int(item.get("pid", -1)) == pid for item in _list_windows_processes())


def _stop_verified_owner(owner: Mapping[str, Any], *, timeout_seconds: float) -> None:
    audit_dir = Path(str(owner.get("audit_dir", ""))).resolve()
    workspace = Path(str(owner.get("workspace", ""))).resolve()
    allowed_root = (workspace / "outputs" / "audits" / "diff_depth_collectors").resolve()
    if not _is_relative_to(audit_dir, allowed_root):
        raise CollectorSupervisorError("recorded collector audit directory is unsafe")
    pid = int(owner.get("pid", -1))
    (audit_dir / "STOP").write_text("stop\n", encoding="ascii")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.5)
    raise CollectorSupervisorError(
        f"verified collector PID {pid} did not stop within timeout"
    )


def _start_collector(
    command: Sequence[str],
    *,
    audit_dir: Path,
) -> int:
    audit_dir.mkdir(parents=True, exist_ok=True)
    stdout = (audit_dir / "collector.stdout.log").open("a", encoding="utf-8")
    stderr = (audit_dir / "collector.stderr.log").open("a", encoding="utf-8")
    creationflags = 0
    for flag in ("CREATE_NEW_PROCESS_GROUP", "CREATE_NO_WINDOW", "DETACHED_PROCESS"):
        creationflags |= int(getattr(subprocess, flag, 0))
    try:
        process = subprocess.Popen(
            list(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            close_fds=True,
        )
    finally:
        stdout.close()
        stderr.close()
    return int(process.pid)


def reconcile_collector(
    *,
    cycle_manifest: Path,
    workspace_root: Path,
    live_root: Path,
    runtime_dir: Path,
    health_timeout_seconds: float,
    health_max_age_seconds: float,
    stop_timeout_seconds: float,
) -> dict[str, Any]:
    workspace = workspace_root.resolve()
    resolved_live_root = live_root.resolve()
    if resolved_live_root != (workspace / "Live").resolve():
        raise CollectorSupervisorError("live_root must be the checkout Live directory")
    resolved_runtime_dir = runtime_dir.resolve()
    allowed_runtime_root = (workspace / "outputs" / "runtime").resolve()
    if not _is_relative_to(resolved_runtime_dir, allowed_runtime_root):
        raise CollectorSupervisorError("runtime_dir must be under outputs/runtime")
    target_path, target_hash, identities, run_id = _load_expected_targets(
        cycle_manifest.resolve(), live_root=resolved_live_root
    )
    collector_script = (workspace / "scripts" / "collect_diff_depth.py").resolve()
    collector_script_hash = _sha256_file(collector_script)
    owner_path = resolved_runtime_dir / "collector_owner.json"
    owner = _strict_json_object(owner_path) if owner_path.exists() else None
    repo_processes = _repo_collector_processes(
        _list_windows_processes(), workspace_root=workspace
    )
    owned_processes = (
        owned_collector_process_chain(
            owner,
            repo_processes,
            workspace_root=workspace,
        )
        if owner is not None
        else []
    )
    if repo_processes and len(owned_processes) != len(repo_processes):
        raise CollectorSupervisorError("multiple or ambiguous repository collectors are running")
    current = owned_processes[0] if owned_processes else None
    if current is not None:
        if owner is None or not owner_matches_process(
            owner, current, workspace_root=workspace
        ):
            raise CollectorSupervisorError(
                "running repository collector ownership is ambiguous"
            )
        replacement_reasons = collector_replacement_reasons(
            owner,
            target_sha256=target_hash,
            collector_script_sha256=collector_script_hash,
        )
        if replacement_reasons:
            mismatch_audit = (
                workspace
                / "outputs"
                / "audits"
                / "diff_depth_collectors"
                / f"replacement_{_utc_now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            _atomic_write_json(
                mismatch_audit,
                {
                    "schema_version": "neutralgrid_diff_depth_replacement_v1",
                    "old_owner": owner,
                    "new_cycle_manifest": str(cycle_manifest.resolve()),
                    "new_target_sha256": target_hash,
                    "new_collector_script_sha256": collector_script_hash,
                    "reasons": replacement_reasons,
                },
            )
            _stop_verified_owner(owner, timeout_seconds=stop_timeout_seconds)
            current = None
        else:
            manifest_path = Path(str(owner["audit_dir"])) / "manifest.json"
            deadline = time.monotonic() + health_timeout_seconds
            last_error = "collector manifest not available"
            while time.monotonic() < deadline:
                try:
                    live_processes = _repo_collector_processes(
                        _list_windows_processes(), workspace_root=workspace
                    )
                    live_owned = owned_collector_process_chain(
                        owner,
                        live_processes,
                        workspace_root=workspace,
                    )
                    if len(live_owned) != len(live_processes) or not live_owned:
                        raise CollectorSupervisorError(
                            "running repository collector ownership is ambiguous"
                        )
                    return validate_collector_health(
                        manifest_path,
                        expected_identities=identities,
                        expected_pids=[int(item["pid"]) for item in live_owned],
                        now=_utc_now(),
                        max_age_seconds=health_max_age_seconds,
                    )
                except CollectorSupervisorError as exc:
                    last_error = str(exc)
                if not _process_exists(int(owner["pid"])):
                    raise CollectorSupervisorError(
                        f"collector exited before becoming healthy: {last_error}"
                    )
                time.sleep(1.0)
            raise CollectorSupervisorError(
                f"collector did not become healthy within timeout: {last_error}"
            )

    if current is None:
        audit_dir = (
            workspace
            / "outputs"
            / "audits"
            / "diff_depth_collectors"
            / f"{target_hash[:8]}_{collector_script_hash[:8]}_{run_id}"
        )
        command = [
            str(workspace / ".venv" / "Scripts" / "python.exe"),
            str(collector_script),
            "--input",
            str(target_path),
            "--live-root",
            str(resolved_live_root),
            "--audit-dir",
            str(audit_dir),
            "--manifest-heartbeat-seconds",
            "5",
            "--fsync-every",
            "1",
        ]
        pid = _start_collector(command, audit_dir=audit_dir)
        owner = {
            "schema_version": OWNER_SCHEMA,
            "workspace": str(workspace),
            "pid": pid,
            "command": command,
            "command_line": windows_command_line(command),
            "audit_dir": str(audit_dir),
            "cycle_manifest": str(cycle_manifest.resolve()),
            "target_csv": str(target_path),
            "target_sha256": target_hash,
            "collector_script_sha256": collector_script_hash,
            "started_at_utc": _utc_now().isoformat(),
        }
        _atomic_write_json(owner_path, owner)

    assert owner is not None
    manifest_path = Path(str(owner["audit_dir"])) / "manifest.json"
    deadline = time.monotonic() + health_timeout_seconds
    last_error = "collector manifest not available"
    while time.monotonic() < deadline:
        try:
            live_processes = _repo_collector_processes(
                _list_windows_processes(), workspace_root=workspace
            )
            live_owned = owned_collector_process_chain(
                owner,
                live_processes,
                workspace_root=workspace,
            )
            if len(live_owned) != len(live_processes) or not live_owned:
                raise CollectorSupervisorError(
                    "running repository collector ownership is ambiguous"
                )
            return validate_collector_health(
                manifest_path,
                expected_identities=identities,
                expected_pids=[int(item["pid"]) for item in live_owned],
                now=_utc_now(),
                max_age_seconds=health_max_age_seconds,
            )
        except CollectorSupervisorError as exc:
            last_error = str(exc)
        if not _process_exists(int(owner["pid"])):
            raise CollectorSupervisorError(
                f"collector exited before becoming healthy: {last_error}"
            )
        time.sleep(1.0)
    raise CollectorSupervisorError(
        f"collector did not become healthy within timeout: {last_error}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / "outputs" / "runtime" / "diff_depth_collector_supervisor",
    )
    parser.add_argument("--health-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--health-max-age-seconds", type=float, default=15.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    for name in (
        "health_timeout_seconds",
        "health_max_age_seconds",
        "stop_timeout_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = reconcile_collector(
            cycle_manifest=args.cycle_manifest,
            workspace_root=args.workspace_root,
            live_root=args.live_root,
            runtime_dir=args.runtime_dir,
            health_timeout_seconds=float(args.health_timeout_seconds),
            health_max_age_seconds=float(args.health_max_age_seconds),
            stop_timeout_seconds=float(args.stop_timeout_seconds),
        )
    except (CollectorSupervisorError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "complete", **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
