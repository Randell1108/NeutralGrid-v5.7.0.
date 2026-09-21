"""Request bounded, exact-owner graceful WebSocket shutdown using STOP files.

This command never signals or kills processes. Legacy manifests without OS
creation identities, inaccessible owners and reused PIDs require investigation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.data.diff_depth import atomic_write_json  # noqa: E402
from neutralgrid.core.process_identity import (  # noqa: E402
    ProcessObservation,
    matches_identity,
    query_process,
)


logger = logging.getLogger(__name__)
SERVICES = {
    "public": ("collect_diff_depth.py", "binance_usdm_diff_depth_manifest_v1", "public_diff_depth"),
    "market": ("collect_market_streams.py", "binance_usdm_market_stream_manifest_v1", "market"),
    "private": ("collect_private_user_stream.py", "binance_usdm_private_service_manifest_v1", "private"),
}


class StopRequestError(RuntimeError):
    """Ownership could not be established without risking an unrelated run."""


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root or path.is_symlink():
        raise StopRequestError(f"path is not an owned descendant of {root}: {path}")
    return resolved


def _read(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise StopRequestError(f"manifest is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _identity(payload: Mapping[str, Any], pid_field: str) -> Mapping[str, Any]:
    identity = payload.get("process_identity")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(identity.get("pid"), int)
        or isinstance(identity.get("pid"), bool)
        or identity["pid"] <= 0
        or identity["pid"] != payload.get(pid_field)
        or not isinstance(identity.get("creation_token"), str)
        or not identity["creation_token"]
        or not isinstance(identity.get("executable"), str)
        or not identity["executable"]
    ):
        raise StopRequestError(f"missing exact process identity for {pid_field}; legacy PID alone is insufficient")
    return identity


def request_stop(
    audit_root: Path,
    *,
    timeout: float = 30.0,
    poll_seconds: float = 0.25,
    observe: Callable[[int], ProcessObservation] = query_process,
) -> dict[str, Any]:
    if not math.isfinite(timeout) or timeout < 0 or not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise StopRequestError("timeout must be finite and non-negative; poll interval must be finite and positive")
    audit_root = _contained(audit_root, ROOT.resolve())
    if not audit_root.is_dir():
        raise StopRequestError(f"audit root does not exist: {audit_root}")
    receipt_path = _contained(audit_root / "stop_receipt.json", audit_root)
    report: dict[str, Any] = {
        "schema_version": "neutralgrid_binance_ws_stop_receipt_v1",
        "audit_root": str(audit_root),
        "requested_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "marker_only_no_signals",
        "timeout_seconds": timeout,
        "status": "blocked",
        "processes": {},
        "markers_written": [],
    }
    try:
        manifest_path = _contained(audit_root / "manifest.json", audit_root)
        supervisor, supervisor_hash = _read(manifest_path)
        if supervisor.get("schema_version") != "neutralgrid_binance_ws_supervisor_v1":
            raise StopRequestError("unexpected supervisor manifest schema")
        if supervisor.get("audit_root") != str(audit_root):
            raise StopRequestError("supervisor audit_root does not match requested root")
        owner = _identity(supervisor, "supervisor_pid")
        services = supervisor.get("services")
        if not isinstance(services, Mapping) or set(services) != set(SERVICES):
            raise StopRequestError("supervisor must own exactly public, market and private")
        identities: dict[str, Mapping[str, Any]] = {"supervisor": owner}
        markers: dict[str, Path] = {"supervisor": _contained(audit_root / "STOP", audit_root)}
        authority_hashes = {manifest_path: supervisor_hash}
        for name, (script, schema, service_name) in SERVICES.items():
            record = services[name]
            if not isinstance(record, Mapping):
                raise StopRequestError(f"invalid service record: {name}")
            service_dir = _contained(audit_root / name, audit_root)
            child_path = _contained(service_dir / "manifest.json", audit_root)
            if record.get("audit_dir") != str(service_dir) or record.get("manifest_path") != str(child_path):
                raise StopRequestError(f"{name}: supervisor path ownership mismatch")
            command = record.get("command")
            if (
                not isinstance(command, list) or len(command) < 4
                or not all(isinstance(arg, str) for arg in command)
                or command[1] != str(ROOT / "scripts" / script)
                or command.count("--audit-dir") != 1
                or command.index("--audit-dir") + 1 >= len(command)
                or command[command.index("--audit-dir") + 1] != str(service_dir)
            ):
                raise StopRequestError(f"{name}: collector command ownership mismatch")
            launcher_identity = record.get("process_identity")
            # Preserve launcher birth identity after exit: the actual Python worker
            # can outlive a launcher on Windows and is checked independently below.
            if launcher_identity is not None:
                launcher_pid = launcher_identity.get("pid") if isinstance(launcher_identity, Mapping) else None
                if record.get("running") is True and record.get("pid") != launcher_pid:
                    raise StopRequestError(f"{name}: running launcher PID disagrees with its identity")
                identities[f"{name}_launcher"] = _identity(
                    {"pid": launcher_pid, "process_identity": launcher_identity}, "pid"
                )
            elif record.get("running") is True or record.get("pid") is not None:
                raise StopRequestError(f"{name}: launcher identity is missing")
            child, child_hash = _read(child_path)
            if child.get("schema_version") != schema or child.get("service") != service_name:
                raise StopRequestError(f"{name}: unexpected collector manifest")
            if child.get("audit_dir") != str(service_dir):
                raise StopRequestError(f"{name}: collector audit_dir ownership mismatch")
            if not isinstance(child.get("run_id"), str) or not child["run_id"]:
                raise StopRequestError(f"{name}: collector run identity is missing")
            identities[name] = _identity(child, "collector_pid")
            authority_hashes[child_path] = child_hash
            markers[name] = _contained(service_dir / "STOP", audit_root)

        # Validate the entire ownership set before making any stop request.
        for name, identity in identities.items():
            observed = observe(identity["pid"])
            if observed.state != "exited" and not matches_identity(observed, identity):
                raise StopRequestError(f"{name}: owner is unknown or PID was reused")
            report["processes"][name] = {"pid": identity["pid"], "initial_state": observed.state}
        for path, digest in authority_hashes.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                # Heartbeats may change during verification. A retry uses a coherent
                # fresh ownership snapshot instead of stopping a different generation.
                raise StopRequestError(f"manifest changed during ownership verification: {path}")
        for name, marker in markers.items():
            # Recheck exact process identity immediately before writing its marker.
            observed = observe(identities[name]["pid"])
            if observed.state != "exited" and not matches_identity(observed, identities[name]):
                raise StopRequestError(f"{name}: owner changed before marker write")
            if observed.state == "running":
                try:
                    with marker.open("x", encoding="ascii") as handle:
                        handle.write("stop requested by verified ownership helper\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except FileExistsError:
                    if marker.is_symlink() or not marker.is_file():
                        raise StopRequestError(f"stop marker is not an owned regular file: {marker}")
                report["markers_written"].append(str(marker))

        deadline = time.monotonic() + timeout
        while True:
            pending = []
            for name, identity in identities.items():
                observed = observe(identity["pid"])
                state = (
                    "exited" if observed.state == "exited"
                    else "still_running" if matches_identity(observed, identity)
                    else "identity_changed_or_unknown"
                )
                report["processes"][name]["final_state"] = state
                if state != "exited":
                    pending.append(name)
            if not pending:
                report["status"] = "stopped"
                break
            if time.monotonic() >= deadline:
                report["status"] = "incomplete"
                report["pending"] = pending
                break
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    except (OSError, ValueError, StopRequestError) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(receipt_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        report = request_stop(args.audit_root, timeout=args.timeout_seconds)
    except (OSError, StopRequestError) as exc:
        logger.error("Stop request failed: %s", exc)
        return 2
    logger.warning("WebSocket stop status=%s; receipt=%s", report["status"], args.audit_root / "stop_receipt.json")
    return 0 if report["status"] == "stopped" else 2


if __name__ == "__main__":
    raise SystemExit(main())
