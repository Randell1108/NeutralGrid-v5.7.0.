"""Own and supervise the three direct Binance USD-M WebSocket services."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, IO, Mapping, Sequence
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.diff_depth import atomic_write_json  # noqa: E402
from neutralgrid.data.private_user_stream import (  # noqa: E402
    PrivateUserStreamError,
    StrategyTarget,
    load_exact_order_linkages,
)
from scripts.collect_diff_depth import _git_output  # noqa: E402
from scripts.collect_market_streams import (  # noqa: E402
    DEFAULT_MARKET_WS_BASE,
)
from scripts.collect_private_user_stream import (  # noqa: E402
    DEFAULT_LIFECYCLE_WS_URL,
    DEFAULT_PRIVATE_WS_BASE,
    load_private_targets,
)
from scripts.collect_diff_depth import (  # noqa: E402
    DEFAULT_REST_BASE,
    DEFAULT_WS_BASE as DEFAULT_PUBLIC_WS_BASE,
)
from neutralgrid.core.process_identity import matches_identity, query_process  # noqa: E402


SUPERVISOR_SCHEMA_VERSION = "neutralgrid_binance_ws_supervisor_v1"
LIMA = ZoneInfo("America/Lima")


class WebSocketSupervisorError(RuntimeError):
    """Service ownership or process supervision failed closed."""


@dataclass
class ServiceProcess:
    name: str
    command: list[str]
    audit_dir: Path
    stdout_path: Path
    stderr_path: Path
    process: subprocess.Popen[str] | None = None
    stdout_handle: IO[str] | None = None
    stderr_handle: IO[str] | None = None
    starts: int = 0
    unexpected_exits: int = 0
    last_exit_code: int | None = None
    last_started_at_utc: str | None = None
    last_exited_at_utc: str | None = None
    next_restart_monotonic: float = 0.0
    restart_backoff_seconds: float = 0.0
    blocked_reason: str | None = None
    process_identity: dict[str, Any] | None = None

    def manifest_record(self) -> dict[str, Any]:
        process = self.process
        running = process is not None and process.poll() is None
        return {
            "name": self.name,
            "command": list(self.command),
            "audit_dir": str(self.audit_dir),
            "manifest_path": str(self.audit_dir / "manifest.json"),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "pid": process.pid if running and process is not None else None,
            "running": running,
            "starts": self.starts,
            "unexpected_exits": self.unexpected_exits,
            "last_exit_code": self.last_exit_code,
            "last_started_at_utc": self.last_started_at_utc,
            "last_exited_at_utc": self.last_exited_at_utc,
            "blocked_reason": self.blocked_reason,
            "process_identity": self.process_identity,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    return query_process(pid).state != "exited"


def _acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pid = -1
        if _pid_is_running(pid):
            raise WebSocketSupervisorError(
                f"Binance WebSocket supervisor already running with PID {pid}"
            )
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def _unique_symbols(targets: Sequence[StrategyTarget]) -> list[str]:
    return list(dict.fromkeys(target.symbol for target in targets))


def build_service_commands(
    args: argparse.Namespace,
    targets: Sequence[StrategyTarget],
) -> dict[str, list[str]]:
    """Build mutually exclusive public, market, and private child commands."""

    symbols = _unique_symbols(targets)
    common = [
        "--duration-seconds",
        "0",
        "--live-root",
        str(Path(args.live_root).resolve()),
        "--fsync-every",
        str(args.fsync_every),
        "--ingestion-date",
        str(args.ingestion_date),
    ]
    audit_root = Path(args.audit_root).resolve()
    commands = {
        "public": [
            sys.executable,
            str(ROOT / "scripts" / "collect_diff_depth.py"),
            "--symbols",
            *symbols,
            *common,
            "--audit-dir",
            str(audit_root / "public"),
            "--ws-base",
            str(args.public_ws_base),
            "--rest-base",
            str(args.rest_base),
            "--no-agg-trades",
            "--no-mark-price-updates",
        ],
        "market": [
            sys.executable,
            str(ROOT / "scripts" / "collect_market_streams.py"),
            "--symbols",
            *symbols,
            *common,
            "--audit-dir",
            str(audit_root / "market"),
            "--ws-base",
            str(args.market_ws_base),
        ],
        "private": [
            sys.executable,
            str(ROOT / "scripts" / "collect_private_user_stream.py"),
            "--input",
            str(Path(args.target_csv).resolve()),
            *common,
            "--audit-dir",
            str(audit_root / "private"),
            "--ws-base",
            str(args.private_ws_base),
            "--lifecycle-ws-url",
            str(args.private_lifecycle_ws_url),
        ],
    }
    if args.allow_nonproduction_endpoints:
        commands["private"].append("--allow-nonproduction-endpoints")
    for path in args.linkage_file:
        commands["private"].extend(["--linkage-file", str(Path(path).resolve())])
    return commands


def _close_logs(service: ServiceProcess) -> None:
    for handle_name in ("stdout_handle", "stderr_handle"):
        handle = getattr(service, handle_name)
        if handle is not None:
            handle.close()
            setattr(service, handle_name, None)


def _start_service(service: ServiceProcess, *, base_backoff: float) -> None:
    service.audit_dir.mkdir(parents=True, exist_ok=True)
    (service.audit_dir / "STOP").unlink(missing_ok=True)
    service.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    _close_logs(service)
    service.stdout_handle = service.stdout_path.open(
        "a", encoding="utf-8", buffering=1
    )
    service.stderr_handle = service.stderr_path.open(
        "a", encoding="utf-8", buffering=1
    )
    service.process = subprocess.Popen(
        service.command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=service.stdout_handle,
        stderr=service.stderr_handle,
        text=True,
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    service.starts += 1
    service.process_identity = query_process(service.process.pid).identity()
    service.last_started_at_utc = _utc_now_iso()
    service.last_exit_code = None
    service.blocked_reason = None
    if service.restart_backoff_seconds <= 0:
        service.restart_backoff_seconds = base_backoff


def _request_child_stop(service: ServiceProcess) -> None:
    service.audit_dir.mkdir(parents=True, exist_ok=True)
    (service.audit_dir / "STOP").write_text(
        "stop requested by supervisor\n", encoding="ascii"
    )


def _stop_services(services: Mapping[str, ServiceProcess], timeout: float) -> None:
    errors: list[str] = []
    workers: dict[str, Mapping[str, Any]] = {}
    for service in services.values():
        try:
            # The supervisor created and owns these exact service directories.
            # A Windows launcher can exit while its worker remains alive.
            _request_child_stop(service)
            child_manifest = service.audit_dir / "manifest.json"
            if child_manifest.is_file():
                payload = json.loads(child_manifest.read_text(encoding="utf-8"))
                identity = payload.get("process_identity") if isinstance(payload, dict) else None
                if (
                    not isinstance(identity, Mapping)
                    or payload.get("audit_dir") != str(service.audit_dir)
                    or payload.get("collector_pid") != identity.get("pid")
                    or not isinstance(identity.get("pid"), int)
                    or isinstance(identity.get("pid"), bool)
                    or identity["pid"] <= 0
                ):
                    errors.append(f"{service.name}: worker identity unavailable during shutdown")
                else:
                    workers[service.name] = identity
        except (OSError, ValueError) as exc:
            errors.append(f"{service.name}: stop request or worker manifest failed: {exc!r}")

    def workers_exited() -> bool:
        return all(query_process(identity["pid"]).state == "exited" for identity in workers.values())

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(
            service.process is None or service.process.poll() is not None
            for service in services.values()
        ) and workers_exited():
            break
        time.sleep(0.1)
    for service in services.values():
        process = service.process
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=min(5.0, max(1.0, timeout)))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=min(5.0, max(1.0, timeout)))
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{service.name}: shutdown failed: {exc!r}")
        finally:
            if process is not None:
                service.last_exit_code = process.poll()
                if service.last_exit_code is not None:
                    service.last_exited_at_utc = _utc_now_iso()
            _close_logs(service)
    for name, identity in workers.items():
        observed = query_process(identity["pid"])
        if observed.state != "exited":
            state = "still running" if matches_identity(observed, identity) else "unverified or reused PID"
            errors.append(f"{name}: worker {state} after bounded shutdown")
    if errors:
        raise WebSocketSupervisorError("; ".join(errors))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--linkage-file", type=Path, action="append", default=[])
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "outputs" / "audits" / "binance_websocket_services_current")
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--ingestion-date",
        help="Freeze a deliberate America/Lima YYYY-MM-DD date for all services",
    )
    parser.add_argument("--log-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--public-ws-base", default=DEFAULT_PUBLIC_WS_BASE)
    parser.add_argument("--market-ws-base", default=DEFAULT_MARKET_WS_BASE)
    parser.add_argument("--private-ws-base", default=DEFAULT_PRIVATE_WS_BASE)
    parser.add_argument(
        "--private-lifecycle-ws-url", default=DEFAULT_LIFECYCLE_WS_URL
    )
    parser.add_argument("--rest-base", default=DEFAULT_REST_BASE)
    parser.add_argument("--allow-nonproduction-endpoints", action="store_true")
    parser.add_argument("--fsync-every", type=int, default=1)
    parser.add_argument("--manifest-heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--restart-base-seconds", type=float, default=1.0)
    parser.add_argument("--restart-max-seconds", type=float, default=30.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    args.public_ws_base = str(args.public_ws_base).rstrip("/")
    args.market_ws_base = str(args.market_ws_base).rstrip("/")
    args.private_ws_base = str(args.private_ws_base).rstrip("/")
    args.private_lifecycle_ws_url = str(args.private_lifecycle_ws_url).rstrip("/")
    args.rest_base = str(args.rest_base).rstrip("/")
    if not args.allow_nonproduction_endpoints and (
        args.public_ws_base != DEFAULT_PUBLIC_WS_BASE
        or args.market_ws_base != DEFAULT_MARKET_WS_BASE
        or args.private_ws_base != DEFAULT_PRIVATE_WS_BASE
        or args.private_lifecycle_ws_url != DEFAULT_LIFECYCLE_WS_URL
        or args.rest_base != DEFAULT_REST_BASE
    ):
        parser.error(
            "non-production Binance endpoints require "
            "--allow-nonproduction-endpoints"
        )
    if not math.isfinite(args.duration_seconds) or args.duration_seconds < 0:
        parser.error("--duration-seconds must be finite and >= 0")
    if args.ingestion_date is None:
        args.ingestion_date = datetime.now(LIMA).date().isoformat()
    else:
        try:
            parsed_date = date.fromisoformat(args.ingestion_date)
        except ValueError:
            parser.error("--ingestion-date must be YYYY-MM-DD")
        if parsed_date.isoformat() != args.ingestion_date:
            parser.error("--ingestion-date must use canonical YYYY-MM-DD form")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be >= 0")
    for field in (
        "manifest_heartbeat_seconds",
        "restart_base_seconds",
        "restart_max_seconds",
        "shutdown_timeout_seconds",
    ):
        value = float(getattr(args, field))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{field.replace('_', '-')} must be finite and positive")
    return args


def supervise(args: argparse.Namespace) -> int:
    targets = load_private_targets(Path(args.target_csv))
    for path in args.linkage_file:
        if not Path(path).is_file():
            raise PrivateUserStreamError(f"linkage file does not exist: {path}")
    load_exact_order_linkages(args.linkage_file, expected_targets=targets)
    audit_root = Path(args.audit_root).resolve()
    audit_root.mkdir(parents=True, exist_ok=True)
    stop_path = audit_root / "STOP"
    lock_path = audit_root / "supervisor.lock"
    manifest_path = audit_root / "manifest.json"
    commands = build_service_commands(args, targets)
    services = {
        name: ServiceProcess(
            name=name,
            command=command,
            audit_dir=audit_root / name,
            stdout_path=Path(args.log_root).resolve() / f"binance_ws_{name}.out.log",
            stderr_path=Path(args.log_root).resolve() / f"binance_ws_{name}.err.log",
        )
        for name, command in commands.items()
    }
    started_at = _utc_now_iso()
    process_identity = query_process(os.getpid()).identity()
    deadline: float | None = None
    stopping = False
    git_head = _git_output(["rev-parse", "--short", "HEAD"])
    target_sha256 = hashlib.sha256(Path(args.target_csv).read_bytes()).hexdigest()

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping, stop_reason
        stopping = True
        stop_reason = f"signal_{_signum}"

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    def write_manifest(status: str, **extra: Any) -> None:
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SUPERVISOR_SCHEMA_VERSION,
                "status": status,
                "started_at_utc": started_at,
                "updated_at_utc": _utc_now_iso(),
                "supervisor_pid": os.getpid(),
                "process_identity": process_identity,
                "target_csv": str(Path(args.target_csv).resolve()),
                "target_sha256": target_sha256,
                "targets": [asdict(target) for target in targets],
                "unique_symbols": _unique_symbols(targets),
                "audit_root": str(audit_root),
                "live_root": str(Path(args.live_root).resolve()),
                "ingestion_date_lima": str(args.ingestion_date),
                "ingestion_timezone": "America/Lima",
                "ingestion_date_basis": "supervisor_frozen",
                "duration_seconds": float(args.duration_seconds),
                "services": {
                    name: service.manifest_record()
                    for name, service in services.items()
                },
                "route_ownership": {
                    "public": "diff-depth only",
                    "market": "aggregate trades, mark prices, closed klines",
                    "private": "authenticated account/order/trade/strategy events",
                },
                "endpoint_authority": {
                    "public_ws_base": str(args.public_ws_base),
                    "market_ws_base": str(args.market_ws_base),
                    "private_ws_base": str(args.private_ws_base),
                    "private_lifecycle_ws_url": str(
                        args.private_lifecycle_ws_url
                    ),
                    "rest_base": str(args.rest_base),
                    "nonproduction_override": bool(
                        args.allow_nonproduction_endpoints
                    ),
                },
                "runtime_effect": "observational_only",
                "git_head": git_head,
                **extra,
            },
        )

    last_manifest_write = float("-inf")
    exit_code = 0
    stop_reason = "requested"
    terminal_error: str | None = None
    lock_descriptor = _acquire_lock(lock_path)
    try:
        stop_path.unlink(missing_ok=True)
        write_manifest("starting")
        for service in services.values():
            _start_service(
                service, base_backoff=float(args.restart_base_seconds)
            )
        deadline = (
            time.monotonic() + float(args.duration_seconds)
            if args.duration_seconds > 0
            else None
        )
        write_manifest("running")
        while not stopping:
            now = time.monotonic()
            if stop_path.exists():
                stopping = True
                stop_reason = "stop_file"
                break
            if deadline is not None and now >= deadline:
                stopping = True
                stop_reason = "duration_elapsed"
                break
            for service in services.values():
                process = service.process
                if process is None:
                    if (
                        service.blocked_reason is None
                        and service.next_restart_monotonic > 0
                        and now >= service.next_restart_monotonic
                    ):
                        _start_service(
                            service,
                            base_backoff=float(args.restart_base_seconds),
                        )
                        service.next_restart_monotonic = 0.0
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                service.last_exit_code = return_code
                service.last_exited_at_utc = _utc_now_iso()
                _close_logs(service)
                service.process = None
                if service.name == "private" and return_code == 3:
                    service.blocked_reason = "missing_api_key_restart_supervisor_after_configuration"
                    continue
                service.unexpected_exits += 1
                if service.next_restart_monotonic <= 0:
                    service.next_restart_monotonic = (
                        now + service.restart_backoff_seconds
                    )
                    service.restart_backoff_seconds = min(
                        service.restart_backoff_seconds * 2.0,
                        float(args.restart_max_seconds),
                    )
            if (
                now - last_manifest_write
                >= float(args.manifest_heartbeat_seconds)
            ):
                status = (
                    "running_private_blocked"
                    if services["private"].blocked_reason
                    else "running"
                )
                write_manifest(status)
                last_manifest_write = now
            time.sleep(0.25)
    except Exception as exc:
        exit_code = 2
        stop_reason = "supervisor_exception"
        terminal_error = repr(exc)
    finally:
        try:
            _stop_services(services, float(args.shutdown_timeout_seconds))
        except Exception as exc:
            exit_code = 2
            stop_reason = "shutdown_exception"
            terminal_error = f"{terminal_error + '; ' if terminal_error else ''}{exc!r}"
        try:
            had_restarts = any(
                service.unexpected_exits for service in services.values()
            )
            private_blocked = services["private"].blocked_reason is not None
            final_status = "failed" if exit_code else (
                "complete_private_blocked"
                if private_blocked
                else (
                    "complete_with_restarts" if had_restarts else "complete"
                )
            )
            write_manifest(
                final_status,
                completed_at_utc=_utc_now_iso(),
                stop_reason=stop_reason,
                exit_code=exit_code,
                error=terminal_error,
            )
        finally:
            os.close(lock_descriptor)
            lock_path.unlink(missing_ok=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    return supervise(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
