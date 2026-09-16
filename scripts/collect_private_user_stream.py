"""Collect Binance USD-M authenticated user-data events as a private service.

The listen key exists only in process memory. Every received text frame is
hashed in its original form, redacted, and made durable before JSON/schema
parsing. Canonical strategy events require exact strategy identity or a
reviewed exchange-order linkage artifact; symbol/time proximity is rejected as
ownership evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.api.binance_client import BinanceClient  # noqa: E402
from neutralgrid.data.diff_depth import atomic_write_json  # noqa: E402
from neutralgrid.data.private_user_stream import (  # noqa: E402
    PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION,
    PrivateStrategyStorage,
    PrivateUserPayloadError,
    PrivateUserStreamError,
    StrategyTarget,
    load_exact_order_linkages,
    normalize_private_user_event,
    redact_secret,
    validate_target,
)
from scripts.collect_diff_depth import _git_output  # noqa: E402


logger = logging.getLogger(__name__)
LIMA = ZoneInfo("America/Lima")
DEFAULT_PRIVATE_WS_BASE = "wss://fstream.binance.com/private/ws"
DEFAULT_LIFECYCLE_WS_URL = "wss://ws-fapi.binance.com/ws-fapi/v1"


@dataclass(frozen=True)
class PrivateRunResult:
    status: str
    connections: int
    wire_events: int
    canonical_records: int
    parse_errors: int
    coverage_gaps: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat()


def _sanitize(value: object, listen_key: str | None) -> str:
    return redact_secret(str(value), listen_key)[0]


def _sanitize_value(value: Any, listen_key: str | None) -> Any:
    if isinstance(value, str):
        return redact_secret(value, listen_key)[0]
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item, listen_key)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, listen_key) for item in value]
    return value


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_private_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise PrivateUserStreamError(
                f"private user-data collector already running with PID {existing_pid}"
            )
        lock_path.unlink(missing_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    os.fsync(descriptor)
    return descriptor


def load_private_targets(path: Path) -> list[StrategyTarget]:
    """Load unique exact symbol/strategy pairs; multiple bots may share a symbol."""

    targets: list[StrategyTarget] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            if "symbol" not in fieldnames or not (
                {"strategy_id", "strategy_number"} & fieldnames
            ):
                raise PrivateUserStreamError(
                    f"{path} must contain symbol and strategy_id/strategy_number"
                )
            for row_index, row in enumerate(reader, start=2):
                strategy_id = (
                    str(row.get("strategy_id", "")).strip()
                    or str(row.get("strategy_number", "")).strip()
                )
                try:
                    target = validate_target(str(row.get("symbol", "")), strategy_id)
                except PrivateUserStreamError as exc:
                    raise PrivateUserStreamError(
                        f"invalid private target at {path}:{row_index}: {exc}"
                    ) from exc
                targets.append(target)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PrivateUserStreamError(f"cannot read private target roster {path}: {exc}") from exc
    if not targets:
        raise PrivateUserStreamError("private target roster contains no rows")
    seen: set[StrategyTarget] = set()
    for target in targets:
        if target in seen:
            raise PrivateUserStreamError(f"duplicate private target: {target}")
        seen.add(target)
    return targets


class PrivateUserCollector:
    """Reconnect, rotate, normalize, and persist one account user-data stream."""

    def __init__(
        self,
        *,
        targets: Sequence[StrategyTarget],
        order_lookup: Mapping[tuple[str, str], StrategyTarget],
        storages: Mapping[StrategyTarget, PrivateStrategyStorage],
        session: aiohttp.ClientSession,
        client: BinanceClient,
        args: argparse.Namespace,
        run_id: str,
        stop_event: asyncio.Event,
        deadline_monotonic: float | None,
        write_run_manifest: Any,
    ) -> None:
        self.targets = tuple(targets)
        self.order_lookup = order_lookup
        self.storages = storages
        self.session = session
        self.client = client
        self.args = args
        self.run_id = run_id
        self.stop_event = stop_event
        self.deadline_monotonic = deadline_monotonic
        self.write_run_manifest = write_run_manifest
        self.started_at_utc = _utc_now_iso()
        self.current_connection_id: str | None = None
        self.current_listen_key: str | None = None
        self.last_event_at_utc: str | None = None
        self.last_error: str | None = None
        self.gap_started_at_utc: str | None = None
        self.counters: dict[str, int] = {
            "connections": 0,
            "listen_keys_created": 0,
            "listen_key_keepalives": 0,
            "listen_key_close_failures": 0,
            "wire_events": 0,
            "canonical_records": 0,
            "duplicate_records_dropped": 0,
            "parse_errors": 0,
            "raw_only_events": 0,
            "unlinked_order_events": 0,
            "coverage_gaps": 0,
            "listen_key_expirations": 0,
        }

    def _time_remaining(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - time.monotonic()

    def _should_stop(self) -> bool:
        remaining = self._time_remaining()
        return self.stop_event.is_set() or (
            remaining is not None and remaining <= 0
        )

    def _targets_for_symbols(
        self, source_symbols: Sequence[str]
    ) -> tuple[StrategyTarget, ...]:
        if not source_symbols:
            return self.targets
        symbol_set = set(source_symbols)
        return tuple(target for target in self.targets if target.symbol in symbol_set)

    def _manifest_payload(self, status: str) -> dict[str, Any]:
        return {
            "status": status,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": _utc_now_iso(),
            "last_event_at_utc": self.last_event_at_utc,
            "current_connection_id": self.current_connection_id,
            "listen_key_present_in_memory": self.current_listen_key is not None,
            "last_error": self.last_error,
            "event_completeness": "unknown",
            "traffic_class": "private",
            "runtime_effect": "observational_only",
            "service_counters": dict(self.counters),
        }

    async def _write_manifests(self, status: str, *, force: bool = False) -> None:
        payload = self._manifest_payload(status)
        for storage in self.storages.values():
            await asyncio.to_thread(storage.write_manifest, payload)
        await self.write_run_manifest(status, force=force, collector=payload)

    def _append_control(self, kind: str, details: Mapping[str, Any]) -> None:
        safe_details = {
            key: _sanitize_value(value, self.current_listen_key)
            for key, value in details.items()
        }
        for storage in self.storages.values():
            storage.append_control(kind, safe_details)

    async def run(self) -> PrivateRunResult:
        backoff = float(self.args.reconnect_base_seconds)
        await self._write_manifests("starting", force=True)
        while not self._should_stop():
            connection_id = f"private-{uuid.uuid4().hex}"
            self.current_connection_id = connection_id
            reason = "connection_exception"
            try:
                reason = await self._run_connection(connection_id)
                backoff = float(self.args.reconnect_base_seconds)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                safe_error = _sanitize(repr(exc), self.current_listen_key)
                self.last_error = safe_error
                self._append_control(
                    "connection_exception",
                    {
                        "connection_id": connection_id,
                        "at_utc": _utc_now_iso(),
                        "error": safe_error,
                    },
                )
                logger.error("Private user-data connection failed: %s", safe_error)
            finally:
                self.current_listen_key = None
                self.current_connection_id = None
            await self._write_manifests("running", force=True)
            if self._should_stop() or reason == "capture_complete":
                break
            if self.gap_started_at_utc is None:
                self.gap_started_at_utc = _utc_now_iso()
                self.counters["coverage_gaps"] += 1
                self._append_control(
                    "coverage_gap_started",
                    {
                        "at_utc": self.gap_started_at_utc,
                        "reason": reason,
                    },
                )
            sleep_seconds = min(backoff, float(self.args.reconnect_max_seconds))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=sleep_seconds)
            except TimeoutError:
                pass
            backoff = min(backoff * 2.0, float(self.args.reconnect_max_seconds))

        if self.counters["parse_errors"] or self.counters["coverage_gaps"]:
            status = "complete_with_labelled_gaps"
        elif self.counters["connections"] == 0:
            status = "complete_without_connection"
        else:
            status = "complete_contiguous_observation"
        await self._write_manifests(status, force=True)
        return PrivateRunResult(
            status=status,
            connections=self.counters["connections"],
            wire_events=self.counters["wire_events"],
            canonical_records=self.counters["canonical_records"],
            parse_errors=self.counters["parse_errors"],
            coverage_gaps=self.counters["coverage_gaps"],
        )

    async def _run_connection(self, connection_id: str) -> str:
        listen_key = await self.client.start_user_data_stream()
        self.current_listen_key = listen_key
        self.counters["listen_keys_created"] += 1
        ws_url = f"{str(self.args.ws_base).rstrip('/')}/{listen_key}"
        ws: aiohttp.ClientWebSocketResponse | None = None
        wire_sequence = 0
        reason = "private_connection_closed"
        try:
            ws = await self.session.ws_connect(
                ws_url,
                heartbeat=float(self.args.heartbeat_seconds),
                autoping=True,
                max_msg_size=int(self.args.max_message_bytes),
            )
            self.counters["connections"] += 1
            connected_monotonic = time.monotonic()
            next_keepalive = connected_monotonic + float(
                self.args.keepalive_seconds
            )
            if self.gap_started_at_utc is not None:
                self._append_control(
                    "coverage_gap_ended",
                    {
                        "gap_started_at_utc": self.gap_started_at_utc,
                        "at_utc": _utc_now_iso(),
                    },
                )
                self.gap_started_at_utc = None
            self._append_control(
                "connection_established",
                {"connection_id": connection_id, "at_utc": _utc_now_iso()},
            )
            await self._write_manifests("running", force=True)
            while not self._should_stop():
                now = time.monotonic()
                if now - connected_monotonic >= float(self.args.rotation_seconds):
                    reason = "scheduled_rotation"
                    break
                if now >= next_keepalive:
                    await self.client.keepalive_user_data_stream(listen_key)
                    self.counters["listen_key_keepalives"] += 1
                    next_keepalive = time.monotonic() + float(
                        self.args.keepalive_seconds
                    )
                    self._append_control(
                        "listen_key_keepalive_succeeded",
                        {"connection_id": connection_id, "at_utc": _utc_now_iso()},
                    )
                receive_timeout = min(1.0, max(0.01, next_keepalive - time.monotonic()))
                remaining = self._time_remaining()
                if remaining is not None:
                    receive_timeout = max(0.01, min(receive_timeout, remaining))
                try:
                    message = await ws.receive(timeout=receive_timeout)
                except TimeoutError:
                    await self._write_manifests("running")
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    wire_sequence += 1
                    raw_text = str(message.data)
                    received_at = _utc_now_iso()
                    received_monotonic_ns = time.monotonic_ns()
                    original_sha256 = hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest()
                    persisted_text, credential_redacted = redact_secret(
                        raw_text, listen_key
                    )
                    for storage in self.storages.values():
                        storage.append_wire(
                            connection_id=connection_id,
                            wire_sequence=wire_sequence,
                            received_at_utc=received_at,
                            received_monotonic_ns=received_monotonic_ns,
                            original_sha256=original_sha256,
                            persisted_text=persisted_text,
                            credential_redacted=credential_redacted,
                        )
                    self.counters["wire_events"] += 1
                    try:
                        decoded = json.loads(raw_text)
                        if not isinstance(decoded, dict):
                            raise PrivateUserPayloadError(
                                "private payload root must be an object"
                            )
                        result = normalize_private_user_event(
                            decoded,
                            run_id=self.run_id,
                            exact_targets=self.targets,
                            order_lookup=self.order_lookup,
                        )
                    except (json.JSONDecodeError, PrivateUserPayloadError) as exc:
                        self._record_parse_error(
                            connection_id, wire_sequence, received_at, str(exc)
                        )
                        reason = "schema_parse_error"
                        break
                    self.last_event_at_utc = received_at
                    if result.source_event_type == "listenKeyExpired":
                        self.counters["listen_key_expirations"] += 1
                        self._append_control(
                            "listen_key_expired",
                            {"connection_id": connection_id, "at_utc": received_at},
                        )
                        reason = "listen_key_expired"
                        break
                    if result.target is not None:
                        storage = self.storages[result.target]
                        for record in result.canonical_records:
                            outcome = storage.append_canonical(record)
                            if outcome == "appended":
                                self.counters["canonical_records"] += 1
                            else:
                                self.counters["duplicate_records_dropped"] += 1
                    elif result.classification == "unlinked_order_preserved_raw_only":
                        self.counters["unlinked_order_events"] += 1
                        for target in self._targets_for_symbols(result.source_symbols):
                            self.storages[target].counters["unlinked_order_events"] += 1
                    else:
                        self.counters["raw_only_events"] += 1
                        relevant = self._targets_for_symbols(result.source_symbols)
                        for target in relevant:
                            self.storages[target].counters["raw_only_events"] += 1
                    await self._write_manifests("running")
                    continue
                if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED}:
                    reason = f"websocket_closed_{ws.close_code}"
                    break
                if message.type == aiohttp.WSMsgType.ERROR:
                    reason = _sanitize(
                        f"websocket_error_{ws.exception()!r}", listen_key
                    )
                    break
                if message.type in {aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG}:
                    continue
                reason = "unexpected_websocket_frame"
                break
            if self._should_stop():
                reason = "capture_complete"
        finally:
            self._append_control(
                "connection_ended",
                {
                    "connection_id": connection_id,
                    "at_utc": _utc_now_iso(),
                    "reason": reason,
                    "wire_events": wire_sequence,
                },
            )
            if ws is not None and not ws.closed:
                await ws.close()
            try:
                await self.client.close_user_data_stream(listen_key)
            except Exception as exc:
                self.counters["listen_key_close_failures"] += 1
                self._append_control(
                    "listen_key_close_failed",
                    {
                        "connection_id": connection_id,
                        "at_utc": _utc_now_iso(),
                        "error": _sanitize(repr(exc), listen_key),
                    },
                )
        return reason

    def _record_parse_error(
        self,
        connection_id: str,
        wire_sequence: int,
        received_at_utc: str,
        error: str,
    ) -> None:
        safe_error = _sanitize(error, self.current_listen_key)
        self.counters["parse_errors"] += 1
        self.last_error = safe_error
        for storage in self.storages.values():
            storage.counters["parse_errors"] += 1
        self._append_control(
            "parse_error",
            {
                "connection_id": connection_id,
                "wire_sequence": wire_sequence,
                "at_utc": received_at_utc,
                "error": safe_error,
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV exact symbol/strategy roster"
    )
    parser.add_argument(
        "--linkage-file",
        type=Path,
        action="append",
        default=[],
        help="Reviewed neutralgrid_strategy_order_linkage_v1 JSON; repeatable",
    )
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--ingestion-date",
        help="Supervisor-frozen America/Lima date in YYYY-MM-DD form",
    )
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--ws-base", default=DEFAULT_PRIVATE_WS_BASE)
    parser.add_argument(
        "--lifecycle-ws-url", default=DEFAULT_LIFECYCLE_WS_URL
    )
    parser.add_argument("--allow-nonproduction-endpoints", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=180.0)
    parser.add_argument("--manifest-heartbeat-seconds", type=float, default=5.0)
    parser.add_argument("--keepalive-seconds", type=float, default=30 * 60)
    parser.add_argument("--rotation-seconds", type=float, default=23 * 3600)
    parser.add_argument("--reconnect-base-seconds", type=float, default=1.0)
    parser.add_argument("--reconnect-max-seconds", type=float, default=30.0)
    parser.add_argument("--max-message-bytes", type=int, default=2_000_000)
    parser.add_argument("--fsync-every", type=int, default=1)
    args = parser.parse_args(argv)
    args.ws_base = str(args.ws_base).rstrip("/")
    args.lifecycle_ws_url = str(args.lifecycle_ws_url).rstrip("/")
    if not args.allow_nonproduction_endpoints and (
        args.ws_base != DEFAULT_PRIVATE_WS_BASE
        or args.lifecycle_ws_url != DEFAULT_LIFECYCLE_WS_URL
    ):
        parser.error(
            "non-production private endpoints require "
            "--allow-nonproduction-endpoints"
        )
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0")
    if args.ingestion_date is not None:
        try:
            parsed_date = date.fromisoformat(args.ingestion_date)
        except ValueError:
            parser.error("--ingestion-date must be YYYY-MM-DD")
        if parsed_date.isoformat() != args.ingestion_date:
            parser.error("--ingestion-date must use canonical YYYY-MM-DD form")
    for field in (
        "heartbeat_seconds",
        "manifest_heartbeat_seconds",
        "keepalive_seconds",
        "rotation_seconds",
        "reconnect_base_seconds",
        "reconnect_max_seconds",
        "max_message_bytes",
    ):
        if float(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.fsync_every < 0:
        parser.error("--fsync-every must be >= 0")
    return args


async def collect_private_user_stream(args: argparse.Namespace) -> int:
    targets = load_private_targets(args.input)
    order_lookup, linkages = load_exact_order_linkages(
        args.linkage_file, expected_targets=targets
    )
    started_at = _utc_now()
    run_id = (
        started_at.strftime("private_user_stream_%Y%m%d_%H%M%S_%f")
        + f"_{os.getpid()}"
    )
    live_date = args.ingestion_date or started_at.astimezone(LIMA).strftime("%Y-%m-%d")
    audit_dir = args.audit_dir or ROOT / "outputs" / "audits" / run_id
    audit_dir.mkdir(parents=True, exist_ok=True)
    stop_path = audit_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    run_manifest = audit_dir / "manifest.json"
    live_root = args.live_root.resolve()
    client = BinanceClient(user_data_ws_api_url=args.lifecycle_ws_url)

    manifest_base: dict[str, Any] = {
        "schema_version": PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION,
        "service": "private",
        "traffic_class": "private",
        "run_id": run_id,
        "started_at_utc": started_at.replace(microsecond=0).isoformat(),
        "live_date_lima": live_date,
        "ingestion_timezone": "America/Lima",
        "ingestion_date_basis": (
            "supervisor_frozen" if args.ingestion_date else "collector_start"
        ),
        "live_root": str(live_root),
        "collector_pid": os.getpid(),
        "targets": [asdict(target) for target in targets],
        "linkages": [
            {
                "target": asdict(item.target),
                "provenance": item.provenance,
                "source_path": str(item.source_path),
                "order_id_count": len(item.order_ids),
            }
            for item in linkages
        ],
        "ws_route": str(args.ws_base),
        "lifecycle_ws_url": str(args.lifecycle_ws_url),
        "lifecycle_transport": (
            "userDataStream.start/ping/stop over Binance USD-M WebSocket API"
        ),
        "event_completeness": "unknown",
        "credential_policy": "environment/config only; listen key never durable",
        "runtime_effect": "observational_only",
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
    }
    if not client.api_key:
        atomic_write_json(
            run_manifest,
            {
                **manifest_base,
                "status": "blocked_missing_api_key",
                "updated_at_utc": _utc_now_iso(),
                "completed_at_utc": _utc_now_iso(),
                "blocker": "BINANCE_API_KEY is not configured",
            },
        )
        await client.close()
        return 3

    lock_path = audit_dir / "collector.lock"
    lock_descriptor = _acquire_private_lock(lock_path)
    stop_event = asyncio.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    storages: dict[StrategyTarget, PrivateStrategyStorage] = {}
    for target in targets:
        run_dir = (
            live_root
            / live_date
            / target.symbol
            / "private_user_stream"
            / target.strategy_id
            / run_id
        ).resolve()
        try:
            run_dir.relative_to(live_root)
        except ValueError as exc:
            raise PrivateUserStreamError(
                f"private stream path escapes live root: {run_dir}"
            ) from exc
        storages[target] = PrivateStrategyStorage(
            run_dir,
            target=target,
            run_id=run_id,
            fsync_every=int(args.fsync_every),
        )
    manifest_last_write = float("-inf")

    async def write_run_manifest(
        status: str, *, force: bool = False, **extra: Any
    ) -> None:
        nonlocal manifest_last_write
        now = time.monotonic()
        if (
            not force
            and now - manifest_last_write
            < float(args.manifest_heartbeat_seconds)
        ):
            return
        await asyncio.to_thread(
            atomic_write_json,
            run_manifest,
            {
                **manifest_base,
                "status": status,
                "updated_at_utc": _utc_now_iso(),
                "symbol_strategy_run_dirs": {
                    f"{target.symbol}:{target.strategy_id}": str(storage.run_dir)
                    for target, storage in storages.items()
                },
                "symbol_strategy_counters": {
                    f"{target.symbol}:{target.strategy_id}": dict(storage.counters)
                    for target, storage in storages.items()
                },
                **extra,
            },
        )
        manifest_last_write = time.monotonic()

    deadline = (
        time.monotonic() + float(args.duration_seconds)
        if args.duration_seconds > 0
        else None
    )
    result: PrivateRunResult | None = None
    await write_run_manifest("starting", force=True)
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            collector = PrivateUserCollector(
                targets=targets,
                order_lookup=order_lookup,
                storages=storages,
                session=session,
                client=client,
                args=args,
                run_id=run_id,
                stop_event=stop_event,
                deadline_monotonic=deadline,
                write_run_manifest=write_run_manifest,
            )
            task = asyncio.create_task(collector.run())
            while not task.done():
                if stop_path.exists():
                    stop_event.set()
                await write_run_manifest("running")
                await asyncio.sleep(0.25)
            result = await task
            await write_run_manifest(
                "complete"
                if result.status == "complete_contiguous_observation"
                else "complete_with_labelled_gaps",
                force=True,
                completed_at_utc=_utc_now_iso(),
                result=asdict(result),
            )
            return 0
    except Exception as exc:
        safe_error = _sanitize(repr(exc), None)
        await write_run_manifest(
            "failed",
            force=True,
            completed_at_utc=_utc_now_iso(),
            error=safe_error,
            result=asdict(result) if result is not None else None,
        )
        logger.error("Private user-data collector failed: %s", safe_error)
        return 2
    finally:
        stop_event.set()
        for storage in storages.values():
            storage.close()
        await client.close()
        os.close(lock_descriptor)
        lock_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(collect_private_user_stream(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
