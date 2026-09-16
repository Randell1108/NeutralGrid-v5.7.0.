"""Fail-closed static/live verifier for the three Binance WebSocket services."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.diff_depth import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION as PUBLIC_MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
)
from neutralgrid.data.market_stream import MARKET_MANIFEST_SCHEMA_VERSION  # noqa: E402
from neutralgrid.data.private_user_stream import (  # noqa: E402
    PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION,
)
from scripts.collect_diff_depth import DEFAULT_WS_BASE as DEFAULT_PUBLIC_WS_BASE  # noqa: E402
from scripts.collect_market_streams import DEFAULT_MARKET_WS_BASE  # noqa: E402
from scripts.collect_private_user_stream import (  # noqa: E402
    DEFAULT_LIFECYCLE_WS_URL,
    DEFAULT_PRIVATE_WS_BASE,
    load_private_targets,
)
from scripts.supervise_binance_websocket_services import (  # noqa: E402
    SUPERVISOR_SCHEMA_VERSION,
)


VERIFICATION_SCHEMA_VERSION = "neutralgrid_binance_ws_verification_v1"
PROMPT_PATH = ROOT / "outputs" / "audits" / "live_data_acquisition_prompt_20260910.md"


class ServiceVerificationError(RuntimeError):
    """A manifest, identity, freshness, or ownership assertion failed."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ServiceVerificationError(f"{path}: non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except ServiceVerificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ServiceVerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ServiceVerificationError(f"{path}: root must be a JSON object")
    return payload


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ServiceVerificationError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ServiceVerificationError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ServiceVerificationError(f"{field} has no timezone authority")
    return parsed.astimezone(timezone.utc)


def _is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == 5
        exit_code = ctypes.c_ulong()
        try:
            success = kernel32.GetExitCodeProcess(
                handle,
                ctypes.byref(exit_code),
            )
            return bool(success) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _manifest_targets(payload: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    raw = payload.get("targets")
    if not isinstance(raw, list):
        raise ServiceVerificationError("manifest targets must be a list")
    targets: list[tuple[str, str | None]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ServiceVerificationError("manifest target must be an object")
        symbol = str(item.get("symbol", "")).strip().upper()
        strategy_raw = item.get("strategy_id")
        strategy = (
            str(strategy_raw).strip()
            if strategy_raw is not None and str(strategy_raw).strip()
            else None
        )
        targets.append((symbol, strategy))
    return targets


def _freshness_check(
    payload: Mapping[str, Any], *, path: Path, max_age_seconds: float
) -> None:
    updated = _parse_utc(payload.get("updated_at_utc"), field=f"{path}:updated_at_utc")
    age = (_utc_now() - updated).total_seconds()
    if age < -5:
        raise ServiceVerificationError(f"{path}: heartbeat is {abs(age):.1f}s in the future")
    if age > max_age_seconds:
        raise ServiceVerificationError(
            f"{path}: heartbeat age {age:.1f}s exceeds {max_age_seconds:.1f}s"
        )


def _static_checks() -> list[Check]:
    required_files = [
        ROOT / "scripts" / "collect_diff_depth.py",
        ROOT / "scripts" / "collect_market_streams.py",
        ROOT / "scripts" / "collect_private_user_stream.py",
        ROOT / "scripts" / "supervise_binance_websocket_services.py",
        ROOT / "scripts" / "start_binance_websocket_services.ps1",
        ROOT / "scripts" / "stop_binance_websocket_services.ps1",
        PROMPT_PATH,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ServiceVerificationError(f"required files are missing: {missing}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    tokens = [
        "start_binance_websocket_services.ps1",
        "verify_binance_websocket_services.py",
        "collect_private_user_stream.py",
        "/public",
        "/market",
        "/private",
        "userDataStream.start",
        "userDataStream.ping",
        "userDataStream.stop",
        "event_completeness",
        "America/Lima",
    ]
    absent = [token for token in tokens if token not in prompt]
    if absent:
        raise ServiceVerificationError(
            f"acquisition prompt lacks operational contract tokens: {absent}"
        )
    return [
        Check(
            "static_contract",
            "PASS",
            "Prompt and all collector/supervisor entry points are present.",
        )
    ]


def _validate_service_processes(
    supervisor: Mapping[str, Any], checks: list[Check]
) -> None:
    status = str(supervisor.get("status", ""))
    running = status.startswith("running")
    if running and not _is_running(supervisor.get("supervisor_pid")):
        raise ServiceVerificationError("running supervisor PID is not live")
    services = supervisor.get("services")
    if not isinstance(services, Mapping) or set(services) != {
        "public",
        "market",
        "private",
    }:
        raise ServiceVerificationError("supervisor must own exactly three services")
    for name in ("public", "market"):
        item = services[name]
        if not isinstance(item, Mapping):
            raise ServiceVerificationError(f"{name} process record is invalid")
        if running and (
            item.get("running") is not True or not _is_running(item.get("pid"))
        ):
            raise ServiceVerificationError(f"{name} service process is not live")
    private = services["private"]
    if not isinstance(private, Mapping):
        raise ServiceVerificationError("private process record is invalid")
    if private.get("blocked_reason"):
        checks.append(
            Check(
                "private_process",
                "FAIL",
                "Private stream is blocked. Configure BINANCE_API_KEY in the existing "
                "secret mechanism and restart the supervisor; never put it on the command line.",
            )
        )
    elif running and (
        private.get("running") is not True or not _is_running(private.get("pid"))
    ):
        raise ServiceVerificationError("private service process is not live")
    else:
        checks.append(Check("process_ownership", "PASS", "One supervisor owns three services."))


def _validate_live_once(
    *,
    target_csv: Path,
    audit_root: Path,
    max_age_seconds: float,
    allow_nonproduction_endpoints: bool,
) -> list[Check]:
    checks: list[Check] = []
    targets = load_private_targets(target_csv)
    exact_targets = [(target.symbol, target.strategy_id) for target in targets]
    symbols = list(dict.fromkeys(target.symbol for target in targets))
    supervisor_path = audit_root / "manifest.json"
    supervisor = _strict_json(supervisor_path)
    if supervisor.get("schema_version") != SUPERVISOR_SCHEMA_VERSION:
        raise ServiceVerificationError("supervisor schema mismatch")
    if Path(str(supervisor.get("target_csv", ""))).resolve() != target_csv.resolve():
        raise ServiceVerificationError("supervisor target CSV path mismatch")
    actual_hash = hashlib.sha256(target_csv.read_bytes()).hexdigest()
    if supervisor.get("target_sha256") != actual_hash:
        raise ServiceVerificationError("supervisor target CSV hash mismatch")
    if _manifest_targets(supervisor) != exact_targets:
        raise ServiceVerificationError("supervisor exact target identity mismatch")
    ingestion_date = str(supervisor.get("ingestion_date_lima", ""))
    if supervisor.get("ingestion_timezone") != "America/Lima" or not ingestion_date:
        raise ServiceVerificationError("supervisor Lima ingestion authority is missing")
    if str(supervisor.get("runtime_effect")) != "observational_only":
        raise ServiceVerificationError("supervisor is not observational-only")
    if str(supervisor.get("status", "")).startswith("running"):
        _freshness_check(
            supervisor, path=supervisor_path, max_age_seconds=max_age_seconds
        )
    checks.append(Check("roster_identity", "PASS", f"{len(exact_targets)} exact targets validated."))
    checks.append(Check("shared_lima_date", "PASS", ingestion_date))
    _validate_service_processes(supervisor, checks)

    manifests = {
        name: _strict_json(audit_root / name / "manifest.json")
        for name in ("public", "market", "private")
    }
    expected_routes = {
        "public": DEFAULT_PUBLIC_WS_BASE,
        "market": DEFAULT_MARKET_WS_BASE,
        "private": DEFAULT_PRIVATE_WS_BASE,
    }
    observed_routes = {
        "public": manifests["public"].get("ws_base"),
        "market": manifests["market"].get("ws_base"),
        "private": manifests["private"].get("ws_route"),
    }
    if not allow_nonproduction_endpoints and observed_routes != expected_routes:
        raise ServiceVerificationError(
            f"route ownership mismatch: observed={observed_routes}"
        )
    lifecycle_url = manifests["private"].get("lifecycle_ws_url")
    if (
        not allow_nonproduction_endpoints
        and lifecycle_url != DEFAULT_LIFECYCLE_WS_URL
    ):
        raise ServiceVerificationError(
            f"private lifecycle route mismatch: observed={lifecycle_url!r}"
        )

    public = manifests["public"]
    if (
        public.get("schema_version") != PUBLIC_MANIFEST_SCHEMA_VERSION
        or public.get("service") != "public_diff_depth"
        or public.get("traffic_class") != "public"
        or public.get("collect_agg_trades") is not False
        or public.get("collect_mark_price_updates") is not False
    ):
        raise ServiceVerificationError("public service route/scope contract failed")
    if [symbol for symbol, _strategy in _manifest_targets(public)] != symbols:
        raise ServiceVerificationError("public service symbol roster mismatch")

    market = manifests["market"]
    if (
        market.get("schema_version") != MARKET_MANIFEST_SCHEMA_VERSION
        or market.get("service") != "market"
        or market.get("traffic_class") != "market"
    ):
        raise ServiceVerificationError("market service route/scope contract failed")
    if [symbol for symbol, _strategy in _manifest_targets(market)] != symbols:
        raise ServiceVerificationError("market service symbol roster mismatch")

    private = manifests["private"]
    if (
        private.get("schema_version") != PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION
        or private.get("service") != "private"
        or private.get("traffic_class") != "private"
        or private.get("event_completeness") != "unknown"
    ):
        raise ServiceVerificationError("private service route/scope contract failed")
    if _manifest_targets(private) != exact_targets:
        raise ServiceVerificationError("private service exact target mismatch")
    if private.get("status") == "blocked_missing_api_key":
        checks.append(
            Check(
                "private_manifest",
                "FAIL",
                "BINANCE_API_KEY is not configured; configure it via the existing "
                "secret mechanism and restart the supervisor.",
            )
        )

    live_root = Path(str(supervisor.get("live_root", ""))).resolve()
    for name, payload in manifests.items():
        manifest_path = audit_root / name / "manifest.json"
        if str(payload.get("status", "")) in {"starting", "running"}:
            _freshness_check(
                payload, path=manifest_path, max_age_seconds=max_age_seconds
            )
        if payload.get("live_date_lima") != ingestion_date:
            raise ServiceVerificationError(f"{name} service ingestion date mismatch")
        if payload.get("ingestion_timezone") != "America/Lima":
            raise ServiceVerificationError(f"{name} service timezone mismatch")
        run_dirs = payload.get("symbol_run_dirs") or payload.get(
            "symbol_strategy_run_dirs"
        )
        if run_dirs is None and name == "private" and payload.get("status") == "blocked_missing_api_key":
            continue
        if not isinstance(run_dirs, Mapping) or not run_dirs:
            raise ServiceVerificationError(f"{name} service run directories missing")
        for value in run_dirs.values():
            run_dir = Path(str(value)).resolve()
            if not _under(run_dir, live_root) or ingestion_date not in run_dir.parts:
                raise ServiceVerificationError(
                    f"{name} run directory violates Live/Lima boundary: {run_dir}"
                )
            if name == "private" and str(payload.get("status", "")) in {
                "starting",
                "running",
            }:
                runtime = _strict_json(run_dir / "manifest.json")
                counters = runtime.get("service_counters")
                if not isinstance(counters, Mapping):
                    raise ServiceVerificationError(
                        f"private collector counters are missing for {run_dir}"
                    )
                connections = counters.get("connections")
                if not isinstance(connections, int) or connections < 1:
                    last_error = runtime.get("last_error")
                    raise ServiceVerificationError(
                        "private collector has not established a user-data connection"
                        + (f": {last_error}" if last_error else "")
                    )
    checks.append(
        Check(
            "exclusive_routes",
            "PASS",
            "/public=diff-depth, /market=trades+marks+klines, /private=user data.",
        )
    )
    checks.append(
        Check(
            "storage_boundary",
            "PASS",
            "All available run directories are beneath the shared Lima Live date.",
        )
    )

    public_counters = public.get("symbol_counters", {})
    if not isinstance(public_counters, Mapping):
        raise ServiceVerificationError("public symbol counters are invalid")
    for symbol, raw in public_counters.items():
        if not isinstance(raw, Mapping):
            raise ServiceVerificationError(f"public counters invalid for {symbol}")
        if raw.get("public_agg_trades", 0) != 0 or raw.get(
            "public_mark_price_updates", 0
        ) != 0:
            raise ServiceVerificationError(
                f"public service crossed into market ownership for {symbol}"
            )
    checks.append(Check("public_market_separation", "PASS", "No market records in public service."))
    return checks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["static", "live"], default="live")
    parser.add_argument("--target-csv", type=Path)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=ROOT / "outputs" / "audits" / "binance_websocket_services_current",
    )
    parser.add_argument("--max-heartbeat-age-seconds", type=float, default=30.0)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-nonproduction-endpoints", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "live" and args.target_csv is None:
        parser.error("--target-csv is required in live mode")
    if args.max_heartbeat_age_seconds <= 0 or args.wait_seconds < 0:
        parser.error("heartbeat age must be positive and wait must be non-negative")
    return args


def verify(args: argparse.Namespace) -> int:
    checks: list[Check] = []
    error: str | None = None
    checks.extend(_static_checks())
    if args.mode == "live":
        deadline = time.monotonic() + float(args.wait_seconds)
        while True:
            try:
                checks.extend(
                    _validate_live_once(
                        target_csv=Path(args.target_csv).resolve(),
                        audit_root=Path(args.audit_root).resolve(),
                        max_age_seconds=float(args.max_heartbeat_age_seconds),
                        allow_nonproduction_endpoints=bool(
                            args.allow_nonproduction_endpoints
                        ),
                    )
                )
                break
            except ServiceVerificationError as exc:
                error = str(exc)
                if time.monotonic() >= deadline:
                    checks.append(Check("live_contract", "FAIL", error))
                    break
                time.sleep(min(0.5, max(0.01, deadline - time.monotonic())))
    passed = not any(check.status == "FAIL" for check in checks)
    report_path = args.report or Path(args.audit_root) / "verification.json"
    report = {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": _utc_now().replace(microsecond=0).isoformat(),
        "mode": args.mode,
        "checks": [asdict(check) for check in checks],
        "last_error": error,
    }
    atomic_write_json(Path(report_path), report)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return verify(parse_args(argv))
    except ServiceVerificationError as exc:
        sys.stderr.write(f"verification failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
