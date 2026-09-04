"""Run fresh private telemetry -> live verdict -> guarded action routing.

The controller treats a freshly validated private-telemetry cycle as the
authority for the active bot set.  Each iteration:

1. loads an explicitly supplied Chrome-plugin cycle manifest, or attempts the
   legacy dedicated-CDP collector when that acquisition mode is selected;
2. falls back only to a still-fresh complete cycle in legacy CDP mode;
3. builds a scanner registry from the exact symbol/strategy IDs in that cycle;
4. runs one live-decision scanner tick and validates its output cardinality;
5. writes ADJUST/END action intents; and
6. automatically invokes the canonical ADJUST executor when an exact external
   approval and an extension-transport command are already configured.

The automatic path remains fail-closed: it handles ADJUST only, discovers one
exact unexpired external approval per intent, requires explicit executor
transport arguments, and accepts success only after a new Chrome telemetry
cycle proves the requested change.  The legacy reviewed-executable path remains
available through ``--allow-actions`` and ``--action-executable``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANONICAL_ADJUST_EXECUTOR = ROOT / "scripts" / "execute_live_telemetry_modifications.py"
ACTION_APPROVAL_SCHEMA = "neutralgrid_action_approval_v1"
PLUGIN_CYCLE_SCHEMA = "neutralgrid_private_telemetry_cycle_v2"
PLUGIN_SNAPSHOT_SCHEMA = "neutralgrid_private_telemetry_snapshot_v2"
DEFAULT_DEBUG_ENDPOINT = "http://127.0.0.1:9222"

from _bot_data_extractor_core import parse_user_text
from neutralgrid.live.decision.private_telemetry import (
    PrivateTelemetryParseError,
    parse_private_telemetry_text,
)
from neutralgrid.live.decision.pnl_history import (
    PnlHistoryError,
    append_pnl_observation,
    build_pnl_observation,
    load_pnl_observations,
)
from neutralgrid.live.decision.pnl_forecast import (
    PnlForecastError,
    predict_shadow_pnl,
)
from neutralgrid.live.decision.volatility import (
    VolatilityError,
    load_price_store_frame,
    load_volatility_contract,
)
from neutralgrid.live.decision.volatility_forecast import (
    predict_shadow_volatility,
)


UTC = timezone.utc
logger = logging.getLogger(__name__)


class ControllerError(RuntimeError):
    """Fail-closed controller error."""


@dataclass(frozen=True)
class TelemetryBot:
    symbol: str
    strategy_id: str
    captured_at_utc: datetime
    raw_text_path: Path
    scanner_entry: dict[str, Any]


@dataclass(frozen=True)
class TelemetryCycle:
    manifest_path: Path
    started_at_utc: datetime
    completed_at_utc: datetime
    bots: tuple[TelemetryBot, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(bot.symbol for bot in self.bots)

    @property
    def strategy_ids(self) -> tuple[str, ...]:
        return tuple(bot.strategy_id for bot in self.bots)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ControllerError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ControllerError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ControllerError(f"{field} must include a timezone: {value!r}")
    return parsed.astimezone(UTC)


def _require_finite_float(data: Mapping[str, Any], key: str, *, positive: bool = False) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ControllerError(f"telemetry field {key} is unavailable")
    number = float(value)
    if not math.isfinite(number):
        raise ControllerError(f"telemetry field {key} is non-finite")
    if positive and number <= 0:
        raise ControllerError(f"telemetry field {key} must be positive")
    return number


def _resolve_cycle_file(path_value: Any, *, manifest_path: Path) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ControllerError(f"{manifest_path}: cycle file path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ControllerError(f"{manifest_path}: cycle file does not exist: {path}")
    return path


def _scanner_entry_from_raw(
    *,
    raw_text: str,
    expected_symbol: str,
    expected_strategy_id: str,
    captured_at_utc: datetime,
    source: str = "dedicated_chrome_cdp",
) -> dict[str, Any]:
    parsed = parse_user_text(raw_text)
    symbol = str(parsed.get("symbol", "")).upper()
    strategy_id = str(parsed.get("strategy_id", ""))
    if symbol != expected_symbol:
        raise ControllerError(
            f"drawer symbol mismatch: manifest={expected_symbol}, parsed={symbol or 'missing'}"
        )
    if strategy_id != expected_strategy_id:
        raise ControllerError(
            "drawer strategy mismatch: "
            f"manifest={expected_strategy_id}, parsed={strategy_id or 'missing'}"
        )

    deploy_ts = parsed.get("start_time_utc")
    if not isinstance(deploy_ts, datetime):
        raise ControllerError(f"{symbol}: Time Created is unavailable")
    if deploy_ts.tzinfo is None:
        raise ControllerError(f"{symbol}: Time Created lacks timezone authority")

    grid_lower = _require_finite_float(parsed, "price_range_low", positive=True)
    grid_upper = _require_finite_float(parsed, "price_range_high", positive=True)
    if grid_lower >= grid_upper:
        raise ControllerError(
            f"{symbol}: invalid grid bounds {grid_lower} >= {grid_upper}"
        )
    num_grids = int(_require_finite_float(parsed, "grids_count", positive=True))
    leverage = _require_finite_float(parsed, "leverage", positive=True)
    capital_usdt = _require_finite_float(
        parsed, "invested_margin_usdt", positive=True
    )

    try:
        private_telemetry = parse_private_telemetry_text(raw_text)
    except PrivateTelemetryParseError as exc:
        raise ControllerError(f"{symbol}: private telemetry parse failed: {exc}") from exc

    return {
        "symbol": symbol,
        "strategy_id": strategy_id,
        "deploy_ts": deploy_ts.astimezone(UTC).isoformat(),
        "grid_lower": grid_lower,
        "grid_upper": grid_upper,
        "num_grids": num_grids,
        "leverage": leverage,
        "capital_usdt": capital_usdt,
        "execution_telemetry": {
            "source": source,
            "captured_at": captured_at_utc.isoformat(),
            **private_telemetry,
        },
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ControllerError(f"{label} contains non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except ControllerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ControllerError(f"{path}: {label} must be an object")
    return payload


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_strict_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def load_complete_cycle(
    manifest_path: Path,
    *,
    allowed_live_root: Path | None = None,
) -> TelemetryCycle:
    """Load and validate one complete private telemetry cycle."""

    manifest_path = manifest_path.resolve()
    payload = _load_json_object(manifest_path, label="cycle manifest")
    if payload.get("status") != "complete":
        raise ControllerError(f"{manifest_path}: telemetry cycle is not complete")
    source = str(payload.get("source") or "dedicated_chrome_cdp")
    plugin_cycle = source == "chrome_plugin"
    resolved_live_root: Path | None = None
    if plugin_cycle:
        if payload.get("schema_version") != PLUGIN_CYCLE_SCHEMA:
            raise ControllerError(f"{manifest_path}: unsupported plugin cycle schema")
        declared_live_root = payload.get("live_root")
        if not isinstance(declared_live_root, str) or not declared_live_root.strip():
            raise ControllerError(f"{manifest_path}: plugin cycle live_root is missing")
        resolved_live_root = Path(declared_live_root).resolve()
        if allowed_live_root is not None:
            expected_live_root = allowed_live_root.resolve()
            if resolved_live_root != expected_live_root:
                raise ControllerError(
                    f"{manifest_path}: plugin cycle live_root mismatch"
                )
        if payload.get("working_row_count") != payload.get("active_bot_count"):
            raise ControllerError(
                f"{manifest_path}: Working-row count does not match active count"
            )

    started_at = _parse_datetime(
        payload.get("cycle_started_at_utc"), field="cycle_started_at_utc"
    )
    completed_at = _parse_datetime(
        payload.get("cycle_completed_at_utc"), field="cycle_completed_at_utc"
    )
    if completed_at < started_at:
        raise ControllerError(f"{manifest_path}: cycle completion predates start")

    files = payload.get("files")
    symbols = payload.get("symbols")
    count = payload.get("active_bot_count")
    if not isinstance(files, list) or not isinstance(symbols, list):
        raise ControllerError(f"{manifest_path}: files/symbols must be lists")
    if not isinstance(count, int) or count <= 0:
        raise ControllerError(f"{manifest_path}: active_bot_count must be positive")
    if len(files) != count or len(symbols) != count:
        raise ControllerError(
            f"{manifest_path}: active count does not match files/symbols"
        )

    expected_symbols = tuple(str(value).upper() for value in symbols)
    if len(set(expected_symbols)) != count:
        raise ControllerError(f"{manifest_path}: active symbols are duplicated")

    bots: list[TelemetryBot] = []
    for item in files:
        if not isinstance(item, dict):
            raise ControllerError(f"{manifest_path}: cycle file entry is not an object")
        symbol = str(item.get("symbol", "")).upper()
        strategy_id = str(item.get("strategy_id", ""))
        if symbol not in expected_symbols or not strategy_id:
            raise ControllerError(
                f"{manifest_path}: invalid symbol/strategy entry {symbol}/{strategy_id}"
            )
        text_path = _resolve_cycle_file(
            item.get("text_path"), manifest_path=manifest_path
        )
        metadata_path = _resolve_cycle_file(
            item.get("json_path"), manifest_path=manifest_path
        )
        if plugin_cycle and resolved_live_root is not None:
            cycle_files = [text_path, metadata_path]
            screenshot_value = item.get("screenshot_path")
            if screenshot_value is not None:
                cycle_files.append(
                    _resolve_cycle_file(screenshot_value, manifest_path=manifest_path)
                )
            for cycle_file in cycle_files:
                if not _is_relative_to(cycle_file, resolved_live_root):
                    raise ControllerError(
                        f"{manifest_path}: plugin cycle file outside live_root: "
                        f"{cycle_file}"
                    )
        metadata = _load_json_object(metadata_path, label="snapshot metadata")
        if str(metadata.get("symbol", "")).upper() != symbol:
            raise ControllerError(f"{metadata_path}: metadata symbol mismatch")
        if str(metadata.get("strategy_id", "")) != strategy_id:
            raise ControllerError(f"{metadata_path}: metadata strategy mismatch")
        if plugin_cycle:
            if metadata.get("schema_version") != PLUGIN_SNAPSHOT_SCHEMA:
                raise ControllerError(f"{metadata_path}: unsupported snapshot schema")
            if metadata.get("source") != "chrome_plugin":
                raise ControllerError(f"{metadata_path}: plugin source mismatch")
            if metadata.get("page_identity") != payload.get("page_identity"):
                raise ControllerError(f"{metadata_path}: page identity mismatch")
            if metadata.get("source_url") != payload.get("source_url"):
                raise ControllerError(f"{metadata_path}: source URL mismatch")
        captured_at = _parse_datetime(
            metadata.get("captured_at_utc"), field=f"{symbol}.captured_at_utc"
        )
        try:
            raw_bytes = text_path.read_bytes()
            raw_text = raw_bytes.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ControllerError(f"{text_path}: raw capture is not valid UTF-8") from exc
        if plugin_cycle:
            expected_hash = metadata.get("raw_text_sha256")
            item_hash = item.get("raw_text_sha256")
            if (
                not _is_strict_sha256(expected_hash)
                or expected_hash != item_hash
            ):
                raise ControllerError(f"{metadata_path}: invalid raw-text hash contract")
            observed_hash = hashlib.sha256(raw_bytes).hexdigest()
            if observed_hash != expected_hash:
                raise ControllerError(f"{text_path}: raw capture SHA-256 mismatch")
            screenshot_path_value = metadata.get("screenshot_path")
            screenshot_hash = metadata.get("screenshot_sha256")
            item_screenshot_path = item.get("screenshot_path")
            item_screenshot_hash = item.get("screenshot_sha256")
            screenshot_contract = (
                screenshot_path_value,
                screenshot_hash,
                item_screenshot_path,
                item_screenshot_hash,
            )
            if any(value is not None for value in screenshot_contract):
                if not all(isinstance(value, str) for value in screenshot_contract):
                    raise ControllerError(
                        f"{metadata_path}: incomplete screenshot hash contract"
                    )
                if (
                    screenshot_path_value != item_screenshot_path
                    or screenshot_hash != item_screenshot_hash
                    or not _is_strict_sha256(screenshot_hash)
                ):
                    raise ControllerError(
                        f"{metadata_path}: inconsistent screenshot hash contract"
                    )
                screenshot_path = _resolve_cycle_file(
                    screenshot_path_value,
                    manifest_path=manifest_path,
                )
                if resolved_live_root is not None and not _is_relative_to(
                    screenshot_path, resolved_live_root
                ):
                    raise ControllerError(
                        f"{metadata_path}: screenshot is outside live_root"
                    )
                if hashlib.sha256(screenshot_path.read_bytes()).hexdigest() != screenshot_hash:
                    raise ControllerError(
                        f"{screenshot_path}: screenshot SHA-256 mismatch"
                    )
        entry = _scanner_entry_from_raw(
            raw_text=raw_text,
            expected_symbol=symbol,
            expected_strategy_id=strategy_id,
            captured_at_utc=captured_at,
            source=source,
        )
        if plugin_cycle and metadata.get("structured_telemetry") != entry:
            raise ControllerError(
                f"{metadata_path}: structured telemetry does not match raw capture"
            )
        bots.append(
            TelemetryBot(
                symbol=symbol,
                strategy_id=strategy_id,
                captured_at_utc=captured_at,
                raw_text_path=text_path,
                scanner_entry=entry,
            )
        )

    if tuple(bot.symbol for bot in bots) != expected_symbols:
        raise ControllerError(f"{manifest_path}: file order/symbol order mismatch")
    if len({bot.strategy_id for bot in bots}) != count:
        raise ControllerError(f"{manifest_path}: strategy IDs are duplicated")
    if plugin_cycle:
        targets_path = _resolve_cycle_file(
            payload.get("collector_targets_csv"), manifest_path=manifest_path
        )
        audit_root = manifest_path.parent.parent.resolve()
        if not _is_relative_to(targets_path, audit_root):
            raise ControllerError(
                f"{manifest_path}: collector targets resolve outside cycle audit root"
            )
        targets_hash = payload.get("collector_targets_sha256")
        if not _is_strict_sha256(targets_hash):
            raise ControllerError(f"{manifest_path}: collector targets hash is invalid")
        targets_bytes = targets_path.read_bytes()
        if hashlib.sha256(targets_bytes).hexdigest() != targets_hash:
            raise ControllerError(f"{targets_path}: collector targets SHA-256 mismatch")
        try:
            decoded_targets = targets_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ControllerError(f"{targets_path}: collector targets are not UTF-8") from exc
        reader = csv.DictReader(decoded_targets.splitlines())
        target_identities = [
            (
                str(row.get("symbol", "")).strip().upper(),
                str(row.get("strategy_id", "")).strip(),
                str(row.get("deploy_ts", "")).strip(),
            )
            for row in reader
        ]
        cycle_identities = [
            (
                bot.symbol,
                bot.strategy_id,
                str(bot.scanner_entry.get("deploy_ts", "")),
            )
            for bot in bots
        ]
        if target_identities != cycle_identities:
            raise ControllerError(
                f"{targets_path}: collector targets do not match cycle identities"
            )
    return TelemetryCycle(
        manifest_path=manifest_path.resolve(),
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        bots=tuple(bots),
    )


def latest_complete_cycle(audit_dir: Path) -> TelemetryCycle:
    paths = sorted(
        (audit_dir / "cycles").glob("cycle_*.json"),
        key=lambda path: path.name,
        reverse=True,
    )
    if not paths:
        raise ControllerError(f"no complete telemetry cycles under {audit_dir}")
    return load_complete_cycle(paths[0])


def validate_cycle_freshness(
    cycle: TelemetryCycle,
    *,
    now: datetime,
    max_age_seconds: float,
) -> None:
    age_seconds = (now - cycle.completed_at_utc).total_seconds()
    if age_seconds < -5:
        raise ControllerError(
            f"telemetry cycle is in the future by {-age_seconds:.1f}s"
        )
    if age_seconds > max_age_seconds:
        raise ControllerError(
            f"telemetry cycle is stale: age={age_seconds:.1f}s "
            f"> max={max_age_seconds:.1f}s"
        )


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    stdin_text: str | None = None,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )


def fetch_private_cycle(args: argparse.Namespace) -> TelemetryCycle:
    """Run one read-only Chrome cycle and return the newly committed manifest."""

    started_at = _utc_now()
    command = [
        str(args.python),
        "scripts/collect_private_grid_telemetry.py",
        "--debug-endpoint",
        args.debug_endpoint,
        "--timeout-seconds",
        str(args.browser_timeout_seconds),
        "--hover-seconds",
        str(args.hover_seconds),
        "--live-root",
        str(args.live_root),
        "--audit-dir",
        str(args.telemetry_audit_dir),
        "--once",
    ]
    result = _run_process(
        command,
        cwd=ROOT,
        timeout_seconds=args.fetch_timeout_seconds,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no process output"
        raise ControllerError(
            f"private telemetry fetch failed with exit {result.returncode}: {detail}"
        )
    cycle = latest_complete_cycle(args.telemetry_audit_dir)
    if cycle.completed_at_utc < started_at.replace(microsecond=0):
        raise ControllerError(
            "private telemetry command returned without committing a new cycle"
        )
    return cycle


def acquire_cycle(args: argparse.Namespace, *, now: datetime) -> tuple[TelemetryCycle, str]:
    if getattr(args, "acquisition_mode", "cdp") == "plugin-manifest":
        manifest_path = getattr(args, "cycle_manifest", None)
        if not isinstance(manifest_path, Path):
            raise ControllerError("plugin-manifest acquisition requires --cycle-manifest")
        audit_dir = getattr(args, "telemetry_audit_dir", manifest_path.parent.parent)
        resolved_manifest = manifest_path.resolve()
        resolved_cycle_dir = (Path(audit_dir).resolve() / "cycles")
        if not _is_relative_to(resolved_manifest, resolved_cycle_dir):
            raise ControllerError(
                "plugin cycle manifest must resolve inside telemetry audit cycles directory"
            )
        cycle = load_complete_cycle(
            resolved_manifest,
            allowed_live_root=Path(args.live_root),
        )
        validate_cycle_freshness(
            cycle, now=now, max_age_seconds=args.max_telemetry_age_seconds
        )
        return cycle, "chrome_plugin_manifest"
    try:
        cycle = fetch_private_cycle(args)
        validate_cycle_freshness(
            cycle, now=now, max_age_seconds=args.max_telemetry_age_seconds
        )
        return cycle, "fresh_fetch"
    except (ControllerError, subprocess.TimeoutExpired) as fetch_error:
        logger.warning("fresh private telemetry unavailable: %s", fetch_error)
        try:
            cached = latest_complete_cycle(args.telemetry_audit_dir)
            validate_cycle_freshness(
                cached, now=now, max_age_seconds=args.max_telemetry_age_seconds
            )
        except ControllerError as cache_error:
            raise ControllerError(
                f"fresh fetch failed ({fetch_error}); no usable cache ({cache_error})"
            ) from cache_error
        return cached, "fresh_cache"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_scanner_registry(cycle: TelemetryCycle, *, runtime_dir: Path) -> Path:
    payload = {"bots": [bot.scanner_entry for bot in cycle.bots]}
    target = runtime_dir / "active_bots_current.yaml"
    _atomic_write_text(
        target,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
    )
    return target


def attach_l2_streams(
    cycle: TelemetryCycle,
    *,
    manifest_paths: Sequence[Path],
    max_age_seconds: float,
    history_window_seconds: float,
    deterioration_min_duration_seconds: float = 60.0,
    deterioration_min_observations: int = 3,
    deterioration_fraction: float = 0.80,
) -> TelemetryCycle:
    """Attach exact collector/run identities to the active scanner entries."""

    if not manifest_paths:
        return cycle
    active_symbols = set(cycle.symbols)
    active_strategy_by_symbol = {
        bot.symbol: bot.strategy_id for bot in cycle.bots
    }
    refs: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        resolved_manifest = manifest_path.resolve()
        try:
            payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(
                f"cannot read diff-depth manifest {resolved_manifest}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ControllerError(f"{resolved_manifest}: manifest must be an object")
        run_id = payload.get("run_id")
        run_dirs = payload.get("symbol_run_dirs")
        if not isinstance(run_id, str) or not run_id:
            raise ControllerError(f"{resolved_manifest}: run_id is missing")
        if not isinstance(run_dirs, dict):
            raise ControllerError(f"{resolved_manifest}: symbol_run_dirs is missing")
        target_strategy_by_symbol: dict[str, str] = {}
        targets = payload.get("targets")
        if isinstance(targets, list):
            for raw_target in targets:
                if not isinstance(raw_target, dict):
                    raise ControllerError(
                        f"{resolved_manifest}: collector target must be an object"
                    )
                target_symbol = str(raw_target.get("symbol", "")).upper()
                target_strategy = str(
                    raw_target.get("strategy_id")
                    or raw_target.get("strategy_number")
                    or ""
                ).strip()
                if target_symbol:
                    if target_symbol in target_strategy_by_symbol:
                        raise ControllerError(
                            f"{resolved_manifest}: duplicate collector target {target_symbol}"
                        )
                    target_strategy_by_symbol[target_symbol] = target_strategy
        for raw_symbol, raw_run_dir in run_dirs.items():
            symbol = str(raw_symbol).upper()
            if symbol not in active_symbols:
                continue
            if symbol in refs:
                raise ControllerError(f"duplicate diff-depth stream for {symbol}")
            if not isinstance(raw_run_dir, str) or not raw_run_dir.strip():
                raise ControllerError(f"{resolved_manifest}: invalid run dir for {symbol}")
            run_dir = Path(raw_run_dir)
            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir
            run_dir = run_dir.resolve()
            feature_path = run_dir / "l2_risk_snapshots.jsonl"
            public_trade_path = run_dir / "public_agg_trades.jsonl"
            symbol_manifest = run_dir / "manifest.json"
            if not feature_path.is_file():
                raise ControllerError(
                    f"{symbol}: collector has no L2 risk derivatives at {feature_path}"
                )
            if not symbol_manifest.is_file():
                raise ControllerError(f"{symbol}: symbol manifest is missing")
            try:
                symbol_payload = json.loads(
                    symbol_manifest.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ControllerError(
                    f"{symbol}: invalid symbol manifest {symbol_manifest}: {exc}"
                ) from exc
            if (
                not isinstance(symbol_payload, dict)
                or str(symbol_payload.get("symbol", "")).upper() != symbol
                or symbol_payload.get("run_id") != run_id
            ):
                raise ControllerError(f"{symbol}: symbol manifest identity mismatch")
            expected_strategy = active_strategy_by_symbol[symbol]
            if public_trade_path.is_file():
                target_strategy = target_strategy_by_symbol.get(symbol, "")
                symbol_target = symbol_payload.get("target")
                symbol_target_strategy = (
                    str(
                        symbol_target.get("strategy_id")
                        or symbol_target.get("strategy_number")
                        or ""
                    ).strip()
                    if isinstance(symbol_target, dict)
                    else ""
                )
                if (
                    target_strategy != expected_strategy
                    or symbol_target_strategy != expected_strategy
                ):
                    raise ControllerError(
                        f"{symbol}: public-trade collector target mismatch: "
                        f"active={expected_strategy}, run={target_strategy or 'missing'}, "
                        f"symbol={symbol_target_strategy or 'missing'}"
                    )
            refs[symbol] = {
                "feature_path": str(feature_path),
                "public_trade_path": (
                    str(public_trade_path) if public_trade_path.is_file() else None
                ),
                "manifest_path": str(symbol_manifest),
                "symbol": symbol,
                "strategy_id": expected_strategy,
                "run_id": run_id,
                "max_age_seconds": max_age_seconds,
                "history_window_seconds": history_window_seconds,
                "deterioration_min_duration_seconds": deterioration_min_duration_seconds,
                "deterioration_min_observations": deterioration_min_observations,
                "deterioration_fraction": deterioration_fraction,
            }

    missing = sorted(active_symbols - set(refs))
    if missing:
        raise ControllerError(
            f"diff-depth manifests do not cover active symbols: {missing}"
        )
    return replace(
        cycle,
        bots=tuple(
            replace(
                bot,
                scanner_entry={**bot.scanner_entry, "l2_stream": refs[bot.symbol]},
            )
            for bot in cycle.bots
        ),
    )


def attach_private_event_streams(
    cycle: TelemetryCycle,
    *,
    manifest_paths: Sequence[Path],
    max_age_seconds: float,
    history_window_seconds: float,
) -> TelemetryCycle:
    """Attach exact symbol/strategy private-event streams to scanner inputs."""

    if not manifest_paths:
        return cycle
    active = {(bot.symbol, bot.strategy_id) for bot in cycle.bots}
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        resolved_manifest = manifest_path.resolve()
        try:
            payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ControllerError(
                f"cannot read private event manifest {resolved_manifest}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ControllerError(f"{resolved_manifest}: manifest must be an object")
        symbol = str(payload.get("symbol", "")).upper()
        strategy_id = str(payload.get("strategy_id", ""))
        identity = (symbol, strategy_id)
        if identity not in active:
            raise ControllerError(
                f"{resolved_manifest}: private event identity is not active: "
                f"{symbol}/{strategy_id or 'missing'}"
            )
        if identity in refs:
            raise ControllerError(
                f"duplicate private event stream for {symbol}/{strategy_id}"
            )
        run_id = payload.get("run_id")
        event_path_raw = payload.get("event_path")
        if not isinstance(run_id, str) or not run_id:
            raise ControllerError(f"{resolved_manifest}: run_id is missing")
        if not isinstance(event_path_raw, str) or not event_path_raw.strip():
            raise ControllerError(f"{resolved_manifest}: event_path is missing")
        event_path = Path(event_path_raw)
        if not event_path.is_absolute():
            event_path = resolved_manifest.parent / event_path
        event_path = event_path.resolve()
        if not event_path.is_file():
            raise ControllerError(
                f"{resolved_manifest}: private event stream does not exist: {event_path}"
            )
        refs[identity] = {
            "event_path": str(event_path),
            "manifest_path": str(resolved_manifest),
            "symbol": symbol,
            "strategy_id": strategy_id,
            "run_id": run_id,
            "max_age_seconds": max_age_seconds,
            "history_window_seconds": history_window_seconds,
        }

    missing = sorted(active - set(refs))
    if missing:
        raise ControllerError(
            f"private event manifests do not cover active bots: {missing}"
        )
    return replace(
        cycle,
        bots=tuple(
            replace(
                bot,
                scanner_entry={
                    **bot.scanner_entry,
                    "private_event_stream": refs[(bot.symbol, bot.strategy_id)],
                },
            )
            for bot in cycle.bots
        ),
    )


def run_scanner_tick(
    args: argparse.Namespace,
    *,
    cycle: TelemetryCycle,
    registry_path: Path,
    iteration_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cycle_logs = args.controller_audit_dir / "scanner" / iteration_id
    cycle_logs.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        "live_decision_scanner.py",
        "--bots",
        str(registry_path),
        "--once",
        "--no-discord",
        "--logs-dir",
        str(cycle_logs),
        "--state-dir",
        str(args.scanner_state_dir),
    ]
    if args.scanner_config_file is not None:
        command.extend(["--config-file", str(args.scanner_config_file)])
    result = _run_process(
        command,
        cwd=ROOT,
        timeout_seconds=args.scanner_timeout_seconds,
    )
    process_evidence = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    _atomic_write_json(cycle_logs / "process.json", process_evidence)
    if result.returncode != 0:
        raise ControllerError(
            f"scanner failed with exit {result.returncode}; "
            f"see {cycle_logs / 'process.json'}"
        )

    log_paths = sorted(cycle_logs.glob("live_decisions_*.jsonl"))
    if len(log_paths) != 1:
        raise ControllerError(
            f"scanner produced {len(log_paths)} JSONL files, expected exactly one"
        )
    rows: list[dict[str, Any]] = []
    for line in log_paths[0].read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ControllerError(f"scanner JSONL is invalid: {exc}") from exc
        if not isinstance(row, dict):
            raise ControllerError("scanner JSONL row is not an object")
        rows.append(row)

    expected = {(bot.symbol, bot.strategy_id) for bot in cycle.bots}
    observed = {
        (str(row.get("symbol", "")).upper(), str(row.get("strategy_id", "")))
        for row in rows
    }
    if len(rows) != len(cycle.bots) or observed != expected:
        raise ControllerError(
            "scanner output does not match active telemetry set: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    return rows, process_evidence


def persist_cycle_pnl_history(
    cycle: TelemetryCycle,
    rows: Sequence[Mapping[str, Any]],
    *,
    live_root: Path,
) -> list[dict[str, Any]]:
    """Commit one immutable private-PnL observation per exact active bot.

    This happens before any action routing.  A missing/conflicting/corrupt PnL
    record therefore blocks the iteration instead of letting a live action run
    without its prospective evidence being durable.
    """

    rows_by_identity = {
        (str(row.get("symbol", "")).upper(), str(row.get("strategy_id", ""))): row
        for row in rows
    }
    outcomes: list[dict[str, Any]] = []
    for bot in cycle.bots:
        row = rows_by_identity.get((bot.symbol, bot.strategy_id))
        if row is None:
            raise ControllerError(
                f"{bot.symbol}/{bot.strategy_id}: scanner row missing for PnL persistence"
            )
        deploy_ts = _parse_datetime(
            bot.scanner_entry.get("deploy_ts"),
            field=f"{bot.symbol}.deploy_ts",
        )
        try:
            snapshot_hash = hashlib.sha256(bot.raw_text_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ControllerError(
                f"{bot.symbol}: cannot hash private telemetry snapshot: {exc}"
            ) from exc
        try:
            observation = build_pnl_observation(
                row,
                deploy_ts=deploy_ts,
                source_cycle_manifest=str(cycle.manifest_path),
                source_snapshot_path=str(bot.raw_text_path),
                source_snapshot_sha256=snapshot_hash,
            )
            result = append_pnl_observation(observation, live_root=live_root)
        except PnlHistoryError as exc:
            raise ControllerError(
                f"{bot.symbol}/{bot.strategy_id}: PnL history commit failed: {exc}"
            ) from exc
        outcomes.append(
            {
                "symbol": bot.symbol,
                "strategy_id": bot.strategy_id,
                "status": result.status,
                "path": str(result.path),
                "observation_id": result.observation_id,
                "bot_identity": result.bot_identity,
                "history_count": result.history_count,
            }
        )
    return outcomes


def build_cycle_shadow_forecasts(
    cycle: TelemetryCycle,
    *,
    live_root: Path,
    artifact_dir: Path | None,
) -> list[dict[str, Any]]:
    """Evaluate an optional OOS-eligible artifact without touching verdict rows."""

    if artifact_dir is None:
        return [
            {
                "symbol": bot.symbol,
                "strategy_id": bot.strategy_id,
                "status": "not_configured",
                "runtime_effect": "none",
            }
            for bot in cycle.bots
        ]
    forecasts: list[dict[str, Any]] = []
    for bot in cycle.bots:
        deploy_ts = _parse_datetime(
            bot.scanner_entry.get("deploy_ts"),
            field=f"{bot.symbol}.deploy_ts",
        )
        try:
            observations = load_pnl_observations(
                live_root=live_root,
                symbol=bot.symbol,
                strategy_id=bot.strategy_id,
                deploy_ts=deploy_ts,
            )
            forecast = predict_shadow_pnl(artifact_dir, observations)
        except (PnlHistoryError, PnlForecastError, OSError) as exc:
            forecasts.append(
                {
                    "symbol": bot.symbol,
                    "strategy_id": bot.strategy_id,
                    "status": "unavailable",
                    "runtime_effect": "none",
                    "reason": str(exc),
                }
            )
            continue
        forecasts.append(forecast)
    return forecasts


def build_cycle_shadow_volatility_forecasts(
    cycle: TelemetryCycle,
    *,
    artifact_dir: Path | None,
    contract_path: Path,
    price_store: Path,
    requested_horizon_minutes: int,
    max_data_age_seconds: float,
    asof_utc: datetime,
) -> list[dict[str, Any]]:
    """Read optional validated volatility artifacts without touching verdict rows."""

    if artifact_dir is None:
        return [
            {
                "symbol": bot.symbol,
                "strategy_id": bot.strategy_id,
                "status": "not_configured",
                "eligibility": False,
                "verdict_influence": False,
                "runtime_effect": "none",
            }
            for bot in cycle.bots
        ]
    try:
        contract = load_volatility_contract(contract_path)
    except VolatilityError as exc:
        return [
            {
                "symbol": bot.symbol,
                "strategy_id": bot.strategy_id,
                "status": "unavailable",
                "reason": str(exc),
                "eligibility": False,
                "verdict_influence": False,
                "runtime_effect": "none",
            }
            for bot in cycle.bots
        ]
    effective_max_data_age_seconds = min(
        max_data_age_seconds,
        float(contract.maximum_source_age_seconds),
    )
    forecasts: list[dict[str, Any]] = []
    for bot in cycle.bots:
        try:
            mark_frame = load_price_store_frame(
                price_store,
                symbol=bot.symbol,
                series_kind=contract.primary_series,
            )
            last_frame = load_price_store_frame(
                price_store,
                symbol=bot.symbol,
                series_kind=contract.diagnostic_series,
            )
            forecast = predict_shadow_volatility(
                artifact_dir,
                contract=contract,
                symbol=bot.symbol,
                strategy_id=bot.strategy_id,
                mark_frame=mark_frame,
                last_frame=last_frame,
                requested_horizon_minutes=requested_horizon_minutes,
                asof_utc=cast(pd.Timestamp, pd.Timestamp(asof_utc)),
            )
            freshness = float(forecast["freshness_seconds"])
            if (
                not math.isfinite(freshness)
                or freshness > effective_max_data_age_seconds
            ):
                raise VolatilityError(
                    f"{bot.symbol}: volatility source is stale: {freshness:.3f}s"
                )
        except (VolatilityError, OSError, ValueError) as exc:
            forecasts.append(
                {
                    "symbol": bot.symbol,
                    "strategy_id": bot.strategy_id,
                    "status": "unavailable",
                    "reason": str(exc),
                    "eligibility": False,
                    "verdict_influence": False,
                    "runtime_effect": "none",
                }
            )
            continue
        forecasts.append(forecast)
    return forecasts


def _finite_optional(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def build_action_intents(
    rows: Sequence[Mapping[str, Any]],
    *,
    iteration_id: str,
) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    for row in rows:
        verdict = str(row.get("verdict", "")).upper()
        if verdict == "CONTINUE":
            continue
        if verdict not in {"ADJUST", "END"}:
            raise ControllerError(f"unsupported scanner verdict: {verdict!r}")
        symbol = str(row.get("symbol", "")).upper()
        strategy_id = str(row.get("strategy_id", ""))
        if not symbol or not strategy_id:
            raise ControllerError("action verdict lacks symbol/strategy_id")
        lower = _finite_optional(row.get("suggested_grid_lower"))
        upper = _finite_optional(row.get("suggested_grid_upper"))
        if verdict == "ADJUST":
            if lower is None or upper is None or lower <= 0 or lower >= upper:
                raise ControllerError(
                    f"{symbol}: ADJUST lacks valid suggested grid bounds"
                )
        else:
            lower = None
            upper = None
        key_material = (
            f"{strategy_id}|{verdict}|{lower if lower is not None else ''}|"
            f"{upper if upper is not None else ''}"
        )
        intents.append(
            {
                "schema_version": "neutralgrid_action_intent_v1",
                "iteration_id": iteration_id,
                "idempotency_key": hashlib.sha256(
                    key_material.encode("utf-8")
                ).hexdigest(),
                "symbol": symbol,
                "strategy_id": strategy_id,
                "action": verdict,
                "suggested_grid_lower": lower,
                "suggested_grid_upper": upper,
                "reasons": list(row.get("reasons") or []),
                "decision_ts": row.get("ts"),
            }
        )
    return intents


def _load_verified_action_keys(ledger_path: Path) -> set[str]:
    if not ledger_path.is_file():
        return set()
    keys: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") == "verified":
            key = row.get("idempotency_key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_action_effect(intent: Mapping[str, Any], cycle: TelemetryCycle) -> None:
    strategy_id = str(intent["strategy_id"])
    active_by_strategy = {bot.strategy_id: bot for bot in cycle.bots}
    action = str(intent["action"])
    if action == "END":
        if strategy_id in active_by_strategy:
            raise ControllerError(
                f"{intent['symbol']}: END was not verified; strategy remains active"
            )
        return
    bot = active_by_strategy.get(strategy_id)
    if bot is None:
        raise ControllerError(
            f"{intent['symbol']}: ADJUST was not verified; strategy is no longer active"
        )
    lower = float(intent["suggested_grid_lower"])
    upper = float(intent["suggested_grid_upper"])
    observed_lower = float(bot.scanner_entry["grid_lower"])
    observed_upper = float(bot.scanner_entry["grid_upper"])
    if (
        not math.isclose(observed_lower, lower, rel_tol=1e-6, abs_tol=1e-12)
        or not math.isclose(observed_upper, upper, rel_tol=1e-6, abs_tol=1e-12)
    ):
        raise ControllerError(
            f"{intent['symbol']}: ADJUST was not verified; "
            f"requested=({lower}, {upper}), "
            f"observed=({observed_lower}, {observed_upper})"
        )


def _verification_intent_from_executor(
    intent: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate executor evidence and return the exact post-check contract."""

    verification_intent = dict(intent)
    if str(intent.get("action")) != "ADJUST":
        return verification_intent
    if response.get("schema_version") != "neutralgrid_action_execution_v1":
        raise ControllerError("executor response schema is unsupported")
    if response.get("preserve_current_position") is not True:
        raise ControllerError("executor did not prove current-position preservation")
    additional = _finite_optional(response.get("additional_investment"))
    if additional is None or not math.isclose(additional, 0.0, abs_tol=1e-12):
        raise ControllerError("executor reported unexpected additional investment")
    executed_lower = _finite_optional(response.get("executed_grid_lower"))
    executed_upper = _finite_optional(response.get("executed_grid_upper"))
    decimals = response.get("price_decimals")
    if (
        executed_lower is None
        or executed_upper is None
        or executed_lower <= 0
        or executed_lower >= executed_upper
        or not isinstance(decimals, int)
        or isinstance(decimals, bool)
        or decimals < 0
        or decimals > 16
    ):
        raise ControllerError("executor returned invalid exchange-applied bounds")
    requested_lower = float(intent["suggested_grid_lower"])
    requested_upper = float(intent["suggested_grid_upper"])
    quantum = 10.0 ** (-decimals)
    tolerance = max(1e-12, quantum * 1e-9)
    lower_delta = requested_lower - executed_lower
    upper_delta = executed_upper - requested_upper
    if (
        lower_delta < -tolerance
        or upper_delta < -tolerance
        or lower_delta >= quantum + tolerance
        or upper_delta >= quantum + tolerance
    ):
        raise ControllerError(
            "executor bounds are not a valid single-quantum outward rounding: "
            f"requested=({requested_lower}, {requested_upper}), "
            f"executed=({executed_lower}, {executed_upper}), "
            f"precision={decimals}"
        )
    verification_intent["suggested_grid_lower"] = executed_lower
    verification_intent["suggested_grid_upper"] = executed_upper
    return verification_intent


def _resolve_exact_action_approval(
    approval_dir: Path,
    intent: Mapping[str, Any],
    *,
    now: datetime,
) -> Path:
    """Return one exact, unexpired external approval for an ADJUST intent."""

    if str(intent.get("action")) != "ADJUST":
        raise ControllerError("automatic action routing supports ADJUST only")
    if not approval_dir.is_dir():
        raise ControllerError(f"action approval directory is unavailable: {approval_dir}")

    expected_lower = _finite_optional(intent.get("suggested_grid_lower"))
    expected_upper = _finite_optional(intent.get("suggested_grid_upper"))
    if expected_lower is None or expected_upper is None:
        raise ControllerError("ADJUST intent lacks finite bounds")

    matches: list[Path] = []
    for path in sorted(approval_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            if payload.get("schema_version") != ACTION_APPROVAL_SCHEMA:
                continue
            if payload.get("idempotency_key") != intent.get("idempotency_key"):
                continue
            if payload.get("symbol") != intent.get("symbol"):
                continue
            if str(payload.get("strategy_id")) != str(intent.get("strategy_id")):
                continue
            if payload.get("action") != "ADJUST":
                continue
            if payload.get("preserve_current_position") is not True:
                continue
            lower = _finite_optional(payload.get("suggested_grid_lower"))
            upper = _finite_optional(payload.get("suggested_grid_upper"))
            if lower is None or upper is None:
                continue
            if not math.isclose(lower, expected_lower, rel_tol=0.0, abs_tol=1e-12):
                continue
            if not math.isclose(upper, expected_upper, rel_tol=0.0, abs_tol=1e-12):
                continue
            approved_at = _parse_datetime(
                payload.get("approved_at_utc"), field="approved_at_utc"
            )
            expires_at = _parse_datetime(
                payload.get("expires_at_utc"), field="expires_at_utc"
            )
            if approved_at > now or expires_at <= now:
                continue
            if (expires_at - approved_at).total_seconds() > 300.0:
                continue
        except (ControllerError, OSError, json.JSONDecodeError):
            continue
        matches.append(path.resolve())

    if not matches:
        raise ControllerError(
            f"{intent.get('symbol')}: exact unexpired {ACTION_APPROVAL_SCHEMA} is missing"
        )
    if len(matches) != 1:
        raise ControllerError(
            f"{intent.get('symbol')}: exact action approval is ambiguous ({len(matches)} matches)"
        )
    return matches[0]


def route_actions(
    args: argparse.Namespace,
    *,
    intents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ledger_path = args.controller_audit_dir / "action_ledger.jsonl"
    verified_keys = _load_verified_action_keys(ledger_path)
    outcomes: list[dict[str, Any]] = []
    for intent in intents:
        key = str(intent["idempotency_key"])
        base = {**intent, "routed_at_utc": _utc_now().isoformat()}
        if key in verified_keys:
            outcome = {**base, "status": "skipped_already_verified"}
            _append_jsonl(ledger_path, outcome)
            outcomes.append(outcome)
            continue
        auto_approval_dir = getattr(args, "action_approval_dir", None)
        action_executable = args.action_executable
        command: list[str]
        if not args.allow_actions and auto_approval_dir is not None:
            try:
                approval_path = _resolve_exact_action_approval(
                    Path(auto_approval_dir), intent, now=_utc_now()
                )
            except ControllerError as exc:
                outcome = {
                    **base,
                    "status": "blocked_exact_approval_unavailable",
                    "approval_error": str(exc),
                }
                _append_jsonl(ledger_path, outcome)
                outcomes.append(outcome)
                continue
            if not args.action_arg:
                outcome = {
                    **base,
                    "status": "blocked_auto_adjust_transport_not_configured",
                    "approval_file": str(approval_path),
                }
                _append_jsonl(ledger_path, outcome)
                outcomes.append(outcome)
                continue
            command = [
                str(args.python),
                str(CANONICAL_ADJUST_EXECUTOR),
                "--approval-file",
                str(approval_path),
                *args.action_arg,
            ]
        else:
            if not args.allow_actions:
                outcome = {**base, "status": "blocked_allow_actions_not_set"}
                _append_jsonl(ledger_path, outcome)
                outcomes.append(outcome)
                continue
            if action_executable is None:
                outcome = {**base, "status": "blocked_no_action_executable"}
                _append_jsonl(ledger_path, outcome)
                outcomes.append(outcome)
                continue
            command = [str(action_executable), *args.action_arg]
        result = _run_process(
            command,
            cwd=ROOT,
            stdin_text=json.dumps(intent),
            timeout_seconds=args.action_timeout_seconds,
        )
        execution = {
            **base,
            "executor_command": command,
            "executor_returncode": result.returncode,
            "executor_stdout": result.stdout,
            "executor_stderr": result.stderr,
        }
        if result.returncode != 0:
            outcome = {**execution, "status": "executor_failed"}
            _append_jsonl(ledger_path, outcome)
            outcomes.append(outcome)
            continue
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            outcome = {**execution, "status": "executor_response_invalid"}
            _append_jsonl(ledger_path, outcome)
            outcomes.append(outcome)
            continue
        if (
            not isinstance(response, dict)
            or response.get("idempotency_key") != key
            or response.get("status") != "executed"
        ):
            outcome = {
                **execution,
                "status": "executor_response_mismatch",
                "executor_response": response,
            }
            _append_jsonl(ledger_path, outcome)
            outcomes.append(outcome)
            continue

        try:
            verification_intent = _verification_intent_from_executor(
                intent,
                response,
            )
            post_cycle = fetch_private_cycle(args)
            _verify_action_effect(verification_intent, post_cycle)
        except (ControllerError, subprocess.TimeoutExpired) as exc:
            outcome = {
                **execution,
                "status": "post_action_verification_failed",
                "verification_error": str(exc),
            }
            _append_jsonl(ledger_path, outcome)
            outcomes.append(outcome)
            continue
        outcome = {
            **execution,
            "status": "verified",
            "verified_at_utc": _utc_now().isoformat(),
            "post_cycle_manifest": str(post_cycle.manifest_path),
            "executor_response": response,
        }
        _append_jsonl(ledger_path, outcome)
        outcomes.append(outcome)
        verified_keys.add(key)
    return outcomes


def route_or_observe_actions(
    args: argparse.Namespace,
    *,
    intents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve action intents without dispatch when observation-only is set."""

    if not getattr(args, "observational_only", False):
        return route_actions(args, intents=intents)
    return [
        {
            "symbol": str(intent.get("symbol", "")),
            "strategy_id": str(intent.get("strategy_id", "")),
            "action": str(intent.get("action", "")),
            "idempotency_key": str(intent.get("idempotency_key", "")),
            "status": "observational_not_executed",
            "runtime_effect": "none",
        }
        for intent in intents
    ]


def run_iteration(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _utc_now()
    iteration_id = started_at.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    report_path = args.controller_audit_dir / "iterations" / f"{iteration_id}.json"
    report: dict[str, Any] = {
        "schema_version": "neutralgrid_private_telemetry_controller_v1",
        "iteration_id": iteration_id,
        "started_at_utc": started_at.isoformat(),
        "status": "running",
    }
    try:
        cycle, source = acquire_cycle(args, now=started_at)
        cycle = attach_l2_streams(
            cycle,
            manifest_paths=args.diff_depth_manifest,
            max_age_seconds=args.max_l2_age_seconds,
            history_window_seconds=args.l2_history_window_seconds,
            deterioration_min_duration_seconds=(
                args.l2_deterioration_min_duration_seconds
            ),
            deterioration_min_observations=(
                args.l2_deterioration_min_observations
            ),
            deterioration_fraction=args.l2_deterioration_fraction,
        )
        cycle = attach_private_event_streams(
            cycle,
            manifest_paths=args.private_event_manifest,
            max_age_seconds=args.max_private_event_age_seconds,
            history_window_seconds=args.private_event_history_window_seconds,
        )
        registry = write_scanner_registry(cycle, runtime_dir=args.runtime_dir)
        rows, process_evidence = run_scanner_tick(
            args,
            cycle=cycle,
            registry_path=registry,
            iteration_id=iteration_id,
        )
        pnl_history = persist_cycle_pnl_history(
            cycle,
            rows,
            live_root=args.live_root,
        )
        shadow_forecasts = build_cycle_shadow_forecasts(
            cycle,
            live_root=args.live_root,
            artifact_dir=args.pnl_forecast_artifact_dir,
        )
        shadow_volatility_forecasts = build_cycle_shadow_volatility_forecasts(
            cycle,
            artifact_dir=getattr(args, "volatility_forecast_artifact_dir", None),
            contract_path=getattr(
                args,
                "volatility_contract",
                ROOT / "config" / "live_volatility_forecast_v1.json",
            ),
            price_store=getattr(args, "volatility_price_store", ROOT / "data" / "price_store"),
            requested_horizon_minutes=getattr(args, "volatility_horizon_minutes", 360),
            max_data_age_seconds=getattr(args, "max_volatility_data_age_seconds", 180.0),
            asof_utc=started_at,
        )
        intents = build_action_intents(rows, iteration_id=iteration_id)
        action_outcomes = route_or_observe_actions(args, intents=intents)
        report.update(
            {
                "status": "complete",
                "cycle_status": "complete",
                "training_readiness": "not_evaluated_by_controller",
                "observational_only": bool(
                    getattr(args, "observational_only", False)
                ),
                "telemetry_source": source,
                "telemetry_manifest": str(cycle.manifest_path),
                "telemetry_completed_at_utc": cycle.completed_at_utc.isoformat(),
                "active_symbols": list(cycle.symbols),
                "active_strategy_ids": list(cycle.strategy_ids),
                "l2_streams_attached": sum(
                    1 for bot in cycle.bots if "l2_stream" in bot.scanner_entry
                ),
                "private_event_streams_attached": sum(
                    1
                    for bot in cycle.bots
                    if "private_event_stream" in bot.scanner_entry
                ),
                "registry_path": str(registry),
                "scanner_returncode": process_evidence["returncode"],
                "pnl_history": pnl_history,
                "pnl_history_appended": sum(
                    item["status"] == "appended" for item in pnl_history
                ),
                "pnl_history_duplicates": sum(
                    item["status"] == "duplicate" for item in pnl_history
                ),
                "shadow_pnl_forecasts": shadow_forecasts,
                "shadow_pnl_forecasts_available": sum(
                    item["status"] == "available" for item in shadow_forecasts
                ),
                "shadow_volatility": {
                    "schema_version": "neutralgrid_shadow_volatility_controller_section_v1",
                    "verdict_influence": False,
                    "forecasts": shadow_volatility_forecasts,
                    "available_count": sum(
                        item["status"] == "available"
                        for item in shadow_volatility_forecasts
                    ),
                },
                "verdicts": [
                    {
                        "symbol": row.get("symbol"),
                        "strategy_id": row.get("strategy_id"),
                        "verdict": row.get("verdict"),
                        "reasons": row.get("reasons"),
                        "suggested_grid_lower": row.get("suggested_grid_lower"),
                        "suggested_grid_upper": row.get("suggested_grid_upper"),
                    }
                    for row in rows
                ],
                "action_outcomes": action_outcomes,
            }
        )
    except (ControllerError, subprocess.TimeoutExpired) as exc:
        report.update(
            {
                "status": "blocked",
                "cycle_status": "blocked",
                "training_readiness": "not_evaluated_by_controller",
                "error": str(exc),
            }
        )
    report["completed_at_utc"] = _utc_now().isoformat()
    _atomic_write_json(report_path, report)
    _atomic_write_json(args.controller_audit_dir / "manifest.json", report)
    return report


def _acquire_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, ValueError, IndexError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
            except OSError:
                pass
            else:
                raise ControllerError(f"controller already running with PID {pid}")
        lock_path.unlink(missing_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, f"{os.getpid()}\n{_utc_now().isoformat()}\n".encode("ascii"))
    return descriptor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acquisition-mode",
        choices=("cdp", "plugin-manifest"),
        default="cdp",
    )
    parser.add_argument(
        "--cycle-manifest",
        type=Path,
        default=None,
        help="Fresh immutable cycle manifest for plugin-manifest acquisition.",
    )
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    parser.add_argument("--max-telemetry-age-seconds", type=float, default=900.0)
    parser.add_argument("--debug-endpoint", default=None)
    parser.add_argument("--browser-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--hover-seconds", type=float, default=0.35)
    parser.add_argument("--fetch-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--scanner-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--action-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".venv" / "Scripts" / "python.exe",
    )
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--telemetry-audit-dir",
        type=Path,
        default=ROOT / "outputs" / "audits" / "private_telemetry_loop_current",
    )
    parser.add_argument(
        "--controller-audit-dir",
        type=Path,
        default=ROOT / "outputs" / "audits" / "live_telemetry_controller_current",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / "outputs" / "runtime" / "live_telemetry_controller",
    )
    parser.add_argument(
        "--scanner-state-dir",
        type=Path,
        default=ROOT / "outputs" / "runtime" / "live_telemetry_controller" / "state",
    )
    parser.add_argument("--scanner-config-file", type=Path, default=None)
    parser.add_argument(
        "--diff-depth-manifest",
        type=Path,
        action="append",
        default=[],
        help="repeatable event-complete collector manifest covering active symbols",
    )
    parser.add_argument("--max-l2-age-seconds", type=float, default=15.0)
    parser.add_argument("--l2-history-window-seconds", type=float, default=300.0)
    parser.add_argument(
        "--l2-deterioration-min-duration-seconds",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--l2-deterioration-min-observations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--l2-deterioration-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--private-event-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "repeatable exact symbol/strategy canonical private-event manifest; "
            "when supplied, every active bot must be covered"
        ),
    )
    parser.add_argument("--max-private-event-age-seconds", type=float, default=600.0)
    parser.add_argument(
        "--private-event-history-window-seconds",
        type=float,
        default=86400.0,
    )
    parser.add_argument(
        "--pnl-forecast-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Optional bot-disjoint OOS-validated shadow PnL artifact. Forecasts "
            "are reported only and never alter verdicts or action intents."
        ),
    )
    parser.add_argument(
        "--volatility-forecast-artifact-dir",
        type=Path,
        default=None,
        help=(
            "Optional frozen-holdout-eligible shadow volatility artifact. "
            "Reported in a separate verdict-inert manifest section."
        ),
    )
    parser.add_argument(
        "--volatility-contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_forecast_v1.json",
    )
    parser.add_argument(
        "--volatility-price-store",
        type=Path,
        default=ROOT / "data" / "price_store",
    )
    parser.add_argument("--volatility-horizon-minutes", type=int, default=360)
    parser.add_argument(
        "--max-volatility-data-age-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--allow-actions",
        action="store_true",
        help="Allow routing ADJUST/END to the configured external executor.",
    )
    parser.add_argument(
        "--observational-only",
        action="store_true",
        help="Persist verdicts and PnL evidence without routing any action intent.",
    )
    parser.add_argument(
        "--action-executable",
        type=Path,
        default=None,
        help=(
            "Separately reviewed executable receiving one action-intent JSON object "
            "on stdin and returning an executed acknowledgement on stdout."
        ),
    )
    parser.add_argument(
        "--action-approval-dir",
        type=Path,
        default=ROOT / "outputs" / "runtime" / "live_telemetry_controller" / "action_approvals",
        help=(
            "Directory containing separate exact unexpired neutralgrid_action_approval_v1 "
            "files. When one exact approval matches an ADJUST intent, invoke the canonical "
            "executor automatically using --action-arg transport settings. END is excluded."
        ),
    )
    parser.add_argument("--action-arg", action="append", default=[])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "interval_seconds",
        "max_telemetry_age_seconds",
        "browser_timeout_seconds",
        "fetch_timeout_seconds",
        "scanner_timeout_seconds",
        "action_timeout_seconds",
        "max_l2_age_seconds",
        "l2_history_window_seconds",
        "max_private_event_age_seconds",
        "private_event_history_window_seconds",
        "max_volatility_data_age_seconds",
    ):
        if float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be > 0")
    if args.hover_seconds < 0:
        parser.error("--hover-seconds must be >= 0")
    if not 0 < args.volatility_horizon_minutes <= 360:
        parser.error("--volatility-horizon-minutes must lie in (0, 360]")
    if args.l2_deterioration_min_duration_seconds < 0:
        parser.error("--l2-deterioration-min-duration-seconds must be >= 0")
    if args.l2_deterioration_min_observations < 2:
        parser.error("--l2-deterioration-min-observations must be >= 2")
    if not 0.5 <= args.l2_deterioration_fraction <= 1.0:
        parser.error("--l2-deterioration-fraction must be in [0.5, 1.0]")
    if args.action_executable is not None and not args.allow_actions:
        parser.error("--action-executable requires --allow-actions")
    if args.acquisition_mode == "plugin-manifest":
        if args.cycle_manifest is None:
            parser.error("--cycle-manifest is required in plugin-manifest mode")
        if args.debug_endpoint is not None:
            parser.error("--debug-endpoint is prohibited in plugin-manifest mode")
        if not args.observational_only:
            parser.error("plugin-manifest mode requires --observational-only")
        if args.allow_actions or args.action_executable is not None or args.action_arg:
            parser.error("action routing is prohibited in plugin-manifest mode")
    else:
        if args.cycle_manifest is not None:
            parser.error("--cycle-manifest requires plugin-manifest mode")
        args.debug_endpoint = args.debug_endpoint or DEFAULT_DEBUG_ENDPOINT
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    lock_path = args.controller_audit_dir / "controller.lock"
    stop_path = args.controller_audit_dir / "STOP"
    descriptor = _acquire_lock(lock_path)
    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    exit_code = 0
    try:
        while True:
            report = run_iteration(args)
            print(json.dumps(report, indent=2, sort_keys=True))
            if report["status"] != "complete":
                exit_code = 2
            if args.once:
                return exit_code
            if stop_requested or stop_path.exists():
                return exit_code
            deadline = time.monotonic() + args.interval_seconds
            while time.monotonic() < deadline:
                if stop_requested or stop_path.exists():
                    return exit_code
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
