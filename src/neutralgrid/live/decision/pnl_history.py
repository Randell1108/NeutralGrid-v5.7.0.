"""Crash-resilient, deployment-scoped live PnL observation history.

The controller's verdict state is intentionally small and mutable.  Forecast
training needs a different contract: immutable observations that survive
controller/runtime-directory changes, preserve the exact strategy deployment,
and cannot silently reinterpret a repeated Chrome snapshot as a new PnL point.

Each observation is one atomically committed JSON file under the repository's
mandated live-data hierarchy::

    Live/YYYY-MM-DD/SYMBOL/pnl_history/<bot_identity>/observations/*.json

The date is the capture date in America/Lima.  ``observation_id`` is derived
from bot identity plus the private-telemetry capture timestamp.  Re-evaluating
the same snapshot is therefore a duplicate no-op; a different snapshot for the
same identity/timestamp is a hard conflict.  Scanner/L2 fields are evidence
attached to that PnL point and remain verdict-inert.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


PNL_OBSERVATION_SCHEMA_VERSION = "neutralgrid_live_pnl_observation_v1"
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{5,24}$")


class PnlHistoryError(RuntimeError):
    """PnL history is ambiguous, corrupt, or cannot be committed safely."""


@dataclass(frozen=True)
class AppendPnlObservationResult:
    status: str
    path: Path
    observation_id: str
    bot_identity: str
    history_count: int


def _lima_timezone() -> timezone | ZoneInfo:
    try:
        return ZoneInfo("America/Lima")
    except ZoneInfoNotFoundError:  # pragma: no cover - Windows tzdata fallback
        return timezone(-timedelta(hours=5))


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PnlHistoryError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PnlHistoryError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise PnlHistoryError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise PnlHistoryError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _finite_or_none(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PnlHistoryError(f"{field} must be numeric when present")
    number = float(value)
    if not math.isfinite(number):
        raise PnlHistoryError(f"{field} must be finite when present")
    return number


def _integer_or_none(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PnlHistoryError(f"{field} must be an integer when present")
    number = int(value)
    if float(number) != float(value):
        raise PnlHistoryError(f"{field} must be an integer when present")
    return number


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _record_integrity_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "record_integrity_sha256",
            "deploy_ts",
            "captured_at",
            "evaluated_at",
            "_source_path",
        }
    }


def pnl_bot_identity(*, symbol: str, strategy_id: str, deploy_ts: datetime) -> str:
    """Return the immutable identity for one exact strategy deployment."""

    symbol_text = symbol.strip().upper()
    strategy_text = strategy_id.strip()
    deployed_at = _require_aware_utc(deploy_ts, field="deploy_ts")
    if not _SYMBOL_RE.fullmatch(symbol_text):
        raise PnlHistoryError("symbol is not a valid live PnL identity component")
    if not strategy_text:
        raise PnlHistoryError("strategy_id is required for PnL identity")
    return hashlib.sha256(
        f"{symbol_text}|{strategy_text}|{deployed_at.isoformat()}".encode("utf-8")
    ).hexdigest()


def _numeric_subset(
    source: Mapping[str, Any],
    keys: Sequence[str],
    *,
    prefix: str,
) -> dict[str, float | None]:
    return {
        key: _finite_or_none(source.get(key), field=f"{prefix}.{key}")
        for key in keys
    }


def build_pnl_observation(
    scanner_row: Mapping[str, Any],
    *,
    deploy_ts: datetime,
    source_cycle_manifest: str,
    source_snapshot_path: str,
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    """Build one canonical observation from an exact scanner result.

    ``source_snapshot_sha256`` binds the record to the verbatim Chrome drawer.
    L2/public/private evidence is copied only when present; missing evidence is
    represented as ``None`` and is never imputed here.
    """

    symbol = str(scanner_row.get("symbol", "")).strip().upper()
    strategy_id = str(scanner_row.get("strategy_id", "")).strip()
    deployed_at = _require_aware_utc(deploy_ts, field="deploy_ts")
    if not symbol or not strategy_id:
        raise PnlHistoryError("scanner row must contain exact symbol and strategy_id")
    if not _HEX_SHA256_RE.fullmatch(source_snapshot_sha256.lower()):
        raise PnlHistoryError("source_snapshot_sha256 must be a lowercase SHA-256")

    telemetry = _mapping(scanner_row.get("execution_telemetry"))
    pnl_raw = _mapping(telemetry.get("pnl"))
    position_raw = _mapping(telemetry.get("position_inventory"))
    grid_raw = _mapping(telemetry.get("grid"))
    captured_at = _parse_utc(
        telemetry.get("captured_at"),
        field="execution_telemetry.captured_at",
    )
    total_profit = _finite_or_none(
        pnl_raw.get("total_profit_usdt"),
        field="execution_telemetry.pnl.total_profit_usdt",
    )
    if total_profit is None:
        raise PnlHistoryError("total_profit_usdt is required for a PnL observation")

    evaluation = _mapping(scanner_row.get("evaluation"))
    evaluated_at = _parse_utc(
        evaluation.get("evaluated_at_utc", scanner_row.get("ts")),
        field="evaluation.evaluated_at_utc",
    )
    if evaluated_at < captured_at - timedelta(seconds=5):
        raise PnlHistoryError("scanner evaluation materially predates telemetry capture")

    l2 = _mapping(evaluation.get("l2_risk"))
    execution = _mapping(evaluation.get("execution_risk"))
    private_events = _mapping(evaluation.get("private_event_evidence"))
    deterioration = _mapping(scanner_row.get("profit_deterioration"))

    pnl = _numeric_subset(
        pnl_raw,
        (
            "total_profit_usdt",
            "total_profit_pct",
            "matched_profit_usdt",
            "matched_profit_pct",
            "realized_profit_usdt",
            "unmatched_pnl_usdt",
            "unmatched_pnl_pct",
            "funding_fee_usdt",
            "funding_fee_pct",
            "transaction_fee_usdt",
        ),
        prefix="pnl",
    )
    position = _numeric_subset(
        position_raw,
        (
            "size_base",
            "size_usdt",
            "entry_price",
            "mark_price",
            "position_pnl_usdt",
            "position_roe_pct",
            "margin_ratio_pct",
            "liquidation_price",
            "isolated_margin_balance_usdt",
        ),
        prefix="position",
    )
    grid: dict[str, Any] = _numeric_subset(
        grid_raw,
        (
            "price_range_lower",
            "price_range_upper",
            "profit_per_grid_pct",
            "invested_margin_usdt",
            "qty_per_order_base",
            "current_leverage",
            "grid_start_price",
        ),
        prefix="grid",
    )
    grid["num_grids"] = _integer_or_none(grid_raw.get("num_grids"), field="grid.num_grids")

    numeric_features: tuple[tuple[str, Mapping[str, Any], str], ...] = (
        ("price", evaluation, "price"),
        ("range_prob", evaluation, "range_prob"),
        ("trend_prob", evaluation, "trend_prob"),
        ("persistence_prob", evaluation, "persistence_prob"),
        ("meta_proba", evaluation, "meta_proba"),
        ("expected_exit_impact_bps", l2, "expected_exit_impact_bps"),
        ("exit_depth_to_position_ratio", l2, "exit_depth_to_position_ratio"),
        ("spread_bps", l2, "spread_bps"),
        ("spread_current_to_median", l2, "spread_current_to_median"),
        ("book_imbalance", l2, "book_imbalance"),
        ("exit_side_removal_to_addition_ratio", l2, "exit_side_removal_to_addition_ratio"),
        ("current_spread_bps", execution, "current_spread_bps"),
        ("baseline_spread_median_bps", execution, "baseline_spread_median_bps"),
        ("exit_depth_current_to_baseline", execution, "exit_depth_current_to_baseline"),
        ("exit_side_imbalance", execution, "exit_side_imbalance"),
        ("recent_spread_worse_fraction", execution, "recent_spread_worse_fraction"),
        ("recent_exit_depth_worse_fraction", execution, "recent_exit_depth_worse_fraction"),
        ("joint_deterioration_trailing_duration_seconds", execution, "joint_deterioration_trailing_duration_seconds"),
        ("aggressive_exit_side_trade_notional_usdt", execution, "aggressive_exit_side_trade_notional_usdt"),
        ("trade_aligned_removal_proxy_usdt", execution, "trade_aligned_removal_proxy_usdt"),
        ("unexplained_removal_proxy_usdt", execution, "unexplained_removal_proxy_usdt"),
        ("refill_proxy_usdt", execution, "refill_proxy_usdt"),
        ("exit_side_net_withdrawal_to_position_ratio", execution, "exit_side_net_withdrawal_to_position_ratio"),
        ("private_cancel_update_fraction", execution, "private_cancel_update_fraction"),
        ("mean_estimated_slippage_bps", execution, "mean_estimated_slippage_bps"),
        ("p90_estimated_slippage_bps", execution, "p90_estimated_slippage_bps"),
        ("mean_adverse_selection_5s_bps", execution, "mean_adverse_selection_5s_bps"),
        ("mean_adverse_selection_30s_bps", execution, "mean_adverse_selection_30s_bps"),
        ("peak_total_profit_usdt", deterioration, "peak_total_profit_usdt"),
        ("gain_giveback_usdt", deterioration, "giveback_usdt"),
        ("gain_giveback_pct", deterioration, "giveback_pct_of_positive_peak"),
        ("private_realized_pnl_usdt", private_events, "realized_pnl_usdt"),
        ("private_commission_usdt", private_events, "commission_usdt"),
        ("private_funding_fee_usdt", private_events, "funding_fee_usdt"),
    )
    features: dict[str, Any] = {
        output: _finite_or_none(source.get(input_key), field=f"features.{output}")
        for output, source, input_key in numeric_features
    }
    for output, source, input_key in (
        ("liquidity_state", execution, "liquidity_state"),
        ("public_trade_status", execution, "public_trade_status"),
        ("private_event_status", execution, "private_event_status"),
        ("private_event_completeness", private_events, "event_completeness"),
        ("l2_run_id", execution, "l2_run_id"),
        ("l2_segment_id", execution, "l2_segment_id"),
        ("private_event_run_id", private_events, "run_id"),
    ):
        features[output] = _clean_text(source.get(input_key))
    for output, source, input_key in (
        ("sustained_joint_deterioration", execution, "sustained_joint_deterioration"),
        ("sustained_spread_deterioration", execution, "sustained_spread_deterioration"),
        ("sustained_exit_depth_deterioration", execution, "sustained_exit_depth_deterioration"),
        ("temporary_joint_deterioration", execution, "temporary_joint_deterioration"),
    ):
        value = source.get(input_key)
        if value is not None and not isinstance(value, bool):
            raise PnlHistoryError(f"features.{output} must be boolean when present")
        features[output] = value

    bot_identity = pnl_bot_identity(
        symbol=symbol,
        strategy_id=strategy_id,
        deploy_ts=deployed_at,
    )
    observation_id = hashlib.sha256(
        f"{bot_identity}|{captured_at.isoformat()}".encode("utf-8")
    ).hexdigest()
    snapshot_basis: dict[str, Any] = {
        "bot_identity": bot_identity,
        "telemetry_captured_at_utc": captured_at.isoformat(),
        "source_snapshot_sha256": source_snapshot_sha256.lower(),
        "pnl": pnl,
        "position": position,
        "grid": grid,
    }
    record: dict[str, Any] = {
        "schema_version": PNL_OBSERVATION_SCHEMA_VERSION,
        "record_type": "live_pnl_observation",
        "observation_id": observation_id,
        "bot_identity": bot_identity,
        "symbol": symbol,
        "strategy_id": strategy_id,
        "candidate_id": _clean_text(scanner_row.get("candidate_id")),
        "deploy_ts_utc": deployed_at.isoformat(),
        "telemetry_captured_at_utc": captured_at.isoformat(),
        "evaluated_at_utc": evaluated_at.isoformat(),
        "source_cycle_manifest": str(source_cycle_manifest),
        "source_snapshot_path": str(source_snapshot_path),
        "source_snapshot_sha256": source_snapshot_sha256.lower(),
        "snapshot_fingerprint": _sha256_payload(snapshot_basis),
        "pnl": pnl,
        "position": position,
        "grid": grid,
        "features": features,
        "decision": {
            "verdict": _clean_text(scanner_row.get("verdict")),
            "reasons": [str(item) for item in scanner_row.get("reasons", [])]
            if isinstance(scanner_row.get("reasons"), list)
            else [],
        },
        "runtime_effect": "observational_only",
    }
    record["record_integrity_sha256"] = _sha256_payload(
        _record_integrity_payload(record)
    )
    return record


def _validate_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != PNL_OBSERVATION_SCHEMA_VERSION:
        raise PnlHistoryError("unsupported PnL observation schema")
    if record.get("record_type") != "live_pnl_observation":
        raise PnlHistoryError("invalid PnL observation record_type")
    integrity_hash = str(record.get("record_integrity_sha256", "")).lower()
    if not _HEX_SHA256_RE.fullmatch(integrity_hash):
        raise PnlHistoryError("invalid record_integrity_sha256")
    if integrity_hash != _sha256_payload(_record_integrity_payload(record)):
        raise PnlHistoryError("PnL observation record_integrity_sha256 is invalid")
    symbol = str(record.get("symbol", "")).strip().upper()
    strategy_id = str(record.get("strategy_id", "")).strip()
    deploy_ts = _parse_utc(record.get("deploy_ts_utc"), field="deploy_ts_utc")
    captured_at = _parse_utc(
        record.get("telemetry_captured_at_utc"),
        field="telemetry_captured_at_utc",
    )
    evaluated_at = _parse_utc(record.get("evaluated_at_utc"), field="evaluated_at_utc")
    if evaluated_at < captured_at - timedelta(seconds=5):
        raise PnlHistoryError("evaluated_at_utc materially predates capture")
    expected_bot = pnl_bot_identity(
        symbol=symbol,
        strategy_id=strategy_id,
        deploy_ts=deploy_ts,
    )
    if record.get("bot_identity") != expected_bot:
        raise PnlHistoryError("PnL observation bot_identity does not match its fields")
    expected_observation = hashlib.sha256(
        f"{expected_bot}|{captured_at.isoformat()}".encode("utf-8")
    ).hexdigest()
    if record.get("observation_id") != expected_observation:
        raise PnlHistoryError("PnL observation_id does not match identity/capture time")
    source_hash = str(record.get("source_snapshot_sha256", "")).lower()
    if not _HEX_SHA256_RE.fullmatch(source_hash):
        raise PnlHistoryError("invalid source_snapshot_sha256")
    pnl = _mapping(record.get("pnl"))
    position = _mapping(record.get("position"))
    grid = _mapping(record.get("grid"))
    total = _finite_or_none(pnl.get("total_profit_usdt"), field="pnl.total_profit_usdt")
    if total is None:
        raise PnlHistoryError("pnl.total_profit_usdt is required")
    snapshot_basis: dict[str, Any] = {
        "bot_identity": expected_bot,
        "telemetry_captured_at_utc": captured_at.isoformat(),
        "source_snapshot_sha256": source_hash,
        "pnl": pnl,
        "position": position,
        "grid": grid,
    }
    if record.get("snapshot_fingerprint") != _sha256_payload(snapshot_basis):
        raise PnlHistoryError("PnL observation snapshot_fingerprint is invalid")
    validated = dict(record)
    validated["deploy_ts"] = deploy_ts
    validated["captured_at"] = captured_at
    validated["evaluated_at"] = evaluated_at
    return validated


def _record_paths(
    *,
    live_root: Path,
    symbol: str,
    bot_identity: str,
) -> list[Path]:
    paths: list[Path] = []
    if not live_root.is_dir():
        return paths
    for date_dir in live_root.iterdir():
        if not date_dir.is_dir() or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_dir.name):
            continue
        observations = (
            date_dir
            / symbol.upper()
            / "pnl_history"
            / bot_identity
            / "observations"
        )
        if observations.is_dir():
            paths.extend(path for path in observations.glob("*.json") if path.is_file())
    return sorted(paths, key=lambda path: str(path))


def _load_record(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PnlHistoryError(f"invalid PnL observation JSON at {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PnlHistoryError(f"PnL observation is not an object at {path}")
    try:
        validated = _validate_observation(cast(dict[str, Any], decoded))
        validated["_source_path"] = str(path)
        return validated
    except PnlHistoryError as exc:
        raise PnlHistoryError(f"{path}: {exc}") from exc


def validate_pnl_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one in-memory observation and return parsed timestamp helpers."""

    return _validate_observation(record)


def load_all_pnl_observations(*, live_root: Path) -> tuple[dict[str, Any], ...]:
    """Load every canonical PnL observation with global deduplication."""

    if not live_root.is_dir():
        return ()
    paths = sorted(
        path
        for path in live_root.rglob("*.json")
        if path.is_file()
        and path.parent.name == "observations"
        and path.parent.parent.parent.name == "pnl_history"
    )
    unique: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = _load_record(path)
        observation_id = str(record["observation_id"])
        prior = unique.get(observation_id)
        if prior is None:
            unique[observation_id] = record
        elif prior["snapshot_fingerprint"] != record["snapshot_fingerprint"]:
            raise PnlHistoryError(
                f"conflicting global PnL observation_id {observation_id}"
            )
    return tuple(
        sorted(
            unique.values(),
            key=lambda record: (
                cast(datetime, record["captured_at"]),
                str(record["bot_identity"]),
            ),
        )
    )


def load_pnl_observations(
    *,
    live_root: Path,
    symbol: str,
    strategy_id: str,
    deploy_ts: datetime,
) -> tuple[dict[str, Any], ...]:
    """Load, validate, deduplicate, and chronologically order one bot history."""

    deployed_at = _require_aware_utc(deploy_ts, field="deploy_ts")
    bot_identity = pnl_bot_identity(
        symbol=symbol,
        strategy_id=strategy_id,
        deploy_ts=deployed_at,
    )
    unique: dict[str, dict[str, Any]] = {}
    for path in _record_paths(
        live_root=live_root,
        symbol=symbol,
        bot_identity=bot_identity,
    ):
        record = _load_record(path)
        if record["bot_identity"] != bot_identity:
            raise PnlHistoryError(f"{path}: record belongs to a different bot identity")
        observation_id = str(record["observation_id"])
        prior = unique.get(observation_id)
        if prior is None:
            unique[observation_id] = record
        elif prior["snapshot_fingerprint"] != record["snapshot_fingerprint"]:
            raise PnlHistoryError(
                f"conflicting duplicate PnL observation_id {observation_id}"
            )
    return tuple(
        sorted(
            unique.values(),
            key=lambda record: (
                cast(datetime, record["captured_at"]),
                str(record["observation_id"]),
            ),
        )
    )


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_bot_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="ascii").splitlines()[0])
        except (OSError, ValueError, IndexError):
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise PnlHistoryError(
                f"PnL history writer already active with PID {existing_pid}"
            )
        lock_path.unlink(missing_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PnlHistoryError("PnL history writer lock was acquired concurrently") from exc
    os.write(
        descriptor,
        f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n".encode("ascii"),
    )
    os.fsync(descriptor)
    return descriptor


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        # The mandated Live hierarchy can produce a valid 259-character final
        # path on Windows. Never embed that final filename in the staging name:
        # doing so crosses MAX_PATH before the atomic commit is attempted.
        prefix=".tmp-",
        suffix=".json",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Link creation is atomic and fails when the immutable target already
            # exists. The temporary file remains in the same directory, so this
            # cannot cross filesystems; cleanup below removes the staging link.
            os.link(temp_name, path)
        except FileExistsError as exc:
            raise PnlHistoryError(
                f"refusing to overwrite immutable observation {path}"
            ) from exc
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def append_pnl_observation(
    observation: Mapping[str, Any],
    *,
    live_root: Path,
) -> AppendPnlObservationResult:
    """Atomically append an immutable observation or return an exact duplicate."""

    validated = _validate_observation(observation)
    symbol = str(validated["symbol"])
    strategy_id = str(validated["strategy_id"])
    deploy_ts = cast(datetime, validated["deploy_ts"])
    captured_at = cast(datetime, validated["captured_at"])
    bot_identity = str(validated["bot_identity"])
    observation_id = str(validated["observation_id"])
    lock_path = live_root / ".pnl_history_locks" / f"{bot_identity}.lock"
    descriptor = _acquire_bot_lock(lock_path)
    try:
        existing = load_pnl_observations(
            live_root=live_root,
            symbol=symbol,
            strategy_id=strategy_id,
            deploy_ts=deploy_ts,
        )
        for record in existing:
            if record["observation_id"] != observation_id:
                continue
            if record["snapshot_fingerprint"] != validated["snapshot_fingerprint"]:
                raise PnlHistoryError(
                    "conflicting observation for the same bot and capture timestamp"
                )
            source_path = Path(str(record.get("_source_path", "")))
            if not source_path.is_file():
                paths = _record_paths(
                    live_root=live_root,
                    symbol=symbol,
                    bot_identity=bot_identity,
                )
                source_path = next(
                    path
                    for path in paths
                    if str(_load_record(path)["observation_id"]) == observation_id
                )
            return AppendPnlObservationResult(
                status="duplicate",
                path=source_path,
                observation_id=observation_id,
                bot_identity=bot_identity,
                history_count=len(existing),
            )

        lima_date = captured_at.astimezone(_lima_timezone()).date().isoformat()
        observations_dir = (
            live_root
            / lima_date
            / symbol
            / "pnl_history"
            / bot_identity
            / "observations"
        )
        timestamp_part = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
        target = observations_dir / f"{timestamp_part}_{observation_id[:16]}.json"
        persisted = {
            key: value
            for key, value in validated.items()
            if key not in {"deploy_ts", "captured_at", "evaluated_at"}
        }
        _atomic_create_json(target, persisted)
        return AppendPnlObservationResult(
            status="appended",
            path=target,
            observation_id=observation_id,
            bot_identity=bot_identity,
            history_count=len(existing) + 1,
        )
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
