"""Immutable, source-checked action evidence. Never routes or promotes actions."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from neutralgrid.core.constants import (
    ACTION_APPROVAL_SCHEMA,
    ACTION_EXECUTION_SCHEMA,
    DECISION_CONTRACT_VERSION,
    STATE_ACTION_REWARD_CONTRACT_VERSION,
    STATE_ACTION_REWARD_SCHEMA,
)
from neutralgrid.live.decision.pnl_forecast import FEATURE_COLUMNS
from neutralgrid.live.decision.action_path_features import PricePoint
from neutralgrid.live.decision.recommender import Verdict

logger = logging.getLogger(__name__)
UTC = timezone.utc
HORIZON = timedelta(minutes=60)
TOLERANCE = timedelta(minutes=5)
RECENTER_CAP = Decimal("0.25")
RECONCILIATION_TOLERANCE = Decimal("0.000001")


class SarError(RuntimeError):
    """Evidence is invalid, conflicted, incomplete or unsupported."""


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not financial evidence")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("non-finite decimal")
    return result


def fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: Any, info: Any) -> Any:
        annotation = cls.model_fields[info.field_name].annotation
        if isinstance(value, bool) and ("Decimal" in str(annotation) or annotation in (float, int)):
            raise ValueError("boolean is not financial evidence")
        return value


class BotIdentity(ContractModel):
    symbol: str = Field(pattern=r"^[A-Z0-9]+USDT$")
    strategy_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    deployed_at: datetime

    _utc = field_validator("deployed_at")(utc)

    @property
    def key(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class SourceRef(ContractModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("source path must be absolute")
        return value


class State(ContractModel):
    bot: BotIdentity
    captured_at: datetime
    available_at: datetime
    feature_window_start: datetime
    sources: tuple[SourceRef, ...] = Field(min_length=1)
    grid_lower: Decimal | None = None
    grid_upper: Decimal | None = None
    num_grids: int | None = Field(default=None, strict=True, gt=0)
    capital_usdt: Decimal | None = Field(default=None, gt=0)
    equity_usdt: Decimal | None = None
    unrealized_usdt: Decimal | None = None
    position_quantity: Decimal | None = None
    entry_price: Decimal | None = Field(default=None, gt=0)
    orders: tuple[dict[str, Any], ...] | None = None
    market_price: Decimal | None = Field(default=None, gt=0)
    market_captured_at: datetime | None = None
    symbol_available: StrictBool | None = None
    liquidation_buffer_pct: Decimal | None = None
    peak_equity_usdt: Decimal | None = Field(default=None, gt=0)
    hmm_lineage: str | None = None
    features: dict[str, float | None] = Field(default_factory=dict)
    price_path: tuple[PricePoint, ...] = ()

    _utc = field_validator("captured_at", "available_at", "feature_window_start")(utc)

    @field_validator("market_captured_at")
    @classmethod
    def market_utc(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @field_validator("features", mode="before")
    @classmethod
    def feature_allowlist(cls, value: Any) -> Any:
        if not isinstance(value, dict) or set(value) - set(FEATURE_COLUMNS):
            raise ValueError("unknown or outcome-derived feature")
        if any(isinstance(item, bool) for item in value.values()):
            raise ValueError("features must be numeric or null")
        return value

    @model_validator(mode="after")
    def temporal_order(self) -> State:
        if self.feature_window_start > self.captured_at or self.available_at < self.captured_at:
            raise ValueError("invalid state availability/window")
        if self.captured_at < self.bot.deployed_at:
            raise ValueError("state predates deployment")
        if self.market_captured_at is not None and self.market_captured_at > self.available_at:
            raise ValueError("market evidence from the future")
        if self.grid_lower is not None and self.grid_upper is not None:
            if not Decimal(0) < self.grid_lower < self.grid_upper:
                raise ValueError("invalid grid bounds")
        return self


class Action(ContractModel):
    verdict: Verdict
    lower: Decimal | None = None
    upper: Decimal | None = None

    @model_validator(mode="after")
    def bounds(self) -> Action:
        if self.verdict == Verdict.ADJUST:
            if self.lower is None or self.upper is None or not Decimal(0) < self.lower < self.upper:
                raise ValueError("ADJUST requires positive ordered bounds")
        elif self.lower is not None or self.upper is not None:
            raise ValueError("bounds apply only to ADJUST")
        return self


class RiskLimits(ContractModel):
    min_liquidation_buffer_pct: Decimal = Field(gt=0)
    max_drawdown_pct: Decimal = Field(gt=0, le=100)
    max_private_age_seconds: float = Field(default=900.0, gt=0)
    max_market_age_seconds: float = Field(default=15.0, gt=0)


def configuration_hash(limits: RiskLimits | None) -> str:
    return fingerprint({"limits": limits.model_dump(mode="json") if limits is not None else None,
                        "horizon_minutes": 60, "endpoint_tolerance_minutes": 5,
                        "recenter_cap": str(RECENTER_CAP), "accounting_tolerance": str(RECONCILIATION_TOLERANCE)})


def feature_definition_hash() -> str:
    return fingerprint({"columns": list(FEATURE_COLUMNS),
                        "contract_source": file_hash(Path(__file__)),
                        "path_source": file_hash(Path(__file__).with_name("action_path_features.py"))})


class Decision(ContractModel):
    decided_at: datetime
    pre_state: State
    incumbent: Action
    shadow: Action
    scanner_contract_version: str = DECISION_CONTRACT_VERSION
    limits: RiskLimits | None = None
    origin: Literal["live", "simulation", "uncontrolled"] = "uncontrolled"
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: tuple[str, ...] = ()

    _utc = field_validator("decided_at")(utc)

    @model_validator(mode="after")
    def asof(self) -> Decision:
        if self.pre_state.available_at > self.decided_at:
            raise ValueError("decision uses unavailable state")
        if self.scanner_contract_version != DECISION_CONTRACT_VERSION:
            raise ValueError("unsupported scanner contract")
        if self.config_hash != configuration_hash(self.limits):
            raise ValueError("configuration hash does not describe the declared profile")
        if self.incumbent.verdict == Verdict.END and self.shadow != self.incumbent:
            raise ValueError("incumbent END cannot be downgraded")
        return self


class Approval(ContractModel):
    action: Action
    approved_at: datetime
    expires_at: datetime
    source: SourceRef

    _utc = field_validator("approved_at", "expires_at")(utc)

    @model_validator(mode="after")
    def expires(self) -> Approval:
        if self.expires_at <= self.approved_at:
            raise ValueError("approval expiry precedes approval")
        return self


class Execution(ContractModel):
    requested: Action
    applied: Action
    approval_event_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_state: State
    post_state: State
    submitted_at: datetime
    verified_at: datetime
    receipt: SourceRef
    quantum: Decimal = Field(gt=0)
    preserve_position: StrictBool
    additional_investment_usdt: Decimal

    _utc = field_validator("submitted_at", "verified_at")(utc)


class Maintenance(ContractModel):
    state: State
    coverage_start: datetime
    coverage_end: datetime
    intervention_coverage_complete: StrictBool
    source: SourceRef

    _utc = field_validator("coverage_start", "coverage_end")(utc)


class CostEvent(ContractModel):
    exchange_id: str = Field(min_length=1)
    kind: Literal["realized", "commission", "funding", "other_trading", "external_flow"]
    occurred_at: datetime
    source_asset: str = Field(min_length=1)
    source_amount: Decimal
    amount_usdt: Decimal | None
    conversion_source: SourceRef | None = None
    source: SourceRef

    _utc = field_validator("occurred_at")(utc)


class Slippage(ContractModel):
    fill_id: str = Field(min_length=1)
    fill_at: datetime
    side: Literal["BUY", "SELL"]
    quantity: Decimal = Field(gt=0)
    fill_price: Decimal = Field(gt=0)
    benchmark_price: Decimal = Field(gt=0)
    benchmark_at: datetime
    estimated: StrictBool
    source: SourceRef
    benchmark_source: SourceRef

    _utc = field_validator("fill_at", "benchmark_at")(utc)

    @model_validator(mode="after")
    def preceding_benchmark(self) -> Slippage:
        if not timedelta(0) <= self.fill_at - self.benchmark_at <= timedelta(seconds=15):
            raise ValueError("slippage benchmark must precede fill by at most 15 seconds")
        return self

    @property
    def cost_usdt(self) -> Decimal:
        return (self.fill_price - self.benchmark_price) * self.quantity * (
            Decimal(1) if self.side == "BUY" else Decimal(-1))


class Outcome(ContractModel):
    state: State
    coverage_start: datetime
    coverage_end: datetime
    event_completeness: Literal["event_complete", "history_complete", "snapshot_only", "unknown"]
    intervention_coverage_complete: StrictBool
    interventions: tuple[datetime, ...] = ()
    realized_includes_commission: StrictBool | None
    realized_includes_funding: StrictBool | None
    costs: tuple[CostEvent, ...] = ()
    slippage: tuple[Slippage, ...] = ()

    _utc = field_validator("coverage_start", "coverage_end")(utc)

    @field_validator("interventions")
    @classmethod
    def intervention_times(cls, values: tuple[datetime, ...]) -> tuple[datetime, ...]:
        return tuple(utc(value) for value in values)


class Finalized(ContractModel):
    terminal_at: datetime
    terminal_status: Literal["ended", "cancelled", "expired"]
    source: SourceRef
    lifetime: Outcome

    _utc = field_validator("terminal_at")(utc)


PAYLOAD_TYPES: dict[str, type[ContractModel]] = {
    "decision": Decision, "approval": Approval, "execution": Execution,
    "maintenance": Maintenance, "outcome": Outcome, "finalized": Finalized,
}


class LedgerEvent(ContractModel):
    schema_version: str = STATE_ACTION_REWARD_SCHEMA
    contract_version: str = STATE_ACTION_REWARD_CONTRACT_VERSION
    bot: BotIdentity
    decision_id: str = Field(min_length=1, max_length=128)
    transition_id: str
    kind: Literal["decision", "approval", "execution", "maintenance", "outcome", "finalized"]
    occurred_at: datetime
    event_id: str
    payload: dict[str, Any]
    record_sha256: str

    _utc = field_validator("occurred_at")(utc)

    @model_validator(mode="after")
    def integrity(self) -> LedgerEvent:
        if (self.schema_version != STATE_ACTION_REWARD_SCHEMA or
                self.contract_version != STATE_ACTION_REWARD_CONTRACT_VERSION):
            raise ValueError("unsupported state/action/reward contract")
        transition = fingerprint({"bot": self.bot.key, "decision_id": self.decision_id})
        event_id = fingerprint({"transition": transition, "kind": self.kind,
                                "occurred_at": self.occurred_at.isoformat()})
        if self.transition_id != transition or self.event_id != event_id:
            raise ValueError("event identity mismatch")
        body = self.model_dump(mode="json", exclude={"record_sha256"})
        if self.record_sha256 != fingerprint(body):
            raise ValueError("event integrity mismatch")
        parsed = PAYLOAD_TYPES[self.kind].model_validate(self.payload)
        for state in _states(parsed):
            if state.bot != self.bot or state.available_at > self.occurred_at:
                raise ValueError("state identity/availability differs from event")
        lifecycle_time = (parsed.decided_at if isinstance(parsed, Decision) else
                          parsed.approved_at if isinstance(parsed, Approval) else
                          parsed.verified_at if isinstance(parsed, Execution) else
                          parsed.terminal_at if isinstance(parsed, Finalized) else None)
        if lifecycle_time is not None and lifecycle_time > self.occurred_at:
            raise ValueError("event precedes its lifecycle evidence")
        return self


def _states(payload: ContractModel) -> tuple[State, ...]:
    if isinstance(payload, Decision):
        return (payload.pre_state,)
    if isinstance(payload, Execution):
        return payload.pre_state, payload.post_state
    if isinstance(payload, (Maintenance, Outcome)):
        return (payload.state,)
    if isinstance(payload, Finalized):
        return (payload.lifetime.state,)
    return ()


def make_event(*, bot: BotIdentity, decision_id: str, kind: str,
               occurred_at: datetime, payload: Mapping[str, Any]) -> LedgerEvent:
    if kind not in PAYLOAD_TYPES:
        raise SarError("unknown lifecycle kind")
    parsed = PAYLOAD_TYPES[kind].model_validate(payload)
    at = utc(occurred_at)
    transition = fingerprint({"bot": bot.key, "decision_id": decision_id})
    body = {"schema_version": STATE_ACTION_REWARD_SCHEMA,
            "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
            "bot": bot.model_dump(mode="json"), "decision_id": decision_id,
            "transition_id": transition, "kind": kind, "occurred_at": at.isoformat(),
            "event_id": fingerprint({"transition": transition, "kind": kind,
                                     "occurred_at": at.isoformat()}),
            "payload": parsed.model_dump(mode="json")}
    # Canonicalize datetime formatting before computing its checksum.
    body["occurred_at"] = at.isoformat().replace("+00:00", "Z")
    return LedgerEvent.model_validate({**body, "record_sha256": fingerprint(body)})


def contract_schema_catalog() -> dict[str, Any]:
    """Independent JSON Schema documents; Python also checks cross-event evidence."""
    return {"schema_version": STATE_ACTION_REWARD_SCHEMA,
            "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
            "event_schema": LedgerEvent.model_json_schema(),
            "payload_schemas": {kind: model.model_json_schema() for kind, model in PAYLOAD_TYPES.items()}}


def verify_sources(event: LedgerEvent) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = Path(value["path"])
                if not path.is_file() or file_hash(path) != value["sha256"]:
                    raise SarError(f"source missing or changed: {path}")
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(event.payload)
    parsed = PAYLOAD_TYPES[event.kind].model_validate(event.payload)
    for state in _states(parsed):
        for point in state.price_path:
            if file_hash(Path(point.source_path)) != point.source_sha256:
                raise SarError("price path source changed")
    if isinstance(parsed, Approval) and parsed.action.verdict == Verdict.ADJUST:
        raw = _native_json(parsed.source)
        _action_key(raw)
        if (raw.get("schema_version") != ACTION_APPROVAL_SCHEMA or raw.get("action") != "ADJUST" or
                raw.get("symbol") != event.bot.symbol or str(raw.get("strategy_id")) != event.bot.strategy_id or
                raw.get("preserve_current_position") is not True or
                decimal(raw.get("suggested_grid_lower")) != parsed.action.lower or
                decimal(raw.get("suggested_grid_upper")) != parsed.action.upper or
                utc(datetime.fromisoformat(str(raw.get("approved_at_utc")).replace("Z", "+00:00"))) != parsed.approved_at or
                utc(datetime.fromisoformat(str(raw.get("expires_at_utc")).replace("Z", "+00:00"))) != parsed.expires_at):
            raise SarError("approval differs from its native source")
    if isinstance(parsed, Execution) and parsed.requested.verdict == Verdict.ADJUST:
        raw = _native_json(parsed.receipt)
        _action_key(raw)
        if (raw.get("schema_version") != ACTION_EXECUTION_SCHEMA or raw.get("status") != "executed" or
                raw.get("action") != "ADJUST" or raw.get("symbol") != event.bot.symbol or
                str(raw.get("strategy_id")) != event.bot.strategy_id or
                raw.get("preserve_current_position") is not True or
                decimal(raw.get("additional_investment")) != parsed.additional_investment_usdt or
                raw.get("num_grids") != parsed.post_state.num_grids or
                decimal(raw.get("requested_grid_lower")) != parsed.requested.lower or
                decimal(raw.get("requested_grid_upper")) != parsed.requested.upper or
                decimal(raw.get("executed_grid_lower")) != parsed.applied.lower or
                decimal(raw.get("executed_grid_upper")) != parsed.applied.upper):
            raise SarError("execution differs from its native verified receipt")
        precision = raw.get("price_decimals")
        if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 16 or parsed.quantum != Decimal(10) ** -precision:
            raise SarError("execution precision differs from receipt")
        for name, state in (("pre_state", parsed.pre_state), ("post_state", parsed.post_state)):
            native = raw.get(name)
            if (not isinstance(native, dict) or native.get("symbol") != event.bot.symbol or
                    str(native.get("strategy_id")) != event.bot.strategy_id or
                    native.get("status") != "Working" or native.get("num_grids") != state.num_grids or
                    decimal(native.get("lower")) != state.grid_lower or
                    decimal(native.get("upper")) != state.grid_upper):
                raise SarError("execution state differs from its native verified receipt")


def _action_key(raw: Mapping[str, Any]) -> str:
    value = raw.get("idempotency_key")
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise SarError("native action identity missing or invalid")
    return value


def _native_json(ref: SourceRef) -> dict[str, Any]:
    path = Path(ref.path)
    if path.stat().st_size > 1024 * 1024:
        raise SarError("oversized native action evidence")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SarError("native evidence must be a JSON object")
    return raw


@contextmanager
def _writer_lock(path: Path) -> Iterator[None]:
    # A fully written inode is published before any contender can open it.
    # Never initialize byte zero through a shared handle: Windows may already
    # have locked that byte between another writer's size check and flush.
    try:
        atomic_create_json(path, {"lock": STATE_ACTION_REWARD_SCHEMA})
    except FileExistsError:
        pass
    except OSError as exc:
        raise SarError("ledger writer busy or unavailable; no event published") from exc
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise SarError("ledger writer busy or unavailable; no event published") from exc
    with handle:
        try:
            if os.fstat(handle.fileno()).st_size == 0:
                raise SarError("ledger writer busy or empty lock; no event published")
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SarError("ledger writer busy; no event published") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(name, path)  # Atomic publication, never overwrites a committed event.
    finally:
        Path(name).unlink(missing_ok=True)


def append_event(event: LedgerEvent, *, live_root: Path,
                 ingested_at: datetime | None = None) -> Path:
    event = LedgerEvent.model_validate(event.model_dump(mode="json"))
    verify_sources(event)
    with _writer_lock(live_root / ".sar-locks" / (event.transition_id + ".lock")):
        pattern = f"????-??-??/{event.bot.symbol}/sar/{event.transition_id[:32]}/{event.event_id}.json"
        existing = list(live_root.glob(pattern))
        if len(existing) > 1:
            raise SarError("duplicate ledger paths; quarantine transition")
        if existing:
            prior = LedgerEvent.model_validate_json(existing[0].read_bytes())
            if prior.record_sha256 != event.record_sha256:
                raise SarError("conflicting immutable event")
            return existing[0]
        date = utc(ingested_at or datetime.now(UTC)).astimezone(
            timezone(timedelta(hours=-5))).date().isoformat()
        target = (live_root / date / event.bot.symbol / "sar" /
                  event.transition_id[:32] / (event.event_id + ".json"))
        atomic_create_json(target, event.model_dump(mode="json"))
        return target


def load_events(live_root: Path) -> tuple[list[LedgerEvent], list[dict[str, str]]]:
    events: list[LedgerEvent] = []
    rejected: list[dict[str, str]] = []
    bad_transitions: set[str] = set()
    for path in sorted(live_root.glob("????-??-??/*/sar/*/*.json")):
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise SarError("oversized ledger event")
            event = LedgerEvent.model_validate_json(path.read_bytes())
            datetime.strptime(path.parents[3].name, "%Y-%m-%d")
            if (path.parent.name != event.transition_id[:32] or path.name != event.event_id + ".json" or
                    path.parents[2].name != event.bot.symbol):
                raise SarError("ledger filename identity mismatch")
            verify_sources(event)
            events.append(event)
        except (OSError, ValueError, SarError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            bad_transitions.add(path.parent.name)
    return [e for e in events if e.transition_id[:32] not in bad_transitions], rejected


def shield_reasons(state: State, limits: RiskLimits | None, *, now: datetime) -> list[str]:
    reasons: list[str] = []
    at = utc(now)
    if limits is None:
        return ["missing_risk_limits"]
    if not timedelta(0) <= at - state.captured_at <= timedelta(seconds=limits.max_private_age_seconds):
        reasons.append("stale_private_data")
    if state.available_at > at:
        reasons.append("unavailable_state")
    if state.market_captured_at is None or not timedelta(0) <= at - state.market_captured_at <= timedelta(seconds=limits.max_market_age_seconds):
        reasons.append("stale_market_data")
    if state.symbol_available is not True:
        reasons.append("symbol_unavailable_or_unknown")
    if state.liquidation_buffer_pct is None or state.liquidation_buffer_pct < limits.min_liquidation_buffer_pct:
        reasons.append("liquidation_buffer_missing_or_breached")
    if state.peak_equity_usdt is None or state.equity_usdt is None:
        reasons.append("drawdown_unknown")
    elif (state.peak_equity_usdt - state.equity_usdt) / state.peak_equity_usdt * 100 > limits.max_drawdown_pct:
        reasons.append("drawdown_breached")
    # Primary operator objective: preserve invested capital. Peak drawdown is a
    # separate constraint and cannot permit consuming the invested principal.
    # This is a pre-state shield, never a filter removing losing outcome labels.
    if state.capital_usdt is None or state.equity_usdt is None:
        reasons.append("invested_capital_floor_unknown")
    elif state.equity_usdt < state.capital_usdt:
        reasons.append("invested_capital_floor_breached")
    if not state.hmm_lineage:
        reasons.append("hmm_lineage_unknown")
    if state.capital_usdt is None or state.position_quantity is None:
        reasons.append("capital_or_position_unknown")
    if state.grid_lower is None or state.grid_upper is None or state.num_grids is None:
        reasons.append("grid_unknown")
    return reasons


def recenter(state: State, *, quantum: Decimal) -> Action:
    if state.grid_lower is None or state.grid_upper is None or state.market_price is None:
        raise SarError("recenter requires observed geometry and price")
    quantum = decimal(quantum)
    if quantum <= 0:
        raise SarError("positive observed price quantum required")
    width = state.grid_upper - state.grid_lower
    center = (state.grid_lower + state.grid_upper) / 2
    shift = min(max(state.market_price - center, -RECENTER_CAP * width), RECENTER_CAP * width)
    lower = ((state.grid_lower + shift) / quantum).to_integral_value(rounding=ROUND_FLOOR) * quantum
    upper = ((state.grid_upper + shift) / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum
    if lower <= 0:
        raise SarError("rounded recenter bound is not positive")
    if abs((lower + upper) / 2 - center) > RECENTER_CAP * width:
        raise SarError("rounding exceeds center cap")
    if shift == 0:
        raise SarError("recenter is a no-op")
    return Action(verdict=Verdict.ADJUST, lower=state.grid_lower + shift,
                  upper=state.grid_upper + shift)


def geometry_reasons(state: State, requested: Action, applied: Action, quantum: Decimal) -> list[str]:
    if state.grid_lower is None or state.grid_upper is None:
        return ["grid_unknown"]
    if requested.verdict != Verdict.ADJUST or applied.verdict != Verdict.ADJUST:
        return ["unsupported_executed_action"]
    if requested.lower is None or requested.upper is None or applied.lower is None or applied.upper is None:
        return ["bounds_missing"]
    width = state.grid_upper - state.grid_lower
    center = (state.grid_lower + state.grid_upper) / 2
    if requested.upper - requested.lower != width:
        return ["requested_width_changed"]
    if abs((applied.lower + applied.upper) / 2 - center) > width * RECENTER_CAP:
        return ["center_cap_exceeded"]
    if not (0 <= requested.lower - applied.lower < quantum and
            0 <= applied.upper - requested.upper < quantum):
        return ["invalid_outward_rounding"]
    return []


@dataclass(frozen=True)
class Projection:
    transition_id: str
    decision: Decision
    behavior: Action | None
    evidence_class: str
    endpoint: datetime | None
    net_usdt: Decimal | None
    normalized_return: Decimal | None
    training_eligible: bool
    reasons: tuple[str, ...]


def project_transition(events: Sequence[LedgerEvent]) -> Projection:
    if not events:
        raise SarError("empty transition")
    unique: dict[str, LedgerEvent] = {}
    conflict = False
    for event in events:
        event = LedgerEvent.model_validate(event.model_dump(mode="json"))
        verify_sources(event)
        if event.transition_id != events[0].transition_id:
            raise SarError("mixed transition identities")
        if event.event_id in unique and unique[event.event_id].record_sha256 != event.record_sha256:
            conflict = True
        unique[event.event_id] = event
    decisions = [Decision.model_validate(e.payload) for e in unique.values() if e.kind == "decision"]
    if len(decisions) != 1:
        raise SarError("transition must contain exactly one decision")
    decision = decisions[0]
    start = decision.pre_state.captured_at
    reasons: list[str] = ["conflicting_event"] if conflict else []
    behavior: Action | None = None
    outcomes = sorted((Outcome.model_validate(e.payload) for e in unique.values() if e.kind == "outcome"),
                      key=lambda o: o.state.captured_at)
    endpoint = next((o for o in outcomes if start + HORIZON <= o.state.captured_at <= start + HORIZON + TOLERANCE), None)
    executions = [Execution.model_validate(e.payload) for e in unique.values() if e.kind == "execution"]
    if len(executions) == 1:
        execution = executions[0]
        approval_event = unique.get(execution.approval_event_id)
        approval = (Approval.model_validate(approval_event.payload)
                    if approval_event is not None and approval_event.kind == "approval" else None)
        if approval is None or approval.action != execution.requested:
            reasons.append("exact_approval_missing")
        elif not (decision.decided_at <= approval.approved_at <= execution.submitted_at < approval.expires_at and
                  approval.expires_at - approval.approved_at <= timedelta(seconds=300) and
                  execution.submitted_at - approval.approved_at <= timedelta(seconds=300)):
            reasons.append("approval_expired_or_out_of_order")
        if approval is not None and execution.requested.verdict == Verdict.ADJUST:
            if _action_key(_native_json(approval.source)) != _action_key(_native_json(execution.receipt)):
                reasons.append("approval_receipt_identity_mismatch")
        if not (decision.decided_at <= execution.submitted_at <= decision.decided_at + timedelta(seconds=180) and
                decision.pre_state.captured_at <= execution.pre_state.captured_at and
                execution.pre_state.available_at <= execution.submitted_at <= execution.post_state.captured_at <=
                execution.post_state.available_at <= execution.verified_at):
            reasons.append("execution_timing_invalid")
        reasons.extend(geometry_reasons(execution.pre_state, execution.requested, execution.applied, execution.quantum))
        if (not execution.preserve_position or execution.additional_investment_usdt != 0 or
                execution.pre_state.num_grids != execution.post_state.num_grids or
                execution.pre_state.capital_usdt != execution.post_state.capital_usdt or
                execution.applied.lower != execution.post_state.grid_lower or
                execution.applied.upper != execution.post_state.grid_upper or
                execution.pre_state.grid_lower != decision.pre_state.grid_lower or
                execution.pre_state.grid_upper != decision.pre_state.grid_upper or
                execution.pre_state.num_grids != decision.pre_state.num_grids or
                execution.pre_state.capital_usdt != decision.pre_state.capital_usdt or
                execution.pre_state.position_quantity is None or execution.post_state.position_quantity is None):
            reasons.append("execution_state_not_preserved")
        if endpoint is not None and (
                endpoint.state.grid_lower != execution.applied.lower or
                endpoint.state.grid_upper != execution.applied.upper or
                endpoint.state.num_grids != execution.post_state.num_grids or
                endpoint.state.capital_usdt != execution.post_state.capital_usdt):
            reasons.append("post_action_geometry_or_capital_changed")
        behavior = execution.applied if not reasons else None
    elif len(executions) > 1:
        reasons.append("multiple_executions")
    else:
        maintenance = [Maintenance.model_validate(e.payload) for e in unique.values() if e.kind == "maintenance"]
        if endpoint is not None and len(maintenance) == 1:
            m = maintenance[0]
            if (m.intervention_coverage_complete and m.coverage_start <= start and
                    m.coverage_end >= endpoint.state.captured_at and
                    m.state.grid_lower == decision.pre_state.grid_lower == endpoint.state.grid_lower and
                    m.state.grid_upper == decision.pre_state.grid_upper == endpoint.state.grid_upper and
                    m.state.num_grids == decision.pre_state.num_grids == endpoint.state.num_grids and
                    m.state.capital_usdt == decision.pre_state.capital_usdt == endpoint.state.capital_usdt):
                behavior = Action(verdict=Verdict.CONTINUE)
        if behavior is None:
            reasons.append("verified_behavior_missing")
    evidence_class = "verified_behavior" if behavior is not None else "unexecuted_proposal"
    if decision.origin != "live":
        evidence_class = decision.origin
        reasons.append("non_live_evidence")
    net: Decimal | None = None
    normalized: Decimal | None = None
    if endpoint is None:
        reasons.append("endpoint_missing")
    elif behavior is not None and not reasons:
        net, accounting_reasons = _reward(decision.pre_state, endpoint, executions)
        reasons.extend(accounting_reasons)
        if net is not None and decision.pre_state.capital_usdt is not None:
            normalized = net / decision.pre_state.capital_usdt
        elif net is not None:
            reasons.append("capital_unknown")
    reasons.extend(shield_reasons(decision.pre_state, decision.limits, now=decision.decided_at))
    return Projection(events[0].transition_id, decision, behavior, evidence_class,
                      endpoint.state.captured_at if endpoint is not None else None,
                      net, normalized, not reasons and net is not None and normalized is not None, tuple(reasons))


def _reward(pre: State, outcome: Outcome, executions: Sequence[Execution]) -> tuple[Decimal | None, list[str]]:
    end = outcome.state.captured_at
    if (outcome.event_completeness != "event_complete" or
            not outcome.intervention_coverage_complete or
            outcome.coverage_start > pre.captured_at or outcome.coverage_end < end):
        return None, ["incomplete_accounting_or_interventions"]
    if any(pre.captured_at < t <= end and
           not any(e.submitted_at == t for e in executions) for t in outcome.interventions):
        return None, ["intervening_action"]
    if outcome.realized_includes_commission is None or outcome.realized_includes_funding is None:
        return None, ["pnl_inclusion_semantics_unknown"]
    if any(v is None for v in (pre.equity_usdt, pre.unrealized_usdt,
                              outcome.state.equity_usdt, outcome.state.unrealized_usdt)):
        return None, ["equity_or_unrealized_unknown"]
    fills: dict[str, Slippage] = {}
    for fill in outcome.slippage:
        if not pre.captured_at < fill.fill_at <= end:
            return None, ["slippage_outside_window"]
        if fill.fill_id in fills and fills[fill.fill_id] != fill:
            return None, ["conflicting_slippage_fill"]
        fills[fill.fill_id] = fill
    totals = {kind: Decimal(0) for kind in ("realized", "commission", "funding", "other_trading", "external_flow")}
    seen: dict[tuple[str, str], CostEvent] = {}
    for cost in outcome.costs:
        key = cost.kind, cost.exchange_id
        if key in seen:
            if seen[key] != cost:
                return None, ["conflicting_exchange_event"]
            continue
        seen[key] = cost
        if not pre.captured_at < cost.occurred_at <= end:
            return None, ["cost_outside_window"]
        if cost.amount_usdt is None or (cost.source_asset != "USDT" and cost.conversion_source is None):
            return None, ["cost_conversion_unknown"]
        if cost.source_asset == "USDT" and cost.source_amount != cost.amount_usdt:
            return None, ["usdt_amount_mismatch"]
        totals[cost.kind] += cost.amount_usdt
    gross = totals["realized"]
    if outcome.realized_includes_commission:
        gross += totals["commission"]
    if outcome.realized_includes_funding:
        gross -= totals["funding"]
    assert pre.equity_usdt is not None and pre.unrealized_usdt is not None
    assert outcome.state.equity_usdt is not None and outcome.state.unrealized_usdt is not None
    net = (gross - totals["commission"] + totals["funding"] + totals["other_trading"] +
           outcome.state.unrealized_usdt - pre.unrealized_usdt)
    equity_net = outcome.state.equity_usdt - pre.equity_usdt - totals["external_flow"]
    if abs(net - equity_net) > RECONCILIATION_TOLERANCE:
        return None, ["equity_reconciliation_failed"]
    return net, []
