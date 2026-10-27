"""Immutable source-bound terminal UI outcomes; no action or training admission."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Context, Decimal, DecimalException, localcontext
import hashlib
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from neutralgrid.live.decision.evidence_cohort import read_bytes, read_json
from neutralgrid.live.decision.exact_bot_ledger import ledger_lock
from neutralgrid.live.decision.native_sar import NativeRecord
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, ContractModel, SarError, SourceRef, atomic_create_json, file_hash, utc,
)

LIMA = timezone(timedelta(hours=-5))
SCHEMA = "neutralgrid_terminal_reported_outcome_v1"
MONEY = r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?"
MAX_FINANCIAL_TOKEN_CHARS = 128  # Resource bound, not an economic/risk threshold.


class TerminalStorageError(SarError):
    """Publication may have succeeded before verification or unlock failed."""

    def __init__(self, message: str, published_path: Path | None = None) -> None:
        super().__init__(message)
        self.published_path = published_path


class TerminalObservation(ContractModel):
    schema_version: Literal["neutralgrid_terminal_reported_outcome_v1"] = SCHEMA
    bot: BotIdentity
    ended_at: datetime
    captured_at: datetime
    status: Literal["canceled", "expired"]
    reported_total_pnl_usdt: Decimal
    reported_invested_margin_usdt: Decimal = Field(gt=0)
    displayed_return_pct: Decimal
    source: SourceRef
    measurement_basis: Literal["terminal_reported_ui_pnl_not_verified_equity"] = "terminal_reported_ui_pnl_not_verified_equity"
    training_eligible: Literal[False] = False
    promotion_eligible: Literal[False] = False
    influence_enabled: Literal[False] = False

    _utc = field_validator("ended_at", "captured_at")(utc)

    @model_validator(mode="after")
    def times(self) -> TerminalObservation:
        if not self.bot.deployed_at <= self.ended_at <= self.captured_at:
            raise ValueError("terminal timing must follow deployment and precede capture")
        return self

    def stable_economics(self) -> dict[str, Any]:
        return self.model_dump(exclude={"source", "captured_at"})


def _live_path(path: Path, live_root: Path) -> Path:
    root, resolved = live_root.resolve(), path.resolve()
    if root.name != "Live" or root not in resolved.parents:
        raise SarError("terminal evidence must remain inside the supplied Live root")
    return resolved


def _field(lines: list[str], name: str) -> str:
    positions = [i for i, line in enumerate(lines) if line == name]
    if len(positions) != 1 or positions[0] + 1 >= len(lines):
        raise SarError(f"missing or ambiguous native field: {name}")
    return lines[positions[0] + 1]


def _local_time(value: str) -> datetime:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value) is None:
        raise SarError("native event time must use the explicit UTC-5 display format")
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=LIMA).astimezone(timezone.utc)


def parse_terminal_capture(record: NativeRecord, *, live_root: Path) -> TerminalObservation:
    path = _live_path(Path(record.source.path), live_root)
    if record.line_number is not None:
        raise SarError("terminal capture must be an original JSON object, not a JSONL row")
    raw = record.read()
    if raw.get("display_timezone") != "UTC-5":
        raise SarError("explicit UTC-5 display timezone required")
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SarError("original native drawer text required")
    for key in ("source", "classification", "scope"):
        if re.search(r"manual.?transcri|transcribed", str(raw.get(key, "")), re.IGNORECASE):
            raise SarError("manual transcription is not an original automated capture")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4 or lines[1:3] != ["Perp", "Futures Grid"]:
        raise SarError("native grid drawer header missing")
    if (lines.count("Futures Grid") != 1 or lines.count("Perp") != 1 or
            sum(re.fullmatch(r"[A-Z0-9]+USDT", line) is not None for line in lines) != 1):
        raise SarError("ambiguous or multiple native drawer headers")
    statuses: dict[str, Literal["canceled", "expired"]] = {"Canceled": "canceled", "Cancelled": "canceled", "Expired": "expired"}
    status = statuses.get(lines[3])
    if status is None:
        raise SarError("native drawer is not terminal")
    if "status" in raw:
        declared = raw["status"]
        if not isinstance(declared, str) or declared.lower().replace("cancelled", "canceled") != status:
            raise SarError("declared status conflicts with native terminal text")
    sid = _field(lines, "Strategy Number")
    if re.fullmatch(r"[0-9]+", sid) is None:
        raise SarError("native strategy ID must be an integer string")
    bot = BotIdentity(symbol=lines[0], strategy_id=sid, deployed_at=_local_time(_field(lines, "Time Created")))
    parts = path.relative_to(live_root.resolve()).parts
    if (len(parts) != 4 or parts[1] != bot.symbol or parts[2] != f"verdict_audit_{sid}" or
            parts[3] != "strategy_detail.json"):
        raise SarError("original capture path does not match its exact native bot")
    datetime.strptime(parts[0], "%Y-%m-%d")
    for key, expected in (("strategy_id", sid), ("symbol", bot.symbol)):
        if key in raw and (not isinstance(raw[key], str) or raw[key] != expected):
            raise SarError(f"declared {key} conflicts with native text")
    timestamps = []
    for key in ("captured_at", "captured_at_utc"):
        if key in raw:
            if not isinstance(raw[key], str):
                raise SarError("capture time must be an aware ISO timestamp")
            timestamps.append(utc(datetime.fromisoformat(raw[key].replace("Z", "+00:00"))))
    if not timestamps or len(set(timestamps)) != 1:
        raise SarError("missing or conflicting capture timestamps")
    profit = _field(lines, "Total Profit (USDT)")
    margin = _field(lines, "Invested Margin (USDT)")
    if len(profit) > 2 * MAX_FINANCIAL_TOKEN_CHARS + 8 or len(margin) > MAX_FINANCIAL_TOKEN_CHARS:
        raise SarError("native financial token exceeds bounded input size")
    match = re.fullmatch(rf"({MONEY})\s+\(({MONEY})%\)", profit)
    if match is None or re.fullmatch(MONEY, margin) is None:
        raise SarError("native PnL or capital amount is malformed")
    if max(len(match[1]), len(match[2])) > MAX_FINANCIAL_TOKEN_CHARS:
        raise SarError("native financial token exceeds bounded input size")
    return TerminalObservation(
        bot=bot, ended_at=_local_time(_field(lines, "Time Ended")), captured_at=timestamps[0],
        status=status, reported_total_pnl_usdt=Decimal(match[1].replace(",", "")),
        reported_invested_margin_usdt=Decimal(margin.replace(",", "")),
        displayed_return_pct=Decimal(match[2].replace(",", "")), source=record.source,
    )


def load_terminal_outcome(path: Path, *, live_root: Path) -> TerminalObservation:
    path = _live_path(path, live_root)
    raw, _ = read_json(path)
    observation = TerminalObservation.model_validate(raw)
    parts = path.relative_to(live_root.resolve()).parts
    if (len(parts) != 4 or parts[1] != observation.bot.symbol or parts[2] != "terminal_outcomes" or
            parts[3] != observation.bot.key + ".json"):
        raise SarError("terminal outcome is misrouted")
    ingestion_date = datetime.strptime(parts[0], "%Y-%m-%d").date()
    if ingestion_date < observation.captured_at.astimezone(LIMA).date():
        raise SarError("terminal ingestion folder predates original capture availability")
    expected = parse_terminal_capture(NativeRecord(observation.source), live_root=live_root)
    if observation != expected:
        raise SarError("terminal outcome differs from its original source")
    return observation


def capture_terminal_outcome(record: NativeRecord, *, live_root: Path,
                             ingested_at: datetime | None = None) -> tuple[Path, bool]:
    observation = parse_terminal_capture(record, live_root=live_root)
    root = live_root.resolve()
    cutoff = utc(ingested_at or datetime.now(timezone.utc))
    if observation.captured_at > cutoff:
        raise SarError("capture availability must not follow ingestion cutoff")
    date = cutoff.astimezone(LIMA).date().isoformat()
    target = _live_path(root / date / observation.bot.symbol / "terminal_outcomes" /
                        (observation.bot.key + ".json"), root)
    published: Path | None = None
    try:
        lock = _live_path(root / ".terminal-outcome-locks" / (observation.bot.key + ".lock"), root)
        with ledger_lock(lock):
            current = parse_terminal_capture(record, live_root=root)
            if observation != current:
                raise SarError("capture changed before publication")
            existing = list(root.glob(f"????-??-??/{observation.bot.symbol}/terminal_outcomes/{observation.bot.key}.json"))
            if len(existing) > 1:
                raise SarError("duplicate terminal outcome paths; quarantine deployment")
            if existing:
                previous = load_terminal_outcome(existing[0], live_root=root)
                if previous.stable_economics() != observation.stable_economics():
                    raise SarError("conflicting terminal economics for exact deployment")
                return existing[0], False
            atomic_create_json(target, observation.model_dump(mode="json"))
            published = target
            load_terminal_outcome(target, live_root=root)
        return target, True
    except (OSError, ValueError, RuntimeError) as error:
        if published is None and target.exists():
            published = target
        raise TerminalStorageError(str(error), published_path=published) from error


def score_terminal_outcome(observation: TerminalObservation, *, live_root: Path) -> dict[str, Any]:
    """Score a witnessed terminal outcome, without counterfactual attribution."""
    expected = parse_terminal_capture(NativeRecord(observation.source), live_root=live_root)
    if observation != expected:
        raise SarError("cannot score terminal economics not proved by original")
    pnl, capital = observation.reported_total_pnl_usdt, observation.reported_invested_margin_usdt
    try:
        with localcontext(Context(prec=50)):
            normalized = pnl / capital
            normalized_pct = 100 * normalized
            deficit = max(Decimal(0), pnl.copy_negate())
            deficit_pct = 100 * deficit / capital
    except DecimalException as error:
        raise SarError("terminal score arithmetic unavailable") from error
    endpoint_hit = normalized >= Decimal("0.03") and observation.ended_at <= observation.bot.deployed_at + timedelta(hours=7)
    return {
        "record_type": "terminal_reported_trajectory_score", "bot": observation.bot.model_dump(mode="json"),
        "ended_at": observation.ended_at.isoformat(), "captured_at": observation.captured_at.isoformat(),
        "reported_total_pnl_usdt": str(pnl), "reported_invested_margin_usdt": str(capital),
        "normalized_return": str(normalized), "normalized_return_pct": str(normalized_pct),
        "displayed_return_pct": str(observation.displayed_return_pct),
        "reported_capital_deficit_usdt": str(deficit),
        "reported_capital_deficit_pct": str(deficit_pct),
        "outcome_class": "positive" if pnl > 0 else "negative" if pnl < 0 else "flat",
        "measurement_basis": observation.measurement_basis,
        "first_passage_status": "path_unobserved",
        "terminal_return_meets_3pct_before_deployment_deadline": endpoint_hit,
        "capital_basis_at_deployment_verified": False,
        "verified_equity": False, "observed_action": None, "causal_reward": None,
        "training_eligible": False, "promotion_eligible": False, "influence_enabled": False,
        "source": observation.source.model_dump(mode="json"),
    }


def audit_terminal_outcomes(live_root: Path) -> dict[str, Any]:
    """Revalidate all terminal records; quarantine invalid or duplicated bots."""
    root = live_root.resolve()
    if root.name != "Live":
        raise SarError("terminal audit requires an explicit Live root")
    started = datetime.now(timezone.utc).isoformat()
    entries: list[dict[str, Any]] = []
    input_sources: list[dict[str, str]] = []
    for path in sorted(root.glob("*/*/terminal_outcomes/*.json")):
        entry: dict[str, Any] = {"path": str(path.resolve()), "keys": set(), "reasons": [], "score": None}
        if re.fullmatch(r"[0-9a-f]{64}", path.stem):
            entry["keys"].add(path.stem)
        try:
            checked_path = _live_path(path, root)
            initial_digest = hashlib.sha256(read_bytes(checked_path)).hexdigest()
            source = SourceRef(path=str(path.resolve()), sha256=initial_digest).model_dump(mode="json")
            input_sources.append(source)
            raw, digest = read_json(checked_path)
            if digest != initial_digest:
                raise SarError("terminal record changed during read")
            # A valid payload identity can quarantine a bad/misrouted filename.
            try:
                entry["keys"].add(BotIdentity.model_validate(raw.get("bot")).key)
            except ValueError:
                pass
            observation = load_terminal_outcome(path, live_root=root)
            entry["keys"].add(observation.bot.key)
            score = score_terminal_outcome(observation, live_root=root)
            if file_hash(path) != digest:
                raise SarError("terminal record changed during audit")
            score["terminal_outcome_source"] = source
            entry["score"] = score
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            entry["reasons"].append(str(error))
        entries.append(entry)
    counts = Counter(key for entry in entries for key in entry["keys"])
    quarantined = {key for entry in entries if entry["reasons"] for key in entry["keys"]}
    quarantined.update(key for key, count in counts.items() if count > 1)
    scores, rejected = [], []
    for entry in entries:
        if any(key in quarantined for key in entry["keys"]):
            entry["reasons"].append("exact_deployment_quarantined_by_invalid_or_duplicate_record")
        if entry["reasons"]:
            rejected.append({"path": entry["path"], "bot_identity_keys": sorted(entry["keys"]), "reasons": entry["reasons"]})
        elif entry["score"] is not None:
            scores.append(entry["score"])
    return {
        "record_type": "terminal_outcome_audit", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": file_hash(Path(__file__)), "input_sources": input_sources,
        "terminal_record_count": len(entries), "accepted_terminal_outcomes": len(scores),
        "rejected_terminal_records": len(rejected), "quarantined_bot_identity_keys": sorted(quarantined),
        "outcome_classes": dict(Counter(score["outcome_class"] for score in scores)),
        "first_passage_statuses": dict(Counter(score["first_passage_status"] for score in scores)),
        "scores": scores, "rejected": rejected,
        "training_eligible": False, "promotion_eligible": False, "influence_enabled": False,
    }
