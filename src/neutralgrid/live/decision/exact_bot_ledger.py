"""Immutable exact-deployment private-evidence ledger, separate from training.

Registration is not ownership. Source-bound facts are not completeness proofs.
No fact here can manufacture a SAR action, reward, approval or safety clearance.
"""
from __future__ import annotations

import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

from pydantic import Field, field_validator, model_validator

from neutralgrid.live.decision.evidence_cohort import read_json, verify_observation_sources
from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, ContractModel, SarError, SourceRef, _writer_lock, atomic_create_json,
    decimal, file_hash, fingerprint, utc,
)

SCHEMA = "neutralgrid_exact_bot_evidence_ledger_v1"
LIMA = timezone(-timedelta(hours=5))


@contextmanager
def ledger_lock(path: Path) -> Iterator[None]:
    # Publish a nonempty lock file atomically. Concurrent first-use initialization
    # must not write/flush byte zero while another Windows handle holds its lock.
    try:
        atomic_create_json(path, {"lock": SCHEMA})
    except FileExistsError:
        pass
    except OSError as exc:
        raise SarError("ledger writer busy or unavailable; retry idempotently") from exc
    try:
        with _writer_lock(path):
            yield
    except OSError as exc:
        raise SarError("ledger I/O unavailable; inspect and retry idempotently") from exc


class CapitalProtectionPolicy(ContractModel):
    reference: Literal["invested_capital"] = "invested_capital"
    maximum_capital_loss_pct: Literal[0] = 0
    minimum_liquidation_buffer_pct: None = None
    horizon_minutes_from_deployment: Literal[420] = 420
    influence_enabled: Literal[False] = False


class PrivateFact(ContractModel):
    schema_version: Literal["neutralgrid_exact_bot_evidence_ledger_v1"] = SCHEMA
    bot: BotIdentity
    account_scope: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["order_ownership", "order_update", "trade_fill", "income", "equity", "capital_flow", "intervention", "coverage"]
    native_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    available_at: datetime
    fields: dict[str, Any]
    source: SourceRef
    source_pointer: str = ""
    ownership_fact: SourceRef | None = None

    @field_validator("occurred_at", "available_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return utc(value)

    @model_validator(mode="after")
    def validate_facts(self) -> PrivateFact:
        if not self.bot.deployed_at <= self.occurred_at <= self.available_at:
            raise ValueError("fact timing must follow deployment and precede availability")
        required = {
            "order_ownership": {"order_id", "ownership_basis"},
            "order_update": {"order_id", "status", "executed_quantity"},
            "trade_fill": {"order_id", "trade_id", "side", "price", "quantity", "realized_pnl_usdt", "commission", "commission_asset"},
            "income": {"transaction_id", "income_type", "amount", "asset"},
            "equity": {"equity_usdt", "invested_capital_usdt", "unrealized_usdt", "peak_equity_usdt"},
            "capital_flow": {"amount", "asset", "direction"},
            "intervention": {"action", "receipt"},
            "coverage": {"start_at", "end_at", "scopes", "gaps", "producer_claims_complete"},
        }[self.kind]
        if set(self.fields) != required:
            raise ValueError(f"{self.kind} fields must exactly match {sorted(required)}")
        for key in ("order_id", "trade_id"):
            if key in self.fields and (not isinstance(self.fields[key], str) or not re.fullmatch(r"[0-9]+", self.fields[key])):
                raise ValueError(f"{key} must be a decimal integer string; no truncation")
        if self.kind == "order_ownership" and self.fields["ownership_basis"] not in {"native_bot_order_receipt", "reviewed_original_order_mapping"}:
            raise ValueError("symbol/time-only ownership is forbidden")
        for key in ("equity_usdt", "invested_capital_usdt", "unrealized_usdt", "price", "quantity", "executed_quantity", "realized_pnl_usdt", "commission", "amount"):
            if key in self.fields:
                number = decimal(self.fields[key])
                if key in {"invested_capital_usdt", "price", "quantity"} and number <= 0:
                    raise ValueError(f"{key} must be positive")
                if key == "commission" and number < 0:
                    raise ValueError("commission must use the positive-cost convention")
                if key == "executed_quantity" and number < 0:
                    raise ValueError("executed quantity must be nonnegative")
        peak = self.fields.get("peak_equity_usdt")
        if peak is not None and (decimal(peak) <= 0 or decimal(peak) < decimal(self.fields["equity_usdt"])):
            raise ValueError("claimed peak equity cannot be below observed equity")
        if self.kind == "trade_fill":
            if self.ownership_fact is None or self.fields["side"] not in {"BUY", "SELL"}:
                raise ValueError("trade fill requires linked ownership and BUY/SELL side")
            if self.native_id != self.fields["trade_id"] or not re.fullmatch(r"[A-Z0-9]+", str(self.fields["commission_asset"])):
                raise ValueError("trade identity or commission asset invalid")
        elif self.kind == "order_update":
            if self.ownership_fact is None or self.fields["status"] not in {"NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "EXPIRED", "REJECTED"}:
                raise ValueError("order update needs exact ownership and a supported exchange status")
        elif self.ownership_fact is not None:
            raise ValueError("ownership_fact applies only to fills/order updates")
        if self.kind == "order_ownership" and self.native_id != self.fields["order_id"]:
            raise ValueError("ownership native_id must equal order_id")
        if self.kind == "capital_flow":
            if self.fields["direction"] not in {"deposit", "withdrawal"} or decimal(self.fields["amount"]) <= 0:
                raise ValueError("capital flow requires positive amount and explicit direction")
            if not re.fullmatch(r"[A-Z0-9]+", str(self.fields["asset"])):
                raise ValueError("capital flow asset invalid")
        if self.kind == "income":
            if (not isinstance(self.fields["transaction_id"], str) or
                    not re.fullmatch(r"[0-9]+", self.fields["transaction_id"]) or self.native_id != self.fields["transaction_id"]):
                raise ValueError("income needs an exact transaction ID")
            if self.fields["income_type"] not in {"FUNDING_FEE", "REALIZED_PNL", "COMMISSION", "OTHER_TRADING"}:
                raise ValueError("unsupported income accounting type")
            if not re.fullmatch(r"[A-Z0-9]+", str(self.fields["asset"])):
                raise ValueError("income asset invalid")
        if self.kind == "intervention":
            if self.fields["action"] not in {"ADJUST", "END", "ADD_CAPITAL", "WITHDRAW_CAPITAL"}:
                raise ValueError("intervention must be an observed intervention, never invented CONTINUE")
            SourceRef.model_validate(self.fields["receipt"])
        if self.kind == "coverage":
            start = utc(datetime.fromisoformat(str(self.fields["start_at"]).replace("Z", "+00:00")))
            end = utc(datetime.fromisoformat(str(self.fields["end_at"]).replace("Z", "+00:00")))
            if not self.bot.deployed_at <= start <= end <= self.occurred_at:
                raise ValueError("invalid coverage interval")
            scopes, gaps = self.fields["scopes"], self.fields["gaps"]
            if not isinstance(scopes, list) or not scopes or any(s not in {"orders", "trades", "income", "capital_flows", "interventions", "equity"} for s in scopes):
                raise ValueError("explicit coverage scopes required")
            if len(scopes) != len(set(scopes)) or not isinstance(gaps, list):
                raise ValueError("coverage scopes/gaps malformed")
            if type(self.fields["producer_claims_complete"]) is not bool:
                raise ValueError("coverage completeness must be a boolean")
            for gap in gaps:
                if not isinstance(gap, dict) or set(gap) != {"start_at", "end_at"}:
                    raise ValueError("coverage gaps need exact time bounds")
                lower = utc(datetime.fromisoformat(gap["start_at"].replace("Z", "+00:00")))
                upper = utc(datetime.fromisoformat(gap["end_at"].replace("Z", "+00:00")))
                if not start <= lower <= upper <= end:
                    raise ValueError("gap outside coverage")
            if gaps and self.fields["producer_claims_complete"]:
                raise ValueError("gapped coverage cannot claim completeness")
        return self

    @property
    def identity(self) -> str:
        # Exchange IDs are scoped to account and symbol, not presumed globally unique.
        return fingerprint({"account": self.account_scope, "symbol": self.bot.symbol,
                            "kind": self.kind, "native_id": self.native_id,
                            "income_type": self.fields.get("income_type")})


def _source_record(ref: SourceRef, pointer: str) -> Mapping[str, Any]:
    payload, digest = read_json(Path(ref.path))
    if digest != ref.sha256:
        raise ValueError("source missing or changed")
    selected: Any = payload
    if pointer:
        if not pointer.startswith("/"):
            raise ValueError("source pointer must be empty or a JSON pointer")
        for part in pointer[1:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(selected, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
                selected = selected[int(key)]
            elif isinstance(selected, dict):
                selected = selected[key]
            else:
                raise ValueError("source pointer cannot resolve")
    if not isinstance(selected, dict):
        raise ValueError("source pointer must resolve to an original record")
    return selected


def verify_fact(fact: PrivateFact, *, live_root: Path) -> None:
    raw = _source_record(fact.source, fact.source_pointer)
    # Exact field binding is deliberately stronger than accepting an unrelated
    # hashed file. It still does not authenticate an external producer's claims.
    expected = {key: fact.model_dump(mode="json")[key] for key in
                ("account_scope", "kind", "native_id", "fields")}
    source_bot = BotIdentity.model_validate(raw.get("bot"))
    source_time = utc(datetime.fromisoformat(str(raw.get("occurred_at")).replace("Z", "+00:00")))
    source_available = utc(datetime.fromisoformat(str(raw.get("available_at")).replace("Z", "+00:00")))
    if (source_bot != fact.bot or source_time != fact.occurred_at or source_available != fact.available_at or
            any(raw.get(key) != value for key, value in expected.items())):
        raise ValueError("original record does not prove the supplied identity and fields")
    if fact.kind == "intervention":
        receipt = SourceRef.model_validate(fact.fields["receipt"])
        if file_hash(Path(receipt.path)) != receipt.sha256:
            raise ValueError("intervention receipt changed")
    if fact.kind in {"trade_fill", "order_update"}:
        assert fact.ownership_fact is not None
        ownership_path = Path(fact.ownership_fact.path).resolve()
        if live_root.resolve() not in ownership_path.parents or ownership_path.parent.name != "facts":
            raise ValueError("trade ownership must reference a committed ledger fact")
        payload, digest = read_json(ownership_path)
        if digest != fact.ownership_fact.sha256:
            raise ValueError("ownership source changed")
        owner = PrivateFact.model_validate(payload["fact"])
        if (ownership_path.parent.parent.name != owner.bot.key or
                ownership_path.parents[2].name != "exact_bot_ledger" or
                ownership_path.parents[3].name != owner.bot.symbol):
            raise ValueError("ownership fact is outside its canonical deployment path")
        if owner.kind != "order_ownership" or owner.bot != fact.bot or owner.account_scope != fact.account_scope or owner.fields["order_id"] != fact.fields["order_id"]:
            raise ValueError("trade ownership/deployment mismatch")
        if owner.available_at > fact.available_at:
            raise ValueError("ownership was unavailable at the fact's available_at")
        if payload.get("record_sha256") != fingerprint(owner.model_dump(mode="json")) or ownership_path.stem != owner.identity:
            raise ValueError("ownership ledger integrity mismatch")
        verify_fact(owner, live_root=live_root)


def append_fact(fact: PrivateFact, *, live_root: Path, now: datetime) -> Path:
    at = utc(now)
    fact = PrivateFact.model_validate(fact.model_dump(mode="json"))
    if fact.available_at > at:
        raise ValueError("future fact is not yet available")
    with ledger_lock(live_root / ".exact_bot_ledger_locks" / (fact.identity + ".lock")):
        verify_fact(fact, live_root=live_root)
        body = fact.model_dump(mode="json")
        checksum = fingerprint(body)
        # Search all deployments to prevent one order being assigned to two bots.
        existing = sorted(live_root.glob(f"????-??-??/{fact.bot.symbol}/exact_bot_ledger/*/facts/{fact.identity}.json"))
        for path in existing:
            prior, _ = read_json(path)
            if prior.get("record_sha256") != checksum or fingerprint(prior.get("fact", {})) != checksum:
                raise ValueError("conflicting native identity or cross-bot ownership")
        if existing:
            return existing[0]
        date = at.astimezone(LIMA).date().isoformat()
        path = live_root / date / fact.bot.symbol / "exact_bot_ledger" / fact.bot.key / "facts" / f"{fact.identity}.json"
        atomic_create_json(path, {"schema_version": SCHEMA, "fact": body, "record_sha256": checksum,
                                 "evidence_class": "source_bound_producer_record",
                                 "training_eligible": False, "event_completeness": "unproven"})
        return path


def verify_registration_source(bot: BotIdentity, source: SourceRef, live_root: Path) -> None:
    raw, digest = read_json(Path(source.path))
    if digest != source.sha256 or raw.get("record_type") != "live_evidence_index":
        raise ValueError("registration needs a verified evidence index")
    if raw.get("sar_bot_key") != bot.key or raw.get("symbol") != bot.symbol or str(raw.get("strategy_id")) != bot.strategy_id:
        raise ValueError("registration identity mismatch")
    identity_proven = False
    for ref in raw.get("sources", []):
        parsed = SourceRef.model_validate(ref)
        if file_hash(Path(parsed.path)) != parsed.sha256:
            raise ValueError("registration original source changed")
        if Path(parsed.path).parent.name == "observations" and not identity_proven:
            observation, _ = read_json(Path(parsed.path))
            record = validate_pnl_observation(observation)
            actual = BotIdentity(symbol=record["symbol"], strategy_id=record["strategy_id"], deployed_at=record["deploy_ts"])
            if actual != bot:
                raise ValueError("registration original deployment mismatch")
            verify_observation_sources(record, live_root=live_root)
            identity_proven = True
    if not identity_proven:
        raise ValueError("registration needs original deployment evidence")


def verify_registration(raw: Mapping[str, Any], path: Path, live_root: Path) -> BotIdentity:
    body = {k: v for k, v in raw.items() if k != "record_sha256"}
    if raw.get("record_sha256") != fingerprint(body) or raw.get("schema_version") != SCHEMA:
        raise ValueError("registration integrity mismatch")
    bot = BotIdentity.model_validate(raw["bot"])
    if path.parent.name != bot.key or path.parents[2].name != bot.symbol:
        raise ValueError("misrouted registration")
    CapitalProtectionPolicy.model_validate(raw["capital_policy"])
    if raw.get("training_eligible") is not False or raw.get("influence_enabled") is not False:
        raise ValueError("registration cannot enable training or influence")
    verify_registration_source(bot, SourceRef.model_validate(raw["identity_source"]), live_root)
    return bot


def register_bot(bot: BotIdentity, *, source: SourceRef, live_root: Path, now: datetime) -> Path:
    at = utc(now)
    verify_registration_source(bot, source, live_root)
    with ledger_lock(live_root / ".exact_bot_ledger_locks" / (bot.key + ".lock")):
        existing = sorted(live_root.glob(f"????-??-??/{bot.symbol}/exact_bot_ledger/{bot.key}/registration.json"))
        for path in existing:
            prior, _ = read_json(path)
            if verify_registration(prior, path, live_root) != bot:
                raise ValueError("registration identity conflict")
        if existing:
            return existing[0]
        path = live_root / at.astimezone(LIMA).date().isoformat() / bot.symbol / "exact_bot_ledger" / bot.key / "registration.json"
        body = {"schema_version": SCHEMA, "bot": bot.model_dump(mode="json"),
                                 "registered_at": at.isoformat(), "identity_source": source.model_dump(mode="json"),
                                 "account_scope": None, "ownership_status": "pending_original_order_mapping",
                                 "capital_policy": CapitalProtectionPolicy().model_dump(mode="json"),
                                 "protected_capital_usdt": None, "event_completeness": "unproven",
                                 "training_eligible": False, "influence_enabled": False}
        atomic_create_json(path, body | {"record_sha256": fingerprint(body)})
        return path


def audit_ledger(live_root: Path) -> dict[str, Any]:
    facts: dict[str, PrivateFact] = {}
    rejected = []
    bad_bots: set[str] = set()
    registrations: set[str] = set()
    for path in sorted(live_root.glob("????-??-??/*/exact_bot_ledger/*/registration.json")):
        try:
            raw, _ = read_json(path)
            bot = verify_registration(raw, path, live_root)
            registrations.add(bot.key)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            bad_bots.add(path.parent.name)
            rejected.append({"path": str(path), "reason": str(exc)})
    for path in sorted(live_root.glob("????-??-??/*/exact_bot_ledger/*/facts/*.json")):
        try:
            raw, _ = read_json(path)
            fact = PrivateFact.model_validate(raw["fact"])
            if path.stem != fact.identity or path.parent.parent.name != fact.bot.key or path.parents[3].name != fact.bot.symbol:
                raise ValueError("misrouted exact-bot fact")
            if raw.get("record_sha256") != fingerprint(fact.model_dump(mode="json")):
                raise ValueError("ledger record hash mismatch")
            verify_fact(fact, live_root=live_root)
            prior = facts.get(fact.identity)
            if prior is not None and prior != fact:
                bad_bots.add(prior.bot.key)
                raise ValueError("conflicting native identity")
            facts[fact.identity] = fact
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, TypeError) as exc:
            bad_bots.add(path.parent.parent.name)
            rejected.append({"path": str(path), "reason": str(exc)})
    accepted = [f for f in facts.values() if f.bot.key not in bad_bots]
    equity_checks = [{"bot_key": f.bot.key, "native_id": f.native_id, "occurred_at": f.occurred_at.isoformat(),
                      "preserves_reported_invested_capital_at_observation": decimal(f.fields["equity_usdt"]) >= decimal(f.fields["invested_capital_usdt"]),
                      "guarantees_future_capital_preservation": False}
                     for f in accepted if f.kind == "equity"]
    return {"schema_version": SCHEMA, "registrations": len(registrations - bad_bots),
            "source_bound_facts": len(accepted), "fact_classes": dict(Counter(f.kind for f in accepted)),
            "fact_bots": len({f.bot.key for f in accepted}), "rejected": rejected,
            "facts": [f.model_dump(mode="json") for f in accepted],
            "capital_preservation_checks": equity_checks,
            "training_eligible": False, "influence_enabled": False,
            "blockers": ["producer_authenticity_and_interval_completeness_require_independent_validation",
                         "no_automatic_SAR_conversion", "minimum_liquidation_buffer_unspecified"]}
