"""Source-bound native telemetry conversion; no action routing or training clearance.

REST pages and producer coverage claims cannot establish a complete intervention
interval. Their accounting can be retained in SAR, with unknown completeness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from neutralgrid.live.decision.action_shadow_io import state_from_observations
from neutralgrid.live.decision.capital_preservation import assess_current_capital
from neutralgrid.live.decision.evidence_cohort import exchange_ticks, read_json, verify_observation_sources
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact, verify_fact
from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import (
    Action, Approval, BotIdentity, CostEvent, Decision, Execution, LedgerEvent,
    Outcome, SarError, SourceRef, State, configuration_hash, decimal,
    feature_definition_hash, file_hash, fingerprint, make_event, utc, verify_sources,
)


@dataclass(frozen=True)
class NativeRecord:
    source: SourceRef
    line_number: int | None = None

    def read(self) -> dict[str, Any]:
        path = Path(self.source.path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise SarError("native record exceeds bounded input size")
        if file_hash(path) != self.source.sha256:
            raise SarError("native source changed")
        if self.line_number is None:
            record, digest = read_json(path)
            if digest != self.source.sha256:
                raise SarError("native source changed during read")
            return record
        if isinstance(self.line_number, bool) or self.line_number < 1:
            raise SarError("JSONL line number must be positive")

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise SarError("duplicate JSON field")
                result[key] = value
            return result

        def invalid(value: str) -> None:
            raise SarError("non-finite JSON number")

        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, 1):
                if index == self.line_number:
                    record = json.loads(line, object_pairs_hook=unique, parse_constant=invalid)
                    if not isinstance(record, dict) or file_hash(path) != self.source.sha256:
                        raise SarError("native source changed or row is not an object")
                    return record
        raise SarError("JSONL line does not exist")


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SarError("explicit source timestamp required")
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _checked(event: LedgerEvent) -> LedgerEvent:
    verify_sources(event)
    return event


def _identity(raw: Mapping[str, Any], bot: BotIdentity) -> None:
    if raw.get("symbol") != bot.symbol or str(raw.get("strategy_id")) != bot.strategy_id:
        raise SarError("native evidence belongs to a different bot")


def _source_closure(refs: Sequence[SourceRef]) -> tuple[SourceRef, ...]:
    """Flatten JSON provenance so ordinary SAR replay checks every original.

    SAR hashes referenced files, not references embedded inside those files.
    Keep the complete source graph explicit, with bounded traversal and cycles.
    """
    pending = [(ref, 0) for ref in refs]
    found: dict[str, SourceRef] = {}
    total_bytes = 0
    while pending:
        ref, depth = pending.pop()
        if ref.path in found:
            if found[ref.path] != ref:
                raise SarError("conflicting provenance source versions")
            continue
        if depth > 32 or len(found) >= 4096:
            raise SarError("provenance graph exceeds bounded traversal")
        path = Path(ref.path)
        total_bytes += path.stat().st_size
        if total_bytes > 64 * 1024 * 1024:
            raise SarError("provenance graph exceeds bounded byte budget")
        raw, digest = read_json(path)
        if digest != ref.sha256:
            raise SarError("provenance original changed")
        found[ref.path] = ref
        values: list[Any] = [raw]
        while values:
            value = values.pop()
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"}:
                    pending.append((SourceRef.model_validate(value), depth + 1))
                else:
                    values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)
    return tuple(found.values())


def native_state(record: NativeRecord, observations: Sequence[Mapping[str, Any]], *,
                 live_root: Path, equity_fact: PrivateFact | None = None) -> State:
    """Construct only from revalidated persisted observations and the exact row.

    Row-reported equity/peak fields are not treated as verified accounting. An
    exact-time, source-bound ledger equity record is required to populate them.
    """
    row = record.read()
    evaluation = row.get("evaluation")
    evaluated = evaluation if isinstance(evaluation, dict) else {}
    asof = _time(evaluated.get("evaluated_at_utc", row.get("ts")))
    originals: list[dict[str, Any]] = []
    native_refs: list[SourceRef] = []
    for observation in observations:
        original_path = Path(str(observation["_source_path"]))
        original, _ = read_json(original_path)
        validated = validate_pnl_observation(original)
        if validated["record_integrity_sha256"] != observation["record_integrity_sha256"]:
            raise SarError("observation differs from persisted original")
        if validated["evaluated_at"] > asof:
            continue
        native_refs.extend(SourceRef.model_validate(ref) for ref in verify_observation_sources(validated, live_root=live_root))
        originals.append(original | {"_source_path": str(original_path)})
    state = state_from_observations(originals, row)
    values = state.model_dump(mode="python")
    values.update(equity_usdt=None, peak_equity_usdt=None,
                  sources=tuple(dict.fromkeys((*state.sources, record.source, *native_refs))))
    if equity_fact is not None:
        equity_fact = PrivateFact.model_validate(equity_fact.model_dump(mode="json"))
        verify_fact(equity_fact, live_root=live_root)
        if (equity_fact.kind != "equity" or equity_fact.bot != state.bot or
                equity_fact.occurred_at != state.captured_at or equity_fact.available_at > state.available_at):
            raise SarError("equity is not exact-bot, exact-time available evidence")
        fields = equity_fact.fields
        if (decimal(fields["invested_capital_usdt"]) != state.capital_usdt or
                decimal(fields["unrealized_usdt"]) != state.unrealized_usdt):
            raise SarError("equity accounting differs from native telemetry")
        values.update(equity_usdt=fields["equity_usdt"], peak_equity_usdt=fields["peak_equity_usdt"],
                      sources=(*values["sources"], *_source_closure((equity_fact.source,))))
    return State.model_validate(values)


def native_decision(record: NativeRecord, observations: Sequence[Mapping[str, Any]], *,
                    live_root: Path, equity_fact: PrivateFact | None = None,
                    intent: NativeRecord | None = None) -> LedgerEvent:
    row = record.read()
    state = native_state(record, observations, live_root=live_root, equity_fact=equity_fact)
    verdict = Verdict(str(row.get("verdict")))
    action = Action(verdict=verdict,
                    lower=row.get("suggested_grid_lower") if verdict == Verdict.ADJUST else None,
                    upper=row.get("suggested_grid_upper") if verdict == Verdict.ADJUST else None)
    if intent is not None:
        proposed = intent.read()
        _identity(proposed, state.bot)
        if proposed.get("action") != action.verdict.value or _time(proposed.get("decision_ts")) != state.available_at:
            raise SarError("native action intent differs from decision")
        if action.verdict == Verdict.ADJUST and (
                decimal(proposed.get("suggested_grid_lower")) != action.lower or
                decimal(proposed.get("suggested_grid_upper")) != action.upper):
            raise SarError("native intent bounds differ from decision")
        state = State.model_validate(state.model_dump(mode="python") | {"sources": (*state.sources, intent.source)})
    capital = assess_current_capital(state, equity_fact, live_root=live_root)
    decision = Decision(decided_at=state.available_at, pre_state=state, incumbent=action,
                        shadow=action, limits=None, origin="live", config_hash=configuration_hash(None),
                        feature_hash=feature_definition_hash(),
                        diagnostics=("native_verdict_is_proposal_not_observed_action", "risk_limits_unspecified",
                                     "primary_objective:preserve_invested_capital", *capital.diagnostics))
    return _checked(make_event(bot=state.bot, decision_id=fingerprint({"bot": state.bot.key, "row": row}),
                               kind="decision", occurred_at=decision.decided_at,
                               payload=decision.model_dump(mode="json")))


def native_approval(record: NativeRecord, decision: LedgerEvent, *, intent: NativeRecord) -> LedgerEvent:
    """Bind a native approval through its original action intent and decision time."""
    if record.line_number is not None:
        raise SarError("native approval must be an original JSON object")
    verify_sources(decision)
    if decision.kind != "decision":
        raise SarError("decision event required")
    parsed = Decision.model_validate(decision.payload)
    if intent.source not in parsed.pre_state.sources:
        raise SarError("decision must retain original intent provenance before approval conversion")
    raw, proposed = record.read(), intent.read()
    _identity(raw, decision.bot)
    _identity(proposed, decision.bot)
    key = raw.get("idempotency_key")
    if (key != proposed.get("idempotency_key") or
            _time(proposed.get("decision_ts")) != parsed.decided_at or
            proposed.get("action") != "ADJUST" or parsed.incumbent.verdict != Verdict.ADJUST):
        raise SarError("approval is not linked to the native decision intent")
    action = Action(verdict=Verdict.ADJUST, lower=raw.get("suggested_grid_lower"),
                    upper=raw.get("suggested_grid_upper"))
    if (action != parsed.incumbent or decimal(proposed.get("suggested_grid_lower")) != action.lower or
            decimal(proposed.get("suggested_grid_upper")) != action.upper):
        raise SarError("approval bounds differ from the linked proposal")
    approval = Approval(action=action, approved_at=_time(raw.get("approved_at_utc")),
                        expires_at=_time(raw.get("expires_at_utc")), source=record.source)
    if approval.approved_at < parsed.decided_at:
        raise SarError("approval predates decision")
    return _checked(make_event(bot=decision.bot, decision_id=decision.decision_id, kind="approval",
                               occurred_at=approval.approved_at, payload=approval.model_dump(mode="json")))


def native_execution(record: NativeRecord, decision: LedgerEvent, approval: LedgerEvent, *,
                     pre_record: NativeRecord, post_record: NativeRecord,
                     pre_observations: Sequence[Mapping[str, Any]],
                     post_observations: Sequence[Mapping[str, Any]], live_root: Path,
                     exchange_info: NativeRecord) -> LedgerEvent:
    """Convert prospective timestamped native receipts; legacy timings stay unknown."""
    if record.line_number is not None:
        raise SarError("native receipt must be an original JSON object")
    for event in (decision, approval):
        verify_sources(event)
    if decision.kind != "decision" or approval.kind != "approval" or decision.transition_id != approval.transition_id:
        raise SarError("approval/decision transition mismatch")
    raw = record.read()
    _identity(raw, decision.bot)
    parsed = Decision.model_validate(decision.payload)
    approved = Approval.model_validate(approval.payload)
    approval_raw = NativeRecord(approved.source).read()
    if (raw.get("idempotency_key") != approval_raw.get("idempotency_key") or
            _time(raw.get("decision_ts")) != parsed.decided_at):
        raise SarError("receipt is not linked to decision and approval")
    submitted, verified = _time(raw.get("submitted_at_utc")), _time(raw.get("verified_at_utc"))
    if not parsed.decided_at <= approved.approved_at <= submitted < approved.expires_at or submitted > verified:
        raise SarError("receipt lifecycle is inconsistent")
    before = native_state(pre_record, pre_observations, live_root=live_root)
    after = native_state(post_record, post_observations, live_root=live_root)
    if before.bot != decision.bot or after.bot != decision.bot:
        raise SarError("receipt states belong to another deployment")
    if not before.available_at <= submitted <= after.captured_at <= after.available_at <= verified:
        raise SarError("receipt timestamps do not enclose native states")
    capture = exchange_info.read()
    received = _time(capture.get("received_at_utc"))
    requested = _time(capture.get("requested_at_utc"))
    if not decision.bot.deployed_at <= requested <= received <= submitted:
        raise SarError("exchange tick evidence is not available in the bot lifetime before submission")
    ticks = exchange_ticks(capture.get("exchange_info", {}))
    quantum = decimal(ticks.get(decision.bot.symbol))
    precision = raw.get("price_decimals")
    if isinstance(precision, bool) or not isinstance(precision, int) or not 0 <= precision <= 16 or quantum != Decimal(10) ** -precision:
        raise SarError("native decimal precision does not match exchange PRICE_FILTER tick")
    # Persist tick provenance in the states so SAR revalidation detects changes.
    before = State.model_validate(before.model_dump(mode="python") | {"sources": (*before.sources, exchange_info.source)})
    if raw.get("preserve_current_position") is not True:
        raise SarError("receipt does not preserve current position")
    executed = Execution(requested=approved.action,
                         applied=Action(verdict=Verdict.ADJUST, lower=raw.get("executed_grid_lower"),
                                        upper=raw.get("executed_grid_upper")), approval_event_id=approval.event_id,
                         pre_state=before, post_state=after, submitted_at=submitted, verified_at=verified,
                         receipt=record.source, quantum=quantum, preserve_position=True,
                         additional_investment_usdt=decimal(raw.get("additional_investment")))
    return _checked(make_event(bot=decision.bot, decision_id=decision.decision_id, kind="execution",
                               occurred_at=verified, payload=executed.model_dump(mode="json")))


def ledger_costs(facts: Sequence[PrivateFact], *, bot: BotIdentity, account_scope: str,
                 start: datetime, end: datetime, available_at: datetime, live_root: Path) -> tuple[CostEvent, ...]:
    """Keep one canonical representation; ambiguous fills/income overlap is rejected."""
    start, end, available_at = utc(start), utc(end), utc(available_at)
    if not bot.deployed_at <= start <= end <= available_at:
        raise SarError("invalid accounting interval")
    seen: dict[str, PrivateFact] = {}
    for supplied in facts:
        fact = PrivateFact.model_validate(supplied.model_dump(mode="json"))
        verify_fact(fact, live_root=live_root)
        if fact.bot != bot or fact.account_scope != account_scope or fact.available_at > available_at:
            raise SarError("accounting identity or availability mismatch")
        if fact.identity in seen and seen[fact.identity] != fact:
            raise SarError("conflicting accounting native identity")
        seen[fact.identity] = fact
    selected = [fact for fact in seen.values() if start < fact.occurred_at <= end]
    if any(f.kind == "trade_fill" for f in selected) and any(
            f.kind == "income" and f.fields["income_type"] in {"REALIZED_PNL", "COMMISSION"} for f in selected):
        raise SarError("fill and income accounting overlap needs explicit exchange linkage")
    costs: list[CostEvent] = []
    for fact in selected:
        fields = fact.fields
        entries: list[tuple[str, str, Decimal]] = []
        if fact.kind == "trade_fill":
            entries = [("realized", "USDT", decimal(fields["realized_pnl_usdt"])),
                       ("commission", str(fields["commission_asset"]), decimal(fields["commission"]))]
        elif fact.kind == "income":
            kind = {"REALIZED_PNL": "realized", "COMMISSION": "commission", "FUNDING_FEE": "funding",
                    "OTHER_TRADING": "other_trading"}[fields["income_type"]]
            amount = decimal(fields["amount"])
            # Native income uses signed account changes; SAR commission is a cost.
            entries = [(kind, str(fields["asset"]), -amount if kind == "commission" else amount)]
        elif fact.kind == "capital_flow":
            entries = [("external_flow", str(fields["asset"]), decimal(fields["amount"]) *
                        (1 if fields["direction"] == "deposit" else -1))]
        for kind, asset, amount in entries:
            costs.append(CostEvent.model_validate(dict(exchange_id=fact.identity, kind=kind,
                occurred_at=fact.occurred_at, source_asset=asset, source_amount=amount,
                amount_usdt=amount if asset == "USDT" else None, source=fact.source)))
    return tuple(sorted(costs, key=lambda cost: (cost.occurred_at, cost.exchange_id, cost.kind)))


def ledger_outcome(record: NativeRecord, observations: Sequence[Mapping[str, Any]], decision: LedgerEvent, *,
                   facts: Sequence[PrivateFact], account_scope: str, live_root: Path,
                   equity_fact: PrivateFact | None = None, ingested_at: datetime) -> LedgerEvent:
    """Persist partial observed accounting without promoting it to a reward label."""
    verify_sources(decision)
    if decision.kind != "decision":
        raise SarError("decision event required")
    parsed = Decision.model_validate(decision.payload)
    state = native_state(record, observations, live_root=live_root, equity_fact=equity_fact)
    at = utc(ingested_at)
    if state.bot != decision.bot or state.available_at > at:
        raise SarError("outcome deployment/availability mismatch")
    if equity_fact is not None and equity_fact.account_scope != account_scope:
        raise SarError("outcome equity account differs from costs")
    costs = ledger_costs(facts, bot=state.bot, account_scope=account_scope,
                         start=parsed.pre_state.captured_at, end=state.captured_at, available_at=at, live_root=live_root)
    accounting_sources: list[SourceRef] = []
    for fact in facts:
        accounting_sources.append(fact.source)
        if fact.ownership_fact is not None:
            accounting_sources.append(fact.ownership_fact)
    state = State.model_validate(state.model_dump(mode="python") | {
        "sources": tuple(dict.fromkeys((*state.sources, *_source_closure(accounting_sources))))})
    outcome = Outcome(state=state, coverage_start=parsed.pre_state.captured_at, coverage_end=state.captured_at,
                      event_completeness="unknown", intervention_coverage_complete=False,
                      realized_includes_commission=None, realized_includes_funding=None, costs=costs)
    return _checked(make_event(bot=decision.bot, decision_id=decision.decision_id, kind="outcome",
                               occurred_at=at, payload=outcome.model_dump(mode="json")))
