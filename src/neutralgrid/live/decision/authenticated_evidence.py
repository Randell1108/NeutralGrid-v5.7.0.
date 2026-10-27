"""Bounded read-only exchange acquisition; attribution and completeness stay separate.

Authenticated REST proves acquisition through a credential, not native grid
ownership. Raw account-symbol rows remain unassigned unless an original reviewed
order mapping supplies that missing relationship. No balances become bot equity.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from neutralgrid.live.decision.evidence_cohort import LIMA, MAX_BYTES, read_json, verify_observation_sources
from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact, append_fact, verify_fact
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, SourceRef, atomic_create_json, decimal, file_hash, fingerprint, utc,
)

SCHEMA = "neutralgrid_authenticated_evidence_v1"
HOST = "https://fapi.binance.com"
WEEK_MS = 7 * 24 * 60 * 60 * 1000
PRODUCER_SHA256 = file_hash(Path(__file__))


def integer_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("identifier must be an integer or decimal integer string")
    text = str(value)
    if not re.fullmatch(r"0|[1-9][0-9]*", text):
        raise ValueError("identifier must be a canonical nonnegative integer")
    return text


def instant_ms(value: Any) -> datetime:
    return datetime.fromtimestamp(int(integer_id(value)) / 1000, tz=timezone.utc)


def credential_scope(api_key: str) -> str:
    if not api_key:
        raise ValueError("authenticated API credential is unavailable")
    # Deliberately not represented as a Binance account UID. Different API keys
    # cannot be merged without a separately verified account identity mapping.
    return "credential_" + hashlib.sha256(api_key.encode()).hexdigest()


def publish(body: Mapping[str, Any], *, bot: BotIdentity, live_root: Path,
            at: datetime, category: str) -> Path:
    if category not in {"responses", "manifests", "producer_records", "pnl_paths", "forecast_contracts"}:
        raise ValueError("unsupported evidence category")
    payload = dict(body)
    if len(json.dumps(payload, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("evidence exceeds bounded storage size")
    digest = fingerprint(payload)
    existing = sorted(live_root.glob(f"????-??-??/{bot.symbol}/authenticated_evidence/{bot.key}/{category}/{digest}.json"))
    if existing:
        if len(existing) != 1 or read_json(existing[0])[0] != payload:
            raise ValueError("conflicting or duplicated immutable evidence")
        return existing[0].resolve()
    path = (live_root.resolve() / utc(at).astimezone(LIMA).date().isoformat() /
            bot.symbol / "authenticated_evidence" / bot.key / category /
            (digest + ".json"))
    try:
        atomic_create_json(path, payload)
    except FileExistsError:
        prior, _ = read_json(path)
        if prior != payload:
            raise ValueError("immutable evidence collision")
    return path


def source_ref(path: Path) -> SourceRef:
    return SourceRef(path=str(path.resolve()), sha256=file_hash(path))


async def collect_history(*, fetch: Callable[[int, int], Awaitable[Any]],
                          endpoint: str, bot: BotIdentity, account_scope: str,
                          start: datetime, end: datetime, live_root: Path,
                          now: Callable[[], datetime], max_requests: int = 128,
                          limit: int = 1000) -> dict[str, Any]:
    """Persist each response before parsing; bounded partitioning never asserts coverage."""
    if endpoint not in {"/fapi/v1/userTrades", "/fapi/v1/income"}:
        raise ValueError("endpoint is not an approved read-only history source")
    start, end = utc(start), utc(end)
    if not bot.deployed_at <= start <= end <= utc(now()):
        raise ValueError("invalid or future acquisition interval")
    if not 1 <= max_requests <= 1024 or not 1 <= limit <= 1000:
        raise ValueError("invalid request budget or response limit")
    pending = [(int(start.timestamp() * 1000), int(end.timestamp() * 1000))]
    refs: list[dict[str, str]] = []
    rows: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = []
    rejected: list[str] = []
    requests = 0
    while pending:
        lower, upper = pending.pop()
        if requests >= max_requests:
            gaps.append({"start_ms": lower, "end_ms": upper, "reason": "request_budget"})
            gaps.extend({"start_ms": a, "end_ms": b, "reason": "request_budget"} for a, b in pending)
            break
        if upper - lower >= WEEK_MS:
            pending.extend([(lower + WEEK_MS, upper), (lower, lower + WEEK_MS - 1)])
            continue
        requested = utc(now())
        requests += 1
        try:
            response = await fetch(lower, upper)
        except Exception as exc:
            # Deliberately omit exception text: HTTP exceptions can contain
            # signed URLs, headers or account payloads. Stop on any transport
            # failure rather than repeatedly retrying a ban/rate-limit.
            gaps.append({"start_ms": lower, "end_ms": upper, "reason": "transport_failure",
                         "error_type": type(exc).__name__})
            gaps.extend({"start_ms": a, "end_ms": b, "reason": "aborted_after_failure"} for a, b in pending)
            break
        received = utc(now())
        captured = {"schema_version": SCHEMA, "record_type": "authenticated_rest_response",
                    "producer_sha256": PRODUCER_SHA256,
                    "account_scope": account_scope, "scope_basis": "credential_fingerprint_not_account_uid",
                    "host": HOST, "endpoint": endpoint,
                    "parameters": {"symbol": bot.symbol, "startTime": lower, "endTime": upper, "limit": limit},
                    "requested_at": requested.isoformat(), "received_at": received.isoformat(),
                    "payload": response, "authentication": "signed_request_tls_local_capture",
                    "server_signed_response": False, "bot_attribution": "unproven"}
        path = publish(captured, bot=bot, live_root=live_root, at=received, category="responses")
        ref = source_ref(path)
        refs.append(ref.model_dump())
        if not isinstance(response, list) or len(response) > limit:
            rejected.append("invalid_response_shape_or_limit")
            break
        invalid = False
        for index, raw in enumerate(response):
            try:
                if not isinstance(raw, dict) or raw.get("symbol") != bot.symbol:
                    raise ValueError("response_symbol_mismatch")
                at = instant_ms(raw.get("time"))
                stamp = int(integer_id(raw.get("time")))
                if not lower <= stamp <= upper or at > received:
                    raise ValueError("response_timestamp_outside_request")
                identity = integer_id(raw.get("id" if endpoint.endswith("userTrades") else "tranId"))
                if endpoint.endswith("income"):
                    if not isinstance(raw.get("incomeType"), str) or not raw["incomeType"]:
                        raise ValueError("missing_income_type")
                    identity += ":" + raw["incomeType"]
                else:
                    integer_id(raw.get("orderId"))
                prior = rows.get(identity)
                if prior and prior["raw"] != raw:
                    raise ValueError("conflicting_native_identity")
                if prior is None:
                    rows[identity] = {"raw": raw, "source": ref.model_dump(),
                                      "pointer": f"/payload/{index}", "available_at": received.isoformat()}
            except (ValueError, TypeError, OverflowError, OSError) as exc:
                rejected.append(str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
                invalid = True
        if invalid:
            break
        if len(response) == limit:
            if lower == upper:
                gaps.append({"start_ms": lower, "end_ms": upper, "reason": "saturated_millisecond"})
            else:
                mid = (lower + upper) // 2
                pending.extend([(mid + 1, upper), (lower, mid)])
    # A finite successful query is retrieval evidence, never proof of retention,
    # every order's ownership, interventions, funding allocation or equity.
    result = {"schema_version": SCHEMA, "bot": bot.model_dump(mode="json"),
              "producer_sha256": PRODUCER_SHA256,
              "account_scope": account_scope, "endpoint": endpoint, "start": start.isoformat(),
              "end": end.isoformat(), "requests": requests, "sources": refs,
              "rows": [] if rejected else list(rows.values()), "rejected": rejected, "gaps": gaps,
              "retrieval_exhausted": not gaps and not rejected,
              "event_completeness": "unproven", "ownership": "unproven",
              "training_eligible": False, "influence_enabled": False}
    manifest = publish(result, bot=bot, live_root=live_root, at=utc(now()), category="manifests")
    return result | {"manifest": source_ref(manifest).model_dump()}


def reviewed_ownership(*, source: SourceRef, bot: BotIdentity, account_scope: str,
                       live_root: Path, now: datetime) -> list[Path]:
    """Admit an explicit reviewed mapping, retaining original native receipt refs.

    A review declaration is not generated here. Every mapped order must occur
    in a hash-bound original containing both strategyId and orderId explicitly.
    No symbol/time inference or ORDER_TRADE_UPDATE.si interpretation is allowed.
    """
    mapping, digest = read_json(Path(source.path))
    if digest != source.sha256 or mapping.get("schema_version") != "neutralgrid_reviewed_native_ownership_v1":
        raise ValueError("reviewed ownership mapping source invalid")
    if BotIdentity.model_validate(mapping.get("bot")) != bot or mapping.get("account_scope") != account_scope:
        raise ValueError("ownership account/deployment mismatch")
    reviewed = utc(datetime.fromisoformat(mapping["reviewed_at"]))
    if reviewed > utc(now) or not isinstance(mapping.get("reviewer"), str) or not mapping["reviewer"].strip():
        raise ValueError("original review identity/time required")
    entries = mapping.get("orders")
    if not isinstance(entries, list) or not entries or len(entries) > 10000:
        raise ValueError("bounded nonempty original order mapping required")
    prepared = []
    seen: set[str] = set()
    for entry in entries:
        ref = SourceRef.model_validate(entry["source"])
        original, checksum = read_json(Path(ref.path))
        if checksum != ref.sha256:
            raise ValueError("original native order receipt changed")
        # This intentionally accepts only explicit receipts, not arbitrary nested
        # fields from account-wide events or client-order naming conventions.
        order = integer_id(original.get("orderId"))
        if (original.get("symbol") != bot.symbol or str(original.get("strategyId")) != bot.strategy_id
                or original.get("account_scope") != account_scope
                or BotIdentity.model_validate(original.get("bot")) != bot):
            raise ValueError("original does not explicitly bind order/account/deployment")
        occurred = instant_ms(original.get("time"))
        available = utc(datetime.fromisoformat(original["available_at"]))
        if order in seen or not bot.deployed_at <= occurred <= available <= reviewed:
            raise ValueError("duplicate mapping or invalid original/review timing")
        seen.add(order)
        body = {"bot": bot.model_dump(mode="json"), "account_scope": account_scope,
                "kind": "order_ownership", "native_id": order, "occurred_at": occurred.isoformat(),
                "available_at": reviewed.isoformat(), "fields": {"order_id": order,
                "ownership_basis": "reviewed_original_order_mapping"},
                "native_provenance": {"mapping": source.model_dump(), "original": ref.model_dump()}}
        prepared.append(body)
    paths = []
    for body in prepared:
        path = publish(body, bot=bot, live_root=live_root, at=now, category="producer_records")
        fact = PrivateFact.model_validate({k: v for k, v in body.items() if k != "native_provenance"} |
                                         {"source": source_ref(path).model_dump()})
        paths.append(append_fact(fact, live_root=live_root, now=now))
    return paths


def admit_fills(history: Mapping[str, Any], *, bot: BotIdentity, account_scope: str,
                ownership_paths: list[Path], live_root: Path, now: datetime) -> dict[str, Any]:
    """Capture exact owned fills; raw source and original mapping remain replayable."""
    if history.get("endpoint") != "/fapi/v1/userTrades" or history.get("rejected"):
        raise ValueError("valid trade history required")
    if history.get("bot") != bot.model_dump(mode="json") or history.get("account_scope") != account_scope:
        raise ValueError("history scope mismatch")
    owners: dict[str, tuple[PrivateFact, SourceRef]] = {}
    for path in ownership_paths:
        path = path.resolve()
        record, _ = read_json(path)
        owner = PrivateFact.model_validate(record["fact"])
        if (live_root.resolve() not in path.parents or path.parent.name != "facts"
                or path.parent.parent.name != owner.bot.key or path.parents[2].name != "exact_bot_ledger"
                or path.parents[3].name != owner.bot.symbol or path.stem != owner.identity
                or record.get("record_sha256") != fingerprint(owner.model_dump(mode="json"))):
            raise ValueError("ownership is not an intact committed canonical fact")
        verify_fact(owner, live_root=live_root)
        if owner.bot != bot or owner.account_scope != account_scope or owner.kind != "order_ownership":
            raise ValueError("ownership fact scope mismatch")
        owners[owner.fields["order_id"]] = (owner, source_ref(path))
    prepared = []
    existing_paths: list[str] = []
    candidates: dict[str, PrivateFact] = {}
    unlinked = 0
    for item in history["rows"]:
        ref = SourceRef.model_validate(item["source"])
        capture, digest = read_json(Path(ref.path))
        if digest != ref.sha256 or capture.get("account_scope") != account_scope or capture.get("endpoint") != "/fapi/v1/userTrades":
            raise ValueError("captured response source mismatch")
        pointer = item["pointer"]
        if not re.fullmatch(r"/payload/(0|[1-9][0-9]*)", pointer):
            raise ValueError("invalid response record pointer")
        raw = capture["payload"][int(pointer.rsplit("/", 1)[1])]
        if raw != item["raw"] or raw.get("symbol") != bot.symbol:
            raise ValueError("captured trade content mismatch")
        order = integer_id(raw["orderId"])
        if order not in owners:
            unlinked += 1
            continue
        owner, ownership = owners[order]
        if raw.get("marginAsset") != "USDT":
            raise ValueError("realized PnL requires explicit USDT settlement denomination")
        available = max(owner.available_at, utc(datetime.fromisoformat(capture["received_at"])))
        fields = {"order_id": order, "trade_id": integer_id(raw["id"]), "side": raw["side"],
                  "price": raw["price"], "quantity": raw["qty"], "realized_pnl_usdt": raw["realizedPnl"],
                  "commission": raw["commission"], "commission_asset": raw["commissionAsset"]}
        body = {"bot": bot.model_dump(mode="json"), "account_scope": account_scope, "kind": "trade_fill",
                "native_id": fields["trade_id"], "occurred_at": instant_ms(raw["time"]).isoformat(),
                "available_at": available.isoformat(), "fields": fields,
                "native_provenance": {"response": ref.model_dump(), "pointer": pointer,
                                      "ownership": ownership.model_dump()}}
        # Validate all rows before committing any fact; no zero imputation.
        candidate = PrivateFact.model_validate({k: v for k, v in body.items() if k != "native_provenance"} |
                                               {"source": ref.model_dump(), "ownership_fact": ownership.model_dump()})
        earlier = candidates.get(candidate.identity)
        if earlier is not None:
            if (earlier.bot != candidate.bot or earlier.fields != candidate.fields
                    or earlier.occurred_at != candidate.occurred_at):
                raise ValueError("conflicting native trades in admission batch")
            continue
        candidates[candidate.identity] = candidate
        prior_paths = sorted(live_root.glob(f"????-??-??/{bot.symbol}/exact_bot_ledger/*/facts/{candidate.identity}.json"))
        if prior_paths:
            if len(prior_paths) != 1:
                raise ValueError("duplicate committed native identity paths")
            prior_record, _ = read_json(prior_paths[0])
            prior = PrivateFact.model_validate(prior_record["fact"])
            verify_fact(prior, live_root=live_root)
            if (prior_record.get("record_sha256") != fingerprint(prior.model_dump(mode="json"))
                    or prior.bot != candidate.bot or prior.account_scope != candidate.account_scope
                    or prior.kind != candidate.kind or prior.fields != candidate.fields
                    or prior.occurred_at != candidate.occurred_at):
                raise ValueError("conflicting committed native trade")
            existing_paths.append(str(prior_paths[0]))
            continue
        prepared.append((body, ownership))
    paths = existing_paths
    for body, ownership in prepared:
        path = publish(body, bot=bot, live_root=live_root, at=now, category="producer_records")
        fact = PrivateFact.model_validate({k: v for k, v in body.items() if k != "native_provenance"} |
                                         {"source": source_ref(path).model_dump(), "ownership_fact": ownership.model_dump()})
        paths.append(str(append_fact(fact, live_root=live_root, now=now)))
    return {"fact_paths": paths, "unlinked_rows": unlinked, "training_eligible": False}


def verify_native_record(raw: Mapping[str, Any], fact: PrivateFact) -> None:
    """Replay source lineage on every ledger read, not only at initial import."""
    provenance = raw["native_provenance"]
    if fact.kind == "order_ownership":
        mapping_ref = SourceRef.model_validate(provenance["mapping"])
        original_ref = SourceRef.model_validate(provenance["original"])
        mapping, checksum = read_json(Path(mapping_ref.path))
        original, original_hash = read_json(Path(original_ref.path))
        if checksum != mapping_ref.sha256 or original_hash != original_ref.sha256:
            raise ValueError("native ownership original changed")
        if (mapping.get("schema_version") != "neutralgrid_reviewed_native_ownership_v1"
                or BotIdentity.model_validate(mapping.get("bot")) != fact.bot
                or mapping.get("account_scope") != fact.account_scope
                or not isinstance(mapping.get("reviewer"), str) or not mapping["reviewer"].strip()
                or not any(entry.get("source") == original_ref.model_dump() for entry in mapping.get("orders", []))):
            raise ValueError("native ownership review mismatch")
        if (integer_id(original.get("orderId")) != fact.fields["order_id"]
                or str(original.get("strategyId")) != fact.bot.strategy_id
                or original.get("symbol") != fact.bot.symbol
                or original.get("account_scope") != fact.account_scope
                or BotIdentity.model_validate(original.get("bot")) != fact.bot
                or instant_ms(original.get("time")) != fact.occurred_at
                or utc(datetime.fromisoformat(mapping["reviewed_at"])) != fact.available_at
                or not fact.occurred_at <= utc(datetime.fromisoformat(original["available_at"])) <= fact.available_at):
            raise ValueError("native ownership content mismatch")
    elif fact.kind == "trade_fill":
        ref = SourceRef.model_validate(provenance["response"])
        capture, checksum = read_json(Path(ref.path))
        pointer = provenance["pointer"]
        if checksum != ref.sha256 or not re.fullmatch(r"/payload/(0|[1-9][0-9]*)", pointer):
            raise ValueError("native trade response changed")
        if (capture.get("schema_version") != SCHEMA or capture.get("host") != HOST
                or capture.get("endpoint") != "/fapi/v1/userTrades"
                or capture.get("account_scope") != fact.account_scope
                or fact.ownership_fact is None or provenance["ownership"] != fact.ownership_fact.model_dump()):
            raise ValueError("native trade acquisition scope mismatch")
        trade = capture["payload"][int(pointer.rsplit("/", 1)[1])]
        fields = {"order_id": integer_id(trade["orderId"]), "trade_id": integer_id(trade["id"]),
                  "side": trade["side"], "price": trade["price"], "quantity": trade["qty"],
                  "realized_pnl_usdt": trade["realizedPnl"], "commission": trade["commission"],
                  "commission_asset": trade["commissionAsset"]}
        if (trade.get("symbol") != fact.bot.symbol or trade.get("marginAsset") != "USDT" or fields != fact.fields
                or instant_ms(trade["time"]) != fact.occurred_at
                or not fact.occurred_at <= utc(datetime.fromisoformat(capture["received_at"])) <= fact.available_at):
            raise ValueError("native trade content mismatch")
    else:
        raise ValueError("unsupported authenticated producer fact kind")


def capture_pnl_path(paths: list[Path], *, bot: BotIdentity, live_root: Path, now: datetime) -> Path:
    """Index source-verified timestamped reported PnL without claiming equity/path coverage."""
    points: dict[str, dict[str, Any]] = {}
    for path in paths:
        original, _ = read_json(path)
        record = validate_pnl_observation(original)
        actual = BotIdentity(symbol=record["symbol"], strategy_id=record["strategy_id"], deployed_at=record["deploy_ts"])
        if actual != bot or record["captured_at"] > utc(now):
            raise ValueError("PnL path deployment or future timestamp mismatch")
        if not record["captured_at"] <= record["evaluated_at"] <= utc(now):
            raise ValueError("PnL evaluation does not prove causal availability")
        originals = verify_observation_sources(record, live_root=live_root)
        key = record["observation_id"]
        point = {"observation_id": key, "occurred_at": record["captured_at"].isoformat(),
                 "available_at": record["evaluated_at"].isoformat(),
                 "reported_total_pnl_usdt": str(decimal(record["pnl"]["total_profit_usdt"])),
                 "reported_invested_margin_usdt": str(decimal(record["grid"]["invested_margin_usdt"])),
                 "observation_source": source_ref(path).model_dump(), "original_sources": originals}
        if key in points and points[key] != point:
            raise ValueError("conflicting PnL observation identity")
        points[key] = point
    if not points:
        raise ValueError("PnL path requires original observations")
    body = {"schema_version": SCHEMA, "record_type": "timestamped_reported_pnl_path",
            "producer_sha256": PRODUCER_SHA256,
            "bot": bot.model_dump(mode="json"), "points": sorted(points.values(), key=lambda p: p["occurred_at"]),
            "measurement": "reported_total_pnl_not_verified_equity",
            "deadline": (bot.deployed_at + timedelta(hours=7)).isoformat(),
            "target_pct": "3", "capital_basis": "reported_invested_margin",
            "path_coverage": "path_unobserved_between_snapshots", "training_eligible": False,
            "influence_enabled": False}
    return publish(body, bot=bot, live_root=live_root, at=now, category="pnl_paths")
