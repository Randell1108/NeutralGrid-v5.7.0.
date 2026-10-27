"""Account-wide income originals, never attributed to a bot by storage location.

Motivating verified capture: reports/live_telemetry_account_scope_20261022/
product_scope_comparison.json records 52 blank-symbol account transfers while
its symbol-filtered income query returned zero. This module never infers bot
ownership from those transfers.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Mapping

from neutralgrid.live.decision.authenticated_evidence import HOST, integer_id, instant_ms, source_ref
from neutralgrid.live.decision.evidence_cohort import LIMA, MAX_BYTES, read_json
from neutralgrid.live.decision.exact_bot_ledger import ledger_lock
from neutralgrid.live.decision.state_action_reward import atomic_create_json, decimal, file_hash, fingerprint, utc

SCHEMA = "neutralgrid_account_income_evidence_v1"
ENDPOINT = "/fapi/v1/income"
PRODUCER_SHA256 = file_hash(Path(__file__))


def validate_context(account_scope: str, storage_symbol: str) -> None:
    if not re.fullmatch(r"credential_[0-9a-f]{64}", account_scope):
        raise ValueError("explicit credential fingerprint scope required")
    validate_storage_symbol(storage_symbol)


def validate_storage_symbol(storage_symbol: str) -> None:
    if not isinstance(storage_symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,32}", storage_symbol):
        raise ValueError("invalid storage context symbol")


def validate_limits(max_requests: int, limit: int, max_records: int) -> None:
    if (type(max_requests) is not int or not 1 <= max_requests <= 1024
            or type(limit) is not int or not 1 <= limit <= 1000
            or type(max_records) is not int or not 1 <= max_records <= 10000):
        raise ValueError("invalid acquisition bounds")


def validate_interval(start: datetime, end: datetime, at: datetime) -> tuple[datetime, datetime]:
    start, end, at = utc(start), utc(end), utc(at)
    if (start.timestamp() < 0 or not start <= end <= at or end - start > timedelta(days=90)
            or start.microsecond % 1000 or end.microsecond % 1000):
        raise ValueError("interval must be past, ordered, millisecond-aligned and at most 90 days")
    return start, end


def publish_account_income(body: Mapping[str, Any], *, account_scope: str, storage_symbol: str,
                           live_root: Path, at: datetime, category: str) -> Path:
    validate_context(account_scope, storage_symbol)
    if category not in {"responses", "records", "manifests"}:
        raise ValueError("unsupported account income category")
    payload = dict(body)
    if len(json.dumps(payload, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("account evidence exceeds storage bound")
    digest = fingerprint(payload)
    root = live_root.resolve()
    with ledger_lock(root / ".account_income_locks" / (digest + ".lock")):
        existing = sorted(root.glob(f"????-??-??/{storage_symbol}/account_income/{account_scope}/{category}/{digest}.json"))
        if existing:
            if len(existing) != 1 or read_json(existing[0])[0] != payload:
                raise ValueError("conflicting immutable account income")
            return existing[0].resolve()
        path = root / utc(at).astimezone(LIMA).date().isoformat() / storage_symbol / "account_income" / account_scope / category / (digest + ".json")
        atomic_create_json(path, payload)
        return path


async def fetch_income(client: Any, lower: int, upper: int, limit: int) -> Any:
    """Intentionally omit symbol and incomeType so blank-symbol flows survive."""
    return await client._request(ENDPOINT, {"startTime": lower, "endTime": upper, "limit": limit},
                                 signed=True, max_retries=0)


def _canonical_records(rows: list[dict[str, Any]], *, account_scope: str, storage_symbol: str,
                       live_root: Path, at: datetime) -> tuple[list[dict[str, Any]], int]:
    """Preserve first availability across refetches and storage-context changes."""
    root = live_root.resolve()
    prepared: list[tuple[Path, dict[str, Any]]] = []
    canonical: list[dict[str, Any]] = []
    with ledger_lock(root / ".account_income_locks" / (account_scope + ".lock")):
        # Validate the whole batch before publishing any canonical identity.
        for row in rows:
            digest = fingerprint({"account_scope": account_scope, "identity": row["identity"]})
            paths = sorted(root.glob(f"????-??-??/*/account_income/{account_scope}/identities/{digest}.json"))
            if paths:
                if len(paths) != 1:
                    raise ValueError("duplicate_committed_income_identity")
                record, _ = read_json(paths[0])
                checksum = record.pop("record_sha256", None)
                if (checksum != fingerprint(record) or record.get("schema_version") != SCHEMA
                        or record.get("account_scope") != account_scope or record.get("record_type") != "canonical_account_income"
                        or record.get("record", {}).get("identity") != row["identity"]
                        or record.get("bot_attribution") != "none" or record.get("training_eligible") is not False
                        or record.get("influence_enabled") is not False):
                    raise ValueError("invalid_committed_income_identity")
                prior = record["record"]
                ref = prior["source"]
                source_path = Path(ref["path"]).resolve()
                parts = source_path.relative_to(root).parts
                if (len(parts) != 6 or parts[2:5] != ("account_income", account_scope, "responses")
                        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[0])):
                    raise ValueError("committed_income_source_outside_capture_store")
                validate_storage_symbol(parts[1])
                original, source_hash = read_json(source_path)
                pointer = prior["pointer"]
                if (source_hash != ref["sha256"] or original.get("account_scope") != account_scope
                        or original.get("schema_version") != SCHEMA or original.get("record_type") != "account_income_response"
                        or original.get("host") != HOST or original.get("endpoint") != ENDPOINT
                        or original.get("bot_attribution") != "none" or original.get("training_eligible") is not False
                        or original.get("influence_enabled") is not False or prior.get("bot_attribution") != "none"
                        or source_path.name != fingerprint(original) + ".json"
                        or not re.fullmatch(r"/payload/(0|[1-9][0-9]*)", pointer)
                        or original["payload"][int(pointer.rsplit("/", 1)[1])] != prior["raw"]):
                    raise ValueError("committed_income_source_changed")
                received = utc(datetime.fromisoformat(original["received_at"]))
                available = utc(datetime.fromisoformat(prior["available_at"]))
                native = prior["raw"]
                params = original["parameters"]
                if (available != received or utc(datetime.fromisoformat(original["requested_at"])) > received
                        or native["incomeType"] + ":" + integer_id(native["tranId"]) != prior["identity"]
                        or not params["startTime"] <= int(integer_id(native["time"])) <= params["endTime"]
                        or instant_ms(native["time"]) > received or "symbol" in params or "incomeType" in params):
                    raise ValueError("committed_income_availability_or_identity_changed")
                if prior["raw"] != row["raw"]:
                    raise ValueError("conflicting_committed_income_identity")
                canonical.append(prior)
            else:
                body = {"schema_version": SCHEMA, "record_type": "canonical_account_income",
                        "account_scope": account_scope, "bot_attribution": "none", "training_eligible": False,
                        "influence_enabled": False, "record": row}
                path = root / utc(at).astimezone(LIMA).date().isoformat() / storage_symbol / "account_income" / account_scope / "identities" / (digest + ".json")
                prepared.append((path, body | {"record_sha256": fingerprint(body)}))
                canonical.append(row)
        for path, body in prepared:
            atomic_create_json(path, body)
    return canonical, len(prepared)


async def collect_account_income(*, fetch: Callable[[int, int, int], Awaitable[Any]],
                                 account_scope: str, storage_symbol: str, start: datetime,
                                 end: datetime, live_root: Path, now: Callable[[], datetime],
                                 max_requests: int = 32, limit: int = 1000,
                                 max_records: int = 4096) -> dict[str, Any]:
    """Bound queries and quarantine conflicts; exhaustion never means full coverage."""
    validate_context(account_scope, storage_symbol)
    start, end = validate_interval(start, end, now())
    validate_limits(max_requests, limit, max_records)
    common = {"schema_version": SCHEMA, "producer_sha256": PRODUCER_SHA256,
              "account_scope": account_scope, "scope_basis": "credential_fingerprint_not_verified_account_uid",
              "storage_context_symbol": storage_symbol, "storage_symbol_is_filter": False,
              "bot_attribution": "none", "training_eligible": False, "influence_enabled": False}
    pending = [(int(start.timestamp() * 1000), int(end.timestamp() * 1000))]
    refs: list[dict[str, Any]] = []
    rows: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = []
    rejected: list[str] = []
    requests = 0
    while pending:
        lower, upper = pending.pop()
        if requests >= max_requests:
            gaps.extend({"start_ms": a, "end_ms": b, "reason": "request_budget"} for a, b in [(lower, upper), *pending])
            break
        # Bounded seven-day partitions are an acquisition choice, not a claim
        # that the income endpoint shares userTrades' seven-day restriction.
        week_ms = 7 * 86400 * 1000
        if upper - lower >= week_ms:
            pending.extend([(lower + week_ms, upper), (lower, lower + week_ms - 1)])
            continue
        requested = utc(now())
        requests += 1
        try:
            response = await fetch(lower, upper, limit)
        except Exception as exc:
            gaps.append({"start_ms": lower, "end_ms": upper, "reason": "transport_failure", "error_type": type(exc).__name__})
            gaps.extend({"start_ms": a, "end_ms": b, "reason": "aborted_after_failure"} for a, b in pending)
            break
        received = utc(now())
        capture = common | {"record_type": "account_income_response", "host": HOST, "endpoint": ENDPOINT,
                            "parameters": {"startTime": lower, "endTime": upper, "limit": limit},
                            "requested_at": requested.isoformat(), "received_at": received.isoformat(),
                            "authentication": "signed_request_tls_local_capture", "server_signed_response": False,
                            "payload": response}
        try:
            path = publish_account_income(capture, account_scope=account_scope, storage_symbol=storage_symbol,
                                          live_root=live_root, at=received, category="responses")
        except (ValueError, TypeError, OSError, RuntimeError):
            rejected.append("response_not_persisted")
            break
        ref = source_ref(path).model_dump()
        refs.append(ref)
        if received < requested or not isinstance(response, list) or len(response) > limit:
            rejected.append("invalid_response_shape_limit_or_clock")
            break
        for index, raw in enumerate(response):
            try:
                if not isinstance(raw, dict) or not isinstance(raw.get("symbol"), str):
                    raise ValueError("invalid_income_record_or_symbol")
                kind = raw.get("incomeType")
                asset = raw.get("asset")
                if (not isinstance(kind, str) or not re.fullmatch(r"[A-Z0-9_]{1,80}", kind)
                        or not isinstance(asset, str) or not re.fullmatch(r"[A-Z0-9]{1,32}", asset)):
                    raise ValueError("invalid_income_type_or_asset")
                decimal(raw.get("income"))
                stamp = int(integer_id(raw.get("time")))
                if not lower <= stamp <= upper or instant_ms(stamp) > received:
                    raise ValueError("income_timestamp_outside_request")
                key = kind + ":" + integer_id(raw.get("tranId"))
                prior = rows.get(key)
                if prior is not None and prior["raw"] != raw:
                    raise ValueError("conflicting_income_identity")
                if prior is None:
                    if len(rows) >= max_records:
                        raise ValueError("record_budget_exceeded")
                    rows[key] = {"identity": key, "raw": raw, "source": ref, "pointer": f"/payload/{index}",
                                 "available_at": received.isoformat(), "bot_attribution": "none"}
            except (ValueError, TypeError, OverflowError, OSError) as exc:
                # Exceptions from financial parsers can embed private values.
                rejected.append(str(exc) if str(exc) in {"invalid_income_record_or_symbol", "invalid_income_type_or_asset",
                    "income_timestamp_outside_request", "conflicting_income_identity", "record_budget_exceeded"} else "invalid_income_value_or_identity")
        if rejected:
            break
        if len(response) == limit:
            if lower == upper:
                gaps.append({"start_ms": lower, "end_ms": upper, "reason": "saturated_millisecond"})
            else:
                mid = (lower + upper) // 2
                pending.extend([(mid + 1, upper), (lower, mid)])
    accepted = [] if rejected else sorted(rows.values(), key=lambda item: (int(item["raw"]["time"]), item["identity"]))
    new_identities: int | None = 0
    if accepted:
        try:
            accepted, new_identities = _canonical_records(accepted, account_scope=account_scope,
                storage_symbol=storage_symbol, live_root=live_root, at=now())
        except (ValueError, TypeError, OSError, RuntimeError, KeyError, IndexError):
            rejected.append("committed_identity_conflict_or_source_invalid")
            accepted = []
            new_identities = None
    record_refs = []
    canonical_records = len(accepted)
    for offset in range(0, len(accepted), 256):
        try:
            path = publish_account_income(common | {"record_type": "account_income_records", "records": accepted[offset:offset + 256]},
                                          account_scope=account_scope, storage_symbol=storage_symbol,
                                          live_root=live_root, at=now(), category="records")
            record_refs.append(source_ref(path).model_dump())
        except (ValueError, TypeError, OSError, RuntimeError):
            rejected.append("record_shard_not_persisted")
            accepted = accepted[:offset]
            break
    result = common | {"record_type": "account_income_manifest", "start_at": start.isoformat(), "end_at": end.isoformat(),
                       "endpoint": ENDPOINT, "requests": requests, "max_requests": max_requests, "max_records": max_records,
                       "sources": refs, "record_shards": record_refs, "accepted_records": len(accepted),
                       "new_canonical_identities": new_identities,
                       "canonical_records_validated": canonical_records,
                       "canonical_commit_status": "partial_or_unknown" if new_identities is None else "complete",
                       "income_types": dict(Counter(row["raw"]["incomeType"] for row in accepted)),
                       "blank_symbol_records": sum(row["raw"]["symbol"] == "" for row in accepted),
                       "rejected": sorted(set(rejected)), "gaps": gaps,
                       "retrieval_exhausted": not gaps and not rejected, "event_completeness": "unknown",
                       "promotion_eligible": False}
    try:
        path = publish_account_income(result, account_scope=account_scope, storage_symbol=storage_symbol,
                                      live_root=live_root, at=now(), category="manifests")
        return result | {"manifest": source_ref(path).model_dump()}
    except (ValueError, TypeError, OSError, RuntimeError):
        return result | {"manifest": None, "rejected": [*result["rejected"], "manifest_not_persisted"],
                         "retrieval_exhausted": False}
