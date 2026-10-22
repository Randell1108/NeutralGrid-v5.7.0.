"""Source-checked research cohorts; never synthesize behavior or train a policy.

The target is reported total bot PnL / invested margin >= 3% no later than
deployment + seven hours. Sparse snapshots can witness hits, never non-hits.
Packets are provenance indexes, deliberately outside canonical training paths.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.private_telemetry import parse_private_telemetry_text
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, atomic_create_json, decimal, fingerprint, utc,
)


SCHEMA = "neutralgrid_live_evidence_cohort_v1"
TARGET_PCT = Decimal("3")
HORIZON = timedelta(hours=7)
MAX_BYTES = 4 * 1024 * 1024
LIMA = timezone(-timedelta(hours=5))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_number(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("source exceeds bounded read size")
    return data


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    data = read_bytes(path)
    payload = json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_pairs,
                         parse_constant=_invalid_number)
    if not isinstance(payload, dict):
        raise ValueError("JSON source must be an object")
    return payload, hashlib.sha256(data).hexdigest()


def _ref(path: Path, digest: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest}


def verify_observation_sources(record: Mapping[str, Any], *, live_root: Path) -> list[dict[str, str]]:
    """Reparse the native drawer and bind identity/time to its capture envelope.

    A page URL is not bot identity: the browser can display a different drawer.
    This proves the stored UI measurement, not exchange-complete accounting.
    """
    root = live_root.resolve()
    raw_path = Path(str(record["source_snapshot_path"])).resolve()
    if root not in raw_path.parents or raw_path.parent.name != record["symbol"]:
        raise ValueError("raw snapshot is outside the exact symbol Live directory")
    data = read_bytes(raw_path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != record["source_snapshot_sha256"]:
        raise ValueError("raw snapshot hash mismatch")
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    strategies = re.findall(r"Strategy Number\s+(\d+)", text)
    created = re.findall(r"Time Created\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if (not lines or lines[0].strip() != record["symbol"] or
            set(strategies) != {str(record["strategy_id"])} or len(set(created)) != 1):
        raise ValueError("native drawer identity is missing or ambiguous")
    deployed = datetime.fromisoformat(created[0]).replace(tzinfo=LIMA).astimezone(timezone.utc)
    if deployed != record["deploy_ts"]:
        raise ValueError("native drawer deployment mismatch")
    envelope_path = raw_path.with_suffix(".json")
    envelope, envelope_hash = read_json(envelope_path)
    if (envelope.get("symbol") != record["symbol"] or
            str(envelope.get("strategy_id")) != record["strategy_id"] or
            envelope.get("raw_text_sha256") != digest):
        raise ValueError("capture envelope identity/hash mismatch")
    captured = utc(datetime.fromisoformat(str(envelope["captured_at_utc"]).replace("Z", "+00:00")))
    if captured != record["captured_at"]:
        raise ValueError("capture envelope time mismatch")
    native = parse_private_telemetry_text(text)
    for section, key in (("pnl", "total_profit_usdt"), ("grid", "invested_margin_usdt")):
        if decimal(native[section].get(key)) != decimal(record[section].get(key)):
            raise ValueError(f"native drawer {section}.{key} mismatch")
    return [_ref(raw_path, digest), _ref(envelope_path, envelope_hash)]


def collect_observations(live_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    unique: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    inputs: list[dict[str, str]] = []
    bad_bots: set[str] = set()
    for path in sorted(live_root.glob("????-??-??/*/pnl_history/*/observations/*.json")):
        bot = path.parent.parent.name if re.fullmatch(r"[0-9a-f]{64}", path.parent.parent.name) else None
        try:
            raw, digest = read_json(path)
            inputs.append(_ref(path, digest))
            record = validate_pnl_observation(raw)
            bot = str(record["bot_identity"])
            # Match append_pnl_observation's timestamp + truncated-id filename.
            expected_name = f"{record['captured_at'].strftime('%Y%m%dT%H%M%S%fZ')}_{record['observation_id'][:16]}.json"
            if (path.parent.parent.name != bot or path.parents[3].name != record["symbol"] or
                    path.name != expected_name):
                raise ValueError("misrouted canonical observation")
            if record["captured_at"] < record["deploy_ts"]:
                raise ValueError("capture precedes deployment")
            sources = verify_observation_sources(record, live_root=live_root)
            record["_source_path"] = str(path.resolve())
            record["_evidence_sources"] = [_ref(path, digest), *sources]
            inputs.extend(sources)
            prior = unique.get(record["observation_id"])
            if prior is not None and prior["record_integrity_sha256"] != record["record_integrity_sha256"]:
                raise ValueError("conflicting full observation identity")
            unique.setdefault(record["observation_id"], record)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            if bot is not None:
                bad_bots.add(bot)
            rejected.append({"path": str(path), "reason": str(exc)})
    # Partial histories must not survive a known conflict in the same deployment.
    accepted = [r for r in unique.values() if r["bot_identity"] not in bad_bots]
    return accepted, rejected, inputs


def target_cohort(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique: dict[str, dict[str, Any]] = {}
    for raw in observations:
        # Audit metadata is not part of the immutable observation schema.
        record = validate_pnl_observation({k: v for k, v in raw.items() if k != "_evidence_sources"})
        prior = unique.get(record["observation_id"])
        if prior is not None and prior["record_integrity_sha256"] != record["record_integrity_sha256"]:
            raise ValueError("conflicting full observation identity")
        unique.setdefault(record["observation_id"], record)
    for record in unique.values():
        grouped[str(record["bot_identity"])].append(record)
    bots, origins = [], []
    for identity, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["captured_at"])
        deadline = rows[0]["deploy_ts"] + HORIZON
        margins: set[Decimal] = set()
        valid_basis = True
        for row in rows:
            try:
                margin = decimal(row["grid"].get("invested_margin_usdt"))
                if margin <= 0:
                    raise ValueError("nonpositive capital")
                margins.add(margin)
            except (ValueError, TypeError):
                valid_basis = False
        valid_basis = valid_basis and len(margins) == 1
        hits = [r for r in rows if valid_basis and r["deploy_ts"] <= r["captured_at"] <= deadline
                and decimal(r["pnl"]["total_profit_usdt"]) * 100 >= TARGET_PCT * next(iter(margins))]
        first_hit = hits[0] if hits else None
        status = ("capital_basis_changed_or_missing" if not valid_basis else
                  "observed_target_hit" if first_hit is not None else "censored_target_path")
        bots.append({"bot_identity": identity, "symbol": rows[0]["symbol"],
                     "strategy_id": rows[0]["strategy_id"], "deploy_ts_utc": rows[0]["deploy_ts_utc"],
                     "deadline_utc": deadline.isoformat(), "classification": status,
                     "observations": len(rows), "first_observed_hit_id": first_hit["observation_id"] if first_hit else None,
                     "continuous_path_complete": False, "initial_capital_coverage_proven": False})
        for row in rows:
            at = row["captured_at"]
            remaining = (deadline - at).total_seconds() / 60
            label = None
            classification = status
            if at < row["deploy_ts"] or at > deadline:
                classification = "outside_target_horizon"
            elif valid_basis and first_hit is not None and at >= first_hit["captured_at"]:
                classification = "already_observed_at_target"
            elif valid_basis and first_hit is not None:
                classification, label = "future_observed_target_hit", 1
            origins.append({"observation_id": row["observation_id"], "bot_identity": identity,
                            "symbol": row["symbol"], "strategy_id": row["strategy_id"],
                            "candidate_id": row.get("candidate_id"), "verdict": row["decision"].get("verdict"),
                            "captured_at_utc": at.isoformat(), "remaining_minutes": remaining,
                            "reported_pnl_pct": float(decimal(row["pnl"]["total_profit_usdt"]) * 100 / next(iter(margins))) if valid_basis else None,
                            "classification": classification, "target_label": label,
                            "behavior": None, "action_reward": None, "action_training_eligible": False})
    return {"schema_version": SCHEMA, "target_pct": str(TARGET_PCT), "horizon_minutes_from_deployment": 420,
            "denominator": "reported invested_margin_usdt; constant across available snapshots",
            "label_semantics": "witnessed UI measurement, not a continuous exchange-equity path",
            "bots": bots, "origins": origins, "positive_origins": sum(r["target_label"] == 1 for r in origins),
            "negative_origins": 0, "fit_eligible": False, "fitted": False,
            "influence_enabled": False, "promotion_eligible": False,
            "blockers": ["no_proven_negative_first_passage_labels", "complete_target_paths_required",
                         "capital_flow_coverage_required", "held_out_two_class_support_not_established"]}


def exchange_ticks(payload: Mapping[str, Any]) -> dict[str, str]:
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError("exchangeInfo.symbols must be a list")
    ticks: dict[str, str] = {}
    for record in symbols:
        symbol = str(record["symbol"])
        if symbol in ticks:
            raise ValueError("duplicate exchange symbol")
        filters = [f for f in record["filters"] if f.get("filterType") == "PRICE_FILTER"]
        if len(filters) != 1 or decimal(filters[0]["tickSize"]) <= 0:
            raise ValueError("one positive PRICE_FILTER tickSize required")
        ticks[symbol] = str(decimal(filters[0]["tickSize"]))
    return ticks


def publish_packet(packet: Mapping[str, Any], *, live_root: Path, now: datetime) -> Path:
    at = utc(now)
    symbol, identity = str(packet["symbol"]), str(packet["bot_identity"])
    if not re.fullmatch(r"[A-Z0-9]+USDT", symbol) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise ValueError("invalid packet identity")
    if len((json.dumps(dict(packet), sort_keys=True, allow_nan=False) + "\n").encode("utf-8")) > MAX_BYTES:
        raise ValueError("evidence packet exceeds bounded read size")
    digest = fingerprint(packet)
    # A stable content address deduplicates reruns without duplicating observations.
    existing = sorted(live_root.glob(f"????-??-??/{symbol}/evidence_cohort/{identity}/{digest}.json"))
    for path in existing:
        prior, _ = read_json(path)
        if fingerprint(prior) != digest:
            raise ValueError("conflicting evidence packet")
    if existing:
        return existing[0]
    destination = live_root / at.astimezone(LIMA).date().isoformat() / symbol / "evidence_cohort" / identity / f"{digest}.json"
    try:
        atomic_create_json(destination, packet)
    except FileExistsError:
        prior, _ = read_json(destination)
        if fingerprint(prior) != digest:
            raise ValueError("concurrent evidence packet conflict")
    return destination


def evidence_packets(observations: Sequence[Mapping[str, Any]], *, transitions: Sequence[Mapping[str, Any]],
                     tick_capture: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[str(row["bot_identity"])].append(row)
    packets = []
    for identity, rows in sorted(grouped.items()):
        first = rows[0]
        sar_bot = BotIdentity(symbol=first["symbol"], strategy_id=first["strategy_id"],
                              deployed_at=first["deploy_ts"])
        linked = [t for t in transitions if t["bot_key"] == sar_bot.key]
        sources = {fingerprint(s): s for row in rows for s in row["_evidence_sources"]}
        symbol_tick = ({"source": tick_capture["source"], "received_at_utc": tick_capture["received_at_utc"],
                        "tick_size": tick_capture["ticks"].get(first["symbol"])} if tick_capture else None)
        packets.append({"schema_version": SCHEMA, "record_type": "live_evidence_index",
                        "symbol": first["symbol"], "strategy_id": first["strategy_id"],
                        "bot_identity": identity, "sar_bot_key": sar_bot.key,
                        "deploy_ts_utc": first["deploy_ts_utc"], "sources": list(sources.values()),
                        "observation_ids": sorted(row["observation_id"] for row in rows),
                        "verdict_counts": dict(Counter(str(row["decision"].get("verdict") or "MISSING") for row in rows)),
                        "transitions": linked, "exchange_metadata_capture": symbol_tick,
                        "exchange_metadata_is_historical": False,
                        "training_eligible": False, "influence_enabled": False,
                        "requirements": ["verified decision with causal features and exact deployment",
                                         "verified execution plus approval/receipt, or complete unchanged-behavior evidence",
                                         "full interval exact-bot private events, interventions and capital flows",
                                         "fee/funding inclusion and equity reconciliation",
                                         "explicit per-bot liquidation and drawdown limits",
                                         "verified equity and peak equity; not sampled PnL peak",
                                         "exchange PRICE_FILTER tickSize available at the decision time",
                                         "bot-disjoint chronological held-out action classes and behavior support"],
                        "missing_action_link": not bool(linked),
                        "eligibility_evidence": {
                            "native_identity_time_pnl_margin": "verified",
                            "candidate_ids": sorted({str(r["candidate_id"]) for r in rows if r.get("candidate_id")}),
                            "observed_action_count": sum(t["observed_action"] is not None for t in linked),
                            "reconciled_reward_count": sum(t["reward_net_usdt"] is not None for t in linked),
                            "eligible_action_count": sum(t["training_eligible"] for t in linked),
                            "risk_profile_count": sum(t["risk_limits"] is not None for t in linked),
                            "equity_count": sum(t["equity_usdt"] is not None for t in linked),
                            "peak_equity_count": sum(t["peak_equity_usdt"] is not None for t in linked),
                            "tick_at_original_decision": "not_proven_by_current_fetch",
                            "private_event_completeness_claims": sorted({str(r["features"].get("private_event_completeness")) for r in rows
                                                                         if r["features"].get("private_event_completeness")}),
                        },
                        "source_verification_scope": "native drawer identity/time and reported PnL/margin; SAR producer assertions retain their existing verification scope"})
    return packets
