"""Capture source-checked live evidence indexes and audit the deployment +7h target.

No execution, scheduling, promotion, model loading or policy enablement occurs.
Use --capture to publish indexes under Live; --ingest-event accepts an original
canonical SAR envelope. Missing evidence is reported, never reconstructed from
subsequent profit or an unexecuted verdict. Public exchange metadata is optional.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.live.decision.action_challenger import evaluate
from neutralgrid.live.decision.evidence_cohort import (
    MAX_BYTES, SCHEMA, collect_observations, evidence_packets, exchange_ticks,
    publish_packet, read_json, target_cohort,
)
from neutralgrid.live.decision.exact_bot_ledger import audit_ledger
from neutralgrid.live.decision.state_action_reward import (
    LedgerEvent, append_event, atomic_create_json, contract_schema_catalog,
    file_hash, load_events, project_transition, verify_sources,
)

logger = logging.getLogger(__name__)
EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"


def fetch_exchange_info() -> dict[str, Any]:
    requested = datetime.now(timezone.utc)
    request = urllib.request.Request(EXCHANGE_INFO_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("exchange metadata exceeds size bound")
    payload = json.loads(data)
    ticks = exchange_ticks(payload)
    return {"url": EXCHANGE_INFO_URL, "requested_at_utc": requested.isoformat(),
            "received_at_utc": datetime.now(timezone.utc).isoformat(),
            "exchange_info": payload, "ticks": ticks,
            "historical_applicability_proven": False}


def research_output(path: Path, live_root: Path) -> Path:
    target = path.resolve()
    # Research output is confined to reports; it cannot overwrite any model store.
    reports = (ROOT / "reports").resolve()
    if reports not in target.parents or target == live_root.resolve() or live_root.resolve() in target.parents:
        raise ValueError("output must be a fresh subdirectory of repository reports, outside Live")
    if target.exists():
        raise ValueError("output directory already exists")
    return target


def validate_ingestion(paths: Sequence[Path]) -> list[LedgerEvent]:
    events = []
    identities: dict[str, str] = {}
    # Validate the entire supplied batch before the first write. Publication is
    # individually atomic and idempotent; a process crash can leave a partial batch.
    for path in paths:
        payload, _ = read_json(path)
        event = LedgerEvent.model_validate(payload)
        verify_sources(event)
        if event.event_id in identities and identities[event.event_id] != event.record_sha256:
            raise ValueError("conflicting event identity in ingestion batch")
        identities[event.event_id] = event.record_sha256
        events.append(event)
    return events


def run(*, live_root: Path, output: Path, capture: bool = False,
        fetch_ticks: bool = False, ingest_events: Sequence[Path] = ()) -> dict[str, Any]:
    output = research_output(output, live_root)
    if (capture or ingest_events) and live_root.resolve() != (ROOT / "Live").resolve():
        raise ValueError("production capture/ingestion requires repository Live")
    prepared = validate_ingestion(ingest_events)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    ingested = []
    for event in prepared:
        ingested.append(str(append_event(event, live_root=live_root)))
    observations, rejected, sources = collect_observations(live_root)
    target = target_cohort(observations)
    events, rejected_events = load_events(live_root)
    grouped: dict[str, list[LedgerEvent]] = defaultdict(list)
    for event in events:
        grouped[event.transition_id].append(event)
    transitions, projections = [], []
    for identity, group in sorted(grouped.items()):
        try:
            projection = project_transition(group)
            projections.append(projection)
            decision = projection.decision
            state = decision.pre_state
            transitions.append({"transition_id": identity, "bot_key": state.bot.key,
                                "incumbent_verdict": decision.incumbent.verdict.value,
                                "observed_action": projection.behavior.model_dump(mode="json") if projection.behavior else None,
                                "reward_net_usdt": str(projection.net_usdt) if projection.net_usdt is not None else None,
                                "training_eligible": projection.training_eligible,
                                "evidence_class": projection.evidence_class, "reasons": list(projection.reasons),
                                "risk_limits": decision.limits.model_dump(mode="json") if decision.limits else None,
                                "equity_usdt": str(state.equity_usdt) if state.equity_usdt is not None else None,
                                "peak_equity_usdt": str(state.peak_equity_usdt) if state.peak_equity_usdt is not None else None,
                                "event_ids": sorted(e.event_id for e in group),
                                "event_records": [e.model_dump(mode="json") for e in group]})
        except (ValueError, OSError, RuntimeError) as exc:
            rejected_events.append({"path": identity, "reason": str(exc)})
    action_report, _ = evaluate(projections)
    private_ledger = audit_ledger(live_root)
    ticks, tick_failure = None, None
    if fetch_ticks:
        try:
            ticks = fetch_exchange_info()
            atomic_create_json(output / "exchange_info_capture.json", ticks)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            tick_failure = f"{type(exc).__name__}: {exc}"
    tick_ref = ({"source": {"path": str(output / "exchange_info_capture.json"),
                            "sha256": file_hash(output / "exchange_info_capture.json")},
                 "received_at_utc": ticks["received_at_utc"], "ticks": ticks["ticks"]} if ticks else None)
    packets = evidence_packets(observations, transitions=transitions, tick_capture=tick_ref)
    packet_paths = [str(publish_packet(packet, live_root=live_root, now=started)) for packet in packets] if capture else []
    # These diagnostic artifacts cannot be discovered by pnl_history or load_events.
    report = {"schema_version": SCHEMA, "started_at_utc": started.isoformat(),
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "valid_observations": len(observations), "exact_deployments": len(packets),
              "rejected_observations": rejected, "rejected_events": rejected_events,
              "verdict_counts": dict(Counter(str(r["decision"].get("verdict") or "MISSING") for r in observations)),
              "target_bot_classes": dict(Counter(b["classification"] for b in target["bots"])),
              "target_origin_classes": dict(Counter(r["classification"] for r in target["origins"])),
              "positive_target_origins": target["positive_origins"], "negative_target_origins": 0,
              "target_experiment_status": "blocked_missing_negative_path_labels",
              "sar_events": len(events), "sar_transitions": len(transitions),
              "verified_action_examples": sum(p.training_eligible for p in projections),
              "ingested_event_paths": ingested, "captured_packet_paths": packet_paths,
              "tick_capture_failure": tick_failure,
              "post_target_horizon": "remaining time strictly before deployment + 7h",
              "post_target_risk_contract": "preserve invested capital; blocked pending verified capital/equity coverage and numerical liquidation buffer",
              "private_ledger_registrations": private_ledger["registrations"],
              "private_ledger_source_bound_facts": private_ledger["source_bound_facts"],
              "influence_enabled": False, "promotion_eligible": False, "fitted": False,
              "capture_scope": "existing original drawers/capture envelopes plus supplied canonical SAR events; no private exchange event collector was started"}
    atomic_create_json(output / "target_cohort.json", target)
    atomic_create_json(output / "action_cohort.json", {"transitions": transitions, "evaluation": action_report})
    atomic_create_json(output / "private_ledger_cohort.json", private_ledger)
    atomic_create_json(output / "evidence_packets.json", {"packets": packets})
    atomic_create_json(output / "required_event_schemas.json", contract_schema_catalog())
    code_paths = [Path(__file__), ROOT / "src/neutralgrid/live/decision/evidence_cohort.py",
                  ROOT / "src/neutralgrid/live/decision/state_action_reward.py",
                  ROOT / "src/neutralgrid/live/decision/action_challenger.py",
                  ROOT / "src/neutralgrid/live/decision/private_telemetry.py"]
    code_paths.append(ROOT / "src/neutralgrid/live/decision/exact_bot_ledger.py")
    atomic_create_json(output / "input_manifest.json", {"sources": sources,
                        "code": [{"path": str(p), "sha256": file_hash(p)} for p in code_paths]})
    atomic_create_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--fetch-exchange-info", action="store_true")
    parser.add_argument("--ingest-event", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    try:
        report = run(live_root=args.live_root, output=args.output_dir, capture=args.capture,
                     fetch_ticks=args.fetch_exchange_info, ingest_events=args.ingest_event)
        print(json.dumps({k: v for k, v in report.items() if k not in
                          {"captured_packet_paths", "rejected_observations", "rejected_events"}}, indent=2))
        # Success means the audit completed, not that a model is eligible.
        return 0 if report["valid_observations"] else 2
    except Exception as exc:
        logger.error("Evidence capture failed: %s", exc)
        print(json.dumps({"completed": False, "influence_enabled": False,
                          "error_type": type(exc).__name__, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
