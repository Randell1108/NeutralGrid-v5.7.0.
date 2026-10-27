"""Convert reviewed original native evidence to SAR without routing actions."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.live.decision.evidence_cohort import collect_observations
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact, audit_ledger
from neutralgrid.live.decision.native_sar import (
    NativeRecord, ledger_outcome, native_approval, native_decision, native_execution,
)
from neutralgrid.live.decision.state_action_reward import (
    SourceRef, append_event, atomic_create_json, file_hash, project_transition, verify_sources,
)

logger = logging.getLogger(__name__)


def _record(path: Path, line: int | None = None) -> NativeRecord:
    path = path.resolve()
    return NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)), line)


def _observations(record: NativeRecord, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = record.read()
    matching = [o for o in observations if o["symbol"] == raw.get("symbol") and str(o["strategy_id"]) == str(raw.get("strategy_id"))]
    identities = {o["bot_identity"] for o in matching}
    if len(identities) != 1:
        raise ValueError("native source must resolve to exactly one verified deployment")
    return matching


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--decision-source", type=Path, required=True)
    parser.add_argument("--decision-line", type=int)
    parser.add_argument("--intent-source", type=Path)
    parser.add_argument("--approval-source", type=Path)
    parser.add_argument("--receipt-source", type=Path)
    parser.add_argument("--pre-source", type=Path)
    parser.add_argument("--pre-line", type=int)
    parser.add_argument("--post-source", type=Path)
    parser.add_argument("--post-line", type=int)
    parser.add_argument("--exchange-info", type=Path)
    parser.add_argument("--outcome-source", type=Path)
    parser.add_argument("--outcome-line", type=int)
    parser.add_argument("--account-scope")
    parser.add_argument("--capture", action="store_true", help="Append verified events to repository Live; default is report only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    published: list[str] = []
    output: Path | None = None
    try:
        candidate = Path(args.output_dir).resolve()
        if (ROOT / "reports").resolve() not in candidate.parents or candidate.exists():
            raise ValueError("output must be a fresh directory under reports")
        if args.capture and args.live_root.resolve() != (ROOT / "Live").resolve():
            raise ValueError("capture is restricted to repository Live")
        if bool(args.approval_source) != bool(args.intent_source):
            raise ValueError("approval conversion needs both original approval and intent")
        if args.receipt_source and not all((args.approval_source, args.pre_source, args.post_source, args.exchange_info)):
            raise ValueError("receipt needs approval, pre/post original states and original exchangeInfo capture")
        if args.outcome_source and not args.account_scope:
            raise ValueError("outcome conversion requires explicit account scope")
        candidate.mkdir(parents=True, exist_ok=False)
        output = candidate
        observations, rejections, _ = collect_observations(args.live_root)
        decision_record = _record(args.decision_source, args.decision_line)
        decision = native_decision(decision_record, _observations(decision_record, observations), live_root=args.live_root,
                                   intent=_record(args.intent_source) if args.intent_source else None)
        events = [decision]
        if args.approval_source:
            approved = native_approval(_record(args.approval_source), decision, intent=_record(args.intent_source))
            events.append(approved)
            if args.receipt_source:
                before, after = _record(args.pre_source, args.pre_line), _record(args.post_source, args.post_line)
                events.append(native_execution(_record(args.receipt_source), decision, approved,
                    pre_record=before, post_record=after, pre_observations=_observations(before, observations),
                    post_observations=_observations(after, observations), live_root=args.live_root,
                    exchange_info=_record(args.exchange_info)))
        ledger_rejections: list[dict[str, Any]] = []
        if args.outcome_source:
            ledger = audit_ledger(args.live_root)
            ledger_rejections = ledger["rejected"]
            if ledger_rejections:
                raise ValueError("private ledger contains rejected evidence; inspect ledger audit before conversion")
            facts = [PrivateFact.model_validate(f) for f in ledger["facts"]
                     if f["account_scope"] == args.account_scope and f["bot"] == decision.bot.model_dump(mode="json")]
            outcome_record = _record(args.outcome_source, args.outcome_line)
            events.append(ledger_outcome(outcome_record, _observations(outcome_record, observations), decision,
                facts=facts, account_scope=args.account_scope, live_root=args.live_root, ingested_at=datetime.now(timezone.utc)))
        for event in events:
            verify_sources(event)
        projection = project_transition(events)
        atomic_create_json(output / "events.json", {"events": [e.model_dump(mode="json") for e in events]})
        # Preserve every original selector, including the intent used to bind approval.
        inputs = []
        for key in ("decision_source", "intent_source", "approval_source", "receipt_source", "pre_source", "post_source", "exchange_info", "outcome_source"):
            if getattr(args, key) is not None:
                ref = _record(getattr(args, key))
                inputs.append({"role": key, **ref.source.model_dump(mode="json")})
        atomic_create_json(output / "inputs.json", {"sources": inputs,
            "selectors": {k: v for k, v in vars(args).items() if k.endswith("_line")}})
        if args.capture:
            lock = args.live_root / ".sar-locks" / (decision.transition_id + ".lock")
            try:
                atomic_create_json(lock, {"lock": "native_sar"})
            except FileExistsError:
                pass
            for event in events:
                published.append(str(append_event(event, live_root=args.live_root)))
        result = {"completed": True, "bot": decision.bot.model_dump(mode="json"), "events": len(events),
            "published_paths": published, "source_rejections": rejections, "ledger_rejections": ledger_rejections,
            "evidence_class": projection.evidence_class,
            "observed_action": projection.behavior.model_dump(mode="json") if projection.behavior else None,
            "net_reward_usdt": str(projection.net_usdt) if projection.net_usdt is not None else None,
            "projection_training_eligible": projection.training_eligible, "blockers": list(projection.reasons),
            "promotion_eligible": False, "influence_enabled": False}
        atomic_create_json(output / "report.json", result)
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        result = {"completed": False, "error_type": type(exc).__name__, "error": str(exc),
                  "published_paths": published, "influence_enabled": False, "promotion_eligible": False}
        if output is not None:
            atomic_create_json(output / "failure.json", result)
        logger.error("Native SAR conversion failed: %s", exc)
        print(json.dumps(result))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
