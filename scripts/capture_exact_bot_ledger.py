"""Initialize exact-bot ledgers and ingest source-bound private facts; never trade."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.live.decision.evidence_cohort import read_json
from neutralgrid.live.decision.exact_bot_ledger import (
    CapitalProtectionPolicy, PrivateFact, append_fact, audit_ledger, register_bot, verify_fact,
)
from neutralgrid.live.decision.state_action_reward import BotIdentity, SourceRef, atomic_create_json

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-capture-report", type=Path)
    parser.add_argument("--ingest", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    live_root = ROOT / "Live"
    try:
        output = args.output_dir.resolve()
        if (ROOT / "reports").resolve() not in output.parents or output.exists():
            raise ValueError("output must be a fresh directory under reports")
        facts = [PrivateFact.model_validate(read_json(p)[0]) for p in args.ingest]
        for fact in facts:
            verify_fact(fact, live_root=live_root)
        packets = []
        if args.bootstrap_capture_report:
            capture, _ = read_json(args.bootstrap_capture_report)
            paths = capture.get("captured_packet_paths")
            if not isinstance(paths, list) or not paths or len(paths) > 10000:
                raise ValueError("bootstrap requires a bounded list of captured evidence indexes")
            for name in paths:
                path = Path(name).resolve()
                if live_root.resolve() not in path.parents or path.parent.parent.name != "evidence_cohort":
                    raise ValueError("bootstrap source must be a committed Live evidence index")
                packet, digest = read_json(path)
                bot = BotIdentity(symbol=packet["symbol"], strategy_id=packet["strategy_id"], deployed_at=packet["deploy_ts_utc"])
                packets.append((bot, SourceRef(path=str(path), sha256=digest)))
        output.mkdir(parents=True, exist_ok=False)
        now = datetime.now(timezone.utc)
        registrations = [str(register_bot(bot, source=source, live_root=live_root, now=now)) for bot, source in packets]
        committed = [str(append_fact(fact, live_root=live_root, now=now)) for fact in facts]
        audit = audit_ledger(live_root)
        atomic_create_json(output / "fact_schema.json", PrivateFact.model_json_schema())
        atomic_create_json(output / "capital_policy.json", CapitalProtectionPolicy().model_dump(mode="json"))
        atomic_create_json(output / "ledger_audit.json", audit)
        result = {"registrations": registrations, "committed_facts": committed,
                  "source_bound_facts": audit["source_bound_facts"], "rejected": audit["rejected"],
                  "training_eligible": False, "influence_enabled": False,
                  "account_ownership_created_from_assumptions": False}
        atomic_create_json(output / "report.json", result)
        print(json.dumps({"registered_deployments": len(registrations), "committed_facts": len(committed),
                          "source_bound_facts": audit["source_bound_facts"], "influence_enabled": False}))
        return 0 if not audit["rejected"] else 2
    except Exception as exc:
        logger.error("Exact-bot ledger operation failed: %s", exc)
        print(json.dumps({"completed": False, "influence_enabled": False, "error_type": type(exc).__name__, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
