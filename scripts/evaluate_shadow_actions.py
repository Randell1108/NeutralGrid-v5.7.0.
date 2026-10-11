"""Offline source audit, event ingestion and bounded action-association evaluation."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.core.constants import STATE_ACTION_REWARD_CONTRACT_VERSION, STATE_ACTION_REWARD_SCHEMA
from neutralgrid.live.decision.action_challenger import evaluate
from neutralgrid.live.decision.action_path_features import path_features
from neutralgrid.live.decision.pnl_forecast import PnlForecastConfig, build_forecast_examples
from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, LedgerEvent, SarError, append_event, atomic_create_json, file_hash,
    load_events, make_event, project_transition,
)

logger = logging.getLogger(__name__)


def research_output(path: Path, live_root: Path) -> Path:
    target = path.resolve()
    for protected in (live_root, ROOT / "artifacts", ROOT / "data", ROOT / "config",
                      ROOT / ".git", ROOT / "src", ROOT / "scripts"):
        protected = protected.resolve()
        if target == protected or protected in target.parents or target in protected.parents:
            raise SarError(f"research output overlaps protected tree: {protected}")
    if target.exists():
        raise SarError("research output must be a fresh directory")
    target.mkdir(parents=True, exist_ok=False)
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ingest-event", type=Path, help="Explicit producer-supplied lifecycle envelope; writes only Live hierarchy")
    parser.add_argument("--path-ablation", action="store_true", help="Prespecified common-cohort snapshot/summary/signature diagnostic")
    args = parser.parse_args(argv)
    if args.ingest_event is None and args.output_dir is None:
        parser.error("--output-dir is required for offline evaluation")
    if args.ingest_event is not None and (args.output_dir is not None or args.path_ablation):
        parser.error("ingestion and research evaluation are separate invocations")
    return args


def run_audit(*, live_root: Path, output_dir: Path, path_ablation: bool = False) -> dict[str, Any]:
    output = research_output(output_dir, live_root)
    events, rejected = load_events(live_root)
    grouped: dict[str, list[LedgerEvent]] = defaultdict(list)
    for event in events:
        grouped[event.transition_id].append(event)
    projections = []
    for transition, group in grouped.items():
        try:
            projections.append(project_transition(group))
        except (ValueError, SarError, OSError) as exc:
            rejected.append({"path": transition, "reason": str(exc)})
    observations = []
    inputs: list[dict[str, str]] = []
    for path in sorted(live_root.glob("????-??-??/*/pnl_history/*/observations/*.json")):
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise SarError("oversized PnL observation")
            digest = file_hash(path)
            observation = validate_pnl_observation(json.loads(path.read_text(encoding="utf-8")))
            if digest != file_hash(path):
                raise SarError("observation changed during audit")
            observations.append(observation)
            inputs.append({"path": str(path.resolve()), "sha256": digest})
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    for path in sorted(live_root.glob("????-??-??/*/sar/*/*.json")):
        try:
            inputs.append({"path": str(path.resolve()), "sha256": file_hash(path)})
        except OSError as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    code_paths = [ROOT / "src" / "neutralgrid" / "core" / "constants.py",
                  *[ROOT / "src" / "neutralgrid" / "live" / "decision" / name for name in
                    ("state_action_reward.py", "action_challenger.py", "action_path_features.py", "action_shadow_io.py")],
                  Path(__file__), ROOT / "backtest" / "backtest_realistic.py", ROOT / "backtest" / "btk_unified_runner.py"]
    manifest = {"inputs": inputs, "code": [{"path": str(p), "sha256": file_hash(p)} for p in code_paths]}
    projection_rows = [{"transition_id": p.transition_id, "evidence_class": p.evidence_class,
                        "behavior": p.behavior.model_dump(mode="json") if p.behavior is not None else None,
                        "net_usdt": str(p.net_usdt) if p.net_usdt is not None else None,
                        "normalized_return": str(p.normalized_return) if p.normalized_return is not None else None,
                        "endpoint": p.endpoint.isoformat() if p.endpoint is not None else None,
                        "training_eligible": p.training_eligible, "reasons": list(p.reasons)} for p in projections]
    # All three prespecified variants use the same path-eligible cohort when requested.
    common = projections
    if path_ablation:
        common = []
        for p in projections:
            try:
                path_features(p.decision.pre_state.price_path, asof=p.decision.pre_state.captured_at)
                common.append(p)
            except ValueError:
                pass
    evaluations = []
    variants = ("snapshot", "summaries", "signature") if path_ablation else ("snapshot",)
    for variant in variants:
        try:
            report, model = evaluate(common, variant=variant)
            if model is not None:
                atomic_create_json(output / variant / "model.json", model.model_dump(mode="json"))
            atomic_create_json(output / variant / "evaluation.json", report)
            evaluations.append(report)
        except Exception as exc:
            evaluations.append({"variant": variant, "evaluated": False, "influence_enabled": False,
                                "failure": type(exc).__name__, "reason": str(exc)})
    try:
        examples, _ = build_forecast_examples(observations, config=PnlForecastConfig(horizon_minutes=60, label_tolerance_minutes=5))
        uncontrolled_pairs = len(examples)
    except RuntimeError as exc:
        uncontrolled_pairs = None
        rejected.append({"path": str(live_root), "reason": str(exc)})
    report = {"schema_version": STATE_ACTION_REWARD_SCHEMA, "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
              "record_type": "isolated_shadow_action_audit", "created_at": datetime.now(timezone.utc).isoformat(),
              "influence_enabled": False, "promotion_eligible": False, "observed_policy_gain": None,
              "ledger_events": len(events), "transitions": len(projections),
              "verified_training_examples": sum(p.training_eligible for p in projections),
              "valid_pnl_observations": len(observations), "uncontrolled_one_hour_pairs": uncontrolled_pairs,
              "pnl_observations_without_private_completeness": sum(not o.get("features", {}).get("private_event_completeness") for o in observations),
              "rejected": rejected, "evidence_classes": dict(Counter(p.evidence_class for p in projections)),
              "ablation_enabled": path_ablation, "prespecified_variants": list(variants),
              "evaluations": evaluations, "replay_entrypoint": "backtest.btk_unified_runner.run_backtest",
              "replay_eligible": False, "off_policy_evaluation_available": False,
              "blockers": ["verified_action_outcomes_required", "behavior_selection_not_identifiable",
                           "existing_live_inventory_equity_order_replay_not_proven"]}
    atomic_create_json(output / "input_manifest.json", manifest)
    atomic_create_json(output / "transitions.json", {"records": projection_rows})
    atomic_create_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.ingest_event is not None:
            if args.live_root.resolve() != ROOT / "Live":
                raise SarError("production ingestion must use the repository Live directory")
            if args.ingest_event.stat().st_size > 4 * 1024 * 1024:
                raise SarError("oversized input event")
            raw = json.loads(args.ingest_event.read_text(encoding="utf-8"))
            event = (LedgerEvent.model_validate(raw) if "record_sha256" in raw else
                     make_event(bot=BotIdentity.model_validate(raw["bot"]), decision_id=raw["decision_id"],
                                kind=raw["kind"], occurred_at=datetime.fromisoformat(raw["occurred_at"].replace("Z", "+00:00")),
                                payload=raw["payload"]))
            path = append_event(event, live_root=args.live_root)
            print(json.dumps({"event_path": str(path), "influence_enabled": False}))
            return 0
        report = run_audit(live_root=args.live_root, output_dir=args.output_dir, path_ablation=args.path_ablation)
        print(json.dumps({k: v for k, v in report.items() if k not in {"evaluations", "rejected"}}, indent=2))
        return 0
    except Exception as exc:
        logger.error("Shadow operation failed: %s", exc)
        print(json.dumps({"completed": False, "influence_enabled": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
