"""Bounded reporting adapter for existing observations; unknown state stays unknown."""
from __future__ import annotations

import json
import math
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from neutralgrid.core.constants import STATE_ACTION_REWARD_CONTRACT_VERSION, STATE_ACTION_REWARD_SCHEMA
from neutralgrid.live.decision.action_challenger import ActionArtifact, predict_shadow, unavailable
from neutralgrid.live.decision.capital_policy import policy_summary
from neutralgrid.live.decision.pnl_forecast import FEATURE_COLUMNS, _feature_rows
from neutralgrid.live.decision.pnl_history import validate_pnl_observation
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import (
    Action, BotIdentity, Decision, RiskLimits, SourceRef, State, append_event,
    configuration_hash, feature_definition_hash, file_hash, make_event, utc,
)


def read_model(path: Path) -> ActionArtifact:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("model exceeds bounded JSON size")
    return ActionArtifact.model_validate_json(path.read_bytes())


def _mapping(value: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, dict) else {}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("explicit timestamp required")
    return utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _decision_time(row: Mapping[str, Any]) -> datetime:
    """Use the per-bot as-of time; legacy rows may only have the batch time."""
    evaluation = _mapping(row.get("evaluation"))
    return _timestamp(evaluation.get("evaluated_at_utc", row.get("ts")))


def _symbol_available(evaluation: Mapping[str, Any]) -> bool | None:
    """Translate the scanner's flags without treating missing data as success."""
    unavailable_flag = evaluation.get("symbol_unavailable")
    transient = evaluation.get("transient_fetch_error")
    if unavailable_flag is True:
        return False
    if unavailable_flag is False and transient is False:
        return True
    return None


def state_from_observations(observations: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> State:
    asof = _decision_time(row)
    past = [o for o in observations if _timestamp(o["evaluated_at_utc"]) <= asof]
    frame, _ = _feature_rows(past)
    if frame.empty:
        raise ValueError("no available PnL observation")
    # All observations supplied by the controller are already exact-bot scoped.
    latest = max(past, key=lambda o: _timestamp(o["telemetry_captured_at_utc"]))
    bot = BotIdentity(symbol=str(latest["symbol"]), strategy_id=str(latest["strategy_id"]),
                      deployed_at=_timestamp(latest["deploy_ts_utc"]))
    sources: dict[str, SourceRef] = {}
    for observation in past:
        identity = BotIdentity(symbol=str(observation["symbol"]), strategy_id=str(observation["strategy_id"]),
                               deployed_at=_timestamp(observation["deploy_ts_utc"]))
        if identity != bot:
            raise ValueError("feature history contains a different bot deployment")
        raw_ref = SourceRef(path=str(observation["source_snapshot_path"]),
                            sha256=str(observation["source_snapshot_sha256"]))
        if raw_ref.path in sources and sources[raw_ref.path] != raw_ref:
            raise ValueError("conflicting feature source identity")
        sources[raw_ref.path] = raw_ref
        record_path = Path(str(observation["_source_path"]))
        digest = file_hash(record_path)
        persisted = validate_pnl_observation(json.loads(record_path.read_text(encoding="utf-8")))
        if (persisted["record_integrity_sha256"] != observation["record_integrity_sha256"] or
                digest != file_hash(record_path)):
            raise ValueError("feature observation differs from its persisted source")
        sources[str(record_path)] = SourceRef(path=str(record_path), sha256=digest)
    if row.get("symbol") != bot.symbol or str(row.get("strategy_id")) != bot.strategy_id:
        raise ValueError("scanner/observation identity mismatch")
    if row.get("deploy_ts") is not None and _timestamp(row["deploy_ts"]) != bot.deployed_at:
        raise ValueError("scanner/observation deployment mismatch")
    numeric = frame.iloc[-1].to_dict()
    features = {c: float(numeric[c]) if c in numeric and math.isfinite(float(numeric[c])) else None for c in FEATURE_COLUMNS}
    grid, position = _mapping(latest["grid"]), _mapping(latest["position"])
    evaluation = _mapping(row.get("evaluation"))
    telemetry = _mapping(row.get("execution_telemetry"))
    risk = _mapping(telemetry.get("risk"))
    l2 = _mapping(evaluation.get("l2_risk"))
    capture = _timestamp(latest["telemetry_captured_at_utc"])
    return State(bot=bot, captured_at=capture, available_at=asof,
                 feature_window_start=min(_timestamp(o["telemetry_captured_at_utc"]) for o in past),
                 sources=tuple(sources.values()), grid_lower=grid.get("price_range_lower"),
                 grid_upper=grid.get("price_range_upper"), num_grids=grid.get("num_grids"),
                 # Row fields cannot certify exact-bot equity. The native
                 # adapter alone admits revalidated source-bound equity facts.
                 capital_usdt=grid.get("invested_margin_usdt"), equity_usdt=None,
                 unrealized_usdt=position.get("position_pnl_usdt"), position_quantity=position.get("size_base"),
                 entry_price=position.get("entry_price"), market_price=evaluation.get("price"),
                 market_captured_at=_timestamp(l2["captured_at_utc"]) if l2.get("captured_at_utc") else None,
                 symbol_available=_symbol_available(evaluation),
                 liquidation_buffer_pct=risk.get("liquidation_distance_to_mark_pct"),
                 peak_equity_usdt=None, hmm_lineage=evaluation.get("active_hmm_version"),
                 features=features)


def report_shadow_row(*, row: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
                      model_path: Path | None, limits_payload: Mapping[str, Any] | None,
                      live_root: Path | None = None) -> dict[str, Any]:
    try:
        incumbent = Verdict(str(row.get("verdict")))
    except (TypeError, ValueError):
        # An invalid proposal is neither CONTINUE nor an observed END action.
        return {"schema_version": STATE_ACTION_REWARD_SCHEMA,
                "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
                "record_type": "shadow_action_decision", "incumbent_verdict": None,
                "verdict": None, "available": False, "influence_enabled": False,
                "execution_eligible": False, "observed_reward": None,
                "diagnostics": ["invalid_incumbent_verdict"]}
    if incumbent == Verdict.END and live_root is None:
        return unavailable(incumbent, "incumbent_end_preserved")
    try:
        action = Action(verdict=incumbent,
                        lower=row.get("suggested_grid_lower") if incumbent == Verdict.ADJUST else None,
                        upper=row.get("suggested_grid_upper") if incumbent == Verdict.ADJUST else None)
        state = state_from_observations(observations, row)
        limits = RiskLimits.model_validate(limits_payload) if limits_payload is not None else None
        model = read_model(model_path) if model_path is not None and incumbent != Verdict.END else None
        if model is not None and model.variant != "snapshot":
            raise ValueError("path ablation cannot enter the controller")
        # Precision is not inferred from the decimal spelling of a market price.
        precision = _mapping(row.get("evaluation")).get("price_quantum")
        quantum = Decimal(str(precision)) if precision is not None else None
        result = predict_shadow(incumbent=action, state=state, limits=limits, quantum=quantum,
                                artifact=model, now=state.available_at)
        result["capital_preservation_policy"] = policy_summary()
        if live_root is not None:
            proposal = Action.model_validate(result["proposal"]) if result.get("available") else action
            decision = Decision(decided_at=state.available_at, pre_state=state,
                                incumbent=action, shadow=proposal, limits=limits, origin="live",
                                config_hash=configuration_hash(limits), feature_hash=feature_definition_hash(),
                                diagnostics=tuple(result["diagnostics"]))
            event = make_event(bot=state.bot, decision_id=f"scanner:{row['ts']}", kind="decision",
                               occurred_at=decision.decided_at, payload=decision.model_dump(mode="json"))
            result["decision_event_path"] = str(append_event(event, live_root=live_root))
        return result
    except Exception as exc:
        return unavailable(incumbent, "shadow_adapter_failure:" + type(exc).__name__)
