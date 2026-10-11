"""Offline action-conditioned associations with support checks, never live policy."""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import Field, StrictBool, model_validator
from sklearn.linear_model import LogisticRegression, Ridge

from neutralgrid.core.constants import STATE_ACTION_REWARD_CONTRACT_VERSION, STATE_ACTION_REWARD_SCHEMA
from neutralgrid.live.decision.action_path_features import SUMMARY_COLUMNS, SIGNATURE_COLUMNS, path_features
from neutralgrid.live.decision.pnl_forecast import FEATURE_COLUMNS
from neutralgrid.live.decision.recommender import Verdict
from neutralgrid.live.decision.state_action_reward import (
    Action, ContractModel, Projection, RiskLimits, SarError, State,
    configuration_hash, feature_definition_hash, file_hash, fingerprint, recenter,
    shield_reasons,
)

logger = logging.getLogger(__name__)
ACTION_COLUMNS = ("capital_usdt", "position_quantity", "price_location", "action_adjust", "center_shift_fraction")
ALL_COLUMNS = FEATURE_COLUMNS + ACTION_COLUMNS + SUMMARY_COLUMNS + SIGNATURE_COLUMNS
Variant = Literal["snapshot", "summaries", "signature"]


class FitConfig(ContractModel):
    min_fit_samples: int = Field(default=30, strict=True, ge=2)
    min_validation_samples: int = Field(default=10, strict=True, ge=2)
    min_test_samples: int = Field(default=10, strict=True, ge=2)
    min_fit_bots: int = Field(default=4, strict=True, ge=2)
    min_validation_bots: int = Field(default=2, strict=True, ge=2)
    min_test_bots: int = Field(default=2, strict=True, ge=2)
    min_action_fit_samples: int = Field(default=5, strict=True, ge=2)


class ActionArtifact(ContractModel):
    schema_version: str = STATE_ACTION_REWARD_SCHEMA
    contract_version: str = STATE_ACTION_REWARD_CONTRACT_VERSION
    record_type: Literal["action_association_model"] = "action_association_model"
    variant: Variant
    columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    ridge_coef: tuple[float, ...]
    ridge_intercept: float
    logistic_coef: tuple[float, ...]
    logistic_intercept: float
    calibration_coef: float
    calibration_intercept: float
    interval_radius_usdt: float = Field(ge=0)
    support_rows: tuple[tuple[float, ...], ...] = Field(max_length=10000)
    support_actions: tuple[Literal["CONTINUE", "ADJUST"], ...]
    support_radii: dict[str, float]
    shift_ranges: dict[str, tuple[float, float]]
    config_hash: str
    feature_hash: str
    code_hash: str
    hmm_lineage: str
    holdout_passed: StrictBool
    influence_enabled: Literal[False] = False
    artifact_id: str

    @model_validator(mode="after")
    def integrity(self) -> ActionArtifact:
        n = len(self.columns)
        if (self.schema_version != STATE_ACTION_REWARD_SCHEMA or
                self.contract_version != STATE_ACTION_REWARD_CONTRACT_VERSION):
            raise ValueError("unsupported artifact contract")
        if not n or len(set(self.columns)) != n or set(self.columns) - set(ALL_COLUMNS):
            raise ValueError("unknown/duplicate model features")
        for values in (self.medians, self.means, self.scales, self.ridge_coef, self.logistic_coef):
            if len(values) != n:
                raise ValueError("artifact coefficient dimensions differ")
        if any(value <= 0 for value in self.scales):
            raise ValueError("artifact scale must be positive")
        if len(self.support_rows) != len(self.support_actions) or any(len(row) != n for row in self.support_rows):
            raise ValueError("support dimensions differ")
        if set(self.support_actions) != {"CONTINUE", "ADJUST"}:
            raise ValueError("both actions must have behavior support")
        if set(self.support_radii) != set(self.support_actions) or set(self.shift_ranges) != set(self.support_actions):
            raise ValueError("support bounds missing")
        if any(v < 0 for v in self.support_radii.values()) or any(a > b for a, b in self.shift_ranges.values()):
            raise ValueError("invalid support bounds")
        if self.artifact_id != fingerprint(self.model_dump(mode="json", exclude={"artifact_id"})):
            raise ValueError("artifact checksum mismatch")
        return self


def feature_row(state: State, action: Action, variant: Variant) -> dict[str, float]:
    row = {key: float(value) if value is not None else math.nan for key, value in state.features.items()}
    width = (state.grid_upper - state.grid_lower if state.grid_upper is not None and state.grid_lower is not None else None)
    center = ((state.grid_upper + state.grid_lower) / 2 if state.grid_upper is not None and state.grid_lower is not None else None)
    shift = ((action.upper + action.lower) / 2 - center if action.upper is not None and action.lower is not None and center is not None else Decimal(0))
    row.update({"capital_usdt": float(state.capital_usdt) if state.capital_usdt is not None else math.nan,
                "position_quantity": float(state.position_quantity) if state.position_quantity is not None else math.nan,
                "price_location": float((state.market_price - center) / width) if state.market_price is not None and center is not None and width else math.nan,
                "action_adjust": float(action.verdict == Verdict.ADJUST),
                "center_shift_fraction": float(shift / width) if width else math.nan})
    if variant != "snapshot":
        history = path_features(state.price_path, asof=state.captured_at)
        row.update({key: history[key] for key in SUMMARY_COLUMNS})
        if variant == "signature":
            row.update({key: history[key] for key in SIGNATURE_COLUMNS})
    return row


def temporal_split(rows: Sequence[Projection]) -> tuple[list[Projection], list[Projection], list[Projection]]:
    ordered = sorted(rows, key=lambda r: r.decision.pre_state.captured_at)
    if len(ordered) < 3:
        return [], [], []
    cut1 = ordered[int(len(ordered) * .6)].decision.pre_state.captured_at
    cut2 = ordered[int(len(ordered) * .8)].decision.pre_state.captured_at
    grouped: dict[tuple[str, str], list[Projection]] = defaultdict(list)
    for row in ordered:
        bot = row.decision.pre_state.bot
        grouped[bot.symbol, bot.strategy_id].append(row)
    fit: list[Projection] = []
    validation: list[Projection] = []
    test: list[Projection] = []
    for group in grouped.values():
        start = min(r.decision.pre_state.feature_window_start for r in group)
        end = max(r.endpoint or r.decision.pre_state.captured_at for r in group)
        if end < cut1:
            fit.extend(group)
        elif start >= cut1 + timedelta(minutes=65) and end < cut2:
            validation.extend(group)
        elif start >= cut2 + timedelta(minutes=65):
            test.extend(group)
    return fit, validation, test


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


def evaluate(rows: Sequence[Projection], *, variant: Variant = "snapshot",
             config: FitConfig | None = None) -> tuple[dict[str, Any], ActionArtifact | None]:
    cfg = config or FitConfig()
    unique: dict[str, Projection] = {}
    conflicting = False
    for row in rows:
        if row.transition_id in unique and unique[row.transition_id] != row:
            conflicting = True
        unique[row.transition_id] = row
    eligible = [r for r in unique.values() if r.training_eligible and not r.reasons and
                r.evidence_class == "verified_behavior" and r.decision.origin == "live" and
                r.net_usdt is not None and r.behavior is not None and
                r.behavior.verdict in {Verdict.CONTINUE, Verdict.ADJUST}]
    report: dict[str, Any] = {
        "schema_version": STATE_ACTION_REWARD_SCHEMA, "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
        "record_type": "action_association_evaluation", "variant": variant,
        "influence_enabled": False, "promotion_eligible": False,
        "evaluated": False, "config": cfg.model_dump(mode="json"),
        "input_count": len(rows), "eligible_count": len(eligible),
        "deduplicated_count": len(rows) - len(unique),
        "rejected_reasons": dict(Counter(reason for r in rows for reason in r.reasons)),
        "policy_value": None, "off_policy_evaluation_available": False,
        "blockers": ["unknown_behavior_propensities", "inventory_equity_order_replay_not_proven"],
    }
    if conflicting:
        report["blockers"].append("conflicting_transition_projection")
        return report, None
    if variant != "snapshot":
        common: list[Projection] = []
        for row in eligible:
            try:
                path_features(row.decision.pre_state.price_path, asof=row.decision.pre_state.captured_at)
                common.append(row)
            except ValueError:
                pass
        eligible = common
        report["path_eligible_count"] = len(common)
    if not eligible:
        report["blockers"].append("no_verified_action_reward_examples")
        return report, None
    if len(eligible) > 10000:
        report["blockers"].append("bounded_training_limit_exceeded")
        return report, None
    lineages = {r.decision.pre_state.hmm_lineage for r in eligible}
    profiles = {r.decision.config_hash for r in eligible}
    features = {r.decision.feature_hash for r in eligible}
    if len(lineages) != 1 or None in lineages or len(profiles) != 1 or features != {feature_definition_hash()}:
        report["blockers"].append("mixed_or_stale_lineage_profile_features")
        return report, None
    fit, validation, test = temporal_split(eligible)
    report["cohorts"] = {name: {"samples": len(group), "bots": len({r.decision.pre_state.bot.key for r in group}),
                                "actions": dict(Counter(r.behavior.verdict.value for r in group if r.behavior is not None)),
                                "transition_ids": [r.transition_id for r in group]}
                         for name, group in (("fit", fit), ("validation", validation), ("test", test))}
    for name, group, min_samples, min_bots in (
        ("fit", fit, cfg.min_fit_samples, cfg.min_fit_bots),
        ("validation", validation, cfg.min_validation_samples, cfg.min_validation_bots),
        ("test", test, cfg.min_test_samples, cfg.min_test_bots),
    ):
        if len(group) < min_samples or len({r.decision.pre_state.bot.key for r in group}) < min_bots:
            report["blockers"].append(f"insufficient_{name}_cohort")
        if name != "fit" and {r.behavior.verdict for r in group if r.behavior is not None} != {Verdict.CONTINUE, Verdict.ADJUST}:
            report["blockers"].append(f"{name}_action_missing")
    counts = Counter(r.behavior.verdict.value for r in fit if r.behavior is not None)
    if any(counts[a] < cfg.min_action_fit_samples for a in ("CONTINUE", "ADJUST")):
        report["blockers"].append("insufficient_action_support")
    if len(report["blockers"]) > 2:
        return report, None
    columns = FEATURE_COLUMNS + ACTION_COLUMNS
    if variant != "snapshot":
        columns += SUMMARY_COLUMNS
    if variant == "signature":
        columns += SIGNATURE_COLUMNS

    def raw(group: Sequence[Projection]) -> np.ndarray:
        values = []
        for row in group:
            assert row.behavior is not None
            features_row = feature_row(row.decision.pre_state, row.behavior, variant)
            values.append([features_row.get(c, math.nan) for c in columns])
        return np.asarray(values, dtype=float)

    raw_fit = raw(fit)
    keep = np.any(np.isfinite(raw_fit), axis=0)
    columns = tuple(c for c, kept in zip(columns, keep) if kept)
    raw_fit = raw_fit[:, keep]
    # All-NaN columns were removed before medians; no warning-driven fallback.
    medians = np.nanmedian(raw_fit, axis=0)
    filled = np.where(np.isfinite(raw_fit), raw_fit, medians)
    variable = np.std(filled, axis=0) > 0
    columns = tuple(c for c, kept in zip(columns, variable) if kept)
    medians = medians[variable]
    filled = filled[:, variable]
    if filled.shape[1] == 0:
        report["blockers"].append("no_variable_features")
        return report, None
    means, scales = np.mean(filled, axis=0), np.std(filled, axis=0)
    x_fit = (filled - means) / scales

    def transform(group: Sequence[Projection]) -> np.ndarray:
        matrix = raw(group)
        return (np.where(np.isfinite(matrix), matrix, medians) - means) / scales

    x_validation, x_test = transform(validation), transform(test)
    def targets(group: Sequence[Projection]) -> np.ndarray:
        values: list[float] = []
        for row in group:
            if row.net_usdt is None:
                raise SarError("eligible row has no observed reward")
            values.append(float(row.net_usdt))
        return np.asarray(values, dtype=float)

    y_fit, y_validation, y_test = targets(fit), targets(validation), targets(test)
    if not all(bool(np.all(np.isfinite(y))) for y in (y_fit, y_validation, y_test)):
        report["blockers"].append("numeric_range_exceeded")
        return report, None
    if len(set(y_fit > 0)) != 2 or len(set(y_validation > 0)) != 2:
        report["blockers"].append("direction_class_missing")
        return report, None
    ridge = Ridge(alpha=1.0).fit(x_fit, y_fit)
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(x_fit, y_fit > 0)
    logits_validation = np.asarray(classifier.decision_function(x_validation), dtype=float).reshape(-1, 1)
    calibration = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(logits_validation, y_validation > 0)
    prediction = np.asarray(ridge.predict(x_test), dtype=float)
    probability = np.asarray(calibration.predict_proba(np.asarray(classifier.decision_function(x_test)).reshape(-1, 1))[:, 1])
    interval = float(np.quantile(np.abs(y_validation - ridge.predict(x_validation)), .8, method="higher"))
    persistence = np.asarray([float(r.decision.pre_state.features.get("pnl_velocity_usdt_per_min") or 0) * 60 for r in test])
    mae = float(np.mean(np.abs(prediction - y_test)))
    zero_mae = float(np.mean(np.abs(y_test)))
    persistence_mae = float(np.mean(np.abs(persistence - y_test)))
    brier = float(np.mean((probability - (y_test > 0)) ** 2))
    baseline_brier = float(np.mean((float(np.mean(y_fit > 0)) - (y_test > 0)) ** 2))
    lower_tail = np.sort(y_test)[:max(1, math.ceil(len(y_test) * .05))]
    forecast_passed = mae < min(zero_mae, persistence_mae) and brier <= baseline_brier
    report.update({"evaluated": True, "forecast_passed": forecast_passed, "mae_usdt": mae,
                   "zero_mae_usdt": zero_mae, "persistence_mae_usdt": persistence_mae,
                   "brier": brier, "prior_brier": baseline_brier,
                   "observed_lower_tail_mean_usdt": float(np.mean(lower_tail)),
                   "policy_tail_change": None, "interval_coverage": float(np.mean(np.abs(y_test-prediction) <= interval)),
                   "feature_columns": list(columns)})
    actions = [r.behavior.verdict.value for r in fit if r.behavior is not None]
    radii: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for action in ("CONTINUE", "ADJUST"):
        selected = x_fit[np.asarray([a == action for a in actions])]
        nearest = []
        for i, value in enumerate(selected):
            distances = np.linalg.norm(selected - value, axis=1)
            distances[i] = np.inf
            nearest.append(float(np.min(distances)))
        radii[action] = max(1e-9, float(np.quantile(nearest, .95)))
        shifts = [feature_row(r.decision.pre_state, r.behavior, variant)["center_shift_fraction"]
                  for r in fit if r.behavior is not None and r.behavior.verdict.value == action]
        ranges[action] = min(shifts), max(shifts)
    supported: dict[str, float] = {}
    for name, group, matrix in (("validation", validation, x_validation), ("test", test, x_test)):
        checks = []
        for row, value in zip(group, matrix):
            assert row.behavior is not None
            action = row.behavior.verdict.value
            selected = x_fit[np.asarray([a == action for a in actions])]
            distance = float(np.min(np.linalg.norm(selected - value, axis=1)))
            shift = feature_row(row.decision.pre_state, row.behavior, variant)["center_shift_fraction"]
            lo, hi = ranges[action]
            checks.append(distance <= radii[action] and lo <= shift <= hi)
        supported[name] = float(np.mean(checks))
    passed = forecast_passed and all(value == 1.0 for value in supported.values())
    report.update({"behavior_support_fraction": supported, "holdout_passed": passed})
    if not all(value == 1.0 for value in supported.values()):
        report["blockers"].append("holdout_behavior_support_missing")
    body = dict(schema_version=STATE_ACTION_REWARD_SCHEMA, contract_version=STATE_ACTION_REWARD_CONTRACT_VERSION,
                record_type="action_association_model", variant=variant, columns=list(columns),
                medians=medians.tolist(), means=means.tolist(), scales=scales.tolist(),
                ridge_coef=np.asarray(ridge.coef_).tolist(), ridge_intercept=float(ridge.intercept_),
                logistic_coef=np.asarray(classifier.coef_)[0].tolist(), logistic_intercept=float(classifier.intercept_[0]),
                calibration_coef=float(calibration.coef_[0, 0]), calibration_intercept=float(calibration.intercept_[0]),
                interval_radius_usdt=interval, support_rows=x_fit.tolist(), support_actions=actions,
                support_radii=radii, shift_ranges={k: list(v) for k, v in ranges.items()},
                config_hash=next(iter(profiles)), feature_hash=next(iter(features)),
                code_hash=file_hash(Path(__file__)), hmm_lineage=next(iter(lineages)),
                holdout_passed=bool(passed), influence_enabled=False)
    artifact = ActionArtifact.model_validate(body | {"artifact_id": fingerprint(body)})
    report["artifact_id"] = artifact.artifact_id
    return report, artifact


def unavailable(incumbent: Verdict, *reasons: str) -> dict[str, Any]:
    return {"schema_version": STATE_ACTION_REWARD_SCHEMA, "contract_version": STATE_ACTION_REWARD_CONTRACT_VERSION,
            "record_type": "shadow_action_decision", "incumbent_verdict": incumbent.value,
            "verdict": incumbent.value, "available": False, "influence_enabled": False,
            "execution_eligible": False, "diagnostics": list(reasons), "observed_reward": None}


def predict_shadow(*, incumbent: Action, state: State, limits: RiskLimits | None,
                   quantum: Decimal | None, artifact: ActionArtifact | None,
                   now: datetime) -> dict[str, Any]:
    """Contain shadow errors at this boundary; all results are verdict-inert."""
    result = unavailable(incumbent.verdict)
    if incumbent.verdict == Verdict.END:
        return unavailable(Verdict.END, "incumbent_end_preserved")
    try:
        reasons = shield_reasons(state, limits, now=now)
        if artifact is None:
            reasons.append("model_unavailable")
        if reasons:
            result["diagnostics"] = reasons
            return result
        assert artifact is not None
        artifact = ActionArtifact.model_validate(artifact.model_dump(mode="json"))
        if (not artifact.holdout_passed or artifact.feature_hash != feature_definition_hash() or
                artifact.code_hash != file_hash(Path(__file__)) or
                artifact.config_hash != configuration_hash(limits) or artifact.hmm_lineage != state.hmm_lineage):
            return unavailable(incumbent.verdict, "model_profile_lineage_or_validation_mismatch")
        if quantum is None:
            return unavailable(incumbent.verdict, "price_quantum_unknown")
        for ref in state.sources:
            if file_hash(Path(ref.path)) != ref.sha256:
                raise SarError("inference source changed")
        if artifact.variant != "snapshot":
            for point in state.price_path:
                if file_hash(Path(point.source_path)) != point.source_sha256:
                    raise SarError("inference path source changed")
        adjust = recenter(state, quantum=quantum)
        predictions: dict[str, dict[str, float]] = {}
        for action in (Action(verdict=Verdict.CONTINUE), adjust):
            row = feature_row(state, action, artifact.variant)
            values = np.asarray([row.get(c, math.nan) for c in artifact.columns], dtype=float)
            if np.count_nonzero(np.isfinite(values)) < len(values) / 2:
                return unavailable(incumbent.verdict, "insufficient_observed_features")
            values = (np.where(np.isfinite(values), values, artifact.medians) - artifact.means) / artifact.scales
            selected = np.asarray([r for r, a in zip(artifact.support_rows, artifact.support_actions) if a == action.verdict.value])
            distance = float(np.min(np.linalg.norm(selected - values, axis=1)))
            lo, hi = artifact.shift_ranges[action.verdict.value]
            if distance > artifact.support_radii[action.verdict.value] or not lo <= row["center_shift_fraction"] <= hi:
                return unavailable(incumbent.verdict, "behavior_support_missing:" + action.verdict.value)
            predicted = float(np.dot(values, artifact.ridge_coef) + artifact.ridge_intercept)
            logit = float(np.dot(values, artifact.logistic_coef) + artifact.logistic_intercept)
            probability = float(_sigmoid(np.asarray([logit * artifact.calibration_coef + artifact.calibration_intercept]))[0])
            if not math.isfinite(predicted) or not math.isfinite(probability):
                raise SarError("non-finite model prediction")
            predictions[action.verdict.value] = {"expected_net_usdt": predicted, "positive_probability": probability}
        difference = predictions["ADJUST"]["expected_net_usdt"] - predictions["CONTINUE"]["expected_net_usdt"]
        if abs(difference) <= 2 * artifact.interval_radius_usdt:
            return unavailable(incumbent.verdict, "forecast_intervals_overlap")
        proposed = adjust if difference > 0 else Action(verdict=Verdict.CONTINUE)
        result.update({"available": True, "verdict": proposed.verdict.value,
                       "proposal": proposed.model_dump(mode="json"), "predictions": predictions,
                       "artifact_id": artifact.artifact_id,
                       "diagnostics": ["action_conditioned_association_only", "live_influence_disabled"]})
        return result
    except Exception as exc:
        logger.warning("Shadow action evaluation failed: %s", exc)
        return unavailable(incumbent.verdict, "shadow_failure:" + type(exc).__name__)
