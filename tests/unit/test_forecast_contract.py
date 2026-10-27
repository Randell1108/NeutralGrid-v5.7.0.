"""Forecast semantics, causal timing and coverage remain distinct from admission."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from neutralgrid.live.decision.forecast_contract import (
    ForecastContract, forecast_from_reported_pnl_path, verify_forecast_sources,
)
from neutralgrid.live.decision.state_action_reward import file_hash


T0 = datetime(2026, 10, 22, tzinfo=timezone.utc)


def evidence(tmp_path: Path, name: str, minute: int, *, available: int | None = None) -> dict:
    path = tmp_path / f"{name}.json"
    path.write_text('{"test_fixture":true}', encoding="utf-8")
    return {"record_id": name, "occurred_at": T0 + timedelta(minutes=minute),
            "available_at": T0 + timedelta(minutes=minute if available is None else available),
            "source": {"path": str(path.resolve()), "sha256": file_hash(path)}}


def payload(tmp_path: Path, family: str = "first_passage_pnl") -> dict:
    feature = evidence(tmp_path, "feature", 60)
    result = {
        "contract_version": "test-contract-v1", "family": family,
        "bot": {"symbol": "BTCUSDT", "strategy_id": "123", "deployed_at": T0},
        "account_scope": "test_account", "origin_at": T0 + timedelta(minutes=60),
        "target_end_at": T0 + timedelta(minutes=420), "units": "binary",
        "capital_basis": "invested_margin_usdt", "invested_margin_usdt": "100",
        "target_pct": "3", "prior_target_state": "unknown", "feature_sources": [feature],
        "label": {"status": "path_unobserved"}, "feature_definition_sha256": "a" * 64,
        "dataset_manifest": feature["source"],
    }
    if family != "first_passage_pnl":
        result.update(target_end_at=T0 + timedelta(minutes=120), horizon_minutes=60,
                      units="USDT", capital_basis="not_normalized",
                      invested_margin_usdt=None, target_pct=None, prior_target_state="not_applicable")
    if family == "volatility":
        result.update(units="fractional_log_return_variance", capital_basis="not_applicable",
                      price_source="mark", sampling_interval_seconds=60,
                      variance_aggregation="cumulative")
    return result


def outcome(tmp_path: Path, minute: int, *, value: str = "1", status: str = "observed_hit") -> dict:
    return {"status": status, "value": value, "observed_at": T0 + timedelta(minutes=minute),
            "sources": [evidence(tmp_path, "outcome", minute)]}


def coverage(tmp_path: Path, start: int, end: int, scope: str = "pnl_path") -> dict:
    return {"start_at": T0 + timedelta(minutes=start), "end_at": T0 + timedelta(minutes=end),
            "basis": "full_interval", "scope": scope,
            "verification": evidence(tmp_path, "coverage", end, available=end + 1)}


def test_deployment_deadline_and_remaining_time(tmp_path):
    raw = payload(tmp_path)
    model = ForecastContract.model_validate(raw)
    assert model.remaining_seconds == 360 * 60
    raw["origin_at"] = T0 + timedelta(minutes=300)
    assert ForecastContract.model_validate(raw).remaining_seconds == 120 * 60
    assert model.training_eligible is model.influence_enabled is model.promotion_enabled is False
    assert len(model.record_sha256) == 64


@pytest.mark.parametrize("field,value", [
    ("target_end_at", T0 + timedelta(minutes=480)), ("target_pct", "4"),
    ("horizon_minutes", 420), ("endpoint_tolerance_seconds", 1),
    ("units", "USDT"), ("capital_basis", "not_normalized"),
    ("invested_margin_usdt", None), ("prior_target_state", "already_hit"),
    ("origin_at", T0 + timedelta(minutes=420)), ("origin_at", T0 - timedelta(seconds=1)),
])
def test_first_passage_cannot_silently_change_semantics(tmp_path, field, value):
    raw = payload(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


def test_deadline_hit_is_admitted_as_evidence_but_late_hit_is_not(tmp_path):
    raw = payload(tmp_path)
    raw["label"] = outcome(tmp_path, 420)
    assert ForecastContract.model_validate(raw).label.value == 1
    raw["label"] = outcome(tmp_path, 421)
    with pytest.raises(ValueError, match="forward interval"):
        ForecastContract.model_validate(raw)


@pytest.mark.parametrize("status,value", [
    ("path_unobserved", "0"), ("pending", "1"), ("observed_hit", "0"),
    ("verified_nonhit", "1"), ("observed_endpoint", "-1"),
])
def test_incompatible_first_passage_label_is_rejected(tmp_path, status, value):
    raw = payload(tmp_path)
    raw["label"] = outcome(tmp_path, 420, status=status, value=value)
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


def test_sparse_points_and_duplicates_never_prove_nonhit(tmp_path):
    raw = payload(tmp_path)
    raw["label"] = outcome(tmp_path, 420, value="0", status="verified_nonhit")
    raw["label"]["coverage"] = {
        "start_at": T0, "end_at": T0 + timedelta(minutes=420),
        "basis": "snapshots", "scope": "pnl_path",
    }
    with pytest.raises(ValueError, match="sparse or partial"):
        ForecastContract.model_validate(raw)
    raw["label"]["sources"] *= 2
    with pytest.raises(ValueError, match="duplicate evidence"):
        ForecastContract.model_validate(raw)


def test_coverage_attestation_is_structural_and_never_training_admission(tmp_path):
    raw = payload(tmp_path)
    raw["label"] = outcome(tmp_path, 420, value="0", status="verified_nonhit")
    raw["label"]["coverage"] = coverage(tmp_path, 0, 420)
    model = ForecastContract.model_validate(raw)
    verify_forecast_sources(model)
    assert model.label.status == "verified_nonhit"
    assert model.training_eligible is False  # A fixture hash cannot prove native coverage.


@pytest.mark.parametrize("edit", ["gap", "partial", "no_verifier", "early_verifier"])
def test_invalid_coverage_is_rejected(tmp_path, edit):
    raw = payload(tmp_path)
    raw["label"] = outcome(tmp_path, 420, value="0", status="verified_nonhit")
    declared = coverage(tmp_path, 0, 420)
    if edit == "gap":
        declared["gap_count"] = 1
    elif edit == "partial":
        declared["start_at"] = T0 + timedelta(minutes=1)
    elif edit == "no_verifier":
        declared["verification"] = None
    else:
        declared["verification"] = evidence(tmp_path, "early", 419)
    raw["label"]["coverage"] = declared
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


@pytest.mark.parametrize("minute,available", [(61, 61), (59, 61), (59, 58)])
def test_event_and_availability_cutoffs_both_apply(tmp_path, minute, available):
    raw = payload(tmp_path)
    raw["feature_sources"] = [evidence(tmp_path, "late", minute, available=available)]
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


def test_verified_prior_nonhit_requires_causal_original(tmp_path):
    raw = payload(tmp_path)
    raw["prior_target_state"] = "verified_not_hit"
    with pytest.raises(ValueError, match="prior nonhit"):
        ForecastContract.model_validate(raw)
    raw["prior_target_verification"] = evidence(tmp_path, "prior", 60)
    assert ForecastContract.model_validate(raw).prior_target_state == "verified_not_hit"
    raw["prior_target_verification"] = evidence(tmp_path, "prior", 60, available=61)
    with pytest.raises(ValueError, match="known at origin"):
        ForecastContract.model_validate(raw)


@pytest.mark.parametrize("family", ["fixed_horizon_pnl", "volatility", "action_reward"])
def test_other_targets_keep_their_fixed_horizons(tmp_path, family):
    raw = payload(tmp_path, family)
    assert ForecastContract.model_validate(raw).remaining_seconds == 3600
    raw["target_end_at"] = T0 + timedelta(minutes=420)
    with pytest.raises(ValueError, match="fixed horizon"):
        ForecastContract.model_validate(raw)


def test_pnl_endpoint_tolerance_does_not_change_first_passage(tmp_path):
    raw = payload(tmp_path, "fixed_horizon_pnl")
    raw["endpoint_tolerance_seconds"] = 300
    raw["label"] = outcome(tmp_path, 125, value="-4", status="observed_endpoint")
    assert ForecastContract.model_validate(raw).label.value == -4
    raw["label"] = outcome(tmp_path, 119, value="-4", status="observed_endpoint")
    with pytest.raises(ValueError, match="precedes the target horizon"):
        ForecastContract.model_validate(raw)


@pytest.mark.parametrize("field,value", [
    ("units", "USDT"), ("price_source", None), ("sampling_interval_seconds", None),
    ("variance_aggregation", None), ("capital_basis", "not_normalized"),
])
def test_volatility_requires_explicit_measurement(tmp_path, field, value):
    raw = payload(tmp_path, "volatility")
    raw[field] = value
    with pytest.raises(ValueError, match="volatility requires"):
        ForecastContract.model_validate(raw)


def test_negative_variance_rejected(tmp_path):
    raw = payload(tmp_path, "volatility")
    raw["label"] = outcome(tmp_path, 120, value="-1", status="observed_endpoint")
    with pytest.raises(ValueError, match="variance cannot be negative"):
        ForecastContract.model_validate(raw)


def test_verdict_is_not_observed_action_or_reward(tmp_path):
    raw = payload(tmp_path, "action_reward")
    raw["proposed_verdict"] = "CONTINUE"
    assert ForecastContract.model_validate(raw).observed_action is None
    raw["label"] = outcome(tmp_path, 120, value="1", status="observed_endpoint")
    with pytest.raises(ValueError, match="observed behavior"):
        ForecastContract.model_validate(raw)
    raw["observed_action"] = "CONTINUE"
    raw["action_evidence"] = [evidence(tmp_path, "maintenance", 120)]
    raw["label"]["coverage"] = coverage(tmp_path, 60, 120, "accounting_and_interventions")
    model = ForecastContract.model_validate(raw)
    assert model.label.value == 1 and model.training_eligible is False


def test_hash_conflict_and_mutation_rejected(tmp_path):
    raw = payload(tmp_path)
    model = ForecastContract.model_validate(raw)
    verify_forecast_sources(model)
    Path(model.dataset_manifest.path).write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_forecast_sources(model)
    raw["dataset_manifest"] = {**raw["dataset_manifest"], "sha256": "b" * 64}
    with pytest.raises(ValueError, match="conflicting hashes"):
        verify_forecast_sources(ForecastContract.model_validate(raw))


@pytest.mark.parametrize("flag", ["training_eligible", "promotion_enabled", "influence_enabled"])
def test_enabling_flags_rejected(tmp_path, flag):
    raw = payload(tmp_path)
    raw[flag] = True
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


@pytest.mark.parametrize("field,value", [
    ("invested_margin_usdt", True), ("invested_margin_usdt", "NaN"),
    ("target_pct", "Infinity"), ("origin_at", datetime(2026, 10, 22, 1)),
])
def test_malformed_numeric_and_time_inputs_rejected(tmp_path, field, value):
    raw = payload(tmp_path)
    raw[field] = value
    with pytest.raises(ValueError):
        ForecastContract.model_validate(raw)


def captured_path(tmp_path: Path, points: list[tuple[int, str, str]]) -> tuple[Path, list[Path]]:
    from neutralgrid.live.decision.authenticated_evidence import capture_pnl_path
    from neutralgrid.live.decision.pnl_history import build_pnl_observation
    from neutralgrid.live.decision.state_action_reward import BotIdentity

    live = tmp_path / "Live"
    symbol_dir = live / "2026-10-22" / "BTCUSDT"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    originals = []
    bot = BotIdentity(symbol="BTCUSDT", strategy_id="123", deployed_at=T0)
    for minute, pnl, margin in points:
        at = (T0 + timedelta(minutes=minute)).isoformat()
        native = symbol_dir / f"native_{minute}.txt"
        native.write_text("\n".join([
            "BTCUSDT", "Time Created", "2026-10-21 19:00:00", "Total Profit (USDT)",
            f"{pnl} (0.00%)", "Positions", "no position", "Pending Order", "Grid Details",
            "Invested Margin", margin, "Advanced (Optional)", "Strategy Number", "123",
        ]), encoding="utf-8")
        native.with_suffix(".json").write_text(json.dumps({"symbol": bot.symbol, "strategy_id": bot.strategy_id,
            "raw_text_sha256": file_hash(native), "captured_at_utc": at}), encoding="utf-8")
        row = build_pnl_observation({"symbol": bot.symbol, "strategy_id": bot.strategy_id, "ts": at,
            "execution_telemetry": {"captured_at": at, "pnl": {"total_profit_usdt": float(pnl)},
                                    "grid": {"invested_margin_usdt": float(margin)}}},
            deploy_ts=T0, source_cycle_manifest="fixture-only", source_snapshot_path=str(native),
            source_snapshot_sha256=file_hash(native))
        observation = symbol_dir / f"observation_{minute}.json"
        observation.write_text(json.dumps(row), encoding="utf-8")
        originals.append(observation)
    path = capture_pnl_path(originals, bot=bot, live_root=live, now=T0 + timedelta(hours=8))
    return path, originals


def test_real_capture_path_exports_inert_common_contract_with_all_originals(tmp_path):
    path, originals = captured_path(tmp_path, [(60, "1.00", "100.00"), (120, "2.00", "100.00")])
    result = forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live")
    assert result.origin_at == T0 + timedelta(minutes=120)
    assert result.remaining_seconds == 300 * 60
    assert len(result.feature_sources) == 6  # Two originals, each with drawer+capture envelope.
    assert result.dataset_manifest.path == str(path.resolve())
    assert result.account_scope_basis == "unverified"
    assert result.measurement_basis == "reported_total_pnl_not_verified_equity"
    assert result.prior_target_state == "unknown" and result.label.value is None
    assert result.label.status == "path_unobserved" and not result.training_eligible
    verify_forecast_sources(result)
    Path(json.loads(originals[0].read_text())["source_snapshot_path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_forecast_sources(result)


def test_future_hit_does_not_supply_conditional_label_when_prior_path_unknown(tmp_path):
    path, originals = captured_path(tmp_path, [(60, "1.00", "100.00"), (400, "3.00", "100.00"),
                                              (420, "-1.00", "100.00")])
    result = forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live")
    assert result.origin_at == T0 + timedelta(minutes=60)
    assert result.label.value is None and result.prior_target_state == "unknown"
    assert all(source.occurred_at <= result.origin_at for source in result.feature_sources)
    hit_id = json.loads(originals[1].read_text())["observation_id"]
    with pytest.raises(ValueError, match="no pre-deadline origin"):
        forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live", origin_observation_id=hit_id)


def test_explicit_origin_uses_only_information_already_available(tmp_path):
    path, originals = captured_path(tmp_path, [(60, "1.00", "100.00"), (120, "2.00", "100.00")])
    first_id = json.loads(originals[0].read_text())["observation_id"]
    result = forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live", origin_observation_id=first_id)
    assert result.origin_at == T0 + timedelta(minutes=60) and len(result.feature_sources) == 3


@pytest.mark.parametrize("points", [
    [(60, "3.00", "100.00")], [(420, "1.00", "100.00")],
    [(60, "1.00", "100.00"), (120, "2.00", "200.00")],
])
def test_unusable_origins_and_changing_capital_cannot_export_a_contract(tmp_path, points):
    path, _ = captured_path(tmp_path, points)
    with pytest.raises(ValueError):
        forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live")


def test_capture_point_deduplication_never_adds_feature_support(tmp_path):
    path, _ = captured_path(tmp_path, [(60, "1.00", "100.00")])
    raw = json.loads(path.read_text())
    raw["points"] *= 2
    path.write_text(json.dumps(raw))
    result = forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live")
    assert len(result.feature_sources) == 3


@pytest.mark.parametrize("change", ["amount", "time", "source", "deadline", "measurement", "early_availability"])
def test_path_reparse_rejects_tampered_measurements_metadata_and_sources(tmp_path, change):
    path, _ = captured_path(tmp_path, [(60, "1.00", "100.00")])
    raw = json.loads(path.read_text())
    if change == "amount":
        raw["points"][0]["reported_total_pnl_usdt"] = "5"
    elif change == "time":
        raw["points"][0]["available_at"] = (T0 + timedelta(minutes=61)).isoformat()
    elif change == "source":
        raw["points"][0]["original_sources"] = []
    elif change == "deadline":
        raw["deadline"] = (T0 + timedelta(hours=8)).isoformat()
    elif change == "measurement":
        raw["measurement"] = "verified_equity"
    else:
        raw["points"][0]["available_at"] = T0.isoformat()
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        forecast_from_reported_pnl_path(path, live_root=tmp_path / "Live")
