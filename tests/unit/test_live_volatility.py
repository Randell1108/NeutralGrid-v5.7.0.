from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

import scripts.train_live_volatility_forecaster as training_script

from neutralgrid.live.decision.volatility import (
    EXPECTED_HAC_LAGS,
    VolatilityError,
    apply_log_interval,
    build_rv_examples,
    calibrate_log_interval,
    contract_digest,
    hac_lag,
    holm_adjust,
    interval_metrics,
    latest_rv_snapshot,
    load_volatility_contract,
    newey_west_mean_test,
    qlike_loss,
    validate_price_frame,
)
from neutralgrid.live.decision.volatility_forecast import (
    VOLATILITY_ARTIFACT_SCHEMA,
    _global_splits,
    _load_artifact,
    select_runtime_horizon,
    train_evaluate_shadow_volatility,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "live_volatility_forecast_v1.json"
UTC = timezone.utc


def _training_cycle_manifest(
    path: Path,
    *,
    page_identity: str = "Authenticated Binance Futures Grid; UM Grid; Running",
    source_url: str = "https://www.binance.bh/en/trading-bots/futures/grid/BTCUSDT",
) -> Path:
    payload = {
        "schema_version": "neutralgrid_private_telemetry_cycle_v2",
        "status": "complete",
        "source": "chrome_plugin",
        "page_identity": page_identity,
        "source_url": source_url,
        "cycle_completed_at_utc": datetime(2026, 8, 23, tzinfo=UTC).isoformat(),
        "active_bot_count": 1,
        "working_row_count": 1,
        "files": [{"symbol": "BTCUSDT", "strategy_id": "413000001"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_training_roster_accepts_descriptive_identity_and_trusted_binance_url(
    tmp_path: Path,
) -> None:
    path = _training_cycle_manifest(tmp_path / "cycle.json")

    assert training_script._load_roster(path) == {"BTCUSDT": "413000001"}


@pytest.mark.parametrize(
    ("page_identity", "source_url", "message"),
    [
        ("", "https://www.binance.com/grid", "page identity is missing"),
        ("Authenticated UM Grid", "http://www.binance.com/grid", "not trusted Binance HTTPS"),
        (
            "Authenticated UM Grid",
            "https://binance.com.evil.example/grid",
            "not trusted Binance HTTPS",
        ),
    ],
)
def test_training_roster_rejects_missing_identity_or_untrusted_source(
    tmp_path: Path,
    page_identity: str,
    source_url: str,
    message: str,
) -> None:
    path = _training_cycle_manifest(
        tmp_path / "cycle.json",
        page_identity=page_identity,
        source_url=source_url,
    )

    with pytest.raises(training_script.VolatilityTrainingError, match=message):
        training_script._load_roster(path)


def _minute_frame(
    *,
    days: int = 5,
    start: str = "2026-01-01T00:00:00Z",
    phase: float = 0.0,
) -> pd.DataFrame:
    count = days * 24 * 60
    times = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    steps = np.arange(count, dtype=float)
    returns = (
        0.00002
        + 0.00035 * np.sin(steps / 17.0 + phase)
        + 0.00012 * np.sin(steps / 113.0 + phase * 0.5)
    )
    closes = 10.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open_time_ms": times.astype("int64") // 1_000_000,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(count),
            "close_time_ms": times.astype("int64") // 1_000_000 + 59_999,
            "is_final": True,
        }
    )


def test_approved_contract_and_hac_lags_are_frozen() -> None:
    contract = load_volatility_contract(CONTRACT_PATH)

    assert contract.horizons_minutes == (30, 60, 180, 360)
    assert {
        horizon: hac_lag(horizon, contract.issuance_cadence_minutes)
        for horizon in contract.horizons_minutes
    } == EXPECTED_HAC_LAGS


def test_contract_hash_detects_mutation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["prediction_interval_coverage"] = 0.8
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VolatilityError, match="SHA-256 mismatch"):
        load_volatility_contract(path)


def test_rehashed_contract_cannot_weaken_evidence_floor(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["minimum_history_days"] = 1
    payload["contract_sha256"] = contract_digest(payload)
    path = tmp_path / "weakened_contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VolatilityError, match="approved 90 days"):
        load_volatility_contract(path)


def test_price_validation_deduplicates_exact_and_rejects_conflict() -> None:
    frame = _minute_frame(days=1)
    exact = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)

    normalized, audit = validate_price_frame(
        exact,
        symbol="TESTUSDT",
        series_kind="mark_kline",
    )

    assert len(normalized) == len(frame)
    assert audit.exact_duplicates == 1
    conflict = frame.iloc[[10]].copy()
    conflict["close"] = conflict["close"] * 1.01
    conflicting = pd.concat([frame, conflict], ignore_index=True)
    with pytest.raises(VolatilityError, match="conflicting duplicate"):
        validate_price_frame(
            conflicting,
            symbol="TESTUSDT",
            series_kind="mark_kline",
        )
    reordered = frame.copy()
    reordered.iloc[[10, 11]] = reordered.iloc[[11, 10]].to_numpy()
    with pytest.raises(VolatilityError, match="unique and monotonic"):
        validate_price_frame(
            reordered,
            symbol="TESTUSDT",
            series_kind="mark_kline",
        )


def test_rv_examples_exclude_windows_touching_a_gap() -> None:
    contract = replace(
        load_volatility_contract(CONTRACT_PATH),
        horizons_minutes=(30,),
        ridge_alphas=(1.0,),
        ewma_lambdas=(0.94,),
    )
    complete = _minute_frame(days=4)
    with_gap = complete.drop(index=3 * 24 * 60).reset_index(drop=True)

    complete_examples, complete_audit = build_rv_examples(
        complete,
        symbol="TESTUSDT",
        contract=contract,
    )
    gap_examples, gap_audit = build_rv_examples(
        with_gap,
        symbol="TESTUSDT",
        contract=contract,
    )

    assert not complete_examples.empty
    assert len(gap_examples) < len(complete_examples)
    assert gap_audit["price_audit"]["gap_count"] == 1
    assert complete_audit["price_audit"]["gap_count"] == 0
    first = complete_examples.iloc[0]
    assert first["target_rv"] >= 0.0
    assert first["rv_30"] > 0.0


def test_forward_rv_uses_exact_five_minute_anchors() -> None:
    contract = load_volatility_contract(CONTRACT_PATH)
    minute = _minute_frame(days=4)
    examples, _ = build_rv_examples(
        minute,
        symbol="TESTUSDT",
        contract=contract,
    )
    row = examples.loc[examples["horizon_minutes"] == 30].iloc[0]
    times = pd.to_datetime(minute["open_time_ms"], unit="ms", utc=True)
    origin_index = int(np.flatnonzero(times == row["origin_utc"])[0])
    sampled = np.asarray(
        minute["close"].iloc[origin_index : origin_index + 31 : 5],
        dtype=float,
    )
    expected = float(np.square(np.diff(np.log(sampled))).sum())

    assert float(row["target_rv"]) == pytest.approx(expected, rel=0.0, abs=1e-18)
    assert float(row["target_volatility_pct"]) == pytest.approx(
        100.0 * math.sqrt(expected),
        rel=0.0,
        abs=1e-15,
    )
    assert row["label_end_utc"] == row["origin_utc"] + pd.Timedelta(minutes=30)


def test_qlike_supports_zero_actual_but_rejects_zero_forecast() -> None:
    loss = qlike_loss([0.0, 0.5], [0.25, 0.5])

    assert np.isfinite(loss).all()
    with pytest.raises(VolatilityError, match="forecast RV"):
        qlike_loss([0.0], [0.0])


def test_newey_west_uses_overlap_lag_and_blocks_short_series() -> None:
    values = np.sin(np.arange(300, dtype=float) / 7.0) * 0.01 - 0.02

    result = newey_west_mean_test(values, horizon_minutes=360)

    assert result["hac_lag"] == 119
    assert result["hac_kernel"] == "bartlett"
    centered = values - float(np.mean(values))
    expected_lrv = float(np.dot(centered, centered) / len(centered))
    for offset in range(1, 120):
        expected_lrv += (
            2.0
            * (1.0 - offset / 120.0)
            * float(np.dot(centered[offset:], centered[:-offset]) / len(centered))
        )
    assert result["long_run_variance"] == pytest.approx(
        expected_lrv,
        rel=0.0,
        abs=1e-18,
    )
    with pytest.raises(VolatilityError, match="insufficient observations"):
        newey_west_mean_test(values[:120], horizon_minutes=360)


def test_interval_width_uses_shared_calibration_median() -> None:
    calibration_actual = np.linspace(0.01, 0.20, 200)
    candidate = calibration_actual * (1.0 + 0.02 * np.sin(np.arange(200)))
    contract = calibrate_log_interval(
        calibration_actual,
        candidate,
        rv_floor=1e-8,
    )
    lower, upper = apply_log_interval(candidate, contract)
    metrics = interval_metrics(
        calibration_actual,
        lower,
        upper,
        scale=float(contract["interval_width_scale"]),
    )

    assert math.isclose(
        float(contract["interval_width_scale"]),
        float(np.median(calibration_actual)),
    )
    assert contract["interval_width_formula"] == (
        "mean((upper - lower) / calibration_median_actual_rv)"
    )
    assert contract["rv_floor"] == pytest.approx(1e-8)
    assert contract["rv_floor_source"] == "development_positive_actual_rv_min_half"
    assert metrics["mean_normalized_interval_width"] >= 0.0


def test_interval_calibration_rejects_zero_actual_median() -> None:
    with pytest.raises(VolatilityError, match="positive"):
        calibrate_log_interval(
            np.concatenate([np.zeros(51), np.ones(49)]),
            np.full(100, 0.01),
            rv_floor=1e-8,
        )


def test_holm_adjustment_and_runtime_horizon_are_frozen() -> None:
    assert holm_adjust([0.01, 0.01, 0.04]) == pytest.approx([0.03, 0.03, 0.04])
    assert select_runtime_horizon(31, [30, 60, 180, 360]) == 60
    assert select_runtime_horizon(360, [360]) == 360
    with pytest.raises(VolatilityError, match=r"\(0, 360\]"):
        select_runtime_horizon(361, [360])


def test_newey_west_rejects_nonpositive_long_run_variance() -> None:
    with pytest.raises(VolatilityError, match="long-run variance"):
        newey_west_mean_test(np.ones(300), horizon_minutes=30)


def test_latest_snapshot_is_causal_and_reports_mark_last_divergence() -> None:
    contract = load_volatility_contract(CONTRACT_PATH)
    mark = _minute_frame(days=4)
    last = mark.copy()
    last["close"] = last["close"] * (1.0 + 0.0001 * np.sin(np.arange(len(last))))
    asof = cast(
        pd.Timestamp,
        pd.Timestamp(mark["open_time_ms"].iloc[-1], unit="ms", tz="UTC"),
    )

    snapshot = latest_rv_snapshot(
        mark,
        last,
        symbol="TESTUSDT",
        contract=contract,
        asof_utc=asof,
    )

    assert snapshot["cutoff_utc"] == asof.isoformat()
    assert set(snapshot["features"]) == {"rv_30", "rv_60", "rv_180", "rv_360"}
    assert snapshot["last_rv"]["360"] is not None
    assert snapshot["mark_volatility_pct"]["360"] == pytest.approx(
        100.0 * math.sqrt(snapshot["mark_rv"]["360"])
    )


def test_shadow_training_writes_frozen_artifacts_without_active_model_path(
    tmp_path: Path,
) -> None:
    base = load_volatility_contract(CONTRACT_PATH)
    contract = replace(
        base,
        horizons_minutes=(30,),
        min_fit_origins=5,
        min_calibration_origins=2,
        min_test_origins=2,
        ridge_alphas=(1.0,),
        ewma_lambdas=(0.94,),
    )
    examples: list[pd.DataFrame] = []
    for symbol, phase in (("AAAUSDT", 0.0), ("BBBUSDT", 0.7)):
        frame, _ = build_rv_examples(
            _minute_frame(days=6, phase=phase),
            symbol=symbol,
            contract=contract,
        )
        examples.append(frame)
    panel = pd.concat(examples, ignore_index=True)

    result = train_evaluate_shadow_volatility(
        panel,
        contract=contract,
        output_dir=tmp_path / "audit",
    )

    assert result["schema_version"] == "neutralgrid_shadow_volatility_oos_report_v1"
    assert result["data_manifest"]["final_test_frozen_before_selection"] is True
    assert len(result["results"]) == 2
    assert all(
        item["tail_diagnostic"]["policy"] == "diagnostic_only"
        and item["tail_diagnostic"]["eligibility_effect"] == "none"
        for item in result["results"]
    )
    assert all(
        item["selection_scope"] == "independent_per_symbol_horizon"
        and item["predictive_accuracy_test_scope"]
        == "symbol_loss_differential_only"
        for item in result["results"]
    )
    assert result["multiple_testing"]["panel_pooled_dm_prohibited"] is True
    assert (tmp_path / "audit" / "model.joblib").is_file()
    assert (tmp_path / "audit" / "metadata.json").is_file()
    assert (tmp_path / "audit" / "rv_examples.parquet").is_file()
    assert "models" not in str(tmp_path / "audit")


def test_artifact_loader_rejects_model_hash_mismatch(tmp_path: Path) -> None:
    (tmp_path / "model.joblib").write_bytes(b"not-the-declared-model")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": VOLATILITY_ARTIFACT_SCHEMA,
                "forecast_eligible": True,
                "verdict_influence": False,
                "model_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(VolatilityError, match="SHA-256 mismatch"):
        _load_artifact(tmp_path)


def test_global_split_purges_boundaries_and_training_lineage_is_fit_only(
    tmp_path: Path,
) -> None:
    base = load_volatility_contract(CONTRACT_PATH)
    contract = replace(
        base,
        horizons_minutes=(30,),
        min_fit_origins=5,
        min_calibration_origins=2,
        min_test_origins=2,
        ridge_alphas=(1.0,),
        ewma_lambdas=(0.94,),
    )
    frames = [
        build_rv_examples(
            _minute_frame(days=6, phase=phase),
            symbol=symbol,
            contract=contract,
        )[0]
        for symbol, phase in (("AAAUSDT", 0.0), ("BBBUSDT", 0.9))
    ]
    panel = pd.concat(frames, ignore_index=True)
    fit, calibration, test, audit = _global_splits(panel, contract)
    fit_cutoff = pd.Timestamp(audit["fit_cutoff_utc"])
    calibration_cutoff = pd.Timestamp(audit["calibration_cutoff_utc"])

    assert cast(pd.Series, fit["label_end_utc"]).max() <= fit_cutoff
    assert cast(pd.Series, calibration["origin_utc"]).min() > fit_cutoff
    assert cast(pd.Series, calibration["label_end_utc"]).max() <= calibration_cutoff
    assert cast(pd.Series, test["origin_utc"]).min() > calibration_cutoff

    result = train_evaluate_shadow_volatility(
        panel,
        contract=contract,
        output_dir=tmp_path / "audit",
    )
    for item in result["results"]:
        lineage = item["training_lineage"]
        lineage_max = cast(
            pd.Timestamp,
            pd.Timestamp(lineage["max_label_end_utc"]),
        )
        assert lineage_max <= fit_cutoff
        if lineage["scope"] == "per_symbol":
            assert lineage["symbols"] == [item["symbol"]]
        else:
            assert lineage["symbols"] == ["AAAUSDT", "BBBUSDT"]


def test_identical_inputs_produce_deterministic_statistical_outputs(
    tmp_path: Path,
) -> None:
    base = load_volatility_contract(CONTRACT_PATH)
    contract = replace(
        base,
        horizons_minutes=(30,),
        min_fit_origins=5,
        min_calibration_origins=2,
        min_test_origins=2,
        ridge_alphas=(1.0,),
        ewma_lambdas=(0.94,),
    )
    panel, _ = build_rv_examples(
        _minute_frame(days=6),
        symbol="AAAUSDT",
        contract=contract,
    )
    first = train_evaluate_shadow_volatility(
        panel,
        contract=contract,
        output_dir=tmp_path / "first",
    )
    second = train_evaluate_shadow_volatility(
        panel,
        contract=contract,
        output_dir=tmp_path / "second",
    )

    first_result = first["results"][0]
    second_result = second["results"][0]
    for key in ("baseline", "challenger", "point_loss", "interval", "gates"):
        assert first_result[key] == second_result[key]
    assert first["data_manifest"]["content_sha256"] == second["data_manifest"][
        "content_sha256"
    ]


def test_training_assembly_excludes_only_the_unready_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = SimpleNamespace(primary_series="mark_kline", minimum_history_days=90)

    def fake_load(
        _root: Path,
        *,
        symbol: str,
        series_kind: str,
    ) -> pd.DataFrame:
        assert series_kind == "mark_kline"
        return pd.DataFrame(
            {
                "symbol": [symbol, symbol],
                "open_time_ms": [0, 1_767_225_600_000],
            }
        )

    def fake_validate(
        frame: pd.DataFrame,
        *,
        symbol: str,
        series_kind: str,
    ) -> tuple[pd.DataFrame, SimpleNamespace]:
        assert str(frame.loc[0, "symbol"]) == symbol
        assert series_kind == "mark_kline"
        assert len(frame) == 1
        return frame, SimpleNamespace(rows=1)

    def fake_examples(
        frame: pd.DataFrame,
        *,
        symbol: str,
        contract: object,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        assert str(frame.loc[0, "symbol"]) == symbol
        return pd.DataFrame({"symbol": [symbol], "value": [1]}), {"rows": 1}

    monkeypatch.setattr(training_script, "load_volatility_contract", lambda _path: contract)
    monkeypatch.setattr(training_script, "load_price_store_frame", fake_load)
    monkeypatch.setattr(training_script, "validate_price_frame", fake_validate)
    monkeypatch.setattr(training_script, "build_rv_examples", fake_examples)
    monkeypatch.setattr(
        training_script,
        "_calendar_day_count",
        lambda frame: 90 if str(frame.loc[0, "symbol"]) == "AAAUSDT" else 89,
    )
    monkeypatch.setattr(
        training_script,
        "audit_single_symbol_readiness",
        lambda _examples, *, symbol, contract: {
            "status": "ready" if symbol == "AAAUSDT" else "blocked",
            "counts": {"fit": 216, "calibration": 72, "test": 72},
        },
    )

    examples, audits = training_script.assemble_training_examples(
        symbols=["AAAUSDT", "BBBUSDT"],
        price_store=tmp_path / "prices",
        contract_path=tmp_path / "contract.json",
        scope_start_dates={
            "AAAUSDT": date(2026, 1, 1),
            "BBBUSDT": date(2026, 1, 1),
        },
    )

    assert examples["symbol"].tolist() == ["AAAUSDT"]
    assert [audit["training_admitted"] for audit in audits] == [True, False]
