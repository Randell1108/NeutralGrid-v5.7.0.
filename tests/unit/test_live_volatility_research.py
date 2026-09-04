from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import scripts.research_live_volatility_v2 as research_script

from neutralgrid.live.decision.volatility import (
    VolatilityError,
    build_rv_examples,
    calibrate_log_interval,
    load_volatility_contract,
)
from neutralgrid.live.decision.volatility_forecast import (
    _development_target_floor,
    _fit_har,
    _global_splits,
    _predict_har,
    _select_baseline,
    _load_artifact,
)
from neutralgrid.live.decision.volatility_research import (
    JUMP_FEATURE_COLUMNS,
    SEMIVARIANCE_FEATURE_COLUMNS,
    VolatilityResearchError,
    _symbol_semivariance_features,
    deduplicate_research_examples,
    load_volatility_research_contract,
)
from neutralgrid.live.decision.volatility_research_evaluation import (
    RESEARCH_ARTIFACT_SCHEMA,
    run_consumed_holdout_research,
    validate_research_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_CONTRACT_PATH = ROOT / "config" / "live_volatility_forecast_v1.json"
RESEARCH_CONTRACT_PATH = ROOT / "config" / "live_volatility_research_v2.json"


def _minute_frame(rows: int = 4_000) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=rows, freq="1min", tz="UTC")
    pattern = np.asarray([0.0004, -0.0003, 0.0002, -0.0001, 0.0005])
    returns = np.resize(pattern, rows)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open_time_ms": (times.astype("int64") // 1_000_000).astype("int64"),
            "close": prices,
            "is_final": True,
        }
    )


def test_research_contract_is_bound_and_verdict_inert() -> None:
    base = load_volatility_contract(BASE_CONTRACT_PATH)
    research = load_volatility_research_contract(
        RESEARCH_CONTRACT_PATH,
        base_contract=base,
    )

    assert research.base_contract_sha256 == base.contract_sha256
    assert research.candidate_families == (
        "har",
        "har_rs",
        "har_j",
        "har_rs_j",
    )
    payload = json.loads(RESEARCH_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert payload["existing_oos_role"] == "consumed_diagnostic_only"
    assert payload["promotion_eligible"] is False
    assert payload["verdict_influence"] is False
    assert payload["target_classification_policy"] == (
        "regression_only_no_rv_classifier"
    )
    assert payload["jump_estimator"] == (
        "bipower_variation_mu1_inverse_square_sum_adjacent_abs_5m_returns"
    )


def test_deduplication_noops_exact_rows_and_rejects_conflicts() -> None:
    row = {
        "symbol": "BTCUSDT",
        "origin_utc": "2026-01-01T00:00:00Z",
        "horizon_minutes": 30,
        "target_rv": 0.01,
    }
    exact = pd.DataFrame([row, dict(row)])
    clean, audit = deduplicate_research_examples(exact)

    assert len(clean) == 1
    assert audit["exact_duplicate_rows_removed"] == 1
    assert audit["conflicting_duplicate_keys"] == 0

    conflict = exact.copy()
    conflict.loc[1, "target_rv"] = 0.02
    with pytest.raises(VolatilityResearchError, match="conflicting research duplicate"):
        deduplicate_research_examples(conflict)


def test_semivariance_features_reconstruct_v1_rv_exactly() -> None:
    base = load_volatility_contract(BASE_CONTRACT_PATH)
    research = load_volatility_research_contract(
        RESEARCH_CONTRACT_PATH,
        base_contract=base,
    )
    mark = _minute_frame()
    examples, _ = build_rv_examples(mark, symbol="BTCUSDT", contract=base)
    assert not examples.empty

    features, audit = _symbol_semivariance_features(
        examples,
        mark,
        symbol="BTCUSDT",
        base_contract=base,
        research_contract=research,
    )

    assert set(SEMIVARIANCE_FEATURE_COLUMNS).issubset(features.columns)
    assert set(JUMP_FEATURE_COLUMNS).issubset(features.columns)
    origins = examples.drop_duplicates("origin_utc").set_index("origin_utc")
    observed = features.set_index("origin_utc")
    for window in (30, 60, 180, 360):
        np.testing.assert_allclose(
            observed[f"rs_pos_{window}"] + observed[f"rs_neg_{window}"],
            origins[f"rv_{window}"],
            rtol=1e-12,
            atol=1e-18,
        )
    assert audit["maximum_semivariance_identity_absolute_error"] <= 1e-18
    assert bool((features[list(JUMP_FEATURE_COLUMNS)] >= 0.0).all().all())


def test_jump_features_use_bipower_variation_on_complete_5m_returns() -> None:
    base = load_volatility_contract(BASE_CONTRACT_PATH)
    research = load_volatility_research_contract(
        RESEARCH_CONTRACT_PATH,
        base_contract=base,
    )
    mark = _minute_frame()
    examples, _ = build_rv_examples(mark, symbol="BTCUSDT", contract=base)
    features, _ = _symbol_semivariance_features(
        examples,
        mark,
        symbol="BTCUSDT",
        base_contract=base,
        research_contract=research,
    )
    origin = cast(pd.Timestamp, features["origin_utc"].iloc[-1])
    samples = pd.date_range(
        origin - pd.Timedelta(minutes=30),
        origin,
        freq="5min",
        tz="UTC",
    )
    prices = pd.Series(
        np.asarray(mark["close"], dtype=float),
        index=pd.to_datetime(mark["open_time_ms"], unit="ms", utc=True),
    ).reindex(samples)
    returns = np.diff(np.log(np.asarray(prices, dtype=float)))
    realized = float(np.sum(np.square(returns)))
    bipower = float((np.pi / 2.0) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])))
    expected_jump = max(realized - bipower, 0.0)
    observed_jump = float(
        features.loc[features["origin_utc"] == origin, "jump_30"].iloc[0]
    )
    assert observed_jump == pytest.approx(expected_jump, rel=1e-12, abs=1e-18)


def test_semivariance_rejects_a_source_gap_instead_of_imputing() -> None:
    base = load_volatility_contract(BASE_CONTRACT_PATH)
    research = load_volatility_research_contract(
        RESEARCH_CONTRACT_PATH,
        base_contract=base,
    )
    mark = _minute_frame()
    examples, _ = build_rv_examples(mark, symbol="BTCUSDT", contract=base)
    assert not examples.empty
    selected = examples.loc[
        examples["origin_utc"] == examples["origin_utc"].iloc[-1]
    ].copy()
    origin_ns = int(cast(pd.Timestamp, selected["origin_utc"].iloc[0]).value)
    gap_ms = int(
        (origin_ns - int(pd.Timedelta(minutes=5).value)) // 1_000_000
    )
    damaged = mark.loc[mark["open_time_ms"] != gap_ms].copy()

    with pytest.raises(VolatilityResearchError, match="touches a gap"):
        _symbol_semivariance_features(
            selected,
            damaged,
            symbol="BTCUSDT",
            base_contract=base,
            research_contract=research,
        )


def test_consumed_holdout_research_is_purged_and_never_promotion_eligible(
    tmp_path: Path,
) -> None:
    loaded_base = load_volatility_contract(BASE_CONTRACT_PATH)
    base = replace(
        loaded_base,
        horizons_minutes=(30,),
        min_fit_origins=10,
        min_calibration_origins=3,
        min_test_origins=3,
        ridge_alphas=(0.1, 1.0),
    )
    research = load_volatility_research_contract(
        RESEARCH_CONTRACT_PATH,
        base_contract=loaded_base,
    )
    mark = _minute_frame(rows=8_000)
    all_examples, _ = build_rv_examples(mark, symbol="BTCUSDT", contract=loaded_base)
    examples = all_examples.loc[all_examples["horizon_minutes"] == 30].copy()
    features, feature_audit = _symbol_semivariance_features(
        examples,
        mark,
        symbol="BTCUSDT",
        base_contract=base,
        research_contract=research,
    )
    examples = examples.merge(
        features,
        on="origin_utc",
        how="left",
        validate="many_to_one",
    )
    fit, calibration, _, split_audit = _global_splits(examples, base)
    baseline = _select_baseline(fit, base)
    original = _fit_har(fit, alpha=1.0, pooled=False)
    calibration_prediction = _predict_har(original, calibration)
    interval_floor = _development_target_floor(fit)
    original["candidate_interval"] = calibrate_log_interval(
        np.asarray(calibration["target_rv"], dtype=float),
        calibration_prediction,
        rv_floor=interval_floor,
        coverage=base.prediction_interval_coverage,
    )
    evidence = {
        "artifact_dir": str(tmp_path / "base"),
        "split_audit": split_audit,
        "model_payload": {"models": {"BTCUSDT|30": original}},
        "report_results": [
            {
                "symbol": "BTCUSDT",
                "horizon_minutes": 30,
                "baseline": baseline,
            }
        ],
    }
    final_root = tmp_path / "final"
    result = run_consumed_holdout_research(
        examples,
        evidence=evidence,
        base_contract=base,
        research_contract=research,
        output_dir=tmp_path / "staging",
        feature_audit={"feature_audits": [feature_audit]},
        artifact_path_root=final_root,
    )

    assert result["status"] == "diagnostic_complete_not_promotion_eligible"
    assert result["promotion_eligible"] is False
    assert result["verdict_influence"] is False
    assert result["runtime_effect"] == "none"
    assert result["summary"]["evaluated_pairs"] == 1
    assert result["summary"]["promotion_eligible_pairs"] == 0
    pair = result["results"][0]
    assert pair["promotion_eligible"] is False
    assert pair["origin_counts"]["fit"] >= 10
    for candidate in pair["selected_candidate"]["candidates"]:
        for fold in candidate["folds"]:
            assert fold["validation_max_label_end_utc"] <= fold[
                "validation_cutoff_utc"
            ]
    staging = tmp_path / "staging"
    staging.replace(final_root)
    metadata = json.loads(
        (final_root / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == RESEARCH_ARTIFACT_SCHEMA
    assert metadata["promotion_eligible"] is False
    assert result["artifact_paths"]["research_report"] == str(
        final_root.resolve() / "research_report.json"
    )
    validation = validate_research_artifact(final_root)
    assert validation["status"] == "valid"
    assert validation["evaluated_pairs"] == 1
    with pytest.raises(
        VolatilityError,
        match="unsupported volatility artifact metadata schema",
    ):
        _load_artifact(final_root)


def test_cli_does_not_delete_a_lock_owned_by_another_process(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / ".locked.lock"
    lock_path.write_text("pid=other\n", encoding="ascii")

    exit_code = research_script.main(
        [
            "--base-artifact-dir",
            str(tmp_path / "base"),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "locked",
        ]
    )

    assert exit_code == 2
    assert lock_path.read_text(encoding="ascii") == "pid=other\n"


def test_cli_removes_only_its_partial_staging_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "preexisting-user-file.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        research_script,
        "load_volatility_contract",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        research_script,
        "load_volatility_research_contract",
        lambda _path, base_contract: object(),
    )
    monkeypatch.setattr(
        research_script,
        "load_consumed_v1_evidence",
        lambda _path, base_contract: {"examples": pd.DataFrame()},
    )
    monkeypatch.setattr(
        research_script,
        "augment_examples_with_semivariance",
        lambda _examples, **_kwargs: (pd.DataFrame(), {}),
    )

    def fail_after_partial_write(
        _examples: pd.DataFrame,
        **kwargs: object,
    ) -> dict[str, object]:
        output_dir = cast(Path, kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "partial.json").write_text("{}", encoding="utf-8")
        raise VolatilityResearchError("injected failure")

    monkeypatch.setattr(
        research_script,
        "run_consumed_holdout_research",
        fail_after_partial_write,
    )

    exit_code = research_script.main(
        [
            "--base-artifact-dir",
            str(tmp_path / "base"),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "cleanup",
        ]
    )

    assert exit_code == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / ".cleanup.lock").exists()
    assert not list(tmp_path.glob(".cleanup.staging-*"))
