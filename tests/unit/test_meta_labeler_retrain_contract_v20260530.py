from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from neutralgrid.models.meta_labeler import (
    ACTIVE_META_TARGET_CONTRACT,
    ACTIVE_META_TARGET_DURATION_COLUMN,
    ACTIVE_META_TARGET_DURATION_MAX_HOURS,
    ACTIVE_META_TARGET_LABEL_COLUMN,
    ACTIVE_SNAPSHOT_META_FEATURES,
    MetaLabeler,
    MetaLabelerMetrics,
    PROMOTION_EVALUATION_CONTRACT,
)
from neutralgrid.models.artifacts import get_active_hmm_version
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder
from neutralgrid.validation.utility import compute_governed_provisional_utility


_CLI_PATH = Path(__file__).resolve().parents[2] / "retrain_meta_labeler.py"
_CLI_SPEC = importlib.util.spec_from_file_location("retrain_meta_labeler", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
retrain_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(retrain_cli)


def _complete_training_df() -> pd.DataFrame:
    row: dict[str, Any] = {feature: 1.0 for feature in ACTIVE_SNAPSHOT_META_FEATURES}
    row["y"] = 1
    row["net_pnl_pct"] = 5.0
    row["duration_hours"] = 6.0
    row["candidate_id"] = "BTCUSDT_20260408_000000_deadbeef"
    row["start_time_utc"] = "2026-04-08T00:00:00+00:00"
    return pd.DataFrame([row, {**row, "candidate_id": "ETHUSDT_20260408_000000_deadbeef"}])


@pytest.mark.parametrize(
    "flag_args",
    [
        ["--accept-version-mismatch"],
        ["--backfill"],
        ["--include-backtest-data", "legacy.csv"],
    ],
)
def test_parse_args_rejects_removed_legacy_flags(monkeypatch, flag_args: list[str]) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["retrain_meta_labeler.py", "--input", "dummy.csv", *flag_args],
    )

    with pytest.raises(SystemExit):
        retrain_cli.parse_args()


def test_parse_args_bare_run_uses_documented_reference_workbook(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["retrain_meta_labeler.py"])

    args = retrain_cli.parse_args()

    assert args.input == "data/new_expired_bots.xlsx"
    assert args.max_rows_per_symbol == 30


def test_parse_args_allows_uncapped_authoritative_meta_pool(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["retrain_meta_labeler.py", "--max-rows-per-symbol", "0"]
    )

    args = retrain_cli.parse_args()

    assert args.max_rows_per_symbol == 0


def test_parse_args_rejects_negative_meta_pool_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["retrain_meta_labeler.py", "--max-rows-per-symbol", "-1"]
    )

    with pytest.raises(SystemExit):
        retrain_cli.parse_args()


def test_inspect_deployed_artifact_health_detects_stale_feature_counts(tmp_path: Path) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    artifact_dir = output_path.parent / "meta_labeler"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.joblib").write_bytes(b"joblib-placeholder")
    stale_features = list(ACTIVE_SNAPSHOT_META_FEATURES[:-1])
    (artifact_dir / "metadata.json").write_text(
        json.dumps({"features": stale_features}),
        encoding="utf-8",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump({"feature_names": stale_features}, fh)

    report = retrain_cli.inspect_deployed_artifact_health(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
    )

    assert report["is_stale"] is True
    assert report["artifact_feature_count"] == len(stale_features)
    assert report["pickle_feature_count"] == len(stale_features)


def _promotion_metrics(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "promotion_status": "pass",
        "oof_auc": 0.72,
        "oof_auc_ci_low": 0.66,
        "oof_auc_ci_high": 0.78,
        "oof_ece": 0.06,
        "oof_deployable_auc": 0.68,
        "oof_deployable_ece": 0.07,
        "promotion_evaluation_contract": PROMOTION_EVALUATION_CONTRACT,
        "train_samples": 240,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_champion_metadata(
    output_path: Path,
    *,
    eval_overrides: dict[str, Any] | None = None,
    hmm_artifact_version: str | None = None,
) -> None:
    artifact_dir = output_path.parent / "meta_labeler"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics: dict[str, Any] = {
        "target_contract": ACTIVE_META_TARGET_CONTRACT,
        "promotion_status": "pass",
        "oof_auc": 0.72,
        "oof_auc_ci_low": 0.66,
        "oof_auc_ci_high": 0.78,
        "oof_ece": 0.06,
        "oof_deployable_auc": 0.68,
        "oof_deployable_ece": 0.07,
        "promotion_evaluation_contract": PROMOTION_EVALUATION_CONTRACT,
    }
    eval_metrics.update(eval_overrides or {})
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": "champion-v1",
                "features": list(ACTIVE_SNAPSHOT_META_FEATURES),
                "total_samples": 240,
                "lineage": {
                    "hmm_artifact_version": (
                        hmm_artifact_version or get_active_hmm_version()
                    ),
                },
                "eval_metrics": eval_metrics,
            }
        ),
        encoding="utf-8",
    )
    output_path.write_bytes(b"champion-remains-unchanged")


def test_champion_challenger_gate_accepts_initial_absolute_pass(tmp_path: Path) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(),
    )
    assert report["status"] == "pass"
    assert report["comparison_scope"] == "initial_deployment"
    assert report["reasons"] == []


def test_champion_challenger_gate_rejects_absolute_failure_before_save(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(promotion_status="fail"),
    )
    assert report["status"] == "fail"
    assert any(
        "candidate_absolute_promotion_gate_not_pass" in reason
        for reason in report["reasons"]
    )
    assert not output_path.exists()


def test_champion_challenger_gate_accepts_exact_non_degradation(tmp_path: Path) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    output_path.parent.mkdir(parents=True)
    _write_champion_metadata(output_path)

    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(),
    )

    assert report["status"] == "pass"
    assert report["comparison_kind"] == "stored_summary_nonpaired"
    assert report["reasons"] == []
    assert output_path.read_bytes() == b"champion-remains-unchanged"


def test_champion_challenger_uses_absolute_gate_for_incomparable_champion(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    output_path.parent.mkdir(parents=True)
    _write_champion_metadata(
        output_path,
        eval_overrides={"promotion_evaluation_contract": "legacy_shuffled_oof"},
        hmm_artifact_version="rolling_180d_stale",
    )

    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(oof_auc=0.60, oof_auc_ci_low=0.55),
    )

    assert report["status"] == "pass"
    assert report["comparison_scope"] == (
        "incomparable_champion_absolute_gate_only"
    )
    assert report["reasons"] == []
    assert len(report["comparison_skipped_reasons"]) == 2


def test_champion_challenger_gate_accumulates_regressions_and_preserves_champion(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    output_path.parent.mkdir(parents=True)
    _write_champion_metadata(output_path)

    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(oof_auc=0.70, oof_auc_ci_low=0.63, oof_ece=0.08),
    )
    decision_path = retrain_cli.write_champion_challenger_decision(
        report, output_path
    )

    assert report["status"] == "fail"
    assert {
        reason.split("(", 1)[0] for reason in report["reasons"]
    } >= {
        "candidate_oof_auc_regressed",
        "candidate_oof_auc_ci_low_regressed",
        "candidate_oof_ece_regressed",
    }
    assert output_path.read_bytes() == b"champion-remains-unchanged"
    assert json.loads(decision_path.read_text(encoding="utf-8"))["status"] == "fail"
    assert not decision_path.with_name(f"{decision_path.name}.tmp").exists()


def test_champion_challenger_gate_preserves_deployable_stratum_metrics(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    _write_champion_metadata(output_path)

    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(oof_deployable_auc=0.66, oof_deployable_ece=0.09),
    )

    assert report["status"] == "fail"
    assert any(
        str(reason).startswith("candidate_oof_deployable_auc_regressed")
        for reason in report["reasons"]
    )
    assert any(
        str(reason).startswith("candidate_oof_deployable_ece_regressed")
        for reason in report["reasons"]
    )
    assert output_path.read_bytes() == b"champion-remains-unchanged"


def test_champion_challenger_gate_fails_closed_on_corrupt_existing_metadata(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"champion-remains-unchanged")
    artifact_dir = output_path.parent / "meta_labeler"
    artifact_dir.mkdir()
    (artifact_dir / "metadata.json").write_text("not-json", encoding="utf-8")

    report = retrain_cli.evaluate_champion_challenger_gate(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
        _promotion_metrics(),
    )

    assert report["status"] == "fail"
    assert report["reasons"] == ["champion_metadata_unreadable:JSONDecodeError"]
    assert output_path.read_bytes() == b"champion-remains-unchanged"


def test_build_feature_verification_report_records_exact_active_feature_contract(tmp_path: Path) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    training_df = _complete_training_df()
    training_df["duration_hours"] = [6.0, 8.0]
    artifact_health = retrain_cli.inspect_deployed_artifact_health(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
    )

    report = retrain_cli.build_feature_verification_report(
        training_df,
        feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
        output_path=output_path,
        artifact_health_before=artifact_health,
    )

    assert report["expected_feature_count"] == len(ACTIVE_SNAPSHOT_META_FEATURES)
    assert report["selected_feature_count"] == len(ACTIVE_SNAPSHOT_META_FEATURES)
    assert report["is_selected_feature_count_exact"] is True
    assert report["is_value_complete"] is True
    assert report["source_training_row_count"] == 2
    assert report["training_row_count"] == 2
    assert report["allow_imputation"] is False
    assert report["imputed_features"] == {}
    # PIPELINE_FIX B.1 AND-label: the 8.0h row is a slow NEGATIVE (not excluded),
    # so both rows are labelable; the 6.0h row is the only fast-winner positive.
    assert report["fast_target_contract"]["eligible_rows"] == 2
    assert report["fast_target_contract"]["positive_count"] == 1
    assert report["fast_target_contract"]["negative_count"] == 1
    assert report["active_training_target"]["target_column"] == retrain_cli.FAST_TARGET_LABEL_COLUMN
    assert report["artifact_loadability_before"]["can_load"] is False
    # UTILFIX-01: "pinned_v0_fallback" was removed; the runtime utility path
    # now fail-closes when current.json is missing, and the verification
    # report records the actual provenance.
    assert report["utility_provenance"]["source"] in {
        "current_artifact",
        "calibrator_unavailable",
    }


def test_training_source_manifest_hashes_exact_builder_inputs_and_population(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "fastwin"
    source_dir.mkdir()
    first = source_dir / "training_data_first.csv"
    second = source_dir / "training_rows_second.csv"
    ignored = source_dir / "unrelated.csv"
    first.write_text("candidate_id,value\na,1\n", encoding="utf-8")
    second.write_text("candidate_id,value\nb,2\n", encoding="utf-8")
    ignored.write_text("candidate_id,value\nc,3\n", encoding="utf-8")
    training_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "start_time_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
            "backtest_timestamp": [
                "2026-02-01T00:00:00Z",
                "2026-02-02T00:00:00Z",
            ],
        }
    )

    manifest = retrain_cli.build_training_source_manifest(source_dir, training_df)

    assert manifest["source_file_count"] == 2
    assert [Path(item["path"]).name for item in manifest["source_files"]] == [
        "training_data_first.csv",
        "training_rows_second.csv",
    ]
    assert all(len(item["sha256"]) == 64 for item in manifest["source_files"])
    assert manifest["source_training_rows"] == 2
    assert manifest["unique_candidate_ids"] == 2
    assert manifest["duplicate_candidate_id_rows"] == 0
    assert manifest["market_event_time_utc"]["min"] == "2026-01-01T00:00:00+00:00"
    assert manifest["market_event_time_utc"]["max"] == "2026-01-02T00:00:00+00:00"
    assert manifest["backtest_write_time_utc"]["max"] == "2026-02-02T00:00:00+00:00"


def test_training_hmm_lineage_gate_rejects_unstamped_hmm_derived_values() -> None:
    training_df = pd.DataFrame({"candidate_id": ["a", "b"], "ev_score": [1.0, 2.0]})

    report = retrain_cli.audit_training_hmm_lineage(
        training_df,
        selected_features=["ev_score", "adx_1h"],
        active_hmm_artifact_version="rolling_active",
    )

    assert report["required"] is True
    assert report["passes"] is False
    assert report["missing_lineage_count"] == 2
    assert report["reasons"] == ["hmm_artifact_version_column_missing"]


def test_training_hmm_lineage_gate_accepts_only_uniform_active_lineage() -> None:
    training_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "ev_score": [1.0, 2.0],
            "hmm_artifact_version": ["rolling_active", "rolling_active"],
            "hmm_trained_at_utc": [
                "2026-08-15T23:14:45.834213+00:00",
                "2026-08-15T23:14:45.834213+00:00",
            ],
            "hmm_feature_semantics_version": [
                "agg_regime_probs_v20260320",
                "agg_regime_probs_v20260320",
            ],
            "hmm_feature_source": ["pinned_artifact_replay"] * 2,
            "feature_cutoff_utc": [
                "2026-07-01T00:00:00+00:00",
                "2026-07-02T00:00:00+00:00",
            ],
            "start_time_utc": [
                "2026-07-01T00:00:00+00:00",
                "2026-07-02T00:00:00+00:00",
            ],
            "range_prob": [0.8, 0.7],
            "trend_prob": [0.2, 0.3],
            "persistence_prob": [0.7, 0.8],
        }
    )

    report = retrain_cli.audit_training_hmm_lineage(
        training_df,
        selected_features=["ev_score", "adx_1h"],
        active_hmm_artifact_version="rolling_active",
        active_hmm_trained_at_utc="2026-08-15T23:14:45.834213+00:00",
    )

    assert report["passes"] is True
    assert report["missing_lineage_count"] == 0
    assert report["mismatched_lineage_count"] == 0
    assert report["observed_versions"] == {"rolling_active": 2}


def test_training_hmm_lineage_gate_rejects_missing_and_mixed_versions() -> None:
    training_df = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "ev_score": [1.0, 2.0, 3.0],
            "hmm_artifact_version": ["rolling_active", "rolling_stale", None],
            "hmm_trained_at_utc": [
                "2026-08-15T23:14:45.834213+00:00",
            ] * 3,
            "hmm_feature_semantics_version": [
                "agg_regime_probs_v20260320",
            ] * 3,
            "hmm_feature_source": ["pinned_artifact_replay"] * 3,
            "feature_cutoff_utc": ["2026-07-01T00:00:00+00:00"] * 3,
            "start_time_utc": ["2026-07-01T00:00:00+00:00"] * 3,
            "range_prob": [0.8, 0.7, 0.6],
            "trend_prob": [0.2, 0.3, 0.4],
            "persistence_prob": [0.7, 0.8, 0.9],
        }
    )

    report = retrain_cli.audit_training_hmm_lineage(
        training_df,
        selected_features=["ev_score"],
        active_hmm_artifact_version="rolling_active",
        active_hmm_trained_at_utc="2026-08-15T23:14:45.834213+00:00",
    )

    assert report["passes"] is False
    assert report["missing_lineage_count"] == 1
    assert report["mismatched_lineage_count"] == 1
    assert report["observed_versions"] == {
        "rolling_active": 1,
        "rolling_stale": 1,
    }
    assert report["reasons"] == [
        "hmm_artifact_version_missing_on_training_rows",
        "training_hmm_lineage_does_not_match_active_hmm",
    ]


@pytest.mark.parametrize(
    ("column", "value", "expected_reason"),
    [
        ("hmm_trained_at_utc", "", "hmm_trained_at_utc_missing_on_training_rows"),
        (
            "hmm_feature_semantics_version",
            "stale_semantics",
            "training_hmm_feature_semantics_version_invalid",
        ),
        ("hmm_feature_source", "", "hmm_feature_source_missing_on_training_rows"),
        ("feature_cutoff_utc", "bad-time", "feature_cutoff_utc_invalid_on_training_rows"),
        ("range_prob", 1.2, "training_hmm_probability_out_of_range"),
    ],
)
def test_training_hmm_lineage_gate_rejects_incomplete_direct_provenance(
    column: str,
    value: object,
    expected_reason: str,
) -> None:
    training_df = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "ev_score": [1.0],
            "hmm_artifact_version": ["rolling_active"],
            "hmm_trained_at_utc": ["2026-08-15T23:14:45.834213+00:00"],
            "hmm_feature_semantics_version": ["agg_regime_probs_v20260320"],
            "hmm_feature_source": ["pinned_artifact_replay"],
            "feature_cutoff_utc": ["2026-07-01T00:00:00+00:00"],
            "start_time_utc": ["2026-07-01T00:00:00+00:00"],
            "range_prob": [0.8],
            "trend_prob": [0.2],
            "persistence_prob": [0.7],
        }
    )
    training_df.loc[0, column] = value

    report = retrain_cli.audit_training_hmm_lineage(
        training_df,
        selected_features=["ev_score"],
        active_hmm_artifact_version="rolling_active",
        active_hmm_trained_at_utc="2026-08-15T23:14:45.834213+00:00",
    )

    assert report["passes"] is False
    assert expected_reason in report["reasons"]


def test_prepare_fast_target_training_frame_applies_and_label_contract() -> None:
    training_df = _complete_training_df()
    losing_row = training_df.iloc[0].copy()
    losing_row["candidate_id"] = "SOLUSDT_20260408_000000_deadbeef"
    losing_row["duration_hours"] = 7.0
    losing_row["net_pnl_pct"] = -1.5
    losing_row["y"] = 0
    # slow-but-profitable: dur 8h > 7h AND pnl 9 >= 3. Under the AND-label this is a
    # NEGATIVE (it is slow, not a fast winner) — it must be kept, not excluded. Its
    # builder y=1 deliberately disagrees with the fast-winner contract target (0).
    slow_row = training_df.iloc[0].copy()
    slow_row["candidate_id"] = "ADAUSDT_20260408_000000_deadbeef"
    slow_row["duration_hours"] = 8.0
    slow_row["net_pnl_pct"] = 9.0
    slow_row["y"] = 1
    training_df = pd.concat(
        [training_df, pd.DataFrame([losing_row, slow_row])],
        ignore_index=True,
    )

    filtered_df, summary = retrain_cli.prepare_fast_target_training_frame(
        training_df,
        pnl_col="net_pnl_pct",
    )

    # All 4 rows are labelable; >7h rows are negatives, not exclusions.
    assert len(filtered_df) == 4
    assert filtered_df["duration_hours"].max() == pytest.approx(8.0)
    assert filtered_df[retrain_cli.FAST_TARGET_LABEL_COLUMN].tolist() == [1, 1, 0, 0]
    assert summary["trained_row_count"] == 4
    assert summary["eligible_rows"] == 4
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 2
    assert summary["negative_rows_duration_gt_7h_count"] == 1
    assert summary["target_column"] == retrain_cli.FAST_TARGET_LABEL_COLUMN


def test_enforce_feature_contract_blocks_reduced_selected_feature_set() -> None:
    training_df = _complete_training_df()

    with pytest.raises(ValueError, match=f"expects {len(ACTIVE_SNAPSHOT_META_FEATURES)} feature\\(s\\)"):
        retrain_cli.enforce_feature_contract(
            training_df,
            feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
            selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES[:-1]),
        )


def test_enforce_feature_contract_blocks_selected_feature_nulls() -> None:
    training_df = _complete_training_df()
    training_df.loc[0, "ou_halflife"] = None

    with pytest.raises(ValueError, match="ou_halflife \\(1/2 null\\)"):
        retrain_cli.enforce_feature_contract(
            training_df,
            feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
            selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
        )


def test_filter_modelable_selected_feature_rows_classifies_incomplete_rows() -> None:
    training_df = _complete_training_df()
    training_df.loc[0, "ou_halflife"] = None

    modelable_df, report = retrain_cli.filter_modelable_selected_feature_rows(
        training_df,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
    )

    assert len(modelable_df) == 1
    assert report["input_rows"] == 2
    assert report["modelable_rows"] == 1
    assert report["excluded_unmodelable_count"] == 1
    excluded = report["excluded_unmodelable_rows"][0]
    assert excluded["candidate_id"] == "BTCUSDT_20260408_000000_deadbeef"
    assert excluded["classification"] == "excluded_unmodelable"
    assert excluded["missing_features"] == ["ou_halflife"]

    retrain_cli.enforce_feature_contract(
        modelable_df,
        feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
    )


def test_allow_imputation_fills_selected_feature_nulls_before_contract() -> None:
    training_df = _complete_training_df()
    training_df.loc[0, "ou_halflife"] = None

    imputed_df, report = retrain_cli.impute_selected_feature_nulls(
        training_df,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
    )

    assert report == {"ou_halflife": {"count": 1, "default": 24.0}}
    assert imputed_df.loc[0, "ou_halflife"] == 24.0
    retrain_cli.enforce_feature_contract(
        imputed_df,
        feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
    )


def test_parse_args_accepts_allow_imputation_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["retrain_meta_labeler.py", "--input", "dummy.csv", "--allow-imputation"],
    )

    args = retrain_cli.parse_args()

    assert args.allow_imputation is True


def test_active_bootstrap_profile_blocks_allow_imputation() -> None:
    with pytest.raises(ValueError, match="must remain exact and non-imputed"):
        retrain_cli.enforce_requested_profile_mode(
            feature_profile=retrain_cli.ACTIVE_BOOTSTRAP_META_FEATURE_PROFILE,
            allow_imputation=True,
        )


def test_write_feature_verification_report_persists_json(tmp_path: Path) -> None:
    output_path = tmp_path / "models" / "meta_labeler.pkl"
    training_df = _complete_training_df()
    training_df["duration_hours"] = [6.0, 6.5]
    artifact_health = retrain_cli.inspect_deployed_artifact_health(
        output_path,
        list(ACTIVE_SNAPSHOT_META_FEATURES),
    )
    report = retrain_cli.build_feature_verification_report(
        training_df,
        feature_profile=retrain_cli.ACTIVE_META_FEATURE_PROFILE,
        selected_features=list(ACTIVE_SNAPSHOT_META_FEATURES),
        output_path=output_path,
        artifact_health_before=artifact_health,
    )

    path = retrain_cli.write_feature_verification_report(report, output_path)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["source_training_row_count"] == 2
    assert persisted["selected_feature_count"] == len(ACTIVE_SNAPSHOT_META_FEATURES)
    assert persisted["is_value_complete"] is True
    assert persisted["allow_imputation"] is False
    assert persisted["imputed_features"] == {}
    assert persisted["fast_target_contract"]["eligible_rows"] == 2
    assert persisted["active_training_target"]["target_column"] == retrain_cli.FAST_TARGET_LABEL_COLUMN
    assert persisted["artifact_loadability_before"]["can_load"] is False
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_meta_labeler_save_persists_fast_target_contract_metadata(monkeypatch, tmp_path: Path) -> None:
    labeler = MetaLabeler()
    labeler._feature_names = list(ACTIVE_SNAPSHOT_META_FEATURES)
    cast(Any, labeler)._model = {"model": "ok"}
    cast(Any, labeler)._imputer = SimpleNamespace()
    labeler._is_trained = True
    labeler._metrics = MetaLabelerMetrics(
        auc_cv=0.61,
        precision_at_5=0.40,
        f1_threshold=0.50,
        f1_score=0.38,
        feature_importance={feature: 0.0 for feature in ACTIVE_SNAPSHOT_META_FEATURES},
        train_samples=10,
        positive_rate=0.50,
        oof_auc=0.704,
        oof_auc_ci_low=0.641,
        oof_auc_ci_high=0.761,
        oof_ece=0.056,
        oof_deployable_auc=0.681,
        oof_deployable_ece=0.061,
        oof_deployable_n=8,
        oof_deployable_n_pos=4,
        oof_deployable_positive_rate=0.5,
        oof_deployable_definition="capital_fraction > 0",
        n_pos=111,
        promotion_status="pass",
        promotion_reasons=[],
    )

    captured: dict[str, object] = {}
    hmm_dir = tmp_path / "hmm"
    hmm_dir.mkdir(parents=True)
    (hmm_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pipeline_version": "6.5.7",
                "trained_at_utc": "2026-04-23T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    import neutralgrid.models.artifacts as artifacts

    def _fake_save_artifact(**kwargs):
        captured["metadata"] = kwargs["metadata"].to_dict()

    monkeypatch.setattr(artifacts, "save_artifact", _fake_save_artifact)
    monkeypatch.setattr(artifacts, "get_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(artifacts, "resolve_hmm_artifact_dir", lambda: hmm_dir)

    output_path = tmp_path / "models" / "meta_labeler.pkl"
    labeler.save(output_path)

    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    eval_metrics = metadata["eval_metrics"]
    assert eval_metrics["target_contract"] == ACTIVE_META_TARGET_CONTRACT
    assert eval_metrics["target_label_column"] == ACTIVE_META_TARGET_LABEL_COLUMN
    assert eval_metrics["target_duration_column"] == ACTIVE_META_TARGET_DURATION_COLUMN
    assert eval_metrics["target_duration_hours_max"] == ACTIVE_META_TARGET_DURATION_MAX_HOURS
    # FASTWIN-01: the promotion gate must be persisted into eval_metrics so a
    # decision-time consumer can read promotion_status from metadata.json.
    assert eval_metrics["promotion_status"] == "pass"
    assert eval_metrics["oof_auc"] == pytest.approx(0.704)
    assert eval_metrics["oof_auc_ci_low"] == pytest.approx(0.641)
    assert eval_metrics["oof_auc_ci_high"] == pytest.approx(0.761)
    assert eval_metrics["oof_ece"] == pytest.approx(0.056)
    assert eval_metrics["oof_deployable_auc"] == pytest.approx(0.681)
    assert eval_metrics["oof_deployable_ece"] == pytest.approx(0.061)
    assert eval_metrics["oof_deployable_n"] == 8
    assert eval_metrics["oof_deployable_n_pos"] == 4
    assert eval_metrics["oof_deployable_positive_rate"] == pytest.approx(0.5)
    assert eval_metrics["oof_deployable_definition"] == "capital_fraction > 0"
    assert eval_metrics["n_pos"] == 111
    assert eval_metrics["promotion_reasons"] == []


def test_join_snapshots_to_outcomes_backfills_grid_spacing_from_candidate_geometry() -> None:
    builder = UnifiedTrainingBuilder()
    snapshot_df = pd.DataFrame(
        [
            {
                "candidate_id": "BTC_1",
                "grid_spacing_pct": None,
                "range_prob": 0.7,
            }
        ]
    )
    backtest_df = pd.DataFrame(
        [
            {
                "candidate_id": "BTC_1",
                "grid_spacing_pct": 0.45,
                "pnl_pct": 1.0,
            }
        ]
    )

    merged = builder._join_snapshots_to_outcomes(snapshot_df, backtest_df)

    assert len(merged) == 1
    assert merged.loc[0, "grid_spacing_pct"] == pytest.approx(0.45)


def test_derive_utility_score_uses_governed_provisional_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_derive_utility_score` calls compute_governed_provisional_utility under
    the hood with `range_prob`/`trend_prob` only, so both should produce the
    same numeric result when a calibrator artifact is available.

    Per UTILFIX-01, the runtime utility path raises
    `UtilityCalibratorUnavailable` when no artifact is present. We provide
    a minimal valid artifact via `artifact_path` to exercise the canonical
    semantics. The training builder catches the exception and returns None
    when no artifact exists; that branch is covered separately in
    `test_utility_calibrator.py::test_scan_handles_utility_calibrator_unavailable`.
    """
    builder = UnifiedTrainingBuilder()
    row = {
        "range_prob": 0.72,
        "trend_prob": 0.18,
        "profit_per_grid_pct": 9.9,
        "num_grids": 77,
        "range_size_pct": 14.5,
        "survival_prob": 0.91,
        "hurst_exponent": 0.62,
    }

    artifact_path = tmp_path / "current.json"
    active_hmm_version = get_active_hmm_version()
    assert active_hmm_version is not None
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "promotable": True,
                "data_contract": {
                    "hmm_artifact_version": active_hmm_version,
                },
                "coefficients": {
                    "lambda_risk": 2.0,
                    "kappa_trend": 1.5,
                    "horizon_hours": 6.0,
                },
                "threshold": {"min_utility": 0.0},
            }
        ),
        encoding="utf-8",
    )

    expected = compute_governed_provisional_utility(
        range_prob=0.72,
        trend_prob=0.18,
        artifact_path=artifact_path,
    ).utility_score

    def _compute_with_test_artifact(*, range_prob: float, trend_prob: float) -> Any:
        return compute_governed_provisional_utility(
            range_prob=range_prob,
            trend_prob=trend_prob,
            artifact_path=artifact_path,
        )

    monkeypatch.setattr(
        "neutralgrid.training.unified_training_builder.compute_governed_provisional_utility",
        _compute_with_test_artifact,
    )
    derived = builder._derive_utility_score(row)
    assert derived == pytest.approx(expected)


# ── FASTWIN-01 promotion gate ────────────────────────────────────────────────


def test_promotion_gate_passes_when_all_three_criteria_met() -> None:
    """At the verified geometric-pool operating point (OOF AUC 0.704
    [0.641, 0.761], ECE 0.056, n_pos 111) the gate must promote."""
    status, reasons = MetaLabeler._evaluate_promotion_gate(
        oof_auc_ci_low=0.641,
        oof_auc_ci_high=0.761,
        n_pos=111,
        oof_ece=0.056,
    )
    assert status == "pass"
    assert reasons == []


@pytest.mark.parametrize(
    "kwargs, expected_reason_substr",
    [
        # CI lower bound at/under 0.50 -> not a discrimination improvement
        (dict(oof_auc_ci_low=0.48, oof_auc_ci_high=0.70, n_pos=111, oof_ece=0.05),
         "auc_ci_includes_0.50"),
        # CI unavailable (bootstrap fail-closed) -> never promote on absent evidence
        (dict(oof_auc_ci_low=None, oof_auc_ci_high=None, n_pos=111, oof_ece=0.05),
         "auc_ci_unavailable"),
        # too few positive events
        (dict(oof_auc_ci_low=0.60, oof_auc_ci_high=0.80, n_pos=40, oof_ece=0.05),
         "n_pos_lt_70"),
        (dict(oof_auc_ci_low=0.60, oof_auc_ci_high=0.80, n_pos=None, oof_ece=0.05),
         "n_pos_unavailable"),
        # miscalibrated
        (dict(oof_auc_ci_low=0.60, oof_auc_ci_high=0.80, n_pos=111, oof_ece=0.20),
         "ece_gt_0.10"),
        (dict(oof_auc_ci_low=0.60, oof_auc_ci_high=0.80, n_pos=111, oof_ece=None),
         "ece_unavailable"),
    ],
)
def test_promotion_gate_fails_closed(kwargs, expected_reason_substr) -> None:
    status, reasons = MetaLabeler._evaluate_promotion_gate(**kwargs)
    assert status == "fail"
    assert any(expected_reason_substr in r for r in reasons)


def test_promotion_gate_accumulates_all_failing_reasons() -> None:
    status, reasons = MetaLabeler._evaluate_promotion_gate(
        oof_auc_ci_low=None,
        oof_auc_ci_high=None,
        n_pos=10,
        oof_ece=0.9,
    )
    assert status == "fail"
    assert len(reasons) == 3


def test_bootstrap_auc_ci_excludes_half_for_discriminating_scores() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 256
    y = (rng.random(n) < 0.43).astype(int)
    # scores strongly correlated with the label
    p = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.15, n), 1e-6, 1 - 1e-6)
    lo, hi = MetaLabeler._bootstrap_auc_ci(y, p, n_boot=1000, random_state=42)
    assert lo is not None and hi is not None
    assert lo > 0.50  # CI excludes the coin-flip line


def test_bootstrap_auc_ci_fails_closed_on_single_class() -> None:
    import numpy as np

    y = np.ones(50, dtype=int)  # one class only -> AUC undefined
    p = np.linspace(0.1, 0.9, 50)
    lo, hi = MetaLabeler._bootstrap_auc_ci(y, p, n_boot=500, random_state=42)
    assert lo is None and hi is None


def test_is_promoted_reflects_promotion_status() -> None:
    labeler = MetaLabeler()
    assert labeler.promotion_status is None
    assert labeler.is_promoted is False
    labeler.promotion_status = "fail"
    assert labeler.is_promoted is False
    labeler.promotion_status = "pass"
    assert labeler.is_promoted is True


def test_promotion_oof_scores_all_rows_and_discriminates() -> None:
    """The study-faithful gate OOF (_evaluate_promotion_oof) must score every
    row (symbol-grouped, all-rows) and recover discrimination on signal-bearing
    data — the divergence the production CPCV could not on a small symbol-diverse
    pool."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n, d = 240, 6
    z = rng.normal(size=n)
    y = (z > 0.1).astype(int)
    X = np.column_stack([z * (0.9 + 0.05 * i) + rng.normal(0, 0.9, n) for i in range(d)])
    syms = np.array([f"S{i%40}" for i in range(n)])  # 40 symbols
    start = np.arange(n).astype("datetime64[h]")
    t1 = start + np.timedelta64(6, "h")

    def factory():
        return LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)

    ml = MetaLabeler()
    auc, lo, hi, ece = ml._evaluate_promotion_oof(X, y, syms, start, t1, factory, n_boot=400)
    assert auc is not None and lo is not None and hi is not None and ece is not None
    assert auc > 0.6          # genuine signal recovered
    assert lo > 0.50          # CI excludes the coin-flip line
    assert 0.0 <= ece <= 0.5


def test_promotion_oof_reports_deployable_population_without_changing_headline() -> None:
    """ERR-075: the deployable diagnostic shares the governed OOF vector but
    remains separate from the four headline promotion values."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(19)
    n, d = 240, 5
    z = rng.normal(size=n)
    y = (z > 0.0).astype(int)
    X = np.column_stack(
        [z * (0.8 + 0.04 * i) + rng.normal(0, 0.8, n) for i in range(d)]
    )
    syms = np.array([f"S{i % 40}" for i in range(n)])
    start = np.arange(n).astype("datetime64[h]")
    t1 = start + np.timedelta64(6, "h")
    deployable = np.ones(n, dtype=bool)
    deployable[::4] = False

    def factory():
        return LogisticRegression(
            class_weight="balanced", solver="liblinear", random_state=42
        )

    ml = MetaLabeler()
    headline = ml._evaluate_promotion_oof(
        X,
        y,
        syms,
        start,
        t1,
        factory,
        deployable_mask=deployable,
        n_boot=200,
    )

    assert len(headline) == 4
    assert all(value is not None for value in headline)
    diagnostic = ml._last_oof_deployable_metrics
    assert diagnostic is not None
    assert diagnostic["available"] is True
    assert diagnostic["definition"] == "capital_fraction > 0"
    assert diagnostic["n"] == int(deployable.sum())
    assert diagnostic["n_pos"] == int(y[deployable].sum())
    assert 0.0 <= diagnostic["auc"] <= 1.0
    assert 0.0 <= diagnostic["ece"] <= 1.0

    status, reasons = MetaLabeler._evaluate_promotion_gate(
        oof_auc_ci_low=headline[1],
        oof_auc_ci_high=headline[2],
        n_pos=int(y.sum()),
        oof_ece=headline[3],
    )
    assert status == "pass"
    assert reasons == []


def test_promotion_oof_fails_closed_on_tiny_pool() -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X = np.random.RandomState(0).normal(size=(12, 4))
    y = np.array([0, 1] * 6)
    out = MetaLabeler()._evaluate_promotion_oof(
        X, y, None, None, None,
        lambda: LogisticRegression(solver="liblinear", random_state=42),
    )
    assert out == (None, None, None, None)
