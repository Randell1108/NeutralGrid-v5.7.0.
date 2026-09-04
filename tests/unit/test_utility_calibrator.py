from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neutralgrid.calibration.utility_calibrator import (
    ELIGIBLE_UNIVERSE_RULE,
    LABEL_COL,
    _load_calibration_pool,
    _select_threshold_fit_only,
    _split_pool,
    _utility_scores,
    build_cli_proof_summary,
    parse_args,
    run_calibration,
)
from neutralgrid.validation.utility import (
    UtilityCalibratorUnavailable,
    UtilityConfig,
    UtilityScorer,
)

TEST_TMP_DIR = Path(__file__).with_name("utility_calibrator_tmp")
TEST_HMM_VERSION = "rolling_180d_20260101_000000"


def _test_path(name: str) -> Path:
    return TEST_TMP_DIR / name


def _base_workbook_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_ids = list(range(100, 108))
    outcomes = pd.DataFrame(
        {
            "strategy_id": strategy_ids,
            "symbol": [f"T{i}USDT" for i in range(8)],
            "pnl_pct": [1.0, -0.3, 0.5, 0.0, 1.2, -0.8, 0.7, -0.2],
            "duration_hours": [6.0] * 8,
            "start_time_utc": pd.date_range("2026-01-01", periods=8, freq="h"),
            "range_size_pct": [999.0] * 8,
            "utility_score": [-999.0] * 8,
        }
    )
    meta = pd.DataFrame(
        {
            "strategy_id": strategy_ids,
            "backfill_status": ["ok"] * 8,
            "num_grids": [20] * 8,
            "range_prob": [0.7] * 8,
            "trend_prob": [0.2] * 8,
            "persistence_prob": [0.8] * 8,
            "hmm_artifact_version": [TEST_HMM_VERSION] * 8,
            "profit_per_grid_pct": [0.8] * 8,
            "range_size_pct": [3.0] * 8,
        }
    )
    return outcomes, meta


def _write_workbook(path: Path, outcomes: pd.DataFrame, meta: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        outcomes.to_excel(writer, sheet_name="General", index=False)
        meta.to_excel(writer, sheet_name="Meta Features", index=False)


def _flat_backfill_frame() -> pd.DataFrame:
    """Single-sheet flat layout matching scripts/backfill_training_features.py output.

    Mirrors the two-sheet `_base_workbook_frames` content joined on strategy_id,
    plus the lineage/source columns the loader uses to derive backfill_status.
    """
    strategy_ids = list(range(100, 108))
    return pd.DataFrame(
        {
            "strategy_id": strategy_ids,
            "symbol": [f"T{i}USDT" for i in range(8)],
            "pnl_pct": [1.0, -0.3, 0.5, 0.0, 1.2, -0.8, 0.7, -0.2],
            "duration_hours": [6.0] * 8,
            "start_time_utc": pd.date_range("2026-01-01", periods=8, freq="h"),
            "num_grids": [20] * 8,
            "range_prob": [0.7] * 8,
            "trend_prob": [0.2] * 8,
            "persistence_prob": [0.8] * 8,
            "hmm_artifact_version": [TEST_HMM_VERSION] * 8,
            "profit_per_grid_pct": [0.8] * 8,
            "range_size_pct": [3.0] * 8,
            "hmm_feature_cutoff_utc": pd.date_range("2026-01-01", periods=8, freq="h"),
            "hmm_feature_source": ["pinned_artifact_replay"] * 8,
        }
    )


def _write_flat_backfill_workbook(path: Path, df: pd.DataFrame) -> None:
    df.to_excel(path, index=False)


def test_from_artifact_raises_when_artifact_missing() -> None:
    """Fail-closed: missing current.json raises UtilityCalibratorUnavailable.

    Replaces the prior silent v0 fallback. Per UTILFIX-01, the runtime path
    must surface the unavailable-calibrator condition explicitly so callers
    can decide whether to fail-closed (decision-time) or emit
    utility_score=None/NaN (offline). The fallback constants remain available
    for the calibrator's internal G6 baseline (direct UtilityConfig(...)),
    which is unaffected by this change.
    """
    missing = _test_path("missing.json")
    assert not missing.exists()

    with pytest.raises(UtilityCalibratorUnavailable) as exc_info:
        UtilityConfig.from_artifact(missing)

    assert str(missing) in str(exc_info.value)
    assert exc_info.value.error_code == "UTILITY_CALIBRATOR_UNAVAILABLE"


def test_from_artifact_raises_on_malformed_json() -> None:
    """Schema validation: malformed payload also fail-closes."""
    bad_path = _test_path("malformed.json")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not valid json", encoding="utf-8")
    try:
        with pytest.raises(UtilityCalibratorUnavailable):
            UtilityConfig.from_artifact(bad_path)
    finally:
        bad_path.unlink(missing_ok=True)


def test_from_artifact_raises_when_required_sections_missing() -> None:
    """Schema validation: missing 'coefficients' / 'threshold' fails closed."""
    incomplete = _test_path("incomplete.json")
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.write_text(json.dumps({"coefficients": {}}), encoding="utf-8")
    try:
        with pytest.raises(UtilityCalibratorUnavailable):
            UtilityConfig.from_artifact(incomplete)
    finally:
        incomplete.unlink(missing_ok=True)


def test_scan_handles_utility_calibrator_unavailable(monkeypatch) -> None:
    """Stage A scan emits utility_score=None when calibrator artifact missing.

    Fail-closed-soft: scan-time feature snapshot must not crash when the
    runtime utility calibrator is unavailable. It logs once and emits
    `utility_score=None` so downstream consumers see the
    feature-missing signal explicitly.
    """
    from neutralgrid.scanner import scan as scan_module

    def _raise_unavailable(*_args, **_kwargs):
        raise UtilityCalibratorUnavailable("/tmp/missing.json", "test fixture")

    monkeypatch.setattr(
        scan_module, "compute_governed_provisional_utility", _raise_unavailable
    )
    # Reset the once-flag so the test sees a fresh emission.
    monkeypatch.setattr(scan_module, "_utility_unavailable_warning_emitted", False)

    # Direct invocation of the inner block: simulate the try/except path.
    # We cannot easily call _compute_stochastic_features without a full df,
    # so we construct a minimal frame and assert the function exits cleanly.
    closes = np.linspace(100.0, 110.0, 400)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "volume": np.full(400, 1000.0),
        }
    )
    out = scan_module._compute_stochastic_features(  # type: ignore[attr-defined]
        df, range_prob=0.7, trend_prob=0.2, range_size_pct=3.0
    )
    # When the calibrator path raises, scan must emit utility_score=None
    # rather than dropping the key or substituting v0.
    assert out.get("utility_score") is None


def test_from_artifact_loads_coefficients_and_explicit_overrides() -> None:
    artifact_path = _test_path("current.json")
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "promotable": True,
                "data_contract": {
                    "hmm_artifact_version": TEST_HMM_VERSION,
                },
                "coefficients": {
                    "lambda_risk": 0.7,
                    "kappa_trend": 0.8,
                    "horizon_hours": 7.5,
                },
                "threshold": {"min_utility": -1.25},
            }
        ),
        encoding="utf-8",
    )

    config = UtilityConfig.from_artifact(
        artifact_path,
        expected_hmm_artifact_version=TEST_HMM_VERSION,
        lambda_risk=2.0,
        min_utility=0.0,
    )

    assert config.lambda_risk == 2.0
    assert config.kappa_trend == 0.8
    assert config.horizon_hours == 7.5
    assert config.min_utility == 0.0

    with pytest.raises(TypeError, match="Unknown UtilityConfig override"):
        UtilityConfig.from_artifact(
            artifact_path,
            expected_hmm_artifact_version=TEST_HMM_VERSION,
            take_profit_pct_shadow=10.0,
        )


def test_from_artifact_rejects_missing_hmm_lineage_contract() -> None:
    artifact_path = _test_path("missing_lineage.json")
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotable": True,
                "coefficients": {
                    "lambda_risk": 0.7,
                    "kappa_trend": 0.8,
                    "horizon_hours": 7.5,
                },
                "threshold": {"min_utility": -1.25},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UtilityCalibratorUnavailable, match="schema_version"):
        UtilityConfig.from_artifact(
            artifact_path,
            expected_hmm_artifact_version=TEST_HMM_VERSION,
        )


def test_from_artifact_rejects_hmm_lineage_mismatch() -> None:
    artifact_path = _test_path("stale_lineage.json")
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "promotable": True,
                "data_contract": {
                    "hmm_artifact_version": "rolling_180d_20260102_000000",
                },
                "coefficients": {
                    "lambda_risk": 0.7,
                    "kappa_trend": 0.8,
                    "horizon_hours": 7.5,
                },
                "threshold": {"min_utility": -1.25},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UtilityCalibratorUnavailable, match="HMM lineage mismatch"):
        UtilityConfig.from_artifact(
            artifact_path,
            expected_hmm_artifact_version=TEST_HMM_VERSION,
        )


def test_from_artifact_rejects_nonpromotable_candidate() -> None:
    artifact_path = _test_path("nonpromotable.json")
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "promotable": False,
                "data_contract": {
                    "hmm_artifact_version": TEST_HMM_VERSION,
                },
                "coefficients": {
                    "lambda_risk": 0.7,
                    "kappa_trend": 0.8,
                    "horizon_hours": 7.5,
                },
                "threshold": {"min_utility": -1.25},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UtilityCalibratorUnavailable, match="not promotable"):
        UtilityConfig.from_artifact(
            artifact_path,
            expected_hmm_artifact_version=TEST_HMM_VERSION,
        )


def test_loader_uses_two_sheets_retains_losers_and_ignores_stored_utility(
) -> None:
    outcomes, meta = _base_workbook_frames()
    workbook = _test_path("expired.xlsx")
    _write_workbook(workbook, outcomes, meta)

    pool = _load_calibration_pool(workbook)

    assert len(pool) == 8
    assert set(pool[LABEL_COL]) == {0, 1}
    assert int(pool.loc[pool["pnl_pct"].eq(0.0), LABEL_COL].iloc[0]) == 0
    assert pool["range_size_pct"].eq(3.0).all()
    assert "utility_score" not in pool.columns
    assert {"survival_prob", "hurst_exponent"}.issubset(pool.columns)
    assert pool["survival_prob"].isna().all()
    coverage = [
        item
        for item in run_calibration(
            expired_bots_path=workbook,
            artifact_dir=TEST_TMP_DIR / "artifacts_loader",
            write_candidate=False,
            promote=False,
            expected_hmm_artifact_version=TEST_HMM_VERSION,
        ).artifact["pool_feature_coverage"]
        if item["feature"] == "num_grids"
    ]
    assert len(coverage) == 1
    assert coverage[0]["non_null_count"] == 8


def test_loader_includes_7h_boundary_and_excludes_gt_7h() -> None:
    outcomes, meta = _base_workbook_frames()
    outcomes.loc[0, "duration_hours"] = 7.0
    outcomes.loc[1, "duration_hours"] = 7.1
    workbook = _test_path("duration_boundary.xlsx")
    _write_workbook(workbook, outcomes, meta)

    pool = _load_calibration_pool(workbook)

    assert 100 in pool["strategy_id"].tolist()
    assert 101 not in pool["strategy_id"].tolist()


def test_loader_rejects_duplicate_strategy_ids() -> None:
    outcomes, meta = _base_workbook_frames()
    outcomes.loc[1, "strategy_id"] = outcomes.loc[0, "strategy_id"]
    workbook = _test_path("duplicate.xlsx")
    _write_workbook(workbook, outcomes, meta)

    with pytest.raises(ValueError, match="Duplicate strategy_id in General"):
        _load_calibration_pool(workbook)

    outcomes, meta = _base_workbook_frames()
    meta.loc[1, "strategy_id"] = meta.loc[0, "strategy_id"]
    meta.loc[0, "start_time_utc"] = "2026-01-02T00:00:00+00:00"
    meta.loc[1, "start_time_utc"] = None
    meta.loc[0, "range_prob"] = 0.9
    meta.loc[1, "range_prob"] = 0.1
    workbook = _test_path("duplicate_meta.xlsx")
    _write_workbook(workbook, outcomes, meta)

    pool = _load_calibration_pool(workbook)

    assert len(pool) == 7
    assert pool.loc[pool["strategy_id"].eq(100), "range_prob"].iloc[0] == pytest.approx(0.9)


def test_loader_rejects_blank_ids_and_missing_status() -> None:
    outcomes, meta = _base_workbook_frames()
    outcomes["strategy_id"] = outcomes["strategy_id"].astype(object)
    outcomes.loc[0, "strategy_id"] = ""
    workbook = _test_path("blank_id.xlsx")
    _write_workbook(workbook, outcomes, meta)

    with pytest.raises(ValueError, match="Blank strategy_id in General"):
        _load_calibration_pool(workbook)

    outcomes, meta = _base_workbook_frames()
    meta.loc[0, "backfill_status"] = ""
    workbook = _test_path("missing_status.xlsx")
    _write_workbook(workbook, outcomes, meta)

    with pytest.raises(ValueError, match="Missing backfill_status"):
        _load_calibration_pool(workbook)


def test_split_rule_is_deterministic_and_keeps_both_classes() -> None:
    outcomes, meta = _base_workbook_frames()
    workbook = _test_path("split.xlsx")
    _write_workbook(workbook, outcomes, meta)

    pool = _load_calibration_pool(workbook)
    fit_df, holdout_df = _split_pool(pool)

    assert len(fit_df) == 6
    assert len(holdout_df) == 2
    assert set(fit_df[LABEL_COL]) == {0, 1}
    assert set(holdout_df[LABEL_COL]) == {0, 1}
    assert fit_df["strategy_id"].tolist() == [100, 101, 102, 103, 104, 105]


def test_vectorized_scores_match_utility_scorer() -> None:
    outcomes, meta = _base_workbook_frames()
    meta["survival_prob"] = [0.6] * 8
    meta["hurst_exponent"] = [0.55] * 8
    workbook = _test_path("vectorized.xlsx")
    _write_workbook(workbook, outcomes, meta)
    pool = _load_calibration_pool(workbook)
    config = UtilityConfig(lambda_risk=1.2, kappa_trend=0.9, horizon_hours=7.0)

    vector_score = _utility_scores(pool.head(1), config)[0]
    row = pool.iloc[0]
    scalar_score = UtilityScorer(config).compute_utility(
        range_prob=float(row["range_prob"]),
        trend_prob=float(row["trend_prob"]),
        profit_per_grid_pct=float(row["profit_per_grid_pct"]),
        num_grids=int(row["num_grids"]),
        range_size_pct=float(row["range_size_pct"]),
        survival_prob=float(row["survival_prob"]),
        hurst_exponent=float(row["hurst_exponent"]),
    ).utility_score

    assert vector_score == pytest.approx(scalar_score)


def test_threshold_selection_is_fit_only() -> None:
    y_fit = np.array([0, 0, 1, 1])
    fit_scores = np.array([-1.0, -0.5, 0.5, 1.0])

    selection = _select_threshold_fit_only(y_fit, fit_scores)

    assert selection.threshold == pytest.approx(0.5)
    assert selection.balanced_accuracy == pytest.approx(1.0)
    assert selection.winner_recall == pytest.approx(1.0)
    assert selection.loser_recall == pytest.approx(1.0)


def test_rejected_candidate_writes_audit_artifact_and_leaves_current_absent(
) -> None:
    outcomes, meta = _base_workbook_frames()
    workbook = _test_path("rejected.xlsx")
    artifact_dir = TEST_TMP_DIR / "artifacts"
    current_path = artifact_dir / "current.json"
    if current_path.exists():
        current_path.unlink()
    _write_workbook(workbook, outcomes, meta)

    result = run_calibration(
        expired_bots_path=workbook,
        artifact_dir=artifact_dir,
        write_candidate=True,
        promote=True,
        expected_hmm_artifact_version=TEST_HMM_VERSION,
    )

    assert result.candidate_path is not None
    assert result.candidate_path.exists()
    assert not result.current_path.exists()
    assert result.artifact["promotable"] is False
    assert result.artifact["gates"]["G3_holdout_auc_gt_0_5"] is False
    proof = build_cli_proof_summary(result)
    assert proof["eligible_universe"] == ELIGIBLE_UNIVERSE_RULE
    assert proof["pool_rows"] == 8
    assert proof["fit_rows"] == 6
    assert proof["holdout_rows"] == 2
    assert proof["positive_count"] == 4
    assert proof["negative_count"] == 4
    assert proof["promotable"] is False
    assert proof["current_json_updated"] is False


def test_parse_args_accepts_cli_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "utility_calibrator.py",
            "--input",
            "custom.xlsx",
            "--artifact-dir",
            "artifacts/utility",
            "--promote",
            "--skip-candidate-write",
        ],
    )

    args = parse_args()

    assert args.input == "custom.xlsx"
    assert args.artifact_dir == "artifacts/utility"
    assert args.promote is True
    assert args.skip_candidate_write is True


def test_parse_args_defaults_to_canonical_workbook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["utility_calibrator.py"])

    args = parse_args()

    assert args.input == "data/new_expired_bots.xlsx"


def test_canonical_workbook_exposes_governed_utility_pool() -> None:
    workbook = Path("data/new_expired_bots.xlsx")
    if not workbook.exists():
        pytest.skip("workspace workbook not available")

    manifest = json.loads(Path("artifact_manifest.json").read_text(encoding="utf-8"))
    active_hmm = str(manifest["hmm"]["active_version"])
    pool = _load_calibration_pool(
        workbook,
        expected_hmm_artifact_version=active_hmm,
    )

    assert not pool.empty
    assert set(pool["hmm_artifact_version"].astype(str)) == {active_hmm}
    assert pool[LABEL_COL].nunique() == 2
    for column in ("range_prob", "trend_prob", "persistence_prob"):
        assert np.isfinite(pd.to_numeric(pool[column], errors="coerce")).all()


def test_producer_paths_use_artifact_loader() -> None:
    root = Path(__file__).resolve().parents[2]
    producer_paths = [
        root / "src" / "neutralgrid" / "scanner" / "scan.py",
        root / "src" / "neutralgrid" / "validation" / "regime_validator.py",
        root / "src" / "neutralgrid" / "training" / "unified_training_builder.py",
        root / "scripts" / "backfill_training_features.py",
        root / "src" / "neutralgrid" / "validation" / "utility.py",
    ]

    for path in producer_paths:
        source = path.read_text(encoding="utf-8")
        assert (
            "UtilityConfig.from_artifact" in source
            or "compute_governed_provisional_utility" in source
        )
        assert "UtilityScorer(UtilityConfig(" not in source
        assert "config or UtilityConfig()" not in source


def test_calibrator_does_not_use_live_outcome_ingestor() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "src" / "neutralgrid" / "calibration" / "utility_calibrator.py"
    ).read_text(encoding="utf-8")

    assert "live_outcome_ingestor import" not in source
    assert "LiveOutcomeIngestor(" not in source


def test_loader_accepts_flat_backfill_workbook() -> None:
    workbook = _test_path("flat.xlsx")
    _write_flat_backfill_workbook(workbook, _flat_backfill_frame())

    pool = _load_calibration_pool(workbook)

    assert len(pool) == 8
    assert set(pool[LABEL_COL]) == {0, 1}
    # Flat layout derives backfill_status from hmm_feature_source.
    assert pool["backfill_status"].eq("complete").all()


def test_loader_excludes_missing_outcome_without_imputing_a_label() -> None:
    df = _flat_backfill_frame()
    df.loc[0, "pnl_pct"] = np.nan
    workbook = _test_path("flat_missing_outcome.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    pool = _load_calibration_pool(workbook)

    assert len(pool) == 7
    assert 100 not in set(pool["strategy_id"])
    assert pool["pnl_pct"].notna().all()


def test_loader_rejects_nonblank_nonnumeric_outcome() -> None:
    df = _flat_backfill_frame()
    df["pnl_pct"] = df["pnl_pct"].astype("object")
    df.loc[0, "pnl_pct"] = "not-a-number"
    workbook = _test_path("flat_invalid_outcome.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    with pytest.raises(ValueError, match="Missing or non-numeric utility inputs"):
        _load_calibration_pool(workbook)


def test_loader_aliases_grids_count_to_num_grids() -> None:
    """Backfill output preserves grids_count from the main extractor sheet but
    the utility calibrator's REQUIRED_FEATURE_COLS uses num_grids; the flat
    loader must alias one to the other."""
    df = _flat_backfill_frame().drop(columns=["num_grids"])
    df["grids_count"] = [20] * 8
    workbook = _test_path("flat_grids_count_alias.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    pool = _load_calibration_pool(workbook)

    assert len(pool) == 8
    assert pool["num_grids"].eq(20).all()


def test_loader_excludes_rows_with_failed_hmm_feature_source() -> None:
    """Rows whose hmm_feature_source is not 'pinned_artifact_replay' map to
    backfill_status='hmm_failed' and are filtered out of the calibration pool."""
    df = _flat_backfill_frame()
    df.loc[0:2, "hmm_feature_source"] = "artifact_unavailable"
    workbook = _test_path("flat_partial_failure.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    pool = _load_calibration_pool(workbook)

    # Three rows mapped to hmm_failed and were excluded; five remain.
    assert len(pool) == 5
    assert pool["backfill_status"].eq("complete").all()
    assert not pool["strategy_id"].isin([100, 101, 102]).any()


def test_loader_rejects_mixed_hmm_lineage() -> None:
    df = _flat_backfill_frame()
    df.loc[0, "hmm_artifact_version"] = "rolling_180d_20260102_000000"
    workbook = _test_path("flat_mixed_hmm_lineage.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    with pytest.raises(ValueError, match="Mixed HMM lineage"):
        _load_calibration_pool(workbook)


def test_loader_rejects_stale_hmm_lineage_against_expected_version() -> None:
    workbook = _test_path("flat_stale_hmm_lineage.xlsx")
    _write_flat_backfill_workbook(workbook, _flat_backfill_frame())

    with pytest.raises(ValueError, match="HMM lineage mismatch"):
        _load_calibration_pool(
            workbook,
            expected_hmm_artifact_version="rolling_180d_20260102_000000",
        )


def test_loader_rejects_nonfinite_hmm_probability() -> None:
    df = _flat_backfill_frame()
    df.loc[0, "persistence_prob"] = np.nan
    workbook = _test_path("flat_nonfinite_hmm_probability.xlsx")
    _write_flat_backfill_workbook(workbook, df)

    with pytest.raises(ValueError, match="Non-finite HMM probabilities"):
        _load_calibration_pool(workbook)
