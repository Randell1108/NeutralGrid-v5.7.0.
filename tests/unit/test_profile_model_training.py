"""Phase 2 unit tests for profile_model training pipeline.

Covers:
  - 2.1: Availability filter is evaluated on df_labeled, not full df.
  - 2.2: NaN imputation uses shared medians (not class-conditional).
  - 2.3: Standardization produces llr ~ 0 for the class-mean midpoint.
  - 2.4: Per-class sample floor max(30, 3*len(feats)) is enforced.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BASE_NUMERIC_FEATURES = [
    "parkinson_vol_ratio_4h_24h_pre",
    "variance_ratio_1m_15m_pre_2h",
    "funding_carry_expected_next_7h",
    "liquidity_stability_z_1h",
]


def _workspace_tmp() -> Path:
    p = Path.cwd() / ".pytest_tmp" / f"phase2_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def _make_balanced_xlsx(tmp: Path, n_rows: int) -> Path:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "strategy_id": f"bot_{i}",
                "symbol": "BTCUSDT",
                "pnl_pct": 5.0 + i * 0.2,
                "profit_factor": 1.0 + (i % 20) * 0.1,
                "duration_hours": 1.0 + (i % 12) * 0.5,
                "parkinson_vol_ratio_4h_24h_pre": 0.8 + (i % 10) * 0.03,
                "variance_ratio_1m_15m_pre_2h": 0.7 + (i % 10) * 0.02,
                "funding_carry_expected_next_7h": -0.04 + (i % 10) * 0.01,
                "liquidity_stability_z_1h": -1.0 + (i % 10) * 0.25,
            }
        )
    df = pd.DataFrame(rows)
    path = tmp / "balanced.xlsx"
    df.to_excel(str(path), index=False, sheet_name="General")
    return path


def test_availability_filter_uses_labeled_universe():
    """Phase 2.1: feature dense outside df_labeled but sparse inside is dropped."""
    tmp = _workspace_tmp()
    try:
        rows = []
        for i in range(100):
            rows.append(
                {
                    "strategy_id": f"short_{i}",
                    "pnl_pct": 5.0 + i * 0.2,
                    "profit_factor": 1.0 + (i % 20) * 0.1,
                    "duration_hours": 2.0 + (i % 10) * 0.3,
                    "parkinson_vol_ratio_4h_24h_pre": 0.8 + (i % 10) * 0.03,
                    "variance_ratio_1m_15m_pre_2h": 0.7 + (i % 10) * 0.02,
                    "funding_carry_expected_next_7h": -0.04 + (i % 10) * 0.01,
                    "liquidity_stability_z_1h": -1.0 + (i % 10) * 0.25,
                    "sparse_in_labeled": float("nan"),
                }
            )
        for i in range(50):
            rows.append(
                {
                    "strategy_id": f"long_{i}",
                    "pnl_pct": 30.0 + i,
                    "profit_factor": 3.0,
                    "duration_hours": 10.0 + i * 0.5,
                    "parkinson_vol_ratio_4h_24h_pre": 1.4,
                    "variance_ratio_1m_15m_pre_2h": 1.1,
                    "funding_carry_expected_next_7h": 0.05,
                    "liquidity_stability_z_1h": 2.0,
                    "sparse_in_labeled": 99.0,
                }
            )
        df = pd.DataFrame(rows)
        path = tmp / "availability.xlsx"
        df.to_excel(str(path), index=False, sheet_name="General")

        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        model = train_profile_model_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=[*BASE_NUMERIC_FEATURES, "sparse_in_labeled"],
        )
        assert "sparse_in_labeled" not in model.features
        for f in BASE_NUMERIC_FEATURES:
            assert f in model.features
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_default_features_match_four_feature_contract():
    """DEFAULT_FEATURES is the active 4-feature profile contract."""
    from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES

    assert DEFAULT_FEATURES == BASE_NUMERIC_FEATURES


def test_training_prefers_general_even_when_legacy_tabs_exist():
    """When General exists it is the canonical training sheet, even beside legacy tabs."""
    tmp = _workspace_tmp()
    try:
        path = _make_balanced_xlsx(tmp, 120)
        with pd.ExcelWriter(path, engine="openpyxl", mode="a") as writer:
            legacy = pd.DataFrame({"strategy_id": ["dup", "dup"]})
            legacy.to_excel(writer, index=False, sheet_name="Entry Validation Metrics")
            legacy.to_excel(writer, index=False, sheet_name="Performance Risk-Adjusted")
            legacy.to_excel(writer, index=False, sheet_name="Market and Volatility")

        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        model = train_profile_model_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=BASE_NUMERIC_FEATURES,
        )
        assert model.features == BASE_NUMERIC_FEATURES
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_shared_median_imputation_no_class_bias():
    """Phase 2.2: shared imputation does not inflate mu_w - mu_l from NaN pattern."""
    tmp = _workspace_tmp()
    try:
        rows = []
        for i in range(50):
            rows.append(
                {
                    "strategy_id": f"w_{i}",
                    "pnl_pct": 30.0 + i * 0.1,
                    "profit_factor": 2.5 + (i % 10) * 0.05,
                    "duration_hours": 2.0 + (i % 10) * 0.3,
                    "parkinson_vol_ratio_4h_24h_pre": 1.1 + (i % 10) * 0.02,
                    "variance_ratio_1m_15m_pre_2h": 0.8 + (i % 10) * 0.02,
                    "funding_carry_expected_next_7h": -0.01 + (i % 10) * 0.002,
                    "liquidity_stability_z_1h": float("nan"),
                }
            )
        for i in range(50):
            rows.append(
                {
                    "strategy_id": f"l_{i}",
                    "pnl_pct": 5.0 + i * 0.05,
                    "profit_factor": 1.0 + (i % 5) * 0.02,
                    "duration_hours": 2.0 + (i % 10) * 0.3,
                    "parkinson_vol_ratio_4h_24h_pre": 0.7 + (i % 10) * 0.01,
                    "variance_ratio_1m_15m_pre_2h": 1.0 + (i % 10) * 0.01,
                    "funding_carry_expected_next_7h": 0.02 + (i % 10) * 0.001,
                    "liquidity_stability_z_1h": 0.5,
                }
            )
        df = pd.DataFrame(rows)
        path = tmp / "bias.xlsx"
        df.to_excel(str(path), index=False, sheet_name="General")

        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        model = train_profile_model_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=BASE_NUMERIC_FEATURES,
        )
        assert "liquidity_stability_z_1h" in model.winner_mu
        assert "liquidity_stability_z_1h" in model.loser_mu
        assert (
            abs(
                model.winner_mu["liquidity_stability_z_1h"]
                - model.loser_mu["liquidity_stability_z_1h"]
            )
            < 1e-9
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_standardization_midpoint_yields_zero_llr():
    """Phase 2.3: midpoint row yields llr ~= log(prior/(1-prior))."""
    tmp = _workspace_tmp()
    try:
        path = _make_balanced_xlsx(tmp, 120)

        from neutralgrid.scanner.profile_model import (
            load_profile_model,
            save_profile_model,
            train_profile_model_from_enhanced_xlsx,
        )

        model = train_profile_model_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=BASE_NUMERIC_FEATURES,
        )
        assert model.feature_mean is not None
        assert model.feature_std is not None
        for f in model.features:
            assert f in model.feature_mean
            assert f in model.feature_std

        assert model.selection_summary is not None
        assert model.selection_summary["top_quantile"] == pytest.approx(0.5)
        out = tmp / "model_with_summary.json"
        save_profile_model(model, out)
        reloaded = load_profile_model(out)
        assert reloaded.selection_summary == model.selection_summary
        assert reloaded.feature_impute == model.feature_impute

        midpoint_std = {
            f: (model.winner_mu[f] + model.loser_mu[f]) / 2.0 for f in model.features
        }
        raw_midpoint = {
            f: midpoint_std[f] * model.feature_std[f] + model.feature_mean[f]
            for f in model.features
        }
        llr = model.llr(raw_midpoint)
        prior = max(0.001, min(0.999, model.prior_winner))
        expected = float(np.log(prior / (1.0 - prior)))
        assert llr is not None
        assert abs(llr - expected) < 1e-9
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_inference_uses_persisted_shared_impute():
    """Missing inference values must use the persisted shared-impute vector."""
    tmp = _workspace_tmp()
    try:
        path = _make_balanced_xlsx(tmp, 120)

        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        model = train_profile_model_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=BASE_NUMERIC_FEATURES,
        )
        assert model.feature_impute is not None

        missing_feature = BASE_NUMERIC_FEATURES[0]
        row = {f: model.feature_mean[f] for f in model.features}
        expected = dict(row)
        expected[missing_feature] = model.feature_impute[missing_feature]
        row[missing_feature] = float("nan")

        llr_missing = model.llr(row)
        llr_expected = model.llr(expected)
        assert llr_missing is not None
        assert llr_expected is not None
        assert abs(llr_missing - llr_expected) < 1e-12
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sample_floor_raises_for_small_n():
    """Phase 2.4: per-class floor max(30, 3*len(feats)) raises on small-n."""
    tmp = _workspace_tmp()
    try:
        path = _make_balanced_xlsx(tmp, 20)
        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        with pytest.raises(ValueError, match=r"need >= max\(30"):
            train_profile_model_from_enhanced_xlsx(
                path,
                max_duration_hours=7.0,
                min_profit_factor=1.0,
                top_quantile=0.5,
                features=BASE_NUMERIC_FEATURES,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pattern_profile_filters_sparse_features_on_labeled_universe():
    """PatternProfile must drop features that fail the labeled-universe availability floor."""
    tmp = _workspace_tmp()
    try:
        rows = []
        for i in range(100):
            rows.append(
                {
                    "strategy_id": f"short_{i}",
                    "pnl_pct": 5.0 + i * 0.2,
                    "profit_factor": 1.0 + (i % 20) * 0.1,
                    "duration_hours": 2.0 + (i % 10) * 0.3,
                    "parkinson_vol_ratio_4h_24h_pre": 0.8 + (i % 10) * 0.03,
                    "variance_ratio_1m_15m_pre_2h": 0.7 + (i % 10) * 0.02,
                    "funding_carry_expected_next_7h": -0.04 + (i % 10) * 0.01,
                    "liquidity_stability_z_1h": -1.0 + (i % 10) * 0.25,
                    "sparse_in_labeled": float("nan"),
                }
            )
        for i in range(50):
            rows.append(
                {
                    "strategy_id": f"long_{i}",
                    "pnl_pct": 30.0 + i,
                    "profit_factor": 3.0,
                    "duration_hours": 10.0 + i * 0.5,
                    "parkinson_vol_ratio_4h_24h_pre": 1.4,
                    "variance_ratio_1m_15m_pre_2h": 1.1,
                    "funding_carry_expected_next_7h": 0.05,
                    "liquidity_stability_z_1h": 2.0,
                    "sparse_in_labeled": 99.0,
                }
            )
        df = pd.DataFrame(rows)
        path = tmp / "pattern_availability.xlsx"
        df.to_excel(str(path), index=False, sheet_name="General")

        from neutralgrid.scanner.pattern_profile import build_profile_from_enhanced_xlsx

        profile = build_profile_from_enhanced_xlsx(
            path,
            max_duration_hours=7.0,
            min_profit_factor=1.0,
            top_quantile=0.5,
            features=[*BASE_NUMERIC_FEATURES, "sparse_in_labeled"],
        )
        assert "sparse_in_labeled" not in profile.features
        assert set(profile.features) == set(BASE_NUMERIC_FEATURES)
        assert set(profile.means) == set(BASE_NUMERIC_FEATURES)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_read_single_sheet_prefers_general_when_present():
    """Single-sheet reader uses General explicitly when that sheet exists."""
    tmp = _workspace_tmp()
    try:
        path = tmp / "general_first.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"value": [1]}).to_excel(writer, index=False, sheet_name="Other")
            pd.DataFrame({"value": [2]}).to_excel(writer, index=False, sheet_name="General")

        from neutralgrid.scanner._xlsx_io import read_single_sheet

        df, sheet_name = read_single_sheet(path)
        assert sheet_name == "General"
        assert int(df.iloc[0]["value"]) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
