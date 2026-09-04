"""Unit tests for ProfileModel.selection_summary label-policy provenance."""
from __future__ import annotations

import json
import shutil
import uuid
import warnings
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

EXPECTED_KEYS = {
    "xlsx",
    "top_quantile",
    "min_profit_factor",
    "min_avg_profit_per_grid",
    "max_duration_hours",
    "duration_band",
    "pnl_threshold",
    "winners_count",
    "losers_count",
    "bounded_universe_size",
    "labeled_universe_size",
    "winners_symbols",
    "shrinkage",
    "label_name",
    "label_definition",
}


def _workspace_tmp() -> Path:
    p = Path.cwd() / ".pytest_tmp" / f"sel_summary_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def _make_balanced_xlsx(tmp: Path, n_rows: int) -> Path:
    rows = []
    for i in range(n_rows):
        rows.append(
            {
                "strategy_id": f"bot_{i}",
                "symbol": f"SYM{i % 5}USDT",
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


def test_train_populates_selection_summary_on_success():
    """Successful training returns a ProfileModel with audit provenance."""
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
        summary = model.selection_summary
        assert summary is not None
        assert set(summary.keys()) == EXPECTED_KEYS
        assert summary["top_quantile"] == pytest.approx(0.5)
        assert summary["min_profit_factor"] == pytest.approx(1.0)
        assert summary["max_duration_hours"] == pytest.approx(7.0)
        assert summary["duration_band"] == {"min_hours": 0.0, "max_hours": 7.0}
        assert summary["shrinkage"] == pytest.approx(0.30)
        assert summary["label_name"] == "fast_completed_winner_under_7h"
        assert summary["label_definition"]["duration_rule"] == "0 <= duration_hours < 7.0"
        assert summary["xlsx"].endswith("balanced.xlsx")
        assert summary["winners_count"] >= 30
        assert summary["losers_count"] >= 30
        assert summary["labeled_universe_size"] <= summary["bounded_universe_size"]
        assert isinstance(summary["winners_symbols"], list)
        assert all(isinstance(s, str) for s in summary["winners_symbols"])
        assert len(summary["winners_symbols"]) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selection_summary_roundtrip():
    """Training -> save_profile_model -> load_profile_model preserves the summary."""
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
        out = tmp / "roundtrip_profile_model.json"
        save_profile_model(model, out)

        raw = json.loads(out.read_text(encoding="utf-8"))
        assert "selection_summary" in raw
        assert set(raw["selection_summary"].keys()) == EXPECTED_KEYS

        reloaded = load_profile_model(out)
        assert reloaded.selection_summary == model.selection_summary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_legacy_profile_model_without_selection_summary():
    """Pre-Phase artifacts lacking selection_summary load as None."""
    from neutralgrid.scanner.profile_model import load_profile_model

    legacy = {
        "features": [
            "parkinson_vol_ratio_4h_24h_pre",
            "funding_carry_expected_next_7h",
        ],
        "winner_mu": {
            "parkinson_vol_ratio_4h_24h_pre": 0.1,
            "funding_carry_expected_next_7h": 0.2,
        },
        "loser_mu": {
            "parkinson_vol_ratio_4h_24h_pre": -0.1,
            "funding_carry_expected_next_7h": -0.2,
        },
        "inv_cov": [[1.0, 0.0], [0.0, 1.0]],
        "prior_winner": 0.4,
        "duration_band": {"min_hours": 0.0, "max_hours": 7.0},
        "feature_mean": {
            "parkinson_vol_ratio_4h_24h_pre": 1.0,
            "funding_carry_expected_next_7h": 0.0,
        },
        "feature_std": {
            "parkinson_vol_ratio_4h_24h_pre": 0.2,
            "funding_carry_expected_next_7h": 0.05,
        },
    }
    tmp = _workspace_tmp()
    try:
        path = tmp / "legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")

        model = load_profile_model(path)
        assert model.selection_summary is None
        assert model.features == [
            "parkinson_vol_ratio_4h_24h_pre",
            "funding_carry_expected_next_7h",
        ]
        assert model.prior_winner == pytest.approx(0.4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_profile_model_rejects_unknown_feature_contract():
    """Artifacts carrying features outside the active 4-feature contract fail closed."""
    from neutralgrid.scanner.profile_model import load_profile_model

    tmp = _workspace_tmp()
    try:
        path = tmp / "invalid_contract.json"
        path.write_text(
            json.dumps(
                {
                    "features": ["trend_structure"],
                    "winner_mu": {"trend_structure": 0.1},
                    "loser_mu": {"trend_structure": -0.1},
                    "inv_cov": [[1.0]],
                    "prior_winner": 0.5,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown features"):
            load_profile_model(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_profile_model_rejects_invalid_matrix_shape():
    """Malformed artifacts fail at load time instead of crashing during scoring."""
    from neutralgrid.scanner.profile_model import load_profile_model

    tmp = _workspace_tmp()
    try:
        path = tmp / "invalid_matrix.json"
        path.write_text(
            json.dumps(
                {
                    "features": BASE_NUMERIC_FEATURES[:2],
                    "winner_mu": {f: 0.1 for f in BASE_NUMERIC_FEATURES[:2]},
                    "loser_mu": {f: -0.1 for f in BASE_NUMERIC_FEATURES[:2]},
                    "inv_cov": [[1.0]],
                    "prior_winner": 0.5,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="inv_cov shape"):
            load_profile_model(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_profile_model_rejects_nonfinite_parameters():
    """NaN/Inf model parameters cannot enter the runtime scoring path."""
    from neutralgrid.scanner.profile_model import load_profile_model

    tmp = _workspace_tmp()
    try:
        path = tmp / "nonfinite.json"
        path.write_text(
            json.dumps(
                {
                    "features": [BASE_NUMERIC_FEATURES[0]],
                    "winner_mu": {BASE_NUMERIC_FEATURES[0]: float("nan")},
                    "loser_mu": {BASE_NUMERIC_FEATURES[0]: -0.1},
                    "inv_cov": [[1.0]],
                    "prior_winner": 0.5,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-finite"):
            load_profile_model(path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_profile_probability_is_stable_for_extreme_negative_llr():
    """The logistic transform must not overflow on extreme valid scores."""
    from neutralgrid.scanner.profile_model import ProfileModel

    feature = BASE_NUMERIC_FEATURES[0]
    model = ProfileModel(
        features=[feature],
        winner_mu={feature: 1000.0},
        loser_mu={feature: -1000.0},
        inv_cov=[[1.0]],
        prior_winner=0.5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probability = model.proba({feature: -1000.0})
    assert probability == pytest.approx(0.0)


def test_top_quantile_zero_collapses_losers_and_fails_closed():
    """At top_quantile=0.0 the losers class collapses and training rejects."""
    tmp = _workspace_tmp()
    try:
        path = _make_balanced_xlsx(tmp, 120)
        from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

        with pytest.raises(ValueError, match=r"losers"):
            train_profile_model_from_enhanced_xlsx(
                path,
                max_duration_hours=7.0,
                min_profit_factor=1.0,
                top_quantile=0.0,
                features=BASE_NUMERIC_FEATURES,
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
