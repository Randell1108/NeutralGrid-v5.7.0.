from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neutralgrid.models.hmm.uncertainty import (
    compute_regime_uncertainty_profile,
    compute_weighted_conditional_tail_risk,
)


def _make_feature_matrix(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Regime 0: lower vol with occasional tail losses.
    r0 = rng.normal(0.0005, 0.004, 180)
    r0[::40] -= 0.03

    # Regime 1: higher vol with heavier downside tails.
    r1 = rng.normal(-0.0002, 0.012, 180)
    r1[::30] -= 0.05

    returns = np.concatenate([r0, r1])
    states = np.array([0] * len(r0) + [1] * len(r1), dtype=int)

    # Build a 5-feature matrix matching HMM schema order.
    vol = np.abs(returns) * 8 + rng.normal(0.0, 0.02, len(returns))
    trend = rng.normal(0.0, 0.4, len(returns))
    adx = np.clip(rng.normal(20.0, 5.0, len(returns)), 0.0, None)
    bbw = np.clip(rng.normal(0.03, 0.01, len(returns)), 0.0, None)
    X = np.column_stack([returns, vol, trend, adx, bbw]).astype(float)
    return X, states


def test_uncertainty_profile_has_excess_kurtosis_and_tail_metrics() -> None:
    X, states = _make_feature_matrix()
    means = np.array(
        [
            [0.0004, 0.25, 0.02, 18.0, 0.02],
            [-0.0003, 0.55, 0.04, 26.0, 0.05],
        ],
        dtype=float,
    )
    transmat = np.array([[0.95, 0.05], [0.08, 0.92]], dtype=float)

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=transmat,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    s0 = profile["regimes"]["state_0"]
    s1 = profile["regimes"]["state_1"]

    assert "excess_kurtosis" in s0["moments"]
    assert "excess_kurtosis" in s1["moments"]
    assert "empirical" in s0["tail_risk"]
    assert "gaussian" in s0["tail_risk"]
    assert "var_95" in s0["tail_risk"]["empirical"]
    assert "cvar_99" in s1["tail_risk"]["empirical"]


def test_uncertainty_profile_supports_exogenous_framework() -> None:
    X, states = _make_feature_matrix(seed=11)
    means = np.array(
        [
            [0.0004, 0.20, 0.01, 17.0, 0.02],
            [-0.0004, 0.60, 0.05, 28.0, 0.06],
        ],
        dtype=float,
    )

    exog = pd.DataFrame(
        {
            "news_risk": np.linspace(0.0, 1.0, X.shape[0]),
            "macro_risk": np.sin(np.linspace(0.0, 6.0, X.shape[0])),
            "political_risk": np.random.default_rng(2).normal(0.0, 0.2, X.shape[0]),
        }
    )

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=None,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
        exogenous_signals=exog,
    )

    exo = profile["exogenous_uncertainty"]
    assert exo["enabled"] is True
    assert len(exo["features_used"]) >= 1
    assert "state_0" in exo["regime_scores"]


def test_weighted_conditional_tail_risk_matches_manual_weighting() -> None:
    X, states = _make_feature_matrix(seed=21)
    means = np.array(
        [
            [0.0004, 0.20, 0.01, 17.0, 0.02],
            [-0.0004, 0.60, 0.05, 28.0, 0.06],
        ],
        dtype=float,
    )
    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=None,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    posterior = np.array([0.25, 0.75], dtype=float)
    weighted = compute_weighted_conditional_tail_risk(posterior, profile)

    s0 = profile["regimes"]["state_0"]["tail_risk"]["empirical"]["cvar_95"]
    s1 = profile["regimes"]["state_1"]["tail_risk"]["empirical"]["cvar_95"]
    expected = 0.25 * float(s0) + 0.75 * float(s1)
    assert weighted["cvar_95"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
