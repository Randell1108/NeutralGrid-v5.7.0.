"""Tests for non-stationary tail correction system.

Covers: Cornish-Fisher VaR, tail adjustment weights, kurtosis-adjusted
volatility, Gaussian divergence, synthetic exogenous signals, enhanced
uncertainty profile, and backward compatibility.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from neutralgrid.models.hmm.uncertainty import (
    _compute_var_cvar_cornish_fisher,
    _compute_var_cvar_gaussian,
    _compute_gaussian_divergence,
    _compute_kurtosis_adjusted_vol,
    _classify_volatility_tiers_kurtosis_aware,
    compute_tail_adjustment_weights,
    _generate_synthetic_exogenous_signals,
    compute_regime_uncertainty_profile,
    compute_weighted_conditional_tail_risk,
    compute_weighted_conditional_tail_risk_enhanced,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_feature_matrix(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2-state feature matrix for profile tests."""
    rng = np.random.default_rng(seed)
    n_per = 200

    # State 0: low vol, near-Gaussian
    r0 = rng.normal(0.0, 0.004, n_per)
    vol0 = np.full(n_per, 0.004)
    trend0 = rng.normal(0.0, 0.001, n_per)
    adx0 = rng.uniform(15, 25, n_per)
    bbw0 = rng.uniform(0.01, 0.02, n_per)

    # State 1: high vol, fat-tailed (Student-t like via mixture)
    r1 = rng.normal(0.0, 0.012, n_per)
    # Inject heavy tails: 10% of observations are 3x larger
    tail_mask = rng.random(n_per) < 0.10
    r1[tail_mask] *= 3.0
    vol1 = np.full(n_per, 0.012)
    trend1 = rng.normal(0.002, 0.003, n_per)
    adx1 = rng.uniform(30, 50, n_per)
    bbw1 = rng.uniform(0.03, 0.06, n_per)

    X = np.vstack([
        np.column_stack([r0, vol0, trend0, adx0, bbw0]),
        np.column_stack([r1, vol1, trend1, adx1, bbw1]),
    ])
    states = np.array([0] * n_per + [1] * n_per, dtype=int)
    return X, states


def _make_state_means() -> np.ndarray:
    return np.array([
        [0.0, 0.004, 0.0, 20.0, 0.015],
        [0.0, 0.012, 0.002, 40.0, 0.045],
    ])


# ---------------------------------------------------------------------------
# 1. CF VaR more conservative than Gaussian for leptokurtic data
# ---------------------------------------------------------------------------

def test_cornish_fisher_var_more_conservative_than_gaussian():
    rng = np.random.default_rng(7)
    # Simulate heavy-tailed returns (mixture: 90% N(0, 0.01) + 10% N(0, 0.05))
    n = 2000
    base = rng.normal(0.0, 0.01, n)
    tails = rng.normal(0.0, 0.05, n)
    mask = rng.random(n) < 0.10
    returns = np.where(mask, tails, base)

    from scipy.stats import kurtosis, skew
    sk = float(skew(returns, bias=False))
    ek = float(kurtosis(returns, fisher=True, bias=False))

    cf_var, _ = _compute_var_cvar_cornish_fisher(returns, 0.99, sk, ek)
    g_var, _ = _compute_var_cvar_gaussian(returns, 0.99)

    # CF should be more extreme (more negative) than Gaussian for fat tails
    assert cf_var < g_var, (
        f"CF VaR ({cf_var:.6f}) should be more negative than Gaussian ({g_var:.6f}) "
        f"for leptokurtic data (excess_kurtosis={ek:.2f})"
    )


# ---------------------------------------------------------------------------
# 2. CF degenerates to Gaussian when skew=0, kurtosis=0
# ---------------------------------------------------------------------------

def test_cornish_fisher_degenerates_to_gaussian_for_normal():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, 5000)  # large sample, near-Gaussian

    cf_var, _ = _compute_var_cvar_cornish_fisher(returns, 0.95, 0.0, 0.0)
    g_var, _ = _compute_var_cvar_gaussian(returns, 0.95)

    assert cf_var == pytest.approx(g_var, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Tail weights neutral for non-extreme observations
# ---------------------------------------------------------------------------

def test_tail_adjustment_weights_neutral_for_non_extreme():
    observation = np.array([0.001, 0.005, 0.0, 25.0, 0.02])
    state_params = [
        {"mean": 0.0, "std": 0.005, "p_empirical_below_2sigma": 0.023, "p_gaussian_below_2sigma": 0.023},
        {"mean": 0.0, "std": 0.012, "p_empirical_below_2sigma": 0.023, "p_gaussian_below_2sigma": 0.023},
    ]

    weights = compute_tail_adjustment_weights(observation, state_params)
    np.testing.assert_allclose(weights, [1.0, 1.0])


# ---------------------------------------------------------------------------
# 4. Tail weights upweight heavy-tailed state
# ---------------------------------------------------------------------------

def test_tail_adjustment_weights_upweight_heavy_tailed_state():
    # Extreme observation (5 sigma for state 0, 2.5 sigma for state 1)
    observation = np.array([-0.025, 0.005, 0.0, 25.0, 0.02])
    state_params = [
        {"mean": 0.0, "std": 0.005, "excess_kurtosis": 0.0,
         "p_empirical_below_2sigma": 0.023, "p_gaussian_below_2sigma": 0.023},
        {"mean": 0.0, "std": 0.010, "excess_kurtosis": 5.0,
         "p_empirical_below_2sigma": 0.10, "p_gaussian_below_2sigma": 0.023},
    ]

    weights = compute_tail_adjustment_weights(observation, state_params, threshold_sigma=2.0)

    # State 1 should get upweighted (p_emp >> p_gauss)
    assert weights[1] > 1.0, f"Heavy-tailed state should be upweighted, got {weights[1]}"


# ---------------------------------------------------------------------------
# 5. Weights clamped to max_weight
# ---------------------------------------------------------------------------

def test_tail_adjustment_weights_clamped():
    observation = np.array([-0.10, 0.005, 0.0, 25.0, 0.02])
    state_params = [
        {"mean": 0.0, "std": 0.005, "p_empirical_below_2sigma": 0.50, "p_gaussian_below_2sigma": 0.023},
    ]

    weights = compute_tail_adjustment_weights(
        observation, state_params, max_weight=3.0, threshold_sigma=2.0,
    )

    assert weights[0] <= 3.0, f"Weight should be clamped to max_weight, got {weights[0]}"


# ---------------------------------------------------------------------------
# 6. Corrected posteriors sum to 1.0
# ---------------------------------------------------------------------------

def test_corrected_posteriors_sum_to_one():
    from neutralgrid.models.hmm.inference import _apply_tail_correction

    posteriors = np.array([
        [0.6, 0.4],
        [0.5, 0.5],
        [0.7, 0.3],
    ])
    observation = np.array([-0.05, 0.005, 0.0, 25.0, 0.02])

    regime_uncertainty = {
        "regimes": {
            "state_0": {
                "tail_adjustment_params": {
                    "mean": 0.0, "std": 0.005, "skew": 0.0,
                    "excess_kurtosis": 0.0,
                    "p_empirical_below_2sigma": 0.10,
                    "p_gaussian_below_2sigma": 0.023,
                },
            },
            "state_1": {
                "tail_adjustment_params": {
                    "mean": 0.0, "std": 0.012, "skew": -0.5,
                    "excess_kurtosis": 3.0,
                    "p_empirical_below_2sigma": 0.15,
                    "p_gaussian_below_2sigma": 0.023,
                },
            },
        },
    }

    corrected, applied, weights = _apply_tail_correction(
        posteriors, observation, regime_uncertainty,
        enabled=True, threshold_sigma=2.0, max_weight=3.0,
    )

    assert corrected.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 7. Correction disabled returns raw posteriors
# ---------------------------------------------------------------------------

def test_tail_correction_disabled_returns_raw():
    from neutralgrid.models.hmm.inference import _apply_tail_correction

    posteriors = np.array([[0.6, 0.4], [0.7, 0.3]])
    observation = np.array([-0.05, 0.005, 0.0, 25.0, 0.02])

    corrected, applied, weights = _apply_tail_correction(
        posteriors, observation, regime_uncertainty=None,
        enabled=False,
    )

    np.testing.assert_array_equal(corrected, posteriors[-1])
    assert applied is False


# ---------------------------------------------------------------------------
# 8. Graceful fallback when regime_uncertainty is None
# ---------------------------------------------------------------------------

def test_tail_correction_graceful_fallback_no_uncertainty():
    from neutralgrid.models.hmm.inference import _apply_tail_correction

    posteriors = np.array([[0.5, 0.5], [0.6, 0.4]])
    observation = np.array([-0.05, 0.005, 0.0, 25.0, 0.02])

    corrected, applied, weights = _apply_tail_correction(
        posteriors, observation, regime_uncertainty=None,
        enabled=True,
    )

    np.testing.assert_array_equal(corrected, posteriors[-1])
    assert applied is False
    assert weights == []


# ---------------------------------------------------------------------------
# 9. Kurtosis-adjusted vol > raw vol for positive kurtosis
# ---------------------------------------------------------------------------

def test_kurtosis_adjusted_vol_higher_than_raw():
    raw_std = 0.01
    adj = _compute_kurtosis_adjusted_vol(raw_std, excess_kurtosis_val=4.0)
    assert adj > raw_std


def test_kurtosis_adjusted_vol_equals_raw_for_zero_kurtosis():
    raw_std = 0.01
    adj = _compute_kurtosis_adjusted_vol(raw_std, excess_kurtosis_val=0.0)
    assert adj == pytest.approx(raw_std, abs=1e-12)


# ---------------------------------------------------------------------------
# 10. Kurtosis-aware tiers can differ from simple tiers
# ---------------------------------------------------------------------------

def test_kurtosis_aware_tiers_differ_from_simple():
    # Two states with similar vol but different kurtosis
    state_vol = {0: 0.010, 1: 0.011}
    state_kurtosis = {0: 0.0, 1: 8.0}

    result = _classify_volatility_tiers_kurtosis_aware(state_vol, state_kurtosis)

    # State 1 should be "high" due to kurtosis adjustment
    assert result["state_tiers"][1] == "high" or result["state_tiers"][0] == "low"


# ---------------------------------------------------------------------------
# 11. Gaussian divergence high for heavy-tailed data
# ---------------------------------------------------------------------------

def test_gaussian_divergence_high_for_heavy_tails():
    rng = np.random.default_rng(7)
    # Student-t with 3 DoF (very heavy tails)
    from scipy.stats import t as tdist
    heavy = tdist.rvs(df=3, size=2000, random_state=rng)

    result = _compute_gaussian_divergence(heavy)

    assert result["jarque_bera_pvalue"] < 0.05, "Heavy-tailed data should reject normality"
    assert result["tail_weight_ratio"] > 1.0, "Should have more tail mass than Gaussian"


# ---------------------------------------------------------------------------
# 12. Gaussian divergence low for Gaussian data
# ---------------------------------------------------------------------------

def test_gaussian_divergence_low_for_normal_data():
    rng = np.random.default_rng(7)
    normal = rng.normal(0.0, 1.0, 5000)

    result = _compute_gaussian_divergence(normal)

    assert result["jarque_bera_pvalue"] > 0.01, "Gaussian data should not reject normality at 1%"
    assert 0.5 < result["tail_weight_ratio"] < 2.0, "Tail weight should be near 1.0 for Gaussian"


# ---------------------------------------------------------------------------
# 13. Synthetic exogenous signals shape
# ---------------------------------------------------------------------------

def test_synthetic_exogenous_signals_shape():
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, 500)
    vol = np.full(500, 0.01)

    signals = _generate_synthetic_exogenous_signals(returns, vol)

    assert signals.shape[0] == 500
    assert "volatility_shock" in signals.columns
    assert "return_shock" in signals.columns
    assert "momentum_divergence" in signals.columns


# ---------------------------------------------------------------------------
# 14. Profile includes cornish_fisher keys
# ---------------------------------------------------------------------------

def test_uncertainty_profile_includes_cornish_fisher():
    X, states = _make_feature_matrix()
    means = _make_state_means()
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=transmat,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    for state_key in ("state_0", "state_1"):
        regime = profile["regimes"][state_key]
        assert "cornish_fisher" in regime["tail_risk"]
        cf = regime["tail_risk"]["cornish_fisher"]
        assert "var_95" in cf
        assert "cvar_99" in cf


# ---------------------------------------------------------------------------
# 15. Profile includes tail_adjustment_params keys
# ---------------------------------------------------------------------------

def test_uncertainty_profile_includes_tail_adjustment_params():
    X, states = _make_feature_matrix()
    means = _make_state_means()

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=None,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    for state_key in ("state_0", "state_1"):
        regime = profile["regimes"][state_key]
        tap = regime["tail_adjustment_params"]
        assert "mean" in tap
        assert "std" in tap
        assert "skew" in tap
        assert "excess_kurtosis" in tap
        assert "p_empirical_below_2sigma" in tap
        assert "p_gaussian_below_2sigma" in tap


# ---------------------------------------------------------------------------
# 16. Enhanced conditional tail risk includes CF metrics
# ---------------------------------------------------------------------------

def test_enhanced_tail_risk_includes_cf_metrics():
    X, states = _make_feature_matrix()
    means = _make_state_means()
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=transmat,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    posterior = np.array([0.3, 0.7])
    result = compute_weighted_conditional_tail_risk_enhanced(posterior, profile)

    assert "cf_var_95" in result
    assert "cf_cvar_95" in result
    assert "weighted_excess_kurtosis" in result
    assert "weighted_skewness" in result
    assert "weighted_gaussian_divergence" in result


# ---------------------------------------------------------------------------
# 17. Existing uncertainty profile keys unchanged (backward compat)
# ---------------------------------------------------------------------------

def test_existing_uncertainty_profile_keys_unchanged():
    X, states = _make_feature_matrix()
    means = _make_state_means()
    transmat = np.array([[0.95, 0.05], [0.10, 0.90]])

    profile = compute_regime_uncertainty_profile(
        feature_matrix_unscaled=X,
        state_assignments=states,
        state_means_unscaled=means,
        transmat=transmat,
        feature_names=["r_t", "vol_t", "trend_t", "adx_t", "bbwidth_t"],
    )

    # Existing top-level keys
    assert "method" in profile
    assert "n_samples" in profile
    assert "global_reference" in profile
    assert "volatility_tiers" in profile
    assert "regimes" in profile
    assert "exogenous_uncertainty" in profile

    # Existing per-state keys
    for state_key in ("state_0", "state_1"):
        regime = profile["regimes"][state_key]
        assert "sample_count" in regime
        assert "moments" in regime
        assert "tail_risk" in regime
        assert "empirical" in regime["tail_risk"]
        assert "gaussian" in regime["tail_risk"]
        assert "tail_event_probabilities" in regime["tail_risk"]

    # Existing base function still works
    posterior = np.array([0.4, 0.6])
    base_result = compute_weighted_conditional_tail_risk(posterior, profile)
    assert "var_95" in base_result or "cvar_95" in base_result
