"""ERR-092 regression tests: de-trended OU half-life estimator and the
advisory (non-gating) half-life window.

Covers:
1. estimate_ou_params_detrended recovers the true half-life on a pure OU
   fixture, and stays close on an OU+trend fixture where the raw estimator's
   half-life inflates from drift attenuation.
2. check_stochastic_regime: with ou_halflife_gate_hard=False (default) an
   out-of-window half-life is advisory (passed stays True when Hurst passes);
   with the flag True the legacy hard rejection is restored.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from neutralgrid.core.config import get_config
from neutralgrid.validation.stochastic import StochasticConfig, StochasticRegimeChecker


def _simulate_ou(
    theta: float,
    sigma: float,
    n: int,
    drift_per_bar: float = 0.0,
    seed: int = 7,
    x0: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = x0
    for i in range(1, n):
        z[i] = z[i - 1] + theta * (0.0 - z[i - 1]) + sigma * rng.standard_normal()
    trend = drift_per_bar * np.arange(n)
    return 4.6 + trend + z  # log-price around ~100


class TestDetrendedOuEstimatorErr092:
    THETA = 0.03  # true half-life = ln(2)/0.03 = 23.1 bars

    def test_pure_ou_recovers_halflife(self):
        checker = StochasticRegimeChecker(StochasticConfig(mc_paths=10))
        lp = _simulate_ou(self.THETA, 0.004, 800)
        theta_dt, _, _ = checker.estimate_ou_params_detrended(lp)
        hl = checker.compute_ou_halflife(theta_dt)
        true_hl = np.log(2) / self.THETA
        assert np.isfinite(hl)
        assert hl == pytest.approx(true_hl, rel=0.40)

    def test_trend_no_longer_inflates_halflife(self):
        """A 0.05%/bar drift (~49% over 800 bars) must not push the de-trended
        half-life toward inf, while the raw estimator inflates materially."""
        checker = StochasticRegimeChecker(StochasticConfig(mc_paths=10))
        lp = _simulate_ou(self.THETA, 0.004, 800, drift_per_bar=0.0005)
        true_hl = np.log(2) / self.THETA

        theta_raw, _, _ = checker.estimate_ou_params(lp)
        hl_raw = checker.compute_ou_halflife(theta_raw)
        theta_dt, _, _ = checker.estimate_ou_params_detrended(lp)
        hl_dt = checker.compute_ou_halflife(theta_dt)

        assert np.isfinite(hl_dt)
        assert hl_dt == pytest.approx(true_hl, rel=0.40)
        # Raw estimator: drift attenuation inflates the half-life well beyond
        # the de-trended estimate (the ERR-092 pathology).
        assert (not np.isfinite(hl_raw)) or hl_raw > 2.0 * hl_dt

    def test_short_series_returns_zero_params(self):
        checker = StochasticRegimeChecker(StochasticConfig(mc_paths=10))
        theta, mu, sigma = checker.estimate_ou_params_detrended(np.ones(5))
        assert (theta, mu, sigma) == (0.0, 0.0, 0.0)


def _slow_ou_df15m(n: int = 800) -> pd.DataFrame:
    """15m frame whose de-trended OU half-life sits above 48 bars while
    Hurst stays below the 0.65 gate (near-random-walk, no strong trend).
    Verified numerically: theta=0.001/seed=7 -> hl_detrended ~92.7,
    hurst ~0.52 (deterministic under the fixed seeds)."""
    lp = _simulate_ou(theta=0.001, sigma=0.003, n=n, seed=7)
    closes = np.exp(lp)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.001,
        "low": closes * 0.999,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


class TestHalflifeAdvisoryGateErr092:
    def _run_check(self, gate_hard: bool):
        from neutralgrid.validation.regime_validator import RegimeValidator

        cfg_obj = copy.deepcopy(get_config())
        cfg_obj.stochastic.ou_halflife_gate_hard = gate_hard
        cfg_obj.stochastic.survival_mc_paths = 50  # keep the test fast
        df = _slow_ou_df15m()
        hi = float(df["high"].max())
        lo = float(df["low"].min())
        with patch(
            "neutralgrid.validation.regime_validator.get_config",
            return_value=cfg_obj,
        ):
            validator = RegimeValidator.__new__(RegimeValidator)  # skip heavy init
            result = validator.check_stochastic_regime(df, range_high=hi, range_low=lo)
        return result

    def test_default_config_demotes_halflife_gate(self):
        assert get_config().stochastic.ou_halflife_gate_hard is False

    def test_out_of_window_halflife_is_advisory_when_soft(self):
        result = self._run_check(gate_hard=False)
        assert result.executed is True
        assert result.metrics is not None
        checks = result.metrics["checks"]
        # Fixture must actually be out-of-window and Hurst-passing, else the
        # test is vacuous.
        assert checks["halflife_ok"] is False
        assert checks["hurst_ok"] is True
        assert checks["halflife_gate_hard"] is False
        # Advisory: overall check passes, telemetry records the excursion.
        assert result.passed is True
        assert result.reason is None
        assert "halflife_advisory" in result.metrics

    def test_out_of_window_halflife_rejects_when_hard(self):
        result = self._run_check(gate_hard=True)
        assert result.executed is True
        assert result.metrics is not None
        assert result.metrics["checks"]["halflife_ok"] is False
        assert result.passed is False
        assert result.reason is not None
        assert "halflife" in result.reason
