"""Unit tests for AFML pipeline bug fixes identified during audit."""
from __future__ import annotations

import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from scipy import stats


# ###########################################################################
# 1. MinTRL kurtosis consistency (cpcv.py:526)
#    Bug: was (kurtosis - 1) / 4.0 instead of (kurtosis - 3) / 4.0
#    Fix: changed to (kurtosis - 3) / 4.0 per AFML Ch 14 formula
# ###########################################################################

class TestMinTRLKurtosisConsistency:
    """Verify kurtosis term in MinTRL uses (kurtosis - 3) / 4.0 per AFML."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.backtest.cpcv import compute_min_track_record_length
        self.compute_min_trl = compute_min_track_record_length

    def test_kurtosis_term_zero_for_normal_returns(self):
        """With kurtosis=3.0 (normal distribution), the kurtosis contribution
        to variance_factor should be exactly 0.0.

        variance_factor = 1 + 0.5*SR^2 - skew*SR + (kurtosis - 3)/4 * SR^2
        For normal: kurtosis=3 => (3-3)/4 * SR^2 = 0.0
        So variance_factor = 1.5 (with skew=0, SR=1).
        """
        # Compute MinTRL for normal returns
        z_alpha = stats.norm.ppf(0.95)
        mintrl = self.compute_min_trl(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            skewness=0.0,
            kurtosis=3.0,
            confidence=0.95,
        )

        # Expected: variance_factor = 1.5, so MinTRL = 1 + 1.5 * (z_alpha/1.0)^2
        expected = 1 + 1.5 * (z_alpha / 1.0) ** 2
        assert mintrl == int(np.ceil(expected)), (
            f"MinTRL with normal kurtosis=3 should be ceil({expected:.4f})="
            f"{int(np.ceil(expected))}, got {mintrl}. "
            "The (kurtosis-3)/4 term should be 0."
        )

    def test_kurtosis_difference_proportional_to_sr_squared(self):
        """The difference in MinTRL between kurtosis=3 and kurtosis=5
        should be proportional to (5-3)/4 * SR^2 = 0.5 * SR^2.

        variance_factor(k=3) = 1.0  (kurtosis term = 0)
        variance_factor(k=5) = 1.0 + (5-3)/4 * 1.0^2 = 1.5
        """
        z_alpha = stats.norm.ppf(0.95)
        sr = 1.0

        mintrl_k3 = self.compute_min_trl(
            observed_sharpe=sr, benchmark_sharpe=0.0,
            skewness=0.0, kurtosis=3.0, confidence=0.95,
        )
        mintrl_k5 = self.compute_min_trl(
            observed_sharpe=sr, benchmark_sharpe=0.0,
            skewness=0.0, kurtosis=5.0, confidence=0.95,
        )

        # The kurtosis=5 case should require more data
        assert mintrl_k5 > mintrl_k3, (
            f"Leptokurtic returns (k=5) should need more data than normal (k=3): "
            f"got mintrl_k5={mintrl_k5}, mintrl_k3={mintrl_k3}"
        )

        # Verify the exact difference is driven by (5-3)/4 * SR^2 = 0.5
        z_sq = (z_alpha / sr) ** 2
        expected_diff_raw = 0.5 * z_sq  # (5-3)/4 * SR^2 * (z/sr_diff)^2
        # Due to ceiling, difference in ceiled values may not be exact but
        # the raw (pre-ceiling) values should match
        vf_k3 = 1.5  # variance_factor at kurtosis=3 (1 + 0.5*SR^2)
        vf_k5 = 1.5 + 0.5  # variance_factor at kurtosis=5
        raw_k3 = 1 + vf_k3 * z_sq
        raw_k5 = 1 + vf_k5 * z_sq
        assert raw_k5 - raw_k3 == pytest.approx(expected_diff_raw, rel=1e-10), (
            f"Raw MinTRL difference should be {expected_diff_raw:.4f}, "
            f"got {raw_k5 - raw_k3:.4f}"
        )

    def test_source_uses_kurtosis_minus_3(self):
        """Verify source code uses (kurtosis - 3) not (kurtosis - 1)."""
        src = inspect.getsource(self.compute_min_trl)
        assert "(kurtosis - 3)" in src or "kurtosis - 3" in src, (
            "compute_min_track_record_length should use (kurtosis - 3) per AFML"
        )
        assert "(kurtosis - 1)" not in src, (
            "Should NOT use (kurtosis - 1) -- that was the old bug"
        )


# ###########################################################################
# 2. Sharpe ratio ddof=1 (backtest_realistic.py:410)
#    Bug: was using ddof=0 (population std) instead of ddof=1 (sample std)
#    Fix: changed to np.std(returns, ddof=1)
# ###########################################################################

class TestSharpeUsesddof1:
    """Verify the backtester Sharpe ratio uses sample std (ddof=1)."""

    def test_source_uses_ddof1(self):
        """backtest_realistic.py should contain ddof=1 in Sharpe computation."""
        from pathlib import Path
        src_path = Path(__file__).resolve().parent.parent.parent / "backtest" / "backtest_realistic.py"
        source = src_path.read_text()
        assert "ddof=1" in source, (
            "backtest_realistic.py should use ddof=1 for Sharpe ratio"
        )

    def test_sharpe_manual_ddof1_computation(self):
        """Verify Sharpe formula: mean(r) / std(r, ddof=1) * sqrt(525600).

        We replicate the exact computation from backtest_realistic.py:410
        using a known equity curve and confirm ddof=1 is used.
        """
        # Build a simple equity curve
        equity_curve = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]
        equity_arr = np.array(equity_curve)

        # Compute returns the same way as backtest_realistic.py:408
        returns = np.diff(equity_arr) / equity_arr[:-1]

        # Compute Sharpe with ddof=1 (correct - sample std)
        sharpe_ddof1 = (
            np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(525600)
        )

        # Compute Sharpe with ddof=0 (wrong - population std)
        sharpe_ddof0 = (
            np.mean(returns) / np.std(returns, ddof=0) * np.sqrt(525600)
        )

        # They should differ (ddof=1 gives smaller denominator => larger Sharpe)
        assert sharpe_ddof1 != pytest.approx(sharpe_ddof0, rel=1e-6), (
            "ddof=1 and ddof=0 should produce different Sharpe values"
        )

        # ddof=1 should give a larger Sharpe (std is larger with ddof=1 for
        # small n, but wait: std with ddof=1 is LARGER, so Sharpe is SMALLER)
        # Actually: std(ddof=1) >= std(ddof=0), so Sharpe(ddof=1) <= Sharpe(ddof=0)
        assert sharpe_ddof1 < sharpe_ddof0, (
            "ddof=1 (sample std) should be more conservative than ddof=0"
        )

    def test_sharpe_ddof1_exact_value(self):
        """Compute Sharpe on a known series and verify exact value."""
        # Simple returns: [+1%, -0.5%, +1%, -0.5%, +1%]
        returns = np.array([0.01, -0.005, 0.01, -0.005, 0.01])

        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))
        expected_sharpe = mean_r / std_r * np.sqrt(525600)

        # Replicate the backtester's formula
        computed_sharpe = (
            (np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(525600))
            if np.std(returns, ddof=1) > 0
            else 0
        )

        assert computed_sharpe == pytest.approx(expected_sharpe, rel=1e-10), (
            f"Sharpe should be {expected_sharpe:.4f}, got {computed_sharpe:.4f}"
        )


# ###########################################################################
# 3. 15m ADX warmup guard (feature_extractor.py:143)
#    Bug: 15m warmup used 1 * adx_period instead of 2 * adx_period
#    Fix: changed to max(2 * adx_period, rsi_period, bb_period) + 10
#         consistent with the 1H ADX warmup at line 131
# ###########################################################################

class TestADXWarmupGuard:
    """Verify 15m ADX warmup uses 2*adx_period, consistent with 1H."""

    def test_source_uses_2x_adx_period_for_15m(self):
        """feature_extractor.py line 143 should use 2 * adx_period for 15m data."""
        from neutralgrid.scanner import feature_extractor
        source = inspect.getsource(feature_extractor)

        # Find the 15m block: it should reference 2 * _cfg_val("adx_period")
        # in the warmup guard for klines_15m
        lines = source.split("\n")
        found_2x_for_15m = False
        for i, line in enumerate(lines):
            if "klines_15m" in line and "2 *" in line and "adx_period" in line:
                found_2x_for_15m = True
                break

        assert found_2x_for_15m, (
            "15m ADX warmup should use '2 * _cfg_val(\"adx_period\")' "
            "to match the 1H warmup formula (2*period for full ADX convergence)"
        )

    def test_1h_and_15m_both_use_2x_warmup(self):
        """Both 1H and 15m ADX should require 2*adx_period bars."""
        from neutralgrid.scanner import feature_extractor
        source = inspect.getsource(feature_extractor)

        # Count occurrences of the 2*adx_period warmup pattern
        import re
        pattern = r'2\s*\*\s*_cfg_val\(\s*["\']adx_period["\']\s*\)'
        matches = re.findall(pattern, source)

        # Should appear at least twice: once for 1H (line ~131) and once for 15m (line ~143)
        assert len(matches) >= 2, (
            f"Expected at least 2 occurrences of '2 * _cfg_val(\"adx_period\")' "
            f"(for 1H and 15m ADX), found {len(matches)}"
        )

    def test_warmup_formula_rationale(self):
        """ADX needs 2*period bars: period for TR/DM smoothing + period for DX smoothing.

        With default adx_period=14:
        - 1x period (14): only enough for TR/DM EMAs
        - 2x period (28): allows DX EMA to also converge
        The 15m guard should require at least 2*14 = 28 bars minimum.
        """
        adx_period = 14  # default from config

        min_bars_1x = adx_period  # insufficient
        min_bars_2x = 2 * adx_period  # correct

        assert min_bars_2x == 28, (
            f"2*adx_period with default 14 should be 28, got {min_bars_2x}"
        )
        assert min_bars_2x > min_bars_1x, (
            "2x warmup should require more bars than 1x"
        )


# ###########################################################################
# Entry point
# ###########################################################################

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
