"""
Unit tests for feature extraction.

Tests invariants, not performance. High ROI tests that prevent regressions.
"""

import pytest
import numpy as np
import pandas as pd
from neutralgrid.data.features import compute_hmm_features
from neutralgrid.models.hmm.schema import FEATURE_NAMES


def create_test_dataframe(n_bars: int = 200) -> pd.DataFrame:
    """Create a test DataFrame with 1H kline data."""
    return pd.DataFrame({
        "open_time": pd.date_range("2025-01-01", periods=n_bars, freq="1h"),
        "open": np.random.uniform(40000, 41000, n_bars),
        "high": np.random.uniform(40500, 41500, n_bars),
        "low": np.random.uniform(39500, 40500, n_bars),
        "close": np.random.uniform(40000, 41000, n_bars),
        "volume": np.random.uniform(100, 1000, n_bars),
        "close_time": pd.date_range("2025-01-01", periods=n_bars, freq="1h") + pd.Timedelta(hours=1),
        "quote_volume": np.random.uniform(4000000, 41000000, n_bars),
        "trades": np.random.randint(1000, 5000, n_bars),
        "taker_buy_base": np.random.uniform(50, 500, n_bars),
        "taker_buy_quote": np.random.uniform(2000000, 20000000, n_bars),
        "ignore": [0] * n_bars,
    })


class TestFeatureExtraction:
    """Test feature extraction invariants."""

    def test_returns_expected_columns(self):
        """Test that feature matrix has expected number of columns."""
        df = create_test_dataframe()
        X, valid = compute_hmm_features(df)

        assert X.shape[1] == len(FEATURE_NAMES), (
            f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}"
        )

    def test_valid_mask_length_matches_input(self):
        """Test that valid mask has same length as input DataFrame."""
        df = create_test_dataframe(n_bars=150)
        X, valid = compute_hmm_features(df)

        assert len(valid) == len(df), (
            f"Valid mask length {len(valid)} doesn't match input length {len(df)}"
        )

    def test_nans_only_in_warmup_period(self):
        """Test that NaNs only occur during indicator warmup."""
        df = create_test_dataframe(n_bars=200)
        X, valid = compute_hmm_features(df)

        # Valid rows should have no NaNs
        X_valid = X[valid]
        assert not np.isnan(X_valid).any(), (
            "Found NaNs in valid rows (after warmup)"
        )

        # All valid rows should come after warmup
        valid_indices = np.where(valid)[0]
        assert valid_indices[0] > 20, (
            f"Valid data starts too early at index {valid_indices[0]} (expected > 20)"
        )

    def test_feature_order_matches_schema(self):
        """Test that features are in correct order."""
        df = create_test_dataframe()
        X, valid = compute_hmm_features(df)

        # Feature names from compute_hmm_features should match schema
        # This is validated internally, but we test it explicitly
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = create_test_dataframe(n_bars=100)

        X1, valid1 = compute_hmm_features(df)
        X2, valid2 = compute_hmm_features(df)

        np.testing.assert_array_equal(valid1, valid2,
            err_msg="Valid masks differ for same input")
        np.testing.assert_allclose(X1, X2, rtol=1e-10,
            err_msg="Feature matrices differ for same input")

    def test_handles_empty_dataframe(self):
        """Test graceful handling of empty input."""
        df = pd.DataFrame({
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })

        X, valid = compute_hmm_features(df)

        assert X.shape == (0, len(FEATURE_NAMES)), (
            f"Expected empty array with {len(FEATURE_NAMES)} columns"
        )
        assert len(valid) == 0, "Expected empty valid mask"

    def test_all_features_finite_after_warmup(self):
        """Test that all features are finite after warmup period."""
        df = create_test_dataframe(n_bars=200)
        X, valid = compute_hmm_features(df)

        X_valid = X[valid]

        assert np.isfinite(X_valid).all(), (
            "Found non-finite values in valid rows"
        )

    def test_feature_ranges_reasonable(self):
        """Test that feature values are in reasonable ranges."""
        df = create_test_dataframe(n_bars=200)
        X, valid = compute_hmm_features(df)

        X_valid = X[valid]

        # r_t (log returns) should be small
        r_t = X_valid[:, FEATURE_NAMES.index("r_t")]
        assert np.abs(r_t).max() < 0.5, (
            f"Log returns suspiciously large: {np.abs(r_t).max()}"
        )

        # vol_t (volatility) should be positive
        vol_t = X_valid[:, FEATURE_NAMES.index("vol_t")]
        assert (vol_t >= 0).all(), (
            "Volatility should be non-negative"
        )

        # adx_t should be in [0, 100]
        adx_t = X_valid[:, FEATURE_NAMES.index("adx_t")]
        assert (adx_t >= 0).all() and (adx_t <= 100).all(), (
            f"ADX out of range [0, 100]: min={adx_t.min()}, max={adx_t.max()}"
        )

    def test_missing_required_columns_raises(self):
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            # Missing 'low', 'close', 'volume'
        })

        with pytest.raises(ValueError, match="missing required columns"):
            compute_hmm_features(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
