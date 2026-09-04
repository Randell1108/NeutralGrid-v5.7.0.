"""
AFML Compliance Tests for NEUTRAL Grid Bot v6.

Comprehensive test suite verifying all critical AFML-alignment fixes applied
to the codebase.  Tests are self-contained (synthetic data only, no external
APIs) and deterministic (fixed seeds).

Modules under test
------------------
1.  Sample Weights          (src/neutralgrid/training/sample_weights.py)
2.  Indicators Warmup       (indicators/technical.py)
3.  Triple Barrier           (src/neutralgrid/models/triple_barrier.py)
4.  CPCV                     (src/neutralgrid/backtest/cpcv.py)
5.  Meta-Labeler             (src/neutralgrid/models/meta_labeler.py)
6.  Funding Rate             (src/neutralgrid/data/funding_rate.py)
7.  Data Curator             (src/neutralgrid/data/curator.py)
8.  Market Data              (src/neutralgrid/data/market_data.py)
9.  Backtest Realistic       (backtest/backtest_realistic.py)
10. Backtest Grid            (backtest/backtest_grid.py)
11. Walk-Forward Evaluation  (src/neutralgrid/backtest/evaluate.py)

Run with:
    pytest tests/test_afml_compliance.py -v
"""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup -- pyproject.toml [tool.pytest.ini_options] sets pythonpath=["."]
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Picklable stubs for meta-labeler imputer mismatch test
# (Must be at module level so pickle can resolve them.)
# ---------------------------------------------------------------------------

class _FakeModel:
    """Picklable stand-in for a trained sklearn model."""
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class _FakeImputer:
    """Picklable stand-in for SimpleImputer with a mismatched n_features_in_."""
    n_features_in_ = 5  # Mismatch: feature list will have 3

    def transform(self, X):
        return np.nan_to_num(X, nan=0.0)


# ###########################################################################
# 1. Sample Weights
# ###########################################################################

class TestSampleWeights:
    """Verify concurrency matrix transposition fix (cond2 bug)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.training.sample_weights import (
            compute_avg_uniqueness,
            compute_concurrency_matrix,
            compute_concurrent_labels,
        )
        self.compute_concurrency_matrix = compute_concurrency_matrix
        self.compute_concurrent_labels = compute_concurrent_labels
        self.compute_avg_uniqueness = compute_avg_uniqueness

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _make_events(intervals):
        """Create t0/t1 Series from list of (start, end) tuples (hours)."""
        base = pd.Timestamp("2024-01-01", tz="UTC")
        t0 = pd.Series([base + timedelta(hours=s) for s, _ in intervals])
        t1 = pd.Series([base + timedelta(hours=e) for _, e in intervals])
        return t0, t1

    # -- tests -----------------------------------------------------------
    def test_concurrency_matrix_symmetry(self):
        """Concurrency matrix must be symmetric after the transposition fix."""
        t0, t1 = self._make_events([(0, 2), (1, 3), (4, 6)])
        mat = self.compute_concurrency_matrix(t0, t1)
        np.testing.assert_array_equal(mat, mat.T)

    def test_concurrency_non_overlapping(self):
        """Two non-overlapping events must have overlap_matrix[0,1] = False."""
        t0, t1 = self._make_events([(0, 1), (2, 3)])
        mat = self.compute_concurrency_matrix(t0, t1)
        assert not mat[0, 1], "Non-overlapping events should not register overlap"
        assert not mat[1, 0], "Symmetry: [1,0] should also be False"

    def test_concurrency_overlapping(self):
        """Two overlapping events must have overlap_matrix[0,1] = True."""
        t0, t1 = self._make_events([(0, 3), (1, 4)])
        mat = self.compute_concurrency_matrix(t0, t1)
        assert mat[0, 1], "Overlapping events should register overlap"
        assert mat[1, 0], "Symmetry: [1,0] should also be True"

    def test_uniqueness_non_overlapping(self):
        """Non-overlapping events should all have uniqueness = 1.0."""
        t0, t1 = self._make_events([(0, 1), (2, 3), (4, 5)])
        uniq = self.compute_avg_uniqueness(t0, t1)
        np.testing.assert_allclose(uniq, 1.0, atol=1e-12)

    def test_uniqueness_fully_overlapping(self):
        """N fully concurrent events should each have uniqueness = 1/N."""
        N = 5
        t0, t1 = self._make_events([(0, 10)] * N)
        uniq = self.compute_avg_uniqueness(t0, t1)
        np.testing.assert_allclose(uniq, 1.0 / N, atol=1e-12)


# ###########################################################################
# 2. Indicators Warmup
# ###########################################################################

class TestIndicatorsWarmup:
    """Verify EMA / ATR / RSI / ADX enforce NaN during warmup period."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        # 200 bars of synthetic close-like data
        self.n = 200
        self.prices = 100.0 + np.cumsum(np.random.randn(self.n) * 0.5)
        self.highs = self.prices + np.abs(np.random.randn(self.n) * 0.3)
        self.lows = self.prices - np.abs(np.random.randn(self.n) * 0.3)
        self.closes = self.prices.copy()

    def test_ema_warmup_nan(self):
        """calc_ema with period=10 should return NaN for bars 0-8."""
        from neutralgrid.indicators.technical import calc_ema

        period = 10
        ema = calc_ema(self.closes, period)
        assert np.all(np.isnan(ema[: period - 1])), (
            f"Bars 0..{period-2} must be NaN; got {ema[:period-1]}"
        )

    def test_ema_warmup_valid(self):
        """calc_ema with period=10 should return non-NaN from bar 9 onwards."""
        from neutralgrid.indicators.technical import calc_ema

        period = 10
        ema = calc_ema(self.closes, period)
        valid_region = ema[period - 1 :]
        assert np.all(np.isfinite(valid_region)), (
            f"Bars {period-1}+ must be finite; found NaN at relative "
            f"indices {np.where(~np.isfinite(valid_region))[0]}"
        )

    def test_atr_warmup_nan(self):
        """calc_atr with period=14 should return NaN for bars 0-13."""
        from neutralgrid.indicators.technical import calc_atr

        period = 14
        atr = calc_atr(self.highs, self.lows, self.closes, period)
        # ATR code sets NaN for [:period] (inclusive of index 0..period-1)
        assert np.all(np.isnan(atr[:period])), (
            f"Bars 0..{period-1} must be NaN; got {atr[:period]}"
        )

    def test_rsi_warmup_nan(self):
        """calc_rsi with period=14 should return NaN for bars 0-13."""
        from neutralgrid.indicators.technical import calc_rsi

        period = 14
        rsi = calc_rsi(self.closes, period)
        assert np.all(np.isnan(rsi[:period])), (
            f"Bars 0..{period-1} must be NaN; got {rsi[:period]}"
        )

    def test_adx_warmup_nan(self):
        """calc_adx with period=14 should return NaN for bars 0-27 (2*period)."""
        from neutralgrid.indicators.technical import calc_adx

        period = 14
        warmup = 2 * period
        adx, plus_di, minus_di = calc_adx(self.highs, self.lows, self.closes, period)
        assert np.all(np.isnan(adx[:warmup])), (
            f"ADX bars 0..{warmup-1} must be NaN; got {adx[:warmup]}"
        )
        assert np.all(np.isnan(plus_di[:warmup])), (
            f"+DI bars 0..{warmup-1} must be NaN"
        )
        assert np.all(np.isnan(minus_di[:warmup])), (
            f"-DI bars 0..{warmup-1} must be NaN"
        )

    def test_ema_slope_nan_init(self):
        """calc_ema_slope should use NaN (not 0) for initialization."""
        from neutralgrid.indicators.technical import calc_ema_slope

        ema = np.arange(50, dtype=float)
        slopes = calc_ema_slope(ema, lookback=5)
        # First `lookback` entries must be NaN (not zero)
        assert np.all(np.isnan(slopes[:5])), (
            f"calc_ema_slope initialises with NaN; got {slopes[:5]}"
        )


# ###########################################################################
# 3. Triple Barrier
# ###########################################################################

class TestTripleBarrier:
    """Verify PT/SL exit at barrier level (not wick extreme), t1 storage,
    double-touch policy, and time barrier labeling."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from neutralgrid.models.triple_barrier import (
            TripleBarrierConfig,
            TripleBarrierLabeler,
        )
        self.TripleBarrierConfig = TripleBarrierConfig
        self.TripleBarrierLabeler = TripleBarrierLabeler

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _flat_prices(n=50, base=100.0):
        """Return close / high / low arrays that stay perfectly flat."""
        closes = np.full(n, base)
        highs = np.full(n, base)
        lows = np.full(n, base)
        return closes, highs, lows

    # -- tests -----------------------------------------------------------
    def test_pt_exit_at_barrier_level(self):
        """When high exceeds PT barrier, exit price == pt_price, NOT the high."""
        cfg = self.TripleBarrierConfig(pt_pct=10.0, sl_pct=-10.0,
                                        time_barrier_hours=100,
                                        timeframe_minutes=1)
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)

        # Bar 5: high shoots to 120 (well past PT at 110)
        highs[5] = 120.0

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows
        )
        assert result is not None
        assert result.label == 1, "Should hit PT"
        expected_pt_price = 100.0 * 1.10  # 110
        assert result.exit_price == pytest.approx(expected_pt_price, abs=1e-6), (
            f"Exit price should be barrier level {expected_pt_price}, "
            f"not the wick extreme {highs[5]}"
        )

    def test_sl_exit_at_barrier_level(self):
        """When low crosses SL barrier, exit price == sl_price, NOT the low."""
        cfg = self.TripleBarrierConfig(pt_pct=10.0, sl_pct=-10.0,
                                        time_barrier_hours=100,
                                        timeframe_minutes=1)
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)

        # Bar 3: low drops to 80 (well past SL at 90)
        lows[3] = 80.0

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows
        )
        assert result is not None
        assert result.label == -1, "Should hit SL"
        expected_sl_price = 100.0 * 0.90  # 90
        assert result.exit_price == pytest.approx(expected_sl_price, abs=1e-6), (
            f"Exit price should be barrier level {expected_sl_price}, "
            f"not the wick extreme {lows[3]}"
        )

    def test_t1_populated(self):
        """t1 should be populated when timestamps provided."""
        cfg = self.TripleBarrierConfig(pt_pct=10.0, sl_pct=-10.0,
                                        time_barrier_hours=100,
                                        timeframe_minutes=1)
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)
        timestamps = pd.date_range("2024-01-01", periods=n, freq="1min").values

        # PT touch at bar 5
        highs[5] = 120.0

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows,
            timestamps=timestamps,
        )
        assert result is not None
        assert result.t1 is not None, "t1 should be populated when timestamps given"

    def test_double_touch_sl_first(self):
        """Same-bar PT and SL touch with sl_first policy => label = -1."""
        cfg = self.TripleBarrierConfig(pt_pct=10.0, sl_pct=-10.0,
                                        time_barrier_hours=100,
                                        timeframe_minutes=1)
        labeler = self.TripleBarrierLabeler(cfg, double_touch_policy="sl_first")

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)

        # Bar 4: touches both barriers
        highs[4] = 115.0  # >= PT at 110
        lows[4] = 85.0    # <= SL at 90

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows
        )
        assert result is not None
        assert result.label == -1, "sl_first policy should label -1"
        assert result.double_touch is True

    def test_time_barrier_label(self):
        """When neither PT nor SL hit, label should be 0."""
        cfg = self.TripleBarrierConfig(pt_pct=10.0, sl_pct=-10.0,
                                        time_barrier_hours=0.5,
                                        timeframe_minutes=1)
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)  # within PT
        lows = np.full(n, 99.5)    # within SL

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows
        )
        assert result is not None
        assert result.label == 0, "Flat price should hit time barrier => label 0"
        assert result.barrier_hit == "time"


# ###########################################################################
# 4. CPCV
# ###########################################################################

class TestCPCV:
    """Verify purge_pct=0.0 fix, deflated Sharpe, PBO, and embargo defaults."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.backtest.cpcv import (
            CPCV,
            CPCVConfig,
            DeflatedSharpeCalculator,
            compute_pbo,
        )
        self.CPCV = CPCV
        self.CPCVConfig = CPCVConfig
        self.DeflatedSharpeCalculator = DeflatedSharpeCalculator
        self.compute_pbo = compute_pbo

    @staticmethod
    def _make_df(n=120):
        """Create a synthetic time-ordered DataFrame."""
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"start_time_utc": ts, "value": np.arange(n)})

    # -- tests -----------------------------------------------------------
    def test_purge_zero_removes_nothing(self):
        """purge_pct=0.0 and embargo_pct=0.0 => no samples should be purged
        (purge_size = 0, not min-clamped to 1)."""
        cfg = self.CPCVConfig(n_groups=4, n_test_groups=1,
                              purge_pct=0.0, embargo_pct=0.0,
                              purge_hours=0.0, embargo_hours=0.0,
                              horizon_hours=0.0)
        cpcv = self.CPCV(cfg)

        df = self._make_df(120)
        total_purged = 0
        for train_idx, test_idx in cpcv.split(df):
            # Every row should appear in either train or test
            all_present = set(train_idx.tolist()) | set(test_idx.tolist())
            total_purged += len(df) - len(all_present)

        # With zero purge/embargo, purge_size and embargo_size should both be 0
        # so every row should be accounted for in every split
        assert total_purged == 0, (
            f"With purge=0 and embargo=0, no samples should be purged; "
            f"got {total_purged} purged across all splits"
        )

    def test_deflated_sharpe_n_trials_1(self):
        """With 1 trial, expected_max_sharpe should be 0.0."""
        calc = self.DeflatedSharpeCalculator(n_trials=1)
        assert calc.expected_max_sharpe() == pytest.approx(0.0, abs=1e-12)

    def test_deflated_sharpe_multiple_trials(self):
        """With many trials, deflated Sharpe should be less than raw."""
        calc = self.DeflatedSharpeCalculator(n_trials=100)
        result = calc.deflate(raw_sharpe=1.5, n_observations=252)
        assert result.deflated_sharpe < result.raw_sharpe, (
            f"Deflated ({result.deflated_sharpe:.4f}) should be < "
            f"raw ({result.raw_sharpe:.4f}) with 100 trials"
        )

    def test_pbo_all_positive_oos(self):
        """If all OOS Sharpe ratios > 0, PBO should be 0.0."""
        # Fake returns that produce positive Sharpe per path
        path_returns: Dict[int, np.ndarray] = {}
        np.random.seed(99)
        for i in range(10):
            # Positive mean -> positive Sharpe
            path_returns[i] = np.random.rand(50) * 0.01 + 0.005
        result = self.compute_pbo(path_returns, n_paths=10)
        assert result.pbo == pytest.approx(0.0, abs=1e-12), (
            f"All-positive OOS Sharpes => pbo=0.0; got {result.pbo}"
        )

    def test_pbo_all_negative_oos(self):
        """If all OOS Sharpe ratios < 0, PBO should be 1.0."""
        path_returns: Dict[int, np.ndarray] = {}
        np.random.seed(99)
        for i in range(10):
            # Negative mean -> negative Sharpe
            path_returns[i] = np.random.rand(50) * 0.01 - 0.02
        result = self.compute_pbo(path_returns, n_paths=10)
        assert result.pbo == pytest.approx(1.0, abs=1e-12), (
            f"All-negative OOS Sharpes => pbo=1.0; got {result.pbo}"
        )

    def test_embargo_defaults_to_one_point_five(self):
        """CPCVConfig().embargo_hours should be 1.5 (25% of 6h holding horizon)."""
        cfg = self.CPCVConfig()
        assert cfg.embargo_hours == pytest.approx(1.5), (
            f"Default embargo_hours should be 1.5; got {cfg.embargo_hours}"
        )

    def test_pbo_detects_negative_sharpe_paths(self):
        """PBO should be > 0 when some paths have negative-mean centered payoffs.

        Creates a mix of positive-mean and negative-mean centered payoff arrays
        to verify that the PBO computation can detect overfitting (i.e., paths
        where OOS Sharpe < 0).
        """
        np.random.seed(42)
        path_returns: Dict[int, np.ndarray] = {}
        # 5 paths with positive-mean centered payoffs
        for i in range(5):
            path_returns[i] = np.random.randn(50) * 0.01 + 0.005
        # 5 paths with negative-mean centered payoffs
        for i in range(5, 10):
            path_returns[i] = np.random.randn(50) * 0.01 - 0.005

        result = self.compute_pbo(path_returns, n_paths=10)
        assert result.pbo > 0, (
            f"PBO should be > 0 with mixed positive/negative paths; got {result.pbo}"
        )
        assert result.n_overfit > 0, (
            f"n_overfit should be > 0 with negative-mean paths; got {result.n_overfit}"
        )


# ###########################################################################
# 5. Meta-Labeler
# ###########################################################################

class TestMetaLabeler:
    """Verify hurdle conversion, feature list, and imputer mismatch warning."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.models.meta_labeler import (
            ACTIVE_SNAPSHOT_META_FEATURES,
            MetaLabeler,
            MetaLabelerConfig,
        )
        self.ACTIVE_SNAPSHOT_META_FEATURES = ACTIVE_SNAPSHOT_META_FEATURES
        self.MetaLabeler = MetaLabeler
        self.MetaLabelerConfig = MetaLabelerConfig

    def test_hurdle_conversion(self):
        """create_labels with hurdle_pct=5.0:
        - pnl_pct=4.9 -> label 0
        - pnl_pct=5.1 -> label 1
        (hurdle_pct=5.0 threshold in percentage units)."""
        config = self.MetaLabelerConfig(hurdle_pct=5.0)
        labeler = self.MetaLabeler(config)

        df = pd.DataFrame({"pnl_pct": [4.9, 5.1]})
        labels = labeler.create_labels(df, pnl_col="pnl_pct", sl_hit_col=None)
        assert labels.iloc[0] == 0, "4.9 < 5.0 hurdle => label 0"
        assert labels.iloc[1] == 1, "5.1 >= 5.0 hurdle => label 1"

    def test_feature_list_length(self):
        """MetaLabelerConfig().features should match the active feature profile.
        primary_pipeline_score excluded (circular dependency).
        hlabel excluded (information leakage: deterministically maps to y)."""
        expected = len(self.ACTIVE_SNAPSHOT_META_FEATURES)
        cfg = self.MetaLabelerConfig()
        assert len(cfg.features) == expected, (
            f"Expected {expected} features; got {len(cfg.features)}: {cfg.features}"
        )
        assert "primary_pipeline_score" not in cfg.features, (
            "'primary_pipeline_score' must NOT be in feature list (circular dependency)"
        )
        assert "hlabel" not in cfg.features, (
            "'hlabel' must NOT be in feature list (information leakage: maps to y)"
        )

    def test_imputer_mismatch_warning(self, caplog):
        """Loading a model with mismatched imputer dimensions should log a
        warning containing 'Imputer dimension mismatch'."""
        import pickle

        state = {
            "model": _FakeModel(),
            "feature_names": ["a", "b", "c"],
            "imputer": _FakeImputer(),
            "is_calibrated": False,
            "config": self.MetaLabelerConfig(),
            "metrics": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(state, f)
            tmp_path = f.name

        import warnings

        try:
            with caplog.at_level(logging.WARNING, logger="neutralgrid.models.meta_labeler"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    self.MetaLabeler.load(Path(tmp_path))
            assert any("Imputer dimension mismatch" in rec.message for rec in caplog.records), (
                "Expected 'Imputer dimension mismatch' warning in logs"
            )
        finally:
            os.unlink(tmp_path)


# ###########################################################################
# 6. Funding Rate
# ###########################################################################

class TestFundingRate:
    """Verify the code uses datetime.now(timezone.utc), not datetime.now()."""

    def test_funding_info_uses_utc(self):
        """Source code of get_funding_info must call datetime.now(timezone.utc)."""
        from neutralgrid.data.funding_rate import FundingRateProvider

        source = inspect.getsource(FundingRateProvider.get_funding_info)
        assert "datetime.now(timezone.utc)" in source, (
            "get_funding_info should use datetime.now(timezone.utc), "
            "not bare datetime.now()"
        )
        # Also verify the naive form is NOT used (no bare `datetime.now()`)
        # We check that every `datetime.now(` has `timezone.utc)` following it.
        # A simple heuristic: count occurrences.
        count_utc = source.count("datetime.now(timezone.utc)")
        count_any = source.count("datetime.now(")
        assert count_utc == count_any, (
            f"Found {count_any} calls to datetime.now() but only {count_utc} "
            f"use timezone.utc"
        )


# ###########################################################################
# 7. Data Curator
# ###########################################################################

class TestDataCurator:
    """Verify clean_ohlcv returns audit trail."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.data.curator import DataCurator
        self.DataCurator = DataCurator

    def test_clean_ohlcv_returns_report(self):
        """clean_ohlcv should return (df, report) where report has 'steps' key."""
        curator = self.DataCurator()
        df = pd.DataFrame({
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [100, 200, 300],
        })
        result = curator.clean_ohlcv(df)
        assert isinstance(result, tuple), "clean_ohlcv should return a tuple"
        assert len(result) == 2, "clean_ohlcv should return (df, report)"

        df_clean, report = result
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(report, dict)
        assert "steps" in report, "report dict must have 'steps' key"

    def test_clean_ohlcv_audit_trail(self):
        """With NaN data, report should log the fill action."""
        curator = self.DataCurator()
        df = pd.DataFrame({
            "open": [1.0, np.nan, 3.0],
            "high": [1.5, np.nan, 3.5],
            "low": [0.5, np.nan, 2.5],
            "close": [1.2, np.nan, 3.2],
            "volume": [100, np.nan, 300],
        })
        _, report = curator.clean_ohlcv(df)
        assert len(report["steps"]) > 0, "NaN data should produce at least one step"
        actions = [s["action"] for s in report["steps"]]
        assert any("fill" in a for a in actions), (
            f"Expected a fill action in steps; got actions: {actions}"
        )


# ###########################################################################
# 8. Market Data
# ###########################################################################

class TestMarketData:
    """Verify save_training_dataset handles single-row DataFrames."""

    def test_save_training_dataset_single_row(self):
        """save_training_dataset with 1-row DataFrame should not raise
        IndexError."""
        from neutralgrid.data.market_data import save_training_dataset

        # Build a minimal single-row DataFrame with the columns the function
        # expects (close_time for timeframe inference).
        ts = pd.Timestamp("2024-06-01 12:00", tz="UTC")
        df = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
            "close_time": [ts],
        })
        datasets = {"TESTUSDT": df}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "single_row_test"
            # This must NOT raise IndexError
            result_path = save_training_dataset(datasets, output_dir)
            assert result_path.exists()


# ###########################################################################
# 9. Backtest Realistic
# ###########################################################################

class TestBacktestRealistic:
    """Verify Sharpe annualization uses 525600 for crypto 24/7 markets."""

    def test_sharpe_annualization_crypto(self):
        """The Sharpe calculation must use sqrt(525600) = sqrt(365*24*60),
        NOT sqrt(362880) = sqrt(252*24*60)."""
        source_path = PROJECT_ROOT / "backtest" / "backtest_realistic.py"
        source = source_path.read_text()

        assert "525600" in source, (
            "backtest_realistic.py must use 525600 (365*24*60) for "
            "crypto annualization"
        )
        # Verify the old incorrect value is NOT present
        assert "362880" not in source, (
            "backtest_realistic.py must NOT use 362880 (252*24*60)"
        )


# ###########################################################################
# 10. Walk-Forward Evaluation
# ###########################################################################

class TestWalkForwardEvaluation:
    """Verify walk-forward splits have proper embargo gap."""

    def test_walk_forward_embargo_gap(self):
        """test_start should be > train_end by embargo_days."""
        from neutralgrid.backtest.evaluate import create_walk_forward_splits

        np.random.seed(42)
        n_symbols = 5
        n_bars = 24 * 365  # ~1 year of 1H data

        datasets = {}
        base_ts = pd.Timestamp("2023-01-01", tz="UTC")
        for i in range(n_symbols):
            ts = pd.date_range(base_ts, periods=n_bars, freq="h")
            prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.1)
            df = pd.DataFrame({
                "open_time": ts,
                "close_time": ts + timedelta(hours=1),
                "open": prices,
                "high": prices + 0.5,
                "low": prices - 0.5,
                "close": prices + 0.05,
                "volume": np.random.rand(n_bars) * 1000,
            })
            datasets[f"SYM{i}USDT"] = df

        embargo_days = 2.0
        splits = create_walk_forward_splits(
            datasets,
            train_days=90,
            test_days=14,
            step_days=14,
            min_train_bars=200,
            embargo_days=embargo_days,
        )

        assert len(splits) > 0, "Should produce at least one split"

        for s in splits:
            gap = (s.test_start - s.train_end).total_seconds() / 86400.0
            assert gap >= embargo_days - 1e-6, (
                f"Split {s.split_id}: test_start - train_end = {gap:.4f} days, "
                f"expected >= {embargo_days}"
            )


# ###########################################################################
# Smoke: source code integrity checks
# ###########################################################################

class TestSourceCodeIntegrity:
    """Quick source-level smoke checks for patterns that confirm fixes."""

    def test_sample_weights_cond2_transposed(self):
        """cond2 in compute_concurrency_matrix must use t0 reshaped as (1, -1)
        and t1 reshaped as (-1, 1), which is the transposed form.

        Before the fix: cond2 was identical to cond1 (both checked t0[i] <= t1[j]).
        After the fix: cond2 checks t0[j] <= t1[i]."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "training" / "sample_weights.py"
        )
        source = source_path.read_text()

        # The fixed cond2 should have the reversed reshape:
        # t0_arr.reshape(1, -1) <= t1_arr.reshape(-1, 1)
        assert "t0_arr.reshape(1, -1)" in source, (
            "cond2 should use t0_arr.reshape(1, -1)"
        )
        assert "t1_arr.reshape(-1, 1)" in source, (
            "cond2 should use t1_arr.reshape(-1, 1)"
        )

    def test_curator_returns_tuple(self):
        """clean_ohlcv return type annotation should include report dict."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "data" / "curator.py"
        )
        source = source_path.read_text()
        assert "Tuple[pd.DataFrame, Dict" in source, (
            "clean_ohlcv should have return type Tuple[pd.DataFrame, Dict[...]]"
        )

    def test_evaluate_embargo_days_parameter(self):
        """create_walk_forward_splits must accept embargo_days parameter."""
        from neutralgrid.backtest.evaluate import create_walk_forward_splits

        sig = inspect.signature(create_walk_forward_splits)
        assert "embargo_days" in sig.parameters, (
            "create_walk_forward_splits must have an embargo_days parameter"
        )
        # Default should be 2.0
        default = sig.parameters["embargo_days"].default
        assert default == pytest.approx(2.0), (
            f"embargo_days default should be 2.0; got {default}"
        )

    def test_meta_labeler_calibration_uses_prefit(self):
        """Meta-labeler calibration should use cv='prefit' (temporal holdout),
        not cv=3 (which leaks future data)."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "models" / "meta_labeler.py"
        )
        source = source_path.read_text()
        assert 'cv="prefit"' in source or "cv='prefit'" in source, (
            "Calibration should use cv='prefit' for temporal holdout"
        )

    def test_meta_labeler_atomic_write(self):
        """Meta-labeler save should use atomic writes (tempfile + os.replace)."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "models" / "meta_labeler.py"
        )
        source = source_path.read_text()
        assert "os.replace" in source, (
            "save() should use os.replace for atomic writes"
        )
        assert "tempfile" in source or "mkstemp" in source, (
            "save() should use tempfile for atomic writes"
        )


# ###########################################################################
# Entry point
# ###########################################################################

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
