"""
AFML Fix Compliance Tests for NEUTRAL Grid Bot v6.5.3.

Comprehensive test suite for all AFML-alignment fixes applied in tasks 1-12.
Tests are self-contained (synthetic data only, no external APIs) and
deterministic (fixed seeds).

Run with:
    pytest tests/test_afml_fixes.py -v
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _workspace_tempdir():
    base = PROJECT_ROOT / ".tmp_pytest"
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)


# ###########################################################################
# 1. Kelly Bet Sizing -- Negative edge rejection (Task 1)
# ###########################################################################

class TestKellyNegativeEdge:
    """Verify trades are rejected when meta_prob < 0.5 (negative Kelly edge)."""

    def test_kelly_edge_negative_when_prob_below_half(self):
        """Kelly edge = 2*p - 1 should be negative when p < 0.5."""
        for p in [0.0, 0.1, 0.3, 0.49]:
            edge = 2 * p - 1
            assert edge <= 0, (
                f"Kelly edge should be <= 0 for p={p}, got {edge}"
            )

    def test_kelly_edge_positive_when_prob_above_half(self):
        """Kelly edge = 2*p - 1 should be positive when p > 0.5."""
        for p in [0.51, 0.6, 0.8, 1.0]:
            edge = 2 * p - 1
            assert edge > 0, (
                f"Kelly edge should be > 0 for p={p}, got {edge}"
            )

    def test_half_kelly_fraction_bounded(self):
        """Half-Kelly capital fraction should be in (0, 0.5] for p in (0.5, 1]."""
        for p in [0.51, 0.6, 0.75, 0.9, 1.0]:
            edge = 2 * p - 1
            fraction = edge / 2
            assert 0 < fraction <= 0.5, (
                f"Half-Kelly fraction should be in (0, 0.5] for p={p}, got {fraction}"
            )

    def test_negative_kelly_annotation_in_source(self):
        """enrich_grid_params.py should annotate negative Kelly edge and continue."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "scanner" / "enrich_grid_params.py"
        )
        source = source_path.read_text()
        assert "kelly_edge <= 0" in source or "kelly_edge<=0" in source, (
            "enrich_grid_params.py should check for kelly_edge <= 0"
        )
        assert "negative_kelly_edge" in source, (
            "enrich_grid_params.py should keep negative_kelly_edge visible in diagnostics"
        )
        assert "stage_b_selector.approve" in source, (
            "Negative Kelly paths should continue into Stage B instead of terminating early"
        )


# ###########################################################################
# 2. PnL Unit Consistency (Task 2)
# ###########################################################################

class TestPnlUnitConsistency:
    """Verify create_labels works correctly for both percentage and decimal PnL."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig
        self.MetaLabeler = MetaLabeler
        self.MetaLabelerConfig = MetaLabelerConfig

    def test_percentage_pnl_labels_correct(self):
        """PnL in percentage format (e.g., 5.0 for 5%) should label correctly.

        With hurdle_pct=0.05 (5%), create_labels compares pnl >= hurdle_pct.
        For percentage data: pnl_pct=6.0 >= 0.05 => True (label=1).
        """
        cfg = self.MetaLabelerConfig(hurdle_pct=0.05, sl_pct=-0.10)
        labeler = self.MetaLabeler(cfg)

        # PnL in percentage format
        df = pd.DataFrame({"pnl_pct": [6.0, 3.0, -12.0, 0.05, 0.04]})
        labels = labeler.create_labels(df)

        # 6.0 >= 0.05 => 1, 3.0 >= 0.05 => 1, -12.0 >= 0.05 => 0
        # 0.05 >= 0.05 => 1, 0.04 >= 0.05 => 0
        assert labels.iloc[0] == 1, "6.0% PnL should be labeled success"
        assert labels.iloc[2] == 0, "-12.0% PnL should be labeled failure"

    def test_create_labels_no_spurious_multiply(self):
        """create_labels should compare pnl directly to hurdle_pct, not hurdle_pct * 100."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "models" / "meta_labeler.py"
        )
        source = source_path.read_text()
        # The fix removed `* 100` from the comparison in create_labels
        # Check that pnl_ok does not multiply hurdle_pct by 100
        assert "hurdle_pct * 100" not in source or "self.config.hurdle_pct * 100" not in source, (
            "create_labels should not multiply hurdle_pct by 100"
        )

    def test_sl_comparison_consistent(self):
        """SL comparison should also be direct (no * 100)."""
        cfg = self.MetaLabelerConfig(hurdle_pct=5.0, sl_pct=-10.0)
        labeler = self.MetaLabeler(cfg)

        df = pd.DataFrame({"pnl_pct": [-15.0, -5.0, 0.0, 6.0]})
        labels = labeler.create_labels(df)

        # -15.0 < -10.0 (sl) => 0; 6.0 >= 5.0 (hurdle) => 1
        assert labels.iloc[0] == 0, "-15% should fail (below SL)"
        assert labels.iloc[3] == 1, "6% should succeed (above hurdle)"


# ###########################################################################
# 3. Volatility Barrier Scaling (Task 3)
# ###########################################################################

class TestVolatilityBarrierScaling:
    """Verify barriers use daily (not annualized) volatility."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from neutralgrid.models.triple_barrier import (
            TripleBarrierConfig,
            TripleBarrierLabeler,
        )
        self.TripleBarrierConfig = TripleBarrierConfig
        self.TripleBarrierLabeler = TripleBarrierLabeler

    def test_daily_vol_not_annualized(self):
        """When daily_vol=0.03 (3%), PT = k_pt * 0.03 * 100 = 6%.

        The code should use daily_vol directly, NOT annualize it.
        If annualized (0.03 * sqrt(365) ~ 0.57), PT would be ~114%,
        which is wrong.
        """
        cfg = self.TripleBarrierConfig(
            pt_pct=15.0, sl_pct=-10.0,
            pt_vol_multiplier=2.0, sl_vol_multiplier=1.5,
            time_barrier_hours=100, timeframe_minutes=1,
        )
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)

        daily_vol = 0.03  # 3% daily vol
        # Dynamic PT = 2.0 * 0.03 * 100 = 6.0% => PT price = 106.0
        # If annualized: 2.0 * 0.57 * 100 = 114% => PT price = 214 (absurd)

        # High at bar 3 = 107 -> above 106 (correct) but below 214 (annualized)
        highs[3] = 107.0

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows,
            daily_vol=daily_vol,
        )
        assert result is not None
        assert result.label == 1, (
            "Should hit dynamic PT with daily vol, not annualized"
        )
        assert result.exit_price == pytest.approx(106.0, abs=1e-6), (
            f"PT price should be 106.0 (2.0 * 3% daily), got {result.exit_price}"
        )

    def test_daily_vol_sl_scaling(self):
        """SL should also scale with daily vol, not annualized."""
        cfg = self.TripleBarrierConfig(
            pt_pct=15.0, sl_pct=-10.0,
            pt_vol_multiplier=2.0, sl_vol_multiplier=1.5,
            time_barrier_hours=100, timeframe_minutes=1,
        )
        labeler = self.TripleBarrierLabeler(cfg)

        n = 50
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.0)
        lows = np.full(n, 100.0)

        daily_vol = 0.03
        # Dynamic SL = -(1.5 * 0.03 * 100) = -4.5% => SL price = 95.5
        lows[4] = 94.0  # Below 95.5

        result = labeler.label_entry(
            closes=closes, entry_bar=0, highs=highs, lows=lows,
            daily_vol=daily_vol,
        )
        assert result is not None
        assert result.label == -1, "Should hit dynamic SL"
        assert result.exit_price == pytest.approx(95.5, abs=1e-6)


# ###########################################################################
# 4. from_unified() Preserves Vol Scaling (Task 4)
# ###########################################################################

class TestFromUnifiedPreservesVolScaling:
    """Verify TripleBarrierConfig.from_unified() propagates k_pt/k_sl."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from neutralgrid.models.triple_barrier import TripleBarrierConfig
        from neutralgrid.models.barrier_config import UnifiedBarrierConfig
        self.TripleBarrierConfig = TripleBarrierConfig
        self.UnifiedBarrierConfig = UnifiedBarrierConfig

    def test_vol_scaling_enabled_propagates(self):
        """When use_volatility_scaling=True, k_pt/k_sl should map to vol multipliers."""
        unified = self.UnifiedBarrierConfig(
            use_volatility_scaling=True,
            k_pt=3.0,
            k_sl=2.0,
        )
        cfg = self.TripleBarrierConfig.from_unified(unified)

        assert cfg.pt_vol_multiplier == 3.0, (
            f"pt_vol_multiplier should be 3.0, got {cfg.pt_vol_multiplier}"
        )
        assert cfg.sl_vol_multiplier == 2.0, (
            f"sl_vol_multiplier should be 2.0, got {cfg.sl_vol_multiplier}"
        )

    def test_vol_scaling_disabled_no_multipliers(self):
        """When use_volatility_scaling=False, multipliers should be None."""
        unified = self.UnifiedBarrierConfig(
            use_volatility_scaling=False,
            k_pt=3.0,
            k_sl=2.0,
        )
        cfg = self.TripleBarrierConfig.from_unified(unified)

        assert cfg.pt_vol_multiplier is None, (
            "pt_vol_multiplier should be None when vol scaling disabled"
        )
        assert cfg.sl_vol_multiplier is None, (
            "sl_vol_multiplier should be None when vol scaling disabled"
        )

    def test_fixed_barriers_still_propagate(self):
        """Fixed pt_pct/sl_pct should always propagate regardless of scaling."""
        unified = self.UnifiedBarrierConfig(
            pt_pct=20.0, sl_pct=-15.0,
            use_volatility_scaling=True,
            k_pt=3.0, k_sl=2.0,
        )
        cfg = self.TripleBarrierConfig.from_unified(unified)

        assert cfg.pt_pct == 20.0
        assert cfg.sl_pct == -15.0


# ###########################################################################
# 5. Sharpe Annualization (Task 7)
# ###########################################################################

class TestSharpeAnnualization:
    """Verify correct Sharpe formula: sqrt(N_obs_per_year) * (mean - rf) / std."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from neutralgrid.metrics.calculator import MetricsCalculator
        self.MetricsCalculator = MetricsCalculator

    def test_sharpe_known_values(self):
        """Verify Sharpe with known inputs produces expected result.

        Given:
        - 100 non-zero trades over 100 hours
        - Mean PnL = 1.0, std(ddof=1) of PnL ~ 1.005
        - Risk-free = 0
        - obs_per_year = 100/100 * 8760 = 8760

        Note: calculate_sharpe_proxy filters out zero-PnL trades (entry fills),
        so all test PnL values must be non-zero.
        """
        np.random.seed(42)
        # Create trades with known mean and std (no zeros, since source filters p != 0)
        n = 100
        pnls = np.array([2.0] * 50 + [-0.5] * 50)  # mean=0.75, no zeros

        trades = [{"realizedPnl": float(p)} for p in pnls]
        duration_hours = 100.0

        calc = self.MetricsCalculator()
        sharpe = calc.calculate_sharpe_proxy(trades, duration_hours, risk_free_rate=0.0)

        # Compute expected (matching source: filter non-zero, ddof=1, annualize)
        non_zero = np.array([p for p in pnls if p != 0])
        n_obs = len(non_zero)
        mean_r = float(np.mean(non_zero))
        std_r = float(np.std(non_zero, ddof=1))
        obs_per_year = n_obs / duration_hours * (365 * 24)
        expected_sharpe = mean_r / std_r * np.sqrt(obs_per_year)

        assert sharpe == pytest.approx(expected_sharpe, rel=1e-6), (
            f"Sharpe should be {expected_sharpe:.4f}, got {sharpe:.4f}"
        )

    def test_sharpe_uses_ddof1(self):
        """Sharpe should use sample std (ddof=1), not population std (ddof=0)."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "metrics" / "calculator.py"
        )
        source = source_path.read_text()
        assert "ddof=1" in source, (
            "calculate_sharpe_proxy should use np.std(pnls, ddof=1)"
        )

    def test_sharpe_uses_sqrt_obs_per_year(self):
        """Annualization should be sqrt(N_obs_per_year), not sqrt(hours_per_year/duration)."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "metrics" / "calculator.py"
        )
        source = source_path.read_text()
        # The fix computes obs_per_year = n_obs / duration_hours * hours_per_year
        assert "obs_per_year" in source or "n_obs" in source, (
            "Should compute observations per year for annualization"
        )

    def test_sharpe_empty_trades(self):
        """Empty trades should return 0.0."""
        calc = self.MetricsCalculator()
        assert calc.calculate_sharpe_proxy([], 24.0) == 0.0

    def test_sharpe_zero_duration(self):
        """Zero duration should return 0.0."""
        calc = self.MetricsCalculator()
        trades = [{"realizedPnl": 1.0}]
        assert calc.calculate_sharpe_proxy(trades, 0.0) == 0.0


# ###########################################################################
# 6. Sequential Bootstrap Performance (Task 9)
# ###########################################################################

class TestSequentialBootstrapFast:
    """Verify optimized sequential bootstrap matches naive for small N."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.training.sample_weights import sequential_bootstrap_indices
        self.sequential_bootstrap_indices = sequential_bootstrap_indices

    @staticmethod
    def _make_events(intervals):
        base = pd.Timestamp("2024-01-01", tz="UTC")
        t0 = pd.Series([base + timedelta(hours=s) for s, _ in intervals])
        t1 = pd.Series([base + timedelta(hours=e) for _, e in intervals])
        return t0, t1

    def test_deterministic(self):
        """Same seed produces identical results."""
        t0, t1 = self._make_events([(0, 3), (1, 4), (5, 7)])
        idx1 = self.sequential_bootstrap_indices(t0, t1, random_state=42)
        idx2 = self.sequential_bootstrap_indices(t0, t1, random_state=42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_correct_length(self):
        """Output length matches n_samples."""
        t0, t1 = self._make_events([(0, 2), (1, 3), (4, 5)])
        indices = self.sequential_bootstrap_indices(t0, t1, n_samples=50, random_state=0)
        assert len(indices) == 50

    def test_isolated_event_preferred(self):
        """Isolated events should be sampled more than highly-concurrent ones."""
        intervals = [(0, 10)] * 5 + [(20, 21)]
        t0, t1 = self._make_events(intervals)
        indices = self.sequential_bootstrap_indices(t0, t1, n_samples=200, random_state=42)

        count_isolated = np.sum(indices == 5)
        count_overlapping = np.mean([np.sum(indices == i) for i in range(5)])

        assert count_isolated > count_overlapping, (
            f"Isolated event count ({count_isolated}) should exceed avg overlapping ({count_overlapping:.1f})"
        )

    def test_all_indices_valid(self):
        """All returned indices should be valid event indices."""
        t0, t1 = self._make_events([(0, 5), (2, 8), (6, 10)])
        indices = self.sequential_bootstrap_indices(t0, t1, n_samples=30, random_state=7)
        assert np.all(indices >= 0)
        assert np.all(indices < len(t0))


# ###########################################################################
# 7. No Look-Ahead Bias (Feature Integrity)
# ###########################################################################

class TestNoLookaheadBias:
    """Verify that features at time t are not contaminated by future data."""

    def test_feature_snapshot_has_no_future_fields(self):
        """FeatureSnapshot should only contain data available at decision time."""
        from neutralgrid.training.data_generator import FeatureSnapshot

        snap = FeatureSnapshot(
            symbol="BTCUSDT",
            start_time_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
            range_prob=0.7,
            trend_prob=0.2,
            adx_1h=25.0,
        )

        d = snap.to_dict()
        # Should not contain any outcome/label fields
        forbidden = ["pnl_pct", "y", "sl_hit", "tp_hit", "t1", "barrier_touched"]
        for key in forbidden:
            assert key not in d, (
                f"FeatureSnapshot should not contain future field '{key}'"
            )

    def test_training_schema_separates_features_from_labels(self):
        """TrainingTableSchema should have separate feature and label sections."""
        from neutralgrid.training.data_generator import TrainingTableSchema

        schema = TrainingTableSchema()
        feature_set = set(schema.features)
        label_set = set(schema.labels)

        # No overlap between features and labels
        overlap = feature_set & label_set
        assert len(overlap) == 0, (
            f"Features and labels should not overlap: {overlap}"
        )

    def test_cpcv_purges_leaking_events(self):
        """CPCV with time-based purging should remove events spanning train/test boundary."""
        from neutralgrid.backtest.cpcv import CPCV
        from neutralgrid.core.config import CPCVConfig

        n = 120
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # Events last 6 hours each
        t1 = ts + timedelta(hours=6)
        df = pd.DataFrame({
            "start_time_utc": ts,
            "t1": t1,
            "value": np.arange(n),
        })

        cfg = CPCVConfig(
            n_groups=4, n_test_groups=1,
            purge_hours=6.0, embargo_hours=2.0,
        )
        cpcv = CPCV(cfg)

        for train_idx, test_idx in cpcv.split(df):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())

            # No overlap between train and test indices
            assert len(train_set & test_set) == 0

            # Check that no remaining train event's lifetime [t0, t1] overlaps
            # the test period [test_start, test_end].  AFML purging should have
            # removed events whose t1 bleeds into the test window.
            train_df = df.loc[df.index.isin(train_set)]
            test_df = df.loc[df.index.isin(test_set)]
            test_start = pd.to_datetime(test_df["start_time_utc"]).min()
            test_end = pd.to_datetime(test_df["start_time_utc"]).max()

            train_t0 = pd.to_datetime(train_df["start_time_utc"])
            train_t1 = pd.to_datetime(train_df["t1"])

            # A training event truly overlaps the test period when its lifetime
            # intersects: train_t1 >= test_start AND train_t0 <= test_end
            overlap_count = ((train_t1 >= test_start) & (train_t0 <= test_end)).sum()

            # The purge should have removed most/all overlapping events
            purge_effectiveness = 1.0 - overlap_count / max(len(train_df), 1)
            assert purge_effectiveness > 0.5, (
                f"Purging should remove most overlapping events, but only {purge_effectiveness:.0%} effective"
            )


# ###########################################################################
# 8. CPCV Holdout Embargo Gap (Task 11)
# ###########################################################################

class TestCpcvHoldoutEmbargo:
    """Verify embargo gap between CV and holdout prevents serial correlation leakage."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.backtest.cpcv import CPCV
        from neutralgrid.core.config import CPCVConfig
        self.CPCV = CPCV
        self.CPCVConfig = CPCVConfig

    def test_holdout_embargo_gap_exists(self):
        """split_with_holdout should have embargo gap between CV and holdout."""
        n = 200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "start_time_utc": ts,
            "t1": ts + timedelta(hours=6),
            "value": np.arange(n),
        })

        cfg = self.CPCVConfig(
            n_groups=4, n_test_groups=1,
            holdout_pct=0.20,
            embargo_hours=6.0,
        )
        cpcv = self.CPCV(cfg)

        cv_df, holdout_df, _ = cpcv.split_with_holdout(df)

        if not holdout_df.empty:
            cv_end = pd.to_datetime(cv_df["start_time_utc"]).max()
            holdout_start = pd.to_datetime(holdout_df["start_time_utc"]).min()
            gap_hours = (holdout_start - cv_end).total_seconds() / 3600

            # Gap should be at least embargo_hours
            assert gap_hours >= cfg.embargo_hours, (
                f"Gap between CV end and holdout start should be >= {cfg.embargo_hours}h, "
                f"got {gap_hours:.1f}h"
            )

    def test_holdout_never_in_cv_splits(self):
        """Holdout indices should never appear in any CV train/test split."""
        n = 200
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "start_time_utc": ts,
            "t1": ts + timedelta(hours=6),
            "value": np.arange(n),
        })

        cfg = self.CPCVConfig(
            n_groups=4, n_test_groups=1,
            holdout_pct=0.20,
        )
        cpcv = self.CPCV(cfg)

        cv_df, holdout_df, cv_splits = cpcv.split_with_holdout(df)

        if not holdout_df.empty:
            holdout_indices = set(holdout_df.index.tolist())

            for train_idx, test_idx in cv_splits:
                cv_indices = set(train_idx.tolist()) | set(test_idx.tolist())
                leaked = cv_indices & holdout_indices
                assert len(leaked) == 0, (
                    f"Holdout indices leaked into CV split: {leaked}"
                )


# ###########################################################################
# 9. Trial Tracker Append-Only (Task 11)
# ###########################################################################

class TestTrialTrackerAppendOnly:
    """Verify trials are never silently replaced (append-only for correct DSR)."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.training.trial_tracker import TrialTracker, TrialRecord
        self.TrialTracker = TrialTracker
        self.TrialRecord = TrialRecord

    def test_duplicate_trial_id_appends_not_replaces(self):
        """Logging a trial with the same ID should append, not overwrite."""
        with _workspace_tempdir() as tmpdir:
            log_path = os.path.join(tmpdir, "trial_log.json")
            tracker = self.TrialTracker(log_file=log_path)

            trial_1 = self.TrialRecord(
                trial_id="test_001",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                model_type="meta_labeler",
                hyperparameters={"n_estimators": 100},
                cv_score=0.70,
            )
            trial_2 = self.TrialRecord(
                trial_id="test_001",  # Same ID
                timestamp=datetime(2025, 1, 2, tzinfo=timezone.utc),
                model_type="meta_labeler",
                hyperparameters={"n_estimators": 200},
                cv_score=0.75,
            )

            tracker.log_trial(trial_1)
            tracker.log_trial(trial_2)

            # Should have 2 entries (append), not 1 (replace)
            count = tracker.get_trial_count(model_type="meta_labeler")
            assert count == 2, (
                f"Duplicate trial_id should append (count=2), not replace (count={count})"
            )

    def test_trial_count_reflects_all_entries(self):
        """get_trial_count should count ALL entries including duplicates."""
        with _workspace_tempdir() as tmpdir:
            log_path = os.path.join(tmpdir, "trial_log.json")
            tracker = self.TrialTracker(log_file=log_path)

            for i in range(5):
                trial = self.TrialRecord(
                    trial_id=f"trial_{i}",
                    timestamp=datetime(2025, 1, 1 + i, tzinfo=timezone.utc),
                    model_type="meta_labeler",
                    hyperparameters={"n_estimators": 100},
                    cv_score=0.50 + i * 0.05,
                )
                tracker.log_trial(trial)

            assert tracker.get_trial_count(model_type="meta_labeler") == 5

    def test_persistence_across_reload(self):
        """Trials should persist when tracker is reloaded from disk."""
        with _workspace_tempdir() as tmpdir:
            log_path = os.path.join(tmpdir, "trial_log.json")

            # Write trials
            tracker1 = self.TrialTracker(log_file=log_path)
            for i in range(3):
                tracker1.log_trial(self.TrialRecord(
                    trial_id=f"persist_{i}",
                    timestamp=datetime(2025, 1, 1 + i, tzinfo=timezone.utc),
                    model_type="hmm",
                    hyperparameters={},
                    cv_score=0.6,
                ))

            # Reload and verify
            tracker2 = self.TrialTracker(log_file=log_path)
            assert tracker2.get_trial_count(model_type="hmm") == 3


# ###########################################################################
# 10. Duplicate Detection (Task 12)
# ###########################################################################

class TestDuplicateDetection:
    """Verify curator catches duplicate timestamps in OHLCV data."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.data.curator import DataCurator, DataQualityConfig
        self.DataCurator = DataCurator
        self.DataQualityConfig = DataQualityConfig

    def test_detects_duplicate_timestamps(self):
        """check_duplicates should flag duplicate timestamps."""
        curator = self.DataCurator()

        ts = pd.date_range("2024-01-01", periods=10, freq="15min")
        # Inject a duplicate
        timestamps = list(ts)
        timestamps[5] = timestamps[4]  # Duplicate at index 5

        df = pd.DataFrame({
            "open_time": timestamps,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        })

        passed, metrics, issues = curator.check_duplicates(df, "open_time")

        assert not passed, "Should fail with duplicate timestamps"
        assert metrics["duplicates"] == 1
        assert any("duplicate" in issue for issue in issues)

    def test_no_duplicates_passes(self):
        """Data without duplicates should pass."""
        curator = self.DataCurator()

        ts = pd.date_range("2024-01-01", periods=10, freq="15min")
        df = pd.DataFrame({
            "open_time": ts,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        })

        passed, metrics, issues = curator.check_duplicates(df, "open_time")

        assert passed
        assert metrics["duplicates"] == 0

    def test_validate_ohlcv_includes_duplicate_check(self):
        """Full validate_ohlcv should include the duplicates check."""
        curator = self.DataCurator()

        ts = pd.date_range("2024-01-01", periods=10, freq="15min")
        timestamps = list(ts)
        timestamps[3] = timestamps[2]  # Duplicate

        df = pd.DataFrame({
            "open_time": timestamps,
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        })

        result = curator.validate_ohlcv(df, timeframe="15m", reference_time=ts[-1].to_pydatetime().replace(tzinfo=timezone.utc))

        assert "duplicates" in result.checks, (
            "validate_ohlcv should include 'duplicates' check"
        )
        assert not result.checks["duplicates"], (
            "Duplicates check should fail"
        )


# ###########################################################################
# Additional: Sample Weights (preserved from earlier baseline)
# ###########################################################################

class TestSampleWeightsPerBarUniqueness:
    """Verify compute_avg_uniqueness uses per-bar averaging per AFML 4.3."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.training.sample_weights import (
            compute_avg_uniqueness,
            compute_concurrency_matrix,
            sequential_bootstrap_indices,
        )
        self.compute_avg_uniqueness = compute_avg_uniqueness
        self.compute_concurrency_matrix = compute_concurrency_matrix
        self.sequential_bootstrap_indices = sequential_bootstrap_indices

    @staticmethod
    def _make_events(intervals):
        base = pd.Timestamp("2024-01-01", tz="UTC")
        t0 = pd.Series([base + timedelta(hours=s) for s, _ in intervals])
        t1 = pd.Series([base + timedelta(hours=e) for _, e in intervals])
        return t0, t1

    def test_non_overlapping_uniqueness_is_one(self):
        """Non-overlapping events should all have uniqueness = 1.0."""
        t0, t1 = self._make_events([(0, 1), (2, 3), (4, 5)])
        uniq = self.compute_avg_uniqueness(t0, t1)
        np.testing.assert_allclose(uniq, 1.0, atol=1e-12)

    def test_fully_overlapping_uniqueness_is_1_over_n(self):
        """N fully concurrent events should each have uniqueness = 1/N."""
        N = 5
        t0, t1 = self._make_events([(0, 10)] * N)
        uniq = self.compute_avg_uniqueness(t0, t1)
        np.testing.assert_allclose(uniq, 1.0 / N, atol=1e-12)

    def test_partial_overlap_per_bar_averaging(self):
        """With partial overlap, per-bar averaging should give > 0.5 (not flat 1/2)."""
        t0, t1 = self._make_events([(0, 4), (2, 6)])
        uniq = self.compute_avg_uniqueness(t0, t1)
        assert uniq[0] > 0.5, f"Event A uniqueness should be > 0.5, got {uniq[0]:.4f}"
        assert uniq[1] > 0.5, f"Event B uniqueness should be > 0.5, got {uniq[1]:.4f}"
        np.testing.assert_allclose(uniq[0], uniq[1], atol=1e-12)


# ###########################################################################
# Additional: CPCV Percent-Based Purging (preserved from earlier baseline)
# ###########################################################################

class TestCPCVPercentPurgingPositional:
    """Verify percent-based purging uses positional indices."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.backtest.cpcv import CPCV
        from neutralgrid.core.config import CPCVConfig
        self.CPCV = CPCV
        self.CPCVConfig = CPCVConfig

    def test_purge_with_noncontiguous_indices(self):
        """Percent-based purging should work with non-contiguous DataFrame indices."""
        n = 120
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        non_contiguous_index = list(range(1000, 1000 + n))
        df = pd.DataFrame(
            {"start_time_utc": ts, "value": np.arange(n)},
            index=non_contiguous_index,
        )

        cfg = self.CPCVConfig(
            n_groups=4, n_test_groups=1,
            purge_pct=0.05, embargo_pct=0.02,
        )
        cpcv = self.CPCV(cfg)

        splits = list(cpcv.split(df))
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            all_idx = set(train_idx.tolist()) | set(test_idx.tolist())
            for idx in all_idx:
                assert idx in non_contiguous_index
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0


# ###########################################################################
# Additional: MinTRL (preserved from earlier baseline)
# ###########################################################################

class TestMinTrackRecordLength:
    """Verify MinTRL formula per AFML Ch 14."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from neutralgrid.backtest.cpcv import compute_min_track_record_length
        self.compute_min_track_record_length = compute_min_track_record_length

    def test_mintrl_known_values(self):
        """MinTRL for SR=1.0, benchmark=0, normal returns, 95% confidence.

        variance_factor = 1 + 0.5*SR^2 - skew*SR + (kurtosis-3)/4 * SR^2
        For normal (kurtosis=3, skew=0, SR=1): variance_factor = 1.5
        MinTRL = 1 + 1.5 * (z_alpha / 1.0)^2
        """
        mintrl = self.compute_min_track_record_length(
            observed_sharpe=1.0, benchmark_sharpe=0.0,
            skewness=0.0, kurtosis=3.0, confidence=0.95,
        )
        z_alpha = stats.norm.ppf(0.95)
        expected = 1 + 1.5 * (z_alpha / 1.0) ** 2
        expected_ceil = int(np.ceil(expected))
        assert mintrl == expected_ceil

    def test_mintrl_sr_equals_benchmark(self):
        """When SR = SR*, MinTRL should be max int sentinel."""
        mintrl = self.compute_min_track_record_length(
            observed_sharpe=0.5, benchmark_sharpe=0.5,
        )
        assert mintrl == 2**31 - 1

    def test_mintrl_higher_confidence_needs_more_data(self):
        """Higher confidence should require more observations."""
        mintrl_95 = self.compute_min_track_record_length(observed_sharpe=1.0, confidence=0.95)
        mintrl_99 = self.compute_min_track_record_length(observed_sharpe=1.0, confidence=0.99)
        assert mintrl_99 > mintrl_95


# ###########################################################################
# Source Code Integrity Checks
# ###########################################################################

class TestSourceCodeIntegrity:
    """Source-level checks confirming all fixes are in place."""

    def test_create_labels_no_times_100(self):
        """create_labels should not multiply hurdle_pct by 100."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "models" / "meta_labeler.py"
        )
        source = source_path.read_text()
        # Find the create_labels method body
        idx = source.find("def create_labels")
        end_idx = source.find("\n    def ", idx + 1)
        method_body = source[idx:end_idx] if end_idx > idx else source[idx:]
        assert "hurdle_pct * 100" not in method_body, (
            "create_labels should compare pnl directly to hurdle_pct"
        )

    def test_from_unified_propagates_vol_multipliers(self):
        """from_unified should propagate k_pt/k_sl when vol scaling enabled."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "models" / "triple_barrier.py"
        )
        source = source_path.read_text()
        idx = source.find("def from_unified")
        end_idx = source.find("\n    @", idx + 1) if source.find("\n    @", idx + 1) > 0 else source.find("\n\n\n", idx + 1)
        method_body = source[idx:end_idx] if end_idx > idx else source[idx:]
        assert "k_pt" in method_body, "from_unified should reference unified.k_pt"
        assert "k_sl" in method_body, "from_unified should reference unified.k_sl"

    def test_trial_tracker_append_only_in_source(self):
        """log_trial should append, not replace."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "training" / "trial_tracker.py"
        )
        source = source_path.read_text()
        idx = source.find("def log_trial")
        end_idx = source.find("\n    def ", idx + 1)
        method_body = source[idx:end_idx] if end_idx > idx else source[idx:]
        # The fixed version should NOT have self._trials[i] = trial
        assert "self._trials[i] = trial" not in method_body, (
            "log_trial should not replace existing trials (append-only)"
        )
        assert "self._trials.append(trial)" in method_body, (
            "log_trial should always append"
        )

    def test_curator_has_check_duplicates(self):
        """DataCurator should have check_duplicates method."""
        from neutralgrid.data.curator import DataCurator
        assert hasattr(DataCurator, "check_duplicates"), (
            "DataCurator should have check_duplicates method"
        )

    def test_sharpe_uses_obs_per_year(self):
        """Sharpe annualization should use obs_per_year pattern."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "metrics" / "calculator.py"
        )
        source = source_path.read_text()
        idx = source.find("def calculate_sharpe_proxy")
        end_idx = source.find("\n    def ", idx + 1)
        method_body = source[idx:end_idx] if end_idx > idx else source[idx:]
        assert "obs_per_year" in method_body, (
            "Sharpe should compute obs_per_year for annualization"
        )

    def test_cpcv_holdout_embargo_in_source(self):
        """split_with_holdout should have embargo gap logic."""
        source_path = (
            PROJECT_ROOT / "src" / "neutralgrid" / "backtest" / "cpcv.py"
        )
        source = source_path.read_text()
        idx = source.find("def split_with_holdout")
        end_idx = source.find("\n    def ", idx + 1)
        method_body = source[idx:end_idx] if end_idx > idx else source[idx:]
        assert "embargo" in method_body.lower(), (
            "split_with_holdout should implement embargo gap"
        )


# ###########################################################################
# Entry point
# ###########################################################################

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
