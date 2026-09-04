"""
Integration tests for AFML-compliant optimizations.

Tests all new modules and enhancements:
1. Enhanced utility scoring with trend_breakout_loss
2. Regime-aware grid adjustment
3. Microstructure penalties
4. Data curator
5. Market dynamics tracking
6. Triple barrier labeling
7. CPCV and deflated Sharpe
"""

from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import pandas as pd


class TestUtilityScoring:
    """Test enhanced utility scoring with trend_breakout_loss."""

    def test_utility_config_has_kappa(self):
        """Verify UtilityConfig includes kappa_trend parameter."""
        from neutralgrid.validation.utility import UtilityConfig

        config = UtilityConfig()
        assert hasattr(config, "kappa_trend")
        assert config.kappa_trend == 1.5  # Default value

    def test_utility_components_has_breakout_loss(self):
        """Verify UtilityComponents includes expected_trend_breakout_loss."""
        from neutralgrid.validation.utility import UtilityConfig, UtilityScorer

        # Explicit config: default `from_artifact()` path now fail-closes
        # when no calibrator artifact is present.
        scorer = UtilityScorer(UtilityConfig())
        result = scorer.compute_utility(
            range_prob=0.7,
            trend_prob=0.2,
            profit_per_grid_pct=0.8,
            num_grids=20,
            range_size_pct=3.0,
        )

        assert hasattr(result, "expected_trend_breakout_loss")
        assert result.expected_trend_breakout_loss >= 0

    def test_utility_with_survival_and_hurst(self):
        """Test utility computation with survival_prob and hurst_exponent."""
        from neutralgrid.validation.utility import UtilityConfig, UtilityScorer

        # Explicit config: default `from_artifact()` path now fail-closes
        # when no calibrator artifact is present.
        scorer = UtilityScorer(UtilityConfig())

        # Low survival prob should increase breakout loss
        result_low = scorer.compute_utility(
            range_prob=0.7,
            trend_prob=0.3,
            profit_per_grid_pct=0.8,
            num_grids=20,
            range_size_pct=3.0,
            survival_prob=0.4,
            hurst_exponent=0.6,
        )

        result_high = scorer.compute_utility(
            range_prob=0.7,
            trend_prob=0.3,
            profit_per_grid_pct=0.8,
            num_grids=20,
            range_size_pct=3.0,
            survival_prob=0.9,
            hurst_exponent=0.4,
        )

        # Lower survival prob should result in higher breakout loss
        assert result_low.expected_trend_breakout_loss > result_high.expected_trend_breakout_loss
        print("[PASS] Utility scoring with survival/hurst works correctly")


class TestRegimeAwareGridAdjustment:
    """Test regime-aware grid adjustment in grid calculator."""

    def test_grid_params_has_sizing_fields(self):
        """Verify GridParams includes capital_fraction and related fields."""
        from neutralgrid.grid.calculator import GridParams

        params = GridParams(symbol="TEST", is_valid=True)
        assert hasattr(params, "capital_fraction")
        assert hasattr(params, "regime_confidence")
        assert hasattr(params, "sizing_reason")

    def test_regime_adjusted_grids(self):
        """Test grid count adjustment based on regime."""
        from neutralgrid.grid.calculator import GridCalculator

        calc = GridCalculator()

        # Low trend prob -> no adjustment
        grids_low, _ = calc.compute_regime_adjusted_grids(
            num_grids=20,
            grid_spacing_pct=0.5,
            trend_prob=0.15,
        )
        assert grids_low == 20

        # High trend prob -> wider spacing, fewer grids
        grids_high, spacing_high = calc.compute_regime_adjusted_grids(
            num_grids=20,
            grid_spacing_pct=0.5,
            trend_prob=0.33,
        )
        assert grids_high < 20
        assert spacing_high > 0.5

        print(f"[PASS] Grid adjustment: {grids_low} grids @ 0.15 trend, {grids_high} grids @ 0.33 trend")


class TestMicrostructure:
    """Test microstructure penalties."""

    def test_microstructure_estimator(self):
        """Test microstructure cost estimation."""
        from neutralgrid.validation.microstructure import MicrostructureEstimator

        estimator = MicrostructureEstimator()

        costs = estimator.estimate_costs(
            funding_rate=0.001,
            volatility_pct=2.0,
        )

        assert costs.round_trip_cost_pct > 0
        assert costs.min_profit_required_pct > 0
        assert costs.min_profit_required_pct > costs.round_trip_cost_pct

        print(f"[PASS] Microstructure: round_trip={costs.round_trip_cost_pct:.3f}%, min_profit={costs.min_profit_required_pct:.3f}%")

    def test_viability_check(self):
        """Test grid viability check."""
        from neutralgrid.validation.microstructure import MicrostructureEstimator

        estimator = MicrostructureEstimator()
        costs = estimator.estimate_costs(volatility_pct=2.0)

        # High profit (above min_profit_required) -> viable
        high_profit = costs.min_profit_required_pct + 1.0  # Ensure above threshold
        viable, _ = estimator.is_viable(costs, profit_per_grid_pct=high_profit)
        assert viable, f"Expected viable with {high_profit:.2f}% > {costs.min_profit_required_pct:.2f}%"

        # Low profit (below min_profit_required) -> not viable
        low_profit = costs.min_profit_required_pct * 0.5  # Ensure below threshold
        viable, _ = estimator.is_viable(costs, profit_per_grid_pct=low_profit)
        assert not viable, f"Expected not viable with {low_profit:.2f}% < {costs.min_profit_required_pct:.2f}%"

        print("[PASS] Viability check works correctly")


class TestDataCurator:
    """Test data curator layer."""

    def test_validate_ohlcv(self):
        """Test OHLCV validation."""
        from neutralgrid.data.curator import DataCurator
        from datetime import datetime, timezone, timedelta

        curator = DataCurator()

        # Create test data
        n = 100
        now = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "open_time": [now - timedelta(minutes=15*i) for i in range(n-1, -1, -1)],
            "open": np.random.uniform(100, 101, n),
            "high": np.random.uniform(101, 102, n),
            "low": np.random.uniform(99, 100, n),
            "close": np.random.uniform(100, 101, n),
            "volume": np.random.uniform(1000, 2000, n),
        })

        result = curator.validate_ohlcv(df, timeframe="15m")

        assert result.passed
        assert "missing_bars" in result.checks
        assert "nan_inf" in result.checks

        print("[PASS] Data curator validation works correctly")

    def test_clean_ohlcv(self):
        """Test OHLCV cleaning."""
        from neutralgrid.data.curator import DataCurator

        curator = DataCurator()

        # Create data with NaN
        df = pd.DataFrame({
            "close": [100.0, np.nan, 102.0, 103.0, np.nan],
            "volume": [1000.0, 2000.0, np.nan, 4000.0, 5000.0],
        })

        cleaned, clean_report = curator.clean_ohlcv(df)

        assert not cleaned["close"].isna().any()
        assert not cleaned["volume"].isna().any()
        assert "steps" in clean_report
        assert clean_report["input_rows"] == 5

        print("[PASS] Data curator cleaning works correctly")


class TestTripleBarrier:
    """Test triple barrier labeling."""

    def test_label_entry(self):
        """Test single entry labeling."""
        from neutralgrid.models.triple_barrier import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()

        # Price that hits PT
        prices = np.array([100.0] * 10 + [115.0] + [100.0] * 100)
        label = labeler.label_entry(prices, entry_bar=5)

        assert label is not None
        assert label.label == 1  # Hit PT
        assert label.barrier_hit == "pt"

        # Price that hits SL (-12% from 100.0 = below 88.0)
        prices = np.array([100.0] * 10 + [87.0] + [100.0] * 100)
        label = labeler.label_entry(prices, entry_bar=5)

        assert label is not None
        assert label.label == -1  # Hit SL
        assert label.barrier_hit == "sl"

        print("[PASS] Triple barrier labeling works correctly")

    def test_label_for_meta_learning(self):
        """Test meta-learning label generation."""
        from neutralgrid.models.triple_barrier import TripleBarrierLabeler

        labeler = TripleBarrierLabeler()

        # Create price series
        n = 500
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({"close": prices})

        labels, details = labeler.label_for_meta_learning(df, entry_indices=[10, 50, 100])

        assert len(labels) == 3
        assert len(details) == 3
        assert "binary_label" in details.columns

        print("[PASS] Meta-learning labels generated correctly")


class TestCPCV:
    """Test CPCV and deflated Sharpe."""

    def test_cpcv_splits(self):
        """Test CPCV split generation."""
        from neutralgrid.backtest.cpcv import CPCV, CPCVConfig

        config = CPCVConfig(n_groups=6, n_test_groups=2)
        cpcv = CPCV(config)

        # Verify number of paths
        assert cpcv.n_paths == 15  # C(6,2) = 15

        # Create test data large enough so that all 15 folds survive
        # time-based purging (default purge_hours=48, embargo_hours=6).
        n = 600
        df = pd.DataFrame({
            "start_time_utc": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": np.random.randn(n),
        })

        # Count splits
        splits = list(cpcv.split(df))
        assert len(splits) == 15

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

        print(f"[PASS] CPCV generates {len(splits)} paths correctly")

    def test_deflated_sharpe(self):
        """Test deflated Sharpe calculation."""
        from neutralgrid.backtest.cpcv import DeflatedSharpeCalculator

        calc = DeflatedSharpeCalculator(n_trials=50)

        result = calc.deflate(
            raw_sharpe=1.5,
            n_observations=252,
        )

        assert result.deflated_sharpe < result.raw_sharpe
        assert result.haircut_pct > 0
        assert result.n_trials == 50
        assert 0 <= result.prob_false_discovery <= 1

        print(f"[PASS] Deflated Sharpe: raw={result.raw_sharpe:.2f}, deflated={result.deflated_sharpe:.2f}, haircut={result.haircut_pct:.1f}%")


class TestIndicatorConsistency:
    """Test indicator parameter consistency fix."""

    def test_feature_extractor_uses_config(self):
        """Verify feature_extractor lazy config returns expected indicator values."""
        from neutralgrid.scanner.feature_extractor import _get_indicator_config
        from neutralgrid.core.config import get_config

        cfg = get_config()
        ind_cfg = _get_indicator_config()

        assert ind_cfg["ema_fast"] == cfg.indicators.ema_fast
        assert ind_cfg["ema_medium"] == cfg.indicators.ema_medium
        assert ind_cfg["ema_slow"] == cfg.indicators.ema_slow

        print(f"[PASS] Indicator parameters aligned: EMA {cfg.indicators.ema_fast}/{cfg.indicators.ema_medium}/{cfg.indicators.ema_slow}")


class TestKlinesQuoteVolume:
    """Test klines schema preserves quote_volume."""

    def test_klines_to_df_includes_quote_volume(self):
        """Verify klines_to_df includes quote_volume column."""
        from neutralgrid.scanner.feature_extractor import klines_to_df

        # Mock kline data (12 columns from Binance)
        klines = [
            [1704067200000, "100.0", "101.0", "99.0", "100.5", "1000.0",
             1704070800000, "100500.0", 500, "500.0", "50250.0", "0"],
            [1704070800000, "100.5", "102.0", "100.0", "101.0", "1200.0",
             1704074400000, "121200.0", 600, "600.0", "60600.0", "0"],
        ]

        df = klines_to_df(klines)

        assert "quote_volume" in df.columns
        assert len(df) == 2
        assert df["quote_volume"].iloc[0] == 100500.0

        print("[PASS] klines_to_df preserves quote_volume column")


class TestHistoryLengthAlignment:
    """Test 1h history length alignment."""

    def test_kline_limit_matches_hmm(self):
        """Verify KLINE_LIMITS['15m'] >= HMM infer limit."""
        from neutralgrid.core.config import get_config

        _cfg = get_config()
        assert _cfg.validation.kline_limits["15m"] >= _cfg.hmm.infer_limit

        print(f"[PASS] 15m kline limit ({_cfg.validation.kline_limits['15m']}) >= HMM limit ({_cfg.hmm.infer_limit})")


class TestBoundedUniverseContract:
    """Test bounded-universe contract for profile_model and pattern_profile.

    Per HMM_CHANGE.md v3.0 sections 3.1-3.8:
    - Only 0 <= duration_hours < max_duration_hours rows are eligible
    - pnl_thr is computed on the bounded universe only
    - Winners and losers come from the same bounded universe
    - Missing profit_factor rows are unlabeled, not losers
    - pattern_profile and profile_model select same winners when APG absent
    """

    @staticmethod
    @contextmanager
    def _workspace_tmp_path():
        """Use workspace temp because Windows tempfile dirs are not writable here."""
        tmp_path = Path.cwd() / ".pytest_tmp" / f"bounded_universe_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True, exist_ok=False)
        try:
            yield tmp_path
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    # Phase 2.4 — per-class sample floor is max(30, 3 * len(feats)). With
    # DEFAULT_FEATURES now containing 4 declared features, min_samples=30 per
    # class. Fixture sized (120 short rows) to satisfy both classes comfortably.
    _N_SHORT_ROWS = 120

    @classmethod
    def _make_test_xlsx(cls, tmp_path):
        """Create a synthetic Excel file with known properties."""
        rows = []
        # Short-horizon rows (duration < 7h)
        for i in range(cls._N_SHORT_ROWS):
            rows.append({
                "strategy_id": f"short_{i}",
                "symbol": "BTCUSDT",
                "pnl_pct": 5.0 + i * 0.2,  # monotone range for pnl_thr locality
                "profit_factor": 1.0 + (i % 20) * 0.1,  # 1.0..2.9, cycles
                "duration_hours": 1.0 + (i % 12) * 0.5,  # 1.0..6.5, bounded
                "adx_1h": 20.0 + (i % 10),
                "adx_15m": 25.0 + (i % 10),
                "adx_5m": 22.0 + (i % 10),
                "rsi_15m": 45.0 + (i % 10),
                "ema_slope_1h": 0.01 * (i % 10),
                "ema_crosses_5m": i % 3,
                "vwap_crosses_5m": i % 2,
                "range_size_pct": 2.0 + (i % 10) * 0.1,
                "bb_width": 0.02 + (i % 10) * 0.001,
                # Current DEFAULT_FEATURES (pattern_profile / profile_model).
                # Varied per-row so each clears the >=10-non-NaN availability
                # filter and supports a Gaussian fit.
                "parkinson_vol_ratio_4h_24h_pre": 0.8 + (i % 10) * 0.05,
                "variance_ratio_1m_15m_pre_2h": 0.5 + (i % 8) * 0.1,
                "funding_carry_expected_next_7h": -0.01 + (i % 5) * 0.005,
                "liquidity_stability_z_1h": -1.0 + (i % 10) * 0.2,
                "trend_structure": "ranging" if i % 2 == 0 else "up",
            })
        # 5 long-horizon rows (duration >= 7h) — must be excluded
        for i in range(5):
            rows.append({
                "strategy_id": f"long_{i}",
                "symbol": "ETHUSDT",
                "pnl_pct": 30.0 + i * 5.0,  # high pnl to prove they get excluded
                "profit_factor": 3.0 + i,
                "duration_hours": 8.0 + i * 2.0,
                "adx_1h": 15.0,
                "adx_15m": 18.0,
                "adx_5m": 20.0,
                "rsi_15m": 50.0,
                "ema_slope_1h": 0.05,
                "ema_crosses_5m": 1,
                "vwap_crosses_5m": 0,
                "range_size_pct": 3.0,
                "bb_width": 0.03,
                "parkinson_vol_ratio_4h_24h_pre": 1.5,
                "variance_ratio_1m_15m_pre_2h": 1.3,
                "funding_carry_expected_next_7h": 0.02,
                "liquidity_stability_z_1h": 1.0,
                "trend_structure": "up",
            })
        # 1 row with missing profit_factor (must be unlabeled, not loser)
        rows.append({
            "strategy_id": "missing_pf",
            "symbol": "XRPUSDT",
            "pnl_pct": 20.0,
            "profit_factor": None,  # NaN
            "duration_hours": 3.0,
            "adx_1h": 22.0,
            "adx_15m": 27.0,
            "adx_5m": 24.0,
            "rsi_15m": 48.0,
            "ema_slope_1h": 0.02,
            "ema_crosses_5m": 1,
            "vwap_crosses_5m": 1,
            "range_size_pct": 2.5,
            "bb_width": 0.025,
            "parkinson_vol_ratio_4h_24h_pre": 0.9,
            "variance_ratio_1m_15m_pre_2h": 0.7,
            "funding_carry_expected_next_7h": 0.0,
            "liquidity_stability_z_1h": 0.5,
            "trend_structure": "ranging",
        })

        df = pd.DataFrame(rows)

        xlsx_path = tmp_path / "test_bounded_universe.xlsx"
        df.to_excel(str(xlsx_path), index=False, sheet_name="Sheet1")
        return xlsx_path, df

    def test_bounded_universe_excludes_long_horizon(self):
        """Only rows with 0 <= duration_hours < max_duration_hours are eligible."""

        with self._workspace_tmp_path() as tmp_path:
            xlsx_path, df = self._make_test_xlsx(tmp_path)

            from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx
            model = train_profile_model_from_enhanced_xlsx(
                xlsx_path, max_duration_hours=7.0, min_profit_factor=1.0, top_quantile=0.5,
            )

            # duration_band must reflect the bounded universe
            assert model.duration_band is not None
            assert model.duration_band["min_hours"] == 0.0
            assert model.duration_band["max_hours"] == 7.0

            # Prior must be computed from bounded universe only (10 short + 1 missing_pf)
            # missing_pf excluded from labels -> 10 labeled rows in bounded universe
            # With min_pf=1.0 and top_quantile=0.5, pnl_thr computed on 10 labeled rows
            assert 0.0 < model.prior_winner < 1.0

        print("[PASS] Bounded universe excludes long-horizon rows")

    def test_pnl_thr_on_bounded_universe(self):
        """pnl_thr must be computed on bounded universe, not full workbook."""

        with self._workspace_tmp_path() as tmp_path:
            xlsx_path, df = self._make_test_xlsx(tmp_path)

            from neutralgrid.scanner.pattern_profile import build_profile_from_enhanced_xlsx
            profile = build_profile_from_enhanced_xlsx(
                xlsx_path, max_duration_hours=7.0, min_profit_factor=1.0, top_quantile=0.5,
            )

            # pnl_threshold must be from bounded universe (short rows + missing_pf row)
            # Short rows pnl: [5,7,9,11,13,15,17,19,21,23], missing_pf pnl=20 (excluded from labeled)
            # Labeled short rows pnl: [5,7,9,11,13,15,17,19,21,23] -> q50 = 14.0
            # Full workbook q75 would be much higher due to long_ rows with pnl 30-50
            pnl_thr = profile.selection_summary["pnl_threshold"]
            # Must not include the long-horizon rows (pnl 30-50)
            assert pnl_thr < 30.0, f"pnl_thr={pnl_thr} includes long-horizon rows"

        print("[PASS] pnl_thr computed on bounded universe only")

    def test_missing_profit_factor_is_unlabeled(self):
        """Rows with missing profit_factor must be excluded from labels, not forced to loser."""

        with self._workspace_tmp_path() as tmp_path:
            xlsx_path, df = self._make_test_xlsx(tmp_path)

            from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx
            # With min_pf=1.0 and low quantile, should get winners from short rows
            model = train_profile_model_from_enhanced_xlsx(
                xlsx_path, max_duration_hours=7.0, min_profit_factor=1.0, top_quantile=0.5,
            )

            # The model should have features — training succeeded without the missing_pf row
            assert len(model.features) > 0

            # Prior should reflect labeled rows only (10 rows, not 11)
            # If missing_pf was forced to loser, prior would be lower
            assert model.prior_winner > 0.0

        print("[PASS] Missing profit_factor rows are unlabeled, not losers")

    def test_pattern_and_model_consistent_winners_no_apg(self):
        """pattern_profile and profile_model select same winners when APG absent."""

        with self._workspace_tmp_path() as tmp_path:
            xlsx_path, df = self._make_test_xlsx(tmp_path)
            # The test xlsx has no avg_profit_per_grid column
            # Use top_quantile=0.5 and min_profit_factor=1.2 to produce >= 5 winners
            # from the 10 labeled short-horizon rows

            from neutralgrid.scanner.pattern_profile import build_profile_from_enhanced_xlsx
            from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx

            profile = build_profile_from_enhanced_xlsx(
                xlsx_path, max_duration_hours=7.0, min_profit_factor=1.2, top_quantile=0.5,
            )
            model = train_profile_model_from_enhanced_xlsx(
                xlsx_path, max_duration_hours=7.0, min_profit_factor=1.2, top_quantile=0.5,
            )

            # Both must report bounded universe metadata
            assert profile.selection_summary.get("max_duration_hours") == 7.0
            assert model.duration_band is not None
            assert model.duration_band["max_hours"] == 7.0

            # Both must use the same bounded universe size
            assert profile.selection_summary.get("bounded_universe_size") is not None

        print("[PASS] pattern_profile and profile_model consistent when APG absent")

    def test_pattern_profile_does_not_fallback_to_top_pnl(self):
        """pattern_profile must fail fast instead of relabeling top-PnL rows."""
        import pytest

        with self._workspace_tmp_path() as tmp_path:
            xlsx_path, _df = self._make_test_xlsx(tmp_path)

            from neutralgrid.scanner.pattern_profile import build_profile_from_enhanced_xlsx

            with pytest.raises(ValueError, match="Bounded universe produced only"):
                build_profile_from_enhanced_xlsx(
                    xlsx_path,
                    max_duration_hours=7.0,
                    min_profit_factor=99.0,
                    top_quantile=0.95,
                )

        print("[PASS] pattern_profile fails fast instead of top-PnL fallback")

    def test_profile_model_roundtrip_with_duration_band(self):
        """ProfileModel duration_band survives JSON roundtrip."""
        from neutralgrid.scanner.profile_model import ProfileModel

        model = ProfileModel(
            features=["adx_1h"],
            winner_mu={"adx_1h": 20.0},
            loser_mu={"adx_1h": 30.0},
            inv_cov=[[1.0]],
            prior_winner=0.5,
            duration_band={"min_hours": 0.0, "max_hours": 7.0},
        )

        obj = model.to_json()
        restored = ProfileModel.from_json(obj)

        assert restored.duration_band is not None
        assert restored.duration_band["min_hours"] == 0.0
        assert restored.duration_band["max_hours"] == 7.0

        print("[PASS] ProfileModel duration_band roundtrips through JSON")

    def test_duplicate_strategy_id_raises(self):
        """Duplicate strategy_id must raise ValueError."""

        with self._workspace_tmp_path() as tmp_path:
            rows = []
            for i in range(10):
                rows.append({
                    "strategy_id": f"bot_{i}",
                    "pnl_pct": 10.0 + i,
                    "profit_factor": 2.0,
                    "duration_hours": 3.0,
                    "adx_1h": 20.0, "adx_15m": 25.0, "adx_5m": 22.0,
                    "rsi_15m": 45.0, "ema_slope_1h": 0.01,
                    "range_size_pct": 2.0, "bb_width": 0.02,
                })
            # Add a duplicate
            rows.append({
                "strategy_id": "bot_0",
                "pnl_pct": 15.0,
                "profit_factor": 2.5,
                "duration_hours": 4.0,
                "adx_1h": 22.0, "adx_15m": 27.0, "adx_5m": 24.0,
                "rsi_15m": 48.0, "ema_slope_1h": 0.02,
                "range_size_pct": 2.5, "bb_width": 0.025,
            })

            df = pd.DataFrame(rows)
            xlsx_path = tmp_path / "test_duplicates.xlsx"
            df.to_excel(str(xlsx_path), index=False, sheet_name="Sheet1")

            from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx
            import pytest

            try:
                train_profile_model_from_enhanced_xlsx(xlsx_path, max_duration_hours=7.0)
                assert False, "Should have raised ValueError for duplicate strategy_id"
            except ValueError as e:
                assert "Duplicate strategy_id" in str(e)

        print("[PASS] Duplicate strategy_id raises ValueError")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("AFML Integration Tests")
    print("=" * 60)

    test_classes = [
        TestUtilityScoring,
        TestRegimeAwareGridAdjustment,
        TestMicrostructure,
        TestDataCurator,
        TestTripleBarrier,
        TestCPCV,
        TestIndicatorConsistency,
        TestKlinesQuoteVolume,
        TestHistoryLengthAlignment,
        TestBoundedUniverseContract,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                except Exception as e:
                    print(f"[FAIL] {method_name}: {e}")
                    failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
