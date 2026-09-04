"""
Unit tests for the unified runner API and label contract.

Tests cover:
- Label contract validation (required fields, types, rejection)
- Engine settings extraction and serialization
- Unified runner output (annotated, validated, self-describing)
- Training config builder (defaults, overrides)
- Integration with candidate pipeline (convert_to_training_row)
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.btk_label_contract import (
    ENGINE_SETTINGS_FIELDS,
    LABEL_CONTRACT_VERSION,
    REQUIRED_LABEL_FIELDS,
    TRAINING_ENGINE_DEFAULTS,
    extract_engine_settings,
    validate_engine_result,
)
from backtest.btk_unified_runner import (
    ENGINE_VERSION,
    build_training_config,
    run_backtest,
)
from backtest.btk_seed_state import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    CANDIDATE_TIME_GEOMETRY_SOURCE,
)
from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.backtest.candidate_pipeline import convert_to_training_row


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_klines(
    closes: list[float],
    start: datetime | None = None,
) -> pd.DataFrame:
    if start is None:
        start = datetime(2026, 1, 1)
    n = len(closes)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1.0] * n,
    })


def _oscillating_klines(n: int = 200, base: float = 105.0, amp: float = 5.0) -> pd.DataFrame:
    """Build klines that oscillate across grid levels [100..110]."""
    import numpy as np
    start = datetime(2026, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    t = np.linspace(0, 6 * np.pi, n)
    closes = base + amp * np.sin(t)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": closes + 0.1,
        "low": closes - 0.1,
        "close": closes,
        "volume": [1.0] * n,
    })


def _base_config(**overrides) -> GridConfig:
    """Config with grid levels at [100, 102, 104, 106, 108, 110]."""
    defaults = dict(
        symbol="TESTUSDT",
        lower=100.0,
        upper=110.0,
        num_grids=5,
        capital=1000.0,
        leverage=10,
        maker_fee=0.0002,
        taker_fee=0.0005,
        order_delay_bars=0,
        funding_rate=0.0001,
        slippage_bps=0.0,
        max_holding_bars=0,
        funding_mode="continuous",
        close_fee_mode="taker",
    )
    defaults.update(overrides)
    return GridConfig(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Label Contract Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelContractValidation:
    """Test validate_engine_result() enforcement."""

    def test_valid_result_passes(self):
        """A complete result dict should pass validation."""
        result = {
            "label_positive_by_horizon": True,
            "final_equity": 410.5,
            "capital_used": 400.0,
            "max_drawdown_pct": 2.3,
            "mae": 9.2,
            "mfe": 12.1,
            "mae_pct_initial": 2.3,
            "mfe_pct_initial": 3.025,
            "net_pnl_pct": 2.625,
            "mode": "geometric",
            "grid_count_semantics": "binance_displayed_intervals",
            "funding_mode": "continuous",
            "funding_fees": 0.05,
            "fees_paid": 1.2,
            "liquidated": False,
            "exit_penalty_pct": 0.0,
            "cb_triggered": False,
            # alignment-v1: realized PnL decomposition
            "realized_net_pnl": 10.5,
            "realized_net_pnl_pct": 2.625,
            "unrealized_fraction": 0.0,
            "horizon_censored": False,
        }
        validate_engine_result(result)  # should not raise

    def test_missing_fields_rejected(self):
        """Missing required fields should raise ValueError."""
        result = {
            "label_positive_by_horizon": True,
            "final_equity": 410.5,
            "liquidated": False,
            # missing: max_drawdown_pct, net_pnl_pct, funding_mode, etc.
        }
        with pytest.raises(ValueError, match="missing required label fields"):
            validate_engine_result(result)

    def test_missing_single_field_rejected(self):
        """Even a single missing field should trigger rejection."""
        result = {
            "label_positive_by_horizon": True,
            "final_equity": 410.5,
            "capital_used": 400.0,
            "max_drawdown_pct": 2.3,
            "mae": 9.2,
            "mfe": 12.1,
            "mae_pct_initial": 2.3,
            "mfe_pct_initial": 3.025,
            "net_pnl_pct": 2.625,
            "mode": "geometric",
            "grid_count_semantics": "binance_displayed_intervals",
            "funding_mode": "continuous",
            "funding_fees": 0.05,
            "liquidated": False,
            "exit_penalty_pct": 0.0,
            "cb_triggered": False,
            "realized_net_pnl": 10.5,
            "realized_net_pnl_pct": 2.625,
            "unrealized_fraction": 0.0,
            "horizon_censored": False,
            # missing: fees_paid
        }
        with pytest.raises(ValueError, match="fees_paid"):
            validate_engine_result(result)

    def test_missing_capital_used_rejected(self):
        """Missing capital_used should trigger rejection."""
        result = {
            "label_positive_by_horizon": True,
            "final_equity": 410.5,
            "max_drawdown_pct": 2.3,
            "mae": 9.2,
            "mfe": 12.1,
            "mae_pct_initial": 2.3,
            "mfe_pct_initial": 3.025,
            "net_pnl_pct": 2.625,
            "mode": "geometric",
            "grid_count_semantics": "binance_displayed_intervals",
            "funding_mode": "continuous",
            "funding_fees": 0.05,
            "fees_paid": 1.2,
            "liquidated": False,
            "exit_penalty_pct": 0.0,
            "cb_triggered": False,
            "realized_net_pnl": 10.5,
            "realized_net_pnl_pct": 2.625,
            "unrealized_fraction": 0.0,
            "horizon_censored": False,
            # missing: capital_used
        }
        with pytest.raises(ValueError, match="capital_used"):
            validate_engine_result(result)

    def test_wrong_type_label_rejected(self):
        """label_positive_by_horizon must be bool, not int."""
        result = {
            "label_positive_by_horizon": 1,  # int, not bool
            "final_equity": 410.5,
            "capital_used": 400.0,
            "max_drawdown_pct": 2.3,
            "mae": 9.2,
            "mfe": 12.1,
            "mae_pct_initial": 2.3,
            "mfe_pct_initial": 3.025,
            "net_pnl_pct": 2.625,
            "mode": "geometric",
            "grid_count_semantics": "binance_displayed_intervals",
            "funding_mode": "continuous",
            "funding_fees": 0.05,
            "fees_paid": 1.2,
            "liquidated": False,
            "exit_penalty_pct": 0.0,
            "cb_triggered": False,
            "realized_net_pnl": 10.5,
            "realized_net_pnl_pct": 2.625,
            "unrealized_fraction": 0.0,
            "horizon_censored": False,
        }
        with pytest.raises(TypeError, match="label_positive_by_horizon must be bool"):
            validate_engine_result(result)

    def test_wrong_type_equity_rejected(self):
        """final_equity must be numeric."""
        result = {
            "label_positive_by_horizon": True,
            "final_equity": "410.5",  # string
            "capital_used": 400.0,
            "max_drawdown_pct": 2.3,
            "mae": 9.2,
            "mfe": 12.1,
            "mae_pct_initial": 2.3,
            "mfe_pct_initial": 3.025,
            "net_pnl_pct": 2.625,
            "mode": "geometric",
            "grid_count_semantics": "binance_displayed_intervals",
            "funding_mode": "continuous",
            "funding_fees": 0.05,
            "fees_paid": 1.2,
            "liquidated": False,
            "exit_penalty_pct": 0.0,
            "cb_triggered": False,
            "realized_net_pnl": 10.5,
            "realized_net_pnl_pct": 2.625,
            "unrealized_fraction": 0.0,
            "horizon_censored": False,
        }
        with pytest.raises(TypeError, match="final_equity must be numeric"):
            validate_engine_result(result)

    def test_required_fields_are_frozen(self):
        """REQUIRED_LABEL_FIELDS should be a frozenset (immutable)."""
        assert isinstance(REQUIRED_LABEL_FIELDS, frozenset)

    def test_contract_version_is_string(self):
        assert isinstance(LABEL_CONTRACT_VERSION, str)
        assert len(LABEL_CONTRACT_VERSION) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Engine Settings Extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineSettingsExtraction:
    """Test extract_engine_settings() from GridConfig."""

    def test_extracts_all_settings(self):
        """All engine settings should be extracted from a config."""
        cfg = _base_config()
        settings = extract_engine_settings(cfg)
        expected_keys = {
            "mode", "grid_count_semantics", "funding_mode", "close_fee_mode", "slippage_bps",
            "order_delay_bars", "maker_fee", "taker_fee",
            "max_holding_bars", "funding_rate", "funding_interval_bars",
            "maintenance_margin_rate", "tick_size", "step_size",
            "price_source", "valuation_price_source",
            "funding_rate_series_len", "margin_mode",
            "spread_bps", "fill_mode", "global_cooldown_bars",
            "capital_fraction", "volatility_target_pct", "volatility_proxy_pct",
            "volatility_scale_min", "volatility_scale_max",
            "cb_enabled", "cb_max_dd_pct", "cb_trailing_activate_pct",
            "cb_trailing_offset_pct",
            "cb_inventory_imbalance_ratio", "cb_inventory_imbalance_dd_pct",
        }
        assert expected_keys == set(settings.keys())

    def test_values_match_config(self):
        """Extracted values should exactly match the config."""
        cfg = _base_config(
            slippage_bps=1.5,
            order_delay_bars=3,
            close_fee_mode="maker",
        )
        settings = extract_engine_settings(cfg)
        assert settings["slippage_bps"] == 1.5
        assert settings["order_delay_bars"] == 3
        assert settings["close_fee_mode"] == "maker"
        assert settings["funding_mode"] == "continuous"
        assert settings["mode"] == "geometric"
        assert settings["grid_count_semantics"] == "binance_displayed_intervals"

    def test_engine_settings_fields_list(self):
        """ENGINE_SETTINGS_FIELDS should list all metadata fields."""
        assert "engine_version" in ENGINE_SETTINGS_FIELDS
        assert "label_contract_version" in ENGINE_SETTINGS_FIELDS
        assert "formula_version" in ENGINE_SETTINGS_FIELDS
        assert "mode" in ENGINE_SETTINGS_FIELDS
        assert "grid_count_semantics" in ENGINE_SETTINGS_FIELDS
        assert "funding_mode" in ENGINE_SETTINGS_FIELDS


# ══════════════════════════════════════════════════════════════════════════════
# Training Config Builder
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTrainingConfig:
    """Test build_training_config() defaults and overrides."""

    def test_training_defaults_applied(self):
        """Config built with no overrides should use training defaults."""
        cfg = build_training_config("BTCUSDT", 40000, 42000, 20)
        assert cfg.funding_mode == "continuous"
        assert cfg.close_fee_mode == "maker"
        assert cfg.order_delay_bars == 2
        assert cfg.max_holding_bars == 360
        assert cfg.maker_fee == 0.0002
        assert cfg.taker_fee == 0.0005
        assert cfg.mode == "geometric"
        assert cfg.grid_count_semantics == "binance_displayed_intervals"
        assert cfg.fill_mode == "wick"
        assert cfg.global_cooldown_bars == 0
        assert cfg.cb_enabled is False
        assert cfg.valuation_price_source == "last"

    def test_override_funding_mode(self):
        """Explicit override should take precedence over training default."""
        cfg = build_training_config("BTCUSDT", 40000, 42000, 20,
                                    funding_mode="snapshot")
        assert cfg.funding_mode == "snapshot"

    def test_override_capital_leverage(self):
        """Capital and leverage are direct overrides."""
        cfg = build_training_config("BTCUSDT", 40000, 42000, 20,
                                    capital=800.0, leverage=20)
        assert cfg.capital == 800.0
        assert cfg.leverage == 20

    def test_none_override_ignored(self):
        """None values should be ignored (training default used)."""
        cfg = build_training_config("BTCUSDT", 40000, 42000, 20,
                                    funding_mode=None)
        assert cfg.funding_mode == "continuous"  # training default, not None

    def test_grid_geometry_set(self):
        """Required grid geometry should be set correctly."""
        cfg = build_training_config("ETHUSDT", 2000, 2200, 10)
        assert cfg.symbol == "ETHUSDT"
        assert cfg.lower == 2000
        assert cfg.upper == 2200
        assert cfg.num_grids == 10

    def test_geometric_grid_levels_are_ratio_spaced(self):
        cfg = build_training_config("TESTUSDT", 100.0, 121.0, 3)
        backtester = RealisticGridBacktester(cfg)
        expected_ratio = (121.0 / 100.0) ** (1.0 / 3)
        assert backtester.grid_levels == pytest.approx(
            [100.0, 100.0 * expected_ratio, 100.0 * expected_ratio**2, 121.0]
        )
        assert backtester.grid_spacing_pct == pytest.approx((expected_ratio - 1.0) * 100.0)

    def test_arithmetic_mode_requires_explicit_override(self):
        cfg = build_training_config("TESTUSDT", 100.0, 121.0, 3, mode="arithmetic")
        backtester = RealisticGridBacktester(cfg)
        assert backtester.grid_levels == pytest.approx([100.0, 107.0, 114.0, 121.0])
        assert getattr(cfg, "_physics_overridden") is True

    def test_funding_rate_override(self):
        """Custom funding rate from scanner should override default."""
        cfg = build_training_config("BTCUSDT", 40000, 42000, 20,
                                    funding_rate=0.0003)
        assert cfg.funding_rate == 0.0003


# ══════════════════════════════════════════════════════════════════════════════
# Unified Runner Output
# ══════════════════════════════════════════════════════════════════════════════

class TestRunBacktest:
    """Test run_backtest() produces validated, annotated output."""

    @pytest.fixture(autouse=True)
    def _run(self):
        cfg = _base_config(funding_mode="continuous")
        # Price oscillates across grid levels to generate trades
        self.klines = _oscillating_klines(200, base=105.0, amp=5.0)
        self.result = run_backtest(cfg, self.klines)

    def test_engine_version_present(self):
        assert self.result["engine_version"] == ENGINE_VERSION

    def test_contract_version_present(self):
        assert self.result["label_contract_version"] == LABEL_CONTRACT_VERSION

    def test_all_engine_settings_serialized(self):
        """Every field from ENGINE_SETTINGS_FIELDS should be in the result."""
        for field in ENGINE_SETTINGS_FIELDS:
            assert field in self.result, f"Missing engine setting: {field}"

    def test_required_label_fields_present(self):
        """All required label fields should be present."""
        for field in REQUIRED_LABEL_FIELDS:
            assert field in self.result, f"Missing required label field: {field}"

    def test_label_is_bool(self):
        import numpy as np
        assert isinstance(self.result["label_positive_by_horizon"], (bool, np.bool_))

    def test_final_equity_is_numeric(self):
        assert isinstance(self.result["final_equity"], (int, float))

    def test_settings_match_config(self):
        """Serialized settings should match the config used."""
        assert self.result["funding_mode"] == "continuous"
        assert self.result["close_fee_mode"] == "taker"
        assert self.result["order_delay_bars"] == 0
        assert self.result["maker_fee"] == 0.0002
        assert self.result["taker_fee"] == 0.0005

    def test_standard_output_fields_preserved(self):
        """Runner should not strip any original engine output fields."""
        assert "symbol" in self.result
        assert "net_pnl" in self.result
        assert "round_trips" in self.result
        assert "sharpe_ratio" in self.result
        assert "equity_curve" not in self.result  # internal, not in output

    def test_net_pnl_equals_final_equity_minus_capital(self):
        assert self.result["net_pnl"] == pytest.approx(
            self.result["final_equity"] - 1000.0  # capital
        )


class TestRunBacktestContractRejection:
    """Verify the runner would reject bad engine output (defensive)."""

    def test_runner_validates_output(self):
        """run_backtest on valid input should not raise."""
        cfg = _base_config()
        klines = _make_klines([99.0, 100.5, 102.5])
        # This should succeed without error
        result = run_backtest(cfg, klines)
        assert "engine_version" in result


class TestRunBacktestWithSeed:
    """Test run_backtest with seed_state parameter."""

    def test_seed_state_none_accepted(self):
        """seed_state=None should work (no seeding)."""
        cfg = _base_config()
        klines = _make_klines([99.0, 100.5, 102.5])
        result = run_backtest(cfg, klines, seed_state=None)
        assert "engine_version" in result

    def test_candidate_time_geometric_profile_seeds_from_first_price(self):
        """Explicit profile builds a partial geometry seed without quantity override."""
        cfg = build_training_config(
            "TESTUSDT",
            100.0,
            110.0,
            6,
            capital=1000.0,
            leverage=10,
            max_holding_bars=0,
        )
        klines = _make_klines([105.0, 106.0, 104.0])
        result = run_backtest(
            cfg,
            klines,
            realism_profile=CANDIDATE_TIME_GEOMETRIC_PROFILE,
        )
        assert result["realism_profile"] == CANDIDATE_TIME_GEOMETRIC_PROFILE
        assert result["seed_state_source"] == CANDIDATE_TIME_GEOMETRY_SOURCE
        assert result["seed_source"] == CANDIDATE_TIME_GEOMETRY_SOURCE
        assert result["position_size_source"] == "model"

    def test_public_market_profile_uses_candidate_time_geometry_seed(self):
        cfg = build_training_config(
            "TESTUSDT",
            100.0,
            110.0,
            6,
            capital=1000.0,
            leverage=10,
            max_holding_bars=0,
        )
        klines = _make_klines([105.0, 106.0, 104.0])
        result = run_backtest(
            cfg,
            klines,
            realism_profile=CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
        )
        assert result["realism_profile"] == CANDIDATE_TIME_PUBLIC_MARKET_PROFILE
        assert result["seed_state_source"] == CANDIDATE_TIME_GEOMETRY_SOURCE
        assert result["seed_source"] == CANDIDATE_TIME_GEOMETRY_SOURCE

    def test_mark_valuation_source_affects_unrealized_pnl_only(self):
        klines = _make_klines([105.0, 95.0])
        klines["mark_close"] = [105.0, 110.0]
        cfg_last = _base_config(
            lower=90.0,
            upper=110.0,
            num_grids=3,
            mode="arithmetic",
            leverage=1,
            maker_fee=0.0,
            taker_fee=0.0,
            funding_rate=0.0,
            max_holding_bars=999,
            valuation_price_source="last",
        )
        cfg_mark = _base_config(
            lower=90.0,
            upper=110.0,
            num_grids=3,
            mode="arithmetic",
            leverage=1,
            maker_fee=0.0,
            taker_fee=0.0,
            funding_rate=0.0,
            max_holding_bars=999,
            valuation_price_source="mark",
        )

        last_result = run_backtest(cfg_last, klines)
        mark_result = run_backtest(cfg_mark, klines)

        assert last_result["fill_price_source"] == "last"
        assert mark_result["fill_price_source"] == "last"
        assert last_result["valuation_price_end"] == pytest.approx(95.0)
        assert mark_result["valuation_price_end"] == pytest.approx(110.0)
        assert mark_result["final_equity"] > last_result["final_equity"]

    def test_unknown_realism_profile_fails_closed(self):
        cfg = _base_config()
        klines = _make_klines([99.0, 100.5, 102.5])
        with pytest.raises(ValueError):
            run_backtest(cfg, klines, realism_profile="unverified_profile")


# ══════════════════════════════════════════════════════════════════════════════
# Integration: Training Config → Runner → Training Row
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingPipelineIntegration:
    """End-to-end: build_training_config → run_backtest → convert_to_training_row."""

    def test_full_pipeline_flow(self):
        """Training config → runner → training row should produce valid output."""
        cfg = build_training_config("TESTUSDT", 100.0, 110.0, 5,
                                    capital=1000.0, leverage=10,
                                    max_holding_bars=0)
        klines = _oscillating_klines(200, base=105.0, amp=5.0)
        result = run_backtest(cfg, klines)

        candidate_row = {
            "symbol": "TESTUSDT",
            "candidate_id": "TESTUSDT_20260101_000000",
            "scan_file": "test.csv",
            "score": 85.0,
            "grid_lower": 100.0,
            "grid_upper": 110.0,
            "num_grids": 5,
        }
        training_row = convert_to_training_row(result, candidate_row)

        # y now comes from hlabel_meta (hierarchical deploy decision).
        # y_horizon preserves the engine's horizon label.
        horizon_y = 1 if result["label_positive_by_horizon"] else 0
        assert training_row.get("y_horizon") == horizon_y
        # hlabel_meta may differ from horizon (it checks L1/L2/L3)
        assert training_row["y"] in (0, 1)
        assert training_row["source"] == "backtest"
        assert training_row["sample_weight_override"] == 0.5
        assert training_row["pnl_pct"] == result["net_pnl_pct"]

    def test_training_config_uses_continuous_funding(self):
        """Training config should default to continuous funding."""
        cfg = build_training_config("TESTUSDT", 100.0, 110.0, 5,
                                    capital=1000.0, leverage=10,
                                    max_holding_bars=0)
        klines = _oscillating_klines(200, base=105.0, amp=5.0)
        result = run_backtest(cfg, klines)
        assert result["funding_mode"] == "continuous"

    def test_engine_settings_in_result(self):
        """Result from training pipeline should include engine settings."""
        cfg = build_training_config("TESTUSDT", 100.0, 110.0, 5,
                                    capital=1000.0, leverage=10)
        klines = _oscillating_klines(100, base=105.0, amp=5.0)
        result = run_backtest(cfg, klines)
        assert result["engine_version"] == ENGINE_VERSION
        assert result["label_contract_version"] == LABEL_CONTRACT_VERSION
        assert result["funding_mode"] == "continuous"


# ══════════════════════════════════════════════════════════════════════════════
# Training Engine Defaults
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingEngineDefaults:
    """Verify TRAINING_ENGINE_DEFAULTS values are sane."""

    def test_mode_is_geometric(self):
        assert TRAINING_ENGINE_DEFAULTS["mode"] == "geometric"

    def test_funding_mode_is_continuous(self):
        assert TRAINING_ENGINE_DEFAULTS["funding_mode"] == "continuous"

    def test_close_fee_mode_is_maker(self):
        assert TRAINING_ENGINE_DEFAULTS["close_fee_mode"] == "maker"

    def test_order_delay_bars_is_two(self):
        assert TRAINING_ENGINE_DEFAULTS["order_delay_bars"] == 2

    def test_fill_mode_is_wick(self):
        assert TRAINING_ENGINE_DEFAULTS["fill_mode"] == "wick"

    def test_global_cooldown_disabled(self):
        assert TRAINING_ENGINE_DEFAULTS["global_cooldown_bars"] == 0

    def test_circuit_breaker_disabled(self):
        assert TRAINING_ENGINE_DEFAULTS["cb_enabled"] is False

    def test_valuation_source_defaults_to_last(self):
        assert TRAINING_ENGINE_DEFAULTS["valuation_price_source"] == "last"

    def test_max_holding_bars_is_360(self):
        """360 bars = 6 hours at 1-min resolution."""
        assert TRAINING_ENGINE_DEFAULTS["max_holding_bars"] == 360

    def test_all_defaults_are_valid_gridconfig_fields(self):
        """Every key in TRAINING_ENGINE_DEFAULTS should be a valid GridConfig field."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(GridConfig)}
        for key in TRAINING_ENGINE_DEFAULTS:
            assert key in valid_fields, (
                f"TRAINING_ENGINE_DEFAULTS key '{key}' is not a GridConfig field"
            )
