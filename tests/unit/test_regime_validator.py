"""
Unit tests for RegimeValidator.

Tests that validator output is stable for fixed input fixtures.
"""

from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from neutralgrid.validation.regime_validator import (
    RegimeValidator,
    ValidationResult,
    TimeframeResult,
    CheckResult,
)


def create_test_market_data(n_bars_1h: int = 200, n_bars_15m: int = 200) -> dict:
    """Create test market data fixture."""
    def make_klines(n_bars):
        return [
            [
                i * 3600000,  # open_time
                40000 + np.sin(i / 10) * 100,  # open
                40000 + np.sin(i / 10) * 100 + 50,  # high
                40000 + np.sin(i / 10) * 100 - 50,  # low
                40000 + np.sin(i / 10) * 100 + 10,  # close
                100 + np.random.uniform(-10, 10),  # volume
                (i + 1) * 3600000,  # close_time
                4000000,  # quote_volume
                1000,  # trades
                50,  # taker_buy_base
                2000000,  # taker_buy_quote
                0,  # ignore
            ]
            for i in range(n_bars)
        ]

    return {
        "symbol": "TESTUSDT",
        "klines": {
            "1h": make_klines(n_bars_1h),
            "15m": make_klines(n_bars_15m),
            "1m": make_klines(100),
        }
    }


class TestRegimeValidator:
    """Test RegimeValidator behavior."""

    def test_validator_initializes(self):
        """Test that validator initializes without errors."""
        validator = RegimeValidator()
        assert validator is not None

    def test_check_result_structure(self):
        """Test that CheckResult has expected structure."""
        check = CheckResult(
            name="test_check",
            executed=True,
            passed=True,
            metrics={"value": 0.5},
            reason=None,
        )

        assert check.name == "test_check"
        assert check.executed is True
        assert check.passed is True
        assert check.metrics == {"value": 0.5}
        assert check.reason is None

    def test_validation_result_structure(self):
        """Test that ValidationResult has expected structure."""
        result = ValidationResult(
            symbol="BTCUSDT",
            is_valid=True,
        )

        assert result.symbol == "BTCUSDT"
        assert result.is_valid is True
        assert result.tf_1h is None
        assert result.tf_15m is None
        assert result.tf_5m is None

    def test_stable_output_for_fixed_input(self):
        """Test that same input produces same output (deterministic)."""
        validator = RegimeValidator()
        market_data = create_test_market_data()

        # Run validation twice
        result1 = validator.validate(market_data)
        result2 = validator.validate(market_data)

        # Should produce identical results
        assert result1.symbol == result2.symbol
        assert result1.is_valid == result2.is_valid

        # If valid, check that extracted values match
        if result1.is_valid:
            assert result1.range_high == result2.range_high
            assert result1.range_low == result2.range_low
            assert result1.current_price == result2.current_price

    def test_missing_1h_data_fails(self):
        """Test that missing 1H data causes validation to fail."""
        validator = RegimeValidator()
        market_data = {
            "symbol": "TESTUSDT",
            "klines": {
                "15m": [[0] * 12 for _ in range(200)],
            }
        }

        result = validator.validate(market_data)

        assert result.is_valid is False
        assert result.tf_1h is not None
        assert result.tf_1h.is_valid is False
        assert result.tf_1h.reason is not None
        assert "missing" in result.tf_1h.reason.lower()

    def test_missing_15m_data_fails(self):
        """Test that missing 15M data causes validation to fail."""
        validator = RegimeValidator()
        market_data = {
            "symbol": "TESTUSDT",
            "klines": {
                "1h": [[0, 40000, 40100, 39900, 40050, 100, 0, 4000000, 1000, 50, 2000000, 0] for _ in range(200)],
            }
        }

        result = validator.validate(market_data)

        # Should fail because 15m data is missing
        # (assuming HMM check passes - if not, it would fail earlier)
        assert result.is_valid is False

    def test_range_quality_check_extracts_bounds(self):
        """Test that range quality check extracts high/low bounds."""
        validator = RegimeValidator()
        market_data = create_test_market_data()

        # Create a fixture where we know the high/low
        highs = [40100, 40200, 40150]
        lows = [39900, 39800, 39850]

        market_data["klines"]["15m"] = [
            [
                i * 900000,
                40000,
                highs[i % len(highs)],
                lows[i % len(lows)],
                40050,
                100,
                (i + 1) * 900000,
                4000000,
                1000,
                50,
                2000000,
                0,
            ]
            for i in range(200)
        ]

        result = validator.validate(market_data)

        # Range should be extracted
        if result.tf_15m and result.tf_15m.is_valid:
            assert result.range_high is not None
            assert result.range_low is not None
            assert result.range_high > result.range_low

    def test_validator_returns_validation_result(self):
        """Test that validate() returns ValidationResult."""
        validator = RegimeValidator()
        market_data = create_test_market_data()

        result = validator.validate(market_data)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, "symbol")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "tf_1h")
        assert hasattr(result, "tf_15m")

    def test_check_methods_return_check_result(self):
        """Test that check methods return CheckResult."""
        validator = RegimeValidator()

        # Create test DataFrame
        df = pd.DataFrame({
            "open": [40000] * 200,
            "high": [40100] * 200,
            "low": [39900] * 200,
            "close": [40050] * 200,
            "volume": [100] * 200,
        })

        # Test range quality check
        check = validator.check_range_quality(df)
        assert isinstance(check, CheckResult)
        assert check.name == "range_quality"
        assert isinstance(check.executed, bool)
        assert isinstance(check.passed, bool)

    def test_insufficient_data_handled_gracefully(self):
        """Test that insufficient data doesn't crash validator."""
        validator = RegimeValidator()

        # Very short data
        market_data = {
            "symbol": "TESTUSDT",
            "klines": {
                "1h": [[0] * 12 for _ in range(10)],  # Only 10 bars
                "15m": [[0] * 12 for _ in range(10)],
            }
        }

        result = validator.validate(market_data)

        # Should fail gracefully, not crash
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False

    def test_stochastic_failure_preserves_computed_metrics(self):
        """Stochastic diagnostics should survive an invalid stochastic gate."""
        validator = RegimeValidator()
        market_data = create_test_market_data()

        dq_ok = CheckResult(
            name="data_quality",
            executed=True,
            passed=True,
            metrics={"issues": []},
            reason=None,
        )
        hmm_ok = CheckResult(
            name="hmm_regime",
            executed=True,
            passed=True,
            metrics={
                "range_prob_agg": 0.42,
                "trend_prob_agg": 0.18,
                "artifact_version": "test",
                "pipeline_version": "test",
            },
            reason=None,
        )
        range_ok = CheckResult(
            name="range_quality",
            executed=True,
            passed=True,
            metrics={
                "range_high": 40200.0,
                "range_low": 39800.0,
                "current_price": 40050.0,
            },
            reason=None,
        )
        vol_skipped = CheckResult(
            name="volatility_bounds",
            executed=False,
            passed=True,
            metrics={},
            reason=None,
        )
        stoch_fail = CheckResult(
            name="stochastic_regime",
            executed=True,
            passed=False,
            metrics={
                "survival_prob": 0.32,
                "hurst_exponent": 0.54,
                "ou_halflife": 12.5,
            },
            reason="stochastic_regime_fail(test)",
        )

        with patch.object(validator, "check_data_quality", return_value=dq_ok), patch.object(
            validator, "check_hmm_regime", return_value=hmm_ok
        ), patch.object(
            validator, "check_range_quality", return_value=range_ok
        ), patch.object(
            validator, "check_volatility_bounds", return_value=vol_skipped
        ), patch.object(
            validator, "check_stochastic_regime", return_value=stoch_fail
        ):
            result = validator.validate(market_data)

        assert result.is_valid is False
        assert result.survival_prob == pytest.approx(0.32)
        assert result.hurst_exponent == pytest.approx(0.54)
        assert result.ou_halflife == pytest.approx(12.5)

    @pytest.mark.parametrize("failure_stage", ["hmm", "range", "volatility", "stochastic"])
    def test_hmm_lineage_preserved_on_hmm_executed_failures(self, failure_stage):
        """Top-level HMM lineage should survive every post-HMM invalid return."""
        validator = RegimeValidator()
        market_data = create_test_market_data()

        dq_ok = CheckResult(
            name="data_quality",
            executed=True,
            passed=True,
            metrics={"issues": []},
            reason=None,
        )
        hmm_check = CheckResult(
            name="hmm_regime",
            executed=True,
            passed=failure_stage != "hmm",
            metrics={
                "range_prob_agg": 0.42,
                "trend_prob_agg": 0.18,
                "artifact_version": "artifact-test",
                "pipeline_version": "pipeline-test",
                "calibration_provenance": {"status": "ok"},
                "regime_conf": 0.77,
                "posterior_mode": "range",
                "persistence_prob": 0.88,
                "trained_at_utc": "2026-04-10T00:00:00+00:00",
                "posteriors": {"range": 0.42, "trend": 0.18},
                "utility_score": 0.12,
            },
            reason="hmm_fail" if failure_stage == "hmm" else None,
        )
        range_check = CheckResult(
            name="range_quality",
            executed=True,
            passed=failure_stage != "range",
            metrics={
                "range_high": 40200.0,
                "range_low": 39800.0,
                "current_price": 40050.0,
            },
            reason="range_fail" if failure_stage == "range" else None,
        )
        vol_check = CheckResult(
            name="volatility_bounds",
            executed=failure_stage == "volatility",
            passed=failure_stage != "volatility",
            metrics={},
            reason="volatility_fail" if failure_stage == "volatility" else None,
        )
        stoch_check = CheckResult(
            name="stochastic_regime",
            executed=failure_stage == "stochastic",
            passed=failure_stage != "stochastic",
            metrics={"survival_prob": 0.32},
            reason="stochastic_fail" if failure_stage == "stochastic" else None,
        )

        with patch.object(validator, "check_data_quality", return_value=dq_ok), patch.object(
            validator, "check_hmm_regime", return_value=hmm_check
        ), patch.object(
            validator, "check_range_quality", return_value=range_check
        ), patch.object(
            validator, "check_volatility_bounds", return_value=vol_check
        ), patch.object(
            validator, "check_stochastic_regime", return_value=stoch_check
        ):
            result = validator.validate(market_data)

        assert result.is_valid is False
        assert result.range_prob == pytest.approx(0.42)
        assert result.trend_prob == pytest.approx(0.18)
        assert result.hmm_artifact_version == "artifact-test"
        assert result.hmm_pipeline_version == "pipeline-test"
        assert result.hmm_calibration_provenance == {"status": "ok"}
        assert result.regime_conf == pytest.approx(0.77)
        assert result.posterior_mode == "range"
        assert result.persistence_prob == pytest.approx(0.88)
        assert result.hmm_trained_at_utc == "2026-04-10T00:00:00+00:00"
        assert result.posteriors == {"range": 0.42, "trend": 0.18}
        assert result.regime_utility == pytest.approx(0.12)

    @pytest.mark.asyncio
    async def test_database_hmm_regime_passed_uses_hmm_stage_result(self):
        """Storage should persist HMM-stage pass even when later gates fail."""
        from neutralgrid.grid.calculator import GridParams
        from neutralgrid.storage.database import Database

        db_dir = Path.cwd() / ".pytest_tmp" / f"hmm_db_{uuid4().hex}"
        db_dir.mkdir(parents=True, exist_ok=False)
        database = Database(str(db_dir / "test.sqlite"))

        try:
            await database.initialize()

            later_gate_fail = ValidationResult(
                symbol="TESTUSDT",
                is_valid=False,
                tf_1h=TimeframeResult("1h", True, checks={"range_prob": 0.7}),
                tf_15m=TimeframeResult("15m", False, reason="range_fail"),
                range_prob=0.7,
            )
            grid_params = GridParams(symbol="TESTUSDT", is_valid=False)

            run_id = await database.save_bot_run(grid_params, later_gate_fail)
            db = await database._get_conn()
            async with db.execute(
                "SELECT hmm_regime_passed, validation_reason FROM bot_runs WHERE id = ?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()

            assert row is not None
            assert row[0] == 1
            assert row[1] == "15M: range_fail"

            hmm_fail = ValidationResult(
                symbol="TESTUSDT",
                is_valid=False,
                tf_1h=TimeframeResult("1h", False, checks={"range_prob": 0.1}, reason="hmm_fail"),
                range_prob=0.1,
            )
            run_id = await database.save_bot_run(grid_params, hmm_fail)
            async with db.execute(
                "SELECT hmm_regime_passed, validation_reason FROM bot_runs WHERE id = ?",
                (run_id,),
            ) as cursor:
                row = await cursor.fetchone()

            assert row is not None
            assert row[0] == 0
            assert row[1] == "HMM: hmm_fail"
        finally:
            await database.close()
            shutil.rmtree(db_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
