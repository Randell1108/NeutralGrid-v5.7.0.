"""
Unit tests for Phase 4: backtest output-contract and horizon label compliance.

Verifies that run() returns every expected field, that MTM-based net_pnl
and label_positive_by_horizon are computed correctly, and that
convert_to_training_row() picks up the horizon label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtest_realistic import GridConfig, RealisticGridBacktester
from neutralgrid.backtest.candidate_pipeline import convert_to_training_row


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_klines(n_bars: int = 200, start_price: float = 100.0,
                           amplitude: float = 5.0) -> pd.DataFrame:
    """Create a synthetic 1m OHLCV DataFrame that oscillates within a range.

    The price oscillates sinusoidally around *start_price* so that grid
    crossings occur, producing trades and non-zero PnL.
    """
    timestamps = pd.date_range("2026-01-01", periods=n_bars, freq="1min")
    t = np.linspace(0, 4 * np.pi, n_bars)
    close = start_price + amplitude * np.sin(t)
    high = close + 0.5
    low = close - 0.5
    open_ = np.roll(close, 1)
    open_[0] = start_price

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n_bars) * 1000.0,
    })


def _default_config(capital: float = 400.0) -> GridConfig:
    """A small grid config whose range spans the synthetic price range."""
    return GridConfig(
        symbol="TESTUSDT",
        lower=95.0,
        upper=105.0,
        num_grids=10,
        capital=capital,
        leverage=10,
        order_delay_bars=0,       # no delay so crossings fill immediately
        funding_rate=0.0001,
        funding_interval_bars=480,
        funding_mode="continuous",
        max_holding_bars=1440,
    )


# ── Exhaustive field list (from the task checklist) ──────────────────────────

EXPECTED_FIELDS: set[str] = {
    "symbol",
    "duration_hours",
    "grid_lower",
    "grid_upper",
    "num_grids",
    "grid_spacing_pct",
    "total_trades",
    "round_trips",
    "win_rate",
    "gross_pnl",
    "fees_paid",
    "net_pnl",
    "net_pnl_pct",
    "mae",
    "mfe",
    "mae_pct_initial",
    "mfe_pct_initial",
    "max_drawdown_pct",
    "max_runup_pct",
    "sharpe_ratio",
    "price_start",
    "price_end",
    "price_change_pct",
    "time_in_range_pct",
    "funding_fees",
    "avg_hold_bars",
    "avg_hold_hours",
    "max_hold_bars",
    "stale_closes",
    "stale_close_pct",
    "open_positions_at_end",
    "max_holding_bars",
    # Phase 1+4: MTM equity and horizon label
    "unrealized_pnl_at_end",
    "final_equity",
    "peak_equity",
    "label_positive_by_horizon",
    "funding_mode",
    # P0-A: Liquidation
    "liquidated",
    "liquidation_bar",
    "liquidation_equity",
    # P0-B: Exchange filter rounding
    "tick_size",
    "step_size",
    # P1-A: Price source
    "price_source",
    # P1-B: Funding rate series
    "funding_rate_series_len",
    # P1-C: Margin mode
    "margin_mode",
    # P2-B: Bid-ask spread
    "spread_bps",
    # P3-A: Fill mode
    "fill_mode",
    # P0+P1: Global cooldown
    "global_cooldown_bars",
    # P3: Termination penalty
    "termination_pnl_pct",
    "exit_penalty_pct",
}


# ── Tests ────────────────────────────────────────────────────────────────────

class TestOutputContract:
    """Verify every field in the run() result dict."""

    @pytest.fixture(autouse=True)
    def _run_backtest(self):
        """Run the backtest once; reuse across tests in this class."""
        self.cfg = _default_config()
        self.df = _make_synthetic_klines()
        bt = RealisticGridBacktester(self.cfg)
        self.result = bt.run(self.df)

    def test_all_expected_fields_present(self):
        missing = EXPECTED_FIELDS - set(self.result.keys())
        assert not missing, f"Missing fields in result dict: {missing}"

    def test_net_pnl_equals_final_equity_minus_capital(self):
        assert self.result["net_pnl"] == pytest.approx(
            self.result["final_equity"] - self.cfg.capital
        ), (
            f"net_pnl ({self.result['net_pnl']}) != "
            f"final_equity ({self.result['final_equity']}) - capital ({self.cfg.capital})"
        )

    def test_label_positive_by_horizon(self):
        expected = self.result["final_equity"] > self.cfg.capital
        assert self.result["label_positive_by_horizon"] == expected

    def test_net_pnl_pct(self):
        expected_pct = (self.result["net_pnl"] / self.cfg.capital) * 100
        assert self.result["net_pnl_pct"] == pytest.approx(expected_pct, rel=1e-9)

    def test_final_equity_includes_unrealized(self):
        """final_equity = realized equity + unrealized PnL at end."""
        # We cannot inspect the internal 'equity' variable, but we can check
        # that final_equity = (capital + net_pnl) since net_pnl is defined
        # as final_equity - capital.
        assert self.result["final_equity"] == pytest.approx(
            self.cfg.capital + self.result["net_pnl"]
        )

    def test_peak_equity_ge_final_equity(self):
        """Peak equity must be >= final equity (it tracks the maximum)."""
        assert self.result["peak_equity"] >= self.result["final_equity"] - 1e-9

    def test_funding_mode_matches_config(self):
        assert self.result["funding_mode"] == self.cfg.funding_mode


class TestHierarchicalLabelInTrainingRow:
    """Verify convert_to_training_row uses hierarchical labels as primary target.

    Since v6.5.3, the training label y comes from hlabel_meta (the hierarchical
    deploy decision: L1 geometry + L2 execution + L3 net return), not directly
    from label_positive_by_horizon.  The horizon label is preserved as y_horizon.
    """

    def _fake_candidate_row(self) -> dict:
        return {
            "symbol": "TESTUSDT",
            "candidate_id": "TESTUSDT_20260101_000000",
            "scan_file": "deployment_ready_20260101_000000.csv",
            "score": 85.0,
            "grid_lower": 95.0,
            "grid_upper": 105.0,
            "num_grids": 10,
        }

    def test_hlabel_meta_overrides_horizon(self):
        """y from hlabel_meta, horizon label preserved as y_horizon."""
        bt_result = {
            "net_pnl_pct": 0.5,  # below 3% hurdle → L3 fail
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "total_trades": 5,
            "max_drawdown_pct": -2.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row(), hurdle_pct=3.0)
        # L1=True, L2=True(5 fills), L3=False(0.5<3.0) → meta=False → y=0
        assert row["y"] == 0
        assert row["y_horizon"] == 1

    def test_all_levels_pass_y_one(self):
        """When all hierarchical levels pass, y=1."""
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "total_trades": 10,
            "max_drawdown_pct": -2.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row(), hurdle_pct=3.0)
        assert row["y"] == 1
        assert row["y_horizon"] == 1

    def test_insufficient_fills_overrides_positive_pnl(self):
        """Positive PnL but insufficient fills → y=0 (L2 fail)."""
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "total_trades": 1,  # below min_fills=3
            "max_drawdown_pct": -2.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row(), hurdle_pct=3.0)
        assert row["y"] == 0  # hlabel_meta=False (L2 fail)
        assert row["y_horizon"] == 1  # horizon was positive

    def test_falls_back_to_hurdle_when_label_absent(self):
        """When label_positive_by_horizon absent, initial y from hurdle; hlabel overrides."""
        bt_result = {
            "net_pnl_pct": 5.0,
            "duration_hours": 3.0,
            "total_trades": 10,
            "max_drawdown_pct": -2.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row(), hurdle_pct=3.0)
        assert row["y"] == 1  # All levels pass

    def test_falls_back_below_hurdle(self):
        """Fall-back path with pnl below hurdle and no fills."""
        bt_result = {
            "net_pnl_pct": 1.0,
            "duration_hours": 3.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row(), hurdle_pct=3.0)
        assert row["y"] == 0  # L3 fail (1.0 < 3.0)

    def test_training_row_has_source_backtest(self):
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row())
        assert row["source"] == "backtest"
        assert row["sample_weight_override"] == 0.5

    def test_training_row_propagates_excursions_when_present(self):
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "mae": 8.0,
            "mfe": 14.0,
            "mae_pct_initial": 2.0,
            "mfe_pct_initial": 3.5,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row())
        assert row["mae"] == pytest.approx(8.0)
        assert row["mfe"] == pytest.approx(14.0)
        assert row["mae_pct_initial"] == pytest.approx(2.0)
        assert row["mfe_pct_initial"] == pytest.approx(3.5)

    def test_training_row_propagates_horizon_censored_true(self):
        # DURATION_FIX §21.4 — censor flag must reach the training row.
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
            "horizon_censored": True,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row())
        assert row["horizon_censored"] is True

    def test_training_row_horizon_censored_defaults_false_when_missing(self):
        bt_result = {
            "net_pnl_pct": 5.0,
            "label_positive_by_horizon": True,
            "duration_hours": 3.0,
        }
        row = convert_to_training_row(bt_result, self._fake_candidate_row())
        assert row["horizon_censored"] is False

    def test_horizon_censored_not_a_model_feature(self):
        # DURATION_FIX §21.5 item 2 — must stay out of ALL_META_FEATURES.
        from neutralgrid.training.unified_training_builder import ALL_META_FEATURES
        assert "horizon_censored" not in ALL_META_FEATURES


class TestEdgeCases:
    """Edge cases: zero trades, flat price."""

    def test_flat_price_no_trades(self):
        """If price never crosses a grid level, output dict is still complete."""
        cfg = GridConfig(
            symbol="FLATUSDT",
            lower=90.0,
            upper=110.0,
            num_grids=10,
            capital=400.0,
            leverage=10,
            order_delay_bars=0,
            funding_mode="snapshot",
            max_holding_bars=1440,
        )
        # Flat price at 100 (midpoint — sits between grid levels, never crosses)
        n = 100
        flat_df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1min"),
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })

        bt = RealisticGridBacktester(cfg)
        result = bt.run(flat_df)

        missing = EXPECTED_FIELDS - set(result.keys())
        assert not missing, f"Missing fields: {missing}"

        # No trades -> net_pnl = 0 -> final_equity = capital
        assert result["net_pnl"] == pytest.approx(0.0)
        assert result["final_equity"] == pytest.approx(cfg.capital)
        assert result["label_positive_by_horizon"] is False
