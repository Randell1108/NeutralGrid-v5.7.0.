"""Unit tests for MTM MAE/MFE excursion engine integration (v20260304)."""

from __future__ import annotations

import pytest

from neutralgrid.metrics.calculator import MetricsCalculator
from neutralgrid.metrics.excursion_mtm_v20260304 import calculate_mtm_excursions


def _mk_mark(close_time_ms: int, close_price: float) -> list:
    """Build a minimal mark-kline row compatible with parser expectations."""
    open_time_ms = max(0, close_time_ms - 60_000)
    return [
        open_time_ms, "0", "0", "0", str(close_price), "0", close_time_ms,
    ]


class TestMtmExcursionsV20260304:
    """Mathematical correctness checks for MAE/MFE on MTM equity."""

    def test_long_inventory_mae_and_mfe(self):
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 95.0),
            _mk_mark(3_000, 110.0),
        ]

        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mae_usdt == pytest.approx(5.0)
        assert m.mfe_usdt == pytest.approx(15.0)
        assert m.max_equity_usdt == pytest.approx(10.0)
        assert m.min_equity_usdt == pytest.approx(-5.0)

    def test_hedge_mode_long_and_short_offset(self):
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "LONG",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
            {
                "time": 1_001,
                "side": "SELL",
                "positionSide": "SHORT",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 110.0),
            _mk_mark(3_000, 90.0),
        ]

        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mae_usdt == pytest.approx(0.0)
        assert m.mfe_usdt == pytest.approx(0.0)
        assert m.max_equity_usdt == pytest.approx(0.0)
        assert m.min_equity_usdt == pytest.approx(0.0)

    def test_commission_and_funding_cashflows_affect_excursions(self):
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "1",
            },
        ]
        funding_fees = [
            {"time": 2_000, "income": "2"},
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 100.0),
        ]

        m = calculate_mtm_excursions(
            trades=trades,
            funding_fees=funding_fees,
            mark_klines=mark_klines,
        )
        assert m.mae_usdt == pytest.approx(1.0)
        assert m.mfe_usdt == pytest.approx(2.0)
        assert m.max_equity_usdt == pytest.approx(1.0)
        assert m.min_equity_usdt == pytest.approx(-1.0)


class TestMtmExcursionsEdgeCases:
    """Edge-case and coverage-gap tests for MAE/MFE engine."""

    def test_empty_inputs_return_zero_metrics(self):
        """MC-1: All-empty inputs hit the early return with zeroed metrics."""
        m = calculate_mtm_excursions(trades=[], funding_fees=[], mark_klines=[])
        assert m.mae_usdt == 0.0
        assert m.mfe_usdt == 0.0
        assert m.max_equity_usdt == 0.0
        assert m.min_equity_usdt == 0.0
        assert m.equity_points == 0
        assert m.mark_points_used == 0
        assert m.used_mark_prices is False

    def test_no_mark_data_falls_back_to_trade_price(self):
        """MC-2: When mark_klines is empty, first trade price becomes mark anchor.

        Equity path (no marks, BUY 1@100):
          trade@1000: mark=None->100, net=(1,100), unrealized=1*(100-100)=0, equity=0
        Single equity point, mark_points_used=0, used_mark_prices=False.
        """
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=[])
        assert m.mae_usdt == pytest.approx(0.0)
        assert m.mfe_usdt == pytest.approx(0.0)
        assert m.equity_points == 1
        assert m.mark_points_used == 0
        assert m.used_mark_prices is False

    def test_position_flip_long_to_short(self):
        """MC-4: BUY 1@100 then SELL 3@100 flips net from +1 to -2.

        Equity path:
          mark@1000:  mark=100, no position, equity=0
          trade@1000: net=(+1, 100), unrealized=0, equity=0
          mark@2000:  mark=90, unrealized=1*(90-100)=-10, equity=-10
          trade@2000: SELL 3 -> flip, net=(-2, 100), unrealized=(-2)*(90-100)=+20, equity=20

        mae = peak(0) - trough(-10) at mark@2000 = 10
        mfe = peak(20) - trough(-10) at trade@2000 = 30
        """
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
            {
                "time": 2_000,
                "side": "SELL",
                "positionSide": "BOTH",
                "qty": "3",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 90.0),
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mae_usdt == pytest.approx(10.0)
        assert m.mfe_usdt == pytest.approx(30.0)
        assert m.max_equity_usdt == pytest.approx(20.0)
        assert m.min_equity_usdt == pytest.approx(-10.0)

    def test_short_only_position(self):
        """MC-6: SELL 1@100 in BOTH mode (short net position).

        Equity path:
          mark@1000:  mark=100, no position, equity=0
          trade@1000: net=(-1, 100), unrealized=(-1)*(100-100)=0, equity=0
          mark@2000:  mark=110, unrealized=(-1)*(110-100)=-10, equity=-10
          mark@3000:  mark=90, unrealized=(-1)*(90-100)=+10, equity=+10

        mae = 0-(-10) = 10, mfe = 10-(-10) = 20
        """
        trades = [
            {
                "time": 1_000,
                "side": "SELL",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 110.0),
            _mk_mark(3_000, 90.0),
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mae_usdt == pytest.approx(10.0)
        assert m.mfe_usdt == pytest.approx(20.0)
        assert m.max_equity_usdt == pytest.approx(10.0)
        assert m.min_equity_usdt == pytest.approx(-10.0)

    def test_open_then_close_position(self):
        """MC-7: BUY 1@100, price dips to 95, rises to 110, SELL 1@110 (rpnl=10).

        Equity path:
          mark@1000:  mark=100, no position, equity=0
          trade@1000: net=(1,100), cash=0, unrealized=0, equity=0
          mark@2000:  mark=95, unrealized=-5, equity=-5
          mark@3000:  mark=110, unrealized=+10, equity=+10
          trade@3000: SELL 1 closes position, net=(0,0), cash=+10, unrealized=0, equity=+10

        mae = 0-(-5) = 5, mfe = 10-(-5) = 15
        """
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
            {
                "time": 3_000,
                "side": "SELL",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "110",
                "realizedPnl": "10",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 95.0),
            _mk_mark(3_000, 110.0),
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mae_usdt == pytest.approx(5.0)
        assert m.mfe_usdt == pytest.approx(15.0)
        assert m.max_equity_usdt == pytest.approx(10.0)
        assert m.min_equity_usdt == pytest.approx(-5.0)

    def test_invalid_trades_are_silently_skipped(self):
        """MC-9: Trades with missing side, zero qty, or negative time produce zero metrics."""
        trades = [
            # Missing "side" key entirely
            {"time": 1_000, "positionSide": "BOTH", "qty": "1", "price": "100",
             "realizedPnl": "0", "commission": "0"},
            # side is empty string
            {"time": 2_000, "side": "", "positionSide": "BOTH", "qty": "1",
             "price": "100", "realizedPnl": "0", "commission": "0"},
            # qty is zero
            {"time": 3_000, "side": "BUY", "positionSide": "BOTH", "qty": "0",
             "price": "100", "realizedPnl": "0", "commission": "0"},
            # price is zero
            {"time": 4_000, "side": "BUY", "positionSide": "BOTH", "qty": "1",
             "price": "0", "realizedPnl": "0", "commission": "0"},
            # time is negative
            {"time": -1, "side": "BUY", "positionSide": "BOTH", "qty": "1",
             "price": "100", "realizedPnl": "0", "commission": "0"},
            # qty is non-numeric
            {"time": 5_000, "side": "BUY", "positionSide": "BOTH", "qty": "abc",
             "price": "100", "realizedPnl": "0", "commission": "0"},
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=[])
        # All trades should be filtered out, resulting in the empty-events early return.
        assert m.mae_usdt == 0.0
        assert m.mfe_usdt == 0.0
        assert m.equity_points == 0

    def test_metadata_fields_mark_count_and_equity_points(self):
        """MC-10: Verify equity_points, mark_points_used, used_mark_prices.

        3 marks + 1 trade = 4 events, all produce equity samples.
        mark_points_used = 3 (all valid marks).
        """
        trades = [
            {
                "time": 1_000,
                "side": "BUY",
                "positionSide": "BOTH",
                "qty": "1",
                "price": "100",
                "realizedPnl": "0",
                "commission": "0",
            },
        ]
        mark_klines = [
            _mk_mark(1_000, 100.0),
            _mk_mark(2_000, 95.0),
            _mk_mark(3_000, 110.0),
        ]
        m = calculate_mtm_excursions(trades=trades, funding_fees=[], mark_klines=mark_klines)
        assert m.mark_points_used == 3
        assert m.used_mark_prices is True
        # 3 mark events + 1 trade event = 4 equity points
        assert m.equity_points == 4


class TestMetricsCalculatorIntegrationV20260304:
    """MetricsCalculator should expose MTM MAE/MFE and pct fields coherently."""

    def test_session_metrics_uses_mtm_excursions(self):
        calc = MetricsCalculator()
        session_data = {
            "symbol": "BTCUSDT",
            "start_time": 1_000,
            "end_time": 3_000,
            "investment_margin": 100.0,
            "trades": [
                {
                    "time": 1_000,
                    "side": "BUY",
                    "positionSide": "BOTH",
                    "qty": "1",
                    "price": "100",
                    "realizedPnl": "0",
                    "commission": "0",
                    "maker": True,
                },
            ],
            "realized_pnl": [],
            "commissions": [],
            "funding_fees": [],
            "mark_klines": [
                _mk_mark(1_000, 100.0),
                _mk_mark(2_000, 95.0),
                _mk_mark(3_000, 110.0),
            ],
        }

        metrics = calc.calculate_session_metrics(session_data=session_data)
        assert metrics.mae == pytest.approx(5.0)
        assert metrics.mfe == pytest.approx(15.0)
        assert metrics.mae_pct_initial == pytest.approx(5.0)
        assert metrics.mfe_pct_initial == pytest.approx(15.0)
