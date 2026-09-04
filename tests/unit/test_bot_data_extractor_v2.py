"""Comprehensive tests for new_bot_data_extractor.py features.

Covers: parse_user_text, parse_trade_fills_from_text, parse_pnl_curve_from_text,
compute_pnl_curve_excursions, compute_derived_fields (mode-aware), _compute_liquidation_features,
search_candidate_match, validate_data_coherence, text_fills_to_dataframe, build_bot_data_from_text.
"""

from __future__ import annotations

import math
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
import pytest

# Project root on sys.path so `from new_bot_data_extractor import ...` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from new_bot_data_extractor import (
    ExtractedBotData,
    ParsedTradeFill,
    TradeMetrics,
    _compute_liquidation_features,
    build_bot_data_from_text,
    compute_pnl_curve_excursions,
    parse_pnl_curve_from_text,
    parse_trade_fills_from_text,
    parse_user_text,
    search_candidate_match,
    text_fills_to_dataframe,
    validate_data_coherence,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def full_bot_text() -> str:
    """A realistic Binance Futures grid bot copy-paste block."""
    return (
        "TAOUSDT Perpetual\n"
        "Neutral Grid\n"
        "Strategy ID: 1234567890\n"
        "Expired\n"
        "Arithmetic\n"
        "Ended on: 2026-03-08 14:30:00\n"
        "5d 12h 30m\n"
        "Invested Margin: 500.00 USDT\n"
        "Initial Leverage: 10x\n"
        "Number of Grids: 50\n"
        "Price Range: 150.50 - 250.75\n"
        "Grid Start Price: 200.00\n"
        "Qty/Order: 0.058\n"
        "Profit/Grid: 0.20% - 0.35%\n"
        "Total Profit: 12.50 USDT (2.50%)\n"
        "Matched Profit: 15.80 USDT\n"
        "Unmatched PNL: -4.30\n"
        "Funding Fee: 1.00\n"
        "Total Trades: 320\n"
        "Annualized Yield: 18.25%\n"
        "Liq Price (Long): 120.50\n"
        "Liq Price (Short): 290.00\n"
        "Trigger Price: 200.25\n"
    )


@pytest.fixture()
def bot_data_basic() -> ExtractedBotData:
    """ExtractedBotData with typical values."""
    return ExtractedBotData(
        strategy_id=1234567890,
        symbol="TAOUSDT",
        status="expired",
        mode="arithmetic",
        end_time_utc=datetime(2026, 3, 8, 14, 30, 0),
        duration_hours=132.5,
        invested_margin_usdt=500.0,
        leverage=10,
        grids_count=50,
        price_range_low=150.50,
        price_range_high=250.75,
        grid_start_price=200.00,
        qty_per_order=0.058,
        total_profit_usdt=12.50,
        pnl_pct=2.50,
        realized_pnl_usdt=15.80,
        unrealized_pnl_usdt=-4.30,
        funding_fee_usdt=1.00,
        total_trades=320,
        apy_pct=18.25,
        liq_price_long=120.50,
        liq_price_short=290.00,
        trigger_price=200.25,
    )


@pytest.fixture
def workspace_tmp_dir() -> Path:
    root = Path("tests") / ".tmp_test_bot_data_extractor_v2"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / uuid4().hex
    case_dir.mkdir()
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.fixture()
def empty_trade_metrics() -> TradeMetrics:
    return TradeMetrics()


# =============================================================================
# 1. parse_user_text()
# =============================================================================

class TestParseUserText:
    """Tests for regex-based field extraction from pasted text."""

    # -- Symbol extraction --

    def test_symbol_taousdt(self):
        data = parse_user_text("TAOUSDT Perpetual\nSome other text")
        assert data["symbol"] == "TAOUSDT"

    def test_symbol_movrusdt(self):
        data = parse_user_text("MOVRUSDT Perpetual\nGrid")
        assert data["symbol"] == "MOVRUSDT"

    def test_symbol_1000pepeusdt(self):
        data = parse_user_text("1000PEPEUSDT Perpetual\nNeutral Grid")
        assert data["symbol"] == "1000PEPEUSDT"

    def test_symbol_only_in_first_15_lines(self):
        """Symbol must appear in first 15 lines."""
        lines = ["line"] * 16 + ["BTCUSDT Perpetual"]
        data = parse_user_text("\n".join(lines))
        assert "symbol" not in data

    def test_symbol_within_15_lines(self):
        lines = ["line"] * 10 + ["ETHUSDT Perpetual"]
        data = parse_user_text("\n".join(lines))
        assert data["symbol"] == "ETHUSDT"

    # -- Strategy ID --

    def test_strategy_id(self):
        data = parse_user_text("Strategy ID: 1234567890")
        assert data["strategy_id"] == 1234567890

    def test_strategy_id_12_digits(self):
        data = parse_user_text("Strategy ID: 123456789012")
        assert data["strategy_id"] == 123456789012

    def test_strategy_id_case_insensitive(self):
        data = parse_user_text("strategy id: 9876543210")
        assert data["strategy_id"] == 9876543210

    def test_strategy_id_missing(self):
        data = parse_user_text("No strategy info here")
        assert "strategy_id" not in data

    # -- Status --

    def test_status_expired(self):
        data = parse_user_text("BTCUSDT\nExpired")
        assert data["status"] == "expired"

    def test_status_cancelled(self):
        data = parse_user_text("BTCUSDT\nCancelled")
        assert data["status"] == "cancelled"

    def test_status_canceled_american(self):
        data = parse_user_text("BTCUSDT\nCanceled")
        assert data["status"] == "cancelled"

    def test_status_ended(self):
        data = parse_user_text("BTCUSDT\nEnded")
        assert data["status"] == "ended"

    def test_status_case_insensitive(self):
        data = parse_user_text("BTCUSDT\nEXPIRED")
        assert data["status"] == "expired"

    def test_status_missing(self):
        data = parse_user_text("BTCUSDT\nActive")
        assert "status" not in data

    def test_unavailable_profit_placeholder_is_not_coerced_to_zero(self):
        data = parse_user_text("Total Matched Profit\n--")
        assert "realized_pnl_usdt" not in data

    # -- Mode --

    def test_mode_arithmetic(self):
        data = parse_user_text("BTCUSDT\nArithmetic\nGrid")
        assert data["mode"] == "arithmetic"

    def test_mode_geometric(self):
        data = parse_user_text("BTCUSDT\nGeometric\nGrid")
        assert data["mode"] == "geometric"

    def test_mode_case_insensitive(self):
        data = parse_user_text("BTCUSDT\nARITHMETIC")
        assert data["mode"] == "arithmetic"

    def test_mode_missing(self):
        data = parse_user_text("BTCUSDT\nNeutral Grid")
        assert "mode" not in data

    # -- End time --

    def test_end_time_ended_time(self):
        """Regex requires 'Ende' or 'Ended' prefix."""
        data = parse_user_text("Ended Time: 2026-03-08 14:30:00")
        # Manual-paste timestamps are interpreted as operator-local (America/Lima,
        # UTC-5) and converted to UTC (commit d1e5591): 14:30 local -> 19:30 UTC.
        assert data["end_time_utc"] == datetime(2026, 3, 8, 19, 30, 0, tzinfo=timezone.utc)

    def test_end_time_ende(self):
        """'Ende' (without d) also matches due to 'Ended?' pattern."""
        data = parse_user_text("Ende: 2026-03-08 14:30:00")
        # America/Lima (UTC-5) -> UTC: 14:30 local -> 19:30 UTC.
        assert data["end_time_utc"] == datetime(2026, 3, 8, 19, 30, 0, tzinfo=timezone.utc)

    def test_end_time_ended_on(self):
        data = parse_user_text("Ended on: 2026-01-15 08:00:00")
        # America/Lima (UTC-5) -> UTC: 08:00 local -> 13:00 UTC.
        assert data["end_time_utc"] == datetime(2026, 1, 15, 13, 0, 0, tzinfo=timezone.utc)

    def test_end_time_missing(self):
        data = parse_user_text("No end time here")
        assert "end_time_utc" not in data

    # -- Duration --

    def test_duration_xd_yh_zm(self):
        data = parse_user_text("5d 12h 30m")
        expected = 5 * 24 + 12 + 30 / 60
        assert data["duration_hours"] == pytest.approx(expected)

    def test_duration_xd_hhmmss(self):
        data = parse_user_text("3D 04:15:30")
        expected = 3 * 24 + 4 + 15 / 60 + 30 / 3600
        assert data["duration_hours"] == pytest.approx(expected)

    def test_duration_decimal_hours(self):
        data = parse_user_text("Duration: 48.5 hours")
        assert data["duration_hours"] == pytest.approx(48.5)

    def test_duration_decimal_hrs(self):
        data = parse_user_text("Duration: 72.0 hrs")
        assert data["duration_hours"] == pytest.approx(72.0)

    def test_duration_hours_minutes_without_trailing_m(self):
        data = parse_user_text("Duration\n6h 23")
        assert data["duration_hours"] == pytest.approx(6 + 23 / 60)

    def test_duration_minutes_only(self):
        data = parse_user_text("Duration\n34m")
        assert data["duration_hours"] == pytest.approx(34 / 60)

    def test_duration_zero_minutes(self):
        data = parse_user_text("0d 5h 0m")
        assert data["duration_hours"] == pytest.approx(5.0)

    def test_duration_missing(self):
        data = parse_user_text("No duration info")
        assert "duration_hours" not in data

    # -- Margin --

    def test_margin_standard(self):
        data = parse_user_text("Invested Margin: 500.00 USDT")
        assert data["invested_margin_usdt"] == pytest.approx(500.0)

    def test_margin_with_comma(self):
        data = parse_user_text("Margin: 1,500.00 USDT")
        assert data["invested_margin_usdt"] == pytest.approx(1500.0)

    def test_margin_no_usdt_suffix(self):
        data = parse_user_text("Margin: 200.50")
        assert data["invested_margin_usdt"] == pytest.approx(200.50)

    def test_invested_margin_does_not_capture_maintenance_margin(self):
        data = parse_user_text(
            """PNL
Invested Margin (USDT)
34.00
Positions
Maintenance Margin
0.9176 USDT
Position Margin
6.11 USDT (Isolated)
"""
        )
        assert data["invested_margin_usdt"] == pytest.approx(34.0)

    # -- Leverage --

    def test_leverage(self):
        data = parse_user_text("Initial Leverage: 10x")
        assert data["leverage"] == 10

    def test_leverage_without_initial(self):
        data = parse_user_text("Leverage: 20X")
        assert data["leverage"] == 20

    # -- Number of grids --

    def test_grids_count(self):
        data = parse_user_text("Number of Grids: 50")
        assert data["grids_count"] == 50

    def test_grids_count_short_form(self):
        data = parse_user_text("Grids: 100")
        assert data["grids_count"] == 100

    # -- Price range --

    def test_price_range_standard(self):
        data = parse_user_text("Price Range: 150.50 - 250.75")
        assert data["price_range_low"] == pytest.approx(150.50)
        assert data["price_range_high"] == pytest.approx(250.75)

    def test_price_range_en_dash(self):
        data = parse_user_text("Price Range: 10.00\u201320.00")
        assert data["price_range_low"] == pytest.approx(10.0)
        assert data["price_range_high"] == pytest.approx(20.0)

    # -- Grid start price --

    def test_grid_start_price(self):
        data = parse_user_text("Grid Start Price: 200.00")
        assert data["grid_start_price"] == pytest.approx(200.00)

    # -- Qty/Order --

    def test_qty_per_order(self):
        data = parse_user_text("Qty/Order: 0.058")
        assert data["qty_per_order"] == pytest.approx(0.058)

    def test_qty_per_order_with_comma(self):
        data = parse_user_text("Qty/Order: 1,000")
        assert data["qty_per_order"] == pytest.approx(1000.0)

    # -- Profit/Grid --

    def test_profit_per_grid_range(self):
        data = parse_user_text("Profit/Grid: 0.20% - 0.35%")
        assert data["profit_per_grid_pct_low"] == pytest.approx(0.20)
        assert data["profit_per_grid_pct_high"] == pytest.approx(0.35)

    def test_profit_per_grid_single_value(self):
        data = parse_user_text("Profit/Grid: 0.50%")
        assert data["profit_per_grid_pct_low"] == pytest.approx(0.50)
        assert "profit_per_grid_pct_high" not in data

    # -- Total Profit --

    def test_total_profit(self):
        data = parse_user_text("Total Profit: 12.50 USDT (2.50%)")
        assert data["total_profit_usdt"] == pytest.approx(12.50)

    def test_total_profit_negative(self):
        data = parse_user_text("Total Profit: -5.80 USDT (-1.16%)")
        assert data["total_profit_usdt"] == pytest.approx(-5.80)

    def test_total_profit_not_matched(self):
        """Ensure 'Total Matched' does NOT match the 'Total Profit' pattern."""
        text = "Total Matched Profit: 99.00 USDT\nTotal Profit: 12.50 USDT"
        data = parse_user_text(text)
        assert data["total_profit_usdt"] == pytest.approx(12.50)

    # -- PnL % --

    def test_pnl_pct_in_parens(self):
        data = parse_user_text("Total Profit: 12.50 USDT (2.50%)")
        assert data["pnl_pct"] == pytest.approx(2.50)

    def test_pnl_pct_standalone(self):
        data = parse_user_text("PnL: -1.50%")
        assert data["pnl_pct"] == pytest.approx(-1.50)

    def test_roi_pct(self):
        data = parse_user_text("ROI: 3.20%")
        assert data["pnl_pct"] == pytest.approx(3.20)

    # -- Matched Profit --

    def test_matched_profit(self):
        data = parse_user_text("Matched Profit: 15.80 USDT")
        assert data["realized_pnl_usdt"] == pytest.approx(15.80)

    def test_matched_profit_negative(self):
        data = parse_user_text("Matched Profit: -2.50")
        assert data["realized_pnl_usdt"] == pytest.approx(-2.50)

    # -- Unmatched PNL --

    def test_unmatched_pnl(self):
        data = parse_user_text("Unmatched PNL: -4.30")
        assert data["unrealized_pnl_usdt"] == pytest.approx(-4.30)

    def test_unmatched_pnl_positive(self):
        data = parse_user_text("Unmatched PnL: 2.10")
        assert data["unrealized_pnl_usdt"] == pytest.approx(2.10)

    # -- Funding Fee --

    def test_funding_fee(self):
        data = parse_user_text("Funding Fee: 1.00")
        assert data["funding_fee_usdt"] == pytest.approx(1.00)

    def test_funding_fee_negative(self):
        data = parse_user_text("Funding Fee: -0.75")
        assert data["funding_fee_usdt"] == pytest.approx(-0.75)

    # -- Total Trades --

    def test_total_trades(self):
        data = parse_user_text("Total Trades: 320")
        assert data["total_trades"] == 320

    def test_total_matched_trades(self):
        data = parse_user_text("Total Matched Trades: 160")
        assert data["total_trades"] == 160

    # -- APY --

    def test_apy(self):
        data = parse_user_text("Annualized Yield: 18.25%")
        assert data["apy_pct"] == pytest.approx(18.25)

    def test_apy_short_form(self):
        data = parse_user_text("APY: 25.00%")
        assert data["apy_pct"] == pytest.approx(25.00)

    def test_apy_negative(self):
        data = parse_user_text("APY: -10.50%")
        assert data["apy_pct"] == pytest.approx(-10.50)

    # -- Liquidation prices --

    def test_liq_price_long(self):
        data = parse_user_text("Liq Price (Long): 120.50")
        assert data["liq_price_long"] == pytest.approx(120.50)

    def test_liq_price_short(self):
        data = parse_user_text("Liq Price (Short): 290.00")
        assert data["liq_price_short"] == pytest.approx(290.00)

    def test_liquidation_price_long_full(self):
        data = parse_user_text("Liquidation Price (Long): 98.00")
        assert data["liq_price_long"] == pytest.approx(98.00)

    def test_liquidation_price_short_full(self):
        data = parse_user_text("Liquidation Price (Short): 350.75")
        assert data["liq_price_short"] == pytest.approx(350.75)

    def test_liq_long_without_parens(self):
        data = parse_user_text("Liq Long: 95.00")
        assert data["liq_price_long"] == pytest.approx(95.00)

    # -- Trigger price --

    def test_trigger_price(self):
        data = parse_user_text("Trigger Price: 200.25")
        assert data["trigger_price"] == pytest.approx(200.25)

    def test_trigger_price_missing(self):
        data = parse_user_text("No trigger here")
        assert "trigger_price" not in data

    # -- Full text integration --

    def test_full_text_all_fields(self, full_bot_text):
        data = parse_user_text(full_bot_text)
        assert data["symbol"] == "TAOUSDT"
        assert data["strategy_id"] == 1234567890
        assert data["status"] == "expired"
        assert data["mode"] == "arithmetic"
        # America/Lima (UTC-5) -> UTC: 14:30 local -> 19:30 UTC (commit d1e5591).
        assert data["end_time_utc"] == datetime(2026, 3, 8, 19, 30, 0, tzinfo=timezone.utc)
        assert data["duration_hours"] == pytest.approx(5 * 24 + 12 + 30 / 60)
        assert data["invested_margin_usdt"] == pytest.approx(500.0)
        assert data["leverage"] == 10
        assert data["grids_count"] == 50
        assert data["price_range_low"] == pytest.approx(150.50)
        assert data["price_range_high"] == pytest.approx(250.75)
        assert data["grid_start_price"] == pytest.approx(200.00)
        assert data["qty_per_order"] == pytest.approx(0.058)
        assert data["profit_per_grid_pct_low"] == pytest.approx(0.20)
        assert data["profit_per_grid_pct_high"] == pytest.approx(0.35)
        assert data["total_profit_usdt"] == pytest.approx(12.50)
        assert data["pnl_pct"] == pytest.approx(2.50)
        assert data["realized_pnl_usdt"] == pytest.approx(15.80)
        assert data["unrealized_pnl_usdt"] == pytest.approx(-4.30)
        assert data["funding_fee_usdt"] == pytest.approx(1.00)
        assert data["total_trades"] == 320
        assert data["apy_pct"] == pytest.approx(18.25)
        assert data["liq_price_long"] == pytest.approx(120.50)
        assert data["liq_price_short"] == pytest.approx(290.00)
        assert data["trigger_price"] == pytest.approx(200.25)

    def test_empty_text_returns_empty_dict(self):
        data = parse_user_text("")
        assert data == {}


# =============================================================================
# 2. parse_trade_fills_from_text()
# =============================================================================

class TestParseTradeFillsFromText:
    """Tests for extracting individual trade fills from user text."""

    def test_standard_format(self):
        text = "Buy 180.50 USDT 0.10 Fee: 0.01 Realized: 1.80"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 1
        f = fills[0]
        assert f.side == "BUY"
        assert f.price == pytest.approx(180.50)
        assert f.quantity == pytest.approx(0.10)
        assert f.fee == pytest.approx(0.01)
        assert f.realized_profit == pytest.approx(1.80)

    def test_uppercase_format(self):
        text = "BUY 180.50 0.10 0.01 0.00"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 1
        assert fills[0].side == "BUY"
        assert fills[0].price == pytest.approx(180.50)

    def test_sell_side(self):
        text = "Sell 200.00 0.05 0.005 2.50"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 1
        assert fills[0].side == "SELL"
        assert fills[0].price == pytest.approx(200.00)
        assert fills[0].quantity == pytest.approx(0.05)

    def test_numbered_format(self):
        text = "1. Buy 180.50 0.058 0.01808 0.00\n2. Sell 182.00 0.058 0.01812 1.50"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 2
        assert fills[0].side == "BUY"
        assert fills[1].side == "SELL"
        assert fills[1].realized_profit == pytest.approx(1.50)

    def test_multiple_fills(self):
        text = (
            "Buy 100.00 1.0\n"
            "Sell 110.00 1.0\n"
            "Buy 105.00 2.0\n"
        )
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 3
        assert fills[0].price == pytest.approx(100.0)
        assert fills[1].price == pytest.approx(110.0)
        assert fills[2].price == pytest.approx(105.0)

    def test_no_fee_no_realized(self):
        text = "Buy 50.00 0.5"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 1
        assert fills[0].fee == 0.0
        assert fills[0].realized_profit == 0.0

    def test_empty_text_returns_empty(self):
        fills = parse_trade_fills_from_text("")
        assert fills == []

    def test_no_matching_lines(self):
        fills = parse_trade_fills_from_text("Random text with no trades")
        assert fills == []

    def test_negative_realized_profit(self):
        text = "Sell 95.00 1.0 0.01 -5.00"
        fills = parse_trade_fills_from_text(text)
        assert len(fills) == 1
        assert fills[0].realized_profit == pytest.approx(-5.00)


# =============================================================================
# 3. parse_pnl_curve_from_text()
# =============================================================================

class TestParsePnlCurveFromText:
    """Tests for extracting PnL curve values from numbered text lines."""

    def test_basic_numbered_entries(self):
        text = "1  +0.50 USDT\n2  1.20\n3  -0.80 USDT"
        values = parse_pnl_curve_from_text(text)
        assert len(values) == 3
        assert values[0] == pytest.approx(0.50)
        assert values[1] == pytest.approx(1.20)
        assert values[2] == pytest.approx(-0.80)

    def test_period_delimited(self):
        text = "1. 2.50\n2. 3.00\n3. 1.80"
        values = parse_pnl_curve_from_text(text)
        assert len(values) == 3
        assert values[0] == pytest.approx(2.50)
        assert values[1] == pytest.approx(3.00)
        assert values[2] == pytest.approx(1.80)

    def test_with_plus_minus(self):
        text = "1) +10.00 USDT\n2) -5.00 USDT\n3) +15.00 USDT"
        values = parse_pnl_curve_from_text(text)
        assert len(values) == 3
        assert values[0] == pytest.approx(10.00)
        assert values[1] == pytest.approx(-5.00)
        assert values[2] == pytest.approx(15.00)

    def test_comma_thousands(self):
        text = "1  1,500.00 USDT\n2  2,000.50"
        values = parse_pnl_curve_from_text(text)
        assert len(values) == 2
        assert values[0] == pytest.approx(1500.00)
        assert values[1] == pytest.approx(2000.50)

    def test_empty_returns_empty(self):
        values = parse_pnl_curve_from_text("")
        assert values == []

    def test_no_matching_lines(self):
        values = parse_pnl_curve_from_text("Just random text with no numbers")
        assert values == []

    def test_single_entry(self):
        text = "1  5.00"
        values = parse_pnl_curve_from_text(text)
        assert len(values) == 1
        assert values[0] == pytest.approx(5.00)

    def test_stops_at_end_of_pnl_curve_section(self):
        text = (
            "PnL Curve:\n"
            "1. 0.00000\n"
            "2. 9.20499\n"
            "3. -5.93318\n"
            "\n"
            "Grid Details\n"
            "Price Range\n"
            "0.03353 - 0.03663 USDT\n"
            "Profit Per Grid\n"
            "1.68% - 1.80%\n"
            "Grid Start Price\n"
            "0.03600 USDT\n"
        )
        values = parse_pnl_curve_from_text(text)
        assert values == pytest.approx([0.0, 9.20499, -5.93318])

    def test_bare_values_require_explicit_curve_heading(self):
        text = (
            "PnL Curve\n"
            "0.000000\n"
            "-0.051474\n"
            "0.113465\n"
            "Grid Details\n"
            "10.00\n"
        )
        values = parse_pnl_curve_from_text(text)
        assert values == pytest.approx([0.0, -0.051474, 0.113465])


# =============================================================================
# 4. compute_pnl_curve_excursions()
# =============================================================================

class TestComputePnlCurveExcursions:
    """Tests for MAE/MFE computation from cumulative PnL curve."""

    def test_monotonically_increasing(self):
        """Monotonically increasing: MAE=0, MFE = total rise."""
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        assert mae == pytest.approx(0.0)
        assert mfe == pytest.approx(5.0)

    def test_monotonically_decreasing(self):
        """Monotonically decreasing: MAE = total drop, MFE=0."""
        values = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        assert mae == pytest.approx(5.0)
        assert mfe == pytest.approx(0.0)

    def test_peak_then_valley(self):
        """Rise to peak then drop: correct drawdown and runup."""
        values = [0.0, 5.0, 10.0, 7.0, 3.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        # Peak at 10, then drops to 3 -> drawdown = 7
        assert mae == pytest.approx(7.0)
        # Trough at 0, rises to 10 -> runup = 10
        assert mfe == pytest.approx(10.0)

    def test_valley_then_peak(self):
        """Drop to valley then rise: correct drawdown and runup."""
        values = [0.0, -5.0, -10.0, -3.0, 5.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        # Peak at 0, drops to -10 -> drawdown = 10
        assert mae == pytest.approx(10.0)
        # Trough at -10, rises to 5 -> runup = 15
        assert mfe == pytest.approx(15.0)

    def test_single_element(self):
        """Single element -> (0, 0)."""
        mae, mfe = compute_pnl_curve_excursions([5.0])
        assert mae == pytest.approx(0.0)
        assert mfe == pytest.approx(0.0)

    def test_empty_list(self):
        mae, mfe = compute_pnl_curve_excursions([])
        assert mae == pytest.approx(0.0)
        assert mfe == pytest.approx(0.0)

    def test_two_elements_up(self):
        mae, mfe = compute_pnl_curve_excursions([0.0, 10.0])
        assert mae == pytest.approx(0.0)
        assert mfe == pytest.approx(10.0)

    def test_two_elements_down(self):
        mae, mfe = compute_pnl_curve_excursions([10.0, 0.0])
        assert mae == pytest.approx(10.0)
        assert mfe == pytest.approx(0.0)

    def test_flat_curve(self):
        values = [5.0, 5.0, 5.0, 5.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        assert mae == pytest.approx(0.0)
        assert mfe == pytest.approx(0.0)

    def test_zigzag(self):
        """Multiple oscillations: find worst drawdown and best runup."""
        values = [0.0, 3.0, 1.0, 6.0, 2.0, 8.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        # Drawdowns: 3->1=2, 6->2=4 -> MAE = 4
        assert mae == pytest.approx(4.0)
        # Runups: 0->3=3, 1->6=5, 2->8=6 -> MFE = 6
        # But also: trough at 0, and value reaches 8 -> runup = 8
        assert mfe == pytest.approx(8.0)

    def test_negative_start(self):
        """Starting from negative: verify math with negative values."""
        values = [-5.0, -3.0, -8.0, -1.0]
        mae, mfe = compute_pnl_curve_excursions(values)
        # Peak at -3, drops to -8 -> drawdown = 5
        assert mae == pytest.approx(5.0)
        # Trough at -8, rises to -1 -> runup = 7
        assert mfe == pytest.approx(7.0)


# =============================================================================
# 5. compute_derived_fields() — MODE-AWARE grid spacing
# =============================================================================

class TestComputeDerivedFields:
    """Tests for ExtractedBotData.compute_derived_fields() mode-aware spacing."""

    def test_arithmetic_mode_explicit(self):
        """Arithmetic: (range/low*100)/(grids-1)."""
        bot = ExtractedBotData(
            mode="arithmetic",
            price_range_low=100.0,
            price_range_high=200.0,
            grids_count=11,
            end_time_utc=datetime(2026, 3, 1, 12, 0, 0),
            duration_hours=24.0,
        )
        bot.compute_derived_fields()
        # (200-100)/100*100 / (11-1) = 100/10 = 10.0
        assert bot.grid_spacing_pct == pytest.approx(10.0)

    def test_arithmetic_mode_none_default(self):
        """When mode is None, arithmetic formula is used (default)."""
        bot = ExtractedBotData(
            mode=None,
            price_range_low=100.0,
            price_range_high=200.0,
            grids_count=11,
        )
        bot.compute_derived_fields()
        assert bot.grid_spacing_pct == pytest.approx(10.0)

    def test_geometric_mode(self):
        """Geometric: ((high/low)^(1/(grids-1))-1)*100."""
        bot = ExtractedBotData(
            mode="geometric",
            price_range_low=100.0,
            price_range_high=200.0,
            grids_count=11,
        )
        bot.compute_derived_fields()
        expected = (math.pow(200.0 / 100.0, 1.0 / 10) - 1.0) * 100
        assert bot.grid_spacing_pct == pytest.approx(expected)

    def test_geometric_mode_small_range(self):
        """Geometric with a smaller range to verify precision."""
        bot = ExtractedBotData(
            mode="geometric",
            price_range_low=150.50,
            price_range_high=250.75,
            grids_count=50,
        )
        bot.compute_derived_fields()
        expected = (math.pow(250.75 / 150.50, 1.0 / 49) - 1.0) * 100
        assert bot.grid_spacing_pct == pytest.approx(expected, rel=1e-6)

    def test_start_time_computed(self):
        """start_time_utc = end_time_utc - duration_hours."""
        bot = ExtractedBotData(
            end_time_utc=datetime(2026, 3, 8, 14, 30, 0),
            duration_hours=48.0,
        )
        bot.compute_derived_fields()
        expected = datetime(2026, 3, 6, 14, 30, 0)
        assert bot.start_time_utc == expected

    def test_no_grid_spacing_when_insufficient_data(self):
        """No spacing computed when data is missing."""
        bot = ExtractedBotData(
            price_range_low=100.0,
            price_range_high=200.0,
            grids_count=None,
        )
        bot.compute_derived_fields()
        assert bot.grid_spacing_pct is None

    def test_no_grid_spacing_single_grid(self):
        """grids_count=1 -> grids-1=0 -> no spacing (division by zero guard)."""
        bot = ExtractedBotData(
            price_range_low=100.0,
            price_range_high=200.0,
            grids_count=1,
        )
        bot.compute_derived_fields()
        assert bot.grid_spacing_pct is None

    def test_no_start_time_without_end_or_duration(self):
        bot = ExtractedBotData(end_time_utc=datetime(2026, 3, 8, 14, 30, 0))
        bot.compute_derived_fields()
        assert bot.start_time_utc is None

    def test_arithmetic_vs_geometric_different(self):
        """For the same range, arithmetic and geometric produce different spacings."""
        low, high, grids = 100.0, 200.0, 21

        bot_arith = ExtractedBotData(
            mode="arithmetic", price_range_low=low, price_range_high=high, grids_count=grids,
        )
        bot_arith.compute_derived_fields()

        bot_geo = ExtractedBotData(
            mode="geometric", price_range_low=low, price_range_high=high, grids_count=grids,
        )
        bot_geo.compute_derived_fields()

        assert bot_arith.grid_spacing_pct is not None
        assert bot_geo.grid_spacing_pct is not None
        assert bot_arith.grid_spacing_pct != pytest.approx(bot_geo.grid_spacing_pct, rel=1e-3)


# =============================================================================
# 6. _compute_liquidation_features()
# =============================================================================

class TestComputeLiquidationFeatures:
    """Tests for liquidation distance feature computation."""

    def test_standard_case(self):
        """All values provided — verify every formula."""
        features = _compute_liquidation_features(
            liq_long=80.0,
            liq_short=300.0,
            grid_low=100.0,
            grid_high=250.0,
        )
        # dist_to_liq_long_pct = (grid_low - liq_long) / grid_low * 100
        assert features["dist_to_liq_long_pct"] == pytest.approx(
            (100.0 - 80.0) / 100.0 * 100, abs=1e-3
        )
        # dist_to_liq_short_pct = (liq_short - grid_high) / grid_high * 100
        assert features["dist_to_liq_short_pct"] == pytest.approx(
            (300.0 - 250.0) / 250.0 * 100, abs=1e-3
        )
        # liq_range_utilization = (grid_range / liq_range) * 100
        grid_range = 250.0 - 100.0
        liq_range = 300.0 - 80.0
        assert features["liq_range_utilization"] == pytest.approx(
            (grid_range / liq_range) * 100, abs=1e-3
        )
        # liq_asymmetry = dist_short / dist_long
        dist_long = (100.0 - 80.0) / 100.0 * 100
        dist_short = (300.0 - 250.0) / 250.0 * 100
        assert features["liq_asymmetry"] == pytest.approx(
            dist_short / dist_long, abs=1e-3
        )

    def test_uses_grid_bounds_not_trigger_price(self):
        """Features use grid_low/grid_high, NOT trigger_price.

        Calling with the same grid bounds but different hypothetical trigger
        prices would yield the same features (trigger_price is not a parameter).
        """
        feat1 = _compute_liquidation_features(80.0, 300.0, 100.0, 250.0)
        feat2 = _compute_liquidation_features(80.0, 300.0, 100.0, 250.0)
        assert feat1 == feat2

    def test_all_none(self):
        features = _compute_liquidation_features(None, None, None, None)
        assert features["liq_price_long"] is None
        assert features["liq_price_short"] is None
        assert features["dist_to_liq_long_pct"] is None
        assert features["dist_to_liq_short_pct"] is None
        assert features["liq_range_utilization"] is None
        assert features["liq_asymmetry"] is None

    def test_only_liq_long(self):
        features = _compute_liquidation_features(
            liq_long=80.0, liq_short=None, grid_low=100.0, grid_high=250.0,
        )
        assert features["dist_to_liq_long_pct"] == pytest.approx(20.0, abs=1e-3)
        assert features["dist_to_liq_short_pct"] is None
        assert features["liq_range_utilization"] is None
        assert features["liq_asymmetry"] is None

    def test_only_liq_short(self):
        features = _compute_liquidation_features(
            liq_long=None, liq_short=300.0, grid_low=100.0, grid_high=250.0,
        )
        assert features["dist_to_liq_long_pct"] is None
        assert features["dist_to_liq_short_pct"] == pytest.approx(20.0, abs=1e-3)
        assert features["liq_range_utilization"] is None
        assert features["liq_asymmetry"] is None

    def test_grid_low_zero(self):
        """grid_low=0 -> guard prevents division by zero."""
        features = _compute_liquidation_features(
            liq_long=0.0, liq_short=100.0, grid_low=0.0, grid_high=50.0,
        )
        assert features["dist_to_liq_long_pct"] is None

    def test_grid_high_zero(self):
        """grid_high=0 -> guard prevents division by zero."""
        features = _compute_liquidation_features(
            liq_long=0.0, liq_short=100.0, grid_low=10.0, grid_high=0.0,
        )
        assert features["dist_to_liq_short_pct"] is None

    def test_liq_range_utilization_inverted_liq(self):
        """If liq_short <= liq_long, no liq_range_utilization."""
        features = _compute_liquidation_features(
            liq_long=300.0, liq_short=80.0, grid_low=100.0, grid_high=250.0,
        )
        assert features["liq_range_utilization"] is None

    def test_asymmetry_zero_dist_long(self):
        """If dist_long == 0, asymmetry is None (division guard)."""
        # liq_long == grid_low -> dist_long = 0
        features = _compute_liquidation_features(
            liq_long=100.0, liq_short=300.0, grid_low=100.0, grid_high=250.0,
        )
        assert features["dist_to_liq_long_pct"] == pytest.approx(0.0)
        assert features["liq_asymmetry"] is None

    def test_symmetric_case(self):
        """Equal distances -> asymmetry ~1.0."""
        # grid_low=100, grid_high=200
        # liq_long=80 -> dist_long = 20%
        # liq_short=240 -> dist_short = (240-200)/200*100 = 20%
        features = _compute_liquidation_features(
            liq_long=80.0, liq_short=240.0, grid_low=100.0, grid_high=200.0,
        )
        assert features["liq_asymmetry"] == pytest.approx(1.0, abs=1e-3)

    def test_rounding(self):
        """Values should be rounded to 4 decimal places."""
        features = _compute_liquidation_features(
            liq_long=79.12345,
            liq_short=301.98765,
            grid_low=100.0,
            grid_high=250.0,
        )
        # Check rounding to 4 decimals
        dist_long_raw = (100.0 - 79.12345) / 100.0 * 100
        assert features["dist_to_liq_long_pct"] == round(dist_long_raw, 4)


# =============================================================================
# 7. search_candidate_match()
# =============================================================================

class TestSearchCandidateMatch:
    """Tests for CSV candidate searching with mock data."""

    def test_exact_match(self, workspace_tmp_dir):
        """Exact match on all criteria returns candidate_id."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "grid_lower": 150.50,
            "grid_upper": 250.75,
            "num_grids": 50,
            "candidate_id": "TAO_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result == "TAO_001"

    def test_no_match_wrong_symbol(self, workspace_tmp_dir):
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "BTCUSDT",
            "grid_lower": 150.50,
            "grid_upper": 250.75,
            "num_grids": 50,
            "candidate_id": "BTC_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result is None

    def test_no_match_wrong_grids(self, workspace_tmp_dir):
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "grid_lower": 150.50,
            "grid_upper": 250.75,
            "num_grids": 100,
            "candidate_id": "TAO_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result is None

    def test_within_tolerance(self, workspace_tmp_dir):
        """Price within tolerance_pct (default 1%) still matches."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "grid_lower": 151.00,  # ~0.33% off from 150.50
            "grid_upper": 251.00,  # ~0.10% off from 250.75
            "num_grids": 50,
            "candidate_id": "TAO_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result == "TAO_001"

    def test_outside_tolerance(self, workspace_tmp_dir):
        """Price difference > tolerance_pct -> no match."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "grid_lower": 160.00,  # ~6.3% off from 150.50 -> exceeds 1%
            "grid_upper": 250.75,
            "num_grids": 50,
            "candidate_id": "TAO_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result is None

    def test_custom_tolerance(self, workspace_tmp_dir):
        """Custom tolerance_pct=5% allows wider match."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "grid_lower": 155.00,  # ~3% off from 150.50
            "grid_upper": 260.00,  # ~3.7% off from 250.75
            "num_grids": 50,
            "candidate_id": "TAO_001",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir, tolerance_pct=5.0,
        )
        assert result == "TAO_001"

    def test_best_match_selected(self, workspace_tmp_dir):
        """When multiple candidates match, the closest (lowest score) wins."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([
            {
                "symbol": "TAOUSDT",
                "grid_lower": 151.00,
                "grid_upper": 251.00,
                "num_grids": 50,
                "candidate_id": "TAO_FAR",
            },
            {
                "symbol": "TAOUSDT",
                "grid_lower": 150.55,
                "grid_upper": 250.80,
                "num_grids": 50,
                "candidate_id": "TAO_CLOSE",
            },
        ])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result == "TAO_CLOSE"

    def test_missing_results_dir(self, workspace_tmp_dir):
        nonexistent = workspace_tmp_dir / "nonexistent"
        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=nonexistent,
        )
        assert result is None

    def test_empty_results_dir(self, workspace_tmp_dir):
        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result is None

    def test_alternative_column_names(self, workspace_tmp_dir):
        """Uses price_range_low / price_range_high / grids_count column names."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "TAOUSDT",
            "price_range_low": 150.50,
            "price_range_high": 250.75,
            "grids_count": 50,
            "candidate_id": "TAO_ALT",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result == "TAO_ALT"

    def test_symbol_case_insensitive(self, workspace_tmp_dir):
        """Symbol matching is case-insensitive."""
        csv_file = workspace_tmp_dir / "deployment_ready_2026-03-08.csv"
        df = pd.DataFrame([{
            "symbol": "taousdt",
            "grid_lower": 150.50,
            "grid_upper": 250.75,
            "num_grids": 50,
            "candidate_id": "TAO_LOWER",
        }])
        df.to_csv(csv_file, index=False)

        result = search_candidate_match(
            "TAOUSDT", 150.50, 250.75, 50, results_dir=workspace_tmp_dir,
        )
        assert result == "TAO_LOWER"


# =============================================================================
# 8. validate_data_coherence()
# =============================================================================

class TestValidateDataCoherence:
    """Tests for cross-validation of data consistency."""

    def test_no_warnings_coherent_data(self):
        """Perfectly coherent data produces no warnings."""
        bot = ExtractedBotData(
            invested_margin_usdt=500.0,
            total_profit_usdt=12.50,
            pnl_pct=2.50,  # 12.50/500 * 100 = 2.50
            realized_pnl_usdt=15.80,
            unrealized_pnl_usdt=-4.30,
            funding_fee_usdt=1.00,  # 15.80 + (-4.30) + 1.00 = 12.50
            grid_spacing_pct=2.0,
            liq_price_long=80.0,
            liq_price_short=300.0,
            price_range_low=100.0,
            price_range_high=250.0,
        )
        metrics = TradeMetrics()
        warnings = validate_data_coherence(bot, metrics)
        assert len(warnings) == 0

    def test_pnl_pct_mismatch(self):
        """Reported PnL% differs from computed by >1%."""
        bot = ExtractedBotData(
            invested_margin_usdt=500.0,
            total_profit_usdt=12.50,
            pnl_pct=5.00,  # Should be 2.50%, diff = 2.50% > 1.0%
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("PnL% mismatch" in w for w in warnings)

    def test_pnl_pct_within_tolerance(self):
        """Small PnL% difference (<= 1%) does NOT trigger warning."""
        bot = ExtractedBotData(
            invested_margin_usdt=500.0,
            total_profit_usdt=12.50,
            pnl_pct=2.80,  # computed=2.50, diff=0.30 < 1.0
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        pnl_warnings = [w for w in warnings if "PnL% mismatch" in w]
        assert len(pnl_warnings) == 0

    def test_profit_decomposition_mismatch(self):
        """total_profit != matched + unmatched + funding by >0.5."""
        bot = ExtractedBotData(
            total_profit_usdt=12.50,
            realized_pnl_usdt=10.00,
            unrealized_pnl_usdt=-1.00,
            funding_fee_usdt=0.50,
            # Expected: 10 + (-1) + 0.5 = 9.50, diff = |9.50 - 12.50| = 3.0 > 0.5
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("decomposition mismatch" in w for w in warnings)

    def test_profit_decomposition_ok(self):
        """total_profit == matched + unmatched + funding within 0.5."""
        bot = ExtractedBotData(
            total_profit_usdt=12.50,
            realized_pnl_usdt=15.80,
            unrealized_pnl_usdt=-4.30,
            funding_fee_usdt=1.00,  # 15.80 + (-4.30) + 1.00 = 12.50
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        decomp_warnings = [w for w in warnings if "decomposition mismatch" in w]
        assert len(decomp_warnings) == 0

    def test_fee_mismatch_from_fills(self):
        """Commission from fills vs trade_metrics differs by >0.01."""
        fills = [
            ParsedTradeFill(side="BUY", price=100.0, quantity=1.0, fee=0.10),
            ParsedTradeFill(side="SELL", price=110.0, quantity=1.0, fee=0.12),
        ]
        metrics = TradeMetrics(commission_usdt=0.50)  # fills total = 0.22
        bot = ExtractedBotData()
        warnings = validate_data_coherence(bot, metrics, fills)
        assert any("Fee mismatch" in w for w in warnings)

    def test_fee_match_from_fills(self):
        """Commission from fills matches trade_metrics."""
        fills = [
            ParsedTradeFill(side="BUY", price=100.0, quantity=1.0, fee=0.10),
            ParsedTradeFill(side="SELL", price=110.0, quantity=1.0, fee=0.12),
        ]
        metrics = TradeMetrics(commission_usdt=0.22)
        bot = ExtractedBotData()
        warnings = validate_data_coherence(bot, metrics, fills)
        fee_warnings = [w for w in warnings if "Fee mismatch" in w]
        assert len(fee_warnings) == 0

    def test_grid_spacing_non_positive(self):
        bot = ExtractedBotData(grid_spacing_pct=-0.5)
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("non-positive" in w for w in warnings)

    def test_grid_spacing_too_large(self):
        bot = ExtractedBotData(grid_spacing_pct=25.0)
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("unusually large" in w for w in warnings)

    def test_grid_spacing_normal_ok(self):
        bot = ExtractedBotData(grid_spacing_pct=1.5)
        warnings = validate_data_coherence(bot, TradeMetrics())
        spacing_warnings = [w for w in warnings if "spacing" in w.lower()]
        assert len(spacing_warnings) == 0

    def test_liq_long_above_grid_low(self):
        """Liq long >= grid low is a warning (should be below)."""
        bot = ExtractedBotData(
            liq_price_long=110.0,
            price_range_low=100.0,
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("Liq long" in w and ">=" in w for w in warnings)

    def test_liq_long_below_grid_low_ok(self):
        bot = ExtractedBotData(
            liq_price_long=80.0,
            price_range_low=100.0,
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        liq_long_warnings = [w for w in warnings if "Liq long" in w]
        assert len(liq_long_warnings) == 0

    def test_liq_short_below_grid_high(self):
        """Liq short <= grid high is a warning (should be above)."""
        bot = ExtractedBotData(
            liq_price_short=200.0,
            price_range_high=250.0,
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert any("Liq short" in w and "<=" in w for w in warnings)

    def test_liq_short_above_grid_high_ok(self):
        bot = ExtractedBotData(
            liq_price_short=300.0,
            price_range_high=250.0,
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        liq_short_warnings = [w for w in warnings if "Liq short" in w]
        assert len(liq_short_warnings) == 0

    def test_multiple_warnings_at_once(self):
        """Multiple issues produce multiple warnings."""
        bot = ExtractedBotData(
            invested_margin_usdt=500.0,
            total_profit_usdt=12.50,
            pnl_pct=10.0,  # mismatch
            grid_spacing_pct=-1.0,  # negative
            liq_price_long=110.0,  # above grid low
            price_range_low=100.0,
        )
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert len(warnings) >= 3

    def test_all_none_no_crash(self):
        """All-None bot data should not crash."""
        bot = ExtractedBotData()
        warnings = validate_data_coherence(bot, TradeMetrics())
        assert isinstance(warnings, list)


# =============================================================================
# 9. text_fills_to_dataframe()
# =============================================================================

class TestTextFillsToDataFrame:
    """Tests for converting ParsedTradeFill list to DataFrame."""

    def test_empty_fills_returns_empty_df(self):
        df = text_fills_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_single_fill(self):
        fills = [ParsedTradeFill(side="BUY", price=100.0, quantity=1.0, fee=0.05)]
        df = text_fills_to_dataframe(fills)
        assert len(df) == 1
        assert df.iloc[0]["Side"] == "Buy"
        assert df.iloc[0]["Price"] == "100.0 USDT"
        assert df.iloc[0]["Quantity"] == "1.0"
        assert df.iloc[0]["Fee"] == "0.05 USDT"
        assert df.iloc[0]["Realized Profit"] == "0.0 USDT"
        assert df.iloc[0]["Maker"] == "TRUE"

    def test_sell_side_capitalization(self):
        fills = [ParsedTradeFill(side="SELL", price=200.0, quantity=0.5)]
        df = text_fills_to_dataframe(fills)
        assert df.iloc[0]["Side"] == "Sell"

    def test_taker_fill(self):
        fills = [ParsedTradeFill(side="BUY", price=100.0, quantity=1.0, maker=False)]
        df = text_fills_to_dataframe(fills)
        assert df.iloc[0]["Maker"] == "FALSE"

    def test_multiple_fills_timestamps(self):
        """Multiple fills get incrementing timestamps."""
        start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        fills = [
            ParsedTradeFill(side="BUY", price=100.0, quantity=1.0),
            ParsedTradeFill(side="SELL", price=110.0, quantity=1.0),
            ParsedTradeFill(side="BUY", price=105.0, quantity=2.0),
        ]
        df = text_fills_to_dataframe(fills, start_time=start)
        assert len(df) == 3
        # Timestamps should be start, start+1m, start+2m
        assert df.iloc[0]["parsed_time"] == start
        assert df.iloc[1]["parsed_time"] == start + timedelta(minutes=1)
        assert df.iloc[2]["parsed_time"] == start + timedelta(minutes=2)

    def test_custom_timestamp_on_fill(self):
        """Explicit timestamp on ParsedTradeFill overrides generated one."""
        custom_ts = datetime(2026, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
        fills = [
            ParsedTradeFill(side="BUY", price=100.0, quantity=1.0, timestamp=custom_ts),
        ]
        df = text_fills_to_dataframe(fills, start_time=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc))
        assert df.iloc[0]["parsed_time"] == custom_ts

    def test_dataframe_columns(self):
        """Verify all expected columns are present."""
        fills = [ParsedTradeFill(side="BUY", price=100.0, quantity=1.0)]
        df = text_fills_to_dataframe(fills)
        expected_cols = {"Side", "Price", "Quantity", "Fee", "Realized Profit", "Maker", "parsed_time"}
        assert set(df.columns) == expected_cols

    def test_realized_profit_preserved(self):
        fills = [
            ParsedTradeFill(side="SELL", price=200.0, quantity=1.0, realized_profit=10.50),
        ]
        df = text_fills_to_dataframe(fills)
        assert df.iloc[0]["Realized Profit"] == "10.5 USDT"


# =============================================================================
# 10. build_bot_data_from_text()
# =============================================================================

class TestBuildBotDataFromText:
    """End-to-end tests for text parsing into structured data."""

    def test_full_parse(self, full_bot_text):
        bot_data, fills, pnl_curve = build_bot_data_from_text(full_bot_text)
        assert bot_data.symbol == "TAOUSDT"
        assert bot_data.strategy_id == 1234567890
        assert bot_data.status == "expired"
        assert bot_data.mode == "arithmetic"
        assert bot_data.leverage == 10
        assert bot_data.grids_count == 50
        assert bot_data.price_range_low == pytest.approx(150.50)
        assert bot_data.price_range_high == pytest.approx(250.75)
        # Derived fields should be computed
        assert bot_data.grid_spacing_pct is not None
        assert bot_data.start_time_utc is not None

    def test_derived_fields_computed(self, full_bot_text):
        bot_data, _, _ = build_bot_data_from_text(full_bot_text)
        # Arithmetic spacing: ((250.75-150.50)/150.50*100)/(50-1)
        expected_spacing = ((250.75 - 150.50) / 150.50 * 100) / 49
        assert bot_data.grid_spacing_pct == pytest.approx(expected_spacing, rel=1e-4)

    def test_fills_parsed_from_combined_text(self):
        text = (
            "ETHUSDT Perpetual\n"
            "Grids: 20\n"
            "Buy 2000.00 0.01\n"
            "Sell 2050.00 0.01\n"
        )
        bot_data, fills, pnl_curve = build_bot_data_from_text(text)
        assert bot_data.symbol == "ETHUSDT"
        assert len(fills) == 2
        assert fills[0].side == "BUY"
        assert fills[1].side == "SELL"

    def test_pnl_curve_parsed_from_combined_text(self):
        text = (
            "BTCUSDT Perpetual\n"
            "Grids: 10\n"
            "1  +0.50 USDT\n"
            "2  1.00 USDT\n"
            "3  -0.20 USDT\n"
        )
        bot_data, fills, pnl_curve = build_bot_data_from_text(text)
        assert len(pnl_curve) == 3
        assert pnl_curve[0] == pytest.approx(0.50)
        assert pnl_curve[1] == pytest.approx(1.00)
        assert pnl_curve[2] == pytest.approx(-0.20)

    def test_empty_text(self):
        bot_data, fills, pnl_curve = build_bot_data_from_text("")
        assert bot_data.symbol is None
        assert fills == []
        assert pnl_curve == []

    def test_returns_tuple_of_three(self, full_bot_text):
        result = build_bot_data_from_text(full_bot_text)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], ExtractedBotData)
        assert isinstance(result[1], list)
        assert isinstance(result[2], list)

    def test_geometric_mode_spacing(self):
        text = (
            "SOLUSDT Perpetual\n"
            "Geometric\n"
            "Price Range: 100.00 - 200.00\n"
            "Number of Grids: 21\n"
        )
        bot_data, _, _ = build_bot_data_from_text(text)
        expected = (math.pow(200.0 / 100.0, 1.0 / 20) - 1.0) * 100
        assert bot_data.mode == "geometric"
        assert bot_data.grid_spacing_pct == pytest.approx(expected, rel=1e-6)

    def test_liq_prices_propagated(self, full_bot_text):
        bot_data, _, _ = build_bot_data_from_text(full_bot_text)
        assert bot_data.liq_price_long == pytest.approx(120.50)
        assert bot_data.liq_price_short == pytest.approx(290.00)
        assert bot_data.trigger_price == pytest.approx(200.25)
