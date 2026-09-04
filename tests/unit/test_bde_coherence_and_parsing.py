"""Unit tests for new_bot_data_extractor audit findings:
  Finding 1: coherence auto-repair + enforcement
  Finding 2: PnL-curve fallback with mtm_used_mark_prices
  Finding 3: parse_screenshot_text case-insensitive + missing fields
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from new_bot_data_extractor import (
    validate_data_coherence,
    auto_repair_coherence,
    parse_screenshot_text,
    TradeMetrics,
    ExtractedBotData,
)


# ═══════════════════════════════════════════════════════════════════════
# FINDING 1: Coherence auto-repair + enforcement
# ═══════════════════════════════════════════════════════════════════════

class TestAutoRepairCoherence:
    """auto_repair_coherence correctly classifies and repairs issues."""

    def _make_bot(self, **overrides) -> ExtractedBotData:
        defaults = dict(
            symbol="TESTUSDT",
            strategy_id=123456789,
            status="expired",
            total_profit_usdt=10.0,
            invested_margin_usdt=100.0,
            pnl_pct=10.0,
            realized_pnl_usdt=8.0,
            unrealized_pnl_usdt=1.5,
            funding_fee_usdt=0.5,
            price_range_low=90.0,
            price_range_high=110.0,
            grid_spacing_pct=2.0,
            liq_price_long=80.0,
            liq_price_short=120.0,
            duration_hours=24.0,
        )
        defaults.update(overrides)
        return ExtractedBotData(**defaults)

    def test_pnl_pct_mismatch_is_repaired(self):
        """PnL% mismatch → auto_repair recomputes pnl_pct from total/margin."""
        bot = self._make_bot(pnl_pct=15.0)  # Wrong: should be 10.0
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        assert any("PnL% mismatch" in w for w in warnings)

        repaired, unresolved = auto_repair_coherence(bot, tm, warnings)
        assert any("PnL% recomputed" in r for r in repaired)
        assert bot.pnl_pct == 10.0  # Repaired in place

    def test_profit_decomposition_within_tolerance(self):
        """Profit decomposition diff <= 5.0 USDT → acknowledged, not unresolved."""
        # total=10.0, sum=8.0+1.5+0.5=10.0 → no mismatch
        # Force a mismatch: total=10, but components sum to 12
        bot = self._make_bot(realized_pnl_usdt=10.0, unrealized_pnl_usdt=1.5, funding_fee_usdt=0.5)
        # sum = 12.0, total=10.0, diff=2.0 < 5.0
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        assert any("Profit decomposition mismatch" in w for w in warnings)

        repaired, unresolved = auto_repair_coherence(bot, tm, warnings)
        assert any("tolerance" in r for r in repaired)
        assert not any("Profit decomposition" in u for u in unresolved)

    def test_profit_decomposition_large_diff_unresolved(self):
        """Profit decomposition diff > 5.0 USDT → unresolved."""
        bot = self._make_bot(realized_pnl_usdt=50.0, unrealized_pnl_usdt=10.0, funding_fee_usdt=0.0)
        # sum = 60.0, total=10.0, diff=50.0 > 5.0
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        assert any("Profit decomposition mismatch" in w for w in warnings)

        repaired, unresolved = auto_repair_coherence(bot, tm, warnings)
        assert any("Profit decomposition" in u for u in unresolved)

    def test_grid_spacing_always_unresolved(self):
        """Grid spacing issues cannot be auto-repaired."""
        bot = self._make_bot(grid_spacing_pct=-1.0)
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        assert any("Grid spacing" in w for w in warnings)

        repaired, unresolved = auto_repair_coherence(bot, tm, warnings)
        assert any("Grid spacing" in u for u in unresolved)

    def test_liquidation_sanity_always_unresolved(self):
        """Liq prices crossing grid bounds → unresolved."""
        bot = self._make_bot(liq_price_long=95.0)  # > grid_low=90
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        assert any("Liq long" in w for w in warnings)

        repaired, unresolved = auto_repair_coherence(bot, tm, warnings)
        assert any("Liq long" in u for u in unresolved)

    def test_no_warnings_returns_empty(self):
        """No warnings → both lists empty."""
        bot = self._make_bot()
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        # With default values, sum = 8+1.5+0.5 = 10 = total. No mismatch.
        assert len(warnings) == 0

    def test_holding_time_ok_not_in_repaired(self):
        """holding_time_ok is not artificially added to repaired list."""
        bot = self._make_bot(pnl_pct=15.0)  # Force a warning
        tm = TradeMetrics()
        warnings = validate_data_coherence(bot, tm)
        repaired, _ = auto_repair_coherence(bot, tm, warnings)
        assert not any("holding_time_ok" in r for r in repaired)


# ═══════════════════════════════════════════════════════════════════════
# FINDING 2: PnL-curve fallback with mtm_used_mark_prices
# ═══════════════════════════════════════════════════════════════════════

class TestMtmUsedMarkPricesFlag:
    """TradeMetrics.mtm_used_mark_prices defaults and threading."""

    def test_default_is_false(self):
        tm = TradeMetrics()
        assert tm.mtm_used_mark_prices is False

    def test_flag_can_be_set_true(self):
        tm = TradeMetrics(mtm_used_mark_prices=True)
        assert tm.mtm_used_mark_prices is True


# ═══════════════════════════════════════════════════════════════════════
# FINDING 3: parse_screenshot_text fixes
# ═══════════════════════════════════════════════════════════════════════

class TestParseScreenshotTextCaseInsensitive:
    """Status and mode detection is now case-insensitive."""

    @pytest.mark.parametrize("text,expected_status", [
        ("Bot EXPIRED on 2026-01-01", "expired"),
        ("Status: expired", "expired"),
        ("CANCELLED by user", "cancelled"),
        ("canceled early", "cancelled"),
        ("Bot ended at 12:00", "ended"),
        ("ENDED on 2026-03-10", "ended"),
    ])
    def test_status_case_insensitive(self, text, expected_status):
        result = parse_screenshot_text(text)
        assert result.get("status") == expected_status

    @pytest.mark.parametrize("text,expected_mode", [
        ("Mode: ARITHMETIC", "arithmetic"),
        ("arithmetic mode", "arithmetic"),
        ("GEOMETRIC spacing", "geometric"),
        ("Geometric", "geometric"),
    ])
    def test_mode_case_insensitive(self, text, expected_mode):
        result = parse_screenshot_text(text)
        assert result.get("mode") == expected_mode


class TestParseScreenshotTextNewFields:
    """Trigger price and liquidation prices are now parsed."""

    def test_trigger_price(self):
        text = "Trigger Price: 1,234.56"
        result = parse_screenshot_text(text)
        assert result["trigger_price"] == pytest.approx(1234.56)

    def test_trigger_price_no_comma(self):
        text = "Trigger Price: 0.5678"
        result = parse_screenshot_text(text)
        assert result["trigger_price"] == pytest.approx(0.5678)

    def test_liq_price_long(self):
        text = "Liquidation Price (Long): 85.50"
        result = parse_screenshot_text(text)
        assert result["liq_price_long"] == pytest.approx(85.50)

    def test_liq_price_short(self):
        text = "Liq Price (Short): 120.75"
        result = parse_screenshot_text(text)
        assert result["liq_price_short"] == pytest.approx(120.75)

    def test_liq_long_short_form(self):
        """Short form 'Liq Long' without 'Price' keyword."""
        text = "Liq Long: 80.00"
        result = parse_screenshot_text(text)
        assert result["liq_price_long"] == pytest.approx(80.00)

    def test_liq_short_short_form(self):
        """Short form 'Liq Short' without 'Price' keyword."""
        text = "Liq Short: 130.00"
        result = parse_screenshot_text(text)
        assert result["liq_price_short"] == pytest.approx(130.00)

    def test_liq_with_commas(self):
        text = "Liquidation Price (Long): 1,234.56\nLiq Price (Short): 2,345.67"
        result = parse_screenshot_text(text)
        assert result["liq_price_long"] == pytest.approx(1234.56)
        assert result["liq_price_short"] == pytest.approx(2345.67)

    def test_all_three_fields_together(self):
        text = (
            "TESTUSDT Expired\n"
            "Trigger Price: 100.50\n"
            "Liquidation Price (Long): 85.00\n"
            "Liquidation Price (Short): 115.00\n"
        )
        result = parse_screenshot_text(text)
        assert result["trigger_price"] == pytest.approx(100.50)
        assert result["liq_price_long"] == pytest.approx(85.00)
        assert result["liq_price_short"] == pytest.approx(115.00)
        assert result["status"] == "expired"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
