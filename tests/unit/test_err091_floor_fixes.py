"""ERR-091 regression tests: profit-floor volatility input scale and the
guaranteed-fail (floor > spacing-cap ceiling) detection.

Covers:
1. _horizon_realized_vol_pct returns bot-horizon realized vol on the floor's
   design scale (~2%), not the half-range-width proxy (14-17).
2. GridCalculator.generate_params reports profit_floor_exceeds_spacing_cap
   when the demanded floor exceeds the max_spacing_pct-implied profit ceiling
   (previously a silent clamp at calculate_grid_spacing followed by a generic
   profit_per_grid_below_min rejection).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from neutralgrid.core.config import get_config
from neutralgrid.grid.calculator import GridCalculator
from neutralgrid.scanner.enrich_grid_params import _horizon_realized_vol_pct


def _make_klines(closes: list[float]) -> list[list]:
    """Minimal kline rows: only index 4 (close) is consumed by the helper."""
    return [[i, "0", "0", "0", str(c), "0"] for i, c in enumerate(closes)]


class TestHorizonRealizedVolErr091:
    def test_known_sigma_recovers_design_scale(self):
        # Alternating +/-0.3% log returns -> per-bar sigma exactly 0.003.
        n = 200
        rets = np.array([0.003 if i % 2 == 0 else -0.003 for i in range(n)])
        closes = 100.0 * np.exp(np.cumsum(rets))
        vol = _horizon_realized_vol_pct(_make_klines(list(closes)), horizon_bars=24)
        assert vol is not None
        expected = 0.003 * math.sqrt(24) * 100.0  # 1.4697%
        assert vol == pytest.approx(expected, rel=1e-6)
        # Design scale: single-digit percent, nowhere near the 14-17 range
        # the half-range proxy produced for wide-range symbols.
        assert vol < 5.0

    def test_insufficient_bars_returns_none(self):
        assert _horizon_realized_vol_pct(_make_klines([100.0] * 10), 24) is None
        assert _horizon_realized_vol_pct(None, 24) is None
        assert _horizon_realized_vol_pct([], 24) is None

    def test_flat_series_returns_none(self):
        # Zero variance -> sigma 0 -> None (caller falls back to 2.0).
        assert _horizon_realized_vol_pct(_make_klines([100.0] * 50), 24) is None

    def test_garbage_rows_return_none(self):
        assert _horizon_realized_vol_pct([["x"], ["y"]] * 30, 24) is None


def _vres(range_low: float, range_high: float) -> SimpleNamespace:
    mid = (range_low + range_high) / 2.0
    return SimpleNamespace(
        symbol="ERR091USDT",
        is_valid=True,
        range_high=range_high,
        range_low=range_low,
        current_price=mid,
        atr_1m=mid * 0.001,
    )


class TestFloorCeilingReconciliationErr091:
    def test_floor_above_ceiling_reports_guaranteed_fail(self):
        """A floor of 2.0% cannot be satisfied under max_spacing_pct=1.15%
        (ceiling ~1.11%); the rejection must name the structural cause."""
        calc = GridCalculator()
        params = calc.generate_params(
            _vres(80.0, 100.0),  # wide 25% range
            min_profit_pct=2.0,
        )
        assert params.is_valid is False
        assert params.reason is not None
        assert params.reason.startswith("profit_floor_exceeds_spacing_cap")
        assert "ceiling=" in params.reason and "max_spacing=" in params.reason

    def test_ceiling_value_matches_formula(self):
        """The reported ceiling equals (1-c)*(1+cap) - 1 - c in percent."""
        calc = GridCalculator()
        params = calc.generate_params(_vres(80.0, 100.0), min_profit_pct=2.0)
        cfg = get_config()
        close_fee = (
            cfg.grid.maker_fee
            if str(getattr(cfg.grid, "close_fee_mode", "maker")).lower() == "maker"
            else cfg.grid.taker_fee
        )
        c = (float(cfg.grid.maker_fee) + float(close_fee)) / 2.0
        ceiling = ((1.0 - c) * (1.0 + float(cfg.grid.max_spacing_pct)) - 1.0 - c) * 100.0
        assert params.reason is not None
        assert f"ceiling={ceiling:.4f}%" in params.reason

    def test_satisfiable_floor_still_produces_valid_grid(self):
        """A design-scale floor (0.30%) on the same geometry must pass —
        proving the new branch does not reject feasible configurations."""
        calc = GridCalculator()
        params = calc.generate_params(_vres(80.0, 100.0), min_profit_pct=0.30)
        assert params.is_valid is True, params.reason
        assert params.profit_per_grid_pct is not None
        assert params.profit_per_grid_pct >= 0.30
