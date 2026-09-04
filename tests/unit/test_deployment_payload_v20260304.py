"""Tests for deployment payload sizing enforcement helpers."""

from __future__ import annotations

from neutralgrid.core.config import get_config
from neutralgrid.live.deployment_payload_v20260304 import (
    build_binance_grid_payload,
    resolve_deployment_sizing,
)


def test_resolve_deployment_sizing_enforces_fraction():
    base = float(get_config().grid.capital)
    sizing = resolve_deployment_sizing({"capital_fraction": 0.30})
    assert sizing.capital_base_usdt == base
    assert sizing.capital_fraction == 0.30
    assert sizing.effective_margin_usdt == base * 0.30


def test_resolve_deployment_sizing_clamps_out_of_range():
    base = float(get_config().grid.capital)
    high = resolve_deployment_sizing({"capital_fraction": 3.0})
    low = resolve_deployment_sizing({"capital_fraction": -1.0})
    assert high.capital_fraction == 1.0
    assert high.effective_margin_usdt == base
    assert low.capital_fraction == 0.0
    assert low.effective_margin_usdt == 0.0


def test_build_binance_grid_payload_uses_effective_margin():
    payload = build_binance_grid_payload(
        {
            "candidate_id": "ETHUSDT_20260304_120000_beefcafe",
            "symbol": "ETHUSDT",
            "grid_lower": 2000.0,
            "grid_upper": 2200.0,
            "num_grids": 20,
            "leverage": 10,
            "capital_fraction": 0.50,
        },
        capital_base_usdt=800.0,
    )
    assert payload["investment_margin_usdt"] == 400.0
    assert payload["capital_fraction"] == 0.50
    assert payload["capital_base_usdt"] == 800.0

