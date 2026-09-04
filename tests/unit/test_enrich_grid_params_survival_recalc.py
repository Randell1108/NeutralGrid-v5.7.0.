from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from neutralgrid.scanner.enrich_grid_params import (
    _RegimeData,
    _apply_ou_params_from_scan_row,
    _grid_bounds_range_size_pct,
    _recalculate_survival_prob_from_grid,
)


def _make_regime_data(*, current_price: float = 100.0, survival_prob: float = 0.41) -> _RegimeData:
    vres = SimpleNamespace(current_price=current_price)
    return _RegimeData(
        market_data={},
        vres=vres,
        survival_prob=survival_prob,
    )


def test_apply_ou_params_from_scan_row_copies_present_values() -> None:
    rd = _make_regime_data()
    scan_row = pd.Series(
        {
            "ou_theta": 0.123456,
            "ou_mu": 10.5,
            "ou_sigma": 0.009876,
        }
    )

    _apply_ou_params_from_scan_row(rd, scan_row)

    assert rd.ou_theta == pytest.approx(0.123456)
    assert rd.ou_mu == pytest.approx(10.5)
    assert rd.ou_sigma == pytest.approx(0.009876)


def test_grid_bounds_recalc_uses_generated_bounds_before_stage_b() -> None:
    rd = _make_regime_data(current_price=100.0, survival_prob=0.22)
    rd.ou_theta = 0.18
    rd.ou_mu = 4.45
    rd.ou_sigma = 0.03
    grid = SimpleNamespace(grid_lower=95.0, grid_upper=105.0)

    cfg = SimpleNamespace(
        stochastic=SimpleNamespace(
            survival_horizon_bars=24,
            survival_mc_paths=1234,
        )
    )
    checker = SimpleNamespace()

    def _compute_survival_probability(**kwargs):
        checker.kwargs = kwargs
        return 0.7312

    checker.compute_survival_probability = _compute_survival_probability

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg):
        with patch(
            "neutralgrid.validation.stochastic.StochasticRegimeChecker",
            return_value=checker,
        ):
            recalculated = _recalculate_survival_prob_from_grid(rd, grid, sym="BTCUSDT")

    assert _grid_bounds_range_size_pct(grid.grid_lower, grid.grid_upper) == pytest.approx(10.0)
    assert recalculated == pytest.approx(0.7312)
    assert checker.kwargs["ou_theta"] == pytest.approx(0.18)
    assert checker.kwargs["ou_mu"] == pytest.approx(4.45)
    assert checker.kwargs["ou_sigma"] == pytest.approx(0.03)
    assert checker.kwargs["current_log_price"] == pytest.approx(4.605170185988092)
    assert checker.kwargs["range_low_log"] == pytest.approx(4.553876891600541)
    assert checker.kwargs["range_high_log"] == pytest.approx(4.653960350157523)
    assert checker.kwargs["horizon"] == 24
    assert checker.kwargs["n_paths"] == 1234


def test_grid_bounds_recalc_fails_closed_and_preserves_original_value() -> None:
    rd = _make_regime_data(current_price=100.0, survival_prob=0.41)
    rd.ou_theta = 0.18
    rd.ou_mu = 4.45
    rd.ou_sigma = 0.03
    grid = SimpleNamespace(grid_lower=95.0, grid_upper=105.0)
    original = rd.survival_prob

    cfg = SimpleNamespace(
        stochastic=SimpleNamespace(
            survival_horizon_bars=24,
            survival_mc_paths=1234,
        )
    )

    class _ExplodingChecker:
        def __init__(self, *_args, **_kwargs):
            pass

        def compute_survival_probability(self, **_kwargs):
            raise RuntimeError("boom")

    with patch("neutralgrid.scanner.enrich_grid_params.get_config", return_value=cfg):
        with patch(
            "neutralgrid.validation.stochastic.StochasticRegimeChecker",
            _ExplodingChecker,
        ):
            recalculated = _recalculate_survival_prob_from_grid(rd, grid, sym="BTCUSDT")
            if recalculated is not None:
                rd.survival_prob = recalculated

    assert recalculated is None
    assert rd.survival_prob == pytest.approx(original)
