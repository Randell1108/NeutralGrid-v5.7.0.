"""Keep default stop-loss units and training entry points consistent."""

from __future__ import annotations

import inspect

import pytest

from neutralgrid.backtest.candidate_pipeline import convert_to_training_row
from neutralgrid.core.config import BarrierConfig, Config, GridConfig
from neutralgrid.models.barrier_config import UnifiedBarrierConfig
from neutralgrid.models.meta_labeler import MetaLabelerConfig, create_meta_labeler
from neutralgrid.models.triple_barrier import (
    TripleBarrierConfig,
    create_triple_barrier_labeler,
)
from neutralgrid.scanner.pnl_ranker import RankingConfig, create_pnl_ranker
from neutralgrid.training.data_generator import LabelConfig, load_and_prepare_training_data
from neutralgrid.validation.utility import UtilityConfig, UtilityScorer


@pytest.mark.parametrize(
    "value",
    [
        BarrierConfig().sl_pct,
        UnifiedBarrierConfig().sl_pct,
        TripleBarrierConfig().sl_pct,
        LabelConfig().sl_pct,
        MetaLabelerConfig().sl_pct,
        UtilityConfig().stop_loss_pct,
        inspect.signature(convert_to_training_row).parameters["sl_pct"].default,
        inspect.signature(create_triple_barrier_labeler).parameters["sl_pct"].default,
        inspect.signature(create_meta_labeler).parameters["sl_pct"].default,
        inspect.signature(load_and_prepare_training_data).parameters["sl_pct"].default,
        inspect.signature(UtilityScorer.compute_utility).parameters["stop_loss_pct"].default,
        inspect.signature(UtilityScorer.expected_drawdown).parameters["stop_loss_pct"].default,
    ],
)
def test_training_percentage_defaults_use_ten_percent(value: float) -> None:
    assert value == -10.0


def test_grid_and_ranker_use_decimal_units() -> None:
    assert GridConfig().stop_loss_pct == -0.10
    assert Config().grid.stop_loss_pct == -0.10
    assert RankingConfig().sl_pct == -0.10
    assert inspect.signature(create_pnl_ranker).parameters["sl_pct"].default == -0.10


def test_explicit_historical_stop_remains_supported() -> None:
    config = Config(barrier=BarrierConfig(sl_pct=-12.0))
    assert config.grid.stop_loss_pct == -0.12
    assert create_meta_labeler(sl_pct=-12.0).config.sl_pct == -12.0
    assert UnifiedBarrierConfig(sl_pct=-12.0).sl_decimal == -0.12


def test_meta_retrain_cli_uses_ten_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    import retrain_meta_labeler

    monkeypatch.setattr("sys.argv", ["retrain_meta_labeler.py"])
    assert retrain_meta_labeler.parse_args().sl_pct == -10.0


def test_stop_default_does_not_enable_or_change_fastwin_execution() -> None:
    from backtest.btk_label_contract import TRAINING_ENGINE_DEFAULTS
    from neutralgrid.core.constants import BOT_HORIZON_HOURS
    from neutralgrid.models.meta_labeler import ACTIVE_META_TARGET_CONTRACT

    assert TRAINING_ENGINE_DEFAULTS["cb_enabled"] is False
    assert TRAINING_ENGINE_DEFAULTS["capital_preservation_enabled"] is False
    assert BOT_HORIZON_HOURS == 6.0
    assert ACTIVE_META_TARGET_CONTRACT == "fast_winner_time_to_3pct_le_7h"
