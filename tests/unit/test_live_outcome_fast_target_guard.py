from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


_CLI_PATH = Path(__file__).resolve().parents[2] / "retrain_meta_labeler.py"
_CLI_SPEC = importlib.util.spec_from_file_location("retrain_meta_labeler_live_guard", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
retrain_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(retrain_cli)


def test_fast_target_rejects_live_rows_without_endogenous_time_to_target() -> None:
    training_df = pd.DataFrame(
        [
            {
                "source": "backtest",
                "duration_hours": 6.0,
                "net_pnl_pct": 4.0,
                "time_to_target_hours": 2.0,
            },
            {
                "source": "backtest",
                "duration_hours": 6.0,
                "net_pnl_pct": -1.0,
                "time_to_target_hours": None,
            },
            {
                "source": "live",
                "duration_hours": 5.0,
                "net_pnl_pct": 5.0,
                "time_to_target_hours": None,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="live outcome row.*missing 'time_to_target_hours'",
    ):
        retrain_cli.prepare_fast_target_training_frame(
            training_df,
            pnl_col="net_pnl_pct",
        )


def test_fast_target_rejects_live_union_when_time_to_target_column_is_absent() -> None:
    training_df = pd.DataFrame(
        [
            {
                "source": "backtest",
                "duration_hours": 6.0,
                "net_pnl_pct": -1.0,
            },
            {
                "source": "live",
                "duration_hours": 5.0,
                "net_pnl_pct": 5.0,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="live outcome row.*missing 'time_to_target_hours'",
    ):
        retrain_cli.prepare_fast_target_training_frame(
            training_df,
            pnl_col="net_pnl_pct",
        )


def test_fast_target_accepts_live_rows_with_endogenous_time_to_target() -> None:
    training_df = pd.DataFrame(
        [
            {
                "source": "backtest",
                "duration_hours": 6.0,
                "net_pnl_pct": 4.0,
                "time_to_target_hours": 2.0,
            },
            {
                "source": "backtest",
                "duration_hours": 6.0,
                "net_pnl_pct": -1.0,
                "time_to_target_hours": None,
            },
            {
                "source": "live",
                "duration_hours": 5.0,
                "net_pnl_pct": 5.0,
                "time_to_target_hours": 3.0,
            },
        ]
    )

    filtered, summary = retrain_cli.prepare_fast_target_training_frame(
        training_df,
        pnl_col="net_pnl_pct",
    )

    assert summary["label_basis"] == "endogenous_time_to_target"
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert filtered[retrain_cli.FAST_TARGET_LABEL_COLUMN].tolist() == [1, 0, 1]
