from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pandas as pd

import run_full_pipeline
from neutralgrid.backtest.candidate_pipeline import convert_to_training_row
from neutralgrid.scanner.empirical_profile_v20260302 import DEFAULT_PROFILE
from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig


def test_ev_contract_fingerprint_is_deterministic_and_profile_sensitive() -> None:
    first = PnLRanker(RankingConfig(use_empirical_alignment=False))
    second = PnLRanker(RankingConfig(use_empirical_alignment=False))
    assert first.ev_contract_fingerprint == second.ev_contract_fingerprint

    first._empirical_profile = DEFAULT_PROFILE
    second._empirical_profile = replace(DEFAULT_PROFILE, fill_rate_scale=1.01)
    assert first.ev_contract_fingerprint != second.ev_contract_fingerprint


def test_post_scoring_stamps_ev_contract_without_changing_score_contract() -> None:
    class DummyRanker:
        ev_contract_fingerprint = "f" * 64

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def compute_score(self, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                rank_score=1.25,
                ev_24h=3.50,
                ev_raw=4.10,
                ev_aligned=3.00,
                expected_fills=1.0,
                fill_rate_scale=1.0,
                fill_rate_scope="global",
                fill_rate_scope_samples=50.0,
                ev_alignment_scope="global",
                ev_alignment_scope_samples=50.0,
                ev_alignment_r2=0.5,
                fill_revenue_pct=2.5,
                funding_cost_pct=0.1,
                boundary_loss_pct=0.2,
                total_penalty=0.3,
            )

    class DummyMeta:
        @staticmethod
        def load(_path: Any) -> SimpleNamespace:
            return SimpleNamespace(is_trained=False)

    row = {
        "symbol": "BTCUSDT",
        "score": 90.0,
        "profit_per_grid_pct": 0.45,
        "num_grids": 20,
        "survival_prob": 0.81,
        "trend_prob": 0.18,
        "range_size_pct": 5.0,
    }
    with (
        patch.object(run_full_pipeline, "PnLRanker", DummyRanker),
        patch.object(run_full_pipeline, "RankingConfig", lambda: None),
        patch.object(run_full_pipeline, "MetaLabeler", DummyMeta),
        patch.object(run_full_pipeline, "AFML_POST_SCORING_AVAILABLE", True),
    ):
        scored = run_full_pipeline._apply_afml_post_scoring(pd.DataFrame([row]))

    assert scored.loc[0, "ev_score"] == 1.25
    assert scored.loc[0, "ev_contract_fingerprint"] == "f" * 64


def test_backtest_training_row_preserves_ev_contract_fingerprint() -> None:
    candidate = {
        "symbol": "BTCUSDT",
        "candidate_id": "BTC-1",
        "start_time_utc": "2026-01-01T00:00:00Z",
        "ev_contract_fingerprint": "a" * 64,
    }
    result = {
        "net_pnl_pct": 4.0,
        "duration_hours": 1.0,
        "barrier_touched": "upper",
        "label_positive_by_horizon": True,
        "is_authoritative": True,
    }
    row = convert_to_training_row(
        cast(dict[str, Any], result),
        cast(dict[str, Any], candidate),
    )
    assert row["ev_contract_fingerprint"] == "a" * 64


def test_backtest_training_row_preserves_hmm_artifact_version() -> None:
    candidate = {
        "symbol": "BTCUSDT",
        "candidate_id": "BTC-1",
        "start_time_utc": "2026-01-01T00:00:00Z",
        "hmm_artifact_version": "rolling_180d_20260809_000434",
    }
    result = {
        "net_pnl_pct": 4.0,
        "duration_hours": 1.0,
        "barrier_touched": "upper",
        "label_positive_by_horizon": True,
        "is_authoritative": True,
    }

    row = convert_to_training_row(
        cast(dict[str, Any], result),
        cast(dict[str, Any], candidate),
    )

    assert row["hmm_artifact_version"] == "rolling_180d_20260809_000434"
