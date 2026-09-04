from __future__ import annotations

import pandas as pd

from neutralgrid.live.decision.execution_outcome_analysis import (
    ExecutionOutcomeAnalysisConfig,
    analyze_joined_execution_outcomes,
)


def _row(
    key: str,
    at: str,
    pnl: float,
    *,
    sustained: bool,
    giveback: float,
) -> dict[str, object]:
    return {
        "join_key": key,
        "join_method": "candidate_id",
        "candidate_id": key,
        "strategy_id": f"strategy-{key}",
        "symbol": "TESTUSDT",
        "ts_utc": at,
        "verdict": "CONTINUE",
        "pnl_pct": pnl,
        "start_time_utc": "2026-08-01T00:00:00+00:00",
        "end_time_utc": "2026-08-04T00:00:00+00:00",
        "execution_risk_source": "sequence_linked_l2_public_private_events",
        "l2_run_id": f"run-{key}",
        "l2_segment_id": "segment-1",
        "private_event_run_id": f"private-{key}",
        "current_total_profit_usdt": 1.0,
        "gain_giveback_usdt": giveback,
        "gain_giveback_pct": giveback * 10.0,
        "sustained_joint_deterioration": sustained,
        "temporary_joint_deterioration": not sustained,
        "exit_depth_current_to_baseline": 0.5 if sustained else 1.1,
        "mean_estimated_slippage_bps": 2.0 if sustained else 0.2,
        "mean_adverse_selection_5s_bps": 1.5 if sustained else -0.1,
        "expected_exit_impact_bps": 3.0 if sustained else 0.3,
        "private_event_completeness": "event_complete",
        "public_trade_status": "available",
    }


def test_empty_analysis_creates_no_gate_or_threshold() -> None:
    result = analyze_joined_execution_outcomes(pd.DataFrame())

    assert result["status"] == "insufficient_no_exact_joined_outcomes"
    assert result["runtime_effect"] == "none"
    assert result["threshold_selected"] is False
    assert result["promotion_gate_created"] is False
    assert result["verdict_influence_validated"] is False


def test_bot_level_temporal_split_deduplicates_ticks_and_has_no_identity_overlap() -> None:
    rows = [
        _row("A", "2026-08-01T01:00:00+00:00", 2.0, sustained=False, giveback=1.0),
        _row("A", "2026-08-01T01:00:00+00:00", 2.0, sustained=False, giveback=1.0),
        _row("A", "2026-08-01T02:00:00+00:00", 2.0, sustained=True, giveback=3.0),
        _row("B", "2026-08-02T01:00:00+00:00", -1.0, sustained=True, giveback=4.0),
        _row("C", "2026-08-03T01:00:00+00:00", 0.5, sustained=False, giveback=0.5),
    ]

    result = analyze_joined_execution_outcomes(
        pd.DataFrame(rows),
        config=ExecutionOutcomeAnalysisConfig(development_fraction=2 / 3),
    )

    assert result["status"] == "observational_temporal_split_available"
    assert result["counts"]["duplicate_evidence_rows_dropped"] == 1
    assert result["counts"]["bot_rows"] == 3
    assert result["all_bots_observational_summary"]["bots"] == 3
    assert result["temporal_split"]["development_join_keys"] == 2
    assert result["temporal_split"]["later_join_keys"] == 1
    assert result["temporal_split"]["overlap_join_key_count"] == 0
    assert result["threshold_selected"] is False
    assert result["verdict_influence_validated"] is False
    bot_a = next(row for row in result["bot_rows"] if row["join_key"] == "A")
    assert bot_a["decision_ticks"] == 2
    assert bot_a["sustained_deterioration_observed"] is True
    assert bot_a["max_gain_giveback_usdt"] == 3.0


def test_post_closure_execution_tick_is_excluded() -> None:
    row = _row(
        "A",
        "2026-08-05T01:00:00+00:00",
        2.0,
        sustained=True,
        giveback=3.0,
    )

    result = analyze_joined_execution_outcomes(pd.DataFrame([row]))

    assert result["status"] == "insufficient_no_execution_evidence"
    assert result["counts"]["eligible_evidence_rows_before_dedup"] == 0
