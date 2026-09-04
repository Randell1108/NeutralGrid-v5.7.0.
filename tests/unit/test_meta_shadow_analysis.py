from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from neutralgrid.live.decision.meta_shadow_analysis import (
    ShadowAnalysisConfig,
    _pending_authoritative_outcomes,
    analyze_joined_shadow,
    join_shadow_to_outcomes,
    load_decision_jsonl,
)


def test_join_uses_deploy_snapshot_candidate_fallback(tmp_path: Path) -> None:
    path = tmp_path / "live_decisions.jsonl"
    record = {
        "ts": "2026-06-21T00:00:00+00:00",
        "symbol": "ENAUSDT",
        "strategy_id": "412730355",
        "candidate_id": None,
        "verdict": "ADJUST",
        "meta_would_tilt": False,
        "meta_influenced_verdict": False,
        "evaluation": {
            "meta_proba": 0.42,
            "meta_authoritative": True,
            "meta_full_fidelity": True,
            "deploy_snapshot": {"candidate_id": "ENAUSDT_20260618_115611_25245053"},
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    decisions = load_decision_jsonl([path])
    outcomes = pd.DataFrame(
        [
            {
                "candidate_id": "ENAUSDT_20260618_115611_25245053",
                "pnl_pct": -1.0,
            }
        ]
    )

    joined = join_shadow_to_outcomes(decisions, outcomes)

    assert len(joined) == 1
    assert joined.iloc[0]["candidate_id"] == "ENAUSDT_20260618_115611_25245053"


def test_join_falls_back_to_strategy_symbol_only_when_decision_candidate_missing() -> None:
    decisions = pd.DataFrame(
        [
            {
                "ts_utc": "2026-06-21T00:00:00+00:00",
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "",
                "verdict": "ADJUST",
                "meta_proba": 0.42,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
            },
            {
                "ts_utc": "2026-06-21T00:01:00+00:00",
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "STALE_CANDIDATE",
                "verdict": "ADJUST",
                "meta_proba": 0.41,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "strategy_id": "412730355",
                "symbol": "ENAUSDT",
                "candidate_id": "ENAUSDT_20260618_115611_25245053",
                "pnl_pct": -1.0,
            }
        ]
    )

    joined = join_shadow_to_outcomes(decisions, outcomes)

    assert len(joined) == 1
    assert joined.iloc[0]["join_method"] == "strategy_symbol"
    assert joined.iloc[0]["join_key"] == "412730355|ENAUSDT"
    assert joined.iloc[0]["candidate_id"] == "ENAUSDT_20260618_115611_25245053"


def test_join_rejects_duplicate_finalized_outcome_identity() -> None:
    decisions = pd.DataFrame(
        [
            {
                "ts_utc": "2026-06-21T00:00:00+00:00",
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "CID-1",
            }
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "CID-1",
                "pnl_pct": 1.0,
            },
            {
                "symbol": "ENAUSDT",
                "strategy_id": "412730356",
                "candidate_id": "CID-1",
                "pnl_pct": -1.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="duplicate candidate_id"):
        join_shadow_to_outcomes(decisions, outcomes)


def test_decision_loader_flattens_verdict_inert_execution_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "live_decisions.jsonl"
    path.write_text(
        json.dumps(
            {
                "ts": "2026-08-01T18:00:00+00:00",
                "symbol": "BANDUSDT",
                "strategy_id": "strategy-1",
                "candidate_id": "candidate-1",
                "verdict": "CONTINUE",
                "profit_deterioration": {
                    "current_total_profit_usdt": 4.0,
                    "peak_total_profit_usdt": 10.0,
                    "giveback_usdt": 6.0,
                    "giveback_pct_of_positive_peak": 60.0,
                },
                "evaluation": {
                    "execution_risk": {
                        "source": "sequence_linked_l2_public_private_events",
                        "l2_run_id": "l2-1",
                        "l2_segment_id": "segment-1",
                        "liquidity_state": "sustained_joint_deterioration",
                        "sustained_joint_deterioration": True,
                        "exit_depth_current_to_baseline": 0.4,
                        "mean_estimated_slippage_bps": 2.5,
                    },
                    "private_event_evidence": {
                        "run_id": "private-1",
                        "event_completeness": "event_complete",
                        "commission_usdt": -0.2,
                        "funding_fee_usdt": -0.1,
                    },
                    "l2_risk": {"expected_exit_impact_bps": 3.2},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_decision_jsonl([path])

    assert loaded.iloc[0]["execution_risk_source"] == (
        "sequence_linked_l2_public_private_events"
    )
    assert loaded.iloc[0]["l2_run_id"] == "l2-1"
    assert loaded.iloc[0]["private_event_completeness"] == "event_complete"
    assert loaded.iloc[0]["mean_estimated_slippage_bps"] == 2.5
    assert loaded.iloc[0]["expected_exit_impact_bps"] == 3.2
    assert loaded.iloc[0]["gain_giveback_pct"] == 60.0


def test_analysis_reports_insufficient_joined_rows_when_no_outcomes() -> None:
    joined = pd.DataFrame()

    result = analyze_joined_shadow(
        joined,
        config=ShadowAnalysisConfig(min_joined_rows=2),
    )

    assert result["status"] == "insufficient_joined_rows"
    assert result["ready_to_enable"] is False
    assert result["d5_calibrated"] is False
    assert result["d6_oos_proven"] is False
    assert result["gate_audit"]["config_action"] == "keep_meta_tilt_enabled_false"
    assert result["recommended_live_config"] == {"meta_tilt_enabled": False}


def test_unique_candidate_count_classifies_blanks_without_future_warning() -> None:
    cases = [
        ([None, ""], 0),
        (["", ""], 0),
        (["A", ""], 1),
        (["A", "A", "B", 123], 3),
    ]

    for candidate_ids, expected_count in cases:
        joined = pd.DataFrame(
            [
                {
                    "join_key": f"BOT_{index}",
                    "candidate_id": candidate_id,
                    "ts_utc": "2026-06-21T00:00:00+00:00",
                    "verdict": "ADJUST",
                    "meta_proba": 0.42,
                    "meta_authoritative": True,
                    "meta_full_fidelity": True,
                    "pnl_pct": -1.0,
                }
                for index, candidate_id in enumerate(candidate_ids)
            ]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = analyze_joined_shadow(
                joined,
                config=ShadowAnalysisConfig(min_joined_rows=10),
            )

        assert result["counts"]["unique_candidates"] == expected_count


def test_analysis_reports_eligibility_funnel_for_joined_but_ineligible_rows() -> None:
    joined = pd.DataFrame(
        [
            {
                "join_key": "BOT_1",
                "candidate_id": "",
                "ts_utc": "2026-06-21T00:00:00+00:00",
                "verdict": "ADJUST",
                "meta_proba": None,
                "meta_authoritative": None,
                "meta_full_fidelity": None,
                "pnl_pct": 1.0,
            },
            {
                "join_key": "BOT_2",
                "candidate_id": "",
                "ts_utc": "2026-06-21T00:01:00+00:00",
                "verdict": "ADJUST",
                "meta_proba": 0.42,
                "meta_authoritative": True,
                "meta_full_fidelity": False,
                "pnl_pct": -1.0,
            },
        ]
    )

    result = analyze_joined_shadow(joined, config=ShadowAnalysisConfig(min_joined_rows=3))

    funnel = result["eligibility_funnel"]
    assert funnel["joined_rows"] == 2
    assert funnel["with_join_key"] == 2
    assert funnel["with_pnl_pct"] == 2
    assert funnel["with_ts_utc"] == 2
    assert funnel["with_meta_proba"] == 1
    assert funnel["meta_authoritative_true"] == 1
    assert funnel["meta_full_fidelity_true"] == 0
    assert funnel["with_authoritative_full_fidelity_meta"] == 0
    assert funnel["all_eligible"] == 0
    assert funnel["failed_reason_counts"]["missing_meta_proba"] == 1
    assert funnel["failed_reason_counts"]["not_full_fidelity"] == 2


def test_post_mortem_ticks_are_excluded_from_eligibility() -> None:
    """Ticks recorded after the joined outcome's end_time_utc (ghost ticks on a
    canceled bot — the ENAUSDT 2026-06-21/07-03 case) must not become eligible
    D5/D6 rows; ticks within the bot's life stay eligible."""
    base = {
        "join_key": "412730355|ENAUSDT",
        "candidate_id": "ENAUSDT_20260618_115611_25245053",
        "verdict": "END",
        "meta_proba": 0.47,
        "meta_authoritative": True,
        "meta_full_fidelity": True,
        "pnl_pct": 2.95,
        "start_time_utc": "2026-06-18T13:01:13+00:00",
        "end_time_utc": "2026-06-18T16:23:58+00:00",
    }
    joined = pd.DataFrame(
        [
            {**base, "ts_utc": "2026-06-18T15:48:19+00:00"},  # within life
            {**base, "ts_utc": "2026-06-21T07:35:37+00:00"},  # post-mortem
            {**base, "ts_utc": "2026-07-03T03:41:22+00:00"},  # post-mortem
            {**base, "ts_utc": "2026-06-18T12:00:00+00:00"},  # pre-deploy
        ]
    )

    result = analyze_joined_shadow(joined, config=ShadowAnalysisConfig(min_joined_rows=5))

    funnel = result["eligibility_funnel"]
    assert funnel["joined_rows"] == 4
    assert funnel["with_authoritative_full_fidelity_meta"] == 4
    assert funnel["failed_reason_counts"]["post_outcome_tick"] == 2
    assert funnel["failed_reason_counts"]["pre_deploy_tick"] == 1
    assert funnel["all_eligible"] == 1
    assert result["counts"]["eligible_rows"] == 1


def test_rows_without_life_window_are_kept_but_counted() -> None:
    """Outcomes lacking end/start times must not silently drop eligible rows —
    they are kept and surfaced via outcome_life_window_missing."""
    joined = pd.DataFrame(
        [
            {
                "join_key": "999|XUSDT",
                "candidate_id": "",
                "ts_utc": "2026-06-21T00:00:00+00:00",
                "verdict": "ADJUST",
                "meta_proba": 0.44,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
                "pnl_pct": -7.25,
            }
        ]
    )

    result = analyze_joined_shadow(joined, config=ShadowAnalysisConfig(min_joined_rows=5))

    funnel = result["eligibility_funnel"]
    assert funnel["all_eligible"] == 1
    assert funnel["outcome_life_window_missing"] == 1
    assert funnel["failed_reason_counts"]["post_outcome_tick"] == 0


def test_pending_authoritative_outcomes_reports_unmatched_d7_rows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "source_path": "live.jsonl",
                "source_line": 1,
                "ts_utc": "2026-06-21T00:00:00+00:00",
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "ENAUSDT_20260618_115611_25245053",
                "meta_proba": 0.46,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
            },
            {
                "source_path": "live.jsonl",
                "source_line": 2,
                "ts_utc": "2026-06-21T00:01:00+00:00",
                "symbol": "ENAUSDT",
                "strategy_id": "412730355",
                "candidate_id": "ENAUSDT_20260618_115611_25245053",
                "meta_proba": 0.45,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
            },
            {
                "source_path": "live.jsonl",
                "source_line": 3,
                "ts_utc": "2026-06-21T00:02:00+00:00",
                "symbol": "AVAXUSDT",
                "strategy_id": "412730999",
                "candidate_id": "",
                "meta_proba": 0.50,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
            },
        ]
    )
    joined = pd.DataFrame(
        [
            {
                "source_path": "live.jsonl",
                "source_line": 1,
                "join_key": "ENAUSDT_20260618_115611_25245053",
            }
        ]
    )

    pending = _pending_authoritative_outcomes(decisions, joined)

    assert pending["authoritative_full_fidelity_rows"] == 3
    assert pending["matched_rows"] == 1
    assert pending["pending_rows"] == 2
    assert pending["pending_unique_join_keys"] == 2
    assert pending["pending_by_join_key"] == [
        {
            "join_key": "412730999|AVAXUSDT",
            "join_method": "strategy_symbol",
            "symbol": "AVAXUSDT",
            "strategy_id": "412730999",
            "candidate_id": "",
            "decision_rows": 1,
            "first_ts_utc": "2026-06-21T00:02:00+00:00",
        },
        {
            "join_key": "ENAUSDT_20260618_115611_25245053",
            "join_method": "candidate_id",
            "symbol": "ENAUSDT",
            "strategy_id": "412730355",
            "candidate_id": "ENAUSDT_20260618_115611_25245053",
            "decision_rows": 1,
            "first_ts_utc": "2026-06-21T00:01:00+00:00",
        },
    ]


def test_analysis_split_keeps_each_join_key_on_one_side() -> None:
    rows = []
    for bot_idx in range(6):
        for tick_idx in range(2):
            rows.append(
                {
                    "join_key": f"BOT_{bot_idx}",
                    "candidate_id": f"CID_{bot_idx}",
                    "ts_utc": f"2026-06-21T00:{bot_idx}{tick_idx}:00+00:00",
                    "verdict": "ADJUST",
                    "meta_proba": 0.40 if bot_idx % 2 == 0 else 0.80,
                    "meta_authoritative": True,
                    "meta_full_fidelity": True,
                    "pnl_pct": -1.0 if bot_idx % 2 == 0 else 2.0,
                }
            )
    joined = pd.DataFrame(rows)

    result = analyze_joined_shadow(
        joined,
        config=ShadowAnalysisConfig(
            min_joined_rows=6,
            min_fit_adjust_rows=1,
            min_fit_adjust_join_keys=1,
            min_oos_rows=1,
            min_oos_join_keys=1,
            min_oos_tilt_rows=1,
            min_oos_tilt_join_keys=1,
            fit_fraction=0.5,
            thresholds=(0.5,),
        ),
    )

    assert result["split_audit"]["split_strategy"] == "temporal_join_key"
    assert result["split_audit"]["overlap_join_keys"] == []
    assert result["counts"]["fit_join_keys"] == 3
    assert result["counts"]["oos_join_keys"] == 3


def test_analysis_calibrates_threshold_and_observes_oos_lift() -> None:
    joined = _synthetic_joined_rows()

    result = analyze_joined_shadow(
        joined,
        config=ShadowAnalysisConfig(
            min_joined_rows=20,
            min_fit_adjust_rows=8,
            min_oos_rows=8,
            min_oos_tilt_rows=3,
            thresholds=(0.30, 0.50, 0.70),
        ),
    )

    assert result["status"] == "oos_lift_observed"
    assert result["d5_calibrated"] is True
    assert result["d6_oos_proven"] is True
    assert result["ready_to_enable"] is True
    assert result["recommended_meta_tilt_low_threshold"] == 0.5
    assert result["split_audit"]["overlap_join_key_count"] == 0
    assert result["gate_audit"]["config_action"] == "operator_may_enable_meta_tilt_after_review"
    assert result["recommended_live_config"] == {
        "meta_tilt_enabled": True,
        "meta_tilt_low_threshold": 0.5,
    }
    assert result["oos_precision_lift_vs_all_adjust"] > 0.0


def _synthetic_joined_rows() -> pd.DataFrame:
    rows = []
    for idx in range(40):
        low_confidence = idx % 2 == 0
        rows.append(
            {
                "candidate_id": f"CID_{idx:03d}",
                "ts_utc": f"2026-06-21T00:{idx:02d}:00+00:00",
                "verdict": "ADJUST",
                "meta_proba": 0.40 if low_confidence else 0.80,
                "meta_authoritative": True,
                "meta_full_fidelity": True,
                "pnl_pct": -1.0 if low_confidence else 2.0,
            }
        )
    return pd.DataFrame(rows)
