from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.validate_backtest_live_reconciliation import (  # noqa: E402
    ManualExports,
    _load_cached_klines_resilient,
    _select_exact_strategy_id_cohort,
    _strategy_id_set,
    _strategy_id_integrity_report,
    _timestamp_coverage_preprobe,
    _timestamp_decision,
    _timestamp_decisions_for_policy,
    _validation_split_summary,
    summarize,
)


def _exports(
    order_time: str | None,
    telemetry_time: str | None = None,
) -> ManualExports:
    if order_time is None:
        orders = pd.DataFrame(
            columns=pd.Index(
                ["_symbol", "_strategy_id", "_time", "_update_time", "_source_file"]
            )
        )
    else:
        orders = pd.DataFrame(
            [
                {
                    "_symbol": "DOGEUSDT",
                    "_strategy_id": "411991896",
                    "_time": pd.Timestamp(order_time, tz="UTC"),
                    "_update_time": pd.Timestamp(order_time, tz="UTC")
                    + pd.Timedelta(hours=2),
                    "_source_file": "Order history 8.csv",
                }
            ]
        )
    telemetry = (
        pd.DataFrame(
            [
                {
                    "_symbol": "DOGEUSDT",
                    "_strategy_id": "411991896",
                    "_created_time": pd.Timestamp(telemetry_time, tz="UTC"),
                    "_source_file": "Live/2026-05-14/DOGEUSDT/snapshot.json",
                }
            ]
        )
        if telemetry_time is not None
        else pd.DataFrame()
    )
    return ManualExports(
        orders=orders,
        trades=pd.DataFrame(),
        transactions=pd.DataFrame(),
        input_trades=pd.DataFrame(),
        telemetry_times=telemetry,
    )


def _row(start: str = "2026-05-14 08:00:00+00:00") -> pd.Series:
    return pd.Series(
        {
            "_row_id": 1,
            "symbol": "DOGEUSDT",
            "strategy_id_text": "411991896",
            "start_time_utc": pd.Timestamp(start),
            "end_time_utc": pd.Timestamp(start) + pd.Timedelta(hours=2),
            "duration_hours": 2.0,
            "pnl_pct": 2.5,
            "mode": "geometric",
        }
    )


def test_stored_policy_keeps_csv_utc_timestamp_unshifted() -> None:
    row = _row()
    decision = _timestamp_decision(
        row,
        _exports("2026-05-14 08:00:00"),
        timestamp_policy="stored_utc",
        selected_time_policy="stored_utc",
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert str(decision.selected_start_utc) == "2026-05-14 08:00:00+00:00"
    assert str(decision.candidate_local_to_utc_start) == "2026-05-14 13:00:00+00:00"
    assert decision.time_evidence_class == "exact_stored_match"
    assert decision.time_delta_seconds_stored_vs_manual == 0.0


def test_evidence_matched_selects_local_offset_only_when_manual_start_matches() -> None:
    row = _row()
    decision = _timestamp_decision(
        row,
        _exports("2026-05-14 13:00:00"),
        timestamp_policy="evidence_matched",
        selected_time_policy="evidence_matched",
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert str(decision.selected_start_utc) == "2026-05-14 13:00:00+00:00"
    assert str(decision.selected_end_utc) == "2026-05-14 15:00:00+00:00"
    assert decision.selected_time_policy == "local_utc_minus_5_to_utc"
    assert decision.time_evidence_class == "exact_local_offset_match"
    assert decision.timestamp_modelable is True
    assert decision.time_delta_seconds_stored_vs_manual == -18000.0
    assert decision.time_delta_seconds_local_adjusted_vs_manual == 0.0


def test_evidence_matched_rejects_missing_manual_order_evidence() -> None:
    decision = _timestamp_decision(
        _row(),
        _exports(None),
        timestamp_policy="evidence_matched",
        selected_time_policy="evidence_matched",
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert decision.selected_start_utc is None
    assert decision.timestamp_modelable is False
    assert decision.time_evidence_class == "missing_manual_evidence"
    assert decision.time_rejection_reason == "time_evidence_class:missing_manual_evidence"


def test_evidence_matched_accepts_exact_identity_telemetry_creation_time() -> None:
    decision = _timestamp_decision(
        _row(),
        _exports(None, telemetry_time="2026-05-14 08:00:00"),
        timestamp_policy="evidence_matched",
        selected_time_policy="evidence_matched",
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert decision.selected_start_utc == pd.Timestamp(
        "2026-05-14 08:00:00", tz="UTC"
    )
    assert decision.timestamp_modelable is True
    assert decision.time_evidence_class == "exact_stored_match"
    assert decision.time_evidence_source == "live_telemetry_created_at_lima"
    assert decision.manual_order_start_utc is None
    assert decision.telemetry_created_at_utc == pd.Timestamp(
        "2026-05-14 08:00:00", tz="UTC"
    )

    preprobe = _timestamp_coverage_preprobe(
        pd.DataFrame([_row()]),
        _exports(None, telemetry_time="2026-05-14 08:00:00"),
        holdout_fraction=0.40,
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )
    assert preprobe["exact_time_evidence_coverage_rows"] == 1
    assert preprobe["exact_manual_order_history_coverage_rows"] == 0
    assert preprobe["exact_live_telemetry_coverage_rows"] == 1


def test_dual_diagnostic_returns_stored_and_local_non_promotable_decisions() -> None:
    decisions = _timestamp_decisions_for_policy(
        _row(),
        _exports("2026-05-14 13:00:00"),
        timestamp_policy="dual_diagnostic",
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert [item.selected_time_policy for item in decisions] == [
        "stored_utc",
        "local_utc_minus_5_to_utc",
    ]
    assert {item.time_rejection_reason for item in decisions} == {
        "diagnostic_only_not_promotable"
    }


def test_timestamp_preprobe_reports_small_holdout_denominator() -> None:
    rows = []
    for idx in range(5):
        item = _row(f"2026-05-14 0{idx}:00:00+00:00")
        item["_row_id"] = idx
        item["pnl_pct"] = 2.0 if idx < 3 else -1.0
        item["duration_hours"] = 2.0
        rows.append(item)
    rows_df = pd.DataFrame(rows)

    out = _timestamp_coverage_preprobe(
        rows_df,
        _exports(None),
        holdout_fraction=0.40,
        duration_source="workbook",
        fixed_duration_hours=6.0,
    )

    assert out["workbook_rows"] == 5
    assert out["diagnostic_fast_winner_contract"] == (
        "terminal_pnl_pct_gt_1_and_duration_hours_lt_7"
    )
    assert out["active_meta_target_contract_evaluated"] is False
    assert out["chronological_holdout_rows"] == 2
    assert out["chronological_holdout_non_fast_non_winner_rows"] == 2
    assert out["holdout_non_winner_specificity_step_pct"] == 50.0


def test_validation_split_summary_reports_holdout_model_metrics() -> None:
    rows = []
    for idx in range(5):
        rows.append(
            {
                "symbol": "DOGEUSDT",
                "strategy_id": str(idx),
                "validation_start_time_utc": f"2026-05-14 0{idx}:00:00+00:00",
                "evidence_class": "partial",
                "model_ran": True,
                "model_pnl_pct": 2.0 if idx < 3 else -0.5,
                "live_pnl_pct": 2.5 if idx < 3 else -1.0,
                "live_duration_hours": 2.0,
            }
        )

    out = _validation_split_summary(
        rows,
        split_mode="chronological",
        holdout_fraction=0.40,
        include_table=False,
    )

    assert out["holdout_rows"] == 2
    assert out["holdout_model_rows"] == 2
    assert out["holdout_model_mean_abs_pnl_error"] == 0.5
    assert out["holdout_non_winner_specificity_pnl_lte_0"] == 1.0


def test_validation_split_uses_stored_chronology_for_rejected_rows() -> None:
    rows = []
    for idx in range(5):
        model_ran = idx in {2, 4}
        rows.append(
            {
                "symbol": "DOGEUSDT",
                "strategy_id": str(idx),
                "validation_start_time_utc": (
                    f"2026-05-14 0{idx}:00:00+00:00" if model_ran else None
                ),
                "validation_split_time_utc": f"2026-05-14 0{idx}:00:00+00:00",
                "evidence_class": "partial" if model_ran else "missing",
                "model_ran": model_ran,
                "model_pnl_pct": 1.0,
                "live_pnl_pct": 1.0,
                "live_duration_hours": 2.0,
            }
        )

    out = _validation_split_summary(
        rows,
        split_mode="chronological",
        holdout_fraction=0.40,
        include_table=True,
    )

    assert out["holdout_rows"] == 2
    assert out["holdout_model_rows"] == 1
    assert [item["strategy_id"] for item in out["split_table"][-2:]] == ["3", "4"]


def test_summarize_all_timestamp_rejected_rows_is_fail_closed_not_exception() -> None:
    rows = [
        {
            "symbol": "DOGEUSDT",
            "strategy_id": "411991896",
            "model_ran": False,
            "klines_available": False,
            "kline_cache_status": "not_attempted_timestamp_rejected",
            "timestamp_modelable": False,
            "time_rejection_reason": "time_evidence_class:missing_manual_evidence",
        }
    ]

    out = summarize(rows, scope="all")

    assert out["rows"] == 1
    assert out["model_rows"] == 0
    assert out["strict_manual_rows"] == 0
    assert out["manual_any_rows"] == 0
    assert out["missing_kline_rows"] == 0
    assert out["kline_not_attempted_timestamp_rejected_rows"] == 1
    assert out["kline_cache_error_rows"] == 0


def test_strategy_id_filter_rejects_non_numeric_tokens() -> None:
    assert _strategy_id_set("413491078, 413490966") == {"413491078", "413490966"}

    try:
        _strategy_id_set("413491078,not-a-strategy")
    except ValueError as exc:
        assert "not-a-strategy" in str(exc)
    else:  # pragma: no cover - regression assertion
        raise AssertionError("invalid strategy ID must fail before cohort selection")


def test_strategy_id_integrity_report_classifies_duplicates_before_dedup() -> None:
    report = _strategy_id_integrity_report(
        pd.DataFrame(
            {
                "strategy_id_text": ["413491078", "413491078", "", "413490966"]
            }
        )
    )

    assert report == {
        "rows": 4,
        "nonblank_strategy_id_rows": 3,
        "blank_strategy_id_rows": 1,
        "unique_strategy_ids": 2,
        "duplicate_strategy_id_rows": 2,
        "duplicate_strategy_id_counts": {"413491078": 2},
    }


def test_exact_strategy_id_cohort_requires_complete_unique_raw_identity() -> None:
    workbook = pd.DataFrame(
        {"strategy_id_text": ["413491078", "413491078", "413490966"]}
    )
    selected = workbook.drop_duplicates("strategy_id_text", keep="first")

    try:
        _select_exact_strategy_id_cohort(
            workbook,
            selected,
            {"413491078", "missing"},
        )
    except ValueError as exc:
        message = str(exc)
        assert "missing_ids=['missing']" in message
        assert "duplicate_id_counts={'413491078': 2}" in message
    else:  # pragma: no cover - regression assertion
        raise AssertionError("incomplete or duplicate exact identity must fail closed")


def test_exact_strategy_id_cohort_rejects_ids_removed_by_filters() -> None:
    workbook = pd.DataFrame(
        {"strategy_id_text": ["413491078", "413490966"]}
    )
    selected = workbook.loc[workbook["strategy_id_text"] == "413491078"].copy()

    try:
        _select_exact_strategy_id_cohort(
            workbook,
            selected,
            {"413491078", "413490966"},
        )
    except ValueError as exc:
        assert "filtered_out_ids=['413490966']" in str(exc)
    else:  # pragma: no cover - regression assertion
        raise AssertionError("filtered exact identity must fail closed")


def test_exact_strategy_id_cohort_reports_complete_unique_match() -> None:
    workbook = pd.DataFrame(
        {"strategy_id_text": ["413491078", "413490966", "unrequested"]}
    )
    selected = workbook.copy()

    cohort, report = _select_exact_strategy_id_cohort(
        workbook,
        selected,
        {"413491078", "413490966"},
    )

    assert set(cohort["strategy_id_text"]) == {"413491078", "413490966"}
    assert report == {
        "requested_strategy_ids": 2,
        "matched_strategy_ids": 2,
        "status": "complete_unique",
    }


def test_corrupt_kline_partition_is_contained_to_one_row(monkeypatch: Any) -> None:
    def _raise(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
        raise OSError("corrupt parquet footer")

    monkeypatch.setattr(
        "scripts.validate_backtest_live_reconciliation.load_cached_klines",
        _raise,
    )
    frame, error = _load_cached_klines_resilient(
        Path("unused"),
        "DOGEUSDT",
        cast(pd.Timestamp, pd.Timestamp("2026-05-14 08:00:00", tz="UTC")),
        cast(pd.Timestamp, pd.Timestamp("2026-05-14 10:00:00", tz="UTC")),
    )

    assert frame.empty
    assert error == "OSError: corrupt parquet footer"
