from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pandas as pd

from neutralgrid.data.depth_shadow import (
    DepthShadowTarget,
    append_jsonl,
    build_candidate_depth_feature_frames,
    derive_position_notional_usdt,
    filter_fresh_depth_shadow_targets,
    load_depth_shadow_targets,
    make_depth_shadow_record,
    parse_candidate_scan_time_utc,
    summarize_order_book,
)


def test_summarize_order_book_computes_depth_and_fill_metrics() -> None:
    order_book = {
        "lastUpdateId": 123,
        "bids": [["99", "2"], ["98", "10"]],
        "asks": [["101", "3"], ["102", "10"]],
    }

    summary = summarize_order_book(
        order_book,
        top_n=2,
        position_notional_usdt=1000.0,
        participation_rate=0.10,
    )

    assert summary["last_update_id"] == 123
    assert summary["best_bid"] == pytest.approx(99.0)
    assert summary["best_ask"] == pytest.approx(101.0)
    assert summary["mid_price"] == pytest.approx(100.0)
    assert summary["spread_pct"] == pytest.approx(2.0)
    assert summary["top_n_bid_depth_usdt"] == pytest.approx(1178.0)
    assert summary["top_n_ask_depth_usdt"] == pytest.approx(1323.0)
    assert summary["top_n_depth_min_usdt"] == pytest.approx(1178.0)
    assert summary["partial_fill_capacity_usdt"] == pytest.approx(117.8)
    assert summary["depth_to_position_min"] == pytest.approx(1.178)
    assert summary["buy_complete_fill"] is True
    assert summary["sell_complete_fill"] is True
    assert summary["min_side_fill_ratio"] == pytest.approx(1.0)
    assert float(summary["max_side_impact_bps"]) > 0.0


def test_summarize_order_book_without_position_keeps_fill_metrics_null() -> None:
    summary = summarize_order_book(
        {"bids": [["99", "2"]], "asks": [["101", "3"]]},
        top_n=1,
    )

    assert summary["position_notional_usdt"] is None
    assert summary["depth_to_position_min"] is None
    assert summary["buy_fill_ratio"] is None
    assert summary["sell_fill_ratio"] is None


def test_load_depth_shadow_targets_reads_candidate_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,candidate_id,scan_time_utc,position_size_usdt",
                "btcusdt,c1,2026-06-25T12:00:00Z,2500",
                "ethusdt,c2,2026-06-25T13:00:00Z,",
            ]
        ),
        encoding="utf-8",
    )

    targets = load_depth_shadow_targets(
        input_path,
        symbols=["BTCUSDT", "ETHUSDT"],
        fallback_position_usdt=1000.0,
    )

    assert [target.symbol for target in targets] == ["BTCUSDT", "ETHUSDT"]
    assert targets[0].candidate_id == "c1"
    assert targets[0].position_notional_usdt == pytest.approx(2500.0)
    assert targets[1].position_notional_usdt == pytest.approx(1000.0)
    assert targets[0].scan_time_utc == "2026-06-25T12:00:00+00:00"


def test_load_depth_shadow_targets_derives_notional_from_deploy_margin_and_leverage(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,candidate_id,scan_time_utc,deploy_margin_usdt,leverage",
                "btcusdt,c1,2026-06-25T12:00:00Z,20,10",
                "ethusdt,c2,2026-06-25T12:00:00Z,0,10",
            ]
        ),
        encoding="utf-8",
    )

    targets = load_depth_shadow_targets(input_path, fallback_position_usdt=1000.0)

    assert targets[0].position_notional_usdt == pytest.approx(200.0)
    assert targets[1].position_notional_usdt == pytest.approx(0.0)


def test_derive_position_notional_prefers_direct_notional_over_margin() -> None:
    row = pd.Series(
        {
            "position_notional_usdt": 123.0,
            "deploy_margin_usdt": 20.0,
            "leverage": 10.0,
        }
    )

    assert derive_position_notional_usdt(row, fallback_position_usdt=1000.0) == pytest.approx(123.0)


def test_load_depth_shadow_targets_derives_scan_time_from_candidate_id(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "\n".join(
            [
                "symbol,candidate_id,position_size_usdt",
                "btcusdt,BTCUSDT_20260625_120000_deadbeef,2500",
            ]
        ),
        encoding="utf-8",
    )

    targets = load_depth_shadow_targets(input_path)

    assert targets[0].scan_time_utc == "2026-06-25T12:00:00+00:00"
    assert parse_candidate_scan_time_utc("bad-id") is None


def test_filter_fresh_depth_shadow_targets_rejects_stale_and_missing_scan_times() -> None:
    now = datetime(2026, 6, 25, 12, 10, 0, tzinfo=timezone.utc)
    targets = [
        DepthShadowTarget(
            symbol="BTCUSDT",
            candidate_id="fresh",
            scan_time_utc="2026-06-25T12:01:00+00:00",
        ),
        DepthShadowTarget(
            symbol="ETHUSDT",
            candidate_id="stale",
            scan_time_utc="2026-06-25T11:00:00+00:00",
        ),
        DepthShadowTarget(symbol="SOLUSDT", candidate_id="missing"),
    ]

    fresh, rejected = filter_fresh_depth_shadow_targets(
        targets,
        now=now,
        max_scan_age_seconds=900,
    )

    assert [target.candidate_id for target in fresh] == ["fresh"]
    assert [(target.candidate_id, target.reason) for target in rejected] == [
        ("stale", "stale_scan_time"),
        ("missing", "missing_scan_time"),
    ]


def test_make_record_and_append_jsonl(tmp_path: Path) -> None:
    target = DepthShadowTarget(
        symbol="BTCUSDT",
        candidate_id="candidate-1",
        scan_time_utc="2026-06-25T12:00:00+00:00",
        position_notional_usdt=1000.0,
    )
    record = make_depth_shadow_record(
        target,
        {"bids": [["99", "2"]], "asks": [["101", "3"]]},
        capture_time_utc="2026-06-25T12:01:00+00:00",
        top_n=1,
    )

    assert record["symbol"] == "BTCUSDT"
    assert record["candidate_id"] == "candidate-1"
    assert record["capture_time_utc"] == "2026-06-25T12:01:00+00:00"
    assert record["bids"] == [["99", "2"]]

    out = tmp_path / "records.jsonl"
    append_jsonl(out, [record])
    loaded = json.loads(out.read_text(encoding="utf-8").strip())
    assert loaded["candidate_id"] == "candidate-1"


def test_build_candidate_depth_feature_frames_splits_first_snapshot_from_window() -> None:
    records = pd.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "candidate_id": "c1",
                "capture_time_utc": "2026-06-25T12:00:00+00:00",
                "iteration": 0,
                "spread_pct": 0.01,
                "top_n_bid_depth_usdt": 1000.0,
                "top_n_ask_depth_usdt": 900.0,
                "top_n_depth_min_usdt": 900.0,
                "top_n_depth_total_usdt": 1900.0,
                "book_imbalance_top_n": 0.1,
                "depth_to_position_min": 9.0,
                "partial_fill_capacity_ratio": 0.9,
                "min_side_fill_ratio": 0.9,
                "max_side_impact_bps": 4.0,
                "funding_rate": 0.0001,
                "estimated_abs_funding_pct": 0.00875,
                "round_trip_fee_pct": 0.04,
                "basis_pct": 0.02,
            },
            {
                "symbol": "BTCUSDT",
                "candidate_id": "c1",
                "capture_time_utc": "2026-06-25T12:01:00+00:00",
                "iteration": 1,
                "spread_pct": 0.03,
                "top_n_bid_depth_usdt": 2000.0,
                "top_n_ask_depth_usdt": 1800.0,
                "top_n_depth_min_usdt": 1800.0,
                "top_n_depth_total_usdt": 3800.0,
                "partial_fill_capacity_ratio": 1.8,
                "min_side_fill_ratio": 1.0,
                "max_side_impact_bps": 2.0,
            },
        ]
    )

    exante, diagnostics = build_candidate_depth_feature_frames(records)

    assert len(exante) == 1
    assert len(diagnostics) == 1
    assert exante.loc[0, "depth_feature_capture_time_utc"] == "2026-06-25T12:00:00+00:00"
    assert exante.loc[0, "depth_scan_spread_pct"] == pytest.approx(0.01)
    assert exante.loc[0, "depth_scan_top_n_depth_min_usdt"] == pytest.approx(900.0)
    assert exante.loc[0, "depth_scan_funding_rate"] == pytest.approx(0.0001)
    assert diagnostics.loc[0, "snapshot_count"] == 2
    assert diagnostics.loc[0, "min_top_n_depth_min_usdt"] == pytest.approx(900.0)
    assert diagnostics.loc[0, "max_spread_pct"] == pytest.approx(0.03)
    assert diagnostics.loc[0, "partial_fill_risk_snapshots"] == 1
