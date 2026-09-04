from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.ingest_private_execution_events import (
    PrivateExecutionIngestError,
    ingest_private_execution_export,
    normalize_strategy_trades,
)
from neutralgrid.live.decision.private_events import (
    PrivateEventStreamRef,
    load_private_event_evidence,
)


UTC = timezone.utc
NOW = datetime(2026, 8, 10, 19, 0, tzinfo=UTC)


def _trade(
    trade_id: int,
    order_id: int,
    *,
    side: str = "BUY",
    commission_asset: str = "USDT",
) -> dict[str, object]:
    return {
        "id": trade_id,
        "orderId": order_id,
        "symbol": "GRVTUSDT",
        "side": side,
        "price": "0.05",
        "qty": "100",
        "realizedPnl": "0.25",
        "commission": "0.01",
        "commissionAsset": commission_asset,
        "time": int((NOW - timedelta(minutes=1)).timestamp() * 1000),
        "maker": True,
    }


def test_normalizer_keeps_only_exact_linked_orders_and_deduplicates() -> None:
    records, audit = normalize_strategy_trades(
        [_trade(1, 10), _trade(1, 10), _trade(2, 99)],
        symbol="GRVTUSDT",
        strategy_id="strategy-1",
        run_id="run-1",
        allowed_order_ids={"10"},
    )

    assert len(records) == 1
    assert records[0]["strategy_id"] == "strategy-1"
    assert records[0]["trade_id"] == "1"
    assert records[0]["order_id"] == "10"
    assert audit == {
        "input_trades": 3,
        "linked_trades": 1,
        "unlinked_trades_filtered": 1,
        "duplicate_trades_dropped": 1,
    }


def test_ingested_export_round_trips_through_private_event_consumer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trades.json"
    source.write_text(json.dumps([_trade(1, 10), _trade(2, 11, side="SELL")]), encoding="utf-8")
    linkage = tmp_path / "linkage.json"
    linkage.write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_strategy_order_linkage_v1",
                "symbol": "GRVTUSDT",
                "strategy_id": "strategy-1",
                "order_ids": ["10", "11"],
                "provenance": "reviewed_grid_order_export",
            }
        ),
        encoding="utf-8",
    )

    result = ingest_private_execution_export(
        source_path=source,
        linkage_path=linkage,
        live_root=tmp_path / "Live",
        run_id="run-1",
        event_completeness="history_complete",
        window_start_utc=NOW - timedelta(hours=1),
        window_end_utc=NOW,
        observed_at_utc=NOW,
    )

    assert result["status"] == "complete"
    assert result["linked_trades"] == 2
    manifest_path = Path(result["manifest_path"])
    event_path = Path(result["event_path"])
    assert "2026-08-10" in manifest_path.parts
    evidence = load_private_event_evidence(
        PrivateEventStreamRef(
            event_path=event_path,
            manifest_path=manifest_path,
            symbol="GRVTUSDT",
            strategy_id="strategy-1",
            run_id="run-1",
            max_age_seconds=60,
            history_window_seconds=3600,
        ),
        now=NOW,
    )
    assert evidence.trade_fill_count == 2
    assert evidence.realized_pnl_usdt == 0.5
    assert evidence.commission_usdt == 0.02
    assert evidence.event_completeness == "history_complete"


def test_missing_or_ambiguous_strategy_order_linkage_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "trades.json"
    source.write_text(json.dumps([_trade(1, 10)]), encoding="utf-8")
    linkage = tmp_path / "linkage.json"
    linkage.write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_strategy_order_linkage_v1",
                "symbol": "GRVTUSDT",
                "strategy_id": "strategy-1",
                "order_ids": [],
                "provenance": "symbol_and_time_only",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PrivateExecutionIngestError, match="ambiguous|order_ids"):
        ingest_private_execution_export(
            source_path=source,
            linkage_path=linkage,
            live_root=tmp_path / "Live",
            run_id="run-1",
            event_completeness="snapshot_only",
            window_start_utc=None,
            window_end_utc=None,
            observed_at_utc=NOW,
        )


def test_static_import_cannot_claim_event_complete(tmp_path: Path) -> None:
    source = tmp_path / "trades.json"
    source.write_text(json.dumps([_trade(1, 10)]), encoding="utf-8")
    linkage = tmp_path / "linkage.json"
    linkage.write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_strategy_order_linkage_v1",
                "symbol": "GRVTUSDT",
                "strategy_id": "strategy-1",
                "order_ids": ["10"],
                "provenance": "reviewed_grid_order_export",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PrivateExecutionIngestError, match="event_complete"):
        ingest_private_execution_export(
            source_path=source,
            linkage_path=linkage,
            live_root=tmp_path / "Live",
            run_id="run-1",
            event_completeness="event_complete",
            window_start_utc=NOW - timedelta(hours=1),
            window_end_utc=NOW,
            observed_at_utc=NOW,
        )


@pytest.mark.parametrize(
    ("linkage_overrides", "run_id", "error"),
    [
        ({"symbol": "../GRVTUSDT"}, "run-1", "symbol"),
        ({"strategy_id": "../strategy-1"}, "run-1", "strategy_id"),
        ({}, "../run-1", "run_id"),
    ],
)
def test_output_identity_components_cannot_escape_live_root(
    tmp_path: Path,
    linkage_overrides: dict[str, str],
    run_id: str,
    error: str,
) -> None:
    source = tmp_path / "trades.json"
    source.write_text(json.dumps([_trade(1, 10)]), encoding="utf-8")
    linkage_payload = {
        "schema_version": "neutralgrid_strategy_order_linkage_v1",
        "symbol": "GRVTUSDT",
        "strategy_id": "strategy-1",
        "order_ids": ["10"],
        "provenance": "reviewed_grid_order_export",
        **linkage_overrides,
    }
    linkage = tmp_path / "linkage.json"
    linkage.write_text(json.dumps(linkage_payload), encoding="utf-8")

    with pytest.raises(PrivateExecutionIngestError, match=error):
        ingest_private_execution_export(
            source_path=source,
            linkage_path=linkage,
            live_root=tmp_path / "Live",
            run_id=run_id,
            event_completeness="snapshot_only",
            window_start_utc=None,
            window_end_utc=None,
            observed_at_utc=NOW,
        )

    assert not (tmp_path / "Live").exists()


def test_naive_or_future_execution_time_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "trades.json"
    future_trade = _trade(1, 10)
    future_trade["time"] = int((NOW + timedelta(minutes=1)).timestamp() * 1000)
    source.write_text(json.dumps([future_trade]), encoding="utf-8")
    linkage = tmp_path / "linkage.json"
    linkage.write_text(
        json.dumps(
            {
                "schema_version": "neutralgrid_strategy_order_linkage_v1",
                "symbol": "GRVTUSDT",
                "strategy_id": "strategy-1",
                "order_ids": ["10"],
                "provenance": "reviewed_grid_order_export",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PrivateExecutionIngestError, match="timezone-aware"):
        ingest_private_execution_export(
            source_path=source,
            linkage_path=linkage,
            live_root=tmp_path / "Live",
            run_id="run-naive",
            event_completeness="history_complete",
            window_start_utc=datetime(2026, 8, 10, 18, 0),
            window_end_utc=NOW,
            observed_at_utc=NOW,
        )

    with pytest.raises(PrivateExecutionIngestError, match="after observed_at_utc"):
        ingest_private_execution_export(
            source_path=source,
            linkage_path=linkage,
            live_root=tmp_path / "Live",
            run_id="run-future",
            event_completeness="snapshot_only",
            window_start_utc=None,
            window_end_utc=None,
            observed_at_utc=NOW,
        )
