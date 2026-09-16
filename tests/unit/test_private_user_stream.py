from __future__ import annotations

import json
from pathlib import Path

import pytest

from neutralgrid.data.private_user_stream import (
    LINKAGE_SCHEMA_VERSION,
    PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION,
    ExactOrderLinkage,
    PrivateStrategyStorage,
    PrivateUserPayloadError,
    PrivateUserStreamError,
    StrategyTarget,
    extract_private_event_symbols,
    load_exact_order_linkages,
    normalize_private_user_event,
    redact_secret,
)
from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
)


TARGET = StrategyTarget("BTCUSDT", "413500001")


def _order_event(
    *,
    order_id: int = 70,
    trade_id: int = 7,
    execution_type: str = "TRADE",
) -> dict[str, object]:
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1_700_000_000_100,
        "T": 1_700_000_000_090,
        "o": {
            "s": "BTCUSDT",
            "c": "grid-order-1",
            "S": "BUY",
            "x": execution_type,
            "X": "FILLED" if execution_type == "TRADE" else "NEW",
            "i": order_id,
            "l": "2",
            "z": "2",
            "L": "100.5",
            "N": "USDT",
            "n": "0.01",
            "T": 1_700_000_000_090,
            "t": trade_id,
            "m": True,
            "rp": "0.25",
        },
    }


def _linkage_file(
    path: Path,
    *,
    strategy_id: str = "413500001",
    order_ids: list[str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": LINKAGE_SCHEMA_VERSION,
                "symbol": "BTCUSDT",
                "strategy_id": strategy_id,
                "order_ids": order_ids or ["70"],
                "provenance": "reviewed_grid_order_export",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_exact_order_linkage_is_roster_bound(tmp_path: Path) -> None:
    lookup, loaded = load_exact_order_linkages(
        [_linkage_file(tmp_path / "linkage.json")],
        expected_targets=[TARGET],
    )

    assert lookup == {("BTCUSDT", "70"): TARGET}
    assert loaded == (
        ExactOrderLinkage(
            target=TARGET,
            order_ids=frozenset({"70"}),
            provenance="reviewed_grid_order_export",
            source_path=(tmp_path / "linkage.json").resolve(),
        ),
    )


def test_linkage_rejects_inactive_target_and_collision(tmp_path: Path) -> None:
    inactive = _linkage_file(
        tmp_path / "inactive.json", strategy_id="413500999"
    )
    with pytest.raises(PrivateUserStreamError, match="not in the exact active roster"):
        load_exact_order_linkages([inactive], expected_targets=[TARGET])

    second_target = StrategyTarget("BTCUSDT", "413500002")
    first = _linkage_file(tmp_path / "first.json")
    second = _linkage_file(
        tmp_path / "second.json", strategy_id=second_target.strategy_id
    )
    with pytest.raises(PrivateUserStreamError, match="collision"):
        load_exact_order_linkages(
            [first, second], expected_targets=[TARGET, second_target]
        )


def test_order_update_and_fill_require_exact_order_linkage() -> None:
    result = normalize_private_user_event(
        _order_event(),
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={("BTCUSDT", "70"): TARGET},
    )

    assert result.classification == "accepted_exact_order_linkage"
    assert result.target == TARGET
    assert [record["event_type"] for record in result.canonical_records] == [
        "order_update",
        "trade_fill",
    ]
    fill = result.canonical_records[1]
    assert fill["trade_id"] == "7"
    assert fill["order_id"] == "70"
    assert fill["price"] == 100.5
    assert fill["qty"] == 2.0
    assert fill["commission_usdt"] == -0.01
    assert fill["realized_pnl_usdt"] == 0.25

    unlinked = normalize_private_user_event(
        _order_event(order_id=71),
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={("BTCUSDT", "70"): TARGET},
    )
    assert unlinked.classification == "unlinked_order_preserved_raw_only"
    assert unlinked.canonical_records == ()


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (_order_event(order_id=0), "i must be positive"),
        (_order_event(trade_id=0), "t must be positive"),
        (dict(_order_event(), E=10**30), "supported timestamp range"),
    ],
)
def test_private_event_rejects_invalid_exchange_identity_or_time(
    event: dict[str, object], message: str
) -> None:
    with pytest.raises(PrivateUserPayloadError, match=message):
        normalize_private_user_event(
            event,
            run_id="private-run",
            exact_targets=[TARGET],
            order_lookup={("BTCUSDT", "70"): TARGET},
        )


def test_strategy_update_requires_exact_symbol_and_strategy() -> None:
    event = {
        "e": "STRATEGY_UPDATE",
        "E": 1_700_000_000_100,
        "T": 1_700_000_000_090,
        "su": {
            "s": "BTCUSDT",
            "si": "413500001",
            "ss": "WORKING",
            "c": 8007,
        },
    }
    accepted = normalize_private_user_event(
        event,
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={},
    )
    assert accepted.classification == "accepted_exact_strategy_identity"
    assert accepted.canonical_records[0]["event_type"] == "account_update"
    assert accepted.canonical_records[0]["strategy_status"] == "WORKING"

    event["su"] = dict(event["su"], si="413500999")
    inactive = normalize_private_user_event(
        event,
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={},
    )
    assert inactive.classification == "inactive_strategy_preserved_raw_only"
    assert inactive.canonical_records == ()


def test_account_event_is_preserved_but_not_assigned_by_symbol() -> None:
    event = {
        "e": "ACCOUNT_UPDATE",
        "E": 1_700_000_000_100,
        "T": 1_700_000_000_090,
        "a": {"m": "ORDER", "P": [{"s": "BTCUSDT", "pa": "1"}]},
    }
    assert extract_private_event_symbols(event) == ("BTCUSDT",)

    result = normalize_private_user_event(
        event,
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={},
    )
    assert result.classification == "account_or_unsupported_event_preserved_raw_only"
    assert result.target is None
    assert result.canonical_records == ()


def test_explicit_funding_and_margin_call_symbols_are_extracted() -> None:
    assert extract_private_event_symbols(
        {
            "e": "ACCOUNT_UPDATE",
            "E": 1_700_000_000_100,
            "a": {"m": "FUNDING_FEE", "S": "ETHUSDT", "P": []},
        }
    ) == ("ETHUSDT",)
    assert extract_private_event_symbols(
        {
            "e": "MARGIN_CALL",
            "E": 1_700_000_000_100,
            "p": [{"s": "BTCUSDT"}, {"s": "ETHUSDT"}],
        }
    ) == ("BTCUSDT", "ETHUSDT")


def test_listen_key_is_redacted_before_persistence() -> None:
    original = '{"e":"listenKeyExpired","listenKey":"top-secret"}'
    redacted, changed = redact_secret(original, "top-secret")

    assert changed is True
    assert "top-secret" not in redacted
    assert "<REDACTED_LISTEN_KEY>" in redacted


def test_private_storage_deduplicates_and_rejects_conflict(tmp_path: Path) -> None:
    storage = PrivateStrategyStorage(
        tmp_path / "run",
        target=TARGET,
        run_id="private-run",
    )
    result = normalize_private_user_event(
        _order_event(),
        run_id="private-run",
        exact_targets=[TARGET],
        order_lookup={("BTCUSDT", "70"): TARGET},
    )
    order_record = result.canonical_records[0]
    assert storage.append_canonical(order_record) == "appended"
    assert storage.append_canonical(order_record) == "duplicate"
    conflicting = dict(order_record, client_order_id="conflict")
    with pytest.raises(PrivateUserStreamError, match="conflicting canonical"):
        storage.append_canonical(conflicting)
    storage.write_manifest(
        {"status": "running", "updated_at_utc": "2026-09-11T01:00:00+00:00"}
    )
    storage.close()

    manifest = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION
    assert manifest["event_completeness"] == "unknown"
    assert manifest["total_records"] == 1
    assert manifest["duplicate_records_dropped"] == 1
    assert manifest["rejected_records"] == 1
    assert PRIVATE_SERVICE_MANIFEST_SCHEMA_VERSION != manifest["schema_version"]


def test_private_storage_never_persists_listen_key(tmp_path: Path) -> None:
    storage = PrivateStrategyStorage(
        tmp_path / "run",
        target=TARGET,
        run_id="private-run",
    )
    original = '{"e":"listenKeyExpired","listenKey":"top-secret"}'
    persisted, redacted = redact_secret(original, "top-secret")
    storage.append_wire(
        connection_id="private-1",
        wire_sequence=1,
        received_at_utc="2026-09-11T01:00:00+00:00",
        received_monotonic_ns=123,
        original_sha256="a" * 64,
        persisted_text=persisted,
        credential_redacted=redacted,
    )
    storage.close()

    wire = (tmp_path / "run" / "wire_events.jsonl").read_text(encoding="utf-8")
    assert "top-secret" not in wire
    assert "REDACTED_LISTEN_KEY" in wire
