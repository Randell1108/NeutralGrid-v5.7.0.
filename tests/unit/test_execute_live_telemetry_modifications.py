from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable

import pytest

from scripts import execute_live_telemetry_modifications as executor


UTC = timezone.utc
NOW = datetime(2026, 7, 31, 16, 0, tzinfo=UTC)


def _key(material: str = "approved-adjustment") -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _intent_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": executor.INTENT_SCHEMA,
        "iteration_id": "iteration-1",
        "idempotency_key": _key(),
        "symbol": "SKYAIUSDT",
        "strategy_id": "413527269",
        "action": "ADJUST",
        "suggested_grid_lower": 0.023595,
        "suggested_grid_upper": 0.025825,
        "decision_ts": (NOW - timedelta(seconds=20)).isoformat(),
    }
    payload.update(overrides)
    return payload


def _approval_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": executor.APPROVAL_SCHEMA,
        "idempotency_key": _key(),
        "symbol": "SKYAIUSDT",
        "strategy_id": "413527269",
        "action": "ADJUST",
        "suggested_grid_lower": 0.023595,
        "suggested_grid_upper": 0.025825,
        "preserve_current_position": True,
        "approved_at_utc": (NOW - timedelta(seconds=10)).isoformat(),
        "expires_at_utc": (NOW + timedelta(seconds=170)).isoformat(),
    }
    payload.update(overrides)
    return payload


def _intent() -> executor.ActionIntent:
    return executor.parse_intent(
        _intent_payload(),
        now=NOW,
        max_age_seconds=180,
    )


def _approval() -> executor.ActionApproval:
    return executor.parse_approval(
        _approval_payload(),
        now=NOW,
        max_lifetime_seconds=300,
    )


class FakeDriver:
    def __init__(
        self,
        *,
        current: executor.BotState | None = None,
        prepared: executor.PreparedAdjustment | None = None,
        post_states: list[executor.BotState] | None = None,
        on_read: Callable[[], None] | None = None,
    ) -> None:
        self.current = current or executor.BotState(
            symbol="SKYAIUSDT",
            strategy_id="413527269",
            status="Working",
            lower=Decimal("0.02479"),
            upper=Decimal("0.02702"),
            num_grids=12,
            position_summary="5,230 SKYAI long",
        )
        self.prepared = prepared or executor.PreparedAdjustment(
            symbol="SKYAIUSDT",
            strategy_id="413527269",
            lower=Decimal("0.02359"),
            upper=Decimal("0.02583"),
            num_grids=12,
            original_num_grids=12,
            preserve_current_position=True,
            additional_investment=Decimal("0.00"),
            price_decimals=5,
            confirm_enabled=True,
        )
        self.post_states = post_states or [
            executor.BotState(
                symbol="SKYAIUSDT",
                strategy_id="413527269",
                status="Working",
                lower=Decimal("0.02359"),
                upper=Decimal("0.02583"),
                num_grids=12,
                position_summary="5,230 SKYAI long",
            )
        ]
        self.on_read = on_read
        self.read_calls = 0
        self.prepare_calls = 0
        self.submit_calls = 0
        self.wait_calls = 0

    def read_state(
        self,
        _intent: executor.ActionIntent,
        _deadline: executor.Deadline,
    ) -> executor.BotState:
        if self.on_read is not None:
            self.on_read()
        self.read_calls += 1
        if self.read_calls == 1:
            return self.current
        index = min(self.read_calls - 2, len(self.post_states) - 1)
        return self.post_states[index]

    def prepare(
        self,
        _intent: executor.ActionIntent,
        _current: executor.BotState,
        _deadline: executor.Deadline,
    ) -> executor.PreparedAdjustment:
        self.prepare_calls += 1
        return self.prepared

    def submit_once(
        self,
        _prepared: executor.PreparedAdjustment,
        _deadline: executor.Deadline,
    ) -> None:
        self.submit_calls += 1

    def wait_before_post_verify(self, _deadline: executor.Deadline) -> None:
        self.wait_calls += 1

    def close(self) -> None:
        pass


class FakeExtensionBridge:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.submitted = False

    def call(
        self,
        action: str,
        payload: dict[str, object],
        *,
        deadline: executor.Deadline,
    ) -> dict[str, object]:
        deadline.require(action)
        self.actions.append(action)
        if action == "hello":
            return {
                "provider": "chrome-extension",
                "url": "https://www.binance.bh/en/trading-bots/futures/grid/SKYAIUSDT",
            }
        if action == "read_state":
            lower = "0.02359" if self.submitted else "0.02479"
            upper = "0.02583" if self.submitted else "0.02702"
            return {
                "raw_text": "\n".join(
                    [
                        "SKYAIUSDT",
                        "Working",
                        "Total Profit (USDT)",
                        "0.00 (0.00%)",
                        "Positions",
                        "5,230 SKYAI Long",
                        "Pending Order",
                        "Grid Details",
                        "Price Range",
                        f"{lower} - {upper} USDT",
                        "Number of Grids",
                        "12",
                        "Order History",
                        "Strategy Number",
                        "413527269",
                    ]
                )
            }
        if action == "open_modify_form":
            return {"lower": "0.02479", "upper": "0.02702", "grids": "12"}
        if action == "ensure_keep_position":
            return {"close_positions": "No"}
        if action == "set_form_inputs":
            assert payload == {"lower": "0.02359", "upper": "0.02583"}
            return {"updated": True}
        if action == "read_form":
            return {
                "lower": "0.02359",
                "upper": "0.02583",
                "grids": "12",
                "close_positions": "No",
                "additional_investment": "0.00",
                "confirm_enabled": True,
            }
        if action == "submit":
            assert not self.submitted
            self.submitted = True
            return {"acknowledged": True}
        if action == "wait":
            return {"waited_seconds": payload["seconds"]}
        raise AssertionError(f"unexpected bridge action: {action}")

    def close(self) -> None:
        pass


def test_parse_intent_rejects_stale_or_end() -> None:
    stale = _intent_payload(decision_ts=(NOW - timedelta(seconds=181)).isoformat())
    with pytest.raises(executor.ExecutionError, match="stale"):
        executor.parse_intent(stale, now=NOW, max_age_seconds=180)

    with pytest.raises(executor.ExecutionError, match="only ADJUST"):
        executor.parse_intent(
            _intent_payload(action="END"),
            now=NOW,
            max_age_seconds=180,
        )


def test_approval_must_match_exactly_and_preserve_position() -> None:
    intent = _intent()
    mismatched = executor.parse_approval(
        _approval_payload(suggested_grid_upper=0.0259),
        now=NOW,
        max_lifetime_seconds=300,
    )
    with pytest.raises(executor.ExecutionError, match="suggested_grid_upper"):
        executor.validate_approval(intent, mismatched)

    with pytest.raises(executor.ExecutionError, match="preserve"):
        executor.parse_approval(
            _approval_payload(preserve_current_position=False),
            now=NOW,
            max_lifetime_seconds=300,
        )


def test_outward_round_bounds_never_narrows_scanner_range() -> None:
    lower, upper = executor.outward_round_bounds(
        Decimal("0.023595"),
        Decimal("0.025825"),
        decimals=5,
    )

    assert lower == Decimal("0.02359")
    assert upper == Decimal("0.02583")


def test_extension_bridge_rejects_non_loopback_or_invalid_token() -> None:
    with pytest.raises(executor.ExecutionError, match="loopback"):
        executor.ExtensionBridgeClient(
            endpoint="https://example.com",
            token=_key(),
            command_timeout_seconds=4,
        )
    with pytest.raises(executor.ExecutionError, match="64 lowercase hex"):
        executor.ExtensionBridgeClient(
            endpoint="http://127.0.0.1:17731",
            token="not-a-token",
            command_timeout_seconds=4,
        )


def test_extension_driver_uses_same_guarded_state_machine() -> None:
    driver = executor.BinanceExtensionAdjustmentDriver(
        bridge_endpoint="http://127.0.0.1:17731",
        bridge_token=_key("bridge"),
        symbol="SKYAIUSDT",
        command_timeout_seconds=4,
    )
    bridge = FakeExtensionBridge()
    driver._bridge = bridge

    result = executor.execute_adjustment(
        _intent(),
        _approval(),
        driver,
        policy=executor.ExecutionPolicy(),
    )

    assert result["status"] == "executed"
    assert result["executed_grid_lower"] == 0.02359
    assert result["executed_grid_upper"] == 0.02583
    assert bridge.actions.count("submit") == 1
    assert bridge.actions[:3] == ["hello", "read_state", "open_modify_form"]


def test_execute_adjustment_submits_once_and_verifies_exact_strategy() -> None:
    driver = FakeDriver()
    fences: list[executor.PreparedAdjustment] = []

    result = executor.execute_adjustment(
        _intent(),
        _approval(),
        driver,
        policy=executor.ExecutionPolicy(),
        before_submit=fences.append,
    )

    assert result["status"] == "executed"
    assert result["executed_grid_lower"] == 0.02359
    assert result["executed_grid_upper"] == 0.02583
    assert result["preserve_current_position"] is True
    assert driver.prepare_calls == 1
    assert driver.submit_calls == 1
    assert driver.read_calls == 2
    assert len(fences) == 1


def test_execute_adjustment_rejects_identity_mismatch_before_prepare() -> None:
    driver = FakeDriver(
        current=executor.BotState(
            symbol="SKYAIUSDT",
            strategy_id="413527999",
            status="Working",
            lower=Decimal("0.02479"),
            upper=Decimal("0.02702"),
            num_grids=12,
            position_summary="position",
        )
    )

    with pytest.raises(executor.ExecutionError, match="identity mismatch"):
        executor.execute_adjustment(
            _intent(),
            _approval(),
            driver,
            policy=executor.ExecutionPolicy(),
        )

    assert driver.prepare_calls == 0
    assert driver.submit_calls == 0


def test_execute_adjustment_rejects_position_close_before_submit() -> None:
    prepared = executor.PreparedAdjustment(
        symbol="SKYAIUSDT",
        strategy_id="413527269",
        lower=Decimal("0.02359"),
        upper=Decimal("0.02583"),
        num_grids=12,
        original_num_grids=12,
        preserve_current_position=False,
        additional_investment=Decimal("0.00"),
        price_decimals=5,
        confirm_enabled=True,
    )
    driver = FakeDriver(prepared=prepared)

    with pytest.raises(executor.ExecutionError, match="close the current position"):
        executor.execute_adjustment(
            _intent(),
            _approval(),
            driver,
            policy=executor.ExecutionPolicy(),
        )

    assert driver.submit_calls == 0


def test_execution_deadline_reserves_post_submit_budget() -> None:
    clock_value = [0.0]

    def clock() -> float:
        return clock_value[0]

    driver = FakeDriver(on_read=lambda: clock_value.__setitem__(0, 8.0))

    with pytest.raises(executor.ExecutionError, match="form preparation"):
        executor.execute_adjustment(
            _intent(),
            _approval(),
            driver,
            policy=executor.ExecutionPolicy(
                deadline_seconds=10,
                min_post_submit_seconds=3,
            ),
            clock=clock,
        )

    assert driver.prepare_calls == 0
    assert driver.submit_calls == 0


def test_post_verification_has_one_bounded_read_retry() -> None:
    old_state = executor.BotState(
        symbol="SKYAIUSDT",
        strategy_id="413527269",
        status="Working",
        lower=Decimal("0.02479"),
        upper=Decimal("0.02702"),
        num_grids=12,
        position_summary="position",
    )
    new_state = executor.BotState(
        symbol="SKYAIUSDT",
        strategy_id="413527269",
        status="Working",
        lower=Decimal("0.02359"),
        upper=Decimal("0.02583"),
        num_grids=12,
        position_summary="position",
    )
    driver = FakeDriver(post_states=[old_state, new_state])

    result = executor.execute_adjustment(
        _intent(),
        _approval(),
        driver,
        policy=executor.ExecutionPolicy(post_verify_attempts=2),
    )

    assert result["status"] == "executed"
    assert driver.submit_calls == 1
    assert driver.wait_calls == 1
    assert driver.read_calls == 3


def test_parse_bot_state_extracts_identity_range_and_grid_count() -> None:
    raw = "\n".join(
        [
            "SKYAIUSDT",
            "Perp",
            "Futures Grid",
            "Working",
            "Total Profit (USDT)",
            "0.00 (0.00%)",
            "Positions",
            "5,230 SKYAI Long",
            "Pending Order",
            "Grid Details",
            "Price Range",
            "0.02359 - 0.02583 USDT",
            "Number of Grids",
            "12",
            "Order History",
            "Strategy Number",
            "413527269",
        ]
    )

    state = executor._parse_bot_state(raw, expected_symbol="SKYAIUSDT")

    assert state.strategy_id == "413527269"
    assert state.lower == Decimal("0.02359")
    assert state.upper == Decimal("0.02583")
    assert state.num_grids == 12
    assert state.position_summary == "5,230 SKYAI Long"


def test_main_contract_writes_submit_fence_and_single_json_response(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    approval_path = tmp_path / "approval.json"
    ledger_path = tmp_path / "execution.jsonl"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    driver = FakeDriver()
    monkeypatch.setattr(executor, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        executor,
        "BinanceCdpAdjustmentDriver",
        lambda **_kwargs: driver,
    )
    monkeypatch.setattr(
        executor.sys,
        "stdin",
        io.StringIO(json.dumps(_intent_payload())),
    )

    returncode = executor.main(
        [
            "--approval-file",
            str(approval_path),
            "--audit-ledger",
            str(ledger_path),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    audit_rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert returncode == 0
    assert response["status"] == "executed"
    assert [row["status"] for row in audit_rows] == ["submit_started", "executed"]
    assert driver.submit_calls == 1


def test_main_selects_extension_transport(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    approval_path = tmp_path / "approval.json"
    token_path = tmp_path / "bridge.token"
    ledger_path = tmp_path / "execution.jsonl"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    token_path.write_text(_key("bridge"), encoding="utf-8")
    driver = FakeDriver()
    created: list[dict[str, object]] = []

    def build_extension_driver(**kwargs: object) -> FakeDriver:
        created.append(kwargs)
        return driver

    monkeypatch.setattr(executor, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        executor,
        "BinanceExtensionAdjustmentDriver",
        build_extension_driver,
    )
    monkeypatch.setattr(
        executor.sys,
        "stdin",
        io.StringIO(json.dumps(_intent_payload())),
    )

    returncode = executor.main(
        [
            "--approval-file",
            str(approval_path),
            "--audit-ledger",
            str(ledger_path),
            "--browser-transport",
            "extension",
            "--extension-token-file",
            str(token_path),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert returncode == 0
    assert response["status"] == "executed"
    assert len(created) == 1
    assert created[0]["bridge_token"] == _key("bridge")
    assert driver.submit_calls == 1


def test_extension_transport_requires_token_file() -> None:
    with pytest.raises(SystemExit):
        executor.parse_args(
            [
                "--approval-file",
                "approval.json",
                "--browser-transport",
                "extension",
            ]
        )


def test_main_blocks_reserved_idempotency_key_without_browser(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    approval_path = tmp_path / "approval.json"
    ledger_path = tmp_path / "execution.jsonl"
    approval_path.write_text(json.dumps(_approval_payload()), encoding="utf-8")
    ledger_path.write_text(
        json.dumps(
            {
                "status": "submit_started",
                "idempotency_key": _key(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(executor, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        executor.sys,
        "stdin",
        io.StringIO(json.dumps(_intent_payload())),
    )

    returncode = executor.main(
        [
            "--approval-file",
            str(approval_path),
            "--audit-ledger",
            str(ledger_path),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert returncode == 2
    assert response["status"] == "blocked"
    assert "already reserved" in response["error"]
