"""Execute one approved Binance Futures Grid adjustment, then verify it.

The script is the reviewed action executable for
``scripts/run_live_telemetry_controller.py``.  It reads exactly one
``neutralgrid_action_intent_v1`` JSON object from stdin and writes exactly one
JSON response to stdout.  Operational logs go to stderr.

Safety contract
---------------
* ADJUST only; END is deliberately unsupported.
* The scanner decision and a separate, exact approval must both be fresh.
* Symbol, strategy ID, bounds, action, and idempotency key must all match.
* One Chrome grid tab is required through either dedicated CDP or an
  authenticated extension bridge; the exact strategy is re-read.
* Current positions are explicitly preserved and verified in the final form.
* Binance price precision is handled by outward rounding only.
* The submit control is clicked at most once.
* A global wall-clock deadline and a reserved post-submit verification budget
  prevent a slow browser session from consuming the action window.
* Success is emitted only after the same active strategy shows the applied
  bounds in a newly opened View Details drawer.

Approval file example::

    {
      "schema_version": "neutralgrid_action_approval_v1",
      "idempotency_key": "<64 hex characters>",
      "symbol": "BTCUSDT",
      "strategy_id": "413500001",
      "action": "ADJUST",
      "suggested_grid_lower": 60000.0,
      "suggested_grid_upper": 70000.0,
      "preserve_current_position": true,
      "approved_at_utc": "2026-07-31T16:00:00+00:00",
      "expires_at_utc": "2026-07-31T16:03:00+00:00"
    }

The approval is intentionally external to the scanner intent.  This prevents
an unattended process from manufacturing its own action-time authority.

Extension-controlled Chrome requires ``--browser-transport extension``, a
loopback bridge endpoint, and a one-run token file.  The bridge never replaces
the scanner intent or approval; it only exposes bounded UI operations to this
same state machine.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.collect_private_grid_telemetry import (
    CDPConnection,
    TelemetryError,
    _close_drawer,
    _discover_rows,
    _drawer_visible,
    _fetch_json,
    _find_view_details_button,
    _read_drawer,
    _validate_drawer_text,
)


UTC = timezone.utc
INTENT_SCHEMA = "neutralgrid_action_intent_v1"
APPROVAL_SCHEMA = "neutralgrid_action_approval_v1"
RESPONSE_SCHEMA = "neutralgrid_action_execution_v1"
EXTENSION_BRIDGE_SCHEMA = "neutralgrid_extension_bridge_v1"
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")
STRATEGY_RE = re.compile(r"^\d+$")
KEY_RE = re.compile(r"^[0-9a-f]{64}$")
RANGE_RE = re.compile(
    r"Price Range\s+([0-9][0-9,]*(?:\.[0-9]+)?)\s*-\s*"
    r"([0-9][0-9,]*(?:\.[0-9]+)?)\s+USDT"
)
GRIDS_RE = re.compile(r"Number of Grids\s+(\d+)")

logger = logging.getLogger("execute_live_telemetry_modifications")


class ExecutionError(RuntimeError):
    """Fail-closed execution error."""


@dataclass(frozen=True)
class ActionIntent:
    idempotency_key: str
    symbol: str
    strategy_id: str
    lower: Decimal
    upper: Decimal
    decision_ts: datetime
    iteration_id: str


@dataclass(frozen=True)
class ActionApproval:
    idempotency_key: str
    symbol: str
    strategy_id: str
    lower: Decimal
    upper: Decimal
    preserve_current_position: bool
    approved_at_utc: datetime
    expires_at_utc: datetime


@dataclass(frozen=True)
class BotState:
    symbol: str
    strategy_id: str
    status: str
    lower: Decimal
    upper: Decimal
    num_grids: int
    position_summary: str


@dataclass(frozen=True)
class PreparedAdjustment:
    symbol: str
    strategy_id: str
    lower: Decimal
    upper: Decimal
    num_grids: int
    original_num_grids: int
    preserve_current_position: bool
    additional_investment: Decimal
    price_decimals: int
    confirm_enabled: bool


@dataclass(frozen=True)
class ExecutionPolicy:
    deadline_seconds: float = 90.0
    max_decision_age_seconds: float = 180.0
    max_approval_lifetime_seconds: float = 300.0
    min_post_submit_seconds: float = 15.0
    post_verify_attempts: int = 2


class AdjustmentDriver(Protocol):
    def read_state(self, intent: ActionIntent, deadline: "Deadline") -> BotState:
        """Read the exact live bot state from a newly opened details drawer."""

    def prepare(
        self,
        intent: ActionIntent,
        current: BotState,
        deadline: "Deadline",
    ) -> PreparedAdjustment:
        """Prepare but do not submit the adjustment form."""

    def submit_once(
        self,
        prepared: PreparedAdjustment,
        deadline: "Deadline",
    ) -> None:
        """Click the final submit control exactly once."""

    def wait_before_post_verify(self, deadline: "Deadline") -> None:
        """Bounded wait used only for UI propagation between verification reads."""

    def close(self) -> None:
        """Release browser resources."""


class Deadline:
    """Monotonic global execution deadline."""

    def __init__(
        self,
        seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ExecutionError("deadline_seconds must be finite and positive")
        self._clock = clock
        self._started = clock()
        self._expires = self._started + seconds

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self._expires - self._clock())

    def require(self, stage: str, *, reserve_seconds: float = 0.0) -> None:
        remaining = self.remaining_seconds
        if remaining <= reserve_seconds:
            raise ExecutionError(
                f"execution deadline exhausted before {stage}: "
                f"remaining={remaining:.2f}s reserve={reserve_seconds:.2f}s"
            )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExecutionError(f"{field} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ExecutionError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _parse_decimal(value: Any, *, field: str, positive: bool = True) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ExecutionError(f"{field} must be numeric")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ExecutionError(f"{field} must be numeric") from exc
    if not number.is_finite():
        raise ExecutionError(f"{field} must be finite")
    if positive and number <= 0:
        raise ExecutionError(f"{field} must be positive")
    return number


def parse_intent(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    max_age_seconds: float,
) -> ActionIntent:
    if payload.get("schema_version") != INTENT_SCHEMA:
        raise ExecutionError("unsupported action-intent schema")
    if str(payload.get("action", "")).upper() != "ADJUST":
        raise ExecutionError("only ADJUST is supported; END remains fail-closed")
    symbol = str(payload.get("symbol", "")).upper()
    strategy_id = str(payload.get("strategy_id", ""))
    key = str(payload.get("idempotency_key", ""))
    if SYMBOL_RE.fullmatch(symbol) is None:
        raise ExecutionError(f"invalid symbol: {symbol!r}")
    if STRATEGY_RE.fullmatch(strategy_id) is None:
        raise ExecutionError(f"invalid strategy_id: {strategy_id!r}")
    if KEY_RE.fullmatch(key) is None:
        raise ExecutionError("idempotency_key must be 64 lowercase hex characters")
    lower = _parse_decimal(payload.get("suggested_grid_lower"), field="lower")
    upper = _parse_decimal(payload.get("suggested_grid_upper"), field="upper")
    if lower >= upper:
        raise ExecutionError("suggested lower bound must be below upper bound")
    decision_ts = _parse_timestamp(payload.get("decision_ts"), field="decision_ts")
    age = (now - decision_ts).total_seconds()
    if age < -5:
        raise ExecutionError(f"decision timestamp is {-age:.1f}s in the future")
    if age > max_age_seconds:
        raise ExecutionError(
            f"decision is stale: age={age:.1f}s > max={max_age_seconds:.1f}s"
        )
    iteration_id = str(payload.get("iteration_id", "")).strip()
    if not iteration_id:
        raise ExecutionError("iteration_id is required")
    return ActionIntent(
        idempotency_key=key,
        symbol=symbol,
        strategy_id=strategy_id,
        lower=lower,
        upper=upper,
        decision_ts=decision_ts,
        iteration_id=iteration_id,
    )


def parse_approval(
    payload: Mapping[str, Any],
    *,
    now: datetime,
    max_lifetime_seconds: float,
) -> ActionApproval:
    if payload.get("schema_version") != APPROVAL_SCHEMA:
        raise ExecutionError("unsupported action-approval schema")
    if str(payload.get("action", "")).upper() != "ADJUST":
        raise ExecutionError("approval must authorize ADJUST")
    approved_at = _parse_timestamp(
        payload.get("approved_at_utc"), field="approved_at_utc"
    )
    expires_at = _parse_timestamp(
        payload.get("expires_at_utc"), field="expires_at_utc"
    )
    if expires_at <= approved_at:
        raise ExecutionError("approval expiry must follow approval time")
    lifetime = (expires_at - approved_at).total_seconds()
    if lifetime > max_lifetime_seconds:
        raise ExecutionError(
            f"approval lifetime {lifetime:.1f}s exceeds {max_lifetime_seconds:.1f}s"
        )
    if (approved_at - now).total_seconds() > 5:
        raise ExecutionError("approval timestamp is in the future")
    if now > expires_at:
        raise ExecutionError("approval has expired")
    preserve = payload.get("preserve_current_position")
    if preserve is not True:
        raise ExecutionError("approval must explicitly preserve the current position")
    symbol = str(payload.get("symbol", "")).upper()
    strategy_id = str(payload.get("strategy_id", ""))
    key = str(payload.get("idempotency_key", ""))
    if SYMBOL_RE.fullmatch(symbol) is None:
        raise ExecutionError(f"invalid approval symbol: {symbol!r}")
    if STRATEGY_RE.fullmatch(strategy_id) is None:
        raise ExecutionError(f"invalid approval strategy_id: {strategy_id!r}")
    if KEY_RE.fullmatch(key) is None:
        raise ExecutionError("approval idempotency_key is invalid")
    lower = _parse_decimal(payload.get("suggested_grid_lower"), field="approval lower")
    upper = _parse_decimal(payload.get("suggested_grid_upper"), field="approval upper")
    if lower >= upper:
        raise ExecutionError("approval lower bound must be below upper bound")
    return ActionApproval(
        idempotency_key=key,
        symbol=symbol,
        strategy_id=strategy_id,
        lower=lower,
        upper=upper,
        preserve_current_position=True,
        approved_at_utc=approved_at,
        expires_at_utc=expires_at,
    )


def validate_approval(intent: ActionIntent, approval: ActionApproval) -> None:
    mismatches: list[str] = []
    for field in ("idempotency_key", "symbol", "strategy_id"):
        if getattr(intent, field) != getattr(approval, field):
            mismatches.append(field)
    if intent.lower != approval.lower:
        mismatches.append("suggested_grid_lower")
    if intent.upper != approval.upper:
        mismatches.append("suggested_grid_upper")
    if mismatches:
        raise ExecutionError(
            "approval does not exactly match the action intent: " + ", ".join(mismatches)
        )


def outward_round_bounds(
    lower: Decimal,
    upper: Decimal,
    *,
    decimals: int,
) -> tuple[Decimal, Decimal]:
    if decimals < 0 or decimals > 16:
        raise ExecutionError(f"unsupported Binance price precision: {decimals}")
    quantum = Decimal(1).scaleb(-decimals)
    rounded_lower = lower.quantize(quantum, rounding=ROUND_FLOOR)
    rounded_upper = upper.quantize(quantum, rounding=ROUND_CEILING)
    if rounded_lower <= 0 or rounded_lower >= rounded_upper:
        raise ExecutionError("outward-rounded bounds are invalid")
    return rounded_lower, rounded_upper


def _state_matches_bounds(
    state: BotState,
    lower: Decimal,
    upper: Decimal,
) -> bool:
    return state.lower == lower and state.upper == upper


def _validate_state_identity(state: BotState, intent: ActionIntent) -> None:
    if state.symbol != intent.symbol or state.strategy_id != intent.strategy_id:
        raise ExecutionError(
            "live bot identity mismatch: "
            f"expected={intent.symbol}/{intent.strategy_id}, "
            f"observed={state.symbol}/{state.strategy_id}"
        )
    if state.status != "Working":
        raise ExecutionError(
            f"{intent.symbol}/{intent.strategy_id} is not Working: {state.status!r}"
        )


def execute_adjustment(
    intent: ActionIntent,
    approval: ActionApproval,
    driver: AdjustmentDriver,
    *,
    policy: ExecutionPolicy,
    clock: Callable[[], float] = time.monotonic,
    before_submit: Callable[[PreparedAdjustment], None] | None = None,
) -> dict[str, Any]:
    """Execute the validated state machine using an injected browser driver."""

    validate_approval(intent, approval)
    deadline = Deadline(policy.deadline_seconds, clock=clock)
    deadline.require("live identity read")
    current = driver.read_state(intent, deadline)
    _validate_state_identity(current, intent)

    deadline.require("form preparation", reserve_seconds=policy.min_post_submit_seconds)
    prepared = driver.prepare(intent, current, deadline)
    if prepared.symbol != intent.symbol or prepared.strategy_id != intent.strategy_id:
        raise ExecutionError("prepared form identity does not match the action intent")
    if prepared.original_num_grids != current.num_grids:
        raise ExecutionError("prepared form changed the original grid count")
    if prepared.num_grids != current.num_grids:
        raise ExecutionError("prepared form would change the grid count")
    if not prepared.preserve_current_position:
        raise ExecutionError("prepared form would close the current position")
    if prepared.additional_investment != Decimal("0"):
        raise ExecutionError("prepared form requires unexpected additional investment")
    if not prepared.confirm_enabled:
        raise ExecutionError("prepared form is not valid or confirm is disabled")
    if prepared.lower > intent.lower or prepared.upper < intent.upper:
        raise ExecutionError("prepared bounds are not an outward rounding of the intent")

    deadline.require(
        "single submit",
        reserve_seconds=policy.min_post_submit_seconds,
    )
    if before_submit is not None:
        before_submit(prepared)
    driver.submit_once(prepared, deadline)

    post: BotState | None = None
    verification_errors: list[str] = []
    for attempt in range(policy.post_verify_attempts):
        deadline.require("post-action verification")
        candidate = driver.read_state(intent, deadline)
        try:
            _validate_state_identity(candidate, intent)
            if not _state_matches_bounds(candidate, prepared.lower, prepared.upper):
                raise ExecutionError(
                    "post-action bounds mismatch: "
                    f"expected=({prepared.lower}, {prepared.upper}) "
                    f"observed=({candidate.lower}, {candidate.upper})"
                )
        except ExecutionError as exc:
            verification_errors.append(str(exc))
            if attempt + 1 >= policy.post_verify_attempts:
                raise ExecutionError(
                    "post-action verification failed: " + " | ".join(verification_errors)
                ) from exc
            driver.wait_before_post_verify(deadline)
            continue
        post = candidate
        break
    if post is None:
        raise ExecutionError("post-action verification produced no state")

    return {
        "schema_version": RESPONSE_SCHEMA,
        "idempotency_key": intent.idempotency_key,
        "status": "executed",
        "symbol": intent.symbol,
        "strategy_id": intent.strategy_id,
        "action": "ADJUST",
        "requested_grid_lower": float(intent.lower),
        "requested_grid_upper": float(intent.upper),
        "executed_grid_lower": float(prepared.lower),
        "executed_grid_upper": float(prepared.upper),
        "price_decimals": prepared.price_decimals,
        "num_grids": prepared.num_grids,
        "preserve_current_position": True,
        "additional_investment": 0.0,
        "pre_state": _json_state(current),
        "post_state": _json_state(post),
        "elapsed_seconds": round(deadline.elapsed_seconds, 3),
    }


def _json_state(state: BotState) -> dict[str, Any]:
    payload = asdict(state)
    payload["lower"] = float(state.lower)
    payload["upper"] = float(state.upper)
    return payload


def _parse_bot_state(raw_text: str, *, expected_symbol: str) -> BotState:
    strategy_id, _created_at = _validate_drawer_text(expected_symbol, raw_text)
    range_match = RANGE_RE.search(raw_text)
    grids_match = GRIDS_RE.search(raw_text)
    if range_match is None or grids_match is None:
        raise ExecutionError(f"{expected_symbol}: grid range/count is not parseable")
    lower = Decimal(range_match.group(1).replace(",", ""))
    upper = Decimal(range_match.group(2).replace(",", ""))
    if lower <= 0 or lower >= upper:
        raise ExecutionError(f"{expected_symbol}: live grid bounds are invalid")
    status = "Working" if re.search(r"(?:^|\n)Working(?:\n|$)", raw_text) else "Unknown"
    position_summary = ""
    position_match = re.search(
        r"(?:^|\n)Positions\n(.*?)(?:\nPending Order\n)",
        raw_text,
        flags=re.DOTALL,
    )
    if position_match is not None:
        position_summary = position_match.group(1).strip()
    return BotState(
        symbol=expected_symbol,
        strategy_id=strategy_id,
        status=status,
        lower=lower,
        upper=upper,
        num_grids=int(grids_match.group(1)),
        position_summary=position_summary,
    )


def _js_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _decimal_places(value: str) -> int:
    normalized = value.strip().replace(",", "")
    if "." not in normalized:
        return 0
    return len(normalized.rsplit(".", 1)[1])


class BinanceCdpAdjustmentDriver:
    """Bounded Binance UI driver over a dedicated Chrome CDP endpoint."""

    def __init__(
        self,
        *,
        debug_endpoint: str,
        symbol: str,
        command_timeout_seconds: float,
        ui_timeout_seconds: float,
        hover_seconds: float,
    ) -> None:
        self._symbol = symbol
        self._ui_timeout_seconds = ui_timeout_seconds
        self._hover_seconds = hover_seconds
        self._submitted = False
        websocket_url, page_url = self._find_single_grid_page(
            debug_endpoint,
            timeout_seconds=command_timeout_seconds,
        )
        self._page_url = page_url
        self._cdp = CDPConnection(
            websocket_url,
            timeout_seconds=command_timeout_seconds,
        )

    @staticmethod
    def _find_single_grid_page(
        debug_endpoint: str,
        *,
        timeout_seconds: float,
    ) -> tuple[str, str]:
        pages = _fetch_json(
            f"{debug_endpoint.rstrip('/')}/json/list",
            timeout_seconds=timeout_seconds,
        )
        if not isinstance(pages, list):
            raise ExecutionError("Chrome /json/list did not return a list")
        matches = [
            page
            for page in pages
            if isinstance(page, dict)
            and page.get("type") == "page"
            and "/en/trading-bots/futures/grid/" in str(page.get("url", ""))
            and page.get("webSocketDebuggerUrl")
        ]
        if len(matches) != 1:
            raise ExecutionError(
                "expected exactly one Binance Futures Grid page in dedicated Chrome; "
                f"found {len(matches)}"
            )
        page = matches[0]
        return str(page["webSocketDebuggerUrl"]), str(page.get("url", ""))

    def close(self) -> None:
        self._cdp.close()

    def _poll(
        self,
        predicate: Callable[[], bool],
        *,
        stage: str,
        deadline: Deadline,
        timeout_seconds: float | None = None,
    ) -> None:
        budget = min(
            timeout_seconds or self._ui_timeout_seconds,
            deadline.remaining_seconds,
        )
        expires = time.monotonic() + budget
        last_error: Exception | None = None
        while time.monotonic() < expires:
            deadline.require(stage)
            try:
                if predicate():
                    return
            except (ExecutionError, TelemetryError) as exc:
                last_error = exc
            time.sleep(0.10)
        detail = f": {last_error}" if last_error is not None else ""
        raise ExecutionError(f"{stage} did not complete within {budget:.1f}s{detail}")

    def _ensure_symbol_page(self, deadline: Deadline) -> None:
        parsed = urlparse(self._page_url)
        if parsed.scheme != "https" or parsed.hostname != "www.binance.bh":
            raise ExecutionError(f"unexpected Binance grid origin: {self._page_url}")
        target = f"https://www.binance.bh/en/trading-bots/futures/grid/{self._symbol}"
        current = self._cdp.evaluate("location.href")
        if current != target:
            deadline.require("symbol-page navigation")
            self._cdp.call("Page.enable")
            self._cdp.call("Page.navigate", {"url": target})
        self._poll(
            lambda: self._cdp.evaluate(
                f"location.href === {_js_string(target)} "
                "&& (document.readyState === 'interactive' || document.readyState === 'complete') "
                "&& Array.from(document.querySelectorAll('[role=\"tab\"]'))"
                ".some((element) => (element.textContent || '').trim().startsWith('UM Grid'))"
            )
            is True,
            stage="exact symbol page load",
            deadline=deadline,
            timeout_seconds=min(20.0, deadline.remaining_seconds),
        )
        self._page_url = target

    def _open_exact_drawer(self, intent: ActionIntent, deadline: Deadline) -> BotState:
        self._ensure_symbol_page(deadline)
        if _drawer_visible(self._cdp):
            _close_drawer(
                self._cdp,
                timeout_seconds=min(self._ui_timeout_seconds, deadline.remaining_seconds),
            )
        _expected_count, rows = _discover_rows(self._cdp)
        matches = [row for row in rows if row.symbol == intent.symbol]
        if len(matches) != 1:
            raise ExecutionError(
                f"expected one Working row for {intent.symbol}; found {len(matches)}"
            )
        row = matches[0]
        button = _find_view_details_button(
            self._cdp,
            row,
            hover_seconds=self._hover_seconds,
        )
        self._cdp.click(button["x"], button["y"])
        self._poll(
            lambda: _drawer_visible(self._cdp),
            stage="View Details drawer open",
            deadline=deadline,
        )
        raw_text = _read_drawer(self._cdp)
        state = _parse_bot_state(raw_text, expected_symbol=intent.symbol)
        _validate_state_identity(state, intent)
        return state

    def read_state(self, intent: ActionIntent, deadline: Deadline) -> BotState:
        return self._open_exact_drawer(intent, deadline)

    def _click_unique_text_button(
        self,
        *,
        root_selector: str,
        text: str,
        stage: str,
        deadline: Deadline,
    ) -> None:
        expression = f"""(() => {{
          const roots = Array.from(document.querySelectorAll({_js_string(root_selector)}))
            .filter((element) => {{
              const rect = element.getBoundingClientRect();
              return rect.width > 0 && rect.height > 0;
            }});
          if (roots.length !== 1) return {{error: 'root_count', count: roots.length}};
          const buttons = Array.from(roots[0].querySelectorAll('button')).filter((button) => {{
            const rect = button.getBoundingClientRect();
            return (button.textContent || '').trim() === {_js_string(text)}
              && rect.width > 0 && rect.height > 0;
          }});
          if (buttons.length !== 1) return {{error: 'button_count', count: buttons.length}};
          const rect = buttons[0].getBoundingClientRect();
          return {{x: rect.x + rect.width / 2, y: rect.y + rect.height / 2}};
        }})()"""
        result = self._cdp.evaluate(expression)
        if not isinstance(result, dict) or "x" not in result or "y" not in result:
            raise ExecutionError(f"{stage} is ambiguous: {result}")
        deadline.require(stage)
        self._cdp.click(float(result["x"]), float(result["y"]))

    def _main_modal_text(self) -> str | None:
        value = self._cdp.evaluate(
            """(() => {
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap.data-size-medium'))
                .filter((element) => {
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0;
                });
              return modals.length === 1 ? (modals[0].innerText || '') : null;
            })()"""
        )
        return value if isinstance(value, str) else None

    def _ensure_keep_position(self, deadline: Deadline) -> None:
        text = self._main_modal_text()
        if text is None:
            raise ExecutionError("Modify Parameters modal is unavailable")
        if re.search(r"Close Your Current Positions\s+No(?:\s|$)", text):
            return
        if not re.search(r"Close Your Current Positions\s+Yes(?:\s|$)", text):
            raise ExecutionError("current-position choice is not parseable")

        opener = self._cdp.evaluate(
            """(() => {
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap.data-size-medium'))
                .filter((element) => {
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0;
                });
              if (modals.length !== 1) return {error: 'main_modal_count', count: modals.length};
              const modal = modals[0];
              const leaves = Array.from(modal.querySelectorAll('*')).filter((element) =>
                element.children.length === 0
                && (element.textContent || '').trim() === 'Close Your Current Positions');
              if (leaves.length !== 1) return {error: 'label_count', count: leaves.length};
              const label = leaves[0];
              let row = label.parentElement;
              for (let depth = 0; row && depth < 7; depth += 1, row = row.parentElement) {
                const rowText = (row.innerText || '').trim();
                const rowRect = row.getBoundingClientRect();
                if (!rowText.includes('Close Your Current Positions') || !/\b(?:Yes|No)\b/.test(rowText)) continue;
                if (rowRect.height <= 0 || rowRect.height > 180) continue;
                const candidates = Array.from(row.querySelectorAll('button,[role="button"],svg'))
                  .filter((element) => {
                    const rect = element.getBoundingClientRect();
                    const style = getComputedStyle(element);
                    return rect.width > 0 && rect.height > 0
                      && style.visibility !== 'hidden' && style.display !== 'none';
                  })
                  .map((element) => {
                    const rect = element.getBoundingClientRect();
                    return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2, area: rect.width * rect.height};
                  })
                  .sort((a, b) => b.x - a.x || a.area - b.area);
                if (candidates.length) return candidates[0];
                const valueLeaves = Array.from(row.querySelectorAll('*')).filter((element) =>
                  element.children.length === 0 && (element.textContent || '').trim() === 'Yes');
                if (valueLeaves.length === 1) {
                  const rect = valueLeaves[0].getBoundingClientRect();
                  return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
                }
              }
              return {error: 'opener_not_found'};
            })()"""
        )
        if not isinstance(opener, dict) or "x" not in opener or "y" not in opener:
            raise ExecutionError(f"position-choice control is ambiguous: {opener}")
        self._cdp.click(float(opener["x"]), float(opener["y"]))
        self._poll(
            lambda: self._cdp.evaluate(
                """(() => Array.from(document.querySelectorAll('.bn-modal-wrap'))
                  .filter((element) => {
                    const rect = element.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                  })
                  .some((element) => (element.innerText || '').includes('No, please keep my positions')))()"""
            )
            is True,
            stage="position-choice modal open",
            deadline=deadline,
        )

        choice = self._cdp.evaluate(
            """(() => {
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap'))
                .filter((element) => {
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0
                    && (element.innerText || '').includes('No, please keep my positions');
                });
              if (modals.length !== 1) return {error: 'choice_modal_count', count: modals.length};
              const radios = Array.from(modals[0].querySelectorAll('[role="radio"]'))
                .filter((element) => (element.innerText || '').trim().startsWith('No, please keep my positions'));
              if (radios.length !== 1) return {error: 'keep_radio_count', count: radios.length};
              const rect = radios[0].getBoundingClientRect();
              return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
            })()"""
        )
        if not isinstance(choice, dict) or "x" not in choice or "y" not in choice:
            raise ExecutionError(f"keep-position option is ambiguous: {choice}")
        self._cdp.click(float(choice["x"]), float(choice["y"]))
        choice_confirm = self._cdp.evaluate(
            """(() => {
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap'))
                .filter((element) => {
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0
                    && (element.innerText || '').includes('No, please keep my positions');
                });
              if (modals.length !== 1) return {error: 'choice_modal_count', count: modals.length};
              const buttons = Array.from(modals[0].querySelectorAll('button')).filter((button) =>
                (button.textContent || '').trim() === 'Confirm');
              if (buttons.length !== 1) return {error: 'confirm_count', count: buttons.length};
              const rect = buttons[0].getBoundingClientRect();
              return {x: rect.x + rect.width / 2, y: rect.y + rect.height / 2};
            })()"""
        )
        if (
            not isinstance(choice_confirm, dict)
            or "x" not in choice_confirm
            or "y" not in choice_confirm
        ):
            raise ExecutionError(
                f"position-choice confirm is ambiguous: {choice_confirm}"
            )
        deadline.require("position-choice confirm")
        self._cdp.click(
            float(choice_confirm["x"]),
            float(choice_confirm["y"]),
        )
        self._poll(
            lambda: (
                (modal_text := self._main_modal_text()) is not None
                and re.search(
                    r"Close Your Current Positions\s+No(?:\s|$)",
                    modal_text,
                )
                is not None
            ),
            stage="position preservation selection",
            deadline=deadline,
        )

    def _read_form(self) -> dict[str, Any]:
        value = self._cdp.evaluate(
            r"""(() => {
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap.data-size-medium'))
                .filter((element) => {
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0;
                });
              if (modals.length !== 1) return {error: 'main_modal_count', count: modals.length};
              const modal = modals[0];
              const uniqueInput = (placeholder) => {
                const inputs = Array.from(modal.querySelectorAll('input'))
                  .filter((input) => input.getAttribute('placeholder') === placeholder);
                return inputs.length === 1 ? inputs[0].value : null;
              };
              const gridInputs = Array.from(modal.querySelectorAll('input')).filter((input) =>
                /^2-/.test(input.getAttribute('placeholder') || ''));
              const confirms = Array.from(modal.querySelectorAll('button')).filter((button) =>
                (button.textContent || '').trim() === 'Confirm');
              const text = modal.innerText || '';
              const closeMatch = text.match(/Close Your Current Positions\s+(Yes|No)(?:\s|$)/);
              const investmentMatch = text.match(/Additional Investment\s+([0-9,.]+)\s+USDT/);
              return {
                text,
                lower: uniqueInput('Lower'),
                upper: uniqueInput('Upper'),
                grids: gridInputs.length === 1 ? gridInputs[0].value : null,
                closePositions: closeMatch ? closeMatch[1] : null,
                additionalInvestment: investmentMatch ? investmentMatch[1] : null,
                confirmCount: confirms.length,
                confirmEnabled: confirms.length === 1
                  && !confirms[0].disabled
                  && confirms[0].getAttribute('aria-disabled') !== 'true'
              };
            })()"""
        )
        if not isinstance(value, dict) or value.get("error"):
            raise ExecutionError(f"Modify Parameters form is unavailable: {value}")
        return value

    def _set_form_input(self, placeholder: str, value: str) -> None:
        result = self._cdp.evaluate(
            f"""(() => {{
              const modals = Array.from(document.querySelectorAll('.bn-modal-wrap.data-size-medium'))
                .filter((element) => {{
                  const rect = element.getBoundingClientRect();
                  return rect.width > 0 && rect.height > 0;
                }});
              if (modals.length !== 1) return {{error: 'main_modal_count', count: modals.length}};
              const inputs = Array.from(modals[0].querySelectorAll('input'))
                .filter((input) => input.getAttribute('placeholder') === {_js_string(placeholder)});
              if (inputs.length !== 1) return {{error: 'input_count', count: inputs.length}};
              const input = inputs[0];
              const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
              setter.call(input, {_js_string(value)});
              input.dispatchEvent(new Event('input', {{bubbles: true}}));
              input.dispatchEvent(new Event('change', {{bubbles: true}}));
              input.blur();
              return {{value: input.value}};
            }})()"""
        )
        if not isinstance(result, dict) or result.get("value") != value:
            raise ExecutionError(
                f"failed to set {placeholder} to {value}: observed={result}"
            )

    def prepare(
        self,
        intent: ActionIntent,
        current: BotState,
        deadline: Deadline,
    ) -> PreparedAdjustment:
        deadline.require("Modify Parameters open")
        self._click_unique_text_button(
            root_selector=".bn-trans.data-show.bn-mask.bn-drawer",
            text="Modify Parameters",
            stage="Modify Parameters open",
            deadline=deadline,
        )
        self._poll(
            lambda: self._main_modal_text() is not None,
            stage="Modify Parameters modal open",
            deadline=deadline,
        )
        original = self._read_form()
        lower_text = original.get("lower")
        upper_text = original.get("upper")
        grids_text = original.get("grids")
        if not isinstance(lower_text, str) or not isinstance(upper_text, str):
            raise ExecutionError("current price inputs are unavailable")
        if not isinstance(grids_text, str) or not grids_text.isdigit():
            raise ExecutionError("current grid count input is unavailable")
        decimals = max(_decimal_places(lower_text), _decimal_places(upper_text))
        rounded_lower, rounded_upper = outward_round_bounds(
            intent.lower,
            intent.upper,
            decimals=decimals,
        )
        self._ensure_keep_position(deadline)
        lower_value = format(rounded_lower, f".{decimals}f")
        upper_value = format(rounded_upper, f".{decimals}f")
        self._set_form_input("Lower", lower_value)
        self._set_form_input("Upper", upper_value)
        self._poll(
            lambda: (
                (form := self._read_form()).get("lower") == lower_value
                and form.get("upper") == upper_value
                and form.get("confirmEnabled") is True
            ),
            stage="prepared form validation",
            deadline=deadline,
        )
        final = self._read_form()
        final_grids = str(final.get("grids", ""))
        investment_text = str(final.get("additionalInvestment", "")).replace(",", "")
        try:
            investment = Decimal(investment_text)
        except InvalidOperation as exc:
            raise ExecutionError("additional investment is not parseable") from exc
        return PreparedAdjustment(
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            lower=rounded_lower,
            upper=rounded_upper,
            num_grids=int(final_grids) if final_grids.isdigit() else -1,
            original_num_grids=int(grids_text),
            preserve_current_position=final.get("closePositions") == "No",
            additional_investment=investment,
            price_decimals=decimals,
            confirm_enabled=final.get("confirmEnabled") is True,
        )

    def submit_once(
        self,
        prepared: PreparedAdjustment,
        deadline: Deadline,
    ) -> None:
        if self._submitted:
            raise ExecutionError("final submit was already attempted")
        form = self._read_form()
        expected_lower = format(prepared.lower, f".{prepared.price_decimals}f")
        expected_upper = format(prepared.upper, f".{prepared.price_decimals}f")
        if (
            form.get("lower") != expected_lower
            or form.get("upper") != expected_upper
            or str(form.get("grids")) != str(prepared.num_grids)
            or form.get("closePositions") != "No"
            or form.get("additionalInvestment") not in {"0", "0.0", "0.00"}
            or form.get("confirmEnabled") is not True
        ):
            raise ExecutionError(f"final form verification failed: {form}")
        deadline.require("final submit")
        self._submitted = True
        self._click_unique_text_button(
            root_selector=".bn-modal-wrap.data-size-medium",
            text="Confirm",
            stage="final submit",
            deadline=deadline,
        )
        self._poll(
            lambda: self._main_modal_text() is None,
            stage="final submit acknowledgement",
            deadline=deadline,
            timeout_seconds=min(12.0, deadline.remaining_seconds),
        )

    def wait_before_post_verify(self, deadline: Deadline) -> None:
        deadline.require("post-action propagation wait")
        time.sleep(min(1.0, max(0.0, deadline.remaining_seconds - 0.1)))


class ExtensionBridgeClient:
    """Authenticated loopback RPC client for an extension-controlled Chrome tab."""

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        command_timeout_seconds: float,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ExecutionError(
                "extension bridge endpoint must be an unauthenticated loopback HTTP URL"
            )
        if KEY_RE.fullmatch(token) is None:
            raise ExecutionError("extension bridge token must be 64 lowercase hex characters")
        self._rpc_url = endpoint.rstrip("/") + "/rpc"
        self._token = token
        self._command_timeout_seconds = command_timeout_seconds
        self._closed = False

    def call(
        self,
        action: str,
        payload: Mapping[str, Any],
        *,
        deadline: Deadline,
    ) -> dict[str, Any]:
        deadline.require(f"extension bridge {action}")
        timeout = min(self._command_timeout_seconds, deadline.remaining_seconds)
        return self._request(action, payload, timeout_seconds=timeout)

    def _request(
        self,
        action: str,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "schema_version": EXTENSION_BRIDGE_SCHEMA,
                "action": action,
                "payload": dict(payload),
            }
        ).encode("utf-8")
        request = Request(
            self._rpc_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Neutralgrid-Bridge-Token": self._token,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=max(0.1, timeout_seconds)) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ExecutionError(
                f"extension bridge {action} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise ExecutionError(f"extension bridge {action} failed: {exc}") from exc
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExecutionError(
                f"extension bridge {action} returned invalid JSON"
            ) from exc
        if not isinstance(message, dict):
            raise ExecutionError(f"extension bridge {action} returned a non-object")
        if message.get("schema_version") != EXTENSION_BRIDGE_SCHEMA:
            raise ExecutionError(f"extension bridge {action} schema mismatch")
        if message.get("ok") is not True:
            raise ExecutionError(
                f"extension bridge {action} blocked: {message.get('error', 'unknown error')}"
            )
        result = message.get("result")
        if not isinstance(result, dict):
            raise ExecutionError(f"extension bridge {action} result is not an object")
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._request("shutdown", {}, timeout_seconds=self._command_timeout_seconds)
        except ExecutionError as exc:
            logger.warning("extension bridge shutdown was not acknowledged: %s", exc)


class BinanceExtensionAdjustmentDriver:
    """Binance adjustment driver using a claimed Chrome extension tab."""

    def __init__(
        self,
        *,
        bridge_endpoint: str,
        bridge_token: str,
        symbol: str,
        command_timeout_seconds: float,
    ) -> None:
        self._symbol = symbol
        self._submitted = False
        self._verified = False
        self._bridge = ExtensionBridgeClient(
            endpoint=bridge_endpoint,
            token=bridge_token,
            command_timeout_seconds=command_timeout_seconds,
        )

    def close(self) -> None:
        self._bridge.close()

    def read_state(self, intent: ActionIntent, deadline: Deadline) -> BotState:
        if not self._verified:
            hello = self._bridge.call("hello", {}, deadline=deadline)
            page_url = str(hello.get("url", ""))
            parsed = urlparse(page_url)
            if (
                hello.get("provider") != "chrome-extension"
                or parsed.scheme != "https"
                or parsed.hostname != "www.binance.bh"
                or "/en/trading-bots/futures/grid/" not in parsed.path
            ):
                raise ExecutionError(
                    f"extension bridge is not attached to a Binance grid page: {hello}"
                )
            self._verified = True
        result = self._bridge.call(
            "read_state",
            {
                "symbol": intent.symbol,
                "strategy_id": intent.strategy_id,
            },
            deadline=deadline,
        )
        raw_text = result.get("raw_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ExecutionError("extension bridge returned empty drawer text")
        state = _parse_bot_state(raw_text, expected_symbol=intent.symbol)
        _validate_state_identity(state, intent)
        return state

    def _read_form(self, deadline: Deadline) -> dict[str, Any]:
        return self._bridge.call("read_form", {}, deadline=deadline)

    def prepare(
        self,
        intent: ActionIntent,
        current: BotState,
        deadline: Deadline,
    ) -> PreparedAdjustment:
        original = self._bridge.call(
            "open_modify_form",
            {
                "symbol": intent.symbol,
                "strategy_id": intent.strategy_id,
            },
            deadline=deadline,
        )
        lower_text = original.get("lower")
        upper_text = original.get("upper")
        grids_text = original.get("grids")
        if not isinstance(lower_text, str) or not isinstance(upper_text, str):
            raise ExecutionError("current extension price inputs are unavailable")
        if not isinstance(grids_text, str) or not grids_text.isdigit():
            raise ExecutionError("current extension grid count input is unavailable")
        if int(grids_text) != current.num_grids:
            raise ExecutionError("extension form grid count differs from live drawer")

        decimals = max(_decimal_places(lower_text), _decimal_places(upper_text))
        rounded_lower, rounded_upper = outward_round_bounds(
            intent.lower,
            intent.upper,
            decimals=decimals,
        )
        lower_value = format(rounded_lower, f".{decimals}f")
        upper_value = format(rounded_upper, f".{decimals}f")
        self._bridge.call("ensure_keep_position", {}, deadline=deadline)
        self._bridge.call(
            "set_form_inputs",
            {"lower": lower_value, "upper": upper_value},
            deadline=deadline,
        )
        final = self._read_form(deadline)
        final_grids = str(final.get("grids", ""))
        investment_text = str(final.get("additional_investment", "")).replace(
            ",", ""
        )
        try:
            investment = Decimal(investment_text)
        except InvalidOperation as exc:
            raise ExecutionError("additional investment is not parseable") from exc
        return PreparedAdjustment(
            symbol=intent.symbol,
            strategy_id=intent.strategy_id,
            lower=rounded_lower,
            upper=rounded_upper,
            num_grids=int(final_grids) if final_grids.isdigit() else -1,
            original_num_grids=int(grids_text),
            preserve_current_position=final.get("close_positions") == "No",
            additional_investment=investment,
            price_decimals=decimals,
            confirm_enabled=final.get("confirm_enabled") is True,
        )

    def submit_once(
        self,
        prepared: PreparedAdjustment,
        deadline: Deadline,
    ) -> None:
        if self._submitted:
            raise ExecutionError("final submit was already attempted")
        form = self._read_form(deadline)
        expected_lower = format(prepared.lower, f".{prepared.price_decimals}f")
        expected_upper = format(prepared.upper, f".{prepared.price_decimals}f")
        if (
            form.get("lower") != expected_lower
            or form.get("upper") != expected_upper
            or str(form.get("grids")) != str(prepared.num_grids)
            or form.get("close_positions") != "No"
            or str(form.get("additional_investment")) not in {"0", "0.0", "0.00"}
            or form.get("confirm_enabled") is not True
        ):
            raise ExecutionError(f"final extension form verification failed: {form}")
        deadline.require("final submit")
        self._submitted = True
        result = self._bridge.call(
            "submit",
            {
                "lower": expected_lower,
                "upper": expected_upper,
                "grids": prepared.num_grids,
                "preserve_current_position": True,
                "additional_investment": "0",
            },
            deadline=deadline,
        )
        if result.get("acknowledged") is not True:
            raise ExecutionError("extension bridge did not acknowledge final submit")

    def wait_before_post_verify(self, deadline: Deadline) -> None:
        self._bridge.call(
            "wait",
            {"seconds": min(1.0, max(0.0, deadline.remaining_seconds - 0.1))},
            deadline=deadline,
        )


def _load_extension_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ExecutionError(f"cannot read extension bridge token {path}: {exc}") from exc
    if KEY_RE.fullmatch(token) is None:
        raise ExecutionError(
            "extension bridge token file must contain one 64-character lowercase hex token"
        )
    return token


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"{path} must contain one JSON object")
    return value


def _read_stdin_object() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ExecutionError("stdin must contain one action-intent JSON object")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ExecutionError(f"stdin action intent is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError("stdin action intent must be one JSON object")
    return value


def _append_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_reserved_action_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("status") not in {"submit_started", "executed"}:
            continue
        key = row.get("idempotency_key")
        if isinstance(key, str):
            keys.add(key)
    return keys


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="execute live telemetry modifications",
        description=__doc__,
    )
    parser.add_argument("--approval-file", type=Path, required=True)
    parser.add_argument(
        "--browser-transport",
        choices=("cdp", "extension"),
        default="cdp",
        help="Chrome control transport. CDP remains the backward-compatible default.",
    )
    parser.add_argument("--debug-endpoint", default="http://127.0.0.1:9222")
    parser.add_argument(
        "--extension-endpoint",
        default="http://127.0.0.1:17731",
        help="Authenticated loopback bridge for a claimed Chrome extension tab.",
    )
    parser.add_argument(
        "--extension-token-file",
        type=Path,
        help="File containing the bridge's one-run 64-character token.",
    )
    parser.add_argument("--deadline-seconds", type=float, default=90.0)
    parser.add_argument("--command-timeout-seconds", type=float, default=4.0)
    parser.add_argument(
        "--extension-command-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-command timeout for higher-level extension bridge operations.",
    )
    parser.add_argument("--ui-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--hover-seconds", type=float, default=0.20)
    parser.add_argument("--max-decision-age-seconds", type=float, default=180.0)
    parser.add_argument("--max-approval-lifetime-seconds", type=float, default=300.0)
    parser.add_argument("--min-post-submit-seconds", type=float, default=15.0)
    parser.add_argument("--post-verify-attempts", type=int, default=2)
    parser.add_argument(
        "--audit-ledger",
        type=Path,
        default=ROOT
        / "outputs"
        / "audits"
        / "live_telemetry_modifications_current"
        / "execution_ledger.jsonl",
    )
    args = parser.parse_args(argv)
    for name in (
        "deadline_seconds",
        "command_timeout_seconds",
        "extension_command_timeout_seconds",
        "ui_timeout_seconds",
        "max_decision_age_seconds",
        "max_approval_lifetime_seconds",
        "min_post_submit_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.deadline_seconds > 180:
        parser.error("--deadline-seconds cannot exceed 180")
    if args.command_timeout_seconds > args.deadline_seconds:
        parser.error("--command-timeout-seconds cannot exceed --deadline-seconds")
    if args.extension_command_timeout_seconds > args.deadline_seconds:
        parser.error(
            "--extension-command-timeout-seconds cannot exceed --deadline-seconds"
        )
    if args.min_post_submit_seconds >= args.deadline_seconds:
        parser.error("--min-post-submit-seconds must be below --deadline-seconds")
    if args.hover_seconds < 0 or args.hover_seconds > 1:
        parser.error("--hover-seconds must be between 0 and 1")
    if args.post_verify_attempts not in {1, 2}:
        parser.error("--post-verify-attempts must be 1 or 2")
    if args.browser_transport == "extension" and args.extension_token_file is None:
        parser.error(
            "--extension-token-file is required when --browser-transport=extension"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = _utc_now()
    audit: dict[str, Any] = {
        "schema_version": RESPONSE_SCHEMA,
        "started_at_utc": started_at.isoformat(),
        "status": "running",
    }
    driver: AdjustmentDriver | None = None
    response: dict[str, Any]
    try:
        intent_payload = _read_stdin_object()
        intent = parse_intent(
            intent_payload,
            now=started_at,
            max_age_seconds=args.max_decision_age_seconds,
        )
        approval = parse_approval(
            _load_json_object(args.approval_file),
            now=started_at,
            max_lifetime_seconds=args.max_approval_lifetime_seconds,
        )
        validate_approval(intent, approval)
        audit.update(
            {
                "idempotency_key": intent.idempotency_key,
                "symbol": intent.symbol,
                "strategy_id": intent.strategy_id,
                "action": "ADJUST",
                "approval_file": str(args.approval_file.resolve()),
                "browser_transport": args.browser_transport,
            }
        )
        if intent.idempotency_key in _load_reserved_action_keys(args.audit_ledger):
            raise ExecutionError(
                "idempotency key is already reserved by a prior submit attempt"
            )
        if args.browser_transport == "extension":
            if args.extension_token_file is None:
                raise ExecutionError("extension bridge token file is required")
            driver = BinanceExtensionAdjustmentDriver(
                bridge_endpoint=args.extension_endpoint,
                bridge_token=_load_extension_token(args.extension_token_file),
                symbol=intent.symbol,
                command_timeout_seconds=args.extension_command_timeout_seconds,
            )
        else:
            driver = BinanceCdpAdjustmentDriver(
                debug_endpoint=args.debug_endpoint,
                symbol=intent.symbol,
                command_timeout_seconds=args.command_timeout_seconds,
                ui_timeout_seconds=args.ui_timeout_seconds,
                hover_seconds=args.hover_seconds,
            )
        response = execute_adjustment(
            intent,
            approval,
            driver,
            policy=ExecutionPolicy(
                deadline_seconds=args.deadline_seconds,
                max_decision_age_seconds=args.max_decision_age_seconds,
                max_approval_lifetime_seconds=args.max_approval_lifetime_seconds,
                min_post_submit_seconds=args.min_post_submit_seconds,
                post_verify_attempts=args.post_verify_attempts,
            ),
            before_submit=lambda prepared: _append_audit(
                args.audit_ledger,
                {
                    **audit,
                    "status": "submit_started",
                    "submit_started_at_utc": _utc_now().isoformat(),
                    "executed_grid_lower": float(prepared.lower),
                    "executed_grid_upper": float(prepared.upper),
                    "price_decimals": prepared.price_decimals,
                    "num_grids": prepared.num_grids,
                    "preserve_current_position": prepared.preserve_current_position,
                    "additional_investment": float(prepared.additional_investment),
                },
            ),
        )
        audit.update(response)
        audit["completed_at_utc"] = _utc_now().isoformat()
        _append_audit(args.audit_ledger, audit)
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        return 0
    except (ExecutionError, TelemetryError, OSError, TimeoutError) as exc:
        logger.error("execution blocked: %s", exc)
        response = {
            "schema_version": RESPONSE_SCHEMA,
            "idempotency_key": audit.get("idempotency_key"),
            "status": "blocked",
            "error": str(exc),
        }
        audit.update(response)
        audit["completed_at_utc"] = _utc_now().isoformat()
        _append_audit(args.audit_ledger, audit)
        sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")
        return 2
    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
