"""Collect read-only Binance Futures Grid telemetry from a dedicated Chrome.

The browser must be launched with a non-default profile and a Chrome DevTools
debugging endpoint.  This collector deliberately avoids cookies, undocumented
HTTP endpoints, and every control that can modify or end a bot.  A cycle is
committed only when all rows in the visible UM Grid Running table produce a
complete View Details drawer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from websockets.sync.client import connect as websocket_connect

from neutralgrid.live.monotonic_schedule import advance_nominal_start


ROOT = Path(__file__).resolve().parents[1]
LIMA = ZoneInfo("America/Lima")
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")
STRATEGY_RE = re.compile(r"Strategy Number\s+(\d+)")
CREATED_RE = re.compile(r"Time Created\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
EXPECTED_MARKERS = (
    "Total Profit (USDT)",
    "Positions",
    "Pending Order",
    "Grid Details",
    "Order History",
    "Strategy Number",
)

logger = logging.getLogger(__name__)


class TelemetryError(RuntimeError):
    """Fail-closed telemetry collection error."""


def _mark_one_shot_finished(
    heartbeat: dict[str, Any],
    *,
    succeeded: bool,
    finished_at: datetime,
) -> None:
    """Replace a transient one-shot state with an explicit terminal marker."""

    heartbeat["status_before_finish"] = heartbeat.get("status")
    heartbeat["status"] = "finished"
    heartbeat["finished_at_utc"] = _iso_seconds(finished_at)
    heartbeat["finish_reason"] = (
        "one_shot_complete" if succeeded else "one_shot_failed"
    )


@dataclass(frozen=True)
class ActiveBotRow:
    symbol: str
    status: str
    row_index: str
    action_buttons: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class DrawerSnapshot:
    symbol: str
    strategy_id: str
    created_at_lima: str | None
    captured_at_utc: str
    captured_at_lima: str
    raw_text: str


class CDPConnection:
    """Minimal synchronous Chrome DevTools Protocol client."""

    def __init__(self, websocket_url: str, *, timeout_seconds: float) -> None:
        self._socket = websocket_connect(
            websocket_url,
            open_timeout=timeout_seconds,
            close_timeout=timeout_seconds,
        )
        self._next_id = 0
        self._timeout_seconds = timeout_seconds

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._socket.send(
            json.dumps(
                {
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
        )
        while True:
            payload = json.loads(self._socket.recv(timeout=self._timeout_seconds))
            if payload.get("id") != request_id:
                continue
            if "error" in payload:
                raise TelemetryError(f"CDP {method} failed: {payload['error']}")
            result = payload.get("result")
            if not isinstance(result, dict):
                raise TelemetryError(f"CDP {method} returned a non-object result")
            return result

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        )
        if result.get("exceptionDetails"):
            raise TelemetryError(f"browser evaluation failed: {result['exceptionDetails']}")
        remote = result.get("result")
        if not isinstance(remote, dict):
            raise TelemetryError("browser evaluation returned no result")
        return remote.get("value")

    def move_mouse(self, x: float, y: float) -> None:
        self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})

    def click(self, x: float, y: float) -> None:
        base = {"x": x, "y": y, "button": "left", "clickCount": 1}
        self.call("Input.dispatchMouseEvent", {"type": "mousePressed", **base})
        self.call("Input.dispatchMouseEvent", {"type": "mouseReleased", **base})

    def press_escape(self) -> None:
        key = {
            "key": "Escape",
            "code": "Escape",
            "windowsVirtualKeyCode": 27,
            "nativeVirtualKeyCode": 27,
        }
        self.call("Input.dispatchKeyEvent", {"type": "keyDown", **key})
        self.call("Input.dispatchKeyEvent", {"type": "keyUp", **key})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_seconds(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat()


def _fetch_json(url: str, *, timeout_seconds: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise TelemetryError(f"Chrome debugging endpoint unavailable: {url}: {exc}") from exc


def _find_grid_page(debug_endpoint: str, *, timeout_seconds: float) -> str:
    pages = _fetch_json(f"{debug_endpoint.rstrip('/')}/json/list", timeout_seconds=timeout_seconds)
    if not isinstance(pages, list):
        raise TelemetryError("Chrome /json/list did not return a list")
    matches = [
        page
        for page in pages
        if isinstance(page, dict)
        and page.get("type") == "page"
        and "binance.bh/en/trading-bots/futures/grid/" in str(page.get("url", ""))
        and page.get("webSocketDebuggerUrl")
    ]
    if len(matches) != 1:
        raise TelemetryError(
            "expected exactly one Binance Futures Grid page in dedicated Chrome; "
            f"found {len(matches)}"
        )
    return str(matches[0]["webSocketDebuggerUrl"])


def _parse_expected_count(tab_texts: Any) -> int:
    if not isinstance(tab_texts, list):
        raise TelemetryError("UM Grid tabs were not readable; sign-in may be required")
    counts: list[int] = []
    for value in tab_texts:
        match = re.fullmatch(r"UM Grid \((\d+)\)", str(value).strip())
        if match:
            counts.append(int(match.group(1)))
    if len(counts) != 1:
        raise TelemetryError(f"expected one UM Grid count; found {counts}")
    return counts[0]


def _discover_rows(cdp: CDPConnection) -> tuple[int, list[ActiveBotRow]]:
    expected_count = _parse_expected_count(
        cdp.evaluate(
            """(() => Array.from(document.querySelectorAll('[role="tab"]'))
                .map((element) => (element.textContent || '').trim())
                .filter((text) => text.startsWith('UM Grid')))()"""
        )
    )
    raw_rows = cdp.evaluate(
        """(() => Array.from(
              document.querySelectorAll('div.bn-virtual-table-row[role="row"]')
            ).map((row) => {
              const cells = Array.from(row.children);
              const symbolText = (cells[0]?.textContent || '').trim();
              const symbolMatch = symbolText.match(/^([A-Z0-9]+USDT)\\s+Perp/);
              const status = (cells[14]?.textContent || '').trim();
              const actionCell = cells[cells.length - 1];
              const buttons = actionCell
                ? Array.from(actionCell.querySelectorAll('button')).map((button) => {
                    const rect = button.getBoundingClientRect();
                    return {
                      x: rect.x + rect.width / 2,
                      y: rect.y + rect.height / 2,
                      width: rect.width,
                      height: rect.height
                    };
                  })
                : [];
              return {
                symbol: symbolMatch ? symbolMatch[1] : null,
                status,
                rowIndex: row.getAttribute('aria-rowindex') || '',
                buttons
              };
            }).filter((row) => row.symbol))()"""
    )
    if not isinstance(raw_rows, list):
        raise TelemetryError("Running grid rows were not readable")
    rows: list[ActiveBotRow] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise TelemetryError("Running grid row is not an object")
        symbol = str(raw.get("symbol", "")).upper()
        status = str(raw.get("status", "")).strip()
        buttons = raw.get("buttons")
        if not SYMBOL_RE.fullmatch(symbol):
            raise TelemetryError(f"invalid active symbol: {symbol!r}")
        if status != "Working":
            raise TelemetryError(f"{symbol} is in Running table but status is {status!r}")
        if not isinstance(buttons, list) or len(buttons) != 4:
            raise TelemetryError(
                f"{symbol} action contract changed: expected 4 buttons, found "
                f"{len(buttons) if isinstance(buttons, list) else 'non-list'}"
            )
        typed_buttons: list[dict[str, float]] = []
        for button in buttons:
            if not isinstance(button, dict):
                raise TelemetryError(f"{symbol} has malformed action button geometry")
            typed_buttons.append(
                {
                    "x": float(button["x"]),
                    "y": float(button["y"]),
                    "width": float(button["width"]),
                    "height": float(button["height"]),
                }
            )
        rows.append(
            ActiveBotRow(
                symbol=symbol,
                status=status,
                row_index=str(raw.get("rowIndex", "")),
                action_buttons=tuple(typed_buttons),
            )
        )
    if expected_count != len(rows):
        raise TelemetryError(
            "visible Running rows do not match UM Grid count: "
            f"expected {expected_count}, visible {len(rows)}; refusing partial cycle"
        )
    if not rows:
        raise TelemetryError("UM Grid Running table contains no active bots")
    if len({row.symbol for row in rows}) != len(rows):
        raise TelemetryError("duplicate active symbol rows are ambiguous")
    return expected_count, rows


def _tooltip_is_view_details(cdp: CDPConnection) -> bool:
    value = cdp.evaluate(
        """(() => Array.from(document.querySelectorAll('[role="tooltip"], .bn-tooltips-content'))
          .some((element) => {
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return (element.textContent || '').trim() === 'View Details'
              && rect.width > 0
              && rect.height > 0
              && style.display !== 'none'
              && style.visibility !== 'hidden';
          }))()"""
    )
    return value is True


def _wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_seconds: float,
    interval_seconds: float = 0.10,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_seconds)
    raise TelemetryError(f"browser state did not become ready within {timeout_seconds:.1f}s")


def _find_view_details_button(
    cdp: CDPConnection,
    row: ActiveBotRow,
    *,
    hover_seconds: float,
) -> dict[str, float]:
    for button in row.action_buttons:
        cdp.move_mouse(10.0, 10.0)
        time.sleep(0.05)
        cdp.move_mouse(button["x"], button["y"])
        time.sleep(hover_seconds)
        if _tooltip_is_view_details(cdp):
            return button
    raise TelemetryError(f"{row.symbol}: no action button verified as View Details")


def _drawer_visible(cdp: CDPConnection) -> bool:
    return (
        cdp.evaluate(
            """(() => {
              const drawer = document.querySelector('.bn-trans.data-show.bn-mask.bn-drawer');
              if (!drawer) return false;
              const rect = drawer.getBoundingClientRect();
              return rect.width > 0 && rect.height > 0;
            })()"""
        )
        is True
    )


def _read_drawer(cdp: CDPConnection) -> str:
    value = cdp.evaluate(
        """(() => {
          const drawer = document.querySelector('.bn-trans.data-show.bn-mask.bn-drawer');
          return drawer ? drawer.innerText : null;
        })()"""
    )
    if not isinstance(value, str):
        raise TelemetryError("View Details drawer text is unavailable")
    return value.replace("\u00a0", " ").strip()


def _validate_drawer_text(symbol: str, raw_text: str) -> tuple[str, str | None]:
    if not raw_text.startswith(f"{symbol}\n"):
        raise TelemetryError(f"{symbol}: drawer symbol does not match")
    missing = [marker for marker in EXPECTED_MARKERS if marker not in raw_text]
    if missing:
        raise TelemetryError(f"{symbol}: incomplete drawer, missing {missing}")
    strategy_match = STRATEGY_RE.search(raw_text)
    if strategy_match is None:
        raise TelemetryError(f"{symbol}: Strategy Number is not parseable")
    created_match = CREATED_RE.search(raw_text)
    return strategy_match.group(1), created_match.group(1) if created_match else None


def _close_drawer(cdp: CDPConnection, *, timeout_seconds: float) -> None:
    cdp.click(50.0, 50.0)
    try:
        _wait_until(
            lambda: not _drawer_visible(cdp),
            timeout_seconds=timeout_seconds,
        )
    except TelemetryError:
        cdp.press_escape()
        _wait_until(
            lambda: not _drawer_visible(cdp),
            timeout_seconds=timeout_seconds,
        )


def _capture_row(
    cdp: CDPConnection,
    row: ActiveBotRow,
    *,
    timeout_seconds: float,
    hover_seconds: float,
) -> DrawerSnapshot:
    button = _find_view_details_button(cdp, row, hover_seconds=hover_seconds)
    cdp.click(button["x"], button["y"])
    _wait_until(
        lambda: _drawer_visible(cdp),
        timeout_seconds=timeout_seconds,
    )
    try:
        raw_text = _read_drawer(cdp)
        strategy_id, created_at_lima = _validate_drawer_text(row.symbol, raw_text)
        captured_at = _utc_now()
        return DrawerSnapshot(
            symbol=row.symbol,
            strategy_id=strategy_id,
            created_at_lima=created_at_lima,
            captured_at_utc=_iso_seconds(captured_at),
            captured_at_lima=_iso_seconds(captured_at.astimezone(LIMA)),
            raw_text=raw_text,
        )
    finally:
        _close_drawer(cdp, timeout_seconds=timeout_seconds)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _commit_cycle(
    snapshots: list[DrawerSnapshot],
    *,
    live_root: Path,
    audit_dir: Path,
    cycle_started_at: datetime,
) -> dict[str, Any]:
    lima_started = cycle_started_at.astimezone(LIMA)
    stamp = lima_started.strftime("%Y%m%d_%H%M%S_lima")
    live_date = lima_started.strftime("%Y-%m-%d")
    files: list[dict[str, str]] = []
    for snapshot in snapshots:
        symbol_dir = live_root / live_date / snapshot.symbol
        base = f"private_telemetry_{snapshot.strategy_id}_{stamp}"
        text_path = symbol_dir / f"{base}.txt"
        json_path = symbol_dir / f"{base}.json"
        _atomic_write_text(text_path, snapshot.raw_text + "\n")
        metadata = {
            **asdict(snapshot),
            "raw_text": None,
            "data_class": "live_bot_telemetry",
            "status": "active_live_snapshot",
            "source": "dedicated_chrome_cdp",
            "required_sections": list(EXPECTED_MARKERS),
        }
        _atomic_write_json(json_path, metadata)
        files.append(
            {
                "symbol": snapshot.symbol,
                "strategy_id": snapshot.strategy_id,
                "text_path": str(text_path),
                "json_path": str(json_path),
            }
        )
    manifest = {
        "status": "complete",
        "cycle_started_at_utc": _iso_seconds(cycle_started_at),
        "cycle_completed_at_utc": _iso_seconds(_utc_now()),
        "active_bot_count": len(snapshots),
        "symbols": [snapshot.symbol for snapshot in snapshots],
        "files": files,
    }
    _atomic_write_json(audit_dir / "cycles" / f"cycle_{stamp}.json", manifest)
    return manifest


def collect_cycle(args: argparse.Namespace) -> dict[str, Any]:
    cycle_started_at = _utc_now()
    websocket_url = _find_grid_page(
        args.debug_endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    cdp = CDPConnection(websocket_url, timeout_seconds=args.timeout_seconds)
    try:
        cdp.call("Runtime.enable")
        cdp.call("Page.bringToFront")
        expected_count, rows = _discover_rows(cdp)
        snapshots = [
            _capture_row(
                cdp,
                row,
                timeout_seconds=args.timeout_seconds,
                hover_seconds=args.hover_seconds,
            )
            for row in rows
        ]
        if len(snapshots) != expected_count:
            raise TelemetryError(
                f"captured {len(snapshots)} drawers for {expected_count} active bots"
            )
    finally:
        cdp.close()
    return _commit_cycle(
        snapshots,
        live_root=Path(args.live_root),
        audit_dir=Path(args.audit_dir),
        cycle_started_at=cycle_started_at,
    )


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            existing_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            existing_pid = -1
        if _pid_is_running(existing_pid):
            raise TelemetryError(
                f"private telemetry collector already running with PID {existing_pid}"
            )
        lock_path.unlink(missing_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode("ascii"))
    return descriptor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug-endpoint",
        default="http://127.0.0.1:9222",
        help="Dedicated Chrome DevTools HTTP endpoint.",
    )
    parser.add_argument("--interval-seconds", type=float, default=180.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--hover-seconds", type=float, default=0.35)
    parser.add_argument("--live-root", default=str(ROOT / "Live"))
    parser.add_argument(
        "--audit-dir",
        default=str(ROOT / "outputs" / "audits" / "private_telemetry_loop_current"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be > 0")
    if args.hover_seconds < 0:
        parser.error("--hover-seconds must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit_dir = Path(args.audit_dir)
    lock_path = audit_dir / "collector.lock"
    stop_path = audit_dir / "STOP"
    descriptor = _acquire_lock(lock_path)
    stop_requested = False

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    consecutive_errors = 0
    completed_cycles = 0
    scheduled_start_monotonic = time.monotonic()
    total_skipped_slots = 0
    try:
        while True:
            actual_start_monotonic = time.monotonic()
            actual_start_utc = _utc_now()
            lateness_seconds = max(
                0.0,
                actual_start_monotonic - scheduled_start_monotonic,
            )
            scheduled_start_utc = actual_start_utc - timedelta(
                seconds=lateness_seconds
            )
            heartbeat: dict[str, Any]
            try:
                manifest = collect_cycle(args)
                completed_cycles += 1
                consecutive_errors = 0
                heartbeat = {
                    "status": "running",
                    "last_cycle": manifest,
                    "completed_cycles": completed_cycles,
                    "consecutive_errors": consecutive_errors,
                    "pid": os.getpid(),
                    "interval_seconds": args.interval_seconds,
                    "data_age_seconds": max(
                        0.0,
                        (
                            _utc_now()
                            - datetime.fromisoformat(
                                str(manifest["cycle_completed_at_utc"])
                            ).astimezone(timezone.utc)
                        ).total_seconds(),
                    ),
                    "failure_class": None,
                }
                logger.info(
                    "private telemetry cycle complete: %d active bots",
                    manifest["active_bot_count"],
                )
            except Exception as exc:
                consecutive_errors += 1
                failed_at = _utc_now()
                heartbeat = {
                    "status": "waiting_for_valid_browser_state",
                    "last_error": repr(exc),
                    "last_error_at_utc": _iso_seconds(failed_at),
                    "completed_cycles": completed_cycles,
                    "consecutive_errors": consecutive_errors,
                    "pid": os.getpid(),
                    "interval_seconds": args.interval_seconds,
                    "data_age_seconds": None,
                    "failure_class": type(exc).__name__,
                }
                _atomic_write_json(
                    audit_dir
                    / "cycles"
                    / f"failed_{failed_at.astimezone(LIMA).strftime('%Y%m%d_%H%M%S_lima')}.json",
                    heartbeat,
                )
                logger.warning("private telemetry cycle rejected: %s", exc)
            actual_end_utc = _utc_now()
            actual_end_monotonic = time.monotonic()
            scheduled_start_monotonic, skipped_slots = advance_nominal_start(
                scheduled_start_monotonic,
                interval_seconds=float(args.interval_seconds),
                completed_at=actual_end_monotonic,
            )
            total_skipped_slots += skipped_slots
            wait_seconds = max(0.0, scheduled_start_monotonic - actual_end_monotonic)
            heartbeat.update(
                {
                    "scheduled_start_utc": _iso_seconds(scheduled_start_utc),
                    "actual_start_utc": _iso_seconds(actual_start_utc),
                    "actual_end_utc": _iso_seconds(actual_end_utc),
                    "duration_seconds": max(
                        0.0,
                        actual_end_monotonic - actual_start_monotonic,
                    ),
                    "lateness_seconds": lateness_seconds,
                    "skipped_slots": skipped_slots,
                    "total_skipped_slots": total_skipped_slots,
                    "next_scheduled_start_utc": _iso_seconds(
                        actual_end_utc + timedelta(seconds=wait_seconds)
                    ),
                }
            )
            if args.once:
                succeeded = consecutive_errors == 0
                _mark_one_shot_finished(
                    heartbeat,
                    succeeded=succeeded,
                    finished_at=_utc_now(),
                )
            _atomic_write_json(audit_dir / "manifest.json", heartbeat)
            if args.once:
                return 0 if succeeded else 2
            if stop_requested or stop_path.exists():
                break
            while time.monotonic() < scheduled_start_monotonic:
                if stop_requested or stop_path.exists():
                    break
                time.sleep(
                    min(
                        1.0,
                        max(0.0, scheduled_start_monotonic - time.monotonic()),
                    )
                )
            if stop_requested or stop_path.exists():
                break
        heartbeat["status"] = "stopped"
        heartbeat["stopped_at_utc"] = _iso_seconds(_utc_now())
        _atomic_write_json(audit_dir / "manifest.json", heartbeat)
        return 0
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
