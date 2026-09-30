"""Consume one fresh Chrome-extension cycle once, with advisory output only.

The recurring Codex task owns browser capture. This entry point never launches
a browser, trains a model, or dispatches an exchange action. Receipts are claimed
before the scanner can advance history: an interrupted cycle requires a NEW
capture, rather than replaying evidence and inflating persistence counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_live_telemetry_controller as controller

logger = logging.getLogger(__name__)
SCHEMA = "neutralgrid_drawer_verdict_receipt_v1"
DEFAULT_AUDIT = ROOT / "outputs" / "audits" / "recurring_drawer_verdict"
MAX_AGE_SECONDS = 900.0


@contextmanager
def exclusive_run_guard(audit_dir: Path) -> Iterator[None]:
    """OS-owned lock releases on process death; the guard file stays in place."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "consumer.guard").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise controller.ControllerError("drawer consumer already running") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise controller.ControllerError("drawer consumer already running") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _observations(cycle: controller.TelemetryCycle) -> list[tuple[str, str, str, str]]:
    return sorted(
        (
            bot.symbol,
            bot.strategy_id,
            bot.captured_at_utc.isoformat(),
            hashlib.sha256(bot.raw_text_path.read_bytes()).hexdigest(),
        )
        for bot in cycle.bots
    )


def capture_key(cycle: controller.TelemetryCycle) -> str:
    """Deduplicate observations, including an identical manifest at a new path."""
    return hashlib.sha256(json.dumps(_observations(cycle)).encode("utf-8")).hexdigest()


def controller_args(args: argparse.Namespace) -> argparse.Namespace:
    command = [
        "--acquisition-mode", "plugin-manifest",
        "--cycle-manifest", str(args.cycle_manifest),
        "--telemetry-audit-dir", str(args.cycle_manifest.parent.parent),
        "--controller-audit-dir", str(args.audit_dir / "controller"),
        "--runtime-dir", str(args.audit_dir / "runtime"),
        "--scanner-state-dir", str(args.audit_dir / "state"),
        "--live-root", str(args.live_root),
        "--max-telemetry-age-seconds", str(MAX_AGE_SECONDS),
        "--observational-only", "--once", "--require-l2-evidence",
    ]
    for manifest in args.diff_depth_manifest:
        command.extend(["--diff-depth-manifest", str(manifest)])
    return controller.parse_args(command)


def consume(args: argparse.Namespace) -> dict[str, Any]:
    """Caller holds the controller lock; any claimed cycle is never replayed."""
    cfg = controller_args(args)
    if not args.diff_depth_manifest:
        raise controller.ControllerError("recurring verdict requires a P0 public L2 manifest")
    payload = controller._load_json_object(args.cycle_manifest, label="cycle manifest")
    if payload.get("source") != "chrome_plugin":
        raise controller.ControllerError("recurring drawer capture requires chrome_plugin source")
    cycle, _ = controller.acquire_cycle(cfg, now=_utc_now())
    key = capture_key(cycle)
    receipt_path = args.audit_dir / "receipts" / f"{key}.json"
    if receipt_path.exists():
        previous = controller._load_json_object(receipt_path, label="cycle receipt")
        if previous.get("schema_version") != SCHEMA or previous.get("capture_key") != key:
            raise controller.ControllerError("cycle receipt identity/schema mismatch")
        return {
            "status": "duplicate" if previous.get("status") == "complete" else "blocked",
            "reason": "capture_already_claimed; acquire a new drawer cycle",
            "receipt_path": str(receipt_path),
            "previous_status": previous.get("status"),
            "verdicts": [],
        }

    observation_paths = [
        args.audit_dir / "observations" / f"{hashlib.sha256(json.dumps(item).encode('utf-8')).hexdigest()}.json"
        for item in _observations(cycle)
    ]
    if any(path.exists() for path in observation_paths):
        raise controller.ControllerError("bot observation already claimed in another cycle; acquire a new full roster")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "capture_key": key,
        "status": "started",
        "started_at_utc": _utc_now().isoformat(),
        "telemetry_manifest": str(cycle.manifest_path),
        "telemetry_manifest_sha256": hashlib.sha256(cycle.manifest_path.read_bytes()).hexdigest(),
        "oldest_capture_at_utc": min(bot.captured_at_utc for bot in cycle.bots).isoformat(),
        "data_class": "live_bot_telemetry",
        "training_eligible": False,
        "observational_only": True,
        "verdicts": [],
    }
    # Atomic claim under the exclusive controller lock. A process death leaves
    # 'started' behind; the next run must not advance the same evidence again.
    controller._atomic_write_json(receipt_path, receipt)
    for path in observation_paths:
        controller._atomic_write_json(path, {"capture_key": key, "receipt_path": str(receipt_path)})
    controller._atomic_write_json(args.audit_dir / "manifest.json", receipt)
    try:
        report = controller.run_iteration(cfg)
        if report.get("status") != "complete":
            raise controller.ControllerError(str(report.get("error", "controller blocked")))
        if report.get("required_evidence_validated") is not True:
            raise controller.ControllerError("controller did not validate required L2 evidence")
        # A slow scan must not turn an expired capture into a current verdict.
        controller.validate_cycle_freshness(cycle, now=_utc_now(), max_age_seconds=MAX_AGE_SECONDS)
        if capture_key(cycle) != key:
            raise controller.ControllerError("capture changed during scanner evaluation")
        if hashlib.sha256(cycle.manifest_path.read_bytes()).hexdigest() != receipt["telemetry_manifest_sha256"]:
            raise controller.ControllerError("manifest changed during scanner evaluation")
        receipt.update(status="complete", verdicts=report["verdicts"], controller_report=report,
                       required_evidence_validated=True)
    except Exception as exc:
        # This is the process boundary: unexpected failures are visible and
        # classified, with no success verdict. KeyboardInterrupt still exits.
        logger.exception("drawer verdict cycle rejected")
        receipt.update(status="blocked", failure_class=type(exc).__name__, error=str(exc))
    receipt["completed_at_utc"] = _utc_now().isoformat()
    controller._atomic_write_json(receipt_path, receipt)
    controller._atomic_write_json(args.audit_dir / "manifest.json", receipt)
    return receipt


def health(audit_dir: Path, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or _utc_now()
    try:
        latest = controller._load_json_object(audit_dir / "manifest.json", label="drawer health")
        if latest.get("schema_version") != SCHEMA:
            raise controller.ControllerError("drawer health schema unavailable")
        if latest.get("status") != "complete":
            return {"status": "blocked", "error": latest.get("error", "last cycle incomplete"), "verdicts": []}
        if latest.get("required_evidence_validated") is not True:
            raise controller.ControllerError("saved cycle lacks required L2 validation; acquire a new capture")
        oldest = controller._parse_datetime(latest.get("oldest_capture_at_utc"), field="oldest_capture_at_utc")
        age = (now - oldest).total_seconds()
        current = -5 <= age <= MAX_AGE_SECONDS
        return {
            "status": "current" if current else "stale",
            "oldest_capture_age_seconds": age,
            "cadence_status": "on_time" if -5 <= age <= 600 else "overdue",
            "verdicts": latest.get("verdicts", []) if current else [],
            "receipt_capture_key": latest.get("capture_key"),
        }
    except (controller.ControllerError, OSError, ValueError) as exc:
        return {"status": "blocked", "error": str(exc), "verdicts": []}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cycle-manifest", type=Path)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--capture-failure", help="Record a failed browser acquisition; publish no verdict.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--diff-depth-manifest", type=Path, action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        result = health(args.audit_dir)
    else:
        lock_path = args.audit_dir / "controller" / "controller.lock"
        descriptor: int | None = None
        try:
            with exclusive_run_guard(args.audit_dir):
                descriptor = controller._acquire_lock(lock_path)
                try:
                    try:
                        if args.capture_failure is not None:
                            result = {"schema_version": SCHEMA, "status": "blocked",
                                      "failure_class": "CaptureFailure", "error": args.capture_failure,
                                      "failed_at_utc": _utc_now().isoformat(), "verdicts": []}
                            controller._atomic_write_json(args.audit_dir / "manifest.json", result)
                        else:
                            result = consume(args)
                    except Exception as exc:
                        logger.exception("drawer cycle unavailable")
                        result = {"schema_version": SCHEMA, "status": "blocked",
                                  "failure_class": type(exc).__name__, "error": str(exc), "verdicts": []}
                        try:
                            controller._atomic_write_json(args.audit_dir / "manifest.json", result)
                        except OSError:
                            logger.exception("cannot persist failure status")
                finally:
                    os.close(descriptor)
                    lock_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error("drawer cycle unavailable: %s", exc)
            result = {"status": "blocked", "failure_class": type(exc).__name__, "error": str(exc), "verdicts": []}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"complete", "current", "duplicate"} else 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
