"""Validate and commit one Chrome-plugin private telemetry capture bundle.

The Chrome plugin owns visible-page acquisition.  This script never opens or
controls a browser.  It accepts only a complete, run-owned capture bundle,
validates the roster fence and every exact bot identity, then commits immutable
snapshot files under ``Live/YYYY-MM-DD/SYMBOL`` and writes the cycle manifest
last.  A committed cycle is suitable for the controller's
``--acquisition-mode plugin-manifest`` path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_live_telemetry_controller import (
    PLUGIN_CYCLE_SCHEMA,
    PLUGIN_SNAPSHOT_SCHEMA,
    _scanner_entry_from_raw,
)


UTC = timezone.utc
LIMA = ZoneInfo("America/Lima")
PLUGIN_CAPTURE_BUNDLE_SCHEMA = "neutralgrid_chrome_plugin_capture_bundle_v1"
SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")
STRATEGY_RE = re.compile(r"^\d+$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CREATED_RE = re.compile(r"Time Created\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TRUSTED_BINANCE_ROOT_HOSTNAMES = frozenset({"binance.com", "binance.bh"})

logger = logging.getLogger(__name__)


class PluginIngestError(RuntimeError):
    """Fail-closed Chrome-plugin capture ingestion error."""


def _reject_nonfinite_json(value: str) -> None:
    raise PluginIngestError(f"non-finite JSON value is prohibited: {value}")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except PluginIngestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PluginIngestError(f"cannot read capture bundle {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PluginIngestError("capture bundle must be a JSON object")
    return payload


def _require_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PluginIngestError(f"{key} must be a non-empty string")
    return value.strip()


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PluginIngestError(f"{field} must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PluginIngestError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PluginIngestError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_inside(
    path_value: str | Path,
    *,
    root: Path,
    label: str,
    must_exist: bool,
) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    resolved_root = root.resolve()
    if not _is_relative_to(resolved, resolved_root):
        raise PluginIngestError(f"{label} resolves outside workspace: {resolved}")
    if must_exist and not resolved.is_file():
        raise PluginIngestError(f"{label} does not exist: {resolved}")
    return resolved


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PluginIngestError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise PluginIngestError("structured telemetry contains a naive datetime")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise PluginIngestError("structured telemetry contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PluginIngestError(
        f"structured telemetry contains unsupported value {type(value).__name__}"
    )


def _atomic_write_new(path: Path, payload: bytes, *, created: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PluginIngestError(f"immutable target already exists: {path}")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise PluginIngestError(f"immutable target appeared concurrently: {path}")
        os.replace(temp_path, path)
        created.append(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_new_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    created: list[Path],
) -> None:
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PluginIngestError(f"cannot serialize {path.name}: {exc}") from exc
    _atomic_write_new(path, encoded, created=created)


@contextmanager
def _exclusive_ingest_lock(audit_dir: Path) -> Iterator[None]:
    lock_path = audit_dir / "plugin_ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PluginIngestError(
            f"plugin ingestion lock already exists and was not removed: {lock_path}"
        ) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _validate_source_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = (parsed.hostname or "").lower()
    trusted_hostname = any(
        hostname == root_hostname or hostname.endswith(f".{root_hostname}")
        for root_hostname in TRUSTED_BINANCE_ROOT_HOSTNAMES
    )
    if parsed.scheme != "https" or not trusted_hostname:
        raise PluginIngestError(
            "source_url must identify an HTTPS Binance page"
        )
    return value


def _validate_roster_symbols(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PluginIngestError(f"{field} must be a non-empty list")
    symbols = tuple(str(item).strip().upper() for item in value)
    if any(SYMBOL_RE.fullmatch(symbol) is None for symbol in symbols):
        raise PluginIngestError(f"{field} contains an invalid symbol")
    if len(set(symbols)) != len(symbols):
        raise PluginIngestError(f"{field} contains duplicate symbols")
    return symbols


def ingest_capture_bundle(
    bundle_manifest_path: Path,
    *,
    workspace_root: Path,
    live_root: Path,
    audit_dir: Path,
) -> Path:
    """Validate a complete plugin bundle and commit its immutable cycle."""

    workspace = workspace_root.resolve()
    bundle_path = _resolve_inside(
        bundle_manifest_path,
        root=workspace,
        label="capture bundle",
        must_exist=True,
    )
    resolved_live_root = _resolve_inside(
        live_root,
        root=workspace,
        label="live root",
        must_exist=False,
    )
    resolved_audit_dir = _resolve_inside(
        audit_dir,
        root=workspace,
        label="audit directory",
        must_exist=False,
    )
    expected_live_root = (workspace / "Live").resolve()
    if resolved_live_root != expected_live_root:
        raise PluginIngestError(
            f"live root must be the checkout Live directory: {expected_live_root}"
        )
    allowed_audit_root = (workspace / "outputs" / "audits").resolve()
    if not _is_relative_to(resolved_audit_dir, allowed_audit_root):
        raise PluginIngestError("audit directory must be under outputs/audits")
    payload = _load_json_object(bundle_path)
    if payload.get("schema_version") != PLUGIN_CAPTURE_BUNDLE_SCHEMA:
        raise PluginIngestError("unsupported capture bundle schema_version")
    if payload.get("status") != "complete":
        raise PluginIngestError("capture bundle status must be complete")
    if payload.get("source") != "chrome_plugin":
        raise PluginIngestError("capture bundle source must be chrome_plugin")
    if payload.get("authenticated") is not True:
        raise PluginIngestError("capture bundle must assert authenticated=true")

    run_id = _require_text(payload, "run_id")
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise PluginIngestError("run_id contains unsupported characters")
    staging_root = (
        workspace / "outputs" / "runtime" / "chrome_plugin_capture" / run_id
    ).resolve()
    if bundle_path.parent != staging_root:
        raise PluginIngestError(
            "capture bundle must be directly inside its run-owned staging directory"
        )
    page_identity = _require_text(payload, "page_identity")
    source_url = _validate_source_url(_require_text(payload, "source_url"))
    started_at = _parse_utc(
        payload.get("cycle_started_at_utc"), field="cycle_started_at_utc"
    )
    completed_at = _parse_utc(
        payload.get("cycle_completed_at_utc"), field="cycle_completed_at_utc"
    )
    if completed_at < started_at:
        raise PluginIngestError("cycle completion predates cycle start")

    roster_before = _validate_roster_symbols(
        payload.get("roster_before_symbols"), field="roster_before_symbols"
    )
    roster_after = _validate_roster_symbols(
        payload.get("roster_after_symbols"), field="roster_after_symbols"
    )
    if roster_before != roster_after:
        raise PluginIngestError("roster changed during capture")
    count = payload.get("working_row_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise PluginIngestError("working_row_count must be a positive integer")
    if len(roster_before) != count:
        raise PluginIngestError("working_row_count does not match roster")
    captures = payload.get("captures")
    if not isinstance(captures, list) or len(captures) != count:
        raise PluginIngestError("capture count does not match Working-row count")

    capture_records: list[dict[str, Any]] = []
    symbols: list[str] = []
    strategy_ids: list[str] = []
    for index, raw_capture in enumerate(captures):
        if not isinstance(raw_capture, dict):
            raise PluginIngestError(f"captures[{index}] must be an object")
        symbol = _require_text(raw_capture, "symbol").upper()
        strategy_id = _require_text(raw_capture, "strategy_id")
        if SYMBOL_RE.fullmatch(symbol) is None:
            raise PluginIngestError(f"captures[{index}] has invalid symbol")
        if STRATEGY_RE.fullmatch(strategy_id) is None:
            raise PluginIngestError(f"captures[{index}] has invalid strategy_id")
        if raw_capture.get("working_status") != "Working":
            raise PluginIngestError(f"{symbol}: status is not exactly Working")
        if raw_capture.get("capture_status") != "complete":
            raise PluginIngestError(f"{symbol}: capture_status is not complete")
        captured_at = _parse_utc(
            raw_capture.get("captured_at_utc"), field=f"{symbol}.captured_at_utc"
        )
        if captured_at < started_at or captured_at > completed_at:
            raise PluginIngestError(f"{symbol}: capture timestamp is outside cycle")
        deployment_time_lima = _require_text(raw_capture, "deployment_time_lima")
        raw_path = _resolve_inside(
            _require_text(raw_capture, "raw_text_path"),
            root=staging_root,
            label=f"{symbol} raw capture",
            must_exist=True,
        )
        raw_bytes = raw_path.read_bytes()
        expected_hash = _validate_sha256(
            raw_capture.get("raw_text_sha256"),
            field=f"{symbol}.raw_text_sha256",
        )
        observed_hash = _sha256_bytes(raw_bytes)
        if observed_hash != expected_hash:
            raise PluginIngestError(f"{symbol}: raw capture SHA-256 mismatch")
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PluginIngestError(f"{symbol}: raw capture is not UTF-8") from exc
        created_match = CREATED_RE.search(raw_text.replace("\u00a0", " "))
        if created_match is None or created_match.group(1) != deployment_time_lima:
            raise PluginIngestError(f"{symbol}: deployment time mismatch")
        try:
            scanner_entry = _scanner_entry_from_raw(
                raw_text=raw_text,
                expected_symbol=symbol,
                expected_strategy_id=strategy_id,
                captured_at_utc=captured_at,
                source="chrome_plugin",
            )
        except Exception as exc:
            raise PluginIngestError(f"{symbol}: drawer validation failed: {exc}") from exc

        screenshot_path: Path | None = None
        screenshot_hash: str | None = None
        if raw_capture.get("screenshot_path") is not None:
            screenshot_path = _resolve_inside(
                _require_text(raw_capture, "screenshot_path"),
                root=staging_root,
                label=f"{symbol} screenshot",
                must_exist=True,
            )
            screenshot_hash = _validate_sha256(
                raw_capture.get("screenshot_sha256"),
                field=f"{symbol}.screenshot_sha256",
            )
            if _sha256_bytes(screenshot_path.read_bytes()) != screenshot_hash:
                raise PluginIngestError(f"{symbol}: screenshot SHA-256 mismatch")

        symbols.append(symbol)
        strategy_ids.append(strategy_id)
        capture_records.append(
            {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "deployment_time_lima": deployment_time_lima,
                "captured_at": captured_at,
                "raw_bytes": raw_bytes,
                "raw_hash": observed_hash,
                "scanner_entry": _json_safe(scanner_entry),
                "screenshot_path": screenshot_path,
                "screenshot_hash": screenshot_hash,
            }
        )

    if tuple(symbols) != roster_before:
        raise PluginIngestError("capture order/symbols do not match roster")
    if len(set(symbols)) != count:
        raise PluginIngestError("capture symbols are duplicated")
    if len(set(strategy_ids)) != count:
        raise PluginIngestError("capture strategy IDs are duplicated")

    started_lima = started_at.astimezone(LIMA)
    live_date = started_lima.strftime("%Y-%m-%d")
    stamp = started_lima.strftime("%Y%m%d_%H%M%S_lima")
    created: list[Path] = []
    with _exclusive_ingest_lock(resolved_audit_dir):
        try:
            files: list[dict[str, Any]] = []
            for record in capture_records:
                symbol = str(record["symbol"])
                strategy_id = str(record["strategy_id"])
                symbol_dir = resolved_live_root / live_date / symbol
                base = f"private_telemetry_{strategy_id}_{stamp}_{run_id}"
                text_path = symbol_dir / f"{base}.txt"
                json_path = symbol_dir / f"{base}.json"
                _atomic_write_new(
                    text_path,
                    bytes(record["raw_bytes"]),
                    created=created,
                )
                metadata: dict[str, Any] = {
                    "schema_version": PLUGIN_SNAPSHOT_SCHEMA,
                    "data_class": "live_bot_telemetry",
                    "status": "active_live_snapshot",
                    "source": "chrome_plugin",
                    "run_id": run_id,
                    "page_identity": page_identity,
                    "source_url": source_url,
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "created_at_lima": record["deployment_time_lima"],
                    "captured_at_utc": record["captured_at"].isoformat(),
                    "captured_at_lima": record["captured_at"].astimezone(LIMA).isoformat(),
                    "raw_text_sha256": record["raw_hash"],
                    "structured_telemetry": record["scanner_entry"],
                }
                file_record: dict[str, Any] = {
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "text_path": str(text_path),
                    "json_path": str(json_path),
                    "raw_text_sha256": record["raw_hash"],
                }
                screenshot_path = record["screenshot_path"]
                if isinstance(screenshot_path, Path):
                    suffix = screenshot_path.suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                        raise PluginIngestError(
                            f"{symbol}: unsupported screenshot extension {suffix}"
                        )
                    final_screenshot = symbol_dir / f"{base}{suffix}"
                    _atomic_write_new(
                        final_screenshot,
                        screenshot_path.read_bytes(),
                        created=created,
                    )
                    metadata["screenshot_path"] = str(final_screenshot)
                    metadata["screenshot_sha256"] = record["screenshot_hash"]
                    file_record["screenshot_path"] = str(final_screenshot)
                    file_record["screenshot_sha256"] = record["screenshot_hash"]
                _atomic_write_new_json(json_path, metadata, created=created)
                files.append(file_record)

            cycle_manifest: dict[str, Any] = {
                "schema_version": PLUGIN_CYCLE_SCHEMA,
                "status": "complete",
                "source": "chrome_plugin",
                "run_id": run_id,
                "page_identity": page_identity,
                "source_url": source_url,
                "capture_bundle_manifest": str(bundle_path),
                "live_root": str(resolved_live_root),
                "cycle_started_at_utc": started_at.isoformat(),
                "cycle_completed_at_utc": completed_at.isoformat(),
                "working_row_count": count,
                "active_bot_count": count,
                "symbols": symbols,
                "strategy_ids": strategy_ids,
                "roster_before_symbols": list(roster_before),
                "roster_after_symbols": list(roster_after),
                "files": files,
            }
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=("symbol", "strategy_id", "deploy_ts"),
                lineterminator="\n",
            )
            writer.writeheader()
            for record in capture_records:
                structured = record["scanner_entry"]
                if not isinstance(structured, dict):
                    raise PluginIngestError("structured scanner entry is unavailable")
                writer.writerow(
                    {
                        "symbol": record["symbol"],
                        "strategy_id": record["strategy_id"],
                        "deploy_ts": structured["deploy_ts"],
                    }
                )
            targets_bytes = csv_buffer.getvalue().encode("utf-8")
            targets_path = (
                resolved_audit_dir / "targets" / f"targets_{stamp}_{run_id}.csv"
            )
            _atomic_write_new(targets_path, targets_bytes, created=created)
            cycle_manifest["collector_targets_csv"] = str(targets_path)
            cycle_manifest["collector_targets_sha256"] = _sha256_bytes(targets_bytes)
            cycle_path = (
                resolved_audit_dir / "cycles" / f"cycle_{stamp}_{run_id}.json"
            )
            _atomic_write_new_json(cycle_path, cycle_manifest, created=created)
            return cycle_path
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "outputs" / "audits" / "chrome_plugin_telemetry",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cycle_path = ingest_capture_bundle(
            args.bundle_manifest,
            workspace_root=args.workspace_root,
            live_root=args.live_root,
            audit_dir=args.audit_dir,
        )
    except PluginIngestError as exc:
        logger.error("Chrome-plugin telemetry bundle rejected: %s", exc)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "source": "chrome_plugin",
                "cycle_manifest": str(cycle_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
