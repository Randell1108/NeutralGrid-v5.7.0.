"""Create or reuse exact-fingerprint validation evidence for PnL acquisition.

The first run for a source fingerprint executes the focused test suite and
Pyright, records exact commands and results, and writes an immutable evidence
artifact.  Later runs reuse only an exact successful artifact for the same
fingerprint.  A missing prior artifact therefore bootstraps validation instead
of creating a permanent fail-closed deadlock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
EVIDENCE_SCHEMA = "neutralgrid_pnl_evidence_validation_v1"
DEFAULT_RELEVANT_FILES = (
    "scripts/validate_pnl_evidence_runtime.py",
    "scripts/ingest_chrome_plugin_telemetry_cycle.py",
    "scripts/run_live_telemetry_controller.py",
    "scripts/collect_diff_depth.py",
    "scripts/supervise_diff_depth_collector.py",
    "scripts/audit_pnl_training_readiness.py",
    "src/neutralgrid/live/decision/pnl_history.py",
    "src/neutralgrid/live/decision/pnl_forecast.py",
    "tests/unit/test_chrome_plugin_telemetry_ingest.py",
    "tests/unit/test_live_pnl_history.py",
    "tests/unit/test_live_pnl_forecast.py",
    "tests/unit/test_live_telemetry_controller.py",
    "tests/unit/test_private_grid_telemetry.py",
    "tests/unit/test_pnl_evidence_validation.py",
    "tests/unit/test_pnl_training_readiness.py",
    "tests/unit/test_diff_depth_collector_supervisor.py",
)
FOCUSED_TESTS = (
    "tests/unit/test_chrome_plugin_telemetry_ingest.py",
    "tests/unit/test_live_pnl_history.py",
    "tests/unit/test_live_pnl_forecast.py",
    "tests/unit/test_live_telemetry_controller.py",
    "tests/unit/test_private_grid_telemetry.py",
    "tests/unit/test_pnl_evidence_validation.py",
    "tests/unit/test_pnl_training_readiness.py",
    "tests/unit/test_diff_depth_collector_supervisor.py",
)


class ValidationEvidenceError(RuntimeError):
    """Fail-closed source-validation evidence error."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValidationEvidenceError(f"non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except ValidationEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationEvidenceError(f"cannot read evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationEvidenceError(f"{path}: evidence must be an object")
    return payload


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValidationEvidenceError(f"immutable evidence already exists: {path}")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise ValidationEvidenceError(
                f"immutable evidence appeared concurrently: {path}"
            )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _default_runner(
    command: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _run_git(workspace_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ValidationEvidenceError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _resolve_relevant_files(
    workspace_root: Path,
    relevant_files: Sequence[Path],
) -> tuple[Path, ...]:
    root = workspace_root.resolve()
    resolved: list[Path] = []
    for value in relevant_files:
        path = value if value.is_absolute() else root / value
        path = path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValidationEvidenceError(
                f"relevant file resolves outside workspace: {path}"
            ) from exc
        if not path.is_file():
            raise ValidationEvidenceError(f"relevant file is missing: {path}")
        resolved.append(path)
    return tuple(resolved)


def source_fingerprint(
    *,
    workspace_root: Path,
    relevant_files: Sequence[Path],
    git_head: str,
    git_status: str,
) -> tuple[str, list[dict[str, str]]]:
    root = workspace_root.resolve()
    file_records: list[dict[str, str]] = []
    for path in _resolve_relevant_files(root, relevant_files):
        file_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    material = {
        "git_head": git_head,
        "git_status": git_status,
        "files": file_records,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), file_records


def ensure_validation_evidence(
    *,
    workspace_root: Path,
    evidence_dir: Path,
    relevant_files: Sequence[Path],
    commands: Sequence[Sequence[str]],
    runner: Runner = _default_runner,
    git_head: str | None = None,
    git_status: str | None = None,
) -> dict[str, Any]:
    """Return exact-fingerprint PASS evidence, running checks when absent."""

    root = workspace_root.resolve()
    resolved_evidence_dir = evidence_dir.resolve()
    try:
        resolved_evidence_dir.relative_to(root)
    except ValueError as exc:
        raise ValidationEvidenceError("evidence directory is outside workspace") from exc
    head = git_head if git_head is not None else _run_git(root, "rev-parse", "HEAD")
    status = (
        git_status
        if git_status is not None
        else _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    )
    fingerprint, file_records = source_fingerprint(
        workspace_root=root,
        relevant_files=relevant_files,
        git_head=head,
        git_status=status,
    )
    pass_path = resolved_evidence_dir / f"validation_{fingerprint}.json"
    if pass_path.exists():
        existing = _strict_json_object(pass_path)
        if (
            existing.get("schema_version") != EVIDENCE_SCHEMA
            or existing.get("status") != "pass"
            or existing.get("source_fingerprint") != fingerprint
        ):
            raise ValidationEvidenceError(
                f"existing exact-fingerprint evidence is invalid: {pass_path}"
            )
        return {
            **existing,
            "evidence_path": str(pass_path),
            "reused": True,
        }

    started_at = _utc_now()
    command_records: list[dict[str, Any]] = []
    all_passed = True
    for command in commands:
        started = time.perf_counter()
        result = runner(command, cwd=root)
        duration = time.perf_counter() - started
        command_records.append(
            {
                "command": list(command),
                "exit_code": int(result.returncode),
                "duration_seconds": round(duration, 6),
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            all_passed = False
            break
    completed_at = _utc_now()
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "pass" if all_passed else "fail",
        "source_fingerprint": fingerprint,
        "git_head": head,
        "git_status": status.splitlines(),
        "relevant_files": file_records,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "commands": command_records,
    }
    if all_passed:
        output_path = pass_path
    else:
        stamp = completed_at.strftime("%Y%m%d_%H%M%S")
        output_path = resolved_evidence_dir / f"validation_{fingerprint}_failed_{stamp}.json"
    _atomic_write_new_json(output_path, evidence)
    result_payload = {
        **evidence,
        "evidence_path": str(output_path),
        "reused": False,
    }
    if not all_passed:
        raise ValidationEvidenceError(
            f"source validation failed; evidence={output_path}"
        )
    return result_payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=ROOT / "outputs" / "audits" / "pnl_evidence_validation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    python = workspace_root / ".venv" / "Scripts" / "python.exe"
    pyright = workspace_root / ".venv" / "Scripts" / "pyright.exe"
    commands = [
        [str(python), "-m", "pytest", *FOCUSED_TESTS, "-q"],
        [str(pyright)],
    ]
    try:
        evidence = ensure_validation_evidence(
            workspace_root=workspace_root,
            evidence_dir=args.evidence_dir,
            relevant_files=[workspace_root / value for value in DEFAULT_RELEVANT_FILES],
            commands=commands,
        )
    except ValidationEvidenceError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "source_fingerprint": evidence["source_fingerprint"],
                "evidence_path": evidence["evidence_path"],
                "reused": evidence["reused"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
