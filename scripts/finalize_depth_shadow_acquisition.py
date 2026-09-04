"""Finalize a completed depth-shadow acquisition into oracle/dataset artifacts.

The long-running collector only captures L2 evidence. This helper performs the
post-run audit steps once the collector manifest is complete: reprice raw books
with corrected sizing, build ex-ante features, build forward outcomes, build
depth-oracle labels, and aggregate oracle bundles into a training frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

ROOT = Path(__file__).resolve().parents[1]


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_step(name: str, command: Sequence[str], *, allow_nonzero: bool = False) -> dict[str, Any]:
    result = subprocess.run(
        list(command),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "name": name,
        "command": list(command),
        "returncode": int(result.returncode),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "allowed_nonzero": bool(allow_nonzero),
        "ok": result.returncode == 0 or allow_nonzero,
    }


def _collector_dir(acquisition_dir: Path) -> Path:
    if (acquisition_dir / "depth_shadow_manifest.json").exists():
        return acquisition_dir
    return acquisition_dir / "collector"


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _depth_window_coverage_seconds(manifest: dict[str, Any]) -> float | None:
    started_at = _parse_utc(manifest.get("started_at_utc"))
    last_capture = _parse_utc(
        manifest.get("last_successful_capture_time_utc")
        or manifest.get("last_capture_time_utc")
    )
    if started_at is None or last_capture is None:
        return None
    return max(0.0, float((last_capture - started_at).total_seconds()))


def _required_depth_window_seconds(manifest: dict[str, Any]) -> float | None:
    raw_duration = manifest.get("duration_seconds")
    if raw_duration is None:
        return None
    try:
        duration = float(raw_duration)
        interval = float(manifest.get("interval_seconds", 60.0))
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return max(0.0, duration - max(0.0, interval))


def _oracle_min_window_hours(required_window_seconds: float | None) -> float | None:
    if required_window_seconds is None:
        return None
    return max(0.0, float(required_window_seconds) / 3600.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--previous-oracle-dir", action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--hours", type=int, default=8)
    parser.add_argument("--min-bars", type=int, default=420)
    parser.add_argument("--max-holding-bars", type=int, default=420)
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be > 0")
    if args.min_bars <= 0:
        parser.error("--min-bars must be > 0")
    if args.max_holding_bars <= 0:
        parser.error("--max-holding-bars must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    acquisition_dir = Path(args.acquisition_dir)
    candidates = Path(args.candidates)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    finalize_manifest_path = output_dir / "depth_shadow_finalize_manifest.json"

    collector_dir = _collector_dir(acquisition_dir)
    collector_manifest_path = collector_dir / "depth_shadow_manifest.json"
    if not collector_manifest_path.exists():
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "blocked_missing_collector_manifest",
            "collector_manifest": str(collector_manifest_path),
        }
        _write_json(finalize_manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    collector_manifest = _read_json(collector_manifest_path)
    depth_window_seconds = _depth_window_coverage_seconds(collector_manifest)
    required_window_seconds = _required_depth_window_seconds(collector_manifest)
    collector_complete = collector_manifest.get("status") == "complete"
    if not collector_complete and not args.allow_incomplete:
        if collector_manifest.get("status") == "complete_depth_window_incomplete":
            manifest = {
                "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "status": "blocked_depth_window_incomplete",
                "collector_status": collector_manifest.get("status"),
                "collector_manifest": str(collector_manifest_path),
                "depth_window_seconds": depth_window_seconds,
                "required_depth_window_seconds": required_window_seconds,
                "note": (
                    "Collector wrote records but marked the depth window incomplete. "
                    "Use --allow-incomplete only for diagnostics; do not train or "
                    "promote from this bundle."
                ),
            }
            _write_json(finalize_manifest_path, manifest)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 2
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "blocked_collector_incomplete",
            "collector_status": collector_manifest.get("status"),
            "collector_manifest": str(collector_manifest_path),
            "note": "Use after the depth-shadow collector writes status=complete, or pass --allow-incomplete for diagnostics only.",
        }
        _write_json(finalize_manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2
    depth_window_complete = (
        depth_window_seconds is not None
        and required_window_seconds is not None
        and depth_window_seconds >= required_window_seconds
    )
    if not depth_window_complete and not args.allow_incomplete:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "blocked_depth_window_incomplete",
            "collector_status": collector_manifest.get("status"),
            "collector_manifest": str(collector_manifest_path),
            "depth_window_seconds": depth_window_seconds,
            "required_depth_window_seconds": required_window_seconds,
            "note": (
                "Collector process completed, but raw depth snapshots do not span "
                "the requested forward window. Use --allow-incomplete only for "
                "diagnostics; do not train or promote from this bundle."
            ),
        }
        _write_json(finalize_manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2

    repriced_dir = output_dir / "repriced_depth"
    features_dir = repriced_dir / "features"
    outcomes_dir = output_dir / "forward_outcomes"
    oracle_dir = output_dir / "oracle"
    dataset_dir = output_dir / "dataset"
    oracle_min_window_hours = _oracle_min_window_hours(required_window_seconds)

    steps: list[dict[str, Any]] = []
    oracle_command = [
        sys.executable,
        str(ROOT / "scripts" / "build_depth_oracle_labels.py"),
        "--depth-input",
        str(repriced_dir),
        "--outcomes",
        str(outcomes_dir / "depth_shadow_forward_outcomes.csv"),
        "--output-dir",
        str(oracle_dir),
        "--pnl-col",
        "net_pnl_pct",
        "--time-col",
        "time_to_target_hours",
        "--tail-col",
        "tail_pnl_pct",
        "--target-hit-col",
        "target_hit",
    ]
    if oracle_min_window_hours is not None:
        oracle_command.extend(["--min-window-hours", str(oracle_min_window_hours)])

    commands: list[tuple[str, list[str], bool]] = [
        (
            "reprice_depth_records",
            [
                sys.executable,
                str(ROOT / "scripts" / "reprice_depth_shadow_records.py"),
                "--records",
                str(collector_dir),
                "--candidates",
                str(candidates),
                "--output-dir",
                str(repriced_dir),
                "--top-n",
                str(collector_manifest.get("top_n", 20)),
                "--participation-rate",
                str(collector_manifest.get("participation_rate", 0.10)),
            ],
            False,
        ),
        (
            "build_depth_features",
            [
                sys.executable,
                str(ROOT / "scripts" / "build_depth_shadow_features.py"),
                "--input",
                str(repriced_dir),
                "--output-dir",
                str(features_dir),
            ],
            False,
        ),
        (
            "build_forward_outcomes",
            [
                sys.executable,
                str(ROOT / "scripts" / "build_depth_shadow_outcomes.py"),
                "--input",
                str(candidates),
                "--output-dir",
                str(outcomes_dir),
                "--hours",
                str(args.hours),
                "--min-bars",
                str(args.min_bars),
                "--max-holding-bars",
                str(args.max_holding_bars),
            ],
            False,
        ),
        (
            "build_depth_oracle",
            oracle_command,
            True,
        ),
    ]
    aggregate_command = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_depth_oracle_dataset.py"),
    ]
    for prev in args.previous_oracle_dir:
        aggregate_command.extend(["--oracle-dir", str(prev)])
    aggregate_command.extend(["--oracle-dir", str(oracle_dir), "--output-dir", str(dataset_dir)])
    commands.append(("aggregate_depth_dataset", aggregate_command, True))

    status = "complete"
    for name, command, allow_nonzero in commands:
        step = _run_step(name, command, allow_nonzero=allow_nonzero)
        steps.append(step)
        if not step["ok"]:
            status = f"failed_{name}"
            break

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "acquisition_dir": str(acquisition_dir),
        "collector_dir": str(collector_dir),
        "candidates": str(candidates),
        "output_dir": str(output_dir),
        "repriced_dir": str(repriced_dir),
        "features_dir": str(features_dir),
        "outcomes_dir": str(outcomes_dir),
        "oracle_dir": str(oracle_dir),
        "dataset_dir": str(dataset_dir),
        "previous_oracle_dirs": [str(path) for path in args.previous_oracle_dir],
        "oracle_min_window_hours": oracle_min_window_hours,
        "collector_manifest": collector_manifest,
        "steps": steps,
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    _write_json(finalize_manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
