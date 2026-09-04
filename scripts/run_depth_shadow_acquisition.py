"""Prepare or start a fresh 7h depth-shadow acquisition run.

This is an orchestration guard for the depth-aware deployable-winner loop. It
does not train, promote, or modify gates. By default it validates that candidate
rows are fresh enough to be ex-ante evidence, writes a reproducible acquisition
manifest, and stops. Use --start to run the underlying collector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.depth_shadow import (  # noqa: E402
    DepthShadowRejectedTarget,
    DepthShadowTarget,
    filter_fresh_depth_shadow_targets,
    load_depth_shadow_targets,
)


DEFAULT_FORWARD_WINDOW_SECONDS = 7 * 60 * 60
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_MAX_SCAN_AGE_SECONDS = 900.0


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _latest_deployment_csv() -> Path:
    paths = sorted((ROOT / "results").glob("deployment_ready_*.csv"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError("No results/deployment_ready_*.csv files found; pass --input or --symbols")
    return paths[-1]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse_now(value: str | None) -> datetime:
    if value is None:
        return _utc_now()
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid --now-utc timestamp: {value}")
    return parsed.to_pydatetime()


def _manual_targets(symbols: Sequence[str], *, now: datetime, fallback_position_usdt: float | None) -> list[DepthShadowTarget]:
    stamp = now.strftime("%Y%m%d_%H%M%S")
    scan_time_utc = now.isoformat()
    return [
        DepthShadowTarget(
            symbol=symbol.upper(),
            candidate_id=f"manual_{symbol.upper()}_{stamp}",
            scan_time_utc=scan_time_utc,
            position_notional_usdt=fallback_position_usdt,
            source_row_index=None,
            source_path="--symbols",
        )
        for symbol in symbols
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_rejected(path: Path, rejected: Sequence[DepthShadowRejectedTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "symbol",
        "candidate_id",
        "scan_time_utc",
        "reason",
        "scan_age_seconds",
        "source_row_index",
        "source_path",
    ]
    pd.DataFrame([asdict(target) for target in rejected], columns=pd.Index(columns)).to_csv(path, index=False)


def _write_targets(path: Path, targets: Sequence[DepthShadowTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "symbol",
        "candidate_id",
        "scan_time_utc",
        "position_notional_usdt",
        "source_row_index",
        "source_path",
    ]
    pd.DataFrame([asdict(target) for target in targets], columns=pd.Index(columns)).to_csv(path, index=False)


def _filter_positive_position_targets(
    targets: Sequence[DepthShadowTarget],
) -> tuple[list[DepthShadowTarget], list[DepthShadowRejectedTarget]]:
    accepted: list[DepthShadowTarget] = []
    rejected: list[DepthShadowRejectedTarget] = []
    for target in targets:
        position = target.position_notional_usdt
        if position is None:
            rejected.append(
                DepthShadowRejectedTarget(
                    symbol=target.symbol,
                    candidate_id=target.candidate_id,
                    scan_time_utc=target.scan_time_utc,
                    reason="missing_position_notional",
                    source_row_index=target.source_row_index,
                    source_path=target.source_path,
                )
            )
            continue
        if position <= 0:
            rejected.append(
                DepthShadowRejectedTarget(
                    symbol=target.symbol,
                    candidate_id=target.candidate_id,
                    scan_time_utc=target.scan_time_utc,
                    reason="non_positive_position_notional",
                    source_row_index=target.source_row_index,
                    source_path=target.source_path,
                )
            )
            continue
        accepted.append(target)
    return accepted, rejected


def _collector_command(args: argparse.Namespace, *, input_path: Path | None, collector_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_depth_shadow.py"),
        "--output-dir",
        str(collector_dir),
        "--duration-seconds",
        str(args.duration_seconds),
        "--interval-seconds",
        str(args.interval_seconds),
        "--iteration-timeout-seconds",
        str(args.iteration_timeout_seconds),
        "--limit",
        str(args.limit),
        "--top-n",
        str(args.top_n),
        "--participation-rate",
        str(args.participation_rate),
        "--max-scan-age-seconds",
        str(args.max_scan_age_seconds),
        "--max-candidates",
        str(args.max_candidates),
    ]
    if args.fallback_position_usdt is not None:
        command.extend(["--fallback-position-usdt", str(args.fallback_position_usdt)])
    if args.symbols:
        command.append("--symbols")
        command.extend([str(symbol).upper() for symbol in args.symbols])
    elif input_path is not None:
        command.extend(["--input", str(input_path)])
    return command


def _load_targets(args: argparse.Namespace, *, input_path: Path | None, now: datetime) -> list[DepthShadowTarget]:
    if args.symbols:
        return _manual_targets(args.symbols, now=now, fallback_position_usdt=args.fallback_position_usdt)
    if input_path is None:
        raise ValueError("input_path is required when --symbols is not supplied")
    return load_depth_shadow_targets(
        input_path,
        max_candidates=None if args.require_positive_position else args.max_candidates,
        fallback_position_usdt=args.fallback_position_usdt,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Fresh deployment_ready CSV. Defaults to latest results/deployment_ready_*.csv")
    parser.add_argument("--symbols", nargs="*", help="Manual live/paper symbols; bypasses deployment CSV")
    parser.add_argument("--output-dir", help="Acquisition audit directory")
    parser.add_argument("--start", action="store_true", help="Actually run collect_depth_shadow.py after validation")
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_FORWARD_WINDOW_SECONDS)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--iteration-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--max-scan-age-seconds", type=float, default=DEFAULT_MAX_SCAN_AGE_SECONDS)
    parser.add_argument("--fallback-position-usdt", type=float)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--now-utc", help="Testing/audit clock override, ISO UTC")
    parser.add_argument(
        "--require-positive-position",
        action="store_true",
        help=(
            "Reject candidates without positive position notional before collection. "
            "Use for depth-oracle label acquisition; leave off for generic book snapshots."
        ),
    )
    args = parser.parse_args(argv)
    if args.input and args.symbols:
        parser.error("Use either --input or --symbols, not both")
    if args.duration_seconds <= 0:
        parser.error("--duration-seconds must be > 0")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be > 0")
    if args.iteration_timeout_seconds <= 0:
        parser.error("--iteration-timeout-seconds must be > 0")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be > 0")
    if args.max_scan_age_seconds <= 0:
        parser.error("--max-scan-age-seconds must be > 0")
    if args.limit <= 0 or args.top_n <= 0:
        parser.error("--limit and --top-n must be > 0")
    if args.top_n > args.limit:
        parser.error("--top-n cannot exceed --limit")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = _parse_now(args.now_utc)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "audits" / f"depth_shadow_acquisition_{run_id}"
    collector_dir = output_dir / "collector"
    rejected_path = output_dir / "rejected_targets.csv"
    targets_path = output_dir / "targets.csv"
    manifest_path = output_dir / "acquisition_manifest.json"
    input_path = None if args.symbols else Path(args.input) if args.input else _latest_deployment_csv()

    targets = _load_targets(args, input_path=input_path, now=now)
    candidate_targets = targets
    rejected: list[DepthShadowRejectedTarget] = []
    if args.require_positive_position:
        candidate_targets, sizing_rejected = _filter_positive_position_targets(candidate_targets)
        rejected.extend(sizing_rejected)

    fresh, freshness_rejected = filter_fresh_depth_shadow_targets(
        candidate_targets,
        now=now,
        max_scan_age_seconds=float(args.max_scan_age_seconds),
    )
    rejected.extend(freshness_rejected)
    fresh = fresh[: args.max_candidates]
    if rejected:
        _write_rejected(rejected_path, rejected)
    _write_targets(targets_path, fresh)

    collector_input_path = None if args.symbols else targets_path
    command = _collector_command(args, input_path=collector_input_path, collector_dir=collector_dir)
    manifest: dict[str, Any] = {
        "created_at_utc": now.isoformat(),
        "status": "ready_not_started" if fresh else "blocked_no_fresh_targets",
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "input_path": str(input_path) if input_path is not None else "--symbols",
        "output_dir": str(output_dir),
        "collector_dir": str(collector_dir),
        "targets_csv": str(targets_path),
        "rejected_targets_csv": str(rejected_path),
        "original_target_count": len(targets),
        "target_count": len(fresh),
        "rejected_target_count": len(rejected),
        "require_positive_position": bool(args.require_positive_position),
        "symbols": sorted({target.symbol for target in fresh}),
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "iteration_timeout_seconds": args.iteration_timeout_seconds,
        "expected_snapshots_per_candidate": int(args.duration_seconds // args.interval_seconds) + 1,
        "max_scan_age_seconds": args.max_scan_age_seconds,
        "collector_command": command,
        "started": bool(args.start),
        "note": (
            "This runner only acquires candidate-keyed L2 shadow evidence. "
            "It does not retrain or modify production model/gate artifacts."
        ),
    }
    if not fresh:
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 2
    if not args.start:
        _write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    result = subprocess.run(command, cwd=ROOT, check=False)
    manifest["collector_returncode"] = result.returncode
    manifest["status"] = "collector_complete" if result.returncode == 0 else "collector_failed"
    _write_json(manifest_path, manifest)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
