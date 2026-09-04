"""Build fail-closed depth-aware deployability labels from L2 shadow data.

The script writes the required oracle artifacts but exits non-zero when the
available evidence cannot produce any labelable rows. It does not modify gates,
configs, or production model artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.depth_oracle import (  # noqa: E402
    DepthOracleColumns,
    DepthOracleConfig,
    build_depth_oracle_labels,
    manifest_with_outputs,
)


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


def _read_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        summary = path / "depth_shadow_summary.csv"
        records = path / "depth_shadow_records.jsonl"
        if summary.exists():
            return pd.read_csv(summary)
        if records.exists():
            return pd.read_json(records, lines=True)
        raise FileNotFoundError(f"No depth_shadow_summary.csv or depth_shadow_records.jsonl in {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {path}")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-input", required=True, help="Depth-shadow audit dir, CSV, JSONL, or Parquet")
    parser.add_argument("--outcomes", required=True, help="Forward outcome table with candidate_id")
    parser.add_argument("--output-dir", required=True, help="Directory for oracle artifacts")
    parser.add_argument("--pnl-col", help="Outcome PnL percent column. Auto-detected only from known names.")
    parser.add_argument("--time-col", help="Time-to-target hours column. Auto-detected only from known names.")
    parser.add_argument("--tail-col", help="Tail-risk PnL percent column. Auto-detected only from known names.")
    parser.add_argument("--target-hit-col", help="Boolean target-hit column. Auto-detected only from known names.")
    parser.add_argument("--horizon-hours", type=float, default=7.0)
    parser.add_argument("--target-pnl-pct", type=float, default=3.0)
    parser.add_argument("--max-tail-loss-pct", type=float, default=-20.0)
    parser.add_argument("--min-fill-ratio", type=float, default=1.0)
    parser.add_argument("--min-capacity-ratio", type=float, default=1.0)
    parser.add_argument("--min-snapshots", type=int, default=2)
    parser.add_argument("--min-window-hours", type=float)
    parser.add_argument("--pnl-is-net-of-costs", action="store_true")
    parser.add_argument("--max-spread-pct", type=float)
    parser.add_argument("--max-impact-bps", type=float)
    args = parser.parse_args(argv)
    if args.horizon_hours <= 0:
        parser.error("--horizon-hours must be > 0")
    if args.min_snapshots <= 0:
        parser.error("--min-snapshots must be > 0")
    if args.min_window_hours is not None and args.min_window_hours <= 0:
        parser.error("--min-window-hours must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    depth_input = Path(args.depth_input)
    outcomes_input = Path(args.outcomes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DepthOracleConfig(
        horizon_hours=args.horizon_hours,
        target_pnl_pct=args.target_pnl_pct,
        max_tail_loss_pct=args.max_tail_loss_pct,
        min_fill_ratio=args.min_fill_ratio,
        min_capacity_ratio=args.min_capacity_ratio,
        min_snapshots=args.min_snapshots,
        min_window_hours=args.min_window_hours,
        pnl_is_net_of_costs=args.pnl_is_net_of_costs,
        max_spread_pct=args.max_spread_pct,
        max_impact_bps=args.max_impact_bps,
    )
    columns = DepthOracleColumns(
        pnl_pct=args.pnl_col,
        time_to_target_hours=args.time_col,
        tail_pnl_pct=args.tail_col,
        target_hit=args.target_hit_col,
    )
    result = build_depth_oracle_labels(
        _read_table(depth_input),
        _read_table(outcomes_input),
        config=config,
        columns=columns,
    )

    labels_path = output_dir / "depth_labels.csv"
    diagnostics_path = output_dir / "replay_diagnostics.csv"
    failed_path = output_dir / "failed_fill_reasons.csv"
    result.labels.to_csv(labels_path, index=False)
    result.replay_diagnostics.to_csv(diagnostics_path, index=False)
    result.failed_fill_reasons.to_csv(failed_path, index=False)

    manifest = manifest_with_outputs(
        result.manifest,
        labels_path=str(labels_path),
        diagnostics_path=str(diagnostics_path),
        failed_reasons_path=str(failed_path),
    )
    manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
            "git_status_short": _git_output(["status", "--short"]),
            "depth_input": str(depth_input),
            "outcomes_input": str(outcomes_input),
            "output_dir": str(output_dir),
            "command": " ".join(sys.argv),
        }
    )
    _write_manifest(output_dir / "depth_oracle_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if result.manifest["labelable_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
