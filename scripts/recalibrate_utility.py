"""Operator CLI for governed utility-score recalibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.calibration.utility_calibrator import run_calibration  # noqa: E402


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalibrate governed utility-score coefficients from expired bot outcomes.",
    )
    parser.add_argument(
        "--expired-bots-path",
        type=Path,
        default=Path("data/new_expired_bots.xlsx"),
        help=(
            "Workbook path: canonical General+Meta Features workbook or a "
            "compatible derived flat backfill staging file."
        ),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/utility"),
        help="Directory for candidate utility artifacts and current.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run calibration and print gates without writing artifacts.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote to current.json only if every promotion gate passes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_calibration(
        expired_bots_path=args.expired_bots_path,
        artifact_dir=args.artifact_dir,
        write_candidate=not args.dry_run,
        promote=args.promote,
    )
    report = {
        "candidate_path": str(result.candidate_path) if result.candidate_path else None,
        "current_path": str(result.current_path),
        "promoted": bool(args.promote and result.promotable),
        "artifact": result.artifact,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
