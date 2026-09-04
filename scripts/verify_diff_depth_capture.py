"""Deterministically replay and verify stored diff-depth capture artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, cast


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.diff_depth import (  # noqa: E402
    DiffDepthError,
    atomic_write_json,
    replay_symbol_capture,
    verification_to_dict,
)


def _load_run_dirs(manifest_path: Path) -> tuple[Path, list[Path]]:
    decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise DiffDepthError("run manifest must be an object")
    payload = cast(dict[str, Any], decoded)
    raw_dirs = payload.get("symbol_run_dirs")
    if not isinstance(raw_dirs, dict) or not raw_dirs:
        raise DiffDepthError("run manifest has no symbol_run_dirs")
    run_dirs = [Path(str(value)) for value in raw_dirs.values()]
    return manifest_path.parent, run_dirs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Collector run manifest")
    source.add_argument("--symbol-run-dir", help="One symbol diff_depth run directory")
    parser.add_argument("--output", help="Verification report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.manifest:
        report_root, run_dirs = _load_run_dirs(Path(args.manifest))
    else:
        run_dir = Path(args.symbol_run_dir)
        report_root = run_dir
        run_dirs = [run_dir]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            result = replay_symbol_capture(run_dir)
        except Exception as exc:
            errors.append({"run_dir": str(run_dir), "error": repr(exc)})
        else:
            results.append(
                {
                    "run_dir": str(run_dir),
                    **verification_to_dict(result),
                }
            )
    passed = not errors and bool(results) and all(
        bool(result["passed"]) for result in results
    )
    report = {
        "passed": passed,
        "symbol_count": len(run_dirs),
        "verified_symbol_count": len(results),
        "results": results,
        "errors": errors,
    }
    output_path = (
        Path(args.output)
        if args.output
        else report_root / "replay_verification.json"
    )
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
