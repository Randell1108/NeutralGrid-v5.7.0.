"""Build candidate-level depth feature tables from depth-shadow snapshots."""

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

from neutralgrid.data.depth_shadow import build_candidate_depth_feature_frames  # noqa: E402


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


def _read_depth_records(path: Path) -> pd.DataFrame:
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
    raise ValueError(f"Unsupported depth-shadow input format: {path}")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Depth-shadow audit dir, CSV, JSONL, or Parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for feature and diagnostic outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_depth_records(input_path)
    exante, diagnostics = build_candidate_depth_feature_frames(records)

    exante_path = output_dir / "depth_exante_features.csv"
    diagnostics_path = output_dir / "depth_window_diagnostics.csv"
    exante.to_csv(exante_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "input_rows": int(len(records)),
        "candidate_rows": int(len(exante)),
        "diagnostic_rows": int(len(diagnostics)),
        "exante_features": str(exante_path),
        "window_diagnostics": str(diagnostics_path),
        "leakage_note": (
            "depth_exante_features uses only the first captured snapshot per candidate; "
            "depth_window_diagnostics uses the full captured window and is not a model feature table."
        ),
        "command": " ".join(sys.argv),
    }
    _write_manifest(output_dir / "depth_feature_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
