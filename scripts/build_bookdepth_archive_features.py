"""Build candidate-safe features from local Binance Vision bookDepth archives."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.bookdepth_archive import (  # noqa: E402
    BOOKDEPTH_SCHEMA_VERSION,
    build_snapshot_features,
    build_window_diagnostics,
    load_archive_frames_for_target,
    load_bookdepth_targets,
)


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate CSV/XLSX/Parquet with symbol and scan timestamp")
    parser.add_argument("--archive-root", required=True, help="Local bookDepth archive root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--fallback-position-usdt", type=float, default=None)
    parser.add_argument("--lookback-hours", type=float, default=1.0)
    parser.add_argument("--forward-hours", type=float, default=7.0)
    parser.add_argument("--max-snapshot-lag-seconds", type=float, default=None)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    archive_root = Path(args.archive_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_bookdepth_targets(
        input_path,
        max_candidates=args.max_candidates,
        fallback_position_usdt=args.fallback_position_usdt,
    )

    frame_cache: dict[Path, pd.DataFrame] = {}
    feature_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []

    for target in targets:
        frames, paths, errors = load_archive_frames_for_target(
            target,
            archive_root,
            lookback_hours=float(args.lookback_hours),
            forward_hours=float(args.forward_hours),
            cache=frame_cache,
        )
        parse_errors.extend(errors)
        snapshot = build_snapshot_features(
            target,
            frames,
            max_snapshot_lag_seconds=args.max_snapshot_lag_seconds,
        )
        snapshot.features["bookdepth_archive_paths"] = ";".join(str(path) for path in paths)
        feature_rows.append(snapshot.features)

        diagnostics = build_window_diagnostics(
            target,
            frames,
            forward_hours=float(args.forward_hours),
        )
        diagnostics["bookdepth_archive_paths"] = ";".join(str(path) for path in paths)
        diagnostic_rows.append(diagnostics)

    features = pd.DataFrame(feature_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    features_path = output_dir / "bookdepth_archive_features.csv"
    diagnostics_path = output_dir / "bookdepth_archive_window_diagnostics.csv"
    features.to_csv(features_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    availability = cast(pd.Series | None, features.get("bookdepth_archive_available"))
    available_rows = (
        int(cast(pd.Series, pd.to_numeric(availability, errors="coerce")).fillna(0).sum())
        if availability is not None
        else 0
    )
    window_availability = cast(pd.Series | None, diagnostics_df.get("bookdepth_window_available"))
    window_rows = (
        int(cast(pd.Series, pd.to_numeric(window_availability, errors="coerce")).fillna(0).sum())
        if window_availability is not None
        else 0
    )
    lag_summary: dict[str, float | int | None] = {"min": None, "median": None, "p95": None, "max": None}
    if "bookdepth_snapshot_lag_seconds" in features.columns:
        lags = cast(pd.Series, pd.to_numeric(cast(pd.Series, features["bookdepth_snapshot_lag_seconds"]), errors="coerce")).dropna()
        if not lags.empty:
            lag_summary = {
                "min": float(lags.min()),
                "median": float(lags.median()),
                "p95": float(lags.quantile(0.95)),
                "max": float(lags.max()),
                "rows_lag_gt_900s": int((lags > 900).sum()),
                "rows_lag_gt_3600s": int((lags > 3600).sum()),
            }

    manifest = {
        "schema_version": BOOKDEPTH_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(input_path),
        "archive_root": str(archive_root),
        "output_dir": str(output_dir),
        "target_count": len(targets),
        "feature_rows": int(len(features)),
        "available_feature_rows": available_rows,
        "feature_coverage_rate": float(available_rows / len(features)) if len(features) else 0.0,
        "window_rows": int(len(diagnostics_df)),
        "available_window_rows": window_rows,
        "window_coverage_rate": float(window_rows / len(diagnostics_df)) if len(diagnostics_df) else 0.0,
        "unique_archive_files_loaded": len(frame_cache),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:50],
        "snapshot_lag_seconds": lag_summary,
        "feature_output": str(features_path),
        "window_diagnostics_output": str(diagnostics_path),
        "note": "Pre-scan bookDepth columns are ex-ante features. Window diagnostics are forward evidence and must not enter training features.",
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
    }
    manifest_path = output_dir / "bookdepth_archive_manifest.json"
    _write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True)[:4000])
    return 0 if available_rows > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
