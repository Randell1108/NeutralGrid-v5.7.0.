"""Recompute depth-shadow fill metrics from raw captured books.

The collector persists raw bid/ask levels. This helper lets an audit correct
candidate sizing without recollecting the forward window: it reloads the raw
JSONL, derives position notional from the original candidate CSV, and rewrites
summary metrics into a new audit directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.depth_shadow import (  # noqa: E402
    DepthShadowTarget,
    append_jsonl,
    load_depth_shadow_targets,
    make_depth_shadow_record,
)


_TARGET_COLUMNS = {
    "symbol",
    "candidate_id",
    "scan_time_utc",
    "position_notional_usdt",
    "source_row_index",
    "source_path",
}
_SUMMARY_COLUMNS = {
    "schema_version",
    "last_update_id",
    "top_n",
    "participation_rate",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread_pct",
    "top_n_bid_depth_usdt",
    "top_n_ask_depth_usdt",
    "top_n_depth_min_usdt",
    "top_n_depth_total_usdt",
    "book_imbalance_top_n",
    "depth_to_position_min",
    "partial_fill_capacity_usdt",
    "partial_fill_capacity_ratio",
    "raw_bid_levels",
    "raw_ask_levels",
    "buy_fill_ratio",
    "buy_impact_bps",
    "buy_levels_consumed",
    "buy_unfilled_notional_usdt",
    "buy_complete_fill",
    "sell_fill_ratio",
    "sell_impact_bps",
    "sell_levels_consumed",
    "sell_unfilled_notional_usdt",
    "sell_complete_fill",
    "max_side_impact_bps",
    "min_side_fill_ratio",
}
_RAW_BOOK_COLUMNS = {"bids", "asks"}


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _records_path(path: Path) -> Path:
    if path.is_dir():
        records = path / "depth_shadow_records.jsonl"
        if not records.exists():
            raise FileNotFoundError(f"No depth_shadow_records.jsonl in {path}")
        return records
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(cast(dict[str, Any], json.loads(text)))
    return records


def _target_from_raw(raw: Mapping[str, Any]) -> DepthShadowTarget:
    position = raw.get("position_notional_usdt")
    try:
        position_notional = float(position) if position is not None and pd.notna(position) else None
    except (TypeError, ValueError):
        position_notional = None
    return DepthShadowTarget(
        symbol=str(raw.get("symbol", "")).upper(),
        candidate_id=str(raw.get("candidate_id", "")),
        scan_time_utc=str(raw.get("scan_time_utc")) if raw.get("scan_time_utc") is not None else None,
        position_notional_usdt=position_notional,
        source_row_index=None,
        source_path="raw_depth_record",
    )


def _market_context(raw: Mapping[str, Any]) -> dict[str, Any]:
    blocked = _TARGET_COLUMNS | _SUMMARY_COLUMNS | _RAW_BOOK_COLUMNS | {"capture_time_utc"}
    return {str(key): value for key, value in raw.items() if key not in blocked}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="Raw depth-shadow JSONL or collector directory")
    parser.add_argument("--candidates", required=True, help="Original candidate CSV/Parquet/XLSX with sizing columns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--participation-rate", type=float, default=0.10)
    parser.add_argument("--fallback-position-usdt", type=float)
    args = parser.parse_args(argv)
    if args.top_n <= 0:
        parser.error("--top-n must be > 0")
    if args.participation_rate <= 0:
        parser.error("--participation-rate must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records_input = _records_path(Path(args.records))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_depth_shadow_targets(
        Path(args.candidates),
        fallback_position_usdt=args.fallback_position_usdt,
    )
    target_by_id = {target.candidate_id: target for target in targets}
    raw_records = _load_jsonl(records_input)

    repriced: list[dict[str, Any]] = []
    missing_target_count = 0
    for raw in raw_records:
        candidate_id = str(raw.get("candidate_id", ""))
        target = target_by_id.get(candidate_id)
        if target is None:
            missing_target_count += 1
            target = _target_from_raw(raw)
        capture_time = str(raw.get("capture_time_utc"))
        order_book = {
            "lastUpdateId": raw.get("last_update_id"),
            "bids": raw.get("bids") or [],
            "asks": raw.get("asks") or [],
        }
        record = make_depth_shadow_record(
            target,
            order_book,
            capture_time_utc=capture_time,
            top_n=int(args.top_n),
            participation_rate=float(args.participation_rate),
            market_context=_market_context(raw),
        )
        repriced.append(record)

    records_path = output_dir / "depth_shadow_records.jsonl"
    if records_path.exists():
        records_path.unlink()
    append_jsonl(records_path, repriced)

    summary_path = output_dir / "depth_shadow_summary.csv"
    if repriced:
        summary_cols = [col for col in repriced[0].keys() if col not in {"bids", "asks"}]
        pd.DataFrame([{col: record.get(col) for col in summary_cols} for record in repriced]).to_csv(
            summary_path,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(summary_path, index=False)

    position_values = [
        target.position_notional_usdt for target in target_by_id.values() if target.position_notional_usdt is not None
    ]
    zero_position_count = sum(1 for value in position_values if value <= 0)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "records_input": str(records_input),
        "candidates_input": str(args.candidates),
        "output_dir": str(output_dir),
        "records_output": str(records_path),
        "summary_output": str(summary_path),
        "input_records": int(len(raw_records)),
        "repriced_records": int(len(repriced)),
        "target_count": int(len(targets)),
        "missing_target_count": int(missing_target_count),
        "zero_or_negative_position_target_count": int(zero_position_count),
        "top_n": int(args.top_n),
        "participation_rate": float(args.participation_rate),
        "fallback_position_usdt": args.fallback_position_usdt,
        "note": (
            "Recomputed from raw bid/ask levels with corrected candidate sizing. "
            "Original collector artifacts are not modified."
        ),
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "command": " ".join(sys.argv),
    }
    (output_dir / "depth_shadow_reprice_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if repriced else 2


if __name__ == "__main__":
    raise SystemExit(main())
