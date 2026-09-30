"""Inventory saved collector manifests against exact finalized bot identities."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_live_scanner_lineage import _identity, _load_finalized, _source_path, _utc


PATTERNS = {
    "diff_depth": "20??-??-??/*/diff_depth/*/manifest.json",
    "market_stream": "20??-??-??/*/market_stream/*/manifest.json",
    "private_user_stream": "20??-??-??/*/private_user_stream/*/*/manifest.json",
}


def audit_manifests(
    *, root: Path, live_root: Path, expired_bots: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outcomes, finalized_ids, workbook_errors = _load_finalized(expired_bots)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for kind, pattern in PATTERNS.items():
        for path in sorted(live_root.glob(pattern)):
            source = _source_path(path, root)
            row: dict[str, Any] = {
                "source_path": source, "source_kind": kind,
                "training_eligible": False,
            }
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("manifest root is not an object")
                target = payload.get("target")
                if not isinstance(target, dict):
                    target = {}
                strategy_id = _identity(payload.get("strategy_id") or target.get("strategy_id"))
                symbol = _identity(payload.get("symbol") or target.get("symbol")).upper()
                expected_symbol = path.relative_to(live_root).parts[1].upper()
                if not symbol or symbol != expected_symbol:
                    raise ValueError("manifest symbol conflicts with Live path")
                row.update({
                    "strategy_id": strategy_id, "symbol": symbol,
                    "run_id": _identity(payload.get("run_id")),
                    "source_status": _identity(payload.get("status")),
                    "started_at_utc": _identity(payload.get("started_at_utc")),
                    "ended_at_utc": _identity(
                        payload.get("completed_at_utc") or payload.get("updated_at_utc")
                        or payload.get("last_event_at_utc")
                    ),
                    "run_file_count": sum(child.is_file() for child in path.parent.iterdir()),
                })
                outcome = outcomes.get((strategy_id, symbol)) if strategy_id else None
                if not strategy_id:
                    classification = "symbol_only"
                elif outcome is None:
                    classification = (
                        "strategy_symbol_conflict" if strategy_id in finalized_ids
                        else "no_finalized_strategy"
                    )
                elif outcome["ambiguous"]:
                    classification = "ambiguous_finalized_identity"
                elif not outcome["valid_window"]:
                    classification = "invalid_outcome_window"
                else:
                    started_at = _utc(payload.get("started_at_utc"))
                    ended_value = (
                        payload.get("completed_at_utc")
                        or payload.get("updated_at_utc")
                        or payload.get("last_event_at_utc")
                    )
                    if ended_value:
                        ended_at = _utc(ended_value)
                        if ended_at < started_at:
                            classification = "invalid_manifest_interval"
                        elif started_at <= outcome["end"] and ended_at >= outcome["start"]:
                            classification = "overlap_valid_window"
                        else:
                            classification = "outside_life_window"
                    elif outcome["start"] <= started_at <= outcome["end"]:
                        classification = "start_inside_valid_window"
                    else:
                        classification = "interval_end_missing"
                row["classification"] = classification
                row["workbook_row"] = outcome["workbook_row"] if outcome else ""
                if classification in {
                    "strategy_symbol_conflict", "ambiguous_finalized_identity",
                    "invalid_manifest_interval",
                }:
                    errors.append({"source": source, "reason": classification})
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                row["classification"] = "invalid_manifest"
                errors.append({"source": source, "reason": f"invalid_manifest: {exc}"})
            rows.append(row)

    files = [path for path in live_root.rglob("*") if path.is_file()]
    summary = {
        "source_class": "read_only_manifest_inventory", "training_eligible": False,
        "live_file_count": len(files),
        "live_suffix_counts": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "manifest_count": len(rows),
        "by_kind": {
            kind: {
                "manifests": sum(row["source_kind"] == kind for row in rows),
                "classifications": dict(sorted(Counter(
                    row["classification"] for row in rows if row["source_kind"] == kind
                ).items())),
                "source_statuses": dict(sorted(Counter(
                    row.get("source_status", "") for row in rows if row["source_kind"] == kind
                ).items())),
            }
            for kind in PATTERNS
        },
        "manifest_errors": errors,
        "workbook_errors": workbook_errors,
    }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    rows, summary = audit_manifests(
        root=root, live_root=root / "Live",
        expired_bots=root / "data/new_expired_bots.xlsx",
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite manifest audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: tuple[str, ...] = (
        "source_path", "source_kind", "run_id", "symbol", "strategy_id",
        "started_at_utc", "ended_at_utc", "source_status", "classification",
        "workbook_row", "run_file_count", "training_eligible",
    )
    with (output_dir / "manifest_links.csv").open("w", encoding="utf-8", newline="") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
