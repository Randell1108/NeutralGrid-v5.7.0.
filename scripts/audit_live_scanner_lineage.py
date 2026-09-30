"""Read-only identity and life-window audit of saved live scanner evidence.

The CSVs are provenance reports, not a training pool. Repeated captures of one
bot remain visible, while bot_links.csv has at most one row per finalized bot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from neutralgrid.live.decision.pnl_history import PnlHistoryError, validate_pnl_observation
from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor


def _identity(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing timestamp")
    timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return timestamp.astimezone(timezone.utc)


def _source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_finalized(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], set[str], list[dict[str, str]]]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name="General")
        excel_serial = True
    else:
        frame = pd.read_csv(path)
        excel_serial = False
    required = {"strategy_id", "symbol", "status", "start_time_utc", "end_time_utc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"finalized outcome source missing columns: {missing}")
    starts = LiveOutcomeIngestor._parse_utc_times(
        cast(pd.Series, frame["start_time_utc"]), excel_serial=excel_serial
    )
    ends = LiveOutcomeIngestor._parse_utc_times(
        cast(pd.Series, frame["end_time_utc"]), excel_serial=excel_serial
    )
    outcomes: dict[tuple[str, str], dict[str, Any]] = {}
    strategy_ids: set[str] = set()
    errors: list[dict[str, str]] = []
    for workbook_row, (index, row) in enumerate(frame.iterrows(), start=2):
        strategy_id = _identity(row["strategy_id"])
        symbol = _identity(row["symbol"]).upper()
        status = _identity(row["status"]).lower()
        if status not in {"cancelled", "canceled", "expired"}:
            continue
        if not strategy_id or not symbol:
            errors.append({"source": f"{path}:row-{workbook_row}", "reason": "missing_finalized_identity"})
            continue
        strategy_ids.add(strategy_id)
        key = (strategy_id, symbol)
        start, end = starts.loc[index], ends.loc[index]
        valid_window = pd.notna(start) and pd.notna(end) and end >= start
        if not valid_window:
            errors.append({"source": f"{path}:row-{workbook_row}", "reason": "invalid_finalized_life_window", "strategy_id": strategy_id, "symbol": symbol})
        if key in outcomes:
            errors.append({"source": f"{path}:row-{workbook_row}", "reason": "duplicate_finalized_identity", "strategy_id": strategy_id, "symbol": symbol})
            outcomes[key]["ambiguous"] = True
            continue
        outcomes[key] = {
            "strategy_id": strategy_id, "symbol": symbol, "status": status,
            "start": start, "end": end, "valid_window": valid_window,
            "ambiguous": False, "workbook_row": workbook_row,
            "pnl_pct": row.get("pnl_pct"),
        }
    return outcomes, strategy_ids, errors


def _classify(
    strategy_id: str, symbol: str, captured_at: datetime,
    outcomes: dict[tuple[str, str], dict[str, Any]], strategy_ids: set[str],
) -> tuple[str, dict[str, Any] | None]:
    outcome = outcomes.get((strategy_id, symbol))
    if outcome is None:
        return ("strategy_symbol_conflict" if strategy_id in strategy_ids else "no_finalized_strategy"), None
    if outcome["ambiguous"]:
        return "ambiguous_finalized_identity", outcome
    if not outcome["valid_window"]:
        return "invalid_finalized_life_window", outcome
    if not outcome["start"] <= captured_at <= outcome["end"]:
        return "outside_life_window", outcome
    return "matched_finalized", outcome


def _deploy_evidence(value: Any, outcome: dict[str, Any]) -> str:
    if value is None or value == "":
        return "missing"
    observed = pd.Timestamp(_utc(value))
    return "exact" if observed == outcome["start"] else "conflict"


def audit(
    *, root: Path, live_root: Path, expired_bots: Path, decision_paths: list[Path],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, Any],
]:
    outcomes, strategy_ids, errors = _load_finalized(expired_bots)
    snapshots: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    pnl_observations: list[dict[str, Any]] = []
    seen_snapshots: set[tuple[str, str, str, str]] = set()
    seen_decisions: set[str] = set()
    seen_pnl: dict[str, str] = {}
    matched_bots: dict[tuple[str, str], dict[str, Any]] = {}

    def record_matched_bot(outcome: dict[str, Any], captured_at: datetime, field: str) -> None:
        bot_key = (outcome["strategy_id"], outcome["symbol"])
        if bot_key not in matched_bots:
            matched_bots[bot_key] = {
                "strategy_id": outcome["strategy_id"], "symbol": outcome["symbol"],
                "finalized_status": outcome["status"],
                "workbook_row": outcome["workbook_row"],
                "start_time_utc": outcome["start"].isoformat(),
                "end_time_utc": outcome["end"].isoformat(),
                "pnl_pct": outcome["pnl_pct"],
                "snapshot_count": 0, "pnl_observation_count": 0, "decision_count": 0,
                "first_capture_utc": captured_at.isoformat(),
                "last_capture_utc": captured_at.isoformat(),
                "training_eligible": False,
            }
        bot = matched_bots[bot_key]
        bot[field] += 1
        bot["first_capture_utc"] = min(bot["first_capture_utc"], captured_at.isoformat())
        bot["last_capture_utc"] = max(bot["last_capture_utc"], captured_at.isoformat())

    snapshot_paths = sorted(live_root.glob("20??-??-??/*/private_telemetry_*.json"))
    for path in snapshot_paths:
        source = _source_path(path, root)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("JSON root is not an object")
            strategy_id = _identity(record.get("strategy_id"))
            symbol = _identity(record.get("symbol")).upper()
            captured_at = _utc(record.get("captured_at_utc"))
            raw_hash = _identity(record.get("raw_text_sha256"))
            if not strategy_id or not symbol or not raw_hash or symbol != path.parent.name.upper():
                raise ValueError("missing or path-conflicting snapshot identity")
            key = (strategy_id, symbol, captured_at.isoformat(), raw_hash)
            if key in seen_snapshots:
                classification, outcome = "duplicate_observation", None
            else:
                seen_snapshots.add(key)
                classification, outcome = _classify(strategy_id, symbol, captured_at, outcomes, strategy_ids)
            deploy_evidence = "not_checked"
            if classification == "matched_finalized" and outcome is not None:
                structured = record.get("structured_telemetry")
                deploy_value = structured.get("deploy_ts") if isinstance(structured, dict) else None
                deploy_evidence = _deploy_evidence(deploy_value, outcome)
                if deploy_evidence == "conflict":
                    classification = "deploy_time_conflict"
                    errors.append({"source": source, "reason": classification, "strategy_id": strategy_id, "symbol": symbol})
            row = {
                "source_path": source, "strategy_id": strategy_id, "symbol": symbol,
                "captured_at_utc": captured_at.isoformat(), "raw_text_sha256": raw_hash,
                "classification": classification, "deploy_time_evidence": deploy_evidence,
                "workbook_row": outcome["workbook_row"] if outcome else "",
                "training_eligible": False,
            }
            snapshots.append(row)
            if classification == "matched_finalized" and outcome is not None:
                record_matched_bot(outcome, captured_at, "snapshot_count")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"source": source, "reason": f"invalid_snapshot: {exc}"})
            snapshots.append({"source_path": source, "classification": "invalid_snapshot", "training_eligible": False})

    pnl_paths = sorted(live_root.glob("20??-??-??/*/pnl_history/*/observations/*.json"))
    for path in pnl_paths:
        source = _source_path(path, root)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("JSON root is not an object")
            record = validate_pnl_observation(raw)
            strategy_id = _identity(record.get("strategy_id"))
            symbol = _identity(record.get("symbol")).upper()
            captured_at = cast(datetime, record["captured_at"])
            observation_id = _identity(record.get("observation_id"))
            fingerprint = _identity(record.get("snapshot_fingerprint"))
            if not strategy_id or not symbol or symbol != path.parents[3].name.upper():
                raise ValueError("missing or path-conflicting PnL identity")
            prior = seen_pnl.get(observation_id)
            if prior is not None:
                if prior != fingerprint:
                    classification, outcome = "conflicting_duplicate_observation", None
                    errors.append({"source": source, "reason": classification, "observation_id": observation_id})
                else:
                    classification, outcome = "duplicate_observation", None
            else:
                seen_pnl[observation_id] = fingerprint
                classification, outcome = _classify(strategy_id, symbol, captured_at, outcomes, strategy_ids)
            deploy_evidence = "not_checked"
            if classification == "matched_finalized" and outcome is not None:
                deploy_evidence = _deploy_evidence(raw.get("deploy_ts_utc"), outcome)
                if deploy_evidence == "conflict":
                    classification = "deploy_time_conflict"
                    errors.append({"source": source, "reason": classification, "strategy_id": strategy_id, "symbol": symbol})
            pnl_observations.append({
                "source_path": source, "strategy_id": strategy_id, "symbol": symbol,
                "captured_at_utc": captured_at.isoformat(), "observation_id": observation_id,
                "classification": classification, "deploy_time_evidence": deploy_evidence,
                "workbook_row": outcome["workbook_row"] if outcome else "",
                "training_eligible": False,
            })
            if classification == "matched_finalized" and outcome is not None:
                record_matched_bot(outcome, captured_at, "pnl_observation_count")
        except (OSError, ValueError, TypeError, KeyError, PnlHistoryError) as exc:
            errors.append({"source": source, "reason": f"invalid_pnl_observation: {exc}"})
            pnl_observations.append({
                "source_path": source, "classification": "invalid_pnl_observation",
                "training_eligible": False,
            })

    for path in sorted(set(decision_paths)):
        source = _source_path(path, root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append({"source": source, "reason": f"decision_read_failed: {exc}"})
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            ref = f"{source}:{line_number}"
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON root is not an object")
                strategy_id = _identity(record.get("strategy_id"))
                symbol = _identity(record.get("symbol")).upper()
                ts = _utc(record.get("ts"))
                if not strategy_id or not symbol:
                    raise ValueError("missing decision identity")
                digest = hashlib.sha256(line.strip().encode("utf-8")).hexdigest()
                if digest in seen_decisions:
                    classification, outcome = "duplicate_observation", None
                else:
                    seen_decisions.add(digest)
                    classification, outcome = _classify(strategy_id, symbol, ts, outcomes, strategy_ids)
                decisions.append({
                    "source_path": ref, "strategy_id": strategy_id, "symbol": symbol,
                    "captured_at_utc": ts.isoformat(), "verdict": _identity(record.get("verdict")),
                    "classification": classification,
                    "workbook_row": outcome["workbook_row"] if outcome else "",
                    "training_eligible": False,
                })
                if classification == "matched_finalized" and outcome is not None:
                    record_matched_bot(outcome, ts, "decision_count")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                errors.append({"source": ref, "reason": f"invalid_decision: {exc}"})
                decisions.append({"source_path": ref, "classification": "invalid_decision", "training_eligible": False})

    bots = [matched_bots[key] for key in sorted(matched_bots)]
    summary = {
        "source_class": "read_only_shadow_audit", "training_eligible": False,
        "expired_bots_path": _source_path(expired_bots, root),
        "snapshot_files": len(snapshot_paths), "pnl_observation_files": len(pnl_paths),
        "decision_files": len(set(decision_paths)),
        "snapshot_classifications": dict(sorted(Counter(row["classification"] for row in snapshots).items())),
        "snapshot_deploy_evidence": dict(sorted(Counter(row.get("deploy_time_evidence", "not_checked") for row in snapshots).items())),
        "pnl_observation_classifications": dict(sorted(Counter(row["classification"] for row in pnl_observations).items())),
        "pnl_deploy_evidence": dict(sorted(Counter(row.get("deploy_time_evidence", "not_checked") for row in pnl_observations).items())),
        "decision_classifications": dict(sorted(Counter(row["classification"] for row in decisions).items())),
        "matched_finalized_bots": len(bots),
        "snapshot_matched_bots": sum(bot["snapshot_count"] > 0 for bot in bots),
        "pnl_matched_bots": sum(bot["pnl_observation_count"] > 0 for bot in bots),
        "decision_matched_bots": sum(bot["decision_count"] > 0 for bot in bots),
        "source_errors": errors,
    }
    return snapshots, pnl_observations, decisions, bots, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    decision_paths = list((root / "logs").glob("live_decisions_*.jsonl"))
    decision_paths += list((root / "outputs/audits/recurring_drawer_verdict/controller/scanner").glob("**/live_decisions_*.jsonl"))
    snapshots, pnl_observations, decisions, bots, summary = audit(
        root=root, live_root=root / "Live",
        expired_bots=root / "data/new_expired_bots.xlsx", decision_paths=decision_paths,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "snapshot_links.csv", snapshots, (
        "source_path", "strategy_id", "symbol", "captured_at_utc", "raw_text_sha256",
        "classification", "deploy_time_evidence", "workbook_row", "training_eligible",
    ))
    _write_csv(output_dir / "pnl_observation_links.csv", pnl_observations, (
        "source_path", "strategy_id", "symbol", "captured_at_utc", "observation_id",
        "classification", "deploy_time_evidence", "workbook_row", "training_eligible",
    ))
    _write_csv(output_dir / "decision_links.csv", decisions, (
        "source_path", "strategy_id", "symbol", "captured_at_utc", "verdict",
        "classification", "workbook_row", "training_eligible",
    ))
    _write_csv(output_dir / "bot_links.csv", bots, (
        "strategy_id", "symbol", "finalized_status", "workbook_row",
        "start_time_utc", "end_time_utc", "pnl_pct", "snapshot_count",
        "pnl_observation_count", "decision_count",
        "first_capture_utc", "last_capture_utc", "training_eligible",
    ))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
