"""Audit immutable live-PnL history without training or modifying artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence, cast


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.live.decision.pnl_forecast import FEATURE_COLUMNS
from neutralgrid.live.decision.pnl_history import (
    PnlHistoryError,
    load_all_pnl_observations,
)


FORECAST_CONTRACT_SCHEMA = "neutralgrid_pnl_forecast_contract_v1"


class ReadinessAuditError(RuntimeError):
    """Fail-closed PnL readiness audit error."""


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReadinessAuditError(f"non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except ReadinessAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReadinessAuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReadinessAuditError(f"{path}: JSON root must be an object")
    return payload


def _physical_observation_paths(live_root: Path) -> tuple[Path, ...]:
    if not live_root.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in live_root.rglob("*.json")
            if path.is_file()
            and path.parent.name == "observations"
            and path.parent.parent.parent.name == "pnl_history"
        )
    )


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _complete_fields(record: Mapping[str, Any], section: str, fields: Sequence[str]) -> bool:
    values = _mapping(record.get(section))
    return all(_finite(values.get(field)) for field in fields)


def _load_contract(path: Path) -> dict[str, Any]:
    payload = _strict_json_object(path)
    if payload.get("schema_version") != FORECAST_CONTRACT_SCHEMA:
        raise ReadinessAuditError("unsupported forecast contract schema")
    if payload.get("status") != "approved":
        raise ReadinessAuditError("forecast contract status must be approved")
    exact_values: dict[str, Any] = {
        "direction_label_definition": "delta_total_profit_usdt_gt_zero",
        "pnl_target_definition": "forward_delta_total_profit",
        "pnl_unit": "USDT",
        "missing_value_policy": "reject_feature_incomplete",
    }
    for key, expected in exact_values.items():
        if payload.get(key) != expected:
            raise ReadinessAuditError(f"unsupported forecast contract {key}")
    numeric_positive = (
        "forecast_horizon_minutes",
        "fit_fraction",
        "calibration_fraction",
        "prediction_interval_coverage",
        "min_fit_bots",
        "min_calibration_bots",
        "min_test_bots",
        "min_fit_samples",
        "min_calibration_samples",
        "min_test_samples",
    )
    for key in numeric_positive:
        value = payload.get(key)
        if not _finite(value):
            raise ReadinessAuditError(f"forecast contract {key} must be positive")
        numeric_value = cast(int | float, value)
        if float(numeric_value) <= 0:
            raise ReadinessAuditError(f"forecast contract {key} must be positive")
    tolerance = payload.get("label_tolerance_minutes")
    if not _finite(tolerance):
        raise ReadinessAuditError(
            "forecast contract label_tolerance_minutes must be non-negative"
        )
    numeric_tolerance = cast(int | float, tolerance)
    if float(numeric_tolerance) < 0:
        raise ReadinessAuditError(
            "forecast contract label_tolerance_minutes must be non-negative"
        )
    if float(payload["fit_fraction"]) + float(payload["calibration_fraction"]) >= 1:
        raise ReadinessAuditError(
            "fit_fraction + calibration_fraction must leave a final test split"
        )
    coverage = float(payload["prediction_interval_coverage"])
    if coverage >= 1:
        raise ReadinessAuditError("prediction_interval_coverage must be below 1")
    return payload


def _features_complete(record: Mapping[str, Any]) -> bool:
    pnl = _mapping(record.get("pnl"))
    position = _mapping(record.get("position"))
    features = _mapping(record.get("features"))
    direct: dict[str, Any] = {
        "current_total_profit_usdt": pnl.get("total_profit_usdt"),
        "matched_profit_usdt": pnl.get("matched_profit_usdt"),
        "unmatched_pnl_usdt": pnl.get("unmatched_pnl_usdt"),
        "position_pnl_usdt": position.get("position_pnl_usdt"),
        "position_notional_abs_usdt": position.get("size_usdt"),
    }
    always_derivable = {
        "pnl_velocity_usdt_per_min",
        "previous_delta_pnl_usdt",
        "previous_direction_positive",
        "public_trade_available",
        "private_event_available",
    }
    ratio_sources = {
        "aggressive_exit_trade_to_position_ratio": (
            "aggressive_exit_side_trade_notional_usdt"
        ),
        "trade_aligned_removal_to_position_ratio": (
            "trade_aligned_removal_proxy_usdt"
        ),
        "unexplained_removal_to_position_ratio": "unexplained_removal_proxy_usdt",
        "refill_to_position_ratio": "refill_proxy_usdt",
    }
    for column in FEATURE_COLUMNS:
        if column in always_derivable:
            continue
        ratio_source = ratio_sources.get(column)
        if ratio_source is not None:
            position_size = position.get("size_usdt")
            if (
                not _finite(position_size)
                or not _finite(features.get(ratio_source))
            ):
                return False
            numeric_position_size = cast(int | float, position_size)
            if abs(float(numeric_position_size)) <= 0.0:
                return False
            continue
        value = direct.get(column, features.get(column))
        if isinstance(value, bool):
            continue
        if not _finite(value):
            return False
    return True


def _classify_labels(
    observations: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, int]:
    horizon = timedelta(minutes=float(contract["forecast_horizon_minutes"]))
    tolerance = timedelta(minutes=float(contract["label_tolerance_minutes"]))
    by_bot: dict[str, list[Mapping[str, Any]]] = {}
    for record in observations:
        by_bot.setdefault(str(record["bot_identity"]), []).append(record)
    counts = {
        "labelable": 0,
        "direction_positive": 0,
        "direction_non_positive": 0,
        "not_yet_matured_or_censored": 0,
        "missing_same_bot_forward_observation": 0,
        "outside_label_tolerance": 0,
        "feature_incomplete": 0,
    }
    for records in by_bot.values():
        ordered = sorted(records, key=lambda item: cast(datetime, item["captured_at"]))
        for index, current in enumerate(ordered):
            if not _features_complete(current):
                counts["feature_incomplete"] += 1
            captured_at = cast(datetime, current["captured_at"])
            target = captured_at + horizon
            later = [
                candidate
                for candidate in ordered[index + 1 :]
                if cast(datetime, candidate["captured_at"]) > captured_at
            ]
            if not later:
                counts["not_yet_matured_or_censored"] += 1
                continue
            eligible = [
                candidate
                for candidate in later
                if target <= cast(datetime, candidate["captured_at"]) <= target + tolerance
            ]
            if not eligible:
                if cast(datetime, later[-1]["captured_at"]) < target:
                    counts["missing_same_bot_forward_observation"] += 1
                else:
                    counts["outside_label_tolerance"] += 1
                continue
            future = eligible[0]
            current_pnl = float(_mapping(current["pnl"])["total_profit_usdt"])
            future_pnl = float(_mapping(future["pnl"])["total_profit_usdt"])
            counts["labelable"] += 1
            if future_pnl - current_pnl > 0:
                counts["direction_positive"] += 1
            else:
                counts["direction_non_positive"] += 1
    return counts


def audit_training_readiness(
    *,
    live_root: Path,
    forecast_contract_path: Path | None,
) -> dict[str, Any]:
    physical_paths = _physical_observation_paths(live_root)
    try:
        observations = load_all_pnl_observations(live_root=live_root)
    except PnlHistoryError as exc:
        raise ReadinessAuditError(str(exc)) from exc
    symbols = sorted({str(record["symbol"]) for record in observations})
    bot_identities = sorted({str(record["bot_identity"]) for record in observations})
    counts: dict[str, Any] = {
        "physical_observation_files": len(physical_paths),
        "schema_valid_observations": len(observations),
        "invalid_observations": 0,
        "exact_duplicates": len(physical_paths) - len(observations),
        "conflicting_duplicates": 0,
        "globally_unique_observations": len(observations),
        "symbols": len(symbols),
        "bot_identities": len(bot_identities),
        "complete_pnl_fields": sum(
            _complete_fields(
                record,
                "pnl",
                ("total_profit_usdt", "matched_profit_usdt", "unmatched_pnl_usdt"),
            )
            for record in observations
        ),
        "complete_position_fields": sum(
            _complete_fields(
                record,
                "position",
                ("size_usdt", "entry_price", "mark_price", "position_pnl_usdt"),
            )
            for record in observations
        ),
        "with_l2_evidence": sum(
            _finite(_mapping(record.get("features")).get("spread_bps"))
            for record in observations
        ),
        "with_aggregate_trade_evidence": sum(
            str(_mapping(record.get("features")).get("public_trade_status", "")).startswith(
                "available"
            )
            for record in observations
        ),
        "with_private_event_evidence": sum(
            str(_mapping(record.get("features")).get("private_event_status", "")).startswith(
                "available"
            )
            for record in observations
        ),
        "feature_complete": sum(_features_complete(record) for record in observations),
        "feature_incomplete": sum(
            not _features_complete(record) for record in observations
        ),
        "labelable": None,
        "direction_positive": None,
        "direction_non_positive": None,
        "not_yet_matured_or_censored": None,
        "missing_same_bot_forward_observation": None,
        "outside_label_tolerance": None,
    }
    capture_times = [cast(datetime, record["captured_at"]) for record in observations]
    observations_per_bot: dict[str, int] = {}
    observations_per_symbol: dict[str, int] = {}
    for record in observations:
        bot = str(record["bot_identity"])
        symbol = str(record["symbol"])
        observations_per_bot[bot] = observations_per_bot.get(bot, 0) + 1
        observations_per_symbol[symbol] = observations_per_symbol.get(symbol, 0) + 1
    report: dict[str, Any] = {
        "schema_version": "neutralgrid_pnl_training_readiness_v1",
        "cycle_status": "complete",
        "training_readiness": "blocked_missing_explicit_forecast_contract",
        "runtime_effect": "none",
        "forecast_contract": None,
        "counts": counts,
        "capture_time_range": {
            "first_utc": min(capture_times).isoformat() if capture_times else None,
            "last_utc": max(capture_times).isoformat() if capture_times else None,
        },
        "observations_per_symbol": observations_per_symbol,
        "observations_per_bot_identity": observations_per_bot,
        "symbols": symbols,
        "bot_identities": bot_identities,
        "limitations": [
            "Physical rows, globally unique observations, labels, and independent bots are distinct quantities.",
            "This audit neither trains nor promotes a PnL forecast artifact.",
        ],
    }
    if forecast_contract_path is None:
        return report
    contract = _load_contract(forecast_contract_path)
    report["forecast_contract"] = {
        "path": str(forecast_contract_path.resolve()),
        "schema_version": contract["schema_version"],
        "status": contract["status"],
    }
    label_counts = _classify_labels(observations, contract=contract)
    counts.update(label_counts)
    required_bot_count = (
        int(contract["min_fit_bots"])
        + int(contract["min_calibration_bots"])
        + int(contract["min_test_bots"])
    )
    required_sample_count = (
        int(contract["min_fit_samples"])
        + int(contract["min_calibration_samples"])
        + int(contract["min_test_samples"])
    )
    if counts["conflicting_duplicates"]:
        readiness = "blocked_conflicting_duplicates"
    elif counts["feature_incomplete"]:
        readiness = "blocked_feature_incomplete"
    elif len(bot_identities) < required_bot_count:
        readiness = "blocked_insufficient_bot_disjoint_splits"
    elif int(counts["labelable"]) < required_sample_count:
        readiness = "blocked_insufficient_labelled_samples"
    else:
        readiness = "ready_for_separate_shadow_training_review"
    report["training_readiness"] = readiness
    report["bot_disjoint_chronological_split_feasible"] = (
        len(bot_identities) >= required_bot_count
    )
    report["label_purge_feasible"] = int(counts["labelable"]) >= required_sample_count
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--forecast-contract", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_training_readiness(
            live_root=args.live_root,
            forecast_contract_path=args.forecast_contract,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ReadinessAuditError(f"refusing to overwrite {args.output}")
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ReadinessAuditError, OSError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
