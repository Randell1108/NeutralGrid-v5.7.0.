"""Explicit-horizon, bot-disjoint shadow forecasts for live grid-bot PnL.

This module never changes a live verdict.  It turns immutable private-PnL
observations into forward dollar/direction labels, fits simple regularized
models, calibrates a dollar interval on a middle cohort, and evaluates exactly
once on a later bot-disjoint cohort.  An artifact is inference-eligible only
when it does not degrade against simple baselines on that final cohort.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neutralgrid.live.decision.pnl_history import (
    PnlHistoryError,
    validate_pnl_observation,
)


PNL_FORECAST_ARTIFACT_SCHEMA_VERSION = "neutralgrid_pnl_forecaster_artifact_v1"
PNL_FORECAST_REPORT_SCHEMA_VERSION = "neutralgrid_pnl_forecast_oos_report_v1"
PNL_FORECAST_OUTPUT_SCHEMA_VERSION = "neutralgrid_shadow_pnl_forecast_v1"


class PnlForecastError(RuntimeError):
    """Forecast data, artifact, or inference contract is invalid."""


@dataclass(frozen=True)
class PnlForecastConfig:
    """All horizon, split, and evidence thresholds are serialized in artifacts."""

    horizon_minutes: float
    label_tolerance_minutes: float
    fit_fraction: float = 0.60
    calibration_fraction: float = 0.20
    prediction_interval_coverage: float = 0.80
    min_fit_bots: int = 4
    min_calibration_bots: int = 2
    min_test_bots: int = 2
    min_fit_samples: int = 30
    min_calibration_samples: int = 10
    min_test_samples: int = 10


FEATURE_COLUMNS: tuple[str, ...] = (
    "current_total_profit_usdt",
    "pnl_velocity_usdt_per_min",
    "previous_delta_pnl_usdt",
    "previous_direction_positive",
    "matched_profit_usdt",
    "unmatched_pnl_usdt",
    "position_pnl_usdt",
    "position_notional_abs_usdt",
    "gain_giveback_usdt",
    "gain_giveback_pct",
    "range_prob",
    "trend_prob",
    "persistence_prob",
    "expected_exit_impact_bps",
    "exit_depth_to_position_ratio",
    "spread_bps",
    "spread_current_to_median",
    "book_imbalance",
    "exit_side_removal_to_addition_ratio",
    "exit_depth_current_to_baseline",
    "exit_side_imbalance",
    "recent_spread_worse_fraction",
    "recent_exit_depth_worse_fraction",
    "joint_deterioration_trailing_duration_seconds",
    "aggressive_exit_trade_to_position_ratio",
    "trade_aligned_removal_to_position_ratio",
    "unexplained_removal_to_position_ratio",
    "refill_to_position_ratio",
    "exit_side_net_withdrawal_to_position_ratio",
    "mean_estimated_slippage_bps",
    "p90_estimated_slippage_bps",
    "mean_adverse_selection_5s_bps",
    "mean_adverse_selection_30s_bps",
    "sustained_joint_deterioration",
    "sustained_spread_deterioration",
    "sustained_exit_depth_deterioration",
    "temporary_joint_deterioration",
    "private_cancel_update_fraction",
    "public_trade_available",
    "private_event_available",
)


def _validate_config(config: PnlForecastConfig) -> None:
    if not math.isfinite(config.horizon_minutes) or config.horizon_minutes <= 0:
        raise PnlForecastError("horizon_minutes must be positive and finite")
    if (
        not math.isfinite(config.label_tolerance_minutes)
        or config.label_tolerance_minutes < 0
    ):
        raise PnlForecastError("label_tolerance_minutes must be non-negative and finite")
    if not 0 < config.fit_fraction < 1:
        raise PnlForecastError("fit_fraction must be between zero and one")
    if not 0 < config.calibration_fraction < 1:
        raise PnlForecastError("calibration_fraction must be between zero and one")
    if config.fit_fraction + config.calibration_fraction >= 1:
        raise PnlForecastError("fit plus calibration fractions must leave a test cohort")
    if not 0 < config.prediction_interval_coverage < 1:
        raise PnlForecastError("prediction_interval_coverage must be between zero and one")
    for name in (
        "min_fit_bots",
        "min_calibration_bots",
        "min_test_bots",
        "min_fit_samples",
        "min_calibration_samples",
        "min_test_samples",
    ):
        if int(getattr(config, name)) < 1:
            raise PnlForecastError(f"{name} must be positive")


def _clean_number(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def _bool_number(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return math.nan


def _available_status(value: Any) -> float:
    return 1.0 if isinstance(value, str) and value.startswith("available") else 0.0


def _validated_unique_observations(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for raw in observations:
        try:
            record = validate_pnl_observation(raw)
        except PnlHistoryError as exc:
            raise PnlForecastError(str(exc)) from exc
        observation_id = str(record["observation_id"])
        prior = unique.get(observation_id)
        if prior is None:
            unique[observation_id] = record
        elif prior["snapshot_fingerprint"] == record["snapshot_fingerprint"]:
            duplicates += 1
        else:
            raise PnlForecastError(
                f"conflicting duplicate observation_id {observation_id}"
            )
    ordered = sorted(
        unique.values(),
        key=lambda record: (
            cast(datetime, record["captured_at"]),
            str(record["bot_identity"]),
        ),
    )
    return ordered, duplicates


def _feature_rows(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    validated, duplicate_count = _validated_unique_observations(observations)
    rows: list[dict[str, Any]] = []
    previous_by_bot: dict[str, tuple[datetime, float]] = {}
    for record in validated:
        bot_identity = str(record["bot_identity"])
        captured_at = cast(datetime, record["captured_at"])
        pnl = cast(dict[str, Any], record["pnl"])
        position = cast(dict[str, Any], record["position"])
        features = cast(dict[str, Any], record["features"])
        current_profit = _clean_number(pnl.get("total_profit_usdt"))
        if not math.isfinite(current_profit):
            continue
        previous = previous_by_bot.get(bot_identity)
        previous_delta = 0.0
        velocity = 0.0
        prior_direction = 0.0
        if previous is not None:
            elapsed_min = (captured_at - previous[0]).total_seconds() / 60.0
            if elapsed_min <= 0:
                raise PnlForecastError(
                    f"non-increasing capture time for bot {bot_identity}"
                )
            previous_delta = current_profit - previous[1]
            velocity = previous_delta / elapsed_min
            prior_direction = float(previous_delta > 0)
        previous_by_bot[bot_identity] = (captured_at, current_profit)

        position_notional = abs(_clean_number(position.get("size_usdt")))
        if not math.isfinite(position_notional) or position_notional <= 0:
            position_notional = math.nan

        def _position_ratio(key: str) -> float:
            notional = _clean_number(features.get(key))
            if not math.isfinite(notional) or not math.isfinite(position_notional):
                return math.nan
            return notional / position_notional

        row: dict[str, Any] = {
            "observation_id": record["observation_id"],
            "record_integrity_sha256": record["record_integrity_sha256"],
            "bot_identity": bot_identity,
            "label_bot_identity": bot_identity,
            "symbol": record["symbol"],
            "strategy_id": record["strategy_id"],
            "deploy_ts_utc": cast(datetime, record["deploy_ts"]),
            "captured_at_utc": captured_at,
            "current_total_profit_usdt": current_profit,
            "pnl_velocity_usdt_per_min": velocity,
            "previous_delta_pnl_usdt": previous_delta,
            "previous_direction_positive": prior_direction,
            "matched_profit_usdt": _clean_number(pnl.get("matched_profit_usdt")),
            "unmatched_pnl_usdt": _clean_number(pnl.get("unmatched_pnl_usdt")),
            "position_pnl_usdt": _clean_number(position.get("position_pnl_usdt")),
            "position_notional_abs_usdt": position_notional,
            "gain_giveback_usdt": _clean_number(features.get("gain_giveback_usdt")),
            "gain_giveback_pct": _clean_number(features.get("gain_giveback_pct")),
            "range_prob": _clean_number(features.get("range_prob")),
            "trend_prob": _clean_number(features.get("trend_prob")),
            "persistence_prob": _clean_number(features.get("persistence_prob")),
            "expected_exit_impact_bps": _clean_number(features.get("expected_exit_impact_bps")),
            "exit_depth_to_position_ratio": _clean_number(features.get("exit_depth_to_position_ratio")),
            "spread_bps": _clean_number(features.get("spread_bps")),
            "spread_current_to_median": _clean_number(features.get("spread_current_to_median")),
            "book_imbalance": _clean_number(features.get("book_imbalance")),
            "exit_side_removal_to_addition_ratio": _clean_number(features.get("exit_side_removal_to_addition_ratio")),
            "exit_depth_current_to_baseline": _clean_number(features.get("exit_depth_current_to_baseline")),
            "exit_side_imbalance": _clean_number(features.get("exit_side_imbalance")),
            "recent_spread_worse_fraction": _clean_number(features.get("recent_spread_worse_fraction")),
            "recent_exit_depth_worse_fraction": _clean_number(features.get("recent_exit_depth_worse_fraction")),
            "joint_deterioration_trailing_duration_seconds": _clean_number(features.get("joint_deterioration_trailing_duration_seconds")),
            "aggressive_exit_trade_to_position_ratio": _position_ratio("aggressive_exit_side_trade_notional_usdt"),
            "trade_aligned_removal_to_position_ratio": _position_ratio("trade_aligned_removal_proxy_usdt"),
            "unexplained_removal_to_position_ratio": _position_ratio("unexplained_removal_proxy_usdt"),
            "refill_to_position_ratio": _position_ratio("refill_proxy_usdt"),
            "exit_side_net_withdrawal_to_position_ratio": _clean_number(features.get("exit_side_net_withdrawal_to_position_ratio")),
            "mean_estimated_slippage_bps": _clean_number(features.get("mean_estimated_slippage_bps")),
            "p90_estimated_slippage_bps": _clean_number(features.get("p90_estimated_slippage_bps")),
            "mean_adverse_selection_5s_bps": _clean_number(features.get("mean_adverse_selection_5s_bps")),
            "mean_adverse_selection_30s_bps": _clean_number(features.get("mean_adverse_selection_30s_bps")),
            "sustained_joint_deterioration": _bool_number(features.get("sustained_joint_deterioration")),
            "sustained_spread_deterioration": _bool_number(features.get("sustained_spread_deterioration")),
            "sustained_exit_depth_deterioration": _bool_number(features.get("sustained_exit_depth_deterioration")),
            "temporary_joint_deterioration": _bool_number(features.get("temporary_joint_deterioration")),
            "private_cancel_update_fraction": _clean_number(features.get("private_cancel_update_fraction")),
            "public_trade_available": _available_status(features.get("public_trade_status")),
            "private_event_available": _available_status(features.get("private_event_status")),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = cast(
            pd.DataFrame,
            frame.sort_values(["captured_at_utc", "bot_identity"]).reset_index(drop=True),
        )
    return frame, {
        "input_observations": len(observations),
        "unique_observations": len(validated),
        "duplicate_observations_dropped": duplicate_count,
        "feature_rows": len(frame),
    }


def build_forecast_examples(
    observations: Sequence[Mapping[str, Any]],
    *,
    config: PnlForecastConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create same-bot forward labels at one explicit horizon."""

    _validate_config(config)
    features, audit = _feature_rows(observations)
    if features.empty:
        audit["labelled_examples"] = 0
        return features, audit
    horizon = timedelta(minutes=config.horizon_minutes)
    tolerance = timedelta(minutes=config.label_tolerance_minutes)
    labelled: list[dict[str, Any]] = []
    for bot_identity, group in features.groupby("bot_identity", sort=False):
        ordered = cast(pd.DataFrame, group.sort_values("captured_at_utc")).reset_index(drop=True)
        for _index, current in ordered.iterrows():
            current_at = cast(datetime, current["captured_at_utc"])
            target = current_at + horizon
            future_rows = cast(
                pd.DataFrame,
                ordered.loc[
                    (cast(pd.Series, ordered["captured_at_utc"]) >= target)
                    & (cast(pd.Series, ordered["captured_at_utc"]) <= target + tolerance)
                ],
            )
            if future_rows.empty:
                continue
            future = future_rows.iloc[0]
            label_at = cast(datetime, future["captured_at_utc"])
            delta = float(future["current_total_profit_usdt"]) - float(
                current["current_total_profit_usdt"]
            )
            row = current.to_dict()
            row.update(
                {
                    "label_bot_identity": str(bot_identity),
                    "label_observation_id": str(future["observation_id"]),
                    "label_captured_at_utc": label_at,
                    "actual_horizon_minutes": (label_at - current_at).total_seconds() / 60.0,
                    "target_delta_pnl_usdt": delta,
                    "target_direction_positive": int(delta > 0),
                    "persistence_baseline_delta_usdt": float(
                        current["pnl_velocity_usdt_per_min"]
                    )
                    * config.horizon_minutes,
                }
            )
            labelled.append(row)
    output = pd.DataFrame(labelled)
    if not output.empty:
        output = cast(
            pd.DataFrame,
            output.sort_values(["captured_at_utc", "bot_identity"]).reset_index(drop=True),
        )
    audit["labelled_examples"] = len(output)
    return output, audit


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_usdt": float(mean_absolute_error(actual, predicted)),
        "rmse_usdt": float(math.sqrt(mean_squared_error(actual, predicted))),
        "sign_accuracy": float(accuracy_score(actual > 0, predicted > 0)),
    }


def _classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(probability.astype(float), 1e-9, 1.0 - 1e-9)
    predicted = clipped >= 0.5
    log_loss = -np.mean(
        actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped)
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "brier": float(brier_score_loss(actual, clipped)),
        "log_loss": float(log_loss),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(examples: pd.DataFrame) -> str:
    payload: list[dict[str, Any]] = []
    for _index, row in examples.iterrows():
        features: dict[str, float | None] = {}
        for column in FEATURE_COLUMNS:
            value = row[column]
            features[column] = None if bool(pd.isna(value)) else float(value)
        payload.append(
            {
                "observation_id": str(row["observation_id"]),
                "record_integrity_sha256": str(row["record_integrity_sha256"]),
                "label_observation_id": str(row["label_observation_id"]),
                "target_delta_pnl_usdt": float(row["target_delta_pnl_usdt"]),
                "features": features,
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _base_report(config: PnlForecastConfig, audit: Mapping[str, int]) -> dict[str, Any]:
    return {
        "schema_version": PNL_FORECAST_REPORT_SCHEMA_VERSION,
        "status": "insufficient_unclassified",
        "forecast_eligible": False,
        "runtime_effect": "none",
        "config": asdict(config),
        "feature_schema": list(FEATURE_COLUMNS),
        "counts": dict(audit),
        "scope_notes": [
            "The horizon is explicit and serialized; no terminal-bot PnL is used as a live forward label.",
            "Fit, interval calibration, and final test identities are bot-disjoint and chronological.",
            "The artifact is shadow-only and never changes CONTINUE, ADJUST, or END.",
        ],
    }


def _write_insufficient(
    report: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    report["artifact_paths"] = {"metadata": str(metadata_path.resolve()), "model": None}
    _atomic_json(metadata_path, report)
    return report


def train_evaluate_shadow_forecaster(
    observations: Sequence[Mapping[str, Any]],
    *,
    config: PnlForecastConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit and strictly evaluate one candidate; never promote a live policy."""

    _validate_config(config)
    existing_artifacts = [
        path
        for path in (output_dir / "metadata.json", output_dir / "model.joblib")
        if path.exists()
    ]
    if existing_artifacts:
        raise PnlForecastError(
            "refusing to overwrite immutable forecast artifact files: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    examples, audit = build_forecast_examples(observations, config=config)
    report = _base_report(config, audit)
    if examples.empty:
        report["status"] = "insufficient_no_forward_labels"
        return _write_insufficient(report, output_dir=output_dir)

    bot_first = cast(
        pd.DataFrame,
        examples.groupby("bot_identity", as_index=False)["captured_at_utc"].min(),
    )
    bot_first = cast(
        pd.DataFrame,
        bot_first.sort_values(
            by=["captured_at_utc", "bot_identity"]
        ).reset_index(drop=True),
    )
    bot_keys = [str(value) for value in cast(pd.Series, bot_first["bot_identity"]).tolist()]
    fit_count = int(math.floor(len(bot_keys) * config.fit_fraction))
    calibration_count = int(math.floor(len(bot_keys) * config.calibration_fraction))
    test_count = len(bot_keys) - fit_count - calibration_count
    report["counts"].update(
        {
            "bots": len(bot_keys),
            "fit_bots": fit_count,
            "calibration_bots": calibration_count,
            "test_bots": test_count,
        }
    )
    if (
        fit_count < config.min_fit_bots
        or calibration_count < config.min_calibration_bots
        or test_count < config.min_test_bots
    ):
        report["status"] = "insufficient_bot_count"
        return _write_insufficient(report, output_dir=output_dir)

    fit_keys = set(bot_keys[:fit_count])
    calibration_keys = set(bot_keys[fit_count : fit_count + calibration_count])
    test_keys = set(bot_keys[fit_count + calibration_count :])
    identity_series = cast(pd.Series, examples["bot_identity"])
    calibration_start = min(
        cast(
            pd.Series,
            examples.loc[
                identity_series.isin(sorted(calibration_keys)),
                "captured_at_utc",
            ],
        )
    )
    test_start = min(
        cast(
            pd.Series,
            examples.loc[
                identity_series.isin(sorted(test_keys)),
                "captured_at_utc",
            ],
        )
    )
    fit = cast(
        pd.DataFrame,
        examples.loc[
            identity_series.isin(sorted(fit_keys))
            & (cast(pd.Series, examples["label_captured_at_utc"]) < calibration_start)
        ].copy(),
    )
    calibration = cast(
        pd.DataFrame,
        examples.loc[
            identity_series.isin(sorted(calibration_keys))
            & (cast(pd.Series, examples["label_captured_at_utc"]) < test_start)
        ].copy(),
    )
    test = cast(
        pd.DataFrame,
        examples.loc[identity_series.isin(sorted(test_keys))].copy(),
    )
    overlap_fit_cal = fit_keys & calibration_keys
    overlap_fit_test = fit_keys & test_keys
    overlap_cal_test = calibration_keys & test_keys
    strict_temporal = bool(
        not fit.empty
        and not calibration.empty
        and not test.empty
        and max(cast(pd.Series, fit["label_captured_at_utc"])) < min(cast(pd.Series, calibration["captured_at_utc"]))
        and max(cast(pd.Series, calibration["label_captured_at_utc"])) < min(cast(pd.Series, test["captured_at_utc"]))
    )
    report["split_audit"] = {
        "strategy": "chronological_bot_identity_fit_calibration_test_with_label_purge",
        "fit_bot_identities": sorted(fit_keys),
        "calibration_bot_identities": sorted(calibration_keys),
        "test_bot_identities": sorted(test_keys),
        "fit_calibration_overlap_count": len(overlap_fit_cal),
        "fit_test_overlap_count": len(overlap_fit_test),
        "calibration_test_overlap_count": len(overlap_cal_test),
        "strict_temporal_order": strict_temporal,
        "calibration_start_utc": cast(datetime, calibration_start).isoformat(),
        "test_start_utc": cast(datetime, test_start).isoformat(),
    }
    report["counts"].update(
        {
            "fit_samples_after_purge": len(fit),
            "calibration_samples_after_purge": len(calibration),
            "test_samples": len(test),
        }
    )
    if (
        len(fit) < config.min_fit_samples
        or len(calibration) < config.min_calibration_samples
        or len(test) < config.min_test_samples
        or not strict_temporal
        or overlap_fit_cal
        or overlap_fit_test
        or overlap_cal_test
    ):
        report["status"] = "insufficient_samples_after_temporal_purge"
        return _write_insufficient(report, output_dir=output_dir)

    y_fit_dollar = np.asarray(fit["target_delta_pnl_usdt"], dtype=float)
    y_fit_direction = np.asarray(fit["target_direction_positive"], dtype=int)
    if len(np.unique(y_fit_direction)) < 2:
        report["status"] = "insufficient_fit_class_diversity"
        return _write_insufficient(report, output_dir=output_dir)

    feature_columns = list(FEATURE_COLUMNS)
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )
    regressor.fit(fit[feature_columns], y_fit_dollar)
    classifier.fit(fit[feature_columns], y_fit_direction)

    calibration_actual = np.asarray(calibration["target_delta_pnl_usdt"], dtype=float)
    calibration_predicted = np.asarray(regressor.predict(calibration[feature_columns]), dtype=float)
    residuals = np.abs(calibration_actual - calibration_predicted)
    interval_radius = float(
        np.quantile(residuals, config.prediction_interval_coverage, method="higher")
    )
    calibration_direction = np.asarray(
        calibration["target_direction_positive"],
        dtype=int,
    )
    if len(np.unique(calibration_direction)) < 2:
        report["status"] = "insufficient_calibration_class_diversity"
        return _write_insufficient(report, output_dir=output_dir)
    calibration_logits = np.asarray(
        classifier.decision_function(calibration[feature_columns]),
        dtype=float,
    ).reshape(-1, 1)
    probability_calibrator = LogisticRegression(max_iter=2000, C=1.0)
    probability_calibrator.fit(calibration_logits, calibration_direction)

    test_actual_dollar = np.asarray(test["target_delta_pnl_usdt"], dtype=float)
    test_actual_direction = np.asarray(test["target_direction_positive"], dtype=int)
    model_dollar = np.asarray(regressor.predict(test[feature_columns]), dtype=float)
    test_logits = np.asarray(
        classifier.decision_function(test[feature_columns]),
        dtype=float,
    ).reshape(-1, 1)
    model_probability = np.asarray(
        probability_calibrator.predict_proba(test_logits)[:, 1],
        dtype=float,
    )
    zero_dollar = np.zeros(len(test), dtype=float)
    persistence_dollar = np.asarray(test["persistence_baseline_delta_usdt"], dtype=float)
    prior_probability_value = float(np.mean(y_fit_direction))
    prior_probability = np.full(len(test), prior_probability_value, dtype=float)
    persistence_probability = (persistence_dollar > 0).astype(float)

    dollar_metrics = {
        "model": _regression_metrics(test_actual_dollar, model_dollar),
        "zero_change_baseline": _regression_metrics(test_actual_dollar, zero_dollar),
        "last_slope_baseline": _regression_metrics(test_actual_dollar, persistence_dollar),
    }
    direction_metrics = {
        "model": _classification_metrics(test_actual_direction, model_probability),
        "training_prior_baseline": _classification_metrics(test_actual_direction, prior_probability),
        "last_direction_baseline": _classification_metrics(test_actual_direction, persistence_probability),
    }
    interval_coverage = float(
        np.mean(
            (test_actual_dollar >= model_dollar - interval_radius)
            & (test_actual_dollar <= model_dollar + interval_radius)
        )
    )
    report["oos_metrics"] = {
        "dollar": dollar_metrics,
        "direction": direction_metrics,
        "prediction_interval": {
            "target_coverage": config.prediction_interval_coverage,
            "observed_coverage": interval_coverage,
            "absolute_residual_radius_usdt": interval_radius,
        },
    }
    best_baseline_mae = min(
        dollar_metrics["zero_change_baseline"]["mae_usdt"],
        dollar_metrics["last_slope_baseline"]["mae_usdt"],
    )
    best_baseline_accuracy = max(
        direction_metrics["training_prior_baseline"]["accuracy"],
        direction_metrics["last_direction_baseline"]["accuracy"],
    )
    best_baseline_brier = min(
        direction_metrics["training_prior_baseline"]["brier"],
        direction_metrics["last_direction_baseline"]["brier"],
    )
    gates = {
        "identity_disjoint": not (overlap_fit_cal or overlap_fit_test or overlap_cal_test),
        "strict_temporal_order": strict_temporal,
        "dollar_non_degradation": dollar_metrics["model"]["mae_usdt"] <= best_baseline_mae,
        "direction_non_degradation": (
            direction_metrics["model"]["accuracy"] >= best_baseline_accuracy
            and direction_metrics["model"]["brier"] <= best_baseline_brier
        ),
        "interval_coverage": interval_coverage + 0.05 >= config.prediction_interval_coverage,
    }
    report["gates"] = gates
    eligible = all(gates.values())
    report["status"] = "shadow_oos_validated" if eligible else "shadow_oos_rejected"
    report["forecast_eligible"] = eligible

    dataset_fingerprint = _dataset_fingerprint(examples)
    artifact_id = hashlib.sha256(
        json.dumps(
            {"config": asdict(config), "dataset_fingerprint": dataset_fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    model_payload = {
        "schema_version": PNL_FORECAST_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "feature_schema": feature_columns,
        "regressor": regressor,
        "classifier": classifier,
        "probability_calibrator": probability_calibrator,
        "probability_calibration_method": "platt_on_bot_disjoint_middle_cohort",
        "dataset_fingerprint": dataset_fingerprint,
        "interval_radius_usdt": interval_radius,
        "training_prior_probability": prior_probability_value,
        "horizon_minutes": config.horizon_minutes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    temp_model = output_dir / f".{model_path.name}.{os.getpid()}.tmp"
    try:
        joblib.dump(model_payload, temp_model, compress=3)
        os.replace(temp_model, model_path)
    finally:
        temp_model.unlink(missing_ok=True)
    report["artifact_id"] = artifact_id
    report["dataset_fingerprint"] = dataset_fingerprint
    report["model_sha256"] = _sha256_file(model_path)
    metadata_path = output_dir / "metadata.json"
    report["artifact_paths"] = {
        "model": str(model_path.resolve()),
        "metadata": str(metadata_path.resolve()),
    }
    _atomic_json(metadata_path, report)
    return report


def predict_shadow_pnl(
    artifact_dir: Path,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one latest-bot shadow forecast from an OOS-validated artifact."""

    metadata_path = artifact_dir / "metadata.json"
    model_path = artifact_dir / "model.joblib"
    try:
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PnlForecastError(f"cannot read forecast metadata: {exc}") from exc
    if not isinstance(metadata_raw, dict):
        raise PnlForecastError("forecast metadata is not an object")
    metadata = cast(dict[str, Any], metadata_raw)
    if metadata.get("schema_version") != PNL_FORECAST_REPORT_SCHEMA_VERSION:
        raise PnlForecastError("unsupported forecast metadata schema")
    if metadata.get("forecast_eligible") is not True:
        raise PnlForecastError(
            f"forecast artifact is not OOS eligible: {metadata.get('status')!r}"
        )
    if not model_path.is_file() or _sha256_file(model_path) != metadata.get("model_sha256"):
        raise PnlForecastError("forecast model is missing or hash-mismatched")
    payload_raw = joblib.load(model_path)
    if not isinstance(payload_raw, dict):
        raise PnlForecastError("forecast model payload is not an object")
    payload = cast(dict[str, Any], payload_raw)
    if payload.get("schema_version") != PNL_FORECAST_ARTIFACT_SCHEMA_VERSION:
        raise PnlForecastError("unsupported forecast model schema")
    if payload.get("artifact_id") != metadata.get("artifact_id"):
        raise PnlForecastError("forecast artifact identity mismatch")
    if payload.get("dataset_fingerprint") != metadata.get("dataset_fingerprint"):
        raise PnlForecastError("forecast dataset fingerprint mismatch")
    if payload.get("feature_schema") != list(FEATURE_COLUMNS):
        raise PnlForecastError("forecast feature schema mismatch")

    frame, audit = _feature_rows(observations)
    if frame.empty:
        raise PnlForecastError("no valid observations are available for inference")
    identities = set(cast(pd.Series, frame["bot_identity"]).astype(str))
    if len(identities) != 1:
        raise PnlForecastError("inference observations must belong to exactly one bot")
    latest = cast(pd.DataFrame, frame.sort_values("captured_at_utc")).iloc[[-1]]
    regressor = payload.get("regressor")
    classifier = payload.get("classifier")
    probability_calibrator = payload.get("probability_calibrator")
    if (
        not hasattr(regressor, "predict")
        or not hasattr(classifier, "decision_function")
        or not hasattr(probability_calibrator, "predict_proba")
    ):
        raise PnlForecastError("forecast estimators are invalid")
    regressor_model = cast(Any, regressor)
    classifier_model = cast(Any, classifier)
    calibrator_model = cast(Any, probability_calibrator)
    predicted = float(regressor_model.predict(latest[list(FEATURE_COLUMNS)])[0])
    latest_logit = np.asarray(
        classifier_model.decision_function(latest[list(FEATURE_COLUMNS)]),
        dtype=float,
    ).reshape(-1, 1)
    probability = float(
        calibrator_model.predict_proba(latest_logit)[0, 1]
    )
    radius = float(payload["interval_radius_usdt"])
    horizon = float(payload["horizon_minutes"])
    persistence = float(latest["pnl_velocity_usdt_per_min"].iloc[0]) * horizon
    captured_at = cast(datetime, latest["captured_at_utc"].iloc[0])
    return {
        "schema_version": PNL_FORECAST_OUTPUT_SCHEMA_VERSION,
        "status": "available",
        "runtime_effect": "none",
        "artifact_id": payload["artifact_id"],
        "bot_identity": str(latest["bot_identity"].iloc[0]),
        "symbol": str(latest["symbol"].iloc[0]),
        "strategy_id": str(latest["strategy_id"].iloc[0]),
        "observation_id": str(latest["observation_id"].iloc[0]),
        "observed_at_utc": captured_at.astimezone(timezone.utc).isoformat(),
        "horizon_minutes": horizon,
        "forecast_target_at_utc": (
            captured_at + timedelta(minutes=horizon)
        ).astimezone(timezone.utc).isoformat(),
        "predicted_delta_pnl_usdt": predicted,
        "prediction_interval_lower_usdt": predicted - radius,
        "prediction_interval_upper_usdt": predicted + radius,
        "probability_positive_delta": probability,
        "zero_change_baseline_delta_usdt": 0.0,
        "last_slope_baseline_delta_usdt": persistence,
        "history_observation_count": int(audit["unique_observations"]),
        "scope_note": (
            "OOS-validated shadow forecast only; it does not authorize or alter a live action."
        ),
    }
