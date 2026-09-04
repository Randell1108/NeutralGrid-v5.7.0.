"""Shadow HAR-RV training, frozen holdout evaluation, and inference.

Artifacts created here are audit outputs only.  They are never resolved through
the active model manifest and cannot affect live verdicts or action routing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neutralgrid.live.decision.volatility import (
    HAR_WINDOWS,
    VOLATILITY_OUTPUT_SCHEMA,
    VolatilityContract,
    VolatilityError,
    apply_log_interval,
    calibrate_log_interval,
    count_non_overlapping_origins,
    hash_frame,
    holm_adjust,
    interval_metrics,
    latest_rv_snapshot,
    newey_west_mean_test,
    qlike_loss,
)


VOLATILITY_ARTIFACT_SCHEMA = "neutralgrid_shadow_volatility_artifact_v1"
VOLATILITY_REPORT_SCHEMA = "neutralgrid_shadow_volatility_oos_report_v1"
FEATURE_COLUMNS: tuple[str, ...] = tuple(f"rv_{window}" for window in HAR_WINDOWS)
UTC = timezone.utc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        joblib.dump(payload, temp_name)
        with open(temp_name, "r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temp_name, index=False)
        with open(temp_name, "r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VolatilityError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except VolatilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VolatilityError(f"{path}: JSON root must be an object")
    return payload


def _positive_floor(frame: pd.DataFrame) -> float:
    values: list[np.ndarray] = []
    for column in (*FEATURE_COLUMNS, "target_rv"):
        array = np.asarray(frame[column], dtype=float)
        values.append(array[np.isfinite(array) & (array > 0.0)])
    positives = np.concatenate([value for value in values if len(value)])
    if len(positives) == 0:
        raise VolatilityError("HAR fit data contains no positive RV")
    floor = float(np.min(positives) * 0.5)
    if not math.isfinite(floor) or floor <= 0.0:
        raise VolatilityError("HAR RV floor is non-positive or non-finite")
    return floor


def _har_frame(frame: pd.DataFrame, *, floor: float, pooled: bool) -> pd.DataFrame:
    transformed = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        values = np.asarray(frame[column], dtype=float)
        if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
            raise VolatilityError(f"HAR feature {column} is invalid")
        transformed[column] = np.log(np.maximum(values, floor))
    if pooled:
        transformed["symbol"] = cast(pd.Series, frame["symbol"]).astype(str)
    return transformed


def _fit_har(
    frame: pd.DataFrame,
    *,
    alpha: float,
    pooled: bool,
) -> dict[str, Any]:
    if frame.empty:
        raise VolatilityError("cannot fit HAR on an empty frame")
    floor = _positive_floor(frame)
    x = _har_frame(frame, floor=floor, pooled=pooled)
    target = np.log(np.maximum(np.asarray(frame["target_rv"], dtype=float), floor))
    if pooled:
        preprocessor = ColumnTransformer(
            transformers=[
                ("rv", StandardScaler(), list(FEATURE_COLUMNS)),
                (
                    "symbol",
                    OneHotEncoder(handle_unknown="error", sparse_output=False),
                    ["symbol"],
                ),
            ],
            remainder="drop",
        )
        pipeline: Pipeline = Pipeline(
            [("preprocessor", preprocessor), ("ridge", Ridge(alpha=alpha))]
        )
    else:
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )
    pipeline.fit(x, target)
    return {
        "pipeline": pipeline,
        "rv_floor": floor,
        "alpha": alpha,
        "scope": "pooled" if pooled else "per_symbol",
    }


def _predict_har(model: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    pipeline = model.get("pipeline")
    floor_value = model.get("rv_floor")
    pooled = model.get("scope") == "pooled"
    if not isinstance(pipeline, Pipeline):
        raise VolatilityError("HAR artifact pipeline is invalid")
    if not isinstance(floor_value, (int, float)) or not math.isfinite(float(floor_value)):
        raise VolatilityError("HAR artifact RV floor is invalid")
    floor = float(floor_value)
    x = _har_frame(frame, floor=floor, pooled=pooled)
    predicted_log = np.asarray(pipeline.predict(x), dtype=float)
    predicted = np.exp(np.clip(predicted_log, -700.0, 700.0))
    predicted = np.maximum(predicted, floor)
    if not bool(np.isfinite(predicted).all()) or bool((predicted <= 0.0).any()):
        raise VolatilityError("HAR prediction is non-positive or non-finite")
    return predicted


def _global_splits(
    examples: pd.DataFrame,
    contract: VolatilityContract,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if examples.empty:
        raise VolatilityError("volatility examples are empty")
    required = {
        "symbol",
        "origin_utc",
        "label_end_utc",
        "horizon_minutes",
        "target_rv",
        *FEATURE_COLUMNS,
    }
    missing = sorted(required - set(examples.columns))
    if missing:
        raise VolatilityError(f"volatility examples lack columns: {missing}")
    frame = examples.copy()
    frame["origin_utc"] = pd.to_datetime(frame["origin_utc"], utc=True, errors="coerce")
    frame["label_end_utc"] = pd.to_datetime(
        frame["label_end_utc"], utc=True, errors="coerce"
    )
    if bool(cast(pd.Series, frame["origin_utc"]).isna().any()) or bool(
        cast(pd.Series, frame["label_end_utc"]).isna().any()
    ):
        raise VolatilityError("volatility examples contain invalid timestamps")
    origins = pd.DatetimeIndex(sorted(cast(pd.Series, frame["origin_utc"]).unique()))
    if len(origins) < 10:
        raise VolatilityError("too few unique origins to split")
    fit_index = max(1, int(math.floor(len(origins) * contract.fit_fraction)))
    calibration_index = max(
        fit_index + 1,
        int(
            math.floor(
                len(origins) * (contract.fit_fraction + contract.calibration_fraction)
            )
        ),
    )
    if calibration_index >= len(origins):
        raise VolatilityError("split fractions leave no final-test origins")
    fit_cutoff = cast(pd.Timestamp, cast(Any, origins[fit_index - 1]))
    calibration_cutoff = cast(pd.Timestamp, cast(Any, origins[calibration_index - 1]))
    fit = frame.loc[
        (cast(pd.Series, frame["origin_utc"]) <= fit_cutoff)
        & (cast(pd.Series, frame["label_end_utc"]) <= fit_cutoff)
    ].copy()
    calibration = frame.loc[
        (cast(pd.Series, frame["origin_utc"]) > fit_cutoff)
        & (cast(pd.Series, frame["origin_utc"]) <= calibration_cutoff)
        & (cast(pd.Series, frame["label_end_utc"]) <= calibration_cutoff)
    ].copy()
    test = frame.loc[
        cast(pd.Series, frame["origin_utc"]) > calibration_cutoff
    ].copy()
    if fit.empty or calibration.empty or test.empty:
        raise VolatilityError("purged 60/20/20 split contains an empty cohort")
    audit = {
        "fit_cutoff_utc": fit_cutoff.isoformat(),
        "calibration_cutoff_utc": calibration_cutoff.isoformat(),
        "fit_rows": len(fit),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "fit_fraction": contract.fit_fraction,
        "calibration_fraction": contract.calibration_fraction,
        "test_fraction": contract.test_fraction,
        "boundary_purge": "label_end_must_not_cross_cohort_cutoff",
    }
    return fit, calibration, test, audit


def _expanding_folds(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    origins = pd.DatetimeIndex(sorted(cast(pd.Series, frame["origin_utc"]).unique()))
    if len(origins) < 30:
        raise VolatilityError("development cohort has too few origins for expanding folds")
    fractions = ((0.50, 2.0 / 3.0), (2.0 / 3.0, 5.0 / 6.0), (5.0 / 6.0, 1.0))
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for train_fraction, validation_fraction in fractions:
        train_index = max(1, int(math.floor(len(origins) * train_fraction))) - 1
        validation_index = min(
            len(origins) - 1,
            max(train_index + 1, int(math.floor(len(origins) * validation_fraction)) - 1),
        )
        folds.append(
            (
                cast(pd.Timestamp, cast(Any, origins[train_index])),
                cast(pd.Timestamp, cast(Any, origins[validation_index])),
            )
        )
    return folds


def _baseline_columns(contract: VolatilityContract) -> tuple[str, ...]:
    return (
        "persistence_forecast_rv",
        "sqrt_time_forecast_rv",
        *(f"ewma_{value:g}_forecast_rv" for value in contract.ewma_lambdas),
    )


def _select_baseline(frame: pd.DataFrame, contract: VolatilityContract) -> dict[str, Any]:
    actual = np.asarray(frame["target_rv"], dtype=float)
    scores: dict[str, float] = {}
    for column in _baseline_columns(contract):
        forecast = np.asarray(frame[column], dtype=float)
        scores[column] = float(np.mean(qlike_loss(actual, forecast)))
    winner = min(scores, key=lambda column: (scores[column], column))
    return {"column": winner, "development_qlike": scores[winner], "scores": scores}


def _development_score(
    development: pd.DataFrame,
    *,
    symbol: str,
    alpha: float,
    pooled: bool,
) -> float:
    symbol_frame = development.loc[
        cast(pd.Series, development["symbol"]).astype(str) == symbol
    ].copy()
    fold_losses: list[float] = []
    for train_cutoff, validation_cutoff in _expanding_folds(symbol_frame):
        if pooled:
            train = development.loc[
                (cast(pd.Series, development["origin_utc"]) <= train_cutoff)
                & (cast(pd.Series, development["label_end_utc"]) <= train_cutoff)
            ].copy()
        else:
            train = symbol_frame.loc[
                (cast(pd.Series, symbol_frame["origin_utc"]) <= train_cutoff)
                & (cast(pd.Series, symbol_frame["label_end_utc"]) <= train_cutoff)
            ].copy()
        validation = symbol_frame.loc[
            (cast(pd.Series, symbol_frame["origin_utc"]) > train_cutoff)
            & (cast(pd.Series, symbol_frame["origin_utc"]) <= validation_cutoff)
        ].copy()
        if len(train) < 20 or len(validation) < 5:
            raise VolatilityError("expanding fold lacks minimum fit/validation rows")
        model = _fit_har(train, alpha=alpha, pooled=pooled)
        prediction = _predict_har(model, validation)
        fold_losses.append(
            float(np.mean(qlike_loss(np.asarray(validation["target_rv"], dtype=float), prediction)))
        )
    return float(np.mean(fold_losses))


def _select_challenger(
    development: pd.DataFrame,
    *,
    symbol: str,
    contract: VolatilityContract,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for scope in ("per_symbol", "pooled"):
        for alpha in contract.ridge_alphas:
            score = _development_score(
                development,
                symbol=symbol,
                alpha=alpha,
                pooled=scope == "pooled",
            )
            candidates.append({"scope": scope, "alpha": alpha, "development_qlike": score})
    winner = min(
        candidates,
        key=lambda item: (
            float(item["development_qlike"]),
            str(item["scope"]),
            float(item["alpha"]),
        ),
    )
    return {**winner, "candidates": candidates}


def _origin_counts(
    cohort: pd.DataFrame,
    *,
    symbol: str,
    horizon: int,
) -> int:
    selected = cohort.loc[
        (cast(pd.Series, cohort["symbol"]).astype(str) == symbol)
        & (pd.to_numeric(cohort["horizon_minutes"], errors="coerce") == horizon)
    ]
    return count_non_overlapping_origins(
        cast(pd.Series, selected["origin_utc"]).tolist(),
        horizon_minutes=horizon,
    )


def _development_target_floor(frame: pd.DataFrame) -> float:
    """Freeze the log-interval floor from positive development actual RV only."""

    actual = np.asarray(frame["target_rv"], dtype=float)
    if not bool(np.isfinite(actual).all()) or bool((actual < 0.0).any()):
        raise VolatilityError("development actual RV must be finite and non-negative")
    positive = actual[actual > 0.0]
    if len(positive) == 0:
        raise VolatilityError("development actual RV has no positive value")
    floor = float(np.min(positive) * 0.5)
    if not math.isfinite(floor) or floor <= 0.0:
        raise VolatilityError("development interval RV floor is invalid")
    return floor


def _evaluate_tuple(
    *,
    symbol: str,
    horizon: int,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    contract: VolatilityContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_h = fit.loc[pd.to_numeric(fit["horizon_minutes"], errors="coerce") == horizon].copy()
    calibration_symbol = calibration.loc[
        (cast(pd.Series, calibration["symbol"]).astype(str) == symbol)
        & (pd.to_numeric(calibration["horizon_minutes"], errors="coerce") == horizon)
    ].copy()
    test_symbol = test.loc[
        (cast(pd.Series, test["symbol"]).astype(str) == symbol)
        & (pd.to_numeric(test["horizon_minutes"], errors="coerce") == horizon)
    ].copy()
    fit_symbol = fit_h.loc[cast(pd.Series, fit_h["symbol"]).astype(str) == symbol].copy()
    origin_counts = {
        "fit": _origin_counts(fit, symbol=symbol, horizon=horizon),
        "calibration": _origin_counts(calibration, symbol=symbol, horizon=horizon),
        "test": _origin_counts(test, symbol=symbol, horizon=horizon),
    }
    required = {
        "fit": contract.min_fit_origins,
        "calibration": contract.min_calibration_origins,
        "test": contract.min_test_origins,
    }
    if any(origin_counts[key] < required[key] for key in required):
        raise VolatilityError(
            f"{symbol}/{horizon}: non-overlapping origin floors fail: "
            f"observed={origin_counts}, required={required}"
        )
    baseline = _select_baseline(fit_symbol, contract)
    challenger = _select_challenger(fit_h, symbol=symbol, contract=contract)
    training_frame = fit_h if challenger["scope"] == "pooled" else fit_symbol
    model = _fit_har(
        training_frame,
        alpha=float(challenger["alpha"]),
        pooled=challenger["scope"] == "pooled",
    )
    calibration_prediction = _predict_har(model, calibration_symbol)
    test_prediction = _predict_har(model, test_symbol)
    baseline_column = str(baseline["column"])
    calibration_baseline = np.asarray(calibration_symbol[baseline_column], dtype=float)
    test_baseline = np.asarray(test_symbol[baseline_column], dtype=float)
    calibration_actual = np.asarray(calibration_symbol["target_rv"], dtype=float)
    test_actual = np.asarray(test_symbol["target_rv"], dtype=float)
    interval_rv_floor = _development_target_floor(fit_symbol)

    candidate_interval = calibrate_log_interval(
        calibration_actual,
        calibration_prediction,
        rv_floor=interval_rv_floor,
        coverage=contract.prediction_interval_coverage,
    )
    baseline_interval = calibrate_log_interval(
        calibration_actual,
        calibration_baseline,
        rv_floor=interval_rv_floor,
        coverage=contract.prediction_interval_coverage,
    )
    candidate_scale = float(candidate_interval["interval_width_scale"])
    baseline_scale = float(baseline_interval["interval_width_scale"])
    if not math.isclose(candidate_scale, baseline_scale, rel_tol=0.0, abs_tol=0.0):
        raise VolatilityError("candidate and baseline interval scales differ")
    candidate_lower, candidate_upper = apply_log_interval(
        test_prediction, candidate_interval
    )
    baseline_lower, baseline_upper = apply_log_interval(test_baseline, baseline_interval)
    candidate_interval_metrics = interval_metrics(
        test_actual,
        candidate_lower,
        candidate_upper,
        scale=candidate_scale,
        target_coverage=contract.prediction_interval_coverage,
    )
    baseline_interval_metrics = interval_metrics(
        test_actual,
        baseline_lower,
        baseline_upper,
        scale=candidate_scale,
        target_coverage=contract.prediction_interval_coverage,
    )
    interval_gates = {
        "wilson_contains_target": bool(
            candidate_interval_metrics["wilson_contains_target"]
        ),
        "coverage_error_improves": float(
            candidate_interval_metrics["absolute_coverage_error"]
        )
        < float(baseline_interval_metrics["absolute_coverage_error"]),
        "normalized_width_improves": float(
            candidate_interval_metrics["mean_normalized_interval_width"]
        )
        < float(baseline_interval_metrics["mean_normalized_interval_width"]),
    }
    candidate_loss = qlike_loss(test_actual, test_prediction)
    baseline_loss = qlike_loss(test_actual, test_baseline)
    dm = newey_west_mean_test(
        candidate_loss - baseline_loss,
        horizon_minutes=horizon,
        issuance_cadence_minutes=contract.issuance_cadence_minutes,
    )

    tail_threshold = float(np.quantile(calibration_actual, 0.90, method="linear"))
    tail_mask = test_actual >= tail_threshold
    tail_count = int(tail_mask.sum())
    tail_diagnostic: dict[str, Any] = {
        "policy": "diagnostic_only",
        "threshold_source": "calibration_actual_rv_p90",
        "threshold": tail_threshold,
        "observation_count": tail_count,
        "eligibility_effect": "none",
    }
    if tail_count:
        tail_loss_differential = candidate_loss[tail_mask] - baseline_loss[tail_mask]
        tail_hac: dict[str, Any]
        try:
            tail_hac = {
                "status": "available",
                **newey_west_mean_test(
                    tail_loss_differential,
                    horizon_minutes=horizon,
                    issuance_cadence_minutes=contract.issuance_cadence_minutes,
                ),
            }
        except VolatilityError as exc:
            tail_hac = {"status": "unavailable", "reason": str(exc)}
        tail_diagnostic.update(
            {
                "candidate_mean_qlike": float(np.mean(candidate_loss[tail_mask])),
                "baseline_mean_qlike": float(np.mean(baseline_loss[tail_mask])),
                "paired_mean_loss_differential": float(
                    np.mean(tail_loss_differential)
                ),
                "hac_uncertainty": tail_hac,
                "candidate_underforecast_fraction": float(
                    np.mean(test_prediction[tail_mask] < test_actual[tail_mask])
                ),
                "baseline_underforecast_fraction": float(
                    np.mean(test_baseline[tail_mask] < test_actual[tail_mask])
                ),
                "candidate_lower_miss_count": int(
                    (test_actual[tail_mask] < candidate_lower[tail_mask]).sum()
                ),
                "candidate_upper_miss_count": int(
                    (test_actual[tail_mask] > candidate_upper[tail_mask]).sum()
                ),
                "baseline_lower_miss_count": int(
                    (test_actual[tail_mask] < baseline_lower[tail_mask]).sum()
                ),
                "baseline_upper_miss_count": int(
                    (test_actual[tail_mask] > baseline_upper[tail_mask]).sum()
                ),
            }
        )

    key = f"{symbol}|{horizon}"
    artifact_model = {
        **model,
        "symbol": symbol,
        "horizon_minutes": horizon,
        "candidate_interval": candidate_interval,
        "baseline_interval": baseline_interval,
        "baseline_column": baseline_column,
    }
    report = {
        "key": key,
        "symbol": symbol,
        "horizon_minutes": horizon,
        "status": "pending_holm",
        "origin_counts": origin_counts,
        "required_origin_counts": required,
        "baseline": baseline,
        "challenger": challenger,
        "selection_scope": "independent_per_symbol_horizon",
        "predictive_accuracy_test_scope": "symbol_loss_differential_only",
        "training_lineage": {
            "cohort": "development_fit_only",
            "scope": challenger["scope"],
            "symbols": sorted(
                cast(pd.Series, training_frame["symbol"]).astype(str).unique()
            ),
            "rows": len(training_frame),
            "max_origin_utc": cast(
                pd.Timestamp,
                cast(pd.Series, training_frame["origin_utc"]).max(),
            ).isoformat(),
            "max_label_end_utc": cast(
                pd.Timestamp,
                cast(pd.Series, training_frame["label_end_utc"]).max(),
            ).isoformat(),
        },
        "point_loss": {
            "metric": "patton_qlike",
            "candidate_mean": float(np.mean(candidate_loss)),
            "baseline_mean": float(np.mean(baseline_loss)),
            "dm_hac": dm,
        },
        "interval": {
            "candidate_calibration": candidate_interval,
            "baseline_calibration": baseline_interval,
            "candidate_test": candidate_interval_metrics,
            "baseline_test": baseline_interval_metrics,
            "gates": interval_gates,
        },
        "tail_diagnostic": tail_diagnostic,
        "verdict_influence": False,
    }
    return report, artifact_model


def train_evaluate_shadow_volatility(
    examples: pd.DataFrame,
    *,
    contract: VolatilityContract,
    output_dir: Path,
    source_audits: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Train both HAR challengers and evaluate one frozen symbol-level champion."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit, calibration, test, split_audit = _global_splits(examples, contract)
    manifest_columns = (
        "symbol",
        "origin_utc",
        "label_end_utc",
        "horizon_minutes",
        "target_rv",
        "target_volatility_pct",
        *FEATURE_COLUMNS,
        *(f"volatility_pct_{window}" for window in HAR_WINDOWS),
    )
    data_manifest = {
        "schema_version": "neutralgrid_shadow_volatility_data_manifest_v1",
        "created_at_utc": _utc_now().isoformat(),
        "contract_sha256": contract.contract_sha256,
        "rows": len(examples),
        "symbols": sorted(cast(pd.Series, examples["symbol"]).astype(str).unique()),
        "content_sha256": hash_frame(examples, manifest_columns),
        "final_test_content_sha256": hash_frame(test, manifest_columns),
        "final_test_frozen_before_selection": True,
        "source_audits": [dict(audit) for audit in source_audits],
    }
    _atomic_parquet(output_dir / "rv_examples.parquet", examples)
    data_manifest["rv_examples_sha256"] = _sha256_file(
        output_dir / "rv_examples.parquet"
    )
    _atomic_json(output_dir / "data_manifest.json", data_manifest)
    _atomic_json(output_dir / "split_audit.json", split_audit)

    reports: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    symbols = sorted(cast(pd.Series, examples["symbol"]).astype(str).unique())
    blockers: list[dict[str, Any]] = []
    for symbol in symbols:
        for horizon in contract.horizons_minutes:
            try:
                report, model = _evaluate_tuple(
                    symbol=symbol,
                    horizon=horizon,
                    fit=fit,
                    calibration=calibration,
                    test=test,
                    contract=contract,
                )
            except VolatilityError as exc:
                blockers.append(
                    {
                        "symbol": symbol,
                        "horizon_minutes": horizon,
                        "reason": str(exc),
                    }
                )
                continue
            reports.append(report)
            models[str(report["key"])] = model

    if reports:
        adjusted = holm_adjust(
            [
                float(cast(Mapping[str, Any], report["point_loss"])["dm_hac"]["one_sided_p_value"])
                for report in reports
            ]
        )
        for report, adjusted_p in zip(reports, adjusted, strict=True):
            point_loss = cast(dict[str, Any], report["point_loss"])
            dm = cast(dict[str, Any], point_loss["dm_hac"])
            dm["holm_adjusted_p_value"] = adjusted_p
            qlike_gate = (
                float(dm["one_sided_upper_confidence_bound"]) <= 0.0
                and adjusted_p <= 0.05
            )
            interval_gates = cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], report["interval"])["gates"],
            )
            interval_gate = all(bool(value) for value in interval_gates.values())
            eligible = qlike_gate and interval_gate
            report["gates"] = {
                "zero_degradation_qlike": qlike_gate,
                "mandatory_interval_quality": interval_gate,
            }
            report["forecast_eligible"] = eligible
            report["status"] = "shadow_oos_validated" if eligible else "shadow_oos_rejected"

    artifact_payload = {
        "schema_version": VOLATILITY_ARTIFACT_SCHEMA,
        "contract_sha256": contract.contract_sha256,
        "created_at_utc": _utc_now().isoformat(),
        "models": models,
    }
    model_path = output_dir / "model.joblib"
    _atomic_joblib(model_path, artifact_payload)
    model_sha = _sha256_file(model_path)
    eligible_keys = sorted(
        str(report["key"])
        for report in reports
        if report.get("forecast_eligible") is True
    )
    metadata = {
        "schema_version": VOLATILITY_ARTIFACT_SCHEMA,
        "created_at_utc": _utc_now().isoformat(),
        "contract_path": str(contract.path),
        "contract_sha256": contract.contract_sha256,
        "data_manifest_sha256": _sha256_file(output_dir / "data_manifest.json"),
        "model_sha256": model_sha,
        "forecast_eligible": bool(eligible_keys),
        "eligible_keys": eligible_keys,
        "verdict_influence": False,
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    result = {
        "schema_version": VOLATILITY_REPORT_SCHEMA,
        "created_at_utc": _utc_now().isoformat(),
        "status": "shadow_oos_validated" if eligible_keys else "shadow_oos_rejected",
        "forecast_eligible": bool(eligible_keys),
        "eligible_keys": eligible_keys,
        "contract": asdict(contract) | {"path": str(contract.path)},
        "data_manifest": data_manifest,
        "split_audit": split_audit,
        "results": reports,
        "multiple_testing": {
            "method": "holm",
            "family": "all_evaluated_symbol_horizon_pairs",
            "dependence_assumption": "valid_under_arbitrary_dependence",
            "panel_pooled_dm_prohibited": True,
        },
        "blockers": blockers,
        "artifact_paths": {
            "data_manifest": str(output_dir / "data_manifest.json"),
            "split_audit": str(output_dir / "split_audit.json"),
            "model": str(model_path),
            "metadata": str(output_dir / "metadata.json"),
            "oos_report": str(output_dir / "oos_report.json"),
            "rv_examples": str(output_dir / "rv_examples.parquet"),
        },
        "verdict_influence": False,
    }
    _atomic_json(output_dir / "oos_report.json", result)
    return result


def _load_artifact(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = artifact_dir.resolve()
    metadata_path = resolved / "metadata.json"
    model_path = resolved / "model.joblib"
    metadata = _strict_json(metadata_path)
    if metadata.get("schema_version") != VOLATILITY_ARTIFACT_SCHEMA:
        raise VolatilityError("unsupported volatility artifact metadata schema")
    if metadata.get("verdict_influence") is not False:
        raise VolatilityError("volatility artifact is not verdict-inert")
    if metadata.get("forecast_eligible") is not True:
        raise VolatilityError("volatility artifact has no eligible symbol/horizon")
    if metadata.get("model_sha256") != _sha256_file(model_path):
        raise VolatilityError("volatility model SHA-256 mismatch")
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise VolatilityError(f"cannot load volatility model: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("schema_version") != VOLATILITY_ARTIFACT_SCHEMA:
        raise VolatilityError("volatility model payload schema is invalid")
    if artifact.get("contract_sha256") != metadata.get("contract_sha256"):
        raise VolatilityError("volatility artifact contract hash mismatch")
    return metadata, artifact


def select_runtime_horizon(requested_minutes: int, eligible_horizons: Sequence[int]) -> int:
    """Return nearest non-shorter horizon; never extrapolate past 360 minutes."""

    if requested_minutes <= 0 or requested_minutes > 360:
        raise VolatilityError("requested runtime horizon must lie in (0, 360]")
    options = sorted({int(value) for value in eligible_horizons if int(value) >= requested_minutes})
    if not options:
        raise VolatilityError("no eligible non-shorter volatility horizon")
    return options[0]


def predict_shadow_volatility(
    artifact_dir: Path,
    *,
    contract: VolatilityContract,
    symbol: str,
    strategy_id: str,
    mark_frame: pd.DataFrame,
    last_frame: pd.DataFrame | None = None,
    requested_horizon_minutes: int = 360,
    asof_utc: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Produce one verdict-inert causal volatility forecast."""

    metadata, artifact = _load_artifact(artifact_dir)
    if metadata.get("contract_sha256") != contract.contract_sha256:
        raise VolatilityError("runtime contract differs from trained artifact")
    prefix = f"{symbol.upper()}|"
    eligible_keys = [
        str(key) for key in metadata.get("eligible_keys", []) if str(key).startswith(prefix)
    ]
    eligible_horizons = [int(key.split("|", 1)[1]) for key in eligible_keys]
    horizon = select_runtime_horizon(requested_horizon_minutes, eligible_horizons)
    key = f"{symbol.upper()}|{horizon}"
    models = artifact.get("models")
    if not isinstance(models, Mapping) or key not in models:
        raise VolatilityError(f"eligible volatility model {key} is missing")
    model = models[key]
    if not isinstance(model, Mapping):
        raise VolatilityError(f"volatility model {key} is invalid")
    snapshot = latest_rv_snapshot(
        mark_frame,
        last_frame,
        symbol=symbol,
        contract=contract,
        asof_utc=asof_utc,
    )
    feature_row = {
        "symbol": symbol.upper(),
        **cast(Mapping[str, Any], snapshot["features"]),
    }
    prediction = float(_predict_har(model, pd.DataFrame([feature_row]))[0])
    interval_contract = model.get("candidate_interval")
    if not isinstance(interval_contract, Mapping):
        raise VolatilityError("volatility model interval calibration is missing")
    lower, upper = apply_log_interval([prediction], interval_contract)
    model_path = artifact_dir.resolve() / "model.joblib"
    return {
        "schema_version": VOLATILITY_OUTPUT_SCHEMA,
        "status": "available",
        "symbol": symbol.upper(),
        "strategy_id": strategy_id,
        "cutoff_utc": snapshot["cutoff_utc"],
        "latest_source_timestamp_utc": snapshot["cutoff_utc"],
        "freshness_seconds": snapshot["freshness_seconds"],
        "requested_horizon_minutes": requested_horizon_minutes,
        "horizon_minutes": horizon,
        "forecast_rv": prediction,
        "forecast_volatility_pct": 100.0 * math.sqrt(prediction),
        "prediction_interval_lower_rv": float(lower[0]),
        "prediction_interval_upper_rv": float(upper[0]),
        "prediction_interval_coverage": contract.prediction_interval_coverage,
        "mark_rv": snapshot["mark_rv"],
        "last_rv": snapshot["last_rv"],
        "mark_volatility_pct": snapshot["mark_volatility_pct"],
        "last_volatility_pct": snapshot["last_volatility_pct"],
        "mark_last_rv_divergence": snapshot["mark_last_rv_divergence"],
        "model_scope": model.get("scope"),
        "contract_sha256": contract.contract_sha256,
        "model_sha256": _sha256_file(model_path),
        "data_manifest_sha256": metadata.get("data_manifest_sha256"),
        "source_quality": {
            "mark_price_audit": snapshot["mark_price_audit"],
            "last_price_audit": snapshot["last_price_audit"],
        },
        "gaps": {
            "mark_gap_count": cast(Mapping[str, Any], snapshot["mark_price_audit"])[
                "gap_count"
            ],
            "mark_missing_minutes": cast(
                Mapping[str, Any], snapshot["mark_price_audit"]
            )["missing_minutes"],
            "last_gap_count": (
                None
                if snapshot["last_price_audit"] is None
                else cast(Mapping[str, Any], snapshot["last_price_audit"])[
                    "gap_count"
                ]
            ),
            "last_missing_minutes": (
                None
                if snapshot["last_price_audit"] is None
                else cast(Mapping[str, Any], snapshot["last_price_audit"])[
                    "missing_minutes"
                ]
            ),
        },
        "eligibility": True,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
