"""Diagnostic-only evaluation for the governed volatility research v2 stage."""

from __future__ import annotations

import hashlib
import json
import math
import time
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
    VolatilityContract,
    apply_log_interval,
    calibrate_log_interval,
    hash_frame,
    holm_adjust,
    interval_metrics,
    newey_west_mean_test,
    qlike_loss,
)
from neutralgrid.live.decision.volatility_forecast import (
    VOLATILITY_ARTIFACT_SCHEMA,
    VOLATILITY_REPORT_SCHEMA,
    _development_target_floor,
    _expanding_folds,
    _global_splits,
    _origin_counts,
    _predict_har,
    _select_baseline,
)
from neutralgrid.live.decision.volatility_research import (
    JUMP_FEATURE_COLUMNS,
    RESEARCH_FEATURE_COLUMNS,
    SEMIVARIANCE_FEATURE_COLUMNS,
    VolatilityResearchContract,
    VolatilityResearchError,
    deduplicate_research_examples,
)


UTC = timezone.utc
RESEARCH_ARTIFACT_SCHEMA = "neutralgrid_live_volatility_research_artifact_v2"
RESEARCH_REPORT_SCHEMA = "neutralgrid_live_volatility_research_report_v2"
RESEARCH_DATA_MANIFEST_SCHEMA = (
    "neutralgrid_live_volatility_research_data_manifest_v2"
)
HAR_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"rv_{window}" for window in HAR_WINDOWS
)
FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "har": HAR_FEATURE_COLUMNS,
    "har_rs": SEMIVARIANCE_FEATURE_COLUMNS,
    "har_j": (*HAR_FEATURE_COLUMNS, *JUMP_FEATURE_COLUMNS),
    "har_rs_j": (*SEMIVARIANCE_FEATURE_COLUMNS, *JUMP_FEATURE_COLUMNS),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VolatilityResearchError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VolatilityResearchError(f"{path}: non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except VolatilityResearchError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityResearchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VolatilityResearchError(f"{path}: JSON root must be an object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise VolatilityResearchError(f"cannot write {path}: {exc}") from exc


def _manifest_columns(include_semivariance: bool) -> tuple[str, ...]:
    columns: tuple[str, ...] = (
        "symbol",
        "origin_utc",
        "label_end_utc",
        "horizon_minutes",
        "target_rv",
        "target_volatility_pct",
        *HAR_FEATURE_COLUMNS,
        *(f"volatility_pct_{window}" for window in HAR_WINDOWS),
    )
    if include_semivariance:
        return (*columns, *RESEARCH_FEATURE_COLUMNS)
    return columns


def load_consumed_v1_evidence(
    artifact_dir: Path,
    *,
    base_contract: VolatilityContract,
) -> dict[str, Any]:
    """Verify and load a v1 OOS artifact as consumed diagnostic evidence."""

    resolved = artifact_dir.resolve()
    paths = {
        "data_manifest": resolved / "data_manifest.json",
        "split_audit": resolved / "split_audit.json",
        "metadata": resolved / "metadata.json",
        "model": resolved / "model.joblib",
        "oos_report": resolved / "oos_report.json",
        "rv_examples": resolved / "rv_examples.parquet",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise VolatilityResearchError(
            f"base volatility artifact lacks required files: {missing}"
        )
    data_manifest = _strict_json(paths["data_manifest"])
    split_audit = _strict_json(paths["split_audit"])
    metadata = _strict_json(paths["metadata"])
    report = _strict_json(paths["oos_report"])
    if data_manifest.get("schema_version") != (
        "neutralgrid_shadow_volatility_data_manifest_v1"
    ):
        raise VolatilityResearchError("unsupported base data manifest schema")
    if metadata.get("schema_version") != VOLATILITY_ARTIFACT_SCHEMA:
        raise VolatilityResearchError("unsupported base model metadata schema")
    if report.get("schema_version") != VOLATILITY_REPORT_SCHEMA:
        raise VolatilityResearchError("unsupported base OOS report schema")
    if metadata.get("verdict_influence") is not False or report.get(
        "verdict_influence"
    ) is not False:
        raise VolatilityResearchError("base volatility evidence is not verdict-inert")
    if metadata.get("contract_sha256") != base_contract.contract_sha256 or (
        data_manifest.get("contract_sha256") != base_contract.contract_sha256
    ):
        raise VolatilityResearchError("base volatility contract hash mismatch")
    if data_manifest.get("final_test_frozen_before_selection") is not True:
        raise VolatilityResearchError("base final test was not frozen before selection")
    if metadata.get("data_manifest_sha256") != _sha256_file(
        paths["data_manifest"]
    ):
        raise VolatilityResearchError("base data manifest SHA-256 mismatch")
    if metadata.get("model_sha256") != _sha256_file(paths["model"]):
        raise VolatilityResearchError("base model SHA-256 mismatch")
    if data_manifest.get("rv_examples_sha256") != _sha256_file(
        paths["rv_examples"]
    ):
        raise VolatilityResearchError("base RV examples SHA-256 mismatch")
    try:
        examples = pd.read_parquet(paths["rv_examples"])
    except (OSError, ValueError, ImportError) as exc:
        raise VolatilityResearchError(
            f"cannot read base RV examples: {exc}"
        ) from exc
    columns = _manifest_columns(include_semivariance=False)
    if data_manifest.get("content_sha256") != hash_frame(examples, columns):
        raise VolatilityResearchError("base RV example content hash mismatch")
    _, _, test_original, observed_split = _global_splits(examples, base_contract)
    if observed_split != split_audit:
        raise VolatilityResearchError("base chronological split audit is not reproducible")
    if data_manifest.get("final_test_content_sha256") != hash_frame(
        test_original, columns
    ):
        raise VolatilityResearchError("base final-test content hash mismatch")
    clean, duplicate_audit = deduplicate_research_examples(examples)
    if duplicate_audit["exact_duplicate_rows_removed"] != 0:
        raise VolatilityResearchError(
            "checksum-verified base RV examples unexpectedly contain duplicates"
        )
    fit, calibration, test, observed_split = _global_splits(clean, base_contract)
    if observed_split != split_audit:
        raise VolatilityResearchError("base chronological split audit is not reproducible")
    try:
        model_payload = joblib.load(paths["model"])
    except Exception as exc:
        raise VolatilityResearchError(f"cannot load base model: {exc}") from exc
    if not isinstance(model_payload, dict) or model_payload.get(
        "schema_version"
    ) != VOLATILITY_ARTIFACT_SCHEMA:
        raise VolatilityResearchError("base model payload schema is invalid")
    if model_payload.get("contract_sha256") != base_contract.contract_sha256:
        raise VolatilityResearchError("base model payload contract hash mismatch")
    models = model_payload.get("models")
    if not isinstance(models, Mapping):
        raise VolatilityResearchError("base model payload has no model mapping")
    report_results = report.get("results")
    if not isinstance(report_results, list):
        raise VolatilityResearchError("base OOS report has no result list")
    return {
        "artifact_dir": str(resolved),
        "paths": {name: str(path) for name, path in paths.items()},
        "data_manifest": data_manifest,
        "split_audit": split_audit,
        "metadata": metadata,
        "report": report,
        "report_results": report_results,
        "model_payload": model_payload,
        "examples": clean,
        "fit": fit,
        "calibration": calibration,
        "test": test,
        "duplicate_audit": duplicate_audit,
        "evidence_role": "consumed_diagnostic_only",
        "promotion_eligible": False,
    }


def _positive_floor(frame: pd.DataFrame, feature_columns: Sequence[str]) -> float:
    positive_arrays: list[np.ndarray] = []
    for column in (*feature_columns, "target_rv"):
        values = np.asarray(frame[column], dtype=float)
        if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
            raise VolatilityResearchError(f"research column {column} is invalid")
        positive = values[values > 0.0]
        if len(positive):
            positive_arrays.append(positive)
    if not positive_arrays:
        raise VolatilityResearchError("research fit has no positive variance value")
    floor = float(np.min(np.concatenate(positive_arrays)) * 0.5)
    if not math.isfinite(floor) or floor <= 0.0:
        raise VolatilityResearchError("research variance floor is invalid")
    return floor


def _transform_features(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    floor: float,
    pooled: bool,
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        values = np.asarray(frame[column], dtype=float)
        if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
            raise VolatilityResearchError(f"research feature {column} is invalid")
        transformed[column] = np.log(np.maximum(values, floor))
    if pooled:
        transformed["symbol"] = cast(pd.Series, frame["symbol"]).astype(str)
    return transformed


def _fit_research_model(
    frame: pd.DataFrame,
    *,
    family: str,
    alpha: float,
    pooled: bool,
) -> dict[str, Any]:
    feature_columns = FEATURE_FAMILIES.get(family)
    if feature_columns is None:
        raise VolatilityResearchError(f"unsupported research family {family}")
    if frame.empty:
        raise VolatilityResearchError("cannot fit research model on an empty frame")
    floor = _positive_floor(frame, feature_columns)
    transformed = _transform_features(
        frame,
        feature_columns=feature_columns,
        floor=floor,
        pooled=pooled,
    )
    target = np.log(np.maximum(np.asarray(frame["target_rv"], dtype=float), floor))
    if pooled:
        preprocessor = ColumnTransformer(
            transformers=[
                ("variance", StandardScaler(), list(feature_columns)),
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
    try:
        pipeline.fit(transformed, target)
    except (ValueError, FloatingPointError) as exc:
        raise VolatilityResearchError(f"research fit failed: {exc}") from exc
    return {
        "pipeline": pipeline,
        "family": family,
        "feature_columns": list(feature_columns),
        "rv_floor": floor,
        "alpha": alpha,
        "scope": "pooled" if pooled else "per_symbol",
        "verdict_influence": False,
    }


def _predict_research_model(
    model: Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    pipeline = model.get("pipeline")
    family = model.get("family")
    floor_value = model.get("rv_floor")
    scope = model.get("scope")
    if not isinstance(pipeline, Pipeline) or not isinstance(family, str):
        raise VolatilityResearchError("research model payload is invalid")
    feature_columns = FEATURE_FAMILIES.get(family)
    if feature_columns is None or model.get("feature_columns") != list(feature_columns):
        raise VolatilityResearchError("research model feature schema is invalid")
    if isinstance(floor_value, (bool, np.bool_)) or not isinstance(
        floor_value, (int, float, np.integer, np.floating)
    ):
        raise VolatilityResearchError("research model RV floor is invalid")
    floor = float(floor_value)
    if not math.isfinite(floor) or floor <= 0.0:
        raise VolatilityResearchError("research model RV floor is invalid")
    transformed = _transform_features(
        frame,
        feature_columns=feature_columns,
        floor=floor,
        pooled=scope == "pooled",
    )
    try:
        predicted_log = np.asarray(pipeline.predict(transformed), dtype=float)
    except (ValueError, FloatingPointError) as exc:
        raise VolatilityResearchError(f"research prediction failed: {exc}") from exc
    predicted = np.maximum(np.exp(np.clip(predicted_log, -700.0, 700.0)), floor)
    if not bool(np.isfinite(predicted).all()) or bool((predicted <= 0.0).any()):
        raise VolatilityResearchError("research prediction is invalid")
    return predicted


def _development_score(
    development: pd.DataFrame,
    *,
    symbol: str,
    family: str,
    alpha: float,
    pooled: bool,
) -> tuple[float, list[dict[str, Any]]]:
    symbol_frame = development.loc[
        cast(pd.Series, development["symbol"]).astype(str) == symbol
    ].copy()
    losses: list[float] = []
    audits: list[dict[str, Any]] = []
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
            & (cast(pd.Series, symbol_frame["label_end_utc"]) <= validation_cutoff)
        ].copy()
        if len(train) < 20 or len(validation) < 5:
            raise VolatilityResearchError(
                "purged expanding fold lacks fit or validation rows"
            )
        model = _fit_research_model(
            train,
            family=family,
            alpha=alpha,
            pooled=pooled,
        )
        prediction = _predict_research_model(model, validation)
        loss = float(
            np.mean(
                qlike_loss(
                    np.asarray(validation["target_rv"], dtype=float), prediction
                )
            )
        )
        losses.append(loss)
        audits.append(
            {
                "train_cutoff_utc": train_cutoff.isoformat(),
                "validation_cutoff_utc": validation_cutoff.isoformat(),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "validation_max_label_end_utc": cast(
                    pd.Timestamp,
                    cast(pd.Series, validation["label_end_utc"]).max(),
                ).isoformat(),
                "mean_qlike": loss,
            }
        )
    return float(np.mean(losses)), audits


def _select_research_candidate(
    development: pd.DataFrame,
    *,
    symbol: str,
    contract: VolatilityContract,
    research_contract: VolatilityResearchContract,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for family in research_contract.candidate_families:
        for scope in ("per_symbol", "pooled"):
            for alpha in contract.ridge_alphas:
                score, folds = _development_score(
                    development,
                    symbol=symbol,
                    family=family,
                    alpha=alpha,
                    pooled=scope == "pooled",
                )
                candidates.append(
                    {
                        "family": family,
                        "scope": scope,
                        "alpha": alpha,
                        "development_qlike": score,
                        "folds": folds,
                    }
                )
    winner = min(
        candidates,
        key=lambda item: (
            float(item["development_qlike"]),
            str(item["family"]),
            str(item["scope"]),
            float(item["alpha"]),
        ),
    )
    return {**winner, "candidates": candidates}


def _residual_statistics(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    floor: float,
) -> dict[str, Any]:
    if len(actual) != len(forecast) or len(actual) == 0:
        raise VolatilityResearchError("residual inputs are empty or misaligned")
    residual = np.log(np.maximum(actual, floor)) - np.log(
        np.maximum(forecast, floor)
    )
    if not bool(np.isfinite(residual).all()):
        raise VolatilityResearchError("signed log residual is non-finite")
    return {
        "observation_count": len(residual),
        "mean_signed_log_residual": float(np.mean(residual)),
        "median_signed_log_residual": float(np.median(residual)),
        "p05_signed_log_residual": float(np.quantile(residual, 0.05)),
        "p95_signed_log_residual": float(np.quantile(residual, 0.95)),
        "underforecast_fraction": float(np.mean(forecast < actual)),
        "zero_actual_count": int((actual == 0.0).sum()),
        "sign_convention": "positive_means_actual_rv_exceeds_forecast_rv",
    }


def _rolling_residual_blocks(
    frame: pd.DataFrame,
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    floor: float,
) -> list[dict[str, Any]]:
    if len(frame) != len(actual):
        raise VolatilityResearchError("residual frame is misaligned")
    blocks: list[dict[str, Any]] = []
    for block_number, indexes in enumerate(np.array_split(np.arange(len(frame)), 4), 1):
        if len(indexes) == 0:
            continue
        block_frame = frame.iloc[indexes]
        blocks.append(
            {
                "block": block_number,
                "first_origin_utc": cast(
                    pd.Timestamp, block_frame["origin_utc"].iloc[0]
                ).isoformat(),
                "last_origin_utc": cast(
                    pd.Timestamp, block_frame["origin_utc"].iloc[-1]
                ).isoformat(),
                **_residual_statistics(
                    actual[indexes], forecast[indexes], floor=floor
                ),
            }
        )
    return blocks


def _model_drift_report(
    calibration_actual: np.ndarray,
    calibration_forecast: np.ndarray,
    test_frame: pd.DataFrame,
    test_actual: np.ndarray,
    test_forecast: np.ndarray,
    *,
    floor: float,
) -> dict[str, Any]:
    calibration_stats = _residual_statistics(
        calibration_actual,
        calibration_forecast,
        floor=floor,
    )
    test_stats = _residual_statistics(test_actual, test_forecast, floor=floor)
    return {
        "calibration": calibration_stats,
        "consumed_test": test_stats,
        "mean_signed_log_residual_shift": float(
            test_stats["mean_signed_log_residual"]
        )
        - float(calibration_stats["mean_signed_log_residual"]),
        "underforecast_fraction_shift": float(
            test_stats["underforecast_fraction"]
        )
        - float(calibration_stats["underforecast_fraction"]),
        "consumed_test_blocks": _rolling_residual_blocks(
            test_frame,
            test_actual,
            test_forecast,
            floor=floor,
        ),
    }


def _find_base_result(
    results: Sequence[Any],
    *,
    symbol: str,
    horizon: int,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in results
        if isinstance(item, Mapping)
        and item.get("symbol") == symbol
        and item.get("horizon_minutes") == horizon
    ]
    if len(matches) != 1:
        raise VolatilityResearchError(
            f"base report must contain exactly one {symbol}/{horizon} result"
        )
    return cast(Mapping[str, Any], matches[0])


def _evaluate_pair(
    *,
    symbol: str,
    horizon: int,
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    base_models: Mapping[str, Any],
    base_results: Sequence[Any],
    contract: VolatilityContract,
    research_contract: VolatilityResearchContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_h = fit.loc[
        pd.to_numeric(fit["horizon_minutes"], errors="coerce") == horizon
    ].copy()
    fit_symbol = fit_h.loc[
        cast(pd.Series, fit_h["symbol"]).astype(str) == symbol
    ].copy()
    calibration_symbol = calibration.loc[
        (cast(pd.Series, calibration["symbol"]).astype(str) == symbol)
        & (pd.to_numeric(calibration["horizon_minutes"], errors="coerce") == horizon)
    ].copy()
    test_symbol = test.loc[
        (cast(pd.Series, test["symbol"]).astype(str) == symbol)
        & (pd.to_numeric(test["horizon_minutes"], errors="coerce") == horizon)
    ].copy()
    if fit_symbol.empty or calibration_symbol.empty or test_symbol.empty:
        raise VolatilityResearchError(f"{symbol}/{horizon}: empty research cohort")
    origin_counts = {
        "fit": _origin_counts(fit, symbol=symbol, horizon=horizon),
        "calibration": _origin_counts(
            calibration, symbol=symbol, horizon=horizon
        ),
        "test": _origin_counts(test, symbol=symbol, horizon=horizon),
    }
    required_origin_counts = {
        "fit": contract.min_fit_origins,
        "calibration": contract.min_calibration_origins,
        "test": contract.min_test_origins,
    }
    if any(
        origin_counts[cohort] < required_origin_counts[cohort]
        for cohort in required_origin_counts
    ):
        raise VolatilityResearchError(
            f"{symbol}/{horizon}: non-overlapping origin floors fail: "
            f"observed={origin_counts}, required={required_origin_counts}"
        )

    baseline = _select_baseline(fit_symbol, contract)
    base_result = _find_base_result(base_results, symbol=symbol, horizon=horizon)
    base_baseline = base_result.get("baseline")
    if not isinstance(base_baseline, Mapping) or base_baseline.get(
        "column"
    ) != baseline["column"]:
        raise VolatilityResearchError(
            f"{symbol}/{horizon}: base baseline selection is not reproducible"
        )
    selected = _select_research_candidate(
        fit_h,
        symbol=symbol,
        contract=contract,
        research_contract=research_contract,
    )
    training_frame = fit_h if selected["scope"] == "pooled" else fit_symbol
    model = _fit_research_model(
        training_frame,
        family=str(selected["family"]),
        alpha=float(selected["alpha"]),
        pooled=selected["scope"] == "pooled",
    )
    calibration_prediction = _predict_research_model(model, calibration_symbol)
    test_prediction = _predict_research_model(model, test_symbol)
    calibration_actual = np.asarray(calibration_symbol["target_rv"], dtype=float)
    test_actual = np.asarray(test_symbol["target_rv"], dtype=float)
    baseline_column = str(baseline["column"])
    calibration_baseline = np.asarray(
        calibration_symbol[baseline_column], dtype=float
    )
    test_baseline = np.asarray(test_symbol[baseline_column], dtype=float)
    interval_floor = _development_target_floor(fit_symbol)
    candidate_interval = calibrate_log_interval(
        calibration_actual,
        calibration_prediction,
        rv_floor=interval_floor,
        coverage=contract.prediction_interval_coverage,
    )
    baseline_interval = calibrate_log_interval(
        calibration_actual,
        calibration_baseline,
        rv_floor=interval_floor,
        coverage=contract.prediction_interval_coverage,
    )
    candidate_lower, candidate_upper = apply_log_interval(
        test_prediction, candidate_interval
    )
    baseline_lower, baseline_upper = apply_log_interval(
        test_baseline, baseline_interval
    )
    scale = float(candidate_interval["interval_width_scale"])
    if scale != float(baseline_interval["interval_width_scale"]):
        raise VolatilityResearchError(
            f"{symbol}/{horizon}: candidate and baseline interval scales differ"
        )
    candidate_interval_metrics = interval_metrics(
        test_actual,
        candidate_lower,
        candidate_upper,
        scale=scale,
        target_coverage=contract.prediction_interval_coverage,
    )
    baseline_interval_metrics = interval_metrics(
        test_actual,
        baseline_lower,
        baseline_upper,
        scale=scale,
        target_coverage=contract.prediction_interval_coverage,
    )
    candidate_loss = qlike_loss(test_actual, test_prediction)
    baseline_loss = qlike_loss(test_actual, test_baseline)
    dm = newey_west_mean_test(
        candidate_loss - baseline_loss,
        horizon_minutes=horizon,
        issuance_cadence_minutes=contract.issuance_cadence_minutes,
    )

    key = f"{symbol}|{horizon}"
    original_model = base_models.get(key)
    if not isinstance(original_model, Mapping):
        raise VolatilityResearchError(f"base model {key} is missing")
    original_calibration_prediction = _predict_har(
        original_model, calibration_symbol
    )
    original_test_prediction = _predict_har(original_model, test_symbol)
    original_interval = original_model.get("candidate_interval")
    if not isinstance(original_interval, Mapping):
        raise VolatilityResearchError(f"base model {key} interval is missing")
    original_lower, original_upper = apply_log_interval(
        original_test_prediction, original_interval
    )
    original_metrics = interval_metrics(
        test_actual,
        original_lower,
        original_upper,
        scale=scale,
        target_coverage=contract.prediction_interval_coverage,
    )
    original_loss = qlike_loss(test_actual, original_test_prediction)
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
    residual_drift = {
        "base_v1_candidate": _model_drift_report(
            calibration_actual,
            original_calibration_prediction,
            test_symbol,
            test_actual,
            original_test_prediction,
            floor=interval_floor,
        ),
        "selected_v2_research_candidate": _model_drift_report(
            calibration_actual,
            calibration_prediction,
            test_symbol,
            test_actual,
            test_prediction,
            floor=interval_floor,
        ),
        "selected_baseline": _model_drift_report(
            calibration_actual,
            calibration_baseline,
            test_symbol,
            test_actual,
            test_baseline,
            floor=interval_floor,
        ),
    }
    report = {
        "key": key,
        "symbol": symbol,
        "horizon_minutes": horizon,
        "status": "pending_holm_diagnostic_only",
        "evidence_role": "consumed_diagnostic_only",
        "origin_counts": origin_counts,
        "required_origin_counts": required_origin_counts,
        "selected_candidate": selected,
        "selected_baseline": baseline,
        "training_rows": len(training_frame),
        "training_max_label_end_utc": cast(
            pd.Timestamp,
            cast(pd.Series, training_frame["label_end_utc"]).max(),
        ).isoformat(),
        "point_loss": {
            "metric": "patton_qlike",
            "candidate_mean": float(np.mean(candidate_loss)),
            "baseline_mean": float(np.mean(baseline_loss)),
            "base_v1_candidate_mean": float(np.mean(original_loss)),
            "candidate_minus_baseline": float(
                np.mean(candidate_loss) - np.mean(baseline_loss)
            ),
            "candidate_minus_base_v1_candidate": float(
                np.mean(candidate_loss) - np.mean(original_loss)
            ),
            "dm_hac": dm,
        },
        "interval": {
            "candidate_calibration": candidate_interval,
            "baseline_calibration": baseline_interval,
            "candidate_consumed_test": candidate_interval_metrics,
            "baseline_consumed_test": baseline_interval_metrics,
            "base_v1_candidate_consumed_test": original_metrics,
            "gates": interval_gates,
        },
        "residual_drift": residual_drift,
        "promotion_eligible": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
    artifact_model = {
        **model,
        "symbol": symbol,
        "horizon_minutes": horizon,
        "candidate_interval": candidate_interval,
        "baseline_interval": baseline_interval,
        "baseline_column": baseline_column,
        "promotion_eligible": False,
        "verdict_influence": False,
    }
    return report, artifact_model


def run_consumed_holdout_research(
    examples: pd.DataFrame,
    *,
    evidence: Mapping[str, Any],
    base_contract: VolatilityContract,
    research_contract: VolatilityResearchContract,
    output_dir: Path,
    feature_audit: Mapping[str, Any],
    artifact_path_root: Path | None = None,
    execution_started_at_utc: datetime | None = None,
    execution_started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Run the v2 ablation without granting any promotion authority."""

    started_at = execution_started_at_utc or datetime.now(UTC)
    started_monotonic = (
        time.monotonic()
        if execution_started_monotonic is None
        else execution_started_monotonic
    )
    if started_at.tzinfo is None or not math.isfinite(started_monotonic):
        raise VolatilityResearchError("research execution timing inputs are invalid")
    clean, duplicate_audit = deduplicate_research_examples(examples)
    required = set(_manifest_columns(include_semivariance=True))
    missing = sorted(required - set(clean.columns))
    if missing:
        raise VolatilityResearchError(f"research examples lack columns: {missing}")
    fit, calibration, test, split_audit = _global_splits(clean, base_contract)
    if split_audit != evidence.get("split_audit"):
        raise VolatilityResearchError(
            "research split differs from checksum-verified v1 evidence"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    examples_path = output_dir / "research_examples.parquet"
    try:
        clean.to_parquet(examples_path, index=False)
    except (OSError, ValueError, ImportError) as exc:
        raise VolatilityResearchError(
            f"cannot write research examples: {exc}"
        ) from exc
    base_models = cast(Mapping[str, Any], evidence["model_payload"])["models"]
    base_results = cast(Sequence[Any], evidence["report_results"])
    reports: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    blockers: list[dict[str, Any]] = []
    symbols = sorted(cast(pd.Series, clean["symbol"]).astype(str).unique())
    for symbol in symbols:
        for horizon in base_contract.horizons_minutes:
            try:
                report, model = _evaluate_pair(
                    symbol=symbol,
                    horizon=horizon,
                    fit=fit,
                    calibration=calibration,
                    test=test,
                    base_models=cast(Mapping[str, Any], base_models),
                    base_results=base_results,
                    contract=base_contract,
                    research_contract=research_contract,
                )
            except (VolatilityResearchError, ValueError, FloatingPointError) as exc:
                blockers.append(
                    {
                        "symbol": symbol,
                        "horizon_minutes": horizon,
                        "failure_class": type(exc).__name__,
                        "reason": str(exc),
                    }
                )
                continue
            reports.append(report)
            models[str(report["key"])] = model
    if not reports:
        raise VolatilityResearchError("all v2 research comparisons were blocked")
    adjusted = holm_adjust(
        [
            float(
                cast(Mapping[str, Any], report["point_loss"])["dm_hac"][
                    "one_sided_p_value"
                ]
            )
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
        interval_gate = all(
            bool(value)
            for value in cast(
                Mapping[str, Any],
                cast(Mapping[str, Any], report["interval"])["gates"],
            ).values()
        )
        report["diagnostic_gates"] = {
            "zero_degradation_qlike": qlike_gate,
            "mandatory_interval_quality": interval_gate,
        }
        report["passes_consumed_holdout_diagnostic"] = qlike_gate and interval_gate
        report["status"] = (
            "diagnostic_pass_not_promotion_eligible"
            if qlike_gate and interval_gate
            else "diagnostic_rejected"
        )

    artifact_payload = {
        "schema_version": RESEARCH_ARTIFACT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_contract_sha256": base_contract.contract_sha256,
        "research_contract_sha256": research_contract.contract_sha256,
        "models": models,
        "promotion_eligible": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
    model_path = output_dir / "research_model.joblib"
    try:
        joblib.dump(artifact_payload, model_path)
    except (OSError, ValueError, TypeError) as exc:
        raise VolatilityResearchError(f"cannot write research model: {exc}") from exc
    columns = _manifest_columns(include_semivariance=True)
    data_manifest = {
        "schema_version": RESEARCH_DATA_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_artifact_dir": evidence["artifact_dir"],
        "base_contract_sha256": base_contract.contract_sha256,
        "research_contract_sha256": research_contract.contract_sha256,
        "rows": len(clean),
        "symbols": symbols,
        "content_sha256": hash_frame(clean, columns),
        "consumed_test_content_sha256": hash_frame(test, columns),
        "research_examples_sha256": _sha256_file(examples_path),
        "duplicate_audit": duplicate_audit,
        "feature_audit": dict(feature_audit),
        "existing_oos_role": "consumed_diagnostic_only",
        "promotion_eligible": False,
        "verdict_influence": False,
    }
    _write_json(output_dir / "data_manifest.json", data_manifest)
    _write_json(output_dir / "split_audit.json", split_audit)
    residual_report = {
        "schema_version": "neutralgrid_live_volatility_residual_drift_v2",
        "evidence_role": "consumed_diagnostic_only",
        "results": [
            {
                "key": report["key"],
                "residual_drift": report["residual_drift"],
            }
            for report in reports
        ],
        "promotion_eligible": False,
        "verdict_influence": False,
    }
    _write_json(output_dir / "residual_drift_report.json", residual_report)
    family_counts: dict[str, int] = {}
    for report in reports:
        selected = cast(Mapping[str, Any], report["selected_candidate"])
        family = str(selected["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    summary = {
        "evaluated_pairs": len(reports),
        "blocked_pairs": len(blockers),
        "selected_family_counts": family_counts,
        "qlike_better_than_base_v1_candidate_pairs": sum(
            float(cast(Mapping[str, Any], report["point_loss"])[
                "candidate_minus_base_v1_candidate"
            ])
            < 0.0
            for report in reports
        ),
        "diagnostic_gate_pass_pairs": sum(
            report.get("passes_consumed_holdout_diagnostic") is True
            for report in reports
        ),
        "promotion_eligible_pairs": 0,
    }
    metadata = {
        "schema_version": RESEARCH_ARTIFACT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "base_contract_sha256": base_contract.contract_sha256,
        "research_contract_sha256": research_contract.contract_sha256,
        "data_manifest_sha256": _sha256_file(output_dir / "data_manifest.json"),
        "research_model_sha256": _sha256_file(model_path),
        "residual_drift_report_sha256": _sha256_file(
            output_dir / "residual_drift_report.json"
        ),
        "promotion_eligible": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
    serialized_root = (
        output_dir if artifact_path_root is None else artifact_path_root.resolve()
    )
    elapsed_seconds = time.monotonic() - started_monotonic
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise VolatilityResearchError("research execution duration is invalid")
    result = {
        "schema_version": RESEARCH_REPORT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "diagnostic_complete_not_promotion_eligible",
        "base_contract": asdict(base_contract) | {"path": str(base_contract.path)},
        "research_contract": asdict(research_contract)
        | {"path": str(research_contract.path)},
        "data_manifest": data_manifest,
        "split_audit": split_audit,
        "summary": summary,
        "execution": {
            "started_at_utc": started_at.astimezone(UTC).isoformat(),
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "clock": "monotonic",
            "measured_through": "before_research_report_serialization",
        },
        "results": reports,
        "blockers": blockers,
        "rejected_hypotheses": [
            {
                "hypothesis": "existing_oos_can_remain_a_promotion_holdout",
                "decision": "rejected",
                "reason": "the existing OOS results informed this research design",
            },
            {
                "hypothesis": "realized_variance_should_be_classified",
                "decision": "rejected",
                "reason": "the approved target is continuous forward realized variance",
            },
            {
                "hypothesis": "jump_features_require_imputation_or_a_live_exception",
                "decision": "rejected",
                "reason": (
                    "HAR-J and HAR-RS-J require complete 5m return windows and "
                    "skip every window touching a gap"
                ),
            },
            {
                "hypothesis": "neural_models_are_required_before_classical_ablation",
                "decision": "rejected",
                "reason": "classical HAR-RS must first prove incremental value",
            },
        ],
        "artifact_paths": {
            "data_manifest": str(serialized_root / "data_manifest.json"),
            "split_audit": str(serialized_root / "split_audit.json"),
            "research_model": str(serialized_root / "research_model.joblib"),
            "metadata": str(serialized_root / "metadata.json"),
            "residual_drift_report": str(
                serialized_root / "residual_drift_report.json"
            ),
            "research_report": str(serialized_root / "research_report.json"),
            "research_examples": str(serialized_root / "research_examples.parquet"),
        },
        "promotion_eligible": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
    _write_json(output_dir / "research_report.json", result)
    metadata["split_audit_sha256"] = _sha256_file(output_dir / "split_audit.json")
    metadata["research_report_sha256"] = _sha256_file(
        output_dir / "research_report.json"
    )
    _write_json(output_dir / "metadata.json", metadata)
    return result


def validate_research_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Fail-closed validate a committed v2 research artifact and its hashes."""

    resolved = artifact_dir.resolve()
    paths = {
        "data_manifest": resolved / "data_manifest.json",
        "metadata": resolved / "metadata.json",
        "research_examples": resolved / "research_examples.parquet",
        "research_model": resolved / "research_model.joblib",
        "research_report": resolved / "research_report.json",
        "residual_drift_report": resolved / "residual_drift_report.json",
        "split_audit": resolved / "split_audit.json",
    }
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise VolatilityResearchError(
            f"research artifact lacks required files: {missing}"
        )
    metadata = _strict_json(paths["metadata"])
    data_manifest = _strict_json(paths["data_manifest"])
    report = _strict_json(paths["research_report"])
    residual_report = _strict_json(paths["residual_drift_report"])
    _strict_json(paths["split_audit"])
    if metadata.get("schema_version") != RESEARCH_ARTIFACT_SCHEMA:
        raise VolatilityResearchError("research metadata schema is invalid")
    if data_manifest.get("schema_version") != RESEARCH_DATA_MANIFEST_SCHEMA:
        raise VolatilityResearchError("research data manifest schema is invalid")
    if report.get("schema_version") != RESEARCH_REPORT_SCHEMA:
        raise VolatilityResearchError("research report schema is invalid")
    if residual_report.get("schema_version") != (
        "neutralgrid_live_volatility_residual_drift_v2"
    ):
        raise VolatilityResearchError("research residual report schema is invalid")
    inert_payloads = (metadata, data_manifest, report, residual_report)
    if any(payload.get("promotion_eligible") is not False for payload in inert_payloads):
        raise VolatilityResearchError("research artifact claims promotion eligibility")
    if any(payload.get("verdict_influence") is not False for payload in inert_payloads):
        raise VolatilityResearchError("research artifact claims verdict influence")
    expected_hashes = {
        "data_manifest_sha256": paths["data_manifest"],
        "research_model_sha256": paths["research_model"],
        "residual_drift_report_sha256": paths["residual_drift_report"],
        "split_audit_sha256": paths["split_audit"],
        "research_report_sha256": paths["research_report"],
    }
    observed_hashes: dict[str, str] = {}
    for field, path in expected_hashes.items():
        observed = _sha256_file(path)
        observed_hashes[field] = observed
        if metadata.get(field) != observed:
            raise VolatilityResearchError(f"research artifact {field} mismatch")
    examples_hash = _sha256_file(paths["research_examples"])
    observed_hashes["research_examples_sha256"] = examples_hash
    if data_manifest.get("research_examples_sha256") != examples_hash:
        raise VolatilityResearchError("research examples SHA-256 mismatch")
    report_manifest = report.get("data_manifest")
    if not isinstance(report_manifest, Mapping) or report_manifest.get(
        "content_sha256"
    ) != data_manifest.get("content_sha256"):
        raise VolatilityResearchError(
            "research report and data manifest content hashes differ"
        )
    try:
        model_payload = joblib.load(paths["research_model"])
    except Exception as exc:
        raise VolatilityResearchError(
            f"cannot load research model payload: {exc}"
        ) from exc
    if not isinstance(model_payload, dict) or model_payload.get(
        "schema_version"
    ) != RESEARCH_ARTIFACT_SCHEMA:
        raise VolatilityResearchError("research model payload schema is invalid")
    if model_payload.get("promotion_eligible") is not False or model_payload.get(
        "verdict_influence"
    ) is not False:
        raise VolatilityResearchError("research model payload is not inert")
    serialized_paths = report.get("artifact_paths")
    if not isinstance(serialized_paths, Mapping):
        raise VolatilityResearchError("research report artifact paths are missing")
    for name, path in paths.items():
        if Path(str(serialized_paths.get(name, ""))).resolve() != path:
            raise VolatilityResearchError(
                f"research report artifact path differs for {name}"
            )
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise VolatilityResearchError("research report summary is missing")
    return {
        "schema_version": "neutralgrid_live_volatility_research_validation_v2",
        "status": "valid",
        "artifact_dir": str(resolved),
        "rows": data_manifest.get("rows"),
        "symbols": data_manifest.get("symbols"),
        "evaluated_pairs": summary.get("evaluated_pairs"),
        "blocked_pairs": summary.get("blocked_pairs"),
        "observed_hashes": observed_hashes,
        "promotion_eligible": False,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
