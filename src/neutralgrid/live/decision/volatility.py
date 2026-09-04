"""Causal realized-volatility contracts, data construction, and statistics.

This module is deliberately independent from live verdict logic.  It validates
finalized one-minute price series, constructs five-minute-sampled forward
realized-variance targets, and implements the frozen QLIKE/HAC/interval rules
used by the shadow volatility forecaster.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd


VOLATILITY_CONTRACT_SCHEMA = "neutralgrid_live_volatility_contract_v1"
VOLATILITY_OUTPUT_SCHEMA = "neutralgrid_shadow_volatility_forecast_v1"
SUPPORTED_HORIZONS: tuple[int, ...] = (30, 60, 180, 360)
HAR_WINDOWS: tuple[int, ...] = (30, 60, 180, 360)
EXPECTED_HAC_LAGS: dict[int, int] = {30: 9, 60: 19, 180: 59, 360: 119}
MINUTE_MS = 60_000


class VolatilityError(RuntimeError):
    """The volatility data, contract, model, or evidence is invalid."""


@dataclass(frozen=True)
class VolatilityContract:
    """Validated immutable volatility research and runtime contract."""

    path: Path
    contract_sha256: str
    primary_series: str
    diagnostic_series: str
    source_interval: str
    sampling_minutes: int
    horizons_minutes: tuple[int, ...]
    issuance_cadence_minutes: int
    minimum_history_days: int
    maximum_source_age_seconds: int
    maximum_roster_age_seconds: int
    fit_fraction: float
    calibration_fraction: float
    prediction_interval_coverage: float
    min_fit_origins: int
    min_calibration_origins: int
    min_test_origins: int
    ridge_alphas: tuple[float, ...]
    ewma_lambdas: tuple[float, ...]

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.fit_fraction - self.calibration_fraction


@dataclass(frozen=True)
class PriceFrameAudit:
    """Evidence produced while validating one price frame."""

    symbol: str
    series_kind: str
    input_rows: int
    output_rows: int
    exact_duplicates: int
    conflicting_duplicates: int
    excluded_non_final: int
    gap_count: int
    missing_minutes: int
    first_open_time_utc: str | None
    last_open_time_utc: str | None


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VolatilityError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except VolatilityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityError(f"cannot read volatility contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VolatilityError("volatility contract root must be an object")
    return payload


def contract_digest(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 excluding the self-referential hash field."""

    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise VolatilityError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise VolatilityError(f"{field} must be finite")
    return number


def _positive_int(value: Any, *, field: str) -> int:
    number = _finite_number(value, field=field)
    integer = int(number)
    if number != integer or integer <= 0:
        raise VolatilityError(f"{field} must be a positive integer")
    return integer


def load_volatility_contract(path: Path) -> VolatilityContract:
    """Load and fail-closed validate the approved versioned contract."""

    resolved = path.resolve()
    payload = _strict_json_object(resolved)
    if payload.get("schema_version") != VOLATILITY_CONTRACT_SCHEMA:
        raise VolatilityError("unsupported volatility contract schema")
    if payload.get("status") != "approved":
        raise VolatilityError("volatility contract status must be approved")
    expected_literals = {
        "primary_series": "mark_kline",
        "diagnostic_series": "last_kline",
        "target_measure": "forward_realized_variance",
        "volatility_transform": "100 * sqrt(realized_variance), non_annualized",
        "implied_volatility_policy": (
            "rejected_no_verified_options_iv_mapping"
        ),
        "source_interval": "1m",
        "missing_value_policy": "exclude_any_window_touching_gap",
        "duplicate_policy": "exact_noop_conflict_reject",
        "point_loss": "patton_qlike",
        "hac_estimator": "newey_west",
        "hac_kernel": "bartlett",
        "hac_lag_rule": "ceil(horizon_minutes / issuance_cadence_minutes) - 1",
        "interval_width_metric": "mean_normalized_interval_width",
        "interval_width_formula": (
            "mean((upper - lower) / calibration_median_actual_rv)"
        ),
        "interval_width_scale_source": "calibration_actual_rv_median",
        "tail_policy": "diagnostic_only_calibration_p90",
        "archive_checksum_policy": "required_sha256_checksum",
        "rest_reconciliation_policy": "archive_to_live_tail_and_audited_gaps_only",
        "non_final_policy": "exclude",
        "split_policy": "chronological_global_60_20_20_with_complete_horizon_purge",
        "statistical_policy": (
            "symbol_horizon_qlike_dm_newey_west_holm_and_interval_gate"
        ),
        "shadow_only_policy": (
            "no_active_scanner_model_sizing_grid_or_execution_influence"
        ),
        "runtime_failure_policy": (
            "explicit_unavailable_never_infer_from_incomplete_evidence"
        ),
        "verdict_influence": False,
    }
    for field, expected in expected_literals.items():
        if payload.get(field) != expected:
            raise VolatilityError(f"unsupported volatility contract {field}")

    declared_digest = payload.get("contract_sha256")
    observed_digest = contract_digest(payload)
    if declared_digest != observed_digest:
        raise VolatilityError(
            "volatility contract SHA-256 mismatch: "
            f"declared={declared_digest!r}, observed={observed_digest}"
        )

    horizons_value = payload.get("horizons_minutes")
    if not isinstance(horizons_value, list):
        raise VolatilityError("horizons_minutes must be a list")
    horizons = tuple(
        _positive_int(value, field="horizons_minutes") for value in horizons_value
    )
    if horizons != SUPPORTED_HORIZONS:
        raise VolatilityError(f"horizons_minutes must equal {SUPPORTED_HORIZONS}")

    sampling = _positive_int(payload.get("sampling_minutes"), field="sampling_minutes")
    cadence = _positive_int(
        payload.get("issuance_cadence_minutes"),
        field="issuance_cadence_minutes",
    )
    if sampling != 5 or cadence != 3:
        raise VolatilityError("only five-minute sampling and three-minute cadence are approved")
    if any(horizon % sampling != 0 for horizon in horizons):
        raise VolatilityError("every horizon must be divisible by sampling_minutes")
    observed_lags = {
        horizon: hac_lag(horizon, cadence) for horizon in horizons
    }
    if observed_lags != EXPECTED_HAC_LAGS:
        raise VolatilityError("contract horizons do not produce approved HAC lags")

    fit_fraction = _finite_number(payload.get("fit_fraction"), field="fit_fraction")
    calibration_fraction = _finite_number(
        payload.get("calibration_fraction"), field="calibration_fraction"
    )
    coverage = _finite_number(
        payload.get("prediction_interval_coverage"),
        field="prediction_interval_coverage",
    )
    if not (0.0 < fit_fraction < 1.0):
        raise VolatilityError("fit_fraction must be between zero and one")
    if not (0.0 < calibration_fraction < 1.0):
        raise VolatilityError("calibration_fraction must be between zero and one")
    if fit_fraction + calibration_fraction >= 1.0:
        raise VolatilityError("fit and calibration fractions must leave a test split")
    if not math.isclose(fit_fraction, 0.60) or not math.isclose(
        calibration_fraction, 0.20
    ):
        raise VolatilityError("only the approved 60/20/20 split is supported")
    if not math.isclose(coverage, 0.90):
        raise VolatilityError("only 90 percent interval coverage is approved")

    origins = payload.get("minimum_non_overlapping_origins")
    if not isinstance(origins, Mapping):
        raise VolatilityError("minimum_non_overlapping_origins must be an object")
    ridge_values = payload.get("ridge_alphas")
    ewma_values = payload.get("ewma_lambdas")
    if not isinstance(ridge_values, list) or not ridge_values:
        raise VolatilityError("ridge_alphas must be a non-empty list")
    if not isinstance(ewma_values, list) or not ewma_values:
        raise VolatilityError("ewma_lambdas must be a non-empty list")
    ridge_alphas = tuple(
        _finite_number(value, field="ridge_alphas") for value in ridge_values
    )
    ewma_lambdas = tuple(
        _finite_number(value, field="ewma_lambdas") for value in ewma_values
    )
    if any(value <= 0 for value in ridge_alphas):
        raise VolatilityError("ridge_alphas must be positive")
    if any(not 0.0 < value < 1.0 for value in ewma_lambdas):
        raise VolatilityError("ewma_lambdas must lie between zero and one")
    freshness = payload.get("freshness_policy")
    if not isinstance(freshness, Mapping):
        raise VolatilityError("freshness_policy must be an object")
    if freshness.get("stale_result") != "unavailable":
        raise VolatilityError("freshness_policy stale_result must be unavailable")
    streams = payload.get("public_event_streams")
    expected_streams = {
        "diff_depth": "@depth@100ms",
        "aggregate_trades": "@aggTrade",
        "mark_price": "@markPrice@1s",
    }
    if streams != expected_streams:
        raise VolatilityError("public_event_streams do not match the approved contract")

    minimum_history_days = _positive_int(
        payload.get("minimum_history_days"), field="minimum_history_days"
    )
    maximum_source_age_seconds = _positive_int(
        freshness.get("maximum_source_age_seconds"),
        field="freshness_policy.maximum_source_age_seconds",
    )
    maximum_roster_age_seconds = _positive_int(
        freshness.get("maximum_roster_age_seconds"),
        field="freshness_policy.maximum_roster_age_seconds",
    )
    min_fit_origins = _positive_int(origins.get("fit"), field="origins.fit")
    min_calibration_origins = _positive_int(
        origins.get("calibration"), field="origins.calibration"
    )
    min_test_origins = _positive_int(origins.get("test"), field="origins.test")
    if minimum_history_days != 90:
        raise VolatilityError("minimum_history_days must equal the approved 90 days")
    if maximum_source_age_seconds != 180:
        raise VolatilityError(
            "maximum_source_age_seconds must equal the approved 180 seconds"
        )
    if (min_fit_origins, min_calibration_origins, min_test_origins) != (
        216,
        72,
        72,
    ):
        raise VolatilityError(
            "minimum_non_overlapping_origins must equal approved 216/72/72"
        )

    return VolatilityContract(
        path=resolved,
        contract_sha256=observed_digest,
        primary_series="mark_kline",
        diagnostic_series="last_kline",
        source_interval="1m",
        sampling_minutes=sampling,
        horizons_minutes=horizons,
        issuance_cadence_minutes=cadence,
        minimum_history_days=minimum_history_days,
        maximum_source_age_seconds=maximum_source_age_seconds,
        maximum_roster_age_seconds=maximum_roster_age_seconds,
        fit_fraction=fit_fraction,
        calibration_fraction=calibration_fraction,
        prediction_interval_coverage=coverage,
        min_fit_origins=min_fit_origins,
        min_calibration_origins=min_calibration_origins,
        min_test_origins=min_test_origins,
        ridge_alphas=ridge_alphas,
        ewma_lambdas=ewma_lambdas,
    )


def _open_time_series(frame: pd.DataFrame) -> pd.Series:
    if "open_time_ms" in frame.columns:
        numeric = cast(
            pd.Series,
            pd.to_numeric(frame["open_time_ms"], errors="coerce"),
        )
        return cast(pd.Series, pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce"))
    if "open_time" in frame.columns:
        return cast(pd.Series, pd.to_datetime(frame["open_time"], utc=True, errors="coerce"))
    raise VolatilityError("price frame lacks open_time_ms/open_time")


def validate_price_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    series_kind: str,
) -> tuple[pd.DataFrame, PriceFrameAudit]:
    """Validate finalized one-minute closes and reject conflicting duplicates."""

    if not isinstance(frame, pd.DataFrame):
        raise VolatilityError("price frame must be a DataFrame")
    if "close" not in frame.columns:
        raise VolatilityError("price frame lacks close")
    work = frame.copy()
    input_rows = len(work)
    work["open_time"] = _open_time_series(work)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    if "is_final" in work.columns:
        final = cast(pd.Series, work["is_final"]).map(
            lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
        )
        excluded_non_final = int((~final).sum())
        work = work.loc[final].copy()
    else:
        excluded_non_final = 0
    if bool(cast(pd.Series, work["open_time"]).isna().any()):
        raise VolatilityError(f"{symbol}/{series_kind}: invalid open_time")
    closes = cast(pd.Series, pd.to_numeric(work["close"], errors="coerce"))
    close_values = closes.to_numpy(dtype=float, na_value=np.nan)
    if not bool(np.isfinite(close_values).all()) or bool((close_values <= 0).any()):
        raise VolatilityError(f"{symbol}/{series_kind}: close must be positive and finite")

    work = work.reset_index(drop=True)
    exact_duplicates = 0
    conflicting_duplicates = 0
    retained: list[int] = []
    compare_columns = [
        column
        for column in ("open", "high", "low", "close", "volume", "close_time_ms", "is_final")
        if column in work.columns
    ]
    for _, group in work.groupby("open_time", sort=False, dropna=False):
        if len(group) == 1:
            retained.append(int(cast(Any, group.index[0])))
            continue
        first = group.iloc[0]
        for row_index in range(1, len(group)):
            current = group.iloc[row_index]
            identical = True
            for column in compare_columns:
                left = first[column]
                right = current[column]
                if pd.isna(left) and pd.isna(right):
                    continue
                if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
                    right, (int, float, np.integer, np.floating)
                ):
                    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0):
                        identical = False
                        break
                elif left != right:
                    identical = False
                    break
            if identical:
                exact_duplicates += 1
            else:
                conflicting_duplicates += 1
        retained.append(int(cast(Any, group.index[0])))
    if conflicting_duplicates:
        raise VolatilityError(
            f"{symbol}/{series_kind}: {conflicting_duplicates} conflicting duplicate candles"
        )

    work = work.loc[retained].copy()
    times = cast(pd.Series, work["open_time"])
    if bool(times.duplicated().any()) or not bool(times.is_monotonic_increasing):
        raise VolatilityError(f"{symbol}/{series_kind}: open_time is not unique and monotonic")
    work = work.reset_index(drop=True)
    times = cast(pd.Series, work["open_time"])
    diffs = cast(pd.Series, times.diff().dt.total_seconds().dropna())
    if bool((diffs <= 0).any()):
        raise VolatilityError(f"{symbol}/{series_kind}: non-increasing open_time")
    gaps = diffs.loc[diffs > 60.0]
    missing_minutes = int(sum(max(0, int(round(float(value) / 60.0)) - 1) for value in gaps))
    normalized = work.loc[:, ["open_time", "close"]].copy()
    normalized["open_time_ms"] = (
        cast(pd.Series, normalized["open_time"]).astype("int64") // 1_000_000
    )
    normalized["is_final"] = True
    normalized = normalized.loc[:, ["open_time_ms", "open_time", "close", "is_final"]]
    first_time = None if normalized.empty else cast(pd.Timestamp, normalized["open_time"].iloc[0]).isoformat()
    last_time = None if normalized.empty else cast(pd.Timestamp, normalized["open_time"].iloc[-1]).isoformat()
    audit = PriceFrameAudit(
        symbol=symbol.upper(),
        series_kind=series_kind,
        input_rows=input_rows,
        output_rows=len(normalized),
        exact_duplicates=exact_duplicates,
        conflicting_duplicates=conflicting_duplicates,
        excluded_non_final=excluded_non_final,
        gap_count=len(gaps),
        missing_minutes=missing_minutes,
        first_open_time_utc=first_time,
        last_open_time_utc=last_time,
    )
    return normalized, audit


def _rv_from_array(
    log_prices: np.ndarray,
    *,
    origin_index: int,
    window_minutes: int,
    sampling_minutes: int,
    forward: bool,
) -> float | None:
    if window_minutes % sampling_minutes:
        raise VolatilityError("RV window is not divisible by sampling interval")
    if forward:
        indexes = np.arange(
            origin_index,
            origin_index + window_minutes + 1,
            sampling_minutes,
            dtype=int,
        )
        full_start = origin_index
        full_end = origin_index + window_minutes
    else:
        indexes = np.arange(
            origin_index - window_minutes,
            origin_index + 1,
            sampling_minutes,
            dtype=int,
        )
        full_start = origin_index - window_minutes
        full_end = origin_index
    if full_start < 0 or full_end >= len(log_prices):
        return None
    full_window = log_prices[full_start : full_end + 1]
    if len(full_window) != window_minutes + 1 or not bool(np.isfinite(full_window).all()):
        return None
    sampled = log_prices[indexes]
    if not bool(np.isfinite(sampled).all()):
        return None
    returns = np.diff(sampled)
    return float(np.dot(returns, returns))


def _sqrt_time_baseline(
    log_prices: np.ndarray,
    *,
    origin_index: int,
    horizon_minutes: int,
) -> float | None:
    estimation_minutes = 48 * 60
    start = origin_index - estimation_minutes
    if start < 0:
        return None
    full = log_prices[start : origin_index + 1]
    if len(full) != estimation_minutes + 1 or not bool(np.isfinite(full).all()):
        return None
    sampled = full[::15]
    returns = np.diff(sampled)
    if len(returns) < 20:
        return None
    sigma = float(np.std(returns))
    forecast = sigma * sigma * max(float(horizon_minutes) / 15.0, 1.0)
    return forecast if math.isfinite(forecast) and forecast > 0.0 else None


def _ewma_baseline(
    log_prices: np.ndarray,
    *,
    origin_index: int,
    horizon_minutes: int,
    sampling_minutes: int,
    decay: float,
) -> float | None:
    lookback_minutes = 48 * 60
    start = origin_index - lookback_minutes
    if start < 0:
        return None
    full = log_prices[start : origin_index + 1]
    if len(full) != lookback_minutes + 1 or not bool(np.isfinite(full).all()):
        return None
    sampled = full[::sampling_minutes]
    squared = np.square(np.diff(sampled))
    if len(squared) < 20:
        return None
    variance = float(squared[0])
    for value in squared[1:]:
        variance = decay * variance + (1.0 - decay) * float(value)
    forecast = variance * (float(horizon_minutes) / sampling_minutes)
    return forecast if math.isfinite(forecast) and forecast > 0.0 else None


def build_rv_examples(
    frame: pd.DataFrame,
    *,
    symbol: str,
    contract: VolatilityContract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build causal HAR features and forward RV labels at three-minute origins."""

    normalized, audit = validate_price_frame(
        frame, symbol=symbol, series_kind=contract.primary_series
    )
    if normalized.empty:
        return pd.DataFrame(), {"price_audit": audit.__dict__, "excluded_origins": 0}
    first = cast(pd.Timestamp, normalized["open_time"].iloc[0]).floor("min")
    last = cast(pd.Timestamp, normalized["open_time"].iloc[-1]).floor("min")
    grid = pd.date_range(first, last, freq="1min", tz="UTC")
    close_series = pd.Series(
        np.asarray(normalized["close"], dtype=float),
        index=pd.DatetimeIndex(normalized["open_time"]),
        dtype=float,
    )
    reindexed = close_series.reindex(grid)
    prices = reindexed.to_numpy(dtype=float, na_value=np.nan)
    log_prices = np.full_like(prices, np.nan, dtype=float)
    valid_prices = np.isfinite(prices) & (prices > 0.0)
    log_prices[valid_prices] = np.log(prices[valid_prices])

    max_lookback = 48 * 60
    max_horizon = max(contract.horizons_minutes)
    first_index = max_lookback
    last_index = len(grid) - max_horizon - 1
    rows: list[dict[str, Any]] = []
    excluded = 0
    for origin_index in range(
        first_index,
        max(first_index, last_index + 1),
        contract.issuance_cadence_minutes,
    ):
        features: dict[str, float] = {}
        feature_complete = True
        for window in HAR_WINDOWS:
            value = _rv_from_array(
                log_prices,
                origin_index=origin_index,
                window_minutes=window,
                sampling_minutes=contract.sampling_minutes,
                forward=False,
            )
            if value is None:
                feature_complete = False
                break
            features[f"rv_{window}"] = value
        if not feature_complete:
            excluded += len(contract.horizons_minutes)
            continue
        origin = grid[origin_index]
        for horizon in contract.horizons_minutes:
            target = _rv_from_array(
                log_prices,
                origin_index=origin_index,
                window_minutes=horizon,
                sampling_minutes=contract.sampling_minutes,
                forward=True,
            )
            persistence = features[f"rv_{horizon}"]
            sqrt_time = _sqrt_time_baseline(
                log_prices,
                origin_index=origin_index,
                horizon_minutes=horizon,
            )
            ewma_values = {
                decay: _ewma_baseline(
                    log_prices,
                    origin_index=origin_index,
                    horizon_minutes=horizon,
                    sampling_minutes=contract.sampling_minutes,
                    decay=decay,
                )
                for decay in contract.ewma_lambdas
            }
            if (
                target is None
                or not math.isfinite(persistence)
                or persistence <= 0.0
                or sqrt_time is None
                or any(value is None for value in ewma_values.values())
            ):
                excluded += 1
                continue
            row: dict[str, Any] = {
                "symbol": symbol.upper(),
                "origin_utc": origin,
                "label_end_utc": origin + pd.Timedelta(minutes=horizon),
                "horizon_minutes": horizon,
                "target_rv": target,
                "target_volatility_pct": 100.0 * math.sqrt(target),
                "persistence_forecast_rv": persistence,
                "sqrt_time_forecast_rv": sqrt_time,
                **features,
            }
            for window, value in (
                (window, features[f"rv_{window}"]) for window in HAR_WINDOWS
            ):
                row[f"volatility_pct_{window}"] = 100.0 * math.sqrt(value)
            for decay, value in ewma_values.items():
                row[f"ewma_{decay:g}_forecast_rv"] = float(cast(float, value))
            rows.append(row)
    examples = pd.DataFrame(rows)
    if not examples.empty:
        examples = examples.sort_values(
            ["origin_utc", "symbol", "horizon_minutes"]
        ).reset_index(drop=True)
    span_days = (last - first).total_seconds() / 86_400.0 if last >= first else 0.0
    return examples, {
        "price_audit": audit.__dict__,
        "calendar_span_days": span_days,
        "eligible_examples": len(examples),
        "excluded_examples": excluded,
    }


def latest_rv_snapshot(
    mark_frame: pd.DataFrame,
    last_frame: pd.DataFrame | None,
    *,
    symbol: str,
    contract: VolatilityContract,
    asof_utc: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Return causal trailing HAR features and mark/last RV diagnostics."""

    mark, mark_audit = validate_price_frame(
        mark_frame,
        symbol=symbol,
        series_kind=contract.primary_series,
    )
    if mark.empty:
        raise VolatilityError(f"{symbol}: no valid finalized mark candles")
    requested_asof = pd.Timestamp.now(tz="UTC") if asof_utc is None else asof_utc
    if requested_asof.tzinfo is None:
        requested_asof = requested_asof.tz_localize("UTC")
    else:
        requested_asof = requested_asof.tz_convert("UTC")
    eligible_mark = mark.loc[
        cast(pd.Series, mark["open_time"]) <= requested_asof.floor("min")
    ].copy()
    if eligible_mark.empty:
        raise VolatilityError(f"{symbol}: mark candles are later than requested as-of")
    origin = cast(pd.Timestamp, eligible_mark["open_time"].iloc[-1]).floor("min")
    start = origin - pd.Timedelta(minutes=48 * 60)
    grid = pd.date_range(start, origin, freq="1min", tz="UTC")
    mark_series = pd.Series(
        np.asarray(eligible_mark["close"], dtype=float),
        index=pd.DatetimeIndex(eligible_mark["open_time"]),
        dtype=float,
    ).reindex(grid)
    mark_prices = mark_series.to_numpy(dtype=float, na_value=np.nan)
    mark_logs = np.full_like(mark_prices, np.nan, dtype=float)
    mark_valid = np.isfinite(mark_prices) & (mark_prices > 0.0)
    mark_logs[mark_valid] = np.log(mark_prices[mark_valid])
    origin_index = len(grid) - 1
    features: dict[str, float] = {}
    mark_rv: dict[str, float] = {}
    for horizon in contract.horizons_minutes:
        value = _rv_from_array(
            mark_logs,
            origin_index=origin_index,
            window_minutes=horizon,
            sampling_minutes=contract.sampling_minutes,
            forward=False,
        )
        if value is None:
            raise VolatilityError(
                f"{symbol}: mark feature window {horizon}m touches a gap"
            )
        features[f"rv_{horizon}"] = value
        mark_rv[str(horizon)] = value

    last_rv: dict[str, float | None] = {
        str(horizon): None for horizon in contract.horizons_minutes
    }
    last_audit: dict[str, Any] | None = None
    if last_frame is not None and not last_frame.empty:
        last, audited_last = validate_price_frame(
            last_frame,
            symbol=symbol,
            series_kind=contract.diagnostic_series,
        )
        last_audit = audited_last.__dict__
        last_series = pd.Series(
            np.asarray(last["close"], dtype=float),
            index=pd.DatetimeIndex(last["open_time"]),
            dtype=float,
        ).reindex(grid)
        last_prices = last_series.to_numpy(dtype=float, na_value=np.nan)
        last_logs = np.full_like(last_prices, np.nan, dtype=float)
        last_valid = np.isfinite(last_prices) & (last_prices > 0.0)
        last_logs[last_valid] = np.log(last_prices[last_valid])
        for horizon in contract.horizons_minutes:
            last_rv[str(horizon)] = _rv_from_array(
                last_logs,
                origin_index=origin_index,
                window_minutes=horizon,
                sampling_minutes=contract.sampling_minutes,
                forward=False,
            )
    divergence = {
        key: (
            None
            if last_rv[key] is None
            else float(mark_rv[key] - cast(float, last_rv[key]))
        )
        for key in mark_rv
    }
    mark_volatility_pct = {
        key: 100.0 * math.sqrt(value) for key, value in mark_rv.items()
    }
    last_volatility_pct = {
        key: None if value is None else 100.0 * math.sqrt(value)
        for key, value in last_rv.items()
    }
    return {
        "symbol": symbol.upper(),
        "cutoff_utc": origin.isoformat(),
        "freshness_seconds": max(
            0.0,
            float((requested_asof - origin).total_seconds()),
        ),
        "features": features,
        "mark_rv": mark_rv,
        "last_rv": last_rv,
        "mark_volatility_pct": mark_volatility_pct,
        "last_volatility_pct": last_volatility_pct,
        "mark_last_rv_divergence": divergence,
        "mark_price_audit": mark_audit.__dict__,
        "last_price_audit": last_audit,
    }


def qlike_loss(actual_rv: Any, forecast_rv: Any) -> np.ndarray:
    """Patton-compatible QLIKE, defined for zero actual and positive forecast."""

    actual = np.asarray(actual_rv, dtype=float)
    forecast = np.asarray(forecast_rv, dtype=float)
    if actual.shape != forecast.shape:
        raise VolatilityError("QLIKE actual and forecast shapes differ")
    if not bool(np.isfinite(actual).all()) or bool((actual < 0.0).any()):
        raise VolatilityError("QLIKE actual RV must be finite and non-negative")
    if not bool(np.isfinite(forecast).all()) or bool((forecast <= 0.0).any()):
        raise VolatilityError("QLIKE forecast RV must be finite and positive")
    return np.log(forecast) + actual / forecast


def hac_lag(horizon_minutes: int, issuance_cadence_minutes: int = 3) -> int:
    """Frozen overlap-aware HAC lag: ceil(horizon/cadence) - 1."""

    if horizon_minutes <= 0 or issuance_cadence_minutes <= 0:
        raise VolatilityError("horizon and cadence must be positive")
    return int(math.ceil(horizon_minutes / issuance_cadence_minutes) - 1)


def newey_west_mean_test(
    loss_differential: Any,
    *,
    horizon_minutes: int,
    issuance_cadence_minutes: int = 3,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """One-sided mean-loss test using Bartlett/Newey-West covariance."""

    values = np.asarray(loss_differential, dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise VolatilityError("loss differential must be a finite one-dimensional array")
    lag = hac_lag(horizon_minutes, issuance_cadence_minutes)
    count = len(values)
    if count <= lag + 1:
        raise VolatilityError(
            f"insufficient observations for HAC: T={count}, lag={lag}"
        )
    mean = float(np.mean(values))
    centered = values - mean
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        weight = 1.0 - float(offset) / float(lag + 1)
        long_run_variance += 2.0 * weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        raise VolatilityError("Newey-West long-run variance is non-positive or non-finite")
    standard_error = math.sqrt(long_run_variance / count)
    statistic = mean / standard_error
    normal = NormalDist()
    critical = normal.inv_cdf(confidence)
    upper = mean + critical * standard_error
    one_sided_p = normal.cdf(statistic)
    return {
        "hac_estimator": "newey_west",
        "hac_kernel": "bartlett",
        "hac_lag_rule": "ceil(horizon_minutes / issuance_cadence_minutes) - 1",
        "hac_lag": lag,
        "observation_count": count,
        "mean_loss_differential": mean,
        "long_run_variance": long_run_variance,
        "standard_error": standard_error,
        "test_statistic": statistic,
        "one_sided_p_value": one_sided_p,
        "one_sided_upper_confidence_bound": upper,
        "confidence": confidence,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, valid under arbitrary dependence."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not bool(np.isfinite(values).all()):
        raise VolatilityError("Holm p-values must be finite and one-dimensional")
    if bool(((values < 0.0) | (values > 1.0)).any()):
        raise VolatilityError("Holm p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(total - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return [float(value) for value in adjusted]


def wilson_interval(successes: int, count: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for empirical prediction-interval coverage."""

    if count <= 0 or successes < 0 or successes > count:
        raise VolatilityError("invalid Wilson successes/count")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def calibrate_log_interval(
    actual_rv: Any,
    forecast_rv: Any,
    *,
    rv_floor: float,
    coverage: float = 0.90,
) -> dict[str, Any]:
    """Calibrate asymmetric signed log-residual quantiles and a shared RV scale."""

    actual = np.asarray(actual_rv, dtype=float)
    forecast = np.asarray(forecast_rv, dtype=float)
    if len(actual) == 0 or actual.shape != forecast.shape:
        raise VolatilityError("interval calibration arrays are empty or misaligned")
    if not bool(np.isfinite(actual).all()) or bool((actual < 0.0).any()):
        raise VolatilityError("calibration actual RV must be finite and non-negative")
    scale = float(np.median(actual))
    if not math.isfinite(scale) or scale <= 0.0:
        raise VolatilityError("calibration median actual RV is non-positive or non-finite")
    if not math.isfinite(rv_floor) or rv_floor <= 0.0:
        raise VolatilityError("development-frozen interval RV floor is invalid")
    safe_actual = np.maximum(actual, rv_floor)
    if not bool(np.isfinite(forecast).all()) or bool((forecast <= 0.0).any()):
        raise VolatilityError("interval calibration forecast must be finite and positive")
    residuals = np.log(safe_actual) - np.log(forecast)
    alpha = 1.0 - coverage
    lower_q = float(np.quantile(residuals, alpha / 2.0, method="linear"))
    upper_q = float(np.quantile(residuals, 1.0 - alpha / 2.0, method="linear"))
    return {
        "coverage": coverage,
        "lower_log_residual_quantile": lower_q,
        "upper_log_residual_quantile": upper_q,
        "rv_floor": rv_floor,
        "rv_floor_source": "development_positive_actual_rv_min_half",
        "interval_width_metric": "mean_normalized_interval_width",
        "interval_width_formula": (
            "mean((upper - lower) / calibration_median_actual_rv)"
        ),
        "interval_width_scale_source": "calibration_actual_rv_median",
        "interval_width_scale": scale,
        "calibration_count": len(actual),
    }


def apply_log_interval(
    forecast_rv: Any,
    calibration: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen asymmetric multiplicative interval factors."""

    forecast = np.asarray(forecast_rv, dtype=float)
    if not bool(np.isfinite(forecast).all()) or bool((forecast <= 0.0).any()):
        raise VolatilityError("interval forecast must be finite and positive")
    lower_q = _finite_number(
        calibration.get("lower_log_residual_quantile"), field="interval.lower_quantile"
    )
    upper_q = _finite_number(
        calibration.get("upper_log_residual_quantile"), field="interval.upper_quantile"
    )
    lower = np.maximum(0.0, forecast * math.exp(lower_q))
    upper = np.maximum(lower, forecast * math.exp(upper_q))
    return lower, upper


def interval_metrics(
    actual_rv: Any,
    lower_rv: Any,
    upper_rv: Any,
    *,
    scale: float,
    target_coverage: float = 0.90,
) -> dict[str, Any]:
    """Evaluate Wilson coverage and calibration-frozen normalized width."""

    actual = np.asarray(actual_rv, dtype=float)
    lower = np.asarray(lower_rv, dtype=float)
    upper = np.asarray(upper_rv, dtype=float)
    if actual.shape != lower.shape or actual.shape != upper.shape or len(actual) == 0:
        raise VolatilityError("interval metric arrays are empty or misaligned")
    if not math.isfinite(scale) or scale <= 0.0:
        raise VolatilityError("interval width scale must be positive and finite")
    if not bool(np.isfinite(actual).all()) or bool((actual < 0.0).any()):
        raise VolatilityError("interval actual RV must be finite and non-negative")
    if not bool(np.isfinite(lower).all()) or not bool(np.isfinite(upper).all()):
        raise VolatilityError("interval bounds must be finite")
    if bool((lower < 0.0).any()) or bool((upper < lower).any()):
        raise VolatilityError("interval bounds are invalid")
    covered = (actual >= lower) & (actual <= upper)
    successes = int(covered.sum())
    empirical = successes / len(actual)
    wilson_low, wilson_high = wilson_interval(successes, len(actual))
    width = float(np.mean((upper - lower) / scale))
    return {
        "target_coverage": target_coverage,
        "empirical_coverage": empirical,
        "absolute_coverage_error": abs(empirical - target_coverage),
        "wilson_lower": wilson_low,
        "wilson_upper": wilson_high,
        "wilson_contains_target": wilson_low <= target_coverage <= wilson_high,
        "mean_normalized_interval_width": width,
        "interval_width_scale": scale,
        "observation_count": len(actual),
    }


def count_non_overlapping_origins(
    origins: Sequence[Any],
    *,
    horizon_minutes: int,
) -> int:
    """Greedily count chronologically non-overlapping forecast origins."""

    ordered = sorted(pd.Timestamp(value) for value in origins)
    count = 0
    next_allowed: pd.Timestamp | None = None
    horizon = pd.Timedelta(minutes=horizon_minutes)
    for origin in ordered:
        if next_allowed is None or origin >= next_allowed:
            count += 1
            candidate = origin + horizon
            if bool(pd.isna(candidate)):
                raise VolatilityError("non-overlapping origin arithmetic produced NaT")
            next_allowed = cast(pd.Timestamp, candidate)
    return count


def hash_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Canonical content hash for selected data-manifest columns."""

    selected = frame.loc[:, list(columns)].copy()
    for column in selected.columns:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = cast(pd.Series, selected[column]).astype("int64")
    encoded = selected.to_csv(
        index=False,
        lineterminator="\n",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_price_store_frame(
    price_store_root: Path,
    *,
    symbol: str,
    series_kind: str,
    interval: str = "1m",
) -> pd.DataFrame:
    """Load the canonical daily PriceSeries Parquet files for one key."""

    directory = price_store_root / symbol.upper() / series_kind / interval
    paths = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, ValueError) as exc:
            raise VolatilityError(f"cannot read price-store file {path}: {exc}") from exc
    return pd.concat(frames, ignore_index=True)


def audit_single_symbol_readiness(
    examples: pd.DataFrame,
    *,
    symbol: str,
    contract: VolatilityContract,
) -> dict[str, Any]:
    """Audit mechanical 60/20/20 non-overlapping origin floors for acquisition."""

    selected = examples.loc[
        (cast(pd.Series, examples.get("symbol", pd.Series(dtype=str))).astype(str) == symbol.upper())
        & (pd.to_numeric(examples.get("horizon_minutes"), errors="coerce") == 360)
    ].copy()
    if selected.empty:
        return {
            "status": "blocked",
            "symbol": symbol.upper(),
            "reason": "no_360_minute_examples",
            "counts": {"fit": 0, "calibration": 0, "test": 0},
        }
    origins = pd.DatetimeIndex(
        sorted(cast(pd.Series, selected["origin_utc"]).unique())
    )
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
        return {
            "status": "blocked",
            "symbol": symbol.upper(),
            "reason": "split_not_feasible",
            "counts": {"fit": 0, "calibration": 0, "test": 0},
        }
    fit_cutoff = cast(pd.Timestamp, cast(Any, origins[fit_index - 1]))
    calibration_cutoff = cast(pd.Timestamp, cast(Any, origins[calibration_index - 1]))
    fit = selected.loc[
        (cast(pd.Series, selected["origin_utc"]) <= fit_cutoff)
        & (cast(pd.Series, selected["label_end_utc"]) <= fit_cutoff)
    ]
    calibration = selected.loc[
        (cast(pd.Series, selected["origin_utc"]) > fit_cutoff)
        & (cast(pd.Series, selected["origin_utc"]) <= calibration_cutoff)
        & (cast(pd.Series, selected["label_end_utc"]) <= calibration_cutoff)
    ]
    test = selected.loc[
        cast(pd.Series, selected["origin_utc"]) > calibration_cutoff
    ]
    counts = {
        "fit": count_non_overlapping_origins(
            cast(pd.Series, fit["origin_utc"]).tolist(), horizon_minutes=360
        ),
        "calibration": count_non_overlapping_origins(
            cast(pd.Series, calibration["origin_utc"]).tolist(), horizon_minutes=360
        ),
        "test": count_non_overlapping_origins(
            cast(pd.Series, test["origin_utc"]).tolist(), horizon_minutes=360
        ),
    }
    required = {
        "fit": contract.min_fit_origins,
        "calibration": contract.min_calibration_origins,
        "test": contract.min_test_origins,
    }
    ready = all(counts[key] >= required[key] for key in required)
    return {
        "status": "ready" if ready else "blocked",
        "symbol": symbol.upper(),
        "counts": counts,
        "required": required,
        "fit_cutoff_utc": fit_cutoff.isoformat(),
        "calibration_cutoff_utc": calibration_cutoff.isoformat(),
    }
