"""Governed v2 research for semivariance and jump volatility challengers.

This module is intentionally separate from the v1 runtime forecaster.  It may
read a checksum-verified v1 research artifact, but every output remains
diagnostic-only and cannot be loaded by the live volatility runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.live.decision.volatility import (
    HAR_WINDOWS,
    VolatilityContract,
    contract_digest,
    load_price_store_frame,
    validate_price_frame,
)


VOLATILITY_RESEARCH_CONTRACT_SCHEMA = (
    "neutralgrid_live_volatility_research_contract_v2"
)
RESEARCH_CANDIDATE_FAMILIES: tuple[str, ...] = (
    "har",
    "har_rs",
    "har_j",
    "har_rs_j",
)
RESEARCH_KEY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "origin_utc",
    "horizon_minutes",
)
SEMIVARIANCE_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    name
    for window in HAR_WINDOWS
    for name in (f"rs_pos_{window}", f"rs_neg_{window}")
)
JUMP_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"jump_{window}" for window in HAR_WINDOWS
)
RESEARCH_FEATURE_COLUMNS: tuple[str, ...] = (
    *SEMIVARIANCE_FEATURE_COLUMNS,
    *JUMP_FEATURE_COLUMNS,
)


class VolatilityResearchError(RuntimeError):
    """The diagnostic research contract, data, or artifact is invalid."""


@dataclass(frozen=True)
class VolatilityResearchContract:
    """Strict immutable contract for the diagnostic-only v2 research stage."""

    path: Path
    contract_sha256: str
    base_contract_sha256: str
    candidate_families: tuple[str, ...]
    semivariance_identity_relative_tolerance: float
    semivariance_identity_absolute_tolerance: float
    jump_estimator: str
    jump_minimum_return_count_policy: str


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise VolatilityResearchError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except VolatilityResearchError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VolatilityResearchError(
            f"cannot read volatility research contract {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise VolatilityResearchError("volatility research contract must be an object")
    return payload


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise VolatilityResearchError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VolatilityResearchError(f"{field} must be finite and non-negative")
    return result


def load_volatility_research_contract(
    path: Path,
    *,
    base_contract: VolatilityContract,
) -> VolatilityResearchContract:
    """Load the research contract and bind it to the exact v1 contract hash."""

    resolved = path.resolve()
    payload = _strict_json_object(resolved)
    if payload.get("schema_version") != VOLATILITY_RESEARCH_CONTRACT_SCHEMA:
        raise VolatilityResearchError("unsupported volatility research schema")
    expected_literals: dict[str, Any] = {
        "status": "research_only",
        "base_contract_schema": "neutralgrid_live_volatility_contract_v1",
        "research_stage": "classical_semivariance_jump_ablation",
        "candidate_selection": (
            "independent_per_symbol_horizon_development_only"
        ),
        "ridge_grid_source": "base_contract",
        "sampling_minutes_source": "base_contract",
        "horizons_source": "base_contract",
        "semivariance_positive_formula": (
            "sum(r_5m^2 * indicator(r_5m >= 0))"
        ),
        "semivariance_negative_formula": (
            "sum(r_5m^2 * indicator(r_5m < 0))"
        ),
        "semivariance_identity": "rs_pos_h + rs_neg_h == rv_h",
        "duplicate_policy": (
            "exact_noop_conflict_reject_on_symbol_origin_horizon"
        ),
        "target_classification_policy": "regression_only_no_rv_classifier",
        "row_classification_policy": "admitted_unique_or_rejected_conflict",
        "jump_estimator": (
            "bipower_variation_mu1_inverse_square_sum_adjacent_abs_5m_returns"
        ),
        "jump_formula": "max(rv_h - bv_h, 0)",
        "jump_sampling": "5m",
        "jump_minimum_return_count_policy": (
            "full_window_5m_return_count_required"
        ),
        "jump_gap_policy": "exclude_any_window_touching_gap",
        "neural_policy": (
            "deferred_until_classical_challenger_passes_new_untouched_holdout"
        ),
        "existing_oos_role": "consumed_diagnostic_only",
        "promotion_holdout_policy": (
            "future_only_after_research_contract_freeze"
        ),
        "promotion_eligible": False,
        "shadow_only": True,
        "verdict_influence": False,
        "runtime_effect": "none",
    }
    for field, expected in expected_literals.items():
        if payload.get(field) != expected:
            raise VolatilityResearchError(
                f"unsupported volatility research contract field {field}"
            )

    declared_base_hash = payload.get("base_contract_sha256")
    if declared_base_hash != base_contract.contract_sha256:
        raise VolatilityResearchError(
            "research contract is not bound to the supplied v1 contract"
        )
    candidates = payload.get("candidate_families")
    if not isinstance(candidates, list) or tuple(candidates) != (
        RESEARCH_CANDIDATE_FAMILIES
    ):
        raise VolatilityResearchError(
            f"candidate_families must equal {RESEARCH_CANDIDATE_FAMILIES}"
        )
    relative_tolerance = _finite_nonnegative(
        payload.get("semivariance_identity_relative_tolerance"),
        field="semivariance_identity_relative_tolerance",
    )
    absolute_tolerance = _finite_nonnegative(
        payload.get("semivariance_identity_absolute_tolerance"),
        field="semivariance_identity_absolute_tolerance",
    )
    if relative_tolerance != 1e-12 or absolute_tolerance != 1e-18:
        raise VolatilityResearchError(
            "semivariance identity tolerances differ from the approved values"
        )
    observed_hash = contract_digest(payload)
    if payload.get("contract_sha256") != observed_hash:
        raise VolatilityResearchError(
            "volatility research contract SHA-256 mismatch"
        )
    return VolatilityResearchContract(
        path=resolved,
        contract_sha256=observed_hash,
        base_contract_sha256=base_contract.contract_sha256,
        candidate_families=RESEARCH_CANDIDATE_FAMILIES,
        semivariance_identity_relative_tolerance=relative_tolerance,
        semivariance_identity_absolute_tolerance=absolute_tolerance,
        jump_estimator=str(payload["jump_estimator"]),
        jump_minimum_return_count_policy=str(
            payload["jump_minimum_return_count_policy"]
        ),
    )


def _rows_equal(group: pd.DataFrame, columns: Sequence[str]) -> bool:
    first = group.iloc[0]
    for column in columns:
        values = cast(pd.Series, group[column])
        expected = first[column]
        equal = values.eq(expected) | (values.isna() & pd.isna(expected))
        if not bool(equal.all()):
            return False
    return True


def deduplicate_research_examples(
    examples: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """No-op exact duplicates and reject conflicting research identities."""

    missing = sorted(set(RESEARCH_KEY_COLUMNS) - set(examples.columns))
    if missing:
        raise VolatilityResearchError(
            f"research examples lack identity columns: {missing}"
        )
    frame = examples.copy()
    frame["symbol"] = cast(pd.Series, frame["symbol"]).astype(str).str.upper()
    frame["origin_utc"] = pd.to_datetime(
        frame["origin_utc"], utc=True, errors="coerce"
    )
    horizons = cast(
        pd.Series,
        pd.to_numeric(
            cast(pd.Series, frame["horizon_minutes"]),
            errors="coerce",
        ),
    )
    if bool(cast(pd.Series, frame["origin_utc"]).isna().any()) or bool(
        horizons.isna().any()
    ):
        raise VolatilityResearchError("research identities contain invalid values")
    horizon_values = np.asarray(horizons, dtype=float)
    if bool((horizon_values <= 0).any()) or bool(
        (horizon_values % 1 != 0).any()
    ):
        raise VolatilityResearchError("research horizons must be positive integers")
    frame["horizon_minutes"] = horizon_values.astype(int)
    if bool((cast(pd.Series, frame["symbol"]).str.len() == 0).any()):
        raise VolatilityResearchError("research symbol cannot be empty")

    duplicate_mask = frame.duplicated(list(RESEARCH_KEY_COLUMNS), keep=False)
    duplicate_rows = frame.loc[duplicate_mask]
    exact_duplicate_rows = 0
    conflicting_keys = 0
    if not duplicate_rows.empty:
        compare_columns = list(frame.columns)
        for key, group in duplicate_rows.groupby(
            list(RESEARCH_KEY_COLUMNS), sort=False, dropna=False
        ):
            if not _rows_equal(group, compare_columns):
                conflicting_keys += 1
                raise VolatilityResearchError(
                    f"conflicting research duplicate for identity {key!r}"
                )
            exact_duplicate_rows += len(group) - 1
    clean = (
        frame.drop_duplicates(list(RESEARCH_KEY_COLUMNS), keep="first")
        .sort_values(list(RESEARCH_KEY_COLUMNS))
        .reset_index(drop=True)
    )
    return clean, {
        "input_rows": len(frame),
        "output_rows": len(clean),
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "conflicting_duplicate_keys": conflicting_keys,
        "row_classification": "admitted_unique",
        "target_classification": "continuous_realized_variance_regression",
    }


def _symbol_semivariance_features(
    examples: pd.DataFrame,
    mark_frame: pd.DataFrame,
    *,
    symbol: str,
    base_contract: VolatilityContract,
    research_contract: VolatilityResearchContract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized, price_audit = validate_price_frame(
        mark_frame,
        symbol=symbol,
        series_kind=base_contract.primary_series,
    )
    if normalized.empty:
        raise VolatilityResearchError(f"{symbol}: mark-price frame is empty")
    first = cast(pd.Timestamp, normalized["open_time"].iloc[0]).floor("min")
    last = cast(pd.Timestamp, normalized["open_time"].iloc[-1]).floor("min")
    grid = pd.date_range(first, last, freq="1min", tz="UTC")
    close_series = pd.Series(
        np.asarray(normalized["close"], dtype=float),
        index=pd.DatetimeIndex(normalized["open_time"]),
        dtype=float,
    )
    prices = close_series.reindex(grid).to_numpy(dtype=float, na_value=np.nan)
    log_prices = np.full_like(prices, np.nan, dtype=float)
    valid = np.isfinite(prices) & (prices > 0.0)
    log_prices[valid] = np.log(prices[valid])

    origins = pd.DatetimeIndex(
        sorted(cast(pd.Series, examples["origin_utc"]).unique())
    )
    offsets_ns = np.asarray(origins.view("int64"), dtype=np.int64) - int(first.value)
    minute_ns = 60_000_000_000
    if bool((offsets_ns % minute_ns != 0).any()):
        raise VolatilityResearchError(f"{symbol}: origin is not minute-aligned")
    origin_indexes = offsets_ns // minute_ns
    if bool((origin_indexes < 0).any()) or bool((origin_indexes >= len(grid)).any()):
        raise VolatilityResearchError(
            f"{symbol}: research origin lies outside the mark-price frame"
        )

    feature_data: dict[str, Any] = {"origin_utc": origins}
    maximum_identity_error = 0.0
    maximum_jump_value = 0.0
    for window in HAR_WINDOWS:
        sample_offsets = np.arange(
            window,
            -1,
            -base_contract.sampling_minutes,
            dtype=np.int64,
        )
        indexes = origin_indexes[:, None] - sample_offsets[None, :]
        if bool((indexes < 0).any()):
            raise VolatilityResearchError(
                f"{symbol}: {window}m feature lacks required history"
            )
        sampled = log_prices[indexes]
        if not bool(np.isfinite(sampled).all()):
            raise VolatilityResearchError(
                f"{symbol}: {window}m semivariance window touches a gap"
            )
        returns = np.diff(sampled, axis=1)
        expected_return_count = window // base_contract.sampling_minutes
        if returns.shape[1] != expected_return_count:
            raise VolatilityResearchError(
                f"{symbol}: {window}m jump window lacks full 5m return count"
            )
        squared = np.square(returns)
        realized = np.sum(squared, axis=1)
        positive = np.sum(np.where(returns >= 0.0, squared, 0.0), axis=1)
        negative = np.sum(np.where(returns < 0.0, squared, 0.0), axis=1)
        mu1 = math.sqrt(2.0 / math.pi)
        bipower = (1.0 / (mu1 * mu1)) * np.sum(
            np.abs(returns[:, 1:]) * np.abs(returns[:, :-1]), axis=1
        )
        jump = np.maximum(realized - bipower, 0.0)
        if not bool(np.isfinite(bipower).all()) or not bool(np.isfinite(jump).all()):
            raise VolatilityResearchError(
                f"{symbol}: {window}m bipower jump estimate is non-finite"
            )
        maximum_jump_value = max(
            maximum_jump_value,
            float(np.max(jump, initial=0.0)),
        )
        identity_error = np.abs((positive + negative) - realized)
        maximum_identity_error = max(
            maximum_identity_error,
            float(np.max(identity_error, initial=0.0)),
        )
        if not bool(
            np.allclose(
                positive + negative,
                realized,
                rtol=research_contract.semivariance_identity_relative_tolerance,
                atol=research_contract.semivariance_identity_absolute_tolerance,
            )
        ):
            raise VolatilityResearchError(
                f"{symbol}: {window}m semivariance identity failed"
            )

        source_column = f"rv_{window}"
        source_values = examples[["origin_utc", source_column]].copy()
        per_origin_counts = source_values.groupby("origin_utc")[source_column].nunique(
            dropna=False
        )
        if bool((per_origin_counts != 1).any()):
            raise VolatilityResearchError(
                f"{symbol}: {source_column} differs across horizon rows"
            )
        unique_origin_mask = ~cast(
            pd.Series, source_values["origin_utc"]
        ).duplicated()
        expected = np.asarray(
            source_values.loc[unique_origin_mask]
            .set_index("origin_utc")
            .reindex(origins)[source_column],
            dtype=float,
        )
        if not bool(
            np.allclose(
                expected,
                realized,
                rtol=research_contract.semivariance_identity_relative_tolerance,
                atol=research_contract.semivariance_identity_absolute_tolerance,
            )
        ):
            raise VolatilityResearchError(
                f"{symbol}: recomputed {source_column} differs from v1 evidence"
            )
        feature_data[f"rs_pos_{window}"] = positive
        feature_data[f"rs_neg_{window}"] = negative
        feature_data[f"jump_{window}"] = jump

    feature_frame = pd.DataFrame(feature_data)
    return feature_frame, {
        "symbol": symbol,
        "origin_count": len(feature_frame),
        "feature_columns": list(RESEARCH_FEATURE_COLUMNS),
        "maximum_semivariance_identity_absolute_error": maximum_identity_error,
        "jump_estimator": research_contract.jump_estimator,
        "jump_minimum_return_count_policy": (
            research_contract.jump_minimum_return_count_policy
        ),
        "maximum_jump_value": maximum_jump_value,
        "price_audit": price_audit.__dict__,
    }


def augment_examples_with_semivariance(
    examples: pd.DataFrame,
    *,
    price_store: Path,
    base_contract: VolatilityContract,
    research_contract: VolatilityResearchContract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Derive causal signed semivariances and jumps from preserved marks."""

    clean, duplicate_audit = deduplicate_research_examples(examples)
    symbols = sorted(cast(pd.Series, clean["symbol"]).astype(str).unique())
    augmented: list[pd.DataFrame] = []
    feature_audits: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_examples = clean.loc[
            cast(pd.Series, clean["symbol"]).astype(str) == symbol
        ].copy()
        mark_frame = load_price_store_frame(
            price_store,
            symbol=symbol,
            series_kind=base_contract.primary_series,
        )
        features, audit = _symbol_semivariance_features(
            symbol_examples,
            mark_frame,
            symbol=symbol,
            base_contract=base_contract,
            research_contract=research_contract,
        )
        merged = symbol_examples.merge(
            features,
            on="origin_utc",
            how="left",
            validate="many_to_one",
        )
        if bool(merged[list(RESEARCH_FEATURE_COLUMNS)].isna().any().any()):
            raise VolatilityResearchError(
                f"{symbol}: research feature merge produced missing values"
            )
        augmented.append(merged)
        feature_audits.append(audit)
    result = (
        pd.concat(augmented, ignore_index=True)
        .sort_values(list(RESEARCH_KEY_COLUMNS))
        .reset_index(drop=True)
    )
    if len(result) != len(clean):
        raise VolatilityResearchError("semivariance augmentation changed row count")
    return result, {
        "duplicate_audit": duplicate_audit,
        "feature_audits": feature_audits,
        "input_rows": len(examples),
        "output_rows": len(result),
        "symbols": symbols,
        "candidate_families": list(research_contract.candidate_families),
        "jump_policy": "bipower_variation_with_full_5m_return_windows",
        "promotion_eligible": False,
        "verdict_influence": False,
    }
