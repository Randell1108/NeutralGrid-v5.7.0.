from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import PROFIT_FACTOR_CAP
from neutralgrid.scanner._xlsx_io import (
    detect_format,
    raise_on_duplicate_strategy_id,
    read_sheet,
    read_single_sheet,
    validate_dataframe,
)

logger = logging.getLogger(__name__)


DEFAULT_FEATURES = [
    "parkinson_vol_ratio_4h_24h_pre",
    "variance_ratio_1m_15m_pre_2h",
    "funding_carry_expected_next_7h",
    "liquidity_stability_z_1h",
]


@dataclass(frozen=True)
class PatternProfile:
    features: list[str]
    means: dict[str, float]
    stds: dict[str, float]
    q10: dict[str, float]
    q90: dict[str, float]
    selection_summary: dict[str, Any]

    def similarity_score(
        self,
        row: Mapping[str, Any],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        if weights is None:
            weights = {}

        dist = 0.0
        wsum = 0.0

        for f in self.features:
            if f not in self.means or f not in self.stds:
                continue

            v = row.get(f, None)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if np.isnan(v):
                continue

            mu = float(self.means[f])
            sd = float(self.stds[f]) if float(self.stds[f]) > 0 else 1.0
            z = (v - mu) / sd

            w = float(weights.get(f, 1.0))
            dist += w * (z * z)
            wsum += w

        if wsum <= 0:
            return 0.0

        dist = dist / wsum
        return float(max(0.0, min(100.0, 100.0 * np.exp(-0.5 * dist))))

    def to_json(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "means": dict(self.means),
            "stds": dict(self.stds),
            "q10": dict(self.q10),
            "q90": dict(self.q90),
            "selection_summary": dict(self.selection_summary),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "PatternProfile":
        return cls(
            features=[str(x) for x in (obj.get("features") or [])],
            means={str(k): float(v) for k, v in (obj.get("means") or {}).items()},
            stds={str(k): float(v) for k, v in (obj.get("stds") or {}).items()},
            q10={str(k): float(v) for k, v in (obj.get("q10") or {}).items()},
            q90={str(k): float(v) for k, v in (obj.get("q90") or {}).items()},
            selection_summary=dict(obj.get("selection_summary") or {}),
        )

    def validate_features(
        self,
        canonical_features: list[str] | None = None,
    ) -> list[str]:
        """Validate that the profile features match a canonical feature list."""
        expected = canonical_features or list(DEFAULT_FEATURES)
        warnings: list[str] = []

        profile_set = set(self.features)
        expected_set = set(expected)

        unknown = profile_set - expected_set
        if unknown:
            raise ValueError(
                f"Pattern profile contains unknown features: {sorted(unknown)}"
            )

        missing = expected_set - profile_set
        if missing:
            warnings.append(
                f"Pattern profile is missing expected features: {sorted(missing)}"
            )

        has_stats = [f for f in expected if f in self.means and f in self.stds]
        if not has_stats:
            raise ValueError(
                "Pattern profile has no statistical data (means/stds) for any "
                f"canonical feature - expected at least one of {expected}"
            )

        missing_stats = [f for f in expected if f not in self.means or f not in self.stds]
        if missing_stats:
            warnings.append(
                f"Pattern profile missing means/stds for features: {sorted(missing_stats)}"
            )

        return warnings

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load_json(
        path: str | Path,
        canonical_features: list[str] | None = None,
    ) -> Optional["PatternProfile"]:
        """Load a pattern profile from JSON with schema validation."""
        path = Path(path)
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read pattern profile from %s: %s", path, exc)
            return None

        try:
            profile = PatternProfile.from_json(obj)
        except Exception as exc:
            logger.warning("Failed to parse pattern profile from %s: %s", path, exc)
            return None

        try:
            expected_features = (
                canonical_features if canonical_features is not None else list(DEFAULT_FEATURES)
            )
            warnings_list = profile.validate_features(expected_features)
            for warning in warnings_list:
                logger.warning("Pattern profile validation (%s): %s", path.name, warning)
        except ValueError as exc:
            logger.warning(
                "Pattern profile validation failed for %s: %s - returning None (fail-closed)",
                path,
                exc,
            )
            return None

        return profile


def build_profile_from_enhanced_xlsx(
    xlsx_path: str | Path,
    *,
    max_duration_hours: float = 7.0,
    min_profit_factor: float = 1.5,
    min_avg_profit_per_grid: Optional[float] = None,
    top_quantile: float = 0.75,
    features: Iterable[str] = DEFAULT_FEATURES,
) -> PatternProfile:
    """Build a statistical profile of winning bots from enhanced xlsx.

    Uses a bounded training universe: only rows with 0 <= duration_hours < max_duration_hours.
    """
    if min_avg_profit_per_grid is None:
        min_avg_profit_per_grid = get_config().grid.profit_grid_min_pct_static_fallback

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Reference xlsx not found: {xlsx_path}")

    fmt = detect_format(xlsx_path)
    if fmt == "multi":
        df_entry = read_sheet(xlsx_path, "Entry Validation Metrics")
        df_perf = read_sheet(xlsx_path, "Performance Risk-Adjusted")

        all_warnings: list[str] = []
        df_entry, w1 = validate_dataframe(df_entry, "Entry Validation Metrics")
        df_perf, w2 = validate_dataframe(df_perf, "Performance Risk-Adjusted")
        all_warnings.extend(w1 + w2)
        raise_on_duplicate_strategy_id(df_entry, context="Entry Validation Metrics")
        raise_on_duplicate_strategy_id(df_perf, context="Performance Risk-Adjusted")

        for warning in all_warnings:
            logger.warning(f"Data validation: {warning}")

        perf_cols_needed = [
            "strategy_id",
            "pnl_pct",
            "profit_factor",
            "duration_hours",
            "total_profit_usdt",
            "avg_profit_per_grid",
            "status",
        ]
        perf_cols_to_merge = ["strategy_id"] + [
            c for c in perf_cols_needed[1:] if c not in df_entry.columns
        ]

        df = df_entry.copy()
        if len(perf_cols_to_merge) > 1:
            df = df.merge(df_perf[perf_cols_to_merge], on="strategy_id", how="left")
    else:
        logger.info("Single-sheet format detected; reading canonical sheet")
        df, sheet_name = read_single_sheet(xlsx_path)
        df, warnings_list = validate_dataframe(df, sheet_name)
        raise_on_duplicate_strategy_id(df, context=sheet_name)
        for warning in warnings_list:
            logger.warning(f"Data validation: {warning}")

    feature_list = list(features)
    numeric_cols = [
        "pnl_pct",
        "profit_factor",
        "duration_hours",
        "total_profit_usdt",
        "avg_profit_per_grid",
        *feature_list,
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "profit_factor" in df.columns:
        df["profit_factor"] = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
        df["profit_factor"] = df["profit_factor"].clip(lower=0.0, upper=PROFIT_FACTOR_CAP)

    duration_series = (
        cast(pd.Series, df["duration_hours"])
        if "duration_hours" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    df_train = cast(
        pd.DataFrame,
        df.loc[(duration_series >= 0.0) & (duration_series < max_duration_hours)].copy(),
    )

    logger.info(
        "Bounded universe: %d rows with 0 <= duration_hours < %.1f (of %d total)",
        len(df_train),
        max_duration_hours,
        len(df),
    )

    pf_train = (
        cast(pd.Series, df_train["profit_factor"])
        if "profit_factor" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    pnl_train = (
        cast(pd.Series, df_train["pnl_pct"])
        if "pnl_pct" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    labelable_mask = pf_train.notna() & pnl_train.notna()
    df_labeled = cast(pd.DataFrame, df_train.loc[labelable_mask].copy())

    logger.info(
        "Labeled subset: %d rows (excluded %d with missing profit_factor or pnl_pct)",
        len(df_labeled),
        len(df_train) - len(df_labeled),
    )

    pnl_thr = (
        float(cast(pd.Series, df_labeled["pnl_pct"]).quantile(top_quantile))
        if len(df_labeled) > 0
        else 0.0
    )

    pf_labeled = cast(pd.Series, df_labeled["profit_factor"])
    pnl_labeled = cast(pd.Series, df_labeled["pnl_pct"])

    winners_mask = (pf_labeled >= min_profit_factor) & (pnl_labeled >= pnl_thr)

    if "avg_profit_per_grid" in df_labeled.columns:
        apg = cast(pd.Series, df_labeled["avg_profit_per_grid"])
        winners_mask = winners_mask & ((apg.isna()) | (apg >= min_avg_profit_per_grid))

    winners = cast(pd.DataFrame, df_labeled.loc[winners_mask].copy())

    if len(winners) < 5:
        raise ValueError(
            f"Bounded universe produced only {len(winners)} winners (need >= 5). "
            f"Cannot build pattern profile on {len(df_labeled)} labeled rows "
            f"with pnl_thr={pnl_thr:.2f}, min_pf={min_profit_factor}."
        )

    feats = [
        f
        for f in feature_list
        if f in df_labeled.columns and int(cast(pd.Series, df_labeled[f]).notna().sum()) >= 10
    ]
    if not feats:
        raise ValueError(
            "No features available in the labeled universe (all dropped by availability filter)."
        )

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    q10: dict[str, float] = {}
    q90: dict[str, float] = {}

    fitted_features: list[str] = []
    for f in feats:
        if f not in winners.columns:
            continue
        feature_series = cast(pd.Series, winners[f])
        dropped_count = int(feature_series.isna().sum())
        if dropped_count > 0:
            logger.warning(
                "build_profile_from_xlsx: dropping %d NaN values in feature '%s'",
                dropped_count,
                f,
            )
        s = cast(pd.Series, feature_series.dropna())
        if s.empty:
            continue
        means[f] = float(cast(pd.Series, s).mean())
        sd = float(np.std(np.asarray(s, dtype=float), ddof=0))
        stds[f] = sd if sd > 0 else 1.0
        q10[f] = float(cast(pd.Series, s).quantile(0.10))
        q90[f] = float(cast(pd.Series, s).quantile(0.90))
        fitted_features.append(f)

    if not fitted_features:
        raise ValueError(
            "No features available in the winner subset after dropping NaN values."
        )

    if "symbol" in winners.columns:
        winner_symbols = cast(pd.Series, winners["symbol"]).astype(str).tolist()
    else:
        winner_symbols = []

    selection_summary = {
        "xlsx": str(xlsx_path),
        "winners_count": int(len(winners)),
        "pnl_threshold": pnl_thr,
        "max_duration_hours": float(max_duration_hours),
        "duration_band": {"min_hours": 0.0, "max_hours": float(max_duration_hours)},
        "min_profit_factor": float(min_profit_factor),
        "min_avg_profit_per_grid": float(min_avg_profit_per_grid),
        "top_quantile": float(top_quantile),
        "bounded_universe_size": len(df_train),
        "labeled_universe_size": len(df_labeled),
        "winners_symbols": sorted(set(winner_symbols)),
        "label_name": "fast_completed_winner_under_7h",
        "label_definition": {
            "duration_rule": (
                f"0 <= duration_hours < {float(max_duration_hours)}"
            ),
            "performance_rule": (
                "profit_factor >= min_profit_factor and "
                "pnl_pct >= train_only_top_quantile_threshold"
            ),
            "time_to_target_claimed": False,
        },
    }

    return PatternProfile(
        features=fitted_features,
        means=means,
        stds=stds,
        q10=q10,
        q90=q90,
        selection_summary=selection_summary,
    )
