"""grid.spacing_profile

Empirical grid spacing prior.

Purpose
-------
Neutral-grid spacing has a direct impact on fill frequency and net profit per
round-trip (after fees). The scanner already uses a *learned* winner profile
for candidate selection; this module extends that idea to grid spacing by
deriving a robust target spacing from historical winners in
``data/new_expired_bots.xlsx``.

Design constraints
------------------
* Deterministic, auditable computation (no randomness).
* Uses only historical summary data (no look-ahead into live candles).
* Caches the computed value to avoid repeated Excel reads.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.grid.formulas import grid_spacing_pct as _grid_spacing_pct

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import BOT_INCLUSION_TOLERANCE_HOURS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpacingProfile:
    """Derived spacing statistics."""

    target_spacing_pct: float
    sample_size: int
    method: str


@dataclass(frozen=True)
class IQRBand:
    """Interquartile band for one numeric winner feature."""

    q1: float
    median: float
    q3: float


@dataclass(frozen=True)
class WinnerIQRBin:
    """Winner geometry profile for one range-size bucket."""

    label: str
    left: float
    right: float
    rows: int
    range_size_pct: IQRBand
    grid_density_per_range_pct: IQRBand
    num_grids: IQRBand
    grid_spacing_pct: IQRBand
    profit_per_grid_pct: IQRBand


@dataclass(frozen=True)
class WinnerIQRProfile:
    """Decile-conditioned winner geometry profile."""

    bins: tuple[WinnerIQRBin, ...]
    range_edges: tuple[float, ...]
    winner_rows: int
    winner_symbols: int
    geometry_rows: int
    dropped_geometry_rows: int
    medians: dict[str, float]
    winner_filter: str


_CACHED: Optional[SpacingProfile] = None

WINNER_IQR_REQUIRED_COLUMNS = (
    "duration_hours",
    "pnl_pct",
    "grids_count",
    "range_size_pct",
    "grid_spacing_pct",
    "profit_per_grid_pct",
)

CANDIDATE_IQR_REQUIRED_COLUMNS = (
    "range_size_pct",
    "num_grids",
    "grid_spacing_pct",
    "profit_per_grid_pct",
)


def _infer_mode(low: float, high: float, num_grids: int, stored_spacing_pct: float) -> str:
    """Classify a row as geometric/arithmetic by minimum residual reconstruction.

    GRIDFIX-001 / GRID_SYNCH §3.2: delegates to
    ``neutralgrid.grid.formulas.grid_spacing_pct`` so the canonical identity
    lives in a single module (was an inline duplicate). The formula module
    expects ``n = grid LINES`` (per Binance convention) and divides by
    ``n - 1`` internally; ``num_grids`` from the workbook is already lines.

    Returns "unknown" when inputs are degenerate (n < 2, low <= 0,
    high <= low, or non-finite inputs). No silent path: the caller must
    handle "unknown" explicitly.
    """
    try:
        n_lines = int(num_grids)
        if n_lines < 2:
            return "unknown"
        if low <= 0 or high <= low or not (
            np.isfinite(low) and np.isfinite(high) and np.isfinite(stored_spacing_pct)
        ):
            return "unknown"
        geom = _grid_spacing_pct(float(low), float(high), n_lines, "geometric")
        arith = _grid_spacing_pct(float(low), float(high), n_lines, "arithmetic")
        return "geometric" if abs(stored_spacing_pct - geom) < abs(stored_spacing_pct - arith) else "arithmetic"
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return "unknown"


def _filter_geometric_dedup(
    df: pd.DataFrame,
    *,
    context: str,
    min_pool_n: int = 20,
) -> pd.DataFrame:
    """Restrict winner pool to geometric rows; deduplicate best-pnl per symbol.

    Fail-closed: raises if the resulting pool falls below ``min_pool_n``
    (production default 20) so that recalibrations against a thin pool
    cannot silently corrupt the spacing target.
    """
    required = {"price_range_low", "price_range_high", "grids_count", "grid_spacing_pct", "pnl_pct", "symbol"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: cannot infer mode; missing column(s): {missing}")

    inferred = [
        _infer_mode(
            float(row["price_range_low"]),
            float(row["price_range_high"]),
            int(row["grids_count"]) if bool(pd.notna(row["grids_count"])) else 0,
            float(row["grid_spacing_pct"]) if bool(pd.notna(row["grid_spacing_pct"])) else float("nan"),
        )
        for _, row in df.iterrows()
    ]
    df = df.assign(_inferred_mode=inferred)
    geom = cast(pd.DataFrame, df.loc[df["_inferred_mode"] == "geometric"].copy())

    if len(geom) < int(min_pool_n):
        raise ValueError(
            f"{context}: geometric subpool size {len(geom)} < {int(min_pool_n)}; refusing to compute spacing target."
        )

    geom = cast(
        pd.DataFrame,
        geom.sort_values("pnl_pct", ascending=False).drop_duplicates(subset="symbol", keep="first"),
    )
    if geom["symbol"].nunique() < int(min_pool_n):
        raise ValueError(
            f"{context}: geometric+dedup unique symbols {geom['symbol'].nunique()} < {int(min_pool_n)}."
        )

    return cast(pd.DataFrame, geom.drop(columns=["_inferred_mode"]))


def _compute_from_xlsx(xlsx_path: Path) -> SpacingProfile:
    df = cast(pd.DataFrame, pd.read_excel(xlsx_path))

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Defensive numeric casting
    for c in [
        "pnl_pct", "grid_spacing_pct", "duration_hours", "leverage",
        "invested_margin_usdt", "price_range_low", "price_range_high", "grids_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter to the canonical bot-lifetime horizon H* (+ inclusion tolerance per
    # D14.1), positive performance, and leverage=10. Capital is not used as a
    # filter because historical records include multiple tested allocations;
    # spacing is dimensionless. The tolerance tau allows historical winners
    # that overran H* by up to BOT_INCLUSION_TOLERANCE_HOURS to remain in the
    # pool — reflects fill/cancel/metric-flush latency.
    _cfg = get_config()
    hours = float(_cfg.grid.max_holding_seconds) / 3600.0
    duration_cap = hours + BOT_INCLUSION_TOLERANCE_HOURS
    lev = float(_cfg.grid.leverage_max)

    base = cast(pd.DataFrame, df.copy())
    if "duration_hours" in base.columns:
        duration_series = cast(pd.Series, base["duration_hours"])
        base = cast(pd.DataFrame, base.loc[duration_series <= duration_cap])
    if "pnl_pct" in base.columns:
        pnl_series = cast(pd.Series, base["pnl_pct"])
        base = cast(pd.DataFrame, base.loc[pnl_series > 0])
    if "leverage" in base.columns:
        leverage_series = cast(pd.Series, base["leverage"])
        base = cast(pd.DataFrame, base.loc[leverage_series == lev])

    # AFML: Log dropped rows instead of silent dropna
    base_pair = cast(pd.DataFrame, base[["grid_spacing_pct", "pnl_pct"]])
    dropped_base = int(cast(pd.Series, base_pair.isna().any(axis=1)).sum())
    if dropped_base > 0:
        logger.warning(f"derive_spacing_profile: dropping {dropped_base} rows with NaN in grid_spacing_pct or pnl_pct")
    base = cast(pd.DataFrame, base.dropna(
        subset=["grid_spacing_pct", "pnl_pct", "price_range_low", "price_range_high", "grids_count"],
        how="any",
    ))
    if base.empty:
        # Fallback: use any rows with a spacing value.
        # AFML: Log dropped rows instead of silent dropna
        dropped_fallback1 = int(cast(pd.Series, df["grid_spacing_pct"]).isna().sum())
        if dropped_fallback1 > 0:
            logger.warning(f"derive_spacing_profile (fallback): dropping {dropped_fallback1} rows with NaN grid_spacing_pct")
        any_rows = cast(pd.DataFrame, df.dropna(subset=["grid_spacing_pct"], how="any"))
        any_rows = cast(pd.DataFrame, any_rows.copy())
        any_rows["grid_spacing_pct"] = pd.to_numeric(any_rows["grid_spacing_pct"], errors="coerce")
        # AFML: Log dropped rows after coercion
        dropped_fallback2 = int(cast(pd.Series, any_rows["grid_spacing_pct"]).isna().sum())
        if dropped_fallback2 > 0:
            logger.warning(f"derive_spacing_profile (fallback): dropping {dropped_fallback2} rows after numeric coercion")
        any_rows = cast(pd.DataFrame, any_rows.dropna(subset=["grid_spacing_pct"], how="any"))
        if any_rows.empty:
            return SpacingProfile(target_spacing_pct=0.90, sample_size=0, method="fallback_constant")
        return SpacingProfile(
            target_spacing_pct=float(cast(pd.Series, any_rows["grid_spacing_pct"]).median()),
            sample_size=int(len(any_rows)),
            method="fallback_median_all",
        )

    # Restrict to geometric+dedup pool (mode-consistency invariant).
    geom = _filter_geometric_dedup(base, context="derive_spacing_profile")

    # Robust location estimator: median spacing across the full geometric pool.
    # Empirical CV across range quintiles is ~3% (range-invariant), so a single
    # global target is statistically indistinguishable from a conditional one
    # at this N and is the conservative choice.
    target = float(cast(pd.Series, geom["grid_spacing_pct"]).median())
    return SpacingProfile(
        target_spacing_pct=target,
        sample_size=int(len(geom)),
        method="geometric_subpool_dedup_median",
    )


def get_spacing_profile(*, refresh: bool = False) -> SpacingProfile:
    """Return cached spacing profile (compute once per process)."""

    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    _cfg = get_config()
    xlsx = _cfg.resolve_path(str(_cfg.grid.grid_spacing_profile_path))
    if not xlsx.exists():
        _CACHED = SpacingProfile(target_spacing_pct=0.90, sample_size=0, method="missing_xlsx_fallback")
        return _CACHED

    _CACHED = _compute_from_xlsx(xlsx)
    return _CACHED


def get_target_spacing_pct(*, refresh: bool = False) -> float:
    """Convenience helper returning the target spacing (in percent, e.g. 0.87)."""

    return float(get_spacing_profile(refresh=refresh).target_spacing_pct)


def _read_frame(data: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            return cast(pd.DataFrame, pd.read_excel(path))
        if path.suffix.lower() == ".csv":
            return cast(pd.DataFrame, pd.read_csv(path))
        raise ValueError(f"Unsupported input file extension: {path.suffix}")

    return cast(pd.DataFrame, data.copy())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = cast(pd.DataFrame, df.copy())
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _require_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required column(s): {missing}")


def _numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = cast(pd.DataFrame, df.copy())
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _iqr(series: pd.Series) -> IQRBand:
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    values = numeric.dropna()
    return IQRBand(
        q1=float(values.quantile(0.25)),
        median=float(values.median()),
        q3=float(values.quantile(0.75)),
    )


def _band_to_dict(band: IQRBand) -> dict[str, float]:
    return {"q1": band.q1, "median": band.median, "q3": band.q3}


def build_winner_iqr_profile(
    workbook: Path | str | pd.DataFrame,
    *,
    deciles: int = 10,
    min_pool_n: int = 20,
) -> WinnerIQRProfile:
    """Build the fast-winner geometry IQR profile.

    Winner label is intentionally fixed to the requested contract:
    ``duration_hours < 7.0`` and ``pnl_pct > 1.0``. Missing geometry rows are
    dropped and counted, never imputed.
    """

    if deciles < 2:
        raise ValueError("deciles must be >= 2")

    df = _normalize_columns(_read_frame(workbook))
    _require_columns(df, WINNER_IQR_REQUIRED_COLUMNS, context="winner workbook")

    if "strategy_id" in df.columns:
        duplicate_count = int(df["strategy_id"].duplicated(keep=False).sum())
        if duplicate_count:
            raise ValueError(f"Duplicate strategy_id in winner workbook: {duplicate_count} rows")

    df = _numeric(
        df,
        list(WINNER_IQR_REQUIRED_COLUMNS) + ["price_range_low", "price_range_high"],
    )
    winners = cast(
        pd.DataFrame,
        df.loc[
            (cast(pd.Series, df["duration_hours"]) < 7.0)
            & (cast(pd.Series, df["pnl_pct"]) > 1.0)
        ].copy(),
    )
    winner_rows = int(len(winners))
    winner_symbols = int(winners["symbol"].nunique()) if "symbol" in winners.columns else 0
    geometry = cast(
        pd.DataFrame,
        winners.dropna(
            subset=[
                "grids_count",
                "range_size_pct",
                "grid_spacing_pct",
                "profit_per_grid_pct",
                "price_range_low",
                "price_range_high",
            ],
            how="any",
        ).copy(),
    )
    dropped_geometry_rows = winner_rows - int(len(geometry))

    if geometry.empty:
        raise ValueError("No winner rows remain after geometry dropna")

    # Restrict to geometric+dedup pool (mode-consistency invariant).
    geometry = _filter_geometric_dedup(
        geometry,
        context="build_winner_iqr_profile",
        min_pool_n=int(min_pool_n),
    )
    # Recount post-filter so dropped_geometry_rows reflects all drop sources
    # (NaN drop + non-geometric mode + best-pnl dedup).
    dropped_geometry_rows = winner_rows - int(len(geometry))

    geometry["grid_density_per_range_pct"] = (
        cast(pd.Series, geometry["grids_count"])
        / cast(pd.Series, geometry["range_size_pct"])
    )
    geometry = cast(
        pd.DataFrame,
        geometry.loc[
            cast(pd.Series, geometry["range_size_pct"]) > 0
        ].copy(),
    )
    if geometry.empty:
        raise ValueError("No winner rows remain after positive range_size_pct filter")

    try:
        bucketed, raw_edges = pd.qcut(
            cast(pd.Series, geometry["range_size_pct"]),
            q=deciles,
            retbins=True,
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError("Unable to compute winner range_size_pct deciles") from exc

    geometry["winner_range_bucket"] = bucketed
    bins: list[WinnerIQRBin] = []
    grouped = geometry.groupby("winner_range_bucket", observed=True)
    for bucket, group in grouped:
        interval = cast(pd.Interval, bucket)
        label = str(interval)
        bins.append(
            WinnerIQRBin(
                label=label,
                left=float(interval.left),
                right=float(interval.right),
                rows=int(len(group)),
                range_size_pct=_iqr(cast(pd.Series, group["range_size_pct"])),
                grid_density_per_range_pct=_iqr(
                    cast(pd.Series, group["grid_density_per_range_pct"])
                ),
                num_grids=_iqr(cast(pd.Series, group["grids_count"])),
                grid_spacing_pct=_iqr(cast(pd.Series, group["grid_spacing_pct"])),
                profit_per_grid_pct=_iqr(cast(pd.Series, group["profit_per_grid_pct"])),
            )
        )

    medians = {
        "num_grids": float(cast(pd.Series, geometry["grids_count"]).median()),
        "range_size_pct": float(cast(pd.Series, geometry["range_size_pct"]).median()),
        "grid_spacing_pct": float(cast(pd.Series, geometry["grid_spacing_pct"]).median()),
        "profit_per_grid_pct": float(cast(pd.Series, geometry["profit_per_grid_pct"]).median()),
        "grid_density_per_range_pct": float(
            cast(pd.Series, geometry["grid_density_per_range_pct"]).median()
        ),
        "grid_density_ratio_of_medians": float(
            cast(pd.Series, geometry["grids_count"]).median()
            / cast(pd.Series, geometry["range_size_pct"]).median()
        ),
    }

    return WinnerIQRProfile(
        bins=tuple(bins),
        range_edges=tuple(float(v) for v in raw_edges),
        winner_rows=winner_rows,
        winner_symbols=winner_symbols,
        geometry_rows=int(len(geometry)),
        dropped_geometry_rows=dropped_geometry_rows,
        medians=medians,
        winner_filter="duration_hours < 7.0 AND pnl_pct > 1.0",
    )


def _find_profile_bin(profile: WinnerIQRProfile, range_size_pct: float) -> WinnerIQRBin | None:
    for idx, bucket in enumerate(profile.bins):
        is_last = idx == len(profile.bins) - 1
        if bucket.left <= range_size_pct < bucket.right or (
            is_last and range_size_pct <= bucket.right
        ):
            return bucket
    return None


def _in_iqr(value: Any, band: IQRBand) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return band.q1 <= v <= band.q3


def _near(value: Any, target: Any, tolerance: float) -> bool:
    try:
        v = float(value)
        t = float(target)
    except (TypeError, ValueError):
        return False
    if t <= 0:
        return False
    return abs(v - t) <= abs(t) * tolerance


def _candidate_calibrated_mask(df: pd.DataFrame) -> pd.Series:
    if "hmm_winner_score_source" in df.columns:
        return cast(pd.Series, df["hmm_winner_score_source"]).astype(str).str.lower().eq("calibrated")
    return pd.Series(True, index=df.index)


def score_candidates_against_winner_iqr(
    candidates: Path | str | pd.DataFrame,
    profile: WinnerIQRProfile,
    *,
    min_grids: int | None = None,
    max_grids: int | None = None,
    require_calibrated: bool = True,
    cost_tolerance: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark candidate rows against the decile-conditioned winner IQR box."""

    df = _normalize_columns(_read_frame(candidates))
    _require_columns(df, CANDIDATE_IQR_REQUIRED_COLUMNS, context="candidate CSV")

    numeric_cols = list(CANDIDATE_IQR_REQUIRED_COLUMNS) + [
        "micro_round_trip_cost_pct",
        "micro_min_profit_required_pct",
        "dynamic_profit_floor_pct",
    ]
    df = _numeric(df, numeric_cols)

    input_rows = int(len(df))
    if require_calibrated:
        calibrated_mask = _candidate_calibrated_mask(df)
        df = cast(pd.DataFrame, df.loc[calibrated_mask].copy())
    calibrated_rows = int(len(df))

    geometry = cast(
        pd.DataFrame,
        df.dropna(subset=list(CANDIDATE_IQR_REQUIRED_COLUMNS), how="any").copy(),
    )
    geometry = cast(
        pd.DataFrame,
        geometry.loc[cast(pd.Series, geometry["range_size_pct"]) > 0].copy(),
    )
    dropped_geometry_rows = calibrated_rows - int(len(geometry))

    cfg = get_config()
    if min_grids is None:
        min_grids = int(cfg.grid.min_grids)
    if max_grids is None:
        max_grids = int(cfg.grid.max_grids)

    winner_bucket_labels: list[str | None] = []
    density_values: list[float | None] = []
    num_flags: list[bool] = []
    spacing_flags: list[bool] = []
    profit_flags: list[bool] = []
    density_flags: list[bool] = []
    tuple_flags: list[bool] = []
    binding: list[str] = []

    for _, row in geometry.iterrows():
        range_size = float(row["range_size_pct"])
        bucket = _find_profile_bin(profile, range_size)
        density = float(row["num_grids"]) / range_size if range_size > 0 else None
        density_values.append(density)

        if bucket is None or density is None:
            winner_bucket_labels.append(None)
            num_flags.append(False)
            spacing_flags.append(False)
            profit_flags.append(False)
            density_flags.append(False)
            tuple_flags.append(False)
            binding.append("outside_winner_range")
            continue

        winner_bucket_labels.append(bucket.label)
        num_ok = _in_iqr(row["num_grids"], bucket.num_grids)
        spacing_ok = _in_iqr(row["grid_spacing_pct"], bucket.grid_spacing_pct)
        profit_ok = _in_iqr(row["profit_per_grid_pct"], bucket.profit_per_grid_pct)
        density_ok = _in_iqr(density, bucket.grid_density_per_range_pct)
        tuple_ok = bool(num_ok and spacing_ok and profit_ok)
        num_flags.append(num_ok)
        spacing_flags.append(spacing_ok)
        profit_flags.append(profit_ok)
        density_flags.append(density_ok)
        tuple_flags.append(tuple_ok)

        if _near(row["grid_spacing_pct"], row.get("micro_round_trip_cost_pct"), cost_tolerance):
            binding.append("round_trip_cost_floor")
        elif _near(row["grid_spacing_pct"], row.get("micro_min_profit_required_pct"), cost_tolerance):
            binding.append("micro_profit_floor")
        elif int(float(row["num_grids"])) == int(max_grids):
            binding.append("max_grids_cap")
        elif (
            int(float(row["num_grids"])) == int(min_grids)
            and density < bucket.grid_density_per_range_pct.q1
        ):
            binding.append("spacing_target_at_min_grids")
        elif density < bucket.grid_density_per_range_pct.q1:
            binding.append("spacing_target_too_wide")
        elif tuple_ok:
            binding.append("winner_iqr_aligned")
        else:
            binding.append("geometry_distribution_mismatch")

    scored = geometry.copy()
    scored["winner_range_bucket"] = winner_bucket_labels
    scored["grid_density_per_range_pct"] = density_values
    scored["num_grids_in_iqr"] = num_flags
    scored["grid_spacing_in_iqr"] = spacing_flags
    scored["profit_per_grid_in_iqr"] = profit_flags
    scored["grid_density_in_iqr"] = density_flags
    scored["tuple_in_iqr"] = tuple_flags
    scored["winner_iqr_binding_constraint"] = binding

    if not scored.empty:
        density_series = cast(pd.Series, scored["grid_density_per_range_pct"])
        candidate_medians = {
            "num_grids": float(cast(pd.Series, scored["num_grids"]).median()),
            "range_size_pct": float(cast(pd.Series, scored["range_size_pct"]).median()),
            "grid_spacing_pct": float(cast(pd.Series, scored["grid_spacing_pct"]).median()),
            "profit_per_grid_pct": float(cast(pd.Series, scored["profit_per_grid_pct"]).median()),
            "grid_density_per_range_pct": float(density_series.median()),
            "grid_density_ratio_of_medians": float(
                cast(pd.Series, scored["num_grids"]).median()
                / cast(pd.Series, scored["range_size_pct"]).median()
            ),
        }
    else:
        candidate_medians = {}

    binding_counts = (
        scored["winner_iqr_binding_constraint"].value_counts(dropna=False).to_dict()
        if "winner_iqr_binding_constraint" in scored.columns
        else {}
    )
    summary: dict[str, Any] = {
        "input_rows": input_rows,
        "calibrated_rows": calibrated_rows,
        "geometry_rows": int(len(scored)),
        "dropped_geometry_rows": dropped_geometry_rows,
        "tuple_in_iqr_rows": int(cast(pd.Series, scored.get("tuple_in_iqr", pd.Series(dtype=bool))).sum())
        if not scored.empty
        else 0,
        "winner_medians": profile.medians,
        "candidate_medians": candidate_medians,
        "binding_counts": {str(k): int(v) for k, v in binding_counts.items()},
    }

    winner_density = profile.medians.get("grid_density_ratio_of_medians")
    candidate_density = candidate_medians.get("grid_density_ratio_of_medians")
    if winner_density and candidate_density is not None:
        summary["candidate_density_share_of_winner"] = float(candidate_density / winner_density)

    return scored, summary


def audit_winner_iqr_alignment(
    workbook: Path | str | pd.DataFrame,
    candidates: Path | str | pd.DataFrame,
    *,
    deciles: int = 10,
    require_calibrated: bool = True,
) -> tuple[WinnerIQRProfile, pd.DataFrame, dict[str, Any]]:
    """Build winner profile and score candidates in one read-only call."""

    profile = build_winner_iqr_profile(workbook, deciles=deciles)
    scored, summary = score_candidates_against_winner_iqr(
        candidates,
        profile,
        require_calibrated=require_calibrated,
    )
    return profile, scored, summary


def winner_iqr_profile_to_dict(profile: WinnerIQRProfile) -> dict[str, Any]:
    """Serialize the profile for proof logging or CLI display."""

    return {
        "winner_filter": profile.winner_filter,
        "winner_rows": profile.winner_rows,
        "winner_symbols": profile.winner_symbols,
        "geometry_rows": profile.geometry_rows,
        "dropped_geometry_rows": profile.dropped_geometry_rows,
        "range_edges": list(profile.range_edges),
        "medians": profile.medians,
        "bins": [
            {
                "label": b.label,
                "left": b.left,
                "right": b.right,
                "rows": b.rows,
                "range_size_pct": _band_to_dict(b.range_size_pct),
                "grid_density_per_range_pct": _band_to_dict(
                    b.grid_density_per_range_pct
                ),
                "num_grids": _band_to_dict(b.num_grids),
                "grid_spacing_pct": _band_to_dict(b.grid_spacing_pct),
                "profit_per_grid_pct": _band_to_dict(b.profit_per_grid_pct),
            }
            for b in profile.bins
        ],
    }


def _latest_candidate_csv(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.glob("deployment_ready_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No deployment_ready_*.csv files under {results_dir}")
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect winner-derived grid spacing and IQR geometry profiles."
    )
    parser.add_argument("--iqr-audit", action="store_true", help="Run winner-IQR candidate audit.")
    parser.add_argument(
        "--workbook",
        default="data/new_expired_bots.xlsx",
        help="Workbook used for the fast-winner IQR profile.",
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Candidate CSV path. Defaults to latest results/deployment_ready_*.csv.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument(
        "--include-uncalibrated",
        action="store_true",
        help="Do not filter to hmm_winner_score_source == calibrated.",
    )
    args = parser.parse_args(argv)

    if not args.iqr_audit:
        profile = get_spacing_profile(refresh=True)
        print(json.dumps(profile.__dict__, indent=2, sort_keys=True))
        return 0

    candidate_path = (
        Path(args.candidates)
        if args.candidates is not None
        else _latest_candidate_csv(Path(args.results_dir))
    )
    profile, scored, summary = audit_winner_iqr_alignment(
        Path(args.workbook),
        candidate_path,
        deciles=int(args.deciles),
        require_calibrated=not bool(args.include_uncalibrated),
    )

    payload = {
        "workbook": str(args.workbook),
        "candidates": str(candidate_path),
        "profile": winner_iqr_profile_to_dict(profile),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    display_cols = [
        c
        for c in [
            "symbol",
            "winner_range_bucket",
            "range_size_pct",
            "num_grids",
            "grid_spacing_pct",
            "profit_per_grid_pct",
            "grid_density_per_range_pct",
            "num_grids_in_iqr",
            "grid_spacing_in_iqr",
            "profit_per_grid_in_iqr",
            "tuple_in_iqr",
            "winner_iqr_binding_constraint",
        ]
        if c in scored.columns
    ]
    if display_cols:
        print(scored[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
