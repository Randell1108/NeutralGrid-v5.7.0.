"""
Empirical profile utilities for EV alignment and payoff-aware sizing.

Builds a lightweight profile from backtest result CSVs and exposes:
- Fill-rate scaling to align analytical EV with simulated fills.
- Linear EV correction against simulated net PnL.
- Generalized Kelly sizing inputs (avg win / avg loss, drawdown profile).
- Context-aware fallbacks (symbol/regime/global) for EV/fill/payoff selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, cast
import logging

import numpy as np
import pandas as pd

from neutralgrid.backtest.realism_governance import (
    LEGACY_REALISM_PROFILE,
    SHADOW_REALISM_PROFILES,
)
from neutralgrid.core.config import get_config
from neutralgrid.core.constants import BOT_HORIZON_HOURS

logger = logging.getLogger(__name__)
SubProfile = Dict[str, float]
FAST_PROFILE_DURATION_MAX_HOURS = 7.0
DEFAULT_SCANNER_LEVERAGE = 10
EMPIRICAL_PROFILE_FINGERPRINT_SCHEMA = (
    "empirical_profile_numeric_v2_realism_authority"
)


@dataclass(frozen=True)
class EmpiricalProfile:
    """Empirical backtest profile for EV alignment and sizing."""

    source: str
    samples: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    payoff_ratio_b: float
    p95_drawdown_pct: float
    fill_rate_scale: float
    ev_alignment_slope: float
    ev_alignment_intercept: float
    ev_alignment_r2: float
    ev_alignment_samples: int
    symbol_profiles: Dict[str, SubProfile]
    regime_profiles: Dict[str, SubProfile]


DEFAULT_PROFILE = EmpiricalProfile(
    source="default",
    samples=0,
    win_rate=0.50,
    avg_win_pct=3.0,
    avg_loss_pct=12.0,
    payoff_ratio_b=0.25,
    p95_drawdown_pct=12.0,
    fill_rate_scale=1.0,
    ev_alignment_slope=1.0,
    ev_alignment_intercept=0.0,
    ev_alignment_r2=0.0,
    ev_alignment_samples=0,
    symbol_profiles={},
    regime_profiles={},
)


def empirical_profile_fingerprint(profile: EmpiricalProfile) -> str:
    """Hash every profile value that can affect PnLRanker output.

    ``source`` is intentionally excluded: moving an identical input directory
    does not change the numeric EV contract. The schema must be bumped whenever
    profile interpretation changes without changing the dataclass values.
    """
    numeric_payload = asdict(profile)
    numeric_payload.pop("source", None)
    payload = {
        "schema": EMPIRICAL_PROFILE_FINGERPRINT_SCHEMA,
        "profile": numeric_payload,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _active_fraction(num_grids: float) -> float:
    """
    Active fraction used by scanner EV logic.

    Keep formula consistent with pnl_ranker.py to avoid drift.
    """
    return min(0.75, (float(num_grids) / 50.0) ** 0.5 * 0.5)


def _compute_profit_per_grid_pct(
    lower: float,
    upper: float,
    num_grids: int,
) -> float:
    """Per-grid profit percentage under Binance's geometric grid identity.

    Delegates to ``neutralgrid.grid.formulas.profit_per_grid_pct`` (single source
    of truth introduced by GRID_SYNCH.md Step 2). Uses the same effective fee
    scalar as the live calculator (``c = (maker_fee + close_fee_rate) / 2``).
    """
    if lower <= 0 or upper <= lower or num_grids <= 1:
        return 0.0

    from neutralgrid.grid.formulas import (
        BINANCE_DISPLAYED_INTERVALS,
        GEOMETRIC,
        profit_per_grid_pct,
    )

    cfg = get_config()
    maker_fee = float(cfg.grid.maker_fee)
    taker_fee = float(cfg.grid.taker_fee)
    close_fee_mode = str(getattr(cfg.grid, "close_fee_mode", "maker")).lower()
    close_fee_rate = maker_fee if close_fee_mode == "maker" else taker_fee
    c = max(0.0, (maker_fee + close_fee_rate) / 2.0)
    return profit_per_grid_pct(
        float(lower),
        float(upper),
        int(num_grids),
        GEOMETRIC,
        c,
        BINANCE_DISPLAYED_INTERVALS,
    )


def _analytic_ev_pct(
    *,
    profit_per_grid_pct: float,
    num_grids: int,
    survival_prob: float,
    funding_rate: float,
    horizon_hours: float,
    leverage: int,
    sl_pct: float,
    baseline_fills_per_hour: float,
    fill_rate_scale: float,
) -> float:
    """Analytical EV (%), matching pnl_ranker core logic."""
    af = _active_fraction(num_grids)
    expected_fills = baseline_fills_per_hour * fill_rate_scale * horizon_hours * af
    fill_revenue = survival_prob * expected_fills * (profit_per_grid_pct / 100.0)
    funding_cost = abs(funding_rate) * leverage * (horizon_hours / 8.0)
    boundary_loss = (1.0 - survival_prob) * abs(sl_pct)
    return float((fill_revenue - funding_cost - boundary_loss) * 100.0)


def _safe_num(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce")).replace([np.inf, -np.inf], np.nan)


def _trend_bucket(trend_prob: float) -> str:
    p = float(np.clip(trend_prob, 0.0, 1.0))
    if p < 0.33:
        return "low"
    if p < 0.66:
        return "mid"
    return "high"


def _filter_fast_horizon_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict fit-time profile rows to the live fast-horizon contract."""
    if "duration_hours" not in df.columns:
        return df.copy()

    duration = _safe_num(cast(pd.Series, df["duration_hours"]))
    mask = duration.gt(0.0) & duration.le(FAST_PROFILE_DURATION_MAX_HOURS)
    return cast(pd.DataFrame, df.loc[mask].copy())


def collect_authoritative_backtest_rows(backtest_dir: Path) -> pd.DataFrame:
    """Load legacy-authority rows, then deduplicate without shadow displacement."""
    frames: list[pd.DataFrame] = []
    historical_unstamped_rows = 0
    admitted_legacy_rows = 0
    excluded_shadow_rows = 0
    excluded_blank_rows = 0
    excluded_unknown_rows = 0

    if not backtest_dir.exists():
        return pd.DataFrame()

    for path in sorted(backtest_dir.glob("backtest_results_*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            logger.debug("Skipping unreadable backtest CSV %s: %s", path.name, exc)
            continue

        if "realism_profile" not in frame.columns:
            historical_unstamped_rows += len(frame)
            frames.append(frame)
            continue

        profile = (
            cast(pd.Series, frame["realism_profile"])
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        legacy_mask = profile.eq(LEGACY_REALISM_PROFILE)
        blank_mask = profile.eq("")
        shadow_mask = profile.isin(SHADOW_REALISM_PROFILES)
        unknown_mask = ~(legacy_mask | blank_mask | shadow_mask)
        admitted_legacy_rows += int(legacy_mask.sum())
        excluded_shadow_rows += int(shadow_mask.sum())
        excluded_blank_rows += int(blank_mask.sum())
        excluded_unknown_rows += int(unknown_mask.sum())
        admitted = cast(pd.DataFrame, frame.loc[legacy_mask].copy())
        if not admitted.empty:
            frames.append(admitted)

    if not frames:
        return pd.DataFrame()

    logger.info(
        "Empirical profile realism admission: historical_unstamped=%d, "
        "legacy=%d, excluded_shadow=%d, excluded_blank=%d, excluded_unknown=%d",
        historical_unstamped_rows,
        admitted_legacy_rows,
        excluded_shadow_rows,
        excluded_blank_rows,
        excluded_unknown_rows,
    )

    df = pd.concat(frames, ignore_index=True)

    # Deterministic dedup: prevent historical re-runs from inflating statistics.
    rows_before = len(df)
    if "candidate_id" in df.columns:
        df = df.drop_duplicates(subset=["candidate_id"], keep="last")
    elif {"symbol", "start_time_utc"}.issubset(set(df.columns)):
        df = df.drop_duplicates(subset=["symbol", "start_time_utc"], keep="last")

    rows_after = len(df)
    if rows_before != rows_after:
        logger.info(
            "Empirical profile dedup: %d → %d rows (%d duplicates removed)",
            rows_before,
            rows_after,
            rows_before - rows_after,
        )

    return df.reset_index(drop=True)


def _fit_linear_alignment(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Fit y ~= a + b*x with robust fallbacks."""
    if len(x) < 6:
        return {"slope": 1.0, "intercept": 0.0, "r2": 0.0, "n": float(len(x))}
    if float(np.nanstd(x)) < 1e-9:
        return {"slope": 1.0, "intercept": 0.0, "r2": 0.0, "n": float(len(x))}

    try:
        slope, intercept = np.polyfit(x, y, deg=1)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
            "n": float(len(x)),
        }
    except Exception:
        return {"slope": 1.0, "intercept": 0.0, "r2": 0.0, "n": float(len(x))}


def _profile_from_rows(df: pd.DataFrame) -> SubProfile:
    """Compute profile statistics from a backtest subset."""
    defaults: SubProfile = {
        "samples": float(DEFAULT_PROFILE.samples),
        "win_rate": float(DEFAULT_PROFILE.win_rate),
        "avg_win_pct": float(DEFAULT_PROFILE.avg_win_pct),
        "avg_loss_pct": float(DEFAULT_PROFILE.avg_loss_pct),
        "payoff_ratio_b": float(DEFAULT_PROFILE.payoff_ratio_b),
        "p95_drawdown_pct": float(DEFAULT_PROFILE.p95_drawdown_pct),
        "fill_rate_scale": float(DEFAULT_PROFILE.fill_rate_scale),
        "ev_alignment_slope": float(DEFAULT_PROFILE.ev_alignment_slope),
        "ev_alignment_intercept": float(DEFAULT_PROFILE.ev_alignment_intercept),
        "ev_alignment_r2": float(DEFAULT_PROFILE.ev_alignment_r2),
        "ev_alignment_samples": float(DEFAULT_PROFILE.ev_alignment_samples),
    }

    if df.empty or "net_pnl_pct" not in df.columns:
        return defaults

    df = _filter_fast_horizon_rows(df)
    if df.empty:
        return defaults

    net = _safe_num(cast(pd.Series, df["net_pnl_pct"])).dropna()
    if net.empty:
        return defaults

    wins = cast(pd.Series, net[net > 0.0])
    losses = cast(pd.Series, net[net < 0.0])

    avg_win = float(wins.mean()) if not wins.empty else defaults["avg_win_pct"]
    avg_loss = float(abs(losses.mean())) if not losses.empty else defaults["avg_loss_pct"]
    avg_loss = max(avg_loss, 1e-6)
    payoff_b = float(avg_win / avg_loss)
    win_rate = float((net > 0.0).mean())

    if "max_drawdown_pct" in df.columns:
        dd = _safe_num(cast(pd.Series, df["max_drawdown_pct"])).dropna()
        p95_dd = (
            float(np.nanpercentile(dd, 95))
            if not dd.empty
            else defaults["p95_drawdown_pct"]
        )
    else:
        p95_dd = defaults["p95_drawdown_pct"]

    fill_scale = 1.0
    if {"round_trips", "duration_hours", "num_grids"}.issubset(set(df.columns)):
        rt = _safe_num(cast(pd.Series, df["round_trips"]))
        dur = _safe_num(cast(pd.Series, df["duration_hours"]))
        ngr = _safe_num(cast(pd.Series, df["num_grids"]))
        mask = (rt > 0) & (dur > 0.0) & (ngr > 0.0)
        if mask.any():
            rt = cast(pd.Series, rt[mask])
            dur = cast(pd.Series, dur[mask])
            ngr = cast(pd.Series, ngr[mask])
            realized_fph = rt / dur
            expected_fph = 2.0 * ngr.map(_active_fraction)
            ratio = cast(pd.Series, (realized_fph / expected_fph)).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                fill_scale = float(np.nanmedian(ratio))
                fill_scale = float(np.clip(fill_scale, 0.30, 2.50))

    required = {
        "grid_lower",
        "grid_upper",
        "num_grids",
        "survival_prob",
        "funding_rate",
        "net_pnl_pct",
    }
    ev_align: Dict[str, float] = {"slope": 1.0, "intercept": 0.0, "r2": 0.0, "n": 0.0}
    if required.issubset(set(df.columns)):
        tmp = df.copy()
        for c in required:
            tmp[c] = _safe_num(cast(pd.Series, tmp[c]))

        if "profit_per_grid_pct" not in tmp.columns:
            def _row_profit(row: Any) -> float:
                try:
                    return _compute_profit_per_grid_pct(
                        float(row["grid_lower"]),
                        float(row["grid_upper"]),
                        int(float(row["num_grids"])),
                    )
                except Exception:
                    return float("nan")

            tmp["profit_per_grid_pct"] = tmp.apply(_row_profit, axis=1)
        else:
            tmp["profit_per_grid_pct"] = _safe_num(cast(pd.Series, tmp["profit_per_grid_pct"]))

        if "leverage" in tmp.columns:
            tmp["leverage"] = _safe_num(cast(pd.Series, tmp["leverage"])).fillna(DEFAULT_SCANNER_LEVERAGE)
        else:
            tmp["leverage"] = float(DEFAULT_SCANNER_LEVERAGE)

        cfg = get_config()
        sl_pct = float(cfg.grid.stop_loss_pct)
        horizon_hours = float(BOT_HORIZON_HOURS)
        baseline_fills_per_hour = 2.0

        tmp["ev_raw_pct"] = tmp.apply(
            lambda r: _analytic_ev_pct(
                profit_per_grid_pct=float(r["profit_per_grid_pct"]),
                num_grids=int(float(r["num_grids"])),
                survival_prob=float(r["survival_prob"]),
                funding_rate=float(r["funding_rate"]),
                horizon_hours=horizon_hours,
                leverage=int(float(r["leverage"])),
                sl_pct=sl_pct,
                baseline_fills_per_hour=baseline_fills_per_hour,
                fill_rate_scale=float(fill_scale),
            ),
            axis=1,
        )

        usable = tmp[["ev_raw_pct", "net_pnl_pct"]].dropna()
        if len(usable) >= 6:
            x = cast(pd.Series, usable["ev_raw_pct"]).to_numpy(dtype=float)
            y = cast(pd.Series, usable["net_pnl_pct"]).to_numpy(dtype=float)
            ev_align = _fit_linear_alignment(x, y)

    return {
        "samples": float(len(net)),
        "win_rate": float(win_rate),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "payoff_ratio_b": float(payoff_b),
        "p95_drawdown_pct": float(p95_dd),
        "fill_rate_scale": float(fill_scale),
        "ev_alignment_slope": float(ev_align["slope"]),
        "ev_alignment_intercept": float(ev_align["intercept"]),
        "ev_alignment_r2": float(ev_align["r2"]),
        "ev_alignment_samples": float(ev_align["n"]),
    }


def _global_as_subprofile(profile: EmpiricalProfile) -> SubProfile:
    return {
        "samples": float(profile.samples),
        "win_rate": float(profile.win_rate),
        "avg_win_pct": float(profile.avg_win_pct),
        "avg_loss_pct": float(profile.avg_loss_pct),
        "payoff_ratio_b": float(profile.payoff_ratio_b),
        "p95_drawdown_pct": float(profile.p95_drawdown_pct),
        "fill_rate_scale": float(profile.fill_rate_scale),
        "ev_alignment_slope": float(profile.ev_alignment_slope),
        "ev_alignment_intercept": float(profile.ev_alignment_intercept),
        "ev_alignment_r2": float(profile.ev_alignment_r2),
        "ev_alignment_samples": float(profile.ev_alignment_samples),
    }


def _select_subprofile(
    profile: EmpiricalProfile,
    *,
    symbol: str | None,
    trend_prob: float | None,
    min_samples: int,
    metric_key: str,
    strict_min: bool = True,
) -> tuple[SubProfile, str]:
    symbol_key = str(symbol).upper() if symbol else None
    if symbol_key:
        sp = profile.symbol_profiles.get(symbol_key)
        if sp is not None and (
            not strict_min or float(sp.get(metric_key, 0.0)) >= float(min_samples)
        ):
            return sp, "symbol"

    if trend_prob is not None:
        bucket = _trend_bucket(float(trend_prob))
        rp = profile.regime_profiles.get(bucket)
        if rp is not None and (
            not strict_min or float(rp.get(metric_key, 0.0)) >= float(min_samples)
        ):
            return rp, f"regime:{bucket}"

    return _global_as_subprofile(profile), "global"


def build_empirical_profile(backtest_dir: Path) -> EmpiricalProfile:
    """
    Build empirical profile from backtest results.

    Uses historical simulation outputs to align analytical EV and infer
    payout asymmetry for generalized Kelly sizing.
    """
    df = collect_authoritative_backtest_rows(Path(backtest_dir))
    if df.empty or "net_pnl_pct" not in df.columns:
        return DEFAULT_PROFILE

    base = _profile_from_rows(df)
    if float(base.get("samples", 0.0)) <= 0:
        return DEFAULT_PROFILE

    symbol_profiles: Dict[str, SubProfile] = {}
    if "symbol" in df.columns:
        tmp = df.copy()
        tmp["_symbol_key"] = tmp["symbol"].astype(str).str.upper()
        for sym, grp in tmp.groupby("_symbol_key"):
            if not sym or sym == "NAN":
                continue
            sub = _profile_from_rows(grp.drop(columns=["_symbol_key"]))
            # Keep small threshold so diagnostics can still be observed.
            if float(sub.get("samples", 0.0)) >= 8.0:
                symbol_profiles[str(sym)] = sub

    regime_profiles: Dict[str, SubProfile] = {}
    trend_col = None
    if "trend_prob" in df.columns:
        trend_col = "trend_prob"
    elif "hmm_trend_prob" in df.columns:
        trend_col = "hmm_trend_prob"

    if trend_col is not None:
        tmp = df.copy()
        trend_vals = _safe_num(cast(pd.Series, tmp[trend_col]))
        tmp["_trend_bucket"] = trend_vals.apply(
            lambda x: _trend_bucket(float(x)) if pd.notna(x) else np.nan
        )
        for bucket in ("low", "mid", "high"):
            grp = tmp.loc[tmp["_trend_bucket"] == bucket]
            if grp.empty:
                continue
            sub = _profile_from_rows(grp.drop(columns=["_trend_bucket"]))
            if float(sub.get("samples", 0.0)) >= 10.0:
                regime_profiles[bucket] = sub

    return EmpiricalProfile(
        source=f"{Path(backtest_dir)}|duration_hours<={FAST_PROFILE_DURATION_MAX_HOURS}",
        samples=int(float(base["samples"])),
        win_rate=float(base["win_rate"]),
        avg_win_pct=float(base["avg_win_pct"]),
        avg_loss_pct=float(base["avg_loss_pct"]),
        payoff_ratio_b=float(base["payoff_ratio_b"]),
        p95_drawdown_pct=float(base["p95_drawdown_pct"]),
        fill_rate_scale=float(base["fill_rate_scale"]),
        ev_alignment_slope=float(base["ev_alignment_slope"]),
        ev_alignment_intercept=float(base["ev_alignment_intercept"]),
        ev_alignment_r2=float(base["ev_alignment_r2"]),
        ev_alignment_samples=int(float(base["ev_alignment_samples"])),
        symbol_profiles=symbol_profiles,
        regime_profiles=regime_profiles,
    )


@lru_cache(maxsize=8)
def load_empirical_profile_cached(backtest_dir: str) -> EmpiricalProfile:
    """Cached profile loader keyed by directory string."""
    try:
        return build_empirical_profile(Path(backtest_dir))
    except Exception as exc:
        logger.warning("Empirical profile build failed: %s", exc)
        return DEFAULT_PROFILE


def align_ev_with_profile(
    ev_raw_pct: float,
    profile: EmpiricalProfile,
    min_samples: int = 20,
) -> float:
    """Apply linear EV alignment learned from simulated outcomes."""
    details = align_ev_with_profile_context(
        ev_raw_pct=ev_raw_pct,
        profile=profile,
        symbol=None,
        trend_prob=None,
        min_samples=min_samples,
    )
    return float(details["ev_aligned"])


def resolve_fill_rate_scale(
    profile: EmpiricalProfile,
    *,
    symbol: str | None = None,
    trend_prob: float | None = None,
    min_samples: int = 20,
) -> Dict[str, float | str]:
    """
    Resolve fill-rate scale using symbol/regime/global context.
    """
    chosen, scope = _select_subprofile(
        profile,
        symbol=symbol,
        trend_prob=trend_prob,
        min_samples=min_samples,
        metric_key="samples",
        strict_min=False,
    )
    samples = float(chosen.get("samples", 0.0))
    raw_scale = float(chosen.get("fill_rate_scale", 1.0))
    if min_samples > 0 and samples < float(min_samples):
        # Soft alignment under low sample count: shrink toward neutral scale=1.0.
        w = float(np.clip(samples / float(min_samples), 0.0, 1.0))
        scale = 1.0 + w * (raw_scale - 1.0)
        scope_out = f"{scope}_shrunk"
    else:
        scale = raw_scale
        scope_out = scope
    return {
        "fill_rate_scale": float(scale),
        "scope": scope_out,
        "samples": float(samples),
    }


def align_ev_with_profile_context(
    *,
    ev_raw_pct: float,
    profile: EmpiricalProfile,
    symbol: str | None = None,
    trend_prob: float | None = None,
    min_samples: int = 20,
) -> Dict[str, float | str]:
    """
    Apply EV alignment with symbol/regime/global context selection.
    """
    chosen, scope = _select_subprofile(
        profile,
        symbol=symbol,
        trend_prob=trend_prob,
        min_samples=min_samples,
        metric_key="ev_alignment_samples",
        strict_min=False,
    )
    n = float(chosen.get("ev_alignment_samples", 0.0))
    slope = float(chosen.get("ev_alignment_slope", 1.0))
    intercept = float(chosen.get("ev_alignment_intercept", 0.0))
    r2 = float(chosen.get("ev_alignment_r2", 0.0))

    if n <= 0:
        return {
            "ev_aligned": float(ev_raw_pct),
            "scope": "none",
            "samples": float(n),
            "slope": 1.0,
            "intercept": 0.0,
            "r2": 0.0,
        }

    aligned_raw = intercept + slope * float(ev_raw_pct)
    if min_samples > 0 and n < float(min_samples):
        # Soft alignment under low sample count: shrink toward identity mapping.
        w = float(np.clip(n / float(min_samples), 0.0, 1.0))
        aligned = float(ev_raw_pct) + w * (aligned_raw - float(ev_raw_pct))
        scope_out = f"{scope}_shrunk"
    else:
        aligned = aligned_raw
        scope_out = scope
    return {
        "ev_aligned": float(aligned),
        "scope": scope_out,
        "samples": float(n),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def resolve_payoff_context(
    profile: EmpiricalProfile,
    *,
    symbol: str | None = None,
    trend_prob: float | None = None,
    min_samples: int = 20,
) -> Dict[str, float | str]:
    """
    Resolve payoff/drawdown context for generalized Kelly sizing.
    """
    chosen, scope = _select_subprofile(
        profile,
        symbol=symbol,
        trend_prob=trend_prob,
        min_samples=min_samples,
        metric_key="samples",
        strict_min=False,
    )
    samples = float(chosen.get("samples", 0.0))
    if min_samples > 0 and samples < float(min_samples):
        # Shrink toward global statistics under small samples.
        w = float(np.clip(samples / float(min_samples), 0.0, 1.0))
        g = _global_as_subprofile(profile)
        avg_win = float(g["avg_win_pct"]) + w * (float(chosen.get("avg_win_pct", g["avg_win_pct"])) - float(g["avg_win_pct"]))
        avg_loss = float(g["avg_loss_pct"]) + w * (float(chosen.get("avg_loss_pct", g["avg_loss_pct"])) - float(g["avg_loss_pct"]))
        payoff_b = float(g["payoff_ratio_b"]) + w * (float(chosen.get("payoff_ratio_b", g["payoff_ratio_b"])) - float(g["payoff_ratio_b"]))
        p95_dd = float(g["p95_drawdown_pct"]) + w * (float(chosen.get("p95_drawdown_pct", g["p95_drawdown_pct"])) - float(g["p95_drawdown_pct"]))
        scope_out = f"{scope}_shrunk"
    else:
        avg_win = float(chosen.get("avg_win_pct", profile.avg_win_pct))
        avg_loss = float(chosen.get("avg_loss_pct", profile.avg_loss_pct))
        payoff_b = float(chosen.get("payoff_ratio_b", profile.payoff_ratio_b))
        p95_dd = float(chosen.get("p95_drawdown_pct", profile.p95_drawdown_pct))
        scope_out = scope
    return {
        "scope": scope_out,
        "samples": float(samples),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "payoff_ratio_b": float(payoff_b),
        "p95_drawdown_pct": float(p95_dd),
    }


def _optimize_fractional_kelly_multiplier(
    *,
    raw_fraction: float,
    p: float,
    b: float,
    drawdown_ref_pct: float,
    drawdown_tolerance_pct: float,
    sweep_min: float,
    sweep_max: float,
    sweep_step: float,
) -> Dict[str, float]:
    """
    Optimize fractional Kelly multiplier with drawdown-constrained log-growth.

    Objective:
      maximize  p*log(1 + f*b) + q*log(1 - f)
    where f = raw_fraction * multiplier, subject to estimated drawdown <= tolerance.
    """
    q = 1.0 - p
    raw = max(0.0, float(raw_fraction))
    b_safe = max(float(b), 1e-9)
    dd_ref = max(float(drawdown_ref_pct), 1e-9)
    dd_tol = max(float(drawdown_tolerance_pct), 1e-9)

    lo = max(0.0, float(sweep_min))
    hi = max(lo, float(sweep_max))
    step = max(float(sweep_step), 1e-6)
    multipliers = np.arange(lo, hi + (step * 0.5), step, dtype=float)
    if multipliers.size == 0:
        multipliers = np.array([0.5], dtype=float)

    best_mult = float(multipliers[0])
    best_growth = float("-inf")
    best_dd = 0.0
    feasible_count = 0
    evaluated_count = 0

    for mult in multipliers:
        evaluated_count += 1
        f = float(np.clip(raw * float(mult), 0.0, 0.95))
        if f <= 0.0:
            growth = 0.0
        else:
            if 1.0 - f <= 0.0:
                continue
            growth = p * float(np.log1p(f * b_safe)) + q * float(np.log1p(-f))
        est_dd = dd_ref * f
        if est_dd > dd_tol:
            continue
        feasible_count += 1
        if growth > best_growth:
            best_growth = growth
            best_mult = float(mult)
            best_dd = float(est_dd)

    if feasible_count == 0:
        # Fall back to tolerance-implied cap on the raw Kelly exposure.
        capped_fraction = min(raw, dd_tol / dd_ref)
        best_mult = 0.0 if raw <= 0 else float(np.clip(capped_fraction / raw, 0.0, hi))
        best_growth = 0.0
        best_dd = dd_ref * capped_fraction

    return {
        "fractional_multiplier": float(best_mult),
        "objective_growth": float(best_growth),
        "estimated_drawdown_pct": float(best_dd),
        "feasible_points": float(feasible_count),
        "evaluated_points": float(evaluated_count),
    }


def generalized_kelly_details(
    *,
    meta_prob: float,
    profile: EmpiricalProfile,
    fractional_kelly: float,
    optimize_fractional: bool = False,
    fractional_sweep_min: float = 0.10,
    fractional_sweep_max: float = 1.00,
    fractional_sweep_step: float = 0.05,
    drawdown_tolerance_pct: float,
    volatility_target_pct: float,
    volatility_proxy_pct: float | None = None,
    symbol: str | None = None,
    trend_prob: float | None = None,
    min_samples: int = 20,
) -> Dict[str, float | str]:
    """
    Compute generalized Kelly details:
        f* = (p*b - q)/b, b = avg_win/avg_loss, q = 1-p
    plus drawdown and volatility targeting scales.
    """
    p = float(np.clip(meta_prob, 0.0, 1.0))
    q = 1.0 - p
    payoff_ctx = resolve_payoff_context(
        profile,
        symbol=symbol,
        trend_prob=trend_prob,
        min_samples=min_samples,
    )
    b = max(float(payoff_ctx["payoff_ratio_b"]), 1e-6)
    raw = (p * b - q) / b

    dd_ref = max(float(payoff_ctx["p95_drawdown_pct"]), 1e-6)
    frac = max(0.0, float(fractional_kelly))
    fractional_mode = "fixed"
    sweep_growth = 0.0
    sweep_estimated_dd = 0.0
    sweep_feasible = 0.0
    sweep_evaluated = 0.0
    if optimize_fractional:
        sweep = _optimize_fractional_kelly_multiplier(
            raw_fraction=float(raw),
            p=p,
            b=b,
            drawdown_ref_pct=dd_ref,
            drawdown_tolerance_pct=float(drawdown_tolerance_pct),
            sweep_min=float(fractional_sweep_min),
            sweep_max=float(fractional_sweep_max),
            sweep_step=float(fractional_sweep_step),
        )
        frac = float(sweep["fractional_multiplier"])
        fractional_mode = "optimized"
        sweep_growth = float(sweep["objective_growth"])
        sweep_estimated_dd = float(sweep["estimated_drawdown_pct"])
        sweep_feasible = float(sweep["feasible_points"])
        sweep_evaluated = float(sweep["evaluated_points"])

    kelly_fraction = max(0.0, raw * frac)

    # In optimized mode, drawdown constraint is already enforced by the sweep.
    if optimize_fractional:
        drawdown_scale = 1.0
    else:
        drawdown_scale = float(np.clip(drawdown_tolerance_pct / dd_ref, 0.05, 1.0))

    vol_scale = 1.0
    if volatility_proxy_pct is not None and volatility_proxy_pct > 0:
        vol_scale = float(np.clip(volatility_target_pct / volatility_proxy_pct, 0.05, 1.0))

    final_fraction = max(0.0, kelly_fraction * drawdown_scale * vol_scale)

    return {
        "meta_prob": p,
        "payoff_ratio_b": b,
        "raw_fraction": float(raw),
        "fractional_multiplier": float(frac),
        "fractional_kelly": float(kelly_fraction),
        "fractional_mode": str(fractional_mode),
        "drawdown_scale": float(drawdown_scale),
        "volatility_scale": float(vol_scale),
        "final_fraction": float(final_fraction),
        "avg_win_pct": float(payoff_ctx["avg_win_pct"]),
        "avg_loss_pct": float(payoff_ctx["avg_loss_pct"]),
        "profile_samples": float(payoff_ctx["samples"]),
        "profile_scope": str(payoff_ctx["scope"]),
        "profile_scope_samples": float(payoff_ctx["samples"]),
        "sweep_objective_growth": float(sweep_growth),
        "sweep_estimated_drawdown_pct": float(sweep_estimated_dd),
        "sweep_feasible_points": float(sweep_feasible),
        "sweep_evaluated_points": float(sweep_evaluated),
    }
