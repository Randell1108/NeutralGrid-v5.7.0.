#!/usr/bin/env python3
"""
Retrain Pattern Profile and Profile Model.

Trains (or retrains) the pattern profile and profile model from
the enhanced expired bots Excel file.

This script generates:
- data/profile/pattern_profile.json
- data/profile/profile_model.json
- data/profile/profile_gate.json

Usage:
    python retrain_scanner.py [--input FILE] [--output-dir DIR]

Examples:
    # Train with default input file
    python retrain_scanner.py

    # Train with custom input file
    python retrain_scanner.py --input data/custom_bots.xlsx

    # Save to custom output directory
    python retrain_scanner.py --output-dir models/v1.0.0/profile/

Input Format:
    Excel file with sheets:
    - "Entry Validation Metrics" - Entry features (ADX, RSI, etc.)
    - "Performance Risk-Adjusted" - PnL, profit factor, duration
    - "Market and Volatility" - Funding rates, OI, ATR, volume

Output:
    - pattern_profile.json: Statistical profile of winning bots
    - profile_model.json: Gaussian discriminant classifier
    - profile_gate.json: Hard threshold gates (ADX, RSI bounds)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import cast

import pandas as pd

# Ensure local `src/` package is used when running this script directly
# (ERR-077: a stale editable install elsewhere must not shadow this tree).
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.data.binance_vision.pipeline import ensure_kline_store
from neutralgrid.data.binance_vision.store import load_parquet_range, parquet_root
from neutralgrid.data.funding_rate import (
    HistoricalFundingLoader,
    realized_funding_carry_proxy_next_hours,
)
from neutralgrid.scanner._xlsx_io import CANONICAL_SINGLE_SHEET_NAME
from neutralgrid.scanner.feature_extractor import compute_profile_feature_block
from neutralgrid.scanner.pattern_profile import (
    DEFAULT_FEATURES,
    PatternProfile,
    build_profile_from_enhanced_xlsx,
)
from neutralgrid.scanner.profile_model import (
    ProfileModel,
    load_profile_model,
    save_profile_model,
    train_profile_model_from_enhanced_xlsx,
)
from neutralgrid.scanner.profile_model_walkforward import (
    promote_profile_version,
    resolve_active_profile_model_path,
    resolve_latest_evaluated_holdout_end,
    walkforward_evaluate,
)
from neutralgrid.validation.profile_gate import ProfileGate


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_scanner")


PROFILE_15M_LOOKBACK_BARS = 100
PROFILE_15M_LOOKBACK_MS = PROFILE_15M_LOOKBACK_BARS * 15 * 60 * 1000
PROFILE_1M_LOOKBACK_MS = 120 * 60 * 1000
PROFILE_FUNDING_HORIZON_HOURS = 7.0
PROFILE_FUNDING_HORIZON_MS = int(PROFILE_FUNDING_HORIZON_HOURS * 3600 * 1000)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain pattern profile and profile model from winners data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/new_expired_bots.xlsx",
        help="Input Excel file with bot performance data (default: data/new_expired_bots.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/profile/",
        help="Output directory for trained artifacts (default: data/profile/)",
    )
    parser.add_argument(
        "--max-duration-hours",
        type=float,
        default=7.0,
        help="Maximum bot duration in hours for bounded training universe (default: 7.0). "
             "Only bots with 0 <= duration_hours < this value are eligible for training.",
    )
    parser.add_argument(
        "--min-profit-factor",
        type=float,
        default=1.5,
        help="Minimum profit factor for winner classification (default: 1.5)",
    )
    parser.add_argument(
        "--min-avg-profit-per-grid",
        type=float,
        default=get_config().grid.profit_grid_min_pct_static_fallback,
        help=f"Minimum average profit per grid for training filter (default: {get_config().grid.profit_grid_min_pct_static_fallback})",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.68,  # SCANNER-TOP-QUANTILE-0.68 (CHANGELOG.md): user-authorized 2026-05-06 to clear max(30, 3*4)=30 floor (26→33 winners)
        help="PnL quantile threshold for winners (default: 0.68, top 32%% — see CHANGELOG SCANNER-TOP-QUANTILE-0.68)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.30,
        help="Covariance shrinkage for profile model (default: 0.30)",
    )
    parser.add_argument(
        "--incumbent-shrinkage",
        type=float,
        default=None,
        help=(
            "Explicit shrinkage used to reproduce an existing incumbent when "
            "its artifact lacks shrinkage provenance. Required for a governed "
            "comparison in that case; never inferred silently."
        ),
    )
    parser.add_argument(
        "--gate-widen",
        type=float,
        default=0.15,
        help="Widening factor for profile gate thresholds (default: 0.15, i.e., 15%% wider)",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip generating profile_gate.json",
    )
    return parser.parse_args()


def _to_utc_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _prepare_history_frame(df: pd.DataFrame, *, interval: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    prepared = df.copy()
    if "open_time" in prepared.columns:
        prepared["open_time"] = _to_utc_series(prepared["open_time"])
    if "close_time" in prepared.columns:
        prepared["close_time"] = _to_utc_series(prepared["close_time"])
    elif "open_time" in prepared.columns:
        interval_minutes = {"15m": 15, "1m": 1}.get(interval)
        if interval_minutes is None:
            raise ValueError(f"Unsupported interval for close_time inference: {interval}")
        prepared["close_time"] = prepared["open_time"] + pd.to_timedelta(
            interval_minutes, unit="m"
        )
    for col in ("open", "high", "low", "close", "volume", "quote_volume"):
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    return prepared.sort_values("open_time").reset_index(drop=True)


async def _ensure_and_load_history(
    symbol: str,
    *,
    interval: str,
    start_ms: int,
    end_ms: int,
    min_years: float,
) -> pd.DataFrame:
    cache_root = Path(get_config().base_dir) / "data" / "cache" / "klines"
    root = parquet_root(str(cache_root), "futures_um", symbol, interval)
    rebuild_from_scratch = False
    try:
        history = _prepare_history_frame(
            load_parquet_range(root, start_ms=start_ms, end_ms=end_ms),
            interval=interval,
        )
    except Exception as exc:
        logger.warning(
            "Unreadable cached %s history for %s at %s: %s. Rebuilding cache slice.",
            interval,
            symbol,
            root,
            exc,
        )
        rebuild_from_scratch = True
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        history = pd.DataFrame()

    start_ts = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
    have_coverage = (
        not history.empty
        and history["open_time"].min() <= start_ts
        and history["close_time"].max() >= end_ts
    )
    if rebuild_from_scratch or not have_coverage:
        logger.info(
            "Ensuring %s %s history in cache for %s -> %s",
            symbol,
            interval,
            start_ts.isoformat(),
            end_ts.isoformat(),
        )
        try:
            await ensure_kline_store(
                symbol=symbol,
                interval=interval,
                market="futures_um",
                min_years=min_years,
                cache_root=str(cache_root),
                mode="backfill" if rebuild_from_scratch else "auto",
                force=rebuild_from_scratch,
            )
        except Exception as exc:
            logger.warning(
                "ensure_kline_store failed for %s %s: %s. "
                "Proceeding with whatever cached history exists; uncovered rows "
                "will retain NaN profile features.",
                symbol,
                interval,
                exc,
            )
        try:
            history = _prepare_history_frame(
                load_parquet_range(root, start_ms=start_ms, end_ms=end_ms),
                interval=interval,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load rebuilt %s history for %s: %s. Returning empty history.",
                interval,
                symbol,
                exc,
            )
            return pd.DataFrame()

    if history.empty:
        logger.warning(
            "No cached %s history available for %s in %s -> %s. "
            "Backfill will leave the profile features missing for uncovered rows.",
            interval,
            symbol,
            start_ts.isoformat(),
            end_ts.isoformat(),
        )
        return history

    covered = (
        history["open_time"].min() <= start_ts
        and history["close_time"].max() >= end_ts
    )
    if not covered:
        logger.warning(
            "Partial %s history for %s covers %s -> %s, requested %s -> %s. "
            "Rows outside the available window will retain NaN profile features.",
            interval,
            symbol,
            history["open_time"].min().isoformat(),
            history["close_time"].max().isoformat(),
            start_ts.isoformat(),
            end_ts.isoformat(),
        )
    return history


async def _backfill_profile_features_async(
    input_path: Path,
    *,
    max_duration_hours: float = 7.0,
) -> None:
    with pd.ExcelFile(input_path) as xl:
        sheet_name = (
            CANONICAL_SINGLE_SHEET_NAME
            if CANONICAL_SINGLE_SHEET_NAME in xl.sheet_names
            else str(xl.sheet_names[0])
        )
        general_df = pd.read_excel(xl, sheet_name=sheet_name)

    general_df.columns = [str(c).strip() for c in general_df.columns]
    if "symbol" not in general_df.columns or "start_time_utc" not in general_df.columns:
        raise ValueError(
            f"{sheet_name} must contain 'symbol' and 'start_time_utc' for profile backfill."
        )

    start_times = pd.to_datetime(general_df["start_time_utc"], utc=True, errors="coerce")
    symbols = general_df["symbol"].astype(str).str.upper()
    valid_mask = symbols.ne("") & start_times.notna()
    if not bool(valid_mask.any()):
        raise ValueError(f"{sheet_name} contains no valid symbol/start_time_utc rows.")

    eligible_mask = valid_mask.copy()
    if "duration_hours" in general_df.columns:
        duration_hours = pd.to_numeric(general_df["duration_hours"], errors="coerce")
        eligible_mask = eligible_mask & duration_hours.ge(0.0) & duration_hours.lt(max_duration_hours)
    if "profit_factor" in general_df.columns and "pnl_pct" in general_df.columns:
        pf = pd.to_numeric(general_df["profit_factor"], errors="coerce")
        pnl = pd.to_numeric(general_df["pnl_pct"], errors="coerce")
        eligible_mask = eligible_mask & pf.notna() & pnl.notna()
    if not bool(eligible_mask.any()):
        raise ValueError(
            f"{sheet_name} contains no bounded/labeled rows eligible for profile backfill."
        )

    logger.info(
        "Profile backfill eligible rows: %d of %d valid General rows",
        int(eligible_mask.sum()),
        int(valid_mask.sum()),
    )

    client = BinanceClient()
    funding_loader = HistoricalFundingLoader(client)
    history_15m: dict[str, pd.DataFrame] = {}
    history_1m: dict[str, pd.DataFrame] = {}
    funding_history: dict[str, list[dict]] = {}

    try:
        for symbol in sorted(set(symbols[eligible_mask].tolist())):
            symbol_mask = eligible_mask & symbols.eq(symbol)
            symbol_start_times = start_times[symbol_mask]
            min_entry_ms = int(cast(pd.Timestamp, symbol_start_times.min()).timestamp() * 1000)
            max_entry_ms = int(cast(pd.Timestamp, symbol_start_times.max()).timestamp() * 1000)
            earliest_needed_ms = min(
                min_entry_ms - PROFILE_15M_LOOKBACK_MS,
                min_entry_ms - PROFILE_1M_LOOKBACK_MS,
            )
            symbol_history_years = max(
                0.01,
                float(max_entry_ms - earliest_needed_ms) / (365.25 * 24 * 3600 * 1000)
                + 0.01,
            )

            history_15m[symbol] = await _ensure_and_load_history(
                symbol,
                interval="15m",
                start_ms=min_entry_ms - PROFILE_15M_LOOKBACK_MS,
                end_ms=max_entry_ms,
                min_years=symbol_history_years,
            )
            history_1m[symbol] = await _ensure_and_load_history(
                symbol,
                interval="1m",
                start_ms=min_entry_ms - PROFILE_1M_LOOKBACK_MS,
                end_ms=max_entry_ms,
                min_years=symbol_history_years,
            )
            funding_history[symbol] = await funding_loader.fetch_raw(
                symbol,
                start_time_ms=min_entry_ms,
                end_time_ms=max_entry_ms + PROFILE_FUNDING_HORIZON_MS,
            )
    finally:
        await client.close()

    for feature_name in DEFAULT_FEATURES:
        if feature_name not in general_df.columns:
            general_df[feature_name] = pd.NA

    for idx, row in general_df.loc[eligible_mask].iterrows():
        symbol = str(row.get("symbol") or "").upper()
        entry_ts = pd.to_datetime(row.get("start_time_utc"), utc=True, errors="coerce")
        if not symbol or pd.isna(entry_ts):
            continue

        entry_ms = int(cast(pd.Timestamp, entry_ts).timestamp() * 1000)
        df_15m_pre = history_15m[symbol].loc[
            history_15m[symbol]["close_time"] < entry_ts
        ].tail(PROFILE_15M_LOOKBACK_BARS)
        df_1m_pre = history_1m[symbol].loc[
            history_1m[symbol]["close_time"] < entry_ts
        ].tail(120)
        funding_carry = realized_funding_carry_proxy_next_hours(
            funding_history.get(symbol, []),
            entry_ms,
            horizon_hours=PROFILE_FUNDING_HORIZON_HOURS,
        )

        feature_block = compute_profile_feature_block(
            klines_15m=df_15m_pre,
            klines_1m=df_1m_pre,
            funding_carry_expected_next_7h=funding_carry,
        )
        for feature_name, feature_value in feature_block.items():
            general_df.at[idx, feature_name] = feature_value

    with pd.ExcelWriter(
        input_path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        general_df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(
        "Backfilled %d profile columns onto %s sheet '%s'",
        len(DEFAULT_FEATURES),
        input_path,
        sheet_name,
    )


def backfill_profile_features(
    input_path: Path,
    *,
    max_duration_hours: float = 7.0,
) -> None:
    asyncio.run(
        _backfill_profile_features_async(
            input_path,
            max_duration_hours=max_duration_hours,
        )
    )


def _write_cold_start_bootstrap_pair(
    output_dir: Path,
    *,
    pattern_profile: PatternProfile,
    profile_model: ProfileModel,
) -> bool:
    """Create an unpromoted bootstrap pair only when neither member exists."""
    pattern_path = output_dir / "pattern_profile.json"
    model_path = output_dir / "profile_model.json"
    if pattern_path.exists() != model_path.exists():
        raise RuntimeError(
            "Incomplete bootstrap pair: pattern_profile.json and "
            "profile_model.json must either both exist or both be absent."
        )
    if pattern_path.exists():
        return False

    pattern_tmp = pattern_path.with_suffix(".json.tmp")
    model_tmp = model_path.with_suffix(".json.tmp")
    try:
        pattern_profile.save_json(pattern_tmp)
        save_profile_model(profile_model, model_tmp)
        pattern_tmp.replace(pattern_path)
        model_tmp.replace(model_path)
    finally:
        pattern_tmp.unlink(missing_ok=True)
        model_tmp.unlink(missing_ok=True)
    return True


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("NEUTRAL Grid Scanner - Retrain Profile Models")
    logger.info("=" * 70)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Expected Excel file with bot performance data")
        sys.exit(1)

    logger.info(f"Input file: {input_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("-" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_pattern_path = output_dir / "pattern_profile.json"
    bootstrap_model_path = output_dir / "profile_model.json"
    current_path = output_dir / "current.json"
    if (
        not current_path.exists()
        and bootstrap_pattern_path.exists() != bootstrap_model_path.exists()
    ):
        logger.error(
            "Incomplete bootstrap pair in %s: pattern_profile.json and "
            "profile_model.json must either both exist or both be absent.",
            output_dir,
        )
        sys.exit(1)
    has_incumbent = current_path.exists() or (
        bootstrap_pattern_path.exists() and bootstrap_model_path.exists()
    )

    logger.info("Backfilling the 4 profile features onto the canonical training sheet...")
    try:
        backfill_profile_features(
            input_path,
            max_duration_hours=args.max_duration_hours,
        )
        logger.info("✓ Backfill complete")
    except Exception as e:
        logger.error(f"Failed to backfill profile features: {e}", exc_info=True)
        sys.exit(1)

    # Training parameters
    logger.info("Training parameters:")
    logger.info(f"  Max duration hours: {args.max_duration_hours}")
    logger.info(f"  Min profit factor: {args.min_profit_factor}")
    logger.info(f"  Min avg profit per grid: {args.min_avg_profit_per_grid}")
    logger.info(f"  Top quantile: {args.top_quantile}")
    logger.info(f"  Features: {len(DEFAULT_FEATURES)} features")
    logger.info("-" * 70)

    # Train pattern profile
    logger.info("Training pattern profile...")
    try:
        pattern_profile = build_profile_from_enhanced_xlsx(
            input_path,
            max_duration_hours=args.max_duration_hours,
            min_profit_factor=args.min_profit_factor,
            min_avg_profit_per_grid=args.min_avg_profit_per_grid,
            top_quantile=args.top_quantile,
            features=DEFAULT_FEATURES,
        )
        logger.info(f"✓ Pattern profile trained")
        logger.info(f"  Winners count: {pattern_profile.selection_summary.get('winners_count', 0)}")
        logger.info(f"  PnL threshold: {pattern_profile.selection_summary.get('pnl_threshold', 0):.2f}%")
        logger.info(f"  Features: {len(pattern_profile.features)}")

        logger.info("Candidate pattern profile retained in memory pending promotion")

    except Exception as e:
        logger.error(f"Failed to train pattern profile: {e}", exc_info=True)
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("Training profile model...")
    logger.info(f"  Shrinkage: {args.shrinkage}")
    try:
        profile_model = train_profile_model_from_enhanced_xlsx(
            input_path,
            max_duration_hours=args.max_duration_hours,
            min_profit_factor=args.min_profit_factor,
            min_avg_profit_per_grid=args.min_avg_profit_per_grid,
            top_quantile=args.top_quantile,
            features=DEFAULT_FEATURES,
            shrinkage=args.shrinkage,
        )
        logger.info(f"✓ Profile model trained")
        logger.info(f"  Features: {len(profile_model.features)}")
        logger.info(f"  Prior winner: {profile_model.prior_winner:.3f}")

        logger.info("Candidate profile model retained in memory pending promotion")

    except Exception as e:
        logger.error(f"Failed to train profile model: {e}", exc_info=True)
        sys.exit(1)

    logger.info("-" * 70)
    logger.info("Running governed walk-forward evaluation...")
    try:
        holdout_start_after_utc = resolve_latest_evaluated_holdout_end(output_dir)
        if holdout_start_after_utc is not None:
            logger.info(
                "OOF freshness boundary: test rows must begin strictly after %s",
                holdout_start_after_utc,
            )
        wf_result = walkforward_evaluate(
            input_path,
            n_folds=5,
            purge_hours=args.max_duration_hours,
            max_duration_hours=args.max_duration_hours,
            min_profit_factor=args.min_profit_factor,
            min_avg_profit_per_grid=args.min_avg_profit_per_grid,
            top_quantile=args.top_quantile,
            features=DEFAULT_FEATURES,
            shrinkage=args.shrinkage,
            holdout_start_after_utc=holdout_start_after_utc,
        )
        incumbent_wf_result = None
        incumbent_shrinkage = None
        if has_incumbent:
            incumbent_path = resolve_active_profile_model_path(output_dir)
            if not incumbent_path.exists():
                raise RuntimeError(
                    f"Incumbent resolver returned missing path: {incumbent_path}"
                )
            incumbent_model = load_profile_model(incumbent_path)
            summary = incumbent_model.selection_summary or {}
            recorded_shrinkage = summary.get("shrinkage")
            if args.incumbent_shrinkage is not None:
                incumbent_shrinkage = float(args.incumbent_shrinkage)
                if (
                    recorded_shrinkage is not None
                    and abs(float(recorded_shrinkage) - incumbent_shrinkage) >= 1e-12
                ):
                    raise RuntimeError(
                        "--incumbent-shrinkage conflicts with incumbent artifact "
                        f"provenance ({incumbent_shrinkage} != {recorded_shrinkage})."
                    )
            elif recorded_shrinkage is not None:
                incumbent_shrinkage = float(recorded_shrinkage)
            else:
                raise RuntimeError(
                    "Existing incumbent lacks shrinkage provenance. Re-run with "
                    "--incumbent-shrinkage set to the verified training value."
                )
            if abs(float(args.shrinkage) - incumbent_shrinkage) < 1e-12:
                incumbent_wf_result = wf_result
            else:
                incumbent_wf_result = walkforward_evaluate(
                    input_path,
                    n_folds=5,
                    purge_hours=args.max_duration_hours,
                    max_duration_hours=args.max_duration_hours,
                    min_profit_factor=args.min_profit_factor,
                    min_avg_profit_per_grid=args.min_avg_profit_per_grid,
                    top_quantile=args.top_quantile,
                    features=DEFAULT_FEATURES,
                    shrinkage=incumbent_shrinkage,
                    holdout_start_after_utc=holdout_start_after_utc,
                )
        promotion_decision = promote_profile_version(
            profile_model,
            requested_features=list(DEFAULT_FEATURES),
            wf_result=wf_result,
            incumbent_wf_result=incumbent_wf_result,
            candidate_pattern_profile=pattern_profile,
            profile_dir=output_dir,
            trial_hyperparameters={
                "n_folds": 5,
                "purge_hours": args.max_duration_hours,
                "max_duration_hours": args.max_duration_hours,
                "min_profit_factor": args.min_profit_factor,
                "min_avg_profit_per_grid": args.min_avg_profit_per_grid,
                "top_quantile": args.top_quantile,
                "shrinkage": args.shrinkage,
                "incumbent_shrinkage": incumbent_shrinkage,
                "holdout_start_after_utc": holdout_start_after_utc,
            },
        )
        logger.info(
            "Walk-forward result: finite_folds=%d/%d, mean_auc=%s, "
            "mean_pass_rate=%.3f, feature_coverage=%.3f",
            sum(1 for value in wf_result.fold_auc if pd.notna(value)),
            len(wf_result.fold_auc),
            (
                f"{wf_result.mean_auc:.3f}"
                if pd.notna(wf_result.mean_auc)
                else "NaN"
            ),
            wf_result.mean_pass_rate,
            wf_result.feature_coverage,
        )
        if promotion_decision.promoted:
            logger.info(
                "Profile model promoted: %s",
                promotion_decision.artifact_filename,
            )
        else:
            bootstrap_created = False
            if not has_incumbent:
                bootstrap_created = _write_cold_start_bootstrap_pair(
                    output_dir,
                    pattern_profile=pattern_profile,
                    profile_model=profile_model,
                )
            logger.warning(
                "Profile bundle not promoted; %s: %s",
                (
                    "cold-start bootstrap pair created"
                    if bootstrap_created
                    else "existing incumbent retained unchanged"
                ),
                promotion_decision.reason,
            )
    except Exception as e:
        logger.error(
            "Governed profile-model evaluation failed: %s",
            e,
            exc_info=True,
        )
        sys.exit(1)

    # Generate profile gate (optional)
    gate_written = False
    if not args.skip_gate:
        legacy_gate_inputs = {"adx_1h", "adx_15m", "adx_5m", "rsi_15m"}
        if not legacy_gate_inputs.issubset(set(pattern_profile.features)):
            logger.warning(
                "Skipping profile_gate generation: current PatternProfile contract "
                "does not carry the legacy ADX/RSI gate inputs (%s).",
                sorted(legacy_gate_inputs),
            )
        else:
            logger.info("-" * 70)
            logger.info("Generating profile gate...")
            logger.info(f"  Widening factor: {args.gate_widen}")
            try:
                gate = ProfileGate.from_pattern_profile(pattern_profile, widen=args.gate_widen)
                logger.info(f"✓ Profile gate generated")
                logger.info(f"  ADX 1h max: {gate.adx_1h_max:.2f}")
                logger.info(f"  ADX 15m max: {gate.adx_15m_max:.2f}")
                logger.info(f"  ADX 5m max: {gate.adx_5m_max:.2f}")
                logger.info(f"  RSI 15m range: [{gate.rsi_15m_low:.2f}, {gate.rsi_15m_high:.2f}]")

                # Save profile gate
                gate_path = output_dir / "profile_gate.json"
                gate_obj = {
                    "adx_1h_max": float(gate.adx_1h_max),
                    "adx_15m_max": float(gate.adx_15m_max),
                    "adx_5m_max": float(gate.adx_5m_max),
                    "rsi_15m_low": float(gate.rsi_15m_low),
                    "rsi_15m_high": float(gate.rsi_15m_high),
                }
                gate_path.write_text(
                    json.dumps(gate_obj, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                logger.info(f"✓ Saved to: {gate_path}")
                gate_written = True

            except Exception as e:
                logger.error(f"Failed to generate profile gate: {e}", exc_info=True)
                logger.warning("Continuing without profile gate")

    # Summary
    logger.info("=" * 70)
    logger.info("Training and governed evaluation complete")
    logger.info("=" * 70)
    logger.info("Governance artifacts:")
    if promotion_decision.evaluation_filename is not None:
        logger.info(
            "  ✓ %s",
            output_dir / "evaluations" / promotion_decision.evaluation_filename,
        )
    if promotion_decision.promoted:
        logger.info("  ✓ %s", output_dir / str(promotion_decision.artifact_filename))
        logger.info("  ✓ %s", output_dir / "current.json")
    else:
        logger.info("  retained bootstrap pattern: %s", bootstrap_pattern_path)
        logger.info("  retained bootstrap model: %s", bootstrap_model_path)
    if gate_written:
        logger.info(f"  ✓ {output_dir / 'profile_gate.json'}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review the evaluation artifact and promotion decision")
    if promotion_decision.promoted:
        logger.info("  2. Run pipeline preflight against the promoted bundle")
    else:
        logger.info("  2. Keep collecting finalized fast-outcome evidence")
        logger.info("  3. Do not deploy the rejected challenger")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
