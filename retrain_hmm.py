#!/usr/bin/env python3
"""
Retrain HMM Regime Detection Model.

This script retrains the HMM model using fresh market data from top symbols.
The HMM is used to detect range vs trending regimes for grid bot deployment.

Usage:
    python retrain_hmm.py [--symbols N] [--bars N] [--force-refresh]

Examples:
    # Default: top 50 symbols, 1000 bars each
    python retrain_hmm.py

    # More symbols and longer history
    python retrain_hmm.py --symbols 100 --bars 2000

    # Force refresh cached data
    python retrain_hmm.py --force-refresh
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import cast
from datetime import datetime, timezone
from pathlib import Path

# Ensure local `src/` package is used when running this script directly.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.models.hmm.train import train_from_market_data
from neutralgrid.scanner.scan import get_top_usdtm_symbols_by_quote_volume
from neutralgrid.models.hmm.canonical_retrain import (
    build_frozen_probe_universe,
    ensure_canonical_store_coverage,
    freeze_run_boundary,
    screen_probe_universe,
    freeze_training_slice,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_hmm")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain HMM Regime Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        type=int,
        default=50,
        help="Number of top symbols by quote volume to train on (default: 50)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=1000,
        help="Number of 15m bars per symbol (default: 1000; canonical mode bypasses this)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of HMM states (default: from config.HMM_N_COMPONENTS)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=300,
        help="Maximum EM iterations (default: 300)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh cached market data",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for market data (default: from config)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Directory to save trained model (default: from config)",
    )
    parser.add_argument(
        "--run-cpcv",
        action="store_true",
        help="Run CPCV evaluation after training (combinatorial purged cross-validation)",
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Run CPCV threshold sweep after CPCV evaluation (requires --run-cpcv)",
    )
    parser.add_argument(
        "--utility-sweep",
        action="store_true",
        help="Run CPCV utility-based threshold sweep (selects threshold by max OOS utility)",
    )
    parser.add_argument(
        "--stability-check",
        action="store_true",
        help="Run walk-forward stability check on regime model",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help=(
            "Save the trained HMM artifact but do not update artifact_manifest.json. "
            "Use for validation sweeps before promotion gates pass."
        ),
    )
    parser.add_argument(
        "--canonical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use canonical retrain pipeline (Vision store, screening, frozen slice). "
             "Pass --no-canonical for legacy API-based flow. Default: True.",
    )
    return parser.parse_args()


async def get_top_symbols(client: BinanceClient, top_n: int = 50) -> list[str]:
    """
    Get top N symbols by 24h quote volume.

    Args:
        client: Binance API client
        top_n: Number of symbols to return

    Returns:
        List of symbol names sorted by volume
    """
    try:
        # Use the existing function from scanner module
        top_symbols = await get_top_usdtm_symbols_by_quote_volume(client, top_n=top_n)
        return [symbol for symbol, _ in top_symbols]

    except Exception as e:
        logger.error(f"Failed to get top symbols: {e}")
        raise


async def main():
    """Main training execution."""
    args = parse_args()

    from neutralgrid.core.config import get_config
    _cfg = get_config()
    # Resolve --n-components from config if not provided on CLI
    if args.n_components is None:
        args.n_components = int(_cfg.hmm.n_components)

    logger.info("=" * 80)
    logger.info("HMM REGIME DETECTION MODEL - RETRAINING")
    logger.info("=" * 80)
    logger.info(f"  Symbols: top {args.symbols} by volume")
    logger.info(f"  Bars per symbol: {args.bars} (15m timeframe)")
    logger.info(f"  HMM components: {args.n_components}")
    logger.info(f"  Max iterations: {args.n_iter}")
    logger.info(f"  Force refresh: {args.force_refresh}")
    logger.info(f"  Canonical mode: {args.canonical}")
    logger.info("=" * 80)

    # Initialize client
    logger.info("")
    logger.info("Connecting to Binance...")
    client = BinanceClient()

    try:
        status = await client.check_connection()
        if not status.get("connected"):
            logger.error("Failed to connect to Binance API")
            sys.exit(1)
        logger.info(f"  Connected (authenticated: {status.get('authenticated', False)})")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        sys.exit(1)

    override_datasets = None
    frozen_slice = None
    canonical_probe_count = None

    if args.canonical:
        # ── Canonical retrain pipeline ──────────────────────────────────
        logger.info("")
        logger.info("Running canonical retrain pipeline (Vision store)...")

        try:
            # Step 1: build probe universe
            probe = await build_frozen_probe_universe(
                client, target_accepted=args.symbols,
            )

            # Step 2: ensure store coverage
            store_roots = await ensure_canonical_store_coverage(
                probe.ranked_symbols,
            )
            # ERR-084: recorded for the strict canonical promotion gate
            canonical_probe_count = len(probe.ranked_symbols)

            # Step 3: freeze run boundary
            boundary = freeze_run_boundary(
                store_roots, probe.selection_time_utc,
            )

            # Step 4: screen to exactly target_accepted symbols
            screening = screen_probe_universe(probe, store_roots, boundary)

            # Step 5: freeze training slice
            frozen_output_dir = (
                _cfg.resolve_path(_cfg.artifacts.artifacts_dir)
                / "hmm"
                / "training_sets"
                / f"canonical_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            frozen_slice = freeze_training_slice(
                screening, store_roots, boundary, frozen_output_dir,
            )

            # Use canonical outputs downstream
            symbols = screening.accepted
            override_datasets = frozen_slice.datasets

        except Exception as e:
            logger.error(f"Canonical pipeline failed: {e}", exc_info=True)
            await client.close()
            sys.exit(1)
    else:
        # ── Legacy API-based flow ───────────────────────────────────────
        logger.info("")
        logger.info(f"Getting top {args.symbols} symbols by volume...")
        try:
            symbols = await get_top_symbols(client, top_n=args.symbols)
            logger.info(f"  Found {len(symbols)} symbols")
            logger.info(f"  Top 10: {', '.join(symbols[:10])}")
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            await client.close()
            sys.exit(1)

    # Train model
    logger.info("")
    logger.info("Starting HMM training...")
    logger.info("-" * 40)

    try:
        from neutralgrid.models.hmm.retrain_orchestration import (
            disable_temperature_scaling_artifact,
            evaluate_and_maybe_promote,
            load_hmm_metadata,
            log_hmm_training_trial,
        )

        if args.artifact_dir:
            artifact_dir = _cfg.resolve_path(args.artifact_dir)
            if artifact_dir.name.lower() == "global":
                raise ValueError(
                    "Legacy HMM global artifact dir is not supported. "
                    "Use a versioned rolling_180d_YYYYMMDD_HHMMSS directory."
                )
            if not artifact_dir.name.startswith("rolling_180d_"):
                raise ValueError(
                    f"Unsupported HMM artifact dir: {artifact_dir}. "
                    "Expected name: rolling_180d_YYYYMMDD_HHMMSS"
                )
            version = artifact_dir.name
        else:
            version = f"rolling_180d_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            artifact_dir = _cfg.resolve_path(_cfg.artifacts.artifacts_dir) / "hmm" / version

        cache_dir = Path(args.cache_dir) if args.cache_dir and not args.force_refresh else (
            Path(_cfg.artifacts.cache_dir) if not args.force_refresh else None
        )

        artifact_path, training_datasets = await train_from_market_data(
            client=client,
            symbols=symbols,
            artifact_dir=artifact_dir,
            timeframe="15m",
            limit=args.bars,
            n_components=args.n_components,
            n_iter=args.n_iter,
            cache_dir=cache_dir,
            version=version,
            override_datasets=override_datasets,
            canonical_mode=args.canonical,
            canonical_metadata=frozen_slice.metadata if frozen_slice is not None else None,
        )

        metadata = load_hmm_metadata(artifact_path)
        actual_window_days = None
        try:
            start_raw = metadata.get("training_start_utc")
            end_raw = metadata.get("training_end_utc")
            if start_raw and end_raw:
                start_dt = datetime.fromisoformat(str(start_raw))
                end_dt = datetime.fromisoformat(str(end_raw))
                actual_window_days = max(
                    0.0,
                    (end_dt - start_dt).total_seconds() / 86400.0,
                )
        except Exception as window_err:
            logger.warning("Could not determine actual HMM training window: %s", window_err)

        disable_temperature_scaling_artifact(artifact_path)
        if args.no_promote:
            eval_metrics = metadata.get("eval_metrics") or {}
            promoted = False
            promotion_reasons = ["no_promote_cli_flag"]
            logger.info(
                "Model saved at %s but promotion skipped by --no-promote.",
                artifact_path,
            )
        else:
            eval_metrics, promoted, promotion_reasons = evaluate_and_maybe_promote(
                version=version,
                artifact_path=artifact_path,
                requested_symbols=len(symbols),
                fetched_datasets=training_datasets,
                window_days=180,
                actual_window_days=actual_window_days,
                # ERR-084: wire the strict canonical gate (accepted==50 AND
                # trained==50) on the canonical path; previously these were
                # never passed, so only the legacy gates ran.
                canonical_mode=args.canonical,
                accepted_count=len(symbols) if args.canonical else None,
                trained_count=len(training_datasets) if args.canonical else None,
                probe_count=canonical_probe_count,
            )
        if promoted:
            logger.info(
                "Quality check passed (mean_pass_rate=%s), promoting model.",
                f"{float(eval_metrics.get('mean_pass_rate', 0.0)):.2%}",
            )
        else:
            logger.warning(
                "Model saved at %s but NOT promoted as active: %s",
                artifact_path,
                ", ".join(promotion_reasons) if promotion_reasons else "unknown policy block",
            )

        # Use canonical artifact path for downstream report artifacts.
        artifact_dir = artifact_path

        log_hmm_training_trial(
            artifact_path=artifact_path,
            version=version,
            requested_symbols=len(symbols),
            fetched_datasets=training_datasets,
            bars_per_symbol=args.bars,
            n_components=int(args.n_components),
            n_iter=int(args.n_iter),
            window_days=180,
            promoted=promoted,
            promotion_reasons=promotion_reasons,
        )

        # Feature multicollinearity diagnostics (AFML Ch 8)
        try:
            import pandas as pd
            from neutralgrid.training.feature_diagnostics import compute_feature_diagnostics
            from neutralgrid.models.hmm.schema import FEATURE_NAMES
            combined = pd.concat(list(training_datasets.values()), ignore_index=True)
            feat_cols = [f for f in FEATURE_NAMES if f in combined.columns]
            if len(feat_cols) >= 2:
                compute_feature_diagnostics(cast(pd.DataFrame, combined[feat_cols]))
        except Exception as _diag_e:
            logger.warning(f"Feature diagnostics skipped: {_diag_e}")

        logger.info("")
        logger.info(
            "Temperature scaling disabled for self-labeled HMM calibration; identity scaler saved."
        )
        logger.info("")
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Artifact saved to: {artifact_path}")

        # Load and display metadata
        import json
        metadata_path = artifact_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info("")
            logger.info("Model Summary:")
            logger.info(f"  Version: {metadata.get('artifact_version')}")
            logger.info(f"  Sequences: {metadata.get('num_sequences')}")
            logger.info(f"  Total samples: {metadata.get('total_samples')}")

            eval_metrics = metadata.get("eval_metrics", {})
            if eval_metrics:
                logger.info("")
                logger.info("Walk-Forward Evaluation:")
                logger.info(f"  Splits: {eval_metrics.get('num_splits')}")
                logger.info(f"  Mean pass rate: {eval_metrics.get('mean_pass_rate', 0):.2%}")
                logger.info(f"  Mean range prob: {eval_metrics.get('mean_range_prob', 0):.4f}")
                logger.info(f"  Mean trend prob: {eval_metrics.get('mean_trend_prob', 0):.4f}")

            state_mapping = metadata.get("state_mapping", {})
            if state_mapping:
                logger.info("")
                logger.info("State Mapping:")
                logger.info(f"  Range state: {state_mapping.get('range')}")
                logger.info(f"  Trend state: {state_mapping.get('trend')}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        await client.close()
        sys.exit(1)

    # Optional CPCV evaluation
    if args.run_cpcv:
        logger.info("")
        logger.info("=" * 80)
        logger.info("CPCV EVALUATION (Combinatorial Purged Cross-Validation)")
        logger.info("=" * 80)
        try:
            from neutralgrid.backtest.evaluate import run_cpcv_hmm_evaluation

            cpcv_results = run_cpcv_hmm_evaluation(
                datasets=training_datasets,
                n_components=args.n_components,
                n_iter=args.n_iter,
            )

            num_paths = cpcv_results.get("num_paths", 0)
            mean_pass = cpcv_results.get("mean_pass_rate", 0.0)
            mean_pass_dom = cpcv_results.get("mean_pass_rate_dominance", 0.0)
            logger.info(f"  CPCV paths: {num_paths}")
            logger.info(f"  Mean pass rate (hard): {mean_pass:.2%}")
            logger.info(f"  Mean pass rate (dominance): {mean_pass_dom:.2%}")

            dist_stats = cpcv_results.get("distribution_stats", {})
            n_total_bars = dist_stats.get("n_total_bars", 0)
            logger.info(f"  Total per-bar evaluations: {n_total_bars}")

            if "range_dominance" in dist_stats:
                dom_stats = dist_stats["range_dominance"]
                logger.info(f"  range_dominance — mean={dom_stats.get('mean', 0):.4f} "
                            f"std={dom_stats.get('std', 0):.4f} "
                            f"p50={dom_stats.get('p50', 0):.4f} "
                            f"p80={dom_stats.get('p80', 0):.4f}")

            deflated = cpcv_results.get("deflated_sharpe")
            if deflated is not None:
                if isinstance(deflated, dict):
                    logger.info(f"  Deflated Sharpe: {deflated.get('deflated_sharpe', 'N/A')}, "
                                f"raw={deflated.get('raw_sharpe', 'N/A')}, "
                                f"haircut={deflated.get('haircut_pct', 'N/A')}%")
                else:
                    logger.info(f"  Deflated Sharpe: {deflated:.4f}")

            pbo = cpcv_results.get("pbo")
            if pbo is not None:
                if isinstance(pbo, dict):
                    logger.info(f"  PBO: {pbo.get('pbo', 'N/A')}, "
                                f"n_overfit={pbo.get('n_overfit', 'N/A')}/{pbo.get('n_paths', 'N/A')}")
                else:
                    logger.info(f"  PBO probability: {pbo:.2%}")

            # Save CPCV results alongside artifact
            import json
            cpcv_path = artifact_dir / "cpcv_results.json"
            cpcv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cpcv_path, "w", encoding="utf-8") as f:
                json.dump(cpcv_results, f, indent=2, default=str)
            logger.info(f"  Results saved to: {cpcv_path}")

        except Exception as e:
            logger.warning(f"CPCV evaluation failed (non-fatal): {e}", exc_info=True)

        # Optional threshold sweep (requires --sweep-thresholds or always runs with --run-cpcv)
        if args.sweep_thresholds:
            logger.info("")
            logger.info("-" * 40)
            logger.info("CPCV THRESHOLD SWEEP")
            logger.info("-" * 40)
            try:
                from neutralgrid.backtest.evaluate import run_cpcv_threshold_sweep

                sweep_target = _cfg.cpcv.sweep_target_pass_rate
                sweep_results = run_cpcv_threshold_sweep(
                    datasets=training_datasets,
                    n_components=args.n_components,
                    n_iter=args.n_iter,
                    target_pass_rate_low=sweep_target[0],
                    target_pass_rate_high=sweep_target[1],
                )

                rec = sweep_results.get("recommended", {})

                # Log dominance recommendation
                dom_rec = rec.get("dominance")
                if dom_rec:
                    logger.info(f"  [dominance] recommended range_dominance_min={dom_rec['range_dominance_min']:.2f} "
                                f"→ pass_rate={dom_rec['pass_rate']:.2%}")

                # Log hard recommendation
                hard_rec = rec.get("hard")
                if hard_rec:
                    logger.info(f"  [hard] recommended range_prob_min={hard_rec['range_prob_min']:.2f} "
                                f"trend_prob_max={hard_rec['trend_prob_max']:.2f} "
                                f"→ pass_rate={hard_rec['pass_rate']:.2%}")

                # Log percentile-anchored
                pct_rec = rec.get("percentile_anchored", {})
                for label, info in pct_rec.items():
                    logger.info(f"  [percentile {label}] dominance_min={info['range_dominance_min']:.4f} "
                                f"→ pass_rate={info['pass_rate']:.2%} ({info['description']})")

                # Log distribution summary
                dist = sweep_results.get("distribution_stats", {})
                n_bars = dist.get("n_total_bars", 0)
                logger.info(f"  Sweep based on {n_bars} per-bar test posteriors")

                # Save sweep results
                import json
                sweep_path = artifact_dir / "cpcv_sweep_results.json"
                sweep_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sweep_path, "w", encoding="utf-8") as f:
                    json.dump(sweep_results, f, indent=2, default=str)
                logger.info(f"  Sweep results saved to: {sweep_path}")

            except Exception as e:
                logger.warning(f"Threshold sweep failed (non-fatal): {e}", exc_info=True)

        # Optional CPCV utility-based threshold sweep
        if args.utility_sweep:
            logger.info("")
            logger.info("-" * 40)
            logger.info("CPCV UTILITY SWEEP (max OOS grid-survival utility)")
            logger.info("-" * 40)
            try:
                from neutralgrid.backtest.evaluate import run_cpcv_utility_sweep

                utility_results = run_cpcv_utility_sweep(
                    datasets=training_datasets,
                    n_components=args.n_components,
                    n_iter=args.n_iter,
                )

                if "error" not in utility_results:
                    sel_t = utility_results["selected_threshold"]
                    sel_u = utility_results["selected_utility"]
                    sel_pr = utility_results["selected_pass_rate"]
                    n_paths = utility_results["n_paths"]
                    logger.info(f"  Selected threshold: {sel_t:.2f}")
                    logger.info(f"  Utility: {sel_u:.6f}")
                    logger.info(f"  Pass rate: {sel_pr:.2%}")
                    logger.info(f"  CPCV paths: {n_paths}")

                    rank_pers = utility_results.get("rank_persistence", {})
                    corr = rank_pers.get("mean_spearman_corr")
                    if corr is not None:
                        logger.info(f"  Rank persistence (Spearman): {corr:.4f}")

                    # Per-threshold detail
                    for entry in utility_results.get("per_threshold", []):
                        t = entry["threshold"]
                        mu = entry["mean_utility"]
                        pr = entry["mean_pass_rate"]
                        mu_str = f"{mu:.6f}" if mu is not None else "N/A"
                        marker = " <-- SELECTED" if t == sel_t else ""
                        logger.info(f"    t={t:.2f}  U={mu_str}  pass_rate={pr:.2%}{marker}")

                    # Save artifact: cpcv_selected_threshold.json
                    import json
                    artifact_output = utility_results["artifact"]
                    threshold_path = artifact_dir / "cpcv_selected_threshold.json"
                    threshold_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(threshold_path, "w", encoding="utf-8") as f:
                        json.dump(artifact_output, f, indent=2, default=str)
                    logger.info(f"  Threshold artifact saved to: {threshold_path}")

                    # Save full utility sweep results
                    utility_path = artifact_dir / "cpcv_utility_sweep.json"
                    utility_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(utility_path, "w", encoding="utf-8") as f:
                        json.dump(utility_results, f, indent=2, default=str)
                    logger.info(f"  Full results saved to: {utility_path}")
                else:
                    logger.warning(f"  Utility sweep error: {utility_results['error']}")

            except Exception as e:
                logger.warning(f"Utility sweep failed (non-fatal): {e}", exc_info=True)

        # Proxy outcome report: forward returns for pass vs fail bars
        logger.info("")
        logger.info("-" * 40)
        logger.info("PROXY OUTCOME REPORT (pass vs fail forward returns)")
        logger.info("-" * 40)
        try:
            from neutralgrid.backtest.evaluate import run_cpcv_proxy_outcome_report

            proxy_report = run_cpcv_proxy_outcome_report(
                datasets=training_datasets,
                n_components=args.n_components,
                n_iter=args.n_iter,
            )

            if "error" not in proxy_report:
                counts = proxy_report.get("counts", {})
                logger.info(f"  Gate: dominance≥{proxy_report['gate_config']['range_dominance_min']} "
                            f"+ trend_prob≤{proxy_report['gate_config']['trend_prob_max_veto']}")
                logger.info(f"  Pass: {counts.get('pass', 0):,}  Fail: {counts.get('fail', 0):,}  "
                            f"Rate: {counts.get('pass_rate', 0):.2%}")

                for _, h_data in proxy_report.get("forward_returns", {}).items():
                    h = h_data["horizon_bars"]
                    p = h_data.get("pass", {})
                    f = h_data.get("fail", {})
                    logger.info(f"  ── {h}-bar forward |log-return| ──")
                    if p.get("count", 0) > 0:
                        logger.info(f"    PASS  mean={p['mean']:.6f}  median={p['median']:.6f}  "
                                    f"std={p['std']:.6f}  n={p['count']:,}")
                    if f.get("count", 0) > 0:
                        logger.info(f"    FAIL  mean={f['mean']:.6f}  median={f['median']:.6f}  "
                                    f"std={f['std']:.6f}  n={f['count']:,}")
                    sep = h_data.get("separation_ratio")
                    if sep is not None:
                        logger.info(f"    separation_ratio={sep:.4f}  "
                                    f"(>1 = gate correctly identifies range)")
                    t_stat = h_data.get("welch_t_stat")
                    p_val = h_data.get("welch_p_value")
                    if t_stat is not None:
                        p_val_str = f"{p_val:.2e}" if p_val is not None else "N/A"
                        logger.info(f"    Welch t={t_stat:.4f}  p={p_val_str}")

                dom_st = proxy_report.get("dominance_stats", {})
                logger.info(f"  dominance — pass_mean={dom_st.get('pass_mean', 'N/A')}  "
                            f"fail_mean={dom_st.get('fail_mean', 'N/A')}")

                # Save proxy outcome report
                import json
                proxy_path = artifact_dir / "cpcv_proxy_outcomes.json"
                proxy_path.parent.mkdir(parents=True, exist_ok=True)
                with open(proxy_path, "w", encoding="utf-8") as f:
                    json.dump(proxy_report, f, indent=2, default=str)
                logger.info(f"  Results saved to: {proxy_path}")
            else:
                logger.warning(f"  Proxy report error: {proxy_report['error']}")

        except Exception as e:
            logger.warning(f"Proxy outcome report failed (non-fatal): {e}", exc_info=True)

    # Walk-forward stability check (runs with --stability-check or --run-cpcv)
    run_stability = args.stability_check or args.run_cpcv
    if run_stability:
        logger.info("")
        logger.info("=" * 80)
        logger.info("WALK-FORWARD STABILITY CHECK")
        logger.info("=" * 80)
        try:
            from neutralgrid.backtest.evaluate import compute_wf_stability_report

            stability_report = compute_wf_stability_report(
                datasets=training_datasets,
                n_components=args.n_components,
                n_iter=args.n_iter,
            )

            stability = stability_report.get("stability", "UNKNOWN")
            logger.info(f"  Stability: {stability}")

            cross = stability_report.get("cross_fold_metrics", {})
            if cross:
                logger.info(f"  Transition matrix max delta: {cross.get('transition_matrix_max_delta', 'N/A')}")
                logger.info(f"  Dominance mean drift: {cross.get('dominance_mean_drift', 'N/A')}")
                logger.info(f"  Duration CV: {cross.get('duration_cv', 'N/A')}")

            # Per-fold summary
            for fold in stability_report.get("folds", []):
                if "error" in fold:
                    logger.info(f"  Fold {fold['fold']}: ERROR - {fold['error']}")
                else:
                    dom = fold.get("dominance", {})
                    logger.info(f"  Fold {fold['fold']} (train={fold['train_frac']:.0%}): "
                                f"dom_mean={dom.get('mean', 'N/A')} "
                                f"range_dur={fold.get('mean_range_duration', 'N/A')}")

            # Save stability report
            import json
            stability_path = artifact_dir / "wf_stability_report.json"
            stability_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stability_path, "w", encoding="utf-8") as f:
                json.dump(stability_report, f, indent=2, default=str)
            logger.info(f"  Report saved to: {stability_path}")

        except Exception as e:
            logger.warning(f"Stability check failed (non-fatal): {e}", exc_info=True)

    # Cleanup
    await client.close()

    logger.info("")
    logger.info("=" * 80)
    logger.info("Done. Run 'python run_full_pipeline.py' to test the new model.")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
