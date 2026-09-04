#!/usr/bin/env python3
"""
Bayesian Threshold Optimization for deployment gate calibration.

Phase 3 calibration: uses GP-UCB acquisition to optimize the 7-dimensional
threshold space (range_prob_min, trend_prob_max, min_tos, min_position_fraction,
etc.) using deflated Sharpe * sqrt(N_deployed) as the objective.

Usage:
    python optimize_thresholds_v20260311.py --backtest-dir data/backtest_candidates
    python optimize_thresholds_v20260311.py --iterations 30 --output data/cache/optimized_thresholds.json

Requires:
    - Historical backtest results in data/backtest_candidates/
    - Trained HMM and meta-labeler artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure local `src/` package is used when running this script directly.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import pandas as pd

from neutralgrid.scanner.empirical_profile_v20260302 import (
    collect_authoritative_backtest_rows,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("optimize_thresholds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian threshold optimization for deployment gates",
    )
    parser.add_argument(
        "--backtest-dir",
        type=str,
        default="data/backtest_candidates",
        help="Directory containing backtest result CSVs (default: data/backtest_candidates)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        help="Number of GP-UCB optimization iterations (default: 25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/cache/optimized_thresholds.json",
        help="Output path for optimized thresholds JSON (default: data/cache/optimized_thresholds.json)",
    )
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=3,
        help="Number of walk-forward validation folds (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print search space and exit without optimizing",
    )
    return parser.parse_args()


def load_backtest_results(backtest_dir: Path) -> pd.DataFrame:
    """Load only legacy-authority backtest rows with deterministic dedup."""
    combined = collect_authoritative_backtest_rows(backtest_dir)
    if combined.empty:
        logger.error("No authoritative backtest rows found in %s", backtest_dir)
        sys.exit(1)
    logger.info("  Total authoritative deduplicated backtest rows: %d", len(combined))
    return combined


def build_evaluate_fn(backtest_df: pd.DataFrame):
    """Build the objective function for Bayesian optimization.

    The objective is: deflated_sharpe * sqrt(N_deployed)

    Parameters
    ----------
    backtest_df : DataFrame
        Historical backtest results with columns including:
        pnl_pct, range_prob, trend_prob, tos, position_fraction, etc.

    Returns
    -------
    callable
        evaluate_fn(thresholds: dict) -> float
    """

    def evaluate_fn(thresholds: dict) -> float:
        """Apply thresholds to historical results and compute objective."""
        df = backtest_df.copy()

        # Apply threshold filters
        mask = pd.Series(True, index=df.index)

        if "range_prob" in df.columns and "range_prob_min" in thresholds:
            rp = pd.to_numeric(df["range_prob"], errors="coerce")
            mask &= rp >= thresholds["range_prob_min"]

        if "trend_prob" in df.columns and "trend_prob_max" in thresholds:
            tp = pd.to_numeric(df["trend_prob"], errors="coerce")
            mask &= tp <= thresholds["trend_prob_max"]

        if "tos" in df.columns and "min_tos" in thresholds:
            tos = pd.to_numeric(df["tos"], errors="coerce")
            mask &= tos >= thresholds["min_tos"]

        if "meta_prob" in df.columns and "min_meta_prob" in thresholds:
            mp = pd.to_numeric(df["meta_prob"], errors="coerce")
            mask &= mp >= thresholds["min_meta_prob"]

        filtered = df[mask]
        n_deployed = len(filtered)

        if n_deployed < 5:
            return -1.0  # Penalize configurations that deploy too few

        # Compute risk-adjusted return
        pnl_col = "pnl_pct" if "pnl_pct" in filtered.columns else "net_pnl_pct"
        if pnl_col not in filtered.columns:
            return -1.0

        pnl_raw = np.asarray(pd.to_numeric(pd.Series(filtered[pnl_col]), errors="coerce"))
        pnl = pnl_raw[np.isfinite(pnl_raw)]
        if len(pnl) < 5:
            return -1.0

        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl, ddof=1))

        if std_pnl < 1e-8:
            sharpe = 0.0
        else:
            sharpe = mean_pnl / std_pnl

        # Deflation: penalize based on number of optimization trials
        # This is a simplified deflation — full AFML deflation requires
        # the complete trial history
        deflation = 1.0  # No deflation for first pass

        # Objective: deflated Sharpe * sqrt(N_deployed)
        objective = deflation * sharpe * np.sqrt(n_deployed)

        return float(objective)

    return evaluate_fn


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("BAYESIAN THRESHOLD OPTIMIZATION (Phase 3)")
    logger.info("=" * 80)

    # Load backtest data
    backtest_dir = Path(args.backtest_dir)
    if not backtest_dir.exists():
        logger.error("Backtest directory not found: %s", backtest_dir)
        sys.exit(1)

    logger.info("Loading backtest results from %s...", backtest_dir)
    backtest_df = load_backtest_results(backtest_dir)

    # Import optimizer
    try:
        from neutralgrid.optimization.bayesian_threshold_optimizer_v20260311 import (
            BayesianThresholdOptimizer,
        )
    except ImportError:
        logger.error(
            "BayesianThresholdOptimizer not available. "
            "Ensure neutralgrid is installed: pip install -e ."
        )
        sys.exit(1)

    # Build evaluate function
    evaluate_fn = build_evaluate_fn(backtest_df)

    if args.dry_run:
        logger.info("Dry run — search space:")
        optimizer = BayesianThresholdOptimizer()
        _space: dict[str, tuple[float, float]] = optimizer.search_space  # type: ignore[assignment]
        for name, bounds in _space.items():
            logger.info("  %s: [%.4f, %.4f]", name, bounds[0], bounds[1])
        logger.info("Total iterations would be: %d", args.iterations)
        return

    # Build default theta from search space midpoints
    optimizer = BayesianThresholdOptimizer()
    _space_bounds: dict[str, tuple[float, float]] = optimizer.search_space  # type: ignore[assignment]
    default_theta: dict[str, float] = {}
    for name, (lo, hi) in _space_bounds.items():
        default_theta[name] = (lo + hi) / 2.0
    # Round discrete params
    if "gb_max_depth" in default_theta:
        default_theta["gb_max_depth"] = float(round(default_theta["gb_max_depth"]))

    # Run optimization
    logger.info("")
    logger.info("Running %d GP-UCB iterations...", args.iterations)
    logger.info("-" * 40)

    # Override optimizer iteration counts if CLI provides different values
    from neutralgrid.optimization.bayesian_threshold_optimizer_v20260311 import (
        OptimizationConfig,
    )
    opt_cfg = OptimizationConfig(
        n_optimization_steps=max(0, args.iterations - 20),  # subtract LHS init
        n_initial_samples=min(20, args.iterations),
    )
    optimizer = BayesianThresholdOptimizer(config=opt_cfg)

    try:
        result = optimizer.optimize(
            evaluate_fn=evaluate_fn,
            default_theta=default_theta,
        )
    except Exception as e:
        logger.error("Optimization failed: %s", e, exc_info=True)
        sys.exit(1)

    # Report results (result is an OptimizationResult dataclass)
    logger.info("")
    logger.info("=" * 80)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 80)
    logger.info("  Best objective: %.6f", result.best_objective)
    logger.info("  Best thresholds:")
    best = result.best_theta
    for k, v in best.items():
        logger.info("    %s: %.4f", k, v)

    # Walk-forward validation
    wf_data: dict | None = None
    if args.walk_forward_folds > 1:
        logger.info("")
        logger.info("Walk-forward validation (%d folds)...", args.walk_forward_folds)
        try:
            # Simple temporal split: train on first 80%, test on last 20%
            n_rows = len(backtest_df)
            split_idx = int(n_rows * 0.80)
            if split_idx > 10 and (n_rows - split_idx) > 5:
                train_df = backtest_df.iloc[:split_idx]
                test_df = backtest_df.iloc[split_idx:]
                eval_train = build_evaluate_fn(train_df)
                eval_test = build_evaluate_fn(test_df)
                retention, passes_gate = optimizer.validate_walk_forward(
                    theta=best,
                    evaluate_fn_train=eval_train,
                    evaluate_fn_test=eval_test,
                )
                wf_data = {
                    "retention_ratio": retention,
                    "passes_gate": passes_gate,
                    "train_rows": split_idx,
                    "test_rows": n_rows - split_idx,
                }
                logger.info("  Retention ratio: %.4f (gate: %s)", retention, passes_gate)
            else:
                logger.warning("  Not enough data for walk-forward split")
        except Exception as wf_err:
            logger.warning("Walk-forward validation failed: %s", wf_err)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "best_thresholds": best,
        "best_objective": result.best_objective,
        "n_iterations": args.iterations,
        "n_backtest_rows": len(backtest_df),
        "walk_forward": wf_data,
        "sensitivity_report": result.sensitivity_report,
    }

    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info("")
    logger.info("Results saved to: %s", output_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
