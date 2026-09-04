"""
CLI for model retraining with rolling windows.

Supports time-decay / rolling retraining to handle non-stationary markets.
AFML principle: Markets drift; models must adapt.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import sys

import pandas as pd

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.models.hmm.train import train_from_market_data
from neutralgrid.models.hmm.retrain_orchestration import (
    disable_temperature_scaling_artifact,
    evaluate_and_maybe_promote,
    log_hmm_training_trial,
)
from neutralgrid.models.hmm.canonical_retrain import (
    build_frozen_probe_universe,
    ensure_canonical_store_coverage,
    freeze_run_boundary,
    screen_probe_universe,
    freeze_training_slice,
)
from neutralgrid.data.market_data import (
    fetch_training_dataset,
    save_training_dataset,
)
from neutralgrid.core.logging import (
    get_logger,
    set_run_id,
    set_artifact_version,
)
from neutralgrid.core.config import get_config


logger = get_logger(__name__)
_ROLLING_VERSION_RE = re.compile(r"^rolling_\d+d_\d{8}_\d{6}$")


def parse_time_window(window_str: str) -> int:
    """
    Parse time window string to days.

    Args:
        window_str: Time window string (e.g., "180d", "90d", "30d")

    Returns:
        Number of days

    Examples:
        >>> parse_time_window("180d")
        180
        >>> parse_time_window("90d")
        90
    """
    if window_str.endswith("d"):
        return int(window_str[:-1])
    elif window_str.endswith("w"):
        return int(window_str[:-1]) * 7
    elif window_str.endswith("m"):
        return int(window_str[:-1]) * 30
    else:
        raise ValueError(f"Invalid time window format: {window_str} (expected: 180d, 12w, 6m)")


async def retrain_with_rolling_window(
    symbols: list[str],
    window_days: int = 180,
    artifact_dir: Optional[Path] = None,
    save_dataset: bool = True,
    dataset_name: Optional[str] = None,
    n_components: Optional[int] = None,
    force_refresh: bool = False,
    _override_datasets: Optional[dict[str, "pd.DataFrame"]] = None,
    _research_only: bool = False,
):
    """
    Retrain HMM model with rolling time window.

    Args:
        symbols: List of symbols to train on
        window_days: Training window size in days (default: 180)
        artifact_dir: Directory to save trained model
        save_dataset: Whether to save training dataset to disk
        dataset_name: Name for saved dataset (default: timestamp)
        n_components: HMM components (default: from config)
        force_refresh: Force refresh data from API even if cached
        _override_datasets: Pre-loaded datasets from canonical pipeline.
            When provided, skips API fetch entirely.
        _research_only: If True, skip promotion (research / explicit-symbol run).

    Returns:
        Path to saved artifact
    """
    run_id = set_run_id()
    logger.info("Starting rolling retraining | run_id=%s, window=%dd", run_id, window_days)

    # Load config
    config = get_config()
    if n_components is None:
        n_components = config.hmm.n_components

    # Calculate time window
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=window_days)

    logger.info("Time window: %s to %s", start_time.date(), end_time.date())

    # Initialize client
    client = BinanceClient()

    # Calculate limit (bars needed for window)
    # 15m bars: 96 bars/day
    bars_needed = window_days * 96 + 100
    limit = min(bars_needed, 1500)  # Cap at Binance max (1500)

    if _override_datasets is not None:
        # Canonical pipeline already provided validated datasets
        datasets = _override_datasets
        logger.info(
            "Using %d pre-loaded canonical datasets (skipping API fetch)",
            len(datasets),
        )
    else:
        if bars_needed > 1500:
            effective_days = (1500 - 100) / 96
            logger.warning(
                "Requested %d-day window requires %d bars, "
                "but Binance API limit is 1500. Training data truncated to ~%.0f days. "
                "Consider using cached datasets or fetching data in chunks for longer windows.",
                window_days, bars_needed, effective_days,
            )

        logger.info("Fetching data for %d symbols (limit: %d bars)", len(symbols), limit)

        # Fetch training data
        datasets = await fetch_training_dataset(
            client=client,
            symbols=symbols,
            timeframe="15m",
            limit=limit,
            force_refresh=force_refresh,
            strict=True,
        )

        logger.info("Fetched %d symbols successfully", len(datasets))

    # Verify actual training span matches requested window
    actual_bars = min(len(df) for df in datasets.values()) if datasets else 0
    actual_days = actual_bars / 96.0
    if actual_days < window_days * 0.9:
        logger.warning(
            "Training data covers %.1f days, less than 90%% of requested %d-day window. "
            "Artifact will not be eligible for promotion.",
            actual_days, window_days,
        )

    # Save dataset if requested
    if save_dataset:
        if dataset_name is None:
            dataset_name = f"rolling_{window_days}d_{end_time.strftime('%Y%m%d')}"

        dataset_dir = config.resolve_path(config.artifacts.training_sets_dir) / dataset_name
        save_training_dataset(
            datasets=datasets,
            output_dir=dataset_dir,
            metadata={
                "description": f"Rolling window training dataset ({window_days} days)",
                "window_days": window_days,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "num_symbols": len(datasets),
                "run_id": run_id,
            }
        )
        logger.info("Dataset saved to: %s", dataset_dir)

    # Generate version (before artifact_dir so we can use it in the path)
    version = f"rolling_{window_days}d_{end_time.strftime('%Y%m%d_%H%M%S')}"
    set_artifact_version(version)

    # Set artifact directory (versioned by default, never legacy "global")
    if artifact_dir is None:
        artifact_dir = config.resolve_path(config.artifacts.artifacts_dir) / "hmm" / version
    else:
        artifact_dir = config.resolve_path(str(artifact_dir))
        if artifact_dir.name.lower() == "global":
            raise ValueError(
                "Legacy HMM global artifact dir is not supported. "
                "Use rolling_180d_YYYYMMDD_HHMMSS directories."
            )
        if _ROLLING_VERSION_RE.match(artifact_dir.name):
            # Caller provided an explicit versioned dir; keep metadata/dir aligned.
            version = artifact_dir.name
            set_artifact_version(version)
        else:
            # Treat custom output as parent directory and create a versioned child.
            artifact_dir = artifact_dir / version

    logger.info("Training HMM model | version=%s", version)

    # Train model (pass pre-fetched datasets to avoid redundant API fetch)
    artifact_path, _ = await train_from_market_data(
        client=client,
        symbols=symbols,
        artifact_dir=artifact_dir,
        timeframe="15m",
        limit=limit,
        n_components=n_components,
        version=version,
        override_datasets=datasets,
    )

    logger.info("Model trained and saved to: %s", artifact_path)

    disable_temperature_scaling_artifact(artifact_path)

    if _research_only:
        # Research-only: skip promotion, log trial as not promoted
        logger.info(
            "Research-only run (explicit --symbols) -- skipping promotion."
        )
        promoted = False
        promotion_reasons = ["research_only_explicit_symbols"]
        eval_metrics: dict[str, object] = {}
    else:
        eval_metrics, promoted, promotion_reasons = evaluate_and_maybe_promote(
            version=version,
            artifact_path=artifact_path,
            requested_symbols=len(symbols),
            fetched_datasets=datasets,
            window_days=window_days,
            actual_window_days=actual_days,
        )
        if promoted:
            logger.info(
                "Quality check passed (mean_pass_rate=%s), promoting model.",
                f"{float(str(eval_metrics.get('mean_pass_rate', 0.0))):.2%}",
            )
        else:
            logger.warning(
                "Model saved at %s but not promoted as active: %s",
                artifact_path,
                ", ".join(promotion_reasons) if promotion_reasons else "unknown policy block",
            )

    log_hmm_training_trial(
        artifact_path=artifact_path,
        version=version,
        requested_symbols=len(symbols),
        fetched_datasets=datasets,
        bars_per_symbol=limit,
        n_components=int(n_components),
        n_iter=int(config.hmm.n_iter),
        window_days=window_days,
        promoted=promoted,
        promotion_reasons=promotion_reasons,
    )

    await client.close()

    return artifact_path


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Retrain HMM model with rolling time window",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain with 180-day rolling window
  python -m neutralgrid.cli.retrain --window 180d

  # Retrain with 90-day window, top 50 symbols
  python -m neutralgrid.cli.retrain --window 90d --top-n 50

  # Retrain with custom output directory
  python -m neutralgrid.cli.retrain --window 180d --output artifacts/hmm/rolling_180d

  # Retrain with dataset saving
  python -m neutralgrid.cli.retrain --window 180d --save-dataset

  # Force refresh data from API
  python -m neutralgrid.cli.retrain --window 180d --force-refresh
        """
    )

    parser.add_argument(
        "--window",
        type=str,
        default="180d",
        help="Training window size (e.g., 180d, 90d, 12w, 6m). Default: 180d"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Specific symbols to train on (e.g., BTCUSDT ETHUSDT). If not specified, uses top-N by volume"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Number of top symbols by volume to use if --symbols not specified. Default: 60"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for trained model. Default: artifacts/hmm/rolling_<window>_<timestamp> (manifest-resolved)"
    )

    parser.add_argument(
        "--n-components",
        type=int,
        help="Number of HMM states. Default: from config (4)"
    )

    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Save training dataset to disk for reproducibility"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name for saved dataset. Default: rolling_<window>_<date>"
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh data from API even if cached"
    )

    args = parser.parse_args()

    # Parse window
    try:
        window_days = parse_time_window(args.window)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Research-only flag: set when --symbols is provided explicitly.
    # Research runs skip promotion to protect production model stability.
    research_only = bool(args.symbols)

    # ── Symbol selection via canonical pipeline or explicit list ──────
    symbols: list[str] = []
    canonical_datasets: dict[str, "pd.DataFrame"] | None = None

    if args.symbols:
        # Explicit symbol list -- research-only run
        symbols = args.symbols
        logger.info(
            "Using %d specified symbols (research-only, promotion skipped)",
            len(symbols),
        )
    else:
        # Canonical pipeline: shared universe selector + Vision store
        logger.info("Running canonical retrain pipeline (Vision store)...")
        client = BinanceClient()
        try:
            probe = await build_frozen_probe_universe(
                client, target_accepted=args.top_n,
            )

            store_roots = await ensure_canonical_store_coverage(
                probe.ranked_symbols,
                min_years=window_days / 365.25,
            )

            boundary = freeze_run_boundary(
                store_roots, probe.selection_time_utc, window_days=window_days,
            )

            screening = screen_probe_universe(probe, store_roots, boundary)

            config = get_config()
            frozen_output_dir = (
                config.resolve_path(config.artifacts.artifacts_dir)
                / "hmm"
                / "training_sets"
                / f"canonical_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            )
            frozen_slice = freeze_training_slice(
                screening, store_roots, boundary, frozen_output_dir,
            )

            symbols = screening.accepted
            canonical_datasets = frozen_slice.datasets

        except Exception as e:
            logger.error("Canonical pipeline failed: %s", e, exc_info=True)
            sys.exit(1)
        finally:
            await client.close()

    if not symbols:
        logger.error("No symbols available for retraining")
        sys.exit(1)

    # Parse output directory
    output_dir = Path(args.output) if args.output else None

    # Run retraining
    try:
        artifact_path = await retrain_with_rolling_window(
            symbols=symbols,
            window_days=window_days,
            artifact_dir=output_dir,
            save_dataset=args.save_dataset,
            dataset_name=args.dataset_name,
            n_components=args.n_components,
            force_refresh=args.force_refresh,
            _override_datasets=canonical_datasets,
            _research_only=research_only,
        )

        logger.info("=" * 60)
        logger.info("[OK] Retraining complete")
        logger.info("Model saved to: %s", artifact_path)
        if research_only:
            logger.info("(Research-only run -- promotion was skipped)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Retraining failed: %s", e, exc_info=True)
        sys.exit(1)


def cli_main():
    """Synchronous entry point for ``[project.scripts]``."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
