#!/usr/bin/env python3
"""
Scan Top 100 USDT-M Perpetual Futures by Quote Volume.

Fetches market data for the top 100 symbols and scores them against
the trained pattern profile and profile model.

Usage:
    python scan_top100.py [--top-n N] [--output FILE]

Examples:
    # Scan top 100 symbols (default)
    python scan_top100.py

    # Scan top 50 symbols
    python scan_top100.py --top-n 50

    # Save results to custom location
    python scan_top100.py --output results/scan_custom.csv

Environment:
    BINANCE_API_KEY (optional) - Binance API key for authenticated endpoints
    BINANCE_API_SECRET (optional) - Binance API secret

Output:
    CSV file with columns:
    - symbol: Trading pair symbol
    - score: Combined similarity + model score (0-100)
    - winner_proba: Probability from profile model (0-1)
    - notes: Qualitative assessment notes
    - [feature columns]: All computed features
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from neutralgrid.core.config import get_config
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.scanner.scan import scan_top_symbols
from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.scanner.profile_model import load_profile_model
from neutralgrid.scanner.profile_model_walkforward import (
    resolve_active_pattern_profile_path,
    resolve_active_profile_model_path,
)

# AFML Training: Feature collection for meta-labeler training
try:
    from neutralgrid.training import FeatureCollector
    FEATURE_LOGGING_AVAILABLE = True
except ImportError:
    FEATURE_LOGGING_AVAILABLE = False
    FeatureCollector = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scan_top100")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan top USDT-M perpetuals and score against winner profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top symbols to scan by quote volume (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results/neutralgrid_candidates_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent API requests (default: 8)",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.05,
        help="Minimum delay between requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help=(
            "Explicit path to pattern profile JSON. Default (None) resolves "
            "the pattern artifact paired with the active model bundle."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit path to profile model JSON. Default (None) uses "
            "resolve_active_profile_model_path(): use data/profile/current.json when "
            "present, otherwise fall back to data/profile/profile_model.json as an "
            "unvalidated bootstrap candidate."
        ),
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading profile model (use pattern profile only)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum score threshold for output (default: None, output all)",
    )
    # AFML H2: AFML scoring enablement flags
    parser.add_argument(
        "--enable-afml-scoring",
        action="store_true",
        help="Enable AFML-compliant EV scoring with PnLRanker and MetaLabeler",
    )
    parser.add_argument(
        "--pnl-ranker-path",
        type=str,
        default="models/pnl_ranker.json",
        help="Path to PnLRanker config (default: models/pnl_ranker.json)",
    )
    parser.add_argument(
        "--meta-labeler-path",
        type=str,
        default="models/meta_labeler.pkl",
        help="Path to trained MetaLabeler (default: models/meta_labeler.pkl)",
    )
    # AFML Training: Feature logging for future training data
    parser.add_argument(
        "--log-features",
        action="store_true",
        default=None,
        help="Enable feature snapshot logging for meta-labeler training (default: from config.ENABLE_FEATURE_LOGGING)",
    )
    parser.add_argument(
        "--no-log-features",
        action="store_true",
        help="Disable feature snapshot logging",
    )
    parser.add_argument(
        "--feature-log-dir",
        type=str,
        default=None,
        help="Directory for feature snapshots (default: from config.FEATURE_LOG_DIR)",
    )
    parser.add_argument(
        "--compute-stochastic",
        action="store_true",
        default=None,
        help="Compute stochastic features (survival_prob, hurst, ou_halflife) - slower but complete",
    )
    parser.add_argument(
        "--no-compute-stochastic",
        action="store_true",
        help="Skip stochastic feature computation for faster scanning",
    )
    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_args()

    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"neutralgrid_candidates_{timestamp}.csv"

    logger.info("=" * 70)
    logger.info("NEUTRAL Grid Scanner - Top Symbols by Quote Volume")
    logger.info("=" * 70)

    # Load profile artifacts
    if args.profile_path is None:
        cfg = get_config()
        profile_path = resolve_active_pattern_profile_path(
            cfg.resolve_path(cfg.artifacts.profile_dir)
        )
        logger.info(
            "Loading default pattern profile via current.json/bootstrap → %s",
            profile_path,
        )
    else:
        profile_path = Path(args.profile_path)
        logger.info("Loading pattern profile from explicit --profile-path: %s", profile_path)
    if not profile_path.exists():
        logger.error(f"Pattern profile not found: {profile_path}")
        logger.error("Run retrain_scanner.py first to generate profile artifacts")
        sys.exit(1)

    try:
        pattern_profile = PatternProfile.load_json(profile_path)
        logger.info(f"✓ Pattern profile loaded ({len(pattern_profile.features)} features)")
    except Exception as e:
        logger.error(f"Failed to load pattern profile: {e}")
        sys.exit(1)

    # Load profile model (optional)
    profile_model = None
    if not args.no_model:
        if args.model_path is None:
            cfg = get_config()
            model_path = resolve_active_profile_model_path(
                cfg.resolve_path(cfg.artifacts.profile_dir)
            )
            logger.info(f"Loading default profile model via current.json/bootstrap → {model_path}")
        else:
            model_path = Path(args.model_path)
            logger.info(f"Loading profile model from explicit --model-path: {model_path}")
        if model_path.exists():
            try:
                profile_model = load_profile_model(model_path)
                logger.info(f"✓ Profile model loaded ({len(profile_model.features)} features)")
            except Exception as e:
                logger.warning(f"Failed to load profile model: {e}")
                logger.warning("Continuing with pattern profile only")
        else:
            logger.warning(f"Profile model not found: {model_path}")
            logger.warning("Continuing with pattern profile only")
    if profile_model is not None and profile_model.features != pattern_profile.features:
        logger.error(
            "Profile bundle feature mismatch: model features %s != pattern features %s",
            profile_model.features,
            pattern_profile.features,
        )
        sys.exit(1)

    # AFML H2: Load AFML scoring components if enabled
    pnl_ranker = None
    meta_labeler = None

    if args.enable_afml_scoring:
        logger.info("AFML scoring enabled - loading components...")

        # Load PnLRanker
        try:
            from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
            pnl_ranker = PnLRanker(RankingConfig())
            logger.info("✓ PnLRanker initialized with default config")
        except ImportError as e:
            logger.warning(f"Failed to import PnLRanker: {e}")
            logger.warning("EV scoring will be unavailable")
        except Exception as e:
            logger.warning(f"Failed to initialize PnLRanker: {e}")

        # Load MetaLabeler if artifact exists
        try:
            from neutralgrid.models.meta_labeler import MetaLabeler
            meta_labeler_path = Path(args.meta_labeler_path)
            if meta_labeler_path.exists():
                try:
                    meta_labeler = MetaLabeler.load(meta_labeler_path)
                    logger.info(f"✓ MetaLabeler loaded from {meta_labeler_path}")
                except Exception as e:
                    logger.warning(f"Failed to load MetaLabeler: {e}")
                    logger.warning("Meta-labeling will use default proba (0.5)")
            else:
                logger.warning(f"MetaLabeler artifact not found: {meta_labeler_path}")
                logger.warning("Meta-labeling will use default proba (0.5)")
        except ImportError as e:
            logger.warning(f"Failed to import MetaLabeler: {e}")

    # AFML Training: Initialize feature collector for training data collection
    feature_collector = None
    if FEATURE_LOGGING_AVAILABLE and FeatureCollector is not None:
        # Determine if feature logging is enabled
        log_features_enabled = args.log_features
        if log_features_enabled is None and not args.no_log_features:
            # Use config default
            log_features_enabled = get_config().feature_logging.enable
        elif args.no_log_features:
            log_features_enabled = False

        if log_features_enabled:
            feature_log_dir = args.feature_log_dir or get_config().feature_logging.log_dir
            feature_collector = FeatureCollector(
                log_dir=feature_log_dir,
                enabled=True,
            )
            logger.info(f"✓ Feature logging enabled -> {feature_log_dir}")
    elif args.log_features:
        logger.warning("Feature logging requested but training module not available")

    # Initialize Binance client
    logger.info("Initializing Binance client...")
    client = BinanceClient()

    # Check connection
    try:
        status = await client.check_connection()
        if not status.get("connected"):
            logger.error("Failed to connect to Binance API")
            sys.exit(1)
        logger.info(f"✓ Connected to Binance (auth: {status.get('authenticated', False)})")
    except Exception as e:
        logger.error(f"Connection check failed: {e}")
        sys.exit(1)

    # Run scan
    logger.info("-" * 70)
    logger.info(f"Scanning top {args.top_n} symbols by quote volume...")
    logger.info(f"Concurrency: {args.max_concurrency}, Min delay: {args.min_delay}s")
    logger.info("-" * 70)

    # Determine if we should compute stochastic features
    compute_stochastic = args.compute_stochastic
    if compute_stochastic is None and not args.no_compute_stochastic:
        # Default: compute stochastic if feature logging is enabled
        compute_stochastic = (feature_collector is not None)
    elif args.no_compute_stochastic:
        compute_stochastic = False

    try:
        df = await scan_top_symbols(
            client=client,
            profile=pattern_profile,
            profile_model=profile_model,
            pnl_ranker=pnl_ranker,              # AFML H2: Pass PnLRanker
            meta_labeler=meta_labeler,           # AFML H2: Pass MetaLabeler
            feature_collector=feature_collector, # AFML Training: Pass FeatureCollector
            compute_stochastic=compute_stochastic,  # AFML Training: Stochastic features
            top_n=args.top_n,
            max_concurrency=args.max_concurrency,
            min_delay_s=args.min_delay,
        )
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        if feature_collector is not None:
            feature_collector.close()
        await client.close()
        sys.exit(1)

    # Filter by minimum score if specified
    if args.min_score is not None and not df.empty:
        original_count = len(df)
        df = df[df["score"] >= args.min_score].copy()
        filtered_count = len(df)
        logger.info(f"Filtered by min_score={args.min_score}: {filtered_count}/{original_count} symbols")

    # Save results
    if df.empty:
        logger.warning("No results to save")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("-" * 70)
        logger.info(f"✓ Results saved to: {output_path}")
        logger.info(f"Total candidates: {len(df)}")

        if len(df) > 0:
            logger.info(f"Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
            logger.info(f"Top 5 symbols by score:")
            for rank, (_, row) in enumerate(df.head(5).iterrows(), start=1):
                notes_str = row.get("notes", "")
                if notes_str and isinstance(notes_str, str):
                    notes_str = f" ({notes_str[:50]}...)" if len(notes_str) > 50 else f" ({notes_str})"
                else:
                    notes_str = ""
                logger.info(f"  {rank}. {row['symbol']:12s} score={row['score']:6.2f}{notes_str}")

    # AFML Training: Flush and report feature collection stats
    if feature_collector is not None:
        feature_collector.close()
        stats = feature_collector.stats
        if stats["total_logged"] > 0:
            logger.info(f"✓ Feature snapshots logged: {stats['total_logged']}")
            if stats["total_skipped"] > 0:
                logger.info(f"  Skipped: {stats['total_skipped']} (validation failures: {stats['validation_failures']})")

    # Cleanup
    await client.close()
    logger.info("=" * 70)
    logger.info("Scan complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
