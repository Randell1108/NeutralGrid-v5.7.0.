"""
CLI entrypoint for the offline order-book replay pipeline.

Usage:
    neutralgrid-replay --input data/manual_exports --output data/replay_outputs
    neutralgrid-replay --input data/manual_exports --sessions data/execution_telemetry.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from neutralgrid.replay.io_scan import scan_directory
from neutralgrid.replay.normalize import (
    load_orders,
    load_trades,
    apply_window_filter,
    load_sessions,
    detect_overlaps,
    filter_overlapping_orders,
)
from neutralgrid.replay.join_trades import build_lifecycles
from neutralgrid.replay.events import build_events, split_events_by_symbol
from neutralgrid.replay.sweep import replay_all
from neutralgrid.replay.export import export_all

logger = logging.getLogger("neutralgrid.replay")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutralgrid-replay",
        description="Offline order-book replay from Binance CSV exports",
    )
    p.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing order/trade CSV exports",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/replay_outputs"),
        help="Output directory for replay results (default: data/replay_outputs)",
    )
    p.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help=(
            "Optional execution telemetry CSV for overlap detection. "
            "Expected columns: symbol, start_time_utc, end_time_utc"
        ),
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Replay window in days (default: 90)",
    )
    p.add_argument(
        "--no-cap-open",
        action="store_false",
        dest="cap_open",
        default=True,
        help="Do not cap still-open orders (omit REMOVE events for NEW orders). "
             "By default, still-open orders are capped at replay window end.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    sessions_path: Path | None = None,
    window_days: int = 90,
    cap_open: bool = True,
) -> dict[str, Path]:
    """Execute the full replay pipeline programmatically.

    Returns a dict of symbol -> output directory path.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")

    t0 = time.time()

    # ── Step 1: Scan input directory ─────────────────────────────────────
    logger.info("=" * 72)
    logger.info("STEP 1: Scanning input directory")
    logger.info("-" * 40)
    scan_result = scan_directory(input_dir)

    if not scan_result.order_files:
        logger.error("No order CSV files found in %s", input_dir)
        raise FileNotFoundError(
            f"No order files found in {input_dir}. "
            "Expected CSVs with columns: Order No, Time(UTC), etc."
        )

    # ── Step 2: Load and normalize ───────────────────────────────────────
    logger.info("")
    logger.info("STEP 2: Loading and normalizing CSV data")
    logger.info("-" * 40)
    orders = load_orders(scan_result.order_files)
    trades = load_trades(scan_result.trade_files)

    # ── Step 3: Apply time window ────────────────────────────────────────
    logger.info("")
    logger.info("STEP 3: Applying %d-day window filter", window_days)
    logger.info("-" * 40)
    orders, trades, _t_start_ms, t_end_ms = apply_window_filter(
        orders, trades, window_days=window_days,
    )

    if not orders:
        logger.warning("No orders remain after window filter")
        return {}

    # ── Step 3b: Overlap detection (optional) ────────────────────────────
    if sessions_path is not None:
        logger.info("")
        logger.info("STEP 3b: Detecting bot-session overlaps")
        logger.info("-" * 40)
        sessions = load_sessions(sessions_path)
        overlaps = detect_overlaps(sessions)
        if overlaps:
            orders, n_disc = filter_overlapping_orders(orders, overlaps)
            logger.info(
                "  %d overlap intervals found, %d orders discarded",
                len(overlaps), n_disc,
            )
        else:
            logger.info("  No overlaps detected")

    # ── Step 4: Join trades and build lifecycles ─────────────────────────
    logger.info("")
    logger.info("STEP 4: Building order lifecycles")
    logger.info("-" * 40)
    lifecycles, anomalies = build_lifecycles(orders, trades)

    # ── Step 5: Generate event stream ────────────────────────────────────
    logger.info("")
    logger.info("STEP 5: Generating event stream")
    logger.info("-" * 40)
    events = build_events(
        lifecycles,
        t_end_ms=t_end_ms if cap_open else None,
    )
    events_by_symbol = split_events_by_symbol(events)
    logger.info("  %d symbols in event stream", len(events_by_symbol))

    # ── Step 6: Sweep-line replay ────────────────────────────────────────
    logger.info("")
    logger.info("STEP 6: Running sweep-line replay")
    logger.info("-" * 40)
    results = replay_all(events_by_symbol)

    # ── Step 7: Export ───────────────────────────────────────────────────
    logger.info("")
    logger.info("STEP 7: Exporting results")
    logger.info("-" * 40)
    symbol_dirs = export_all(results, trades, anomalies, output_dir)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "Replay pipeline complete: %d symbols, %.1fs elapsed",
        len(symbol_dirs), elapsed,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 72)

    return symbol_dirs


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(verbose=args.verbose)

    try:
        run_pipeline(
            input_dir=args.input,
            output_dir=args.output,
            sessions_path=args.sessions,
            window_days=args.window_days,
            cap_open=args.cap_open,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
