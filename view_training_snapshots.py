#!/usr/bin/env python3
"""
View and manage training data snapshots.

CLI tool for inspecting feature snapshots collected during scanning.

Usage:
    python view_training_snapshots.py                     # Summary of all snapshots
    python view_training_snapshots.py --detailed          # Detailed feature availability
    python view_training_snapshots.py --export FILE.csv   # Export to CSV for analysis
    python view_training_snapshots.py --date 2026-01-19   # View specific date
    python view_training_snapshots.py --tail 10           # Show last N snapshots
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from neutralgrid.core.config import get_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="View and manage training data snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory containing snapshots (default: from config.FEATURE_LOG_DIR)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed feature availability report",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export all snapshots to CSV file",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Filter to specific date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Show snapshots from last N days",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Show last N snapshots",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific symbols",
    )
    return parser.parse_args()


def load_snapshots(log_dir: Path, date_filter: Optional[str] = None, days: Optional[int] = None) -> pd.DataFrame:
    """Load snapshots from parquet files."""
    if not log_dir.exists():
        return pd.DataFrame()

    files = sorted(log_dir.glob("snapshots_*.parquet"))
    if not files:
        return pd.DataFrame()

    # Filter by date if specified
    if date_filter:
        files = [f for f in files if date_filter in f.name]
    elif days:
        cutoff = datetime.now() - timedelta(days=days)
        filtered = []
        for f in files:
            try:
                date_str = f.stem.replace("snapshots_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff:
                    filtered.append(f)
            except ValueError:
                continue
        files = filtered

    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Warning: Could not read {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("=" * 70)
    print("TRAINING DATA SNAPSHOTS SUMMARY")
    print("=" * 70)
    print(f"Total snapshots: {len(df)}")

    if df.empty:
        print("No snapshots found.")
        return

    # Date range
    if "start_time_utc" in df.columns:
        df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])
        min_date = df["start_time_utc"].min()
        max_date = df["start_time_utc"].max()
        print(f"Date range: {min_date.strftime('%Y-%m-%d %H:%M')} to {max_date.strftime('%Y-%m-%d %H:%M')}")

    # Unique symbols
    if "symbol" in df.columns:
        unique_symbols = df["symbol"].nunique()
        print(f"Unique symbols: {unique_symbols}")

    print()


def print_feature_availability(df: pd.DataFrame, detailed: bool = False):
    """Print feature availability report."""
    if df.empty:
        return

    feature_cols = [
        "range_prob", "trend_prob", "utility_score",
        "survival_prob", "hurst_exponent", "ou_halflife",
        "profit_per_grid_pct", "num_grids", "range_size_pct",
        "adx_1h", "adx_15m", "rsi_15m",
        "funding_rate", "ev_score",
    ]

    print("FEATURE AVAILABILITY")
    print("-" * 70)

    total = len(df)
    available_count = 0
    partial_count = 0
    missing_count = 0

    for col in feature_cols:
        if col not in df.columns:
            status = "MISSING"
            pct = 0.0
            missing_count += 1
        else:
            non_null = df[col].notna().sum()
            pct = (non_null / total) * 100 if total > 0 else 0.0
            if pct == 0:
                status = "MISSING"
                missing_count += 1
            elif pct == 100:
                status = "OK"
                available_count += 1
            else:
                status = "PARTIAL"
                partial_count += 1

        if detailed:
            if status == "OK":
                marker = "[OK]"
            elif status == "PARTIAL":
                marker = "[~]"
            else:
                marker = "[X]"
            print(f"  {marker} {col:25s}: {pct:5.1f}% available")
        elif status != "OK":
            if status == "PARTIAL":
                print(f"  [~] {col:25s}: {pct:5.1f}% available")
            else:
                print(f"  [X] {col:25s}: MISSING")

    print()
    print(f"Summary: {available_count}/14 fully available, {partial_count} partial, {missing_count} missing")
    print()


def print_recent_snapshots(df: pd.DataFrame, n: int = 10):
    """Print recent snapshots."""
    if df.empty:
        return

    print(f"RECENT SNAPSHOTS (last {n})")
    print("-" * 70)

    # Sort by time
    if "start_time_utc" in df.columns:
        df = df.sort_values("start_time_utc", ascending=False)

    recent = df.head(n)

    cols_to_show = ["symbol", "start_time_utc", "range_prob", "trend_prob", "ev_score"]
    cols_available = [c for c in cols_to_show if c in recent.columns]

    if not cols_available:
        print("No data to display.")
        return

    # Format for display
    display_df = recent[cols_available].copy()
    if "start_time_utc" in display_df.columns:
        display_df["start_time_utc"] = pd.to_datetime(display_df["start_time_utc"]).dt.strftime("%Y-%m-%d %H:%M")
    if "range_prob" in display_df.columns:
        display_df["range_prob"] = display_df["range_prob"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    if "trend_prob" in display_df.columns:
        display_df["trend_prob"] = display_df["trend_prob"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    if "ev_score" in display_df.columns:
        display_df["ev_score"] = display_df["ev_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    print(display_df.to_string(index=False))
    print()


def main():
    """Main function."""
    args = parse_args()

    # Get log directory
    log_dir = Path(args.log_dir) if args.log_dir else Path(
        get_config().feature_logging.log_dir
    )

    print(f"Loading snapshots from: {log_dir}")
    print()

    # Load snapshots
    df = load_snapshots(log_dir, date_filter=args.date, days=args.days)

    # Filter by symbols if specified
    if args.symbols and not df.empty and "symbol" in df.columns:
        df = df[df["symbol"].isin(args.symbols)]

    # Print summary
    print_summary(df)

    # Feature availability
    print_feature_availability(df, detailed=args.detailed)

    # Recent snapshots
    if args.tail:
        print_recent_snapshots(df, n=args.tail)
    elif not args.detailed and not args.export and len(df) > 0:
        print_recent_snapshots(df, n=5)

    # Export if requested
    if args.export and not df.empty:
        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_path, index=False)
        print(f"Exported {len(df)} snapshots to: {export_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
