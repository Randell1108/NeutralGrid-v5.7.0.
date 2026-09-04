#!/usr/bin/env python3
"""
Step 1: Match deployed bots to their pipeline scan candidates.

Links each deployed bot (from new_expired_bots.xlsx, Feb 11+) to the
deployment_ready CSV candidate that preceded it, enabling prediction
accuracy analysis.

Output: data/candidate_execution_linkage.csv
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_bots(min_date: str = "2026-02-11") -> pd.DataFrame:
    """Load deployed bots from Excel, filtering to start_time >= min_date."""
    path = PROJECT_ROOT / "data" / "new_expired_bots.xlsx"
    df = pd.read_excel(path)

    # Parse start_time_utc to datetime
    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)
    df["end_time_utc"] = pd.to_datetime(df["end_time_utc"], utc=True)

    cutoff = pd.Timestamp(min_date, tz="UTC")
    df = df[df["start_time_utc"] >= cutoff].copy()
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} bots with start_time >= {min_date}")
    return df


def parse_csv_timestamps(results_dir: Path) -> list[tuple[datetime, Path]]:
    """Parse all deployment_ready CSVs and extract their timestamps."""
    pattern = re.compile(r"deployment_ready_(\d{8}_\d{6})\.csv$")
    csv_entries = []

    for f in sorted(results_dir.glob("deployment_ready_*.csv")):
        m = pattern.search(f.name)
        if m:
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
            csv_entries.append((ts, f))

    csv_entries.sort(key=lambda x: x[0])
    print(f"Found {len(csv_entries)} timestamped deployment_ready CSVs")
    return csv_entries


# Candidate columns to extract (use .get() for columns that may not exist)
CANDIDATE_FIELDS = {
    "cand_score": "score",
    "cand_ev_score": "ev_score",
    "cand_ev_24h": "ev_24h",
    "cand_meta_prob": "meta_prob",
    "cand_winner_proba": "winner_proba",
    "cand_grid_lower": "grid_lower",
    "cand_grid_upper": "grid_upper",
    "cand_num_grids": "num_grids",
    "cand_grid_spacing_pct": "grid_spacing_pct",
    "cand_profit_per_grid_pct": "profit_per_grid_pct",
    "cand_leverage": "leverage",
    "cand_range_prob": "range_prob",
    "cand_trend_prob": "trend_prob",
    "cand_survival_prob": "survival_prob",
    "cand_hurst_exponent": "hurst_exponent",
    "cand_grid_is_valid": "grid_is_valid",
    "cand_net_edge_pct": "net_edge_pct",
    "cand_micro_viable": "micro_viable",
    "cand_regime_conf": "regime_conf",
}


def find_candidate_in_csv(
    symbol: str, csv_path: Path
) -> dict | None:
    """Look up a symbol in a CSV. Return candidate fields if found and grid_is_valid."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Warning: Could not read {csv_path.name}: {e}")
        return None

    row = df[df["symbol"] == symbol]
    if row.empty:
        return None

    row = row.iloc[0]

    # Check grid_is_valid — handle string "True"/"False" and boolean
    grid_valid = row.get("grid_is_valid", False)
    if isinstance(grid_valid, str):
        grid_valid = grid_valid.strip().lower() == "true"

    if not grid_valid:
        return None

    # Extract candidate fields
    result = {}
    for out_col, src_col in CANDIDATE_FIELDS.items():
        result[out_col] = row.get(src_col, np.nan)

    # Handle ev_24h vs ev_48h for older CSVs
    if pd.isna(result.get("cand_ev_24h")):
        result["cand_ev_24h"] = row.get("ev_48h", np.nan)

    return result


def match_bots_to_candidates(
    bots: pd.DataFrame, csv_entries: list[tuple[datetime, Path]]
) -> pd.DataFrame:
    """For each bot, find the matching candidate from the closest preceding CSV."""
    results = []

    for _, bot in bots.iterrows():
        bot_start = bot["start_time_utc"]
        symbol = bot["symbol"]

        # Find CSVs that precede this bot's start time
        preceding = [
            (ts, path) for ts, path in csv_entries if ts < bot_start
        ]
        # Sort by timestamp descending (most recent first)
        preceding.sort(key=lambda x: x[0], reverse=True)

        matched = False
        match_method = np.nan
        cand_csv_file = np.nan
        cand_csv_timestamp = np.nan
        hours_between = np.nan
        cand_fields = {k: np.nan for k in CANDIDATE_FIELDS}

        # Try up to 3 closest CSVs
        for i, (csv_ts, csv_path) in enumerate(preceding[:3]):
            candidate = find_candidate_in_csv(symbol, csv_path)
            if candidate is not None:
                matched = True
                match_method = ["closest", "2nd", "3rd"][i]
                cand_csv_file = csv_path.name
                cand_csv_timestamp = csv_ts.isoformat()
                hours_between = (bot_start - csv_ts).total_seconds() / 3600
                cand_fields = candidate
                break

        # Build output row
        row = {
            # Bot fields
            "strategy_id": bot.get("strategy_id", np.nan),
            "symbol": symbol,
            "start_time_utc": bot["start_time_utc"],
            "end_time_utc": bot.get("end_time_utc", np.nan),
            "duration_hours": bot.get("duration_hours", np.nan),
            "invested_margin_usdt": bot.get("invested_margin_usdt", np.nan),
            "leverage": bot.get("leverage", np.nan),
            "grids_count": bot.get("grids_count", np.nan),
            "price_range_low": bot.get("price_range_low", np.nan),
            "price_range_high": bot.get("price_range_high", np.nan),
            "total_profit_usdt": bot.get("total_profit_usdt", np.nan),
            "pnl_pct": bot.get("pnl_pct", np.nan),
            "realized_pnl_usdt": bot.get("realized_pnl_usdt", np.nan),
            "commission_usdt": bot.get("commission_usdt", np.nan),
            "profit_factor": bot.get("profit_factor", np.nan),
            "total_trades": bot.get("total_trades", np.nan),
            "trend_structure": bot.get("trend_structure", np.nan),
            # Candidate fields
            **cand_fields,
            # Linkage metadata
            "cand_csv_file": cand_csv_file,
            "cand_csv_timestamp": cand_csv_timestamp,
            "matched": matched,
            "match_method": match_method,
            "hours_between_scan_and_deploy": hours_between,
        }
        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("STEP 1: Match Candidates to Deployed Bots")
    print("=" * 70)

    bots = load_bots()
    csv_entries = parse_csv_timestamps(PROJECT_ROOT / "results")
    linkage = match_bots_to_candidates(bots, csv_entries)

    # Summary
    n_matched = linkage["matched"].sum()
    n_total = len(linkage)
    print(f"\nResults: {n_matched}/{n_total} bots matched to candidates")

    unmatched = linkage[~linkage["matched"]]["symbol"].tolist()
    if unmatched:
        print(f"Unmatched symbols: {', '.join(unmatched)}")

    matched_rows = linkage[linkage["matched"]]
    if not matched_rows.empty:
        print(f"\nMatch methods:")
        print(matched_rows["match_method"].value_counts().to_string())
        print(f"\nMean hours between scan and deploy: "
              f"{matched_rows['hours_between_scan_and_deploy'].mean():.1f}h")

    # Save
    out_path = PROJECT_ROOT / "data" / "candidate_execution_linkage.csv"
    linkage.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Columns: {len(linkage.columns)}")


if __name__ == "__main__":
    main()
