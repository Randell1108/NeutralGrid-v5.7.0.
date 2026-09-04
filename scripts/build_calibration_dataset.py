#!/usr/bin/env python3
"""
Step 3: Build ML-ready calibration dataset from execution telemetry.

Transforms telemetry into a feature matrix suitable for threshold
tuning and feature importance analysis.

Output: data/calibration_dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Feature columns (X)
FEATURE_COLS = [
    "cand_score",
    "cand_ev_24h",
    "cand_meta_prob",
    "cand_winner_proba",
    "cand_grid_spacing_pct",
    "cand_profit_per_grid_pct",
    "cand_range_prob",
    "cand_trend_prob",
    "cand_survival_prob",
    "cand_hurst_exponent",
    "cand_regime_conf",
    "cand_net_edge_pct",
    "cand_micro_viable",
    "duration_hours",
    "grids_count",
]

# Label columns (y)
LABEL_COLS = [
    "outcome_label",
    "pnl_pct",
    "outcome_tier",
    "profit_factor",
]

# Weight column
WEIGHT_COL = "invested_margin_usdt"


def main():
    print("=" * 70)
    print("STEP 3: Build Calibration Dataset")
    print("=" * 70)

    in_path = PROJECT_ROOT / "data" / "execution_telemetry.csv"
    if not in_path.exists():
        print(f"Error: {in_path} not found. Run build_execution_telemetry.py first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from telemetry")

    # Keep only matched rows
    df = df[df["matched"] == True].copy()  # noqa: E712
    print(f"After filtering matched-only: {len(df)} rows")

    if df.empty:
        print("No matched rows found. Cannot build calibration dataset.")
        sys.exit(1)

    # Select columns
    keep_cols = FEATURE_COLS + LABEL_COLS + [WEIGHT_COL]
    # Only keep columns that actually exist
    available = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"Note: Missing columns (will be NaN): {missing}")

    cal = df[available].copy()

    # Add missing columns as NaN
    for c in missing:
        cal[c] = np.nan

    # Convert cand_micro_viable to numeric (boolean -> 0/1)
    if "cand_micro_viable" in cal.columns:
        cal["cand_micro_viable"] = pd.to_numeric(
            cal["cand_micro_viable"].map(
                {True: 1, False: 0, "True": 1, "False": 0}
            ),
            errors="coerce",
        )

    # Fill NaN features with column medians
    for col in FEATURE_COLS:
        if col in cal.columns and cal[col].isna().any():
            median_val = cal[col].median()
            n_filled = cal[col].isna().sum()
            cal[col] = cal[col].fillna(median_val)
            if n_filled > 0:
                print(f"  Filled {n_filled} NaN in {col} with median={median_val:.4f}")

    # Add interaction terms
    cal["score_x_ev"] = cal["cand_score"] * cal["cand_ev_24h"]
    cal["meta_x_survival"] = cal["cand_meta_prob"] * cal["cand_survival_prob"]

    # Reorder: features, interactions, labels, weight
    interaction_cols = ["score_x_ev", "meta_x_survival"]
    final_order = FEATURE_COLS + interaction_cols + LABEL_COLS + [WEIGHT_COL]
    final_order = [c for c in final_order if c in cal.columns]
    cal = cal[final_order]

    # Summary
    print(f"\nCalibration dataset: {len(cal)} rows x {len(cal.columns)} columns")
    print(f"Features: {len(FEATURE_COLS)} + {len(interaction_cols)} interactions")
    print(f"Win rate: {(cal['outcome_label'] == 1).mean():.1%}")
    print(f"Mean PnL%: {cal['pnl_pct'].mean():.2f}%")

    # Save
    out_path = PROJECT_ROOT / "data" / "calibration_dataset.csv"
    cal.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
