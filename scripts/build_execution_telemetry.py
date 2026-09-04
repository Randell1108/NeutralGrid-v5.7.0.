#!/usr/bin/env python3
"""
Step 2: Build execution telemetry — predicted vs actual deltas.

Reads candidate_execution_linkage.csv and computes prediction accuracy
metrics for each matched bot-candidate pair.

Output: data/execution_telemetry.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute predicted-vs-actual deltas for matched rows."""
    df = df.copy()

    # Grid parameter deltas (only for matched rows)
    matched = df["matched"] == True  # noqa: E712

    # Grid lower delta
    df.loc[matched, "grid_lower_delta_pct"] = (
        (df.loc[matched, "price_range_low"] - df.loc[matched, "cand_grid_lower"])
        / df.loc[matched, "cand_grid_lower"]
        * 100
    )

    # Grid upper delta
    df.loc[matched, "grid_upper_delta_pct"] = (
        (df.loc[matched, "price_range_high"] - df.loc[matched, "cand_grid_upper"])
        / df.loc[matched, "cand_grid_upper"]
        * 100
    )

    # Grids count delta
    df.loc[matched, "grids_count_delta"] = (
        df.loc[matched, "grids_count"] - df.loc[matched, "cand_num_grids"]
    )

    # Actual profit per grid (simplified)
    df["actual_profit_per_grid_pct"] = np.where(
        df["grids_count"] > 0,
        df["pnl_pct"] / df["grids_count"],
        np.nan,
    )

    # Predicted vs actual profit
    df.loc[matched, "predicted_vs_actual_profit_pct"] = (
        df.loc[matched, "cand_profit_per_grid_pct"]
        - df.loc[matched, "actual_profit_per_grid_pct"]
    )

    # Outcome label (binary)
    df["outcome_label"] = np.where(df["pnl_pct"] > 0, 1, 0)

    # Outcome tier
    conditions = [
        df["pnl_pct"] > 5,
        df["pnl_pct"] > 0,
        df["pnl_pct"] >= -3,
        df["pnl_pct"] < -3,
    ]
    choices = ["big_win", "small_win", "small_loss", "big_loss"]
    df["outcome_tier"] = np.select(conditions, choices, default="unknown")

    # Score accuracy: did high score predict win?
    df.loc[matched, "score_accuracy"] = np.where(
        (df.loc[matched, "cand_score"] >= 45) & (df.loc[matched, "pnl_pct"] > 0),
        1,
        np.where(
            (df.loc[matched, "cand_score"] < 45) & (df.loc[matched, "pnl_pct"] <= 0),
            1,
            0,
        ),
    )

    # EV accuracy: positive EV -> positive PnL?
    df.loc[matched, "ev_accuracy"] = np.where(
        (df.loc[matched, "cand_ev_24h"] > 0) & (df.loc[matched, "pnl_pct"] > 0),
        1,
        np.where(
            (df.loc[matched, "cand_ev_24h"] <= 0) & (df.loc[matched, "pnl_pct"] <= 0),
            1,
            0,
        ),
    )

    # Meta-prob accuracy: meta_prob > 0.5 -> win?
    df.loc[matched, "meta_prob_accuracy"] = np.where(
        (df.loc[matched, "cand_meta_prob"] > 0.5) & (df.loc[matched, "pnl_pct"] > 0),
        1,
        np.where(
            (df.loc[matched, "cand_meta_prob"] <= 0.5)
            & (df.loc[matched, "pnl_pct"] <= 0),
            1,
            0,
        ),
    )

    return df


def print_summary(df: pd.DataFrame):
    """Print console summary of telemetry."""
    matched = df[df["matched"] == True].copy()  # noqa: E712
    if matched.empty:
        print("No matched rows to summarize.")
        return

    print("\n" + "=" * 70)
    print("EXECUTION TELEMETRY SUMMARY")
    print("=" * 70)

    # Overall stats
    n = len(matched)
    win_rate = (matched["outcome_label"] == 1).mean()
    mean_pnl = matched["pnl_pct"].mean()
    print(f"\nOverall: {n} matched bots, win rate={win_rate:.1%}, mean PnL={mean_pnl:.2f}%")

    # Win rate by score bucket
    print("\n--- Win Rate by Score Bucket ---")
    bins = [0, 45, 55, 65, 100]
    labels = ["<45", "45-55", "55-65", "65+"]
    matched["score_bucket"] = pd.cut(
        matched["cand_score"], bins=bins, labels=labels, right=False
    )
    for bucket in labels:
        subset = matched[matched["score_bucket"] == bucket]
        if len(subset) > 0:
            wr = (subset["outcome_label"] == 1).mean()
            mp = subset["pnl_pct"].mean()
            print(f"  {bucket:>6s}: n={len(subset):2d}, win_rate={wr:.1%}, mean_pnl={mp:+.2f}%")
        else:
            print(f"  {bucket:>6s}: n= 0")

    # EV prediction accuracy
    print("\n--- EV Prediction Accuracy ---")
    ev_valid = matched.dropna(subset=["cand_ev_24h"])
    if len(ev_valid) > 0:
        pos_ev = ev_valid[ev_valid["cand_ev_24h"] > 0]
        if len(pos_ev) > 0:
            pos_ev_wr = (pos_ev["pnl_pct"] > 0).mean()
            print(f"  Positive EV candidates: {len(pos_ev)}, actual win rate: {pos_ev_wr:.1%}")
        neg_ev = ev_valid[ev_valid["cand_ev_24h"] <= 0]
        if len(neg_ev) > 0:
            neg_ev_wr = (neg_ev["pnl_pct"] > 0).mean()
            print(f"  Negative/zero EV candidates: {len(neg_ev)}, actual win rate: {neg_ev_wr:.1%}")
        overall_ev_acc = ev_valid["ev_accuracy"].mean()
        print(f"  Overall EV accuracy: {overall_ev_acc:.1%}")
    else:
        print("  No EV data available.")

    # Meta-labeler precision/recall
    print("\n--- Meta-Labeler Performance ---")
    meta_valid = matched.dropna(subset=["cand_meta_prob"])
    if len(meta_valid) > 0:
        pred_pos = meta_valid[meta_valid["cand_meta_prob"] > 0.5]
        actual_pos = meta_valid[meta_valid["outcome_label"] == 1]

        if len(pred_pos) > 0:
            tp = ((pred_pos["outcome_label"] == 1).sum())
            precision = tp / len(pred_pos)
            print(f"  Precision (meta>0.5 -> win): {precision:.1%} ({tp}/{len(pred_pos)})")
        else:
            print("  No predictions > 0.5")

        if len(actual_pos) > 0:
            tp_recall = (actual_pos["cand_meta_prob"] > 0.5).sum()
            recall = tp_recall / len(actual_pos)
            print(f"  Recall (win -> meta>0.5): {recall:.1%} ({tp_recall}/{len(actual_pos)})")
        else:
            print("  No actual wins.")

        overall_meta_acc = meta_valid["meta_prob_accuracy"].mean()
        print(f"  Overall meta-prob accuracy: {overall_meta_acc:.1%}")
    else:
        print("  No meta-prob data available.")

    # Outcome tier distribution
    print("\n--- Outcome Tier Distribution ---")
    tier_counts = matched["outcome_tier"].value_counts()
    for tier in ["big_win", "small_win", "small_loss", "big_loss"]:
        count = tier_counts.get(tier, 0)
        print(f"  {tier:>12s}: {count}")

    # Grid parameter accuracy
    print("\n--- Grid Parameter Deltas (matched) ---")
    for col in ["grid_lower_delta_pct", "grid_upper_delta_pct", "grids_count_delta"]:
        valid = matched[col].dropna()
        if len(valid) > 0:
            print(f"  {col}: mean={valid.mean():.2f}, std={valid.std():.2f}")

    print()


def main():
    print("=" * 70)
    print("STEP 2: Build Execution Telemetry")
    print("=" * 70)

    in_path = PROJECT_ROOT / "data" / "candidate_execution_linkage.csv"
    if not in_path.exists():
        print(f"Error: {in_path} not found. Run match_candidates_to_bots.py first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from linkage file")

    df = compute_telemetry(df)

    print_summary(df)

    out_path = PROJECT_ROOT / "data" / "execution_telemetry.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
