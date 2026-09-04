#!/usr/bin/env python3
"""
Step 4: Calibrate pipeline thresholds from execution data.

Analyzes the calibration dataset to recommend better min_score, EV floor,
and meta_prob floor. Runs feature importance via GradientBoosting.

Output: data/calibration/calibrated_constants.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Current defaults from run_full_pipeline.py --min-score
CURRENT_MIN_SCORE = 45.0


def score_threshold_sweep(df: pd.DataFrame) -> list[dict]:
    """Test min_score thresholds from 40 to 70 in steps of 5."""
    results = []
    for threshold in range(40, 75, 5):
        subset = df[df["cand_score"] >= threshold]
        n = len(subset)
        if n == 0:
            results.append({
                "threshold": threshold,
                "n": 0,
                "win_rate": 0.0,
                "mean_pnl": 0.0,
                "objective": 0.0,
            })
            continue
        win_rate = (subset["outcome_label"] == 1).mean()
        mean_pnl = subset["pnl_pct"].mean()
        # Objective: win_rate * mean_pnl (only positive if both are good)
        objective = win_rate * mean_pnl if mean_pnl > 0 else -abs(win_rate * mean_pnl)
        results.append({
            "threshold": threshold,
            "n": int(n),
            "win_rate": round(win_rate, 4),
            "mean_pnl": round(mean_pnl, 4),
            "objective": round(objective, 4),
        })
    return results


def find_ev_threshold(df: pd.DataFrame) -> float:
    """Find EV cutoff that best separates winners from losers."""
    ev_col = df["cand_ev_24h"].dropna()
    if len(ev_col) < 4:
        return 0.0

    best_cutoff = 0.0
    best_separation = -np.inf

    for pct in np.arange(0.1, 0.9, 0.1):
        cutoff = ev_col.quantile(pct)
        above = df[df["cand_ev_24h"] >= cutoff]
        below = df[df["cand_ev_24h"] < cutoff]
        if len(above) == 0 or len(below) == 0:
            continue
        separation = above["pnl_pct"].mean() - below["pnl_pct"].mean()
        if separation > best_separation:
            best_separation = separation
            best_cutoff = cutoff

    return round(float(best_cutoff), 4)


def find_meta_prob_threshold(df: pd.DataFrame) -> float:
    """Find meta_prob cutoff that maximizes F1-score for outcome_label."""
    meta_col = df["cand_meta_prob"].dropna()
    if len(meta_col) < 4:
        return 0.5

    best_f1 = -1.0
    best_cutoff = 0.5

    for cutoff in np.arange(0.05, 0.95, 0.05):
        valid = df.dropna(subset=["cand_meta_prob"])
        pred = (valid["cand_meta_prob"] > cutoff).astype(int)
        actual = valid["outcome_label"]

        tp = ((pred == 1) & (actual == 1)).sum()
        fp = ((pred == 1) & (actual == 0)).sum()
        fn = ((pred == 0) & (actual == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_cutoff = cutoff

    return round(float(best_cutoff), 2)


def compute_feature_importances(df: pd.DataFrame) -> dict[str, float]:
    """Run GradientBoosting to get feature importances."""
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        print("  Warning: scikit-learn not installed. Skipping feature importance.")
        return {}

    feature_cols = [
        "cand_score", "cand_ev_24h", "cand_meta_prob", "cand_winner_proba",
        "cand_grid_spacing_pct", "cand_profit_per_grid_pct",
        "cand_range_prob", "cand_trend_prob", "cand_survival_prob",
        "cand_hurst_exponent", "cand_regime_conf", "cand_net_edge_pct",
        "cand_micro_viable", "duration_hours", "grids_count",
        "score_x_ev", "meta_x_survival",
    ]
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df["outcome_label"].copy()

    # Drop rows with any remaining NaN
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if len(X) < 5:
        print("  Too few samples for feature importance analysis.")
        return {}

    # Use conservative settings for small sample
    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    importances = dict(zip(available, [round(float(v), 4) for v in model.feature_importances_]))
    # Sort by importance descending
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    return importances


def main():
    print("=" * 70)
    print("STEP 4: Calibrate from Execution Data")
    print("=" * 70)

    in_path = PROJECT_ROOT / "data" / "calibration_dataset.csv"
    if not in_path.exists():
        print(f"Error: {in_path} not found. Run build_calibration_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(in_path)
    n = len(df)
    print(f"Loaded {n} rows from calibration dataset")

    if n < 3:
        print("Too few samples for calibration. Need at least 3.")
        sys.exit(1)

    # 1. Score threshold sweep
    print("\n--- Score Threshold Sweep ---")
    sweep = score_threshold_sweep(df)
    for s in sweep:
        marker = " <-- current" if s["threshold"] == CURRENT_MIN_SCORE else ""
        print(f"  score>={s['threshold']:2d}: n={s['n']:2d}, "
              f"win_rate={s['win_rate']:.1%}, mean_pnl={s['mean_pnl']:+.2f}%, "
              f"obj={s['objective']:+.4f}{marker}")

    # Find best threshold (highest objective with n >= 3)
    valid_sweep = [s for s in sweep if s["n"] >= 3]
    if valid_sweep:
        best = max(valid_sweep, key=lambda x: x["objective"])
        recommended_score = best["threshold"]
        win_rate_at_rec = best["win_rate"]
        mean_pnl_at_rec = best["mean_pnl"]
    else:
        recommended_score = CURRENT_MIN_SCORE
        win_rate_at_rec = 0.0
        mean_pnl_at_rec = 0.0

    print(f"\n  Recommended min_score: {recommended_score} "
          f"(current: {CURRENT_MIN_SCORE})")

    # 2. EV threshold
    print("\n--- EV Threshold Analysis ---")
    ev_floor = find_ev_threshold(df)
    print(f"  Recommended ev_floor: {ev_floor}")

    # 3. Meta-prob threshold
    print("\n--- Meta-Prob Threshold Analysis ---")
    meta_floor = find_meta_prob_threshold(df)
    print(f"  Recommended meta_prob_floor: {meta_floor}")

    # 4. Grid parameter accuracy
    print("\n--- Grid Parameter Correlation ---")
    if "cand_grid_spacing_pct" in df.columns:
        # We don't have actual grid_spacing_pct in calibration dataset directly,
        # but we can check grid count alignment via grids_count
        from scipy import stats
        if df["cand_grid_spacing_pct"].notna().sum() >= 3:
            corr, pval = stats.pearsonr(
                df["cand_grid_spacing_pct"].dropna(),
                df.loc[df["cand_grid_spacing_pct"].notna(), "pnl_pct"],
            )
            print(f"  Grid spacing vs PnL correlation: r={corr:.3f}, p={pval:.3f}")

    # 5. Feature importance
    print("\n--- Feature Importance (GradientBoosting) ---")
    importances = compute_feature_importances(df)
    for feat, imp in list(importances.items())[:10]:
        bar = "#" * int(imp * 50)
        print(f"  {feat:>30s}: {imp:.4f} {bar}")

    # Build output JSON
    output = {
        "calibration_date": str(date.today()),
        "sample_size": int(n),
        "recommended_min_score": int(recommended_score),
        "recommended_ev_floor": float(ev_floor),
        "recommended_meta_prob_floor": float(meta_floor),
        "win_rate_at_recommended": float(round(win_rate_at_rec, 4)),
        "mean_pnl_at_recommended": float(round(mean_pnl_at_rec, 4)),
        "current_min_score": float(CURRENT_MIN_SCORE),
        "feature_importances": importances,
        "score_sweep_results": sweep,
        "notes": (
            f"Sample size is small (n={n}). "
            "Treat as directional guidance, not definitive. "
            "Re-run as more bots accumulate for increasingly reliable calibrations."
        ),
    }

    # Save
    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibrated_constants.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"  Min Score:    current={CURRENT_MIN_SCORE} -> recommended={recommended_score}")
    print(f"  EV Floor:     recommended={ev_floor}")
    print(f"  Meta Floor:   recommended={meta_floor}")
    print(f"  Sample Size:  {n}")
    if n < 30:
        print(f"\n  WARNING: Sample size ({n}) is too small for statistical significance.")
        print("  These are directional recommendations only. Accumulate more")
        print("  bot data before adjusting production thresholds.")
    print()


if __name__ == "__main__":
    main()
