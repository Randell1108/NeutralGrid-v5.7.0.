"""
Validate MC Range-Containment Accuracy for survival_prob >= 0.60.

TRENDING_OSCILLATOR.md Validation Criterion 4:
    "For trending symbols with survival_prob >= 0.60, verify MC
     range-containment accuracy."

This script evaluates whether the Monte Carlo survival_prob metric
is a statistically meaningful discriminator at the 0.60 threshold
for predicting range containment in grid bot outcomes.

Data sources used:
  1. data/training_sets/combined_training_20260228.csv
       -- Full survival_prob range [0, 1], 274 rows, with y/pnl_pct
  2. data/new_expired_bots.xlsx
       -- 139 live-traded bots with survival_prob and PnL outcomes
  3. data/backtest_candidates/backtest_results_*.csv (all dates)
       -- 3800+ backtests with survival_prob and time_in_range_pct

Output: results/mc_containment_validation.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from neutralgrid.core.config import get_config
except Exception:  # pragma: no cover - script fallback when src import unavailable
    get_config = None  # type: ignore[assignment]

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BACKTEST_DIR = DATA_DIR / "backtest_candidates"

THRESHOLD = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MC containment against outcome datasets",
    )
    parser.add_argument(
        "--backtest-dir",
        default=str(DEFAULT_BACKTEST_DIR),
        help="Directory containing backtest result CSVs",
    )
    parser.add_argument(
        "--backtest-pattern",
        default="backtest_results_*.csv",
        help="Glob for backtest result files inside --backtest-dir",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only the latest matching backtest results file",
    )
    return parser.parse_args()


def current_stochastic_settings() -> tuple[int, float, int]:
    """Return configured survival horizon in bars/hours and MC paths.

    Fails closed: if config cannot be loaded, raise rather than silently
    falling back to a historical literal (prior 56-bar fallback was
    removed per DURATION_FIX.md Step 5 / B9).
    """
    if get_config is None:
        raise RuntimeError(
            "neutralgrid.core.config.get_config is unavailable; "
            "cannot determine canonical survival horizon. Refusing to use a literal fallback."
        )
    cfg = get_config()
    horizon_bars = int(cfg.stochastic.survival_horizon_bars)
    mc_paths = int(cfg.stochastic.survival_mc_paths)
    horizon_hours = (horizon_bars * 15) / 60.0
    return horizon_bars, horizon_hours, mc_paths


def describe_backtest_survival_range(df: pd.DataFrame) -> str:
    """Summarise whether current backtest data still appears truncated."""
    if df.empty or "survival_prob" not in df.columns:
        return "backtest survival_prob range unavailable"
    sp_min = float(df["survival_prob"].min())
    sp_max = float(df["survival_prob"].max())
    if sp_min >= 0.49:
        return (
            f"observed range [{sp_min:.4f}, {sp_max:.4f}] still appears truncated "
            "near 0.50"
        )
    return (
        f"observed range [{sp_min:.4f}, {sp_max:.4f}] extends below 0.50; "
        "the truncation fix is visible in current data"
    )


# ── Data loaders ─────────��──────────────────────────────────────────────

def load_combined_training() -> pd.DataFrame:
    """Load Feb combined training set (full survival_prob range)."""
    path = DATA_DIR / "training_sets" / "combined_training_20260228.csv"
    if not path.exists():
        logger.warning("Combined training set not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.dropna(subset=["survival_prob", "y"])
    logger.info("Combined training: %d rows (survival_prob %.4f to %.4f)",
                len(df), df["survival_prob"].min(), df["survival_prob"].max())
    return df


def load_expired_bots() -> pd.DataFrame:
    """Load backfilled expired bots (live outcomes)."""
    path = DATA_DIR / "new_expired_bots.xlsx"
    if not path.exists():
        logger.warning("Expired bots not found: %s", path)
        return pd.DataFrame()
    df = pd.read_excel(path)
    df = df.dropna(subset=["survival_prob"])
    logger.info("Expired bots: %d rows (survival_prob %.4f to %.4f)",
                len(df), df["survival_prob"].min(), df["survival_prob"].max())
    return df


def load_all_backtest_results(
    backtest_dir: Path,
    filename_pattern: str = "backtest_results_*.csv",
    *,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Load all backtest_results CSVs that have both survival_prob and time_in_range_pct."""
    if not backtest_dir.exists():
        logger.warning("Backtest candidates dir not found: %s", backtest_dir)
        return pd.DataFrame()
    files = sorted(backtest_dir.glob(filename_pattern))
    if latest_only:
        files = files[-1:]
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if "survival_prob" in d.columns and "time_in_range_pct" in d.columns:
                cols = ["symbol", "survival_prob", "time_in_range_pct",
                        "net_pnl_pct", "max_drawdown_pct", "grid_lower",
                        "grid_upper", "price_start", "price_end"]
                sel = [c for c in cols if c in d.columns]
                sub = d[sel].copy()
                sub["source"] = f.name
                dfs.append(sub)
        except Exception as exc:
            logger.debug("Skipping %s: %s", f.name, exc)

    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["survival_prob", "time_in_range_pct"])
    logger.info("Backtest results: %d rows from %d files (survival_prob %.4f to %.4f)",
                len(combined), len(dfs),
                combined["survival_prob"].min(), combined["survival_prob"].max())
    return combined


# ── Analysis functions ──────────────────────────────────────────────────

def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict:
    """Compute accuracy, precision, recall, F1 for binary predictions."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def analyze_dataset(
    df: pd.DataFrame,
    outcome_col: str,
    dataset_name: str,
    threshold: float = THRESHOLD,
) -> list[dict]:
    """Analyze survival_prob as discriminator for a binary outcome column."""
    rows = []
    if df.empty or outcome_col not in df.columns:
        return rows

    above = df[df["survival_prob"] >= threshold]
    below = df[df["survival_prob"] < threshold]

    # Binary classification metrics
    y_true = df[outcome_col].astype(int).values
    y_pred = (df["survival_prob"] >= threshold).astype(int).values
    metrics = binary_classification_metrics(y_true, y_pred)

    # Chi-square test
    try:
        ct = pd.crosstab(df["survival_prob"] >= threshold, df[outcome_col])
        chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    except Exception:
        chi2, p_chi = float("nan"), float("nan")

    # Spearman correlation
    try:
        r_sp, p_sp = stats.spearmanr(df["survival_prob"], df[outcome_col])
    except Exception:
        r_sp, p_sp = float("nan"), float("nan")

    # Group rates
    above_rate = float(above[outcome_col].mean()) if len(above) > 0 else float("nan")
    below_rate = float(below[outcome_col].mean()) if len(below) > 0 else float("nan")

    rows.append({
        "dataset": dataset_name,
        "outcome": outcome_col,
        "threshold": threshold,
        "n_total": len(df),
        "n_above": len(above),
        "n_below": len(below),
        "rate_above": round(above_rate, 4),
        "rate_below": round(below_rate, 4),
        "rate_delta": round(above_rate - below_rate, 4),
        "chi2": round(chi2, 4) if np.isfinite(chi2) else None,
        "chi2_pvalue": round(p_chi, 6) if np.isfinite(p_chi) else None,
        "spearman_r": round(r_sp, 4) if np.isfinite(r_sp) else None,
        "spearman_p": round(p_sp, 6) if np.isfinite(p_sp) else None,
        **metrics,
    })
    return rows


def decile_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    dataset_name: str,
) -> list[dict]:
    """Compute outcome rate by survival_prob decile."""
    rows = []
    if df.empty or outcome_col not in df.columns:
        return rows

    # Use fixed bins to make results comparable
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = [
        "0.00-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
        "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00",
    ]
    df = df.copy()
    df["sp_bin"] = pd.cut(df["survival_prob"], bins=bins, labels=labels, right=False)

    for name, grp in df.groupby("sp_bin", observed=True):
        n = len(grp)
        rate = float(grp[outcome_col].mean()) if n > 0 else float("nan")
        rows.append({
            "dataset": dataset_name,
            "outcome": outcome_col,
            "sp_bin": str(name),
            "n": n,
            "outcome_rate": round(rate, 4),
            "type": "decile_analysis",
        })
    return rows


def threshold_scan(
    df: pd.DataFrame,
    outcome_col: str,
    dataset_name: str,
) -> list[dict]:
    """Scan multiple thresholds to find optimal discriminator."""
    rows = []
    if df.empty or outcome_col not in df.columns:
        return rows

    for thresh in np.arange(0.30, 0.85, 0.05):
        thresh = round(float(thresh), 2)
        above = df[df["survival_prob"] >= thresh]
        below = df[df["survival_prob"] < thresh]
        if len(above) < 5 or len(below) < 5:
            continue

        try:
            ct = pd.crosstab(df["survival_prob"] >= thresh, df[outcome_col])
            chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        except Exception:
            chi2, p_chi = float("nan"), float("nan")

        rows.append({
            "dataset": dataset_name,
            "outcome": outcome_col,
            "threshold": thresh,
            "n_above": len(above),
            "n_below": len(below),
            "rate_above": round(float(above[outcome_col].mean()), 4),
            "rate_below": round(float(below[outcome_col].mean()), 4),
            "chi2": round(chi2, 4) if np.isfinite(chi2) else None,
            "chi2_pvalue": round(p_chi, 6) if np.isfinite(p_chi) else None,
            "type": "threshold_scan",
        })
    return rows


# ── Main ────────────────────���────────────────��──────────────────────────

def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    logger.info("=" * 72)
    logger.info("MC Range-Containment Validation (Criterion 4)")
    logger.info("Threshold under test: survival_prob >= %.2f", THRESHOLD)
    logger.info("=" * 72)
    print()

    all_results: list[dict] = []

    # ── Dataset 1: Combined Training (full range) ─���────────────────────
    logger.info("--- Dataset 1: Combined Training (Feb 2026) ---")
    df_train = load_combined_training()
    if not df_train.empty:
        df_train["profitable"] = (df_train["pnl_pct"] > 0).astype(int)
        results_1 = analyze_dataset(df_train, "y", "combined_training_y")
        results_1b = analyze_dataset(df_train, "profitable", "combined_training_profitable")
        all_results.extend(results_1)
        all_results.extend(results_1b)

        for r in results_1 + results_1b:
            logger.info(
                "  [%s] >= %.2f: rate=%.4f (n=%d) | < %.2f: rate=%.4f (n=%d) | "
                "chi2=%.4f p=%.6f | spearman_r=%.4f",
                r["outcome"], THRESHOLD, r["rate_above"], r["n_above"],
                THRESHOLD, r["rate_below"], r["n_below"],
                r.get("chi2") or 0, r.get("chi2_pvalue") or 0,
                r.get("spearman_r") or 0,
            )

        all_results.extend(decile_analysis(df_train, "y", "combined_training_y"))
        all_results.extend(threshold_scan(df_train, "y", "combined_training_y"))
    print()

    # ─��� Dataset 2: Expired Bots (live outcomes) ───────��────────────────
    logger.info("--- Dataset 2: Expired Bots (Live, n=139) ---")
    df_bots = load_expired_bots()
    if not df_bots.empty:
        df_bots["profitable"] = (df_bots["pnl_pct"] > 0).astype(int)

        # Construct containment proxy:
        # MAE as fraction of grid range -- contained if MAE < grid range
        df_bots["grid_range_pct"] = (
            (df_bots["price_range_high"] - df_bots["price_range_low"])
            / ((df_bots["price_range_high"] + df_bots["price_range_low"]) / 2)
            * 100
        )
        df_bots["mae_range_ratio"] = (
            df_bots["mae_pct_initial"] / df_bots["grid_range_pct"]
        )
        df_bots["contained"] = (df_bots["mae_range_ratio"] < 1.0).astype(int)

        results_2a = analyze_dataset(df_bots, "profitable", "expired_bots_profitable")
        results_2b = analyze_dataset(df_bots, "contained", "expired_bots_contained")
        all_results.extend(results_2a)
        all_results.extend(results_2b)

        for r in results_2a + results_2b:
            logger.info(
                "  [%s] >= %.2f: rate=%.4f (n=%d) | < %.2f: rate=%.4f (n=%d) | "
                "chi2=%.4f p=%.6f | spearman_r=%.4f",
                r["outcome"], THRESHOLD, r["rate_above"], r["n_above"],
                THRESHOLD, r["rate_below"], r["n_below"],
                r.get("chi2") or 0, r.get("chi2_pvalue") or 0,
                r.get("spearman_r") or 0,
            )

        all_results.extend(decile_analysis(df_bots, "profitable", "expired_bots_profitable"))
        all_results.extend(decile_analysis(df_bots, "contained", "expired_bots_contained"))
        all_results.extend(threshold_scan(df_bots, "profitable", "expired_bots_profitable"))
    print()

    # ── Dataset 3: Backtest Results (time_in_range_pct) ────────────────
    logger.info("--- Dataset 3: Backtest Results (n~3800) ---")
    df_bt = load_all_backtest_results(
        backtest_dir,
        args.backtest_pattern,
        latest_only=args.latest_only,
    )
    if not df_bt.empty:
        # Containment: time_in_range_pct >= 50% (matches range_breakout threshold)
        df_bt["contained_50"] = (df_bt["time_in_range_pct"] >= 50.0).astype(int)
        df_bt["contained_80"] = (df_bt["time_in_range_pct"] >= 80.0).astype(int)

        # Price exit at end of backtest
        if "grid_lower" in df_bt.columns and "price_end" in df_bt.columns:
            df_bt["price_exited"] = (
                (df_bt["price_end"] < df_bt["grid_lower"])
                | (df_bt["price_end"] > df_bt["grid_upper"])
            ).astype(int)
            results_3c = analyze_dataset(df_bt, "price_exited", "backtest_price_exited")
            all_results.extend(results_3c)
            for r in results_3c:
                logger.info(
                    "  [%s] >= %.2f: rate=%.4f (n=%d) | < %.2f: rate=%.4f (n=%d) | "
                    "spearman_r=%.4f",
                    r["outcome"], THRESHOLD, r["rate_above"], r["n_above"],
                    THRESHOLD, r["rate_below"], r["n_below"],
                    r.get("spearman_r") or 0,
                )

        # Spearman: survival_prob vs continuous time_in_range_pct
        r_cont, p_cont = stats.spearmanr(df_bt["survival_prob"], df_bt["time_in_range_pct"])
        logger.info(
            "  Spearman (survival_prob vs time_in_range_pct): r=%.4f, p=%.6f",
            r_cont, p_cont,
        )
        all_results.append({
            "dataset": "backtest_continuous",
            "outcome": "time_in_range_pct",
            "threshold": THRESHOLD,
            "n_total": len(df_bt),
            "spearman_r": round(r_cont, 4),
            "spearman_p": round(p_cont, 6),
            "type": "continuous_correlation",
            "note": describe_backtest_survival_range(df_bt),
        })

        results_3a = analyze_dataset(df_bt, "contained_50", "backtest_contained_50")
        results_3b = analyze_dataset(df_bt, "contained_80", "backtest_contained_80")
        all_results.extend(results_3a)
        all_results.extend(results_3b)

        for r in results_3a + results_3b:
            logger.info(
                "  [%s] >= %.2f: rate=%.4f (n=%d) | < %.2f: rate=%.4f (n=%d) | "
                "spearman_r=%.4f",
                r["outcome"], THRESHOLD, r["rate_above"], r["n_above"],
                THRESHOLD, r["rate_below"], r["n_below"],
                r.get("spearman_r") or 0,
            )

        sp_min = float(df_bt["survival_prob"].min())
        sp_max = float(df_bt["survival_prob"].max())
        if sp_min >= 0.49:
            logger.info(
                "  WARNING: Backtest data survival_prob range [%.4f, %.4f] still "
                "looks truncated near 0.50. Limited below-threshold power remains.",
                sp_min, sp_max,
            )
        else:
            logger.info(
                "  INFO: Backtest data survival_prob range [%.4f, %.4f] extends "
                "below 0.50, so the truncation fix is visible in current data.",
                sp_min, sp_max,
            )
    print()

    # ── Summary Report ─────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("SUMMARY: MC Containment Validation at threshold=%.2f", THRESHOLD)
    logger.info("=" * 72)
    print()

    summary_rows = [r for r in all_results if "type" not in r]
    if summary_rows:
        for r in summary_rows:
            sig = "***" if (r.get("chi2_pvalue") or 1.0) < 0.001 else (
                "**" if (r.get("chi2_pvalue") or 1.0) < 0.01 else (
                    "*" if (r.get("chi2_pvalue") or 1.0) < 0.05 else "ns"
                )
            )
            logger.info(
                "  %-35s  rate_above=%.4f  rate_below=%.4f  delta=%.4f  chi2_p=%s  [%s]",
                r["dataset"],
                r.get("rate_above", 0),
                r.get("rate_below", 0),
                r.get("rate_delta", 0),
                f'{r.get("chi2_pvalue", "N/A"):.6f}' if r.get("chi2_pvalue") is not None else "N/A",
                sig,
            )
    print()

    # ── MC Methodology Assessment ──────────────────────────────────────
    logger.info("=" * 72)
    logger.info("MC METHODOLOGY ASSESSMENT")
    logger.info("=" * 72)
    horizon_bars, horizon_hours, mc_paths = current_stochastic_settings()
    backtest_range_note = describe_backtest_survival_range(df_bt) if 'df_bt' in locals() else "backtest survival_prob range unavailable"
    logger.info("")
    logger.info("1. CURRENT SIMULATION CONFIG: Euler-Maruyama OU process, %d paths,", mc_paths)
    logger.info(
        "   %d-bar horizon (%.1fh at 15min bars).",
        horizon_bars, horizon_hours,
    )
    logger.info("")
    logger.info("2. BACKTEST RANGE DIAGNOSTIC: %s.", backtest_range_note)
    logger.info("")
    logger.info("3. INTERPRETATION: Use the summary metrics above as the primary")
    logger.info("   evidence. Positive monotonic correlation and stronger >= %.2f", THRESHOLD)
    logger.info("   split performance indicate the containment model is aligning better.")
    logger.info("")
    logger.info("4. OU ASSUMPTION: The MC assumes Ornstein-Uhlenbeck dynamics")
    logger.info("   (mean-reverting). For trending micro-oscillators, this may")
    logger.info("   still understate drift risk even after the configuration fixes.")
    logger.info("")
    logger.info("5. MODEL ROLE: If correlation remains weak after these fixes,")
    logger.info("   treat survival_prob as a general quality predictor rather than")
    logger.info("   a literal containment probability.")

    # ── Save results ──────────────���────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "mc_containment_validation.csv"
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(out_path, index=False)
    logger.info("")
    logger.info("Results saved to: %s", out_path)
    logger.info("Total result rows: %d", len(df_out))


if __name__ == "__main__":
    main()
