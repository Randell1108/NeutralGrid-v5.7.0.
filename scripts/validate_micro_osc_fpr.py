"""Validation Criterion 3 -- False Positive Rate for Micro-Oscillator Bypass Candidates.

Computes FPR by retroactively identifying bypass candidates (micro_osc_score >= 0.45
AND survival_prob >= 0.60) among historically backtested symbols, then measuring
how many were unprofitable.

Approach:
    1. Load ALL deployment_ready CSVs (scan-time features including vwap_crosses_5m,
       ema_crosses_5m, hurst_exponent, survival_prob, and candidate_id).
    2. Load ALL backtest_results CSVs (PnL outcomes and candidate_id).
    3. Join on candidate_id to pair scan-time features with backtest PnL.
    4. Retroactively compute micro_osc_score using the TRENDING_OSCILLATOR.md formula
       (for CSVs produced before the column was added natively).
    5. Filter to bypass candidates and compute FPR + distributional stats.

Output:
    - Console summary
    - results/micro_osc_fpr_analysis.csv

Usage:
    python scripts/validate_micro_osc_fpr.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BT_DIR = BASE_DIR / "data" / "backtest_candidates"
OUTPUT_PATH = BASE_DIR / "results" / "micro_osc_fpr_analysis.csv"

# Bypass thresholds (from MicroOscConfig in config.py)
MIN_SCORE = 0.45
MIN_SURVIVAL_PROB = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze bypass false-positive rate from scan/backtest joins",
    )
    parser.add_argument(
        "--backtest-dir",
        default=str(DEFAULT_BT_DIR),
        help="Directory containing backtest result CSVs",
    )
    parser.add_argument(
        "--backtest-pattern",
        default="backtest_results_*.csv",
        help="Glob for backtest result files inside --backtest-dir",
    )
    parser.add_argument(
        "--scan-pattern",
        default="deployment_ready_*.csv",
        help="Glob for scan files inside results/",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Use only the latest backtest file and latest scan file matching the globs",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# micro_osc_score formula (mirrors scan.py / TRENDING_OSCILLATOR.md Step 1)
# ---------------------------------------------------------------------------
def compute_micro_osc_score(
    vwap_crosses_5m: float,
    ema_crosses_5m: float,
    hurst_exponent: float | None,
    *,
    k_vwap: float = 25.0,
    k_ema: float = 20.0,
) -> float | None:
    """Compute the micro-oscillator score from raw components."""
    if pd.isna(vwap_crosses_5m) or pd.isna(ema_crosses_5m):
        return None
    vwap_norm = vwap_crosses_5m / (vwap_crosses_5m + k_vwap)
    ema_norm = ema_crosses_5m / (ema_crosses_5m + k_ema)

    if hurst_exponent is not None and not np.isnan(hurst_exponent):
        hurst_norm = max(0.0, 1.0 - 2.0 * abs(hurst_exponent - 0.5))
        return round(0.45 * vwap_norm + 0.35 * hurst_norm + 0.20 * ema_norm, 4)
    else:
        return round(0.60 * vwap_norm + 0.40 * ema_norm, 4)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_backtest_results(
    backtest_dir: Path,
    filename_pattern: str = "backtest_results_*.csv",
    *,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Load and deduplicate all backtest_results CSVs."""
    bt_files = sorted(backtest_dir.glob(filename_pattern))
    if not bt_files:
        logger.error(
            "No backtest result files matching %s found in %s",
            filename_pattern,
            backtest_dir,
        )
        sys.exit(1)
    if latest_only:
        bt_files = bt_files[-1:]

    frames = []
    for f in bt_files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
            continue
        df = df.copy()
        df["_bt_source"] = f.name
        frames.append(df)

    bt = pd.concat(frames, ignore_index=True)
    # Dedup: keep latest backtest per candidate_id
    if "backtest_timestamp" in bt.columns:
        bt = bt.sort_values("backtest_timestamp").drop_duplicates(
            subset=["candidate_id"], keep="last"
        )
    else:
        bt = bt.drop_duplicates(subset=["candidate_id"], keep="last")

    logger.info(
        "Loaded %d unique backtest results from %d files",
        len(bt), len(bt_files),
    )
    return bt


def load_all_scan_features(
    filename_pattern: str = "deployment_ready_*.csv",
    *,
    latest_only: bool = False,
) -> pd.DataFrame:
    """Load scan-time features from deployment_ready CSVs."""
    scan_files = sorted(RESULTS_DIR.glob(filename_pattern))
    if not scan_files:
        logger.error("No scan files matching %s found in %s", filename_pattern, RESULTS_DIR)
        sys.exit(1)
    if latest_only:
        scan_files = scan_files[-1:]

    # Columns we need from scan data
    needed_cols = {
        "candidate_id", "symbol",
        "vwap_crosses_5m", "ema_crosses_5m", "hurst_exponent",
        "survival_prob", "micro_osc_score", "micro_osc_bypass",
    }

    frames = []
    for f in scan_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if "candidate_id" not in df.columns:
                continue
            available = needed_cols & set(df.columns)
            frames.append(df[list(available)])
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)

    scan = pd.concat(frames, ignore_index=True)
    scan = scan.drop_duplicates(subset=["candidate_id"], keep="last")
    logger.info(
        "Loaded %d unique scan candidates from %d files",
        len(scan), len(scan_files),
    )
    return scan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    bt = load_all_backtest_results(
        backtest_dir,
        args.backtest_pattern,
        latest_only=args.latest_only,
    )
    scan = load_all_scan_features(args.scan_pattern, latest_only=args.latest_only)

    # ------------------------------------------------------------------
    # 2. Join on candidate_id
    # ------------------------------------------------------------------
    merged = bt.merge(scan, on="candidate_id", how="inner", suffixes=("_bt", "_scan"))
    merged = merged.copy()
    logger.info("Joined %d rows (backtest + scan features)", len(merged))

    if len(merged) == 0:
        logger.error("No candidate_id overlap between backtest results and scan data.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Retroactively compute micro_osc_score where missing
    # ------------------------------------------------------------------
    # Prefer native micro_osc_score from scan; compute from components otherwise
    hurst_col = (
        "hurst_exponent_scan"
        if "hurst_exponent_scan" in merged.columns
        else "hurst_exponent"
    )

    def _resolve_mos(row: pd.Series) -> float | None:
        native = row.get("micro_osc_score")
        if pd.notna(native):
            return float(native)
        return compute_micro_osc_score(
            row.get("vwap_crosses_5m", np.nan),
            row.get("ema_crosses_5m", np.nan),
            row.get(hurst_col, None),
        )

    merged["micro_osc_score_resolved"] = merged.apply(_resolve_mos, axis=1)

    mos_nn = merged["micro_osc_score_resolved"].notna().sum()
    logger.info("micro_osc_score resolved: %d / %d non-null", mos_nn, len(merged))

    # Use survival_prob from backtest results (always populated for backtested rows)
    surv_col = (
        "survival_prob_bt"
        if "survival_prob_bt" in merged.columns
        else "survival_prob"
    )
    sp_nn = merged[surv_col].notna().sum()
    logger.info("survival_prob available: %d / %d non-null", sp_nn, len(merged))

    # ------------------------------------------------------------------
    # 4. Identify bypass candidates
    # ------------------------------------------------------------------
    valid_mask = merged["micro_osc_score_resolved"].notna() & merged[surv_col].notna()
    bypass_mask = (
        valid_mask
        & (merged["micro_osc_score_resolved"] >= MIN_SCORE)
        & (merged[surv_col] >= MIN_SURVIVAL_PROB)
    )
    non_bypass_mask = valid_mask & ~bypass_mask

    bypass_flag_col = None
    for candidate in ("micro_osc_bypass_scan", "micro_osc_bypass_bt", "micro_osc_bypass"):
        if candidate in merged.columns:
            bypass_flag_col = candidate
            break
    if bypass_flag_col is not None:
        actual_bypass_mask = bypass_mask & merged[bypass_flag_col].fillna(False).astype(bool)
    else:
        actual_bypass_mask = pd.Series(False, index=merged.index)
    proxy_bypass_mask = bypass_mask & ~actual_bypass_mask

    n_valid = int(valid_mask.sum())
    n_bypass = int(bypass_mask.sum())
    n_non_bypass = int(non_bypass_mask.sum())
    n_actual_bypass = int(actual_bypass_mask.sum())
    n_proxy_bypass = int(proxy_bypass_mask.sum())

    logger.info(
        "Valid for analysis: %d  |  Bypass candidates: %d  |  Non-bypass: %d",
        n_valid, n_bypass, n_non_bypass,
    )

    if n_bypass == 0:
        logger.error(
            "No bypass candidates found. Cannot compute FPR. "
            "This may indicate micro_osc_score or survival_prob data is insufficient."
        )
        sys.exit(1)

    bypass_df = merged.loc[bypass_mask].copy()
    non_bypass_df = merged.loc[non_bypass_mask].copy()

    # ------------------------------------------------------------------
    # 5. Compute FPR and PnL statistics
    # ------------------------------------------------------------------
    pnl_col = "net_pnl_pct"

    # --- Bypass candidates ---
    bp_profitable = bypass_df[pnl_col] > 0
    bp_unprofitable = ~bp_profitable
    bp_fpr = float(bp_unprofitable.sum()) / n_bypass

    bp_mean = float(bypass_df[pnl_col].mean())
    bp_median = float(bypass_df[pnl_col].median())
    bp_std = float(bypass_df[pnl_col].std())
    bp_min = float(bypass_df[pnl_col].min())
    bp_max = float(bypass_df[pnl_col].max())
    bp_q05 = float(bypass_df[pnl_col].quantile(0.05))
    bp_q25 = float(bypass_df[pnl_col].quantile(0.25))
    bp_q75 = float(bypass_df[pnl_col].quantile(0.75))
    bp_q95 = float(bypass_df[pnl_col].quantile(0.95))

    # --- Non-bypass candidates (for comparison) ---
    if n_non_bypass > 0:
        nb_profitable = non_bypass_df[pnl_col] > 0
        nb_fpr = float((~nb_profitable).sum()) / n_non_bypass
        nb_mean = float(non_bypass_df[pnl_col].mean())
        nb_median = float(non_bypass_df[pnl_col].median())
    else:
        nb_fpr = float("nan")
        nb_mean = float("nan")
        nb_median = float("nan")

    # --- Overall pipeline (all backtested, for reference) ---
    overall_n = len(bt)
    overall_profitable = int((bt[pnl_col] > 0).sum())
    overall_fpr = (overall_n - overall_profitable) / overall_n if overall_n > 0 else float("nan")

    # --- Hurdle-based analysis (meta_hurdle_pct ~ 3.0%) ---
    hurdle_pct = 3.0
    bp_above_hurdle = int((bypass_df[pnl_col] >= hurdle_pct).sum())
    bp_hurdle_rate = bp_above_hurdle / n_bypass

    # ------------------------------------------------------------------
    # 6. Print report
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("VALIDATION CRITERION 3: FALSE POSITIVE RATE FOR MICRO-OSC BYPASS CANDIDATES")
    print("=" * 80)
    print()
    print("Definition: A bypass candidate has micro_osc_score >= 0.45 AND survival_prob >= 0.60.")
    print("Definition: A false positive is a bypass candidate with net_pnl_pct <= 0.")
    print()
    backtest_file_count = len(list(backtest_dir.glob(args.backtest_pattern)))
    scan_file_count = len(list(RESULTS_DIR.glob(args.scan_pattern)))
    print(f"Data sources:")
    print(
        f"  Backtest results:     {len(bt)} unique candidates across {backtest_file_count} files"
    )
    print(
        f"  Scan features:        {len(scan)} unique candidates across {scan_file_count} files"
    )
    print(f"  Joined:               {len(merged)} candidates with both scan features and backtest outcomes")
    print(f"  Valid for analysis:   {n_valid} (micro_osc_score AND survival_prob both non-null)")
    print()

    print("-" * 80)
    print("BYPASS CANDIDATES (micro_osc_score >= 0.45 AND survival_prob >= 0.60)")
    print("-" * 80)
    print(f"  Count:                {n_bypass}")
    print(f"  Actual bypass rows:   {n_actual_bypass}")
    print(f"  Proxy-only rows:      {n_proxy_bypass}")
    print(f"  Profitable (PnL>0):   {int(bp_profitable.sum())} / {n_bypass} ({bp_profitable.sum()/n_bypass*100:.1f}%)")
    print(f"  Unprofitable (PnL<=0):{int(bp_unprofitable.sum())} / {n_bypass} ({bp_unprofitable.sum()/n_bypass*100:.1f}%)")
    print(f"  FALSE POSITIVE RATE:  {int(bp_unprofitable.sum())}/{n_bypass} = {bp_fpr*100:.1f}%")
    print()
    print(f"  PnL Distribution:")
    print(f"    Mean:     {bp_mean:+.3f}%")
    print(f"    Median:   {bp_median:+.3f}%")
    print(f"    Std Dev:  {bp_std:.3f}%")
    print(f"    Min:      {bp_min:+.3f}%")
    print(f"    Max:      {bp_max:+.3f}%")
    print(f"    P5:       {bp_q05:+.3f}%")
    print(f"    P25:      {bp_q25:+.3f}%")
    print(f"    P75:      {bp_q75:+.3f}%")
    print(f"    P95:      {bp_q95:+.3f}%")
    print()
    print(f"  Above hurdle ({hurdle_pct}%): {bp_above_hurdle}/{n_bypass} = {bp_hurdle_rate*100:.1f}%")
    print()

    print("-" * 80)
    print("COMPARISON: NON-BYPASS CANDIDATES")
    print("-" * 80)
    if n_non_bypass > 0:
        print(f"  Count:              {n_non_bypass}")
        print(f"  FPR:                {(~nb_profitable).sum()}/{n_non_bypass} = {nb_fpr*100:.1f}%")
        print(f"  Mean PnL:           {nb_mean:+.3f}%")
        print(f"  Median PnL:         {nb_median:+.3f}%")
    else:
        print("  (No non-bypass candidates in joined dataset)")
    print()

    print("-" * 80)
    print("COMPARISON: OVERALL PIPELINE (ALL BACKTESTED CANDIDATES)")
    print("-" * 80)
    print(f"  Count:              {overall_n}")
    print(f"  Profitable:         {overall_profitable}/{overall_n} = {overall_profitable/overall_n*100:.1f}%")
    print(f"  FPR:                {overall_n - overall_profitable}/{overall_n} = {overall_fpr*100:.1f}%")
    print(f"  Mean PnL:           {float(bt[pnl_col].mean()):+.3f}%")
    print(f"  Median PnL:         {float(bt[pnl_col].median()):+.3f}%")
    print()

    # ------------------------------------------------------------------
    # 7. Interpretation
    # ------------------------------------------------------------------
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()

    fpr_diff = bp_fpr - overall_fpr
    if abs(fpr_diff) < 0.03:
        fpr_verdict = (
            f"Bypass FPR ({bp_fpr*100:.1f}%) is comparable to overall pipeline FPR "
            f"({overall_fpr*100:.1f}%). Difference: {fpr_diff*100:+.1f}pp."
        )
    elif fpr_diff > 0:
        fpr_verdict = (
            f"Bypass FPR ({bp_fpr*100:.1f}%) is HIGHER than overall pipeline FPR "
            f"({overall_fpr*100:.1f}%) by {fpr_diff*100:+.1f}pp. "
            f"Bypass candidates carry slightly elevated false positive risk."
        )
    else:
        fpr_verdict = (
            f"Bypass FPR ({bp_fpr*100:.1f}%) is LOWER than overall pipeline FPR "
            f"({overall_fpr*100:.1f}%). Bypass candidates are at least as selective."
        )
    print(f"  {fpr_verdict}")
    print()

    mean_diff = bp_mean - float(bt[pnl_col].mean())
    if abs(mean_diff) < 0.5:
        pnl_verdict = (
            f"Bypass mean PnL ({bp_mean:+.3f}%) is comparable to overall "
            f"({float(bt[pnl_col].mean()):+.3f}%). Difference: {mean_diff:+.3f}pp."
        )
    elif mean_diff > 0:
        pnl_verdict = (
            f"Bypass mean PnL ({bp_mean:+.3f}%) exceeds overall "
            f"({float(bt[pnl_col].mean()):+.3f}%) by {mean_diff:+.3f}pp."
        )
    else:
        pnl_verdict = (
            f"Bypass mean PnL ({bp_mean:+.3f}%) is below overall "
            f"({float(bt[pnl_col].mean()):+.3f}%) by {mean_diff:+.3f}pp."
        )
    print(f"  {pnl_verdict}")
    print()

    # ------------------------------------------------------------------
    # 8. Methodological caveats
    # ------------------------------------------------------------------
    print("=" * 80)
    print("METHODOLOGICAL CAVEATS")
    print("=" * 80)
    print()
    print("  1. RETROACTIVE COMPUTATION: micro_osc_score was computed retroactively for")
    print("     most candidates using scan-time vwap_crosses_5m, ema_crosses_5m, and")
    print("     hurst_exponent joined from deployment_ready CSVs. Only the most recent")
    print("     scans (2026-03-29) have native micro_osc_score values.")
    print()
    if n_actual_bypass > 0 and n_proxy_bypass > 0:
        print("  2. COHORT MIXING: This dataset contains BOTH actual bypass rows")
        print("     (micro_osc_bypass=True in scan output) and older proxy rows")
        print("     identified retroactively from historical standard-pipeline scans.")
    elif n_actual_bypass > 0:
        print("  2. ACTUAL BYPASS COHORT: This dataset includes genuine bypass rows")
        print("     marked micro_osc_bypass=True in scan output, so the analysis is")
        print("     no longer purely retroactive.")
    else:
        print("  2. SELECTION BIAS: Backtested candidates already passed grid_is_valid=True")
        print("     via the standard pipeline. The bypass feature is designed for candidates")
        print("     that FAIL standard gates but pass micro_osc criteria. This run still")
        print("     reflects only retroactive proxy analysis, not genuine bypass-only rows.")
    print()
    print("  3. SURVIVAL_PROB SOURCE: We use survival_prob from backtest results (always")
    print("     populated). This is the scan-time value propagated through the pipeline,")
    print("     not a re-computed value.")
    print()
    print("  4. HURST AVAILABILITY: Not all scan candidates had hurst_exponent computed")
    print("     (requires 300+ 15m bars). For those without hurst, micro_osc_score is")
    print("     computed using the 2-component formula (vwap + ema only), which may")
    print("     produce different score distributions.")
    print()
    print("  5. TEMPORAL SCOPE: Backtest results span multiple dates (Feb-Mar 2026).")
    print("     Market regime shifts within this period are not controlled for.")
    print()

    # ------------------------------------------------------------------
    # 9. Save detailed results
    # ------------------------------------------------------------------
    sym_col = "symbol_bt" if "symbol_bt" in bypass_df.columns else "symbol"
    output_cols = [
        sym_col, "candidate_id", "micro_osc_score_resolved", surv_col,
        pnl_col, "_bt_source",
    ]
    # Add y if available
    if "y" in bypass_df.columns:
        output_cols.append("y")

    save_df = bypass_df[output_cols].copy()
    save_df = save_df.rename(columns={
        sym_col: "symbol",
        "micro_osc_score_resolved": "micro_osc_score",
        surv_col: "survival_prob",
    })
    save_df["actual_bypass_row"] = False
    if bypass_flag_col is not None and bypass_flag_col in bypass_df.columns:
        save_df["actual_bypass_row"] = bypass_df[bypass_flag_col].fillna(False).astype(bool).values
    save_df["is_profitable"] = save_df[pnl_col] > 0
    save_df["is_false_positive"] = ~save_df["is_profitable"]

    # Add summary row
    summary = pd.DataFrame([{
        "symbol": "---SUMMARY---",
        "candidate_id": f"n={n_bypass}",
        "micro_osc_score": float(bypass_df["micro_osc_score_resolved"].mean()),
        "survival_prob": float(bypass_df[surv_col].mean()),
        pnl_col: bp_mean,
        "_bt_source": f"FPR={bp_fpr:.4f};actual={n_actual_bypass};proxy={n_proxy_bypass}",
        "actual_bypass_row": None,
        "is_profitable": None,
        "is_false_positive": None,
    }])
    if "y" in save_df.columns:
        summary["y"] = None
    save_df = pd.concat([save_df, summary], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved FPR analysis to %s (%d rows)", OUTPUT_PATH, len(save_df))


if __name__ == "__main__":
    main()
