"""Validation Criterion 2 — Threshold Calibration for micro_osc_score.

Primary calibration uses the joined scan + backtest population so the sweep
covers both admitted and rejected candidates across the widest available score
range. The expired-bots workbook is still loaded as a reference dataset when
present, but it is no longer the sole calibration source.

Outputs:
    - Console summary per dataset
    - results/micro_osc_threshold_calibration.csv
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

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BACKTEST_DIR = BASE_DIR / "data" / "backtest_candidates"
EXPIRED_PATH = BASE_DIR / "data" / "new_expired_bots.xlsx"
EXPIRED_PATH_FALLBACK = BASE_DIR / "data" / "new_expired_bots.xlsx"
OUTPUT_PATH = BASE_DIR / "results" / "micro_osc_threshold_calibration.csv"

DEFAULT_THRESHOLD = 0.45
SURVIVAL_THRESHOLD = 0.60
THRESHOLDS = np.arange(0.10, 0.91, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate micro_osc_score thresholds on scan/backtest outcomes",
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

    if hurst_exponent is not None and not pd.isna(hurst_exponent):
        hurst_norm = max(0.0, 1.0 - 2.0 * abs(float(hurst_exponent) - 0.5))
        return round(0.45 * vwap_norm + 0.35 * hurst_norm + 0.20 * ema_norm, 4)
    return round(0.60 * vwap_norm + 0.40 * ema_norm, 4)


def _first_non_null(row: pd.Series, *columns: str) -> float | str | None:
    """Return the first non-null value from the requested columns."""
    for col in columns:
        if col not in row.index:
            continue
        value = row[col]
        if pd.notna(value):
            return value
    return None


def confusion_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute confusion-matrix metrics at a given threshold."""
    predicted_positive = scores >= threshold
    tp = int(np.sum(predicted_positive & labels))
    fp = int(np.sum(predicted_positive & ~labels))
    fn = int(np.sum(~predicted_positive & labels))
    tn = int(np.sum(~predicted_positive & ~labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": round(float(threshold), 2),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "n_admitted": tp + fp,
        "n_rejected": tn + fn,
    }


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

    frames: list[pd.DataFrame] = []
    for file_path in bt_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            logger.warning("Skipping %s: %s", file_path.name, exc)
            continue
        if "candidate_id" not in df.columns:
            continue
        df = df.copy()
        df["_bt_source"] = file_path.name
        frames.append(df)

    if not frames:
        logger.error("No readable backtest result files with candidate_id were found.")
        sys.exit(1)

    bt = pd.concat(frames, ignore_index=True)
    if "backtest_timestamp" in bt.columns:
        bt = bt.sort_values("backtest_timestamp").drop_duplicates(
            subset=["candidate_id"], keep="last"
        )
    else:
        bt = bt.drop_duplicates(subset=["candidate_id"], keep="last")

    logger.info(
        "Loaded %d unique backtest rows from %d files",
        len(bt), len(frames),
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

    needed_cols = {
        "candidate_id",
        "symbol",
        "micro_osc_score",
        "micro_osc_bypass",
        "vwap_crosses_5m",
        "ema_crosses_5m",
        "hurst_exponent",
        "survival_prob",
    }

    frames: list[pd.DataFrame] = []
    for file_path in scan_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:
            logger.warning("Skipping %s: %s", file_path.name, exc)
            continue
        if "candidate_id" not in df.columns:
            continue
        available = [col for col in needed_cols if col in df.columns]
        if not available:
            continue
        frames.append(df[available].copy())

    if not frames:
        logger.error("No readable deployment_ready files with candidate_id were found.")
        sys.exit(1)

    scan = pd.concat(frames, ignore_index=True)
    scan = scan.drop_duplicates(subset=["candidate_id"], keep="last")
    logger.info(
        "Loaded %d unique scan rows from %d files",
        len(scan), len(frames),
    )
    return scan


def build_backtest_scan_population(
    *,
    backtest_dir: Path,
    backtest_pattern: str = "backtest_results_*.csv",
    scan_pattern: str = "deployment_ready_*.csv",
    latest_only: bool = False,
) -> pd.DataFrame:
    """Build the primary calibration population from scan + backtest joins."""
    bt = load_all_backtest_results(
        backtest_dir,
        backtest_pattern,
        latest_only=latest_only,
    )
    scan = load_all_scan_features(scan_pattern, latest_only=latest_only)
    merged = bt.merge(scan, on="candidate_id", how="inner", suffixes=("_bt", "_scan"))
    logger.info("Joined %d scan/backtest rows on candidate_id", len(merged))

    if merged.empty:
        logger.error("No candidate_id overlap between backtest results and scan data.")
        sys.exit(1)

    def _resolve_score(row: pd.Series) -> float | None:
        native = _first_non_null(row, "micro_osc_score_scan", "micro_osc_score_bt", "micro_osc_score")
        if native is not None:
            return float(native)
        hurst_value = _first_non_null(row, "hurst_exponent_scan", "hurst_exponent_bt", "hurst_exponent")
        return compute_micro_osc_score(
            row.get("vwap_crosses_5m_scan", row.get("vwap_crosses_5m", np.nan)),
            row.get("ema_crosses_5m_scan", row.get("ema_crosses_5m", np.nan)),
            None if hurst_value is None else float(hurst_value),
        )

    population = merged.copy()
    population["dataset"] = "backtest_scan_population"
    population["symbol"] = population.apply(
        lambda row: _first_non_null(row, "symbol_scan", "symbol_bt", "symbol"),
        axis=1,
    )
    population["micro_osc_score"] = population.apply(_resolve_score, axis=1)
    population["survival_prob"] = pd.to_numeric(
        population.apply(
            lambda row: _first_non_null(row, "survival_prob_bt", "survival_prob_scan", "survival_prob"),
            axis=1,
        ),
        errors="coerce",
    )
    population["net_pnl_pct"] = pd.to_numeric(population.get("net_pnl_pct"), errors="coerce")
    population["profitable"] = population["net_pnl_pct"] > 0

    cols = [
        "dataset",
        "candidate_id",
        "symbol",
        "micro_osc_score",
        "survival_prob",
        "net_pnl_pct",
        "profitable",
    ]
    return population[cols].copy()


def load_expired_bots_reference() -> pd.DataFrame:
    """Load expired bots as a reference-only calibration population."""
    data_path = EXPIRED_PATH if EXPIRED_PATH.exists() else EXPIRED_PATH_FALLBACK
    if not data_path.exists():
        logger.warning("Expired-bots workbook not found; skipping reference dataset.")
        return pd.DataFrame()

    df = pd.read_excel(data_path)
    required = {"symbol", "vwap_crosses_5m", "ema_crosses_5m", "hurst_exponent", "total_profit_usdt"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "Expired-bots workbook missing %s; skipping reference dataset.",
            sorted(missing),
        )
        return pd.DataFrame()

    ref = df.copy()
    ref["dataset"] = "expired_bots_reference"
    ref["candidate_id"] = ref.get("candidate_id")
    ref["micro_osc_score"] = ref.apply(
        lambda row: compute_micro_osc_score(
            row["vwap_crosses_5m"],
            row["ema_crosses_5m"],
            row["hurst_exponent"],
        ),
        axis=1,
    )
    ref["survival_prob"] = pd.to_numeric(ref.get("survival_prob"), errors="coerce")
    ref["net_pnl_pct"] = pd.to_numeric(ref.get("pnl_pct"), errors="coerce")
    ref["profitable"] = pd.to_numeric(ref["total_profit_usdt"], errors="coerce") > 0

    cols = [
        "dataset",
        "candidate_id",
        "symbol",
        "micro_osc_score",
        "survival_prob",
        "net_pnl_pct",
        "profitable",
    ]
    return ref[cols].copy()


def sweep_thresholds(dataset_name: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run the threshold sweep for a single dataset."""
    clean = df.dropna(subset=["micro_osc_score", "profitable"]).copy()
    clean["micro_osc_score"] = pd.to_numeric(clean["micro_osc_score"], errors="coerce")
    clean = clean[np.isfinite(clean["micro_osc_score"])]

    if clean.empty:
        raise ValueError(f"{dataset_name}: no finite micro_osc_score values available")

    scores = clean["micro_osc_score"].to_numpy(dtype=float)
    labels = clean["profitable"].to_numpy(dtype=bool)

    sweep_rows: list[dict[str, float | int | str]] = []
    for threshold in THRESHOLDS:
        row = confusion_at_threshold(scores, labels, float(threshold))
        row["dataset"] = dataset_name
        row["type"] = "sweep"
        sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)
    best = sweep_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    default_row = sweep_df.loc[np.isclose(sweep_df["threshold"], DEFAULT_THRESHOLD)].iloc[0]

    below_default = int(np.sum(scores < DEFAULT_THRESHOLD))
    with_survival = clean["survival_prob"].notna()
    dual_gate_mask = (
        with_survival
        & (clean["micro_osc_score"] >= DEFAULT_THRESHOLD)
        & (clean["survival_prob"] >= SURVIVAL_THRESHOLD)
    )

    summary: dict[str, float | int | str] = {
        "dataset": dataset_name,
        "type": "summary",
        "n_total": int(len(clean)),
        "n_profitable": int(labels.sum()),
        "n_unprofitable": int((~labels).sum()),
        "score_min": round(float(np.min(scores)), 4),
        "score_max": round(float(np.max(scores)), 4),
        "score_mean": round(float(np.mean(scores)), 4),
        "n_below_default": below_default,
        "optimal_threshold": round(float(best["threshold"]), 2),
        "optimal_f1": round(float(best["f1"]), 4),
        "optimal_precision": round(float(best["precision"]), 4),
        "optimal_recall": round(float(best["recall"]), 4),
        "default_threshold": DEFAULT_THRESHOLD,
        "default_f1": round(float(default_row["f1"]), 4),
        "default_precision": round(float(default_row["precision"]), 4),
        "default_recall": round(float(default_row["recall"]), 4),
        "dual_gate_count": int(dual_gate_mask.sum()),
        "dual_gate_profitable": int((dual_gate_mask & clean["profitable"]).sum()),
        "survival_non_null": int(with_survival.sum()),
    }
    return sweep_df, summary


def print_dataset_report(
    dataset_name: str,
    df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    summary: dict[str, float | int | str],
) -> None:
    """Print a compact console report for one dataset."""
    print()
    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)
    print(f"Rows: {summary['n_total']}  |  Profitable: {summary['n_profitable']}  |  Unprofitable: {summary['n_unprofitable']}")
    print(
        "micro_osc_score range: "
        f"{summary['score_min']:.4f} to {summary['score_max']:.4f} "
        f"(mean {summary['score_mean']:.4f})"
    )
    print(f"Below default threshold 0.45: {summary['n_below_default']}")
    print(
        "Dual gate (score>=0.45 AND survival_prob>=0.60): "
        f"{summary['dual_gate_count']} rows, "
        f"{summary['dual_gate_profitable']} profitable"
    )
    print()

    print(
        "Optimal threshold: "
        f"{summary['optimal_threshold']:.2f}  "
        f"F1={summary['optimal_f1']:.4f}  "
        f"precision={summary['optimal_precision']:.4f}  "
        f"recall={summary['optimal_recall']:.4f}"
    )
    print(
        "Default threshold: "
        f"{DEFAULT_THRESHOLD:.2f}  "
        f"F1={summary['default_f1']:.4f}  "
        f"precision={summary['default_precision']:.4f}  "
        f"recall={summary['default_recall']:.4f}"
    )
    print()

    display_thresholds = set(np.arange(0.10, 0.91, 0.05).round(2))
    display_thresholds.add(round(float(summary["optimal_threshold"]), 2))
    display_thresholds.add(DEFAULT_THRESHOLD)
    display_df = sweep_df[
        sweep_df["threshold"].round(2).isin(display_thresholds)
    ].sort_values("threshold")
    print("Selected sweep rows:")
    print(
        display_df[
            ["threshold", "TP", "FP", "TN", "FN", "precision", "recall", "f1", "n_admitted"]
        ].to_string(index=False)
    )

    recommendation_gap = abs(float(summary["optimal_threshold"]) - DEFAULT_THRESHOLD)
    if recommendation_gap > 0.05:
        print()
        print(
            "Recommendation: the optimal threshold differs from the default by "
            f"{recommendation_gap:.2f}; review MicroOscConfig.min_score."
        )


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    primary = build_backtest_scan_population(
        backtest_dir=backtest_dir,
        backtest_pattern=args.backtest_pattern,
        scan_pattern=args.scan_pattern,
        latest_only=args.latest_only,
    )
    datasets: list[pd.DataFrame] = [primary]

    reference = load_expired_bots_reference()
    if not reference.empty:
        datasets.append(reference)

    all_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []

    for dataset in datasets:
        dataset_name = str(dataset["dataset"].iloc[0])
        sweep_df, summary = sweep_thresholds(dataset_name, dataset)
        print_dataset_report(dataset_name, dataset, sweep_df, summary)
        all_rows.extend(sweep_df.to_dict("records"))
        all_rows.append(summary)
        summaries.append(summary)

    primary_summary = next(
        summary for summary in summaries if summary["dataset"] == "backtest_scan_population"
    )

    print()
    print("=" * 80)
    print("CRITERION 2 CHECKPOINT")
    print("=" * 80)
    print(
        "Primary calibration population size: "
        f"{primary_summary['n_total']} rows"
    )
    print(
        "Primary score range: "
        f"{primary_summary['score_min']:.4f} to {primary_summary['score_max']:.4f}"
    )
    print(
        "Primary optimal threshold: "
        f"{primary_summary['optimal_threshold']:.2f}"
    )
    print(
        "Primary precision / recall at optimum: "
        f"{primary_summary['optimal_precision']:.4f} / "
        f"{primary_summary['optimal_recall']:.4f}"
    )
    if int(primary_summary["n_total"]) < 100:
        print("WARNING: sample size is still below the 100-row target in the plan.")
    if float(primary_summary["score_min"]) >= DEFAULT_THRESHOLD:
        print("WARNING: the primary population still lacks below-threshold candidates.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
    logger.info("Saved threshold calibration results to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
