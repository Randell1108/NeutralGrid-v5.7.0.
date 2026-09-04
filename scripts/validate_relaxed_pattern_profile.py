"""Validate that known fast winners still rank high under a relaxed pattern profile.

Context
-------
The scanner pattern profile was retrained with `top_quantile=0.0` (PnL quantile
gate removed) so the winner centroid could be estimated from more rows while
the training pool grows. The safety question: does the relaxed profile still
score the original strict-policy winners near the top of the labeled universe?
If not, the relaxation has diluted the centroid past its discriminative value.

Inputs
------
  --reference-profile   JSON with selection_summary.winners_symbols listing the
                        canonical winners under the strict policy. Typically
                        `data/profile/_archive/pattern_profile_pre_relaxation_*.json`.
  --relaxed-profile     JSON for the current (relaxed) profile. Typically
                        `data/profile/pattern_profile.json`.
  --xlsx                The labeled-bot xlsx used to train both profiles.
  --top-n-threshold     Top-N size for the overlap metric (default: 20).
  --min-rank-percentile Median winner rank must fall within this top quantile
                        of the labeled universe (default: 0.75 — i.e. the
                        reference winners' median rank must be in the top 25%).

Outputs
-------
  - Per-reference-winner diagnostic: symbol, rank (1-based), percentile.
  - Top-N overlap: fraction of reference winners present in top-N relaxed.
  - Distribution summary: min / median / max similarity for reference winners
    vs. non-reference rows.

Exit codes
----------
  0  Reference winners' median rank sits inside the top `min-rank-percentile`
     quantile. Relaxation is discriminatively valid.
  1  Median rank fell out of the top quantile. Full per-symbol diagnostic is
     printed; investigate before keeping the relaxed profile in production.

Weights for `similarity_score` mirror the hard-coded weights at
`src/neutralgrid/scanner/scan.py:647-656`. If scan.py changes, update here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Allow running from repo root without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neutralgrid.scanner._xlsx_io import (  # noqa: E402
    MULTI_SHEET_NAMES,
    detect_format,
    raise_on_duplicate_strategy_id,
    read_sheet,
    read_single_sheet,
    validate_dataframe,
)
from neutralgrid.scanner.pattern_profile import PatternProfile  # noqa: E402


logger = logging.getLogger("validate_relaxed_pattern_profile")


# Mirrors scan.py:647-656 exactly. Keep in sync manually — if the production
# scorer changes weights, the validation must use the new weights to stay
# faithful to the deployed ranking logic.
_SCAN_WEIGHTS: dict[str, float] = {
    "adx_1h": 1.2,
    "adx_15m": 1.0,
    "adx_5m": 0.7,
    "ema_slope_1h": 1.2,
    "bb_width": 1.0,
    "range_size_pct": 1.0,
    "ema_crosses_5m": 0.6,
    "vwap_crosses_5m": 0.6,
}


def _load_labeled_universe(xlsx_path: Path) -> pd.DataFrame:
    """Load every labeled row — same readers as profile training."""
    fmt = detect_format(xlsx_path)
    if fmt == "multi":
        frames = [read_sheet(xlsx_path, s) for s in MULTI_SHEET_NAMES]
        merged = frames[0]
        for df_next in frames[1:]:
            merged = merged.merge(df_next, on="strategy_id", how="outer", suffixes=("", "_dup"))
        df_val, _ = validate_dataframe(merged, "merged multi-sheet")
        raise_on_duplicate_strategy_id(df_val, context="merged multi-sheet")
        return df_val
    df, name = read_single_sheet(xlsx_path)
    df_val, _ = validate_dataframe(df, name)
    raise_on_duplicate_strategy_id(df_val, context=name)
    return df_val


def _profile_from_path(path: Path) -> PatternProfile:
    profile = PatternProfile.load_json(path)
    if profile is None:
        raise RuntimeError(
            f"Failed to load pattern profile from {path} — load_json returned None "
            "(critical features missing)."
        )
    return profile


def _winners_from_reference(profile: PatternProfile) -> list[str]:
    summary = profile.selection_summary or {}
    raw = summary.get("winners_symbols") or []
    winners = sorted({str(s) for s in raw})
    if not winners:
        raise RuntimeError(
            "Reference profile has no selection_summary.winners_symbols — cannot "
            "establish the canonical winner set. Use a pre-relaxation artifact."
        )
    return winners


def _score_universe(
    df: pd.DataFrame,
    profile: PatternProfile,
) -> pd.DataFrame:
    """Compute similarity_score for every row, descending rank."""
    scores: list[float] = []
    for _, row in df.iterrows():
        scores.append(float(profile.similarity_score(row.to_dict(), weights=_SCAN_WEIGHTS)))
    out = df.copy()
    out["similarity_score"] = scores
    out = out.sort_values("similarity_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)
    out["percentile"] = 1.0 - (out["rank"] - 1) / max(len(out) - 1, 1)
    return out


def _summarize(
    ranked: pd.DataFrame,
    reference_winners: list[str],
    top_n: int,
) -> dict[str, object]:
    symbol_col = "symbol" if "symbol" in ranked.columns else None
    if symbol_col is None:
        raise RuntimeError("Labeled xlsx has no 'symbol' column — cannot locate reference winners.")

    syms = cast(pd.Series, ranked[symbol_col]).astype(str)
    ref_mask = syms.isin(reference_winners)
    ref_rows = ranked.loc[ref_mask].copy()
    non_ref_rows = ranked.loc[~ref_mask].copy()

    ref_ranks = cast(pd.Series, ref_rows["rank"]).astype(float).tolist() if not ref_rows.empty else []
    median_rank = float(np.median(ref_ranks)) if ref_ranks else float("nan")
    median_pct = 1.0 - (median_rank - 1) / max(len(ranked) - 1, 1) if ref_ranks else float("nan")

    top_n_syms = set(cast(pd.Series, ranked.head(top_n)[symbol_col]).astype(str).tolist())
    overlap = sorted(set(reference_winners) & top_n_syms)
    missing = sorted(set(reference_winners) - set(syms.tolist()))

    def _stats(series: pd.Series) -> dict[str, float]:
        if series.empty:
            return {"min": float("nan"), "median": float("nan"), "max": float("nan")}
        vals = series.astype(float).to_numpy()
        return {
            "min": float(np.min(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "universe_size": int(len(ranked)),
        "reference_winners": reference_winners,
        "reference_found": int(len(ref_rows)),
        "reference_missing_from_xlsx": missing,
        "per_winner": [
            {
                "symbol": str(r[symbol_col]),
                "rank": int(r["rank"]),
                "percentile": float(r["percentile"]),
                "similarity": float(r["similarity_score"]),
            }
            for _, r in ref_rows.sort_values("rank").iterrows()
        ],
        "median_rank": median_rank,
        "median_percentile": median_pct,
        "top_n_threshold": int(top_n),
        "top_n_overlap_count": len(overlap),
        "top_n_overlap_fraction": (
            len(overlap) / len(reference_winners) if reference_winners else 0.0
        ),
        "top_n_overlap_symbols": overlap,
        "similarity_stats_reference": _stats(cast(pd.Series, ref_rows["similarity_score"])),
        "similarity_stats_non_reference": _stats(cast(pd.Series, non_ref_rows["similarity_score"])),
    }


def _print_report(report: dict[str, object]) -> None:
    print("=" * 72)
    print("Relaxed pattern-profile validation — reference-winner ranking")
    print("=" * 72)
    print(f"Labeled universe size      : {report['universe_size']}")
    print(f"Reference winners (strict) : {len(cast(list, report['reference_winners']))}")
    print(f"Found in labeled xlsx      : {report['reference_found']}")
    missing = cast(list, report["reference_missing_from_xlsx"])
    if missing:
        print(f"Missing from xlsx          : {missing}")
    print()
    print("Per-winner ranks under relaxed profile:")
    for entry in cast(list, report["per_winner"]):
        e = cast(dict, entry)
        print(
            f"  {e['symbol']:<14} rank={e['rank']:>4}  "
            f"percentile={e['percentile']:.3f}  sim={e['similarity']:.2f}"
        )
    print()
    print(f"Median winner rank         : {report['median_rank']}")
    print(f"Median winner percentile   : {report['median_percentile']}")
    print(f"Top-{report['top_n_threshold']} overlap                : "
          f"{report['top_n_overlap_count']} / {len(cast(list, report['reference_winners']))} "
          f"({report['top_n_overlap_fraction']:.2%})")
    print(f"Top-N overlap symbols      : {report['top_n_overlap_symbols']}")
    print()
    print("Similarity distribution:")
    ref = cast(dict, report["similarity_stats_reference"])
    non = cast(dict, report["similarity_stats_non_reference"])
    print(f"  reference winners  min={ref['min']:.2f}  med={ref['median']:.2f}  max={ref['max']:.2f}")
    print(f"  non-reference rows min={non['min']:.2f}  med={non['median']:.2f}  max={non['max']:.2f}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--xlsx", default="data/new_expired_bots.xlsx", type=Path)
    ap.add_argument(
        "--reference-profile",
        default="data/profile/_archive/pattern_profile_pre_relaxation_20260420.json",
        type=Path,
    )
    ap.add_argument(
        "--relaxed-profile",
        default="data/profile/pattern_profile.json",
        type=Path,
    )
    ap.add_argument("--top-n-threshold", type=int, default=20)
    ap.add_argument(
        "--min-rank-percentile",
        type=float,
        default=0.75,
        help="Median reference-winner percentile must be >= this value (0.75 = top quartile).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for label, path in (
        ("xlsx", args.xlsx),
        ("reference-profile", args.reference_profile),
        ("relaxed-profile", args.relaxed_profile),
    ):
        if not Path(path).exists():
            logger.error("Missing %s input: %s", label, path)
            return 2

    reference = _profile_from_path(args.reference_profile)
    relaxed = _profile_from_path(args.relaxed_profile)
    winners = _winners_from_reference(reference)
    logger.info("Loaded %d reference winners from %s", len(winners), args.reference_profile.name)

    df = _load_labeled_universe(args.xlsx)
    ranked = _score_universe(df, relaxed)
    report = _summarize(ranked, winners, args.top_n_threshold)
    _print_report(report)

    median_pct = float(cast(float, report["median_percentile"]))
    if not np.isfinite(median_pct):
        logger.error("Median percentile is NaN — no reference winners found in xlsx.")
        return 1

    if median_pct + 1e-9 < args.min_rank_percentile:
        logger.error(
            "FAIL: median reference-winner percentile %.3f < required %.3f. "
            "Relaxed profile has demoted the canonical winners; investigate before deploying.",
            median_pct, args.min_rank_percentile,
        )
        return 1

    logger.info(
        "PASS: median reference-winner percentile %.3f >= required %.3f.",
        median_pct, args.min_rank_percentile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
