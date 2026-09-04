#!/usr/bin/env python3
"""Reproducible feature study for the fast-winner meta-labeler (AFML precision filter).

Usage::

    python scripts/meta_labeler_feature_study.py [--data-dir data/backtest_candidates]
        [--pool geometric|all|both] [--pnl-hurdle 3.0] [--max-duration-hours 7.0]
        [--out data/meta_labeler_feature_study.json]

What this script answers
------------------------
"Which ex-ante features make the meta-labeler healthy, accurate and helpful?" It
builds the fast-winner label ``y = (eff_pnl >= hurdle) & (duration <= H)`` on the
labeled backtest pool and measures, leakage-free:

* univariate ROC-AUC + mutual information + bootstrap 95% CI per feature, and
* out-of-fold ROC-AUC for candidate models under symbol-grouped, time-purged CV.

Why a strict allowlist (load-bearing safety detail)
---------------------------------------------------
The backtest training CSVs carry *post-hoc outcome* columns
(``hlabel_detail_*``, ``realized_net_pnl``, ``mfe*``, ``pnl_curve_*`` ...) that are
NOT in ``_KNOWN_LABEL_COLUMNS`` but ARE label leakage (one of them equals the PnL,
univariate AUC == 1.0). A naive "all columns minus ``_KNOWN_LABEL_COLUMNS``"
feature set therefore leaks. This study uses a curated EX-ANTE allowlist plus an
outcome-column regex, and asserts that no surviving feature has univariate
AUC > ``--leak-auc`` (default 0.90). This is the same discipline the snapshot-only
training architecture enforces, reproduced here so the result is auditable.

The numbers this reproduces (geometric pool, leakage-screened, as of 2026-05-29):
calibrated logistic OOF AUC ~0.70 [0.635, 0.765]; broader pool ~0.82 [0.78, 0.857].
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# ── Curated EX-ANTE feature allowlist (decision-time only) ──────────────────
# Grouped to match the plan's formula. NEVER replace this with
# "all columns minus _KNOWN_LABEL_COLUMNS" — the CSVs carry outcome columns.
EXANTE_ALLOWLIST: list[str] = [
    # primary signal. NOTE: primary_pipeline_score is DELIBERATELY EXCLUDED (see
    # CIRCULAR_EXCLUDE) because the scanner ranking score can blend meta_prob,
    # which would make it a circular meta-feature. ev_score (expected-value edge)
    # is an independent signal and is retained.
    "ev_score",
    # regime / volatility / mean-reversion
    "adx_1h", "adx_15m", "adx_5m", "atr_pct_15m", "ou_halflife", "rsi_15m",
    "ema_slope_1h", "ema_crosses_5m", "vwap_crosses_5m", "hurst_exponent",
    "range_size_pct", "scan_range_size_pct", "bb_width", "bb_width_ratio_1h_15m",
    # grid economics
    "grid_spacing_pct", "profit_per_grid_pct", "num_grids",
    # liquidity / context
    "quote_volume_24h", "open_interest", "micro_round_trip_cost_pct", "funding_rate",
    # microstructure / flow (usually sparse in the backtest pool; kept for when backfilled)
    "funding_rate_zscore", "open_interest_change_pct", "top_account_lsr_log",
    "top_position_lsr_log", "global_lsr_log", "taker_imbalance", "taker_buy_sell_log",
    "basis_pct", "spread_pct", "top20_bid_depth_usdt", "top20_ask_depth_usdt",
    "book_imbalance_top20", "oi_to_volume", "micro_osc_score",
]

# Features owned by the HMM winner calibrator / dependent on HMM lineage. Excluded
# to keep the meta-labeler non-redundant and lineage-invariant.
WINNER_OR_HMM = {
    "range_prob", "trend_prob", "persistence_prob", "survival_prob",
    "regime_conf", "hmm_tail_cvar_95", "tos_gcf",
    "hmm_range_prob_state", "hmm_trend_prob_state",
}

# Circular meta-features: the primary pipeline / deployment score can incorporate
# meta_prob (see mi_weighted_scorer signals = similarity, profile_proba, ev_score,
# meta_prob). Using it as a meta-labeler feature would feed the model's own output
# back to it. This is an enforced AFML invariant (tests/test_afml_compliance.py
# asserts "primary_pipeline_score" not in cfg.features; CHANGELOG records the
# removal as "a circular feature"). Keep it out regardless of its raw correlation.
CIRCULAR_EXCLUDE = {
    "primary_pipeline_score", "deployment_score", "score", "scan_score",
}

# Outcome / excursion / label-derived column substrings — belt-and-suspenders
# leakage screen on top of the allowlist.
OUTCOME_PATTERN = re.compile(
    r"(hlabel|pnl|realized|unrealized|mfe|mae|excursion|pnl_curve|rotation|sl_hit|"
    r"duration|time_to_peak|total_fills|total_costs|net_return|capital_fraction|"
    r"drawdown|liquidat|^y$|^y_|label)",
    re.IGNORECASE,
)


def _known_label_columns() -> frozenset[str]:
    """Best-effort import of the canonical label-leakage guard set."""
    try:
        from neutralgrid.models.meta_labeler import _KNOWN_LABEL_COLUMNS

        return frozenset(_KNOWN_LABEL_COLUMNS)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not import _KNOWN_LABEL_COLUMNS (%s); using regex only", exc)
        return frozenset()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return cast(pd.Series, pd.to_numeric(cast(pd.Series, df[col]), errors="coerce"))
    return pd.Series(np.nan, index=df.index)


def _effective_pnl(df: pd.DataFrame) -> pd.Series:
    net = _num(df, "net_pnl_pct")
    raw = _num(df, "pnl_pct")
    return net.where(net.notna(), raw)


def _bootstrap_auc_ci(
    y: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.RandomState
) -> tuple[float | None, float | None]:
    aucs: list[float] = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        bi = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[bi])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[bi], score[bi])))
    if not aucs:
        return None, None
    return round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    err = 0.0
    n = len(y)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        err += abs(y[m].mean() - p[m].mean()) * m.sum() / n
    return round(float(err), 3)


def _load_pool(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "training_data_*.csv")))
    if not files:
        raise FileNotFoundError(f"No training_data_*.csv under {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if "candidate_id" in df.columns:
        df = df.drop_duplicates(subset=["candidate_id"], keep="last").reset_index(drop=True)
    logger.info("Loaded %d files -> %d distinct rows", len(files), len(df))
    return df


def _modelable_features(df: pd.DataFrame) -> list[str]:
    label_cols = _known_label_columns()
    feats: list[str] = []
    for col in EXANTE_ALLOWLIST:
        if col not in df.columns:
            continue
        if (
            col in WINNER_OR_HMM
            or col in CIRCULAR_EXCLUDE
            or col in label_cols
            or OUTCOME_PATTERN.search(col)
        ):
            continue
        s = _num(df, col)
        if s.notna().mean() < 0.5 or s.nunique(dropna=True) <= 1:
            continue
        feats.append(col)
    return feats


def _cv_auc(
    Xv: pd.DataFrame,
    cols: list[str],
    y: np.ndarray,
    groups: np.ndarray,
    start: np.ndarray,
    t1: np.ndarray,
    estimator_fn,
    calibrate: bool,
    rng: np.random.RandomState,
) -> dict | None:
    cols = [c for c in cols if c in Xv.columns]
    if not cols:
        return None
    X = Xv[cols].to_numpy()
    oof = np.full(len(y), np.nan)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in splitter.split(X, y, groups=groups):
        ts0, ts1 = start[te].min(), t1[te].max()
        keep = ~((t1[tr] >= ts0) & (start[tr] <= ts1))  # purge train rows overlapping test window
        tri = tr[keep] if keep.sum() > 15 else tr
        if len(np.unique(y[tri])) < 2:
            continue
        imp = SimpleImputer(strategy="median").fit(X[tri])
        scl = StandardScaler().fit(imp.transform(X[tri]))
        est = estimator_fn()
        if calibrate:
            est = CalibratedClassifierCV(est, method="sigmoid", cv=3)
        est.fit(scl.transform(imp.transform(X[tri])), y[tri])
        oof[te] = est.predict_proba(scl.transform(imp.transform(X[te])))[:, 1]
    m = ~np.isnan(oof)
    if len(np.unique(y[m])) < 2:
        return None
    auc = float(roc_auc_score(y[m], oof[m]))
    lo, hi = _bootstrap_auc_ci(y[m], oof[m], 500, rng)
    k = max(1, int(0.1 * m.sum()))
    thr = np.sort(oof[m])[::-1][k - 1]
    top = oof[m] >= thr
    base = float(y[m].mean())
    return {
        "auc": round(auc, 3), "ci_lo": lo, "ci_hi": hi,
        "ece": _ece(y[m], oof[m]),
        "decile_precision": round(float(y[m][top].mean()), 3),
        "decile_lift": round(float(y[m][top].mean() / max(base, 1e-9)), 2),
        "n_scored": int(m.sum()),
    }


def _study(df: pd.DataFrame, tag: str, args: argparse.Namespace) -> dict:
    rng = np.random.RandomState(RANDOM_STATE)
    eff = _effective_pnl(df)
    dur = _num(df, "duration_hours")
    y = ((eff >= args.pnl_hurdle) & (dur <= args.max_duration_hours)).astype(int).to_numpy()
    result: dict = {
        "tag": tag, "n": int(len(df)), "n_pos": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "duration_unique": int(dur.nunique()),
        "pnl_hurdle": args.pnl_hurdle, "max_duration_hours": args.max_duration_hours,
    }
    if y.sum() < 8:
        result["note"] = "too few positives to model"
        return result

    feats = _modelable_features(df)
    result["n_features"] = len(feats)
    result["features"] = feats
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    Xv = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=pd.Index(feats))

    # Univariate screen on the true target
    mi = mutual_info_classif(Xv.to_numpy(), y, random_state=RANDOM_STATE)
    uni: list[dict] = []
    for j, col in enumerate(feats):
        a = float(roc_auc_score(y, Xv[col].to_numpy()))
        score = Xv[col].to_numpy() if a >= 0.5 else -Xv[col].to_numpy()
        lo, hi = _bootstrap_auc_ci(y, score, 500, rng)
        uni.append({
            "name": col, "dir_auc": round(max(a, 1 - a), 3), "mi": round(float(mi[j]), 3),
            "ci_lo": lo, "ci_hi": hi, "significant": bool(lo is not None and lo > 0.5),
        })
    uni.sort(key=lambda d: -d["dir_auc"])
    result["univariate_top"] = uni[:12]
    result["n_significant"] = int(sum(d["significant"] for d in uni))

    # MANDATORY leakage assertion
    leaks = [d["name"] for d in uni if d["dir_auc"] > args.leak_auc]
    result["leak_check_auc_gt_threshold"] = leaks
    if leaks:
        raise RuntimeError(
            f"[{tag}] LEAKAGE: features with univariate AUC > {args.leak_auc}: {leaks}. "
            "A label-derived/outcome column survived the allowlist+regex screen."
        )

    # Models under symbol-grouped, time-purged CV
    groups = df["symbol"].astype(str).to_numpy()
    start = pd.to_datetime(df["start_time_utc"], errors="coerce", utc=True).to_numpy()
    t1 = (
        pd.to_datetime(df["start_time_utc"], errors="coerce", utc=True)
        + pd.to_timedelta(np.minimum(dur.fillna(6.0).to_numpy(), args.max_duration_hours), unit="h")
    ).to_numpy()

    def logit():
        return LogisticRegression(class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)

    def gbm():
        return GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        )

    k_budget = max(2, int(y.sum() // 10))
    topk = [d["name"] for d in uni[:k_budget]]
    result["k_budget"] = k_budget
    result["topk_features"] = topk
    result["cv"] = {
        "all_features_logistic_calibrated": _cv_auc(Xv, feats, y, groups, start, t1, logit, True, rng),
        "topk_logistic_calibrated": _cv_auc(Xv, topk, y, groups, start, t1, logit, True, rng),
        "all_features_gbm": _cv_auc(Xv, feats, y, groups, start, t1, gbm, False, rng),
    }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/backtest_candidates"))
    p.add_argument("--pool", choices=["geometric", "all", "both"], default="both",
                   help="geometric = authoritative (mode==geometric); all = all distinct rows")
    p.add_argument("--pnl-hurdle", type=float, default=3.0)
    p.add_argument("--max-duration-hours", type=float, default=7.0)
    p.add_argument("--leak-auc", type=float, default=0.90,
                   help="fail if any surviving feature has univariate AUC above this")
    p.add_argument("--out", type=Path, default=Path("data/meta_labeler_feature_study.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = _load_pool(args.data_dir)
    out: dict = {}
    if args.pool in ("geometric", "both"):
        if "mode" in df.columns:
            geo = cast(pd.DataFrame, df[df["mode"] == "geometric"]).reset_index(drop=True)
        else:
            geo = df.iloc[0:0]
        out["geometric"] = _study(geo, "geometric", args)
    if args.pool in ("all", "both"):
        out["all_distinct"] = _study(df, "all_distinct", args)

    payload = json.dumps(out, indent=2, default=str)
    print(payload)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        logger.info("Wrote report to %s", args.out)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("Could not write %s: %s", args.out, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
