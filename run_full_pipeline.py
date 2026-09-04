#!/usr/bin/env python3
"""
Full NEUTRAL Grid Bot Pipeline: Scan -> Enrich -> Deploy.

Runs the complete workflow from scanning top symbols to producing
deployment-ready grid parameters.

Usage:
    python run_full_pipeline.py [--top-n N] [--min-score N] [--output FILE]

Examples:
    # Default: scan top 250 symbols, enrich, display results
    python run_full_pipeline.py

    # Scan top 100 with minimum score threshold
    python run_full_pipeline.py --top-n 100 --min-score 40

    # Save to specific file
    python run_full_pipeline.py --output results/deployment_candidates.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

# Ensure local `src/` package is used when running this script directly.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from neutralgrid import __version__ as PIPELINE_VERSION
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.scanner.scan import scan_top_symbols
from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.scanner.profile_model import load_profile_model
from neutralgrid.scanner.profile_model_walkforward import (
    resolve_active_pattern_profile_path,
    resolve_active_profile_model_path,
)
from neutralgrid.scanner.enrich_grid_params import enrich_with_grid_params, EnrichConfig
from neutralgrid.core.candidate_id import make_candidate_id_from_row
from neutralgrid.validation.hmm_regime import ensure_hmm_model

# AFML post-enrichment scoring (EV + meta-labeling)
try:
    from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
    from neutralgrid.models.meta_labeler import MetaLabeler
    AFML_POST_SCORING_AVAILABLE = True
except ImportError:
    PnLRanker = None
    RankingConfig = None
    MetaLabeler = None
    AFML_POST_SCORING_AVAILABLE = False

try:
    from neutralgrid.training import build_feature_snapshot, create_collector
    FINAL_SNAPSHOT_LOGGING_AVAILABLE = True
except ImportError:
    build_feature_snapshot = None
    create_collector = None
    FINAL_SNAPSHOT_LOGGING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("full_pipeline")
# httpx logs the entire request URL at INFO, including signed Binance query
# parameters.  Request outcome details remain available from this module's
# explicit connection/error logs without exposing credentials or signatures.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def _close_client_safely(client: Any) -> None:
    """Best-effort close that preserves the primary pipeline error."""

    try:
        await client.close()
    except Exception as exc:
        logger.warning("Failed to close Binance client cleanly: %s", exc)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse JSON artifact %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_hmm_artifact_version(hmm_dir: Path) -> str | None:
    metadata = _read_json_dict(hmm_dir / "metadata.json")
    version = metadata.get("artifact_version") if metadata else None
    if version is None:
        return hmm_dir.name if hmm_dir.name else None
    version_str = str(version).strip()
    return version_str if version_str else None


def _apply_afml_post_scoring(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute EV-based ranking after grid enrichment.

    Ranks candidates by expected net return (ev_24h with risk penalty)
    instead of a composite weighted score.

    EV = P(stay) x fills x profit - funding_cost - P(breakout) x |SL|
    ev_score = EV x (1 - risk_penalty)

    This function is strict: it does not impute missing inputs; it skips
    EV computation when required inputs are absent.
    """
    if not AFML_POST_SCORING_AVAILABLE:
        return df

    # Type narrowing: guaranteed non-None after AFML_POST_SCORING_AVAILABLE check
    assert PnLRanker is not None
    assert RankingConfig is not None
    assert MetaLabeler is not None

    out = df.copy()
    if 'score' in out.columns and 'scan_score' not in out.columns:
        out['scan_score'] = out['score']

    # Instantiate ranker
    ranker = PnLRanker(RankingConfig())
    ev_contract_fingerprint = getattr(ranker, "ev_contract_fingerprint", None)

    # Load meta-labeler if present so post-score output can retain diagnostics.
    # The current meta-labeler is not an admission/ranking/sizing authority.
    meta = None
    meta_path = Path('models/meta_labeler.pkl')
    if meta_path.exists():
        try:
            meta = MetaLabeler.load(meta_path)
        except Exception as exc:
            logger.warning("Failed to load meta-labeler for post-scoring: %s", exc)
            meta = None

    # FASTWIN-01: meta_prob is an authoritative deployment signal only when the
    # loaded model passed the promotion gate (promotion_status == "pass"). This
    # is the SAME model file enrich loads, so the two authority sources agree;
    # fail-closed to diagnostic_only when meta is absent/unpromoted.
    meta_authoritative = (getattr(meta, "promotion_status", None) == "pass")

    ev_scores = []
    ev_24h_vals = []
    ev_raw_vals = []
    ev_aligned_vals = []
    ev_expected_fills_vals = []
    ev_fill_scale_vals = []
    ev_fill_scope_vals = []
    ev_fill_scope_samples_vals = []
    ev_alignment_scope_vals = []
    ev_alignment_scope_samples_vals = []
    ev_alignment_r2_vals = []
    fill_revenue_vals = []
    funding_cost_vals = []
    boundary_loss_vals = []
    penalty_vals = []
    meta_probs = []
    meta_prob_sources = []
    ev_errors = []

    for _, row in out.iterrows():
        ev = None
        evh = None
        ev_raw = None
        ev_aligned = None
        ev_expected_fills = None
        ev_fill_scale = None
        ev_fill_scope = None
        ev_fill_scope_samples = None
        ev_alignment_scope = None
        ev_alignment_scope_samples = None
        ev_alignment_r2 = None
        fill_rev = None
        fund_cost = None
        bnd_loss = None
        penalty = None
        ev_failed = False
        meta_prob_source = "missing"
        try:
            profit_per_grid_pct = row.get('profit_per_grid_pct')
            num_grids = row.get('num_grids')
            survival_prob = row.get('survival_prob')
            trend_prob = row.get('trend_prob')
            hmm_trend_prob = row.get('hmm_trend_prob')
            hmm_range_prob = row.get('hmm_range_prob')
            range_prob = row.get('range_prob')
            if hmm_trend_prob is not None and pd.notna(hmm_trend_prob):
                if trend_prob is None or pd.isna(trend_prob):
                    raise AssertionError("Canonical post-enrich contract missing bare trend_prob")
                if float(trend_prob) != float(hmm_trend_prob):
                    raise AssertionError("Canonical post-enrich contract mismatch: trend_prob != hmm_trend_prob")
            if hmm_range_prob is not None and pd.notna(hmm_range_prob):
                if range_prob is None or pd.isna(range_prob):
                    raise AssertionError("Canonical post-enrich contract missing bare range_prob")
                if float(range_prob) != float(hmm_range_prob):
                    raise AssertionError("Canonical post-enrich contract mismatch: range_prob != hmm_range_prob")
            funding_rate = row.get('funding_rate')
            range_size_pct = row.get('range_size_pct')

            required = [profit_per_grid_pct, num_grids, survival_prob, trend_prob, range_size_pct]
            if all(v is not None and pd.notna(v) for v in required):
                # Type narrowing (validated by condition above)
                assert profit_per_grid_pct is not None
                assert num_grids is not None
                assert survival_prob is not None
                assert trend_prob is not None
                assert range_size_pct is not None
                _lev = row.get("leverage")
                _leverage = int(float(_lev)) if _lev is not None and pd.notna(_lev) else 10
                res = ranker.compute_score(
                    profit_per_grid_pct=float(profit_per_grid_pct),
                    num_grids=int(num_grids),
                    survival_prob=float(survival_prob),
                    trend_prob=float(trend_prob),
                    funding_rate=float(funding_rate) if funding_rate is not None and pd.notna(funding_rate) else None,
                    range_size_pct=float(range_size_pct),
                    symbol=str(row.get("symbol")) if row.get("symbol") is not None else None,
                    leverage=_leverage,
                )
                evh = round(float(res.ev_24h), 4)
                ev = round(float(res.rank_score), 4)
                ev_raw = round(float(res.ev_raw), 4)
                ev_aligned = round(float(res.ev_aligned), 4)
                ev_expected_fills = round(float(res.expected_fills), 4)
                ev_fill_scale = round(float(res.fill_rate_scale), 4)
                ev_fill_scope = str(res.fill_rate_scope)
                ev_fill_scope_samples = round(float(res.fill_rate_scope_samples), 1)
                ev_alignment_scope = str(res.ev_alignment_scope)
                ev_alignment_scope_samples = round(float(res.ev_alignment_scope_samples), 1)
                ev_alignment_r2 = round(float(res.ev_alignment_r2), 4)
                fill_rev = round(float(res.fill_revenue_pct), 4)
                fund_cost = round(float(res.funding_cost_pct), 4)
                bnd_loss = round(float(res.boundary_loss_pct), 4)
                penalty = round(float(res.total_penalty), 4)
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("EV computation failed for %s: %s", row.get('symbol', '?'), exc)
            ev = None
            ev_failed = True

        # Meta probability for deployment ranking and audit logging.
        mp = row.get("meta_prob")
        mp = float(mp) if mp is not None and pd.notna(mp) else None
        existing_source = row.get("meta_prob_source")
        if mp is not None:
            if existing_source in {"enrich", "post_scoring_backfill"}:
                meta_prob_source = str(existing_source)
            else:
                meta_prob_source = "enrich"
        if mp is None and meta is not None and getattr(meta, 'is_trained', False):
            try:
                feature_row = dict(row)
                if not meta.get_missing_feature_names(feature_row):
                    mp = float(meta.predict_proba(feature_row))
                    meta_prob_source = "post_scoring_backfill"
            except Exception:
                mp = None

        ev_scores.append(ev)
        ev_24h_vals.append(evh)
        ev_raw_vals.append(ev_raw)
        ev_aligned_vals.append(ev_aligned)
        ev_expected_fills_vals.append(ev_expected_fills)
        ev_fill_scale_vals.append(ev_fill_scale)
        ev_fill_scope_vals.append(ev_fill_scope)
        ev_fill_scope_samples_vals.append(ev_fill_scope_samples)
        ev_alignment_scope_vals.append(ev_alignment_scope)
        ev_alignment_scope_samples_vals.append(ev_alignment_scope_samples)
        ev_alignment_r2_vals.append(ev_alignment_r2)
        fill_revenue_vals.append(fill_rev)
        funding_cost_vals.append(fund_cost)
        boundary_loss_vals.append(bnd_loss)
        penalty_vals.append(penalty)
        ev_errors.append(ev_failed)
        meta_probs.append(mp)
        meta_prob_sources.append(meta_prob_source)

    out['ev_error'] = ev_errors
    out['ev_contract_fingerprint'] = ev_contract_fingerprint
    out['ev_score'] = ev_scores
    out['ev_24h'] = ev_24h_vals
    out['ev_raw'] = ev_raw_vals
    out['ev_aligned'] = ev_aligned_vals
    out['ev_expected_fills'] = ev_expected_fills_vals
    out['ev_fill_rate_scale'] = ev_fill_scale_vals
    out['ev_fill_context_scope'] = ev_fill_scope_vals
    out['ev_fill_context_samples'] = ev_fill_scope_samples_vals
    out['ev_alignment_context_scope'] = ev_alignment_scope_vals
    out['ev_alignment_context_samples'] = ev_alignment_scope_samples_vals
    out['ev_alignment_context_r2'] = ev_alignment_r2_vals
    out['ev_fill_revenue_pct'] = fill_revenue_vals
    out['ev_funding_cost_pct'] = funding_cost_vals
    out['ev_boundary_loss_pct'] = boundary_loss_vals
    out['ev_risk_penalty'] = penalty_vals
    out['meta_prob'] = meta_probs
    out['meta_prob_source'] = meta_prob_sources
    out['meta_prob_authority'] = (
        "authoritative" if meta_authoritative else "diagnostic_only"
    )

    # Observability backfill: fill meta_prob for any deployable row that enrich left
    # null. As of ERR-059, enrich Stage-12 computes ev_score BEFORE the meta probe,
    # so meta_prob is normally populated there (source "enrich") and IS consumed by
    # gate + Kelly sizing at enrich time. This post-scoring pass therefore only
    # back-fills the residual rows where the enrich-time probe could not run (e.g. a
    # missing meta-feature), purely so the output column is observable. Deployment
    # RANKING remains ev_score-based (deployment_score_source="ev_score"), not meta_prob.
    if meta is not None and getattr(meta, 'is_trained', False):
        for idx, row in out.iterrows():
            existing = out.at[idx, 'meta_prob']
            if existing is not None and pd.notna(existing):
                continue
            try:
                feature_row = dict(row)
                if not meta.get_missing_feature_names(feature_row):
                    out.at[idx, 'meta_prob'] = float(meta.predict_proba(feature_row))
                    out.at[idx, 'meta_prob_source'] = "post_scoring_backfill"
            except Exception as exc:
                logger.warning(
                    "Meta-labeler post-scoring backfill failed for %s: %s",
                    row.get('symbol', '?'), exc,
                )

    # Gate 4: Rank valid deployment candidates by expected value (EV).
    # Preserve the original scan-time composite score in score_original.
    if 'score' in out.columns:
        out['score_original'] = out['score']
        out['deployment_score'] = float('nan')
        out['deployment_score_source'] = None

        valid_mask = pd.Series(True, index=out.index)
        if "grid_is_valid" in out.columns:
            valid_mask = valid_mask & (out["grid_is_valid"] == True)  # noqa: E712
        err_mask = out['ev_error'] == True  # noqa: E712  (pandas Series)

        rank_mask = valid_mask & ~err_mask & out['ev_score'].notna()

        if bool(rank_mask.any()):
            ranked = out.loc[rank_mask].copy()
            ranked["__ev_sort"] = pd.to_numeric(ranked["ev_score"], errors="coerce")
            ranked["__score_sort"] = pd.to_numeric(ranked["score_original"], errors="coerce")
            ranked["__candidate_sort"] = (
                ranked["candidate_id"].astype(str)
                if "candidate_id" in ranked.columns
                else ranked["symbol"].astype(str)
            )
            ranked["__row_order"] = list(range(len(ranked)))
            ranked = ranked.sort_values(
                by=[
                    "__ev_sort",
                    "__score_sort",
                    "__candidate_sort",
                    "__row_order",
                ],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            total_ranked = len(ranked)
            deployment_scores = [
                50.0 + 50.0 * ((total_ranked - pos) / total_ranked)
                for pos in range(total_ranked)
            ]
            out.loc[ranked.index, 'deployment_score'] = deployment_scores
            out.loc[ranked.index, "deployment_score_source"] = "ev_score"

    # Safety assertion: no invalid candidate should have a deployment_score.
    if "grid_is_valid" in out.columns and "deployment_score" in out.columns:
        _invalid_scored = (
            (out["grid_is_valid"] != True)  # noqa: E712
            & out["deployment_score"].notna()
        )
        assert not bool(_invalid_scored.any()), (
            "BUG: %d invalid candidates have deployment_score" % int(_invalid_scored.sum())
        )

    if getattr(ranker, "empirical_profile", None) is not None:
        ep = ranker.empirical_profile
        assert ep is not None
        out["ev_alignment_samples"] = int(ep.ev_alignment_samples)
        out["ev_alignment_slope"] = float(ep.ev_alignment_slope)
        out["ev_alignment_intercept"] = float(ep.ev_alignment_intercept)
        out["ev_alignment_r2"] = float(ep.ev_alignment_r2)
        out["payoff_ratio_b"] = float(ep.payoff_ratio_b)

    return out


def _build_potential_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rejected-but-diagnostic rows for separate near-miss review."""
    if df.empty or "grid_is_valid" not in df.columns:
        return pd.DataFrame()

    invalid_mask = cast(pd.Series, df["grid_is_valid"]) != True  # noqa: E712
    potential = cast(pd.DataFrame, df.loc[invalid_mask].copy())
    if potential.empty:
        return potential

    evidence_mask = pd.Series(False, index=potential.index)
    for col in ["ev_score", "stage_b_reason", "hard_gate_reason", "rejection_reasons"]:
        if col in potential.columns:
            evidence_mask = evidence_mask | cast(pd.Series, potential[col]).notna()
    if "failure_stage" in potential.columns:
        failure_stage = cast(pd.Series, potential["failure_stage"]).astype(str)
        evidence_mask = evidence_mask | (
            failure_stage.notna()
            & ~failure_stage.str.lower().isin({"", "none", "nan", "not_enriched", "below_threshold"})
        )
    potential = cast(pd.DataFrame, potential.loc[evidence_mask].copy())
    if potential.empty:
        return potential

    if "deployment_score" in potential.columns:
        potential["deployment_score"] = float("nan")
    potential["potential_candidate"] = True

    for col in ["capital_fraction", "profit_per_grid_pct", "profit_per_grid_min_pct", "dynamic_profit_floor_pct"]:
        if col in potential.columns:
            potential[col] = pd.to_numeric(cast(pd.Series, potential[col]), errors="coerce")

    if "capital_fraction" in potential.columns:
        potential["distance_to_position_size_min"] = potential["capital_fraction"] - 0.05
    else:
        potential["distance_to_position_size_min"] = float("nan")

    if "profit_per_grid_pct" in potential.columns and "profit_per_grid_min_pct" in potential.columns:
        potential["distance_to_profit_floor_pct"] = (
            potential["profit_per_grid_pct"] - potential["profit_per_grid_min_pct"]
        )
    elif "profit_per_grid_pct" in potential.columns and "dynamic_profit_floor_pct" in potential.columns:
        potential["distance_to_profit_floor_pct"] = (
            potential["profit_per_grid_pct"] - potential["dynamic_profit_floor_pct"]
        )
    else:
        potential["distance_to_profit_floor_pct"] = float("nan")

    for col in ["ev_score", "score", "scan_score"]:
        if col in potential.columns:
            potential[col] = pd.to_numeric(cast(pd.Series, potential[col]), errors="coerce")
    potential["potential_score_source"] = "diagnostic"

    sort_cols = [c for c in ["ev_score", "score", "scan_score"] if c in potential.columns]
    if sort_cols:
        potential = potential.sort_values(
            by=sort_cols,
            ascending=[False] * len(sort_cols),
            kind="mergesort",
        )
        total = len(potential)
        potential["potential_score"] = [
            50.0 + 50.0 * ((total - pos) / total)
            for pos in range(total)
        ]
    else:
        potential["potential_score"] = float("nan")

    return potential


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full NEUTRAL Grid Bot Pipeline: Scan -> Enrich -> Deploy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=250,
        help="Number of top symbols to scan by quote volume (default: 250)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results/deployment_ready_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Optional isolated training-run bundle directory. When set, the "
            "deployment CSV is written as deployment_ready_<timestamp>.csv "
            "inside this directory and feature snapshots are written under "
            "<run-dir>/snapshots."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help=(
            "Minimum score threshold for enrichment "
            "(default: 45.0, or 20.0 with --discovery-mode)"
        ),
    )
    parser.add_argument(
        "--discovery-mode",
        action="store_true",
        default=False,
        help=(
            "Generate non-deployment discovery candidates by preserving gate "
            "verdicts but relaxing deployment-gate enforcement."
        ),
    )
    parser.add_argument(
        "--max-enrichment",
        type=int,
        default=250,
        help="Maximum symbols to enrich with grid params (default: 250)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent API requests (default: 8)",
    )
    parser.add_argument(
        "--compute-stochastic",
        action="store_true",
        default=None,
        help="Compute stochastic features (survival prob, Hurst, OU halflife)",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default=None,
        help=(
            "Explicit path to pattern profile JSON. Default (None) resolves "
            "the pattern artifact paired with the active model bundle."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit path to profile model JSON. Default (None) uses "
            "resolve_active_profile_model_path(): use data/profile/current.json when "
            "present, otherwise fall back to data/profile/profile_model.json as an "
            "unvalidated bootstrap candidate."
        ),
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all candidates (not just valid ones)",
    )
    parser.add_argument(
        "--strict-holdout",
        action="store_true",
        help="Treat holdout contamination as a hard failure (exit 1)",
    )
    parser.add_argument(
        "--diagnostic-surrogate-artifact",
        type=Path,
        default=None,
        help=(
            "Optional quarantined diagnostic surrogate. It writes a separate shadow "
            "report only and cannot affect meta gates, sizing, ranking, or deployment."
        ),
    )
    parser.add_argument(
        "--diagnostic-surrogate-output-dir",
        type=Path,
        default=Path("artifacts/diagnostics/meta_prob_shadow"),
        help="Diagnostic-only root for an opt-in surrogate shadow report.",
    )
    parser.add_argument(
        "--diagnostic-surrogate-allow-hmm-lineage-mismatch",
        action="store_true",
        help=(
            "Permit an explicitly extrapolative diagnostic shadow score when the "
            "pipeline HMM differs from the surrogate teacher HMM."
        ),
    )
    args = parser.parse_args(argv)
    if args.run_dir and args.output:
        parser.error("--run-dir cannot be combined with --output; use one provenance root")
    if args.min_score is None:
        args.min_score = 20.0 if args.discovery_mode else 45.0
    return args


def _resolve_pipeline_output_paths(args: argparse.Namespace, timestamp: str) -> tuple[Path, Path]:
    """Resolve deployment CSV and feature snapshot directories for one run."""
    run_dir_raw = getattr(args, "run_dir", None)
    if run_dir_raw:
        run_dir = Path(str(run_dir_raw))
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError(
                f"--run-dir must be new or empty to prevent provenance mixing: {run_dir}"
            )
        return run_dir / f"deployment_ready_{timestamp}.csv", run_dir / "snapshots"

    output_raw = getattr(args, "output", None)
    if output_raw:
        return Path(str(output_raw)), Path("data/training_snapshots")

    results_dir = Path("results")
    return results_dir / f"deployment_ready_{timestamp}.csv", Path("data/training_snapshots")


def _write_diagnostic_surrogate_shadow(
    *,
    args: argparse.Namespace,
    pipeline_output: Path,
    timestamp: str,
) -> None:
    """Emit a quarantined shadow report without changing the pipeline result."""

    artifact_path = getattr(args, "diagnostic_surrogate_artifact", None)
    if artifact_path is None:
        return
    try:
        from neutralgrid.diagnostics.meta_prob_shadow import score_csv_to_shadow_report

        output_dir = Path(args.diagnostic_surrogate_output_dir)
        output_path = output_dir / f"pipeline_{timestamp}" / "meta_prob_shadow.csv"
        summary = score_csv_to_shadow_report(
            artifact_path=Path(artifact_path),
            input_path=pipeline_output,
            output_path=output_path,
            allow_hmm_lineage_mismatch=bool(
                args.diagnostic_surrogate_allow_hmm_lineage_mismatch
            ),
            input_context="post_enrichment_pipeline_output",
        )
        logger.info(
            "Diagnostic surrogate shadow written: %s (%d/%d rows scored; cross-lineage=%s)",
            summary["output"],
            summary["scored_rows"],
            summary["rows"],
            summary["cross_lineage_extrapolation"],
        )
    except Exception as exc:
        logger.error(
            "Diagnostic surrogate shadow failed; deployment output remains unchanged: %s",
            exc,
            exc_info=True,
        )


def _log_run_health_summary(df: pd.DataFrame, profile_model: Any) -> None:
    """Log a concise run-health summary so silently-degraded subsystems are visible.

    Surfaces: whether the profile model was active (vs similarity-only scan), how
    many deployable rows received a meta_prob, and how many deployable candidates
    were dropped from ranking for missing EV inputs. Read-only / best-effort.
    """
    try:
        valid = (
            (df["grid_is_valid"] == True)  # noqa: E712
            if "grid_is_valid" in df.columns
            else pd.Series(False, index=df.index)
        )
        n_valid = int(valid.sum())
        if profile_model is not None:
            pm = f"active ({len(getattr(profile_model, 'features', []))} features)"
        else:
            pm = "ABSENT -> scan ran similarity-only (degraded)"
        if "meta_prob" in df.columns:
            mp = cast(pd.Series, pd.to_numeric(cast(pd.Series, df["meta_prob"]), errors="coerce"))
            mp_cov = f"{int((mp.notna() & valid).sum())}/{n_valid} deployable rows have meta_prob"
        else:
            mp_cov = "meta_prob column absent"
        if "deployment_score" in df.columns:
            ds = cast(pd.Series, pd.to_numeric(cast(pd.Series, df["deployment_score"]), errors="coerce"))
            excluded = int((valid & ds.isna()).sum())
        else:
            excluded = n_valid
        logger.info("")
        logger.info("RUN HEALTH SUMMARY")
        logger.info("  profile_model      : %s", pm)
        logger.info("  meta_prob coverage : %s", mp_cov)
        logger.info(
            "  ranking-excluded   : %d deployable candidate(s) had no deployment_score (missing EV inputs)",
            excluded,
        )
    except Exception as exc:
        logger.warning("Run health summary failed: %s", exc)


def display_deployment_summary(
    df: pd.DataFrame,
    show_all: bool = False,
    *,
    discovery_mode: bool = False,
) -> None:
    """Display deployment-ready candidates summary."""
    import sys
    # Handle Windows encoding issues with special characters
    if sys.platform == "win32":
        try:
            import io
            if isinstance(sys.stdout, io.TextIOWrapper):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    print()
    print("=" * 120)
    title = "DISCOVERY GRID CANDIDATES" if discovery_mode else "DEPLOYMENT-READY GRID PARAMETERS"
    print(f"{title}  (pipeline v{PIPELINE_VERSION})")
    print("=" * 120)
    print()

    # Filter for valid grids only (unless show_all)
    bt_col = "below_threshold_tag"
    has_bt = bt_col in df.columns
    # Gate 4 (Fix 1b): Sort by deployment_score (EV-ranked) for display
    _sort_col = "deployment_score" if "deployment_score" in df.columns else "score"
    if show_all:
        valid_df = df.sort_values(_sort_col, ascending=False)
        print(f"Showing all {len(valid_df)} candidates (including invalid)")
    else:
        valid_df = cast(pd.DataFrame, df[df["grid_is_valid"] == True]).sort_values(by=_sort_col, ascending=False)  # noqa: E712
        total_valid = len(valid_df)
        total_candidates = len(df)
        # Count below-threshold tagged (enriched but not auto-valid)
        tagged_count = 0
        if has_bt:
            tag_series = cast(pd.Series, valid_df[bt_col])
            tagged_count = int(tag_series.fillna(False).astype(bool).sum())
        auto_deploy = total_valid - tagged_count
        print(f"Valid Candidates: {total_valid} / {total_candidates}")
        if tagged_count:
            print(f"  Auto-deploy: {auto_deploy}  |  Below-threshold tagged: {tagged_count}")

    if valid_df.empty:
        print("No valid deployment candidates found.")
        print()
        if not show_all:
            print("Run with --show-all to see why candidates were rejected.")
        return

    print()
    print("-" * 140)
    print(f"{'Symbol':<12} {'EV%':>7} {'EV24h%':>7} {'Valid':>6} {'Lower':>10} {'Upper':>10} {'Grids':>6} {'Profit/G':>9} {'Edge':>4} {'NetE%':>6} {'SurvP':>6} {'RangeP':>7} {'Reason':<20}")
    print("-" * 140)

    display_count = min(20, len(valid_df))
    for _, row in valid_df.head(display_count).iterrows():
        symbol = row["symbol"]
        ev_score = row.get("ev_score")
        ev_24h = row.get("ev_24h")
        ev_str = f"{ev_score:.2f}" if ev_score is not None and pd.notna(ev_score) else f"{row['score']:.2f}"
        ev24_str = f"{ev_24h:.2f}" if ev_24h is not None and pd.notna(ev_24h) else "-"
        is_valid = "YES" if row.get("grid_is_valid") else "NO"
        if row.get("below_threshold_tag"):
            is_valid = "TAG"

        grid_lower = f"${row['grid_lower']:.4f}" if bool(pd.notna(row.get("grid_lower"))) else "-"
        grid_upper = f"${row['grid_upper']:.4f}" if bool(pd.notna(row.get("grid_upper"))) else "-"
        num_grids_raw = row.get("num_grids")
        num_grids = "-"
        if num_grids_raw is not None and bool(pd.notna(num_grids_raw)):
            num_grids = f"{int(float(cast(Any, num_grids_raw)))}"
        profit = f"{row['profit_per_grid_pct']:.2f}%" if bool(pd.notna(row.get("profit_per_grid_pct"))) else "-"
        surv_prob = f"{row['survival_prob']:.3f}" if bool(pd.notna(row.get("survival_prob"))) else "-"
        range_prob = f"{row['hmm_range_prob']:.3f}" if bool(pd.notna(row.get("hmm_range_prob"))) else "-"
        reason = str(row.get("grid_reason", "-"))[:20]

        # Edge tier columns
        edge_raw = row.get("edge_tier_chosen")
        edge = {"BIG": "BIG", "MEDIUM": "MED"}.get(edge_raw, "-") if edge_raw and bool(pd.notna(edge_raw)) else "-"
        net_e = row.get("net_edge_pct")
        net_e_str = f"{net_e:.2f}" if net_e is not None and pd.notna(net_e) else "-"

        print(f"{symbol:<12} {ev_str:>7} {ev24_str:>7} {is_valid:>6} {grid_lower:>10} {grid_upper:>10} {num_grids:>6} {profit:>9} {edge:>4} {net_e_str:>6} {surv_prob:>6} {range_prob:>7} {reason:<20}")

    if len(valid_df) > display_count:
        print(f"... and {len(valid_df) - display_count} more candidates")

    # Detailed view for top 5 valid candidates
    print()
    print("=" * 120)
    print("DETAILED DEPLOYMENT PARAMETERS - TOP 5 VALID CANDIDATES")
    print("=" * 120)

    _sort_col_detail = "deployment_score" if "deployment_score" in df.columns else "score"
    valid_only = cast(pd.DataFrame, df[df["grid_is_valid"] == True]).sort_values(by=_sort_col_detail, ascending=False)  # noqa: E712
    for _, row in valid_only.head(5).iterrows():
        print()
        print("=" * 80)
        ev_disp = row.get('ev_score')
        ev_label = f"EV: {ev_disp:.2f}%" if ev_disp is not None and pd.notna(ev_disp) else f"Score: {row['score']:.2f}"
        status_label = "DISCOVERY CANDIDATE" if discovery_mode else "READY FOR DEPLOYMENT"
        print(f"  {row['symbol']} | {ev_label} | {status_label}")
        print("=" * 80)

        print("  EV DECOMPOSITION:")
        if bool(pd.notna(row.get("ev_24h"))):
            print(f"    Raw EV (24h):      {row.get('ev_24h', 0):+.2f}%")
        if bool(pd.notna(row.get("ev_fill_revenue_pct"))):
            print(f"    Fill Revenue:      +{row.get('ev_fill_revenue_pct', 0):.2f}%")
        if bool(pd.notna(row.get("ev_funding_cost_pct"))):
            print(f"    Funding Cost:      -{row.get('ev_funding_cost_pct', 0):.2f}%")
        if bool(pd.notna(row.get("ev_boundary_loss_pct"))):
            print(f"    Boundary Loss:     -{row.get('ev_boundary_loss_pct', 0):.2f}%")
        if bool(pd.notna(row.get("ev_risk_penalty"))):
            print(f"    Risk Penalty:       {row.get('ev_risk_penalty', 0):.2f}")
        if bool(pd.notna(row.get("ev_score"))):
            print(f"    Risk-adj EV:       {row.get('ev_score', 0):+.2f}%")

        print("  GRID PARAMETERS:")
        print(f"    Grid Lower:         ${row.get('grid_lower', 0):.6f}")
        print(f"    Grid Upper:         ${row.get('grid_upper', 0):.6f}")
        print(f"    Number of Grids:    {int(row.get('num_grids') or 0)}")
        print(f"    Grid Spacing:       {row.get('grid_spacing_pct', 0):.2f}%")
        print(f"    Profit per Grid:    {row.get('profit_per_grid_pct', 0):.2f}%")
        print(f"    Leverage:           {int(row.get('leverage') or 10)}x")
        print(f"    Stop Loss:          {row.get('stop_loss_pct', -10):.1f}%")

        print("  EDGE TIER:")
        edge_chosen = row.get("edge_tier_chosen", "-")
        print(f"    Tier Chosen:        {edge_chosen if bool(pd.notna(edge_chosen)) else '-'}")
        if bool(pd.notna(row.get("net_edge_pct"))):
            print(f"    Net Edge:           {row.get('net_edge_pct', 0):.4f}%")
        if bool(pd.notna(row.get("edge_upgrade_success"))):
            print(f"    Upgrade Success:    {'YES' if row.get('edge_upgrade_success') else 'NO'}")
        if bool(pd.notna(row.get("edge_fallback_reason"))) and row.get("edge_fallback_reason"):
            print(f"    Fallback Reason:    {str(row.get('edge_fallback_reason'))[:60]}")
        if row.get("below_threshold_tag"):
            print(f"    Below-Threshold:    YES (enriched but not auto-valid)")

        print("  REGIME METRICS:")
        print(f"    HMM Range Prob:     {row.get('hmm_range_prob', 0):.4f}")
        print(f"    HMM Trend Prob:     {row.get('hmm_trend_prob', 0):.4f}")
        if bool(pd.notna(row.get("survival_prob"))):
            print(f"    Survival Prob:      {row.get('survival_prob', 0):.3f}")
        if bool(pd.notna(row.get("hurst_exponent"))):
            print(f"    Hurst Exponent:     {row.get('hurst_exponent', 0):.3f}")
        if bool(pd.notna(row.get("ou_halflife"))):
            print(f"    OU Halflife:        {row.get('ou_halflife', 0):.1f} bars")

        print("  TECHNICAL INDICATORS:")
        print(f"    ADX (1h):           {row.get('adx_1h', 0):.1f}")
        print(f"    RSI (15m):          {row.get('rsi_15m', 0):.1f}")
        print(f"    BB Width:           {row.get('bb_width', 0):.4f}")
        print(f"    Range Size:         {row.get('range_size_pct', 0):.2f}%")

        print("  MICROSTRUCTURE (if available):")
        if bool(pd.notna(row.get("micro_round_trip_cost_pct"))):
            print(f"    Round-trip Cost:    {row.get('micro_round_trip_cost_pct', 0):.4f}%")
        if bool(pd.notna(row.get("micro_viable"))):
            print(f"    Micro Viable:       {'YES' if row.get('micro_viable') else 'NO'}")

        print("  MARKET DATA:")
        if bool(pd.notna(row.get("last_price"))):
            print(f"    Last Price:         ${row.get('last_price', 0):.6f}")
        if bool(pd.notna(row.get("quote_volume_24h"))):
            vol = float(row.get("quote_volume_24h") or 0)
            print(f"    24h Volume:         ${vol/1e6:.2f}M")
        if bool(pd.notna(row.get("funding_rate"))):
            print(f"    Funding Rate:       {float(row.get('funding_rate') or 0)*100:.4f}%")

    print()
    print("=" * 120)


def _check_artifact_version_consistency() -> None:
    """Warn on lineage/pipeline incompatibilities across deployed artifacts."""
    import json as _json
    from neutralgrid.models.artifacts import resolve_hmm_artifact_dir
    from neutralgrid.models.artifact_compat import evaluate_hmm_meta_labeler_compatibility

    def _read_metadata(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    try:
        hmm_metadata_path = resolve_hmm_artifact_dir() / "metadata.json"
    except Exception:
        return  # HMM not configured: skip cross-artifact check

    meta_metadata_path = Path("models/meta_labeler/metadata.json")
    hmm_metadata = _read_metadata(hmm_metadata_path)
    meta_metadata = _read_metadata(meta_metadata_path)
    if not hmm_metadata or not meta_metadata:
        return

    issues = evaluate_hmm_meta_labeler_compatibility(
        hmm_metadata=hmm_metadata,
        meta_metadata=meta_metadata,
    )
    if issues:
        logger.warning("Artifact compatibility issues detected:")
        for issue in issues:
            logger.warning("  %s", issue)

async def main():
    """Main pipeline execution."""
    args = parse_args()

    # Setup output path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path, snapshot_dir = _resolve_pipeline_output_paths(args, timestamp)

    logger.info("=" * 80)
    logger.info("NEUTRAL GRID BOT - FULL PIPELINE  v%s", PIPELINE_VERSION)
    logger.info("Scan -> Validate -> Enrich -> Deploy")
    logger.info("=" * 80)
    if args.discovery_mode:
        logger.warning(
            "DISCOVERY MODE ENABLED: gate verdicts are preserved, but deployment "
            "gate enforcement is relaxed for data generation."
        )

    # =========================================================================
    # STEP 1: Load Artifacts
    # =========================================================================
    logger.info("")
    logger.info("STEP 1: Loading model artifacts...")
    logger.info("-" * 40)

    if args.profile_path is None:
        cfg = get_config()
        profile_path = resolve_active_pattern_profile_path(
            cfg.resolve_path(cfg.artifacts.profile_dir)
        )
        logger.info(
            "  Resolving default pattern profile via current.json/bootstrap → %s",
            profile_path,
        )
    else:
        profile_path = Path(args.profile_path)
        logger.info("  Loading pattern profile from explicit --profile-path: %s", profile_path)
    if not profile_path.exists():
        logger.error(f"Pattern profile not found: {profile_path}")
        logger.error("Run: python retrain_scanner.py --input data/new_expired_bots.xlsx")
        sys.exit(1)

    try:
        loaded_profile = PatternProfile.load_json(profile_path)
        if loaded_profile is None:
            raise ValueError(f"Pattern profile could not be parsed: {profile_path}")
        pattern_profile = loaded_profile
        logger.info(f"  Pattern profile loaded ({len(pattern_profile.features)} features)")
    except Exception as e:
        logger.error(f"Failed to load pattern profile: {e}")
        sys.exit(1)

    profile_model = None
    if args.model_path is None:
        cfg = get_config()
        model_path = resolve_active_profile_model_path(
            cfg.resolve_path(cfg.artifacts.profile_dir)
        )
        logger.info(f"  Resolving default profile model via current.json/bootstrap → {model_path}")
    else:
        model_path = Path(args.model_path)
        logger.info(f"  Loading profile model from explicit --model-path: {model_path}")
    if model_path.exists():
        try:
            profile_model = load_profile_model(model_path)
            logger.info(f"  Profile model loaded ({len(profile_model.features)} features)")
        except Exception as e:
            logger.warning(f"Failed to load profile model: {e}")
            profile_model = None
    if profile_model is not None and profile_model.features != pattern_profile.features:
        logger.error(
            "Profile bundle feature mismatch: model features %s != pattern features %s",
            profile_model.features,
            pattern_profile.features,
        )
        sys.exit(1)

    # Load meta-labeler for scan-phase scoring and bet sizing
    scan_meta_labeler = None
    if AFML_POST_SCORING_AVAILABLE and MetaLabeler is not None:
        meta_path = Path('models/meta_labeler.pkl')
        if meta_path.exists():
            try:
                scan_meta_labeler = MetaLabeler.load(meta_path)
                if getattr(scan_meta_labeler, 'is_trained', False):
                    logger.info("  Meta-labeler loaded for scan-phase scoring")
                else:
                    scan_meta_labeler = None
                    logger.info("  Meta-labeler found but not trained, skipping")
            except Exception as e:
                logger.warning(f"Failed to load meta-labeler: {e}")

    # Cross-model version consistency check
    _check_artifact_version_consistency()

    # Fail-fast: production pipeline requires an active rolling HMM artifact.
    try:
        hmm_dir = await ensure_hmm_model(force=False)
        logger.info("  HMM artifact ready: %s", hmm_dir)
    except Exception as e:
        logger.error("HMM artifact check failed: %s", e)
        sys.exit(1)
    active_hmm_version = _resolve_hmm_artifact_version(Path(hmm_dir))

    # Phase 2: keep runtime MI artifact deactivated until it is verifiably
    # aligned with scan-time signal definitions and a non-null MI target.
    mi_weights_for_scan: dict[str, float] | None = None
    try:
        from neutralgrid.scanner.mi_weighted_scorer_v20260311 import (
            default_runtime_mi_weights as _default_runtime_mi_weights,
        )
        _mi_path = Path("data/cache/mi_weights.json")
        if _mi_path.exists():
            logger.warning(
                "  MI weights artifact present but runtime artifact mode is disabled "
                "until source columns/raw MI/target semantics are verifiable; "
                "scan will use uniform runtime weights: %s",
                {k: f"{v:.3f}" for k, v in _default_runtime_mi_weights().items()},
            )
        else:
            logger.info(
                "  MI weights file not found — scan will use uniform runtime MI weights: %s",
                {k: f"{v:.3f}" for k, v in _default_runtime_mi_weights().items()},
            )
    except Exception as _mi_err:
        logger.debug("MI weights not available: %s", _mi_err)

    # =========================================================================
    # STEP 1.5: Validate holdout integrity (AFML Station 4)
    # =========================================================================
    try:
        from neutralgrid.training.holdout_validator import HoldoutValidator
        validator = HoldoutValidator()
        result = validator.validate_integrity()
        if result['contaminated']:
            logger.warning(
                "Holdout contamination detected: %d premature access(es)",
                result['premature_access_count'],
            )
            if args.strict_holdout:
                logger.error("--strict-holdout enabled: aborting due to contamination")
                sys.exit(1)
        elif result['valid']:
            logger.info("Holdout integrity: VALID")
    except Exception as e:
        logger.debug("Holdout validation skipped: %s", e)

    # =========================================================================
    # STEP 2: Initialize Binance Client
    # =========================================================================
    logger.info("")
    logger.info("STEP 2: Connecting to Binance...")
    logger.info("-" * 40)

    client = BinanceClient()

    try:
        status = await client.check_connection()
        if not status.get("connected"):
            logger.error("Failed to connect to Binance API")
            await _close_client_safely(client)
            sys.exit(1)
        logger.info(f"  Connected (authenticated: {status.get('authenticated', False)})")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        await _close_client_safely(client)
        sys.exit(1)

    # Capital source: fixed base capital from config (account-balance sizing removed).
    # Per-bot deployed margin is still scaled downstream by the dynamic capital_fraction.
    base_capital = float(get_config().grid.capital)
    logger.info("  Capital source: fixed ($%.2f)", base_capital)

    # =========================================================================
    # STEP 3: Scan Top Symbols
    # =========================================================================
    logger.info("")
    logger.info(f"STEP 3: Scanning top {args.top_n} symbols by quote volume...")
    logger.info("-" * 40)

    scan_data_cache: dict[str, dict[str, Any]] = {}
    try:
        df = await scan_top_symbols(
            client=client,
            profile=pattern_profile,
            profile_model=profile_model,
            meta_labeler=scan_meta_labeler,
            compute_stochastic=args.compute_stochastic,
            top_n=args.top_n,
            max_concurrency=args.max_concurrency,
            min_delay_s=0.05,
            mi_weights=mi_weights_for_scan,
            scan_data_out=scan_data_cache,
        )
        logger.info(f"  Scanned {len(df)} symbols")
        logger.info(f"  Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        await _close_client_safely(client)
        sys.exit(1)

    # =========================================================================
    # STEP 4: Enrich with Grid Parameters
    # =========================================================================
    logger.info("")
    logger.info(f"STEP 4: Enriching candidates with grid parameters...")
    logger.info(f"        (score >= {args.min_score}, max {args.max_enrichment} symbols)")
    logger.info("-" * 40)

    enrich_cfg = EnrichConfig(
        score_threshold=args.min_score,
        max_symbols=args.max_enrichment,
        concurrency=min(5, args.max_concurrency),
        capital_base_usdt=base_capital,
        discovery_mode=bool(args.discovery_mode),
    )

    try:
        df_enriched = await enrich_with_grid_params(
            df_candidates=df,
            client=client,
            pattern_profile=pattern_profile,
            cfg=enrich_cfg,
            scan_data_cache=scan_data_cache,
            meta_labeler=scan_meta_labeler,
        )

        # AFML: compute EV/meta probabilities after enrichment (strict, no imputation for EV)
        df_enriched = _apply_afml_post_scoring(df_enriched)

        valid_count = len(df_enriched[df_enriched["grid_is_valid"] == True])  # noqa: E712
        enriched_count = len(df_enriched[df_enriched["grid_reason"].notna()])
        logger.info(f"  Enriched {enriched_count} candidates")
        logger.info(f"  Valid for deployment: {valid_count}")
    except Exception as e:
        logger.error(f"Enrichment failed: {e}", exc_info=True)
        await _close_client_safely(client)
        sys.exit(1)

    # No later phase performs network I/O.  Close before snapshot/CSV writes so
    # a local persistence error cannot strand the HTTP client or its sockets.
    await _close_client_safely(client)

    # =========================================================================
    # STEP 5: Save and Display Results
    # =========================================================================
    logger.info("")
    logger.info("STEP 5: Saving results...")
    logger.info("-" * 40)

    # Stamp pipeline version into output for traceability
    df_enriched["pipeline_version"] = PIPELINE_VERSION
    df_enriched["pipeline_run_id"] = timestamp
    df_enriched["discovery_mode"] = bool(args.discovery_mode)
    df_enriched["active_hmm_artifact_version"] = active_hmm_version
    # FASTWIN auditability (METALABELER_LIVETELEMETRY.md Done Criterion 6): make every
    # meta-labeler-influenced deployment decision auditable from the results CSV. The
    # meta_prob / meta_prob_source / meta_prob_authority / deployment_score_source /
    # meta_size_factor columns are already stamped by _apply_afml_post_scoring; here we
    # add the lineage / feature-profile / Stage-B-gate context. When the meta-labeler is
    # loaded, its HMM lineage equals active_hmm_version (the artifact_compat load gate
    # enforces the match; a mismatch raises and scan_meta_labeler stays None).
    if scan_meta_labeler is not None:
        from neutralgrid.models.meta_labeler import META_FEATURE_PROFILES
        _meta_feats = tuple(getattr(scan_meta_labeler, "_feature_names", []) or [])
        df_enriched["meta_feature_profile"] = next(
            (name for name, feats in META_FEATURE_PROFILES.items() if tuple(feats) == _meta_feats),
            None,
        )
        df_enriched["meta_labeler_hmm_artifact_version"] = active_hmm_version
    else:
        df_enriched["meta_feature_profile"] = None
        df_enriched["meta_labeler_hmm_artifact_version"] = None
    df_enriched["stage_b_meta_gate_enabled"] = bool(
        getattr(get_config().two_stage, "meta_gate_enabled", False)
    )
    # base_capital is the fixed config capital (account-balance sizing removed)
    if "capital_fraction" in df_enriched.columns:
        cap_frac = cast(pd.Series, pd.to_numeric(df_enriched["capital_fraction"], errors="coerce")).fillna(1.0)
    else:
        cap_frac = pd.Series(1.0, index=df_enriched.index, dtype=float)
    cap_frac = cast(pd.Series, cap_frac).clip(0.0, 1.0)
    df_enriched["capital_base_usdt"] = base_capital
    df_enriched["deploy_margin_usdt"] = (cap_frac * base_capital).round(6)
    df_enriched["candidate_id"] = df_enriched.apply(
        lambda row: make_candidate_id_from_row(row, timestamp), axis=1
    )

    if FINAL_SNAPSHOT_LOGGING_AVAILABLE and build_feature_snapshot is not None and create_collector is not None:
        collector = create_collector(log_dir=str(snapshot_dir), enabled=True)
        snapshot_ts = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        snapshot_ts = snapshot_ts.replace(tzinfo=timezone.utc)
        try:
            for _, row in df_enriched.iterrows():
                snapshot = build_feature_snapshot(
                    symbol=str(row.get("symbol", "")),
                    scanner_row=row.to_dict(),
                    validation_result=None,
                    timestamp=snapshot_ts,
                )
                collector.log(snapshot)
        finally:
            collector.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    potential_candidates = _build_potential_candidates(df_enriched)
    potential_path = output_path.with_name(f"potential_candidates_{timestamp}.csv")
    if not potential_candidates.empty:
        potential_candidates.to_csv(potential_path, index=False)
        logger.info("  Potential candidates saved to: %s", potential_path)
    else:
        logger.info("  Potential candidates: none")
    df_enriched.to_csv(output_path, index=False)
    logger.info(f"  Results saved to: {output_path}")
    _write_diagnostic_surrogate_shadow(
        args=args,
        pipeline_output=output_path,
        timestamp=timestamp,
    )

    # Display deployment summary
    display_deployment_summary(
        df_enriched,
        show_all=args.show_all,
        discovery_mode=bool(args.discovery_mode),
    )

    _log_run_health_summary(df_enriched, profile_model)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Pipeline v%s complete. Results: %s", PIPELINE_VERSION, output_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
