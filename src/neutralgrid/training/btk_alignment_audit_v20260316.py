"""
Backtest-to-Live Alignment Audit (alignment-v1).

Observational module that compares backtest predictions against live deployment
outcomes.  Builds on :class:`LiveOutcomeIngestor` linkage infrastructure
instead of duplicating matching logic.

**Compound keys:**

- Backtest side: ``(candidate_id, backtest_run_id)`` — unique per execution.
- Live side: ``(candidate_id, strategy_id)`` — unique per deployment.
- Join: ``candidate_id`` links the two sides (one-to-many: one backtest →
  N live deployments).

**This module NEVER modifies production outputs.**  It is purely observational
and produces a report DataFrame for human review.

AFML Compliance:
- Enforces calibration eligibility criteria distinct from backtest eligibility.
- Tracks ``formula_version`` to prevent mixing incompatible label semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlignmentAuditor:
    """Compare backtest predictions against live outcomes.

    Parameters
    ----------
    ingestor : LiveOutcomeIngestor
        Pre-configured ingestor for expired bot outcomes.  The auditor
        delegates all linkage / forensic matching to the ingestor's
        existing infrastructure.
    """

    def __init__(self, ingestor: object) -> None:
        # Accept the ingestor instance; avoid circular import by using object
        self._ingestor = ingestor

    # ── Public API ─────────────────────────────────────────────────────────

    def compute_alignment_report(
        self,
        backtest_results_dir: Path,
        *,
        min_round_trips: int = 2,
        max_unrealized_fraction: float = 0.50,
        required_formula_version: Optional[str] = None,
    ) -> pd.DataFrame:
        """Join live outcomes + backtest results, compute directional bias.

        Calibration eligibility (per live deployment row) requires ALL:

        1. Both live outcome AND matching backtest exist (join succeeds).
        2. Live bot had ``>= min_round_trips`` completed round trips.
        3. Backtest ``unrealized_fraction <= max_unrealized_fraction``.
        4. Live outcome has non-null ``realized_pnl_usdt``.
        5. ``formula_version`` matches ``required_formula_version`` (if set).

        Parameters
        ----------
        backtest_results_dir
            Directory containing ``backtest_results_*.csv`` files.
        min_round_trips
            Minimum completed round trips for a live bot to qualify.
        max_unrealized_fraction
            Maximum backtest ``unrealized_fraction`` for eligibility.
        required_formula_version
            If set, only include rows with this formula version.

        Returns
        -------
        pd.DataFrame
            One row per ``(candidate_id, strategy_id)`` pair.
        """
        live_df = self._load_live_outcomes()
        bt_df = self._load_backtest_results(backtest_results_dir)

        if live_df.empty or bt_df.empty:
            logger.warning(
                "Alignment audit: no data to join "
                "(live=%d rows, backtest=%d rows)",
                len(live_df),
                len(bt_df),
            )
            return pd.DataFrame()

        # Dedup backtests: keep latest per candidate_id
        if "backtest_timestamp" in bt_df.columns:
            bt_df = bt_df.sort_values("backtest_timestamp")
            bt_df = bt_df.drop_duplicates(subset=["candidate_id"], keep="last")

        # Join on candidate_id (one-to-many: one backtest → N live deploys)
        merged = pd.merge(
            live_df,
            bt_df,
            on="candidate_id",
            how="inner",
            suffixes=("_live", "_bt"),
        )

        if merged.empty:
            logger.info("Alignment audit: 0 matched pairs after join.")
            return pd.DataFrame()

        # Compute alignment metrics
        report = self._compute_metrics(merged)

        # Apply calibration eligibility gates
        report["calibration_eligible"] = self._apply_eligibility(
            report,
            min_round_trips=min_round_trips,
            max_unrealized_fraction=max_unrealized_fraction,
            required_formula_version=required_formula_version,
        )

        eligible_count = int(cast(pd.Series, report["calibration_eligible"]).sum())
        logger.info(
            "Alignment audit: %d total pairs, %d calibration-eligible",
            len(report),
            eligible_count,
        )
        return report

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_live_outcomes(self) -> pd.DataFrame:
        """Delegate to ingestor to get live outcome rows."""
        try:
            result = self._ingestor.ingest()  # type: ignore[attr-defined]
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                return pd.DataFrame()
            return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            logger.error("Failed to load live outcomes: %s", exc)
            return pd.DataFrame()

    @staticmethod
    def _load_backtest_results(results_dir: Path) -> pd.DataFrame:
        """Load and concatenate all backtest_results_*.csv files."""
        csv_files = sorted(results_dir.glob("backtest_results_*.csv"))
        if not csv_files:
            logger.warning("No backtest_results_*.csv in %s", results_dir)
            return pd.DataFrame()

        frames = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and "candidate_id" in df.columns:
                    frames.append(df)
            except Exception as exc:
                logger.warning("Skipping %s: %s", csv_path.name, exc)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
        """Compute alignment metrics from joined data."""
        rows = []
        for _, r in merged.iterrows():
            bt_net = _safe_float(r.get("net_pnl_pct_bt") or r.get("net_pnl_pct"))
            bt_real = _safe_float(
                r.get("realized_net_pnl_pct_bt") or r.get("realized_net_pnl_pct")
            )
            bt_uf = _safe_float(
                r.get("unrealized_fraction_bt") or r.get("unrealized_fraction")
            )
            live_pnl = _safe_float(r.get("pnl_pct_live") or r.get("pnl_pct"))

            # Live realized PnL %
            live_realized = _safe_float(r.get("realized_pnl_usdt"))
            live_invested = _safe_float(r.get("invested_margin_usdt"))
            live_realized_pct = (
                (live_realized / live_invested) * 100.0
                if live_realized is not None and live_invested and live_invested > 0
                else None
            )

            # Live unrealized fraction
            live_unrealized = _safe_float(r.get("unrealized_pnl_usdt"))
            live_uf = None
            if live_realized is not None and live_unrealized is not None:
                _abs_r = abs(live_realized)
                _abs_u = abs(live_unrealized)
                live_uf = _abs_u / (_abs_r + _abs_u) if (_abs_r + _abs_u) > 0 else 0.0

            # Directional bias = backtest realized − live realized
            directional_bias = (
                bt_real - live_realized_pct
                if bt_real is not None and live_realized_pct is not None
                else None
            )

            rows.append(
                {
                    "candidate_id": r.get("candidate_id"),
                    "strategy_id": r.get("strategy_id"),
                    "backtest_run_id": r.get("backtest_run_id"),
                    "backtest_timestamp": r.get("backtest_timestamp"),
                    "bt_net_pnl_pct": bt_net,
                    "bt_realized_net_pnl_pct": bt_real,
                    "bt_unrealized_fraction": bt_uf,
                    "live_pnl_pct": live_pnl,
                    "live_realized_pnl_pct": live_realized_pct,
                    "live_unrealized_fraction": live_uf,
                    "directional_bias": directional_bias,
                    "live_round_trips": _safe_int(r.get("total_trades_live") or r.get("total_trades")),
                    "live_duration_hours": _safe_float(r.get("duration_hours_live") or r.get("duration_hours")),
                    "formula_version": r.get("formula_version"),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _apply_eligibility(
        report: pd.DataFrame,
        *,
        min_round_trips: int,
        max_unrealized_fraction: float,
        required_formula_version: Optional[str],
    ) -> pd.Series:
        """Return boolean Series of calibration eligibility."""
        eligible = pd.Series(True, index=report.index)

        # Gate 1: live round trips
        if "live_round_trips" in report.columns:
            eligible &= report["live_round_trips"].fillna(0) >= min_round_trips

        # Gate 2: backtest unrealized fraction
        if "bt_unrealized_fraction" in report.columns:
            eligible &= report["bt_unrealized_fraction"].fillna(1.0) <= max_unrealized_fraction

        # Gate 3: live realized PnL available
        if "live_realized_pnl_pct" in report.columns:
            eligible &= report["live_realized_pnl_pct"].notna()

        # Gate 4: directional bias computable
        if "directional_bias" in report.columns:
            eligible &= report["directional_bias"].notna()

        # Gate 5: formula version match
        if required_formula_version is not None and "formula_version" in report.columns:
            eligible &= report["formula_version"] == required_formula_version

        return eligible


# ── Module-level helpers ──────────────────────────────────────────────────────

def _safe_float(val: object) -> Optional[float]:
    """Convert to float, returning None for NaN/None/invalid."""
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: object) -> Optional[int]:
    """Convert to int, returning None for NaN/None/invalid."""
    if val is None:
        return None
    try:
        f = float(val)  # type: ignore[arg-type]
        return int(f) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
