"""
Live outcome ingestor — joins expired bot data with deploy linkage.

Reads ``new_expired_bots.xlsx`` (or CSV) and the deploy linkage log to
produce outcome records keyed by ``candidate_id``.  When linkage is
unavailable, falls back to forensic timestamp matching (same logic as
``scripts/match_candidates_to_bots.py``).

The output is a DataFrame with:
- candidate_id (from linkage or forensic match)
- scan-time features (from the deployment_ready CSV)
- live outcome fields (pnl_pct, duration, trades, etc.)
- provenance metadata (source="live", match_method, etc.)

Usage::

    from neutralgrid.training.live_outcome_ingestor import LiveOutcomeIngestor

    ingestor = LiveOutcomeIngestor(
        expired_bots_path="data/new_expired_bots.xlsx",
        linkage_dir="data/linkage",
        scanner_results_dir="results",
    )
    outcomes_df = ingestor.ingest()
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from neutralgrid.models.meta_labeler import normalize_inference_feature_dict

logger = logging.getLogger(__name__)

_FILENAME_TS_RE = re.compile(r"(\d{8}_\d{6})")

# Fields to extract from scanner CSVs as scan-time features
_SCAN_FEATURE_FIELDS = [
    "score",
    "range_prob",
    "trend_prob",
    "hmm_range_prob",
    "hmm_trend_prob",
    "utility_score",
    "regime_utility",
    "survival_prob",
    "hurst_exponent",
    "ou_halflife",
    "profit_per_grid_pct",
    "num_grids",
    "range_size_pct",
    "adx_1h",
    "adx_15m",
    "adx_5m",
    "rsi_15m",
    "ema_slope_1h",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "bb_width",
    "atr_pct_15m",
    "funding_rate",
    "open_interest",
    "quote_volume_24h",
    "ev_score",
    "ev_24h",
    "meta_prob",
    "meta_prob_source",
    "deployment_score",
    "scan_meta_prob",
    "winner_proba",
    "grid_lower",
    "grid_upper",
    "grid_spacing_pct",
    "leverage",
    "regime_conf",
    "micro_round_trip_cost_pct",
    "net_edge_pct",
    "capital_fraction",
    # Step 7: enriched/context fields for live→training parity
    "long_short_ratio",
    "funding_rate_zscore",
    "open_interest_change_pct",
    "bb_width_ratio_1h_15m",
    "tos_gcf",
    "hmm_tail_cvar_95",
    "persistence_prob",
]

# Live outcome fields from expired bots
_OUTCOME_FIELDS = [
    "strategy_id",
    "symbol",
    "pnl_pct",
    "total_profit_usdt",
    "realized_pnl_usdt",
    "unrealized_pnl_usdt",
    "commission_usdt",
    "funding_fee_usdt",
    "duration_hours",
    "total_trades",
    "maker_count",
    "taker_count",
    "grids_count",
    "invested_margin_usdt",
    "leverage",
    "price_range_low",
    "price_range_high",
    "grid_spacing_pct",
    "profit_factor",
    "mae",
    "mfe",
    "mae_pct_initial",
    "mfe_pct_initial",
    "start_time_utc",
    "end_time_utc",
    "trend_structure",
    "liq_price_long",
    "liq_price_short",
    "dist_to_liq_long_pct",
    "dist_to_liq_short_pct",
    "liq_range_utilization",
    "liq_asymmetry",
    "trigger_price",
]


class LiveOutcomeIngestor:
    """Joins expired bot outcomes with scanner candidate data.

    Parameters
    ----------
    expired_bots_path : Path
        Path to ``new_expired_bots.xlsx`` or CSV with live bot results.
    linkage_dir : Path, optional
        Directory containing ``deploy_linkage_log.csv``.
    scanner_results_dir : Path, optional
        Directory containing ``deployment_ready_*.csv`` files.
    min_bot_date : str
        Only process bots with ``start_time_utc >= min_bot_date``.
    """

    def __init__(
        self,
        expired_bots_path: Path,
        linkage_dir: Optional[Path] = None,
        scanner_results_dir: Optional[Path] = None,
        min_bot_date: str = "2026-02-01",
    ) -> None:
        self._bots_path = Path(expired_bots_path)
        self._linkage_dir = Path(linkage_dir) if linkage_dir else Path("data/linkage")
        self._scanner_dir = Path(scanner_results_dir) if scanner_results_dir else Path("results")
        self._min_date = min_bot_date
        self._linkage_conflict_strategy_ids: set[str] = set()

    def ingest(self) -> pd.DataFrame:
        """Run the full ingestion pipeline.

        Returns
        -------
        pd.DataFrame
            Outcome records with candidate_id, scan features, and live results.
        """
        bots_df = self._load_bots()
        if bots_df.empty:
            logger.warning("No bots loaded — returning empty DataFrame")
            return pd.DataFrame()

        # Try linkage-based join first
        linkage_df = self._load_linkage()
        scanner_csvs = self._parse_scanner_csvs()

        rows: List[Dict[str, Any]] = []
        linked_count = 0
        forensic_count = 0
        unmatched_count = 0
        conflict_count = 0

        for _, bot in bots_df.iterrows():
            strategy_id = str(bot.get("strategy_id", ""))
            _symbol = str(bot["symbol"])

            if strategy_id in self._linkage_conflict_strategy_ids:
                candidate_id = ""
                scan_features = {}
                match_method = "linkage_conflict"
                conflict_count += 1
            else:
                # Method 1: Direct linkage (deploy_linkage_log.csv)
                candidate_id, scan_features = self._match_via_linkage(
                    strategy_id, linkage_df, scanner_csvs
                )
                if candidate_id is not None:
                    match_method = "linkage"
                    linked_count += 1
                else:
                    # Method 2: Forensic timestamp matching
                    candidate_id, scan_features = self._match_via_forensic(
                        bot, scanner_csvs
                    )
                    if candidate_id is not None:
                        match_method = "forensic"
                        forensic_count += 1
                    else:
                        match_method = "unmatched"
                        candidate_id = ""
                        scan_features = {}
                        unmatched_count += 1

            row = self._build_outcome_row(
                bot, candidate_id, scan_features, match_method
            )
            rows.append(row)

        logger.info(
            "Ingested %d bots: %d linkage, %d forensic, %d unmatched, %d linkage conflicts",
            len(rows), linked_count, forensic_count, unmatched_count, conflict_count,
        )

        out = pd.DataFrame(rows)
        # Backward-compat schema stabilization: always expose known outcome fields.
        for field in _OUTCOME_FIELDS:
            if field not in out.columns:
                out[field] = np.nan
        return out

    # ── Internal methods ─────────────────────────────────────────────────

    def _load_bots(self) -> pd.DataFrame:
        """Load expired bots from Excel or CSV."""
        path = self._bots_path
        if not path.exists():
            logger.error("Expired bots file not found: %s", path)
            return pd.DataFrame()

        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        if "start_time_utc" in df.columns:
            df["start_time_utc"] = pd.to_datetime(df["start_time_utc"], utc=True)
        if "end_time_utc" in df.columns:
            df["end_time_utc"] = pd.to_datetime(df["end_time_utc"], utc=True)

        # Filter by date
        if self._min_date and "start_time_utc" in df.columns:
            cutoff = pd.Timestamp(self._min_date, tz="UTC")
            df = df[df["start_time_utc"] >= cutoff]

        logger.info("Loaded %d expired bots from %s", len(df), path.name)
        return cast(pd.DataFrame, df.reset_index(drop=True))

    def _load_linkage(self) -> pd.DataFrame:
        """Load deploy linkage log if it exists."""
        self._linkage_conflict_strategy_ids.clear()
        path = self._linkage_dir / "deploy_linkage_log.csv"
        if not path.exists():
            logger.info("No deploy linkage file found at %s", path)
            return pd.DataFrame()

        df = pd.read_csv(path)
        required = {"strategy_id", "candidate_id"}
        if not required.issubset(df.columns):
            logger.error(
                "Deploy linkage file is missing required columns %s: %s",
                sorted(required - set(df.columns)),
                path,
            )
            return pd.DataFrame()

        strategy = cast(pd.Series, df["strategy_id"]).fillna("").astype(str).str.strip()
        candidate = cast(pd.Series, df["candidate_id"]).fillna("").astype(str).str.strip()
        notes = (
            cast(pd.Series, df["notes"]).fillna("").astype(str).str.lower()
            if "notes" in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        basis = (
            cast(pd.Series, df["linkage_basis"]).fillna("").astype(str).str.lower()
            if "linkage_basis" in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        forensic_mask = notes.str.contains("match_method=geometry", regex=False) | basis.str.startswith(
            "forensic"
        )
        invalid_identity = (
            strategy.eq("")
            | strategy.str.lower().isin({"nan", "none", "null"})
            | candidate.eq("")
            | candidate.str.lower().isin({"nan", "none", "null"})
        )
        direct = cast(pd.DataFrame, df.loc[~forensic_mask & ~invalid_identity].copy())
        if direct.empty:
            logger.warning(
                "Loaded %d linkage records from %s, but none are direct identity links",
                len(df),
                path.name,
            )
            return direct

        direct_strategy = cast(pd.Series, direct["strategy_id"]).astype(str).str.strip()
        direct_candidate = cast(pd.Series, direct["candidate_id"]).astype(str).str.strip()
        pairs = pd.DataFrame(
            {"strategy_id": direct_strategy, "candidate_id": direct_candidate},
            index=direct.index,
        )
        candidate_counts = pairs.groupby("strategy_id")["candidate_id"].nunique()
        self._linkage_conflict_strategy_ids = set(
            candidate_counts.loc[candidate_counts > 1].index.astype(str)
        )
        if self._linkage_conflict_strategy_ids:
            direct = cast(
                pd.DataFrame,
                direct.loc[~direct_strategy.isin(self._linkage_conflict_strategy_ids)].copy(),
            )
            logger.error(
                "Rejected direct linkage for %d strategy_id conflict(s): %s",
                len(self._linkage_conflict_strategy_ids),
                sorted(self._linkage_conflict_strategy_ids),
            )

        logger.info(
            "Loaded %d governed direct linkage records from %s "
            "(excluded forensic=%d, invalid_identity=%d)",
            len(direct),
            path.name,
            int(forensic_mask.sum()),
            int(invalid_identity.sum()),
        )
        return direct

    def _parse_scanner_csvs(self) -> List[Tuple[datetime, Path]]:
        """Parse all deployment_ready CSVs with their timestamps."""
        pattern = re.compile(r"deployment_ready_(\d{8}_\d{6})\.csv$")
        entries: List[Tuple[datetime, Path]] = []

        if not self._scanner_dir.exists():
            logger.warning("Scanner results dir not found: %s", self._scanner_dir)
            return entries

        for f in sorted(self._scanner_dir.rglob("deployment_ready_*.csv")):
            m = pattern.search(f.name)
            if m:
                ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").replace(
                    tzinfo=timezone.utc
                )
                entries.append((ts, f))

        entries.sort(key=lambda x: x[0])
        logger.info("Found %d scanner CSVs", len(entries))
        return entries

    def _match_via_linkage(
        self,
        strategy_id: str,
        linkage_df: pd.DataFrame,
        scanner_csvs: List[Tuple[datetime, Path]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Try to match bot to candidate via deploy linkage log."""
        if linkage_df.empty or not strategy_id:
            return None, {}

        strategy_ids = cast(pd.Series, linkage_df["strategy_id"]).astype(str)
        match = linkage_df.loc[strategy_ids == strategy_id]
        if match.empty:
            return None, {}

        # deploy_linkage_log.csv is append-only. A strategy_id can reappear
        # after a re-deploy, so use the latest row just like MonitorContext.
        row = match.iloc[-1]
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id.lower() == "nan":
            return None, {}

        # Extract scan features from the corresponding scanner CSV
        scan_ts_str = str(row.get("scan_ts", ""))
        scan_features = self._extract_scan_features(
            str(row.get("symbol", "")),
            scan_ts_str,
            scanner_csvs,
            candidate_id=candidate_id,
        )

        # Also include config fields from the linkage itself as fallback
        for field in _SCAN_FEATURE_FIELDS:
            if field not in scan_features and field in row.index:
                val = row[field]
                if pd.notna(val):
                    scan_features[field] = val

        return candidate_id, scan_features

    @staticmethod
    def _grid_match_score(
        bot_low: float,
        bot_high: float,
        bot_grids: int,
        cand_low: float,
        cand_high: float,
        cand_grids: int,
    ) -> float:
        """Distance score between bot geometry and candidate geometry."""
        low_scale = max(abs(bot_low), 1e-9)
        high_scale = max(abs(bot_high), 1e-9)
        grids_scale = max(abs(bot_grids), 1)
        low_err = abs(cand_low - bot_low) / low_scale
        high_err = abs(cand_high - bot_high) / high_scale
        grids_err = abs(cand_grids - bot_grids) / grids_scale
        return float(low_err + high_err + 0.5 * grids_err)

    def _match_via_forensic(
        self,
        bot: Any,
        scanner_csvs: List[Tuple[datetime, Path]],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Forensic matching: closest preceding scanner CSV by time and
        closest candidate by grid geometry (lower/upper/num_grids).
        """
        symbol = str(bot.get("symbol", ""))
        bot_start = bot.get("start_time_utc")
        if not scanner_csvs:
            return None, {}

        try:
            bot_start_ts = pd.Timestamp(bot_start)
            if bot_start_ts.tzinfo is None:
                bot_start_ts = bot_start_ts.tz_localize("UTC")
        except Exception:
            bot_start_ts = None
        if bot_start_ts is None:
            return None, {}

        # Find CSVs that precede this bot
        preceding = [
            (ts, path) for ts, path in scanner_csvs if ts < bot_start_ts
        ]
        preceding.sort(key=lambda x: x[0], reverse=True)

        # Try a small rolling window of preceding scans.
        for _, csv_path in preceding[:5]:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            row_match = cast(pd.DataFrame, df[df["symbol"] == symbol].copy())
            if row_match.empty:
                continue

            # Keep only valid grid candidates.
            if "grid_is_valid" in row_match.columns:
                _gv = row_match["grid_is_valid"]
                if _gv.dtype == object:
                    row_match = cast(
                        pd.DataFrame,
                        row_match[_gv.astype(str).str.strip().str.lower() == "true"].copy(),
                    )
                else:
                    row_match = cast(pd.DataFrame, row_match[_gv == True].copy())  # noqa: E712
            if row_match.empty:
                continue

            # If geometry is available on both sides, match by grid boundaries/count.
            bot_low = cast(pd.Series, pd.to_numeric(pd.Series([bot.get("price_range_low")]), errors="coerce")).iloc[0]
            bot_high = cast(pd.Series, pd.to_numeric(pd.Series([bot.get("price_range_high")]), errors="coerce")).iloc[0]
            bot_grids = cast(pd.Series, pd.to_numeric(pd.Series([bot.get("grids_count")]), errors="coerce")).iloc[0]

            if (
                pd.notna(bot_low)
                and pd.notna(bot_high)
                and pd.notna(bot_grids)
                and {"grid_lower", "grid_upper", "num_grids"}.issubset(set(row_match.columns))
            ):
                _tmp = row_match.copy()
                _tmp["grid_lower"] = pd.to_numeric(_tmp["grid_lower"], errors="coerce")
                _tmp["grid_upper"] = pd.to_numeric(_tmp["grid_upper"], errors="coerce")
                _tmp["num_grids"] = pd.to_numeric(_tmp["num_grids"], errors="coerce")
                _tmp = cast(pd.DataFrame, _tmp.dropna(subset=["grid_lower", "grid_upper", "num_grids"]))
                if not _tmp.empty:
                    _tmp["grid_match_score"] = _tmp.apply(
                        lambda r: self._grid_match_score(
                            float(bot_low),
                            float(bot_high),
                            int(float(bot_grids)),
                            float(r["grid_lower"]),
                            float(r["grid_upper"]),
                            int(float(r["num_grids"])),
                        ),
                        axis=1,
                    )
                    _tmp = cast(pd.DataFrame, _tmp.sort_values("grid_match_score", ascending=True))
                    row = _tmp.iloc[0]
                    # If nearest geometry is too far, skip this CSV and keep searching.
                    if float(row["grid_match_score"]) > 0.08:
                        continue
                else:
                    row = row_match.iloc[0]
            else:
                row = row_match.iloc[0]

            # Build candidate_id from the scanner row
            m = _FILENAME_TS_RE.search(csv_path.stem)
            file_ts = m.group(1) if m else csv_path.stem

            # Use existing candidate_id if present, otherwise synthesise
            cid = row.get("candidate_id")
            if cid is None or (isinstance(cid, float) and np.isnan(cid)):
                cid = f"{symbol}_{file_ts}"

            normalized_row = normalize_inference_feature_dict(
                cast(Dict[str, Any], row.to_dict())
            )
            scan_features = {
                f: normalized_row[f] for f in _SCAN_FEATURE_FIELDS
                if f in normalized_row and pd.notna(normalized_row[f])
            }

            return str(cid), scan_features

        return None, {}

    def _extract_scan_features(
        self,
        symbol: str,
        scan_ts_str: str,
        scanner_csvs: List[Tuple[datetime, Path]],
        *,
        candidate_id: str = "",
    ) -> Dict[str, Any]:
        """Extract scan-time features from the matching scanner CSV."""
        if not scan_ts_str or not scanner_csvs:
            return {}

        # Find the CSV matching scan_ts
        try:
            target_ts = datetime.strptime(scan_ts_str, "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return {}

        for csv_ts, csv_path in scanner_csvs:
            if abs((csv_ts - target_ts).total_seconds()) < 60:
                try:
                    df = pd.read_csv(csv_path)
                    row: pd.DataFrame
                    if candidate_id and "candidate_id" in df.columns:
                        candidate_ids = cast(pd.Series, df["candidate_id"]).fillna("").astype(str)
                        row = cast(pd.DataFrame, df.loc[candidate_ids == candidate_id])
                        if row.empty:
                            continue
                    elif "symbol" in df.columns:
                        symbols = cast(pd.Series, df["symbol"]).fillna("").astype(str)
                        row = cast(pd.DataFrame, df.loc[symbols == symbol])
                    else:
                        row = pd.DataFrame()
                    if not row.empty:
                        r = row.iloc[0]
                        normalized_row = normalize_inference_feature_dict(
                            cast(Dict[str, Any], r.to_dict())
                        )
                        return {
                            f: normalized_row[f] for f in _SCAN_FEATURE_FIELDS
                            if f in normalized_row and pd.notna(normalized_row[f])
                        }
                except Exception:
                    pass

        return {}

    def _build_outcome_row(
        self,
        bot: Any,
        candidate_id: str,
        scan_features: Dict[str, Any],
        match_method: str,
    ) -> Dict[str, Any]:
        """Build a single outcome row combining bot data + scan features."""
        row: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "match_method": match_method,
            "source": "live",
            "sample_weight_override": 1.0,
        }

        # Copy outcome fields from bot
        for field in _OUTCOME_FIELDS:
            val = bot.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                # Keep timestamps as strings for clean DataFrame construction
                if field in ("start_time_utc", "end_time_utc") and hasattr(val, "isoformat"):
                    row[field] = str(val)
                else:
                    row[field] = val

        # Copy scan-time features (prefixed with scan_ for clarity)
        for field, val in scan_features.items():
            row[f"scan_{field}"] = val

        return row
