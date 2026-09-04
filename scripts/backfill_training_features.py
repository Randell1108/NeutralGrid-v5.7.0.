"""
Backfill historically computable training features.

This script backfills decision-time features without inventing proxy values for
utility, profit-per-grid, or funding. It reuses the same shared feature and
scoring logic used by the live pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import pandas as pd

# Ensure THIS repo's `src/` package is used when running this script directly
# (ERR-077: a stale editable install elsewhere must not shadow this tree).
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.candidate_id import extract_parts
from neutralgrid.core.config import get_config
from neutralgrid.data.binance_vision.downloader import download_kline_batch
from neutralgrid.data.binance_vision.ingest import ingest_zips
from neutralgrid.data.binance_vision.urls import daily_dates
from neutralgrid.models.hmm.inference import HMMRegimePredictor
from neutralgrid.scanner.feature_extractor import compute_features, klines_to_df
from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
from neutralgrid.training.data_generator import (
    ExistingDataMapper,
    HMM_FEATURE_SEMANTICS_VERSION,
)
from neutralgrid.validation.stochastic import StochasticConfig, StochasticRegimeChecker
from neutralgrid.validation.utility import (
    UtilityCalibratorUnavailable,
    compute_governed_provisional_utility,
)

# Module-level guard so the utility-unavailable warning is emitted once per
# backfill run; subsequent rows log at debug level to avoid flooding.
_utility_unavailable_warning_emitted = False

logger = logging.getLogger(__name__)

BASELINE_BACKFILL_COLUMNS = [
    "range_prob",
    "trend_prob",
    "utility_score",
    "survival_prob",
    "hurst_exponent",
    "ou_halflife",
    "profit_per_grid_pct",
    "range_size_pct",
    "funding_rate",
    "ev_score",
    "ev_contract_fingerprint",
    "primary_pipeline_score",
]

LIVE_PLUS_BACKFILL_COLUMNS_V20260312 = [
    "persistence_prob",
    "adx_1h",
    "adx_15m",
    "adx_5m",
    "rsi_15m",
    "ema_slope_1h",
    "ema_crosses_5m",
    "vwap_crosses_5m",
    "bb_width",
    "atr_pct_15m",
    "quote_volume_24h",
    "regime_conf",
]

HMM_LINEAGE_COLUMNS = [
    "hmm_trained_at_utc",
    "hmm_artifact_version",
    "hmm_pipeline_version",
    "hmm_feature_semantics_version",
    "hmm_feature_source",
    "hmm_replay_scope",
    "hmm_feature_cutoff_utc",
    "feature_cutoff_utc",
    "hmm_calibration_status",
]

TEXT_BACKFILL_COLUMNS = set(HMM_LINEAGE_COLUMNS) | {"ev_contract_fingerprint"}

HMM_DERIVED_COLUMNS = ["range_prob", "trend_prob", "persistence_prob"]

# These values either come directly from the HMM or are calculated from its
# probabilities.  A stale value must never survive a requested pinned replay
# when causal market data cannot be re-inferenced for that row.
HMM_DEPENDENT_BACKFILL_COLUMNS = (
    HMM_DERIVED_COLUMNS
    + HMM_LINEAGE_COLUMNS
    + ["utility_score", "ev_score", "regime_conf", "hmm_tail_cvar_95"]
)
FEATURE_CUTOFF_SOURCES = {"start_time_utc", "candidate_id_scan_time"}
REPLAY_SCOPES = {"full_feature_refresh", "hmm_lineage_only"}


def _to_timestamp(value: Any) -> Optional[datetime]:
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
        return parsed.to_pydatetime()
    return None


def _candidate_scan_timestamp(value: Any) -> Optional[datetime]:
    """Return the canonical UTC scan timestamp embedded in a candidate ID."""
    scan_ts = extract_parts(_non_empty_str(value)).get("scan_ts", "")
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(scan_ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _as_float(value: Any) -> float | None:
    numeric_series = cast(pd.Series, pd.to_numeric(pd.Series([value]), errors="coerce"))
    numeric = numeric_series.iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None or not np.isfinite(numeric):
        return None
    return int(round(numeric))


def _non_empty_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "<na>", "nat", "none"} else text


def _write_dataframe_atomic(df: pd.DataFrame, output_path: Path) -> None:
    """Publish a dataframe through a same-directory durable temporary file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_suffix = f".{uuid.uuid4().hex}.tmp{output_path.suffix}"
    temp_path = output_path.with_name(f".{output_path.stem}{temp_suffix}")
    try:
        if output_path.suffix.lower() in {".xlsx", ".xls"}:
            df.to_excel(temp_path, index=False)
            with temp_path.open("r+b") as handle:
                os.fsync(handle.fileno())
        else:
            with temp_path.open("w", encoding="utf-8", newline="") as handle:
                df.to_csv(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


class TrainingDataBackfiller:
    """Backfill historically computable training features."""

    def __init__(
        self,
        default_artifact_version: str = "",
        skip_if_fresh: bool = False,
        hmm_only: bool = False,
        max_concurrency: int = 1,
        require_fresh_output: bool = False,
        feature_cutoff_source: str = "start_time_utc",
        replay_scope: str = "full_feature_refresh",
    ) -> None:
        self.hmm_predictor = None
        self._hmm_predictors: Dict[str, Optional[HMMRegimePredictor]] = {}
        self.stochastic_checker = StochasticRegimeChecker(StochasticConfig())
        self.pnl_ranker = PnLRanker(RankingConfig())
        self.data_mapper = ExistingDataMapper()
        self.client: Optional[BinanceClient] = None
        self.default_artifact_version = _non_empty_str(default_artifact_version)
        self.skip_if_fresh = bool(skip_if_fresh)
        self.hmm_only = bool(hmm_only)
        if max_concurrency < 1 or max_concurrency > 8:
            raise ValueError("max_concurrency must be between 1 and 8")
        self.max_concurrency = int(max_concurrency)
        self.require_fresh_output = bool(require_fresh_output)
        normalized_cutoff_source = str(feature_cutoff_source).strip()
        if normalized_cutoff_source not in FEATURE_CUTOFF_SOURCES:
            raise ValueError(
                "feature_cutoff_source must be one of "
                f"{sorted(FEATURE_CUTOFF_SOURCES)}"
            )
        self.feature_cutoff_source = (
            "candidate_id_scan_time"
            if self.hmm_only and normalized_cutoff_source == "start_time_utc"
            else normalized_cutoff_source
        )
        normalized_replay_scope = str(replay_scope).strip()
        if normalized_replay_scope not in REPLAY_SCOPES:
            raise ValueError(
                f"replay_scope must be one of {sorted(REPLAY_SCOPES)}"
            )
        self.replay_scope = (
            "hmm_lineage_only" if self.hmm_only else normalized_replay_scope
        )

    def _resolve_feature_cutoff(
        self,
        row: pd.Series,
        event_start: datetime,
    ) -> datetime | None:
        if self.feature_cutoff_source == "start_time_utc":
            return event_start
        candidate_id = _non_empty_str(row.get("candidate_id"))
        # Legacy expired-bot records predate canonical candidate IDs.  Their
        # event start is the only recorded decision-time boundary, while a
        # malformed non-empty ID still fails closed below.
        if not candidate_id:
            return event_start
        cutoff = _candidate_scan_timestamp(candidate_id)
        if cutoff is None:
            return event_start if self.hmm_only else None
        if cutoff > event_start:
            return None
        return cutoff

    def _get_hmm_predictor(self, artifact_version: str | None) -> Optional[HMMRegimePredictor]:
        version = _non_empty_str(artifact_version)
        if not version:
            return None
        if version in self._hmm_predictors:
            return self._hmm_predictors[version]

        cfg = get_config()
        artifact_dir = cfg.resolve_path(cfg.artifacts.artifacts_dir) / "hmm" / version
        try:
            predictor = HMMRegimePredictor(artifact_dir)
            self._hmm_predictors[version] = predictor
            logger.info("Loaded HMM predictor for %s", version)
        except Exception as exc:
            logger.warning("Could not load HMM predictor for %s: %s", version, exc)
            self._hmm_predictors[version] = None
        return self._hmm_predictors[version]

    async def init_client(self) -> None:
        self.client = BinanceClient()

    async def close_client(self) -> None:
        if self.client is not None:
            await self.client.close()

    @staticmethod
    def _interval_delta(interval: str, bars: int) -> timedelta:
        if interval == "1h":
            return timedelta(hours=bars)
        if interval == "15m":
            return timedelta(minutes=15 * bars)
        if interval == "5m":
            return timedelta(minutes=5 * bars)
        if interval == "1m":
            return timedelta(minutes=bars)
        raise ValueError(f"Unsupported interval: {interval}")

    async def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        lookback_bars: int,
    ) -> pd.DataFrame:
        if self.client is None:
            raise RuntimeError("Client not initialized")

        fetch_start = start_time - self._interval_delta(interval, lookback_bars + 5)
        try:
            klines = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=int(fetch_start.timestamp() * 1000),
                end_time=int(start_time.timestamp() * 1000),
                limit=lookback_bars + 5,
            )
            rest_df = klines_to_df(klines)
        except Exception as exc:
            logger.warning(
                "REST kline fetch failed for %s %s; retaining archive fallback eligibility: %s",
                symbol,
                interval,
                exc,
            )
            rest_df = pd.DataFrame()

        completed_rest = self._select_completed_klines(
            rest_df,
            interval=interval,
            start_time=start_time,
            lookback_bars=lookback_bars,
        )
        if interval != "15m" or len(completed_rest) >= lookback_bars:
            return completed_rest

        vision_df = await self._fetch_binance_vision_klines(
            symbol,
            interval,
            start_time,
            lookback_bars,
        )
        if vision_df.empty:
            return completed_rest

        merged = (
            vision_df.copy()
            if completed_rest.empty
            else pd.concat([vision_df, completed_rest], ignore_index=True, sort=False)
        )
        completed = self._select_completed_klines(
            merged,
            interval=interval,
            start_time=start_time,
            lookback_bars=lookback_bars,
        )
        logger.info(
            "Historical 15m reconstruction for %s: REST=%d Vision+REST=%d required=%d",
            symbol,
            len(completed_rest),
            len(completed),
            lookback_bars,
        )
        return completed

    def _select_completed_klines(
        self,
        df: pd.DataFrame,
        *,
        interval: str,
        start_time: datetime,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Select unique bars fully closed before the decision timestamp."""
        if df.empty or "open_time" not in df.columns:
            return pd.DataFrame(columns=df.columns)

        prepared = df.copy()
        prepared["open_time"] = pd.to_datetime(
            prepared["open_time"], utc=True, errors="coerce"
        )
        inferred_close = prepared["open_time"] + self._interval_delta(interval, 1)
        if "close_time" in prepared.columns:
            reported_close = pd.to_datetime(
                prepared["close_time"], utc=True, errors="coerce"
            )
            effective_close = reported_close.fillna(inferred_close)
        else:
            effective_close = inferred_close

        decision_ts = pd.Timestamp(start_time)
        if decision_ts.tzinfo is None:
            decision_ts = decision_ts.tz_localize("UTC")
        else:
            decision_ts = decision_ts.tz_convert("UTC")
        completed_mask = prepared["open_time"].notna() & (effective_close <= decision_ts)
        prepared = cast(pd.DataFrame, prepared.loc[completed_mask])
        prepared = cast(
            pd.DataFrame,
            prepared.sort_values("open_time")
            .drop_duplicates(subset=["open_time"], keep="last")
            .tail(lookback_bars)
            .reset_index(drop=True),
        )
        return prepared

    async def _fetch_binance_vision_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        lookback_bars: int,
    ) -> pd.DataFrame:
        """Fetch checksum-verified daily archives for a short REST history."""
        fetch_start = start_time - self._interval_delta(interval, lookback_bars + 5)
        dates = daily_dates(fetch_start.date(), start_time.date())
        cfg = get_config()
        cache_dir = cfg.resolve_path(cfg.artifacts.cache_dir) / "klines"
        try:
            paths = await download_kline_batch(
                symbol=symbol,
                interval=interval,
                dates=dates,
                market="futures_um",
                granularity="daily",
                cache_dir=cache_dir,
            )
            if not paths:
                return pd.DataFrame()
            vision_df, report = await asyncio.to_thread(ingest_zips, paths)
            logger.info(
                "Loaded %d checksum-verified Vision rows for %s %s from %d archive(s); duplicates_removed=%s",
                len(vision_df),
                symbol,
                interval,
                len(paths),
                report.get("duplicates_removed", 0),
            )
            return vision_df
        except Exception as exc:
            logger.warning(
                "Binance Vision fallback failed for %s %s; row remains explicitly incomplete: %s",
                symbol,
                interval,
                exc,
            )
            return pd.DataFrame()

    async def fetch_historical_funding_rate(
        self,
        symbol: str,
        start_time: datetime,
    ) -> float | None:
        if self.client is None:
            raise RuntimeError("Client not initialized")

        window_start = start_time - timedelta(days=3)
        rows = await self.client.get_funding_rate(
            symbol=symbol,
            start_time=int(window_start.timestamp() * 1000),
            end_time=int(start_time.timestamp() * 1000),
            limit=1000,
        )
        if not rows:
            return None

        best_rate = None
        best_time = -1
        entry_ms = int(start_time.timestamp() * 1000)
        for row in rows:
            funding_time = int(row.get("fundingTime", 0) or 0)
            if funding_time <= 0 or funding_time > entry_ms:
                continue
            try:
                rate = float(row["fundingRate"])
            except (KeyError, TypeError, ValueError):
                continue
            if funding_time >= best_time:
                best_time = funding_time
                best_rate = rate
        return best_rate

    @staticmethod
    def _grid_lower(row: pd.Series) -> float | None:
        return _as_float(row.get("price_range_low")) or _as_float(row.get("grid_lower"))

    @staticmethod
    def _grid_upper(row: pd.Series) -> float | None:
        return _as_float(row.get("price_range_high")) or _as_float(row.get("grid_upper"))

    def _derive_profit_per_grid_pct(self, row: pd.Series) -> float | None:
        lower = self._grid_lower(row)
        upper = self._grid_upper(row)
        num_grids = _as_int(row.get("grids_count")) or _as_int(row.get("num_grids"))
        mode_val = row.get("mode")
        if lower is None or upper is None or num_grids is None or bool(pd.isna(mode_val)):
            return None
        return float(
            self.data_mapper.compute_profit_per_grid(
                grid_lower=lower,
                grid_upper=upper,
                num_grids=num_grids,
                mode=str(mode_val),
            )
        )

    def _derive_range_size_pct(
        self,
        row: pd.Series,
        market_range_size_pct: float | None,
    ) -> float | None:
        existing = _as_float(row.get("range_size_pct"))
        if existing is not None:
            return existing
        if market_range_size_pct is not None:
            return market_range_size_pct
        lower = self._grid_lower(row)
        upper = self._grid_upper(row)
        if lower is None or upper is None:
            return None
        reference_price = (lower + upper) / 2.0
        if reference_price <= 0:
            return None
        return float(((upper - lower) / float(reference_price)) * 100.0)

    def _derive_regime_conf(self, range_prob: float | None, trend_prob: float | None) -> float | None:
        if range_prob is None or trend_prob is None:
            return None
        dominance = range_prob / max(range_prob + trend_prob, 1e-10)
        cfg = get_config()
        soft_lower = float(cfg.hmm.soft_gate_lower)
        soft_upper = float(cfg.hmm.soft_gate_upper)
        if soft_upper <= soft_lower:
            return float(1.0 if dominance >= soft_upper else 0.0)
        return float(max(0.0, min(1.0, (dominance - soft_lower) / (soft_upper - soft_lower))))

    async def backfill_single_bot(self, row: pd.Series) -> Dict[str, Any]:
        symbol = str(row.get("symbol", "")).strip().upper()
        start_time = _to_timestamp(row.get("start_time_utc"))
        # Deployment snapshots are scanner-origin records.  They do not carry a
        # later bot/event start, but their canonical candidate ID records the
        # decision-time scan boundary.  Accept it only in the explicit scan-time
        # mode; malformed/noncanonical IDs still fail closed below.
        if start_time is None and self.feature_cutoff_source == "candidate_id_scan_time":
            start_time = _candidate_scan_timestamp(row.get("candidate_id"))
        if not symbol or start_time is None:
            invalid_result: Dict[str, Any] = {
                column: np.nan
                for column in BASELINE_BACKFILL_COLUMNS + LIVE_PLUS_BACKFILL_COLUMNS_V20260312 + HMM_LINEAGE_COLUMNS
            }
            if symbol and self.feature_cutoff_source == "candidate_id_scan_time":
                invalid_result["hmm_feature_source"] = "invalid_candidate_id_scan_time"
            return invalid_result

        feature_cutoff = self._resolve_feature_cutoff(row, start_time)
        if feature_cutoff is None:
            invalid_result: Dict[str, Any] = {
                column: np.nan
                for column in BASELINE_BACKFILL_COLUMNS
                + LIVE_PLUS_BACKFILL_COLUMNS_V20260312
                + HMM_LINEAGE_COLUMNS
            }
            invalid_result["hmm_feature_source"] = (
                "invalid_candidate_id_scan_time"
                if self.feature_cutoff_source == "candidate_id_scan_time"
                else "invalid_feature_cutoff"
            )
            return invalid_result

        if self.client is None:
            raise RuntimeError("Client not initialized")

        inference_time = feature_cutoff
        if self.replay_scope == "hmm_lineage_only":
            # A pinned-HMM refresh changes only the causal 15-minute regime
            # inference and fields derived from it.  Retaining all other
            # point-in-time features avoids silently substituting new market
            # data for an unchanged historical snapshot.  Candidate IDs are
            # minted at scan time; use that identity timestamp rather than
            # the later backtest start so the feature window is causal.
            df_15m = await self.fetch_historical_klines(
                symbol, "15m", inference_time, lookback_bars=800
            )
            df_1h = pd.DataFrame()
            df_5m = pd.DataFrame()
            df_1m = pd.DataFrame()
            funding_rate = _as_float(row.get("funding_rate"))
        else:
            df_1h, df_15m, df_5m, df_1m, funding_rate = await asyncio.gather(
                self.fetch_historical_klines(symbol, "1h", feature_cutoff, lookback_bars=250),
                self.fetch_historical_klines(symbol, "15m", feature_cutoff, lookback_bars=800),
                self.fetch_historical_klines(symbol, "5m", feature_cutoff, lookback_bars=350),
                self.fetch_historical_klines(symbol, "1m", feature_cutoff, lookback_bars=500),
                self.fetch_historical_funding_rate(symbol, feature_cutoff),
            )

        result: Dict[str, Any] = {
            column: np.nan
            for column in BASELINE_BACKFILL_COLUMNS + LIVE_PLUS_BACKFILL_COLUMNS_V20260312 + HMM_LINEAGE_COLUMNS
        }

        range_prob = None
        trend_prob = None
        row_artifact_version = _non_empty_str(row.get("hmm_artifact_version"))
        # UTILFIX-01: an explicit CLI default is the replay authority, including
        # for stale lineage carried by the INPUT file (not only values preserved
        # from a pre-existing output). Otherwise a fresh output path could still
        # select the old row-stamped artifact before the merge invalidation logic
        # has a chance to run.
        artifact_version = self.default_artifact_version or row_artifact_version
        predictor = self._get_hmm_predictor(artifact_version)
        if artifact_version:
            result["hmm_artifact_version"] = artifact_version
        result["hmm_feature_semantics_version"] = HMM_FEATURE_SEMANTICS_VERSION

        if predictor is not None:
            try:
                hmm = predictor.predict(df_15m)
                range_prob = float(hmm.range_prob_agg)
                trend_prob = float(hmm.trend_prob_agg)
                result["range_prob"] = range_prob
                result["trend_prob"] = trend_prob
                result["persistence_prob"] = float(hmm.persistence_prob)
                result["hmm_trained_at_utc"] = str(hmm.trained_at_utc)
                result["hmm_artifact_version"] = str(hmm.artifact_version)
                result["hmm_pipeline_version"] = str(hmm.pipeline_version)
                result["hmm_feature_source"] = "pinned_artifact_replay"
                result["hmm_replay_scope"] = self.replay_scope
                result["feature_cutoff_utc"] = feature_cutoff.isoformat()
                result["hmm_feature_cutoff_utc"] = inference_time.isoformat()
                if isinstance(hmm.calibration_provenance, dict):
                    status = hmm.calibration_provenance.get("status")
                    if status is not None:
                        result["hmm_calibration_status"] = str(status)
            except Exception as exc:
                logger.debug("HMM inference failed for %s: %s", symbol, exc)
                result["hmm_feature_source"] = "artifact_inference_failed"
        elif artifact_version:
            result["hmm_feature_source"] = "artifact_unavailable"
        else:
            result["hmm_feature_source"] = "missing_artifact_version"

        if self.replay_scope == "hmm_lineage_only":
            # An HMM-only replay intentionally retains the historical
            # decision-time feature snapshot.  Prefer its explicitly stored
            # grid return before falling back to geometry, which may not be
            # present in compact training rows.
            profit_per_grid_pct = _as_float(row.get("profit_per_grid_pct"))
            if profit_per_grid_pct is None:
                profit_per_grid_pct = self._derive_profit_per_grid_pct(row)
            range_size_pct = self._derive_range_size_pct(
                row=row,
                market_range_size_pct=_as_float(row.get("range_size_pct")),
            )
            survival_prob = _as_float(row.get("survival_prob"))
            num_grids = _as_int(row.get("grids_count")) or _as_int(row.get("num_grids"))
            if funding_rate is not None:
                result["funding_rate"] = float(funding_rate)
            if None not in (range_prob, trend_prob):
                try:
                    utility = compute_governed_provisional_utility(
                        range_prob=cast(float, range_prob),
                        trend_prob=cast(float, trend_prob),
                    )
                    result["utility_score"] = float(utility.utility_score)
                except UtilityCalibratorUnavailable:
                    result["utility_score"] = float("nan")
            if None not in (profit_per_grid_pct, num_grids, survival_prob, trend_prob, range_size_pct):
                leverage = _as_int(row.get("leverage")) or 10
                ev = self.pnl_ranker.compute_score(
                    profit_per_grid_pct=cast(float, profit_per_grid_pct),
                    num_grids=cast(int, num_grids),
                    survival_prob=cast(float, survival_prob),
                    trend_prob=cast(float, trend_prob),
                    funding_rate=funding_rate,
                    range_size_pct=cast(float, range_size_pct),
                    symbol=symbol,
                    leverage=leverage,
                )
                result["ev_score"] = float(ev.rank_score)
                result["ev_contract_fingerprint"] = self.pnl_ranker.ev_contract_fingerprint
            regime_conf = self._derive_regime_conf(range_prob, trend_prob)
            if regime_conf is not None:
                result["regime_conf"] = float(regime_conf)
            return result

        market_features = compute_features(
            symbol,
            klines_1h=df_1h,
            klines_15m=df_15m,
            klines_5m=df_5m,
            klines_1m=df_1m,
            funding_rate=funding_rate,
        )
        for column in LIVE_PLUS_BACKFILL_COLUMNS_V20260312:
            value = getattr(market_features, column, None)
            if value is not None and np.isfinite(float(value)):
                result[column] = float(value)

        lower = self._grid_lower(row)
        upper = self._grid_upper(row)
        if lower is not None and upper is not None and len(df_15m) >= 100:
            try:
                close_series = cast(pd.Series, df_15m["close"])
                numeric_closes = cast(
                    pd.Series, pd.to_numeric(close_series, errors="coerce")
                )
                closes = np.asarray(numeric_closes, dtype=float)
                log_prices = np.log(closes[np.isfinite(closes) & (closes > 0)])
                if log_prices.size >= 50:
                    stochastic = self.stochastic_checker.analyze(
                        log_prices=log_prices,
                        range_high_log=np.log(upper),
                        range_low_log=np.log(lower),
                    )
                    result["survival_prob"] = float(stochastic.survival_prob)
                    result["hurst_exponent"] = float(stochastic.hurst_exponent)
                    if np.isfinite(stochastic.ou_halflife):
                        result["ou_halflife"] = float(stochastic.ou_halflife)
            except Exception as exc:
                logger.debug("Stochastic analysis failed for %s: %s", symbol, exc)

        profit_per_grid_pct = self._derive_profit_per_grid_pct(row)
        if profit_per_grid_pct is not None:
            result["profit_per_grid_pct"] = float(profit_per_grid_pct)

        range_size_pct = self._derive_range_size_pct(
            row=row,
            market_range_size_pct=_as_float(result.get("range_size_pct")),
        )
        if range_size_pct is not None:
            result["range_size_pct"] = float(range_size_pct)

        if funding_rate is not None:
            result["funding_rate"] = float(funding_rate)

        num_grids = _as_int(row.get("grids_count")) or _as_int(row.get("num_grids"))
        survival_prob = _as_float(result.get("survival_prob"))
        if None not in (range_prob, trend_prob):
            try:
                utility = compute_governed_provisional_utility(
                    range_prob=cast(float, range_prob),
                    trend_prob=cast(float, trend_prob),
                )
                result["utility_score"] = float(utility.utility_score)
            except UtilityCalibratorUnavailable as exc:
                # Offline backfill: missing calibrator yields utility_score=NaN
                # (column is float-typed). No silent v0 substitution. Emit one
                # warning per run to avoid flooding on per-row iteration.
                global _utility_unavailable_warning_emitted
                if not _utility_unavailable_warning_emitted:
                    logger.warning(
                        "utility calibrator unavailable; utility_score=NaN for "
                        "remaining rows (first hit at %s): %s",
                        symbol,
                        exc,
                    )
                    _utility_unavailable_warning_emitted = True
                else:
                    logger.debug(
                        "utility calibrator unavailable for %s: %s", symbol, exc
                    )
                result["utility_score"] = float("nan")

        if None not in (
            profit_per_grid_pct,
            num_grids,
            survival_prob,
            trend_prob,
            range_size_pct,
        ):
            ev_profit_per_grid_pct = cast(float, profit_per_grid_pct)
            ev_num_grids = cast(int, num_grids)
            ev_survival_prob = cast(float, survival_prob)
            ev_trend_prob = cast(float, trend_prob)
            ev_range_size_pct = cast(float, range_size_pct)
            leverage = _as_int(row.get("leverage"))
            if leverage is None:
                leverage = 10
            ev = self.pnl_ranker.compute_score(
                profit_per_grid_pct=ev_profit_per_grid_pct,
                num_grids=ev_num_grids,
                survival_prob=ev_survival_prob,
                trend_prob=ev_trend_prob,
                funding_rate=funding_rate,
                range_size_pct=ev_range_size_pct,
                symbol=symbol,
                leverage=leverage,
            )
            # Match the canonical serve-time/unified-builder contract:
            # ev_score is PnLRanker.rank_score (penalized aligned EV), while
            # ev_24h is a distinct diagnostic and must not be written here.
            result["ev_score"] = float(ev.rank_score)
            result["ev_contract_fingerprint"] = (
                self.pnl_ranker.ev_contract_fingerprint
            )

        regime_conf = self._derive_regime_conf(range_prob, trend_prob)
        if regime_conf is not None:
            result["regime_conf"] = float(regime_conf)

        existing_primary = _as_float(row.get("primary_pipeline_score"))
        if existing_primary is None:
            existing_primary = _as_float(row.get("score"))
        if existing_primary is not None:
            result["primary_pipeline_score"] = float(existing_primary)

        return result

    async def backfill_all(self, input_file: str, output_file: str) -> None:
        input_path = Path(input_file)
        output_path = Path(output_file)

        if self.require_fresh_output and output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing backfill output: {output_path}"
            )

        if input_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)

        # UTILFIX-02: apply the explicit replay authority to the input frame
        # itself, not only to values preserved from an existing output.  Without
        # this invalidation a failed re-inference can leave old range/trend/EV
        # values in place while the result is labelled with the new artifact.
        if self.default_artifact_version and "hmm_artifact_version" in df.columns:
            input_versions = (
                cast(pd.Series, df["hmm_artifact_version"])
                .fillna("")
                .astype(str)
                .str.strip()
            )
            stale_input_mask = cast(
                pd.Series,
                input_versions.ne(self.default_artifact_version),
            )
            stale_input_count = int(stale_input_mask.sum())
            if stale_input_count > 0:
                for column in HMM_DEPENDENT_BACKFILL_COLUMNS:
                    if column not in df.columns:
                        continue
                    if column in TEXT_BACKFILL_COLUMNS:
                        df.loc[stale_input_mask, column] = pd.NA
                    else:
                        df.loc[stale_input_mask, column] = np.nan
                logger.info(
                    "Invalidated stale input HMM-derived values for %d row(s) "
                    "(requested=%s); re-inference is required.",
                    stale_input_count,
                    self.default_artifact_version,
                )

        # Merge with existing backfilled file to preserve previously computed
        # features for delisted or temporarily unavailable symbols.
        all_backfill_cols = (
            BASELINE_BACKFILL_COLUMNS
            + LIVE_PLUS_BACKFILL_COLUMNS_V20260312
            + HMM_LINEAGE_COLUMNS
        )
        if output_path.exists() and output_path != input_path:
            try:
                if output_path.suffix.lower() in {".xlsx", ".xls"}:
                    existing = pd.read_excel(output_path)
                else:
                    existing = pd.read_csv(output_path)
                preserved = 0
                required_keys = ["symbol", "start_time_utc"]
                if all(key in df.columns for key in required_keys) and all(
                    key in existing.columns for key in required_keys
                ):
                    current = df.copy()
                    current["_merge_symbol"] = (
                        current["symbol"].fillna("").astype(str).str.strip().str.upper()
                    )
                    current["_merge_start_time_utc"] = pd.to_datetime(
                        current["start_time_utc"], utc=True, errors="coerce", format="mixed"
                    )
                    existing = existing.copy()
                    existing["_merge_symbol"] = (
                        existing["symbol"].fillna("").astype(str).str.strip().str.upper()
                    )
                    existing["_merge_start_time_utc"] = pd.to_datetime(
                        existing["start_time_utc"], utc=True, errors="coerce", format="mixed"
                    )
                    temp_merge_cols = ["_merge_symbol", "_merge_start_time_utc"]
                    merge_keys = ["_merge_symbol", "_merge_start_time_utc"]
                    for optional_key in ("candidate_id", "strategy_id"):
                        if optional_key in current.columns and optional_key in existing.columns:
                            current_key = f"_merge_{optional_key}"
                            existing_key = f"_merge_{optional_key}"
                            temp_merge_cols.append(current_key)
                            current[current_key] = (
                                current[optional_key].fillna("").astype(str).str.strip()
                            )
                            existing[existing_key] = (
                                existing[optional_key].fillna("").astype(str).str.strip()
                            )
                            if bool(current[current_key].ne("").any()) or bool(
                                existing[existing_key].ne("").any()
                            ):
                                merge_keys.append(current_key)
                    existing = cast(
                        pd.DataFrame,
                        existing.loc[
                            existing["_merge_symbol"].ne("")
                            & existing["_merge_start_time_utc"].notna()
                        ]
                        .sort_values(merge_keys)
                        .drop_duplicates(subset=merge_keys, keep="last"),
                    )
                    preserve_cols = [
                        col for col in all_backfill_cols if col in existing.columns
                    ]
                    existing_subset = cast(pd.DataFrame, existing[merge_keys + preserve_cols])
                    existing_preserve = cast(
                        pd.DataFrame,
                        existing_subset.rename(
                            columns={col: f"existing__{col}" for col in preserve_cols}
                        ),
                    )
                    merged = cast(
                        pd.DataFrame,
                        current.merge(
                            existing_preserve,
                            on=merge_keys,
                            how="left",
                        ),
                    )
                    existing_version_normalized = (
                        cast(
                            pd.Series,
                            merged.get(
                                "existing__hmm_artifact_version",
                                pd.Series("", index=merged.index),
                            ),
                        )
                        .fillna("")
                        .astype(str)
                        .str.strip()
                    )
                    has_hmm_lineage = cast(
                        pd.Series, existing_version_normalized.ne("")
                    )
                    # UTILFIX-01: when --default-artifact-version is set, it is
                    # AUTHORITATIVE. A preserved per-row hmm_artifact_version that
                    # differs from the explicit default must NOT contaminate the
                    # new run. Invalidate HMM_DERIVED + HMM_LINEAGE preservation
                    # for stale rows so backfill_single_bot re-attempts inference
                    # against the explicit default.
                    explicit_default = self.default_artifact_version
                    if explicit_default:
                        existing_lineage_matches_default = cast(
                            pd.Series,
                            existing_version_normalized.eq(explicit_default),
                        )
                        stale_lineage_count = int(
                            (has_hmm_lineage & ~existing_lineage_matches_default).sum()
                        )
                        if stale_lineage_count > 0:
                            stale_versions = sorted(
                                set(
                                    existing_version_normalized.loc[
                                        has_hmm_lineage & ~existing_lineage_matches_default
                                    ].tolist()
                                )
                            )
                            logger.info(
                                "Merge invalidated %d row(s) with stale HMM lineage "
                                "(preserved=%s vs requested=%s); re-inference will "
                                "run for those rows.",
                                stale_lineage_count,
                                stale_versions,
                                explicit_default,
                            )
                    else:
                        # Legacy path: no explicit default. The lineage-match
                        # mask is unused below (gated on `if explicit_default`),
                        # so we don't compute it. Initialised to a typed empty
                        # Series for static-analysis transparency only.
                        existing_lineage_matches_default = cast(
                            pd.Series,
                            pd.Series(False, index=merged.index, dtype=bool),
                        )
                    invalidated_columns = set(HMM_DEPENDENT_BACKFILL_COLUMNS)
                    for column in preserve_cols:
                        existing_col = f"existing__{column}"
                        if column not in merged.columns:
                            if pd.api.types.is_numeric_dtype(merged[existing_col]):
                                merged[column] = np.nan
                            else:
                                merged[column] = pd.Series(
                                    pd.NA,
                                    index=merged.index,
                                    dtype="object",
                                )
                        fill_mask = cast(
                            pd.Series,
                            merged[column].isna() & merged[existing_col].notna(),
                        )
                        if column in HMM_DERIVED_COLUMNS:
                            fill_mask = cast(pd.Series, fill_mask & has_hmm_lineage)
                        if explicit_default and column in invalidated_columns:
                            fill_mask = cast(
                                pd.Series,
                                fill_mask & existing_lineage_matches_default,
                            )
                        if fill_mask.any():
                            merged.loc[fill_mask, column] = merged.loc[fill_mask, existing_col]
                            preserved += int(fill_mask.sum())
                        merged = cast(pd.DataFrame, merged.drop(columns=[existing_col]))
                    merged = cast(
                        pd.DataFrame,
                        merged.drop(
                            columns=[
                                key
                                for key in temp_merge_cols
                                if key in merged.columns
                            ],
                        ),
                    )
                    df = merged
                if preserved > 0:
                    logger.info(
                        "Preserved %d previously backfilled values from %s",
                        preserved, output_path.name,
                    )
            except Exception as exc:
                logger.warning("Could not merge existing backfill: %s", exc)

        await self.init_client()
        try:
            for column in all_backfill_cols:
                if column not in df.columns:
                    if column in TEXT_BACKFILL_COLUMNS:
                        df[column] = pd.Series(pd.NA, index=df.index, dtype="object")
                    else:
                        df[column] = np.nan
                elif column in TEXT_BACKFILL_COLUMNS and not pd.api.types.is_object_dtype(df[column]):
                    df[column] = df[column].astype("object")

            success_count = 0
            skipped_fresh = 0
            pending: list[tuple[Any, pd.Series]] = []
            for idx, row in df.iterrows():
                if self.skip_if_fresh and self.default_artifact_version:
                    preserved_version = row.get("hmm_artifact_version")
                    preserved_version_str = (
                        str(preserved_version).strip()
                        if preserved_version is not None and pd.notna(preserved_version)
                        else ""
                    )
                    if preserved_version_str == self.default_artifact_version:
                        derived = [row.get(col) for col in HMM_DERIVED_COLUMNS]
                        if all(
                            v is not None
                            and pd.notna(v)
                            and isinstance(v, (int, float, np.integer, np.floating))
                            and np.isfinite(float(v))
                            for v in derived
                        ):
                            skipped_fresh += 1
                            continue
                pending.append((idx, row.copy()))

            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _run_one(
                idx: Any,
                row: pd.Series,
            ) -> tuple[Any, Dict[str, Any]]:
                async with semaphore:
                    try:
                        features = await self.backfill_single_bot(row)
                        return idx, features
                    except Exception as exc:
                        logger.error(
                            "Backfill failed for row %s (%s): %s",
                            idx,
                            row.get("symbol"),
                            exc,
                        )
                        return idx, {}
                    finally:
                        await asyncio.sleep(0.1)

            results = await asyncio.gather(
                *(_run_one(idx, row) for idx, row in pending)
            )
            for idx, features in results:
                for column, value in features.items():
                    if value is not None and not pd.isna(value):
                        df.at[idx, column] = value
                if any(
                    value is not None and not pd.isna(value)
                    for value in features.values()
                ):
                    success_count += 1
        finally:
            try:
                await self.close_client()
            except Exception as exc:
                logger.error("Failed to close Binance client cleanly: %s", exc)

        if self.require_fresh_output and output_path.exists():
            raise FileExistsError(
                f"Refusing to replace output created during replay: {output_path}"
            )
        _write_dataframe_atomic(df, output_path)

        logger.info("Backfill complete: %s", output_path)
        logger.info(
            "Rows with at least one computed feature: %d / %d (%d skipped: fresh lineage)",
            success_count,
            len(df),
            skipped_fresh,
        )
        for column in BASELINE_BACKFILL_COLUMNS + LIVE_PLUS_BACKFILL_COLUMNS_V20260312:
            series = cast(pd.Series, df[column])
            coverage = float(series.notna().mean()) if len(df) else 0.0
            logger.info("%s coverage: %.1f%%", column, coverage * 100.0)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historically computable training features into the expired-bots workbook.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/new_expired_bots.xlsx"),
        help="Path to the source workbook (default: data/new_expired_bots.xlsx).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/new_expired_bots_backfilled.xlsx"),
        help="Path for the backfilled workbook (default: data/new_expired_bots_backfilled.xlsx).",
    )
    parser.add_argument(
        "--default-artifact-version",
        type=str,
        default="",
        help=(
            "Pinned HMM artifact version. When set, this is AUTHORITATIVE: "
            "rows whose preserved (existing-output) hmm_artifact_version "
            "differs from this value have their preserved HMM-derived "
            "features and lineage invalidated, forcing re-inference against "
            "the explicit version. Rows where klines are unavailable (e.g. "
            "delisted symbols) end up with hmm_feature_source="
            "'artifact_unavailable' rather than silently retaining a stale "
            "lineage. Empty (default) preserves prior per-row lineage with "
            "no invalidation."
        ),
    )
    parser.add_argument(
        "--skip-if-fresh",
        action="store_true",
        help=(
            "Skip per-row HMM inference when the post-merge row already has "
            "hmm_artifact_version == --default-artifact-version AND finite "
            "range_prob / trend_prob / persistence_prob. Stale-lineage rows "
            "are still re-inferenced because the merge clears their HMM "
            "columns. No-op when --default-artifact-version is empty."
        ),
    )
    parser.add_argument(
        "--hmm-only",
        action="store_true",
        help=(
            "Refresh only active-HMM-derived lineage, probabilities, utility, EV, and regime confidence. "
            "Uses the stored ex-ante non-HMM feature snapshot and only fetches the causal 15m HMM window."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Bounded concurrent row replays (1-8; default 1).",
    )
    parser.add_argument(
        "--require-fresh-output",
        action="store_true",
        help="Refuse to read, merge, or replace an existing output path.",
    )
    parser.add_argument(
        "--feature-cutoff-source",
        choices=sorted(FEATURE_CUTOFF_SOURCES),
        default="start_time_utc",
        help=(
            "Timestamp governing all point-in-time feature fetches. Use "
            "candidate_id_scan_time for scanner-origin backtest rows; the "
            "default start_time_utc preserves expired-bot compatibility."
        ),
    )
    parser.add_argument(
        "--replay-scope",
        choices=sorted(REPLAY_SCOPES),
        default="full_feature_refresh",
        help=(
            "full_feature_refresh recomputes all historically available "
            "market features. hmm_lineage_only fetches only 15m HMM context, "
            "preserves independent scanner-snapshot features, and recomputes "
            "HMM-transitive EV fields."
        ),
    )
    return parser.parse_args(argv)


async def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args(argv)
    backfiller = TrainingDataBackfiller(
        default_artifact_version=args.default_artifact_version,
        skip_if_fresh=args.skip_if_fresh,
        hmm_only=args.hmm_only,
        max_concurrency=args.max_concurrency,
        require_fresh_output=args.require_fresh_output,
        feature_cutoff_source=args.feature_cutoff_source,
        replay_scope=args.replay_scope,
    )
    await backfiller.backfill_all(
        str(args.input),
        str(args.output),
    )


if __name__ == "__main__":
    asyncio.run(main())
