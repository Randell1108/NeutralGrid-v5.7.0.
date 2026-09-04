"""
Scanner Integration for Training Data Collection.

Provides utilities to log feature snapshots from the scanner
for future meta-labeler training.

Usage in scanner/scan.py:
    from neutralgrid.training.scanner_integration import (
        FeatureCollector,
        build_feature_snapshot,
    )

    # Create collector at scanner startup
    collector = FeatureCollector(log_dir="data/training_snapshots")

    # In fetch_one(), after computing features:
    snapshot = build_feature_snapshot(
        symbol=symbol,
        scanner_row=row,
        validation_result=validation_result,  # If available
    )
    collector.log(snapshot)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import math

from neutralgrid.grid.formulas import ARITHMETIC, GEOMETRIC, grid_spacing_pct
from neutralgrid.training.data_generator import (
    FeatureSnapshot,
    FeatureSnapshotLogger,
    HMM_FEATURE_SEMANTICS_VERSION,
)
from neutralgrid.models.meta_labeler import normalize_inference_feature_dict

logger = logging.getLogger(__name__)
_VALID_GRID_MODES = {ARITHMETIC, GEOMETRIC}


def build_feature_snapshot(
    symbol: str,
    scanner_row: Dict[str, Any],
    validation_result: Optional[Any] = None,
    timestamp: Optional[datetime] = None,
) -> FeatureSnapshot:
    """
    Build a FeatureSnapshot from scanner output and validation result.

    This creates a complete feature snapshot suitable for logging
    and future training use.

    Args:
        symbol: Trading pair symbol
        scanner_row: Dictionary of features from scanner
        validation_result: ValidationResult from regime_validator (optional)
        timestamp: Override timestamp (default: now)

    Returns:
        FeatureSnapshot with all available features
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    scanner_row = normalize_inference_feature_dict(scanner_row)

    # Helper to get first non-null value from multiple possible field names
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(isinstance(value, float) and math.isnan(value))
        except TypeError:
            return False

    def get_first(*keys):
        for key in keys:
            val = scanner_row.get(key)
            if not _is_missing(val):
                return val
        return None

    def _derive_grid_spacing_pct() -> Optional[float]:
        explicit = _safe_float(get_first("grid_spacing_pct", "scan_grid_spacing_pct"))
        if explicit is not None:
            return explicit
        lower = _safe_float(get_first("grid_lower", "range_low"))
        upper = _safe_float(get_first("grid_upper", "range_high"))
        grids = _safe_int(scanner_row.get("num_grids"))
        raw_mode = get_first("mode", "backtest_mode")
        mode = str(raw_mode).strip().lower() if raw_mode is not None else ""
        if lower is None or upper is None or grids is None:
            return None
        if mode not in _VALID_GRID_MODES:
            logger.warning(
                "Cannot derive grid_spacing_pct for %s without explicit valid mode",
                symbol,
            )
            return None
        if lower <= 0 or upper <= lower or grids < 2:
            return None
        return float(grid_spacing_pct(lower, upper, grids, mode))

    # Start with scanner row values - handle multiple field name conventions
    snapshot = FeatureSnapshot(
        symbol=symbol,
        start_time_utc=timestamp,
        candidate_id=scanner_row.get("candidate_id"),
        # HMM/Regime features are canonicalized before snapshot construction.
        range_prob=_safe_float(get_first("range_prob", "hmm_range_prob")),
        trend_prob=_safe_float(get_first("trend_prob", "hmm_trend_prob")),
        hmm_trained_at_utc=(
            None if get_first("hmm_trained_at_utc") is None else str(get_first("hmm_trained_at_utc"))
        ),
        hmm_artifact_version=(
            None if get_first("hmm_artifact_version") is None else str(get_first("hmm_artifact_version"))
        ),
        hmm_pipeline_version=(
            None if get_first("hmm_pipeline_version") is None else str(get_first("hmm_pipeline_version"))
        ),
        hmm_feature_semantics_version=(
            str(get_first("hmm_feature_semantics_version") or HMM_FEATURE_SEMANTICS_VERSION)
        ),
        # Gate 5 (Fix 5a): Default hmm_feature_source to "live_scan"
        # rather than leaving it as None (silent lineage gap).
        hmm_feature_source=str(
            get_first("hmm_feature_source") or "live_scan"
        ),
        hmm_calibration_status=(
            None if get_first("hmm_calibration_status") is None else str(get_first("hmm_calibration_status"))
        ),
        # Stochastic features
        survival_prob=_safe_float(get_first("survival_prob", "stochastic_survival_prob")),
        hurst_exponent=_safe_float(scanner_row.get("hurst_exponent")),
        ou_halflife=_safe_float(scanner_row.get("ou_halflife")),
        # Grid features (computed when AFML scoring is enabled)
        profit_per_grid_pct=_safe_float(scanner_row.get("profit_per_grid_pct")),
        num_grids=_safe_int(scanner_row.get("num_grids")),
        grid_spacing_pct=_derive_grid_spacing_pct(),
        range_size_pct=_safe_float(get_first("range_size_pct", "scan_range_size_pct")),
        # Technical indicators - handle both naming conventions
        adx_1h=_safe_float(scanner_row.get("adx_1h")),
        adx_15m=_safe_float(scanner_row.get("adx_15m")),
        rsi_15m=_safe_float(scanner_row.get("rsi_15m")),
        adx_5m=_safe_float(scanner_row.get("adx_5m")),
        ema_slope_1h=_safe_float(scanner_row.get("ema_slope_1h")),
        ema_crosses_5m=_safe_int(scanner_row.get("ema_crosses_5m")),
        vwap_crosses_5m=_safe_int(scanner_row.get("vwap_crosses_5m")),
        bb_width=_safe_float(scanner_row.get("bb_width")),
        atr_pct_15m=_safe_float(scanner_row.get("atr_pct_15m")),
        quote_volume_24h=_safe_float(scanner_row.get("quote_volume_24h")),
        open_interest=_safe_float(scanner_row.get("open_interest")),
        regime_conf=_safe_float(scanner_row.get("regime_conf")),
        # Market features
        funding_rate=_safe_float(scanner_row.get("funding_rate")),
        micro_round_trip_cost_pct=_safe_float(scanner_row.get("micro_round_trip_cost_pct")),
        # afml_enriched_v20260318 additions
        hmm_tail_cvar_95=_safe_float(scanner_row.get("hmm_tail_cvar_95")),
        tos_gcf=_safe_float(scanner_row.get("tos_gcf")),
        persistence_prob=_safe_float(scanner_row.get("persistence_prob")),
        long_short_ratio=_safe_float(scanner_row.get("long_short_ratio")),
        funding_rate_zscore=_safe_float(scanner_row.get("funding_rate_zscore")),
        open_interest_change_pct=_safe_float(scanner_row.get("open_interest_change_pct")),
        bb_width_ratio_1h_15m=_safe_float(scanner_row.get("bb_width_ratio_1h_15m")),
        micro_osc_score=_safe_float(scanner_row.get("micro_osc_score")),
        # Binance-derived crowding / flow context
        top_account_lsr_log=_safe_float(scanner_row.get("top_account_lsr_log")),
        top_position_lsr_log=_safe_float(scanner_row.get("top_position_lsr_log")),
        top_position_vs_account_delta=_safe_float(scanner_row.get("top_position_vs_account_delta")),
        global_lsr_log=_safe_float(scanner_row.get("global_lsr_log")),
        taker_imbalance=_safe_float(scanner_row.get("taker_imbalance")),
        taker_buy_sell_log=_safe_float(scanner_row.get("taker_buy_sell_log")),
        basis_pct=_safe_float(scanner_row.get("basis_pct")),
        # Binance-derived execution / liquidity context
        spread_pct=_safe_float(scanner_row.get("spread_pct")),
        top20_bid_depth_usdt=_safe_float(scanner_row.get("top20_bid_depth_usdt")),
        top20_ask_depth_usdt=_safe_float(scanner_row.get("top20_ask_depth_usdt")),
        top20_depth_min_usdt=_safe_float(scanner_row.get("top20_depth_min_usdt")),
        book_imbalance_top20=_safe_float(scanner_row.get("book_imbalance_top20")),
        oi_notional=_safe_float(scanner_row.get("oi_notional")),
        oi_to_volume=_safe_float(scanner_row.get("oi_to_volume")),
        # Pattern-profile inputs remain provenance-only until a challenger
        # passes an untouched exact-target promotion gate.
        parkinson_vol_ratio_4h_24h_pre=_safe_float(
            scanner_row.get("parkinson_vol_ratio_4h_24h_pre")
        ),
        variance_ratio_1m_15m_pre_2h=_safe_float(
            scanner_row.get("variance_ratio_1m_15m_pre_2h")
        ),
        funding_carry_expected_next_7h=_safe_float(
            scanner_row.get("funding_carry_expected_next_7h")
        ),
        liquidity_stability_z_1h=_safe_float(
            scanner_row.get("liquidity_stability_z_1h")
        ),
        ev_score=_safe_float(get_first("ev_score")),
        ev_contract_fingerprint=(
            str(scanner_row["ev_contract_fingerprint"]).strip()
            if scanner_row.get("ev_contract_fingerprint") is not None
            and str(scanner_row["ev_contract_fingerprint"]).strip()
            and str(scanner_row["ev_contract_fingerprint"]).strip().lower()
            not in {"nan", "none", "<na>"}
            else None
        ),
        primary_pipeline_score=_safe_float(get_first("primary_pipeline_score", "scan_score", "score")),
        # Range bounds for label computation
        range_high=_safe_float(get_first("range_high", "grid_upper")),
        range_low=_safe_float(get_first("range_low", "grid_lower")),
        entry_price=_safe_float(get_first("last_price", "current_price", "entry_price")),
    )

    # Override with ValidationResult if available (more authoritative)
    if validation_result is not None:
        if hasattr(validation_result, "range_prob") and validation_result.range_prob is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "range_prob": validation_result.range_prob}
            )
        if hasattr(validation_result, "trend_prob") and validation_result.trend_prob is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "trend_prob": validation_result.trend_prob}
            )
        if hasattr(validation_result, "survival_prob") and validation_result.survival_prob is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "survival_prob": validation_result.survival_prob}
            )
        if hasattr(validation_result, "hurst_exponent") and validation_result.hurst_exponent is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "hurst_exponent": validation_result.hurst_exponent}
            )
        if hasattr(validation_result, "ou_halflife") and validation_result.ou_halflife is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "ou_halflife": validation_result.ou_halflife}
            )
        if hasattr(validation_result, "range_high") and validation_result.range_high is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "range_high": validation_result.range_high}
            )
        if hasattr(validation_result, "range_low") and validation_result.range_low is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "range_low": validation_result.range_low}
            )
        if hasattr(validation_result, "current_price") and validation_result.current_price is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(),
                   "entry_price": validation_result.current_price}
            )
        if hasattr(validation_result, "hmm_artifact_version") and validation_result.hmm_artifact_version is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(), "hmm_artifact_version": validation_result.hmm_artifact_version}
            )
        if hasattr(validation_result, "hmm_pipeline_version") and validation_result.hmm_pipeline_version is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(), "hmm_pipeline_version": validation_result.hmm_pipeline_version}
            )
        # Read HMM lineage from top-level fields first, fallback to tf_1h.checks
        trained_at = getattr(validation_result, "hmm_trained_at_utc", None)
        cal_prov = getattr(validation_result, "hmm_calibration_provenance", None)
        if trained_at is None and hasattr(validation_result, "tf_1h") and getattr(validation_result, "tf_1h") is not None:
            checks = getattr(validation_result.tf_1h, "checks", {})
            if isinstance(checks, dict):
                trained_at = checks.get("trained_at_utc")
                if cal_prov is None:
                    cal_prov = checks.get("calibration_provenance")
        if trained_at is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(), "hmm_trained_at_utc": str(trained_at)}
            )
        if isinstance(cal_prov, dict) and cal_prov.get("status") is not None:
            snapshot = FeatureSnapshot(
                **{**snapshot.to_dict(), "hmm_calibration_status": str(cal_prov.get("status"))}
            )

    return snapshot


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert to float, returning None if not possible."""
    if value is None:
        return None
    try:
        f = float(value)
        if not math.isfinite(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Safely convert to int, returning None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


class FeatureCollector:
    """
    High-level API for collecting features during scanning.

    Wraps FeatureSnapshotLogger with additional tracking and validation.

    Usage:
        collector = FeatureCollector(log_dir="data/training_snapshots")

        # During scanning:
        snapshot = build_feature_snapshot(symbol, row, validation_result)
        collector.log(snapshot)

        # At end of session:
        collector.close()
        print(f"Logged {collector.stats['total_logged']} snapshots")
    """

    def __init__(
        self,
        log_dir: str | Path = "data/training_snapshots",
        enabled: bool = True,
        validate_snapshots: bool = True,
    ):
        """
        Initialize collector.

        Args:
            log_dir: Directory to store snapshot parquet files
            enabled: Whether to actually log (set False to disable)
            validate_snapshots: Whether to validate snapshots before logging
        """
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.validate_snapshots = validate_snapshots

        self._logger: Optional[FeatureSnapshotLogger] = None
        self._stats = {
            "total_logged": 0,
            "total_skipped": 0,
            "validation_failures": 0,
        }

        if self.enabled:
            self._logger = FeatureSnapshotLogger(self.log_dir)
            logger.info(f"FeatureCollector initialized, logging to {self.log_dir}")

    @property
    def stats(self) -> Dict[str, int]:
        """Get collection statistics."""
        return self._stats.copy()

    def log(
        self,
        snapshot: FeatureSnapshot,
        min_features_required: Optional[int] = None,
    ) -> bool:
        """
        Log a feature snapshot.

        Args:
            snapshot: FeatureSnapshot to log
            min_features_required: Minimum non-null active-profile features to log.
                Defaults to the full active profile size.

        Returns:
            True if logged successfully, False if skipped
        """
        if not self.enabled or self._logger is None:
            return False

        from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
        if min_features_required is None:
            min_features_required = len(ACTIVE_SNAPSHOT_META_FEATURES)

        # Validate if enabled
        if self.validate_snapshots:
            issues = self._validate_snapshot(snapshot, min_features_required)
            if issues:
                self._stats["validation_failures"] += 1
                logger.debug(
                    "Skipping snapshot for %s: %s",
                    snapshot.symbol,
                    "; ".join(issues),
                )
                self._stats["total_skipped"] += 1
                return False

        # Log the snapshot
        try:
            self._logger.log_snapshot(snapshot)
            self._stats["total_logged"] += 1
            return True
        except Exception as e:
            logger.warning("Failed to log snapshot for %s: %s", snapshot.symbol, e)
            self._stats["total_skipped"] += 1
            return False

    def _validate_snapshot(
        self,
        snapshot: FeatureSnapshot,
        min_features: int,
    ) -> list[str]:
        """Validate snapshot for logging."""
        from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES

        issues = []

        # Check symbol
        if not snapshot.symbol:
            issues.append("missing_symbol")

        # Count non-null features against the active production feature profile.
        snapshot_dict = snapshot.to_dict()
        missing_features = [
            f for f in ACTIVE_SNAPSHOT_META_FEATURES
            if snapshot_dict.get(f) is None
        ]
        non_null = len(ACTIVE_SNAPSHOT_META_FEATURES) - len(missing_features)

        if non_null < min_features:
            issues.append(f"insufficient_features({non_null}/{min_features})")
        if missing_features:
            issues.append(
                "missing_profile_features(" + ",".join(missing_features) + ")"
            )

        return issues

    def flush(self) -> None:
        """Flush any buffered snapshots to disk."""
        if self._logger is not None:
            self._logger.flush()

    def close(self) -> None:
        """Close the collector and flush remaining data."""
        self.flush()
        logger.info(
            "FeatureCollector closed: logged=%d, skipped=%d, validation_failures=%d",
            self._stats["total_logged"],
            self._stats["total_skipped"],
            self._stats["validation_failures"],
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Convenience function for scanner integration
def create_collector(
    log_dir: str = "data/training_snapshots",
    enabled: bool = True,
    validate_snapshots: bool = False,
) -> FeatureCollector:
    """
    Create a feature collector for scanner integration.

    Args:
        log_dir: Directory to store snapshots
        enabled: Whether to enable logging
        validate_snapshots: Whether to skip snapshots with missing active
            profile fields. Defaults to False so logging preserves observed
            nulls for retrain-time contract checks instead of fabricating
            completeness.

    Returns:
        Configured FeatureCollector
    """
    return FeatureCollector(
        log_dir=log_dir,
        enabled=enabled,
        validate_snapshots=validate_snapshots,
    )
