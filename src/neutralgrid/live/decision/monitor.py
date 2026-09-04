"""Per-bot live evaluator -- produces a `BotEvaluation` from current market state.

Wires the existing scanner / HMM / meta-labeler / utility components into a
single `evaluate_bot()` coroutine. Stateless per call; heavy resources
(HMM model, meta-labeler, utility config) are loaded once into a
`MonitorContext` at scanner startup.

Microstructure hard gate is deferred to Phase D -- the recommender already
treats `micro_gate_pass=None` as "couldn't evaluate".

See LIVE_DECISION.md for the full design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.grid.formulas import (
    GEOMETRIC,
    grid_spacing_pct as _grid_spacing_pct,
    profit_per_grid_pct as _profit_per_grid_pct,
)
from neutralgrid.live.candidate_deploy_linker import DeployLinker
from neutralgrid.live.decision.loader import LiveBotSpec
from neutralgrid.live.decision.l2_risk import (
    L2RiskError,
    PositionNormalizedL2Risk,
    load_validated_l2_window,
    position_normalized_l2_risk_from_validated_window,
)
from neutralgrid.live.decision.recommender import BotEvaluation
from neutralgrid.live.decision.private_events import (
    PrivateEventError,
    PrivateEventEvidence,
    load_private_event_evidence,
)
from neutralgrid.live.decision.execution_risk import (
    ExecutionRiskError,
    ExecutionRiskEvidence,
    execution_risk_from_validated_l2_window,
)
from neutralgrid.models.meta_labeler import META_FEATURE_PROFILES
from neutralgrid.scanner.feature_extractor import compute_features, klines_to_df
from neutralgrid.scanner.pnl_ranker import PnLRanker, RankingConfig
from neutralgrid.scanner.scan import _compute_stochastic_features
from neutralgrid.validation.hmm_regime import load_hmm_model
from neutralgrid.validation.microstructure import MicrostructureEstimator
from neutralgrid.validation.microstructure_hard_gate import MicrostructureHardGate
from neutralgrid.validation.utility import (
    UtilityCalibratorUnavailable,
    UtilityConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_META_LABELER_PATH = Path("models") / "meta_labeler.pkl"

# When more than half of the meta-labeler's required features are missing the
# diagnostic overlay is skipped (D1). The AUTHORITATIVE tilt additionally
# requires strict full fidelity (0 missing); see LIVE_DECISION.md /
# METALABELER_LIVETELEMETRY.md.
_META_INACTIVE_MISSING_RATIO = 0.5


@dataclass
class MonitorContext:
    """Cached heavy resources reused across ticks.

    Each field may be ``None`` when the corresponding artifact is unavailable
    at startup; `evaluate_bot()` surfaces the absence as an explicit
    `*_artifact_missing` flag or `*_unavailable` diagnostic, never silently.
    """

    hmm: Any  # neutralgrid.models.hmm.inference.HMMRegimePredictor | None
    meta_labeler: Any  # neutralgrid.models.meta_labeler.MetaLabeler | None
    utility_config: Optional[UtilityConfig]
    hmm_unavailable: bool
    meta_unavailable: bool
    utility_unavailable: bool
    # Phase D additions
    micro_estimator: Optional[MicrostructureEstimator] = None
    micro_gate: Optional[MicrostructureHardGate] = None
    # Meta-labeler live-verdict wiring (D2/D3/D7). meta_promoted mirrors the
    # enrich authority gate (promotion_status == "pass", fail-closed). pnl_ranker
    # produces the meta `ev_score` feature exactly as run_full_pipeline does.
    # active_hmm_version / meta_feature_profile are captured once for the audit
    # trail; the MetaLabeler.load gate guarantees the meta lineage == active HMM.
    meta_promoted: bool = False
    pnl_ranker: Any = None  # neutralgrid.scanner.pnl_ranker.PnLRanker | None
    active_hmm_version: Optional[str] = None
    meta_feature_profile: Optional[str] = None
    # Linker rows keyed by strategy_id and candidate_id; loaded eagerly at
    # startup. Operationally: restart the scanner after deploying new bots so
    # the linkage cache picks them up.
    linker_by_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    linker_by_candidate: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        meta_labeler_path: Optional[Path] = None,
    ) -> MonitorContext:
        """Eagerly load HMM, meta-labeler, utility, microstructure, and linkage.

        Tolerates each artifact being absent. Logs WARNING per missing piece;
        the scanner still produces useful ADJUST recommendations carrying the
        corresponding diagnostic codes.
        """
        path = meta_labeler_path or DEFAULT_META_LABELER_PATH

        # -- HMM ----------------------------------------------------------
        hmm: Any = None
        hmm_unavailable = False
        try:
            hmm = load_hmm_model()
            if hmm is None:
                hmm_unavailable = True
                logger.warning(
                    "HMM artifact unavailable; range_prob/trend_prob will be NaN per tick"
                )
        except Exception as e:  # pragma: no cover - defensive
            hmm_unavailable = True
            logger.warning("HMM load failed: %s", e)

        # Active HMM version for the live audit trail (best-effort; the predictor
        # exposes metadata.artifact_version -- see models/hmm/inference.py).
        active_hmm_version = getattr(
            getattr(hmm, "metadata", None), "artifact_version", None
        )

        # -- Meta-labeler -------------------------------------------------
        meta: Any = None
        meta_unavailable = False
        try:
            from neutralgrid.models.meta_labeler import MetaLabeler

            meta = MetaLabeler.load(path)
        except FileNotFoundError:
            meta_unavailable = True
            logger.warning("Meta-labeler artifact at %s not found; overlay inactive", path)
        except Exception as e:
            meta_unavailable = True
            logger.warning("Meta-labeler load failed: %s; overlay inactive", e)

        # Authority mirror of enrich_grid_params.py:1892-1894 (promotion_status
        # == "pass", fail-closed for None/legacy/Mock). Reuses MetaLabeler.is_promoted.
        meta_promoted = bool(getattr(meta, "is_promoted", False)) if meta is not None else False
        # Feature-profile name for the audit trail -- same recipe as
        # run_full_pipeline.py:1071-1076 (match _feature_names to META_FEATURE_PROFILES).
        meta_feature_profile: Optional[str] = None
        if meta is not None:
            _feats = tuple(getattr(meta, "_feature_names", []) or [])
            meta_feature_profile = next(
                (name for name, feats in META_FEATURE_PROFILES.items() if tuple(feats) == _feats),
                None,
            )

        # -- Utility calibrator (offline-caller pattern: catch + warn) ----
        utility: Optional[UtilityConfig] = None
        utility_unavailable = False
        try:
            utility = UtilityConfig.from_artifact()
        except UtilityCalibratorUnavailable as e:
            utility_unavailable = True
            logger.warning(
                "Utility calibrator unavailable (%s); utility_score will be NaN", e
            )
        except Exception as e:  # pragma: no cover - defensive
            utility_unavailable = True
            logger.warning("Utility calibrator load failed: %s", e)

        # -- Microstructure (Phase D) -------------------------------------
        micro_estimator: Optional[MicrostructureEstimator] = None
        micro_gate: Optional[MicrostructureHardGate] = None
        try:
            micro_estimator = MicrostructureEstimator()
            micro_gate = MicrostructureHardGate()
        except Exception as e:
            logger.warning(
                "Microstructure gate unavailable: %s; micro_gate_pass will be None", e
            )

        # -- EV ranker (meta `ev_score` feature) --------------------------
        # Same construction as the scan/enrich path (run_full_pipeline.py:127).
        pnl_ranker: Any = None
        try:
            pnl_ranker = PnLRanker(RankingConfig())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("PnLRanker unavailable: %s; ev_score will be missing", e)

        # -- Deploy-time linkage (Phase D) --------------------------------
        # Last-match-wins: deploy_linkage_log.csv is append-only chronological,
        # so a re-deploy with the same strategy_id should attribute deltas to
        # the LATEST row, not the earliest. Plain assignment overwrites.
        linker_by_strategy: dict[str, dict[str, Any]] = {}
        linker_by_candidate: dict[str, dict[str, Any]] = {}
        try:
            linker = DeployLinker()
            for row in linker.load_all():
                sid = str(row.get("strategy_id", "")).strip()
                cid = str(row.get("candidate_id", "")).strip()
                if sid and sid.lower() != "nan":
                    linker_by_strategy[sid] = row
                if cid and cid.lower() != "nan":
                    linker_by_candidate[cid] = row
        except Exception as e:
            logger.warning(
                "DeployLinker load failed: %s; deploy-time deltas unavailable", e
            )

        return cls(
            hmm=hmm,
            meta_labeler=meta,
            utility_config=utility,
            hmm_unavailable=hmm_unavailable,
            meta_unavailable=meta_unavailable,
            utility_unavailable=utility_unavailable,
            micro_estimator=micro_estimator,
            micro_gate=micro_gate,
            meta_promoted=meta_promoted,
            pnl_ranker=pnl_ranker,
            active_hmm_version=active_hmm_version,
            meta_feature_profile=meta_feature_profile,
            linker_by_strategy=linker_by_strategy,
            linker_by_candidate=linker_by_candidate,
        )

    def lookup_deploy_row(self, spec: LiveBotSpec) -> Optional[dict[str, Any]]:
        """Best-effort linkage lookup: candidate_id first, then strategy_id."""
        if spec.candidate_id and spec.candidate_id in self.linker_by_candidate:
            return self.linker_by_candidate[spec.candidate_id]
        if spec.strategy_id and spec.strategy_id in self.linker_by_strategy:
            return self.linker_by_strategy[spec.strategy_id]
        return None


async def evaluate_bot(
    spec: LiveBotSpec,
    *,
    context: MonitorContext,
    client: BinanceClient,
    now: Optional[datetime] = None,
) -> BotEvaluation:
    """Run one full evaluation for a single bot. Never raises -- failures are
    encoded as flags / diagnostics on the returned `BotEvaluation`.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    diagnostics: list[str] = []

    private_event_evidence: Optional[PrivateEventEvidence] = None
    if spec.private_event_stream is not None:
        try:
            private_event_evidence = load_private_event_evidence(
                spec.private_event_stream,
                now=now,
            )
        except PrivateEventError as exc:
            diagnostics.append(f"private_event_stream_unavailable:{exc}")

    l2_risk: Optional[PositionNormalizedL2Risk] = None
    execution_risk: Optional[ExecutionRiskEvidence] = None
    if spec.l2_stream is not None:
        position = (
            spec.execution_telemetry.position_inventory
            if spec.execution_telemetry is not None
            else None
        )
        try:
            l2_window = load_validated_l2_window(spec.l2_stream, now=now)
        except L2RiskError as exc:
            diagnostics.append(f"l2_stream_unavailable:{exc}")
        else:
            try:
                l2_risk = position_normalized_l2_risk_from_validated_window(
                    l2_window,
                    now=now,
                    position_size_base=(
                        position.size_base if position is not None else None
                    ),
                    position_size_usdt=(
                        position.size_usdt if position is not None else None
                    ),
                )
            except L2RiskError as exc:
                diagnostics.append(f"l2_stream_unavailable:{exc}")
            try:
                execution_risk = execution_risk_from_validated_l2_window(
                    spec.l2_stream,
                    l2_window,
                    private_ref=spec.private_event_stream,
                    now=now,
                    position_size_base=(
                        position.size_base if position is not None else None
                    ),
                    position_size_usdt=(
                        position.size_usdt if position is not None else None
                    ),
                )
            except ExecutionRiskError as exc:
                diagnostics.append(f"execution_risk_unavailable:{exc}")

    # -- 1. Fetch market data ----------------------------------------------
    try:
        market = await client.get_all_market_data(spec.symbol)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else None
        if status is not None and 400 <= status < 500:
            return _empty_eval(
                spec, now, symbol_unavailable=True,
                diagnostics=tuple([*diagnostics, f"binance_4xx:{status}"]),
                l2_risk=l2_risk,
                private_event_evidence=private_event_evidence,
                execution_risk=execution_risk,
            )
        return _empty_eval(
            spec,
            now,
            transient_fetch_error=True,
            diagnostics=tuple([*diagnostics, f"http_status_{status}"]),
            l2_risk=l2_risk,
            private_event_evidence=private_event_evidence,
            execution_risk=execution_risk,
        )
    except (httpx.HTTPError, TimeoutError, OSError) as e:
        return _empty_eval(
            spec, now, transient_fetch_error=True,
            diagnostics=tuple([*diagnostics, f"fetch_error:{type(e).__name__}"]),
            l2_risk=l2_risk,
            private_event_evidence=private_event_evidence,
            execution_risk=execution_risk,
        )

    # -- 2. Convert klines to DataFrames -----------------------------------
    klines_raw = market.get("klines", {}) or {}
    try:
        klines_1h = klines_to_df(klines_raw.get("1h") or [])
        klines_15m = klines_to_df(klines_raw.get("15m") or [])
        klines_5m = klines_to_df(klines_raw.get("5m") or [])
        klines_1m = klines_to_df(klines_raw.get("1m") or [])
    except Exception as e:
        logger.warning("klines_to_df failed for %s: %s", spec.symbol, e)
        return _empty_eval(
            spec, now, transient_fetch_error=True,
            diagnostics=tuple([*diagnostics, "klines_parse_failed"]),
            l2_risk=l2_risk,
            private_event_evidence=private_event_evidence,
            execution_risk=execution_risk,
        )

    if klines_1h.empty:
        diagnostics.append("data_missing:1h")
    if klines_15m.empty:
        diagnostics.append("data_missing:15m")

    # -- 3. Compute scanner features ---------------------------------------
    # funding_rate / open_interest arrive as raw Binance payloads (a history list
    # and an OI dict). Extract the scalars scan.py uses so the meta-labeler
    # features -- and the micro gate / ev_score below -- receive scalars, not
    # containers (a list/dict would break get_missing_feature_names/predict_proba).
    funding_rate_scalar = _extract_funding_rate(market)
    open_interest_scalar = _extract_open_interest(market)
    sym_feats: Any = None
    try:
        sym_feats = compute_features(
            spec.symbol,
            klines_1h=klines_1h,
            klines_15m=klines_15m,
            klines_5m=klines_5m,
            klines_1m=klines_1m,
            funding_rate=funding_rate_scalar,
            funding_info=market.get("funding_info"),
            open_interest=open_interest_scalar,
            quote_volume_24h=_extract_quote_volume_24h(market),
        )
    except Exception as e:
        logger.warning("compute_features failed for %s: %s", spec.symbol, e)
        diagnostics.append("feature_extraction_failed")

    # -- 4. HMM regime inference -------------------------------------------
    hmm_result: Any = None
    hmm_artifact_missing = context.hmm_unavailable
    if context.hmm is not None and not klines_15m.empty:
        try:
            hmm_result = context.hmm.predict(df=klines_15m)
        except ValueError as e:
            # Insufficient bars / data quality -- treat as data_missing, not END
            diagnostics.append("data_missing:hmm_input")
            logger.debug("HMM predict failed for %s: %s", spec.symbol, e)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("HMM predict raised unexpectedly for %s: %s", spec.symbol, e)
            diagnostics.append("hmm_predict_failed")

    range_prob = float(hmm_result.range_prob) if hmm_result is not None else None
    trend_prob = float(hmm_result.trend_prob) if hmm_result is not None else None
    persistence_prob = (
        float(hmm_result.persistence_prob) if hmm_result is not None else None
    )

    # -- 5. Price-vs-grid math ---------------------------------------------
    last_price = _resolve_last_price(market, klines_1m, klines_15m)
    pct_inside, dist_lower, dist_upper = _price_vs_grid(spec, last_price)

    # -- 6. Microstructure costs (estimated ONCE; reused by gate + meta) ---
    micro_costs, micro_cost_reasons = _estimate_micro_costs(
        spec=spec, market=market, sym_feats=sym_feats, context=context,
    )
    micro_round_trip_cost_pct: Optional[float] = None
    if micro_costs is not None:
        _rt = getattr(micro_costs, "round_trip_cost_pct", None)
        if _rt is not None:
            try:
                micro_round_trip_cost_pct = round(float(_rt), 4)
            except (TypeError, ValueError):
                micro_round_trip_cost_pct = None

    # -- 7. Stochastic + geometric-grid + EV features (canonical reuse) ----
    range_size_pct = (
        getattr(sym_feats, "range_size_pct", None) if sym_feats is not None else None
    )
    stochastic: dict[str, Any] = {}
    if not klines_15m.empty:
        try:
            # >=300-bar guard lives inside; returns {} / partial below that.
            stochastic = _compute_stochastic_features(
                klines_15m, range_prob, trend_prob, range_size_pct
            )
        except Exception as e:
            logger.debug("stochastic features unavailable for %s: %s", spec.symbol, e)
            stochastic = {}
    survival_prob = stochastic.get("survival_prob")
    hurst_exponent = stochastic.get("hurst_exponent")
    ou_halflife = stochastic.get("ou_halflife")

    grid_spacing_pct, profit_per_grid_pct = _geometric_grid_features(spec)
    ev_score = _compute_ev_score(
        spec=spec,
        ranker=context.pnl_ranker,
        survival_prob=survival_prob,
        profit_per_grid_pct=profit_per_grid_pct,
        trend_prob=trend_prob,
        range_size_pct=range_size_pct,
        funding_rate=funding_rate_scalar,
    )

    # -- 8. Meta-labeler overlay (fail-closed feature guard) ---------------
    # meta_proba stays DIAGNOSTIC while up to 50% of features are imputed (D1);
    # the AUTHORITATIVE tilt (D4) additionally requires strict full fidelity
    # (0 missing), mirroring the enrich backfill gate
    # (run_full_pipeline.py:309-323). Missing features are left absent (not
    # imputed) so get_missing_feature_names() flags them.
    meta_proba: Optional[float] = None
    meta_missing_features: tuple[str, ...] = ()
    meta_full_fidelity = False
    if context.meta_labeler is not None:
        feature_dict = _build_meta_feature_dict(
            sym_feats=sym_feats,
            spec=spec,
            range_prob=range_prob,
            trend_prob=trend_prob,
            last_price=last_price,
            hurst_exponent=hurst_exponent,
            ou_halflife=ou_halflife,
            micro_round_trip_cost_pct=micro_round_trip_cost_pct,
            grid_spacing_pct=grid_spacing_pct,
            profit_per_grid_pct=profit_per_grid_pct,
            ev_score=ev_score,
        )
        try:
            missing = context.meta_labeler.get_missing_feature_names(feature_dict)
            meta_missing_features = tuple(missing)
            # Total required features = everything reported missing for an empty
            # probe dict. The loaded MetaLabeler exposes get_missing_feature_names
            # but NOT a get_feature_names accessor; calling the latter here used to
            # raise AttributeError and silently bypass this guard (fail-OPEN). See
            # METALABELER_LIVETELEMETRY.md D1.
            n_required = max(len(context.meta_labeler.get_missing_feature_names({})), 1)
            if len(missing) / n_required > _META_INACTIVE_MISSING_RATIO:
                diagnostics.append("meta_overlay_inactive")
            else:
                meta_proba = float(context.meta_labeler.predict_proba(feature_dict))
                if missing:
                    diagnostics.append(f"meta_imputed:{len(missing)}")
                else:
                    meta_full_fidelity = True
        except AttributeError:
            # A MetaLabeler build that does not expose get_missing_feature_names
            # cannot be feature-gated, so fail CLOSED (skip the overlay) rather
            # than predict on silently-imputed features.
            diagnostics.append("meta_overlay_inactive")
        except Exception as e:
            logger.warning("meta_labeler probe failed for %s: %s", spec.symbol, e)
            diagnostics.append("meta_predict_failed")
    elif context.meta_unavailable:
        diagnostics.append("meta_overlay_inactive")

    # Authority mirror of enrich (promotion_status == "pass", fail-closed) AND
    # strict full fidelity. Only an authoritative + full-fidelity meta_proba may
    # influence a verdict (D4).
    meta_authoritative = bool(
        context.meta_promoted and meta_proba is not None and meta_full_fidelity
    )
    # The MetaLabeler.load gate guarantees meta lineage == active HMM, so the
    # linked version equals the active version when a meta-labeler is loaded
    # (mirrors run_full_pipeline.py:1077).
    linked_hmm_version = (
        context.active_hmm_version if context.meta_labeler is not None else None
    )

    # -- 9. Utility score (offline-caller pattern) -------------------------
    utility_score: Optional[float] = None
    if context.utility_unavailable:
        diagnostics.append("utility_calibrator_unavailable")
    elif context.utility_config is not None and range_prob is not None and trend_prob is not None:
        # Diagnostic-grade hint from the calibrated lambda; the EV-based
        # `ev_score` meta feature now comes from PnLRanker (step 7, D2).
        utility_score = _approx_utility_hint(
            range_prob=range_prob,
            trend_prob=trend_prob,
            cfg=context.utility_config,
        )

    # -- 10. Microstructure hard gate (reuses step-6 costs) ----------------
    micro_gate_pass, micro_gate_reasons = _run_microstructure_gate(
        spec=spec, costs=micro_costs, context=context,
    )
    micro_reasons = micro_cost_reasons + micro_gate_reasons

    # -- 11. Deploy-time linkage snapshot + delta (Phase D) ----------------
    deploy_snapshot = _build_deploy_snapshot(
        spec=spec, context=context, current_meta_proba=meta_proba,
        diagnostics=diagnostics,
    )

    return BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=now,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        deploy_ts=spec.deploy_ts,
        price=last_price,
        pct_inside_grid=pct_inside,
        dist_to_lower_pct=dist_lower,
        dist_to_upper_pct=dist_upper,
        range_prob=range_prob,
        trend_prob=trend_prob,
        persistence_prob=persistence_prob,
        meta_proba=meta_proba,
        utility_score=utility_score,
        micro_gate_pass=micro_gate_pass,
        micro_reasons=tuple(micro_reasons),
        symbol_unavailable=False,
        transient_fetch_error=False,
        hmm_artifact_missing=hmm_artifact_missing,
        diagnostics=tuple(diagnostics),
        deploy_snapshot=deploy_snapshot,
        execution_telemetry=spec.execution_telemetry,
        l2_risk=l2_risk,
        private_event_evidence=private_event_evidence,
        execution_risk=execution_risk,
        meta_authoritative=meta_authoritative,
        meta_full_fidelity=meta_full_fidelity,
        meta_missing_features=meta_missing_features,
        meta_source="live_overlay" if meta_proba is not None else None,
        active_hmm_version=context.active_hmm_version,
        linked_hmm_version=linked_hmm_version,
        meta_feature_profile=context.meta_feature_profile,
    )


# -- Helpers -------------------------------------------------------------------


def _empty_eval(
    spec: LiveBotSpec,
    now: datetime,
    *,
    symbol_unavailable: bool = False,
    transient_fetch_error: bool = False,
    diagnostics: tuple[str, ...] = (),
    l2_risk: Optional[PositionNormalizedL2Risk] = None,
    private_event_evidence: Optional[PrivateEventEvidence] = None,
    execution_risk: Optional[ExecutionRiskEvidence] = None,
) -> BotEvaluation:
    """Build a minimal BotEvaluation when the fetch pipeline could not run."""
    return BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=now,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        deploy_ts=spec.deploy_ts,
        symbol_unavailable=symbol_unavailable,
        transient_fetch_error=transient_fetch_error,
        diagnostics=diagnostics,
        execution_telemetry=spec.execution_telemetry,
        l2_risk=l2_risk,
        private_event_evidence=private_event_evidence,
        execution_risk=execution_risk,
    )


def _extract_quote_volume_24h(market: dict[str, Any]) -> Optional[float]:
    ticker = market.get("ticker") or market.get("ticker_24h") or {}
    if isinstance(ticker, dict):
        v = ticker.get("quoteVolume") or ticker.get("quote_volume")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _extract_funding_rate(market: dict[str, Any]) -> Optional[float]:
    """Current funding rate as a SCALAR.

    ``get_all_market_data`` returns ``funding_rate`` as the raw Binance history
    *list*; the model feature is a scalar. Mirror scan.py (``prem["lastFundingRate"]``,
    scan.py:408) and fall back to the most recent history entry.
    """
    prem = market.get("premium_index")
    if isinstance(prem, dict) and prem.get("lastFundingRate") is not None:
        try:
            return float(prem["lastFundingRate"])
        except (TypeError, ValueError):
            pass
    hist = market.get("funding_rate")
    if isinstance(hist, list) and hist:
        try:
            return float(hist[-1]["fundingRate"])
        except (TypeError, ValueError, KeyError, IndexError):
            pass
    if isinstance(hist, (int, float)):
        return float(hist)
    return None


def _extract_open_interest(market: dict[str, Any]) -> Optional[float]:
    """Open interest as a SCALAR.

    ``get_all_market_data`` returns ``open_interest`` as the raw Binance *dict*;
    the model feature is a scalar. Mirror scan.py (``float(oi["openInterest"])``,
    scan.py:417).
    """
    oi = market.get("open_interest")
    if isinstance(oi, dict) and oi.get("openInterest") is not None:
        try:
            return float(oi["openInterest"])
        except (TypeError, ValueError):
            pass
    if isinstance(oi, (int, float)):
        return float(oi)
    return None


def _resolve_last_price(
    market: dict[str, Any],
    klines_1m: pd.DataFrame,
    klines_15m: pd.DataFrame,
) -> Optional[float]:
    """Best-effort last-price resolution. Prefer ticker, then 1m close, then 15m close."""
    ticker = market.get("ticker") or {}
    if isinstance(ticker, dict):
        for key in ("lastPrice", "last_price", "close"):
            v = ticker.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    for kdf in (klines_1m, klines_15m):
        if not kdf.empty and "close" in kdf.columns:
            try:
                return float(kdf["close"].iloc[-1])
            except (TypeError, ValueError, IndexError):
                continue
    return None


def _price_vs_grid(
    spec: LiveBotSpec, price: Optional[float]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (pct_inside_grid, dist_to_lower_pct, dist_to_upper_pct).

    pct_inside_grid = 100*(price-lower)/(upper-lower); <0 or >100 ⇒ outside.
    dist_to_lower/upper_pct are computed only when price is inside the grid
    (negative-distance does not make sense once the price is outside).
    """
    if price is None:
        return None, None, None
    width = spec.grid_upper - spec.grid_lower
    if width <= 0:
        return None, None, None
    pct_inside = 100.0 * (price - spec.grid_lower) / width
    if pct_inside < 0 or pct_inside > 100:
        return pct_inside, None, None
    dist_lower = pct_inside  # distance from lower in %-of-width
    dist_upper = 100.0 - pct_inside
    return pct_inside, dist_lower, dist_upper


def _build_meta_feature_dict(
    *,
    sym_feats: Any,
    spec: LiveBotSpec,
    range_prob: Optional[float],
    trend_prob: Optional[float],
    last_price: Optional[float],
    hurst_exponent: Optional[float] = None,
    ou_halflife: Optional[float] = None,
    micro_round_trip_cost_pct: Optional[float] = None,
    grid_spacing_pct: Optional[float] = None,
    profit_per_grid_pct: Optional[float] = None,
    ev_score: Optional[float] = None,
) -> dict[str, Any]:
    """Assemble the meta-labeler feature dict from canonical, reused sources.

    `SymbolFeatures.as_dict()` supplies the 13 technical/market features; the HMM
    probabilities and spec-derived `num_grids` are added here. The remaining
    features are computed live by the caller via the SAME code paths the
    training/scan pipeline uses (no train-serve skew) and passed in:

      - hurst_exponent, ou_halflife          -> scanner.scan._compute_stochastic_features
      - micro_round_trip_cost_pct            -> MicrostructureEstimator.estimate_costs
      - grid_spacing_pct, profit_per_grid_pct -> geometric grid.formulas (PERCENT)
      - ev_score                             -> PnLRanker.compute_score().rank_score

    A feature that could not be computed (e.g. <300 15m bars -> stochastic
    unavailable, or degenerate grid geometry) is left ABSENT (not imputed) so
    `get_missing_feature_names()` flags it and the caller can fail closed --
    never silently imputed into an authoritative decision (see
    safety-invariants.md / METALABELER_LIVETELEMETRY.md / LIVE_DECISION.md).

    This function NEVER includes `hlabel` or any other label column.
    """
    out: dict[str, Any] = {}
    if sym_feats is not None and hasattr(sym_feats, "as_dict"):
        out.update(sym_feats.as_dict())
    if range_prob is not None:
        out["range_prob"] = range_prob
    if trend_prob is not None:
        out["trend_prob"] = trend_prob
    out["num_grids"] = float(spec.num_grids)
    # Geometric grid geometry (PERCENT) matching the canonical training/scan
    # computation; left absent when the geometry could not be computed.
    if grid_spacing_pct is not None:
        out["grid_spacing_pct"] = grid_spacing_pct
    if profit_per_grid_pct is not None:
        out["profit_per_grid_pct"] = profit_per_grid_pct
    if hurst_exponent is not None:
        out["hurst_exponent"] = hurst_exponent
    if ou_halflife is not None:
        out["ou_halflife"] = ou_halflife
    if micro_round_trip_cost_pct is not None:
        out["micro_round_trip_cost_pct"] = micro_round_trip_cost_pct
    if ev_score is not None:
        out["ev_score"] = ev_score
    if last_price is not None and "last_price" not in out:
        out["last_price"] = last_price
    return out


def _geometric_grid_features(spec: LiveBotSpec) -> tuple[Optional[float], Optional[float]]:
    """Geometric ``(grid_spacing_pct, profit_per_grid_pct)`` in PERCENT.

    Reuses the single-source formulas in ``grid.formulas`` with GEOMETRIC mode,
    the default legacy-line-count semantics, and config fees
    ``c = (maker_fee + close_fee_rate) / 2`` -- identical to the expired-bot
    training computation (`data_generator.compute_profit_per_grid`). Returns
    ``(None, None)`` on degenerate geometry (e.g. num_grids < 2) so the features
    are marked genuinely-missing rather than fabricated.
    """
    try:
        cfg_grid = get_config().grid
        maker = float(cfg_grid.maker_fee)
        taker = float(cfg_grid.taker_fee)
        close_mode = str(getattr(cfg_grid, "close_fee_mode", "maker")).lower()
        close_rate = maker if close_mode == "maker" else taker
        c = max(0.0, (maker + close_rate) / 2.0)
        spacing = _grid_spacing_pct(
            spec.grid_lower, spec.grid_upper, int(spec.num_grids), GEOMETRIC
        )
        profit = _profit_per_grid_pct(
            spec.grid_lower, spec.grid_upper, int(spec.num_grids), GEOMETRIC, c
        )
        return float(spacing), float(profit)
    except Exception as e:
        logger.debug("geometric grid features unavailable for %s: %s", spec.symbol, e)
        return None, None


def _compute_ev_score(
    *,
    spec: LiveBotSpec,
    ranker: Any,
    survival_prob: Optional[float],
    profit_per_grid_pct: Optional[float],
    trend_prob: Optional[float],
    range_size_pct: Optional[float],
    funding_rate: Optional[float],
) -> Optional[float]:
    """Meta ``ev_score`` via ``PnLRanker.compute_score().rank_score`` -- the
    EV-based, meta-independent ranking score the scan/enrich path uses
    (run_full_pipeline.py:214-225). Returns None when any required input is
    absent, so ev_score is genuinely missing rather than fabricated. Fees are
    NOT supplied by the live spec; the ranker uses config fees internally.
    """
    if (
        ranker is None
        or survival_prob is None
        or profit_per_grid_pct is None
        or trend_prob is None
    ):
        return None
    try:
        res = ranker.compute_score(
            profit_per_grid_pct=float(profit_per_grid_pct),
            num_grids=int(spec.num_grids),
            survival_prob=float(survival_prob),
            trend_prob=float(trend_prob),
            funding_rate=float(funding_rate) if funding_rate is not None else None,
            range_size_pct=float(range_size_pct) if range_size_pct is not None else 3.0,
            symbol=spec.symbol,
            leverage=int(spec.leverage),
        )
        return round(float(res.rank_score), 4)
    except Exception as e:
        logger.debug("ev_score computation failed for %s: %s", spec.symbol, e)
        return None


def _approx_utility_hint(
    *, range_prob: float, trend_prob: float, cfg: UtilityConfig
) -> float:
    """Diagnostic-grade utility hint.

    Combines the calibrated `lambda_risk` and `kappa_trend` into a quick
    range-vs-trend balance score, NaN-safe. This is *not* the EV-based score
    from `pnl_ranker.compute_score()` -- that requires survival_prob which
    we do not plumb in this read-only path. See LIVE_DECISION.md Open Items.
    """
    return float(range_prob - cfg.kappa_trend * trend_prob)


def _estimate_micro_costs(
    *,
    spec: LiveBotSpec,
    market: dict[str, Any],
    sym_feats: Any,
    context: MonitorContext,
) -> tuple[Any, list[str]]:
    """Estimate microstructure costs ONCE per tick from the order book.

    Returns ``(costs, reasons)``. ``costs`` is None when the estimator is
    unavailable, the order book is missing, or estimation fails; ``reasons``
    carries the diagnostic codes. The single estimate is reused by BOTH the
    microstructure hard gate and the meta-labeler ``micro_round_trip_cost_pct``
    feature (no duplicate estimation).
    """
    if context.micro_estimator is None or context.micro_gate is None:
        return None, []

    order_book = market.get("order_book")
    if not order_book or not isinstance(order_book, dict):
        return None, ["data_missing:order_book"]

    funding_rate = _extract_funding_rate(market)

    # Volatility input: prefer SymbolFeatures.atr_pct_15m when available,
    # fall back to a neutral 2.0% to keep the gate from misfiring on missing data.
    volatility_pct: float = 2.0
    if sym_feats is not None:
        atr = getattr(sym_feats, "atr_pct_15m", None)
        if atr is not None:
            try:
                volatility_pct = float(atr)
            except (TypeError, ValueError):
                pass

    try:
        costs = context.micro_estimator.estimate_costs(
            order_book=order_book,
            funding_rate=funding_rate,
            volatility_pct=volatility_pct,
        )
    except Exception as e:
        logger.debug("MicrostructureEstimator.estimate_costs failed for %s: %s", spec.symbol, e)
        return None, [f"micro_estimate_failed:{type(e).__name__}"]

    return costs, []


def _run_microstructure_gate(
    *,
    spec: LiveBotSpec,
    costs: Any,
    context: MonitorContext,
) -> tuple[Optional[bool], list[str]]:
    """Run the microstructure hard gate against pre-computed ``costs``.

    Returns ``(passed, rejection_codes)``. `passed=None` means the gate
    couldn't run (no gate, no costs). The recommender treats None as
    "couldn't evaluate" -- it does NOT cascade to an END verdict.
    """
    if context.micro_gate is None or costs is None:
        return None, []

    # Approximate profit_per_grid_pct from the spec (arithmetic grid) for the
    # gate's conservative spread-to-profit floor ONLY. This is intentionally NOT
    # the meta `profit_per_grid_pct` feature (which is geometric, see
    # _geometric_grid_features). None is also acceptable per the gate code.
    midpoint = (spec.grid_lower + spec.grid_upper) / 2.0
    profit_per_grid_pct: Optional[float] = None
    if midpoint > 0 and spec.num_grids > 0:
        profit_per_grid_pct = (
            (spec.grid_upper - spec.grid_lower) / spec.num_grids / midpoint * 100.0
        )

    try:
        result = context.micro_gate.evaluate(
            round_trip_cost_pct=costs.round_trip_cost_pct,
            spread_cost_pct=costs.spread_cost_pct,
            funding_cost_pct=costs.funding_cost_pct,
            sufficient_liquidity=costs.sufficient_liquidity,
            profit_per_grid_pct=profit_per_grid_pct,
            dynamic_profit_floor_pct=None,
        )
    except Exception as e:
        logger.debug("MicrostructureHardGate.evaluate failed for %s: %s", spec.symbol, e)
        return None, [f"micro_gate_failed:{type(e).__name__}"]

    return bool(result.passed), list(result.rejection_codes)


# Linkage-row keys we copy into the deploy snapshot. Kept narrow so the JSONL
# stays compact and we do not leak unrelated fields.
_DEPLOY_SNAPSHOT_KEYS: tuple[str, ...] = (
    "deploy_time_utc",
    "candidate_id",
    "strategy_id",
    "meta_prob",
    "score",
    "scan_score",
    "ev_score",
    "ev_contract_fingerprint",
    "ev_24h",
    "deployment_score",
    "grid_lower",
    "grid_upper",
    "num_grids",
    "leverage",
    "grid_spacing_pct",
    "profit_per_grid_pct",
)


def _build_deploy_snapshot(
    *,
    spec: LiveBotSpec,
    context: MonitorContext,
    current_meta_proba: Optional[float],
    diagnostics: list[str],
) -> Optional[dict[str, Any]]:
    """Look up the deployment row and build a compact snapshot for the digest.

    Computes ``delta_meta_prob = current_meta_proba - deploy.meta_prob`` when
    both are non-None. Mutates *diagnostics* with ``candidate_link_missing``
    when the YAML provides a candidate_id / strategy_id but the linker
    returns nothing -- this is a soft signal, NOT a verdict trigger.
    """
    # Only attempt lookup when the user gave us at least one identifier
    has_identifier = bool(spec.candidate_id or spec.strategy_id)
    if not has_identifier:
        return None

    row = context.lookup_deploy_row(spec)
    if row is None:
        diagnostics.append("candidate_link_missing")
        return None

    snapshot: dict[str, Any] = {}
    for key in _DEPLOY_SNAPSHOT_KEYS:
        if key not in row:
            continue
        value = row[key]
        # CSV reads back as strings or NaN -- coerce numeric-looking values
        if isinstance(value, str):
            try:
                value = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                pass  # leave as string (e.g. ISO timestamps)
        # Skip NaN markers from pandas (float('nan'))
        if isinstance(value, float) and value != value:
            continue
        snapshot[key] = value

    deploy_meta_prob_raw = snapshot.get("meta_prob")
    delta: Optional[float] = None
    try:
        if (
            current_meta_proba is not None
            and deploy_meta_prob_raw is not None
            and deploy_meta_prob_raw == deploy_meta_prob_raw  # NaN-safe
        ):
            delta = float(current_meta_proba) - float(deploy_meta_prob_raw)
    except (TypeError, ValueError):
        pass

    if delta is not None:
        snapshot["delta_meta_prob"] = delta

    return snapshot or None
