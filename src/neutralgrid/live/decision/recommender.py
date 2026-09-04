"""Pure verdict logic for the tactical live decision tool.

Given a `BotEvaluation` (snapshot of one bot at one tick) and a `BotHistory`
(persisted per-bot state), `decide()` returns:

* a `Recommendation` (verdict + reasons + suggested re-centered bounds)
* whether to emit it on this tick (after cool-down)
* a fresh `BotHistory` with counters / last-emitted advanced

The function is pure; the caller persists the new history.

See LIVE_DECISION.md (project root) for the verdict-mapping table.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from neutralgrid.live.decision.loader import LiveExecutionTelemetry
from neutralgrid.live.decision.l2_risk import PositionNormalizedL2Risk
from neutralgrid.live.decision.private_events import PrivateEventEvidence
from neutralgrid.live.decision.execution_risk import ExecutionRiskEvidence
from neutralgrid.live.decision.state_store import (
    BotHistory,
    TickSummary,
    append_tick,
)


class Verdict(str, Enum):
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    END = "END"


@dataclass(frozen=True)
class RecommenderConfig:
    """Tunable thresholds + cool-downs. Defaults match LIVE_DECISION.md."""

    min_range_prob: float = 0.45
    adjust_range_prob_floor: float = 0.30
    end_trend_prob_threshold: float = 0.40
    boundary_proximity_pct: float = 10.0
    end_cooldown_min: int = 30
    adjust_escalate_after: int = 3
    continue_heartbeat_min: int = 60
    microstructure_failures_to_end: int = 3
    transient_failures_to_end: int = 3
    emit_every_tick: bool = False
    # D4 caution-only meta tilt. Default DISABLED + threshold 0.0 => inert
    # (an authoritative meta_proba < 0.0 is impossible), so verdicts are
    # unchanged until an operating point is calibrated from shadow data (D5).
    # `from_file` auto-loads these (it iterates dataclass fields generically).
    meta_tilt_enabled: bool = False
    meta_tilt_low_threshold: float = 0.0
    # ── Hysteresis END trigger (contract v1.1, GATEFIX-02) ──────────────
    # Replay evidence (638 engine-verified pseudo-deployments, 2026-07-13):
    # the v1.0 single-tick outside END had exit precision 0.38-0.47 across
    # every cohort split and destroyed winner upside ~3.7x larger than the
    # hysteresis rule's (kept_W -$10,372 vs -$2,786 on the 484-deployment
    # cohort). The replacement END fires on EITHER 180 wall-clock minutes
    # continuously outside (observed on >= 3 consecutive ticks, gap-reset
    # below) OR displacement >= 2.0 half-widths from grid center. Precision
    # replicated 0.61-0.71 on held-out splits; displacement neighbors 1.25,
    # 1.5, 2.5, 3.0 and persistence neighbors 15/30/60min all tested and
    # inferior. Set end_on_first_outside_tick=True to restore exact v1.0
    # behavior.
    end_on_first_outside_tick: bool = False
    end_outside_persistence_min: float = 180.0
    end_outside_min_ticks: int = 3
    # NOTE: the displacement stop is unreachable on the DOWNSIDE when
    # (upper+lower)/(upper-lower) < end_displacement_multiple (price >= 0
    # caps downside displacement); for grids with upper > 3*lower a total
    # collapse exits via persistence instead.
    end_displacement_multiple: float = 2.0
    # Unobserved time must not count as "continuously outside": if the gap
    # since the last recorded tick exceeds this, the price-outside state is
    # reset before updating (scanner outage / laptop sleep safety).
    end_state_max_gap_min: float = 15.0
    # Once a price END has fired it stays latched (no CONTINUE retraction /
    # 180-min re-arm from a single inside tick); cleared after this many
    # consecutive inside ticks.
    end_latch_clear_ticks: int = 12

    def __post_init__(self) -> None:
        # Type-then-range validation, fail-closed at construction. bool is
        # excluded from the numeric check (bool subclasses int) so a YAML
        # "true" cannot silently pass as a threshold.
        for name, minimum, exclusive in (
            ("end_outside_persistence_min", 0.0, True),
            ("end_outside_min_ticks", 1, False),
            ("end_displacement_multiple", 1.0, True),
            ("end_state_max_gap_min", 0.0, True),
            ("end_latch_clear_ticks", 1, False),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"RecommenderConfig.{name} must be numeric, got "
                    f"{type(value).__name__}: {value!r}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(f"RecommenderConfig.{name} must be finite, got {value!r}")
            if (value <= minimum) if exclusive else (value < minimum):
                op = ">" if exclusive else ">="
                raise ValueError(
                    f"RecommenderConfig.{name} must be {op} {minimum}, got {value!r}"
                )
        if not isinstance(self.end_on_first_outside_tick, bool):
            raise ValueError(
                "RecommenderConfig.end_on_first_outside_tick must be a bool, got "
                f"{type(self.end_on_first_outside_tick).__name__}"
            )

    @classmethod
    def from_file(cls, path: Path) -> RecommenderConfig:
        """Load thresholds from a YAML or JSON file. Unknown keys logged + ignored.

        Schema is a flat mapping of any subset of `RecommenderConfig` field
        names to their values (e.g. ``min_range_prob: 0.50``). Missing keys
        keep their defaults.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        ValueError
            If the file cannot be parsed or contains a non-mapping payload.
        """
        import json
        import logging as _logging

        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # local import keeps pyyaml optional for callers

            data = yaml.safe_load(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ValueError(
                f"--config-file must be .yaml/.yml/.json, got {path.suffix!r}"
            )

        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"config file root must be a mapping, got {type(data).__name__}")

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        overrides: dict[str, Any] = {}
        log = _logging.getLogger(__name__)
        for k, v in data.items():
            if k not in valid_keys:
                log.warning("config file: ignoring unknown key %r", k)
                continue
            overrides[k] = v
        return cls(**overrides)


@dataclass(frozen=True)
class BotEvaluation:
    """One-tick snapshot for one bot.

    Phase A: produced synthetically by tests / CLI dry-run.
    Phase B: produced by `monitor.evaluate_bot()`.
    Phase D: gains optional `deploy_snapshot` for cross-time deltas.
    """

    symbol: str
    evaluated_at_utc: datetime
    grid_lower: float
    grid_upper: float

    # Deploy timestamp of the bot spec (v1.1). Used to invalidate stale
    # price-outside state inherited through a reused strategy_id state key:
    # a price_outside_since predating deploy_ts belongs to a previous
    # deployment and is reset. None (e.g. old callers) skips the check.
    deploy_ts: Optional[datetime] = None

    # Market snapshot (None when the corresponding fetch failed)
    price: Optional[float] = None
    pct_inside_grid: Optional[float] = None  # 100*(price-lower)/(upper-lower); <0 or >100 = outside
    dist_to_lower_pct: Optional[float] = None  # only populated when price inside grid
    dist_to_upper_pct: Optional[float] = None

    # Regime / model outputs
    range_prob: Optional[float] = None
    trend_prob: Optional[float] = None
    persistence_prob: Optional[float] = None
    meta_proba: Optional[float] = None
    utility_score: Optional[float] = None

    # Gates (None = couldn't evaluate, e.g. due to data_missing)
    micro_gate_pass: Optional[bool] = None
    micro_reasons: tuple[str, ...] = ()

    # Operational flags
    symbol_unavailable: bool = False
    transient_fetch_error: bool = False
    hmm_artifact_missing: bool = False

    # Free-form diagnostic codes ("data_missing:1h", "meta_overlay_inactive", etc.)
    diagnostics: tuple[str, ...] = ()

    # Phase D: deploy-time snapshot via candidate_deploy_linker. Populated
    # only when the YAML carries a candidate_id (or strategy_id) that
    # resolves to a linkage row. Keys we care about:
    #   meta_prob, score, ev_score, ev_24h, deploy_time_utc
    #   delta_meta_prob (computed: current.meta_proba - deploy.meta_prob)
    deploy_snapshot: Optional[dict[str, Any]] = None

    # User/account supplied live execution telemetry from the active bot registry
    # or canonical Live/<date>/<symbol>/live_bot_data_scanner.yaml.
    execution_telemetry: Optional[LiveExecutionTelemetry] = None

    # Fresh sequence-linked L2 evidence. It is observational in this phase and
    # is not read by the verdict rules.
    l2_risk: Optional[PositionNormalizedL2Risk] = None

    # Exact-strategy private order/fill/income evidence. Completeness is kept
    # explicit and this field is observational: verdict rules do not read it.
    private_event_evidence: Optional[PrivateEventEvidence] = None

    # Sequence-linked L2/public/private execution-quality evidence. This is
    # observational and cannot influence verdict mapping in this phase.
    execution_risk: Optional[ExecutionRiskEvidence] = None

    # Meta-labeler live-verdict wiring (D3/D7). `meta_authoritative` mirrors the
    # enrich gate (promotion_status=="pass", fail-closed); `meta_full_fidelity`
    # means 0 required features were missing/imputed. Only an authoritative +
    # full-fidelity meta_proba may influence a verdict (D4). The remaining fields
    # are write-only audit evidence (never read back into a verdict or scan).
    meta_authoritative: bool = False
    meta_full_fidelity: bool = False
    meta_missing_features: tuple[str, ...] = ()
    meta_source: Optional[str] = None
    active_hmm_version: Optional[str] = None
    linked_hmm_version: Optional[str] = None
    meta_feature_profile: Optional[str] = None

    @property
    def grid_width(self) -> float:
        return self.grid_upper - self.grid_lower


@dataclass(frozen=True)
class Recommendation:
    verdict: Verdict
    reasons: tuple[str, ...]
    suggested_grid_lower: Optional[float]
    suggested_grid_upper: Optional[float]
    consecutive_count: int
    escalated: bool
    # D4 shadow audit: `meta_would_tilt` is computed every tick regardless of
    # the enable flag; `meta_influenced_verdict` is True only when the tilt was
    # actually applied (enabled). Both are write-only audit evidence (D7).
    meta_would_tilt: bool = False
    meta_influenced_verdict: bool = False
    profit_deterioration: Optional[ProfitDeteriorationEvidence] = None


@dataclass(frozen=True)
class ProfitDeteriorationEvidence:
    """Deduplicated observed PnL path; never a verdict threshold."""

    observation_status: str
    captured_at_utc: Optional[datetime]
    current_total_profit_usdt: float
    previous_total_profit_usdt: Optional[float]
    change_from_previous_usdt: Optional[float]
    peak_total_profit_usdt: Optional[float]
    peak_total_profit_at: Optional[datetime]
    giveback_usdt: Optional[float]
    giveback_pct_of_positive_peak: Optional[float]
    current_is_profitable: bool
    observation_count: int
    matched_profit_usdt: Optional[float]
    unmatched_pnl_usdt: Optional[float]
    funding_fee_usdt: Optional[float]
    transaction_fee_usdt: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("captured_at_utc", "peak_total_profit_at"):
            value = payload[key]
            if isinstance(value, datetime):
                payload[key] = value.astimezone(timezone.utc).isoformat()
        return payload


@dataclass(frozen=True)
class RecommenderResult:
    recommendation: Recommendation
    should_emit: bool
    new_history: BotHistory
    diagnostics: tuple[str, ...] = field(default=())


def decide(
    evaluation: BotEvaluation,
    history: BotHistory,
    config: RecommenderConfig,
    now: datetime,
) -> RecommenderResult:
    """Compute the verdict for one tick and return updated history.

    Pure function. Caller is responsible for persisting `result.new_history`
    via `state_store.save_history()`.
    """
    (
        profit_evidence,
        profit_deploy_ts,
        peak_total_profit_usdt,
        peak_total_profit_at,
        last_total_profit_usdt,
        last_total_profit_at,
        pnl_observation_count,
    ) = _update_profit_deterioration(evaluation, history)

    # -- Cross-tick counters ------------------------------------------------
    consecutive_fetch_errors = (
        history.consecutive_fetch_errors + 1 if evaluation.transient_fetch_error else 0
    )
    consecutive_micro_failures = (
        history.consecutive_micro_failures + 1 if evaluation.micro_gate_pass is False else 0
    )

    # -- Price-outside state machine (contract v1.1) -------------------------
    # Tri-state: True/False when price and pct_inside_grid are present and
    # finite; None (unknown) otherwise. Unknown neither advances nor resets
    # the persistence state — a data gap cannot END a bot and cannot erase an
    # accrued timer.
    outside_now: Optional[bool] = None
    if (
        evaluation.price is not None
        and evaluation.pct_inside_grid is not None
        and math.isfinite(evaluation.price)
        and math.isfinite(evaluation.pct_inside_grid)
    ):
        outside_now = (
            evaluation.pct_inside_grid < 0 or evaluation.pct_inside_grid > 100
        )

    price_outside_since = history.price_outside_since
    consecutive_price_outside = history.consecutive_price_outside
    consecutive_price_inside = history.consecutive_price_inside
    price_end_latched = history.price_end_latched

    # Staleness guard 1: state predating this deployment (stale strategy_id
    # state-key reuse) is discarded entirely, latch included.
    if (
        evaluation.deploy_ts is not None
        and price_outside_since is not None
        and price_outside_since < evaluation.deploy_ts
    ):
        price_outside_since = None
        consecutive_price_outside = 0
        consecutive_price_inside = 0
        price_end_latched = False

    # Staleness guard 2: unobserved time must not count as "continuously
    # outside". If the gap since the last recorded tick exceeds the limit,
    # reset the accumulation counters (the latch survives — a gap does not
    # un-break an already-ENDed bot).
    if history.ticks:
        last_observed = history.ticks[-1].evaluated_at_utc
        gap_min = (now - last_observed).total_seconds() / 60.0
        if gap_min > config.end_state_max_gap_min:
            price_outside_since = None
            consecutive_price_outside = 0
            consecutive_price_inside = 0

    just_flipped_outside = outside_now is True and consecutive_price_outside == 0
    if outside_now is True:
        consecutive_price_outside += 1
        consecutive_price_inside = 0
        if price_outside_since is None:
            price_outside_since = now
    elif outside_now is False:
        consecutive_price_outside = 0
        consecutive_price_inside += 1
        price_outside_since = None
        if (
            price_end_latched
            and consecutive_price_inside >= config.end_latch_clear_ticks
        ):
            price_end_latched = False
    # outside_now is None: carry all price-state fields unchanged.

    end_reasons: list[str] = []
    adjust_reasons: list[str] = []
    # D4: computed inside the ADJUST-only block below; stays False whenever the
    # bot is END (the END layer is untouchable by the meta tilt).
    meta_would_tilt = False

    # -- END conditions ----------------------------------------------------
    if evaluation.symbol_unavailable:
        end_reasons.append("symbol_unavailable")

    if config.end_on_first_outside_tick:
        # Legacy v1.0 behavior: a single outside tick ENDs the bot. State
        # counters above are still maintained so toggling the flag does not
        # destroy accrued timers.
        if outside_now is True:
            end_reasons.append(
                f"price_outside_grid:{evaluation.pct_inside_grid:.1f}pct"
            )
    else:
        # v1.1 hysteresis END. Evidence: see RecommenderConfig field comment.
        if price_end_latched:
            # A prior price END stays END until the price has been back
            # inside for end_latch_clear_ticks consecutive ticks — no
            # CONTINUE retraction / fresh 180-min re-arm from one inside tick.
            end_reasons.append("price_end_latched")
        # Displacement disaster stop (stateless, high precision: 0.74-0.80
        # across cohorts).
        if (
            evaluation.price is not None
            and math.isfinite(evaluation.price)
            and evaluation.grid_width > 0
        ):
            _half_width = evaluation.grid_width / 2.0
            _center = (evaluation.grid_lower + evaluation.grid_upper) / 2.0
            displacement = abs(evaluation.price - _center) / _half_width
            if displacement >= config.end_displacement_multiple:
                end_reasons.append(f"price_displacement:{displacement:.2f}x")
                price_end_latched = True
        # Persistence: continuously outside for the configured wall-clock
        # span, observed on enough consecutive ticks (gap-reset above keeps
        # wall-clock an honest proxy for observed continuity).
        if outside_now is True and price_outside_since is not None:
            elapsed_min = (now - price_outside_since).total_seconds() / 60.0
            if (
                elapsed_min >= config.end_outside_persistence_min
                and consecutive_price_outside >= config.end_outside_min_ticks
            ):
                end_reasons.append(f"price_outside_persistent:{elapsed_min:.0f}min")
                price_end_latched = True

    if (
        evaluation.range_prob is not None
        and evaluation.trend_prob is not None
        and evaluation.range_prob < config.min_range_prob
        and evaluation.trend_prob > config.end_trend_prob_threshold
    ):
        end_reasons.append(
            f"regime_flipped:range={evaluation.range_prob:.2f},"
            f"trend={evaluation.trend_prob:.2f}"
        )

    if consecutive_micro_failures >= config.microstructure_failures_to_end:
        end_reasons.append(
            f"microstructure_persistent_fail:{consecutive_micro_failures}x"
        )

    if consecutive_fetch_errors >= config.transient_failures_to_end:
        end_reasons.append(f"persistent_fetch_failure:{consecutive_fetch_errors}x")

    end_reasons.extend(_telemetry_end_reasons(evaluation.execution_telemetry))

    regime_reasons = [r for r in end_reasons if r.startswith("regime_flipped:")]
    if regime_reasons and len(end_reasons) == len(regime_reasons):
        telemetry_status = _telemetry_activity_status(evaluation.execution_telemetry)
        if telemetry_status == "harvesting_low_risk":
            end_reasons = []
            adjust_reasons.append("telemetry_harvesting_offsets_regime_flip")
        elif telemetry_status == "no_harvest":
            end_reasons.append("telemetry_no_harvest")

    # -- ADJUST conditions (only if not already END) -----------------------
    if not end_reasons:
        if outside_now is True and not config.end_on_first_outside_tick:
            # v1.1 watch state: outside the range but below the hysteresis
            # thresholds. The operator gets the re-centered suggested bounds
            # immediately (ADJUST machinery below) while the bot stays alive.
            # Name deliberately does NOT contain "price_outside_grid" as a
            # leading substring, so colon-less v1.0 log greps cannot match it.
            adjust_reasons.append(
                f"price_outside_watch:{evaluation.pct_inside_grid:.1f}pct"
            )
        if evaluation.transient_fetch_error:
            adjust_reasons.append("transient_fetch_error")
        if evaluation.hmm_artifact_missing:
            adjust_reasons.append("hmm_artifact_missing")
        if evaluation.micro_gate_pass is False:
            # Single-tick failure (the 3-in-a-row case escalates to END above)
            adjust_reasons.append("microstructure_fail")
        if (
            evaluation.range_prob is not None
            and config.adjust_range_prob_floor
            <= evaluation.range_prob
            < config.min_range_prob
        ):
            adjust_reasons.append(f"range_prob_borderline:{evaluation.range_prob:.2f}")
        if (
            evaluation.dist_to_lower_pct is not None
            and 0 <= evaluation.dist_to_lower_pct < config.boundary_proximity_pct
        ):
            adjust_reasons.append(
                f"price_near_lower:{evaluation.dist_to_lower_pct:.1f}pct"
            )
        if (
            evaluation.dist_to_upper_pct is not None
            and 0 <= evaluation.dist_to_upper_pct < config.boundary_proximity_pct
        ):
            adjust_reasons.append(
                f"price_near_upper:{evaluation.dist_to_upper_pct:.1f}pct"
            )
        for diag in evaluation.diagnostics:
            if diag.startswith("data_missing:"):
                adjust_reasons.append(diag)

        # -- D4: bounded caution-only meta tilt (shadow-capable) -----------
        # Only modulates an ADJUST that ALREADY exists from another reason
        # (`bool(adjust_reasons)` guard); never originates or suppresses an
        # ADJUST, and -- being inside `if not end_reasons` -- can never create
        # or cancel an END. Requires an authoritative, full-fidelity meta_proba
        # below the calibrated low threshold. `meta_would_tilt` is computed
        # every tick for shadow logging; it is only APPLIED when enabled.
        meta_would_tilt = (
            evaluation.meta_authoritative
            and evaluation.meta_full_fidelity
            and evaluation.meta_proba is not None
            and evaluation.meta_proba < config.meta_tilt_low_threshold
            and bool(adjust_reasons)
        )
        if meta_would_tilt and config.meta_tilt_enabled:
            adjust_reasons.append("meta_low_confidence")

    # -- Verdict assembly --------------------------------------------------
    if end_reasons:
        verdict = Verdict.END
        reasons = sorted(set(end_reasons))
    elif adjust_reasons:
        verdict = Verdict.ADJUST
        reasons = sorted(set(adjust_reasons))
    else:
        verdict = Verdict.CONTINUE
        reasons = []

    # Streak counter
    if history.last_verdict == verdict.value:
        consecutive = history.consecutive + 1
    else:
        consecutive = 1

    # Escalation flag (ADJUST only). When the meta tilt is applied, escalate one
    # tick sooner (more cautious) -- caution-only; it never delays escalation.
    meta_tilt_applied = meta_would_tilt and config.meta_tilt_enabled
    effective_escalate_after = (
        max(1, config.adjust_escalate_after - 1)
        if meta_tilt_applied
        else config.adjust_escalate_after
    )
    escalated = (
        verdict is Verdict.ADJUST and consecutive >= effective_escalate_after
    )

    # -- Suggested re-centered bounds (ADJUST only) ------------------------
    suggested_lower: Optional[float] = None
    suggested_upper: Optional[float] = None
    if verdict is Verdict.ADJUST and evaluation.price is not None:
        half_width = evaluation.grid_width / 2.0
        suggested_lower = evaluation.price - half_width
        suggested_upper = evaluation.price + half_width

    # -- Cool-down: should we emit this tick? ------------------------------
    # The first tick of a fresh excursion must reach the operator even when a
    # prior ADJUST streak has already escalated (which normally suppresses
    # further ADJUSTs until the verdict changes).
    force_emit = (
        verdict is Verdict.ADJUST
        and just_flipped_outside
        and not config.end_on_first_outside_tick
    )
    should_emit = _should_emit(
        verdict=verdict,
        reasons=reasons,
        escalated=escalated,
        history=history,
        config=config,
        now=now,
        force=force_emit,
    )

    # -- Build new history snapshot ----------------------------------------
    tick = TickSummary(
        evaluated_at_utc=evaluation.evaluated_at_utc,
        verdict=verdict.value,
        reasons=list(reasons),
        price=evaluation.price,
        range_prob=evaluation.range_prob,
        meta_proba=evaluation.meta_proba,
    )

    history_with_tick = append_tick(history, tick)
    new_history = BotHistory(
        last_verdict=verdict.value,
        consecutive=consecutive,
        last_emitted_at=now if should_emit else history.last_emitted_at,
        last_emitted_verdict=(
            verdict.value if should_emit else history.last_emitted_verdict
        ),
        last_reasons=list(reasons) if should_emit else list(history.last_reasons),
        last_escalated=escalated if should_emit else history.last_escalated,
        consecutive_fetch_errors=consecutive_fetch_errors,
        consecutive_micro_failures=consecutive_micro_failures,
        price_outside_since=price_outside_since,
        consecutive_price_outside=consecutive_price_outside,
        consecutive_price_inside=consecutive_price_inside,
        price_end_latched=price_end_latched,
        profit_deploy_ts=profit_deploy_ts,
        peak_total_profit_usdt=peak_total_profit_usdt,
        peak_total_profit_at=peak_total_profit_at,
        last_total_profit_usdt=last_total_profit_usdt,
        last_total_profit_at=last_total_profit_at,
        pnl_observation_count=pnl_observation_count,
        ticks=history_with_tick.ticks,
    )

    rec = Recommendation(
        verdict=verdict,
        reasons=tuple(reasons),
        suggested_grid_lower=suggested_lower,
        suggested_grid_upper=suggested_upper,
        consecutive_count=consecutive,
        escalated=escalated,
        meta_would_tilt=meta_would_tilt,
        meta_influenced_verdict=meta_tilt_applied,
        profit_deterioration=profit_evidence,
    )
    return RecommenderResult(rec, should_emit, new_history)


def _update_profit_deterioration(
    evaluation: BotEvaluation,
    history: BotHistory,
) -> tuple[
    Optional[ProfitDeteriorationEvidence],
    Optional[datetime],
    Optional[float],
    Optional[datetime],
    Optional[float],
    Optional[datetime],
    int,
]:
    """Update peak/giveback evidence only for fresh private snapshots."""

    profit_deploy_ts = history.profit_deploy_ts
    peak = history.peak_total_profit_usdt
    peak_at = history.peak_total_profit_at
    previous = history.last_total_profit_usdt
    previous_at = history.last_total_profit_at
    count = history.pnl_observation_count

    if evaluation.deploy_ts is not None and profit_deploy_ts != evaluation.deploy_ts:
        profit_deploy_ts = evaluation.deploy_ts
        peak = None
        peak_at = None
        previous = None
        previous_at = None
        count = 0

    telemetry = evaluation.execution_telemetry
    if telemetry is None or telemetry.pnl is None:
        return None, profit_deploy_ts, peak, peak_at, previous, previous_at, count
    current = telemetry.pnl.total_profit_usdt
    if current is None or not math.isfinite(current):
        return None, profit_deploy_ts, peak, peak_at, previous, previous_at, count

    captured_at = telemetry.captured_at
    status = "new"
    is_new = captured_at is not None
    if captured_at is None:
        status = "missing_timestamp"
        is_new = False
    elif evaluation.deploy_ts is not None and captured_at < evaluation.deploy_ts:
        status = "predates_deployment"
        is_new = False
    elif previous_at is not None and captured_at == previous_at:
        status = "duplicate" if previous == current else "duplicate_conflict"
        is_new = False
    elif previous_at is not None and captured_at < previous_at:
        status = "out_of_order"
        is_new = False

    prior_for_change = previous
    if is_new:
        count += 1
        previous = current
        previous_at = captured_at
        if peak is None or current > peak:
            peak = current
            peak_at = captured_at

    giveback = max(peak - current, 0.0) if peak is not None else None
    giveback_pct = (
        giveback / peak * 100.0
        if giveback is not None and peak is not None and peak > 0
        else None
    )
    pnl = telemetry.pnl
    evidence = ProfitDeteriorationEvidence(
        observation_status=status,
        captured_at_utc=captured_at,
        current_total_profit_usdt=current,
        previous_total_profit_usdt=prior_for_change,
        change_from_previous_usdt=(
            current - prior_for_change if prior_for_change is not None else None
        ),
        peak_total_profit_usdt=peak,
        peak_total_profit_at=peak_at,
        giveback_usdt=giveback,
        giveback_pct_of_positive_peak=giveback_pct,
        current_is_profitable=current > 0,
        observation_count=count,
        matched_profit_usdt=pnl.matched_profit_usdt,
        unmatched_pnl_usdt=pnl.unmatched_pnl_usdt,
        funding_fee_usdt=pnl.funding_fee_usdt,
        transaction_fee_usdt=pnl.transaction_fee_usdt,
    )
    return evidence, profit_deploy_ts, peak, peak_at, previous, previous_at, count


def _telemetry_end_reasons(
    telemetry: Optional[LiveExecutionTelemetry],
) -> list[str]:
    if telemetry is None or telemetry.pnl is None or telemetry.tp_sl is None:
        return []

    total_profit = telemetry.pnl.total_profit_usdt
    if total_profit is None:
        return []

    reasons: list[str] = []
    stop_loss = telemetry.tp_sl.stop_loss
    if (
        stop_loss is not None
        and stop_loss.pnl_usdt is not None
        and stop_loss.pnl_usdt < 0
        and total_profit <= stop_loss.pnl_usdt
    ):
        reasons.append(
            f"telemetry_stop_loss_reached:{total_profit:.2f}<={stop_loss.pnl_usdt:.2f}"
        )

    take_profit = telemetry.tp_sl.take_profit
    if (
        take_profit is not None
        and take_profit.pnl_usdt is not None
        and take_profit.pnl_usdt > 0
        and total_profit >= take_profit.pnl_usdt
    ):
        reasons.append(
            f"telemetry_take_profit_reached:{total_profit:.2f}>={take_profit.pnl_usdt:.2f}"
        )

    return reasons


def _telemetry_activity_status(
    telemetry: Optional[LiveExecutionTelemetry],
) -> Optional[str]:
    if telemetry is None or telemetry.pnl is None:
        return None

    pnl = telemetry.pnl
    has_positive_harvest = any(
        value is not None and value > 0
        for value in (
            pnl.matched_profit_usdt,
            pnl.matched_profit_pct,
            pnl.realized_profit_usdt,
        )
    )
    has_no_harvest_evidence = all(
        value is not None and value <= 0
        for value in (pnl.matched_profit_usdt, pnl.matched_profit_pct)
    )

    ladder = telemetry.open_order_ladder
    has_ladder = ladder is not None and (bool(ladder.buy) or bool(ladder.sell))

    risk_label = None
    if telemetry.risk is not None:
        risk_label = telemetry.risk.risk_label
    low_risk = bool(risk_label is not None and "low" in risk_label.lower())

    if has_positive_harvest and has_ladder and low_risk:
        return "harvesting_low_risk"
    if has_no_harvest_evidence and has_ladder:
        return "no_harvest"
    return None


def _should_emit(
    *,
    verdict: Verdict,
    reasons: list[str],
    escalated: bool,
    history: BotHistory,
    config: RecommenderConfig,
    now: datetime,
    force: bool = False,
) -> bool:
    """Apply the cool-down policy. See LIVE_DECISION.md §Concurrency, state, cool-down."""
    if config.emit_every_tick:
        return True

    # Rule 0 (v1.1): caller-forced emission (first tick of a fresh price
    # excursion) bypasses the escalated-ADJUST suppression below.
    if force:
        return True

    # Rule 1: first tick ever -- always emit (baseline)
    if history.last_verdict is None:
        return True

    # Rule 2: verdict transition -- always emit
    if history.last_verdict != verdict.value:
        return True

    # Same verdict as the previous tick.
    if verdict is Verdict.END:
        if history.last_emitted_at is None:
            return True
        elapsed_min = (now - history.last_emitted_at).total_seconds() / 60.0
        return elapsed_min >= config.end_cooldown_min

    if verdict is Verdict.ADJUST:
        # Newly escalated this tick (e.g. 3rd consecutive ADJUST)
        if escalated and not history.last_escalated:
            return True
        # Already escalated -- suppress until the verdict changes
        if history.last_escalated:
            return False
        # Not yet escalated and reasons changed mid-stream -- emit
        if set(reasons) != set(history.last_reasons):
            return True
        return False

    # CONTINUE -- heartbeat only
    if verdict is Verdict.CONTINUE:
        if history.last_emitted_at is None:
            return True
        elapsed_min = (now - history.last_emitted_at).total_seconds() / 60.0
        return elapsed_min >= config.continue_heartbeat_min

    return False  # unreachable
