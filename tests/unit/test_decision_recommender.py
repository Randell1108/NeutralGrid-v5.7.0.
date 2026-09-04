"""Tests for live.decision.recommender (Phase A) — pure verdict logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from neutralgrid.live.decision.loader import (
    LiveExecutionTelemetry,
    OpenOrderLadder,
    OpenOrderLadderEntry,
    PnlTelemetry,
    RiskTelemetry,
    TpSlLeg,
    TpSlTelemetry,
)
from neutralgrid.live.decision.execution_risk import ExecutionRiskEvidence
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    RecommenderConfig,
    Verdict,
    decide,
)
from neutralgrid.live.decision.private_events import PrivateEventEvidence
from neutralgrid.live.decision.state_store import BotHistory, TickSummary


_T0 = datetime(2026, 5, 6, 14, 0, 0, tzinfo=timezone.utc)
_DEFAULT_CFG = RecommenderConfig()


def _tick(at: datetime, verdict: str = "ADJUST") -> TickSummary:
    """Minimal recorded tick, used to anchor the v1.1 gap-staleness guard."""
    return TickSummary(
        evaluated_at_utc=at, verdict=verdict, reasons=[], price=65_000.0,
        range_prob=None, meta_proba=None,
    )


def _eval(**overrides: Any) -> BotEvaluation:
    """Build a healthy CONTINUE-shaped BotEvaluation, override fields as needed."""
    base: dict[str, Any] = dict(
        symbol="BTCUSDT",
        evaluated_at_utc=_T0,
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        price=65_000.0,
        pct_inside_grid=50.0,
        dist_to_lower_pct=50.0,
        dist_to_upper_pct=50.0,
        range_prob=0.70,
        trend_prob=0.20,
        persistence_prob=0.60,
        meta_proba=0.55,
        utility_score=0.10,
        micro_gate_pass=True,
    )
    base.update(overrides)
    return BotEvaluation(**base)


def _harvesting_telemetry() -> LiveExecutionTelemetry:
    return LiveExecutionTelemetry(
        pnl=PnlTelemetry(
            total_profit_usdt=0.05,
            matched_profit_usdt=0.06,
            matched_profit_pct=1.11,
            realized_profit_usdt=0.04,
            unmatched_pnl_usdt=-0.01,
        ),
        open_order_ladder=OpenOrderLadder(
            buy=(OpenOrderLadderEntry(side="buy", level=1, price=0.11294),),
            sell=(OpenOrderLadderEntry(side="sell", level=1, price=0.11445),),
        ),
        risk=RiskTelemetry(risk_label="Low Risk", risk_ratio=2.3),
        tp_sl=TpSlTelemetry(
            stop_loss=TpSlLeg(pnl_usdt=-0.60, roi_pct=-10.0, price_type="Mark"),
            take_profit=TpSlLeg(pnl_usdt=0.90, roi_pct=15.0, price_type="Mark"),
        ),
    )


def _no_harvest_telemetry() -> LiveExecutionTelemetry:
    return LiveExecutionTelemetry(
        pnl=PnlTelemetry(
            total_profit_usdt=0.0,
            matched_profit_usdt=0.0,
            matched_profit_pct=0.0,
            realized_profit_usdt=0.0,
        ),
        open_order_ladder=OpenOrderLadder(
            buy=(OpenOrderLadderEntry(side="buy", level=1, price=0.11294),),
            sell=(OpenOrderLadderEntry(side="sell", level=1, price=0.11445),),
        ),
        risk=RiskTelemetry(risk_label="Low Risk", risk_ratio=2.3),
    )


# ── Verdict path coverage ────────────────────────────────────────────────────


def test_first_tick_continue_emits_baseline() -> None:
    res = decide(_eval(), BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.CONTINUE
    assert res.recommendation.reasons == ()
    assert res.should_emit is True  # first tick always emits
    assert res.new_history.last_verdict == "CONTINUE"
    assert res.new_history.consecutive == 1


def test_private_event_evidence_is_observational_and_verdict_inert() -> None:
    evidence = PrivateEventEvidence(
        source="canonical_private_event_stream",
        observed_at_utc=_T0,
        age_seconds=0.0,
        run_id="run-1",
        symbol="BTCUSDT",
        strategy_id="strategy-1",
        capture_mode="binance_user_data_stream",
        event_completeness="event_complete",
        source_scopes=("orders", "trades", "income"),
        manifest_total_records=4,
        records_in_window=4,
        history_coverage_seconds=300.0,
        order_update_count=2,
        order_status_counts={"FILLED": 2},
        trade_fill_count=2,
        unique_trade_order_count=2,
        maker_fill_count=2,
        taker_fill_count=0,
        unknown_liquidity_fill_count=0,
        buy_fill_notional_usdt=100.0,
        sell_fill_notional_usdt=101.0,
        realized_pnl_usdt=1.0,
        commission_usdt=-0.04,
        funding_fee_usdt=-0.01,
        other_income_usdt=0.0,
        duplicate_records_dropped=0,
        rejected_records=0,
        first_event_at_utc=_T0 - timedelta(minutes=5),
        last_event_at_utc=_T0,
        last_fill_at_utc=_T0,
    )

    baseline = decide(_eval(), BotHistory.empty(), _DEFAULT_CFG, _T0)
    observed = decide(
        _eval(private_event_evidence=evidence),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )

    assert observed.recommendation.verdict is baseline.recommendation.verdict
    assert observed.recommendation.reasons == baseline.recommendation.reasons
    assert observed.recommendation.suggested_grid_lower == (
        baseline.recommendation.suggested_grid_lower
    )
    assert observed.recommendation.suggested_grid_upper == (
        baseline.recommendation.suggested_grid_upper
    )


def test_execution_risk_evidence_is_observational_and_verdict_inert() -> None:
    evidence = ExecutionRiskEvidence(
        source="sequence_linked_l2_public_private_events",
        captured_at_utc=_T0,
        l2_run_id="l2-run",
        l2_segment_id="segment-1",
        strategy_id="strategy-1",
        l2_snapshot_count=60,
        history_coverage_seconds=300.0,
        position_side="long",
        exit_book_side="bids",
        position_notional_usdt=100_000.0,
        current_spread_bps=20.0,
        baseline_spread_median_bps=2.0,
        current_exit_depth_notional_usdt=100.0,
        baseline_exit_depth_median_usdt=10_000.0,
        exit_depth_current_to_baseline=0.01,
        exit_side_imbalance=-0.95,
        recent_observation_count=30,
        recent_spread_worse_fraction=1.0,
        recent_exit_depth_worse_fraction=1.0,
        sustained_spread_deterioration=True,
        sustained_exit_depth_deterioration=True,
        sustained_joint_deterioration=True,
        temporary_joint_deterioration=False,
        joint_deterioration_trailing_duration_seconds=300.0,
        deterioration_min_duration_seconds=60.0,
        deterioration_min_observations=3,
        deterioration_fraction_threshold=0.8,
        liquidity_state="sustained_joint_deterioration",
        public_trade_identity_status="exact_collector_target",
        public_trade_status="available",
        public_trade_count=20,
        public_trade_notional_usdt=50_000.0,
        aggressive_exit_side_trade_notional_usdt=45_000.0,
        aggressive_exit_side_trade_to_position_ratio=0.45,
        exit_side_removed_notional_usdt=60_000.0,
        exit_side_removed_to_position_ratio=0.60,
        exit_side_added_notional_usdt=1_000.0,
        exit_side_added_to_position_ratio=0.01,
        exit_side_net_withdrawal_notional_usdt=59_000.0,
        exit_side_net_withdrawal_to_position_ratio=0.59,
        trade_aligned_removal_proxy_usdt=45_000.0,
        trade_aligned_removal_to_position_ratio=0.45,
        unexplained_removal_proxy_usdt=15_000.0,
        unexplained_removal_to_position_ratio=0.15,
        refill_proxy_usdt=500.0,
        refill_to_position_ratio=0.005,
        sweep_proxy_interval_count=12,
        private_event_status="available",
        private_fill_count=3,
        private_order_update_count=10,
        private_cancel_update_count=8,
        private_cancel_update_fraction=0.8,
        fill_l2_linked_count=3,
        fill_l2_unlinked_count=0,
        mean_estimated_slippage_bps=15.0,
        p90_estimated_slippage_bps=25.0,
        mean_adverse_selection_5s_bps=10.0,
        mean_adverse_selection_30s_bps=30.0,
        latest_fill_estimate=None,
    )

    baseline = decide(_eval(), BotHistory.empty(), _DEFAULT_CFG, _T0)
    observed = decide(
        _eval(execution_risk=evidence),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )

    assert observed.recommendation.verdict is baseline.recommendation.verdict
    assert observed.recommendation.reasons == baseline.recommendation.reasons
    assert observed.recommendation.suggested_grid_lower == (
        baseline.recommendation.suggested_grid_lower
    )
    assert observed.recommendation.suggested_grid_upper == (
        baseline.recommendation.suggested_grid_upper
    )


def test_profit_deterioration_tracks_peak_and_giveback_without_changing_verdict() -> None:
    deploy_ts = _T0 - timedelta(hours=1)

    def evaluation(total: float, at: datetime) -> BotEvaluation:
        return _eval(
            evaluated_at_utc=at,
            deploy_ts=deploy_ts,
            execution_telemetry=LiveExecutionTelemetry(
                source="dedicated_chrome_cdp",
                captured_at=at,
                pnl=PnlTelemetry(
                    total_profit_usdt=total,
                    matched_profit_usdt=6.37,
                    unmatched_pnl_usdt=total - 6.37,
                    funding_fee_usdt=0.0,
                    transaction_fee_usdt=-0.48914617,
                ),
            ),
        )

    first = decide(evaluation(2.12, _T0), BotHistory.empty(), _DEFAULT_CFG, _T0)
    peak = decide(
        evaluation(6.74, _T0 + timedelta(minutes=5)),
        first.new_history,
        _DEFAULT_CFG,
        _T0 + timedelta(minutes=5),
    )
    deteriorated = decide(
        evaluation(0.90, _T0 + timedelta(minutes=10)),
        peak.new_history,
        _DEFAULT_CFG,
        _T0 + timedelta(minutes=10),
    )

    evidence = deteriorated.recommendation.profit_deterioration
    assert deteriorated.recommendation.verdict is Verdict.CONTINUE
    assert evidence is not None
    assert evidence.observation_status == "new"
    assert evidence.peak_total_profit_usdt == 6.74
    assert evidence.giveback_usdt == pytest.approx(5.84)
    assert evidence.giveback_pct_of_positive_peak == pytest.approx(86.64688427)
    assert evidence.current_is_profitable is True
    assert deteriorated.new_history.pnl_observation_count == 3

    duplicate = decide(
        evaluation(0.90, _T0 + timedelta(minutes=10)),
        deteriorated.new_history,
        _DEFAULT_CFG,
        _T0 + timedelta(minutes=11),
    )
    assert duplicate.recommendation.profit_deterioration is not None
    assert duplicate.recommendation.profit_deterioration.observation_status == "duplicate"
    assert duplicate.new_history.pnl_observation_count == 3


def test_price_outside_first_tick_yields_watch_adjust() -> None:
    """v1.1: a single outside tick is ADJUST (watch + recenter suggestion),
    not END. GATEFIX-02 replay evidence: the single-tick END had exit
    precision 0.38-0.47 and destroyed winners."""
    res = decide(
        _eval(price=58_000.0, pct_inside_grid=-20.0, dist_to_lower_pct=None, dist_to_upper_pct=None),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert any(r.startswith("price_outside_watch:") for r in res.recommendation.reasons)
    # recenter suggestion accompanies the watch
    assert res.recommendation.suggested_grid_lower == pytest.approx(58_000.0 - 5_000.0)
    assert res.recommendation.suggested_grid_upper == pytest.approx(58_000.0 + 5_000.0)
    # persistence state started
    assert res.new_history.consecutive_price_outside == 1
    assert res.new_history.price_outside_since == _T0


def test_price_outside_grid_yields_end_legacy_flag() -> None:
    """end_on_first_outside_tick=True restores the exact v1.0 behavior."""
    legacy = RecommenderConfig(end_on_first_outside_tick=True)
    res = decide(
        _eval(price=58_000.0, pct_inside_grid=-20.0, dist_to_lower_pct=None, dist_to_upper_pct=None),
        BotHistory.empty(),
        legacy,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_outside_grid:") for r in res.recommendation.reasons)
    # counters are maintained even in legacy mode (flag toggles must not
    # destroy accrued state)
    assert res.new_history.consecutive_price_outside == 1
    assert res.new_history.price_outside_since == _T0


def test_regime_flip_yields_end() -> None:
    res = decide(
        _eval(range_prob=0.30, trend_prob=0.55),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("regime_flipped:") for r in res.recommendation.reasons)


def test_regime_flip_with_harvesting_low_risk_telemetry_yields_adjust() -> None:
    res = decide(
        _eval(
            range_prob=0.30,
            trend_prob=0.55,
            execution_telemetry=_harvesting_telemetry(),
        ),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "telemetry_harvesting_offsets_regime_flip" in res.recommendation.reasons


def test_regime_flip_without_harvest_keeps_end() -> None:
    res = decide(
        _eval(
            range_prob=0.30,
            trend_prob=0.55,
            execution_telemetry=_no_harvest_telemetry(),
        ),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert "telemetry_no_harvest" in res.recommendation.reasons


def test_harvesting_telemetry_does_not_override_price_end() -> None:
    """Price-evidence ENDs are not softenable by harvesting telemetry —
    same invariant as v1.0, now exercised through the persistence trigger."""
    hist = BotHistory(
        last_verdict="ADJUST",
        consecutive=2,
        price_outside_since=_T0 - timedelta(minutes=200),
        consecutive_price_outside=10,
        ticks=[_tick(_T0 - timedelta(minutes=5))],
    )
    res = decide(
        _eval(
            price=58_000.0,
            pct_inside_grid=-20.0,
            dist_to_lower_pct=None,
            dist_to_upper_pct=None,
            range_prob=0.30,
            trend_prob=0.55,
            execution_telemetry=_harvesting_telemetry(),
        ),
        hist,
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_outside_persistent:") for r in res.recommendation.reasons)


def test_harvesting_telemetry_does_not_override_price_outside_grid_legacy() -> None:
    legacy = RecommenderConfig(end_on_first_outside_tick=True)
    res = decide(
        _eval(
            price=58_000.0,
            pct_inside_grid=-20.0,
            dist_to_lower_pct=None,
            dist_to_upper_pct=None,
            range_prob=0.30,
            trend_prob=0.55,
            execution_telemetry=_harvesting_telemetry(),
        ),
        BotHistory.empty(),
        legacy,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_outside_grid:") for r in res.recommendation.reasons)


def test_telemetry_stop_loss_reached_yields_end() -> None:
    telemetry = LiveExecutionTelemetry(
        pnl=PnlTelemetry(total_profit_usdt=-0.61),
        tp_sl=TpSlTelemetry(stop_loss=TpSlLeg(pnl_usdt=-0.60, price_type="Mark")),
    )
    res = decide(
        _eval(execution_telemetry=telemetry),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("telemetry_stop_loss_reached:") for r in res.recommendation.reasons)


def test_symbol_unavailable_yields_end() -> None:
    res = decide(
        _eval(symbol_unavailable=True),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert "symbol_unavailable" in res.recommendation.reasons


def test_borderline_range_prob_yields_adjust_with_recentered_bounds() -> None:
    res = decide(
        _eval(range_prob=0.40, trend_prob=0.20),  # in [0.30, 0.45) → ADJUST
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert any(r.startswith("range_prob_borderline:") for r in res.recommendation.reasons)
    # Suggested bounds preserve width, recenter on price 65000, width 10000 → [60000, 70000]
    assert res.recommendation.suggested_grid_lower == pytest.approx(60_000.0)
    assert res.recommendation.suggested_grid_upper == pytest.approx(70_000.0)


def test_price_near_upper_yields_adjust() -> None:
    res = decide(
        _eval(dist_to_upper_pct=5.0, dist_to_lower_pct=95.0),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert any(r.startswith("price_near_upper:") for r in res.recommendation.reasons)


def test_data_missing_diagnostic_yields_adjust() -> None:
    res = decide(
        _eval(diagnostics=("data_missing:1h",)),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "data_missing:1h" in res.recommendation.reasons


def test_hmm_artifact_missing_yields_adjust_not_silent_continue() -> None:
    """Fail-closed: a missing HMM artifact must not silently emit CONTINUE."""
    res = decide(
        _eval(hmm_artifact_missing=True, range_prob=None, trend_prob=None),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "hmm_artifact_missing" in res.recommendation.reasons


def test_micro_gate_single_failure_is_adjust() -> None:
    res = decide(
        _eval(micro_gate_pass=False),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "microstructure_fail" in res.recommendation.reasons


def test_micro_gate_three_consecutive_failures_escalates_to_end() -> None:
    cfg = RecommenderConfig(microstructure_failures_to_end=3)
    history = BotHistory.empty()
    eval_failing = _eval(micro_gate_pass=False)

    # First two failures → ADJUST, counters incrementing
    for i in range(2):
        res = decide(eval_failing, history, cfg, _T0 + timedelta(minutes=5 * i))
        assert res.recommendation.verdict is Verdict.ADJUST
        history = res.new_history

    # Third failure → END
    res = decide(eval_failing, history, cfg, _T0 + timedelta(minutes=15))
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("microstructure_persistent_fail:") for r in res.recommendation.reasons)


def test_three_consecutive_fetch_errors_escalates_to_end() -> None:
    cfg = RecommenderConfig(transient_failures_to_end=3)
    history = BotHistory.empty()
    eval_fetch_err = _eval(transient_fetch_error=True)

    for i in range(2):
        res = decide(eval_fetch_err, history, cfg, _T0 + timedelta(minutes=5 * i))
        assert res.recommendation.verdict is Verdict.ADJUST
        history = res.new_history

    res = decide(eval_fetch_err, history, cfg, _T0 + timedelta(minutes=15))
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("persistent_fetch_failure:") for r in res.recommendation.reasons)


# ── Cool-down / emission policy ──────────────────────────────────────────────


def test_continue_heartbeat_suppresses_then_emits() -> None:
    cfg = RecommenderConfig(continue_heartbeat_min=60)
    h = BotHistory.empty()

    # First tick: emits baseline
    res = decide(_eval(), h, cfg, _T0)
    assert res.should_emit is True
    h = res.new_history

    # 5 minutes later: same CONTINUE → suppressed
    res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=5))
    assert res.should_emit is False
    h = res.new_history

    # 65 minutes later: heartbeat elapsed → emit
    res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=65))
    assert res.should_emit is True


def test_emit_every_tick_overrides_continue_heartbeat() -> None:
    cfg = RecommenderConfig(continue_heartbeat_min=60, emit_every_tick=True)
    h = BotHistory.empty()

    res = decide(_eval(), h, cfg, _T0)
    assert res.should_emit is True
    h = res.new_history

    res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=5))
    assert res.recommendation.verdict is Verdict.CONTINUE
    assert res.should_emit is True


def test_end_cooldown_suppresses_then_emits() -> None:
    cfg = RecommenderConfig(end_cooldown_min=30)
    h = BotHistory.empty()
    end_eval = _eval(symbol_unavailable=True)

    # First END: emits
    res = decide(end_eval, h, cfg, _T0)
    assert res.should_emit is True
    h = res.new_history

    # 10 min later: still END, within cool-down → suppressed
    res = decide(end_eval, h, cfg, _T0 + timedelta(minutes=10))
    assert res.should_emit is False
    h = res.new_history

    # 35 min later: cool-down elapsed → emit
    res = decide(end_eval, h, cfg, _T0 + timedelta(minutes=35))
    assert res.should_emit is True


def test_adjust_escalates_at_third_consecutive_then_suppresses() -> None:
    cfg = RecommenderConfig(adjust_escalate_after=3)
    h = BotHistory.empty()
    adjust_eval = _eval(range_prob=0.40, trend_prob=0.20)  # ADJUST via borderline range_prob

    # Tick 1: ADJUST, emit (transition from None)
    res = decide(adjust_eval, h, cfg, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST
    assert res.recommendation.consecutive_count == 1
    assert res.recommendation.escalated is False
    assert res.should_emit is True
    h = res.new_history

    # Tick 2: same ADJUST + same reasons → suppressed (not yet escalated)
    res = decide(adjust_eval, h, cfg, _T0 + timedelta(minutes=5))
    assert res.recommendation.consecutive_count == 2
    assert res.recommendation.escalated is False
    assert res.should_emit is False
    h = res.new_history

    # Tick 3: third consecutive ADJUST → escalation, emit
    res = decide(adjust_eval, h, cfg, _T0 + timedelta(minutes=10))
    assert res.recommendation.consecutive_count == 3
    assert res.recommendation.escalated is True
    assert res.should_emit is True
    h = res.new_history

    # Tick 4: still ADJUST, already escalated → suppressed
    res = decide(adjust_eval, h, cfg, _T0 + timedelta(minutes=15))
    assert res.recommendation.escalated is True
    assert res.should_emit is False


def test_adjust_reasons_changing_mid_stream_emits() -> None:
    cfg = RecommenderConfig(adjust_escalate_after=10)  # high so we don't escalate
    h = BotHistory.empty()

    # Tick 1: ADJUST due to borderline range_prob
    e1 = _eval(range_prob=0.40, trend_prob=0.20)
    res = decide(e1, h, cfg, _T0)
    assert res.should_emit is True
    h = res.new_history

    # Tick 2: still ADJUST but new reason (price near upper) — emit
    e2 = _eval(range_prob=0.40, trend_prob=0.20, dist_to_upper_pct=5.0, dist_to_lower_pct=95.0)
    res = decide(e2, h, cfg, _T0 + timedelta(minutes=5))
    assert res.recommendation.verdict is Verdict.ADJUST
    assert res.should_emit is True


def test_transition_always_emits() -> None:
    cfg = RecommenderConfig()
    h = BotHistory.empty()

    # Tick 1: CONTINUE
    res = decide(_eval(), h, cfg, _T0)
    h = res.new_history

    # Tick 2: ADJUST — transition, emit even if 1 minute later
    res = decide(_eval(range_prob=0.40, trend_prob=0.20), h, cfg, _T0 + timedelta(minutes=1))
    assert res.recommendation.verdict is Verdict.ADJUST
    assert res.should_emit is True


def test_reasons_are_deterministically_ordered() -> None:
    """Reasons must be sorted so digest diffs are stable."""
    e = _eval(
        range_prob=0.40,  # range_prob_borderline
        trend_prob=0.20,
        dist_to_upper_pct=5.0,  # price_near_upper
        dist_to_lower_pct=95.0,
        diagnostics=("data_missing:1h",),
    )
    res = decide(e, BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST
    # sorted() ordering is deterministic
    assert list(res.recommendation.reasons) == sorted(set(res.recommendation.reasons))


def test_end_takes_precedence_over_adjust_conditions() -> None:
    """If both END and ADJUST conditions fire on the same tick, END wins."""
    e = _eval(
        symbol_unavailable=True,  # END
        range_prob=0.40,  # would be ADJUST
        trend_prob=0.20,
    )
    res = decide(e, BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.END
    # No ADJUST reasons leaked
    assert not any(r.startswith("range_prob_borderline:") for r in res.recommendation.reasons)


def test_history_counters_advance_correctly() -> None:
    cfg = RecommenderConfig()
    h = BotHistory.empty()

    # Two consecutive fetch errors → counter at 2
    e_err = _eval(transient_fetch_error=True)
    res = decide(e_err, h, cfg, _T0)
    h = res.new_history
    assert h.consecutive_fetch_errors == 1

    res = decide(e_err, h, cfg, _T0 + timedelta(minutes=5))
    h = res.new_history
    assert h.consecutive_fetch_errors == 2

    # Now a clean tick → counter resets
    res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=10))
    h = res.new_history
    assert h.consecutive_fetch_errors == 0


def test_suggested_bounds_only_for_adjust_with_price() -> None:
    # CONTINUE → no suggestions
    res = decide(_eval(), BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.suggested_grid_lower is None
    assert res.recommendation.suggested_grid_upper is None

    # END → no suggestions
    res = decide(
        _eval(symbol_unavailable=True),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.suggested_grid_lower is None

    # ADJUST without price → no suggestions
    res = decide(
        _eval(range_prob=0.40, trend_prob=0.20, price=None, pct_inside_grid=None,
              dist_to_lower_pct=None, dist_to_upper_pct=None),
        BotHistory.empty(),
        _DEFAULT_CFG,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert res.recommendation.suggested_grid_lower is None


# ── D4: bounded caution-only meta tilt (shadow + enabled) ────────────────────


def _adjust_meta_eval(**overrides: Any) -> BotEvaluation:
    """Borderline-ADJUST eval with an authoritative, full-fidelity, LOW meta_proba."""
    base: dict[str, Any] = dict(
        range_prob=0.40,  # 0.30 <= 0.40 < 0.45 -> range_prob_borderline ADJUST
        trend_prob=0.20,  # < end_trend_prob_threshold -> no regime_flipped END
        meta_proba=0.20,
        meta_authoritative=True,
        meta_full_fidelity=True,
    )
    base.update(overrides)
    return _eval(**base)


_TILT_ON = RecommenderConfig(meta_tilt_enabled=True, meta_tilt_low_threshold=0.50)


def test_meta_tilt_adds_reason_when_enabled_and_authoritative() -> None:
    res = decide(_adjust_meta_eval(), BotHistory.empty(), _TILT_ON, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "meta_low_confidence" in res.recommendation.reasons
    assert res.recommendation.meta_would_tilt is True
    assert res.recommendation.meta_influenced_verdict is True


def test_meta_tilt_shadow_logged_but_inert_when_disabled() -> None:
    """Default (disabled): tilt is COMPUTED for shadow logging but never applied."""
    res = decide(
        _adjust_meta_eval(),
        BotHistory.empty(),
        RecommenderConfig(meta_tilt_low_threshold=0.50),  # enabled defaults False
        _T0,
    )
    assert res.recommendation.verdict is Verdict.ADJUST
    assert "meta_low_confidence" not in res.recommendation.reasons
    assert res.recommendation.meta_would_tilt is True  # shadow signal recorded
    assert res.recommendation.meta_influenced_verdict is False


def test_meta_tilt_requires_authority() -> None:
    res = decide(
        _adjust_meta_eval(meta_authoritative=False),
        BotHistory.empty(),
        _TILT_ON,
        _T0,
    )
    assert res.recommendation.meta_would_tilt is False
    assert "meta_low_confidence" not in res.recommendation.reasons


def test_meta_tilt_requires_full_fidelity() -> None:
    res = decide(
        _adjust_meta_eval(meta_full_fidelity=False),
        BotHistory.empty(),
        _TILT_ON,
        _T0,
    )
    assert res.recommendation.meta_would_tilt is False
    assert "meta_low_confidence" not in res.recommendation.reasons


def test_meta_tilt_never_originates_adjust() -> None:
    """A healthy CONTINUE with low meta_proba stays CONTINUE -- the tilt only
    modulates an ADJUST that already exists."""
    res = decide(
        _eval(meta_proba=0.20, meta_authoritative=True, meta_full_fidelity=True),
        BotHistory.empty(),
        _TILT_ON,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.CONTINUE
    assert res.recommendation.meta_would_tilt is False
    assert "meta_low_confidence" not in res.recommendation.reasons


@pytest.mark.parametrize(
    "end_overrides",
    [
        {"symbol_unavailable": True},
        {"range_prob": 0.30, "trend_prob": 0.50},  # regime_flipped END
        # v1.1: price END fixture is the displacement disaster stop
        # (|50k-65k|/5k = 3.0 >= 2.0); a mere single tick outside is now a
        # watch-ADJUST, covered by test_price_outside_first_tick_yields_watch_adjust.
        {"price": 50_000.0, "pct_inside_grid": -100.0,
         "dist_to_lower_pct": None, "dist_to_upper_pct": None},  # price displacement END
    ],
)
def test_meta_tilt_never_flips_end(end_overrides: dict[str, Any]) -> None:
    """No low meta_proba can cancel or alter an END verdict (END layer is untouchable)."""
    res = decide(
        _eval(meta_proba=0.01, meta_authoritative=True, meta_full_fidelity=True, **end_overrides),
        BotHistory.empty(),
        _TILT_ON,
        _T0,
    )
    assert res.recommendation.verdict is Verdict.END
    assert res.recommendation.meta_would_tilt is False
    assert res.recommendation.meta_influenced_verdict is False
    assert "meta_low_confidence" not in res.recommendation.reasons


def test_meta_tilt_default_threshold_is_inert_even_if_enabled() -> None:
    """meta_tilt_low_threshold defaults to 0.0; an authoritative meta_proba can
    never be < 0.0, so flipping only the enable flag changes nothing."""
    res = decide(
        _adjust_meta_eval(meta_proba=0.20),
        BotHistory.empty(),
        RecommenderConfig(meta_tilt_enabled=True),  # threshold defaults 0.0
        _T0,
    )
    assert res.recommendation.meta_would_tilt is False
    assert "meta_low_confidence" not in res.recommendation.reasons


def test_meta_tilt_escalates_adjust_one_tick_sooner() -> None:
    eval_b = _adjust_meta_eval()

    # Disabled baseline: 2nd consecutive ADJUST is NOT escalated (needs 3rd).
    cfg_off = RecommenderConfig(adjust_escalate_after=3)
    h = BotHistory.empty()
    h = decide(eval_b, h, cfg_off, _T0).new_history
    r2_off = decide(eval_b, h, cfg_off, _T0 + timedelta(minutes=5))
    assert r2_off.recommendation.consecutive_count == 2
    assert r2_off.recommendation.escalated is False

    # Enabled: caution-only tilt escalates one tick sooner (effective 2).
    cfg_on = RecommenderConfig(
        adjust_escalate_after=3, meta_tilt_enabled=True, meta_tilt_low_threshold=0.50
    )
    h = BotHistory.empty()
    r1_on = decide(eval_b, h, cfg_on, _T0)
    assert r1_on.recommendation.escalated is False  # consecutive 1 < effective 2
    h = r1_on.new_history
    r2_on = decide(eval_b, h, cfg_on, _T0 + timedelta(minutes=5))
    assert r2_on.recommendation.consecutive_count == 2
    assert r2_on.recommendation.escalated is True
    assert r2_on.recommendation.meta_influenced_verdict is True
    assert "meta_low_confidence" in r2_on.recommendation.reasons


# -- v1.1 hysteresis END state machine (GATEFIX-02) ---------------------------


def _outside_eval(**overrides: Any) -> BotEvaluation:
    """Outside-the-range evaluation: price below lower bound, no disaster
    displacement (|58k - 65k| / 5k = 1.4 < 2.0)."""
    base: dict[str, Any] = dict(price=58_000.0, pct_inside_grid=-20.0,
                                dist_to_lower_pct=None, dist_to_upper_pct=None)
    base.update(overrides)
    return _eval(**base)


def test_persistence_end_requires_both_elapsed_and_ticks() -> None:
    # elapsed satisfied, ticks not: 181 min but only 2 observed ticks
    h = BotHistory(last_verdict="ADJUST", consecutive=1,
                   price_outside_since=_T0 - timedelta(minutes=181),
                   consecutive_price_outside=1,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    res = decide(_outside_eval(), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST
    assert res.new_history.consecutive_price_outside == 2

    # ticks satisfied, elapsed not: 100 min outside across 25 ticks
    h = BotHistory(last_verdict="ADJUST", consecutive=1,
                   price_outside_since=_T0 - timedelta(minutes=100),
                   consecutive_price_outside=24,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    res = decide(_outside_eval(), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST

    # both satisfied -> END, latched
    h = BotHistory(last_verdict="ADJUST", consecutive=1,
                   price_outside_since=_T0 - timedelta(minutes=181),
                   consecutive_price_outside=24,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    res = decide(_outside_eval(), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_outside_persistent:") for r in res.recommendation.reasons)
    assert res.new_history.price_end_latched is True


def test_displacement_end_fires_immediately_and_latches() -> None:
    # price 3 half-widths below center: |50k-65k|/5k = 3.0 >= 2.0
    res = decide(_eval(price=50_000.0, pct_inside_grid=-100.0,
                       dist_to_lower_pct=None, dist_to_upper_pct=None),
                 BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_displacement:3.00x") for r in res.recommendation.reasons)
    assert res.new_history.price_end_latched is True


def test_displacement_boundary_value_exact_multiple_fires() -> None:
    # exactly 2.0 half-widths: price 55k, center 65k, half 5k
    res = decide(_eval(price=55_000.0, pct_inside_grid=-50.0,
                       dist_to_lower_pct=None, dist_to_upper_pct=None),
                 BotHistory.empty(), _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.END
    assert any(r.startswith("price_displacement:2.00x") for r in res.recommendation.reasons)


def test_inside_tick_resets_persistence_state() -> None:
    h = BotHistory(last_verdict="ADJUST", consecutive=2,
                   price_outside_since=_T0 - timedelta(minutes=100),
                   consecutive_price_outside=20,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    res = decide(_eval(), h, _DEFAULT_CFG, _T0)  # healthy inside eval
    assert res.new_history.price_outside_since is None
    assert res.new_history.consecutive_price_outside == 0
    assert res.new_history.consecutive_price_inside == 1


def test_none_and_nonfinite_price_carries_state_unchanged() -> None:
    since = _T0 - timedelta(minutes=100)
    h = BotHistory(last_verdict="ADJUST", consecutive=1,
                   price_outside_since=since, consecutive_price_outside=7,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    # missing price (transient fetch error shape)
    res = decide(_eval(price=None, pct_inside_grid=None,
                       dist_to_lower_pct=None, dist_to_upper_pct=None,
                       range_prob=None, trend_prob=None,
                       transient_fetch_error=True),
                 h, _DEFAULT_CFG, _T0)
    assert res.new_history.price_outside_since == since
    assert res.new_history.consecutive_price_outside == 7
    # non-finite price must behave like missing, not like a displacement END
    res = decide(_eval(price=float("inf"), pct_inside_grid=float("inf"),
                       dist_to_lower_pct=None, dist_to_upper_pct=None),
                 h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is not Verdict.END
    assert res.new_history.price_outside_since == since
    assert res.new_history.consecutive_price_outside == 7


def test_gap_staleness_resets_accumulation() -> None:
    """Unobserved time (scanner outage) must not count as continuously
    outside: gap > end_state_max_gap_min resets the accumulators."""
    h = BotHistory(last_verdict="ADJUST", consecutive=1,
                   price_outside_since=_T0 - timedelta(minutes=300),
                   consecutive_price_outside=50,
                   ticks=[_tick(_T0 - timedelta(minutes=120))])  # 2h blackout
    res = decide(_outside_eval(), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST  # NOT an instant END
    assert res.new_history.consecutive_price_outside == 1
    assert res.new_history.price_outside_since == _T0


def test_stale_state_from_previous_deployment_is_discarded() -> None:
    """A reused strategy_id state key must not carry a previous deployment
    outside-timer into a fresh deployment."""
    deploy_ts = _T0 - timedelta(minutes=10)
    h = BotHistory(last_verdict="END", consecutive=3,
                   price_outside_since=_T0 - timedelta(days=5),
                   consecutive_price_outside=200, price_end_latched=True,
                   ticks=[_tick(_T0 - timedelta(minutes=5), verdict="END")])
    res = decide(_outside_eval(deploy_ts=deploy_ts), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST  # fresh watch, no END
    assert res.new_history.consecutive_price_outside == 1
    assert res.new_history.price_end_latched is False


def test_end_latch_holds_until_clear_ticks_inside() -> None:
    cfg = RecommenderConfig(end_latch_clear_ticks=3)
    h = BotHistory(last_verdict="END", consecutive=1, price_end_latched=True,
                   ticks=[_tick(_T0 - timedelta(minutes=5), verdict="END")])
    # inside ticks 1 and 2: still END via latch
    for i in (1, 2):
        res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=5 * i))
        assert res.recommendation.verdict is Verdict.END
        assert "price_end_latched" in res.recommendation.reasons
        h = res.new_history
    # third consecutive inside tick clears the latch
    res = decide(_eval(), h, cfg, _T0 + timedelta(minutes=15))
    assert res.recommendation.verdict is Verdict.CONTINUE
    assert res.new_history.price_end_latched is False


def test_watch_flip_tick_force_emits_despite_escalated_adjust_streak() -> None:
    """The first tick of a fresh excursion must reach the operator even when
    a prior ADJUST streak already escalated (which normally suppresses
    further ADJUST emissions until the verdict changes)."""
    h = BotHistory(last_verdict="ADJUST", consecutive=4,
                   last_emitted_at=_T0 - timedelta(minutes=5),
                   last_emitted_verdict="ADJUST",
                   last_reasons=["price_near_lower:5.0pct"],
                   last_escalated=True,
                   ticks=[_tick(_T0 - timedelta(minutes=5))])
    res = decide(_outside_eval(), h, _DEFAULT_CFG, _T0)
    assert res.recommendation.verdict is Verdict.ADJUST
    assert any(r.startswith("price_outside_watch:") for r in res.recommendation.reasons)
    assert res.should_emit is True


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RecommenderConfig(end_displacement_multiple=1.0)  # must be > 1.0
    with pytest.raises(ValueError):
        RecommenderConfig(end_displacement_multiple="2.5")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RecommenderConfig(end_displacement_multiple=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RecommenderConfig(end_outside_persistence_min=0.0)
    with pytest.raises(ValueError):
        RecommenderConfig(end_outside_min_ticks=0)
    with pytest.raises(ValueError):
        RecommenderConfig(end_state_max_gap_min=float("nan"))
    with pytest.raises(ValueError):
        RecommenderConfig(end_latch_clear_ticks=0)
    with pytest.raises(ValueError):
        RecommenderConfig(end_on_first_outside_tick="yes")  # type: ignore[arg-type]
