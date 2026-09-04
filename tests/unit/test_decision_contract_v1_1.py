"""Pinning contract test for the live decision tool's v1.1 wire format.

When `DECISION_CONTRACT_VERSION` is bumped, this test must be reviewed
against the change set. Bumping the constant without updating this test (or
updating this test without justification in CHANGELOG.md) is a contract
violation that will silently break:

* persisted state files at ``data/live_decisions/state/<key>.json``
* the JSONL audit log at ``logs/live_decisions_YYYYMMDD.jsonl``
* the Discord embed shape consumers may pattern-match on
* the recommender reason-code vocabulary downstream tools may parse

If you change any of those shapes, bump the constant AND update this test
in the same commit.

v1.0 -> v1.1 (GATEFIX-02, 2026-07-13): hysteresis END trigger. The v1.0
single-tick `price_outside_grid:` END is emitted only under the legacy
`end_on_first_outside_tick=True` config; the default vocabulary gains
`price_outside_watch:` (ADJUST) and `price_outside_persistent:` /
`price_displacement:` / `price_end_latched` (END). BotHistory gains four
additive state fields (pre-1.1 state files load with defaults). Justified in
CHANGELOG.md (GATEFIX-02) with 638 engine-verified replay deployments.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from neutralgrid.core.constants import DECISION_CONTRACT_VERSION
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    RecommenderConfig,
    Verdict,
    decide,
)
from neutralgrid.live.decision.state_store import BotHistory, TickSummary


def test_decision_contract_version_pinned_to_1_1() -> None:
    assert DECISION_CONTRACT_VERSION == "1.1", (
        "DECISION_CONTRACT_VERSION changed. Review LIVE_DECISION.md, the "
        "JSONL schema, persisted state files, and Discord-embed consumers "
        "before updating this test."
    )


def test_verdict_vocabulary_pinned() -> None:
    """Verdict values are wire-format. Renaming = contract bump."""
    assert {v.value for v in Verdict} == {"CONTINUE", "ADJUST", "END"}
    assert Verdict.CONTINUE.value == "CONTINUE"
    assert Verdict.ADJUST.value == "ADJUST"
    assert Verdict.END.value == "END"


def test_recommender_config_default_thresholds_pinned() -> None:
    """LIVE_DECISION.md publishes these defaults; changing them is a wire-shape change."""
    cfg = RecommenderConfig()
    assert cfg.min_range_prob == 0.45
    assert cfg.adjust_range_prob_floor == 0.30
    assert cfg.end_trend_prob_threshold == 0.40
    assert cfg.boundary_proximity_pct == 10.0
    assert cfg.end_cooldown_min == 30
    assert cfg.adjust_escalate_after == 3
    assert cfg.continue_heartbeat_min == 60
    assert cfg.microstructure_failures_to_end == 3
    assert cfg.transient_failures_to_end == 3
    # v1.1 hysteresis END defaults (replay-anchored; see CHANGELOG GATEFIX-02)
    assert cfg.end_on_first_outside_tick is False
    assert cfg.end_outside_persistence_min == 180.0
    assert cfg.end_outside_min_ticks == 3
    assert cfg.end_displacement_multiple == 2.0
    assert cfg.end_state_max_gap_min == 15.0
    assert cfg.end_latch_clear_ticks == 12


def test_recommender_result_field_set_pinned() -> None:
    """JSONL consumers depend on this exact field set on Recommendation."""
    eval = BotEvaluation(
        symbol="BTCUSDT",
        evaluated_at_utc=datetime(2026, 5, 7, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
    )
    res = decide(eval, BotHistory.empty(), RecommenderConfig(), datetime(2026, 5, 7, tzinfo=timezone.utc))
    expected_fields = {
        "verdict",
        "reasons",
        "suggested_grid_lower",
        "suggested_grid_upper",
        "consecutive_count",
        "escalated",
    }
    actual = {f.name for f in res.recommendation.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    assert expected_fields.issubset(actual), (
        f"Recommendation field set changed. Bump DECISION_CONTRACT_VERSION. "
        f"Missing: {expected_fields - actual}, Extra: {actual - expected_fields}"
    )


def test_bot_history_field_set_pinned() -> None:
    """Persisted state files key off this field set; renaming breaks reload."""
    fields = {f.name for f in BotHistory.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    expected = {
        "last_verdict",
        "consecutive",
        "last_emitted_at",
        "last_emitted_verdict",
        "last_reasons",
        "last_escalated",
        "consecutive_fetch_errors",
        "consecutive_micro_failures",
        # v1.1 hysteresis state (additive; absent keys default on load)
        "price_outside_since",
        "consecutive_price_outside",
        "consecutive_price_inside",
        "price_end_latched",
        "ticks",
    }
    assert expected.issubset(fields), (
        "BotHistory field set changed. Persisted state files at "
        "data/live_decisions/state/*.json will fail to round-trip. "
        f"Missing: {expected - fields}, Extra: {fields - expected}"
    )


def test_tick_summary_field_set_pinned() -> None:
    fields = {f.name for f in TickSummary.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    expected = {
        "evaluated_at_utc",
        "verdict",
        "reasons",
        "price",
        "range_prob",
        "meta_proba",
    }
    assert fields == expected, (
        f"TickSummary fields changed: missing={expected - fields}, extra={fields - expected}"
    )


def test_reason_code_prefixes_pinned() -> None:
    """Downstream consumers pattern-match on these prefixes; preserve them.

    Asserts the canonical prefix vocabulary by triggering each on synthetic input
    and checking that the prefix appears verbatim in the reasons output.
    """
    cfg = RecommenderConfig()
    now = datetime(2026, 5, 7, tzinfo=timezone.utc)
    base = dict(
        symbol="BTCUSDT",
        evaluated_at_utc=now,
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        price=65_000.0,
        pct_inside_grid=50.0,
        dist_to_lower_pct=50.0,
        dist_to_upper_pct=50.0,
    )

    # END prefixes -- v1.1 vocabulary
    # Displacement disaster stop: price 3x half-width below center.
    e_disp = BotEvaluation(**{**base, "price": 50_000.0, "pct_inside_grid": -100.0,
                              "dist_to_lower_pct": None, "dist_to_upper_pct": None})
    r = decide(e_disp, BotHistory.empty(), cfg, now)
    assert any(s.startswith("price_displacement:") for s in r.recommendation.reasons)
    assert r.recommendation.verdict is Verdict.END

    # Persistence: outside for > 180 min across >= 3 observed ticks.
    e_out = BotEvaluation(**{**base, "price": 59_000.0, "pct_inside_grid": -10.0,
                             "dist_to_lower_pct": None, "dist_to_upper_pct": None})
    hist = BotHistory(
        last_verdict="ADJUST",
        consecutive=2,
        price_outside_since=now - timedelta(minutes=181),
        consecutive_price_outside=5,
        ticks=[TickSummary(evaluated_at_utc=now - timedelta(minutes=5),
                           verdict="ADJUST", reasons=[], price=59_000.0,
                           range_prob=None, meta_proba=None)],
    )
    r = decide(e_out, hist, cfg, now)
    assert any(s.startswith("price_outside_persistent:") for s in r.recommendation.reasons)
    assert r.recommendation.verdict is Verdict.END

    # Latch: a fired price END persists as `price_end_latched`.
    e_in = BotEvaluation(**base)
    hist_latched = BotHistory(last_verdict="END", consecutive=1, price_end_latched=True,
                              ticks=[TickSummary(evaluated_at_utc=now - timedelta(minutes=5),
                                                 verdict="END", reasons=[], price=59_000.0,
                                                 range_prob=None, meta_proba=None)])
    r = decide(e_in, hist_latched, cfg, now)
    assert "price_end_latched" in r.recommendation.reasons
    assert r.recommendation.verdict is Verdict.END

    # Legacy mode: the v1.0 single-tick END reason is preserved verbatim.
    legacy_cfg = RecommenderConfig(end_on_first_outside_tick=True)
    e_outside = BotEvaluation(**{**base, "price": 58_000.0, "pct_inside_grid": -20.0,
                                 "dist_to_lower_pct": None, "dist_to_upper_pct": None})
    r = decide(e_outside, BotHistory.empty(), legacy_cfg, now)
    assert any(s.startswith("price_outside_grid:") for s in r.recommendation.reasons)
    assert r.recommendation.verdict is Verdict.END

    e_flip = BotEvaluation(**{**base, "range_prob": 0.30, "trend_prob": 0.50})
    r = decide(e_flip, BotHistory.empty(), cfg, now)
    assert any(s.startswith("regime_flipped:") for s in r.recommendation.reasons)

    e_unavail = BotEvaluation(**{**base, "symbol_unavailable": True})
    r = decide(e_unavail, BotHistory.empty(), cfg, now)
    assert "symbol_unavailable" in r.recommendation.reasons

    # ADJUST prefixes
    # v1.1 watch: first outside tick is ADJUST (not END) with the watch code,
    # whose name is deliberately NOT prefixed by "price_outside_grid".
    e_watch = BotEvaluation(**{**base, "price": 59_000.0, "pct_inside_grid": -10.0,
                               "dist_to_lower_pct": None, "dist_to_upper_pct": None})
    r = decide(e_watch, BotHistory.empty(), cfg, now)
    assert any(s.startswith("price_outside_watch:") for s in r.recommendation.reasons)
    assert r.recommendation.verdict is Verdict.ADJUST
    assert not any(s.startswith("price_outside_grid") for s in r.recommendation.reasons)

    e_borderline = BotEvaluation(**{**base, "range_prob": 0.40, "trend_prob": 0.20})
    r = decide(e_borderline, BotHistory.empty(), cfg, now)
    assert any(s.startswith("range_prob_borderline:") for s in r.recommendation.reasons)

    e_near_upper = BotEvaluation(**{**base, "dist_to_upper_pct": 5.0, "dist_to_lower_pct": 95.0})
    r = decide(e_near_upper, BotHistory.empty(), cfg, now)
    assert any(s.startswith("price_near_upper:") for s in r.recommendation.reasons)

    e_near_lower = BotEvaluation(**{**base, "dist_to_lower_pct": 5.0, "dist_to_upper_pct": 95.0})
    r = decide(e_near_lower, BotHistory.empty(), cfg, now)
    assert any(s.startswith("price_near_lower:") for s in r.recommendation.reasons)

    e_data_missing = BotEvaluation(**{**base, "diagnostics": ("data_missing:1h",)})
    r = decide(e_data_missing, BotHistory.empty(), cfg, now)
    assert "data_missing:1h" in r.recommendation.reasons

    e_hmm_missing = BotEvaluation(**{**base, "hmm_artifact_missing": True})
    r = decide(e_hmm_missing, BotHistory.empty(), cfg, now)
    assert "hmm_artifact_missing" in r.recommendation.reasons

    e_transient = BotEvaluation(**{**base, "transient_fetch_error": True})
    r = decide(e_transient, BotHistory.empty(), cfg, now)
    assert "transient_fetch_error" in r.recommendation.reasons

    e_micro = BotEvaluation(**{**base, "micro_gate_pass": False})
    r = decide(e_micro, BotHistory.empty(), cfg, now)
    assert "microstructure_fail" in r.recommendation.reasons


def test_jsonl_record_top_level_keys_pinned() -> None:
    """Wire format for JSONL consumers (BI, dashboards, downstream agents)."""
    from neutralgrid.live.decision.loader import LiveBotSpec
    from neutralgrid.live.decision.recommender import Recommendation
    from neutralgrid.live.decision.renderer import ScanResult, _to_jsonl_record

    spec = LiveBotSpec(
        symbol="BTCUSDT",
        strategy_id="x",
        deploy_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        num_grids=50,
        leverage=5,
        capital_usdt=200.0,
        candidate_id=None,
    )
    eval = BotEvaluation(
        symbol="BTCUSDT",
        evaluated_at_utc=datetime(2026, 5, 7, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
    )
    rec = Recommendation(
        verdict=Verdict.CONTINUE,
        reasons=(),
        suggested_grid_lower=None,
        suggested_grid_upper=None,
        consecutive_count=1,
        escalated=False,
    )
    result = ScanResult(spec=spec, evaluation=eval, recommendation=rec, should_emit=True)
    record = _to_jsonl_record(result, datetime(2026, 5, 7, tzinfo=timezone.utc))

    expected_top_keys = {
        "ts",
        "contract_version",
        "symbol",
        "strategy_id",
        "candidate_id",
        "verdict",
        "reasons",
        "should_emit",
        "consecutive_count",
        "escalated",
        "suggested_grid_lower",
        "suggested_grid_upper",
        "evaluation",
    }
    assert expected_top_keys.issubset(set(record.keys())), (
        f"JSONL top-level keys changed. Missing: {expected_top_keys - set(record.keys())}"
    )
    assert record["contract_version"] == DECISION_CONTRACT_VERSION
