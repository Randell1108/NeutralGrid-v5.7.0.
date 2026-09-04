"""Phase D tests: deploy-time deltas, --config-file overrides, microstructure gate.

Monitor-level integration tests with the real microstructure estimator and a
synthetic linker cache. No live network."""
from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from neutralgrid.live.decision.loader import LiveBotSpec
from neutralgrid.live.decision.monitor import MonitorContext, evaluate_bot
from neutralgrid.live.decision.recommender import RecommenderConfig
from neutralgrid.live.decision.renderer import (
    JsonlWriter,
    ScanResult,
    _to_jsonl_record,
)
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    Recommendation,
    Verdict,
)


_T0 = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)


def _spec(symbol: str = "BTCUSDT", **overrides: Any) -> LiveBotSpec:
    base: dict[str, Any] = dict(
        symbol=symbol,
        strategy_id="strat-1",
        deploy_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        num_grids=50,
        leverage=5,
        capital_usdt=200.0,
        candidate_id=None,
    )
    base.update(overrides)
    return LiveBotSpec(**base)


def _kline_row(ts_ms: int, close: float) -> list[Any]:
    return [
        ts_ms, str(close), str(close + 100), str(close - 100), str(close),
        "10.0", ts_ms + 60_000, "650000.0", 100, "5.0", "325000.0", "0",
    ]


def _canned_market_data(price: float = 65_000.0) -> dict[str, Any]:
    base_ms = int(_T0.timestamp() * 1000)
    klines = {
        tf: [_kline_row(base_ms - i * 60_000, price) for i in range(120, 0, -1)]
        for tf in ("1h", "15m", "5m", "1m")
    }
    return {
        "symbol": "BTCUSDT",
        "klines": klines,
        "ticker": {"lastPrice": str(price)},
        "funding_rate": 0.0001,
        "funding_info": {},
        "open_interest": 50_000_000.0,
        "premium_index": 0.0,
        # Tight 0.005% spread; deep book → liquidity passes
        "order_book": {
            "bids": [[price - 1, 100.0], [price - 2, 100.0], [price - 3, 100.0]],
            "asks": [[price + 1, 100.0], [price + 2, 100.0], [price + 3, 100.0]],
        },
        "long_short_ratio": {},
        "taker_volume": {},
        "open_interest_hist": [],
    }


def _bare_context(
    *,
    micro_estimator: Any = None,
    micro_gate: Any = None,
    linker_by_strategy: dict[str, dict[str, Any]] | None = None,
    linker_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> MonitorContext:
    return MonitorContext(
        hmm=None,
        meta_labeler=None,
        utility_config=None,
        hmm_unavailable=True,
        meta_unavailable=True,
        utility_unavailable=True,
        micro_estimator=micro_estimator,
        micro_gate=micro_gate,
        linker_by_strategy=linker_by_strategy or {},
        linker_by_candidate=linker_by_candidate or {},
    )


# -- --config-file -----------------------------------------------------------


def test_recommender_config_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "thresholds.yaml"
    p.write_text(
        textwrap.dedent(
            """
            min_range_prob: 0.50
            adjust_escalate_after: 5
            continue_heartbeat_min: 90
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = RecommenderConfig.from_file(p)
    assert cfg.min_range_prob == 0.50
    assert cfg.adjust_escalate_after == 5
    assert cfg.continue_heartbeat_min == 90
    # Unspecified fields keep defaults
    assert cfg.end_cooldown_min == 30


def test_recommender_config_loads_meta_tilt_enablement_fields(tmp_path: Path) -> None:
    p = tmp_path / "meta_tilt_thresholds.yaml"
    p.write_text(
        textwrap.dedent(
            """
            meta_tilt_enabled: true
            meta_tilt_low_threshold: 0.45
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = RecommenderConfig.from_file(p)
    assert cfg.meta_tilt_enabled is True
    assert cfg.meta_tilt_low_threshold == 0.45


def test_recommender_config_from_json(tmp_path: Path) -> None:
    p = tmp_path / "thresholds.json"
    p.write_text(json.dumps({"min_range_prob": 0.55, "boundary_proximity_pct": 5.0}),
                 encoding="utf-8")
    cfg = RecommenderConfig.from_file(p)
    assert cfg.min_range_prob == 0.55
    assert cfg.boundary_proximity_pct == 5.0


def test_recommender_config_unknown_keys_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "thresholds.yaml"
    p.write_text(
        textwrap.dedent(
            """
            min_range_prob: 0.50
            unknown_field: 99
            """
        ).strip(),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        cfg = RecommenderConfig.from_file(p)
    assert cfg.min_range_prob == 0.50
    assert any("unknown_field" in r.message for r in caplog.records)


def test_recommender_config_rejects_bad_extension(tmp_path: Path) -> None:
    p = tmp_path / "thresholds.txt"
    p.write_text("min_range_prob: 0.5", encoding="utf-8")
    with pytest.raises(ValueError, match="must be .yaml/.yml/.json"):
        RecommenderConfig.from_file(p)


def test_recommender_config_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "thresholds.yaml"
    p.write_text("- 1\n- 2", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        RecommenderConfig.from_file(p)


def test_recommender_config_empty_file_yields_defaults(tmp_path: Path) -> None:
    p = tmp_path / "thresholds.yaml"
    p.write_text("", encoding="utf-8")
    cfg = RecommenderConfig.from_file(p)
    assert cfg == RecommenderConfig()


# -- Deploy snapshot --------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_snapshot_populated_when_strategy_id_resolves() -> None:
    linker_row = {
        "strategy_id": "strat-1",
        "candidate_id": "cand-abc",
        "deploy_time_utc": "2026-05-01T14:00:00Z",
        "meta_prob": 0.65,
        "score": 1.20,
        "ev_score": 0.85,
        "ev_24h": 12.5,
        "grid_lower": 60_000.0,
        "grid_upper": 70_000.0,
        "num_grids": 50,
        "leverage": 5,
    }
    ctx = _bare_context(linker_by_strategy={"strat-1": linker_row})
    spec = _spec(strategy_id="strat-1", candidate_id=None)

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.deploy_snapshot is not None
    assert result.deploy_snapshot["strategy_id"] == "strat-1"
    assert result.deploy_snapshot["meta_prob"] == 0.65
    assert result.deploy_snapshot["ev_score"] == 0.85
    assert "candidate_link_missing" not in result.diagnostics


@pytest.mark.asyncio
async def test_candidate_id_takes_precedence_over_strategy_id() -> None:
    cand_row = {"strategy_id": "ignored", "candidate_id": "cand-A", "meta_prob": 0.50}
    strat_row = {"strategy_id": "strat-X", "candidate_id": "other", "meta_prob": 0.10}
    ctx = _bare_context(
        linker_by_candidate={"cand-A": cand_row},
        linker_by_strategy={"strat-X": strat_row},
    )
    spec = _spec(strategy_id="strat-X", candidate_id="cand-A")

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.deploy_snapshot is not None
    assert result.deploy_snapshot["meta_prob"] == 0.50  # candidate-row wins


@pytest.mark.asyncio
async def test_candidate_link_missing_diagnostic_when_lookup_fails() -> None:
    """When the YAML provides an identifier but the linker has no row → diagnostic, no fail."""
    ctx = _bare_context(linker_by_strategy={})
    spec = _spec(strategy_id="strat-not-in-linker")

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.deploy_snapshot is None
    assert "candidate_link_missing" in result.diagnostics


@pytest.mark.asyncio
async def test_no_identifier_means_no_diagnostic() -> None:
    """If the YAML omits both strategy_id and candidate_id, we don't ask the linker."""
    spec = _spec(strategy_id=None, candidate_id=None)
    ctx = _bare_context()

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.deploy_snapshot is None
    assert "candidate_link_missing" not in result.diagnostics


@pytest.mark.asyncio
async def test_delta_meta_prob_only_when_both_sides_present() -> None:
    # current_meta_proba is None (no meta-labeler) → no delta
    linker_row = {"strategy_id": "strat-1", "candidate_id": "", "meta_prob": 0.65}
    ctx = _bare_context(linker_by_strategy={"strat-1": linker_row})
    spec = _spec(strategy_id="strat-1", candidate_id=None)

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.deploy_snapshot is not None
    assert "delta_meta_prob" not in result.deploy_snapshot


@pytest.mark.asyncio
async def test_linker_cache_uses_latest_row_when_strategy_id_repeats() -> None:
    """Append-only chronological log: re-deploys with the same strategy_id must
    attribute deltas to the LATEST row, not the earliest (BLOCKER from Phase E
    data-curator review).
    """
    earlier = {"strategy_id": "strat-1", "candidate_id": "", "meta_prob": 0.10,
               "deploy_time_utc": "2026-04-01T10:00:00Z", "score": 0.5}
    later = {"strategy_id": "strat-1", "candidate_id": "", "meta_prob": 0.65,
             "deploy_time_utc": "2026-05-05T10:00:00Z", "score": 1.2}

    # Manually replicate MonitorContext.create()'s last-match-wins indexing
    linker_by_strategy: dict[str, dict[str, Any]] = {}
    for row in (earlier, later):
        linker_by_strategy[row["strategy_id"]] = row  # type: ignore[index]

    ctx = _bare_context(linker_by_strategy=linker_by_strategy)
    spec = _spec(strategy_id="strat-1", candidate_id=None)

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.deploy_snapshot is not None
    # The LATEST deploy's meta_prob (0.65) must win, not the earlier 0.10
    assert result.deploy_snapshot["meta_prob"] == 0.65
    assert result.deploy_snapshot["score"] == 1.2


def test_deploy_snapshot_skips_nan_values() -> None:
    """Pandas-loaded CSV rows often have NaN floats; those must not leak into the snapshot."""
    from neutralgrid.live.decision.monitor import _build_deploy_snapshot

    spec = _spec(strategy_id="strat-1")
    nan = float("nan")
    linker_row = {"strategy_id": "strat-1", "meta_prob": nan, "score": 1.5,
                   "candidate_id": ""}
    ctx = _bare_context(linker_by_strategy={"strat-1": linker_row})

    diags: list[str] = []
    snap = _build_deploy_snapshot(
        spec=spec, context=ctx, current_meta_proba=0.7, diagnostics=diags,
    )
    assert snap is not None
    assert "meta_prob" not in snap  # NaN dropped
    assert snap["score"] == 1.5


# -- Microstructure gate ----------------------------------------------------


@pytest.mark.asyncio
async def test_microstructure_gate_runs_with_real_components() -> None:
    """End-to-end: real MicrostructureEstimator + MicrostructureHardGate against canned book."""
    from neutralgrid.validation.microstructure import MicrostructureEstimator
    from neutralgrid.validation.microstructure_hard_gate import MicrostructureHardGate

    ctx = _bare_context(
        micro_estimator=MicrostructureEstimator(),
        micro_gate=MicrostructureHardGate(),
    )
    spec = _spec(strategy_id="strat-1", candidate_id=None)

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    # Real components should produce a definite True or False (not None)
    assert result.micro_gate_pass in (True, False)
    # If it failed, micro_reasons should be populated
    if result.micro_gate_pass is False:
        assert len(result.micro_reasons) > 0


@pytest.mark.asyncio
async def test_microstructure_skipped_when_order_book_missing() -> None:
    """No order_book in market data → micro_gate_pass=None, data_missing diagnostic."""
    from neutralgrid.validation.microstructure import MicrostructureEstimator
    from neutralgrid.validation.microstructure_hard_gate import MicrostructureHardGate

    ctx = _bare_context(
        micro_estimator=MicrostructureEstimator(),
        micro_gate=MicrostructureHardGate(),
    )
    spec = _spec()
    market = _canned_market_data()
    market["order_book"] = {}  # empty / missing

    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=market)

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.micro_gate_pass is None
    assert "data_missing:order_book" in result.micro_reasons


@pytest.mark.asyncio
async def test_microstructure_skipped_when_no_estimator() -> None:
    """Context with micro_estimator=None → micro_gate_pass=None, no crash."""
    ctx = _bare_context(micro_estimator=None, micro_gate=None)
    spec = _spec()
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.micro_gate_pass is None


# -- JSONL schema -----------------------------------------------------------


def test_jsonl_includes_deploy_snapshot_and_micro_reasons(tmp_path: Path) -> None:
    spec = _spec(strategy_id="strat-1")
    eval = BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=_T0,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        deploy_snapshot={
            "candidate_id": "cand-abc",
            "meta_prob": 0.6,
            "delta_meta_prob": 0.05,
        },
        micro_gate_pass=False,
        micro_reasons=("spread_to_profit_too_high(0.5>0.3)",),
    )
    rec = Recommendation(
        verdict=Verdict.ADJUST,
        reasons=("microstructure_fail",),
        suggested_grid_lower=None,
        suggested_grid_upper=None,
        consecutive_count=1,
        escalated=False,
    )
    result = ScanResult(spec=spec, evaluation=eval, recommendation=rec, should_emit=True)

    record = _to_jsonl_record(result, _T0)
    assert record["candidate_id"] == "cand-abc"
    assert record["evaluation"]["deploy_snapshot"] == {
        "candidate_id": "cand-abc",
        "meta_prob": 0.6,
        "delta_meta_prob": 0.05,
    }
    assert record["evaluation"]["micro_reasons"] == ["spread_to_profit_too_high(0.5>0.3)"]

    # Round-trip through JsonlWriter
    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(result, now=_T0)
    writer.close()
    line = writer.path_for(_T0).read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["evaluation"]["deploy_snapshot"]["delta_meta_prob"] == 0.05


def test_jsonl_candidate_id_prefers_explicit_spec_over_deploy_snapshot() -> None:
    spec = _spec(strategy_id="strat-1", candidate_id="explicit-candidate")
    eval = BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=_T0,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        deploy_snapshot={"candidate_id": "snapshot-candidate"},
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

    record = _to_jsonl_record(result, _T0)

    assert record["candidate_id"] == "explicit-candidate"
