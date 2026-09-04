"""Tests for live.decision.renderer (Phase B) — console + JSONL."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from neutralgrid.core.constants import DECISION_CONTRACT_VERSION
from neutralgrid.live.decision.loader import (
    LiveBotSpec,
    LiveExecutionTelemetry,
    PnlTelemetry,
)
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    Recommendation,
    Verdict,
)
from neutralgrid.live.decision.renderer import (
    ConsoleRenderer,
    JsonlWriter,
    ScanResult,
)


_T0 = datetime(2026, 5, 6, 14, 30, 0, tzinfo=timezone.utc)


def _spec(
    symbol: str = "BTCUSDT",
    execution_telemetry: LiveExecutionTelemetry | None = None,
) -> LiveBotSpec:
    return LiveBotSpec(
        symbol=symbol,
        strategy_id="strat-1",
        deploy_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        num_grids=50,
        leverage=5,
        capital_usdt=200.0,
        candidate_id=None,
        execution_telemetry=execution_telemetry,
    )


def _eval(symbol: str = "BTCUSDT", price: float = 65_000.0) -> BotEvaluation:
    return BotEvaluation(
        symbol=symbol,
        evaluated_at_utc=_T0,
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        price=price,
        pct_inside_grid=50.0,
        dist_to_lower_pct=50.0,
        dist_to_upper_pct=50.0,
        range_prob=0.7,
        trend_prob=0.2,
        persistence_prob=0.55,
        meta_proba=0.6,
        utility_score=0.1,
        micro_gate_pass=True,
    )


def _result(
    *,
    verdict: Verdict = Verdict.CONTINUE,
    reasons: tuple[str, ...] = (),
    should_emit: bool = True,
    consecutive: int = 1,
    escalated: bool = False,
    symbol: str = "BTCUSDT",
) -> ScanResult:
    return ScanResult(
        spec=_spec(symbol),
        evaluation=_eval(symbol=symbol),
        recommendation=Recommendation(
            verdict=verdict,
            reasons=reasons,
            suggested_grid_lower=None,
            suggested_grid_upper=None,
            consecutive_count=consecutive,
            escalated=escalated,
        ),
        should_emit=should_emit,
    )


# ── Console renderer ─────────────────────────────────────────────────────────


def test_console_renders_header_and_aggregates() -> None:
    results = [
        _result(verdict=Verdict.CONTINUE),
        _result(verdict=Verdict.ADJUST, reasons=("range_prob_borderline:0.40",)),
        _result(verdict=Verdict.END, reasons=("symbol_unavailable",), should_emit=True),
    ]
    out = ConsoleRenderer().render(results, now=_T0)
    assert DECISION_CONTRACT_VERSION in out
    assert "CONTINUE=1" in out and "ADJUST=1" in out and "END=1" in out
    assert "emitted=3" in out
    assert "SYMBOL" in out and "VERDICT" in out and "REASONS" in out


def test_console_handles_empty_results() -> None:
    out = ConsoleRenderer().render([], now=_T0)
    assert "(no bots to evaluate)" in out


def test_console_marks_non_emitted_rows() -> None:
    """Suppressed rows (cool-down) should still render but without the * emit marker."""
    results = [_result(should_emit=False)]
    out = ConsoleRenderer().render(results, now=_T0)
    assert "BTCUSDT" in out
    # Body row for non-emit should not start with the * marker
    body = [line for line in out.splitlines() if "BTCUSDT" in line][0]
    assert not body.lstrip().startswith("*")


def test_console_renders_escalation_marker() -> None:
    results = [_result(verdict=Verdict.ADJUST, escalated=True, consecutive=3)]
    out = ConsoleRenderer().render(results, now=_T0)
    body = [line for line in out.splitlines() if "BTCUSDT" in line][0]
    # ESCAL column should show "Y" for escalated
    assert "Y" in body


def test_console_handles_none_fields() -> None:
    """No price, no probs → renders '--' rather than crashing."""
    spec = _spec()
    e = BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=_T0,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
    )
    rec = Recommendation(
        verdict=Verdict.ADJUST,
        reasons=("data_missing:1h",),
        suggested_grid_lower=None,
        suggested_grid_upper=None,
        consecutive_count=1,
        escalated=False,
    )
    result = ScanResult(spec=spec, evaluation=e, recommendation=rec, should_emit=True)
    out = ConsoleRenderer().render([result], now=_T0)
    assert "--" in out  # placeholder for None numeric fields


# ── JSONL writer ─────────────────────────────────────────────────────────────


def test_jsonl_appends_one_record_per_call(tmp_path: Path) -> None:
    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(_result(symbol="BTCUSDT"), now=_T0)
    writer.append(_result(symbol="ETHUSDT", verdict=Verdict.ADJUST,
                          reasons=("range_prob_borderline:0.40",)), now=_T0)
    writer.close()

    path = writer.path_for(_T0)
    assert path.is_file()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    rec_a = json.loads(lines[0])
    rec_b = json.loads(lines[1])
    assert rec_a["symbol"] == "BTCUSDT"
    assert rec_a["verdict"] == "CONTINUE"
    assert rec_a["contract_version"] == DECISION_CONTRACT_VERSION
    assert rec_b["symbol"] == "ETHUSDT"
    assert rec_b["reasons"] == ["range_prob_borderline:0.40"]


def test_jsonl_includes_full_evaluation_fields(tmp_path: Path) -> None:
    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(_result(verdict=Verdict.CONTINUE), now=_T0)
    writer.close()

    line = writer.path_for(_T0).read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    e = rec["evaluation"]
    expected_keys = {
        "evaluated_at_utc", "grid_lower", "grid_upper", "price",
        "pct_inside_grid", "dist_to_lower_pct", "dist_to_upper_pct",
        "range_prob", "trend_prob", "persistence_prob", "meta_proba",
        "utility_score", "micro_gate_pass", "symbol_unavailable",
        "transient_fetch_error", "hmm_artifact_missing", "diagnostics",
        "private_event_evidence",
    }
    assert expected_keys.issubset(set(e.keys()))


def test_jsonl_includes_execution_telemetry_when_present(tmp_path: Path) -> None:
    telemetry = LiveExecutionTelemetry(
        source="user_provided_binance_ui",
        pnl=PnlTelemetry(realized_profit_usdt=24.03, matched_profit_usdt=24.05),
    )
    spec = _spec(symbol="COSUSDT", execution_telemetry=telemetry)
    result = ScanResult(
        spec=spec,
        evaluation=_eval(symbol="COSUSDT"),
        recommendation=Recommendation(
            verdict=Verdict.CONTINUE,
            reasons=(),
            suggested_grid_lower=None,
            suggested_grid_upper=None,
            consecutive_count=1,
            escalated=False,
        ),
        should_emit=True,
    )

    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(result, now=_T0)
    writer.close()

    line = writer.path_for(_T0).read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["execution_telemetry"]["source"] == "user_provided_binance_ui"
    assert rec["execution_telemetry"]["pnl"]["realized_profit_usdt"] == 24.03
    assert rec["execution_telemetry"]["pnl"]["matched_profit_usdt"] == 24.05


def test_jsonl_rolls_over_on_utc_date_change(tmp_path: Path) -> None:
    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(_result(symbol="BTCUSDT"), now=_T0)
    next_day = _T0 + timedelta(days=1, hours=1)
    writer.append(_result(symbol="ETHUSDT"), now=next_day)
    writer.close()

    path_day1 = writer.path_for(_T0)
    path_day2 = writer.path_for(next_day)
    assert path_day1.is_file()
    assert path_day2.is_file()
    assert path_day1 != path_day2

    rec1 = json.loads(path_day1.read_text(encoding="utf-8").strip())
    rec2 = json.loads(path_day2.read_text(encoding="utf-8").strip())
    assert rec1["symbol"] == "BTCUSDT"
    assert rec2["symbol"] == "ETHUSDT"


def test_jsonl_context_manager_closes_handle(tmp_path: Path) -> None:
    with JsonlWriter(base_dir=tmp_path) as writer:
        writer.append(_result(), now=_T0)
    # After exit, the handle should be closed (idempotent close is safe)
    writer.close()  # second call must not raise


def test_jsonl_creates_logs_dir_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "deeply" / "nested" / "logs"
    assert not target.exists()
    writer = JsonlWriter(base_dir=target)
    writer.append(_result(), now=_T0)
    writer.close()
    assert target.is_dir()
    assert any(target.iterdir())


def test_jsonl_handles_none_fields_gracefully(tmp_path: Path) -> None:
    """Records with None probabilities should serialize as null, not crash."""
    spec = _spec()
    e = BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=_T0,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        diagnostics=("data_missing:1h",),
    )
    rec = Recommendation(
        verdict=Verdict.ADJUST,
        reasons=("data_missing:1h",),
        suggested_grid_lower=None,
        suggested_grid_upper=None,
        consecutive_count=1,
        escalated=False,
    )
    result = ScanResult(spec=spec, evaluation=e, recommendation=rec, should_emit=True)

    writer = JsonlWriter(base_dir=tmp_path)
    writer.append(result, now=_T0)
    writer.close()

    line = writer.path_for(_T0).read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["evaluation"]["price"] is None
    assert parsed["evaluation"]["range_prob"] is None
    assert parsed["evaluation"]["diagnostics"] == ["data_missing:1h"]
