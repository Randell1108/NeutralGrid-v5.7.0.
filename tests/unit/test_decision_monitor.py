"""Tests for live.decision.monitor (Phase B) — no live network."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from neutralgrid.live.decision.loader import LiveBotSpec
from neutralgrid.data.diff_depth import LocalOrderBook
from neutralgrid.live.decision.l2_risk import (
    L2IntervalAccumulator,
    L2StreamRef,
    build_l2_risk_record,
)
from neutralgrid.live.decision.monitor import (
    MonitorContext,
    _build_meta_feature_dict,
    _compute_ev_score,
    _extract_funding_rate,
    _extract_open_interest,
    _geometric_grid_features,
    evaluate_bot,
)
from neutralgrid.live.decision.private_events import (
    PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
    PRIVATE_EVENT_SCHEMA_VERSION,
    PrivateEventStreamRef,
)


_T0 = datetime(2026, 5, 6, 14, 30, 0, tzinfo=timezone.utc)


def _spec(symbol: str = "BTCUSDT", lower: float = 60_000.0, upper: float = 70_000.0) -> LiveBotSpec:
    return LiveBotSpec(
        symbol=symbol,
        strategy_id="test-1",
        deploy_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        grid_lower=lower,
        grid_upper=upper,
        num_grids=50,
        leverage=5,
        capital_usdt=200.0,
        candidate_id=None,
    )


def _empty_context() -> MonitorContext:
    """All artifacts unavailable — exercises the offline / fail-soft paths."""
    return MonitorContext(
        hmm=None,
        meta_labeler=None,
        utility_config=None,
        hmm_unavailable=True,
        meta_unavailable=True,
        utility_unavailable=True,
    )


class _RecordingHMM:
    def __init__(self) -> None:
        self.last_close_seen: float | None = None

    def predict(self, *, df: Any) -> SimpleNamespace:
        self.last_close_seen = float(df["close"].iloc[-1])
        return SimpleNamespace(
            range_prob=0.75,
            trend_prob=0.15,
            persistence_prob=0.65,
        )


def _kline_row(ts_ms: int, close: float) -> list[Any]:
    """Minimal kline row in Binance's 12-field format."""
    return [
        ts_ms,
        str(close),  # open
        str(close + 100),  # high
        str(close - 100),  # low
        str(close),  # close
        "10.0",  # volume
        ts_ms + 60_000,  # close_time
        "650000.0",  # quote_volume
        100,  # num_trades
        "5.0",  # taker_buy_base_volume
        "325000.0",  # taker_buy_quote_volume
        "0",  # ignore
    ]


def _canned_market_data(*, symbol: str = "BTCUSDT", price: float = 65_000.0) -> dict[str, Any]:
    base_ms = int(_T0.timestamp() * 1000)
    klines_1h = [_kline_row(base_ms - i * 3_600_000, price) for i in range(120, 0, -1)]
    klines_15m = [_kline_row(base_ms - i * 900_000, price) for i in range(120, 0, -1)]
    klines_5m = [_kline_row(base_ms - i * 300_000, price) for i in range(120, 0, -1)]
    klines_1m = [_kline_row(base_ms - i * 60_000, price) for i in range(120, 0, -1)]
    return {
        "symbol": symbol,
        "klines": {
            "1h": klines_1h,
            "15m": klines_15m,
            "5m": klines_5m,
            "1m": klines_1m,
        },
        "ticker": {"lastPrice": str(price)},
        "funding_rate": 0.0001,
        "funding_info": {},
        "open_interest": 50_000_000.0,
        "premium_index": 0.0,
        "order_book": {"bids": [[price - 5, 1.0]], "asks": [[price + 5, 1.0]]},
        "long_short_ratio": {},
        "taker_volume": {},
        "open_interest_hist": [],
    }


# ── Successful path ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_successful_fetch_emits_price_vs_grid_with_no_models() -> None:
    spec = _spec(lower=60_000, upper=70_000)
    ctx = _empty_context()
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data(price=65_000))

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.symbol == "BTCUSDT"
    assert result.price == pytest.approx(65_000.0)
    # 65_000 is exactly the midpoint of [60k, 70k] → 50%
    assert result.pct_inside_grid == pytest.approx(50.0)
    assert result.dist_to_lower_pct == pytest.approx(50.0)
    assert result.dist_to_upper_pct == pytest.approx(50.0)
    # No artifacts → these stay None / flagged
    assert result.range_prob is None
    assert result.trend_prob is None
    assert result.meta_proba is None
    assert result.utility_score is None
    assert result.hmm_artifact_missing is True
    assert "meta_overlay_inactive" in result.diagnostics
    assert "utility_calibrator_unavailable" in result.diagnostics
    # Operational flags clean
    assert result.symbol_unavailable is False
    assert result.transient_fetch_error is False


@pytest.mark.asyncio
async def test_evaluate_bot_attaches_fresh_sequence_linked_l2_evidence(
    tmp_path: Path,
) -> None:
    stream_path = tmp_path / "l2_risk_snapshots.jsonl"
    record = build_l2_risk_record(
        book=LocalOrderBook(
            bids={Decimal("64999"): Decimal("1")},
            asks={Decimal("65001"): Decimal("1")},
        ),
        accumulator=L2IntervalAccumulator(),
        symbol="BTCUSDT",
        run_id="run-1",
        connection_id="connection",
        segment_id="segment-1",
        captured_at_utc=_T0.isoformat(),
        wire_sequence=1,
        final_update_id=1,
        top_n=10,
    )
    stream_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    spec = replace(
        _spec(),
        l2_stream=L2StreamRef(
            feature_path=stream_path,
            symbol="BTCUSDT",
            run_id="run-1",
        ),
    )
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(
        return_value=_canned_market_data(price=65_000)
    )

    result = await evaluate_bot(
        spec, context=_empty_context(), client=client_mock, now=_T0
    )

    assert result.l2_risk is not None
    assert result.execution_risk is not None
    assert result.l2_risk.run_id == "run-1"
    assert result.execution_risk.l2_run_id == result.l2_risk.run_id
    assert result.execution_risk.l2_segment_id == result.l2_risk.segment_id
    assert result.execution_risk.l2_snapshot_count == result.l2_risk.snapshot_count
    assert result.l2_risk.spread_bps == pytest.approx(0.3076923077)
    assert not any(
        code.startswith("l2_stream_unavailable:") for code in result.diagnostics
    )


@pytest.mark.asyncio
async def test_evaluate_bot_attaches_private_events_without_changing_models(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "private_events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_EVENT_SCHEMA_VERSION,
                "event_type": "trade_fill",
                "symbol": "BTCUSDT",
                "strategy_id": "test-1",
                "run_id": "private-run",
                "event_time_utc": _T0.isoformat(),
                "trade_id": "1",
                "order_id": "10",
                "side": "SELL",
                "price": 65000.0,
                "qty": 0.01,
                "maker": True,
                "commission_usdt": -0.01,
                "realized_pnl_usdt": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "private_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": PRIVATE_EVENT_MANIFEST_SCHEMA_VERSION,
                "run_id": "private-run",
                "symbol": "BTCUSDT",
                "strategy_id": "test-1",
                "status": "running",
                "updated_at_utc": _T0.isoformat(),
                "capture_mode": "binance_user_data_stream",
                "event_completeness": "event_complete",
                "source_scopes": ["trades"],
                "total_records": 1,
                "duplicate_records_dropped": 0,
                "rejected_records": 0,
            }
        ),
        encoding="utf-8",
    )
    spec = replace(
        _spec(),
        private_event_stream=PrivateEventStreamRef(
            event_path=event_path,
            manifest_path=manifest_path,
            symbol="BTCUSDT",
            strategy_id="test-1",
            run_id="private-run",
        ),
    )
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(
        return_value=_canned_market_data(price=65_000)
    )

    result = await evaluate_bot(
        spec,
        context=_empty_context(),
        client=client_mock,
        now=_T0,
    )

    assert result.private_event_evidence is not None
    assert result.private_event_evidence.trade_fill_count == 1
    assert result.private_event_evidence.strategy_id == "test-1"
    assert result.range_prob is None


@pytest.mark.asyncio
async def test_price_outside_grid_pct_inside_negative_or_above_100() -> None:
    spec = _spec(lower=60_000, upper=70_000)
    ctx = _empty_context()
    # Price below grid → pct_inside_grid < 0
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(
        return_value=_canned_market_data(price=58_000.0)
    )
    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)
    assert result.price == pytest.approx(58_000.0)
    assert result.pct_inside_grid is not None and result.pct_inside_grid < 0
    # Distance fields are None when price is outside the grid (negative
    # distance is meaningless)
    assert result.dist_to_lower_pct is None
    assert result.dist_to_upper_pct is None


# ── Failure paths ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_4xx_response_yields_symbol_unavailable() -> None:
    spec = _spec(symbol="NOTREAL")
    ctx = _empty_context()
    client_mock = AsyncMock()
    request = httpx.Request("GET", "https://api.binance.test/fapi/v1/exchangeInfo")
    response = httpx.Response(status_code=400, request=request)
    client_mock.get_all_market_data = AsyncMock(
        side_effect=httpx.HTTPStatusError("400 Bad Request", request=request, response=response)
    )

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.symbol_unavailable is True
    assert result.transient_fetch_error is False
    assert any(d.startswith("binance_4xx:") for d in result.diagnostics)


@pytest.mark.asyncio
async def test_5xx_response_yields_transient_fetch_error() -> None:
    spec = _spec()
    ctx = _empty_context()
    client_mock = AsyncMock()
    request = httpx.Request("GET", "https://api.binance.test/fapi/v1/exchangeInfo")
    response = httpx.Response(status_code=503, request=request)
    client_mock.get_all_market_data = AsyncMock(
        side_effect=httpx.HTTPStatusError("503 Service Unavailable", request=request, response=response)
    )

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.transient_fetch_error is True
    assert result.symbol_unavailable is False


@pytest.mark.asyncio
async def test_network_error_yields_transient_fetch_error() -> None:
    spec = _spec()
    ctx = _empty_context()
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(side_effect=httpx.ConnectError("DNS lookup failed"))

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.transient_fetch_error is True
    assert result.symbol_unavailable is False
    # Diagnostic preserves the underlying error class for forensics
    assert any("ConnectError" in d for d in result.diagnostics)


@pytest.mark.asyncio
async def test_empty_klines_yields_data_missing_diagnostic() -> None:
    """Fail-closed: empty 1h klines must surface as data_missing, not silent CONTINUE."""
    spec = _spec()
    ctx = _empty_context()
    market = _canned_market_data()
    market["klines"]["1h"] = []
    market["klines"]["15m"] = []
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=market)

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert "data_missing:1h" in result.diagnostics
    assert "data_missing:15m" in result.diagnostics
    # Even with 1m klines, range_prob can't be computed without 15m HMM input
    assert result.range_prob is None


@pytest.mark.asyncio
async def test_falls_back_to_kline_close_when_ticker_missing() -> None:
    spec = _spec()
    ctx = _empty_context()
    market = _canned_market_data(price=65_500.0)
    market.pop("ticker", None)
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=market)

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.price == pytest.approx(65_500.0)


# ── HMM / utility diagnostics ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hmm_unavailable_propagates_artifact_missing_flag() -> None:
    spec = _spec()
    ctx = MonitorContext(
        hmm=None,
        meta_labeler=None,
        utility_config=None,
        hmm_unavailable=True,
        meta_unavailable=False,
        utility_unavailable=False,
    )
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.hmm_artifact_missing is True
    assert result.range_prob is None


@pytest.mark.asyncio
async def test_hmm_uses_15m_klines_not_1h_klines() -> None:
    spec = _spec()
    hmm = _RecordingHMM()
    ctx = MonitorContext(
        hmm=hmm,
        meta_labeler=None,
        utility_config=None,
        hmm_unavailable=False,
        meta_unavailable=True,
        utility_unavailable=True,
    )
    market = _canned_market_data(price=65_000.0)
    base_ms = int(_T0.timestamp() * 1000)
    market["klines"]["1h"] = [
        _kline_row(base_ms - i * 3_600_000, 61_000.0) for i in range(120, 0, -1)
    ]
    market["klines"]["15m"] = [
        _kline_row(base_ms - i * 900_000, 66_000.0) for i in range(120, 0, -1)
    ]
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=market)

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert hmm.last_close_seen == pytest.approx(66_000.0)
    assert result.range_prob == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_utility_unavailable_emits_diagnostic_not_silent_default() -> None:
    """Per safety-invariants.md: missing utility calibrator → diagnostic, never v0 default."""
    spec = _spec()
    ctx = MonitorContext(
        hmm=None,
        meta_labeler=None,
        utility_config=None,
        hmm_unavailable=False,
        meta_unavailable=False,
        utility_unavailable=True,
    )
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(spec, context=ctx, client=client_mock, now=_T0)

    assert result.utility_score is None
    assert "utility_calibrator_unavailable" in result.diagnostics


# ── Meta-labeler overlay feature-guard (D1: fail-closed) ─────────────────────


class _FakeMeta:
    """Minimal meta-labeler stub mirroring the loaded MetaLabeler surface.

    Exposes ``get_missing_feature_names`` but deliberately NOT a
    ``get_feature_names`` accessor (the real artifact has none). An empty probe
    dict reports the full required set; a populated dict reports ``missing`` of
    them absent.
    """

    REQUIRED = 20

    def __init__(self, missing: int) -> None:
        self._missing = missing
        self.predict_calls = 0

    def get_missing_feature_names(self, features: dict[str, Any]) -> list[str]:
        if not features:
            return [f"f{i}" for i in range(self.REQUIRED)]
        return [f"f{i}" for i in range(self._missing)]

    def predict_proba(self, features: dict[str, Any]) -> float:
        self.predict_calls += 1
        return 0.83


class _FakeMetaNoIntrospection:
    """Old-style stub exposing no feature accessor → must fail CLOSED."""

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict_proba(self, features: dict[str, Any]) -> float:
        self.predict_calls += 1
        return 0.83


def _meta_context(meta: Any, *, promoted: bool = False) -> MonitorContext:
    return MonitorContext(
        hmm=None,
        meta_labeler=meta,
        utility_config=None,
        hmm_unavailable=True,
        meta_unavailable=False,
        utility_unavailable=True,
        meta_promoted=promoted,
        active_hmm_version="rolling_180d_test",
        meta_feature_profile="fastwin_test",
    )


@pytest.mark.asyncio
async def test_meta_overlay_inactive_when_majority_features_missing() -> None:
    """>50% required features missing → overlay fail-closed, predict NOT called.

    Regression for the fail-OPEN guard: the monitor used to call a non-existent
    get_feature_names(), raise AttributeError, and fall through to predict_proba
    on silently-imputed features. The guard must now skip the overlay instead.
    """
    meta = _FakeMeta(missing=15)  # 15/20 = 0.75 > 0.5
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(_spec(), context=_meta_context(meta), client=client_mock, now=_T0)

    assert result.meta_proba is None
    assert "meta_overlay_inactive" in result.diagnostics
    assert meta.predict_calls == 0


@pytest.mark.asyncio
async def test_meta_overlay_active_with_minority_features_missing() -> None:
    """<50% required features missing → overlay predicts and flags imputation."""
    meta = _FakeMeta(missing=5)  # 5/20 = 0.25 < 0.5
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(_spec(), context=_meta_context(meta), client=client_mock, now=_T0)

    assert result.meta_proba == pytest.approx(0.83)
    assert "meta_imputed:5" in result.diagnostics
    assert meta.predict_calls == 1


@pytest.mark.asyncio
async def test_meta_overlay_fails_closed_without_introspection_method() -> None:
    """A meta-labeler lacking get_missing_feature_names must fail CLOSED, not impute."""
    meta = _FakeMetaNoIntrospection()
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(_spec(), context=_meta_context(meta), client=client_mock, now=_T0)

    assert result.meta_proba is None
    assert "meta_overlay_inactive" in result.diagnostics
    assert meta.predict_calls == 0


# ── Meta-labeler authority + fidelity (D3) ───────────────────────────────────


@pytest.mark.asyncio
async def test_meta_authoritative_when_promoted_and_full_fidelity() -> None:
    """0 missing features + promoted model → authoritative, full-fidelity."""
    meta = _FakeMeta(missing=0)
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(
        _spec(), context=_meta_context(meta, promoted=True), client=client_mock, now=_T0
    )

    assert result.meta_proba == pytest.approx(0.83)
    assert result.meta_full_fidelity is True
    assert result.meta_authoritative is True
    assert result.meta_missing_features == ()
    assert "meta_imputed:0" not in result.diagnostics
    # D7 audit evidence propagated from the context.
    assert result.meta_source == "live_overlay"
    assert result.active_hmm_version == "rolling_180d_test"
    assert result.linked_hmm_version == "rolling_180d_test"
    assert result.meta_feature_profile == "fastwin_test"


@pytest.mark.asyncio
async def test_meta_not_authoritative_when_unpromoted() -> None:
    """Full fidelity but NOT promoted → fail-closed: never authoritative."""
    meta = _FakeMeta(missing=0)
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(
        _spec(), context=_meta_context(meta, promoted=False), client=client_mock, now=_T0
    )

    assert result.meta_full_fidelity is True
    assert result.meta_authoritative is False


@pytest.mark.asyncio
async def test_meta_not_authoritative_when_imputed() -> None:
    """Promoted but features imputed (<50% missing) → not full-fidelity, not authoritative."""
    meta = _FakeMeta(missing=5)
    client_mock = AsyncMock()
    client_mock.get_all_market_data = AsyncMock(return_value=_canned_market_data())

    result = await evaluate_bot(
        _spec(), context=_meta_context(meta, promoted=True), client=client_mock, now=_T0
    )

    assert result.meta_proba == pytest.approx(0.83)  # diagnostic value still produced
    assert result.meta_full_fidelity is False
    assert result.meta_authoritative is False
    assert len(result.meta_missing_features) == 5


# ── Meta feature builders (D2) ───────────────────────────────────────────────


def test_geometric_grid_features_are_percent_and_not_arithmetic() -> None:
    """grid_spacing_pct / profit_per_grid_pct use the geometric formula, not the
    old arithmetic approximation."""
    from neutralgrid.grid.formulas import (
        GEOMETRIC,
        grid_spacing_pct as _gsp,
        profit_per_grid_pct as _ppg,
    )
    from neutralgrid.core.config import get_config

    spec = _spec(lower=60_000.0, upper=70_000.0)
    spacing, profit = _geometric_grid_features(spec)

    assert spacing is not None and profit is not None
    # Matches the canonical single-source geometric formula.
    expected_spacing = _gsp(spec.grid_lower, spec.grid_upper, spec.num_grids, GEOMETRIC)
    cfg_grid = get_config().grid
    close_mode = str(getattr(cfg_grid, "close_fee_mode", "maker")).lower()
    close_rate = cfg_grid.maker_fee if close_mode == "maker" else cfg_grid.taker_fee
    c = max(0.0, (cfg_grid.maker_fee + close_rate) / 2.0)
    expected_profit = _ppg(spec.grid_lower, spec.grid_upper, spec.num_grids, GEOMETRIC, c)
    assert spacing == pytest.approx(expected_spacing)
    assert profit == pytest.approx(expected_profit)

    # Differs from the previous arithmetic approximation (proves the skew fix).
    midpoint = (spec.grid_lower + spec.grid_upper) / 2.0
    arithmetic = (spec.grid_upper - spec.grid_lower) / spec.num_grids / midpoint * 100.0
    assert spacing != pytest.approx(arithmetic)


def test_geometric_grid_features_degenerate_returns_none() -> None:
    """num_grids < 2 is degenerate for the geometric formula → genuinely missing."""
    spec = _spec()
    spec_one = LiveBotSpec(
        symbol=spec.symbol,
        strategy_id=spec.strategy_id,
        deploy_ts=spec.deploy_ts,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        num_grids=1,
        leverage=spec.leverage,
        capital_usdt=spec.capital_usdt,
        candidate_id=spec.candidate_id,
    )
    assert _geometric_grid_features(spec_one) == (None, None)


def test_compute_ev_score_uses_rank_score_and_fails_closed() -> None:
    """ev_score == round(rank_score, 4); None when any required input absent."""

    class _FakeRanker:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def compute_score(self, **kw: Any) -> SimpleNamespace:
            self.calls.append(kw)
            return SimpleNamespace(rank_score=1.234567)

    ranker = _FakeRanker()
    ev = _compute_ev_score(
        spec=_spec(),
        ranker=ranker,
        survival_prob=0.8,
        profit_per_grid_pct=0.5,
        trend_prob=0.2,
        range_size_pct=3.0,
        funding_rate=0.0001,
    )
    assert ev == pytest.approx(1.2346)
    assert ranker.calls[0]["profit_per_grid_pct"] == 0.5
    assert ranker.calls[0]["leverage"] == int(_spec().leverage)

    # Fail-closed: a missing required input yields None (no fabrication).
    assert (
        _compute_ev_score(
            spec=_spec(), ranker=ranker, survival_prob=None,
            profit_per_grid_pct=0.5, trend_prob=0.2, range_size_pct=3.0, funding_rate=None,
        )
        is None
    )
    assert (
        _compute_ev_score(
            spec=_spec(), ranker=None, survival_prob=0.8,
            profit_per_grid_pct=0.5, trend_prob=0.2, range_size_pct=3.0, funding_rate=None,
        )
        is None
    )


def test_build_meta_feature_dict_uses_geometric_and_excludes_labels() -> None:
    """Computed features land in the dict; grid_spacing_pct is the geometric
    value passed in; no label column is ever included (leakage invariant)."""
    sym_feats = SimpleNamespace(as_dict=lambda: {"adx_1h": 20.0, "range_size_pct": 3.0})
    out = _build_meta_feature_dict(
        sym_feats=sym_feats,
        spec=_spec(),
        range_prob=0.7,
        trend_prob=0.2,
        last_price=65_000.0,
        hurst_exponent=0.45,
        ou_halflife=24.0,
        micro_round_trip_cost_pct=0.12,
        grid_spacing_pct=0.5,
        profit_per_grid_pct=0.4,
        ev_score=1.1,
    )
    assert out["hurst_exponent"] == 0.45
    assert out["ou_halflife"] == 24.0
    assert out["micro_round_trip_cost_pct"] == 0.12
    assert out["grid_spacing_pct"] == 0.5  # geometric value passed in, not arithmetic
    assert out["profit_per_grid_pct"] == 0.4
    assert out["ev_score"] == 1.1
    assert out["num_grids"] == float(_spec().num_grids)
    for label in ("hlabel", "hlabel_meta", "y", "net_pnl_pct", "pnl_pct"):
        assert label not in out


def test_live_feature_dict_full_fidelity_no_train_serve_gap_err048() -> None:
    """ERR-048: when live data is sufficient the monitor's feature builder produces
    EVERY meta-labeler feature via the same canonical code paths as training
    (option C), so there is NO train/serve gap / silent imputation. With all 13
    SymbolFeatures.as_dict() technicals + the 7 caller-computed features supplied,
    MetaLabeler.get_missing_feature_names must report zero missing."""
    from neutralgrid.models.meta_labeler import MetaLabeler

    sym_feats = SimpleNamespace(as_dict=lambda: {
        "adx_1h": 22.0, "adx_15m": 20.0, "adx_5m": 24.0, "atr_pct_15m": 0.01,
        "rsi_15m": 50.0, "ema_slope_1h": 0.0, "ema_crosses_5m": 1.0,
        "vwap_crosses_5m": 1.0, "range_size_pct": 5.0, "bb_width": 0.03,
        "quote_volume_24h": 5.0e7, "open_interest": 5.0e7, "funding_rate": 0.0001,
    })
    out = _build_meta_feature_dict(
        sym_feats=sym_feats, spec=_spec(), range_prob=0.7, trend_prob=0.2,
        last_price=65_000.0, hurst_exponent=0.45, ou_halflife=24.0,
        micro_round_trip_cost_pct=0.12, grid_spacing_pct=0.5,
        profit_per_grid_pct=0.4, ev_score=1.1,
    )
    # Fresh MetaLabeler falls back to config.features (the active FASTWIN 20-set).
    missing = MetaLabeler().get_missing_feature_names(out)
    assert missing == [], f"unexpected live train/serve gap: {missing}"


def test_extract_funding_rate_returns_scalar_not_list() -> None:
    """Regression: get_all_market_data returns funding_rate as a raw history LIST;
    the feature must be the scalar lastFundingRate (a list/dict would break the
    meta-labeler probe with an ambiguous-truth-value error)."""
    # premium_index is the canonical source (mirrors scan.py).
    market = {
        "premium_index": {"lastFundingRate": "0.0001"},
        "funding_rate": [{"fundingRate": "0.0009"}, {"fundingRate": "0.0007"}],
    }
    assert _extract_funding_rate(market) == pytest.approx(0.0001)
    # Fallback to the latest history entry when premium_index is absent.
    assert _extract_funding_rate(
        {"funding_rate": [{"fundingRate": "0.0009"}, {"fundingRate": "0.0007"}]}
    ) == pytest.approx(0.0007)
    # Already-scalar and missing cases.
    assert _extract_funding_rate({"funding_rate": 0.0005}) == pytest.approx(0.0005)
    assert _extract_funding_rate({}) is None


def test_extract_open_interest_returns_scalar_not_dict() -> None:
    """Regression: get_all_market_data returns open_interest as a raw DICT; the
    feature must be the scalar float(openInterest)."""
    assert _extract_open_interest(
        {"open_interest": {"symbol": "ENAUSDT", "openInterest": "490365122"}}
    ) == pytest.approx(490365122.0)
    assert _extract_open_interest({"open_interest": 123.0}) == pytest.approx(123.0)
    assert _extract_open_interest({}) is None
