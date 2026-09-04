"""Tests for live.decision.discord_sink.DiscordDigestSink (Phase C)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from neutralgrid.live.decision.discord_sink import (
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    DiscordDigestSink,
)
from neutralgrid.live.decision.loader import LiveBotSpec
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    Recommendation,
    Verdict,
)
from neutralgrid.live.decision.renderer import ScanResult


_T0 = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)


def _spec(symbol: str = "BTCUSDT") -> LiveBotSpec:
    return LiveBotSpec(
        symbol=symbol,
        strategy_id=f"strat-{symbol}",
        deploy_ts=datetime(2026, 5, 1, tzinfo=timezone.utc),
        grid_lower=60_000.0,
        grid_upper=70_000.0,
        num_grids=50,
        leverage=5,
        capital_usdt=200.0,
        candidate_id=None,
    )


def _result(
    *,
    verdict: Verdict = Verdict.CONTINUE,
    reasons: tuple[str, ...] = (),
    should_emit: bool = True,
    consecutive: int = 1,
    escalated: bool = False,
    suggested_lower: float | None = None,
    suggested_upper: float | None = None,
    symbol: str = "BTCUSDT",
) -> ScanResult:
    spec = _spec(symbol)
    e = BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=_T0,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        price=65_000.0,
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
    rec = Recommendation(
        verdict=verdict,
        reasons=reasons,
        suggested_grid_lower=suggested_lower,
        suggested_grid_upper=suggested_upper,
        consecutive_count=consecutive,
        escalated=escalated,
    )
    return ScanResult(spec=spec, evaluation=e, recommendation=rec, should_emit=should_emit)


def _mock_async_client() -> MagicMock:
    """A MagicMock that pretends to be an httpx.AsyncClient with a 2xx POST."""
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.status_code = 204
    client.post = AsyncMock(return_value=response)
    client.aclose = AsyncMock()
    return client


# -- Missing webhook ---------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_webhook_returns_false_without_crashing() -> None:
    sink = DiscordDigestSink(webhook_url=None)
    sent = await sink.send([_result()], now=_T0)
    assert sent is False
    # Second call must not re-warn (one-time warning only)
    sent2 = await sink.send([_result()], now=_T0)
    assert sent2 is False


@pytest.mark.asyncio
async def test_no_emit_results_skips_send() -> None:
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=_mock_async_client())
    results = [_result(should_emit=False), _result(should_emit=False)]
    sent = await sink.send(results, now=_T0)
    assert sent is False
    sink._client.post.assert_not_called()  # type: ignore[attr-defined]


# -- Successful send ---------------------------------------------------------


@pytest.mark.asyncio
async def test_send_posts_one_embed_with_aggregate_footer() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    results = [
        _result(symbol="BTCUSDT", verdict=Verdict.CONTINUE, should_emit=True),
        _result(
            symbol="ETHUSDT",
            verdict=Verdict.ADJUST,
            reasons=("range_prob_borderline:0.40",),
            should_emit=True,
        ),
        _result(symbol="SOLUSDT", verdict=Verdict.END, should_emit=False),
    ]
    sent = await sink.send(results, now=_T0)
    assert sent is True

    call_kwargs = client.post.call_args.kwargs
    assert call_kwargs["json"]["embeds"][0]["title"].startswith("Tactical Scan ")
    embed = call_kwargs["json"]["embeds"][0]
    # Only emit-true bots get fields (ETH and BTC; SOL suppressed)
    assert len(embed["fields"]) == 2
    field_names = [f["name"] for f in embed["fields"]]
    assert any("BTCUSDT" in n for n in field_names)
    assert any("ETHUSDT" in n for n in field_names)
    assert not any("SOLUSDT" in n for n in field_names)
    # Footer aggregates across the WHOLE fleet, not just emit-true
    footer = embed["footer"]["text"]
    assert "CONTINUE=1" in footer
    assert "ADJUST=1" in footer
    assert "END=1" in footer


@pytest.mark.asyncio
async def test_color_red_when_any_emit_is_end() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    await sink.send(
        [_result(verdict=Verdict.END, reasons=("symbol_unavailable",), should_emit=True)],
        now=_T0,
    )
    assert client.post.call_args.kwargs["json"]["embeds"][0]["color"] == COLOR_RED


@pytest.mark.asyncio
async def test_color_yellow_when_any_emit_is_adjust() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    await sink.send(
        [
            _result(verdict=Verdict.ADJUST, reasons=("range_prob_borderline:0.40",), should_emit=True),
            _result(verdict=Verdict.CONTINUE, should_emit=True),
        ],
        now=_T0,
    )
    assert client.post.call_args.kwargs["json"]["embeds"][0]["color"] == COLOR_YELLOW


@pytest.mark.asyncio
async def test_color_green_when_all_emit_continue() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    await sink.send([_result(verdict=Verdict.CONTINUE, should_emit=True)], now=_T0)
    assert client.post.call_args.kwargs["json"]["embeds"][0]["color"] == COLOR_GREEN


@pytest.mark.asyncio
async def test_escalated_flag_visible_in_field_name() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    await sink.send(
        [_result(verdict=Verdict.ADJUST, reasons=("range_prob_borderline:0.40",),
                  consecutive=3, escalated=True, should_emit=True)],
        now=_T0,
    )
    fields = client.post.call_args.kwargs["json"]["embeds"][0]["fields"]
    assert "ESCALATED" in fields[0]["name"]


@pytest.mark.asyncio
async def test_suggested_bounds_appear_in_field_value() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    await sink.send(
        [_result(verdict=Verdict.ADJUST, reasons=("price_near_upper:5.0pct",),
                  suggested_lower=58_000.0, suggested_upper=68_000.0, should_emit=True)],
        now=_T0,
    )
    field_value = client.post.call_args.kwargs["json"]["embeds"][0]["fields"][0]["value"]
    assert "suggested grid" in field_value
    assert "58000" in field_value and "68000" in field_value


# -- Token bucket / rate limit ----------------------------------------------


@pytest.mark.asyncio
async def test_token_bucket_spaces_consecutive_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _mock_async_client()
    sleep_calls: list[float] = []

    async def _fake_sleep(s: float) -> None:
        sleep_calls.append(s)

    monkeypatch.setattr("neutralgrid.live.decision.discord_sink.asyncio.sleep", _fake_sleep)

    sink = DiscordDigestSink(
        webhook_url="https://discord.test/x", client=client, min_interval_seconds=15.0
    )
    # First send: no wait (last_send is -inf)
    await sink.send([_result()], now=_T0)
    # Force the sink to think the previous send was just now
    sink._last_send_monotonic = time.monotonic()
    # Second send: should sleep for ~15s
    await sink.send([_result()], now=_T0)

    assert sleep_calls, "expected at least one asyncio.sleep call to throttle"
    assert sleep_calls[0] > 0


# -- Failure handling --------------------------------------------------------


@pytest.mark.asyncio
async def test_4xx_response_is_logged_but_swallowed() -> None:
    client = MagicMock()
    response = MagicMock()
    response.status_code = 401
    response.text = "unauthorized"

    import httpx

    request = httpx.Request("POST", "https://discord.test/x")
    err = httpx.HTTPStatusError("401 Unauthorized", request=request, response=httpx.Response(401, request=request))
    response.raise_for_status = MagicMock(side_effect=err)
    client.post = AsyncMock(return_value=response)
    client.aclose = AsyncMock()

    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    sent = await sink.send([_result()], now=_T0)
    assert sent is False  # Failure is reported via False, not exception


@pytest.mark.asyncio
async def test_network_error_is_logged_but_swallowed() -> None:
    import httpx

    client = MagicMock()
    client.post = AsyncMock(side_effect=httpx.ConnectError("DNS lookup failed"))
    client.aclose = AsyncMock()

    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    sent = await sink.send([_result()], now=_T0)
    assert sent is False


# -- Lifecycle ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_closes_owned_client() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    sink._own_client = True  # mark as ours so aclose actually closes it
    await sink.aclose()
    client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_long_field_value_is_truncated() -> None:
    client = _mock_async_client()
    sink = DiscordDigestSink(webhook_url="https://discord.test/x", client=client)
    huge_reason = "x" * 4000
    await sink.send(
        [_result(verdict=Verdict.ADJUST, reasons=(huge_reason,), should_emit=True)],
        now=_T0,
    )
    field_value = client.post.call_args.kwargs["json"]["embeds"][0]["fields"][0]["value"]
    assert len(field_value) <= 1000
    assert field_value.endswith("...")
