"""Async Discord webhook sink for per-tick decision digests.

One embed per tick. Fields are emitted ONLY for results whose
``should_emit`` is true (the recommender's cool-down logic decides this).
Footer carries the aggregate count across the whole fleet so the user knows
how many bots are being monitored even when most are suppressed.

Token bucket: 1 message per ``min_interval_seconds`` (default 15s). Discord
allows ~30/min per webhook; a 5-minute scan cadence won't get close, but the
limiter keeps us safe under burstier configurations or after re-emit storms.

Failure handling: invalid/unreachable webhook -> log warning + swallow. The
console + JSONL sinks always continue to flow.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from neutralgrid.core.constants import DECISION_CONTRACT_VERSION
from neutralgrid.live.decision.renderer import ScanResult

logger = logging.getLogger(__name__)

# Discord embed colors. Severity-style mapping driven by the worst-verdict
# present in the digest (END > ADJUST > CONTINUE).
COLOR_GREEN = 0x2ECC71
COLOR_YELLOW = 0xF1C40F
COLOR_RED = 0xE74C3C
COLOR_GREY = 0x95A5A6

# Discord field-value caps at 1024 chars; we cap at 1000 to leave room.
_FIELD_VALUE_CAP = 1000


class DiscordDigestSink:
    """Async sink that posts one digest embed per tick."""

    def __init__(
        self,
        webhook_url: Optional[str],
        *,
        min_interval_seconds: float = 15.0,
        post_timeout_seconds: float = 10.0,
        client: Optional[Any] = None,  # httpx.AsyncClient | None
    ) -> None:
        self.webhook_url = webhook_url
        self._min_interval = float(min_interval_seconds)
        self._post_timeout = float(post_timeout_seconds)
        self._client = client
        self._own_client = client is None
        self._lock = asyncio.Lock()
        self._last_send_monotonic: float = float("-inf")
        self._warned_missing = False

    async def send(self, results: Sequence[ScanResult], *, now: datetime) -> bool:
        """Send a digest for *results*. Returns True if a message was sent.

        Skips silently when no result has should_emit=True (no signal).
        Logs a one-time warning when no webhook is configured.
        """
        if not self.webhook_url:
            if not self._warned_missing:
                logger.warning(
                    "No Discord webhook configured; skipping digest. "
                    "Set NEUTRALGRID_DISCORD_WEBHOOK or pass --discord-webhook."
                )
                self._warned_missing = True
            return False

        emit = [r for r in results if r.should_emit]
        if not emit:
            return False

        payload = self._build_payload(results=results, emit=emit, now=now)
        async with self._lock:
            await self._wait_for_token()
            ok = await self._post(payload)
            self._last_send_monotonic = time.monotonic()
        return ok

    async def aclose(self) -> None:
        if self._client is not None and self._own_client:
            try:
                await self._client.aclose()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("DiscordDigestSink.aclose() raised: %s", e)
            self._client = None

    # ------------------------------------------------------------------

    async def _wait_for_token(self) -> None:
        elapsed = time.monotonic() - self._last_send_monotonic
        wait = self._min_interval - elapsed
        if wait > 0:
            await asyncio.sleep(wait)

    def _build_payload(
        self,
        *,
        results: Sequence[ScanResult],
        emit: Sequence[ScanResult],
        now: datetime,
    ) -> dict[str, Any]:
        n_continue = sum(1 for r in results if r.recommendation.verdict.value == "CONTINUE")
        n_adjust = sum(1 for r in results if r.recommendation.verdict.value == "ADJUST")
        n_end = sum(1 for r in results if r.recommendation.verdict.value == "END")
        color = _color_for(emit)

        fields: list[dict[str, Any]] = []
        for r in emit:
            fields.append(_field_for(r))

        title = (
            f"Tactical Scan {now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            f" -- {len(emit)}/{len(results)} bots emitting"
        )
        footer_text = (
            f"CONTINUE={n_continue} | ADJUST={n_adjust} | END={n_end}"
            f" | contract {DECISION_CONTRACT_VERSION}"
        )

        return {
            "embeds": [
                {
                    "title": title,
                    "color": color,
                    "timestamp": now.astimezone(timezone.utc).isoformat(),
                    "fields": fields[:25],  # Discord caps at 25 fields per embed
                    "footer": {"text": footer_text},
                }
            ]
        }

    async def _post(self, payload: dict[str, Any]) -> bool:
        import httpx  # lazy import: keeps cost off when no webhook is set

        client = self._client
        own = self._own_client and client is None
        if client is None:
            client = httpx.AsyncClient(timeout=self._post_timeout)
            self._client = client
            self._own_client = True

        try:
            assert self.webhook_url is not None
            resp = await client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "?"
            # Don't log e.response.text directly: Discord error JSON has been
            # observed to echo webhook-id fragments. Cap and keep generic.
            logger.warning("Discord digest POST returned status %s", status)
        except (httpx.HTTPError, OSError, asyncio.TimeoutError) as e:
            # Don't log %s of the exception: httpx ConnectError messages
            # include the full URL on DNS / connection-refused failures.
            logger.warning(
                "Discord digest POST failed: %s", type(e).__name__
            )
        finally:
            # If we own the client and it was a one-shot (no future calls expected),
            # we keep it open for reuse across ticks; aclose() handles teardown.
            _ = own  # documented intentional reuse
        return False


def _color_for(emit: Sequence[ScanResult]) -> int:
    """Worst-verdict-wins coloring."""
    if not emit:
        return COLOR_GREY
    if any(r.recommendation.verdict.value == "END" for r in emit):
        return COLOR_RED
    if any(r.recommendation.verdict.value == "ADJUST" for r in emit):
        return COLOR_YELLOW
    return COLOR_GREEN


def _field_for(r: ScanResult) -> dict[str, Any]:
    rec = r.recommendation
    e = r.evaluation
    name = f"{r.spec.symbol} [{rec.verdict.value}]"
    if rec.escalated:
        name += " [ESCALATED]"

    parts: list[str] = []
    if rec.reasons:
        reasons_joined = ", ".join(rec.reasons)
        parts.append(f"reasons: {reasons_joined}")
    parts.append(f"consecutive: {rec.consecutive_count}")
    if e.price is not None and e.pct_inside_grid is not None:
        parts.append(f"price: {e.price:g} (pct_inside={e.pct_inside_grid:.1f}%)")
    elif e.price is not None:
        parts.append(f"price: {e.price:g}")
    if e.range_prob is not None:
        parts.append(f"range_prob: {e.range_prob:.2f}")
    if e.trend_prob is not None:
        parts.append(f"trend_prob: {e.trend_prob:.2f}")
    if rec.suggested_grid_lower is not None and rec.suggested_grid_upper is not None:
        parts.append(
            f"suggested grid: [{rec.suggested_grid_lower:g}, {rec.suggested_grid_upper:g}]"
        )
    if e.deploy_snapshot is not None:
        delta = e.deploy_snapshot.get("delta_meta_prob")
        if isinstance(delta, (int, float)):
            sign = "+" if delta >= 0 else ""
            parts.append(f"delta_meta_prob: {sign}{delta:.3f}")
    if e.micro_gate_pass is False and e.micro_reasons:
        parts.append("micro_fail: " + ", ".join(e.micro_reasons))
    if r.spec.strategy_id:
        parts.append(f"strategy_id: {r.spec.strategy_id}")

    value = "\n".join(parts)
    if len(value) > _FIELD_VALUE_CAP:
        value = value[: _FIELD_VALUE_CAP - 3] + "..."

    return {"name": name[:256], "value": value or "-", "inline": False}
