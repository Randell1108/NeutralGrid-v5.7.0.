"""Tests for live.decision.state_store (Phase A)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.state_store import (
    BotHistory,
    TickSummary,
    TICK_BUFFER_MAX,
    append_tick,
    load_history,
    sanitize_state_key,
    save_history,
    state_path,
)

_TICK_BASE = datetime(2026, 5, 6, 14, 30, 0, tzinfo=timezone.utc)


def _tick(offset_seconds: int = 0, verdict: str = "CONTINUE") -> TickSummary:
    return TickSummary(
        evaluated_at_utc=_TICK_BASE + timedelta(seconds=offset_seconds),
        verdict=verdict,
        reasons=[f"reason_{offset_seconds}"],
        price=100.0 + offset_seconds,
        range_prob=0.5,
        meta_proba=0.6,
    )


def test_round_trip_empty_history(tmp_path: Path) -> None:
    save_history("BOT-1", BotHistory.empty(), base_dir=tmp_path)
    loaded = load_history("BOT-1", base_dir=tmp_path)
    assert loaded == BotHistory.empty()


def test_round_trip_full_history(tmp_path: Path) -> None:
    h = BotHistory(
        last_verdict="ADJUST",
        consecutive=2,
        last_emitted_at=datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc),
        last_emitted_verdict="ADJUST",
        last_reasons=["range_prob_borderline:0.40", "price_near_upper:7.5pct"],
        last_escalated=False,
        consecutive_fetch_errors=1,
        consecutive_micro_failures=0,
        ticks=[_tick(0, "ADJUST"), _tick(300, "ADJUST")],
    )
    save_history("BOT-1", h, base_dir=tmp_path)
    loaded = load_history("BOT-1", base_dir=tmp_path)

    assert loaded.last_verdict == "ADJUST"
    assert loaded.consecutive == 2
    assert loaded.last_emitted_at == datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc)
    assert loaded.last_reasons == ["range_prob_borderline:0.40", "price_near_upper:7.5pct"]
    assert loaded.consecutive_fetch_errors == 1
    assert len(loaded.ticks) == 2
    assert loaded.ticks[0].verdict == "ADJUST"
    assert loaded.ticks[0].price == 100.0


def test_load_history_returns_empty_when_file_missing(tmp_path: Path) -> None:
    loaded = load_history("nonexistent-bot", base_dir=tmp_path)
    assert loaded == BotHistory.empty()


def test_load_history_returns_empty_on_corrupt_file(tmp_path: Path) -> None:
    path = state_path("BOT-1", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")

    loaded = load_history("BOT-1", base_dir=tmp_path)
    assert loaded == BotHistory.empty()


def test_atomic_write_failure_leaves_original_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = BotHistory(last_verdict="CONTINUE", consecutive=1, last_reasons=["initial"])
    save_history("BOT-1", initial, base_dir=tmp_path)
    target = state_path("BOT-1", base_dir=tmp_path)
    original_payload = target.read_text(encoding="utf-8")

    # Force os.replace to fail mid-write
    real_replace = os.replace

    def _boom(src: str, dst: str) -> None:  # pragma: no cover - lambda-ish
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", _boom)

    new = BotHistory(last_verdict="END", consecutive=1, last_reasons=["price_outside_grid:120"])
    with pytest.raises(OSError):
        save_history("BOT-1", new, base_dir=tmp_path)

    # Restore replace and confirm the original file is unchanged
    monkeypatch.setattr(os, "replace", real_replace)
    assert target.read_text(encoding="utf-8") == original_payload

    # And the orphan tempfile is cleaned up
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".") and p.name.endswith(".tmp")]
    assert leftover == []


def test_append_tick_caps_at_buffer_max() -> None:
    h = BotHistory.empty()
    for i in range(TICK_BUFFER_MAX + 5):
        h = append_tick(h, _tick(i))
    assert len(h.ticks) == TICK_BUFFER_MAX
    # Oldest entries should have been dropped FIFO. We appended offsets 0..24;
    # the buffer keeps the last 20 → offsets 5..24.
    assert h.ticks[0].evaluated_at_utc == _TICK_BASE + timedelta(seconds=5)
    assert h.ticks[-1].evaluated_at_utc == _TICK_BASE + timedelta(seconds=TICK_BUFFER_MAX + 4)


def test_sanitize_state_key_strips_reserved_chars() -> None:
    raw = "ETHUSDT__2026-05-06T14:30:00+00:00"
    sanitized = sanitize_state_key(raw)
    for ch in (":", "/", "\\", "*", "?", '"', "<", ">", "|"):
        assert ch not in sanitized
    assert "ETHUSDT" in sanitized


def test_state_path_uses_default_dir_when_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Don't actually write — just confirm the path resolves under DEFAULT_STATE_DIR
    monkeypatch.chdir(tmp_path)
    p = state_path("BOT-1")
    assert "live_decisions" in str(p)
    assert "BOT-1.json" in p.name


def test_history_to_dict_round_trip_preserves_types() -> None:
    h = BotHistory(
        last_verdict="END",
        consecutive=4,
        last_emitted_at=datetime(2026, 5, 6, 14, 30, tzinfo=timezone.utc),
        last_emitted_verdict="END",
        last_reasons=["price_outside_grid:120"],
        last_escalated=True,
        consecutive_fetch_errors=0,
        consecutive_micro_failures=2,
        ticks=[_tick(60, "ADJUST")],
    )
    d = h.to_dict()
    # Must be JSON-serializable
    json.dumps(d)
    h2 = BotHistory.from_dict(d)
    assert h2 == h


# -- v1.1 hysteresis state fields (GATEFIX-02) --------------------------------


def test_history_round_trip_with_v1_1_price_state() -> None:
    h = BotHistory(
        last_verdict="ADJUST",
        consecutive=2,
        price_outside_since=datetime(2026, 7, 13, 10, 0, tzinfo=timezone.utc),
        consecutive_price_outside=7,
        consecutive_price_inside=0,
        price_end_latched=True,
        ticks=[_tick(60, "ADJUST")],
    )
    d = h.to_dict()
    json.dumps(d)
    h2 = BotHistory.from_dict(d)
    assert h2 == h
    assert h2.price_outside_since is not None
    assert h2.price_outside_since.tzinfo is not None


def test_history_round_trip_with_profit_observation_state() -> None:
    observed_at = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)
    h = BotHistory(
        profit_deploy_ts=observed_at - timedelta(hours=1),
        peak_total_profit_usdt=6.74,
        peak_total_profit_at=observed_at,
        last_total_profit_usdt=0.90,
        last_total_profit_at=observed_at + timedelta(minutes=10),
        pnl_observation_count=3,
    )

    h2 = BotHistory.from_dict(h.to_dict())

    assert h2 == h


def test_pre_v1_1_state_dict_loads_with_defaults() -> None:
    """A v1.0-era state file (no hysteresis keys) must load unchanged with the
    new fields defaulted — backward compatibility by construction."""
    d = {
        "last_verdict": "CONTINUE",
        "consecutive": 3,
        "last_emitted_at": "2026-05-06T14:30:00+00:00",
        "last_emitted_verdict": "CONTINUE",
        "last_reasons": [],
        "last_escalated": False,
        "consecutive_fetch_errors": 0,
        "consecutive_micro_failures": 0,
        "ticks": [],
    }
    h = BotHistory.from_dict(d)
    assert h.price_outside_since is None
    assert h.consecutive_price_outside == 0
    assert h.consecutive_price_inside == 0
    assert h.price_end_latched is False


def test_naive_timestamps_are_normalized_to_utc_on_load() -> None:
    """A NAIVE ISO timestamp in a persisted file must load as aware UTC —
    otherwise decide()'s aware-minus-naive subtraction would crash the whole
    scanner loop (adversarial review finding, 2026-07-13)."""
    d = {
        "last_verdict": "ADJUST",
        "consecutive": 1,
        "last_emitted_at": "2026-07-13T05:00:00",       # naive
        "price_outside_since": "2026-07-13T04:00:00",   # naive
        "consecutive_price_outside": 2,
        "ticks": [
            {
                "evaluated_at_utc": "2026-07-13T05:00:00",  # naive
                "verdict": "ADJUST",
                "reasons": [],
                "price": 1.0,
                "range_prob": None,
                "meta_proba": None,
            }
        ],
    }
    h = BotHistory.from_dict(d)
    assert h.last_emitted_at is not None and h.last_emitted_at.tzinfo is not None
    assert h.price_outside_since is not None and h.price_outside_since.tzinfo is not None
    assert h.ticks[0].evaluated_at_utc.tzinfo is not None
    # aware arithmetic must not raise
    _ = datetime(2026, 7, 13, 6, 0, tzinfo=timezone.utc) - h.price_outside_since
