"""Per-bot tick history with atomic-write JSON persistence.

Each live bot has one state file at
``data/live_decisions/state/<sanitized_state_key>.json``. The recommender
treats history as immutable and produces a new `BotHistory` per tick; the
caller (monitor / CLI loop) is responsible for `save_history()`.

Atomic write: a temporary file in the same directory is fsynced and then
``os.replace``d over the target. A crash mid-write leaves the prior file
intact.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _parse_utc(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string into an AWARE UTC datetime, or None.

    A naive timestamp in a persisted state file (hand-edited, or written by a
    pre-v1.1 build) would otherwise crash the recommender's aware-minus-naive
    subtraction at decide() time — after load_history()'s except clause has
    already passed. Mirrors loader._coerce_deploy_ts semantics: naive is
    interpreted as UTC, aware is converted to UTC.
    """
    if not isinstance(raw, str):
        return None
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# Maximum ticks retained per bot. Older entries are dropped FIFO.
TICK_BUFFER_MAX: int = 20

# Default base directory for state files. Callers can override.
DEFAULT_STATE_DIR: Path = Path("data") / "live_decisions" / "state"

# Filesystem characters that are illegal on Windows or awkward in shells.
_RESERVED_CHARS: tuple[str, ...] = (":", "/", "\\", "*", "?", '"', "<", ">", "|")


@dataclass(frozen=True)
class TickSummary:
    """Lightweight per-tick record stored in the ring buffer."""

    evaluated_at_utc: datetime
    verdict: str  # Verdict.value
    reasons: list[str]
    price: Optional[float]
    range_prob: Optional[float]
    meta_proba: Optional[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_at_utc": self.evaluated_at_utc.isoformat(),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "price": self.price,
            "range_prob": self.range_prob,
            "meta_proba": self.meta_proba,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TickSummary:
        evaluated_at = _parse_utc(d["evaluated_at_utc"])
        if evaluated_at is None:
            raise ValueError(f"evaluated_at_utc is not an ISO string: {d['evaluated_at_utc']!r}")
        return cls(
            evaluated_at_utc=evaluated_at,
            verdict=str(d["verdict"]),
            reasons=list(d.get("reasons", [])),
            price=d.get("price"),
            range_prob=d.get("range_prob"),
            meta_proba=d.get("meta_proba"),
        )


@dataclass
class BotHistory:
    """Persisted per-bot state used by the recommender's cool-down logic.

    Verdicts are stored as plain strings (not the `Verdict` enum) so this
    module remains free of upward dependencies on `recommender`.
    """

    last_verdict: Optional[str] = None
    consecutive: int = 0
    last_emitted_at: Optional[datetime] = None
    last_emitted_verdict: Optional[str] = None
    last_reasons: list[str] = field(default_factory=list)
    last_escalated: bool = False
    consecutive_fetch_errors: int = 0
    consecutive_micro_failures: int = 0
    # Hysteresis END state (contract v1.1). Absent keys in pre-v1.1 state
    # files load as these defaults — backward compatible by construction.
    price_outside_since: Optional[datetime] = None
    consecutive_price_outside: int = 0
    consecutive_price_inside: int = 0
    price_end_latched: bool = False
    # Deduplicated private-PnL observation state. These fields are evidence
    # only; no threshold or verdict policy is encoded here.
    profit_deploy_ts: Optional[datetime] = None
    peak_total_profit_usdt: Optional[float] = None
    peak_total_profit_at: Optional[datetime] = None
    last_total_profit_usdt: Optional[float] = None
    last_total_profit_at: Optional[datetime] = None
    pnl_observation_count: int = 0
    ticks: list[TickSummary] = field(default_factory=list)

    @classmethod
    def empty(cls) -> BotHistory:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_verdict": self.last_verdict,
            "consecutive": self.consecutive,
            "last_emitted_at": (
                self.last_emitted_at.isoformat() if self.last_emitted_at is not None else None
            ),
            "last_emitted_verdict": self.last_emitted_verdict,
            "last_reasons": list(self.last_reasons),
            "last_escalated": bool(self.last_escalated),
            "consecutive_fetch_errors": int(self.consecutive_fetch_errors),
            "consecutive_micro_failures": int(self.consecutive_micro_failures),
            "price_outside_since": (
                self.price_outside_since.isoformat()
                if self.price_outside_since is not None else None
            ),
            "consecutive_price_outside": int(self.consecutive_price_outside),
            "consecutive_price_inside": int(self.consecutive_price_inside),
            "price_end_latched": bool(self.price_end_latched),
            "profit_deploy_ts": (
                self.profit_deploy_ts.isoformat()
                if self.profit_deploy_ts is not None
                else None
            ),
            "peak_total_profit_usdt": self.peak_total_profit_usdt,
            "peak_total_profit_at": (
                self.peak_total_profit_at.isoformat()
                if self.peak_total_profit_at is not None
                else None
            ),
            "last_total_profit_usdt": self.last_total_profit_usdt,
            "last_total_profit_at": (
                self.last_total_profit_at.isoformat()
                if self.last_total_profit_at is not None
                else None
            ),
            "pnl_observation_count": int(self.pnl_observation_count),
            "ticks": [t.to_dict() for t in self.ticks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BotHistory:
        return cls(
            last_verdict=d.get("last_verdict"),
            consecutive=int(d.get("consecutive", 0)),
            last_emitted_at=_parse_utc(d.get("last_emitted_at")),
            last_emitted_verdict=d.get("last_emitted_verdict"),
            last_reasons=list(d.get("last_reasons", [])),
            last_escalated=bool(d.get("last_escalated", False)),
            consecutive_fetch_errors=int(d.get("consecutive_fetch_errors", 0)),
            consecutive_micro_failures=int(d.get("consecutive_micro_failures", 0)),
            price_outside_since=_parse_utc(d.get("price_outside_since")),
            consecutive_price_outside=int(d.get("consecutive_price_outside", 0)),
            consecutive_price_inside=int(d.get("consecutive_price_inside", 0)),
            price_end_latched=bool(d.get("price_end_latched", False)),
            profit_deploy_ts=_parse_utc(d.get("profit_deploy_ts")),
            peak_total_profit_usdt=d.get("peak_total_profit_usdt"),
            peak_total_profit_at=_parse_utc(d.get("peak_total_profit_at")),
            last_total_profit_usdt=d.get("last_total_profit_usdt"),
            last_total_profit_at=_parse_utc(d.get("last_total_profit_at")),
            pnl_observation_count=int(d.get("pnl_observation_count", 0)),
            ticks=[TickSummary.from_dict(t) for t in d.get("ticks", [])],
        )


def sanitize_state_key(key: str) -> str:
    """Strip filesystem-reserved characters from a state key."""
    out = key
    for ch in _RESERVED_CHARS:
        out = out.replace(ch, "-")
    return out


def state_path(state_key: str, base_dir: Optional[Path] = None) -> Path:
    """Return the filesystem path for a given state key."""
    base = base_dir if base_dir is not None else DEFAULT_STATE_DIR
    return base / f"{sanitize_state_key(state_key)}.json"


def load_history(state_key: str, base_dir: Optional[Path] = None) -> BotHistory:
    """Load history for *state_key* or return an empty `BotHistory` if absent/corrupt."""
    path = state_path(state_key, base_dir)
    if not path.is_file():
        return BotHistory.empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return BotHistory.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning("Failed to load history at %s; starting empty: %s", path, e)
        return BotHistory.empty()


def save_history(
    state_key: str,
    history: BotHistory,
    base_dir: Optional[Path] = None,
) -> Path:
    """Persist *history* to disk atomically (tempfile + os.replace).

    Returns the final path written.
    """
    path = state_path(state_key, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(history.to_dict(), indent=2)

    # Use a tempfile in the same directory so os.replace is atomic on Windows.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the orphan tempfile; the original file is untouched.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def append_tick(history: BotHistory, tick: TickSummary) -> BotHistory:
    """Return a new `BotHistory` with *tick* appended (FIFO ring buffer)."""
    new_ticks = list(history.ticks)
    new_ticks.append(tick)
    if len(new_ticks) > TICK_BUFFER_MAX:
        new_ticks = new_ticks[-TICK_BUFFER_MAX:]
    return replace(history, ticks=new_ticks)
