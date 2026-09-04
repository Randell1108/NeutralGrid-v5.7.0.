"""Output renderers for the tactical live decision tool.

Two sinks:
* `ConsoleRenderer` -- formats a list of `ScanResult`s into a human-readable
  table, mirroring the style of `run_full_pipeline._display_results`.
* `JsonlWriter` -- appends one JSON object per `ScanResult` to a daily
  rollover file at ``logs/live_decisions_YYYYMMDD.jsonl``.

Discord webhook output lives in `discord_sink.py` (Phase C).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from neutralgrid.core.constants import DECISION_CONTRACT_VERSION
from neutralgrid.live.decision.loader import LiveBotSpec
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    Recommendation,
)

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("logs")


@dataclass(frozen=True)
class ScanResult:
    """One bot's evaluation + recommendation + emission decision for one tick."""

    spec: LiveBotSpec
    evaluation: BotEvaluation
    recommendation: Recommendation
    should_emit: bool
    diagnostics: tuple[str, ...] = field(default=())


# -- Console ----------------------------------------------------------------


class ConsoleRenderer:
    """Renders a tick's results to a single string suitable for stdout."""

    HEADER = (
        f"{'SYMBOL':<10} {'VERDICT':<9} {'CONSEC':<6} {'ESCAL':<5} "
        f"{'PRICE':>10} {'PCT_IN':>7} {'RNG_P':>6} {'TRD_P':>6} REASONS"
    )

    def render(self, results: Sequence[ScanResult], *, now: datetime) -> str:
        lines: list[str] = []
        lines.append(
            f"=== live_decision_scanner tick @ {now.isoformat()} "
            f"(contract {DECISION_CONTRACT_VERSION}) ==="
        )
        if not results:
            lines.append("(no bots to evaluate)")
            return "\n".join(lines)

        n_continue = sum(1 for r in results if r.recommendation.verdict.value == "CONTINUE")
        n_adjust = sum(1 for r in results if r.recommendation.verdict.value == "ADJUST")
        n_end = sum(1 for r in results if r.recommendation.verdict.value == "END")
        n_emit = sum(1 for r in results if r.should_emit)
        lines.append(
            f"bots: {len(results)} | "
            f"CONTINUE={n_continue} ADJUST={n_adjust} END={n_end} | "
            f"emitted={n_emit}"
        )
        lines.append(self.HEADER)
        lines.append("-" * len(self.HEADER))
        for r in results:
            lines.append(self._render_row(r))
        return "\n".join(lines)

    def _render_row(self, r: ScanResult) -> str:
        e = r.evaluation
        rec = r.recommendation
        symbol = r.spec.symbol[:10]
        verdict = rec.verdict.value
        consec = str(rec.consecutive_count)
        escal = "Y" if rec.escalated else "-"
        price = _fmt_optional(e.price, "{:>10.4f}", "{:>10}")
        pct_in = _fmt_optional(e.pct_inside_grid, "{:>7.1f}", "{:>7}")
        rng_p = _fmt_optional(e.range_prob, "{:>6.2f}", "{:>6}")
        trd_p = _fmt_optional(e.trend_prob, "{:>6.2f}", "{:>6}")
        reasons_str = ", ".join(rec.reasons) if rec.reasons else "-"
        emit_marker = "*" if r.should_emit else " "
        return (
            f"{emit_marker}{symbol:<9} {verdict:<9} {consec:<6} {escal:<5} "
            f"{price} {pct_in} {rng_p} {trd_p} {reasons_str}"
        )


def _fmt_optional(v: Optional[float], fmt: str, na_fmt: str) -> str:
    if v is None:
        return na_fmt.format("--")
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return na_fmt.format("--")


# -- JSONL ------------------------------------------------------------------


class JsonlWriter:
    """Append-only JSONL sink with UTC daily rollover.

    Files live at ``<base_dir>/live_decisions_YYYYMMDD.jsonl``. Each call to
    `append()` writes exactly one line (one JSON object) and flushes.

    Concurrency: append-mode opens are atomic for line-sized writes on POSIX
    and on Windows (NTFS) for our payload sizes (well under PIPE_BUF / 4 KB
    sector). One file handle is kept open and rotated when the UTC date
    changes mid-run.
    """

    def __init__(self, base_dir: Path = DEFAULT_LOG_DIR) -> None:
        self._base_dir = base_dir
        self._current_date: Optional[str] = None
        self._fh: Any = None  # io.TextIOBase | None

    def path_for(self, dt: datetime) -> Path:
        """Return the JSONL file path for the UTC date of *dt*."""
        date_part = dt.astimezone(timezone.utc).strftime("%Y%m%d")
        return self._base_dir / f"live_decisions_{date_part}.jsonl"

    def append(self, result: ScanResult, *, now: datetime) -> None:
        """Append one result as a single JSON line. Rotates on UTC date change."""
        date_part = now.astimezone(timezone.utc).strftime("%Y%m%d")
        if self._current_date != date_part:
            self._rotate(date_part)
        assert self._fh is not None  # set by _rotate
        line = json.dumps(_to_jsonl_record(result, now), default=_json_default)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._fh = None
            self._current_date = None

    def _rotate(self, date_part: str) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._base_dir.mkdir(parents=True, exist_ok=True)
        path = self._base_dir / f"live_decisions_{date_part}.jsonl"
        self._fh = path.open("a", encoding="utf-8")
        self._current_date = date_part

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _to_jsonl_record(result: ScanResult, now: datetime) -> dict[str, Any]:
    e = result.evaluation
    rec = result.recommendation
    candidate_id = _resolved_candidate_id(result)
    return {
        "ts": now.astimezone(timezone.utc).isoformat(),
        "contract_version": DECISION_CONTRACT_VERSION,
        "symbol": result.spec.symbol,
        "strategy_id": result.spec.strategy_id,
        "candidate_id": candidate_id,
        "execution_telemetry": (
            result.spec.execution_telemetry.to_dict()
            if result.spec.execution_telemetry is not None
            else None
        ),
        "verdict": rec.verdict.value,
        "reasons": list(rec.reasons),
        "should_emit": result.should_emit,
        "consecutive_count": rec.consecutive_count,
        "escalated": rec.escalated,
        "suggested_grid_lower": rec.suggested_grid_lower,
        "suggested_grid_upper": rec.suggested_grid_upper,
        # D4/D7 shadow audit: whether the meta tilt would fire and whether it
        # actually influenced this verdict (only when meta_tilt_enabled).
        "meta_would_tilt": rec.meta_would_tilt,
        "meta_influenced_verdict": rec.meta_influenced_verdict,
        "profit_deterioration": (
            rec.profit_deterioration.to_dict()
            if rec.profit_deterioration is not None
            else None
        ),
        "evaluation": {
            "evaluated_at_utc": e.evaluated_at_utc.astimezone(timezone.utc).isoformat(),
            "grid_lower": e.grid_lower,
            "grid_upper": e.grid_upper,
            "price": e.price,
            "pct_inside_grid": e.pct_inside_grid,
            "dist_to_lower_pct": e.dist_to_lower_pct,
            "dist_to_upper_pct": e.dist_to_upper_pct,
            "range_prob": e.range_prob,
            "trend_prob": e.trend_prob,
            "persistence_prob": e.persistence_prob,
            "meta_proba": e.meta_proba,
            "utility_score": e.utility_score,
            "micro_gate_pass": e.micro_gate_pass,
            "micro_reasons": list(e.micro_reasons),
            "symbol_unavailable": e.symbol_unavailable,
            "transient_fetch_error": e.transient_fetch_error,
            "hmm_artifact_missing": e.hmm_artifact_missing,
            "diagnostics": list(e.diagnostics),
            "deploy_snapshot": e.deploy_snapshot,
            # D7 live meta audit evidence (write-only; never read into a verdict).
            "meta_authoritative": e.meta_authoritative,
            "meta_full_fidelity": e.meta_full_fidelity,
            "meta_missing_features": list(e.meta_missing_features),
            "meta_source": e.meta_source,
            "active_hmm_version": e.active_hmm_version,
            "linked_hmm_version": e.linked_hmm_version,
            "meta_feature_profile": e.meta_feature_profile,
            "l2_risk": e.l2_risk.to_dict() if e.l2_risk is not None else None,
            "private_event_evidence": (
                e.private_event_evidence.to_dict()
                if e.private_event_evidence is not None
                else None
            ),
            "execution_risk": (
                e.execution_risk.to_dict()
                if e.execution_risk is not None
                else None
            ),
        },
    }


def _resolved_candidate_id(result: ScanResult) -> Optional[str]:
    if result.spec.candidate_id:
        return result.spec.candidate_id
    deploy_snapshot = result.evaluation.deploy_snapshot
    if not isinstance(deploy_snapshot, dict):
        return None
    candidate = deploy_snapshot.get("candidate_id")
    if candidate is None:
        return None
    candidate_text = str(candidate).strip()
    return candidate_text or None


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.astimezone(timezone.utc).isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON-serializable")
