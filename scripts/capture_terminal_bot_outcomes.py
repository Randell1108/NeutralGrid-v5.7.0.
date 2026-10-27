"""Inspect original terminal UI captures; --capture publishes immutable outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neutralgrid.live.decision.evidence_cohort import read_json
from neutralgrid.live.decision.native_sar import NativeRecord
from neutralgrid.live.decision.state_action_reward import SourceRef, atomic_create_json
from neutralgrid.live.decision.terminal_outcomes import (
    TerminalObservation, TerminalStorageError, capture_terminal_outcome, parse_terminal_capture,
    score_terminal_outcome,
)

logger = logging.getLogger(__name__)


def run(sources: Sequence[Path], *, output_dir: Path, capture: bool = False,
        root: Path = ROOT) -> dict[str, Any]:
    """Root is an explicit application/test boundary, never a CLI input."""
    root = root.resolve()
    live = (root / "Live").resolve()
    reports = root / "reports"
    output = output_dir.resolve()
    if live != root / "Live" or reports.resolve() != reports:
        raise ValueError("canonical Live/reports roots must not redirect through links")
    if reports not in output.parents or output.exists():
        raise ValueError("output must be a fresh directory strictly under repository reports")
    if not sources:
        raise ValueError("at least one original source is required")
    output.mkdir(parents=True, exist_ok=False)
    cutoff = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "record_type": "terminal_outcome_capture_report", "started_at_utc": cutoff.isoformat(),
        "mode": "capture" if capture else "report_only", "results": [], "errors": [],
        "duplicates": [], "duplicate_source_count": 0,
        "training_eligible": False, "promotion_eligible": False, "influence_enabled": False,
    }
    groups: dict[str, list[tuple[NativeRecord, TerminalObservation]]] = defaultdict(list)
    bad_routes: set[tuple[str, str]] = set()
    # Validate and classify the entire batch before any terminal record writes.
    for source in sources:
        try:
            if live not in source.resolve().parents:
                raise ValueError("source must be inside canonical repository Live")
            # Bounded strict read detects duplicate JSON keys and nonfinite data.
            _, digest = read_json(source)
            native = NativeRecord(SourceRef(path=str(source.resolve()), sha256=digest))
            observation = parse_terminal_capture(native, live_root=live)
            if observation.captured_at > cutoff:
                raise ValueError("capture availability follows run cutoff")
            groups[observation.bot.key].append((native, observation))
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            if live in source.resolve().parents:
                parts = source.resolve().relative_to(live).parts
                if len(parts) == 4 and parts[2].startswith("verdict_audit_"):
                    bad_routes.add((parts[1], parts[2].removeprefix("verdict_audit_")))
            report["errors"].append({"source_path": str(source.resolve()), "error_type": type(error).__name__, "reason": str(error)})
    accepted: list[list[tuple[NativeRecord, TerminalObservation]]] = []
    for group in groups.values():
        group.sort(key=lambda item: (item[1].captured_at, item[0].source.path))
        first = group[0][1]
        if ((first.bot.symbol, first.bot.strategy_id) in bad_routes or
                any(item[1].stable_economics() != first.stable_economics() for item in group)):
            for native, _ in group:
                report["errors"].append({"source_path": native.source.path, "reason": "batch exact-bot conflict or invalid related capture; all scores quarantined"})
            continue
        accepted.append(group)
    checkpoint = 0
    for group in accepted:
        native, observation = group[0]
        entry: dict[str, Any] = {"source_path": native.source.path, "stored_path": None, "created": False}
        try:
            for member, observed in group:
                if parse_terminal_capture(member, live_root=live) != observed:
                    raise ValueError("batch source changed after validation")
            entry["score"] = score_terminal_outcome(observation, live_root=live)
            if capture:
                stored, created = capture_terminal_outcome(native, live_root=live, ingested_at=cutoff)
                entry.update(stored_path=str(stored), created=created)
            report["results"].append(entry)
            for duplicate, _ in group[1:]:
                report["duplicates"].append({"source": duplicate.source.model_dump(mode="json"), "retained_source": native.source.model_dump(mode="json"), "bot_identity": observation.bot.key})
            report["duplicate_source_count"] = len(report["duplicates"])
        except (OSError, ValueError, RuntimeError, TypeError) as error:
            failed: dict[str, Any] = {"source_path": native.source.path, "error_type": type(error).__name__, "reason": str(error)}
            if isinstance(error, TerminalStorageError) and error.published_path is not None:
                failed["existing_or_published_path"] = str(error.published_path)
                failed["publication_requires_revalidation"] = True
            report["errors"].append(failed)
        # Checkpoint each result so later failures cannot hide earlier writes.
        checkpoint += 1
        try:
            atomic_create_json(output / f"progress_{checkpoint:04d}.json", report)
        except OSError as error:
            published = Path(entry["stored_path"]) if entry["stored_path"] else None
            raise TerminalStorageError("report checkpoint failed; source outcomes may already be published", published) from error
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "partial_failure" if report["errors"] and report["results"] else "failed" if report["errors"] else "completed"
    atomic_create_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true", help="Publish verified terminal observations; default only reports scores")
    args = parser.parse_args(argv)
    try:
        report = run(args.source, output_dir=args.output_dir, capture=args.capture)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        logger.error("Terminal outcome run failed: %s. Earlier progress reports or published outcomes may exist; revalidate before retrying.", error)
        if isinstance(error, TerminalStorageError) and error.published_path is not None:
            logger.error("Existing or published outcome requiring revalidation: %s", error.published_path)
        return 2
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
