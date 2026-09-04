"""Run the governed diagnostic-only HAR versus HAR-RS research stage."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.live.decision.volatility import (
    VolatilityError,
    load_volatility_contract,
)
from neutralgrid.live.decision.volatility_research import (
    VolatilityResearchError,
    augment_examples_with_semivariance,
    load_volatility_research_contract,
)
from neutralgrid.live.decision.volatility_research_evaluation import (
    load_consumed_v1_evidence,
    run_consumed_holdout_research,
)


UTC = timezone.utc
logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--base-contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_forecast_v1.json",
    )
    parser.add_argument(
        "--research-contract",
        type=Path,
        default=ROOT / "config" / "live_volatility_research_v2.json",
    )
    parser.add_argument(
        "--price-store",
        type=Path,
        default=ROOT / "data" / "price_store",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "audits" / "live_volatility_research",
    )
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def _safe_run_id(value: str) -> str:
    if Path(value).name != value or value in {"", ".", ".."}:
        raise VolatilityResearchError("--run-id must be one safe path component")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    execution_started_at = datetime.now(UTC)
    execution_started_monotonic = time.monotonic()
    args = parse_args(argv)
    run_id = _safe_run_id(
        args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    staging_dir = output_root / f".{run_id}.staging-{os.getpid()}"
    lock_path = output_root / f".{run_id}.lock"
    lock_fd: int | None = None
    try:
        if final_dir.exists():
            raise VolatilityResearchError(f"research run already exists: {final_dir}")
        try:
            lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise VolatilityResearchError(
                f"research run lock already exists: {lock_path}"
            ) from exc
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("ascii"))
        base_contract = load_volatility_contract(args.base_contract.resolve())
        research_contract = load_volatility_research_contract(
            args.research_contract.resolve(),
            base_contract=base_contract,
        )
        evidence = load_consumed_v1_evidence(
            args.base_artifact_dir.resolve(),
            base_contract=base_contract,
        )
        examples, feature_audit = augment_examples_with_semivariance(
            evidence["examples"],
            price_store=args.price_store.resolve(),
            base_contract=base_contract,
            research_contract=research_contract,
        )
        result = run_consumed_holdout_research(
            examples,
            evidence=evidence,
            base_contract=base_contract,
            research_contract=research_contract,
            output_dir=staging_dir,
            feature_audit=feature_audit,
            artifact_path_root=final_dir,
            execution_started_at_utc=execution_started_at,
            execution_started_monotonic=execution_started_monotonic,
        )
        staging_dir.replace(final_dir)
        terminal_summary = {
            "schema_version": result["schema_version"],
            "status": result["status"],
            "summary": result["summary"],
            "execution": result["execution"],
            "artifact_paths": result["artifact_paths"],
            "promotion_eligible": result["promotion_eligible"],
            "verdict_influence": result["verdict_influence"],
            "runtime_effect": result["runtime_effect"],
        }
        print(json.dumps(terminal_summary, indent=2, sort_keys=True, default=str))
        return 0
    except (
        VolatilityError,
        VolatilityResearchError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        logger.error("volatility research blocked: %s", exc)
        return 2
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.error("cannot remove research lock %s: %s", lock_path, exc)
        if staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except OSError as exc:
                logger.error("cannot remove research staging directory %s: %s", staging_dir, exc)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
