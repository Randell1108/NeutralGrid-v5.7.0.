"""Generate missing frozen-cohort outcomes through the canonical backtest wrapper."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import fetch_historical_klines, resolve_backtest_start_timestamp, run_single_backtest
from neutralgrid.core.constants import ENGINE_VERSION, FORMULA_VERSION, LABEL_CONTRACT_VERSION
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES

OUT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("backtest_realistic").setLevel(logging.WARNING)
LOG = logging.getLogger("profile_additional_backtest")


async def main() -> None:
    wanted = set(json.loads((OUT / "additional_candidate_ids.json").read_text()))
    candidates = [json.loads(line) for line in (OUT / "frozen_candidate_payloads.jsonl").read_text().splitlines()]
    candidates = [row for row in candidates if row["candidate_id"] in wanted]
    results_path = OUT / "additional_backtest_results.jsonl"
    completed = set()
    if results_path.exists():
        completed = {json.loads(line)["candidate_id"] for line in results_path.read_text().splitlines()}
    pending = [row for row in candidates if row["candidate_id"] not in completed]
    client = BinanceClient()
    semaphore = asyncio.Semaphore(4)
    failures = []
    finished = len(completed)
    with results_path.open("a", encoding="utf-8") as output:
        async def one(row: dict) -> None:
            nonlocal finished
            cid = row["candidate_id"]
            async with semaphore:
                for attempt in range(1, 4):
                    try:
                        start = resolve_backtest_start_timestamp(row)
                        klines = await fetch_historical_klines(client, row["symbol"], start, hours=7)
                        if len(klines) < 420:
                            raise ValueError(f"Incomplete seven-hour window: {len(klines)} bars")
                        result = run_single_backtest(row, klines, capital=400.0, leverage=10, max_holding_bars=420, realism_profile="legacy")
                        for key, expected in {"engine_version": ENGINE_VERSION, "formula_version": FORMULA_VERSION, "label_contract_version": LABEL_CONTRACT_VERSION, "realism_profile": "legacy", "mode": "geometric"}.items():
                            assert result[key] == expected, (cid, key, result[key])
                        assert bool(result["is_authoritative"])
                        result.update({feature: float(row[feature]) for feature in DEFAULT_FEATURES})
                        result.update({"candidate_id": cid, "source": "backtest", "start_time_utc": start.isoformat(), "scan_timestamp": row["scan_timestamp"], "profile_split": row["split"], "original_source_path": row["original_source_path"], "original_source_sha256": row["original_source_sha256"], "profile_feature_source": "recorded_scanner_snapshot", "canonical_wrapper": "neutralgrid.backtest.candidate_pipeline.run_single_backtest", "attempt": attempt})
                        output.write(json.dumps(result, default=str, ensure_ascii=True) + "\n")
                        output.flush()
                        finished += 1
                        if finished % 25 == 0 or finished == len(candidates):
                            LOG.info("Saved %d/%d additional outcomes", finished, len(candidates))
                        return
                    except Exception as exc:
                        LOG.warning("Candidate %s attempt %d failed: %r", cid, attempt, exc)
                        if attempt == 3:
                            failures.append({"candidate_id": cid, "attempts": attempt, "error": repr(exc)})
                        else:
                            await asyncio.sleep(2 * attempt)
        try:
            await asyncio.gather(*(one(row) for row in pending))
        finally:
            await client.close()
    manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "requested_candidates": len(candidates), "successful_rows": finished, "failures": failures, "hours": 7, "min_bars": 420, "realism_profile": "legacy", "capital": 400, "leverage": 10, "max_concurrency": 4, "entrypoint": "run_single_backtest -> build_training_config -> run_backtest", "results_path": str(results_path), "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest()}
    (OUT / "additional_backtest_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info("Backtesting finished: %s", manifest)


if __name__ == "__main__":
    asyncio.run(main())
