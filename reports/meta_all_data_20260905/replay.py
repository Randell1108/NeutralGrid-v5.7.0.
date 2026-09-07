"""Resumable isolated FASTWIN replays of saved scanner configurations."""
from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["NEUTRALGRID_BASE_DIR"] = str(ROOT)
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import fetch_historical_klines, run_single_backtest, convert_to_training_row


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def backtest(row, bars):
    logging.getLogger().setLevel(logging.ERROR)
    result = run_single_backtest(row, bars, capital=400., leverage=10, max_holding_bars=420, realism_profile="legacy")
    train = convert_to_training_row(result, row, horizon_hours=7.)
    train["study_source_pool"] = "new_snapshot_replay"
    train["study_snapshot_path"] = row["study_snapshot_path"]
    train["study_snapshot_sha256"] = row["study_snapshot_sha256"]
    train["study_replay_contract"] = "existing_scan_geometry_7h_legacy_research_v1"
    train["study_promotion_eligible"] = False
    return safe({"status": "success", "candidate_id": row["candidate_id"], "training": train,
                 "outcome_start": str(bars.timestamp.iloc[0]), "outcome_end": str(bars.timestamp.iloc[-1]),
                 "bars": len(bars), "fetched_at_utc": datetime.now(timezone.utc).isoformat()})


async def main():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    queue = cast(pd.DataFrame, pd.read_pickle(OUT / "_tmp/snapshot_replay_queue.pkl")).sort_values("scan_time", ascending=False)
    checkpoints = OUT / "_tmp/replay_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    client = BinanceClient()
    client.api_key = ""
    client.api_secret = ""
    lock = asyncio.Lock()
    last_request = 0.
    semaphore = asyncio.Semaphore(6)
    statuses = []
    completed = 0

    async def one(row, executor):
        nonlocal last_request, completed
        cid = str(row["candidate_id"])
        path = checkpoints / (hashlib.sha256(cid.encode()).hexdigest() + ".json.gz")
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                status = json.load(f)
            if status["status"] != "pending_7h_window":
                return status
        start = pd.Timestamp(row["scan_time"])
        assert isinstance(start, pd.Timestamp)
        if start + pd.Timedelta(hours=7, minutes=2) > pd.Timestamp.now(tz="UTC"):
            return {"candidate_id": cid, "status": "pending_7h_window", "matures_utc": cast(pd.Timestamp, start + pd.Timedelta(hours=7, minutes=2)).isoformat()}
        async with semaphore:
            status = None
            for attempt in range(3):
                try:
                    async with lock:
                        # Keep combined request weight below the shared-IP
                        # budget while the HMM backfill is running.
                        interval = .20 if (OUT / "existing_pool_active_hmm.csv").exists() else 1.0
                        pause = interval - (time.monotonic() - last_request)
                        if pause > 0:
                            await asyncio.sleep(pause)
                        last_request = time.monotonic()
                    bars = await fetch_historical_klines(client, str(row["symbol"]), cast(datetime, start.to_pydatetime()), hours=7)
                    if len(bars) != 420:
                        status = {"candidate_id": cid, "status": "incomplete_market_window", "bars": len(bars)}
                        break
                    ts = pd.to_datetime(bars.timestamp, utc=True)
                    if not bool(ts.diff().dropna().eq(pd.Timedelta(minutes=1)).all()):
                        status = {"candidate_id": cid, "status": "noncontiguous_market_window"}
                        break
                    if ts.iloc[0] < start or ts.iloc[-1] > start + pd.Timedelta(hours=7, minutes=1):
                        status = {"candidate_id": cid, "status": "market_window_mismatch"}
                        break
                    status = await asyncio.get_running_loop().run_in_executor(executor, backtest, row, bars)
                    break
                except Exception as exc:
                    status = {"candidate_id": cid, "status": "replay_error", "error": str(exc), "attempts": attempt + 1}
                    if "Invalid symbol" in str(exc) or "-1121" in str(exc):
                        break
                    await asyncio.sleep(2 * (attempt + 1))
            assert status is not None
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(status, f, default=str, allow_nan=False)
            completed += 1
            if completed % 50 == 0:
                print(json.dumps({"completed": completed, "queue": len(queue), "last_id": cid, "last_status": status["status"]}), flush=True)
            return status

    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            tasks = [one(row, executor) for row in queue.to_dict("records")]
            for task in asyncio.as_completed(tasks):
                statuses.append(await task)
    finally:
        await client.close()
    trained = [s["training"] for s in statuses if s["status"] == "success"]
    pd.DataFrame(trained).to_csv(OUT / "new_replay_training.csv", index=False)
    ledger = [{k: v for k, v in s.items() if k != "training"} for s in statuses]
    pd.DataFrame(ledger).to_csv(OUT / "new_replay_row_dispositions.csv", index=False)
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "queue_rows": len(queue),
        "status_counts": pd.Series([s["status"] for s in statuses]).value_counts().to_dict(),
        "contract": "existing_scan_geometry_7h_legacy_research_v1", "promotion_eligible": False,
        "outcomes_are_simulations_not_live_records": True, "features_from_recorded_snapshots": True, "capital": 400, "leverage": 10,
        "engine_parameters": "repository build_training_config defaults with explicit seven-hour horizon"}
    (OUT / "new_replay_manifest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
