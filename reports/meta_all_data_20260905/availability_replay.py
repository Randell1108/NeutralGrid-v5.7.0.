"""Independent latest-cohort replay after recorded artifact availability."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pandas as pd
import replay


def main():
    out = Path(__file__).resolve().parent
    queue = cast(pd.DataFrame, pd.read_pickle(out / "_tmp/snapshot_replay_queue.pkl"))
    queue = queue.loc[queue.scan_time >= pd.Timestamp("2026-09-04", tz="UTC")].copy()
    records = []
    for row in queue.to_dict("records"):
        source = Path(row["study_snapshot_path"])
        start = pd.Timestamp(row["study_snapshot_mtime"])
        manifest = source.parent / "validation/pipeline_run_manifest.json"
        if manifest.exists():
            generated = json.loads(manifest.read_text()).get("generated_at_utc")
            if generated:
                start = max(start, pd.Timestamp(generated))
        row["scan_time"] = start
        assert isinstance(start, pd.Timestamp)
        row["candidate_available_ts_utc"] = start.isoformat()
        row["candidate_available_source"] = "recorded_artifact_mtime_and_validation_manifest"
        records.append(row)
    target = out / "availability"
    (target / "_tmp").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_pickle(target / "_tmp/snapshot_replay_queue.pkl")
    pd.DataFrame(records)[["candidate_id", "candidate_available_ts_utc", "candidate_available_source", "study_snapshot_path"]].to_csv(target / "availability_evidence.csv", index=False)
    replay.OUT = target
    asyncio.run(replay.main())


if __name__ == "__main__":
    main()
