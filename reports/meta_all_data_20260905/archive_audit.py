"""Retain older bot identities absent from the current workbook."""
from __future__ import annotations

import json
import pandas as pd
import numpy as np

import experiment as ex
from workbook_audit import utc


def main():
    archive = pd.read_csv(ex.OUT / "archived_extra_bot_rows.csv", low_memory=False)
    current = pd.read_csv(ex.OUT / "workbook_actual_evidence.csv")
    assert not set(archive.strategy_id) & set(current.strategy_id)
    rows = []
    allowed = ["adx_1h", "adx_15m", "adx_5m", "rsi_15m", "ema_slope_1h", "ema_crosses_5m", "vwap_crosses_5m", "range_size_pct", "bb_width", "grid_spacing_pct"]
    for row in archive.to_dict("records"):
        start, end = utc(row["start_time_utc"]), utc(row["end_time_utc"])
        hours = (end - start).total_seconds() / 3600
        positive = 0 < hours <= 7 and 0 < float(row["duration_hours"]) <= 7 and float(row["pnl_pct"]) >= 3
        result = {"strategy_id": row["strategy_id"], "candidate_id": row.get("candidate_id"), "symbol": row["symbol"], "mode": row.get("mode"),
            "scan_time": start, "event_end": start + pd.Timedelta(hours=7), "actual_end_time": end,
            "elapsed_hours": hours, "recorded_duration_hours": row["duration_hours"], "recorded_pnl_pct": row["pnl_pct"],
            "fast_winner_target": 1. if positive else np.nan,
            "label_evidence": "archived net terminal profit >=3% within exact seven-hour window" if positive else "unknown: full timed net-MTM path not established",
            "label_source": row["archive_source"], "time_to_target_hours": np.nan,
            "study_source_pool": "archived_actual_bot_positive_bound"}
        result.update({f: row.get(f, np.nan) if f in allowed else np.nan for f in ex.ALL})
        result["num_grids"] = row.get("grids_count", np.nan)
        rows.append(result)
    data = pd.DataFrame(rows)
    data.to_csv(ex.OUT / "archived_bot_evidence.csv", index=False)
    ex.write("archived_bot_audit.json", {"additional_strategy_ids": len(data), "proven_positive_labels": int(data.fast_winner_target.eq(1).sum()),
        "unknown_labels": int(data.fast_winner_target.isna().sum()),
        "timing_policy": "Only within-bot elapsed timestamps establish positive labels; saved measurements are diagnostic, with availability unverified.",
        "not_merged_into_original_workbook": True})
    print(json.dumps({"archive_rows": len(data), "positives": int(data.fast_winner_target.eq(1).sum())}), flush=True)


if __name__ == "__main__":
    main()
