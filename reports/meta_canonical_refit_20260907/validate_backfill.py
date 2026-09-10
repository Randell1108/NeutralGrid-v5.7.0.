"""Read-only lineage, feature-preservation and timestamp audit for this refit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.training.data_generator import HMM_FEATURE_SEMANTICS_VERSION

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent


def lineage_verdict(frame: pd.DataFrame, active: str) -> str:
    if "hmm_artifact_version" not in frame:
        return "INCOMPLETE"
    versions = frame.hmm_artifact_version.astype("string")
    if versions.nunique(dropna=False) != 1:
        return "SPLIT"
    if not versions.eq(active).all():
        return "STALE"
    for column in ("range_prob", "trend_prob", "persistence_prob"):
        if column not in frame or not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            return "INCOMPLETE"
    return "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backfilled", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    active = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    metadata = json.loads((ROOT / "artifacts/hmm" / active / "metadata.json").read_text())
    source = pd.read_csv(args.source, low_memory=False).set_index("candidate_id")
    frame = pd.read_csv(args.backfilled, low_memory=False).set_index("candidate_id")
    assert source.index.is_unique and frame.index.is_unique
    assert set(source.index) == set(frame.index)
    frame = frame.loc[source.index]
    verdict = lineage_verdict(frame, active)
    synthetic = pd.DataFrame({"hmm_artifact_version": [active, "different"], "range_prob": [0.5, 0.5], "trend_prob": [0.5, 0.5], "persistence_prob": [0.5, 0.5]})
    assert lineage_verdict(synthetic, active) == "SPLIT"
    synthetic.loc[1, "hmm_artifact_version"] = active
    synthetic.loc[1, "range_prob"] = np.nan
    assert lineage_verdict(synthetic, active) == "INCOMPLETE"
    expected = {
        "hmm_feature_source": "pinned_artifact_replay",
        "hmm_replay_scope": "hmm_lineage_only",
        "hmm_feature_semantics_version": HMM_FEATURE_SEMANTICS_VERSION,
    }
    checks = {column: bool(frame[column].astype(str).eq(str(value)).all()) for column, value in expected.items()}
    checks["trained_at_matches_artifact"] = bool(pd.to_datetime(frame.hmm_trained_at_utc, utc=True, format="ISO8601").eq(pd.Timestamp(metadata["trained_at_utc"])).all())
    parts = frame.index.to_series().astype("string").str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scan = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True)
    cutoff = pd.to_datetime(frame.hmm_feature_cutoff_utc, utc=True, format="ISO8601")
    checks["cutoff_equals_scan_timestamp"] = bool(cutoff.eq(scan).all())
    checks["cutoff_no_later_than_outcome_start"] = bool(cutoff.le(pd.to_datetime(frame.start_time_utc, utc=True, format="ISO8601")).all())
    changed, missing = {}, {}
    for feature in ACTIVE_SNAPSHOT_META_FEATURES:
        before = pd.to_numeric(source[feature], errors="coerce")
        after = pd.to_numeric(frame[feature], errors="coerce")
        changed[feature] = int((~np.isclose(before, after, equal_nan=True, rtol=1e-10, atol=1e-12)).sum())
        missing[feature] = int((~np.isfinite(after)).sum())
    checks["independent_active_features_preserved"] = all(count == 0 for feature, count in changed.items() if feature != "ev_score")
    original_columns = [c for c in ("label", "meta_label", "time_to_target_hours", "pnl_pct", "duration_hours", "backtest_start_ts_utc", "start_time_utc") if c in source]
    outcome_numeric_differences = {}
    for column in original_columns:
        if pd.api.types.is_numeric_dtype(source[column]) and pd.api.types.is_numeric_dtype(frame[column]):
            outcome_numeric_differences[column] = float((source[column] - frame[column]).abs().max())
            checks[f"outcome_preserved:{column}"] = bool(np.isclose(source[column], frame[column], equal_nan=True, rtol=1e-12, atol=1e-12).all())
        else:
            checks[f"outcome_preserved:{column}"] = bool(source[column].fillna("<NA>").astype(str).eq(frame[column].fillna("<NA>").astype(str)).all())
    audit = {
        "source": str(args.source.resolve()), "backfilled": str(args.backfilled.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "backfilled_sha256": hashlib.sha256(args.backfilled.read_bytes()).hexdigest(),
        "rows": len(frame), "active_hmm": active, "lineage_verdict": verdict,
        "versions": frame.hmm_artifact_version.value_counts(dropna=False).to_dict(),
        "checks": checks, "active_feature_changes": changed, "missing_active_features": missing,
        "outcome_numeric_max_absolute_differences": outcome_numeric_differences,
        "negative_control_tests": "mixed lineage -> SPLIT; missing probability -> INCOMPLETE",
        "scan_start_utc": scan.min().isoformat(), "scan_end_utc": scan.max().isoformat(),
        "historical_replay_caveat": "Active HMM was fitted after most historical scans; this is the canonical pinned-artifact refit, not a point-in-time HMM training simulation.",
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    assert verdict == "PASS" and all(checks.values()), "Input validation failed; inspect audit"


if __name__ == "__main__":
    main()
