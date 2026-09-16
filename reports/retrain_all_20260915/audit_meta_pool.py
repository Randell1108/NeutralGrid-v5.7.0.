"""Validate all-row lineage and preservation before canonical pool finalization."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    manifest = json.loads((OUT / "fresh_production_pool/backtest_run_manifest.json").read_text())
    source = Path(manifest["training_file"])
    before = pd.read_csv(source, low_memory=False)
    after_path = OUT / "fresh_meta_validated_backfilled.csv"
    after = pd.read_csv(after_path, low_memory=False)
    assert before.candidate_id.is_unique and after.candidate_id.is_unique
    assert set(before.candidate_id) == set(after.candidate_id)
    assert len(after) == manifest["successful_rows"]
    before = before.set_index("candidate_id").sort_index()
    after = after.set_index("candidate_id").sort_index()
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    finite = np.isfinite(after[["range_prob", "trend_prob", "persistence_prob"]].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    valid = finite & after.hmm_artifact_version.eq(version).to_numpy() & after.hmm_feature_source.eq("pinned_artifact_replay").to_numpy()
    parts = after.index.to_series().str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scans = pd.to_datetime(parts[0] + parts[1], format="%Y%m%d%H%M%S", utc=True)
    cutoffs = pd.to_datetime(after.hmm_feature_cutoff_utc, format="mixed", utc=True)
    valid &= scans.eq(cutoffs).to_numpy()
    audit = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "active_hmm": version, "rows": len(after), "valid_rows": int(valid.sum()), "invalid_candidate_ids": after.index[~valid].tolist(), "versions": after.hmm_artifact_version.value_counts(dropna=False).to_dict(), "feature_nonfinite_counts": {f: int((~np.isfinite(pd.to_numeric(after[f], errors="coerce"))).sum()) for f in ACTIVE_SNAPSHOT_META_FEATURES}, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "backfill_sha256": hashlib.sha256(after_path.read_bytes()).hexdigest()}
    (OUT / "meta_lineage_audit.json").write_text(json.dumps(audit, indent=2, default=str)+"\n")
    assert valid.all(), "Incomplete HMM replay; do not finalize"
    assert audit["feature_nonfinite_counts"]["ev_score"] == 0, "HMM-dependent EV must be observed"
    retained = [f for f in ACTIVE_SNAPSHOT_META_FEATURES if f != "ev_score"]
    retained += [c for c in ("time_to_target_hours", "pnl_pct", "realized_net_pnl_pct", "y", "label_positive_by_horizon") if c in before]
    for column in retained:
        old = pd.to_numeric(before[column], errors="coerce").to_numpy(dtype=float)
        new = pd.to_numeric(after[column], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(old, new, equal_nan=True, rtol=1e-10, atol=1e-12), column
    audit.update(status="PASS", all_ids_preserved=True, independent_features_and_outcomes_preserved=retained, feature_cutoff="recorded candidate scan time", lineage_caveat="Pinned active-artifact replay of historical inputs; not a claim the new HMM existed at historical scan time")
    (OUT / "meta_lineage_audit.json").write_text(json.dumps(audit, indent=2, default=str)+"\n")
    print(json.dumps(audit, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
