"""Reuse verified identical active-HMM replays, then run canonical finalization/refit."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.backfill_training_features import HMM_DEPENDENT_BACKFILL_COLUMNS


def run(args: list[str], log: str) -> None:
    with (OUT / log).open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (log, result.returncode)


def main() -> None:
    manifest_path = OUT / "fresh_production_pool/backtest_run_manifest.json"
    replay_log = OUT / "meta_replay_queue.log"
    while not manifest_path.exists() or "Fresh meta pool replay completed" not in replay_log.read_text(encoding="utf-8"):
        time.sleep(5)
    manifest = json.loads(manifest_path.read_text())
    source = pd.read_csv(manifest["training_file"], low_memory=False).set_index("candidate_id")
    cache = pd.read_csv(OUT / "fresh_meta_backfilled.csv",low_memory=False).set_index("candidate_id")
    assert source.index.is_unique and cache.index.is_unique
    cache = cache.reindex(source.index)
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    probability_columns = ["range_prob", "trend_prob", "persistence_prob", "ev_score"]
    reusable = cache.hmm_artifact_version.eq(version) & cache.hmm_feature_source.eq("pinned_artifact_replay")
    reusable &= np.isfinite(cache[probability_columns].apply(pd.to_numeric,errors="coerce").to_numpy()).all(axis=1)
    parts = source.index.to_series().str.extract(r"^[^_]+_(\d{8})_(\d{6})(?:_|$)")
    scan = pd.to_datetime(parts[0]+parts[1],format="%Y%m%d%H%M%S",utc=True)
    start = pd.to_datetime(source.start_time_utc,format="mixed",utc=True)
    assert scan.le(start).all(), "Corrected fresh source still contains noncausal timestamps"
    reusable &= pd.to_datetime(cache.hmm_feature_cutoff_utc,format="mixed",utc=True,errors="coerce").eq(scan)
    checked = list(dict.fromkeys([f for f in ACTIVE_SNAPSHOT_META_FEATURES if f != "ev_score"] + ["survival_prob", "grid_lower", "grid_upper", "lower", "upper", "grids_count", "leverage"]))
    for column in checked:
        old = pd.to_numeric(cache[column],errors="coerce").to_numpy(dtype=float) if column in cache else np.full(len(cache),np.nan)
        new = pd.to_numeric(source[column],errors="coerce").to_numpy(dtype=float) if column in source else np.full(len(source),np.nan)
        reusable &= np.isclose(old,new,rtol=1e-12,atol=1e-14,equal_nan=True)
    for column in [*HMM_DEPENDENT_BACKFILL_COLUMNS,"ev_contract_fingerprint"]:
        if column not in cache:
            continue
        if column not in source:
            source[column] = pd.Series(index=source.index,dtype=cache[column].dtype)
        source.loc[reusable,column] = cache.loc[reusable,column]
    input_path = OUT / "fresh_meta_validated_input.csv"
    source.reset_index().to_csv(input_path,index=False)
    record = {"rows":len(source),"reused_identical_active_hmm_rows":int(reusable.sum()),"rows_requiring_inference":int((~reusable).sum()),"checked_independent_inputs":checked,"rule":"Same candidate scan time, active HMM, finite replay outputs and unchanged inputs to HMM-derived EV; fresh backtest outcomes are never copied from the diagnostic run."}
    (OUT / "meta_replay_reuse_audit.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)
    run(["scripts/backfill_training_features.py","--input",str(input_path),"--output",str(OUT / "fresh_meta_validated_backfilled.csv"),"--default-artifact-version",version,"--hmm-only","--feature-cutoff-source","candidate_id_scan_time","--replay-scope","hmm_lineage_only","--require-fresh-output","--skip-if-fresh","--max-concurrency","4"],"meta_validated_backfill.log")
    run([str(OUT / "audit_meta_pool.py")],"meta_validated_audit.log")
    audit = json.loads((OUT / "meta_lineage_audit.json").read_text())
    missing = {k:v for k,v in audit["feature_nonfinite_counts"].items() if v}
    assert set(missing) <= {"ou_halflife"}, missing
    args = ["scripts/finalize_fresh_authoritative_meta_pool.py","--source",str(OUT / "fresh_meta_validated_backfilled.csv"),"--run-manifest",str(manifest_path),"--output-dir",str(OUT / "finalized_fresh_pool"),"--start-date","2026-02-18","--end-date","2026-09-11","--active-hmm-artifact-version",version]
    if missing:
        args.append("--allow-active-feature-imputation")
    run(args,"meta_finalization.log")
    args = ["retrain_meta_labeler.py","--input","data/new_expired_bots.xlsx","--backtest-results-dir",str(OUT / "finalized_fresh_pool"),"--max-rows-per-symbol","0","--estimator","vote_logit_hgb","--export-training-data",str(OUT / "prepared_training.csv")]
    if missing:
        args.append("--allow-imputation")
    (OUT / "meta_retrain_command.json").write_text(json.dumps(args,indent=2)+"\n")
    run(args,"meta_retrain.log")
    print("Canonical meta-labeler refit completed; runtime validation remains.",flush=True)


if __name__ == "__main__":
    main()
