"""Refit the canonical profile recipe without reusing disclosed validation rows."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, build_profile_from_enhanced_xlsx
from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx, save_profile_model
from neutralgrid.scanner.profile_model_walkforward import walkforward_evaluate, promote_profile_version

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "profile_candidate"


def main() -> None:
    OUT.mkdir(exist_ok=False)
    source = ROOT / "data/new_expired_bots.xlsx"
    previous = json.loads((ROOT / "reports/profile_training_20260908/workbook_baseline/evaluation.json").read_text())
    prior_ends = [s for r in previous["trials"] for s in r["walkforward"]["fold_test_end_utc"]]
    cutoff = max(prior_ends)
    model = train_profile_model_from_enhanced_xlsx(source, top_quantile=.68, shrinkage=.3)
    summary = dict(model.selection_summary or {}, artifact_role="shadow", hmm_features_used=False, prior_disclosure_end_utc=cutoff, funding_availability="not_verified; no realized-funding backfill performed")
    model = dataclasses.replace(model, selection_summary=summary)
    pattern = build_profile_from_enhanced_xlsx(source, top_quantile=.68)
    pattern = dataclasses.replace(pattern, selection_summary=summary)
    save_profile_model(model, OUT / "profile_model.json")
    pattern.save_json(OUT / "pattern_profile.json")
    evaluation = walkforward_evaluate(source, top_quantile=.68, shrinkage=.3, n_folds=5, purge_hours=7, holdout_start_after_utc=cutoff)
    # This is an isolated rejection audit, not a production incumbent comparison.
    decision = promote_profile_version(model, requested_features=list(DEFAULT_FEATURES), wf_result=evaluation, candidate_pattern_profile=pattern, profile_dir=OUT / "promotion_audit")
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "selection_summary": summary, "walkforward": dataclasses.asdict(evaluation), "isolated_promotion_decision": dataclasses.asdict(decision), "production_activated": False, "blockers": ["Entry-time funding availability not verified", "Production pattern-only bootstrap has no paired profile model"], "profile_gate": "No legacy profile gate emitted for the current four-feature PatternProfile contract"}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
