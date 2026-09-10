"""Evaluate existing workbook profile recipes without backfilling or promotion."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES, build_profile_from_enhanced_xlsx
from neutralgrid.scanner.profile_model import train_profile_model_from_enhanced_xlsx, save_profile_model
from neutralgrid.scanner.profile_model_walkforward import walkforward_evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "data/new_expired_bots.xlsx"
    target = OUT / "workbook_baseline"
    target.mkdir(exist_ok=True)
    records = []
    for shrinkage in (0.0, 0.1, 0.3, 0.6, 0.9):
        model = train_profile_model_from_enhanced_xlsx(source, shrinkage=shrinkage, top_quantile=.68)
        model = dataclasses.replace(model, selection_summary=dict(model.selection_summary or {}, artifact_role="shadow_diagnostic", blocker="historical_funding_availability_unverified; canonical_backfill_uses_future_realized_proxy"))
        save_profile_model(model, target / f"profile_model_shrinkage_{shrinkage:.1f}.json")
        wf = walkforward_evaluate(source, n_folds=5, purge_hours=7, shrinkage=shrinkage, top_quantile=.68)
        records.append({"shrinkage": shrinkage, "walkforward": dataclasses.asdict(wf)})
        print(json.dumps({"shrinkage": shrinkage, "mean_auc": wf.mean_auc, "pass_rate": wf.mean_pass_rate, "fold_auc": wf.fold_auc}), flush=True)
    pattern = build_profile_from_enhanced_xlsx(source, top_quantile=.68)
    pattern.save_json(target / "pattern_profile.json")
    ablation = [f for f in DEFAULT_FEATURES if f != "funding_carry_expected_next_7h"]
    wf = walkforward_evaluate(source, n_folds=5, purge_hours=7, features=ablation, shrinkage=.3, top_quantile=.68)
    (target / "evaluation.json").write_text(json.dumps({"source": str(source), "status": "shadow_only", "trials": records, "funding_excluded_diagnostic": dataclasses.asdict(wf), "blockers": ["Historical funding proxy is not proven available at entry", "Three-feature ablation is below required 90% feature retention", "These workbook outcomes have prior evaluation exposure"]}, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
