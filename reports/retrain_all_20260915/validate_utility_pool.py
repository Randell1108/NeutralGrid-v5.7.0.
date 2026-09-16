"""Compare the canonical and derived utility admission paths exactly."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from pandas.testing import assert_frame_equal
from neutralgrid.calibration.utility_calibrator import _load_calibration_pool, _split_pool, LABEL_COL

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    version = json.loads((ROOT / "artifact_manifest.json").read_text())["hmm"]["active_version"]
    canonical = _load_calibration_pool(ROOT / "data/new_expired_bots.xlsx", expected_hmm_artifact_version=version)
    derived = _load_calibration_pool(OUT / "utility_complete_features.xlsx", expected_hmm_artifact_version=version)
    assert_frame_equal(canonical, derived, check_dtype=False, check_exact=True)
    fit, holdout = _split_pool(canonical)
    canonical.to_csv(OUT / "utility_admitted_pool.csv", index=False)
    general = pd.read_excel(ROOT / "data/new_expired_bots.xlsx", sheet_name="General")
    excluded = general.loc[~general.strategy_id.isin(canonical.strategy_id)].copy()
    excluded.to_csv(OUT / "utility_excluded_rows.csv", index=False)
    result = {"status": "PASS", "active_hmm": version, "canonical_and_flat_pool_equal": True, "workbook_rows": len(general), "admitted_rows": len(canonical), "excluded_rows": len(excluded), "label_counts": canonical[LABEL_COL].value_counts().to_dict(), "fit_rows": len(fit), "holdout_rows": len(holdout), "fit_label_counts": fit[LABEL_COL].value_counts().to_dict(), "holdout_label_counts": holdout[LABEL_COL].value_counts().to_dict()}
    (OUT / "utility_pool_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
