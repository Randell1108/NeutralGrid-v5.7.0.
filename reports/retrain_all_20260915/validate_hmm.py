"""Validate the canonical HMM rotation and record its actual training boundary."""
from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from neutralgrid.models.artifacts import load_artifact

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main() -> None:
    old = json.loads((OUT / "artifact_manifest_before.json").read_text())["hmm"]["active_version"]
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())
    version = manifest["hmm"]["active_version"]
    assert version != old and re.fullmatch(r"rolling_180d_\d{8}_\d{6}", version)
    directory = ROOT / "artifacts/hmm" / version
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((directory / "eval.json").read_text())
    temperature = json.loads((directory / "temperature_scaler.json").read_text())
    assert evaluation["mean_pass_rate"] >= .5
    assert temperature["temperature"] == 1.0 and temperature["fitted"] is False
    assert temperature["status"] == "disabled_self_supervised"
    artifact = load_artifact(directory, expected_version=version)
    model, scaler = artifact["model"], artifact["scaler"]
    for value in [model.startprob_, model.transmat_, model.means_, model.covars_, scaler.center_, scaler.scale_]:
        assert np.isfinite(value).all()
    assert np.allclose(model.transmat_.sum(axis=1), 1)
    assert np.isclose(model.startprob_.sum(), 1)
    assert (np.diagonal(model.covars_, axis1=1, axis2=2) > 0).all()
    assert np.allclose(scaler.inverse_transform(model.means_), artifact["state_means_unscaled"])
    result = {"status": "PASS", "checked_at_utc": datetime.now(timezone.utc).isoformat(), "prior_hmm": old, "active_hmm": version, "rotated": True, "artifact_dir": str(directory), "eval_metrics": evaluation, "temperature": temperature, "metadata": metadata, "parameter_integrity": "PASS", "gate_limitation": "Soft-mode pass rate is unconditional, not predictive accuracy"}
    (OUT / "active_hmm_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"metadata", "eval_metrics"}}, indent=2))


if __name__ == "__main__":
    main()
