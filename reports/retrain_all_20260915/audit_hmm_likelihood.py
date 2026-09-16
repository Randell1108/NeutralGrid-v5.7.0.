"""Frozen-model conditional likelihood check on closed post-training bars."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.data.features import compute_hmm_features, FEATURE_NAMES
from neutralgrid.scanner.feature_extractor import klines_to_df

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


async def main() -> None:
    active = json.loads((OUT / "active_hmm_validation.json").read_text(encoding="utf-8"))
    versions = {"old": active["prior_hmm"], "new": active["active_hmm"]}
    models = {}
    metadata = {}
    hashes = {}
    for role, version in versions.items():
        directory = ROOT / "artifacts/hmm" / version
        metadata[role] = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert metadata[role]["features"] == list(FEATURE_NAMES)
        assert metadata[role]["feature_schema"] == get_config().hmm.feature_schema
        model = joblib.load(directory / "model.joblib")
        scaler = joblib.load(directory / "scaler.joblib")
        for value in [model.startprob_, model.transmat_, model.means_, model.covars_, scaler.scale_, scaler.center_]:
            assert np.isfinite(value).all()
        assert np.allclose(model.transmat_.sum(axis=1), 1)
        assert np.isclose(model.startprob_.sum(), 1)
        assert (model.transmat_ >= 0).all() and (model.startprob_ >= 0).all()
        assert (np.diagonal(model.covars_, axis1=1, axis2=2) > 0).all()
        assert (scaler.scale_ > 0).all()
        models[role] = (model, scaler)
        hashes[role] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    boundary = pd.Timestamp(metadata["new"]["training_end_utc"])
    prior_path = OUT / "hmm_post_training_audit.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else None
    asof = pd.Timestamp(prior["frozen_asof_utc"]) if prior else pd.Timestamp.now(tz="UTC").floor("15min")
    destination = OUT / "hmm_post_training_data"
    destination.mkdir(exist_ok=prior is not None)
    rows = []
    failures = []
    client = BinanceClient()
    try:
        for symbol in metadata["new"]["symbols_used"]:
            try:
                data_path = destination / f"{symbol}.json"
                if data_path.exists() and prior is not None:
                    recorded = next(r for r in prior["per_symbol"] if r["symbol"] == symbol)
                    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == recorded["source_sha256"]
                    raw = json.loads(data_path.read_text(encoding="utf-8"))
                else:
                    raw = await client.get_klines(symbol, "15m", limit=800, start_time=int((asof-pd.Timedelta(minutes=15*800)).timestamp()*1000), end_time=int(asof.timestamp()*1000)-1)
                assert all(int(k[6]) < int(asof.timestamp()*1000) for k in raw)
                frame = klines_to_df(raw).sort_values("open_time").reset_index(drop=True)
                assert frame.open_time.is_unique
                assert frame.open_time.diff().dropna().eq(pd.Timedelta(minutes=15)).all()
                data_path.write_text(json.dumps(raw, separators=(",", ":")), encoding="utf-8")
                matrix, valid = compute_hmm_features(frame)
                features = matrix[valid]
                times = frame.loc[valid, "open_time"]
                count_prefix = int(times.le(boundary).sum())
                count_oos = int(times.gt(boundary).sum())
                assert count_prefix > 0 and count_oos > 0
                assert times.iloc[count_prefix] > boundary
                row = {"symbol": symbol, "prefix_rows": count_prefix, "oos_rows": count_oos, "oos_start": str(times.iloc[count_prefix]), "oos_end": str(times.iloc[-1]), "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()}
                for role, (model, scaler) in models.items():
                    transformed = scaler.transform(features)
                    conditional = model.score(transformed) - model.score(transformed[:count_prefix])
                    raw_density = conditional - count_oos * float(np.log(scaler.scale_).sum())
                    row[role + "_log_density_per_bar"] = float(raw_density/count_oos)
                row["new_minus_old_per_bar"] = row["new_log_density_per_bar"] - row["old_log_density_per_bar"]
                assert np.isfinite(row["new_minus_old_per_bar"])
                rows.append(row)
                print(json.dumps(row), flush=True)
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})
                print(json.dumps(failures[-1]), flush=True)
            await asyncio.sleep(1)
    finally:
        await client.close()
    result = {"status": "PASS" if len(rows) == 50 and not failures else "INCOMPLETE", "versions": versions, "frozen_asof_utc": str(asof), "new_training_last_bar_open_utc": str(boundary), "symbols_evaluated": len(rows), "oos_rows": sum(r["oos_rows"] for r in rows), "method": "Conditional log likelihood = score(prefix+OOS)-score(prefix), adjusted by minus n_OOS*sum(log(scaler.scale_)) for raw feature density comparability. Same features and observed bars for both frozen models.", "limitations": ["Short single-day sample; correlated symbols; no statistical significance or profit claim", "Supplemental diagnostic, not a changed promotion gate", "Configured soft-mode walk-forward pass rate is always one by implementation, so it is not predictive accuracy"], "parameter_integrity": "PASS", "artifact_sha256": hashes, "failures": failures, "per_symbol": rows}
    if rows:
        deltas = np.array([r["new_minus_old_per_bar"] for r in rows])
        result.update(mean_log_density_gain_per_bar=float(deltas.mean()), median_log_density_gain_per_bar=float(np.median(deltas)), fraction_symbols_improved=float((deltas > 0).mean()))
    (OUT / "hmm_post_training_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"artifact_sha256", "per_symbol"}}, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
