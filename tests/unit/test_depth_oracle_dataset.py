from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.aggregate_depth_oracle_dataset import main as aggregate_main


def test_aggregate_depth_oracle_dataset_blocks_without_oos_classes(tmp_path: Path) -> None:
    depth_input = tmp_path / "depth_shadow"
    feature_dir = depth_input / "features"
    feature_dir.mkdir(parents=True)
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    output_dir = tmp_path / "dataset"

    labels = pd.DataFrame(
        [
            {
                "symbol": "LINKUSDT",
                "candidate_id": "LINKUSDT_20260626_002406_aaa",
                "depth_oracle_label": 1,
                "label_status": "positive",
            },
            {
                "symbol": "TAOUSDT",
                "candidate_id": "TAOUSDT_20260626_002406_bbb",
                "depth_oracle_label": 0,
                "label_status": "negative",
            },
            {
                "symbol": "XRPUSDT",
                "candidate_id": "XRPUSDT_20260626_002406_ccc",
                "depth_oracle_label": 1,
                "label_status": "positive",
            },
        ]
    )
    labels.to_csv(oracle_dir / "depth_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "LINKUSDT_20260626_002406_aaa",
                "depth_scan_spread_pct": 0.01,
            },
            {
                "candidate_id": "TAOUSDT_20260626_002406_bbb",
                "depth_scan_spread_pct": 0.02,
            },
            {
                "candidate_id": "XRPUSDT_20260626_002406_ccc",
                "depth_scan_spread_pct": 0.03,
            },
        ]
    ).to_csv(feature_dir / "depth_exante_features.csv", index=False)
    (oracle_dir / "depth_oracle_manifest.json").write_text(
        json.dumps(
            {
                "depth_labels": str(oracle_dir / "depth_labels.csv"),
                "depth_input": str(depth_input),
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    exit_code = aggregate_main(
        [
            "--oracle-dir",
            str(oracle_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / "depth_oracle_dataset_manifest.json").read_text(encoding="utf-8"))
    training = pd.read_csv(output_dir / "depth_training_frame.csv")
    assert exit_code == 2
    assert manifest["status"] == "blocked_insufficient_oos_depth_labels"
    assert manifest["label_summary"]["labelable_rows"] == 3
    assert manifest["oos_split"]["ready"] is False
    assert manifest["oos_split"]["reason"] == "eval_split_missing_class"
    assert "depth_scan_spread_pct" in training.columns
