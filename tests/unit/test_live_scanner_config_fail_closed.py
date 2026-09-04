"""Regression: an explicitly passed --config-file that fails to load must
abort the scanner (fail-closed), not silently fall back to default
RecommenderConfig — in scheduled/loop mode a silent fallback would run every
subsequent tick on wrong thresholds while exiting 0."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live_decision_scanner import _load_recommender_config
from neutralgrid.live.decision.recommender import RecommenderConfig


def _args(config_file):
    return argparse.Namespace(config_file=config_file)


def test_no_config_file_uses_defaults():
    cfg = _load_recommender_config(_args(None))
    assert isinstance(cfg, RecommenderConfig)


def test_missing_explicit_config_file_aborts():
    with pytest.raises(SystemExit):
        _load_recommender_config(_args(Path("does_not_exist_config.yaml")))


def test_valid_config_file_loads(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text("meta_tilt_low_threshold: 0.41\n", encoding="utf-8")
    cfg = _load_recommender_config(_args(p))
    assert cfg.meta_tilt_low_threshold == pytest.approx(0.41)
