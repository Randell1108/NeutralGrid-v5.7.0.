"""Regression tests: every production consumer of profile_model.json must
route through resolve_active_profile_model_path() rather than assembling a
raw path. The resolver may use profile_model.json only as the explicit
bootstrap fallback when current.json is absent.

These are source-level assertions (string search) because:
  - Importing run_full_pipeline / scan_top100 pulls heavy dependency graphs.
  - app.py startup executes Binance/FastAPI bootstrap.
  - The invariant we need to lock is literal: the resolver is called and
    no direct `"profile_model.json"` path-assembly remains in a loader site.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _workspace_tmp() -> Path:
    p = Path.cwd() / ".pytest_tmp" / f"resolver_{uuid.uuid4().hex}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def test_scan_top100_imports_resolver():
    src = _read("scan_top100.py")
    assert "from neutralgrid.scanner.profile_model_walkforward import (" in src
    assert "resolve_active_profile_model_path" in src
    assert "resolve_active_profile_model_path()" in src
    assert "resolve_active_pattern_profile_path" in src


def test_scan_top100_model_path_default_is_none():
    """CLI default None means the resolver is used; explicit override still honored."""
    src = _read("scan_top100.py")
    assert 'default="data/profile/profile_model.json"' not in src
    assert 'default="data/profile/pattern_profile.json"' not in src
    # Argparse block uses default=None for --model-path.
    # Guard against regression: ensure the explicit hardcoded default does not return.


def test_run_full_pipeline_imports_resolver():
    src = _read("run_full_pipeline.py")
    assert "from neutralgrid.scanner.profile_model_walkforward import (" in src
    assert "resolve_active_profile_model_path" in src
    assert "resolve_active_profile_model_path()" in src


def test_run_full_pipeline_resolves_paired_pattern_profile():
    src = _read("run_full_pipeline.py")
    assert "resolve_active_pattern_profile_path" in src
    assert 'default="data/profile/pattern_profile.json"' not in src


def test_run_full_pipeline_model_path_default_is_none():
    src = _read("run_full_pipeline.py")
    assert 'default="data/profile/profile_model.json"' not in src


def test_api_app_uses_resolver():
    src = _read("src/neutralgrid/api/app.py")
    assert "resolve_active_profile_model_path" in src
    # Direct path assembly must not reappear:
    assert '_profile_dir / "profile_model.json"' not in src
    assert "resolve_active_pattern_profile_path" in src
    assert '_profile_dir / "pattern_profile.json"' not in src


def test_api_app_fail_closed_message_mentions_promotion():
    """When the resolver returns a sentinel, app startup must fail with a
    message pointing to bootstrap creation or promotion workflow."""
    src = _read("src/neutralgrid/api/app.py")
    assert "no promoted or bootstrap profile " in src
    assert "model is active" in src
    assert "profile_model.json" in src
    assert "retrain_scanner to create" in src
    assert "promote_profile_version" in src


def test_resolver_bootstraps_when_profile_model_json_exists():
    """End-to-end invariant: absent current.json may bootstrap from
    profile_model.json so the pipeline can run before first promotion."""
    from neutralgrid.scanner.profile_model_walkforward import (
        resolve_active_profile_model_path,
    )

    tmp = _workspace_tmp()
    try:
        (tmp / "profile_model.json").write_text("{}")
        resolved = resolve_active_profile_model_path(tmp)
        assert resolved.name == "profile_model.json"
        assert resolved.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
