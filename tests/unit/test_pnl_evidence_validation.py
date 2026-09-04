from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from scripts import validate_pnl_evidence_runtime as validation


def test_validation_evidence_bootstraps_then_reuses_exact_fingerprint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def successful_runner(
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "passed", "")

    first = validation.ensure_validation_evidence(
        workspace_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        relevant_files=[source],
        commands=[["python", "-m", "pytest"], ["pyright"]],
        runner=successful_runner,
        git_head="head-1",
        git_status=" M source.py",
    )
    second = validation.ensure_validation_evidence(
        workspace_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        relevant_files=[source],
        commands=[["python", "-m", "pytest"], ["pyright"]],
        runner=successful_runner,
        git_head="head-1",
        git_status=" M source.py",
    )

    assert first["status"] == "pass"
    assert first["reused"] is False
    assert second["status"] == "pass"
    assert second["reused"] is True
    assert len(calls) == 2
    evidence_path = Path(str(first["evidence_path"]))
    persisted = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert persisted["source_fingerprint"] == first["source_fingerprint"]
    assert [item["exit_code"] for item in persisted["commands"]] == [0, 0]


def test_validation_evidence_does_not_reuse_after_source_change(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    calls = 0

    def successful_runner(
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, "passed", "")

    first = validation.ensure_validation_evidence(
        workspace_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        relevant_files=[source],
        commands=[["check"]],
        runner=successful_runner,
        git_head="head-1",
        git_status="",
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second = validation.ensure_validation_evidence(
        workspace_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        relevant_files=[source],
        commands=[["check"]],
        runner=successful_runner,
        git_head="head-1",
        git_status="",
    )

    assert first["source_fingerprint"] != second["source_fingerprint"]
    assert calls == 2

