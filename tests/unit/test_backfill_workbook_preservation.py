from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "backfill_workbook_preservation", _ROOT / "scripts/backfill_training_features.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_FIXTURE = _ROOT / "tests/fixtures/backfill_preservation_five_sheets.xlsx"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_fixture(tmp_path: Path, name: str = "source.xlsx") -> Path:
    path = tmp_path / name
    shutil.copyfile(_FIXTURE, path)
    return path


def _isolated_backfiller(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    backfiller = _MODULE.TrainingDataBackfiller(**kwargs)

    async def noop() -> None:
        pass

    async def no_features(row: pd.Series) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(backfiller, "init_client", noop)
    monkeypatch.setattr(backfiller, "close_client", noop)
    monkeypatch.setattr(backfiller, "backfill_single_bot", no_features)
    return backfiller


def test_five_sheet_source_requires_explicit_flat_export_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_fixture(tmp_path)
    before = _digest(source)
    destination = tmp_path / "flat.xlsx"
    backfiller = _isolated_backfiller(monkeypatch)

    async def forbidden_network() -> None:
        pytest.fail("Workbook preservation must be checked before network initialization")

    monkeypatch.setattr(backfiller, "init_client", forbidden_network)
    with pytest.raises(ValueError, match="allow-flat-workbook-export"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert _digest(source) == before
    assert not destination.exists()


@pytest.mark.parametrize("suffix", [".xlsx", ".csv"])
def test_explicit_flat_export_preserves_all_source_bytes_and_stale_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    source = _copy_fixture(tmp_path)
    before = _digest(source)
    destination = tmp_path / ("flat" + suffix)
    backfiller = _isolated_backfiller(
        monkeypatch, allow_flat_workbook_export=True,
        default_artifact_version="new_hmm", skip_if_fresh=True,
    )
    asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    result = pd.read_excel(destination) if suffix == ".xlsx" else pd.read_csv(destination)
    assert result["strategy_id"].tolist() == [1001, 1002]
    assert result["label"].tolist() == [1, 0]
    assert pd.isna(result.loc[0, "range_prob"])
    assert pd.isna(result.loc[0, "hmm_artifact_version"])
    assert result.loc[1, "range_prob"] == pytest.approx(0.6)
    assert result.loc[1, "hmm_artifact_version"] == "new_hmm"
    assert _digest(source) == before
    with pd.ExcelFile(source) as book:
        assert len(book.sheet_names) == 5


def test_flat_opt_in_never_allows_overwriting_multisheet_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_fixture(tmp_path)
    before = _digest(source)
    backfiller = _isolated_backfiller(monkeypatch, allow_flat_workbook_export=True)
    with pytest.raises((ValueError, FileExistsError), match="workbook|fresh"):
        asyncio.run(backfiller.backfill_all(str(source), str(source)))
    assert _digest(source) == before


@pytest.mark.parametrize("suffix", [".xlsx", ".csv"])
def test_multisheet_export_rejects_existing_flat_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str,
) -> None:
    source = _copy_fixture(tmp_path)
    destination = tmp_path / ("existing" + suffix)
    # Even an unreadable existing target must not be overwritten.
    destination.write_bytes(b"existing destination")
    backfiller = _isolated_backfiller(monkeypatch, allow_flat_workbook_export=True)
    with pytest.raises(FileExistsError, match="fresh"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert destination.read_bytes() == b"existing destination"


def test_csv_input_cannot_replace_existing_multisheet_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = _copy_fixture(tmp_path, "canonical.xlsx")
    before = _digest(destination)
    source = tmp_path / "rows.csv"
    pd.DataFrame([{"symbol": "BTCUSDT"}]).to_csv(source, index=False)
    backfiller = _isolated_backfiller(monkeypatch)
    with pytest.raises(ValueError, match="multi-sheet"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert _digest(destination) == before


def test_destination_created_during_backfill_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _copy_fixture(tmp_path)
    destination = tmp_path / "new.csv"
    backfiller = _isolated_backfiller(monkeypatch, allow_flat_workbook_export=True)

    async def concurrent_destination() -> None:
        destination.write_bytes(b"published by another worker")

    monkeypatch.setattr(backfiller, "init_client", concurrent_destination)
    with pytest.raises(FileExistsError, match="fresh|created during"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert destination.read_bytes() == b"published by another worker"


def test_multisheet_destination_created_during_tabular_backfill_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "rows.csv"
    destination = tmp_path / "new.xlsx"
    pd.DataFrame([{"symbol": "BTCUSDT"}]).to_csv(source, index=False)
    backfiller = _isolated_backfiller(monkeypatch)

    async def concurrent_destination() -> None:
        shutil.copyfile(_FIXTURE, destination)

    monkeypatch.setattr(backfiller, "init_client", concurrent_destination)
    with pytest.raises(ValueError, match="multi-sheet"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert _digest(destination) == _digest(_FIXTURE)


def test_fresh_publication_refuses_last_moment_destination_without_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "fresh.csv"
    original_to_csv = pd.DataFrame.to_csv

    def racing_to_csv(self: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        destination.write_bytes(b"last-moment concurrent result")
        return original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", racing_to_csv)
    with pytest.raises(FileExistsError):
        _MODULE._write_dataframe_atomic(
            pd.DataFrame([{"a": 1}]), destination, require_fresh=True,
        )
    assert destination.read_bytes() == b"last-moment concurrent result"
    assert not list(tmp_path.glob(".*.tmp*"))


def test_atomic_export_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "existing.csv"
    destination.write_bytes(b"previous valid result")

    def failed_export(self: pd.DataFrame, *args: Any, **kwargs: Any) -> None:
        raise OSError("simulated export I/O failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_export)
    with pytest.raises(OSError, match="simulated"):
        _MODULE._write_dataframe_atomic(pd.DataFrame([{"a": 1}]), destination)
    assert destination.read_bytes() == b"previous valid result"
    assert not list(tmp_path.glob(".*.tmp*"))


def test_fresh_publication_does_not_fallback_when_hard_links_are_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "fresh.csv"

    def unsupported_link(*args: Any, **kwargs: Any) -> None:
        raise OSError("hard links unavailable on this filesystem")

    monkeypatch.setattr(_MODULE.os, "link", unsupported_link)
    with pytest.raises(OSError, match="hard links unavailable"):
        _MODULE._write_dataframe_atomic(
            pd.DataFrame([{"a": 1}]), destination, require_fresh=True,
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp*"))


def test_unreadable_destination_is_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.csv"
    destination = tmp_path / "canonical.xlsx"
    pd.DataFrame([{"symbol": "BTCUSDT"}]).to_csv(source, index=False)
    destination.write_bytes(b"not a valid workbook")
    backfiller = _isolated_backfiller(monkeypatch)

    async def forbidden_network() -> None:
        pytest.fail("Unreadable workbook must fail before network initialization")

    monkeypatch.setattr(backfiller, "init_client", forbidden_network)
    with pytest.raises(ValueError, match="format"):
        asyncio.run(backfiller.backfill_all(str(source), str(destination)))
    assert destination.read_bytes() == b"not a valid workbook"


def test_cli_exposes_flat_export_opt_in() -> None:
    assert _MODULE._parse_args(["--allow-flat-workbook-export"]).allow_flat_workbook_export
