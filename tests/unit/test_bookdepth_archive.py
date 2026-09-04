from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from neutralgrid.data.bookdepth_archive import (
    bookdepth_checksum_url,
    bookdepth_filename,
    bookdepth_url,
    build_snapshot_features,
    build_window_diagnostics,
    file_requests_for_targets,
    parse_bookdepth_zip,
    parse_checksum_text,
)
from neutralgrid.data.depth_shadow import DepthShadowTarget


def _write_bookdepth_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv = "\n".join(
        [
            "timestamp,percentage,depth,notional",
            "2026-06-25 23:59:50,-1,10,1000",
            "2026-06-25 23:59:50,1,8,800",
            "2026-06-26 00:00:10,-1,11,1100",
            "2026-06-26 00:00:10,1,7,700",
            "2026-06-26 00:01:10,-1,12,1200",
            "2026-06-26 00:01:10,1,6,600",
        ]
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-bookDepth-2026-06-26.csv", csv)


def test_bookdepth_url_builders() -> None:
    assert bookdepth_filename("btcusdt", date(2026, 6, 25)) == "BTCUSDT-bookDepth-2026-06-25.zip"
    assert (
        bookdepth_url("btcusdt", date(2026, 6, 25))
        == "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2026-06-25.zip"
    )
    encoded = bookdepth_url("\u9f99\u867eUSDT", date(2026, 6, 25))
    assert "%E9%BE%99%E8%99%BEUSDT" in encoded
    assert bookdepth_checksum_url("btcusdt", date(2026, 6, 25)).endswith(".zip.CHECKSUM")


def test_parse_checksum_text_extracts_hash() -> None:
    digest = "a" * 64
    assert parse_checksum_text(f"{digest}  BTCUSDT-bookDepth-2026-06-25.zip") == digest
    assert parse_checksum_text("not-a-checksum") is None


def test_parse_bookdepth_zip_normalizes_columns(tmp_path: Path) -> None:
    zip_path = tmp_path / "BTCUSDT-bookDepth-2026-06-26.zip"
    _write_bookdepth_zip(zip_path)

    df = parse_bookdepth_zip(zip_path)

    assert list(df.columns) == ["timestamp", "percentage", "depth", "notional"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["percentage"].tolist()[:2] == [-1, 1]


def test_file_requests_cover_lookback_and_forward_dates() -> None:
    target = DepthShadowTarget(
        symbol="BTCUSDT",
        candidate_id="BTCUSDT_20260626_000005",
        scan_time_utc="2026-06-26T00:00:05+00:00",
    )

    requests = file_requests_for_targets([target], lookback_hours=1.0, forward_hours=7.0)

    assert [(req.symbol, req.date.isoformat()) for req in requests] == [
        ("BTCUSDT", "2026-06-25"),
        ("BTCUSDT", "2026-06-26"),
    ]


def test_build_snapshot_features_uses_only_pre_scan_rows(tmp_path: Path) -> None:
    zip_path = tmp_path / "BTCUSDT-bookDepth-2026-06-26.zip"
    _write_bookdepth_zip(zip_path)
    frame = parse_bookdepth_zip(zip_path)
    target = DepthShadowTarget(
        symbol="BTCUSDT",
        candidate_id="BTCUSDT_20260626_000005",
        scan_time_utc="2026-06-26T00:00:05+00:00",
        position_notional_usdt=100.0,
    )

    result = build_snapshot_features(target, [frame])

    assert result.features["bookdepth_archive_available"] == 1
    assert result.features["bookdepth_snapshot_time_utc"] == "2026-06-25T23:59:50+00:00"
    assert result.features["bookdepth_snapshot_lag_seconds"] == 15.0
    assert result.features["bookdepth_min_1_notional"] == 800.0
    assert result.features["bookdepth_min_1_to_position"] == 8.0
    assert result.features["bookdepth_imbalance_1"] == (1000.0 - 800.0) / 1800.0


def test_build_window_diagnostics_keeps_forward_evidence_separate(tmp_path: Path) -> None:
    zip_path = tmp_path / "BTCUSDT-bookDepth-2026-06-26.zip"
    _write_bookdepth_zip(zip_path)
    frame = parse_bookdepth_zip(zip_path)
    target = DepthShadowTarget(
        symbol="BTCUSDT",
        candidate_id="BTCUSDT_20260626_000005",
        scan_time_utc="2026-06-26T00:00:05+00:00",
    )

    diagnostics = build_window_diagnostics(target, [frame], forward_hours=1.0)

    assert diagnostics["bookdepth_window_available"] == 1
    assert diagnostics["bookdepth_window_snapshot_count"] == 2
    assert diagnostics["bookdepth_window_min_1_notional_min"] == 600.0
