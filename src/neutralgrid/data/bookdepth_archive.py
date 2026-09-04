"""Binance Vision futures bookDepth archive helpers.

Binance ``bookDepth`` daily files are percentage-bucket depth snapshots, not
raw price-level order-book ladders.  They are still useful as historical,
candidate-keyed liquidity evidence when joined strictly at or before scan time.
"""

from __future__ import annotations

import hashlib
import io
import urllib.parse
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.data.binance_vision.urls import BASE_URL
from neutralgrid.data.depth_shadow import DepthShadowTarget, load_depth_shadow_targets


BOOKDEPTH_SCHEMA_VERSION = "bookdepth_archive_v1"
BOOKDEPTH_COLUMNS = ["timestamp", "percentage", "depth", "notional"]
DEFAULT_BOOKDEPTH_FORWARD_HOURS = 7.0
DEFAULT_BOOKDEPTH_LOOKBACK_HOURS = 1.0


@dataclass(frozen=True)
class BookDepthFileRequest:
    """One Binance Vision bookDepth file needed by candidate windows."""

    symbol: str
    date: date

    @property
    def filename(self) -> str:
        return bookdepth_filename(self.symbol, self.date)

    @property
    def url(self) -> str:
        return bookdepth_url(self.symbol, self.date)

    @property
    def checksum_url(self) -> str:
        return bookdepth_checksum_url(self.symbol, self.date)


@dataclass(frozen=True)
class BookDepthSnapshotResult:
    """Candidate-level pre-scan bookDepth feature result."""

    features: dict[str, Any]
    diagnostics: dict[str, Any]


def bookdepth_filename(symbol: str, dt: date) -> str:
    """Return the Binance Vision daily bookDepth ZIP filename."""
    sym = symbol.upper()
    return f"{sym}-bookDepth-{dt.year:04d}-{dt.month:02d}-{dt.day:02d}.zip"


def bookdepth_relative_path(symbol: str, dt: date) -> str:
    """Return the Binance Vision relative path for a bookDepth ZIP."""
    sym = symbol.upper()
    return f"data/futures/um/daily/bookDepth/{sym}/{bookdepth_filename(sym, dt)}"


def bookdepth_url(symbol: str, dt: date) -> str:
    """Return the Binance Vision URL for a bookDepth ZIP."""
    return BASE_URL + urllib.parse.quote(bookdepth_relative_path(symbol, dt), safe="/")


def bookdepth_checksum_url(symbol: str, dt: date) -> str:
    """Return the Binance Vision URL for a bookDepth ZIP checksum."""
    return bookdepth_url(symbol, dt) + ".CHECKSUM"


def local_bookdepth_zip(root: Path, symbol: str, dt: date) -> Path:
    """Return the local archive path for a bookDepth ZIP."""
    sym = symbol.upper()
    return Path(root) / sym / bookdepth_filename(sym, dt)


def sha256_file(path: Path) -> str:
    """Compute a file SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_text(text: str) -> str | None:
    """Extract the SHA-256 digest from a Binance ``.CHECKSUM`` file."""
    first = text.strip().split()[0] if text.strip() else ""
    if len(first) == 64 and all(char in "0123456789abcdefABCDEF" for char in first):
        return first.lower()
    return None


def verify_checksum(zip_path: Path, checksum_path: Path) -> tuple[bool, str | None, str | None]:
    """Verify a local ZIP against its Binance checksum file."""
    if not Path(zip_path).exists() or not Path(checksum_path).exists():
        return False, None, None
    expected = parse_checksum_text(Path(checksum_path).read_text(encoding="utf-8", errors="replace"))
    actual = sha256_file(Path(zip_path))
    return expected is not None and actual.lower() == expected.lower(), expected, actual


def parse_bookdepth_zip(zip_path: Path) -> pd.DataFrame:
    """Parse a Binance Vision bookDepth ZIP into a normalized DataFrame."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected 1 CSV in {zip_path.name}, found {len(names)}")
        raw = archive.read(names[0])

    df = pd.read_csv(io.BytesIO(raw))
    if list(df.columns) != BOOKDEPTH_COLUMNS:
        if len(df.columns) != len(BOOKDEPTH_COLUMNS):
            raise ValueError(f"{zip_path.name}: expected {len(BOOKDEPTH_COLUMNS)} columns, got {len(df.columns)}")
        df.columns = BOOKDEPTH_COLUMNS

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ("percentage", "depth", "notional"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "percentage"]).sort_values(["timestamp", "percentage"])
    return out.reset_index(drop=True)


def load_bookdepth_targets(
    path: Path,
    *,
    max_candidates: int | None = None,
    fallback_position_usdt: float | None = None,
) -> list[DepthShadowTarget]:
    """Load candidate rows using the repo-standard candidate target contract."""
    return load_depth_shadow_targets(
        Path(path),
        max_candidates=max_candidates,
        fallback_position_usdt=fallback_position_usdt,
    )


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _target_scan_timestamp(target: DepthShadowTarget) -> pd.Timestamp | None:
    if target.scan_time_utc is None:
        return None
    parsed = pd.to_datetime(target.scan_time_utc, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return cast(pd.Timestamp, parsed)


def file_requests_for_targets(
    targets: Sequence[DepthShadowTarget],
    *,
    lookback_hours: float = DEFAULT_BOOKDEPTH_LOOKBACK_HOURS,
    forward_hours: float = DEFAULT_BOOKDEPTH_FORWARD_HOURS,
) -> list[BookDepthFileRequest]:
    """Return unique symbol-date archive requests covering candidate windows."""
    requests: dict[tuple[str, date], BookDepthFileRequest] = {}
    for target in targets:
        scan_ts = _target_scan_timestamp(target)
        if scan_ts is None:
            continue
        start_ts = scan_ts - pd.Timedelta(hours=float(lookback_hours))
        end_ts = scan_ts + pd.Timedelta(hours=float(forward_hours))
        start_date = cast(date, start_ts.date())
        end_date = cast(date, end_ts.date())
        for dt in _date_range(start_date, end_date):
            key = (target.symbol.upper(), dt)
            requests[key] = BookDepthFileRequest(symbol=key[0], date=key[1])
    return [requests[key] for key in sorted(requests)]


def _bucket_token(value: float) -> str:
    text = f"{abs(value):g}".replace(".", "p")
    return text


def _numeric_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _empty_feature_row(target: DepthShadowTarget, reason: str, zip_count: int) -> BookDepthSnapshotResult:
    base = _base_candidate_fields(target)
    base.update(
        {
            "bookdepth_archive_available": 0,
            "bookdepth_candidate_zip_count": zip_count,
            "bookdepth_snapshot_time_utc": None,
            "bookdepth_snapshot_lag_seconds": None,
            "bookdepth_bucket_count": 0,
            "bookdepth_available_percentages": "",
            "bookdepth_missing_reason": reason,
        }
    )
    return BookDepthSnapshotResult(features=base, diagnostics=dict(base))


def _base_candidate_fields(target: DepthShadowTarget) -> dict[str, Any]:
    return {
        "candidate_id": target.candidate_id,
        "symbol": target.symbol.upper(),
        "scan_time_utc": target.scan_time_utc,
        "position_notional_usdt": target.position_notional_usdt,
        "source_row_index": target.source_row_index,
        "source_path": target.source_path,
    }


def build_snapshot_features(
    target: DepthShadowTarget,
    frames: Sequence[pd.DataFrame],
    *,
    max_snapshot_lag_seconds: float | None = None,
) -> BookDepthSnapshotResult:
    """Build leakage-safe pre-scan bookDepth features for one target."""
    scan_ts = _target_scan_timestamp(target)
    if scan_ts is None:
        return _empty_feature_row(target, "missing_scan_time", len(frames))
    if not frames:
        return _empty_feature_row(target, "missing_archive_files", 0)

    combined = pd.concat(list(frames), ignore_index=True)
    if combined.empty:
        return _empty_feature_row(target, "empty_archive_files", len(frames))

    ts_col = cast(pd.Series, combined["timestamp"])
    pre = combined.loc[ts_col <= scan_ts].copy()
    if pre.empty:
        return _empty_feature_row(target, "no_snapshot_at_or_before_scan", len(frames))

    snapshot_time = cast(pd.Timestamp, pre["timestamp"].max())
    lag_seconds = float((scan_ts - snapshot_time).total_seconds())
    if max_snapshot_lag_seconds is not None and lag_seconds > max_snapshot_lag_seconds:
        return _empty_feature_row(target, "snapshot_lag_exceeds_limit", len(frames))

    snap = pre.loc[cast(pd.Series, pre["timestamp"]) == snapshot_time].copy()
    snap = snap.dropna(subset=["percentage"])
    features = _base_candidate_fields(target)
    features.update(
        {
            "bookdepth_archive_available": 1,
            "bookdepth_candidate_zip_count": len(frames),
            "bookdepth_snapshot_time_utc": snapshot_time.isoformat(),
            "bookdepth_snapshot_lag_seconds": lag_seconds,
            "bookdepth_bucket_count": int(len(snap)),
            "bookdepth_available_percentages": ";".join(f"{float(v):g}" for v in sorted(snap["percentage"].tolist())),
            "bookdepth_missing_reason": None,
        }
    )

    by_pct: dict[float, Mapping[str, Any]] = {}
    for _, row in snap.iterrows():
        pct = _numeric_or_none(row.get("percentage"))
        if pct is None:
            continue
        by_pct[pct] = cast(Mapping[str, Any], row)
        side = "neg" if pct < 0 else "pos"
        token = _bucket_token(pct)
        features[f"bookdepth_{side}_{token}_notional"] = _numeric_or_none(row.get("notional"))
        features[f"bookdepth_{side}_{token}_depth"] = _numeric_or_none(row.get("depth"))

    for abs_pct in sorted({abs(pct) for pct in by_pct if pct != 0}):
        neg = _numeric_or_none(by_pct.get(-abs_pct, {}).get("notional"))
        pos = _numeric_or_none(by_pct.get(abs_pct, {}).get("notional"))
        token = _bucket_token(abs_pct)
        total: float | None = None
        min_notional: float | None = None
        imbalance: float | None = None
        if neg is not None and pos is not None:
            total = neg + pos
            min_notional = min(neg, pos)
            imbalance = (neg - pos) / total if total > 0 else None
        features[f"bookdepth_total_{token}_notional"] = total
        features[f"bookdepth_min_{token}_notional"] = min_notional
        features[f"bookdepth_imbalance_{token}"] = imbalance
        position = target.position_notional_usdt
        if min_notional is not None and position is not None and position > 0:
            features[f"bookdepth_min_{token}_to_position"] = min_notional / position

    neg_values: list[float] = []
    pos_values: list[float] = []
    for pct, row in by_pct.items():
        value = _numeric_or_none(row.get("notional"))
        if value is None:
            continue
        if pct < 0:
            neg_values.append(value)
        elif pct > 0:
            pos_values.append(value)
    neg_sum = sum(neg_values)
    pos_sum = sum(pos_values)
    if neg_sum > 0 or pos_sum > 0:
        total_sum = neg_sum + pos_sum
        features["bookdepth_neg_all_notional"] = neg_sum
        features["bookdepth_pos_all_notional"] = pos_sum
        features["bookdepth_total_all_notional"] = total_sum
        features["bookdepth_imbalance_all"] = (neg_sum - pos_sum) / total_sum if total_sum > 0 else None

    return BookDepthSnapshotResult(features=features, diagnostics=dict(features))


def build_window_diagnostics(
    target: DepthShadowTarget,
    frames: Sequence[pd.DataFrame],
    *,
    forward_hours: float = DEFAULT_BOOKDEPTH_FORWARD_HOURS,
) -> dict[str, Any]:
    """Summarize forward-window archive coverage for diagnostics only."""
    base = _base_candidate_fields(target)
    scan_ts = _target_scan_timestamp(target)
    if scan_ts is None:
        base.update({"bookdepth_window_available": 0, "bookdepth_window_missing_reason": "missing_scan_time"})
        return base
    if not frames:
        base.update({"bookdepth_window_available": 0, "bookdepth_window_missing_reason": "missing_archive_files"})
        return base

    combined = pd.concat(list(frames), ignore_index=True)
    if combined.empty:
        base.update({"bookdepth_window_available": 0, "bookdepth_window_missing_reason": "empty_archive_files"})
        return base

    end_ts = scan_ts + pd.Timedelta(hours=float(forward_hours))
    ts_col = cast(pd.Series, combined["timestamp"])
    window = combined.loc[(ts_col >= scan_ts) & (ts_col <= end_ts)].copy()
    if window.empty:
        base.update({"bookdepth_window_available": 0, "bookdepth_window_missing_reason": "no_forward_window_rows"})
        return base

    timestamps = pd.Series(window["timestamp"]).dropna().drop_duplicates().sort_values()
    base.update(
        {
            "bookdepth_window_available": 1,
            "bookdepth_window_missing_reason": None,
            "bookdepth_window_start_utc": timestamps.iloc[0].isoformat() if not timestamps.empty else None,
            "bookdepth_window_end_utc": timestamps.iloc[-1].isoformat() if not timestamps.empty else None,
            "bookdepth_window_snapshot_count": int(len(timestamps)),
            "bookdepth_window_hours": (
                float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600.0)
                if len(timestamps) > 1
                else 0.0
            ),
            "bookdepth_window_expected_hours": float(forward_hours),
        }
    )

    pivot = window.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    for abs_pct in sorted({abs(float(col)) for col in pivot.columns if float(col) != 0.0}):
        neg_col = -abs_pct
        pos_col = abs_pct
        if neg_col not in pivot.columns or pos_col not in pivot.columns:
            continue
        min_series = pd.concat([pivot[neg_col], pivot[pos_col]], axis=1).min(axis=1).dropna()
        if min_series.empty:
            continue
        token = _bucket_token(abs_pct)
        base[f"bookdepth_window_min_{token}_notional_min"] = float(min_series.min())
        base[f"bookdepth_window_min_{token}_notional_median"] = float(min_series.median())
        base[f"bookdepth_window_min_{token}_notional_p10"] = float(min_series.quantile(0.10))
    return base


def load_archive_frames_for_target(
    target: DepthShadowTarget,
    archive_root: Path,
    *,
    lookback_hours: float = DEFAULT_BOOKDEPTH_LOOKBACK_HOURS,
    forward_hours: float = DEFAULT_BOOKDEPTH_FORWARD_HOURS,
    cache: dict[Path, pd.DataFrame] | None = None,
) -> tuple[list[pd.DataFrame], list[Path], list[str]]:
    """Load local archive frames needed by one candidate target."""
    requests = file_requests_for_targets(
        [target],
        lookback_hours=lookback_hours,
        forward_hours=forward_hours,
    )
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    errors: list[str] = []
    frame_cache = cache if cache is not None else {}
    for request in requests:
        path = local_bookdepth_zip(Path(archive_root), request.symbol, request.date)
        if not path.exists():
            continue
        try:
            frame = frame_cache.get(path)
            if frame is None:
                frame = parse_bookdepth_zip(path)
                frame_cache[path] = frame
            frames.append(frame)
            paths.append(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return frames, paths, errors
