"""Candidate-keyed order-book depth shadow capture helpers.

The depth-aware deployability work needs time-series L2 snapshots keyed to the
candidate that would have been evaluated.  This module deliberately captures
data only at collection time; it does not infer historical L2 or change any
production gate/model behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from neutralgrid.core.candidate_id import extract_parts


DEPTH_SHADOW_SCHEMA_VERSION = "depth_shadow_v1"
_SCAN_TIME_COLUMNS = (
    "scan_time_utc",
    "scan_timestamp_utc",
    "scan_timestamp",
    "snapshot_matched_at_utc",
    "start_time_utc",
    "timestamp_utc",
    "timestamp",
    "created_at_utc",
)
_POSITION_NOTIONAL_USDT_COLUMNS = (
    "position_size_usdt",
    "position_notional_usdt",
    "notional_usdt",
    "order_notional_usdt",
)
_MARGIN_USDT_COLUMNS = (
    "deploy_margin_usdt",
    "allocated_capital_usdt",
    "capital_allocated_usdt",
    "investment_usdt",
    "invested_margin_usdt",
)
_CAPITAL_FRACTION_COLUMNS = (
    "capital_fraction",
    "ps_fraction",
)
_CAPITAL_BASE_USDT_COLUMNS = (
    "capital_base_usdt",
    "account_equity_usdt",
)
_LEVERAGE_COLUMNS = (
    "leverage",
    "grid_leverage",
)


@dataclass(frozen=True)
class DepthShadowTarget:
    """Candidate row that should receive depth-shadow snapshots."""

    symbol: str
    candidate_id: str
    scan_time_utc: str | None = None
    position_notional_usdt: float | None = None
    source_row_index: int | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class DepthShadowRejectedTarget:
    """Depth-shadow target that cannot be used for ex-ante capture."""

    symbol: str
    candidate_id: str
    scan_time_utc: str | None
    reason: str
    scan_age_seconds: float | None = None
    source_row_index: int | None = None
    source_path: str | None = None


def utc_now_iso() -> str:
    """Return an ISO UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _non_empty(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _row_index_or_none(value: Any) -> int | None:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _first_column_value(row: pd.Series, columns: Sequence[str]) -> Any:
    for col in columns:
        if col in row.index:
            value = row[col]
            if _non_empty(value) is not None:
                return value
    return None


def _first_numeric_value(row: pd.Series, columns: Sequence[str]) -> float | None:
    value = _first_column_value(row, columns)
    return _float_or_none(value)


def derive_position_notional_usdt(row: pd.Series, *, fallback_position_usdt: float | None = None) -> float | None:
    """Derive order-book exposure from explicit candidate sizing columns.

    Direct notional columns win. Margin columns need leverage before they can be
    treated as notional; an explicit zero margin is authoritative and must not
    fall through to a fallback position.
    """
    notional = _first_numeric_value(row, _POSITION_NOTIONAL_USDT_COLUMNS)
    if notional is not None:
        return notional

    margin = _first_numeric_value(row, _MARGIN_USDT_COLUMNS)
    leverage = _first_numeric_value(row, _LEVERAGE_COLUMNS)
    if margin is not None:
        if margin <= 0:
            return 0.0
        if leverage is not None and leverage > 0:
            return float(margin * leverage)
        return None

    fraction = _first_numeric_value(row, _CAPITAL_FRACTION_COLUMNS)
    capital_base = _first_numeric_value(row, _CAPITAL_BASE_USDT_COLUMNS)
    if fraction is not None and capital_base is not None:
        if fraction <= 0 or capital_base <= 0:
            return 0.0
        if leverage is not None and leverage > 0:
            return float(fraction * capital_base * leverage)
        return None

    return fallback_position_usdt


def _read_candidate_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported candidate input format: {path}")


def parse_candidate_scan_time_utc(candidate_id: str) -> str | None:
    """Extract an ISO UTC scan timestamp from a repo-standard candidate_id."""
    scan_ts = extract_parts(candidate_id).get("scan_ts", "")
    if not scan_ts:
        return None
    try:
        parsed = datetime.strptime(scan_ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parsed.isoformat()


def _target_scan_datetime(target: DepthShadowTarget) -> datetime | None:
    if target.scan_time_utc is None:
        return None
    parsed = pd.to_datetime(target.scan_time_utc, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return cast(datetime, parsed.to_pydatetime())


def filter_fresh_depth_shadow_targets(
    targets: Sequence[DepthShadowTarget],
    *,
    now: datetime,
    max_scan_age_seconds: float,
) -> tuple[list[DepthShadowTarget], list[DepthShadowRejectedTarget]]:
    """Keep only targets whose scan time is close enough to collection time."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    fresh: list[DepthShadowTarget] = []
    rejected: list[DepthShadowRejectedTarget] = []
    for target in targets:
        scan_dt = _target_scan_datetime(target)
        if scan_dt is None:
            rejected.append(
                DepthShadowRejectedTarget(
                    symbol=target.symbol,
                    candidate_id=target.candidate_id,
                    scan_time_utc=target.scan_time_utc,
                    reason="missing_scan_time",
                    source_row_index=target.source_row_index,
                    source_path=target.source_path,
                )
            )
            continue

        age_seconds = (now - scan_dt).total_seconds()
        if age_seconds < 0:
            rejected.append(
                DepthShadowRejectedTarget(
                    symbol=target.symbol,
                    candidate_id=target.candidate_id,
                    scan_time_utc=target.scan_time_utc,
                    reason="scan_time_in_future",
                    scan_age_seconds=float(age_seconds),
                    source_row_index=target.source_row_index,
                    source_path=target.source_path,
                )
            )
            continue
        if age_seconds > max_scan_age_seconds:
            rejected.append(
                DepthShadowRejectedTarget(
                    symbol=target.symbol,
                    candidate_id=target.candidate_id,
                    scan_time_utc=target.scan_time_utc,
                    reason="stale_scan_time",
                    scan_age_seconds=float(age_seconds),
                    source_row_index=target.source_row_index,
                    source_path=target.source_path,
                )
            )
            continue
        fresh.append(target)
    return fresh, rejected


def load_depth_shadow_targets(
    path: Path,
    *,
    max_candidates: int | None = None,
    symbols: Iterable[str] | None = None,
    fallback_position_usdt: float | None = None,
) -> list[DepthShadowTarget]:
    """Load candidate-keyed depth targets from a CSV, Parquet, or Excel file."""
    df = _read_candidate_frame(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Candidate input {path} is missing required column 'symbol'")

    symbol_filter = {s.upper() for s in symbols} if symbols is not None else None
    targets: list[DepthShadowTarget] = []
    for idx, row in df.iterrows():
        symbol = _non_empty(row.get("symbol"))
        if symbol is None:
            continue
        symbol = symbol.upper()
        if symbol_filter is not None and symbol not in symbol_filter:
            continue

        candidate_id = _non_empty(row.get("candidate_id")) or f"{symbol}_row{idx}"
        scan_raw = _first_column_value(cast(pd.Series, row), _SCAN_TIME_COLUMNS)
        scan_ts = None
        if scan_raw is not None:
            parsed = pd.to_datetime(scan_raw, utc=True, errors="coerce")
            if not pd.isna(parsed):
                scan_ts = parsed.isoformat()
        if scan_ts is None:
            scan_ts = parse_candidate_scan_time_utc(str(candidate_id))

        position_notional = derive_position_notional_usdt(
            cast(pd.Series, row),
            fallback_position_usdt=fallback_position_usdt,
        )

        targets.append(
            DepthShadowTarget(
                symbol=symbol,
                candidate_id=str(candidate_id),
                scan_time_utc=scan_ts,
                position_notional_usdt=position_notional,
                source_row_index=_row_index_or_none(idx),
                source_path=str(path),
            )
        )
        if max_candidates is not None and len(targets) >= max_candidates:
            break
    return targets


def _level_depth_usdt(levels: Sequence[Sequence[Any]], top_n: int) -> float:
    total = 0.0
    for level in levels[:top_n]:
        if len(level) < 2:
            continue
        price = _float_or_none(level[0])
        qty = _float_or_none(level[1])
        if price is None or qty is None:
            continue
        total += price * qty
    return float(total)


def _fill_side_metrics(
    levels: Sequence[Sequence[Any]],
    *,
    target_notional_usdt: float | None,
    side: str,
) -> dict[str, float | int | bool | None]:
    if target_notional_usdt is None or target_notional_usdt <= 0:
        return {
            f"{side}_fill_ratio": None,
            f"{side}_impact_bps": None,
            f"{side}_levels_consumed": None,
            f"{side}_unfilled_notional_usdt": None,
            f"{side}_complete_fill": None,
        }
    if not levels:
        return {
            f"{side}_fill_ratio": 0.0,
            f"{side}_impact_bps": None,
            f"{side}_levels_consumed": 0,
            f"{side}_unfilled_notional_usdt": float(target_notional_usdt),
            f"{side}_complete_fill": False,
        }

    best = _float_or_none(levels[0][0] if len(levels[0]) >= 1 else None)
    filled_notional = 0.0
    filled_qty = 0.0
    levels_consumed = 0
    for level in levels:
        if len(level) < 2:
            continue
        price = _float_or_none(level[0])
        qty = _float_or_none(level[1])
        if price is None or qty is None or price <= 0 or qty <= 0:
            continue
        remaining = target_notional_usdt - filled_notional
        if remaining <= 0:
            break
        level_notional = price * qty
        take_notional = min(level_notional, remaining)
        filled_notional += take_notional
        filled_qty += take_notional / price
        levels_consumed += 1

    fill_ratio = min(1.0, filled_notional / target_notional_usdt)
    vwap = filled_notional / filled_qty if filled_qty > 0 else None
    impact_bps = None
    if best is not None and best > 0 and vwap is not None:
        if side == "buy":
            impact_bps = max(0.0, (vwap - best) / best * 10000.0)
        else:
            impact_bps = max(0.0, (best - vwap) / best * 10000.0)

    return {
        f"{side}_fill_ratio": float(fill_ratio),
        f"{side}_impact_bps": float(impact_bps) if impact_bps is not None else None,
        f"{side}_levels_consumed": int(levels_consumed),
        f"{side}_unfilled_notional_usdt": float(max(0.0, target_notional_usdt - filled_notional)),
        f"{side}_complete_fill": bool(fill_ratio >= 0.999999),
    }


def summarize_order_book(
    order_book: Mapping[str, Any],
    *,
    top_n: int = 20,
    position_notional_usdt: float | None = None,
    participation_rate: float = 0.10,
) -> dict[str, Any]:
    """Compute leakage-safe depth features from a single order-book snapshot."""
    bids = cast(Sequence[Sequence[Any]], order_book.get("bids") or [])
    asks = cast(Sequence[Sequence[Any]], order_book.get("asks") or [])
    best_bid = _float_or_none(bids[0][0] if bids and len(bids[0]) >= 1 else None)
    best_ask = _float_or_none(asks[0][0] if asks and len(asks[0]) >= 1 else None)
    mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    spread_pct = (
        ((best_ask - best_bid) / mid * 100.0)
        if mid is not None and mid > 0 and best_bid is not None and best_ask is not None
        else None
    )

    bid_depth = _level_depth_usdt(bids, top_n)
    ask_depth = _level_depth_usdt(asks, top_n)
    min_depth = min(bid_depth, ask_depth)
    total_depth = bid_depth + ask_depth
    available_capacity = min_depth * participation_rate
    depth_to_position = (
        min_depth / position_notional_usdt
        if position_notional_usdt is not None and position_notional_usdt > 0
        else None
    )
    capacity_ratio = (
        available_capacity / position_notional_usdt
        if position_notional_usdt is not None and position_notional_usdt > 0
        else None
    )

    out: dict[str, Any] = {
        "schema_version": DEPTH_SHADOW_SCHEMA_VERSION,
        "last_update_id": order_book.get("lastUpdateId"),
        "top_n": int(top_n),
        "participation_rate": float(participation_rate),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_pct": float(spread_pct) if spread_pct is not None else None,
        "top_n_bid_depth_usdt": float(bid_depth),
        "top_n_ask_depth_usdt": float(ask_depth),
        "top_n_depth_min_usdt": float(min_depth),
        "top_n_depth_total_usdt": float(total_depth),
        "book_imbalance_top_n": float((bid_depth - ask_depth) / total_depth) if total_depth > 0 else None,
        "position_notional_usdt": position_notional_usdt,
        "depth_to_position_min": float(depth_to_position) if depth_to_position is not None else None,
        "partial_fill_capacity_usdt": float(available_capacity),
        "partial_fill_capacity_ratio": float(capacity_ratio) if capacity_ratio is not None else None,
        "raw_bid_levels": len(bids),
        "raw_ask_levels": len(asks),
    }
    out.update(_fill_side_metrics(asks, target_notional_usdt=position_notional_usdt, side="buy"))
    out.update(_fill_side_metrics(bids, target_notional_usdt=position_notional_usdt, side="sell"))
    buy_impact = _float_or_none(out.get("buy_impact_bps"))
    sell_impact = _float_or_none(out.get("sell_impact_bps"))
    impacts = [v for v in (buy_impact, sell_impact) if v is not None]
    fill_ratios = [
        v for v in (_float_or_none(out.get("buy_fill_ratio")), _float_or_none(out.get("sell_fill_ratio")))
        if v is not None
    ]
    out["max_side_impact_bps"] = max(impacts) if impacts else None
    out["min_side_fill_ratio"] = min(fill_ratios) if fill_ratios else None
    return out


def make_depth_shadow_record(
    target: DepthShadowTarget,
    order_book: Mapping[str, Any],
    *,
    capture_time_utc: str,
    top_n: int = 20,
    participation_rate: float = 0.10,
    market_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one candidate-keyed depth-shadow record."""
    summary = summarize_order_book(
        order_book,
        top_n=top_n,
        position_notional_usdt=target.position_notional_usdt,
        participation_rate=participation_rate,
    )
    return {
        **asdict(target),
        "capture_time_utc": capture_time_utc,
        **summary,
        **dict(market_context or {}),
        "bids": order_book.get("bids", [])[:top_n],
        "asks": order_book.get("asks", [])[:top_n],
    }


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return cast(pd.Series, pd.to_numeric(df[column], errors="coerce"))


def build_candidate_depth_feature_frames(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split depth-shadow records into ex-ante features and window diagnostics.

    The returned feature frame uses only the first captured snapshot per
    candidate. The diagnostics frame may use all snapshots and must not be fed
    to the model as ex-ante features.
    """
    required = {"symbol", "candidate_id", "capture_time_utc"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Depth-shadow records missing required columns: {missing}")
    if records.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = records.copy()
    df["_capture_ts"] = pd.to_datetime(df["capture_time_utc"], utc=True, errors="coerce")
    if "iteration" in df.columns:
        iteration = cast(pd.Series, pd.to_numeric(df["iteration"], errors="coerce"))
        df["_iteration_sort"] = iteration.fillna(0)
    else:
        df["_iteration_sort"] = 0
    df = cast(pd.DataFrame, df.sort_values(["candidate_id", "_capture_ts", "_iteration_sort"]))
    first = cast(pd.DataFrame, df.groupby("candidate_id", sort=False).head(1).copy())

    feature_cols = [
        "spread_pct",
        "top_n_bid_depth_usdt",
        "top_n_ask_depth_usdt",
        "top_n_depth_min_usdt",
        "top_n_depth_total_usdt",
        "book_imbalance_top_n",
        "depth_to_position_min",
        "partial_fill_capacity_ratio",
        "min_side_fill_ratio",
        "max_side_impact_bps",
        "funding_rate",
        "estimated_abs_funding_pct",
        "round_trip_fee_pct",
        "basis_pct",
    ]
    feature_rows: dict[str, Any] = {
        "symbol": first["symbol"],
        "candidate_id": first["candidate_id"],
        "depth_feature_capture_time_utc": first["capture_time_utc"],
    }
    if "scan_time_utc" in first.columns:
        feature_rows["scan_time_utc"] = first["scan_time_utc"]
    for col in feature_cols:
        if col in first.columns:
            feature_rows[f"depth_scan_{col}"] = first[col]
    exante = pd.DataFrame(feature_rows)

    diag_rows: list[dict[str, Any]] = []
    for candidate_id, group in df.groupby("candidate_id", sort=False):
        g = cast(pd.DataFrame, group)
        depth = _numeric_series(g, "top_n_depth_min_usdt")
        spread = _numeric_series(g, "spread_pct")
        impact = _numeric_series(g, "max_side_impact_bps")
        capacity = _numeric_series(g, "partial_fill_capacity_ratio")
        fill_ratio = _numeric_series(g, "min_side_fill_ratio")
        depth_mean = float(depth.mean()) if bool(depth.notna().any()) else np.nan
        depth_std = float(depth.std(ddof=0)) if bool(depth.notna().any()) else np.nan
        diag_rows.append(
            {
                "symbol": g["symbol"].iloc[0],
                "candidate_id": candidate_id,
                "snapshot_count": int(len(g)),
                "first_capture_time_utc": g["capture_time_utc"].iloc[0],
                "last_capture_time_utc": g["capture_time_utc"].iloc[-1],
                "min_top_n_depth_min_usdt": float(depth.min()) if bool(depth.notna().any()) else np.nan,
                "median_top_n_depth_min_usdt": float(depth.median()) if bool(depth.notna().any()) else np.nan,
                "depth_stability_cv": (
                    float(depth_std / depth_mean)
                    if np.isfinite(depth_mean) and depth_mean > 0 and np.isfinite(depth_std)
                    else np.nan
                ),
                "median_spread_pct": float(spread.median()) if bool(spread.notna().any()) else np.nan,
                "max_spread_pct": float(spread.max()) if bool(spread.notna().any()) else np.nan,
                "median_max_side_impact_bps": float(impact.median()) if bool(impact.notna().any()) else np.nan,
                "max_side_impact_bps": float(impact.max()) if bool(impact.notna().any()) else np.nan,
                "min_partial_fill_capacity_ratio": float(capacity.min()) if bool(capacity.notna().any()) else np.nan,
                "min_side_fill_ratio": float(fill_ratio.min()) if bool(fill_ratio.notna().any()) else np.nan,
                "partial_fill_risk_snapshots": int((fill_ratio < 1.0).sum()) if bool(fill_ratio.notna().any()) else 0,
                "low_capacity_snapshots": int((capacity < 1.0).sum()) if bool(capacity.notna().any()) else 0,
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    return exante.reset_index(drop=True), diagnostics.reset_index(drop=True)


def append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append JSONL records, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str))
            handle.write("\n")
