# pyright: reportArgumentType=false, reportReturnType=false, reportOptionalMemberAccess=false
"""No-save validation of backtest behavior against expired live-bot evidence.

This diagnostic reads the expired-bot workbook, Binance manual exports, local
1-minute kline cache, and optional replay snapshots. It does not modify pipeline
inputs, model artifacts, deploy-ready outputs, or training data.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ERR-087 documented exemption (AUDIT_01 A02-F03): this diagnostic script
# imports the engine directly ONLY to (a) read grid_levels for seed
# construction and (b) extract the raw trade tape (backtester.trades) for
# timing metrics — neither is exposed by run_backtest(), and no label or
# training row is derived from these direct runs. All label-bearing results
# in this script go through run_backtest() below.
from backtest.backtest_realistic import RealisticGridBacktester
from backtest.btk_seed_state import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    LEGACY_REALISM_PROFILE,
    REALISM_PROFILES,
    SeedState,
    build_candidate_time_geometry_seed,
)
from backtest.btk_replay_seed_loader import load_seed_from_order_history_exports
from backtest.btk_unified_runner import build_training_config, run_backtest
from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.backtest.candidate_pipeline import (
    attach_mark_close,
    extract_exchange_filter_settings,
    funding_rate_series_from_rows,
    run_single_backtest,
)
from neutralgrid.backtest.realism_governance import validate_realism_output_path

logging.getLogger("backtest_realistic").setLevel(logging.WARNING)
logging.getLogger("backtest.btk_unified_runner").setLevel(logging.ERROR)


REQUIRED_WORKBOOK_FIELDS = (
    "symbol",
    "start_time_utc",
    "end_time_utc",
    "duration_hours",
    "invested_margin_usdt",
    "leverage",
    "grids_count",
    "price_range_low",
    "price_range_high",
    "pnl_pct",
    "total_profit_usdt",
    "total_trades",
    "mode",
)


@dataclass(frozen=True)
class ManualExports:
    orders: pd.DataFrame
    trades: pd.DataFrame
    transactions: pd.DataFrame
    input_trades: pd.DataFrame
    telemetry_times: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class TimestampDecision:
    stored_start_utc: pd.Timestamp
    stored_end_utc: pd.Timestamp
    candidate_local_to_utc_start: pd.Timestamp
    candidate_local_to_utc_end: pd.Timestamp
    manual_order_start_utc: pd.Timestamp | None
    manual_order_last_update_utc: pd.Timestamp | None
    telemetry_created_at_utc: pd.Timestamp | None
    selected_start_utc: pd.Timestamp | None
    selected_end_utc: pd.Timestamp | None
    selected_time_policy: str
    time_evidence_class: str
    time_evidence_source: str
    time_delta_seconds_stored_vs_manual: float | None
    time_delta_seconds_local_adjusted_vs_manual: float | None
    time_delta_seconds_stored_vs_evidence: float | None
    time_delta_seconds_local_adjusted_vs_evidence: float | None
    time_rejection_reason: str
    timestamp_modelable: bool


TIMESTAMP_POLICIES = (
    "stored_utc",
    "local_utc_minus_5_to_utc",
    "evidence_matched",
    "dual_diagnostic",
)
REALISM_ABLATIONS = (
    "profile",
    "legacy",
    "exchange_filters_only",
    "funding_series_only",
    "mark_valuation_only",
    "geometry_seed_only",
    "combined_public",
)
_LOCAL_TO_UTC_OFFSET = pd.Timedelta(hours=5)


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_csv_parts(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for path in paths:
        part = pd.read_csv(path, dtype=str)
        part["_source_file"] = path.name
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out.columns = out.columns.str.strip()
    return out


def _numeric(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().replace(",", "")
    if not text or text.lower() == "nan":
        return math.nan
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return math.nan


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.extract(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", expand=False),
        errors="coerce",
    )


def _utc_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if bool(pd.isna(ts)):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _required_utc_timestamp(value: Any, field_name: str) -> pd.Timestamp:
    ts = _utc_timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Missing required UTC timestamp field: {field_name}")
    return cast(pd.Timestamp, ts)


def _timestamp_second(value: pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    ts = _utc_timestamp(value)
    if pd.isna(ts):
        return None
    return ts.floor("s")


def _timestamp_text(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(_utc_timestamp(value))


def _delta_seconds(left: pd.Timestamp | None, right: pd.Timestamp | None) -> float | None:
    left_s = _timestamp_second(left)
    right_s = _timestamp_second(right)
    if left_s is None or right_s is None:
        return None
    return float((left_s - right_s).total_seconds())


def _timestamps_equal_at_second(left: pd.Timestamp | None, right: pd.Timestamp | None) -> bool:
    left_s = _timestamp_second(left)
    right_s = _timestamp_second(right)
    return left_s is not None and right_s is not None and left_s == right_s


def _bool_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.upper()
    return values.map({"TRUE": True, "FALSE": False, "1": True, "0": False})


def _strategy_id_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        fval = float(value)
        if math.isfinite(fval) and fval.is_integer():
            return str(int(fval))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_workbook_rows(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="General")
    df = df.copy()
    df["_row_id"] = np.arange(len(df), dtype=int)

    for col in ("start_time_utc", "end_time_utc", "last_updated_utc"):
        if col in df.columns:
            df[col] = _utc_series(df[col])

    for col in (
        "duration_hours",
        "invested_margin_usdt",
        "leverage",
        "grids_count",
        "price_range_low",
        "price_range_high",
        "pnl_pct",
        "total_profit_usdt",
        "realized_pnl_usdt",
        "unrealized_pnl_usdt",
        "funding_fee_usdt",
        "commission_usdt",
        "maker_count",
        "taker_count",
        "total_trades",
        "mae",
        "mfe",
        "mae_pct_initial",
        "mfe_pct_initial",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["strategy_id_text"] = df.get("strategy_id", pd.Series("", index=df.index)).map(
        _strategy_id_text
    )
    df["candidate_id_text"] = df.get("candidate_id", pd.Series("", index=df.index)).map(
        _strategy_id_text
    )
    df["mode"] = df.get("mode", pd.Series("", index=df.index)).astype(str).str.lower().str.strip()
    return df


def select_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    required_missing = [col for col in REQUIRED_WORKBOOK_FIELDS if col not in df.columns]
    if required_missing:
        raise ValueError(f"Workbook missing required fields: {required_missing}")

    eligible = df.copy()
    for col in REQUIRED_WORKBOOK_FIELDS:
        if col == "symbol":
            eligible = eligible.loc[eligible[col].astype(str).str.len() > 0]
        elif col == "mode":
            eligible = eligible.loc[eligible[col].isin(["arithmetic", "geometric"])]
        else:
            eligible = eligible.loc[eligible[col].notna()]

    if scope == "latest20":
        winners = eligible.loc[eligible["pnl_pct"] > 0].sort_values(
            "end_time_utc", ascending=False
        )
        return winners.drop_duplicates("symbol", keep="first").head(20).copy()

    dedup_key = np.where(
        eligible["strategy_id_text"].astype(str).str.len() > 0,
        "sid:" + eligible["strategy_id_text"].astype(str),
        (
            "fallback:"
            + eligible["candidate_id_text"].astype(str)
            + "|"
            + eligible["symbol"].astype(str)
            + "|"
            + eligible["start_time_utc"].astype(str)
        ),
    )
    eligible = eligible.assign(_dedup_key=dedup_key)
    return eligible.sort_values("end_time_utc", ascending=False).drop_duplicates(
        "_dedup_key", keep="first"
    )


def load_manual_exports(
    exports_dir: Path,
    live_telemetry_root: Path | None = None,
) -> ManualExports:
    csv_paths = sorted(exports_dir.glob("*.csv"))
    order_files = [p for p in csv_paths if "order history" in p.name.lower()]
    trade_files = [p for p in csv_paths if "trade history" in p.name.lower()]
    transaction_files = [p for p in csv_paths if "transaction history" in p.name.lower()]

    orders = _read_csv_parts(order_files)
    if not orders.empty:
        orders["_time"] = _utc_series(orders["Time(UTC)"])
        orders["_update_time"] = _utc_series(orders.get("Update Time", orders["Time(UTC)"]))
        orders["_symbol"] = orders["Symbol"].astype(str).str.strip().str.upper()
        orders["_order_id"] = orders["Order No"].astype(str).str.strip()
        orders["_strategy_id"] = orders.get("Strategy Id", pd.Series("", index=orders.index)).map(
            _strategy_id_text
        )
        orders["_executed_amount"] = _numeric_series(
            orders.get("Executed Amount", pd.Series(np.nan, index=orders.index))
        )
        orders["_price"] = _numeric_series(orders.get("Price", pd.Series(np.nan, index=orders.index)))
        orders["_avg_price"] = _numeric_series(
            orders.get("Average Price", pd.Series(np.nan, index=orders.index))
        )
        orders["_amount"] = _numeric_series(orders.get("Amount", pd.Series(np.nan, index=orders.index)))
        orders = (
            orders.sort_values(["_order_id", "_update_time", "_executed_amount"])
            .drop_duplicates("_order_id", keep="last")
            .reset_index(drop=True)
        )

    trades = _read_csv_parts(trade_files)
    if not trades.empty:
        trades["_time"] = _utc_series(trades["Time(UTC)"])
        trades["_symbol"] = trades["Symbol"].astype(str).str.strip().str.upper()
        trades["_order_id"] = trades["Order Id"].astype(str).str.strip()
        trades["_trade_id"] = trades.get("Trade Id", pd.Series("", index=trades.index)).astype(str).str.strip()
        trades["_price"] = _numeric_series(trades.get("Price", pd.Series(np.nan, index=trades.index)))
        trades["_qty"] = _numeric_series(trades.get("Quantity", pd.Series(np.nan, index=trades.index)))
        trades["_amount"] = _numeric_series(trades.get("Amount", pd.Series(np.nan, index=trades.index)))
        trades["_fee"] = _numeric_series(trades.get("Fee", pd.Series(np.nan, index=trades.index)))
        trades["_realized_profit"] = _numeric_series(
            trades.get("Realized Profit", pd.Series(np.nan, index=trades.index))
        )
        trades["_maker"] = _bool_series(trades.get("Maker", pd.Series("", index=trades.index)))
        trades = trades.drop_duplicates(
            subset=[
                "_trade_id",
                "_order_id",
                "_symbol",
                "_time",
                "_price",
                "_qty",
                "_fee",
                "_realized_profit",
            ],
            keep="first",
        ).reset_index(drop=True)

    transactions = _read_csv_parts(transaction_files)
    if not transactions.empty:
        transactions["_time"] = _utc_series(transactions["Date(UTC)"])
        transactions["_symbol"] = transactions.get("Symbol", pd.Series("", index=transactions.index)).astype(str).str.strip().str.upper()
        transactions["_type"] = transactions.get("type", pd.Series("", index=transactions.index)).astype(str).str.strip().str.upper()
        transactions["_amount"] = _numeric_series(
            transactions.get("Amount", pd.Series(np.nan, index=transactions.index))
        )
        transactions["_transaction_id"] = transactions.get(
            "Transaction ID", pd.Series("", index=transactions.index)
        ).astype(str).str.strip()
        transactions = transactions.drop_duplicates(
            subset=["_transaction_id", "_time", "_type", "_symbol", "_amount"],
            keep="first",
        ).reset_index(drop=True)

    input_trades = load_manual_input_trades(ROOT / "data" / "manual_input")
    return ManualExports(
        orders=orders,
        trades=trades,
        transactions=transactions,
        input_trades=input_trades,
        telemetry_times=load_live_telemetry_times(
            live_telemetry_root if live_telemetry_root is not None else ROOT / "Live"
        ),
    )


def load_live_telemetry_times(live_root: Path) -> pd.DataFrame:
    """Read exact bot creation times from policy-compliant live snapshots.

    Only ``Live/YYYY-MM-DD/SYMBOL/private_telemetry_*.json`` is considered.
    Rows without exact strategy identity or a parseable Lima creation time are
    not timestamp evidence.
    """
    rows: list[dict[str, Any]] = []
    if not live_root.exists():
        return pd.DataFrame()
    for path in sorted(live_root.glob("20??-??-??/*/private_telemetry_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).warning(
                "Skipping unreadable live telemetry snapshot %s: %s",
                path,
                exc,
            )
            continue
        if not isinstance(payload, dict) or payload.get("data_class") != "live_bot_telemetry":
            continue
        symbol = str(payload.get("symbol", "")).strip().upper()
        strategy_id = _strategy_id_text(payload.get("strategy_id"))
        created_text = str(payload.get("created_at_lima", "")).strip()
        if not symbol or not strategy_id or not created_text:
            continue
        try:
            created = pd.Timestamp(created_text)
            if bool(pd.isna(created)):
                continue
            if created.tzinfo is None:
                created = created.tz_localize("America/Lima")
            created = created.tz_convert("UTC")
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "_symbol": symbol,
                "_strategy_id": strategy_id,
                "_created_time": created,
                "_source_file": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_manual_input_trades(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not input_dir.exists():
        return pd.DataFrame()
    trade_re = re.compile(
        r"(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+"
        r"(Buy|Sell)\s+Limit\s+"
        r"([0-9.]+)\s+USDT\s+"
        r"([0-9.]+)\s+([A-Z0-9]+)\s+"
        r"([0-9.]+)\s+USDT\s+"
        r"([0-9.]+)\s+USDT"
    )
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        first_symbol = re.search(r"\b([A-Z0-9]+USDT)\b", text)
        strategy_match = re.search(r"Strategy Number\s+(\d+)", text)
        symbol = first_symbol.group(1).upper() if first_symbol else path.stem.split("_")[0].upper()
        strategy_id = strategy_match.group(1) if strategy_match else ""
        for match in trade_re.finditer(text):
            rows.append(
                {
                    "_source_file": str(path),
                    "_time": pd.to_datetime(match.group(1), utc=True, errors="coerce"),
                    "_symbol": symbol,
                    "_strategy_id": strategy_id,
                    "_side": match.group(2).lower(),
                    "_price": float(match.group(3)),
                    "_qty": float(match.group(4)),
                    "_asset": match.group(5),
                    "_amount": float(match.group(6)),
                    "_fee": float(match.group(7)),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.drop_duplicates(
        subset=["_source_file", "_time", "_side", "_price", "_qty", "_amount", "_fee"],
        keep="first",
    ).reset_index(drop=True)


def _row_end(
    row: pd.Series,
    duration_source: str,
    fixed_hours: float,
    start: pd.Timestamp | None = None,
) -> pd.Timestamp:
    start_ts = pd.Timestamp(row["start_time_utc"]) if start is None else start
    if duration_source == "fixed":
        return start_ts + pd.Timedelta(hours=fixed_hours)
    duration_hours = row.get("duration_hours", math.nan)
    if start_ts != pd.Timestamp(row["start_time_utc"]) and _finite(duration_hours):
        return start_ts + pd.Timedelta(hours=float(duration_hours))
    return pd.Timestamp(row["end_time_utc"])


def _order_history_strategy_window(
    row: pd.Series,
    exports: ManualExports,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start, end, _class, _source_count = _manual_order_time_evidence(row, exports)
    return start, end


def _manual_order_time_evidence(
    row: pd.Series,
    exports: ManualExports,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str, int]:
    if exports.orders.empty:
        return None, None, "missing_manual_evidence", 0
    strategy_id = str(row.get("strategy_id_text", "") or "")
    if not strategy_id:
        return None, None, "missing_manual_evidence", 0
    symbol = str(row["symbol"])
    orders = exports.orders
    matched = orders.loc[
        (orders["_symbol"] == symbol) & (orders["_strategy_id"] == strategy_id)
    ].copy()
    if matched.empty:
        return None, None, "missing_manual_evidence", 0
    source_col = "_source_file" if "_source_file" in matched.columns else None
    if source_col is not None:
        starts_by_source = (
            matched.dropna(subset=["_time"])
            .groupby(source_col)["_time"]
            .min()
            .dropna()
            .map(_timestamp_second)
        )
        unique_source_starts = {
            item for item in starts_by_source.tolist() if item is not None
        }
        source_count = len(unique_source_starts)
        if source_count > 1:
            start = matched["_time"].dropna().min()
            end = matched["_update_time"].dropna().max()
            start_ts = pd.Timestamp(start) if pd.notna(start) else None
            end_ts = pd.Timestamp(end) if pd.notna(end) else None
            return start_ts, end_ts, "ambiguous_manual_evidence", source_count
    else:
        source_count = 1
    start = matched["_time"].dropna().min()
    end = matched["_update_time"].dropna().max()
    start_ts = pd.Timestamp(start) if pd.notna(start) else None
    end_ts = pd.Timestamp(end) if pd.notna(end) else None
    return start_ts, end_ts, "manual_evidence_present", source_count


def _telemetry_time_evidence(
    row: pd.Series,
    exports: ManualExports,
) -> tuple[pd.Timestamp | None, str, int]:
    telemetry = exports.telemetry_times
    if telemetry.empty:
        return None, "missing_telemetry_evidence", 0
    strategy_id = str(row.get("strategy_id_text", "") or "")
    if not strategy_id:
        return None, "missing_telemetry_evidence", 0
    matched = telemetry.loc[
        (telemetry["_symbol"] == str(row["symbol"]))
        & (telemetry["_strategy_id"] == strategy_id)
    ].copy()
    if matched.empty:
        return None, "missing_telemetry_evidence", 0
    starts = {
        item
        for item in matched["_created_time"].dropna().map(_timestamp_second).tolist()
        if item is not None
    }
    if len(starts) != 1:
        return None, "ambiguous_telemetry_evidence", len(starts)
    return next(iter(starts)), "telemetry_evidence_present", 1


def _timestamp_decision(
    row: pd.Series,
    exports: ManualExports,
    *,
    timestamp_policy: str,
    selected_time_policy: str,
    duration_source: str,
    fixed_duration_hours: float,
) -> TimestampDecision:
    stored_start = _required_utc_timestamp(row["start_time_utc"], "start_time_utc")
    stored_end = cast(pd.Timestamp, (
        stored_start + pd.Timedelta(hours=fixed_duration_hours)
        if duration_source == "fixed"
        else _required_utc_timestamp(row["end_time_utc"], "end_time_utc")
    ))
    local_start = cast(pd.Timestamp, stored_start + _LOCAL_TO_UTC_OFFSET)
    local_end = cast(pd.Timestamp, (
        local_start + pd.Timedelta(hours=fixed_duration_hours)
        if duration_source == "fixed"
        else stored_end + _LOCAL_TO_UTC_OFFSET
    ))
    manual_start, manual_end, manual_class, _source_count = _manual_order_time_evidence(row, exports)
    telemetry_start, telemetry_class, _telemetry_source_count = _telemetry_time_evidence(
        row, exports
    )
    evidence_start = manual_start
    evidence_class = manual_class
    evidence_source = (
        "manual_order_history"
        if manual_class != "missing_manual_evidence"
        else "none"
    )
    if manual_class == "missing_manual_evidence" and telemetry_class != "missing_telemetry_evidence":
        evidence_start = telemetry_start
        evidence_class = telemetry_class
        evidence_source = "live_telemetry_created_at_lima"
    if evidence_class in {"manual_evidence_present", "telemetry_evidence_present"}:
        if _timestamps_equal_at_second(evidence_start, stored_start):
            time_class = "exact_stored_match"
        elif _timestamps_equal_at_second(evidence_start, local_start):
            time_class = "exact_local_offset_match"
        else:
            time_class = "conflicting_time_evidence"
    else:
        time_class = evidence_class

    selected_start: pd.Timestamp | None
    selected_end: pd.Timestamp | None
    rejection = ""
    modelable = True
    if selected_time_policy == "stored_utc":
        selected_start = stored_start
        selected_end = stored_end
    elif selected_time_policy == "local_utc_minus_5_to_utc":
        selected_start = local_start
        selected_end = local_end
    elif selected_time_policy == "evidence_matched":
        if time_class == "exact_stored_match":
            selected_start = stored_start
            selected_end = stored_end
            selected_time_policy = "stored_utc"
        elif time_class == "exact_local_offset_match":
            selected_start = local_start
            selected_end = local_end
            selected_time_policy = "local_utc_minus_5_to_utc"
        else:
            selected_start = None
            selected_end = None
            modelable = False
            rejection = f"time_evidence_class:{time_class}"
    else:
        raise ValueError(f"Unsupported selected_time_policy: {selected_time_policy!r}")

    if timestamp_policy == "dual_diagnostic":
        rejection = "diagnostic_only_not_promotable"

    return TimestampDecision(
        stored_start_utc=stored_start,
        stored_end_utc=stored_end,
        candidate_local_to_utc_start=local_start,
        candidate_local_to_utc_end=local_end,
        manual_order_start_utc=manual_start,
        manual_order_last_update_utc=manual_end,
        telemetry_created_at_utc=telemetry_start,
        selected_start_utc=selected_start,
        selected_end_utc=selected_end,
        selected_time_policy=selected_time_policy,
        time_evidence_class=time_class,
        time_evidence_source=evidence_source,
        time_delta_seconds_stored_vs_manual=_delta_seconds(stored_start, manual_start),
        time_delta_seconds_local_adjusted_vs_manual=_delta_seconds(local_start, manual_start),
        time_delta_seconds_stored_vs_evidence=_delta_seconds(stored_start, evidence_start),
        time_delta_seconds_local_adjusted_vs_evidence=_delta_seconds(
            local_start, evidence_start
        ),
        time_rejection_reason=rejection,
        timestamp_modelable=modelable,
    )


def _timestamp_decisions_for_policy(
    row: pd.Series,
    exports: ManualExports,
    *,
    timestamp_policy: str,
    duration_source: str,
    fixed_duration_hours: float,
) -> list[TimestampDecision]:
    if timestamp_policy == "dual_diagnostic":
        return [
            _timestamp_decision(
                row,
                exports,
                timestamp_policy=timestamp_policy,
                selected_time_policy="stored_utc",
                duration_source=duration_source,
                fixed_duration_hours=fixed_duration_hours,
            ),
            _timestamp_decision(
                row,
                exports,
                timestamp_policy=timestamp_policy,
                selected_time_policy="local_utc_minus_5_to_utc",
                duration_source=duration_source,
                fixed_duration_hours=fixed_duration_hours,
            ),
        ]
    return [
        _timestamp_decision(
            row,
            exports,
            timestamp_policy=timestamp_policy,
            selected_time_policy=timestamp_policy,
            duration_source=duration_source,
            fixed_duration_hours=fixed_duration_hours,
        )
    ]


def _timestamp_decision_fields(decision: TimestampDecision) -> dict[str, Any]:
    return {
        "stored_start_utc": _timestamp_text(decision.stored_start_utc),
        "stored_end_utc": _timestamp_text(decision.stored_end_utc),
        "candidate_local_to_utc_start": _timestamp_text(
            decision.candidate_local_to_utc_start
        ),
        "candidate_local_to_utc_end": _timestamp_text(decision.candidate_local_to_utc_end),
        "manual_order_start_utc": _timestamp_text(decision.manual_order_start_utc),
        "manual_order_last_update_utc": _timestamp_text(
            decision.manual_order_last_update_utc
        ),
        "telemetry_created_at_utc": _timestamp_text(decision.telemetry_created_at_utc),
        "selected_start_utc": _timestamp_text(decision.selected_start_utc),
        "selected_end_utc": _timestamp_text(decision.selected_end_utc),
        "selected_time_policy": decision.selected_time_policy,
        "time_evidence_class": decision.time_evidence_class,
        "time_evidence_source": decision.time_evidence_source,
        "time_delta_seconds_stored_vs_manual": decision.time_delta_seconds_stored_vs_manual,
        "time_delta_seconds_local_adjusted_vs_manual": (
            decision.time_delta_seconds_local_adjusted_vs_manual
        ),
        "time_delta_seconds_stored_vs_evidence": (
            decision.time_delta_seconds_stored_vs_evidence
        ),
        "time_delta_seconds_local_adjusted_vs_evidence": (
            decision.time_delta_seconds_local_adjusted_vs_evidence
        ),
        "time_rejection_reason": decision.time_rejection_reason,
        "timestamp_modelable": bool(decision.timestamp_modelable),
    }


def _manual_metrics(
    row: pd.Series,
    exports: ManualExports,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    symbol = str(row["symbol"])
    strategy_id = str(row.get("strategy_id_text", "") or "")

    orders = exports.orders
    order_symbol = pd.DataFrame()
    order_strategy = pd.DataFrame()
    if not orders.empty:
        order_symbol = orders.loc[
            (orders["_symbol"] == symbol)
            & (orders["_time"] >= start)
            & (orders["_time"] <= end)
        ].copy()
        if strategy_id:
            order_strategy = order_symbol.loc[order_symbol["_strategy_id"] == strategy_id].copy()

    order_ids = set(order_strategy["_order_id"].astype(str)) if not order_strategy.empty else set()

    trades = exports.trades
    trade_symbol = pd.DataFrame()
    trade_strategy = pd.DataFrame()
    if not trades.empty:
        trade_symbol = trades.loc[
            (trades["_symbol"] == symbol)
            & (trades["_time"] >= start)
            & (trades["_time"] <= end)
        ].copy()
        if order_ids:
            trade_strategy = trade_symbol.loc[trade_symbol["_order_id"].astype(str).isin(order_ids)].copy()

    transactions = exports.transactions
    tx_symbol = pd.DataFrame()
    if not transactions.empty:
        tx_symbol = transactions.loc[
            (transactions["_symbol"].isin([symbol, ""]))
            & (transactions["_time"] >= start)
            & (transactions["_time"] <= end)
        ].copy()
        if not tx_symbol.empty:
            symbol_specific = tx_symbol.loc[tx_symbol["_symbol"] == symbol]
            if not symbol_specific.empty:
                tx_symbol = symbol_specific

    input_trades = exports.input_trades
    input_strategy = pd.DataFrame()
    if not input_trades.empty:
        input_symbol = input_trades.loc[
            (input_trades["_symbol"] == symbol)
            & (input_trades["_time"] >= start)
            & (input_trades["_time"] <= end)
        ].copy()
        if strategy_id:
            input_strategy = input_symbol.loc[input_symbol["_strategy_id"] == strategy_id].copy()
        else:
            input_strategy = input_symbol

    fee_raw = float(trade_strategy["_fee"].dropna().sum()) if not trade_strategy.empty else math.nan
    fee_cost = float(trade_strategy["_fee"].dropna().abs().sum()) if not trade_strategy.empty else math.nan
    realized_sum = (
        float(trade_strategy["_realized_profit"].dropna().sum()) if not trade_strategy.empty else math.nan
    )
    maker_count = int((trade_strategy["_maker"] == True).sum()) if not trade_strategy.empty else 0
    taker_count = int((trade_strategy["_maker"] == False).sum()) if not trade_strategy.empty else 0
    trade_timing = _trade_timing_metrics(trade_strategy["_time"]) if not trade_strategy.empty else {}
    input_timing = _trade_timing_metrics(input_strategy["_time"]) if not input_strategy.empty else {}
    any_trade_count = int(len(trade_strategy)) if not trade_strategy.empty else int(len(input_strategy))
    any_timing = trade_timing if trade_timing else input_timing
    any_source = "manual_exports" if not trade_strategy.empty else ("manual_input" if not input_strategy.empty else "")

    tx_realized = math.nan
    tx_commission = math.nan
    tx_funding = math.nan
    tx_net_income = math.nan
    if not tx_symbol.empty:
        tx_realized = float(tx_symbol.loc[tx_symbol["_type"] == "REALIZED_PNL", "_amount"].dropna().sum())
        tx_commission = float(tx_symbol.loc[tx_symbol["_type"] == "COMMISSION", "_amount"].dropna().sum())
        tx_funding = float(tx_symbol.loc[tx_symbol["_type"] == "FUNDING_FEE", "_amount"].dropna().sum())
        tx_net_income = float(
            tx_symbol.loc[
                tx_symbol["_type"].isin(["REALIZED_PNL", "COMMISSION", "FUNDING_FEE"]),
                "_amount",
            ].dropna().sum()
        )

    strict = bool(order_ids and not trade_strategy.empty)
    trade_plus_tx = math.nan
    if math.isfinite(realized_sum):
        funding_component = tx_funding if math.isfinite(tx_funding) else 0.0
        fee_component = fee_cost if math.isfinite(fee_cost) else 0.0
        trade_plus_tx = realized_sum + funding_component - fee_component

    out = {
        "manual_order_symbol_time_count": int(len(order_symbol)),
        "manual_order_strategy_count": int(len(order_strategy)),
        "manual_trade_symbol_time_count": int(len(trade_symbol)),
        "manual_trade_strategy_count": int(len(trade_strategy)),
        "manual_input_trade_strategy_count": int(len(input_strategy)),
        "manual_any_trade_count": any_trade_count,
        "manual_any_trade_source": any_source,
        "manual_tx_symbol_time_count": int(len(tx_symbol)),
        "manual_strict_reconstructable": strict,
        "manual_any_reconstructable": bool(any_trade_count > 0),
        "manual_maker_count": maker_count,
        "manual_taker_count": taker_count,
        "manual_trade_fee_raw_sum_usdt": fee_raw,
        "manual_trade_fee_cost_usdt": fee_cost,
        "manual_trade_realized_pnl_sum_usdt": realized_sum,
        "manual_tx_realized_pnl_usdt": tx_realized,
        "manual_tx_commission_usdt": tx_commission,
        "manual_tx_funding_fee_usdt": tx_funding,
        "manual_tx_net_income_usdt": tx_net_income,
        "manual_trade_plus_funding_minus_fee_usdt": trade_plus_tx,
    }
    out.update({f"manual_{key}": value for key, value in trade_timing.items()})
    out.update({f"manual_input_{key}": value for key, value in input_timing.items()})
    out.update({f"manual_any_{key}": value for key, value in any_timing.items()})
    return out


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    start_month = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    months = pd.date_range(start_month, end_month, freq="MS", tz="UTC")
    return [(f"{dt.year:04d}", f"{dt.month:02d}") for dt in months]


def load_cached_klines(cache_root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year, month in _months_between(start, end):
        month_dir = cache_root / symbol / "1m" / "parquet" / f"year={year}" / f"month={month}"
        if not month_dir.exists():
            continue
        for path in sorted(month_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame()

    klines = pd.concat(parts, ignore_index=True)
    if "open_time" not in klines.columns:
        return pd.DataFrame()
    klines["timestamp"] = pd.to_datetime(klines["open_time"], utc=True, errors="coerce")
    filtered = klines.loc[(klines["timestamp"] >= start) & (klines["timestamp"] <= end)].copy()
    if filtered.empty:
        return filtered
    return filtered[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


def _load_cached_klines_resilient(
    cache_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str | None]:
    """Contain a corrupt cache partition to one diagnostic row.

    The reconciliation is an evidence audit across independent bot rows.  A
    broken parquet file must be recorded explicitly, but it must not erase the
    audit evidence available for every other row.
    """
    try:
        return load_cached_klines(cache_root, symbol, start, end), None
    except Exception as exc:
        return pd.DataFrame(), f"{type(exc).__name__}: {exc}"


def _expand_ohlc_path(klines: pd.DataFrame, path_mode: str) -> pd.DataFrame:
    if path_mode == "current" or klines.empty:
        return klines
    sequence = ("open", "high", "low", "close") if path_mode == "ohlc" else ("open", "low", "high", "close")
    rows: list[dict[str, Any]] = []
    for _, row in klines.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        values = [float(row[col]) for col in sequence]
        prev = values[0]
        for offset, value in enumerate(values):
            rows.append(
                {
                    "timestamp": timestamp + pd.Timedelta(seconds=offset * 15),
                    "open": prev,
                    "high": max(prev, value),
                    "low": min(prev, value),
                    "close": value,
                    "volume": float(row.get("volume", 0.0)) / len(values),
                }
            )
            prev = value
    return pd.DataFrame(rows)


def _build_model_config(
    row: pd.Series,
    physics_overrides: dict[str, Any] | None = None,
):
    overrides: dict[str, Any] = {
        "capital": float(row["invested_margin_usdt"]),
        "leverage": int(row["leverage"]),
        "mode": str(row["mode"]),
    }
    if physics_overrides:
        overrides.update(physics_overrides)
    return build_training_config(
        str(row["symbol"]),
        float(row["price_range_low"]),
        float(row["price_range_high"]),
        int(row["grids_count"]),
        **overrides,
    )


def _run_model(
    row: pd.Series,
    klines: pd.DataFrame,
    physics_overrides: dict[str, Any] | None = None,
    seed_state: SeedState | None = None,
    realism_profile: str = LEGACY_REALISM_PROFILE,
) -> dict[str, Any]:
    cfg = _build_model_config(row, physics_overrides)
    return run_backtest(
        cfg,
        klines,
        seed_state=seed_state,
        realism_profile=realism_profile,
    )


def _candidate_time_geometry_seed(
    row: pd.Series,
    klines: pd.DataFrame,
    validation_start: pd.Timestamp,
    physics_overrides: dict[str, Any] | None = None,
) -> SeedState | None:
    if klines.empty:
        return None
    mode = str(row.get("mode", "")).strip().lower()
    if mode != "geometric":
        return None
    first_close = _numeric(klines.iloc[0].get("close"))
    if not _finite(first_close) or first_close <= 0:
        return None

    cfg = _build_model_config(row, physics_overrides)
    backtester = RealisticGridBacktester(cfg)
    return build_candidate_time_geometry_seed(
        symbol=str(row["symbol"]),
        grid_levels=backtester.grid_levels,
        start_price=first_close,
        t0=validation_start.to_pydatetime(),
    )


def _trade_timing_metrics(times: pd.Series | list[Any]) -> dict[str, Any]:
    series = pd.to_datetime(pd.Series(times), utc=True, errors="coerce").dropna().sort_values()
    if series.empty:
        return {}
    diffs = series.diff().dt.total_seconds().dropna()
    per_minute = series.dt.floor("min").value_counts()
    return {
        "first_trade_time_utc": str(series.iloc[0]),
        "last_trade_time_utc": str(series.iloc[-1]),
        "trade_span_minutes": float((series.iloc[-1] - series.iloc[0]).total_seconds() / 60.0),
        "min_intertrade_seconds": float(diffs.min()) if not diffs.empty else None,
        "median_intertrade_seconds": float(diffs.median()) if not diffs.empty else None,
        "mean_intertrade_seconds": float(diffs.mean()) if not diffs.empty else None,
        "max_same_minute_trades": int(per_minute.max()) if not per_minute.empty else 0,
        "sub_120s_gap_count": int((diffs < 120.0).sum()) if not diffs.empty else 0,
        "zero_second_gap_count": int((diffs == 0.0).sum()) if not diffs.empty else 0,
    }


def _run_model_trade_metrics(
    row: pd.Series,
    klines: pd.DataFrame,
    physics_overrides: dict[str, Any] | None = None,
    seed_state: SeedState | None = None,
) -> dict[str, Any]:
    cfg = _build_model_config(row, physics_overrides)
    # ERR-087 exemption: direct engine use to read the raw trade tape
    # (run_backtest() does not expose backtester.trades). Diagnostic only —
    # output feeds timing metrics, never labels or training rows.
    backtester = RealisticGridBacktester(cfg)
    if seed_state is not None:
        backtester.seed_from_state(seed_state)
    backtester.run(klines)
    times = [trade.timestamp for trade in backtester.trades]
    out = {f"model_{key}": value for key, value in _trade_timing_metrics(times).items()}
    out["model_trade_count_from_tape"] = int(len(backtester.trades))
    return out


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _evidence_class(metrics: dict[str, Any], seed_state: SeedState | None) -> str:
    if seed_state is not None and seed_state.evidence_class == "conflicting":
        return "conflicting"
    if seed_state is None and not bool(metrics.get("manual_any_reconstructable", False)):
        return "missing"

    live_fill_count = metrics.get("live_fill_count", math.nan)
    manual_trade_count = metrics.get("manual_trade_strategy_count", 0)
    tx_count = metrics.get("manual_tx_symbol_time_count", 0)
    strict = bool(metrics.get("manual_strict_reconstructable", False))
    if (
        seed_state is not None
        and strict
        and _finite(live_fill_count)
        and float(live_fill_count) > 0
        and int(manual_trade_count or 0) >= float(live_fill_count)
        and int(tx_count or 0) > 0
    ):
        return "complete"
    return "partial"


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if "kline_cache_status" in df.columns:
        kline_status = cast(pd.Series, df["kline_cache_status"]).fillna("").astype(str)
        missing_kline_mask = kline_status.eq("missing")
        kline_error_rows = int(kline_status.eq("error").sum())
        kline_not_attempted_rows = int(
            kline_status.eq("not_attempted_timestamp_rejected").sum()
        )
    else:
        missing_kline_mask = (
            ~cast(pd.Series, df["klines_available"]).fillna(False).astype(bool)
            if "klines_available" in df.columns
            else pd.Series(False, index=df.index, dtype=bool)
        )
        kline_error_rows = 0
        kline_not_attempted_rows = 0
    out: dict[str, Any] = {
        "scope": scope,
        "diagnostic_fast_winner_contract": "terminal_pnl_pct_gt_1_and_duration_hours_lt_7",
        "active_meta_target_contract_evaluated": False,
        "active_meta_target_contract_limitation": (
            "this validator does not reconstruct time_to_3pct; its fast-winner metric "
            "must not be interpreted as fast_winner_time_to_3pct_le_7h accuracy"
        ),
        "rows": int(len(df)),
        "model_rows": int(df["model_ran"].sum()) if "model_ran" in df else 0,
        "strict_manual_rows": int(df["manual_strict_reconstructable"].sum())
        if "manual_strict_reconstructable" in df
        else 0,
        "manual_any_rows": int(df["manual_any_reconstructable"].sum())
        if "manual_any_reconstructable" in df
        else 0,
        "missing_kline_rows": int(missing_kline_mask.sum()),
        "kline_cache_error_rows": kline_error_rows,
        "kline_not_attempted_timestamp_rejected_rows": kline_not_attempted_rows,
        "symbols_missing_klines": sorted(
            df.loc[missing_kline_mask, "symbol"].astype(str).unique().tolist()
        )
        if "symbol" in df.columns
        else [],
    }
    model_df = df.loc[df["model_ran"]].copy() if "model_ran" in df else pd.DataFrame()
    if not model_df.empty:
        model_df["abs_model_pnl_error"] = (model_df["model_pnl_pct"] - model_df["live_pnl_pct"]).abs()
        model_df["abs_model_trade_error"] = (
            model_df["model_total_trades"] - model_df["live_fill_count"]
        ).abs()
        out.update(
            {
                "model_mean_pnl_pct": float(model_df["model_pnl_pct"].mean()),
                "live_mean_pnl_pct": float(model_df["live_pnl_pct"].mean()),
                "model_mean_abs_pnl_error": float(model_df["abs_model_pnl_error"].mean()),
                "model_median_abs_pnl_error": float(model_df["abs_model_pnl_error"].median()),
                "model_pnl_sign_match_rate": float(
                    (
                        np.sign(model_df["model_pnl_pct"].to_numpy(dtype=float))
                        == np.sign(model_df["live_pnl_pct"].to_numpy(dtype=float))
                    ).mean()
                ),
                "model_mean_trade_count": float(model_df["model_total_trades"].mean()),
                "live_mean_trade_count": float(model_df["live_fill_count"].mean()),
                "model_mean_abs_trade_count_error": float(model_df["abs_model_trade_error"].mean()),
            }
        )
        live_positive = model_df["live_pnl_pct"].astype(float) > 0.0
        model_positive = model_df["model_pnl_pct"].astype(float) > 0.0
        live_fast_winner = (model_df["live_pnl_pct"].astype(float) > 1.0) & (
            model_df["live_duration_hours"].astype(float) < 7.0
        )
        model_fast_winner = model_df["model_pnl_pct"].astype(float) > 1.0
        out.update(
            {
                "live_positive_rows_modelable": int(live_positive.sum()),
                "live_non_positive_rows_modelable": int((~live_positive).sum()),
                "model_positive_rows": int(model_positive.sum()),
                "model_non_positive_rows": int((~model_positive).sum()),
                "winner_recall_pnl_gt_0": float(
                    ((live_positive) & (model_positive)).sum() / live_positive.sum()
                )
                if int(live_positive.sum()) > 0
                else None,
                "non_winner_specificity_pnl_lte_0": float(
                    ((~live_positive) & (~model_positive)).sum() / (~live_positive).sum()
                )
                if int((~live_positive).sum()) > 0
                else None,
                "live_fast_winner_rows_modelable": int(live_fast_winner.sum()),
                "model_fast_winner_rows": int(model_fast_winner.sum()),
                "fast_winner_recall_pnl_gt_1_duration_lt_7": float(
                    ((live_fast_winner) & (model_fast_winner)).sum() / live_fast_winner.sum()
                )
                if int(live_fast_winner.sum()) > 0
                else None,
            }
        )
        capital_error = (
            model_df["model_capital_used_usdt"].astype(float)
            - model_df["live_invested_margin_usdt"].astype(float)
        ).abs()
        out["capital_abs_error_mean"] = float(capital_error.mean())
        top_errors = model_df.assign(
            abs_model_pnl_error=(
                model_df["model_pnl_pct"].astype(float)
                - model_df["live_pnl_pct"].astype(float)
            ).abs()
        ).sort_values("abs_model_pnl_error", ascending=False)
        out["top_model_pnl_errors"] = [
            {
                "symbol": str(item["symbol"]),
                "strategy_id": str(item.get("strategy_id", "")),
                "live_pnl_pct": float(item["live_pnl_pct"]),
                "model_pnl_pct": float(item["model_pnl_pct"]),
                "abs_error": float(item["abs_model_pnl_error"]),
                "evidence_class": str(item.get("evidence_class", "")),
                "seed_state_source": str(item.get("seed_state_source", "")),
            }
            for item in top_errors.head(10).to_dict(orient="records")
        ]
        if "evidence_class" in model_df.columns:
            out["evidence_class_counts"] = {
                str(key): int(value)
                for key, value in model_df["evidence_class"].value_counts(dropna=False).to_dict().items()
            }
        if "seed_state_source" in model_df.columns:
            seeded = model_df.loc[model_df["seed_state_source"].astype(str) != "none"].copy()
            out["seeded_model_rows"] = int(len(seeded))
            if not seeded.empty:
                seeded["abs_model_pnl_error"] = (
                    seeded["model_pnl_pct"].astype(float) - seeded["live_pnl_pct"].astype(float)
                ).abs()
                seeded["abs_seed_trade_error"] = (
                    seeded["model_total_trades"].astype(float)
                    - seeded["live_fill_count"].astype(float)
                ).abs()
                out["seeded_model_mean_abs_pnl_error"] = float(
                    seeded["abs_model_pnl_error"].mean()
                )
                out["seeded_model_median_abs_pnl_error"] = float(
                    seeded["abs_model_pnl_error"].median()
                )
                out["seeded_model_mean_abs_trade_count_error"] = float(
                    seeded["abs_seed_trade_error"].mean()
                )
                out["seeded_position_size_source_counts"] = {
                    str(key): int(value)
                    for key, value in seeded["model_position_size_source"].value_counts(
                        dropna=False
                    ).to_dict().items()
                }
        if {"model_mae_pct_initial", "live_mae_pct_initial"}.issubset(model_df.columns):
            mae_pairs = model_df.loc[
                model_df["model_mae_pct_initial"].notna() & model_df["live_mae_pct_initial"].notna()
            ].copy()
            if not mae_pairs.empty:
                out["mae_pct_initial_pair_rows"] = int(len(mae_pairs))
                out["mae_pct_initial_mean_abs_error"] = float(
                    (
                        mae_pairs["model_mae_pct_initial"].astype(float)
                        - mae_pairs["live_mae_pct_initial"].astype(float)
                    ).abs().mean()
                )

    strict_mask = (
        cast(pd.Series, df["manual_strict_reconstructable"])
        .astype("boolean")
        .fillna(False)
        .astype(bool)
        if "manual_strict_reconstructable" in df.columns
        else pd.Series(False, index=df.index, dtype=bool)
    )
    strict_df = df.loc[strict_mask].copy()
    if not strict_df.empty:
        strict_df["manual_trade_coverage_ratio"] = strict_df["manual_trade_strategy_count"] / strict_df[
            "live_fill_count"
        ].replace(0, np.nan)
        if "model_pnl_pct" in strict_df:
            strict_model = strict_df.loc[strict_df["model_ran"]].copy()
            if not strict_model.empty:
                strict_model["abs_model_pnl_error"] = (
                    strict_model["model_pnl_pct"] - strict_model["live_pnl_pct"]
                ).abs()
                out["strict_model_mean_abs_pnl_error"] = float(
                    strict_model["abs_model_pnl_error"].mean()
                )
                strict_model["abs_trade_count_error"] = (
                    strict_model["model_total_trades"] - strict_model["manual_trade_strategy_count"]
                ).abs()
                out["strict_manual_vs_model_mean_abs_trade_row_error"] = float(
                    strict_model["abs_trade_count_error"].mean()
                )
                if {
                    "manual_sub_120s_gap_count",
                    "manual_trade_strategy_count",
                    "model_sub_120s_gap_count",
                    "model_trade_count_from_tape",
                }.issubset(strict_model.columns):
                    manual_sub_gap_share = (
                        strict_model["manual_sub_120s_gap_count"].astype(float)
                        / strict_model["manual_trade_strategy_count"].astype(float).replace(0.0, np.nan)
                    )
                    model_sub_gap_share = (
                        strict_model["model_sub_120s_gap_count"].astype(float)
                        / strict_model["model_trade_count_from_tape"].astype(float).replace(0.0, np.nan)
                    )
                    out["strict_manual_sub_120s_gap_share_mean"] = float(manual_sub_gap_share.mean())
                    out["strict_model_sub_120s_gap_share_mean"] = float(model_sub_gap_share.mean())
                if {
                    "manual_max_same_minute_trades",
                    "model_max_same_minute_trades",
                }.issubset(strict_model.columns):
                    out["strict_manual_max_same_minute_trades_mean"] = float(
                        strict_model["manual_max_same_minute_trades"].astype(float).mean()
                    )
                    out["strict_model_max_same_minute_trades_mean"] = float(
                        strict_model["model_max_same_minute_trades"].astype(float).mean()
                    )
        out.update(
            {
                "strict_manual_mean_trade_rows": float(strict_df["manual_trade_strategy_count"].mean()),
                "strict_manual_mean_maker_ratio": float(
                    (
                        strict_df["manual_maker_count"]
                        / (strict_df["manual_maker_count"] + strict_df["manual_taker_count"]).replace(0, np.nan)
                    ).mean()
                ),
                "strict_manual_mean_trade_coverage_ratio": float(
                    strict_df["manual_trade_coverage_ratio"].mean()
                ),
            }
        )

    any_mask = (
        cast(pd.Series, df["manual_any_reconstructable"])
        .astype("boolean")
        .fillna(False)
        .astype(bool)
        if "manual_any_reconstructable" in df.columns
        else pd.Series(False, index=df.index, dtype=bool)
    )
    any_df = df.loc[any_mask].copy()
    if not any_df.empty:
        any_model = any_df.loc[any_df["model_ran"]].copy()
        if not any_model.empty:
            any_model["abs_model_pnl_error"] = (
                any_model["model_pnl_pct"] - any_model["live_pnl_pct"]
            ).abs()
            any_model["manual_any_trade_coverage_ratio"] = any_model["manual_any_trade_count"] / any_model[
                "live_fill_count"
            ].replace(0, np.nan)
            out["manual_any_model_mean_abs_pnl_error"] = float(
                any_model["abs_model_pnl_error"].mean()
            )
            out["manual_any_mean_trade_rows"] = float(any_model["manual_any_trade_count"].mean())
            out["manual_any_mean_trade_coverage_ratio"] = float(
                any_model["manual_any_trade_coverage_ratio"].mean()
            )
            if {
                "manual_any_sub_120s_gap_count",
                "manual_any_trade_count",
                "model_sub_120s_gap_count",
                "model_trade_count_from_tape",
            }.issubset(any_model.columns):
                manual_any_sub_gap_share = (
                    any_model["manual_any_sub_120s_gap_count"].astype(float)
                    / any_model["manual_any_trade_count"].astype(float).replace(0.0, np.nan)
                )
                model_sub_gap_share = (
                    any_model["model_sub_120s_gap_count"].astype(float)
                    / any_model["model_trade_count_from_tape"].astype(float).replace(0.0, np.nan)
                )
                out["manual_any_sub_120s_gap_share_mean"] = float(manual_any_sub_gap_share.mean())
                out["manual_any_model_sub_120s_gap_share_mean"] = float(model_sub_gap_share.mean())
            if {
                "manual_any_max_same_minute_trades",
                "model_max_same_minute_trades",
            }.issubset(any_model.columns):
                out["manual_any_max_same_minute_trades_mean"] = float(
                    any_model["manual_any_max_same_minute_trades"].astype(float).mean()
                )
                out["manual_any_model_max_same_minute_trades_mean"] = float(
                    any_model["model_max_same_minute_trades"].astype(float).mean()
                )

    return out


def _diagnostic_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.override_fill_mode != "current":
        overrides["fill_mode"] = args.override_fill_mode
    if args.override_close_fee_mode != "current":
        overrides["close_fee_mode"] = args.override_close_fee_mode
    if args.override_funding_mode != "current":
        overrides["funding_mode"] = args.override_funding_mode
    if args.override_maintenance_margin_rate is not None:
        overrides["maintenance_margin_rate"] = float(args.override_maintenance_margin_rate)
    if args.bar_path != "current" and "fill_mode" not in overrides:
        overrides["fill_mode"] = "close"
    if args.override_global_cooldown_bars is not None:
        overrides["global_cooldown_bars"] = int(args.override_global_cooldown_bars)
    if args.override_order_delay_bars is not None:
        overrides["order_delay_bars"] = int(args.override_order_delay_bars)
    return overrides


def _ablation_needs_public_inputs(ablation: str) -> bool:
    return ablation in {
        "exchange_filters_only",
        "funding_series_only",
        "mark_valuation_only",
        "combined_public",
    }


def _ablation_uses_geometry_seed(ablation: str) -> bool:
    return ablation in {"geometry_seed_only", "combined_public"}


def _ablation_overrides(
    ablation: str,
    public_payload: dict[str, Any],
    base_overrides: dict[str, Any],
) -> dict[str, Any]:
    overrides = dict(base_overrides)
    if ablation in {"exchange_filters_only", "combined_public"}:
        tick_size = public_payload.get("tick_size")
        step_size = public_payload.get("step_size")
        if _finite(tick_size) and float(tick_size) > 0:
            overrides["tick_size"] = float(tick_size)
        if _finite(step_size) and float(step_size) > 0:
            overrides["step_size"] = float(step_size)
    if ablation in {"funding_series_only", "combined_public"}:
        funding_series = public_payload.get("funding_rate_series")
        if isinstance(funding_series, list) and funding_series:
            overrides["funding_rate_series"] = [float(item) for item in funding_series]
    if ablation in {"mark_valuation_only", "combined_public"}:
        overrides["valuation_price_source"] = "mark"
    return overrides


def _symbol_set(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _strategy_id_set(value: str | None) -> set[str]:
    if value is None:
        return set()
    strategy_ids = {
        _strategy_id_text(item)
        for item in value.split(",")
        if str(item).strip()
    }
    invalid = sorted(item for item in strategy_ids if re.fullmatch(r"\d+", item) is None)
    if invalid:
        raise ValueError(f"Invalid strategy IDs: {invalid}")
    return strategy_ids


def _strategy_id_integrity_report(df: pd.DataFrame) -> dict[str, Any]:
    """Classify workbook strategy-ID completeness before any row deduplication."""
    if "strategy_id_text" not in df.columns:
        return {
            "rows": int(len(df)),
            "nonblank_strategy_id_rows": 0,
            "blank_strategy_id_rows": int(len(df)),
            "unique_strategy_ids": 0,
            "duplicate_strategy_id_rows": 0,
            "duplicate_strategy_id_counts": {},
        }
    strategy_ids = (
        cast(pd.Series, df["strategy_id_text"])
        .fillna("")
        .astype(str)
        .str.strip()
    )
    nonblank = strategy_ids.ne("")
    counts = strategy_ids.loc[nonblank].value_counts(dropna=False)
    duplicate_counts = {
        str(strategy_id): int(count)
        for strategy_id, count in counts.items()
        if int(count) > 1
    }
    duplicate_rows = int(
        strategy_ids.loc[nonblank].duplicated(keep=False).sum()
    )
    return {
        "rows": int(len(df)),
        "nonblank_strategy_id_rows": int(nonblank.sum()),
        "blank_strategy_id_rows": int((~nonblank).sum()),
        "unique_strategy_ids": int(counts.size),
        "duplicate_strategy_id_rows": duplicate_rows,
        "duplicate_strategy_id_counts": dict(sorted(duplicate_counts.items())),
    }


def _select_exact_strategy_id_cohort(
    workbook_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    requested_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a complete one-row-per-ID cohort or fail before replay.

    ``select_scope`` intentionally deduplicates ordinary audits. Exact-ID
    cohorts need a stronger contract: duplicates in the raw workbook, IDs that
    do not exist, and IDs removed by eligibility/mode/symbol filters must stay
    visible and must never be silently converted into a smaller cohort.
    """
    if not requested_ids:
        return selected_df, {
            "requested_strategy_ids": 0,
            "matched_strategy_ids": 0,
            "status": "not_requested",
        }

    workbook_ids = (
        cast(pd.Series, workbook_df["strategy_id_text"])
        .fillna("")
        .astype(str)
        .str.strip()
    )
    workbook_counts = workbook_ids.value_counts(dropna=False)
    missing_ids = sorted(
        strategy_id
        for strategy_id in requested_ids
        if int(workbook_counts.get(strategy_id, 0)) == 0
    )
    duplicate_counts = {
        strategy_id: int(workbook_counts.get(strategy_id, 0))
        for strategy_id in sorted(requested_ids)
        if int(workbook_counts.get(strategy_id, 0)) > 1
    }
    if missing_ids or duplicate_counts:
        raise ValueError(
            "Exact strategy-ID cohort failed raw-workbook identity checks: "
            f"missing_ids={missing_ids}, duplicate_id_counts={duplicate_counts}"
        )

    matched = selected_df.loc[
        cast(pd.Series, selected_df["strategy_id_text"])
        .fillna("")
        .astype(str)
        .isin(requested_ids)
    ].copy()
    matched_ids = set(
        cast(pd.Series, matched["strategy_id_text"])
        .fillna("")
        .astype(str)
        .tolist()
    )
    filtered_out_ids = sorted(requested_ids - matched_ids)
    if filtered_out_ids:
        raise ValueError(
            "Exact strategy-ID cohort became incomplete after eligibility, mode, "
            f"or symbol filters: filtered_out_ids={filtered_out_ids}"
        )
    if len(matched) != len(requested_ids):
        raise ValueError(
            "Exact strategy-ID cohort is not one-row-per-ID after selection: "
            f"requested={len(requested_ids)}, matched_rows={len(matched)}"
        )
    return matched, {
        "requested_strategy_ids": int(len(requested_ids)),
        "matched_strategy_ids": int(len(matched_ids)),
        "status": "complete_unique",
    }


def _timestamp_ms(value: Any) -> int:
    ts = pd.Timestamp(value)
    if bool(pd.isna(ts)) or not isinstance(ts, pd.Timestamp):
        raise ValueError(f"Invalid timestamp for public market evidence: {value!r}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _find_symbol_exchange_info(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    symbols = exchange_info.get("symbols")
    if isinstance(symbols, list):
        for item in symbols:
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol.upper():
                return item
    if str(exchange_info.get("symbol", "")).upper() == symbol.upper():
        return exchange_info
    return None


def _exchange_filter_diagnostics(exchange_info: dict[str, Any], symbol: str) -> dict[str, Any]:
    return extract_exchange_filter_settings(exchange_info, symbol)


def _public_market_defaults(source: str = "not_requested") -> dict[str, Any]:
    return {
        "exchange_filter_source": source,
        "tick_size_source": source,
        "step_size_source": source,
        "min_notional_source": source,
        "tick_size_value": None,
        "step_size_value": None,
        "min_notional_value": None,
        "exchange_filter_validation_status": source,
        "exchange_filter_rejection_reason": "",
        "mark_price_source": source,
        "mark_price_rows": 0,
        "funding_series_source": source,
        "funding_series_status": source,
        "funding_series_rows": 0,
        "historical_depth_source": "missing",
    }


def _klines_from_raw(raw_klines: list[Any]) -> pd.DataFrame:
    if not raw_klines:
        return pd.DataFrame()
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw_klines, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = cast(pd.DataFrame, df[["timestamp", "open", "high", "low", "close", "volume"]].copy())
    valid = cast(pd.Series, out["timestamp"]).notna() & cast(pd.Series, out["close"]).notna()
    return cast(pd.DataFrame, out.loc[valid].reset_index(drop=True))


async def _fetch_exchange_info_once() -> dict[str, Any] | None:
    client = BinanceClient()
    try:
        return await client.get_exchange_info()
    except Exception:
        return None
    finally:
        await client.close()


async def _fetch_public_market_model_inputs(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_info: dict[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    diagnostics = _public_market_defaults("missing")
    if exchange_info is not None:
        diagnostics.update(extract_exchange_filter_settings(exchange_info, symbol))

    start_ms = _timestamp_ms(start)
    end_ms = _timestamp_ms(end)
    minutes = max(1, int(math.ceil((end_ms - start_ms) / 60_000)))
    limit = max(1, min(1500, minutes + 2))

    client = BinanceClient()
    mark_df = pd.DataFrame()
    try:
        try:
            raw_mark_klines = await client.get_mark_price_klines(
                symbol,
                "1m",
                limit=limit,
                start_time=start_ms,
                end_time=end_ms,
                include_current=False,
            )
            mark_df = _klines_from_raw(raw_mark_klines or [])
            diagnostics["mark_price_source"] = (
                "binance_mark_price_klines" if not mark_df.empty else "missing"
            )
            diagnostics["mark_price_rows"] = int(len(mark_df))
        except Exception as exc:
            diagnostics["mark_price_source"] = "error"
            diagnostics["mark_price_error"] = repr(exc)

        try:
            funding_rows = await client.get_funding_rate(
                symbol,
                limit=1000,
                start_time=start_ms,
                end_time=end_ms,
            )
            funding_series, funding_status, funding_count = funding_rate_series_from_rows(
                funding_rows or []
            )
            if funding_series is not None:
                diagnostics["funding_rate_series"] = funding_series
            diagnostics["funding_series_source"] = "binance_funding_rate"
            diagnostics["funding_series_status"] = funding_status
            diagnostics["funding_series_rows"] = funding_count
        except Exception as exc:
            diagnostics["funding_series_source"] = "error"
            diagnostics["funding_series_status"] = "missing"
            diagnostics["funding_series_error"] = repr(exc)
    finally:
        await client.close()

    diagnostics["historical_depth_source"] = "missing"
    return diagnostics, mark_df


async def _fetch_public_market_evidence(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    client = BinanceClient()
    evidence: dict[int, dict[str, Any]] = {}
    try:
        symbols = sorted({str(row.get("symbol", "")).upper() for row in rows if str(row.get("symbol", "")).strip()})
        exchange_info = await client.get_exchange_info()
        exchange_by_symbol = {
            symbol: _exchange_filter_diagnostics(exchange_info, symbol)
            for symbol in symbols
        }
        for idx, row in enumerate(rows):
            symbol = str(row.get("symbol", "")).upper()
            diagnostics = _public_market_defaults("missing")
            if symbol in exchange_by_symbol:
                diagnostics.update(exchange_by_symbol[symbol])
            try:
                start_ms = _timestamp_ms(row.get("validation_start_time_utc"))
                end_ms = _timestamp_ms(row.get("validation_end_time_utc"))
            except ValueError as exc:
                diagnostics["mark_price_source"] = "skipped"
                diagnostics["funding_series_source"] = "skipped"
                diagnostics["funding_series_status"] = "skipped"
                diagnostics["public_market_skip_reason"] = repr(exc)
                evidence[idx] = diagnostics
                continue
            minutes = max(1, int(math.ceil((end_ms - start_ms) / 60_000)))
            limit = max(1, min(1500, minutes + 2))
            try:
                mark_klines = await client.get_mark_price_klines(
                    symbol,
                    "1m",
                    limit=limit,
                    start_time=start_ms,
                    end_time=end_ms,
                    include_current=False,
                )
                diagnostics["mark_price_source"] = "binance_mark_price_klines" if mark_klines else "missing"
                diagnostics["mark_price_rows"] = int(len(mark_klines or []))
            except Exception as exc:
                diagnostics["mark_price_source"] = "error"
                diagnostics["mark_price_error"] = repr(exc)
            try:
                funding_rows = await client.get_funding_rate(
                    symbol,
                    limit=1000,
                    start_time=start_ms,
                    end_time=end_ms,
                )
                _funding_series, funding_status, funding_count = funding_rate_series_from_rows(
                    funding_rows or []
                )
                diagnostics["funding_series_source"] = "binance_funding_rate"
                diagnostics["funding_series_status"] = funding_status
                diagnostics["funding_series_rows"] = funding_count
            except Exception as exc:
                diagnostics["funding_series_source"] = "error"
                diagnostics["funding_series_status"] = "missing"
                diagnostics["funding_series_error"] = repr(exc)
            diagnostics["historical_depth_source"] = "missing"
            evidence[idx] = diagnostics
    finally:
        await client.close()
    return evidence


def _apply_public_market_evidence(rows: list[dict[str, Any]], fetch: bool) -> None:
    if not fetch:
        for row in rows:
            if "exchange_filter_source" not in row:
                row.update(_public_market_defaults("not_requested"))
        return
    fetched = asyncio.run(_fetch_public_market_evidence(rows))
    for idx, row in enumerate(rows):
        row.update(fetched.get(idx, _public_market_defaults("missing")))


def _validation_split_summary(
    rows: list[dict[str, Any]],
    split_mode: str,
    holdout_fraction: float,
    include_table: bool,
) -> dict[str, Any]:
    if split_mode == "none":
        for row in rows:
            row["validation_split"] = "none"
        return {"validation_split": "none"}
    if split_mode != "chronological":
        raise ValueError(f"Unsupported validation split: {split_mode!r}")
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be > 0 and < 1 for chronological split")
    if not rows:
        return {
            "validation_split": "chronological",
            "holdout_fraction": float(holdout_fraction),
            "calibration_rows": 0,
            "holdout_rows": 0,
            "split_table": [] if include_table else None,
        }

    indexed: list[tuple[pd.Timestamp, str, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        split_time = row.get("validation_split_time_utc")
        if not split_time:
            split_time = row.get("validation_start_time_utc")
        ts = pd.Timestamp(split_time)
        if pd.isna(ts):
            ts = pd.Timestamp.max.tz_localize("UTC")
        elif ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        indexed.append((ts, str(row.get("symbol", "")), idx, row))
    indexed.sort(key=lambda item: (item[0], item[1], item[2]))

    total = len(indexed)
    holdout_count = int(math.ceil(total * holdout_fraction))
    holdout_count = max(1, min(total, holdout_count))
    calibration_count = total - holdout_count
    holdout_start_idx = calibration_count

    for pos, (_, _, _, row) in enumerate(indexed):
        row["validation_split"] = "holdout" if pos >= holdout_start_idx else "calibration"
        row["validation_split_order"] = pos

    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "validation_split": "chronological",
        "holdout_fraction": float(holdout_fraction),
        "calibration_rows": int((df["validation_split"] == "calibration").sum()),
        "holdout_rows": int((df["validation_split"] == "holdout").sum()),
    }
    for split_name in ("calibration", "holdout"):
        split_df = df.loc[df["validation_split"] == split_name].copy()
        prefix = f"{split_name}_"
        if split_df.empty:
            out[prefix + "first_start"] = None
            out[prefix + "last_start"] = None
            out[prefix + "evidence_class_counts"] = {}
            out[prefix + "identifiers"] = []
            continue
        split_time_column = (
            "validation_split_time_utc"
            if "validation_split_time_utc" in split_df.columns
            else "validation_start_time_utc"
        )
        starts = pd.to_datetime(split_df[split_time_column], utc=True, errors="coerce")
        out[prefix + "first_start"] = str(starts.min())
        out[prefix + "last_start"] = str(starts.max())
        out[prefix + "evidence_class_counts"] = {
            str(key): int(value)
            for key, value in split_df["evidence_class"].value_counts(dropna=False).to_dict().items()
        } if "evidence_class" in split_df.columns else {}
        out[prefix + "identifiers"] = [
            {
                "symbol": str(item.get("symbol", "")),
                "strategy_id": str(item.get("strategy_id", "")),
                "validation_start_time_utc": str(item.get("validation_start_time_utc", "")),
                "validation_split_time_utc": str(
                    item.get("validation_split_time_utc", "")
                ),
                "evidence_class": str(item.get("evidence_class", "")),
            }
            for item in split_df.sort_values("validation_split_order").to_dict(orient="records")
        ]
        model_split = (
            split_df.loc[split_df["model_ran"] == True].copy()
            if "model_ran" in split_df.columns
            else pd.DataFrame()
        )
        out[prefix + "model_rows"] = int(len(model_split))
        if not model_split.empty and {
            "model_pnl_pct",
            "live_pnl_pct",
            "live_duration_hours",
        }.issubset(model_split.columns):
            model_pnl = model_split["model_pnl_pct"].astype(float)
            live_pnl = model_split["live_pnl_pct"].astype(float)
            abs_error = (model_pnl - live_pnl).abs()
            live_positive = live_pnl > 0.0
            model_positive = model_pnl > 0.0
            live_fast_winner = (live_pnl > 1.0) & (
                model_split["live_duration_hours"].astype(float) < 7.0
            )
            model_fast_winner = model_pnl > 1.0
            out[prefix + "model_mean_abs_pnl_error"] = float(abs_error.mean())
            out[prefix + "model_median_abs_pnl_error"] = float(abs_error.median())
            out[prefix + "model_pnl_sign_match_rate"] = float(
                (np.sign(model_pnl.to_numpy(dtype=float)) == np.sign(live_pnl.to_numpy(dtype=float))).mean()
            )
            out[prefix + "winner_recall_pnl_gt_0"] = (
                float(((live_positive) & (model_positive)).sum() / live_positive.sum())
                if int(live_positive.sum()) > 0
                else None
            )
            out[prefix + "non_winner_specificity_pnl_lte_0"] = (
                float(((~live_positive) & (~model_positive)).sum() / (~live_positive).sum())
                if int((~live_positive).sum()) > 0
                else None
            )
            out[prefix + "fast_winner_recall_pnl_gt_1_duration_lt_7"] = (
                float(((live_fast_winner) & (model_fast_winner)).sum() / live_fast_winner.sum())
                if int(live_fast_winner.sum()) > 0
                else None
            )

    if include_table:
        out["split_table"] = [
            {
                "split": str(item.get("validation_split", "")),
                "order": int(item.get("validation_split_order", -1)),
                "symbol": str(item.get("symbol", "")),
                "strategy_id": str(item.get("strategy_id", "")),
                "validation_start_time_utc": str(item.get("validation_start_time_utc", "")),
                "validation_split_time_utc": str(
                    item.get("validation_split_time_utc", "")
                ),
                "evidence_class": str(item.get("evidence_class", "")),
            }
            for item in df.sort_values("validation_split_order").to_dict(orient="records")
        ]
    return out


def _timestamp_coverage_preprobe(
    rows_df: pd.DataFrame,
    exports: ManualExports,
    *,
    holdout_fraction: float,
    duration_source: str,
    fixed_duration_hours: float,
) -> dict[str, Any]:
    if rows_df.empty:
        return {
            "workbook_rows": 0,
            "geometric_rows": 0,
            "exact_time_evidence_coverage_rows": 0,
            "exact_manual_order_history_coverage_rows": 0,
            "exact_live_telemetry_coverage_rows": 0,
            "modelable_evidence_matched_rows": 0,
            "diagnostic_fast_winner_contract": "terminal_pnl_pct_gt_1_and_duration_hours_lt_7",
            "active_meta_target_contract_evaluated": False,
            "fast_winner_rows": 0,
            "non_fast_non_winner_rows": 0,
            "chronological_holdout_rows": 0,
            "chronological_holdout_fast_winner_rows": 0,
            "chronological_holdout_non_fast_non_winner_rows": 0,
            "holdout_non_winner_specificity_step_pct": None,
            "time_evidence_class_counts": {},
        }

    decisions: list[TimestampDecision] = []
    rows_for_split: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        decision = _timestamp_decision(
            row,
            exports,
            timestamp_policy="evidence_matched",
            selected_time_policy="evidence_matched",
            duration_source=duration_source,
            fixed_duration_hours=fixed_duration_hours,
        )
        decisions.append(decision)
        live_pnl = float(row["pnl_pct"])
        live_duration = float(row["duration_hours"])
        rows_for_split.append(
            {
                "validation_start_time_utc": _timestamp_text(decision.selected_start_utc)
                or _timestamp_text(decision.stored_start_utc),
                "symbol": str(row["symbol"]),
                "fast_winner": bool(live_pnl > 1.0 and live_duration < 7.0),
                "winner": bool(live_pnl > 1.0),
            }
        )

    classes = [decision.time_evidence_class for decision in decisions]
    exact_coverage = sum(
        cls in {"exact_stored_match", "exact_local_offset_match"}
        for cls in classes
    )
    exact_manual_coverage = sum(
        decision.time_evidence_class
        in {"exact_stored_match", "exact_local_offset_match"}
        and decision.time_evidence_source == "manual_order_history"
        for decision in decisions
    )
    exact_telemetry_coverage = sum(
        decision.time_evidence_class
        in {"exact_stored_match", "exact_local_offset_match"}
        and decision.time_evidence_source == "live_telemetry_created_at_lima"
        for decision in decisions
    )
    fast_winners = sum(bool(item["fast_winner"]) for item in rows_for_split)
    non_fast = len(rows_for_split) - fast_winners

    holdout_count = 0
    holdout_fast = 0
    holdout_non_fast = 0
    if rows_for_split and 0.0 < holdout_fraction < 1.0:
        ordered = sorted(
            rows_for_split,
            key=lambda item: (
                pd.Timestamp(item["validation_start_time_utc"]),
                str(item["symbol"]),
            ),
        )
        holdout_count = max(1, min(len(ordered), int(math.ceil(len(ordered) * holdout_fraction))))
        holdout = ordered[-holdout_count:]
        holdout_fast = sum(bool(item["fast_winner"]) for item in holdout)
        holdout_non_fast = holdout_count - holdout_fast

    specificity_step = (
        float(100.0 / holdout_non_fast) if holdout_non_fast > 0 else None
    )
    class_counts = {
        str(key): int(value)
        for key, value in pd.Series(classes, dtype="object")
        .value_counts(dropna=False)
        .to_dict()
        .items()
    }
    return {
        "workbook_rows": int(len(rows_df)),
        "geometric_rows": int((rows_df["mode"].astype(str).str.lower() == "geometric").sum())
        if "mode" in rows_df.columns
        else 0,
        "exact_time_evidence_coverage_rows": int(exact_coverage),
        "exact_manual_order_history_coverage_rows": int(exact_manual_coverage),
        "exact_live_telemetry_coverage_rows": int(exact_telemetry_coverage),
        "modelable_evidence_matched_rows": int(exact_coverage),
        "diagnostic_fast_winner_contract": "terminal_pnl_pct_gt_1_and_duration_hours_lt_7",
        "active_meta_target_contract_evaluated": False,
        "fast_winner_rows": int(fast_winners),
        "non_fast_non_winner_rows": int(non_fast),
        "chronological_holdout_rows": int(holdout_count),
        "chronological_holdout_fast_winner_rows": int(holdout_fast),
        "chronological_holdout_non_fast_non_winner_rows": int(holdout_non_fast),
        "holdout_non_winner_specificity_step_pct": specificity_step,
        "time_evidence_class_counts": class_counts,
    }


def run_validation(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        validate_realism_output_path(args.realism_profile, output_dir)
    if bool(args.seed_from_manual_exports) and str(args.realism_profile) != "legacy":
        raise ValueError("--seed-from-manual-exports cannot be combined with a non-legacy realism profile")
    ablation = str(args.realism_ablation)
    if ablation != "profile" and str(args.realism_profile) != LEGACY_REALISM_PROFILE:
        raise ValueError("--realism-ablation must use --realism-profile legacy")
    public_profile = str(args.realism_profile) == CANDIDATE_TIME_PUBLIC_MARKET_PROFILE
    public_ablation = _ablation_needs_public_inputs(ablation)
    include_strategy_ids = _strategy_id_set(args.include_strategy_ids)
    if include_strategy_ids and str(args.scope) != "all":
        raise ValueError(
            "--include-strategy-ids requires --scope all so the requested exact "
            "cohort cannot be reduced by latest20 winner/symbol selection"
        )
    workbook = load_workbook_rows(_as_path(args.workbook))
    workbook_strategy_id_integrity = _strategy_id_integrity_report(workbook)
    rows_df = select_scope(workbook, args.scope)
    if args.mode_filter != "all":
        rows_df = rows_df.loc[rows_df["mode"].astype(str).str.lower() == args.mode_filter].copy()
    include_symbols = _symbol_set(args.include_symbols)
    exclude_symbols = _symbol_set(args.exclude_symbols)
    if include_symbols:
        rows_df = rows_df.loc[rows_df["symbol"].astype(str).str.upper().isin(include_symbols)].copy()
    if exclude_symbols:
        rows_df = rows_df.loc[~rows_df["symbol"].astype(str).str.upper().isin(exclude_symbols)].copy()
    rows_df, exact_strategy_id_integrity = _select_exact_strategy_id_cohort(
        workbook,
        rows_df,
        include_strategy_ids,
    )
    exports = load_manual_exports(
        _as_path(args.manual_exports),
        _as_path(args.live_telemetry_root),
    )
    cache_root = _as_path(args.kline_cache)
    physics_overrides = _diagnostic_overrides(args)
    timestamp_preprobe = _timestamp_coverage_preprobe(
        rows_df,
        exports,
        holdout_fraction=float(args.holdout_fraction),
        duration_source=str(args.duration_source),
        fixed_duration_hours=float(args.fixed_duration_hours),
    )
    public_exchange_info = (
        asyncio.run(_fetch_exchange_info_once())
        if public_profile or public_ablation
        else None
    )

    rows: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        timestamp_decisions = _timestamp_decisions_for_policy(
            row,
            exports,
            timestamp_policy=str(args.timestamp_policy),
            duration_source=str(args.duration_source),
            fixed_duration_hours=float(args.fixed_duration_hours),
        )
        for decision in timestamp_decisions:
            validation_start = decision.selected_start_utc
            end = decision.selected_end_utc
            base = {
                "row_id": int(row["_row_id"]),
                "strategy_id": row.get("strategy_id_text", ""),
                "symbol": str(row["symbol"]),
                "mode": str(row["mode"]),
                "start_time_utc": str(row["start_time_utc"]),
                "timestamp_policy_requested": str(args.timestamp_policy),
                "validation_start_time_utc": _timestamp_text(validation_start),
                "validation_split_time_utc": _timestamp_text(
                    validation_start
                    if validation_start is not None
                    else decision.stored_start_utc
                ),
                "order_history_start_time_utc": _timestamp_text(decision.manual_order_start_utc),
                "order_history_last_update_time_utc": _timestamp_text(
                    decision.manual_order_last_update_utc
                ),
                "order_history_start_used": bool(
                    _timestamps_equal_at_second(
                        validation_start, decision.manual_order_start_utc
                    )
                ),
                "validation_end_time_utc": _timestamp_text(end),
                "duration_source": args.duration_source,
                "mode_filter": args.mode_filter,
                "live_duration_hours": float(row["duration_hours"]),
                "live_pnl_pct": float(row["pnl_pct"]),
                "live_total_profit_usdt": float(row["total_profit_usdt"]),
                "live_realized_pnl_usdt": float(row.get("realized_pnl_usdt", math.nan)),
                "live_unrealized_pnl_usdt": float(row.get("unrealized_pnl_usdt", math.nan)),
                "live_funding_fee_usdt": float(row.get("funding_fee_usdt", math.nan)),
                "live_commission_usdt": float(row.get("commission_usdt", math.nan)),
                "live_mae": float(row.get("mae", math.nan)),
                "live_mfe": float(row.get("mfe", math.nan)),
                "live_mae_pct_initial": float(row.get("mae_pct_initial", math.nan)),
                "live_mfe_pct_initial": float(row.get("mfe_pct_initial", math.nan)),
                "live_total_trades": float(row["total_trades"]),
                "live_maker_count": float(row.get("maker_count", math.nan)),
                "live_taker_count": float(row.get("taker_count", math.nan)),
                "live_fill_count": _live_fill_count(row),
                "live_invested_margin_usdt": float(row["invested_margin_usdt"]),
            }
            base.update(_timestamp_decision_fields(decision))
            base["model_ran"] = False

            if (
                not decision.timestamp_modelable
                or validation_start is None
                or end is None
                or pd.isna(validation_start)
                or pd.isna(end)
            ):
                base["klines_available"] = False
                base["klines_rows"] = 0
                base["kline_cache_status"] = "not_attempted_timestamp_rejected"
                base["kline_cache_error"] = ""
                base.update(_public_market_defaults("not_requested"))
                rows.append(base)
                continue

            manual = _manual_metrics(row, exports, validation_start, end)
            base.update(manual)

            klines, kline_error = _load_cached_klines_resilient(
                cache_root,
                str(row["symbol"]),
                validation_start,
                end,
            )
            base["kline_cache_status"] = "error" if kline_error else (
                "available" if not klines.empty else "missing"
            )
            base["kline_cache_error"] = kline_error or ""
            klines = _expand_ohlc_path(klines, args.bar_path)
            base["klines_available"] = not klines.empty
            base["klines_rows"] = int(len(klines))

            public_payload: dict[str, Any] = {}
            if public_profile or public_ablation:
                public_payload, mark_klines = asyncio.run(
                    _fetch_public_market_model_inputs(
                        str(row["symbol"]),
                        validation_start,
                        end,
                        public_exchange_info,
                    )
                )
                base.update(public_payload)
                if (
                    not klines.empty
                    and not mark_klines.empty
                    and (
                        public_profile
                        or ablation in {"mark_valuation_only", "combined_public"}
                    )
                ):
                    klines = attach_mark_close(klines, mark_klines)

            seed_state = None
            if args.seed_from_manual_exports:
                seed_state = load_seed_from_order_history_exports(
                    _as_path(args.manual_exports),
                    str(row["symbol"]),
                    validation_start.to_pydatetime(),
                    strategy_id=row.get("strategy_id_text", ""),
                )
            elif (
                str(args.realism_profile) == CANDIDATE_TIME_GEOMETRIC_PROFILE
                or _ablation_uses_geometry_seed(ablation)
            ):
                seed_state = _candidate_time_geometry_seed(
                    row,
                    klines,
                    validation_start,
                    _ablation_overrides(ablation, public_payload, physics_overrides),
                )
            base.update(
                {
                    "realism_profile": str(args.realism_profile),
                    "realism_ablation": ablation,
                    "seed_from_manual_exports": bool(args.seed_from_manual_exports),
                    "seed_state_source": seed_state.source if seed_state is not None else "none",
                    "seed_evidence_class": seed_state.evidence_class if seed_state is not None else "missing",
                    "seed_buy_levels": len(seed_state.open_buy_levels) if seed_state is not None else 0,
                    "seed_sell_levels": len(seed_state.open_sell_levels) if seed_state is not None else 0,
                    "seed_active_level_count": (
                        len(seed_state.open_buy_levels) + len(seed_state.open_sell_levels)
                        if seed_state is not None
                        else 0
                    ),
                    "seed_qty_per_order": seed_state.qty_per_order if seed_state is not None else None,
                    "seed_qty_source": seed_state.qty_source if seed_state is not None else "none",
                }
            )
            base["evidence_class"] = _evidence_class(base, seed_state)

            if not klines.empty:
                try:
                    if public_profile and ablation == "profile":
                        candidate_payload = {
                            "symbol": str(row["symbol"]),
                            "candidate_id": f"{row['symbol']}_{validation_start.strftime('%Y%m%d_%H%M%S')}",
                            "grid_lower": float(row["price_range_low"]),
                            "grid_upper": float(row["price_range_high"]),
                            "num_grids": int(row["grids_count"]),
                            "mode": str(row["mode"]),
                        }
                        candidate_payload.update(public_payload)
                        model = run_single_backtest(
                            candidate_row=candidate_payload,
                            klines_df=klines,
                            capital=float(row["invested_margin_usdt"]),
                            leverage=int(row["leverage"]),
                            realism_profile=str(args.realism_profile),
                        )
                        model_trade_metrics = {
                            "model_trade_count_from_tape": int(model.get("total_trades", 0) or 0)
                        }
                    else:
                        model_physics_overrides = _ablation_overrides(
                            ablation,
                            public_payload,
                            physics_overrides,
                        )
                        model_profile = (
                            str(args.realism_profile)
                            if ablation == "profile"
                            else LEGACY_REALISM_PROFILE
                        )
                        model_seed_state = (
                            None
                            if (
                                ablation == "profile"
                                and str(args.realism_profile) == CANDIDATE_TIME_GEOMETRIC_PROFILE
                            )
                            else seed_state
                        )
                        model = _run_model(
                            row,
                            klines,
                            model_physics_overrides,
                            model_seed_state,
                            model_profile,
                        )
                        model_trade_metrics = _run_model_trade_metrics(
                            row,
                            klines,
                            model_physics_overrides,
                            seed_state,
                        )
                    base.update(
                        {
                            "model_ran": True,
                            "model_pnl_pct": float(model.get("net_pnl_pct", math.nan)),
                            "model_total_trades": float(model.get("total_trades", math.nan)),
                            "model_max_dd_pct": float(model.get("max_drawdown_pct", math.nan)),
                            "model_mae_pct_initial": float(model.get("mae_pct_initial", math.nan)),
                            "model_mfe_pct_initial": float(model.get("mfe_pct_initial", math.nan)),
                            "model_capital_used_usdt": float(model.get("capital_used", math.nan)),
                            "model_fees_paid_usdt": float(model.get("fees_paid", math.nan)),
                            "model_funding_fees_usdt": float(model.get("funding_fees", math.nan)),
                            "model_is_authoritative": bool(model.get("is_authoritative", False)),
                            "model_fill_mode": model.get("fill_mode"),
                            "model_close_fee_mode": model.get("close_fee_mode"),
                            "model_global_cooldown_bars": model.get("global_cooldown_bars"),
                            "model_position_size": model.get("position_size"),
                            "model_position_size_source": model.get("position_size_source"),
                            "model_seed_restrict_active_ladder": model.get("seed_restrict_active_ladder"),
                            "model_seeded_active_level_count": model.get("seeded_active_level_count"),
                            "model_seed_evidence_class": model.get("seed_evidence_class"),
                            "model_fill_price_source": model.get("fill_price_source"),
                            "model_valuation_price_source": model.get("valuation_price_source"),
                            "model_exchange_filter_validation_status": model.get("exchange_filter_validation_status"),
                            "model_funding_series_status": model.get("funding_series_status"),
                        }
                    )
                    base.update(model_trade_metrics)
                except Exception as exc:
                    base["model_error"] = repr(exc)

            for key, value in list(base.items()):
                if isinstance(value, float) and not _finite(value):
                    base[key] = None
            rows.append(base)

    if not public_profile:
        _apply_public_market_evidence(rows, bool(args.fetch_public_market_evidence))
    split_summary = _validation_split_summary(
        rows,
        str(args.validation_split),
        float(args.holdout_fraction),
        bool(args.print_split_table),
    )
    summary = summarize(rows, args.scope)
    summary["mode_filter"] = args.mode_filter
    summary["include_symbols"] = sorted(include_symbols)
    summary["exclude_symbols"] = sorted(exclude_symbols)
    summary["include_strategy_ids"] = sorted(include_strategy_ids)
    summary["workbook_strategy_id_integrity"] = workbook_strategy_id_integrity
    summary["exact_strategy_id_integrity"] = exact_strategy_id_integrity
    summary["bar_path"] = args.bar_path
    summary["realism_profile"] = str(args.realism_profile)
    summary["realism_ablation"] = ablation
    summary["public_market_evidence_requested"] = bool(
        args.fetch_public_market_evidence or public_profile or public_ablation
    )
    if rows:
        rows_df_out = pd.DataFrame(rows)
        for col in (
            "exchange_filter_source",
            "tick_size_source",
            "step_size_source",
            "min_notional_source",
            "mark_price_source",
            "funding_series_source",
            "historical_depth_source",
            "timestamp_policy_requested",
            "selected_time_policy",
            "time_evidence_class",
            "time_evidence_source",
            "time_rejection_reason",
            "kline_cache_status",
        ):
            if col in rows_df_out.columns:
                summary[f"{col}_counts"] = {
                    str(key): int(value)
                    for key, value in rows_df_out[col].value_counts(dropna=False).to_dict().items()
                }
        if {"model_ran", "time_evidence_class"}.issubset(rows_df_out.columns):
            model_rows_out = rows_df_out.loc[rows_df_out["model_ran"] == True].copy()
            summary["model_time_evidence_class_counts"] = {
                str(key): int(value)
                for key, value in model_rows_out["time_evidence_class"]
                .value_counts(dropna=False)
                .to_dict()
                .items()
            }
        if "timestamp_modelable" in rows_df_out.columns:
            summary["timestamp_modelable_rows"] = int(rows_df_out["timestamp_modelable"].sum())
            summary["timestamp_rejected_rows"] = int((~rows_df_out["timestamp_modelable"]).sum())
    summary["timestamp_preprobe"] = timestamp_preprobe
    summary["diagnostic_overrides"] = physics_overrides
    summary.update(split_summary)
    summary["mode_values_tested"] = sorted(
        pd.DataFrame(rows)["mode"].astype(str).str.lower().unique().tolist()
    ) if rows else []
    return rows, summary


def _live_fill_count(row: pd.Series) -> float:
    maker = row.get("maker_count", math.nan)
    taker = row.get("taker_count", math.nan)
    if _finite(maker) and _finite(taker):
        fill_count = float(maker) + float(taker)
        if fill_count > 0:
            return fill_count
    return float(row["total_trades"])


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "reconciliation_rows.csv", index=False)
    (output_dir / "reconciliation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default="data/new_expired_bots.xlsx")
    parser.add_argument("--manual-exports", default="data/manual_exports")
    parser.add_argument(
        "--live-telemetry-root",
        default="Live",
        help=(
            "Policy-compliant Live/YYYY-MM-DD telemetry root used only for "
            "exact strategy creation-time validation."
        ),
    )
    parser.add_argument("--kline-cache", default="data/cache/klines/futures_um")
    parser.add_argument("--scope", choices=("latest20", "all"), default="latest20")
    parser.add_argument("--mode-filter", choices=("all", "geometric", "arithmetic"), default="all")
    parser.add_argument("--include-symbols", default=None)
    parser.add_argument("--exclude-symbols", default=None)
    parser.add_argument(
        "--include-strategy-ids",
        default=None,
        help=(
            "Comma-separated exact strategy IDs. Use this for immutable cohort "
            "audits so older bots sharing a symbol cannot enter the replay."
        ),
    )
    parser.add_argument("--duration-source", choices=("workbook", "fixed"), default="workbook")
    parser.add_argument("--fixed-duration-hours", type=float, default=6.0)
    parser.add_argument(
        "--timestamp-policy",
        choices=TIMESTAMP_POLICIES,
        default="stored_utc",
        help=(
            "Select the UTC validation window: stored workbook UTC, explicit "
            "local UTC-5 to Binance UTC diagnostic shift, exact manual-evidence "
            "match, or non-promotable dual diagnostic."
        ),
    )
    parser.add_argument("--bar-path", choices=("current", "ohlc", "olhc"), default="current")
    parser.add_argument(
        "--realism-profile",
        choices=REALISM_PROFILES,
        default=LEGACY_REALISM_PROFILE,
    )
    parser.add_argument(
        "--realism-ablation",
        choices=REALISM_ABLATIONS,
        default="profile",
        help=(
            "Diagnostic-only component ablation. 'profile' preserves the "
            "selected realism profile; other choices isolate public-market "
            "or geometry-seed components without changing pipeline defaults."
        ),
    )
    parser.add_argument("--override-fill-mode", choices=("current", "close", "wick"), default="current")
    parser.add_argument("--override-close-fee-mode", choices=("current", "maker", "taker"), default="current")
    parser.add_argument("--override-funding-mode", choices=("current", "continuous", "snapshot"), default="current")
    parser.add_argument("--override-maintenance-margin-rate", type=float, default=None)
    parser.add_argument("--override-global-cooldown-bars", type=int, default=None)
    parser.add_argument("--override-order-delay-bars", type=int, default=None)
    parser.add_argument("--validation-split", choices=("none", "chronological"), default="none")
    parser.add_argument("--holdout-fraction", type=float, default=0.40)
    parser.add_argument("--print-split-table", action="store_true")
    parser.add_argument("--fetch-public-market-evidence", action="store_true")
    parser.add_argument(
        "--seed-from-manual-exports",
        action="store_true",
        help="Seed each modelable row from exact Order history exports when symbol/strategy/time evidence exists.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows, summary = run_validation(args)
    if args.output_dir:
        _write_outputs(_as_path(args.output_dir), rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
