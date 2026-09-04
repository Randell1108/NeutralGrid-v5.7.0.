"""Build oracle-compatible forward outcomes for depth-shadow candidate rows.

This is an audit helper for the depth-aware deployable-winner loop. It uses the
existing backtest engine for rows with complete grid geometry and leaves missing
geometry rows explicitly unlabeled instead of fabricating outcomes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.api.binance_client import BinanceClient  # noqa: E402
from neutralgrid.backtest.candidate_pipeline import (  # noqa: E402
    _parse_scan_timestamp,
    fetch_historical_klines,
    run_single_backtest,
)
from neutralgrid.backtest.realism_governance import (  # noqa: E402
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    REALISM_PROFILES,
    validate_realism_output_path,
)


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def _read_candidates(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported candidate input format: {path}")


def _is_geometry_valid(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    lower = _float_or_none(row.get("grid_lower"))
    upper = _float_or_none(row.get("grid_upper"))
    grids = _int_or_none(row.get("num_grids"))
    if lower is None:
        return False, "missing_grid_lower"
    if upper is None:
        return False, "missing_grid_upper"
    if grids is None:
        return False, "missing_num_grids"
    if lower <= 0 or upper <= lower:
        return False, "invalid_grid_bounds"
    if grids < 2:
        return False, "invalid_num_grids"
    return True, None


def _candidate_start(row: Mapping[str, Any]) -> datetime | None:
    for col in ("candidate_available_ts_utc", "backtest_start_ts_utc", "scan_time_utc", "start_time_utc"):
        raw = row.get(col)
        if raw is None or pd.isna(raw):
            continue
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if not pd.isna(parsed):
            return cast(pd.Timestamp, parsed).to_pydatetime()
    candidate_id = str(row.get("candidate_id", ""))
    return _parse_scan_timestamp(candidate_id)


def _tail_pnl_pct(result: Mapping[str, Any]) -> float | None:
    for col in ("max_drawdown_pct", "mae_pct_initial", "mae_pct", "tail_pnl_pct", "min_pnl_pct"):
        value = _float_or_none(result.get(col))
        if value is None:
            continue
        return -abs(value) if col.startswith(("max_drawdown", "mae")) and value > 0 else value
    return None


def _outcome_row(
    row: Mapping[str, Any],
    *,
    status: str,
    reason: str | None = None,
    result: Mapping[str, Any] | None = None,
    klines_rows: int | None = None,
) -> dict[str, Any]:
    result = result or {}
    pnl = _float_or_none(result.get("net_pnl_pct"))
    time_to_target = _float_or_none(result.get("time_to_target_hours"))
    target_hit = bool(time_to_target is not None and time_to_target <= 7.0)
    tail = _tail_pnl_pct(result)
    return {
        "candidate_id": row.get("candidate_id"),
        "symbol": row.get("symbol"),
        "outcome_status": status,
        "outcome_reason": reason,
        "pnl_pct": pnl,
        "net_pnl_pct": pnl,
        "time_to_target_hours": time_to_target,
        "target_hit": target_hit if status == "backtested" else None,
        "tail_pnl_pct": tail,
        "klines_rows": klines_rows,
        "price_start": result.get("price_start"),
        "price_end": result.get("price_end"),
        "backtest_start_ts_utc": result.get("backtest_start_ts_utc"),
        "barrier_touched": result.get("barrier_touched"),
        "grid_lower": row.get("grid_lower"),
        "grid_upper": row.get("grid_upper"),
        "num_grids": row.get("num_grids"),
    }


def _capital_fraction(row: pd.Series) -> float | None:
    capital_fraction = _float_or_none(row.get("capital_fraction"))
    if capital_fraction is None:
        capital_fraction = _float_or_none(row.get("ps_fraction"))
    if capital_fraction is None:
        deploy_margin = _float_or_none(row.get("deploy_margin_usdt"))
        capital_base = _float_or_none(row.get("capital_base_usdt"))
        if deploy_margin is not None and capital_base is not None and capital_base > 0:
            capital_fraction = deploy_margin / capital_base
    return capital_fraction


def _candidate_payload(row: pd.Series, start: datetime, *, capital_fraction: float) -> dict[str, Any]:
    funding_rate = _float_or_none(row.get("funding_rate")) or 0.0001
    return {
        "candidate_id": row.get("candidate_id"),
        "symbol": row.get("symbol"),
        "grid_lower": row.get("grid_lower"),
        "grid_upper": row.get("grid_upper"),
        "num_grids": row.get("num_grids"),
        "mode": row.get("mode"),
        "funding_rate": funding_rate,
        "capital_fraction": capital_fraction,
        "candidate_available_ts_utc": start.isoformat(),
        "candidate_available_source": "candidate_id_or_scan_timestamp",
    }


async def build_outcomes(args: argparse.Namespace) -> int:
    validate_realism_output_path(args.realism_profile, args.output_dir)
    candidates = _read_candidates(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = BinanceClient()
    rows: list[dict[str, Any]] = []
    try:
        for _, raw_row in candidates.iterrows():
            row = cast(pd.Series, raw_row)
            mapping = cast(Mapping[str, Any], row)
            geometry_ok, geometry_reason = _is_geometry_valid(mapping)
            start = _candidate_start(mapping)
            if not geometry_ok:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason=geometry_reason))
                continue
            if start is None:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason="missing_start_timestamp"))
                continue
            capital_fraction = _capital_fraction(row)
            if capital_fraction is None:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason="missing_capital_fraction"))
                continue
            if capital_fraction <= 0:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason="non_positive_capital_fraction"))
                continue
            try:
                klines = await fetch_historical_klines(client, str(row["symbol"]), start, hours=int(args.hours))
            except Exception as exc:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason=f"kline_fetch_failed:{type(exc).__name__}"))
                continue
            if klines.empty or len(klines) < int(args.min_bars):
                rows.append(_outcome_row(mapping, status="unbacktestable", reason="insufficient_klines", klines_rows=len(klines)))
                continue
            payload = _candidate_payload(row, start, capital_fraction=capital_fraction)
            try:
                result = run_single_backtest(
                    candidate_row=payload,
                    klines_df=klines,
                    capital=float(args.capital),
                    leverage=int(args.leverage),
                    max_holding_bars=int(args.max_holding_bars),
                    realism_profile=str(args.realism_profile),
                )
            except Exception as exc:
                rows.append(_outcome_row(mapping, status="unbacktestable", reason=f"backtest_failed:{type(exc).__name__}", klines_rows=len(klines)))
                continue
            rows.append(_outcome_row(mapping, status="backtested", result=result, klines_rows=len(klines)))
            await asyncio.sleep(float(args.sleep_seconds))
    finally:
        try:
            await client.close()
        except Exception:
            pass

    outcomes = pd.DataFrame(rows)
    outcomes_path = output_dir / "depth_shadow_forward_outcomes.csv"
    outcomes.to_csv(outcomes_path, index=False)
    status_counts = outcomes["outcome_status"].value_counts(dropna=False).to_dict() if "outcome_status" in outcomes else {}
    reason_counts = outcomes["outcome_reason"].value_counts(dropna=False).to_dict() if "outcome_reason" in outcomes else {}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "outcomes": str(outcomes_path),
        "candidate_rows": int(len(candidates)),
        "outcome_rows": int(len(outcomes)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "hours": int(args.hours),
        "min_bars": int(args.min_bars),
        "realism_profile": str(args.realism_profile),
        "note": "Only rows with complete scanner geometry are backtested. Missing geometry remains explicit missing evidence.",
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
    }
    manifest_path = output_dir / "depth_shadow_forward_outcomes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if int(status_counts.get("backtested", 0)) > 0 else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hours", type=int, default=8)
    parser.add_argument("--min-bars", type=int, default=420)
    parser.add_argument("--capital", type=float, default=400.0)
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--max-holding-bars", type=int, default=420)
    parser.add_argument(
        "--realism-profile",
        choices=REALISM_PROFILES,
        default=CANDIDATE_TIME_GEOMETRIC_PROFILE,
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        validate_realism_output_path(args.realism_profile, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(build_outcomes(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
