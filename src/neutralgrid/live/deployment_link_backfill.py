"""Backfill deploy-link records from canonical live telemetry YAML files.

This module repairs manual-deployment workflows where a bot was created from a
scanner candidate, but the deploy-time ``deploy_linkage_log.csv`` row was not
written. It does not infer a deployment from a symbol alone: the live YAML must
carry a candidate_id, and that exact ID must exist in a deployment-ready CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import pandas as pd

from neutralgrid.core.candidate_id import extract_parts
from neutralgrid.live.candidate_deploy_linker import DeployLinker
from neutralgrid.live.decision.loader import LiveBotSpec, load_bot_specs


class LinkBackfillError(ValueError):
    """Raised when a live YAML cannot be safely linked to a candidate row."""


@dataclass(frozen=True)
class LinkBackfillResult:
    yaml_path: Path
    candidate_id: str
    strategy_id: str
    deployment_ready_path: Path
    linkage_path: Path
    wrote_row: bool
    reason: str


def backfill_link_from_live_yaml(
    yaml_path: Path,
    *,
    results_dir: Path = Path("results"),
    linkage_dir: Optional[Path] = None,
    allow_geometry_match: bool = False,
    grid_tolerance_pct: float = 0.01,
    dry_run: bool = False,
) -> LinkBackfillResult:
    """Write the missing deploy-link row for one canonical live YAML.

    The function is intentionally fail-closed:
    - live YAML must parse to exactly one bot spec;
    - both candidate_id and strategy_id must be present;
    - the candidate_id must match exactly one row across deployment-ready CSVs;
    - an existing strategy_id linked to a different candidate_id is rejected.
    """

    specs, warnings = load_bot_specs(yaml_path)
    if warnings:
        codes = ", ".join(w.code for w in warnings)
        raise LinkBackfillError(f"live YAML has loader warnings: {codes}")
    if len(specs) != 1:
        raise LinkBackfillError(f"expected exactly one live bot spec, found {len(specs)}")

    spec = specs[0]
    if spec.strategy_id is None or not spec.strategy_id.strip():
        raise LinkBackfillError("live YAML is missing strategy_id")

    strategy_id = spec.strategy_id.strip()
    if spec.candidate_id is not None and spec.candidate_id.strip():
        candidate_id = spec.candidate_id.strip()
        candidate_row, source_path = _find_candidate_row(candidate_id, results_dir)
        if not _candidate_scan_is_causal(candidate_id, spec.deploy_ts):
            raise LinkBackfillError(
                f"deployment time {spec.deploy_ts.isoformat()} precedes candidate scan"
            )
        match_note = f"backfilled_from_live_yaml={yaml_path.as_posix()}"
    else:
        if not allow_geometry_match:
            raise LinkBackfillError("live YAML is missing candidate_id")
        candidate_row, source_path = _find_candidate_row_by_geometry(
            spec, results_dir, grid_tolerance_pct=grid_tolerance_pct
        )
        candidate_id = str(candidate_row.get("candidate_id", "")).strip()
        match_note = (
            f"backfilled_from_live_yaml={yaml_path.as_posix()};"
            f"linkage_basis=forensic_geometry;"
            f"match_method=geometry;"
            f"grid_tolerance_pct={grid_tolerance_pct:g}"
        )
    symbol = str(candidate_row.get("symbol", "")).upper()
    if symbol and symbol != spec.symbol:
        raise LinkBackfillError(
            f"candidate symbol {symbol} does not match live symbol {spec.symbol}"
        )

    linker = DeployLinker(linkage_dir=linkage_dir)
    existing = linker.load_all()
    for row in existing:
        existing_strategy = _clean_text(row.get("strategy_id"))
        existing_candidate = _clean_text(row.get("candidate_id"))
        if existing_strategy == strategy_id and existing_candidate != candidate_id:
            raise LinkBackfillError(
                f"strategy_id {strategy_id} is already linked to {existing_candidate}"
            )
        if existing_strategy == strategy_id and existing_candidate == candidate_id:
            return LinkBackfillResult(
                yaml_path=yaml_path,
                candidate_id=candidate_id,
                strategy_id=strategy_id,
                deployment_ready_path=source_path,
                linkage_path=linker.path,
                wrote_row=False,
                reason="link already exists",
            )

    if not dry_run:
        linker.log_deployment_from_row(
            candidate_row,
            strategy_id,
            pipeline_version=_clean_text(candidate_row.get("pipeline_version")),
            margin_usdt=spec.capital_usdt,
            deploy_time_utc=spec.deploy_ts,
            notes=match_note,
        )

    return LinkBackfillResult(
        yaml_path=yaml_path,
        candidate_id=candidate_id,
        strategy_id=strategy_id,
        deployment_ready_path=source_path,
        linkage_path=linker.path,
        wrote_row=not dry_run,
        reason="link written" if not dry_run else "dry run",
    )


def _find_candidate_row(candidate_id: str, results_dir: Path) -> tuple[dict[str, Any], Path]:
    if not results_dir.is_dir():
        raise LinkBackfillError(f"results directory does not exist: {results_dir}")

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(results_dir.rglob("deployment_ready_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise LinkBackfillError(f"failed reading {path}: {exc}") from exc
        if "candidate_id" not in df.columns:
            continue
        candidate_ids = cast(pd.Series, df["candidate_id"]).fillna("").astype(str)
        hit_df = df.loc[candidate_ids == candidate_id]
        for _, row in hit_df.iterrows():
            candidate_row = dict(row)
            _require_terminal_admission(candidate_row, candidate_id)
            matches.append((path, candidate_row))

    if not matches:
        raise LinkBackfillError(f"candidate_id not found in deployment_ready CSVs: {candidate_id}")
    unique_paths = {path for path, _ in matches}
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in sorted(unique_paths))
        raise LinkBackfillError(
            f"candidate_id matched {len(matches)} deployment-ready rows: {paths}"
        )
    return matches[0][1], matches[0][0]


def _find_candidate_row_by_geometry(
    spec: LiveBotSpec,
    results_dir: Path,
    *,
    grid_tolerance_pct: float,
) -> tuple[dict[str, Any], Path]:
    """Resolve a missing candidate_id only when geometry gives one causal match."""

    if not results_dir.is_dir():
        raise LinkBackfillError(f"results directory does not exist: {results_dir}")
    if grid_tolerance_pct < 0:
        raise LinkBackfillError("grid_tolerance_pct must be non-negative")

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(results_dir.rglob("deployment_ready_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise LinkBackfillError(f"failed reading {path}: {exc}") from exc
        required = {"symbol", "candidate_id", "grid_lower", "grid_upper", "num_grids", "leverage"}
        if not required.issubset(df.columns):
            continue
        symbol_col = cast(pd.Series, df["symbol"]).fillna("").astype(str).str.upper()
        symbol_df = df.loc[symbol_col == spec.symbol]
        if symbol_df.empty:
            continue
        for _, row in symbol_df.iterrows():
            candidate_id = _clean_text(row.get("candidate_id"))
            if not candidate_id:
                continue
            try:
                _require_terminal_admission(dict(row), candidate_id)
            except LinkBackfillError:
                continue
            if not _candidate_scan_is_causal(candidate_id, spec.deploy_ts):
                continue
            if not _geometry_matches(spec, dict(row), grid_tolerance_pct):
                continue
            matches.append((path, dict(row)))

    if not matches:
        raise LinkBackfillError(
            "candidate_id not found by geometry; provide candidate_id in the live YAML "
            "or verify the bot was deployed from a scanner row"
        )
    if len(matches) > 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise LinkBackfillError(
            f"geometry matched {len(matches)} deployment-ready rows: {paths}"
        )
    return matches[0][1], matches[0][0]


def _candidate_scan_is_causal(candidate_id: str, deploy_ts: datetime) -> bool:
    scan_ts = extract_parts(candidate_id).get("scan_ts", "")
    try:
        scan_dt = datetime.strptime(scan_ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return scan_dt <= deploy_ts


def _require_terminal_admission(row: dict[str, Any], candidate_id: str) -> None:
    required = ("grid_is_valid", "hard_gate_passed", "stage_b_approved")
    if not all(_is_true(row.get(field)) for field in required):
        raise LinkBackfillError(
            f"candidate_id {candidate_id} did not pass all terminal admission gates"
        )


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _geometry_matches(
    spec: LiveBotSpec,
    row: dict[str, Any],
    grid_tolerance_pct: float,
) -> bool:
    lower = _to_finite_float(row.get("grid_lower"))
    upper = _to_finite_float(row.get("grid_upper"))
    grids = _to_finite_float(row.get("num_grids"))
    leverage = _to_finite_float(row.get("leverage"))
    if lower is None or upper is None or grids is None or leverage is None:
        return False
    tolerance = grid_tolerance_pct / 100.0
    lower_ok = _relative_abs_diff(lower, spec.grid_lower) <= tolerance
    upper_ok = _relative_abs_diff(upper, spec.grid_upper) <= tolerance
    grids_ok = int(round(grids)) == int(spec.num_grids)
    leverage_ok = math.isclose(leverage, float(spec.leverage), rel_tol=0.0, abs_tol=1e-9)
    return bool(lower_ok and upper_ok and grids_ok and leverage_ok)


def _relative_abs_diff(value: float, target: float) -> float:
    denom = max(abs(target), 1e-12)
    return abs(value - target) / denom


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill deploy_linkage_log.csv from a canonical live telemetry YAML."
    )
    parser.add_argument("yaml_path", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--linkage-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-geometry-match",
        action="store_true",
        help=(
            "When the YAML lacks candidate_id, infer it only from one causal "
            "deployment_ready row with matching symbol/grid/leverage."
        ),
    )
    parser.add_argument(
        "--grid-tolerance-pct",
        type=float,
        default=0.01,
        help="Relative grid-bound tolerance in percent for --allow-geometry-match.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = backfill_link_from_live_yaml(
        args.yaml_path,
        results_dir=args.results_dir,
        linkage_dir=args.linkage_dir,
        allow_geometry_match=bool(args.allow_geometry_match),
        grid_tolerance_pct=float(args.grid_tolerance_pct),
        dry_run=bool(args.dry_run),
    )
    print(f"candidate_id={result.candidate_id}")
    print(f"strategy_id={result.strategy_id}")
    print(f"deployment_ready_path={result.deployment_ready_path}")
    print(f"linkage_path={result.linkage_path}")
    print(f"wrote_row={result.wrote_row}")
    print(f"reason={result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
