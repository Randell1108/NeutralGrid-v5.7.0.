"""
Deploy-time linkage logger.

Records the exact moment a scanner candidate becomes a live bot by writing
an append-only CSV row with::

    candidate_id, strategy_id, deploy_time_utc, grid config snapshot

This is the **source-of-truth** link between scanner predictions and live
execution.  All downstream analysis (outcome matching, training table
building) keys off this linkage file.

Usage at deploy time::

    from neutralgrid.live.candidate_deploy_linker import DeployLinker

    linker = DeployLinker()
    linker.log_deployment(
        candidate_id="BTCUSDT_20260227_143000_a1b2c3d4",
        strategy_id="ng_12345678",
        grid_lower=40000.0,
        grid_upper=42000.0,
        num_grids=30,
        leverage=10,
    )
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import msvcrt  # type: ignore[import-not-found,reportUnusedImport]
except ImportError:
    msvcrt = None  # type: ignore[assignment]
try:
    import fcntl  # type: ignore[import-not-found]
except ImportError:
    fcntl = None  # type: ignore[assignment]

from neutralgrid.core.candidate_id import extract_parts, make_candidate_id
from neutralgrid.live.deployment_payload_v20260304 import resolve_deployment_sizing

logger = logging.getLogger(__name__)

# Default linkage file lives alongside other data outputs
_DEFAULT_LINKAGE_DIR = Path("data/linkage")

_CANDIDATE_ID_RE = re.compile(
    r"^(?P<symbol>[^_]+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<hash>[0-9a-fA-F]{8})$"
)

# CSV column order (append-only, so this is also the header for new files)
_LINKAGE_COLUMNS = [
    "candidate_id",
    "strategy_id",
    "deploy_time_utc",
    "symbol",
    "scan_ts",
    "config_hash",
    "grid_lower",
    "grid_upper",
    "num_grids",
    "leverage",
    "grid_spacing_pct",
    "profit_per_grid_pct",
    "score",
    "scan_score",
    "ev_score",
    "ev_24h",
    "meta_prob",
    "meta_prob_source",
    "deployment_score",
    "pipeline_version",
    "capital_fraction",
    "capital_base_usdt",
    "volatility_scale_applied",
    "margin_usdt",
    "notes",
]


class DeployLinkIntegrityError(ValueError):
    """Raised when a deployment would corrupt the identity ledger."""


class DeployLinker:
    """Append-only CSV logger that records candidate → live-bot linkage.

    Parameters
    ----------
    linkage_dir : Path, optional
        Directory for linkage CSV files.  Defaults to ``data/linkage/``.
    """

    def __init__(self, linkage_dir: Optional[Path] = None) -> None:
        self._dir = Path(linkage_dir) if linkage_dir else _DEFAULT_LINKAGE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "deploy_linkage_log.csv"

    @property
    def path(self) -> Path:
        """Path to the linkage CSV file."""
        return self._path

    def log_deployment(
        self,
        candidate_id: str,
        strategy_id: str,
        *,
        deploy_time_utc: Optional[datetime | str] = None,
        grid_lower: Optional[float] = None,
        grid_upper: Optional[float] = None,
        num_grids: Optional[int] = None,
        leverage: Optional[int] = None,
        grid_spacing_pct: Optional[float] = None,
        profit_per_grid_pct: Optional[float] = None,
        score: Optional[float] = None,
        scan_score: Optional[float] = None,
        ev_score: Optional[float] = None,
        ev_24h: Optional[float] = None,
        meta_prob: Optional[float] = None,
        meta_prob_source: str = "missing",
        deployment_score: Optional[float] = None,
        pipeline_version: str = "",
        capital_fraction: Optional[float] = None,
        capital_base_usdt: Optional[float] = None,
        volatility_scale_applied: Optional[float] = None,
        margin_usdt: Optional[float] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record a single deployment event.

        Returns the row dict that was written.
        """
        candidate_id = str(candidate_id).strip()
        strategy_id = str(strategy_id).strip()
        deploy_time = self._format_deploy_time(deploy_time_utc)
        self._validate_identity_and_geometry(
            candidate_id=candidate_id,
            strategy_id=strategy_id,
            deploy_time_utc=deploy_time,
            grid_lower=grid_lower,
            grid_upper=grid_upper,
            num_grids=num_grids,
            leverage=leverage,
        )
        parts = extract_parts(candidate_id)

        row: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "strategy_id": strategy_id,
            "deploy_time_utc": deploy_time,
            "symbol": parts["symbol"],
            "scan_ts": parts["scan_ts"],
            "config_hash": parts["config_hash"],
            "grid_lower": grid_lower,
            "grid_upper": grid_upper,
            "num_grids": num_grids,
            "leverage": leverage,
            "grid_spacing_pct": grid_spacing_pct,
            "profit_per_grid_pct": profit_per_grid_pct,
            "score": score,
            "scan_score": scan_score,
            "ev_score": ev_score,
            "ev_24h": ev_24h,
            "meta_prob": meta_prob,
            "meta_prob_source": meta_prob_source,
            "deployment_score": deployment_score,
            "pipeline_version": pipeline_version,
            "capital_fraction": capital_fraction,
            "capital_base_usdt": capital_base_usdt,
            "volatility_scale_applied": volatility_scale_applied,
            "margin_usdt": margin_usdt,
            "notes": notes,
        }

        wrote_row = self._append_row(row)
        if not wrote_row:
            logger.info(
                "Deployment link already exists; no duplicate written: %s -> %s",
                candidate_id,
                strategy_id,
            )
            return row
        logger.info(
            "Logged deployment: %s → %s at %s",
            candidate_id, strategy_id, row["deploy_time_utc"],
        )
        return row

    def log_deployment_from_row(
        self,
        candidate_row: Dict[str, Any],
        strategy_id: str,
        pipeline_version: str = "",
        margin_usdt: Optional[float] = None,
        deploy_time_utc: Optional[datetime | str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Record deployment from a scanner DataFrame row (dict-like).

        Extracts all relevant fields from the candidate row automatically.
        """
        sizing = resolve_deployment_sizing(candidate_row)
        vol_scale = self._to_float(candidate_row.get("kelly_volatility_scale"), default=1.0)
        effective_margin = margin_usdt
        if effective_margin is None:
            effective_margin = sizing.effective_margin_usdt

        return self.log_deployment(
            candidate_id=str(candidate_row.get("candidate_id", "")),
            strategy_id=strategy_id,
            deploy_time_utc=deploy_time_utc,
            grid_lower=candidate_row.get("grid_lower"),
            grid_upper=candidate_row.get("grid_upper"),
            num_grids=candidate_row.get("num_grids"),
            leverage=candidate_row.get("leverage"),
            grid_spacing_pct=candidate_row.get("grid_spacing_pct"),
            profit_per_grid_pct=candidate_row.get("profit_per_grid_pct"),
            score=candidate_row.get("score"),
            scan_score=candidate_row.get("scan_score", candidate_row.get("score")),
            ev_score=candidate_row.get("ev_score"),
            ev_24h=candidate_row.get("ev_24h"),
            meta_prob=candidate_row.get("meta_prob"),
            meta_prob_source=self._to_source(candidate_row.get("meta_prob_source")),
            deployment_score=candidate_row.get("deployment_score"),
            pipeline_version=pipeline_version,
            capital_fraction=sizing.capital_fraction,
            capital_base_usdt=sizing.capital_base_usdt,
            volatility_scale_applied=vol_scale,
            margin_usdt=effective_margin,
            notes=notes,
        )

    def load_all(self) -> list[Dict[str, Any]]:
        """Load all linkage records as a list of dicts."""
        if not self._path.exists():
            return []
        import pandas as pd
        df = pd.read_csv(self._path)
        return df.to_dict(orient="records")

    def get_strategy_ids_for_candidate(self, candidate_id: str) -> list[str]:
        """Return all strategy_ids linked to a given candidate_id."""
        records = self.load_all()
        return [
            str(r["strategy_id"])
            for r in records
            if str(r.get("candidate_id", "")) == candidate_id
        ]

    def get_candidate_id_for_strategy(self, strategy_id: str) -> Optional[str]:
        """Return the candidate_id linked to a given strategy_id (first match)."""
        records = self.load_all()
        for r in records:
            if str(r.get("strategy_id", "")) == strategy_id:
                return str(r["candidate_id"])
        return None

    # ── Internal ──────────────────────────────────────────────────────────

    def _append_row(self, row: Dict[str, Any]) -> bool:
        """Append one identity-safe row, or return ``False`` if it exists."""
        with open(self._path, "a+", newline="", encoding="utf-8") as fh:
            try:
                if msvcrt is not None:
                    # Seek to start so all threads lock the same byte region
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                elif fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]

                # Check header inside the lock to avoid TOCTOU race
                write_header = os.path.getsize(self._path) == 0

                if not write_header:
                    fh.seek(0)
                    reader = csv.DictReader(fh)
                    if reader.fieldnames != _LINKAGE_COLUMNS:
                        raise DeployLinkIntegrityError(
                            "deploy linkage header does not match the governed schema"
                        )
                    candidate_id = str(row.get("candidate_id", "")).strip()
                    strategy_id = str(row.get("strategy_id", "")).strip()
                    for existing in reader:
                        existing_strategy = str(existing.get("strategy_id", "")).strip()
                        if existing_strategy != strategy_id:
                            continue
                        existing_candidate = str(existing.get("candidate_id", "")).strip()
                        if existing_candidate == candidate_id:
                            return False
                        raise DeployLinkIntegrityError(
                            f"strategy_id {strategy_id} is already linked to "
                            f"{existing_candidate}"
                        )

                fh.seek(0, os.SEEK_END)
                writer = csv.DictWriter(fh, fieldnames=_LINKAGE_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in _LINKAGE_COLUMNS})
                fh.flush()
                os.fsync(fh.fileno())
                return True
            finally:
                if msvcrt is not None:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                elif fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]

    @staticmethod
    def _validate_identity_and_geometry(
        *,
        candidate_id: str,
        strategy_id: str,
        deploy_time_utc: str,
        grid_lower: Optional[float],
        grid_upper: Optional[float],
        num_grids: Optional[int],
        leverage: Optional[int],
    ) -> None:
        match = _CANDIDATE_ID_RE.fullmatch(candidate_id)
        if match is None:
            raise DeployLinkIntegrityError(
                "candidate_id must use SYMBOL_YYYYMMDD_HHMMSS_hash8 format"
            )
        if not strategy_id or strategy_id.lower() in {"nan", "none", "null"}:
            raise DeployLinkIntegrityError("strategy_id must be non-empty")

        try:
            scan_time = datetime.strptime(
                f"{match.group('date')}_{match.group('time')}",
                "%Y%m%d_%H%M%S",
            ).replace(tzinfo=timezone.utc)
            deploy_time = datetime.fromisoformat(deploy_time_utc)
        except ValueError as exc:
            raise DeployLinkIntegrityError(f"invalid deployment timestamp: {exc}") from exc
        if deploy_time < scan_time:
            raise DeployLinkIntegrityError(
                f"deployment time {deploy_time_utc} precedes candidate scan "
                f"{scan_time.isoformat()}"
            )

        if (
            grid_lower is None
            or grid_upper is None
            or num_grids is None
            or leverage is None
        ):
            raise DeployLinkIntegrityError(
                "grid_lower, grid_upper, num_grids, and leverage are required"
            )
        try:
            lower = float(grid_lower)
            upper = float(grid_upper)
            grids_float = float(num_grids)
            leverage_float = float(leverage)
        except (TypeError, ValueError) as exc:
            raise DeployLinkIntegrityError("deployment geometry must be numeric") from exc
        if not all(math.isfinite(v) for v in (lower, upper, grids_float, leverage_float)):
            raise DeployLinkIntegrityError("deployment geometry must be finite")
        if lower <= 0 or upper <= lower:
            raise DeployLinkIntegrityError("deployment grid bounds are invalid")
        if not grids_float.is_integer() or int(grids_float) < 2:
            raise DeployLinkIntegrityError("num_grids must be an integer >= 2")
        if not leverage_float.is_integer() or int(leverage_float) <= 0:
            raise DeployLinkIntegrityError("leverage must be a positive integer")

        expected = make_candidate_id(
            match.group("symbol"),
            f"{match.group('date')}_{match.group('time')}",
            grid_lower=lower,
            grid_upper=upper,
            num_grids=int(grids_float),
            leverage=int(leverage_float),
        )
        if expected.lower() != candidate_id.lower():
            raise DeployLinkIntegrityError(
                "candidate_id config hash does not match deployment geometry"
            )

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _to_source(value: Any, default: str = "missing") -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return default
        return text

    @staticmethod
    def _format_deploy_time(value: Optional[datetime | str]) -> str:
        if value is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(value, datetime):
            dt = value
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        text = str(value).strip()
        if not text:
            return datetime.now(timezone.utc).isoformat()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DeployLinkIntegrityError(f"invalid deploy_time_utc: {text}") from exc
        if dt.tzinfo is None or dt.utcoffset() is None:
            raise DeployLinkIntegrityError("deploy_time_utc string must include a timezone")
        return dt.astimezone(timezone.utc).isoformat()
