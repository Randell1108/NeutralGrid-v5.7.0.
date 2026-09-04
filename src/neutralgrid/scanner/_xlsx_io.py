"""Shared xlsx I/O + validation for scanner profile builders.

Single source for sheet reading, duplicate-id guards, schema validation, and
format detection used by both `pattern_profile.build_profile_from_enhanced_xlsx`
and `profile_model.train_profile_model_from_enhanced_xlsx`.

Prior to Phase 1, these helpers were duplicated in both modules with subtle
divergences (profile_model.py had no validation; pattern_profile.py's
format-detect leaned on a ValueError catch). This module consolidates them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd

logger = logging.getLogger(__name__)

MULTI_SHEET_NAMES = (
    "Entry Validation Metrics",
    "Performance Risk-Adjusted",
    "Market and Volatility",
)

CANONICAL_SINGLE_SHEET_NAME = "General"


def read_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a single sheet and normalize column names (strip whitespace)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def raise_on_duplicate_strategy_id(df: pd.DataFrame, *, context: str) -> None:
    """Raise ValueError if strategy_id column contains duplicates.

    Operates only when the column is present; sheets without strategy_id are
    no-ops (caller decides whether that is acceptable).
    """
    if "strategy_id" not in df.columns:
        return
    dup_ids = df["strategy_id"].dropna()
    if dup_ids.duplicated().any():
        n_dups = int(dup_ids.duplicated().sum())
        raise ValueError(
            f"Duplicate strategy_id detected ({n_dups} duplicates) in {context}. "
            f"Cannot proceed with label construction."
        )


def validate_dataframe(
    df: pd.DataFrame, sheet_name: str
) -> tuple[pd.DataFrame, list[str]]:
    """Validate xlsx sheet and return (possibly-modified df, warnings).

    Fail-closed invariants (raise ValueError):
      - win_rate > 100  — out-of-range; data ingest error
      - profit_factor < 0  — undefined; previously logged-only
      - duration_hours < 0  — undefined; previously logged-only

    Soft fixes (warn, do not raise):
      - win_rate < 0 → clipped to 0.0 (pre-Phase-1 behavior retained)
      - strategy_id NULL → warn only (upstream guarantees uniqueness)
    """
    warnings: list[str] = []
    df = df.copy()

    if "win_rate" in df.columns:
        win_rate_series = cast(pd.Series, df["win_rate"])
        over = win_rate_series > 100
        if bool(over.any()):
            n = int(over.sum())
            raise ValueError(
                f"[{sheet_name}] {n} rows have win_rate > 100; invariant violation."
            )
        under = win_rate_series < 0
        if bool(under.any()):
            n = int(under.sum())
            warnings.append(f"[{sheet_name}] Set {n} negative win_rate values to 0%")
            df.loc[under, "win_rate"] = 0.0

    if "profit_factor" in df.columns:
        pf_series = cast(pd.Series, df["profit_factor"])
        negative = pf_series < 0
        if bool(negative.any()):
            n = int(negative.sum())
            raise ValueError(
                f"[{sheet_name}] {n} rows have negative profit_factor; invariant violation."
            )

    if "duration_hours" in df.columns:
        dur_series = cast(pd.Series, df["duration_hours"])
        negative = dur_series < 0
        if bool(negative.any()):
            n = int(negative.sum())
            raise ValueError(
                f"[{sheet_name}] {n} rows have negative duration_hours; invariant violation."
            )

    if "strategy_id" in df.columns:
        null_mask = cast(pd.Series, df["strategy_id"].isna())
        if bool(null_mask.any()):
            n = int(null_mask.sum())
            warnings.append(f"[{sheet_name}] {n} rows have NULL strategy_id")

    return df, warnings


def detect_format(xlsx_path: Path) -> Literal["multi", "single"]:
    """Inspect sheet names to decide multi-sheet vs single-sheet layout.

    Does NOT rely on a ValueError catch from read_sheet — this makes the
    control flow explicit and avoids masking unrelated ValueErrors.

    When a workbook carries the canonical ``General`` sheet, it wins even if
    legacy multi-sheet tabs are also present. Pattern-profile retraining now
    backfills features directly onto ``General`` and should not silently fall
    back to the older merged-sheet contract.
    """
    with pd.ExcelFile(xlsx_path) as xl:
        sheet_names = set(str(s) for s in xl.sheet_names)
    if CANONICAL_SINGLE_SHEET_NAME in sheet_names:
        return "single"
    if set(MULTI_SHEET_NAMES).issubset(sheet_names):
        return "multi"
    return "single"


def read_single_sheet(xlsx_path: Path) -> tuple[pd.DataFrame, str]:
    """Read the canonical sheet of a single-sheet xlsx.

    When a workbook carries a ``General`` sheet, use it explicitly. Otherwise
    fall back to the first sheet to preserve legacy single-sheet behavior.
    Returns ``(df, sheet_name)``.
    """
    with pd.ExcelFile(xlsx_path) as xl:
        if CANONICAL_SINGLE_SHEET_NAME in xl.sheet_names:
            sheet_name = CANONICAL_SINGLE_SHEET_NAME
        else:
            sheet_name = str(xl.sheet_names[0])
        df = pd.read_excel(xl, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sheet_name
