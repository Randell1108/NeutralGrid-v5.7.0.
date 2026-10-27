from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from neutralgrid.models.meta_labeler import (
    ACTIVE_META_TARGET_CONTRACT,
    ACTIVE_SNAPSHOT_META_FEATURES,
)

SHEET_NAME = "FASTWIN Simulations"
LABEL_SOURCE = "canonical_configuration_simulation_v1"
EVIDENCE_SCHEMA = "canonical_fastwin_simulation_evidence_v1"
LABEL_FIELDS = (
    "strategy_id", "candidate_id", "symbol", "start_time_utc", "t1",
    "net_pnl_pct", "time_to_target_hours", "target_reached", "duration_hours",
)


def _same_number(left: Any, right: Any) -> bool:
    try:
        if pd.isna(left) and (right is None or pd.isna(right)):
            return True
        return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
    except (TypeError, ValueError):
        return False


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(result):
        raise ValueError("Canonical simulation has an invalid event timestamp")
    return cast(pd.Timestamp, result)


def apply_canonical_simulation_labels(live: pd.DataFrame, workbook_path: Path) -> pd.DataFrame:
    """Opt-in simulated labels; never rewrite recorded live outcomes.

    The workbook sheet selects a complete, hash-bound evidence cohort. Evidence
    binds engine output to bot configuration and the exact ex-ante feature vector.
    Missing target times are valid negatives only after this validation. Source
    weight 1.0 is an operator preference for canonical executions, not a claim
    that modeled outcomes are observed. Ordinary live ingestion is unchanged.
    """
    before = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
    labels = pd.read_excel(workbook_path, sheet_name=SHEET_NAME, dtype={"strategy_id": str})
    general = pd.read_excel(workbook_path, sheet_name="General", dtype={"strategy_id": str})
    if hashlib.sha256(workbook_path.read_bytes()).hexdigest() != before:
        raise ValueError("Canonical workbook changed during simulation label read")
    required = {*LABEL_FIELDS, "label_source", "evidence_path", "evidence_sha256"}
    if labels.empty or not required.issubset(labels.columns):
        raise ValueError("Canonical simulation label sheet is empty or incomplete")
    for column in ("evidence_path", "evidence_sha256", "label_source"):
        series = cast(pd.Series, labels[column])
        if bool(series.isna().any()) or series.nunique() != 1:
            raise ValueError(f"Canonical simulation {column} must be uniform and complete")
    if labels.iloc[0]["label_source"] != LABEL_SOURCE:
        raise ValueError("Unsupported canonical simulation label source")
    evidence_path = Path(str(labels.iloc[0]["evidence_path"]))
    if not evidence_path.is_absolute():
        evidence_path = workbook_path.parent / evidence_path
    raw = evidence_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != str(labels.iloc[0]["evidence_sha256"]):
        raise ValueError("Canonical simulation evidence hash mismatch")
    evidence = json.loads(raw)
    if (evidence.get("schema") != EVIDENCE_SCHEMA
            or evidence.get("target_contract") != ACTIVE_META_TARGET_CONTRACT
            or evidence.get("horizon_hours") != 7.0
            or evidence.get("complete_market_windows") is not True
            or evidence.get("deterministic_engine_repeat") is not True
            or evidence.get("configuration_checks_passed") is not True):
        raise ValueError("Canonical simulation evidence contract is incomplete")
    sources = evidence.get("source_hashes")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Canonical simulation evidence has no source bindings")
    for filename, expected in sources.items():
        source_path = evidence_path.parent / filename
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Canonical simulation source hash mismatch: {filename}")
    records = evidence.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Canonical simulation evidence has no records")
    evidence_rows = {str(row["strategy_id"]): row for row in records}
    if len(evidence_rows) != len(records):
        raise ValueError("Duplicate strategy in canonical simulation evidence")
    ids = cast(pd.Series, labels["strategy_id"]).astype(str)
    if ids.duplicated().any() or set(ids) != set(evidence_rows):
        raise ValueError("Canonical simulation cohort has missing or duplicate executions")
    selected = live.copy()
    selected["strategy_id"] = cast(pd.Series, selected["strategy_id"]).astype(str)
    selected = selected.loc[selected["strategy_id"].isin(ids)].copy()
    if selected["strategy_id"].duplicated().any() or set(selected["strategy_id"]) != set(ids):
        raise ValueError("Canonical simulation executions failed identity/feature admission")
    if general["strategy_id"].duplicated().any():
        raise ValueError("Duplicate canonical General strategy")
    general = general.set_index("strategy_id")
    selected = selected.set_index("strategy_id", drop=False)
    result: list[dict[str, Any]] = []
    for label in labels.to_dict("records"):
        sid = str(label["strategy_id"])
        expected = evidence_rows[sid]
        row = cast(pd.Series, selected.loc[sid]).to_dict()
        bot = cast(pd.Series, general.loc[sid])
        for field in LABEL_FIELDS:
            actual, reference = label[field], expected[field]
            if field in ("strategy_id", "candidate_id", "symbol"):
                matches = str(actual) == str(reference) == str(row[field])
            elif field in ("start_time_utc", "t1"):
                matches = _timestamp(actual) == _timestamp(reference)
            else:
                matches = _same_number(actual, reference)
            if not matches:
                raise ValueError(f"Canonical simulation evidence differs: {sid}/{field}")
        start, end = _timestamp(label["start_time_utc"]), _timestamp(label["t1"])
        if start != _timestamp(row["start_time_utc"]) or end != start.ceil("min") + pd.Timedelta(hours=7):
            raise ValueError(f"Canonical simulation event interval differs: {sid}")
        ttt = label["time_to_target_hours"]
        reached = label["target_reached"]
        if not isinstance(reached, (bool, np.bool_)):
            raise ValueError(f"Canonical simulation target flag is not boolean: {sid}")
        valid_time = pd.notna(ttt) and np.isfinite(float(ttt)) and 0 <= float(ttt) <= 7
        if bool(reached) != bool(valid_time) or (not reached and pd.notna(ttt)):
            raise ValueError(f"Canonical simulation target/time disagreement: {sid}")
        if not np.isfinite(float(label["net_pnl_pct"])) or not 0 <= float(label["duration_hours"]) <= 7:
            raise ValueError(f"Canonical simulation outcome is invalid: {sid}")
        configuration = expected["canonical_configuration"]
        capital_fraction = expected.get("capital_fraction")
        capital_used = expected.get("capital_used")
        if not _same_number(capital_fraction, 1.0) or not _same_number(capital_used, configuration["invested_margin_usdt"]):
            raise ValueError(f"Canonical simulation capital evidence is incomplete: {sid}")
        for field in ("price_range_low", "price_range_high", "grids_count", "invested_margin_usdt", "leverage"):
            if not _same_number(bot[field], configuration[field]):
                raise ValueError(f"Canonical simulation configuration changed: {sid}/{field}")
        if str(bot["mode"]).lower() != "geometric" or configuration["mode"] != "geometric":
            raise ValueError(f"Canonical simulation is not geometric: {sid}")
        for feature in ACTIVE_SNAPSHOT_META_FEATURES:
            if not _same_number(row[feature], expected["features"][feature]) or not np.isfinite(float(row[feature])):
                raise ValueError(f"Canonical simulation feature changed: {sid}/{feature}")
        if str(row["hmm_artifact_version"]) != str(expected["hmm_artifact_version"]):
            raise ValueError(f"Canonical simulation HMM lineage differs: {sid}")
        for cutoff in ("feature_cutoff_utc", "hmm_feature_cutoff_utc"):
            if _timestamp(row[cutoff]) != _timestamp(expected[cutoff]):
                raise ValueError(f"Canonical simulation feature cutoff differs: {sid}/{cutoff}")
        row.update({
            "hlabel_detail_observed_net_pnl_pct": row.get("net_pnl_pct"),
            "hlabel_detail_observed_t1": row.get("t1"),
            "source": "canonical_simulation", "source_class": LABEL_SOURCE,
            "observed_live_outcome": False, "sample_weight_override": 1.0,
            "capital_fraction": float(capital_fraction), "capital_used": float(capital_used),
            "training_event_id": f"canonical_simulation:{sid}",
            "net_pnl_pct": float(label["net_pnl_pct"]), "pnl_pct": float(label["net_pnl_pct"]),
            "time_to_target_hours": float(ttt) if reached else np.nan,
            "target_reached": bool(reached), "duration_hours": float(label["duration_hours"]),
            "t1": end.isoformat(), "t1_source": LABEL_SOURCE, "t1_is_synthetic": True,
            "label_source": LABEL_SOURCE, "label_evidence_sha256": digest,
        })
        result.append(row)
    return pd.DataFrame(result).reset_index(drop=True)
