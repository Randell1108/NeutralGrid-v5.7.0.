"""PnL curve feature extraction for training enrichment.

Extracts structural features from user-supplied cumulative PnL curves
(hourly Binance snapshots) that reveal grid bot behavior patterns not
captured by scalar MAE/MFE metrics.

No Binance API calls.  Pure numerical computation.

Features
--------
pnl_curve_fill_concentration : float
    Fraction of total PnL from the single largest absolute move.
    Range [0, 1].  1.0 = all profit from one fill.  Closer to 0 = many
    small fills contributing evenly.

pnl_curve_active_ratio : float
    Fraction of bars with non-zero PnL changes.  Range [0, 1].
    Low values indicate the grid was idle most of the time.

pnl_curve_time_to_peak : float
    Normalized index (0–1) where the final peak PnL was first reached.
    0 = peak at start, 1 = peak at end.  Early peaks followed by
    flatlines suggest the grid captured profit early then stalled.

pnl_curve_flatline_pct : float
    Fraction of consecutive identical values in the curve.  Range [0, 1].
    High values indicate the bot stopped generating new fills.

pnl_curve_entropy : float
    Shannon entropy of discretized returns (10-bin histogram).
    Higher entropy = more unpredictable path.  Zero entropy = all
    returns in one bin (constant or single-step curve).

pnl_curve_shape_class : str
    Categorical label: "step", "gradual", "volatile", "flat".
    Determined by thresholds on active_ratio and fill_concentration.

pnl_curve_points : int
    Number of data points in the original curve.

pnl_curve_raw_json : str
    JSON-serialized raw curve for post-hoc analysis.  Allows
    reconstruction without re-fetching from Binance.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Minimum curve length for meaningful feature extraction
_MIN_CURVE_LEN = 3


@dataclass(frozen=True)
class PnlCurveFeatures:
    """Extracted features from a cumulative PnL curve."""

    fill_concentration: float
    active_ratio: float
    time_to_peak: float
    flatline_pct: float
    entropy: float
    shape_class: str
    points: int
    raw_json: str

    def to_dict(self, prefix: str = "pnl_curve_") -> Dict[str, Any]:
        """Flatten to a dict with prefixed keys for training row merging."""
        return {
            f"{prefix}fill_concentration": self.fill_concentration,
            f"{prefix}active_ratio": self.active_ratio,
            f"{prefix}time_to_peak": self.time_to_peak,
            f"{prefix}flatline_pct": self.flatline_pct,
            f"{prefix}entropy": self.entropy,
            f"{prefix}shape_class": self.shape_class,
            f"{prefix}points": self.points,
            f"{prefix}raw_json": self.raw_json,
        }


def extract_pnl_curve_features(
    pnl_values: Sequence[float],
) -> Optional[PnlCurveFeatures]:
    """Extract structural features from a cumulative PnL curve.

    Parameters
    ----------
    pnl_values
        Cumulative PnL at each snapshot (hourly or per-bar).
        Must have at least ``_MIN_CURVE_LEN`` points.

    Returns
    -------
    PnlCurveFeatures or None
        None if the curve is too short for meaningful extraction.
    """
    n = len(pnl_values)
    if n < _MIN_CURVE_LEN:
        logger.debug("PnL curve too short (%d < %d) for feature extraction", n, _MIN_CURVE_LEN)
        return None

    # ── Sanitize NaN / Inf values ──────────────────────────────────────
    values: List[float] = []
    nan_count = 0
    for v in pnl_values:
        if math.isnan(v) or math.isinf(v):
            nan_count += 1
            values.append(0.0)
        else:
            values.append(float(v))
    if nan_count > 0:
        logger.warning(
            "Replaced %d NaN/Inf values with 0.0 in PnL curve (%d points)", nan_count, n
        )

    # ── Compute bar-to-bar changes ───────────────────────────────────
    changes = [values[i + 1] - values[i] for i in range(n - 1)]
    abs_changes = [abs(c) for c in changes]
    total_abs_change = sum(abs_changes)

    # ── Fill concentration ───────────────────────────────────────────
    if total_abs_change > 0:
        fill_concentration = max(abs_changes) / total_abs_change
    else:
        fill_concentration = 0.0

    # ── Active ratio ─────────────────────────────────────────────────
    # A bar is "active" if the PnL changed (with tolerance for float noise)
    epsilon = max(total_abs_change * 1e-10, 1e-12)
    active_bars = sum(1 for c in abs_changes if c > epsilon)
    active_ratio = active_bars / len(changes)

    # ── Time to peak ─────────────────────────────────────────────────
    peak_val = max(values)
    # Find the FIRST index where peak was reached
    peak_idx = 0
    for i, v in enumerate(values):
        if abs(v - peak_val) < epsilon:
            peak_idx = i
            break
    time_to_peak = peak_idx / (n - 1) if n > 1 else 0.0

    # ── Flatline percentage ──────────────────────────────────────────
    # Count bars where value is identical to the previous bar
    flatline_count = sum(1 for c in abs_changes if c <= epsilon)
    flatline_pct = flatline_count / len(changes)

    # ── Shannon entropy of return distribution ───────────────────────
    entropy = _compute_return_entropy(changes)

    # ── Shape classification ─────────────────────────────────────────
    shape_class = _classify_shape(
        active_ratio=active_ratio,
        fill_concentration=fill_concentration,
        flatline_pct=flatline_pct,
        entropy=entropy,
    )

    # ── Raw curve persistence ────────────────────────────────────────
    raw_json = json.dumps([round(v, 8) for v in values])

    return PnlCurveFeatures(
        fill_concentration=round(fill_concentration, 6),
        active_ratio=round(active_ratio, 6),
        time_to_peak=round(time_to_peak, 6),
        flatline_pct=round(flatline_pct, 6),
        entropy=round(entropy, 6),
        shape_class=shape_class,
        points=n,
        raw_json=raw_json,
    )


def extract_time_to_threshold(
    pnl_values: Sequence[float],
    starting_capital: float,
    *,
    threshold_pct: float = 3.0,
    bar_minutes: float = 1.0,
) -> Dict[str, Any]:
    """First bar at which cumulative net PnL reaches ``threshold_pct`` of capital.

    This is the ENDOGENOUS time-to-target used to label fast winners (PIPELINE_FIX
    v2). ``pnl_values`` is the per-bar cumulative NET PnL in the SAME units as the
    curve built by ``btk_unified_runner`` (``equity - starting_capital``), i.e. the
    mark-to-market equity net of fees and funding — the SAME basis as ``net_pnl_pct``
    (``backtest_realistic.run``: ``net_pnl_pct = (final_equity - starting)/starting``).
    The target is therefore ``pnl_values[i] >= threshold_pct/100 * starting_capital``,
    equivalent to net ``pnl_pct >= threshold_pct``.

    Parameters
    ----------
    pnl_values
        Per-bar cumulative net PnL (dollars).
    starting_capital
        Starting capital (dollars); used to convert the percent threshold.
    threshold_pct
        Profit target in percent of starting capital (default 3.0).
    bar_minutes
        Minutes per bar (1.0 for 1m klines).

    Returns
    -------
    dict with ``time_to_target_bars`` (int; -1 if never reached),
    ``time_to_target_hours`` (float; NaN if never reached), and
    ``target_reached`` (bool). Pure function; no side effects.
    """
    out: Dict[str, Any] = {
        "time_to_target_bars": -1,
        "time_to_target_hours": float("nan"),
        "target_reached": False,
    }
    if not pnl_values or not (starting_capital > 0) or not math.isfinite(starting_capital):
        return out
    threshold_abs = (float(threshold_pct) / 100.0) * float(starting_capital)
    for i, v in enumerate(pnl_values):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv >= threshold_abs:
            out["time_to_target_bars"] = i
            out["time_to_target_hours"] = round(i * float(bar_minutes) / 60.0, 6)
            out["target_reached"] = True
            return out
    return out


def _compute_return_entropy(changes: List[float], n_bins: int = 10) -> float:
    """Shannon entropy of discretized returns.

    Uses a fixed 10-bin histogram.  Returns 0.0 if all changes are zero
    or if there's only one unique bin.  Maximum is log2(n_bins).
    """
    if not changes:
        return 0.0

    abs_max = max(abs(c) for c in changes)
    if abs_max == 0:
        return 0.0

    # Discretize into n_bins uniform bins over [-abs_max, abs_max]
    bin_width = (2 * abs_max) / n_bins
    counts = [0] * n_bins
    for c in changes:
        # Map change to bin index [0, n_bins-1]
        idx = int((c + abs_max) / bin_width)
        idx = max(0, min(idx, n_bins - 1))  # clamp both edges
        counts[idx] += 1

    # Compute Shannon entropy
    total = len(changes)
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _classify_shape(
    *,
    active_ratio: float,
    fill_concentration: float,
    flatline_pct: float,
    entropy: float,
) -> str:
    """Classify PnL curve shape into categorical label.

    Categories
    ----------
    "flat"      : Nearly no activity (active_ratio < 0.1)
    "step"      : One or two large moves, then flat (fill_concentration > 0.7)
    "volatile"  : Many moves, high entropy (entropy > 2.0 and active_ratio > 0.5)
    "gradual"   : Steady accumulation (everything else)

    Note: ``flatline_pct`` is accepted for API stability and future
    shape-refinement rules but is not used in the current classification.
    """
    _ = flatline_pct  # reserved for future refinement
    if active_ratio < 0.1:
        return "flat"
    if fill_concentration > 0.7:
        return "step"
    if entropy > 2.0 and active_ratio > 0.5:
        return "volatile"
    return "gradual"
