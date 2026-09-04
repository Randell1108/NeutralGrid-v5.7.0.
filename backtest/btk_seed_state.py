"""
Seed state dataclasses for initialising RealisticGridBacktester from replay data.

Decoupled from ``src/neutralgrid/replay`` to keep the ``backtest/`` package
independent.  The loader in ``btk_replay_seed_loader.py`` constructs these
from exported replay CSV files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    LEGACY_REALISM_PROFILE,
    REALISM_PROFILES,
    validate_realism_profile,
)

__all__ = [
    "CANDIDATE_TIME_GEOMETRIC_PROFILE",
    "CANDIDATE_TIME_PUBLIC_MARKET_PROFILE",
    "LEGACY_REALISM_PROFILE",
    "REALISM_PROFILES",
    "validate_realism_profile",
    "SeedOrderLevel",
    "SeedState",
    "build_candidate_time_geometry_seed",
]

CANDIDATE_TIME_GEOMETRY_SOURCE = "candidate_time_geometry"


@dataclass(frozen=True)
class SeedOrderLevel:
    """A single price level from a replay-exported order book ladder."""
    side: str           # "BUY" or "SELL"
    price: float        # Price level
    qty_total: float    # Aggregate quantity at this level
    order_count: int = 1


@dataclass
class SeedState:
    """Snapshot of active grid levels at a given point in time.

    Used by :meth:`RealisticGridBacktester.seed_from_state` to initialise
    the simulator's ``_level_available_bar`` map so that levels already in the
    live order book at *t0* are recognised as immediately fillable.

    ``open_positions`` is intentionally left empty unless a deterministic
    source of position data is available — the backtester must not fabricate
    inventory.
    """
    t0: datetime
    open_buy_levels: List[SeedOrderLevel] = field(default_factory=list)
    open_sell_levels: List[SeedOrderLevel] = field(default_factory=list)
    open_positions: List[dict] = field(default_factory=list)
    symbol: Optional[str] = None
    qty_per_order: Optional[float] = None
    qty_source: str = "none"
    evidence_class: str = "missing"
    source: str = ""
    time_provenance: List[dict[str, str]] = field(default_factory=list)


def build_candidate_time_geometry_seed(
    *,
    symbol: str,
    grid_levels: Sequence[float],
    start_price: float,
    t0: datetime,
) -> SeedState | None:
    """Build a partial seed from known candidate-time geometry.

    This seed is intentionally diagnostic: it uses only the configured grid
    levels and the first replay price. Quantity remains model-sized because
    no live order quantity is verified by this source.
    """
    if not math.isfinite(start_price) or start_price <= 0:
        return None

    buy_levels: list[SeedOrderLevel] = []
    sell_levels: list[SeedOrderLevel] = []
    for raw_level in grid_levels:
        level = float(raw_level)
        if not math.isfinite(level) or level <= 0:
            continue
        if level < start_price:
            buy_levels.append(
                SeedOrderLevel(side="BUY", price=level, qty_total=0.0)
            )
        elif level > start_price:
            sell_levels.append(
                SeedOrderLevel(side="SELL", price=level, qty_total=0.0)
            )

    if not buy_levels and not sell_levels:
        return None

    return SeedState(
        t0=t0,
        open_buy_levels=buy_levels,
        open_sell_levels=sell_levels,
        symbol=symbol,
        qty_per_order=None,
        qty_source="model",
        evidence_class="partial",
        source=CANDIDATE_TIME_GEOMETRY_SOURCE,
    )
