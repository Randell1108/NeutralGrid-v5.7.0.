from __future__ import annotations

"""
Offline order-book replay pipeline.

Reconstructs the aggregate open order book per symbol from Binance
CSV exports (Order History + Trade History) over the last 90 days.

No live API calls.  Fully deterministic and auditable.

Usage (CLI):
    neutralgrid-replay --input data/manual_exports --output data/replay_outputs

Usage (programmatic):
    from neutralgrid.replay.cli import run_pipeline
    run_pipeline(input_dir=Path("data/manual_exports"),
                 output_dir=Path("data/replay_outputs"))
"""
from neutralgrid.replay.schema import (
    OrderRow,
    TradeRow,
    OrderLifecycle,
    Event,
    BookSnapshot,
    BookLevel,
    Anomaly,
    BotSession,
    OverlapInterval,
    OPEN_SENTINEL_MS,
)

__all__ = [
    "OrderRow",
    "TradeRow",
    "OrderLifecycle",
    "Event",
    "BookSnapshot",
    "BookLevel",
    "Anomaly",
    "BotSession",
    "OverlapInterval",
    "OPEN_SENTINEL_MS",
]
