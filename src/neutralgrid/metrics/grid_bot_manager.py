"""
Binance Grid Bot Manager.

Manages expired/canceled grid bots, gathers metrics from Binance API,
and exports comprehensive data to CSV for analysis.
"""

import csv
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional, cast
from pathlib import Path

import logging

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.metrics.excursion_mtm_v20260304 import calculate_mtm_excursions

logger = logging.getLogger(__name__)


@dataclass
class GridBotMetrics:
    """Comprehensive metrics for an expired/canceled grid bot."""

    # Identifiers
    symbol: str
    strategy_id: int
    status: str  # "expired" or "cancelled"

    # Time data
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0

    # Investment
    investment_margin: float = 0.0
    leverage: int = 1

    # Position data (for active bots)
    position_size: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    liquidation_price: float = 0.0
    unrealized_pnl: float = 0.0

    # Grid configuration
    grid_lines: int = 0
    pending_buy_orders: int = 0
    pending_sell_orders: int = 0
    grid_low: float = 0.0
    grid_high: float = 0.0
    grid_spacing_pct: float = 0.0

    # Performance metrics
    matched_profit: float = 0.0
    funding_fees: float = 0.0
    commissions: float = 0.0
    total_profit: float = 0.0
    apy_pct: float = 0.0
    mae: float = 0.0  # Max Adverse Excursion (MTM, USDT)
    mfe: float = 0.0  # Max Favorable Excursion (MTM, USDT)
    mae_pct_initial: Optional[float] = None
    mfe_pct_initial: Optional[float] = None
    profit_factor: float = 0.0  # Gross profits / gross losses

    # Trade statistics
    total_trades: int = 0
    maker_count: int = 0
    taker_count: int = 0
    executed_qty: float = 0.0
    notional_completed: float = 0.0

    # Metadata
    last_updated: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        # Convert datetimes to ISO format
        for key in ["start_time", "end_time", "last_updated"]:
            if result[key]:
                result[key] = result[key].isoformat()
        return result

    def to_csv_row(self) -> dict[str, Any]:
        """Convert to CSV row format for export."""
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "strategy_type": "grid",
            "status": self.status,
            "start_time_utc": self.start_time.isoformat() if self.start_time else "",
            "end_time_utc": self.end_time.isoformat() if self.end_time else "",
            "duration_hours": round(self.duration_hours, 2),
            "invested_margin_usdt": round(self.investment_margin, 2),
            "leverage": self.leverage,
            "grids_count": self.grid_lines,
            "price_range_low": round(self.grid_low, 6),
            "price_range_high": round(self.grid_high, 6),
            "grid_spacing_pct": round(self.grid_spacing_pct, 4),
            "total_profit_usdt": round(self.total_profit, 4),
            "pnl_pct": round((self.total_profit / self.investment_margin * 100) if self.investment_margin > 0 else 0, 2),
            "realized_pnl_usdt": round(self.matched_profit, 4),
            "unrealized_pnl_usdt": round(self.unrealized_pnl, 4),
            "funding_fee_usdt": round(self.funding_fees, 4),
            "commission_usdt": round(self.commissions, 4),
            "mae": round(self.mae, 4),
            "mfe": round(self.mfe, 4),
            "mae_pct_initial": round(self.mae_pct_initial, 4) if self.mae_pct_initial is not None else "",
            "mfe_pct_initial": round(self.mfe_pct_initial, 4) if self.mfe_pct_initial is not None else "",
            "profit_factor": round(self.profit_factor, 4),
            "maker_count": self.maker_count,
            "taker_count": self.taker_count,
            "total_trades": self.total_trades,
            "executed_qty": round(self.executed_qty, 4),
            "notional_completed": round(self.notional_completed, 4),
            "holding_time_ok": "true" if self.duration_hours < 8 * 24 else "false",
            "last_updated_utc": self.last_updated.isoformat() if self.last_updated else "",
        }


class BinanceGridBotManager:
    """
    Manager for expired/canceled Binance Futures grid bots.

    Gathers comprehensive metrics from Binance API and exports to CSV.
    """

    CSV_HEADERS = [
        "strategy_id", "symbol", "strategy_type", "status",
        "start_time_utc", "end_time_utc", "duration_hours",
        "invested_margin_usdt", "leverage", "grids_count",
        "price_range_low", "price_range_high", "grid_spacing_pct",
        "total_profit_usdt", "pnl_pct", "realized_pnl_usdt",
        "unrealized_pnl_usdt", "funding_fee_usdt", "commission_usdt",
        "mae", "mfe", "mae_pct_initial", "mfe_pct_initial",
        "profit_factor", "maker_count", "taker_count",
        "total_trades", "executed_qty", "notional_completed",
        "holding_time_ok", "last_updated_utc"
    ]

    def __init__(self, client: Optional[BinanceClient] = None):
        """
        Initialize the Grid Bot Manager.

        Args:
            client: Optional BinanceClient instance
        """
        self.client = client or BinanceClient()
        self.bots: dict[int, GridBotMetrics] = {}  # strategy_id -> metrics
        cfg = get_config()
        self.csv_path = cfg.resolve_path(cfg.database.path).parent / "new_expired_bots.csv"
        self._csv_lock = asyncio.Lock()

    async def add_expired_bot(
        self,
        symbol: str,
        strategy_id: int,
        status: str,
        investment_margin: float,
        leverage: int = 1,
        grid_low: Optional[float] = None,
        grid_high: Optional[float] = None,
        num_grids: Optional[int] = None,
        end_time: Optional[datetime] = None
    ) -> GridBotMetrics:
        """
        Add an expired or canceled bot for metrics collection.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            strategy_id: Binance strategy ID
            status: "expired" or "cancelled"
            investment_margin: Initial margin invested
            leverage: Leverage used
            grid_low: Lower price bound (optional)
            grid_high: Upper price bound (optional)
            num_grids: Number of grids (optional)
            end_time: When the bot ended (optional, defaults to now)

        Returns:
            GridBotMetrics with populated data
        """
        if status not in ["expired", "cancelled"]:
            raise ValueError("Status must be 'expired' or 'cancelled'")

        # Create initial metrics object
        metrics = GridBotMetrics(
            symbol=symbol.upper(),
            strategy_id=strategy_id,
            status=status,
            investment_margin=investment_margin,
            leverage=leverage,
            grid_low=grid_low or 0.0,
            grid_high=grid_high or 0.0,
            grid_lines=num_grids or 0,
            end_time=end_time or datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )

        # Fetch metrics from Binance
        await self._fetch_bot_metrics(metrics)

        # Store in memory
        self.bots[strategy_id] = metrics

        # Append to CSV
        await self._append_to_csv(metrics)

        return metrics

    async def _fetch_bot_metrics(self, metrics: GridBotMetrics) -> None:
        """
        Fetch all metrics for a bot from Binance API.

        Args:
            metrics: GridBotMetrics to populate
        """
        symbol = metrics.symbol

        # Ensure end_time is set (always populated by add_expired_bot)
        if metrics.end_time is None:
            metrics.end_time = datetime.now(timezone.utc)

        # Calculate time range (last 180 days or from strategy start)
        end_time_ms = int(metrics.end_time.timestamp() * 1000)
        start_time_ms = end_time_ms - (180 * 24 * 60 * 60 * 1000)  # 180 days lookback

        # 1. Find bot start time from order history
        bot_start = await self._find_bot_start_time(
            symbol, metrics.strategy_id, start_time_ms, end_time_ms
        )
        if bot_start:
            metrics.start_time = bot_start
            start_time_ms = int(bot_start.timestamp() * 1000)

            # Calculate duration
            duration = metrics.end_time - bot_start
            metrics.duration_hours = duration.total_seconds() / 3600

        # 2. Fetch complete trade history for matched profit and commissions
        trades = await self.client.get_user_trades_all(
            symbol, start_time_ms, end_time_ms
        )

        matched_profit = 0.0
        commissions = 0.0
        total_qty = 0.0
        total_notional = 0.0
        gross_profits = 0.0
        gross_losses = 0.0
        maker_count = 0
        taker_count = 0

        for trade in trades:
            pnl = float(trade.get("realizedPnl", 0))
            matched_profit += pnl
            commissions += abs(float(trade.get("commission", 0)))
            qty = float(trade.get("qty", 0))
            price = float(trade.get("price", 0))
            total_qty += qty
            total_notional += qty * price

            # Profit factor components
            if pnl > 0:
                gross_profits += pnl
            else:
                gross_losses += abs(pnl)

            # Maker/Taker counts
            if trade.get("maker", False):
                maker_count += 1
            else:
                taker_count += 1

        metrics.matched_profit = matched_profit
        metrics.commissions = commissions
        metrics.total_trades = len(trades)
        metrics.executed_qty = total_qty
        metrics.notional_completed = total_notional
        metrics.maker_count = maker_count
        metrics.taker_count = taker_count

        # Calculate profit factor
        if gross_losses > 0:
            metrics.profit_factor = gross_profits / gross_losses
        elif gross_profits > 0:
            metrics.profit_factor = 999.99  # Infinite (all profits, no losses)
        else:
            metrics.profit_factor = 0.0

        # 3. Fetch funding and mark-price series for MTM MAE/MFE.
        funding_result, mark_result = await asyncio.gather(
            self.client.get_income_history_all(
                symbol=symbol,
                income_type="FUNDING_FEE",
                start_time=start_time_ms,
                end_time=end_time_ms,
            ),
            self.client.get_mark_price_klines_all(
                symbol=symbol,
                interval="1m",
                start_time=start_time_ms,
                end_time=end_time_ms,
            ),
            return_exceptions=True,
        )

        funding_income: list[dict]
        if isinstance(funding_result, Exception):
            logger.warning(
                "Failed to fetch funding history for %s strategy %s: %s",
                symbol, metrics.strategy_id, funding_result,
            )
            funding_income = []
        else:
            funding_income = cast(list[dict], funding_result)

        mark_klines: list[list]
        if isinstance(mark_result, Exception):
            logger.warning(
                "Failed to fetch mark klines for %s strategy %s: %s",
                symbol, metrics.strategy_id, mark_result,
            )
            mark_klines = []
        else:
            mark_klines = cast(list[list], mark_result)

        excursion = calculate_mtm_excursions(
            trades=trades,
            funding_fees=funding_income,
            mark_klines=mark_klines,
        )
        metrics.mae = excursion.mae_usdt
        metrics.mfe = excursion.mfe_usdt
        if metrics.investment_margin > 0:
            metrics.mae_pct_initial = (metrics.mae / metrics.investment_margin) * 100
            metrics.mfe_pct_initial = (metrics.mfe / metrics.investment_margin) * 100

        metrics.funding_fees = sum(float(f.get("income", 0)) for f in funding_income)

        # 4. Calculate grid spacing if we have grid bounds
        if metrics.grid_high > 0 and metrics.grid_low > 0 and metrics.grid_lines > 0:
            range_pct = (metrics.grid_high - metrics.grid_low) / metrics.grid_low * 100
            metrics.grid_spacing_pct = range_pct / metrics.grid_lines

        # 5. Calculate total profit (including commission costs)
        # For expired bots, unrealized PnL should be 0 (position closed)
        metrics.total_profit = (
            metrics.matched_profit +
            metrics.unrealized_pnl +
            metrics.funding_fees -
            metrics.commissions
        )

        # 6. Calculate APY
        if metrics.investment_margin > 0 and metrics.duration_hours > 0:
            duration_days = metrics.duration_hours / 24
            roi = metrics.total_profit / metrics.investment_margin
            metrics.apy_pct = ((1 + roi) ** (365 / duration_days) - 1) * 100

    async def _find_bot_start_time(
        self,
        symbol: str,
        strategy_id: int,
        start_time_ms: int,
        end_time_ms: int
    ) -> Optional[datetime]:
        """
        Find the start time of a bot by looking for first order with strategyId.

        Note: strategyId is found in clientOrderId for grid bots.

        Args:
            symbol: Trading pair
            strategy_id: Strategy ID to match
            start_time_ms: Search start time
            end_time_ms: Search end time

        Returns:
            Start time datetime or None
        """
        try:
            orders = await self.client.get_all_orders(
                symbol, start_time_ms, end_time_ms
            )

            # Look for orders with matching strategyId in clientOrderId
            # Grid bot orders typically have format: x-strategy_id-xxx
            strategy_str = str(strategy_id)

            earliest_time = None
            for order in orders:
                client_order_id = order.get("clientOrderId", "")
                order_time = order.get("time", 0)

                # Check if this order belongs to our strategy
                if strategy_str in client_order_id:
                    order_dt = datetime.fromtimestamp(order_time / 1000, tz=timezone.utc)
                    if earliest_time is None or order_dt < earliest_time:
                        earliest_time = order_dt

            return earliest_time

        except Exception as e:
            logger.warning("Failed to find bot start time for %s (strategy %s): %s", symbol, strategy_id, e)
            return None

    async def _append_to_csv(self, metrics: GridBotMetrics) -> None:
        """Append bot metrics to CSV file."""
        async with self._csv_lock:
            await asyncio.to_thread(self._append_to_csv_sync, metrics)

    def _append_to_csv_sync(self, metrics: GridBotMetrics) -> None:
        """Synchronous CSV append, run via to_thread to avoid blocking the event loop."""
        csv_exists = self.csv_path.exists()

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)

            if not csv_exists:
                writer.writeheader()

            writer.writerow(metrics.to_csv_row())

    async def export_all_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        Export all stored bots to CSV.

        Args:
            filepath: Optional path (defaults to data/new_expired_bots.csv)

        Returns:
            Path to the written CSV file
        """
        path = Path(filepath) if filepath else self.csv_path

        # Snapshot the values under the lock, then write in a thread.
        async with self._csv_lock:
            rows = [m.to_csv_row() for m in self.bots.values()]
            await asyncio.to_thread(self._export_rows_sync, path, rows)

        return str(path)

    def _export_rows_sync(self, path: Path, rows: list[dict]) -> None:
        """Synchronous bulk CSV write, run via to_thread."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def get_bot_report(self, strategy_id: int) -> Optional[dict]:
        """
        Get complete metrics report for a bot.

        Args:
            strategy_id: Strategy ID to look up

        Returns:
            Full metrics dictionary or None
        """
        metrics = self.bots.get(strategy_id)
        if metrics:
            return metrics.to_dict()
        return None

    def list_bots(self) -> list[dict]:
        """Get summary of all tracked bots."""
        return [
            {
                "strategy_id": m.strategy_id,
                "symbol": m.symbol,
                "status": m.status,
                "total_profit": m.total_profit,
                "apy_pct": m.apy_pct,
                "duration_hours": m.duration_hours,
            }
            for m in self.bots.values()
        ]


# Convenience function for synchronous usage
def create_grid_bot_manager() -> BinanceGridBotManager:
    """Create a BinanceGridBotManager instance."""
    return BinanceGridBotManager()
