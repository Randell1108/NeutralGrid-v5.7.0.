"""
Metrics Calculator for Grid Bot Performance Analysis.

Calculates key performance metrics from Binance trade fills and income data.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional
from datetime import datetime, timezone

from neutralgrid.metrics.excursion_mtm_v20260304 import calculate_mtm_excursions

@dataclass
class SessionMetrics:
    """Calculated metrics for a grid bot session."""

    # Core metrics
    symbol: str
    session_start: datetime
    session_end: datetime
    duration_hours: float

    # PnL metrics
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    funding_fees: float = 0.0
    net_pnl: float = 0.0

    # Performance metrics
    mae: float = 0.0  # Max Adverse Excursion (MTM, USDT)
    mfe: float = 0.0  # Max Favorable Excursion (MTM, USDT)
    mae_pct_initial: Optional[float] = None
    mfe_pct_initial: Optional[float] = None
    profit_factor: float = 0.0
    win_rate: float = 0.0
    sharpe_proxy: float = 0.0

    # Trade statistics
    total_fills: int = 0
    maker_count: int = 0
    taker_count: int = 0
    notional_completed: float = 0.0
    avg_profit_per_grid: float = 0.0

    # Grid params used
    grid_lower: Optional[float] = None
    grid_upper: Optional[float] = None
    num_grids: Optional[int] = None
    spacing_pct: Optional[float] = None
    leverage: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        # Convert datetimes to ISO format
        result["session_start"] = self.session_start.isoformat() if self.session_start else None
        result["session_end"] = self.session_end.isoformat() if self.session_end else None
        return result


class MetricsCalculator:
    """
    Calculate grid bot performance metrics from trade and income data.
    """

    def calculate_pnl(
        self,
        realized_pnl: list[dict],
        commissions: list[dict],
        funding_fees: list[dict]
    ) -> tuple[float, float, float, float]:
        """
        Calculate PnL metrics from income history.

        Args:
            realized_pnl: List of REALIZED_PNL income records
            commissions: List of COMMISSION income records
            funding_fees: List of FUNDING_FEE income records

        Returns:
            Tuple of (gross_pnl, total_fees, funding_fees_total, net_pnl)
        """
        gross_pnl = sum(float(r.get("income", 0)) for r in realized_pnl)
        total_fees = sum(abs(float(r.get("income", 0))) for r in commissions)
        funding_total = sum(float(r.get("income", 0)) for r in funding_fees)

        # Net PnL = Gross PnL - Fees + Funding (funding can be positive or negative)
        net_pnl = gross_pnl - total_fees + funding_total

        return gross_pnl, total_fees, funding_total, net_pnl

    def calculate_mae(
        self,
        trades: list[dict],
        entry_price: Optional[float] = None,
        funding_fees: Optional[list[dict]] = None,
        mark_klines: Optional[list[list]] = None,
    ) -> float:
        """
        Calculate Maximum Adverse Excursion (MTM).

        Uses a mark-to-market equity path built from:
        - trade realized PnL and commissions
        - funding cash flows
        - mark-price updates (when available)

        Args:
            trades: List of trade records
            entry_price: Optional entry price for reference

        Returns:
            MAE in quote-currency units (USDT for USDT-M futures)
        """
        _ = entry_price  # Backward-compatible argument; not used by MTM path.
        excursion = calculate_mtm_excursions(
            trades=trades,
            funding_fees=funding_fees or [],
            mark_klines=mark_klines or [],
        )
        return excursion.mae_usdt

    def calculate_mfe(
        self,
        trades: list[dict],
        funding_fees: Optional[list[dict]] = None,
        mark_klines: Optional[list[list]] = None,
    ) -> float:
        """
        Calculate Maximum Favorable Excursion (MTM) in quote-currency units.
        """
        excursion = calculate_mtm_excursions(
            trades=trades,
            funding_fees=funding_fees or [],
            mark_klines=mark_klines or [],
        )
        return excursion.mfe_usdt

    def calculate_profit_factor(self, trades: list[dict]) -> float:
        """
        Calculate Profit Factor = Gross Profits / Gross Losses.

        Args:
            trades: List of trade records

        Returns:
            Profit factor (> 1 means profitable)
        """
        if not trades:
            return 0.0

        gross_profits = 0.0
        gross_losses = 0.0

        for trade in trades:
            pnl = float(trade.get("realizedPnl", 0))
            if pnl > 0:
                gross_profits += pnl
            else:
                gross_losses += abs(pnl)

        if gross_losses == 0:
            return float("inf") if gross_profits > 0 else 0.0

        return gross_profits / gross_losses

    def calculate_maker_taker_counts(self, trades: list[dict]) -> tuple[int, int]:
        """
        Count maker vs taker trades.

        Args:
            trades: List of trade records

        Returns:
            Tuple of (maker_count, taker_count)
        """
        maker_count = sum(1 for t in trades if t.get("maker", False))
        taker_count = len(trades) - maker_count
        return maker_count, taker_count

    def calculate_notional(self, trades: list[dict]) -> float:
        """
        Calculate total notional for completed trades.

        Args:
            trades: List of trade records

        Returns:
            Total notional (price x quantity summed)
        """
        total = 0.0
        for trade in trades:
            price = float(trade.get("price", 0))
            qty = float(trade.get("qty", 0))
            total += price * qty
        return total

    def calculate_win_rate(self, trades: list[dict]) -> float:
        """
        Calculate win rate.

        Args:
            trades: List of trade records

        Returns:
            Win rate as percentage (0-100)
        """
        if not trades:
            return 0.0

        # Count trades with positive realized PnL
        winning_trades = sum(1 for t in trades if float(t.get("realizedPnl", 0)) > 0)
        # Only count trades that have realized PnL (exclude entry trades)
        trades_with_pnl = [t for t in trades if float(t.get("realizedPnl", 0)) != 0]

        if not trades_with_pnl:
            return 0.0

        return (winning_trades / len(trades_with_pnl)) * 100

    def calculate_sharpe_proxy(
        self,
        trades: list[dict],
        duration_hours: float,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate a proxy for Sharpe ratio.

        Uses simple return / volatility estimate.

        Args:
            trades: List of trade records
            duration_hours: Session duration in hours
            risk_free_rate: Annualized risk-free rate

        Returns:
            Sharpe ratio proxy
        """
        if not trades or duration_hours <= 0:
            return 0.0

        import numpy as np

        # Get PnL per trade
        pnls = [float(t.get("realizedPnl", 0)) for t in trades]
        pnls = [p for p in pnls if p != 0]  # Exclude zero PnL trades

        if not pnls:
            return 0.0

        n_obs = len(pnls)
        mean_return = np.mean(pnls)
        std_return = np.std(pnls, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe (assuming 24/7 trading)
        hours_per_year = 365 * 24
        obs_per_year = (n_obs / duration_hours * hours_per_year) if duration_hours > 0 else n_obs
        annualization_factor = np.sqrt(obs_per_year)

        # Convert annual risk-free rate to per-observation rate
        rf_per_obs = risk_free_rate / obs_per_year if obs_per_year > 0 else 0

        sharpe = (mean_return - rf_per_obs) / std_return * annualization_factor
        return float(sharpe)

    def calculate_session_metrics(
        self,
        session_data: dict,
        grid_params: Optional[dict] = None
    ) -> SessionMetrics:
        """
        Calculate all metrics for a grid bot session.

        Args:
            session_data: Data from get_grid_bot_session_data()
            grid_params: Optional grid parameters used

        Returns:
            SessionMetrics dataclass with all calculated metrics
        """
        symbol = session_data.get("symbol", "UNKNOWN")
        start_time = session_data.get("start_time", 0)
        end_time = session_data.get("end_time", 0)
        trades = session_data.get("trades", [])
        realized_pnl = session_data.get("realized_pnl", [])
        commissions = session_data.get("commissions", [])
        funding_fees = session_data.get("funding_fees", [])
        mark_klines = session_data.get("mark_klines", [])
        investment_margin = float(session_data.get("investment_margin", 0) or 0.0)

        # Calculate duration
        duration_ms = end_time - start_time
        duration_hours = duration_ms / (1000 * 60 * 60) if duration_ms > 0 else 0

        # Convert timestamps to datetime
        session_start = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc) if start_time else datetime.now(timezone.utc)
        session_end = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc) if end_time else datetime.now(timezone.utc)

        # Calculate PnL metrics
        gross_pnl, total_fees, funding_total, net_pnl = self.calculate_pnl(
            realized_pnl, commissions, funding_fees
        )

        # Calculate performance metrics
        excursion = calculate_mtm_excursions(
            trades=trades,
            funding_fees=funding_fees,
            mark_klines=mark_klines,
        )
        mae = excursion.mae_usdt
        mfe = excursion.mfe_usdt
        mae_pct_initial: Optional[float] = None
        mfe_pct_initial: Optional[float] = None
        if investment_margin > 0:
            mae_pct_initial = (mae / investment_margin) * 100
            mfe_pct_initial = (mfe / investment_margin) * 100
        profit_factor = self.calculate_profit_factor(trades)
        win_rate = self.calculate_win_rate(trades)
        sharpe_proxy = self.calculate_sharpe_proxy(trades, duration_hours)

        # Calculate trade statistics
        maker_count, taker_count = self.calculate_maker_taker_counts(trades)
        notional = self.calculate_notional(trades)
        total_fills = len(trades)

        # Calculate average profit per grid
        trades_with_pnl = len([t for t in trades if float(t.get("realizedPnl", 0)) != 0])
        avg_profit = net_pnl / trades_with_pnl if trades_with_pnl > 0 else 0

        # Build metrics object
        metrics = SessionMetrics(
            symbol=symbol,
            session_start=session_start,
            session_end=session_end,
            duration_hours=round(duration_hours, 2),
            gross_pnl=round(gross_pnl, 4),
            total_fees=round(total_fees, 4),
            funding_fees=round(funding_total, 4),
            net_pnl=round(net_pnl, 4),
            mae=round(mae, 4),
            mfe=round(mfe, 4),
            mae_pct_initial=round(mae_pct_initial, 4) if mae_pct_initial is not None else None,
            mfe_pct_initial=round(mfe_pct_initial, 4) if mfe_pct_initial is not None else None,
            profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else 999.99,
            win_rate=round(win_rate, 2),
            sharpe_proxy=round(sharpe_proxy, 4),
            total_fills=total_fills,
            maker_count=maker_count,
            taker_count=taker_count,
            notional_completed=round(notional, 4),
            avg_profit_per_grid=round(avg_profit, 6),
        )

        # Add grid params if provided
        if grid_params:
            metrics.grid_lower = grid_params.get("grid_lower")
            metrics.grid_upper = grid_params.get("grid_upper")
            metrics.num_grids = grid_params.get("num_grids")
            metrics.spacing_pct = grid_params.get("spacing_pct")
            metrics.leverage = grid_params.get("leverage")

        return metrics
