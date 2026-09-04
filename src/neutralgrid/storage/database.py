"""
SQLite Database for Performance Metrics Storage.

Stores bot runs and post-trade performance metrics
for statistical evaluation and guideline refinement.
"""

from __future__ import annotations

import aiosqlite
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from neutralgrid.core.config import get_config
from neutralgrid.grid.calculator import GridParams

if TYPE_CHECKING:
    from neutralgrid.validation.regime_validator import ValidationResult
    from neutralgrid.metrics.calculator import SessionMetrics


class Database:
    """Async SQLite database for metrics storage.

    Uses a persistent connection with WAL mode and busy_timeout
    to avoid connection-per-call overhead and improve concurrency.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_config().database.path
        self._conn: Optional[aiosqlite.Connection] = None
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create a persistent database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    async def close(self):
        """Close the persistent database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def initialize(self):
        """Create database tables if they don't exist."""
        db = await self._get_conn()

        # Bot runs table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validation_status TEXT NOT NULL,
                grid_lower REAL,
                grid_upper REAL,
                num_grids INTEGER,
                spacing_pct REAL,
                profit_per_grid_pct REAL,
                leverage INTEGER,
                total_notional REAL,
                stop_loss_pct REAL,
                take_profit_pct REAL,
                max_hold_time TEXT,
                expected_return_pct REAL,
                validation_1h_passed INTEGER,
                validation_15m_passed INTEGER,
                validation_5m_passed INTEGER,
                hmm_regime_passed INTEGER,
                validation_reason TEXT
            )
        """)

        # Migration: add hmm_regime_passed to existing bot_runs tables
        try:
            await db.execute("ALTER TABLE bot_runs ADD COLUMN hmm_regime_passed INTEGER")
        except Exception:
            pass  # Column already exists

        # Bot performance metrics table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_run_id INTEGER REFERENCES bot_runs(id),
                symbol TEXT,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sharpe_proxy REAL,
                max_adverse_excursion REAL,
                max_favorable_excursion REAL,
                mae_pct_initial REAL,
                mfe_pct_initial REAL,
                profit_factor REAL,
                regime_stability_score REAL,
                duration_hours REAL,
                total_fills INTEGER,
                total_fees_paid REAL,
                funding_fees REAL,
                net_pnl REAL,
                gross_pnl REAL,
                win_rate REAL,
                avg_profit_per_grid REAL,
                maker_count INTEGER,
                taker_count INTEGER,
                notional_completed REAL,
                grid_lower REAL,
                grid_upper REAL,
                num_grids INTEGER,
                spacing_pct REAL,
                leverage INTEGER
            )
        """)

        # Backward-compatible migration for databases created before
        # max_favorable_excursion / pct excursion fields existed.
        await self._ensure_bot_metrics_columns(db)

        # Validation history table (for analytics)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS validation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_valid INTEGER NOT NULL,
                adx_1h REAL,
                adx_15m REAL,
                adx_5m REAL,
                rsi_15m REAL,
                ema_crosses_5m INTEGER,
                vwap_crosses_5m INTEGER,
                range_size_pct REAL,
                failure_reason TEXT
            )
        """)

        await db.commit()

    async def _ensure_bot_metrics_columns(self, db: aiosqlite.Connection) -> None:
        """Add bot_metrics columns that may be missing in older databases."""
        async with db.execute("PRAGMA table_info(bot_metrics)") as cursor:
            rows = await cursor.fetchall()

        existing = {str(row[1]) for row in rows}
        required: dict[str, str] = {
            "max_favorable_excursion": "REAL",
            "mae_pct_initial": "REAL",
            "mfe_pct_initial": "REAL",
        }
        for column, ddl_type in required.items():
            if column in existing:
                continue
            await db.execute(
                f"ALTER TABLE bot_metrics ADD COLUMN {column} {ddl_type}"
            )

    async def save_bot_run(
        self,
        grid_params: GridParams,
        validation_result: Optional[ValidationResult] = None
    ) -> int:
        """
        Save a bot run (validation attempt) to the database.

        Args:
            grid_params: Grid parameters (may be invalid)
            validation_result: Full validation result

        Returns:
            The ID of the inserted bot run
        """
        db = await self._get_conn()

        # Extract validation details
        val_1h = validation_result.tf_1h if validation_result else None
        val_15m = validation_result.tf_15m if validation_result else None
        val_5m = validation_result.tf_5m if validation_result else None

        reason = None
        if not grid_params.is_valid:
            reasons = []
            if val_1h and not val_1h.is_valid:
                reasons.append(f"HMM: {val_1h.reason}")
            if val_15m and not val_15m.is_valid:
                reasons.append(f"15M: {val_15m.reason}")
            if val_5m and not val_5m.is_valid:
                reasons.append(f"5M: {val_5m.reason}")
            reason = " | ".join(reasons) if reasons else "Unknown"

        # Derive hmm_regime_passed from the actual HMM check result
        hmm_passed = 1 if val_1h and val_1h.is_valid else 0

        cursor = await db.execute("""
            INSERT INTO bot_runs (
                symbol, validation_status, grid_lower, grid_upper,
                num_grids, spacing_pct, profit_per_grid_pct, leverage,
                total_notional, stop_loss_pct, take_profit_pct,
                max_hold_time, expected_return_pct,
                validation_1h_passed, validation_15m_passed,
                validation_5m_passed, hmm_regime_passed, validation_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            grid_params.symbol,
            "VALID" if grid_params.is_valid else "INVALID",
            grid_params.grid_lower,
            grid_params.grid_upper,
            grid_params.num_grids,
            grid_params.grid_spacing_pct,
            grid_params.profit_per_grid_pct,
            grid_params.leverage,
            grid_params.total_notional,
            grid_params.stop_loss_pct * 100 if grid_params.stop_loss_pct is not None else None,
            grid_params.take_profit_pct * 100 if grid_params.take_profit_pct is not None else None,
            grid_params.max_holding_time,
            grid_params.expected_net_return_pct,
            1 if val_1h and val_1h.is_valid else 0,
            1 if val_15m and val_15m.is_valid else 0,
            1 if val_5m and val_5m.is_valid else 0,
            hmm_passed,
            reason
        ))

        await db.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    async def save_session_metrics(
        self,
        metrics: SessionMetrics,
        bot_run_id: Optional[int] = None
    ) -> int:
        """
        Save session metrics for a completed grid bot run.

        Args:
            metrics: SessionMetrics object with calculated metrics
            bot_run_id: Optional ID of the associated bot run

        Returns:
            The ID of the inserted metrics record
        """
        db = await self._get_conn()

        cursor = await db.execute("""
            INSERT INTO bot_metrics (
                bot_run_id, symbol, session_start, session_end,
                sharpe_proxy, max_adverse_excursion, max_favorable_excursion,
                mae_pct_initial, mfe_pct_initial, profit_factor,
                duration_hours, total_fills, total_fees_paid,
                funding_fees, net_pnl, gross_pnl, win_rate,
                avg_profit_per_grid, maker_count, taker_count,
                notional_completed, grid_lower, grid_upper,
                num_grids, spacing_pct, leverage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bot_run_id,
            metrics.symbol,
            metrics.session_start.isoformat() if metrics.session_start else None,
            metrics.session_end.isoformat() if metrics.session_end else None,
            metrics.sharpe_proxy,
            metrics.mae,
            metrics.mfe,
            metrics.mae_pct_initial,
            metrics.mfe_pct_initial,
            metrics.profit_factor,
            metrics.duration_hours,
            metrics.total_fills,
            metrics.total_fees,
            metrics.funding_fees,
            metrics.net_pnl,
            metrics.gross_pnl,
            metrics.win_rate,
            metrics.avg_profit_per_grid,
            metrics.maker_count,
            metrics.taker_count,
            metrics.notional_completed,
            metrics.grid_lower,
            metrics.grid_upper,
            metrics.num_grids,
            metrics.spacing_pct,
            metrics.leverage,
        ))

        await db.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    async def get_session_metrics(self, limit: int = 20) -> list[dict]:
        """Get recent session metrics."""
        db = await self._get_conn()
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM bot_metrics
            ORDER BY recorded_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def save_validation_history(
        self,
        symbol: str,
        is_valid: bool,
        validation_result: Optional[ValidationResult] = None
    ):
        """Save validation attempt for analytics."""
        db = await self._get_conn()

        # Extract check values
        adx_1h = None
        adx_15m = None
        adx_5m = None
        rsi_15m = None
        ema_crosses = None
        vwap_crosses = None
        range_size_pct = None
        failure_reason = None

        if validation_result:
            if validation_result.tf_1h:
                adx_1h = validation_result.tf_1h.checks.get("adx")
                if not validation_result.tf_1h.is_valid:
                    failure_reason = validation_result.tf_1h.reason

            if validation_result.tf_15m:
                adx_15m = validation_result.tf_15m.checks.get("adx")
                rsi_15m = validation_result.tf_15m.checks.get("rsi_mean")
                range_size_pct = validation_result.tf_15m.checks.get("range_size_pct")
                if not validation_result.tf_15m.is_valid and not failure_reason:
                    failure_reason = validation_result.tf_15m.reason

            if validation_result.tf_5m:
                adx_5m = validation_result.tf_5m.checks.get("adx")
                ema_crosses = validation_result.tf_5m.checks.get("ema_crosses")
                vwap_crosses = validation_result.tf_5m.checks.get("vwap_crosses")
                if not validation_result.tf_5m.is_valid and not failure_reason:
                    failure_reason = validation_result.tf_5m.reason

        await db.execute("""
            INSERT INTO validation_history (
                symbol, is_valid, adx_1h, adx_15m, adx_5m,
                rsi_15m, ema_crosses_5m, vwap_crosses_5m,
                range_size_pct, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, 1 if is_valid else 0, adx_1h, adx_15m, adx_5m,
            rsi_15m, ema_crosses, vwap_crosses, range_size_pct,
            failure_reason
        ))

        await db.commit()

    async def get_recent_runs(self, limit: int = 20) -> list[dict]:
        """Get recent bot runs for display."""
        db = await self._get_conn()
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT * FROM bot_runs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_run_with_metrics(self, run_id: int) -> dict[str, Any]:
        """Get a specific run with its metrics."""
        db = await self._get_conn()
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT * FROM bot_runs WHERE id = ?", (run_id,)
        ) as cursor:
            run = await cursor.fetchone()

        async with db.execute(
            "SELECT * FROM bot_metrics WHERE bot_run_id = ?", (run_id,)
        ) as cursor:
            metrics = await cursor.fetchall()

        return {
            "run": dict(run) if run else None,
            "metrics": [dict(m) for m in metrics]
        }

    async def get_validation_stats(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """Get validation statistics."""
        db = await self._get_conn()
        if symbol:
            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(is_valid) as valid_count,
                    AVG(adx_1h) as avg_adx_1h,
                    AVG(adx_15m) as avg_adx_15m,
                    AVG(adx_5m) as avg_adx_5m
                FROM validation_history
                WHERE symbol = ?
            """
            async with db.execute(query, (symbol,)) as cursor:
                row = await cursor.fetchone()
        else:
            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(is_valid) as valid_count,
                    AVG(adx_1h) as avg_adx_1h,
                    AVG(adx_15m) as avg_adx_15m,
                    AVG(adx_5m) as avg_adx_5m
                FROM validation_history
            """
            async with db.execute(query) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return {
                "total": 0,
                "valid_count": 0,
                "valid_rate": 0,
                "avg_adx_1h": None,
                "avg_adx_15m": None,
                "avg_adx_5m": None,
            }

        return {
            "total": row[0],
            "valid_count": row[1],
            "valid_rate": row[1] / row[0] if row[0] > 0 else 0,
            "avg_adx_1h": row[2],
            "avg_adx_15m": row[3],
            "avg_adx_5m": row[4]
        }
