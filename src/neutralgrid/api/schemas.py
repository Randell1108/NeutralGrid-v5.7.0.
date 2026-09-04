"""
API Request and Response Schemas.

Pydantic models for FastAPI endpoints.
Separated from business logic for clarity and reusability.
"""

from pydantic import BaseModel
from typing import Any, Optional


class ValidationRequest(BaseModel):
    """Request to validate a trading pair."""
    symbol: str


class ValidationResponse(BaseModel):
    """
    Response from validation endpoint.

    Truthfully reports which checks were executed and which were skipped.
    See API_TRUTHFULNESS.md for details.
    """
    symbol: str
    is_valid: bool
    grid_params: Optional[dict[str, Any]] = None
    validation_details: Optional[dict[str, Any]] = None
    executed_checks: list[str] = []  # List of checks that were actually run
    skipped_checks: list[dict[str, Any]] = []  # List of checks skipped with reasons
    message: str


class MetricsIngestRequest(BaseModel):
    """Request model for metrics ingestion."""
    symbol: str
    start_time: int  # Milliseconds since epoch
    end_time: int    # Milliseconds since epoch
    bot_run_id: Optional[int] = None
    grid_params: Optional[dict[str, Any]] = None  # Optional grid params used


class ExpiredBotRequest(BaseModel):
    """Request model for adding an expired/canceled bot."""
    symbol: str
    strategy_id: int
    status: str  # "expired" or "cancelled"
    investment_margin: float
    leverage: int = 1
    grid_low: Optional[float] = None
    grid_high: Optional[float] = None
    num_grids: Optional[int] = None


class ErrorResponse(BaseModel):
    """Structured error returned by all API endpoints."""
    error_code: str          # machine-readable: "DATA_FETCH_ERROR", "MODEL_LOAD_FAILED", etc.
    detail: str              # human-readable message
    status_code: int         # HTTP status (mirrors the actual response status)
    context: dict[str, Any] = {}  # error-type-specific: {symbol, binance_code, artifact_path, ...}
