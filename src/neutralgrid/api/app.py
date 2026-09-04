"""
FastAPI application - thin routing layer only.

This module contains ONLY:
- FastAPI app initialization
- Request parsing
- Response formatting
- Delegation to business logic modules

NO business logic should live here.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from pathlib import Path
import json

# Import schemas from new location
from neutralgrid.api.schemas import (
    ValidationRequest,
    ValidationResponse,
    MetricsIngestRequest,
    ExpiredBotRequest,
    ErrorResponse,
)

# Import exception hierarchy
from neutralgrid.core.exceptions import (
    NeutralGridError,
    ModelLoadError,
    InferenceError,
    ValidationPipelineError,
    ArtifactVersionError,
)
from neutralgrid.api.binance_client import BinanceAPIError

# Import logging from new location
from neutralgrid.core.logging import get_logger

# Import utils from new location
from neutralgrid.core.utils import convert_numpy_types

# Config (lightweight, idempotent — just env var reads)
from neutralgrid.core.config import get_config

# Get logger instance
logger = get_logger("api")


def _get_dashboard_path() -> Path:
    """Resolve dashboard path lazily at call time, not import time."""
    return get_config().resolve_path("dashboard")


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup and shutdown hooks."""
    # ----- Startup: all business-logic imports & instance creation -----
    from neutralgrid.api.binance_client import BinanceClient
    from neutralgrid.validation.regime_validator import RegimeValidator
    from neutralgrid.grid.calculator import GridCalculator
    from neutralgrid.storage.database import Database
    from neutralgrid.metrics.calculator import MetricsCalculator
    from neutralgrid.metrics.grid_bot_manager import BinanceGridBotManager

    # Create service instances
    _cfg = get_config()
    binance_client = BinanceClient()
    grid_calculator = GridCalculator()
    database = Database(db_path=str(_cfg.resolve_path(_cfg.database.path)))
    metrics_calculator = MetricsCalculator()

    # Profile artifact loading — hard errors when enabled but missing/corrupt
    _profile_dir = _cfg.resolve_path(_cfg.artifacts.profile_dir)
    _pattern_profile = None
    _profile_model = None
    _gate = None

    if _cfg.validation.enable_profile_gate:
        from neutralgrid.validation.profile_gate import ProfileGate
        from neutralgrid.scanner.pattern_profile import PatternProfile
        from neutralgrid.scanner.profile_model_walkforward import (
            resolve_active_pattern_profile_path,
        )

        pp_path = resolve_active_pattern_profile_path(_profile_dir)
        if not pp_path.exists():
            raise RuntimeError(
                "enable_profile_gate=True but no promoted or bootstrap pattern "
                f"profile is active (resolver returned {pp_path.name})"
            )
        _pattern_profile = PatternProfile.load_json(pp_path)

        gate_path = _profile_dir / "profile_gate.json"
        if gate_path.exists():
            obj = json.loads(gate_path.read_text(encoding="utf-8"))
            _gate = ProfileGate(
                adx_1h_max=float(obj.get("adx_1h_max")),
                adx_15m_max=float(obj.get("adx_15m_max")),
                adx_5m_max=float(obj.get("adx_5m_max")),
                rsi_15m_low=float(obj.get("rsi_15m_low")),
                rsi_15m_high=float(obj.get("rsi_15m_high")),
            )
        else:
            _gate = ProfileGate.from_pattern_profile(_pattern_profile, widen=0.15)

    if _cfg.validation.enable_profile_model:
        from neutralgrid.scanner.profile_model import load_profile_model
        from neutralgrid.scanner.profile_model_walkforward import resolve_active_profile_model_path

        # Route through the resolver: prefer current.json when present, but
        # allow bootstrap use of profile_model.json while no pointer exists yet.
        model_path = resolve_active_profile_model_path(_profile_dir)
        if not model_path.exists():
            raise RuntimeError(
                f"enable_profile_model=True but no promoted or bootstrap profile "
                f"model is active (resolver returned {model_path.name}). Run "
                f"retrain_scanner to create {_profile_dir / 'profile_model.json'} or "
                f"promote_profile_version before starting the API."
            )
        _profile_model = load_profile_model(model_path)
        if (
            _pattern_profile is not None
            and _profile_model.features != _pattern_profile.features
        ):
            raise RuntimeError(
                "Promoted profile bundle feature mismatch: model features "
                f"{_profile_model.features} != pattern features "
                f"{_pattern_profile.features}"
            )

    # Startup validation: check critical paths
    from neutralgrid.models.artifacts import resolve_hmm_artifact_dir
    try:
        _hmm_dir = resolve_hmm_artifact_dir()
        if not (_hmm_dir / "model.joblib").exists():
            logger.warning(
                "HMM artifact not found at %s — regime inference will be unavailable "
                "until model is trained (run neutralgrid-retrain)", _hmm_dir,
            )
    except Exception:
        logger.warning(
            "No HMM artifact directory configured — regime inference will be "
            "unavailable until model is trained (run neutralgrid-retrain)",
        )

    _db_dir = _cfg.resolve_path(_cfg.database.path).parent
    _db_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(str(_db_dir), os.W_OK):
        raise RuntimeError(
            f"Database directory is not writable: {_db_dir}"
        )

    # Initialize validator and bot manager
    validator = RegimeValidator(
        gate=_gate,
        pattern_profile=_pattern_profile,
        profile_model=_profile_model,
    )
    grid_bot_manager = BinanceGridBotManager(binance_client)

    # Store all services on app.state
    app.state.binance_client = binance_client
    app.state.grid_calculator = grid_calculator
    app.state.database = database
    app.state.metrics_calculator = metrics_calculator
    app.state.validator = validator
    app.state.grid_bot_manager = grid_bot_manager

    # Database initialization
    logger.info("=" * 60)
    logger.info("NEUTRAL Grid Bot Validator starting up...")
    await database.initialize()
    logger.info(f"Database initialized at: {database.db_path}")
    logger.info(f"API Authentication: {'ENABLED' if binance_client.is_authenticated else 'DISABLED (public endpoints only)'}")
    logger.info("Server ready to accept requests")
    logger.info("=" * 60)

    yield

    # ----- Shutdown -----
    logger.info("Shutting down...")
    await database.close()
    await binance_client.close()
    logger.info("Server stopped")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="NEUTRAL Grid Bot Validator",
    description="Real-time NEUTRAL Futures Grid Bot Validator for Binance USDT-M",
    version="1.0.0",
    lifespan=lifespan
)

# =============================================================================
# EXCEPTION HANDLERS — structured JSON errors for domain exceptions
# =============================================================================

@app.exception_handler(BinanceAPIError)
async def binance_api_error_handler(request: Request, exc: BinanceAPIError):
    # Rate-limit / IP-ban codes from Binance
    is_rate_limit = exc.status == 429 or exc.code in (-1003, -1015)
    status = 429 if is_rate_limit else 502
    body = ErrorResponse(
        error_code="BINANCE_API_ERROR",
        detail=str(exc),
        status_code=status,
        context={
            "binance_code": exc.code,
            "binance_status": exc.status,
            "endpoint": exc.endpoint,
        },
    )
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ArtifactVersionError)
async def artifact_version_error_handler(request: Request, exc: ArtifactVersionError):
    body = ErrorResponse(
        error_code=exc.error_code,
        detail=str(exc),
        status_code=503,
        context={
            "expected_version": exc.expected,
            "found_version": exc.found,
            "artifact_dir": exc.artifact_dir,
        },
    )
    return JSONResponse(status_code=503, content=body.model_dump())


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    body = ErrorResponse(
        error_code=exc.error_code,
        detail=str(exc),
        status_code=503,
        context={"artifact_path": exc.artifact_path},
    )
    return JSONResponse(status_code=503, content=body.model_dump())


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    body = ErrorResponse(
        error_code=exc.error_code,
        detail=str(exc),
        status_code=500,
        context={"symbol": exc.symbol, "model_type": exc.model_type},
    )
    return JSONResponse(status_code=500, content=body.model_dump())


@app.exception_handler(ValidationPipelineError)
async def validation_pipeline_error_handler(request: Request, exc: ValidationPipelineError):
    body = ErrorResponse(
        error_code=exc.error_code,
        detail=str(exc),
        status_code=500,
        context={"symbol": exc.symbol, "stage": exc.stage},
    )
    return JSONResponse(status_code=500, content=body.model_dump())


@app.exception_handler(NeutralGridError)
async def neutralgrid_error_handler(request: Request, exc: NeutralGridError):
    body = ErrorResponse(
        error_code=getattr(exc, "error_code", "NEUTRALGRID_ERROR"),
        detail=str(exc),
        status_code=500,
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# Mount dashboard static files if present
_dashboard_path = _get_dashboard_path()
if _dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(_dashboard_path)), name="static")


# =============================================================================
# VALIDATION ENDPOINTS
# =============================================================================

@app.post("/api/validate", response_model=ValidationResponse)
async def validate_pair(body: ValidationRequest, request: Request):
    """
    Validate a trading pair for NEUTRAL grid bot deployment.

    Current validation pipeline:
    - HMM regime classification (range vs. trend, using 15m klines)
    - 15M: Range bounds extraction for grid parameterization
    - 5M: NOT IMPLEMENTED (reported as skipped)

    The response truthfully reports:
    - executed_checks: List of checks that actually ran
    - skipped_checks: List of checks not implemented/not run with reasons
    - validation_details: Per-timeframe results with executed=True/False flag

    Returns VALID + grid parameters or INVALID with failure reasons.
    """
    binance_client = request.app.state.binance_client
    validator = request.app.state.validator
    grid_calculator = request.app.state.grid_calculator
    database = request.app.state.database

    symbol = body.symbol.upper()
    logger.info(f"Validation request received for {symbol}")

    try:
        # Step 1: Fetch market data
        market_data = await binance_client.get_all_market_data(symbol)

        # Step 2: Run validation (delegates to RegimeValidator)
        validation_result = validator.validate(market_data)

        # Step 3: Generate grid parameters (delegates to GridCalculator)
        grid_params = grid_calculator.generate_params(
            validation_result,
            range_prob=validation_result.range_prob,
            trend_prob=validation_result.trend_prob,
            survival_prob=validation_result.survival_prob,
            hurst_exponent=validation_result.hurst_exponent,
        )

        # Step 4: Save to database
        await database.save_bot_run(grid_params, validation_result)
        await database.save_validation_history(
            symbol, validation_result.is_valid, validation_result
        )

        # Step 5: Build response with truthful check execution tracking
        executed_checks = []
        skipped_checks = []
        details = {}

        # Track what was actually executed
        if validation_result.tf_1h is not None:
            executed_checks.append("hmm_regime")
            details["hmm_regime"] = {
                "executed": True,
                "passed": validation_result.tf_1h.is_valid,
                "checks": validation_result.tf_1h.checks,
            }
            if not validation_result.tf_1h.is_valid:
                details["hmm_regime"]["reason"] = validation_result.tf_1h.reason

        if validation_result.tf_15m is not None:
            executed_checks.append("15m_range")
            details["15m"] = {
                "executed": True,
                "passed": validation_result.tf_15m.is_valid,
                "checks": validation_result.tf_15m.checks,
            }
            if not validation_result.tf_15m.is_valid:
                details["15m"]["reason"] = validation_result.tf_15m.reason

        # 5m check is not currently implemented
        if validation_result.tf_5m is not None:
            executed_checks.append("5m_validation")
            details["5m"] = {
                "executed": True,
                "passed": validation_result.tf_5m.is_valid,
                "checks": validation_result.tf_5m.checks,
            }
            if not validation_result.tf_5m.is_valid:
                details["5m"]["reason"] = validation_result.tf_5m.reason
        else:
            # Be truthful: 5m check is not implemented
            skipped_checks.append({
                "name": "5m_validation",
                "reason": "not_implemented",
                "note": "5m timeframe validation is not currently implemented in the validation pipeline"
            })
            details["5m"] = {
                "executed": False,
                "passed": None,
                "checks": {},
                "reason": "not_implemented"
            }

        if validation_result.is_valid:
            return ValidationResponse(
                symbol=symbol,
                is_valid=True,
                grid_params=convert_numpy_types(grid_params.to_dict()),
                validation_details=convert_numpy_types(details),
                executed_checks=executed_checks,
                skipped_checks=skipped_checks,
                message="VALID"
            )
        else:
            return ValidationResponse(
                symbol=symbol,
                is_valid=False,
                validation_details=convert_numpy_types(details),
                executed_checks=executed_checks,
                skipped_checks=skipped_checks,
                message="INVALID"
            )

    except NeutralGridError:
        raise  # handled by exception handlers above
    except Exception as e:
        logger.error(f"Unexpected error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/history")
async def get_history(request: Request, limit: int = 20):
    """Get recent validation history."""
    database = request.app.state.database
    runs = await database.get_recent_runs(limit)
    return {"runs": runs}


@app.get("/api/stats")
async def get_stats(request: Request, symbol: Optional[str] = None):
    """Get validation statistics."""
    database = request.app.state.database
    stats = await database.get_validation_stats(symbol)
    return stats


@app.get("/api/status")
async def get_api_status(request: Request):
    """
    Check API connection and authentication status.

    Returns connection info and whether authenticated endpoints are available.
    """
    binance_client = request.app.state.binance_client
    status = await binance_client.check_connection()
    return {
        "connected": status.get("connected", False),
        "authenticated": status.get("authenticated", False),
        "server_time": status.get("server_time"),
        "account_balance": status.get("account_balance"),
        "error": status.get("error"),
        "auth_error": status.get("auth_error"),
    }


@app.get("/api/account")
async def get_account(request: Request):
    """
    Get account information (requires API keys).

    Returns account balance and position information.
    """
    binance_client = request.app.state.binance_client
    if not binance_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
        )

    try:
        balance = await binance_client.get_account_balance()
        positions = await binance_client.get_positions()

        # Filter to only show open positions
        open_positions = [
            p for p in positions
            if float(p.get("positionAmt", 0)) != 0
        ]

        return {
            "balance": balance,
            "open_positions": convert_numpy_types(open_positions),
        }
    except NeutralGridError:
        raise  # handled by exception handlers above
    except Exception as e:
        logger.error(f"Unexpected account fetch error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/run/{run_id}")
async def get_run(run_id: int, request: Request):
    """Get a specific run with metrics."""
    database = request.app.state.database
    data = await database.get_run_with_metrics(run_id)
    if not data["run"]:
        raise HTTPException(status_code=404, detail="Run not found")
    return data


# =============================================================================
# METRICS INGESTION ENDPOINTS
# =============================================================================

@app.post("/api/metrics/ingest")
async def ingest_metrics(body: MetricsIngestRequest, request: Request):
    """
    Ingest metrics for a completed grid bot session.

    Fetches trade fills and income history from Binance,
    calculates performance metrics, and saves to database.
    """
    binance_client = request.app.state.binance_client
    metrics_calculator = request.app.state.metrics_calculator
    database = request.app.state.database

    if not binance_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET."
        )

    try:
        logger.info(f"Ingesting metrics for {body.symbol} from {body.start_time} to {body.end_time}")

        # Fetch session data from Binance
        session_data = await binance_client.get_grid_bot_session_data(
            body.symbol,
            body.start_time,
            body.end_time
        )

        # Calculate metrics (delegates to MetricsCalculator)
        metrics = metrics_calculator.calculate_session_metrics(
            session_data,
            body.grid_params
        )

        # Save to database
        metrics_id = await database.save_session_metrics(
            metrics,
            body.bot_run_id
        )

        logger.info(f"Metrics saved with ID {metrics_id}: NetPnL={metrics.net_pnl}, Fills={metrics.total_fills}")

        return {
            "success": True,
            "metrics_id": metrics_id,
            "summary": {
                "symbol": metrics.symbol,
                "duration_hours": metrics.duration_hours,
                "net_pnl": metrics.net_pnl,
                "gross_pnl": metrics.gross_pnl,
                "total_fees": metrics.total_fees,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "mae": metrics.mae,
                "mfe": metrics.mfe,
                "mae_pct_initial": metrics.mae_pct_initial,
                "mfe_pct_initial": metrics.mfe_pct_initial,
                "total_fills": metrics.total_fills,
                "maker_count": metrics.maker_count,
                "taker_count": metrics.taker_count,
                "notional_completed": metrics.notional_completed,
            }
        }

    except NeutralGridError:
        raise  # handled by exception handlers above
    except Exception as e:
        logger.error(f"Unexpected error for {body.symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/metrics")
async def get_metrics(request: Request, limit: int = 20):
    """Get recent session metrics."""
    database = request.app.state.database
    metrics = await database.get_session_metrics(limit)
    return {"metrics": metrics}


# =============================================================================
# EXPIRED BOT MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/expired-bots")
async def add_expired_bot(body: ExpiredBotRequest, request: Request):
    """
    Add an expired or canceled grid bot for metrics collection.

    Fetches trade history and calculates comprehensive metrics.
    Appends to data/new_expired_bots.csv
    """
    binance_client = request.app.state.binance_client
    grid_bot_manager = request.app.state.grid_bot_manager

    if not binance_client.is_authenticated:
        raise HTTPException(
            status_code=401,
            detail="API keys not configured. Set BINANCE_API_KEY and BINANCE_API_SECRET."
        )

    try:
        logger.info(f"Adding {body.status} bot: {body.symbol} (ID: {body.strategy_id})")

        metrics = await grid_bot_manager.add_expired_bot(
            symbol=body.symbol,
            strategy_id=body.strategy_id,
            status=body.status,
            investment_margin=body.investment_margin,
            leverage=body.leverage,
            grid_low=body.grid_low,
            grid_high=body.grid_high,
            num_grids=body.num_grids,
        )

        logger.info(f"Bot added: TotalProfit={metrics.total_profit}, APY={metrics.apy_pct:.2f}%")

        return {
            "success": True,
            "strategy_id": metrics.strategy_id,
            "summary": {
                "symbol": metrics.symbol,
                "status": metrics.status,
                "total_profit": metrics.total_profit,
                "apy_pct": round(metrics.apy_pct, 2),
                "duration_hours": metrics.duration_hours,
                "matched_profit": metrics.matched_profit,
                "funding_fees": metrics.funding_fees,
                "commissions": metrics.commissions,
                "mae": metrics.mae,
                "mfe": metrics.mfe,
                "total_trades": metrics.total_trades,
            }
        }

    except NeutralGridError:
        raise  # handled by exception handlers above
    except Exception as e:
        logger.error(f"Unexpected error adding expired bot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/expired-bots")
async def get_expired_bots(request: Request):
    """Get list of all tracked expired/canceled bots."""
    grid_bot_manager = request.app.state.grid_bot_manager
    return {"bots": grid_bot_manager.list_bots()}


@app.get("/api/expired-bots/{strategy_id}")
async def get_expired_bot_details(strategy_id: int, request: Request):
    """Get detailed metrics for a specific expired bot."""
    grid_bot_manager = request.app.state.grid_bot_manager
    report = grid_bot_manager.get_bot_report(strategy_id)
    if not report:
        raise HTTPException(status_code=404, detail="Bot not found")
    return report


# =============================================================================
# DASHBOARD SERVING
# =============================================================================

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    index_path = _get_dashboard_path() / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        content={"message": "Dashboard not found. Use /api/validate endpoint."},
        status_code=200
    )
