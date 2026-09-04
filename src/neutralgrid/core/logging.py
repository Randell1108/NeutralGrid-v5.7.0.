"""
Structured logging configuration for NEUTRAL Grid Bot.

Provides consistent logging across all modules with:
- Console output for development
- Rotating file logs for production
- Error-specific logging
- Structured context (run_id, symbol, artifact_version)
"""

import logging
import threading
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
from contextvars import ContextVar


# Context variables for structured logging
_run_id_var: ContextVar[Optional[str]] = ContextVar("run_id", default=None)
_symbol_var: ContextVar[Optional[str]] = ContextVar("symbol", default=None)
_timeframe_var: ContextVar[Optional[str]] = ContextVar("timeframe", default=None)
_artifact_version_var: ContextVar[Optional[str]] = ContextVar("artifact_version", default=None)


class StructuredFormatter(logging.Formatter):
    """
    Formatter that includes structured context in log messages.

    Adds run_id, symbol, timeframe, and artifact_version to logs when available.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Get context values
        run_id = _run_id_var.get()
        symbol = _symbol_var.get()
        timeframe = _timeframe_var.get()
        artifact_version = _artifact_version_var.get()

        # Build context string
        context_parts = []
        if run_id:
            context_parts.append(f"run_id={run_id}")
        if symbol:
            context_parts.append(f"symbol={symbol}")
        if timeframe:
            context_parts.append(f"tf={timeframe}")
        if artifact_version:
            context_parts.append(f"ver={artifact_version}")

        context = f" | {', '.join(context_parts)}" if context_parts else ""

        # Work on a copy to avoid mutating the shared record for other handlers
        import copy
        record = copy.copy(record)
        original_msg = record.getMessage()
        record.msg = f"{original_msg}{context}"
        record.args = ()

        return super().format(record)


def setup_logging(
    log_dir: str = "logs",
    logger_name: str = "neutralgrid",
    level: int = logging.INFO,
    use_structured: bool = True,
) -> logging.Logger:
    """
    Configure application logging with optional structured logging.

    Args:
        log_dir: Directory for log files (created if doesn't exist)
        logger_name: Name for the logger
        level: Logging level (default: INFO)
        use_structured: Use structured logging with context (default: True)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_path = Path(log_dir)
    logs_path.mkdir(exist_ok=True)

    # Choose formatter
    if use_structured:
        log_format = StructuredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        log_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # File handler for errors (rotating, max 5MB, keep 5 backups)
    error_log_path = logs_path / "errors.log"
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)

    # File handler for all logs (rotating, max 10MB, keep 3 backups)
    all_log_path = logs_path / "validator.log"
    file_handler = RotatingFileHandler(
        all_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_format)

    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Clear existing handlers
    logger.handlers.clear()
    logger.addHandler(error_handler)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Also configure uvicorn loggers to use same handlers
    logging.getLogger("uvicorn.error").handlers = [console_handler, file_handler]
    logging.getLogger("uvicorn.access").handlers = [console_handler, file_handler]

    return logger


# Singleton logger instance
_logger: logging.Logger | None = None
_logger_lock = threading.Lock()


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get or create application logger.

    Args:
        name: Optional logger name (default: "neutralgrid")

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                from neutralgrid.core.config import get_config
                _cfg = get_config()
                _resolved_log_dir = str(_cfg.resolve_path(_cfg.logging.log_dir))
                _logger = setup_logging(
                    log_dir=_resolved_log_dir,
                    logger_name="neutralgrid",
                )
    return _logger if name is None else _logger.getChild(name)


# Context management functions

def set_run_id(run_id: Optional[str] = None) -> str:
    """
    Set run ID for structured logging.

    Args:
        run_id: Run ID (generates UUID if None)

    Returns:
        The run ID that was set
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    _run_id_var.set(run_id)
    return run_id


def set_symbol(symbol: str):
    """Set symbol for structured logging."""
    _symbol_var.set(symbol)


def set_timeframe(timeframe: str):
    """Set timeframe for structured logging."""
    _timeframe_var.set(timeframe)


def set_artifact_version(version: str):
    """Set artifact version for structured logging."""
    _artifact_version_var.set(version)


def clear_context():
    """Clear all structured logging context."""
    _run_id_var.set(None)
    _symbol_var.set(None)
    _timeframe_var.set(None)
    _artifact_version_var.set(None)


def get_context() -> Dict[str, Any]:
    """
    Get current logging context.

    Returns:
        Dictionary with current context values
    """
    return {
        "run_id": _run_id_var.get(),
        "symbol": _symbol_var.get(),
        "timeframe": _timeframe_var.get(),
        "artifact_version": _artifact_version_var.get(),
    }
