"""
Alerting hooks for model load failures and inference errors.

AFML Compliance:
Production models must have proper alerting when they fail to load or predict.
Silent failures can lead to stale model usage or missing predictions without
operators being aware.

Design principles:
- Pluggable alert handlers (logging, webhook, email, etc.)
- Severity levels for different failure types
- Context-rich alert messages for debugging
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of model-related alerts."""
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    SCHEMA_MISMATCH = "schema_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    INFERENCE_FAILED = "inference_failed"
    STALE_MODEL = "stale_model"
    DATA_QUALITY = "data_quality"


@dataclass
class ModelAlert:
    """Alert payload for model-related issues."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp_utc: str
    context: Dict[str, Any]
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp_utc": self.timestamp_utc,
            "context": self.context,
            "error": self.error,
            "error_type": self.error_type,
        }


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    @abstractmethod
    def handle(self, alert: ModelAlert) -> None:
        """Handle an alert."""
        pass


class LoggingAlertHandler(AlertHandler):
    """
    Default alert handler that logs to Python logging.

    This ensures alerts are visible in application logs even if no
    external alerting system is configured.
    """

    def __init__(self, logger_name: Optional[str] = None):
        self.logger = logging.getLogger(logger_name or __name__)

    def handle(self, alert: ModelAlert) -> None:
        """Log alert at appropriate level."""
        log_msg = (
            f"[{alert.alert_type.value}] {alert.message} | "
            f"context={alert.context}"
        )
        if alert.error:
            log_msg += f" | error={alert.error}"

        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(log_msg)
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)


class CallbackAlertHandler(AlertHandler):
    """
    Alert handler that calls a custom callback function.

    Useful for:
    - Sending webhooks to Slack/Discord/PagerDuty
    - Writing to external monitoring systems
    - Custom alerting logic
    """

    def __init__(self, callback: Callable[[ModelAlert], None]):
        self.callback = callback

    def handle(self, alert: ModelAlert) -> None:
        """Invoke callback with alert."""
        try:
            self.callback(alert)
        except Exception as e:
            # Don't let callback errors propagate
            logger.error(f"Alert callback failed: {e}")


# Discord embed colors keyed on AlertSeverity. Hex values are the conventional
# Discord palette (info-blue / warn-yellow / error-red / critical-dark-red).
_DISCORD_COLOR_BY_SEVERITY: Dict[AlertSeverity, int] = {
    AlertSeverity.INFO: 0x3498DB,
    AlertSeverity.WARNING: 0xF1C40F,
    AlertSeverity.ERROR: 0xE74C3C,
    AlertSeverity.CRITICAL: 0x992D22,
}


class DiscordWebhookHandler(AlertHandler):
    """
    Alert handler that posts ModelAlerts to a Discord webhook as embeds.

    Synchronous: uses ``httpx.Client`` so it can be invoked from any thread.
    For per-tick decision digests (which need async + rate-limiting), see
    ``neutralgrid.live.decision.discord_sink.DiscordDigestSink``.

    On webhook failures (network error, non-2xx response) the handler logs and
    swallows the exception -- alerting must never break the caller. Set
    ``raise_on_error=True`` only in tests.
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        timeout_seconds: float = 5.0,
        raise_on_error: bool = False,
        client: Optional[Any] = None,
    ):
        if not webhook_url:
            raise ValueError("webhook_url must be a non-empty string")
        self.webhook_url = webhook_url
        self._timeout = timeout_seconds
        self._raise_on_error = raise_on_error
        self._client = client  # injected for tests

    def handle(self, alert: ModelAlert) -> None:
        try:
            payload = self._build_payload(alert)
            self._post(payload)
        except Exception as e:
            logger.error("Discord webhook delivery failed: %s", e)
            if self._raise_on_error:
                raise

    def _build_payload(self, alert: ModelAlert) -> Dict[str, Any]:
        color = _DISCORD_COLOR_BY_SEVERITY.get(alert.severity, 0x95A5A6)
        fields: List[Dict[str, Any]] = []
        for k, v in (alert.context or {}).items():
            value_str = str(v)
            # Discord field values cap at 1024 chars
            if len(value_str) > 1000:
                value_str = value_str[:997] + "..."
            fields.append({"name": str(k), "value": value_str, "inline": True})
        if alert.error:
            err_str = alert.error
            if len(err_str) > 1000:
                err_str = err_str[:997] + "..."
            fields.append({"name": "error", "value": err_str, "inline": False})
        return {
            "embeds": [
                {
                    "title": f"[{alert.severity.value.upper()}] {alert.alert_type.value}",
                    "description": alert.message,
                    "color": color,
                    "timestamp": alert.timestamp_utc,
                    "fields": fields,
                    "footer": {"text": "neutralgrid.alerts"},
                }
            ]
        }

    def _post(self, payload: Dict[str, Any]) -> None:
        # Imported lazily so importing alerts.py doesn't pull httpx for callers
        # that don't use the Discord handler.
        import httpx

        client = self._client
        own_client = client is None
        if client is None:
            client = httpx.Client(timeout=self._timeout)
        try:
            resp = client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
        finally:
            if own_client:
                client.close()


class AlertManager:
    """
    Central manager for model alerts.

    Maintains a list of handlers and dispatches alerts to all of them.
    By default, includes a logging handler so alerts are never lost.
    """

    _instance: Optional["AlertManager"] = None
    _handlers: List[AlertHandler]

    def __init__(self):
        self._handlers = [LoggingAlertHandler()]

    @classmethod
    def get_instance(cls) -> "AlertManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def clear_handlers(self) -> None:
        """Remove all handlers (except default logger)."""
        self._handlers = [LoggingAlertHandler()]

    def alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> ModelAlert:
        """
        Send an alert to all handlers.

        Args:
            alert_type: Type of alert
            severity: Alert severity level
            message: Human-readable message
            context: Additional context data
            error: Optional exception that caused the alert

        Returns:
            The ModelAlert that was sent
        """
        alert = ModelAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            context=context or {},
            error=str(error) if error else None,
            error_type=type(error).__name__ if error else None,
        )

        for handler in self._handlers:
            try:
                handler.handle(alert)
            except Exception as e:
                # Log handler errors but don't fail
                logger.error(f"Alert handler {type(handler).__name__} failed: {e}")

        return alert


# Convenience functions for common alert patterns


def alert_model_not_found(
    artifact_dir: str,
    model_type: str = "HMM",
) -> ModelAlert:
    """Alert when model artifact is not found."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.MODEL_NOT_FOUND,
        severity=AlertSeverity.WARNING,
        message=f"{model_type} model artifact not found",
        context={
            "artifact_dir": str(artifact_dir),
            "model_type": model_type,
        },
    )


def alert_model_load_failed(
    artifact_dir: str,
    error: Exception,
    model_type: str = "HMM",
) -> ModelAlert:
    """Alert when model loading fails."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.MODEL_LOAD_FAILED,
        severity=AlertSeverity.ERROR,
        message=f"Failed to load {model_type} model",
        context={
            "artifact_dir": str(artifact_dir),
            "model_type": model_type,
        },
        error=error,
    )


def alert_schema_mismatch(
    expected_features: List[str],
    found_features: List[str],
    model_type: str = "HMM",
) -> ModelAlert:
    """Alert when feature schema doesn't match."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.SCHEMA_MISMATCH,
        severity=AlertSeverity.ERROR,
        message=f"{model_type} feature schema mismatch - potential training/inference skew",
        context={
            "expected_features": expected_features,
            "found_features": found_features,
            "model_type": model_type,
        },
    )


def alert_version_mismatch(
    expected_version: str,
    found_version: str,
    artifact_dir: str,
) -> ModelAlert:
    """Alert when model version doesn't match expected."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.VERSION_MISMATCH,
        severity=AlertSeverity.WARNING,
        message="Model version mismatch",
        context={
            "expected_version": expected_version,
            "found_version": found_version,
            "artifact_dir": str(artifact_dir),
        },
    )


def alert_inference_failed(
    symbol: str,
    error: Exception,
    model_type: str = "HMM",
) -> ModelAlert:
    """Alert when inference fails."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.INFERENCE_FAILED,
        severity=AlertSeverity.WARNING,
        message=f"{model_type} inference failed for {symbol}",
        context={
            "symbol": symbol,
            "model_type": model_type,
        },
        error=error,
    )


def alert_stale_model(
    trained_at_utc: str,
    staleness_days: float,
    threshold_days: float,
    model_type: str = "HMM",
) -> ModelAlert:
    """Alert when model is stale (hasn't been retrained recently)."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.STALE_MODEL,
        severity=AlertSeverity.WARNING,
        message=f"{model_type} model is {staleness_days:.1f} days old (threshold: {threshold_days} days)",
        context={
            "trained_at_utc": trained_at_utc,
            "staleness_days": staleness_days,
            "threshold_days": threshold_days,
            "model_type": model_type,
        },
    )


def alert_data_quality(
    message: str,
    context: Dict[str, Any],
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> ModelAlert:
    """Alert for data quality issues."""
    return AlertManager.get_instance().alert(
        alert_type=AlertType.DATA_QUALITY,
        severity=severity,
        message=message,
        context=context,
    )
