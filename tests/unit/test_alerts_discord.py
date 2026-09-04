"""Tests for the DiscordWebhookHandler in src/neutralgrid/models/alerts.py."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from neutralgrid.models.alerts import (
    AlertSeverity,
    AlertType,
    DiscordWebhookHandler,
    ModelAlert,
)


def _alert(severity: AlertSeverity = AlertSeverity.WARNING) -> ModelAlert:
    return ModelAlert(
        alert_type=AlertType.MODEL_NOT_FOUND,
        severity=severity,
        message="HMM artifact missing under artifacts/hmm/",
        timestamp_utc=datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).isoformat(),
        context={"artifact_dir": "artifacts/hmm", "model_type": "HMM"},
        error=None,
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    client.post = MagicMock(return_value=response)
    return client


def test_constructor_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        DiscordWebhookHandler("")


def test_handle_posts_one_message_per_alert() -> None:
    client = _mock_client()
    handler = DiscordWebhookHandler("https://discord.test/webhooks/x/y", client=client)
    handler.handle(_alert())
    client.post.assert_called_once()


def test_payload_shape() -> None:
    client = _mock_client()
    handler = DiscordWebhookHandler("https://discord.test/webhooks/x/y", client=client)
    handler.handle(_alert(severity=AlertSeverity.ERROR))

    _, kwargs = client.post.call_args
    payload: dict[str, Any] = kwargs.get("json") or client.post.call_args.kwargs.get("json")
    assert "embeds" in payload
    embed = payload["embeds"][0]
    assert embed["title"].startswith("[ERROR]")
    assert "HMM artifact missing" in embed["description"]
    # ERROR maps to red palette per Discord conventions
    assert embed["color"] == 0xE74C3C
    field_names = {f["name"] for f in embed["fields"]}
    assert "artifact_dir" in field_names
    assert "model_type" in field_names


def test_field_value_truncated_when_too_long() -> None:
    client = _mock_client()
    huge = "x" * 5000
    alert = ModelAlert(
        alert_type=AlertType.INFERENCE_FAILED,
        severity=AlertSeverity.WARNING,
        message="...",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        context={"trace": huge},
    )
    handler = DiscordWebhookHandler("https://discord.test/webhooks/x/y", client=client)
    handler.handle(alert)

    payload = client.post.call_args.kwargs["json"]
    field_value = payload["embeds"][0]["fields"][0]["value"]
    assert len(field_value) <= 1000
    assert field_value.endswith("...")


def test_handle_swallows_post_failure() -> None:
    client = MagicMock()
    client.post.side_effect = RuntimeError("network down")
    handler = DiscordWebhookHandler("https://discord.test/webhooks/x/y", client=client)
    # Must NOT raise -- alerting failures cannot break the caller
    handler.handle(_alert())


def test_raise_on_error_re_raises() -> None:
    client = MagicMock()
    client.post.side_effect = RuntimeError("network down")
    handler = DiscordWebhookHandler(
        "https://discord.test/webhooks/x/y", client=client, raise_on_error=True
    )
    with pytest.raises(RuntimeError):
        handler.handle(_alert())


def test_severity_color_mapping() -> None:
    severities = {
        AlertSeverity.INFO: 0x3498DB,
        AlertSeverity.WARNING: 0xF1C40F,
        AlertSeverity.ERROR: 0xE74C3C,
        AlertSeverity.CRITICAL: 0x992D22,
    }
    for sev, expected_color in severities.items():
        client = _mock_client()
        handler = DiscordWebhookHandler("https://discord.test/webhooks/x/y", client=client)
        handler.handle(_alert(severity=sev))
        assert client.post.call_args.kwargs["json"]["embeds"][0]["color"] == expected_color
