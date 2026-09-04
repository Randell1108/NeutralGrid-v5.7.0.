from __future__ import annotations

import logging

import pytest

from run_full_pipeline import _close_client_safely


class _ClosingClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


@pytest.mark.asyncio
async def test_close_client_safely_closes_client() -> None:
    client = _ClosingClient()

    await _close_client_safely(client)

    assert client.closed is True


@pytest.mark.asyncio
async def test_close_client_safely_preserves_primary_failure(caplog: pytest.LogCaptureFixture) -> None:
    client = _ClosingClient(fail=True)
    caplog.set_level(logging.WARNING, logger="full_pipeline")

    await _close_client_safely(client)

    assert client.closed is True
    assert "Failed to close Binance client cleanly" in caplog.text


def test_http_request_line_loggers_do_not_emit_info() -> None:
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() >= logging.WARNING
