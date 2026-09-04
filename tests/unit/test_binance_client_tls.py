"""TLS transport tests for BinanceClient."""

from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

import pytest

from neutralgrid.api import binance_client
from neutralgrid.api.binance_client import BinanceClient


@pytest.mark.asyncio
async def test_client_uses_system_trust_store_context() -> None:
    """The Binance client must retain TLS verification via Windows CryptoAPI."""
    ssl_context = MagicMock()
    http_client = MagicMock()
    http_client.is_closed = False

    with patch.object(
        binance_client.truststore,
        "SSLContext",
        return_value=ssl_context,
    ) as ssl_context_factory, patch.object(
        binance_client.httpx,
        "AsyncClient",
        return_value=http_client,
    ) as async_client:
        client = BinanceClient(api_key="key", api_secret="secret")
        result = await client._get_client()

    assert result is http_client
    ssl_context_factory.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    async_client.assert_called_once_with(
        timeout=30.0,
        headers={"X-MBX-APIKEY": "key"},
        verify=ssl_context,
    )
