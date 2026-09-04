"""
Binance Futures USDT-M API Client.
Fetches real-time market data for regime validation and grid parameterization.
Supports both public and authenticated (signed) API requests.
"""

import asyncio
import hmac
import hashlib
import logging
import ssl
import time
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import truststore

from neutralgrid.core.config import get_config

logger = logging.getLogger(__name__)


class BinanceAPIError(Exception):
    """Structured error from Binance API."""

    def __init__(self, status: int, code: int | None, msg: str, endpoint: str, params_snippet: str):
        self.status = status
        self.code = code          # Binance error code (-1021, -1003, etc.) or None
        self.msg = msg
        self.endpoint = endpoint
        self.params_snippet = params_snippet  # first 200 chars of params repr
        super().__init__(f"[{status}] {code}: {msg} | {endpoint}")

    @property
    def is_ip_ban(self) -> bool:
        return self.status == 418

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429

    @property
    def is_timestamp_error(self) -> bool:
        return self.code in (-1021, -1022)


class BinanceClient:
    """Async client for Binance Futures USDT-M API with authentication support."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Binance Futures client.
        
        Args:
            api_key: Binance API key (optional, uses config if not provided)
            api_secret: Binance API secret (optional, uses config if not provided)
            base_url: Binance Futures API base URL (optional, uses config if not provided)
        """
        cfg = get_config()
        self._cfg = cfg
        self.base_url = base_url or cfg.binance.futures_base_url
        self.api_key = api_key or cfg.binance.api_key
        self.api_secret = api_secret or cfg.binance.api_secret
        self._client: Optional[httpx.AsyncClient] = None
        self._time_offset_ms: int = 0
        self._circuit_open_until: float = 0.0  # time.monotonic() when circuit reopens
        # Rate-limit tracking: sliding window of (monotonic_ts, weight) tuples
        self._weight_log: list[tuple[float, int]] = []
        self._weight_limit: int = 1200  # Binance 1-minute weight cap
        self._weight_warn: int = 1000   # Start sleeping when approaching limit
        self._weight_lock = asyncio.Lock()

    @property
    def is_authenticated(self) -> bool:
        """Check if API keys are configured."""
        return bool(self.api_key and self.api_secret)

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds, adjusted by server time offset."""
        return int(time.time() * 1000) + self._time_offset_ms

    def _generate_signature(self, query_string: str) -> str:
        """
        Generate HMAC-SHA256 signature for authenticated requests.
        
        Args:
            query_string: The query string to sign
            
        Returns:
            Hex-encoded signature
        """
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with API key headers."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            # Use the operating system trust store. The default certifi bundle
            # cannot validate every chain accepted by the local system store.
            # Verification remains on.
            ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers,
                verify=ssl_context,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Time-sync
    # ------------------------------------------------------------------

    async def sync_time(self) -> int:
        """Fetch Binance server time and compute local clock offset."""
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/fapi/v1/time")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "serverTime" not in data:
            raise ValueError(f"Invalid time sync response: missing 'serverTime' key")
        server_time = int(data["serverTime"])
        local_time = int(time.time() * 1000)
        self._time_offset_ms = server_time - local_time
        logger.info("Time sync: offset=%dms", self._time_offset_ms)
        return self._time_offset_ms

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _check_circuit(self, endpoint: str) -> None:
        """Raise if circuit breaker is open (IP banned)."""
        remaining = self._circuit_open_until - time.monotonic()
        if remaining > 0:
            raise BinanceAPIError(
                status=418, code=None,
                msg=f"Circuit breaker open, retry in {remaining:.0f}s",
                endpoint=endpoint, params_snippet="",
            )

    def _trip_circuit(self, retry_after: int | None) -> None:
        """Trip the circuit breaker on 418 IP ban."""
        wait = retry_after or 120  # default 2 minutes if no Retry-After header
        self._circuit_open_until = time.monotonic() + wait
        logger.error("Circuit breaker tripped for %ds (IP ban)", wait)

    # ------------------------------------------------------------------
    # Rate-limit tracking
    # ------------------------------------------------------------------

    async def _check_rate_limit(self, weight: int = 1) -> None:
        """Prune stale entries and block until room is available in the 1-minute budget."""
        while True:
            sleep_for = 0.0
            current = 0
            async with self._weight_lock:
                now = time.monotonic()
                cutoff = now - 60.0
                self._weight_log = [(ts, w) for ts, w in self._weight_log if ts > cutoff]
                current = sum(w for _, w in self._weight_log)

                if current + weight < self._weight_warn:
                    self._weight_log.append((now, weight))
                    return

                # Wait until the oldest in-window event falls out.
                if self._weight_log:
                    sleep_for = max(self._weight_log[0][0] - cutoff, 0.5)
                else:
                    sleep_for = 1.0

            logger.warning(
                "Rate limit approaching (%d/%d), sleeping %.1fs",
                current, self._weight_limit, sleep_for,
            )
            await asyncio.sleep(sleep_for)

    async def _update_weight_from_response(self, response: httpx.Response) -> None:
        """Parse X-MBX-USED-WEIGHT-1M and reconcile local tracking with server state."""
        header = response.headers.get("X-MBX-USED-WEIGHT-1M")
        if header is not None:
            try:
                server_weight = int(header)
                async with self._weight_lock:
                    now = time.monotonic()
                    cutoff = now - 60.0
                    self._weight_log = [(ts, w) for ts, w in self._weight_log if ts > cutoff]
                    local_weight = sum(w for _, w in self._weight_log)
                    if server_weight > local_weight:
                        diff = server_weight - local_weight
                        self._weight_log.append((now, diff))
                    elif server_weight < local_weight:
                        # Server header is authoritative for current 1-minute usage.
                        self._weight_log = [(now, server_weight)] if server_weight > 0 else []
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Centralized request logic
    # ------------------------------------------------------------------

    # Default endpoint weights (approximate Binance API weights)
    _ENDPOINT_WEIGHTS: dict[str, int] = {
        "/fapi/v1/klines": 5,
        "/fapi/v1/depth": 5,
        "/fapi/v1/ticker/24hr": 1,
        "/fapi/v1/exchangeInfo": 10,
        "/fapi/v1/openInterest": 1,
        "/fapi/v1/premiumIndex": 1,
        "/fapi/v1/fundingRate": 1,
        "/fapi/v1/markPriceKlines": 5,
    }

    async def _do_request(
        self,
        method: str,
        endpoint: str,
        params: dict | None,
        signed: bool,
        max_retries: int,
    ) -> Any:
        """
        Execute a GET or POST request with retry, circuit breaker, rate-limit, and time-sync.

        Handles:
        - Rate-limit tracking (sliding 1-minute weight window).
        - 418 IP ban → trips circuit breaker, raises immediately.
        - 429 / 5xx → exponential backoff retry (respects Retry-After).
        - -1021/-1022 timestamp errors → sync_time() then retry (once).
        - Connection/timeout errors → backoff retry.
        - Other 4xx → raises immediately.
        """
        self._check_circuit(endpoint)
        weight = self._ENDPOINT_WEIGHTS.get(endpoint, 1)
        await self._check_rate_limit(weight)

        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"

        if params is None:
            params = {}

        params_snippet = repr(params)[:200]
        last_exc: Exception | None = None
        time_synced = False  # only allow one auto-sync per call

        for attempt in range(max_retries + 1):
            try:
                request_params = dict(params)

                if signed:
                    if not self.is_authenticated:
                        raise ValueError("API keys required for authenticated requests")
                    request_params["recvWindow"] = 10000
                    request_params["timestamp"] = self._get_timestamp()
                    query_string = urlencode(request_params)
                    request_params["signature"] = self._generate_signature(query_string)

                if method == "POST":
                    response = await client.post(url, params=request_params)
                else:
                    response = await client.get(url, params=request_params)

                await self._update_weight_from_response(response)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # Try to parse Binance error body
                binance_code = None
                binance_msg = ""
                try:
                    body = e.response.json()
                    binance_code = body.get("code")
                    binance_msg = body.get("msg", "")
                except Exception:
                    binance_msg = e.response.text[:300]

                # 418 — IP ban: trip circuit breaker, do NOT retry
                if status == 418:
                    retry_after = None
                    ra_header = e.response.headers.get("Retry-After")
                    if ra_header:
                        try:
                            retry_after = int(ra_header)
                        except ValueError:
                            pass
                    self._trip_circuit(retry_after)
                    raise BinanceAPIError(
                        status=status, code=binance_code, msg=binance_msg,
                        endpoint=endpoint, params_snippet=params_snippet,
                    ) from e

                # Timestamp drift — sync once then retry
                if binance_code in (-1021, -1022) and not time_synced:
                    time_synced = True
                    logger.warning(
                        "Timestamp error (%s) on %s, syncing time and retrying",
                        binance_code, endpoint,
                    )
                    try:
                        await self.sync_time()
                    except Exception:
                        logger.warning("Time sync failed, continuing with retries")
                    continue

                # 429 or 5xx — backoff + retry
                if status == 429 or status >= 500:
                    await self._update_weight_from_response(e.response)
                    last_exc = BinanceAPIError(
                        status=status, code=binance_code, msg=binance_msg,
                        endpoint=endpoint, params_snippet=params_snippet,
                    )
                    if attempt < max_retries:
                        # Respect Retry-After header on 429
                        backoff = min(2 ** attempt, 30)
                        if status == 429:
                            ra_header = e.response.headers.get("Retry-After")
                            if ra_header:
                                try:
                                    backoff = max(backoff, int(ra_header))
                                except ValueError:
                                    pass
                        logger.warning(
                            "API %s returned %d, retrying in %ds (attempt %d/%d)",
                            endpoint, status, backoff, attempt + 1, max_retries,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise last_exc from e

                # Other 4xx — no retry
                raise BinanceAPIError(
                    status=status, code=binance_code, msg=binance_msg,
                    endpoint=endpoint, params_snippet=params_snippet,
                ) from e

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                last_exc = e
                if attempt < max_retries:
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "API %s connection error: %s, retrying in %ds (attempt %d/%d)",
                        endpoint, type(e).__name__, backoff, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(backoff)
                    continue
                raise

        if last_exc is None:
            raise RuntimeError(f"Unexpected: request loop completed without result for {endpoint}")
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public wrappers (backward-compatible signatures)
    # ------------------------------------------------------------------

    async def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        signed: bool = False,
        max_retries: int = 3,
    ) -> Any:
        """Make a GET request to the API with retry on transient failures."""
        return await self._do_request("GET", endpoint, params, signed, max_retries)

    async def _post_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        signed: bool = True,
        max_retries: int = 3,
    ) -> Any:
        """Make a POST request to the API with retry on transient failures."""
        return await self._do_request("POST", endpoint, params, signed, max_retries)

    # =========================================================================
    # PUBLIC ENDPOINTS (No authentication required)
    # =========================================================================

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        include_current: bool = False
    ) -> list[list]:
        """
        Fetch kline/candlestick data with optional current forming candle.

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
            limit: Number of klines to fetch
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            include_current: If True, includes the current forming candle using real-time price

        Returns list of:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }

        # Add optional time parameters
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        klines = await self._request(self._cfg.binance.endpoints["klines"], params)
        
        # The last kline from the API may be the current forming candle.
        # Drop it when include_current=False to avoid incomplete-bar contamination.
        if not include_current and klines and len(klines[-1]) > 6:
            last_close_time = int(klines[-1][6])
            if last_close_time > self._get_timestamp():
                klines = klines[:-1]
        elif include_current and klines and len(klines[-1]) > 4:
            ticker = await self.get_ticker(symbol)
            if ticker:
                current_price = float(ticker.get("lastPrice", 0))
                _current_high = float(ticker.get("highPrice", 0))
                _current_low = float(ticker.get("lowPrice", 0))
                
                # Update the last candle with real-time data
                # The last kline's close should be the current price
                last_kline = klines[-1]
                last_kline[4] = str(current_price)  # Update close to current price
                
                # Update high if current price is higher
                if current_price > float(last_kline[2]):
                    last_kline[2] = str(current_price)
                
                # Update low if current price is lower
                if current_price < float(last_kline[3]):
                    last_kline[3] = str(current_price)
        
        return klines

    async def get_mark_price_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        include_current: bool = False,
    ) -> list[list]:
        """
        Fetch mark-price kline data.

        Same schema and incomplete-bar logic as ``get_klines`` but uses the
        ``/fapi/v1/markPriceKlines`` endpoint which returns candles based on
        the mark price rather than the last traded price.

        Returns list of:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        params: dict[str, str | int] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        klines = await self._request(
            self._cfg.binance.endpoints["mark_price_klines"], params,
        )

        # Drop incomplete (still-forming) bar — same logic as get_klines
        if not include_current and klines and len(klines[-1]) > 6:
            last_close_time = int(klines[-1][6])
            if last_close_time > self._get_timestamp():
                klines = klines[:-1]

        return klines

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get current real-time ticker data for a symbol.
        
        Returns: {"symbol", "lastPrice", "highPrice", "lowPrice", "volume", etc.}
        """
        params = {"symbol": symbol.upper()}
        return await self._request(self._cfg.binance.endpoints["ticker"], params)

    async def get_open_interest(self, symbol: str) -> dict[str, Any]:
        """
        Get current open interest for a symbol.
        
        Returns: {"symbol": "BTCUSDT", "openInterest": "12345.678", "time": 1234567890}
        """
        params = {"symbol": symbol.upper()}
        return await self._request(self._cfg.binance.endpoints["open_interest"], params)

    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict]:
        """
        Get historical open interest statistics.
        
        Period options: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        """
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["open_interest_hist"], params)

    async def get_funding_rate(
        self,
        symbol: str,
        limit: int = 10,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[dict]:
        """
        Get funding rate history.

        Args:
            symbol: Trading pair symbol
            limit: Number of records (default 10, max 1000)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)

        Returns list of: {"symbol", "fundingRate", "fundingTime", "markPrice"}
        """
        params = {
            "symbol": symbol.upper(),
            "limit": limit
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return await self._request(self._cfg.binance.endpoints["funding_rate"], params)

    async def get_premium_index(self, symbol: str) -> dict[str, Any]:
        """
        Get premium index (mark price, index price, settlement price).
        Used for basis calculation.
        
        Returns: {"symbol", "markPrice", "indexPrice", "estimatedSettlePrice", 
                  "lastFundingRate", "nextFundingTime", "interestRate"}
        """
        params = {"symbol": symbol.upper()}
        return await self._request(self._cfg.binance.endpoints["premium_index"], params)

    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20
    ) -> dict[str, Any]:
        """
        Get order book depth.
        
        Returns: {"lastUpdateId", "bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        params = {
            "symbol": symbol.upper(),
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["depth"], params)

    async def get_global_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict]:
        """
        Get long/short account ratio for all traders.
        
        Returns list of: {"symbol", "longShortRatio", "longAccount", "shortAccount", "timestamp"}
        """
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["global_long_short_ratio"], params)

    async def get_top_long_short_account_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict]:
        """
        Get long/short ratio of top traders (by account).
        """
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["top_long_short_account"], params)

    async def get_top_long_short_position_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict]:
        """
        Get long/short ratio of top traders (by position).
        """
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["top_long_short_position"], params)

    async def get_taker_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> list[dict]:
        """
        Get taker buy/sell volume ratio.
        
        Returns list of: {"buySellRatio", "buyVol", "sellVol", "timestamp"}
        """
        params = {
            "symbol": symbol.upper(),
            "period": period,
            "limit": limit
        }
        return await self._request(self._cfg.binance.endpoints["taker_long_short"], params)

    async def get_exchange_info(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Returns: {"timezone", "serverTime", "rateLimits", "exchangeFilters", "symbols"}
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self._request(self._cfg.binance.endpoints["exchange_info"], params)

    # =========================================================================
    # AUTHENTICATED ENDPOINTS (Require API keys)
    # =========================================================================

    async def get_account_info(self) -> dict[str, Any]:
        """
        Get current account information (requires authentication).
        
        Returns account balances, positions, and margin info.
        """
        return await self._request(self._cfg.binance.endpoints["account"], signed=True)

    async def get_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Get current position risk information (requires authentication).
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns list of position data with unrealized PnL, leverage, etc.
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self._request(self._cfg.binance.endpoints["positions"], params, signed=True)

    async def get_account_balance(self) -> dict[str, Any]:
        """
        Get account balance summary (requires authentication).
        
        Returns dict with total wallet balance, available balance, etc.
        """
        account = await self.get_account_info()
        return {
            "totalWalletBalance": float(account.get("totalWalletBalance", 0)),
            "totalUnrealizedProfit": float(account.get("totalUnrealizedProfit", 0)),
            "totalMarginBalance": float(account.get("totalMarginBalance", 0)),
            "availableBalance": float(account.get("availableBalance", 0)),
            "maxWithdrawAmount": float(account.get("maxWithdrawAmount", 0)),
        }

    async def get_user_trades(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
        from_id: Optional[int] = None,
    ) -> list[dict]:
        """
        Get account trade list for a symbol (requires authentication).
        
        Used to fetch grid bot fills and calculate metrics.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Max number of trades (default 1000, max 1000)
            from_id: Trade ID to fetch from (cannot be combined with time range
                on Binance; retained for compatibility with direct API semantics)
        
        Returns list of trades with:
        {id, symbol, orderId, side, price, qty, realizedPnl, 
         marginAsset, quoteQty, commission, commissionAsset,
         time, positionSide, maker, buyer}
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        if from_id is not None:
            params["fromId"] = from_id
        return await self._request(self._cfg.binance.endpoints["user_trades"], params, signed=True)

    async def get_user_trades_all(
        self,
        symbol: str,
        start_time: int,
        end_time: int,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch complete trade history for a time range without silent truncation.

        Binance ``/fapi/v1/userTrades`` limits each request to a 7-day window and
        max ``limit`` rows. This method recursively splits windows that saturate
        ``limit`` to avoid dropping trades.
        """
        if start_time > end_time:
            return []

        max_window_ms = 7 * 24 * 60 * 60 * 1000
        pending: list[tuple[int, int]] = [(int(start_time), int(end_time))]
        all_trades: list[dict] = []
        seen: set[tuple[int, int, int]] = set()

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        while pending:
            window_start, window_end = pending.pop()
            if window_start > window_end:
                continue

            # Respect API's max 7-day range constraint per request.
            if (window_end - window_start) > max_window_ms:
                split_end = window_start + max_window_ms
                pending.append((split_end + 1, window_end))
                pending.append((window_start, split_end))
                continue

            batch = await self.get_user_trades(
                symbol=symbol,
                start_time=window_start,
                end_time=window_end,
                limit=limit,
            )

            # Saturated window: split to avoid silently losing fills.
            if len(batch) >= limit and (window_end - window_start) > 1000:
                mid = window_start + (window_end - window_start) // 2
                pending.append((mid + 1, window_end))
                pending.append((window_start, mid))
                continue

            for trade in batch:
                key = (
                    _as_int(trade.get("id", 0), 0),
                    _as_int(trade.get("time", 0), 0),
                    _as_int(trade.get("orderId", 0), 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                all_trades.append(trade)

        all_trades.sort(
            key=lambda t: (_as_int(t.get("time", 0), 0), _as_int(t.get("id", 0), 0)),
        )
        return all_trades

    async def get_income_history(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
        page: int = 1,
    ) -> list[dict]:
        """
        Get income history (requires authentication).
        
        Income types include: REALIZED_PNL, FUNDING_FEE, COMMISSION, etc.
        
        Args:
            symbol: Optional trading pair symbol
            income_type: Filter by type (REALIZED_PNL, FUNDING_FEE, COMMISSION, etc.)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Max number of records
            page: Result page index (1-based)
        
        Returns list of income records:
        {symbol, incomeType, income, asset, time, info, tranId, tradeId}
        """
        params: dict[str, str | int] = {"limit": limit, "page": page}
        if symbol:
            params["symbol"] = symbol.upper()
        if income_type:
            params["incomeType"] = income_type
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return await self._request(self._cfg.binance.endpoints["income"], params, signed=True)

    async def get_income_history_all(
        self,
        symbol: Optional[str] = None,
        income_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
        max_pages: int = 200,
    ) -> list[dict]:
        """Fetch all pages from income history for the selected filter."""
        all_rows: list[dict] = []
        seen: set[tuple[str, int, str, str]] = set()

        def _as_int(value: Any, default: int = 0) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        for page in range(1, max_pages + 1):
            batch = await self.get_income_history(
                symbol=symbol,
                income_type=income_type,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                page=page,
            )
            if not batch:
                break

            for row in batch:
                key = (
                    str(row.get("tranId", "")),
                    _as_int(row.get("time", 0), 0),
                    str(row.get("incomeType", "")),
                    str(row.get("tradeId", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                all_rows.append(row)

            if len(batch) < limit:
                break
        else:
            logger.warning(
                "Income history pagination reached max_pages=%d for symbol=%s type=%s",
                max_pages, symbol, income_type,
            )

        all_rows.sort(key=lambda r: _as_int(r.get("time", 0), 0))
        return all_rows

    async def get_mark_price_klines_all(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1500,
        include_current: bool = False,
    ) -> list[list]:
        """Fetch full mark-price kline range using time-cursor pagination."""
        if start_time > end_time:
            return []

        rows: list[list] = []
        cursor = int(start_time)
        last_open_time = -1

        while cursor <= end_time:
            batch = await self.get_mark_price_klines(
                symbol=symbol,
                interval=interval,
                limit=limit,
                start_time=cursor,
                end_time=end_time,
                include_current=include_current,
            )
            if not batch:
                break

            for row in batch:
                open_time = int(row[0]) if len(row) > 0 else 0
                if open_time <= last_open_time:
                    continue
                rows.append(row)
                last_open_time = open_time

            last_close_time = int(batch[-1][6]) if len(batch[-1]) > 6 else cursor
            next_cursor = last_close_time + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < limit:
                break

        return rows

    async def get_grid_bot_session_data(
        self,
        symbol: str,
        start_time: int,
        end_time: int
    ) -> dict[str, Any]:
        """
        Fetch all data needed for grid bot metrics calculation.
        
        Args:
            symbol: Trading pair symbol
            start_time: Session start time in milliseconds
            end_time: Session end time in milliseconds
        
        Returns dict with trades and income for the session.
        """
        trades, realized_pnl, commissions, funding_fees, mark_klines = await asyncio.gather(
            self.get_user_trades_all(symbol, start_time, end_time),
            self.get_income_history_all(symbol, "REALIZED_PNL", start_time, end_time),
            self.get_income_history_all(symbol, "COMMISSION", start_time, end_time),
            self.get_income_history_all(symbol, "FUNDING_FEE", start_time, end_time),
            self.get_mark_price_klines_all(
                symbol=symbol,
                interval="1m",
                start_time=start_time,
                end_time=end_time,
                include_current=False,
            ),
        )
        
        return {
            "symbol": symbol.upper(),
            "start_time": start_time,
            "end_time": end_time,
            "trades": trades,
            "realized_pnl": realized_pnl,
            "commissions": commissions,
            "funding_fees": funding_fees,
            "mark_klines": mark_klines,
        }

    async def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Get all open orders for a symbol (requires authentication).
        
        Used to count grid lines and analyze pending orders.
        
        Args:
            symbol: Optional trading pair symbol (if None, gets all)
        
        Returns list of open orders with:
        {orderId, symbol, status, clientOrderId, price, origQty, 
         side, positionSide, type, time, updateTime}
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self._request(self._cfg.binance.endpoints["open_orders"], params, signed=True)

    async def get_all_orders(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> list[dict]:
        """
        Get all orders (active, canceled, filled) for a symbol.
        
        Used to find bot start time by matching strategyId.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Max number of orders
        
        Returns list of all orders with:
        {orderId, symbol, status, clientOrderId, price, origQty,
         executedQty, side, type, time, updateTime}
        """
        params = {"symbol": symbol.upper(), "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return await self._request(self._cfg.binance.endpoints["all_orders"], params, signed=True)

    # =========================================================================
    # DATA AGGREGATION
    # =========================================================================

    async def get_all_market_data(self, symbol: str) -> dict[str, Any]:
        """
        Fetch all required market data for a symbol.

        Returns dict with all timeframes and futures metrics.
        All independent API calls run concurrently via asyncio.gather.
        """
        (
            klines_1h, klines_15m, klines_5m, klines_1m,
            open_interest, funding_rate, premium_index,
            order_book, long_short_ratio, taker_volume,
            open_interest_hist,
        ) = await asyncio.gather(
            self.get_klines(symbol, "1h", self._cfg.validation.kline_limits["1h"], include_current=False),
            self.get_klines(symbol, "15m", self._cfg.validation.kline_limits["15m"], include_current=False),
            self.get_klines(symbol, "5m", self._cfg.validation.kline_limits["5m"], include_current=False),
            self.get_klines(symbol, "1m", self._cfg.validation.kline_limits["1m"], include_current=False),
            self.get_open_interest(symbol),
            self.get_funding_rate(symbol, limit=90),
            self.get_premium_index(symbol),
            self.get_order_book(symbol),
            self.get_global_long_short_ratio(symbol),
            self.get_taker_long_short_ratio(symbol),
            self.get_open_interest_hist(symbol, period="1h", limit=24),
        )

        return {
            "symbol": symbol.upper(),
            "klines": {
                "1h": klines_1h,
                "15m": klines_15m,
                "5m": klines_5m,
                "1m": klines_1m,
            },
            "open_interest": open_interest,
            "funding_rate": funding_rate,
            "premium_index": premium_index,
            "order_book": order_book,
            "long_short_ratio": long_short_ratio,
            "taker_volume": taker_volume,
            "open_interest_hist": open_interest_hist,
        }

    async def get_enrichment_market_data(
        self,
        symbol: str,
        *,
        pre_fetched_klines: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch market data for enrichment, reusing pre-fetched klines from scan.

        When *pre_fetched_klines* is provided (keys: '1h', '15m', '5m'),
        those kline fetches are skipped, saving ~3 API calls per symbol.
        Only missing data is fetched from the API.
        """
        have_1h = bool(pre_fetched_klines and "1h" in pre_fetched_klines)
        have_15m = bool(pre_fetched_klines and "15m" in pre_fetched_klines)
        have_5m = bool(pre_fetched_klines and "5m" in pre_fetched_klines)

        coros: list[Any] = []
        labels: list[str] = []

        if not have_1h:
            coros.append(self.get_klines(symbol, "1h", self._cfg.validation.kline_limits["1h"], include_current=False))
            labels.append("klines_1h")
        if not have_15m:
            coros.append(self.get_klines(symbol, "15m", self._cfg.validation.kline_limits["15m"], include_current=False))
            labels.append("klines_15m")
        if not have_5m:
            coros.append(self.get_klines(symbol, "5m", self._cfg.validation.kline_limits["5m"], include_current=False))
            labels.append("klines_5m")

        # Always fetch these (not cached from scan)
        coros.extend([
            self.get_klines(symbol, "1m", self._cfg.validation.kline_limits["1m"], include_current=False),
            self.get_open_interest(symbol),
            self.get_funding_rate(symbol, limit=90),  # 90 periods for z-score computation
            self.get_premium_index(symbol),
            self.get_order_book(symbol),
            self.get_global_long_short_ratio(symbol),
            self.get_taker_long_short_ratio(symbol),
            self.get_open_interest_hist(symbol, period="1h", limit=24),  # OI history for change %
        ])
        labels.extend([
            "klines_1m", "open_interest", "funding_rate", "premium_index",
            "order_book", "long_short_ratio", "taker_volume", "open_interest_hist",
        ])

        results = await asyncio.gather(*coros)
        fetched = dict(zip(labels, results))

        pfk = pre_fetched_klines or {}
        klines_1h = pfk["1h"] if have_1h else fetched["klines_1h"]
        klines_15m = pfk["15m"] if have_15m else fetched["klines_15m"]
        klines_5m = pfk["5m"] if have_5m else fetched["klines_5m"]

        return {
            "symbol": symbol.upper(),
            "klines": {
                "1h": klines_1h,
                "15m": klines_15m,
                "5m": klines_5m,
                "1m": fetched["klines_1m"],
            },
            "open_interest": fetched["open_interest"],
            "funding_rate": fetched["funding_rate"],
            "premium_index": fetched["premium_index"],
            "order_book": fetched["order_book"],
            "long_short_ratio": fetched["long_short_ratio"],
            "taker_volume": fetched["taker_volume"],
            "open_interest_hist": fetched["open_interest_hist"],
        }

    async def check_connection(self) -> dict[str, Any]:
        """
        Test API connection and authentication status.
        
        Returns connection status and account info if authenticated.
        """
        result = {
            "connected": False,
            "authenticated": False,
            "server_time": None,
            "account_balance": None,
        }
        
        try:
            # Test public endpoint
            info = await self.get_exchange_info()
            result["connected"] = True
            result["server_time"] = info.get("serverTime")
            
            # Test authenticated endpoint if keys are configured
            if self.is_authenticated:
                try:
                    balance = await self.get_account_balance()
                    result["authenticated"] = True
                    result["account_balance"] = balance
                except Exception as e:
                    result["auth_error"] = str(e)
        except Exception as e:
            result["error"] = str(e)
        
        return result
