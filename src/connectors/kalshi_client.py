"""Kalshi exchange connector — REST API with RSA-PSS authentication.

Handles:
  - Market discovery (list active markets)
  - Price fetching (bid/ask/mid)
  - Order placement (paper mode by default)
  - Position management

Kalshi API docs: https://docs.kalshi.com/
Authentication: RSA-PSS signing of ``timestamp + method + path``.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.observability.metrics import track_latency

log = get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on transient errors — skip 4xx client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return True


# ── Data Models ──────────────────────────────────────────────────────


@dataclass
class KalshiMarket:
    """A single Kalshi market (binary contract)."""
    ticker: str
    title: str
    category: str = ""
    status: str = "open"          # "open" | "settled" | "closed"
    yes_bid: float = 0.0
    yes_ask: float = 1.0
    no_bid: float = 0.0
    no_ask: float = 1.0
    volume: int = 0
    open_interest: int = 0
    expiration_time: str = ""
    result: str | None = None    # "yes" | "no" | None
    subtitle: str = ""
    event_ticker: str = ""

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "title": self.title,
            "category": self.category,
            "status": self.status,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "mid": self.mid,
            "spread": self.spread,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "expiration_time": self.expiration_time,
            "event_ticker": self.event_ticker,
        }


@dataclass
class KalshiOrderResult:
    """Result of a Kalshi order (real or simulated)."""
    order_id: str = ""
    status: str = ""             # "simulated" | "submitted" | "filled" | "failed"
    fill_price: float = 0.0
    fill_size: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class KalshiPosition:
    """A position on Kalshi."""
    ticker: str = ""
    side: str = ""               # "yes" | "no"
    quantity: int = 0
    avg_price: float = 0.0
    market_value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ── Client ───────────────────────────────────────────────────────────


def _kalshi_price(raw: dict, dollars_key: str, cents_key: str, default: float) -> float:
    """Extract price from Kalshi API response, handling both dollar and cent formats.

    Post March 2026 API returns ``*_dollars`` fields (decimal 0.0-1.0).
    Legacy API returned integer cents (0-100).
    """
    if dollars_key in raw:
        return float(raw[dollars_key])
    cents_val = raw.get(cents_key)
    if cents_val is not None:
        return float(cents_val) / 100
    return default


def _parse_kalshi_market(raw: dict[str, Any]) -> KalshiMarket:
    """Parse a raw API response dict into a KalshiMarket."""
    return KalshiMarket(
        ticker=raw.get("ticker", ""),
        title=raw.get("title", ""),
        category=raw.get("category", ""),
        status=raw.get("status", "open"),
        yes_bid=_kalshi_price(raw, "yes_bid_dollars", "yes_bid", 0.0),
        yes_ask=_kalshi_price(raw, "yes_ask_dollars", "yes_ask", 1.0),
        no_bid=_kalshi_price(raw, "no_bid_dollars", "no_bid", 0.0),
        no_ask=_kalshi_price(raw, "no_ask_dollars", "no_ask", 1.0),
        volume=int(float(raw.get("volume_fp", raw.get("volume", 0)))),
        open_interest=int(float(raw.get("open_interest_fp", raw.get("open_interest", 0)))),
        expiration_time=raw.get("expiration_time", ""),
        result=raw.get("result"),
        subtitle=raw.get("subtitle", raw.get("yes_sub_title", "")),
        event_ticker=raw.get("event_ticker", ""),
    )


class KalshiClient:
    """Async client for the Kalshi REST API."""

    def __init__(
        self,
        base_url: str = "https://api.elections.kalshi.com",
        api_key_id: str = "",
        private_key_path: str = "",
        paper_mode: bool = True,
        timeout: float = 15.0,
    ):
        self._base = base_url.rstrip("/")
        self._api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "")
        self._private_key_path = private_key_path or os.environ.get(
            "KALSHI_PRIVATE_KEY_PATH", "",
        )
        self._paper_mode = paper_mode
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._rsa_key: Any = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    def _load_rsa_key(self) -> Any:
        """Load RSA private key for API authentication.

        Lazy-imports ``cryptography`` to avoid hard dependency.
        """
        if self._rsa_key is not None:
            return self._rsa_key

        try:
            from cryptography.hazmat.primitives.serialization import (
                load_pem_private_key,
            )
        except ImportError:
            raise RuntimeError(
                "cryptography is required for Kalshi API authentication. "
                "Install it with: pip install 'bot[kalshi]'"
            )

        # Try env var first, then file path
        key_pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
        if key_pem:
            key_data = key_pem.encode()
        elif self._private_key_path and os.path.isfile(self._private_key_path):
            with open(self._private_key_path, "rb") as f:
                key_data = f.read()
        else:
            raise RuntimeError(
                "Kalshi RSA key not found. Set KALSHI_PRIVATE_KEY env var "
                "or provide kalshi_private_key_path in config."
            )

        self._rsa_key = load_pem_private_key(key_data, password=None)
        return self._rsa_key

    def _build_auth_headers(
        self, method: str, path: str, timestamp: str | None = None,
    ) -> dict[str, str]:
        """Build RSA-PSS signed authentication headers."""
        ts = timestamp or str(int(time.time() * 1000))
        message = ts + method.upper() + path

        key = self._load_rsa_key()

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        import base64
        sig_b64 = base64.b64encode(signature).decode()

        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    # ── HTTP helpers ─────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    async def _get(
        self, path: str, params: dict[str, Any] | None = None,
    ) -> Any:
        """Authenticated GET request."""
        await rate_limiter.get("kalshi").acquire()
        client = self._ensure_client()

        headers = {}
        if self._api_key_id:
            headers = self._build_auth_headers("GET", path)

        with track_latency("kalshi"):
            resp = await client.get(path, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception(_is_retryable),
    )
    async def _post(
        self, path: str, body: dict[str, Any] | None = None,
    ) -> Any:
        """Authenticated POST request."""
        await rate_limiter.get("kalshi").acquire()
        client = self._ensure_client()

        headers = {"Content-Type": "application/json"}
        if self._api_key_id:
            headers.update(self._build_auth_headers("POST", path))

        with track_latency("kalshi"):
            resp = await client.post(path, json=body or {}, headers=headers)
            resp.raise_for_status()
            return resp.json()

    # ── Market data endpoints ────────────────────────────────────────

    async def list_markets(
        self,
        status: str = "open",
        limit: int = 100,
        cursor: str = "",
    ) -> list[KalshiMarket]:
        """List Kalshi markets with optional filtering."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        data = await self._get("/trade-api/v2/markets", params=params)
        markets_raw = data.get("markets", [])
        return [_parse_kalshi_market(m) for m in markets_raw]

    async def list_markets_paginated(
        self,
        status: str = "open",
        max_markets: int = 500,
    ) -> list[KalshiMarket]:
        """Paginate through Kalshi markets using cursor-based pagination."""
        all_markets: list[KalshiMarket] = []
        cursor = ""
        page_size = min(100, max_markets)

        for _ in range(max_markets // page_size + 1):
            params: dict[str, Any] = {"limit": page_size}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor

            data = await self._get("/trade-api/v2/markets", params=params)
            markets_raw = data.get("markets", [])
            all_markets.extend(_parse_kalshi_market(m) for m in markets_raw)

            cursor = data.get("cursor", "")
            if not cursor or len(markets_raw) < page_size:
                break
            if len(all_markets) >= max_markets:
                break

        return all_markets

    async def get_market(self, ticker: str) -> KalshiMarket:
        """Get a single market by ticker."""
        data = await self._get(f"/trade-api/v2/markets/{ticker}")
        market_raw = data.get("market", data)
        return _parse_kalshi_market(market_raw)

    async def get_market_orderbook(self, ticker: str) -> dict[str, Any]:
        """Get the orderbook for a market."""
        return await self._get(f"/trade-api/v2/markets/{ticker}/orderbook")

    # ── Order placement ──────────────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        order_type: str = "limit",
    ) -> KalshiOrderResult:
        """Place an order on Kalshi.

        In paper mode, returns a simulated result without hitting the API.
        """
        if self._paper_mode:
            return KalshiOrderResult(
                order_id=f"paper-{uuid.uuid4().hex[:12]}",
                status="simulated",
                fill_price=price,
                fill_size=quantity,
            )

        body = {
            "ticker": ticker,
            "action": "buy" if side.lower() in ("buy", "yes") else "sell",
            "side": "yes" if side.lower() in ("buy", "yes") else "no",
            "count": quantity,
            "type": order_type,
            "yes_price": int(price * 100),  # cents for legacy API
            "yes_price_dollars": str(round(price, 4)),  # dollars for new API
        }

        try:
            data = await self._post("/trade-api/v2/portfolio/orders", body=body)
            order_data = data.get("order", data)
            fill_price = _kalshi_price(
                order_data, "yes_price_dollars", "yes_price", price,
            )
            return KalshiOrderResult(
                order_id=order_data.get("order_id", ""),
                status=order_data.get("status", "submitted"),
                fill_price=fill_price,
                fill_size=int(order_data.get("count", 0)),
            )
        except Exception as e:
            log.warning("kalshi.order_error", ticker=ticker, error=str(e))
            return KalshiOrderResult(
                order_id="",
                status="failed",
                error=str(e),
            )

    # ── Position management ──────────────────────────────────────────

    async def get_positions(self) -> list[KalshiPosition]:
        """Get current portfolio positions."""
        if self._paper_mode:
            return []

        data = await self._get("/trade-api/v2/portfolio/positions")
        positions_raw = data.get("market_positions", [])
        return [
            KalshiPosition(
                ticker=p.get("ticker", ""),
                side="yes" if p.get("position", 0) > 0 else "no",
                quantity=abs(int(p.get("position", 0))),
                avg_price=float(p.get("average_price", 0)) / 100,
                market_value=float(p.get("market_value", 0)) / 100,
            )
            for p in positions_raw
        ]

    # ── Cleanup ──────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
