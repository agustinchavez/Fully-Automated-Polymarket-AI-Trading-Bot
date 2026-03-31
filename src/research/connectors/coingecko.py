"""CoinGecko research connector — crypto price data for CRYPTO markets.

Uses the free CoinGecko Demo API (30 calls/min) to fetch real-time
prices, 24h changes, and 30-day OHLCV for cryptocurrencies.
"""

from __future__ import annotations

import os
import time
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Ticker / name → CoinGecko coin ID
_COIN_MAP: dict[str, str] = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "cardano": "cardano",
    "ada": "cardano",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "xrp": "ripple",
    "ripple": "ripple",
    "polkadot": "polkadot",
    "dot": "polkadot",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
    "polygon": "matic-network",
    "matic": "matic-network",
    "chainlink": "chainlink",
    "link": "chainlink",
    "litecoin": "litecoin",
    "ltc": "litecoin",
    "uniswap": "uniswap",
    "uni": "uniswap",
}

_CRYPTO_KEYWORDS: list[str] = [
    "btc", "eth", "sol", "bitcoin", "ethereum", "solana",
    "blockchain", "defi", "nft", "token", "protocol",
    "crypto", "coin", "cardano", "dogecoin", "xrp",
]

# Simple response cache: {coin_id: (timestamp, data)}
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 60  # seconds


class CoinGeckoConnector(BaseResearchConnector):
    """Fetch cryptocurrency data from CoinGecko's free API."""

    @property
    def name(self) -> str:
        return "coingecko"

    def relevant_categories(self) -> set[str]:
        return {"CRYPTO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _CRYPTO_KEYWORDS)

    def _get_api_key(self) -> str:
        return os.environ.get("COINGECKO_API_KEY", "")

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        max_coins = getattr(self._config, "coingecko_max_coins", 3)
        coin_ids = self._extract_coins(question, max_coins)
        if not coin_ids:
            return []

        sources: list[FetchedSource] = []
        for coin_id in coin_ids:
            source = await self._fetch_price(coin_id)
            if source:
                sources.append(source)
        return sources

    def _extract_coins(self, question: str, max_coins: int) -> list[str]:
        """Extract CoinGecko coin IDs from the question."""
        q_lower = question.lower()
        found: list[str] = []
        seen: set[str] = set()

        # Check each token against the map
        for token in q_lower.split():
            token_clean = token.strip("?.,!;:'\"()")
            cg_id = _COIN_MAP.get(token_clean)
            if cg_id and cg_id not in seen:
                seen.add(cg_id)
                found.append(cg_id)

        # Also check multi-word names
        for name, cg_id in _COIN_MAP.items():
            if len(name) > 3 and name in q_lower and cg_id not in seen:
                seen.add(cg_id)
                found.append(cg_id)

        return found[:max_coins]

    async def _fetch_price(self, coin_id: str) -> FetchedSource | None:
        """Fetch current price + 24h change for a coin."""
        # Check cache
        now = time.monotonic()
        cached = _cache.get(coin_id)
        if cached and (now - cached[0]) < _CACHE_TTL:
            return self._format_price(coin_id, cached[1])

        await rate_limiter.get("coingecko").acquire()

        client = self._get_client()
        headers: dict[str, str] = {}
        api_key = self._get_api_key()
        if api_key:
            headers["x-cg-demo-api-key"] = api_key

        resp = await client.get(
            f"{_COINGECKO_BASE}/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        coin_data = data.get(coin_id)
        if not coin_data:
            return None

        # Cache the result
        _cache[coin_id] = (now, coin_data)
        return self._format_price(coin_id, coin_data)

    def _format_price(self, coin_id: str, data: dict) -> FetchedSource:
        """Format price data into a FetchedSource."""
        price = data.get("usd", 0)
        change_24h = data.get("usd_24h_change", 0)
        market_cap = data.get("usd_market_cap", 0)
        volume = data.get("usd_24h_vol", 0)

        change_sign = "+" if change_24h >= 0 else ""
        display_name = coin_id.replace("-", " ").title()

        content = (
            f"CoinGecko Price Data: {display_name}\n"
            f"Current Price: ${price:,.2f}\n"
            f"24h Change: {change_sign}{change_24h:.2f}%\n"
            f"Market Cap: ${market_cap:,.0f}\n"
            f"24h Volume: ${volume:,.0f}\n"
            f"Source: CoinGecko (aggregated from 1,700+ exchanges)"
        )

        snippet = f"{display_name}: ${price:,.2f} ({change_sign}{change_24h:.2f}%)"

        return self._make_source(
            title=f"CoinGecko: {display_name} Price",
            url=f"https://www.coingecko.com/en/coins/{coin_id}",
            snippet=snippet,
            publisher="CoinGecko",
            content=content,
        )
