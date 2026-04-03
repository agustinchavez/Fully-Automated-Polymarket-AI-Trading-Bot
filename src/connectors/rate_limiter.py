"""Token-bucket rate limiter for all external API calls.

Prevents bans from SerpAPI, OpenAI, Polymarket, and other endpoints.
Thread-safe, per-endpoint rate limiting with configurable burst.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class BucketConfig:
    """Configuration for a single rate-limit bucket."""
    tokens_per_second: float
    max_burst: int
    name: str = ""


# Default rate limits per endpoint category
DEFAULT_LIMITS: dict[str, BucketConfig] = {
    "gamma": BucketConfig(tokens_per_second=5.0, max_burst=10, name="Polymarket Gamma"),
    "clob": BucketConfig(tokens_per_second=10.0, max_burst=20, name="Polymarket CLOB"),
    "openai": BucketConfig(tokens_per_second=3.0, max_burst=5, name="OpenAI"),
    "anthropic": BucketConfig(tokens_per_second=2.0, max_burst=4, name="Anthropic"),
    "google": BucketConfig(tokens_per_second=2.0, max_burst=4, name="Google AI"),
    "serpapi": BucketConfig(tokens_per_second=1.0, max_burst=3, name="SerpAPI"),
    "bing": BucketConfig(tokens_per_second=3.0, max_burst=5, name="Bing Search"),
    "tavily": BucketConfig(tokens_per_second=2.0, max_burst=4, name="Tavily"),
    "web_fetch": BucketConfig(tokens_per_second=5.0, max_burst=15, name="Web Fetch"),
    "open_meteo": BucketConfig(tokens_per_second=2.0, max_burst=5, name="Open-Meteo"),
    "binance": BucketConfig(tokens_per_second=5.0, max_burst=10, name="Binance"),
    "kalshi": BucketConfig(tokens_per_second=3.0, max_burst=6, name="Kalshi"),
    "fred": BucketConfig(tokens_per_second=2.0, max_burst=4, name="FRED"),
    "coingecko": BucketConfig(tokens_per_second=0.5, max_burst=2, name="CoinGecko"),
    "congress": BucketConfig(tokens_per_second=1.4, max_burst=3, name="Congress.gov"),
    "gdelt": BucketConfig(tokens_per_second=1.0, max_burst=3, name="GDELT"),
    "courtlistener": BucketConfig(tokens_per_second=1.0, max_burst=3, name="CourtListener"),
    "duckduckgo": BucketConfig(tokens_per_second=1.0, max_burst=3, name="DuckDuckGo"),
    "searxng": BucketConfig(tokens_per_second=5.0, max_burst=10, name="SearXNG"),
    "edgar": BucketConfig(tokens_per_second=2.0, max_burst=4, name="SEC EDGAR"),
    "arxiv": BucketConfig(tokens_per_second=0.33, max_burst=1, name="arXiv"),
    "openfda": BucketConfig(tokens_per_second=4.0, max_burst=8, name="openFDA"),
    "worldbank": BucketConfig(tokens_per_second=2.0, max_burst=4, name="World Bank"),
    "kalshi_prior": BucketConfig(tokens_per_second=2.0, max_burst=4, name="Kalshi Prior"),
    "metaculus": BucketConfig(tokens_per_second=0.5, max_burst=2, name="Metaculus"),
    "wikipedia": BucketConfig(tokens_per_second=1.0, max_burst=2, name="Wikipedia"),
    "google_trends": BucketConfig(tokens_per_second=0.5, max_burst=1, name="Google Trends"),
    "reddit": BucketConfig(tokens_per_second=1.0, max_burst=3, name="Reddit"),
    "pubmed": BucketConfig(tokens_per_second=3.0, max_burst=5, name="PubMed"),
    "xai": BucketConfig(tokens_per_second=2.0, max_burst=4, name="xAI Grok"),
    "deepseek": BucketConfig(tokens_per_second=3.0, max_burst=5, name="DeepSeek"),
    "manifold": BucketConfig(tokens_per_second=5.0, max_burst=10, name="Manifold"),
    "predictit": BucketConfig(tokens_per_second=0.5, max_burst=1, name="PredictIt"),
}


class TokenBucket:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, config: BucketConfig):
        self._config = config
        self._tokens: float = float(config.max_burst)
        self._last_refill: float = time.monotonic()
        self._lock = Lock()
        self._total_requests: int = 0
        self._total_waits: int = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self._config.max_burst),
            self._tokens + elapsed * self._config.tokens_per_second,
        )
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking. Returns True if acquired."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._total_requests += 1
                return True
            return False

    def wait_time(self) -> float:
        """Return seconds until a token is available."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                return 0.0
            deficit = 1.0 - self._tokens
            return deficit / self._config.tokens_per_second

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        while True:
            wt = self.wait_time()
            if wt <= 0:
                if self.try_acquire():
                    return
            else:
                self._total_waits += 1
                await asyncio.sleep(wt)

    def acquire_sync(self) -> None:
        """Synchronous version of acquire."""
        while True:
            wt = self.wait_time()
            if wt <= 0:
                if self.try_acquire():
                    return
            else:
                self._total_waits += 1
                time.sleep(wt)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
        }


class RateLimiterRegistry:
    """Global registry of per-endpoint rate limiters."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def get(self, endpoint: str) -> TokenBucket:
        """Get or create a rate limiter for an endpoint."""
        with self._lock:
            if endpoint not in self._buckets:
                config = DEFAULT_LIMITS.get(
                    endpoint,
                    BucketConfig(tokens_per_second=5.0, max_burst=10, name=endpoint),
                )
                self._buckets[endpoint] = TokenBucket(config)
            return self._buckets[endpoint]

    def configure(self, endpoint: str, tokens_per_second: float, max_burst: int) -> None:
        """Override rate limit for an endpoint."""
        with self._lock:
            config = BucketConfig(
                tokens_per_second=tokens_per_second,
                max_burst=max_burst,
                name=endpoint,
            )
            self._buckets[endpoint] = TokenBucket(config)

    def stats(self) -> dict[str, dict[str, int]]:
        """Get stats for all buckets."""
        with self._lock:
            return {name: bucket.stats for name, bucket in self._buckets.items()}


# Global singleton
rate_limiter = RateLimiterRegistry()
