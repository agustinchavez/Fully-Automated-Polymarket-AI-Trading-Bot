"""Kalshi prior connector — cross-platform consensus prices.

Wraps existing KalshiClient + MarketMatcher to fetch the current
Kalshi price for a matched market. Returns a FetchedSource plus
a ``consensus_signal`` dict in ``raw`` for the signal aggregator.

Requires RSA-PSS authentication (KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

# All categories — Kalshi covers elections, macro, tech, etc.
_ALL_CATEGORIES: set[str] = {
    "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
    "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
    "ENTERTAINMENT", "TECH", "REGULATION", "CULTURE", "UNKNOWN",
}


class KalshiPriorConnector(BaseResearchConnector):
    """Fetch current Kalshi price for cross-platform consensus signal."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._kalshi_client: Any = None
        self._matcher: Any = None

    @property
    def name(self) -> str:
        return "kalshi_prior"

    def relevant_categories(self) -> set[str]:
        return _ALL_CATEGORIES

    def _get_kalshi_client(self) -> Any:
        """Lazy-init KalshiClient in paper mode (read-only)."""
        if self._kalshi_client is None:
            from src.connectors.kalshi_client import KalshiClient
            self._kalshi_client = KalshiClient(paper_mode=True)
        return self._kalshi_client

    def _get_matcher(self) -> Any:
        """Lazy-init MarketMatcher."""
        if self._matcher is None:
            from src.connectors.market_matcher import MarketMatcher
            self._matcher = MarketMatcher(min_confidence=0.3)
        return self._matcher

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        await rate_limiter.get("kalshi_prior").acquire()

        client = self._get_kalshi_client()
        matcher = self._get_matcher()

        # Paginate to get full market pool (~500 markets).
        # Let exceptions propagate to BaseResearchConnector.fetch()
        # so the circuit breaker trips on persistent failures (e.g. 401).
        kalshi_markets = await client.list_markets_paginated(
            status="open", max_markets=500,
        )

        if not kalshi_markets:
            return []

        # Create a pseudo Polymarket object for matching
        poly_proxy = _PolyProxy(question=question)
        matches = matcher.find_matches([poly_proxy], kalshi_markets)

        if not matches:
            return []

        # Take best match
        best = matches[0]
        kalshi_market = None
        for km in kalshi_markets:
            if km.ticker == best.kalshi_ticker:
                kalshi_market = km
                break

        if kalshi_market is None:
            return []

        mid = kalshi_market.mid
        spread = kalshi_market.spread

        content_lines = [
            f"Cross-Platform Price: Kalshi",
            f"Market: {kalshi_market.title}",
            f"YES mid-price: {mid:.1%}",
            f"Spread: {spread:.1%}",
            f"Match confidence: {best.match_confidence:.2f}",
            f"Match method: {best.match_method}",
            f"Source: Kalshi Exchange",
        ]
        content = "\n".join(content_lines)
        snippet = f"Kalshi: {mid:.1%} YES (spread {spread:.1%})"

        return [self._make_source(
            title=f"Kalshi Price: {kalshi_market.title}",
            url=f"https://kalshi.com/markets/{kalshi_market.ticker}",
            snippet=snippet,
            publisher="Kalshi Exchange",
            date="",
            content=content,
            authority_score=0.9,
            raw={
                "consensus_signal": {
                    "platform": "kalshi",
                    "price": round(mid, 4),
                    "spread_pp": round(spread * 100, 1),
                    "match_confidence": round(best.match_confidence, 2),
                },
            },
        )]


@dataclass
class _PolyProxy:
    """Minimal proxy for MarketMatcher compatibility."""
    question: str
    id: str = "proxy"
    condition_id: str = "proxy"
