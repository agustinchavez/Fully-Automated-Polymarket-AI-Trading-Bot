"""Manifold Markets consensus connector — rationalist/forecasting community.

Free API, no auth, 500 req/min.
Endpoint: api.manifold.markets/v0/search-markets
Returns BINARY market probabilities as consensus signals.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_BASE = "https://api.manifold.markets/v0"
_MIN_VOLUME = 100

_ALL_CATEGORIES: set[str] = {
    "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
    "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
    "ENTERTAINMENT", "TECH", "REGULATION", "CULTURE", "UNKNOWN",
}


class ManifoldConnector(BaseResearchConnector):
    """Fetch consensus probability from Manifold Markets."""

    name = "manifold"

    def relevant_categories(self) -> set[str]:
        return _ALL_CATEGORIES

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        await rate_limiter.get("manifold").acquire()

        params = {"term": question[:120], "limit": "5", "sort": "relevance"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{_BASE}/search-markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
        except Exception as exc:
            log.warning("manifold.fetch_failed", error=str(exc))
            return []

        for m in data or []:
            if m.get("isResolved"):
                continue
            if m.get("outcomeType") != "BINARY":
                continue
            prob = m.get("probability")
            if prob is None or m.get("volume", 0) < _MIN_VOLUME:
                continue

            traders = m.get("uniqueBettorCount", 0)
            slug = m.get("slug", "")
            creator = m.get("creatorUsername", "")
            url = f"https://manifold.markets/{creator}/{slug}" if slug else ""

            return [FetchedSource(
                url=url,
                title=m.get("question", ""),
                publisher="Manifold Markets",
                content=f"Manifold community: {prob:.1%} ({traders} traders)",
                authority_score=0.72,
                raw={
                    "consensus_signal": {
                        "platform": "manifold",
                        "price": float(prob),
                        "volume": m.get("volume", 0),
                        "traders": traders,
                    },
                },
            )]

        return []
