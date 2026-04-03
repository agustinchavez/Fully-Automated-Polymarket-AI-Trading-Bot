"""Wikipedia pageviews connector — attention spike detection.

Uses the free Wikimedia REST API (no key required):
- Endpoint: wikimedia.org/api/rest_v1/metrics/pageviews
- Spike detection: 7-day avg / 30-day avg ratio
- 4-hour in-memory cache (pageview data refreshes daily)
"""

from __future__ import annotations

import re
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_WIKIMEDIA_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews"
    "/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
)
_USER_AGENT = "PolymarketBot/1.0 (research@example.com)"

# Common entities → Wikipedia article titles
_ENTITY_MAP: dict[str, str] = {
    "fed": "Federal_Reserve",
    "federal reserve": "Federal_Reserve",
    "trump": "Donald_Trump",
    "biden": "Joe_Biden",
    "harris": "Kamala_Harris",
    "desantis": "Ron_DeSantis",
    "elon musk": "Elon_Musk",
    "musk": "Elon_Musk",
    "tesla": "Tesla,_Inc.",
    "apple": "Apple_Inc.",
    "google": "Google",
    "microsoft": "Microsoft",
    "amazon": "Amazon_(company)",
    "meta": "Meta_Platforms",
    "nvidia": "Nvidia",
    "openai": "OpenAI",
    "chatgpt": "ChatGPT",
    "gpt": "GPT-4",
    "bitcoin": "Bitcoin",
    "btc": "Bitcoin",
    "ethereum": "Ethereum",
    "eth": "Ethereum",
    "ukraine": "Russian_invasion_of_Ukraine",
    "russia": "Russia",
    "china": "China",
    "taiwan": "Taiwan",
    "nato": "NATO",
    "supreme court": "Supreme_Court_of_the_United_States",
    "scotus": "Supreme_Court_of_the_United_States",
    "s&p 500": "S%26P_500",
    "inflation": "Inflation",
    "recession": "Recession",
    "gdp": "Gross_domestic_product",
    "cpi": "Consumer_price_index",
    "unemployment": "Unemployment",
    "interest rate": "Interest_rate",
    "mortgage": "Mortgage",
    "climate change": "Climate_change",
    "covid": "COVID-19_pandemic",
    "world cup": "FIFA_World_Cup",
    "olympics": "Olympic_Games",
    "super bowl": "Super_Bowl",
    "spacex": "SpaceX",
    "boeing": "Boeing",
    "pfizer": "Pfizer",
    "moderna": "Moderna",
}


class WikipediaPageviewsConnector(BaseResearchConnector):
    """Detect attention spikes via Wikipedia pageview data."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._cache: dict[str, tuple[float, dict]] = {}  # article -> (ts, data)

    @property
    def name(self) -> str:
        return "wikipedia_pageviews"

    def relevant_categories(self) -> set[str]:
        return {
            "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
            "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
            "ENTERTAINMENT", "TECH", "REGULATION", "UNKNOWN",
        }

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        article = self._extract_article(question)
        if not article:
            return []

        cache_ttl = getattr(self._config, "wikipedia_cache_ttl_secs", 14400)

        # Check cache
        cached = self._cache.get(article)
        if cached:
            ts, data = cached
            if _time.monotonic() - ts < cache_ttl:
                return self._build_source(article, data)

        await rate_limiter.get("wikipedia").acquire()

        # Fetch 35 days of data (30d baseline + recent 7d)
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=35)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")

        url = _WIKIMEDIA_API.format(article=article, start=start, end=end)
        client = self._get_client()
        resp = await client.get(
            url,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            return []

        # Compute spike ratio
        views = [item.get("views", 0) for item in items]
        spike_data = self._compute_spike(views)

        # Cache
        self._cache[article] = (_time.monotonic(), spike_data)

        return self._build_source(article, spike_data)

    def _extract_article(self, question: str) -> str:
        """Map question to Wikipedia article title."""
        q_lower = question.lower()

        # Check entity map (longest match first)
        sorted_entities = sorted(_ENTITY_MAP.keys(), key=len, reverse=True)
        for entity in sorted_entities:
            if entity in q_lower:
                return _ENTITY_MAP[entity]

        # Fallback: extract capitalized noun phrases
        words = question.split()
        proper_nouns = [
            w for w in words
            if w[0].isupper() and len(w) > 2
            and w.lower() not in {
                "will", "the", "does", "has", "are", "can",
                "should", "what", "when", "how", "is",
            }
        ]
        if proper_nouns:
            return "_".join(proper_nouns[:2])

        return ""

    def _compute_spike(self, views: list[int]) -> dict:
        """Compute 7d/30d spike ratio from daily view counts."""
        if len(views) < 7:
            return {"spike_ratio": 1.0, "avg_7d": 0, "avg_30d": 0}

        avg_7d = sum(views[-7:]) / 7
        avg_30d = sum(views[-30:]) / min(30, len(views[-30:])) if len(views) >= 7 else avg_7d
        spike_ratio = avg_7d / avg_30d if avg_30d > 0 else 1.0

        return {
            "spike_ratio": round(spike_ratio, 2),
            "avg_7d": round(avg_7d),
            "avg_30d": round(avg_30d),
        }

    def _build_source(self, article: str, spike_data: dict) -> list[FetchedSource]:
        """Build FetchedSource from spike data."""
        spike = spike_data.get("spike_ratio", 1.0)
        avg_7d = spike_data.get("avg_7d", 0)
        avg_30d = spike_data.get("avg_30d", 0)

        if spike < 1.5:
            direction = "normal"
        elif spike < 2.0:
            direction = "elevated"
        elif spike < 3.0:
            direction = "strong"
        else:
            direction = "viral"

        article_display = article.replace("_", " ").replace("%26", "&")
        content_lines = [
            f"Wikipedia Attention Signal: {article_display}",
            f"Spike ratio: {spike:.1f}x ({direction.upper()})",
            f"7-day avg views: {avg_7d:,}",
            f"30-day avg views: {avg_30d:,}",
            f"Source: Wikimedia Pageviews API",
        ]
        content = "\n".join(content_lines)
        snippet = f"Wikipedia '{article_display}': {spike:.1f}x spike ({direction})"

        return [self._make_source(
            title=f"Wikipedia Pageviews: {article_display}",
            url=f"https://en.wikipedia.org/wiki/{article}",
            snippet=snippet,
            publisher="Wikipedia/Wikimedia",
            content=content,
            authority_score=0.7,
            raw={
                "behavioral_signal": {
                    "source": "wikipedia",
                    "signal_type": "attention_spike",
                    "value": spike,
                    "direction": direction,
                    "article": article,
                    "avg_7d": avg_7d,
                    "avg_30d": avg_30d,
                },
            },
        )]
