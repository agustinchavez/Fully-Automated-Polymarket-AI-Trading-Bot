"""Google Trends research connector — search interest spike detection.

Dual-source architecture:
- **SerpAPI** (gated: only when SERPAPI_KEY env var present + market volume above threshold):
  GET https://serpapi.com/search.json?engine=google_trends&q={keyword}&data_type=TIMESERIES&date=today 3-m&geo=US
- **Tavily** (always runs when TAVILY_API_KEY present):
  search for "{keyword} trends analysis 2026"
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_SERPAPI_URL = "https://serpapi.com/search.json"
_TAVILY_URL = "https://api.tavily.com/search"

# Stop words removed during keyword extraction
_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "will", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "shall", "should",
    "would", "could", "can", "may", "might", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "about", "if", "then", "that", "this", "these",
    "those", "what", "which", "who", "whom", "how", "when", "where",
    "why", "its", "it", "he", "she", "they", "them", "his", "her",
    "their", "my", "your", "our", "me", "us", "up", "out", "over",
}


class GoogleTrendsConnector(BaseResearchConnector):
    """Detect search interest spikes via Google Trends (SerpAPI + Tavily)."""

    @property
    def name(self) -> str:
        return "google_trends"

    def relevant_categories(self) -> set[str]:
        return {
            "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
            "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
            "ENTERTAINMENT", "TECH", "REGULATION", "CULTURE", "UNKNOWN",
        }

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
        **kwargs: Any,
    ) -> list[FetchedSource]:
        keyword = self._extract_keyword(question)
        if not keyword:
            return []

        volume_threshold = getattr(
            self._config, "google_trends_volume_threshold_usd", 50_000.0,
        )
        volume_usd: float = kwargs.get(
            "volume_usd",
            getattr(self._config, "volume_usd", 0.0),
        )

        serpapi_key = os.environ.get("SERPAPI_KEY", "")
        tavily_key = os.environ.get("TAVILY_API_KEY", "")

        spike_ratio: float = 1.0
        current_index: int = 0
        serpapi_ok = False
        narrative_context: str = ""

        # ── SerpAPI (gated) ───────────────────────────────────────
        if serpapi_key and volume_usd >= volume_threshold:
            try:
                serpapi_data = await self._fetch_serpapi(keyword, serpapi_key)
                if serpapi_data is not None:
                    spike_ratio = serpapi_data["spike_ratio"]
                    current_index = serpapi_data["current_index"]
                    serpapi_ok = True
            except Exception as exc:
                log.warning(
                    "google_trends.serpapi_failed",
                    keyword=keyword,
                    error=str(exc),
                )

        # ── Tavily (always when key present) ──────────────────────
        if tavily_key:
            try:
                narrative_context = await self._fetch_tavily(keyword, tavily_key)
            except Exception as exc:
                log.warning(
                    "google_trends.tavily_failed",
                    keyword=keyword,
                    error=str(exc),
                )

        # Need at least one source to produce a result
        if not serpapi_ok and not narrative_context:
            return []

        return self._build_source(
            keyword=keyword,
            spike_ratio=spike_ratio,
            current_index=current_index,
            narrative_context=narrative_context,
            serpapi_ok=serpapi_ok,
        )

    # ── SerpAPI ───────────────────────────────────────────────────

    async def _fetch_serpapi(
        self, keyword: str, api_key: str,
    ) -> dict[str, Any] | None:
        """Fetch Google Trends timeseries from SerpAPI."""
        await rate_limiter.get("google_trends").acquire()

        client = self._get_client()
        resp = await client.get(
            _SERPAPI_URL,
            params={
                "engine": "google_trends",
                "q": keyword,
                "data_type": "TIMESERIES",
                "date": "today 3-m",
                "geo": "US",
                "api_key": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        timeline_data = (
            data.get("interest_over_time", {}).get("timeline_data", [])
        )
        if not timeline_data:
            return None

        # Extract interest values from timeline points
        values: list[int] = []
        for point in timeline_data:
            point_values = point.get("values", [])
            if point_values:
                # First query's extracted_value
                raw_val = point_values[0].get("extracted_value", 0)
                values.append(int(raw_val))

        if not values:
            return None

        return self._compute_spike(values)

    # ── Tavily ────────────────────────────────────────────────────

    async def _fetch_tavily(self, keyword: str, api_key: str) -> str:
        """Search Tavily for trend narrative context."""
        await rate_limiter.get("google_trends").acquire()

        client = self._get_client()
        resp = await client.post(
            _TAVILY_URL,
            json={
                "api_key": api_key,
                "query": f"{keyword} trends analysis 2026",
                "max_results": 2,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return ""

        parts: list[str] = []
        for r in results[:2]:
            title = r.get("title", "")
            snippet = r.get("content", r.get("snippet", ""))
            if title or snippet:
                parts.append(f"{title}: {snippet}" if title else snippet)

        return " | ".join(parts)

    # ── Keyword extraction ────────────────────────────────────────

    def _extract_keyword(self, question: str) -> str:
        """Extract the most relevant 1-3 word keyword from the question.

        Heuristic: remove stop words, punctuation, and leading verbs,
        then take the first significant phrase (up to 3 words).
        """
        # Strip trailing punctuation and leading question structure
        q = question.strip().rstrip("?")
        q = re.sub(
            r"^(Will|Is|Does|Has|Are|Do|Can|Should|What|When|Where|How|Why)"
            r"\s+(the\s+)?",
            "", q, flags=re.I,
        )

        # Tokenize and filter stop words
        tokens = re.findall(r"[A-Za-z0-9]+", q)
        significant = [
            t for t in tokens
            if t.lower() not in _STOP_WORDS and len(t) > 1
        ]

        if not significant:
            return ""

        # Take up to 3 significant words
        keyword = " ".join(significant[:3])
        return keyword

    # ── Spike computation ─────────────────────────────────────────

    def _compute_spike(self, values: list[int]) -> dict[str, Any]:
        """Compute 7d/30d spike ratio and current index from timeseries.

        SerpAPI returns weekly data points for a 3-month window, so we
        approximate:
        - recent 7d ~ last 1 data point
        - recent 30d ~ last 4 data points
        """
        if not values:
            return {"spike_ratio": 1.0, "current_index": 0}

        current_index = values[-1]

        # Average of last ~4 points (roughly 30 days of weekly data)
        recent_count = min(4, len(values))
        avg_recent = sum(values[-recent_count:]) / recent_count

        # Average of the full series (baseline)
        avg_full = sum(values) / len(values)

        spike_ratio = round(avg_recent / avg_full, 2) if avg_full > 0 else 1.0

        return {
            "spike_ratio": spike_ratio,
            "current_index": current_index,
        }

    # ── Source builder ────────────────────────────────────────────

    def _build_source(
        self,
        *,
        keyword: str,
        spike_ratio: float,
        current_index: int,
        narrative_context: str,
        serpapi_ok: bool,
    ) -> list[FetchedSource]:
        """Build a single FetchedSource combining both data streams."""
        if spike_ratio < 1.5:
            direction = "normal"
        elif spike_ratio < 2.0:
            direction = "elevated"
        elif spike_ratio < 3.0:
            direction = "strong"
        else:
            direction = "viral"

        content_lines = [
            f"Google Trends Signal: {keyword}",
        ]
        if serpapi_ok:
            content_lines.extend([
                f"Spike ratio: {spike_ratio:.1f}x ({direction.upper()})",
                f"Current index: {current_index}",
            ])
        if narrative_context:
            content_lines.append(f"Narrative: {narrative_context[:500]}")
        content_lines.append("Source: Google Trends (SerpAPI + Tavily)")

        content = "\n".join(content_lines)

        if serpapi_ok:
            snippet = (
                f"Google Trends '{keyword}': {spike_ratio:.1f}x spike "
                f"({direction}), index={current_index}"
            )
        else:
            snippet = (
                f"Google Trends '{keyword}': narrative context available"
            )

        return [self._make_source(
            title=f"Google Trends: {keyword}",
            url=f"https://trends.google.com/trends/explore?q={keyword}&geo=US",
            snippet=snippet,
            publisher="Google Trends",
            content=content,
            authority_score=0.6,
            raw={
                "behavioral_signal": {
                    "source": "google_trends",
                    "signal_type": "search_trend",
                    "value": spike_ratio,
                    "current_index": current_index,
                    "narrative": narrative_context[:300] if narrative_context else "",
                },
            },
        )]
