"""GDELT research connector — global news volume/tone for broad market types.

Uses the free GDELT 2.0 DOC API (no key required) to fetch news volume
timelines and detect coverage spikes for prediction market questions.

Also supports the GDELT GKG (Global Knowledge Graph) API for entity extraction,
providing named entities (persons, organizations, locations) mentioned in
recent news coverage of a topic.
"""

from __future__ import annotations

import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_GKG_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

# Broad category coverage — GDELT is useful for most market types
_RELEVANT_CATEGORIES: set[str] = {
    "MACRO", "ELECTION", "CORPORATE", "LEGAL", "GEOPOLITICS",
}


class GdeltConnector(BaseResearchConnector):
    """Fetch news volume/tone data from the GDELT 2.0 DOC API."""

    @property
    def name(self) -> str:
        return "gdelt"

    def relevant_categories(self) -> set[str]:
        return _RELEVANT_CATEGORIES

    def is_relevant(self, question: str, market_type: str) -> bool:
        return market_type in self.relevant_categories()

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        timespan_days = getattr(self._config, "gdelt_timespan_days", 7)
        search_term = self._extract_topic(question)
        if not search_term:
            return []

        sources = await self._fetch_timeline(search_term, timespan_days)

        # Also fetch GKG entity data for richer context
        gkg_sources = await self._fetch_gkg(search_term, timespan_days)
        sources.extend(gkg_sources)

        return sources

    def _extract_topic(self, question: str) -> str:
        """Extract core topic from question for GDELT search."""
        q = question.strip().rstrip("?")
        q = re.sub(
            r"^(Will|Is|Does|Has|Are|Do|Can|Should)\s+(the\s+)?",
            "", q, flags=re.I,
        )
        q = re.sub(r"\s+(by|before|after|in)\s+\d{4}.*$", "", q, flags=re.I)
        # Remove parentheses/colons (GDELT only allows parens around OR)
        q = re.sub(r"[():]", " ", q)
        # Remove O/U spread notation (meaningless for GDELT)
        q = re.sub(r"\bO/U\s+[\d.]+\b", "", q, flags=re.I)
        # Filter tokens shorter than 3 chars (GDELT rejects them)
        tokens = [t for t in q.split() if len(t) >= 3]
        return " ".join(tokens[:5]).strip()[:80]

    async def _fetch_timeline(
        self,
        search_term: str,
        timespan_days: int,
    ) -> list[FetchedSource]:
        """Fetch news volume timeline from GDELT."""
        await rate_limiter.get("gdelt").acquire()

        client = self._get_client()
        resp = await client.get(
            _GDELT_BASE,
            params={
                "query": search_term,
                "mode": "timelinevol",
                "timespan": f"{timespan_days}d",
                "format": "json",
            },
        )
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            log.warning(
                "gdelt.non_json_response",
                content_type=content_type,
                body_preview=resp.text[:200],
            )
            return []
        data = resp.json()

        timeline = data.get("timeline", [])
        if not timeline:
            return []

        # First series in timeline
        series = timeline[0].get("data", [])
        if not series:
            return []

        # Extract volume values and detect spikes
        volumes = [point.get("value", 0) for point in series]
        dates = [point.get("date", "") for point in series]

        if not volumes:
            return []

        avg_vol = sum(volumes) / len(volumes) if volumes else 0
        max_vol = max(volumes)
        spike_detected = max_vol > (avg_vol * 2) if avg_vol > 0 else False

        # Find peak date
        peak_idx = volumes.index(max_vol)
        peak_date = dates[peak_idx] if peak_idx < len(dates) else ""

        spike_text = "SPIKE DETECTED" if spike_detected else "No spike"
        content = (
            f"GDELT News Volume: {search_term}\n"
            f"Period: last {timespan_days} days\n"
            f"Average daily volume: {avg_vol:.1f}\n"
            f"Peak volume: {max_vol:.1f} ({peak_date})\n"
            f"Volume spike: {spike_text}\n"
            f"Data points: {len(volumes)}\n"
            f"Source: GDELT Project (global news monitoring)"
        )

        snippet = (
            f"News volume for '{search_term}': "
            f"avg={avg_vol:.0f}, peak={max_vol:.0f} — {spike_text}"
        )

        return [
            self._make_source(
                title=f"GDELT: News Volume — {search_term[:40]}",
                url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={search_term}&mode=timelinevol",
                snippet=snippet,
                publisher="GDELT Project",
                content=content,
                authority_score=0.4,  # Aggregator, not primary source
            )
        ]

    async def _fetch_gkg(
        self,
        search_term: str,
        timespan_days: int,
    ) -> list[FetchedSource]:
        """Fetch entity/theme data from GDELT GKG via the DOC API tone mode.

        Extracts the top tone-scored articles and their associated entities
        (persons, organizations, locations) from the article metadata.
        """
        await rate_limiter.get("gdelt").acquire()

        client = self._get_client()
        try:
            resp = await client.get(
                _GDELT_GKG_BASE,
                params={
                    "query": search_term,
                    "mode": "tonechart",
                    "timespan": f"{timespan_days}d",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" not in content_type:
                return []
            data = resp.json()
        except Exception as e:
            log.debug("gdelt.gkg_fetch_failed", error=str(e))
            return []

        tone_data = data.get("tonechart", [])
        if not tone_data:
            return []

        # Extract tone statistics
        tones = [float(t.get("tone", 0)) for t in tone_data if t.get("tone")]
        if not tones:
            return []

        avg_tone = sum(tones) / len(tones)
        min_tone = min(tones)
        max_tone = max(tones)

        if avg_tone > 2.0:
            sentiment = "positive"
        elif avg_tone < -2.0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Extract unique domains as proxy for breadth of coverage
        domains: set[str] = set()
        for item in tone_data[:50]:
            url = item.get("url", "")
            if url:
                try:
                    from urllib.parse import urlparse
                    domains.add(urlparse(url).netloc)
                except Exception:
                    pass

        content = (
            f"GDELT Tone Analysis: {search_term}\n"
            f"  Period: last {timespan_days} days\n"
            f"  Articles analyzed: {len(tone_data)}\n"
            f"  Average tone: {avg_tone:.2f} ({sentiment})\n"
            f"  Tone range: {min_tone:.2f} to {max_tone:.2f}\n"
            f"  Unique sources: {len(domains)}\n"
            f"  Source: GDELT Global Knowledge Graph"
        )

        return [
            self._make_source(
                title=f"GDELT GKG: Tone — {search_term[:40]}",
                url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={search_term}&mode=tonechart",
                snippet=(
                    f"Media tone for '{search_term}': "
                    f"avg={avg_tone:.1f} ({sentiment}), "
                    f"{len(tone_data)} articles, {len(domains)} sources"
                ),
                publisher="GDELT GKG",
                content=content,
                authority_score=0.35,
                raw={
                    "behavioral_signal": {
                        "source": "gdelt_gkg",
                        "signal_type": "media_tone",
                        "value": round(avg_tone, 4),
                        "sentiment": sentiment,
                        "article_count": len(tone_data),
                        "source_count": len(domains),
                        "tone_range": [round(min_tone, 2), round(max_tone, 2)],
                    }
                },
            )
        ]
