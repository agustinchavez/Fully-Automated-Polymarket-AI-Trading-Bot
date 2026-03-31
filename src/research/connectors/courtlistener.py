"""CourtListener research connector — legal case data for LEGAL markets.

Uses the free CourtListener API (token required, free signup) to search
court opinions and docket entries from the RECAP archive.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_CL_BASE = "https://www.courtlistener.com/api/rest/v4/search/"

_LEGAL_KEYWORDS: list[str] = [
    "court", "ruling", "decision", "lawsuit", "case", "appeal",
    "supreme court", "judge", "verdict", "injunction", "ruling",
    "litigation", "plaintiff", "defendant", "opinion",
]

_MAX_OPINION_CHARS = 2_000


class CourtListenerConnector(BaseResearchConnector):
    """Fetch legal case data from the CourtListener API."""

    @property
    def name(self) -> str:
        return "courtlistener"

    def relevant_categories(self) -> set[str]:
        return {"LEGAL"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _LEGAL_KEYWORDS)

    def _get_api_key(self) -> str:
        return os.environ.get("COURTLISTENER_API_KEY", "")

    def _is_maintenance_window(self) -> bool:
        """Check if CourtListener is in Thursday maintenance (21:00-23:59 PT)."""
        now_utc = datetime.now(timezone.utc)
        # PT is UTC-8 (PST) or UTC-7 (PDT) — use UTC-8 as conservative bound
        pt_hour = (now_utc.hour - 8) % 24
        pt_weekday = now_utc.weekday()
        # Adjust weekday if hour wrap changes the day
        if now_utc.hour < 8:
            pt_weekday = (pt_weekday - 1) % 7

        # Thursday = 3, maintenance 21:00-23:59 PT
        return pt_weekday == 3 and pt_hour >= 21

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        api_key = self._get_api_key()
        if not api_key:
            log.debug("courtlistener.no_api_key")
            return []

        if self._is_maintenance_window():
            log.debug("courtlistener.maintenance_window")
            return []

        search_term = self._extract_legal_term(question)
        if not search_term:
            return []

        # Try opinions first, then dockets
        sources = await self._search_opinions(api_key, search_term)
        if not sources:
            sources = await self._search_dockets(api_key, search_term)

        return sources

    def _extract_legal_term(self, question: str) -> str:
        """Extract legal search term from question."""
        import re

        q = question.strip().rstrip("?")
        q = re.sub(
            r"^(Will|Is|Does|Has|Are|Do|Can|Should)\s+(the\s+)?",
            "", q, flags=re.I,
        )
        q = re.sub(r"\s+(by|before|after|in)\s+\d{4}.*$", "", q, flags=re.I)
        q = re.sub(r"\s+(rule|decide|overturn|uphold).*$", "", q, flags=re.I)
        return q.strip()[:100]

    async def _search_opinions(
        self,
        api_key: str,
        search_term: str,
        max_results: int = 3,
    ) -> list[FetchedSource]:
        """Search CourtListener for court opinions."""
        await rate_limiter.get("courtlistener").acquire()

        client = self._get_client()
        resp = await client.get(
            _CL_BASE,
            params={
                "q": search_term,
                "type": "o",  # opinions
                "order_by": "score desc",
            },
            headers={"Authorization": f"Token {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return []

        sources: list[FetchedSource] = []
        for result in results[:max_results]:
            source = self._format_opinion(result)
            if source:
                sources.append(source)
        return sources

    async def _search_dockets(
        self,
        api_key: str,
        search_term: str,
        max_results: int = 3,
    ) -> list[FetchedSource]:
        """Search CourtListener for docket entries."""
        await rate_limiter.get("courtlistener").acquire()

        client = self._get_client()
        resp = await client.get(
            _CL_BASE,
            params={
                "q": search_term,
                "type": "r",  # RECAP dockets
                "order_by": "score desc",
            },
            headers={"Authorization": f"Token {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return []

        sources: list[FetchedSource] = []
        for result in results[:max_results]:
            source = self._format_docket(result)
            if source:
                sources.append(source)
        return sources

    def _format_opinion(self, result: dict) -> FetchedSource | None:
        """Format a CourtListener opinion result."""
        case_name = result.get("caseName", "")
        if not case_name:
            return None

        court = result.get("court", "")
        date_filed = result.get("dateFiled", "")
        snippet_text = result.get("snippet", "")
        # Truncate opinion text
        if len(snippet_text) > _MAX_OPINION_CHARS:
            snippet_text = snippet_text[:_MAX_OPINION_CHARS] + "..."

        abs_url = result.get("absolute_url", "")
        url = f"https://www.courtlistener.com{abs_url}" if abs_url else "https://www.courtlistener.com"

        content = (
            f"CourtListener Opinion: {case_name}\n"
            f"Court: {court}\n"
            f"Date Filed: {date_filed}\n"
            f"Excerpt: {snippet_text}\n"
        )

        return self._make_source(
            title=f"CourtListener: {case_name[:60]}",
            url=url,
            snippet=f"{case_name[:50]} — {court}",
            publisher="CourtListener",
            date=date_filed,
            content=content,
        )

    def _format_docket(self, result: dict) -> FetchedSource | None:
        """Format a CourtListener docket result."""
        case_name = result.get("caseName", "")
        if not case_name:
            return None

        court = result.get("court", "")
        date_filed = result.get("dateFiled", "")
        docket_number = result.get("docketNumber", "")

        abs_url = result.get("absolute_url", "")
        url = f"https://www.courtlistener.com{abs_url}" if abs_url else "https://www.courtlistener.com"

        content = (
            f"CourtListener Docket: {case_name}\n"
            f"Docket Number: {docket_number}\n"
            f"Court: {court}\n"
            f"Date Filed: {date_filed}\n"
        )

        return self._make_source(
            title=f"CourtListener Docket: {case_name[:50]}",
            url=url,
            snippet=f"{case_name[:40]} — Docket {docket_number}",
            publisher="CourtListener",
            date=date_filed,
            content=content,
        )
