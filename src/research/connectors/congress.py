"""Congress.gov research connector — legislative data for ELECTION/LEGAL markets.

Uses the free Congress.gov API (5,000 req/hr) to fetch bill status,
vote records, and nomination updates from the Library of Congress.
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

_CONGRESS_BASE = "https://api.congress.gov/v3"

_POLITICS_KEYWORDS: list[str] = [
    "senate", "congress", "house", "vote", "bill", "legislation",
    "confirmation", "nomination", "law", "policy", "filibuster",
    "amendment", "representative", "senator", "speaker",
    "bipartisan", "veto", "override",
]


class CongressConnector(BaseResearchConnector):
    """Fetch legislative data from the Congress.gov API."""

    @property
    def name(self) -> str:
        return "congress"

    def relevant_categories(self) -> set[str]:
        return {"ELECTION", "LEGAL"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _POLITICS_KEYWORDS)

    def _get_api_key(self) -> str:
        return os.environ.get("CONGRESS_API_KEY", "")

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        api_key = self._get_api_key()
        if not api_key:
            log.debug("congress.no_api_key")
            return []

        max_bills = getattr(self._config, "congress_max_bills", 5)

        # Check for nomination-related keywords first
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["nomination", "confirmation", "confirm", "nominee"]):
            return await self._search_nominations(api_key, question, max_bills)

        # Default: search bills
        search_term = self._extract_search_term(question)
        if not search_term:
            return []

        return await self._search_bills(api_key, search_term, max_bills)

    def _extract_search_term(self, question: str) -> str:
        """Extract core subject from question for Congress.gov search."""
        # Remove question framing
        q = question.strip().rstrip("?")
        q = re.sub(
            r"^(Will|Is|Does|Has|Are|Do|Can|Should)\s+(the\s+)?",
            "", q, flags=re.I,
        )
        # Remove prediction market framing
        q = re.sub(r"\s+(by|before|after|in)\s+\d{4}.*$", "", q, flags=re.I)
        q = re.sub(r"\s+(pass|fail|be\s+signed|become\s+law).*$", "", q, flags=re.I)
        return q.strip()[:100]  # Cap length for API query

    async def _search_bills(
        self,
        api_key: str,
        search_term: str,
        max_bills: int,
    ) -> list[FetchedSource]:
        """Search Congress.gov for bills matching the search term."""
        await rate_limiter.get("congress").acquire()

        client = self._get_client()
        resp = await client.get(
            f"{_CONGRESS_BASE}/bill",
            params={
                "query": search_term,
                "sort": "updateDate+desc",
                "limit": max_bills,
                "api_key": api_key,
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        bills = data.get("bills", [])
        if not bills:
            return []

        sources: list[FetchedSource] = []
        for bill in bills[:max_bills]:
            source = self._format_bill(bill)
            if source:
                sources.append(source)
        return sources

    async def _search_nominations(
        self,
        api_key: str,
        question: str,
        max_results: int,
    ) -> list[FetchedSource]:
        """Search Congress.gov for nominations."""
        await rate_limiter.get("congress").acquire()

        client = self._get_client()
        resp = await client.get(
            f"{_CONGRESS_BASE}/nomination",
            params={
                "limit": max_results,
                "sort": "updateDate+desc",
                "api_key": api_key,
                "format": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        nominations = data.get("nominations", [])
        if not nominations:
            return []

        sources: list[FetchedSource] = []
        for nom in nominations[:max_results]:
            source = self._format_nomination(nom)
            if source:
                sources.append(source)
        return sources

    def _format_bill(self, bill: dict) -> FetchedSource | None:
        """Format a bill result into a FetchedSource."""
        title = bill.get("title", "")
        if not title:
            return None

        bill_type = bill.get("type", "")
        bill_number = bill.get("number", "")
        congress = bill.get("congress", "")
        latest_action = bill.get("latestAction", {})
        action_text = latest_action.get("text", "No recent action")
        action_date = latest_action.get("actionDate", "")

        content = (
            f"Congress.gov: {title}\n"
            f"Bill: {bill_type} {bill_number} ({congress}th Congress)\n"
            f"Latest Action: {action_text}\n"
            f"Action Date: {action_date}\n"
            f"URL: https://congress.gov/bill/{congress}th-congress/"
            f"{bill_type.lower()}-bill/{bill_number}"
        )

        snippet = f"{bill_type}{bill_number}: {action_text[:80]}"

        # Use public congress.gov URL (not API URL which requires auth)
        public_url = (
            f"https://congress.gov/bill/{congress}th-congress/"
            f"{bill_type.lower()}-bill/{bill_number}"
        )

        return self._make_source(
            title=f"Congress.gov: {bill_type}{bill_number} — {title[:60]}",
            url=public_url,
            snippet=snippet,
            publisher="Congress.gov",
            date=action_date,
            content=content,
        )

    def _format_nomination(self, nom: dict) -> FetchedSource | None:
        """Format a nomination result into a FetchedSource."""
        description = nom.get("description", "")
        if not description:
            return None

        latest_action = nom.get("latestAction", {})
        action_text = latest_action.get("text", "No recent action")
        action_date = latest_action.get("actionDate", "")
        congress = nom.get("congress", "")

        content = (
            f"Congress.gov Nomination: {description}\n"
            f"Latest Action: {action_text}\n"
            f"Action Date: {action_date}\n"
            f"Congress: {congress}"
        )

        snippet = f"Nomination: {description[:60]} — {action_text[:40]}"

        # Use public congress.gov URL (not API URL which requires auth)
        nom_number = nom.get("number", "")
        public_url = (
            f"https://congress.gov/nomination/{congress}th-congress/{nom_number}"
            if congress and nom_number
            else "https://congress.gov"
        )

        return self._make_source(
            title=f"Congress.gov Nomination: {description[:60]}",
            url=public_url,
            snippet=snippet,
            publisher="Congress.gov",
            date=action_date,
            content=content,
        )
