"""PubMed research connector -- NIH Entrez API for SCIENCE markets.

Uses the free NCBI Entrez e-utilities (optional API key for higher limits):
- Step 1: esearch.fcgi -- search PubMed and retrieve PMIDs
- Step 2: esummary.fcgi -- fetch article summaries by PMID
- Rate limit: 3 req/s without key, 10 req/s with key
"""

from __future__ import annotations

import os
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_PUBMED_KEYWORDS: list[str] = [
    "fda", "drug", "trial", "vaccine", "clinical", "phase 3",
    "cancer", "treatment", "therapy", "disease", "virus", "pandemic",
    "health", "medical", "protein", "gene", "cell", "mutation",
    "study", "research",
]


class PubMedConnector(BaseResearchConnector):
    """Fetch recent PubMed article summaries from the NIH Entrez API."""

    @property
    def name(self) -> str:
        return "pubmed"

    def relevant_categories(self) -> set[str]:
        return {"SCIENCE"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type not in self.relevant_categories():
            return False
        return self._question_matches_keywords(question, _PUBMED_KEYWORDS)

    def _get_api_key(self) -> str:
        """Read NCBI API key from config or environment (optional)."""
        if self._config:
            key = getattr(self._config, "pubmed_api_key", "")
            if key:
                return key
        return os.environ.get("NCBI_API_KEY", "")

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        max_results = getattr(self._config, "pubmed_max_results", 5)
        api_key = self._get_api_key()

        # Step 1: search for PMIDs
        pmids = await self._esearch(question, max_results, api_key)
        if not pmids:
            return []

        # Step 2: fetch article summaries
        return await self._esummary(pmids, api_key)

    async def _esearch(
        self,
        question: str,
        max_results: int,
        api_key: str,
    ) -> list[str]:
        """Search PubMed and return a list of PMID strings."""
        await rate_limiter.get("pubmed").acquire()

        client = self._get_client()
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": question,
            "retmax": max_results,
            "retmode": "json",
            "sort": "date",
        }
        if api_key:
            params["api_key"] = api_key

        resp = await client.get(_ESEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list

    async def _esummary(
        self,
        pmids: list[str],
        api_key: str,
    ) -> list[FetchedSource]:
        """Fetch article summaries for a list of PMIDs."""
        await rate_limiter.get("pubmed").acquire()

        client = self._get_client()
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key

        resp = await client.get(_ESUMMARY_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        result_block = data.get("result", {})
        uid_list = result_block.get("uids", [])

        sources: list[FetchedSource] = []
        for uid in uid_list:
            article = result_block.get(uid)
            if not article:
                continue

            source = self._parse_article(article)
            if source:
                sources.append(source)

        return sources

    def _parse_article(self, article: dict[str, Any]) -> FetchedSource | None:
        """Parse a single esummary article record into a FetchedSource."""
        pmid = article.get("uid", "")
        title = article.get("title", "Untitled")
        pubdate = article.get("pubdate", "")
        journal = article.get("source", "")

        if not pmid:
            return None

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        content_lines = [
            f"Title: {title}",
            f"Journal: {journal}",
            f"Published: {pubdate}",
            f"PMID: {pmid}",
            f"URL: {url}",
            "Source: PubMed (NIH/NLM)",
        ]
        content = "\n".join(content_lines)
        snippet = f"{title} -- {journal} ({pubdate})"

        return self._make_source(
            title=f"PubMed: {title}",
            url=url,
            snippet=snippet,
            publisher=journal,
            date=pubdate,
            content=content,
            authority_score=1.0,
        )
