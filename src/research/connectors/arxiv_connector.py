"""arXiv research connector — preprints for SCIENCE markets.

Uses the free arXiv API (no key required, 1 req per 3 seconds):
- Endpoint: export.arxiv.org/api/query
- Response: Atom XML parsed with stdlib xml.etree.ElementTree
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = "{http://www.w3.org/2005/Atom}"

_SCIENCE_KEYWORDS: list[str] = [
    "fda", "trial", "study", "research", "paper", "drug", "vaccine",
    "ai model", "gpt", "claude", "benchmark", "published", "journal",
    "clinical", "scientific", "experiment", "preprint", "arxiv",
    "machine learning", "deep learning", "neural", "quantum",
    "fusion", "crispr", "genome", "protein", "particle",
]


class ArxivConnector(BaseResearchConnector):
    """Fetch recent preprints from arXiv."""

    @property
    def name(self) -> str:
        return "arxiv"

    def relevant_categories(self) -> set[str]:
        return {"SCIENCE"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _SCIENCE_KEYWORDS)

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        max_results = getattr(self._config, "arxiv_max_results", 5)
        query = self._build_query(question)
        if not query:
            return []

        await rate_limiter.get("arxiv").acquire()

        client = self._get_client()
        resp = await client.get(
            _ARXIV_API,
            params={
                "search_query": query,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        resp.raise_for_status()

        return self._parse_atom(resp.text, max_results)

    def _build_query(self, question: str) -> str:
        """Extract key terms from question and format as arXiv query."""
        # Remove common question words
        cleaned = re.sub(
            r"\b(will|the|a|an|in|of|to|for|and|or|is|be|by|it|this|that|"
            r"does|do|can|should|has|have|what|when|where|how|why|which)\b",
            "",
            question.lower(),
        )
        terms = [t for t in cleaned.split() if len(t) > 2]
        if not terms:
            return ""

        # Use top 4 terms for title/abstract search
        terms = terms[:4]
        parts = []
        for term in terms:
            parts.append(f"ti:{term} OR abs:{term}")
        return " AND ".join(f"({p})" for p in parts)

    def _parse_atom(self, xml_text: str, max_results: int) -> list[FetchedSource]:
        """Parse Atom XML response from arXiv API."""
        sources: list[FetchedSource] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            log.warning("arxiv.xml_parse_error")
            return []

        entries = root.findall(f"{_ATOM_NS}entry")
        for entry in entries[:max_results]:
            title_el = entry.find(f"{_ATOM_NS}title")
            summary_el = entry.find(f"{_ATOM_NS}summary")
            published_el = entry.find(f"{_ATOM_NS}published")
            id_el = entry.find(f"{_ATOM_NS}id")

            title = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"
            abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
            published = published_el.text.strip()[:10] if published_el is not None and published_el.text else ""
            arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""

            # Authors (top 3)
            author_els = entry.findall(f"{_ATOM_NS}author")
            authors = []
            for a in author_els[:3]:
                name_el = a.find(f"{_ATOM_NS}name")
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            if len(author_els) > 3:
                authors.append(f"et al. (+{len(author_els) - 3})")

            # Categories
            cat_els = entry.findall("{http://arxiv.org/schemas/atom}primary_category")
            categories = []
            for cat in cat_els:
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            # Build content
            abstract_trimmed = abstract[:600]
            if len(abstract) > 600:
                abstract_trimmed += "..."

            content_lines = [
                f"Title: {title}",
                f"Authors: {', '.join(authors)}",
                f"Published: {published}",
            ]
            if categories:
                content_lines.append(f"Categories: {', '.join(categories)}")
            content_lines.append(f"\nAbstract: {abstract_trimmed}")
            content_lines.append(f"\nURL: {arxiv_url}")
            content_lines.append("Source: arXiv.org")

            content = "\n".join(content_lines)
            snippet = f"{title} — {', '.join(authors[:2])} ({published})"

            sources.append(self._make_source(
                title=f"arXiv: {title}",
                url=arxiv_url,
                snippet=snippet,
                publisher="arXiv.org",
                date=published,
                content=content,
            ))

        return sources
