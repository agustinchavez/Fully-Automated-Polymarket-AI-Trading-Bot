"""Base class for structured API research connectors.

Each connector fetches data from a free public API and returns
``FetchedSource`` objects with ``extraction_method='api'``.  The
base class wraps every call in a try/except so a failing connector
can never break the research pipeline.
"""

from __future__ import annotations

import abc
from typing import Any

import httpx

from src.observability.logger import get_logger
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)


class BaseResearchConnector(abc.ABC):
    """Abstract base for all research API connectors."""

    def __init__(self, config: Any = None) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None

    # ── Identity ─────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and config keys."""

    @abc.abstractmethod
    def relevant_categories(self) -> set[str]:
        """Market categories this connector can serve."""

    # ── Core interface ───────────────────────────────────────────

    @abc.abstractmethod
    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        """Subclass implementation — may raise."""

    async def fetch(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        """Safe wrapper — catches all exceptions, returns [] on failure."""
        try:
            return await self._fetch_impl(question, market_type)
        except Exception as e:
            log.warning(
                f"research_connector.{self.name}.failed",
                error=str(e),
            )
            return []

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self, timeout: float = 15.0) -> httpx.AsyncClient:
        """Lazy-initialise a shared httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    # ── Helpers ──────────────────────────────────────────────────

    def _make_source(
        self,
        *,
        title: str,
        url: str,
        snippet: str,
        publisher: str,
        date: str = "",
        content: str = "",
        authority_score: float = 1.0,
        query_intent: str = "primary",
        raw: dict[str, Any] | None = None,
    ) -> FetchedSource:
        """Create a ``FetchedSource`` with ``extraction_method='api'``."""
        return FetchedSource(
            title=title,
            url=url,
            snippet=snippet,
            publisher=publisher,
            date=date,
            content=content,
            authority_score=authority_score,
            query_intent=query_intent,
            extraction_method="api",
            content_length=len(content),
            raw=raw or {},
        )

    def is_relevant(self, question: str, market_type: str) -> bool:
        """Check if this connector should run for the given question.

        Default: returns True if ``market_type`` is in ``relevant_categories()``.
        Override in subclass to also check question keywords.
        """
        return market_type in self.relevant_categories()

    def _question_matches_keywords(
        self, question: str, keywords: list[str],
    ) -> bool:
        """Return True if any keyword appears in the question (case-insensitive)."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in keywords)
