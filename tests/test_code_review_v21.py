"""Code review v21 fixes — page content shared timeout removed,
daily budget raised to 25 USD.

Tests cover:
  1. Page content fetch no longer uses shared asyncio.wait_for wrapper (Issue 1)
  2. daily_limit_usd raised to 25 for continuous operation (Issue 2)
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Issue 1: Page content fetch — no shared timeout ─────────────────


class TestPageContentNoSharedTimeout:
    """Content fetch should rely on per-request httpx timeout, not shared gather wrapper."""

    def test_no_wait_for_in_content_fetch(self) -> None:
        """fetch_sources content section has no asyncio.wait_for wrapping gather."""
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_sources)
        # The old pattern: asyncio.wait_for(asyncio.gather(...), timeout=30.0)
        # Should no longer exist in the content fetch section
        assert "timeout=30.0" not in source

    def test_content_fetch_uses_bare_gather(self) -> None:
        """Content fetch uses asyncio.gather directly without wait_for."""
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_sources)
        # Should have gather with return_exceptions=True for content
        assert "asyncio.gather(*content_tasks, return_exceptions=True)" in source

    @pytest.mark.asyncio
    async def test_slow_page_does_not_cancel_fast_pages(self) -> None:
        """One slow page fetch doesn't cancel fast ones."""
        from src.research.source_fetcher import SourceFetcher, FetchedSource
        from src.connectors.web_search import SearchResult

        config = MagicMock()
        config.source_timeout_secs = 1
        config.max_sources = 10
        config.fetch_full_content = True
        config.content_fetch_top_n = 3
        config.primary_domains = {}
        config.secondary_domains = []
        config.blocked_domains = []

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = config
        fetcher._connectors = []

        # Pre-populate top sources (skip search phase)
        fast_source1 = FetchedSource(
            title="fast1", url="http://fast1.com/article", snippet="ok",
            authority_score=1.0,
        )
        fast_source2 = FetchedSource(
            title="fast2", url="http://fast2.com/article", snippet="ok",
            authority_score=0.9,
        )
        slow_source = FetchedSource(
            title="slow", url="http://slow.com/article", snippet="ok",
            authority_score=0.8,
        )

        async def mock_fetch_page(url: str) -> str:
            if "slow" in url:
                # Simulates per-request httpx timeout — the page is slow
                # and eventually the httpx client returns empty.
                await asyncio.sleep(0.5)
                return ""  # httpx timeout → empty content
            return f"Content from {url}"

        fetcher.fetch_page_content = mock_fetch_page
        fetcher._run_query = AsyncMock(return_value=[])

        # Directly test the content fetch portion by calling fetch_sources
        # with queries that return no results (content fetch uses pre-populated top)
        from src.research.query_builder import SearchQuery
        queries = [SearchQuery(text="test", intent="primary")]

        # We need to mock _run_query to return results that populate `top`
        fast_result1 = SearchResult(
            title="fast1", url="http://fast1.com/article", snippet="ok",
            source="fast1.com", date="", position=1, raw={},
        )
        fast_result2 = SearchResult(
            title="fast2", url="http://fast2.com/article", snippet="ok",
            source="fast2.com", date="", position=2, raw={},
        )
        slow_result = SearchResult(
            title="slow", url="http://slow.com/article", snippet="ok",
            source="slow.com", date="", position=3, raw={},
        )

        fetcher._run_query = AsyncMock(
            return_value=[fast_result1, fast_result2, slow_result]
        )

        # Use a timeout to prevent test hanging — if content fetch
        # uses shared timeout, fast pages would get cancelled
        results = await asyncio.wait_for(
            fetcher.fetch_sources(queries),
            timeout=5.0,  # generous test timeout
        )

        # Fast pages should have content populated
        fast_results = [r for r in results if "fast" in r.title]
        assert len(fast_results) == 2
        for r in fast_results:
            assert r.content != ""
            assert r.extraction_method == "html"

    def test_no_content_fetch_timeout_log(self) -> None:
        """The old 'content_fetch_timeout' warning log is removed."""
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_sources)
        assert "content_fetch_timeout" not in source


# ── Issue 2: daily_limit_usd raised to 25 ───────────────────────────


class TestDailyBudgetRaised:
    """Daily budget should be 25 USD for continuous operation with gating."""

    def test_daily_limit_25(self) -> None:
        """config.yaml has daily_limit_usd: 25.0."""
        from src.config import load_config

        config = load_config()
        assert config.budget.daily_limit_usd == 25.0

    def test_budget_headroom_above_gating_cost(self) -> None:
        """Budget has headroom above estimated gated cost (~12-18 USD/day)."""
        from src.config import load_config

        config = load_config()
        # With gating + 5 markets, expected cost ~12-18 USD/day
        # Budget should be above the high estimate
        assert config.budget.daily_limit_usd >= 20.0

    def test_warning_threshold_fires_before_limit(self) -> None:
        """Warning at 80% of 25 = 20 USD — alerts before hitting limit."""
        from src.config import load_config

        config = load_config()
        warning_usd = config.budget.daily_limit_usd * config.budget.warning_pct
        assert warning_usd >= 18.0  # fires at $20, above typical daily cost
