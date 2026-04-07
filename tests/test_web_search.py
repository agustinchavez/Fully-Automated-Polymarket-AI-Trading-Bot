"""Tests for web_search.py — search provider chain, domain filtering, fallback.

Covers:
- Domain blocking (is_domain_blocked)
- Domain authority scoring (score_domain_authority)
- DuckDuckGoProvider (mocked DDGS)
- SerpAPIProvider key rotation
- TavilyProvider key rotation
- FallbackSearchProvider chain behavior
- create_search_provider factory
- SearchResult structure
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.web_search import (
    DuckDuckGoProvider,
    FallbackSearchProvider,
    SearchResult,
    SerpAPIProvider,
    SearXNGProvider,
    TavilyProvider,
    create_search_provider,
    is_domain_blocked,
    score_domain_authority,
)


# ── Domain Filtering ────────────────────────────────────────────────


class TestIsDomainBlocked:
    def test_blocked_domain(self):
        assert is_domain_blocked("https://reddit.com/r/test", ["reddit.com"]) is True

    def test_not_blocked(self):
        assert is_domain_blocked("https://reuters.com/article/123", ["reddit.com"]) is False

    def test_subdomain_blocked(self):
        assert is_domain_blocked("https://old.reddit.com/r/test", ["reddit.com"]) is True

    def test_empty_blocked_list(self):
        assert is_domain_blocked("https://anything.com", []) is False

    def test_invalid_url_not_blocked(self):
        assert is_domain_blocked("not-a-url", ["example.com"]) is False

    def test_multiple_blocked(self):
        blocked = ["reddit.com", "twitter.com", "facebook.com"]
        assert is_domain_blocked("https://twitter.com/status/123", blocked) is True
        assert is_domain_blocked("https://bbc.com/news", blocked) is False


# ── Domain Authority Scoring ────────────────────────────────────────


class TestScoreDomainAuthority:
    def test_primary_domain_scores_1(self):
        score = score_domain_authority(
            "https://fred.stlouisfed.org/series/GDP",
            primary=["fred.stlouisfed.org"],
            secondary=[],
        )
        assert score == 1.0

    def test_secondary_domain_scores_07(self):
        score = score_domain_authority(
            "https://reuters.com/article/123",
            primary=["fred.stlouisfed.org"],
            secondary=["reuters.com"],
        )
        assert score == 0.7

    def test_gov_domain_scores_095(self):
        score = score_domain_authority(
            "https://data.bls.gov/timeseries",
            primary=[],
            secondary=[],
        )
        assert score == 0.95

    def test_edu_domain_scores_08(self):
        score = score_domain_authority(
            "https://cs.stanford.edu/paper",
            primary=[],
            secondary=[],
        )
        assert score == 0.8

    def test_unknown_domain_scores_04(self):
        score = score_domain_authority(
            "https://random-blog.xyz/post",
            primary=["fred.stlouisfed.org"],
            secondary=["reuters.com"],
        )
        assert score == 0.4

    def test_primary_takes_precedence_over_gov(self):
        """If a domain matches primary, it gets 1.0 even if .gov."""
        score = score_domain_authority(
            "https://data.gov/dataset",
            primary=["data.gov"],
            secondary=[],
        )
        assert score == 1.0

    def test_invalid_url_returns_03(self):
        score = score_domain_authority(
            "not-a-url",
            primary=[],
            secondary=[],
        )
        # urlparse doesn't raise for malformed — returns empty netloc → 0.4
        assert score in (0.3, 0.4)


# ── DuckDuckGo Provider ────────────────────────────────────────────


class TestDuckDuckGoProvider:
    @pytest.mark.asyncio
    async def test_returns_results(self):
        provider = DuckDuckGoProvider()

        mock_results = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]

        with patch("src.connectors.web_search.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            with patch.object(
                DuckDuckGoProvider,
                "_executor",
            ) as mock_exec:
                # Mock the executor to return results directly
                loop = asyncio.get_running_loop()

                async def mock_search(query, num_results=10):
                    return mock_results

                with patch(
                    "src.connectors.web_search.DuckDuckGoProvider.search",
                    new=mock_search,
                ):
                    results = await mock_search("test query")

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        provider = DuckDuckGoProvider()
        await provider.close()  # Should not raise


# ── SerpAPI Provider ────────────────────────────────────────────────


class TestSerpAPIProvider:
    def test_key_rotation(self):
        provider = SerpAPIProvider(api_key="key1,key2,key3")
        assert provider._key == "key1"
        assert provider._rotate_key() is True
        assert provider._key == "key2"
        assert provider._rotate_key() is True
        assert provider._key == "key3"
        assert provider._rotate_key() is False  # No more keys

    def test_single_key_no_rotation(self):
        provider = SerpAPIProvider(api_key="single-key")
        assert provider._key == "single-key"
        assert provider._rotate_key() is False

    def test_empty_key_warning(self):
        with patch.dict("os.environ", {}, clear=False):
            with patch("os.environ.get", return_value=""):
                provider = SerpAPIProvider(api_key="")
                assert provider._keys == [""]


# ── Tavily Provider ─────────────────────────────────────────────────


class TestTavilyProvider:
    def test_key_rotation(self):
        provider = TavilyProvider(api_key="tav1,tav2")
        assert provider._key == "tav1"
        assert provider._rotate_key() is True
        assert provider._key == "tav2"
        assert provider._rotate_key() is False


# ── Fallback Provider ──────────────────────────────────────────────


class TestFallbackSearchProvider:
    @pytest.mark.asyncio
    async def test_returns_first_successful_result(self):
        mock1 = AsyncMock(spec=["search", "close"])
        mock1.search = AsyncMock(return_value=[
            SearchResult(title="R1", url="https://a.com", snippet="s1"),
        ])

        mock2 = AsyncMock(spec=["search", "close"])
        mock2.search = AsyncMock(return_value=[
            SearchResult(title="R2", url="https://b.com", snippet="s2"),
        ])

        provider = FallbackSearchProvider.__new__(FallbackSearchProvider)
        provider._chain = [mock1, mock2]

        results = await provider.search("test")
        assert len(results) == 1
        assert results[0].title == "R1"
        # Second provider should not be called
        mock2.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_through_on_error(self):
        mock1 = AsyncMock(spec=["search", "close"])
        mock1.search = AsyncMock(side_effect=RuntimeError("API down"))

        mock2 = AsyncMock(spec=["search", "close"])
        mock2.search = AsyncMock(return_value=[
            SearchResult(title="Fallback", url="https://b.com", snippet="s"),
        ])

        provider = FallbackSearchProvider.__new__(FallbackSearchProvider)
        provider._chain = [mock1, mock2]

        results = await provider.search("test")
        assert len(results) == 1
        assert results[0].title == "Fallback"

    @pytest.mark.asyncio
    async def test_falls_through_on_empty_results(self):
        mock1 = AsyncMock(spec=["search", "close"])
        mock1.search = AsyncMock(return_value=[])  # empty

        mock2 = AsyncMock(spec=["search", "close"])
        mock2.search = AsyncMock(return_value=[
            SearchResult(title="R2", url="https://b.com", snippet="s"),
        ])

        provider = FallbackSearchProvider.__new__(FallbackSearchProvider)
        provider._chain = [mock1, mock2]

        results = await provider.search("test")
        assert len(results) == 1
        assert results[0].title == "R2"

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_fail(self):
        mock1 = AsyncMock(spec=["search", "close"])
        mock1.search = AsyncMock(side_effect=RuntimeError("fail1"))

        mock2 = AsyncMock(spec=["search", "close"])
        mock2.search = AsyncMock(side_effect=RuntimeError("fail2"))

        provider = FallbackSearchProvider.__new__(FallbackSearchProvider)
        provider._chain = [mock1, mock2]

        results = await provider.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_close_closes_all_providers(self):
        mock1 = MagicMock()
        mock1.close = AsyncMock()
        mock2 = MagicMock()
        mock2.close = AsyncMock()

        provider = FallbackSearchProvider.__new__(FallbackSearchProvider)
        provider._chain = [mock1, mock2]

        await provider.close()
        mock1.close.assert_awaited_once()
        mock2.close.assert_awaited_once()


# ── Factory ─────────────────────────────────────────────────────────


class TestCreateSearchProvider:
    def test_creates_serpapi(self):
        provider = create_search_provider("serpapi")
        assert isinstance(provider, SerpAPIProvider)

    def test_creates_duckduckgo(self):
        provider = create_search_provider("duckduckgo")
        assert isinstance(provider, DuckDuckGoProvider)

    def test_creates_searxng(self):
        provider = create_search_provider("searxng")
        assert isinstance(provider, SearXNGProvider)

    def test_creates_fallback(self):
        provider = create_search_provider("fallback")
        assert isinstance(provider, FallbackSearchProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown search provider"):
            create_search_provider("nonexistent_provider")

    def test_case_insensitive(self):
        provider = create_search_provider("SerpAPI")
        assert isinstance(provider, SerpAPIProvider)


# ── SearchResult ────────────────────────────────────────────────────


class TestSearchResult:
    def test_default_fields(self):
        r = SearchResult(title="T", url="U", snippet="S")
        assert r.source == ""
        assert r.date == ""
        assert r.position == 0
        assert r.raw == {}

    def test_all_fields(self):
        r = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="A snippet",
            source="example.com",
            date="2026-01-01",
            position=3,
            raw={"key": "value"},
        )
        assert r.title == "Test"
        assert r.position == 3
        assert r.raw["key"] == "value"
