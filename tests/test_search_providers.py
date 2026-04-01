"""Tests for DuckDuckGo and SearXNG search providers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.connectors.web_search import (
    DuckDuckGoProvider,
    FallbackSearchProvider,
    SearchResult,
    SearXNGProvider,
    create_search_provider,
    is_domain_blocked,
    score_domain_authority,
)


# ── DuckDuckGo Provider ────────────────────────────────────────────


class TestDuckDuckGoProvider:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_results = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]
        with patch("src.connectors.web_search.DDGS", create=True) as MockDDGS:
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            MockDDGS.return_value = mock_ddgs

            # Patch the import inside the method
            provider = DuckDuckGoProvider()
            with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=MockDDGS)}):
                results = await provider.search("test query", num_results=5)

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert results[0].position == 1
        assert results[1].position == 2

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock()}):
            import sys
            mock_module = sys.modules["duckduckgo_search"]
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = []
            mock_module.DDGS.return_value = mock_ddgs

            provider = DuckDuckGoProvider()
            results = await provider.search("no results query")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_extracts_source_from_url(self):
        mock_results = [
            {"title": "Test", "href": "https://www.reuters.com/article/123", "body": "News"},
        ]
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock()}):
            import sys
            mock_module = sys.modules["duckduckgo_search"]
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_module.DDGS.return_value = mock_ddgs

            provider = DuckDuckGoProvider()
            results = await provider.search("reuters news")

        assert results[0].source == "www.reuters.com"

    @pytest.mark.asyncio
    async def test_search_missing_import_raises(self):
        """DDG provider raises RuntimeError when library not installed."""
        # Directly test the import guard without retry overhead
        provider = DuckDuckGoProvider()
        original_search = provider.search.__wrapped__  # unwrap tenacity
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            with pytest.raises(RuntimeError, match="duckduckgo-search not installed"):
                await original_search(provider, "test")

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        provider = DuckDuckGoProvider()
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_search_handles_link_key_fallback(self):
        """DDG sometimes returns 'link' instead of 'href'."""
        mock_results = [
            {"title": "Alt", "link": "https://example.com/alt", "snippet": "Alt snippet"},
        ]
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock()}):
            import sys
            mock_module = sys.modules["duckduckgo_search"]
            mock_ddgs = MagicMock()
            mock_ddgs.text.return_value = mock_results
            mock_module.DDGS.return_value = mock_ddgs

            provider = DuckDuckGoProvider()
            results = await provider.search("alt query")

        assert results[0].url == "https://example.com/alt"
        assert results[0].snippet == "Alt snippet"


# ── SearXNG Provider ──────────────────────────────────────────────


class TestSearXNGProvider:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "SearXNG Result",
                    "url": "https://example.com/searxng",
                    "content": "Found via SearXNG",
                    "engine": "google",
                    "publishedDate": "2025-01-01",
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        provider = SearXNGProvider(base_url="http://test:8080")
        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            results = await provider.search("test query")

        assert len(results) == 1
        assert results[0].title == "SearXNG Result"
        assert results[0].url == "https://example.com/searxng"
        assert results[0].snippet == "Found via SearXNG"
        assert results[0].source == "google"
        assert results[0].date == "2025-01-01"
        assert results[0].position == 1

    @pytest.mark.asyncio
    async def test_search_respects_num_results(self):
        mock_response = MagicMock()
        items = [
            {"title": f"R{i}", "url": f"https://example.com/{i}", "content": f"S{i}"}
            for i in range(20)
        ]
        mock_response.json.return_value = {"results": items}
        mock_response.raise_for_status = MagicMock()

        provider = SearXNGProvider(base_url="http://test:8080")
        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            results = await provider.search("test", num_results=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_sends_correct_params(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        provider = SearXNGProvider(base_url="http://searxng:8080/")
        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            await provider.search("prediction markets", num_results=10)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "http://searxng:8080/search"
        assert call_args[1]["params"]["q"] == "prediction markets"
        assert call_args[1]["params"]["format"] == "json"

    @pytest.mark.asyncio
    async def test_default_url_from_env(self):
        with patch.dict("os.environ", {"SEARXNG_URL": "http://custom:9090"}):
            provider = SearXNGProvider()
        assert provider._base_url == "http://custom:9090"

    @pytest.mark.asyncio
    async def test_default_url_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            provider = SearXNGProvider()
        assert provider._base_url == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        provider = SearXNGProvider(base_url="http://test:8080/")
        assert provider._base_url == "http://test:8080"

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        provider = SearXNGProvider(base_url="http://test:8080")
        with patch.object(provider._client, "aclose", new_callable=AsyncMock) as mock_close:
            await provider.close()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        provider = SearXNGProvider(base_url="http://test:8080")
        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            results = await provider.search("nothing")

        assert results == []

    @pytest.mark.asyncio
    async def test_engines_list_as_source(self):
        """When source is a list (engines field), extract first item."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Multi",
                    "url": "https://example.com",
                    "content": "Multi-engine",
                    "engines": ["bing", "google"],
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        provider = SearXNGProvider(base_url="http://test:8080")
        with patch.object(provider._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            results = await provider.search("multi")

        assert results[0].source == "bing"


# ── Factory ───────────────────────────────────────────────────────


class TestSearchFactory:
    def test_create_duckduckgo(self):
        provider = create_search_provider("duckduckgo")
        assert isinstance(provider, DuckDuckGoProvider)

    def test_create_searxng(self):
        provider = create_search_provider("searxng")
        assert isinstance(provider, SearXNGProvider)

    def test_create_fallback(self):
        provider = create_search_provider("fallback")
        assert isinstance(provider, FallbackSearchProvider)

    def test_fallback_default_chain_starts_with_duckduckgo(self):
        provider = FallbackSearchProvider()
        assert isinstance(provider._chain[0], DuckDuckGoProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown search provider"):
            create_search_provider("nonexistent")

    def test_case_insensitive(self):
        provider = create_search_provider("DuckDuckGo")
        assert isinstance(provider, DuckDuckGoProvider)


# ── Domain Filtering ──────────────────────────────────────────────


class TestDomainFiltering:
    def test_blocked_domain(self):
        assert is_domain_blocked("https://reddit.com/r/test", ["reddit.com"])

    def test_allowed_domain(self):
        assert not is_domain_blocked("https://reuters.com/article", ["reddit.com"])

    def test_primary_authority(self):
        score = score_domain_authority(
            "https://reuters.com/news", ["reuters.com"], []
        )
        assert score == 1.0

    def test_secondary_authority(self):
        score = score_domain_authority(
            "https://bbc.co.uk/news", [], ["bbc.co.uk"]
        )
        assert score == 0.7

    def test_gov_domain(self):
        score = score_domain_authority(
            "https://data.gov/dataset", [], []
        )
        assert score == 0.95

    def test_edu_domain(self):
        score = score_domain_authority(
            "https://mit.edu/research", [], []
        )
        assert score == 0.8

    def test_unknown_domain(self):
        score = score_domain_authority(
            "https://random-blog.com", [], []
        )
        assert score == 0.4


# ── Fallback Chain ────────────────────────────────────────────────


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_fallback_tries_next_on_failure(self):
        provider = FallbackSearchProvider(chain=["duckduckgo", "serpapi"])
        # Make first provider fail, second succeed
        mock_result = SearchResult(
            title="Fallback", url="https://example.com", snippet="test"
        )
        provider._chain[0].search = AsyncMock(side_effect=RuntimeError("DDG down"))
        provider._chain[1].search = AsyncMock(return_value=[mock_result])

        results = await provider.search("test")
        assert len(results) == 1
        assert results[0].title == "Fallback"

    @pytest.mark.asyncio
    async def test_fallback_returns_empty_on_all_fail(self):
        provider = FallbackSearchProvider(chain=["duckduckgo"])
        provider._chain[0].search = AsyncMock(side_effect=RuntimeError("DDG down"))

        results = await provider.search("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_fallback_skips_empty_results(self):
        provider = FallbackSearchProvider(chain=["duckduckgo", "serpapi"])
        mock_result = SearchResult(
            title="From SerpAPI", url="https://example.com", snippet="test"
        )
        provider._chain[0].search = AsyncMock(return_value=[])
        provider._chain[1].search = AsyncMock(return_value=[mock_result])

        results = await provider.search("test")
        assert len(results) == 1
        assert results[0].title == "From SerpAPI"
