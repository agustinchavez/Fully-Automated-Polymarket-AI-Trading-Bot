"""Tests for BaseResearchConnector ABC and helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource


# ── Concrete subclass for testing ──────────────────────────────────


class _StubConnector(BaseResearchConnector):
    """Minimal concrete connector for testing the base class."""

    def __init__(self, *, fail: bool = False, config=None):
        super().__init__(config)
        self._fail = fail

    @property
    def name(self) -> str:
        return "stub"

    def relevant_categories(self) -> set[str]:
        return {"TEST_CAT"}

    async def _fetch_impl(self, question, market_type):
        if self._fail:
            raise RuntimeError("boom")
        return [
            self._make_source(
                title="Stub",
                url="https://example.com/data",
                snippet="stub snippet",
                publisher="StubAPI",
                content="full content here",
            )
        ]


class _KeywordConnector(BaseResearchConnector):
    """Connector with keyword-based relevance."""

    @property
    def name(self) -> str:
        return "keyword_stub"

    def relevant_categories(self) -> set[str]:
        return {"MACRO"}

    def is_relevant(self, question, market_type):
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, ["inflation", "gdp"])

    async def _fetch_impl(self, question, market_type):
        return []


# ── Tests ──────────────────────────────────────────────────────────


class TestMakeSource:
    def test_extraction_method_is_api(self) -> None:
        c = _StubConnector()
        src = c._make_source(
            title="T", url="https://x.com", snippet="s", publisher="P",
        )
        assert src.extraction_method == "api"

    def test_authority_score_default_is_1(self) -> None:
        c = _StubConnector()
        src = c._make_source(
            title="T", url="https://x.com", snippet="s", publisher="P",
        )
        assert src.authority_score == 1.0

    def test_custom_authority_score(self) -> None:
        c = _StubConnector()
        src = c._make_source(
            title="T", url="https://x.com", snippet="s", publisher="P",
            authority_score=0.4,
        )
        assert src.authority_score == 0.4

    def test_content_length_set(self) -> None:
        c = _StubConnector()
        src = c._make_source(
            title="T", url="https://x.com", snippet="s", publisher="P",
            content="hello world",
        )
        assert src.content_length == len("hello world")

    def test_returns_fetched_source(self) -> None:
        c = _StubConnector()
        src = c._make_source(
            title="T", url="https://x.com", snippet="s", publisher="P",
        )
        assert isinstance(src, FetchedSource)


class TestErrorSwallowing:
    def test_failing_connector_returns_empty(self) -> None:
        c = _StubConnector(fail=True)
        result = asyncio.run(
            c.fetch("some question", "TEST_CAT")
        )
        assert result == []

    def test_success_returns_sources(self) -> None:
        c = _StubConnector(fail=False)
        result = asyncio.run(
            c.fetch("some question", "TEST_CAT")
        )
        assert len(result) == 1
        assert result[0].extraction_method == "api"


class TestRelevantCategories:
    def test_relevant_categories_contract(self) -> None:
        c = _StubConnector()
        cats = c.relevant_categories()
        assert isinstance(cats, set)
        assert "TEST_CAT" in cats

    def test_is_relevant_default(self) -> None:
        c = _StubConnector()
        assert c.is_relevant("anything", "TEST_CAT")
        assert not c.is_relevant("anything", "OTHER")


class TestKeywordMatching:
    def test_keyword_match(self) -> None:
        c = _KeywordConnector()
        assert c.is_relevant("Will inflation rise?", "UNKNOWN")
        assert c.is_relevant("GDP growth forecast", "UNKNOWN")

    def test_keyword_no_match(self) -> None:
        c = _KeywordConnector()
        assert not c.is_relevant("Will BTC hit 100k?", "UNKNOWN")

    def test_category_match_overrides_keywords(self) -> None:
        c = _KeywordConnector()
        assert c.is_relevant("random question", "MACRO")


class TestClose:
    def test_close_with_no_client(self) -> None:
        c = _StubConnector()
        asyncio.run(c.close())
        assert c._client is None

    def test_close_shuts_client(self) -> None:
        c = _StubConnector()
        mock_client = AsyncMock()
        c._client = mock_client
        asyncio.run(c.close())
        mock_client.aclose.assert_awaited_once()
        assert c._client is None
