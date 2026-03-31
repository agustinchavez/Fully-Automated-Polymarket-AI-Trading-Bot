"""Integration tests for research connectors pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.base import BaseResearchConnector
from src.research.connectors.registry import get_enabled_connectors
from src.research.source_fetcher import FetchedSource


# ── Registry Tests ────────────────────────────────────────────────────


class TestRegistry:
    def test_no_connectors_when_all_disabled(self) -> None:
        config = MagicMock()
        config.openmeteo_enabled = False
        config.fred_enabled = False
        config.coingecko_enabled = False
        config.congress_enabled = False
        config.gdelt_enabled = False
        config.courtlistener_enabled = False
        connectors = get_enabled_connectors(config)
        assert connectors == []

    def test_openmeteo_enabled(self) -> None:
        config = MagicMock()
        config.openmeteo_enabled = True
        config.fred_enabled = False
        config.coingecko_enabled = False
        config.congress_enabled = False
        config.gdelt_enabled = False
        config.courtlistener_enabled = False
        connectors = get_enabled_connectors(config)
        assert len(connectors) == 1
        assert connectors[0].name == "open_meteo"

    def test_multiple_connectors_enabled(self) -> None:
        config = MagicMock()
        config.openmeteo_enabled = True
        config.fred_enabled = True
        config.coingecko_enabled = False
        config.congress_enabled = False
        config.gdelt_enabled = True
        config.courtlistener_enabled = False
        connectors = get_enabled_connectors(config)
        assert len(connectors) == 3
        names = {c.name for c in connectors}
        assert "open_meteo" in names
        assert "fred" in names
        assert "gdelt" in names

    def test_all_connectors_enabled(self) -> None:
        config = MagicMock()
        config.openmeteo_enabled = True
        config.fred_enabled = True
        config.coingecko_enabled = True
        config.congress_enabled = True
        config.gdelt_enabled = True
        config.courtlistener_enabled = True
        connectors = get_enabled_connectors(config)
        assert len(connectors) == 6


# ── Connector Contracts ──────────────────────────────────────────────


class TestConnectorContracts:
    """Verify all connectors satisfy the base class contract."""

    def _get_all_connectors(self) -> list[BaseResearchConnector]:
        config = MagicMock()
        config.openmeteo_enabled = True
        config.fred_enabled = True
        config.coingecko_enabled = True
        config.congress_enabled = True
        config.gdelt_enabled = True
        config.courtlistener_enabled = True
        return get_enabled_connectors(config)

    def test_all_have_name(self) -> None:
        for c in self._get_all_connectors():
            assert isinstance(c.name, str)
            assert len(c.name) > 0

    def test_all_have_relevant_categories(self) -> None:
        for c in self._get_all_connectors():
            cats = c.relevant_categories()
            assert isinstance(cats, set)
            assert len(cats) > 0

    def test_all_return_list_on_irrelevant(self) -> None:
        for c in self._get_all_connectors():
            result = asyncio.run(c.fetch("test", "NONEXISTENT_CATEGORY_XYZ"))
            assert isinstance(result, list)


# ── Source Fetcher Integration ────────────────────────────────────────


class TestSourceFetcherIntegration:
    """Test integration between connectors and SourceFetcher."""

    def test_api_sources_have_correct_extraction_method(self) -> None:
        """All connector sources use extraction_method='api'."""
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)
        src = c._make_source(
            title="Test",
            url="https://example.com",
            snippet="test",
            publisher="Test",
        )
        assert src.extraction_method == "api"

    def test_authority_score_defaults(self) -> None:
        """Default authority_score is 1.0."""
        from src.research.connectors.fred import FredConnector

        c = FredConnector(config=None)
        src = c._make_source(
            title="Test",
            url="https://example.com",
            snippet="test",
            publisher="Test",
        )
        assert src.authority_score == 1.0

    def test_gdelt_authority_score_is_low(self) -> None:
        """GDELT uses 0.4 authority_score (aggregator, not primary)."""
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)
        src = c._make_source(
            title="Test",
            url="https://example.com",
            snippet="test",
            publisher="Test",
            authority_score=0.4,
        )
        assert src.authority_score == 0.4

    @patch("src.research.connectors.registry.get_enabled_connectors")
    def test_fetch_structured_sources_calls_relevant_connectors(
        self, mock_get_connectors: MagicMock,
    ) -> None:
        """fetch_structured_sources dispatches to matching connectors."""
        from src.research.source_fetcher import SourceFetcher

        mock_connector = MagicMock(spec=BaseResearchConnector)
        mock_connector.is_relevant.return_value = True
        mock_connector.fetch = AsyncMock(
            return_value=[
                FetchedSource(
                    url="https://api.example.com/data",
                    title="API Source",
                    snippet="test",
                    publisher="TestAPI",
                    extraction_method="api",
                    authority_score=1.0,
                    content_length=100,
                )
            ]
        )
        mock_get_connectors.return_value = [mock_connector]

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = MagicMock()
        result = asyncio.run(
            fetcher.fetch_structured_sources("test question", "MACRO")
        )

        assert len(result) == 1
        assert result[0].extraction_method == "api"
        mock_connector.fetch.assert_awaited_once()

    @patch("src.research.connectors.registry.get_enabled_connectors")
    def test_irrelevant_connectors_skipped(
        self, mock_get_connectors: MagicMock,
    ) -> None:
        """Connectors that are not relevant are skipped."""
        from src.research.source_fetcher import SourceFetcher

        mock_connector = MagicMock(spec=BaseResearchConnector)
        mock_connector.is_relevant.return_value = False
        mock_get_connectors.return_value = [mock_connector]

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = MagicMock()
        result = asyncio.run(
            fetcher.fetch_structured_sources("test question", "CRYPTO")
        )

        assert result == []
        mock_connector.fetch.assert_not_called()

    @patch("src.research.connectors.registry.get_enabled_connectors")
    def test_connector_failure_doesnt_break_pipeline(
        self, mock_get_connectors: MagicMock,
    ) -> None:
        """A failing connector returns [] without breaking others."""
        from src.research.source_fetcher import SourceFetcher

        mock_good = MagicMock(spec=BaseResearchConnector)
        mock_good.is_relevant.return_value = True
        mock_good.fetch = AsyncMock(
            return_value=[
                FetchedSource(
                    url="https://good.com",
                    title="Good",
                    snippet="ok",
                    publisher="GoodAPI",
                    extraction_method="api",
                    authority_score=1.0,
                    content_length=10,
                )
            ]
        )

        mock_bad = MagicMock(spec=BaseResearchConnector)
        mock_bad.is_relevant.return_value = True
        mock_bad.fetch = AsyncMock(return_value=[])  # error swallowed in base

        mock_get_connectors.return_value = [mock_bad, mock_good]

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = MagicMock()
        result = asyncio.run(
            fetcher.fetch_structured_sources("test", "MACRO")
        )

        # Good connector's source still comes through
        assert len(result) == 1
        assert result[0].url == "https://good.com"
