"""Integration tests for research connectors pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

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
        config.edgar_enabled = False
        config.arxiv_enabled = False
        config.openfda_enabled = False
        config.worldbank_enabled = False
        config.kalshi_prior_enabled = False
        config.metaculus_enabled = False
        config.wikipedia_pageviews_enabled = False
        config.google_trends_enabled = False
        config.reddit_sentiment_enabled = False
        config.pubmed_enabled = False
        config.manifold_enabled = False
        config.predictit_enabled = False
        config.sports_odds_enabled = False
        config.sports_stats_enabled = False
        config.spotify_charts_enabled = False
        config.kronos_enabled = False
        config.crypto_futures_enabled = False
        config.defillama_enabled = False
        config.acled_enabled = False
        config.github_activity_enabled = False
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
        config.edgar_enabled = False
        config.arxiv_enabled = False
        config.openfda_enabled = False
        config.worldbank_enabled = False
        config.kalshi_prior_enabled = False
        config.metaculus_enabled = False
        config.wikipedia_pageviews_enabled = False
        config.google_trends_enabled = False
        config.reddit_sentiment_enabled = False
        config.pubmed_enabled = False
        config.manifold_enabled = False
        config.predictit_enabled = False
        config.sports_odds_enabled = False
        config.sports_stats_enabled = False
        config.spotify_charts_enabled = False
        config.kronos_enabled = False
        config.crypto_futures_enabled = False
        config.defillama_enabled = False
        config.acled_enabled = False
        config.github_activity_enabled = False
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
        config.edgar_enabled = False
        config.arxiv_enabled = False
        config.openfda_enabled = False
        config.worldbank_enabled = False
        config.kalshi_prior_enabled = False
        config.metaculus_enabled = False
        config.wikipedia_pageviews_enabled = False
        config.google_trends_enabled = False
        config.reddit_sentiment_enabled = False
        config.pubmed_enabled = False
        config.manifold_enabled = False
        config.predictit_enabled = False
        config.sports_odds_enabled = False
        config.sports_stats_enabled = False
        config.spotify_charts_enabled = False
        config.kronos_enabled = False
        config.crypto_futures_enabled = False
        config.defillama_enabled = False
        config.acled_enabled = False
        config.github_activity_enabled = False
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
        config.edgar_enabled = True
        config.arxiv_enabled = True
        config.openfda_enabled = True
        config.worldbank_enabled = True
        config.kalshi_prior_enabled = True
        config.metaculus_enabled = True
        config.wikipedia_pageviews_enabled = True
        config.google_trends_enabled = True
        config.pubmed_enabled = True
        config.reddit_sentiment_enabled = True
        config.manifold_enabled = True
        config.predictit_enabled = True
        config.sports_odds_enabled = True
        config.sports_stats_enabled = True
        config.spotify_charts_enabled = True
        config.kronos_enabled = True
        config.crypto_futures_enabled = True
        config.defillama_enabled = True
        config.acled_enabled = True
        config.github_activity_enabled = True
        connectors = get_enabled_connectors(config)
        assert len(connectors) == 26

    def test_new_connectors_enabled_individually(self) -> None:
        for connector_flag, expected_name in [
            ("edgar_enabled", "edgar"),
            ("arxiv_enabled", "arxiv"),
            ("openfda_enabled", "openfda"),
            ("worldbank_enabled", "worldbank"),
            ("kalshi_prior_enabled", "kalshi_prior"),
            ("metaculus_enabled", "metaculus"),
            ("wikipedia_pageviews_enabled", "wikipedia_pageviews"),
            ("google_trends_enabled", "google_trends"),
            ("pubmed_enabled", "pubmed"),
            ("reddit_sentiment_enabled", "reddit_sentiment"),
            ("manifold_enabled", "manifold"),
            ("predictit_enabled", "predictit"),
            ("sports_odds_enabled", "sports_odds"),
            ("sports_stats_enabled", "sports_stats"),
            ("spotify_charts_enabled", "spotify_charts"),
        ]:
            config = MagicMock()
            config.openmeteo_enabled = False
            config.fred_enabled = False
            config.coingecko_enabled = False
            config.congress_enabled = False
            config.gdelt_enabled = False
            config.courtlistener_enabled = False
            config.edgar_enabled = False
            config.arxiv_enabled = False
            config.openfda_enabled = False
            config.worldbank_enabled = False
            config.kalshi_prior_enabled = False
            config.metaculus_enabled = False
            config.wikipedia_pageviews_enabled = False
            config.google_trends_enabled = False
            config.reddit_sentiment_enabled = False
            config.pubmed_enabled = False
            config.manifold_enabled = False
            config.predictit_enabled = False
            config.sports_odds_enabled = False
            config.sports_stats_enabled = False
            config.spotify_charts_enabled = False
            config.kronos_enabled = False
            config.crypto_futures_enabled = False
            config.defillama_enabled = False
            config.acled_enabled = False
            config.github_activity_enabled = False
            setattr(config, connector_flag, True)
            connectors = get_enabled_connectors(config)
            assert len(connectors) == 1
            assert connectors[0].name == expected_name


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
        config.edgar_enabled = True
        config.arxiv_enabled = True
        config.openfda_enabled = True
        config.worldbank_enabled = True
        config.kalshi_prior_enabled = True
        config.metaculus_enabled = True
        config.wikipedia_pageviews_enabled = True
        config.google_trends_enabled = True
        config.pubmed_enabled = True
        config.reddit_sentiment_enabled = True
        config.manifold_enabled = True
        config.predictit_enabled = True
        config.sports_odds_enabled = True
        config.sports_stats_enabled = True
        config.spotify_charts_enabled = True
        config.kronos_enabled = True
        config.crypto_futures_enabled = True
        config.defillama_enabled = True
        config.acled_enabled = True
        config.github_activity_enabled = True
        return get_enabled_connectors(config)

    def test_all_have_name(self) -> None:
        for c in self._get_all_connectors():
            assert isinstance(c.name, str)
            assert len(c.name) > 0

    def test_all_have_relevant_categories(self) -> None:
        for c in self._get_all_connectors():
            cats = c.relevant_categories()
            assert isinstance(cats, set)
            # Empty set means "all categories" (e.g., manifold)
            # Non-empty set means specific categories

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

    def test_fetch_structured_sources_calls_relevant_connectors(self) -> None:
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

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._connectors = [mock_connector]
        fetcher._config = MagicMock(source_timeout_secs=15)
        result = asyncio.run(
            fetcher.fetch_structured_sources("test question", "MACRO")
        )

        assert len(result) == 1
        assert result[0].extraction_method == "api"
        mock_connector.fetch.assert_awaited_once()

    def test_irrelevant_connectors_skipped(self) -> None:
        """Connectors that are not relevant are skipped."""
        from src.research.source_fetcher import SourceFetcher

        mock_connector = MagicMock(spec=BaseResearchConnector)
        mock_connector.is_relevant.return_value = False

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._connectors = [mock_connector]
        fetcher._config = MagicMock(source_timeout_secs=15)
        result = asyncio.run(
            fetcher.fetch_structured_sources("test question", "CRYPTO")
        )

        assert result == []
        mock_connector.fetch.assert_not_called()

    def test_connector_failure_doesnt_break_pipeline(self) -> None:
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

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._connectors = [mock_bad, mock_good]
        fetcher._config = MagicMock(source_timeout_secs=15)
        result = asyncio.run(
            fetcher.fetch_structured_sources("test", "MACRO")
        )

        # Good connector's source still comes through
        assert len(result) == 1
        assert result[0].url == "https://good.com"
