"""Tests for WikipediaPageviewsConnector — attention spike detection."""

from __future__ import annotations

import asyncio
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.wikipedia_pageviews import WikipediaPageviewsConnector
from src.research.source_fetcher import FetchedSource


def _make_pageviews_response(
    avg_daily: int = 5000,
    spike_views: int = 15000,
) -> dict:
    items = []
    # 28 days of baseline
    for _ in range(28):
        items.append({"views": avg_daily})
    # 7 days of spike
    for _ in range(7):
        items.append({"views": spike_views})
    return {"items": items}


class TestEntityExtraction:
    def test_fed_maps_to_federal_reserve(self) -> None:
        c = WikipediaPageviewsConnector()
        assert c._extract_article("Will the Fed cut rates?") == "Federal_Reserve"

    def test_bitcoin_maps_correctly(self) -> None:
        c = WikipediaPageviewsConnector()
        assert c._extract_article("Will bitcoin hit 100k?") == "Bitcoin"

    def test_btc_maps_to_bitcoin(self) -> None:
        c = WikipediaPageviewsConnector()
        assert c._extract_article("Will BTC reach new highs?") == "Bitcoin"

    def test_trump_maps_correctly(self) -> None:
        c = WikipediaPageviewsConnector()
        assert c._extract_article("Will Trump win the election?") == "Donald_Trump"

    def test_unknown_entity_single_word_skipped(self) -> None:
        c = WikipediaPageviewsConnector()
        # Single proper noun isn't enough — requires >= 2 to avoid
        # garbage articles like "Heat_Hornets" from sports matchups
        result = c._extract_article("Will Zylothon succeed?")
        assert result == ""

    def test_unknown_entity_multi_word_fallback(self) -> None:
        c = WikipediaPageviewsConnector()
        # Two qualifying proper nouns produce a valid fallback
        result = c._extract_article("Will Zylothon Corporation expand?")
        assert result == "Zylothon_Corporation"

    def test_empty_question_returns_empty(self) -> None:
        c = WikipediaPageviewsConnector()
        # All lowercase stop words, no capitalized nouns, no entity match
        result = c._extract_article("will the at to for")
        assert result == ""


class TestSpikeComputation:
    def test_3x_spike_detected(self) -> None:
        c = WikipediaPageviewsConnector()
        # 28 days of 1000 views, 7 days of 3000 views
        views = [1000] * 28 + [3000] * 7
        spike = c._compute_spike(views)
        assert spike["spike_ratio"] > 2.0
        assert spike["avg_7d"] == 3000

    def test_no_spike_ratio_near_one(self) -> None:
        c = WikipediaPageviewsConnector()
        # Flat views — no spike
        views = [1000] * 35
        spike = c._compute_spike(views)
        assert spike["spike_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_less_than_7_items_returns_default(self) -> None:
        c = WikipediaPageviewsConnector()
        spike = c._compute_spike([100, 200, 300])
        assert spike["spike_ratio"] == 1.0
        assert spike["avg_7d"] == 0
        assert spike["avg_30d"] == 0

    def test_empty_list(self) -> None:
        c = WikipediaPageviewsConnector()
        spike = c._compute_spike([])
        assert spike["spike_ratio"] == 1.0

    def test_exactly_7_items(self) -> None:
        c = WikipediaPageviewsConnector()
        views = [500] * 7
        spike = c._compute_spike(views)
        assert spike["avg_7d"] == 500


class TestDirectionLabels:
    def test_below_1_5_is_normal(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 1.2, "avg_7d": 1200, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert len(sources) == 1
        assert sources[0].raw["behavioral_signal"]["direction"] == "normal"

    def test_1_5_to_2_0_is_elevated(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 1.7, "avg_7d": 1700, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "elevated"

    def test_2_0_to_3_0_is_strong(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 2.5, "avg_7d": 2500, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "strong"

    def test_above_3_0_is_viral(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 4.0, "avg_7d": 4000, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "viral"

    def test_exactly_1_5_is_elevated(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 1.5, "avg_7d": 1500, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "elevated"

    def test_exactly_2_0_is_strong(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 2.0, "avg_7d": 2000, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "strong"

    def test_exactly_3_0_is_viral(self) -> None:
        c = WikipediaPageviewsConnector()
        spike_data = {"spike_ratio": 3.0, "avg_7d": 3000, "avg_30d": 1000}
        sources = c._build_source("Test_Article", spike_data)
        assert sources[0].raw["behavioral_signal"]["direction"] == "viral"


class TestFetch:
    def test_successful_fetch_with_spike(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_pageviews_response(
            avg_daily=5000, spike_views=15000
        )
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will bitcoin price rise?", "CRYPTO")
            )

        assert len(sources) == 1
        src = sources[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert "behavioral_signal" in src.raw
        signal = src.raw["behavioral_signal"]
        assert signal["source"] == "wikipedia"
        assert signal["signal_type"] == "attention_spike"
        assert signal["value"] > 1.0

    def test_no_article_extracted_returns_empty(self) -> None:
        c = WikipediaPageviewsConnector()
        # All lowercase stop words with no entity match and no capitalized nouns
        sources = asyncio.run(
            c._fetch_impl("will the at to for", "UNKNOWN")
        )
        assert sources == []

    def test_api_error_returns_empty_via_fetch(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("Wikimedia down"))
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            # Use .fetch() to go through base class circuit breaker
            sources = asyncio.run(
                c.fetch("Will Trump win?", "ELECTION")
            )

        assert sources == []

    def test_empty_items_returns_empty(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will Trump win?", "ELECTION")
            )

        assert sources == []

    def test_publisher_is_wikipedia(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_pageviews_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will bitcoin rally?", "CRYPTO")
            )

        assert len(sources) == 1
        assert sources[0].publisher == "Wikipedia/Wikimedia"

    def test_authority_score(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_pageviews_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will bitcoin rally?", "CRYPTO")
            )

        assert len(sources) == 1
        assert sources[0].authority_score == 0.7


class TestCache:
    def test_cache_hit_avoids_second_http_call(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_pageviews_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()

            # First call — should hit API
            sources1 = asyncio.run(
                c._fetch_impl("Will bitcoin rally?", "CRYPTO")
            )
            assert len(sources1) == 1

            # Second call — should use cache (no additional HTTP call)
            sources2 = asyncio.run(
                c._fetch_impl("Will bitcoin rally?", "CRYPTO")
            )
            assert len(sources2) == 1

        # mock_client.get should have been called only once
        assert mock_client.get.call_count == 1

    def test_cache_expires_after_ttl(self) -> None:
        c = WikipediaPageviewsConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_pageviews_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.wikipedia_pageviews.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()

            # First call populates cache
            asyncio.run(c._fetch_impl("Will bitcoin rally?", "CRYPTO"))

            # Manually expire cache by setting timestamp far in the past
            for key in c._cache:
                ts, data = c._cache[key]
                c._cache[key] = (ts - 20000, data)

            # Second call should hit API again
            asyncio.run(c._fetch_impl("Will bitcoin rally?", "CRYPTO"))

        assert mock_client.get.call_count == 2
