"""Tests for FredConnector — Federal Reserve Economic Data research."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.fred import FredConnector
from src.research.source_fetcher import FetchedSource


def _make_fred_response(series_id="UNRATE"):
    """Fake FRED API response."""
    return {
        "observations": [
            {"date": "2026-02-01", "value": "3.9"},
            {"date": "2026-01-01", "value": "4.0"},
            {"date": "2025-12-01", "value": "4.1"},
        ]
    }


class TestSeriesMapping:
    def test_unemployment_maps_to_unrate(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will unemployment fall below 4%?", max_series=3)
        series_ids = [m[0] for m in matches]
        assert "UNRATE" in series_ids

    def test_inflation_maps_to_cpi(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will inflation exceed 3%?", max_series=3)
        series_ids = [m[0] for m in matches]
        assert "CPIAUCSL" in series_ids

    def test_fed_rate_maps_to_fedfunds(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will the Fed cut rates?", max_series=3)
        series_ids = [m[0] for m in matches]
        assert "FEDFUNDS" in series_ids

    def test_gdp_maps_correctly(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will GDP growth slow in Q2?", max_series=3)
        series_ids = [m[0] for m in matches]
        assert "GDP" in series_ids

    def test_treasury_maps_to_yields(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will the yield curve invert?", max_series=3)
        series_ids = [m[0] for m in matches]
        assert "T10Y2Y" in series_ids or "DGS10" in series_ids

    def test_no_match_returns_empty(self) -> None:
        c = FredConnector()
        matches = c._match_series("Will BTC hit 100k?", max_series=3)
        assert matches == []

    def test_max_series_limit(self) -> None:
        c = FredConnector()
        # "economy recession GDP" should match multiple series
        matches = c._match_series("economy recession GDP treasury yield", max_series=2)
        assert len(matches) <= 2


class TestRelevance:
    def test_relevant_for_macro_category(self) -> None:
        c = FredConnector()
        assert c.is_relevant("any question", "MACRO")

    def test_relevant_for_macro_keywords(self) -> None:
        c = FredConnector()
        assert c.is_relevant("Will inflation rise?", "UNKNOWN")
        assert c.is_relevant("Will the Fed cut rates?", "UNKNOWN")

    def test_not_relevant_for_crypto(self) -> None:
        c = FredConnector()
        assert not c.is_relevant("Will BTC hit 100k?", "CRYPTO")


class TestFetch:
    def test_returns_empty_without_api_key(self) -> None:
        c = FredConnector()
        with patch.dict(os.environ, {}, clear=True):
            sources = asyncio.run(
                c._fetch_impl("Will unemployment fall below 4%?", "MACRO")
            )
        assert sources == []

    def test_successful_fetch(self) -> None:
        c = FredConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fred_response("UNRATE")
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("src.research.connectors.fred.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will unemployment fall below 4%?", "MACRO")
                )

        assert len(sources) >= 1
        src = sources[0]
        assert src.extraction_method == "api"
        assert src.authority_score == 1.0
        assert src.publisher == "Federal Reserve Bank of St. Louis"
        assert "3.9" in src.content
        assert "UNRATE" in src.title

    def test_returns_empty_for_non_macro(self) -> None:
        c = FredConnector()
        sources = asyncio.run(
            c._fetch_impl("Will BTC hit 100k?", "CRYPTO")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = FredConnector()
        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("src.research.connectors.fred.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
                c._client = mock_client

                sources = asyncio.run(
                    c.fetch("Will unemployment fall?", "MACRO")
                )
        assert sources == []

    def test_multiple_series_from_one_question(self) -> None:
        c = FredConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fred_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch.dict(os.environ, {"FRED_API_KEY": "test-key"}):
            with patch("src.research.connectors.fred.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl(
                        "Will treasury yield and mortgage rates rise?", "MACRO"
                    )
                )

        # Should have multiple sources (treasury + mortgage)
        assert len(sources) >= 2
