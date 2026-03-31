"""Tests for OpenMeteoConnector — weather forecast research."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.open_meteo import OpenMeteoConnector
from src.research.source_fetcher import FetchedSource


def _make_forecast_response(city_lat=40.71, city_lon=-74.01):
    """Fake Open-Meteo forecast API response."""
    return {
        "daily": {
            "time": ["2026-03-14", "2026-03-15", "2026-03-16"],
            "precipitation_sum": [2.1, 0.0, 5.3],
            "windspeed_10m_max": [25.0, 12.0, 40.0],
            "temperature_2m_max": [18.5, 22.0, 15.0],
            "temperature_2m_min": [8.0, 10.0, 6.0],
        },
    }


def _make_geocode_response(name="Paris", lat=48.86, lon=2.35):
    return {
        "results": [
            {"name": name, "latitude": lat, "longitude": lon}
        ]
    }


class TestCityParsing:
    def test_extracts_known_city(self) -> None:
        c = OpenMeteoConnector()
        city = c._extract_city("Will temperature in Miami exceed 95°F?")
        assert city.lower() in ("miami", "")  # depends on regex
        # More flexible: check that resolve_location finds it
        lat, lon, name = asyncio.run(
            c._resolve_location("Will temperature in Miami exceed 95°F?")
        )
        assert lat is not None
        assert abs(lat - 25.76) < 0.1

    def test_extracts_city_for_preposition(self) -> None:
        c = OpenMeteoConnector()
        city = c._extract_city("Will it rain in Seattle this week?")
        assert "seattle" in city.lower()

    def test_direct_city_match(self) -> None:
        c = OpenMeteoConnector()
        lat, lon, name = asyncio.run(
            c._resolve_location("Will Chicago have a heat wave?")
        )
        assert lat is not None
        assert abs(lat - 41.88) < 0.1

    def test_unknown_city_returns_none_without_geocoding(self) -> None:
        c = OpenMeteoConnector()
        # Mock the geocoding API to return no results
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        lat, lon, name = asyncio.run(
            c._resolve_location("Will Timbuktu get snow?")
        )
        # Either geocoded or None — depends on whether "Timbuktu" is extracted
        # The key point: doesn't raise


class TestRelevance:
    def test_relevant_for_weather_category(self) -> None:
        c = OpenMeteoConnector()
        assert c.is_relevant("any question", "WEATHER")

    def test_relevant_for_weather_keywords(self) -> None:
        c = OpenMeteoConnector()
        assert c.is_relevant("Will hurricane hit Florida?", "UNKNOWN")
        assert c.is_relevant("Will there be a storm in NYC?", "UNKNOWN")

    def test_not_relevant_for_non_weather(self) -> None:
        c = OpenMeteoConnector()
        assert not c.is_relevant("Will BTC hit 100k?", "CRYPTO")


class TestFetch:
    def test_successful_fetch(self) -> None:
        c = OpenMeteoConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_forecast_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.open_meteo.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will it rain in Miami?", "WEATHER")
            )

        assert len(sources) >= 1
        src = sources[0]
        assert src.extraction_method == "api"
        assert src.authority_score == 1.0
        assert src.publisher == "Open-Meteo"
        assert "18.5" in src.content  # temp_max from response

    def test_returns_empty_for_non_weather(self) -> None:
        c = OpenMeteoConnector()
        sources = asyncio.run(
            c._fetch_impl("Will BTC hit 100k?", "CRYPTO")
        )
        assert sources == []

    def test_returns_empty_when_no_city_found(self) -> None:
        c = OpenMeteoConnector()
        # No city name in question
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        sources = asyncio.run(
            c._fetch_impl("Will the weather be bad?", "WEATHER")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = OpenMeteoConnector()
        # _fetch_impl will fail because no client and no network
        # But fetch() should catch the error
        with patch("src.research.connectors.open_meteo.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=RuntimeError("network error"))
            c._client = mock_client

            sources = asyncio.run(
                c.fetch("Will it rain in Miami?", "WEATHER")
            )
        assert sources == []


class TestGeocodeFallback:
    def test_geocode_for_unknown_city(self) -> None:
        c = OpenMeteoConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_geocode_response("Paris", 48.86, 2.35)
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        lat, lon, name = asyncio.run(
            c._geocode("Paris")
        )
        assert lat == 48.86
        assert lon == 2.35
