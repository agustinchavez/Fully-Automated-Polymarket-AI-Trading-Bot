"""Tests for WorldBankConnector — development indicators."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.worldbank import WorldBankConnector, _COUNTRY_MAP


# ── Fake responses ─────────────────────────────────────────────────


def _make_wb_response(country="China", indicator="GDP Growth (annual %)", code="NY.GDP.MKTP.KD.ZG"):
    """Fake World Bank API response."""
    return [
        {"page": 1, "pages": 1, "total": 3},
        [
            {"date": "2025", "value": 5.2, "indicator": {"value": indicator}},
            {"date": "2024", "value": 5.0, "indicator": {"value": indicator}},
            {"date": "2023", "value": 5.4, "indicator": {"value": indicator}},
        ],
    ]


# ── Country extraction ─────────────────────────────────────────────


class TestCountryExtraction:
    def test_extract_china(self) -> None:
        c = WorldBankConnector()
        result = c._extract_country("Will China GDP growth slow?")
        assert result is not None
        assert result[1] == "CN"

    def test_extract_uk_full_name(self) -> None:
        c = WorldBankConnector()
        result = c._extract_country("Will the United Kingdom economy recover?")
        assert result is not None
        assert result[1] == "GB"

    def test_extract_south_korea(self) -> None:
        c = WorldBankConnector()
        result = c._extract_country("Will South Korea exports grow?")
        assert result is not None
        assert result[1] == "KR"

    def test_no_country_returns_none(self) -> None:
        c = WorldBankConnector()
        result = c._extract_country("Will inflation rise in the US?")
        # US is not in the map (this connector is for non-US)
        assert result is None

    def test_country_map_has_major_economies(self) -> None:
        for country in ["china", "india", "brazil", "germany", "japan", "france"]:
            assert country in _COUNTRY_MAP


# ── Indicator matching ─────────────────────────────────────────────


class TestIndicatorMatching:
    def test_gdp_growth_match(self) -> None:
        c = WorldBankConnector()
        matches = c._match_indicators("Will GDP growth in China slow?")
        codes = [m[0] for m in matches]
        assert "NY.GDP.MKTP.KD.ZG" in codes

    def test_inflation_match(self) -> None:
        c = WorldBankConnector()
        matches = c._match_indicators("Will inflation rise in India?")
        codes = [m[0] for m in matches]
        assert "FP.CPI.TOTL.ZG" in codes

    def test_unemployment_match(self) -> None:
        c = WorldBankConnector()
        matches = c._match_indicators("Will unemployment fall in Brazil?")
        codes = [m[0] for m in matches]
        assert "SL.UEM.TOTL.ZS" in codes

    def test_default_to_gdp_when_no_keyword(self) -> None:
        c = WorldBankConnector()
        matches = c._match_indicators("How is the economy in Germany?")
        codes = [m[0] for m in matches]
        # "economy" doesn't exactly match, but default fallback is GDP
        assert len(matches) >= 1

    def test_max_indicators_limit(self) -> None:
        c = WorldBankConnector()
        matches = c._match_indicators(
            "GDP growth inflation unemployment exports", max_indicators=2
        )
        assert len(matches) <= 2


# ── Relevance ──────────────────────────────────────────────────────


class TestRelevance:
    def test_relevant_for_macro_with_country(self) -> None:
        c = WorldBankConnector()
        assert c.is_relevant("Will China GDP growth slow?", "MACRO")

    def test_relevant_for_geopolitics_with_country(self) -> None:
        c = WorldBankConnector()
        assert c.is_relevant("Will India trade exports grow?", "GEOPOLITICS")

    def test_not_relevant_without_country(self) -> None:
        c = WorldBankConnector()
        assert not c.is_relevant("Will inflation rise?", "MACRO")

    def test_not_relevant_for_sports(self) -> None:
        c = WorldBankConnector()
        assert not c.is_relevant("Will Japan win the World Cup?", "SPORTS")


# ── Fetch pipeline ─────────────────────────────────────────────────


class TestFetch:
    def test_successful_fetch(self) -> None:
        c = WorldBankConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_wb_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.worldbank.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will China GDP growth slow in 2026?", "MACRO")
            )

        assert len(sources) >= 1
        s = sources[0]
        assert s.publisher == "World Bank"
        assert s.extraction_method == "api"
        assert s.authority_score == 1.0
        assert "China" in s.title
        assert "5.20" in s.content or "5.2" in s.content

    def test_returns_empty_for_us_question(self) -> None:
        c = WorldBankConnector()
        sources = asyncio.run(
            c._fetch_impl("Will US GDP growth slow?", "MACRO")
        )
        assert sources == []

    def test_returns_empty_for_irrelevant_category(self) -> None:
        c = WorldBankConnector()
        sources = asyncio.run(
            c._fetch_impl("Will China win?", "SPORTS")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = WorldBankConnector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        with patch("src.research.connectors.worldbank.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c.fetch("Will India inflation rise?", "MACRO")
            )
        assert sources == []

    def test_empty_data_returns_none(self) -> None:
        c = WorldBankConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"page": 1}, None]
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.worldbank.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will Brazil GDP grow?", "MACRO")
            )
        assert sources == []

    def test_respects_mrv_config(self) -> None:
        config = MagicMock()
        config.worldbank_mrv = 5
        c = WorldBankConnector(config=config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_wb_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.worldbank.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            asyncio.run(
                c._fetch_impl("Will India GDP growth slow?", "MACRO")
            )

        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["mrv"] == 5
