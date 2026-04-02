"""Tests for EdgarConnector — SEC EDGAR filings and XBRL data."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.edgar import EdgarConnector, _CIK_MAP


# ── Fake responses ─────────────────────────────────────────────────


def _make_xbrl_response():
    """Fake XBRL companyfacts response."""
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"val": 94836000000, "end": "2026-03-29", "form": "10-Q"},
                            {"val": 119575000000, "end": "2025-12-28", "form": "10-K"},
                            {"val": 85777000000, "end": "2025-06-28", "form": "10-Q"},
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {"val": 23636000000, "end": "2026-03-29", "form": "10-Q"},
                        ]
                    }
                },
            }
        }
    }


def _make_efts_response(n=2):
    """Fake EFTS full-text search response."""
    hits = []
    for i in range(n):
        hits.append({
            "_source": {
                "entity_name": f"Test Corp {i}",
                "form_type": "8-K",
                "file_date": f"2026-03-0{i + 1}",
                "file_num": f"001-{40000 + i}",
            }
        })
    return {"hits": {"hits": hits}}


# ── Company extraction ─────────────────────────────────────────────


class TestCompanyExtraction:
    def test_extract_apple(self) -> None:
        c = EdgarConnector()
        name, cik = c._extract_company("Will Apple report record revenue?")
        assert cik == "0000320193"

    def test_extract_tesla_by_ticker(self) -> None:
        c = EdgarConnector()
        name, cik = c._extract_company("Will TSLA stock reach $300?")
        assert cik == "0001318605"

    def test_extract_nvidia(self) -> None:
        c = EdgarConnector()
        name, cik = c._extract_company("Will Nvidia beat earnings expectations?")
        assert cik == "0001045810"

    def test_no_company_returns_empty(self) -> None:
        c = EdgarConnector()
        name, cik = c._extract_company("Will inflation rise above 3%?")
        assert cik == ""
        assert name == ""

    def test_cik_map_has_major_companies(self) -> None:
        # Ensure the map covers key companies
        assert "apple" in _CIK_MAP
        assert "microsoft" in _CIK_MAP
        assert "google" in _CIK_MAP
        assert "amazon" in _CIK_MAP
        assert "meta" in _CIK_MAP
        assert "tesla" in _CIK_MAP
        assert "nvidia" in _CIK_MAP


# ── Numeric question detection ─────────────────────────────────────


class TestNumericDetection:
    def test_revenue_is_numeric(self) -> None:
        c = EdgarConnector()
        assert c._is_numeric_question("Will Apple revenue exceed $100B?")

    def test_earnings_is_numeric(self) -> None:
        c = EdgarConnector()
        assert c._is_numeric_question("Will Tesla report positive earnings?")

    def test_generic_not_numeric(self) -> None:
        c = EdgarConnector()
        assert not c._is_numeric_question("Will Apple announce a new product?")


# ── Relevance ──────────────────────────────────────────────────────


class TestRelevance:
    def test_relevant_for_corporate(self) -> None:
        c = EdgarConnector()
        assert c.is_relevant("any question", "CORPORATE")

    def test_relevant_for_tech(self) -> None:
        c = EdgarConnector()
        assert c.is_relevant("any question", "TECH")

    def test_relevant_by_keyword(self) -> None:
        c = EdgarConnector()
        assert c.is_relevant("Will earnings beat expectations?", "UNKNOWN")
        assert c.is_relevant("Will the CEO resign?", "UNKNOWN")
        assert c.is_relevant("Will the merger go through?", "UNKNOWN")

    def test_not_relevant_for_weather(self) -> None:
        c = EdgarConnector()
        assert not c.is_relevant("Will it rain tomorrow?", "WEATHER")


# ── XBRL fetch ─────────────────────────────────────────────────────


class TestXBRLFetch:
    def test_xbrl_revenue_fetch(self) -> None:
        c = EdgarConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_xbrl_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            source = asyncio.run(
                c._fetch_xbrl("0000320193", "Apple", "Will Apple revenue grow?")
            )

        assert source is not None
        assert source.extraction_method == "api"
        assert source.authority_score == 1.0
        assert source.publisher == "SEC EDGAR"
        assert "Revenue" in source.title
        assert "94,836,000,000" in source.content or "94836000000" in source.content

    def test_xbrl_metric_not_found(self) -> None:
        c = EdgarConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"facts": {"us-gaap": {}}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            source = asyncio.run(
                c._fetch_xbrl("0000320193", "Apple", "Will Apple revenue grow?")
            )

        assert source is None


# ── Filing search ──────────────────────────────────────────────────


class TestFilingSearch:
    def test_search_filings_by_cik(self) -> None:
        c = EdgarConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_efts_response(2)
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._search_filings("Apple", "0000320193", max_results=3)
            )

        assert len(sources) == 2
        assert sources[0].publisher == "SEC EDGAR"
        assert "8-K" in sources[0].title
        assert sources[0].extraction_method == "api"

    def test_efts_full_text_search(self) -> None:
        c = EdgarConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_efts_response(1)
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._search_efts("corporate layoffs tech sector", max_results=3)
            )

        assert len(sources) == 1
        assert sources[0].publisher == "SEC EDGAR"


# ── Full fetch pipeline ────────────────────────────────────────────


class TestFetchPipeline:
    def test_full_fetch_with_company(self) -> None:
        c = EdgarConnector()

        xbrl_resp = MagicMock()
        xbrl_resp.json.return_value = _make_xbrl_response()
        xbrl_resp.raise_for_status = MagicMock()

        efts_resp = MagicMock()
        efts_resp.json.return_value = _make_efts_response(1)
        efts_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        # First call = XBRL, second call = EFTS
        mock_client.get = AsyncMock(side_effect=[xbrl_resp, efts_resp])
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl(
                    "Will Apple revenue exceed expectations?", "CORPORATE"
                )
            )

        assert len(sources) >= 1
        # Should have XBRL source and/or filing source
        assert all(s.publisher == "SEC EDGAR" for s in sources)

    def test_full_fetch_no_company_uses_efts(self) -> None:
        c = EdgarConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_efts_response(2)
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will there be major SEC enforcement?", "CORPORATE")
            )

        assert len(sources) >= 1

    def test_returns_empty_for_irrelevant(self) -> None:
        c = EdgarConnector()
        sources = asyncio.run(
            c._fetch_impl("Will it snow in Miami?", "WEATHER")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = EdgarConnector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        with patch("src.research.connectors.edgar.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c.fetch("Will Apple stock crash?", "CORPORATE")
            )
        assert sources == []
