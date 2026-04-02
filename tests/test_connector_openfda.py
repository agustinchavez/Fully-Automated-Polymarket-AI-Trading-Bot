"""Tests for OpenFDAConnector — FDA drug approval data."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.openfda import OpenFDAConnector


# ── Fake responses ─────────────────────────────────────────────────


def _make_fda_response():
    """Fake openFDA drug response."""
    return {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Keytruda"],
                    "generic_name": ["pembrolizumab"],
                },
                "sponsor_name": "MERCK SHARP DOHME",
                "application_number": "BLA125514",
                "submissions": [
                    {
                        "submission_status_date": "20260115",
                        "submission_type": "SUPPL",
                    }
                ],
                "products": [
                    {
                        "route": "INTRAVENOUS",
                        "dosage_form": "INJECTION",
                    }
                ],
            },
            {
                "openfda": {
                    "brand_name": ["Opdivo"],
                    "generic_name": ["nivolumab"],
                },
                "sponsor_name": "BRISTOL MYERS SQUIBB",
                "application_number": "BLA125554",
                "submissions": [
                    {
                        "submission_status_date": "20250801",
                        "submission_type": "ORIG",
                    }
                ],
                "products": [],
            },
        ]
    }


# ── Drug term extraction ──────────────────────────────────────────


class TestDrugTermExtraction:
    def test_extract_quoted_term(self) -> None:
        c = OpenFDAConnector()
        term = c._extract_drug_term('Will "Keytruda" get FDA approval?')
        assert term == "Keytruda"

    def test_extract_proper_noun(self) -> None:
        c = OpenFDAConnector()
        term = c._extract_drug_term("Will Moderna vaccine get approved?")
        assert "Moderna" in term

    def test_extract_fallback(self) -> None:
        c = OpenFDAConnector()
        term = c._extract_drug_term("will the new cancer drug get approved by fda?")
        assert len(term) > 0  # Should extract some terms

    def test_empty_returns_empty(self) -> None:
        c = OpenFDAConnector()
        term = c._extract_drug_term("will the fda?")
        assert term == ""


# ── Relevance ──────────────────────────────────────────────────────


class TestRelevance:
    def test_relevant_for_fda_keyword(self) -> None:
        c = OpenFDAConnector()
        assert c.is_relevant("Will the FDA approve this drug?", "SCIENCE")
        assert c.is_relevant("Phase 3 clinical trial results", "SCIENCE")

    def test_relevant_for_drug_keyword(self) -> None:
        c = OpenFDAConnector()
        assert c.is_relevant("Will the new vaccine work?", "UNKNOWN")

    def test_not_relevant_for_generic_science(self) -> None:
        c = OpenFDAConnector()
        # No FDA keywords present
        assert not c.is_relevant("Will quantum computing advance?", "SCIENCE")

    def test_not_relevant_for_sports(self) -> None:
        c = OpenFDAConnector()
        assert not c.is_relevant("Will the Lakers win?", "SPORTS")


# ── Result parsing ─────────────────────────────────────────────────


class TestParsing:
    def test_parse_drug_result(self) -> None:
        c = OpenFDAConnector()
        result = _make_fda_response()["results"][0]
        source = c._parse_drug_result(result)
        assert source is not None
        assert "Keytruda" in source.title
        assert source.publisher == "openFDA"
        assert source.extraction_method == "api"
        assert source.authority_score == 1.0
        assert "pembrolizumab" in source.content
        assert "MERCK" in source.content
        assert "BLA125514" in source.content

    def test_parse_result_with_approval_date(self) -> None:
        c = OpenFDAConnector()
        result = _make_fda_response()["results"][0]
        source = c._parse_drug_result(result)
        assert source is not None
        assert "20260115" in source.content


# ── Fetch pipeline ─────────────────────────────────────────────────


class TestFetch:
    def test_successful_fetch(self) -> None:
        c = OpenFDAConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fda_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.openfda.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl(
                    "Will Keytruda get FDA approval for new indication?",
                    "SCIENCE",
                )
            )

        assert len(sources) >= 1
        assert all(s.publisher == "openFDA" for s in sources)

    def test_uses_api_key_from_config(self) -> None:
        config = MagicMock()
        config.openfda_api_key = "test-key-123"
        c = OpenFDAConnector(config=config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fda_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.openfda.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            asyncio.run(
                c._fetch_impl(
                    "Will Pfizer drug get FDA approval?", "SCIENCE",
                )
            )

        # Check that api_key was passed
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["api_key"] == "test-key-123"

    def test_returns_empty_for_irrelevant(self) -> None:
        c = OpenFDAConnector()
        sources = asyncio.run(
            c._fetch_impl("Will quantum computing advance?", "SCIENCE")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = OpenFDAConnector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        with patch("src.research.connectors.openfda.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c.fetch("Will the FDA approve this drug?", "SCIENCE")
            )
        assert sources == []

    def test_empty_results(self) -> None:
        c = OpenFDAConnector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.openfda.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will Keytruda get approved?", "SCIENCE")
            )
        assert sources == []
