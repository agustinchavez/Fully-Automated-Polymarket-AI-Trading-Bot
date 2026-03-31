"""Tests for CongressConnector — legislative data."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.congress import CongressConnector
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector(*, api_key: str = "test-key") -> CongressConnector:
    c = CongressConnector(config=None)
    c._get_api_key = lambda: api_key  # type: ignore[assignment]
    return c


def _bills_response() -> dict:
    return {
        "bills": [
            {
                "title": "Inflation Reduction Act of 2022",
                "type": "HR",
                "number": "5376",
                "congress": "117",
                "latestAction": {
                    "text": "Became Public Law No: 117-169.",
                    "actionDate": "2022-08-16",
                },
                "url": "https://api.congress.gov/v3/bill/117/hr/5376",
            },
        ]
    }


def _nominations_response() -> dict:
    return {
        "nominations": [
            {
                "description": "John Doe, of California, to be Ambassador",
                "congress": "118",
                "latestAction": {
                    "text": "Confirmed by the Senate by Voice Vote.",
                    "actionDate": "2024-03-15",
                },
                "url": "https://api.congress.gov/v3/nomination/118/123",
            },
        ]
    }


# ── Relevance ─────────────────────────────────────────────────────────


class TestRelevance:
    def test_election_category_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any question", "ELECTION")

    def test_legal_category_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any question", "LEGAL")

    def test_non_matching_category_not_relevant(self) -> None:
        c = _make_connector()
        assert not c.is_relevant("Will Bitcoin hit 100k?", "CRYPTO")

    def test_keyword_makes_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("Will the Senate pass the bill?", "UNKNOWN")
        assert c.is_relevant("Will the House vote on legislation?", "UNKNOWN")
        assert c.is_relevant("Will the nomination be confirmed?", "UNKNOWN")


# ── Search Term Extraction ────────────────────────────────────────────


class TestSearchTermExtraction:
    def test_strips_question_framing(self) -> None:
        c = _make_connector()
        term = c._extract_search_term("Will the infrastructure bill pass?")
        assert "Will" not in term
        assert "infrastructure bill" in term.lower()

    def test_strips_date_framing(self) -> None:
        c = _make_connector()
        term = c._extract_search_term(
            "Will climate legislation pass by 2025?"
        )
        assert "2025" not in term
        assert "climate legislation" in term.lower()

    def test_strips_outcome_framing(self) -> None:
        c = _make_connector()
        term = c._extract_search_term(
            "Will the CHIPS Act become law?"
        )
        assert "become law" not in term.lower()

    def test_caps_length(self) -> None:
        c = _make_connector()
        long_q = "Will " + "a" * 200 + " pass?"
        term = c._extract_search_term(long_q)
        assert len(term) <= 100


# ── Fetch ─────────────────────────────────────────────────────────────


class TestFetch:
    def test_returns_empty_when_not_relevant(self) -> None:
        c = _make_connector()
        result = asyncio.run(c.fetch("Will Bitcoin go up?", "CRYPTO"))
        assert result == []

    def test_returns_empty_when_no_api_key(self) -> None:
        c = _make_connector(api_key="")
        result = asyncio.run(c.fetch("Will the Senate pass the bill?", "ELECTION"))
        assert result == []

    @patch("src.research.connectors.congress.rate_limiter")
    def test_bill_search(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _bills_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will the Inflation Reduction Act pass?", "ELECTION")
        )

        assert len(result) == 1
        src = result[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert src.authority_score == 1.0
        assert "Inflation Reduction Act" in src.content
        assert "HR" in src.content
        assert "5376" in src.content
        assert "Became Public Law" in src.content
        assert src.publisher == "Congress.gov"

    @patch("src.research.connectors.congress.rate_limiter")
    def test_nomination_search(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _nominations_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will the nomination be confirmed?", "ELECTION")
        )

        assert len(result) == 1
        src = result[0]
        assert "Nomination" in src.title
        assert "John Doe" in src.content
        assert "Confirmed" in src.content

    @patch("src.research.connectors.congress.rate_limiter")
    def test_api_error_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will the Senate pass the bill?", "ELECTION")
        )
        assert result == []

    @patch("src.research.connectors.congress.rate_limiter")
    def test_empty_bills_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"bills": []}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will climate legislation pass?", "ELECTION")
        )
        assert result == []


# ── Format ────────────────────────────────────────────────────────────


class TestFormat:
    def test_format_bill(self) -> None:
        c = _make_connector()
        bill = _bills_response()["bills"][0]
        src = c._format_bill(bill)
        assert src is not None
        assert "HR" in src.content
        assert "5376" in src.content
        assert "117th Congress" in src.content
        assert "congress.gov" in src.url.lower()

    def test_format_bill_no_title(self) -> None:
        c = _make_connector()
        src = c._format_bill({"number": "123"})
        assert src is None

    def test_format_nomination(self) -> None:
        c = _make_connector()
        nom = _nominations_response()["nominations"][0]
        src = c._format_nomination(nom)
        assert src is not None
        assert "John Doe" in src.content
        assert "Congress.gov" in src.publisher

    def test_format_nomination_no_description(self) -> None:
        c = _make_connector()
        src = c._format_nomination({"congress": "118"})
        assert src is None
