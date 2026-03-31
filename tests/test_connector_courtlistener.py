"""Tests for CourtListenerConnector — legal case data."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.courtlistener import CourtListenerConnector
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector(*, api_key: str = "test-token") -> CourtListenerConnector:
    c = CourtListenerConnector(config=None)
    c._get_api_key = lambda: api_key  # type: ignore[assignment]
    return c


def _opinion_response() -> dict:
    return {
        "results": [
            {
                "caseName": "Smith v. United States",
                "court": "Supreme Court of the United States",
                "dateFiled": "2024-06-15",
                "snippet": "The court held that the statute applies...",
                "absolute_url": "/opinion/12345/smith-v-united-states/",
            }
        ]
    }


def _docket_response() -> dict:
    return {
        "results": [
            {
                "caseName": "Doe v. State of California",
                "court": "U.S. District Court, Central District of California",
                "dateFiled": "2024-05-10",
                "docketNumber": "2:24-cv-01234",
                "absolute_url": "/docket/67890/doe-v-state-of-california/",
            }
        ]
    }


# ── Relevance ─────────────────────────────────────────────────────────


class TestRelevance:
    def test_legal_category_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any", "LEGAL")

    def test_non_legal_not_relevant(self) -> None:
        c = _make_connector()
        assert not c.is_relevant("Will GDP grow?", "MACRO")

    def test_keyword_makes_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("Will the Supreme Court rule on this?", "UNKNOWN")
        assert c.is_relevant("Will the lawsuit be dismissed?", "UNKNOWN")
        assert c.is_relevant("Will the court ruling stand?", "UNKNOWN")


# ── Maintenance Window ────────────────────────────────────────────────


class TestMaintenanceWindow:
    @patch("src.research.connectors.courtlistener.datetime")
    def test_thursday_maintenance_detected(self, mock_dt: MagicMock) -> None:
        from datetime import datetime, timezone

        # Thursday 21:00 PT = Friday 05:00 UTC
        mock_now = MagicMock()
        mock_now.hour = 5  # 05:00 UTC
        mock_now.weekday.return_value = 4  # Friday UTC
        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        c = _make_connector()
        # PT hour = (5 - 8) % 24 = 21, weekday adjusted: (4 - 1) % 7 = 3 (Thursday)
        assert c._is_maintenance_window()

    @patch("src.research.connectors.courtlistener.datetime")
    def test_non_thursday_not_maintenance(self, mock_dt: MagicMock) -> None:
        from datetime import datetime, timezone

        # Monday 15:00 UTC
        mock_now = MagicMock()
        mock_now.hour = 15
        mock_now.weekday.return_value = 0  # Monday
        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        c = _make_connector()
        assert not c._is_maintenance_window()


# ── Fetch ─────────────────────────────────────────────────────────────


class TestFetch:
    def test_returns_empty_when_not_relevant(self) -> None:
        c = _make_connector()
        result = asyncio.run(c.fetch("Will BTC hit 100k?", "CRYPTO"))
        assert result == []

    def test_returns_empty_when_no_api_key(self) -> None:
        c = _make_connector(api_key="")
        result = asyncio.run(c.fetch("Will the court rule?", "LEGAL"))
        assert result == []

    @patch("src.research.connectors.courtlistener.rate_limiter")
    def test_opinion_search(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()
        c._is_maintenance_window = lambda: False

        mock_resp = MagicMock()
        mock_resp.json.return_value = _opinion_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will the Supreme Court rule on Smith?", "LEGAL")
        )

        assert len(result) == 1
        src = result[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert src.authority_score == 1.0
        assert "Smith v. United States" in src.content
        assert src.publisher == "CourtListener"
        assert "courtlistener.com" in src.url

    @patch("src.research.connectors.courtlistener.rate_limiter")
    def test_falls_back_to_dockets(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()
        c._is_maintenance_window = lambda: False

        # First call (opinions) returns empty, second call (dockets) returns data
        mock_resp_empty = MagicMock()
        mock_resp_empty.json.return_value = {"results": []}
        mock_resp_empty.raise_for_status = MagicMock()

        mock_resp_docket = MagicMock()
        mock_resp_docket.json.return_value = _docket_response()
        mock_resp_docket.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[mock_resp_empty, mock_resp_docket]
        )
        c._client = mock_client

        result = asyncio.run(
            c.fetch("Will the Doe lawsuit be dismissed?", "LEGAL")
        )

        assert len(result) == 1
        assert "Doe v. State of California" in result[0].content
        assert "2:24-cv-01234" in result[0].content

    @patch("src.research.connectors.courtlistener.rate_limiter")
    def test_api_error_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()
        c._is_maintenance_window = lambda: False

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        result = asyncio.run(c.fetch("Will the court rule?", "LEGAL"))
        assert result == []


# ── Format ────────────────────────────────────────────────────────────


class TestFormat:
    def test_format_opinion(self) -> None:
        c = _make_connector()
        opinion = _opinion_response()["results"][0]
        src = c._format_opinion(opinion)
        assert src is not None
        assert "Smith v. United States" in src.content
        assert "Supreme Court" in src.content
        assert "courtlistener.com" in src.url

    def test_format_opinion_no_case_name(self) -> None:
        c = _make_connector()
        src = c._format_opinion({"court": "test"})
        assert src is None

    def test_format_docket(self) -> None:
        c = _make_connector()
        docket = _docket_response()["results"][0]
        src = c._format_docket(docket)
        assert src is not None
        assert "Doe v. State" in src.content
        assert "2:24-cv-01234" in src.content

    def test_format_docket_no_case_name(self) -> None:
        c = _make_connector()
        src = c._format_docket({"docketNumber": "123"})
        assert src is None

    def test_long_snippet_truncated(self) -> None:
        c = _make_connector()
        opinion = _opinion_response()["results"][0].copy()
        opinion["snippet"] = "x" * 3000
        src = c._format_opinion(opinion)
        assert src is not None
        assert len(src.content) < 3000 + 200  # overhead for labels
