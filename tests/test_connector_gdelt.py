"""Tests for GdeltConnector — global news volume/tone."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.gdelt import GdeltConnector
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector() -> GdeltConnector:
    return GdeltConnector(config=None)


def _timeline_response(*, spike: bool = False) -> dict:
    """Build a GDELT timeline vol response."""
    if spike:
        data = [
            {"date": "2024-03-01", "value": 10},
            {"date": "2024-03-02", "value": 12},
            {"date": "2024-03-03", "value": 80},  # spike
            {"date": "2024-03-04", "value": 11},
            {"date": "2024-03-05", "value": 9},
        ]
    else:
        data = [
            {"date": "2024-03-01", "value": 10},
            {"date": "2024-03-02", "value": 12},
            {"date": "2024-03-03", "value": 11},
            {"date": "2024-03-04", "value": 13},
            {"date": "2024-03-05", "value": 10},
        ]
    return {"timeline": [{"data": data}]}


# ── Relevance ─────────────────────────────────────────────────────────


class TestRelevance:
    def test_macro_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any", "MACRO")

    def test_election_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any", "ELECTION")

    def test_legal_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any", "LEGAL")

    def test_crypto_not_relevant(self) -> None:
        c = _make_connector()
        assert not c.is_relevant("any", "CRYPTO")

    def test_weather_not_relevant(self) -> None:
        c = _make_connector()
        assert not c.is_relevant("any", "WEATHER")


# ── Topic Extraction ─────────────────────────────────────────────────


class TestTopicExtraction:
    def test_strips_question_framing(self) -> None:
        c = _make_connector()
        topic = c._extract_topic("Will the trade war escalate?")
        assert "Will" not in topic
        assert "trade war" in topic.lower()

    def test_caps_length(self) -> None:
        c = _make_connector()
        long_q = "Will " + "x" * 200 + " happen?"
        topic = c._extract_topic(long_q)
        assert len(topic) <= 80


# ── Fetch ─────────────────────────────────────────────────────────────


class TestFetch:
    def test_returns_empty_when_not_relevant(self) -> None:
        c = _make_connector()
        result = asyncio.run(c.fetch("Will BTC hit 100k?", "CRYPTO"))
        assert result == []

    @patch("src.research.connectors.gdelt.rate_limiter")
    def test_successful_fetch_no_spike(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _timeline_response(spike=False)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(c.fetch("Will the trade war escalate?", "MACRO"))

        assert len(result) == 1
        src = result[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert src.authority_score == 0.4  # GDELT is aggregator
        assert "No spike" in src.content
        assert src.publisher == "GDELT Project"

    @patch("src.research.connectors.gdelt.rate_limiter")
    def test_spike_detection(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _timeline_response(spike=True)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(c.fetch("Will the sanctions expand?", "GEOPOLITICS"))

        assert len(result) == 1
        assert "SPIKE DETECTED" in result[0].content

    @patch("src.research.connectors.gdelt.rate_limiter")
    def test_empty_timeline_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"timeline": []}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(c.fetch("Will trade war escalate?", "MACRO"))
        assert result == []

    @patch("src.research.connectors.gdelt.rate_limiter")
    def test_non_json_response_returns_empty(self, mock_rl: MagicMock) -> None:
        """GDELT sometimes returns HTML error pages instead of JSON."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.text = "<html><body>Service Unavailable</body></html>"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(c.fetch("Will trade war escalate?", "MACRO"))
        assert result == []

    @patch("src.research.connectors.gdelt.rate_limiter")
    def test_api_error_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        result = asyncio.run(c.fetch("Will trade war escalate?", "MACRO"))
        assert result == []
