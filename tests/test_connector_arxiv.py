"""Tests for ArxivConnector — arXiv preprint search."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.arxiv_connector import ArxivConnector


# ── Fake responses ─────────────────────────────────────────────────

_ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2603.12345v1</id>
    <title>Advances in Quantum Error Correction for Topological Codes</title>
    <summary>We present novel techniques for quantum error correction
using topological codes. Our approach achieves a 10x improvement in
logical error rates compared to previous methods. We demonstrate this
on a 72-qubit processor.</summary>
    <published>2026-03-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <author><name>Carol White</name></author>
    <author><name>Dave Brown</name></author>
    <arxiv:primary_category term="quant-ph"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2603.12346v1</id>
    <title>Neural Network Scaling Laws Revisited</title>
    <summary>An updated analysis of scaling laws for large language models.</summary>
    <published>2026-03-14T00:00:00Z</published>
    <author><name>Eve Green</name></author>
    <arxiv:primary_category term="cs.LG"/>
  </entry>
</feed>"""

_EMPTY_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


# ── Query building ─────────────────────────────────────────────────


class TestQueryBuilding:
    def test_builds_query_from_question(self) -> None:
        c = ArxivConnector()
        query = c._build_query("Will quantum computing break RSA encryption?")
        assert "ti:" in query
        assert "abs:" in query
        # Should contain key terms
        assert "quantum" in query.lower()

    def test_empty_question_returns_empty(self) -> None:
        c = ArxivConnector()
        query = c._build_query("will the a")
        assert query == ""

    def test_limits_to_4_terms(self) -> None:
        c = ArxivConnector()
        query = c._build_query(
            "quantum computing machine learning neural network scaling laws"
        )
        # Count AND parts (should be at most 4)
        parts = query.split(" AND ")
        assert len(parts) <= 4


# ── Relevance ──────────────────────────────────────────────────────


class TestRelevance:
    def test_relevant_for_science(self) -> None:
        c = ArxivConnector()
        assert c.is_relevant("any question", "SCIENCE")

    def test_relevant_by_keyword(self) -> None:
        c = ArxivConnector()
        assert c.is_relevant("Will a new AI model beat GPT?", "TECH")
        assert c.is_relevant("Will CRISPR cure cancer?", "UNKNOWN")
        assert c.is_relevant("new machine learning benchmark", "UNKNOWN")

    def test_not_relevant_for_sports(self) -> None:
        c = ArxivConnector()
        assert not c.is_relevant("Will the Lakers win?", "SPORTS")


# ── XML parsing ────────────────────────────────────────────────────


class TestParsing:
    def test_parse_atom_entries(self) -> None:
        c = ArxivConnector()
        sources = c._parse_atom(_ATOM_RESPONSE, max_results=5)
        assert len(sources) == 2

        s1 = sources[0]
        assert "Quantum Error Correction" in s1.title
        assert s1.publisher == "arXiv.org"
        assert s1.extraction_method == "api"
        assert s1.authority_score == 1.0
        assert "2026-03-15" in s1.date
        assert "Alice Smith" in s1.content
        assert "et al." in s1.content  # 4 authors, should show 3 + et al.
        assert "quant-ph" in s1.content

    def test_parse_empty_feed(self) -> None:
        c = ArxivConnector()
        sources = c._parse_atom(_EMPTY_ATOM, max_results=5)
        assert sources == []

    def test_parse_invalid_xml(self) -> None:
        c = ArxivConnector()
        sources = c._parse_atom("not xml at all", max_results=5)
        assert sources == []

    def test_max_results_limits_output(self) -> None:
        c = ArxivConnector()
        sources = c._parse_atom(_ATOM_RESPONSE, max_results=1)
        assert len(sources) == 1


# ── Fetch pipeline ─────────────────────────────────────────────────


class TestFetch:
    def test_successful_fetch(self) -> None:
        c = ArxivConnector()

        mock_resp = MagicMock()
        mock_resp.text = _ATOM_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.arxiv_connector.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will quantum computing advance in 2026?", "SCIENCE")
            )

        assert len(sources) == 2
        assert all(s.publisher == "arXiv.org" for s in sources)

    def test_returns_empty_for_irrelevant(self) -> None:
        c = ArxivConnector()
        sources = asyncio.run(
            c._fetch_impl("Will the Lakers win the championship?", "SPORTS")
        )
        assert sources == []

    def test_error_swallowed_by_base(self) -> None:
        c = ArxivConnector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        with patch("src.research.connectors.arxiv_connector.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c.fetch("Will fusion energy work?", "SCIENCE")
            )
        assert sources == []

    def test_respects_max_results_config(self) -> None:
        config = MagicMock()
        config.arxiv_max_results = 1
        c = ArxivConnector(config=config)

        mock_resp = MagicMock()
        mock_resp.text = _ATOM_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.arxiv_connector.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("quantum computing", "SCIENCE")
            )

        # Check that max_results was passed to API
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["max_results"] == 1
