"""Tests for PubMedConnector -- NIH Entrez API for SCIENCE markets."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.pubmed import PubMedConnector, _PUBMED_KEYWORDS
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector(config: object | None = None) -> PubMedConnector:
    return PubMedConnector(config=config)


def _esearch_response(pmids: list[str] | None = None) -> dict:
    if pmids is None:
        pmids = ["12345678", "87654321"]
    return {"esearchresult": {"idlist": pmids}}


def _esummary_response(
    uids: list[str] | None = None,
    articles: dict | None = None,
) -> dict:
    if uids is None:
        uids = ["12345678", "87654321"]
    if articles is None:
        articles = {}
        for uid in uids:
            articles[uid] = {
                "uid": uid,
                "title": f"Study on drug treatment {uid}",
                "pubdate": "2026 Mar",
                "source": "Nature Medicine",
            }
    result = {"uids": uids}
    result.update(articles)
    return {"result": result}


# ── Two-Step Fetch ───────────────────────────────────────────────────


class TestTwoStepFetch:
    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_esearch_then_esummary(self, mock_rl: MagicMock) -> None:
        """PubMed uses two sequential API calls: esearch -> esummary."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        esearch_resp = MagicMock()
        esearch_resp.json.return_value = _esearch_response(["11111111"])
        esearch_resp.raise_for_status = MagicMock()

        esummary_resp = MagicMock()
        esummary_resp.json.return_value = _esummary_response(
            uids=["11111111"],
            articles={
                "11111111": {
                    "uid": "11111111",
                    "title": "FDA approves new cancer drug",
                    "pubdate": "2026 Jan",
                    "source": "The Lancet",
                },
            },
        )
        esummary_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[esearch_resp, esummary_resp]
        )
        c._client = mock_client

        result = asyncio.run(c._fetch_impl("Will the FDA approve the new cancer drug?", "SCIENCE"))

        assert len(result) == 1
        assert result[0].title == "PubMed: FDA approves new cancer drug"
        # Two GET calls: esearch + esummary
        assert mock_client.get.call_count == 2


# ── Category Relevance ───────────────────────────────────────────────


class TestRelevance:
    def test_science_category_triggers(self) -> None:
        """SCIENCE category with matching keywords is relevant."""
        c = _make_connector()
        assert c.is_relevant("Will the FDA approve the new drug?", "SCIENCE")

    def test_science_with_trial_keyword(self) -> None:
        c = _make_connector()
        assert c.is_relevant("Will the clinical trial succeed?", "SCIENCE")

    def test_non_science_not_relevant(self) -> None:
        """Non-SCIENCE categories are never relevant."""
        c = _make_connector()
        assert not c.is_relevant("Will the FDA approve the new drug?", "MACRO")
        assert not c.is_relevant("Will the FDA approve the new drug?", "CRYPTO")
        assert not c.is_relevant("Will the FDA approve the new drug?", "ELECTION")

    def test_science_without_medical_keywords(self) -> None:
        """SCIENCE category but no matching keywords is not relevant."""
        c = _make_connector()
        # This question has no PubMed keywords
        assert not c.is_relevant("Will the rocket launch succeed?", "SCIENCE")

    def test_relevant_categories_is_science_only(self) -> None:
        c = _make_connector()
        assert c.relevant_categories() == {"SCIENCE"}


# ── API Key Handling ─────────────────────────────────────────────────


class TestAPIKey:
    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_api_key_appended_when_present(self, mock_rl: MagicMock) -> None:
        """When NCBI_API_KEY is set, it is appended to API params."""
        mock_rl.get.return_value.acquire = AsyncMock()

        with patch.dict(os.environ, {"NCBI_API_KEY": "my-ncbi-key"}, clear=False):
            c = _make_connector()

            esearch_resp = MagicMock()
            esearch_resp.json.return_value = _esearch_response([])
            esearch_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=esearch_resp)
            c._client = mock_client

            asyncio.run(c._fetch_impl("Will the new vaccine work?", "SCIENCE"))

            # Check that api_key param was included in the esearch call
            call_args = mock_client.get.call_args
            params = call_args.kwargs.get("params", call_args.args[1] if len(call_args.args) > 1 else {})
            assert params.get("api_key") == "my-ncbi-key"

    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_no_api_key_param_when_absent(self, mock_rl: MagicMock) -> None:
        """When no API key is available, api_key param is not included."""
        mock_rl.get.return_value.acquire = AsyncMock()

        with patch.dict(os.environ, {}, clear=False):
            # Ensure NCBI_API_KEY is not set
            os.environ.pop("NCBI_API_KEY", None)
            c = _make_connector()

            esearch_resp = MagicMock()
            esearch_resp.json.return_value = _esearch_response([])
            esearch_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=esearch_resp)
            c._client = mock_client

            asyncio.run(c._fetch_impl("Will the new vaccine work?", "SCIENCE"))

            call_args = mock_client.get.call_args
            params = call_args.kwargs.get("params", call_args.args[1] if len(call_args.args) > 1 else {})
            assert "api_key" not in params


# ── Article Formatting ───────────────────────────────────────────────


class TestFormatting:
    def test_formats_article_content(self) -> None:
        """Article content includes title, journal, date, PMID, and URL."""
        c = _make_connector()
        article = {
            "uid": "99999999",
            "title": "New mRNA therapy shows promise",
            "pubdate": "2026 Feb 15",
            "source": "Cell",
        }
        src = c._parse_article(article)
        assert src is not None
        assert "Title: New mRNA therapy shows promise" in src.content
        assert "Journal: Cell" in src.content
        assert "Published: 2026 Feb 15" in src.content
        assert "PMID: 99999999" in src.content
        assert "https://pubmed.ncbi.nlm.nih.gov/99999999/" in src.content

    def test_publisher_is_journal_name(self) -> None:
        """Publisher field is set to the journal name."""
        c = _make_connector()
        article = {
            "uid": "11111111",
            "title": "Study title",
            "pubdate": "2026 Mar",
            "source": "The New England Journal of Medicine",
        }
        src = c._parse_article(article)
        assert src is not None
        assert src.publisher == "The New England Journal of Medicine"

    def test_snippet_format(self) -> None:
        """Snippet includes title, journal, and date."""
        c = _make_connector()
        article = {
            "uid": "22222222",
            "title": "Test title",
            "pubdate": "2026 Jan",
            "source": "Science",
        }
        src = c._parse_article(article)
        assert src is not None
        assert "Test title" in src.snippet
        assert "Science" in src.snippet
        assert "2026 Jan" in src.snippet


# ── Authority Score ──────────────────────────────────────────────────


class TestAuthorityScore:
    def test_authority_score_is_1_0(self) -> None:
        c = _make_connector()
        article = {
            "uid": "33333333",
            "title": "High quality research",
            "pubdate": "2026 Mar",
            "source": "Nature",
        }
        src = c._parse_article(article)
        assert src is not None
        assert src.authority_score == 1.0

    def test_extraction_method_is_api(self) -> None:
        c = _make_connector()
        article = {
            "uid": "44444444",
            "title": "Another study",
            "pubdate": "2026 Mar",
            "source": "BMJ",
        }
        src = c._parse_article(article)
        assert src is not None
        assert src.extraction_method == "api"


# ── Empty Results ────────────────────────────────────────────────────


class TestEmptyResults:
    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_returns_empty_when_no_pmids(self, mock_rl: MagicMock) -> None:
        """Returns empty list when esearch finds no PMIDs."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        esearch_resp = MagicMock()
        esearch_resp.json.return_value = _esearch_response([])
        esearch_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=esearch_resp)
        c._client = mock_client

        result = asyncio.run(c._fetch_impl("Will the vaccine be approved?", "SCIENCE"))
        assert result == []
        # Only esearch should be called, not esummary
        assert mock_client.get.call_count == 1

    def test_returns_empty_when_not_relevant(self) -> None:
        c = _make_connector()
        result = asyncio.run(c._fetch_impl("Will the market crash?", "MACRO"))
        assert result == []

    def test_parse_article_returns_none_without_uid(self) -> None:
        c = _make_connector()
        result = c._parse_article({"title": "No UID article"})
        assert result is None


# ── Error Handling ───────────────────────────────────────────────────


class TestErrorHandling:
    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_api_failure_returns_empty_via_fetch(self, mock_rl: MagicMock) -> None:
        """API errors are caught by the base class fetch() wrapper."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API unavailable"))
        c._client = mock_client

        # Use the safe fetch() wrapper which catches exceptions
        result = asyncio.run(c.fetch("Will the new drug trial succeed?", "SCIENCE"))
        assert result == []

    @patch("src.research.connectors.pubmed.rate_limiter")
    def test_esummary_failure_raises(self, mock_rl: MagicMock) -> None:
        """If esearch succeeds but esummary fails, exception propagates from _fetch_impl."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        esearch_resp = MagicMock()
        esearch_resp.json.return_value = _esearch_response(["11111111"])
        esearch_resp.raise_for_status = MagicMock()

        esummary_resp = MagicMock()
        esummary_resp.raise_for_status.side_effect = RuntimeError("esummary error")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[esearch_resp, esummary_resp]
        )
        c._client = mock_client

        with pytest.raises(RuntimeError, match="esummary error"):
            asyncio.run(c._fetch_impl("Will the vaccine work?", "SCIENCE"))
