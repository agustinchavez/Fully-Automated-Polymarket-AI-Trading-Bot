"""Tests for MetaculusConnector — expert forecaster probabilities."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.metaculus import MetaculusConnector
from src.research.source_fetcher import FetchedSource

# All TestFetch tests need an API key to pass the early-exit guard
_KEY_PATCH = patch.object(MetaculusConnector, "_get_api_key", return_value="test-key")


def _make_metaculus_response(
    title: str = "Will the Fed cut rates?",
    prob: float = 0.61,
    forecasters: int = 847,
) -> dict:
    return {
        "results": [
            {
                "id": 12345,
                "title": title,
                "community_prediction": {"full": {"q2": prob}},
                "number_of_forecasters": forecasters,
            }
        ]
    }


class TestTokenize:
    def test_basic_tokenization(self) -> None:
        c = MetaculusConnector()
        tokens = c._tokenize("Hello World test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stop_words_removed(self) -> None:
        c = MetaculusConnector()
        tokens = c._tokenize("Will the Fed be cutting rates?")
        assert "will" not in tokens
        assert "the" not in tokens
        assert "be" not in tokens
        assert "fed" in tokens
        assert "cutting" in tokens
        assert "rates" in tokens

    def test_single_char_words_removed(self) -> None:
        c = MetaculusConnector()
        tokens = c._tokenize("a b c real word")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "real" in tokens
        assert "word" in tokens

    def test_empty_string(self) -> None:
        c = MetaculusConnector()
        tokens = c._tokenize("")
        assert tokens == set()


class TestJaccard:
    def test_identical_sets(self) -> None:
        c = MetaculusConnector()
        a = {"fed", "cut", "rates"}
        b = {"fed", "cut", "rates"}
        assert c._jaccard_similarity(a, b) == 1.0

    def test_disjoint_sets(self) -> None:
        c = MetaculusConnector()
        a = {"fed", "cut", "rates"}
        b = {"bitcoin", "price", "moon"}
        assert c._jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        c = MetaculusConnector()
        a = {"fed", "cut", "rates"}
        b = {"fed", "raise", "rates"}
        # intersection = {fed, rates} = 2, union = {fed, cut, rates, raise} = 4
        assert c._jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_empty_set_returns_zero(self) -> None:
        c = MetaculusConnector()
        assert c._jaccard_similarity(set(), {"hello"}) == 0.0
        assert c._jaccard_similarity({"hello"}, set()) == 0.0
        assert c._jaccard_similarity(set(), set()) == 0.0


class TestRelevance:
    def test_relevant_for_macro(self) -> None:
        c = MetaculusConnector()
        assert c.is_relevant("Will inflation rise?", "MACRO")

    def test_relevant_for_election(self) -> None:
        c = MetaculusConnector()
        assert c.is_relevant("Who wins the election?", "ELECTION")

    def test_relevant_for_science(self) -> None:
        c = MetaculusConnector()
        assert c.is_relevant("Will AI pass the Turing test?", "SCIENCE")

    def test_relevant_for_all_categories(self) -> None:
        c = MetaculusConnector()
        for cat in [
            "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
            "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
            "ENTERTAINMENT", "TECH", "REGULATION", "UNKNOWN",
        ]:
            assert c.is_relevant("any question", cat)


class TestFetch:
    def test_match_above_jaccard_returns_source(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response(
                title="Will the Fed cut rates?", prob=0.61, forecasters=847
            )
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert len(sources) == 1
        src = sources[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert "consensus_signal" in src.raw
        signal = src.raw["consensus_signal"]
        assert signal["platform"] == "metaculus"
        assert signal["price"] == 0.61
        assert signal["forecasters"] == 847

    def test_match_below_jaccard_returns_empty(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response(
                title="Completely different unrelated topic here",
                prob=0.50,
                forecasters=100,
            )
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert sources == []

    def test_below_min_forecasters_returns_empty(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response(
                title="Will the Fed cut rates?",
                prob=0.61,
                forecasters=5,  # Below default min of 20
            )
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert sources == []

    def test_no_api_results_returns_empty(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert sources == []

    def test_api_error_returns_empty_via_fetch(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                # Use .fetch() to go through base class circuit breaker
                sources = asyncio.run(
                    c.fetch("Will the Fed cut rates?", "MACRO")
                )

        assert sources == []

    def test_prediction_missing_returns_empty(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "results": [
                    {
                        "id": 12345,
                        "title": "Will the Fed cut rates?",
                        "community_prediction": {"full": {}},  # No q2
                        "number_of_forecasters": 847,
                    }
                ]
            }
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert sources == []

    def test_authority_score(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response()
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert len(sources) == 1
        assert sources[0].authority_score == 0.95

    def test_confidence_is_jaccard_in_raw(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response(
                title="Will the Fed cut rates?", prob=0.70, forecasters=200
            )
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert len(sources) == 1
        signal = sources[0].raw["consensus_signal"]
        # Jaccard of identical question to matching title should be 1.0
        assert signal["confidence"] > 0.0

    def test_url_contains_question_id(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()

            mock_resp = MagicMock()
            mock_resp.json.return_value = _make_metaculus_response()
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            c._client = mock_client

            with patch("src.research.connectors.metaculus.rate_limiter") as mock_rl:
                mock_rl.get.return_value.acquire = AsyncMock()
                sources = asyncio.run(
                    c._fetch_impl("Will the Fed cut rates?", "MACRO")
                )

        assert len(sources) == 1
        assert "12345" in sources[0].url

    def test_empty_search_terms_returns_empty(self) -> None:
        with _KEY_PATCH:
            c = MetaculusConnector()
            # All stop words — _extract_search_terms returns ""
            sources = asyncio.run(
                c._fetch_impl("will the a an be?", "MACRO")
            )
        assert sources == []

    def test_no_api_key_returns_empty(self) -> None:
        """Connector should return [] when no API key is configured."""
        c = MetaculusConnector()
        sources = asyncio.run(
            c._fetch_impl("Will the Fed cut rates?", "MACRO")
        )
        assert sources == []
