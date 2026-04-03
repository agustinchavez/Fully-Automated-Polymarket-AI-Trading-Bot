"""Tests for GoogleTrendsConnector -- search interest spike detection."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.google_trends import GoogleTrendsConnector
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector(config: object | None = None) -> GoogleTrendsConnector:
    return GoogleTrendsConnector(config=config)


def _serpapi_response(values: list[int] | None = None) -> dict:
    """Build a minimal SerpAPI interest_over_time response."""
    if values is None:
        values = [30, 35, 40, 38, 42, 45, 50, 55, 60, 70, 80, 90]
    timeline_data = [
        {"values": [{"extracted_value": v}]} for v in values
    ]
    return {"interest_over_time": {"timeline_data": timeline_data}}


def _tavily_response(title: str = "Trend article", content: str = "Big spike") -> dict:
    return {"results": [{"title": title, "content": content}]}


# ── Keyword Extraction ───────────────────────────────────────────────


class TestKeywordExtraction:
    def test_simple_question(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("Will Bitcoin reach 100k?")
        assert "Bitcoin" in kw

    def test_leading_verb_stripped(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("Will the Fed raise rates?")
        assert kw.lower().startswith("fed")

    def test_stop_words_removed(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("Is the economy in a recession?")
        # "economy" and "recession" should be in result, not "is", "the", "in", "a"
        words = kw.lower().split()
        assert "economy" in words
        assert "the" not in words
        assert "is" not in words

    def test_up_to_three_words(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("Will Federal Reserve raise interest rates again this year?")
        words = kw.split()
        assert len(words) <= 3

    def test_empty_question(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("")
        assert kw == ""

    def test_only_stop_words(self) -> None:
        c = _make_connector()
        kw = c._extract_keyword("Will the a be?")
        assert kw == ""


# ── Spike Computation ────────────────────────────────────────────────


class TestSpikeComputation:
    def test_spike_ratio_calculation(self) -> None:
        c = _make_connector()
        # Values: baseline flat at 10, recent spike to 40
        values = [10, 10, 10, 10, 10, 10, 10, 10, 40, 40, 40, 40]
        result = c._compute_spike(values)
        # avg_recent (last 4) = 40, avg_full = (8*10 + 4*40)/12 = 240/12 = 20
        # spike_ratio = 40 / 20 = 2.0
        assert result["spike_ratio"] >= 2.0
        assert result["current_index"] == 40

    def test_flat_trend_ratio_near_one(self) -> None:
        c = _make_connector()
        values = [50, 50, 50, 50, 50, 50, 50, 50]
        result = c._compute_spike(values)
        assert result["spike_ratio"] == 1.0
        assert result["current_index"] == 50

    def test_empty_values(self) -> None:
        c = _make_connector()
        result = c._compute_spike([])
        assert result["spike_ratio"] == 1.0
        assert result["current_index"] == 0

    def test_single_value(self) -> None:
        c = _make_connector()
        result = c._compute_spike([75])
        assert result["spike_ratio"] == 1.0
        assert result["current_index"] == 75

    def test_seven_day_thirty_day_ratio_semantics(self) -> None:
        """Recent 4 points (30d) vs full series (baseline)."""
        c = _make_connector()
        # 8 points baseline=20, last 4 at 60
        values = [20, 20, 20, 20, 20, 20, 20, 20, 60, 60, 60, 60]
        result = c._compute_spike(values)
        # avg_recent = 60, avg_full = (8*20+4*60)/12 = 400/12 ~ 33.33
        # spike_ratio = round(60 / 33.33, 2) = 1.8
        assert result["spike_ratio"] == 1.8


# ── SerpAPI Gating ───────────────────────────────────────────────────


class TestSerpAPIGating:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "test-serp-key", "TAVILY_API_KEY": ""}, clear=False)
    def test_serpapi_gated_by_env_var(self, mock_rl: MagicMock) -> None:
        """SerpAPI only called when SERPAPI_KEY env var is present."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _serpapi_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )

        assert len(result) == 1
        mock_client.get.assert_called_once()

    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "", "TAVILY_API_KEY": ""}, clear=False)
    def test_serpapi_not_called_without_key(self, mock_rl: MagicMock) -> None:
        """SerpAPI not called when SERPAPI_KEY env var is empty."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )
        # No API keys -> empty result
        assert result == []
        mock_client.get.assert_not_called()

    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "test-key", "TAVILY_API_KEY": ""}, clear=False)
    def test_serpapi_gated_by_volume_threshold(self, mock_rl: MagicMock) -> None:
        """SerpAPI skipped when market volume below threshold (default 50k)."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        c._client = mock_client

        # volume_usd=10_000 is below the 50k default threshold
        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=10_000.0)
        )
        # SerpAPI skipped + no Tavily key -> empty
        assert result == []
        mock_client.get.assert_not_called()


# ── Tavily Always Runs ───────────────────────────────────────────────


class TestTavily:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "", "TAVILY_API_KEY": "test-tavily-key"}, clear=False)
    def test_tavily_runs_when_key_present(self, mock_rl: MagicMock) -> None:
        """Tavily always runs when TAVILY_API_KEY is set, regardless of volume."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_resp = MagicMock()
        mock_resp.json.return_value = _tavily_response()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=0.0)
        )
        assert len(result) == 1
        mock_client.post.assert_called_once()
        # Snippet should note narrative context (no SerpAPI data)
        assert "narrative context" in result[0].snippet


# ── Combined Output ──────────────────────────────────────────────────


class TestCombinedOutput:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "serp-key", "TAVILY_API_KEY": "tavily-key"}, clear=False)
    def test_combined_serpapi_and_tavily(self, mock_rl: MagicMock) -> None:
        """Both sources contribute to a single FetchedSource."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        serpapi_resp = MagicMock()
        serpapi_resp.json.return_value = _serpapi_response()
        serpapi_resp.raise_for_status = MagicMock()

        tavily_resp = MagicMock()
        tavily_resp.json.return_value = _tavily_response("Major spike", "Interest is surging")
        tavily_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=serpapi_resp)
        mock_client.post = AsyncMock(return_value=tavily_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )
        assert len(result) == 1
        src = result[0]
        # Spike ratio from SerpAPI present
        assert "Spike ratio" in src.content
        # Narrative from Tavily present
        assert "Narrative" in src.content


# ── Graceful Degradation ─────────────────────────────────────────────


class TestGracefulDegradation:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "", "TAVILY_API_KEY": "tavily-key"}, clear=False)
    def test_tavily_only_when_no_serpapi_key(self, mock_rl: MagicMock) -> None:
        """Connector works with Tavily-only when SerpAPI key is absent."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        tavily_resp = MagicMock()
        tavily_resp.json.return_value = _tavily_response()
        tavily_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=tavily_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO")
        )
        assert len(result) == 1
        # Should not contain SerpAPI data
        assert "Spike ratio" not in result[0].content

    @patch.dict(os.environ, {"SERPAPI_KEY": "", "TAVILY_API_KEY": ""}, clear=False)
    def test_returns_empty_when_no_api_keys(self) -> None:
        """Returns empty list when neither API key is present."""
        c = _make_connector()
        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO")
        )
        assert result == []


# ── Error Handling ───────────────────────────────────────────────────


class TestErrorHandling:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "serp-key", "TAVILY_API_KEY": "tavily-key"}, clear=False)
    def test_serpapi_fails_tavily_still_works(self, mock_rl: MagicMock) -> None:
        """When SerpAPI raises, Tavily results are still returned."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        tavily_resp = MagicMock()
        tavily_resp.json.return_value = _tavily_response()
        tavily_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("SerpAPI down"))
        mock_client.post = AsyncMock(return_value=tavily_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )
        assert len(result) == 1
        # SerpAPI failed, so no spike ratio in content
        assert "Spike ratio" not in result[0].content
        assert "Narrative" in result[0].content

    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "serp-key", "TAVILY_API_KEY": "tavily-key"}, clear=False)
    def test_both_fail_returns_empty(self, mock_rl: MagicMock) -> None:
        """When both APIs fail, returns empty list."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("SerpAPI down"))
        mock_client.post = AsyncMock(side_effect=RuntimeError("Tavily down"))
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )
        assert result == []


# ── behavioral_signal and Authority Score ─────────────────────────────


class TestMetadata:
    @patch("src.research.connectors.google_trends.rate_limiter")
    @patch.dict(os.environ, {"SERPAPI_KEY": "key", "TAVILY_API_KEY": "key"}, clear=False)
    def test_behavioral_signal_in_raw(self, mock_rl: MagicMock) -> None:
        """raw dict contains behavioral_signal with expected fields."""
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        serpapi_resp = MagicMock()
        serpapi_resp.json.return_value = _serpapi_response()
        serpapi_resp.raise_for_status = MagicMock()

        tavily_resp = MagicMock()
        tavily_resp.json.return_value = _tavily_response()
        tavily_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=serpapi_resp)
        mock_client.post = AsyncMock(return_value=tavily_resp)
        c._client = mock_client

        result = asyncio.run(
            c._fetch_impl("Will Bitcoin crash?", "CRYPTO", volume_usd=100_000.0)
        )
        assert len(result) == 1
        raw = result[0].raw
        assert "behavioral_signal" in raw
        signal = raw["behavioral_signal"]
        assert signal["source"] == "google_trends"
        assert signal["signal_type"] == "search_trend"
        assert "value" in signal
        assert "current_index" in signal

    def test_authority_score_is_0_6(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="Bitcoin",
            spike_ratio=2.0,
            current_index=80,
            narrative_context="Some context",
            serpapi_ok=True,
        )
        assert len(sources) == 1
        assert sources[0].authority_score == 0.6

    def test_extraction_method_is_api(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="Bitcoin",
            spike_ratio=1.0,
            current_index=50,
            narrative_context="context",
            serpapi_ok=False,
        )
        assert sources[0].extraction_method == "api"


# ── Relevant Categories ──────────────────────────────────────────────


class TestRelevantCategories:
    def test_common_categories_included(self) -> None:
        c = _make_connector()
        cats = c.relevant_categories()
        for expected in ["MACRO", "ELECTION", "CRYPTO", "TECHNOLOGY", "SPORTS", "UNKNOWN"]:
            assert expected in cats

    def test_is_relevant_for_supported_category(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any question", "CRYPTO")
        assert c.is_relevant("any question", "ELECTION")

    def test_not_relevant_for_unsupported_category(self) -> None:
        c = _make_connector()
        # Should only be not relevant for categories NOT in the set
        cats = c.relevant_categories()
        fake_cat = "NICHE_NONSENSE"
        assert fake_cat not in cats
        assert not c.is_relevant("any question", fake_cat)


# ── Direction Labels ─────────────────────────────────────────────────


class TestDirectionLabels:
    def test_normal_direction(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="test", spike_ratio=1.0, current_index=50,
            narrative_context="ctx", serpapi_ok=True,
        )
        assert "NORMAL" in sources[0].content

    def test_elevated_direction(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="test", spike_ratio=1.7, current_index=50,
            narrative_context="ctx", serpapi_ok=True,
        )
        assert "ELEVATED" in sources[0].content

    def test_strong_direction(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="test", spike_ratio=2.5, current_index=50,
            narrative_context="ctx", serpapi_ok=True,
        )
        assert "STRONG" in sources[0].content

    def test_viral_direction(self) -> None:
        c = _make_connector()
        sources = c._build_source(
            keyword="test", spike_ratio=4.0, current_index=50,
            narrative_context="ctx", serpapi_ok=True,
        )
        assert "VIRAL" in sources[0].content
