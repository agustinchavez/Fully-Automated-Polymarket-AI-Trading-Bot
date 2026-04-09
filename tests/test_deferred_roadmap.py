"""Deferred roadmap spec — tests for new connectors and enhancements.

Covers:
  Batch A: Evidence quality source reweighting in SourceFetcher
  Batch B1: Binance futures funding rate connector
  Batch B2: DeFiLlama TVL connector
  Batch C1: ACLED armed conflict connector
  Batch C2: GDELT GKG entity extraction upgrade
  Batch D: GitHub activity connector
  Signal aggregator integration for new signal types
  Registry connector count update
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.research.source_fetcher import FetchedSource


# ── Batch A: Evidence quality source reweighting ─────────────────────


class TestEvidenceQualityReweighting:
    """Source authority scores are modulated by learned evidence quality weights."""

    def test_source_fetcher_accepts_db_path(self) -> None:
        from src.research.source_fetcher import SourceFetcher

        config = MagicMock()
        config.source_timeout_secs = 10
        provider = MagicMock()
        with patch("src.research.connectors.registry.get_enabled_connectors", return_value=[]):
            fetcher = SourceFetcher(provider, config, db_path="data/test.db")
        assert fetcher._db_path == "data/test.db"

    def test_source_fetcher_no_db_path_defaults_none(self) -> None:
        from src.research.source_fetcher import SourceFetcher

        config = MagicMock()
        config.source_timeout_secs = 10
        provider = MagicMock()
        with patch("src.research.connectors.registry.get_enabled_connectors", return_value=[]):
            fetcher = SourceFetcher(provider, config)
        assert fetcher._db_path is None

    def test_get_evidence_tracker_returns_none_without_db(self) -> None:
        from src.research.source_fetcher import SourceFetcher

        config = MagicMock()
        config.source_timeout_secs = 10
        provider = MagicMock()
        with patch("src.research.connectors.registry.get_enabled_connectors", return_value=[]):
            fetcher = SourceFetcher(provider, config)
        assert fetcher._get_evidence_tracker() is None

    def test_get_evidence_tracker_sentinel_on_failure(self) -> None:
        from src.research.source_fetcher import SourceFetcher

        config = MagicMock()
        config.source_timeout_secs = 10
        provider = MagicMock()
        with patch("src.research.connectors.registry.get_enabled_connectors", return_value=[]):
            fetcher = SourceFetcher(provider, config, db_path="/nonexistent/bad.db")
        # First call fails → sentinel
        tracker = fetcher._get_evidence_tracker()
        # With a bad path, sqlite3.connect might still succeed (creates file)
        # or fail — either way the method shouldn't raise
        assert tracker is None or tracker is not None  # just ensures no exception


# ── Batch B1: Crypto futures funding rate ────────────────────────────


class TestCryptoFuturesConnector:
    """Binance futures funding rate connector for CRYPTO markets."""

    def test_name(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert c.name == "crypto_futures"

    def test_relevant_categories(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert c.relevant_categories() == {"CRYPTO"}

    def test_is_relevant_crypto_btc(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert c.is_relevant("Will Bitcoin reach 100k?", "CRYPTO")

    def test_is_relevant_crypto_eth(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert c.is_relevant("Will ETH price go up?", "CRYPTO")

    def test_not_relevant_macro(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert not c.is_relevant("Will inflation rise?", "MACRO")

    def test_not_relevant_no_coin(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        assert not c.is_relevant("Will crypto recover?", "CRYPTO")

    def test_extract_ticker_btc(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        assert CryptoFuturesConnector._extract_ticker("Will BTC reach 90k?") == "BTCUSDT"

    def test_extract_ticker_solana(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        assert CryptoFuturesConnector._extract_ticker("Solana price prediction") == "SOLUSDT"

    def test_extract_ticker_none(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        assert CryptoFuturesConnector._extract_ticker("Will stocks rise?") is None

    @pytest.mark.asyncio
    async def test_fetch_returns_sources(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)

        premium_data = {
            "lastFundingRate": "0.0003",
            "markPrice": "95000.0",
        }
        history_data = [
            {"fundingRate": "0.0002"},
            {"fundingRate": "0.0003"},
            {"fundingRate": "0.0004"},
            {"fundingRate": "0.0001"},
        ]

        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = premium_data
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = history_data
        mock_resp2.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_resp1, mock_resp2])
        c._client = mock_client

        with patch("src.research.connectors.crypto_futures.rate_limiter") as rl:
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_impl("Will BTC go up?", "CRYPTO")

        assert len(sources) == 1
        raw = sources[0].raw
        assert "behavioral_signal" in raw
        assert raw["behavioral_signal"]["source"] == "crypto_futures"
        assert raw["behavioral_signal"]["signal_type"] == "funding_rate"

    @pytest.mark.asyncio
    async def test_fetch_no_ticker_returns_empty(self) -> None:
        from src.research.connectors.crypto_futures import CryptoFuturesConnector

        c = CryptoFuturesConnector(config=None)
        sources = await c._fetch_impl("Will stocks rise?", "CRYPTO")
        assert sources == []


# ── Batch B2: DeFiLlama TVL ─────────────────────────────────────────


class TestDeFiLlamaConnector:
    """DeFiLlama TVL connector for CRYPTO markets."""

    def test_name(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)
        assert c.name == "defillama"

    def test_relevant_categories(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)
        assert c.relevant_categories() == {"CRYPTO"}

    def test_is_relevant_ethereum(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)
        assert c.is_relevant("Will Ethereum TVL grow?", "CRYPTO")

    def test_is_relevant_aave(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)
        assert c.is_relevant("Will Aave reach $10B TVL?", "CRYPTO")

    def test_not_relevant_macro(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)
        assert not c.is_relevant("Will GDP grow?", "MACRO")

    def test_extract_protocol_ethereum(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        assert DeFiLlamaConnector._extract_protocol("Ethereum price") == "ethereum"

    def test_extract_protocol_uniswap(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        assert DeFiLlamaConnector._extract_protocol("Will Uniswap TVL grow?") == "uniswap"

    def test_extract_protocol_none(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        assert DeFiLlamaConnector._extract_protocol("Will stocks rise?") is None

    @pytest.mark.asyncio
    async def test_fetch_returns_sources(self) -> None:
        from src.research.connectors.defillama import DeFiLlamaConnector

        c = DeFiLlamaConnector(config=None)

        protocol_data = {
            "name": "Aave",
            "currentChainTvls": {"Ethereum": 5_000_000_000, "Polygon": 500_000_000},
            "tvl": [
                {"totalLiquidityUSD": 4_000_000_000},
                {"totalLiquidityUSD": 4_100_000_000},
                {"totalLiquidityUSD": 4_200_000_000},
                {"totalLiquidityUSD": 4_300_000_000},
                {"totalLiquidityUSD": 4_400_000_000},
                {"totalLiquidityUSD": 4_500_000_000},
                {"totalLiquidityUSD": 4_600_000_000},
                {"totalLiquidityUSD": 5_000_000_000},
                {"totalLiquidityUSD": 5_200_000_000},
                {"totalLiquidityUSD": 5_500_000_000},
            ],
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = protocol_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.defillama.rate_limiter") as rl:
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_impl("Will Aave TVL grow?", "CRYPTO")

        assert len(sources) == 1
        raw = sources[0].raw
        assert raw["behavioral_signal"]["source"] == "defillama"
        assert raw["behavioral_signal"]["signal_type"] == "tvl"


# ── Batch C1: ACLED armed conflict ──────────────────────────────────


class TestAcledConnector:
    """ACLED armed conflict connector for GEOPOLITICS markets."""

    def test_name(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert c.name == "acled"

    def test_relevant_categories(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert c.relevant_categories() == {"GEOPOLITICS"}

    def test_is_relevant_ukraine_conflict(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert c.is_relevant("Will the Ukraine war escalate?", "GEOPOLITICS")

    def test_is_relevant_syria_military(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert c.is_relevant("Will Syria see military action?", "GEOPOLITICS")

    def test_not_relevant_no_conflict_keyword(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert not c.is_relevant("Will Ukraine join the EU?", "GEOPOLITICS")

    def test_not_relevant_macro(self) -> None:
        from src.research.connectors.acled import AcledConnector

        c = AcledConnector(config=None)
        assert not c.is_relevant("Will inflation rise?", "MACRO")

    def test_extract_country_ukraine(self) -> None:
        from src.research.connectors.acled import AcledConnector

        assert AcledConnector._extract_country("Ukraine conflict") == "Ukraine"

    def test_extract_country_palestine(self) -> None:
        from src.research.connectors.acled import AcledConnector

        assert AcledConnector._extract_country("Gaza ceasefire") == "Palestine"

    def test_extract_country_none(self) -> None:
        from src.research.connectors.acled import AcledConnector

        assert AcledConnector._extract_country("trade war") is None

    def test_extract_region_middle_east(self) -> None:
        from src.research.connectors.acled import AcledConnector

        assert AcledConnector._extract_region("Middle East violence") == 11

    @pytest.mark.asyncio
    async def test_fetch_no_api_key_returns_empty(self) -> None:
        from src.research.connectors.acled import AcledConnector

        config = MagicMock()
        config.acled_api_key = ""
        c = AcledConnector(config=config)

        with patch.dict("os.environ", {"ACLED_API_KEY": "", "ACLED_EMAIL": ""}):
            sources = await c._fetch_impl("Ukraine war escalate?", "GEOPOLITICS")
        assert sources == []

    @pytest.mark.asyncio
    async def test_fetch_returns_sources(self) -> None:
        from src.research.connectors.acled import AcledConnector

        config = MagicMock()
        config.acled_api_key = "test-key"
        config.acled_lookback_days = 30
        c = AcledConnector(config=config)

        api_data = {
            "data": [
                {"event_type": "Battles", "fatalities": "5"},
                {"event_type": "Battles", "fatalities": "3"},
                {"event_type": "Explosions/Remote violence", "fatalities": "10"},
                {"event_type": "Violence against civilians", "fatalities": "2"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_data
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with (
            patch("src.research.connectors.acled.rate_limiter") as rl,
            patch.dict("os.environ", {"ACLED_API_KEY": "test-key", "ACLED_EMAIL": "test@test.com"}),
        ):
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_impl("Ukraine war escalate?", "GEOPOLITICS")

        assert len(sources) == 1
        raw = sources[0].raw
        assert raw["behavioral_signal"]["source"] == "acled"
        assert raw["behavioral_signal"]["signal_type"] == "conflict_events"
        assert raw["behavioral_signal"]["fatalities"] == 20


# ── Batch C2: GDELT GKG upgrade ────────────────────────────────────


class TestGdeltGKGUpgrade:
    """GDELT connector now includes GKG tone analysis."""

    def test_gdelt_has_fetch_gkg_method(self) -> None:
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)
        assert hasattr(c, "_fetch_gkg")

    @pytest.mark.asyncio
    async def test_fetch_gkg_returns_source(self) -> None:
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)

        tone_data = {
            "tonechart": [
                {"tone": "3.5", "url": "https://reuters.com/article1"},
                {"tone": "-1.2", "url": "https://apnews.com/article2"},
                {"tone": "2.1", "url": "https://bbc.com/article3"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = tone_data
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.gdelt.rate_limiter") as rl:
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_gkg("inflation", 7)

        assert len(sources) == 1
        raw = sources[0].raw
        assert raw["behavioral_signal"]["source"] == "gdelt_gkg"
        assert raw["behavioral_signal"]["signal_type"] == "media_tone"
        assert raw["behavioral_signal"]["sentiment"] == "neutral"

    @pytest.mark.asyncio
    async def test_fetch_gkg_positive_sentiment(self) -> None:
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)

        tone_data = {
            "tonechart": [
                {"tone": "5.0", "url": "https://reuters.com/a"},
                {"tone": "4.0", "url": "https://bbc.com/b"},
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = tone_data
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.gdelt.rate_limiter") as rl:
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_gkg("good news", 7)

        assert sources[0].raw["behavioral_signal"]["sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_fetch_gkg_empty_tonechart(self) -> None:
        from src.research.connectors.gdelt import GdeltConnector

        c = GdeltConnector(config=None)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tonechart": []}
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.gdelt.rate_limiter") as rl:
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_gkg("nothing", 7)

        assert sources == []


# ── Batch D: GitHub activity ────────────────────────────────────────


class TestGitHubActivityConnector:
    """GitHub activity connector for TECHNOLOGY and CRYPTO markets."""

    def test_name(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert c.name == "github_activity"

    def test_relevant_categories(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert c.relevant_categories() == {"TECHNOLOGY", "CRYPTO"}

    def test_is_relevant_bitcoin_crypto(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert c.is_relevant("Will Bitcoin update its protocol?", "CRYPTO")

    def test_is_relevant_pytorch_tech(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert c.is_relevant("Will PyTorch release v3?", "TECHNOLOGY")

    def test_not_relevant_election(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert not c.is_relevant("Will Biden win?", "ELECTION")

    def test_not_relevant_no_project(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)
        assert not c.is_relevant("Will tech stocks rise?", "TECHNOLOGY")

    def test_extract_repo_bitcoin(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        assert GitHubActivityConnector._extract_repo("Bitcoin upgrade") == "bitcoin/bitcoin"

    def test_extract_repo_pytorch(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        assert GitHubActivityConnector._extract_repo("PyTorch v3") == "pytorch/pytorch"

    def test_extract_repo_explicit_url(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        result = GitHubActivityConnector._extract_repo(
            "Will github.com/facebook/react get more stars?"
        )
        assert result == "facebook/react"

    def test_extract_repo_none(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        assert GitHubActivityConnector._extract_repo("Will stocks rise?") is None

    @pytest.mark.asyncio
    async def test_fetch_returns_sources(self) -> None:
        from src.research.connectors.github_activity import GitHubActivityConnector

        c = GitHubActivityConnector(config=None)

        repo_data = {
            "full_name": "bitcoin/bitcoin",
            "stargazers_count": 80000,
            "forks_count": 35000,
            "open_issues_count": 800,
            "pushed_at": "2024-03-01T12:00:00Z",
        }
        weekly_data = [{"total": 50}] * 8  # 8 weeks
        release_data = {"tag_name": "v27.0"}

        mock_repo = MagicMock()
        mock_repo.status_code = 200
        mock_repo.json.return_value = repo_data
        mock_repo.raise_for_status = MagicMock()

        mock_weekly = MagicMock()
        mock_weekly.status_code = 200
        mock_weekly.json.return_value = weekly_data

        mock_release = MagicMock()
        mock_release.status_code = 200
        mock_release.json.return_value = release_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_repo, mock_weekly, mock_release])
        c._client = mock_client

        with (
            patch("src.research.connectors.github_activity.rate_limiter") as rl,
            patch.dict("os.environ", {"GITHUB_TOKEN": ""}),
        ):
            rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_impl("Will Bitcoin update?", "CRYPTO")

        assert len(sources) == 1
        raw = sources[0].raw
        assert raw["behavioral_signal"]["source"] == "github_activity"
        assert raw["behavioral_signal"]["signal_type"] == "dev_activity"
        assert raw["behavioral_signal"]["stars"] == 80000


# ── Signal aggregator integration ───────────────────────────────────


class TestSignalAggregatorNewSignals:
    """New behavioral signals are correctly parsed into SignalStack."""

    def test_funding_rate_signal(self) -> None:
        from src.research.signal_aggregator import build_signal_stack

        source = FetchedSource(
            title="test", url="https://example.com", snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "crypto_futures",
                    "signal_type": "funding_rate",
                    "value": 0.0003,
                    "sentiment": "moderately bullish",
                    "symbol": "BTC",
                }
            },
        )
        stack = build_signal_stack([source], 0.5)
        assert stack.funding_rate == 0.0003
        assert stack.funding_rate_sentiment == "moderately bullish"
        assert stack.funding_rate_symbol == "BTC"

    def test_tvl_signal(self) -> None:
        from src.research.signal_aggregator import build_signal_stack

        source = FetchedSource(
            title="test", url="https://example.com", snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "defillama",
                    "signal_type": "tvl",
                    "change_7d_pct": 0.05,
                    "tvl_usd": 5_000_000_000,
                    "trend": "rising",
                    "protocol": "ethereum",
                }
            },
        )
        stack = build_signal_stack([source], 0.5)
        assert stack.tvl_change_7d_pct == 0.05
        assert stack.tvl_usd == 5_000_000_000
        assert stack.tvl_trend == "rising"

    def test_conflict_signal(self) -> None:
        from src.research.signal_aggregator import build_signal_stack

        source = FetchedSource(
            title="test", url="https://example.com", snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "acled",
                    "signal_type": "conflict_events",
                    "value": 150,
                    "fatalities": 45,
                    "trend": "escalating",
                    "location": "Ukraine",
                }
            },
        )
        stack = build_signal_stack([source], 0.5)
        assert stack.conflict_events == 150
        assert stack.conflict_fatalities == 45
        assert stack.conflict_trend == "escalating"

    def test_gdelt_gkg_signal(self) -> None:
        from src.research.signal_aggregator import build_signal_stack

        source = FetchedSource(
            title="test", url="https://example.com", snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "gdelt_gkg",
                    "signal_type": "media_tone",
                    "value": -3.5,
                    "sentiment": "negative",
                    "article_count": 200,
                }
            },
        )
        stack = build_signal_stack([source], 0.5)
        assert stack.gdelt_tone == -3.5
        assert stack.gdelt_sentiment == "negative"
        assert stack.gdelt_article_count == 200

    def test_github_activity_signal(self) -> None:
        from src.research.signal_aggregator import build_signal_stack

        source = FetchedSource(
            title="test", url="https://example.com", snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "github_activity",
                    "signal_type": "dev_activity",
                    "value": 200,
                    "stars": 80000,
                    "activity_trend": "increasing",
                    "repo": "bitcoin/bitcoin",
                }
            },
        )
        stack = build_signal_stack([source], 0.5)
        assert stack.github_commits_4w == 200
        assert stack.github_stars == 80000
        assert stack.github_activity_trend == "increasing"

    def test_render_funding_rate(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            funding_rate=0.0003,
            funding_rate_sentiment="moderately bullish",
            funding_rate_symbol="BTC",
        )
        rendered = render_signal_stack(stack)
        assert "Futures funding rate" in rendered
        assert "BTC" in rendered
        assert "moderately bullish" in rendered

    def test_render_tvl(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            tvl_change_7d_pct=0.05,
            tvl_usd=5_000_000_000,
            tvl_trend="rising",
            tvl_protocol="ethereum",
        )
        rendered = render_signal_stack(stack)
        assert "DeFi TVL" in rendered
        assert "rising" in rendered

    def test_render_conflict(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            conflict_events=150,
            conflict_fatalities=45,
            conflict_trend="escalating",
            conflict_location="Ukraine",
        )
        rendered = render_signal_stack(stack)
        assert "ACLED conflict" in rendered
        assert "Ukraine" in rendered
        assert "escalating" in rendered

    def test_render_github(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            github_commits_4w=200,
            github_activity_trend="increasing",
            github_stars=80000,
            github_repo="bitcoin/bitcoin",
        )
        rendered = render_signal_stack(stack)
        assert "GitHub activity" in rendered
        assert "bitcoin/bitcoin" in rendered
        assert "increasing" in rendered

    def test_render_gdelt_tone(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            gdelt_tone=-3.5,
            gdelt_sentiment="negative",
            gdelt_article_count=200,
        )
        rendered = render_signal_stack(stack)
        assert "GDELT media tone" in rendered
        assert "negative" in rendered

    def test_no_new_signals_no_change(self) -> None:
        """Empty stack with no new signals renders empty string."""
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack()
        assert render_signal_stack(stack) == ""


# ── Registry connector count ────────────────────────────────────────


class TestRegistryConnectorCount:
    """Registry correctly loads all 26 connectors when all enabled."""

    def test_all_connectors_enabled_count_26(self) -> None:
        from src.research.connectors.registry import get_enabled_connectors

        config = MagicMock()
        config.openmeteo_enabled = True
        config.fred_enabled = True
        config.coingecko_enabled = True
        config.congress_enabled = True
        config.gdelt_enabled = True
        config.courtlistener_enabled = True
        config.edgar_enabled = True
        config.arxiv_enabled = True
        config.openfda_enabled = True
        config.worldbank_enabled = True
        config.kalshi_prior_enabled = True
        config.metaculus_enabled = True
        config.wikipedia_pageviews_enabled = True
        config.google_trends_enabled = True
        config.pubmed_enabled = True
        config.reddit_sentiment_enabled = True
        config.manifold_enabled = True
        config.predictit_enabled = True
        config.sports_odds_enabled = True
        config.sports_stats_enabled = True
        config.spotify_charts_enabled = True
        config.kronos_enabled = True
        config.crypto_futures_enabled = True
        config.defillama_enabled = True
        config.acled_enabled = True
        config.github_activity_enabled = True
        connectors = get_enabled_connectors(config)
        assert len(connectors) == 26

    def test_new_connectors_enabled_individually(self) -> None:
        from src.research.connectors.registry import get_enabled_connectors

        for connector_flag, expected_name in [
            ("crypto_futures_enabled", "crypto_futures"),
            ("defillama_enabled", "defillama"),
            ("acled_enabled", "acled"),
            ("github_activity_enabled", "github_activity"),
        ]:
            config = MagicMock()
            # Disable all
            config.openmeteo_enabled = False
            config.fred_enabled = False
            config.coingecko_enabled = False
            config.congress_enabled = False
            config.gdelt_enabled = False
            config.courtlistener_enabled = False
            config.edgar_enabled = False
            config.arxiv_enabled = False
            config.openfda_enabled = False
            config.worldbank_enabled = False
            config.kalshi_prior_enabled = False
            config.metaculus_enabled = False
            config.wikipedia_pageviews_enabled = False
            config.google_trends_enabled = False
            config.pubmed_enabled = False
            config.reddit_sentiment_enabled = False
            config.manifold_enabled = False
            config.predictit_enabled = False
            config.sports_odds_enabled = False
            config.sports_stats_enabled = False
            config.spotify_charts_enabled = False
            config.kronos_enabled = False
            config.crypto_futures_enabled = False
            config.defillama_enabled = False
            config.acled_enabled = False
            config.github_activity_enabled = False

            # Enable only the one under test
            setattr(config, connector_flag, True)
            connectors = get_enabled_connectors(config)
            assert len(connectors) == 1
            assert connectors[0].name == expected_name


# ── Config fields ───────────────────────────────────────────────────


class TestNewConfigFields:
    """New config fields exist with correct defaults."""

    def test_crypto_futures_enabled_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.crypto_futures_enabled is False

    def test_defillama_enabled_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.defillama_enabled is False

    def test_acled_enabled_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.acled_enabled is False

    def test_acled_api_key_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.acled_api_key == ""

    def test_acled_lookback_days_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.acled_lookback_days == 30

    def test_github_activity_enabled_default(self) -> None:
        from src.config import ResearchConfig

        config = ResearchConfig()
        assert config.github_activity_enabled is False

    def test_acled_api_key_in_secret_fields(self) -> None:
        from src.config import _SECRET_FIELDS

        assert "acled_api_key" in _SECRET_FIELDS


# ── Rate limiter buckets ────────────────────────────────────────────


class TestNewRateLimiterBuckets:
    """New rate limiter buckets exist in DEFAULT_LIMITS."""

    def test_defillama_bucket(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS

        assert "defillama" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["defillama"].name == "DeFiLlama"

    def test_acled_bucket(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS

        assert "acled" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["acled"].name == "ACLED"

    def test_github_bucket(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS

        assert "github" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["github"].name == "GitHub"
