"""Tests for CoinGeckoConnector — crypto price data."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.coingecko import (
    CoinGeckoConnector,
    _COIN_MAP,
)
from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_connector(*, api_key: str = "test-key") -> CoinGeckoConnector:
    c = CoinGeckoConnector(config=None)
    c._get_api_key = lambda: api_key  # type: ignore[assignment]
    return c


def _price_response(coin_id: str = "bitcoin") -> dict:
    return {
        coin_id: {
            "usd": 67_500.42,
            "usd_24h_change": 2.35,
            "usd_market_cap": 1_325_000_000_000,
            "usd_24h_vol": 28_500_000_000,
        }
    }


# ── Relevance ─────────────────────────────────────────────────────────


class TestRelevance:
    def test_crypto_category_is_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("any question", "CRYPTO")

    def test_non_crypto_category_not_relevant(self) -> None:
        c = _make_connector()
        assert not c.is_relevant("Will GDP grow?", "MACRO")

    def test_keyword_makes_relevant(self) -> None:
        c = _make_connector()
        assert c.is_relevant("Will bitcoin hit 100k?", "UNKNOWN")
        assert c.is_relevant("ETH price prediction", "UNKNOWN")


# ── Coin Extraction ──────────────────────────────────────────────────


class TestCoinExtraction:
    def test_extract_single_ticker(self) -> None:
        c = _make_connector()
        coins = c._extract_coins("Will BTC hit 100k?", 3)
        assert coins == ["bitcoin"]

    def test_extract_full_name(self) -> None:
        c = _make_connector()
        coins = c._extract_coins("Will ethereum reach $5000?", 3)
        assert coins == ["ethereum"]

    def test_extract_multiple_coins(self) -> None:
        c = _make_connector()
        coins = c._extract_coins("Will BTC outperform ETH this year?", 5)
        assert "bitcoin" in coins
        assert "ethereum" in coins

    def test_deduplication(self) -> None:
        c = _make_connector()
        # "bitcoin" and "btc" both map to "bitcoin"
        coins = c._extract_coins("Will bitcoin BTC reach 100k?", 5)
        assert coins.count("bitcoin") == 1

    def test_max_coins_limit(self) -> None:
        c = _make_connector()
        coins = c._extract_coins(
            "BTC ETH SOL ADA DOGE XRP DOT", 2,
        )
        assert len(coins) <= 2

    def test_no_coins_found(self) -> None:
        c = _make_connector()
        coins = c._extract_coins("Will the market crash?", 3)
        assert coins == []


# ── Fetch ─────────────────────────────────────────────────────────────


class TestFetch:
    def test_returns_empty_when_not_relevant(self) -> None:
        c = _make_connector()
        result = asyncio.run(c.fetch("Will GDP grow?", "MACRO"))
        assert result == []

    def test_returns_empty_when_no_api_key(self) -> None:
        c = _make_connector(api_key="")
        # Even with CRYPTO type, no key still works (CoinGecko demo allows keyless)
        # but _extract_coins must find something
        result = asyncio.run(c.fetch("Will bitcoin go up?", "CRYPTO"))
        # Should attempt fetch — test that it doesn't crash
        assert isinstance(result, list)

    @patch("src.research.connectors.coingecko.rate_limiter")
    def test_successful_fetch(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        # httpx Response methods (json, raise_for_status) are sync
        mock_resp = MagicMock()
        mock_resp.json.return_value = _price_response("bitcoin")
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        # Clear cache to force API call
        c._cache.clear()
        result = asyncio.run(c.fetch("Will bitcoin hit 100k?", "CRYPTO"))

        assert len(result) == 1
        src = result[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert src.authority_score == 1.0
        assert "67,500.42" in src.content
        assert "+2.35%" in src.content
        assert src.publisher == "CoinGecko"

    @patch("src.research.connectors.coingecko.rate_limiter")
    def test_cache_hit_skips_api_call(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        # Pre-fill cache on the instance
        c._cache["bitcoin"] = (
            time.monotonic(),
            _price_response("bitcoin")["bitcoin"],
        )

        mock_client = AsyncMock()
        c._client = mock_client

        result = asyncio.run(c.fetch("Will bitcoin hit 100k?", "CRYPTO"))

        assert len(result) == 1
        # Client.get should NOT have been called (cache hit)
        mock_client.get.assert_not_called()

    @patch("src.research.connectors.coingecko.rate_limiter")
    def test_api_error_returns_empty(self, mock_rl: MagicMock) -> None:
        mock_rl.get.return_value.acquire = AsyncMock()
        c = _make_connector()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("API down"))
        c._client = mock_client

        c._cache.clear()
        result = asyncio.run(c.fetch("Will bitcoin hit 100k?", "CRYPTO"))
        assert result == []


# ── Format ────────────────────────────────────────────────────────────


class TestFormat:
    def test_format_price_content(self) -> None:
        c = _make_connector()
        data = _price_response("bitcoin")["bitcoin"]
        src = c._format_price("bitcoin", data)
        assert "Bitcoin" in src.title
        assert "$67,500.42" in src.content
        assert "+2.35%" in src.content
        assert "coingecko.com" in src.url

    def test_format_negative_change(self) -> None:
        c = _make_connector()
        data = {
            "usd": 100.0,
            "usd_24h_change": -5.2,
            "usd_market_cap": 1_000_000,
            "usd_24h_vol": 500_000,
        }
        src = c._format_price("solana", data)
        assert "-5.20%" in src.content
        assert "Solana" in src.title
