"""Tests for KalshiPriorConnector — cross-platform consensus prices."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.kalshi_prior import KalshiPriorConnector
from src.research.source_fetcher import FetchedSource


@dataclass
class FakeKalshiMarket:
    ticker: str = "FED-RATE-CUT"
    title: str = "Will the Fed cut rates?"
    yes_bid: float = 0.40
    yes_ask: float = 0.44
    no_bid: float = 0.56
    no_ask: float = 0.60
    volume: int = 1000

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid


@dataclass
class FakeMatch:
    polymarket_id: str = "proxy"
    polymarket_question: str = "Will the Fed cut rates?"
    kalshi_ticker: str = "FED-RATE-CUT"
    kalshi_title: str = "Will the Fed cut rates?"
    match_method: str = "keyword"
    match_confidence: float = 0.75


class TestRelevance:
    def test_relevant_for_macro(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Will the Fed cut rates?", "MACRO")

    def test_relevant_for_election(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Who will win the election?", "ELECTION")

    def test_relevant_for_corporate(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Will Tesla earnings beat?", "CORPORATE")

    def test_relevant_for_science(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Will SpaceX land on Mars?", "SCIENCE")

    def test_relevant_for_crypto(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Will BTC hit 100k?", "CRYPTO")

    def test_relevant_for_technology(self) -> None:
        c = KalshiPriorConnector()
        assert c.is_relevant("Will GPT-5 release?", "TECHNOLOGY")


class TestFetch:
    def _make_connector(
        self,
        kalshi_markets: list | None = None,
        matches: list | None = None,
    ) -> KalshiPriorConnector:
        c = KalshiPriorConnector()
        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(
            return_value=kalshi_markets if kalshi_markets is not None else []
        )
        c._kalshi_client = mock_kalshi
        c._matcher = MagicMock()
        c._matcher.find_matches.return_value = matches if matches is not None else []
        return c

    def test_match_found_returns_fetched_source(self) -> None:
        market = FakeKalshiMarket()
        match = FakeMatch()
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert len(sources) == 1
        src = sources[0]
        assert isinstance(src, FetchedSource)
        assert src.extraction_method == "api"
        assert "consensus_signal" in src.raw
        assert src.raw["consensus_signal"]["platform"] == "kalshi"

    def test_no_match_returns_empty(self) -> None:
        market = FakeKalshiMarket()
        c = self._make_connector(kalshi_markets=[market], matches=[])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert sources == []

    def test_empty_kalshi_markets_returns_empty(self) -> None:
        c = self._make_connector(kalshi_markets=[], matches=[])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert sources == []

    def test_client_error_returns_empty_via_fetch(self) -> None:
        c = KalshiPriorConnector()
        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(
            side_effect=RuntimeError("Kalshi API down")
        )
        c._kalshi_client = mock_kalshi

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            # Use .fetch() to go through base class circuit breaker
            sources = asyncio.run(
                c.fetch("Will the Fed cut rates?", "MACRO")
            )

        assert sources == []

    def test_price_mid_calculation(self) -> None:
        market = FakeKalshiMarket(yes_bid=0.30, yes_ask=0.40)
        match = FakeMatch(kalshi_ticker=market.ticker)
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert len(sources) == 1
        signal = sources[0].raw["consensus_signal"]
        expected_mid = (0.30 + 0.40) / 2  # 0.35
        assert signal["price"] == round(expected_mid, 4)

    def test_spread_pp_is_spread_times_100(self) -> None:
        market = FakeKalshiMarket(yes_bid=0.40, yes_ask=0.44)
        match = FakeMatch(kalshi_ticker=market.ticker)
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert len(sources) == 1
        signal = sources[0].raw["consensus_signal"]
        expected_spread_pp = round((0.44 - 0.40) * 100, 1)  # 4.0
        assert signal["spread_pp"] == expected_spread_pp

    def test_match_confidence_in_raw(self) -> None:
        market = FakeKalshiMarket()
        match = FakeMatch(match_confidence=0.82)
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert len(sources) == 1
        signal = sources[0].raw["consensus_signal"]
        assert signal["match_confidence"] == 0.82

    def test_url_contains_ticker(self) -> None:
        market = FakeKalshiMarket(ticker="MY-TICKER")
        match = FakeMatch(kalshi_ticker="MY-TICKER")
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will it happen?", "MACRO")
            )

        assert len(sources) == 1
        assert "MY-TICKER" in sources[0].url

    def test_authority_score(self) -> None:
        market = FakeKalshiMarket()
        match = FakeMatch()
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert sources[0].authority_score == 0.9

    def test_ticker_not_found_in_markets_returns_empty(self) -> None:
        """Match references a ticker that doesn't exist in kalshi_markets."""
        market = FakeKalshiMarket(ticker="OTHER-TICKER")
        match = FakeMatch(kalshi_ticker="NON-EXISTENT")
        c = self._make_connector(kalshi_markets=[market], matches=[match])

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert sources == []

    def test_list_markets_exception_returns_empty(self) -> None:
        """list_markets raising inside _fetch_impl returns [] (internal catch)."""
        c = KalshiPriorConnector()
        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(
            side_effect=ConnectionError("network error")
        )
        c._kalshi_client = mock_kalshi

        with patch("src.research.connectors.kalshi_prior.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            # _fetch_impl has its own try/except for list_markets
            sources = asyncio.run(
                c._fetch_impl("Will the Fed cut rates?", "MACRO")
            )

        assert sources == []
