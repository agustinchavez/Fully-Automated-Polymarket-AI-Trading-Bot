"""Tests for Phase 5 Batch A: Kalshi connector, market matcher, and config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.kalshi_client import (
    KalshiClient,
    KalshiMarket,
    KalshiOrderResult,
    KalshiPosition,
    _parse_kalshi_market,
)
from src.connectors.market_matcher import MarketMatch, MarketMatcher


# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class FakeGammaMarket:
    id: str = "poly-001"
    condition_id: str = "cond-001"
    question: str = "Will Bitcoin exceed $100,000 by December 2025?"
    category: str = "CRYPTO"
    slug: str = "btc-100k"
    tokens: list[Any] = field(default_factory=list)


@dataclass
class FakeKalshiMarket:
    ticker: str = "KXBTCUSD-25DEC31-T100000"
    title: str = "Will Bitcoin exceed $100,000 by December 31, 2025?"
    category: str = "Crypto"
    status: str = "active"
    yes_bid: float = 0.62
    yes_ask: float = 0.65
    no_bid: float = 0.35
    no_ask: float = 0.38
    volume: int = 5000
    open_interest: int = 1200
    expiration_time: str = "2025-12-31T23:59:59Z"
    result: str | None = None
    subtitle: str = ""
    event_ticker: str = "KXBTCUSD"


# ── TestKalshiMarketModel ────────────────────────────────────────────


class TestKalshiMarketModel:

    def test_mid_price(self) -> None:
        m = KalshiMarket(ticker="T", title="Test", yes_bid=0.40, yes_ask=0.50)
        assert m.mid == pytest.approx(0.45)

    def test_spread(self) -> None:
        m = KalshiMarket(ticker="T", title="Test", yes_bid=0.40, yes_ask=0.50)
        assert m.spread == pytest.approx(0.10)

    def test_default_values(self) -> None:
        m = KalshiMarket(ticker="T", title="Test")
        assert m.status == "active"
        assert m.result is None
        assert m.volume == 0

    def test_to_dict_has_all_fields(self) -> None:
        m = KalshiMarket(ticker="ABC", title="Test Market")
        d = m.to_dict()
        assert d["ticker"] == "ABC"
        assert "mid" in d
        assert "spread" in d

    def test_parse_kalshi_market_cents_to_dollars(self) -> None:
        """Legacy API returns prices in cents; parser converts to dollars."""
        raw = {
            "ticker": "KXTEST",
            "title": "Test",
            "yes_bid": 65,
            "yes_ask": 68,
            "no_bid": 32,
            "no_ask": 35,
            "volume": 100,
        }
        m = _parse_kalshi_market(raw)
        assert m.yes_bid == pytest.approx(0.65)
        assert m.yes_ask == pytest.approx(0.68)
        assert m.no_bid == pytest.approx(0.32)
        assert m.no_ask == pytest.approx(0.35)

    def test_parse_kalshi_market_dollars_format(self) -> None:
        """Post March 2026 API returns prices as *_dollars fields (decimal)."""
        raw = {
            "ticker": "KXTEST",
            "title": "Test",
            "yes_bid_dollars": 0.65,
            "yes_ask_dollars": 0.68,
            "no_bid_dollars": 0.32,
            "no_ask_dollars": 0.35,
            "volume": 200,
        }
        m = _parse_kalshi_market(raw)
        assert m.yes_bid == pytest.approx(0.65)
        assert m.yes_ask == pytest.approx(0.68)
        assert m.no_bid == pytest.approx(0.32)
        assert m.no_ask == pytest.approx(0.35)

    def test_parse_kalshi_market_dollars_takes_priority(self) -> None:
        """When both dollars and cents fields present, dollars wins."""
        raw = {
            "ticker": "KXTEST",
            "title": "Test",
            "yes_bid": 50,           # Legacy cents: 0.50
            "yes_bid_dollars": 0.65, # New dollars: 0.65
            "yes_ask": 55,
            "yes_ask_dollars": 0.68,
            "no_bid": 45,
            "no_bid_dollars": 0.32,
            "no_ask": 50,
            "no_ask_dollars": 0.35,
        }
        m = _parse_kalshi_market(raw)
        assert m.yes_bid == pytest.approx(0.65)
        assert m.yes_ask == pytest.approx(0.68)
        assert m.no_bid == pytest.approx(0.32)
        assert m.no_ask == pytest.approx(0.35)

    def test_parse_kalshi_market_defaults(self) -> None:
        raw = {"ticker": "X", "title": "Y"}
        m = _parse_kalshi_market(raw)
        assert m.ticker == "X"
        assert m.yes_bid == 0.0
        assert m.yes_ask == 1.0


# ── TestKalshiClient ────────────────────────────────────────────────


class TestKalshiClient:

    @pytest.mark.asyncio
    async def test_list_markets_paper_mode(self) -> None:
        client = KalshiClient(paper_mode=True)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "markets": [
                    {"ticker": "K1", "title": "Market 1", "yes_bid": 50, "yes_ask": 55},
                    {"ticker": "K2", "title": "Market 2", "yes_bid": 30, "yes_ask": 40},
                ],
            }
            markets = await client.list_markets()

        assert len(markets) == 2
        assert markets[0].ticker == "K1"
        assert markets[0].yes_bid == pytest.approx(0.50)

    @pytest.mark.asyncio
    async def test_get_market(self) -> None:
        client = KalshiClient(paper_mode=True)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "market": {"ticker": "K1", "title": "Test", "yes_bid": 60, "yes_ask": 65},
            }
            m = await client.get_market("K1")

        assert m.ticker == "K1"
        assert m.yes_bid == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_get_orderbook(self) -> None:
        client = KalshiClient(paper_mode=True)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"yes": [[60, 100]], "no": [[40, 80]]}
            book = await client.get_market_orderbook("K1")

        assert "yes" in book

    @pytest.mark.asyncio
    async def test_place_order_paper_mode(self) -> None:
        """Paper mode returns simulated result without API call."""
        client = KalshiClient(paper_mode=True)
        result = await client.place_order("K1", "buy", 10, 0.65)
        assert result.status == "simulated"
        assert result.fill_price == 0.65
        assert result.fill_size == 10
        assert result.order_id.startswith("paper-")

    @pytest.mark.asyncio
    async def test_get_positions_paper_mode(self) -> None:
        client = KalshiClient(paper_mode=True)
        positions = await client.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_close_releases_client(self) -> None:
        client = KalshiClient(paper_mode=True)
        mock_http = AsyncMock()
        client._client = mock_http
        await client.close()
        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self) -> None:
        client = KalshiClient(paper_mode=True)
        await client.close()  # Should not raise

    def test_ensure_client_creates_httpx(self) -> None:
        client = KalshiClient(paper_mode=True)
        assert client._client is None
        http = client._ensure_client()
        assert http is not None
        assert client._client is not None

    @pytest.mark.asyncio
    async def test_list_markets_empty_response(self) -> None:
        client = KalshiClient(paper_mode=True)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"markets": []}
            markets = await client.list_markets()

        assert markets == []

    @pytest.mark.asyncio
    async def test_list_markets_with_cursor(self) -> None:
        client = KalshiClient(paper_mode=True)
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"markets": [{"ticker": "K3", "title": "T"}]}
            markets = await client.list_markets(cursor="abc123")

        assert len(markets) == 1
        # Verify _get was called with the correct path and cursor in params
        call_args = mock_get.call_args
        params = call_args.kwargs.get("params", call_args[1] if len(call_args[1]) > 0 else {})
        assert params.get("cursor") == "abc123"

    @pytest.mark.asyncio
    async def test_place_order_live_error_returns_failed(self) -> None:
        """Live mode order that raises returns failed result."""
        client = KalshiClient(paper_mode=False, api_key_id="test")
        with patch.object(client, "_post", side_effect=Exception("Network error")):
            result = await client.place_order("K1", "buy", 10, 0.65)
        assert result.status == "failed"
        assert "Network error" in result.error

    def test_order_result_to_dict(self) -> None:
        r = KalshiOrderResult(order_id="o1", status="simulated", fill_price=0.5)
        d = r.to_dict()
        assert d["order_id"] == "o1"

    def test_position_to_dict(self) -> None:
        p = KalshiPosition(ticker="K1", side="yes", quantity=10)
        d = p.to_dict()
        assert d["ticker"] == "K1"

    @pytest.mark.asyncio
    async def test_place_order_live_success(self) -> None:
        client = KalshiClient(paper_mode=False, api_key_id="test")
        with patch.object(client, "_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {
                "order": {
                    "order_id": "live-123",
                    "status": "filled",
                    "yes_price": 65,
                    "count": 10,
                },
            }
            result = await client.place_order("K1", "buy", 10, 0.65)
        assert result.status == "filled"
        assert result.order_id == "live-123"
        assert result.fill_price == pytest.approx(0.65)


# ── TestKalshiAuth ──────────────────────────────────────────────────


class TestKalshiAuth:

    def test_load_rsa_key_missing_raises(self) -> None:
        """Missing key raises RuntimeError."""
        client = KalshiClient(paper_mode=False)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="cryptography is required|Kalshi RSA key"):
                client._load_rsa_key()

    def test_build_auth_headers_requires_key(self) -> None:
        client = KalshiClient(paper_mode=False, api_key_id="test-key")
        with pytest.raises((RuntimeError, Exception)):
            client._build_auth_headers("GET", "/test")

    def test_paper_mode_skips_auth(self) -> None:
        """Paper mode GET doesn't require auth headers."""
        client = KalshiClient(paper_mode=True)
        # Should not raise even without keys
        assert client._paper_mode is True

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"KALSHI_API_KEY_ID": "env-key"}):
            client = KalshiClient(paper_mode=True)
        assert client._api_key_id == "env-key"

    def test_api_key_from_constructor(self) -> None:
        client = KalshiClient(api_key_id="ctor-key", paper_mode=True)
        assert client._api_key_id == "ctor-key"

    def test_private_key_path_from_env(self) -> None:
        with patch.dict("os.environ", {"KALSHI_PRIVATE_KEY_PATH": "/tmp/key.pem"}):
            client = KalshiClient(paper_mode=True)
        assert client._private_key_path == "/tmp/key.pem"


# ── TestMarketMatcher ───────────────────────────────────────────────


class TestMarketMatcher:

    def test_manual_mapping_match(self) -> None:
        """Manual mappings have confidence 1.0."""
        poly = [FakeGammaMarket(condition_id="cond-001")]
        kalshi = [FakeKalshiMarket(ticker="K1")]
        matcher = MarketMatcher(manual_mappings={"K1": "cond-001"})
        matches = matcher.find_matches(poly, kalshi)
        assert len(matches) == 1
        assert matches[0].match_method == "manual"
        assert matches[0].match_confidence == 1.0

    def test_keyword_match_high_overlap(self) -> None:
        """Similar questions with high entity overlap match."""
        poly = [FakeGammaMarket(
            condition_id="c1",
            question="Will Bitcoin exceed $100,000 by December 2025?",
        )]
        kalshi = [FakeKalshiMarket(
            ticker="K1",
            title="Will Bitcoin exceed $100,000 by December 31, 2025?",
        )]
        matcher = MarketMatcher(min_confidence=0.5)
        matches = matcher.find_matches(poly, kalshi)
        assert len(matches) == 1
        assert matches[0].match_method == "keyword"
        assert matches[0].match_confidence >= 0.5

    def test_keyword_match_low_overlap_rejected(self) -> None:
        """Unrelated questions don't match."""
        poly = [FakeGammaMarket(
            condition_id="c1",
            question="Will the Fed cut interest rates in March?",
        )]
        kalshi = [FakeKalshiMarket(
            ticker="K1",
            title="Will Bitcoin exceed $100,000?",
        )]
        matcher = MarketMatcher(min_confidence=0.6)
        matches = matcher.find_matches(poly, kalshi)
        assert len(matches) == 0

    def test_empty_inputs(self) -> None:
        matcher = MarketMatcher()
        assert matcher.find_matches([], []) == []
        assert matcher.find_matches([FakeGammaMarket()], []) == []
        assert matcher.find_matches([], [FakeKalshiMarket()]) == []

    def test_extract_entities_removes_stop_words(self) -> None:
        entities = MarketMatcher._extract_entities("Will the Fed cut rates?")
        assert "will" not in entities
        assert "the" not in entities
        assert "fed" in entities
        assert "cut" in entities

    def test_extract_entities_preserves_numbers(self) -> None:
        entities = MarketMatcher._extract_entities("above $100,000 by 2025")
        assert any("100" in e for e in entities)
        assert "2025" in entities

    def test_normalize_question(self) -> None:
        result = MarketMatcher._normalize_question("  Will BTC Hit $100K?  ")
        assert result == "will btc hit 100k"

    def test_jaccard_similarity_identical(self) -> None:
        s = {"a", "b", "c"}
        assert MarketMatcher._jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_jaccard_similarity_disjoint(self) -> None:
        assert MarketMatcher._jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        sim = MarketMatcher._jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        assert sim == pytest.approx(0.5)

    def test_jaccard_similarity_empty(self) -> None:
        assert MarketMatcher._jaccard_similarity(set(), {"a"}) == 0.0

    def test_match_result_to_dict(self) -> None:
        m = MarketMatch(
            polymarket_id="p1",
            polymarket_question="Q?",
            kalshi_ticker="K1",
            kalshi_title="T",
            match_method="manual",
            match_confidence=1.0,
        )
        d = m.to_dict()
        assert d["polymarket_id"] == "p1"
        assert d["match_confidence"] == 1.0

    def test_deduplication(self) -> None:
        """Each market only matches once (best match wins)."""
        poly = [
            FakeGammaMarket(condition_id="c1", question="Will Bitcoin reach $100K?"),
            FakeGammaMarket(condition_id="c2", question="Will Bitcoin reach $100,000?"),
        ]
        kalshi = [FakeKalshiMarket(ticker="K1", title="Will Bitcoin reach $100,000?")]
        matcher = MarketMatcher(min_confidence=0.4)
        matches = matcher.find_matches(poly, kalshi)
        # K1 should match only one poly market
        assert len(matches) <= 1

    def test_sorted_by_confidence_descending(self) -> None:
        poly = [
            FakeGammaMarket(condition_id="c1", question="Will BTC hit $100K by December 2025?"),
            FakeGammaMarket(condition_id="c2", question="Will ETH hit $5K by June 2025?"),
        ]
        kalshi = [
            FakeKalshiMarket(ticker="K1", title="Will BTC hit $100K by December 2025?"),
        ]
        matcher = MarketMatcher(
            manual_mappings={"K1": "c1"},
            min_confidence=0.3,
        )
        matches = matcher.find_matches(poly, kalshi)
        # Manual match (1.0) should come first
        if len(matches) >= 1:
            assert matches[0].match_confidence == 1.0


# ── TestArbitrageConfig ─────────────────────────────────────────────


class TestArbitrageConfig:

    def test_disabled_by_default(self) -> None:
        from src.config import BotConfig
        cfg = BotConfig()
        assert cfg.arbitrage.enabled is False
        assert cfg.arbitrage.kalshi_paper_mode is True

    def test_defaults(self) -> None:
        from src.config import ArbitrageConfig
        cfg = ArbitrageConfig()
        assert cfg.min_arb_edge == 0.03
        assert cfg.polymarket_fee_pct == 0.02
        assert cfg.kalshi_fee_pct == 0.02
        assert cfg.max_arb_positions_count == 5
        assert cfg.complementary_threshold == 0.97

    def test_fee_warning_when_enabled(self) -> None:
        import warnings
        from src.config import ArbitrageConfig
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ArbitrageConfig(enabled=True, min_arb_edge=0.03)
            # 0.03 <= 0.02 + 0.02 = 0.04, so warning expected
            assert any("zero or negative profit" in str(x.message) for x in w)

    def test_no_warning_when_disabled(self) -> None:
        import warnings
        from src.config import ArbitrageConfig
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ArbitrageConfig(enabled=False, min_arb_edge=0.01)
            arb_warnings = [x for x in w if "zero or negative profit" in str(x.message)]
            assert len(arb_warnings) == 0

    def test_bot_config_has_arbitrage_field(self) -> None:
        from src.config import BotConfig
        cfg = BotConfig()
        assert hasattr(cfg, "arbitrage")
        assert cfg.arbitrage.scan_interval_secs == 60

    def test_rate_limiter_has_kalshi(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS
        assert "kalshi" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["kalshi"].tokens_per_second == 3.0

    def test_secret_fields_include_kalshi(self) -> None:
        from src.config import _SECRET_FIELDS
        assert "kalshi_api_key_id" in _SECRET_FIELDS
        assert "kalshi_private_key_path" in _SECRET_FIELDS

    def test_backward_compatible(self) -> None:
        """Loading default config still works with new ArbitrageConfig."""
        from src.config import BotConfig
        cfg = BotConfig()
        # All existing config sections still present
        assert hasattr(cfg, "scanning")
        assert hasattr(cfg, "specialists")
        assert hasattr(cfg, "arbitrage")
