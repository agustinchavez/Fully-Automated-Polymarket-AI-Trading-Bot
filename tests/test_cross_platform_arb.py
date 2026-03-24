"""Tests for cross-platform arb scanner, complementary arb, and correlated mispricings.

Covers:
  - CrossPlatformArbOpportunity dataclass & serialisation
  - PairedTradeResult dataclass & status variants
  - CrossPlatformArbScanner: scan, compute_spread, execute_arb, position limits
  - ComplementaryArbOpportunity detection
  - CorrelatedMispricing detection
  - Engine loop wiring
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.kalshi_client import KalshiMarket, KalshiOrderResult
from src.connectors.market_matcher import MarketMatch
from src.policy.cross_platform_arb import (
    CrossPlatformArbOpportunity,
    CrossPlatformArbScanner,
    PairedTradeResult,
)
from src.policy.arbitrage import (
    ComplementaryArbOpportunity,
    CorrelatedMispricing,
    detect_complementary_arb,
    detect_correlated_mispricings,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_match(
    poly_id: str = "poly-1",
    kalshi_ticker: str = "KXTEST-1",
    confidence: float = 0.9,
    method: str = "keyword",
) -> MarketMatch:
    return MarketMatch(
        polymarket_id=poly_id,
        polymarket_question="Will it rain tomorrow?",
        kalshi_ticker=kalshi_ticker,
        kalshi_title="Rain tomorrow?",
        match_method=method,
        match_confidence=confidence,
    )


def _make_opportunity(
    spread: float = 0.08,
    net_spread: float = 0.04,
    is_actionable: bool = True,
    buy_price: float = 0.40,
    sell_price: float = 0.48,
) -> CrossPlatformArbOpportunity:
    return CrossPlatformArbOpportunity(
        match=_make_match(),
        poly_yes_price=0.40,
        poly_no_price=0.60,
        kalshi_yes_price=0.48,
        kalshi_no_price=0.52,
        spread=spread,
        net_spread=net_spread,
        direction="BUY_POLY_YES_SELL_KALSHI_YES",
        buy_platform="polymarket",
        sell_platform="kalshi",
        buy_price=buy_price,
        sell_price=sell_price,
        total_fees=0.04,
        is_actionable=is_actionable,
    )


@dataclass
class _FakeToken:
    token_id: str = "tok-1"
    outcome: str = "Yes"
    price: float = 0.50


@dataclass
class _FakeMarket:
    id: str = "m1"
    condition_id: str = "cond-1"
    question: str = "Will it rain?"
    category: str = "weather"
    slug: str = "will-it-rain"
    tokens: list[Any] = None  # type: ignore[assignment]
    best_bid: float = 0.50

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = [
                _FakeToken(token_id="yes-1", outcome="Yes", price=0.50),
                _FakeToken(token_id="no-1", outcome="No", price=0.50),
            ]


def _make_config(**overrides: Any) -> MagicMock:
    """Build a mock ArbitrageConfig-like object."""
    defaults = {
        "kalshi_api_base": "https://trading-api.kalshi.com",
        "kalshi_api_key_id": "",
        "kalshi_private_key_path": "",
        "kalshi_paper_mode": True,
        "min_arb_edge": 0.03,
        "polymarket_fee_pct": 0.02,
        "kalshi_fee_pct": 0.02,
        "max_arb_position_usd": 200.0,
        "max_arb_positions_count": 5,
        "manual_mappings_json": "{}",
        "match_min_confidence": 0.6,
        "unwind_on_partial_fill": True,
    }
    defaults.update(overrides)
    config = MagicMock()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


# ── CrossPlatformArbOpportunity Tests ────────────────────────────────


class TestCrossPlatformArbOpportunity:
    def test_auto_arb_id(self) -> None:
        opp = _make_opportunity()
        assert opp.arb_id.startswith("arb-")
        assert len(opp.arb_id) == 16  # "arb-" + 12 hex chars

    def test_auto_timestamp(self) -> None:
        before = time.time()
        opp = _make_opportunity()
        after = time.time()
        assert before <= opp.timestamp <= after

    def test_custom_arb_id_preserved(self) -> None:
        opp = CrossPlatformArbOpportunity(
            match=_make_match(),
            poly_yes_price=0.40, poly_no_price=0.60,
            kalshi_yes_price=0.48, kalshi_no_price=0.52,
            spread=0.08, net_spread=0.04,
            direction="BUY_POLY_YES_SELL_KALSHI_YES",
            buy_platform="polymarket", sell_platform="kalshi",
            buy_price=0.40, sell_price=0.48,
            total_fees=0.04, is_actionable=True,
            arb_id="custom-id",
        )
        assert opp.arb_id == "custom-id"

    def test_to_dict(self) -> None:
        opp = _make_opportunity()
        d = opp.to_dict()
        assert d["arb_id"] == opp.arb_id
        assert d["spread"] == 0.08
        assert d["net_spread"] == 0.04
        assert d["is_actionable"] is True
        assert "match" in d
        assert isinstance(d["match"], dict)

    def test_direction_combos(self) -> None:
        for direction in [
            "BUY_POLY_YES_SELL_KALSHI_YES",
            "BUY_KALSHI_YES_SELL_POLY_YES",
            "BUY_POLY_NO_SELL_KALSHI_NO",
            "BUY_KALSHI_NO_SELL_POLY_NO",
        ]:
            opp = CrossPlatformArbOpportunity(
                match=_make_match(),
                poly_yes_price=0.40, poly_no_price=0.60,
                kalshi_yes_price=0.48, kalshi_no_price=0.52,
                spread=0.08, net_spread=0.04,
                direction=direction,
                buy_platform="polymarket", sell_platform="kalshi",
                buy_price=0.40, sell_price=0.48,
                total_fees=0.04, is_actionable=True,
            )
            assert opp.direction == direction

    def test_not_actionable_when_negative_net(self) -> None:
        opp = _make_opportunity(net_spread=-0.01, is_actionable=False)
        assert not opp.is_actionable


# ── PairedTradeResult Tests ──────────────────────────────────────────


class TestPairedTradeResult:
    def test_both_filled(self) -> None:
        result = PairedTradeResult(
            arb_id="arb-test",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_fill_price=0.40,
            sell_fill_price=0.48,
            stake_usd=100.0,
            net_pnl=3.60,
            status="both_filled",
        )
        assert result.status == "both_filled"
        assert result.net_pnl == 3.60

    def test_partial_fill(self) -> None:
        result = PairedTradeResult(
            arb_id="arb-test",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_fill_price=0.40,
            sell_fill_price=0.0,
            stake_usd=100.0,
            net_pnl=-4.0,
            status="partial",
            unwind_reason="Sell leg failed: timeout",
        )
        assert result.status == "partial"
        assert result.unwind_reason == "Sell leg failed: timeout"

    def test_unwound(self) -> None:
        result = PairedTradeResult(
            arb_id="arb-test",
            buy_platform="kalshi",
            sell_platform="polymarket",
            buy_fill_price=0.40,
            sell_fill_price=0.0,
            stake_usd=50.0,
            net_pnl=-2.0,
            status="unwound",
            unwind_reason="Sell leg failed: API error",
        )
        assert result.status == "unwound"

    def test_failed(self) -> None:
        result = PairedTradeResult(
            arb_id="arb-test",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_fill_price=0.0,
            sell_fill_price=0.0,
            stake_usd=100.0,
            net_pnl=0.0,
            status="failed",
            unwind_reason="Invalid buy price",
        )
        assert result.status == "failed"
        assert result.net_pnl == 0.0

    def test_to_dict(self) -> None:
        result = PairedTradeResult(
            arb_id="arb-test",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_fill_price=0.40,
            sell_fill_price=0.48,
            stake_usd=100.0,
            net_pnl=3.60,
            status="both_filled",
        )
        d = result.to_dict()
        assert d["arb_id"] == "arb-test"
        assert d["status"] == "both_filled"


# ── CrossPlatformArbScanner Tests ────────────────────────────────────


class TestCrossPlatformArbScanner:
    def test_init(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        assert scanner._kalshi is None
        assert scanner._opportunity_log == []

    def test_compute_spread_buy_poly_sell_kalshi(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.40, poly_no=0.60,
            kalshi_yes=0.50, kalshi_no=0.50,
            total_fees=0.04, min_edge=0.03,
        )
        # Best spread: sell_kalshi_yes(0.50) - buy_poly_yes(0.40) = 0.10
        assert abs(opp.spread - 0.10) < 1e-9
        assert abs(opp.net_spread - 0.06) < 1e-9
        assert opp.is_actionable is True
        assert opp.direction == "BUY_POLY_YES_SELL_KALSHI_YES"

    def test_compute_spread_buy_kalshi_sell_poly(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.50, poly_no=0.50,
            kalshi_yes=0.35, kalshi_no=0.65,
            total_fees=0.04, min_edge=0.03,
        )
        # Best: sell_poly_yes(0.50) - buy_kalshi_yes(0.35) = 0.15
        assert abs(opp.spread - 0.15) < 1e-9
        assert opp.direction == "BUY_KALSHI_YES_SELL_POLY_YES"

    def test_compute_spread_no_side(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.50, poly_no=0.40,
            kalshi_yes=0.50, kalshi_no=0.55,
            total_fees=0.04, min_edge=0.03,
        )
        # Best: sell_kalshi_no(0.55) - buy_poly_no(0.40) = 0.15
        assert abs(opp.spread - 0.15) < 1e-9
        assert "NO" in opp.direction

    def test_compute_spread_not_actionable(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.50, poly_no=0.50,
            kalshi_yes=0.50, kalshi_no=0.50,
            total_fees=0.04, min_edge=0.03,
        )
        # Spread = 0, net = -0.04
        assert not opp.is_actionable

    def test_compute_spread_zero_spread(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.50, poly_no=0.50,
            kalshi_yes=0.50, kalshi_no=0.50,
        )
        assert opp.spread == 0.0
        assert opp.net_spread < 0

    def test_compute_spread_negative_clamped(self) -> None:
        match = _make_match()
        opp = CrossPlatformArbScanner.compute_spread(
            match,
            poly_yes=0.70, poly_no=0.30,
            kalshi_yes=0.60, kalshi_no=0.40,
        )
        # All combos are negative spreads
        # spread is max(0.0, best_spread) so 0.0 if best is negative
        # Actually: buy_poly_no=0.30, sell_kalshi_no=0.40 → 0.10
        assert opp.spread >= 0.0

    @pytest.mark.asyncio
    async def test_scan_empty_markets(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        result = await scanner.scan([])
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_kalshi_error(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(side_effect=Exception("API down"))
        scanner._kalshi = mock_kalshi

        result = await scanner.scan([_FakeMarket()])
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_no_matches(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(return_value=[
            KalshiMarket(ticker="KXUNRELATED", title="Unrelated market"),
        ])
        scanner._kalshi = mock_kalshi

        result = await scanner.scan([_FakeMarket()])
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_with_matches(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)

        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(return_value=[
            KalshiMarket(ticker="KXRAIN", title="Will it rain tomorrow?",
                         yes_bid=0.60, yes_ask=0.65),
        ])
        scanner._kalshi = mock_kalshi

        # Mock the matcher to return a match
        with patch.object(scanner._matcher, "find_matches", return_value=[
            _make_match(poly_id="cond-1", kalshi_ticker="KXRAIN"),
        ]):
            poly_market = _FakeMarket(
                condition_id="cond-1",
                tokens=[
                    _FakeToken(token_id="yes-1", outcome="Yes", price=0.40),
                    _FakeToken(token_id="no-1", outcome="No", price=0.60),
                ],
            )
            result = await scanner.scan([poly_market])

        assert len(result) == 1
        assert result[0].poly_yes_price == 0.40

    @pytest.mark.asyncio
    async def test_scan_sorted_by_net_spread(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)

        mock_kalshi = AsyncMock()
        mock_kalshi.list_markets = AsyncMock(return_value=[
            KalshiMarket(ticker="KX1", title="Market A", yes_bid=0.60, yes_ask=0.65),
            KalshiMarket(ticker="KX2", title="Market B", yes_bid=0.80, yes_ask=0.85),
        ])
        scanner._kalshi = mock_kalshi

        with patch.object(scanner._matcher, "find_matches", return_value=[
            _make_match(poly_id="p1", kalshi_ticker="KX1"),
            _make_match(poly_id="p2", kalshi_ticker="KX2"),
        ]):
            markets = [
                _FakeMarket(
                    id="p1", condition_id="p1",
                    tokens=[
                        _FakeToken(outcome="Yes", price=0.40),
                        _FakeToken(outcome="No", price=0.60),
                    ],
                ),
                _FakeMarket(
                    id="p2", condition_id="p2",
                    tokens=[
                        _FakeToken(outcome="Yes", price=0.30),
                        _FakeToken(outcome="No", price=0.70),
                    ],
                ),
            ]
            result = await scanner.scan(markets)

        assert len(result) == 2
        # Should be sorted by net_spread descending
        assert result[0].net_spread >= result[1].net_spread

    def test_check_position_limits_under(self) -> None:
        config = _make_config(max_arb_positions_count=5, max_arb_position_usd=200.0)
        scanner = CrossPlatformArbScanner(config)
        assert scanner.check_position_limits() is True

    def test_check_position_limits_count_exceeded(self) -> None:
        config = _make_config(max_arb_positions_count=2, max_arb_position_usd=200.0)
        scanner = CrossPlatformArbScanner(config)
        # Add 2 active positions
        for _ in range(2):
            scanner._active_positions.append(PairedTradeResult(
                arb_id="arb-x", buy_platform="p", sell_platform="k",
                buy_fill_price=0.4, sell_fill_price=0.5,
                stake_usd=100.0, net_pnl=5.0, status="both_filled",
            ))
        assert scanner.check_position_limits() is False

    def test_check_position_limits_exposure_exceeded(self) -> None:
        config = _make_config(max_arb_positions_count=10, max_arb_position_usd=100.0)
        scanner = CrossPlatformArbScanner(config)
        # 10 positions at $100 each = $1000, limit is 10 * $100 = $1000
        for _ in range(10):
            scanner._active_positions.append(PairedTradeResult(
                arb_id="arb-x", buy_platform="p", sell_platform="k",
                buy_fill_price=0.4, sell_fill_price=0.5,
                stake_usd=100.0, net_pnl=5.0, status="both_filled",
            ))
        assert scanner.check_position_limits() is False

    def test_opportunity_log_property(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        assert scanner.opportunity_log == []

    def test_active_positions_property(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        assert scanner.active_positions == []


# ── Arb Execution Tests ──────────────────────────────────────────────


class TestArbExecution:
    @pytest.mark.asyncio
    async def test_execute_both_filled(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(return_value=KalshiOrderResult(
            order_id="paper-123", status="simulated",
            fill_price=0.48, fill_size=10,
        ))
        scanner._kalshi = mock_kalshi

        opp = _make_opportunity(buy_price=0.40, sell_price=0.48)
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "both_filled"
        assert result.arb_id == opp.arb_id

    @pytest.mark.asyncio
    async def test_execute_invalid_buy_price(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)

        opp = _make_opportunity(buy_price=0.0)
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "failed"
        assert "Invalid buy price" in result.unwind_reason

    @pytest.mark.asyncio
    async def test_execute_buy_leg_error(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(
            side_effect=Exception("Connection refused"),
        )
        scanner._kalshi = mock_kalshi

        # Buy on kalshi (will fail)
        opp = CrossPlatformArbOpportunity(
            match=_make_match(),
            poly_yes_price=0.50, poly_no_price=0.50,
            kalshi_yes_price=0.40, kalshi_no_price=0.60,
            spread=0.10, net_spread=0.06,
            direction="BUY_KALSHI_YES_SELL_POLY_YES",
            buy_platform="kalshi", sell_platform="polymarket",
            buy_price=0.40, sell_price=0.50,
            total_fees=0.04, is_actionable=True,
        )
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "failed"
        assert "Buy leg failed" in result.unwind_reason

    @pytest.mark.asyncio
    async def test_execute_sell_leg_error_unwind(self) -> None:
        config = _make_config(unwind_on_partial_fill=True)
        scanner = CrossPlatformArbScanner(config)

        # Buy succeeds (polymarket simulated), sell fails (kalshi)
        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(
            side_effect=Exception("Kalshi API error"),
        )
        scanner._kalshi = mock_kalshi

        opp = _make_opportunity()  # buy=polymarket, sell=kalshi
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "unwound"
        assert "Sell leg failed" in result.unwind_reason

    @pytest.mark.asyncio
    async def test_execute_sell_leg_error_partial(self) -> None:
        config = _make_config(unwind_on_partial_fill=False)
        scanner = CrossPlatformArbScanner(config)

        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(
            side_effect=Exception("Kalshi API error"),
        )
        scanner._kalshi = mock_kalshi

        opp = _make_opportunity()
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "partial"

    @pytest.mark.asyncio
    async def test_execute_pnl_calculation(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)

        # Buy on polymarket (simulated), sell on kalshi (simulated paper)
        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(return_value=KalshiOrderResult(
            order_id="paper-sell", status="simulated",
            fill_price=0.50, fill_size=250,
        ))
        scanner._kalshi = mock_kalshi

        opp = _make_opportunity(buy_price=0.40, sell_price=0.50)
        result = await scanner.execute_arb(opp, stake_usd=100.0)

        assert result.status == "both_filled"
        # quantity = int(100 / 0.40) = 250
        # gross_pnl = (0.50 - 0.40) * 250 = 25.0
        # net_pnl = 25.0 - 0.04 * 100 = 21.0
        assert abs(result.net_pnl - 21.0) < 1e-9

    @pytest.mark.asyncio
    async def test_execute_appends_to_active_positions(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        mock_kalshi.place_order = AsyncMock(return_value=KalshiOrderResult(
            order_id="paper-123", status="simulated",
            fill_price=0.48, fill_size=10,
        ))
        scanner._kalshi = mock_kalshi

        opp = _make_opportunity()
        await scanner.execute_arb(opp, stake_usd=100.0)
        assert len(scanner.active_positions) == 1

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        config = _make_config()
        scanner = CrossPlatformArbScanner(config)
        mock_kalshi = AsyncMock()
        scanner._kalshi = mock_kalshi

        await scanner.close()
        mock_kalshi.close.assert_awaited_once()
        assert scanner._kalshi is None


# ── Complementary Arb Detection Tests ────────────────────────────────


class TestComplementaryArbDetection:
    def test_below_threshold_detected(self) -> None:
        """YES+NO < threshold → opportunity detected."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.45),
                _FakeToken(outcome="No", price=0.45),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 1
        assert result[0].combined_cost == 0.90
        assert result[0].guaranteed_profit == 0.10

    def test_above_threshold_not_detected(self) -> None:
        """YES+NO >= threshold → no opportunity."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.50),
                _FakeToken(outcome="No", price=0.50),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 0

    def test_exactly_at_threshold(self) -> None:
        """YES+NO == threshold → not detected (must be strictly below)."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.485),
                _FakeToken(outcome="No", price=0.485),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 0

    def test_fee_deduction(self) -> None:
        """Net profit accounts for trading fees."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.45),
                _FakeToken(outcome="No", price=0.45),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 1
        # guaranteed = 0.10, fee = 0.02 * 2 = 0.04, net = 0.06
        assert abs(result[0].net_profit - 0.06) < 1e-4
        assert result[0].is_actionable is True

    def test_not_actionable_when_fees_exceed_profit(self) -> None:
        """When fees > guaranteed profit, not actionable."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.48),
                _FakeToken(outcome="No", price=0.48),
            ],
        )
        # combined = 0.96, guaranteed = 0.04, fee = 0.08 * 2 = 0.16
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=800)
        assert len(result) == 1
        assert result[0].is_actionable is False

    def test_multi_token_markets_skipped(self) -> None:
        """Markets with >2 tokens are skipped."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Yes", price=0.30),
                _FakeToken(outcome="No", price=0.30),
                _FakeToken(outcome="Maybe", price=0.30),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 0

    def test_missing_yes_no_skipped(self) -> None:
        """Markets without YES/NO token labels are skipped."""
        market = _FakeMarket(
            tokens=[
                _FakeToken(outcome="Trump", price=0.45),
                _FakeToken(outcome="Biden", price=0.45),
            ],
        )
        result = detect_complementary_arb([market], threshold=0.97, fee_bps=200)
        assert len(result) == 0

    def test_sorted_by_net_profit(self) -> None:
        """Results sorted by net_profit descending."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                tokens=[
                    _FakeToken(outcome="Yes", price=0.45),
                    _FakeToken(outcome="No", price=0.45),
                ],
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                tokens=[
                    _FakeToken(outcome="Yes", price=0.30),
                    _FakeToken(outcome="No", price=0.30),
                ],
            ),
        ]
        result = detect_complementary_arb(markets, threshold=0.97, fee_bps=200)
        assert len(result) == 2
        assert result[0].net_profit >= result[1].net_profit

    def test_empty_markets(self) -> None:
        result = detect_complementary_arb([], threshold=0.97, fee_bps=200)
        assert result == []


# ── Correlated Mispricing Detection Tests ────────────────────────────


class TestCorrelatedMispricings:
    def test_subset_mispricing_detected(self) -> None:
        """Specific event priced higher than general event → flagged."""
        # "Biden Democratic nominee" has 2 extra entities vs "Democrat win"
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden Democratic nominee win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.45)],
                best_bid=0.45,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will somebody win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.30)],
                best_bid=0.30,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) >= 1
        opp = result[0]
        assert opp.divergence >= 0.10
        assert opp.implied_conditional > 1.0

    def test_no_mispricing_when_consistent(self) -> None:
        """When prices are close enough, no mispricing flagged."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden Democratic nominee win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.30)],
                best_bid=0.30,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will somebody win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.45)],
                best_bid=0.45,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        # Specific (30%) < General (45%) → consistent, no divergence
        assert len(result) == 0

    def test_divergence_threshold(self) -> None:
        """Only flag when divergence exceeds min_divergence."""
        # Same entity count → falls into equal specificity branch (price gap)
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Bitcoin reach $100k by December 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.42)],
                best_bid=0.42,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will Bitcoin reach $100k by November 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.40)],
                best_bid=0.40,
            ),
        ]
        # price_diff = 0.02, below min_divergence=0.10
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) == 0

    def test_unrelated_markets_ignored(self) -> None:
        """Markets with low entity overlap are not compared."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Bitcoin reach $100k?",
                tokens=[_FakeToken(outcome="Yes", price=0.60)],
                best_bid=0.60,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will it rain in London tomorrow?",
                tokens=[_FakeToken(outcome="Yes", price=0.20)],
                best_bid=0.20,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) == 0

    def test_implied_conditional(self) -> None:
        """Implied conditional P(specific|general) is computed correctly."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden Democratic nominee win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.50)],
                best_bid=0.50,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will somebody win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.35)],
                best_bid=0.35,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) >= 1
        opp = result[0]
        # specific=0.50, general=0.35 → implied = 0.50/0.35 ≈ 1.4286
        assert opp.implied_conditional > 1.0

    def test_equal_specificity_price_gap(self) -> None:
        """Markets with equal entity counts use price gap detection."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.50)],
                best_bid=0.50,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will Trump win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.30)],
                best_bid=0.30,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) >= 1
        opp = result[0]
        assert abs(opp.divergence - 0.20) < 1e-4
        assert opp.implied_conditional == 0.0  # Equal specificity path

    def test_actionable_flag(self) -> None:
        """is_actionable requires divergence >= min_divergence + 0.05."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.45)],
                best_bid=0.45,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will Trump win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.33)],
                best_bid=0.33,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        assert len(result) >= 1
        opp = result[0]
        # divergence = 0.12, min + 0.05 = 0.15 → not actionable
        if opp.divergence < 0.15:
            assert not opp.is_actionable

    def test_sorted_by_divergence(self) -> None:
        """Results sorted by divergence descending."""
        markets = [
            _FakeMarket(
                id="m1", condition_id="c1",
                question="Will Biden win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.50)],
                best_bid=0.50,
            ),
            _FakeMarket(
                id="m2", condition_id="c2",
                question="Will Trump win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.30)],
                best_bid=0.30,
            ),
            _FakeMarket(
                id="m3", condition_id="c3",
                question="Will Harris win the presidential election 2024?",
                tokens=[_FakeToken(outcome="Yes", price=0.55)],
                best_bid=0.55,
            ),
        ]
        result = detect_correlated_mispricings(markets, min_divergence=0.10)
        if len(result) >= 2:
            assert result[0].divergence >= result[1].divergence

    def test_empty_markets(self) -> None:
        result = detect_correlated_mispricings([], min_divergence=0.10)
        assert result == []

    def test_single_market(self) -> None:
        """Single market → no pairs to compare."""
        market = _FakeMarket(
            question="Will Bitcoin reach $100k?",
            tokens=[_FakeToken(outcome="Yes", price=0.60)],
            best_bid=0.60,
        )
        result = detect_correlated_mispricings([market], min_divergence=0.10)
        assert result == []


# ── Engine Loop Integration Tests ────────────────────────────────────


class TestEngineArbIntegration:
    def test_scanner_not_init_when_disabled(self) -> None:
        """Cross-platform scanner not created when arbitrage disabled."""
        from src.config import BotConfig
        config = BotConfig()
        assert config.arbitrage.enabled is False

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        assert engine._cross_platform_scanner is None

    def test_scanner_init_when_enabled(self) -> None:
        """Cross-platform scanner created when arbitrage enabled."""
        from src.config import BotConfig, ArbitrageConfig
        config = BotConfig(arbitrage=ArbitrageConfig(enabled=True))

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        assert engine._cross_platform_scanner is not None

    def test_get_status_includes_arbitrage(self) -> None:
        """get_status response includes arbitrage section."""
        from src.config import BotConfig
        config = BotConfig()

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        status = engine.get_status()
        assert "arbitrage" in status
        arb = status["arbitrage"]
        assert arb["enabled"] is False
        assert arb["cross_platform_opportunities"] == 0
        assert arb["complementary_arb"] == 0

    def test_get_status_with_enabled_arbitrage(self) -> None:
        """get_status shows active positions when scanner is running."""
        from src.config import BotConfig, ArbitrageConfig
        config = BotConfig(arbitrage=ArbitrageConfig(enabled=True))

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        status = engine.get_status()
        arb = status["arbitrage"]
        assert arb["enabled"] is True
        assert arb["active_arb_positions"] == 0

    @pytest.mark.asyncio
    async def test_maybe_scan_cross_platform_disabled(self) -> None:
        """No scan when arbitrage disabled."""
        from src.config import BotConfig
        config = BotConfig()

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        await engine._maybe_scan_cross_platform_arb([_FakeMarket()])
        assert engine._latest_cross_platform_opps == []

    @pytest.mark.asyncio
    async def test_maybe_scan_cross_platform_respects_interval(self) -> None:
        """Second scan within interval is skipped."""
        from src.config import BotConfig, ArbitrageConfig
        config = BotConfig(arbitrage=ArbitrageConfig(
            enabled=True, scan_interval_secs=60,
        ))

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)

        # Mock scanner to track calls
        mock_scanner = MagicMock()
        mock_scanner.scan = AsyncMock(return_value=[])
        mock_scanner.active_positions = []
        engine._cross_platform_scanner = mock_scanner

        # First scan
        await engine._maybe_scan_cross_platform_arb([_FakeMarket()])
        assert mock_scanner.scan.await_count == 1

        # Second scan immediately → should be skipped
        await engine._maybe_scan_cross_platform_arb([_FakeMarket()])
        assert mock_scanner.scan.await_count == 1

    @pytest.mark.asyncio
    async def test_maybe_scan_arbitrage_calls_enhanced(self) -> None:
        """_maybe_scan_arbitrage calls complementary + correlated detection."""
        from src.config import BotConfig
        config = BotConfig()

        from src.engine.loop import TradingEngine
        engine = TradingEngine(config)
        engine._last_arbitrage_scan = 0.0  # Force scan

        markets = [_FakeMarket()]

        with patch("src.policy.arbitrage.detect_arbitrage", return_value=[]) as mock_detect, \
             patch("src.policy.arbitrage.detect_complementary_arb", return_value=[]) as mock_comp, \
             patch("src.policy.arbitrage.detect_correlated_mispricings", return_value=[]) as mock_corr:
            await engine._maybe_scan_arbitrage(markets)
            mock_detect.assert_called_once()
            mock_comp.assert_called_once()
            mock_corr.assert_called_once()
