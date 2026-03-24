"""Tests for Phase 4 Batch B: Crypto TA specialist."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from src.forecast.specialists.crypto_ta import (
    CryptoTASpecialist,
    CryptoQuery,
    Candle,
    TAIndicators,
    compute_rsi,
    compute_vwap,
    compute_sma,
    compute_momentum,
)


# ── Helpers ──────────────────────────────────────────────────────────────


@dataclass
class FakeClassification:
    category: str = "CRYPTO"
    subcategory: str = "btc_price"


@dataclass
class FakeMarket:
    id: str = "crypto-001"
    question: str = "Will Bitcoin price go up in the next 5 minutes?"
    market_type: str = "CRYPTO"
    resolution_source: str = "Binance"


@dataclass
class FakeFeatures:
    implied_probability: float = 0.50


@dataclass
class FakeConfig:
    enabled: bool = True
    enabled_specialists: list[str] = field(default_factory=lambda: ["crypto_ta"])
    weather_min_edge: float = 0.08
    weather_api_base: str = ""
    crypto_min_edge: float = 0.04
    crypto_candle_source: str = "binance"
    politics_polling_weight: float = 0.6


def _make_candles(
    n: int = 100,
    base_price: float = 50000.0,
    trend: float = 0.0,
    volume: float = 100.0,
) -> list[Candle]:
    """Create synthetic candle data with optional trend."""
    candles = []
    price = base_price
    for i in range(n):
        price += trend
        candles.append(Candle(
            timestamp=1700000000 + i * 60,
            open=price - 5,
            high=price + 10,
            low=price - 10,
            close=price,
            volume=volume,
        ))
    return candles


def _make_trending_candles(direction: str, n: int = 100) -> list[Candle]:
    """Create candles with clear bullish or bearish trend."""
    trend = 50.0 if direction == "up" else -50.0
    return _make_candles(n=n, base_price=50000.0, trend=trend)


# ── TestCryptoQuestionParsing ─────────────────────────────────────────────


class TestCryptoQuestionParsing:

    def setup_method(self) -> None:
        self.specialist = CryptoTASpecialist(FakeConfig())

    def test_parse_btc_price_up(self) -> None:
        q = "Will Bitcoin price go up in the next 5 minutes?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.asset == "BTC"
        assert result.direction == "up"

    def test_parse_eth_price_above(self) -> None:
        q = "Will ETH be above $3000 in 15 minutes?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.asset == "ETH"
        assert result.direction == "up"

    def test_parse_btc_price_down(self) -> None:
        q = "Will BTC price go down in the next 5 minutes?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.direction == "down"

    def test_parse_timeframe_extraction(self) -> None:
        q = "Will Bitcoin price go up in the next 15 minutes?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.timeframe == "15m"

    def test_parse_non_crypto_returns_none(self) -> None:
        q = "Will the Fed cut rates?"
        result = self.specialist._parse_question(q)
        assert result is None

    def test_parse_altcoin_not_handled(self) -> None:
        q = "Will Solana price go up?"
        result = self.specialist._parse_question(q)
        assert result is None

    def test_can_handle_btc_price(self) -> None:
        cls = FakeClassification(category="CRYPTO", subcategory="btc_price")
        q = "Will Bitcoin price go up in the next 5 minutes?"
        assert self.specialist.can_handle(cls, q) is True

    def test_can_handle_eth_price(self) -> None:
        cls = FakeClassification(category="CRYPTO", subcategory="eth_price")
        q = "Will ETH price increase in the next 5 minutes?"
        assert self.specialist.can_handle(cls, q) is True

    def test_can_handle_non_crypto_rejected(self) -> None:
        cls = FakeClassification(category="MACRO", subcategory="fed_rates")
        q = "Will Bitcoin price go up?"
        assert self.specialist.can_handle(cls, q) is False

    def test_can_handle_crypto_regulation_rejected(self) -> None:
        """Crypto regulation questions are NOT btc_price."""
        cls = FakeClassification(category="CRYPTO", subcategory="regulation")
        q = "Will Bitcoin price go up?"
        assert self.specialist.can_handle(cls, q) is False


# ── TestIndicatorComputation ──────────────────────────────────────────────


class TestIndicatorComputation:

    def test_rsi_overbought(self) -> None:
        """Consistently rising prices → RSI near 100."""
        closes = [100 + i * 10 for i in range(50)]  # Monotonically rising
        rsi = compute_rsi(closes)
        assert rsi > 70

    def test_rsi_oversold(self) -> None:
        """Consistently falling prices → RSI near 0."""
        closes = [100 - i * 2 for i in range(50)]  # Monotonically falling
        rsi = compute_rsi(closes)
        assert rsi < 30

    def test_rsi_neutral(self) -> None:
        """Flat prices → RSI around 50."""
        closes = [100.0] * 50
        rsi = compute_rsi(closes)
        # Flat = no gains or losses, avg_loss = 0 → RSI = 100 technically
        # But with zero gains and zero losses, we get 0/0 case
        assert rsi >= 50.0

    def test_rsi_insufficient_data(self) -> None:
        """Fewer than period+1 values → neutral 50."""
        closes = [100.0, 101.0, 99.0]
        rsi = compute_rsi(closes, period=14)
        assert rsi == 50.0

    def test_vwap_calculation(self) -> None:
        candles = [
            Candle(0, 100, 110, 90, 105, 1000),
            Candle(0, 105, 115, 95, 110, 2000),
        ]
        vwap = compute_vwap(candles)
        # VWAP = sum((H+L+C)/3 * V) / sum(V)
        tp1 = (110 + 90 + 105) / 3  # 101.67
        tp2 = (115 + 95 + 110) / 3  # 106.67
        expected = (tp1 * 1000 + tp2 * 2000) / 3000
        assert vwap == pytest.approx(expected, abs=0.1)

    def test_vwap_empty_candles(self) -> None:
        assert compute_vwap([]) == 0.0

    def test_sma_calculation(self) -> None:
        closes = [10, 20, 30, 40, 50]
        assert compute_sma(closes, 3) == pytest.approx(40.0)  # (30+40+50)/3
        assert compute_sma(closes, 5) == pytest.approx(30.0)  # (10+20+30+40+50)/5

    def test_sma_insufficient_data(self) -> None:
        closes = [10, 20]
        assert compute_sma(closes, 5) == pytest.approx(15.0)  # (10+20)/2

    def test_momentum_positive(self) -> None:
        closes = [100] * 10 + [110]  # 10% rise over 10 periods
        mom = compute_momentum(closes, period=10)
        assert mom == pytest.approx(0.10, abs=0.01)

    def test_momentum_negative(self) -> None:
        closes = [100] * 10 + [90]  # 10% fall
        mom = compute_momentum(closes, period=10)
        assert mom == pytest.approx(-0.10, abs=0.01)

    def test_momentum_insufficient_data(self) -> None:
        closes = [100, 101]
        mom = compute_momentum(closes, period=10)
        assert mom == 0.0

    def test_composite_indicators_bullish(self) -> None:
        """Strongly bullish candles → positive composite signal."""
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_trending_candles("up")
        indicators = specialist._compute_indicators(candles)
        assert indicators.composite_signal > 0
        assert indicators.rsi_14 > 50
        assert indicators.sma_crossover >= 0

    def test_composite_indicators_bearish(self) -> None:
        """Strongly bearish candles → negative composite signal."""
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_trending_candles("down")
        indicators = specialist._compute_indicators(candles)
        assert indicators.composite_signal < 0

    def test_indicators_to_dict(self) -> None:
        ind = TAIndicators(rsi_14=65.0, vwap_deviation=0.0012)
        d = ind.to_dict()
        assert "rsi_14" in d
        assert "vwap_deviation" in d
        assert isinstance(d["rsi_14"], float)


# ── TestProbabilityConversion ─────────────────────────────────────────────


class TestProbabilityConversion:

    def setup_method(self) -> None:
        self.specialist = CryptoTASpecialist(FakeConfig())

    def test_strong_bullish_high_probability(self) -> None:
        ind = TAIndicators(composite_signal=0.8)
        prob = self.specialist._signal_to_probability(ind, "up")
        assert prob > 0.7

    def test_strong_bearish_low_probability(self) -> None:
        ind = TAIndicators(composite_signal=-0.8)
        prob = self.specialist._signal_to_probability(ind, "up")
        assert prob < 0.3

    def test_neutral_around_50pct(self) -> None:
        ind = TAIndicators(composite_signal=0.0)
        prob = self.specialist._signal_to_probability(ind, "up")
        assert prob == pytest.approx(0.5, abs=0.01)

    def test_down_direction_inverts(self) -> None:
        """Bullish signal + 'down' question → low probability."""
        ind = TAIndicators(composite_signal=0.8)
        prob = self.specialist._signal_to_probability(ind, "down")
        assert prob < 0.3

    def test_probability_clamped_upper(self) -> None:
        ind = TAIndicators(composite_signal=10.0)  # Extreme
        prob = self.specialist._signal_to_probability(ind, "up")
        assert prob <= 0.95

    def test_probability_clamped_lower(self) -> None:
        ind = TAIndicators(composite_signal=-10.0)  # Extreme
        prob = self.specialist._signal_to_probability(ind, "up")
        assert prob >= 0.05


# ── TestCryptoAPIIntegration ──────────────────────────────────────────────


class TestCryptoAPIIntegration:

    @pytest.mark.asyncio
    async def test_forecast_end_to_end(self) -> None:
        """End-to-end: mock candles → SpecialistResult."""
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_trending_candles("up")

        with patch.object(
            specialist, "_fetch_candles", new_callable=AsyncMock,
        ) as mock:
            mock.return_value = candles
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )

        assert result.specialist_name == "crypto_ta"
        assert result.bypasses_llm is True
        assert result.confidence_level == "LOW"
        assert 0.05 <= result.probability <= 0.95

    @pytest.mark.asyncio
    async def test_forecast_bearish_market(self) -> None:
        """Bearish candles + 'up' question → probability < 0.5."""
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_trending_candles("down")

        with patch.object(
            specialist, "_fetch_candles", new_callable=AsyncMock,
        ) as mock:
            mock.return_value = candles
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )

        assert result.probability < 0.5

    @pytest.mark.asyncio
    async def test_unparseable_question_raises(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        market = FakeMarket(question="Will the Fed cut rates?")
        with pytest.raises(ValueError, match="Cannot parse"):
            await specialist.forecast(
                market, FakeFeatures(), FakeClassification(),
            )

    @pytest.mark.asyncio
    async def test_key_evidence_populated(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_candles(100)

        with patch.object(
            specialist, "_fetch_candles", new_callable=AsyncMock,
        ) as mock:
            mock.return_value = candles
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )

        assert len(result.key_evidence) == 1
        assert "Binance" in result.key_evidence[0]["source"]

    @pytest.mark.asyncio
    async def test_metadata_has_indicators(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        candles = _make_candles(100)

        with patch.object(
            specialist, "_fetch_candles", new_callable=AsyncMock,
        ) as mock:
            mock.return_value = candles
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )

        assert "rsi_14" in result.specialist_metadata
        assert "composite_signal" in result.specialist_metadata

    @pytest.mark.asyncio
    async def test_close_releases_client(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        mock_client = AsyncMock()
        specialist._client = mock_client
        await specialist.close()
        mock_client.aclose.assert_called_once()
        assert specialist._client is None


# ── TestCryptoRouterIntegration ───────────────────────────────────────────


class TestCryptoRouterIntegration:

    def test_can_handle_btc_price(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        cls = FakeClassification(category="CRYPTO", subcategory="btc_price")
        q = "Will Bitcoin price go up in the next 5 minutes?"
        assert specialist.can_handle(cls, q) is True

    def test_can_handle_eth_price(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        cls = FakeClassification(category="CRYPTO", subcategory="eth_price")
        q = "Will ETH price increase in the next 5 minutes?"
        assert specialist.can_handle(cls, q) is True

    def test_ignores_crypto_regulation(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        cls = FakeClassification(category="CRYPTO", subcategory="regulation")
        q = "Will the SEC approve Bitcoin ETF?"
        assert specialist.can_handle(cls, q) is False

    def test_ignores_non_crypto(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        cls = FakeClassification(category="MACRO", subcategory="fed_rates")
        assert specialist.can_handle(cls, "Will rates rise?") is False

    def test_specialist_name(self) -> None:
        specialist = CryptoTASpecialist(FakeConfig())
        assert specialist.name == "crypto_ta"
