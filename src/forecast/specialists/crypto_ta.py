"""Crypto price market specialist — technical analysis-based forecasting.

Fetches real-time 1-minute candles from Binance and computes a composite
signal from RSI(14), VWAP deviation, SMA crossover (9/21), and momentum.
Converts the composite signal to a probability via logistic function.

Completely bypasses the LLM pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.connectors.rate_limiter import rate_limiter
from src.forecast.specialists.base import BaseSpecialist, SpecialistResult
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class Candle:
    """Single candlestick data point."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class CryptoQuery:
    """Parsed crypto question components."""
    asset: str          # "BTC" or "ETH"
    direction: str      # "up" or "down" (the question asks about)
    timeframe: str      # "5m", "15m", "1h", etc.


@dataclass
class TAIndicators:
    """Technical analysis indicator values."""
    rsi_14: float = 50.0
    vwap: float = 0.0
    vwap_deviation: float = 0.0
    sma_9: float = 0.0
    sma_21: float = 0.0
    sma_crossover: float = 0.0   # 1.0 bullish, -1.0 bearish, 0 neutral
    momentum: float = 0.0        # rate of change
    composite_signal: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rsi_14": round(self.rsi_14, 2),
            "vwap_deviation": round(self.vwap_deviation, 4),
            "sma_9": round(self.sma_9, 2),
            "sma_21": round(self.sma_21, 2),
            "sma_crossover": round(self.sma_crossover, 2),
            "momentum": round(self.momentum, 4),
            "composite_signal": round(self.composite_signal, 4),
        }


# Question parsing patterns
_CRYPTO_ASSET_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth)\b",
    re.IGNORECASE,
)

_TIMEFRAME_RE = re.compile(
    r"\b(\d+)\s*(?:-?\s*)?(minutes?|mins?|hours?|hrs?|h|m)\b",
    re.IGNORECASE,
)

_ASSET_MAP = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
}

_BULLISH_PATTERNS = re.compile(
    r"\b(go\s+up|above|higher|increase|rise|up)\b", re.IGNORECASE,
)
_BEARISH_PATTERNS = re.compile(
    r"\b(go\s+down|below|lower|decrease|fall|down|drop)\b", re.IGNORECASE,
)
_PRICE_CONTEXT_RE = re.compile(
    r"\b(price|trading|valued|worth)\b", re.IGNORECASE,
)


def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Compute Relative Strength Index."""
    if len(closes) < period + 1:
        return 50.0  # Neutral when insufficient data

    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))

    # Use simple moving average for initial RSI
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Smooth with exponential-style for remaining data
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_vwap(candles: list[Candle]) -> float:
    """Compute Volume Weighted Average Price."""
    if not candles:
        return 0.0

    total_vol = sum(c.volume for c in candles)
    if total_vol == 0:
        return candles[-1].close

    weighted = sum(
        ((c.high + c.low + c.close) / 3) * c.volume
        for c in candles
    )
    return weighted / total_vol


def compute_sma(closes: list[float], period: int) -> float:
    """Compute Simple Moving Average over the last `period` values."""
    if len(closes) < period:
        return sum(closes) / len(closes) if closes else 0.0
    return sum(closes[-period:]) / period


def compute_momentum(closes: list[float], period: int = 10) -> float:
    """Compute rate of change over `period` bars."""
    if len(closes) < period + 1:
        return 0.0
    old = closes[-(period + 1)]
    if old == 0:
        return 0.0
    return (closes[-1] - old) / old


class CryptoTASpecialist(BaseSpecialist):
    """Technical analysis-based crypto price specialist."""

    def __init__(self, config: Any):
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._candle_source = getattr(config, "crypto_candle_source", "binance")

    @property
    def name(self) -> str:
        return "crypto_ta"

    def can_handle(self, classification: Any, question: str) -> bool:
        cat = getattr(classification, "category", "")
        sub = getattr(classification, "subcategory", "")
        if cat != "CRYPTO":
            return False
        if sub not in ("btc_price", "eth_price"):
            return False
        return self._parse_question(question) is not None

    def _parse_question(self, question: str) -> CryptoQuery | None:
        """Extract asset, direction, and timeframe from question."""
        # 1. Find crypto asset
        m = _CRYPTO_ASSET_RE.search(question)
        if not m:
            return None

        asset_raw = m.group(1).lower()
        asset = _ASSET_MAP.get(asset_raw)
        if not asset:
            return None

        # Must have price context (price, go up/down, above/below, etc.)
        has_price = _PRICE_CONTEXT_RE.search(question) is not None
        has_bullish = _BULLISH_PATTERNS.search(question)
        has_bearish = _BEARISH_PATTERNS.search(question)

        if not has_price and not has_bullish and not has_bearish:
            return None

        # 2. Determine direction
        if has_bearish:
            direction = "down"
        elif has_bullish:
            direction = "up"
        else:
            direction = "up"  # Default

        # 3. Timeframe extraction
        timeframe = "5m"  # Default
        tm = _TIMEFRAME_RE.search(question)
        if tm:
            num = int(tm.group(1))
            unit = tm.group(2).lower()
            if unit in ("h", "hr", "hrs", "hour", "hours"):
                timeframe = f"{num}h"
            else:
                timeframe = f"{num}m"

        return CryptoQuery(asset=asset, direction=direction, timeframe=timeframe)

    def _compute_indicators(self, candles: list[Candle]) -> TAIndicators:
        """Compute all TA indicators from candle data."""
        if not candles:
            return TAIndicators()

        closes = [c.close for c in candles]

        rsi = compute_rsi(closes, period=14)
        vwap = compute_vwap(candles)
        current_price = closes[-1]
        vwap_dev = (current_price - vwap) / vwap if vwap > 0 else 0.0

        sma_9 = compute_sma(closes, 9)
        sma_21 = compute_sma(closes, 21)

        # SMA crossover signal
        if sma_9 > sma_21 * 1.001:  # Bullish with small buffer
            sma_cross = 1.0
        elif sma_9 < sma_21 * 0.999:  # Bearish
            sma_cross = -1.0
        else:
            sma_cross = 0.0

        mom = compute_momentum(closes, period=10)

        # Composite signal (weighted sum)
        rsi_norm = (rsi - 50) / 50
        vwap_norm = max(-1.0, min(1.0, vwap_dev * 10))
        mom_norm = max(-1.0, min(1.0, mom * 100))

        composite = (
            0.30 * rsi_norm
            + 0.25 * vwap_norm
            + 0.25 * sma_cross
            + 0.20 * mom_norm
        )

        return TAIndicators(
            rsi_14=rsi,
            vwap=vwap,
            vwap_deviation=vwap_dev,
            sma_9=sma_9,
            sma_21=sma_21,
            sma_crossover=sma_cross,
            momentum=mom,
            composite_signal=composite,
        )

    def _signal_to_probability(
        self, indicators: TAIndicators, direction: str,
    ) -> float:
        """Convert composite signal to probability via logistic function."""
        signal = indicators.composite_signal

        # If question asks about "down", invert the signal
        if direction == "down":
            signal = -signal

        # Logistic sigmoid: 1 / (1 + exp(-k*x)), k=3 for moderate steepness
        probability = 1.0 / (1.0 + math.exp(-3.0 * signal))
        return max(0.05, min(0.95, probability))

    async def _fetch_candles(
        self, asset: str, interval: str = "1m", limit: int = 100,
    ) -> list[Candle]:
        """Fetch candles from Binance public API."""
        await rate_limiter.get("binance").acquire()

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)

        symbol = f"{asset.upper()}USDT"
        resp = await self._client.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
            },
        )
        resp.raise_for_status()

        return [
            Candle(
                timestamp=row[0] / 1000,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
            for row in resp.json()
        ]

    async def forecast(
        self,
        market: Any,
        features: Any,
        classification: Any,
    ) -> SpecialistResult:
        """Produce a crypto TA forecast."""
        query = self._parse_question(market.question)
        if query is None:
            raise ValueError(f"Cannot parse crypto question: {market.question}")

        candles = await self._fetch_candles(query.asset, interval="1m", limit=100)
        indicators = self._compute_indicators(candles)
        probability = self._signal_to_probability(indicators, query.direction)

        return SpecialistResult(
            probability=probability,
            confidence_level="LOW",  # TA on short timeframes is low confidence
            reasoning=(
                f"Composite TA: RSI={indicators.rsi_14:.1f}, "
                f"VWAP_dev={indicators.vwap_deviation:.3f}, "
                f"SMA_cross={indicators.sma_crossover:.0f}, "
                f"momentum={indicators.momentum:.4f}"
            ),
            evidence_quality=0.4,
            specialist_name="crypto_ta",
            specialist_metadata=indicators.to_dict(),
            bypasses_llm=True,
            key_evidence=[{
                "source": f"Binance {query.asset}/USDT 1m candles",
                "text": (
                    f"Composite signal={indicators.composite_signal:.3f} → "
                    f"P({query.direction})={probability:.1%}"
                ),
            }],
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
