"""Binance futures funding rate connector — sentiment signal for CRYPTO.

Fetches the current and recent funding rates for perpetual futures contracts
from Binance's public API.  A persistently positive funding rate implies
the market is net-long (bullish sentiment), while negative means net-short.

No API key required.  Rate-limited via the existing ``binance`` bucket.
"""

from __future__ import annotations

import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

# Coin → Binance futures ticker
_COIN_TICKERS: dict[str, str] = {
    "bitcoin": "BTCUSDT", "btc": "BTCUSDT",
    "ethereum": "ETHUSDT", "eth": "ETHUSDT",
    "solana": "SOLUSDT", "sol": "SOLUSDT",
    "bnb": "BNBUSDT",
    "xrp": "XRPUSDT", "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT", "doge": "DOGEUSDT",
    "cardano": "ADAUSDT", "ada": "ADAUSDT",
    "avalanche": "AVAXUSDT", "avax": "AVAXUSDT",
    "polygon": "MATICUSDT", "matic": "MATICUSDT",
}


class CryptoFuturesConnector(BaseResearchConnector):
    """Binance perpetual futures funding rate signal for CRYPTO markets."""

    @property
    def name(self) -> str:
        return "crypto_futures"

    def relevant_categories(self) -> set[str]:
        return {"CRYPTO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type != "CRYPTO":
            return False
        return self._extract_ticker(question) is not None

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        ticker = self._extract_ticker(question)
        if not ticker:
            return []

        await rate_limiter.get("binance").acquire()

        client = self._get_client(timeout=10.0)

        # Fetch current premium index (includes current funding rate)
        resp = await client.get(
            _PREMIUM_URL,
            params={"symbol": ticker},
        )
        resp.raise_for_status()
        premium = resp.json()

        current_rate = float(premium.get("lastFundingRate", 0))

        # Fetch recent funding rate history (last 8 periods = 24 hours)
        await rate_limiter.get("binance").acquire()
        resp2 = await client.get(
            _FUNDING_URL,
            params={"symbol": ticker, "limit": 8},
        )
        resp2.raise_for_status()
        history = resp2.json()

        rates = [float(r["fundingRate"]) for r in history]
        avg_rate = sum(rates) / len(rates) if rates else 0.0

        # Interpret sentiment
        if avg_rate > 0.0005:
            sentiment = "strongly bullish"
        elif avg_rate > 0.0001:
            sentiment = "moderately bullish"
        elif avg_rate < -0.0005:
            sentiment = "strongly bearish"
        elif avg_rate < -0.0001:
            sentiment = "moderately bearish"
        else:
            sentiment = "neutral"

        annualized = avg_rate * 3 * 365  # 3 funding periods/day

        symbol = ticker.replace("USDT", "")
        content = (
            f"Binance Futures Funding Rate: {symbol}\n"
            f"  Current rate: {current_rate:.6f} ({current_rate * 100:.4f}%)\n"
            f"  24h average: {avg_rate:.6f} ({avg_rate * 100:.4f}%)\n"
            f"  Annualized: {annualized:.1%}\n"
            f"  Sentiment: {sentiment}\n"
            f"  Interpretation: {'Longs pay shorts' if avg_rate > 0 else 'Shorts pay longs'}\n"
            f"  Source: Binance Futures (perpetual swap)"
        )

        return [
            self._make_source(
                title=f"Funding Rate: {symbol} Perps",
                url=f"https://www.binance.com/en/futures/{symbol}USDT",
                snippet=(
                    f"{symbol} funding rate: {avg_rate * 100:.4f}% "
                    f"(24h avg, {sentiment})"
                ),
                publisher="Binance Futures",
                content=content,
                authority_score=0.65,
                raw={
                    "behavioral_signal": {
                        "source": "crypto_futures",
                        "signal_type": "funding_rate",
                        "value": round(avg_rate, 8),
                        "current_rate": round(current_rate, 8),
                        "annualized": round(annualized, 4),
                        "sentiment": sentiment,
                        "symbol": symbol,
                        "periods": len(rates),
                    }
                },
            )
        ]

    @staticmethod
    def _extract_ticker(question: str) -> str | None:
        """Extract Binance futures ticker from question text."""
        q = question.lower()
        for keyword, ticker in _COIN_TICKERS.items():
            if keyword in q:
                return ticker
        return None
