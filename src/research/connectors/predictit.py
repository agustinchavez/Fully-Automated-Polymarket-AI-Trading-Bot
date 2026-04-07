"""PredictIt consensus connector — US political prediction market.

Free API, no auth. Fetches all markets in a single call (cached 60s).
Binary market matching via text similarity.
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_ALL_MARKETS_URL = "https://www.predictit.org/api/marketdata/all/"
_CACHE_TTL = 60  # seconds — matches API rate limit
_MIN_SIMILARITY = 0.45


class PredictItConnector(BaseResearchConnector):
    """Fetch consensus probability from PredictIt."""

    name = "predictit"

    def relevant_categories(self) -> set[str]:
        return {"ELECTION", "GEOPOLITICS"}

    _cache_data: list[Any] | None = None
    _cache_time: float = 0.0

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        await rate_limiter.get("predictit").acquire()

        markets = await self._get_all_markets()
        best = self._find_best_match(question, markets)
        if best is None:
            return []

        market, contract, prob = best
        url = market.get("url", "")

        return [self._make_source(
            url=url,
            title=market.get("name", ""),
            snippet=f"PredictIt: {prob:.1%}",
            publisher="PredictIt",
            content=f"PredictIt: {prob:.1%}",
            authority_score=0.68,
            raw={
                "consensus_signal": {
                    "platform": "predictit",
                    "price": prob,
                    "market_name": market.get("name", ""),
                },
            },
        )]

    async def _get_all_markets(self) -> list[Any]:
        now = time.monotonic()
        if (
            PredictItConnector._cache_data is not None
            and now - PredictItConnector._cache_time < _CACHE_TTL
        ):
            return PredictItConnector._cache_data

        try:
            client = self._get_client()
            resp = await client.get(_ALL_MARKETS_URL)
            resp.raise_for_status()
            data = resp.json()
            PredictItConnector._cache_data = data.get("markets", [])
            PredictItConnector._cache_time = now
        except Exception as exc:
            log.warning("predictit.fetch_failed", error=str(exc))
            PredictItConnector._cache_data = []
            PredictItConnector._cache_time = now

        return PredictItConnector._cache_data  # type: ignore[return-value]

    @staticmethod
    def _find_best_match(
        question: str,
        markets: list[Any],
    ) -> tuple[dict, dict, float] | None:
        q = question.lower()
        best_score = _MIN_SIMILARITY
        best: tuple[dict, dict, float] | None = None

        for market in markets:
            name = (market.get("name") or "").lower()
            score = SequenceMatcher(None, q[:80], name[:80]).ratio()
            if score < best_score:
                continue

            contracts = market.get("contracts", [])
            # Binary: single YES/NO contract
            if len(contracts) == 1:
                c = contracts[0]
                buy = c.get("bestBuyYesCost")
                sell = c.get("bestSellYesCost")
                if buy and sell:
                    best_score = score
                    best = (market, c, (buy + sell) / 2)
            # Binary: two mutually exclusive contracts summing to ~1
            elif len(contracts) == 2:
                p0 = contracts[0].get("bestBuyYesCost", 0) or 0
                p1 = contracts[1].get("bestBuyYesCost", 0) or 0
                if 0.85 <= p0 + p1 <= 1.15:
                    best_score = score
                    best = (market, contracts[0], p0)

        return best
