"""DeFiLlama TVL connector — total value locked signal for CRYPTO markets.

Fetches protocol-level TVL data from the free DeFiLlama API (no key required).
TVL trends indicate ecosystem health — rising TVL is bullish, falling is bearish.

Rate-limited via a dedicated ``defillama`` bucket.
"""

from __future__ import annotations

import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_PROTOCOLS_URL = "https://api.llama.fi/protocols"
_TVL_URL = "https://api.llama.fi/tvl/{protocol}"
_PROTOCOL_URL = "https://api.llama.fi/protocol/{protocol}"

# Keyword → DeFiLlama protocol slug mapping
_PROTOCOL_MAP: dict[str, str] = {
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "avalanche": "avalanche",
    "avax": "avalanche",
    "polygon": "polygon",
    "matic": "polygon",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "bnb": "bsc",
    "bsc": "bsc",
    "binance": "bsc",
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "cardano": "cardano",
    "ada": "cardano",
    # DeFi protocols
    "aave": "aave",
    "uniswap": "uniswap",
    "lido": "lido",
    "makerdao": "makerdao",
    "maker": "makerdao",
    "compound": "compound",
    "curve": "curve-dex",
}


class DeFiLlamaConnector(BaseResearchConnector):
    """DeFiLlama TVL data for CRYPTO markets."""

    @property
    def name(self) -> str:
        return "defillama"

    def relevant_categories(self) -> set[str]:
        return {"CRYPTO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type != "CRYPTO":
            return False
        return self._extract_protocol(question) is not None

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        protocol = self._extract_protocol(question)
        if not protocol:
            return []

        await rate_limiter.get("defillama").acquire()

        client = self._get_client(timeout=10.0)

        # Check if this is a chain (use /v2/chains) or protocol
        # Try protocol endpoint first
        try:
            resp = await client.get(
                f"https://api.llama.fi/protocol/{protocol}"
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            # Try as chain TVL
            try:
                resp = await client.get(
                    f"https://api.llama.fi/v2/historicalChainTvl/{protocol}"
                )
                resp.raise_for_status()
                chain_data = resp.json()
                if not chain_data:
                    return []
                return self._build_chain_source(protocol, chain_data)
            except Exception:
                return []

        current_tvl = data.get("currentChainTvls", {})
        total_tvl = sum(
            v for k, v in current_tvl.items()
            if isinstance(v, (int, float)) and not k.endswith("-borrowed")
            and not k.endswith("-staking")
        )
        if total_tvl <= 0:
            total_tvl = data.get("tvl", 0)

        # Get TVL history for trend
        tvl_history = data.get("tvl", [])
        if isinstance(tvl_history, list) and len(tvl_history) >= 7:
            recent = tvl_history[-1].get("totalLiquidityUSD", 0)
            week_ago = tvl_history[-7].get("totalLiquidityUSD", 0)
            if week_ago > 0:
                change_7d = (recent - week_ago) / week_ago
            else:
                change_7d = 0.0
        else:
            change_7d = 0.0
            recent = total_tvl

        trend = "rising" if change_7d > 0.02 else "falling" if change_7d < -0.02 else "stable"
        name = data.get("name", protocol)

        content = (
            f"DeFiLlama TVL: {name}\n"
            f"  Current TVL: ${total_tvl:,.0f}\n"
            f"  7-day change: {change_7d:+.1%}\n"
            f"  Trend: {trend}\n"
            f"  Source: DeFiLlama (defillama.com)"
        )

        return [
            self._make_source(
                title=f"TVL: {name}",
                url=f"https://defillama.com/protocol/{protocol}",
                snippet=f"{name} TVL: ${total_tvl:,.0f} ({change_7d:+.1%} 7d, {trend})",
                publisher="DeFiLlama",
                content=content,
                authority_score=0.70,
                raw={
                    "behavioral_signal": {
                        "source": "defillama",
                        "signal_type": "tvl",
                        "value": round(change_7d, 4),
                        "tvl_usd": round(total_tvl, 2),
                        "change_7d_pct": round(change_7d, 4),
                        "trend": trend,
                        "protocol": protocol,
                    }
                },
            )
        ]

    def _build_chain_source(
        self, chain: str, history: list[dict],
    ) -> list[FetchedSource]:
        """Build source from chain-level TVL history."""
        if not history:
            return []

        current = history[-1].get("tvl", 0)
        if len(history) >= 7:
            week_ago = history[-7].get("tvl", 0)
            change_7d = (current - week_ago) / week_ago if week_ago > 0 else 0.0
        else:
            change_7d = 0.0

        trend = "rising" if change_7d > 0.02 else "falling" if change_7d < -0.02 else "stable"

        content = (
            f"DeFiLlama Chain TVL: {chain.title()}\n"
            f"  Current TVL: ${current:,.0f}\n"
            f"  7-day change: {change_7d:+.1%}\n"
            f"  Trend: {trend}\n"
            f"  Source: DeFiLlama (defillama.com)"
        )

        return [
            self._make_source(
                title=f"Chain TVL: {chain.title()}",
                url=f"https://defillama.com/chain/{chain.title()}",
                snippet=f"{chain.title()} chain TVL: ${current:,.0f} ({change_7d:+.1%} 7d)",
                publisher="DeFiLlama",
                content=content,
                authority_score=0.65,
                raw={
                    "behavioral_signal": {
                        "source": "defillama",
                        "signal_type": "tvl",
                        "value": round(change_7d, 4),
                        "tvl_usd": round(current, 2),
                        "change_7d_pct": round(change_7d, 4),
                        "trend": trend,
                        "protocol": chain,
                    }
                },
            )
        ]

    @staticmethod
    def _extract_protocol(question: str) -> str | None:
        """Extract DeFiLlama protocol/chain slug from question text."""
        q = question.lower()
        for keyword, slug in _PROTOCOL_MAP.items():
            if keyword in q:
                return slug
        return None
