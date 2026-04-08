"""FRED research connector — Federal Reserve Economic Data for MACRO markets.

Uses the free FRED API (120 req/min, instant key signup) to fetch
official US economic time series: CPI, GDP, unemployment, Fed funds rate, etc.
"""

from __future__ import annotations

import os
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_SERIES_INFO = "https://api.stlouisfed.org/fred/series"

# Keyword → FRED series mapping (from spec)
_KEYWORD_SERIES: list[tuple[list[str], list[str], str]] = [
    # (keywords, series_ids, human_name)
    (
        ["fed rate", "federal funds", "rate cut", "rate hike", "interest rate",
         "the fed", "fed cut", "fed raise"],
        ["FEDFUNDS"],
        "Federal Funds Rate",
    ),
    (
        ["unemployment", "jobs", "jobless", "labor market"],
        ["UNRATE"],
        "Unemployment Rate",
    ),
    (
        ["inflation", "cpi", "consumer price"],
        ["CPIAUCSL"],
        "CPI All Urban Consumers",
    ),
    (
        ["gdp", "economic growth", "recession", "economy"],
        ["GDP"],
        "Gross Domestic Product",
    ),
    (
        ["mortgage", "housing", "home price"],
        ["MORTGAGE30US"],
        "30-Year Mortgage Rate",
    ),
    (
        ["treasury", "yield", "t-bill", "yield curve"],
        ["T10Y2Y", "DGS10"],
        "Treasury Yield",
    ),
    (
        ["consumer sentiment", "consumer confidence"],
        ["UMCSENT"],
        "Consumer Sentiment Index",
    ),
    (
        ["s&p", "stock market", "equities", "sp500"],
        ["SP500"],
        "S&P 500 Index",
    ),
    (
        ["pce", "personal consumption", "pce inflation"],
        ["PCEPI"],
        "PCE Price Index",
    ),
    (
        ["manufacturing employment", "factory workers", "manufacturing jobs"],
        ["MANEMP"],
        "Manufacturing Employment",
    ),
    (
        ["oil", "crude", "petroleum", "wti", "energy price"],
        ["DCOILWTICO"],
        "Crude Oil Price (WTI)",
    ),
    (
        ["retail sales", "consumer spending", "retail"],
        ["MRTSSM44X72USS"],
        "Retail Sales",
    ),
    (
        ["jobless claims", "initial claims", "weekly claims"],
        ["ICSA"],
        "Initial Jobless Claims",
    ),
    (
        ["money supply", "m2", "monetary"],
        ["M2SL"],
        "M2 Money Supply",
    ),
    (
        ["t-bill", "30-day", "short-term rate", "3-month treasury"],
        ["DTB3"],
        "3-Month Treasury Bill Rate",
    ),
]

_MACRO_KEYWORDS: list[str] = [
    "inflation", "cpi", "unemployment", "gdp", "interest rate",
    "federal reserve", "fed", "mortgage", "treasury", "yield",
    "recession", "consumer price", "rate cut", "rate hike",
    "jobs", "jobless", "consumer sentiment", "stock market",
    "pce", "manufacturing employment", "oil", "crude", "retail sales",
    "jobless claims", "money supply", "t-bill",
]


class FredConnector(BaseResearchConnector):
    """Fetch US economic data from the FRED API."""

    @property
    def name(self) -> str:
        return "fred"

    def relevant_categories(self) -> set[str]:
        return {"MACRO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _MACRO_KEYWORDS)

    def _get_api_key(self) -> str:
        """Read FRED API key from env."""
        return os.environ.get("FRED_API_KEY", "")

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        api_key = self._get_api_key()
        if not api_key:
            log.debug("fred.no_api_key")
            return []

        max_series = getattr(self._config, "fred_max_series", 3)
        series_ids = self._match_series(question, max_series)
        if not series_ids:
            return []

        sources: list[FetchedSource] = []
        for series_id, series_name in series_ids:
            source = await self._fetch_series(api_key, series_id, series_name)
            if source:
                sources.append(source)

        return sources

    def _match_series(
        self, question: str, max_series: int,
    ) -> list[tuple[str, str]]:
        """Map question keywords to FRED series IDs."""
        q_lower = question.lower()
        matched: list[tuple[str, str]] = []
        seen: set[str] = set()

        for keywords, series_ids, human_name in _KEYWORD_SERIES:
            if any(kw in q_lower for kw in keywords):
                for sid in series_ids:
                    if sid not in seen:
                        seen.add(sid)
                        matched.append((sid, human_name))

        return matched[:max_series]

    async def _fetch_series(
        self,
        api_key: str,
        series_id: str,
        series_name: str,
    ) -> FetchedSource | None:
        """Fetch latest 3 observations for a FRED series."""
        await rate_limiter.get("fred").acquire()

        client = self._get_client()
        resp = await client.get(
            _FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "sort_order": "desc",
                "limit": 3,
                "file_type": "json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            return None

        # Build content
        lines = [
            f"FRED Data: {series_name}",
            f"Series ID: {series_id}",
        ]
        for i, obs in enumerate(observations):
            label = "Latest" if i == 0 else f"Prior ({i})"
            lines.append(f"{label}: {obs['value']} ({obs['date']})")
        lines.append("Source: Federal Reserve Bank of St. Louis")

        content = "\n".join(lines)
        latest = observations[0]
        snippet = f"{series_name}: {latest['value']} as of {latest['date']}"

        return self._make_source(
            title=f"FRED: {series_name} ({series_id})",
            url=f"https://fred.stlouisfed.org/series/{series_id}",
            snippet=snippet,
            publisher="Federal Reserve Bank of St. Louis",
            date=latest["date"],
            content=content,
        )
