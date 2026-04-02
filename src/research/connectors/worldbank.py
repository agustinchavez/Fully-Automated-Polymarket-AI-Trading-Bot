"""World Bank research connector — development indicators for MACRO/GEOPOLITICS.

Uses the free World Bank API (no key required):
- Endpoint: api.worldbank.org/v2/country/{iso2}/indicator/{code}
- Returns JSON with recent data values
"""

from __future__ import annotations

import re

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_WB_API = "https://api.worldbank.org/v2/country/{iso2}/indicator/{code}"

# Country names → ISO-2 codes
_COUNTRY_MAP: dict[str, str] = {
    "china": "CN",
    "india": "IN",
    "brazil": "BR",
    "russia": "RU",
    "germany": "DE",
    "france": "FR",
    "uk": "GB",
    "united kingdom": "GB",
    "britain": "GB",
    "japan": "JP",
    "south korea": "KR",
    "korea": "KR",
    "mexico": "MX",
    "canada": "CA",
    "australia": "AU",
    "indonesia": "ID",
    "turkey": "TR",
    "saudi arabia": "SA",
    "argentina": "AR",
    "south africa": "ZA",
    "nigeria": "NG",
    "egypt": "EG",
    "italy": "IT",
    "spain": "ES",
    "poland": "PL",
    "netherlands": "NL",
    "sweden": "SE",
    "switzerland": "CH",
    "norway": "NO",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
    "vietnam": "VN",
    "thailand": "TH",
    "philippines": "PH",
    "malaysia": "MY",
    "pakistan": "PK",
    "bangladesh": "BD",
    "kenya": "KE",
    "ethiopia": "ET",
    "ukraine": "UA",
    "iran": "IR",
    "iraq": "IQ",
    "israel": "IL",
    "taiwan": "TW",
    "singapore": "SG",
    "hong kong": "HK",
    "new zealand": "NZ",
    "greece": "GR",
    "portugal": "PT",
    "czech republic": "CZ",
    "romania": "RO",
    "hungary": "HU",
}

# Keyword → (indicator code, display name)
_INDICATOR_MAP: list[tuple[list[str], str, str]] = [
    (
        ["gdp growth", "economic growth", "gdp"],
        "NY.GDP.MKTP.KD.ZG",
        "GDP Growth (annual %)",
    ),
    (
        ["inflation", "cpi", "consumer price"],
        "FP.CPI.TOTL.ZG",
        "Inflation, consumer prices (annual %)",
    ),
    (
        ["unemployment", "jobless"],
        "SL.UEM.TOTL.ZS",
        "Unemployment, total (% of labor force)",
    ),
    (
        ["exports", "trade", "import"],
        "NE.EXP.GNFS.ZS",
        "Exports of goods and services (% of GDP)",
    ),
    (
        ["debt", "government debt", "public debt"],
        "GC.DOD.TOTL.GD.ZS",
        "Central government debt (% of GDP)",
    ),
    (
        ["gdp per capita", "income per capita", "living standard"],
        "NY.GDP.PCAP.CD",
        "GDP per capita (current US$)",
    ),
    (
        ["population", "population growth"],
        "SP.POP.TOTL",
        "Population, total",
    ),
    (
        ["poverty", "poverty rate"],
        "SI.POV.DDAY",
        "Poverty headcount ratio at $2.15/day",
    ),
]


class WorldBankConnector(BaseResearchConnector):
    """Fetch development indicators from the World Bank API."""

    @property
    def name(self) -> str:
        return "worldbank"

    def relevant_categories(self) -> set[str]:
        return {"MACRO", "GEOPOLITICS"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type not in self.relevant_categories():
            return False
        # Only trigger if question mentions a non-US country
        return self._extract_country(question) is not None

    def _extract_country(self, question: str) -> tuple[str, str] | None:
        """Find a country mention in the question → (name, iso2)."""
        q_lower = question.lower()
        # Check multi-word countries first (longest match)
        sorted_countries = sorted(_COUNTRY_MAP.keys(), key=len, reverse=True)
        for country in sorted_countries:
            if country in q_lower:
                return country, _COUNTRY_MAP[country]
        return None

    def _match_indicators(
        self, question: str, max_indicators: int = 2,
    ) -> list[tuple[str, str]]:
        """Map question keywords to World Bank indicator codes."""
        q_lower = question.lower()
        matched: list[tuple[str, str]] = []

        for keywords, code, display_name in _INDICATOR_MAP:
            if any(kw in q_lower for kw in keywords):
                matched.append((code, display_name))

        if not matched:
            # Default to GDP growth for MACRO questions
            matched.append(("NY.GDP.MKTP.KD.ZG", "GDP Growth (annual %)"))

        return matched[:max_indicators]

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        country_match = self._extract_country(question)
        if not country_match:
            return []

        country_name, iso2 = country_match
        mrv = getattr(self._config, "worldbank_mrv", 3)
        indicators = self._match_indicators(question)

        sources: list[FetchedSource] = []
        for code, display_name in indicators:
            source = await self._fetch_indicator(
                iso2, country_name, code, display_name, mrv,
            )
            if source:
                sources.append(source)

        return sources

    async def _fetch_indicator(
        self,
        iso2: str,
        country_name: str,
        code: str,
        display_name: str,
        mrv: int,
    ) -> FetchedSource | None:
        """Fetch a single indicator for a country."""
        await rate_limiter.get("worldbank").acquire()

        client = self._get_client()
        url = _WB_API.format(iso2=iso2, code=code)
        resp = await client.get(
            url,
            params={
                "format": "json",
                "mrv": mrv,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # World Bank API returns [metadata, data_array]
        if not isinstance(data, list) or len(data) < 2:
            return None
        records = data[1]
        if not records:
            return None

        lines = [
            f"World Bank Data: {country_name.title()}",
            f"Indicator: {display_name}",
        ]
        latest_date = ""
        for i, rec in enumerate(records):
            year = rec.get("date", "N/A")
            value = rec.get("value")
            label = "Latest" if i == 0 else f"Prior ({i})"
            val_str = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
            lines.append(f"{label}: {val_str} ({year})")
            if i == 0:
                latest_date = str(year)

        lines.append("Source: World Bank Open Data")

        content = "\n".join(lines)
        latest_val = records[0].get("value", "N/A")
        snippet = f"{country_name.title()} {display_name}: {latest_val} ({latest_date})"

        return self._make_source(
            title=f"World Bank: {country_name.title()} — {display_name}",
            url=f"https://data.worldbank.org/indicator/{code}?locations={iso2}",
            snippet=snippet,
            publisher="World Bank",
            date=latest_date,
            content=content,
        )
