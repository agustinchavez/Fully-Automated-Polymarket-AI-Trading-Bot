"""SEC EDGAR research connector — corporate filings and financial data.

Uses the free SEC EDGAR APIs (no key required):
- Full-text search: efts.sec.gov/LATEST/search-index
- XBRL facts: data.sec.gov/api/xbrl/companyfacts
Rate limit: 10 req/sec with User-Agent header (SEC policy).
"""

from __future__ import annotations

import re

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_EFTS_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_XBRL_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SEC_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K"

# Major public companies → CIK numbers
_CIK_MAP: dict[str, str] = {
    "apple": "0000320193",
    "aapl": "0000320193",
    "microsoft": "0000789019",
    "msft": "0000789019",
    "google": "0001652044",
    "alphabet": "0001652044",
    "googl": "0001652044",
    "amazon": "0001018724",
    "amzn": "0001018724",
    "meta": "0001326801",
    "facebook": "0001326801",
    "tesla": "0001318605",
    "tsla": "0001318605",
    "nvidia": "0001045810",
    "nvda": "0001045810",
    "jpmorgan": "0000019617",
    "jpm": "0000019617",
    "berkshire": "0001067983",
    "brk": "0001067983",
    "johnson": "0000200406",
    "jnj": "0000200406",
    "walmart": "0000104169",
    "wmt": "0000104169",
    "visa": "0001403161",
    "disney": "0001744489",
    "dis": "0001744489",
    "netflix": "0001065280",
    "nflx": "0001065280",
    "intel": "0000050863",
    "intc": "0000050863",
    "amd": "0000002488",
    "adobe": "0000796343",
    "salesforce": "0001108524",
    "crm": "0001108524",
    "pfizer": "0000078003",
    "pfe": "0000078003",
    "moderna": "0001682852",
    "mrna": "0001682852",
    "boeing": "0000012927",
    "ba": "0000012927",
    "goldman": "0000886982",
    "gs": "0000886982",
    "uber": "0001543151",
    "lyft": "0001759509",
    "coinbase": "0001679788",
    "coin": "0001679788",
    "palantir": "0001321655",
    "pltr": "0001321655",
    "snap": "0001564408",
    "spotify": "0001639920",
    "spot": "0001639920",
}

_CORPORATE_KEYWORDS: list[str] = [
    "earnings", "revenue", "profit", "merger", "acquisition",
    "ipo", "ceo", "layoffs", "stock", "shares", "quarterly",
    "annual report", "sec filing", "10-k", "10-q", "8-k",
    "dividend", "buyback", "restructuring", "bankruptcy",
]

_NUMERIC_KEYWORDS: list[str] = [
    "earnings", "revenue", "profit", "income", "sales",
    "eps", "margin", "assets", "debt", "cash flow",
]

_SEC_USER_AGENT = "PolymarketBot/1.0 (research@example.com)"


class EdgarConnector(BaseResearchConnector):
    """Fetch SEC filings and XBRL financial data from EDGAR."""

    @property
    def name(self) -> str:
        return "edgar"

    def relevant_categories(self) -> set[str]:
        return {"CORPORATE", "TECH"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _CORPORATE_KEYWORDS)

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        max_filings = getattr(self._config, "edgar_max_filings", 3)
        company, cik = self._extract_company(question)
        sources: list[FetchedSource] = []

        if cik and self._is_numeric_question(question):
            source = await self._fetch_xbrl(cik, company, question)
            if source:
                sources.append(source)

        if cik:
            filing_sources = await self._search_filings(
                company, cik, max_filings,
            )
            sources.extend(filing_sources)

        if not cik:
            # Fall back to full-text search
            filing_sources = await self._search_efts(question, max_filings)
            sources.extend(filing_sources)

        return sources[:max_filings]

    def _extract_company(self, question: str) -> tuple[str, str]:
        """Match question against company keywords → (name, CIK)."""
        q_lower = question.lower()
        for keyword, cik in _CIK_MAP.items():
            if keyword in q_lower:
                return keyword.title(), cik
        return "", ""

    def _is_numeric_question(self, question: str) -> bool:
        """Detect earnings/revenue/profit keywords for XBRL mode."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in _NUMERIC_KEYWORDS)

    async def _fetch_xbrl(
        self, cik: str, company: str, question: str,
    ) -> FetchedSource | None:
        """Fetch XBRL financial facts for a company."""
        await rate_limiter.get("edgar").acquire()

        client = self._get_client()
        url = _XBRL_FACTS.format(cik=cik)
        resp = await client.get(
            url,
            headers={"User-Agent": _SEC_USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        facts = data.get("facts", {}).get("us-gaap", {})
        if not facts:
            return None

        # Find relevant metric
        metric_name, metric_key = self._match_xbrl_metric(question, facts)
        if not metric_key:
            return None

        units = facts[metric_key].get("units", {})
        values_key = "USD" if "USD" in units else next(iter(units), None)
        if not values_key:
            return None

        entries = units[values_key]
        recent = sorted(entries, key=lambda e: e.get("end", ""), reverse=True)[:3]
        if not recent:
            return None

        lines = [
            f"SEC EDGAR XBRL Data: {company}",
            f"Metric: {metric_name}",
        ]
        for i, entry in enumerate(recent):
            label = "Latest" if i == 0 else f"Prior ({i})"
            val = entry.get("val", "N/A")
            period = entry.get("end", "N/A")
            form = entry.get("form", "")
            lines.append(f"{label}: {val:,} ({period}, {form})" if isinstance(val, (int, float)) else f"{label}: {val} ({period})")
        lines.append("Source: SEC EDGAR (XBRL)")

        content = "\n".join(lines)
        snippet = f"{company} {metric_name}: {recent[0].get('val', 'N/A')} ({recent[0].get('end', '')})"

        return self._make_source(
            title=f"SEC EDGAR: {company} — {metric_name}",
            url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            snippet=snippet,
            publisher="SEC EDGAR",
            date=recent[0].get("end", ""),
            content=content,
        )

    def _match_xbrl_metric(
        self, question: str, facts: dict,
    ) -> tuple[str, str]:
        """Map question keywords to an XBRL metric key."""
        q_lower = question.lower()
        # Priority order of metrics to check
        metric_map: list[tuple[list[str], str, str]] = [
            (["revenue", "sales"], "Revenues", "Revenue"),
            (["revenue", "sales"], "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenue"),
            (["earnings", "income", "profit"], "NetIncomeLoss", "Net Income"),
            (["eps", "earnings per share"], "EarningsPerShareBasic", "EPS (Basic)"),
            (["assets", "total assets"], "Assets", "Total Assets"),
            (["debt", "liabilities"], "Liabilities", "Total Liabilities"),
            (["cash", "cash flow"], "CashAndCashEquivalentsAtCarryingValue", "Cash & Equivalents"),
        ]

        for keywords, fact_key, display_name in metric_map:
            if any(kw in q_lower for kw in keywords) and fact_key in facts:
                return display_name, fact_key

        return "", ""

    async def _search_filings(
        self, company: str, cik: str, max_results: int,
    ) -> list[FetchedSource]:
        """Search for recent 8-K filings by CIK."""
        await rate_limiter.get("edgar").acquire()

        client = self._get_client()
        resp = await client.get(
            _EFTS_SEARCH,
            params={
                "q": company,
                "dateRange": "custom",
                "startdt": "2025-01-01",
                "enddt": "2026-12-31",
                "forms": "8-K",
                "from": 0,
                "size": max_results,
            },
            headers={"User-Agent": _SEC_USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        return self._parse_filing_results(data, max_results)

    async def _search_efts(
        self, question: str, max_results: int,
    ) -> list[FetchedSource]:
        """Full-text search on EDGAR for any question."""
        await rate_limiter.get("edgar").acquire()

        # Extract key terms (skip common words)
        terms = re.sub(r"\b(will|the|a|an|in|of|to|for|and|or|is|be|by)\b", "", question.lower())
        terms = " ".join(terms.split()[:5])

        client = self._get_client()
        resp = await client.get(
            _EFTS_SEARCH,
            params={
                "q": terms,
                "forms": "8-K,10-K,10-Q",
                "from": 0,
                "size": max_results,
            },
            headers={"User-Agent": _SEC_USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()

        return self._parse_filing_results(data, max_results)

    def _parse_filing_results(
        self, data: dict, max_results: int,
    ) -> list[FetchedSource]:
        """Parse EFTS search results into FetchedSource list."""
        hits = data.get("hits", {}).get("hits", [])
        sources: list[FetchedSource] = []

        for hit in hits[:max_results]:
            src = hit.get("_source", {})
            entity = src.get("entity_name", "Unknown")
            form_type = src.get("form_type", "Filing")
            filed_date = src.get("file_date", "")
            file_num = src.get("file_num", "")

            content_lines = [
                f"SEC Filing: {form_type}",
                f"Company: {entity}",
                f"Filed: {filed_date}",
                f"File Number: {file_num}",
            ]
            content = "\n".join(content_lines)
            snippet = f"{entity} filed {form_type} on {filed_date}"

            sources.append(self._make_source(
                title=f"SEC {form_type}: {entity} ({filed_date})",
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={file_num}",
                snippet=snippet,
                publisher="SEC EDGAR",
                date=filed_date,
                content=content,
            ))

        return sources
