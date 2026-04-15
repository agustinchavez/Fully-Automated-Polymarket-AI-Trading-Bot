"""openFDA research connector — FDA drug data for SCIENCE markets.

Uses the free openFDA API (no key required, optional key for higher limits):
- Endpoint: api.fda.gov/drug/drugsfda.json
- Free tier: 240 req/min without key, 120 req/min with key per IP
"""

from __future__ import annotations

import os
import re

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_OPENFDA_DRUGS = "https://api.fda.gov/drug/drugsfda.json"

_FDA_KEYWORDS: list[str] = [
    "fda", "drug", "approval", "nda", "bla", "pdufa",
    "phase 3", "clinical", "vaccine", "biotech", "pharma",
    "cancer", "device", "therapeutic", "medication",
    "pharmaceutical", "biologic", "generic", "recall",
]


class OpenFDAConnector(BaseResearchConnector):
    """Fetch FDA drug approval and safety data from openFDA."""

    @property
    def name(self) -> str:
        return "openfda"

    def relevant_categories(self) -> set[str]:
        return {"SCIENCE"}

    _EXCLUDED_TYPES: set[str] = {"SPORTS", "CULTURE", "ELECTION", "CRYPTO", "WEATHER", "GEOPOLITICS"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        # Skip categories that will never have FDA-relevant content
        if market_type.upper() in self._EXCLUDED_TYPES:
            return False
        # Only trigger on FDA-related keywords (not all SCIENCE)
        return self._question_matches_keywords(question, _FDA_KEYWORDS)

    def _get_api_key(self) -> str:
        """Read openFDA API key from config or env (optional)."""
        if self._config:
            key = getattr(self._config, "openfda_api_key", "")
            if key:
                return key
        return os.environ.get("OPENFDA_API_KEY", "")

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        drug_term = self._extract_drug_term(question)
        if not drug_term:
            return []

        await rate_limiter.get("openfda").acquire()

        client = self._get_client()
        params: dict[str, str | int] = {
            "search": f'openfda.brand_name:"{drug_term}" OR openfda.generic_name:"{drug_term}" OR sponsor_name:"{drug_term}"',
            "limit": 5,
        }
        api_key = self._get_api_key()
        if api_key:
            params["api_key"] = api_key

        resp = await client.get(_OPENFDA_DRUGS, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return []

        sources: list[FetchedSource] = []
        for result in results[:3]:
            source = self._parse_drug_result(result)
            if source:
                sources.append(source)

        return sources

    def _extract_drug_term(self, question: str) -> str:
        """Pull drug or company name from question for search."""
        # Try to find quoted terms first
        quoted = re.findall(r'"([^"]+)"', question)
        if quoted:
            return quoted[0]

        # Try to find capitalized words (likely proper nouns = drug/company names)
        words = question.split()
        proper_nouns = [
            w for w in words
            if w[0].isupper() and len(w) > 2
            and w.lower() not in {
                "will", "the", "fda", "does", "has", "are", "can",
                "should", "what", "when", "phase", "drug",
            }
        ]
        if proper_nouns:
            return " ".join(proper_nouns[:2])

        # Fall back: extract key content words
        cleaned = re.sub(
            r"\b(will|the|a|an|in|of|to|for|and|or|is|be|by|fda|"
            r"drug|approval|approve|approved|get|receive)\b",
            "",
            question.lower(),
        )
        terms = [t for t in cleaned.split() if len(t) > 3]
        return " ".join(terms[:2]) if terms else ""

    def _parse_drug_result(self, result: dict) -> FetchedSource | None:
        """Parse a single openFDA drug result into a FetchedSource."""
        openfda = result.get("openfda", {})
        brand_names = openfda.get("brand_name", [])
        generic_names = openfda.get("generic_name", [])
        brand = brand_names[0] if brand_names else "Unknown"
        generic = generic_names[0] if generic_names else ""

        sponsor = result.get("sponsor_name", "Unknown")
        app_number = result.get("application_number", "")

        # Get submission info
        submissions = result.get("submissions", [])
        approval_date = ""
        submission_type = ""
        if submissions:
            latest = submissions[0]
            approval_date = latest.get("submission_status_date", "")
            submission_type = latest.get("submission_type", "")

        # Products info
        products = result.get("products", [])
        indications = []
        for prod in products[:3]:
            route = prod.get("route", "")
            dosage = prod.get("dosage_form", "")
            if route or dosage:
                indications.append(f"{dosage} ({route})")

        content_lines = [
            f"FDA Drug Data: {brand}",
            f"Generic Name: {generic}",
            f"Sponsor: {sponsor}",
            f"Application Number: {app_number}",
        ]
        if approval_date:
            content_lines.append(f"Approval Date: {approval_date}")
        if submission_type:
            content_lines.append(f"Submission Type: {submission_type}")
        if indications:
            content_lines.append(f"Dosage Forms: {', '.join(indications)}")
        content_lines.append("Source: openFDA (U.S. Food & Drug Administration)")

        content = "\n".join(content_lines)
        snippet = f"{brand} ({generic}) by {sponsor}"
        if approval_date:
            snippet += f" — approved {approval_date}"

        return self._make_source(
            title=f"FDA: {brand} ({app_number})",
            url=f"https://api.fda.gov/drug/drugsfda.json?search=application_number:{app_number}",
            snippet=snippet,
            publisher="openFDA",
            date=approval_date,
            content=content,
        )
