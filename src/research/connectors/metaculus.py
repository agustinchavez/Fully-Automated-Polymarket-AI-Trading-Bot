"""Metaculus consensus connector — expert forecaster probabilities.

Uses the Metaculus API:
- Endpoint: metaculus.com/api2/questions/
- Jaccard similarity filter for question matching
- Minimum forecasters gate for result quality
- Returns community_prediction.full.q2 (median probability)

API key required (set METACULUS_API_KEY env var).
"""

from __future__ import annotations

import os
import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_METACULUS_API = "https://www.metaculus.com/api2/questions/"

_STOP_WORDS: set[str] = {
    "will", "the", "a", "an", "be", "is", "are", "in", "on", "at",
    "to", "for", "of", "by", "this", "that", "or", "and", "not",
    "yes", "no", "?", "how", "what", "when", "where", "which",
    "do", "does", "has", "have", "it", "its",
}


class MetaculusConnector(BaseResearchConnector):
    """Fetch expert consensus probabilities from Metaculus."""

    @property
    def name(self) -> str:
        return "metaculus"

    def relevant_categories(self) -> set[str]:
        return {
            "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
            "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
            "ENTERTAINMENT", "TECH", "REGULATION", "UNKNOWN",
        }

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        key = getattr(self._config, "metaculus_api_key", "") or ""
        if not key:
            key = os.environ.get("METACULUS_API_KEY", "")
        return key

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        api_key = self._get_api_key()
        if not api_key:
            return []

        min_forecasters = getattr(self._config, "metaculus_min_forecasters", 20)
        min_jaccard = getattr(self._config, "metaculus_min_jaccard", 0.60)

        # Build search query from question keywords
        search_terms = self._extract_search_terms(question)
        if not search_terms:
            return []

        await rate_limiter.get("metaculus").acquire()

        client = self._get_client()
        headers = {
            "Accept": "application/json",
            "Authorization": f"Token {api_key}",
            "User-Agent": "PolymarketBot/1.0 (research; +https://github.com)",
        }
        resp = await client.get(
            _METACULUS_API,
            params={
                "search": search_terms,
                "status": "open",
                "limit": 5,
                "type": "forecast",
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return []

        # Find best match by Jaccard similarity
        q_tokens = self._tokenize(question)
        best_match = None
        best_jaccard = 0.0

        for result in results:
            title = result.get("title", "")
            title_tokens = self._tokenize(title)
            jaccard = self._jaccard_similarity(q_tokens, title_tokens)
            if jaccard >= min_jaccard and jaccard > best_jaccard:
                best_jaccard = jaccard
                best_match = result

        if best_match is None:
            return []

        # Extract community prediction
        prediction = best_match.get("community_prediction", {})
        full_pred = prediction.get("full", {})
        probability = full_pred.get("q2")  # median

        if probability is None:
            return []

        forecasters = best_match.get("number_of_forecasters", 0)
        if forecasters < min_forecasters:
            log.debug(
                "metaculus.below_min_forecasters",
                forecasters=forecasters,
                min_required=min_forecasters,
            )
            return []

        title = best_match.get("title", "Unknown")
        question_id = best_match.get("id", 0)

        content_lines = [
            f"Metaculus Expert Forecast",
            f"Question: {title}",
            f"Community prediction (median): {probability:.1%}",
            f"Forecasters: {forecasters:,}",
            f"Match confidence (Jaccard): {best_jaccard:.2f}",
            f"Source: Metaculus (calibration Brier ≈ 0.111)",
        ]
        content = "\n".join(content_lines)
        snippet = f"Metaculus: {probability:.1%} ({forecasters:,} forecasters)"

        return [self._make_source(
            title=f"Metaculus: {title}",
            url=f"https://www.metaculus.com/questions/{question_id}/",
            snippet=snippet,
            publisher="Metaculus",
            date="",
            content=content,
            authority_score=0.95,
            raw={
                "consensus_signal": {
                    "platform": "metaculus",
                    "price": round(probability, 4),
                    "forecasters": forecasters,
                    "confidence": round(best_jaccard, 2),
                },
            },
        )]

    def _extract_search_terms(self, question: str) -> str:
        """Extract key search terms from question."""
        tokens = self._tokenize(question)
        return " ".join(list(tokens)[:6])

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into meaningful words."""
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return {w for w in words if w not in _STOP_WORDS and len(w) > 1}

    def _jaccard_similarity(self, a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0
