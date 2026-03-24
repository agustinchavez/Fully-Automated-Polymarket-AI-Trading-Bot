"""Cross-platform market matcher — maps Kalshi tickers to Polymarket markets.

Three matching strategies, in priority order:
  1. Manual mappings (highest confidence, from config JSON)
  2. Keyword/entity matching with Jaccard similarity
  3. Title fuzzy matching (lowest confidence)

Only matches exceeding ``match_min_confidence`` are returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


# Stop words removed before entity extraction (shared with arbitrage.py)
_STOP_WORDS: set[str] = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "by", "before", "after",
    "this", "that", "or", "and", "not", "no", "yes", "?", "how",
    "what", "when", "where", "which", "who", "do", "does", "has",
    "have", "been", "being", "if", "it", "its", "than", "then",
    "can", "could", "would", "should", "may", "might",
}

# Numbers, dates, and named entities carry more matching weight
_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s*\d{1,2}\b|\b\d{4}\b",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


@dataclass
class MarketMatch:
    """A matched pair of markets across platforms."""
    polymarket_id: str
    polymarket_question: str
    kalshi_ticker: str
    kalshi_title: str
    match_method: str        # "manual" | "keyword" | "fuzzy"
    match_confidence: float  # 0.0–1.0
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class MarketMatcher:
    """Matches Kalshi markets to Polymarket markets."""

    def __init__(
        self,
        manual_mappings: dict[str, str] | None = None,
        min_confidence: float = 0.6,
    ):
        # {kalshi_ticker: polymarket_condition_id}
        self._manual = manual_mappings or {}
        self._min_confidence = min_confidence
        self._cache: dict[str, MarketMatch | None] = {}

    def find_matches(
        self,
        poly_markets: list[Any],
        kalshi_markets: list[Any],
    ) -> list[MarketMatch]:
        """Find overlapping markets between platforms.

        Args:
            poly_markets: list of GammaMarket (or any obj with .id, .question, .condition_id)
            kalshi_markets: list of KalshiMarket (or any obj with .ticker, .title)

        Returns:
            Sorted list of MarketMatch by confidence descending.
        """
        if not poly_markets or not kalshi_markets:
            return []

        matches: list[MarketMatch] = []
        matched_poly_ids: set[str] = set()
        matched_kalshi_tickers: set[str] = set()

        # Build poly lookup
        poly_by_id: dict[str, Any] = {}
        for pm in poly_markets:
            pid = getattr(pm, "condition_id", None) or getattr(pm, "id", "")
            poly_by_id[pid] = pm

        # Strategy 1: Manual mappings
        for kticker, pid in self._manual.items():
            kalshi_obj = next(
                (k for k in kalshi_markets if k.ticker == kticker), None,
            )
            poly_obj = poly_by_id.get(pid)
            if kalshi_obj and poly_obj:
                matches.append(MarketMatch(
                    polymarket_id=pid,
                    polymarket_question=getattr(poly_obj, "question", ""),
                    kalshi_ticker=kticker,
                    kalshi_title=getattr(kalshi_obj, "title", ""),
                    match_method="manual",
                    match_confidence=1.0,
                    category=getattr(poly_obj, "category", ""),
                ))
                matched_poly_ids.add(pid)
                matched_kalshi_tickers.add(kticker)

        # Strategy 2: Keyword/entity matching
        # Pre-compute entities for all markets
        poly_entities: list[tuple[Any, str, set[str]]] = []
        for pm in poly_markets:
            pid = getattr(pm, "condition_id", None) or getattr(pm, "id", "")
            if pid in matched_poly_ids:
                continue
            q = getattr(pm, "question", "")
            ents = self._extract_entities(q)
            if len(ents) >= 2:
                poly_entities.append((pm, pid, ents))

        for km in kalshi_markets:
            if km.ticker in matched_kalshi_tickers:
                continue
            k_ents = self._extract_entities(km.title)
            if len(k_ents) < 2:
                continue

            best_score = 0.0
            best_poly: Any = None
            best_pid = ""

            for pm, pid, p_ents in poly_entities:
                if pid in matched_poly_ids:
                    continue
                score = self._jaccard_similarity(p_ents, k_ents)
                if score > best_score:
                    best_score = score
                    best_poly = pm
                    best_pid = pid

            if best_score >= self._min_confidence and best_poly is not None:
                matches.append(MarketMatch(
                    polymarket_id=best_pid,
                    polymarket_question=getattr(best_poly, "question", ""),
                    kalshi_ticker=km.ticker,
                    kalshi_title=km.title,
                    match_method="keyword",
                    match_confidence=round(best_score, 3),
                    category=getattr(best_poly, "category", ""),
                ))
                matched_poly_ids.add(best_pid)
                matched_kalshi_tickers.add(km.ticker)

        if matches:
            log.info(
                "market_matcher.matches_found",
                total=len(matches),
                manual=sum(1 for m in matches if m.match_method == "manual"),
                keyword=sum(1 for m in matches if m.match_method == "keyword"),
            )

        return sorted(matches, key=lambda m: m.match_confidence, reverse=True)

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract meaningful entities from market text."""
        # Normalize
        normalized = text.lower().strip()
        normalized = re.sub(r"[^\w\s%$.]", " ", normalized)

        words = set(normalized.split()) - _STOP_WORDS

        # Also extract dates and numbers as entities
        dates = _DATE_RE.findall(text)
        numbers = _NUMBER_RE.findall(text)

        entities = words | {d.lower() for d in dates} | {n for n in numbers}
        # Remove very short tokens
        return {e for e in entities if len(e) >= 2}

    @staticmethod
    def _normalize_question(text: str) -> str:
        """Normalize question text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity between two entity sets."""
        if not a or not b:
            return 0.0
        intersection = a & b
        union = a | b
        return len(intersection) / len(union) if union else 0.0
