"""Sports odds research connector — sportsbook consensus prices.

Uses The Odds API (v4) to fetch real-time odds from multiple sportsbooks,
removes vig to compute fair probabilities, and produces a weighted consensus
price.  The consensus is emitted as a ``consensus_signal`` in
``FetchedSource.raw`` so it flows into the SignalStack and LLM prompt.

Free tier: 500 requests/month, no credit card required.
https://the-odds-api.com/
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_CACHE_TTL_SECS = 600  # 10-minute cache per sport key

_API_BASE = "https://api.the-odds-api.com/v4/sports"

# ── Sport key mapping ────────────────────────────────────────────────
# Maps question keywords → The Odds API sport key.
_SPORT_KEYWORDS: dict[str, str] = {
    "nfl": "americanfootball_nfl",
    "super bowl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "world series": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "stanley cup": "icehockey_nhl",
    "premier league": "soccer_epl",
    "epl": "soccer_epl",
    "champions league": "soccer_uefa_champions_league",
    "la liga": "soccer_spain_la_liga",
    "serie a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue 1": "soccer_france_ligue_one",
    "mls": "soccer_usa_mls",
    "liga mx": "soccer_mexico_ligamx",
    "world cup": "soccer_fifa_world_cup",
    "ufc": "mma_mixed_martial_arts",
    "mma": "mma_mixed_martial_arts",
    "boxing": "boxing_boxing",
    "f1": "motorsport_formula_one",
    "formula 1": "motorsport_formula_one",
    "nascar": "motorsport_nascar",
}

# ── Team extraction regex ────────────────────────────────────────────
_TEAM_RE = re.compile(
    r"(?:will\s+)?(.+?)\s+(?:vs\.?|v\.?|versus)\s+(.+?)(?:\s+(?:win|lose|draw|match|game|in\b)|\?|$)",
    re.IGNORECASE,
)

# Simple fallback: "Will X win ..."
_TEAM_WIN_RE = re.compile(
    r"(?:will\s+)?(?:the\s+)?(.+?)\s+win\b",
    re.IGNORECASE,
)


class SportsOddsConnector(BaseResearchConnector):
    """Fetch sportsbook odds and compute vig-free consensus probability."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        # Cache: sport_key → (timestamp, events_json)
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    @property
    def name(self) -> str:
        return "sports_odds"

    def relevant_categories(self) -> set[str]:
        return {"SPORTS"}

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        key = getattr(self._config, "sports_odds_api_key", "") or ""
        if not key:
            key = os.environ.get("ODDS_API_KEY", "")
        return key

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        api_key = self._get_api_key()
        if not api_key:
            return []

        sport_key = self._extract_sport_key(question)
        if not sport_key:
            return []

        teams = self._extract_teams(question)

        # Check cache before making HTTP request
        events = self._get_cached(sport_key)
        if events is None:
            await rate_limiter.get("sports_odds").acquire()

            client = self._get_client()
            resp = await client.get(
                f"{_API_BASE}/{sport_key}/odds/",
                params={
                    "apiKey": api_key,
                    "regions": "us,eu",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
            )
            resp.raise_for_status()
            events = resp.json()
            self._set_cached(sport_key, events)

        if not events:
            return []

        # Find matching event
        matched_event = self._match_event(events, teams, question)
        if not matched_event:
            return []

        # Compute consensus
        book_weights = getattr(self._config, "sports_book_weights", {}) or {
            "pinnacle": 0.40, "betfair": 0.30,
            "draftkings": 0.20, "fanduel": 0.10,
        }
        min_books = getattr(self._config, "sports_min_books", 2)

        result = self._compute_consensus(matched_event, book_weights, min_books)
        if result is None:
            return []

        consensus_prob, spread_pp, n_books, sharp_prob, summary = result

        return [
            self._make_source(
                title=f"Sportsbook Consensus: {matched_event.get('home_team', '?')} vs {matched_event.get('away_team', '?')}",
                url=f"https://the-odds-api.com/sports/{sport_key}/",
                snippet=f"Consensus: {consensus_prob:.1%} ({n_books} books, spread {spread_pp:.1f}pp)",
                publisher="The Odds API",
                content=summary,
                raw={
                    "consensus_signal": {
                        "platform": "sportsbooks",
                        "price": round(consensus_prob, 4),
                        "spread_pp": round(spread_pp, 1),
                        "books": n_books,
                        "sharp_book": round(sharp_prob, 4) if sharp_prob is not None else None,
                    },
                },
            )
        ]

    def _get_cached(self, sport_key: str) -> list[dict[str, Any]] | None:
        """Return cached events if still within TTL, else None."""
        entry = self._cache.get(sport_key)
        if entry is None:
            return None
        ts, events = entry
        if time.monotonic() - ts > _CACHE_TTL_SECS:
            del self._cache[sport_key]
            return None
        return events

    def _set_cached(self, sport_key: str, events: list[dict[str, Any]]) -> None:
        """Store events in cache with current timestamp."""
        self._cache[sport_key] = (time.monotonic(), events)

    @staticmethod
    def _extract_sport_key(question: str) -> str | None:
        """Map question keywords to The Odds API sport key."""
        q_lower = question.lower()
        # Try longest keywords first to avoid partial matches
        for keyword in sorted(_SPORT_KEYWORDS, key=len, reverse=True):
            if keyword in q_lower:
                return _SPORT_KEYWORDS[keyword]
        return None

    @staticmethod
    def _extract_teams(question: str) -> list[str]:
        """Extract team names from question text."""
        m = _TEAM_RE.search(question)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        m = _TEAM_WIN_RE.search(question)
        if m:
            return [m.group(1).strip()]
        return []

    @staticmethod
    def _match_event(
        events: list[dict[str, Any]],
        teams: list[str],
        question: str,
    ) -> dict[str, Any] | None:
        """Find the best matching event from the API response."""
        if not teams:
            # Try matching by any word overlap with event teams
            q_lower = question.lower()
            for event in events:
                home = event.get("home_team", "").lower()
                away = event.get("away_team", "").lower()
                if home and home in q_lower or away and away in q_lower:
                    return event
            return None

        for event in events:
            home = event.get("home_team", "").lower()
            away = event.get("away_team", "").lower()
            matched = 0
            for team in teams:
                t = team.lower()
                if t in home or t in away or home in t or away in t:
                    matched += 1
            if matched >= 1:
                return event

        return None

    @staticmethod
    def _remove_vig(decimal_odds: list[float]) -> list[float]:
        """Remove bookmaker vig from decimal odds to get fair probabilities.

        Formula: implied[i] = 1 / odds[i]; fair[i] = implied[i] / sum(implied)
        """
        if not decimal_odds or any(o <= 1.0 for o in decimal_odds):
            return []
        implied = [1.0 / o for o in decimal_odds]
        total = sum(implied)
        if total == 0:
            return []
        return [p / total for p in implied]

    @staticmethod
    def _compute_consensus(
        event: dict[str, Any],
        book_weights: dict[str, float],
        min_books: int,
    ) -> tuple[float, float, int, float | None, str] | None:
        """Compute weighted consensus probability from bookmaker odds.

        Returns (consensus_prob, spread_pp, n_books, sharp_prob, summary)
        or None if insufficient data.
        """
        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return None

        home_team = event.get("home_team", "Unknown")
        away_team = event.get("away_team", "Unknown")

        # Extract fair home-win probability from each bookmaker
        book_probs: list[tuple[str, float]] = []
        sharp_prob: float | None = None

        for bm in bookmakers:
            bm_key = bm.get("key", "").lower()
            markets = bm.get("markets", [])

            for market in markets:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                if len(outcomes) < 2:
                    continue

                # Get decimal odds for all outcomes
                decimal_odds = [o.get("price", 0.0) for o in outcomes]
                fair_probs = SportsOddsConnector._remove_vig(decimal_odds)
                if not fair_probs:
                    continue

                # Find home team probability (first outcome is typically home)
                home_prob = fair_probs[0]
                for i, o in enumerate(outcomes):
                    if o.get("name", "").lower() == home_team.lower():
                        home_prob = fair_probs[i]
                        break

                book_probs.append((bm_key, home_prob))
                if bm_key == "pinnacle":
                    sharp_prob = home_prob
                break

        if len(book_probs) < min_books:
            return None

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for bm_key, prob in book_probs:
            w = book_weights.get(bm_key, 0.05)  # default weight for unknown books
            weighted_sum += prob * w
            total_weight += w

        if total_weight == 0:
            return None

        consensus_prob = weighted_sum / total_weight
        probs_only = [p for _, p in book_probs]
        spread_pp = (max(probs_only) - min(probs_only)) * 100

        # Build summary
        lines = [
            f"Sportsbook Consensus: {home_team} vs {away_team}",
            f"Consensus home-win probability: {consensus_prob:.1%}",
            f"Books surveyed: {len(book_probs)}",
            f"Spread: {spread_pp:.1f} percentage points",
            "",
        ]
        for bm_key, prob in sorted(book_probs, key=lambda x: x[0]):
            lines.append(f"  {bm_key}: {prob:.1%}")
        if sharp_prob is not None:
            lines.append(f"\nPinnacle (sharp): {sharp_prob:.1%}")

        summary = "\n".join(lines)

        return consensus_prob, spread_pp, len(book_probs), sharp_prob, summary
