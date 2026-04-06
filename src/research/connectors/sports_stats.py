"""Sports stats research connector — team form, H2H, and context.

Uses API-Football (v3) to fetch recent form, head-to-head records, and
fixture details for soccer markets.  Non-soccer sports are not covered
by this API, so the connector gracefully returns [].

The stats context is emitted as a ``behavioral_signal`` in
``FetchedSource.raw`` so it flows into the SignalStack and LLM prompt.

Free tier: 100 requests/day, no credit card required.
https://www.api-football.com/

API call budget per market:
  - Cached team ID: 2 calls (fixtures + H2H)
  - Uncached team ID: 3 calls (team search + fixtures + H2H)
  - Form is derived from H2H results to avoid 2 extra /statistics calls.
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

_API_BASE = "https://v3.football.api-sports.io"

# Keywords that indicate soccer content
_SOCCER_KEYWORDS: list[str] = [
    "soccer", "football match", "serie a", "serie b", "calcio",
    "ligue 1", "bundesliga", "eredivisie", "primeira liga",
    "la liga", "el clasico", "mls", "liga mx", "premier league",
    "champions league", "europa league", "fc", "match draw",
]

# ── Team extraction regex ────────────────────────────────────────────
_TEAM_RE = re.compile(
    r"(?:will\s+)?(.+?)\s+(?:vs\.?|v\.?|versus)\s+(.+?)(?:\s+(?:win|lose|draw|match|game|in\b)|\?|$)",
    re.IGNORECASE,
)


class SportsStatsConnector(BaseResearchConnector):
    """Fetch soccer team form and H2H from API-Football."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        # Persistent cache: normalized team name → API-Football team ID
        self._team_id_cache: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "sports_stats"

    def relevant_categories(self) -> set[str]:
        return {"SPORTS"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        """Only relevant for soccer-related markets."""
        if market_type not in self.relevant_categories():
            return False
        return self._question_matches_keywords(question, _SOCCER_KEYWORDS)

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        key = getattr(self._config, "sports_stats_api_key", "") or ""
        if not key:
            key = os.environ.get("API_FOOTBALL_KEY", "")
        return key

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        api_key = self._get_api_key()
        if not api_key:
            return []

        if not self.is_relevant(question, market_type):
            return []

        teams = self._extract_teams(question)
        if not teams:
            return []

        client = self._get_client()
        headers = {"x-apisports-key": api_key}

        # Resolve team ID (cached or via API)
        search_team = teams[0]
        team_id = await self._resolve_team_id(search_team, client, headers)
        if not team_id:
            return []

        # Fetch upcoming fixtures for this team (1 API call)
        await rate_limiter.get("sports_stats").acquire()
        resp = await client.get(
            f"{_API_BASE}/fixtures",
            params={"team": team_id, "next": 5},
            headers=headers,
        )
        resp.raise_for_status()
        fixtures = resp.json().get("response", [])

        # Try to find fixture matching the away team too
        matched_fixture = None
        away_team_id = None
        if len(teams) >= 2:
            away_search = teams[1].lower()
            for fix in fixtures:
                home = fix.get("teams", {}).get("home", {})
                away = fix.get("teams", {}).get("away", {})
                home_name = home.get("name", "").lower()
                away_name = away.get("name", "").lower()
                if away_search in home_name or away_search in away_name:
                    matched_fixture = fix
                    if away_search in away_name:
                        away_team_id = away.get("id")
                    else:
                        away_team_id = home.get("id")
                    break

        if matched_fixture is None and fixtures:
            matched_fixture = fixtures[0]
            away_info = matched_fixture.get("teams", {}).get("away", {})
            away_team_id = away_info.get("id")

        if matched_fixture is None:
            return []

        # Extract fixture context
        home_info = matched_fixture.get("teams", {}).get("home", {})
        away_info = matched_fixture.get("teams", {}).get("away", {})
        home_name = home_info.get("name", "Home")
        away_name = away_info.get("name", "Away")
        home_team_id = home_info.get("id")
        league_info = matched_fixture.get("league", {})
        league_name = league_info.get("name", "")

        # Fetch H2H — also derives form from recent results (1 API call)
        h2h_record = {"home_wins": 0, "draws": 0, "away_wins": 0}
        home_form = ""
        away_form = ""
        if home_team_id and away_team_id:
            try:
                await rate_limiter.get("sports_stats").acquire()
                resp = await client.get(
                    f"{_API_BASE}/fixtures/headtohead",
                    params={"h2h": f"{home_team_id}-{away_team_id}", "last": 5},
                    headers=headers,
                )
                resp.raise_for_status()
                h2h_fixtures = resp.json().get("response", [])
                h2h_record = self._parse_h2h(h2h_fixtures, home_team_id)
                # Derive form from H2H results (avoids 2 extra API calls)
                home_form = self._derive_form(h2h_fixtures, home_team_id)
                away_form = self._derive_form(h2h_fixtures, away_team_id)
            except Exception as e:
                log.debug("sports_stats.h2h_failed", error=str(e))

        # Compute form score (W=3, D=1, L=0)
        form_score = self._compute_form_score(home_form)

        # Build content summary
        lines = [f"Sports Context: {home_name} vs {away_name}"]
        if league_name:
            lines.append(f"League: {league_name}")
        if home_form:
            lines.append(f"Home form (last {len(home_form)}): {home_form}")
        if away_form:
            lines.append(f"Away form (last {len(away_form)}): {away_form}")
        h2h_total = h2h_record["home_wins"] + h2h_record["draws"] + h2h_record["away_wins"]
        if h2h_total > 0:
            lines.append(
                f"H2H (last {h2h_total}): {home_name} {h2h_record['home_wins']}W "
                f"{h2h_record['draws']}D {h2h_record['away_wins']}L"
            )

        content = "\n".join(lines)

        return [
            self._make_source(
                title=f"Sports Context: {home_name} vs {away_name}",
                url="https://www.api-football.com/",
                snippet=f"Form: {home_form or 'N/A'} vs {away_form or 'N/A'}, H2H: {h2h_record['home_wins']}-{h2h_record['draws']}-{h2h_record['away_wins']}",
                publisher="API-Football",
                content=content,
                raw={
                    "behavioral_signal": {
                        "source": "sports_stats",
                        "signal_type": "sports_context",
                        "value": form_score,
                        "home_form": home_form,
                        "away_form": away_form,
                        "h2h_home_wins": h2h_record["home_wins"],
                        "h2h_draws": h2h_record["draws"],
                        "h2h_away_wins": h2h_record["away_wins"],
                    },
                },
            )
        ]

    async def _resolve_team_id(
        self,
        team_name: str,
        client: Any,
        headers: dict[str, str],
    ) -> int | None:
        """Resolve team name to API-Football team ID, with persistent cache."""
        cache_key = team_name.lower().strip()
        if cache_key in self._team_id_cache:
            return self._team_id_cache[cache_key]

        await rate_limiter.get("sports_stats").acquire()
        resp = await client.get(
            f"{_API_BASE}/teams",
            params={"search": team_name},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        team_results = data.get("response", [])
        if not team_results:
            return None

        team_id = team_results[0].get("team", {}).get("id")
        if team_id:
            self._team_id_cache[cache_key] = team_id
        return team_id

    @staticmethod
    def _extract_teams(question: str) -> list[str]:
        """Extract team names from question text."""
        m = _TEAM_RE.search(question)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        return []

    @staticmethod
    def _parse_h2h(
        fixtures: list[dict[str, Any]],
        home_team_id: int,
    ) -> dict[str, int]:
        """Parse H2H fixture results into win/draw/loss record."""
        record = {"home_wins": 0, "draws": 0, "away_wins": 0}
        for fix in fixtures:
            goals = fix.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            if home_goals is None or away_goals is None:
                continue

            teams = fix.get("teams", {})
            fixture_home_id = teams.get("home", {}).get("id")

            if home_goals == away_goals:
                record["draws"] += 1
            elif fixture_home_id == home_team_id:
                if home_goals > away_goals:
                    record["home_wins"] += 1
                else:
                    record["away_wins"] += 1
            else:
                # Teams are swapped in this fixture
                if home_goals > away_goals:
                    record["away_wins"] += 1
                else:
                    record["home_wins"] += 1

        return record

    @staticmethod
    def _derive_form(
        fixtures: list[dict[str, Any]],
        team_id: int,
    ) -> str:
        """Derive W/D/L form string from H2H fixtures for a given team."""
        form_chars: list[str] = []
        for fix in fixtures:
            goals = fix.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            if home_goals is None or away_goals is None:
                continue

            teams = fix.get("teams", {})
            fixture_home_id = teams.get("home", {}).get("id")
            is_home = fixture_home_id == team_id

            if home_goals == away_goals:
                form_chars.append("D")
            elif is_home:
                form_chars.append("W" if home_goals > away_goals else "L")
            else:
                form_chars.append("W" if away_goals > home_goals else "L")

        return "".join(form_chars)

    @staticmethod
    def _compute_form_score(form: str) -> float:
        """Convert form string (e.g. 'WWDLW') to numeric score.

        W=3, D=1, L=0.  Normalized to [0, 1] range.
        """
        if not form:
            return 0.5  # neutral default
        points = 0
        total = 0
        for ch in form.upper():
            if ch == "W":
                points += 3
                total += 3
            elif ch == "D":
                points += 1
                total += 3
            elif ch == "L":
                total += 3
        if total == 0:
            return 0.5
        return points / total
