"""Tests for SportsStatsConnector — soccer form, H2H, and context."""

from __future__ import annotations

import pytest

from src.research.connectors.sports_stats import SportsStatsConnector


# ── Fixtures ───────────────────────────────────────────────────────

def _make_connector(api_key: str = "test-key"):
    class _Cfg:
        sports_stats_api_key = api_key
    return SportsStatsConnector(_Cfg())


# ═══════════════════════════════════════════════════════════════════
#  Relevance Checks
# ═══════════════════════════════════════════════════════════════════


class TestRelevance:
    def test_soccer_relevant(self):
        c = _make_connector()
        assert c.is_relevant("Serie A: Napoli vs Juventus?", "SPORTS") is True

    def test_premier_league_relevant(self):
        c = _make_connector()
        assert c.is_relevant("Will Arsenal win the Premier League?", "SPORTS") is True

    def test_nfl_not_relevant(self):
        c = _make_connector()
        assert c.is_relevant("NFL Super Bowl winner?", "SPORTS") is False

    def test_nba_not_relevant(self):
        c = _make_connector()
        assert c.is_relevant("NBA Finals MVP?", "SPORTS") is False

    def test_non_sports_not_relevant(self):
        c = _make_connector()
        assert c.is_relevant("Will Bitcoin reach $100k?", "CRYPTO") is False

    def test_fc_keyword(self):
        c = _make_connector()
        assert c.is_relevant("Barcelona FC vs Real Madrid FC?", "SPORTS") is True

    def test_match_draw(self):
        c = _make_connector()
        assert c.is_relevant("Will the match draw?", "SPORTS") is True


# ═══════════════════════════════════════════════════════════════════
#  Team Extraction
# ═══════════════════════════════════════════════════════════════════


class TestExtractTeams:
    def test_vs_pattern(self):
        teams = SportsStatsConnector._extract_teams("Barcelona vs Real Madrid?")
        assert len(teams) == 2
        assert "Barcelona" in teams[0]
        assert "Real Madrid" in teams[1]

    def test_no_teams(self):
        teams = SportsStatsConnector._extract_teams("Will the Premier League season end early?")
        assert teams == []


# ═══════════════════════════════════════════════════════════════════
#  H2H Parsing
# ═══════════════════════════════════════════════════════════════════


class TestParseH2H:
    def test_basic_record(self):
        fixtures = [
            {"goals": {"home": 2, "away": 1}, "teams": {"home": {"id": 10}, "away": {"id": 20}}},
            {"goals": {"home": 1, "away": 1}, "teams": {"home": {"id": 10}, "away": {"id": 20}}},
            {"goals": {"home": 0, "away": 3}, "teams": {"home": {"id": 10}, "away": {"id": 20}}},
        ]
        record = SportsStatsConnector._parse_h2h(fixtures, home_team_id=10)
        assert record["home_wins"] == 1
        assert record["draws"] == 1
        assert record["away_wins"] == 1

    def test_swapped_teams(self):
        """When home team plays away in the fixture, results should still be correct."""
        fixtures = [
            # Team 10 is AWAY here but scored more
            {"goals": {"home": 1, "away": 2}, "teams": {"home": {"id": 20}, "away": {"id": 10}}},
        ]
        record = SportsStatsConnector._parse_h2h(fixtures, home_team_id=10)
        assert record["home_wins"] == 1
        assert record["away_wins"] == 0

    def test_missing_goals(self):
        """Fixtures with None goals should be skipped."""
        fixtures = [
            {"goals": {"home": None, "away": None}, "teams": {"home": {"id": 10}, "away": {"id": 20}}},
        ]
        record = SportsStatsConnector._parse_h2h(fixtures, home_team_id=10)
        assert record == {"home_wins": 0, "draws": 0, "away_wins": 0}

    def test_empty_fixtures(self):
        record = SportsStatsConnector._parse_h2h([], home_team_id=10)
        assert record == {"home_wins": 0, "draws": 0, "away_wins": 0}


# ═══════════════════════════════════════════════════════════════════
#  Form Score
# ═══════════════════════════════════════════════════════════════════


class TestFormScore:
    def test_all_wins(self):
        assert SportsStatsConnector._compute_form_score("WWWWW") == 1.0

    def test_all_losses(self):
        assert SportsStatsConnector._compute_form_score("LLLLL") == 0.0

    def test_all_draws(self):
        score = SportsStatsConnector._compute_form_score("DDDDD")
        assert abs(score - 1 / 3) < 0.01

    def test_mixed(self):
        # WWDLW = 3+3+1+0+3 = 10 out of 15
        score = SportsStatsConnector._compute_form_score("WWDLW")
        assert abs(score - 10 / 15) < 0.01

    def test_empty_form(self):
        assert SportsStatsConnector._compute_form_score("") == 0.5

    def test_case_insensitive(self):
        assert SportsStatsConnector._compute_form_score("wwdlw") == SportsStatsConnector._compute_form_score("WWDLW")


# ═══════════════════════════════════════════════════════════════════
#  Connector Identity
# ═══════════════════════════════════════════════════════════════════


class TestConnectorIdentity:
    def test_name(self):
        c = _make_connector()
        assert c.name == "sports_stats"

    def test_relevant_categories(self):
        c = _make_connector()
        assert c.relevant_categories() == {"SPORTS"}

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        c = _make_connector(api_key="")
        result = await c._fetch_impl("Serie A: Napoli vs Juventus?", "SPORTS")
        assert result == []

    @pytest.mark.asyncio
    async def test_non_soccer_returns_empty(self):
        c = _make_connector()
        result = await c._fetch_impl("NFL Super Bowl winner?", "SPORTS")
        assert result == []
