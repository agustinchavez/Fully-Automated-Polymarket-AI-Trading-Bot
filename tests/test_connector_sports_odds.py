"""Tests for SportsOddsConnector — sportsbook consensus probabilities."""

from __future__ import annotations

import pytest

from src.research.connectors.sports_odds import SportsOddsConnector


# ── Fixtures ───────────────────────────────────────────────────────

def _make_connector(api_key: str = "test-key", **kw):
    """Create a connector with a minimal config stub."""
    class _Cfg:
        sports_odds_api_key = api_key
        sports_book_weights = {
            "pinnacle": 0.40, "betfair": 0.30,
            "draftkings": 0.20, "fanduel": 0.10,
        }
        sports_min_books = 2
    for k, v in kw.items():
        setattr(_Cfg, k, v)
    return SportsOddsConnector(_Cfg())


def _make_event(
    home: str = "Team A",
    away: str = "Team B",
    bookmakers: list | None = None,
) -> dict:
    if bookmakers is None:
        bookmakers = [
            _make_bookmaker("pinnacle", home, 1.80, away, 2.10),
            _make_bookmaker("draftkings", home, 1.75, away, 2.15),
        ]
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": bookmakers,
    }


def _make_bookmaker(key: str, home: str, home_odds: float, away: str, away_odds: float) -> dict:
    return {
        "key": key,
        "markets": [{
            "key": "h2h",
            "outcomes": [
                {"name": home, "price": home_odds},
                {"name": away, "price": away_odds},
            ],
        }],
    }


# ═══════════════════════════════════════════════════════════════════
#  Sport Key Extraction
# ═══════════════════════════════════════════════════════════════════


class TestExtractSportKey:
    def test_nfl(self):
        assert SportsOddsConnector._extract_sport_key("Will the NFL game end in OT?") == "americanfootball_nfl"

    def test_nba(self):
        assert SportsOddsConnector._extract_sport_key("NBA Finals winner?") == "basketball_nba"

    def test_premier_league(self):
        assert SportsOddsConnector._extract_sport_key("Will Arsenal win the Premier League?") == "soccer_epl"

    def test_champions_league(self):
        assert SportsOddsConnector._extract_sport_key("Champions League final winner?") == "soccer_uefa_champions_league"

    def test_la_liga(self):
        assert SportsOddsConnector._extract_sport_key("La Liga match: Barcelona vs Real Madrid") == "soccer_spain_la_liga"

    def test_serie_a(self):
        assert SportsOddsConnector._extract_sport_key("Serie A: Napoli vs Juventus") == "soccer_italy_serie_a"

    def test_ufc(self):
        assert SportsOddsConnector._extract_sport_key("UFC 300 main event winner?") == "mma_mixed_martial_arts"

    def test_f1(self):
        assert SportsOddsConnector._extract_sport_key("F1 Monaco GP winner?") == "motorsport_formula_one"

    def test_no_match(self):
        assert SportsOddsConnector._extract_sport_key("Will Bitcoin reach $100k?") is None

    def test_case_insensitive(self):
        assert SportsOddsConnector._extract_sport_key("Who will win the NFL?") == "americanfootball_nfl"


# ═══════════════════════════════════════════════════════════════════
#  Team Extraction
# ═══════════════════════════════════════════════════════════════════


class TestExtractTeams:
    def test_vs_pattern(self):
        teams = SportsOddsConnector._extract_teams("Lakers vs Celtics game?")
        assert len(teams) == 2
        assert "Lakers" in teams[0]
        assert "Celtics" in teams[1]

    def test_versus_pattern(self):
        teams = SportsOddsConnector._extract_teams("Lakers versus Celtics?")
        assert len(teams) == 2

    def test_win_pattern(self):
        teams = SportsOddsConnector._extract_teams("Will the Lakers win?")
        assert len(teams) >= 1
        assert "Lakers" in teams[0]

    def test_no_teams(self):
        teams = SportsOddsConnector._extract_teams("What is the weather?")
        assert teams == []


# ═══════════════════════════════════════════════════════════════════
#  Vig Removal
# ═══════════════════════════════════════════════════════════════════


class TestRemoveVig:
    def test_fair_odds_sum_to_one(self):
        """Fair probabilities must sum to 1.0 after vig removal."""
        fair = SportsOddsConnector._remove_vig([1.80, 2.10])
        assert len(fair) == 2
        assert abs(sum(fair) - 1.0) < 1e-10

    def test_even_odds(self):
        """Equal odds (2.0, 2.0) → 50/50 after vig removal."""
        fair = SportsOddsConnector._remove_vig([2.0, 2.0])
        assert abs(fair[0] - 0.5) < 1e-10
        assert abs(fair[1] - 0.5) < 1e-10

    def test_heavy_favorite(self):
        """1.20 vs 5.00 → strong favorite ~80%."""
        fair = SportsOddsConnector._remove_vig([1.20, 5.00])
        assert fair[0] > 0.75
        assert fair[1] < 0.25

    def test_three_way(self):
        """Three-way market (home/draw/away)."""
        fair = SportsOddsConnector._remove_vig([2.50, 3.20, 3.00])
        assert len(fair) == 3
        assert abs(sum(fair) - 1.0) < 1e-10

    def test_invalid_odds(self):
        """Odds <= 1.0 are invalid."""
        assert SportsOddsConnector._remove_vig([0.5, 2.0]) == []

    def test_empty(self):
        assert SportsOddsConnector._remove_vig([]) == []


# ═══════════════════════════════════════════════════════════════════
#  Event Matching
# ═══════════════════════════════════════════════════════════════════


class TestMatchEvent:
    def test_exact_team_match(self):
        events = [_make_event("Lakers", "Celtics")]
        result = SportsOddsConnector._match_event(events, ["Lakers", "Celtics"], "")
        assert result is not None
        assert result["home_team"] == "Lakers"

    def test_partial_team_match(self):
        events = [_make_event("Los Angeles Lakers", "Boston Celtics")]
        result = SportsOddsConnector._match_event(events, ["Lakers"], "")
        assert result is not None

    def test_no_match(self):
        events = [_make_event("Lakers", "Celtics")]
        result = SportsOddsConnector._match_event(events, ["Warriors"], "")
        assert result is None

    def test_no_teams_fallback_question(self):
        events = [_make_event("Lakers", "Celtics")]
        result = SportsOddsConnector._match_event(events, [], "Will the Lakers win?")
        assert result is not None

    def test_empty_events(self):
        result = SportsOddsConnector._match_event([], ["Lakers"], "")
        assert result is None


# ═══════════════════════════════════════════════════════════════════
#  Consensus Computation
# ═══════════════════════════════════════════════════════════════════


class TestComputeConsensus:
    def test_two_books(self):
        event = _make_event("Team A", "Team B", [
            _make_bookmaker("pinnacle", "Team A", 1.80, "Team B", 2.10),
            _make_bookmaker("draftkings", "Team A", 1.75, "Team B", 2.15),
        ])
        result = SportsOddsConnector._compute_consensus(
            event,
            {"pinnacle": 0.40, "draftkings": 0.20},
            min_books=2,
        )
        assert result is not None
        prob, spread_pp, n_books, sharp, summary = result
        assert 0.0 < prob < 1.0
        assert n_books == 2
        assert spread_pp >= 0
        assert sharp is not None  # pinnacle present

    def test_min_books_gate(self):
        """Should return None if fewer books than min_books."""
        event = _make_event("Team A", "Team B", [
            _make_bookmaker("pinnacle", "Team A", 1.80, "Team B", 2.10),
        ])
        result = SportsOddsConnector._compute_consensus(
            event,
            {"pinnacle": 0.40},
            min_books=2,
        )
        assert result is None

    def test_no_bookmakers(self):
        event = {"home_team": "A", "away_team": "B", "bookmakers": []}
        assert SportsOddsConnector._compute_consensus(event, {}, 1) is None

    def test_sharp_book_identified(self):
        event = _make_event("Team A", "Team B", [
            _make_bookmaker("pinnacle", "Team A", 1.80, "Team B", 2.10),
            _make_bookmaker("betfair", "Team A", 1.85, "Team B", 2.05),
        ])
        result = SportsOddsConnector._compute_consensus(
            event,
            {"pinnacle": 0.40, "betfair": 0.30},
            min_books=2,
        )
        assert result is not None
        _, _, _, sharp, _ = result
        assert sharp is not None
        # Pinnacle at 1.80 → implied ~55.6%, after vig removal ~56.8%
        assert 0.50 < sharp < 0.65

    def test_unknown_book_gets_default_weight(self):
        event = _make_event("Team A", "Team B", [
            _make_bookmaker("pinnacle", "Team A", 2.00, "Team B", 2.00),
            _make_bookmaker("unknown_book", "Team A", 2.00, "Team B", 2.00),
        ])
        result = SportsOddsConnector._compute_consensus(
            event,
            {"pinnacle": 0.40},
            min_books=2,
        )
        assert result is not None
        prob, _, _, _, _ = result
        assert abs(prob - 0.5) < 0.01  # even odds → ~50%


# ═══════════════════════════════════════════════════════════════════
#  Connector Identity
# ═══════════════════════════════════════════════════════════════════


class TestConnectorIdentity:
    def test_name(self):
        c = _make_connector()
        assert c.name == "sports_odds"

    def test_relevant_categories(self):
        c = _make_connector()
        assert c.relevant_categories() == {"SPORTS"}

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        c = _make_connector(api_key="")
        result = await c._fetch_impl("NFL game?", "SPORTS")
        assert result == []
