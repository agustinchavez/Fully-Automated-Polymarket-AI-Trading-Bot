"""Research analytics spec v2 fixes — config corrections, Reddit rewrite,
FRED expansion, continuous learning activation.

Tests cover:
  1. min_edge > total fees (zero-profit bug fix)
  2. daily_limit_usd raised to 25
  3. continuous_learning modules enabled
  4. Reddit connector uses public JSON API (no praw)
  5. Expanded subreddit map (SPORTS, GEOPOLITICS, CULTURE, WEATHER)
  6. Expanded FRED series (7 new economic indicators)
"""

from __future__ import annotations

import inspect

import pytest


# ── Fix 1: min_edge > total fees ───────────────────────────────────


class TestMinEdgeAboveFees:
    """min_edge must be strictly greater than total fees."""

    def test_min_edge_is_006(self) -> None:
        """config.yaml has min_edge: 0.06."""
        from src.config import load_config
        config = load_config()
        assert config.risk.min_edge == 0.06

    def test_min_edge_exceeds_total_fees(self) -> None:
        """min_edge > transaction_fee_pct + exit_fee_pct."""
        from src.config import load_config
        config = load_config()
        total_fees = config.risk.transaction_fee_pct + config.risk.exit_fee_pct
        assert config.risk.min_edge > total_fees

    def test_net_profit_floor_positive(self) -> None:
        """Net profit floor (min_edge - total_fees) is at least 2%."""
        from src.config import load_config
        config = load_config()
        total_fees = config.risk.transaction_fee_pct + config.risk.exit_fee_pct
        net_floor = config.risk.min_edge - total_fees
        assert net_floor >= 0.02 - 1e-9


# ── Fix 2: daily_limit_usd = 25 ───────────────────────────────────


class TestDailyBudget:
    """Daily budget at 25 for headroom above gated cost."""

    def test_daily_limit_25(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.budget.daily_limit_usd == 25.0


# ── Fix 3: continuous_learning enabled ────────────────────────────


class TestContinuousLearningEnabled:
    """post_mortem, weekly_summary, evidence_tracking all enabled."""

    def test_post_mortem_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.post_mortem_enabled is True

    def test_weekly_summary_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.weekly_summary_enabled is True

    def test_evidence_tracking_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.evidence_tracking_enabled is True

    def test_smart_retrain_still_disabled(self) -> None:
        """smart_retrain needs 30 trades — stays disabled for now."""
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.smart_retrain_enabled is False

    def test_param_optimizer_still_disabled(self) -> None:
        """param_optimizer needs 30 trades — stays disabled for now."""
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.param_optimizer_enabled is False


# ── Item 4: Reddit uses public JSON API ────────────────────────────


class TestRedditPublicJsonApi:
    """Reddit connector uses public JSON API, no praw."""

    def test_no_praw_import(self) -> None:
        """Module does not reference _HAS_PRAW."""
        import src.research.connectors.reddit_sentiment as mod
        assert not hasattr(mod, "_HAS_PRAW")

    def test_uses_httpx_client(self) -> None:
        """_search_subreddits is async and uses the base class httpx client."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector
        source = inspect.getsource(RedditSentimentConnector._search_subreddits)
        assert "self._get_client" in source
        assert "search.json" in source

    def test_no_asyncio_to_thread(self) -> None:
        """_fetch_impl no longer uses asyncio.to_thread (was for praw sync calls)."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector
        source = inspect.getsource(RedditSentimentConnector._fetch_impl)
        assert "asyncio.to_thread" not in source

    def test_no_client_id_check(self) -> None:
        """_fetch_impl does not check for REDDIT_CLIENT_ID."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector
        source = inspect.getsource(RedditSentimentConnector._fetch_impl)
        assert "client_id" not in source

    def test_user_agent_set(self) -> None:
        """Module defines a User-Agent string."""
        from src.research.connectors.reddit_sentiment import _USER_AGENT
        assert "polymarket" in _USER_AGENT.lower()


# ── Item 5: Expanded subreddit map ─────────────────────────────────


class TestExpandedSubredditMap:
    """Subreddit map covers high-signal subs for all major categories."""

    def test_sports_has_nba_soccer_nfl(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        sports = _SUBREDDIT_MAP["SPORTS"]
        assert "nba" in sports
        assert "soccer" in sports
        assert "nfl" in sports

    def test_geopolitics_has_worldnews(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        geo = _SUBREDDIT_MAP["GEOPOLITICS"]
        assert "worldnews" in geo
        assert "geopolitics" in geo

    def test_culture_has_music_movies(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        culture = _SUBREDDIT_MAP["CULTURE"]
        assert "Music" in culture
        assert "movies" in culture

    def test_weather_has_weather_climatology(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        weather = _SUBREDDIT_MAP["WEATHER"]
        assert "weather" in weather
        assert "climatology" in weather

    def test_election_has_political_discussion(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        election = _SUBREDDIT_MAP["ELECTION"]
        assert "PoliticalDiscussion" in election

    def test_crypto_has_bitcoin_ethereum(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        crypto = _SUBREDDIT_MAP["CRYPTO"]
        assert "Bitcoin" in crypto
        assert "ethereum" in crypto

    def test_total_categories_with_subreddits(self) -> None:
        """At least 12 categories have dedicated subreddit lists."""
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert len(_SUBREDDIT_MAP) >= 12


# ── Item 10: Expanded FRED series ──────────────────────────────────


class TestExpandedFredSeries:
    """FRED connector maps 7 new economic series."""

    def test_pce_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will PCE inflation rise?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "PCEPI" in series_ids

    def test_manufacturing_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will manufacturing employment rise?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "MANEMP" in series_ids

    def test_oil_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will crude oil prices fall?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "DCOILWTICO" in series_ids

    def test_retail_sales_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will retail sales drop?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "MRTSSM44X72USS" in series_ids

    def test_jobless_claims_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will initial jobless claims rise?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "ICSA" in series_ids

    def test_money_supply_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will M2 money supply expand?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "M2SL" in series_ids

    def test_tbill_series_mapped(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will 3-month treasury bill rates rise?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "DTB3" in series_ids

    def test_total_series_count(self) -> None:
        """FRED connector has 15 keyword-series mappings (8 original + 7 new)."""
        from src.research.connectors.fred import _KEYWORD_SERIES
        assert len(_KEYWORD_SERIES) == 15

    def test_macro_keywords_include_new_terms(self) -> None:
        """_MACRO_KEYWORDS includes new terms for relevance matching."""
        from src.research.connectors.fred import _MACRO_KEYWORDS
        for term in ["pce", "manufacturing employment", "oil", "crude", "retail sales",
                     "jobless claims", "money supply", "t-bill"]:
            assert term in _MACRO_KEYWORDS

    def test_oil_question_is_relevant(self) -> None:
        """Oil/crude questions trigger FRED connector relevance."""
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        assert c.is_relevant("Will oil prices crash?", "UNKNOWN")
