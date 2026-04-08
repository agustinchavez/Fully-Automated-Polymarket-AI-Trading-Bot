"""Tests for RedditSentimentConnector -- crowd sentiment from Reddit.

Rewritten for the public JSON API version (no praw dependency).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.source_fetcher import FetchedSource


# ── Helpers ───────────────────────────────────────────────────────────


def _make_post(
    title: str = "Test post",
    selftext: str = "",
    score: int = 10,
    upvote_ratio: float = 0.9,
    age_hours: float = 12.0,
    permalink: str = "/r/test/comments/abc/test_post",
    subreddit: str = "test",
) -> dict:
    """Build a post dict matching RedditSentimentConnector._search_subreddits output."""
    return {
        "title": title,
        "selftext": selftext,
        "score": score,
        "upvote_ratio": upvote_ratio,
        "created_utc": time.time() - (age_hours * 3600),
        "url": f"https://reddit.com{permalink}",
        "subreddit": subreddit,
    }


def _make_reddit_json_response(posts: list[dict]) -> dict:
    """Build a Reddit JSON API response structure."""
    children = []
    for post in posts:
        children.append({
            "kind": "t3",
            "data": {
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "score": post.get("score", 1),
                "upvote_ratio": post.get("upvote_ratio", 0.5),
                "created_utc": post.get("created_utc", time.time()),
                "permalink": post.get("permalink", "/r/test/comments/abc/post"),
            },
        })
    return {"data": {"children": children}}


# ── No praw dependency ─────────────────────────────────────────────


class TestNoPrawDependency:
    def test_no_praw_import(self) -> None:
        """Connector no longer imports praw."""
        import src.research.connectors.reddit_sentiment as mod
        assert not hasattr(mod, "_HAS_PRAW")
        assert "praw" not in dir(mod)

    def test_no_credentials_required(self) -> None:
        """Connector does not require REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET."""
        import inspect
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector
        source = inspect.getsource(RedditSentimentConnector._fetch_impl)
        assert "REDDIT_CLIENT_ID" not in source
        assert "client_secret" not in source


# ── Subreddit Routing ────────────────────────────────────────────────


class TestSubredditRouting:
    def test_election_routes_to_politics(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert "politics" in _SUBREDDIT_MAP["ELECTION"]
        assert "PoliticalDiscussion" in _SUBREDDIT_MAP["ELECTION"]

    def test_crypto_routes_to_cryptocurrency(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert "cryptocurrency" in _SUBREDDIT_MAP["CRYPTO"]
        assert "Bitcoin" in _SUBREDDIT_MAP["CRYPTO"]
        assert "ethereum" in _SUBREDDIT_MAP["CRYPTO"]

    def test_macro_routes_to_economics_investing(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert "economics" in _SUBREDDIT_MAP["MACRO"]
        assert "investing" in _SUBREDDIT_MAP["MACRO"]

    def test_science_routes_to_science(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["SCIENCE"] == ["science"]

    def test_corporate_routes_to_stocks(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["CORPORATE"] == ["stocks"]

    def test_legal_routes_to_law(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["LEGAL"] == ["law"]

    def test_technology_routes_to_technology(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["TECHNOLOGY"] == ["technology"]
        assert _SUBREDDIT_MAP["TECH"] == ["technology"]

    def test_fallback_for_unknown_category(self) -> None:
        from src.research.connectors.reddit_sentiment import (
            _SUBREDDIT_MAP,
            _DEFAULT_SUBREDDITS,
        )
        assert "WEIRD_CATEGORY" not in _SUBREDDIT_MAP
        assert _DEFAULT_SUBREDDITS == ["polymarket"]

    # ── New subreddit map entries ─────────────────────────────────

    def test_sports_subreddits(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        sports = _SUBREDDIT_MAP["SPORTS"]
        assert "nba" in sports
        assert "soccer" in sports
        assert "nfl" in sports

    def test_geopolitics_subreddits(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        geo = _SUBREDDIT_MAP["GEOPOLITICS"]
        assert "worldnews" in geo
        assert "geopolitics" in geo

    def test_culture_subreddits(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        culture = _SUBREDDIT_MAP["CULTURE"]
        assert "Music" in culture
        assert "movies" in culture

    def test_weather_subreddits(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        weather = _SUBREDDIT_MAP["WEATHER"]
        assert "weather" in weather
        assert "climatology" in weather


# ── Sentiment Scoring ────────────────────────────────────────────────


class TestSentimentScoring:
    def test_bullish_posts(self) -> None:
        """Posts with bullish keywords produce positive sentiment."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        posts = [
            _make_post(
                title="Bitcoin will rise and gain support!",
                selftext="Very bullish and positive outlook",
                score=100,
                upvote_ratio=0.95,
            ),
        ]
        score, direction, count = RedditSentimentConnector._score_sentiment(posts)
        assert score > 0
        assert direction == "bullish"
        assert count == 1

    def test_bearish_posts(self) -> None:
        """Posts with bearish keywords produce negative sentiment."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        posts = [
            _make_post(
                title="Market will fall and decline sharply",
                selftext="Very bearish, expect a drop",
                score=50,
                upvote_ratio=0.8,
            ),
        ]
        score, direction, count = RedditSentimentConnector._score_sentiment(posts)
        assert score < 0
        assert direction == "bearish"
        assert count == 1

    def test_neutral_mixed_posts(self) -> None:
        """Equal bullish and bearish keywords produce neutral sentiment."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        posts = [
            _make_post(
                title="Could rise or fall depending on outcome",
                selftext="Bullish scenario vs bearish scenario",
                score=10,
                upvote_ratio=0.7,
            ),
        ]
        score, direction, count = RedditSentimentConnector._score_sentiment(posts)
        # "rise" + "bullish" = 2 bull, "fall" + "bearish" = 2 bear => neutral
        assert direction == "neutral"
        assert count == 1

    def test_no_sentiment_keywords_returns_neutral(self) -> None:
        """Posts without any sentiment keywords return 0.0 neutral."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        posts = [
            _make_post(title="Random discussion about weather", selftext="Check it here"),
        ]
        score, direction, count = RedditSentimentConnector._score_sentiment(posts)
        assert score == 0.0
        assert direction == "neutral"

    def test_weighted_by_score_and_upvote_ratio(self) -> None:
        """High-score posts carry more weight than low-score posts."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        posts = [
            _make_post(title="Bullish win likely", score=1000, upvote_ratio=0.95),
            _make_post(title="Bearish decline fail", score=5, upvote_ratio=0.5),
        ]
        score, direction, count = RedditSentimentConnector._score_sentiment(posts)
        # High-score bullish post should dominate
        assert score > 0
        assert direction == "bullish"
        assert count == 2


# ── 48h Time Filter ──────────────────────────────────────────────────


class TestTimeFilter:
    def test_48h_cutoff_constant(self) -> None:
        from src.research.connectors.reddit_sentiment import _48H_SECS
        assert _48H_SECS == 48 * 3600


# ── behavioral_signal ────────────────────────────────────────────────


class TestBehavioralSignal:
    def test_behavioral_signal_structure(self) -> None:
        """raw dict contains behavioral_signal with expected fields."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Will Bitcoin crash?",
            sentiment_score=0.5,
            direction="bullish",
            post_count=10,
            subreddits=["cryptocurrency"],
        )
        assert len(sources) == 1
        raw = sources[0].raw
        assert "behavioral_signal" in raw
        signal = raw["behavioral_signal"]
        assert signal["source"] == "reddit"
        assert signal["signal_type"] == "sentiment"
        assert signal["value"] == 0.5
        assert signal["direction"] == "bullish"
        assert signal["post_count"] == 10

    def test_post_count_in_behavioral_signal(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Will GDP grow?",
            sentiment_score=-0.3,
            direction="bearish",
            post_count=25,
            subreddits=["economics"],
        )
        signal = sources[0].raw["behavioral_signal"]
        assert signal["post_count"] == 25

    def test_content_includes_post_count(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Will Bitcoin crash?",
            sentiment_score=0.2,
            direction="bullish",
            post_count=7,
            subreddits=["cryptocurrency"],
        )
        assert "7" in sources[0].content


# ── Authority Score ──────────────────────────────────────────────────


class TestAuthorityScore:
    def test_authority_score_is_0_5(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test question",
            sentiment_score=0.0,
            direction="neutral",
            post_count=5,
            subreddits=["polymarket"],
        )
        assert sources[0].authority_score == 0.5

    def test_extraction_method_is_api(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test question",
            sentiment_score=0.0,
            direction="neutral",
            post_count=5,
            subreddits=["polymarket"],
        )
        assert sources[0].extraction_method == "api"


# ── Keyword Extraction ──────────────────────────────────────────────


class TestKeywordExtraction:
    def test_stop_words_stripped(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        query = RedditSentimentConnector._extract_search_query("Will the market crash by the end?")
        words = query.lower().split()
        assert "will" not in words
        assert "the" not in words
        assert "by" not in words
        assert "market" in words
        assert "crash" in words

    def test_question_mark_removed(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        query = RedditSentimentConnector._extract_search_query("Will Bitcoin hit 100k?")
        assert "?" not in query

    def test_truncated_to_120_chars(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        long_q = "word " * 100  # 500 chars
        query = RedditSentimentConnector._extract_search_query(long_q)
        assert len(query) <= 120


# ── Relevant Categories ─────────────────────────────────────────────


class TestRelevantCategories:
    def test_broad_category_coverage(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        cats = c.relevant_categories()
        for expected in [
            "MACRO", "ELECTION", "CORPORATE", "CRYPTO",
            "TECHNOLOGY", "SPORTS", "UNKNOWN",
        ]:
            assert expected in cats

    def test_name_is_reddit_sentiment(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        assert c.name == "reddit_sentiment"


# ── Source Builder ───────────────────────────────────────────────────


class TestSourceBuilder:
    def test_url_points_to_first_subreddit(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test",
            sentiment_score=0.5,
            direction="bullish",
            post_count=3,
            subreddits=["economics", "investing"],
        )
        assert sources[0].url == "https://reddit.com/r/economics"

    def test_snippet_contains_subreddit_and_direction(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test",
            sentiment_score=-0.5,
            direction="bearish",
            post_count=8,
            subreddits=["cryptocurrency"],
        )
        snippet = sources[0].snippet
        assert "r/cryptocurrency" in snippet
        assert "bearish" in snippet
        assert "8 posts" in snippet

    def test_publisher_is_reddit(self) -> None:
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test",
            sentiment_score=0.0,
            direction="neutral",
            post_count=1,
            subreddits=["polymarket"],
        )
        assert sources[0].publisher == "Reddit"

    def test_content_uses_public_json_api(self) -> None:
        """Content says 'public JSON API' not 'via PRAW'."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        sources = c._build_source(
            question="Test",
            sentiment_score=0.0,
            direction="neutral",
            post_count=1,
            subreddits=["polymarket"],
        )
        assert "public JSON API" in sources[0].content
        assert "PRAW" not in sources[0].content


# ── JSON API integration ─────────────────────────────────────────────


class TestJsonApiIntegration:
    @pytest.mark.asyncio
    async def test_search_uses_json_endpoint(self) -> None:
        """_search_subreddits calls reddit.com/r/{sub}/search.json."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)

        json_data = _make_reddit_json_response([
            {"title": "Test post", "score": 10, "upvote_ratio": 0.9,
             "created_utc": time.time() - 3600},
        ])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.reddit_sentiment.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            posts = await c._search_subreddits(["cryptocurrency"], "bitcoin")

        assert len(posts) == 1
        assert posts[0]["title"] == "Test post"
        # Verify the URL called
        call_args = mock_client.get.call_args
        assert "search.json" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_old_posts_filtered(self) -> None:
        """Posts older than 48h are excluded."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)

        json_data = _make_reddit_json_response([
            {"title": "Old post", "score": 10, "upvote_ratio": 0.9,
             "created_utc": time.time() - 72 * 3600},  # 72h ago
            {"title": "Recent post", "score": 10, "upvote_ratio": 0.9,
             "created_utc": time.time() - 3600},  # 1h ago
        ])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.reddit_sentiment.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            posts = await c._search_subreddits(["test"], "query")

        assert len(posts) == 1
        assert posts[0]["title"] == "Recent post"

    @pytest.mark.asyncio
    async def test_fetch_impl_returns_sources(self) -> None:
        """_fetch_impl returns FetchedSource list from JSON API."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)

        json_data = _make_reddit_json_response([
            {"title": "Bullish rise growth", "score": 50, "upvote_ratio": 0.9,
             "created_utc": time.time() - 3600},
        ])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.reddit_sentiment.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            sources = await c._fetch_impl("Will Bitcoin rise?", "CRYPTO")

        assert len(sources) == 1
        assert sources[0].publisher == "Reddit"
        signal = sources[0].raw["behavioral_signal"]
        assert signal["source"] == "reddit"
        assert signal["value"] > 0  # bullish

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        """Empty search query returns []."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)
        # All stop words → empty query
        sources = await c._fetch_impl("Will the?", "MACRO")
        assert sources == []

    @pytest.mark.asyncio
    async def test_api_error_returns_empty_posts(self) -> None:
        """HTTP error from Reddit returns empty posts list."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("429 Too Many Requests")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        with patch("src.research.connectors.reddit_sentiment.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            posts = await c._search_subreddits(["test"], "query")

        assert posts == []

    @pytest.mark.asyncio
    async def test_rate_limiter_called_per_subreddit(self) -> None:
        """Rate limiter acquire() is called once per subreddit."""
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector

        c = RedditSentimentConnector(config=None)

        json_data = _make_reddit_json_response([])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = json_data

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        c._client = mock_client

        mock_acquire = AsyncMock()
        with patch("src.research.connectors.reddit_sentiment.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = mock_acquire
            await c._search_subreddits(["sub1", "sub2", "sub3"], "query")

        assert mock_acquire.call_count == 3
