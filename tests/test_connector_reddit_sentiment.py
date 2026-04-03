"""Tests for RedditSentimentConnector -- crowd sentiment from Reddit."""

from __future__ import annotations

import asyncio
import os
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


def _make_submission(
    title: str = "Test post",
    selftext: str = "",
    score: int = 10,
    upvote_ratio: float = 0.9,
    age_hours: float = 12.0,
    permalink: str = "/r/test/comments/abc/test_post",
) -> MagicMock:
    """Build a mock praw Submission object."""
    sub = MagicMock()
    sub.title = title
    sub.selftext = selftext
    sub.score = score
    sub.upvote_ratio = upvote_ratio
    sub.created_utc = time.time() - (age_hours * 3600)
    sub.permalink = permalink
    return sub


# ── praw ImportError Handling ────────────────────────────────────────


class TestPrawImport:
    def test_praw_import_error_returns_empty(self) -> None:
        """When praw is not installed, connector returns []."""
        import src.research.connectors.reddit_sentiment as mod

        original = mod._HAS_PRAW
        try:
            mod._HAS_PRAW = False
            from src.research.connectors.reddit_sentiment import RedditSentimentConnector
            c = RedditSentimentConnector(config=None)
            result = asyncio.run(c._fetch_impl("Will Bitcoin crash?", "CRYPTO"))
            assert result == []
        finally:
            mod._HAS_PRAW = original


# ── Missing Credentials ─────────────────────────────────────────────


class TestMissingCredentials:
    def test_returns_empty_without_client_id(self) -> None:
        """Returns empty when reddit_client_id is missing."""
        import src.research.connectors.reddit_sentiment as mod

        original = mod._HAS_PRAW
        try:
            mod._HAS_PRAW = True
            from src.research.connectors.reddit_sentiment import RedditSentimentConnector
            c = RedditSentimentConnector(config=None)

            with patch.dict(os.environ, {"REDDIT_CLIENT_ID": "", "REDDIT_CLIENT_SECRET": "secret"}, clear=False):
                result = asyncio.run(c._fetch_impl("Will Bitcoin crash?", "CRYPTO"))
                assert result == []
        finally:
            mod._HAS_PRAW = original

    def test_returns_empty_without_client_secret(self) -> None:
        """Returns empty when reddit_client_secret is missing."""
        import src.research.connectors.reddit_sentiment as mod

        original = mod._HAS_PRAW
        try:
            mod._HAS_PRAW = True
            from src.research.connectors.reddit_sentiment import RedditSentimentConnector
            c = RedditSentimentConnector(config=None)

            with patch.dict(os.environ, {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": ""}, clear=False):
                result = asyncio.run(c._fetch_impl("Will Bitcoin crash?", "CRYPTO"))
                assert result == []
        finally:
            mod._HAS_PRAW = original


# ── Subreddit Routing ────────────────────────────────────────────────


class TestSubredditRouting:
    def test_election_routes_to_politics(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["ELECTION"] == ["politics"]

    def test_crypto_routes_to_cryptocurrency(self) -> None:
        from src.research.connectors.reddit_sentiment import _SUBREDDIT_MAP
        assert _SUBREDDIT_MAP["CRYPTO"] == ["cryptocurrency"]

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

    @patch("src.research.connectors.reddit_sentiment.rate_limiter")
    @patch.dict(os.environ, {"REDDIT_CLIENT_ID": "id", "REDDIT_CLIENT_SECRET": "secret"}, clear=False)
    def test_old_posts_filtered_in_search(self, mock_rl: MagicMock) -> None:
        """Posts older than 48h are excluded by _search_subreddits."""
        mock_rl.get.return_value.acquire = AsyncMock()

        import src.research.connectors.reddit_sentiment as mod

        original = mod._HAS_PRAW
        try:
            mod._HAS_PRAW = True

            from src.research.connectors.reddit_sentiment import RedditSentimentConnector
            c = RedditSentimentConnector(config=None)

            # Create mock reddit and submissions
            old_submission = _make_submission(
                title="Old post bullish",
                age_hours=72.0,  # 72h > 48h cutoff
            )
            recent_submission = _make_submission(
                title="Recent post bullish",
                age_hours=12.0,  # 12h < 48h cutoff
            )

            mock_subreddit = MagicMock()
            mock_subreddit.search.return_value = [old_submission, recent_submission]

            mock_reddit = MagicMock()
            mock_reddit.subreddit.return_value = mock_subreddit
            c._reddit = mock_reddit

            posts = c._search_subreddits(["test"], "bullish query")
            # Only the recent post should be included
            assert len(posts) == 1
            assert posts[0]["title"] == "Recent post bullish"
        finally:
            mod._HAS_PRAW = original


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
