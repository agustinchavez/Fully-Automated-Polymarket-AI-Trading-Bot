"""Reddit sentiment connector — crowd sentiment signal from subreddit discussions.

Uses Reddit's public JSON API (no library or credentials required) to search
subreddits for recent posts matching a market question, then scores overall
sentiment based on bullish/bearish keyword frequency weighted by post score.

Rate-limited at 60 requests/minute via the ``reddit`` rate limiter bucket.
"""

from __future__ import annotations

import time as _time
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

# ── Subreddit routing ────────────────────────────────────────────────

_SUBREDDIT_MAP: dict[str, list[str]] = {
    "ELECTION": ["politics", "PoliticalDiscussion"],
    "CRYPTO": ["cryptocurrency", "Bitcoin", "ethereum", "CryptoMarkets"],
    "MACRO": ["economics", "investing"],
    "SCIENCE": ["science"],
    "CORPORATE": ["stocks"],
    "LEGAL": ["law"],
    "TECHNOLOGY": ["technology"],
    "TECH": ["technology"],
    "SPORTS": ["nba", "soccer", "nfl", "mlb", "tennis", "leagueoflegends"],
    "GEOPOLITICS": ["worldnews", "geopolitics", "CredibleDefense"],
    "CULTURE": ["Music", "popheads", "movies", "hiphopheads"],
    "WEATHER": ["weather", "climatology"],
}

_DEFAULT_SUBREDDITS: list[str] = ["polymarket"]

# ── Sentiment keywords ───────────────────────────────────────────────

_BULLISH_KEYWORDS: set[str] = {
    "bullish", "up", "rise", "gain", "positive", "growth",
    "win", "pass", "approve", "support", "yes", "likely",
}

_BEARISH_KEYWORDS: set[str] = {
    "bearish", "down", "fall", "drop", "negative", "decline",
    "lose", "fail", "reject", "oppose", "no", "unlikely",
}

# Words stripped when extracting a search query from the question
_STOP_WORDS: set[str] = {
    "will", "is", "does", "has", "are", "do", "can", "should",
    "what", "when", "how", "the", "a", "an", "of", "in", "to",
    "by", "before", "after", "be", "been", "being",
}

# 48 hours in seconds
_48H_SECS = 48 * 3600

_USER_AGENT = "polymarket-bot/1.0"


class RedditSentimentConnector(BaseResearchConnector):
    """Fetch crowd sentiment from Reddit discussions via public JSON API."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)

    # ── Identity ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "reddit_sentiment"

    def relevant_categories(self) -> set[str]:
        return {
            "MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY",
            "SCIENCE", "GEOPOLITICS", "CRYPTO", "WEATHER", "SPORTS",
            "ENTERTAINMENT", "TECH", "REGULATION", "CULTURE", "UNKNOWN",
        }

    # ── Core fetch ───────────────────────────────────────────────────

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        search_query = self._extract_search_query(question)
        if not search_query:
            return []

        subreddits = _SUBREDDIT_MAP.get(market_type, _DEFAULT_SUBREDDITS)

        posts = await self._search_subreddits(subreddits, search_query)

        if not posts:
            return []

        sentiment_score, direction, post_count = self._score_sentiment(posts)

        return self._build_source(
            question=question,
            sentiment_score=sentiment_score,
            direction=direction,
            post_count=post_count,
            subreddits=subreddits,
        )

    # ── Async Reddit JSON API search ─────────────────────────────────

    async def _search_subreddits(
        self,
        subreddits: list[str],
        query: str,
    ) -> list[dict[str, Any]]:
        """Search subreddits via Reddit's public JSON API, filter to 48h window."""
        now = _time.time()
        cutoff = now - _48H_SECS
        posts: list[dict[str, Any]] = []

        client = self._get_client(timeout=15.0)

        for sub_name in subreddits:
            try:
                await rate_limiter.get("reddit").acquire()

                url = f"https://www.reddit.com/r/{sub_name}/search.json"
                resp = await client.get(
                    url,
                    params={"q": query, "sort": "new", "limit": 25, "t": "week",
                            "restrict_sr": "on"},
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()

                children = data.get("data", {}).get("children", [])
                for child in children:
                    post_data = child.get("data", {})
                    created_utc = post_data.get("created_utc", 0)
                    if created_utc < cutoff:
                        continue
                    posts.append({
                        "title": post_data.get("title", ""),
                        "selftext": post_data.get("selftext", ""),
                        "score": post_data.get("score", 1),
                        "upvote_ratio": post_data.get("upvote_ratio", 0.5),
                        "created_utc": created_utc,
                        "url": f"https://reddit.com{post_data.get('permalink', '')}",
                        "subreddit": sub_name,
                    })
            except Exception as exc:
                log.warning(
                    "reddit_sentiment.subreddit_search_failed",
                    subreddit=sub_name,
                    error=str(exc),
                )

        return posts

    # ── Keyword extraction ───────────────────────────────────────────

    @staticmethod
    def _extract_search_query(question: str) -> str:
        """Strip common question words to produce a Reddit search query."""
        q = question.strip().rstrip("?")
        words = q.split()
        filtered = [w for w in words if w.lower() not in _STOP_WORDS]
        return " ".join(filtered)[:120]

    # ── Sentiment scoring ────────────────────────────────────────────

    @staticmethod
    def _score_sentiment(
        posts: list[dict[str, Any]],
    ) -> tuple[float, str, int]:
        """Compute weighted sentiment score from posts.

        Returns
        -------
        sentiment_score : float
            Value in [-1.0, +1.0].
        direction : str
            ``"bullish"`` / ``"bearish"`` / ``"neutral"``.
        post_count : int
            Number of posts analysed.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for post in posts:
            text = (post["title"] + " " + post["selftext"]).lower()
            weight = max(post["score"], 1) * post["upvote_ratio"]

            bull_hits = sum(1 for kw in _BULLISH_KEYWORDS if kw in text)
            bear_hits = sum(1 for kw in _BEARISH_KEYWORDS if kw in text)

            if bull_hits or bear_hits:
                post_signal = (bull_hits - bear_hits) / (bull_hits + bear_hits)
                weighted_sum += post_signal * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0, "neutral", len(posts)

        raw_score = weighted_sum / total_weight
        sentiment_score = max(-1.0, min(1.0, raw_score))

        if sentiment_score > 0.1:
            direction = "bullish"
        elif sentiment_score < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"

        return round(sentiment_score, 4), direction, len(posts)

    # ── Source builder ───────────────────────────────────────────────

    def _build_source(
        self,
        *,
        question: str,
        sentiment_score: float,
        direction: str,
        post_count: int,
        subreddits: list[str],
    ) -> list[FetchedSource]:
        subs_display = ", ".join(f"r/{s}" for s in subreddits)
        content_lines = [
            f"Reddit Sentiment Signal",
            f"Subreddits: {subs_display}",
            f"Posts analysed (48h window): {post_count}",
            f"Sentiment score: {sentiment_score:+.4f} ({direction.upper()})",
            f"Source: Reddit (public JSON API)",
        ]
        content = "\n".join(content_lines)
        snippet = (
            f"Reddit sentiment ({subs_display}): "
            f"{sentiment_score:+.4f} ({direction}), {post_count} posts"
        )

        return [self._make_source(
            title=f"Reddit Sentiment: {question[:60]}",
            url=f"https://reddit.com/r/{subreddits[0]}",
            snippet=snippet,
            publisher="Reddit",
            content=content,
            authority_score=0.5,
            raw={
                "behavioral_signal": {
                    "source": "reddit",
                    "signal_type": "sentiment",
                    "value": sentiment_score,
                    "direction": direction,
                    "post_count": post_count,
                },
            },
        )]
