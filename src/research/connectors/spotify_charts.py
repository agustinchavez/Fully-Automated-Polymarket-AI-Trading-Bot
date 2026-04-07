"""Spotify Charts connector — artist ranking data from kworb.net.

Scrapes kworb.net (free, no auth, static HTML) for:
- Top artists by monthly listeners (kworb.net/spotify/listeners2.html)
- Daily streaming chart positions (kworb.net/spotify/country/global_daily.html)

kworb.net updates every ~15 minutes from official Spotify data.
No API key required. Uses a 30-minute cache.
"""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_LISTENERS_URL = "https://kworb.net/spotify/listeners.html"
_DAILY_CHART_URL = "https://kworb.net/spotify/country/global_daily.html"
_USER_AGENT = "PolymarketBot/1.0 (research connector)"
_CACHE_TTL = 1800  # 30 minutes
_MIN_SIMILARITY = 0.50

# Regex to extract artist names from common Polymarket question patterns
_ARTIST_PATTERNS = [
    # "Will Drake be #1 on Spotify monthly listeners?"
    re.compile(r"(?:Will|Does|Is|Has)\s+(.+?)\s+(?:be|become|have|reach|hit|stay|remain)\b", re.IGNORECASE),
    # "Drake Spotify monthly listeners"
    re.compile(r"^(.+?)\s+(?:spotify|monthly\s+listeners|streaming)", re.IGNORECASE),
    # "most Spotify monthly listeners" — extract from "Will X have the most..."
    re.compile(r"(?:Will|Does)\s+(.+?)\s+(?:have\s+the\s+most|top|lead)", re.IGNORECASE),
]


class SpotifyChartsConnector(BaseResearchConnector):
    """Fetch Spotify artist ranking data from kworb.net."""

    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        self._listeners_cache: list[dict[str, Any]] | None = None
        self._listeners_cache_time: float = 0.0
        self._daily_cache: list[dict[str, Any]] | None = None
        self._daily_cache_time: float = 0.0

    @property
    def name(self) -> str:
        return "spotify_charts"

    def relevant_categories(self) -> set[str]:
        return {"CULTURE"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        """Only run for CULTURE markets mentioning Spotify/streaming keywords."""
        if market_type not in self.relevant_categories():
            return False
        return self._question_matches_keywords(question, [
            "spotify", "monthly listeners", "streaming", "top artist",
            "billboard", "charts", "number one", "#1",
        ])

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        await rate_limiter.get("spotify_charts").acquire()

        artist = self._extract_artist(question)
        if not artist:
            log.debug("spotify_charts.no_artist_extracted", question=question[:80])
            return []

        # Fetch both data sources
        listeners = await self._get_listeners_ranking()
        daily = await self._get_daily_chart()

        # Find artist in monthly listeners ranking
        listener_match = self._find_artist(artist, listeners)
        daily_match = self._find_artist(artist, daily)

        if listener_match is None and daily_match is None:
            log.debug("spotify_charts.no_match", artist=artist)
            return []

        # Build result
        content_lines = [f"Spotify Charts Data: {artist}"]
        rank = None
        monthly_listeners = ""
        daily_rank = None
        daily_streams = ""

        if listener_match is not None:
            rank = listener_match.get("rank", 0)
            monthly_listeners = listener_match.get("listeners", "")
            content_lines.append(
                f"Monthly listeners rank: #{rank}"
            )
            content_lines.append(
                f"Monthly listeners: {monthly_listeners}"
            )

        if daily_match is not None:
            daily_rank = daily_match.get("rank", 0)
            daily_streams = daily_match.get("streams", "")
            content_lines.append(
                f"Global daily chart rank: #{daily_rank}"
            )
            if daily_streams:
                content_lines.append(f"Daily streams: {daily_streams}")

        content_lines.append("Source: kworb.net (Spotify Charts)")
        content = "\n".join(content_lines)

        # Build snippet
        parts = []
        if rank is not None:
            parts.append(f"#{rank} monthly listeners")
        if daily_rank is not None:
            parts.append(f"#{daily_rank} daily chart")
        snippet = f"Spotify: {artist} — {', '.join(parts)}"

        return [self._make_source(
            title=f"Spotify Charts: {artist}",
            url=_LISTENERS_URL,
            snippet=snippet,
            publisher="kworb.net / Spotify Charts",
            content=content,
            authority_score=0.80,
            raw={
                "behavioral_signal": {
                    "source": "spotify_charts",
                    "signal_type": "chart_position",
                    "value": float(rank) if rank else 0.0,
                    "artist": artist,
                    "monthly_listeners_rank": rank,
                    "monthly_listeners": monthly_listeners,
                    "daily_chart_rank": daily_rank,
                    "daily_streams": daily_streams,
                },
            },
        )]

    # ── Data fetching ────────────────────────────────────────────────

    async def _get_listeners_ranking(self) -> list[dict[str, Any]]:
        """Fetch top artists by monthly listeners from kworb.net."""
        now = time.monotonic()
        if (
            self._listeners_cache is not None
            and now - self._listeners_cache_time < _CACHE_TTL
        ):
            return self._listeners_cache

        try:
            client = self._get_client()
            resp = await client.get(
                _LISTENERS_URL,
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            data = self._parse_listeners_html(resp.text)
            self._listeners_cache = data
            self._listeners_cache_time = now
        except Exception as exc:
            log.warning("spotify_charts.listeners_fetch_failed", error=str(exc))
            if self._listeners_cache is None:
                self._listeners_cache = []
                self._listeners_cache_time = now

        return self._listeners_cache

    async def _get_daily_chart(self) -> list[dict[str, Any]]:
        """Fetch global daily streaming chart from kworb.net."""
        now = time.monotonic()
        if (
            self._daily_cache is not None
            and now - self._daily_cache_time < _CACHE_TTL
        ):
            return self._daily_cache

        try:
            client = self._get_client()
            resp = await client.get(
                _DAILY_CHART_URL,
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            data = self._parse_daily_html(resp.text)
            self._daily_cache = data
            self._daily_cache_time = now
        except Exception as exc:
            log.warning("spotify_charts.daily_fetch_failed", error=str(exc))
            if self._daily_cache is None:
                self._daily_cache = []
                self._daily_cache_time = now

        return self._daily_cache

    # ── HTML parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_listeners_html(html: str) -> list[dict[str, Any]]:
        """Parse kworb.net/spotify/listeners.html into ranked artist list.

        Actual HTML structure:
        <tr><td>1</td><td class="text"><div><a href="...">Artist</a></div></td>
            <td>134,630,482</td>...</tr>
        """
        results: list[dict[str, Any]] = []
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
        cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
        link_pattern = re.compile(r"<a[^>]*>(.*?)</a>", re.DOTALL)

        for i, row_match in enumerate(row_pattern.finditer(html)):
            if i == 0:
                continue  # Skip header row
            cells = cell_pattern.findall(row_match.group(1))
            if len(cells) < 3:
                continue

            # First cell: rank number
            rank_str = re.sub(r"<[^>]+>", "", cells[0]).strip()
            try:
                rank = int(rank_str)
            except ValueError:
                continue

            # Second cell: artist name (inside <div><a>)
            name_cell = cells[1]
            link_match = link_pattern.search(name_cell)
            name = link_match.group(1).strip() if link_match else name_cell.strip()
            name = re.sub(r"<[^>]+>", "", name).strip()

            # Third cell: listener count
            listeners = re.sub(r"<[^>]+>", "", cells[2]).strip()

            if name:
                results.append({
                    "rank": rank,
                    "name": name,
                    "listeners": listeners,
                })

            if rank >= 200:  # Top 200 is plenty
                break

        return results

    @staticmethod
    def _parse_daily_html(html: str) -> list[dict[str, Any]]:
        """Parse kworb.net/spotify/country/global_daily.html into chart list.

        Actual HTML structure:
        <tr><td class="np">1</td><td class="np">=</td>
            <td class="text mp"><div><a>Artist</a> - <a>Song</a></div></td>
            <td>days</td><td>peak</td><td>...</td><td>streams</td>...</tr>
        """
        results: list[dict[str, Any]] = []
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
        cell_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
        link_pattern = re.compile(r"<a[^>]*>(.*?)</a>", re.DOTALL)

        for i, row_match in enumerate(row_pattern.finditer(html)):
            if i == 0:
                continue  # Skip header
            cells = cell_pattern.findall(row_match.group(1))
            if len(cells) < 7:
                continue

            # Cell[0]: rank
            rank_str = re.sub(r"<[^>]+>", "", cells[0]).strip()
            try:
                rank = int(rank_str)
            except ValueError:
                continue

            # Cell[2]: "Artist - Song" (two <a> tags inside <div>)
            name_cell = cells[2]
            # Extract all link texts
            links = link_pattern.findall(name_cell)
            if links:
                artist = links[0].strip()  # First <a> = artist
                song_title = links[1].strip() if len(links) > 1 else ""
                full_name = f"{artist} - {song_title}" if song_title else artist
            else:
                full_text = re.sub(r"<[^>]+>", "", name_cell).strip()
                artist = full_text.split(" - ")[0].strip() if " - " in full_text else full_text
                full_name = full_text

            # Cell[6]: daily stream count
            streams = re.sub(r"<[^>]+>", "", cells[6]).strip()

            if artist:
                results.append({
                    "rank": rank,
                    "name": artist,
                    "song": full_name,
                    "streams": streams,
                })

            if rank >= 200:
                break

        return results

    # ── Artist extraction & matching ─────────────────────────────────

    @staticmethod
    def _extract_artist(question: str) -> str:
        """Extract artist name from a market question.

        Tries regex patterns, then falls back to extracting capitalized
        words before Spotify-related keywords.
        """
        for pattern in _ARTIST_PATTERNS:
            match = pattern.search(question)
            if match:
                artist = match.group(1).strip()
                # Remove common filler words
                artist = re.sub(
                    r"\b(the|a|an|have|has|had|get|most|more|top)\b",
                    "", artist, flags=re.IGNORECASE,
                ).strip()
                if artist and len(artist) > 1:
                    return artist

        # Fallback: extract capitalized words
        words = question.split()
        proper = [
            w for w in words
            if w[0].isupper() and len(w) > 1
            and w.lower() not in {
                "will", "the", "does", "has", "is", "are", "can",
                "spotify", "billboard", "monthly", "listeners",
            }
        ]
        if proper:
            return " ".join(proper[:3])

        return ""

    @staticmethod
    def _find_artist(
        target: str,
        rankings: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find the best-matching artist in a ranking list."""
        target_lower = target.lower()
        best_score = _MIN_SIMILARITY
        best_match: dict[str, Any] | None = None

        for entry in rankings:
            name = entry.get("name", "")
            name_lower = name.lower()

            # Exact substring match (high confidence)
            if target_lower in name_lower or name_lower in target_lower:
                return entry

            # Fuzzy match
            score = SequenceMatcher(None, target_lower, name_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = entry

        return best_match
