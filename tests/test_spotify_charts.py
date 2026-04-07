"""Tests for SpotifyChartsConnector — kworb.net scraper.

Covers:
- Artist name extraction from market questions
- HTML parsing (listeners table, daily chart table)
- Artist matching (exact, substring, fuzzy)
- FetchedSource output structure
- Cache behavior
- is_relevant filtering
- SignalStack integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.connectors.spotify_charts import SpotifyChartsConnector


# ── Sample HTML ──────────────────────────────────────────────────────

SAMPLE_LISTENERS_HTML = """
<html><body>
<table class="sortable">
<thead><tr><th>#</th><th class="text">Artist</th><th>Listeners</th><th>Daily +/-</th><th>PkListeners</th></tr></thead>
<tbody>
<tr><td>1</td><td class="text"><div><a href="artist/abc_songs.html">The Weeknd</a></div></td><td>115,234,567</td><td>+50,000</td><td class="smaller">126,000,000</td></tr>
<tr><td>2</td><td class="text"><div><a href="artist/def_songs.html">Taylor Swift</a></div></td><td>102,345,678</td><td>+30,000</td><td class="smaller">116,000,000</td></tr>
<tr><td>3</td><td class="text"><div><a href="artist/ghi_songs.html">Drake</a></div></td><td>91,234,567</td><td>-10,000</td><td class="smaller">95,000,000</td></tr>
<tr><td>4</td><td class="text"><div><a href="artist/jkl_songs.html">Bad Bunny</a></div></td><td>82,123,456</td><td>+5,000</td><td class="smaller">90,000,000</td></tr>
<tr><td>5</td><td class="text"><div><a href="artist/mno_songs.html">Ed Sheeran</a></div></td><td>78,901,234</td><td>+20,000</td><td class="smaller">85,000,000</td></tr>
</tbody>
</table>
</body></html>
"""

SAMPLE_DAILY_HTML = """
<html><body>
<table class="sortable" id="spotifydaily">
<thead><tr><th class="np">Pos</th><th class="np">P+</th><th class="mp text">Artist and Title</th><th>Days</th><th>Pk</th><th class="mini text">(x?)</th><th>Streams</th><th>Streams+</th><th>7Day</th><th>7Day+</th><th>Total</th></tr></thead>
<tbody>
<tr><td class="np">1</td><td class="np">=</td><td class="text mp"><div><a href="artist/abc.html">The Weeknd</a> - <a href="track/xyz.html">Blinding Lights</a></div></td><td>100</td><td>1</td><td class="np mini text">(x50)</td><td>8,234,567</td><td>+100,000</td><td class="smaller">55,000,000</td><td class="smaller">+500,000</td><td>500,000,000</td></tr>
<tr><td class="np">2</td><td class="np">=</td><td class="text mp"><div><a href="artist/def.html">Taylor Swift</a> - <a href="track/uvw.html">Anti-Hero</a></div></td><td>200</td><td>1</td><td class="np mini text">(x10)</td><td>7,123,456</td><td>+50,000</td><td class="smaller">48,000,000</td><td class="smaller">+200,000</td><td>400,000,000</td></tr>
<tr><td class="np">3</td><td class="np">=</td><td class="text mp"><div><a href="artist/ghi.html">Drake</a> - <a href="track/rst.html">One Dance</a></div></td><td>300</td><td>1</td><td class="np mini text">(x5)</td><td>6,543,210</td><td>-10,000</td><td class="smaller">42,000,000</td><td class="smaller">-100,000</td><td>350,000,000</td></tr>
</tbody>
</table>
</body></html>
"""


# ── HTML Parsing ─────────────────────────────────────────────────────


class TestParseListenersHTML:
    """Parse kworb.net listeners table."""

    def test_parses_artists_and_listeners(self):
        result = SpotifyChartsConnector._parse_listeners_html(SAMPLE_LISTENERS_HTML)
        assert len(result) == 5
        assert result[0]["name"] == "The Weeknd"
        assert result[0]["rank"] == 1
        assert result[0]["listeners"] == "115,234,567"

    def test_all_artists_present(self):
        result = SpotifyChartsConnector._parse_listeners_html(SAMPLE_LISTENERS_HTML)
        names = [r["name"] for r in result]
        assert "Drake" in names
        assert "Bad Bunny" in names
        assert "Ed Sheeran" in names

    def test_ranks_sequential(self):
        result = SpotifyChartsConnector._parse_listeners_html(SAMPLE_LISTENERS_HTML)
        for i, entry in enumerate(result):
            assert entry["rank"] == i + 1

    def test_empty_html(self):
        result = SpotifyChartsConnector._parse_listeners_html("")
        assert result == []

    def test_no_table(self):
        result = SpotifyChartsConnector._parse_listeners_html("<html><body>no table</body></html>")
        assert result == []


class TestParseDailyHTML:
    """Parse kworb.net daily chart table."""

    def test_parses_songs_and_artists(self):
        result = SpotifyChartsConnector._parse_daily_html(SAMPLE_DAILY_HTML)
        assert len(result) == 3
        assert result[0]["name"] == "The Weeknd"
        assert result[0]["song"] == "The Weeknd - Blinding Lights"
        assert result[0]["streams"] == "8,234,567"

    def test_extracts_artist_from_song_title(self):
        result = SpotifyChartsConnector._parse_daily_html(SAMPLE_DAILY_HTML)
        assert result[1]["name"] == "Taylor Swift"
        assert result[2]["name"] == "Drake"

    def test_empty_html(self):
        result = SpotifyChartsConnector._parse_daily_html("")
        assert result == []


# ── Artist Extraction ────────────────────────────────────────────────


class TestExtractArtist:
    """Extract artist name from market questions."""

    def test_will_be_pattern(self):
        artist = SpotifyChartsConnector._extract_artist(
            "Will Drake be #1 on Spotify monthly listeners?"
        )
        assert "Drake" in artist

    def test_will_have_pattern(self):
        artist = SpotifyChartsConnector._extract_artist(
            "Will The Weeknd have the most Spotify monthly listeners?"
        )
        assert "Weeknd" in artist

    def test_will_reach_pattern(self):
        artist = SpotifyChartsConnector._extract_artist(
            "Will Taylor Swift reach 100M Spotify monthly listeners?"
        )
        assert "Taylor Swift" in artist

    def test_stay_pattern(self):
        artist = SpotifyChartsConnector._extract_artist(
            "Will Bad Bunny stay in the top 5 Spotify artists?"
        )
        assert "Bad Bunny" in artist

    def test_fallback_proper_nouns(self):
        artist = SpotifyChartsConnector._extract_artist(
            "Adele Spotify monthly listeners ranking"
        )
        assert "Adele" in artist

    def test_empty_question(self):
        artist = SpotifyChartsConnector._extract_artist("")
        assert artist == ""


# ── Artist Matching ──────────────────────────────────────────────────


class TestFindArtist:
    """Find artist in ranking list."""

    def test_exact_match(self):
        rankings = [
            {"rank": 1, "name": "Drake", "listeners": "90M"},
            {"rank": 2, "name": "Taylor Swift", "listeners": "100M"},
        ]
        result = SpotifyChartsConnector._find_artist("Drake", rankings)
        assert result is not None
        assert result["rank"] == 1

    def test_substring_match(self):
        rankings = [
            {"rank": 1, "name": "The Weeknd", "listeners": "110M"},
        ]
        result = SpotifyChartsConnector._find_artist("Weeknd", rankings)
        assert result is not None
        assert result["rank"] == 1

    def test_case_insensitive(self):
        rankings = [
            {"rank": 1, "name": "Bad Bunny", "listeners": "80M"},
        ]
        result = SpotifyChartsConnector._find_artist("bad bunny", rankings)
        assert result is not None

    def test_no_match(self):
        rankings = [
            {"rank": 1, "name": "Drake", "listeners": "90M"},
        ]
        result = SpotifyChartsConnector._find_artist("Nonexistent Artist XYZ", rankings)
        assert result is None

    def test_empty_rankings(self):
        result = SpotifyChartsConnector._find_artist("Drake", [])
        assert result is None


# ── is_relevant ──────────────────────────────────────────────────────


class TestIsRelevant:
    """Keyword-based relevance check."""

    def test_culture_with_spotify(self):
        conn = SpotifyChartsConnector()
        assert conn.is_relevant("Will Drake be #1 on Spotify?", "CULTURE") is True

    def test_culture_with_monthly_listeners(self):
        conn = SpotifyChartsConnector()
        assert conn.is_relevant("Monthly listeners for Taylor Swift", "CULTURE") is True

    def test_culture_without_keywords(self):
        conn = SpotifyChartsConnector()
        assert conn.is_relevant("Will the Oscars have record viewership?", "CULTURE") is False

    def test_non_culture_category(self):
        conn = SpotifyChartsConnector()
        assert conn.is_relevant("Will Spotify reach $400 stock price?", "CORPORATE") is False

    def test_streaming_keyword(self):
        conn = SpotifyChartsConnector()
        assert conn.is_relevant("Will this artist top the streaming charts?", "CULTURE") is True


# ── Connector Integration ────────────────────────────────────────────


class TestFetchImpl:
    """Full connector fetch flow with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_returns_source_when_artist_found(self):
        conn = SpotifyChartsConnector()

        mock_resp_listeners = MagicMock()
        mock_resp_listeners.text = SAMPLE_LISTENERS_HTML
        mock_resp_listeners.raise_for_status = MagicMock()

        mock_resp_daily = MagicMock()
        mock_resp_daily.text = SAMPLE_DAILY_HTML
        mock_resp_daily.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_resp_listeners, mock_resp_daily])
        conn._client = mock_client

        with patch("src.research.connectors.spotify_charts.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            result = await conn._fetch_impl(
                "Will Drake be #1 on Spotify monthly listeners?", "CULTURE",
            )

        assert len(result) == 1
        source = result[0]
        assert "Drake" in source.title
        assert source.publisher == "kworb.net / Spotify Charts"
        assert source.authority_score == 0.80

        # Check raw signal structure
        bs = source.raw.get("behavioral_signal", {})
        assert bs["source"] == "spotify_charts"
        assert bs["signal_type"] == "chart_position"
        assert bs["artist"] == "Drake"
        assert bs["monthly_listeners_rank"] == 3

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_artist_extracted(self):
        conn = SpotifyChartsConnector()

        with patch("src.research.connectors.spotify_charts.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            result = await conn._fetch_impl("some random question", "CULTURE")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_artist_not_in_charts(self):
        conn = SpotifyChartsConnector()

        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_LISTENERS_HTML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        conn._client = mock_client

        with patch("src.research.connectors.spotify_charts.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            result = await conn._fetch_impl(
                "Will Nonexistent XYZ Artist be #1 on Spotify?", "CULTURE",
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_cache_prevents_refetch(self):
        conn = SpotifyChartsConnector()

        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_LISTENERS_HTML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        conn._client = mock_client

        with patch("src.research.connectors.spotify_charts.rate_limiter") as mock_rl:
            mock_rl.get.return_value.acquire = AsyncMock()
            # First call populates cache
            await conn._fetch_impl(
                "Will Drake be #1 on Spotify monthly listeners?", "CULTURE",
            )
            call_count_after_first = mock_client.get.call_count

            # Second call should use cache (no new HTTP calls for listeners)
            await conn._fetch_impl(
                "Will Drake be #1 on Spotify monthly listeners?", "CULTURE",
            )
            # Should not double the HTTP call count
            assert mock_client.get.call_count <= call_count_after_first + 1


# ── SignalStack Integration ──────────────────────────────────────────


class TestSignalStackIntegration:
    """Spotify data flows into SignalStack correctly."""

    def test_build_signal_stack_with_spotify(self):
        from src.research.signal_aggregator import build_signal_stack
        from src.research.source_fetcher import FetchedSource

        source = FetchedSource(
            title="Spotify Charts: Drake",
            url="https://kworb.net/spotify/listeners2.html",
            snippet="Spotify: Drake — #3 monthly listeners",
            publisher="kworb.net",
            content="Spotify Charts Data: Drake\nMonthly listeners rank: #3",
            authority_score=0.80,
            extraction_method="api",
            content_length=50,
            raw={
                "behavioral_signal": {
                    "source": "spotify_charts",
                    "signal_type": "chart_position",
                    "value": 3.0,
                    "artist": "Drake",
                    "monthly_listeners_rank": 3,
                    "monthly_listeners": "91,234,567",
                    "daily_chart_rank": 5,
                    "daily_streams": "6,543,210",
                },
            },
        )

        stack = build_signal_stack([source], poly_price=0.50)

        assert stack.spotify_artist == "Drake"
        assert stack.spotify_listeners_rank == 3
        assert stack.spotify_daily_rank == 5
        assert stack.spotify_monthly_listeners == "91,234,567"

    def test_render_signal_stack_includes_spotify(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            spotify_artist="Drake",
            spotify_listeners_rank=3,
            spotify_daily_rank=5,
            spotify_monthly_listeners="91,234,567",
        )

        rendered = render_signal_stack(stack)
        assert "Spotify Charts: Drake" in rendered
        assert "#3 monthly listeners" in rendered
        assert "#5 daily chart" in rendered

    def test_render_signal_stack_omits_when_empty(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack()
        rendered = render_signal_stack(stack)
        assert "Spotify" not in rendered

    def test_build_signal_stack_no_spotify_no_change(self):
        from src.research.signal_aggregator import build_signal_stack

        stack = build_signal_stack([], poly_price=0.50)
        assert stack.spotify_artist == ""
        assert stack.spotify_listeners_rank is None
        assert stack.spotify_daily_rank is None


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    """Connector loads via registry when enabled."""

    def test_registry_loads_spotify_charts(self):
        from src.research.connectors.registry import get_enabled_connectors

        config = MagicMock()
        config.spotify_charts_enabled = True
        # Disable everything else
        for attr in dir(config):
            if attr.endswith("_enabled") and attr != "spotify_charts_enabled":
                setattr(config, attr, False)

        connectors = get_enabled_connectors(config)
        names = [c.name for c in connectors]
        assert "spotify_charts" in names

    def test_registry_skips_when_disabled(self):
        from src.research.connectors.registry import get_enabled_connectors

        config = MagicMock()
        config.spotify_charts_enabled = False
        for attr in dir(config):
            if attr.endswith("_enabled"):
                setattr(config, attr, False)

        connectors = get_enabled_connectors(config)
        names = [c.name for c in connectors]
        assert "spotify_charts" not in names
