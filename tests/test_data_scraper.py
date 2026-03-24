"""Tests for historical data scraper (Phase 1)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backtest.data_scraper import HistoricalDataScraper, ScrapeResult
from src.backtest.database import BacktestDatabase
from src.connectors.polymarket_gamma import GammaMarket, GammaToken


@pytest.fixture
def db() -> BacktestDatabase:
    bdb = BacktestDatabase(db_path=":memory:")
    bdb.connect()
    yield bdb
    bdb.close()


def _make_market(
    cid: str = "0x1",
    question: str = "Will X happen?",
    tokens: list[GammaToken] | None = None,
    volume: float = 5000.0,
    **kwargs: Any,
) -> GammaMarket:
    """Create a fake GammaMarket for testing."""
    if tokens is None:
        tokens = [
            GammaToken(token_id="t1", outcome="Yes", price=0.95, winner=True),
            GammaToken(token_id="t2", outcome="No", price=0.05, winner=False),
        ]
    defaults = dict(
        id=cid,
        condition_id=cid,
        question=question,
        description="Test market",
        category="Test",
        market_type="UNKNOWN",
        volume=volume,
        liquidity=1000.0,
        tokens=tokens,
        slug=f"test-{cid}",
        closed=True,
        active=False,
        raw={"id": cid, "question": question},
    )
    defaults.update(kwargs)
    return GammaMarket(**defaults)


# ── Resolution Parsing ────────────────────────────────────────────────


class TestParseResolution:

    def test_yes_winner(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", winner=True, price=1.0),
            GammaToken(outcome="No", winner=False, price=0.0),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "YES"

    def test_no_winner(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", winner=False, price=0.0),
            GammaToken(outcome="No", winner=True, price=1.0),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "NO"

    def test_no_winner_set(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", winner=None, price=0.5),
            GammaToken(outcome="No", winner=None, price=0.5),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "AMBIGUOUS"

    def test_all_false_winners(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", winner=False, price=0.5),
            GammaToken(outcome="No", winner=False, price=0.5),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "AMBIGUOUS"

    def test_lowercase_yes(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="yes", winner=True, price=1.0),
            GammaToken(outcome="no", winner=False, price=0.0),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "YES"

    def test_non_standard_outcome_winner(self) -> None:
        """Non-standard outcome names with a winner default to YES."""
        market = _make_market(tokens=[
            GammaToken(outcome="Trump", winner=True, price=1.0),
            GammaToken(outcome="Biden", winner=False, price=0.0),
        ])
        assert HistoricalDataScraper._parse_resolution(market) == "YES"


# ── Record Conversion ────────────────────────────────────────────────


class TestToHistoricalRecord:

    def test_basic_conversion(self) -> None:
        market = _make_market()
        record = HistoricalDataScraper._to_historical_record(market, "YES")
        assert record.condition_id == "0x1"
        assert record.question == "Will X happen?"
        assert record.resolution == "YES"
        assert record.volume_usd == 5000.0
        assert record.scraped_at != ""

    def test_prices_json(self) -> None:
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", price=0.75, winner=True),
            GammaToken(outcome="No", price=0.25, winner=False),
        ])
        record = HistoricalDataScraper._to_historical_record(market, "YES")
        import json
        prices = json.loads(record.final_prices_json)
        assert prices["Yes"] == 0.75
        assert prices["No"] == 0.25

    def test_tokens_json(self) -> None:
        market = _make_market()
        record = HistoricalDataScraper._to_historical_record(market, "YES")
        import json
        tokens = json.loads(record.tokens_json)
        assert len(tokens) == 2
        assert tokens[0]["outcome"] == "Yes"


# ── Process Market (filter logic) ────────────────────────────────────


class TestProcessMarket:

    def test_valid_market_passes(self, db: BacktestDatabase) -> None:
        scraper = HistoricalDataScraper(db)
        result = ScrapeResult()
        market = _make_market()
        record = scraper._process_market(market, result)
        assert record is not None
        assert record.resolution == "YES"

    def test_non_binary_market_skipped(self, db: BacktestDatabase) -> None:
        scraper = HistoricalDataScraper(db)
        result = ScrapeResult()
        market = _make_market(tokens=[
            GammaToken(outcome="A", price=0.3),
            GammaToken(outcome="B", price=0.3),
            GammaToken(outcome="C", price=0.4),
        ])
        record = scraper._process_market(market, result)
        assert record is None
        assert result.invalid_skipped == 1

    def test_ambiguous_resolution_skipped(self, db: BacktestDatabase) -> None:
        scraper = HistoricalDataScraper(db)
        result = ScrapeResult()
        market = _make_market(tokens=[
            GammaToken(outcome="Yes", winner=None, price=0.5),
            GammaToken(outcome="No", winner=None, price=0.5),
        ])
        record = scraper._process_market(market, result)
        assert record is None
        assert result.invalid_skipped == 1

    def test_low_volume_skipped(self, db: BacktestDatabase) -> None:
        scraper = HistoricalDataScraper(db, min_volume=5000.0)
        result = ScrapeResult()
        market = _make_market(volume=100.0)
        record = scraper._process_market(market, result)
        assert record is None
        assert result.invalid_skipped == 1


# ── Full Scrape ───────────────────────────────────────────────────────


class TestScrape:

    def test_scrape_stores_markets(self, db: BacktestDatabase) -> None:
        """Scrape inserts valid markets into the database."""
        markets_page1 = [_make_market(f"m{i}") for i in range(3)]

        async def mock_list_markets(**kwargs):
            if kwargs.get("offset", 0) == 0:
                return markets_page1
            return []

        with patch(
            "src.backtest.data_scraper.GammaClient",
        ) as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list_markets)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db, min_volume=0.0)
            result = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=10, batch_size=10)
            )

        assert result.new_inserted == 3
        assert db.count_historical_markets() == 3

    def test_scrape_deduplication(self, db: BacktestDatabase) -> None:
        """Second scrape of same markets counts as duplicates."""
        markets = [_make_market("dup")]

        async def mock_list(**kwargs):
            if kwargs.get("offset", 0) == 0:
                return markets
            return []

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db, min_volume=0.0)
            # First scrape
            r1 = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=10, batch_size=10)
            )
            assert r1.new_inserted == 1

            # Second scrape
            r2 = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=10, batch_size=10)
            )
            assert r2.duplicates_skipped == 1
            assert r2.new_inserted == 0

    def test_scrape_progress_callback(self, db: BacktestDatabase) -> None:
        """Progress callback is called with correct counts."""
        markets = [_make_market(f"p{i}") for i in range(2)]
        progress_calls: list[tuple[int, int]] = []

        async def mock_list(**kwargs):
            if kwargs.get("offset", 0) == 0:
                return markets
            return []

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db, min_volume=0.0)
            asyncio.new_event_loop().run_until_complete(
                scraper.scrape(
                    max_markets=10, batch_size=10,
                    progress_callback=lambda c, t: progress_calls.append((c, t)),
                )
            )

        assert len(progress_calls) >= 1
        # Last call should show collected count
        assert progress_calls[-1][0] == 2

    def test_scrape_api_error(self, db: BacktestDatabase) -> None:
        """API errors are captured without crashing."""

        async def mock_list(**kwargs):
            raise ConnectionError("API down")

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db)
            result = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=10, batch_size=10)
            )

        assert len(result.errors) == 1
        assert "API down" in result.errors[0]

    def test_scrape_pagination(self, db: BacktestDatabase) -> None:
        """Scraper paginates through multiple batches."""
        call_count = 0

        async def mock_list(**kwargs):
            nonlocal call_count
            call_count += 1
            offset = kwargs.get("offset", 0)
            if offset == 0:
                return [_make_market(f"batch1_{i}") for i in range(3)]
            elif offset == 3:
                return [_make_market(f"batch2_{i}") for i in range(2)]
            return []

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db, min_volume=0.0)
            result = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=100, batch_size=3)
            )

        assert result.new_inserted == 5
        assert call_count >= 2

    def test_scrape_respects_max_markets(self, db: BacktestDatabase) -> None:
        """Scraper stops after reaching max_markets."""

        async def mock_list(**kwargs):
            offset = kwargs.get("offset", 0)
            return [_make_market(f"m{offset + i}") for i in range(5)]

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db, min_volume=0.0)
            result = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=3, batch_size=5)
            )

        # Should have collected exactly 3 (capped by max_markets)
        assert db.count_historical_markets() == 3

    def test_scrape_duration_tracked(self, db: BacktestDatabase) -> None:
        async def mock_list(**kwargs):
            return []

        with patch("src.backtest.data_scraper.GammaClient") as MockClient:
            instance = MockClient.return_value
            instance.list_markets = AsyncMock(side_effect=mock_list)
            instance.close = AsyncMock()

            scraper = HistoricalDataScraper(db)
            result = asyncio.new_event_loop().run_until_complete(
                scraper.scrape(max_markets=10)
            )

        assert result.duration_secs >= 0.0
