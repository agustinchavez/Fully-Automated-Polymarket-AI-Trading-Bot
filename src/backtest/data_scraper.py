"""Historical data scraper — fetches resolved markets from the Gamma API.

Uses the existing GammaClient to paginate through all resolved markets
and stores them in the backtest database for replay.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.backtest.database import BacktestDatabase
from src.backtest.models import HistoricalMarketRecord
from src.connectors.polymarket_gamma import GammaClient, GammaMarket, GammaToken
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class ScrapeResult:
    """Summary of a scrape operation."""
    total_fetched: int = 0
    new_inserted: int = 0
    duplicates_skipped: int = 0
    invalid_skipped: int = 0      # No clear resolution
    errors: list[str] = field(default_factory=list)
    duration_secs: float = 0.0


class HistoricalDataScraper:
    """Fetches resolved markets from the Polymarket Gamma API."""

    def __init__(
        self,
        db: BacktestDatabase,
        min_volume: float = 1000.0,
    ):
        self._db = db
        self._min_volume = min_volume
        self._client: GammaClient | None = None

    async def scrape(
        self,
        max_markets: int = 10000,
        batch_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ScrapeResult:
        """Fetch resolved markets in batches using offset pagination.

        Paginates through closed markets ordered by volume (highest first).
        Skips markets already in the database (idempotent via upsert).
        """
        result = ScrapeResult()
        start = time.monotonic()
        client = GammaClient()
        offset = 0
        collected = 0

        try:
            while collected < max_markets:
                batch_limit = min(batch_size, max_markets - collected)
                try:
                    markets = await client.list_markets(
                        limit=batch_limit,
                        offset=offset,
                        active=False,
                        closed=True,
                        order="volume",
                        ascending=False,
                    )
                except Exception as e:
                    result.errors.append(f"API error at offset {offset}: {e}")
                    log.error(
                        "scraper.api_error", offset=offset, error=str(e),
                    )
                    break

                if not markets:
                    log.info("scraper.no_more_markets", offset=offset)
                    break

                for market in markets:
                    if collected >= max_markets:
                        break
                    result.total_fetched += 1
                    record = self._process_market(market, result)
                    if record:
                        existing = self._db.get_historical_market(
                            record.condition_id,
                        )
                        if existing:
                            result.duplicates_skipped += 1
                        else:
                            result.new_inserted += 1
                        # Upsert regardless (may update volume/liquidity)
                        self._db.upsert_historical_market(record)
                        collected += 1

                offset += len(markets)

                if progress_callback:
                    progress_callback(collected, max_markets)

                log.info(
                    "scraper.batch_complete",
                    batch_size=len(markets),
                    collected=collected,
                    offset=offset,
                )

                # API returned fewer than requested → we've exhausted results
                if len(markets) < batch_limit:
                    break

        finally:
            await client.close()

        result.duration_secs = round(time.monotonic() - start, 2)
        log.info(
            "scraper.complete",
            total_fetched=result.total_fetched,
            new_inserted=result.new_inserted,
            duplicates=result.duplicates_skipped,
            invalid=result.invalid_skipped,
            duration=result.duration_secs,
        )
        return result

    async def scrape_recent(self, days_back: int = 7) -> ScrapeResult:
        """Incremental scraper for newly resolved markets.

        Fetches the most recently closed markets (by startDate desc)
        and stops when it hits markets older than ``days_back`` days.
        """
        result = ScrapeResult()
        start = time.monotonic()
        client = GammaClient()
        offset = 0
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)

        try:
            while True:
                try:
                    markets = await client.list_markets(
                        limit=100,
                        offset=offset,
                        active=False,
                        closed=True,
                        order="startDate",
                        ascending=False,
                    )
                except Exception as e:
                    result.errors.append(f"API error: {e}")
                    break

                if not markets:
                    break

                past_cutoff = False
                for market in markets:
                    result.total_fetched += 1
                    # Check if market is older than cutoff
                    if market.end_date and market.end_date < cutoff:
                        past_cutoff = True
                        break
                    record = self._process_market(market, result)
                    if record:
                        existing = self._db.get_historical_market(
                            record.condition_id,
                        )
                        if existing:
                            result.duplicates_skipped += 1
                        else:
                            result.new_inserted += 1
                        self._db.upsert_historical_market(record)

                if past_cutoff:
                    break
                offset += len(markets)
                if len(markets) < 100:
                    break

        finally:
            await client.close()

        result.duration_secs = round(time.monotonic() - start, 2)
        return result

    def _process_market(
        self,
        market: GammaMarket,
        result: ScrapeResult,
    ) -> HistoricalMarketRecord | None:
        """Validate and convert a GammaMarket to a HistoricalMarketRecord.

        Returns None for markets that should be skipped.
        """
        # Only binary markets (exactly 2 outcomes)
        if len(market.tokens) != 2:
            result.invalid_skipped += 1
            return None

        # Must have clear resolution
        resolution = self._parse_resolution(market)
        if resolution == "AMBIGUOUS":
            result.invalid_skipped += 1
            return None

        # Volume filter
        if market.volume < self._min_volume:
            result.invalid_skipped += 1
            return None

        return self._to_historical_record(market, resolution)

    @staticmethod
    def _parse_resolution(market: GammaMarket) -> str:
        """Determine resolution from token.winner field.

        Returns:
            "YES" if the Yes token won
            "NO" if the No token won
            "AMBIGUOUS" if no clear winner
        """
        for token in market.tokens:
            if token.winner is True:
                if token.outcome.lower() in ("yes", "y"):
                    return "YES"
                elif token.outcome.lower() in ("no", "n"):
                    return "NO"
                else:
                    # Non-standard outcome name but still a winner
                    return "YES"
        return "AMBIGUOUS"

    @staticmethod
    def _to_historical_record(
        market: GammaMarket,
        resolution: str,
    ) -> HistoricalMarketRecord:
        """Convert a GammaMarket to a HistoricalMarketRecord."""
        # Build prices map
        prices: dict[str, float] = {}
        tokens_data: list[dict[str, Any]] = []
        for t in market.tokens:
            prices[t.outcome] = t.price
            tokens_data.append({
                "token_id": t.token_id,
                "outcome": t.outcome,
                "price": t.price,
                "winner": t.winner,
            })

        outcomes = [t.outcome for t in market.tokens]

        return HistoricalMarketRecord(
            condition_id=market.condition_id or market.id,
            question=market.question,
            description=market.description,
            category=market.category,
            market_type=market.market_type,
            resolution=resolution,
            resolved_at=(
                market.end_date.isoformat() if market.end_date else ""
            ),
            created_at=(
                market.created_at.isoformat() if market.created_at else ""
            ),
            end_date=(
                market.end_date.isoformat() if market.end_date else ""
            ),
            volume_usd=market.volume,
            liquidity_usd=market.liquidity,
            slug=market.slug,
            outcomes_json=json.dumps(outcomes),
            final_prices_json=json.dumps(prices),
            tokens_json=json.dumps(tokens_data),
            raw_json=json.dumps(market.raw, default=str),
            scraped_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
