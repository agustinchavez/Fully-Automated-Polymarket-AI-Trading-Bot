"""Economic & political calendar awareness.

Monitors upcoming scheduled events (FOMC, CPI, earnings) and injects
temporal context into the LLM prompt. Reduces position size when a
high-impact event is within 24 hours.

Data sources: FRED release calendar (free, no key).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)

# Common keywords per category for event matching
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "FED": ["fomc", "federal reserve", "interest rate", "fed rate", "monetary policy"],
    "ECONOMIC": ["cpi", "inflation", "gdp", "jobs", "unemployment", "nonfarm", "ppi", "retail sales"],
    "EARNINGS": ["earnings", "quarterly report", "revenue"],
    "POLITICAL": ["supreme court", "congress", "senate", "election", "vote", "impeach"],
}


@dataclass
class CalendarEvent:
    """A scheduled economic or political event."""
    name: str
    date: datetime
    category: str       # 'FED' | 'ECONOMIC' | 'EARNINGS' | 'POLITICAL'
    hours_away: float
    impact: str          # 'low' | 'medium' | 'high'
    keywords: list[str] = field(default_factory=list)


class EventCalendar:
    """Calendar of upcoming economic and political events."""

    def __init__(self, refresh_interval_hours: int = 6, lookahead_days: int = 14) -> None:
        self._events: list[CalendarEvent] = []
        self._last_refresh: float = 0.0
        self._interval = refresh_interval_hours * 3600
        self._lookahead_days = lookahead_days

    async def refresh(self) -> None:
        """Refresh events from data sources."""
        now = time.time()
        if now - self._last_refresh < self._interval:
            return

        events: list[CalendarEvent] = []
        try:
            events += await self._fetch_fred_releases()
        except Exception as exc:
            log.warning("calendar.fred_fetch_failed", error=str(exc))

        self._events = events
        self._last_refresh = now
        log.info("calendar.refresh_complete", event_count=len(events))

    async def _fetch_fred_releases(self) -> list[CalendarEvent]:
        """Fetch upcoming releases from FRED API."""
        import os

        fred_key = os.environ.get("FRED_API_KEY", "")
        if not fred_key:
            return []

        import aiohttp

        now = datetime.now(timezone.utc)
        end_date = now.replace(day=min(now.day + self._lookahead_days, 28))

        url = "https://api.stlouisfed.org/fred/releases/dates"
        params = {
            "api_key": fred_key,
            "file_type": "json",
            "realtime_start": now.strftime("%Y-%m-%d"),
            "realtime_end": end_date.strftime("%Y-%m-%d"),
            "limit": "50",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

        events: list[CalendarEvent] = []
        for item in data.get("release_dates", []):
            release_name = item.get("release_name", "")
            date_str = item.get("date", "")
            if not date_str:
                continue

            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            hours = (dt - now).total_seconds() / 3600
            if hours < 0:
                continue

            # Determine impact and category
            name_lower = release_name.lower()
            impact = "high" if any(
                kw in name_lower for kw in ["fomc", "cpi", "employment", "gdp", "nonfarm"]
            ) else "medium"

            category = "ECONOMIC"
            if "fomc" in name_lower or "fed" in name_lower:
                category = "FED"

            keywords = [kw for kw in _CATEGORY_KEYWORDS.get(category, [])
                        if kw in name_lower]

            events.append(CalendarEvent(
                name=release_name,
                date=dt,
                category=category,
                hours_away=round(hours, 1),
                impact=impact,
                keywords=keywords,
            ))

        return events

    def get_events_for_market(
        self,
        question: str,
        category: str,
    ) -> list[CalendarEvent]:
        """Return events relevant to a market question."""
        q = question.lower()
        return [
            e for e in self._events
            if any(kw in q for kw in e.keywords)
            or e.category.upper() == category.upper()
        ]
