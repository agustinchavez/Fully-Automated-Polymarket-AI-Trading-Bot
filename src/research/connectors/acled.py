"""ACLED armed conflict connector — conflict event data for GEOPOLITICS markets.

Uses the ACLED (Armed Conflict Location & Event Data) API to fetch recent
conflict events for countries/regions mentioned in prediction market questions.
Provides event counts, fatality estimates, and trend analysis.

Requires an API key (free academic access at acleddata.com).
Rate-limited via a dedicated ``acled`` bucket.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_ACLED_BASE = "https://api.acleddata.com/acled/read"

# Country name normalization for ACLED API
_COUNTRY_ALIASES: dict[str, str] = {
    "ukraine": "Ukraine",
    "russia": "Russia",
    "israel": "Israel",
    "palestine": "Palestine",
    "gaza": "Palestine",
    "syria": "Syria",
    "iraq": "Iraq",
    "iran": "Iran",
    "yemen": "Yemen",
    "sudan": "Sudan",
    "ethiopia": "Ethiopia",
    "myanmar": "Myanmar",
    "burma": "Myanmar",
    "afghanistan": "Afghanistan",
    "congo": "Democratic Republic of Congo",
    "drc": "Democratic Republic of Congo",
    "somalia": "Somalia",
    "nigeria": "Nigeria",
    "libya": "Libya",
    "mali": "Mali",
    "haiti": "Haiti",
    "china": "China",
    "taiwan": "Taiwan",
    "north korea": "North Korea",
    "south korea": "South Korea",
    "india": "India",
    "pakistan": "Pakistan",
    "mexico": "Mexico",
    "colombia": "Colombia",
    "venezuela": "Venezuela",
}

# Regions for broader queries
_REGION_CODES: dict[str, int] = {
    "middle east": 11,
    "north africa": 1,
    "west africa": 4,
    "east africa": 2,
    "central africa": 3,
    "southern africa": 5,
    "south asia": 9,
    "southeast asia": 10,
    "central asia": 12,
    "europe": 13,
    "caucasus": 14,
    "central america": 15,
    "south america": 16,
}


class AcledConnector(BaseResearchConnector):
    """ACLED conflict event data for GEOPOLITICS markets."""

    @property
    def name(self) -> str:
        return "acled"

    def relevant_categories(self) -> set[str]:
        return {"GEOPOLITICS"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type != "GEOPOLITICS":
            return False
        # Need either a country or region match, plus conflict-related keywords
        q = question.lower()
        has_location = (
            self._extract_country(question) is not None
            or self._extract_region(question) is not None
        )
        conflict_keywords = [
            "war", "conflict", "attack", "invasion", "cease",
            "military", "troops", "battle", "violence", "fighting",
            "escalat", "offensive", "casualties", "rebel",
        ]
        has_conflict = any(kw in q for kw in conflict_keywords)
        return has_location and has_conflict

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        api_key = getattr(self._config, "acled_api_key", "") or os.environ.get(
            "ACLED_API_KEY", ""
        )
        acled_email = os.environ.get("ACLED_EMAIL", "")
        if not api_key or not acled_email:
            return []

        country = self._extract_country(question)
        region_code = self._extract_region(question)

        if not country and region_code is None:
            return []

        await rate_limiter.get("acled").acquire()

        client = self._get_client(timeout=15.0)
        lookback = getattr(self._config, "acled_lookback_days", 30)
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=lookback)
        ).strftime("%Y-%m-%d")

        params: dict[str, Any] = {
            "key": api_key,
            "email": acled_email,
            "event_date": f"{start_date}|",
            "event_date_where": "BETWEEN",
            "limit": 500,
        }
        if country:
            params["country"] = country
        elif region_code is not None:
            params["region"] = region_code

        resp = await client.get(_ACLED_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("data", [])
        if not events:
            return []

        # Aggregate statistics
        total_events = len(events)
        total_fatalities = sum(int(e.get("fatalities", 0)) for e in events)

        # Event type breakdown
        type_counts: dict[str, int] = {}
        for e in events:
            etype = e.get("event_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        # Trend: compare first half vs second half of period
        mid = total_events // 2
        first_half = total_events - mid  # older events
        second_half = mid  # newer events
        if first_half > 0:
            trend_ratio = second_half / first_half
            if trend_ratio > 1.2:
                trend = "escalating"
            elif trend_ratio < 0.8:
                trend = "de-escalating"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"

        location = country or f"Region {region_code}"

        # Top event types
        top_types = sorted(type_counts.items(), key=lambda x: -x[1])[:3]
        type_summary = ", ".join(f"{t}: {c}" for t, c in top_types)

        content = (
            f"ACLED Conflict Data: {location}\n"
            f"  Period: last {lookback} days\n"
            f"  Total events: {total_events}\n"
            f"  Total fatalities: {total_fatalities}\n"
            f"  Event types: {type_summary}\n"
            f"  Trend: {trend}\n"
            f"  Source: ACLED (Armed Conflict Location & Event Data)"
        )

        return [
            self._make_source(
                title=f"ACLED: Conflict Events — {location}",
                url=f"https://acleddata.com/dashboard/#/dashboard",
                snippet=(
                    f"{location}: {total_events} conflict events, "
                    f"{total_fatalities} fatalities ({lookback}d, {trend})"
                ),
                publisher="ACLED",
                content=content,
                authority_score=0.80,
                raw={
                    "behavioral_signal": {
                        "source": "acled",
                        "signal_type": "conflict_events",
                        "value": total_events,
                        "fatalities": total_fatalities,
                        "trend": trend,
                        "location": location,
                        "period_days": lookback,
                        "event_types": dict(top_types),
                    }
                },
            )
        ]

    @staticmethod
    def _extract_country(question: str) -> str | None:
        """Extract ACLED country name from question text."""
        q = question.lower()
        for keyword, country in _COUNTRY_ALIASES.items():
            if keyword in q:
                return country
        return None

    @staticmethod
    def _extract_region(question: str) -> int | None:
        """Extract ACLED region code from question text."""
        q = question.lower()
        for keyword, code in _REGION_CODES.items():
            if keyword in q:
                return code
        return None
