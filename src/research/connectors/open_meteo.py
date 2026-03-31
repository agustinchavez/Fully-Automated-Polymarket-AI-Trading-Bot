"""Open-Meteo research connector — weather forecast data for WEATHER markets.

Uses the free Open-Meteo API (no key required, 10K calls/day) to fetch
7-day weather forecasts with daily precipitation, wind speed, and temperature.
Falls back to the Open-Meteo geocoding API for cities not in the static map.
"""

from __future__ import annotations

import re
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.forecast.specialists.weather import CITY_COORDS
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

_FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"
_GEOCODE_BASE = "https://geocoding-api.open-meteo.com/v1/search"

_WEATHER_KEYWORDS: list[str] = [
    "hurricane", "storm", "flood", "temperature", "rainfall",
    "weather", "tornado", "drought", "heat wave", "snowfall",
    "precipitation", "wind", "rain", "snow", "hail",
]

# City extraction regex — "in <city>" or "for <city>"
_CITY_RE = re.compile(
    r"(?:in|for|at|near)\s+([A-Za-z\s]+?)(?:\s+(?:on|by|before|after|this|next|exceed|reach|temperature|high|low|will|today|tomorrow)|\?|$)",
    re.IGNORECASE,
)


class OpenMeteoConnector(BaseResearchConnector):
    """Fetch weather forecast data from Open-Meteo (free, no key)."""

    @property
    def name(self) -> str:
        return "open_meteo"

    def relevant_categories(self) -> set[str]:
        return {"WEATHER"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        """Check if this connector should run for the given question."""
        if market_type in self.relevant_categories():
            return True
        return self._question_matches_keywords(question, _WEATHER_KEYWORDS)

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        if not self.is_relevant(question, market_type):
            return []

        lat, lon, city_name = await self._resolve_location(question)
        if lat is None or lon is None:
            return []

        await rate_limiter.get("open_meteo").acquire()

        client = self._get_client()
        resp = await client.get(
            _FORECAST_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum,windspeed_10m_max,temperature_2m_max,temperature_2m_min",
                "forecast_days": 7,
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("windspeed_10m_max", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])

        if not dates:
            return []

        # Build readable content
        lines = [
            f"Open-Meteo 7-Day Forecast for {city_name.title()}",
            f"Coordinates: {lat:.2f}°N, {lon:.2f}°E",
            "",
        ]
        for i, d in enumerate(dates):
            t_hi = f"{temp_max[i]:.1f}°C" if i < len(temp_max) else "N/A"
            t_lo = f"{temp_min[i]:.1f}°C" if i < len(temp_min) else "N/A"
            p = f"{precip[i]:.1f}mm" if i < len(precip) else "N/A"
            w = f"{wind[i]:.1f}km/h" if i < len(wind) else "N/A"
            lines.append(f"{d}: High {t_hi}, Low {t_lo}, Precip {p}, Wind {w}")

        content = "\n".join(lines)
        snippet = (
            f"7-day forecast for {city_name.title()}: "
            f"High {temp_max[0]:.1f}°C, Precip {precip[0]:.1f}mm"
            if temp_max and precip else f"7-day forecast for {city_name.title()}"
        )

        return [
            self._make_source(
                title=f"Open-Meteo Forecast: {city_name.title()}",
                url=f"{_FORECAST_BASE}?latitude={lat}&longitude={lon}",
                snippet=snippet,
                publisher="Open-Meteo",
                content=content,
            )
        ]

    async def _resolve_location(
        self, question: str,
    ) -> tuple[float | None, float | None, str]:
        """Extract city from question, resolve to lat/lon."""
        city_name = self._extract_city(question)
        if not city_name:
            return None, None, ""

        city_key = city_name.lower().strip()

        # Try static map first
        coords = CITY_COORDS.get(city_key)
        if coords is None:
            for name, c in CITY_COORDS.items():
                if city_key in name or name in city_key:
                    coords = c
                    city_name = name
                    break

        if coords:
            return coords[0], coords[1], city_name

        # Fall back to Open-Meteo geocoding
        return await self._geocode(city_name)

    def _extract_city(self, question: str) -> str:
        """Extract city name from question text."""
        # Try matching any known city name directly (longest match first)
        q_lower = question.lower()
        for city in sorted(CITY_COORDS.keys(), key=len, reverse=True):
            if city in q_lower:
                return city
        # Fall back to regex extraction
        m = _CITY_RE.search(question)
        if m:
            return m.group(1).strip()
        return ""

    async def _geocode(
        self, city: str,
    ) -> tuple[float | None, float | None, str]:
        """Geocode a city name via Open-Meteo's free geocoding API."""
        try:
            client = self._get_client()
            resp = await client.get(
                _GEOCODE_BASE,
                params={"name": city, "count": 1},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                r = results[0]
                return r["latitude"], r["longitude"], r.get("name", city)
        except Exception as e:
            log.warning("open_meteo.geocode_failed", city=city, error=str(e))
        return None, None, city
