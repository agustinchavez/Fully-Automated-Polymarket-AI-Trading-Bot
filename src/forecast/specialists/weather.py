"""Weather market specialist — GFS ensemble-based temperature forecasting.

Uses the Open-Meteo free API to fetch 31-member GFS ensemble forecasts.
For temperature threshold markets, the fraction of ensemble members
exceeding the threshold IS the probability estimate.

Completely bypasses the LLM pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

from src.connectors.rate_limiter import rate_limiter
from src.forecast.specialists.base import BaseSpecialist, SpecialistResult
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class WeatherQuery:
    """Parsed weather question components."""
    city: str
    lat: float
    lon: float
    date: str              # YYYY-MM-DD
    threshold_f: float     # Temperature threshold in Fahrenheit
    operator: str          # "above" | "below"
    days_ahead: int


# Top 50 US cities + common abbreviations
CITY_COORDS: dict[str, tuple[float, float]] = {
    "new york": (40.71, -74.01),
    "nyc": (40.71, -74.01),
    "new york city": (40.71, -74.01),
    "los angeles": (34.05, -118.24),
    "la": (34.05, -118.24),
    "chicago": (41.88, -87.63),
    "houston": (29.76, -95.37),
    "phoenix": (33.45, -112.07),
    "philadelphia": (39.95, -75.17),
    "philly": (39.95, -75.17),
    "san antonio": (29.42, -98.49),
    "san diego": (32.72, -117.16),
    "dallas": (32.78, -96.80),
    "san jose": (37.34, -121.89),
    "austin": (30.27, -97.74),
    "jacksonville": (30.33, -81.66),
    "fort worth": (32.76, -97.33),
    "columbus": (39.96, -82.99),
    "charlotte": (35.23, -80.84),
    "san francisco": (37.77, -122.42),
    "sf": (37.77, -122.42),
    "indianapolis": (39.77, -86.16),
    "seattle": (47.61, -122.33),
    "denver": (39.74, -104.99),
    "washington": (38.91, -77.04),
    "dc": (38.91, -77.04),
    "washington dc": (38.91, -77.04),
    "nashville": (36.16, -86.78),
    "oklahoma city": (35.47, -97.52),
    "el paso": (31.76, -106.44),
    "boston": (42.36, -71.06),
    "portland": (45.52, -122.68),
    "las vegas": (36.17, -115.14),
    "vegas": (36.17, -115.14),
    "memphis": (35.15, -90.05),
    "louisville": (38.25, -85.76),
    "baltimore": (39.29, -76.61),
    "milwaukee": (43.04, -87.91),
    "albuquerque": (35.08, -106.65),
    "tucson": (32.22, -110.93),
    "fresno": (36.74, -119.77),
    "sacramento": (38.58, -121.49),
    "mesa": (33.41, -111.83),
    "kansas city": (39.10, -94.58),
    "atlanta": (33.75, -84.39),
    "omaha": (41.26, -95.94),
    "colorado springs": (38.83, -104.82),
    "raleigh": (35.78, -78.64),
    "miami": (25.76, -80.19),
    "minneapolis": (44.98, -93.27),
    "tampa": (27.95, -82.46),
    "new orleans": (29.95, -90.07),
    "cleveland": (41.50, -81.69),
    "pittsburgh": (40.44, -79.99),
    "cincinnati": (39.10, -84.51),
    "st louis": (38.63, -90.20),
    "saint louis": (38.63, -90.20),
    "detroit": (42.33, -83.05),
    "honolulu": (21.31, -157.86),
    "anchorage": (61.22, -149.90),
}

# Regex patterns for temperature threshold questions
_TEMP_ABOVE_RE = re.compile(
    r"(?:will|does|can)\s+(?:the\s+)?(?:high\s+)?(?:temperature\s+)?(?:in\s+)?"
    r"(.+?)\s+(?:high\s+)?(?:temperature\s+)?"
    r"(?:exceed|be\s+above|be\s+over|reach|surpass|top|hit)\s+"
    r"(\d+)\s*(?:°?\s*[fF]|degrees?\s*(?:fahrenheit)?)",
    re.IGNORECASE,
)

_TEMP_BELOW_RE = re.compile(
    r"(?:will|does|can)\s+(?:the\s+)?(?:low\s+)?(?:temperature\s+)?(?:in\s+)?"
    r"(.+?)\s+(?:low\s+)?(?:temperature\s+)?"
    r"(?:be\s+below|drop\s+below|fall\s+below|be\s+under|dip\s+below)\s+"
    r"(\d+)\s*(?:°?\s*[fF]|degrees?\s*(?:fahrenheit)?)",
    re.IGNORECASE,
)

# Date extraction patterns
_DATE_MONTH_DAY_RE = re.compile(
    r"(?:on|by)\s+"
    r"(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+(\d{1,2})",
    re.IGNORECASE,
)

_DATE_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _fahrenheit_to_celsius(f: float) -> float:
    return (f - 32) * 5 / 9


class WeatherSpecialist(BaseSpecialist):
    """GFS ensemble-based weather forecasting specialist."""

    def __init__(self, config: Any):
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._api_base = getattr(
            config, "weather_api_base",
            "https://ensemble-api.open-meteo.com/v1/ensemble",
        )

    @property
    def name(self) -> str:
        return "weather"

    def can_handle(self, classification: Any, question: str) -> bool:
        """Handle WEATHER markets with parseable temperature threshold."""
        cat = getattr(classification, "category", "")
        if cat != "WEATHER":
            return False
        return self._parse_question(question) is not None

    def _parse_question(self, question: str) -> WeatherQuery | None:
        """Extract city, date, temperature threshold, and operator."""
        # Try "above" patterns first
        operator = "above"
        m = _TEMP_ABOVE_RE.search(question)
        if not m:
            operator = "below"
            m = _TEMP_BELOW_RE.search(question)
        if not m:
            return None

        city_raw = m.group(1).strip().rstrip("'s").strip()
        threshold_f = float(m.group(2))

        # City lookup (case-insensitive)
        city_key = city_raw.lower()
        coords = CITY_COORDS.get(city_key)
        if coords is None:
            # Try partial match
            for name, c in CITY_COORDS.items():
                if city_key in name or name in city_key:
                    coords = c
                    city_key = name
                    break
        if coords is None:
            return None

        lat, lon = coords

        # Date extraction
        target_date = self._extract_date(question)
        if target_date is None:
            # Default to tomorrow
            target_date = datetime.now() + timedelta(days=1)

        date_str = target_date.strftime("%Y-%m-%d")
        days_ahead = max(0, (target_date.date() - datetime.now().date()).days)

        return WeatherQuery(
            city=city_key,
            lat=lat,
            lon=lon,
            date=date_str,
            threshold_f=threshold_f,
            operator=operator,
            days_ahead=days_ahead,
        )

    def _extract_date(self, question: str) -> datetime | None:
        """Extract target date from question text."""
        # ISO format: 2024-07-04
        m = _DATE_ISO_RE.search(question)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y-%m-%d")
            except ValueError:
                pass

        # Month day: "June 15", "July 4th"
        m = _DATE_MONTH_DAY_RE.search(question)
        if m:
            month_name = m.group(1).lower()
            day = int(m.group(2))
            month = _MONTH_MAP.get(month_name)
            if month:
                year = datetime.now().year
                try:
                    dt = datetime(year, month, day)
                    # If date is in the past, assume next year
                    if dt.date() < datetime.now().date():
                        dt = datetime(year + 1, month, day)
                    return dt
                except ValueError:
                    pass

        return None

    def _count_threshold_exceedance(
        self,
        ensemble_data: dict,
        threshold_f: float,
        operator: str,
    ) -> float:
        """Count fraction of ensemble members exceeding/below the threshold.

        The Open-Meteo ensemble API returns hourly data per member.
        We extract the daily high (max hourly temp) per member and check
        against the threshold.
        """
        threshold_c = _fahrenheit_to_celsius(threshold_f)

        # Parse ensemble member temperatures
        hourly = ensemble_data.get("hourly", {})

        # Find all ensemble member keys (temperature_2m_member01, etc.)
        member_keys = [
            k for k in hourly
            if k.startswith("temperature_2m_member")
        ]

        if not member_keys:
            log.warning("weather.no_ensemble_members")
            return 0.5  # Neutral when no data

        # For each member, get the daily max temperature
        exceedance_count = 0
        total_members = len(member_keys)

        for key in member_keys:
            temps = hourly[key]
            if not temps:
                continue
            # Filter out None values
            valid_temps = [t for t in temps if t is not None]
            if not valid_temps:
                continue

            if operator == "above":
                daily_max = max(valid_temps)
                if daily_max > threshold_c:
                    exceedance_count += 1
            else:  # below
                daily_min = min(valid_temps)
                if daily_min < threshold_c:
                    exceedance_count += 1

        if total_members == 0:
            return 0.5

        return exceedance_count / total_members

    async def _fetch_ensemble(
        self, lat: float, lon: float, date: str,
    ) -> dict:
        """Fetch GFS ensemble data from Open-Meteo."""
        await rate_limiter.get("open_meteo").acquire()

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)

        resp = await self._client.get(
            self._api_base,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "models": "gfs_seamless",
                "start_date": date,
                "end_date": date,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def forecast(
        self,
        market: Any,
        features: Any,
        classification: Any,
    ) -> SpecialistResult:
        """Produce a weather probability forecast."""
        query = self._parse_question(market.question)
        if query is None:
            raise ValueError(f"Cannot parse weather question: {market.question}")

        ensemble_data = await self._fetch_ensemble(
            query.lat, query.lon, query.date,
        )
        fraction = self._count_threshold_exceedance(
            ensemble_data, query.threshold_f, query.operator,
        )

        # Clamp probability
        probability = max(0.01, min(0.99, fraction))

        # Confidence based on forecast horizon
        if query.days_ahead <= 3:
            confidence = "HIGH"
        elif query.days_ahead <= 7:
            confidence = "HIGH"
        elif query.days_ahead <= 14:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Evidence quality degrades with forecast horizon
        evidence_quality = max(0.3, 1.0 - query.days_ahead * 0.05)

        member_count = len([
            k for k in ensemble_data.get("hourly", {})
            if k.startswith("temperature_2m_member")
        ])
        exceedance_count = int(round(fraction * member_count))

        return SpecialistResult(
            probability=probability,
            confidence_level=confidence,
            reasoning=(
                f"GFS ensemble: {exceedance_count}/{member_count} members "
                f"{query.operator} {query.threshold_f:.0f}°F for {query.city} "
                f"on {query.date}"
            ),
            evidence_quality=evidence_quality,
            specialist_name="weather",
            specialist_metadata={
                "ensemble_members": member_count,
                "exceedance_count": exceedance_count,
                "threshold_f": query.threshold_f,
                "operator": query.operator,
                "city": query.city,
                "days_ahead": query.days_ahead,
                "date": query.date,
            },
            bypasses_llm=True,
            key_evidence=[{
                "source": "Open-Meteo GFS Ensemble",
                "text": (
                    f"{exceedance_count}/{member_count} ensemble members forecast "
                    f"temperature {query.operator} {query.threshold_f:.0f}°F"
                ),
            }],
            invalidation_triggers=[
                "Significant weather system change",
                "Model initialization data update",
            ],
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
