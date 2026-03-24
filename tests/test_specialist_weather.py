"""Tests for Phase 4 Batch A: Specialist infrastructure + Weather specialist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.forecast.specialists.base import BaseSpecialist, SpecialistResult
from src.forecast.specialists.weather import (
    WeatherSpecialist,
    WeatherQuery,
    CITY_COORDS,
    _fahrenheit_to_celsius,
)
from src.forecast.specialist_router import SpecialistRouter


# ── Helpers ──────────────────────────────────────────────────────────────


@dataclass
class FakeClassification:
    category: str = "WEATHER"
    subcategory: str = "forecast"


@dataclass
class FakeMarket:
    id: str = "weather-001"
    question: str = "Will NYC high temperature exceed 75°F on June 15?"
    market_type: str = "WEATHER"
    resolution_source: str = "NOAA"
    category: str = "WEATHER"


@dataclass
class FakeFeatures:
    implied_probability: float = 0.50
    question: str = "Will NYC high temperature exceed 75°F on June 15?"


@dataclass
class FakeConfig:
    enabled: bool = True
    enabled_specialists: list[str] = field(default_factory=lambda: ["weather"])
    weather_min_edge: float = 0.08
    weather_api_base: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    crypto_min_edge: float = 0.04
    crypto_candle_source: str = "binance"
    politics_polling_weight: float = 0.6


def _make_ensemble_response(
    member_count: int = 31,
    temps_per_member: list[float] | None = None,
) -> dict:
    """Build a mock Open-Meteo ensemble API response."""
    hourly: dict[str, Any] = {
        "time": ["2024-06-15T00:00", "2024-06-15T06:00",
                 "2024-06-15T12:00", "2024-06-15T18:00"],
    }
    for i in range(member_count):
        key = f"temperature_2m_member{i:02d}"
        if temps_per_member and i < len(temps_per_member):
            # Use specified temp as the max hourly value
            max_t = temps_per_member[i]
            hourly[key] = [max_t - 5, max_t - 2, max_t, max_t - 3]
        else:
            # Default: 24°C (75.2°F)
            hourly[key] = [18.0, 20.0, 24.0, 22.0]
    return {"hourly": hourly}


# ── TestSpecialistResult ──────────────────────────────────────────────────


class TestSpecialistResult:

    def test_to_ensemble_result_model_name(self) -> None:
        """Conversion creates specialist:name model forecast."""
        sr = SpecialistResult(
            probability=0.87,
            confidence_level="HIGH",
            reasoning="Test",
            evidence_quality=0.9,
            specialist_name="weather",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.individual_forecasts[0].model_name == "specialist:weather"

    def test_to_ensemble_result_probability(self) -> None:
        sr = SpecialistResult(
            probability=0.87, confidence_level="HIGH",
            reasoning="r", evidence_quality=0.9, specialist_name="weather",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.model_probability == 0.87

    def test_to_ensemble_result_spread_zero(self) -> None:
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.5, specialist_name="test",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.spread == 0.0

    def test_to_ensemble_result_agreement_one(self) -> None:
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.5, specialist_name="test",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.agreement_score == 1.0

    def test_to_ensemble_result_aggregation_method(self) -> None:
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.5, specialist_name="crypto_ta",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.aggregation_method == "specialist:crypto_ta"

    def test_to_ensemble_result_models_count(self) -> None:
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.5, specialist_name="test",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.models_succeeded == 1
        assert er.models_failed == 0

    def test_to_base_rate_match(self) -> None:
        sr = SpecialistResult(
            probability=0.65, confidence_level="MEDIUM",
            reasoning="Polling avg", evidence_quality=0.8,
            specialist_name="politics", bypasses_llm=False,
        )
        brm = BaseSpecialist.to_base_rate_match(sr)
        assert brm.base_rate == 0.65
        assert brm.confidence == 0.8
        assert brm.source == "politics"

    def test_specialist_result_defaults(self) -> None:
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.5, specialist_name="test",
        )
        assert sr.bypasses_llm is True
        assert sr.key_evidence == []
        assert sr.invalidation_triggers == []
        assert sr.specialist_metadata == {}


# ── TestWeatherQuestionParsing ────────────────────────────────────────────


class TestWeatherQuestionParsing:

    def setup_method(self) -> None:
        self.specialist = WeatherSpecialist(FakeConfig())

    def test_parse_nyc_temperature_above(self) -> None:
        q = "Will NYC high temperature exceed 75°F on June 15?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.city == "nyc"
        assert result.threshold_f == 75.0
        assert result.operator == "above"

    def test_parse_chicago_high_exceed(self) -> None:
        q = "Will Chicago's high exceed 90°F on July 4?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.city == "chicago"
        assert result.threshold_f == 90.0
        assert result.operator == "above"

    def test_parse_temperature_below(self) -> None:
        q = "Will the temperature in Denver drop below 32°F on December 25?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.city == "denver"
        assert result.threshold_f == 32.0
        assert result.operator == "below"

    def test_parse_degrees_word(self) -> None:
        q = "Will Boston's temperature be above 80 degrees fahrenheit on August 1?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.threshold_f == 80.0
        assert result.operator == "above"

    def test_parse_unknown_city_returns_none(self) -> None:
        q = "Will Timbuktu's high exceed 100°F on March 1?"
        result = self.specialist._parse_question(q)
        assert result is None

    def test_parse_no_threshold_returns_none(self) -> None:
        q = "Will it rain in NYC on June 15?"
        result = self.specialist._parse_question(q)
        assert result is None

    def test_parse_date_extraction_month_day(self) -> None:
        q = "Will Miami's high exceed 95°F on September 10?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert "-09-10" in result.date

    def test_parse_date_iso_format(self) -> None:
        q = "Will Seattle's temperature be above 70°F on 2025-07-20?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.date == "2025-07-20"

    def test_city_lookup_case_insensitive(self) -> None:
        q = "Will NEW YORK's high exceed 80°F on July 1?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.lat == pytest.approx(40.71, abs=0.1)

    def test_city_abbreviation_sf(self) -> None:
        q = "Will SF's high exceed 70°F on August 15?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.city == "sf"

    def test_parse_reach_keyword(self) -> None:
        q = "Will the temperature in Atlanta reach 95°F on June 20?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.operator == "above"

    def test_can_handle_weather_category(self) -> None:
        cls = FakeClassification(category="WEATHER")
        q = "Will NYC high temperature exceed 75°F on June 15?"
        assert self.specialist.can_handle(cls, q) is True

    def test_can_handle_non_weather_rejected(self) -> None:
        cls = FakeClassification(category="ELECTION")
        q = "Will NYC high temperature exceed 75°F on June 15?"
        assert self.specialist.can_handle(cls, q) is False


# ── TestWeatherEnsembleProbability ────────────────────────────────────────


class TestWeatherEnsembleProbability:

    def setup_method(self) -> None:
        self.specialist = WeatherSpecialist(FakeConfig())

    def test_all_members_above_threshold(self) -> None:
        """All members exceed threshold → probability near 1.0."""
        # 31 members, all with max 30°C (86°F), threshold 75°F (23.9°C)
        data = _make_ensemble_response(31, [30.0] * 31)
        frac = self.specialist._count_threshold_exceedance(data, 75.0, "above")
        assert frac == pytest.approx(1.0)

    def test_no_members_above_threshold(self) -> None:
        """No members exceed threshold → probability near 0.0."""
        # 31 members, all with max 20°C (68°F), threshold 75°F (23.9°C)
        data = _make_ensemble_response(31, [20.0] * 31)
        frac = self.specialist._count_threshold_exceedance(data, 75.0, "above")
        assert frac == pytest.approx(0.0)

    def test_fraction_calculation_27_of_31(self) -> None:
        """27/31 members exceed → ~87%."""
        temps = [30.0] * 27 + [20.0] * 4  # 27 above, 4 below
        data = _make_ensemble_response(31, temps)
        frac = self.specialist._count_threshold_exceedance(data, 75.0, "above")
        assert frac == pytest.approx(27 / 31, abs=0.01)

    def test_below_operator(self) -> None:
        """Below operator checks daily min."""
        # All members: min hourly is threshold - 5 = 13.0°C (55.4°F)
        # threshold 60°F = 15.6°C, so min(13°C) < threshold → exceedance
        temps = [18.0] * 31  # max is 18, but the response builder puts min = max-5 = 13
        data = _make_ensemble_response(31, temps)
        frac = self.specialist._count_threshold_exceedance(data, 60.0, "below")
        assert frac == pytest.approx(1.0)

    def test_below_operator_none_below(self) -> None:
        """Below operator: all members above threshold."""
        temps = [30.0] * 31  # min = 25°C (77°F)
        data = _make_ensemble_response(31, temps)
        frac = self.specialist._count_threshold_exceedance(data, 32.0, "below")
        # min is 25°C = 77°F, threshold is 32°F = 0°C → no members below
        assert frac == pytest.approx(0.0)

    def test_no_ensemble_members_neutral(self) -> None:
        """No ensemble members → return 0.5 (neutral)."""
        data = {"hourly": {"time": ["2024-06-15T00:00"]}}
        frac = self.specialist._count_threshold_exceedance(data, 75.0, "above")
        assert frac == pytest.approx(0.5)

    def test_mixed_members(self) -> None:
        """10/20 members exceed → 0.5."""
        temps = [30.0] * 10 + [20.0] * 10
        data = _make_ensemble_response(20, temps)
        frac = self.specialist._count_threshold_exceedance(data, 75.0, "above")
        assert frac == pytest.approx(0.5)

    def test_fahrenheit_to_celsius(self) -> None:
        assert _fahrenheit_to_celsius(32) == pytest.approx(0.0)
        assert _fahrenheit_to_celsius(212) == pytest.approx(100.0)


# ── TestWeatherAPIIntegration ─────────────────────────────────────────────


class TestWeatherAPIIntegration:

    @pytest.mark.asyncio
    async def test_forecast_calls_api_and_returns_result(self) -> None:
        """End-to-end: mock API → valid SpecialistResult."""
        specialist = WeatherSpecialist(FakeConfig())
        mock_resp = _make_ensemble_response(31, [30.0] * 27 + [20.0] * 4)

        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            # Use a short-horizon query so confidence is predictable
            query = WeatherQuery(
                city="nyc", lat=40.71, lon=-74.01, date="2024-06-15",
                threshold_f=75.0, operator="above", days_ahead=3,
            )
            with patch.object(specialist, "_parse_question", return_value=query):
                result = await specialist.forecast(
                    FakeMarket(), FakeFeatures(), FakeClassification(),
                )

        assert result.specialist_name == "weather"
        assert result.probability == pytest.approx(27 / 31, abs=0.02)
        assert result.bypasses_llm is True
        assert result.confidence_level == "HIGH"
        assert result.specialist_metadata["ensemble_members"] == 31

    @pytest.mark.asyncio
    async def test_forecast_high_confidence_short_horizon(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        mock_resp = _make_ensemble_response(31, [25.0] * 31)

        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            # Patch _parse_question to return a short-horizon query
            query = WeatherQuery(
                city="nyc", lat=40.71, lon=-74.01, date="2024-06-15",
                threshold_f=75.0, operator="above", days_ahead=2,
            )
            with patch.object(specialist, "_parse_question", return_value=query):
                result = await specialist.forecast(
                    FakeMarket(), FakeFeatures(), FakeClassification(),
                )
        assert result.confidence_level == "HIGH"
        assert result.evidence_quality == pytest.approx(0.9, abs=0.1)

    @pytest.mark.asyncio
    async def test_forecast_medium_confidence_long_horizon(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        mock_resp = _make_ensemble_response(31, [25.0] * 31)

        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            query = WeatherQuery(
                city="nyc", lat=40.71, lon=-74.01, date="2024-06-15",
                threshold_f=75.0, operator="above", days_ahead=10,
            )
            with patch.object(specialist, "_parse_question", return_value=query):
                result = await specialist.forecast(
                    FakeMarket(), FakeFeatures(), FakeClassification(),
                )
        assert result.confidence_level == "MEDIUM"

    @pytest.mark.asyncio
    async def test_forecast_low_confidence_very_long_horizon(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        mock_resp = _make_ensemble_response(31, [25.0] * 31)

        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            query = WeatherQuery(
                city="nyc", lat=40.71, lon=-74.01, date="2024-06-15",
                threshold_f=75.0, operator="above", days_ahead=20,
            )
            with patch.object(specialist, "_parse_question", return_value=query):
                result = await specialist.forecast(
                    FakeMarket(), FakeFeatures(), FakeClassification(),
                )
        assert result.confidence_level == "LOW"

    @pytest.mark.asyncio
    async def test_unparseable_question_raises(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        market = FakeMarket(question="Will it rain tomorrow?")
        with pytest.raises(ValueError, match="Cannot parse"):
            await specialist.forecast(
                market, FakeFeatures(), FakeClassification(),
            )

    @pytest.mark.asyncio
    async def test_key_evidence_populated(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        mock_resp = _make_ensemble_response(31, [30.0] * 31)
        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
        assert len(result.key_evidence) == 1
        assert "Open-Meteo" in result.key_evidence[0]["source"]

    @pytest.mark.asyncio
    async def test_probability_clamped(self) -> None:
        """Probability is clamped to [0.01, 0.99]."""
        specialist = WeatherSpecialist(FakeConfig())
        # All members exceed → fraction = 1.0 → clamped to 0.99
        mock_resp = _make_ensemble_response(31, [40.0] * 31)
        with patch.object(specialist, "_fetch_ensemble", new_callable=AsyncMock) as mock:
            mock.return_value = mock_resp
            result = await specialist.forecast(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
        assert result.probability <= 0.99

    @pytest.mark.asyncio
    async def test_close_releases_client(self) -> None:
        specialist = WeatherSpecialist(FakeConfig())
        mock_client = AsyncMock()
        specialist._client = mock_client
        await specialist.close()
        mock_client.aclose.assert_called_once()
        assert specialist._client is None


# ── TestSpecialistRouter ──────────────────────────────────────────────────


class TestSpecialistRouter:

    def test_router_matches_weather(self) -> None:
        """Router finds weather specialist for WEATHER market."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = MagicMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_specialist.can_handle.return_value = True
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig())
            result = router.match(FakeClassification(), "Will NYC exceed 75°F?")
            assert result is not None
            assert result.name == "weather"

    def test_router_no_match_for_election(self) -> None:
        """Router returns None for non-matching category."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = MagicMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_specialist.can_handle.return_value = False
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig())
            result = router.match(
                FakeClassification(category="ELECTION"),
                "Will Biden win?",
            )
            assert result is None

    def test_router_disabled_config(self) -> None:
        """Router with empty enabled_specialists has no specialists."""
        config = FakeConfig(enabled_specialists=[])
        router = SpecialistRouter(config)
        assert router.match(FakeClassification(), "test?") is None

    @pytest.mark.asyncio
    async def test_router_route_returns_result(self) -> None:
        """route() returns SpecialistResult on match."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = AsyncMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_specialist.can_handle.return_value = True
            mock_specialist.forecast.return_value = SpecialistResult(
                probability=0.87, confidence_level="HIGH",
                reasoning="test", evidence_quality=0.9,
                specialist_name="weather",
            )
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig())
            result = await router.route(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
            assert result is not None
            assert result.probability == 0.87

    @pytest.mark.asyncio
    async def test_router_route_returns_none_on_no_match(self) -> None:
        config = FakeConfig(enabled_specialists=[])
        router = SpecialistRouter(config)
        result = await router.route(
            FakeMarket(), FakeFeatures(), FakeClassification(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_router_falls_back_on_error(self) -> None:
        """Specialist error → None (fall back to ensemble)."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = AsyncMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_specialist.can_handle.return_value = True
            mock_specialist.forecast.side_effect = RuntimeError("API down")
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig())
            result = await router.route(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_router_close_delegates(self) -> None:
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = AsyncMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig())
            await router.close()
            mock_specialist.close.assert_called_once()

    def test_router_unknown_specialist_ignored(self) -> None:
        """Unknown specialist name is safely ignored."""
        config = FakeConfig(enabled_specialists=["unknown_specialist"])
        router = SpecialistRouter(config)
        assert len(router._specialists) == 0

    def test_router_multiple_specialists(self) -> None:
        """Multiple specialists can be loaded."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_w = MagicMock(spec=BaseSpecialist)
            mock_w.name = "weather"
            mock_c = MagicMock(spec=BaseSpecialist)
            mock_c.name = "crypto_ta"
            mock_load.side_effect = [mock_w, mock_c]

            config = FakeConfig(enabled_specialists=["weather", "crypto_ta"])
            router = SpecialistRouter(config)
            assert len(router._specialists) == 2

    @pytest.mark.asyncio
    async def test_router_bypass_mode_result(self) -> None:
        """Bypass mode specialist result has bypasses_llm=True."""
        sr = SpecialistResult(
            probability=0.87, confidence_level="HIGH",
            reasoning="GFS", evidence_quality=0.9,
            specialist_name="weather", bypasses_llm=True,
        )
        assert sr.bypasses_llm is True

    @pytest.mark.asyncio
    async def test_router_augment_mode_result(self) -> None:
        """Augment mode specialist result has bypasses_llm=False."""
        sr = SpecialistResult(
            probability=0.65, confidence_level="MEDIUM",
            reasoning="Polling", evidence_quality=0.8,
            specialist_name="politics", bypasses_llm=False,
        )
        assert sr.bypasses_llm is False


# ── TestSpecialistsConfig ─────────────────────────────────────────────────


class TestSpecialistsConfig:

    def test_specialists_disabled_by_default(self) -> None:
        from src.config import SpecialistsConfig
        cfg = SpecialistsConfig()
        assert cfg.enabled is False

    def test_specialists_enabled_list_empty_by_default(self) -> None:
        from src.config import SpecialistsConfig
        cfg = SpecialistsConfig()
        assert cfg.enabled_specialists == []

    def test_weather_min_edge_default(self) -> None:
        from src.config import SpecialistsConfig
        cfg = SpecialistsConfig()
        assert cfg.weather_min_edge == 0.08

    def test_bot_config_has_specialists(self) -> None:
        from src.config import BotConfig
        cfg = BotConfig()
        assert hasattr(cfg, "specialists")
        assert cfg.specialists.enabled is False

    def test_backward_compatible(self) -> None:
        """BotConfig loads without specialists section in YAML."""
        from src.config import BotConfig
        cfg = BotConfig()
        assert cfg.specialists.enabled_specialists == []
        assert cfg.risk.bankroll > 0  # Other sections unaffected

    def test_rate_limiter_has_open_meteo(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS
        assert "open_meteo" in DEFAULT_LIMITS

    def test_rate_limiter_has_binance(self) -> None:
        from src.connectors.rate_limiter import DEFAULT_LIMITS
        assert "binance" in DEFAULT_LIMITS


# ── TestCityCoords ────────────────────────────────────────────────────────


class TestCityCoords:

    def test_major_cities_present(self) -> None:
        for city in ["new york", "los angeles", "chicago", "miami", "seattle"]:
            assert city in CITY_COORDS, f"{city} not in CITY_COORDS"

    def test_abbreviations_present(self) -> None:
        for abbr in ["nyc", "la", "sf", "dc"]:
            assert abbr in CITY_COORDS, f"{abbr} not in CITY_COORDS"

    def test_coords_reasonable(self) -> None:
        """All coords are within continental US + Alaska + Hawaii."""
        for city, (lat, lon) in CITY_COORDS.items():
            assert 18 < lat < 72, f"{city} lat {lat} out of range"
            assert -180 < lon < -60, f"{city} lon {lon} out of range"
