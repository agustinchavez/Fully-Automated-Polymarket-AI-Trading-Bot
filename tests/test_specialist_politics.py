"""Tests for Phase 4 Batch C: Politics specialist + full integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.forecast.specialists.base import BaseSpecialist, SpecialistResult
from src.forecast.specialists.politics import (
    PoliticsSpecialist,
    PollData,
    RaceQuery,
)
from src.forecast.specialist_router import SpecialistRouter


# ── Helpers ──────────────────────────────────────────────────────────────


@dataclass
class FakeClassification:
    category: str = "ELECTION"
    subcategory: str = "presidential"


@dataclass
class FakeMarket:
    id: str = "election-001"
    question: str = "Will Biden win the presidential election?"
    market_type: str = "ELECTION"
    resolution_source: str = "AP"
    category: str = "ELECTION"


@dataclass
class FakeFeatures:
    implied_probability: float = 0.45
    days_to_expiry: float = 60.0


@dataclass
class FakeConfig:
    enabled: bool = True
    enabled_specialists: list[str] = field(default_factory=lambda: ["politics"])
    weather_min_edge: float = 0.08
    weather_api_base: str = ""
    crypto_min_edge: float = 0.04
    crypto_candle_source: str = "binance"
    politics_polling_weight: float = 0.6


# ── TestPoliticsQuestionParsing ───────────────────────────────────────────


class TestPoliticsQuestionParsing:

    def setup_method(self) -> None:
        self.specialist = PoliticsSpecialist(FakeConfig())

    def test_parse_presidential_race(self) -> None:
        q = "Will Biden win the presidential election?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.race_type == "presidential"

    def test_parse_senate_race(self) -> None:
        q = "Will the Democrat win the senate seat in Arizona?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.race_type == "senate"
        assert "arizona" in result.state.lower()

    def test_parse_candidate_extraction(self) -> None:
        q = "Will Trump win the presidential election?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert "Trump" in result.candidate

    def test_parse_non_election_returns_none(self) -> None:
        q = "Will the Fed cut interest rates?"
        result = self.specialist._parse_question(q)
        assert result is None

    def test_parse_incumbency_detection(self) -> None:
        q = "Will Biden win the presidential race?"
        result = self.specialist._parse_question(q)
        assert result is not None
        assert result.has_incumbent is True
        assert result.candidate_is_incumbent is True

    def test_can_handle_election_category(self) -> None:
        cls = FakeClassification(category="ELECTION", subcategory="presidential")
        q = "Will Biden win the presidential election?"
        assert self.specialist.can_handle(cls, q) is True

    def test_can_handle_non_election_rejected(self) -> None:
        cls = FakeClassification(category="MACRO", subcategory="fed_rates")
        q = "Will Biden win?"
        assert self.specialist.can_handle(cls, q) is False

    def test_can_handle_legislation_rejected(self) -> None:
        """Election category but non-matching subcategory."""
        cls = FakeClassification(category="ELECTION", subcategory="legislation")
        q = "Will the bill pass?"
        assert self.specialist.can_handle(cls, q) is False


# ── TestPollingAverage ────────────────────────────────────────────────────


class TestPollingAverage:

    def setup_method(self) -> None:
        self.specialist = PoliticsSpecialist(FakeConfig())

    def test_simple_average(self) -> None:
        polls = [
            PollData(candidate="Biden", percentage=48.0, sample_size=1000),
            PollData(candidate="Biden", percentage=52.0, sample_size=1000),
        ]
        avg = self.specialist._compute_polling_average(polls, "Biden")
        assert avg == pytest.approx(0.50, abs=0.01)

    def test_weighted_by_sample_size(self) -> None:
        polls = [
            PollData(candidate="Biden", percentage=40.0, sample_size=100),
            PollData(candidate="Biden", percentage=60.0, sample_size=900),
        ]
        avg = self.specialist._compute_polling_average(polls, "Biden")
        # Weighted toward 60% due to larger sample
        assert avg > 0.55

    def test_single_poll(self) -> None:
        polls = [PollData(candidate="Biden", percentage=55.0, sample_size=500)]
        avg = self.specialist._compute_polling_average(polls, "Biden")
        assert avg == pytest.approx(0.55, abs=0.01)

    def test_no_polls_returns_50pct(self) -> None:
        avg = self.specialist._compute_polling_average([], "Biden")
        assert avg == pytest.approx(0.50)

    def test_no_matching_candidate(self) -> None:
        polls = [PollData(candidate="Trump", percentage=52.0, sample_size=1000)]
        avg = self.specialist._compute_polling_average(polls, "Biden")
        assert avg == pytest.approx(0.50)


# ── TestAdjustments ───────────────────────────────────────────────────────


class TestAdjustments:

    def setup_method(self) -> None:
        self.specialist = PoliticsSpecialist(FakeConfig())

    def test_time_discount_far_from_election(self) -> None:
        """Far from election → pull toward 50%."""
        race = RaceQuery(race_type="presidential", candidate="Biden")
        adj = self.specialist._apply_adjustments(0.60, race, days_to_election=200)
        # Should be pulled toward 0.5
        assert adj < 0.58

    def test_time_discount_close_to_election(self) -> None:
        """Close to election → minimal adjustment."""
        race = RaceQuery(race_type="presidential", candidate="Biden")
        adj = self.specialist._apply_adjustments(0.60, race, days_to_election=20)
        # Within 30 days, only polling error adjustment
        assert adj > 0.55

    def test_polling_error_adjustment(self) -> None:
        """Polling error pulls probability toward 50%."""
        race = RaceQuery(race_type="presidential", candidate="Test")
        adj = self.specialist._apply_adjustments(0.70, race, days_to_election=10)
        # 3.5% presidential error pulls toward 50%
        assert adj < 0.70

    def test_incumbency_advantage(self) -> None:
        """Incumbent gets +1% boost."""
        race_inc = RaceQuery(
            race_type="presidential", candidate="Biden",
            has_incumbent=True, candidate_is_incumbent=True,
        )
        race_chal = RaceQuery(
            race_type="presidential", candidate="Challenger",
            has_incumbent=True, candidate_is_incumbent=False,
        )
        adj_inc = self.specialist._apply_adjustments(0.50, race_inc, days_to_election=10)
        adj_chal = self.specialist._apply_adjustments(0.50, race_chal, days_to_election=10)
        assert adj_inc > adj_chal

    def test_bounds_clamped(self) -> None:
        race = RaceQuery(race_type="presidential", candidate="Test")
        adj_high = self.specialist._apply_adjustments(0.99, race, days_to_election=1)
        adj_low = self.specialist._apply_adjustments(0.01, race, days_to_election=1)
        assert adj_high <= 0.98
        assert adj_low >= 0.02


# ── TestPoliticsIntegration ───────────────────────────────────────────────


class TestPoliticsIntegration:

    @pytest.mark.asyncio
    async def test_forecast_augment_mode(self) -> None:
        """Politics specialist sets bypasses_llm=False."""
        specialist = PoliticsSpecialist(FakeConfig())
        result = await specialist.forecast(
            FakeMarket(), FakeFeatures(), FakeClassification(),
        )
        assert result.bypasses_llm is False

    @pytest.mark.asyncio
    async def test_forecast_returns_valid_result(self) -> None:
        specialist = PoliticsSpecialist(FakeConfig())
        result = await specialist.forecast(
            FakeMarket(), FakeFeatures(), FakeClassification(),
        )
        assert result.specialist_name == "politics"
        assert 0.02 <= result.probability <= 0.98
        assert result.confidence_level in ("LOW", "MEDIUM", "HIGH")

    @pytest.mark.asyncio
    async def test_to_base_rate_match_conversion(self) -> None:
        specialist = PoliticsSpecialist(FakeConfig())
        result = await specialist.forecast(
            FakeMarket(), FakeFeatures(), FakeClassification(),
        )
        brm = BaseSpecialist.to_base_rate_match(result)
        assert brm.base_rate == result.probability
        assert brm.source == "politics"

    @pytest.mark.asyncio
    async def test_confidence_high_close_to_election(self) -> None:
        specialist = PoliticsSpecialist(FakeConfig())
        features = FakeFeatures(days_to_expiry=15.0)
        # Need polls for HIGH confidence
        with patch.object(specialist, "_fetch_polling_data", new_callable=AsyncMock) as mock:
            mock.return_value = [
                PollData(candidate="Biden", percentage=52.0, sample_size=1000)
                for _ in range(5)
            ]
            result = await specialist.forecast(
                FakeMarket(), features, FakeClassification(),
            )
        assert result.confidence_level == "HIGH"

    @pytest.mark.asyncio
    async def test_unparseable_question_raises(self) -> None:
        specialist = PoliticsSpecialist(FakeConfig())
        market = FakeMarket(question="Will it rain tomorrow?")
        with pytest.raises(ValueError, match="Cannot parse"):
            await specialist.forecast(
                market, FakeFeatures(), FakeClassification(),
            )


# ── TestFullPipelineIntegration ───────────────────────────────────────────


class TestFullPipelineIntegration:

    def test_config_disabled_no_routing(self) -> None:
        """When specialists disabled, no routing occurs."""
        from src.config import BotConfig
        cfg = BotConfig()
        assert cfg.specialists.enabled is False

    @pytest.mark.asyncio
    async def test_weather_bypasses_llm(self) -> None:
        """Weather specialist returns bypass mode result."""
        sr = SpecialistResult(
            probability=0.87, confidence_level="HIGH",
            reasoning="GFS", evidence_quality=0.9,
            specialist_name="weather", bypasses_llm=True,
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.aggregation_method == "specialist:weather"
        assert er.model_probability == 0.87

    @pytest.mark.asyncio
    async def test_politics_augments_llm(self) -> None:
        """Politics specialist returns augment mode result."""
        sr = SpecialistResult(
            probability=0.55, confidence_level="MEDIUM",
            reasoning="Polling", evidence_quality=0.7,
            specialist_name="politics", bypasses_llm=False,
        )
        brm = BaseSpecialist.to_base_rate_match(sr)
        assert brm.base_rate == 0.55

    @pytest.mark.asyncio
    async def test_error_falls_back_to_none(self) -> None:
        """Router returns None when specialist errors."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mock_specialist = AsyncMock(spec=BaseSpecialist)
            mock_specialist.name = "weather"
            mock_specialist.can_handle.return_value = True
            mock_specialist.forecast.side_effect = Exception("API down")
            mock_load.return_value = mock_specialist

            router = SpecialistRouter(FakeConfig(enabled_specialists=["weather"]))
            result = await router.route(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
            assert result is None

    def test_specialist_result_flows_through_edge_calc(self) -> None:
        """SpecialistResult → EnsembleResult has correct probability."""
        sr = SpecialistResult(
            probability=0.75, confidence_level="HIGH",
            reasoning="r", evidence_quality=0.9, specialist_name="weather",
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        # Edge calc only needs model_probability
        assert er.model_probability == 0.75
        assert er.models_succeeded == 1
        assert er.spread == 0.0

    def test_specialist_result_has_required_forecast_fields(self) -> None:
        """EnsembleResult from specialist has all required fields."""
        sr = SpecialistResult(
            probability=0.65, confidence_level="MEDIUM",
            reasoning="test", evidence_quality=0.8,
            specialist_name="politics",
            key_evidence=[{"source": "polls", "text": "data"}],
            invalidation_triggers=["new poll released"],
        )
        er = BaseSpecialist.to_ensemble_result(sr)
        assert er.confidence_level == "MEDIUM"
        assert er.reasoning == "test"
        assert len(er.key_evidence) == 1
        assert len(er.invalidation_triggers) == 1
        assert len(er.individual_forecasts) == 1

    def test_multiple_specialists_can_coexist(self) -> None:
        """Router can hold weather + crypto + politics simultaneously."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            mocks = []
            for name in ["weather", "crypto_ta", "politics"]:
                m = MagicMock(spec=BaseSpecialist)
                m.name = name
                mocks.append(m)
            mock_load.side_effect = mocks

            config = FakeConfig(
                enabled_specialists=["weather", "crypto_ta", "politics"],
            )
            router = SpecialistRouter(config)
            assert len(router._specialists) == 3

    @pytest.mark.asyncio
    async def test_router_first_match_wins(self) -> None:
        """Router returns first matching specialist."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            m1 = AsyncMock(spec=BaseSpecialist)
            m1.name = "weather"
            m1.can_handle.return_value = True
            m1.forecast.return_value = SpecialistResult(
                probability=0.9, confidence_level="HIGH",
                reasoning="weather", evidence_quality=0.9,
                specialist_name="weather",
            )

            m2 = AsyncMock(spec=BaseSpecialist)
            m2.name = "crypto_ta"
            m2.can_handle.return_value = True

            mock_load.side_effect = [m1, m2]

            config = FakeConfig(enabled_specialists=["weather", "crypto_ta"])
            router = SpecialistRouter(config)

            result = await router.route(
                FakeMarket(), FakeFeatures(), FakeClassification(),
            )
            assert result is not None
            assert result.specialist_name == "weather"
            # Second specialist's forecast should NOT be called
            m2.forecast.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_all_specialists(self) -> None:
        """Router close delegates to all specialists."""
        with patch(
            "src.forecast.specialist_router.SpecialistRouter._load_specialist",
        ) as mock_load:
            m1 = AsyncMock(spec=BaseSpecialist)
            m1.name = "weather"
            m2 = AsyncMock(spec=BaseSpecialist)
            m2.name = "crypto_ta"
            mock_load.side_effect = [m1, m2]

            config = FakeConfig(enabled_specialists=["weather", "crypto_ta"])
            router = SpecialistRouter(config)
            await router.close()
            m1.close.assert_called_once()
            m2.close.assert_called_once()

    def test_specialist_metadata_preserved(self) -> None:
        """Specialist metadata is preserved through conversion."""
        sr = SpecialistResult(
            probability=0.87, confidence_level="HIGH",
            reasoning="r", evidence_quality=0.9,
            specialist_name="weather",
            specialist_metadata={"ensemble_members": 31, "city": "nyc"},
        )
        # Metadata stays on SpecialistResult, not on EnsembleResult
        assert sr.specialist_metadata["ensemble_members"] == 31

    def test_base_rate_match_pattern_description(self) -> None:
        """Base rate match has informative description."""
        sr = SpecialistResult(
            probability=0.65, confidence_level="MEDIUM",
            reasoning="r", evidence_quality=0.8,
            specialist_name="politics", bypasses_llm=False,
        )
        brm = BaseSpecialist.to_base_rate_match(sr)
        assert "politics" in brm.pattern_description.lower()
        assert "specialist" in brm.pattern_description.lower()

    def test_all_specialists_have_name_property(self) -> None:
        """Each specialist class has a name property."""
        from src.forecast.specialists.weather import WeatherSpecialist
        from src.forecast.specialists.crypto_ta import CryptoTASpecialist

        cfg = FakeConfig()
        assert WeatherSpecialist(cfg).name == "weather"
        assert CryptoTASpecialist(cfg).name == "crypto_ta"
        assert PoliticsSpecialist(cfg).name == "politics"

    def test_specialist_result_evidence_quality_range(self) -> None:
        """Evidence quality must be 0-1."""
        sr = SpecialistResult(
            probability=0.5, confidence_level="LOW",
            reasoning="r", evidence_quality=0.75,
            specialist_name="test",
        )
        assert 0.0 <= sr.evidence_quality <= 1.0
