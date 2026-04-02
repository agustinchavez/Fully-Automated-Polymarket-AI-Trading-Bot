"""Tests for ensemble integration, adaptive weights fix, prompt v2 + decomposition wiring
(Phase 2 — Batch C)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import EnsembleConfig, ForecastingConfig
from src.forecast.base_rates import BaseRateMatch
from src.forecast.ensemble import (
    EnsembleForecaster,
    EnsembleResult,
    ModelForecast,
    _build_prompt,
)
from src.forecast.feature_builder import MarketFeatures
from src.research.evidence_extractor import EvidencePackage


def _make_features(**kwargs) -> MarketFeatures:
    defaults = dict(
        question="Will the Fed cut rates?",
        market_type="MACRO",
        volume_usd=50000.0,
        liquidity_usd=10000.0,
        spread_pct=0.02,
        days_to_expiry=30.0,
        price_momentum=0.01,
        evidence_quality=0.7,
        num_sources=5,
        top_bullets=["CPI fell to 2.1%"],
    )
    defaults.update(kwargs)
    return MarketFeatures(**defaults)


def _make_evidence(**kwargs) -> EvidencePackage:
    defaults = dict(
        market_id="test-market",
        question="Will the Fed cut rates?",
        summary="Summary.",
    )
    defaults.update(kwargs)
    return EvidencePackage(**defaults)


def _make_model_forecast(model: str, prob: float, **kwargs) -> ModelForecast:
    defaults = dict(
        model_name=model,
        model_probability=prob,
        confidence_level="MEDIUM",
        reasoning="Test reasoning.",
    )
    defaults.update(kwargs)
    return ModelForecast(**defaults)


def _make_ensemble_config(**kwargs) -> EnsembleConfig:
    defaults = dict(
        enabled=True,
        models=["gpt-4o", "claude-sonnet-4-6", "gemini-2.0-flash"],
        aggregation="trimmed_mean",
        weights={"gpt-4o": 0.40, "claude-sonnet-4-6": 0.35, "gemini-2.0-flash": 0.25},
        min_models_required=1,
    )
    defaults.update(kwargs)
    return EnsembleConfig(**defaults)


def _make_forecast_config(**kwargs) -> ForecastingConfig:
    defaults = dict(prompt_version="v1")
    defaults.update(kwargs)
    return ForecastingConfig(**defaults)


# ── Adaptive Weights Fix Tests ───────────────────────────────────────


class TestAdaptiveWeightsAutoSwitch:

    def test_trimmed_mean_with_adaptive_weights_switches_to_weighted(self) -> None:
        """When adaptive weights are set and aggregation is trimmed_mean,
        auto-switches to weighted."""
        ens_cfg = _make_ensemble_config(aggregation="trimmed_mean")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        weights = {"gpt-4o": 0.5, "claude-sonnet-4-6": 0.3, "gemini-2.0-flash": 0.2}
        forecaster.set_adaptive_weights(weights)

        model_probs = [("gpt-4o", 0.8), ("claude-sonnet-4-6", 0.6), ("gemini-2.0-flash", 0.4)]
        prob, method = forecaster._aggregate(model_probs)

        assert method == "weighted"
        # Weighted: 0.8*0.5 + 0.6*0.3 + 0.4*0.2 = 0.4 + 0.18 + 0.08 = 0.66
        assert prob == pytest.approx(0.66, abs=0.001)

    def test_weighted_with_adaptive_weights_stays_weighted(self) -> None:
        """When aggregation is already weighted, stays weighted."""
        ens_cfg = _make_ensemble_config(aggregation="weighted")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        weights = {"gpt-4o": 0.5, "claude-sonnet-4-6": 0.3, "gemini-2.0-flash": 0.2}
        forecaster.set_adaptive_weights(weights)

        model_probs = [("gpt-4o", 0.8), ("claude-sonnet-4-6", 0.6), ("gemini-2.0-flash", 0.4)]
        prob, method = forecaster._aggregate(model_probs)

        assert method == "weighted"

    def test_median_with_adaptive_weights_stays_median(self) -> None:
        """When aggregation is median, stays median even with adaptive weights."""
        ens_cfg = _make_ensemble_config(aggregation="median")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        weights = {"gpt-4o": 0.5, "claude-sonnet-4-6": 0.3, "gemini-2.0-flash": 0.2}
        forecaster.set_adaptive_weights(weights)

        model_probs = [("gpt-4o", 0.8), ("claude-sonnet-4-6", 0.6), ("gemini-2.0-flash", 0.4)]
        prob, method = forecaster._aggregate(model_probs)

        assert method == "median"
        assert prob == pytest.approx(0.6, abs=0.001)  # median of [0.4, 0.6, 0.8]

    def test_no_adaptive_weights_uses_configured_method(self) -> None:
        """Without adaptive weights, uses configured aggregation method."""
        ens_cfg = _make_ensemble_config(aggregation="trimmed_mean")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        model_probs = [("gpt-4o", 0.8), ("claude-sonnet-4-6", 0.6), ("gemini-2.0-flash", 0.4)]
        prob, method = forecaster._aggregate(model_probs)

        assert method == "trimmed_mean"

    def test_weighted_aggregation_correct_calculation(self) -> None:
        """Weighted aggregation produces correct weighted average."""
        ens_cfg = _make_ensemble_config(
            aggregation="weighted",
            weights={"gpt-4o": 0.6, "claude-sonnet-4-6": 0.4},
            models=["gpt-4o", "claude-sonnet-4-6"],
        )
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        model_probs = [("gpt-4o", 0.9), ("claude-sonnet-4-6", 0.3)]
        prob, method = forecaster._aggregate(model_probs)

        # (0.9*0.6 + 0.3*0.4) / (0.6+0.4) = (0.54 + 0.12) / 1.0 = 0.66
        assert prob == pytest.approx(0.66, abs=0.001)
        assert method == "weighted"

    def test_unknown_model_gets_equal_weight(self) -> None:
        """Model not in weights dict gets equal weight."""
        ens_cfg = _make_ensemble_config(aggregation="weighted")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        model_probs = [("gpt-4o", 0.8), ("unknown-model", 0.4)]
        prob, method = forecaster._aggregate(model_probs)

        assert method == "weighted"
        assert 0.4 < prob < 0.8  # somewhere between the two

    def test_single_model_returns_its_prob(self) -> None:
        """Single model returns its probability regardless of method."""
        ens_cfg = _make_ensemble_config(aggregation="trimmed_mean")
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)
        forecaster.set_adaptive_weights({"gpt-4o": 1.0})

        model_probs = [("gpt-4o", 0.75)]
        prob, method = forecaster._aggregate(model_probs)

        assert prob == 0.75

    def test_empty_model_probs(self) -> None:
        """Empty model probs returns 0.5."""
        ens_cfg = _make_ensemble_config()
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        prob, method = forecaster._aggregate([])
        assert prob == 0.5

    @pytest.mark.asyncio
    async def test_ensemble_result_reflects_actual_method(self) -> None:
        """EnsembleResult.aggregation_method reflects auto-switched method."""
        ens_cfg = _make_ensemble_config(
            aggregation="trimmed_mean",
            models=["model-a", "model-b"],
        )
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)
        forecaster.set_adaptive_weights({"model-a": 0.6, "model-b": 0.4})

        # Mock _query_model to return successful forecasts
        with patch("src.forecast.ensemble._query_model", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = [
                _make_model_forecast("model-a", 0.7),
                _make_model_forecast("model-b", 0.5),
            ]
            result = await forecaster.forecast(
                _make_features(), _make_evidence(),
            )

        assert result.aggregation_method == "weighted"

    @pytest.mark.asyncio
    async def test_ensemble_without_adaptive_weights_reports_configured_method(self) -> None:
        """Without adaptive weights, reports configured aggregation method."""
        ens_cfg = _make_ensemble_config(
            aggregation="trimmed_mean",
            models=["model-a", "model-b"],
        )
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        with patch("src.forecast.ensemble._query_model", new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = [
                _make_model_forecast("model-a", 0.7),
                _make_model_forecast("model-b", 0.5),
            ]
            result = await forecaster.forecast(
                _make_features(), _make_evidence(),
            )

        assert result.aggregation_method == "trimmed_mean"


# ── Prompt V2 Integration Tests ──────────────────────────────────────


class TestPromptV2Integration:

    @pytest.mark.asyncio
    async def test_forecast_with_v1_prompt(self) -> None:
        """Forecast with prompt_version=v1 uses legacy prompt."""
        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config(prompt_version="v1")
        forecaster = EnsembleForecaster(ens_cfg, fc)

        prompts_used = []

        async def capture_prompt(model, prompt, config, timeout_secs=30):
            prompts_used.append(prompt)
            return _make_model_forecast(model, 0.6)

        with patch("src.forecast.ensemble._query_model", side_effect=capture_prompt):
            await forecaster.forecast(
                _make_features(), _make_evidence(),
                prompt_version="v1",
            )

        assert len(prompts_used) == 1
        assert "superforecasting" not in prompts_used[0]
        assert "TASK:" in prompts_used[0]

    @pytest.mark.asyncio
    async def test_forecast_with_v2_prompt(self) -> None:
        """Forecast with prompt_version=v2 uses structured prompt."""
        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config(prompt_version="v2")
        forecaster = EnsembleForecaster(ens_cfg, fc)

        prompts_used = []

        async def capture_prompt(model, prompt, config, timeout_secs=30):
            prompts_used.append(prompt)
            return _make_model_forecast(model, 0.6)

        with patch("src.forecast.ensemble._query_model", side_effect=capture_prompt):
            await forecaster.forecast(
                _make_features(), _make_evidence(),
                prompt_version="v2",
            )

        assert len(prompts_used) == 1
        assert "superforecasting" in prompts_used[0]
        assert "START WITH THE BASE RATE" in prompts_used[0]

    @pytest.mark.asyncio
    async def test_forecast_with_base_rate_info(self) -> None:
        """Base rate info is included in v2 prompt."""
        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config(prompt_version="v2")
        forecaster = EnsembleForecaster(ens_cfg, fc)

        base_rate = BaseRateMatch(
            base_rate=0.25,
            pattern_description="Fed rate cut frequency",
            source="FOMC data",
            confidence=0.7,
        )

        prompts_used = []

        async def capture_prompt(model, prompt, config, timeout_secs=30):
            prompts_used.append(prompt)
            return _make_model_forecast(model, 0.4)

        with patch("src.forecast.ensemble._query_model", side_effect=capture_prompt):
            await forecaster.forecast(
                _make_features(), _make_evidence(),
                base_rate_info=base_rate, prompt_version="v2",
            )

        assert "Base rate: 25%" in prompts_used[0]
        assert "Fed rate cut frequency" in prompts_used[0]

    @pytest.mark.asyncio
    async def test_forecast_v2_without_base_rate(self) -> None:
        """v2 prompt works without base rate info."""
        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config(prompt_version="v2")
        forecaster = EnsembleForecaster(ens_cfg, fc)

        prompts_used = []

        async def capture_prompt(model, prompt, config, timeout_secs=30):
            prompts_used.append(prompt)
            return _make_model_forecast(model, 0.5)

        with patch("src.forecast.ensemble._query_model", side_effect=capture_prompt):
            await forecaster.forecast(
                _make_features(), _make_evidence(),
                base_rate_info=None, prompt_version="v2",
            )

        assert "No specific base rate available" in prompts_used[0]

    @pytest.mark.asyncio
    async def test_v2_response_populates_new_fields(self) -> None:
        """v2 response fields (base_rate, evidence_for/against) are populated."""
        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config()
        forecaster = EnsembleForecaster(ens_cfg, fc)

        v2_forecast = _make_model_forecast(
            "model-a", 0.4,
            base_rate=0.25,
            evidence_for=["CPI dropped"],
            evidence_against=["Labor strong"],
        )

        with patch("src.forecast.ensemble._query_model", new_callable=AsyncMock) as mock:
            mock.return_value = v2_forecast
            result = await forecaster.forecast(
                _make_features(), _make_evidence(),
            )

        assert result.individual_forecasts[0].base_rate == 0.25
        assert result.individual_forecasts[0].evidence_for == ["CPI dropped"]


# ── Decomposition Integration Tests ─────────────────────────────────


class TestDecompositionIntegration:

    def test_should_decompose_called_correctly(self) -> None:
        """should_decompose filters by question length and category."""
        from src.forecast.decomposer import should_decompose

        assert should_decompose("Will the Federal Reserve cut interest rates?")
        assert not should_decompose("Yes?")
        assert not should_decompose("Will team win?", market_type="SPORTS")

    @pytest.mark.asyncio
    async def test_decompose_with_mocked_llm(self) -> None:
        """Full decomposition pipeline with mocked LLM."""
        from src.forecast.decomposer import (
            QuestionDecomposer,
            combine_sub_forecasts,
        )

        config = ForecastingConfig(
            decomposition_enabled=True,
            max_sub_questions=3,
        )
        decomposer = QuestionDecomposer(config)

        mock_response = {
            "sub_questions": [
                {"text": "Will inflation drop?", "weight": 0.6},
                {"text": "Will unemployment rise?", "weight": 0.4},
            ],
            "decomposition_quality": 0.7,
        }

        with patch.object(decomposer, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            decomp = await decomposer.decompose("Will the Fed cut rates?", "MACRO")

        assert len(decomp.sub_questions) == 2

        # Combine with sub-probabilities
        combined = combine_sub_forecasts(decomp, [0.7, 0.3])
        assert 0.3 < combined < 0.7

    @pytest.mark.asyncio
    async def test_decomposition_failure_is_non_fatal(self) -> None:
        """Decomposition failure doesn't crash the pipeline."""
        from src.forecast.decomposer import QuestionDecomposer

        config = ForecastingConfig(decomposition_enabled=True)
        decomposer = QuestionDecomposer(config)

        with patch.object(decomposer, "_call_llm", new_callable=AsyncMock) as mock:
            mock.side_effect = RuntimeError("API error")
            decomp = await decomposer.decompose("Will the Fed cut rates?")

        assert decomp.sub_questions == []
        assert "API error" in decomp.error

    def test_combine_with_decomposition_disabled(self) -> None:
        """Config disabled by default — no decomposition."""
        config = ForecastingConfig()
        assert config.decomposition_enabled is False


# ── Cached Forecaster Integration ────────────────────────────────────


class TestCachedForecasterIntegration:

    @pytest.mark.asyncio
    async def test_cached_forecaster_accepts_new_params(self) -> None:
        """CachedEnsembleForecaster.forecast() accepts base_rate_info and prompt_version."""
        from src.backtest.cached_forecaster import CachedEnsembleForecaster
        from src.backtest.llm_cache import LLMResponseCache
        from src.backtest.database import BacktestDatabase

        db = BacktestDatabase(":memory:")
        db.connect()
        cache = LLMResponseCache(db)

        ens_cfg = _make_ensemble_config(models=["model-a"])
        fc = _make_forecast_config(prompt_version="v2")
        cached = CachedEnsembleForecaster(cache, ens_cfg, fc)

        base_rate = BaseRateMatch(
            base_rate=0.25,
            pattern_description="Fed cut rate",
            source="FOMC data",
            confidence=0.7,
        )

        with patch("src.backtest.cached_forecaster._query_model", new_callable=AsyncMock) as mock:
            mock.return_value = _make_model_forecast("model-a", 0.6)
            result = await cached.forecast(
                _make_features(), _make_evidence(),
                base_rate_info=base_rate, prompt_version="v2",
            )

        assert result.model_probability == pytest.approx(0.6, abs=0.001)
        db.close()

    @pytest.mark.asyncio
    async def test_cached_forecaster_aggregate_returns_tuple(self) -> None:
        """CachedEnsembleForecaster handles _aggregate returning tuple."""
        from src.backtest.cached_forecaster import CachedEnsembleForecaster
        from src.backtest.llm_cache import LLMResponseCache
        from src.backtest.database import BacktestDatabase

        db = BacktestDatabase(":memory:")
        db.connect()
        cache = LLMResponseCache(db)

        ens_cfg = _make_ensemble_config(models=["model-a", "model-b"])
        fc = _make_forecast_config()
        cached = CachedEnsembleForecaster(cache, ens_cfg, fc)

        with patch("src.backtest.cached_forecaster._query_model", new_callable=AsyncMock) as mock:
            mock.side_effect = [
                _make_model_forecast("model-a", 0.7),
                _make_model_forecast("model-b", 0.5),
            ]
            result = await cached.forecast(
                _make_features(), _make_evidence(),
            )

        assert 0.5 <= result.model_probability <= 0.7
        assert result.aggregation_method in ("trimmed_mean", "weighted", "median")
        db.close()


# ── Config Validation Tests ──────────────────────────────────────────


class TestConfigValidation:

    def test_prompt_version_default(self) -> None:
        """prompt_version defaults to v1."""
        fc = ForecastingConfig()
        assert fc.prompt_version == "v1"

    def test_base_rate_disabled_by_default(self) -> None:
        """base_rate_enabled defaults to False."""
        fc = ForecastingConfig()
        assert fc.base_rate_enabled is False

    def test_decomposition_disabled_by_default(self) -> None:
        """decomposition_enabled defaults to False."""
        fc = ForecastingConfig()
        assert fc.decomposition_enabled is False

    def test_all_features_can_be_enabled(self) -> None:
        """All Phase 2 features can be enabled via config."""
        fc = ForecastingConfig(
            prompt_version="v2",
            base_rate_enabled=True,
            decomposition_enabled=True,
            max_sub_questions=5,
        )
        assert fc.prompt_version == "v2"
        assert fc.base_rate_enabled is True
        assert fc.decomposition_enabled is True
        assert fc.max_sub_questions == 5
