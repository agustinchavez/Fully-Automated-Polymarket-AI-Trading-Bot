"""Cache-wrapped ensemble forecaster for backtesting.

Wraps the existing EnsembleForecaster with a cache layer that:
  - Checks the LLM cache before making API calls
  - On cache miss, calls the actual LLM and stores the result
  - Supports partial cache hits (some models cached, others live)
  - Has a force_cache_only mode that fails on cache miss
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.backtest.llm_cache import LLMResponseCache
from src.config import EnsembleConfig, ForecastingConfig
from src.forecast.ensemble import (
    EnsembleForecaster,
    EnsembleResult,
    ModelForecast,
    _build_prompt,
    _query_model,
)
from src.forecast.feature_builder import MarketFeatures
from src.research.evidence_extractor import EvidencePackage
from src.observability.logger import get_logger

log = get_logger(__name__)


class CacheMissError(Exception):
    """Raised in force_cache_only mode when a cache miss occurs."""

    def __init__(self, model_name: str, question: str):
        self.model_name = model_name
        self.question_preview = question[:80]
        super().__init__(
            f"Cache miss for model '{model_name}' on question: {self.question_preview}"
        )


class CachedEnsembleForecaster:
    """Ensemble forecaster with transparent LLM cache layer.

    For each model in the ensemble:
      1. Check cache for (question, model_name)
      2. If hit → return cached ModelForecast instantly
      3. If miss → call actual LLM, cache the result, return

    Partial cache hits are supported: if 2 of 3 models are cached,
    only 1 API call is made.
    """

    def __init__(
        self,
        cache: LLMResponseCache,
        ensemble_config: EnsembleConfig,
        forecast_config: ForecastingConfig,
        force_cache_only: bool = False,
    ):
        self._cache = cache
        self._ensemble_config = ensemble_config
        self._forecast_config = forecast_config
        self._force_cache_only = force_cache_only
        # Reuse the existing EnsembleForecaster for aggregation
        self._ensemble = EnsembleForecaster(ensemble_config, forecast_config)

    async def forecast(
        self,
        features: MarketFeatures,
        evidence: EvidencePackage,
        base_rate_info: Any = None,
        prompt_version: str = "v1",
    ) -> EnsembleResult:
        """Run ensemble forecast with cache layer."""
        question = features.question
        prompt = _build_prompt(features, evidence, base_rate_info, prompt_version)
        timeout = self._ensemble_config.timeout_per_model_secs

        forecasts: list[ModelForecast] = []

        for model in self._ensemble_config.models:
            # Check cache first
            cached = self._cache.get(question, model)
            if cached is not None:
                log.info(
                    "cached_forecaster.cache_hit",
                    model=model,
                    question=question[:60],
                )
                forecasts.append(cached)
                continue

            # Cache miss
            if self._force_cache_only:
                raise CacheMissError(model, question)

            # Call actual LLM
            log.info(
                "cached_forecaster.cache_miss",
                model=model,
                question=question[:60],
            )
            result = await _query_model(model, prompt, self._forecast_config, timeout)

            # Cache if successful
            if not result.error:
                self._cache.put(question, model, result, prompt)

            forecasts.append(result)

        # Use the existing aggregation logic
        successes = [f for f in forecasts if not f.error]
        failures = [f for f in forecasts if f.error]

        if not successes:
            return EnsembleResult(
                model_probability=0.5,
                confidence_level="LOW",
                models_succeeded=0,
                models_failed=len(failures),
                reasoning="All models failed",
            )

        # Aggregate probabilities using the existing method
        model_probs = [(f.model_name, f.model_probability) for f in successes]
        agg_prob, agg_method = self._ensemble._aggregate(model_probs)

        # Aggregate confidence
        conf_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        conf_values = [conf_order.get(f.confidence_level, 0) for f in successes]
        avg_conf = sum(conf_values) / len(conf_values)
        if avg_conf >= 1.5:
            agg_confidence = "HIGH"
        elif avg_conf >= 0.5:
            agg_confidence = "MEDIUM"
        else:
            agg_confidence = "LOW"

        prob_values = [f.model_probability for f in successes]
        spread = max(prob_values) - min(prob_values) if len(prob_values) > 1 else 0.0
        agreement = max(0.0, 1.0 - spread * 2)

        if spread > 0.15:
            agg_confidence = "LOW"

        all_reasoning = [f.reasoning for f in successes if f.reasoning]

        return EnsembleResult(
            model_probability=agg_prob,
            confidence_level=agg_confidence,
            individual_forecasts=list(forecasts),
            models_succeeded=len(successes),
            models_failed=len(failures),
            aggregation_method=agg_method,
            spread=round(spread, 4),
            agreement_score=round(agreement, 3),
            reasoning=" | ".join(all_reasoning[:3]),
        )
