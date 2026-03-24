"""Tests for LLM response cache and cached forecaster (Phase 1)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.backtest.cached_forecaster import CachedEnsembleForecaster, CacheMissError
from src.backtest.database import BacktestDatabase
from src.backtest.llm_cache import LLMResponseCache
from src.config import EnsembleConfig, ForecastingConfig
from src.forecast.ensemble import ModelForecast


@pytest.fixture
def db() -> BacktestDatabase:
    bdb = BacktestDatabase(db_path=":memory:")
    bdb.connect()
    yield bdb
    bdb.close()


@pytest.fixture
def cache(db: BacktestDatabase) -> LLMResponseCache:
    return LLMResponseCache(db, template_version="v1")


def _make_forecast(
    model: str = "gpt-4o",
    prob: float = 0.7,
    **kwargs,
) -> ModelForecast:
    defaults = dict(
        model_name=model,
        model_probability=prob,
        confidence_level="MEDIUM",
        reasoning="Test reasoning",
        latency_ms=500.0,
    )
    defaults.update(kwargs)
    return ModelForecast(**defaults)


# ── Cache Key Generation ─────────────────────────────────────────────


class TestCacheKey:

    def test_deterministic(self, cache: LLMResponseCache) -> None:
        k1 = cache.make_cache_key("Will X?", "gpt-4o")
        k2 = cache.make_cache_key("Will X?", "gpt-4o")
        assert k1 == k2

    def test_different_question(self, cache: LLMResponseCache) -> None:
        k1 = cache.make_cache_key("Will X?", "gpt-4o")
        k2 = cache.make_cache_key("Will Y?", "gpt-4o")
        assert k1 != k2

    def test_different_model(self, cache: LLMResponseCache) -> None:
        k1 = cache.make_cache_key("Will X?", "gpt-4o")
        k2 = cache.make_cache_key("Will X?", "claude-3-5-sonnet")
        assert k1 != k2

    def test_different_version(self, db: BacktestDatabase) -> None:
        c1 = LLMResponseCache(db, template_version="v1")
        c2 = LLMResponseCache(db, template_version="v2")
        k1 = c1.make_cache_key("Will X?", "gpt-4o")
        k2 = c2.make_cache_key("Will X?", "gpt-4o")
        assert k1 != k2


# ── Cache Put/Get ─────────────────────────────────────────────────────


class TestCachePutGet:

    def test_put_and_get(self, cache: LLMResponseCache) -> None:
        forecast = _make_forecast(prob=0.75)
        cache.put("Will X?", "gpt-4o", forecast)
        result = cache.get("Will X?", "gpt-4o")
        assert result is not None
        assert result.model_probability == 0.75
        assert result.confidence_level == "MEDIUM"
        assert result.model_name == "gpt-4o"

    def test_cache_miss(self, cache: LLMResponseCache) -> None:
        result = cache.get("nonexistent question", "gpt-4o")
        assert result is None

    def test_roundtrip_preserves_fields(self, cache: LLMResponseCache) -> None:
        forecast = _make_forecast(
            prob=0.82,
            reasoning="Strong evidence",
            invalidation_triggers=["event A", "event B"],
            key_evidence=[{"text": "fact", "source": "test"}],
        )
        cache.put("Q?", "gpt-4o", forecast, prompt="the prompt")
        result = cache.get("Q?", "gpt-4o")
        assert result is not None
        assert result.reasoning == "Strong evidence"
        assert len(result.invalidation_triggers) == 2
        assert len(result.key_evidence) == 1

    def test_overwrite_existing(self, cache: LLMResponseCache) -> None:
        cache.put("Q?", "gpt-4o", _make_forecast(prob=0.5))
        cache.put("Q?", "gpt-4o", _make_forecast(prob=0.9))
        result = cache.get("Q?", "gpt-4o")
        assert result is not None
        assert result.model_probability == 0.9


# ── Cache Stats ───────────────────────────────────────────────────────


class TestCacheStats:

    def test_initial_stats(self, cache: LLMResponseCache) -> None:
        s = cache.stats
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["hit_rate"] == 0.0

    def test_stats_after_miss(self, cache: LLMResponseCache) -> None:
        cache.get("X?", "gpt-4o")
        assert cache.stats["misses"] == 1

    def test_stats_after_hit(self, cache: LLMResponseCache) -> None:
        cache.put("X?", "gpt-4o", _make_forecast())
        cache.get("X?", "gpt-4o")
        assert cache.stats["hits"] == 1
        assert cache.stats["hit_rate"] == 1.0

    def test_stats_mixed(self, cache: LLMResponseCache) -> None:
        cache.put("A?", "gpt-4o", _make_forecast())
        cache.get("A?", "gpt-4o")  # hit
        cache.get("B?", "gpt-4o")  # miss
        cache.get("C?", "gpt-4o")  # miss
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 2
        assert cache.stats["hit_rate"] == pytest.approx(1 / 3, abs=0.01)


# ── Cached Ensemble Forecaster ────────────────────────────────────────


class TestCachedEnsembleForecaster:

    def _make_forecaster(
        self,
        cache: LLMResponseCache,
        models: list[str] | None = None,
        force_cache_only: bool = False,
    ) -> CachedEnsembleForecaster:
        if models is None:
            models = ["gpt-4o"]
        ensemble_cfg = EnsembleConfig(
            models=models,
            min_models_required=1,
            aggregation="trimmed_mean",
        )
        forecast_cfg = ForecastingConfig()
        return CachedEnsembleForecaster(
            cache=cache,
            ensemble_config=ensemble_cfg,
            forecast_config=forecast_cfg,
            force_cache_only=force_cache_only,
        )

    def _make_features(self, question: str = "Will X?"):
        from src.forecast.feature_builder import MarketFeatures
        return MarketFeatures(
            question=question,
            market_type="UNKNOWN",
            volume_usd=5000.0,
            liquidity_usd=1000.0,
        )

    def _make_evidence(self):
        from src.research.evidence_extractor import EvidencePackage
        return EvidencePackage(
            market_id="test",
            question="Will X?",
            summary="Test evidence",
            quality_score=0.5,
        )

    def test_cache_hit_skips_api(self, cache: LLMResponseCache) -> None:
        """When the cache has the result, no API call is made."""
        cache.put("Will X?", "gpt-4o", _make_forecast(prob=0.8))
        forecaster = self._make_forecaster(cache, models=["gpt-4o"])

        # No mocking needed — if it calls the API, it would fail
        # because there's no OpenAI key configured
        result = asyncio.new_event_loop().run_until_complete(
            forecaster.forecast(self._make_features("Will X?"), self._make_evidence())
        )
        assert result.model_probability == pytest.approx(0.8, abs=0.01)
        assert result.models_succeeded == 1

    def test_force_cache_only_raises_on_miss(
        self, cache: LLMResponseCache,
    ) -> None:
        """force_cache_only=True raises CacheMissError on cache miss."""
        forecaster = self._make_forecaster(
            cache, models=["gpt-4o"], force_cache_only=True,
        )
        with pytest.raises(CacheMissError, match="gpt-4o"):
            asyncio.new_event_loop().run_until_complete(
                forecaster.forecast(
                    self._make_features("Unknown question?"),
                    self._make_evidence(),
                )
            )

    def test_partial_cache_hit(self, cache: LLMResponseCache) -> None:
        """Some models cached, others would need API calls.

        In force_cache_only=False, the uncached model call goes through
        the normal flow. We mock the API call for the uncached model.
        """
        cache.put("Will X?", "model-a", _make_forecast(model="model-a", prob=0.7))

        forecaster = self._make_forecaster(
            cache, models=["model-a", "model-b"],
        )

        mock_forecast = _make_forecast(model="model-b", prob=0.6)

        with patch(
            "src.backtest.cached_forecaster._query_model",
            new_callable=AsyncMock,
            return_value=mock_forecast,
        ):
            result = asyncio.new_event_loop().run_until_complete(
                forecaster.forecast(
                    self._make_features("Will X?"),
                    self._make_evidence(),
                )
            )

        assert result.models_succeeded == 2
        # Check that model-b was cached for future use
        cached_b = cache.get("Will X?", "model-b")
        assert cached_b is not None
        assert cached_b.model_probability == 0.6

    def test_all_cached_multiple_models(
        self, cache: LLMResponseCache,
    ) -> None:
        """All models cached — no API calls, proper aggregation."""
        cache.put("Will X?", "m1", _make_forecast(model="m1", prob=0.6))
        cache.put("Will X?", "m2", _make_forecast(model="m2", prob=0.8))

        forecaster = self._make_forecaster(cache, models=["m1", "m2"])

        result = asyncio.new_event_loop().run_until_complete(
            forecaster.forecast(self._make_features("Will X?"), self._make_evidence())
        )
        assert result.models_succeeded == 2
        # Aggregated probability should be between 0.6 and 0.8
        assert 0.55 <= result.model_probability <= 0.85
