"""Tests for Phase 3: Silicon Crowd Expansion — Grok + DeepSeek + median aggregation.

Covers:
  - Provider routing for new models
  - No-key graceful degradation (Grok and DeepSeek)
  - Circuit breaker integration
  - Rate limiter buckets
  - DeepSeek category gating (GEOPOLITICS/ELECTION exclusion)
  - Median aggregation
  - Config validation
  - Mocked API calls (base_url, cost tracking)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import EnsembleConfig, ModelTierConfig
from src.connectors.rate_limiter import DEFAULT_LIMITS
from src.forecast.ensemble import (
    EnsembleForecaster,
    ModelForecast,
    _query_deepseek,
    _query_model,
    _query_xai,
    _route_model,
)
from src.forecast.feature_builder import MarketFeatures
from src.observability.circuit_breaker import DEFAULT_BREAKER_CONFIGS
from src.research.evidence_extractor import EvidencePackage


# ── Helpers ──────────────────────────────────────────────────────────


def _make_features(**overrides) -> MarketFeatures:
    defaults = dict(
        market_id="test",
        question="Will X happen?",
        market_type="MACRO",
        implied_probability=0.50,
        volume_usd=5000.0,
        category="MACRO",
    )
    defaults.update(overrides)
    return MarketFeatures(**defaults)


def _make_evidence() -> EvidencePackage:
    return EvidencePackage(
        market_id="test",
        question="Will X happen?",
        summary="Test evidence.",
        contradictions=[],
    )


def _make_forecast_config():
    from src.config import ForecastingConfig
    return ForecastingConfig()


# ── Provider routing ──────────────────────────────────────────────────


class TestProviderRouting:
    def test_route_grok(self) -> None:
        assert _route_model("grok-4-fast-reasoning") == "xai"

    def test_route_grok_case_insensitive(self) -> None:
        assert _route_model("Grok-4") == "xai"

    def test_route_deepseek(self) -> None:
        assert _route_model("deepseek-chat") == "deepseek"

    def test_route_deepseek_case_insensitive(self) -> None:
        assert _route_model("DeepSeek-V3") == "deepseek"

    def test_existing_routes_unchanged(self) -> None:
        assert _route_model("gpt-4o") == "openai"
        assert _route_model("claude-sonnet-4-6") == "anthropic"
        assert _route_model("gemini-2.5-flash") == "google"


# ── No-key behavior — Grok ───────────────────────────────────────────


class TestNoKeyGrok:
    def _unset_xai_key(self):
        """Helper to ensure XAI_API_KEY is absent for no-key tests."""
        import os
        saved = os.environ.pop("XAI_API_KEY", None)
        return saved

    def _restore_xai_key(self, saved):
        import os
        if saved is not None:
            os.environ["XAI_API_KEY"] = saved

    def test_no_key_returns_error(self) -> None:
        saved = self._unset_xai_key()
        try:
            result = asyncio.run(
                _query_xai("grok-4-fast-reasoning", "test prompt", _make_forecast_config())
            )
            assert result.error == "XAI_API_KEY not set"
            assert result.model_probability == 0.5
        finally:
            self._restore_xai_key(saved)

    def test_no_key_no_network_call(self) -> None:
        saved = self._unset_xai_key()
        try:
            with patch("openai.AsyncOpenAI") as mock_client:
                result = asyncio.run(
                    _query_xai("grok-4-fast-reasoning", "test", _make_forecast_config())
                )
                mock_client.assert_not_called()
        finally:
            self._restore_xai_key(saved)

    def test_no_key_model_name_preserved(self) -> None:
        saved = self._unset_xai_key()
        try:
            result = asyncio.run(
                _query_xai("grok-4-fast-reasoning", "test", _make_forecast_config())
            )
            assert result.model_name == "grok-4-fast-reasoning"
        finally:
            self._restore_xai_key(saved)


# ── No-key behavior — DeepSeek ────────────────────────────────────────


class TestNoKeyDeepSeek:
    def _unset_deepseek_key(self):
        import os
        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        return saved

    def _restore_deepseek_key(self, saved):
        import os
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved

    def test_no_key_returns_error(self) -> None:
        saved = self._unset_deepseek_key()
        try:
            result = asyncio.run(
                _query_deepseek("deepseek-chat", "test prompt", _make_forecast_config())
            )
            assert result.error == "DEEPSEEK_API_KEY not set"
            assert result.model_probability == 0.5
        finally:
            self._restore_deepseek_key(saved)

    def test_no_key_no_network_call(self) -> None:
        saved = self._unset_deepseek_key()
        try:
            with patch("openai.AsyncOpenAI") as mock_client:
                result = asyncio.run(
                    _query_deepseek("deepseek-chat", "test", _make_forecast_config())
                )
                mock_client.assert_not_called()
        finally:
            self._restore_deepseek_key(saved)

    def test_no_key_model_name_preserved(self) -> None:
        saved = self._unset_deepseek_key()
        try:
            result = asyncio.run(
                _query_deepseek("deepseek-chat", "test", _make_forecast_config())
            )
            assert result.model_name == "deepseek-chat"
        finally:
            self._restore_deepseek_key(saved)


# ── Circuit breaker ───────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_xai_circuit_breaker_exists(self) -> None:
        assert "xai" in DEFAULT_BREAKER_CONFIGS
        assert DEFAULT_BREAKER_CONFIGS["xai"].name == "xAI Grok"

    def test_deepseek_circuit_breaker_exists(self) -> None:
        assert "deepseek" in DEFAULT_BREAKER_CONFIGS
        assert DEFAULT_BREAKER_CONFIGS["deepseek"].name == "DeepSeek"

    def test_xai_cb_open_returns_error(self) -> None:
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("xai")
            # Force open
            original_allow = cb.allow_request
            cb.allow_request = lambda: False
            cb.time_until_retry = lambda: 10.0
            try:
                result = asyncio.run(
                    _query_xai("grok-4-fast-reasoning", "test", _make_forecast_config())
                )
                assert "Circuit breaker open" in result.error
            finally:
                cb.allow_request = original_allow

    def test_deepseek_cb_open_returns_error(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("deepseek")
            original_allow = cb.allow_request
            cb.allow_request = lambda: False
            cb.time_until_retry = lambda: 10.0
            try:
                result = asyncio.run(
                    _query_deepseek("deepseek-chat", "test", _make_forecast_config())
                )
                assert "Circuit breaker open" in result.error
            finally:
                cb.allow_request = original_allow


# ── Rate limiter ──────────────────────────────────────────────────────


class TestRateLimiter:
    def test_xai_bucket_exists(self) -> None:
        assert "xai" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["xai"].tokens_per_second == 2.0

    def test_deepseek_bucket_exists(self) -> None:
        assert "deepseek" in DEFAULT_LIMITS
        assert DEFAULT_LIMITS["deepseek"].tokens_per_second == 3.0


# ── DeepSeek category gating ─────────────────────────────────────────


class TestDeepSeekCategoryGating:
    def _make_ensemble(self, **overrides) -> EnsembleForecaster:
        ensemble_defaults = dict(
            models=["gpt-4o", "deepseek-chat"],
            aggregation="median",
            min_models_required=1,
        )
        ensemble_defaults.update(overrides)
        ensemble_cfg = EnsembleConfig(**ensemble_defaults)
        forecast_cfg = _make_forecast_config()
        return EnsembleForecaster(ensemble_cfg, forecast_cfg)

    @patch("src.forecast.ensemble._query_model")
    def test_deepseek_excluded_from_geopolitics(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="gpt-4o", model_probability=0.6
        )

        forecaster = self._make_ensemble()
        features = _make_features(category="GEOPOLITICS")
        evidence = _make_evidence()

        asyncio.run(
            forecaster.forecast(features, evidence)
        )

        # Should only query gpt-4o, not deepseek-chat
        called_models = [call.args[0] for call in mock_query.call_args_list]
        assert "gpt-4o" in called_models
        assert "deepseek-chat" not in called_models

    @patch("src.forecast.ensemble._query_model")
    def test_deepseek_excluded_from_election(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="gpt-4o", model_probability=0.6
        )

        forecaster = self._make_ensemble()
        features = _make_features(category="ELECTION")
        evidence = _make_evidence()

        asyncio.run(
            forecaster.forecast(features, evidence)
        )

        called_models = [call.args[0] for call in mock_query.call_args_list]
        assert "deepseek-chat" not in called_models

    @patch("src.forecast.ensemble._query_model")
    def test_deepseek_not_excluded_from_macro(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="test", model_probability=0.6
        )

        forecaster = self._make_ensemble()
        features = _make_features(category="MACRO")
        evidence = _make_evidence()

        asyncio.run(
            forecaster.forecast(features, evidence)
        )

        called_models = [call.args[0] for call in mock_query.call_args_list]
        assert "deepseek-chat" in called_models

    @patch("src.forecast.ensemble._query_model")
    def test_deepseek_not_excluded_from_science(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="test", model_probability=0.6
        )

        forecaster = self._make_ensemble()
        features = _make_features(category="SCIENCE")
        evidence = _make_evidence()

        asyncio.run(
            forecaster.forecast(features, evidence)
        )

        called_models = [call.args[0] for call in mock_query.call_args_list]
        assert "deepseek-chat" in called_models

    @patch("src.forecast.ensemble._query_model")
    def test_deepseek_not_excluded_from_crypto(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="test", model_probability=0.6
        )

        forecaster = self._make_ensemble()
        features = _make_features(category="CRYPTO")
        evidence = _make_evidence()

        asyncio.run(
            forecaster.forecast(features, evidence)
        )

        called_models = [call.args[0] for call in mock_query.call_args_list]
        assert "deepseek-chat" in called_models

    @patch("src.forecast.ensemble._query_model")
    def test_grok_not_excluded_from_any_category(self, mock_query) -> None:
        mock_query.return_value = ModelForecast(
            model_name="test", model_probability=0.6
        )

        ensemble_cfg = EnsembleConfig(
            models=["gpt-4o", "grok-4-fast-reasoning"],
            aggregation="median",
            min_models_required=1,
        )
        forecast_cfg = _make_forecast_config()
        forecaster = EnsembleForecaster(ensemble_cfg, forecast_cfg)

        for cat in ["GEOPOLITICS", "ELECTION", "MACRO", "SCIENCE", "CRYPTO"]:
            mock_query.reset_mock()
            features = _make_features(category=cat)
            evidence = _make_evidence()

            asyncio.run(
                forecaster.forecast(features, evidence)
            )

            called_models = [call.args[0] for call in mock_query.call_args_list]
            assert "grok-4-fast-reasoning" in called_models, f"Grok excluded from {cat}"


# ── Median aggregation ────────────────────────────────────────────────


class TestMedianAggregation:
    def _make_forecaster(self) -> EnsembleForecaster:
        cfg = EnsembleConfig(
            models=["m1", "m2", "m3", "m4", "m5"],
            aggregation="median",
            min_models_required=1,
        )
        return EnsembleForecaster(cfg, _make_forecast_config())

    def test_median_of_5(self) -> None:
        forecaster = self._make_forecaster()
        probs = [("m1", 0.4), ("m2", 0.5), ("m3", 0.6), ("m4", 0.7), ("m5", 0.8)]
        result, method = forecaster._aggregate(probs)
        assert result == 0.6
        assert method == "median"

    def test_median_of_4(self) -> None:
        forecaster = self._make_forecaster()
        probs = [("m1", 0.4), ("m2", 0.5), ("m3", 0.7), ("m4", 0.8)]
        result, method = forecaster._aggregate(probs)
        assert result == 0.6  # (0.5 + 0.7) / 2
        assert method == "median"

    def test_median_of_2(self) -> None:
        forecaster = self._make_forecaster()
        probs = [("m1", 0.4), ("m2", 0.6)]
        result, method = forecaster._aggregate(probs)
        assert result == 0.5  # (0.4 + 0.6) / 2

    def test_median_of_3(self) -> None:
        forecaster = self._make_forecaster()
        probs = [("m1", 0.3), ("m2", 0.6), ("m3", 0.9)]
        result, method = forecaster._aggregate(probs)
        assert result == 0.6


# ── Config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_ensemble_config_deepseek_excluded_defaults(self) -> None:
        cfg = EnsembleConfig()
        assert cfg.deepseek_excluded_categories == ["GEOPOLITICS", "ELECTION"]

    def test_ensemble_config_custom_exclusion(self) -> None:
        cfg = EnsembleConfig(deepseek_excluded_categories=["GEOPOLITICS", "ELECTION", "MACRO"])
        assert "MACRO" in cfg.deepseek_excluded_categories

    def test_model_tier_config_premium_includes_new_models(self) -> None:
        # Check that the default ModelTierConfig can accept new models
        cfg = ModelTierConfig(
            premium_models=["gpt-4o", "claude-sonnet-4-6", "gemini-2.5-flash",
                            "grok-4-fast-reasoning", "deepseek-chat"]
        )
        assert "grok-4-fast-reasoning" in cfg.premium_models
        assert "deepseek-chat" in cfg.premium_models


# ── Mocked API calls ─────────────────────────────────────────────────


class TestMockedApiCalls:
    def test_query_xai_uses_correct_base_url(self) -> None:
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("xai")
            original = cb.allow_request
            cb.allow_request = lambda: True
            try:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = '{"probability": 0.6, "confidence": "MEDIUM"}'
                mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_cls:
                    result = asyncio.run(
                        _query_xai("grok-4-fast-reasoning", "test", _make_forecast_config())
                    )
                    mock_cls.assert_called_once_with(
                        api_key="test-key",
                        base_url="https://api.x.ai/v1",
                    )
                    assert not result.error
            finally:
                cb.allow_request = original

    def test_query_deepseek_uses_correct_base_url(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("deepseek")
            original = cb.allow_request
            cb.allow_request = lambda: True
            try:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = '{"probability": 0.55, "confidence": "MEDIUM"}'
                mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_cls:
                    result = asyncio.run(
                        _query_deepseek("deepseek-chat", "test", _make_forecast_config())
                    )
                    mock_cls.assert_called_once_with(
                        api_key="test-key",
                        base_url="https://api.deepseek.com/v1",
                    )
                    assert not result.error
            finally:
                cb.allow_request = original

    def test_xai_cost_tracker_records_call(self) -> None:
        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("xai")
            original = cb.allow_request
            cb.allow_request = lambda: True
            try:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = '{"probability": 0.6}'
                mock_resp.usage = MagicMock(prompt_tokens=200, completion_tokens=100)

                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                with patch("openai.AsyncOpenAI", return_value=mock_client):
                    with patch("src.forecast.ensemble.cost_tracker") as mock_ct:
                        asyncio.run(
                            _query_xai("grok-4-fast-reasoning", "test", _make_forecast_config())
                        )
                        mock_ct.record_call.assert_called_once_with(
                            "grok-4-fast-reasoning",
                            input_tokens=200,
                            output_tokens=100,
                        )
            finally:
                cb.allow_request = original

    def test_deepseek_records_success_on_cb(self) -> None:
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False):
            from src.observability.circuit_breaker import circuit_breakers
            cb = circuit_breakers.get("deepseek")
            original_allow = cb.allow_request
            original_success = cb.record_success
            cb.allow_request = lambda: True
            cb.record_success = MagicMock()
            try:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock()]
                mock_resp.choices[0].message.content = '{"probability": 0.5}'
                mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

                mock_client = AsyncMock()
                mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                with patch("openai.AsyncOpenAI", return_value=mock_client):
                    asyncio.run(
                        _query_deepseek("deepseek-chat", "test", _make_forecast_config())
                    )
                    cb.record_success.assert_called_once()
            finally:
                cb.allow_request = original_allow
                cb.record_success = original_success
