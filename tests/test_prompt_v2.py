"""Tests for prompt v2 and _build_prompt updates (Phase 2 — Batch A)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.forecast.base_rates import BaseRateMatch
from src.forecast.ensemble import (
    ModelForecast,
    _build_model_forecast,
    _build_prompt,
    _FORECAST_PROMPT,
    _FORECAST_PROMPT_V2,
    _parse_llm_json,
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
        top_bullets=["CPI fell to 2.1%", "Fed signaled possible cut"],
    )
    defaults.update(kwargs)
    return MarketFeatures(**defaults)


def _make_evidence(**kwargs) -> EvidencePackage:
    defaults = dict(
        market_id="test-market",
        question="Will the Fed cut rates?",
        summary="CPI trends and Fed commentary.",
    )
    defaults.update(kwargs)
    return EvidencePackage(**defaults)


def _make_base_rate_info(**kwargs) -> BaseRateMatch:
    defaults = dict(
        base_rate=0.25,
        pattern_description="Fed cuts rates at any given meeting",
        source="FOMC historical decisions 1990-2024",
        confidence=0.7,
        category="MACRO",
        sample_size=270,
    )
    defaults.update(kwargs)
    return BaseRateMatch(**defaults)


# ── Prompt template tests ────────────────────────────────────────────


class TestPromptTemplates:

    def test_v1_template_has_task_section(self) -> None:
        """v1 template contains TASK section."""
        assert "TASK:" in _FORECAST_PROMPT
        assert "MARKET QUESTION:" in _FORECAST_PROMPT

    def test_v2_template_has_base_rate_block(self) -> None:
        """v2 template has {base_rate_block} placeholder."""
        assert "{base_rate_block}" in _FORECAST_PROMPT_V2

    def test_v2_template_has_structured_reasoning(self) -> None:
        """v2 template enforces reasoning chain."""
        assert "START WITH THE BASE RATE" in _FORECAST_PROMPT_V2
        assert "EVIDENCE FOR" in _FORECAST_PROMPT_V2
        assert "EVIDENCE AGAINST" in _FORECAST_PROMPT_V2
        assert "ADJUSTMENT" in _FORECAST_PROMPT_V2

    def test_v2_template_requires_structured_json(self) -> None:
        """v2 template specifies new JSON fields."""
        assert '"base_rate"' in _FORECAST_PROMPT_V2
        assert '"evidence_for"' in _FORECAST_PROMPT_V2
        assert '"evidence_against"' in _FORECAST_PROMPT_V2
        assert '"adjustment_reasoning"' in _FORECAST_PROMPT_V2


# ── _build_prompt v1 backward compatibility ──────────────────────────


class TestBuildPromptV1:

    def test_v1_default(self) -> None:
        """Default (no version arg) produces v1 prompt."""
        features = _make_features()
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence)
        assert "MARKET QUESTION: Will the Fed cut rates?" in prompt
        assert "superforecasting" not in prompt

    def test_v1_explicit(self) -> None:
        """Explicit version='v1' produces v1 prompt."""
        features = _make_features()
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, prompt_version="v1")
        assert "TASK:" in prompt
        assert "START WITH THE BASE RATE" not in prompt

    def test_v1_includes_evidence_bullets(self) -> None:
        """v1 prompt includes evidence bullets."""
        features = _make_features(top_bullets=["CPI fell to 2.1%"])
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, prompt_version="v1")
        assert "CPI fell to 2.1%" in prompt

    def test_v1_no_bullets(self) -> None:
        """v1 prompt handles empty bullets."""
        features = _make_features(top_bullets=[])
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, prompt_version="v1")
        assert "No evidence bullets available." in prompt

    def test_v1_ignores_base_rate_info(self) -> None:
        """v1 prompt ignores base_rate_info even if provided."""
        features = _make_features()
        evidence = _make_evidence()
        base_rate = _make_base_rate_info()
        prompt = _build_prompt(features, evidence, base_rate_info=base_rate, prompt_version="v1")
        assert "HISTORICAL BASE RATE" not in prompt


# ── _build_prompt v2 ─────────────────────────────────────────────────


class TestBuildPromptV2:

    def test_v2_with_base_rate(self) -> None:
        """v2 prompt includes base rate block when provided."""
        features = _make_features()
        evidence = _make_evidence()
        base_rate = _make_base_rate_info(base_rate=0.25)
        prompt = _build_prompt(features, evidence, base_rate_info=base_rate, prompt_version="v2")

        assert "HISTORICAL BASE RATE:" in prompt
        assert "Base rate: 25%" in prompt
        assert "Fed cuts rates at any given meeting" in prompt
        assert "FOMC historical decisions" in prompt
        assert "Start from this base rate" in prompt

    def test_v2_without_base_rate(self) -> None:
        """v2 prompt handles None base_rate_info gracefully."""
        features = _make_features()
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, base_rate_info=None, prompt_version="v2")

        assert "HISTORICAL BASE RATE:" in prompt
        assert "No specific base rate available" in prompt
        assert "Estimate an appropriate base rate" in prompt

    def test_v2_includes_structured_reasoning(self) -> None:
        """v2 prompt includes structured reasoning chain."""
        features = _make_features()
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, prompt_version="v2")

        assert "START WITH THE BASE RATE" in prompt
        assert "EVIDENCE FOR" in prompt
        assert "EVIDENCE AGAINST" in prompt
        assert "superforecasting" in prompt

    def test_v2_includes_market_features(self) -> None:
        """v2 prompt includes market features."""
        features = _make_features(volume_usd=50000.0)
        evidence = _make_evidence()
        prompt = _build_prompt(features, evidence, prompt_version="v2")

        assert "$50,000" in prompt
        assert "Days to expiry" in prompt

    def test_v2_includes_evidence(self) -> None:
        """v2 prompt includes evidence bullets."""
        features = _make_features(top_bullets=["CPI fell to 2.1%"])
        evidence = _make_evidence(summary="Inflation data analysis.")
        prompt = _build_prompt(features, evidence, prompt_version="v2")

        assert "CPI fell to 2.1%" in prompt
        assert "Inflation data analysis." in prompt


# ── _parse_llm_json ──────────────────────────────────────────────────


class TestParseLLMJson:

    def test_v1_response(self) -> None:
        """Parses v1 response (no extra fields)."""
        response = json.dumps({
            "model_probability": 0.65,
            "confidence_level": "MEDIUM",
            "reasoning": "Evidence supports yes.",
        })
        parsed = _parse_llm_json(response)
        assert parsed["model_probability"] == 0.65
        assert "base_rate" not in parsed

    def test_v2_response(self) -> None:
        """Parses v2 response with extra fields."""
        response = json.dumps({
            "base_rate": 0.25,
            "base_rate_reasoning": "Historical Fed rate cut frequency",
            "evidence_for": ["CPI dropped", "Fed signaled"],
            "evidence_against": ["Labor market strong"],
            "adjustment_reasoning": "Evidence shifts probability up from base rate",
            "model_probability": 0.40,
            "confidence_level": "MEDIUM",
            "reasoning": "Adjusted from 25% base rate.",
        })
        parsed = _parse_llm_json(response)
        assert parsed["model_probability"] == 0.40
        assert parsed["base_rate"] == 0.25
        assert len(parsed["evidence_for"]) == 2
        assert len(parsed["evidence_against"]) == 1

    def test_with_markdown_fences(self) -> None:
        """Handles markdown-fenced JSON."""
        response = '```json\n{"model_probability": 0.5}\n```'
        parsed = _parse_llm_json(response)
        assert parsed["model_probability"] == 0.5


# ── _build_model_forecast ────────────────────────────────────────────


class TestBuildModelForecast:

    def test_v1_parsed_data(self) -> None:
        """Builds ModelForecast from v1 parsed data."""
        parsed = {
            "model_probability": 0.65,
            "confidence_level": "MEDIUM",
            "reasoning": "Test.",
        }
        mf = _build_model_forecast("gpt-4o", parsed, 100.0)
        assert mf.model_probability == 0.65
        assert mf.base_rate == 0.0
        assert mf.evidence_for == []
        assert mf.evidence_against == []

    def test_v2_parsed_data(self) -> None:
        """Builds ModelForecast from v2 parsed data."""
        parsed = {
            "model_probability": 0.40,
            "confidence_level": "MEDIUM",
            "reasoning": "Adjusted from base rate.",
            "base_rate": 0.25,
            "evidence_for": ["CPI dropped"],
            "evidence_against": ["Labor strong"],
        }
        mf = _build_model_forecast("gpt-4o", parsed, 150.0)
        assert mf.model_probability == 0.40
        assert mf.base_rate == 0.25
        assert mf.evidence_for == ["CPI dropped"]
        assert mf.evidence_against == ["Labor strong"]
        assert mf.latency_ms == 150.0

    def test_clamps_probability(self) -> None:
        """Probability is clamped to [0.01, 0.99]."""
        parsed = {"model_probability": 1.5}
        mf = _build_model_forecast("gpt-4o", parsed, 0.0)
        assert mf.model_probability == 0.99

        parsed = {"model_probability": -0.5}
        mf = _build_model_forecast("gpt-4o", parsed, 0.0)
        assert mf.model_probability == 0.01

    def test_defaults_on_missing(self) -> None:
        """Missing fields get sensible defaults."""
        mf = _build_model_forecast("gpt-4o", {}, 0.0)
        assert mf.model_probability == 0.5
        assert mf.confidence_level == "LOW"
        assert mf.reasoning == ""
        assert mf.base_rate == 0.0


# ── ModelForecast new fields ─────────────────────────────────────────


class TestModelForecastFields:

    def test_default_fields(self) -> None:
        """New v2 fields default correctly."""
        mf = ModelForecast(model_name="test", model_probability=0.5)
        assert mf.base_rate == 0.0
        assert mf.evidence_for == []
        assert mf.evidence_against == []

    def test_set_v2_fields(self) -> None:
        """New v2 fields can be set."""
        mf = ModelForecast(
            model_name="test",
            model_probability=0.5,
            base_rate=0.25,
            evidence_for=["point1"],
            evidence_against=["point2"],
        )
        assert mf.base_rate == 0.25
        assert mf.evidence_for == ["point1"]
        assert mf.evidence_against == ["point2"]
