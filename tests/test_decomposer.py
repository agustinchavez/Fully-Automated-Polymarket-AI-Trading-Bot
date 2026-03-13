"""Tests for question decomposition (Phase 2 — Batch B)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ForecastingConfig
from src.forecast.decomposer import (
    DecompositionResult,
    QuestionDecomposer,
    SubQuestion,
    _MIN_QUESTION_LENGTH,
    combine_sub_forecasts,
    should_decompose,
)


@pytest.fixture
def config() -> ForecastingConfig:
    return ForecastingConfig(
        decomposition_enabled=True,
        decomposition_model="gpt-4o-mini",
        max_sub_questions=3,
    )


@pytest.fixture
def decomposer(config: ForecastingConfig) -> QuestionDecomposer:
    return QuestionDecomposer(config)


# ── should_decompose heuristics ──────────────────────────────────────


class TestShouldDecompose:

    def test_normal_question(self) -> None:
        """Normal question should be decomposed."""
        assert should_decompose("Will the Federal Reserve cut interest rates in June?")

    def test_short_question(self) -> None:
        """Short question should not be decomposed."""
        assert not should_decompose("Yes or no?")

    def test_empty_question(self) -> None:
        """Empty question should not be decomposed."""
        assert not should_decompose("")

    def test_sports_category_skipped(self) -> None:
        """SPORTS category should not be decomposed."""
        assert not should_decompose(
            "Will the Lakers win the championship?",
            market_type="SPORTS",
        )

    def test_macro_category_allowed(self) -> None:
        """MACRO category should be decomposed."""
        assert should_decompose(
            "Will GDP growth exceed estimates?",
            market_type="MACRO",
        )

    def test_below_min_length(self) -> None:
        """Questions shorter than minimum length are skipped."""
        short = "x" * (_MIN_QUESTION_LENGTH - 1)
        assert not should_decompose(short)

    def test_at_min_length(self) -> None:
        """Questions at exactly minimum length are allowed."""
        exact = "x" * _MIN_QUESTION_LENGTH
        assert should_decompose(exact)


# ── combine_sub_forecasts ────────────────────────────────────────────


class TestCombineSubForecasts:

    def test_weighted_average(self) -> None:
        """Produces weighted average of sub-probabilities."""
        decomp = DecompositionResult(
            sub_questions=[
                SubQuestion(text="Q1", weight=0.6),
                SubQuestion(text="Q2", weight=0.4),
            ],
        )
        result = combine_sub_forecasts(decomp, [0.8, 0.2])
        expected = 0.8 * 0.6 + 0.2 * 0.4  # 0.56
        assert result == pytest.approx(expected, abs=0.001)

    def test_equal_weights(self) -> None:
        """Equal weights produce simple average."""
        decomp = DecompositionResult(
            sub_questions=[
                SubQuestion(text="Q1", weight=0.5),
                SubQuestion(text="Q2", weight=0.5),
            ],
        )
        result = combine_sub_forecasts(decomp, [0.6, 0.4])
        assert result == pytest.approx(0.5, abs=0.001)

    def test_single_sub_question(self) -> None:
        """Single sub-question returns its probability."""
        decomp = DecompositionResult(
            sub_questions=[SubQuestion(text="Q1", weight=1.0)],
        )
        result = combine_sub_forecasts(decomp, [0.7])
        assert result == pytest.approx(0.7, abs=0.001)

    def test_empty_probs(self) -> None:
        """Empty probabilities return 0.5."""
        decomp = DecompositionResult()
        assert combine_sub_forecasts(decomp, []) == 0.5

    def test_zero_weights_fallback(self) -> None:
        """Zero weights fall back to simple mean."""
        decomp = DecompositionResult(
            sub_questions=[
                SubQuestion(text="Q1", weight=0.0),
                SubQuestion(text="Q2", weight=0.0),
            ],
        )
        result = combine_sub_forecasts(decomp, [0.6, 0.4])
        assert result == pytest.approx(0.5, abs=0.001)

    def test_mismatched_lengths(self) -> None:
        """Mismatched lengths fall back to simple mean of probs."""
        decomp = DecompositionResult(
            sub_questions=[SubQuestion(text="Q1", weight=1.0)],
        )
        result = combine_sub_forecasts(decomp, [0.6, 0.4])
        assert result == pytest.approx(0.5, abs=0.001)

    def test_clamped_result(self) -> None:
        """Result is clamped to [0.01, 0.99]."""
        decomp = DecompositionResult(
            sub_questions=[SubQuestion(text="Q1", weight=1.0)],
        )
        # Extreme probability
        result = combine_sub_forecasts(decomp, [0.001])
        assert result >= 0.01

    def test_three_sub_questions(self) -> None:
        """Three sub-questions with varying weights."""
        decomp = DecompositionResult(
            sub_questions=[
                SubQuestion(text="Q1", weight=0.5),
                SubQuestion(text="Q2", weight=0.3),
                SubQuestion(text="Q3", weight=0.2),
            ],
        )
        result = combine_sub_forecasts(decomp, [0.8, 0.5, 0.3])
        expected = (0.8 * 0.5 + 0.5 * 0.3 + 0.3 * 0.2) / 1.0  # 0.61
        assert result == pytest.approx(expected, abs=0.001)


# ── QuestionDecomposer._parse_result ─────────────────────────────────


class TestParseResult:

    def test_basic_parsing(self, decomposer: QuestionDecomposer) -> None:
        """Parses valid decomposition response."""
        parsed = {
            "sub_questions": [
                {"text": "Sub Q1?", "weight": 0.6, "dependency_type": "independent"},
                {"text": "Sub Q2?", "weight": 0.4, "dependency_type": "independent"},
            ],
            "combination_method": "weighted_average",
            "decomposition_quality": 0.8,
        }
        result = decomposer._parse_result("Main question?", parsed)

        assert len(result.sub_questions) == 2
        assert result.sub_questions[0].text == "Sub Q1?"
        assert result.decomposition_quality == 0.8
        assert result.error == ""

    def test_weights_normalized(self, decomposer: QuestionDecomposer) -> None:
        """Weights are normalized to sum to 1.0."""
        parsed = {
            "sub_questions": [
                {"text": "Q1?", "weight": 0.3},
                {"text": "Q2?", "weight": 0.3},
            ],
        }
        result = decomposer._parse_result("Main?", parsed)
        total = sum(sq.weight for sq in result.sub_questions)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_max_sub_questions_enforced(self, decomposer: QuestionDecomposer) -> None:
        """Truncates to max_sub_questions."""
        parsed = {
            "sub_questions": [
                {"text": f"Q{i}?", "weight": 0.2}
                for i in range(10)
            ],
        }
        result = decomposer._parse_result("Main?", parsed)
        assert len(result.sub_questions) <= 3

    def test_empty_text_skipped(self, decomposer: QuestionDecomposer) -> None:
        """Sub-questions with empty text are skipped."""
        parsed = {
            "sub_questions": [
                {"text": "Valid Q?", "weight": 0.5},
                {"text": "", "weight": 0.5},
                {"text": "   ", "weight": 0.5},
            ],
        }
        result = decomposer._parse_result("Main?", parsed)
        assert len(result.sub_questions) == 1

    def test_quality_clamped(self, decomposer: QuestionDecomposer) -> None:
        """Quality is clamped to [0, 1]."""
        parsed = {
            "sub_questions": [{"text": "Q?", "weight": 1.0}],
            "decomposition_quality": 2.5,
        }
        result = decomposer._parse_result("Main?", parsed)
        assert result.decomposition_quality == 1.0

    def test_weight_clamped(self, decomposer: QuestionDecomposer) -> None:
        """Individual weights are clamped to [0, 1]."""
        parsed = {
            "sub_questions": [
                {"text": "Q?", "weight": 5.0},
            ],
        }
        result = decomposer._parse_result("Main?", parsed)
        assert result.sub_questions[0].weight <= 1.0

    def test_missing_fields_default(self, decomposer: QuestionDecomposer) -> None:
        """Missing fields get defaults."""
        parsed = {"sub_questions": [{"text": "Q?"}]}
        result = decomposer._parse_result("Main?", parsed)
        assert result.sub_questions[0].dependency_type == "independent"
        assert result.combination_method == "weighted_average"
        assert result.decomposition_quality == 0.5


# ── QuestionDecomposer.decompose (with mocked LLM) ──────────────────


class TestDecomposeAsync:

    @pytest.mark.asyncio
    async def test_decompose_success(self, decomposer: QuestionDecomposer) -> None:
        """Successful decomposition returns sub-questions."""
        mock_response = {
            "sub_questions": [
                {"text": "Will inflation drop?", "weight": 0.5},
                {"text": "Will unemployment rise?", "weight": 0.5},
            ],
            "decomposition_quality": 0.7,
        }

        with patch.object(decomposer, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            result = await decomposer.decompose(
                "Will the Fed cut rates in June?", "MACRO",
            )

        assert len(result.sub_questions) == 2
        assert result.error == ""
        assert result.decomposition_quality == 0.7

    @pytest.mark.asyncio
    async def test_decompose_llm_failure(self, decomposer: QuestionDecomposer) -> None:
        """LLM failure returns empty result with error."""
        with patch.object(decomposer, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = RuntimeError("API timeout")
            result = await decomposer.decompose("Will the Fed cut rates?")

        assert result.sub_questions == []
        assert "API timeout" in result.error

    @pytest.mark.asyncio
    async def test_decompose_sets_original_question(self, decomposer: QuestionDecomposer) -> None:
        """Result contains the original question."""
        mock_response = {
            "sub_questions": [{"text": "Sub Q?", "weight": 1.0}],
        }
        with patch.object(decomposer, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response
            result = await decomposer.decompose("Original question here?")

        assert result.original_question == "Original question here?"


# ── Config integration ───────────────────────────────────────────────


class TestDecomposerConfig:

    def test_disabled_by_default(self) -> None:
        """Decomposition is disabled by default in config."""
        cfg = ForecastingConfig()
        assert cfg.decomposition_enabled is False

    def test_default_model(self) -> None:
        """Default decomposition model is gpt-4o-mini."""
        cfg = ForecastingConfig()
        assert cfg.decomposition_model == "gpt-4o-mini"

    def test_max_sub_questions_default(self) -> None:
        """Default max sub-questions is 3."""
        cfg = ForecastingConfig()
        assert cfg.max_sub_questions == 3

    def test_decomposer_uses_config_model(self, config: ForecastingConfig) -> None:
        """Decomposer uses model from config."""
        d = QuestionDecomposer(config)
        assert d._model == "gpt-4o-mini"

    def test_decomposer_uses_config_max_sub(self) -> None:
        """Decomposer uses max_sub from config."""
        cfg = ForecastingConfig(max_sub_questions=5)
        d = QuestionDecomposer(cfg)
        assert d._max_sub == 5
