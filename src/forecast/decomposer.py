"""Question decomposition for superforecasting-style sub-question analysis.

Breaks complex market questions into 2-3 sub-questions that are
individually easier to estimate. Sub-probabilities are combined
via weighted average to produce the final forecast.

Uses a cheap model (gpt-4o-mini) for decomposition — ~$0.0003/call.
Sub-questions share the parent market's evidence (no re-search).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from src.config import ForecastingConfig
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class SubQuestion:
    """A single sub-question from decomposition."""
    text: str
    weight: float = 1.0
    dependency_type: str = "independent"  # independent | conditional


@dataclass
class DecompositionResult:
    """Result of question decomposition."""
    original_question: str = ""
    sub_questions: list[SubQuestion] = field(default_factory=list)
    combination_method: str = "weighted_average"
    decomposition_quality: float = 0.5
    error: str = ""


_DECOMPOSITION_PROMPT = """\
You are an expert at breaking complex prediction market questions into
simpler sub-questions whose answers inform the main question.

QUESTION: {question}
MARKET TYPE: {market_type}

Break this question into 2-{max_sub} sub-questions that:
1. Are individually easier to estimate than the main question
2. Together cover the key factors that determine the main outcome
3. Each have a weight (0.0-1.0) reflecting their importance to the main question

Return valid JSON:
{{
  "sub_questions": [
    {{
      "text": "Sub-question text?",
      "weight": <0.0-1.0>,
      "dependency_type": "independent"
    }}
  ],
  "combination_method": "weighted_average",
  "decomposition_quality": <0.0-1.0, your confidence this decomposition is useful>
}}

RULES:
- Generate exactly 2 to {max_sub} sub-questions.
- Weights should sum to approximately 1.0.
- Each sub-question must be a clear yes/no question.
- dependency_type is "independent" for most sub-questions.
- decomposition_quality should be LOW (<0.3) if the question doesn't
  decompose well (e.g., simple binary questions).
- Return ONLY valid JSON, no markdown fences.
"""

# Categories where decomposition is unlikely to help
_SKIP_CATEGORIES = {"SPORTS"}

# Minimum question length for decomposition
_MIN_QUESTION_LENGTH = 20


def should_decompose(question: str, market_type: str = "") -> bool:
    """Heuristic filter: should this question be decomposed?

    Returns False for questions that are too simple or in categories
    where decomposition doesn't help.
    """
    if not question or len(question) < _MIN_QUESTION_LENGTH:
        return False

    if market_type in _SKIP_CATEGORIES:
        return False

    return True


def combine_sub_forecasts(
    decomposition: DecompositionResult,
    sub_probs: list[float],
) -> float:
    """Combine sub-question probabilities into a single forecast.

    Uses weighted average of sub-probabilities. Falls back to simple
    mean if weights sum to 0.

    Args:
        decomposition: The decomposition result with sub-questions and weights.
        sub_probs: Probability estimates for each sub-question.

    Returns:
        Combined probability estimate.
    """
    if not sub_probs:
        return 0.5

    sqs = decomposition.sub_questions
    if len(sub_probs) != len(sqs):
        # Mismatch — just average
        return sum(sub_probs) / len(sub_probs)

    total_weight = sum(sq.weight for sq in sqs)
    if total_weight <= 0:
        return sum(sub_probs) / len(sub_probs)

    weighted_sum = sum(p * sq.weight for p, sq in zip(sub_probs, sqs))
    result = weighted_sum / total_weight
    return max(0.01, min(0.99, result))


class QuestionDecomposer:
    """Decomposes market questions into sub-questions using a cheap LLM.

    Uses gpt-4o-mini (configurable) for decomposition prompts.
    Results can be cached via the LLM response cache.
    """

    def __init__(self, config: ForecastingConfig):
        self._config = config
        self._model = config.decomposition_model
        self._max_sub = config.max_sub_questions

    async def decompose(
        self, question: str, market_type: str = "",
    ) -> DecompositionResult:
        """Decompose a question into sub-questions.

        Args:
            question: The market question to decompose.
            market_type: Optional market type for context.

        Returns:
            DecompositionResult with sub-questions and combination logic.
            On error, returns empty result with error message.
        """
        prompt = _DECOMPOSITION_PROMPT.format(
            question=question,
            market_type=market_type or "UNKNOWN",
            max_sub=self._max_sub,
        )

        try:
            parsed = await self._call_llm(prompt)
            return self._parse_result(question, parsed)
        except Exception as e:
            log.warning(
                "decomposer.failed",
                question=question[:80],
                error=str(e),
            )
            return DecompositionResult(
                original_question=question,
                error=str(e),
            )

    async def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the decomposition LLM and parse JSON response."""
        from src.forecast.ensemble import _parse_llm_json, _query_model

        result = await _query_model(
            self._model, prompt, self._config,
            timeout_secs=15,
        )

        if result.error:
            raise RuntimeError(f"LLM call failed: {result.error}")

        return result.raw_response

    def _parse_result(
        self, question: str, parsed: dict[str, Any],
    ) -> DecompositionResult:
        """Parse LLM response into DecompositionResult."""
        raw_sqs = parsed.get("sub_questions", [])

        sub_questions: list[SubQuestion] = []
        for sq_data in raw_sqs[:self._max_sub]:
            text = sq_data.get("text", "").strip()
            if not text:
                continue
            weight = float(sq_data.get("weight", 1.0 / max(len(raw_sqs), 1)))
            weight = max(0.0, min(1.0, weight))
            dep_type = sq_data.get("dependency_type", "independent")
            sub_questions.append(SubQuestion(
                text=text,
                weight=weight,
                dependency_type=dep_type,
            ))

        # Normalize weights to sum to 1.0
        total_w = sum(sq.weight for sq in sub_questions)
        if total_w > 0 and abs(total_w - 1.0) > 0.01:
            for sq in sub_questions:
                sq.weight = sq.weight / total_w

        quality = float(parsed.get("decomposition_quality", 0.5))
        quality = max(0.0, min(1.0, quality))

        method = parsed.get("combination_method", "weighted_average")

        log.info(
            "decomposer.result",
            question=question[:80],
            num_sub_questions=len(sub_questions),
            quality=round(quality, 2),
        )

        return DecompositionResult(
            original_question=question,
            sub_questions=sub_questions,
            combination_method=method,
            decomposition_quality=quality,
        )
