"""Base class and output contract for domain-specific forecasting specialists.

All specialists return a SpecialistResult that can be converted to
EnsembleResult (for bypass mode) or BaseRateMatch (for augment mode),
ensuring downstream edge calc, risk checks, and position sizing work
identically regardless of the forecast source.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpecialistResult:
    """Output from a domain-specific specialist."""
    probability: float              # 0.01–0.99
    confidence_level: str           # LOW | MEDIUM | HIGH
    reasoning: str
    evidence_quality: float         # 0.0–1.0
    specialist_name: str            # "weather", "crypto_ta", "politics"
    specialist_metadata: dict[str, Any] = field(default_factory=dict)
    bypasses_llm: bool = True       # False for augment-mode specialists
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    invalidation_triggers: list[str] = field(default_factory=list)


class BaseSpecialist(abc.ABC):
    """Abstract base for domain-specific forecasting specialists."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier: 'weather', 'crypto_ta', 'politics'."""
        ...

    @abc.abstractmethod
    def can_handle(self, classification: Any, question: str) -> bool:
        """Return True if this specialist can forecast the given market."""
        ...

    @abc.abstractmethod
    async def forecast(
        self,
        market: Any,
        features: Any,
        classification: Any,
    ) -> SpecialistResult:
        """Produce a specialist forecast for the given market."""
        ...

    async def close(self) -> None:
        """Release resources (HTTP clients, etc). Override if needed."""
        pass

    @staticmethod
    def to_ensemble_result(result: SpecialistResult) -> Any:
        """Convert SpecialistResult to EnsembleResult for downstream compat."""
        from src.forecast.ensemble import EnsembleResult, ModelForecast

        model_forecast = ModelForecast(
            model_name=f"specialist:{result.specialist_name}",
            model_probability=result.probability,
            confidence_level=result.confidence_level,
            reasoning=result.reasoning,
            key_evidence=result.key_evidence,
            invalidation_triggers=result.invalidation_triggers,
        )
        return EnsembleResult(
            model_probability=result.probability,
            confidence_level=result.confidence_level,
            individual_forecasts=[model_forecast],
            models_succeeded=1,
            models_failed=0,
            aggregation_method=f"specialist:{result.specialist_name}",
            spread=0.0,
            agreement_score=1.0,
            reasoning=result.reasoning,
            invalidation_triggers=result.invalidation_triggers,
            key_evidence=result.key_evidence,
        )

    @staticmethod
    def to_base_rate_match(result: SpecialistResult) -> Any:
        """Convert SpecialistResult to BaseRateMatch for augment-mode."""
        from src.forecast.base_rates import BaseRateMatch

        return BaseRateMatch(
            base_rate=result.probability,
            pattern_description=f"Specialist:{result.specialist_name} forecast",
            source=result.specialist_name,
            confidence=result.evidence_quality,
        )
