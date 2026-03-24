"""Specialist router — maps market classifications to domain specialists.

After market classification, the router checks if any enabled specialist
can handle the market. If so, the specialist produces a forecast that
either bypasses or augments the general LLM ensemble pipeline.
"""

from __future__ import annotations

from typing import Any

from src.forecast.specialists.base import BaseSpecialist, SpecialistResult
from src.observability.logger import get_logger

log = get_logger(__name__)


class SpecialistRouter:
    """Routes markets to domain-specific specialists when available."""

    def __init__(self, config: Any):
        self._config = config
        self._specialists: list[BaseSpecialist] = []
        self._init_specialists()

    def _init_specialists(self) -> None:
        """Lazy-import and instantiate enabled specialists."""
        for name in self._config.enabled_specialists:
            try:
                specialist = self._load_specialist(name)
                if specialist:
                    self._specialists.append(specialist)
                    log.info("specialist_router.loaded", specialist=name)
            except Exception as e:
                log.warning(
                    "specialist_router.load_error",
                    specialist=name,
                    error=str(e),
                )

    def _load_specialist(self, name: str) -> BaseSpecialist | None:
        """Import and instantiate a specialist by name."""
        if name == "weather":
            from src.forecast.specialists.weather import WeatherSpecialist
            return WeatherSpecialist(self._config)
        if name == "crypto_ta":
            from src.forecast.specialists.crypto_ta import CryptoTASpecialist
            return CryptoTASpecialist(self._config)
        if name == "politics":
            from src.forecast.specialists.politics import PoliticsSpecialist
            return PoliticsSpecialist(self._config)
        log.warning("specialist_router.unknown_specialist", name=name)
        return None

    def match(
        self,
        classification: Any,
        question: str,
    ) -> BaseSpecialist | None:
        """Find the first specialist that can handle this market."""
        for specialist in self._specialists:
            try:
                if specialist.can_handle(classification, question):
                    return specialist
            except Exception as e:
                log.warning(
                    "specialist_router.match_error",
                    specialist=specialist.name,
                    error=str(e),
                )
        return None

    async def route(
        self,
        market: Any,
        features: Any,
        classification: Any,
    ) -> SpecialistResult | None:
        """Route to specialist and get result, or None to fall back."""
        specialist = self.match(classification, market.question)
        if specialist is None:
            return None

        try:
            result = await specialist.forecast(market, features, classification)
            log.info(
                "specialist_router.matched",
                specialist=specialist.name,
                market_id=getattr(market, "id", "?"),
                probability=round(result.probability, 3),
                bypasses_llm=result.bypasses_llm,
            )
            return result
        except Exception as e:
            log.warning(
                "specialist_router.forecast_error",
                specialist=specialist.name,
                market_id=getattr(market, "id", "?"),
                error=str(e),
            )
            return None  # Fall back to general pipeline

    async def close(self) -> None:
        """Release resources held by all specialists."""
        for s in self._specialists:
            try:
                await s.close()
            except Exception as e:
                log.warning(
                    "specialist_router.close_error",
                    specialist=s.name,
                    error=str(e),
                )
