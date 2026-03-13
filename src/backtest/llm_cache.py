"""LLM response cache for deterministic, cost-free backtest replay.

Cache key: SHA-256(question + model_name + prompt_template_version)

Deliberately excludes volatile market features (volume, liquidity, spread)
from the cache key so that cached responses can be reused across different
config variations. The forecaster's output should be similar for the same
question regardless of microstructure data.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from src.backtest.database import BacktestDatabase
from src.backtest.models import LLMCacheRecord
from src.forecast.ensemble import ModelForecast
from src.observability.logger import get_logger

log = get_logger(__name__)


class LLMResponseCache:
    """Deterministic LLM response cache backed by the backtest database."""

    def __init__(
        self,
        db: BacktestDatabase,
        template_version: str = "v1",
    ):
        self._db = db
        self._template_version = template_version
        self._hits = 0
        self._misses = 0

    def make_cache_key(self, question: str, model_name: str) -> str:
        """Generate deterministic cache key from question + model + version."""
        raw = f"{question}|{model_name}|{self._template_version}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, question: str, model_name: str) -> ModelForecast | None:
        """Look up cached response. Returns None on cache miss."""
        key = self.make_cache_key(question, model_name)
        record = self._db.get_llm_cache(key)
        if record is None:
            self._misses += 1
            return None

        self._hits += 1
        try:
            data = json.loads(record.response_json)
            return ModelForecast(
                model_name=data.get("model_name", model_name),
                model_probability=data.get("model_probability", 0.5),
                confidence_level=data.get("confidence_level", "LOW"),
                reasoning=data.get("reasoning", ""),
                invalidation_triggers=data.get("invalidation_triggers", []),
                key_evidence=data.get("key_evidence", []),
                raw_response=data.get("raw_response", {}),
                latency_ms=record.latency_ms,
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("llm_cache.deserialize_error", key=key, error=str(e))
            self._misses += 1
            self._hits -= 1  # undo the hit
            return None

    def put(
        self,
        question: str,
        model_name: str,
        forecast: ModelForecast,
        prompt: str = "",
    ) -> None:
        """Store a forecast in the cache."""
        import datetime as dt

        key = self.make_cache_key(question, model_name)
        question_hash = hashlib.sha256(question.encode()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest() if prompt else ""

        response_data = {
            "model_name": forecast.model_name,
            "model_probability": forecast.model_probability,
            "confidence_level": forecast.confidence_level,
            "reasoning": forecast.reasoning,
            "invalidation_triggers": forecast.invalidation_triggers,
            "key_evidence": forecast.key_evidence,
            "raw_response": forecast.raw_response,
        }

        record = LLMCacheRecord(
            cache_key=key,
            market_question_hash=question_hash,
            model_name=model_name,
            prompt_hash=prompt_hash,
            response_json=json.dumps(response_data, default=str),
            input_tokens=0,
            output_tokens=0,
            latency_ms=forecast.latency_ms,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._db.upsert_llm_cache(record)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, total),
        }
