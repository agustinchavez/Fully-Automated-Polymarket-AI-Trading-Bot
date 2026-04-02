"""Multi-model ensemble forecaster — queries multiple LLMs in parallel.

Supports:
  - GPT-4o (OpenAI)
  - Claude 3.5 Sonnet (Anthropic)
  - Gemini 1.5 Pro (Google)

Aggregation methods:
  - trimmed_mean: Remove highest and lowest, average the rest
  - median: Take the median probability
  - weighted: Use configurable per-model weights

Gracefully degrades if some models fail — requires min_models_required
to produce a forecast, otherwise falls back to single model.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any

from src.config import EnsembleConfig, ForecastingConfig
from src.forecast.feature_builder import MarketFeatures
from src.research.evidence_extractor import EvidencePackage
from src.observability.logger import get_logger
from src.connectors.rate_limiter import rate_limiter
from src.observability.metrics import cost_tracker
from src.observability.circuit_breaker import circuit_breakers

log = get_logger(__name__)


@dataclass
class ModelForecast:
    """Forecast from a single model."""
    model_name: str
    model_probability: float
    confidence_level: str = "LOW"
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: float = 0.0
    # Phase 2: Structured reasoning fields (populated by prompt v2)
    base_rate: float = 0.0
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Aggregated result from multiple models."""
    model_probability: float
    confidence_level: str = "LOW"
    individual_forecasts: list[ModelForecast] = field(default_factory=list)
    models_succeeded: int = 0
    models_failed: int = 0
    aggregation_method: str = "trimmed_mean"
    spread: float = 0.0  # max - min probability across models
    agreement_score: float = 0.0  # 1.0 = perfect agreement
    reasoning: str = ""
    invalidation_triggers: list[str] = field(default_factory=list)
    key_evidence: list[dict[str, Any]] = field(default_factory=list)


_FORECAST_PROMPT = """\
You are an expert probabilistic forecaster analyzing a prediction market.

MARKET QUESTION: {question}
MARKET TYPE: {market_type}

EVIDENCE SUMMARY:
{evidence_summary}

TOP EVIDENCE BULLETS:
{evidence_bullets}

{contradictions_block}

MARKET FEATURES:
- Volume: ${volume_usd:,.0f}
- Liquidity: ${liquidity_usd:,.0f}
- Spread: {spread_pct:.1%}
- Days to expiry: {days_to_expiry:.0f}
- Price momentum (24h): {price_momentum:+.3f}
- Evidence quality score: {evidence_quality:.2f}
- Sources analyzed: {num_sources}

TASK:
Based ONLY on the evidence above, produce an independent probability
estimate for the question. Do NOT try to guess or anchor to any market
price — form your own view from the evidence.

Return valid JSON:
{{
  "model_probability": <0.01-0.99>,
  "confidence_level": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "2-4 sentence explanation of your probability estimate",
  "invalidation_triggers": [
    "specific event/data that would change this forecast significantly"
  ],
  "key_evidence": [
    {{
      "text": "evidence bullet",
      "source": "publisher name",
      "impact": "supports/opposes/neutral"
    }}
  ]
}}

RULES:
- Your probability must be between 0.01 and 0.99.
- Form your estimate independently from evidence — do NOT anchor to any
  external price or implied probability.
- If evidence is weak (quality < 0.3), bias toward 0.50.
- If evidence contradicts itself, widen uncertainty toward 0.50.
- confidence_level:
  - HIGH = authoritative primary source data directly answers the question
  - MEDIUM = strong secondary sources with consistent direction
  - LOW = limited/conflicting/stale evidence
- Never claim certainty. Express epistemic humility.
- Do NOT hallucinate data not present in the evidence.

Return ONLY valid JSON, no markdown fences.
"""


_FORECAST_PROMPT_V2 = """\
You are an expert probabilistic forecaster using superforecasting methodology.

MARKET QUESTION: {question}
MARKET TYPE: {market_type}

{base_rate_block}

EVIDENCE SUMMARY:
{evidence_summary}

TOP EVIDENCE BULLETS:
{evidence_bullets}

{contradictions_block}

MARKET FEATURES:
- Volume: ${volume_usd:,.0f}
- Liquidity: ${liquidity_usd:,.0f}
- Spread: {spread_pct:.1%}
- Days to expiry: {days_to_expiry:.0f}
- Price momentum (24h): {price_momentum:+.3f}
- Evidence quality score: {evidence_quality:.2f}
- Sources analyzed: {num_sources}

TASK:
Follow this structured reasoning chain to produce your probability estimate:

1. START WITH THE BASE RATE: What is the historical frequency of this type
   of event? Use the base rate provided above if available, or estimate one
   from your knowledge. This is your starting anchor.

2. EVIDENCE FOR: List the strongest evidence supporting YES resolution.

3. EVIDENCE AGAINST: List the strongest evidence supporting NO resolution.

4. ADJUSTMENT: Explain how the specific evidence shifts the probability
   away from the base rate. Be explicit about the direction and magnitude
   of each adjustment.

5. FINAL PROBABILITY: Your calibrated estimate after adjustments.

Return valid JSON:
{{
  "base_rate": <float, the starting base rate you used>,
  "base_rate_reasoning": "1-2 sentence explanation of why this base rate applies",
  "evidence_for": [
    "strongest evidence point supporting YES"
  ],
  "evidence_against": [
    "strongest evidence point supporting NO"
  ],
  "adjustment_reasoning": "how evidence shifts from base rate to final probability",
  "model_probability": <0.01-0.99>,
  "confidence_level": "LOW" | "MEDIUM" | "HIGH",
  "reasoning": "2-4 sentence final summary",
  "invalidation_triggers": [
    "specific event/data that would change this forecast significantly"
  ],
  "key_evidence": [
    {{
      "text": "evidence bullet",
      "source": "publisher name",
      "impact": "supports/opposes/neutral"
    }}
  ]
}}

RULES:
- Your probability must be between 0.01 and 0.99.
- START from the base rate and adjust — do not ignore the anchor.
- Form your estimate independently from evidence — do NOT anchor to any
  external price or implied probability.
- If evidence is weak (quality < 0.3), stay closer to the base rate.
- If evidence contradicts itself, widen uncertainty toward the base rate.
- confidence_level:
  - HIGH = authoritative primary source data directly answers the question
  - MEDIUM = strong secondary sources with consistent direction
  - LOW = limited/conflicting/stale evidence
- Never claim certainty. Express epistemic humility.
- Do NOT hallucinate data not present in the evidence.

Return ONLY valid JSON, no markdown fences.
"""


def _build_prompt(
    features: MarketFeatures,
    evidence: EvidencePackage,
    base_rate_info: Any | None = None,
    prompt_version: str = "v1",
) -> str:
    """Build the forecast prompt from features and evidence.

    Args:
        features: Market features for the forecast.
        evidence: Evidence package from research.
        base_rate_info: Optional BaseRateMatch with historical base rate.
        prompt_version: "v1" (legacy) or "v2" (structured reasoning chain).
    """
    evidence_bullets = "\n".join(
        f"- {b}" for b in features.top_bullets
    ) if features.top_bullets else "No evidence bullets available."

    contradictions_block = ""
    if evidence.contradictions:
        lines = ["CONTRADICTIONS DETECTED:"]
        for c in evidence.contradictions:
            lines.append(
                f"- {c.claim_a} ({c.source_a.publisher}) vs "
                f"{c.claim_b} ({c.source_b.publisher}): {c.description}"
            )
        contradictions_block = "\n".join(lines)

    format_kwargs = dict(
        question=features.question,
        market_type=features.market_type,
        evidence_summary=evidence.summary or "No summary available.",
        evidence_bullets=evidence_bullets,
        contradictions_block=contradictions_block,
        volume_usd=features.volume_usd,
        liquidity_usd=features.liquidity_usd,
        spread_pct=features.spread_pct,
        days_to_expiry=features.days_to_expiry,
        price_momentum=features.price_momentum,
        evidence_quality=features.evidence_quality,
        num_sources=features.num_sources,
    )

    if prompt_version == "v2":
        # Build base rate block
        if base_rate_info is not None:
            base_rate_block = (
                f"HISTORICAL BASE RATE:\n"
                f"- Base rate: {base_rate_info.base_rate:.0%}\n"
                f"- Pattern: {base_rate_info.pattern_description}\n"
                f"- Source: {base_rate_info.source}\n"
                f"- Match confidence: {base_rate_info.confidence:.0%}\n"
                f"Start from this base rate and adjust based on the evidence below."
            )
        else:
            base_rate_block = (
                "HISTORICAL BASE RATE:\n"
                "No specific base rate available for this question type.\n"
                "Estimate an appropriate base rate from your knowledge before adjusting."
            )
        format_kwargs["base_rate_block"] = base_rate_block
        return _FORECAST_PROMPT_V2.format(**format_kwargs)

    return _FORECAST_PROMPT.format(**format_kwargs)


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    """Parse LLM response JSON with markdown fence handling."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    return json.loads(raw_text.strip())


def _build_model_forecast(model: str, parsed: dict[str, Any], latency_ms: float) -> ModelForecast:
    """Build a ModelForecast from parsed LLM JSON (works for v1 and v2)."""
    return ModelForecast(
        model_name=model,
        model_probability=max(0.01, min(0.99, float(parsed.get("model_probability", 0.5)))),
        confidence_level=parsed.get("confidence_level", "LOW"),
        reasoning=parsed.get("reasoning", ""),
        invalidation_triggers=parsed.get("invalidation_triggers", []),
        key_evidence=parsed.get("key_evidence", []),
        raw_response=parsed,
        latency_ms=latency_ms,
        # v2 structured fields (default gracefully for v1 responses)
        base_rate=float(parsed.get("base_rate", 0.0)),
        evidence_for=parsed.get("evidence_for", []),
        evidence_against=parsed.get("evidence_against", []),
    )


async def _query_openai(model: str, prompt: str, config: ForecastingConfig, timeout_secs: int = 30) -> ModelForecast:
    """Query an OpenAI model."""
    import time
    from openai import AsyncOpenAI

    provider_cb = circuit_breakers.get("openai")
    if not provider_cb.allow_request():
        return ModelForecast(
            model_name=model, model_probability=0.5,
            error=f"Circuit breaker open for openai (retry after {provider_cb.time_until_retry():.0f}s)",
            latency_ms=0.0,
        )
    start = time.monotonic()
    try:
        await rate_limiter.get("openai").acquire()
        client = AsyncOpenAI()
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                messages=[
                    {"role": "system", "content": "You are a calibrated probabilistic forecaster. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            ),
            timeout=timeout_secs,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = _parse_llm_json(raw)
        usage = getattr(resp, "usage", None)
        cost_tracker.record_call(
            model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        provider_cb.record_success()
        return _build_model_forecast(model, parsed, (time.monotonic() - start) * 1000)
    except Exception as e:
        provider_cb.record_failure()
        return ModelForecast(
            model_name=model, model_probability=0.5, error=str(e),
            latency_ms=(time.monotonic() - start) * 1000,
        )


async def _query_anthropic(model: str, prompt: str, config: ForecastingConfig, timeout_secs: int = 30) -> ModelForecast:
    """Query an Anthropic Claude model."""
    import time

    provider_cb = circuit_breakers.get("anthropic")
    if not provider_cb.allow_request():
        return ModelForecast(
            model_name=model, model_probability=0.5,
            error=f"Circuit breaker open for anthropic (retry after {provider_cb.time_until_retry():.0f}s)",
            latency_ms=0.0,
        )
    start = time.monotonic()
    try:
        import anthropic
        await rate_limiter.get("anthropic").acquire()
        client = anthropic.AsyncAnthropic()
        resp = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature,
                system="You are a calibrated probabilistic forecaster. Return only valid JSON.",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout_secs,
        )
        raw = resp.content[0].text if resp.content else "{}"
        parsed = _parse_llm_json(raw)
        usage = getattr(resp, "usage", None)
        cost_tracker.record_call(
            model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        provider_cb.record_success()
        return _build_model_forecast(model, parsed, (time.monotonic() - start) * 1000)
    except Exception as e:
        provider_cb.record_failure()
        return ModelForecast(
            model_name=model, model_probability=0.5, error=str(e),
            latency_ms=(time.monotonic() - start) * 1000,
        )


async def _query_google(model: str, prompt: str, config: ForecastingConfig, timeout_secs: int = 30) -> ModelForecast:
    """Query a Google Gemini model."""
    import time

    provider_cb = circuit_breakers.get("google")
    if not provider_cb.allow_request():
        return ModelForecast(
            model_name=model, model_probability=0.5,
            error=f"Circuit breaker open for google (retry after {provider_cb.time_until_retry():.0f}s)",
            latency_ms=0.0,
        )
    start = time.monotonic()
    try:
        import google.generativeai as genai
        await rate_limiter.get("google").acquire()
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        gmodel = genai.GenerativeModel(model)
        # Configure per-request to avoid thread-safety issues with global state
        gmodel._client = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
        resp = await asyncio.wait_for(
            asyncio.to_thread(
                gmodel.generate_content,
                f"You are a calibrated probabilistic forecaster. Return only valid JSON.\n\n{prompt}",
            ),
            timeout=timeout_secs,
        )
        raw = resp.text or "{}"
        parsed = _parse_llm_json(raw)
        usage_meta = getattr(resp, "usage_metadata", None)
        cost_tracker.record_call(
            model,
            input_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
        )
        provider_cb.record_success()
        return _build_model_forecast(model, parsed, (time.monotonic() - start) * 1000)
    except Exception as e:
        provider_cb.record_failure()
        return ModelForecast(
            model_name=model, model_probability=0.5, error=str(e),
            latency_ms=(time.monotonic() - start) * 1000,
        )


def _route_model(model: str) -> str:
    """Determine which provider a model name belongs to."""
    if "claude" in model.lower():
        return "anthropic"
    elif "gemini" in model.lower():
        return "google"
    else:
        return "openai"


async def _query_model(
    model: str, prompt: str, config: ForecastingConfig, timeout_secs: int = 30,
) -> ModelForecast:
    """Route a model query to the appropriate provider."""
    provider = _route_model(model)
    if provider == "anthropic":
        return await _query_anthropic(model, prompt, config, timeout_secs)
    elif provider == "google":
        return await _query_google(model, prompt, config, timeout_secs)
    else:
        return await _query_openai(model, prompt, config, timeout_secs)


class EnsembleForecaster:
    """Multi-model ensemble forecaster."""

    def __init__(self, ensemble_config: EnsembleConfig, forecast_config: ForecastingConfig):
        self._ensemble = ensemble_config
        self._forecast = forecast_config
        self._external_weights: dict[str, float] | None = None

    def set_adaptive_weights(self, weights: dict[str, float]) -> None:
        """Inject learned per-category weights from AdaptiveModelWeighter."""
        self._external_weights = weights

    async def forecast(
        self,
        features: MarketFeatures,
        evidence: EvidencePackage,
        base_rate_info: Any | None = None,
        prompt_version: str = "v1",
    ) -> EnsembleResult:
        """Query all configured models in parallel and aggregate."""
        prompt = _build_prompt(features, evidence, base_rate_info, prompt_version)

        # Query all models concurrently
        timeout = self._ensemble.timeout_per_model_secs
        tasks = [
            _query_model(model, prompt, self._forecast, timeout)
            for model in self._ensemble.models
        ]
        forecasts = await asyncio.gather(*tasks)

        # Separate successes and failures
        successes = [f for f in forecasts if not f.error]
        failures = [f for f in forecasts if f.error]

        for f in failures:
            log.warning("ensemble.model_failed", model=f.model_name, error=f.error)

        # Check minimum models
        if len(successes) < self._ensemble.min_models_required:
            log.warning(
                "ensemble.insufficient_models",
                succeeded=len(successes),
                required=self._ensemble.min_models_required,
            )
            # Fallback — try a different provider if the primary fallback failed
            fallback_model = self._ensemble.fallback_model
            fallback_provider = _route_model(fallback_model)
            failed_providers = {_route_model(f.model_name) for f in failures}
            if fallback_provider in failed_providers:
                # Switch fallback to a provider that didn't fail
                alt_models = {
                    "openai": "gpt-4o",
                    "anthropic": "claude-sonnet-4-6",
                    "google": "gemini-2.0-flash",
                }
                for prov, mdl in alt_models.items():
                    if prov not in failed_providers:
                        fallback_model = mdl
                        break
            fallback = await _query_model(
                fallback_model, prompt, self._forecast, timeout,
            )
            if fallback.error:
                return EnsembleResult(
                    model_probability=0.5,
                    confidence_level="LOW",
                    models_succeeded=0,
                    models_failed=len(forecasts) + 1,
                    reasoning="All models failed",
                )
            successes = [fallback]

        # Warn when ensemble is degraded (fewer than 2 models succeeded)
        if len(successes) < 2 and len(self._ensemble.models) >= 2:
            log.warning(
                "ensemble.degraded_single_model",
                succeeded=len(successes),
                configured=len(self._ensemble.models),
                model=successes[0].model_name if successes else "none",
            )

        # Aggregate probabilities
        model_probs = [(f.model_name, f.model_probability) for f in successes]
        agg_prob, agg_method_used = self._aggregate(model_probs)

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

        # Model spread = disagreement indicator
        prob_values = [f.model_probability for f in successes]
        spread = max(prob_values) - min(prob_values) if len(prob_values) > 1 else 0.0
        agreement = max(0.0, 1.0 - spread * 2)  # spread of 0.5 = 0 agreement

        # If models disagree strongly, reduce confidence
        if spread > 0.15:
            agg_confidence = "LOW"

        # Merge reasoning, triggers, evidence from all models
        all_reasoning = [f.reasoning for f in successes if f.reasoning]
        all_triggers = []
        all_evidence = []
        seen_triggers: set[str] = set()
        for f in successes:
            for t in f.invalidation_triggers:
                if t.lower() not in seen_triggers:
                    seen_triggers.add(t.lower())
                    all_triggers.append(t)
            all_evidence.extend(f.key_evidence)

        result = EnsembleResult(
            model_probability=agg_prob,
            confidence_level=agg_confidence,
            individual_forecasts=list(forecasts),
            models_succeeded=len(successes),
            models_failed=len(failures),
            aggregation_method=agg_method_used,
            spread=round(spread, 4),
            agreement_score=round(agreement, 3),
            reasoning=" | ".join(all_reasoning[:3]),
            invalidation_triggers=all_triggers[:5],
            key_evidence=all_evidence[:8],
        )

        log.info(
            "ensemble.result",
            agg_prob=round(agg_prob, 3),
            confidence=agg_confidence,
            models_ok=len(successes),
            models_fail=len(failures),
            spread=round(spread, 3),
            method=agg_method_used,
        )
        return result

    def _aggregate(self, model_probs: list[tuple[str, float]]) -> tuple[float, str]:
        """Aggregate probabilities using configured method.

        When adaptive weights are injected via set_adaptive_weights()
        and aggregation is trimmed_mean, auto-switches to weighted
        aggregation so the learned weights are actually used.

        Args:
            model_probs: List of (model_name, probability) tuples.

        Returns:
            Tuple of (aggregated_probability, method_used).
        """
        if not model_probs:
            return 0.5, self._ensemble.aggregation

        probs = [p for _, p in model_probs]

        if len(probs) == 1:
            return probs[0], self._ensemble.aggregation

        method = self._ensemble.aggregation

        # Auto-switch to weighted when adaptive weights are available
        # and current method is trimmed_mean (the default). This fixes
        # the bug where learned weights were injected but never used.
        if method == "trimmed_mean" and self._external_weights:
            method = "weighted"
            log.info("ensemble.auto_switch_to_weighted")

        if method == "median":
            sorted_p = sorted(probs)
            mid = len(sorted_p) // 2
            if len(sorted_p) % 2 == 0:
                return (sorted_p[mid - 1] + sorted_p[mid]) / 2, method
            return sorted_p[mid], method

        elif method == "weighted":
            # Use adaptive (learned) weights if available, else config weights
            weights = self._external_weights or self._ensemble.weights
            total_weight = 0.0
            weighted_sum = 0.0
            for model_name, p in model_probs:
                w = weights.get(model_name, 1.0 / len(model_probs))
                weighted_sum += p * w
                total_weight += w
            prob = weighted_sum / total_weight if total_weight > 0 else 0.5
            return prob, method

        else:  # trimmed_mean (default)
            if len(probs) <= 2:
                return sum(probs) / len(probs), method
            sorted_p = sorted(probs)
            trim = max(1, int(len(sorted_p) * self._ensemble.trim_fraction))
            trimmed = sorted_p[trim:-trim] if trim < len(sorted_p) // 2 else sorted_p
            if not trimmed:
                trimmed = sorted_p
            return sum(trimmed) / len(trimmed), method
