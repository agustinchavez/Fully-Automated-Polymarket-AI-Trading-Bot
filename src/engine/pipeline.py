"""Pipeline runner — executes the research-to-trade pipeline stages.

Extracted from TradingEngine to reduce god-class complexity.
Receives dependencies via constructor injection.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any

from src.config import BotConfig
from src.observability.circuit_breaker import circuit_breakers
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.analytics.adaptive_weights import AdaptiveModelWeighter
    from src.analytics.calibration_feedback import CalibrationFeedbackLoop
    from src.analytics.regime_detector import RegimeState
    from src.analytics.smart_entry import SmartEntryCalculator
    from src.analytics.wallet_scanner import WalletScanner
    from src.connectors.ws_feed import WebSocketFeed
    from src.engine.loop import PipelineContext
    from src.execution.exit_finalizer import ExitFinalizer
    from src.execution.fill_tracker import FillTracker
    from src.execution.plan_controller import PlanController
    from src.forecast.specialist_router import SpecialistRouter
    from src.policy.drawdown import DrawdownManager
    from src.policy.portfolio_risk import PositionSnapshot
    from src.storage.audit import AuditTrail
    from src.storage.database import Database

log = get_logger(__name__)


class PipelineRunner:
    """Executes pipeline stages on a PipelineContext.

    Extracted from TradingEngine to reduce god-class complexity.
    Receives dependencies via constructor injection.
    """

    def __init__(
        self,
        config: BotConfig,
        db: Database | None,
        audit: AuditTrail | None,
        drawdown: DrawdownManager,
        calibration_loop: CalibrationFeedbackLoop,
        adaptive_weighter: AdaptiveModelWeighter,
        smart_entry: SmartEntryCalculator,
        specialist_router: SpecialistRouter | None,
        fill_tracker: FillTracker | None,
        plan_controller: PlanController | None,
        exit_finalizer: ExitFinalizer | None,
        current_regime: RegimeState | None,
        ws_feed: WebSocketFeed,
        wallet_scanner: WalletScanner,
        positions: list[PositionSnapshot],
        latest_scan_result: Any,
    ) -> None:
        self.config = config
        self._db = db
        self._audit = audit
        self.drawdown = drawdown
        self._calibration_loop = calibration_loop
        self._adaptive_weighter = adaptive_weighter
        self._smart_entry = smart_entry
        self._specialist_router = specialist_router
        self._fill_tracker = fill_tracker
        self._plan_controller = plan_controller
        self._exit_finalizer = exit_finalizer
        self._current_regime = current_regime
        self._ws_feed = ws_feed
        self._wallet_scanner = wallet_scanner
        self._positions = positions
        self._latest_scan_result = latest_scan_result

        # Singletons for external-API modules — instantiated once, reuse cache/interval
        self._uma_monitor: Any = None
        if config.uma.enabled:
            try:
                from src.analytics.uma_monitor import UMAMonitor
                self._uma_monitor = UMAMonitor(
                    refresh_interval_mins=config.uma.refresh_interval_mins,
                )
            except Exception:
                log.debug("engine.uma_monitor_init_skipped")

        self._event_calendar: Any = None
        if config.calendar.enabled:
            try:
                from src.analytics.event_calendar import EventCalendar
                self._event_calendar = EventCalendar(
                    refresh_interval_hours=config.calendar.refresh_interval_hours,
                    lookahead_days=config.calendar.lookahead_days,
                )
            except Exception:
                log.debug("engine.event_calendar_init_skipped")

        # Research infrastructure — singleton, reused across all market evals.
        # Previously recreated on every stage_research() call (561ms overhead).
        try:
            from src.connectors.web_search import create_search_provider
            from src.research.source_fetcher import SourceFetcher
            from src.research.evidence_extractor import EvidenceExtractor

            self._search_provider = create_search_provider(
                config.research.search_provider
            )
            self._source_fetcher = SourceFetcher(
                self._search_provider, config.research,
                db_path=config.storage.sqlite_path,
                auto_weight_sources=config.continuous_learning.auto_weight_sources,
            )
            self._evidence_extractor = EvidenceExtractor(config.forecasting)
        except Exception:
            log.debug("engine.research_infra_init_skipped")
            self._search_provider = None
            self._source_fetcher = None
            self._evidence_extractor = None

    async def close(self) -> None:
        """Close shared research infrastructure — called at engine shutdown."""
        if self._source_fetcher is not None:
            try:
                await self._source_fetcher.close()
            except Exception:
                log.warning("engine.source_fetcher_close_error", exc_info=True)
        if self._search_provider is not None:
            try:
                await self._search_provider.close()
            except Exception:
                log.warning("engine.search_provider_close_error", exc_info=True)

    # ── Pipeline Stage Methods ────────────────────────────────────────

    def stage_classify(self, ctx: PipelineContext) -> None:
        """Stage 0: Classify the market."""
        from src.engine.market_classifier import classify_and_log
        ctx.classification = classify_and_log(ctx.market)
        log.info(
            "engine.pipeline_start",
            market_id=ctx.market_id,
            question=ctx.question[:80],
            market_type=ctx.market.market_type,
            category=ctx.classification.category,
            subcategory=ctx.classification.subcategory,
            researchability=ctx.classification.researchability,
        )

    async def stage_research(self, ctx: PipelineContext) -> bool:
        """Stage 1: Research. Returns False if research failed and pipeline should abort."""
        cb = circuit_breakers.get("research")
        if not cb.allow_request():
            log.warning(
                "engine.research_circuit_open",
                market_id=ctx.market_id,
                retry_after=cb.time_until_retry(),
            )
            self._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                reason="Research circuit breaker open",
                                classification=ctx.classification)
            return False

        from src.research.query_builder import build_queries

        # Use shared singleton instances (initialised in __init__)
        source_fetcher = self._source_fetcher
        extractor = self._evidence_extractor

        if source_fetcher is None or extractor is None:
            # Lazy fallback if singleton init failed
            from src.connectors.web_search import create_search_provider
            from src.research.source_fetcher import SourceFetcher
            from src.research.evidence_extractor import EvidenceExtractor

            search_provider = create_search_provider(self.config.research.search_provider)
            source_fetcher = SourceFetcher(
                search_provider, self.config.research,
                db_path=self.config.storage.sqlite_path,
                auto_weight_sources=self.config.continuous_learning.auto_weight_sources,
            )
            extractor = EvidenceExtractor(self.config.forecasting)

        try:
            max_q = ctx.classification.recommended_queries
            queries = build_queries(
                ctx.market, max_queries=max_q,
                category=ctx.classification.category,
                researchability=ctx.classification.researchability,
            )
            ctx.sources = await source_fetcher.fetch_sources(
                queries,
                market_type=ctx.classification.category or ctx.market.market_type,
                max_sources=self.config.research.max_sources,
                market_question=ctx.question,
            )
            ctx.evidence = await extractor.extract(
                market_id=ctx.market_id, question=ctx.question,
                sources=ctx.sources, market_type=ctx.market.market_type,
            )
            cb.record_success()
        except Exception as e:
            cb.record_failure()
            log.error("engine.research_failed", market_id=ctx.market_id, error=str(e))
            self._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                reason=f"Research failed: {e}",
                                classification=ctx.classification)
            return False

        # Build signal stack from research sources + conviction + calendar
        try:
            from src.research.signal_aggregator import build_signal_stack
            poly_price = (
                ctx.features.implied_probability
                if ctx.features else 0.5
            )
            # Gather conviction signals from latest wallet scan
            conviction_signals = None
            if (self.config.wallet_scanner.enabled
                    and self.config.wallet_scanner.whale_in_prompt
                    and self._latest_scan_result
                    and hasattr(self._latest_scan_result, "conviction_signals")):
                conviction_signals = self._latest_scan_result.conviction_signals

            # Gather calendar events (uses shared singleton)
            calendar_events = None
            if self.config.calendar.enabled and self._event_calendar is not None:
                try:
                    await self._event_calendar.refresh()
                    cat = ctx.classification.category if ctx.classification else ""
                    calendar_events = self._event_calendar.get_events_for_market(ctx.question, cat) or None
                except Exception:
                    log.debug("engine.calendar_events_skipped", market_id=ctx.market_id)

            # Build lightweight microstructure signals from ws_feed
            micro_signals = None
            try:
                token_id = ""
                for t in getattr(ctx.market, "tokens", []):
                    if getattr(t, "outcome", "").lower() == "yes":
                        token_id = t.token_id
                        break
                if not token_id and getattr(ctx.market, "tokens", []):
                    token_id = ctx.market.tokens[0].token_id
                if token_id and self._ws_feed:
                    from src.connectors.microstructure import MicrostructureSignals, FlowImbalance
                    last_tick = self._ws_feed.get_last_price(token_id)
                    twap = self._ws_feed.get_twap(token_id, window_hours=2.0)
                    if last_tick and twap and twap > 0:
                        ms = MicrostructureSignals(token_id=token_id)
                        ms.vwap = twap
                        ms.vwap_divergence = last_tick.mid - twap
                        ms.vwap_divergence_pct = ms.vwap_divergence / twap
                        micro_signals = ms
            except Exception:
                pass  # microstructure signals are best-effort

            ctx._signal_stack = build_signal_stack(
                ctx.sources, poly_price,
                micro_signals=micro_signals,
                conviction_signals=conviction_signals,
                calendar_events=calendar_events,
            )
        except Exception:
            log.debug("engine.signal_stack_build_skipped", market_id=ctx.market_id)

        log.info(
            "engine.research_done", market_id=ctx.market_id,
            sources=len(ctx.sources), bullets=len(ctx.evidence.bullets),
            quality=round(ctx.evidence.quality_score, 3),
        )
        return True

    async def stage_forecast(self, ctx: PipelineContext) -> bool:
        """Stage 3: Run ensemble or single-model forecast. Returns False on failure."""
        cb = circuit_breakers.get("forecast")
        if not cb.allow_request():
            log.warning(
                "engine.forecast_circuit_open",
                market_id=ctx.market_id,
                retry_after=cb.time_until_retry(),
            )
            return False

        try:
            return await self._run_forecast(ctx)
        except Exception as e:
            cb.record_failure()
            log.error("engine.forecast_failed", market_id=ctx.market_id, error=str(e))
            return False

    async def _run_forecast(self, ctx: PipelineContext) -> bool:
        """Inner forecast logic — separated for circuit breaker wrapping."""
        cb = circuit_breakers.get("forecast")

        # ── Phase 4: Specialist routing ─────────────────────────────
        if self.config.specialists.enabled and self._specialist_router:
            specialist_result = await self._specialist_router.route(
                ctx.market, ctx.features, ctx.classification,
            )
            if specialist_result is not None:
                if specialist_result.bypasses_llm:
                    # Bypass mode: specialist provides complete forecast
                    from src.forecast.specialists.base import BaseSpecialist
                    from src.forecast.llm_forecaster import ForecastResult
                    ens_result = BaseSpecialist.to_ensemble_result(specialist_result)
                    ctx.forecast = ForecastResult(
                        market_id=ctx.market_id,
                        question=ctx.question,
                        market_type=ctx.market.market_type,
                        resolution_source=ctx.market.resolution_source,
                        implied_probability=ctx.features.implied_probability,
                        model_probability=ens_result.model_probability,
                        edge=ens_result.model_probability - ctx.features.implied_probability,
                        confidence_level=ens_result.confidence_level,
                        evidence=ens_result.key_evidence,
                        invalidation_triggers=ens_result.invalidation_triggers,
                        reasoning=ens_result.reasoning,
                        evidence_quality=specialist_result.evidence_quality,
                        num_sources=1,
                        raw_llm_response={
                            "specialist": True,
                            "specialist_name": specialist_result.specialist_name,
                            "specialist_metadata": specialist_result.specialist_metadata,
                            "spread": 0.0,
                            "agreement": 1.0,
                        },
                    )
                    ctx._specialist_used = specialist_result.specialist_name
                    log.info(
                        "engine.specialist_bypass",
                        specialist=specialist_result.specialist_name,
                        market_id=ctx.market_id,
                        probability=round(specialist_result.probability, 3),
                    )
                    cb.record_success()
                    return True
                else:
                    # Augment mode: inject as base rate, continue to ensemble
                    from src.forecast.specialists.base import BaseSpecialist
                    ctx._specialist_base_rate = BaseSpecialist.to_base_rate_match(
                        specialist_result,
                    )
                    ctx._specialist_used = specialist_result.specialist_name
                    log.info(
                        "engine.specialist_augment",
                        specialist=specialist_result.specialist_name,
                        market_id=ctx.market_id,
                        base_rate=round(specialist_result.probability, 3),
                    )

        if self.config.ensemble.enabled:
            from src.forecast.ensemble import EnsembleForecaster
            from src.forecast.model_router import select_tier

            # Select model tier based on opportunity characteristics
            ens_cfg = self.config.ensemble
            tier_used = None
            if self.config.model_tiers.enabled:
                rough_edge = abs(ctx.features.implied_probability - 0.5)
                tier = select_tier(
                    ctx.features, self.config.model_tiers, rough_edge,
                )
                log.info(
                    "engine.tier_selected",
                    market_id=ctx.market_id,
                    tier=tier.tier,
                    models=tier.models,
                    reason=tier.reason,
                )
                tier_used = tier.tier
                if tier.models != ens_cfg.models:
                    ens_cfg = ens_cfg.model_copy(
                        update={"models": tier.models},
                    )

            ens_forecaster = EnsembleForecaster(
                ens_cfg, self.config.forecasting,
            )
            # Inject learned adaptive weights if available
            try:
                if self._db:
                    cat = ctx.classification.category if ctx.classification else "UNKNOWN"
                    adaptive_result = self._adaptive_weighter.get_weights(
                        self._db.conn, cat,
                    )
                    if adaptive_result.data_available:
                        ens_forecaster.set_adaptive_weights(adaptive_result.weights)
                        log.info(
                            "engine.adaptive_weights_injected",
                            category=cat,
                            blend=round(adaptive_result.blend_factor, 3),
                            weights={k: round(v, 3)
                                     for k, v in adaptive_result.weights.items()},
                        )
            except Exception as e:
                log.warning("engine.adaptive_weights_inject_error", error=str(e))

            # Phase 2: Base rate lookup (specialist augment overrides)
            base_rate_info = None
            if hasattr(ctx, "_specialist_base_rate") and ctx._specialist_base_rate:
                base_rate_info = ctx._specialist_base_rate
                ctx._base_rate_value = base_rate_info.base_rate
                log.info(
                    "engine.specialist_base_rate_injected",
                    market_id=ctx.market_id,
                    base_rate=base_rate_info.base_rate,
                    source=base_rate_info.source,
                )
            elif self.config.forecasting.base_rate_enabled:
                try:
                    from src.forecast.base_rates import BaseRateRegistry
                    br_registry = BaseRateRegistry()
                    cat = ctx.classification.category if ctx.classification else "UNKNOWN"
                    base_rate_info = br_registry.match(ctx.question, category=cat)
                    if base_rate_info:
                        ctx._base_rate_value = base_rate_info.base_rate
                        log.info(
                            "engine.base_rate_found",
                            market_id=ctx.market_id,
                            base_rate=base_rate_info.base_rate,
                            pattern=base_rate_info.pattern_description,
                        )
                except Exception as e:
                    log.warning("engine.base_rate_lookup_error", error=str(e))

            # Phase 2: Question decomposition
            if self.config.forecasting.decomposition_enabled:
                try:
                    from src.forecast.decomposer import (
                        QuestionDecomposer,
                        combine_sub_forecasts,
                        should_decompose,
                    )
                    mtype = ctx.market.market_type if ctx.market else ""
                    if should_decompose(ctx.question, mtype):
                        decomposer = QuestionDecomposer(self.config.forecasting)
                        decomp = await decomposer.decompose(ctx.question, mtype)
                        if decomp.sub_questions:
                            sub_probs = []
                            for sq in decomp.sub_questions:
                                import copy
                                sub_features = copy.copy(ctx.features)
                                sub_features.question = sq.text
                                sub_result = await ens_forecaster.forecast(
                                    features=sub_features,
                                    evidence=ctx.evidence,
                                    base_rate_info=base_rate_info,
                                    prompt_version=self.config.forecasting.prompt_version,
                                )
                                sub_probs.append(sub_result.model_probability)
                            ctx._decomposition_sub_probs = sub_probs
                            decomposed_prob = combine_sub_forecasts(decomp, sub_probs)
                            log.info(
                                "engine.decomposition_result",
                                market_id=ctx.market_id,
                                num_sub=len(decomp.sub_questions),
                                decomposed_prob=round(decomposed_prob, 3),
                            )
                except Exception as e:
                    log.warning("engine.decomposition_error", error=str(e))

            evidence_quality = ctx.evidence.quality_score if ctx.evidence else 0.5
            ens_result = await ens_forecaster.forecast(
                features=ctx.features,
                evidence=ctx.evidence,
                base_rate_info=base_rate_info,
                prompt_version=self.config.forecasting.prompt_version,
                signal_stack=getattr(ctx, "_signal_stack", None),
                evidence_quality=evidence_quality,
            )
            from src.forecast.llm_forecaster import ForecastResult
            ctx.forecast = ForecastResult(
                market_id=ctx.market_id,
                question=ctx.question,
                market_type=ctx.market.market_type,
                resolution_source=ctx.market.resolution_source,
                implied_probability=ctx.features.implied_probability,
                model_probability=ens_result.model_probability,
                edge=ens_result.model_probability - ctx.features.implied_probability,
                confidence_level=ens_result.confidence_level,
                evidence=ens_result.key_evidence,
                invalidation_triggers=ens_result.invalidation_triggers,
                reasoning=ens_result.reasoning,
                evidence_quality=ctx.evidence.quality_score,
                num_sources=ctx.evidence.num_sources,
                raw_llm_response={
                    "ensemble": True,
                    "models_succeeded": ens_result.models_succeeded,
                    "models_failed": ens_result.models_failed,
                    "spread": ens_result.spread,
                    "agreement": ens_result.agreement_score,
                    "aggregation": ens_result.aggregation_method,
                },
                model_forecasts={
                    f.model_name: f.model_probability
                    for f in ens_result.individual_forecasts
                },
            )
            # Apply low-evidence penalty
            if ctx.evidence.quality_score < self.config.forecasting.min_evidence_quality:
                penalty = self.config.forecasting.low_evidence_penalty
                old_prob = ctx.forecast.model_probability
                ctx.forecast.model_probability = old_prob * (1 - penalty) + 0.5 * penalty
                ctx.forecast.edge = ctx.forecast.model_probability - ctx.features.implied_probability
                log.info("engine.ensemble_low_evidence_penalty",
                         original=round(old_prob, 3),
                         adjusted=round(ctx.forecast.model_probability, 3))

            # Scout tier confidence gate: reject LOW confidence scout forecasts
            if tier_used == "scout" and ctx.forecast.confidence_level == "LOW":
                log.info(
                    "engine.scout_confidence_gate_rejected",
                    market_id=ctx.market_id,
                    confidence=ctx.forecast.confidence_level,
                )
                return False
        else:
            from src.forecast.llm_forecaster import LLMForecaster
            forecaster = LLMForecaster(self.config.forecasting)
            ctx.forecast = await forecaster.forecast(
                features=ctx.features, evidence=ctx.evidence,
                resolution_source=ctx.market.resolution_source,
            )

        log.info(
            "engine.forecast_done", market_id=ctx.market_id,
            implied=round(ctx.forecast.implied_probability, 3),
            model=round(ctx.forecast.model_probability, 3),
            edge=round(ctx.forecast.edge, 3),
            confidence=ctx.forecast.confidence_level,
        )
        cb.record_success()
        return True

    def stage_calibrate(self, ctx: PipelineContext) -> None:
        """Stage 3b: Apply probability calibration."""
        try:
            from src.forecast.calibrator import calibrate as apply_calibration
            ensemble_spread = 0.0
            if hasattr(ctx.forecast, "raw_llm_response") and isinstance(ctx.forecast.raw_llm_response, dict):
                ensemble_spread = ctx.forecast.raw_llm_response.get("spread", 0.0)
            cal_result = apply_calibration(
                raw_prob=ctx.forecast.model_probability,
                evidence_quality=ctx.evidence.quality_score,
                num_contradictions=(
                    len(ctx.evidence.contradictions)
                    if hasattr(ctx.evidence, "contradictions") else 0
                ),
                method=self.config.forecasting.calibration_method,
                low_evidence_penalty=self.config.forecasting.low_evidence_penalty,
                ensemble_spread=ensemble_spread,
            )
            if abs(cal_result.calibrated_probability - ctx.forecast.model_probability) > 0.005:
                log.info(
                    "engine.calibration_applied",
                    market_id=ctx.market_id,
                    raw=round(ctx.forecast.model_probability, 4),
                    calibrated=round(cal_result.calibrated_probability, 4),
                    adjustments=cal_result.adjustments,
                )
                ctx.forecast.model_probability = cal_result.calibrated_probability
                ctx.forecast.edge = ctx.forecast.model_probability - ctx.forecast.implied_probability
        except Exception as e:
            log.warning("engine.calibration_apply_error", error=str(e))

    def stage_edge_calc(self, ctx: PipelineContext) -> None:
        """Stage 4: Edge calculation + whale/smart-money adjustment.

        Fixes applied:
          - Match by market_slug OR condition_id (not market_id)
          - Match direction BULLISH/BEARISH (not BUY/SELL)
          - Whale-edge convergence: when whale signal agrees with model edge,
            use a lower min_edge threshold for higher conviction trades
        """
        from src.policy.edge_calc import calculate_edge

        # TWAP reference price: use time-weighted avg instead of spot if available
        implied_for_edge = ctx.forecast.implied_probability
        if self.config.risk.use_twap_edge:
            try:
                token_id = ""
                for t in getattr(ctx.market, "tokens", []):
                    if getattr(t, "outcome", "").lower() == "yes":
                        token_id = t.token_id
                        break
                if not token_id and getattr(ctx.market, "tokens", []):
                    token_id = ctx.market.tokens[0].token_id
                if token_id:
                    twap = self._ws_feed.get_twap(
                        token_id, window_hours=self.config.risk.twap_window_hours,
                    )
                    if twap is not None:
                        divergence = abs(twap - implied_for_edge)
                        max_div = getattr(self.config.risk, "twap_max_divergence", 0.08)
                        if divergence <= max_div:
                            implied_for_edge = twap
                            log.info(
                                "engine.twap_reference",
                                market_id=ctx.market_id,
                                spot=round(ctx.forecast.implied_probability, 4),
                                twap=round(twap, 4),
                            )
            except Exception:
                pass  # fallback to spot price

        ctx.edge_result = calculate_edge(
            implied_prob=implied_for_edge,
            model_prob=ctx.forecast.model_probability,
            transaction_fee_pct=self.config.risk.transaction_fee_pct,
            gas_cost_usd=self.config.risk.gas_cost_usd,
            holding_hours=ctx.features.hours_to_resolution,
            use_probability_space_costs=self.config.risk.use_probability_space_costs,
        )

        # Dual edge: if sportsbook consensus available, use max(model_edge, book_edge * 0.7)
        signal_stack = getattr(ctx, "_signal_stack", None)
        book_consensus = getattr(signal_stack, "sportsbook_consensus", None) if signal_stack else None
        if book_consensus is not None:
            poly_price = implied_for_edge
            book_edge = abs(book_consensus - poly_price)
            book_edge_discounted = book_edge * 0.7
            model_edge = ctx.edge_result.abs_net_edge
            if book_edge_discounted > model_edge:
                # Re-calc edge using sportsbook-anchored probability
                book_model_prob = book_consensus
                ctx.edge_result = calculate_edge(
                    implied_prob=implied_for_edge,
                    model_prob=book_model_prob,
                    transaction_fee_pct=self.config.risk.transaction_fee_pct,
                    gas_cost_usd=self.config.risk.gas_cost_usd,
                    holding_hours=ctx.features.hours_to_resolution,
                    use_probability_space_costs=self.config.risk.use_probability_space_costs,
                )
                log.info(
                    "engine.sportsbook_dual_edge",
                    market_id=ctx.market_id,
                    model_edge=round(model_edge, 4),
                    book_edge=round(book_edge, 4),
                    book_edge_discounted=round(book_edge_discounted, 4),
                    used="sportsbook",
                )

        # Track whale convergence for min_edge override later
        ctx_whale_converged = False

        # Whale / Smart-Money Edge Adjustment
        if (self.config.wallet_scanner.enabled
                and self._latest_scan_result
                and hasattr(self._latest_scan_result, "conviction_signals")):
            whale_cfg = self.config.wallet_scanner
            # Phase 7: Use enhanced thresholds when quality scoring enabled
            if whale_cfg.whale_quality_scoring_enabled:
                _whale_boost = whale_cfg.enhanced_conviction_edge_boost
                _whale_penalty = whale_cfg.enhanced_conviction_edge_penalty
            else:
                _whale_boost = whale_cfg.conviction_edge_boost
                _whale_penalty = whale_cfg.conviction_edge_penalty
            # Double-counting fix: when whale data is in the LLM prompt,
            # cap the numeric edge adjustment to avoid double-counting
            if whale_cfg.whale_in_prompt:
                _whale_boost = min(_whale_boost, 0.02)
                _whale_penalty = min(_whale_penalty, 0.02)
            market_slug = getattr(ctx.market, "slug", "") or ""
            market_cid = getattr(ctx.market, "condition_id", "") or ""

            for sig in self._latest_scan_result.conviction_signals:
                # Match by slug, condition_id, or title substring
                sig_slug = getattr(sig, "market_slug", "") or ""
                sig_cid = getattr(sig, "condition_id", "") or ""
                matched = (
                    (sig_slug and market_slug and sig_slug == market_slug)
                    or (sig_cid and market_cid and sig_cid == market_cid)
                )
                if not matched:
                    continue

                # Direction matching: BULLISH→BUY_YES, BEARISH→BUY_NO
                whale_agrees = (
                    (sig.direction == "BULLISH" and ctx.edge_result.direction == "BUY_YES")
                    or (sig.direction == "BEARISH" and ctx.edge_result.direction == "BUY_NO")
                )
                if whale_agrees:
                    boost = _whale_boost
                    # Scale boost by conviction strength
                    strength_mult = (
                        1.5 if sig.signal_strength == "STRONG"
                        else 1.0 if sig.signal_strength == "MODERATE"
                        else 0.6
                    )
                    scaled_boost = boost * strength_mult
                    ctx.edge_result = calculate_edge(
                        implied_prob=ctx.forecast.implied_probability,
                        model_prob=(
                            min(0.99, ctx.forecast.model_probability + scaled_boost)
                            if ctx.edge_result.direction == "BUY_YES"
                            else max(0.01, ctx.forecast.model_probability - scaled_boost)
                        ),
                        transaction_fee_pct=self.config.risk.transaction_fee_pct,
                        gas_cost_usd=self.config.risk.gas_cost_usd,
                        holding_hours=ctx.features.hours_to_resolution,
                        use_probability_space_costs=self.config.risk.use_probability_space_costs,
                    )
                    ctx.edge_result.pre_whale_probability = ctx.forecast.model_probability
                    ctx.edge_result.whale_adjustment = scaled_boost
                    ctx_whale_converged = True
                    ctx.whale_converged = True
                    log.info("engine.whale_edge_boost", market_id=ctx.market_id,
                             boost=round(scaled_boost, 4),
                             strength=sig.signal_strength,
                             whale_count=sig.whale_count,
                             new_edge=round(ctx.edge_result.abs_net_edge, 4))
                else:
                    penalty = _whale_penalty
                    ctx.edge_result = calculate_edge(
                        implied_prob=ctx.forecast.implied_probability,
                        model_prob=(
                            max(0.01, ctx.forecast.model_probability - penalty)
                            if ctx.edge_result.direction == "BUY_YES"
                            else min(0.99, ctx.forecast.model_probability + penalty)
                        ),
                        transaction_fee_pct=self.config.risk.transaction_fee_pct,
                        gas_cost_usd=self.config.risk.gas_cost_usd,
                        holding_hours=ctx.features.hours_to_resolution,
                        use_probability_space_costs=self.config.risk.use_probability_space_costs,
                    )
                    ctx.edge_result.pre_whale_probability = ctx.forecast.model_probability
                    ctx.edge_result.whale_adjustment = -penalty
                    log.info("engine.whale_edge_penalty", market_id=ctx.market_id,
                             penalty=penalty, new_edge=round(ctx.edge_result.abs_net_edge, 4))
                break  # only apply first matching signal

        # Determine if we have edge — use lower threshold when whales agree
        min_edge = self.config.risk.min_edge
        if ctx_whale_converged:
            min_edge = self.config.wallet_scanner.whale_convergence_min_edge
            log.info("engine.whale_convergence",
                     market_id=ctx.market_id,
                     normal_min_edge=self.config.risk.min_edge,
                     whale_min_edge=min_edge,
                     edge=round(ctx.edge_result.abs_net_edge, 4))

        edge_for_threshold = ctx.edge_result.abs_net_edge
        if ctx.edge_result.effective_edge is not None:
            edge_for_threshold = ctx.edge_result.effective_edge
        ctx.has_edge = (
            ctx.edge_result.is_positive
            and edge_for_threshold >= min_edge
        )

    def stage_uncertainty_adjustment(self, ctx: PipelineContext) -> None:
        """Stage 4b: Apply edge uncertainty penalty (if enabled)."""
        if not self.config.risk.uncertainty_enabled:
            return

        from src.policy.edge_uncertainty import compute_and_adjust, UncertaintyInputs

        # Gather ensemble spread from raw_llm_response
        ensemble_spread = 0.0
        if (ctx.forecast
                and hasattr(ctx.forecast, "raw_llm_response")
                and isinstance(ctx.forecast.raw_llm_response, dict)):
            ensemble_spread = ctx.forecast.raw_llm_response.get("spread", 0.0)

        evidence_quality = ctx.features.evidence_quality if ctx.features else 0.5
        base_rate = getattr(ctx, "_base_rate_value", 0.5)
        sub_probs = getattr(ctx, "_decomposition_sub_probs", [])

        inputs = UncertaintyInputs(
            ensemble_spread=ensemble_spread,
            evidence_quality=evidence_quality,
            base_rate=base_rate,
            model_probability=ctx.forecast.model_probability,
            decomposition_sub_probs=sub_probs,
        )

        result = compute_and_adjust(
            inputs=inputs,
            raw_edge=ctx.edge_result.abs_net_edge,
            penalty_factor=self.config.risk.uncertainty_penalty_factor,
        )

        ctx.edge_result.effective_edge = result.effective_edge
        ctx._uncertainty_result = result

        log.info(
            "engine.uncertainty_adjustment",
            market_id=ctx.market_id,
            uncertainty=round(result.uncertainty_score, 3),
            raw_edge=round(result.raw_edge, 4),
            effective_edge=round(result.effective_edge, 4),
        )

    def stage_risk_checks(self, ctx: PipelineContext) -> None:
        """Stage 5: Risk limit checks."""
        from src.policy.risk_limits import check_risk_limits
        daily_pnl = self._db.get_daily_pnl() if self._db else 0.0
        open_positions = self._db.get_open_positions_count() if self._db else 0

        # When whales agree with our model, use a lower min_edge threshold
        whale_min_edge = None
        if ctx.whale_converged:
            whale_min_edge = self.config.wallet_scanner.whale_convergence_min_edge

        # Phase 4: When a specialist handled this market, bypass its
        # category restriction so the market type check passes.
        restricted = self.config.scanning.restricted_types or None
        specialist_used = getattr(ctx, "_specialist_used", None)
        if specialist_used and restricted and ctx.classification:
            restricted = [
                t for t in restricted
                if t != ctx.classification.category
            ] or None

        ctx.risk_result = check_risk_limits(
            edge=ctx.edge_result, features=ctx.features,
            risk_config=self.config.risk,
            forecast_config=self.config.forecasting,
            current_open_positions=open_positions,
            daily_pnl=daily_pnl,
            market_type=ctx.market.market_type,
            allowed_types=self.config.scanning.preferred_types or None,
            restricted_types=restricted,
            drawdown_state=self.drawdown.state,
            confidence_level=ctx.forecast.confidence_level if ctx.forecast else "LOW",
            min_edge_override=whale_min_edge,
        )

    async def stage_uma_check(self, ctx: PipelineContext) -> None:
        """Block markets with active UMA Oracle disputes (high tail risk)."""
        if self._uma_monitor is None:
            return
        if not ctx.risk_result or not ctx.risk_result.allowed:
            return

        try:
            await self._uma_monitor.refresh_disputes()
            cid = getattr(ctx.market, "condition_id", "") or ""
            if cid and self._uma_monitor.is_disputed(cid):
                ctx.risk_result.allowed = False
                ctx.risk_result.violations.append("UMA_DISPUTE: active dispute on market")
                log.warning(
                    "engine.uma_dispute_blocked",
                    market_id=ctx.market_id,
                    condition_id=cid[:16],
                )
        except Exception as e:
            log.debug("engine.uma_check_skipped", error=str(e))

    def stage_persist_forecast(self, ctx: PipelineContext) -> None:
        """Persist forecast and market records to DB."""
        if not self._db:
            return
        from src.storage.models import ForecastRecord, MarketRecord
        self._db.upsert_market(MarketRecord(
            id=ctx.market_id, condition_id=ctx.market.condition_id,
            question=ctx.question, market_type=ctx.market.market_type,
            category=ctx.market.category, volume=ctx.market.volume,
            liquidity=ctx.market.liquidity,
            end_date=ctx.market.end_date.isoformat() if ctx.market.end_date else "",
            resolution_source=ctx.market.resolution_source,
        ))
        self._db.insert_forecast(ForecastRecord(
            id=str(uuid.uuid4()), market_id=ctx.market_id,
            question=ctx.question, market_type=ctx.market.market_type,
            implied_probability=ctx.forecast.implied_probability,
            model_probability=ctx.forecast.model_probability,
            edge=ctx.forecast.edge,
            confidence_level=ctx.forecast.confidence_level,
            evidence_quality=ctx.evidence.quality_score,
            num_sources=ctx.evidence.num_sources,
            decision=ctx.risk_result.decision,
            reasoning=ctx.forecast.reasoning[:500],
            evidence_json=json.dumps(ctx.forecast.evidence[:5]),
            invalidation_triggers_json=json.dumps(ctx.forecast.invalidation_triggers),
            research_evidence_json=json.dumps({
                **ctx.evidence.to_dict(),
                "classification": ctx.classification.to_dict(),
            }),
        ))

    def stage_correlation_check(self, ctx: PipelineContext) -> None:
        """Check portfolio correlation before allowing entry."""
        if not self._positions or not ctx.risk_result.allowed:
            return
        from src.policy.portfolio_risk import check_correlation
        corr_ok, corr_reason = check_correlation(
            existing_positions=self._positions,
            new_question=ctx.question,
            new_category=ctx.classification.category if ctx.classification else "",
            new_event_slug=ctx.market.slug or "",
            similarity_threshold=self.config.portfolio.correlation_similarity_threshold,
        )
        if not corr_ok:
            ctx.risk_result.allowed = False
            ctx.risk_result.violations.append(f"Correlation: {corr_reason}")
            log.info("engine.correlation_blocked",
                     market_id=ctx.market_id, reason=corr_reason)

    def stage_var_gate(self, ctx: PipelineContext) -> None:
        """Check portfolio VaR limit before allowing entry."""
        if not self.config.portfolio.var_gate_enabled:
            return
        if not ctx.risk_result or not ctx.risk_result.allowed:
            return

        from src.policy.correlation import EventCorrelationScorer
        from src.policy.portfolio_risk import check_var_gate, PositionSnapshot
        from src.policy.position_sizer import calculate_position_size

        scorer = EventCorrelationScorer(self.config.portfolio)

        # Preliminary Kelly estimate for VaR sizing (replaces hardcoded 50.0)
        estimated_size = 50.0
        if ctx.edge_result and ctx.forecast:
            try:
                prelim = calculate_position_size(
                    edge=ctx.edge_result,
                    risk_config=self.config.risk,
                    confidence_level=getattr(ctx.forecast, "confidence_level", "LOW"),
                )
                estimated_size = max(1.0, prelim.stake_usd)
            except Exception:
                pass  # fall back to 50.0 on any error

        new_pos = PositionSnapshot(
            market_id=ctx.market_id,
            question=ctx.question,
            category=ctx.classification.category if ctx.classification else "",
            event_slug=getattr(ctx.market, "slug", "") or "",
            side=ctx.edge_result.direction.replace("BUY_", "") if ctx.edge_result else "YES",
            size_usd=estimated_size,
            entry_price=ctx.edge_result.implied_probability if ctx.edge_result else 0.5,
            current_price=ctx.edge_result.implied_probability if ctx.edge_result else 0.5,
        )

        allowed, reason, details = check_var_gate(
            positions=self._positions,
            new_position=new_pos,
            bankroll=self.config.risk.bankroll,
            max_var_pct=self.config.portfolio.max_portfolio_var_pct,
            correlation_scorer=scorer,
        )

        if not allowed:
            ctx.risk_result.allowed = False
            ctx.risk_result.violations.append(f"VAR_GATE: {reason}")
            log.info(
                "engine.var_gate_blocked",
                market_id=ctx.market_id,
                projected_var=details["projected_var"],
                limit=details["var_limit"],
            )

    def stage_position_sizing(self, ctx: PipelineContext) -> None:
        """Stage 6: Calculate position size. Sets ctx.position to None if too small."""
        from src.policy.position_sizer import calculate_position_size
        regime_kelly = (
            self._current_regime.kelly_multiplier
            if self._current_regime else 1.0
        )
        regime_size = (
            self._current_regime.size_multiplier
            if self._current_regime else 1.0
        )
        # Category-weighted stake multiplier
        category = (ctx.classification.category
                    if ctx.classification else ctx.market.category or "")
        cat_mults = getattr(self.config.risk, "category_stake_multipliers", {})
        cat_mult = cat_mults.get(category, 1.0)
        # Phase 3: Uncertainty multiplier for position sizing
        unc_mult = 1.0
        unc_result = getattr(ctx, "_uncertainty_result", None)
        if unc_result is not None:
            unc_mult = max(0.5, 1.0 - unc_result.uncertainty_score * 0.5)
        # Phase 2 signal confluence: reduce sizing when signals diverge
        conf_mult = 1.0
        signal_stack = getattr(ctx, "_signal_stack", None)
        if signal_stack is not None:
            conf_mult = getattr(signal_stack, "recommended_kelly_multiplier", 1.0)
        # Calendar: reduce size when high-impact event is within 24h
        cal_mult = 1.0
        if self.config.calendar.enabled and signal_stack is not None:
            cal_events = getattr(signal_stack, "calendar_events", [])
            for evt in cal_events:
                impact = getattr(evt, "impact", "")
                hours = getattr(evt, "hours_away", 999)
                if impact == "high" and hours < 24:
                    cal_mult = self.config.calendar.pre_event_size_reduction
                    log.info(
                        "engine.calendar_size_reduction",
                        market_id=ctx.market_id,
                        event=getattr(evt, "name", ""),
                        hours=round(hours, 1),
                        multiplier=cal_mult,
                    )
                    break
        ctx.position = calculate_position_size(
            edge=ctx.edge_result, risk_config=self.config.risk,
            confidence_level=ctx.forecast.confidence_level,
            drawdown_multiplier=self.drawdown.state.kelly_multiplier,
            timeline_multiplier=ctx.features.time_decay_multiplier,
            price_volatility=ctx.features.price_volatility,
            regime_multiplier=regime_kelly * regime_size * cal_mult,
            category_multiplier=cat_mult,
            uncertainty_multiplier=unc_mult,
            confluence_multiplier=conf_mult,
        )
        if ctx.position.stake_usd < 1.0:
            log.info("engine.stake_too_small", market_id=ctx.market_id,
                     stake=ctx.position.stake_usd)
            self._log_candidate(
                ctx.cycle_id, ctx.market, forecast=ctx.forecast,
                evidence=ctx.evidence, edge_result=ctx.edge_result,
                decision="NO TRADE", reason="Stake too small",
                stake=ctx.position.stake_usd,
                classification=ctx.classification,
            )
            ctx.position = None

    async def stage_execute_order(self, ctx: PipelineContext) -> None:
        """Stage 7: Build and route orders."""
        from src.execution.order_builder import build_order
        from src.execution.order_router import OrderRouter
        from src.connectors.polymarket_clob import CLOBClient

        market = ctx.market
        forecast = ctx.forecast
        edge_result = ctx.edge_result
        position = ctx.position

        yes_tokens = [t for t in market.tokens if t.outcome.lower() == "yes"]
        no_tokens = [t for t in market.tokens if t.outcome.lower() == "no"]
        if not yes_tokens and not no_tokens and len(market.tokens) >= 2:
            yes_tokens = [market.tokens[0]]
            no_tokens = [market.tokens[1]]
        elif not yes_tokens and market.tokens:
            yes_tokens = [market.tokens[0]]

        if edge_result.direction == "BUY_YES" and yes_tokens:
            token_id = yes_tokens[0].token_id
            implied_price = yes_tokens[0].price or forecast.implied_probability
        elif edge_result.direction == "BUY_NO" and no_tokens:
            token_id = no_tokens[0].token_id
            implied_price = no_tokens[0].price or (1 - forecast.implied_probability)
        else:
            token_id = (yes_tokens[0].token_id if yes_tokens
                        else (no_tokens[0].token_id if no_tokens else ""))
            implied_price = forecast.implied_probability

        if not token_id:
            log.warning("engine.no_token_id", market_id=ctx.market_id)
            self._log_candidate(
                ctx.cycle_id, market, forecast=forecast, evidence=ctx.evidence,
                edge_result=edge_result, decision="NO TRADE",
                reason="No token ID available",
                classification=ctx.classification,
            )
            return

        # Smart Entry: Calculate optimal entry price
        execution_strategy = "simple"
        try:
            regime_patience = (
                self._current_regime.entry_patience
                if self._current_regime else 1.0
            )
            entry_plan = self._smart_entry.calculate_entry(
                market_id=ctx.market_id,
                side=edge_result.direction,
                current_price=implied_price,
                fair_value=forecast.model_probability,
                edge=edge_result.abs_net_edge,
                spread=getattr(ctx.features, "spread", 0.0),
                hours_to_resolution=getattr(ctx.features, "hours_to_resolution", 720.0),
                regime_patience=regime_patience,
            )
            if entry_plan and entry_plan.recommended_price > 0:
                old_price = implied_price
                implied_price = entry_plan.recommended_price
                if entry_plan.recommended_strategy == "twap":
                    execution_strategy = "twap"
                log.info(
                    "engine.smart_entry", market_id=ctx.market_id,
                    old_price=round(old_price, 4),
                    new_price=round(implied_price, 4),
                    strategy=entry_plan.recommended_strategy,
                    improvement_bps=round(entry_plan.expected_improvement_bps, 1),
                )
        except Exception as e:
            log.warning("engine.smart_entry_error", error=str(e))

        # Phase 6: Patience window — wait for better price if enabled
        if self.config.execution.patience_window_enabled:
            try:
                _patience_clob = CLOBClient()
                try:
                    async def _get_price():
                        ob = _patience_clob.get_orderbook(token_id)
                        if asyncio.iscoroutine(ob):
                            ob = await ob
                        return ob.mid if hasattr(ob, "mid") and ob.mid > 0 else implied_price

                    patience_result = await self._smart_entry.patience_monitor(
                        market_id=ctx.market_id,
                        side=edge_result.direction,
                        min_edge=self.config.risk.min_edge,
                        current_edge=edge_result.abs_net_edge,
                        get_current_price=_get_price,
                        get_model_prob=lambda: forecast.model_probability,
                        max_wait_secs=self.config.execution.patience_window_max_secs,
                        check_interval_secs=self.config.execution.patience_check_interval_secs,
                        immediate_multiplier=self.config.execution.edge_immediate_multiplier,
                    )

                    if patience_result.action == "cancelled":
                        log.info(
                            "engine.patience_cancelled",
                            market_id=ctx.market_id,
                            reason=patience_result.reason,
                            wait=patience_result.wait_time_secs,
                        )
                        self._log_candidate(
                            ctx.cycle_id, market, forecast=forecast,
                            evidence=ctx.evidence, edge_result=edge_result,
                            decision="NO TRADE",
                            reason=f"Patience cancelled: {patience_result.reason}",
                            classification=ctx.classification,
                        )
                        return
                    elif patience_result.entry_price > 0:
                        implied_price = patience_result.entry_price
                        log.info(
                            "engine.patience_entered",
                            market_id=ctx.market_id,
                            action=patience_result.action,
                            wait=patience_result.wait_time_secs,
                            price=patience_result.entry_price,
                        )
                finally:
                    if hasattr(_patience_clob, "close"):
                        try:
                            close_result = _patience_clob.close()
                            if asyncio.iscoroutine(close_result):
                                await close_result
                        except Exception:
                            pass
            except Exception as e:
                log.warning("engine.patience_error", error=str(e))

        # Phase 6: Auto strategy selection based on liquidity + historical quality
        if self.config.execution.auto_strategy_selection_enabled:
            try:
                from src.execution.strategy_selector import ExecutionStrategySelector

                selector = ExecutionStrategySelector(
                    thin_depth_usd=self.config.execution.auto_strategy_thin_depth_usd,
                    large_order_pct=self.config.execution.auto_strategy_large_order_pct,
                    learning_enabled=self.config.execution.auto_strategy_learning_enabled,
                    min_samples=self.config.execution.auto_strategy_min_samples,
                )

                # Estimate depth from features or use default
                depth_usd = getattr(ctx.features, "depth_usd", 0.0)
                if depth_usd <= 0:
                    depth_usd = getattr(ctx.features, "volume_24h", 50000.0)

                # Get historical quality if learning is enabled
                historical_quality = None
                if (
                    self.config.execution.auto_strategy_learning_enabled
                    and hasattr(self, "_fill_tracker")
                    and self._fill_tracker is not None
                ):
                    historical_quality = self._fill_tracker.get_quality()

                recommendation = selector.select(
                    order_size_usd=position.stake_usd,
                    depth_usd=depth_usd,
                    historical_quality=historical_quality,
                )
                execution_strategy = recommendation.strategy
                log.info(
                    "engine.strategy_selected",
                    market_id=ctx.market_id,
                    strategy=recommendation.strategy,
                    reason=recommendation.reason,
                    confidence=recommendation.confidence,
                    depth_usd=recommendation.depth_usd,
                )
            except Exception as e:
                log.warning("engine.strategy_selector_error", error=str(e))

        orders = build_order(
            market_id=ctx.market_id, token_id=token_id,
            position=position, implied_price=implied_price,
            config=self.config.execution, execution_strategy=execution_strategy,
        )

        clob = CLOBClient()
        router = OrderRouter(clob, self.config.execution)
        ctx._order_statuses = []  # list[str]
        ctx._token_id = token_id
        try:
            # Phase 10E: Plan orchestration — sequential child submission
            if self._plan_controller and len(orders) > 1:
                await self._submit_with_plan(
                    ctx, orders, router, edge_result, position, token_id,
                    execution_strategy,
                )
            else:
                # Simple path: submit all orders immediately
                for order in orders:
                    order_result = await router.submit_order(order)
                    ctx._order_statuses.append(order_result.status)
                    log.info(
                        "engine.order_result", market_id=ctx.market_id,
                        order_id=order_result.order_id[:8],
                        status=order_result.status,
                        fill_price=order_result.fill_price,
                        fill_size=order_result.fill_size,
                        clob_order_id=order_result.clob_order_id[:8] if order_result.clob_order_id else "",
                    )
                    if self._db:
                        from src.storage.models import TradeRecord, PositionRecord, OrderRecord
                        self._record_order_result(
                            ctx, order, order_result, edge_result, position, token_id,
                        )
                    if order_result.status in ("simulated", "filled"):
                        ctx.result["trade_executed"] = True
                        self._ws_feed.subscribe(token_id)
                    elif order_result.status in ("submitted", "pending"):
                        # Order is on the book — position will be created on fill confirmation
                        ctx.result["trade_attempted"] = True
        finally:
            await clob.close()

    async def _submit_with_plan(
        self,
        ctx: PipelineContext,
        orders: list,
        router: Any,
        edge_result: Any,
        position: Any,
        token_id: str,
        execution_strategy: str,
    ) -> None:
        """Create an execution plan and submit only the first child order.

        Remaining children are submitted sequentially by the reconciliation
        loop as each child fills.
        """
        plan_metadata = None
        if execution_strategy == "twap":
            plan_metadata = {
                "min_child_interval_secs": self.config.execution.twap_interval_secs,
            }
        plan = self._plan_controller.create_plan(
            orders, execution_strategy, metadata=plan_metadata,
        )

        first_spec = self._plan_controller.get_first_child_spec(plan)
        order_result = await router.submit_order(first_spec)
        ctx._order_statuses.append(order_result.status)

        log.info(
            "engine.plan_first_child",
            plan_id=plan.plan_id[:8],
            order_id=order_result.order_id[:8],
            status=order_result.status,
            strategy=execution_strategy,
            total_children=plan.total_children,
        )

        if self._db:
            self._record_order_result(
                ctx, first_spec, order_result, edge_result, position, token_id,
                parent_plan_id=plan.plan_id, child_index=0,
            )

        if order_result.status in ("simulated", "filled"):
            ctx.result["trade_executed"] = True
            self._ws_feed.subscribe(token_id)
        elif order_result.status in ("submitted", "pending"):
            ctx.result["trade_attempted"] = True

        # Activate plan so reconciliation knows to advance children
        self._plan_controller.activate_plan(
            plan.plan_id, order_result.order_id,
        )

    def _record_order_result(
        self,
        ctx: PipelineContext,
        order: Any,
        order_result: Any,
        edge_result: Any,
        position: Any,
        token_id: str,
        parent_plan_id: str = "",
        child_index: int = 0,
    ) -> None:
        """Record order result to DB based on fill status.

        - simulated: instant fill (paper mode) → insert trade + upsert position
        - filled: confirmed fill (live) → insert trade + upsert position
        - submitted/pending: on the book → insert order only, no position yet
        - failed: log for audit → insert order with error
        """
        from src.storage.models import TradeRecord, PositionRecord, OrderRecord
        from src.execution.direction import parse_direction

        # Prefer canonical fields from OrderSpec if available, fallback to parse
        action_side = getattr(order, "action_side", "") or ""
        outcome_side = getattr(order, "outcome_side", "") or ""
        if not action_side or not outcome_side:
            action_side, outcome_side = parse_direction(edge_result.direction)

        if order_result.status in ("simulated", "filled"):
            # Confirmed fill — create trade record + position
            self._db.insert_trade(TradeRecord(
                id=str(uuid.uuid4()),
                order_id=order_result.order_id,
                market_id=ctx.market_id, token_id=token_id,
                side=edge_result.direction,
                price=order_result.fill_price,
                size=order_result.fill_size,
                stake_usd=position.stake_usd,
                status=order_result.status.upper(),
                dry_run=order_result.status == "simulated",
                action_side=action_side,
                outcome_side=outcome_side,
            ))
            self._db.upsert_position(PositionRecord(
                market_id=ctx.market_id, token_id=token_id,
                direction=edge_result.direction,
                entry_price=order_result.fill_price,
                size=order_result.fill_size,
                stake_usd=position.stake_usd,
                current_price=order_result.fill_price, pnl=0.0,
                action_side=action_side,
                outcome_side=outcome_side,
                question=ctx.question[:200] if ctx.question else "",
                market_type=getattr(ctx.market, "market_type", ""),
            ))
        elif order_result.status in ("submitted", "pending"):
            # Order on the book — track in open_orders, no position yet
            self._db.insert_order(OrderRecord(
                order_id=order_result.order_id,
                clob_order_id=order_result.clob_order_id,
                market_id=ctx.market_id,
                token_id=token_id,
                side=edge_result.direction,
                order_type=getattr(order, "order_type", "limit"),
                price=getattr(order, "price", 0.0),
                size=getattr(order, "size", 0.0),
                stake_usd=position.stake_usd,
                status=order_result.status,
                dry_run=False,
                ttl_secs=getattr(order, "ttl_secs", 0),
                action_side=action_side,
                outcome_side=outcome_side,
                parent_plan_id=parent_plan_id,
                child_index=child_index,
            ))
            log.info(
                "engine.order_awaiting_fill",
                market_id=ctx.market_id,
                order_id=order_result.order_id[:8],
                clob_order_id=order_result.clob_order_id[:8] if order_result.clob_order_id else "",
            )
        elif order_result.status == "failed":
            # Failed order — record for audit trail
            self._db.insert_order(OrderRecord(
                order_id=order_result.order_id,
                clob_order_id=order_result.clob_order_id,
                market_id=ctx.market_id,
                token_id=token_id,
                side=edge_result.direction,
                order_type=getattr(order, "order_type", "limit"),
                price=getattr(order, "price", 0.0),
                size=getattr(order, "size", 0.0),
                stake_usd=position.stake_usd,
                status="failed",
                dry_run=False,
                error=order_result.error,
                action_side=action_side,
                outcome_side=outcome_side,
            ))
            log.error(
                "engine.order_failed",
                market_id=ctx.market_id,
                order_id=order_result.order_id[:8],
                error=order_result.error,
            )

    def stage_audit_and_log(self, ctx: PipelineContext) -> None:
        """Stage 8: Audit trail + logging + adaptive weight recording."""
        order_statuses = getattr(ctx, "_order_statuses", [])
        token_id = getattr(ctx, "_token_id", "")

        if self._audit:
            self._audit.record_trade_decision(
                market_id=ctx.market_id, question=ctx.question,
                model_prob=ctx.forecast.model_probability,
                implied_prob=ctx.forecast.implied_probability,
                edge=ctx.forecast.edge,
                confidence=ctx.forecast.confidence_level,
                risk_result=ctx.risk_result.to_dict(),
                position_size=ctx.position.stake_usd if ctx.position else 0.0,
                order_id="",
                evidence_summary=ctx.evidence.summary[:200],
            )

        self._log_candidate(
            ctx.cycle_id, ctx.market, forecast=ctx.forecast,
            evidence=ctx.evidence, edge_result=ctx.edge_result,
            decision="TRADE", reason="All checks passed",
            stake=ctx.position.stake_usd if ctx.position else 0.0,
            order_status=order_statuses[0] if order_statuses else "",
            classification=ctx.classification,
        )
        if self._db:
            mode = (
                "\U0001f9ea Paper"
                if order_statuses and order_statuses[0] == "simulated"
                else "\U0001f4b0 Live"
            )
            self._db.insert_alert(
                "info",
                f'{mode} trade: {ctx.edge_result.direction} on '
                f'"{ctx.question[:60]}" '
                f"\u2014 stake ${ctx.position.stake_usd:.2f}, "
                f"edge {ctx.forecast.edge:+.3f}, "
                f"confidence {ctx.forecast.confidence_level}",
                "trade", ctx.market_id,
            )
        log.info(
            "engine.trade_executed", market_id=ctx.market_id,
            direction=ctx.edge_result.direction,
            stake=ctx.position.stake_usd if ctx.position else 0.0,
            edge=round(ctx.forecast.edge, 3), status=order_statuses,
        )

        # Record for adaptive weighting (model accuracy log)
        try:
            if self._db and hasattr(ctx.forecast, 'model_forecasts'):
                for model_name, prob in (ctx.forecast.model_forecasts or {}).items():
                    self._db.conn.execute("""
                        INSERT INTO model_forecast_log
                            (model_name, market_id, category, forecast_prob,
                             actual_outcome, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        model_name, ctx.market_id,
                        ctx.classification.category if ctx.classification else "UNKNOWN",
                        prob, None,
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    ))
                    self._db.conn.commit()
        except Exception as e:
            log.warning("engine.model_forecast_log_error", error=str(e))

    def _log_candidate(
        self, cycle_id: int, market: Any,
        forecast: Any = None, evidence: Any = None,
        edge_result: Any = None,
        decision: str = "SKIP", reason: str = "",
        stake: float = 0.0, order_status: str = "",
        classification: Any = None,
    ) -> None:
        if not self._db:
            return
        try:
            # Prefer rich classifier category over legacy market_type
            effective_type = (
                classification.category
                if classification
                and getattr(classification, "category", "UNKNOWN")
                not in ("UNKNOWN", "")
                else market.market_type
            )
            self._db.insert_candidate(
                cycle_id=cycle_id, market_id=market.id,
                question=market.question[:200],
                market_type=effective_type,
                implied_prob=(getattr(forecast, "implied_probability", market.best_bid)
                              if forecast else market.best_bid),
                model_prob=(getattr(forecast, "model_probability", 0.0)
                            if forecast else 0.0),
                edge=getattr(forecast, "edge", 0.0) if forecast else 0.0,
                evidence_quality=(getattr(evidence, "quality_score", 0.0)
                                  if evidence else 0.0),
                num_sources=(getattr(evidence, "num_sources", 0)
                             if evidence else 0),
                confidence=(getattr(forecast, "confidence_level", "")
                            if forecast else ""),
                decision=decision, decision_reasons=reason[:300],
                stake_usd=stake, order_status=order_status,
            )
        except Exception as e:
            log.warning("engine.log_candidate_error", error=str(e))

    def record_performance_log(
        self,
        pos: Any,
        exit_price: float,
        pnl: float,
        mkt_record: Any = None,
    ) -> None:
        """Write a closed position to the performance_log table for analytics.

        Gathers forecast data (probability, edge, confidence, evidence quality)
        from the forecasts table and computes holding duration.
        """
        if not self._db:
            return
        try:
            import datetime as _dt
            from src.storage.models import PerformanceLogRecord

            # Look up the most recent forecast for this market
            forecasts = self._db.get_forecasts(market_id=pos.market_id, limit=1)
            fc = forecasts[0] if forecasts else None

            # Compute holding hours
            holding_hours = 0.0
            try:
                opened = _dt.datetime.fromisoformat(
                    pos.opened_at.replace("Z", "+00:00")
                )
                now = _dt.datetime.now(_dt.timezone.utc)
                holding_hours = (now - opened).total_seconds() / 3600
            except Exception:
                pass

            # Determine category from market record or forecast
            category = "UNKNOWN"
            if mkt_record and getattr(mkt_record, "category", None):
                category = mkt_record.category
            elif fc and getattr(fc, "market_type", None):
                category = fc.market_type

            # Determine actual outcome from exit price (for resolved markets)
            actual_outcome = None
            if exit_price >= 0.98:
                actual_outcome = 1.0
            elif exit_price <= 0.02:
                actual_outcome = 0.0

            self._db.insert_performance_log(PerformanceLogRecord(
                market_id=pos.market_id,
                question=(
                    getattr(mkt_record, "question", "")
                    if mkt_record else getattr(pos, "question", "")
                ),
                category=category,
                forecast_prob=fc.model_probability if fc else 0.0,
                actual_outcome=actual_outcome,
                edge_at_entry=fc.edge if fc else 0.0,
                confidence=fc.confidence_level if fc else "LOW",
                evidence_quality=fc.evidence_quality if fc else 0.0,
                stake_usd=pos.stake_usd,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                holding_hours=round(holding_hours, 2),
            ))

            log.info(
                "engine.performance_log_recorded",
                market_id=pos.market_id[:8],
                pnl=pnl,
                holding_hours=round(holding_hours, 1),
                category=category,
            )

            # Backfill model_forecast_log with actual outcome so model
            # accuracy queries (IS NOT NULL) include resolved rows.
            if actual_outcome is not None:
                resolved_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
                self._db.conn.execute(
                    "UPDATE model_forecast_log "
                    "SET actual_outcome = ?, resolved_at = ? "
                    "WHERE market_id = ? AND actual_outcome IS NULL",
                    (actual_outcome, resolved_at, pos.market_id),
                )
                self._db.conn.commit()
                log.info(
                    "engine.model_forecast_log_resolved",
                    market_id=pos.market_id[:8],
                    actual_outcome=actual_outcome,
                )
        except Exception as e:
            log.warning("engine.performance_log_error", error=str(e))

    def maybe_run_post_mortem(self, market_id: str) -> None:
        """Run post-mortem analysis on a resolved market (Phase 8)."""
        cl_cfg = self.config.continuous_learning
        if not cl_cfg.post_mortem_enabled:
            return
        if not self._db:
            return
        try:
            from src.analytics.post_mortem import PostMortemAnalyzer

            conn = self._db._conn  # reuse existing connection
            analyzer = PostMortemAnalyzer(conn)
            analysis = analyzer.analyze_market(market_id)
            if analysis and analysis.was_confident_and_wrong:
                log.warning(
                    "engine.confident_wrong_trade",
                    market_id=market_id[:8],
                    forecast=analysis.forecast_prob,
                    outcome=analysis.actual_outcome,
                )
        except Exception as e:
            log.warning("engine.post_mortem_error", error=str(e))
