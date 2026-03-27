"""Continuous trading loop — the brain of the bot.

Runs on a configurable cycle (default 5 minutes):
  1. Discover & filter markets
  2. Build features for each market
  3. Research top candidates (evidence gathering)
  4. Forecast probabilities
  5. Calculate edges
  6. Check risk limits
  7. Size positions
  8. Route orders (paper or live)
  9. Monitor existing positions for exits

Between cycles:
  - Check drawdown state
  - Monitor position exits (stop-loss, resolution/100%, time-based)
  - Persist engine state to DB for dashboard
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config import BotConfig, load_config, is_live_trading_enabled
from src.policy.drawdown import DrawdownManager
from src.policy.portfolio_risk import PortfolioRiskManager, PositionSnapshot
from src.engine.market_filter import ResearchCache, filter_markets, FilterStats
from src.analytics.regime_detector import RegimeDetector, RegimeState
from src.analytics.calibration_feedback import CalibrationFeedbackLoop
from src.analytics.adaptive_weights import AdaptiveModelWeighter
from src.analytics.smart_entry import SmartEntryCalculator
from src.analytics.wallet_scanner import WalletScanner, save_scan_result
from src.connectors.ws_feed import WebSocketFeed, PriceTick
from src.observability.logger import get_logger
from src.observability.metrics import cost_tracker
from src.observability.circuit_breaker import circuit_breakers

log = get_logger(__name__)


@dataclass
class CycleResult:
    """Summary of one trading cycle."""
    cycle_id: int
    started_at: float
    ended_at: float = 0.0
    duration_secs: float = 0.0
    markets_scanned: int = 0
    markets_researched: int = 0
    edges_found: int = 0
    trades_attempted: int = 0
    trades_executed: int = 0
    errors: list[str] = field(default_factory=list)
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class PipelineContext:
    """Carries state through the processing pipeline stages."""
    market: Any
    cycle_id: int
    market_id: str = ""
    question: str = ""
    classification: Any = None
    sources: list[Any] = field(default_factory=list)
    evidence: Any = None
    features: Any = None
    forecast: Any = None
    edge_result: Any = None
    has_edge: bool = False
    risk_result: Any = None
    position: Any = None
    whale_converged: bool = False   # True when whale signal agrees with model edge
    result: dict[str, Any] = field(default_factory=lambda: {
        "has_edge": False, "trade_attempted": False, "trade_executed": False,
    })


class TradingEngine:
    """Continuous trading engine that coordinates all bot components."""

    def __init__(self, config: Any | None = None):
        self.config: BotConfig = config or load_config()
        self._running = False
        self._cycle_count = 0
        self._cycle_history: list[CycleResult] = []

        bankroll = self.config.risk.bankroll
        self.drawdown = DrawdownManager(bankroll, self.config)
        self.portfolio = PortfolioRiskManager(bankroll, self.config)

        self._pre_cycle_hooks: list[Callable] = []
        self._post_cycle_hooks: list[Callable] = []
        self._positions: list[PositionSnapshot] = []

        # Pre-research filter
        self._research_cache = ResearchCache(
            cooldown_minutes=self.config.scanning.research_cooldown_minutes,
        )
        self._last_filter_stats: FilterStats | None = None

        # ── Analytics & Intelligence Layer ──
        self._regime_detector = RegimeDetector()
        self._calibration_loop = CalibrationFeedbackLoop()
        self._adaptive_weighter = AdaptiveModelWeighter(self.config.ensemble)
        self._smart_entry = SmartEntryCalculator()
        self._current_regime: RegimeState | None = None

        # ── Wallet / Whale Scanner ──
        self._wallet_scanner = WalletScanner(
            min_whale_count=self.config.wallet_scanner.min_whale_count,
            min_conviction_score=self.config.wallet_scanner.min_conviction_score,
        )
        self._last_wallet_scan: float = 0.0
        self._latest_scan_result: Any = None

        # ── WebSocket price feed ──
        self._ws_feed = WebSocketFeed()
        self._ws_task: asyncio.Task[None] | None = None

        # ── Rebalance / Arbitrage tracking ──
        self._last_rebalance_check: float = 0.0
        self._last_arbitrage_scan: float = 0.0
        self._latest_arb_opportunities: list[Any] = []

        # ── Phase 5: Cross-Platform Arbitrage ──
        self._cross_platform_scanner: Any = None
        self._last_cross_platform_scan: float = 0.0
        self._latest_cross_platform_opps: list[Any] = []
        self._latest_complementary_arb: list[Any] = []
        self._latest_correlated_mispricings: list[Any] = []
        if self.config.arbitrage.enabled:
            try:
                from src.policy.cross_platform_arb import CrossPlatformArbScanner
                self._cross_platform_scanner = CrossPlatformArbScanner(
                    self.config.arbitrage,
                )
            except Exception as e:
                log.warning("engine.cross_platform_scanner_init_error", error=str(e))

        # Database (initialised in start())
        self._db: Any = None
        self._audit: Any = None
        self._alert_manager: Any = None

        # Phase 9: Daily summary tracking
        self._last_daily_summary_date: str = ""

        # Phase 9: Graduated deployment + Telegram bot
        self._deployment_manager: Any = None
        self._telegram_bot: Any = None
        self._telegram_task: asyncio.Task[None] | None = None

        # Phase 10: Reconciliation loop
        self._reconciliation_task: asyncio.Task[None] | None = None
        self._reconciliation_stop: asyncio.Event | None = None

        # Phase 10B: Shared exit finalizer (initialised in start())
        self._exit_finalizer: Any = None

        # Phase 10E: Execution plan controller (initialised in start())
        self._plan_controller: Any = None

        # Phase 6: Execution fill tracker (optional)
        self._fill_tracker: Any = None
        if self.config.execution.auto_strategy_selection_enabled:
            from src.execution.fill_tracker import FillTracker
            self._fill_tracker = FillTracker()

        # ── Phase 4: Specialist Router ──
        self._specialist_router: Any = None
        if self.config.specialists.enabled:
            try:
                from src.forecast.specialist_router import SpecialistRouter
                self._specialist_router = SpecialistRouter(self.config.specialists)
            except Exception as e:
                log.warning("engine.specialist_router_init_error", error=str(e))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def cycle_history(self) -> list[CycleResult]:
        return list(self._cycle_history)

    def add_pre_cycle_hook(self, fn: Callable) -> None:
        self._pre_cycle_hooks.append(fn)

    def add_post_cycle_hook(self, fn: Callable) -> None:
        self._post_cycle_hooks.append(fn)

    # ── Lifecycle ────────────────────────────────────────────────────

    def _init_db(self) -> None:
        from src.storage.database import Database
        from src.storage.audit import AuditTrail
        self._db = Database(self.config.storage)
        self._db.connect()
        self._audit = AuditTrail()
        log.info("engine.db_connected", path=self.config.storage.sqlite_path)

    def _persist_engine_state(self, extra: dict[str, Any] | None = None) -> None:
        if not self._db:
            return
        try:
            state = {
                "running": self._running,
                "cycle_count": self._cycle_count,
                "live_trading": is_live_trading_enabled(),
                "paper_mode": self.config.engine.paper_mode,
                "last_cycle": (
                    self._cycle_history[-1].to_dict()
                    if self._cycle_history else None
                ),
                "positions": len(self._positions),
                "scan_interval_minutes": self.config.engine.scan_interval_minutes,
                "max_markets_per_cycle": self.config.engine.max_markets_per_cycle,
                "auto_start": self.config.engine.auto_start,
                "filter_stats": (
                    self._last_filter_stats.__dict__
                    if self._last_filter_stats else None
                ),
                "research_cache_size": self._research_cache.size(),
            }
            if extra:
                state.update(extra)
            self._db.set_engine_state("engine_status", json.dumps(state))
            dd = self.drawdown.state
            self._db.set_engine_state("drawdown", json.dumps(dd.to_dict()))
        except Exception as e:
            log.warning("engine.persist_state_error", error=str(e))

    async def start(self) -> None:
        self._running = True
        interval = self.config.engine.cycle_interval_secs
        self._init_db()
        self._restore_kill_switch_state()

        # Phase 10B: Create shared exit finalizer
        if self._db:
            from src.execution.exit_finalizer import ExitFinalizer
            self._exit_finalizer = ExitFinalizer(self._db, self.config)

        # Phase 10E: Create plan controller if enabled
        if self.config.execution.plan_orchestration_enabled and self._db:
            try:
                from src.execution.plan_controller import PlanController
                self._plan_controller = PlanController(self._db, self.config.execution)
                log.info("engine.plan_controller_initialized")
            except Exception as e:
                log.warning("engine.plan_controller_init_error", error=str(e))

        # Phase 10: Wire FillTracker to DB for persistence + load history
        if self._fill_tracker and self._db:
            self._fill_tracker._db = self._db
            try:
                loaded = self._fill_tracker.load_from_db(lookback_hours=24)
                if loaded > 0:
                    log.info("engine.fill_tracker_loaded", count=loaded)
            except Exception as e:
                log.warning("engine.fill_tracker_load_error", error=str(e))

        # Init alert manager
        try:
            from src.observability.alerts import AlertManager
            self._alert_manager = AlertManager(self.config)
        except Exception as e:
            log.warning("engine.alert_manager_init_error", error=str(e))

        # Phase 9: Graduated deployment — apply stage limits
        if self.config.production.enabled:
            try:
                from src.policy.graduated_deployment import GraduatedDeploymentManager
                conn = self._db._conn if self._db else None
                self._deployment_manager = GraduatedDeploymentManager(
                    self.config, conn,
                )
                bankroll, max_stake = self._deployment_manager.apply_stage_limits()
                self.config.risk.bankroll = bankroll
                self.config.risk.max_stake_per_market = max_stake
                self.drawdown = DrawdownManager(bankroll, self.config)
                self.portfolio = PortfolioRiskManager(bankroll, self.config)
            except Exception as e:
                log.warning("engine.deployment_manager_init_error", error=str(e))

        self._db.insert_alert("info", "\U0001f916 Trading engine started", "system")
        log.info(
            "engine.starting",
            interval_secs=interval,
            live_trading=is_live_trading_enabled(),
            bankroll=self.config.risk.bankroll,
        )
        self._persist_engine_state()

        # Graceful shutdown on SIGTERM / SIGINT
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except (NotImplementedError, RuntimeError):
                pass  # Windows or non-main thread (e.g. dashboard)

        # Start WebSocket price feed in background
        try:
            self._ws_task = asyncio.create_task(self._ws_feed.start())
            log.info("engine.ws_feed_started")
        except Exception as e:
            log.warning("engine.ws_feed_start_error", error=str(e))

        # Phase 9: Start Telegram kill bot in background
        prod = self.config.production
        if prod.telegram_kill_enabled and prod.telegram_kill_token:
            try:
                from src.observability.telegram_bot import TelegramKillBot
                self._telegram_bot = TelegramKillBot(
                    token=prod.telegram_kill_token,
                    chat_id=prod.telegram_kill_chat_id,
                    engine=self,
                )
                self._telegram_task = asyncio.create_task(
                    self._telegram_bot.start()
                )
                log.info("engine.telegram_bot_started")
            except Exception as e:
                log.warning("engine.telegram_bot_start_error", error=str(e))

        # Phase 10: Start reconciliation loop in background
        if (
            self.config.execution.reconciliation_enabled
            and is_live_trading_enabled()
            and not self.config.execution.dry_run
            and self._db
        ):
            try:
                from src.execution.reconciliation import run_reconciliation_loop
                from src.connectors.polymarket_clob import CLOBClient

                recon_clob = CLOBClient()
                self._reconciliation_stop = asyncio.Event()
                self._reconciliation_task = asyncio.create_task(
                    run_reconciliation_loop(
                        db=self._db,
                        clob=recon_clob,
                        config=self.config.execution,
                        stop_event=self._reconciliation_stop,
                        fill_tracker=self._fill_tracker,
                        on_buy_fill=lambda token_id: self._ws_feed.subscribe(token_id),
                        exit_finalizer=self._exit_finalizer,
                        plan_controller=self._plan_controller,
                    )
                )
                log.info("engine.reconciliation_loop_started")
            except Exception as e:
                log.warning("engine.reconciliation_start_error", error=str(e))

        while self._running:
            try:
                await self._run_cycle()
            except Exception as e:
                log.error("engine.cycle_error", error=str(e))
                traceback.print_exc()
                if self._db:
                    self._db.insert_alert("error", f"Cycle error: {e}", "system")
            self._persist_engine_state()
            if self._running:
                log.info("engine.sleeping", seconds=interval)
                await asyncio.sleep(interval)

        log.info("engine.stopped", total_cycles=self._cycle_count)
        if self._db:
            self._db.insert_alert("info", "\U0001f6d1 Trading engine stopped", "system")
            self._persist_engine_state({"running": False})

    def stop(self) -> None:
        log.info("engine.stop_requested")
        self._running = False
        # Stop WebSocket feed
        if self._ws_task and not self._ws_task.done():
            asyncio.ensure_future(self._ws_feed.stop())
        # Stop Telegram bot
        if self._telegram_bot:
            self._telegram_bot.stop()
        if self._telegram_task and not self._telegram_task.done():
            self._telegram_task.cancel()
        # Stop reconciliation loop
        if self._reconciliation_stop:
            self._reconciliation_stop.set()
        if self._reconciliation_task and not self._reconciliation_task.done():
            self._reconciliation_task.cancel()

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        log.info("engine.signal_received", signal=sig.name)
        self.stop()

    # ── Phase 9: Kill Switch Persistence ────────────────────────────

    def _restore_kill_switch_state(self) -> None:
        """Restore kill switch from DB on engine startup."""
        if not self._db or not self.config.production.enabled:
            return
        if not self.config.production.persist_kill_switch:
            return
        try:
            state = self._db.get_kill_switch_state()
            if state["is_killed"]:
                self.drawdown.state.is_killed = True
                self.drawdown.state.kelly_multiplier = 0.0
                self.config.risk.kill_switch = True
                log.warning(
                    "engine.kill_switch_restored",
                    reason=state["kill_reason"],
                    killed_at=state["killed_at"],
                )
        except Exception as e:
            log.warning("engine.kill_switch_restore_error", error=str(e))

    def _persist_kill_switch(self, reason: str, killed_by: str) -> None:
        """Write kill switch state to DB."""
        if not self._db or not self.config.production.enabled:
            return
        if not self.config.production.persist_kill_switch:
            return
        try:
            daily_pnl = self._db.get_daily_pnl()
            self._db.set_kill_switch(
                is_killed=True,
                reason=reason,
                killed_by=killed_by,
                daily_pnl=daily_pnl,
                bankroll=self.config.risk.bankroll,
            )
        except Exception as e:
            log.warning("engine.persist_kill_error", error=str(e))

    def _check_daily_pnl_kill(self) -> bool:
        """Check if daily P&L loss exceeds percentage threshold.

        Returns True if the kill switch was triggered.
        """
        if not self._db or not self.config.production.enabled:
            return False
        if not self.config.production.daily_loss_kill_enabled:
            return False
        if self.drawdown.state.is_killed:
            return False  # already killed

        try:
            daily_pnl = self._db.get_daily_pnl()
            bankroll = self.config.risk.bankroll
            if bankroll <= 0:
                return False
            loss_pct = abs(daily_pnl) / bankroll
            threshold = self.config.production.daily_loss_kill_pct

            if daily_pnl < 0 and loss_pct >= threshold:
                self.drawdown.state.is_killed = True
                self.drawdown.state.kelly_multiplier = 0.0
                self.config.risk.kill_switch = True
                reason = (
                    f"Daily P&L loss {loss_pct:.1%} exceeds "
                    f"{threshold:.1%} threshold (${daily_pnl:.2f})"
                )
                log.critical("engine.daily_pnl_kill", reason=reason)
                self._persist_kill_switch(reason, "daily_pnl_auto")
                if self._db:
                    self._db.insert_alert("critical", reason, "risk")
                return True
        except Exception as e:
            log.warning("engine.daily_pnl_kill_check_error", error=str(e))
        return False

    async def _maybe_send_daily_summary(self) -> None:
        """Send daily summary at the configured hour (once per day)."""
        if not self._db:
            return
        if not self.config.alerts.daily_summary_enabled:
            return

        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Already sent today
        if self._last_daily_summary_date == today:
            return
        # Not yet the configured hour
        if now.hour < self.config.alerts.daily_summary_hour:
            return

        try:
            from src.observability.daily_summary import DailySummaryGenerator
            generator = DailySummaryGenerator(
                self._db.conn,
                bankroll=self.config.risk.bankroll,
            )
            summary = generator.generate(today)
            generator.persist(summary)

            if self._alert_manager:
                await generator.send_summary(summary, self._alert_manager)

            self._last_daily_summary_date = today
            log.info("engine.daily_summary_generated", date=today)
        except Exception as e:
            log.warning("engine.daily_summary_error", error=str(e))

    def _maybe_check_invariants(self) -> None:
        """Run invariant checks every N cycles when enabled."""
        if not self._db:
            return
        if not self.config.execution.invariant_checks_enabled:
            return
        interval = self.config.execution.invariant_check_interval_cycles
        if self._cycle_count % interval != 0:
            return

        try:
            from src.observability.invariant_checker import check_invariants
            from src.observability.metrics import metrics as _inv_metrics
            violations = check_invariants(self._db)
            if violations:
                for v in violations:
                    log.warning(
                        "engine.invariant_violation",
                        check=v.check,
                        severity=v.severity,
                        market_id=v.market_id[:8],
                        message=v.message,
                    )
                    _inv_metrics.incr(f"invariant.violations.{v.check}")
                    _inv_metrics.incr(f"invariant.violations_by_severity.{v.severity}")
                    if v.severity == "critical":
                        self._db.insert_alert("critical", v.message, "invariant_checker")
        except Exception as e:
            log.warning("engine.invariant_check_error", error=str(e))

    # ── Cycle ────────────────────────────────────────────────────────

    async def _run_cycle(self) -> CycleResult:
        self._cycle_count += 1
        cycle = CycleResult(cycle_id=self._cycle_count, started_at=time.time())
        log.info("engine.cycle_start", cycle_id=cycle.cycle_id)

        try:
            for hook in self._pre_cycle_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    log.warning("engine.hook_error", hook=str(hook), error=str(e))

            can_trade, dd_reason = self.drawdown.can_trade()
            if not can_trade:
                log.warning("engine.drawdown_halt", reason=dd_reason)
                cycle.status = "skipped"
                cycle.errors.append(f"Drawdown halt: {dd_reason}")
                if self._db:
                    self._db.insert_alert("warning", f"Cycle skipped: {dd_reason}", "risk")
                # Persist kill switch to DB if drawdown killed
                if self.drawdown.state.is_killed:
                    self._persist_kill_switch(dd_reason, "drawdown_auto")
                self._finish_cycle(cycle)
                return cycle

            # ── Daily P&L Kill Check ─────────────────────────────────
            if self._check_daily_pnl_kill():
                cycle.status = "skipped"
                cycle.errors.append("Daily P&L kill switch triggered")
                self._finish_cycle(cycle)
                return cycle

            # ── Budget Check ─────────────────────────────────────────
            if self.config.budget.enabled:
                can_spend, remaining = cost_tracker.check_budget(
                    self.config.budget.daily_limit_usd,
                )
                if not can_spend:
                    log.warning(
                        "engine.budget_exhausted",
                        daily_limit=self.config.budget.daily_limit_usd,
                    )
                    cycle.status = "skipped"
                    cycle.errors.append("Budget exhausted for today")
                    if self._db:
                        self._db.insert_alert(
                            "warning", "Daily API budget exhausted", "budget",
                        )
                    self._finish_cycle(cycle)
                    return cycle

            # ── Regime Detection ─────────────────────────────────────
            try:
                if self._db:
                    self._current_regime = self._regime_detector.detect(
                        self._db.conn,
                    )
                    # Persist regime state for dashboard
                    import datetime as _dt
                    self._db.conn.execute("""
                        INSERT INTO regime_history
                            (regime, confidence, kelly_multiplier,
                             size_multiplier, explanation, detected_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        self._current_regime.regime,
                        self._current_regime.confidence,
                        self._current_regime.kelly_multiplier,
                        self._current_regime.size_multiplier,
                        self._current_regime.explanation,
                        _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    ))
                    self._db.conn.commit()
            except Exception as e:
                log.warning("engine.regime_detection_error", error=str(e))

            # ── Calibration check ────────────────────────────────────
            try:
                if self._db and self._cycle_count % 10 == 0:
                    self._calibration_loop.retrain_calibrator(self._db.conn)
            except Exception as e:
                log.warning("engine.calibration_retrain_error", error=str(e))
                return cycle

            markets = await self._discover_markets()
            cycle.markets_scanned = len(markets)

            if not markets:
                log.info("engine.no_markets")
                cycle.status = "completed"
                self._finish_cycle(cycle)
                return cycle

            # Pre-research filter — skip low-quality markets before SerpAPI
            blocked_types = set(self.config.scanning.filter_blocked_types)
            preferred_types = self.config.scanning.preferred_types or None
            min_score = self.config.scanning.filter_min_score
            max_per_cycle = self.config.engine.max_markets_per_cycle
            max_age_hours = self.config.scanning.max_market_age_hours

            self._research_cache.clear_stale()

            filtered, fstats = filter_markets(
                markets,
                min_score=min_score,
                max_pass=max_per_cycle,
                research_cache=self._research_cache,
                blocked_types=blocked_types,
                preferred_types=preferred_types,
                max_market_age_hours=max_age_hours,
            )
            self._last_filter_stats = fstats
            cycle.markets_researched = len(filtered)

            if not filtered:
                log.info("engine.all_filtered", stats=fstats.__dict__)
                cycle.status = "completed"
                self._finish_cycle(cycle)
                return cycle

            for candidate in filtered:
                try:
                    result = await self._process_candidate(candidate, cycle.cycle_id)
                    # Mark as researched so it's skipped for cooldown period
                    self._research_cache.mark_researched(
                        getattr(candidate, "id", ""),
                    )
                    if result.get("has_edge"):
                        cycle.edges_found += 1
                    if result.get("trade_attempted"):
                        cycle.trades_attempted += 1
                    if result.get("trade_executed"):
                        cycle.trades_executed += 1
                except Exception as e:
                    log.error(
                        "engine.candidate_error",
                        market_id=getattr(candidate, "id", "?"),
                        error=str(e),
                    )
                    cycle.errors.append(str(e))
                    traceback.print_exc()

            await self._check_positions()
            await self._confirm_pending_orders()
            await self._maybe_rebalance()
            await self._maybe_scan_wallets()
            await self._maybe_scan_arbitrage(markets)
            await self._maybe_scan_cross_platform_arb(markets)

            # ── Daily Summary ────────────────────────────────────────
            await self._maybe_send_daily_summary()

            # ── Invariant Checks ───────────────────────────────────
            self._maybe_check_invariants()

            cycle.status = "completed"

        except Exception as e:
            cycle.status = "error"
            cycle.errors.append(str(e))
            log.error("engine.cycle_failed", error=str(e))

        for hook in self._post_cycle_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            except Exception as e:
                log.warning("engine.hook_error", hook=str(hook), error=str(e))

        self._finish_cycle(cycle)
        return cycle

    def _finish_cycle(self, cycle: CycleResult) -> None:
        cycle.ended_at = time.time()
        cycle.duration_secs = round(cycle.ended_at - cycle.started_at, 2)
        self._cycle_history.append(cycle)
        if len(self._cycle_history) > 100:
            self._cycle_history = self._cycle_history[-50:]

        # Collect API cost summary for this cycle
        cycle_costs = cost_tracker.end_cycle()

        log.info(
            "engine.cycle_complete",
            cycle_id=cycle.cycle_id,
            duration=cycle.duration_secs,
            scanned=cycle.markets_scanned,
            researched=cycle.markets_researched,
            edges=cycle.edges_found,
            trades=cycle.trades_executed,
            status=cycle.status,
            cycle_cost_usd=cycle_costs["cycle_cost_usd"],
            total_cost_usd=cycle_costs["total_cost_usd"],
        )
        if self._db:
            self._db.insert_alert(
                "info",
                f"Cycle {cycle.cycle_id}: scanned={cycle.markets_scanned} "
                f"researched={cycle.markets_researched} edges={cycle.edges_found} "
                f"trades={cycle.trades_executed} ({cycle.duration_secs:.1f}s)",
                "engine",
            )

    # ── Market Discovery ─────────────────────────────────────────────

    async def _discover_markets(self) -> list[Any]:
        from src.connectors.polymarket_gamma import fetch_active_markets
        cb = circuit_breakers.get("gamma")
        if not cb.allow_request():
            log.warning(
                "engine.discovery_circuit_open",
                retry_after=cb.time_until_retry(),
            )
            return []
        try:
            markets = await fetch_active_markets(
                min_volume=self.config.risk.min_liquidity, limit=200,
            )
            cb.record_success()
            return markets
        except Exception as e:
            cb.record_failure()
            log.error("engine.discovery_error", error=str(e))
            return []

    async def _rank_markets(self, markets: list[Any]) -> list[Any]:
        scored = []
        for m in markets:
            score = (
                m.volume * 0.3
                + m.liquidity * 0.5
                + (1.0 if m.has_clear_resolution else 0.0) * 0.2
            )
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ── Full Pipeline ────────────────────────────────────────────────

    async def _process_candidate(self, market: Any, cycle_id: int) -> dict[str, Any]:
        """Process a single market through the full research-to-trade pipeline.

        Orchestrates stages via PipelineContext and dedicated stage methods.
        """
        ctx = PipelineContext(market=market, cycle_id=cycle_id,
                              market_id=market.id, question=market.question)

        # ── Early exit: skip if we already hold a position or have a pending order
        if self._db:
            existing = [p for p in (self._db.get_open_positions())
                        if p.market_id == market.id]
            if existing:
                log.info("engine.duplicate_skip", market_id=market.id[:8],
                         msg="Already have open position — skipping")
                ctx.result["skipped"] = "duplicate_position"
                return ctx.result

            # Also skip if there's already a submitted/pending order for this market
            try:
                if self._db.has_active_order_for_market(market.id):
                    log.info("engine.duplicate_order_skip", market_id=market.id[:8],
                             msg="Already have pending order — skipping")
                    ctx.result["skipped"] = "duplicate_order"
                    return ctx.result
            except Exception:
                pass  # If open_orders query fails, proceed anyway

        # ── Stage 0: Classification ──────────────────────────────────
        self._stage_classify(ctx)

        # ── Stage 1: Research ────────────────────────────────────────
        ok = await self._stage_research(ctx)
        if not ok:
            return ctx.result

        # ── Stage 2: Build Features ──────────────────────────────────
        from src.forecast.feature_builder import build_features
        ctx.features = build_features(market=market, evidence=ctx.evidence)

        # ── Stage 3: Forecast ────────────────────────────────────────
        forecast_ok = await self._stage_forecast(ctx)
        if not forecast_ok:
            self._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                reason="Forecast failed or circuit open")
            return ctx.result

        # ── Stage 3b: Apply Calibration ──────────────────────────────
        self._stage_calibrate(ctx)

        # ── Stage 4: Edge Calculation + Whale Adjustment ─────────────
        self._stage_edge_calc(ctx)

        # ── Stage 4b: Edge Uncertainty Adjustment ─────────────────
        self._stage_uncertainty_adjustment(ctx)
        ctx.result["has_edge"] = ctx.has_edge

        # ── Stage 5: Risk Checks ─────────────────────────────────────
        self._stage_risk_checks(ctx)

        # ── Persist forecast to DB ───────────────────────────────────
        self._stage_persist_forecast(ctx)

        # ── Portfolio Correlation Check ──────────────────────────────
        self._stage_correlation_check(ctx)

        # ── Portfolio VaR Gate ─────────────────────────────────────
        self._stage_var_gate(ctx)

        # ── Decision Gate ────────────────────────────────────────────
        if not ctx.risk_result.allowed:
            log.info("engine.no_trade", market_id=ctx.market_id,
                     violations=ctx.risk_result.violations)
            self._log_candidate(
                cycle_id, market, forecast=ctx.forecast, evidence=ctx.evidence,
                edge_result=ctx.edge_result, decision="NO TRADE",
                reason="; ".join(ctx.risk_result.violations),
            )
            if self._audit:
                self._audit.record_trade_decision(
                    market_id=ctx.market_id, question=ctx.question,
                    model_prob=ctx.forecast.model_probability,
                    implied_prob=ctx.forecast.implied_probability,
                    edge=ctx.forecast.edge,
                    confidence=ctx.forecast.confidence_level,
                    risk_result=ctx.risk_result.to_dict(), position_size=0.0,
                    evidence_summary=ctx.evidence.summary[:200],
                )
            return ctx.result

        # ── Stage 6: Position Sizing ─────────────────────────────────
        self._stage_position_sizing(ctx)
        if ctx.position is None:
            return ctx.result

        ctx.result["trade_attempted"] = True

        # ── Stage 7: Build & Route Order ─────────────────────────────
        await self._stage_execute_order(ctx)

        # ── Stage 8: Audit + Log ─────────────────────────────────────
        self._stage_audit_and_log(ctx)

        return ctx.result

    # ── Pipeline Stage Methods ────────────────────────────────────────

    def _stage_classify(self, ctx: PipelineContext) -> None:
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

    async def _stage_research(self, ctx: PipelineContext) -> bool:
        """Stage 1: Research. Returns False if research failed and pipeline should abort."""
        cb = circuit_breakers.get("research")
        if not cb.allow_request():
            log.warning(
                "engine.research_circuit_open",
                market_id=ctx.market_id,
                retry_after=cb.time_until_retry(),
            )
            self._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                reason="Research circuit breaker open")
            return False

        from src.research.query_builder import build_queries
        from src.research.source_fetcher import SourceFetcher
        from src.research.evidence_extractor import EvidenceExtractor
        from src.connectors.web_search import create_search_provider

        search_provider = create_search_provider(self.config.research.search_provider)
        source_fetcher = SourceFetcher(search_provider, self.config.research)

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
            )
            extractor = EvidenceExtractor(self.config.forecasting)
            ctx.evidence = await extractor.extract(
                market_id=ctx.market_id, question=ctx.question,
                sources=ctx.sources, market_type=ctx.market.market_type,
            )
            cb.record_success()
        except Exception as e:
            cb.record_failure()
            log.error("engine.research_failed", market_id=ctx.market_id, error=str(e))
            self._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                reason=f"Research failed: {e}")
            return False
        finally:
            await source_fetcher.close()
            await search_provider.close()

        log.info(
            "engine.research_done", market_id=ctx.market_id,
            sources=len(ctx.sources), bullets=len(ctx.evidence.bullets),
            quality=round(ctx.evidence.quality_score, 3),
        )
        return True

    async def _stage_forecast(self, ctx: PipelineContext) -> bool:
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

            ens_result = await ens_forecaster.forecast(
                features=ctx.features,
                evidence=ctx.evidence,
                base_rate_info=base_rate_info,
                prompt_version=self.config.forecasting.prompt_version,
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

    def _stage_calibrate(self, ctx: PipelineContext) -> None:
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

    def _stage_edge_calc(self, ctx: PipelineContext) -> None:
        """Stage 4: Edge calculation + whale/smart-money adjustment.

        Fixes applied:
          - Match by market_slug OR condition_id (not market_id)
          - Match direction BULLISH/BEARISH (not BUY/SELL)
          - Whale-edge convergence: when whale signal agrees with model edge,
            use a lower min_edge threshold for higher conviction trades
        """
        from src.policy.edge_calc import calculate_edge
        ctx.edge_result = calculate_edge(
            implied_prob=ctx.forecast.implied_probability,
            model_prob=ctx.forecast.model_probability,
            transaction_fee_pct=self.config.risk.transaction_fee_pct,
            gas_cost_usd=self.config.risk.gas_cost_usd,
            holding_hours=ctx.features.hours_to_resolution,
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
                    )
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
                    )
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

    def _stage_uncertainty_adjustment(self, ctx: PipelineContext) -> None:
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

    def _stage_risk_checks(self, ctx: PipelineContext) -> None:
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

    def _stage_persist_forecast(self, ctx: PipelineContext) -> None:
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

    def _stage_correlation_check(self, ctx: PipelineContext) -> None:
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

    def _stage_var_gate(self, ctx: PipelineContext) -> None:
        """Check portfolio VaR limit before allowing entry."""
        if not self.config.portfolio.var_gate_enabled:
            return
        if not ctx.risk_result or not ctx.risk_result.allowed:
            return

        from src.policy.correlation import EventCorrelationScorer
        from src.policy.portfolio_risk import check_var_gate, PositionSnapshot

        scorer = EventCorrelationScorer(self.config.portfolio)

        new_pos = PositionSnapshot(
            market_id=ctx.market_id,
            question=ctx.question,
            category=ctx.classification.category if ctx.classification else "",
            event_slug=getattr(ctx.market, "slug", "") or "",
            side=ctx.edge_result.direction.replace("BUY_", "") if ctx.edge_result else "YES",
            size_usd=50.0,  # estimate; actual size computed later
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

    def _stage_position_sizing(self, ctx: PipelineContext) -> None:
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
        ctx.position = calculate_position_size(
            edge=ctx.edge_result, risk_config=self.config.risk,
            confidence_level=ctx.forecast.confidence_level,
            drawdown_multiplier=self.drawdown.state.kelly_multiplier,
            timeline_multiplier=ctx.features.time_decay_multiplier,
            price_volatility=ctx.features.price_volatility,
            regime_multiplier=regime_kelly * regime_size,
            category_multiplier=cat_mult,
            uncertainty_multiplier=unc_mult,
        )
        if ctx.position.stake_usd < 1.0:
            log.info("engine.stake_too_small", market_id=ctx.market_id,
                     stake=ctx.position.stake_usd)
            self._log_candidate(
                ctx.cycle_id, ctx.market, forecast=ctx.forecast,
                evidence=ctx.evidence, edge_result=ctx.edge_result,
                decision="NO TRADE", reason="Stake too small",
                stake=ctx.position.stake_usd,
            )
            ctx.position = None

    async def _stage_execute_order(self, ctx: PipelineContext) -> None:
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
        plan = self._plan_controller.create_plan(
            orders, execution_strategy,
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

    def _stage_audit_and_log(self, ctx: PipelineContext) -> None:
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
                        prob, -1.0,
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
    ) -> None:
        if not self._db:
            return
        try:
            self._db.insert_candidate(
                cycle_id=cycle_id, market_id=market.id,
                question=market.question[:200],
                market_type=market.market_type,
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

    def _record_performance_log(
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
        except Exception as e:
            log.warning("engine.performance_log_error", error=str(e))

    def _maybe_run_post_mortem(self, market_id: str) -> None:
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

    async def _check_positions(self) -> None:
        """Fetch live prices for all open positions and update PNL.

        Uses WebSocket feed when available for instant pricing,
        falls back to Gamma REST API when WS prices are stale or unavailable.
        """
        if not self._db:
            return

        positions = self._db.get_open_positions()
        if not positions:
            self._positions = []
            return

        from src.connectors.polymarket_gamma import GammaClient
        from src.execution.direction import parse_direction as _parse_dir

        client = GammaClient()
        snapshots: list[PositionSnapshot] = []
        ws_hits = 0
        rest_hits = 0
        try:
            for pos in positions:
                try:
                    # Try WebSocket feed first for instant pricing
                    current_price = None
                    ws_tick = self._ws_feed.get_last_price(pos.token_id)
                    if ws_tick and (time.time() - ws_tick.timestamp) < 60:
                        current_price = ws_tick.mid or ws_tick.best_bid
                        ws_hits += 1
                    
                    # Fall back to REST API if WS price unavailable or stale
                    market = None
                    if current_price is None:
                        market = await client.get_market(pos.market_id)
                        current_price = pos.current_price  # fallback
                        for tok in market.tokens:
                            if tok.token_id == pos.token_id:
                                current_price = tok.price
                                break
                        rest_hits += 1

                    # Calculate PNL based on outcome side
                    _o = getattr(pos, "outcome_side", "")
                    _a, _o = (getattr(pos, "action_side", ""), _o) if _o else _parse_dir(pos.direction)
                    if _o == "NO":
                        pnl = (pos.entry_price - current_price) * pos.size
                    else:
                        # YES or unknown — long position
                        pnl = (current_price - pos.entry_price) * pos.size

                    self._db.update_position_price(
                        pos.market_id, current_price, round(pnl, 4),
                    )

                    # Fetch market metadata (needed for snapshots + exit trades)
                    if market is None:
                        market = await client.get_market(pos.market_id)
                    mkt_record = self._db.get_market(pos.market_id)

                    # ── Determine exit reason (if any) ───────────────
                    sl_pct = getattr(self.config.risk, "stop_loss_pct", 0.0)
                    tp_pct = getattr(self.config.risk, "take_profit_pct", 0.0)
                    max_hold = getattr(self.config.risk, "max_holding_hours", 72.0)
                    pnl_pct = pnl / pos.stake_usd if pos.stake_usd > 0 else 0.0
                    exit_reason = ""

                    # Stop-loss
                    if sl_pct > 0 and pnl_pct <= -sl_pct:
                        exit_reason = f"STOP_LOSS: {pnl_pct:.1%} <= -{sl_pct:.0%}"
                    # Take-profit
                    elif tp_pct > 0 and pnl_pct >= tp_pct:
                        exit_reason = f"TAKE_PROFIT: {pnl_pct:.1%} >= +{tp_pct:.0%}"
                    # Market resolved (price at 0 or 1)
                    elif current_price is not None and (current_price >= 0.98 or current_price <= 0.02):
                        exit_reason = f"MARKET_RESOLVED: price={current_price:.4f}"
                    # Max holding period exceeded
                    elif max_hold > 0:
                        try:
                            import datetime as _dt
                            opened = _dt.datetime.fromisoformat(pos.opened_at.replace("Z", "+00:00"))
                            now = _dt.datetime.now(_dt.timezone.utc)
                            holding_hours = (now - opened).total_seconds() / 3600
                            if holding_hours >= max_hold:
                                exit_reason = f"MAX_HOLDING: {holding_hours:.1f}h >= {max_hold:.0f}h"
                        except Exception:
                            pass

                    if exit_reason:
                        log.info(
                            "engine.auto_exit",
                            market_id=pos.market_id[:8],
                            reason=exit_reason,
                            pnl=round(pnl, 4),
                            pnl_pct=f"{pnl_pct:.1%}",
                        )

                        # ── Route exit order (live or simulated) ───────
                        exit_confirmed = await self._route_exit_order(
                            pos, current_price, pnl, exit_reason, mkt_record,
                        )
                        if exit_confirmed:
                            continue  # skip snapshot — position closed
                        # If not confirmed (live SELL pending), keep position open

                    # Build snapshot for portfolio risk
                    snapshots.append(PositionSnapshot(
                        market_id=pos.market_id,
                        question=mkt_record.question if mkt_record else "",
                        category=mkt_record.category if mkt_record else "",
                        event_slug=market.slug or "",
                        side=_o if _o else ("YES" if pos.direction in ("BUY_YES", "BUY") else "NO"),
                        size_usd=pos.stake_usd,
                        entry_price=pos.entry_price,
                        current_price=current_price,
                        unrealised_pnl=round(pnl, 4),
                    ))

                    log.info(
                        "engine.position_update",
                        market_id=pos.market_id[:8],
                        entry=pos.entry_price,
                        current=current_price,
                        pnl=round(pnl, 4),
                        source="ws" if ws_tick else "rest",
                    )

                except Exception as e:
                    log.warning(
                        "engine.position_price_error",
                        market_id=pos.market_id[:8],
                        error=str(e),
                    )
                    # Keep stale snapshot
                    _o2 = getattr(pos, "outcome_side", "")
                    _a2, _o2 = (getattr(pos, "action_side", ""), _o2) if _o2 else _parse_dir(pos.direction)
                    snapshots.append(PositionSnapshot(
                        market_id=pos.market_id,
                        question="",
                        category="",
                        event_slug="",
                        side=_o2 if _o2 else ("YES" if pos.direction in ("BUY_YES", "BUY") else "NO"),
                        size_usd=pos.stake_usd,
                        entry_price=pos.entry_price,
                        current_price=pos.current_price,
                        unrealised_pnl=pos.pnl,
                    ))

        finally:
            await client.close()

        self._positions = snapshots
        log.info(
            "engine.positions_checked",
            count=len(snapshots),
            total_pnl=round(sum(s.unrealised_pnl for s in snapshots), 4),
            ws_hits=ws_hits,
            rest_hits=rest_hits,
        )

    async def _route_exit_order(
        self,
        pos: Any,
        current_price: float,
        pnl: float,
        exit_reason: str,
        mkt_record: Any,
    ) -> bool:
        """Route an exit order — live SELL or simulated close.

        Returns True if the position was closed (simulated or live fill confirmed).
        Returns False if the exit order is pending (live SELL submitted but not yet filled).
        """
        use_live_exit = (
            is_live_trading_enabled()
            and self.config.execution.live_exit_routing
            and not self.config.execution.dry_run
        )

        if use_live_exit:
            # Route a real SELL order through the OrderRouter
            try:
                from src.execution.order_builder import build_exit_order
                from src.execution.order_router import OrderRouter
                from src.connectors.polymarket_clob import CLOBClient

                from src.execution.direction import parse_direction as _pd_exit
                _exit_os = getattr(pos, "outcome_side", "") or _pd_exit(pos.direction)[1]
                exit_order = build_exit_order(
                    market_id=pos.market_id,
                    token_id=pos.token_id,
                    size=pos.size,
                    current_price=current_price,
                    config=self.config.execution,
                    exit_reason=exit_reason.split(":")[0],
                    outcome_side=_exit_os,
                )
                clob = CLOBClient()
                try:
                    router = OrderRouter(clob, self.config.execution)
                    result = await router.submit_order(exit_order)

                    if result.status == "filled":
                        # SELL confirmed — close position
                        self._finalize_exit(
                            pos, result.fill_price or current_price,
                            pnl, exit_reason, mkt_record,
                        )
                        return True
                    elif result.status in ("submitted", "pending"):
                        # SELL order on the book — reconciliation will handle
                        from src.storage.models import OrderRecord
                        self._db.insert_order(OrderRecord(
                            order_id=result.order_id,
                            clob_order_id=result.clob_order_id,
                            market_id=pos.market_id,
                            token_id=pos.token_id,
                            side="SELL",
                            order_type=exit_order.order_type,
                            price=exit_order.price,
                            size=pos.size,
                            stake_usd=pos.stake_usd,
                            status=result.status,
                            dry_run=False,
                            action_side=exit_order.action_side,
                            outcome_side=exit_order.outcome_side or "YES",
                        ))
                        log.info(
                            "engine.exit_order_pending",
                            market_id=pos.market_id[:8],
                            order_id=result.order_id[:8],
                        )
                        return False
                    else:
                        # SELL failed — will retry next cycle
                        log.warning(
                            "engine.exit_order_failed",
                            market_id=pos.market_id[:8],
                            error=result.error,
                        )
                        return False
                finally:
                    await clob.close()
            except Exception as e:
                log.error("engine.live_exit_error", market_id=pos.market_id[:8], error=str(e))
                return False
        else:
            # Paper mode / live_exit_routing disabled — simulated exit
            from src.storage.models import TradeRecord
            from src.execution.direction import parse_direction as _pd2
            _sim_outcome = getattr(pos, "outcome_side", "") or _pd2(pos.direction)[1]
            self._db.insert_trade(TradeRecord(
                id=f"exit-{pos.market_id[:8]}-{int(time.time())}",
                order_id=f"auto-exit-{pos.market_id[:8]}",
                market_id=pos.market_id,
                token_id=pos.token_id,
                side="SELL",
                price=current_price,
                size=pos.size,
                stake_usd=pos.stake_usd,
                status=f"SIMULATED|{exit_reason}",
                dry_run=True,
                action_side="SELL",
                outcome_side=_sim_outcome or "YES",
            ))
            self._finalize_exit(pos, current_price, pnl, exit_reason, mkt_record)
            return True

    def _finalize_exit(
        self,
        pos: Any,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        mkt_record: Any,
    ) -> None:
        """Archive position, record performance, clean up.

        Delegates to ExitFinalizer for the full 5-step pipeline.
        Falls back to inline logic if finalizer not yet initialised.
        """
        if self._exit_finalizer:
            self._exit_finalizer.finalize(
                pos=pos,
                exit_price=exit_price,
                pnl=pnl,
                close_reason=exit_reason,
                mkt_record=mkt_record,
            )
        else:
            # Fallback (e.g. tests that skip start())
            self._db.archive_position(
                pos=pos,
                exit_price=exit_price,
                pnl=round(pnl, 4),
                close_reason=exit_reason.split(":")[0],
            )
            self._record_performance_log(
                pos=pos,
                exit_price=exit_price,
                pnl=round(pnl, 4),
                mkt_record=mkt_record,
            )
            self._maybe_run_post_mortem(pos.market_id)
            self._db.remove_position(pos.market_id)
            self._db.insert_alert(
                "warning",
                f"Auto-exit {pos.market_id[:8]}: {exit_reason} "
                f"(PNL ${pnl:.2f})",
                "engine",
            )

    async def _confirm_pending_orders(self) -> None:
        """Confirm pending orders from the open_orders table.

        Paper mode: auto-fill pending orders immediately.
        Live mode: defer to reconciliation loop (Batch C).
        """
        if not self._db:
            return

        try:
            pending = self._db.get_open_orders(status="submitted")
            pending += self._db.get_open_orders(status="pending")
        except Exception:
            return

        if not pending:
            return

        is_live = is_live_trading_enabled() and not self.config.execution.dry_run

        for order in pending:
            if is_live:
                # Live mode — reconciliation loop will handle this
                continue

            # Paper mode — auto-confirm as filled
            from src.storage.models import TradeRecord, PositionRecord
            from src.execution.direction import parse_direction as _pd_paper
            fill_price = order.price
            fill_size = order.size
            _paper_a = getattr(order, "action_side", "") or ""
            _paper_o = getattr(order, "outcome_side", "") or ""
            if not _paper_a:
                _paper_a, _paper_o = _pd_paper(order.side)

            self._db.update_order_status(
                order.order_id, "filled",
                filled_size=fill_size, avg_fill_price=fill_price,
            )
            self._db.insert_trade(TradeRecord(
                id=str(uuid.uuid4()),
                order_id=order.order_id,
                market_id=order.market_id,
                token_id=order.token_id,
                side=order.side,
                price=fill_price,
                size=fill_size,
                stake_usd=order.stake_usd,
                status="SIMULATED",
                dry_run=True,
                action_side=_paper_a,
                outcome_side=_paper_o,
            ))
            self._db.upsert_position(PositionRecord(
                market_id=order.market_id,
                token_id=order.token_id,
                direction=order.side,
                entry_price=fill_price,
                size=fill_size,
                stake_usd=order.stake_usd,
                current_price=fill_price,
                pnl=0.0,
                action_side=_paper_a,
                outcome_side=_paper_o,
            ))
            self._ws_feed.subscribe(order.token_id)
            log.info(
                "engine.paper_auto_fill",
                order_id=order.order_id[:8],
                market_id=order.market_id[:8],
            )

    async def _maybe_rebalance(self) -> None:
        """Check for portfolio drift and log rebalance signals."""
        interval = self.config.portfolio.rebalance_check_interval_minutes * 60
        now = time.time()
        if now - self._last_rebalance_check < interval:
            return

        self._last_rebalance_check = now
        if not self._positions:
            return

        try:
            signals = self.portfolio.check_rebalance(self._positions)
            if signals:
                for sig in signals:
                    log.warning(
                        "engine.rebalance_signal",
                        type=sig.signal_type,
                        urgency=sig.urgency,
                        description=sig.description,
                    )
                    if self._db:
                        self._db.insert_alert(
                            "warning",
                            f"⚖️ Rebalance: {sig.description}",
                            "risk",
                        )
        except Exception as e:
            log.warning("engine.rebalance_error", error=str(e))

    async def _maybe_scan_arbitrage(self, markets: list[Any]) -> None:
        """Scan for arbitrage opportunities across discovered markets."""
        interval = self.config.portfolio.rebalance_check_interval_minutes * 60
        now = time.time()
        if now - self._last_arbitrage_scan < interval:
            return

        self._last_arbitrage_scan = now
        if not markets:
            return

        try:
            from src.policy.arbitrage import (
                detect_arbitrage,
                detect_complementary_arb,
                detect_correlated_mispricings,
            )
            fee_bps = int(self.config.risk.transaction_fee_pct * 10000)
            opps = detect_arbitrage(markets, fee_bps=fee_bps)
            self._latest_arb_opportunities = opps
            if opps:
                actionable = [o for o in opps if o.is_actionable]
                log.info(
                    "engine.arbitrage_scan",
                    total=len(opps),
                    actionable=len(actionable),
                )
                for opp in actionable[:3]:
                    if self._db:
                        self._db.insert_alert(
                            "info",
                            f"🔀 Arb: {opp.description}",
                            "arbitrage",
                        )

            # Complementary arb (YES+NO < threshold)
            comp_opps = detect_complementary_arb(
                markets,
                threshold=self.config.arbitrage.complementary_threshold,
                fee_bps=fee_bps,
            )
            self._latest_complementary_arb = comp_opps
            for co in comp_opps[:3]:
                if co.is_actionable and self._db:
                    self._db.insert_alert(
                        "info",
                        f"🎯 Complementary arb: {co.question[:60]} "
                        f"(net profit: {co.net_profit:.4f})",
                        "arbitrage",
                    )

            # Correlated mispricings
            corr_opps = detect_correlated_mispricings(
                markets,
                min_divergence=self.config.arbitrage.correlated_min_divergence,
            )
            self._latest_correlated_mispricings = corr_opps
            for cm in corr_opps[:3]:
                if cm.is_actionable and self._db:
                    self._db.insert_alert(
                        "info",
                        f"📊 Correlated mispricing: {cm.explanation[:80]}",
                        "arbitrage",
                    )
        except Exception as e:
            log.warning("engine.arbitrage_scan_error", error=str(e))

    async def _maybe_scan_cross_platform_arb(
        self, markets: list[Any],
    ) -> None:
        """Scan for cross-platform arb opportunities (Polymarket vs Kalshi)."""
        if not self.config.arbitrage.enabled or not self._cross_platform_scanner:
            return

        interval = self.config.arbitrage.scan_interval_secs
        now = time.time()
        if now - self._last_cross_platform_scan < interval:
            return

        self._last_cross_platform_scan = now

        try:
            opps = await self._cross_platform_scanner.scan(markets)
            self._latest_cross_platform_opps = opps

            actionable = [o for o in opps if o.is_actionable]
            if actionable:
                log.info(
                    "engine.cross_platform_arb_found",
                    total=len(opps),
                    actionable=len(actionable),
                )

                # Auto-execute within position limits
                for opp in actionable:
                    if not self._cross_platform_scanner.check_position_limits():
                        log.info("engine.cross_platform_arb_position_limit_reached")
                        break

                    stake = min(
                        self.config.arbitrage.max_arb_position_usd,
                        self.config.risk.max_stake_per_market,
                    )
                    result = await self._cross_platform_scanner.execute_arb(
                        opp, stake,
                    )

                    if self._db:
                        self._db.insert_alert(
                            "info" if result.status == "both_filled" else "warning",
                            f"⚡ Cross-platform arb {result.status}: "
                            f"net_pnl=${result.net_pnl:.4f}",
                            "arbitrage",
                        )
        except Exception as e:
            log.warning("engine.cross_platform_arb_error", error=str(e))

    async def _maybe_scan_wallets(self) -> None:
        """Run wallet scanner if enabled and interval elapsed."""
        if not self.config.wallet_scanner.enabled:
            return

        interval = self.config.wallet_scanner.scan_interval_minutes * 60
        now = time.time()
        if now - self._last_wallet_scan < interval:
            return

        log.info("engine.wallet_scan_start")
        try:
            result = await self._wallet_scanner.scan()
            self._latest_scan_result = result
            self._last_wallet_scan = now

            # Persist to database
            if self._db:
                import sqlite3
                db_path = self.config.storage.sqlite_path
                conn = sqlite3.connect(db_path)
                try:
                    save_scan_result(conn, result)

                    # Phase 7: Quality scoring + timing snapshots
                    whale_cfg = self.config.wallet_scanner
                    if whale_cfg.whale_quality_scoring_enabled:
                        from src.analytics.whale_scorer import WhaleScorer
                        scorer = WhaleScorer(
                            conn,
                            lookback_days=whale_cfg.whale_quality_lookback_days,
                            timing_favorable_threshold=whale_cfg.whale_timing_favorable_threshold,
                        )
                        quality_scores = scorer.score_all(result.tracked_wallets)
                        result.quality_scores = quality_scores
                        scorer.save_scores(conn, quality_scores)

                        # Update pending 24h snapshots
                        def _price_from_scan(slug: str) -> float:
                            for sig in result.conviction_signals:
                                if sig.market_slug == slug:
                                    return sig.current_price
                            return 0.0
                        scorer.update_pending_snapshots(conn, _price_from_scan)

                        # Re-compute conviction with quality filter
                        qualified = scorer.get_top_percentile(
                            quality_scores, whale_cfg.whale_quality_min_percentile,
                        )
                        if qualified and hasattr(self._wallet_scanner, "_last_all_positions"):
                            qualified_addrs = {s.address for s in qualified}
                            quality_wts = {
                                s.address: s.composite_score / 100.0
                                for s in qualified
                            }
                            result.conviction_signals = (
                                self._wallet_scanner._compute_conviction(
                                    self._wallet_scanner._last_all_positions,
                                    result.scanned_at,
                                    qualified_addresses=qualified_addrs,
                                    quality_weights=quality_wts,
                                )
                            )
                            log.info(
                                "engine.whale_quality_filter",
                                total=len(quality_scores),
                                qualified=len(qualified),
                                signals=len(result.conviction_signals),
                            )
                finally:
                    conn.close()

            log.info(
                "engine.wallet_scan_complete",
                wallets=result.wallets_scanned,
                signals=len(result.conviction_signals),
                deltas=len(result.deltas),
            )
        except Exception as e:
            log.warning("engine.wallet_scan_error", error=str(e))

    def get_status(self) -> dict[str, Any]:
        dd_state = self.drawdown.state
        pr_report = self.portfolio.assess(self._positions)
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "live_trading": is_live_trading_enabled(),
            "drawdown": dd_state.to_dict(),
            "portfolio": pr_report.to_dict(),
            "last_cycle": (
                self._cycle_history[-1].to_dict()
                if self._cycle_history else None
            ),
            "positions": len(self._positions),
            "filter_stats": (
                self._last_filter_stats.__dict__
                if self._last_filter_stats else None
            ),
            "research_cache_size": self._research_cache.size(),
            "circuit_breakers": circuit_breakers.stats(),
            "arbitrage": {
                "enabled": self.config.arbitrage.enabled,
                "intra_platform_opportunities": len(self._latest_arb_opportunities),
                "cross_platform_opportunities": len(self._latest_cross_platform_opps),
                "complementary_arb": len(self._latest_complementary_arb),
                "correlated_mispricings": len(self._latest_correlated_mispricings),
                "active_arb_positions": (
                    len(self._cross_platform_scanner.active_positions)
                    if self._cross_platform_scanner else 0
                ),
            },
        }
