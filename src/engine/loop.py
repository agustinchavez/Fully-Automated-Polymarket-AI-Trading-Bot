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
from typing import TYPE_CHECKING, Any, Callable

from src.config import BotConfig, load_config, is_live_trading_enabled

if TYPE_CHECKING:
    from src.connectors.polymarket_gamma import GammaMarket
    from src.engine.market_classifier import MarketClassification
    from src.execution.exit_finalizer import ExitFinalizer
    from src.execution.fill_tracker import FillTracker
    from src.execution.plan_controller import PlanController
    from src.forecast.feature_builder import MarketFeatures
    from src.forecast.llm_forecaster import ForecastResult
    from src.forecast.specialist_router import SpecialistRouter
    from src.forecast.specialists.base import SpecialistResult
    from src.policy.cross_platform_arb import CrossPlatformArbScanner
    from src.policy.edge_calc import EdgeResult
    from src.policy.position_sizer import PositionSize
    from src.policy.risk_limits import RiskCheckResult
    from src.research.evidence_extractor import EvidencePackage
    from src.research.source_fetcher import FetchedSource
    from src.storage.audit import AuditTrail
    from src.storage.database import Database

from src.engine.event_monitor import EventMonitor
from src.engine.pipeline import PipelineRunner
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
    market: GammaMarket
    cycle_id: int
    market_id: str = ""
    question: str = ""
    classification: MarketClassification | None = None
    sources: list[FetchedSource] = field(default_factory=list)
    evidence: EvidencePackage | None = None
    features: MarketFeatures | None = None
    forecast: ForecastResult | SpecialistResult | None = None
    edge_result: EdgeResult | None = None
    has_edge: bool = False
    risk_result: RiskCheckResult | None = None
    position: PositionSize | None = None
    whale_converged: bool = False   # True when whale signal agrees with model edge
    _signal_stack: Any = None       # Phase 2 SignalStack from signal_aggregator
    result: dict[str, Any] = field(default_factory=lambda: {
        "has_edge": False, "trade_attempted": False, "trade_executed": False,
    })


class TradingEngine:
    """Continuous trading engine that coordinates all bot components."""

    def __init__(self, config: BotConfig | None = None):
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
        self._latest_scan_result: dict | None = None

        # ── WebSocket price feed ──
        self._ws_feed = WebSocketFeed()
        self._ws_task: asyncio.Task[None] | None = None

        # ── Rebalance / Arbitrage tracking ──
        self._last_rebalance_check: float = 0.0
        self._last_arbitrage_scan: float = 0.0
        self._latest_arb_opportunities: list[Any] = []

        # ── Phase 5: Cross-Platform Arbitrage ──
        self._cross_platform_scanner: CrossPlatformArbScanner | None = None
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
        self._db: Database | None = None
        self._audit: AuditTrail | None = None
        self._alert_manager: Any = None

        # Phase 9: Daily summary tracking
        self._last_daily_summary_date: str = ""
        self._last_weekly_digest_week: str = ""

        # Phase 9: Graduated deployment + bots (optional deps — keep Any)
        self._deployment_manager: Any = None
        self._telegram_bot: Any = None
        self._telegram_task: asyncio.Task[None] | None = None
        self._discord_bot: Any = None
        self._discord_task: asyncio.Task[None] | None = None
        self._slack_bot: Any = None
        self._slack_task: asyncio.Task[None] | None = None

        # Phase 10: Reconciliation loop
        self._reconciliation_task: asyncio.Task[None] | None = None
        self._reconciliation_stop: asyncio.Event | None = None

        # Phase 10B: Shared exit finalizer (initialised in start())
        self._exit_finalizer: ExitFinalizer | None = None

        # Phase 10E: Execution plan controller (initialised in start())
        self._plan_controller: PlanController | None = None

        # Phase 6: Execution fill tracker (optional)
        self._fill_tracker: FillTracker | None = None
        if self.config.execution.auto_strategy_selection_enabled:
            from src.execution.fill_tracker import FillTracker
            self._fill_tracker = FillTracker()

        # ── Phase 4: Specialist Router ──
        self._specialist_router: SpecialistRouter | None = None
        if self.config.specialists.enabled:
            try:
                from src.forecast.specialist_router import SpecialistRouter
                self._specialist_router = SpecialistRouter(self.config.specialists)
            except Exception as e:
                log.warning("engine.specialist_router_init_error", error=str(e))

        # ── Event Monitor (triggers re-research on price/volume/whale changes) ──
        self._event_monitor = EventMonitor()

        # ── Phase 12: Pipeline Runner (initialized in start() after DB) ──
        self._pipeline: PipelineRunner | None = None

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

            # Compute paper Sharpe from daily summaries (every 10 cycles)
            if self._cycle_count % 10 == 0:
                self._compute_paper_sharpe()
        except Exception as e:
            log.warning("engine.persist_state_error", error=str(e))

    def _compute_paper_sharpe(self) -> None:
        """Compute paper Sharpe from daily summaries and store in engine_state."""
        if not self._db:
            return
        try:
            rows = self._db.conn.execute(
                "SELECT total_pnl FROM daily_summaries "
                "ORDER BY summary_date DESC LIMIT 30"
            ).fetchall()
            if len(rows) >= 7:  # need at least a week of data
                import math
                pnls = [r["total_pnl"] for r in rows]
                mean_pnl = sum(pnls) / len(pnls)
                variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
                std = math.sqrt(variance) if variance > 0 else 0.0
                paper_sharpe = (mean_pnl / std) if std > 0 else 0.0
                self._db.set_engine_state(
                    "paper_sharpe", str(round(paper_sharpe, 4)),
                )
        except Exception:
            pass  # daily_summaries table may not exist yet

    async def start(self) -> None:
        self._running = True
        interval = self.config.engine.cycle_interval_secs
        self._init_db()
        self._restore_kill_switch_state()

        # Compute paper Sharpe at startup to avoid stale data
        self._compute_paper_sharpe()

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

        # ── Phase 12: Initialize pipeline runner ──
        self._pipeline = PipelineRunner(
            config=self.config,
            db=self._db,
            audit=self._audit,
            drawdown=self.drawdown,
            calibration_loop=self._calibration_loop,
            adaptive_weighter=self._adaptive_weighter,
            smart_entry=self._smart_entry,
            specialist_router=self._specialist_router,
            fill_tracker=self._fill_tracker,
            plan_controller=self._plan_controller,
            exit_finalizer=self._exit_finalizer,
            current_regime=self._current_regime,
            ws_feed=self._ws_feed,
            wallet_scanner=self._wallet_scanner,
            positions=self._positions,
            latest_scan_result=self._latest_scan_result,
        )

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

        # Discord kill bot
        if prod.discord_kill_enabled and prod.discord_kill_token:
            try:
                from src.observability.discord_bot import DiscordKillBot

                self._discord_bot = DiscordKillBot(
                    token=prod.discord_kill_token,
                    channel_id=prod.discord_kill_channel_id,
                    engine=self,
                )
                self._discord_task = asyncio.create_task(
                    self._discord_bot.start()
                )
                log.info("engine.discord_bot_started")
            except Exception as e:
                log.warning("engine.discord_bot_start_error", error=str(e))

        # Slack kill bot
        if prod.slack_kill_enabled and prod.slack_kill_bot_token:
            try:
                from src.observability.slack_bot import SlackKillBot

                self._slack_bot = SlackKillBot(
                    bot_token=prod.slack_kill_bot_token,
                    app_token=prod.slack_kill_app_token,
                    channel_id=prod.slack_kill_channel_id,
                    engine=self,
                )
                self._slack_task = asyncio.create_task(
                    self._slack_bot.start()
                )
                log.info("engine.slack_bot_started")
            except Exception as e:
                log.warning("engine.slack_bot_start_error", error=str(e))

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

        # Close shared research infrastructure (httpx clients, etc.)
        if self._pipeline:
            try:
                await self._pipeline.close()
            except Exception:
                log.warning("engine.pipeline_close_error", exc_info=True)

        if self._db:
            self._db.insert_alert("info", "\U0001f6d1 Trading engine stopped", "system")
            self._persist_engine_state({"running": False})

    def stop(self) -> None:
        log.info("engine.stop_requested")
        self._running = False
        # Stop WebSocket feed
        if self._ws_task and not self._ws_task.done():
            asyncio.ensure_future(self._ws_feed.stop())
        # Stop bots
        if self._telegram_bot:
            self._telegram_bot.stop()
        if self._telegram_task and not self._telegram_task.done():
            self._telegram_task.cancel()
        if self._discord_bot:
            self._discord_bot.stop()
        if self._discord_task and not self._discord_task.done():
            self._discord_task.cancel()
        if self._slack_bot:
            self._slack_bot.stop()
        if self._slack_task and not self._slack_task.done():
            self._slack_task.cancel()
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

    async def _maybe_send_weekly_digest(self) -> None:
        """Send weekly digest at the configured day/hour via AlertManager."""
        if not self._db:
            return
        digest_cfg = getattr(self.config, "digest", None)
        if not digest_cfg or not digest_cfg.enabled:
            return

        import datetime as _dt

        now = _dt.datetime.now(_dt.timezone.utc)

        # Check day of week
        day_map = {
            "mon": 0, "tue": 1, "wed": 2, "thu": 3,
            "fri": 4, "sat": 5, "sun": 6,
        }
        target_day = day_map.get(digest_cfg.schedule_day_of_week, 0)
        if now.weekday() != target_day:
            return
        if now.hour != digest_cfg.schedule_hour:
            return

        # Already sent this week
        week_key = now.strftime("%Y-W%W")
        if self._last_weekly_digest_week == week_key:
            return

        try:
            from src.observability.reports import WeeklyDigestGenerator

            gen = WeeklyDigestGenerator(
                conn=self._db.conn,
                bankroll=self.config.risk.bankroll,
                transaction_fee_pct=self.config.risk.transaction_fee_pct,
            )
            digest = gen.generate(days=digest_cfg.lookback_days)
            if self._alert_manager:
                await gen.send_via_alert_manager(digest, self._alert_manager)
            self._last_weekly_digest_week = week_key
            log.info("engine.weekly_digest_sent", week=week_key)
        except Exception as e:
            log.warning("engine.weekly_digest_error", error=str(e))

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

            # Sync mutable state to pipeline runner before processing
            if self._pipeline:
                self._pipeline._current_regime = self._current_regime
                self._pipeline._positions = self._positions
                self._pipeline._latest_scan_result = self._latest_scan_result

            # ── Per-cycle dedup: skip duplicate market IDs ──────────
            seen_ids: set[str] = set()
            deduped: list[Any] = []
            for c in filtered:
                mid = getattr(c, "id", "")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    deduped.append(c)
            if len(deduped) < len(filtered):
                log.info(
                    "engine.cycle_dedup",
                    original=len(filtered),
                    deduped=len(deduped),
                )
            filtered = deduped

            # ── Phase 1: Research + Forecast (parallel) ─────────────
            # These stages are read-only and stateless per candidate.
            # Running them concurrently cuts cycle time by ~N×.
            phase1_tasks = [
                asyncio.wait_for(
                    self._run_phase1(candidate, cycle.cycle_id),
                    timeout=300,
                )
                for candidate in filtered
            ]
            phase1_results = await asyncio.gather(
                *phase1_tasks, return_exceptions=True,
            )

            # ── Phase 2: Risk + Execution (sequential) ────────────
            # These stages read/write shared portfolio state.
            for candidate, p1 in zip(filtered, phase1_results):
                mid = getattr(candidate, "id", "?")
                self._research_cache.mark_researched(
                    getattr(candidate, "id", ""),
                )

                if isinstance(p1, asyncio.TimeoutError):
                    log.warning("engine.candidate_timeout", market_id=mid)
                    cycle.errors.append(f"Timeout: {mid[:8]}")
                    continue
                if isinstance(p1, BaseException):
                    log.error("engine.candidate_error", market_id=mid,
                              error=str(p1))
                    cycle.errors.append(str(p1))
                    traceback.print_exc()
                    continue

                ctx = p1  # PipelineContext from Phase 1
                if ctx is None:
                    # Phase 1 returned None → early exit (duplicate/skip)
                    continue

                try:
                    result = await self._run_phase2(ctx, cycle.cycle_id)
                    if result.get("has_edge"):
                        cycle.edges_found += 1
                    if result.get("trade_attempted"):
                        cycle.trades_attempted += 1
                    if result.get("trade_executed"):
                        cycle.trades_executed += 1
                except Exception as e:
                    log.error("engine.candidate_error", market_id=mid,
                              error=str(e))
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

            # ── Weekly Digest via AlertManager ─────────────────────
            await self._maybe_send_weekly_digest()

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

    async def _run_phase1(
        self, market: Any, cycle_id: int,
    ) -> PipelineContext | None:
        """Phase 1: classify → research → features → forecast.

        Stateless per candidate — safe to run concurrently.
        Returns a PipelineContext ready for Phase 2, or None if
        the candidate should be skipped.
        """
        ctx = PipelineContext(market=market, cycle_id=cycle_id,
                              market_id=market.id, question=market.question)

        # ── Early exit: skip if we already hold a position or pending order
        if self._db:
            existing = [p for p in (self._db.get_open_positions())
                        if p.market_id == market.id]
            if existing:
                log.info("engine.duplicate_skip", market_id=market.id[:8],
                         msg="Already have open position — skipping")
                return None

            try:
                if self._db.has_active_order_for_market(market.id):
                    log.info("engine.duplicate_order_skip", market_id=market.id[:8],
                             msg="Already have pending order — skipping")
                    return None
            except Exception:
                pass

        pipeline = self._pipeline

        # ── Stage 0: Classification
        pipeline.stage_classify(ctx)

        # ── Stage 1: Research
        ok = await pipeline.stage_research(ctx)
        if not ok:
            return None

        # ── Stage 2: Build Features
        from src.forecast.feature_builder import build_features
        ctx.features = build_features(market=market, evidence=ctx.evidence)

        # ── Stage 3: Forecast
        forecast_ok = await pipeline.stage_forecast(ctx)
        if not forecast_ok:
            pipeline._log_candidate(ctx.cycle_id, ctx.market, decision="SKIP",
                                    reason="Forecast failed or circuit open",
                                    classification=ctx.classification)
            return None

        return ctx

    async def _run_phase2(
        self, ctx: PipelineContext, cycle_id: int,
    ) -> dict[str, Any]:
        """Phase 2: calibrate → edge → risk → execute.

        Reads/writes shared portfolio state — must run sequentially.
        """
        pipeline = self._pipeline
        market = ctx.market

        # ── Stage 3b: Apply Calibration
        pipeline.stage_calibrate(ctx)

        # ── Stage 4: Edge Calculation + Whale Adjustment
        pipeline.stage_edge_calc(ctx)

        # ── Stage 4b: Edge Uncertainty Adjustment
        pipeline.stage_uncertainty_adjustment(ctx)
        ctx.result["has_edge"] = ctx.has_edge

        # ── Stage 5: Risk Checks
        pipeline.stage_risk_checks(ctx)

        # ── Persist forecast to DB
        pipeline.stage_persist_forecast(ctx)

        # ── Portfolio Correlation Check
        pipeline.stage_correlation_check(ctx)

        # ── Portfolio VaR Gate
        pipeline.stage_var_gate(ctx)

        # ── UMA Dispute Check
        await pipeline.stage_uma_check(ctx)

        # ── Decision Gate
        if not ctx.risk_result.allowed:
            log.info("engine.no_trade", market_id=ctx.market_id,
                     violations=ctx.risk_result.violations)
            pipeline._log_candidate(
                cycle_id, market, forecast=ctx.forecast, evidence=ctx.evidence,
                edge_result=ctx.edge_result, decision="NO TRADE",
                reason="; ".join(ctx.risk_result.violations),
                classification=ctx.classification,
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

        # ── Stage 6: Position Sizing
        pipeline.stage_position_sizing(ctx)
        if ctx.position is None:
            return ctx.result

        ctx.result["trade_attempted"] = True

        # ── Stage 7: Build & Route Order
        await pipeline.stage_execute_order(ctx)

        # ── Stage 8: Audit + Log
        pipeline.stage_audit_and_log(ctx)

        return ctx.result

    async def _process_candidate(self, market: Any, cycle_id: int) -> dict[str, Any]:
        """Process a single market through the full research-to-trade pipeline.

        Backward-compatible wrapper that runs Phase 1 + Phase 2 sequentially.
        The main cycle loop uses the parallel Phase 1 / sequential Phase 2
        split directly.
        """
        ctx = await self._run_phase1(market, cycle_id)
        if ctx is None:
            return {"has_edge": False, "trade_attempted": False, "trade_executed": False}
        return await self._run_phase2(ctx, cycle_id)

    # ── Pipeline stage methods — moved to src/engine/pipeline.py ──────
    # Thin delegation wrappers kept for backward compatibility (tests, _finalize_exit).

    def _record_performance_log(
        self, pos: Any, exit_price: float, pnl: float, mkt_record: Any = None,
    ) -> None:
        if self._pipeline:
            self._pipeline.record_performance_log(pos, exit_price, pnl, mkt_record)

    def _maybe_run_post_mortem(self, market_id: str) -> None:
        if self._pipeline:
            self._pipeline.maybe_run_post_mortem(market_id)

    def _record_order_result(self, ctx: Any, order: Any, order_result: Any,
                             edge_result: Any, position: Any, token_id: str,
                             parent_plan_id: str = "", child_index: int = 0) -> None:
        """Delegate to PipelineRunner (backward compat for tests)."""
        runner = self._pipeline or self._make_ephemeral_pipeline()
        runner._record_order_result(
            ctx, order, order_result, edge_result, position, token_id,
            parent_plan_id, child_index,
        )

    async def _stage_execute_order(self, ctx: Any) -> None:
        """Delegate to PipelineRunner (backward compat for tests)."""
        runner = self._pipeline or self._make_ephemeral_pipeline()
        await runner.stage_execute_order(ctx)

    def _log_candidate(self, cycle_id: int, market: Any, **kwargs: Any) -> None:
        """Delegate to PipelineRunner (backward compat)."""
        runner = self._pipeline or self._make_ephemeral_pipeline()
        runner._log_candidate(cycle_id, market, **kwargs)

    def _make_ephemeral_pipeline(self) -> PipelineRunner:
        """Create a PipelineRunner on the fly (for tests that skip start())."""
        return PipelineRunner(
            config=self.config,
            db=self._db,
            audit=self._audit,
            drawdown=self.drawdown,
            calibration_loop=self._calibration_loop,
            adaptive_weighter=self._adaptive_weighter,
            smart_entry=self._smart_entry,
            specialist_router=self._specialist_router,
            fill_tracker=self._fill_tracker,
            plan_controller=self._plan_controller,
            exit_finalizer=self._exit_finalizer,
            current_regime=self._current_regime,
            ws_feed=self._ws_feed,
            wallet_scanner=self._wallet_scanner,
            positions=self._positions,
            latest_scan_result=self._latest_scan_result,
        )

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

                    # ── Event Monitor: check for re-research triggers ──
                    trigger = self._event_monitor.check_price_move(
                        pos.market_id, current_price,
                    )
                    if trigger:
                        self._research_cache.invalidate(pos.market_id)

                    # Fetch market metadata (needed for snapshots + exit trades)
                    if market is None:
                        market = await client.get_market(pos.market_id)
                    mkt_record = self._db.get_market(pos.market_id)

                    # ── Event Monitor: volume, resolution, whale triggers ──
                    try:
                        mkt_volume = float(getattr(market, "volume", 0) or 0)
                        if mkt_volume > 0:
                            vol_trigger = self._event_monitor.check_volume_spike(
                                pos.market_id, mkt_volume,
                            )
                            if vol_trigger:
                                self._research_cache.invalidate(pos.market_id)
                    except (TypeError, ValueError):
                        pass

                    try:
                        mkt_end = getattr(market, "end_date", None)
                        if mkt_end is not None and hasattr(mkt_end, "timestamp"):
                            import datetime as _dt
                            now = _dt.datetime.now(_dt.timezone.utc)
                            if mkt_end.tzinfo is None:
                                mkt_end = mkt_end.replace(tzinfo=_dt.timezone.utc)
                            hours_remaining = (mkt_end - now).total_seconds() / 3600
                            if hours_remaining > 0:
                                res_trigger = self._event_monitor.check_resolution_approaching(
                                    pos.market_id, hours_remaining,
                                )
                                if res_trigger:
                                    self._research_cache.invalidate(pos.market_id)
                    except (TypeError, ValueError, AttributeError):
                        pass

                    try:
                        if self._latest_scan_result and isinstance(self._latest_scan_result, dict):
                            whale_signals = self._latest_scan_result.get("signals", [])
                            for ws in whale_signals:
                                if ws.get("market_id") == pos.market_id:
                                    whale_trigger = self._event_monitor.check_whale_activity(
                                        pos.market_id,
                                        whale_count=ws.get("whale_count", 0),
                                        whale_volume_pct=ws.get("whale_volume_pct", 0.0),
                                    )
                                    if whale_trigger:
                                        self._research_cache.invalidate(pos.market_id)
                                    break
                    except (TypeError, ValueError):
                        pass

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
