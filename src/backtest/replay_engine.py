"""Replay engine — runs the trading pipeline on historical resolved markets.

Pipeline per market (simplified version of the live engine):
  1. Load historical market
  2. Convert to GammaMarket for pipeline compatibility
  3. Build features with synthetic implied probability
  4. Build mock evidence (no web search)
  5. Run forecast via CachedEnsembleForecaster
  6. Calculate edge
  7. Size position (if edge found)
  8. Simulate execution (compare direction to known resolution)
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.backtest.cached_forecaster import CachedEnsembleForecaster
from src.backtest.database import BacktestDatabase
from src.backtest.llm_cache import LLMResponseCache
from src.backtest.mock_evidence import build_mock_evidence
from src.backtest.models import (
    BacktestRunRecord,
    BacktestTradeRecord,
    HistoricalMarketRecord,
)
from src.config import BotConfig
from src.connectors.polymarket_gamma import GammaMarket, GammaToken, classify_market_type
from src.forecast.feature_builder import MarketFeatures
from src.policy.edge_calc import EdgeResult, calculate_edge
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    run_id: str = ""
    config_name: str = ""

    # Aggregate metrics
    total_markets: int = 0
    markets_traded: int = 0
    markets_skipped: int = 0

    # P&L
    total_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0

    # Calibration
    brier_score: float = 0.0
    calibration_buckets: list[dict[str, Any]] = field(default_factory=list)

    # Per-category breakdown
    category_stats: list[dict[str, Any]] = field(default_factory=list)

    # Equity curve
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    # Trades list
    trades: list[BacktestTradeRecord] = field(default_factory=list)

    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_secs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "total_markets": self.total_markets,
            "markets_traded": self.markets_traded,
            "markets_skipped": self.markets_skipped,
            "total_pnl": round(self.total_pnl, 2),
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "brier_score": round(self.brier_score, 4),
            "calibration_buckets": self.calibration_buckets,
            "category_stats": self.category_stats,
            "equity_curve": self.equity_curve,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_secs": round(self.duration_secs, 2),
        }


@dataclass
class _RunState:
    """Internal state tracker during a backtest run."""

    bankroll: float = 10000.0
    equity: float = 10000.0
    peak_equity: float = 10000.0
    trades: list[BacktestTradeRecord] = field(default_factory=list)
    pnls: list[float] = field(default_factory=list)


class ReplayEngine:
    """Replays the trading pipeline on historical resolved markets."""

    def __init__(
        self,
        config: BotConfig,
        backtest_db: BacktestDatabase,
        cache: LLMResponseCache,
        force_cache_only: bool = False,
    ):
        self._config = config
        self._db = backtest_db
        self._cache = cache
        self._force_cache_only = force_cache_only

        self._forecaster = CachedEnsembleForecaster(
            cache=cache,
            ensemble_config=config.ensemble,
            forecast_config=config.forecasting,
            force_cache_only=force_cache_only,
        )

        # Phase 6: Optional realistic fill simulator
        self._fill_simulator = None
        if config.backtest.realistic_fills_enabled:
            from src.backtest.fill_simulator import FillSimulator, FillSimulationConfig
            sim_config = FillSimulationConfig.from_backtest_config(config.backtest)
            self._fill_simulator = FillSimulator(sim_config)

    async def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        min_volume: float = 0.0,
        category: str | None = None,
        max_markets: int = 0,
        name: str = "",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BacktestResult:
        """Execute the backtest."""
        run_id = str(uuid.uuid4())[:8]
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()

        # Fetch historical markets
        limit = max_markets if max_markets > 0 else 100000
        markets = self._db.get_historical_markets(
            start_date=start_date,
            end_date=end_date,
            min_volume=min_volume,
            category=category,
            limit=limit,
        )

        log.info(
            "replay.starting",
            run_id=run_id,
            markets=len(markets),
            start=start_date,
            end=end_date,
        )

        # Create run record
        run_record = BacktestRunRecord(
            run_id=run_id,
            name=name or f"backtest-{run_id}",
            config_json=self._config.model_dump_json(),
            status="running",
            start_date=start_date or "",
            end_date=end_date or "",
            started_at=started_at,
        )
        self._db.insert_backtest_run(run_record)

        state = _RunState(
            bankroll=self._config.risk.bankroll,
            equity=self._config.risk.bankroll,
            peak_equity=self._config.risk.bankroll,
        )

        for idx, hist_market in enumerate(markets):
            if progress_callback:
                progress_callback(idx + 1, len(markets), hist_market.question[:60])

            try:
                trade = await self._process_market(hist_market, state)
                if trade:
                    state.trades.append(trade)
                    state.pnls.append(trade.pnl)
                    state.equity += trade.pnl
                    state.peak_equity = max(state.peak_equity, state.equity)
                    self._db.insert_backtest_trade(trade)
            except Exception as e:
                log.warning(
                    "replay.market_error",
                    condition_id=hist_market.condition_id,
                    error=str(e),
                )

        completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        duration = time.time() - time.mktime(
            dt.datetime.fromisoformat(started_at).timetuple()
        ) if started_at else 0.0

        # Compute results
        result = self._compute_results(
            run_id=run_id,
            name=name or f"backtest-{run_id}",
            state=state,
            total_markets=len(markets),
            started_at=started_at,
            completed_at=completed_at,
            duration_secs=duration,
        )

        # Update run record
        self._db.update_backtest_run(run_id, {
            "status": "completed",
            "markets_processed": result.total_markets,
            "markets_traded": result.markets_traded,
            "total_pnl": result.total_pnl,
            "brier_score": result.brier_score,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
            "results_json": json.dumps(result.to_dict()),
            "completed_at": completed_at,
            "duration_secs": result.duration_secs,
        })

        # Store backtest Sharpe in main DB for preflight validation
        if result.sharpe_ratio != 0.0:
            try:
                from src.storage.database import Database
                main_db = Database(self._config.storage)
                main_db.connect()
                main_db.set_engine_state(
                    "last_backtest_sharpe",
                    str(round(result.sharpe_ratio, 4)),
                )
                main_db.close()
            except Exception as e:
                log.warning("backtest.sharpe_persist_error", error=str(e))

        log.info(
            "replay.complete",
            run_id=run_id,
            markets=result.total_markets,
            traded=result.markets_traded,
            pnl=round(result.total_pnl, 2),
            brier=round(result.brier_score, 4),
        )
        return result

    async def _process_market(
        self,
        hist_market: HistoricalMarketRecord,
        state: _RunState,
    ) -> BacktestTradeRecord | None:
        """Process a single historical market through the pipeline.

        Returns a trade record if a trade was taken, None if skipped.
        """
        # Build features
        implied_prob = self._config.backtest.default_implied_prob
        features = MarketFeatures(
            market_id=hist_market.condition_id,
            question=hist_market.question,
            market_type=hist_market.market_type or classify_market_type(
                hist_market.question, hist_market.category,
            ),
            volume_usd=hist_market.volume_usd,
            liquidity_usd=hist_market.liquidity_usd,
            implied_probability=implied_prob,
            category=hist_market.category,
            evidence_quality=self._config.backtest.mock_evidence_quality,
            num_sources=1,
        )

        # Build mock evidence
        evidence = build_mock_evidence(
            hist_market,
            quality_score=self._config.backtest.mock_evidence_quality,
        )

        # Run forecast
        ensemble_result = await self._forecaster.forecast(features, evidence)
        if ensemble_result.models_succeeded == 0:
            return None

        model_prob = ensemble_result.model_probability
        confidence = ensemble_result.confidence_level

        # Calculate edge
        edge_result = calculate_edge(
            implied_prob=implied_prob,
            model_prob=model_prob,
            transaction_fee_pct=self._config.risk.transaction_fee_pct,
            exit_fee_pct=self._config.risk.exit_fee_pct,
            hold_to_resolution=True,
        )

        # Check minimum edge
        min_edge = self._config.risk.min_edge
        if abs(edge_result.net_edge) < min_edge:
            return None

        # Determine direction
        direction = edge_result.direction  # "BUY_YES" or "BUY_NO"

        # Simple position sizing: use a fixed fraction of bankroll
        stake = min(
            state.bankroll * self._config.risk.kelly_fraction * 0.5,
            self._config.risk.max_stake_per_market,
        )
        if stake <= 0:
            return None

        # Simulate execution
        actual_outcome = 1.0 if hist_market.resolution == "YES" else 0.0
        entry_price = implied_prob if direction == "BUY_YES" else (1.0 - implied_prob)

        # Determine if forecast was correct
        if direction == "BUY_YES":
            forecast_correct = hist_market.resolution == "YES"
        else:
            forecast_correct = hist_market.resolution == "NO"

        # Phase 6: Use realistic fill simulator if enabled
        if self._fill_simulator is not None:
            sim_fill = self._fill_simulator.simulate(
                direction=direction,
                entry_price=entry_price,
                resolution=hist_market.resolution,
                stake_usd=stake,
                available_liquidity_usd=hist_market.liquidity_usd,
            )
            return BacktestTradeRecord(
                run_id="",
                market_condition_id=hist_market.condition_id,
                question=hist_market.question,
                category=hist_market.category,
                direction=direction,
                model_probability=model_prob,
                implied_probability=implied_prob,
                edge=edge_result.net_edge,
                confidence_level=confidence,
                stake_usd=sim_fill.filled_size_usd,
                entry_price=sim_fill.entry_price,
                exit_price=sim_fill.exit_price,
                pnl=sim_fill.pnl,
                resolution=hist_market.resolution,
                actual_outcome=actual_outcome,
                forecast_correct=forecast_correct,
                created_at=hist_market.resolved_at,
                slippage_bps=sim_fill.slippage_bps,
                fill_rate=sim_fill.fill_rate,
                simulated_impact_pct=sim_fill.price_impact_pct,
                fill_delay_ms=sim_fill.fill_delay_ms,
                fee_paid_usd=sim_fill.fee_paid_usd,
            )

        # Default: flat slippage model
        slippage = self._config.backtest.default_slippage_pct
        exit_price, pnl = self._simulate_execution(
            direction=direction,
            entry_price=entry_price,
            resolution=hist_market.resolution,
            stake_usd=stake,
            slippage_pct=slippage,
        )

        return BacktestTradeRecord(
            run_id="",
            market_condition_id=hist_market.condition_id,
            question=hist_market.question,
            category=hist_market.category,
            direction=direction,
            model_probability=model_prob,
            implied_probability=implied_prob,
            edge=edge_result.net_edge,
            confidence_level=confidence,
            stake_usd=stake,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            resolution=hist_market.resolution,
            actual_outcome=actual_outcome,
            forecast_correct=forecast_correct,
            created_at=hist_market.resolved_at,
        )

    @staticmethod
    def _simulate_execution(
        direction: str,
        entry_price: float,
        resolution: str,
        stake_usd: float,
        slippage_pct: float = 0.005,
    ) -> tuple[float, float]:
        """Simulate trade execution.

        Returns (exit_price, pnl).
        """
        entry_with_slip = entry_price * (1 + slippage_pct)
        entry_with_slip = min(entry_with_slip, 0.99)

        if direction == "BUY_YES":
            exit_price = 1.0 if resolution == "YES" else 0.0
        else:
            exit_price = 1.0 if resolution == "NO" else 0.0

        if entry_with_slip <= 0:
            return exit_price, 0.0

        # Number of contracts = stake / entry_price
        contracts = stake_usd / entry_with_slip
        pnl = (exit_price - entry_with_slip) * contracts

        return exit_price, round(pnl, 2)

    def _compute_results(
        self,
        run_id: str,
        name: str,
        state: _RunState,
        total_markets: int,
        started_at: str,
        completed_at: str,
        duration_secs: float,
    ) -> BacktestResult:
        """Compute aggregate results from all trades."""
        trades = state.trades
        pnls = state.pnls

        # Set run_id on all trades
        for t in trades:
            t.run_id = run_id

        result = BacktestResult(
            run_id=run_id,
            config_name=name,
            total_markets=total_markets,
            markets_traded=len(trades),
            markets_skipped=total_markets - len(trades),
            started_at=started_at,
            completed_at=completed_at,
            duration_secs=duration_secs,
            trades=trades,
        )

        if not trades:
            return result

        # P&L metrics
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        result.total_pnl = sum(pnls)
        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.win_rate = len(wins) / len(pnls) if pnls else 0.0
        result.avg_win = sum(wins) / len(wins) if wins else 0.0
        result.avg_loss = sum(losses) / len(losses) if losses else 0.0
        result.largest_win = max(wins) if wins else 0.0
        result.largest_loss = min(losses) if losses else 0.0
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = (
            total_wins / total_losses if total_losses > 0 else float("inf")
        )

        # Risk metrics
        risk = self._compute_risk_metrics(pnls, state.bankroll)
        result.sharpe_ratio = risk["sharpe"]
        result.sortino_ratio = risk["sortino"]
        result.max_drawdown_pct = risk["max_drawdown_pct"]
        result.calmar_ratio = risk["calmar"]

        # Calibration
        result.brier_score = self._compute_brier_score(trades)
        result.calibration_buckets = self._compute_calibration_buckets(trades)

        # Category breakdown
        result.category_stats = self._compute_category_stats(trades)

        # Equity curve
        result.equity_curve = self._compute_equity_curve(trades, state.bankroll)

        return result

    @staticmethod
    def _compute_brier_score(trades: list[BacktestTradeRecord]) -> float:
        """Compute Brier score: mean of (forecast_prob - actual_outcome)^2.

        For BUY_YES trades, forecast_prob = model_probability.
        For BUY_NO trades, forecast_prob = 1 - model_probability.
        """
        if not trades:
            return 0.0
        total = 0.0
        for t in trades:
            # Use the probability relevant to the YES outcome
            forecast_yes_prob = t.model_probability
            actual = t.actual_outcome
            total += (forecast_yes_prob - actual) ** 2
        return total / len(trades)

    @staticmethod
    def _compute_calibration_buckets(
        trades: list[BacktestTradeRecord],
        n_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """Bin forecasts by probability, compare avg to actual rate."""
        if not trades:
            return []

        bin_width = 1.0 / n_bins
        buckets: list[dict[str, Any]] = []

        for i in range(n_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width
            in_bin = [
                t for t in trades
                if bin_start <= t.model_probability < bin_end
                or (i == n_bins - 1 and t.model_probability == 1.0)
            ]
            if not in_bin:
                continue
            avg_forecast = sum(t.model_probability for t in in_bin) / len(in_bin)
            actual_rate = sum(t.actual_outcome for t in in_bin) / len(in_bin)
            buckets.append({
                "bin_start": round(bin_start, 2),
                "bin_end": round(bin_end, 2),
                "avg_forecast": round(avg_forecast, 4),
                "actual_rate": round(actual_rate, 4),
                "count": len(in_bin),
            })

        return buckets

    @staticmethod
    def _compute_equity_curve(
        trades: list[BacktestTradeRecord],
        bankroll: float,
    ) -> list[dict[str, Any]]:
        """Build equity curve from trade list."""
        curve: list[dict[str, Any]] = []
        cumulative = 0.0
        for i, t in enumerate(trades):
            cumulative += t.pnl
            curve.append({
                "market_index": i,
                "question": t.question[:60],
                "pnl": round(t.pnl, 2),
                "cumulative_pnl": round(cumulative, 2),
                "equity": round(bankroll + cumulative, 2),
            })
        return curve

    @staticmethod
    def _compute_risk_metrics(
        pnls: list[float],
        bankroll: float,
    ) -> dict[str, float]:
        """Compute Sharpe, Sortino, max drawdown, Calmar."""
        if not pnls or len(pnls) < 2:
            return {
                "sharpe": 0.0, "sortino": 0.0,
                "max_drawdown_pct": 0.0, "calmar": 0.0,
            }

        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Sharpe (annualize roughly: assume ~250 trades/year)
        sharpe = (mean_pnl / std) if std > 0 else 0.0

        # Sortino (downside deviation only)
        downside = [p for p in pnls if p < 0]
        if downside:
            down_var = sum(p ** 2 for p in downside) / len(pnls)
            down_std = math.sqrt(down_var)
            sortino = (mean_pnl / down_std) if down_std > 0 else 0.0
        else:
            sortino = float("inf") if mean_pnl > 0 else 0.0

        # Max drawdown
        equity = bankroll
        peak = bankroll
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Calmar
        total_return = sum(pnls) / bankroll if bankroll > 0 else 0.0
        calmar = (total_return / max_dd) if max_dd > 0 else 0.0

        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "calmar": round(calmar, 4),
        }

    @staticmethod
    def _compute_category_stats(
        trades: list[BacktestTradeRecord],
    ) -> list[dict[str, Any]]:
        """Compute per-category statistics."""
        by_cat: dict[str, list[BacktestTradeRecord]] = {}
        for t in trades:
            cat = t.category or "UNKNOWN"
            by_cat.setdefault(cat, []).append(t)

        stats: list[dict[str, Any]] = []
        for cat, cat_trades in sorted(by_cat.items()):
            pnls = [t.pnl for t in cat_trades]
            wins = [p for p in pnls if p > 0]
            stats.append({
                "category": cat,
                "total_trades": len(cat_trades),
                "wins": len(wins),
                "losses": len(pnls) - len(wins),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0.0,
            })
        return stats
