"""Tests for replay engine (Phase 1)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.backtest.database import BacktestDatabase
from src.backtest.llm_cache import LLMResponseCache
from src.backtest.models import BacktestTradeRecord, HistoricalMarketRecord
from src.backtest.replay_engine import BacktestResult, ReplayEngine
from src.config import BotConfig
from src.forecast.ensemble import ModelForecast

import warnings


@pytest.fixture
def db() -> BacktestDatabase:
    bdb = BacktestDatabase(db_path=":memory:")
    bdb.connect()
    yield bdb
    bdb.close()


@pytest.fixture
def config() -> BotConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BotConfig()


@pytest.fixture
def cache(db: BacktestDatabase) -> LLMResponseCache:
    return LLMResponseCache(db, template_version="v1")


def _seed_markets(db: BacktestDatabase, count: int = 5) -> list[HistoricalMarketRecord]:
    """Seed the database with historical markets."""
    markets = []
    for i in range(count):
        resolution = "YES" if i % 2 == 0 else "NO"
        m = HistoricalMarketRecord(
            condition_id=f"market-{i}",
            question=f"Will event {i} happen?",
            description=f"Description for event {i}.",
            category="Test",
            market_type="UNKNOWN",
            resolution=resolution,
            resolved_at=f"2024-{(i % 12) + 1:02d}-15T00:00:00Z",
            volume_usd=5000.0 + i * 1000,
        )
        db.upsert_historical_market(m)
        markets.append(m)
    return markets


def _seed_cache(
    cache: LLMResponseCache,
    questions: list[str],
    models: list[str],
    prob: float = 0.7,
) -> None:
    """Pre-seed the LLM cache for deterministic testing."""
    for q in questions:
        for m in models:
            forecast = ModelForecast(
                model_name=m,
                model_probability=prob,
                confidence_level="MEDIUM",
                reasoning="Test cached reasoning",
            )
            cache.put(q, m, forecast)


# ── Execution Simulation ─────────────────────────────────────────────


class TestSimulateExecution:

    def test_buy_yes_wins(self) -> None:
        exit_p, pnl = ReplayEngine._simulate_execution(
            direction="BUY_YES", entry_price=0.50,
            resolution="YES", stake_usd=100.0, slippage_pct=0.0,
        )
        assert exit_p == 1.0
        assert pnl == 100.0  # (1.0 - 0.5) * (100 / 0.5) = 100

    def test_buy_yes_loses(self) -> None:
        exit_p, pnl = ReplayEngine._simulate_execution(
            direction="BUY_YES", entry_price=0.50,
            resolution="NO", stake_usd=100.0, slippage_pct=0.0,
        )
        assert exit_p == 0.0
        assert pnl == -100.0  # (0.0 - 0.5) * (100 / 0.5) = -100

    def test_buy_no_wins(self) -> None:
        exit_p, pnl = ReplayEngine._simulate_execution(
            direction="BUY_NO", entry_price=0.50,
            resolution="NO", stake_usd=100.0, slippage_pct=0.0,
        )
        assert exit_p == 1.0
        assert pnl == 100.0

    def test_buy_no_loses(self) -> None:
        exit_p, pnl = ReplayEngine._simulate_execution(
            direction="BUY_NO", entry_price=0.50,
            resolution="YES", stake_usd=100.0, slippage_pct=0.0,
        )
        assert exit_p == 0.0
        assert pnl == -100.0

    def test_slippage_reduces_profit(self) -> None:
        _, pnl_no_slip = ReplayEngine._simulate_execution(
            direction="BUY_YES", entry_price=0.50,
            resolution="YES", stake_usd=100.0, slippage_pct=0.0,
        )
        _, pnl_with_slip = ReplayEngine._simulate_execution(
            direction="BUY_YES", entry_price=0.50,
            resolution="YES", stake_usd=100.0, slippage_pct=0.05,
        )
        assert pnl_with_slip < pnl_no_slip

    def test_zero_entry_price(self) -> None:
        """Zero entry price should not crash."""
        exit_p, pnl = ReplayEngine._simulate_execution(
            direction="BUY_YES", entry_price=0.0,
            resolution="YES", stake_usd=100.0, slippage_pct=0.0,
        )
        # With slippage=0 and entry=0, entry_with_slip=0, returns 0 pnl
        assert pnl == 0.0


# ── Brier Score ───────────────────────────────────────────────────────


class TestBrierScore:

    def test_perfect_prediction(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m",
                model_probability=1.0, actual_outcome=1.0,
            ),
        ]
        assert ReplayEngine._compute_brier_score(trades) == 0.0

    def test_worst_prediction(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m",
                model_probability=1.0, actual_outcome=0.0,
            ),
        ]
        assert ReplayEngine._compute_brier_score(trades) == 1.0

    def test_50_50_prediction(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m",
                model_probability=0.5, actual_outcome=1.0,
            ),
        ]
        assert ReplayEngine._compute_brier_score(trades) == 0.25

    def test_empty_trades(self) -> None:
        assert ReplayEngine._compute_brier_score([]) == 0.0

    def test_multiple_trades(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m1",
                model_probability=0.8, actual_outcome=1.0,
            ),
            BacktestTradeRecord(
                run_id="r", market_condition_id="m2",
                model_probability=0.2, actual_outcome=0.0,
            ),
        ]
        # (0.8-1.0)^2 + (0.2-0.0)^2 = 0.04 + 0.04 = 0.08 / 2 = 0.04
        assert ReplayEngine._compute_brier_score(trades) == pytest.approx(0.04)


# ── Calibration Buckets ──────────────────────────────────────────────


class TestCalibrationBuckets:

    def test_basic_buckets(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id=f"m{i}",
                model_probability=0.15, actual_outcome=float(i % 5 == 0),
            )
            for i in range(10)
        ]
        buckets = ReplayEngine._compute_calibration_buckets(trades)
        assert len(buckets) >= 1
        # All trades in 0.1-0.2 bucket
        b = buckets[0]
        assert b["count"] == 10
        assert b["bin_start"] == 0.1

    def test_empty_trades(self) -> None:
        assert ReplayEngine._compute_calibration_buckets([]) == []


# ── Risk Metrics ──────────────────────────────────────────────────────


class TestRiskMetrics:

    def test_all_wins(self) -> None:
        pnls = [10.0, 20.0, 15.0, 25.0]
        metrics = ReplayEngine._compute_risk_metrics(pnls, 10000.0)
        assert metrics["sharpe"] > 0
        assert metrics["max_drawdown_pct"] == 0.0

    def test_all_losses(self) -> None:
        pnls = [-10.0, -20.0, -15.0]
        metrics = ReplayEngine._compute_risk_metrics(pnls, 10000.0)
        assert metrics["sharpe"] < 0
        assert metrics["max_drawdown_pct"] > 0

    def test_mixed_pnls(self) -> None:
        pnls = [100.0, -50.0, 75.0, -25.0, 150.0]
        metrics = ReplayEngine._compute_risk_metrics(pnls, 10000.0)
        assert metrics["sharpe"] > 0
        assert metrics["max_drawdown_pct"] > 0

    def test_empty_pnls(self) -> None:
        metrics = ReplayEngine._compute_risk_metrics([], 10000.0)
        assert metrics["sharpe"] == 0.0

    def test_single_pnl(self) -> None:
        metrics = ReplayEngine._compute_risk_metrics([50.0], 10000.0)
        assert metrics["sharpe"] == 0.0  # need at least 2


# ── Equity Curve ──────────────────────────────────────────────────────


class TestEquityCurve:

    def test_basic_curve(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m1",
                question="Q1", pnl=100.0,
            ),
            BacktestTradeRecord(
                run_id="r", market_condition_id="m2",
                question="Q2", pnl=-50.0,
            ),
        ]
        curve = ReplayEngine._compute_equity_curve(trades, 10000.0)
        assert len(curve) == 2
        assert curve[0]["cumulative_pnl"] == 100.0
        assert curve[0]["equity"] == 10100.0
        assert curve[1]["cumulative_pnl"] == 50.0
        assert curve[1]["equity"] == 10050.0


# ── Category Stats ────────────────────────────────────────────────────


class TestCategoryStats:

    def test_single_category(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m1",
                category="Crypto", pnl=50.0,
            ),
            BacktestTradeRecord(
                run_id="r", market_condition_id="m2",
                category="Crypto", pnl=-20.0,
            ),
        ]
        stats = ReplayEngine._compute_category_stats(trades)
        assert len(stats) == 1
        assert stats[0]["category"] == "Crypto"
        assert stats[0]["total_trades"] == 2
        assert stats[0]["total_pnl"] == 30.0

    def test_multiple_categories(self) -> None:
        trades = [
            BacktestTradeRecord(
                run_id="r", market_condition_id="m1",
                category="Crypto", pnl=50.0,
            ),
            BacktestTradeRecord(
                run_id="r", market_condition_id="m2",
                category="Politics", pnl=-20.0,
            ),
        ]
        stats = ReplayEngine._compute_category_stats(trades)
        assert len(stats) == 2


# ── Full Replay Integration ──────────────────────────────────────────


class TestReplayIntegration:

    def test_full_replay_with_cache(
        self,
        db: BacktestDatabase,
        config: BotConfig,
        cache: LLMResponseCache,
    ) -> None:
        """Full replay run with pre-seeded cache."""
        markets = _seed_markets(db, count=3)

        # Pre-seed cache for all markets and models
        questions = [m.question for m in markets]
        models = config.ensemble.models
        _seed_cache(cache, questions, models, prob=0.7)

        engine = ReplayEngine(
            config=config, backtest_db=db,
            cache=cache, force_cache_only=True,
        )

        result = asyncio.new_event_loop().run_until_complete(
            engine.run(max_markets=3, name="test-run")
        )

        assert result.total_markets == 3
        assert result.run_id != ""
        assert result.brier_score >= 0.0

        # Verify run was persisted
        run = db.get_backtest_run(result.run_id)
        assert run is not None
        assert run.status == "completed"

    def test_empty_market_set(
        self,
        db: BacktestDatabase,
        config: BotConfig,
        cache: LLMResponseCache,
    ) -> None:
        """Replay with no markets should produce empty result."""
        engine = ReplayEngine(config=config, backtest_db=db, cache=cache)

        result = asyncio.new_event_loop().run_until_complete(
            engine.run(max_markets=10, name="empty-run")
        )

        assert result.total_markets == 0
        assert result.markets_traded == 0
        assert result.total_pnl == 0.0

    def test_progress_callback(
        self,
        db: BacktestDatabase,
        config: BotConfig,
        cache: LLMResponseCache,
    ) -> None:
        """Progress callback fires for each market."""
        _seed_markets(db, count=2)
        questions = [f"Will event {i} happen?" for i in range(2)]
        _seed_cache(cache, questions, config.ensemble.models)

        calls: list[tuple[int, int, str]] = []

        engine = ReplayEngine(
            config=config, backtest_db=db,
            cache=cache, force_cache_only=True,
        )

        asyncio.new_event_loop().run_until_complete(
            engine.run(
                max_markets=2, name="progress-test",
                progress_callback=lambda c, t, q: calls.append((c, t, q)),
            )
        )

        assert len(calls) == 2


# ── BacktestResult ────────────────────────────────────────────────────


class TestBacktestResult:

    def test_to_dict(self) -> None:
        result = BacktestResult(
            run_id="r1",
            total_markets=100,
            markets_traded=50,
            total_pnl=250.0,
            brier_score=0.22,
        )
        d = result.to_dict()
        assert d["run_id"] == "r1"
        assert d["total_pnl"] == 250.0
        assert d["brier_score"] == 0.22
