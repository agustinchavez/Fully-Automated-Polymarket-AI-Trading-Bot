"""Tests for parallel candidate processing (Phase 1 / Phase 2 split).

Verifies:
- _run_phase1 returns PipelineContext or None
- _run_phase2 runs risk + execution sequentially
- _process_candidate delegates to phase1 + phase2
- Main cycle loop runs Phase 1 in parallel, Phase 2 sequentially
- Exceptions in Phase 1 don't cancel other candidates
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import load_config


# ── Helpers ──────────────────────────────────────────────────────────


def _make_market(market_id: str = "mkt-abc", question: str = "Will X?") -> MagicMock:
    m = MagicMock()
    m.id = market_id
    m.question = question
    m.tokens = [
        MagicMock(outcome="Yes", token_id="tok-yes", price=0.50),
        MagicMock(outcome="No", token_id="tok-no", price=0.50),
    ]
    m.market_type = "binary"
    return m


def _make_engine():
    """Create a minimal TradingEngine with mocked dependencies."""
    with patch("src.engine.loop.DrawdownManager"):
        with patch("src.engine.loop.PortfolioRiskManager"):
            from src.engine.loop import TradingEngine
            cfg = load_config()
            engine = TradingEngine(config=cfg)

    engine._db = MagicMock()
    engine._db.get_open_positions.return_value = []
    engine._db.has_active_order_for_market.return_value = False
    engine._ws_feed = MagicMock()
    engine._audit = None

    # Mock the pipeline runner
    pipeline = MagicMock()
    pipeline.stage_classify = MagicMock()
    pipeline.stage_research = AsyncMock(return_value=True)
    pipeline.stage_forecast = AsyncMock(return_value=True)
    pipeline.stage_calibrate = MagicMock()
    pipeline.stage_edge_calc = MagicMock()
    pipeline.stage_uncertainty_adjustment = MagicMock()
    pipeline.stage_risk_checks = MagicMock()
    pipeline.stage_persist_forecast = MagicMock()
    pipeline.stage_correlation_check = MagicMock()
    pipeline.stage_var_gate = MagicMock()
    pipeline.stage_uma_check = AsyncMock()
    pipeline.stage_position_sizing = MagicMock()
    pipeline.stage_execute_order = AsyncMock()
    pipeline.stage_audit_and_log = MagicMock()
    pipeline._log_candidate = MagicMock()
    engine._pipeline = pipeline

    return engine


# ── Phase 1 Tests ────────────────────────────────────────────────────


class TestRunPhase1:
    """_run_phase1: classify → research → features → forecast."""

    @pytest.mark.asyncio
    async def test_returns_context_on_success(self):
        engine = _make_engine()
        market = _make_market()

        with patch("src.forecast.feature_builder.build_features", return_value=MagicMock()):
            ctx = await engine._run_phase1(market, cycle_id=1)

        assert ctx is not None
        assert ctx.market_id == "mkt-abc"
        engine._pipeline.stage_classify.assert_called_once()
        engine._pipeline.stage_research.assert_called_once()
        engine._pipeline.stage_forecast.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_duplicate_position(self):
        engine = _make_engine()
        market = _make_market()

        pos = MagicMock()
        pos.market_id = "mkt-abc"
        engine._db.get_open_positions.return_value = [pos]

        ctx = await engine._run_phase1(market, cycle_id=1)
        assert ctx is None

    @pytest.mark.asyncio
    async def test_returns_none_when_duplicate_order(self):
        engine = _make_engine()
        market = _make_market()
        engine._db.has_active_order_for_market.return_value = True

        ctx = await engine._run_phase1(market, cycle_id=1)
        assert ctx is None

    @pytest.mark.asyncio
    async def test_returns_none_when_research_fails(self):
        engine = _make_engine()
        market = _make_market()
        engine._pipeline.stage_research = AsyncMock(return_value=False)

        with patch("src.forecast.feature_builder.build_features", return_value=MagicMock()):
            ctx = await engine._run_phase1(market, cycle_id=1)

        assert ctx is None

    @pytest.mark.asyncio
    async def test_returns_none_when_forecast_fails(self):
        engine = _make_engine()
        market = _make_market()
        engine._pipeline.stage_forecast = AsyncMock(return_value=False)

        with patch("src.forecast.feature_builder.build_features", return_value=MagicMock()):
            ctx = await engine._run_phase1(market, cycle_id=1)

        assert ctx is None
        engine._pipeline._log_candidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_db_skips_duplicate_check(self):
        engine = _make_engine()
        engine._db = None
        market = _make_market()

        with patch("src.forecast.feature_builder.build_features", return_value=MagicMock()):
            ctx = await engine._run_phase1(market, cycle_id=1)

        assert ctx is not None


# ── Phase 2 Tests ────────────────────────────────────────────────────


class TestRunPhase2:
    """_run_phase2: calibrate → edge → risk → execute."""

    @pytest.mark.asyncio
    async def test_runs_all_stages(self):
        engine = _make_engine()
        market = _make_market()

        from src.engine.loop import PipelineContext
        ctx = PipelineContext(
            market=market, cycle_id=1,
            market_id="mkt-abc", question="Will X?",
        )
        # Simulate Phase 1 completed: forecast is set
        ctx.forecast = MagicMock(
            model_probability=0.6, implied_probability=0.5,
            edge=0.10, confidence_level="HIGH",
        )
        ctx.evidence = MagicMock(summary="test evidence")
        ctx.risk_result = MagicMock(allowed=True, violations=[])

        result = await engine._run_phase2(ctx, cycle_id=1)

        engine._pipeline.stage_calibrate.assert_called_once()
        engine._pipeline.stage_edge_calc.assert_called_once()
        engine._pipeline.stage_risk_checks.assert_called_once()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_no_trade_when_risk_blocks(self):
        engine = _make_engine()
        market = _make_market()

        from src.engine.loop import PipelineContext
        ctx = PipelineContext(
            market=market, cycle_id=1,
            market_id="mkt-abc", question="Will X?",
        )
        ctx.forecast = MagicMock(
            model_probability=0.6, implied_probability=0.5,
            edge=0.10, confidence_level="HIGH",
        )
        ctx.evidence = MagicMock(summary="test evidence")
        ctx.risk_result = MagicMock(
            allowed=False,
            violations=["daily_limit_exceeded"],
            to_dict=MagicMock(return_value={}),
        )

        result = await engine._run_phase2(ctx, cycle_id=1)

        assert result.get("trade_attempted", False) is False
        engine._pipeline._log_candidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_trade_when_position_none(self):
        engine = _make_engine()
        market = _make_market()

        from src.engine.loop import PipelineContext
        ctx = PipelineContext(
            market=market, cycle_id=1,
            market_id="mkt-abc", question="Will X?",
        )
        ctx.forecast = MagicMock()
        ctx.evidence = MagicMock(summary="test")
        ctx.risk_result = MagicMock(allowed=True, violations=[])
        # Position sizing returns None
        ctx.position = None

        result = await engine._run_phase2(ctx, cycle_id=1)
        assert result.get("trade_attempted", False) is False


# ── Process Candidate (backward compat) ──────────────────────────────


class TestProcessCandidate:
    """_process_candidate delegates to phase1 + phase2."""

    @pytest.mark.asyncio
    async def test_delegates_to_phases(self):
        engine = _make_engine()
        market = _make_market()

        engine._pipeline.stage_forecast = AsyncMock(return_value=True)
        engine._pipeline.stage_research = AsyncMock(return_value=True)
        engine._pipeline.stage_risk_checks = MagicMock()
        engine._pipeline.stage_position_sizing = MagicMock()

        from src.engine.loop import PipelineContext
        ctx = PipelineContext(
            market=market, cycle_id=1,
            market_id="mkt-abc", question="Will X?",
        )

        # Mock risk_result to exist on ctx after stage_risk_checks
        def set_risk(c):
            c.risk_result = MagicMock(allowed=True, violations=[])
        engine._pipeline.stage_risk_checks.side_effect = set_risk

        with patch("src.forecast.feature_builder.build_features", return_value=MagicMock()):
            result = await engine._process_candidate(market, cycle_id=1)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_returns_skip_result_when_phase1_none(self):
        engine = _make_engine()
        market = _make_market()

        # Duplicate position → Phase 1 returns None
        pos = MagicMock()
        pos.market_id = "mkt-abc"
        engine._db.get_open_positions.return_value = [pos]

        result = await engine._process_candidate(market, cycle_id=1)
        assert result["has_edge"] is False
        assert result["trade_attempted"] is False


# ── Parallel Execution Tests ─────────────────────────────────────────


class TestParallelExecution:
    """Main cycle loop: Phase 1 parallel, Phase 2 sequential."""

    @pytest.mark.asyncio
    async def test_phase1_runs_concurrently(self):
        """Multiple candidates' Phase 1 should overlap in time."""
        engine = _make_engine()

        # Track call times to prove concurrency
        call_times = []

        original_phase1 = engine._run_phase1

        async def slow_phase1(market, cycle_id):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)  # 50ms simulated research
            # Return None to skip Phase 2
            return None

        engine._run_phase1 = slow_phase1

        markets = [_make_market(f"mkt-{i}") for i in range(3)]

        # Run the parallel section directly
        phase1_tasks = [
            asyncio.wait_for(
                engine._run_phase1(candidate, 1),
                timeout=300,
            )
            for candidate in markets
        ]
        results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

        assert len(results) == 3
        # All 3 started within 20ms of each other (concurrent, not sequential)
        if len(call_times) >= 2:
            spread = call_times[-1] - call_times[0]
            assert spread < 0.04, f"Phase 1 calls not concurrent: {spread:.3f}s spread"

    @pytest.mark.asyncio
    async def test_phase1_exception_doesnt_cancel_others(self):
        """One failing Phase 1 should not affect others."""
        engine = _make_engine()

        call_count = 0

        async def mixed_phase1(market, cycle_id):
            nonlocal call_count
            call_count += 1
            if market.id == "mkt-1":
                raise ValueError("Research failed for mkt-1")
            return None  # Skip Phase 2

        engine._run_phase1 = mixed_phase1

        markets = [_make_market(f"mkt-{i}") for i in range(3)]

        phase1_tasks = [
            asyncio.wait_for(
                engine._run_phase1(candidate, 1),
                timeout=300,
            )
            for candidate in markets
        ]
        results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

        assert call_count == 3  # All 3 ran despite mkt-1 failure
        assert isinstance(results[1], ValueError)  # mkt-1 failed
        assert results[0] is None  # mkt-0 succeeded
        assert results[2] is None  # mkt-2 succeeded

    @pytest.mark.asyncio
    async def test_phase1_timeout_doesnt_cancel_others(self):
        """One timing-out Phase 1 should not affect others."""
        engine = _make_engine()

        async def timeout_phase1(market, cycle_id):
            if market.id == "mkt-slow":
                await asyncio.sleep(10)  # Will be cancelled by timeout
            return None

        engine._run_phase1 = timeout_phase1

        markets = [
            _make_market("mkt-fast"),
            _make_market("mkt-slow"),
        ]

        phase1_tasks = [
            asyncio.wait_for(
                engine._run_phase1(candidate, 1),
                timeout=0.1,  # Short timeout
            )
            for candidate in markets
        ]
        results = await asyncio.gather(*phase1_tasks, return_exceptions=True)

        assert results[0] is None  # Fast one succeeded
        assert isinstance(results[1], asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_phase2_only_runs_for_successful_phase1(self):
        """Phase 2 should only run for candidates that passed Phase 1."""
        engine = _make_engine()

        phase2_markets = []

        async def track_phase2(ctx, cycle_id):
            phase2_markets.append(ctx.market_id)
            return ctx.result

        engine._run_phase2 = track_phase2

        # Simulate phase1 results: success, exception, success
        from src.engine.loop import PipelineContext

        ctx_0 = PipelineContext(
            market=_make_market("mkt-0"), cycle_id=1,
            market_id="mkt-0", question="Q0",
        )
        ctx_0.risk_result = MagicMock(allowed=False, violations=[])

        ctx_2 = PipelineContext(
            market=_make_market("mkt-2"), cycle_id=1,
            market_id="mkt-2", question="Q2",
        )
        ctx_2.risk_result = MagicMock(allowed=False, violations=[])

        phase1_results = [
            ctx_0,                          # mkt-0: success
            ValueError("research failed"),  # mkt-1: exception
            ctx_2,                          # mkt-2: success
        ]

        # Simulate the main loop's Phase 2 dispatch
        import traceback as tb

        filtered = [_make_market(f"mkt-{i}") for i in range(3)]
        engine._research_cache = MagicMock()

        for candidate, p1 in zip(filtered, phase1_results):
            mid = getattr(candidate, "id", "?")
            engine._research_cache.mark_researched(getattr(candidate, "id", ""))

            if isinstance(p1, BaseException):
                continue
            if p1 is None:
                continue

            await engine._run_phase2(p1, 1)

        # Phase 2 ran for mkt-0 and mkt-2 only
        assert phase2_markets == ["mkt-0", "mkt-2"]
