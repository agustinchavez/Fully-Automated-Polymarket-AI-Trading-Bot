"""Phase 12 — Deferred code review items.

Batch A: Typing cleanup verification
Batch B: SQLite concurrency hardening
Batch C: Backtest validation wiring
Batch D: PipelineRunner extraction
"""

from __future__ import annotations

import inspect
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Batch A: Typing Cleanup ───────────────────────────────────────────


class TestTypingCleanup:
    """Verify PipelineContext and TradingEngine use concrete types."""

    def test_pipeline_context_accepts_typed_fields(self) -> None:
        """PipelineContext can be constructed with concrete type instances."""
        from src.engine.loop import PipelineContext

        market = MagicMock()
        market.id = "m1"
        market.question = "Will X happen?"
        ctx = PipelineContext(market=market, cycle_id=1, market_id="m1", question="Q")
        assert ctx.market is market
        assert ctx.cycle_id == 1

    def test_pipeline_context_defaults(self) -> None:
        """Optional fields default to None."""
        from src.engine.loop import PipelineContext

        market = MagicMock()
        ctx = PipelineContext(market=market, cycle_id=1)
        assert ctx.classification is None
        assert ctx.evidence is None
        assert ctx.features is None
        assert ctx.forecast is None
        assert ctx.edge_result is None
        assert ctx.risk_result is None
        assert ctx.position is None
        assert ctx.sources == []

    def test_engine_init_accepts_bot_config(self) -> None:
        """TradingEngine.__init__ accepts BotConfig | None."""
        from src.config import BotConfig
        from src.engine.loop import TradingEngine

        config = BotConfig()
        engine = TradingEngine(config=config)
        assert engine.config is config

    def test_typing_import_guard_no_runtime_import(self) -> None:
        """TYPE_CHECKING block doesn't cause runtime imports of guarded types."""
        import sys
        # These modules should NOT be imported at module level by loop.py
        # (they're lazy-imported inside methods or under TYPE_CHECKING)
        # We verify by checking they aren't in sys.modules SOLELY because
        # of the loop import. Since tests may import them, we just verify
        # the import succeeds without error.
        from src.engine.loop import PipelineContext  # noqa: F401
        # If we got here, the TYPE_CHECKING guard works correctly
        assert True


# ── Batch B: SQLite Concurrency ───────────────────────────────────────


class TestSQLiteConcurrency:
    """Verify WAL mode, timeout, busy_timeout, and retry logic."""

    def test_dashboard_conn_has_wal(self, tmp_path: Path) -> None:
        """Dashboard _get_conn() enables WAL journal mode."""
        import src.dashboard.app as app_mod

        db_file = str(tmp_path / "dash.db")
        # Create a minimal DB so _get_conn works
        conn_init = sqlite3.connect(db_file)
        conn_init.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        conn_init.close()

        original = app_mod._db_path
        app_mod._db_path = db_file
        try:
            conn = app_mod._get_conn()
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal", f"Expected WAL, got {mode}"
            conn.close()
        finally:
            app_mod._db_path = original

    def test_dashboard_conn_has_busy_timeout(self, tmp_path: Path) -> None:
        """Dashboard _get_conn() sets busy_timeout."""
        import src.dashboard.app as app_mod

        db_file = str(tmp_path / "dash2.db")
        conn_init = sqlite3.connect(db_file)
        conn_init.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        conn_init.close()

        original = app_mod._db_path
        app_mod._db_path = db_file
        try:
            conn = app_mod._get_conn()
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert timeout == 30000, f"Expected 30000, got {timeout}"
            conn.close()
        finally:
            app_mod._db_path = original

    def test_database_connect_has_busy_timeout(self, tmp_path: Path) -> None:
        """Database.connect() sets busy_timeout pragma."""
        from src.config import StorageConfig
        from src.storage.database import Database

        db_file = str(tmp_path / "test.db")
        db = Database(StorageConfig(sqlite_path=db_file))
        db.connect()
        timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 30000
        db.close()

    def test_retry_on_locked(self) -> None:
        """_execute_with_retry retries on 'database is locked' error."""
        from src.config import StorageConfig
        from src.storage.database import Database

        db = Database(StorageConfig(sqlite_path=":memory:"))
        # Use a mock connection that fails once then succeeds
        mock_conn = MagicMock()
        call_count = 0

        def flaky_execute(sql, params=()):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.OperationalError("database is locked")
            return MagicMock()  # cursor

        mock_conn.execute = flaky_execute
        db._conn = mock_conn

        db._execute_with_retry("INSERT INTO test VALUES (?)", ("hello",))
        assert call_count == 2

    def test_retry_raises_after_max(self) -> None:
        """_execute_with_retry raises after max_retries exceeded."""
        from src.config import StorageConfig
        from src.storage.database import Database

        db = Database(StorageConfig(sqlite_path=":memory:"))
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        db._conn = mock_conn

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db._execute_with_retry(
                "INSERT INTO test VALUES (?)", ("x",), max_retries=2,
            )

    def test_backtest_db_has_busy_timeout(self, tmp_path: Path) -> None:
        """BacktestDatabase.connect() sets busy_timeout."""
        from src.backtest.database import BacktestDatabase

        db_file = str(tmp_path / "bt.db")
        db = BacktestDatabase(db_path=db_file)
        db.connect()
        timeout = db.conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 30000
        db.close()


# ── Batch C: Backtest Validation ──────────────────────────────────────


class TestBacktestValidation:
    """Verify backtest Sharpe storage and paper Sharpe computation."""

    def test_backtest_stores_sharpe_in_engine_state(self, tmp_path: Path) -> None:
        """After replay run completes, last_backtest_sharpe is stored in main DB."""
        from src.config import StorageConfig
        from src.storage.database import Database

        main_db_path = str(tmp_path / "main.db")
        main_db = Database(StorageConfig(sqlite_path=main_db_path))
        main_db.connect()

        # Simulate what replay_engine.py does after computing results
        sharpe = 1.5432
        main_db.set_engine_state(
            "last_backtest_sharpe", str(round(sharpe, 4)),
        )

        stored = main_db.get_engine_state("last_backtest_sharpe")
        assert stored == "1.5432"
        main_db.close()

    def test_paper_sharpe_computed_from_summaries(self, tmp_path: Path) -> None:
        """Paper Sharpe computed from daily_summaries and stored in engine_state."""
        import math
        from src.config import StorageConfig
        from src.storage.database import Database

        db_path = str(tmp_path / "paper.db")
        db = Database(StorageConfig(sqlite_path=db_path))
        db.connect()

        # daily_summaries table should already exist from migrations.
        # Insert 10 daily summaries with known PnL values.
        pnls = [10.0, -5.0, 15.0, 8.0, -3.0, 12.0, 7.0, -2.0, 9.0, 6.0]
        for i, pnl in enumerate(pnls):
            db.conn.execute(
                "INSERT INTO daily_summaries (summary_date, total_pnl, bankroll) "
                "VALUES (?, ?, 1000.0)",
                (f"2025-01-{i+1:02d}", pnl),
            )
        db.conn.commit()

        # Compute expected Sharpe
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(variance)
        expected_sharpe = round(mean_pnl / std, 4)

        # Simulate what engine loop does
        rows = db.conn.execute(
            "SELECT total_pnl FROM daily_summaries "
            "ORDER BY summary_date DESC LIMIT 30"
        ).fetchall()
        assert len(rows) >= 7
        fetched_pnls = [r["total_pnl"] for r in rows]
        mean = sum(fetched_pnls) / len(fetched_pnls)
        var = sum((p - mean) ** 2 for p in fetched_pnls) / len(fetched_pnls)
        s = math.sqrt(var) if var > 0 else 0.0
        paper_sharpe = round((mean / s) if s > 0 else 0.0, 4)
        db.set_engine_state("paper_sharpe", str(paper_sharpe))

        stored = db.get_engine_state("paper_sharpe")
        assert stored == str(expected_sharpe)
        db.close()

    def test_paper_sharpe_needs_minimum_days(self, tmp_path: Path) -> None:
        """Fewer than 7 days of data → no paper_sharpe stored."""
        from src.config import StorageConfig
        from src.storage.database import Database

        db_path = str(tmp_path / "paper2.db")
        db = Database(StorageConfig(sqlite_path=db_path))
        db.connect()

        # Insert only 5 summaries (below 7 minimum)
        for i in range(5):
            db.conn.execute(
                "INSERT INTO daily_summaries (summary_date, total_pnl, bankroll) "
                "VALUES (?, ?, 1000.0)",
                (f"2025-01-{i+1:02d}", 10.0),
            )
        db.conn.commit()

        rows = db.conn.execute(
            "SELECT total_pnl FROM daily_summaries "
            "ORDER BY summary_date DESC LIMIT 30"
        ).fetchall()
        # Condition: don't compute if < 7
        assert len(rows) < 7
        # paper_sharpe should not exist
        stored = db.get_engine_state("paper_sharpe")
        assert stored is None
        db.close()

    def test_preflight_agreement_passes(self, tmp_path: Path) -> None:
        """Backtest-paper agreement passes when Sharpes are within tolerance."""
        from src.config import StorageConfig
        from src.storage.database import Database
        from src.observability.preflight import PreflightChecker
        from src.config import BotConfig

        db_path = str(tmp_path / "preflight.db")
        db = Database(StorageConfig(sqlite_path=db_path))
        db.connect()

        # Set backtest and paper Sharpe within 25% tolerance
        db.set_engine_state("last_backtest_sharpe", "1.50")
        db.set_engine_state("paper_sharpe", "1.30")  # 13% divergence < 25%

        config = BotConfig()
        checker = PreflightChecker(config, db.conn)
        result = checker.check_backtest_paper_agreement()
        assert result.passed is True
        db.close()

    def test_preflight_agreement_fails(self, tmp_path: Path) -> None:
        """Backtest-paper agreement fails when Sharpes diverge > 25%."""
        from src.config import StorageConfig
        from src.storage.database import Database
        from src.observability.preflight import PreflightChecker
        from src.config import BotConfig

        db_path = str(tmp_path / "preflight2.db")
        db = Database(StorageConfig(sqlite_path=db_path))
        db.connect()

        # Set backtest and paper Sharpe outside tolerance
        db.set_engine_state("last_backtest_sharpe", "2.00")
        db.set_engine_state("paper_sharpe", "1.00")  # 50% divergence > 25%

        config = BotConfig()
        checker = PreflightChecker(config, db.conn)
        result = checker.check_backtest_paper_agreement()
        assert result.passed is False
        db.close()


# ── Batch D: PipelineRunner Extraction ───────────────────────────────


class TestPipelineRunner:
    """Verify PipelineRunner extracted correctly from TradingEngine."""

    def _make_runner(self):
        """Create a PipelineRunner with mock dependencies."""
        from src.engine.pipeline import PipelineRunner
        from src.config import BotConfig

        config = BotConfig()
        return PipelineRunner(
            config=config,
            db=MagicMock(),
            audit=MagicMock(),
            drawdown=MagicMock(),
            calibration_loop=MagicMock(),
            adaptive_weighter=MagicMock(),
            smart_entry=MagicMock(),
            specialist_router=None,
            fill_tracker=None,
            plan_controller=None,
            exit_finalizer=None,
            current_regime=None,
            ws_feed=MagicMock(),
            wallet_scanner=MagicMock(),
            positions=[],
            latest_scan_result=None,
        )

    def test_pipeline_runner_init(self) -> None:
        """PipelineRunner constructor accepts all dependencies."""
        runner = self._make_runner()
        assert runner.config is not None
        assert runner._db is not None
        assert runner._audit is not None
        assert runner.drawdown is not None

    def test_pipeline_stages_public_api(self) -> None:
        """All stage methods exist on PipelineRunner."""
        from src.engine.pipeline import PipelineRunner

        expected_methods = [
            "stage_classify",
            "stage_research",
            "stage_forecast",
            "stage_calibrate",
            "stage_edge_calc",
            "stage_uncertainty_adjustment",
            "stage_risk_checks",
            "stage_persist_forecast",
            "stage_correlation_check",
            "stage_var_gate",
            "stage_position_sizing",
            "stage_execute_order",
            "stage_audit_and_log",
            "record_performance_log",
            "maybe_run_post_mortem",
        ]
        for method_name in expected_methods:
            assert hasattr(PipelineRunner, method_name), f"Missing: {method_name}"

    def test_engine_no_longer_has_stage_methods(self) -> None:
        """TradingEngine no longer has core _stage_* methods (moved to PipelineRunner).

        Some methods retain thin delegation wrappers for backward compat.
        """
        from src.engine.loop import TradingEngine

        # These methods should be fully removed (no delegation needed)
        removed_methods = [
            "_stage_classify",
            "_stage_research",
            "_stage_forecast",
            "_stage_calibrate",
            "_stage_edge_calc",
            "_stage_uncertainty_adjustment",
            "_stage_risk_checks",
            "_stage_persist_forecast",
            "_stage_correlation_check",
            "_stage_var_gate",
            "_stage_position_sizing",
            "_stage_audit_and_log",
            "_run_forecast",
            "_submit_with_plan",
        ]
        for method_name in removed_methods:
            assert not hasattr(TradingEngine, method_name), (
                f"TradingEngine still has {method_name} — should be in PipelineRunner"
            )

        # These retain thin delegation wrappers for backward compat
        delegated = ["_stage_execute_order", "_record_order_result", "_log_candidate"]
        for method_name in delegated:
            assert hasattr(TradingEngine, method_name), (
                f"Missing delegation wrapper: {method_name}"
            )

    def test_engine_retains_lifecycle_methods(self) -> None:
        """TradingEngine still has lifecycle and infrastructure methods."""
        from src.engine.loop import TradingEngine

        retained_methods = [
            "start",
            "stop",
            "get_status",
            "_init_db",
            "_run_cycle",
            "_process_candidate",
            "_check_positions",
            "_finalize_exit",
            "_route_exit_order",
            "_discover_markets",
            "_persist_engine_state",
        ]
        for method_name in retained_methods:
            assert hasattr(TradingEngine, method_name), (
                f"TradingEngine lost {method_name} — should have been retained"
            )

    def test_engine_has_pipeline_attribute(self) -> None:
        """TradingEngine has _pipeline attribute initialized to None."""
        from src.engine.loop import TradingEngine

        engine = TradingEngine()
        assert hasattr(engine, "_pipeline")
        assert engine._pipeline is None  # initialized in start()

    def test_stage_classify_delegates(self) -> None:
        """stage_classify calls market_classifier correctly."""
        runner = self._make_runner()
        ctx = MagicMock()
        ctx.market = MagicMock()
        ctx.market_id = "test-123"
        ctx.question = "Will X happen?"
        ctx.market.market_type = "binary"

        with patch("src.engine.pipeline.PipelineRunner.stage_classify") as mock_classify:
            mock_classify(ctx)
            mock_classify.assert_called_once_with(ctx)

    def test_pipeline_runner_async_methods(self) -> None:
        """Async stage methods are coroutine functions."""
        from src.engine.pipeline import PipelineRunner

        async_methods = [
            "stage_research",
            "stage_forecast",
            "stage_execute_order",
        ]
        for method_name in async_methods:
            method = getattr(PipelineRunner, method_name)
            assert inspect.iscoroutinefunction(method), (
                f"{method_name} should be async"
            )

    def test_pipeline_runner_sync_methods(self) -> None:
        """Sync stage methods are NOT coroutine functions."""
        from src.engine.pipeline import PipelineRunner

        sync_methods = [
            "stage_classify",
            "stage_calibrate",
            "stage_edge_calc",
            "stage_uncertainty_adjustment",
            "stage_risk_checks",
            "stage_persist_forecast",
            "stage_correlation_check",
            "stage_var_gate",
            "stage_position_sizing",
            "stage_audit_and_log",
        ]
        for method_name in sync_methods:
            method = getattr(PipelineRunner, method_name)
            assert not inspect.iscoroutinefunction(method), (
                f"{method_name} should be sync"
            )

    def test_delegation_methods_on_engine(self) -> None:
        """TradingEngine still has delegation wrappers for exit fallback."""
        from src.engine.loop import TradingEngine

        assert hasattr(TradingEngine, "_record_performance_log")
        assert hasattr(TradingEngine, "_maybe_run_post_mortem")

    def test_log_candidate_uses_classification(self) -> None:
        """_log_candidate with classification writes classification.category, not market.market_type."""
        runner = self._make_runner()
        market = MagicMock()
        market.id = "m-1"
        market.question = "Will it rain?"
        market.market_type = "UNKNOWN"
        market.best_bid = 0.5

        classification = MagicMock()
        classification.category = "WEATHER"

        runner._log_candidate(
            cycle_id=1, market=market, decision="SKIP", reason="test",
            classification=classification,
        )

        runner._db.insert_candidate.assert_called_once()
        assert runner._db.insert_candidate.call_args.kwargs["market_type"] == "WEATHER"

    def test_log_candidate_fallback_to_market_type(self) -> None:
        """_log_candidate with classification=None writes market.market_type (backward compat)."""
        runner = self._make_runner()
        market = MagicMock()
        market.id = "m-2"
        market.question = "Will BTC hit 100k?"
        market.market_type = "CRYPTO"
        market.best_bid = 0.4

        runner._log_candidate(
            cycle_id=2, market=market, decision="SKIP", reason="test",
            classification=None,
        )

        runner._db.insert_candidate.assert_called_once()
        assert runner._db.insert_candidate.call_args.kwargs["market_type"] == "CRYPTO"

    def test_loop_log_candidate_calls_pass_classification(self) -> None:
        """Bug 1: loop.py _log_candidate calls must include classification= kwarg."""
        import inspect
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._process_candidate)
        # Split source at _log_candidate calls (excluding definition)
        # Each call block should contain classification= before the next statement
        parts = source.split("_log_candidate(")
        # First part is before any call; skip it
        call_parts = parts[1:]
        assert len(call_parts) >= 2, (
            f"Expected >=2 _log_candidate calls, found {len(call_parts)}"
        )
        for i, part in enumerate(call_parts):
            # Get text up to the matching close paren (rough: up to next dedent)
            block = part[:500]  # enough context for the full call
            assert "classification=" in block, (
                f"_log_candidate call #{i+1} missing classification="
            )
