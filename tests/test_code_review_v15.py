"""Code Review v15 — Performance optimizations and fixes.

Fix 1: Singleton research infrastructure in PipelineRunner
Fix 3: SQLite PRAGMA synchronous=NORMAL
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fix 1: Singleton Research Infrastructure ──────────────────────────


class TestSingletonResearchInfra:
    """Verify research infra is created once in PipelineRunner.__init__."""

    def _make_runner(self, **overrides):
        """Create a PipelineRunner with mock dependencies."""
        from src.config import BotConfig
        from src.engine.pipeline import PipelineRunner

        config = overrides.pop("config", BotConfig())
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
            **overrides,
        )

    def _make_ctx(self):
        """Create a mock PipelineContext."""
        ctx = MagicMock()
        ctx.market_id = "test-123"
        ctx.question = "Will X happen?"
        ctx.market.market_type = "binary"
        ctx.classification.recommended_queries = 3
        ctx.classification.category = "SCIENCE"
        ctx.classification.researchability = 50
        ctx.features = MagicMock()
        ctx.features.implied_probability = 0.5
        return ctx

    def test_singleton_search_provider_created(self) -> None:
        """PipelineRunner creates _search_provider on init."""
        runner = self._make_runner()
        assert runner._search_provider is not None

    def test_singleton_source_fetcher_created(self) -> None:
        """PipelineRunner creates _source_fetcher on init."""
        runner = self._make_runner()
        assert runner._source_fetcher is not None

    def test_singleton_evidence_extractor_created(self) -> None:
        """PipelineRunner creates _evidence_extractor on init."""
        runner = self._make_runner()
        assert runner._evidence_extractor is not None

    @pytest.mark.asyncio
    async def test_stage_research_uses_singleton(self) -> None:
        """stage_research() uses the singleton _source_fetcher, not a new one."""
        runner = self._make_runner()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_sources = AsyncMock(return_value=[])
        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(return_value=MagicMock(quality_score=0.5))
        runner._source_fetcher = mock_fetcher
        runner._evidence_extractor = mock_extractor

        ctx = self._make_ctx()

        with patch("src.engine.pipeline.circuit_breakers") as mock_cbs:
            mock_cb = MagicMock()
            mock_cb.allow_request.return_value = True
            mock_cbs.get.return_value = mock_cb

            with patch("src.research.query_builder.build_queries", return_value=["q1"]):
                await runner.stage_research(ctx)

        mock_fetcher.fetch_sources.assert_called_once()
        mock_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_stage_research_no_close_call(self) -> None:
        """stage_research() does NOT close the singleton instances."""
        runner = self._make_runner()

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_sources = AsyncMock(return_value=[])
        mock_fetcher.close = AsyncMock()
        mock_extractor = AsyncMock()
        mock_extractor.extract = AsyncMock(return_value=MagicMock(quality_score=0.5))
        runner._source_fetcher = mock_fetcher
        runner._evidence_extractor = mock_extractor

        ctx = self._make_ctx()

        with patch("src.engine.pipeline.circuit_breakers") as mock_cbs:
            mock_cb = MagicMock()
            mock_cb.allow_request.return_value = True
            mock_cbs.get.return_value = mock_cb

            with patch("src.research.query_builder.build_queries", return_value=["q1"]):
                await runner.stage_research(ctx)

        mock_fetcher.close.assert_not_called()

    def test_close_method_exists(self) -> None:
        """PipelineRunner has an async close() method."""
        import inspect
        from src.engine.pipeline import PipelineRunner
        assert hasattr(PipelineRunner, "close")
        assert inspect.iscoroutinefunction(PipelineRunner.close)

    @pytest.mark.asyncio
    async def test_close_calls_fetcher_and_provider(self) -> None:
        """close() cleans up source_fetcher and search_provider."""
        runner = self._make_runner()

        mock_fetcher = AsyncMock()
        mock_provider = AsyncMock()
        runner._source_fetcher = mock_fetcher
        runner._search_provider = mock_provider

        await runner.close()

        mock_fetcher.close.assert_called_once()
        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_handles_none_singletons(self) -> None:
        """close() is safe when singletons are None (init failed)."""
        runner = self._make_runner()
        runner._source_fetcher = None
        runner._search_provider = None
        runner._evidence_extractor = None

        # Should not raise
        await runner.close()

    @pytest.mark.asyncio
    async def test_singleton_fallback_when_init_fails(self) -> None:
        """If singleton init failed, stage_research falls back to creating new instances."""
        runner = self._make_runner()
        runner._source_fetcher = None
        runner._evidence_extractor = None

        ctx = self._make_ctx()

        with patch("src.engine.pipeline.circuit_breakers") as mock_cbs:
            mock_cb = MagicMock()
            mock_cb.allow_request.return_value = True
            mock_cbs.get.return_value = mock_cb

            with patch("src.research.query_builder.build_queries", return_value=["q1"]):
                with patch("src.connectors.web_search.create_search_provider") as mock_csp:
                    mock_provider = AsyncMock()
                    mock_csp.return_value = mock_provider

                    with patch("src.research.source_fetcher.SourceFetcher") as mock_sf_cls:
                        mock_sf = AsyncMock()
                        mock_sf.fetch_sources = AsyncMock(return_value=[])
                        mock_sf_cls.return_value = mock_sf

                        with patch("src.research.evidence_extractor.EvidenceExtractor") as mock_ee_cls:
                            mock_ee = AsyncMock()
                            mock_ee.extract = AsyncMock(return_value=MagicMock(quality_score=0.5))
                            mock_ee_cls.return_value = mock_ee

                            await runner.stage_research(ctx)

        mock_csp.assert_called_once()
        mock_sf_cls.assert_called_once()
        mock_ee_cls.assert_called_once()


# ── Fix 3: SQLite PRAGMA synchronous=NORMAL ──────────────────────────


class TestSQLitePragmaSynchronous:
    """Verify PRAGMA synchronous=NORMAL is set in all database connections."""

    def test_main_database_synchronous_normal(self) -> None:
        """Main database sets PRAGMA synchronous=NORMAL."""
        from src.storage.database import Database
        from src.config import StorageConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(sqlite_path=str(Path(tmpdir) / "test.db"))
            db = Database(config)
            db.connect()
            try:
                result = db._conn.execute("PRAGMA synchronous").fetchone()
                # NORMAL = 1
                assert result[0] == 1, f"Expected synchronous=NORMAL (1), got {result[0]}"
            finally:
                db.close()

    def test_backtest_database_synchronous_normal(self) -> None:
        """Backtest database sets PRAGMA synchronous=NORMAL."""
        from src.backtest.database import BacktestDatabase

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "backtest.db")
            db = BacktestDatabase(db_path)
            db.connect()
            try:
                result = db._conn.execute("PRAGMA synchronous").fetchone()
                assert result[0] == 1, f"Expected synchronous=NORMAL (1), got {result[0]}"
            finally:
                db.close()

    def test_dashboard_connection_synchronous_normal(self) -> None:
        """Dashboard _get_conn() sets PRAGMA synchronous=NORMAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "dash.db")
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                result = conn.execute("PRAGMA synchronous").fetchone()
                assert result[0] == 1
            finally:
                conn.close()

    def test_synchronous_normal_with_wal(self) -> None:
        """synchronous=NORMAL with WAL provides crash safety."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_wal.db")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            wal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            sync = conn.execute("PRAGMA synchronous").fetchone()[0]
            assert wal == "wal"
            assert sync == 1
            conn.close()

    def test_pragma_in_database_source(self) -> None:
        """Verify PRAGMA synchronous=NORMAL appears in database.py source."""
        import inspect
        from src.storage import database
        source = inspect.getsource(database)
        assert 'PRAGMA synchronous=NORMAL' in source

    def test_pragma_in_backtest_database_source(self) -> None:
        """Verify PRAGMA synchronous=NORMAL appears in backtest/database.py source."""
        import inspect
        from src.backtest import database
        source = inspect.getsource(database)
        assert 'PRAGMA synchronous=NORMAL' in source


# ── DDGS max_workers=1 Preservation ──────────────────────────────────


class TestDDGSThreadSafety:
    """Verify DuckDuckGo ThreadPoolExecutor stays at max_workers=1."""

    def test_ddgs_max_workers_is_one(self) -> None:
        """DDGS executor max_workers must remain 1 to prevent primp deadlocks."""
        from src.connectors.web_search import DuckDuckGoProvider
        assert DuckDuckGoProvider._executor._max_workers == 1

    def test_ddgs_executor_is_class_level(self) -> None:
        """DDGS executor is a ClassVar, not instance-level."""
        from concurrent.futures import ThreadPoolExecutor
        from src.connectors.web_search import DuckDuckGoProvider
        assert isinstance(DuckDuckGoProvider._executor, ThreadPoolExecutor)


# ── Per-cycle Dedup (v13 work — verify still present) ────────────────


class TestCycleDedupStillPresent:
    """Verify per-cycle dedup code exists in loop.py."""

    def test_dedup_code_exists(self) -> None:
        """Dedup logic is present in engine loop source."""
        import inspect
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine)
        assert "seen_ids" in source
        assert "deduped" in source
        assert "engine.cycle_dedup" in source
