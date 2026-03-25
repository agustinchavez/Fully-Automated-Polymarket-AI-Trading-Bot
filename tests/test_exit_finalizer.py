"""Tests for Phase 10B Batch B: ExitFinalizer shared exit pipeline."""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.config import StorageConfig, ExecutionConfig
from src.execution.exit_finalizer import ExitFinalizer
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import PositionRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_db() -> Database:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


def _make_position(**overrides) -> PositionRecord:
    defaults = dict(
        market_id="mkt-test",
        token_id="tok-test",
        direction="BUY_YES",
        entry_price=0.50,
        size=100.0,
        stake_usd=50.0,
        current_price=0.60,
        pnl=10.0,
        action_side="BUY",
        outcome_side="YES",
        opened_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return PositionRecord(**defaults)


class FakeConfig:
    """Minimal config mock for ExitFinalizer."""
    class CL:
        post_mortem_enabled = False
    continuous_learning = CL()


class FakeConfigWithPostMortem:
    class CL:
        post_mortem_enabled = True
    continuous_learning = CL()


# ── Core Finalization Tests ─────────────────────────────────────


class TestExitFinalizer:
    def test_finalize_archives_position(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        # Position should be archived
        rows = db._conn.execute(
            "SELECT * FROM closed_positions WHERE market_id='mkt-test'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["close_reason"] == "TP_HIT"
        assert rows[0]["pnl"] == 10.0

    def test_finalize_removes_open_position(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        assert db.get_position("mkt-test") is None

    def test_finalize_records_performance_log(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT * FROM performance_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["market_id"] == "mkt-test"
        assert rows[0]["pnl"] == 10.0

    def test_finalize_inserts_alert(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT * FROM alerts_log").fetchall()
        assert len(rows) >= 1
        alert_msg = rows[-1]["message"]
        assert "Auto-exit" in alert_msg
        assert "TP_HIT" in alert_msg

    def test_finalize_strips_colon_suffix_from_reason(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="SL_HIT:details")

        rows = db._conn.execute("SELECT close_reason FROM closed_positions").fetchall()
        assert rows[0]["close_reason"] == "SL_HIT"

    def test_finalize_rounds_pnl(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.123456, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT pnl FROM closed_positions").fetchall()
        assert rows[0]["pnl"] == 10.1235  # rounded to 4 decimal places


# ── Post-Mortem Tests ───────────────────────────────────────────


class TestPostMortem:
    def test_skips_post_mortem_when_disabled(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        # Should not raise even without PostMortemAnalyzer
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

    def test_skips_post_mortem_when_no_config(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        class NoPostMortemConfig:
            pass  # no continuous_learning attribute

        finalizer = ExitFinalizer(db, NoPostMortemConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

    @patch("src.execution.exit_finalizer.ExitFinalizer._maybe_run_post_mortem")
    def test_runs_post_mortem_when_enabled(self, mock_pm):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfigWithPostMortem())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        mock_pm.assert_called_once_with("mkt-test")


# ── Performance Log Details ─────────────────────────────────────


class TestPerformanceLogDetails:
    def test_resolved_market_sets_actual_outcome_1(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.99, pnl=49.0, close_reason="RESOLVED")

        rows = db._conn.execute("SELECT actual_outcome FROM performance_log").fetchall()
        assert rows[0]["actual_outcome"] == 1.0

    def test_resolved_market_sets_actual_outcome_0(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.01, pnl=-49.0, close_reason="RESOLVED")

        rows = db._conn.execute("SELECT actual_outcome FROM performance_log").fetchall()
        assert rows[0]["actual_outcome"] == 0.0

    def test_mid_price_sets_actual_outcome_none(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT actual_outcome FROM performance_log").fetchall()
        assert rows[0]["actual_outcome"] is None

    def test_holding_hours_calculated(self):
        db = _make_db()
        # Position opened 2 hours ago
        opened = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
        pos = _make_position(opened_at=opened)
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT holding_hours FROM performance_log").fetchall()
        assert rows[0]["holding_hours"] >= 1.9  # approximately 2 hours

    def test_category_from_market_record(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        @dataclass
        class FakeMkt:
            category: str = "crypto"
            question: str = "Will BTC hit 100k?"

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(
            pos, exit_price=0.60, pnl=10.0,
            close_reason="TP_HIT", mkt_record=FakeMkt(),
        )

        rows = db._conn.execute("SELECT category FROM performance_log").fetchall()
        assert rows[0]["category"] == "crypto"

    def test_category_fallback_to_unknown(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        rows = db._conn.execute("SELECT category FROM performance_log").fetchall()
        assert rows[0]["category"] == "UNKNOWN"


# ── Error Resilience Tests ──────────────────────────────────────


class TestErrorResilience:
    def test_survives_archive_error(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        # Make archive_position fail
        db.archive_position = MagicMock(side_effect=Exception("archive boom"))

        finalizer = ExitFinalizer(db, FakeConfig())
        # Should not raise — continues to remaining steps
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        # Performance log should still be recorded
        rows = db._conn.execute("SELECT * FROM performance_log").fetchall()
        assert len(rows) == 1

    def test_survives_performance_log_error(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        # Make perf log fail
        db.insert_performance_log = MagicMock(side_effect=Exception("perf boom"))

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        # Position should still be archived and removed
        rows = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(rows) == 1
        assert db.get_position("mkt-test") is None

    def test_survives_alert_error(self):
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        db.insert_alert = MagicMock(side_effect=Exception("alert boom"))

        finalizer = ExitFinalizer(db, FakeConfig())
        finalizer.finalize(pos, exit_price=0.60, pnl=10.0, close_reason="TP_HIT")

        # Should still archive and remove
        rows = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(rows) == 1


# ── Reconciler Integration Tests ────────────────────────────────


class TestReconcilerWithFinalizer:
    def test_sell_fill_uses_finalizer(self):
        """Reconciler SELL fill delegates to ExitFinalizer when provided."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult
        from src.storage.models import OrderRecord

        db = _make_db()
        config = ExecutionConfig()

        # Create position
        pos = _make_position(market_id="mkt-recon")
        db.upsert_position(pos)

        # Create SELL order
        order = OrderRecord(
            order_id="ord-sell-recon",
            clob_order_id="clob-sell-recon",
            market_id="mkt-recon",
            token_id="tok-test",
            side="SELL",
            order_type="limit",
            price=0.60,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
            action_side="SELL",
            outcome_side="YES",
        )
        db.insert_order(order)

        finalizer = ExitFinalizer(db, FakeConfig())
        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "matched", "takingAmount": "60.0"}

        reconciler = OrderReconciler(
            db, FakeCLOB(), config,
            exit_finalizer=finalizer,
        )
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Position should be archived via finalizer (full 5-step pipeline)
        rows = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(rows) == 1

        # Performance log should exist (finalizer records it)
        perf = db._conn.execute("SELECT * FROM performance_log").fetchall()
        assert len(perf) == 1

        # Alert should exist
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%Auto-exit%'"
        ).fetchall()
        assert len(alerts) >= 1

    def test_sell_fill_without_finalizer_falls_back(self):
        """Reconciler SELL fill without finalizer uses archive-only path."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult
        from src.storage.models import OrderRecord

        db = _make_db()
        config = ExecutionConfig()

        pos = _make_position(market_id="mkt-nofin")
        db.upsert_position(pos)

        order = OrderRecord(
            order_id="ord-sell-nofin",
            clob_order_id="clob-sell-nofin",
            market_id="mkt-nofin",
            token_id="tok-test",
            side="SELL",
            order_type="limit",
            price=0.60,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
            action_side="SELL",
            outcome_side="YES",
        )
        db.insert_order(order)

        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "matched", "takingAmount": "60.0"}

        # No exit_finalizer provided
        reconciler = OrderReconciler(db, FakeCLOB(), config)
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Position should be archived (archive-only fallback)
        rows = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(rows) == 1

    def test_orphan_sell_fill_inserts_critical_alert(self):
        """Reconciler orphan SELL fill creates alert with SELL_FILLED_ORPHAN."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult
        from src.storage.models import OrderRecord

        db = _make_db()
        config = ExecutionConfig()

        # No position — orphan SELL
        order = OrderRecord(
            order_id="ord-orphan",
            clob_order_id="clob-orphan",
            market_id="mkt-orphan",
            token_id="tok-orphan",
            side="SELL",
            order_type="limit",
            price=0.60,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
            action_side="SELL",
            outcome_side="YES",
        )
        db.insert_order(order)

        finalizer = ExitFinalizer(db, FakeConfig())
        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "matched", "takingAmount": "60.0"}

        reconciler = OrderReconciler(
            db, FakeCLOB(), config,
            exit_finalizer=finalizer,
        )
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Should have SELL_FILLED_ORPHAN in closed_positions
        rows = db._conn.execute(
            "SELECT close_reason FROM closed_positions WHERE market_id='mkt-orphan'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["close_reason"] == "SELL_FILLED_ORPHAN"

        # Should have critical alert
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE level='critical'"
        ).fetchall()
        assert len(alerts) >= 1
        assert "Orphan SELL fill" in alerts[0]["message"]


# ── Engine Delegation Test ──────────────────────────────────────


class TestEngineDelegation:
    def test_finalize_exit_delegates_to_finalizer(self):
        """Engine _finalize_exit delegates to ExitFinalizer when available."""
        db = _make_db()
        pos = _make_position()
        db.upsert_position(pos)

        finalizer = ExitFinalizer(db, FakeConfig())
        mock_finalize = MagicMock(wraps=finalizer.finalize)
        finalizer.finalize = mock_finalize

        # Simulate engine with _exit_finalizer set
        from src.engine.loop import TradingEngine
        engine = TradingEngine.__new__(TradingEngine)
        engine._db = db
        engine._exit_finalizer = finalizer
        engine.config = type("C", (), {"continuous_learning": FakeConfig.CL()})()

        engine._finalize_exit(
            pos, exit_price=0.60, pnl=10.0,
            exit_reason="TP_HIT", mkt_record=None,
        )

        mock_finalize.assert_called_once()
