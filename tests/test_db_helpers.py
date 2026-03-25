"""Tests for Phase 10B Batch C: DB helper methods."""

from __future__ import annotations

import sqlite3

import pytest

from src.config import StorageConfig
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import OrderRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_db() -> Database:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


def _insert_order(db, order_id, market_id, status="submitted"):
    db.insert_order(OrderRecord(
        order_id=order_id,
        clob_order_id=f"clob-{order_id}",
        market_id=market_id,
        token_id=f"tok-{order_id}",
        side="BUY_YES",
        order_type="limit",
        price=0.50,
        size=100.0,
        stake_usd=50.0,
        status=status,
        dry_run=False,
    ))


# ── prune_terminal_orders Tests ─────────────────────────────────


class TestPruneTerminalOrders:
    def test_prunes_filled_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "filled")
        _insert_order(db, "ord-2", "mkt-2", "filled")

        count = db.prune_terminal_orders()
        assert count == 2

        rows = db._conn.execute("SELECT * FROM open_orders").fetchall()
        assert len(rows) == 0

    def test_prunes_cancelled_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "cancelled")

        count = db.prune_terminal_orders()
        assert count == 1

    def test_prunes_expired_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "expired")

        count = db.prune_terminal_orders()
        assert count == 1

    def test_prunes_failed_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "failed")

        count = db.prune_terminal_orders()
        assert count == 1

    def test_keeps_submitted_orders(self):
        db = _make_db()
        _insert_order(db, "ord-active", "mkt-1", "submitted")
        _insert_order(db, "ord-terminal", "mkt-2", "filled")

        count = db.prune_terminal_orders()
        assert count == 1

        rows = db._conn.execute("SELECT * FROM open_orders").fetchall()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-active"

    def test_keeps_pending_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "pending")

        count = db.prune_terminal_orders()
        assert count == 0

    def test_keeps_partial_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "partial")

        count = db.prune_terminal_orders()
        assert count == 0

    def test_empty_table_returns_zero(self):
        db = _make_db()
        count = db.prune_terminal_orders()
        assert count == 0


# ── has_active_order_for_market Tests ───────────────────────────


class TestHasActiveOrderForMarket:
    def test_returns_true_for_submitted(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "submitted")

        assert db.has_active_order_for_market("mkt-1") is True

    def test_returns_true_for_pending(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "pending")

        assert db.has_active_order_for_market("mkt-1") is True

    def test_returns_true_for_partial(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "partial")

        assert db.has_active_order_for_market("mkt-1") is True

    def test_returns_false_for_terminal_only(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "filled")
        _insert_order(db, "ord-2", "mkt-1", "cancelled")

        assert db.has_active_order_for_market("mkt-1") is False

    def test_returns_false_for_no_orders(self):
        db = _make_db()
        assert db.has_active_order_for_market("mkt-1") is False

    def test_returns_false_for_different_market(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-other", "submitted")

        assert db.has_active_order_for_market("mkt-1") is False


# ── get_active_orders Tests ─────────────────────────────────────


class TestGetActiveOrders:
    def test_returns_active_orders(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "submitted")
        _insert_order(db, "ord-2", "mkt-2", "pending")
        _insert_order(db, "ord-3", "mkt-3", "partial")

        active = db.get_active_orders()
        assert len(active) == 3

    def test_excludes_terminal_orders(self):
        db = _make_db()
        _insert_order(db, "ord-active", "mkt-1", "submitted")
        _insert_order(db, "ord-filled", "mkt-2", "filled")
        _insert_order(db, "ord-cancelled", "mkt-3", "cancelled")
        _insert_order(db, "ord-expired", "mkt-4", "expired")
        _insert_order(db, "ord-failed", "mkt-5", "failed")

        active = db.get_active_orders()
        assert len(active) == 1
        assert active[0].order_id == "ord-active"

    def test_empty_table_returns_empty(self):
        db = _make_db()
        assert db.get_active_orders() == []

    def test_returns_order_records(self):
        db = _make_db()
        _insert_order(db, "ord-1", "mkt-1", "submitted")

        active = db.get_active_orders()
        assert isinstance(active[0], OrderRecord)
        assert active[0].order_id == "ord-1"


# ── Reconciler Uses DB Method ──────────────────────────────────


class TestReconcilerUsesDBMethod:
    def test_reconciler_prune_uses_db_method(self):
        """Reconciler _prune_terminal_orders delegates to db.prune_terminal_orders."""
        from src.execution.reconciliation import OrderReconciler
        from src.config import ExecutionConfig

        db = _make_db()
        _insert_order(db, "ord-filled", "mkt-1", "filled")
        _insert_order(db, "ord-active", "mkt-2", "submitted")

        class FakeCLOB:
            pass

        reconciler = OrderReconciler(db, FakeCLOB(), ExecutionConfig())
        pruned = reconciler._prune_terminal_orders()

        assert pruned == 1
        rows = db._conn.execute("SELECT * FROM open_orders").fetchall()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-active"
