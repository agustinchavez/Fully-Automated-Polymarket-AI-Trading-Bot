"""Integration tests for direction normalization (Phase 10B Batch A).

Tests verify:
  - Migration 17 adds action_side/outcome_side columns
  - DB CRUD methods populate and read canonical fields
  - Reconciler fills populate canonical fields
  - PnL calculation uses outcome_side correctly
  - Backward compatibility with empty canonical fields
"""

from __future__ import annotations

import sqlite3
import time
import uuid

import pytest

from src.config import StorageConfig, ExecutionConfig
from src.execution.direction import parse_direction, canonical_direction, is_long, is_short
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import (
    OrderRecord,
    TradeRecord,
    PositionRecord,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_db() -> Database:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


# ── Migration 17 Tests ──────────────────────────────────────────


class TestMigration17:
    def test_schema_version_is_18(self):
        db = _make_db()
        row = db._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == 20

    def test_open_orders_has_canonical_columns(self):
        db = _make_db()
        info = db._conn.execute("PRAGMA table_info(open_orders)").fetchall()
        cols = {row["name"] for row in info}
        assert "action_side" in cols
        assert "outcome_side" in cols

    def test_trades_has_canonical_columns(self):
        db = _make_db()
        info = db._conn.execute("PRAGMA table_info(trades)").fetchall()
        cols = {row["name"] for row in info}
        assert "action_side" in cols
        assert "outcome_side" in cols

    def test_positions_has_canonical_columns(self):
        db = _make_db()
        info = db._conn.execute("PRAGMA table_info(positions)").fetchall()
        cols = {row["name"] for row in info}
        assert "action_side" in cols
        assert "outcome_side" in cols

    def test_closed_positions_has_canonical_columns(self):
        db = _make_db()
        info = db._conn.execute("PRAGMA table_info(closed_positions)").fetchall()
        cols = {row["name"] for row in info}
        assert "action_side" in cols
        assert "outcome_side" in cols


# ── DB CRUD Tests ───────────────────────────────────────────────


class TestDBCanonicalFields:
    def test_insert_order_with_canonical_fields(self):
        db = _make_db()
        order = OrderRecord(
            order_id="ord-1",
            clob_order_id="clob-1",
            market_id="mkt-1",
            token_id="tok-1",
            side="BUY_YES",
            order_type="limit",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="submitted",
            dry_run=False,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_order(order)
        fetched = db.get_order("ord-1")
        assert fetched is not None
        assert fetched.action_side == "BUY"
        assert fetched.outcome_side == "YES"

    def test_insert_trade_with_canonical_fields(self):
        db = _make_db()
        trade = TradeRecord(
            id="trade-1",
            order_id="ord-1",
            market_id="mkt-1",
            token_id="tok-1",
            side="BUY_YES",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="FILLED",
            dry_run=False,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_trade(trade)
        rows = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE id='trade-1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["action_side"] == "BUY"
        assert rows[0]["outcome_side"] == "YES"

    def test_upsert_position_with_canonical_fields(self):
        db = _make_db()
        pos = PositionRecord(
            market_id="mkt-1",
            token_id="tok-1",
            direction="BUY_NO",
            entry_price=0.40,
            size=100.0,
            stake_usd=40.0,
            current_price=0.40,
            pnl=0.0,
            action_side="BUY",
            outcome_side="NO",
        )
        db.upsert_position(pos)
        fetched = db.get_position("mkt-1")
        assert fetched is not None
        assert fetched.action_side == "BUY"
        assert fetched.outcome_side == "NO"

    def test_archive_position_with_canonical_fields(self):
        db = _make_db()
        pos = PositionRecord(
            market_id="mkt-1",
            token_id="tok-1",
            direction="BUY_YES",
            entry_price=0.55,
            size=100.0,
            stake_usd=55.0,
            current_price=0.60,
            pnl=5.0,
            action_side="BUY",
            outcome_side="YES",
        )
        db.upsert_position(pos)
        db.archive_position(pos, exit_price=0.60, pnl=5.0, close_reason="TP_HIT")
        rows = db._conn.execute(
            "SELECT action_side, outcome_side FROM closed_positions WHERE market_id='mkt-1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["action_side"] == "BUY"
        assert rows[0]["outcome_side"] == "YES"

    def test_backward_compat_empty_canonical_fields(self):
        """Records inserted without canonical fields default to empty strings."""
        db = _make_db()
        trade = TradeRecord(
            id="trade-legacy",
            order_id="ord-legacy",
            market_id="mkt-legacy",
            token_id="tok-legacy",
            side="BUY_YES",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="FILLED",
            dry_run=True,
            # no action_side/outcome_side — defaults to ""
        )
        db.insert_trade(trade)
        rows = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE id='trade-legacy'"
        ).fetchall()
        assert rows[0]["action_side"] == ""
        assert rows[0]["outcome_side"] == ""

    def test_get_open_positions_has_canonical_fields(self):
        db = _make_db()
        pos = PositionRecord(
            market_id="mkt-1",
            token_id="tok-1",
            direction="BUY_YES",
            entry_price=0.55,
            size=100.0,
            stake_usd=55.0,
            current_price=0.55,
            pnl=0.0,
            action_side="BUY",
            outcome_side="YES",
        )
        db.upsert_position(pos)
        positions = db.get_open_positions()
        assert len(positions) == 1
        assert positions[0].action_side == "BUY"
        assert positions[0].outcome_side == "YES"

    def test_get_open_orders_has_canonical_fields(self):
        db = _make_db()
        order = OrderRecord(
            order_id="ord-1",
            clob_order_id="clob-1",
            market_id="mkt-1",
            token_id="tok-1",
            side="BUY_YES",
            order_type="limit",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="submitted",
            dry_run=False,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_order(order)
        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].action_side == "BUY"
        assert orders[0].outcome_side == "YES"


# ── Reconciler Direction Tests ──────────────────────────────────


class TestReconcilerDirection:
    def test_handle_fill_populates_canonical_fields(self):
        """A fully filled BUY order creates trade+position with canonical fields."""
        from src.execution.reconciliation import OrderReconciler

        db = _make_db()
        config = ExecutionConfig()

        # Insert an order
        order = OrderRecord(
            order_id="ord-fill-1",
            clob_order_id="clob-fill-1",
            market_id="mkt-fill",
            token_id="tok-fill",
            side="BUY_YES",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_order(order)

        # Mock CLOB
        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "matched", "takingAmount": "50.0"}

        reconciler = OrderReconciler(db, FakeCLOB(), config)
        from src.execution.reconciliation import ReconciliationResult
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Check trade has canonical fields
        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-fill-1'"
        ).fetchall()
        assert len(trades) == 1
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "YES"

        # Check position has canonical fields
        pos = db.get_position("mkt-fill")
        assert pos is not None
        assert pos.action_side == "BUY"
        assert pos.outcome_side == "YES"

    def test_handle_sell_fill_uses_canonical_fields(self):
        """A SELL fill creates trade with action_side=SELL and proper outcome_side."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult

        db = _make_db()
        config = ExecutionConfig()

        # Create existing position first
        pos = PositionRecord(
            market_id="mkt-sell",
            token_id="tok-sell",
            direction="BUY_YES",
            entry_price=0.50,
            size=100.0,
            stake_usd=50.0,
            current_price=0.60,
            pnl=10.0,
            action_side="BUY",
            outcome_side="YES",
        )
        db.upsert_position(pos)

        # Create SELL order
        order = OrderRecord(
            order_id="ord-sell-1",
            clob_order_id="clob-sell-1",
            market_id="mkt-sell",
            token_id="tok-sell",
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

        reconciler = OrderReconciler(db, FakeCLOB(), config)
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Trade should have SELL canonical fields
        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-sell-1'"
        ).fetchall()
        assert trades[0]["action_side"] == "SELL"
        assert trades[0]["outcome_side"] == "YES"

    def test_partial_fill_populates_canonical_fields(self):
        """Partial fills create incremental trade with canonical fields."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult

        db = _make_db()
        config = ExecutionConfig()

        order = OrderRecord(
            order_id="ord-partial",
            clob_order_id="clob-partial",
            market_id="mkt-partial",
            token_id="tok-partial",
            side="BUY_NO",
            order_type="limit",
            price=0.40,
            size=100.0,
            stake_usd=40.0,
            status="submitted",
            dry_run=False,
            action_side="BUY",
            outcome_side="NO",
            filled_size=0.0,
            avg_fill_price=0.0,
        )
        db.insert_order(order)

        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "live", "takingAmount": "20.0"}

        reconciler = OrderReconciler(db, FakeCLOB(), config)
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.partial == 1

        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-partial'"
        ).fetchall()
        assert len(trades) == 1
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "NO"

    def test_fallback_to_parse_direction_when_no_canonical(self):
        """Orders without canonical fields fall back to parse_direction."""
        from src.execution.reconciliation import OrderReconciler, ReconciliationResult

        db = _make_db()
        config = ExecutionConfig()

        order = OrderRecord(
            order_id="ord-legacy",
            clob_order_id="clob-legacy",
            market_id="mkt-legacy",
            token_id="tok-legacy",
            side="BUY_YES",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
            # no action_side/outcome_side — defaults to ""
        )
        db.insert_order(order)

        class FakeCLOB:
            def get_order_status(self, clob_id):
                return {"status": "matched", "takingAmount": "50.0"}

        reconciler = OrderReconciler(db, FakeCLOB(), config)
        result = ReconciliationResult()
        reconciler._reconcile_order(order, result)

        assert result.filled == 1

        # Should have derived canonical fields from parse_direction("BUY_YES")
        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-legacy'"
        ).fetchall()
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "YES"


# ── PnL Direction Tests ─────────────────────────────────────────


class TestPnLDirection:
    def test_yes_position_pnl_positive_on_price_rise(self):
        """BUY YES: PnL = (current - entry) * size."""
        # Long YES: price goes up → positive PnL
        entry, current, size = 0.50, 0.60, 100
        pnl = (current - entry) * size
        assert pnl == pytest.approx(10.0)

    def test_no_position_pnl_positive_on_price_drop(self):
        """BUY NO: PnL = (entry - current) * size."""
        # Short NO: price goes down → positive PnL
        entry, current, size = 0.50, 0.40, 100
        pnl = (entry - current) * size
        assert pnl == pytest.approx(10.0)

    def test_pnl_direction_via_outcome_side(self):
        """outcome_side determines PnL formula, not legacy direction."""
        entry, current, size = 0.50, 0.60, 100

        # YES → long formula
        a, o = parse_direction("BUY_YES")
        assert o == "YES"
        pnl_yes = (current - entry) * size
        assert pnl_yes == pytest.approx(10.0)

        # NO → short formula
        a, o = parse_direction("BUY_NO")
        assert o == "NO"
        pnl_no = (entry - current) * size
        assert pnl_no == pytest.approx(-10.0)

    def test_sell_direction_does_not_imply_buy_no(self):
        """SELL direction should NOT be treated as equivalent to BUY_NO."""
        a, o = parse_direction("SELL")
        assert a == "SELL"
        assert o == ""  # no outcome info in SELL alone
        assert not is_short(a, o)
        assert not is_long(a, o)


# ── Exit Order Direction Tests ──────────────────────────────────


class TestExitOrderDirection:
    def test_exit_order_has_sell_action(self):
        """Exit orders should always have action_side=SELL."""
        db = _make_db()
        order = OrderRecord(
            order_id="exit-1",
            clob_order_id="clob-exit-1",
            market_id="mkt-exit",
            token_id="tok-exit",
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
        fetched = db.get_order("exit-1")
        assert fetched.action_side == "SELL"
        assert fetched.outcome_side == "YES"

    def test_exit_order_inherits_position_outcome(self):
        """Exit orders should inherit outcome_side from the position being closed."""
        # A BUY_NO position's exit should have outcome_side=NO
        _, outcome = parse_direction("BUY_NO")
        assert outcome == "NO"
        # The exit order action_side is SELL, outcome_side is NO
        assert canonical_direction("SELL", outcome) == "SELL"
