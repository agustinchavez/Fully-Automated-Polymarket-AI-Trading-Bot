"""Tests for Phase 10 Batch C: Reconciliation loop + stale order cancellation."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.config import ExecutionConfig
from src.execution.reconciliation import OrderReconciler, ReconciliationResult, run_reconciliation_loop
from src.storage.models import OrderRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**kwargs) -> ExecutionConfig:
    defaults = dict(
        dry_run=False,
        reconciliation_enabled=True,
        reconciliation_interval_secs=1,
        stale_order_cancel_enabled=False,
        stale_order_cancel_secs=600,
    )
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


def _make_db():
    from src.storage.migrations import run_migrations
    from src.storage.database import Database
    from src.config import StorageConfig
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


def _make_clob(**method_results):
    clob = MagicMock()
    clob.get_order_status = MagicMock(
        return_value=method_results.get("get_order_status", {})
    )
    clob.cancel_order = MagicMock(
        return_value=method_results.get("cancel_order", {"cancelled": True})
    )
    return clob


def _insert_order(db, **kwargs):
    defaults = dict(
        order_id="ord-001",
        clob_order_id="clob-001",
        market_id="market-abc",
        token_id="token-xyz",
        side="BUY_YES",
        order_type="limit",
        price=0.50,
        size=100.0,
        stake_usd=50.0,
        status="submitted",
        dry_run=False,
    )
    defaults.update(kwargs)
    db.insert_order(OrderRecord(**defaults))


# ── Basic Reconciliation Tests ──────────────────────────────────


class TestReconcileOnce:
    def test_empty_no_orders(self):
        db = _make_db()
        clob = _make_clob()
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        result = reconciler.reconcile_once()

        assert result.checked == 0
        assert result.filled == 0
        assert result.errors == 0

    def test_filled_order_creates_trade_and_position(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "50",
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.checked == 1
        assert result.filled == 1

        # Terminal order pruned — verify via trade + position instead
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1
        assert trades[0]["status"] == "FILLED"
        assert trades[0]["size"] == 100.0  # 50 / 0.50

        # Position should be created (BUY order)
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1
        assert positions[0]["entry_price"] == 0.50

        # Filled order should be pruned from open_orders
        assert result.pruned >= 1

    def test_filled_order_no_taking_amount_uses_full_size(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "0",
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db, size=200.0, price=0.40)

        result = reconciler.reconcile_once()

        assert result.filled == 1
        # Verify full size via trade record (order is pruned)
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1
        assert trades[0]["size"] == 200.0

    def test_live_order_no_fill_unchanged(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "live",
            "takingAmount": "0",
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.checked == 1
        assert result.filled == 0
        assert result.partial == 0

        # Live order is NOT terminal — should still be in open_orders
        order = db.get_order("ord-001")
        assert order.status == "submitted"  # unchanged

    def test_partial_fill_updates_order(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "live",
            "takingAmount": "25",  # 25 / 0.50 = 50 tokens filled out of 100
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db, filled_size=0.0)

        result = reconciler.reconcile_once()

        assert result.partial == 1

        # Partial order is NOT terminal — should still be in open_orders
        order = db.get_order("ord-001")
        assert order.status == "partial"
        assert order.filled_size == 50.0  # 25 / 0.50

        # Position should be created with partial size
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1
        assert positions[0]["size"] == 50.0

        # Incremental trade record should be created
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1
        assert trades[0]["status"] == "PARTIAL_FILL"
        assert trades[0]["size"] == 50.0

    def test_partial_fill_average_price(self):
        """Partial fill should compute weighted average price correctly."""
        db = _make_db()

        # First partial: 30 tokens at 0.50
        _insert_order(db, filled_size=30.0, avg_fill_price=0.50, price=0.50)

        # Now CLOB shows 80 tokens filled (50 more at 0.50)
        clob = _make_clob(get_order_status={
            "status": "live",
            "takingAmount": "40",  # 40 / 0.50 = 80 tokens
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        result = reconciler.reconcile_once()

        assert result.partial == 1
        order = db.get_order("ord-001")
        assert order.filled_size == 80.0
        # avg = (0.50 * 30 + 0.50 * 50) / 80 = 0.50
        assert order.avg_fill_price == 0.50

    def test_cancelled_by_exchange(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "cancelled"})
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.cancelled == 1
        # Cancelled is terminal — order pruned
        assert result.pruned >= 1
        # No trades or positions created
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0

    def test_expired_by_exchange(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "expired"})
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.cancelled == 1
        assert result.pruned >= 1

    def test_clob_api_error_counted(self):
        db = _make_db()
        clob = _make_clob()
        clob.get_order_status.side_effect = Exception("API timeout")
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.errors == 1
        assert result.checked == 1

        # Order should remain unchanged (not terminal, not pruned)
        order = db.get_order("ord-001")
        assert order.status == "submitted"


# ── Fill Tracker Notification Tests ─────────────────────────────


class TestFillTrackerNotification:
    def test_fill_tracker_notified_on_fill(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "50",
        })
        config = _make_config()
        tracker = MagicMock()
        reconciler = OrderReconciler(db, clob, config, fill_tracker=tracker)

        _insert_order(db)
        reconciler.reconcile_once()

        # Should register then record fill (3 args, not 4)
        tracker.register_order.assert_called_once()
        tracker.record_fill.assert_called_once()
        # Verify record_fill has exactly 3 positional args (order_id, fill_price, fill_size)
        args = tracker.record_fill.call_args[0]
        assert len(args) == 3

    def test_fill_tracker_notified_on_cancel(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "cancelled"})
        config = _make_config()
        tracker = MagicMock()
        reconciler = OrderReconciler(db, clob, config, fill_tracker=tracker)

        _insert_order(db)
        reconciler.reconcile_once()

        tracker.record_unfilled.assert_called_once()


# ── Stale Order Cancellation Tests ──────────────────────────────


class TestStaleCancellation:
    def test_stale_order_cancelled_when_enabled(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "live", "takingAmount": "0"})
        config = _make_config(
            stale_order_cancel_enabled=True,
            stale_order_cancel_secs=0,  # immediate
        )
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.stale_cancelled == 1
        clob.cancel_order.assert_called_once_with("clob-001")
        # Cancelled is terminal — pruned
        assert result.pruned >= 1

    def test_stale_cancellation_disabled_leaves_orders(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "live", "takingAmount": "0"})
        config = _make_config(stale_order_cancel_enabled=False)
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.stale_cancelled == 0
        clob.cancel_order.assert_not_called()

    def test_stale_cancel_failure_counted_as_error(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "live", "takingAmount": "0"})
        clob.cancel_order.side_effect = Exception("Network error")
        config = _make_config(
            stale_order_cancel_enabled=True,
            stale_order_cancel_secs=0,
        )
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)

        result = reconciler.reconcile_once()

        assert result.errors >= 1
        assert result.stale_cancelled == 0


# ── SELL Order Fill Tests ───────────────────────────────────────


class TestSellOrderFill:
    def test_sell_fill_archives_position_with_real_pnl(self):
        """SELL fill should use actual position's entry price for PnL."""
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "30",  # 30 / 0.60 = 50 tokens
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        # Insert a real position with entry_price=0.50
        from src.storage.models import PositionRecord
        db.upsert_position(PositionRecord(
            market_id="market-abc",
            token_id="token-xyz",
            direction="BUY_YES",
            entry_price=0.50,
            size=50.0,
            stake_usd=25.0,
        ))

        _insert_order(db, side="SELL", price=0.60, size=50.0, stake_usd=30.0)

        result = reconciler.reconcile_once()

        assert result.filled == 1

        # Position should be removed (archived)
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        # Closed position should have real PnL: (0.60 - 0.50) * 50 = 5.0
        closed = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(closed) == 1
        assert closed[0]["pnl"] == 5.0
        assert closed[0]["entry_price"] == 0.50  # from real position
        assert closed[0]["exit_price"] == 0.60
        assert closed[0]["close_reason"] == "SELL_FILLED"

    def test_sell_fill_no_existing_position_uses_order_data(self):
        """If position is already gone, archive with order data and pnl=0."""
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "25",
        })
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        # No position in DB
        _insert_order(db, side="SELL", price=0.50, size=50.0)

        result = reconciler.reconcile_once()
        assert result.filled == 1


# ── WS Subscription on BUY Fill Tests ──────────────────────────


class TestWSSubscription:
    def test_buy_fill_triggers_ws_callback(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "50",
        })
        config = _make_config()
        ws_callback = MagicMock()
        reconciler = OrderReconciler(db, clob, config, on_buy_fill=ws_callback)

        _insert_order(db, token_id="token-ws-test")

        reconciler.reconcile_once()

        ws_callback.assert_called_once_with("token-ws-test")

    def test_sell_fill_no_ws_callback(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "matched",
            "takingAmount": "30",
        })
        config = _make_config()
        ws_callback = MagicMock()
        reconciler = OrderReconciler(db, clob, config, on_buy_fill=ws_callback)

        _insert_order(db, side="SELL", price=0.60, size=50.0)

        reconciler.reconcile_once()

        ws_callback.assert_not_called()

    def test_partial_fill_first_triggers_ws_callback(self):
        db = _make_db()
        clob = _make_clob(get_order_status={
            "status": "live",
            "takingAmount": "25",
        })
        config = _make_config()
        ws_callback = MagicMock()
        reconciler = OrderReconciler(db, clob, config, on_buy_fill=ws_callback)

        _insert_order(db, filled_size=0.0, token_id="token-partial")

        reconciler.reconcile_once()

        ws_callback.assert_called_once_with("token-partial")


# ── Batch Mixed Status Tests ───────────────────────────────────


class TestBatchMixedStatus:
    def test_batch_of_five_orders_mixed(self):
        db = _make_db()
        config = _make_config()

        # Insert 5 orders with different CLOB statuses
        _insert_order(db, order_id="ord-1", clob_order_id="clob-1")
        _insert_order(db, order_id="ord-2", clob_order_id="clob-2")
        _insert_order(db, order_id="ord-3", clob_order_id="clob-3")
        _insert_order(db, order_id="ord-4", clob_order_id="clob-4")
        _insert_order(db, order_id="ord-5", clob_order_id="clob-5")

        responses = {
            "clob-1": {"status": "matched", "takingAmount": "50"},
            "clob-2": {"status": "live", "takingAmount": "0"},
            "clob-3": {"status": "cancelled"},
            "clob-4": {"status": "matched", "takingAmount": "25"},
            "clob-5": {"status": "live", "takingAmount": "10"},  # partial
        }

        clob = MagicMock()
        clob.get_order_status = MagicMock(
            side_effect=lambda oid: responses.get(oid, {})
        )
        reconciler = OrderReconciler(db, clob, config)

        result = reconciler.reconcile_once()

        assert result.checked == 5
        assert result.filled == 2  # ord-1, ord-4
        assert result.cancelled == 1  # ord-3
        assert result.partial == 1  # ord-5 (partial fill)

        # Terminal orders (filled, cancelled) are pruned
        # Non-terminal (submitted, partial) remain
        assert db.get_order("ord-2").status == "submitted"  # live, no fill
        assert db.get_order("ord-5").status == "partial"

        # Filled/cancelled are pruned
        assert db.get_order("ord-1") is None
        assert db.get_order("ord-3") is None
        assert db.get_order("ord-4") is None


# ── Terminal Order Pruning Tests ───────────────────────────────


class TestTerminalPruning:
    def test_pruning_removes_terminal_orders(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "matched", "takingAmount": "50"})
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)
        result = reconciler.reconcile_once()

        assert result.pruned == 1
        assert db.get_order("ord-001") is None

    def test_nonterminal_orders_not_pruned(self):
        db = _make_db()
        clob = _make_clob(get_order_status={"status": "live", "takingAmount": "0"})
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db)
        result = reconciler.reconcile_once()

        assert result.pruned == 0
        assert db.get_order("ord-001") is not None


# ── Reconciliation Loop Tests ──────────────────────────────────


class TestReconciliationLoop:
    @pytest.mark.asyncio
    async def test_loop_runs_and_stops(self):
        db = _make_db()
        clob = _make_clob()
        config = _make_config(reconciliation_interval_secs=0)
        stop_event = asyncio.Event()

        # Stop after a short delay
        async def stop_after_delay():
            await asyncio.sleep(0.1)
            stop_event.set()

        asyncio.create_task(stop_after_delay())

        await run_reconciliation_loop(db, clob, config, stop_event)

        # If we got here, the loop stopped correctly

    @pytest.mark.asyncio
    async def test_loop_survives_errors(self):
        db = _make_db()
        clob = _make_clob()
        config = _make_config(reconciliation_interval_secs=0)
        stop_event = asyncio.Event()

        # Insert an order but make CLOB error
        _insert_order(db)
        clob.get_order_status.side_effect = Exception("BOOM")

        async def stop_after_delay():
            await asyncio.sleep(0.15)
            stop_event.set()

        asyncio.create_task(stop_after_delay())

        # Should not crash
        await run_reconciliation_loop(db, clob, config, stop_event)

    @pytest.mark.asyncio
    async def test_loop_not_started_in_paper_mode(self):
        """Verify the engine does NOT start reconciliation in paper mode."""
        with patch("src.engine.loop.load_config") as mock_load:
            from src.config import load_config
            cfg = load_config()
            cfg.execution.reconciliation_enabled = True
            cfg.execution.dry_run = True  # paper mode
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)

        assert engine._reconciliation_task is None


# ── Config Defaults Tests ──────────────────────────────────────


class TestReconciliationConfig:
    def test_reconciliation_disabled_by_default(self):
        config = ExecutionConfig()
        assert config.reconciliation_enabled is False

    def test_stale_cancel_disabled_by_default(self):
        config = ExecutionConfig()
        assert config.stale_order_cancel_enabled is False

    def test_stale_cancel_secs_default(self):
        config = ExecutionConfig()
        assert config.stale_order_cancel_secs == 600

    def test_reconciliation_interval_default(self):
        config = ExecutionConfig()
        assert config.reconciliation_interval_secs == 30


# ── Order Without CLOB ID Tests ────────────────────────────────


class TestOrderWithoutClobId:
    def test_pending_order_without_clob_id_skipped(self):
        """Pending orders without a CLOB ID should not be polled."""
        db = _make_db()
        clob = _make_clob()
        config = _make_config()
        reconciler = OrderReconciler(db, clob, config)

        _insert_order(db, status="pending", clob_order_id="")

        result = reconciler.reconcile_once()

        # Should not try to poll CLOB for orders without a CLOB ID
        clob.get_order_status.assert_not_called()
        assert result.checked == 0
