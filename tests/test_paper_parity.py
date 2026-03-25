"""Tests for Phase 10B Batch C: Paper/live direction parity."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.config import StorageConfig, ExecutionConfig
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


def _make_engine(db):
    """Create a minimal TradingEngine for paper-fill testing."""
    from src.engine.loop import TradingEngine

    engine = TradingEngine.__new__(TradingEngine)
    engine._db = db
    engine._exit_finalizer = None

    # Mock ws_feed
    engine._ws_feed = MagicMock()
    engine._ws_feed.subscribe = MagicMock()

    # Mock config
    engine.config = MagicMock()
    engine.config.execution.dry_run = True
    engine.config.execution.live_exit_routing_enabled = False

    return engine


# ── Paper Auto-Fill Direction Tests ─────────────────────────────


class TestPaperAutoFillDirection:
    @pytest.mark.asyncio
    async def test_paper_fill_buy_yes_populates_canonical_fields(self):
        db = _make_db()
        engine = _make_engine(db)

        order = OrderRecord(
            order_id="ord-paper-1",
            clob_order_id="clob-paper-1",
            market_id="mkt-paper",
            token_id="tok-paper",
            side="BUY_YES",
            order_type="limit",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="submitted",
            dry_run=True,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_order(order)

        # Mock is_live_trading_enabled to return False (paper mode)
        with patch("src.engine.loop.is_live_trading_enabled", return_value=False):
            await engine._confirm_pending_orders()

        # Check trade has canonical fields
        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-paper-1'"
        ).fetchall()
        assert len(trades) == 1
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "YES"

        # Check position has canonical fields
        pos = db.get_position("mkt-paper")
        assert pos is not None
        assert pos.action_side == "BUY"
        assert pos.outcome_side == "YES"

    @pytest.mark.asyncio
    async def test_paper_fill_buy_no_populates_canonical_fields(self):
        db = _make_db()
        engine = _make_engine(db)

        order = OrderRecord(
            order_id="ord-paper-no",
            clob_order_id="clob-paper-no",
            market_id="mkt-paper-no",
            token_id="tok-paper-no",
            side="BUY_NO",
            order_type="limit",
            price=0.40,
            size=100.0,
            stake_usd=40.0,
            status="submitted",
            dry_run=True,
            action_side="BUY",
            outcome_side="NO",
        )
        db.insert_order(order)

        with patch("src.engine.loop.is_live_trading_enabled", return_value=False):
            await engine._confirm_pending_orders()

        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-paper-no'"
        ).fetchall()
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "NO"

    @pytest.mark.asyncio
    async def test_paper_fill_legacy_order_derives_canonical_fields(self):
        """Legacy orders without canonical fields get them from parse_direction."""
        db = _make_db()
        engine = _make_engine(db)

        order = OrderRecord(
            order_id="ord-legacy-paper",
            clob_order_id="clob-legacy-paper",
            market_id="mkt-legacy-paper",
            token_id="tok-legacy-paper",
            side="BUY_YES",
            order_type="limit",
            price=0.55,
            size=100.0,
            stake_usd=55.0,
            status="submitted",
            dry_run=True,
            # no action_side/outcome_side
        )
        db.insert_order(order)

        with patch("src.engine.loop.is_live_trading_enabled", return_value=False):
            await engine._confirm_pending_orders()

        trades = db._conn.execute(
            "SELECT action_side, outcome_side FROM trades WHERE order_id='ord-legacy-paper'"
        ).fetchall()
        assert trades[0]["action_side"] == "BUY"
        assert trades[0]["outcome_side"] == "YES"

    @pytest.mark.asyncio
    async def test_paper_fill_subscribes_ws(self):
        db = _make_db()
        engine = _make_engine(db)

        order = OrderRecord(
            order_id="ord-ws",
            clob_order_id="clob-ws",
            market_id="mkt-ws",
            token_id="tok-ws",
            side="BUY_YES",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=True,
            action_side="BUY",
            outcome_side="YES",
        )
        db.insert_order(order)

        with patch("src.engine.loop.is_live_trading_enabled", return_value=False):
            await engine._confirm_pending_orders()

        engine._ws_feed.subscribe.assert_called_once_with("tok-ws")


# ── Engine Uses has_active_order_for_market ─────────────────────


class TestEngineUsesDBHelper:
    def test_has_active_order_used_for_duplicate_check(self):
        """Engine should use has_active_order_for_market for dedup."""
        db = _make_db()

        # Insert an active order
        db.insert_order(OrderRecord(
            order_id="ord-dedup",
            clob_order_id="clob-dedup",
            market_id="mkt-dedup",
            token_id="tok-dedup",
            side="BUY_YES",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="submitted",
            dry_run=False,
        ))

        assert db.has_active_order_for_market("mkt-dedup") is True

        # Terminal order should not block
        db.insert_order(OrderRecord(
            order_id="ord-done",
            clob_order_id="clob-done",
            market_id="mkt-done",
            token_id="tok-done",
            side="BUY_YES",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="filled",
            dry_run=False,
        ))

        assert db.has_active_order_for_market("mkt-done") is False
