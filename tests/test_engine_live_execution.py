"""Tests for Phase 10 Batch B: Engine loop split (submit vs confirm) + live exit routing."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ExecutionConfig, load_config
from src.execution.order_builder import OrderSpec, build_exit_order
from src.execution.order_router import OrderResult
from src.storage.models import OrderRecord, TradeRecord, PositionRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**kwargs) -> ExecutionConfig:
    defaults = dict(dry_run=True, max_retries=1, retry_backoff_secs=0.01)
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


def _make_order_result(**kwargs) -> OrderResult:
    defaults = dict(
        order_id=str(uuid.uuid4()),
        status="simulated",
        fill_price=0.50,
        fill_size=100.0,
        timestamp="2024-01-01T00:00:00Z",
        clob_order_id="",
    )
    defaults.update(kwargs)
    return OrderResult(**defaults)


def _make_order_spec(**kwargs) -> OrderSpec:
    defaults = dict(
        order_id="test-order-1234",
        market_id="market-abc",
        token_id="token-xyz",
        side="BUY",
        order_type="limit",
        price=0.50,
        size=100.0,
        stake_usd=50.0,
        ttl_secs=300,
        dry_run=False,
    )
    defaults.update(kwargs)
    return OrderSpec(**defaults)


def _make_db():
    """Create in-memory Database with migrations applied."""
    from src.storage.migrations import run_migrations
    from src.storage.database import Database
    from src.config import StorageConfig
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


@dataclass
class FakePosition:
    market_id: str = "market-abc"
    token_id: str = "token-xyz"
    direction: str = "BUY_YES"
    entry_price: float = 0.50
    size: float = 100.0
    stake_usd: float = 50.0
    current_price: float = 0.55
    pnl: float = 5.0
    opened_at: str = "2024-01-01T00:00:00Z"


@dataclass
class FakeMarketRecord:
    question: str = "Will X happen?"
    category: str = "crypto"


@dataclass
class FakeEdgeResult:
    direction: str = "BUY_YES"
    abs_net_edge: float = 0.10


@dataclass
class FakePositionSize:
    stake_usd: float = 50.0
    token_quantity: float = 100.0
    kelly_fraction_used: float = 0.05
    capped_by: str = "kelly"
    direction: str = "BUY_YES"


@dataclass
class FakeForecast:
    model_probability: float = 0.60
    implied_probability: float = 0.50


# ── build_exit_order Tests ──────────────────────────────────────


class TestBuildExitOrder:
    def test_sell_order_created(self):
        config = _make_config(default_order_type="limit", slippage_tolerance=0.005)
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config)
        assert order.side == "SELL"
        assert order.market_id == "mkt-1"
        assert order.token_id == "tok-1"
        assert order.size == 50.0
        assert order.execution_strategy == "simple"

    def test_limit_price_below_current(self):
        config = _make_config(default_order_type="limit", slippage_tolerance=0.01)
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.80, config)
        # Limit SELL price should be below current price
        assert order.price < 0.80
        expected = round(0.80 * (1 - 0.01), 4)
        assert order.price == expected

    def test_market_order_price_zero(self):
        config = _make_config(default_order_type="market", slippage_tolerance=0.01)
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.80, config)
        assert order.price == 0.0

    def test_exit_reason_in_metadata(self):
        config = _make_config()
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.80, config, exit_reason="STOP_LOSS")
        assert order.metadata["exit_reason"] == "STOP_LOSS"

    def test_stake_usd_calculated(self):
        config = _make_config()
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.60, config)
        assert order.stake_usd == round(50.0 * 0.60, 2)

    def test_dry_run_flag_from_config(self):
        config = _make_config(dry_run=True)
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.60, config)
        assert order.dry_run is True

        config2 = _make_config(dry_run=False)
        order2 = build_exit_order("mkt-1", "tok-1", 50.0, 0.60, config2)
        assert order2.dry_run is False


# ── _record_order_result Tests ──────────────────────────────────


class TestRecordOrderResult:
    """Tests for the _record_order_result branching logic."""

    def _make_engine(self, db):
        """Create a minimal TradingEngine with a mocked DB."""
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    def _make_ctx(self):
        """Create a minimal PipelineContext."""
        from src.engine.loop import PipelineContext
        return PipelineContext(
            market=MagicMock(market_type="binary"),
            cycle_id=1,
            market_id="market-abc",
            question="Will X happen?",
        )

    def test_simulated_creates_trade_and_position(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()
        order = _make_order_spec()
        result = _make_order_result(status="simulated", fill_price=0.50, fill_size=100.0)
        edge = FakeEdgeResult()
        pos = FakePositionSize()

        engine._record_order_result(ctx, order, result, edge, pos, "token-xyz")

        # Trade should be created
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1
        assert trades[0]["side"] == "BUY_YES"
        assert trades[0]["dry_run"] == 1

        # Position should be created
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1
        assert positions[0]["entry_price"] == 0.50

    def test_filled_creates_trade_and_position(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()
        order = _make_order_spec()
        result = _make_order_result(
            status="filled", fill_price=0.52, fill_size=96.0,
            clob_order_id="clob-999",
        )
        edge = FakeEdgeResult()
        pos = FakePositionSize()

        engine._record_order_result(ctx, order, result, edge, pos, "token-xyz")

        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1
        assert trades[0]["price"] == 0.52
        assert trades[0]["size"] == 96.0
        assert trades[0]["dry_run"] == 0  # live fill

        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1
        assert positions[0]["entry_price"] == 0.52

    def test_submitted_creates_order_no_position(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()
        order = _make_order_spec()
        result = _make_order_result(
            status="submitted", fill_price=0.0, fill_size=0.0,
            clob_order_id="clob-111",
        )
        edge = FakeEdgeResult()
        pos = FakePositionSize()

        engine._record_order_result(ctx, order, result, edge, pos, "token-xyz")

        # No trade or position yet
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0

        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        # But open_orders should have the order
        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "submitted"
        assert orders[0].clob_order_id == "clob-111"

    def test_pending_creates_order_no_position(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()
        order = _make_order_spec()
        result = _make_order_result(status="pending", fill_price=0.0, fill_size=0.0)
        edge = FakeEdgeResult()
        pos = FakePositionSize()

        engine._record_order_result(ctx, order, result, edge, pos, "token-xyz")

        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "pending"

    def test_failed_creates_order_with_error(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()
        order = _make_order_spec()
        result = _make_order_result(
            status="failed", fill_price=0.0, fill_size=0.0,
            error="Insufficient balance",
        )
        edge = FakeEdgeResult()
        pos = FakePositionSize()

        engine._record_order_result(ctx, order, result, edge, pos, "token-xyz")

        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "failed"
        assert "Insufficient" in orders[0].error


# ── _route_exit_order Tests ─────────────────────────────────────


class TestRouteExitOrder:
    """Tests for live vs simulated exit routing."""

    def _make_engine(self, db):
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_simulated_exit_creates_trade_and_archives(self):
        """Paper mode exit should create a simulated SELL trade and close position."""
        db = _make_db()
        engine = self._make_engine(db)
        pos = FakePosition()
        mkt = FakeMarketRecord()

        # Insert position first so archive works
        db.upsert_position(PositionRecord(
            market_id=pos.market_id, token_id=pos.token_id,
            direction=pos.direction, entry_price=pos.entry_price,
            size=pos.size, stake_usd=pos.stake_usd,
        ))

        # Mock _finalize_exit to verify it's called
        engine._finalize_exit = MagicMock()

        result = await engine._route_exit_order(
            pos, current_price=0.55, pnl=5.0,
            exit_reason="STOP_LOSS: -10%", mkt_record=mkt,
        )

        assert result is True
        # Should have inserted a SELL trade
        trades = db._conn.execute("SELECT * FROM trades WHERE side='SELL'").fetchall()
        assert len(trades) == 1
        assert "SIMULATED" in trades[0]["status"]
        # finalize_exit should be called
        engine._finalize_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulated_exit_when_live_exit_disabled(self):
        """Even in live mode, if live_exit_routing is disabled, use simulated exit."""
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.live_exit_routing = False
        pos = FakePosition()
        mkt = FakeMarketRecord()

        engine._finalize_exit = MagicMock()

        result = await engine._route_exit_order(
            pos, current_price=0.55, pnl=5.0,
            exit_reason="TAKE_PROFIT", mkt_record=mkt,
        )

        assert result is True

    @pytest.mark.asyncio
    @patch("src.engine.loop.is_live_trading_enabled", return_value=True)
    async def test_live_exit_filled_closes_position(self, mock_live):
        """Live exit with immediate fill should close the position."""
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.live_exit_routing = True
        engine.config.execution.dry_run = False
        pos = FakePosition()
        mkt = FakeMarketRecord()

        engine._finalize_exit = MagicMock()

        mock_result = _make_order_result(status="filled", fill_price=0.55, fill_size=100.0)

        with patch("src.execution.order_builder.build_exit_order") as mock_build, \
             patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls:
            mock_build.return_value = _make_order_spec(side="SELL")
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=mock_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            result = await engine._route_exit_order(
                pos, current_price=0.55, pnl=5.0,
                exit_reason="RESOLVED", mkt_record=mkt,
            )

        assert result is True
        engine._finalize_exit.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.engine.loop.is_live_trading_enabled", return_value=True)
    async def test_live_exit_submitted_keeps_position_open(self, mock_live):
        """Live exit with SELL on book should keep position open."""
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.live_exit_routing = True
        engine.config.execution.dry_run = False
        pos = FakePosition()
        mkt = FakeMarketRecord()

        mock_result = _make_order_result(
            status="submitted", fill_price=0.0, fill_size=0.0,
            clob_order_id="clob-sell-001",
        )

        with patch("src.execution.order_builder.build_exit_order") as mock_build, \
             patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls:
            mock_build.return_value = _make_order_spec(side="SELL")
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=mock_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            result = await engine._route_exit_order(
                pos, current_price=0.55, pnl=5.0,
                exit_reason="TAKE_PROFIT", mkt_record=mkt,
            )

        assert result is False  # Position stays open
        # SELL order should be tracked in open_orders
        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].status == "submitted"

    @pytest.mark.asyncio
    @patch("src.engine.loop.is_live_trading_enabled", return_value=True)
    async def test_live_exit_failed_keeps_position_open(self, mock_live):
        """Live exit that fails should keep position open for retry."""
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.live_exit_routing = True
        engine.config.execution.dry_run = False
        pos = FakePosition()
        mkt = FakeMarketRecord()

        mock_result = _make_order_result(
            status="failed", fill_price=0.0, fill_size=0.0,
            error="Network timeout",
        )

        with patch("src.execution.order_builder.build_exit_order") as mock_build, \
             patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls:
            mock_build.return_value = _make_order_spec(side="SELL")
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=mock_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            result = await engine._route_exit_order(
                pos, current_price=0.55, pnl=5.0,
                exit_reason="STOP_LOSS", mkt_record=mkt,
            )

        assert result is False

    @pytest.mark.asyncio
    @patch("src.engine.loop.is_live_trading_enabled", return_value=True)
    async def test_live_exit_exception_keeps_position_open(self, mock_live):
        """Exception during live exit routing should not crash — returns False."""
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.live_exit_routing = True
        engine.config.execution.dry_run = False
        pos = FakePosition()
        mkt = FakeMarketRecord()

        with patch("src.execution.order_builder.build_exit_order", side_effect=ImportError("test")):
            result = await engine._route_exit_order(
                pos, current_price=0.55, pnl=5.0,
                exit_reason="STOP_LOSS", mkt_record=mkt,
            )

        assert result is False


# ── _finalize_exit Tests ────────────────────────────────────────


class TestFinalizeExit:
    def _make_engine(self, db):
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    def test_finalize_archives_and_cleans_up(self):
        db = _make_db()
        engine = self._make_engine(db)
        pos = FakePosition()
        mkt = FakeMarketRecord()

        # Insert position
        db.upsert_position(PositionRecord(
            market_id=pos.market_id, token_id=pos.token_id,
            direction=pos.direction, entry_price=pos.entry_price,
            size=pos.size, stake_usd=pos.stake_usd,
        ))

        # Mock _record_performance_log and _maybe_run_post_mortem
        engine._record_performance_log = MagicMock()
        engine._maybe_run_post_mortem = MagicMock()

        engine._finalize_exit(pos, exit_price=0.55, pnl=5.0,
                              exit_reason="STOP_LOSS: -10%", mkt_record=mkt)

        # Position should be removed
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        # Closed positions should have entry
        closed = db._conn.execute("SELECT * FROM closed_positions").fetchall()
        assert len(closed) == 1
        assert closed[0]["close_reason"] == "STOP_LOSS"

        # Alert should be inserted
        alerts = db._conn.execute("SELECT * FROM alerts_log").fetchall()
        assert len(alerts) >= 1

    def test_finalize_calls_post_mortem(self):
        db = _make_db()
        engine = self._make_engine(db)
        pos = FakePosition()

        db.upsert_position(PositionRecord(
            market_id=pos.market_id, token_id=pos.token_id,
            direction=pos.direction, entry_price=pos.entry_price,
            size=pos.size, stake_usd=pos.stake_usd,
        ))

        engine._record_performance_log = MagicMock()
        engine._maybe_run_post_mortem = MagicMock()

        engine._finalize_exit(pos, exit_price=0.60, pnl=10.0,
                              exit_reason="TAKE_PROFIT", mkt_record=FakeMarketRecord())

        engine._maybe_run_post_mortem.assert_called_once_with(pos.market_id)


# ── _confirm_pending_orders Tests ───────────────────────────────


class TestConfirmPendingOrders:
    def _make_engine(self, db):
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_no_pending_orders_is_noop(self):
        db = _make_db()
        engine = self._make_engine(db)
        # No orders in DB
        await engine._confirm_pending_orders()
        # No crash, no trades
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0

    @pytest.mark.asyncio
    async def test_paper_mode_auto_fills_submitted_orders(self):
        db = _make_db()
        engine = self._make_engine(db)

        # Insert a submitted order
        db.insert_order(OrderRecord(
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
            dry_run=True,
        ))

        await engine._confirm_pending_orders()

        # Order should be updated to filled
        order = db.get_order("ord-001")
        assert order.status == "filled"
        assert order.filled_size == 100.0
        assert order.avg_fill_price == 0.50

        # Trade and position should be created
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1

        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_paper_mode_auto_fills_pending_orders(self):
        db = _make_db()
        engine = self._make_engine(db)

        db.insert_order(OrderRecord(
            order_id="ord-002",
            market_id="market-def",
            token_id="token-abc",
            side="BUY_NO",
            price=0.40,
            size=200.0,
            stake_usd=80.0,
            status="pending",
            dry_run=True,
        ))

        await engine._confirm_pending_orders()

        order = db.get_order("ord-002")
        assert order.status == "filled"

    @pytest.mark.asyncio
    @patch("src.engine.loop.is_live_trading_enabled", return_value=True)
    async def test_live_mode_defers_to_reconciliation(self, mock_live):
        db = _make_db()
        engine = self._make_engine(db)
        engine.config.execution.dry_run = False

        db.insert_order(OrderRecord(
            order_id="ord-003",
            market_id="market-ghi",
            token_id="token-live",
            side="BUY_YES",
            price=0.60,
            size=50.0,
            stake_usd=30.0,
            status="submitted",
            dry_run=False,
        ))

        await engine._confirm_pending_orders()

        # Order should NOT be auto-filled in live mode
        order = db.get_order("ord-003")
        assert order.status == "submitted"  # unchanged

        # No trade or position created
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0

    @pytest.mark.asyncio
    async def test_paper_auto_fill_subscribes_ws(self):
        db = _make_db()
        engine = self._make_engine(db)

        db.insert_order(OrderRecord(
            order_id="ord-ws",
            market_id="market-ws",
            token_id="token-ws",
            side="BUY_YES",
            price=0.55,
            size=80.0,
            stake_usd=44.0,
            status="submitted",
            dry_run=True,
        ))

        await engine._confirm_pending_orders()

        engine._ws_feed.subscribe.assert_called_with("token-ws")

    @pytest.mark.asyncio
    async def test_multiple_pending_orders_all_filled(self):
        db = _make_db()
        engine = self._make_engine(db)

        for i in range(3):
            db.insert_order(OrderRecord(
                order_id=f"ord-multi-{i}",
                market_id=f"market-multi-{i}",
                token_id=f"token-multi-{i}",
                side="BUY_YES",
                price=0.50 + i * 0.05,
                size=100.0,
                stake_usd=50.0,
                status="submitted" if i % 2 == 0 else "pending",
                dry_run=True,
            ))

        await engine._confirm_pending_orders()

        for i in range(3):
            order = db.get_order(f"ord-multi-{i}")
            assert order.status == "filled"

        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 3

    @pytest.mark.asyncio
    async def test_no_db_is_noop(self):
        """Engine without DB should not crash."""
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = None
        engine._ws_feed = MagicMock()

        await engine._confirm_pending_orders()  # should not crash

    @pytest.mark.asyncio
    async def test_filled_orders_not_reprocessed(self):
        """Orders already filled should not be picked up."""
        db = _make_db()
        engine = self._make_engine(db)

        db.insert_order(OrderRecord(
            order_id="ord-done",
            market_id="market-done",
            token_id="token-done",
            side="BUY_YES",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="filled",
            dry_run=True,
        ))

        await engine._confirm_pending_orders()

        # No new trades should be created (order was already filled)
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0


# ── _stage_execute_order Integration Tests ──────────────────────


class TestStageExecuteOrderBranching:
    """Integration tests verifying _stage_execute_order branches correctly."""

    def _make_engine(self, db):
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    def _make_ctx(self, market_id="market-test"):
        from src.engine.loop import PipelineContext
        market = MagicMock()
        market.tokens = [
            MagicMock(outcome="Yes", token_id="tok-yes", price=0.50),
            MagicMock(outcome="No", token_id="tok-no", price=0.50),
        ]
        market.market_type = "binary"
        ctx = PipelineContext(
            market=market,
            cycle_id=1,
            market_id=market_id,
            question="Test question?",
        )
        ctx.forecast = FakeForecast()
        ctx.edge_result = FakeEdgeResult()
        ctx.position = FakePositionSize()
        return ctx

    @pytest.mark.asyncio
    async def test_simulated_order_creates_position_immediately(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()

        simulated_result = _make_order_result(
            status="simulated", fill_price=0.50, fill_size=100.0,
        )

        with patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls, \
             patch("src.execution.order_builder.build_order") as mock_build:
            mock_build.return_value = [_make_order_spec()]
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=simulated_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            await engine._stage_execute_order(ctx)

        assert ctx.result["trade_executed"] is True
        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 1

    @pytest.mark.asyncio
    async def test_submitted_order_no_position(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()

        submitted_result = _make_order_result(
            status="submitted", fill_price=0.0, fill_size=0.0,
            clob_order_id="clob-new",
        )

        with patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls, \
             patch("src.execution.order_builder.build_order") as mock_build:
            mock_build.return_value = [_make_order_spec()]
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=submitted_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            await engine._stage_execute_order(ctx)

        assert ctx.result.get("trade_executed") is not True
        assert ctx.result.get("trade_attempted") is True

        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "submitted"

    @pytest.mark.asyncio
    async def test_failed_order_no_position_or_trade(self):
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()

        failed_result = _make_order_result(
            status="failed", fill_price=0.0, fill_size=0.0,
            error="Rejected by CLOB",
        )

        with patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls, \
             patch("src.execution.order_builder.build_order") as mock_build:
            mock_build.return_value = [_make_order_spec()]
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(return_value=failed_result)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            await engine._stage_execute_order(ctx)

        positions = db._conn.execute("SELECT * FROM positions").fetchall()
        assert len(positions) == 0

        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 0

        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "failed"
        assert "Rejected" in orders[0].error


# ── Config Defaults Tests ───────────────────────────────────────


class TestConfigDefaults:
    def test_live_exit_routing_default_false(self):
        config = ExecutionConfig()
        assert config.live_exit_routing is False

    def test_reconciliation_enabled_default_false(self):
        config = ExecutionConfig()
        assert config.reconciliation_enabled is False

    def test_reconciliation_interval_default(self):
        config = ExecutionConfig()
        assert config.reconciliation_interval_secs == 30

    def test_stale_order_cancel_default_false(self):
        config = ExecutionConfig()
        assert config.stale_order_cancel_enabled is False


# ── TWAP Mixed Status Tests ────────────────────────────────────


class TestTWAPMixedStatus:
    """Test handling of TWAP orders where slices have different fill statuses."""

    def _make_engine(self, db):
        with patch("src.engine.loop.load_config") as mock_load:
            cfg = load_config()
            mock_load.return_value = cfg
            with patch("src.engine.loop.DrawdownManager"):
                with patch("src.engine.loop.PortfolioRiskManager"):
                    from src.engine.loop import TradingEngine
                    engine = TradingEngine(config=cfg)
        engine._db = db
        engine._ws_feed = MagicMock()
        return engine

    def _make_ctx(self):
        from src.engine.loop import PipelineContext
        market = MagicMock()
        market.tokens = [
            MagicMock(outcome="Yes", token_id="tok-yes", price=0.50),
        ]
        market.market_type = "binary"
        ctx = PipelineContext(
            market=market, cycle_id=1, market_id="market-twap",
            question="TWAP test?",
        )
        ctx.forecast = FakeForecast()
        ctx.edge_result = FakeEdgeResult()
        ctx.position = FakePositionSize()
        return ctx

    @pytest.mark.asyncio
    async def test_twap_mixed_filled_and_submitted(self):
        """First TWAP slice filled, second submitted — only partial position."""
        db = _make_db()
        engine = self._make_engine(db)
        ctx = self._make_ctx()

        filled_result = _make_order_result(
            status="filled", fill_price=0.50, fill_size=50.0,
        )
        submitted_result = _make_order_result(
            status="submitted", fill_price=0.0, fill_size=0.0,
            clob_order_id="clob-twap-2",
        )

        call_count = 0

        async def side_effect(order):
            nonlocal call_count
            call_count += 1
            return filled_result if call_count == 1 else submitted_result

        with patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls, \
             patch("src.execution.order_builder.build_order") as mock_build:
            mock_build.return_value = [
                _make_order_spec(order_id="slice-1"),
                _make_order_spec(order_id="slice-2"),
            ]
            mock_router = AsyncMock()
            mock_router.submit_order = AsyncMock(side_effect=side_effect)
            mock_router_cls.return_value = mock_router
            mock_clob = AsyncMock()
            mock_clob_cls.return_value = mock_clob

            await engine._stage_execute_order(ctx)

        # Should have both trade_executed and trade_attempted
        assert ctx.result["trade_executed"] is True
        assert ctx.result.get("trade_attempted") is True

        # 1 trade (filled) + 1 open_order (submitted)
        trades = db._conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) == 1

        orders = db.get_open_orders()
        assert len(orders) == 1
        assert orders[0].status == "submitted"
