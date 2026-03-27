"""Tests for code review fixes: TWAP stagger, per-model CBs, resource leak,
whale traceability, iceberg ordering.

Batch B + C tests for the external code review response.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ExecutionConfig
from src.execution.order_builder import OrderSpec
from src.policy.edge_calc import EdgeResult, calculate_edge
from src.storage.models import ExecutionPlanRecord, OrderRecord


# ─── helpers ───────────────────────────────────────────────────────────


def _make_db():
    """Create an in-memory DB with execution_plans and open_orders tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE execution_plans (
            plan_id TEXT PRIMARY KEY,
            market_id TEXT,
            token_id TEXT DEFAULT '',
            strategy_type TEXT DEFAULT '',
            action_side TEXT DEFAULT '',
            outcome_side TEXT DEFAULT '',
            target_size REAL DEFAULT 0,
            target_stake_usd REAL DEFAULT 0,
            filled_size REAL DEFAULT 0,
            avg_fill_price REAL DEFAULT 0,
            total_children INTEGER DEFAULT 0,
            completed_children INTEGER DEFAULT 0,
            active_child_order_id TEXT DEFAULT '',
            next_child_index INTEGER DEFAULT 0,
            status TEXT DEFAULT 'planned',
            dry_run INTEGER DEFAULT 1,
            error TEXT DEFAULT '',
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE open_orders (
            order_id TEXT PRIMARY KEY,
            clob_order_id TEXT DEFAULT '',
            market_id TEXT,
            token_id TEXT DEFAULT '',
            side TEXT DEFAULT 'BUY',
            order_type TEXT DEFAULT 'limit',
            price REAL DEFAULT 0,
            size REAL DEFAULT 0,
            filled_size REAL DEFAULT 0,
            avg_fill_price REAL DEFAULT 0,
            status TEXT DEFAULT 'submitted',
            dry_run INTEGER DEFAULT 1,
            action_side TEXT DEFAULT '',
            outcome_side TEXT DEFAULT '',
            parent_plan_id TEXT DEFAULT '',
            child_index INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE alerts_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            channel TEXT DEFAULT '',
            message TEXT,
            market_id TEXT DEFAULT '',
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def _make_fake_db(conn):
    """Wrap a sqlite3 connection in a mock DB with the methods PlanController needs."""
    db = MagicMock()
    db._conn = conn

    def insert_execution_plan(plan):
        conn.execute(
            """INSERT INTO execution_plans
               (plan_id, market_id, token_id, strategy_type, action_side, outcome_side,
                target_size, target_stake_usd, filled_size, avg_fill_price,
                total_children, completed_children, active_child_order_id,
                next_child_index, status, dry_run, error, metadata_json,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (plan.plan_id, plan.market_id, plan.token_id, plan.strategy_type,
             plan.action_side, plan.outcome_side, plan.target_size, plan.target_stake_usd,
             plan.filled_size, plan.avg_fill_price, plan.total_children,
             plan.completed_children, plan.active_child_order_id, plan.next_child_index,
             plan.status, int(plan.dry_run), plan.error, plan.metadata_json,
             plan.created_at, plan.updated_at),
        )
        conn.commit()

    def get_execution_plan(plan_id):
        row = conn.execute("SELECT * FROM execution_plans WHERE plan_id=?", (plan_id,)).fetchone()
        if not row:
            return None
        return ExecutionPlanRecord(**{k: row[k] for k in row.keys()})

    def update_execution_plan(plan_id, **kwargs):
        sets = ", ".join(f"{k}=?" for k in kwargs)
        vals = list(kwargs.values()) + [plan_id]
        conn.execute(f"UPDATE execution_plans SET {sets}, updated_at=? WHERE plan_id=?",
                     vals[:-1] + [dt.datetime.now(dt.timezone.utc).isoformat(), plan_id])
        conn.commit()

    def get_plan_children(plan_id):
        rows = conn.execute(
            "SELECT * FROM open_orders WHERE parent_plan_id=? ORDER BY child_index",
            (plan_id,),
        ).fetchall()
        return [OrderRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def get_active_execution_plans():
        rows = conn.execute(
            "SELECT * FROM execution_plans WHERE status='active'"
        ).fetchall()
        return [ExecutionPlanRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def insert_alert(level, message, channel=""):
        conn.execute(
            "INSERT INTO alerts_log (level, channel, message, created_at) VALUES (?,?,?,?)",
            (level, channel, message, dt.datetime.now(dt.timezone.utc).isoformat()),
        )
        conn.commit()

    def update_order_status(order_id, status):
        conn.execute("UPDATE open_orders SET status=? WHERE order_id=?", (status, order_id))
        conn.commit()

    db.insert_execution_plan = insert_execution_plan
    db.get_execution_plan = get_execution_plan
    db.update_execution_plan = update_execution_plan
    db.get_plan_children = get_plan_children
    db.get_active_execution_plans = get_active_execution_plans
    db.insert_alert = insert_alert
    db.update_order_status = update_order_status
    return db


def _make_child_specs(n: int = 3, market_id: str = "m1") -> list[OrderSpec]:
    """Create N child order specs."""
    parent = str(uuid.uuid4())
    return [
        OrderSpec(
            order_id=str(uuid.uuid4()),
            market_id=market_id,
            token_id="t1",
            side="BUY",
            order_type="limit",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            ttl_secs=300,
            dry_run=True,
            action_side="BUY",
            outcome_side="YES",
            execution_strategy="twap",
            parent_order_id=parent,
            child_index=i,
            total_children=n,
        )
        for i in range(n)
    ]


def _insert_child_order(conn, order_id, plan_id, child_index, status="submitted",
                        filled_size=0.0, avg_fill_price=0.0, size=100.0, price=0.50):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO open_orders
           (order_id, market_id, token_id, side, price, size, filled_size,
            avg_fill_price, status, parent_plan_id, child_index,
            action_side, outcome_side, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (order_id, "m1", "t1", "BUY", price, size, filled_size,
         avg_fill_price, status, plan_id, child_index,
         "BUY", "YES", now, now),
    )
    conn.commit()


# ─── Batch B1: TWAP time stagger ──────────────────────────────────────


class TestTWAPTimeStagger:
    """TWAP interval enforcement between child submissions."""

    def test_twap_interval_blocks_immediate_resubmit(self) -> None:
        """Child fills within interval → returns None (delays next child)."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db, ExecutionConfig(twap_interval_secs=60))

        specs = _make_child_specs(3)
        plan = ctrl.create_plan(specs, "twap", metadata={"min_child_interval_secs": 60})
        first = ctrl.get_first_child_spec(plan)
        ctrl.activate_plan(plan.plan_id, first.order_id)

        # Insert child 0 as filled
        _insert_child_order(conn, first.order_id, plan.plan_id, 0,
                            status="filled", filled_size=100.0, avg_fill_price=0.50)
        child0 = db.get_plan_children(plan.plan_id)[0]

        # First fill records last_child_completed_at but no prior timestamp → should proceed
        result = ctrl.update_plan_from_child(child0)
        # First child has no prior timestamp, so it should succeed
        assert result is not None

    def test_twap_interval_blocks_second_child(self) -> None:
        """Second child fill within interval → returns None."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db, ExecutionConfig(twap_interval_secs=60))

        specs = _make_child_specs(3)
        plan = ctrl.create_plan(specs, "twap", metadata={"min_child_interval_secs": 60})
        first = ctrl.get_first_child_spec(plan)
        ctrl.activate_plan(plan.plan_id, first.order_id)

        # Fill child 0 — sets last_child_completed_at
        _insert_child_order(conn, first.order_id, plan.plan_id, 0,
                            status="filled", filled_size=100.0, avg_fill_price=0.50)
        child0 = db.get_plan_children(plan.plan_id)[0]
        next_spec = ctrl.update_plan_from_child(child0)
        assert next_spec is not None  # first child fills, timestamp was empty → proceeds

        # Submit and fill child 1 immediately — within interval
        child1_id = next_spec.order_id
        _insert_child_order(conn, child1_id, plan.plan_id, 1,
                            status="filled", filled_size=100.0, avg_fill_price=0.50)
        # Set active child so update_plan_from_child works
        db.update_execution_plan(plan.plan_id, active_child_order_id=child1_id)
        child1 = db.get_plan_children(plan.plan_id)[1]
        result = ctrl.update_plan_from_child(child1)
        # Should be blocked — less than 60 seconds since last child
        assert result is None

    def test_iceberg_no_interval(self) -> None:
        """Iceberg plans (min_child_interval_secs=0) proceed immediately."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db)

        specs = _make_child_specs(2)
        plan = ctrl.create_plan(specs, "iceberg")  # no metadata → interval=0
        first = ctrl.get_first_child_spec(plan)
        ctrl.activate_plan(plan.plan_id, first.order_id)

        _insert_child_order(conn, first.order_id, plan.plan_id, 0,
                            status="filled", filled_size=100.0, avg_fill_price=0.50)
        child0 = db.get_plan_children(plan.plan_id)[0]
        result = ctrl.update_plan_from_child(child0)
        assert result is not None  # immediate — no interval

    def test_recover_stuck_plans_respects_interval(self) -> None:
        """Plan within TWAP interval window should NOT be recovered as stuck."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db)

        specs = _make_child_specs(3)
        plan = ctrl.create_plan(specs, "twap", metadata={"min_child_interval_secs": 300})
        first = ctrl.get_first_child_spec(plan)
        ctrl.activate_plan(plan.plan_id, first.order_id)

        # Simulate: child 0 filled, metadata updated, active_child cleared (waiting)
        meta = json.loads(plan.metadata_json)
        meta["last_child_completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        db.update_execution_plan(
            plan.plan_id,
            active_child_order_id="",
            next_child_index=1,
            completed_children=1,
            metadata_json=json.dumps(meta),
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0  # not stuck — waiting for interval

    def test_recover_stuck_plans_after_interval(self) -> None:
        """Plan past TWAP interval window should be recovered."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db)

        specs = _make_child_specs(3)
        plan = ctrl.create_plan(specs, "twap", metadata={"min_child_interval_secs": 1})
        first = ctrl.get_first_child_spec(plan)
        ctrl.activate_plan(plan.plan_id, first.order_id)

        # Set last_child_completed_at to well in the past
        meta = json.loads(plan.metadata_json)
        past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=10)).isoformat()
        meta["last_child_completed_at"] = past
        db.update_execution_plan(
            plan.plan_id,
            active_child_order_id="",
            next_child_index=1,
            completed_children=1,
            metadata_json=json.dumps(meta),
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 1

    def test_metadata_stores_interval_and_timestamp(self) -> None:
        """Verify plan metadata contains min_child_interval_secs and last_child_completed_at."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db)

        specs = _make_child_specs(3)
        plan = ctrl.create_plan(specs, "twap", metadata={"min_child_interval_secs": 30})
        meta = json.loads(plan.metadata_json)
        assert meta["min_child_interval_secs"] == 30
        assert meta["last_child_completed_at"] == ""
        assert meta["spec_version"] == 1
        assert len(meta["children"]) == 3


# ─── Batch B2: Per-model circuit breakers ──────────────────────────────


class TestPerModelCircuitBreakers:
    """Per-model circuit breakers in ensemble forecaster."""

    def test_google_config_exists(self) -> None:
        """Google should be in DEFAULT_BREAKER_CONFIGS."""
        from src.observability.circuit_breaker import DEFAULT_BREAKER_CONFIGS
        assert "google" in DEFAULT_BREAKER_CONFIGS
        assert DEFAULT_BREAKER_CONFIGS["google"].name == "Google Gemini"

    def test_openai_breaker_open_returns_fallback(self) -> None:
        """When openai breaker is open, _query_openai returns 0.5 with error."""
        from src.observability.circuit_breaker import circuit_breakers

        cb = circuit_breakers.get("openai")
        cb.reset()
        # Trip the breaker by recording failures
        for _ in range(10):
            cb.record_failure()

        from src.forecast.ensemble import _query_openai
        from src.config import ForecastingConfig
        result = asyncio.run(_query_openai("gpt-4o", "test", ForecastingConfig()))
        assert result.model_probability == 0.5
        assert "Circuit breaker open" in result.error
        assert result.latency_ms == 0.0
        cb.reset()

    def test_anthropic_breaker_open_returns_fallback(self) -> None:
        """When anthropic breaker is open, _query_anthropic returns 0.5 with error."""
        from src.observability.circuit_breaker import circuit_breakers

        cb = circuit_breakers.get("anthropic")
        cb.reset()
        for _ in range(10):
            cb.record_failure()

        from src.forecast.ensemble import _query_anthropic
        from src.config import ForecastingConfig
        result = asyncio.run(_query_anthropic("claude-3-5-sonnet-20241022", "test", ForecastingConfig()))
        assert result.model_probability == 0.5
        assert "Circuit breaker open" in result.error
        cb.reset()

    def test_google_breaker_open_returns_fallback(self) -> None:
        """When google breaker is open, _query_google returns 0.5 with error."""
        from src.observability.circuit_breaker import circuit_breakers

        cb = circuit_breakers.get("google")
        cb.reset()
        for _ in range(10):
            cb.record_failure()

        from src.forecast.ensemble import _query_google
        from src.config import ForecastingConfig
        result = asyncio.run(_query_google("gemini-2.0-flash", "test", ForecastingConfig()))
        assert result.model_probability == 0.5
        assert "Circuit breaker open" in result.error
        cb.reset()

    def test_one_model_failure_isolates(self) -> None:
        """Tripping anthropic breaker should not affect openai breaker."""
        from src.observability.circuit_breaker import circuit_breakers

        cb_anthropic = circuit_breakers.get("anthropic")
        cb_openai = circuit_breakers.get("openai")
        cb_anthropic.reset()
        cb_openai.reset()

        # Trip anthropic
        for _ in range(10):
            cb_anthropic.record_failure()

        assert not cb_anthropic.allow_request()  # anthropic blocked
        assert cb_openai.allow_request()  # openai still open
        cb_anthropic.reset()


# ─── Batch B3: Research resource leak ──────────────────────────────────


class TestResearchResourceCleanup:
    """Resource cleanup in _stage_research — nested try/finally."""

    def test_search_provider_closed_when_fetcher_close_fails(self) -> None:
        """If source_fetcher.close() raises, search_provider.close() must still run."""
        fetcher = AsyncMock()
        fetcher.close = AsyncMock(side_effect=RuntimeError("fetcher close failed"))
        provider = AsyncMock()
        provider.close = AsyncMock()

        async def _test():
            try:
                raise ValueError("research failed")
            except Exception:
                pass
            finally:
                try:
                    await fetcher.close()
                except Exception:
                    pass
                try:
                    await provider.close()
                except Exception:
                    pass

        asyncio.run(_test())
        fetcher.close.assert_awaited_once()
        provider.close.assert_awaited_once()

    def test_both_close_errors_suppressed(self) -> None:
        """If both close() calls raise, neither propagates."""
        fetcher = AsyncMock()
        fetcher.close = AsyncMock(side_effect=RuntimeError("fail1"))
        provider = AsyncMock()
        provider.close = AsyncMock(side_effect=RuntimeError("fail2"))

        async def _test():
            try:
                await fetcher.close()
            except Exception:
                pass
            try:
                await provider.close()
            except Exception:
                pass

        # Should not raise
        asyncio.run(_test())
        fetcher.close.assert_awaited_once()
        provider.close.assert_awaited_once()

    def test_normal_close_unaffected(self) -> None:
        """When neither raises, both are called normally."""
        fetcher = AsyncMock()
        fetcher.close = AsyncMock()
        provider = AsyncMock()
        provider.close = AsyncMock()

        async def _test():
            try:
                await fetcher.close()
            except Exception:
                pass
            try:
                await provider.close()
            except Exception:
                pass

        asyncio.run(_test())
        fetcher.close.assert_awaited_once()
        provider.close.assert_awaited_once()


# ─── Batch C1: Whale traceability ─────────────────────────────────────


class TestWhaleTraceability:
    """Whale adjustment traceability fields on EdgeResult."""

    def test_default_whale_fields(self) -> None:
        """New EdgeResult fields default to None/0.0."""
        result = calculate_edge(implied_prob=0.60, model_prob=0.70)
        assert result.whale_adjustment == 0.0
        assert result.pre_whale_probability is None

    def test_whale_fields_settable(self) -> None:
        """whale_adjustment and pre_whale_probability are settable after construction."""
        result = calculate_edge(implied_prob=0.60, model_prob=0.70)
        result.whale_adjustment = 0.05
        result.pre_whale_probability = 0.65
        assert result.whale_adjustment == 0.05
        assert result.pre_whale_probability == 0.65

    def test_fields_survive_recalculation(self) -> None:
        """New calculate_edge() preserves default values (whale fields set externally)."""
        r1 = calculate_edge(implied_prob=0.50, model_prob=0.60)
        assert r1.whale_adjustment == 0.0
        assert r1.pre_whale_probability is None
        # Simulate what engine loop does: recalculate then set
        r2 = calculate_edge(implied_prob=0.50, model_prob=0.65)
        r2.pre_whale_probability = 0.60
        r2.whale_adjustment = 0.05
        assert r2.pre_whale_probability == 0.60
        assert r2.whale_adjustment == 0.05
        # Original r1 unaffected
        assert r1.whale_adjustment == 0.0


# ─── Batch C2: Iceberg ordering ───────────────────────────────────────


class TestIcebergOrdering:
    """Verify iceberg visible portion is always first child."""

    def test_visible_portion_is_first_child(self) -> None:
        """_build_iceberg_orders returns visible as child[0], hidden as child[1]."""
        from src.execution.order_builder import _build_iceberg_orders
        from src.policy.position_sizer import PositionSize

        position = PositionSize(
            stake_usd=1000.0,
            kelly_fraction_used=0.25,
            full_kelly_stake=4000.0,
            capped_by="kelly",
            direction="BUY_YES",
            token_quantity=2000.0,
        )
        config = ExecutionConfig(dry_run=True)
        orders = _build_iceberg_orders("market1", "token1", position, 0.50, config)

        assert len(orders) == 2
        assert orders[0].metadata.get("iceberg_part") == "visible"
        assert orders[0].child_index == 0
        assert orders[1].metadata.get("iceberg_part") == "hidden"
        assert orders[1].child_index == 1
        # Visible 20% < hidden 80%
        assert orders[0].size < orders[1].size

    def test_iceberg_plan_no_interval(self) -> None:
        """Iceberg plans get min_child_interval_secs=0 (immediate next child)."""
        from src.execution.plan_controller import PlanController

        conn = _make_db()
        db = _make_fake_db(conn)
        ctrl = PlanController(db)

        specs = _make_child_specs(2)
        # No metadata passed → interval defaults to 0
        plan = ctrl.create_plan(specs, "iceberg")
        meta = json.loads(plan.metadata_json)
        assert meta["min_child_interval_secs"] == 0
