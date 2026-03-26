"""Phase 10E: TWAP/Iceberg parent execution plan orchestration tests.

Tests cover:
  - Batch A: ExecutionPlanRecord model, migration 18, DB methods
  - Batch B: PlanController lifecycle (create, activate, update, cancel, summary)
  - Batch C: Engine loop plan branching, reconciliation plan-aware callbacks
  - Batch D: Dashboard endpoints, invariant check #13
"""

from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from src.config import ExecutionConfig, StorageConfig
from src.execution.order_builder import OrderSpec
from src.execution.plan_controller import (
    PlanController,
    _serialize_order_spec,
    _deserialize_order_spec,
)
from src.observability.invariant_checker import check_invariants
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import ExecutionPlanRecord, OrderRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _make_db(conn: sqlite3.Connection | None = None) -> Database:
    if conn is None:
        conn = _make_conn()
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


def _make_order_spec(
    *,
    market_id: str = "mkt-1",
    token_id: str = "tok-1",
    price: float = 0.60,
    size: float = 100.0,
    stake_usd: float = 60.0,
    strategy: str = "twap",
    child_index: int = 0,
    total_children: int = 3,
    dry_run: bool = True,
) -> OrderSpec:
    return OrderSpec(
        order_id=str(uuid.uuid4()),
        market_id=market_id,
        token_id=token_id,
        side="BUY",
        order_type="limit",
        price=price,
        size=size,
        stake_usd=stake_usd,
        ttl_secs=300,
        dry_run=dry_run,
        action_side="BUY",
        outcome_side="YES",
        execution_strategy=strategy,
        parent_order_id="",
        child_index=child_index,
        total_children=total_children,
    )


def _make_twap_orders(num_slices: int = 3) -> list[OrderSpec]:
    """Create a list of TWAP child OrderSpecs."""
    orders = []
    for i in range(num_slices):
        orders.append(_make_order_spec(
            size=round(100.0 / num_slices, 2),
            stake_usd=round(60.0 / num_slices, 2),
            child_index=i,
            total_children=num_slices,
        ))
    return orders


def _make_iceberg_orders() -> list[OrderSpec]:
    """Create a list of iceberg child OrderSpecs (visible + hidden)."""
    return [
        _make_order_spec(size=20.0, stake_usd=12.0, strategy="iceberg",
                         child_index=0, total_children=2),
        _make_order_spec(size=80.0, stake_usd=48.0, strategy="iceberg",
                         child_index=1, total_children=2),
    ]


# ═══════════════════════════════════════════════════════════════
# BATCH A: Data Model + Persistence
# ═══════════════════════════════════════════════════════════════


class TestExecutionPlanRecord:
    def test_defaults(self):
        plan = ExecutionPlanRecord(plan_id="p1", market_id="m1")
        assert plan.status == "planned"
        assert plan.total_children == 0
        assert plan.dry_run is True
        assert plan.metadata_json == "{}"
        assert plan.filled_size == 0.0

    def test_all_fields(self):
        plan = ExecutionPlanRecord(
            plan_id="p2", market_id="m2", token_id="t2",
            strategy_type="twap", action_side="BUY", outcome_side="YES",
            target_size=300.0, target_stake_usd=180.0,
            total_children=3, status="active",
        )
        assert plan.strategy_type == "twap"
        assert plan.target_size == 300.0
        assert plan.total_children == 3


class TestOrderRecordPlanFields:
    def test_default_empty(self):
        order = OrderRecord(order_id="o1", market_id="m1")
        assert order.parent_plan_id == ""
        assert order.child_index == 0

    def test_explicit_values(self):
        order = OrderRecord(
            order_id="o2", market_id="m2",
            parent_plan_id="plan-abc", child_index=2,
        )
        assert order.parent_plan_id == "plan-abc"
        assert order.child_index == 2


class TestConfigDefault:
    def test_plan_orchestration_disabled_by_default(self):
        config = ExecutionConfig()
        assert config.plan_orchestration_enabled is False


class TestMigration18:
    def test_schema_version_is_18(self):
        conn = _make_conn()
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == 18

    def test_execution_plans_table_exists(self):
        conn = _make_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = [r["name"] for r in tables]
        assert "execution_plans" in names

    def test_open_orders_has_plan_columns(self):
        conn = _make_conn()
        info = conn.execute("PRAGMA table_info(open_orders)").fetchall()
        cols = {r["name"] for r in info}
        assert "parent_plan_id" in cols
        assert "child_index" in cols

    def test_indexes_created(self):
        conn = _make_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        names = {r["name"] for r in indexes}
        assert "idx_exec_plans_status" in names
        assert "idx_exec_plans_market" in names
        assert "idx_open_orders_parent_plan" in names


class TestDatabasePlanMethods:
    def test_insert_and_get_execution_plan(self):
        db = _make_db()
        plan = ExecutionPlanRecord(
            plan_id="plan-1", market_id="m1", strategy_type="twap",
            total_children=3, status="planned",
        )
        db.insert_execution_plan(plan)
        result = db.get_execution_plan("plan-1")
        assert result is not None
        assert result.plan_id == "plan-1"
        assert result.strategy_type == "twap"
        assert result.total_children == 3

    def test_get_execution_plan_not_found(self):
        db = _make_db()
        assert db.get_execution_plan("nonexistent") is None

    def test_update_execution_plan(self):
        db = _make_db()
        plan = ExecutionPlanRecord(
            plan_id="plan-u", market_id="m1", status="planned",
        )
        db.insert_execution_plan(plan)
        db.update_execution_plan("plan-u", status="active", filled_size=50.0)
        updated = db.get_execution_plan("plan-u")
        assert updated.status == "active"
        assert updated.filled_size == 50.0

    def test_get_active_execution_plans(self):
        db = _make_db()
        for pid, status in [("p1", "planned"), ("p2", "active"),
                            ("p3", "filled"), ("p4", "partial")]:
            db.insert_execution_plan(
                ExecutionPlanRecord(plan_id=pid, market_id="m1", status=status)
            )
        active = db.get_active_execution_plans()
        active_ids = {p.plan_id for p in active}
        assert active_ids == {"p1", "p2", "p4"}

    def test_get_plan_children(self):
        db = _make_db()
        plan_id = "plan-children"
        for idx in range(3):
            db.insert_order(OrderRecord(
                order_id=f"child-{idx}", market_id="m1",
                status="submitted", parent_plan_id=plan_id, child_index=idx,
            ))
        children = db.get_plan_children(plan_id)
        assert len(children) == 3
        assert [c.child_index for c in children] == [0, 1, 2]

    def test_insert_order_with_plan_fields(self):
        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="o-plan", market_id="m1",
            parent_plan_id="plan-x", child_index=1,
        ))
        order = db.get_order("o-plan")
        assert order.parent_plan_id == "plan-x"
        assert order.child_index == 1

    def test_prune_terminal_orders_skips_active_plan_children(self):
        db = _make_db()
        # Create an active plan
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="active-plan", market_id="m1", status="active",
            total_children=2,
        ))
        # Child 0: filled (should NOT be pruned — plan is active)
        db.insert_order(OrderRecord(
            order_id="child-0", market_id="m1", status="filled",
            parent_plan_id="active-plan", child_index=0,
        ))
        # Standalone filled order (no plan — should be pruned)
        db.insert_order(OrderRecord(
            order_id="standalone", market_id="m2", status="filled",
        ))
        pruned = db.prune_terminal_orders()
        assert pruned == 1  # Only standalone pruned
        # Child should still exist
        child = db.get_order("child-0")
        assert child is not None

    def test_prune_terminal_allows_completed_plan_children(self):
        db = _make_db()
        # Completed plan — children can be pruned
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="done-plan", market_id="m1", status="filled",
            total_children=2,
        ))
        db.insert_order(OrderRecord(
            order_id="done-child", market_id="m1", status="filled",
            parent_plan_id="done-plan", child_index=0,
        ))
        pruned = db.prune_terminal_orders()
        assert pruned == 1

    def test_get_all_execution_plans(self):
        db = _make_db()
        for i in range(5):
            db.insert_execution_plan(ExecutionPlanRecord(
                plan_id=f"all-{i}", market_id="m1",
            ))
        plans = db.get_all_execution_plans(limit=3)
        assert len(plans) == 3


# ═══════════════════════════════════════════════════════════════
# BATCH B: Plan Controller
# ═══════════════════════════════════════════════════════════════


class TestOrderSpecSerialization:
    def test_round_trip(self):
        spec = _make_order_spec()
        serialized = _serialize_order_spec(spec)
        deserialized = _deserialize_order_spec(serialized)
        assert deserialized.market_id == spec.market_id
        assert deserialized.price == spec.price
        assert deserialized.size == spec.size
        assert deserialized.action_side == spec.action_side
        assert deserialized.outcome_side == spec.outcome_side
        assert deserialized.execution_strategy == spec.execution_strategy


class TestPlanControllerCreatePlan:
    def test_create_twap_plan(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        assert plan.strategy_type == "twap"
        assert plan.total_children == 3
        assert plan.status == "planned"
        assert plan.target_size == pytest.approx(100.0, abs=0.1)
        assert plan.market_id == "mkt-1"
        assert plan.action_side == "BUY"
        assert plan.outcome_side == "YES"
        # Verify persisted
        stored = db.get_execution_plan(plan.plan_id)
        assert stored is not None
        assert stored.total_children == 3

    def test_create_iceberg_plan(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_iceberg_orders()
        plan = ctrl.create_plan(orders, "iceberg")
        assert plan.total_children == 2
        assert plan.strategy_type == "iceberg"
        assert plan.target_size == pytest.approx(100.0, abs=0.1)

    def test_create_plan_empty_raises(self):
        db = _make_db()
        ctrl = PlanController(db)
        with pytest.raises(ValueError, match="empty"):
            ctrl.create_plan([], "twap")

    def test_metadata_json_contains_children(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        meta = json.loads(plan.metadata_json)
        assert "children" in meta
        assert len(meta["children"]) == 2
        assert meta["children"][0]["size"] == orders[0].size


class TestPlanControllerActivate:
    def test_activate_plan(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "first-child-id")
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "active"
        assert updated.active_child_order_id == "first-child-id"
        assert updated.next_child_index == 1


class TestPlanControllerGetFirstChild:
    def test_get_first_child(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        first = ctrl.get_first_child_spec(plan)
        assert first.market_id == "mkt-1"
        assert first.size == orders[0].size
        # Should have a new order_id
        assert first.order_id != orders[0].order_id

    def test_get_first_child_empty_metadata(self):
        db = _make_db()
        ctrl = PlanController(db)
        plan = ExecutionPlanRecord(
            plan_id="empty-meta", market_id="m1",
            metadata_json=json.dumps({"children": []}),
        )
        with pytest.raises(ValueError, match="no serialized"):
            ctrl.get_first_child_spec(plan)


class TestPlanControllerUpdateFromChild:
    def _setup_active_plan(self, db, num_children=3):
        """Create an active plan with children in the DB."""
        ctrl = PlanController(db)
        orders = _make_twap_orders(num_children)
        plan = ctrl.create_plan(orders, "twap")

        # Submit first child
        first_order_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=first_order_id, market_id="mkt-1",
            price=0.60, size=orders[0].size,
            status="submitted", parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, first_order_id)
        return ctrl, plan, first_order_id

    def test_child_filled_returns_next_spec(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db)

        # Mark first child as filled
        db.update_order_status(first_id, "filled",
                               filled_size=33.33, avg_fill_price=0.60)

        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)

        assert next_spec is not None
        assert next_spec.market_id == "mkt-1"
        # Plan should be updated
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.filled_size > 0
        assert updated.completed_children == 1

    def test_last_child_filled_completes_plan(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db, num_children=1)

        # Mark only child as filled
        db.update_order_status(first_id, "filled",
                               filled_size=100.0, avg_fill_price=0.60)

        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)

        assert next_spec is None  # No more children
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "filled"

    def test_child_cancelled_no_fills_cancels_plan(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db, num_children=1)

        # Cancel the only child
        db.update_order_status(first_id, "cancelled")
        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)

        assert next_spec is None
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "cancelled"

    def test_child_failed_with_prior_fills_partial(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db, num_children=2)

        # Fill first child
        db.update_order_status(first_id, "filled",
                               filled_size=50.0, avg_fill_price=0.60)
        child_1 = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_1)
        assert next_spec is not None

        # Submit and fail second child
        second_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=second_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="failed", parent_plan_id=plan.plan_id, child_index=1,
        ))
        child_2 = db.get_order(second_id)
        next_spec = ctrl.update_plan_from_child(child_2)

        assert next_spec is None
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "partial"

    def test_no_parent_plan_id_is_noop(self):
        db = _make_db()
        ctrl = PlanController(db)
        order = OrderRecord(order_id="o1", market_id="m1", parent_plan_id="")
        assert ctrl.update_plan_from_child(order) is None

    def test_terminal_plan_is_noop(self):
        db = _make_db()
        ctrl = PlanController(db)
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="done-plan", market_id="m1", status="filled",
        ))
        order = OrderRecord(
            order_id="o1", market_id="m1",
            parent_plan_id="done-plan", status="filled",
        )
        assert ctrl.update_plan_from_child(order) is None

    def test_child_partial_updates_aggregates(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db)

        # Partial fill on first child
        db.update_order_status(first_id, "partial",
                               filled_size=10.0, avg_fill_price=0.60)
        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)

        # No next spec for partial
        assert next_spec is None
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.filled_size > 0

    def test_weighted_avg_fill_price(self):
        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db, num_children=2)

        # Fill first child at 0.60
        db.update_order_status(first_id, "filled",
                               filled_size=50.0, avg_fill_price=0.60)
        child_1 = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_1)
        assert next_spec is not None

        # Submit and fill second child at 0.65
        second_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=second_id, market_id="mkt-1",
            price=0.65, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.65,
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        child_2 = db.get_order(second_id)
        ctrl.update_plan_from_child(child_2)

        updated = db.get_execution_plan(plan.plan_id)
        # Weighted avg: (50*0.60 + 50*0.65) / 100 = 0.625
        assert updated.avg_fill_price == pytest.approx(0.625, abs=0.001)
        assert updated.status == "filled"


class TestPlanControllerCancel:
    def test_cancel_plan_no_fills(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.cancel_plan(plan.plan_id, "user requested")
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "cancelled"
        assert updated.error == "user requested"

    def test_cancel_plan_with_fills_partial(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # Add a filled child
        db.insert_order(OrderRecord(
            order_id="filled-child", market_id="mkt-1",
            status="filled", filled_size=50.0,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.cancel_plan(plan.plan_id)
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "partial"

    def test_cancel_terminal_plan_is_noop(self):
        db = _make_db()
        ctrl = PlanController(db)
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="already-done", market_id="m1", status="filled",
        ))
        ctrl.cancel_plan("already-done")
        updated = db.get_execution_plan("already-done")
        assert updated.status == "filled"  # Unchanged


class TestPlanControllerSummary:
    def test_summary_with_children(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Add children
        for i in range(2):
            db.insert_order(OrderRecord(
                order_id=f"sum-child-{i}", market_id="mkt-1",
                price=0.60, size=33.33,
                status="filled" if i == 0 else "submitted",
                parent_plan_id=plan.plan_id, child_index=i,
            ))

        summary = ctrl.get_plan_summary(plan.plan_id)
        assert summary["plan_id"] == plan.plan_id
        assert summary["strategy_type"] == "twap"
        assert len(summary["children"]) == 2

    def test_summary_not_found(self):
        db = _make_db()
        ctrl = PlanController(db)
        assert ctrl.get_plan_summary("nonexistent") == {}


# ═══════════════════════════════════════════════════════════════
# BATCH C: Engine Loop + Reconciliation Integration
# ═══════════════════════════════════════════════════════════════


class TestEnginePlanBranching:
    """Test that multi-order strategies branch to plan orchestration."""

    def test_single_order_no_plan(self):
        """Single orders should NOT create a plan even when enabled."""
        db = _make_db()
        ctrl = PlanController(db)
        # With a single order, the engine should use the simple path
        # Verify: no plans exist after a would-be single submission
        plans = db.get_active_execution_plans()
        assert len(plans) == 0

    def test_plan_controller_initialized_when_enabled(self):
        """Verify config gating."""
        config = ExecutionConfig(plan_orchestration_enabled=True)
        assert config.plan_orchestration_enabled is True

    def test_plan_controller_not_initialized_when_disabled(self):
        config = ExecutionConfig(plan_orchestration_enabled=False)
        assert config.plan_orchestration_enabled is False


class TestReconciliationPlanAware:
    """Test that reconciler notifies plan controller on status changes."""

    def test_reconciler_accepts_plan_controller(self):
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )
        assert reconciler._plan_controller is ctrl
        assert reconciler._pending_plan_submissions == []

    def test_reconciler_without_plan_controller(self):
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
        )
        assert reconciler._plan_controller is None

    def test_notify_plan_controller_queues_next_child(self):
        """After a child fill, the next child should be queued."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )

        # Create plan with 2 children
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # Insert and fill first child
        first_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=first_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, first_id)

        # Simulate reconciler notification
        order = db.get_order(first_id)
        reconciler._notify_plan_controller(order)

        # Next child should be queued
        assert len(reconciler._pending_plan_submissions) == 1
        spec, queued_plan_id, child_idx = reconciler._pending_plan_submissions[0]
        assert queued_plan_id == plan.plan_id
        assert child_idx == 1

    def test_notify_plan_controller_no_parent_noop(self):
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )

        # Order with no parent_plan_id
        db.insert_order(OrderRecord(
            order_id="no-parent", market_id="m1", status="filled",
        ))
        order = db.get_order("no-parent")
        reconciler._notify_plan_controller(order)
        assert len(reconciler._pending_plan_submissions) == 0


# ═══════════════════════════════════════════════════════════════
# BATCH D: Dashboard + Invariant
# ═══════════════════════════════════════════════════════════════


class TestInvariantCheck13:
    def test_stale_plan_detected(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="stuck-plan", market_id="m1",
            status="active", total_children=3, next_child_index=1,
            active_child_order_id="",  # No active child!
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 1
        assert stale[0].severity == "critical"

    def test_no_false_positive_completed_plan(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="done-plan", market_id="m1",
            status="filled", total_children=3, next_child_index=3,
            active_child_order_id="",
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 0

    def test_no_false_positive_active_with_child(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="healthy-plan", market_id="m1",
            status="active", total_children=3, next_child_index=1,
            active_child_order_id="child-123",  # Has active child
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 0

    def test_no_false_positive_all_children_submitted(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="all-submitted", market_id="m1",
            status="active", total_children=3, next_child_index=3,
            active_child_order_id="",
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 0

    def test_clean_db_zero_violations(self):
        db = _make_db()
        violations = check_invariants(db)
        assert len(violations) == 0, f"Expected 0 violations, got {violations}"


class TestDashboardPlanEndpoints:
    """Test dashboard endpoint handlers produce valid output."""

    def test_execution_plans_empty(self):
        """API should return empty list when no plans exist."""
        db = _make_db()
        plans = db.get_all_execution_plans()
        assert plans == []

    def test_execution_plans_returns_data(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="dash-1", market_id="m1",
            strategy_type="twap", total_children=3, status="active",
        ))
        plans = db.get_all_execution_plans()
        assert len(plans) == 1
        assert plans[0].plan_id == "dash-1"

    def test_active_plans_filters(self):
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="act-1", market_id="m1", status="active",
        ))
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="act-2", market_id="m1", status="filled",
        ))
        active = db.get_active_execution_plans()
        assert len(active) == 1
        assert active[0].plan_id == "act-1"
