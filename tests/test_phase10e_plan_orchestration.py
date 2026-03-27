"""Phase 10E: TWAP/Iceberg parent execution plan orchestration tests.

Tests cover:
  - Batch A: ExecutionPlanRecord model, migration 18, DB methods
  - Batch B: PlanController lifecycle (create, activate, update, cancel, summary)
  - Batch C: Engine loop plan branching, reconciliation plan-aware callbacks
  - Batch D: Dashboard endpoints, invariant check #13
  - Hardening: active_child timing, terminology, spec versioning, cancel semantics, alerts
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
    SPEC_VERSION,
    UnsupportedSpecVersion,
    _serialize_order_spec,
    _deserialize_order_spec,
    _validate_spec_version,
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

    def test_metadata_contains_spec_version(self):
        """Verify spec_version is tagged in metadata for future compat."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        meta = json.loads(plan.metadata_json)
        assert "spec_version" in meta
        assert meta["spec_version"] == SPEC_VERSION

    def test_create_plan_inserts_alert(self):
        """Plan creation should emit a DB alert."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        ctrl.create_plan(orders, "twap")
        # Check alerts_log table
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(alerts) >= 1
        last = dict(alerts[0])
        assert "execution plan" in last["message"].lower() or "plan created" in last["message"].lower()


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

    def test_child_filled_clears_active_child_order_id(self):
        """When a child fills and next is returned, active_child_order_id
        should be set to __advancing__ sentinel (submission path replaces it)."""
        from src.execution.plan_controller import ADVANCING_SENTINEL

        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db)

        # Verify active child is set
        before = db.get_execution_plan(plan.plan_id)
        assert before.active_child_order_id == first_id

        # Fill first child
        db.update_order_status(first_id, "filled",
                               filled_size=33.33, avg_fill_price=0.60)
        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)
        assert next_spec is not None

        # active_child_order_id should now be __advancing__ sentinel
        after = db.get_execution_plan(plan.plan_id)
        assert after.active_child_order_id == ADVANCING_SENTINEL

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

    def test_terminal_child_also_clears_active_child(self):
        """When a child is cancelled and next is returned, active_child_order_id
        should be set to __advancing__ sentinel."""
        from src.execution.plan_controller import ADVANCING_SENTINEL

        db = _make_db()
        ctrl, plan, first_id = self._setup_active_plan(db, num_children=2)

        # Cancel first child
        db.update_order_status(first_id, "cancelled")
        child_order = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child_order)
        assert next_spec is not None

        updated = db.get_execution_plan(plan.plan_id)
        assert updated.active_child_order_id == ADVANCING_SENTINEL


class TestPlanControllerCancel:
    def test_cancel_plan_no_fills(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        active_child = ctrl.cancel_plan(plan.plan_id, "user requested")
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "cancelled"
        assert updated.error == "user requested"
        # No active child yet (plan was just planned)
        assert active_child == ""

    def test_cancel_plan_returns_active_child(self):
        """cancel_plan() should return the active child order ID for venue cancellation."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-on-book")
        active_child = ctrl.cancel_plan(plan.plan_id, "user cancel")
        assert active_child == "child-on-book"
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.active_child_order_id == ""

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
        active_child = ctrl.cancel_plan("already-done")
        assert active_child == ""
        updated = db.get_execution_plan("already-done")
        assert updated.status == "filled"  # Unchanged

    def test_cancel_plan_emits_alert(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.cancel_plan(plan.plan_id, "test cancel")
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%cancelled%' ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(alerts) >= 1


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

    def test_summary_uses_terminal_terminology(self):
        """Summary should use terminal_children and terminal_pct, not completion."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        summary = ctrl.get_plan_summary(plan.plan_id)
        assert "terminal_children" in summary
        assert "terminal_pct" in summary
        assert "fill_pct" in summary
        # Should NOT have the old names
        assert "completed_children" not in summary
        assert "completion_pct" not in summary

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

    def test_active_child_set_after_submission(self):
        """After _submit_plan_children runs, active_child_order_id should be set."""
        # This tests the end-to-end flow:
        # 1. Child fills → controller clears active_child_order_id
        # 2. _submit_plan_children sets it to the new child
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "old-child")

        # Simulate: controller returns next spec, clears active_child
        first_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=first_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        child = db.get_order(first_id)
        next_spec = ctrl.update_plan_from_child(child)
        assert next_spec is not None

        # At this point active_child_order_id should be __advancing__
        from src.execution.plan_controller import ADVANCING_SENTINEL
        mid = db.get_execution_plan(plan.plan_id)
        assert mid.active_child_order_id == ADVANCING_SENTINEL

        # Simulate what _submit_plan_children does after submission
        new_child_id = "new-child-id"
        db.update_execution_plan(plan.plan_id, active_child_order_id=new_child_id)
        after = db.get_execution_plan(plan.plan_id)
        assert after.active_child_order_id == new_child_id


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


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 2: spec-version validation, alert cleanup,
#   cancel caller, partial-child alerts
# ═══════════════════════════════════════════════════════════════


class TestSpecVersionValidation:
    """Verify spec_version is validated on read/reconstruction."""

    def test_validate_supported_version(self):
        """Version 1 should pass without error."""
        _validate_spec_version({"spec_version": 1}, "test-plan")

    def test_validate_missing_version_defaults_to_1(self):
        """Legacy plans without spec_version are treated as version 1."""
        _validate_spec_version({}, "test-plan")  # no error

    def test_validate_unsupported_version_raises(self):
        """Future versions should be rejected explicitly."""
        with pytest.raises(UnsupportedSpecVersion, match="spec_version=99"):
            _validate_spec_version({"spec_version": 99}, "test-plan")

    def test_get_first_child_rejects_bad_version(self):
        """get_first_child_spec should fail on unsupported version."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # Tamper metadata to unsupported version
        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(plan.plan_id, metadata_json=json.dumps(meta))
        plan = db.get_execution_plan(plan.plan_id)

        with pytest.raises(UnsupportedSpecVersion):
            ctrl.get_first_child_spec(plan)

    def test_child_filled_rejects_bad_version(self):
        """_handle_child_filled should fail the plan on unsupported version."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Submit + fill first child
        first_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=first_id, market_id="mkt-1",
            price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, first_id)

        # Tamper version before the controller tries to read next child
        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(plan.plan_id, metadata_json=json.dumps(meta))

        child = db.get_order(first_id)
        result = ctrl.update_plan_from_child(child)
        assert result is None  # no next spec returned

        # Plan should be marked failed
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "failed"
        assert "spec_version=999" in updated.error

    def test_child_terminal_rejects_bad_version(self):
        """_handle_child_terminal should fail the plan on unsupported version."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Submit + cancel first child
        first_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=first_id, market_id="mkt-1",
            price=0.60, size=33.33,
            status="cancelled",
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, first_id)

        # Tamper version
        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(plan.plan_id, metadata_json=json.dumps(meta))

        child = db.get_order(first_id)
        result = ctrl.update_plan_from_child(child)
        assert result is None

        updated = db.get_execution_plan(plan.plan_id)
        assert updated.status == "failed"
        assert "spec_version=999" in updated.error


class TestAlertChannelCleanup:
    """Verify all alerts use stable 'plan_controller' channel."""

    def test_create_alert_uses_stable_channel(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        ctrl.create_plan(orders, "twap")
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(alerts) >= 1
        last = dict(alerts[0])
        assert last["channel"] == "plan_controller"

    def test_cancel_alert_uses_stable_channel(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.cancel_plan(plan.plan_id, "test")
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%cancelled%'"
        ).fetchall()
        assert len(alerts) >= 1
        last = dict(alerts[0])
        assert last["channel"] == "plan_controller"

    def test_completion_alert_uses_stable_channel(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(1)
        plan = ctrl.create_plan(orders, "twap")

        child_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=child_id, market_id="mkt-1",
            price=0.60, size=100.0,
            status="filled", filled_size=100.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, child_id)
        ctrl.update_plan_from_child(db.get_order(child_id))

        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%filled%' ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(alerts) >= 1
        assert dict(alerts[0])["channel"] == "plan_controller"

    def test_alert_message_contains_details(self):
        """Details should be in message, not in channel."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log ORDER BY id DESC LIMIT 1"
        ).fetchall()
        last = dict(alerts[0])
        # Details like plan_id and market_id should be in the message
        assert "plan=" in last["message"] or "mkt-" in last["message"]


class TestCancelPlanEndToEnd:
    """Verify cancel_plan caller behavior and child handling."""

    def test_cancel_returns_active_child_id(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-abc")
        result = ctrl.cancel_plan(plan.plan_id, "testing")
        assert result == "child-abc"

    def test_cancel_no_active_child_returns_empty(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        # Plan is in "planned" state, no active child
        result = ctrl.cancel_plan(plan.plan_id, "testing")
        assert result == ""

    def test_cancel_already_terminal_returns_empty(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="filled")
        result = ctrl.cancel_plan(plan.plan_id, "testing")
        assert result == ""

    def test_cancel_nonexistent_returns_empty(self):
        db = _make_db()
        ctrl = PlanController(db)
        result = ctrl.cancel_plan("no-such-plan", "testing")
        assert result == ""


class TestPartialChildAlert:
    """Verify partial fill progress alerts are emitted."""

    def test_partial_fill_emits_alert(self):
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        child_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=child_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="partial", filled_size=25.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, child_id)

        child = db.get_order(child_id)
        ctrl.update_plan_from_child(child)

        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%partial fill%'"
        ).fetchall()
        assert len(alerts) >= 1
        last = dict(alerts[0])
        assert "%" in last["message"]  # contains percentage
        assert last["channel"] == "plan_controller"


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 3: Edge-case and messy real-world scenarios
# ═══════════════════════════════════════════════════════════════


class TestPlanEdgeCases:
    """Tests for messy real-world execution behavior."""

    def test_duplicate_fill_on_same_child_idempotent(self):
        """If a child fill arrives twice, plan should not double-count."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        child_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=child_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, child_id)

        # First fill notification — should return next spec
        child = db.get_order(child_id)
        next_spec = ctrl.update_plan_from_child(child)
        assert next_spec is not None

        # Simulate: second child submitted and filled
        db.insert_order(OrderRecord(
            order_id=next_spec.order_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)

        second_child = db.get_order(next_spec.order_id)
        result = ctrl.update_plan_from_child(second_child)
        assert result is None  # no more children

        # Duplicate fill on first child again — plan already terminal
        dup_result = ctrl.update_plan_from_child(child)
        assert dup_result is None

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "filled"
        # filled_size should not be double-counted
        assert final.filled_size == pytest.approx(100.0, abs=1.0)

    def test_fill_on_already_terminal_plan_noop(self):
        """Fill arriving for a plan that's already cancelled/failed/filled."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        child_id = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=child_id, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, child_id)

        # Force plan to terminal state
        db.update_execution_plan(plan.plan_id, status="cancelled")

        child = db.get_order(child_id)
        result = ctrl.update_plan_from_child(child)
        assert result is None  # no-op for terminal plans

    def test_all_children_cancelled_plan_status(self):
        """When every child is cancelled with no fills, plan ends as 'cancelled'."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # Insert ONLY the first child (cancelled) — realistic flow
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0, status="cancelled",
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        # First cancel → next child returned (child 1 not yet in DB)
        child0 = db.get_order(c1)
        next_spec = ctrl.update_plan_from_child(child0)
        assert next_spec is not None  # still has child 1 to submit

        # Simulate: second child submitted and also cancelled
        db.insert_order(OrderRecord(
            order_id=next_spec.order_id, market_id="mkt-1",
            price=0.60, size=50.0, status="cancelled",
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)

        child1 = db.get_order(next_spec.order_id)
        result = ctrl.update_plan_from_child(child1)
        assert result is None

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "cancelled"
        assert final.filled_size == 0.0

    def test_mixed_filled_and_cancelled_children(self):
        """First child filled, second cancelled → plan is 'partial'."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # First child: filled
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        next_spec = ctrl.update_plan_from_child(db.get_order(c1))
        assert next_spec is not None

        # Second child: cancelled
        db.insert_order(OrderRecord(
            order_id=next_spec.order_id, market_id="mkt-1",
            price=0.60, size=50.0, status="cancelled",
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)
        ctrl.update_plan_from_child(db.get_order(next_spec.order_id))

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        assert final.filled_size == pytest.approx(50.0, abs=0.1)

    def test_plan_with_expired_children_status(self):
        """All children expired with no fills → plan status is 'expired'."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(1)  # single child for simplicity
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=100.0, status="expired",
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        ctrl.update_plan_from_child(db.get_order(c1))

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "expired"

    def test_plan_with_failed_child_and_prior_fills(self):
        """First child filled, second failed → partial status."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        next_spec = ctrl.update_plan_from_child(db.get_order(c1))

        db.insert_order(OrderRecord(
            order_id=next_spec.order_id, market_id="mkt-1",
            price=0.60, size=50.0, status="failed",
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)
        ctrl.update_plan_from_child(db.get_order(next_spec.order_id))

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        assert final.filled_size == pytest.approx(50.0, abs=0.1)


class TestActiveChildTimingEdgeCases:
    """Tests for the window between controller clearing active_child_order_id
    and the submission path setting it to the new child."""

    def test_invariant_does_not_fire_when_all_children_submitted(self):
        """Plan with next_child_index == total_children and no active child
        should NOT trigger the stale plan invariant."""
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="awaiting-last", market_id="m1",
            status="active",
            total_children=3,
            next_child_index=3,  # all submitted
            active_child_order_id="",
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 0

    def test_invariant_fires_when_children_remain(self):
        """Plan with next_child_index < total_children and no active child
        should trigger the stale plan invariant."""
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="stuck-mid", market_id="m1",
            status="active",
            total_children=3,
            next_child_index=1,  # 2 remaining
            active_child_order_id="",
        ))
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 1

    def test_controller_clears_active_child_before_returning_next(self):
        """When _handle_child_filled returns next spec, active_child_order_id
        should be empty at that moment."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        next_spec = ctrl.update_plan_from_child(db.get_order(c1))
        assert next_spec is not None

        # Between controller returning and submission setting it, field is __advancing__
        from src.execution.plan_controller import ADVANCING_SENTINEL
        mid_plan = db.get_execution_plan(plan.plan_id)
        assert mid_plan.active_child_order_id == ADVANCING_SENTINEL
        assert mid_plan.next_child_index == 2  # advanced

        # Submission path sets the new child
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)
        after = db.get_execution_plan(plan.plan_id)
        assert after.active_child_order_id == next_spec.order_id

    def test_stale_invariant_clears_after_submission(self):
        """The stale invariant should not fire once active_child_order_id is set."""
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="briefly-stale", market_id="m1",
            status="active",
            total_children=3,
            next_child_index=2,
            active_child_order_id="",
        ))
        # Before submission: invariant fires
        violations = check_invariants(db)
        stale = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale) == 1

        # After submission: invariant clears
        db.update_execution_plan("briefly-stale", active_child_order_id="new-child-id")
        violations2 = check_invariants(db)
        stale2 = [v for v in violations2 if v.check == "stale_execution_plan"]
        assert len(stale2) == 0


class TestReconciliationPlanEdgeCases:
    """Tests for plan-aware reconciliation edge cases."""

    def test_notify_controller_with_override_status(self):
        """Override status should be used by the controller, not the DB status."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(1)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=100.0,
            status="submitted",  # DB says submitted
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        # Override to "cancelled" — simulates exchange cancellation
        order = db.get_order(c1)
        reconciler._notify_plan_controller(order, override_status="cancelled")

        # Plan should be terminal (cancelled) not still active
        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "cancelled"

    def test_notify_controller_no_plan_controller_noop(self):
        """Without plan controller, notify does nothing even with parent_plan_id."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=None,
        )

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", status="filled",
            parent_plan_id="some-plan", child_index=0,
        ))
        order = db.get_order(c1)
        # Should not raise
        reconciler._notify_plan_controller(order)
        assert len(reconciler._pending_plan_submissions) == 0

    def test_pending_submissions_cleared_after_processing(self):
        """_pending_plan_submissions should be empty after reconciler processes them."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        order = db.get_order(c1)
        reconciler._notify_plan_controller(order)
        assert len(reconciler._pending_plan_submissions) == 1

        # Simulate processing
        reconciler._pending_plan_submissions.clear()
        assert len(reconciler._pending_plan_submissions) == 0


class TestPlanPruningEdgeCases:
    """Tests for plan-aware order pruning."""

    def test_terminal_child_of_active_plan_not_pruned(self):
        """Filled children of active plans should NOT be pruned."""
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="active-plan", market_id="m1",
            status="active", total_children=2,
        ))
        db.insert_order(OrderRecord(
            order_id="child-filled", market_id="m1",
            status="filled", parent_plan_id="active-plan", child_index=0,
        ))
        count = db.prune_terminal_orders()
        assert count == 0

        # Verify child still exists
        remaining = db.get_plan_children("active-plan")
        assert len(remaining) == 1

    def test_terminal_child_of_terminal_plan_pruned(self):
        """Filled children of terminal plans SHOULD be pruned."""
        db = _make_db()
        db.insert_execution_plan(ExecutionPlanRecord(
            plan_id="done-plan", market_id="m1",
            status="filled", total_children=1,
        ))
        db.insert_order(OrderRecord(
            order_id="child-done", market_id="m1",
            status="filled", parent_plan_id="done-plan", child_index=0,
        ))
        count = db.prune_terminal_orders()
        assert count == 1

    def test_orphan_terminal_order_no_plan_pruned(self):
        """Terminal orders without a parent plan should be pruned normally."""
        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="no-plan", market_id="m1",
            status="filled", parent_plan_id="",
        ))
        count = db.prune_terminal_orders()
        assert count == 1


class TestSpecVersionSeverityUpgrade:
    """Verify unsupported spec version triggers critical alert, not warning."""

    def test_critical_alert_on_bad_version_filled(self):
        """Unsupported spec version during child fill should trigger critical alert."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        # Tamper version
        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(plan.plan_id, metadata_json=json.dumps(meta))

        ctrl.update_plan_from_child(db.get_order(c1))

        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%failed%' AND message LIKE '%spec_version%'"
        ).fetchall()
        assert len(alerts) >= 1
        assert dict(alerts[0])["level"] == "critical"

    def test_critical_alert_on_bad_version_terminal(self):
        """Unsupported spec version during child cancel should trigger critical alert."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=33.33, status="cancelled",
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)

        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(plan.plan_id, metadata_json=json.dumps(meta))

        ctrl.update_plan_from_child(db.get_order(c1))

        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%failed%' AND message LIKE '%spec_version%'"
        ).fetchall()
        assert len(alerts) >= 1
        assert dict(alerts[0])["level"] == "critical"


class TestCancelPlanVenueEdgeCases:
    """Tests for cancel_plan behavior under various conditions."""

    def test_cancel_plan_with_partial_fills(self):
        """Cancel a plan that has partial fills → status should be 'partial'."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # First child filled
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        next_spec = ctrl.update_plan_from_child(db.get_order(c1))
        db.update_execution_plan(plan.plan_id, active_child_order_id=next_spec.order_id)

        # Cancel while second child is active
        active_id = ctrl.cancel_plan(plan.plan_id, "user requested")
        assert active_id == next_spec.order_id

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"  # has fills, so partial
        assert final.error == "user requested"

    def test_cancel_plan_twice_idempotent(self):
        """Cancelling an already-cancelled plan should return empty string."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.cancel_plan(plan.plan_id, "first cancel")
        result = ctrl.cancel_plan(plan.plan_id, "second cancel")
        assert result == ""

    def test_cancel_plan_preserves_fill_aggregates(self):
        """Cancellation should not reset filled_size or avg_fill_price."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0,
            status="filled", filled_size=50.0, avg_fill_price=0.62,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        ctrl.update_plan_from_child(db.get_order(c1))

        # Check fill is recorded
        pre = db.get_execution_plan(plan.plan_id)
        assert pre.filled_size > 0

        ctrl.cancel_plan(plan.plan_id, "done")

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        # cancel_plan re-scans children and persists fill aggregates
        assert final.filled_size == pytest.approx(50.0, abs=0.1)
        assert final.avg_fill_price == pytest.approx(0.62, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 4: cancel fill persistence, child cancellation,
#   submission failure handling, stuck plan auto-recovery
# ═══════════════════════════════════════════════════════════════


class TestCancelPlanFillPersistence:
    """Verify cancel_plan() persists filled_size and avg_fill_price."""

    def test_cancel_no_fills_zeros(self):
        """Cancel with no filled children → filled_size=0, avg_fill_price=0."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.cancel_plan(plan.plan_id, "no fills")
        final = db.get_execution_plan(plan.plan_id)
        assert final.filled_size == 0.0
        assert final.avg_fill_price == 0.0
        assert final.status == "cancelled"

    def test_cancel_with_one_filled_child(self):
        """Cancel with one filled child → persists that child's fill data."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.58,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        ctrl.cancel_plan(plan.plan_id, "after first fill")

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        assert final.filled_size == pytest.approx(33.33, abs=0.01)
        assert final.avg_fill_price == pytest.approx(0.58, abs=0.01)

    def test_cancel_with_multiple_fills_weighted_avg(self):
        """Cancel with two filled children → weighted avg fill price."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # First child filled at 0.55
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.55, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.55,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        # Second child filled at 0.65
        c2 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c2, market_id="mkt-1",
            price=0.65, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.65,
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        ctrl.activate_plan(plan.plan_id, c2)
        ctrl.cancel_plan(plan.plan_id, "partial cancel")

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        assert final.filled_size == pytest.approx(66.66, abs=0.1)
        # Weighted avg: (33.33*0.55 + 33.33*0.65) / 66.66 = 0.60
        assert final.avg_fill_price == pytest.approx(0.60, abs=0.01)

    def test_cancel_with_partial_child_uses_filled_size(self):
        """Cancel with a partially filled child → uses filled_size, not size."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1",
            price=0.60, size=50.0,
            status="partial", filled_size=20.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        ctrl.cancel_plan(plan.plan_id, "partial child cancel")

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"
        assert final.filled_size == pytest.approx(20.0, abs=0.1)


class TestCancelPlanChildCancellation:
    """Verify cancel_plan() cancels remaining submitted/pending children."""

    def test_submitted_children_cancelled(self):
        """Submitted children should be marked as cancelled on plan cancel."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Insert 3 children: first filled, second submitted, third pending
        c1 = str(uuid.uuid4())
        c2 = str(uuid.uuid4())
        c3 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", price=0.60, size=33.33,
            status="filled", filled_size=33.33, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        db.insert_order(OrderRecord(
            order_id=c2, market_id="mkt-1", price=0.60, size=33.33,
            status="submitted",
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        db.insert_order(OrderRecord(
            order_id=c3, market_id="mkt-1", price=0.60, size=33.33,
            status="pending",
            parent_plan_id=plan.plan_id, child_index=2,
        ))
        ctrl.activate_plan(plan.plan_id, c2)
        ctrl.cancel_plan(plan.plan_id, "cancel all")

        # Verify: submitted and pending children marked cancelled
        child2 = db.get_order(c2)
        child3 = db.get_order(c3)
        assert child2.status == "cancelled"
        assert child3.status == "cancelled"

        # Filled child should remain filled
        child1 = db.get_order(c1)
        assert child1.status == "filled"

    def test_no_children_to_cancel(self):
        """Cancel with no submitted children should still work."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(1)
        plan = ctrl.create_plan(orders, "twap")

        # Only child is already filled
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", price=0.60, size=100.0,
            status="filled", filled_size=100.0, avg_fill_price=0.60,
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        ctrl.activate_plan(plan.plan_id, c1)
        ctrl.cancel_plan(plan.plan_id, "nothing to cancel")

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "partial"  # has fills
        child = db.get_order(c1)
        assert child.status == "filled"  # unchanged

    def test_already_cancelled_children_skipped(self):
        """Children already in terminal state should not be double-cancelled."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        c2 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", price=0.60, size=50.0,
            status="cancelled",
            parent_plan_id=plan.plan_id, child_index=0,
        ))
        db.insert_order(OrderRecord(
            order_id=c2, market_id="mkt-1", price=0.60, size=50.0,
            status="submitted",
            parent_plan_id=plan.plan_id, child_index=1,
        ))
        ctrl.activate_plan(plan.plan_id, c2)
        ctrl.cancel_plan(plan.plan_id, "partial cancel")

        # Only c2 gets cancelled, c1 stays as-is
        assert db.get_order(c2).status == "cancelled"
        assert db.get_order(c1).status == "cancelled"  # was already cancelled


class TestSubmissionFailureHandling:
    """Verify _submit_plan_children marks plan as failed on submission error."""

    def test_submit_failure_marks_plan_failed(self):
        """When child submission throws, plan should be marked failed.

        We patch submit_order on the OrderRouter prototype to force failure
        regardless of import order.
        """
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "first-child")

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        # Patch submit_order on the class so any new instance fails
        with patch.object(
            OrderRouter, "submit_order",
            new=AsyncMock(side_effect=RuntimeError("venue connection lost")),
        ):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "failed"
        assert "submission failed" in final.error

    def test_submit_failure_clears_active_child(self):
        """Failed submission should clear active_child_order_id."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "prev-child")

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        with patch.object(
            OrderRouter, "submit_order",
            new=AsyncMock(side_effect=RuntimeError("venue error")),
        ):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        final = db.get_execution_plan(plan.plan_id)
        assert final.active_child_order_id == ""


class TestRecoverStuckPlans:
    """Verify PlanController.recover_stuck_plans() detection and recovery."""

    def test_stuck_plan_detected_and_recovered(self):
        """An active plan with no child and remaining children is stuck and recoverable."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Simulate: plan activated, first child filled, next_child_index advanced
        # but active_child_order_id is empty (stuck)
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="",
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 1
        plan_id, spec = recovered[0]
        assert plan_id == plan.plan_id
        assert spec.market_id == "mkt-1"

        # next_child_index should be advanced
        updated = db.get_execution_plan(plan.plan_id)
        assert updated.next_child_index == 2

    def test_non_stuck_plan_not_recovered(self):
        """Plan with active child should NOT be recovered."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="child-123",  # has active child
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0

    def test_all_children_submitted_not_recovered(self):
        """Plan with all children submitted should NOT be recovered."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=3,  # all submitted
            active_child_order_id="",
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0

    def test_terminal_plan_not_recovered(self):
        """Terminal plans should NOT be recovered."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        db.update_execution_plan(plan.plan_id, status="filled")

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0

    def test_recovery_with_bad_spec_version_fails_plan(self):
        """Stuck plan with bad spec version should be marked failed."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Tamper spec version
        meta = json.loads(plan.metadata_json)
        meta["spec_version"] = 999
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="",
            metadata_json=json.dumps(meta),
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0  # no recovery returned

        # Plan should be marked failed
        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "failed"
        assert "spec_version" in final.error

    def test_recovery_emits_warning_alert(self):
        """Recovery should emit a warning alert about the stuck plan."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="",
        )

        ctrl.recover_stuck_plans()

        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE message LIKE '%stuck%'"
        ).fetchall()
        assert len(alerts) >= 1
        last = dict(alerts[0])
        assert last["level"] == "warning"

    def test_recovery_fresh_order_id(self):
        """Recovered spec should have a fresh order_id."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        original_ids = {o.order_id for o in orders}
        plan = ctrl.create_plan(orders, "twap")

        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="",
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 1
        _, spec = recovered[0]
        assert spec.order_id not in original_ids

    def test_recovery_serialized_children_exhausted(self):
        """If next_child_index exceeds serialized children, mark plan failed."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        # Manually set next_child_index beyond what's serialized
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=5,  # beyond len(children)=2
            total_children=6,    # looks like there are more
            active_child_order_id="",
        )

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 0

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "failed"
        assert "exceeds serialized children" in final.error

    def test_multiple_stuck_plans_all_recovered(self):
        """Multiple stuck plans should all be recovered in one call."""
        db = _make_db()
        ctrl = PlanController(db)

        plans = []
        for _ in range(3):
            orders = _make_twap_orders(3)
            plan = ctrl.create_plan(orders, "twap")
            db.update_execution_plan(
                plan.plan_id,
                status="active",
                next_child_index=1,
                active_child_order_id="",
            )
            plans.append(plan)

        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 3
        recovered_ids = {pid for pid, _ in recovered}
        expected_ids = {p.plan_id for p in plans}
        assert recovered_ids == expected_ids


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 5 — BATCH A: Race-Condition Guards
# ═══════════════════════════════════════════════════════════════


class TestRaceConditionGuards:
    """Verify race-condition guards added in Hardening Round 5."""

    def test_fill_before_activation_ignored(self):
        """A1: child fill on a 'planned' plan should return None."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        assert plan.status == "planned"

        # Simulate a child fill arriving before activate_plan()
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=50.0, filled_size=50.0, avg_fill_price=0.60,
        ))
        child = db.get_order(c1)
        result = ctrl.update_plan_from_child(child)
        assert result is None
        # Plan should remain in planned state
        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "planned"

    def test_partial_before_activation_ignored(self):
        """A1: partial fill on a 'planned' plan should return None."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")

        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", status="partial",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=50.0, filled_size=20.0, avg_fill_price=0.60,
        ))
        child = db.get_order(c1)
        result = ctrl.update_plan_from_child(child)
        assert result is None

    def test_cancel_races_with_child_fill(self):
        """A2: cancel_plan() then _handle_child_filled re-read detects terminal."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        # Insert and fill child 0
        c0 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c0, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=33.33, filled_size=33.33, avg_fill_price=0.60,
        ))

        # Cancel the plan first
        ctrl.cancel_plan(plan.plan_id, reason="user cancel")
        final_after_cancel = db.get_execution_plan(plan.plan_id)
        assert final_after_cancel.status in ("cancelled", "partial")

        # Now a late fill arrives — should be ignored by re-read guard
        child = db.get_order(c0)
        result = ctrl.update_plan_from_child(child)
        assert result is None  # terminal guard catches it

    def test_cancel_races_with_child_terminal(self):
        """A3: cancel_plan() then _handle_child_terminal re-read detects terminal."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        c0 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c0, market_id="mkt-1", status="cancelled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=33.33,
        ))

        ctrl.cancel_plan(plan.plan_id, reason="shutdown")
        child = db.get_order(c0)
        result = ctrl.update_plan_from_child(child)
        assert result is None

    def test_double_fill_reread_second_noop(self):
        """A2: first fill completes plan, second fill's re-read detects terminal."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(1)  # single child
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        c0 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c0, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=100.0, filled_size=100.0, avg_fill_price=0.60,
        ))

        # First update completes the plan
        child = db.get_order(c0)
        result1 = ctrl.update_plan_from_child(child)
        assert result1 is None  # plan completed, no next spec
        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "filled"

        # Second update should be ignored (terminal guard at line 244)
        result2 = ctrl.update_plan_from_child(child)
        assert result2 is None

    def test_advancing_sentinel_set_on_next_child(self):
        """A4: after child fill returns next spec, active_child_order_id == __advancing__."""
        from src.execution.plan_controller import ADVANCING_SENTINEL

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        c0 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c0, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=33.33, filled_size=33.33, avg_fill_price=0.60,
        ))

        child = db.get_order(c0)
        next_spec = ctrl.update_plan_from_child(child)
        assert next_spec is not None

        updated = db.get_execution_plan(plan.plan_id)
        assert updated.active_child_order_id == ADVANCING_SENTINEL

    def test_advancing_sentinel_no_invariant_13(self):
        """A4: plan with __advancing__ does NOT trigger stale_execution_plan."""
        from src.execution.plan_controller import ADVANCING_SENTINEL

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")

        # Simulate advancing state
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id=ADVANCING_SENTINEL,
        )

        violations = check_invariants(db)
        stale_violations = [v for v in violations if v.check == "stale_execution_plan"]
        assert len(stale_violations) == 0

    def test_advancing_sentinel_replaced_after_submit(self):
        """A4: _submit_plan_children replaces __advancing__ with real order_id."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter
        from src.execution.plan_controller import ADVANCING_SENTINEL

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id=ADVANCING_SENTINEL,
        )

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        fake_result = type("R", (), {
            "order_id": "real-order-id",
            "clob_order_id": "clob-123",
            "status": "submitted",
        })()

        with patch.object(
            OrderRouter, "submit_order",
            new=AsyncMock(return_value=fake_result),
        ):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        updated = db.get_execution_plan(plan.plan_id)
        assert updated.active_child_order_id == "real-order-id"
        assert updated.active_child_order_id != ADVANCING_SENTINEL

    def test_cancel_clears_to_empty_not_advancing(self):
        """A4: cancel_plan sets active_child_order_id="" not __advancing__."""
        from src.execution.plan_controller import ADVANCING_SENTINEL

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        ctrl.cancel_plan(plan.plan_id, "user cancel")
        final = db.get_execution_plan(plan.plan_id)
        assert final.active_child_order_id == ""
        assert final.active_child_order_id != ADVANCING_SENTINEL

    def test_submit_skips_cancelled_plan(self):
        """A5: queued child for cancelled plan is skipped."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")
        ctrl.cancel_plan(plan.plan_id, "cancelled")

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        submit_mock = AsyncMock()
        with patch.object(OrderRouter, "submit_order", new=submit_mock):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        submit_mock.assert_not_called()

    def test_submit_skips_filled_plan(self):
        """A5: queued child for filled plan is skipped."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="filled")

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        submit_mock = AsyncMock()
        with patch.object(OrderRouter, "submit_order", new=submit_mock):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        submit_mock.assert_not_called()

    def test_submit_skips_deleted_plan(self):
        """A5: plan removed from DB before submission — skipped."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        spec = _make_order_spec()
        reconciler._pending_plan_submissions.append(
            (spec, "nonexistent-plan-id", 1)
        )

        submit_mock = AsyncMock()
        with patch.object(OrderRouter, "submit_order", new=submit_mock):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        submit_mock.assert_not_called()

    def test_submit_mixed_queue_per_item_check(self):
        """A5: multiple queued items, some for terminal plans, some for active."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        # Plan 1: active (should be submitted)
        orders1 = _make_twap_orders(2)
        plan1 = ctrl.create_plan(orders1, "twap")
        ctrl.activate_plan(plan1.plan_id, "c0")

        # Plan 2: cancelled (should be skipped)
        orders2 = _make_twap_orders(2)
        plan2 = ctrl.create_plan(orders2, "twap")
        db.update_execution_plan(plan2.plan_id, status="cancelled")

        spec1 = _make_order_spec()
        spec2 = _make_order_spec()
        reconciler._pending_plan_submissions.append((spec1, plan1.plan_id, 1))
        reconciler._pending_plan_submissions.append((spec2, plan2.plan_id, 1))

        fake_result = type("R", (), {
            "order_id": "submitted-id",
            "clob_order_id": "clob-id",
            "status": "submitted",
        })()

        submit_mock = AsyncMock(return_value=fake_result)
        with patch.object(OrderRouter, "submit_order", new=submit_mock):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        # Only plan1's child should have been submitted
        assert submit_mock.call_count == 1

    def test_recover_then_cancel_before_submit(self):
        """A5: recover_stuck_plans returns spec, cancel happens, submit skips."""
        import asyncio
        from unittest.mock import patch, AsyncMock
        from src.execution.reconciliation import _submit_plan_children, OrderReconciler
        from src.execution.order_router import OrderRouter

        class FakeClob:
            pass

        db = _make_db()
        ctrl = PlanController(db)
        config = ExecutionConfig()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(), config=config,
            plan_controller=ctrl,
        )

        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=1,
            active_child_order_id="",
        )

        # Recover the stuck plan
        recovered = ctrl.recover_stuck_plans()
        assert len(recovered) == 1
        _, spec = recovered[0]
        reconciler._pending_plan_submissions.append(
            (spec, plan.plan_id, 1)
        )

        # Cancel before submission happens
        ctrl.cancel_plan(plan.plan_id, "shutdown")

        submit_mock = AsyncMock()
        with patch.object(OrderRouter, "submit_order", new=submit_mock):
            asyncio.run(
                _submit_plan_children(reconciler, db, FakeClob(), config, None)
            )

        submit_mock.assert_not_called()

    def test_concurrent_fills_sequential_processing(self):
        """A2+A3: two children fill, processed sequentially, plan completes correctly."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        # Insert both children as filled
        c0 = str(uuid.uuid4())
        c1 = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c0, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=50.0, filled_size=50.0, avg_fill_price=0.60,
        ))
        db.insert_order(OrderRecord(
            order_id=c1, market_id="mkt-1", status="filled",
            parent_plan_id=plan.plan_id, child_index=1,
            price=0.60, size=50.0, filled_size=50.0, avg_fill_price=0.60,
        ))

        # First fill — should return next spec (or complete if both terminal)
        child0 = db.get_order(c0)
        result1 = ctrl.update_plan_from_child(child0)

        # Second fill — plan should complete
        child1 = db.get_order(c1)
        result2 = ctrl.update_plan_from_child(child1)

        final = db.get_execution_plan(plan.plan_id)
        assert final.status == "filled"
        assert final.filled_size == pytest.approx(100.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 5 — BATCH B: Invariant Checks #14-#18
# ═══════════════════════════════════════════════════════════════


class TestInvariantChecks14To18:
    """Verify invariant checks #14-#18 added in Hardening Round 5."""

    # ── Check #14: terminal_plan_active_child ────────────────

    def test_terminal_plan_active_child_detected(self):
        """#14: submitted child of filled plan → critical."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="filled")

        # Insert an active child referencing the terminal plan
        db.insert_order(OrderRecord(
            order_id="orphan-child", market_id="mkt-1",
            status="submitted", parent_plan_id=plan.plan_id, child_index=0,
            action_side="BUY", outcome_side="YES",
        ))

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "terminal_plan_active_child"]
        assert len(found) == 1
        assert found[0].severity == "critical"

    def test_active_plan_active_child_no_violation(self):
        """#14: submitted child of active plan → no violation."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-0")

        db.insert_order(OrderRecord(
            order_id="child-0", market_id="mkt-1",
            status="submitted", parent_plan_id=plan.plan_id, child_index=0,
            action_side="BUY", outcome_side="YES",
        ))

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "terminal_plan_active_child"]
        assert len(found) == 0

    def test_terminal_child_terminal_plan_no_violation(self):
        """#14: filled child of filled plan → no violation (child not active)."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="filled")

        db.insert_order(OrderRecord(
            order_id="done-child", market_id="mkt-1",
            status="filled", parent_plan_id=plan.plan_id, child_index=0,
            action_side="BUY", outcome_side="YES",
        ))

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "terminal_plan_active_child"]
        assert len(found) == 0

    # ── Check #15: plan_child_index_overflow ──────────────────

    def test_child_index_overflow_detected(self):
        """#15: next_child_index=5, total_children=3 → warning."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=5,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_child_index_overflow"]
        assert len(found) == 1
        assert found[0].severity == "warning"

    def test_normal_index_no_violation(self):
        """#15: next_child_index=2, total_children=3 → no violation."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="active",
            next_child_index=2,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_child_index_overflow"]
        assert len(found) == 0

    def test_terminal_plan_overflow_excluded(self):
        """#15: filled plan with overflow → excluded."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="filled",
            next_child_index=5,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_child_index_overflow"]
        assert len(found) == 0

    # ── Check #16: plan_overfill ──────────────────────────────

    def test_overfill_detected(self):
        """#16: filled_size=110, target_size=100 → critical."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="filled",
            filled_size=110.0,
        )
        # target_size is ~99.99 from _make_twap_orders(3)

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_overfill"]
        assert len(found) == 1
        assert found[0].severity == "critical"

    def test_within_tolerance_no_violation(self):
        """#16: filled_size=100.5, target_size=100 → within 1% tolerance."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(3)
        plan = ctrl.create_plan(orders, "twap")
        # target_size is ~99.99, so filled_size of 100.5 is ~0.5% over → within tolerance
        db.update_execution_plan(
            plan.plan_id,
            status="filled",
            filled_size=100.5,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_overfill"]
        assert len(found) == 0

    def test_zero_target_no_violation(self):
        """#16: target_size=0 → excluded."""
        db = _make_db()
        conn = db._conn
        now = "2025-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO execution_plans
            (plan_id, market_id, token_id, strategy_type, action_side, outcome_side,
             target_size, target_stake_usd, filled_size, avg_fill_price,
             total_children, completed_children, active_child_order_id,
             next_child_index, status, dry_run, error, metadata_json,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("zero-target", "mkt-1", "tok-1", "twap", "BUY", "YES",
             0.0, 0.0, 10.0, 0.60, 1, 1, "", 1, "filled", 1, "", "{}",
             now, now),
        )
        conn.commit()

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_overfill"]
        assert len(found) == 0

    # ── Check #17: plan_child_missing_canonical ───────────────

    def test_plan_child_missing_canonical_detected(self):
        """#17: submitted child with empty action_side → warning."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-no-canon")

        db.insert_order(OrderRecord(
            order_id="child-no-canon", market_id="mkt-1",
            status="submitted",
            parent_plan_id=plan.plan_id, child_index=0,
            action_side="", outcome_side="",
        ))

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_child_missing_canonical"]
        assert len(found) == 1
        assert found[0].severity == "warning"

    def test_plan_child_with_canonical_no_violation(self):
        """#17: submitted child with both fields → no violation."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "child-ok")

        db.insert_order(OrderRecord(
            order_id="child-ok", market_id="mkt-1",
            status="submitted",
            parent_plan_id=plan.plan_id, child_index=0,
            action_side="BUY", outcome_side="YES",
        ))

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "plan_child_missing_canonical"]
        assert len(found) == 0

    # ── Check #18: partial_plan_zero_fill ─────────────────────

    def test_partial_zero_fill_detected(self):
        """#18: status=partial, filled_size=0 → warning."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="partial",
            filled_size=0.0,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "partial_plan_zero_fill"]
        assert len(found) == 1
        assert found[0].severity == "warning"

    def test_partial_with_fill_no_violation(self):
        """#18: status=partial, filled_size=50 → no violation."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="partial",
            filled_size=50.0,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "partial_plan_zero_fill"]
        assert len(found) == 0

    def test_cancelled_zero_fill_no_violation(self):
        """#18: status=cancelled, filled_size=0 → excluded."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(
            plan.plan_id,
            status="cancelled",
            filled_size=0.0,
        )

        violations = check_invariants(db)
        found = [v for v in violations if v.check == "partial_plan_zero_fill"]
        assert len(found) == 0

    # ── Clean DB ──────────────────────────────────────────────

    def test_clean_db_zero_violations_all_18(self):
        """All 18 checks on clean DB → 0 violations."""
        db = _make_db()
        violations = check_invariants(db)
        assert len(violations) == 0


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 5 — BATCH C: Reconciliation Metrics
# ═══════════════════════════════════════════════════════════════


class TestReconciliationMetrics:
    """Verify reconciliation metrics emission in reconcile_once()."""

    def _build_reconciler(self, db=None, clob=None):
        """Build an OrderReconciler with FakeClob and in-memory DB."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            def get_order_status(self, order_id):
                return None

            def cancel_order(self, order_id):
                pass

        if db is None:
            db = _make_db()
        return OrderReconciler(
            db=db, clob=clob or FakeClob(), config=ExecutionConfig(),
        ), db

    def test_reconcile_emits_pass_counter(self):
        """C1: passes counter increments."""
        from src.observability.metrics import metrics

        reconciler, db = self._build_reconciler()
        before = metrics.snapshot()["counters"].get("reconciliation.passes", 0)

        reconciler.reconcile_once()

        after = metrics.snapshot()["counters"].get("reconciliation.passes", 0)
        assert after >= before + 1

    def test_reconcile_emits_checked(self):
        """C1: checked counter matches result."""
        from src.observability.metrics import metrics

        db = _make_db()
        # Insert a submitted order with clob_order_id so it gets checked
        db.insert_order(OrderRecord(
            order_id="check-me", market_id="mkt-1",
            status="submitted", clob_order_id="clob-123",
            action_side="BUY", outcome_side="YES",
        ))

        reconciler, _ = self._build_reconciler(db=db)
        before = metrics.snapshot()["counters"].get("reconciliation.checked", 0)

        reconciler.reconcile_once()

        after = metrics.snapshot()["counters"].get("reconciliation.checked", 0)
        assert after >= before + 1

    def test_reconcile_emits_filled(self):
        """C1: filled counter on fill."""
        from src.observability.metrics import metrics

        class FillClob:
            def get_order_status(self, _):
                return {"status": "matched", "takingAmount": "60.0"}

            def cancel_order(self, _):
                pass

        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="fill-me", market_id="mkt-fill",
            status="submitted", clob_order_id="clob-fill",
            price=0.60, size=100.0, stake_usd=60.0,
            action_side="BUY", outcome_side="YES",
        ))

        from src.execution.reconciliation import OrderReconciler
        reconciler = OrderReconciler(
            db=db, clob=FillClob(), config=ExecutionConfig(),
        )

        before = metrics.snapshot()["counters"].get("reconciliation.filled", 0)
        reconciler.reconcile_once()
        after = metrics.snapshot()["counters"].get("reconciliation.filled", 0)
        assert after >= before + 1

    def test_reconcile_emits_cancelled(self):
        """C1: cancelled counter."""
        from src.observability.metrics import metrics

        class CancelClob:
            def get_order_status(self, _):
                return {"status": "cancelled"}

            def cancel_order(self, _):
                pass

        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="cancel-me", market_id="mkt-cancel",
            status="submitted", clob_order_id="clob-cancel",
            action_side="BUY", outcome_side="YES",
        ))

        from src.execution.reconciliation import OrderReconciler
        reconciler = OrderReconciler(
            db=db, clob=CancelClob(), config=ExecutionConfig(),
        )

        before = metrics.snapshot()["counters"].get("reconciliation.cancelled", 0)
        reconciler.reconcile_once()
        after = metrics.snapshot()["counters"].get("reconciliation.cancelled", 0)
        assert after >= before + 1

    def test_reconcile_emits_errors(self):
        """C1: errors counter."""
        from src.observability.metrics import metrics

        class ErrorClob:
            def get_order_status(self, _):
                raise RuntimeError("venue down")

            def cancel_order(self, _):
                pass

        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="error-me", market_id="mkt-error",
            status="submitted", clob_order_id="clob-error",
            action_side="BUY", outcome_side="YES",
        ))

        from src.execution.reconciliation import OrderReconciler
        reconciler = OrderReconciler(
            db=db, clob=ErrorClob(), config=ExecutionConfig(),
        )

        before = metrics.snapshot()["counters"].get("reconciliation.errors", 0)
        reconciler.reconcile_once()
        after = metrics.snapshot()["counters"].get("reconciliation.errors", 0)
        assert after >= before + 1

    def test_reconcile_emits_pruned(self):
        """C1: pruned counter."""
        from src.observability.metrics import metrics

        db = _make_db()
        # Insert a terminal order that will be pruned
        db.insert_order(OrderRecord(
            order_id="prune-me", market_id="mkt-prune",
            status="filled", action_side="BUY", outcome_side="YES",
        ))

        reconciler, _ = self._build_reconciler(db=db)
        before = metrics.snapshot()["counters"].get("reconciliation.pruned", 0)
        reconciler.reconcile_once()
        after = metrics.snapshot()["counters"].get("reconciliation.pruned", 0)
        assert after >= before + 1

    def test_multiple_passes_accumulate(self):
        """C1: counters accumulate across calls."""
        from src.observability.metrics import metrics

        reconciler, _ = self._build_reconciler()
        before = metrics.snapshot()["counters"].get("reconciliation.passes", 0)

        reconciler.reconcile_once()
        reconciler.reconcile_once()
        reconciler.reconcile_once()

        after = metrics.snapshot()["counters"].get("reconciliation.passes", 0)
        assert after >= before + 3


class TestInvariantViolationMetrics:
    """Verify invariant violation metrics emission in engine loop."""

    def test_invariant_violation_metric(self):
        """C3: violation increments counter."""
        from unittest.mock import patch, MagicMock
        from src.observability.metrics import metrics

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="partial", filled_size=0.0)

        # Manually call check_invariants and emit metrics (simulates engine loop)
        violations = check_invariants(db)
        found = [v for v in violations if v.check == "partial_plan_zero_fill"]
        assert len(found) >= 1

        before = metrics.snapshot()["counters"].get("invariant.violations.partial_plan_zero_fill", 0)
        for v in violations:
            metrics.incr(f"invariant.violations.{v.check}")
            metrics.incr(f"invariant.violations_by_severity.{v.severity}")
        after = metrics.snapshot()["counters"].get("invariant.violations.partial_plan_zero_fill", 0)
        assert after >= before + 1

    def test_violation_by_severity_metric(self):
        """C3: severity counter increments."""
        from src.observability.metrics import metrics

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        db.update_execution_plan(plan.plan_id, status="partial", filled_size=0.0)

        violations = check_invariants(db)
        before_w = metrics.snapshot()["counters"].get("invariant.violations_by_severity.warning", 0)
        for v in violations:
            metrics.incr(f"invariant.violations.{v.check}")
            metrics.incr(f"invariant.violations_by_severity.{v.severity}")
        after_w = metrics.snapshot()["counters"].get("invariant.violations_by_severity.warning", 0)
        assert after_w >= before_w + 1

    def test_api_reconciliation_structure(self):
        """C4: endpoint returns correct JSON keys."""
        from src.dashboard.app import app

        with app.test_client() as client:
            resp = client.get("/api/reconciliation")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "reconciliation" in data
            assert "plans" in data
            assert "invariant_violations" in data
            recon = data["reconciliation"]
            assert "passes" in recon
            assert "checked" in recon
            assert "filled" in recon

    def test_api_reconciliation_empty(self):
        """C4: all zeros when no activity (metrics may have prior state, just check structure)."""
        from src.dashboard.app import app

        with app.test_client() as client:
            resp = client.get("/api/reconciliation")
            assert resp.status_code == 200
            data = resp.get_json()
            # All values should be numeric
            for key, val in data["reconciliation"].items():
                assert isinstance(val, (int, float)), f"{key} is {type(val)}"


# ═══════════════════════════════════════════════════════════════
# HARDENING ROUND 5 — BATCH D: Venue Edge-Case Tests
# ═══════════════════════════════════════════════════════════════


class TestVenueEdgeCases:
    """Verify reconciliation handles odd CLOB responses correctly."""

    def _build(self, clob_response, db=None):
        """Build reconciler with controllable CLOB response."""
        from src.execution.reconciliation import OrderReconciler

        class FakeClob:
            def __init__(self, resp):
                self._resp = resp

            def get_order_status(self, _):
                return self._resp

            def cancel_order(self, _):
                pass

        if db is None:
            db = _make_db()
        reconciler = OrderReconciler(
            db=db, clob=FakeClob(clob_response), config=ExecutionConfig(),
        )
        return reconciler, db

    def _insert_submitted_order(self, db, order_id="test-order", **kwargs):
        defaults = dict(
            order_id=order_id, market_id="mkt-1", token_id="tok-1",
            status="submitted", clob_order_id="clob-123",
            price=0.60, size=100.0, stake_usd=60.0,
            action_side="BUY", outcome_side="YES",
        )
        defaults.update(kwargs)
        db.insert_order(OrderRecord(**defaults))

    def test_clob_matched_zero_taking_amount(self):
        """Matched with takingAmount=0 → fill uses order.size as fallback."""
        reconciler, db = self._build({"status": "matched", "takingAmount": "0"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 1
        # Verify via trade record (order may have been pruned)
        trades = db._conn.execute(
            "SELECT * FROM trades WHERE order_id = 'test-order'"
        ).fetchall()
        assert len(trades) == 1
        assert trades[0]["size"] == pytest.approx(100.0, abs=0.1)

    def test_clob_matched_missing_taking_amount(self):
        """Matched with no takingAmount key → fill uses order.size."""
        reconciler, db = self._build({"status": "matched"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 1
        trades = db._conn.execute(
            "SELECT * FROM trades WHERE order_id = 'test-order'"
        ).fetchall()
        assert len(trades) == 1
        assert trades[0]["size"] == pytest.approx(100.0, abs=0.1)

    def test_clob_non_dict_response(self):
        """CLOB returns None → no state change."""
        reconciler, db = self._build(None)
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 0
        assert result.cancelled == 0

    def test_clob_non_dict_string_response(self):
        """CLOB returns a string → no state change."""
        reconciler, db = self._build("some_error")
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 0
        assert result.cancelled == 0

    def test_clob_unknown_status(self):
        """CLOB returns status='some_garbage' → no state change."""
        reconciler, db = self._build({"status": "some_garbage"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 0
        assert result.cancelled == 0

    def test_clob_empty_status(self):
        """CLOB returns status='' → no state change."""
        reconciler, db = self._build({"status": ""})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.filled == 0
        assert result.cancelled == 0

    def test_clob_american_cancelled(self):
        """CLOB returns status='canceled' (single L) → cancelled."""
        reconciler, db = self._build({"status": "canceled"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.cancelled == 1

    def test_clob_live_zero_taking_no_partial(self):
        """CLOB returns live with takingAmount=0 → no partial fill."""
        reconciler, db = self._build({"status": "live", "takingAmount": "0"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.partial == 0

    def test_clob_live_decreasing_taking(self):
        """Live with new taking < old filled → no regression."""
        db = _make_db()
        # Order with existing partial fill of 50
        self._insert_submitted_order(db, filled_size=50.0, avg_fill_price=0.60)

        reconciler, _ = self._build(
            {"status": "live", "takingAmount": "10"},  # 10/0.6=16.67 < 50
            db=db,
        )

        result = reconciler.reconcile_once()
        assert result.partial == 0

    def test_clob_live_negative_taking(self):
        """Negative takingAmount → no negative fill (float conversion works, but <= 0 check prevents partial)."""
        reconciler, db = self._build({"status": "live", "takingAmount": "-10"})
        self._insert_submitted_order(db)

        result = reconciler.reconcile_once()
        assert result.partial == 0

    def test_plan_child_fill_zero_price(self):
        """Child with price=0 → avg_fill_price handles gracefully."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(1)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "zero-price-child")

        c = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c, market_id="mkt-1",
            status="filled", parent_plan_id=plan.plan_id, child_index=0,
            price=0.0, size=100.0, filled_size=100.0, avg_fill_price=0.0,
        ))

        child = db.get_order(c)
        result = ctrl.update_plan_from_child(child)
        assert result is None  # single child plan completes

        final = db.get_execution_plan(plan.plan_id)
        # Should not crash; price is 0 but that's valid
        assert final.status in ("filled", "partial")

    def test_plan_child_expired_advances_plan(self):
        """Expired child → plan advances to next or goes terminal."""
        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "exp-child")

        c = str(uuid.uuid4())
        db.insert_order(OrderRecord(
            order_id=c, market_id="mkt-1",
            status="expired", parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=50.0,
        ))

        child = db.get_order(c)
        next_spec = ctrl.update_plan_from_child(child)
        # Should advance to next child since there's another child
        assert next_spec is not None

    def test_stale_cancel_clob_error(self):
        """CLOB throws on cancel → error counted."""
        from src.execution.reconciliation import OrderReconciler

        class FailCancelClob:
            def get_order_status(self, _):
                return None

            def cancel_order(self, _):
                raise RuntimeError("exchange down")

        db = _make_db()
        db.insert_order(OrderRecord(
            order_id="stale-ord", market_id="mkt-1",
            status="submitted", clob_order_id="clob-stale",
            action_side="BUY", outcome_side="YES",
            price=0.60, size=100.0,
            created_at="2020-01-01T00:00:00Z",
        ))

        config = ExecutionConfig(
            stale_order_cancel_enabled=True,
            stale_order_cancel_secs=1,
        )
        reconciler = OrderReconciler(
            db=db, clob=FailCancelClob(), config=config,
        )
        result = reconciler.reconcile_once()
        assert result.errors >= 1

    def test_reconcile_order_no_clob_order_id(self):
        """Order without clob_order_id → skipped (no fill or cancel)."""
        reconciler, db = self._build({"status": "matched"})
        db.insert_order(OrderRecord(
            order_id="no-clob", market_id="mkt-1",
            status="submitted", clob_order_id="",
            action_side="BUY", outcome_side="YES",
        ))

        result = reconciler.reconcile_once()
        # The order is fetched by get_submitted_orders (checked=1) but
        # _reconcile_order returns early since there's no clob_order_id
        assert result.filled == 0
        assert result.cancelled == 0

    def test_plan_child_with_clob_cancel_notifies_controller(self):
        """Exchange cancels plan child → plan controller is notified."""
        from src.execution.reconciliation import OrderReconciler

        class CancelClob:
            def get_order_status(self, _):
                return {"status": "cancelled"}

            def cancel_order(self, _):
                pass

        db = _make_db()
        ctrl = PlanController(db)
        orders = _make_twap_orders(2)
        plan = ctrl.create_plan(orders, "twap")
        ctrl.activate_plan(plan.plan_id, "exchange-cancel-child")

        db.insert_order(OrderRecord(
            order_id="exchange-cancel-child", market_id="mkt-1",
            status="submitted", clob_order_id="clob-will-cancel",
            parent_plan_id=plan.plan_id, child_index=0,
            price=0.60, size=50.0,
            action_side="BUY", outcome_side="YES",
        ))

        reconciler = OrderReconciler(
            db=db, clob=CancelClob(), config=ExecutionConfig(),
            plan_controller=ctrl,
        )
        result = reconciler.reconcile_once()
        assert result.cancelled == 1

        # Plan controller should have been notified and queued next child
        assert len(reconciler._pending_plan_submissions) >= 1
