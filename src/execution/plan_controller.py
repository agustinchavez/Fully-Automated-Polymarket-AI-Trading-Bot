"""Execution plan controller — orchestrates TWAP/iceberg parent lifecycle.

Phase 10E: Manages persistent parent execution plans with sequential child
submission.  Simple orders (single child) skip this layer entirely.

Lifecycle:
  create_plan()        → status=planned, children serialized in metadata_json
  activate_plan()      → status=active, first child submitted
  update_plan_from_child() → child fill/cancel → update parent, return next child
  cancel_plan()        → mark plan + remaining children as cancelled
"""

from __future__ import annotations

import json
import uuid
import datetime as dt
from typing import Any

from src.config import ExecutionConfig
from src.execution.order_builder import OrderSpec
from src.storage.models import ExecutionPlanRecord, OrderRecord
from src.observability.logger import get_logger

log = get_logger(__name__)

# Incremented when the serialized OrderSpec schema changes.
SPEC_VERSION = 1
_SUPPORTED_SPEC_VERSIONS = {1}


class UnsupportedSpecVersion(Exception):
    """Raised when plan metadata has an unrecognized spec_version."""


def _serialize_order_spec(spec: OrderSpec) -> dict[str, Any]:
    """Serialize an OrderSpec to a JSON-safe dict (exclude non-serializable metadata)."""
    return {
        "order_id": spec.order_id,
        "market_id": spec.market_id,
        "token_id": spec.token_id,
        "side": spec.side,
        "order_type": spec.order_type,
        "price": spec.price,
        "size": spec.size,
        "stake_usd": spec.stake_usd,
        "ttl_secs": spec.ttl_secs,
        "dry_run": spec.dry_run,
        "action_side": spec.action_side,
        "outcome_side": spec.outcome_side,
        "execution_strategy": spec.execution_strategy,
        "parent_order_id": spec.parent_order_id,
        "child_index": spec.child_index,
        "total_children": spec.total_children,
    }


def _deserialize_order_spec(d: dict[str, Any]) -> OrderSpec:
    """Reconstruct an OrderSpec from a serialized dict."""
    return OrderSpec(
        order_id=d.get("order_id", str(uuid.uuid4())),
        market_id=d.get("market_id", ""),
        token_id=d.get("token_id", ""),
        side=d.get("side", "BUY"),
        order_type=d.get("order_type", "limit"),
        price=d.get("price", 0.0),
        size=d.get("size", 0.0),
        stake_usd=d.get("stake_usd", 0.0),
        ttl_secs=d.get("ttl_secs", 0),
        dry_run=d.get("dry_run", True),
        action_side=d.get("action_side", ""),
        outcome_side=d.get("outcome_side", ""),
        execution_strategy=d.get("execution_strategy", "simple"),
        parent_order_id=d.get("parent_order_id", ""),
        child_index=d.get("child_index", 0),
        total_children=d.get("total_children", 1),
    )


def _validate_spec_version(meta: dict[str, Any], plan_id: str = "") -> None:
    """Raise UnsupportedSpecVersion if the metadata version is unrecognized.

    Plans serialized without a spec_version (legacy, pre-versioning) are
    treated as version 1.
    """
    version = meta.get("spec_version", 1)
    if version not in _SUPPORTED_SPEC_VERSIONS:
        raise UnsupportedSpecVersion(
            f"Plan {plan_id[:8] if plan_id else '?'} has spec_version={version}, "
            f"supported={_SUPPORTED_SPEC_VERSIONS}"
        )


class PlanController:
    """Orchestrate parent execution plans for multi-child strategies."""

    def __init__(self, db: object, config: ExecutionConfig | None = None):
        self._db = db
        self._config = config or ExecutionConfig()

    # ── Helpers ──────────────────────────────────────────────────

    def _alert(self, level: str, message: str, channel: str = "plan_controller") -> None:
        """Insert a DB alert if the database supports it."""
        try:
            self._db.insert_alert(level, message, channel)
        except Exception:
            pass  # alerts are best-effort

    # ── Lifecycle ────────────────────────────────────────────────

    def create_plan(
        self,
        orders: list[OrderSpec],
        strategy_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionPlanRecord:
        """Create a new execution plan from a list of child OrderSpecs.

        The child specs are serialized into metadata_json for deferred
        reconstruction — no need to re-run build_order().

        Raises ValueError if orders is empty.
        """
        if not orders:
            raise ValueError("Cannot create plan with empty orders list")

        plan_id = str(uuid.uuid4())
        first = orders[0]

        # Compute totals from child specs
        target_size = sum(o.size for o in orders)
        target_stake = sum(o.stake_usd for o in orders)

        # Serialize children for deferred reconstruction
        serialized_children = [_serialize_order_spec(o) for o in orders]
        plan_metadata: dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "children": serialized_children,
        }
        if metadata:
            plan_metadata["extra"] = {
                k: v for k, v in metadata.items()
                if isinstance(v, (str, int, float, bool, type(None), list, dict))
            }

        now = dt.datetime.now(dt.timezone.utc).isoformat()

        plan = ExecutionPlanRecord(
            plan_id=plan_id,
            market_id=first.market_id,
            token_id=first.token_id,
            strategy_type=strategy_type,
            action_side=first.action_side,
            outcome_side=first.outcome_side,
            target_size=round(target_size, 6),
            target_stake_usd=round(target_stake, 2),
            filled_size=0.0,
            avg_fill_price=0.0,
            total_children=len(orders),
            completed_children=0,
            active_child_order_id="",
            next_child_index=0,
            status="planned",
            dry_run=first.dry_run,
            error="",
            metadata_json=json.dumps(plan_metadata),
            created_at=now,
            updated_at=now,
        )

        self._db.insert_execution_plan(plan)

        log.info(
            "plan_controller.created",
            plan_id=plan_id[:8],
            strategy=strategy_type,
            children=len(orders),
            target_size=round(target_size, 4),
            market_id=first.market_id,
        )
        self._alert(
            "info",
            f"Execution plan created: {strategy_type} with {len(orders)} children "
            f"(plan={plan_id[:8]} market={first.market_id[:8]})",
        )
        return plan

    def get_first_child_spec(self, plan: ExecutionPlanRecord) -> OrderSpec:
        """Return the first child OrderSpec from the plan's metadata.

        Assigns a fresh order_id so the child can be submitted.
        Raises UnsupportedSpecVersion if metadata version is unrecognized.
        """
        meta = json.loads(plan.metadata_json)
        _validate_spec_version(meta, plan.plan_id)
        children = meta.get("children", [])
        if not children:
            raise ValueError(f"Plan {plan.plan_id} has no serialized children")

        spec = _deserialize_order_spec(children[0])
        # Fresh order_id for the child to be submitted
        spec.order_id = str(uuid.uuid4())
        return spec

    def activate_plan(self, plan_id: str, first_child_order_id: str) -> None:
        """Transition plan from planned → active after first child is submitted."""
        self._db.update_execution_plan(
            plan_id,
            status="active",
            active_child_order_id=first_child_order_id,
            next_child_index=1,
        )
        log.info(
            "plan_controller.activated",
            plan_id=plan_id[:8],
            first_child=first_child_order_id[:8],
        )

    def update_plan_from_child(self, order: OrderRecord) -> OrderSpec | None:
        """Called when a child order's status changes.

        Returns the next OrderSpec to submit, or None if no action needed.
        """
        plan_id = getattr(order, "parent_plan_id", "")
        if not plan_id:
            return None

        plan = self._db.get_execution_plan(plan_id)
        if plan is None:
            log.warning(
                "plan_controller.plan_not_found",
                plan_id=plan_id,
                order_id=order.order_id,
            )
            return None

        # Skip if plan is already terminal
        if plan.status in ("filled", "cancelled", "failed", "expired"):
            return None

        if order.status == "filled":
            return self._handle_child_filled(plan, order)
        elif order.status == "partial":
            self._handle_child_partial(plan, order)
            return None
        elif order.status in ("cancelled", "expired", "failed"):
            return self._handle_child_terminal(plan, order)
        return None

    def _handle_child_filled(
        self, plan: ExecutionPlanRecord, order: OrderRecord
    ) -> OrderSpec | None:
        """Handle a child order that has been fully filled."""
        # Recalculate aggregate from all children
        children = self._db.get_plan_children(plan.plan_id)
        total_filled = 0.0
        total_cost = 0.0
        terminal = 0

        for child in children:
            if child.status in ("filled", "partial"):
                child_fill = child.filled_size if child.filled_size > 0 else child.size
                child_price = child.avg_fill_price if child.avg_fill_price > 0 else child.price
                total_filled += child_fill
                total_cost += child_fill * child_price
            if child.status in ("filled", "cancelled", "expired", "failed"):
                terminal += 1

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        # Check if all children are done
        if terminal >= plan.total_children:
            # All children processed — determine final status
            fill_ratio = total_filled / plan.target_size if plan.target_size > 0 else 0.0
            final_status = "filled" if fill_ratio >= 0.99 else "partial"

            self._db.update_execution_plan(
                plan.plan_id,
                status=final_status,
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
            )
            log.info(
                "plan_controller.completed",
                plan_id=plan.plan_id[:8],
                status=final_status,
                filled_size=round(total_filled, 4),
                fill_ratio=round(fill_ratio, 4),
            )
            self._alert(
                "info",
                f"Execution plan {final_status}: {plan.plan_id[:8]} "
                f"(filled={round(total_filled, 2)} fill_ratio={round(fill_ratio, 3)})",
            )
            return None

        # More children to submit — get the next one
        next_idx = plan.next_child_index
        meta = json.loads(plan.metadata_json)
        try:
            _validate_spec_version(meta, plan.plan_id)
        except UnsupportedSpecVersion as e:
            self._db.update_execution_plan(
                plan.plan_id,
                status="failed",
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
                error=str(e),
            )
            self._alert("warning", f"Plan {plan.plan_id[:8]} failed: {e}")
            log.warning("plan_controller.unsupported_spec_version", plan_id=plan.plan_id[:8], error=str(e))
            return None
        serialized = meta.get("children", [])

        if next_idx >= len(serialized):
            # No more children available — shouldn't happen but handle gracefully
            self._db.update_execution_plan(
                plan.plan_id,
                status="partial" if total_filled > 0 else "failed",
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
                error="next_child_index exceeds serialized children count",
            )
            log.warning(
                "plan_controller.no_more_children",
                plan_id=plan.plan_id[:8],
                next_idx=next_idx,
                serialized_count=len(serialized),
            )
            return None

        next_spec = _deserialize_order_spec(serialized[next_idx])
        next_spec.order_id = str(uuid.uuid4())

        # Clear active_child_order_id — the submission path will set it
        # to the new child's order_id once actually submitted.
        self._db.update_execution_plan(
            plan.plan_id,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
            completed_children=terminal,
            next_child_index=next_idx + 1,
            active_child_order_id="",
        )

        log.info(
            "plan_controller.next_child",
            plan_id=plan.plan_id[:8],
            child_index=next_idx,
            child_order_id=next_spec.order_id[:8],
        )
        self._alert(
            "info",
            f"Plan {plan.plan_id[:8]} advancing to child {next_idx + 1}/{plan.total_children}",
        )
        return next_spec

    def _handle_child_partial(
        self, plan: ExecutionPlanRecord, order: OrderRecord
    ) -> None:
        """Handle a child order with partial fill — update parent aggregates."""
        children = self._db.get_plan_children(plan.plan_id)
        total_filled = 0.0
        total_cost = 0.0
        for child in children:
            fill = child.filled_size if child.filled_size > 0 else 0.0
            price = child.avg_fill_price if child.avg_fill_price > 0 else child.price
            total_filled += fill
            total_cost += fill * price

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        self._db.update_execution_plan(
            plan.plan_id,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
        )

        fill_pct = total_filled / plan.target_size * 100 if plan.target_size > 0 else 0.0
        log.debug(
            "plan_controller.child_partial",
            plan_id=plan.plan_id[:8],
            order_id=order.order_id[:8],
            total_filled=round(total_filled, 4),
            fill_pct=round(fill_pct, 1),
        )
        self._alert(
            "info",
            f"Plan {plan.plan_id[:8]} partial fill progress: "
            f"{round(fill_pct, 1)}% of target ({round(total_filled, 2)}/{round(plan.target_size, 2)})",
        )

    def _handle_child_terminal(
        self, plan: ExecutionPlanRecord, order: OrderRecord
    ) -> OrderSpec | None:
        """Handle a child that was cancelled/expired/failed."""
        children = self._db.get_plan_children(plan.plan_id)
        total_filled = 0.0
        total_cost = 0.0
        terminal = 0

        for child in children:
            if child.status in ("filled", "partial"):
                child_fill = child.filled_size if child.filled_size > 0 else child.size
                child_price = child.avg_fill_price if child.avg_fill_price > 0 else child.price
                total_filled += child_fill
                total_cost += child_fill * child_price
            if child.status in ("filled", "cancelled", "expired", "failed"):
                terminal += 1

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0

        if terminal >= plan.total_children:
            # All done
            final_status = "partial" if total_filled > 0 else order.status
            self._db.update_execution_plan(
                plan.plan_id,
                status=final_status,
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
            )
            log.info(
                "plan_controller.terminal",
                plan_id=plan.plan_id[:8],
                status=final_status,
                reason=order.status,
            )
            self._alert(
                "warning" if total_filled == 0 else "info",
                f"Execution plan {final_status}: {plan.plan_id[:8]} "
                f"(reason={order.status} filled={round(total_filled, 2)})",
            )
            return None

        # More children remain — continue submitting
        next_idx = plan.next_child_index
        meta = json.loads(plan.metadata_json)
        try:
            _validate_spec_version(meta, plan.plan_id)
        except UnsupportedSpecVersion as e:
            self._db.update_execution_plan(
                plan.plan_id,
                status="failed",
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
                error=str(e),
            )
            self._alert("warning", f"Plan {plan.plan_id[:8]} failed: {e}")
            log.warning("plan_controller.unsupported_spec_version", plan_id=plan.plan_id[:8], error=str(e))
            return None
        serialized = meta.get("children", [])

        if next_idx >= len(serialized):
            final_status = "partial" if total_filled > 0 else order.status
            self._db.update_execution_plan(
                plan.plan_id,
                status=final_status,
                filled_size=round(total_filled, 6),
                avg_fill_price=round(avg_price, 6),
                completed_children=terminal,
                active_child_order_id="",
            )
            return None

        next_spec = _deserialize_order_spec(serialized[next_idx])
        next_spec.order_id = str(uuid.uuid4())

        # Clear active_child_order_id — submission path will set it
        self._db.update_execution_plan(
            plan.plan_id,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
            completed_children=terminal,
            next_child_index=next_idx + 1,
            active_child_order_id="",
        )

        log.info(
            "plan_controller.next_child_after_terminal",
            plan_id=plan.plan_id[:8],
            child_index=next_idx,
            terminal_reason=order.status,
        )
        return next_spec

    def cancel_plan(self, plan_id: str, reason: str = "") -> str:
        """Cancel a plan and mark it as cancelled.

        Returns the active_child_order_id if one exists, so the caller
        can cancel it on the venue.  Returns empty string if no active child.
        """
        plan = self._db.get_execution_plan(plan_id)
        if plan is None:
            return ""
        if plan.status in ("filled", "cancelled", "failed", "expired"):
            return ""

        active_child = plan.active_child_order_id or ""

        children = self._db.get_plan_children(plan_id)
        total_filled = sum(
            (c.filled_size if c.filled_size > 0 else 0.0)
            for c in children
            if c.status in ("filled", "partial")
        )

        final_status = "partial" if total_filled > 0 else "cancelled"
        self._db.update_execution_plan(
            plan_id,
            status=final_status,
            active_child_order_id="",
            error=reason or "cancelled",
        )
        log.info(
            "plan_controller.cancelled",
            plan_id=plan_id[:8],
            status=final_status,
            filled_size=round(total_filled, 4),
            active_child=active_child[:8] if active_child else "",
        )
        self._alert(
            "warning",
            f"Execution plan cancelled: {plan_id[:8]} "
            f"(reason={reason or 'cancelled'} filled={round(total_filled, 2)})",
        )
        return active_child

    def get_plan_summary(self, plan_id: str) -> dict[str, Any]:
        """Return a dashboard-friendly summary of a plan and its children."""
        plan = self._db.get_execution_plan(plan_id)
        if plan is None:
            return {}

        children = self._db.get_plan_children(plan_id)
        terminal_pct = (
            plan.completed_children / plan.total_children * 100
            if plan.total_children > 0
            else 0.0
        )
        fill_pct = (
            plan.filled_size / plan.target_size * 100
            if plan.target_size > 0
            else 0.0
        )

        return {
            "plan_id": plan.plan_id,
            "market_id": plan.market_id,
            "strategy_type": plan.strategy_type,
            "status": plan.status,
            "action_side": plan.action_side,
            "outcome_side": plan.outcome_side,
            "target_size": plan.target_size,
            "target_stake_usd": plan.target_stake_usd,
            "filled_size": plan.filled_size,
            "avg_fill_price": plan.avg_fill_price,
            "total_children": plan.total_children,
            "terminal_children": plan.completed_children,
            "terminal_pct": round(terminal_pct, 1),
            "fill_pct": round(fill_pct, 1),
            "dry_run": plan.dry_run,
            "error": plan.error,
            "created_at": plan.created_at,
            "updated_at": plan.updated_at,
            "children": [
                {
                    "order_id": c.order_id,
                    "child_index": c.child_index,
                    "status": c.status,
                    "price": c.price,
                    "size": c.size,
                    "filled_size": c.filled_size,
                    "avg_fill_price": c.avg_fill_price,
                }
                for c in children
            ],
        }
