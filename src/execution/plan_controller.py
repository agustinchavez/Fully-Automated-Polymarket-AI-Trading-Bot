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

# Sentinel value for active_child_order_id during the timing window between
# the controller returning the next child spec and the submission path setting
# the real order_id. Prevents invariant #13 false-positives.
ADVANCING_SENTINEL = "__advancing__"

# Terminal plan statuses — once a plan reaches any of these, no further
# child updates should mutate it.
_TERMINAL_PLAN_STATUSES = frozenset({"filled", "partial", "cancelled", "failed", "expired"})


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
        # TWAP time staggering: store minimum interval between child submissions
        plan_metadata["min_child_interval_secs"] = 0
        if metadata and "min_child_interval_secs" in metadata:
            plan_metadata["min_child_interval_secs"] = metadata["min_child_interval_secs"]
        plan_metadata["last_child_completed_at"] = ""
        if metadata:
            plan_metadata["extra"] = {
                k: v for k, v in metadata.items()
                if k != "min_child_interval_secs"
                and isinstance(v, (str, int, float, bool, type(None), list, dict))
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
        if plan.status in _TERMINAL_PLAN_STATUSES:
            return None

        # Guard: child fill can arrive before activate_plan() is called
        if plan.status == "planned":
            log.info(
                "plan_controller.child_update_before_activation",
                plan_id=plan_id[:8],
                order_id=order.order_id[:8],
                child_status=order.status,
            )
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

        # Re-read guard: detect concurrent cancel/terminal between initial read and now
        plan = self._db.get_execution_plan(plan.plan_id)
        if plan is None or plan.status in _TERMINAL_PLAN_STATUSES:
            log.info(
                "plan_controller.reread_terminal_in_handle_filled",
                plan_id=(plan.plan_id[:8] if plan else "?"),
            )
            return None

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
            self._alert("critical", f"Plan {plan.plan_id[:8]} failed: {e}")
            log.error("plan_controller.unsupported_spec_version", plan_id=plan.plan_id[:8], error=str(e))
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

        # TWAP time stagger: enforce minimum interval between child submissions
        min_interval = meta.get("min_child_interval_secs", 0)
        now_ts = dt.datetime.now(dt.timezone.utc)
        if min_interval > 0:
            last_completed = meta.get("last_child_completed_at", "")
            if last_completed:
                try:
                    last_ts = dt.datetime.fromisoformat(last_completed)
                    elapsed = (now_ts - last_ts).total_seconds()
                    if elapsed < min_interval:
                        # Not enough time elapsed — record fill, let recovery pick up later
                        meta["last_child_completed_at"] = now_ts.isoformat()
                        self._db.update_execution_plan(
                            plan.plan_id,
                            filled_size=round(total_filled, 6),
                            avg_fill_price=round(avg_price, 6),
                            completed_children=terminal,
                            active_child_order_id="",
                            metadata_json=json.dumps(meta),
                        )
                        log.info(
                            "plan_controller.twap_interval_wait",
                            plan_id=plan.plan_id[:8],
                            elapsed=round(elapsed, 1),
                            interval=min_interval,
                        )
                        return None
                except (ValueError, TypeError):
                    pass  # bad timestamp — proceed normally
        # Record completion timestamp for future interval checks
        meta["last_child_completed_at"] = now_ts.isoformat()

        next_spec = _deserialize_order_spec(serialized[next_idx])
        next_spec.order_id = str(uuid.uuid4())

        # Use __advancing__ sentinel instead of "" to prevent invariant #13
        # from firing during the natural timing window between controller
        # clearing and submission setting the real order_id.
        self._db.update_execution_plan(
            plan.plan_id,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
            completed_children=terminal,
            next_child_index=next_idx + 1,
            active_child_order_id=ADVANCING_SENTINEL,
            metadata_json=json.dumps(meta),
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
        """Handle a child order with partial fill — update parent aggregates.

        Note: This method does not read metadata_json, so no spec-version
        validation is needed here.  If this method ever evolves to fetch
        next-child specs or depends on serialized metadata, add
        _validate_spec_version() at the read site.
        """
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

        # Re-read guard: detect concurrent cancel/terminal between initial read and now
        plan = self._db.get_execution_plan(plan.plan_id)
        if plan is None or plan.status in _TERMINAL_PLAN_STATUSES:
            log.info(
                "plan_controller.reread_terminal_in_handle_terminal",
                plan_id=(plan.plan_id[:8] if plan else "?"),
            )
            return None

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
            self._alert("critical", f"Plan {plan.plan_id[:8]} failed: {e}")
            log.error("plan_controller.unsupported_spec_version", plan_id=plan.plan_id[:8], error=str(e))
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

        # Use __advancing__ sentinel (same rationale as _handle_child_filled)
        self._db.update_execution_plan(
            plan.plan_id,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
            completed_children=terminal,
            next_child_index=next_idx + 1,
            active_child_order_id=ADVANCING_SENTINEL,
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
        if plan.status in _TERMINAL_PLAN_STATUSES:
            return ""

        active_child = plan.active_child_order_id or ""

        children = self._db.get_plan_children(plan_id)
        total_filled = 0.0
        total_cost = 0.0
        for c in children:
            if c.status in ("filled", "partial"):
                fill = c.filled_size if c.filled_size > 0 else 0.0
                price = c.avg_fill_price if c.avg_fill_price > 0 else c.price
                total_filled += fill
                total_cost += fill * price

        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        final_status = "partial" if total_filled > 0 else "cancelled"
        self._db.update_execution_plan(
            plan_id,
            status=final_status,
            filled_size=round(total_filled, 6),
            avg_fill_price=round(avg_price, 6),
            active_child_order_id="",
            error=reason or "cancelled",
        )
        # Cancel remaining non-terminal children in open_orders.
        # "partial" children still have an open venue order for the remaining
        # quantity, so we cancel those too — their filled portion is already
        # captured in the aggregate above.
        cancelled_children = 0
        for c in children:
            if c.status in ("submitted", "pending", "partial"):
                try:
                    self._db.update_order_status(c.order_id, "cancelled")
                    cancelled_children += 1
                except Exception:
                    pass  # best-effort

        log.info(
            "plan_controller.cancelled",
            plan_id=plan_id[:8],
            status=final_status,
            filled_size=round(total_filled, 4),
            active_child=active_child[:8] if active_child else "",
            cancelled_children=cancelled_children,
        )
        self._alert(
            "warning",
            f"Execution plan cancelled: {plan_id[:8]} "
            f"(reason={reason or 'cancelled'} filled={round(total_filled, 2)})",
        )
        return active_child

    def recover_stuck_plans(self) -> list[tuple[str, "OrderSpec"]]:
        """Detect and recover stuck plans — active with no active child and remaining children.

        For each stuck plan, attempts to reconstruct and return the next child
        OrderSpec.  If reconstruction fails (e.g. bad spec version), marks the
        plan as failed.

        Returns a list of (plan_id, next_spec) tuples for the caller to submit.
        """
        recovered: list[tuple[str, OrderSpec]] = []

        try:
            active_plans = self._db.get_active_execution_plans()
        except Exception:
            return recovered

        for plan in active_plans:
            if plan.status != "active":
                continue
            if plan.active_child_order_id:
                continue  # has an active child or is advancing — not stuck
            if plan.next_child_index >= plan.total_children:
                continue  # all children submitted — awaiting last fill

            # Check TWAP interval: plan may be deliberately waiting, not stuck
            try:
                _meta = json.loads(plan.metadata_json)
                _min_interval = _meta.get("min_child_interval_secs", 0)
                _last_completed = _meta.get("last_child_completed_at", "")
                if _min_interval > 0 and _last_completed:
                    _last_ts = dt.datetime.fromisoformat(_last_completed)
                    _elapsed = (dt.datetime.now(dt.timezone.utc) - _last_ts).total_seconds()
                    if _elapsed < _min_interval:
                        continue  # still waiting for interval, not stuck
            except (json.JSONDecodeError, ValueError, TypeError):
                pass  # proceed with normal recovery

            # This plan is stuck — attempt recovery
            try:
                meta = json.loads(plan.metadata_json)
                _validate_spec_version(meta, plan.plan_id)
                serialized = meta.get("children", [])

                next_idx = plan.next_child_index
                if next_idx >= len(serialized):
                    self._db.update_execution_plan(
                        plan.plan_id,
                        status="failed",
                        active_child_order_id="",
                        error="Recovery failed: next_child_index exceeds serialized children",
                    )
                    self._alert("warning", f"Plan {plan.plan_id[:8]} recovery failed: no more serialized children")
                    continue

                next_spec = _deserialize_order_spec(serialized[next_idx])
                next_spec.order_id = str(uuid.uuid4())

                self._db.update_execution_plan(
                    plan.plan_id,
                    next_child_index=next_idx + 1,
                    active_child_order_id=ADVANCING_SENTINEL,
                )

                recovered.append((plan.plan_id, next_spec))
                log.info(
                    "plan_controller.stuck_plan_recovered",
                    plan_id=plan.plan_id[:8],
                    child_index=next_idx,
                )
                self._alert(
                    "warning",
                    f"Plan {plan.plan_id[:8]} was stuck — recovering child {next_idx + 1}/{plan.total_children}",
                )

            except UnsupportedSpecVersion as e:
                self._db.update_execution_plan(
                    plan.plan_id,
                    status="failed",
                    active_child_order_id="",
                    error=str(e),
                )
                self._alert("critical", f"Plan {plan.plan_id[:8]} recovery failed: {e}")

            except Exception as e:
                log.warning(
                    "plan_controller.recovery_error",
                    plan_id=plan.plan_id[:8],
                    error=str(e),
                )

        return recovered

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

    def emit_plan_metrics(self) -> dict[str, float]:
        """Compute and emit plan-level metrics. Returns the metrics dict."""
        from src.observability.metrics import metrics

        try:
            all_plans = self._db._conn.execute(
                "SELECT status, strategy_type, filled_size, target_size, "
                "total_children, created_at, updated_at FROM execution_plans"
            ).fetchall()
        except Exception:
            return {}

        active = 0
        completed = 0
        cancelled = 0
        failed = 0
        total_fill_pct = 0.0
        fill_pct_count = 0
        total_children_sum = 0
        completed_plans = 0
        strategy_completions: dict[str, int] = {}
        recovered = metrics.snapshot().get("counters", {}).get(
            "plan_controller.stuck_plan_recovered", 0
        )

        for p in all_plans:
            status = p["status"]
            if status == "active":
                active += 1
            elif status in ("filled", "partial"):
                completed += 1
                completed_plans += 1
                total_children_sum += p["total_children"]
                strategy = p["strategy_type"] or "unknown"
                strategy_completions[strategy] = strategy_completions.get(strategy, 0) + 1
            elif status == "cancelled":
                cancelled += 1
            elif status == "failed":
                failed += 1

            if p["target_size"] and p["target_size"] > 0:
                fill_pct = p["filled_size"] / p["target_size"] * 100
                total_fill_pct += fill_pct
                fill_pct_count += 1

        avg_fill_pct = total_fill_pct / fill_pct_count if fill_pct_count > 0 else 0.0
        avg_children = total_children_sum / completed_plans if completed_plans > 0 else 0.0
        total = active + completed + cancelled + failed
        completion_rate = completed / total * 100 if total > 0 else 0.0
        cancel_rate = cancelled / total * 100 if total > 0 else 0.0

        metrics.gauge("plans.active_count", active)
        metrics.gauge("plans.avg_fill_pct", round(avg_fill_pct, 1))
        metrics.gauge("plans.completion_rate", round(completion_rate, 1))
        metrics.gauge("plans.cancel_rate", round(cancel_rate, 1))
        metrics.gauge("plans.avg_children_per_completed", round(avg_children, 1))

        for strategy, count in strategy_completions.items():
            metrics.gauge(f"plans.completions_by_strategy.{strategy}", count)

        result = {
            "active": active,
            "completed": completed,
            "cancelled": cancelled,
            "failed": failed,
            "avg_fill_pct": round(avg_fill_pct, 1),
            "completion_rate": round(completion_rate, 1),
            "cancel_rate": round(cancel_rate, 1),
            "avg_children_per_completed": round(avg_children, 1),
        }
        return result
