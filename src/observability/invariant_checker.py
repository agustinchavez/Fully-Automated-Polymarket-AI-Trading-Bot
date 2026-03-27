"""Invariant checker — detects data consistency issues.

Checks:
  1. Duplicate positions (same market_id)
  2. Orphaned positions (no matching trade)
  3. Conflicting SELL orders (multiple active SELL for same market)
  4. Direction field mismatches (action_side/outcome_side vs direction)
  5. Stale positions (open > 7 days)
  6. SELL order without position (dangling exit)
  7. Multiple entry orders for same market
  8. Filled trades missing canonical fields
  9. Filled BUY trade with no position or closed position
  10. Position with empty outcome_side
  11. Active orders missing canonical fields
  12. Reconciled SELL fills that hit orphan fallback
  13. Stale execution plan (active but no active child and remaining children)
  14. Active child order of terminal plan
  15. Plan child index overflow (next_child_index > total_children)
  16. Plan overfill (filled_size > target_size * 1.01)
  17. Plan child missing canonical fields
  18. Partial plan with zero fill
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)

STALE_HOURS = 168  # 7 days


@dataclass
class InvariantViolation:
    """A single invariant violation."""
    check: str
    severity: str  # "critical" or "warning"
    market_id: str
    message: str


def check_invariants(db: object) -> list[InvariantViolation]:
    """Run all invariant checks against the database.

    Returns a list of violations found. Empty list means all checks pass.
    """
    violations: list[InvariantViolation] = []

    try:
        violations.extend(_check_duplicate_positions(db))
    except Exception as e:
        log.warning("invariant_checker.duplicate_positions_error", error=str(e))

    try:
        violations.extend(_check_orphaned_positions(db))
    except Exception as e:
        log.warning("invariant_checker.orphaned_positions_error", error=str(e))

    try:
        violations.extend(_check_conflicting_sell_orders(db))
    except Exception as e:
        log.warning("invariant_checker.conflicting_sell_error", error=str(e))

    try:
        violations.extend(_check_direction_mismatches(db))
    except Exception as e:
        log.warning("invariant_checker.direction_mismatch_error", error=str(e))

    try:
        violations.extend(_check_stale_positions(db))
    except Exception as e:
        log.warning("invariant_checker.stale_positions_error", error=str(e))

    try:
        violations.extend(_check_sell_without_position(db))
    except Exception as e:
        log.warning("invariant_checker.sell_without_position_error", error=str(e))

    try:
        violations.extend(_check_multiple_entry_orders(db))
    except Exception as e:
        log.warning("invariant_checker.multiple_entry_orders_error", error=str(e))

    try:
        violations.extend(_check_missing_canonical_fields(db))
    except Exception as e:
        log.warning("invariant_checker.missing_canonical_error", error=str(e))

    try:
        violations.extend(_check_filled_trade_no_position(db))
    except Exception as e:
        log.warning("invariant_checker.filled_no_position_error", error=str(e))

    try:
        violations.extend(_check_empty_outcome_position(db))
    except Exception as e:
        log.warning("invariant_checker.empty_outcome_error", error=str(e))

    try:
        violations.extend(_check_active_order_missing_canonical(db))
    except Exception as e:
        log.warning("invariant_checker.active_order_canonical_error", error=str(e))

    try:
        violations.extend(_check_orphan_sell_fallback(db))
    except Exception as e:
        log.warning("invariant_checker.orphan_sell_fallback_error", error=str(e))

    try:
        violations.extend(_check_stale_execution_plan(db))
    except Exception as e:
        log.warning("invariant_checker.stale_execution_plan_error", error=str(e))

    try:
        violations.extend(_check_terminal_plan_active_child(db))
    except Exception as e:
        log.warning("invariant_checker.terminal_plan_active_child_error", error=str(e))

    try:
        violations.extend(_check_plan_child_index_overflow(db))
    except Exception as e:
        log.warning("invariant_checker.plan_child_index_overflow_error", error=str(e))

    try:
        violations.extend(_check_plan_overfill(db))
    except Exception as e:
        log.warning("invariant_checker.plan_overfill_error", error=str(e))

    try:
        violations.extend(_check_plan_child_missing_canonical(db))
    except Exception as e:
        log.warning("invariant_checker.plan_child_missing_canonical_error", error=str(e))

    try:
        violations.extend(_check_partial_plan_zero_fill(db))
    except Exception as e:
        log.warning("invariant_checker.partial_plan_zero_fill_error", error=str(e))

    return violations


def _check_duplicate_positions(db: object) -> list[InvariantViolation]:
    """Detect multiple positions for the same market."""
    rows = db._conn.execute(
        """SELECT market_id, COUNT(*) as cnt FROM positions
        GROUP BY market_id HAVING cnt > 1"""
    ).fetchall()

    return [
        InvariantViolation(
            check="duplicate_positions",
            severity="critical",
            market_id=r["market_id"],
            message=f"Market {r['market_id'][:8]} has {r['cnt']} positions",
        )
        for r in rows
    ]


def _check_orphaned_positions(db: object) -> list[InvariantViolation]:
    """Detect positions with no matching trade."""
    rows = db._conn.execute(
        """SELECT p.market_id FROM positions p
        LEFT JOIN trades t ON p.market_id = t.market_id
        WHERE t.id IS NULL"""
    ).fetchall()

    return [
        InvariantViolation(
            check="orphaned_position",
            severity="warning",
            market_id=r["market_id"],
            message=f"Position {r['market_id'][:8]} has no matching trade",
        )
        for r in rows
    ]


def _check_conflicting_sell_orders(db: object) -> list[InvariantViolation]:
    """Detect multiple active SELL orders for the same market."""
    rows = db._conn.execute(
        """SELECT market_id, COUNT(*) as cnt FROM open_orders
        WHERE side = 'SELL'
        AND status IN ('submitted', 'pending')
        GROUP BY market_id HAVING cnt > 1"""
    ).fetchall()

    return [
        InvariantViolation(
            check="conflicting_sell_orders",
            severity="critical",
            market_id=r["market_id"],
            message=f"Market {r['market_id'][:8]} has {r['cnt']} active SELL orders",
        )
        for r in rows
    ]


def _check_direction_mismatches(db: object) -> list[InvariantViolation]:
    """Detect positions where action_side/outcome_side disagree with direction."""
    from src.execution.direction import parse_direction

    rows = db._conn.execute(
        """SELECT market_id, direction, action_side, outcome_side
        FROM positions
        WHERE action_side != '' AND outcome_side != ''"""
    ).fetchall()

    violations = []
    for r in rows:
        expected_a, expected_o = parse_direction(r["direction"])
        if expected_a and (expected_a != r["action_side"] or expected_o != r["outcome_side"]):
            violations.append(InvariantViolation(
                check="direction_mismatch",
                severity="warning",
                market_id=r["market_id"],
                message=(
                    f"Position {r['market_id'][:8]}: direction={r['direction']} "
                    f"but action_side={r['action_side']}, outcome_side={r['outcome_side']}"
                ),
            ))
    return violations


def _check_stale_positions(db: object) -> list[InvariantViolation]:
    """Detect positions open for more than STALE_HOURS hours."""
    import datetime as _dt

    cutoff = (
        _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=STALE_HOURS)
    ).isoformat()

    rows = db._conn.execute(
        """SELECT market_id, opened_at FROM positions
        WHERE opened_at < ? AND opened_at != ''""",
        (cutoff,),
    ).fetchall()

    return [
        InvariantViolation(
            check="stale_position",
            severity="warning",
            market_id=r["market_id"],
            message=f"Position {r['market_id'][:8]} open since {r['opened_at'][:10]}",
        )
        for r in rows
    ]


def _check_sell_without_position(db: object) -> list[InvariantViolation]:
    """Detect active SELL orders for markets with no open position."""
    rows = db._conn.execute(
        """SELECT o.market_id, o.order_id FROM open_orders o
        WHERE o.side = 'SELL'
        AND o.status IN ('submitted', 'pending')
        AND o.market_id NOT IN (SELECT market_id FROM positions)"""
    ).fetchall()

    return [
        InvariantViolation(
            check="sell_without_position",
            severity="critical",
            market_id=r["market_id"],
            message=f"Active SELL order {r['order_id'][:8]} for market {r['market_id'][:8]} with no position",
        )
        for r in rows
    ]


def _check_multiple_entry_orders(db: object) -> list[InvariantViolation]:
    """Detect multiple active BUY orders for the same market."""
    rows = db._conn.execute(
        """SELECT market_id, COUNT(*) as cnt FROM open_orders
        WHERE side != 'SELL'
        AND status IN ('submitted', 'pending')
        GROUP BY market_id HAVING cnt > 1"""
    ).fetchall()

    return [
        InvariantViolation(
            check="multiple_entry_orders",
            severity="warning",
            market_id=r["market_id"],
            message=f"Market {r['market_id'][:8]} has {r['cnt']} active entry orders",
        )
        for r in rows
    ]


def _check_missing_canonical_fields(db: object) -> list[InvariantViolation]:
    """Detect filled trades missing action_side or outcome_side."""
    rows = db._conn.execute(
        """SELECT id, market_id FROM trades
        WHERE status = 'FILLED'
        AND (action_side = '' OR outcome_side = '')"""
    ).fetchall()

    return [
        InvariantViolation(
            check="missing_canonical_fields",
            severity="warning",
            market_id=r["market_id"],
            message=f"Filled trade {r['id'][:8]} missing canonical direction fields",
        )
        for r in rows
    ]


def _check_filled_trade_no_position(db: object) -> list[InvariantViolation]:
    """Detect filled BUY trades with no matching position or closed position."""
    rows = db._conn.execute(
        """SELECT t.id, t.market_id FROM trades t
        WHERE t.status = 'FILLED'
        AND t.side != 'SELL'
        AND t.market_id NOT IN (SELECT market_id FROM positions)
        AND t.market_id NOT IN (SELECT market_id FROM closed_positions)"""
    ).fetchall()

    return [
        InvariantViolation(
            check="filled_trade_no_position",
            severity="warning",
            market_id=r["market_id"],
            message=f"Filled trade {r['id'][:8]} for market {r['market_id'][:8]} has no position",
        )
        for r in rows
    ]


def _check_empty_outcome_position(db: object) -> list[InvariantViolation]:
    """Detect positions with empty outcome_side."""
    rows = db._conn.execute(
        """SELECT market_id FROM positions WHERE outcome_side = ''"""
    ).fetchall()

    return [
        InvariantViolation(
            check="empty_outcome_position",
            severity="warning",
            market_id=r["market_id"],
            message=f"Position {r['market_id'][:8]} has empty outcome_side",
        )
        for r in rows
    ]


def _check_active_order_missing_canonical(db: object) -> list[InvariantViolation]:
    """Detect active orders missing action_side or outcome_side."""
    rows = db._conn.execute(
        """SELECT order_id, market_id FROM open_orders
        WHERE status IN ('submitted', 'pending')
        AND (action_side = '' OR outcome_side = '')"""
    ).fetchall()

    return [
        InvariantViolation(
            check="active_order_missing_canonical",
            severity="warning",
            market_id=r["market_id"],
            message=f"Active order {r['order_id'][:8]} missing canonical direction fields",
        )
        for r in rows
    ]


def _check_orphan_sell_fallback(db: object) -> list[InvariantViolation]:
    """Detect closed positions with SELL_FILLED_ORPHAN close reason.

    These indicate reconciled SELL fills that had no matching position,
    suggesting a state inconsistency at the time of the fill.
    """
    rows = db._conn.execute(
        """SELECT market_id FROM closed_positions
        WHERE close_reason = 'SELL_FILLED_ORPHAN'"""
    ).fetchall()

    return [
        InvariantViolation(
            check="orphan_sell_fallback",
            severity="warning",
            market_id=r["market_id"],
            message=f"Market {r['market_id'][:8]} had a SELL fill with no matching position",
        )
        for r in rows
    ]


def _check_stale_execution_plan(db: object) -> list[InvariantViolation]:
    """Detect active execution plans stuck with no active child order.

    An active plan with no active_child_order_id and remaining children
    indicates the next child was never submitted — the plan is stuck.
    """
    # Check if execution_plans table exists
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT plan_id, market_id FROM execution_plans
        WHERE status = 'active'
        AND (active_child_order_id = '' OR active_child_order_id IS NULL)
        AND next_child_index < total_children"""
    ).fetchall()

    return [
        InvariantViolation(
            check="stale_execution_plan",
            severity="critical",
            market_id=r["market_id"],
            message=f"Plan {r['plan_id'][:8]} is active but has no active child and remaining children",
        )
        for r in rows
    ]


def _check_terminal_plan_active_child(db: object) -> list[InvariantViolation]:
    """#14: Detect active child orders whose parent plan is terminal.

    An order with status submitted/pending whose parent_plan_id references
    a terminal plan indicates an orphaned child that should have been cancelled.
    """
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT o.order_id, o.market_id, o.parent_plan_id, ep.status AS plan_status
        FROM open_orders o
        JOIN execution_plans ep ON o.parent_plan_id = ep.plan_id
        WHERE o.status IN ('submitted', 'pending')
        AND o.parent_plan_id != ''
        AND ep.status IN ('filled', 'partial', 'cancelled', 'failed', 'expired')"""
    ).fetchall()

    return [
        InvariantViolation(
            check="terminal_plan_active_child",
            severity="critical",
            market_id=r["market_id"],
            message=(
                f"Order {r['order_id'][:8]} is active but parent plan "
                f"{r['parent_plan_id'][:8]} is {r['plan_status']}"
            ),
        )
        for r in rows
    ]


def _check_plan_child_index_overflow(db: object) -> list[InvariantViolation]:
    """#15: Detect non-terminal plans where next_child_index > total_children."""
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT plan_id, market_id, next_child_index, total_children
        FROM execution_plans
        WHERE status NOT IN ('filled', 'partial', 'cancelled', 'failed', 'expired')
        AND next_child_index > total_children"""
    ).fetchall()

    return [
        InvariantViolation(
            check="plan_child_index_overflow",
            severity="warning",
            market_id=r["market_id"],
            message=(
                f"Plan {r['plan_id'][:8]} has next_child_index={r['next_child_index']} "
                f"but total_children={r['total_children']}"
            ),
        )
        for r in rows
    ]


def _check_plan_overfill(db: object) -> list[InvariantViolation]:
    """#16: Detect plans where filled_size exceeds target_size by >1%."""
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT plan_id, market_id, filled_size, target_size
        FROM execution_plans
        WHERE target_size > 0
        AND filled_size > target_size * 1.01"""
    ).fetchall()

    return [
        InvariantViolation(
            check="plan_overfill",
            severity="critical",
            market_id=r["market_id"],
            message=(
                f"Plan {r['plan_id'][:8]} overfilled: "
                f"{r['filled_size']:.2f} > target {r['target_size']:.2f}"
            ),
        )
        for r in rows
    ]


def _check_plan_child_missing_canonical(db: object) -> list[InvariantViolation]:
    """#17: Detect active plan children missing canonical direction fields."""
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT order_id, market_id FROM open_orders
        WHERE parent_plan_id != ''
        AND status IN ('submitted', 'pending')
        AND (action_side = '' OR outcome_side = '')"""
    ).fetchall()

    return [
        InvariantViolation(
            check="plan_child_missing_canonical",
            severity="warning",
            market_id=r["market_id"],
            message=f"Plan child {r['order_id'][:8]} missing canonical direction fields",
        )
        for r in rows
    ]


def _check_partial_plan_zero_fill(db: object) -> list[InvariantViolation]:
    """#18: Detect plans with status='partial' but zero filled_size."""
    tbl = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_plans'"
    ).fetchone()
    if not tbl:
        return []

    rows = db._conn.execute(
        """SELECT plan_id, market_id FROM execution_plans
        WHERE status = 'partial'
        AND filled_size <= 0"""
    ).fetchall()

    return [
        InvariantViolation(
            check="partial_plan_zero_fill",
            severity="warning",
            market_id=r["market_id"],
            message=f"Plan {r['plan_id'][:8]} is partial but has zero filled_size",
        )
        for r in rows
    ]
