"""Order reconciliation loop — polls CLOB for fill status.

Checks submitted/pending orders against the CLOB, handling:
  - Full fills: creates trade + position
  - Partial fills: updates order, creates/updates position, records incremental trade
  - Cancelled/expired by exchange: marks order status
  - Stale unfilled orders: optional auto-cancellation
  - Terminal order pruning: archives filled/cancelled/expired/failed orders

All behavior is config-gated and disabled by default.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config import ExecutionConfig, is_live_trading_enabled
from src.execution.direction import parse_direction
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """Summary of one reconciliation pass."""
    checked: int = 0
    filled: int = 0
    partial: int = 0
    cancelled: int = 0
    stale_cancelled: int = 0
    pruned: int = 0
    errors: int = 0


class OrderReconciler:
    """Reconciles open orders against the CLOB."""

    def __init__(
        self,
        db: object,
        clob: object,
        config: ExecutionConfig,
        fill_tracker: object | None = None,
        on_buy_fill: Callable[[str], None] | None = None,
        exit_finalizer: object | None = None,
    ):
        self._db = db
        self._clob = clob
        self._config = config
        self._fill_tracker = fill_tracker
        self._on_buy_fill = on_buy_fill  # callback(token_id) for WS subscription
        self._exit_finalizer = exit_finalizer

    def reconcile_once(self) -> ReconciliationResult:
        """Run a single reconciliation pass over all open orders.

        Returns a ReconciliationResult summarizing what happened.
        """
        result = ReconciliationResult()

        try:
            pending = self._db.get_submitted_orders()
            pending += [
                o for o in self._db.get_open_orders(status="pending")
                if o.clob_order_id  # only poll orders with a CLOB ID
            ]
        except Exception as e:
            log.error("reconciler.db_error", error=str(e))
            result.errors += 1
            return result

        for order in pending:
            result.checked += 1
            try:
                self._reconcile_order(order, result)
            except Exception as e:
                log.warning(
                    "reconciler.order_error",
                    order_id=order.order_id[:8],
                    error=str(e),
                )
                result.errors += 1

        # Stale order cancellation
        if self._config.stale_order_cancel_enabled:
            try:
                stale_secs = getattr(self._config, "stale_order_cancel_secs", 3600)
                stale_orders = self._db.get_stale_orders(stale_secs)
                for order in stale_orders:
                    self._cancel_stale_order(order, result)
            except Exception as e:
                log.warning("reconciler.stale_error", error=str(e))
                result.errors += 1

        # Prune terminal orders
        pruned = self._prune_terminal_orders()
        result.pruned = pruned

        log.info(
            "reconciler.pass_complete",
            checked=result.checked,
            filled=result.filled,
            partial=result.partial,
            cancelled=result.cancelled,
            stale_cancelled=result.stale_cancelled,
            pruned=result.pruned,
            errors=result.errors,
        )
        return result

    def _reconcile_order(self, order: object, result: ReconciliationResult) -> None:
        """Check a single order against the CLOB and update accordingly."""
        if not order.clob_order_id:
            return

        clob_data = self._clob.get_order_status(order.clob_order_id)
        if not isinstance(clob_data, dict):
            return

        clob_status = str(clob_data.get("status", "")).lower()

        if clob_status == "matched":
            self._handle_fill(order, clob_data, result)
        elif clob_status == "live":
            # Still on book — check for partial fills
            taking = clob_data.get("takingAmount", clob_data.get("taking_amount", "0"))
            try:
                taking_val = float(taking)
            except (TypeError, ValueError):
                taking_val = 0.0

            if taking_val > 0 and order.price > 0:
                fill_size = taking_val / order.price
                if fill_size > order.filled_size:
                    self._handle_partial_fill(order, fill_size, result)
        elif clob_status in ("cancelled", "canceled"):
            self._db.update_order_status(order.order_id, "cancelled")
            result.cancelled += 1
            self._notify_unfilled(order.order_id)
            log.info("reconciler.order_cancelled_by_exchange", order_id=order.order_id[:8])
        elif clob_status == "expired":
            self._db.update_order_status(order.order_id, "expired")
            result.cancelled += 1
            self._notify_unfilled(order.order_id)
            log.info("reconciler.order_expired", order_id=order.order_id[:8])

    def _get_order_direction(self, order: object) -> tuple[str, str]:
        """Extract canonical (action_side, outcome_side) from an order."""
        action = getattr(order, "action_side", "") or ""
        outcome = getattr(order, "outcome_side", "") or ""
        if action:
            return (action, outcome)
        return parse_direction(order.side)

    def _handle_fill(self, order: object, clob_data: dict, result: ReconciliationResult) -> None:
        """Handle a fully filled order."""
        from src.storage.models import TradeRecord, PositionRecord

        action_side, outcome_side = self._get_order_direction(order)

        # Parse fill data
        taking = clob_data.get("takingAmount", clob_data.get("taking_amount", "0"))
        try:
            taking_val = float(taking)
        except (TypeError, ValueError):
            taking_val = 0.0

        if taking_val > 0 and order.price > 0:
            fill_size = taking_val / order.price
            fill_price = order.price
        else:
            fill_size = order.size
            fill_price = order.price

        # Update order status
        self._db.update_order_status(
            order.order_id, "filled",
            filled_size=fill_size, avg_fill_price=fill_price,
        )

        # Create trade record
        self._db.insert_trade(TradeRecord(
            id=str(uuid.uuid4()),
            order_id=order.order_id,
            market_id=order.market_id,
            token_id=order.token_id,
            side=order.side,
            price=fill_price,
            size=fill_size,
            stake_usd=order.stake_usd,
            status="FILLED",
            dry_run=False,
            action_side=action_side,
            outcome_side=outcome_side,
        ))

        # Create or update position
        is_sell = action_side == "SELL" or order.side == "SELL"
        if is_sell:
            self._handle_sell_fill(order, fill_price, fill_size)
        else:
            # BUY fill — create position
            self._db.upsert_position(PositionRecord(
                market_id=order.market_id,
                token_id=order.token_id,
                direction=order.side,
                entry_price=fill_price,
                size=fill_size,
                stake_usd=order.stake_usd,
                current_price=fill_price,
                pnl=0.0,
                action_side=action_side,
                outcome_side=outcome_side,
            ))
            # Subscribe to WS feed for price monitoring
            if self._on_buy_fill and order.token_id:
                try:
                    self._on_buy_fill(order.token_id)
                except Exception:
                    pass

        self._notify_fill(order, fill_price, fill_size)

        result.filled += 1
        log.info(
            "reconciler.order_filled",
            order_id=order.order_id[:8],
            fill_price=fill_price,
            fill_size=fill_size,
        )

    def _handle_sell_fill(self, order: object, fill_price: float, fill_size: float) -> None:
        """Handle a SELL fill using the actual open position for accurate PnL."""
        try:
            real_pos = self._db.get_position(order.market_id)
            if real_pos:
                # Compute realized PnL from actual entry vs fill exit
                pnl = (fill_price - real_pos.entry_price) * fill_size
                if self._exit_finalizer:
                    self._exit_finalizer.finalize(
                        pos=real_pos,
                        exit_price=fill_price,
                        pnl=round(pnl, 4),
                        close_reason="SELL_FILLED",
                    )
                else:
                    self._db.archive_position(
                        pos=real_pos,
                        exit_price=fill_price,
                        pnl=round(pnl, 4),
                        close_reason="SELL_FILLED",
                    )
                    self._db.remove_position(order.market_id)
            else:
                # Orphan SELL fill — no position found
                log.warning(
                    "reconciler.orphan_sell_fill",
                    order_id=order.order_id[:8],
                    market_id=order.market_id[:8],
                )
                self._db.archive_position(
                    pos=type("P", (), {
                        "market_id": order.market_id,
                        "token_id": order.token_id,
                        "direction": order.side,
                        "entry_price": order.price,
                        "size": fill_size,
                        "stake_usd": order.stake_usd,
                        "opened_at": order.created_at,
                        "action_side": getattr(order, "action_side", "SELL"),
                        "outcome_side": getattr(order, "outcome_side", ""),
                    })(),
                    exit_price=fill_price,
                    pnl=0.0,
                    close_reason="SELL_FILLED_ORPHAN",
                )
                self._db.remove_position(order.market_id)
                if self._exit_finalizer:
                    try:
                        self._db.insert_alert(
                            "critical",
                            f"Orphan SELL fill: {order.market_id[:8]} — no position found",
                            "reconciler",
                        )
                    except Exception:
                        pass
        except Exception as e:
            log.warning(
                "reconciler.sell_fill_archive_error",
                order_id=order.order_id[:8],
                error=str(e),
            )

    def _handle_partial_fill(
        self, order: object, new_fill_size: float, result: ReconciliationResult,
    ) -> None:
        """Handle a partial fill — update order and create/update position."""
        from src.storage.models import PositionRecord, TradeRecord

        action_side, outcome_side = self._get_order_direction(order)

        old_filled = order.filled_size
        old_avg = order.avg_fill_price if order.avg_fill_price > 0 else order.price
        incremental = new_fill_size - old_filled

        if incremental <= 0:
            return

        # Weighted average price
        new_avg = (old_avg * old_filled + order.price * incremental) / new_fill_size

        self._db.update_order_status(
            order.order_id, "partial",
            filled_size=new_fill_size, avg_fill_price=round(new_avg, 6),
        )

        # Record an incremental trade for this partial fill event
        incremental_stake = order.stake_usd * (incremental / order.size) if order.size > 0 else 0
        self._db.insert_trade(TradeRecord(
            id=str(uuid.uuid4()),
            order_id=order.order_id,
            market_id=order.market_id,
            token_id=order.token_id,
            side=order.side,
            price=order.price,
            size=incremental,
            stake_usd=round(incremental_stake, 2),
            status="PARTIAL_FILL",
            dry_run=False,
            action_side=action_side,
            outcome_side=outcome_side,
        ))

        # Create/update position with partial fill
        is_sell = action_side == "SELL" or order.side == "SELL"
        if not is_sell:
            partial_stake = order.stake_usd * (new_fill_size / order.size) if order.size > 0 else 0
            self._db.upsert_position(PositionRecord(
                market_id=order.market_id,
                token_id=order.token_id,
                direction=order.side,
                entry_price=round(new_avg, 6),
                size=new_fill_size,
                stake_usd=round(partial_stake, 2),
                current_price=round(new_avg, 6),
                pnl=0.0,
                action_side=action_side,
                outcome_side=outcome_side,
            ))
            # Subscribe to WS feed on first partial fill
            if old_filled == 0 and self._on_buy_fill and order.token_id:
                try:
                    self._on_buy_fill(order.token_id)
                except Exception:
                    pass

        result.partial += 1
        log.info(
            "reconciler.partial_fill",
            order_id=order.order_id[:8],
            old_filled=old_filled,
            new_filled=new_fill_size,
            incremental=incremental,
            avg_price=round(new_avg, 6),
        )

    def _cancel_stale_order(self, order: object, result: ReconciliationResult) -> None:
        """Cancel a stale unfilled order."""
        try:
            self._clob.cancel_order(order.clob_order_id)
            self._db.update_order_status(order.order_id, "cancelled")
            result.stale_cancelled += 1
            self._notify_unfilled(order.order_id)
            log.info(
                "reconciler.stale_cancelled",
                order_id=order.order_id[:8],
                clob_order_id=order.clob_order_id[:8],
            )
        except Exception as e:
            log.warning(
                "reconciler.cancel_failed",
                order_id=order.order_id[:8],
                error=str(e),
            )
            result.errors += 1

    def _notify_fill(self, order: object, fill_price: float, fill_size: float) -> None:
        """Notify FillTracker of a fill, registering the order first if needed."""
        if not self._fill_tracker or not hasattr(self._fill_tracker, "record_fill"):
            return
        # FillTracker.record_fill() requires the order to be registered first
        pending = getattr(self._fill_tracker, "_pending_orders", {})
        if order.order_id not in pending:
            self._fill_tracker.register_order(
                order.order_id, order.market_id, order.price,
                order.size, "reconciled",
            )
        self._fill_tracker.record_fill(order.order_id, fill_price, fill_size)

    def _notify_unfilled(self, order_id: str) -> None:
        """Notify FillTracker of an unfilled/cancelled order."""
        if self._fill_tracker and hasattr(self._fill_tracker, "record_unfilled"):
            self._fill_tracker.record_unfilled(order_id)

    def _prune_terminal_orders(self) -> int:
        """Remove terminal orders (filled/cancelled/expired/failed) from open_orders.

        Returns the number of orders pruned.
        """
        try:
            count = self._db.prune_terminal_orders()
            if count > 0:
                log.info("reconciler.pruned_terminal_orders", count=count)
            return count
        except Exception as e:
            log.warning("reconciler.prune_error", error=str(e))
            return 0


async def run_reconciliation_loop(
    db: object,
    clob: object,
    config: ExecutionConfig,
    stop_event: asyncio.Event,
    fill_tracker: object | None = None,
    on_buy_fill: Callable[[str], None] | None = None,
    exit_finalizer: object | None = None,
) -> None:
    """Background loop that periodically reconciles open orders.

    Runs until stop_event is set. Only active in live mode with
    reconciliation_enabled=True.
    """
    reconciler = OrderReconciler(
        db, clob, config, fill_tracker, on_buy_fill,
        exit_finalizer=exit_finalizer,
    )
    interval = config.reconciliation_interval_secs

    log.info("reconciler.loop_started", interval_secs=interval)

    while not stop_event.is_set():
        try:
            reconciler.reconcile_once()
        except Exception as e:
            log.error("reconciler.loop_error", error=str(e))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break  # stop_event was set
        except asyncio.TimeoutError:
            pass  # timeout expired — run another pass

    log.info("reconciler.loop_stopped")
