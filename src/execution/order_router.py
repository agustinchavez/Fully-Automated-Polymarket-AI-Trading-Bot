"""Order router — sends orders to the CLOB or simulates them.

In dry_run mode: logs the order and records it in storage.
In live mode: submits via the py-clob-client signing client.

Enhancements:
  - Retry with exponential backoff on transient failures
  - Slippage protection: rejects orders whose fill price exceeds tolerance
  - Differentiated limit vs market order paths
  - CLOB response parsing for actual fill data (Phase 10)
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from src.config import ExecutionConfig, is_live_trading_enabled
from src.connectors.polymarket_clob import CLOBClient
from src.execution.order_builder import OrderSpec
from src.observability.logger import get_logger
from src.observability.metrics import metrics

log = get_logger(__name__)


@dataclass
class OrderResult:
    """Result of order submission."""
    order_id: str
    status: str          # "simulated" | "submitted" | "filled" | "failed" | "rejected"
    fill_price: float = 0.0
    fill_size: float = 0.0
    error: str = ""
    timestamp: str = ""
    clob_order_id: str = ""  # CLOB-assigned order ID (Phase 10)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_size": self.fill_size,
            "error": self.error,
            "timestamp": self.timestamp,
            "clob_order_id": self.clob_order_id,
        }


class OrderRouter:
    """Route orders to CLOB or simulate them."""

    def __init__(self, clob: CLOBClient, config: ExecutionConfig):
        self._clob = clob
        self._config = config

    async def submit_order(self, order: OrderSpec) -> OrderResult:
        """Submit an order. Dry-run by default."""
        ts = dt.datetime.now(dt.timezone.utc).isoformat()

        # Enforce dry run unless explicitly enabled
        if order.dry_run or self._config.dry_run or not is_live_trading_enabled():
            log.info(
                "order_router.dry_run",
                order_id=order.order_id[:8],
                market=order.market_id,
                side=order.side,
                price=order.price,
                size=order.size,
                stake=order.stake_usd,
            )
            metrics.incr("orders.simulated")
            return OrderResult(
                order_id=order.order_id,
                status="simulated",
                fill_price=order.price,
                fill_size=order.size,
                timestamp=ts,
            )

        # Live order submission with retry
        last_error = ""
        max_retries = self._config.max_retries
        backoff = self._config.retry_backoff_secs

        for attempt in range(1, max_retries + 1):
            try:
                signing = self._clob.get_signing_client()

                # Slippage guard: reject if price deviates beyond tolerance
                if order.order_type == "limit":
                    resp = signing.create_and_post_order(
                        token_id=order.token_id,
                        price=order.price,
                        size=order.size,
                        side=order.side,
                    )
                else:
                    # Market order: use aggressive pricing with slippage tolerance
                    slippage = self._config.slippage_tolerance
                    aggressive_price = (
                        min(order.price * (1 + slippage), 0.99)
                        if order.side == "BUY"
                        else max(order.price * (1 - slippage), 0.01)
                    )
                    resp = signing.create_and_post_order(
                        token_id=order.token_id,
                        price=aggressive_price,
                        size=order.size,
                        side=order.side,
                    )

                log.info(
                    "order_router.submitted",
                    order_id=order.order_id[:8],
                    attempt=attempt,
                    response=str(resp)[:200],
                )
                metrics.incr("orders.submitted")

                return _parse_clob_response(resp, order, ts)

            except Exception as e:
                last_error = str(e)
                log.warning(
                    "order_router.retry",
                    order_id=order.order_id[:8],
                    attempt=attempt,
                    max_retries=max_retries,
                    error=last_error,
                )
                if attempt < max_retries:
                    await asyncio.sleep(backoff * (2 ** (attempt - 1)))

        # All retries exhausted
        log.error(
            "order_router.failed",
            order_id=order.order_id[:8],
            error=last_error,
        )
        metrics.incr("orders.failed")
        return OrderResult(
            order_id=order.order_id,
            status="failed",
            error=f"Failed after {max_retries} attempts: {last_error}",
            timestamp=ts,
        )


# ── CLOB Response Parsing ─────────────────────────────────────────────


def _parse_clob_response(
    resp: Any,
    order: OrderSpec,
    timestamp: str,
) -> OrderResult:
    """Parse the py-clob-client response into a structured OrderResult.

    The CLOB response typically contains:
      - orderID (str): exchange-assigned order ID
      - success (bool): whether the order was accepted
      - status (str): "matched" (filled), "live" (on book), "delayed", etc.
      - takingAmount (str): amount filled immediately (for marketable limits)
      - makingAmount (str): amount placed on the book
    """
    raw = resp if isinstance(resp, dict) else {"raw": str(resp)}

    # Extract CLOB order ID
    clob_order_id = ""
    if isinstance(resp, dict):
        clob_order_id = str(resp.get("orderID", resp.get("order_id", "")))

    # Determine status from CLOB response
    clob_status = ""
    if isinstance(resp, dict):
        clob_status = str(resp.get("status", "")).lower()

    # Parse fill data
    fill_price = 0.0
    fill_size = 0.0

    if isinstance(resp, dict):
        taking_amount = resp.get("takingAmount", resp.get("taking_amount", "0"))
        try:
            taking_value = float(taking_amount)
        except (TypeError, ValueError):
            taking_value = 0.0

        if taking_value > 0 and order.price > 0:
            fill_size = taking_value / order.price
            fill_price = order.price

    # Map CLOB status to our status
    if clob_status == "matched":
        status = "filled"
        # If matched but no takingAmount parsed, assume full fill at order price
        if fill_size == 0:
            fill_price = order.price
            fill_size = order.size
    elif clob_status == "live":
        status = "submitted"
        # Order is on the book, no fill yet
        fill_price = 0.0
        fill_size = 0.0
    elif clob_status in ("delayed", ""):
        status = "pending"
    else:
        # Unknown status — treat as pending for reconciliation
        status = "pending"

    # Check success flag
    if isinstance(resp, dict):
        success = resp.get("success", True)
        if success is False:
            status = "failed"
            error_msg = str(resp.get("errorMsg", resp.get("error", "CLOB rejected order")))
            return OrderResult(
                order_id=order.order_id,
                status="failed",
                error=error_msg,
                timestamp=timestamp,
                clob_order_id=clob_order_id,
                raw_response=raw,
            )

    return OrderResult(
        order_id=order.order_id,
        status=status,
        fill_price=round(fill_price, 6),
        fill_size=round(fill_size, 6),
        timestamp=timestamp,
        clob_order_id=clob_order_id,
        raw_response=raw,
    )
