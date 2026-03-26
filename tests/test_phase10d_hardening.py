"""Tests for Phase 10D: Router/Venue Hardening & Invariant Expansion.

Covers:
  1. OrderResult canonical fields propagated from OrderSpec
  2. Market-order reference price (no more price=0.0)
  3. CLOB response parsing hardening
  4. Two new invariant checks (11-12)
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.config import ExecutionConfig, StorageConfig
from src.execution.order_builder import (
    OrderSpec,
    build_exit_order,
    build_order,
)
from src.execution.order_router import OrderResult, OrderRouter, _parse_clob_response
from src.observability.invariant_checker import check_invariants
from src.policy.position_sizer import PositionSize
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import OrderRecord, PositionRecord, TradeRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**overrides) -> ExecutionConfig:
    defaults = dict(
        default_order_type="limit",
        slippage_tolerance=0.005,
        limit_order_ttl_secs=120,
        dry_run=True,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _make_position(direction="BUY_YES", **overrides) -> PositionSize:
    defaults = dict(
        stake_usd=50.0,
        kelly_fraction_used=0.05,
        full_kelly_stake=100.0,
        capped_by="kelly",
        direction=direction,
        token_quantity=100.0,
    )
    defaults.update(overrides)
    return PositionSize(**defaults)


def _make_order_spec(side="BUY", action_side="BUY", outcome_side="YES", **overrides):
    defaults = dict(
        order_id=str(uuid.uuid4()),
        market_id="mkt-test",
        token_id="tok-test",
        side=side,
        order_type="limit",
        price=0.55,
        size=100.0,
        stake_usd=55.0,
        ttl_secs=120,
        dry_run=False,
        action_side=action_side,
        outcome_side=outcome_side,
    )
    defaults.update(overrides)
    return OrderSpec(**defaults)


def _make_db() -> Database:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


def _insert_order(db, order_id, market_id, side="SELL", status="submitted",
                  action_side="", outcome_side=""):
    db.insert_order(OrderRecord(
        order_id=order_id,
        clob_order_id=f"clob-{order_id}",
        market_id=market_id,
        token_id=f"tok-{market_id}",
        side=side,
        order_type="limit",
        price=0.60,
        size=100.0,
        stake_usd=50.0,
        status=status,
        dry_run=False,
        action_side=action_side,
        outcome_side=outcome_side,
    ))


def _insert_position(db, market_id, **overrides):
    defaults = dict(
        market_id=market_id,
        token_id=f"tok-{market_id}",
        direction="BUY_YES",
        entry_price=0.50,
        size=100.0,
        stake_usd=50.0,
        current_price=0.55,
        pnl=5.0,
        action_side="BUY",
        outcome_side="YES",
        opened_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    db.upsert_position(PositionRecord(**defaults))


# ═══════════════════════════════════════════════════════════════════
# 1. OrderResult Canonical Fields
# ═══════════════════════════════════════════════════════════════════


class TestOrderResultCanonicalFields:
    def test_default_empty(self):
        result = OrderResult(order_id="test", status="filled")
        assert result.action_side == ""
        assert result.outcome_side == ""

    def test_fields_stored(self):
        result = OrderResult(
            order_id="test", status="filled",
            action_side="BUY", outcome_side="YES",
        )
        assert result.action_side == "BUY"
        assert result.outcome_side == "YES"

    def test_to_dict_includes_canonical(self):
        result = OrderResult(
            order_id="test", status="filled",
            action_side="SELL", outcome_side="NO",
        )
        d = result.to_dict()
        assert d["action_side"] == "SELL"
        assert d["outcome_side"] == "NO"

    @pytest.mark.asyncio
    async def test_dry_run_propagates_canonical(self):
        config = _make_config(dry_run=True)
        clob = MagicMock()
        router = OrderRouter(clob, config)

        order = _make_order_spec(action_side="BUY", outcome_side="YES", dry_run=True)
        result = await router.submit_order(order)

        assert result.status == "simulated"
        assert result.action_side == "BUY"
        assert result.outcome_side == "YES"

    def test_parse_clob_response_propagates_canonical(self):
        order = _make_order_spec(action_side="BUY", outcome_side="NO")
        resp = {"orderID": "clob-123", "status": "live", "success": True}

        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.action_side == "BUY"
        assert result.outcome_side == "NO"

    def test_parse_clob_failed_propagates_canonical(self):
        order = _make_order_spec(action_side="SELL", outcome_side="YES")
        resp = {"success": False, "errorMsg": "insufficient funds"}

        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.status == "failed"
        assert result.action_side == "SELL"
        assert result.outcome_side == "YES"

    def test_parse_clob_matched_propagates_canonical(self):
        order = _make_order_spec(
            action_side="BUY", outcome_side="YES",
            price=0.50, size=100.0,
        )
        resp = {
            "orderID": "clob-456",
            "status": "matched",
            "takingAmount": "50.0",
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.action_side == "BUY"
        assert result.outcome_side == "YES"


# ═══════════════════════════════════════════════════════════════════
# 2. Market Order Reference Price
# ═══════════════════════════════════════════════════════════════════


class TestMarketOrderPrice:
    def test_simple_order_market_has_reference_price(self):
        config = _make_config(default_order_type="market")
        pos = _make_position(direction="BUY_YES")
        orders = build_order("mkt-1", "tok-1", pos, 0.55, config, execution_strategy="simple")

        assert orders[0].price > 0
        assert orders[0].price == round(0.55, 4)

    def test_exit_order_market_has_reference_price(self):
        config = _make_config(default_order_type="market")
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config, outcome_side="YES")

        assert order.price > 0
        assert order.price == round(0.70, 4)

    def test_simple_order_market_zero_implied_uses_floor(self):
        config = _make_config(default_order_type="market")
        pos = _make_position(direction="BUY_YES")
        orders = build_order("mkt-1", "tok-1", pos, 0.0, config, execution_strategy="simple")

        assert orders[0].price == 0.01

    def test_exit_order_market_zero_price_uses_floor(self):
        config = _make_config(default_order_type="market")
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.0, config, outcome_side="YES")

        assert order.price == 0.01


# ═══════════════════════════════════════════════════════════════════
# 3. CLOB Response Parsing Hardening
# ═══════════════════════════════════════════════════════════════════


class TestCLOBResponseParsing:
    def test_matched_valid_taking_amount_happy_path(self):
        """Normal matched order with valid takingAmount derives correct fill."""
        order = _make_order_spec(price=0.50, size=100.0)
        # takingAmount=25.0 at price=0.50 → fill_size = 50 tokens
        resp = {
            "orderID": "clob-hp",
            "status": "matched",
            "takingAmount": "25.0",
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.fill_size == pytest.approx(50.0, abs=0.01)
        assert result.fill_price == 0.50

    def test_partial_fill_taking_amount(self):
        """takingAmount smaller than full order → partial fill preserved."""
        order = _make_order_spec(price=0.40, size=200.0)
        # takingAmount=20.0 at price=0.40 → fill_size = 50 tokens (partial)
        resp = {
            "orderID": "clob-pf",
            "status": "matched",
            "takingAmount": "20.0",
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.fill_size == pytest.approx(50.0, abs=0.01)
        assert result.fill_size < order.size  # partial, not full

    def test_fill_size_capped_at_order_size(self):
        """fill_size derived from takingAmount should not exceed order.size."""
        order = _make_order_spec(price=0.50, size=100.0)
        # takingAmount implies 200 tokens which is 2x order size
        resp = {
            "orderID": "clob-1",
            "status": "matched",
            "takingAmount": "100.0",  # 100.0 / 0.50 = 200 tokens
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        assert result.fill_size <= 100.0  # capped at order.size

    def test_zero_price_no_division_error(self):
        """order.price=0 should not cause ZeroDivisionError."""
        order = _make_order_spec(price=0.0, size=100.0)
        resp = {
            "orderID": "clob-2",
            "status": "matched",
            "takingAmount": "50.0",
            "success": True,
        }
        # Should not raise
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        # fill_size falls back to order.size for matched without fill
        assert result.status == "filled"

    def test_matched_without_taking_amount_assumes_full_fill(self):
        order = _make_order_spec(price=0.55, size=100.0)
        resp = {"orderID": "clob-3", "status": "matched", "success": True}
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")

        assert result.status == "filled"
        assert result.fill_size == 100.0
        assert result.fill_price == 0.55

    def test_non_dict_response_handled(self):
        order = _make_order_spec(price=0.55, size=100.0)
        resp = "unexpected string response"
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")

        assert result.status == "pending"
        assert result.fill_size == 0.0

    def test_invalid_taking_amount_handled(self):
        order = _make_order_spec(price=0.55, size=100.0)
        resp = {
            "orderID": "clob-4",
            "status": "matched",
            "takingAmount": "not_a_number",
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")
        # Falls back to full fill assumption
        assert result.status == "filled"
        assert result.fill_size == 100.0

    def test_live_status_no_fill(self):
        order = _make_order_spec(price=0.55, size=100.0)
        resp = {
            "orderID": "clob-5",
            "status": "live",
            "takingAmount": "0",
            "success": True,
        }
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")

        assert result.status == "submitted"
        assert result.fill_size == 0.0
        assert result.fill_price == 0.0

    def test_unknown_status_treated_as_pending(self):
        order = _make_order_spec(price=0.55, size=100.0)
        resp = {"orderID": "clob-6", "status": "something_new", "success": True}
        result = _parse_clob_response(resp, order, "2025-01-01T00:00:00Z")

        assert result.status == "pending"


# ═══════════════════════════════════════════════════════════════════
# 4. New Invariant Checks (11-12)
# ═══════════════════════════════════════════════════════════════════


class TestActiveOrderMissingCanonical:
    def test_active_order_missing_fields_detected(self):
        db = _make_db()
        _insert_order(db, "ord-missing", "mkt-miss",
                      side="BUY_YES", status="submitted",
                      action_side="", outcome_side="")

        violations = check_invariants(db)
        amc = [v for v in violations if v.check == "active_order_missing_canonical"]
        assert len(amc) == 1
        assert amc[0].severity == "warning"

    def test_active_order_with_fields_no_violation(self):
        db = _make_db()
        _insert_order(db, "ord-ok", "mkt-ok",
                      side="BUY_YES", status="submitted",
                      action_side="BUY", outcome_side="YES")

        violations = check_invariants(db)
        amc = [v for v in violations if v.check == "active_order_missing_canonical"]
        assert len(amc) == 0

    def test_terminal_order_missing_fields_no_violation(self):
        db = _make_db()
        _insert_order(db, "ord-filled", "mkt-done",
                      side="BUY_YES", status="filled",
                      action_side="", outcome_side="")

        violations = check_invariants(db)
        amc = [v for v in violations if v.check == "active_order_missing_canonical"]
        assert len(amc) == 0

    def test_partial_action_side_only_detected(self):
        """Missing outcome_side alone should trigger."""
        db = _make_db()
        _insert_order(db, "ord-partial", "mkt-partial",
                      side="BUY_YES", status="pending",
                      action_side="BUY", outcome_side="")

        violations = check_invariants(db)
        amc = [v for v in violations if v.check == "active_order_missing_canonical"]
        assert len(amc) == 1


class TestOrphanSellFallback:
    def test_orphan_sell_detected(self):
        db = _make_db()
        db._conn.execute(
            """INSERT INTO closed_positions
            (market_id, token_id, direction, entry_price, exit_price,
             size, stake_usd, pnl, close_reason, action_side, outcome_side)
            VALUES ('mkt-orphan', 'tok-orphan', 'BUY_YES', 0.50, 0.60,
                    100, 50, 10, 'SELL_FILLED_ORPHAN', 'SELL', 'YES')"""
        )
        db._conn.commit()

        violations = check_invariants(db)
        osf = [v for v in violations if v.check == "orphan_sell_fallback"]
        assert len(osf) == 1
        assert osf[0].severity == "warning"
        assert "mkt-orph" in osf[0].message

    def test_normal_close_no_violation(self):
        db = _make_db()
        db._conn.execute(
            """INSERT INTO closed_positions
            (market_id, token_id, direction, entry_price, exit_price,
             size, stake_usd, pnl, close_reason, action_side, outcome_side)
            VALUES ('mkt-normal', 'tok-normal', 'BUY_YES', 0.50, 0.60,
                    100, 50, 10, 'RESOLVED', 'SELL', 'YES')"""
        )
        db._conn.commit()

        violations = check_invariants(db)
        osf = [v for v in violations if v.check == "orphan_sell_fallback"]
        assert len(osf) == 0

    def test_no_closed_positions_no_violation(self):
        db = _make_db()
        violations = check_invariants(db)
        osf = [v for v in violations if v.check == "orphan_sell_fallback"]
        assert len(osf) == 0


# ═══════════════════════════════════════════════════════════════════
# Combined: All 12 checks on clean DB
# ═══════════════════════════════════════════════════════════════════


class TestAllChecksCleanDB:
    def test_no_violations_on_empty_db(self):
        db = _make_db()
        violations = check_invariants(db)
        assert violations == []

    def test_no_violations_on_consistent_data(self):
        db = _make_db()
        _insert_position(db, "mkt-1")
        # Add a trade with canonical fields
        db.insert_trade(TradeRecord(
            id=str(uuid.uuid4()),
            order_id="ord-1",
            market_id="mkt-1",
            token_id="tok-mkt-1",
            side="BUY_YES",
            price=0.50,
            size=100.0,
            stake_usd=50.0,
            status="FILLED",
            dry_run=True,
            action_side="BUY",
            outcome_side="YES",
        ))
        violations = check_invariants(db)
        assert violations == []
