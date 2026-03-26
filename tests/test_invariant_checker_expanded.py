"""Tests for Phase 10C Batch C: Expanded invariant checks (checks 6-10)."""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid

import pytest

from src.config import StorageConfig
from src.observability.invariant_checker import (
    InvariantViolation,
    check_invariants,
)
from src.storage.database import Database
from src.storage.migrations import run_migrations
from src.storage.models import OrderRecord, PositionRecord, TradeRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_db() -> Database:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


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


def _insert_trade(db, market_id, **overrides):
    defaults = dict(
        id=str(uuid.uuid4()),
        order_id=f"ord-{market_id}",
        market_id=market_id,
        token_id=f"tok-{market_id}",
        side="BUY_YES",
        price=0.50,
        size=100.0,
        stake_usd=50.0,
        status="FILLED",
        dry_run=True,
        action_side="BUY",
        outcome_side="YES",
    )
    defaults.update(overrides)
    db.insert_trade(TradeRecord(**defaults))


def _insert_order(db, order_id, market_id, side="SELL", status="submitted"):
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
    ))


def _insert_closed_position(db, market_id):
    db._conn.execute(
        """INSERT INTO closed_positions
        (market_id, token_id, direction, entry_price, exit_price,
         size, stake_usd, pnl, close_reason, action_side, outcome_side)
        VALUES (?, ?, 'BUY_YES', 0.50, 0.70, 100, 50, 20, 'RESOLVED', 'BUY', 'YES')""",
        (market_id, f"tok-{market_id}"),
    )
    db._conn.commit()


# ── SELL Without Position Tests ──────────────────────────────────


class TestSellWithoutPosition:
    def test_sell_order_no_position_detected(self):
        db = _make_db()
        _insert_order(db, "sell-orphan", "mkt-no-pos", side="SELL", status="submitted")

        violations = check_invariants(db)
        swp = [v for v in violations if v.check == "sell_without_position"]
        assert len(swp) == 1
        assert swp[0].severity == "critical"
        assert "mkt-no-p" in swp[0].message

    def test_sell_order_with_position_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-has-pos")
        _insert_trade(db, "mkt-has-pos")
        _insert_order(db, "sell-ok", "mkt-has-pos", side="SELL", status="submitted")

        violations = check_invariants(db)
        swp = [v for v in violations if v.check == "sell_without_position"]
        assert len(swp) == 0

    def test_filled_sell_order_no_violation(self):
        db = _make_db()
        _insert_order(db, "sell-done", "mkt-no-pos", side="SELL", status="filled")

        violations = check_invariants(db)
        swp = [v for v in violations if v.check == "sell_without_position"]
        assert len(swp) == 0


# ── Multiple Entry Orders Tests ──────────────────────────────────


class TestMultipleEntryOrders:
    def test_two_buy_orders_same_market_detected(self):
        db = _make_db()
        _insert_order(db, "buy-1", "mkt-dup", side="BUY_YES", status="submitted")
        _insert_order(db, "buy-2", "mkt-dup", side="BUY_YES", status="pending")

        violations = check_invariants(db)
        meo = [v for v in violations if v.check == "multiple_entry_orders"]
        assert len(meo) == 1
        assert meo[0].severity == "warning"

    def test_single_buy_order_no_violation(self):
        db = _make_db()
        _insert_order(db, "buy-1", "mkt-single", side="BUY_YES", status="submitted")

        violations = check_invariants(db)
        meo = [v for v in violations if v.check == "multiple_entry_orders"]
        assert len(meo) == 0

    def test_two_buy_orders_different_markets_no_violation(self):
        db = _make_db()
        _insert_order(db, "buy-1", "mkt-a", side="BUY_YES", status="submitted")
        _insert_order(db, "buy-2", "mkt-b", side="BUY_YES", status="submitted")

        violations = check_invariants(db)
        meo = [v for v in violations if v.check == "multiple_entry_orders"]
        assert len(meo) == 0

    def test_sell_orders_not_counted(self):
        db = _make_db()
        _insert_order(db, "sell-1", "mkt-x", side="SELL", status="submitted")
        _insert_order(db, "sell-2", "mkt-x", side="SELL", status="submitted")

        violations = check_invariants(db)
        meo = [v for v in violations if v.check == "multiple_entry_orders"]
        assert len(meo) == 0


# ── Missing Canonical Fields Tests ───────────────────────────────


class TestMissingCanonicalFields:
    def test_filled_trade_empty_action_side_detected(self):
        db = _make_db()
        _insert_position(db, "mkt-canon")
        _insert_trade(db, "mkt-canon", action_side="", outcome_side="")

        violations = check_invariants(db)
        mcf = [v for v in violations if v.check == "missing_canonical_fields"]
        assert len(mcf) == 1
        assert mcf[0].severity == "warning"

    def test_filled_trade_with_fields_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-ok")
        _insert_trade(db, "mkt-ok", action_side="BUY", outcome_side="YES")

        violations = check_invariants(db)
        mcf = [v for v in violations if v.check == "missing_canonical_fields"]
        assert len(mcf) == 0

    def test_non_filled_trade_empty_fields_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-pending")
        _insert_trade(
            db, "mkt-pending",
            status="SUBMITTED", action_side="", outcome_side="",
        )

        violations = check_invariants(db)
        mcf = [v for v in violations if v.check == "missing_canonical_fields"]
        assert len(mcf) == 0


# ── Filled Trade No Position Tests ───────────────────────────────


class TestFilledTradeNoPosition:
    def test_buy_trade_no_position_detected(self):
        db = _make_db()
        _insert_trade(db, "mkt-lost", side="BUY_YES")
        # No position for mkt-lost

        violations = check_invariants(db)
        ftnp = [v for v in violations if v.check == "filled_trade_no_position"]
        assert len(ftnp) == 1
        assert ftnp[0].severity == "warning"

    def test_trade_with_position_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-has")
        _insert_trade(db, "mkt-has", side="BUY_YES")

        violations = check_invariants(db)
        ftnp = [v for v in violations if v.check == "filled_trade_no_position"]
        assert len(ftnp) == 0

    def test_trade_with_closed_position_no_violation(self):
        db = _make_db()
        _insert_trade(db, "mkt-closed", side="BUY_YES")
        _insert_closed_position(db, "mkt-closed")

        violations = check_invariants(db)
        ftnp = [v for v in violations if v.check == "filled_trade_no_position"]
        assert len(ftnp) == 0

    def test_sell_trade_no_position_no_violation(self):
        db = _make_db()
        _insert_trade(db, "mkt-sell", side="SELL")
        # No position — but it's a SELL trade, so that's expected

        violations = check_invariants(db)
        ftnp = [v for v in violations if v.check == "filled_trade_no_position"]
        assert len(ftnp) == 0


# ── Empty Outcome Position Tests ─────────────────────────────────


class TestEmptyOutcomePosition:
    def test_empty_outcome_side_detected(self):
        db = _make_db()
        _insert_position(db, "mkt-empty-os", outcome_side="")
        _insert_trade(db, "mkt-empty-os")

        violations = check_invariants(db)
        eop = [v for v in violations if v.check == "empty_outcome_position"]
        assert len(eop) == 1
        assert eop[0].severity == "warning"

    def test_populated_outcome_side_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-ok", outcome_side="YES")
        _insert_trade(db, "mkt-ok")

        violations = check_invariants(db)
        eop = [v for v in violations if v.check == "empty_outcome_position"]
        assert len(eop) == 0


# ── Combined / Error Resilience ──────────────────────────────────


class TestCombinedChecks:
    def test_clean_db_no_violations(self):
        db = _make_db()
        violations = check_invariants(db)
        assert violations == []

    def test_consistent_data_no_violations(self):
        db = _make_db()
        _insert_position(db, "mkt-1", outcome_side="YES")
        _insert_trade(db, "mkt-1", action_side="BUY", outcome_side="YES")
        violations = check_invariants(db)
        assert violations == []

    def test_individual_check_failure_doesnt_break_others(self):
        """If one check's table is missing, others still run."""
        db = _make_db()
        # Drop trades table to break some checks
        db._conn.execute("DROP TABLE trades")
        db._conn.commit()

        # Insert data that would trigger violations in other checks
        _insert_position(db, "mkt-stale-pos", outcome_side="",
                        opened_at=(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat())

        # Should not raise — returns partial results
        violations = check_invariants(db)
        assert isinstance(violations, list)
        # Should at least find the stale position and empty outcome
        checks = {v.check for v in violations}
        assert "stale_position" in checks or "empty_outcome_position" in checks
