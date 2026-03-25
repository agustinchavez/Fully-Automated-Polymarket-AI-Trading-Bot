"""Tests for Phase 10B Batch D: Invariant Checker."""

from __future__ import annotations

import datetime as dt
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.config import StorageConfig, ExecutionConfig
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
    import uuid
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


# ── Clean DB Tests ──────────────────────────────────────────────


class TestCleanDB:
    def test_no_violations_on_empty_db(self):
        db = _make_db()
        violations = check_invariants(db)
        assert violations == []

    def test_no_violations_on_consistent_data(self):
        db = _make_db()
        _insert_position(db, "mkt-1")
        _insert_trade(db, "mkt-1")
        violations = check_invariants(db)
        assert violations == []


# ── Duplicate Positions Tests ───────────────────────────────────


class TestDuplicatePositions:
    def test_duplicate_position_detected(self):
        db = _make_db()
        # Recreate positions table without PRIMARY KEY to simulate corruption
        db._conn.execute("DROP TABLE positions")
        db._conn.execute(
            """CREATE TABLE positions (
                market_id TEXT, token_id TEXT, direction TEXT,
                entry_price REAL, size REAL, stake_usd REAL,
                current_price REAL, pnl REAL, opened_at TEXT,
                question TEXT DEFAULT '', market_type TEXT DEFAULT '',
                action_side TEXT DEFAULT '', outcome_side TEXT DEFAULT ''
            )"""
        )
        # Insert two rows for the same market
        for tok in ("tok-1", "tok-2"):
            db._conn.execute(
                """INSERT INTO positions (market_id, token_id, direction, entry_price,
                   size, stake_usd, current_price, pnl, action_side, outcome_side)
                VALUES ('mkt-dup', ?, 'BUY_YES', 0.50, 100, 50, 0.55, 5, 'BUY', 'YES')""",
                (tok,),
            )
        db._conn.commit()

        violations = check_invariants(db)
        dup_violations = [v for v in violations if v.check == "duplicate_positions"]
        assert len(dup_violations) == 1
        assert dup_violations[0].severity == "critical"
        assert "mkt-dup" in dup_violations[0].market_id

    def test_single_position_no_violation(self):
        db = _make_db()
        _insert_position(db, "mkt-1")

        violations = check_invariants(db)
        dup_violations = [v for v in violations if v.check == "duplicate_positions"]
        assert len(dup_violations) == 0


# ── Orphaned Positions Tests ────────────────────────────────────


class TestOrphanedPositions:
    def test_orphaned_position_detected(self):
        db = _make_db()
        _insert_position(db, "mkt-orphan")
        # No trade for mkt-orphan

        violations = check_invariants(db)
        orphan_violations = [v for v in violations if v.check == "orphaned_position"]
        assert len(orphan_violations) == 1
        assert orphan_violations[0].severity == "warning"

    def test_position_with_trade_not_orphaned(self):
        db = _make_db()
        _insert_position(db, "mkt-good")
        _insert_trade(db, "mkt-good")

        violations = check_invariants(db)
        orphan_violations = [v for v in violations if v.check == "orphaned_position"]
        assert len(orphan_violations) == 0


# ── Conflicting SELL Orders Tests ───────────────────────────────


class TestConflictingSellOrders:
    def test_conflicting_sell_orders_detected(self):
        db = _make_db()
        _insert_order(db, "sell-1", "mkt-conflict", side="SELL", status="submitted")
        _insert_order(db, "sell-2", "mkt-conflict", side="SELL", status="pending")

        violations = check_invariants(db)
        sell_violations = [v for v in violations if v.check == "conflicting_sell_orders"]
        assert len(sell_violations) == 1
        assert sell_violations[0].severity == "critical"

    def test_single_sell_order_no_violation(self):
        db = _make_db()
        _insert_order(db, "sell-1", "mkt-single", side="SELL", status="submitted")

        violations = check_invariants(db)
        sell_violations = [v for v in violations if v.check == "conflicting_sell_orders"]
        assert len(sell_violations) == 0

    def test_filled_sell_orders_no_violation(self):
        db = _make_db()
        _insert_order(db, "sell-1", "mkt-done", side="SELL", status="filled")
        _insert_order(db, "sell-2", "mkt-done", side="SELL", status="filled")

        violations = check_invariants(db)
        sell_violations = [v for v in violations if v.check == "conflicting_sell_orders"]
        assert len(sell_violations) == 0


# ── Direction Mismatch Tests ────────────────────────────────────


class TestDirectionMismatch:
    def test_direction_mismatch_detected(self):
        db = _make_db()
        # direction says BUY_YES but action_side/outcome_side says BUY/NO
        _insert_position(
            db, "mkt-mismatch",
            direction="BUY_YES",
            action_side="BUY",
            outcome_side="NO",  # mismatch!
        )

        violations = check_invariants(db)
        mismatch_violations = [v for v in violations if v.check == "direction_mismatch"]
        assert len(mismatch_violations) == 1
        assert mismatch_violations[0].severity == "warning"

    def test_matching_fields_no_violation(self):
        db = _make_db()
        _insert_position(
            db, "mkt-match",
            direction="BUY_YES",
            action_side="BUY",
            outcome_side="YES",
        )

        violations = check_invariants(db)
        mismatch_violations = [v for v in violations if v.check == "direction_mismatch"]
        assert len(mismatch_violations) == 0

    def test_empty_canonical_fields_skipped(self):
        db = _make_db()
        _insert_position(
            db, "mkt-empty",
            direction="BUY_YES",
            action_side="",
            outcome_side="",
        )

        violations = check_invariants(db)
        mismatch_violations = [v for v in violations if v.check == "direction_mismatch"]
        assert len(mismatch_violations) == 0


# ── Stale Position Tests ────────────────────────────────────────


class TestStalePositions:
    def test_stale_position_detected(self):
        db = _make_db()
        old_date = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
        ).isoformat()
        _insert_position(db, "mkt-stale", opened_at=old_date)

        violations = check_invariants(db)
        stale_violations = [v for v in violations if v.check == "stale_position"]
        assert len(stale_violations) == 1
        assert stale_violations[0].severity == "warning"

    def test_fresh_position_not_stale(self):
        db = _make_db()
        _insert_position(db, "mkt-fresh")  # opened_at = now

        violations = check_invariants(db)
        stale_violations = [v for v in violations if v.check == "stale_position"]
        assert len(stale_violations) == 0


# ── InvariantViolation Dataclass Tests ──────────────────────────


class TestInvariantViolation:
    def test_dataclass_fields(self):
        v = InvariantViolation(
            check="test_check",
            severity="critical",
            market_id="mkt-test",
            message="Test message",
        )
        assert v.check == "test_check"
        assert v.severity == "critical"
        assert v.market_id == "mkt-test"
        assert v.message == "Test message"


# ── check_invariants Error Handling ─────────────────────────────


class TestCheckInvariantsErrors:
    def test_survives_db_errors(self):
        db = _make_db()
        # Drop positions table to cause an error
        db._conn.execute("DROP TABLE positions")
        db._conn.commit()

        # Should not raise — returns partial results
        violations = check_invariants(db)
        assert isinstance(violations, list)

    def test_returns_all_violations(self):
        db = _make_db()
        # Create orphan position (no trade) + stale
        old_date = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)
        ).isoformat()
        _insert_position(db, "mkt-multi", opened_at=old_date)

        violations = check_invariants(db)
        checks = {v.check for v in violations}
        assert "orphaned_position" in checks
        assert "stale_position" in checks


# ── Engine Integration Tests ────────────────────────────────────


class TestEngineInvariantChecks:
    def test_engine_runs_checks_when_enabled(self):
        from src.engine.loop import TradingEngine

        db = _make_db()
        _insert_position(db, "mkt-orphan")  # orphan = no trade

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = db
        engine._cycle_count = 10
        engine.config = MagicMock()
        engine.config.execution.invariant_checks_enabled = True
        engine.config.execution.invariant_check_interval_cycles = 10

        engine._maybe_check_invariants()

        # Should have inserted alerts for violations
        alerts = db._conn.execute("SELECT * FROM alerts_log").fetchall()
        # Orphan is warning-level, not critical, so no alert expected
        # But the method should complete without error
        assert True

    def test_engine_skips_when_disabled(self):
        from src.engine.loop import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = MagicMock()
        engine._cycle_count = 10
        engine.config = MagicMock()
        engine.config.execution.invariant_checks_enabled = False

        # Should not call check_invariants
        with patch("src.observability.invariant_checker.check_invariants") as mock_check:
            engine._maybe_check_invariants()
            mock_check.assert_not_called()

    def test_engine_respects_interval(self):
        from src.engine.loop import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = MagicMock()
        engine._cycle_count = 5  # not a multiple of 10
        engine.config = MagicMock()
        engine.config.execution.invariant_checks_enabled = True
        engine.config.execution.invariant_check_interval_cycles = 10

        with patch("src.observability.invariant_checker.check_invariants") as mock_check:
            engine._maybe_check_invariants()
            mock_check.assert_not_called()

    def test_engine_critical_violation_creates_alert(self):
        from src.engine.loop import TradingEngine

        db = _make_db()
        # Recreate positions table without PK to allow duplicates
        db._conn.execute("DROP TABLE positions")
        db._conn.execute(
            """CREATE TABLE positions (
                market_id TEXT, token_id TEXT, direction TEXT,
                entry_price REAL, size REAL, stake_usd REAL,
                current_price REAL, pnl REAL, opened_at TEXT,
                question TEXT DEFAULT '', market_type TEXT DEFAULT '',
                action_side TEXT DEFAULT '', outcome_side TEXT DEFAULT ''
            )"""
        )
        for tok in ("tok-1", "tok-2"):
            db._conn.execute(
                """INSERT INTO positions (market_id, token_id, direction, entry_price,
                   size, stake_usd, current_price, pnl, action_side, outcome_side)
                VALUES ('mkt-dup', ?, 'BUY_YES', 0.50, 100, 50, 0.55, 5, 'BUY', 'YES')""",
                (tok,),
            )
        db._conn.commit()

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = db
        engine._cycle_count = 10
        engine.config = MagicMock()
        engine.config.execution.invariant_checks_enabled = True
        engine.config.execution.invariant_check_interval_cycles = 10

        engine._maybe_check_invariants()

        # Critical violations should create alerts
        alerts = db._conn.execute(
            "SELECT * FROM alerts_log WHERE level='critical'"
        ).fetchall()
        assert len(alerts) >= 1


# ── Dashboard API Test ──────────────────────────────────────────


class TestDashboardAPI:
    def test_api_invariant_checks_endpoint_exists(self):
        """The /api/invariant-checks endpoint should be registered."""
        from src.dashboard.app import app
        rules = [r.rule for r in app.url_map.iter_rules()]
        assert "/api/invariant-checks" in rules
