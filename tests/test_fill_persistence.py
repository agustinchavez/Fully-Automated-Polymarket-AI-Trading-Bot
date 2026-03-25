"""Tests for Phase 10 Batch D: Fill persistence to execution_fills table."""

from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pytest

from src.config import ExecutionConfig
from src.execution.fill_tracker import FillTracker, FillRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_db():
    from src.storage.migrations import run_migrations
    from src.storage.database import Database
    from src.config import StorageConfig
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


# ── FillTracker DB Integration Tests ────────────────────────────


class TestFillPersistence:
    def test_fill_persisted_to_db(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        tracker.register_order("ord-1", "market-a", 0.50, 100.0, "simple")
        tracker.record_fill("ord-1", 0.51, 100.0)

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-1"
        assert rows[0]["market_id"] == "market-a"
        assert rows[0]["expected_price"] == 0.50
        assert rows[0]["fill_price"] == 0.51
        assert rows[0]["size_filled"] == 100.0
        assert rows[0]["execution_strategy"] == "simple"

    def test_fill_without_db_works(self):
        """FillTracker without DB should still work in-memory."""
        tracker = FillTracker()  # no db

        tracker.register_order("ord-2", "market-b", 0.60, 50.0, "twap")
        record = tracker.record_fill("ord-2", 0.61, 50.0)

        assert record is not None
        assert record.order_id == "ord-2"
        assert len(tracker._fills) == 1

    def test_unfilled_record_persisted(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        tracker.register_order("ord-3", "market-c", 0.40, 200.0, "iceberg")
        tracker.record_unfilled("ord-3")

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "ord-3"
        assert rows[0]["fill_price"] == 0.0
        assert rows[0]["size_filled"] == 0.0

    def test_all_fill_record_fields_mapped(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        tracker.register_order("ord-fields", "market-f", 0.55, 80.0, "adaptive")
        tracker.record_fill("ord-fields", 0.56, 80.0)

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        row = dict(rows[0])

        assert row["order_id"] == "ord-fields"
        assert row["market_id"] == "market-f"
        assert row["expected_price"] == 0.55
        assert row["fill_price"] == 0.56
        assert row["size_ordered"] == 80.0
        assert row["size_filled"] == 80.0
        assert row["is_partial"] == 0  # not partial
        assert row["slippage_bps"] > 0  # positive slippage
        assert row["time_to_fill_secs"] >= 0
        assert row["execution_strategy"] == "adaptive"
        assert row["fill_rate"] == 1.0
        assert row["created_at"] != ""

    def test_multiple_fills_persisted(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        for i in range(5):
            oid = f"ord-multi-{i}"
            tracker.register_order(oid, f"market-{i}", 0.50, 100.0, "simple")
            tracker.record_fill(oid, 0.50 + i * 0.01, 100.0)

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        assert len(rows) == 5

    def test_partial_fill_is_partial_flag(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        tracker.register_order("ord-partial", "market-p", 0.50, 100.0, "simple")
        # Only 50 out of 100 filled = partial
        tracker.record_fill("ord-partial", 0.50, 50.0)

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        assert rows[0]["is_partial"] == 1


# ── Load from DB Tests ──────────────────────────────────────────


class TestLoadFromDB:
    def test_load_empty_db(self):
        db = _make_db()
        tracker = FillTracker(db=db)

        loaded = tracker.load_from_db(lookback_hours=24)

        assert loaded == 0
        assert len(tracker._fills) == 0

    def test_load_recent_fills(self):
        db = _make_db()

        # Insert fills directly into DB (simulating previous session)
        now = dt.datetime.now(dt.timezone.utc)
        for i in range(3):
            db.insert_execution_fill({
                "order_id": f"old-{i}",
                "market_id": f"market-old-{i}",
                "expected_price": 0.50,
                "fill_price": 0.51,
                "size_ordered": 100.0,
                "size_filled": 100.0,
                "is_partial": False,
                "slippage_bps": 20.0,
                "time_to_fill_secs": 1.5,
                "execution_strategy": "simple",
                "fill_rate": 1.0,
                "created_at": (now - dt.timedelta(hours=1)).isoformat(),
            })

        tracker = FillTracker(db=db)
        loaded = tracker.load_from_db(lookback_hours=24)

        assert loaded == 3
        assert len(tracker._fills) == 3
        assert tracker._fills[0].order_id == "old-0"
        assert tracker._fills[0].fill_price == 0.51

    def test_load_respects_lookback(self):
        db = _make_db()

        now = dt.datetime.now(dt.timezone.utc)

        # Recent fill (1 hour ago)
        db.insert_execution_fill({
            "order_id": "recent",
            "market_id": "market-r",
            "expected_price": 0.50,
            "fill_price": 0.51,
            "size_ordered": 100.0,
            "size_filled": 100.0,
            "slippage_bps": 20.0,
            "time_to_fill_secs": 1.0,
            "execution_strategy": "simple",
            "fill_rate": 1.0,
            "created_at": (now - dt.timedelta(hours=1)).isoformat(),
        })

        # Old fill (48 hours ago)
        db.insert_execution_fill({
            "order_id": "old",
            "market_id": "market-o",
            "expected_price": 0.50,
            "fill_price": 0.52,
            "size_ordered": 100.0,
            "size_filled": 100.0,
            "slippage_bps": 40.0,
            "time_to_fill_secs": 2.0,
            "execution_strategy": "twap",
            "fill_rate": 1.0,
            "created_at": (now - dt.timedelta(hours=48)).isoformat(),
        })

        tracker = FillTracker(db=db)
        loaded = tracker.load_from_db(lookback_hours=24)

        assert loaded == 1
        assert tracker._fills[0].order_id == "recent"

    def test_load_idempotent(self):
        db = _make_db()

        now = dt.datetime.now(dt.timezone.utc)
        db.insert_execution_fill({
            "order_id": "idem-1",
            "market_id": "market-i",
            "expected_price": 0.50,
            "fill_price": 0.50,
            "size_ordered": 100.0,
            "size_filled": 100.0,
            "slippage_bps": 0.0,
            "time_to_fill_secs": 0.5,
            "execution_strategy": "simple",
            "fill_rate": 1.0,
            "created_at": now.isoformat(),
        })

        tracker = FillTracker(db=db)

        loaded1 = tracker.load_from_db()
        loaded2 = tracker.load_from_db()  # second call

        assert loaded1 == 1
        assert loaded2 == 0  # no duplicates
        assert len(tracker._fills) == 1

    def test_loaded_fills_usable_in_get_quality(self):
        db = _make_db()

        now = dt.datetime.now(dt.timezone.utc)
        for i in range(5):
            db.insert_execution_fill({
                "order_id": f"quality-{i}",
                "market_id": f"market-q-{i}",
                "expected_price": 0.50,
                "fill_price": 0.51,
                "size_ordered": 100.0,
                "size_filled": 100.0,
                "slippage_bps": 20.0,
                "time_to_fill_secs": 1.0,
                "execution_strategy": "simple",
                "fill_rate": 1.0,
                "created_at": now.isoformat(),
            })

        tracker = FillTracker(db=db)
        tracker.load_from_db()

        quality = tracker.get_quality(lookback_hours=24)

        assert quality.total_orders == 5
        assert quality.total_fills == 5
        assert quality.avg_slippage_bps == 20.0

    def test_load_without_db_returns_zero(self):
        tracker = FillTracker()  # no DB

        loaded = tracker.load_from_db()

        assert loaded == 0


# ── DB Insert/Query Methods Tests ──────────────────────────────


class TestDBMethods:
    def test_insert_execution_fill(self):
        db = _make_db()

        db.insert_execution_fill({
            "order_id": "db-1",
            "market_id": "market-db",
            "expected_price": 0.55,
            "fill_price": 0.56,
            "size_ordered": 200.0,
            "size_filled": 200.0,
            "is_partial": False,
            "slippage_bps": 18.2,
            "time_to_fill_secs": 0.8,
            "execution_strategy": "twap",
            "fill_rate": 1.0,
            "created_at": "2024-01-01T00:00:00Z",
        })

        rows = db._conn.execute("SELECT * FROM execution_fills").fetchall()
        assert len(rows) == 1
        assert rows[0]["order_id"] == "db-1"

    def test_get_execution_fills_limit(self):
        db = _make_db()

        for i in range(10):
            db.insert_execution_fill({
                "order_id": f"fill-{i}",
                "market_id": f"market-{i}",
                "created_at": f"2024-01-{i+1:02d}T00:00:00Z",
            })

        result = db.get_execution_fills(limit=5)
        assert len(result) == 5

    def test_get_execution_fills_since(self):
        db = _make_db()

        db.insert_execution_fill({
            "order_id": "old",
            "created_at": "2024-01-01T00:00:00Z",
        })
        db.insert_execution_fill({
            "order_id": "new",
            "created_at": "2024-06-01T00:00:00Z",
        })

        result = db.get_execution_fills_since("2024-03-01T00:00:00Z")
        assert len(result) == 1
        assert result[0]["order_id"] == "new"

    def test_get_execution_fills_empty(self):
        db = _make_db()

        result = db.get_execution_fills()
        assert result == []

    def test_get_execution_fills_since_empty(self):
        db = _make_db()

        result = db.get_execution_fills_since("2024-01-01T00:00:00Z")
        assert result == []


# ── Slippage Recomputation on Load Tests ────────────────────────


class TestSlippageRecompute:
    def test_slippage_recomputed_on_load(self):
        db = _make_db()

        now = dt.datetime.now(dt.timezone.utc)
        db.insert_execution_fill({
            "order_id": "slippage-test",
            "market_id": "market-s",
            "expected_price": 0.50,
            "fill_price": 0.52,
            "size_ordered": 100.0,
            "size_filled": 100.0,
            "slippage_bps": 40.0,
            "time_to_fill_secs": 1.0,
            "execution_strategy": "simple",
            "fill_rate": 1.0,
            "created_at": now.isoformat(),
        })

        tracker = FillTracker(db=db)
        tracker.load_from_db()

        assert len(tracker._fills) == 1
        fill = tracker._fills[0]
        assert fill.slippage == pytest.approx(0.02, abs=0.001)  # 0.52 - 0.50
        assert fill.slippage_bps == 40.0  # from DB
