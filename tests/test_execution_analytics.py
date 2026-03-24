"""Tests for Phase 6 Batch C: Execution analytics and dashboard."""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from src.execution.fill_tracker import (
    ExecutionQuality,
    FillRecord,
    FillTracker,
)


# ── DetailedExecutionQuality ─────────────────────────────────────


class TestDetailedExecutionQuality:
    def _make_tracker_with_fills(self) -> FillTracker:
        """Create a tracker with some fills."""
        tracker = FillTracker()
        fills = [
            ("o1", "m1", 0.50, 100, "simple", 0.51, 95),
            ("o2", "m2", 0.60, 50, "twap", 0.60, 50),
            ("o3", "m3", 0.45, 300, "iceberg", 0.47, 280),
            ("o4", "m4", 0.55, 80, "simple", 0.54, 80),
            ("o5", "m5", 0.70, 150, "twap", 0.72, 145),
        ]
        for oid, mid, ep, size, strat, fp, filled in fills:
            tracker.register_order(oid, mid, ep, size, strat)
            tracker.record_fill(oid, fp, filled)
        return tracker

    def test_slippage_distribution(self):
        """Slippage distribution should contain all values."""
        tracker = self._make_tracker_with_fills()
        quality = tracker.get_detailed_quality(lookback_hours=24)
        assert len(quality.slippage_distribution) == 5

    def test_worst_best_slippage(self):
        """Worst and best slippage should be correct."""
        tracker = self._make_tracker_with_fills()
        quality = tracker.get_detailed_quality(lookback_hours=24)
        assert quality.worst_slippage_bps >= quality.best_slippage_bps

    def test_size_buckets(self):
        """Fill rate by size bucket should have categories."""
        tracker = self._make_tracker_with_fills()
        quality = tracker.get_detailed_quality(lookback_hours=24)
        # Should have at least one bucket populated
        assert len(quality.fill_rate_by_size_bucket) > 0

    def test_realized_spread(self):
        """Realized vs expected spread should be computed."""
        tracker = self._make_tracker_with_fills()
        quality = tracker.get_detailed_quality(lookback_hours=24)
        # spread is average of (fill-expected)/expected
        assert isinstance(quality.realized_vs_expected_spread, float)

    def test_empty_tracker(self):
        """Empty tracker should return empty quality."""
        tracker = FillTracker()
        quality = tracker.get_detailed_quality(lookback_hours=24)
        assert quality.total_orders == 0
        assert quality.slippage_distribution == []
        assert quality.worst_slippage_bps == 0.0
        assert quality.best_slippage_bps == 0.0


# ── EnhancedFillTracker ──────────────────────────────────────────


class TestEnhancedFillTracker:
    def test_detailed_extends_basic(self):
        """get_detailed_quality should include basic quality fields."""
        tracker = FillTracker()
        tracker.register_order("o1", "m1", 0.50, 100, "simple")
        tracker.record_fill("o1", 0.52, 100)

        basic = tracker.get_quality()
        detailed = tracker.get_detailed_quality()

        # Detailed should have same base fields
        assert detailed.total_orders == basic.total_orders
        assert detailed.total_fills == basic.total_fills
        assert detailed.avg_slippage_bps == basic.avg_slippage_bps

        # Plus extra fields
        assert len(detailed.slippage_distribution) == 1

    def test_bucket_categorization(self):
        """Orders should be categorized into size buckets."""
        tracker = FillTracker()
        # Small order: size=10 * price=0.50 = $5
        tracker.register_order("o1", "m1", 0.50, 10, "simple")
        tracker.record_fill("o1", 0.51, 10)

        # Medium order: size=200 * price=0.50 = $100
        tracker.register_order("o2", "m2", 0.50, 200, "simple")
        tracker.record_fill("o2", 0.51, 200)

        # Large order: size=1000 * price=0.50 = $500
        tracker.register_order("o3", "m3", 0.50, 1000, "simple")
        tracker.record_fill("o3", 0.51, 1000)

        quality = tracker.get_detailed_quality()
        assert "small" in quality.fill_rate_by_size_bucket
        assert "medium" in quality.fill_rate_by_size_bucket
        assert "large" in quality.fill_rate_by_size_bucket

    def test_backward_compat(self):
        """get_quality() should still work without new fields populated."""
        tracker = FillTracker()
        tracker.register_order("o1", "m1", 0.50, 100, "simple")
        tracker.record_fill("o1", 0.52, 100)

        quality = tracker.get_quality()
        # New fields should be at default values
        assert quality.slippage_distribution == []
        assert quality.fill_rate_by_size_bucket == {}
        assert quality.worst_slippage_bps == 0.0


# ── Execution Quality Endpoint ───────────────────────────────────


class TestExecutionQualityEndpoint:
    @pytest.fixture
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def _create_execution_fills_table(self, conn: sqlite3.Connection):
        """Create execution_fills table and add test data."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS execution_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                market_id TEXT DEFAULT '',
                expected_price REAL DEFAULT 0,
                fill_price REAL DEFAULT 0,
                size_ordered REAL DEFAULT 0,
                size_filled REAL DEFAULT 0,
                is_partial INTEGER DEFAULT 0,
                slippage_bps REAL DEFAULT 0,
                time_to_fill_secs REAL DEFAULT 0,
                execution_strategy TEXT DEFAULT 'simple',
                strategy_selected_by TEXT DEFAULT 'manual',
                fill_rate REAL DEFAULT 1.0,
                fees_usd REAL DEFAULT 0,
                realized_spread_bps REAL DEFAULT 0,
                created_at TEXT
            )
        """)
        conn.commit()

    def test_no_table(self, client):
        """Should return zeros when no table exists."""
        with patch("src.dashboard.app._get_conn") as mock_conn:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            mock_conn.return_value = conn

            resp = client.get("/api/execution-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total_orders"] == 0
            assert data["total_fills"] == 0
            assert data["strategy_stats"] == {}
            assert data["slippage_distribution"] == []

    def test_empty_fills(self, client):
        """Should return zeros when table exists but is empty."""
        with patch("src.dashboard.app._get_conn") as mock_conn:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            self._create_execution_fills_table(conn)
            mock_conn.return_value = conn

            resp = client.get("/api/execution-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total_orders"] == 0

    def test_strategy_breakdown(self, client):
        """Should provide per-strategy stats."""
        with patch("src.dashboard.app._get_conn") as mock_conn:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            self._create_execution_fills_table(conn)

            # Insert test fills
            conn.execute("""
                INSERT INTO execution_fills
                (order_id, market_id, expected_price, fill_price,
                 size_ordered, size_filled, slippage_bps, time_to_fill_secs,
                 execution_strategy, fill_rate)
                VALUES ('o1', 'm1', 0.50, 0.51, 100, 100, 20.0, 1.5, 'simple', 1.0)
            """)
            conn.execute("""
                INSERT INTO execution_fills
                (order_id, market_id, expected_price, fill_price,
                 size_ordered, size_filled, slippage_bps, time_to_fill_secs,
                 execution_strategy, fill_rate)
                VALUES ('o2', 'm2', 0.60, 0.61, 200, 200, 16.7, 2.0, 'twap', 1.0)
            """)
            conn.commit()
            mock_conn.return_value = conn

            resp = client.get("/api/execution-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total_orders"] == 2
            assert data["total_fills"] == 2
            assert "simple" in data["strategy_stats"]
            assert "twap" in data["strategy_stats"]
            assert data["strategy_stats"]["simple"]["count"] == 1
            assert data["strategy_stats"]["twap"]["count"] == 1

    def test_slippage_distribution_endpoint(self, client):
        """Should return slippage distribution buckets."""
        with patch("src.dashboard.app._get_conn") as mock_conn:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            self._create_execution_fills_table(conn)

            # Insert fills with various slippages
            for i, slip in enumerate([-2.0, 3.0, 7.0, 15.0, 30.0, 60.0]):
                conn.execute(
                    """
                    INSERT INTO execution_fills
                    (order_id, market_id, expected_price, fill_price,
                     size_ordered, size_filled, slippage_bps, time_to_fill_secs,
                     execution_strategy, fill_rate)
                    VALUES (?, ?, 0.50, 0.51, 100, 100, ?, 1.0, 'simple', 1.0)
                    """,
                    (f"o{i}", f"m{i}", slip),
                )
            conn.commit()
            mock_conn.return_value = conn

            resp = client.get("/api/execution-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)

            dist = data["slippage_distribution"]
            assert len(dist) == 6  # 6 buckets
            # Each bucket should have a label and count
            for bucket in dist:
                assert "label" in bucket
                assert "count" in bucket
            # Verify the total matches
            total_in_buckets = sum(b["count"] for b in dist)
            assert total_in_buckets == 6

    def test_median_time_to_fill(self, client):
        """Should compute median time to fill."""
        with patch("src.dashboard.app._get_conn") as mock_conn:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            self._create_execution_fills_table(conn)

            for i, ttf in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
                conn.execute(
                    """
                    INSERT INTO execution_fills
                    (order_id, market_id, expected_price, fill_price,
                     size_ordered, size_filled, slippage_bps, time_to_fill_secs,
                     execution_strategy, fill_rate)
                    VALUES (?, ?, 0.50, 0.51, 100, 100, 10.0, ?, 'simple', 1.0)
                    """,
                    (f"o{i}", f"m{i}", ttf),
                )
            conn.commit()
            mock_conn.return_value = conn

            resp = client.get("/api/execution-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["median_time_to_fill"] == 3.0  # median of [1,2,3,4,5]


# ── Migration 12 ─────────────────────────────────────────────────


class TestMigration12:
    def test_execution_fills_table_created(self):
        """Migration 12 should create the execution_fills table."""
        from src.storage.migrations import run_migrations

        conn = sqlite3.connect(":memory:")
        run_migrations(conn)

        # Table should exist
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_fills'"
        ).fetchone()
        assert row is not None

        # Indexes should exist
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_exec_fills%'"
        ).fetchall()
        assert len(indexes) == 3
        conn.close()

    def test_schema_version_12(self):
        """Schema version should be 12 after migrations."""
        from src.storage.migrations import SCHEMA_VERSION

        assert SCHEMA_VERSION >= 12


# ── Slippage Distribution Helper ─────────────────────────────────


class TestSlippageDistribution:
    def test_empty_slippages(self):
        from src.dashboard.app import _compute_slippage_distribution

        result = _compute_slippage_distribution([])
        assert result == []

    def test_all_buckets_populated(self):
        from src.dashboard.app import _compute_slippage_distribution

        slippages = [-5.0, 2.0, 7.0, 15.0, 30.0, 60.0]
        result = _compute_slippage_distribution(slippages)
        assert len(result) == 6
        for bucket in result:
            assert bucket["count"] == 1

    def test_single_bucket_concentration(self):
        from src.dashboard.app import _compute_slippage_distribution

        slippages = [1.0, 2.0, 3.0, 4.0]  # all in 0-5 bucket
        result = _compute_slippage_distribution(slippages)
        bucket_0_5 = next(b for b in result if b["label"] == "0-5 bps")
        assert bucket_0_5["count"] == 4

    def test_negative_slippage_improvement(self):
        from src.dashboard.app import _compute_slippage_distribution

        slippages = [-10.0, -5.0, -1.0]
        result = _compute_slippage_distribution(slippages)
        improvement = next(b for b in result if "improvement" in b["label"])
        assert improvement["count"] == 3
