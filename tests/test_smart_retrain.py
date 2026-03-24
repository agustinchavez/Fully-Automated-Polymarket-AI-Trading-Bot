"""Tests for Phase 8 Batch C: Smart calibration retraining + A/B testing."""

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import patch

import pytest

from src.analytics.smart_retrain import (
    ABTestResult,
    RetrainTrigger,
    SmartRetrainManager,
)


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """In-memory DB with required tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # calibration_history (migration 2)
    conn.execute("""
        CREATE TABLE calibration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_prob REAL NOT NULL,
            actual_outcome REAL NOT NULL,
            recorded_at REAL NOT NULL,
            market_id TEXT
        )
    """)

    # engine_state (migration 3)
    conn.execute("""
        CREATE TABLE engine_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at REAL
        )
    """)

    # calibration_ab_results (migration 14)
    conn.execute("""
        CREATE TABLE calibration_ab_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_id TEXT UNIQUE NOT NULL,
            calibrated_brier REAL DEFAULT 0,
            uncalibrated_brier REAL DEFAULT 0,
            calibrated_count INTEGER DEFAULT 0,
            uncalibrated_count INTEGER DEFAULT 0,
            calibration_helps INTEGER DEFAULT 1,
            delta_brier REAL DEFAULT 0,
            trigger_reason TEXT DEFAULT '',
            started_at TEXT,
            completed_at TEXT
        )
    """)
    conn.commit()
    return conn


def _insert_calibration_rows(
    conn: sqlite3.Connection,
    n: int = 40,
    forecast_prob: float = 0.7,
    actual_outcome: float = 1.0,
    base_time: float | None = None,
) -> None:
    """Insert calibration history rows."""
    base = base_time or time.time()
    for i in range(n):
        conn.execute("""
            INSERT INTO calibration_history
                (forecast_prob, actual_outcome, recorded_at, market_id)
            VALUES (?, ?, ?, ?)
        """, (forecast_prob, actual_outcome, base - (n - i) * 3600, f"m{i}"))
    conn.commit()


# ── RetrainTrigger ───────────────────────────────────────────────


class TestRetrainTrigger:
    def test_defaults(self):
        t = RetrainTrigger()
        assert t.should_retrain is False
        assert t.reason == "none"
        assert t.details == {}

    def test_should_retrain(self):
        t = RetrainTrigger(should_retrain=True, reason="resolution_count")
        assert t.should_retrain is True

    def test_reason_values(self):
        for reason in ("resolution_count", "brier_degradation", "specialist_enabled", "none"):
            t = RetrainTrigger(reason=reason)
            assert t.reason == reason


# ── Check Retrain Trigger ────────────────────────────────────────


class TestCheckRetrainTrigger:
    def test_resolution_count_trigger(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=35)
        manager = SmartRetrainManager(conn, retrain_resolution_count=30)
        trigger = manager.check_retrain_trigger()
        assert trigger.should_retrain is True
        assert trigger.reason == "resolution_count"
        assert trigger.details["count"] >= 30

    def test_no_trigger(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=10)
        manager = SmartRetrainManager(conn, retrain_resolution_count=30)
        trigger = manager.check_retrain_trigger()
        assert trigger.should_retrain is False
        assert trigger.reason == "none"

    def test_specialist_trigger(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=5)
        # Set current specialists
        conn.execute("""
            INSERT INTO engine_state (key, value, updated_at)
            VALUES ('enabled_specialists', 'weather,crypto', ?)
        """, (time.time(),))
        # Set different last-retrain specialists
        conn.execute("""
            INSERT INTO engine_state (key, value, updated_at)
            VALUES ('last_retrain_specialists', 'weather', ?)
        """, (time.time(),))
        conn.commit()
        manager = SmartRetrainManager(conn, retrain_resolution_count=100)
        trigger = manager.check_retrain_trigger()
        assert trigger.should_retrain is True
        assert trigger.reason == "specialist_enabled"

    def test_recently_retrained(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=5)
        # Mark recent retrain
        conn.execute("""
            INSERT INTO engine_state (key, value, updated_at)
            VALUES ('last_retrain_time', ?, ?)
        """, (time.time() + 100, time.time()))
        conn.commit()
        manager = SmartRetrainManager(conn, retrain_resolution_count=30)
        trigger = manager.check_retrain_trigger()
        # Only 5 rows after last_retrain_time (none, since last_retrain is in future)
        assert trigger.should_retrain is False

    def test_empty_history(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        trigger = manager.check_retrain_trigger()
        assert trigger.should_retrain is False


# ── Rolling Brier ────────────────────────────────────────────────


class TestRollingBrier:
    def test_computes_correctly(self):
        conn = _create_test_db()
        # Insert recent rows within 7-day window
        now = time.time()
        for i in range(10):
            conn.execute("""
                INSERT INTO calibration_history
                    (forecast_prob, actual_outcome, recorded_at, market_id)
                VALUES (?, ?, ?, ?)
            """, (0.8, 1.0, now - i * 3600, f"m{i}"))
        conn.commit()
        manager = SmartRetrainManager(conn)
        brier = manager.get_rolling_brier(window_days=7)
        # Brier = (0.8 - 1.0)^2 = 0.04
        assert abs(brier - 0.04) < 0.001

    def test_empty_history(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        assert manager.get_rolling_brier() == 0.0

    def test_window_filtering(self):
        conn = _create_test_db()
        now = time.time()
        # Insert old rows (outside 7-day window)
        for i in range(10):
            conn.execute("""
                INSERT INTO calibration_history
                    (forecast_prob, actual_outcome, recorded_at, market_id)
                VALUES (?, ?, ?, ?)
            """, (0.8, 1.0, now - 30 * 86400, f"old{i}"))
        conn.commit()
        manager = SmartRetrainManager(conn)
        assert manager.get_rolling_brier(window_days=7) == 0.0

    def test_single_sample(self):
        conn = _create_test_db()
        conn.execute("""
            INSERT INTO calibration_history
                (forecast_prob, actual_outcome, recorded_at, market_id)
            VALUES (0.5, 1.0, ?, 'm1')
        """, (time.time(),))
        conn.commit()
        manager = SmartRetrainManager(conn)
        brier = manager.get_rolling_brier(window_days=7)
        assert abs(brier - 0.25) < 0.001  # (0.5 - 1.0)^2 = 0.25


# ── ABTestResult ─────────────────────────────────────────────────


class TestABTestResult:
    def test_defaults(self):
        r = ABTestResult()
        assert r.calibration_helps is True
        assert r.delta_brier == 0.0

    def test_to_dict(self):
        r = ABTestResult(
            test_id="ab-1", calibrated_brier=0.15,
            uncalibrated_brier=0.20, calibration_helps=True,
        )
        d = r.to_dict()
        assert d["test_id"] == "ab-1"
        assert d["calibration_helps"] is True
        assert d["calibrated_brier"] == 0.15

    def test_calibration_helps_flag(self):
        r = ABTestResult(calibration_helps=False, delta_brier=0.05)
        assert r.calibration_helps is False
        assert r.delta_brier > 0  # positive = calibration hurts


# ── Retrain with A/B Test ────────────────────────────────────────


class TestRetrainWithABTest:
    def test_runs_ab_test(self):
        conn = _create_test_db()
        # Insert enough rows with varying forecasts
        now = time.time()
        for i in range(50):
            prob = 0.3 + (i % 10) * 0.05
            outcome = 1.0 if prob > 0.5 else 0.0
            conn.execute("""
                INSERT INTO calibration_history
                    (forecast_prob, actual_outcome, recorded_at, market_id)
                VALUES (?, ?, ?, ?)
            """, (prob, outcome, now - i * 3600, f"m{i}"))
        conn.commit()
        manager = SmartRetrainManager(conn, ab_min_samples=20)
        trigger = RetrainTrigger(should_retrain=True, reason="resolution_count")
        result = manager.retrain_with_ab_test(trigger)
        assert result.test_id.startswith("ab-")
        assert result.calibrated_count > 0
        assert result.uncalibrated_count > 0
        assert result.trigger_reason == "resolution_count"

    def test_insufficient_samples(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=5)
        manager = SmartRetrainManager(conn, ab_min_samples=20)
        trigger = RetrainTrigger(should_retrain=True, reason="resolution_count")
        result = manager.retrain_with_ab_test(trigger)
        # Not enough samples for A/B
        assert result.calibrated_count == 0
        assert result.uncalibrated_count == 0

    def test_saves_ab_result(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=50)
        manager = SmartRetrainManager(conn, ab_min_samples=10)
        trigger = RetrainTrigger(should_retrain=True, reason="resolution_count")
        result = manager.retrain_with_ab_test(trigger)
        # Check it was persisted
        row = conn.execute(
            "SELECT * FROM calibration_ab_results WHERE test_id = ?",
            (result.test_id,),
        ).fetchone()
        assert row is not None

    def test_holdout_split_deterministic(self):
        conn = _create_test_db()
        _insert_calibration_rows(conn, n=50)
        manager = SmartRetrainManager(conn, ab_min_samples=10)
        trigger = RetrainTrigger(should_retrain=True, reason="test")
        r1 = manager.retrain_with_ab_test(trigger)
        # Holdout count should be deterministic for same market_ids
        conn2 = _create_test_db()
        _insert_calibration_rows(conn2, n=50)
        manager2 = SmartRetrainManager(conn2, ab_min_samples=10)
        r2 = manager2.retrain_with_ab_test(trigger)
        assert r1.uncalibrated_count == r2.uncalibrated_count


# ── A/B History ──────────────────────────────────────────────────


class TestABHistory:
    def test_fetches_all(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        for i in range(3):
            conn.execute("""
                INSERT INTO calibration_ab_results
                    (test_id, calibrated_brier, uncalibrated_brier,
                     calibrated_count, uncalibrated_count,
                     calibration_helps, delta_brier, trigger_reason,
                     started_at, completed_at)
                VALUES (?, 0.15, 0.20, 40, 10, 1, -0.05, 'test', '', '')
            """, (f"ab-{i}",))
        conn.commit()
        history = manager.get_ab_history()
        assert len(history) == 3

    def test_empty(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        assert manager.get_ab_history() == []

    def test_sorted_by_date(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        conn.execute("""
            INSERT INTO calibration_ab_results
                (test_id, calibrated_brier, uncalibrated_brier,
                 calibrated_count, uncalibrated_count,
                 calibration_helps, delta_brier, trigger_reason,
                 started_at, completed_at)
            VALUES ('ab-old', 0.15, 0.20, 40, 10, 1, -0.05, 'test', '', '2025-01-01')
        """)
        conn.execute("""
            INSERT INTO calibration_ab_results
                (test_id, calibrated_brier, uncalibrated_brier,
                 calibrated_count, uncalibrated_count,
                 calibration_helps, delta_brier, trigger_reason,
                 started_at, completed_at)
            VALUES ('ab-new', 0.10, 0.20, 40, 10, 1, -0.10, 'test', '', '2025-06-01')
        """)
        conn.commit()
        history = manager.get_ab_history()
        assert history[0].test_id == "ab-new"  # most recent first


# ── Save Retrain State ──────────────────────────────────────────


class TestSaveRetrainState:
    def test_saves_trigger(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        trigger = RetrainTrigger(
            should_retrain=True, reason="resolution_count",
        )
        manager.save_retrain_state(trigger)
        row = conn.execute(
            "SELECT value FROM engine_state WHERE key = 'last_retrain_reason'"
        ).fetchone()
        assert row["value"] == "resolution_count"

    def test_updates_time(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        trigger = RetrainTrigger(should_retrain=True, reason="test")
        manager.save_retrain_state(trigger)
        row = conn.execute(
            "SELECT value FROM engine_state WHERE key = 'last_retrain_time'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self):
        conn = _create_test_db()
        manager = SmartRetrainManager(conn)
        trigger = RetrainTrigger(should_retrain=True, reason="test1")
        manager.save_retrain_state(trigger)
        trigger2 = RetrainTrigger(should_retrain=True, reason="test2")
        manager.save_retrain_state(trigger2)
        row = conn.execute(
            "SELECT value FROM engine_state WHERE key = 'last_retrain_reason'"
        ).fetchone()
        assert row["value"] == "test2"


# ── Calibration Feedback Integration ─────────────────────────────


class TestCalibrationFeedbackIntegration:
    def test_smart_retrain_path(self):
        """When smart_retrain_enabled, should use SmartRetrainManager."""
        from src.analytics.calibration_feedback import (
            CalibrationFeedbackLoop,
            ResolutionRecord,
        )
        conn = _create_test_db()
        # Add performance_log and model_forecast_log tables
        conn.execute("""
            CREATE TABLE performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                question TEXT, category TEXT DEFAULT 'UNKNOWN',
                forecast_prob REAL, actual_outcome REAL,
                edge_at_entry REAL, confidence TEXT DEFAULT 'LOW',
                evidence_quality REAL DEFAULT 0, stake_usd REAL DEFAULT 0,
                entry_price REAL DEFAULT 0, exit_price REAL DEFAULT 0,
                pnl REAL DEFAULT 0, holding_hours REAL DEFAULT 0,
                resolved_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE model_forecast_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL, market_id TEXT NOT NULL,
                category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
                actual_outcome REAL, recorded_at TEXT
            )
        """)
        conn.commit()

        feedback = CalibrationFeedbackLoop(retrain_interval=10)
        record = ResolutionRecord(
            market_id="m1", question="Test?", category="MACRO",
            forecast_prob=0.7, actual_outcome=1.0, edge_at_entry=0.05,
            confidence="HIGH", evidence_quality=0.6, stake_usd=50.0,
            entry_price=0.5, exit_price=1.0, pnl=50.0, holding_hours=24.0,
        )
        # Should not crash when smart_retrain_enabled
        feedback.record_resolution(conn, record, smart_retrain_enabled=True)
        # Verify calibration_history was still populated
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM calibration_history"
        ).fetchone()
        assert row["cnt"] == 1

    def test_classic_retrain_path(self):
        """When smart_retrain_enabled=False, should use classic modulo retrain."""
        from src.analytics.calibration_feedback import (
            CalibrationFeedbackLoop,
            ResolutionRecord,
        )
        conn = _create_test_db()
        conn.execute("""
            CREATE TABLE performance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT, question TEXT, category TEXT DEFAULT 'UNKNOWN',
                forecast_prob REAL, actual_outcome REAL,
                edge_at_entry REAL, confidence TEXT DEFAULT 'LOW',
                evidence_quality REAL DEFAULT 0, stake_usd REAL DEFAULT 0,
                entry_price REAL DEFAULT 0, exit_price REAL DEFAULT 0,
                pnl REAL DEFAULT 0, holding_hours REAL DEFAULT 0,
                resolved_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE model_forecast_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT, market_id TEXT,
                category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
                actual_outcome REAL, recorded_at TEXT
            )
        """)
        conn.commit()

        feedback = CalibrationFeedbackLoop(retrain_interval=5)
        for i in range(6):
            record = ResolutionRecord(
                market_id=f"m{i}", question="Test?", category="MACRO",
                forecast_prob=0.7, actual_outcome=1.0, edge_at_entry=0.05,
                confidence="HIGH", evidence_quality=0.6, stake_usd=50.0,
                entry_price=0.5, exit_price=1.0, pnl=50.0, holding_hours=24.0,
            )
            feedback.record_resolution(conn, record, smart_retrain_enabled=False)
        # After 5 resolutions, classic retrain should have been triggered
        # (we can't directly verify retrain happened, but no crash = success)


# ── Dashboard Endpoint ───────────────────────────────────────────


def _create_dashboard_test_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


class TestDashboardEndpoint:
    @pytest.fixture
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_retrain_history_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/calibration/retrain-history")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["ab_tests"] == []
            assert data["total"] == 0

    def test_retrain_history_no_table(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            mock.return_value = conn
            resp = client.get("/api/calibration/retrain-history")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["ab_tests"] == []

    def test_retrain_history_with_data(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            conn.execute("""
                INSERT INTO calibration_ab_results
                    (test_id, calibrated_brier, uncalibrated_brier,
                     calibrated_count, uncalibrated_count,
                     calibration_helps, delta_brier, trigger_reason,
                     started_at, completed_at)
                VALUES ('ab-1', 0.15, 0.20, 40, 10, 1, -0.05, 'resolution_count', '', '')
            """)
            conn.commit()
            mock.return_value = conn
            resp = client.get("/api/calibration/retrain-history")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total"] == 1
            assert data["ab_tests"][0]["test_id"] == "ab-1"
