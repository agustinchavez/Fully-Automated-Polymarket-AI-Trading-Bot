"""Tests for Phase 9 Batch B: Chaos testing framework."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from src.observability.chaos import (
    ChaosTestResult,
    ChaosTestRunner,
    ChaosTestSuite,
    FailureType,
)


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


# ── ChaosTestResult ──────────────────────────────────────────────


class TestChaosTestResult:
    def test_defaults(self):
        r = ChaosTestResult()
        assert r.test_name == ""
        assert r.passed is False

    def test_to_dict(self):
        r = ChaosTestResult(
            test_name="test_db", component="engine",
            failure_type="db_disconnect", passed=True,
        )
        d = r.to_dict()
        assert d["test_name"] == "test_db"
        assert d["passed"] is True

    def test_with_error(self):
        r = ChaosTestResult(
            test_name="test_fail", passed=False,
            error_message="Something broke",
        )
        assert r.error_message == "Something broke"


# ── ChaosTestSuite ───────────────────────────────────────────────


class TestChaosTestSuite:
    def test_all_passed_true(self):
        suite = ChaosTestSuite(results=[
            ChaosTestResult(passed=True),
            ChaosTestResult(passed=True),
        ])
        assert suite.all_passed is True
        assert len(suite.passed) == 2
        assert len(suite.failed) == 0

    def test_all_passed_false(self):
        suite = ChaosTestSuite(results=[
            ChaosTestResult(passed=True),
            ChaosTestResult(passed=False),
        ])
        assert suite.all_passed is False

    def test_empty_suite(self):
        suite = ChaosTestSuite()
        assert suite.all_passed is False

    def test_to_dict(self):
        suite = ChaosTestSuite(
            run_id="test-run",
            results=[ChaosTestResult(passed=True)],
        )
        d = suite.to_dict()
        assert d["run_id"] == "test-run"
        assert d["total"] == 1
        assert d["all_passed"] is True


# ── Individual Chaos Tests ───────────────────────────────────────


class TestChaosDbDisconnect:
    def test_passes(self):
        runner = ChaosTestRunner()
        result = runner.test_db_disconnect()
        assert result.passed is True
        assert result.component == "engine"

    def test_failure_type(self):
        runner = ChaosTestRunner()
        result = runner.test_db_disconnect()
        assert result.failure_type == FailureType.DB_DISCONNECT

    def test_no_crash_with_no_db(self):
        runner = ChaosTestRunner()
        result = runner.test_db_disconnect()
        assert "No crash" in result.actual_behavior


class TestChaosApiTimeout:
    def test_circuit_breaker_trips(self):
        runner = ChaosTestRunner()
        result = runner.test_api_timeout()
        assert result.passed is True

    def test_circuit_breaker_component(self):
        runner = ChaosTestRunner()
        result = runner.test_api_timeout()
        assert result.component == "circuit_breaker"

    def test_failure_type(self):
        runner = ChaosTestRunner()
        result = runner.test_api_timeout()
        assert result.failure_type == FailureType.API_TIMEOUT


class TestChaosDrawdownSpike:
    def test_kill_engages(self):
        runner = ChaosTestRunner()
        result = runner.test_drawdown_spike()
        assert result.passed is True
        assert "engaged" in result.actual_behavior.lower()

    def test_component(self):
        runner = ChaosTestRunner()
        result = runner.test_drawdown_spike()
        assert result.component == "drawdown"

    def test_failure_type(self):
        runner = ChaosTestRunner()
        result = runner.test_drawdown_spike()
        assert result.failure_type == FailureType.DRAWDOWN_SPIKE


class TestChaosCorruptData:
    def test_handles_corrupt(self):
        runner = ChaosTestRunner()
        result = runner.test_corrupt_market_data()
        assert result.passed is True

    def test_component(self):
        runner = ChaosTestRunner()
        result = runner.test_corrupt_market_data()
        assert result.component == "market_filter"


class TestChaosKillSwitchRoundtrip:
    def test_round_trip(self):
        runner = ChaosTestRunner()
        result = runner.test_kill_switch_persistence()
        assert result.passed is True

    def test_component(self):
        runner = ChaosTestRunner()
        result = runner.test_kill_switch_persistence()
        assert result.component == "database"

    def test_failure_type(self):
        runner = ChaosTestRunner()
        result = runner.test_kill_switch_persistence()
        assert result.failure_type == FailureType.KILL_SWITCH


class TestChaosDailyPnlKill:
    def test_triggers(self):
        runner = ChaosTestRunner()
        result = runner.test_daily_pnl_kill()
        assert result.passed is True

    def test_component(self):
        runner = ChaosTestRunner()
        result = runner.test_daily_pnl_kill()
        assert result.component == "engine"


class TestChaosCascade:
    def test_cascade_passes(self):
        runner = ChaosTestRunner()
        result = runner.test_circuit_breaker_cascade()
        assert result.passed is True


class TestChaosGracefulShutdown:
    def test_shutdown(self):
        runner = ChaosTestRunner()
        result = runner.test_graceful_shutdown()
        assert result.passed is True
        assert result.component == "engine"


# ── Full Runner ──────────────────────────────────────────────────


class TestChaosTestRunner:
    def test_run_all(self):
        runner = ChaosTestRunner()
        suite = runner.run_all()
        assert len(suite.results) == 8
        assert suite.run_id.startswith("chaos-")

    def test_all_pass(self):
        runner = ChaosTestRunner()
        suite = runner.run_all()
        assert suite.all_passed is True

    def test_durations_populated(self):
        runner = ChaosTestRunner()
        suite = runner.run_all()
        for r in suite.results:
            assert r.duration_secs >= 0

    def test_persist_results(self):
        conn = _create_test_db()
        runner = ChaosTestRunner(conn)
        suite = runner.run_all()
        runner.persist_results(suite)

        rows = conn.execute(
            "SELECT * FROM chaos_test_results WHERE run_id = ?",
            (suite.run_id,),
        ).fetchall()
        assert len(rows) == 8
        assert all(r["passed"] for r in rows)
