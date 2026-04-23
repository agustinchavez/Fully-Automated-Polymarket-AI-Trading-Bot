"""Tests for Phase 9 Batch B: Pre-flight readiness checklist."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from src.config import BotConfig, ProductionConfig, AlertsConfig, BudgetConfig
from src.observability.preflight import (
    CheckResult,
    PreflightChecker,
    PreflightReport,
)


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _cfg(**overrides) -> BotConfig:
    prod_kw = {"enabled": True, **overrides}
    return BotConfig(production=ProductionConfig(**prod_kw))


# ── CheckResult ──────────────────────────────────────────────────


class TestCheckResult:
    def test_defaults(self):
        r = CheckResult(name="test", passed=True, message="ok")
        assert r.required is True

    def test_to_dict(self):
        r = CheckResult(name="test", passed=False, message="fail", required=False)
        d = r.to_dict()
        assert d["name"] == "test"
        assert d["passed"] is False
        assert d["required"] is False


# ── PreflightReport ──────────────────────────────────────────────


class TestPreflightReport:
    def test_ready_all_pass(self):
        report = PreflightReport(checks=[
            CheckResult("a", True, "ok"),
            CheckResult("b", True, "ok"),
        ])
        assert report.ready_for_live is True

    def test_not_ready_required_fails(self):
        report = PreflightReport(checks=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "fail", required=True),
        ])
        assert report.ready_for_live is False
        assert len(report.blocking_failures) == 1

    def test_ready_optional_fails(self):
        report = PreflightReport(checks=[
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "fail", required=False),
        ])
        assert report.ready_for_live is True
        assert len(report.warnings) == 1

    def test_to_dict(self):
        report = PreflightReport(checks=[
            CheckResult("a", True, "ok"),
        ])
        d = report.to_dict()
        assert d["ready_for_live"] is True
        assert d["total_checks"] == 1


# ── Backtest Sharpe ──────────────────────────────────────────────


class TestPreflightSharpe:
    def test_passes_above_threshold(self):
        conn = _create_test_db()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('last_backtest_sharpe', '1.5', ?)", (time.time(),)
        )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_min_sharpe=1.0), conn)
        result = checker.check_backtest_sharpe()
        assert result.passed is True
        assert "1.50" in result.message

    def test_fails_below_threshold(self):
        conn = _create_test_db()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('last_backtest_sharpe', '0.5', ?)", (time.time(),)
        )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_min_sharpe=1.0), conn)
        result = checker.check_backtest_sharpe()
        assert result.passed is False

    def test_fails_no_data(self):
        conn = _create_test_db()
        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_backtest_sharpe()
        assert result.passed is False
        assert "No backtest" in result.message


# ── Paper P&L Duration ───────────────────────────────────────────


class TestPreflightPaperDays:
    def test_passes_with_enough_days(self):
        conn = _create_test_db()
        # Insert 35 profitable days
        for i in range(35):
            conn.execute(
                "INSERT INTO daily_summaries (summary_date, total_pnl, created_at) "
                "VALUES (?, ?, ?)",
                (f"2024-01-{i+1:02d}", 10.0, "2024-01-01T00:00:00"),
            )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_min_paper_days=30), conn)
        result = checker.check_paper_pnl_duration()
        assert result.passed is True
        assert "35 days" in result.message

    def test_fails_not_enough_days(self):
        conn = _create_test_db()
        for i in range(10):
            conn.execute(
                "INSERT INTO daily_summaries (summary_date, total_pnl, created_at) "
                "VALUES (?, ?, ?)",
                (f"2024-01-{i+1:02d}", 10.0, "2024-01-01T00:00:00"),
            )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_min_paper_days=30), conn)
        result = checker.check_paper_pnl_duration()
        assert result.passed is False

    def test_fails_no_data(self):
        conn = _create_test_db()
        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_paper_pnl_duration()
        assert result.passed is False


# ── Backtest-Paper Agreement ─────────────────────────────────────


class TestPreflightAgreement:
    def test_passes_within_tolerance(self):
        conn = _create_test_db()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('last_backtest_sharpe', '1.5', ?)", (time.time(),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('paper_sharpe', '1.3', ?)", (time.time(),)
        )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_backtest_paper_tolerance=0.25), conn)
        result = checker.check_backtest_paper_agreement()
        assert result.passed is True

    def test_fails_outside_tolerance(self):
        conn = _create_test_db()
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('last_backtest_sharpe', '2.0', ?)", (time.time(),)
        )
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('paper_sharpe', '0.5', ?)", (time.time(),)
        )
        conn.commit()

        checker = PreflightChecker(_cfg(preflight_backtest_paper_tolerance=0.25), conn)
        result = checker.check_backtest_paper_agreement()
        assert result.passed is False

    def test_no_data(self):
        conn = _create_test_db()
        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_backtest_paper_agreement()
        assert result.passed is False


# ── Chaos Tests Passed ───────────────────────────────────────────


class TestPreflightChaosTests:
    def test_passes_all_green(self):
        conn = _create_test_db()
        for i in range(5):
            conn.execute(
                "INSERT INTO chaos_test_results "
                "(run_id, test_name, passed, created_at) "
                "VALUES ('run-1', ?, 1, '2024-01-15T00:00:00')",
                (f"test_{i}",),
            )
        conn.commit()

        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_chaos_tests_passed()
        assert result.passed is True
        assert "5/5" in result.message

    def test_fails_with_failures(self):
        conn = _create_test_db()
        conn.execute(
            "INSERT INTO chaos_test_results "
            "(run_id, test_name, passed, created_at) "
            "VALUES ('run-1', 'test_1', 1, '2024-01-15T00:00:00')"
        )
        conn.execute(
            "INSERT INTO chaos_test_results "
            "(run_id, test_name, passed, created_at) "
            "VALUES ('run-1', 'test_2', 0, '2024-01-15T00:00:00')"
        )
        conn.commit()

        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_chaos_tests_passed()
        assert result.passed is False

    def test_no_results(self):
        conn = _create_test_db()
        checker = PreflightChecker(_cfg(), conn)
        result = checker.check_chaos_tests_passed()
        assert result.passed is False
        assert "No chaos test" in result.message


# ── DB Backup ────────────────────────────────────────────────────


class TestPreflightDbBackup:
    def test_no_backup(self):
        checker = PreflightChecker(_cfg())
        result = checker.check_db_backup()
        assert result.passed is False
        assert result.required is False

    def test_backup_exists(self, tmp_path):
        # Create a fake backup file
        backup = tmp_path / "bot.db.bak"
        backup.write_text("backup")

        cfg = _cfg()
        cfg.storage.sqlite_path = str(tmp_path / "bot.db")
        checker = PreflightChecker(cfg)
        result = checker.check_db_backup()
        assert result.passed is True  # age < 24h since we just created it


# ── Budget Caps ──────────────────────────────────────────────────


class TestPreflightBudget:
    def test_enabled(self):
        cfg = _cfg()
        cfg.budget.enabled = True
        cfg.budget.daily_limit_usd = 5.0
        checker = PreflightChecker(cfg)
        result = checker.check_budget_caps()
        assert result.passed is True

    def test_disabled(self):
        cfg = _cfg()
        cfg.budget.enabled = False
        checker = PreflightChecker(cfg)
        result = checker.check_budget_caps()
        assert result.passed is False


# ── Alert Channels ───────────────────────────────────────────────


class TestPreflightAlerts:
    def test_no_channels(self):
        cfg = _cfg()
        checker = PreflightChecker(cfg)
        result = checker.check_alert_channels()
        assert result.passed is False
        assert result.required is False

    def test_telegram_configured(self):
        cfg = _cfg()
        cfg.alerts.telegram_bot_token = "token123"
        cfg.alerts.telegram_chat_id = "12345"
        checker = PreflightChecker(cfg)
        result = checker.check_alert_channels()
        assert result.passed is True
        assert "telegram" in result.message


# ── Full Run ─────────────────────────────────────────────────────


class TestPreflightFullRun:
    def test_run_all_returns_report(self):
        conn = _create_test_db()
        checker = PreflightChecker(_cfg(), conn)
        # Mock model availability to avoid real API calls in tests
        mock_result = CheckResult(
            name="model_availability", passed=True,
            message="All models OK (mocked)",
        )
        checker.check_model_availability = lambda: mock_result
        report = checker.run_all()
        assert len(report.checks) == 8

    def test_run_without_conn(self):
        checker = PreflightChecker(_cfg())
        mock_result = CheckResult(
            name="model_availability", passed=True,
            message="All models OK (mocked)",
        )
        checker.check_model_availability = lambda: mock_result
        report = checker.run_all()
        assert len(report.checks) == 8
        # Most DB-dependent checks should fail
        assert not report.ready_for_live


# ── CLI Commands ─────────────────────────────────────────────────


class TestCLICommands:
    def test_preflight_command(self):
        from click.testing import CliRunner
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["production", "preflight"])
        assert result.exit_code == 0
        assert "Pre-Flight" in result.output

    def test_chaos_test_command(self):
        from click.testing import CliRunner
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["production", "chaos-test"])
        assert result.exit_code == 0
        assert "Chaos Test" in result.output

    def test_production_group_help(self):
        from click.testing import CliRunner
        from src.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["production", "--help"])
        assert result.exit_code == 0
        assert "preflight" in result.output
        assert "chaos-test" in result.output
