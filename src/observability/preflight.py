"""Pre-flight readiness checklist — validate before going live.

Checks:
  1. Backtest Sharpe ratio above threshold
  2. Sufficient paper trading days with positive P&L
  3. Backtest-paper agreement within tolerance
  4. Most recent chaos tests all passed
  5. Database backup exists and is recent
  6. Budget caps are configured
  7. At least one alert channel is configured
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import BotConfig
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    required: bool = True  # If True, failure blocks live trading

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "required": self.required,
        }


@dataclass
class PreflightReport:
    """Complete pre-flight readiness report."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ready_for_live(self) -> bool:
        """True only if all required checks pass."""
        return all(c.passed for c in self.checks if c.required)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.required and not c.passed]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.required and not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_for_live": self.ready_for_live,
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "blocking_failures": len(self.blocking_failures),
            "warnings": len(self.warnings),
            "checks": [c.to_dict() for c in self.checks],
        }


class PreflightChecker:
    """Run pre-flight readiness checks."""

    def __init__(
        self,
        config: BotConfig,
        conn: sqlite3.Connection | None = None,
    ):
        self._config = config
        self._conn = conn

    def run_all(self) -> PreflightReport:
        """Run all pre-flight checks."""
        report = PreflightReport()
        checks = [
            self.check_backtest_sharpe,
            self.check_paper_pnl_duration,
            self.check_backtest_paper_agreement,
            self.check_chaos_tests_passed,
            self.check_db_backup,
            self.check_budget_caps,
            self.check_alert_channels,
        ]

        for check_fn in checks:
            try:
                result = check_fn()
            except Exception as e:
                result = CheckResult(
                    name=check_fn.__name__,
                    passed=False,
                    message=f"Check error: {e}",
                )
            report.checks.append(result)

        return report

    def check_backtest_sharpe(self) -> CheckResult:
        """Check that backtest Sharpe ratio meets threshold."""
        min_sharpe = self._config.production.preflight_min_sharpe

        if not self._conn:
            return CheckResult(
                name="backtest_sharpe",
                passed=False,
                message="No database connection for backtest check",
            )

        try:
            # Look for recent backtest runs via engine_state
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'last_backtest_sharpe'"
            ).fetchone()
            if row:
                sharpe = float(row["value"])
                passed = sharpe >= min_sharpe
                return CheckResult(
                    name="backtest_sharpe",
                    passed=passed,
                    message=f"Backtest Sharpe={sharpe:.2f} (min={min_sharpe:.2f})",
                )
        except (sqlite3.OperationalError, ValueError):
            pass

        return CheckResult(
            name="backtest_sharpe",
            passed=False,
            message=f"No backtest Sharpe data found (min required: {min_sharpe:.2f})",
        )

    def check_paper_pnl_duration(self) -> CheckResult:
        """Check that paper trading has been profitable for enough days."""
        min_days = self._config.production.preflight_min_paper_days

        if not self._conn:
            return CheckResult(
                name="paper_pnl_duration",
                passed=False,
                message="No database connection",
            )

        try:
            rows = self._conn.execute(
                "SELECT summary_date, total_pnl FROM daily_summaries "
                "ORDER BY summary_date DESC LIMIT ?",
                (min_days * 2,),
            ).fetchall()

            if not rows:
                return CheckResult(
                    name="paper_pnl_duration",
                    passed=False,
                    message=f"No daily summaries found (need {min_days}+ days)",
                )

            profitable_days = sum(1 for r in rows if float(r["total_pnl"]) > 0)
            total_days = len(rows)

            passed = total_days >= min_days and profitable_days > total_days // 2
            return CheckResult(
                name="paper_pnl_duration",
                passed=passed,
                message=(
                    f"{total_days} days tracked, {profitable_days} profitable "
                    f"(min {min_days} days required)"
                ),
            )
        except sqlite3.OperationalError:
            return CheckResult(
                name="paper_pnl_duration",
                passed=False,
                message="daily_summaries table not found",
            )

    def check_backtest_paper_agreement(self) -> CheckResult:
        """Check that backtest and paper results agree within tolerance."""
        tolerance = self._config.production.preflight_backtest_paper_tolerance

        if not self._conn:
            return CheckResult(
                name="backtest_paper_agreement",
                passed=False,
                message="No database connection",
                required=False,
            )

        try:
            bt_row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'last_backtest_sharpe'"
            ).fetchone()
            paper_row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'paper_sharpe'"
            ).fetchone()

            if not bt_row or not paper_row:
                return CheckResult(
                    name="backtest_paper_agreement",
                    passed=False,
                    message="Missing backtest or paper Sharpe data",
                    required=False,
                )

            bt_sharpe = float(bt_row["value"])
            paper_sharpe = float(paper_row["value"])

            if bt_sharpe == 0:
                return CheckResult(
                    name="backtest_paper_agreement",
                    passed=False,
                    message="Backtest Sharpe is zero",
                    required=False,
                )

            divergence = abs(bt_sharpe - paper_sharpe) / abs(bt_sharpe)
            passed = divergence <= tolerance
            return CheckResult(
                name="backtest_paper_agreement",
                passed=passed,
                message=(
                    f"Divergence={divergence:.1%} "
                    f"(bt={bt_sharpe:.2f}, paper={paper_sharpe:.2f}, "
                    f"tolerance={tolerance:.0%})"
                ),
                required=False,
            )
        except (sqlite3.OperationalError, ValueError):
            return CheckResult(
                name="backtest_paper_agreement",
                passed=False,
                message="Error reading Sharpe data",
                required=False,
            )

    def check_chaos_tests_passed(self) -> CheckResult:
        """Check that most recent chaos test run passed."""
        if not self._conn:
            return CheckResult(
                name="chaos_tests_passed",
                passed=False,
                message="No database connection",
            )

        try:
            rows = self._conn.execute(
                """SELECT run_id, passed, COUNT(*) as cnt
                FROM chaos_test_results
                GROUP BY run_id
                ORDER BY created_at DESC
                LIMIT 1"""
            ).fetchall()

            if not rows:
                return CheckResult(
                    name="chaos_tests_passed",
                    passed=False,
                    message="No chaos test results found — run `bot production chaos-test`",
                )

            # Check if all tests in the latest run passed
            run_id = rows[0]["run_id"]
            all_results = self._conn.execute(
                "SELECT passed FROM chaos_test_results WHERE run_id = ?",
                (run_id,),
            ).fetchall()

            all_passed = all(r["passed"] for r in all_results)
            total = len(all_results)
            passed_count = sum(1 for r in all_results if r["passed"])

            return CheckResult(
                name="chaos_tests_passed",
                passed=all_passed,
                message=f"Latest chaos run: {passed_count}/{total} passed (run_id={run_id})",
            )
        except sqlite3.OperationalError:
            return CheckResult(
                name="chaos_tests_passed",
                passed=False,
                message="chaos_test_results table not found",
            )

    def check_db_backup(self) -> CheckResult:
        """Check that a database backup exists and is recent (< 24h)."""
        db_path = Path(self._config.storage.sqlite_path)
        backup_path = db_path.with_suffix(".db.bak")

        if not backup_path.exists():
            return CheckResult(
                name="db_backup",
                passed=False,
                message=f"No backup found at {backup_path}",
                required=False,
            )

        age_hours = (time.time() - backup_path.stat().st_mtime) / 3600
        passed = age_hours < 24
        return CheckResult(
            name="db_backup",
            passed=passed,
            message=f"Backup age: {age_hours:.1f}h (max 24h)",
            required=False,
        )

    def check_budget_caps(self) -> CheckResult:
        """Check that budget caps are configured."""
        budget = self._config.budget

        if budget.enabled and budget.daily_limit_usd > 0:
            return CheckResult(
                name="budget_caps",
                passed=True,
                message=f"Budget cap: ${budget.daily_limit_usd:.2f}/day",
            )
        return CheckResult(
            name="budget_caps",
            passed=False,
            message="Budget caps not enabled or limit is zero",
        )

    def check_alert_channels(self) -> CheckResult:
        """Check that at least one alert channel is configured."""
        alerts = self._config.alerts

        channels = []
        if alerts.telegram_bot_token and alerts.telegram_chat_id:
            channels.append("telegram")
        if alerts.discord_webhook_url:
            channels.append("discord")
        if alerts.slack_webhook_url:
            channels.append("slack")
        if alerts.email_smtp_host and alerts.email_to:
            channels.append("email")

        if channels:
            return CheckResult(
                name="alert_channels",
                passed=True,
                message=f"Configured channels: {', '.join(channels)}",
                required=False,
            )
        return CheckResult(
            name="alert_channels",
            passed=False,
            message="No alert channels configured — configure at least one",
            required=False,
        )
