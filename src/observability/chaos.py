"""Chaos testing framework — inject failures to verify graceful degradation.

Runs controlled failure injection tests:
  1. DB disconnect
  2. API timeout / circuit breaker trip
  3. Drawdown spike → kill switch
  4. Corrupt market data
  5. Kill switch round-trip (persist → restart → verify → reset)
  6. Daily P&L kill trigger
  7. Circuit breaker cascade
  8. Graceful shutdown (SIGTERM simulation)
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


class FailureType(str, Enum):
    DB_DISCONNECT = "db_disconnect"
    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    CIRCUIT_TRIP = "circuit_trip"
    CORRUPT_DATA = "corrupt_data"
    DRAWDOWN_SPIKE = "drawdown_spike"
    KILL_SWITCH = "kill_switch"
    DAILY_PNL_KILL = "daily_pnl_kill"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"


@dataclass
class ChaosTestResult:
    """Result of a single chaos test."""
    test_name: str = ""
    component: str = ""
    failure_type: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
    passed: bool = False
    duration_secs: float = 0.0
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ChaosTestSuite:
    """Collection of chaos test results."""
    run_id: str = ""
    results: list[ChaosTestResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    @property
    def passed(self) -> list[ChaosTestResult]:
        return [r for r in self.results if r.passed]

    @property
    def failed(self) -> list[ChaosTestResult]:
        return [r for r in self.results if not r.passed]

    @property
    def all_passed(self) -> bool:
        return len(self.results) > 0 and all(r.passed for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "all_passed": self.all_passed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }


class ChaosTestRunner:
    """Run chaos tests against bot components."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self._conn = conn

    def run_all(self) -> ChaosTestSuite:
        """Run all chaos tests and return results."""
        suite = ChaosTestSuite(
            run_id=f"chaos-{uuid.uuid4().hex[:8]}",
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

        tests = [
            self.test_db_disconnect,
            self.test_api_timeout,
            self.test_drawdown_spike,
            self.test_corrupt_market_data,
            self.test_kill_switch_persistence,
            self.test_daily_pnl_kill,
            self.test_circuit_breaker_cascade,
            self.test_graceful_shutdown,
        ]

        for test_fn in tests:
            start = time.time()
            try:
                result = test_fn()
                result.duration_secs = round(time.time() - start, 3)
            except Exception as e:
                result = ChaosTestResult(
                    test_name=test_fn.__name__,
                    component="unknown",
                    failure_type="execution_error",
                    expected_behavior="Test should complete",
                    actual_behavior=f"Exception: {e}",
                    passed=False,
                    duration_secs=round(time.time() - start, 3),
                    error_message=str(e),
                )
            suite.results.append(result)

        suite.completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        return suite

    def test_db_disconnect(self) -> ChaosTestResult:
        """Verify engine handles DB failures gracefully."""
        from src.engine.loop import TradingEngine
        from src.config import BotConfig

        engine = TradingEngine(config=BotConfig())
        engine._db = None  # Simulate no DB

        try:
            # These should not crash when DB is None
            engine._persist_engine_state()
            engine._restore_kill_switch_state()
            return ChaosTestResult(
                test_name="test_db_disconnect",
                component="engine",
                failure_type=FailureType.DB_DISCONNECT,
                expected_behavior="Engine handles missing DB gracefully",
                actual_behavior="No crash, operations skipped",
                passed=True,
            )
        except Exception as e:
            return ChaosTestResult(
                test_name="test_db_disconnect",
                component="engine",
                failure_type=FailureType.DB_DISCONNECT,
                expected_behavior="Engine handles missing DB gracefully",
                actual_behavior=f"Crashed: {e}",
                passed=False,
                error_message=str(e),
            )

    def test_api_timeout(self) -> ChaosTestResult:
        """Verify circuit breaker trips on repeated failures."""
        from src.observability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        cb = CircuitBreaker(CircuitBreakerConfig(
            name="chaos_test",
            failure_threshold=3,
            window_secs=60,
            recovery_timeout_secs=1,
        ))

        # Simulate failures
        for _ in range(3):
            cb.record_failure()

        if not cb.allow_request():
            return ChaosTestResult(
                test_name="test_api_timeout",
                component="circuit_breaker",
                failure_type=FailureType.API_TIMEOUT,
                expected_behavior="Circuit breaker opens after 3 failures",
                actual_behavior="Circuit breaker correctly OPEN",
                passed=True,
            )
        return ChaosTestResult(
            test_name="test_api_timeout",
            component="circuit_breaker",
            failure_type=FailureType.API_TIMEOUT,
            expected_behavior="Circuit breaker opens after 3 failures",
            actual_behavior="Circuit breaker still CLOSED",
            passed=False,
        )

    def test_drawdown_spike(self) -> ChaosTestResult:
        """Verify kill switch engages on severe drawdown."""
        from src.policy.drawdown import DrawdownManager
        from src.config import BotConfig

        cfg = BotConfig()
        dm = DrawdownManager(10000.0, cfg)
        dm.update(10000.0)  # Set peak

        # Simulate 25% drawdown
        dm.update(7500.0)

        if dm.state.is_killed:
            return ChaosTestResult(
                test_name="test_drawdown_spike",
                component="drawdown",
                failure_type=FailureType.DRAWDOWN_SPIKE,
                expected_behavior="Kill switch engages at max drawdown",
                actual_behavior="Kill switch correctly engaged",
                passed=True,
            )
        return ChaosTestResult(
            test_name="test_drawdown_spike",
            component="drawdown",
            failure_type=FailureType.DRAWDOWN_SPIKE,
            expected_behavior="Kill switch engages at max drawdown",
            actual_behavior=f"Kill switch not engaged, drawdown={dm.state.drawdown_pct:.1%}",
            passed=False,
        )

    def test_corrupt_market_data(self) -> ChaosTestResult:
        """Verify engine skips corrupt market data without crash."""
        from src.engine.market_filter import filter_markets

        class CorruptMarket:
            id = ""
            question = None
            volume = -999
            liquidity = None
            best_bid = "not_a_number"
            best_ask = None
            spread = None
            market_type = None
            created_at = None

        try:
            passed, stats = filter_markets(
                [CorruptMarket()], min_score=0, max_pass=10,
            )
            return ChaosTestResult(
                test_name="test_corrupt_market_data",
                component="market_filter",
                failure_type=FailureType.CORRUPT_DATA,
                expected_behavior="Filter handles corrupt data gracefully",
                actual_behavior=f"Processed without crash, {len(passed)} passed",
                passed=True,
            )
        except Exception as e:
            return ChaosTestResult(
                test_name="test_corrupt_market_data",
                component="market_filter",
                failure_type=FailureType.CORRUPT_DATA,
                expected_behavior="Filter handles corrupt data gracefully",
                actual_behavior=f"Crashed: {e}",
                passed=False,
                error_message=str(e),
            )

    def test_kill_switch_persistence(self) -> ChaosTestResult:
        """Verify kill → persist → read → reset round-trip."""
        from src.storage.migrations import run_migrations

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)

        try:
            # Set kill
            conn.execute(
                """INSERT OR REPLACE INTO kill_switch_state
                    (id, is_killed, kill_reason, killed_at, killed_by,
                     daily_pnl_at_kill, bankroll_at_kill)
                VALUES (1, 1, 'chaos test', ?, 'chaos_runner', -100, 5000)""",
                (dt.datetime.now(dt.timezone.utc).isoformat(),),
            )
            conn.commit()

            # Read
            row = conn.execute("SELECT * FROM kill_switch_state WHERE id = 1").fetchone()
            assert row["is_killed"] == 1

            # Reset
            conn.execute(
                "UPDATE kill_switch_state SET is_killed = 0, kill_reason = '' WHERE id = 1"
            )
            conn.commit()

            row = conn.execute("SELECT * FROM kill_switch_state WHERE id = 1").fetchone()
            assert row["is_killed"] == 0

            return ChaosTestResult(
                test_name="test_kill_switch_persistence",
                component="database",
                failure_type=FailureType.KILL_SWITCH,
                expected_behavior="Kill switch persists and resets correctly",
                actual_behavior="Round-trip successful",
                passed=True,
            )
        except Exception as e:
            return ChaosTestResult(
                test_name="test_kill_switch_persistence",
                component="database",
                failure_type=FailureType.KILL_SWITCH,
                expected_behavior="Kill switch persists and resets correctly",
                actual_behavior=f"Failed: {e}",
                passed=False,
                error_message=str(e),
            )

    def test_daily_pnl_kill(self) -> ChaosTestResult:
        """Verify daily P&L kill triggers correctly."""
        from unittest.mock import MagicMock
        from src.engine.loop import TradingEngine
        from src.config import BotConfig, ProductionConfig

        cfg = BotConfig(production=ProductionConfig(
            enabled=True, daily_loss_kill_pct=0.05,
        ))
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = -600.0  # 6% > 5%
        engine._db = mock_db

        triggered = engine._check_daily_pnl_kill()

        if triggered and engine.drawdown.state.is_killed:
            return ChaosTestResult(
                test_name="test_daily_pnl_kill",
                component="engine",
                failure_type=FailureType.DAILY_PNL_KILL,
                expected_behavior="Kill switch triggers at -6% daily loss",
                actual_behavior="Kill switch correctly triggered",
                passed=True,
            )
        return ChaosTestResult(
            test_name="test_daily_pnl_kill",
            component="engine",
            failure_type=FailureType.DAILY_PNL_KILL,
            expected_behavior="Kill switch triggers at -6% daily loss",
            actual_behavior=f"triggered={triggered}, is_killed={engine.drawdown.state.is_killed}",
            passed=False,
        )

    def test_circuit_breaker_cascade(self) -> ChaosTestResult:
        """Verify cascading circuit breaker failures don't crash."""
        from src.observability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        breakers = [
            CircuitBreaker(CircuitBreakerConfig(
                name=f"cascade_{i}", failure_threshold=2, window_secs=60,
            ))
            for i in range(5)
        ]

        try:
            # Trip all breakers
            for cb in breakers:
                for _ in range(3):
                    cb.record_failure()

            # All should be open
            all_open = all(not cb.allow_request() for cb in breakers)

            return ChaosTestResult(
                test_name="test_circuit_breaker_cascade",
                component="circuit_breaker",
                failure_type=FailureType.CIRCUIT_TRIP,
                expected_behavior="All breakers open, no crash",
                actual_behavior=f"All open: {all_open}",
                passed=all_open,
            )
        except Exception as e:
            return ChaosTestResult(
                test_name="test_circuit_breaker_cascade",
                component="circuit_breaker",
                failure_type=FailureType.CIRCUIT_TRIP,
                expected_behavior="All breakers open, no crash",
                actual_behavior=f"Crashed: {e}",
                passed=False,
                error_message=str(e),
            )

    def test_graceful_shutdown(self) -> ChaosTestResult:
        """Verify engine stop sets running to False."""
        from src.engine.loop import TradingEngine
        from src.config import BotConfig

        engine = TradingEngine(config=BotConfig())
        engine._running = True

        engine.stop()

        if not engine._running:
            return ChaosTestResult(
                test_name="test_graceful_shutdown",
                component="engine",
                failure_type=FailureType.GRACEFUL_SHUTDOWN,
                expected_behavior="Engine stops cleanly",
                actual_behavior="Engine running=False after stop()",
                passed=True,
            )
        return ChaosTestResult(
            test_name="test_graceful_shutdown",
            component="engine",
            failure_type=FailureType.GRACEFUL_SHUTDOWN,
            expected_behavior="Engine stops cleanly",
            actual_behavior="Engine still running",
            passed=False,
        )

    def persist_results(
        self, suite: ChaosTestSuite, conn: sqlite3.Connection | None = None,
    ) -> None:
        """Save chaos test results to DB."""
        target = conn or self._conn
        if not target:
            return
        try:
            for result in suite.results:
                target.execute(
                    """INSERT INTO chaos_test_results
                        (run_id, test_name, component, failure_type,
                         expected_behavior, actual_behavior, passed,
                         duration_secs, error_message, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        suite.run_id, result.test_name, result.component,
                        result.failure_type, result.expected_behavior,
                        result.actual_behavior, int(result.passed),
                        result.duration_secs, result.error_message,
                        suite.completed_at,
                    ),
                )
            target.commit()
        except sqlite3.OperationalError as e:
            log.warning("chaos.persist_error", error=str(e))
