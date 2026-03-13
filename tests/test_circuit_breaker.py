"""Tests for circuit breaker module (Phase 0B)."""

from __future__ import annotations

import time

import pytest

from src.observability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreakerStates:

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_threshold(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, window_secs=60.0))
        cb.allow_request()
        cb.record_failure()
        cb.allow_request()
        cb.record_failure()
        # 2 failures, threshold is 3
        assert cb.state == CircuitState.CLOSED

    def test_opens_at_threshold(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, window_secs=60.0))
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout_secs=0.05,
        ))
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_calls(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_secs=0.01,
            half_open_max_calls=2,
        ))
        cb.allow_request()
        cb.record_failure()
        time.sleep(0.02)
        # First two should be allowed
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        # Third should be rejected
        assert cb.allow_request() is False

    def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout_secs=0.01,
        ))
        cb.allow_request()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout_secs=0.01,
        ))
        cb.allow_request()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


class TestSlidingWindow:

    def test_old_failures_expire(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3, window_secs=0.05,
        ))
        cb.allow_request()
        cb.record_failure()
        cb.allow_request()
        cb.record_failure()
        time.sleep(0.06)  # window expires
        # One more failure — total in window is now 1, not 3
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_time_until_retry(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout_secs=1.0,
        ))
        cb.allow_request()
        cb.record_failure()
        remaining = cb.time_until_retry()
        assert 0.5 < remaining <= 1.0

    def test_time_until_retry_when_closed(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        assert cb.time_until_retry() == 0.0


class TestCircuitBreakerRegistry:

    def test_get_creates_default(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("gamma")
        assert cb.state == CircuitState.CLOSED

    def test_get_returns_same_instance(self) -> None:
        reg = CircuitBreakerRegistry()
        cb1 = reg.get("gamma")
        cb2 = reg.get("gamma")
        assert cb1 is cb2

    def test_get_unknown_creates_default(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("unknown_endpoint")
        assert cb.state == CircuitState.CLOSED

    def test_configure_overrides(self) -> None:
        reg = CircuitBreakerRegistry()
        reg.configure("custom", failure_threshold=1)
        cb = reg.get("custom")
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stats(self) -> None:
        reg = CircuitBreakerRegistry()
        reg.get("gamma")
        s = reg.stats()
        assert "gamma" in s
        assert s["gamma"]["state"] == "closed"

    def test_reset_all(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("test")
        # Force open
        for _ in range(10):
            cb.allow_request()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        reg.reset_all()
        assert cb.state == CircuitState.CLOSED


class TestReset:

    def test_reset_clears_state(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.stats["window_failures"] == 0


class TestCircuitOpenError:

    def test_error_message(self) -> None:
        err = CircuitOpenError("gamma", 25.3)
        assert "gamma" in str(err)
        assert "OPEN" in str(err)
        assert err.name == "gamma"
        assert err.retry_after_secs == 25.3


class TestStats:

    def test_stats_count_correctly(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))
        cb.allow_request()
        cb.record_success()
        cb.allow_request()
        cb.record_failure()
        s = cb.stats
        assert s["total_calls"] == 2
        assert s["total_successes"] == 1
        assert s["total_failures"] == 1
        assert s["total_rejections"] == 0

    def test_rejection_counted(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.allow_request()
        cb.record_failure()
        cb.allow_request()  # rejected
        s = cb.stats
        assert s["total_rejections"] == 1


class TestConfigLoads:

    def test_circuit_breaker_config_in_botconfig(self) -> None:
        from src.config import load_config
        cfg = load_config()
        assert cfg.circuit_breakers.enabled is True
        assert cfg.circuit_breakers.default_failure_threshold == 5
