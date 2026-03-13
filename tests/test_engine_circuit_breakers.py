"""Tests for circuit breaker integration in the engine loop."""

from __future__ import annotations

import pytest

from src.observability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestDiscoveryCircuitBreaker:
    """Gamma circuit breaker controls market discovery."""

    def test_open_circuit_rejects(self) -> None:
        """When gamma circuit is open, requests are rejected."""
        reg = CircuitBreakerRegistry()
        cb = reg.get("gamma")
        for _ in range(5):
            cb.allow_request()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_success_keeps_closed(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("gamma")
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestResearchCircuitBreaker:
    """Research circuit breaker protects search/extraction pipeline."""

    def test_trips_after_threshold(self) -> None:
        """Research CB has threshold=3, should trip after 3 failures."""
        reg = CircuitBreakerRegistry()
        cb = reg.get("research")
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_further_research(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("research")
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()
        assert cb.allow_request() is False

    def test_success_resets_from_half_open(self) -> None:
        import time
        reg = CircuitBreakerRegistry()
        reg.configure("research_test", failure_threshold=1, recovery_timeout_secs=0.01)
        cb = reg.get("research_test")
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestForecastCircuitBreaker:
    """Forecast circuit breaker protects LLM calls."""

    def test_trips_after_threshold(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("forecast")
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects(self) -> None:
        reg = CircuitBreakerRegistry()
        cb = reg.get("forecast")
        for _ in range(3):
            cb.allow_request()
            cb.record_failure()
        assert cb.allow_request() is False


class TestMultipleBreakersIndependent:
    """Circuit breakers for different endpoints are independent."""

    def test_gamma_open_doesnt_affect_research(self) -> None:
        reg = CircuitBreakerRegistry()
        gamma = reg.get("gamma")
        research = reg.get("research")
        # Trip gamma
        for _ in range(5):
            gamma.allow_request()
            gamma.record_failure()
        assert gamma.state == CircuitState.OPEN
        # Research should still be fine
        assert research.state == CircuitState.CLOSED
        assert research.allow_request() is True
