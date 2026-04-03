"""Circuit breaker for external API and pipeline stage protection.

States:
  CLOSED    — Normal operation; failures are counted in a sliding window
  OPEN      — All calls rejected; waiting for recovery timeout
  HALF_OPEN — Allow limited probe calls; success resets, failure re-opens

Thread-safe, in-memory state (no database persistence needed).
Follows the same registry/singleton pattern as rate_limiter.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any

from src.observability.logger import get_logger
from src.observability.metrics import metrics

log = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""

    def __init__(self, name: str, retry_after_secs: float):
        self.name = name
        self.retry_after_secs = retry_after_secs
        super().__init__(
            f"Circuit breaker '{name}' is OPEN. "
            f"Retry after {retry_after_secs:.1f}s"
        )


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""

    failure_threshold: int = 5
    window_secs: float = 60.0
    recovery_timeout_secs: float = 30.0
    half_open_max_calls: int = 2
    name: str = ""


# Default configs per endpoint/stage
DEFAULT_BREAKER_CONFIGS: dict[str, CircuitBreakerConfig] = {
    "gamma": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="Polymarket Gamma",
    ),
    "clob": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="Polymarket CLOB",
    ),
    "research": CircuitBreakerConfig(
        failure_threshold=3, window_secs=120.0,
        recovery_timeout_secs=60.0, half_open_max_calls=1, name="Research Pipeline",
    ),
    "forecast": CircuitBreakerConfig(
        failure_threshold=3, window_secs=120.0,
        recovery_timeout_secs=60.0, half_open_max_calls=1, name="Forecast Pipeline",
    ),
    "openai": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="OpenAI",
    ),
    "anthropic": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="Anthropic",
    ),
    "google": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="Google Gemini",
    ),
    "serpapi": CircuitBreakerConfig(
        failure_threshold=3, window_secs=120.0,
        recovery_timeout_secs=60.0, half_open_max_calls=1, name="SerpAPI",
    ),
    "xai": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="xAI Grok",
    ),
    "deepseek": CircuitBreakerConfig(
        failure_threshold=5, window_secs=60.0,
        recovery_timeout_secs=30.0, half_open_max_calls=2, name="DeepSeek",
    ),
}


class CircuitBreaker:
    """Thread-safe circuit breaker with sliding window failure counting."""

    def __init__(self, config: CircuitBreakerConfig):
        self._config = config
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failures: deque[float] = deque()  # timestamps of failures
        self._opened_at: float = 0.0
        self._half_open_calls: int = 0
        # Stats
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_rejections: int = 0
        self._total_successes: int = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition()
            return self._state

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_rejections": self._total_rejections,
                "total_successes": self._total_successes,
                "window_failures": len(self._failures),
                "failure_threshold": self._config.failure_threshold,
            }

    def _maybe_transition(self) -> None:
        """Check if state should transition. Must be called under lock."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout_secs:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                log.info(
                    "circuit_breaker.half_open",
                    name=self._config.name,
                    recovery_secs=self._config.recovery_timeout_secs,
                )
                metrics.gauge(
                    "circuit_breaker.state", 0.5,
                    endpoint=self._config.name,
                )

    def _purge_old_failures(self) -> None:
        """Remove failures outside the sliding window. Must be called under lock."""
        cutoff = time.monotonic() - self._config.window_secs
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns True if the request can proceed, False if the circuit is open.
        Call record_success() or record_failure() after the call completes.
        """
        with self._lock:
            self._maybe_transition()

            if self._state == CircuitState.CLOSED:
                self._total_calls += 1
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._config.half_open_max_calls:
                    self._half_open_calls += 1
                    self._total_calls += 1
                    return True
                self._total_rejections += 1
                return False

            # OPEN
            self._total_rejections += 1
            return False

    def record_success(self) -> None:
        """Record a successful call. Resets circuit if in HALF_OPEN."""
        with self._lock:
            self._total_successes += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures.clear()
                self._half_open_calls = 0
                log.info("circuit_breaker.closed", name=self._config.name)
                metrics.gauge(
                    "circuit_breaker.state", 0.0,
                    endpoint=self._config.name,
                )

    def record_failure(self) -> None:
        """Record a failed call. May trip the breaker."""
        with self._lock:
            now = time.monotonic()
            self._total_failures += 1
            self._failures.append(now)
            self._purge_old_failures()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = now
                log.warning(
                    "circuit_breaker.reopened",
                    name=self._config.name,
                )
                metrics.gauge(
                    "circuit_breaker.state", 1.0,
                    endpoint=self._config.name,
                )
                return

            if self._state == CircuitState.CLOSED:
                if len(self._failures) >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = now
                    log.warning(
                        "circuit_breaker.opened",
                        name=self._config.name,
                        failures=len(self._failures),
                        window_secs=self._config.window_secs,
                    )
                    metrics.gauge(
                        "circuit_breaker.state", 1.0,
                        endpoint=self._config.name,
                    )

    def time_until_retry(self) -> float:
        """Seconds until the circuit transitions from OPEN to HALF_OPEN."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, self._config.recovery_timeout_secs - elapsed)

    def reset(self) -> None:
        """Force-reset the circuit breaker to CLOSED. For testing/admin."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures.clear()
            self._half_open_calls = 0
            self._opened_at = 0.0


class CircuitBreakerRegistry:
    """Global registry of per-endpoint circuit breakers.

    Follows the same pattern as RateLimiterRegistry.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        with self._lock:
            if name not in self._breakers:
                config = DEFAULT_BREAKER_CONFIGS.get(
                    name,
                    CircuitBreakerConfig(name=name),
                )
                self._breakers[name] = CircuitBreaker(config)
            return self._breakers[name]

    def configure(
        self,
        name: str,
        failure_threshold: int = 5,
        window_secs: float = 60.0,
        recovery_timeout_secs: float = 30.0,
        half_open_max_calls: int = 2,
    ) -> None:
        """Override circuit breaker config for an endpoint."""
        with self._lock:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                window_secs=window_secs,
                recovery_timeout_secs=recovery_timeout_secs,
                half_open_max_calls=half_open_max_calls,
                name=name,
            )
            self._breakers[name] = CircuitBreaker(config)

    def stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        with self._lock:
            return {name: cb.stats for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED. For testing/admin."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()


# Global singleton (mirrors rate_limiter pattern)
circuit_breakers = CircuitBreakerRegistry()
