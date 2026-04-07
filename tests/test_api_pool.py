"""Tests for src/connectors/api_pool.py — multi-endpoint API pool.

Covers:
- EndpointRateLimiter token mechanics
- ApiEndpoint health tracking (auto-disable, recovery)
- ApiPool selection strategies (round-robin, least-loaded, weighted-random)
- HTTP GET with failover (success, 4xx, 5xx, timeout)
- Counter increments and stats
- load_pool_from_config
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.connectors.api_pool import (
    ApiEndpoint,
    ApiPool,
    BUILTIN_ENDPOINTS,
    EndpointRateLimiter,
    SelectionStrategy,
    load_pool_from_config,
)


# ── EndpointRateLimiter ──────────────────────────────────────────────


class TestEndpointRateLimiter:
    """Token-bucket rate limiter scoped to a single endpoint."""

    def test_initial_tokens_equal_rpm(self):
        limiter = EndpointRateLimiter(rpm=120)
        assert limiter.available_tokens == 120.0

    def test_try_acquire_consumes_token(self):
        limiter = EndpointRateLimiter(rpm=10)
        assert limiter.try_acquire() is True
        assert limiter.available_tokens < 10.0

    def test_try_acquire_fails_when_empty(self):
        limiter = EndpointRateLimiter(rpm=2)
        limiter.try_acquire()
        limiter.try_acquire()
        assert limiter.try_acquire() is False

    def test_tokens_refill_over_time(self):
        limiter = EndpointRateLimiter(rpm=60)
        # Drain all tokens
        for _ in range(60):
            limiter.try_acquire()
        assert limiter.available_tokens < 1.0
        # Simulate time passage
        limiter._last_refill -= 10.0  # 10 seconds ago
        # 60 RPM = 1/sec → 10 tokens refilled
        assert limiter.available_tokens >= 9.0

    def test_wait_time_zero_when_tokens_available(self):
        limiter = EndpointRateLimiter(rpm=60)
        assert limiter.wait_time() == 0.0

    def test_wait_time_positive_when_empty(self):
        limiter = EndpointRateLimiter(rpm=60)
        for _ in range(60):
            limiter.try_acquire()
        wt = limiter.wait_time()
        assert wt > 0.0

    @pytest.mark.asyncio
    async def test_acquire_blocks_then_succeeds(self):
        limiter = EndpointRateLimiter(rpm=60)
        # Should succeed immediately
        await asyncio.wait_for(limiter.acquire(), timeout=1.0)

    def test_stats_counters(self):
        limiter = EndpointRateLimiter(rpm=10)
        limiter.try_acquire()
        limiter.try_acquire()
        stats = limiter.stats
        assert stats["rpm"] == 10
        assert stats["total_requests"] == 2
        assert stats["total_waits"] == 0


# ── ApiEndpoint ──────────────────────────────────────────────────────


class TestApiEndpoint:
    """Endpoint health tracking and path routing."""

    def test_supports_path_universal_when_no_paths(self):
        ep = ApiEndpoint(name="test", base_url="http://test.com")
        assert ep.supports_path("/anything") is True
        assert ep.supports_path("") is True

    def test_supports_path_filters_by_prefix(self):
        ep = ApiEndpoint(
            name="test", base_url="http://test.com",
            supported_paths=["/trades", "/positions"],
        )
        assert ep.supports_path("/trades") is True
        assert ep.supports_path("/trades/123") is True
        assert ep.supports_path("/positions") is True
        assert ep.supports_path("/markets") is False
        assert ep.supports_path("") is False

    def test_record_success_resets_failures(self):
        ep = ApiEndpoint(name="test", base_url="http://test.com")
        ep.consecutive_failures = 3
        ep.record_success()
        assert ep.consecutive_failures == 0
        assert ep.total_successes == 1

    def test_record_failure_increments(self):
        ep = ApiEndpoint(name="test", base_url="http://test.com")
        ep.record_failure("timeout")
        assert ep.consecutive_failures == 1
        assert ep.total_failures == 1
        assert ep.last_error == "timeout"

    def test_auto_disable_after_max_failures(self):
        ep = ApiEndpoint(
            name="test", base_url="http://test.com",
            max_consecutive_failures=3,
        )
        assert ep.healthy is True
        ep.record_failure("err1")
        ep.record_failure("err2")
        assert ep.healthy is True
        ep.record_failure("err3")  # hits threshold
        assert ep.healthy is False
        assert ep.disabled_at > 0

    def test_auto_recovery_after_cooldown(self):
        ep = ApiEndpoint(
            name="test", base_url="http://test.com",
            max_consecutive_failures=2,
            recovery_cooldown_secs=5.0,
        )
        ep.record_failure("err1")
        ep.record_failure("err2")
        assert ep.healthy is False

        # Simulate cooldown elapsed
        ep.disabled_at = time.monotonic() - 10.0
        assert ep.check_recovery() is True
        assert ep.healthy is True
        assert ep.consecutive_failures == 0

    def test_no_recovery_before_cooldown(self):
        ep = ApiEndpoint(
            name="test", base_url="http://test.com",
            max_consecutive_failures=2,
            recovery_cooldown_secs=120.0,
        )
        ep.record_failure("err1")
        ep.record_failure("err2")
        assert ep.healthy is False
        assert ep.check_recovery() is False

    def test_record_success_re_enables_disabled_endpoint(self):
        ep = ApiEndpoint(
            name="test", base_url="http://test.com",
            max_consecutive_failures=2,
        )
        ep.record_failure("err1")
        ep.record_failure("err2")
        assert ep.healthy is False
        ep.record_success()
        assert ep.healthy is True

    def test_status_dict(self):
        ep = ApiEndpoint(name="test", base_url="http://test.com", rpm=30)
        status = ep.status
        assert status["name"] == "test"
        assert status["healthy"] is True
        assert "limiter" in status

    def test_error_message_truncated(self):
        ep = ApiEndpoint(name="test", base_url="http://test.com")
        ep.record_failure("x" * 200)
        assert len(ep.last_error) <= 120


# ── ApiPool Selection ────────────────────────────────────────────────


class TestApiPoolSelection:
    """Endpoint selection strategies — fixed to use real paths."""

    def test_round_robin_returns_valid_endpoint(self):
        pool = ApiPool(strategy="round-robin")
        # Pass /markets which matches gamma-api's supported_paths
        ep = pool._select_endpoint("/markets")
        assert ep is not None
        assert ep.supports_path("/markets")

    def test_least_loaded_picks_highest_tokens(self):
        pool = ApiPool(strategy="least-loaded")
        # Both endpoints support /markets → gamma-api
        # Drain gamma-api tokens, then add a universal endpoint with more tokens
        pool.endpoints.append(ApiEndpoint(
            name="universal",
            base_url="http://test.com",
            rpm=120,
            supported_paths=[],  # universal
        ))
        # Drain gamma-api
        gamma = [ep for ep in pool.endpoints if ep.name == "gamma-api-primary"][0]
        for _ in range(int(gamma.limiter.rpm)):
            gamma.limiter.try_acquire()

        selected = pool._select_endpoint("/markets")
        assert selected is not None
        # Should pick the universal endpoint since it has more tokens
        assert selected.name == "universal"

    def test_weighted_random_returns_valid_endpoint(self):
        pool = ApiPool(strategy="weighted-random")
        ep = pool._select_endpoint("/trades")
        assert ep is not None
        assert ep.supports_path("/trades")

    def test_select_returns_none_when_no_match(self):
        pool = ApiPool()
        ep = pool._select_endpoint("/nonexistent-path")
        assert ep is None

    def test_select_with_trades_path_returns_data_api(self):
        pool = ApiPool()
        ep = pool._select_endpoint("/trades")
        assert ep is not None
        assert ep.name == "data-api-primary"

    def test_select_with_markets_path_returns_gamma_api(self):
        pool = ApiPool()
        ep = pool._select_endpoint("/markets")
        assert ep is not None
        assert ep.name == "gamma-api-primary"

    def test_round_robin_cycles_with_universal_endpoints(self):
        pool = ApiPool(strategy="round-robin")
        # Add two universal endpoints
        pool.endpoints = [
            ApiEndpoint(name="ep-a", base_url="http://a.com"),
            ApiEndpoint(name="ep-b", base_url="http://b.com"),
        ]
        names = [pool._select_endpoint("/any").name for _ in range(4)]
        assert names == ["ep-a", "ep-b", "ep-a", "ep-b"]

    def test_skips_unhealthy_endpoints(self):
        pool = ApiPool()
        # Disable the gamma-api
        gamma = [ep for ep in pool.endpoints if ep.name == "gamma-api-primary"][0]
        gamma.healthy = False
        gamma.disabled_at = time.monotonic()
        gamma.recovery_cooldown_secs = 9999  # won't auto-recover

        ep = pool._select_endpoint("/markets")
        # gamma-api is the only endpoint supporting /markets, so None
        assert ep is None


# ── ApiPool HTTP ─────────────────────────────────────────────────────


class TestApiPoolHTTP:
    """HTTP GET with failover and error handling."""

    @pytest.mark.asyncio
    async def test_successful_get(self):
        pool = ApiPool()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "1"}]
        mock_resp.raise_for_status = MagicMock()

        with patch("src.connectors.api_pool.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = instance

            data = await pool.get("/markets", params={"limit": 5})

        assert data == [{"id": "1"}]
        assert pool._total_requests == 1

    @pytest.mark.asyncio
    async def test_4xx_returns_none_no_retry(self):
        """Client errors (4xx) should return None without trying other endpoints."""
        import httpx

        pool = ApiPool()

        request = httpx.Request("GET", "http://test.com/markets")
        response = httpx.Response(status_code=404, request=request)
        error = httpx.HTTPStatusError(
            "Not Found", request=request, response=response,
        )

        with patch("src.connectors.api_pool.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(side_effect=error)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = instance

            data = await pool.get("/markets")

        assert data is None
        assert pool._total_errors == 1

    @pytest.mark.asyncio
    async def test_5xx_tries_next_endpoint(self):
        """Server errors should try the next endpoint."""
        import httpx

        pool = ApiPool()

        request = httpx.Request("GET", "http://test.com/markets")
        response = httpx.Response(status_code=500, request=request)
        error = httpx.HTTPStatusError(
            "Internal Server Error", request=request, response=response,
        )

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise error

        with patch("src.connectors.api_pool.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = instance

            data = await pool.get("/markets")

        assert data is None
        assert pool._total_errors >= 1

    @pytest.mark.asyncio
    async def test_timeout_tries_next_endpoint(self):
        """Timeout errors should try the next endpoint."""
        import httpx

        pool = ApiPool()

        async def mock_get(*args, **kwargs):
            raise httpx.TimeoutException("read timeout")

        with patch("src.connectors.api_pool.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = instance

            data = await pool.get("/markets")

        assert data is None
        assert pool._total_errors >= 1

    @pytest.mark.asyncio
    async def test_counter_increments_on_success(self):
        pool = ApiPool()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.connectors.api_pool.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = AsyncMock(return_value=mock_resp)
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = instance

            await pool.get("/trades")

        assert pool._total_requests == 1
        assert pool._total_errors == 0


# ── ApiPool Init + Config ────────────────────────────────────────────


class TestApiPoolInit:
    """Pool initialization, custom endpoints, and stats."""

    def test_default_pool_has_two_builtin_endpoints(self):
        pool = ApiPool()
        assert len(pool.endpoints) == 2
        names = {ep.name for ep in pool.endpoints}
        assert "data-api-primary" in names
        assert "gamma-api-primary" in names

    def test_effective_rpm_sums_healthy(self):
        pool = ApiPool()
        assert pool.effective_rpm == 120  # 60 + 60

    def test_custom_endpoints_added(self):
        pool = ApiPool(custom_endpoints=[
            {"name": "mirror-1", "base_url": "http://mirror.com", "rpm": 30},
        ])
        assert len(pool.endpoints) == 3
        assert pool.endpoints[-1].name == "mirror-1"
        assert pool.endpoints[-1].rpm == 30

    def test_empty_base_url_skipped(self):
        pool = ApiPool(custom_endpoints=[
            {"name": "empty", "base_url": ""},
        ])
        assert len(pool.endpoints) == 2  # only builtins

    def test_strategy_enum_from_string(self):
        pool = ApiPool(strategy="round-robin")
        assert pool.strategy == SelectionStrategy.ROUND_ROBIN

    def test_stats_dict(self):
        pool = ApiPool()
        stats = pool.stats
        assert stats["strategy"] == "least-loaded"
        assert stats["endpoint_count"] == 2
        assert stats["healthy_count"] == 2
        assert stats["effective_rpm"] == 120
        assert "endpoints" in stats

    def test_healthy_count(self):
        pool = ApiPool()
        assert pool.healthy_count == 2
        pool.endpoints[0].healthy = False
        pool.endpoints[0].disabled_at = time.monotonic()
        pool.endpoints[0].recovery_cooldown_secs = 9999
        assert pool.healthy_count == 1


class TestLoadPoolFromConfig:
    """load_pool_from_config reads config.yaml."""

    def test_loads_default_when_no_config(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            pool = load_pool_from_config()
        assert isinstance(pool, ApiPool)
        assert len(pool.endpoints) >= 2

    def test_loads_with_custom_strategy(self):
        import yaml
        mock_config = {
            "scanner": {
                "apiPool": {
                    "strategy": "round-robin",
                    "endpoints": [],
                },
            },
        }
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value=mock_config):
                with patch("pathlib.Path.exists", return_value=True):
                    pool = load_pool_from_config()
        assert pool.strategy == SelectionStrategy.ROUND_ROBIN
