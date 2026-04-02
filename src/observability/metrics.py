"""Simple in-process metrics collection.

Stores counters, gauges, and histograms in memory.
Can be dumped to JSON for reporting.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from threading import Lock
from typing import Any, Generator

_MAX_EVENTS = 10_000  # Cap event history to bound memory usage


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile from pre-sorted data using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def _histogram_stats(values: list[float]) -> dict[str, Any]:
    """Compute histogram statistics including percentiles."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
    s = sorted(values)
    return {
        "count": len(s),
        "min": s[0],
        "max": s[-1],
        "avg": sum(s) / len(s),
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
    }


@dataclass
class MetricPoint:
    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Thread-safe in-process metrics collector."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._events: list[MetricPoint] = []

    def incr(self, name: str, value: float = 1.0, **tags: str) -> None:
        with self._lock:
            self._counters[name] += value
            self._events.append(MetricPoint(name=name, value=value, tags=tags))
            if len(self._events) > _MAX_EVENTS:
                self._events = self._events[-(_MAX_EVENTS // 2):]

    def gauge(self, name: str, value: float, **tags: str) -> None:
        with self._lock:
            self._gauges[name] = value
            self._events.append(MetricPoint(name=name, value=value, tags=tags))
            if len(self._events) > _MAX_EVENTS:
                self._events = self._events[-(_MAX_EVENTS // 2):]

    def histogram(self, name: str, value: float, **tags: str) -> None:
        with self._lock:
            self._histograms[name].append(value)
            self._events.append(MetricPoint(name=name, value=value, tags=tags))
            if len(self._events) > _MAX_EVENTS:
                self._events = self._events[-(_MAX_EVENTS // 2):]

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics with percentiles."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: _histogram_stats(v)
                    for k, v in self._histograms.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._events.clear()


# Global singleton
metrics = MetricsCollector()


# ── API Latency Tracking ─────────────────────────────────────────────

@contextlib.contextmanager
def track_latency(endpoint: str) -> Generator[None, None, None]:
    """Context manager that records API call latency to the metrics histogram.

    Usage:
        with track_latency("gamma"):
            result = await client._get("/markets")
    """
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000
        metrics.histogram(f"api_latency_ms.{endpoint}", elapsed_ms)


# ── API Cost Tracking ────────────────────────────────────────────────

# Approximate per-call costs (USD) for common APIs
_DEFAULT_COSTS: dict[str, float] = {
    "gpt-4o": 0.005,                    # ~$5/1M input tokens, ~1K tokens/call
    "gpt-4o-mini": 0.0005,
    "claude-3-5-sonnet-20241022": 0.005,
    "claude-sonnet-4-6": 0.008,
    "gemini-1.5-pro": 0.003,
    "gemini-2.0-flash": 0.0003,
    "serpapi": 0.005,                    # $50/5K searches
    "bing": 0.003,
    "tavily": 0.005,
}

# Per-token costs: (input_cost_per_token, output_cost_per_token)
_TOKEN_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50e-6, 10.00e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "claude-sonnet-4-6": (3.00e-6, 15.00e-6),
    "claude-3-5-sonnet-20241022": (3.00e-6, 15.00e-6),
    "gemini-2.0-flash": (0.10e-6, 0.40e-6),
    "gemini-1.5-pro": (1.25e-6, 5.00e-6),
}


class CostTracker:
    """Track API costs per cycle, daily, and cumulative with budget enforcement."""

    def __init__(
        self,
        cost_map: dict[str, float] | None = None,
        token_cost_map: dict[str, tuple[float, float]] | None = None,
    ):
        self._costs = cost_map or dict(_DEFAULT_COSTS)
        self._token_costs = token_cost_map or dict(_TOKEN_COSTS)
        self._lock = Lock()
        # Cycle counters
        self._cycle_calls: dict[str, int] = defaultdict(int)
        self._cycle_cost: float = 0.0
        # Daily counters
        self._daily_calls: dict[str, int] = defaultdict(int)
        self._daily_cost: float = 0.0
        self._current_date: str = date.today().isoformat()
        # Session/total counters
        self._total_calls: dict[str, int] = defaultdict(int)
        self._total_cost: float = 0.0
        # Token counters
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def _maybe_roll_day(self) -> None:
        """Reset daily counters if the date has changed. Must be called under lock."""
        today = date.today().isoformat()
        if today != self._current_date:
            self._daily_calls = defaultdict(int)
            self._daily_cost = 0.0
            self._current_date = today

    def record_call(
        self,
        api_name: str,
        count: int = 1,
        actual_cost: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record an API call and its estimated cost.

        Cost priority: actual_cost > token-based > flat estimate.

        Args:
            api_name: The API/model identifier.
            count: Number of calls.
            actual_cost: If provided, use this exact cost instead of estimates.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens generated.
        """
        if actual_cost is not None:
            cost = actual_cost
        elif input_tokens > 0 or output_tokens > 0:
            tok = self._token_costs.get(api_name)
            if tok:
                cost = input_tokens * tok[0] + output_tokens * tok[1]
            else:
                cost = self._costs.get(api_name, 0.001) * count
        else:
            cost = self._costs.get(api_name, 0.001) * count
        with self._lock:
            self._maybe_roll_day()
            self._cycle_calls[api_name] += count
            self._daily_calls[api_name] += count
            self._total_calls[api_name] += count
            self._cycle_cost += cost
            self._daily_cost += cost
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

    @property
    def daily_cost(self) -> float:
        with self._lock:
            self._maybe_roll_day()
            return self._daily_cost

    def check_budget(self, daily_limit: float) -> tuple[bool, float]:
        """Check if we can still spend today.

        Returns (can_spend, remaining_usd).
        """
        with self._lock:
            self._maybe_roll_day()
            remaining = max(0.0, daily_limit - self._daily_cost)
            return (remaining > 0, remaining)

    def end_cycle(self) -> dict[str, Any]:
        """End the current cycle and return cost summary. Resets cycle counters."""
        with self._lock:
            self._maybe_roll_day()
            summary = {
                "cycle_cost_usd": round(self._cycle_cost, 4),
                "cycle_calls": dict(self._cycle_calls),
                "daily_cost_usd": round(self._daily_cost, 4),
                "daily_calls": dict(self._daily_calls),
                "current_date": self._current_date,
                "total_cost_usd": round(self._total_cost, 4),
                "total_calls": dict(self._total_calls),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            }
            self._cycle_calls = defaultdict(int)
            self._cycle_cost = 0.0
            return summary

    def snapshot(self) -> dict[str, Any]:
        """Return current cost state without resetting."""
        with self._lock:
            self._maybe_roll_day()
            return {
                "cycle_cost_usd": round(self._cycle_cost, 4),
                "cycle_calls": dict(self._cycle_calls),
                "daily_cost_usd": round(self._daily_cost, 4),
                "daily_calls": dict(self._daily_calls),
                "current_date": self._current_date,
                "total_cost_usd": round(self._total_cost, 4),
                "total_calls": dict(self._total_calls),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            }


# Global cost tracker singleton
cost_tracker = CostTracker()
