"""Tests for API latency tracking (Phase 0B)."""

from __future__ import annotations

import time

import pytest

from src.observability.metrics import MetricsCollector, track_latency


class TestTrackLatency:

    def test_records_histogram(self) -> None:
        """track_latency records elapsed time into metrics histogram."""
        import src.observability.metrics as mod
        original = mod.metrics
        m = MetricsCollector()
        mod.metrics = m
        try:
            with track_latency("test_endpoint"):
                time.sleep(0.01)
            snap = m.snapshot()
            key = "api_latency_ms.test_endpoint"
            assert key in snap["histograms"]
            stats = snap["histograms"][key]
            assert stats["count"] == 1
            assert stats["p50"] >= 10.0  # at least 10ms
        finally:
            mod.metrics = original

    def test_records_on_exception(self) -> None:
        """Latency is recorded even when the wrapped code raises."""
        import src.observability.metrics as mod
        original = mod.metrics
        m = MetricsCollector()
        mod.metrics = m
        try:
            with pytest.raises(ValueError):
                with track_latency("failing"):
                    raise ValueError("boom")
            snap = m.snapshot()
            assert "api_latency_ms.failing" in snap["histograms"]
            assert snap["histograms"]["api_latency_ms.failing"]["count"] == 1
        finally:
            mod.metrics = original

    def test_multiple_calls_accumulate(self) -> None:
        """Multiple track_latency calls accumulate in the same histogram."""
        import src.observability.metrics as mod
        original = mod.metrics
        m = MetricsCollector()
        mod.metrics = m
        try:
            for _ in range(5):
                with track_latency("multi"):
                    pass  # near-zero latency
            snap = m.snapshot()
            assert snap["histograms"]["api_latency_ms.multi"]["count"] == 5
        finally:
            mod.metrics = original

    def test_different_endpoints_separate(self) -> None:
        """Different endpoint names produce separate histogram keys."""
        import src.observability.metrics as mod
        original = mod.metrics
        m = MetricsCollector()
        mod.metrics = m
        try:
            with track_latency("gamma"):
                pass
            with track_latency("clob"):
                pass
            snap = m.snapshot()
            assert "api_latency_ms.gamma" in snap["histograms"]
            assert "api_latency_ms.clob" in snap["histograms"]
        finally:
            mod.metrics = original


class TestMetricsHistogramPercentiles:

    def test_percentile_calculation(self) -> None:
        m = MetricsCollector()
        for v in [10, 20, 30, 40, 50]:
            m.histogram("test", float(v))
        snap = m.snapshot()
        stats = snap["histograms"]["test"]
        assert stats["count"] == 5
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["p50"] == pytest.approx(30.0, abs=5)

    def test_empty_histogram(self) -> None:
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["histograms"] == {}
