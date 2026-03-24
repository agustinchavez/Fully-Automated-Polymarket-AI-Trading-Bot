"""Tests for Phase 6 Batch C: Execution strategy auto-selection."""

from __future__ import annotations

import pytest

from src.config import ExecutionConfig
from src.execution.strategy_selector import (
    ExecutionStrategySelector,
    StrategyRecommendation,
)


# ── StrategyRecommendation ────────────────────────────────────────


class TestStrategyRecommendation:
    def test_defaults(self):
        rec = StrategyRecommendation(
            strategy="simple",
            reason="Normal",
            confidence=0.75,
        )
        assert rec.strategy == "simple"
        assert rec.alternative == ""
        assert rec.depth_usd == 0.0
        assert rec.order_size_usd == 0.0
        assert rec.order_pct_of_depth == 0.0

    def test_to_dict(self):
        rec = StrategyRecommendation(
            strategy="twap",
            reason="Large order",
            confidence=0.80,
            alternative="iceberg",
            depth_usd=10000.0,
            order_size_usd=2000.0,
            order_pct_of_depth=0.2,
        )
        d = rec.to_dict()
        assert d["strategy"] == "twap"
        assert d["reason"] == "Large order"
        assert d["confidence"] == 0.8
        assert d["alternative"] == "iceberg"
        assert d["depth_usd"] == 10000.0
        assert d["order_size_usd"] == 2000.0
        assert d["order_pct_of_depth"] == 0.2


# ── ExecutionStrategySelector ─────────────────────────────────────


class TestExecutionStrategySelector:
    def _make(self, **kw) -> ExecutionStrategySelector:
        return ExecutionStrategySelector(**kw)

    def test_thin_liquidity_iceberg(self):
        """Depth below threshold should recommend iceberg."""
        sel = self._make(thin_depth_usd=5000.0)
        rec = sel.select(order_size_usd=100.0, depth_usd=2000.0)
        assert rec.strategy == "iceberg"
        assert rec.confidence == 0.85
        assert rec.alternative == "simple"
        assert "Thin liquidity" in rec.reason

    def test_large_order_twap(self):
        """Large order (> 10% of depth) should recommend TWAP."""
        sel = self._make(thin_depth_usd=1000.0, large_order_pct=0.10)
        rec = sel.select(order_size_usd=2000.0, depth_usd=10000.0)
        assert rec.strategy == "twap"
        assert rec.confidence == 0.80
        assert rec.alternative == "iceberg"
        assert "Large order" in rec.reason

    def test_normal_conditions_simple(self):
        """Normal conditions should recommend simple."""
        sel = self._make(thin_depth_usd=5000.0, large_order_pct=0.10)
        rec = sel.select(order_size_usd=100.0, depth_usd=50000.0)
        assert rec.strategy == "simple"
        assert rec.confidence == 0.75
        assert "Normal conditions" in rec.reason

    def test_zero_depth(self):
        """Zero depth should recommend iceberg (thin liquidity)."""
        sel = self._make(thin_depth_usd=5000.0)
        rec = sel.select(order_size_usd=100.0, depth_usd=0.0)
        assert rec.strategy == "iceberg"

    def test_both_triggers_thin_wins(self):
        """When both thin and large trigger, thin (iceberg) has priority."""
        sel = self._make(thin_depth_usd=5000.0, large_order_pct=0.10)
        # depth=3000 < 5000 threshold, AND order=500 > 10% of 3000
        rec = sel.select(order_size_usd=500.0, depth_usd=3000.0)
        assert rec.strategy == "iceberg"  # Rule 1 fires first

    def test_custom_thresholds(self):
        """Custom thresholds should work correctly."""
        sel = self._make(thin_depth_usd=1000.0, large_order_pct=0.50)
        # Depth 2000 > 1000 threshold, order 500 = 25% < 50% threshold
        rec = sel.select(order_size_usd=500.0, depth_usd=2000.0)
        assert rec.strategy == "simple"

    def test_threshold_edge_case_exact_pct(self):
        """Order at exactly the threshold should still trigger TWAP."""
        sel = self._make(thin_depth_usd=1000.0, large_order_pct=0.10)
        # order=1001, depth=10000, pct=0.1001 > 0.10
        rec = sel.select(order_size_usd=1001.0, depth_usd=10000.0)
        assert rec.strategy == "twap"

    def test_just_below_pct_threshold(self):
        """Order just below threshold should be simple."""
        sel = self._make(thin_depth_usd=1000.0, large_order_pct=0.10)
        # order=999, depth=10000, pct=0.0999 < 0.10
        rec = sel.select(order_size_usd=999.0, depth_usd=10000.0)
        assert rec.strategy == "simple"


# ── Learning Mode ─────────────────────────────────────────────────


class TestStrategySelectorLearning:
    def test_learning_disabled(self):
        """Learning disabled should fall through to default."""
        sel = ExecutionStrategySelector(
            learning_enabled=False,
        )
        rec = sel.select(
            order_size_usd=100.0,
            depth_usd=50000.0,
            historical_quality="dummy",
        )
        assert rec.strategy == "simple"

    def test_learning_with_mock_quality(self):
        """Learning with clear winner should recommend that strategy."""

        class MockQuality:
            strategy_stats = {
                "simple": {
                    "count": 20,
                    "avg_slippage_bps": 5.0,
                    "avg_fill_rate": 0.95,
                },
                "twap": {
                    "count": 15,
                    "avg_slippage_bps": 2.0,
                    "avg_fill_rate": 0.90,
                },
            }

        sel = ExecutionStrategySelector(
            learning_enabled=True,
            min_samples=10,
        )
        rec = sel.select(
            order_size_usd=100.0,
            depth_usd=50000.0,
            historical_quality=MockQuality(),
        )
        assert rec.strategy == "twap"  # Lower slippage
        assert "Learning mode" in rec.reason
        assert rec.confidence == 0.70

    def test_learning_insufficient_samples(self):
        """Learning with too few samples should fall through."""

        class MockQuality:
            strategy_stats = {
                "simple": {
                    "count": 3,  # below min_samples=10
                    "avg_slippage_bps": 1.0,
                    "avg_fill_rate": 0.95,
                },
            }

        sel = ExecutionStrategySelector(
            learning_enabled=True,
            min_samples=10,
        )
        rec = sel.select(
            order_size_usd=100.0,
            depth_usd=50000.0,
            historical_quality=MockQuality(),
        )
        assert rec.strategy == "simple"
        assert "Normal conditions" in rec.reason  # Default path

    def test_learning_no_clear_winner(self):
        """When all strategies have poor fill rate, fall through."""

        class MockQuality:
            strategy_stats = {
                "simple": {
                    "count": 20,
                    "avg_slippage_bps": 1.0,
                    "avg_fill_rate": 0.50,  # below 0.80 threshold
                },
                "twap": {
                    "count": 20,
                    "avg_slippage_bps": 2.0,
                    "avg_fill_rate": 0.60,  # below 0.80 threshold
                },
            }

        sel = ExecutionStrategySelector(
            learning_enabled=True,
            min_samples=10,
        )
        rec = sel.select(
            order_size_usd=100.0,
            depth_usd=50000.0,
            historical_quality=MockQuality(),
        )
        assert rec.strategy == "simple"
        assert "Normal conditions" in rec.reason

    def test_learning_no_strategy_stats(self):
        """Quality object with no strategy_stats should fall through."""

        class MockQuality:
            pass  # no strategy_stats attribute

        sel = ExecutionStrategySelector(
            learning_enabled=True,
            min_samples=10,
        )
        rec = sel.select(
            order_size_usd=100.0,
            depth_usd=50000.0,
            historical_quality=MockQuality(),
        )
        assert rec.strategy == "simple"


# ── Recommendation Fields ─────────────────────────────────────────


class TestRecommendationFields:
    def test_depth_and_size_recorded(self):
        """Depth and size should be recorded in recommendation."""
        sel = ExecutionStrategySelector()
        rec = sel.select(order_size_usd=250.0, depth_usd=30000.0)
        assert rec.depth_usd == 30000.0
        assert rec.order_size_usd == 250.0
        assert abs(rec.order_pct_of_depth - 250.0 / 30000.0) < 1e-6

    def test_confidence_range(self):
        """All strategies should have confidence in [0, 1]."""
        sel = ExecutionStrategySelector()
        for depth in [100, 5000, 50000]:
            for size in [10, 500, 5000]:
                rec = sel.select(order_size_usd=size, depth_usd=depth)
                assert 0.0 <= rec.confidence <= 1.0

    def test_alternative_set_for_non_simple(self):
        """Non-simple strategies should have an alternative."""
        sel = ExecutionStrategySelector(thin_depth_usd=5000.0)
        rec = sel.select(order_size_usd=100.0, depth_usd=2000.0)
        assert rec.strategy == "iceberg"
        assert rec.alternative != ""


# ── Config Integration ────────────────────────────────────────────


class TestStrategyConfig:
    def test_default_strategy_config(self):
        cfg = ExecutionConfig()
        assert cfg.auto_strategy_selection_enabled is False
        assert cfg.auto_strategy_thin_depth_usd == 5000.0
        assert cfg.auto_strategy_large_order_pct == 0.10
        assert cfg.auto_strategy_learning_enabled is False
        assert cfg.auto_strategy_min_samples == 10

    def test_custom_strategy_config(self):
        cfg = ExecutionConfig(
            auto_strategy_selection_enabled=True,
            auto_strategy_thin_depth_usd=2000.0,
            auto_strategy_large_order_pct=0.20,
            auto_strategy_learning_enabled=True,
            auto_strategy_min_samples=5,
        )
        assert cfg.auto_strategy_selection_enabled is True
        assert cfg.auto_strategy_thin_depth_usd == 2000.0
        assert cfg.auto_strategy_large_order_pct == 0.20
        assert cfg.auto_strategy_learning_enabled is True
        assert cfg.auto_strategy_min_samples == 5
