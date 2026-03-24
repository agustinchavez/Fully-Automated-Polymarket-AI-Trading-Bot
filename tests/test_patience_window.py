"""Tests for Phase 6 Batch B: Smart entry patience window."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.analytics.smart_entry import (
    PatienceResult,
    SmartEntryCalculator,
)
from src.config import ExecutionConfig


# ── PatienceResult ───────────────────────────────────────────────


class TestPatienceResult:
    def test_defaults(self):
        pr = PatienceResult(
            action="enter_immediately",
            wait_time_secs=0.0,
            entry_price=0.55,
            edge_at_entry=0.12,
        )
        assert pr.action == "enter_immediately"
        assert pr.edge_trajectory == []
        assert pr.reason == ""

    def test_to_dict(self):
        pr = PatienceResult(
            action="cancelled",
            wait_time_secs=15.3,
            entry_price=0.0,
            edge_at_entry=0.02,
            edge_trajectory=[0.05, 0.03, 0.02],
            reason="Edge dropped",
        )
        d = pr.to_dict()
        assert d["action"] == "cancelled"
        assert d["wait_time_secs"] == 15.3
        assert d["entry_price"] == 0.0
        assert len(d["edge_trajectory"]) == 3
        assert d["reason"] == "Edge dropped"


# ── patience_monitor ─────────────────────────────────────────────


class TestPatienceMonitor:
    def _make_calculator(self) -> SmartEntryCalculator:
        return SmartEntryCalculator()

    @pytest.fixture
    def calc(self):
        return self._make_calculator()

    @pytest.mark.asyncio
    async def test_enter_immediately_large_edge(self, calc):
        """Edge > 2 * min_edge should enter immediately."""
        get_price = AsyncMock(return_value=0.50)
        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.12,  # 3x min_edge
            get_current_price=get_price,
            get_model_prob=lambda: 0.62,
            immediate_multiplier=2.0,
        )
        assert result.action == "enter_immediately"
        assert result.wait_time_secs == 0.0
        assert result.entry_price == 0.50
        assert result.edge_at_entry == 0.12

    @pytest.mark.asyncio
    async def test_cancel_edge_deteriorates(self, calc):
        """Edge dropping below min_edge should cancel."""
        prices = iter([0.55, 0.60, 0.65])  # price moving against us

        async def get_price():
            return next(prices)

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,  # near threshold, not immediate
            get_current_price=get_price,
            get_model_prob=lambda: 0.62,
            max_wait_secs=1,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        assert result.action == "cancelled"
        assert result.entry_price == 0.0

    @pytest.mark.asyncio
    async def test_timeout_with_valid_edge(self, calc):
        """Should enter at timeout if edge is still valid."""
        get_price = AsyncMock(return_value=0.55)

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,
            get_current_price=get_price,
            get_model_prob=lambda: 0.60,
            max_wait_secs=0.3,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        assert result.action == "timeout"
        assert result.entry_price == 0.55
        assert result.wait_time_secs >= 0.3

    @pytest.mark.asyncio
    async def test_wait_then_enter_improved_edge(self, calc):
        """Edge improving to > 2*min_edge after wait should enter."""
        call_count = 0

        async def get_price():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return 0.55  # initial: edge = 0.62-0.55 = 0.07
            return 0.45  # improved: edge = 0.62-0.45 = 0.17

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.07,  # above min but below 2x
            get_current_price=get_price,
            get_model_prob=lambda: 0.62,
            max_wait_secs=2,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        assert result.action == "entered_after_wait"
        assert result.wait_time_secs > 0
        assert result.entry_price == 0.45

    @pytest.mark.asyncio
    async def test_edge_trajectory_tracked(self, calc):
        """Edge values should be tracked over time."""
        get_price = AsyncMock(return_value=0.55)

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,
            get_current_price=get_price,
            get_model_prob=lambda: 0.60,
            max_wait_secs=0.25,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        # Initial edge + at least 1 check
        assert len(result.edge_trajectory) >= 2
        assert result.edge_trajectory[0] == 0.05

    @pytest.mark.asyncio
    async def test_callback_error_handling(self, calc):
        """Price callback errors should not crash the monitor."""
        calls = [0]

        async def flaky_price():
            calls[0] += 1
            if calls[0] <= 2:
                raise ConnectionError("API down")
            return 0.55

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,
            get_current_price=flaky_price,
            get_model_prob=lambda: 0.60,
            max_wait_secs=0.5,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        # Should complete without crashing
        assert result.action in ("timeout", "cancelled", "entered_after_wait")

    @pytest.mark.asyncio
    async def test_zero_min_edge(self, calc):
        """Zero min_edge: immediate check is skipped, monitors until timeout."""
        get_price = AsyncMock(return_value=0.50)
        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.0,
            current_edge=0.01,
            get_current_price=get_price,
            get_model_prob=lambda: 0.51,
            max_wait_secs=0.3,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        # With min_edge=0, immediate multiplier check is skipped.
        # Edge stays positive, so should timeout with valid entry.
        assert result.action in ("timeout", "entered_after_wait")
        assert result.entry_price == 0.50

    @pytest.mark.asyncio
    async def test_negative_edge_cancelled(self, calc):
        """Negative edge (below min_edge) should cancel on first check."""
        get_price = AsyncMock(return_value=0.70)

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,  # initially above
            get_current_price=get_price,  # 0.60-0.70=-0.10 on recheck
            get_model_prob=lambda: 0.60,
            max_wait_secs=1,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        assert result.action == "cancelled"

    @pytest.mark.asyncio
    async def test_buy_no_direction(self, calc):
        """BUY_NO should compute edge correctly."""
        get_price = AsyncMock(return_value=0.30)  # YES price = 0.30, NO price = 0.70

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_NO",
            min_edge=0.04,
            current_edge=0.15,  # large edge
            get_current_price=get_price,
            get_model_prob=lambda: 0.25,  # model thinks YES is 25%, so NO is 75%
            max_wait_secs=1,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        assert result.action == "enter_immediately"

    @pytest.mark.asyncio
    async def test_wait_time_within_bounds(self, calc):
        """Wait time should not exceed max_wait_secs significantly."""
        get_price = AsyncMock(return_value=0.55)

        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.05,
            get_current_price=get_price,
            get_model_prob=lambda: 0.60,
            max_wait_secs=0.5,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        # Allow some tolerance for sleep imprecision
        assert result.wait_time_secs <= 1.0


# ── Integration ──────────────────────────────────────────────────


class TestPatienceMonitorIntegration:
    @pytest.mark.asyncio
    async def test_config_gated_disabled(self):
        """When disabled, patience should not run (tested via engine)."""
        cfg = ExecutionConfig()
        assert cfg.patience_window_enabled is False
        assert cfg.patience_window_max_secs == 300
        assert cfg.patience_check_interval_secs == 5

    @pytest.mark.asyncio
    async def test_config_gated_enabled(self):
        """Custom patience config should load correctly."""
        cfg = ExecutionConfig(
            patience_window_enabled=True,
            patience_window_max_secs=60,
            patience_check_interval_secs=2,
            edge_immediate_multiplier=3.0,
            edge_deterioration_cancel=False,
        )
        assert cfg.patience_window_enabled is True
        assert cfg.patience_window_max_secs == 60
        assert cfg.edge_immediate_multiplier == 3.0

    @pytest.mark.asyncio
    async def test_patience_with_smart_entry_plan(self):
        """Patience monitor should work after calculate_entry."""
        calc = SmartEntryCalculator()
        plan = calc.calculate_entry(
            market_id="test",
            side="BUY_YES",
            current_price=0.55,
            fair_value=0.62,
            edge=0.07,
        )
        assert plan.recommended_price > 0

        # Now run patience monitor
        get_price = AsyncMock(return_value=0.55)
        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.12,  # large edge
            get_current_price=get_price,
            get_model_prob=lambda: 0.62,
            immediate_multiplier=2.0,
        )
        assert result.action == "enter_immediately"

    @pytest.mark.asyncio
    async def test_edge_recalculation_uses_new_price(self):
        """Edge should be recalculated with the fetched price."""
        prices = iter([0.56, 0.57, 0.58, 0.60, 0.62])

        async def get_price():
            try:
                return next(prices)
            except StopIteration:
                return 0.62

        calc = SmartEntryCalculator()
        result = await calc.patience_monitor(
            market_id="test",
            side="BUY_YES",
            min_edge=0.04,
            current_edge=0.06,  # 0.62-0.56=0.06
            get_current_price=get_price,
            get_model_prob=lambda: 0.62,
            max_wait_secs=0.5,
            check_interval_secs=0.1,
            immediate_multiplier=2.0,
        )
        # Price moves from 0.56 to 0.62, edge shrinks to 0
        assert result.action == "cancelled"
        assert len(result.edge_trajectory) > 1


# ── Config ───────────────────────────────────────────────────────


class TestPatienceConfig:
    def test_default_patience_config(self):
        cfg = ExecutionConfig()
        assert cfg.patience_window_enabled is False
        assert cfg.patience_window_max_secs == 300
        assert cfg.patience_check_interval_secs == 5
        assert cfg.edge_immediate_multiplier == 2.0
        assert cfg.edge_deterioration_cancel is True

    def test_custom_patience_config(self):
        cfg = ExecutionConfig(
            patience_window_enabled=True,
            patience_window_max_secs=120,
            patience_check_interval_secs=10,
            edge_immediate_multiplier=1.5,
            edge_deterioration_cancel=False,
        )
        assert cfg.patience_window_enabled is True
        assert cfg.patience_window_max_secs == 120
        assert cfg.edge_immediate_multiplier == 1.5
