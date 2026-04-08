"""Tests for budget tracking and daily cost limits."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import pytest

from src.observability.metrics import CostTracker
from src.config import BudgetConfig


class TestDailyTracking:
    """Daily cost accumulation and day rollover."""

    def test_daily_cost_accumulates(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a")
        ct.record_call("api_a")
        assert ct.daily_cost == pytest.approx(0.02, abs=1e-6)

    def test_daily_rollover_resets(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a")
        assert ct.daily_cost == pytest.approx(0.01)
        # Simulate date change
        with ct._lock:
            ct._current_date = "2000-01-01"
        # Next call triggers rollover
        ct.record_call("api_a")
        assert ct.daily_cost == pytest.approx(0.01)  # reset + new call

    def test_total_survives_rollover(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a", count=3)
        with ct._lock:
            ct._current_date = "2000-01-01"
        ct.record_call("api_a", count=2)
        snap = ct.snapshot()
        assert snap["total_cost_usd"] == pytest.approx(0.05, abs=1e-4)
        assert snap["daily_cost_usd"] == pytest.approx(0.02, abs=1e-4)


class TestCheckBudget:
    """Budget check returns (can_spend, remaining)."""

    def test_under_limit_ok(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a")  # $0.01
        can, remaining = ct.check_budget(1.00)
        assert can is True
        assert remaining == pytest.approx(0.99, abs=0.01)

    def test_at_limit_exhausted(self) -> None:
        ct = CostTracker({"api_a": 1.00})
        ct.record_call("api_a")  # $1.00
        can, remaining = ct.check_budget(1.00)
        assert can is False
        assert remaining == 0.0

    def test_over_limit_exhausted(self) -> None:
        ct = CostTracker({"api_a": 0.60})
        ct.record_call("api_a")
        ct.record_call("api_a")  # $1.20
        can, remaining = ct.check_budget(1.00)
        assert can is False
        assert remaining == 0.0


class TestActualCostOverride:
    """actual_cost parameter overrides flat-rate estimate."""

    def test_actual_cost_used(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a", actual_cost=0.05)
        assert ct.daily_cost == pytest.approx(0.05)

    def test_actual_cost_none_falls_back(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a", actual_cost=None)
        assert ct.daily_cost == pytest.approx(0.01)


class TestSnapshotAndEndCycle:
    """Snapshot includes daily fields, end_cycle resets cycle but not daily."""

    def test_snapshot_has_daily_fields(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a")
        snap = ct.snapshot()
        assert "daily_cost_usd" in snap
        assert "daily_calls" in snap
        assert "current_date" in snap

    def test_end_cycle_resets_cycle_not_daily(self) -> None:
        ct = CostTracker({"api_a": 0.01})
        ct.record_call("api_a", count=2)
        summary = ct.end_cycle()
        assert summary["cycle_cost_usd"] == pytest.approx(0.02, abs=1e-4)
        assert summary["daily_cost_usd"] == pytest.approx(0.02, abs=1e-4)
        # After end_cycle, cycle is reset but daily is not
        snap = ct.snapshot()
        assert snap["cycle_cost_usd"] == 0.0
        assert snap["daily_cost_usd"] == pytest.approx(0.02, abs=1e-4)


class TestBudgetConfig:
    """BudgetConfig defaults."""

    def test_defaults(self) -> None:
        bc = BudgetConfig()
        assert bc.enabled is True
        assert bc.daily_limit_usd == 5.0
        assert bc.warning_pct == 0.80

    def test_config_loads(self) -> None:
        from src.config import load_config
        cfg = load_config()
        assert cfg.budget.enabled is True
        assert cfg.budget.daily_limit_usd == 15.0


class TestTokenBasedCost:
    """Token-based cost calculation."""

    def test_token_cost_gpt4o(self) -> None:
        """gpt-4o: 1000 input ($2.50e-3) + 500 output ($5.00e-3) = $0.0075."""
        ct = CostTracker()
        ct.record_call("gpt-4o", input_tokens=1000, output_tokens=500)
        assert ct.daily_cost == pytest.approx(0.0075, abs=1e-5)

    def test_token_cost_mini(self) -> None:
        """gpt-4o-mini: 1000 input ($1.5e-4) + 500 output ($3.0e-4) = $0.00045."""
        ct = CostTracker()
        ct.record_call("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert ct.daily_cost == pytest.approx(0.00045, abs=1e-6)

    def test_token_fallback_to_flat(self) -> None:
        """No tokens → falls back to flat estimate."""
        ct = CostTracker()
        ct.record_call("gpt-4o")  # no tokens
        assert ct.daily_cost == pytest.approx(0.005, abs=1e-4)

    def test_token_counters_in_snapshot(self) -> None:
        ct = CostTracker()
        ct.record_call("gpt-4o", input_tokens=1000, output_tokens=500)
        snap = ct.snapshot()
        assert snap["total_input_tokens"] == 1000
        assert snap["total_output_tokens"] == 500

    def test_actual_cost_overrides_tokens(self) -> None:
        """actual_cost takes priority over token-based calculation."""
        ct = CostTracker()
        ct.record_call("gpt-4o", actual_cost=0.99, input_tokens=1000, output_tokens=500)
        assert ct.daily_cost == pytest.approx(0.99)

    def test_unknown_model_tokens_fallback(self) -> None:
        """Unknown model with tokens but no token pricing → flat estimate."""
        ct = CostTracker()
        ct.record_call("some-unknown-model", input_tokens=1000, output_tokens=500)
        # Falls back to default flat rate of $0.001
        assert ct.daily_cost == pytest.approx(0.001, abs=1e-5)
