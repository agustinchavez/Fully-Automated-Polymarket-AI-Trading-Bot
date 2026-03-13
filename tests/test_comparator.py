"""Tests for A/B backtest comparator (Phase 1 — Batch 5)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.backtest.comparator import (
    BacktestComparator,
    ComparisonResult,
    _diff_configs,
    _paired_ttest,
    _significance_level,
)
from src.backtest.database import BacktestDatabase
from src.backtest.models import BacktestRunRecord, BacktestTradeRecord


@pytest.fixture
def db() -> BacktestDatabase:
    d = BacktestDatabase(":memory:")
    d.connect()
    yield d
    d.close()


def _make_run(run_id: str, name: str = "", **kwargs: object) -> BacktestRunRecord:
    defaults = {
        "run_id": run_id,
        "name": name or f"run-{run_id}",
        "status": "completed",
        "started_at": "2024-01-01T00:00:00",
    }
    defaults.update(kwargs)
    return BacktestRunRecord(**defaults)


def _make_trade(
    run_id: str, market_id: str, pnl: float,
    category: str = "crypto", direction: str = "BUY_YES",
    model_prob: float = 0.7, resolution: str = "YES",
) -> BacktestTradeRecord:
    return BacktestTradeRecord(
        run_id=run_id,
        market_condition_id=market_id,
        question=f"Test market {market_id}?",
        category=category,
        direction=direction,
        model_probability=model_prob,
        implied_probability=0.5,
        edge=0.1,
        confidence_level="MEDIUM",
        stake_usd=100.0,
        entry_price=0.5,
        exit_price=1.0 if pnl > 0 else 0.0,
        pnl=pnl,
        resolution=resolution,
        actual_outcome=1.0 if resolution == "YES" else 0.0,
        forecast_correct=pnl > 0,
        created_at="2024-06-01T00:00:00",
    )


class TestPairedTtest:

    def test_identical_values(self) -> None:
        """Identical paired values → p = 1.0."""
        p = _paired_ttest([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert p == 1.0

    def test_small_sample(self) -> None:
        """Single observation → p = 1.0 (too few)."""
        p = _paired_ttest([1.0], [2.0])
        assert p == 1.0

    def test_large_difference(self) -> None:
        """Large consistent difference → small p-value."""
        a = [1.0] * 20
        b = [10.0] * 20
        p = _paired_ttest(a, b)
        assert p < 0.01

    def test_no_difference_noisy(self) -> None:
        """Mixed differences → large p-value."""
        a = [1.0, -1.0, 1.0, -1.0, 1.0]
        b = [-1.0, 1.0, -1.0, 1.0, -1.0]
        p = _paired_ttest(a, b)
        assert p > 0.05

    def test_constant_difference(self) -> None:
        """Constant non-zero difference with zero variance → p = 0."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        p = _paired_ttest(a, b)
        assert p == 0.0


class TestSignificanceLevel:

    def test_strong(self) -> None:
        assert _significance_level(0.005) == "strong"

    def test_moderate(self) -> None:
        assert _significance_level(0.03) == "moderate"

    def test_weak(self) -> None:
        assert _significance_level(0.08) == "weak"

    def test_none(self) -> None:
        assert _significance_level(0.5) == "none"


class TestDiffConfigs:

    def test_empty_configs(self) -> None:
        assert _diff_configs("{}", "{}") == {}

    def test_same_configs(self) -> None:
        c = json.dumps({"a": 1, "b": 2})
        assert _diff_configs(c, c) == {}

    def test_different_values(self) -> None:
        a = json.dumps({"edge": 0.04, "bankroll": 10000})
        b = json.dumps({"edge": 0.06, "bankroll": 10000})
        diff = _diff_configs(a, b)
        assert "edge" in diff
        assert diff["edge"]["a"] == 0.04
        assert diff["edge"]["b"] == 0.06
        assert "bankroll" not in diff

    def test_invalid_json(self) -> None:
        assert _diff_configs("not json", "{}") == {}


class TestBacktestComparator:

    def test_compare_basic(self, db: BacktestDatabase) -> None:
        """Basic comparison produces correct deltas."""
        run_a = _make_run("aaa", total_pnl=100.0, brier_score=0.25,
                          sharpe_ratio=1.0, win_rate=0.6, max_drawdown_pct=0.1)
        run_b = _make_run("bbb", total_pnl=200.0, brier_score=0.15,
                          sharpe_ratio=1.5, win_rate=0.7, max_drawdown_pct=0.05)
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")

        assert result.pnl_delta == pytest.approx(100.0)
        assert result.brier_delta == pytest.approx(-0.10, abs=0.001)
        assert result.sharpe_delta == pytest.approx(0.5)
        assert result.win_rate_delta == pytest.approx(0.1)

    def test_compare_run_not_found(self, db: BacktestDatabase) -> None:
        """Raises ValueError for missing run."""
        run_a = _make_run("aaa")
        db.insert_backtest_run(run_a)

        comp = BacktestComparator(db)
        with pytest.raises(ValueError, match="not found"):
            comp.compare("aaa", "missing")

    def test_overlapping_markets_paired_test(self, db: BacktestDatabase) -> None:
        """Paired t-test is performed on overlapping markets."""
        run_a = _make_run("aaa", total_pnl=50.0)
        run_b = _make_run("bbb", total_pnl=100.0)
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        # Both runs trade same 5 markets, B is consistently better
        for i in range(5):
            db.insert_backtest_trade(_make_trade("aaa", f"m{i}", pnl=10.0))
            db.insert_backtest_trade(_make_trade("bbb", f"m{i}", pnl=20.0))

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")
        assert result.overlapping_markets == 5
        assert result.p_value < 0.05
        assert result.significance in ("moderate", "strong")

    def test_no_overlapping_markets(self, db: BacktestDatabase) -> None:
        """No overlap → p=1.0, significance=none."""
        run_a = _make_run("aaa")
        run_b = _make_run("bbb")
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        db.insert_backtest_trade(_make_trade("aaa", "m1", pnl=10.0))
        db.insert_backtest_trade(_make_trade("bbb", "m2", pnl=20.0))

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")
        assert result.overlapping_markets == 0
        assert result.p_value == 1.0
        assert result.significance == "none"

    def test_category_comparison(self, db: BacktestDatabase) -> None:
        """Per-category comparison includes all categories from both runs."""
        run_a = _make_run("aaa")
        run_b = _make_run("bbb")
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        db.insert_backtest_trade(_make_trade("aaa", "m1", pnl=10.0, category="crypto"))
        db.insert_backtest_trade(_make_trade("aaa", "m2", pnl=-5.0, category="politics"))
        db.insert_backtest_trade(_make_trade("bbb", "m1", pnl=20.0, category="crypto"))

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")

        cats = {c["category"] for c in result.category_comparison}
        assert "crypto" in cats
        assert "politics" in cats

    def test_to_dict(self, db: BacktestDatabase) -> None:
        """ComparisonResult.to_dict() is JSON-serializable."""
        run_a = _make_run("aaa", total_pnl=50.0)
        run_b = _make_run("bbb", total_pnl=100.0)
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")
        d = result.to_dict()
        # Should be JSON serializable
        serialized = json.dumps(d)
        assert "run_a" in serialized
        assert "deltas" in serialized

    def test_config_diff_included(self, db: BacktestDatabase) -> None:
        """Config differences are included when configs differ."""
        config_a = json.dumps({"risk": {"min_edge": 0.04}})
        config_b = json.dumps({"risk": {"min_edge": 0.06}})
        run_a = _make_run("aaa", config_json=config_a)
        run_b = _make_run("bbb", config_json=config_b)
        db.insert_backtest_run(run_a)
        db.insert_backtest_run(run_b)

        comp = BacktestComparator(db)
        result = comp.compare("aaa", "bbb")
        assert "risk" in result.config_diff
