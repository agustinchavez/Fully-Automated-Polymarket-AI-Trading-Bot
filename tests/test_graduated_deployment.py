"""Tests for Phase 9 Batch C: Graduated deployment."""

from __future__ import annotations

import sqlite3

import pytest

from src.config import BotConfig, ProductionConfig
from src.policy.graduated_deployment import (
    DeploymentStage,
    GraduatedDeploymentManager,
    STAGES,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**prod_kwargs) -> BotConfig:
    return BotConfig(production=ProductionConfig(enabled=True, **prod_kwargs))


def _make_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


# ── DeploymentStage ──────────────────────────────────────────────


class TestDeploymentStage:
    def test_defaults(self):
        stage = DeploymentStage(name="test", bankroll=100.0, max_stake=5.0)
        assert stage.min_duration_days == 7
        assert stage.criteria == ""

    def test_custom_fields(self):
        stage = DeploymentStage(
            name="week2", bankroll=500.0, max_stake=25.0,
            min_duration_days=14, criteria="Must be profitable",
        )
        assert stage.min_duration_days == 14
        assert "profitable" in stage.criteria


# ── Stage Config ─────────────────────────────────────────────────


class TestStageConfig:
    def test_paper_stage(self):
        cfg = _make_config(deployment_stage="paper")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "paper"
        assert sc.bankroll == cfg.risk.bankroll

    def test_week1_stage(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "week1"
        assert sc.bankroll == 100.0
        assert sc.max_stake == 5.0

    def test_week2_stage(self):
        cfg = _make_config(deployment_stage="week2")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "week2"
        assert sc.bankroll == 500.0
        assert sc.max_stake == 25.0

    def test_week3_4_stage(self):
        cfg = _make_config(deployment_stage="week3_4")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "week3_4"
        assert sc.bankroll == 2000.0
        assert sc.max_stake == 50.0

    def test_month2_plus_stage(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "month2_plus"
        assert sc.bankroll == 2000.0  # base = week3_4

    def test_unknown_stage_defaults(self):
        cfg = _make_config(deployment_stage="unknown_stage")
        mgr = GraduatedDeploymentManager(cfg)
        sc = mgr.get_stage_config()
        assert sc.name == "unknown_stage"
        assert sc.bankroll == cfg.risk.bankroll


# ── Stage Advancement ────────────────────────────────────────────


class TestStageAdvancement:
    def test_paper_no_advance(self):
        cfg = _make_config(deployment_stage="paper")
        mgr = GraduatedDeploymentManager(cfg)
        assert mgr.check_advancement(100.0, 1.5, 30) is None

    def test_week1_advance_on_good_performance(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        # Loss is within 10% threshold
        result = mgr.check_advancement(-5.0, 0.0, 7)
        assert result == "week2"

    def test_week1_no_advance_too_early(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(10.0, 0.0, 3)
        assert result is None

    def test_week1_no_advance_large_loss(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        # Loss > 10% of $100 bankroll
        result = mgr.check_advancement(-15.0, 0.0, 7)
        assert result is None

    def test_week2_advance_positive_pnl(self):
        cfg = _make_config(deployment_stage="week2")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(50.0, 0.0, 7)
        assert result == "week3_4"

    def test_week2_no_advance_negative_pnl(self):
        cfg = _make_config(deployment_stage="week2")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(-10.0, 0.0, 7)
        assert result is None

    def test_week3_4_advance(self):
        cfg = _make_config(deployment_stage="week3_4")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(100.0, 1.5, 14)
        assert result == "month2_plus"

    def test_week3_4_no_advance_too_early(self):
        cfg = _make_config(deployment_stage="week3_4")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(100.0, 1.5, 10)
        assert result is None

    def test_month2_plus_no_advance(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        result = mgr.check_advancement(1000.0, 2.0, 90)
        assert result is None


# ── Apply Stage Limits ───────────────────────────────────────────


class TestApplyStageLimits:
    def test_returns_stage_limits(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll, max_stake = mgr.apply_stage_limits()
        assert bankroll == 100.0
        assert max_stake == 5.0

    def test_disabled_returns_config_defaults(self):
        cfg = BotConfig(production=ProductionConfig(enabled=False))
        mgr = GraduatedDeploymentManager(cfg)
        bankroll, max_stake = mgr.apply_stage_limits()
        assert bankroll == cfg.risk.bankroll
        assert max_stake == cfg.risk.max_stake_per_market

    def test_week2_limits(self):
        cfg = _make_config(deployment_stage="week2")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll, max_stake = mgr.apply_stage_limits()
        assert bankroll == 500.0
        assert max_stake == 25.0


# ── Month2+ Sharpe Scaling ───────────────────────────────────────


class TestMonth2SharpeScaling:
    def test_positive_sharpe_scales(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll = mgr.get_effective_bankroll(sharpe=2.0)
        base = 2000.0
        expected = base + (2.0 * base * 0.5)
        assert bankroll == expected

    def test_zero_sharpe_returns_base(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll = mgr.get_effective_bankroll(sharpe=0.0)
        assert bankroll == 2000.0

    def test_capped_at_5x(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll = mgr.get_effective_bankroll(sharpe=100.0)
        assert bankroll == 2000.0 * 5

    def test_negative_sharpe_returns_base(self):
        cfg = _make_config(deployment_stage="month2_plus")
        mgr = GraduatedDeploymentManager(cfg)
        bankroll = mgr.get_effective_bankroll(sharpe=-1.0)
        assert bankroll == 2000.0


# ── Stage Advance & History (DB) ─────────────────────────────────


class TestStageAdvanceDB:
    def test_advance_persists(self):
        cfg = _make_config(deployment_stage="week1")
        conn = _make_db()
        mgr = GraduatedDeploymentManager(cfg, conn)
        mgr.advance_stage("week2", cumulative_pnl=10.0, sharpe=0.5)

        assert cfg.production.deployment_stage == "week2"
        rows = conn.execute("SELECT * FROM deployment_stages").fetchall()
        assert len(rows) >= 1
        assert rows[0]["stage"] == "week2"

    def test_advance_without_db(self):
        cfg = _make_config(deployment_stage="week1")
        mgr = GraduatedDeploymentManager(cfg)
        # Should not crash
        mgr.advance_stage("week2")
        assert cfg.production.deployment_stage == "week2"

    def test_stage_history_empty(self):
        cfg = _make_config()
        mgr = GraduatedDeploymentManager(cfg)
        assert mgr.get_stage_history() == []

    def test_stage_history_returns_rows(self):
        cfg = _make_config(deployment_stage="week1")
        conn = _make_db()
        mgr = GraduatedDeploymentManager(cfg, conn)
        mgr.advance_stage("week2", cumulative_pnl=10.0)
        mgr.advance_stage("week3_4", cumulative_pnl=50.0)

        history = mgr.get_stage_history()
        assert len(history) >= 2


# ── STAGES constant ──────────────────────────────────────────────


class TestStagesConstant:
    def test_stages_list(self):
        assert STAGES == ["paper", "week1", "week2", "week3_4", "month2_plus"]

    def test_all_stages_have_config(self):
        cfg = _make_config()
        mgr = GraduatedDeploymentManager(cfg)
        for stage in STAGES:
            sc = mgr.get_stage_config(stage)
            assert sc.name == stage
            assert sc.bankroll > 0
            assert sc.max_stake > 0
