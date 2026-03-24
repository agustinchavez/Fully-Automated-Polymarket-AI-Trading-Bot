"""Tests for Phase 8 Batch B: Automatic strategy parameter tuning."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from src.analytics.param_optimizer import (
    OptimizationResult,
    ParameterOptimizer,
    TUNABLE_PARAMS,
)


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """In-memory DB with Phase 8 tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE param_optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'pending',
            num_perturbations INTEGER DEFAULT 0,
            best_sharpe REAL DEFAULT 0,
            baseline_sharpe REAL DEFAULT 0,
            sharpe_improvement_pct REAL DEFAULT 0,
            p_value REAL DEFAULT 1.0,
            significance TEXT DEFAULT 'none',
            best_config_json TEXT DEFAULT '{}',
            config_diff_json TEXT DEFAULT '{}',
            applied INTEGER DEFAULT 0,
            started_at TEXT,
            completed_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE param_optimization_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            config_json TEXT DEFAULT '{}',
            sharpe_ratio REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            max_drawdown_pct REAL DEFAULT 0,
            brier_score REAL DEFAULT 1.0,
            created_at TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_por_run ON param_optimization_results(run_id)"
    )
    conn.commit()
    return conn


def _make_baseline_config() -> dict[str, float]:
    """Default tunable parameter values."""
    return {
        "min_edge": 0.04,
        "kelly_fraction": 0.25,
        "min_evidence_quality": 0.55,
        "stop_loss_pct": 0.20,
        "take_profit_pct": 0.30,
        "max_stake_per_market": 50.0,
    }


# ── OptimizationResult ──────────────────────────────────────────


class TestOptimizationResult:
    def test_defaults(self):
        r = OptimizationResult()
        assert r.status == "pending"
        assert r.run_id == ""
        assert r.significance == "none"
        assert r.results == []

    def test_to_dict(self):
        r = OptimizationResult(
            run_id="test-1", status="completed", best_sharpe=1.5
        )
        d = r.to_dict()
        assert d["run_id"] == "test-1"
        assert d["status"] == "completed"
        assert d["best_sharpe"] == 1.5

    def test_status_values(self):
        for s in ("pending", "completed", "failed", "no_improvement"):
            r = OptimizationResult(status=s)
            assert r.status == s


# ── Generate Perturbations ───────────────────────────────────────


class TestGeneratePerturbations:
    def test_correct_count(self):
        config = _make_baseline_config()
        perturbations = ParameterOptimizer.generate_perturbations(config, n=20)
        assert len(perturbations) == 20

    def test_within_range(self):
        config = _make_baseline_config()
        perturbations = ParameterOptimizer.generate_perturbations(
            config, n=50, range_pct=0.20, seed=42,
        )
        for p in perturbations:
            for name, value in p.items():
                if name in TUNABLE_PARAMS:
                    _, _, param_min, param_max = TUNABLE_PARAMS[name]
                    assert value >= param_min, f"{name}={value} < {param_min}"
                    assert value <= param_max, f"{name}={value} > {param_max}"

    def test_clamped_to_bounds(self):
        # Set config at extremes
        config = {
            "min_edge": 0.01,  # at minimum
            "kelly_fraction": 0.49,  # near maximum
            "min_evidence_quality": 0.55,
            "stop_loss_pct": 0.20,
            "take_profit_pct": 0.30,
            "max_stake_per_market": 50.0,
        }
        perturbations = ParameterOptimizer.generate_perturbations(
            config, n=100, range_pct=0.50, seed=42,
        )
        for p in perturbations:
            assert p["min_edge"] >= 0.01
            assert p["kelly_fraction"] <= 0.50

    def test_reproducible_seed(self):
        config = _make_baseline_config()
        p1 = ParameterOptimizer.generate_perturbations(config, n=10, seed=42)
        p2 = ParameterOptimizer.generate_perturbations(config, n=10, seed=42)
        assert p1 == p2

    def test_all_params_varied(self):
        config = _make_baseline_config()
        perturbations = ParameterOptimizer.generate_perturbations(
            config, n=50, seed=42,
        )
        for name in TUNABLE_PARAMS:
            values = {p[name] for p in perturbations}
            # With 50 perturbations, should have at least 2 unique values
            assert len(values) > 1, f"Param {name} not varied"


# ── Extract Current Params ───────────────────────────────────────


class TestExtractCurrentParams:
    def test_reads_from_config(self):
        from src.config import BotConfig
        cfg = BotConfig()
        params = ParameterOptimizer.extract_current_params(cfg)
        assert "min_edge" in params
        assert "kelly_fraction" in params
        assert params["min_edge"] == 0.04
        assert params["kelly_fraction"] == 0.25

    def test_defaults(self):
        from src.config import BotConfig
        params = ParameterOptimizer.extract_current_params(BotConfig())
        assert params["stop_loss_pct"] == 0.20
        assert params["take_profit_pct"] == 0.30

    def test_custom_values(self):
        from src.config import BotConfig, RiskConfig
        cfg = BotConfig(risk=RiskConfig(
            min_edge=0.08, kelly_fraction=0.10,
            max_stake_per_market=100.0,
        ))
        params = ParameterOptimizer.extract_current_params(cfg)
        assert params["min_edge"] == 0.08
        assert params["kelly_fraction"] == 0.10
        assert params["max_stake_per_market"] == 100.0


# ── Config Diff ──────────────────────────────────────────────────


class TestConfigDiff:
    def test_computes_diff(self):
        baseline = {"min_edge": 0.04, "kelly_fraction": 0.25}
        best = {"min_edge": 0.05, "kelly_fraction": 0.25}
        diff = ParameterOptimizer.compute_config_diff(baseline, best)
        assert "min_edge" in diff
        assert "kelly_fraction" not in diff
        assert diff["min_edge"]["baseline"] == 0.04
        assert diff["min_edge"]["optimized"] == 0.05
        assert diff["min_edge"]["change_pct"] == 25.0

    def test_no_diff(self):
        config = _make_baseline_config()
        diff = ParameterOptimizer.compute_config_diff(config, config)
        assert diff == {}


# ── Significance Level ───────────────────────────────────────────


class TestSignificanceLevel:
    def test_strong(self):
        assert ParameterOptimizer.significance_level(0.005) == "strong"

    def test_moderate(self):
        assert ParameterOptimizer.significance_level(0.03) == "moderate"

    def test_weak(self):
        assert ParameterOptimizer.significance_level(0.08) == "weak"

    def test_none(self):
        assert ParameterOptimizer.significance_level(0.15) == "none"


# ── Paired T-Test ────────────────────────────────────────────────


class TestPairedTTest:
    def test_identical_values(self):
        p = ParameterOptimizer.paired_ttest([1, 2, 3, 4], [1, 2, 3, 4])
        assert p == 1.0

    def test_clear_improvement(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        b = [5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1]
        p = ParameterOptimizer.paired_ttest(a, b)
        assert p < 0.05

    def test_insufficient_data(self):
        p = ParameterOptimizer.paired_ttest([1], [2])
        assert p == 1.0

    def test_no_difference(self):
        import random
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(50)]
        b = [rng.gauss(0, 1) for _ in range(50)]
        p = ParameterOptimizer.paired_ttest(a, b)
        # Should not be highly significant for independent noise
        assert p > 0.01


# ── DB Persistence ───────────────────────────────────────────────


class TestDBPersistence:
    def test_save_run(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        result = OptimizationResult(
            run_id="test-1",
            status="completed",
            best_sharpe=1.5,
            baseline_sharpe=1.0,
            sharpe_improvement_pct=50.0,
        )
        optimizer.save_run(result)
        row = conn.execute(
            "SELECT * FROM param_optimization_runs WHERE run_id = 'test-1'"
        ).fetchone()
        assert row is not None
        assert row["status"] == "completed"
        assert float(row["best_sharpe"]) == 1.5

    def test_save_perturbation_result(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        optimizer.save_perturbation_result(
            run_id="test-1",
            config={"min_edge": 0.05},
            sharpe=1.2,
            total_pnl=100.0,
        )
        row = conn.execute(
            "SELECT * FROM param_optimization_results WHERE run_id = 'test-1'"
        ).fetchone()
        assert row is not None
        assert float(row["sharpe_ratio"]) == 1.2

    def test_upsert_run(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        r1 = OptimizationResult(run_id="test-1", status="pending")
        optimizer.save_run(r1)
        r2 = OptimizationResult(run_id="test-1", status="completed", best_sharpe=2.0)
        optimizer.save_run(r2)
        rows = conn.execute(
            "SELECT * FROM param_optimization_runs WHERE run_id = 'test-1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "completed"


# ── Pending Suggestions ─────────────────────────────────────────


class TestPendingSuggestions:
    def test_returns_unapplied(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        r = OptimizationResult(
            run_id="test-1", status="completed",
            significance="moderate", sharpe_improvement_pct=15.0,
            best_config={"min_edge": 0.05},
            config_diff={"min_edge": {"baseline": 0.04, "optimized": 0.05}},
        )
        optimizer.save_run(r)
        suggestions = optimizer.get_pending_suggestions()
        assert len(suggestions) == 1
        assert suggestions[0]["run_id"] == "test-1"

    def test_filters_insignificant(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        r = OptimizationResult(
            run_id="test-1", status="completed",
            significance="none", sharpe_improvement_pct=5.0,
        )
        optimizer.save_run(r)
        suggestions = optimizer.get_pending_suggestions()
        assert len(suggestions) == 0

    def test_filters_applied(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        r = OptimizationResult(
            run_id="test-1", status="completed",
            significance="strong", sharpe_improvement_pct=20.0,
        )
        optimizer.save_run(r)
        optimizer.apply_suggestion("test-1")
        suggestions = optimizer.get_pending_suggestions()
        assert len(suggestions) == 0

    def test_empty(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        assert optimizer.get_pending_suggestions() == []


# ── Apply Suggestion ─────────────────────────────────────────────


class TestApplySuggestion:
    def test_marks_applied(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        r = OptimizationResult(run_id="test-1", status="completed")
        optimizer.save_run(r)
        success = optimizer.apply_suggestion("test-1")
        assert success is True
        row = conn.execute(
            "SELECT applied FROM param_optimization_runs WHERE run_id = 'test-1'"
        ).fetchone()
        assert row["applied"] == 1

    def test_invalid_run_id(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        # Should not crash; just returns True (no matching row but no error)
        success = optimizer.apply_suggestion("nonexistent")
        assert success is True  # SQL UPDATE with no matching rows is not an error


# ── All Runs ─────────────────────────────────────────────────────


class TestGetAllRuns:
    def test_returns_all(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        for i in range(3):
            optimizer.save_run(OptimizationResult(
                run_id=f"test-{i}", status="completed",
            ))
        runs = optimizer.get_all_runs()
        assert len(runs) == 3

    def test_empty(self):
        conn = _create_test_db()
        optimizer = ParameterOptimizer(conn)
        assert optimizer.get_all_runs() == []


# ── Dashboard Endpoints ──────────────────────────────────────────


def _create_dashboard_test_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


class TestDashboardEndpoints:
    @pytest.fixture
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_runs_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/param-optimization/runs")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["runs"] == []

    def test_suggestions_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/param-optimization/suggestions")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["suggestions"] == []

    def test_apply_endpoint(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            optimizer = ParameterOptimizer(conn)
            optimizer.save_run(OptimizationResult(
                run_id="test-1", status="completed",
            ))
            mock.return_value = conn
            resp = client.post("/api/param-optimization/apply/test-1")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["success"] is True

    def test_runs_no_table(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            mock.return_value = conn
            resp = client.get("/api/param-optimization/runs")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["runs"] == []
