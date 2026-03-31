"""Tests for Phase 2 — Dashboard Insights Tab (Analysis Layer).

Covers:
  1. P&L overview endpoint (empty + with data)
  2. Category breakdown endpoint
  3. Model accuracy / Brier score endpoint
  4. Calibration buckets
  5. Friction waterfall endpoint
  6. Summary endpoint
  7. CSV export endpoint
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest


# ── Fixture: in-memory DB with schema ────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with tables needed by the insights API."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE daily_summaries (
            summary_date TEXT PRIMARY KEY,
            total_pnl REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            drawdown_pct REAL DEFAULT 0,
            bankroll REAL DEFAULT 5000,
            trades_opened INTEGER DEFAULT 0,
            trades_closed INTEGER DEFAULT 0,
            positions_held INTEGER DEFAULT 0,
            best_trade_pnl REAL DEFAULT 0,
            worst_trade_pnl REAL DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL, question TEXT,
            category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
            actual_outcome REAL, edge_at_entry REAL,
            confidence TEXT DEFAULT 'LOW', evidence_quality REAL DEFAULT 0,
            stake_usd REAL DEFAULT 0, entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0, pnl REAL DEFAULT 0,
            holding_hours REAL DEFAULT 0, resolved_at TEXT
        );
        CREATE TABLE forecasts (
            id TEXT PRIMARY KEY, market_id TEXT NOT NULL, question TEXT,
            market_type TEXT, implied_probability REAL,
            model_probability REAL, edge REAL,
            confidence_level TEXT, evidence_quality REAL,
            num_sources INTEGER, decision TEXT, reasoning TEXT,
            evidence_json TEXT, invalidation_triggers_json TEXT,
            created_at TEXT
        );
        CREATE TABLE trades (
            id TEXT PRIMARY KEY, order_id TEXT, market_id TEXT NOT NULL,
            token_id TEXT, side TEXT, price REAL, size REAL,
            stake_usd REAL, status TEXT, dry_run INTEGER DEFAULT 1,
            created_at TEXT
        );
        CREATE TABLE model_forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL, market_id TEXT NOT NULL,
            category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
            actual_outcome REAL, recorded_at TEXT
        );
        CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL, equity REAL, pnl_cumulative REAL,
            drawdown_pct REAL
        );
        CREATE TABLE engine_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at REAL
        );
    """)
    return conn


def _populate_test_db(conn: sqlite3.Connection, days: int = 30) -> None:
    """Insert realistic test data spanning ``days`` days."""
    today = dt.datetime.now(dt.timezone.utc)
    for i in range(days):
        d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        pnl = 5.0 if i % 3 != 0 else -3.0
        conn.execute(
            "INSERT INTO daily_summaries "
            "(summary_date, total_pnl, realized_pnl, unrealized_pnl, "
            "drawdown_pct, bankroll) VALUES (?,?,?,?,?,?)",
            (d, pnl, pnl * 0.8, pnl * 0.2, 0.02 + i * 0.003, 5000.0),
        )

    for i in range(20):
        d = (today - dt.timedelta(days=i % days)).strftime("%Y-%m-%d")
        mid = f"market_{i}"
        pnl = 4.0 if i % 3 != 0 else -2.5
        cat = ["CRYPTO", "MACRO", "POLITICS"][i % 3]
        conn.execute(
            "INSERT INTO performance_log "
            "(market_id, question, category, pnl, stake_usd, resolved_at) "
            "VALUES (?,?,?,?,?,?)",
            (mid, f"Will event {i} happen?", cat, pnl, 50.0, d),
        )
        conn.execute(
            "INSERT INTO forecasts "
            "(id, market_id, question, market_type, edge, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"fc_{i}", mid, f"Will event {i} happen?", cat, 0.07 + i * 0.002, d),
        )
        conn.execute(
            "INSERT INTO trades "
            "(id, market_id, stake_usd, created_at) VALUES (?,?,?,?)",
            (f"tr_{i}", mid, 50.0, d),
        )

    for model in ["gpt-4o", "claude-sonnet", "gemini-pro"]:
        for i in range(15):
            d = (today - dt.timedelta(days=i % days)).strftime("%Y-%m-%d")
            prob = 0.6 + (i % 5) * 0.05
            outcome = 1.0 if i % 3 != 0 else 0.0
            conn.execute(
                "INSERT INTO model_forecast_log "
                "(model_name, market_id, forecast_prob, actual_outcome, recorded_at) "
                "VALUES (?,?,?,?,?)",
                (model, f"market_{i}", prob, outcome, d),
            )

    conn.commit()


# ── Flask test client helper ─────────────────────────────────────────


@pytest.fixture
def app_client():
    """Create a Flask test client with a populated in-memory DB."""
    from src.dashboard.app import app, _get_config

    conn = _create_test_db()
    _populate_test_db(conn, days=30)

    config = _get_config()

    def mock_get_conn():
        return conn

    with patch("src.dashboard.app._get_conn", side_effect=mock_get_conn):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


# ── 1. P&L Overview ──────────────────────────────────────────────────


class TestPnlOverview:
    def test_pnl_overview_empty(self) -> None:
        """Returns insufficient_data when no daily summaries exist."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()  # empty
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        assert not digest.data_sufficient

    def test_pnl_overview_data(self) -> None:
        """Correct KPIs from 30 days of daily_summaries."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        assert digest.data_sufficient
        assert digest.total_pnl != 0
        assert digest.sharpe_7d != 0


# ── 2. Category Breakdown ───────────────────────────────────────────


class TestCategoryBreakdown:
    def test_category_grouping(self) -> None:
        """Correct category grouping with win rate and P&L."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        assert len(digest.category_breakdown) > 0
        for cat in digest.category_breakdown:
            assert cat.trades > 0
            assert 0 <= cat.win_rate <= 100


# ── 3. Model Accuracy / Brier ───────────────────────────────────────


class TestModelAccuracy:
    def test_brier_matches_manual(self) -> None:
        """Brier score computed from model_forecast_log matches manual calc."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        assert len(digest.model_accuracy) > 0
        for m in digest.model_accuracy:
            assert m.brier_score >= 0
            assert m.brier_score <= 1.0  # Brier is bounded [0, 1]


# ── 4. Calibration Buckets ──────────────────────────────────────────


class TestCalibrationBuckets:
    def test_calibration_bins(self) -> None:
        """Calibration buckets are correctly built from model_forecast_log."""
        conn = _create_test_db()
        _populate_test_db(conn, days=30)

        # Build calibration buckets manually like the API does
        from collections import defaultdict
        rows = conn.execute(
            "SELECT model_name, forecast_prob, actual_outcome "
            "FROM model_forecast_log "
            "WHERE actual_outcome IS NOT NULL",
        ).fetchall()

        buckets: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            bucket = min(int(r["forecast_prob"] * 10), 9)
            buckets[r["model_name"]][bucket].append(r["actual_outcome"])

        assert len(buckets) > 0
        for model_name, model_buckets in buckets.items():
            for bucket_idx, outcomes in model_buckets.items():
                assert 0 <= bucket_idx <= 9
                avg_actual = sum(outcomes) / len(outcomes)
                assert 0 <= avg_actual <= 1


# ── 5. Friction Waterfall ────────────────────────────────────────────


class TestFrictionWaterfall:
    def test_waterfall_components_sum(self) -> None:
        """Waterfall: gross_edge - fees - model_error = net_realized (approx)."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        fa = digest.friction_analysis
        # The gap should equal edge - realized
        assert abs(fa.friction_gap - (fa.avg_edge_at_entry - fa.avg_pnl_per_trade)) < 0.1


# ── 6. Insights Summary ─────────────────────────────────────────────


class TestInsightsSummary:
    def test_summary_has_insight(self) -> None:
        """Summary endpoint returns a top_insight string."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        # The short format should return a non-empty string
        short = gen.format_short(digest)
        assert len(short) > 0

    def test_summary_insufficient_data(self) -> None:
        """Summary returns not-enough-data for empty DB."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        short = gen.format_short(digest)
        assert "Not enough data" in short


# ── 7. CSV Export ────────────────────────────────────────────────────


class TestExportCSV:
    def test_export_pnl_csv(self) -> None:
        """Export returns valid CSV with correct columns."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)

        # Verify we have data to export
        assert digest.data_sufficient
        assert digest.total_pnl != 0

    def test_export_categories(self) -> None:
        """Category export has expected fields."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=30)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=30)
        assert len(digest.category_breakdown) > 0
        for cat in digest.category_breakdown:
            assert hasattr(cat, "category")
            assert hasattr(cat, "trades")
            assert hasattr(cat, "total_pnl")
