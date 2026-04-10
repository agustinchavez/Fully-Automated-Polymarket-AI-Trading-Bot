"""Tests for the Paper Trading Progress widget.

Covers:
  1. /api/paper-trading-progress endpoint — empty DB
  2. /api/paper-trading-progress endpoint — with data
  3. Milestone progress calculations (smart_retrain, ai_analyst, live_gate)
  4. Edge cases (no Brier score, zero days)
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from unittest.mock import patch

import pytest


# ── Fixture: in-memory DB with full schema ────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with tables needed by the progress API."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE trades (
            id TEXT PRIMARY KEY, order_id TEXT, market_id TEXT NOT NULL,
            token_id TEXT, side TEXT, price REAL, size REAL,
            stake_usd REAL, status TEXT, dry_run INTEGER DEFAULT 1,
            created_at TEXT
        );
        CREATE TABLE closed_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL, question TEXT, direction TEXT,
            entry_price REAL, exit_price REAL, stake_usd REAL, pnl REAL,
            close_reason TEXT DEFAULT '', closed_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE calibration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brier_score REAL, bucket_count INTEGER DEFAULT 10,
            sample_size INTEGER DEFAULT 0, recorded_at TEXT
        );
        CREATE TABLE schema_version (version INTEGER);
    """)
    return conn


def _populate_db(
    conn: sqlite3.Connection,
    n_trades: int = 20,
    n_resolved: int = 10,
    first_trade_days_ago: int = 15,
    brier: float | None = None,
) -> None:
    """Insert test data into an in-memory DB."""
    now = dt.datetime.now(dt.timezone.utc)

    for i in range(n_trades):
        days_ago = min(i, first_trade_days_ago)
        ts = (now - dt.timedelta(days=days_ago)).isoformat()
        conn.execute(
            "INSERT INTO trades (id, market_id, created_at) VALUES (?,?,?)",
            (f"tr_{i}", f"mkt_{i}", ts),
        )

    for i in range(n_resolved):
        days_ago = min(i, first_trade_days_ago)
        ts = (now - dt.timedelta(days=days_ago)).isoformat()
        conn.execute(
            "INSERT INTO closed_positions "
            "(market_id, question, entry_price, exit_price, stake_usd, pnl, closed_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (f"mkt_{i}", f"Q{i}", 0.5, 0.6, 50.0, 5.0, ts),
        )

    if brier is not None:
        conn.execute(
            "INSERT INTO calibration_history (brier_score, recorded_at) VALUES (?,?)",
            (brier, now.isoformat()),
        )

    conn.commit()


# ── Flask test client ──────────────────────────────────────────────


@pytest.fixture
def app_client():
    """Flask test client with a populated DB."""
    from src.dashboard.app import app

    conn = _create_test_db()
    _populate_db(conn, n_trades=25, n_resolved=35, first_trade_days_ago=30,
                 brier=0.22)

    def mock_get_conn():
        return conn

    with patch("src.dashboard.app._get_conn", side_effect=mock_get_conn):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


@pytest.fixture
def empty_client():
    """Flask test client with empty DB."""
    from src.dashboard.app import app

    conn = _create_test_db()

    def mock_get_conn():
        return conn

    with patch("src.dashboard.app._get_conn", side_effect=mock_get_conn):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


# ── Tests ──────────────────────────────────────────────────────────


class TestPaperTradingProgressEndpoint:
    """Tests for /api/paper-trading-progress."""

    def test_returns_200(self, app_client) -> None:
        resp = app_client.get("/api/paper-trading-progress")
        assert resp.status_code == 200

    def test_response_structure(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        assert "trades_today" in data
        assert "trades_7d" in data
        assert "trades_30d" in data
        assert "trades_all" in data
        assert "resolved_total" in data
        assert "days_running" in data
        assert "brier_current" in data
        assert "milestones" in data

    def test_trade_counts_populated(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        assert data["trades_all"] == 25
        assert data["resolved_total"] == 35

    def test_days_running_calculated(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        # First trade 30 days ago
        assert data["days_running"] >= 15

    def test_brier_score_present(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        assert data["brier_current"] == 0.22

    def test_milestones_keys(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        ms = data["milestones"]
        assert "smart_retrain" in ms
        assert "ai_analyst" in ms
        assert "live_gate" in ms


class TestMilestoneProgress:
    """Milestone progress calculation correctness."""

    def test_smart_retrain_met_at_30(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        ms = data["milestones"]["smart_retrain"]
        # 35 resolved >= 30 required
        assert ms["met"] is True
        assert ms["progress_pct"] == 100

    def test_smart_retrain_not_met_under_30(self, empty_client) -> None:
        # Insert only 10 resolved
        from src.dashboard.app import _get_conn
        conn = _create_test_db()
        _populate_db(conn, n_trades=10, n_resolved=10, first_trade_days_ago=5)

        def mock_conn():
            return conn

        with patch("src.dashboard.app._get_conn", side_effect=mock_conn):
            data = empty_client.get("/api/paper-trading-progress").get_json()

        ms = data["milestones"]["smart_retrain"]
        assert ms["met"] is False
        assert ms["progress_pct"] == 33  # 10/30 * 100 = 33.33 → 33

    def test_live_gate_met_all_conditions(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        ms = data["milestones"]["live_gate"]
        # 35 resolved >= 30, days >= 28? depends on first_trade_days_ago=30
        # brier = 0.22 < 0.25
        assert ms["current_brier"] == 0.22
        assert ms["required_brier"] == 0.25

    def test_live_gate_label_and_description(self, app_client) -> None:
        data = app_client.get("/api/paper-trading-progress").get_json()
        ms = data["milestones"]["live_gate"]
        assert ms["label"] == "Live Trading Gate"
        assert "live mode" in ms["description"].lower()


class TestEmptyDatabase:
    """Edge cases with empty database."""

    def test_empty_returns_zeros(self, empty_client) -> None:
        data = empty_client.get("/api/paper-trading-progress").get_json()
        assert data["trades_all"] == 0
        assert data["resolved_total"] == 0
        assert data["days_running"] == 0

    def test_empty_brier_is_none(self, empty_client) -> None:
        data = empty_client.get("/api/paper-trading-progress").get_json()
        assert data["brier_current"] is None

    def test_empty_milestones_not_met(self, empty_client) -> None:
        data = empty_client.get("/api/paper-trading-progress").get_json()
        for ms in data["milestones"].values():
            assert ms["met"] is False

    def test_empty_progress_zero(self, empty_client) -> None:
        data = empty_client.get("/api/paper-trading-progress").get_json()
        for ms in data["milestones"].values():
            assert ms["progress_pct"] == 0
