"""Tests for backtest dashboard API endpoints (Phase 1 — Batch 5)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.backtest.models import (
    BacktestRunRecord,
    BacktestTradeRecord,
    HistoricalMarketRecord,
)


@pytest.fixture
def app():
    """Flask test app with mocked config."""
    from src.dashboard.app import app as flask_app

    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _mock_db():
    """Create a mock BacktestDatabase."""
    return MagicMock()


class TestBacktestRunsEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_list_runs_empty(self, mock_get_db, client) -> None:
        db = _mock_db()
        db.get_backtest_runs.return_value = []
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["runs"] == []
        assert data["count"] == 0

    @patch("src.dashboard.app._get_backtest_db")
    def test_list_runs_with_data(self, mock_get_db, client) -> None:
        db = _mock_db()
        run = BacktestRunRecord(
            run_id="abc", name="test", status="completed",
            total_pnl=100.0, brier_score=0.2, started_at="2024-01-01T00:00:00",
        )
        db.get_backtest_runs.return_value = [run]
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        assert data["runs"][0]["run_id"] == "abc"

    @patch("src.dashboard.app._get_backtest_db")
    def test_list_runs_with_limit(self, mock_get_db, client) -> None:
        db = _mock_db()
        db.get_backtest_runs.return_value = []
        mock_get_db.return_value = db

        client.get("/api/backtest/runs?limit=10")
        db.get_backtest_runs.assert_called_once_with(limit=10)


class TestBacktestRunDetailEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_run_detail_found(self, mock_get_db, client) -> None:
        db = _mock_db()
        run = BacktestRunRecord(
            run_id="abc", name="test", status="completed",
            started_at="2024-01-01T00:00:00",
        )
        db.get_backtest_run.return_value = run
        db.get_backtest_trades.return_value = []
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs/abc")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["run"]["run_id"] == "abc"
        assert data["trades_count"] == 0

    @patch("src.dashboard.app._get_backtest_db")
    def test_run_detail_not_found(self, mock_get_db, client) -> None:
        db = _mock_db()
        db.get_backtest_run.return_value = None
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs/missing")
        assert resp.status_code == 404


class TestBacktestTradesEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_paginated_trades(self, mock_get_db, client) -> None:
        db = _mock_db()
        trades = [
            BacktestTradeRecord(
                run_id="abc", market_condition_id=f"m{i}",
                pnl=float(i), created_at="2024-01-01",
            )
            for i in range(5)
        ]
        db.get_backtest_trades.return_value = trades
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs/abc/trades?limit=2&offset=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 5
        assert len(data["trades"]) == 2
        assert data["offset"] == 1


class TestBacktestCalibrationEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_calibration_data(self, mock_get_db, client) -> None:
        db = _mock_db()
        results = {
            "calibration_buckets": [
                {"bin_start": 0.0, "bin_end": 0.1, "avg_forecast": 0.05, "actual_rate": 0.03, "count": 10},
            ],
        }
        run = BacktestRunRecord(
            run_id="abc", name="test", status="completed",
            brier_score=0.2, results_json=json.dumps(results),
            started_at="2024-01-01T00:00:00",
        )
        db.get_backtest_run.return_value = run
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs/abc/calibration")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["brier_score"] == 0.2
        assert len(data["calibration_buckets"]) == 1

    @patch("src.dashboard.app._get_backtest_db")
    def test_calibration_not_found(self, mock_get_db, client) -> None:
        db = _mock_db()
        db.get_backtest_run.return_value = None
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/runs/missing/calibration")
        assert resp.status_code == 404


class TestBacktestCompareEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_compare_missing_params(self, mock_get_db, client) -> None:
        resp = client.get("/api/backtest/compare")
        assert resp.status_code == 400

    @patch("src.dashboard.app._get_backtest_db")
    def test_compare_success(self, mock_get_db, client) -> None:
        from src.backtest.comparator import ComparisonResult

        db = _mock_db()
        run_a = BacktestRunRecord(
            run_id="aaa", name="A", status="completed",
            total_pnl=100.0, started_at="2024-01-01T00:00:00",
        )
        run_b = BacktestRunRecord(
            run_id="bbb", name="B", status="completed",
            total_pnl=200.0, started_at="2024-01-01T00:00:00",
        )
        db.get_backtest_run.side_effect = lambda rid: run_a if rid == "aaa" else run_b
        db.get_backtest_trades.return_value = []
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/compare?a=aaa&b=bbb")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "deltas" in data
        assert data["deltas"]["pnl"] == 100.0


class TestBacktestCacheStatsEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_cache_stats(self, mock_get_db, client) -> None:
        db = _mock_db()
        db.get_llm_cache_stats.return_value = {
            "total_entries": 100,
            "distinct_models": 3,
        }
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/cache-stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_entries"] == 100


class TestBacktestMarketsEndpoint:

    @patch("src.dashboard.app._get_backtest_db")
    def test_list_markets(self, mock_get_db, client) -> None:
        db = _mock_db()
        market = HistoricalMarketRecord(
            condition_id="cond1", question="Will it rain?",
            resolution="YES", volume_usd=5000.0,
        )
        db.get_historical_markets.return_value = [market]
        db.count_historical_markets.return_value = 42
        mock_get_db.return_value = db

        resp = client.get("/api/backtest/markets?limit=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 1
        assert data["total"] == 42
        assert data["markets"][0]["condition_id"] == "cond1"
