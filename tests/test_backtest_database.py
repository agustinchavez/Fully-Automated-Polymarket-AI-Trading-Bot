"""Tests for backtest database CRUD operations (Phase 1)."""

from __future__ import annotations

import sqlite3

import pytest

from src.backtest.database import BacktestDatabase
from src.backtest.models import (
    BacktestRunRecord,
    BacktestTradeRecord,
    HistoricalMarketRecord,
    LLMCacheRecord,
)


@pytest.fixture
def db() -> BacktestDatabase:
    """In-memory backtest database for testing."""
    bdb = BacktestDatabase(db_path=":memory:")
    bdb.connect()
    yield bdb
    bdb.close()


# ── Migrations ────────────────────────────────────────────────────────


class TestMigrations:

    def test_tables_created(self, db: BacktestDatabase) -> None:
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "historical_markets" in tables
        assert "llm_cache" in tables
        assert "backtest_runs" in tables
        assert "backtest_trades" in tables
        assert "schema_version" in tables

    def test_schema_version(self, db: BacktestDatabase) -> None:
        from src.backtest.migrations import SCHEMA_VERSION
        row = db.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        assert row[0] == SCHEMA_VERSION

    def test_idempotent_migrations(self, db: BacktestDatabase) -> None:
        """Running migrations again should not error."""
        from src.backtest.migrations import run_migrations, SCHEMA_VERSION
        run_migrations(db.conn)  # second run
        row = db.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        assert row[0] == SCHEMA_VERSION


# ── Historical Markets ────────────────────────────────────────────────


class TestHistoricalMarkets:

    def _make_market(self, cid: str = "0x1", **kwargs) -> HistoricalMarketRecord:
        defaults = dict(
            condition_id=cid,
            question="Will X happen?",
            resolution="YES",
            volume_usd=5000.0,
            resolved_at="2024-06-15T00:00:00Z",
        )
        defaults.update(kwargs)
        return HistoricalMarketRecord(**defaults)

    def test_upsert_and_get(self, db: BacktestDatabase) -> None:
        market = self._make_market()
        db.upsert_historical_market(market)
        result = db.get_historical_market("0x1")
        assert result is not None
        assert result.question == "Will X happen?"
        assert result.resolution == "YES"

    def test_upsert_idempotent(self, db: BacktestDatabase) -> None:
        market = self._make_market()
        db.upsert_historical_market(market)
        db.upsert_historical_market(market)
        assert db.count_historical_markets() == 1

    def test_upsert_updates_fields(self, db: BacktestDatabase) -> None:
        m1 = self._make_market(volume_usd=1000.0)
        db.upsert_historical_market(m1)
        m2 = self._make_market(volume_usd=9999.0)
        db.upsert_historical_market(m2)
        result = db.get_historical_market("0x1")
        assert result is not None
        assert result.volume_usd == 9999.0

    def test_get_nonexistent(self, db: BacktestDatabase) -> None:
        assert db.get_historical_market("nonexistent") is None

    def test_count(self, db: BacktestDatabase) -> None:
        assert db.count_historical_markets() == 0
        db.upsert_historical_market(self._make_market("a"))
        db.upsert_historical_market(self._make_market("b"))
        assert db.count_historical_markets() == 2

    def test_query_all(self, db: BacktestDatabase) -> None:
        for i in range(5):
            db.upsert_historical_market(
                self._make_market(f"m{i}", resolved_at=f"2024-0{i+1}-01T00:00:00Z")
            )
        results = db.get_historical_markets()
        assert len(results) == 5

    def test_query_date_range(self, db: BacktestDatabase) -> None:
        db.upsert_historical_market(
            self._make_market("early", resolved_at="2024-01-01T00:00:00Z")
        )
        db.upsert_historical_market(
            self._make_market("mid", resolved_at="2024-06-15T00:00:00Z")
        )
        db.upsert_historical_market(
            self._make_market("late", resolved_at="2024-12-01T00:00:00Z")
        )
        results = db.get_historical_markets(
            start_date="2024-05-01", end_date="2024-07-01",
        )
        assert len(results) == 1
        assert results[0].condition_id == "mid"

    def test_query_min_volume(self, db: BacktestDatabase) -> None:
        db.upsert_historical_market(self._make_market("low", volume_usd=100.0))
        db.upsert_historical_market(self._make_market("high", volume_usd=50000.0))
        results = db.get_historical_markets(min_volume=1000.0)
        assert len(results) == 1
        assert results[0].condition_id == "high"

    def test_query_category(self, db: BacktestDatabase) -> None:
        db.upsert_historical_market(self._make_market("a", category="Crypto"))
        db.upsert_historical_market(self._make_market("b", category="Politics"))
        results = db.get_historical_markets(category="Crypto")
        assert len(results) == 1

    def test_query_limit_offset(self, db: BacktestDatabase) -> None:
        for i in range(10):
            db.upsert_historical_market(
                self._make_market(f"m{i:02d}", resolved_at=f"2024-01-{i+1:02d}T00:00:00Z")
            )
        page1 = db.get_historical_markets(limit=3, offset=0)
        page2 = db.get_historical_markets(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].condition_id != page2[0].condition_id


# ── LLM Cache ─────────────────────────────────────────────────────────


class TestLLMCache:

    def _make_cache(self, key: str = "k1", **kwargs) -> LLMCacheRecord:
        defaults = dict(
            cache_key=key,
            market_question_hash="qhash",
            model_name="gpt-4o",
            prompt_hash="phash",
            response_json='{"model_probability": 0.7}',
        )
        defaults.update(kwargs)
        return LLMCacheRecord(**defaults)

    def test_upsert_and_get(self, db: BacktestDatabase) -> None:
        record = self._make_cache()
        db.upsert_llm_cache(record)
        result = db.get_llm_cache("k1")
        assert result is not None
        assert result.model_name == "gpt-4o"

    def test_cache_miss(self, db: BacktestDatabase) -> None:
        assert db.get_llm_cache("nonexistent") is None

    def test_upsert_overwrites(self, db: BacktestDatabase) -> None:
        db.upsert_llm_cache(self._make_cache(response_json='{"p": 0.5}'))
        db.upsert_llm_cache(self._make_cache(response_json='{"p": 0.9}'))
        result = db.get_llm_cache("k1")
        assert result is not None
        assert "0.9" in result.response_json

    def test_stats(self, db: BacktestDatabase) -> None:
        db.upsert_llm_cache(self._make_cache("k1", model_name="gpt-4o"))
        db.upsert_llm_cache(self._make_cache("k2", model_name="claude"))
        stats = db.get_llm_cache_stats()
        assert stats["total_entries"] == 2
        assert stats["distinct_models"] == 2


# ── Backtest Runs ─────────────────────────────────────────────────────


class TestBacktestRuns:

    def _make_run(self, run_id: str = "run-001", **kwargs) -> BacktestRunRecord:
        defaults = dict(
            run_id=run_id,
            name="test run",
            status="pending",
        )
        defaults.update(kwargs)
        return BacktestRunRecord(**defaults)

    def test_insert_and_get(self, db: BacktestDatabase) -> None:
        run = self._make_run()
        db.insert_backtest_run(run)
        result = db.get_backtest_run("run-001")
        assert result is not None
        assert result.name == "test run"

    def test_get_nonexistent(self, db: BacktestDatabase) -> None:
        assert db.get_backtest_run("nope") is None

    def test_update(self, db: BacktestDatabase) -> None:
        db.insert_backtest_run(self._make_run())
        db.update_backtest_run("run-001", {
            "status": "completed",
            "total_pnl": 42.5,
            "brier_score": 0.22,
        })
        result = db.get_backtest_run("run-001")
        assert result is not None
        assert result.status == "completed"
        assert result.total_pnl == 42.5

    def test_update_empty_dict(self, db: BacktestDatabase) -> None:
        db.insert_backtest_run(self._make_run())
        db.update_backtest_run("run-001", {})  # should not error

    def test_list_runs(self, db: BacktestDatabase) -> None:
        db.insert_backtest_run(self._make_run("r1", started_at="2024-01-01"))
        db.insert_backtest_run(self._make_run("r2", started_at="2024-06-01"))
        runs = db.get_backtest_runs()
        assert len(runs) == 2
        # Most recent first
        assert runs[0].run_id == "r2"

    def test_list_runs_limit(self, db: BacktestDatabase) -> None:
        for i in range(5):
            db.insert_backtest_run(self._make_run(f"r{i}"))
        runs = db.get_backtest_runs(limit=2)
        assert len(runs) == 2

    def test_insert_returns_run_id(self, db: BacktestDatabase) -> None:
        rid = db.insert_backtest_run(self._make_run("my-id"))
        assert rid == "my-id"


# ── Backtest Trades ───────────────────────────────────────────────────


class TestBacktestTrades:

    def _setup_run(self, db: BacktestDatabase) -> None:
        db.insert_backtest_run(
            BacktestRunRecord(run_id="run-001", name="test")
        )

    def test_insert_and_get(self, db: BacktestDatabase) -> None:
        self._setup_run(db)
        trade = BacktestTradeRecord(
            run_id="run-001",
            market_condition_id="0xabc",
            direction="BUY_YES",
            pnl=50.0,
            forecast_correct=True,
        )
        db.insert_backtest_trade(trade)
        trades = db.get_backtest_trades("run-001")
        assert len(trades) == 1
        assert trades[0].direction == "BUY_YES"
        assert trades[0].forecast_correct is True

    def test_multiple_trades(self, db: BacktestDatabase) -> None:
        self._setup_run(db)
        for i in range(5):
            db.insert_backtest_trade(
                BacktestTradeRecord(
                    run_id="run-001",
                    market_condition_id=f"0x{i}",
                    pnl=float(i * 10),
                )
            )
        trades = db.get_backtest_trades("run-001")
        assert len(trades) == 5

    def test_empty_trades(self, db: BacktestDatabase) -> None:
        self._setup_run(db)
        trades = db.get_backtest_trades("run-001")
        assert trades == []

    def test_trades_isolated_by_run(self, db: BacktestDatabase) -> None:
        db.insert_backtest_run(BacktestRunRecord(run_id="r1", name="a"))
        db.insert_backtest_run(BacktestRunRecord(run_id="r2", name="b"))
        db.insert_backtest_trade(
            BacktestTradeRecord(run_id="r1", market_condition_id="m1")
        )
        db.insert_backtest_trade(
            BacktestTradeRecord(run_id="r2", market_condition_id="m2")
        )
        assert len(db.get_backtest_trades("r1")) == 1
        assert len(db.get_backtest_trades("r2")) == 1

    def test_forecast_correct_bool_roundtrip(self, db: BacktestDatabase) -> None:
        """forecast_correct stored as int in SQLite, converted back to bool."""
        self._setup_run(db)
        db.insert_backtest_trade(
            BacktestTradeRecord(
                run_id="run-001",
                market_condition_id="m1",
                forecast_correct=True,
            )
        )
        db.insert_backtest_trade(
            BacktestTradeRecord(
                run_id="run-001",
                market_condition_id="m2",
                forecast_correct=False,
            )
        )
        trades = db.get_backtest_trades("run-001")
        assert trades[0].forecast_correct is True
        assert trades[1].forecast_correct is False


# ── Connection Errors ─────────────────────────────────────────────────


class TestConnectionErrors:

    def test_conn_property_raises_before_connect(self) -> None:
        bdb = BacktestDatabase(db_path=":memory:")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = bdb.conn


# ── Config ────────────────────────────────────────────────────────────


class TestBacktestConfig:

    def test_default_config_valid(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from src.config import load_config
            cfg = load_config()
            assert cfg.backtest.db_path == "data/backtest.db"
            assert cfg.backtest.cache_llm_responses is True
            assert cfg.backtest.default_implied_prob == 0.5
