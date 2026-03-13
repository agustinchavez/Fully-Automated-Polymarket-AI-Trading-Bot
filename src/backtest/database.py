"""Backtest database — separate SQLite persistence for historical data.

Isolates backtest data from the live trading database to prevent
contamination. Follows the same patterns as src/storage/database.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.backtest.migrations import run_migrations
from src.backtest.models import (
    BacktestRunRecord,
    BacktestTradeRecord,
    HistoricalMarketRecord,
    LLMCacheRecord,
)
from src.observability.logger import get_logger

log = get_logger(__name__)


class BacktestDatabase:
    """SQLite database for the backtesting subsystem."""

    def __init__(self, db_path: str = "data/backtest.db"):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open database connection and run migrations."""
        path = Path(self._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        run_migrations(self._conn)
        log.info("backtest_db.connected", path=str(path))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Backtest database not connected. Call connect() first.")
        return self._conn

    # ── Historical Markets ────────────────────────────────────────────

    def upsert_historical_market(self, record: HistoricalMarketRecord) -> None:
        """Insert or update a historical market record."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO historical_markets
                (condition_id, question, description, category, market_type,
                 resolution, resolved_at, created_at, end_date,
                 volume_usd, liquidity_usd, slug,
                 outcomes_json, final_prices_json, tokens_json,
                 raw_json, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.condition_id, record.question, record.description,
                record.category, record.market_type, record.resolution,
                record.resolved_at, record.created_at, record.end_date,
                record.volume_usd, record.liquidity_usd, record.slug,
                record.outcomes_json, record.final_prices_json,
                record.tokens_json, record.raw_json, record.scraped_at,
            ),
        )
        self.conn.commit()

    def get_historical_market(
        self, condition_id: str,
    ) -> HistoricalMarketRecord | None:
        """Fetch a single historical market by condition_id."""
        row = self.conn.execute(
            "SELECT * FROM historical_markets WHERE condition_id = ?",
            (condition_id,),
        ).fetchone()
        if not row:
            return None
        return HistoricalMarketRecord(**dict(row))

    def get_historical_markets(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        min_volume: float = 0.0,
        category: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[HistoricalMarketRecord]:
        """Query historical markets with optional filters."""
        conditions: list[str] = []
        params: list[Any] = []

        if start_date:
            conditions.append("resolved_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("resolved_at <= ?")
            params.append(end_date)
        if min_volume > 0:
            conditions.append("volume_usd >= ?")
            params.append(min_volume)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM historical_markets
            {where}
            ORDER BY resolved_at ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self.conn.execute(query, params).fetchall()
        return [HistoricalMarketRecord(**dict(r)) for r in rows]

    def count_historical_markets(self) -> int:
        """Count total historical markets in the database."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM historical_markets"
        ).fetchone()
        return row[0] if row else 0

    # ── LLM Cache ─────────────────────────────────────────────────────

    def upsert_llm_cache(self, record: LLMCacheRecord) -> None:
        """Insert or update a cached LLM response."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
                (cache_key, market_question_hash, model_name, prompt_hash,
                 response_json, input_tokens, output_tokens, latency_ms,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.cache_key, record.market_question_hash,
                record.model_name, record.prompt_hash,
                record.response_json, record.input_tokens,
                record.output_tokens, record.latency_ms,
                record.created_at,
            ),
        )
        self.conn.commit()

    def get_llm_cache(self, cache_key: str) -> LLMCacheRecord | None:
        """Fetch a cached LLM response by cache key."""
        row = self.conn.execute(
            "SELECT * FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        return LLMCacheRecord(**dict(row))

    def get_llm_cache_stats(self) -> dict[str, Any]:
        """Get aggregate cache statistics."""
        row = self.conn.execute(
            "SELECT COUNT(*) as total, COUNT(DISTINCT model_name) as models "
            "FROM llm_cache"
        ).fetchone()
        return {
            "total_entries": row[0] if row else 0,
            "distinct_models": row[1] if row else 0,
        }

    # ── Backtest Runs ─────────────────────────────────────────────────

    def insert_backtest_run(self, record: BacktestRunRecord) -> str:
        """Insert a new backtest run. Returns the run_id."""
        self.conn.execute(
            """
            INSERT INTO backtest_runs
                (run_id, name, config_json, config_diff_json,
                 start_date, end_date, status,
                 markets_processed, markets_traded, total_pnl,
                 brier_score, win_rate, sharpe_ratio, max_drawdown_pct,
                 results_json, started_at, completed_at, duration_secs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id, record.name, record.config_json,
                record.config_diff_json, record.start_date, record.end_date,
                record.status, record.markets_processed, record.markets_traded,
                record.total_pnl, record.brier_score, record.win_rate,
                record.sharpe_ratio, record.max_drawdown_pct,
                record.results_json, record.started_at, record.completed_at,
                record.duration_secs,
            ),
        )
        self.conn.commit()
        return record.run_id

    def update_backtest_run(self, run_id: str, updates: dict[str, Any]) -> None:
        """Update fields on an existing backtest run."""
        if not updates:
            return
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [run_id]
        self.conn.execute(
            f"UPDATE backtest_runs SET {set_clauses} WHERE run_id = ?",
            values,
        )
        self.conn.commit()

    def get_backtest_runs(self, limit: int = 50) -> list[BacktestRunRecord]:
        """List backtest runs, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM backtest_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [BacktestRunRecord(**dict(r)) for r in rows]

    def get_backtest_run(self, run_id: str) -> BacktestRunRecord | None:
        """Fetch a single backtest run by ID."""
        row = self.conn.execute(
            "SELECT * FROM backtest_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return BacktestRunRecord(**dict(row))

    # ── Backtest Trades ───────────────────────────────────────────────

    def insert_backtest_trade(self, record: BacktestTradeRecord) -> None:
        """Insert a single trade for a backtest run."""
        self.conn.execute(
            """
            INSERT INTO backtest_trades
                (run_id, market_condition_id, question, category,
                 direction, model_probability, implied_probability,
                 edge, confidence_level, stake_usd,
                 entry_price, exit_price, pnl,
                 resolution, actual_outcome, forecast_correct,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.run_id, record.market_condition_id, record.question,
                record.category, record.direction, record.model_probability,
                record.implied_probability, record.edge,
                record.confidence_level, record.stake_usd,
                record.entry_price, record.exit_price, record.pnl,
                record.resolution, record.actual_outcome,
                int(record.forecast_correct), record.created_at,
            ),
        )
        self.conn.commit()

    def get_backtest_trades(
        self, run_id: str,
    ) -> list[BacktestTradeRecord]:
        """Fetch all trades for a backtest run."""
        rows = self.conn.execute(
            "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        results: list[BacktestTradeRecord] = []
        for r in rows:
            d = dict(r)
            # SQLite stores bool as int
            d["forecast_correct"] = bool(d.get("forecast_correct", 0))
            # Remove autoincrement 'id' not in model
            d.pop("id", None)
            results.append(BacktestTradeRecord(**d))
        return results
