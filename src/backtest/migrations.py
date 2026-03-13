"""Database migrations for the backtest subsystem.

Separate from the live trading database — uses its own schema versioning.
"""

from __future__ import annotations

import sqlite3

from src.observability.logger import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 1

_MIGRATIONS: dict[int, list[str]] = {
    1: [
        # Schema version tracking
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );
        """,

        # Historical resolved markets scraped from Gamma API
        """
        CREATE TABLE IF NOT EXISTS historical_markets (
            condition_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            description TEXT DEFAULT '',
            category TEXT DEFAULT '',
            market_type TEXT DEFAULT '',
            resolution TEXT DEFAULT '',
            resolved_at TEXT DEFAULT '',
            created_at TEXT DEFAULT '',
            end_date TEXT DEFAULT '',
            volume_usd REAL DEFAULT 0,
            liquidity_usd REAL DEFAULT 0,
            slug TEXT DEFAULT '',
            outcomes_json TEXT DEFAULT '[]',
            final_prices_json TEXT DEFAULT '{}',
            tokens_json TEXT DEFAULT '[]',
            raw_json TEXT DEFAULT '{}',
            scraped_at TEXT DEFAULT ''
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_hist_resolved
            ON historical_markets(resolved_at);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_hist_volume
            ON historical_markets(volume_usd);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_hist_category
            ON historical_markets(category);
        """,

        # LLM response cache for deterministic replay
        """
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            market_question_hash TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            response_json TEXT DEFAULT '{}',
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            created_at TEXT DEFAULT ''
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cache_model
            ON llm_cache(model_name);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cache_question
            ON llm_cache(market_question_hash);
        """,

        # Backtest run metadata
        """
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            config_json TEXT DEFAULT '{}',
            config_diff_json TEXT DEFAULT '{}',
            start_date TEXT DEFAULT '',
            end_date TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            markets_processed INTEGER DEFAULT 0,
            markets_traded INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            brier_score REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            max_drawdown_pct REAL DEFAULT 0,
            results_json TEXT DEFAULT '{}',
            started_at TEXT DEFAULT '',
            completed_at TEXT DEFAULT '',
            duration_secs REAL DEFAULT 0
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_runs_status
            ON backtest_runs(status);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_runs_started
            ON backtest_runs(started_at);
        """,

        # Individual trades per backtest run
        """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            market_condition_id TEXT NOT NULL,
            question TEXT DEFAULT '',
            category TEXT DEFAULT '',
            direction TEXT DEFAULT '',
            model_probability REAL DEFAULT 0.5,
            implied_probability REAL DEFAULT 0.5,
            edge REAL DEFAULT 0,
            confidence_level TEXT DEFAULT 'LOW',
            stake_usd REAL DEFAULT 0,
            entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0,
            pnl REAL DEFAULT 0,
            resolution TEXT DEFAULT '',
            actual_outcome REAL DEFAULT 0,
            forecast_correct INTEGER DEFAULT 0,
            created_at TEXT DEFAULT '',
            FOREIGN KEY (run_id) REFERENCES backtest_runs(run_id)
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_bt_trades_run
            ON backtest_trades(run_id);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_bt_trades_market
            ON backtest_trades(market_condition_id);
        """,
    ],
}


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run all pending migrations on the backtest database."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)"
    )
    conn.commit()

    current = _get_current_version(conn)

    for version in sorted(_MIGRATIONS.keys()):
        if version <= current:
            continue
        log.info("backtest_migrations.running", version=version)
        for sql in _MIGRATIONS[version]:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    log.info(
                        "backtest_migrations.column_exists_skip",
                        version=version,
                        error=str(e),
                    )
                else:
                    raise
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (version,),
        )
        conn.commit()
        log.info("backtest_migrations.applied", version=version)

    final = _get_current_version(conn)
    log.info("backtest_migrations.complete", version=final)


def _get_current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return row[0] if row and row[0] else 0
    except Exception:
        return 0
