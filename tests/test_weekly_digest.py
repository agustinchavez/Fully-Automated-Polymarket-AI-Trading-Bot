"""Tests for Phase 1 — Weekly Digest (Analysis Layer).

Covers:
  1. WeeklyDigestGenerator data queries
  2. Digest formatting (Telegram, short)
  3. Message splitting
  4. Telegram bot commands (/weekly, /report, /insights, /models)
  5. Config loading (DigestConfig)
  6. APScheduler wiring
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixture: in-memory DB with schema ────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with the tables needed by the digest."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE daily_summaries (
            summary_date TEXT PRIMARY KEY,
            total_pnl REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            trades_opened INTEGER DEFAULT 0,
            trades_closed INTEGER DEFAULT 0,
            positions_held INTEGER DEFAULT 0,
            drawdown_pct REAL DEFAULT 0,
            bankroll REAL DEFAULT 5000,
            best_trade_pnl REAL DEFAULT 0,
            worst_trade_pnl REAL DEFAULT 0,
            created_at TEXT
        );

        CREATE TABLE performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            question TEXT,
            category TEXT DEFAULT 'UNKNOWN',
            forecast_prob REAL,
            actual_outcome REAL,
            edge_at_entry REAL,
            confidence TEXT DEFAULT 'LOW',
            evidence_quality REAL DEFAULT 0,
            stake_usd REAL DEFAULT 0,
            entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0,
            pnl REAL DEFAULT 0,
            holding_hours REAL DEFAULT 0,
            resolved_at TEXT
        );

        CREATE TABLE forecasts (
            id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            question TEXT,
            market_type TEXT,
            implied_probability REAL,
            model_probability REAL,
            edge REAL,
            confidence_level TEXT,
            evidence_quality REAL,
            num_sources INTEGER,
            decision TEXT,
            reasoning TEXT,
            evidence_json TEXT,
            invalidation_triggers_json TEXT,
            created_at TEXT
        );

        CREATE TABLE trades (
            id TEXT PRIMARY KEY,
            order_id TEXT,
            market_id TEXT NOT NULL,
            token_id TEXT,
            side TEXT,
            price REAL,
            size REAL,
            stake_usd REAL,
            status TEXT,
            dry_run INTEGER DEFAULT 1,
            created_at TEXT
        );

        CREATE TABLE model_forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            market_id TEXT NOT NULL,
            category TEXT DEFAULT 'UNKNOWN',
            forecast_prob REAL,
            actual_outcome REAL,
            recorded_at TEXT
        );
    """)
    return conn


def _populate_test_db(conn: sqlite3.Connection, days: int = 7) -> None:
    """Insert realistic test data spanning ``days`` days."""
    today = dt.datetime.now(dt.timezone.utc)

    # daily_summaries
    for i in range(days):
        d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        pnl = 5.0 if i % 3 != 0 else -3.0
        conn.execute(
            "INSERT INTO daily_summaries "
            "(summary_date, total_pnl, realized_pnl, unrealized_pnl, "
            "drawdown_pct, bankroll) VALUES (?,?,?,?,?,?)",
            (d, pnl, pnl * 0.8, pnl * 0.2, 0.02 + i * 0.005, 5000.0),
        )

    # performance_log + forecasts + trades
    for i in range(15):
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

    # model_forecast_log — 12 forecasts per model
    for model in ["gpt-4o", "claude-sonnet", "gemini-pro"]:
        for i in range(12):
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


# ── 1. Empty Database ────────────────────────────────────────────────


class TestDigestEmptyDB:
    def test_returns_not_enough_data(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        assert not digest.data_sufficient
        assert digest.data_days_available == 0

    def test_format_shows_not_enough_data(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        msg = gen.format_telegram(digest)
        assert "Not enough data" in msg


# ── 2. Below Minimum Days ───────────────────────────────────────────


class TestDigestBelowMinDays:
    def test_two_days_insufficient(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=2)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        # With only 2 days of data, should be insufficient
        assert digest.data_days_available <= 2


# ── 3. Full Week Digest ─────────────────────────────────────────────


class TestDigestFullWeek:
    def test_pnl_aggregation(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        assert digest.data_sufficient
        assert digest.total_pnl != 0
        assert digest.realized_pnl != 0

    def test_roi_calculated(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        expected_roi = digest.total_pnl / 5000.0 * 100
        assert abs(digest.roi_pct - expected_roi) < 0.01

    def test_sharpe_computed(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        # Sharpe should be non-zero with varying daily P&L
        assert digest.sharpe_7d != 0.0


# ── 4. Category Breakdown ───────────────────────────────────────────


class TestCategoryBreakdown:
    def test_categories_populated(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        assert len(digest.category_breakdown) > 0
        cats = {c.category for c in digest.category_breakdown}
        assert "CRYPTO" in cats or "MACRO" in cats

    def test_win_rate_computed(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        for cat in digest.category_breakdown:
            if cat.trades > 0:
                assert 0 <= cat.win_rate <= 100


# ── 5. Model Accuracy ───────────────────────────────────────────────


class TestModelAccuracy:
    def test_brier_score_correct(self) -> None:
        """Brier score should be computed from model_forecast_log using recorded_at."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        assert len(digest.model_accuracy) > 0
        for m in digest.model_accuracy:
            assert m.brier_score >= 0  # Brier is always non-negative
            assert m.forecasts >= 10  # min trades gate

    def test_directional_accuracy(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        for m in digest.model_accuracy:
            assert 0 <= m.directional_accuracy <= 100


# ── 6. Friction Analysis ────────────────────────────────────────────


class TestFrictionAnalysis:
    def test_friction_gap(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        fa = digest.friction_analysis
        # Gap should equal edge - roi
        assert abs(fa.friction_gap - (fa.avg_edge_at_entry - fa.avg_pnl_per_trade)) < 0.1

    def test_fee_cost_uses_config(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0, transaction_fee_pct=0.03)
        digest = gen.generate(days=7)
        # Fee cost should be positive (we have trades)
        assert digest.friction_analysis.fee_cost_total > 0


# ── 7. Telegram Format ──────────────────────────────────────────────


class TestFormatTelegram:
    def test_fits_4096_chars(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator, TELEGRAM_MAX_LEN
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        msg = gen.format_telegram(digest)
        assert len(msg) <= TELEGRAM_MAX_LEN

    def test_contains_key_sections(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        msg = gen.format_telegram(digest)
        assert "*P&L*" in msg
        assert "Weekly Digest" in msg


# ── 8. Message Splitting ────────────────────────────────────────────


class TestFormatSplit:
    def test_short_message_not_split(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        parts = WeeklyDigestGenerator.split_message("Hello world", max_len=100)
        assert len(parts) == 1

    def test_long_message_split(self) -> None:
        from src.observability.reports import WeeklyDigestGenerator
        long_text = "\n".join([f"Line {i}" for i in range(200)])
        parts = WeeklyDigestGenerator.split_message(long_text, max_len=100)
        assert len(parts) > 1
        for part in parts:
            assert len(part) <= 100


# ── 9. Telegram /weekly Command ─────────────────────────────────────


class TestWeeklyCommand:
    @pytest.mark.asyncio
    async def test_weekly_returns_digest(self) -> None:
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="test", chat_id="123")

        # Mock engine with DB
        conn = _create_test_db()
        _populate_test_db(conn, days=7)
        engine = MagicMock()
        engine._db.conn = conn
        engine.config.risk.bankroll = 5000.0
        engine.config.risk.transaction_fee_pct = 0.02
        bot._engine = engine

        result = await bot._cmd_weekly()
        assert "Weekly Digest" in result or "Not enough data" in result


# ── 10. Telegram /report Command ────────────────────────────────────


class TestReportCommand:
    @pytest.mark.asyncio
    async def test_report_accepts_days(self) -> None:
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="test", chat_id="123")

        conn = _create_test_db()
        _populate_test_db(conn, days=14)
        engine = MagicMock()
        engine._db.conn = conn
        engine.config.risk.bankroll = 5000.0
        engine.config.risk.transaction_fee_pct = 0.02
        bot._engine = engine

        result = await bot._cmd_report(days=14)
        assert isinstance(result, str)
        assert len(result) > 0


# ── 11. Config Loading ──────────────────────────────────────────────


class TestDigestConfig:
    def test_config_loads_digest(self) -> None:
        from src.config import load_config
        config = load_config()
        assert hasattr(config, "digest")
        assert config.digest.enabled is True
        assert config.digest.schedule_hour == 8
        assert config.digest.schedule_day_of_week == "mon"

    def test_config_loads_analyst(self) -> None:
        from src.config import load_config
        config = load_config()
        assert hasattr(config, "analyst")
        assert config.analyst.enabled is False
        assert config.analyst.provider == "anthropic"


# ── 12. APScheduler Wiring ──────────────────────────────────────────


class TestSchedulerWiring:
    def test_scheduler_attribute_exists(self) -> None:
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="test", chat_id="123")
        assert hasattr(bot, "_scheduler")
        assert bot._scheduler is None

    def test_start_scheduler_with_config(self) -> None:
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="test", chat_id="123")

        engine = MagicMock()
        engine.config.digest.enabled = True
        engine.config.digest.schedule_day_of_week = "mon"
        engine.config.digest.schedule_hour = 8
        bot._engine = engine

        with patch("src.observability.telegram_bot.AsyncIOScheduler", create=True) as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched_cls.return_value = mock_sched

            # Patch the import inside the method
            with patch.dict("sys.modules", {
                "apscheduler": MagicMock(),
                "apscheduler.schedulers": MagicMock(),
                "apscheduler.schedulers.asyncio": MagicMock(AsyncIOScheduler=mock_sched_cls),
            }):
                bot._start_scheduler()

    def test_scheduler_not_started_without_engine(self) -> None:
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="test", chat_id="123")
        bot._start_scheduler()
        assert bot._scheduler is None


# ── Markdown Escape Tests ──────────────────────────────────────


class TestMarkdownEscape:
    def test_escape_md_asterisk(self) -> None:
        from src.observability.reports import _escape_md
        assert _escape_md("Will *BTC* rise?") == r"Will \*BTC\* rise?"

    def test_escape_md_brackets(self) -> None:
        from src.observability.reports import _escape_md
        assert _escape_md("Will [BTC] hit $100k?") == r"Will \[BTC] hit $100k?"

    def test_escape_md_underscore(self) -> None:
        from src.observability.reports import _escape_md
        assert _escape_md("some_variable_name") == r"some\_variable\_name"

    def test_escape_md_backtick(self) -> None:
        from src.observability.reports import _escape_md
        assert _escape_md("code `block`") == r"code \`block\`"

    def test_best_worst_trade_escapes_question(self) -> None:
        """best_trade and worst_trade strings have Markdown chars escaped."""
        from src.observability.reports import WeeklyDigestGenerator
        conn = _create_test_db()
        today = dt.datetime.now(dt.timezone.utc)
        # Populate min data days
        for i in range(5):
            d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO daily_summaries "
                "(summary_date, total_pnl, realized_pnl) VALUES (?,?,?)",
                (d, 5.0, 5.0),
            )
        # Insert a trade with Markdown-special question
        d = today.strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO performance_log "
            "(market_id, question, pnl, stake_usd, resolved_at) "
            "VALUES (?,?,?,?,?)",
            ("mkt-md", "Will [BTC] hit *$100k* before_halving?", 10.0, 50.0, d),
        )
        conn.commit()
        gen = WeeklyDigestGenerator(conn, bankroll=5000.0)
        digest = gen.generate(days=7)
        # The asterisks and brackets should be escaped
        assert r"\*" in digest.best_trade
        assert r"\[" in digest.best_trade
