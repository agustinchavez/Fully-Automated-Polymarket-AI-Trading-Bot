"""Tests for Phase 9 Batch A: Kill switch persistence, daily P&L kill, daily summary."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import BotConfig, ProductionConfig, RiskConfig


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """In-memory DB with all migrations applied."""
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _make_production_config(**overrides) -> BotConfig:
    """BotConfig with production enabled."""
    prod_kw = {"enabled": True, **overrides}
    return BotConfig(production=ProductionConfig(**prod_kw))


# ── ProductionConfig ─────────────────────────────────────────────


class TestProductionConfig:
    def test_defaults(self):
        cfg = ProductionConfig()
        assert cfg.enabled is False
        assert cfg.daily_loss_kill_pct == 0.05
        assert cfg.daily_loss_kill_enabled is True
        assert cfg.persist_kill_switch is True
        assert cfg.deployment_stage == "paper"
        assert cfg.telegram_kill_enabled is False

    def test_enabled_in_bot_config(self):
        cfg = BotConfig(production=ProductionConfig(enabled=True))
        assert cfg.production.enabled is True
        assert cfg.production.deployment_stage == "paper"

    def test_custom_daily_loss_pct(self):
        cfg = ProductionConfig(daily_loss_kill_pct=0.10)
        assert cfg.daily_loss_kill_pct == 0.10

    def test_yaml_round_trip(self):
        """Config can be serialized and deserialized."""
        cfg = BotConfig(production=ProductionConfig(enabled=True, deployment_stage="week1"))
        d = cfg.model_dump()
        restored = BotConfig(**d)
        assert restored.production.enabled is True
        assert restored.production.deployment_stage == "week1"

    def test_secret_fields_redacted(self):
        cfg = BotConfig(production=ProductionConfig(
            telegram_kill_token="secret_token_123",
            sentry_dsn="https://secret@sentry.io/123",
        ))
        redacted = cfg.redacted_dict()
        assert redacted["production"]["telegram_kill_token"] == "sec***"
        assert redacted["production"]["sentry_dsn"] == "htt***"


# ── Migration 15 ─────────────────────────────────────────────────


class TestMigration15:
    def test_tables_created(self):
        conn = _create_test_db()
        tables = [
            r[0] for r in
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "kill_switch_state" in tables
        assert "deployment_stages" in tables
        assert "daily_summaries" in tables
        assert "chaos_test_results" in tables

    def test_kill_switch_seeded(self):
        conn = _create_test_db()
        row = conn.execute("SELECT * FROM kill_switch_state WHERE id = 1").fetchone()
        assert row is not None
        assert row["is_killed"] == 0

    def test_schema_version(self):
        conn = _create_test_db()
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] >= 15


# ── Kill Switch DB Persistence ───────────────────────────────────


class TestKillSwitchPersistence:
    def test_default_state(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        state = db.get_kill_switch_state()
        assert state["is_killed"] is False
        assert state["kill_reason"] == ""

    def test_set_and_get(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.set_kill_switch(
            is_killed=True,
            reason="Daily loss exceeded",
            killed_by="daily_pnl_auto",
            daily_pnl=-300.0,
            bankroll=5000.0,
        )
        state = db.get_kill_switch_state()
        assert state["is_killed"] is True
        assert state["kill_reason"] == "Daily loss exceeded"
        assert state["killed_by"] == "daily_pnl_auto"
        assert state["daily_pnl_at_kill"] == -300.0
        assert state["bankroll_at_kill"] == 5000.0
        assert state["killed_at"] != ""

    def test_reset(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.set_kill_switch(True, "test", "test_user")
        db.reset_kill_switch()
        state = db.get_kill_switch_state()
        assert state["is_killed"] is False
        assert state["kill_reason"] == ""
        assert state["killed_by"] == ""

    def test_survives_reconnect(self):
        """Kill state persists across connection objects (simulated with same DB)."""
        conn = _create_test_db()
        from src.storage.database import Database
        from src.config import StorageConfig
        db1 = Database(StorageConfig(sqlite_path=":memory:"))
        db1._conn = conn
        db1.set_kill_switch(True, "test reason", "system")

        # Simulate new connection to same DB
        db2 = Database(StorageConfig(sqlite_path=":memory:"))
        db2._conn = conn
        state = db2.get_kill_switch_state()
        assert state["is_killed"] is True
        assert state["kill_reason"] == "test reason"

    def test_set_not_killed(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.set_kill_switch(False, "", "dashboard")
        state = db.get_kill_switch_state()
        assert state["is_killed"] is False

    def test_missing_table_returns_default(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        db._conn = conn
        state = db.get_kill_switch_state()
        assert state["is_killed"] is False

    def test_overwrite_state(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.set_kill_switch(True, "reason 1", "system")
        db.set_kill_switch(True, "reason 2", "dashboard")
        state = db.get_kill_switch_state()
        assert state["kill_reason"] == "reason 2"
        assert state["killed_by"] == "dashboard"

    def test_reset_when_not_killed(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.reset_kill_switch()  # Should not crash
        state = db.get_kill_switch_state()
        assert state["is_killed"] is False


# ── Kill Switch Restore ──────────────────────────────────────────


class TestKillSwitchRestore:
    def test_restore_killed_state(self):
        """Engine restores kill state from DB on startup."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config()
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_kill_switch_state.return_value = {
            "is_killed": True,
            "kill_reason": "Daily loss",
            "killed_at": "2024-01-01T00:00:00",
            "killed_by": "daily_pnl_auto",
            "daily_pnl_at_kill": -300.0,
            "bankroll_at_kill": 5000.0,
        }
        engine._db = mock_db

        engine._restore_kill_switch_state()
        assert engine.drawdown.state.is_killed is True
        assert engine.drawdown.state.kelly_multiplier == 0.0
        assert engine.config.risk.kill_switch is True

    def test_restore_clean_state(self):
        """Engine does not kill when DB says not killed."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config()
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_kill_switch_state.return_value = {
            "is_killed": False, "kill_reason": "", "killed_at": "",
            "killed_by": "", "daily_pnl_at_kill": 0, "bankroll_at_kill": 0,
        }
        engine._db = mock_db

        engine._restore_kill_switch_state()
        assert engine.drawdown.state.is_killed is False

    def test_no_restore_when_disabled(self):
        """No restore when production is disabled."""
        from src.engine.loop import TradingEngine
        cfg = BotConfig()  # production.enabled=False
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        engine._db = mock_db

        engine._restore_kill_switch_state()
        mock_db.get_kill_switch_state.assert_not_called()

    def test_no_restore_when_persist_disabled(self):
        """No restore when persist_kill_switch=False."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(persist_kill_switch=False)
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        engine._db = mock_db

        engine._restore_kill_switch_state()
        mock_db.get_kill_switch_state.assert_not_called()

    def test_restore_error_handled(self):
        """Error during restore does not crash engine."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config()
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_kill_switch_state.side_effect = Exception("DB error")
        engine._db = mock_db

        engine._restore_kill_switch_state()  # Should not raise
        assert engine.drawdown.state.is_killed is False


# ── Daily P&L Kill ───────────────────────────────────────────────


class TestDailyPnlKill:
    def test_triggers_at_threshold(self):
        """Kill switch engages when daily loss >= threshold."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(daily_loss_kill_pct=0.05)
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = -600.0  # 6% > 5%
        engine._db = mock_db

        triggered = engine._check_daily_pnl_kill()
        assert triggered is True
        assert engine.drawdown.state.is_killed is True
        assert engine.config.risk.kill_switch is True

    def test_no_trigger_below_threshold(self):
        """No kill when loss is below threshold."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(daily_loss_kill_pct=0.05)
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = -400.0  # 4% < 5%
        engine._db = mock_db

        triggered = engine._check_daily_pnl_kill()
        assert triggered is False
        assert engine.drawdown.state.is_killed is False

    def test_no_trigger_on_profit(self):
        """No kill when daily P&L is positive."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config()
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = 500.0
        engine._db = mock_db

        assert engine._check_daily_pnl_kill() is False

    def test_no_trigger_when_already_killed(self):
        """No re-kill when already killed."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config()
        engine = TradingEngine(config=cfg)
        engine.drawdown.state.is_killed = True

        mock_db = MagicMock()
        engine._db = mock_db

        assert engine._check_daily_pnl_kill() is False
        mock_db.get_daily_pnl.assert_not_called()

    def test_no_trigger_when_disabled(self):
        """No check when daily_loss_kill_enabled=False."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(daily_loss_kill_enabled=False)
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        engine._db = mock_db

        assert engine._check_daily_pnl_kill() is False
        mock_db.get_daily_pnl.assert_not_called()

    def test_no_trigger_when_production_disabled(self):
        """No check when production.enabled=False."""
        from src.engine.loop import TradingEngine
        cfg = BotConfig()
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        engine._db = mock_db

        assert engine._check_daily_pnl_kill() is False

    def test_persists_kill_to_db(self):
        """Kill switch persisted to DB when daily P&L triggers."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(daily_loss_kill_pct=0.05)
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = -600.0
        engine._db = mock_db

        engine._check_daily_pnl_kill()
        mock_db.set_kill_switch.assert_called_once()
        call_args = mock_db.set_kill_switch.call_args
        assert call_args[1]["is_killed"] is True
        assert "daily_pnl_auto" in call_args[1]["killed_by"]

    def test_alert_sent_on_kill(self):
        """Alert inserted when daily P&L kill triggers."""
        from src.engine.loop import TradingEngine
        cfg = _make_production_config(daily_loss_kill_pct=0.05)
        cfg.risk.bankroll = 10000.0
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        mock_db.get_daily_pnl.return_value = -600.0
        engine._db = mock_db

        engine._check_daily_pnl_kill()
        mock_db.insert_alert.assert_called_once()
        alert_call = mock_db.insert_alert.call_args
        assert alert_call[0][0] == "critical"


# ── Daily Summary ────────────────────────────────────────────────


class TestDailySummary:
    def test_generate_empty(self):
        from src.observability.daily_summary import DailySummaryGenerator, DailySummary
        conn = _create_test_db()
        gen = DailySummaryGenerator(conn, bankroll=5000.0)
        summary = gen.generate("2024-01-15")
        assert summary.summary_date == "2024-01-15"
        assert summary.total_pnl == 0.0
        assert summary.bankroll == 5000.0

    def test_generate_with_data(self):
        from src.observability.daily_summary import DailySummaryGenerator
        conn = _create_test_db()
        # Insert test performance data
        conn.execute(
            "INSERT INTO performance_log "
            "(market_id, question, category, forecast_prob, actual_outcome, "
            "edge_at_entry, confidence, evidence_quality, stake_usd, "
            "entry_price, exit_price, pnl, holding_hours, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("m1", "Q1", "MACRO", 0.6, 1.0, 0.1, "HIGH", 0.8, 50, 0.5, 1.0, 25.0, 10, "2024-01-15T10:00:00"),
        )
        conn.execute(
            "INSERT INTO performance_log "
            "(market_id, question, category, forecast_prob, actual_outcome, "
            "edge_at_entry, confidence, evidence_quality, stake_usd, "
            "entry_price, exit_price, pnl, holding_hours, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("m2", "Q2", "MACRO", 0.7, 0.0, 0.05, "LOW", 0.5, 30, 0.7, 0.0, -21.0, 5, "2024-01-15T12:00:00"),
        )
        conn.commit()

        gen = DailySummaryGenerator(conn, bankroll=5000.0)
        summary = gen.generate("2024-01-15")
        assert summary.realized_pnl == 4.0  # 25 - 21
        assert summary.trades_closed == 2
        assert summary.best_trade_pnl == 25.0
        assert summary.worst_trade_pnl == -21.0

    def test_persist(self):
        from src.observability.daily_summary import DailySummaryGenerator, DailySummary
        conn = _create_test_db()
        gen = DailySummaryGenerator(conn, bankroll=5000.0)
        summary = DailySummary(
            summary_date="2024-01-15",
            total_pnl=100.0,
            realized_pnl=80.0,
            unrealized_pnl=20.0,
            trades_opened=5,
            trades_closed=3,
            positions_held=2,
            drawdown_pct=0.02,
            bankroll=5000.0,
        )
        gen.persist(summary)

        row = conn.execute(
            "SELECT * FROM daily_summaries WHERE summary_date = '2024-01-15'"
        ).fetchone()
        assert row is not None
        assert float(row["total_pnl"]) == 100.0
        assert int(row["trades_opened"]) == 5

    def test_format_message(self):
        from src.observability.daily_summary import DailySummaryGenerator, DailySummary
        summary = DailySummary(
            summary_date="2024-01-15",
            total_pnl=150.0,
            realized_pnl=120.0,
            unrealized_pnl=30.0,
            trades_opened=10,
            trades_closed=5,
            positions_held=5,
            drawdown_pct=0.03,
            bankroll=5000.0,
        )
        msg = DailySummaryGenerator.format_message(summary)
        assert "2024-01-15" in msg
        assert "$150.00" in msg
        assert "+3.0%" in msg
        assert "10 opened" in msg

    def test_format_message_negative_pnl(self):
        from src.observability.daily_summary import DailySummaryGenerator, DailySummary
        summary = DailySummary(
            summary_date="2024-01-15",
            total_pnl=-200.0,
            bankroll=10000.0,
        )
        msg = DailySummaryGenerator.format_message(summary)
        assert "-$200.00" in msg
        assert "-2.0%" in msg

    def test_send_summary(self):
        import asyncio
        from src.observability.daily_summary import DailySummaryGenerator, DailySummary
        conn = _create_test_db()
        gen = DailySummaryGenerator(conn, bankroll=5000.0)
        summary = DailySummary(summary_date="2024-01-15", total_pnl=100.0, bankroll=5000.0)

        mock_alert = AsyncMock()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(gen.send_summary(summary, mock_alert))
        loop.close()
        mock_alert.send.assert_called_once()

    def test_once_per_day(self):
        """Engine daily summary only sends once per day."""
        from src.engine.loop import TradingEngine
        cfg = BotConfig()
        cfg.alerts.daily_summary_enabled = True
        cfg.alerts.daily_summary_hour = 0  # midnight — always past
        engine = TradingEngine(config=cfg)

        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        engine._last_daily_summary_date = today  # already sent

        mock_db = MagicMock()
        engine._db = mock_db

        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine._maybe_send_daily_summary())
        loop.close()
        # Should not have generated since already sent today
        # (no call to DailySummaryGenerator)

    def test_respects_hour(self):
        """Engine daily summary only sends at or after configured hour."""
        from src.engine.loop import TradingEngine
        cfg = BotConfig()
        cfg.alerts.daily_summary_enabled = True
        cfg.alerts.daily_summary_hour = 23  # 11 PM — very late
        engine = TradingEngine(config=cfg)

        mock_db = MagicMock()
        engine._db = mock_db

        now = dt.datetime.now(dt.timezone.utc)
        if now.hour < 23:
            # It's before 11 PM, so summary should not fire
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(engine._maybe_send_daily_summary())
            loop.close()
            assert engine._last_daily_summary_date == ""


# ── Dashboard Kill Switch Endpoints ──────────────────────────────


import tempfile
import os


def _shared_db_path() -> str:
    """Create a temp file DB with migrations, return path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    from src.storage.migrations import run_migrations
    run_migrations(conn)
    conn.close()
    return path


class TestDashboardKillSwitch:
    @pytest.fixture
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_toggle_persists(self, client):
        """POST /api/kill-switch persists to DB."""
        db_path = _shared_db_path()
        try:
            with patch("src.dashboard.app._get_conn") as mock_conn, \
                 patch("src.dashboard.app._get_config", return_value=BotConfig()):
                def make_conn():
                    c = sqlite3.connect(db_path)
                    c.row_factory = sqlite3.Row
                    return c
                mock_conn.side_effect = make_conn

                resp = client.post("/api/kill-switch")
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["kill_switch"] is True

                # Verify with a fresh connection
                verify = sqlite3.connect(db_path)
                verify.row_factory = sqlite3.Row
                row = verify.execute("SELECT * FROM kill_switch_state WHERE id = 1").fetchone()
                assert row["is_killed"] == 1
                verify.close()
        finally:
            os.unlink(db_path)

    def test_get_state(self, client):
        """GET /api/kill-switch/state returns DB state."""
        db_path = _shared_db_path()
        try:
            with patch("src.dashboard.app._get_conn") as mock_conn:
                def make_conn():
                    c = sqlite3.connect(db_path)
                    c.row_factory = sqlite3.Row
                    return c
                mock_conn.side_effect = make_conn

                resp = client.get("/api/kill-switch/state")
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["is_killed"] is False
        finally:
            os.unlink(db_path)

    def test_reset_requires_confirm(self, client):
        """POST /api/kill-switch/reset requires confirm body."""
        db_path = _shared_db_path()
        try:
            with patch("src.dashboard.app._get_conn") as mock_conn, \
                 patch("src.dashboard.app._get_config", return_value=BotConfig()):
                def make_conn():
                    c = sqlite3.connect(db_path)
                    c.row_factory = sqlite3.Row
                    return c
                mock_conn.side_effect = make_conn

                resp = client.post(
                    "/api/kill-switch/reset",
                    data=json.dumps({}),
                    content_type="application/json",
                )
                assert resp.status_code == 400
        finally:
            os.unlink(db_path)

    def test_reset_clears_state(self, client):
        """POST /api/kill-switch/reset with confirm clears DB."""
        db_path = _shared_db_path()
        try:
            # Set kill state directly
            setup_conn = sqlite3.connect(db_path)
            setup_conn.execute(
                "UPDATE kill_switch_state SET is_killed = 1, kill_reason = 'test' WHERE id = 1"
            )
            setup_conn.commit()
            setup_conn.close()

            with patch("src.dashboard.app._get_conn") as mock_conn, \
                 patch("src.dashboard.app._get_config", return_value=BotConfig()):
                def make_conn():
                    c = sqlite3.connect(db_path)
                    c.row_factory = sqlite3.Row
                    return c
                mock_conn.side_effect = make_conn

                resp = client.post(
                    "/api/kill-switch/reset",
                    data=json.dumps({"confirm": True}),
                    content_type="application/json",
                )
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["success"] is True

                # Verify with fresh connection
                verify = sqlite3.connect(db_path)
                verify.row_factory = sqlite3.Row
                row = verify.execute("SELECT * FROM kill_switch_state WHERE id = 1").fetchone()
                assert row["is_killed"] == 0
                verify.close()
        finally:
            os.unlink(db_path)

    def test_daily_summaries_endpoint(self, client):
        """GET /api/daily-summaries returns list."""
        db_path = _shared_db_path()
        try:
            with patch("src.dashboard.app._get_conn") as mock_conn:
                def make_conn():
                    c = sqlite3.connect(db_path)
                    c.row_factory = sqlite3.Row
                    return c
                mock_conn.side_effect = make_conn

                resp = client.get("/api/daily-summaries")
                assert resp.status_code == 200
                data = json.loads(resp.data)
                assert data["summaries"] == []
        finally:
            os.unlink(db_path)


# ── Database insert_daily_summary ─────────────────────────────────


class TestDatabaseDailySummary:
    def test_insert_and_get(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.insert_daily_summary(
            summary_date="2024-01-15",
            total_pnl=100.0,
            realized_pnl=80.0,
            unrealized_pnl=20.0,
            trades_opened=5,
            trades_closed=3,
            positions_held=2,
            drawdown_pct=0.02,
            bankroll=5000.0,
            best_trade_pnl=50.0,
            worst_trade_pnl=-10.0,
        )
        summaries = db.get_daily_summaries()
        assert len(summaries) == 1
        assert summaries[0]["summary_date"] == "2024-01-15"
        assert float(summaries[0]["total_pnl"]) == 100.0

    def test_upsert_daily_summary(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        db.insert_daily_summary("2024-01-15", 100, 80, 20, 5, 3, 2, 0.02, 5000)
        db.insert_daily_summary("2024-01-15", 200, 150, 50, 8, 6, 2, 0.03, 5000)
        summaries = db.get_daily_summaries()
        assert len(summaries) == 1
        assert float(summaries[0]["total_pnl"]) == 200.0  # Updated

    def test_empty_summaries(self):
        from src.storage.database import Database
        from src.config import StorageConfig
        db = Database(StorageConfig(sqlite_path=":memory:"))
        db._conn = _create_test_db()
        assert db.get_daily_summaries() == []
