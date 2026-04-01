"""Tests for Slack kill-switch bot."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.slack_bot import SlackKillBot, _COMMAND_PATTERN


# ── Helpers ──────────────────────────────────────────────────────


def _make_bot(engine=None) -> SlackKillBot:
    return SlackKillBot(
        bot_token="xoxb-test-token",
        app_token="xapp-test-token",
        channel_id="C123456",
        engine=engine,
    )


def _make_engine():
    engine = MagicMock()
    engine.is_running = True
    engine._cycle_count = 42
    engine.drawdown = MagicMock()
    engine.drawdown.state = MagicMock()
    engine.drawdown.state.is_killed = False
    engine.drawdown.state.kill_reason = ""
    engine.drawdown.state.drawdown_pct = 0.05
    engine.drawdown.state.peak_value = 10000.0
    engine.drawdown.state.current_value = 9500.0
    engine._db = MagicMock()
    engine._db.get_daily_pnl.return_value = 150.0
    engine.config = MagicMock()
    engine.config.risk.bankroll = 10000.0
    return engine


# ── Init ─────────────────────────────────────────────────────────


class TestSlackBotInit:
    def test_creates_with_params(self):
        bot = _make_bot()
        assert bot._bot_token == "xoxb-test-token"
        assert bot._app_token == "xapp-test-token"
        assert bot._channel_id == "C123456"
        assert bot._running is False

    def test_has_command_handler(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        assert bot._commands is not None
        assert bot._commands._engine is engine


# ── Command Pattern ──────────────────────────────────────────────


class TestSlackCommandPattern:
    def test_matches_kill(self):
        assert _COMMAND_PATTERN.match("!kill")

    def test_matches_status(self):
        assert _COMMAND_PATTERN.match("!status")

    def test_matches_pnl(self):
        assert _COMMAND_PATTERN.match("!pnl")

    def test_matches_report_with_args(self):
        assert _COMMAND_PATTERN.match("!report 14")

    def test_matches_analyze_with_args(self):
        assert _COMMAND_PATTERN.match("!analyze 60")

    def test_no_match_without_bang(self):
        assert not _COMMAND_PATTERN.match("kill")

    def test_no_match_random_text(self):
        assert not _COMMAND_PATTERN.match("hello world")

    def test_all_commands_match(self):
        for cmd in ("kill", "status", "pnl", "resume", "weekly",
                     "report", "insights", "models", "analyze",
                     "provider", "help"):
            assert _COMMAND_PATTERN.match(f"!{cmd}"), f"!{cmd} should match"


# ── Command Delegation ───────────────────────────────────────────


class TestSlackCommandDelegation:
    @pytest.mark.asyncio
    async def test_kill_command(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        result = await bot._commands.dispatch("kill")
        assert "KILL SWITCH ACTIVATED" in result

    @pytest.mark.asyncio
    async def test_status_command(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        result = await bot._commands.dispatch("status")
        assert "ENGINE STATUS" in result

    @pytest.mark.asyncio
    async def test_help_command(self):
        bot = _make_bot()
        result = await bot._commands.dispatch("help")
        assert "kill" in result

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        bot = _make_bot()
        result = await bot._commands.dispatch("xyz")
        assert "Unknown command" in result


# ── Lifecycle ────────────────────────────────────────────────────


class TestSlackBotLifecycle:
    def test_stop_sets_running_false(self):
        bot = _make_bot()
        bot._running = True
        bot._handler = None
        bot.stop()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_stop_closes_handler(self):
        bot = _make_bot()
        bot._running = True
        mock_handler = MagicMock()
        mock_handler.close_async = AsyncMock()
        bot._handler = mock_handler
        bot.stop()
        assert bot._running is False
