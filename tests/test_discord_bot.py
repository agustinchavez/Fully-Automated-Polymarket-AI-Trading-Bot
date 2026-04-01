"""Tests for Discord kill-switch bot."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.discord_bot import (
    DiscordKillBot,
    _split_message,
    _to_discord_bold,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_bot(engine=None) -> DiscordKillBot:
    return DiscordKillBot(
        token="discord-test-token",
        channel_id="123456789",
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


class TestDiscordBotInit:
    def test_creates_with_params(self):
        bot = _make_bot()
        assert bot._token == "discord-test-token"
        assert bot._channel_id == 123456789
        assert bot._running is False

    def test_has_command_handler(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        assert bot._commands is not None
        assert bot._commands._engine is engine

    def test_empty_channel_id(self):
        bot = DiscordKillBot(token="tok", channel_id="", engine=None)
        assert bot._channel_id == 0


# ── Command Delegation ───────────────────────────────────────────


class TestDiscordCommandDelegation:
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
    async def test_pnl_command(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        result = await bot._commands.dispatch("pnl")
        assert "+$150.00" in result

    @pytest.mark.asyncio
    async def test_help_command(self):
        bot = _make_bot()
        result = await bot._commands.dispatch("help")
        assert "kill" in result
        assert "status" in result

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        bot = _make_bot()
        result = await bot._commands.dispatch("foobar")
        assert "Unknown command" in result

    @pytest.mark.asyncio
    async def test_resume_command(self):
        engine = _make_engine()
        engine.drawdown.state.is_killed = True
        bot = _make_bot(engine)
        result = await bot._commands.dispatch("resume")
        assert "KILL SWITCH RESET" in result


# ── Bold Formatting ──────────────────────────────────────────────


class TestDiscordBoldFormatting:
    def test_converts_single_bold(self):
        result = _to_discord_bold("*Hello*")
        assert result == "**Hello**"

    def test_converts_multiple_bold(self):
        result = _to_discord_bold("*Hello* world *test*")
        assert result == "**Hello** world **test**"

    def test_no_asterisks_unchanged(self):
        result = _to_discord_bold("Hello world")
        assert result == "Hello world"

    def test_real_command_output(self):
        text = "*AI Analyst Config*\nEnabled: True\nProvider: anthropic"
        result = _to_discord_bold(text)
        assert "**AI Analyst Config**" in result
        assert "Enabled: True" in result


# ── Message Splitting ────────────────────────────────────────────


class TestDiscordSplitMessage:
    def test_short_message_no_split(self):
        result = _split_message("hello", 2000)
        assert result == ["hello"]

    def test_long_message_splits(self):
        msg = "\n".join([f"line{i}" for i in range(500)])
        result = _split_message(msg, 2000)
        assert len(result) >= 2
        # All content preserved
        joined = "\n".join(result)
        assert "line0" in joined
        assert "line499" in joined

    def test_no_newline_splits_at_limit(self):
        msg = "a" * 4000
        result = _split_message(msg, 2000)
        assert len(result) == 2
        assert len(result[0]) == 2000


# ── Lifecycle ────────────────────────────────────────────────────


class TestDiscordBotLifecycle:
    def test_stop_sets_running_false(self):
        bot = _make_bot()
        bot._running = True
        bot._client = None
        bot.stop()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_stop_closes_client(self):
        bot = _make_bot()
        bot._running = True
        mock_client = MagicMock()
        mock_client.is_closed.return_value = False
        mock_client.close = AsyncMock()
        bot._client = mock_client
        bot.stop()
        assert bot._running is False
