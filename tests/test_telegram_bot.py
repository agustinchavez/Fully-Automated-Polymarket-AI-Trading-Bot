"""Tests for Phase 9 Batch C: Telegram kill-switch bot."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.telegram_bot import TelegramKillBot


# ── Helpers ──────────────────────────────────────────────────────


def _make_bot(engine=None) -> TelegramKillBot:
    return TelegramKillBot(
        token="test-token-123",
        chat_id="12345",
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


def _make_update(text: str, chat_id: str = "12345") -> dict:
    return {
        "update_id": 1,
        "message": {
            "chat": {"id": int(chat_id) if chat_id.isdigit() else 0},
            "text": text,
        },
    }


# ── TelegramKillBot Init ────────────────────────────────────────


class TestTelegramBotInit:
    def test_creates_with_params(self):
        bot = _make_bot()
        assert bot._token == "test-token-123"
        assert bot._chat_id == "12345"
        assert bot._running is False

    def test_base_url(self):
        bot = _make_bot()
        assert "test-token-123" in bot._base_url


# ── Security ─────────────────────────────────────────────────────


class TestTelegramSecurity:
    @pytest.mark.asyncio
    async def test_rejects_unauthorized_chat(self):
        bot = _make_bot()
        update = _make_update("/status", chat_id="99999")
        # Should not crash and should not send a response
        with patch.object(bot, "_send_message", new_callable=AsyncMock) as mock_send:
            await bot._handle_update(update)
            mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepts_authorized_chat(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        update = _make_update("/status", chat_id="12345")
        with patch.object(bot, "_send_message", new_callable=AsyncMock) as mock_send:
            await bot._handle_update(update)
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_ignores_non_commands(self):
        bot = _make_bot()
        update = _make_update("hello world", chat_id="12345")
        with patch.object(bot, "_send_message", new_callable=AsyncMock) as mock_send:
            await bot._handle_update(update)
            mock_send.assert_not_called()


# ── Kill Command ─────────────────────────────────────────────────


class TestTelegramKillCommand:
    @pytest.mark.asyncio
    async def test_kill_activates(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        response = await bot._cmd_kill()
        assert "KILL SWITCH ACTIVATED" in response
        engine.drawdown.kill.assert_called_once()
        engine._persist_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_no_engine(self):
        bot = _make_bot()
        response = await bot._cmd_kill()
        assert "No engine" in response


# ── Status Command ───────────────────────────────────────────────


class TestTelegramStatusCommand:
    @pytest.mark.asyncio
    async def test_status_returns_info(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        response = await bot._cmd_status()
        assert "ENGINE STATUS" in response
        assert "Running: True" in response
        assert "Cycles: 42" in response

    @pytest.mark.asyncio
    async def test_status_no_engine(self):
        bot = _make_bot()
        response = await bot._cmd_status()
        assert "No engine" in response


# ── PnL Command ──────────────────────────────────────────────────


class TestTelegramPnlCommand:
    @pytest.mark.asyncio
    async def test_pnl_positive(self):
        engine = _make_engine()
        engine._db.get_daily_pnl.return_value = 150.0
        bot = _make_bot(engine)
        response = await bot._cmd_pnl()
        assert "+$150.00" in response

    @pytest.mark.asyncio
    async def test_pnl_negative(self):
        engine = _make_engine()
        engine._db.get_daily_pnl.return_value = -200.0
        bot = _make_bot(engine)
        response = await bot._cmd_pnl()
        assert "-$200.00" in response

    @pytest.mark.asyncio
    async def test_pnl_no_engine(self):
        bot = _make_bot()
        response = await bot._cmd_pnl()
        assert "No engine" in response


# ── Resume Command ───────────────────────────────────────────────


class TestTelegramResumeCommand:
    @pytest.mark.asyncio
    async def test_resume_resets(self):
        engine = _make_engine()
        engine.drawdown.state.is_killed = True
        bot = _make_bot(engine)
        response = await bot._cmd_resume()
        assert "KILL SWITCH RESET" in response
        assert engine.drawdown.state.is_killed is False
        engine._db.reset_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_no_engine(self):
        bot = _make_bot()
        response = await bot._cmd_resume()
        assert "No engine" in response


# ── Help Command ─────────────────────────────────────────────────


class TestTelegramHelpCommand:
    @pytest.mark.asyncio
    async def test_help_lists_commands(self):
        bot = _make_bot()
        response = await bot._cmd_help()
        assert "/kill" in response
        assert "/status" in response
        assert "/pnl" in response
        assert "/resume" in response


# ── Unknown Command ──────────────────────────────────────────────


class TestTelegramUnknownCommand:
    @pytest.mark.asyncio
    async def test_unknown(self):
        bot = _make_bot()
        response = await bot._cmd_unknown()
        assert "Unknown command" in response


# ── Bot Lifecycle ────────────────────────────────────────────────


class TestTelegramBotLifecycle:
    def test_stop(self):
        bot = _make_bot()
        bot._running = True
        bot.stop()
        assert bot._running is False

    @pytest.mark.asyncio
    async def test_handle_update_routes_command(self):
        engine = _make_engine()
        bot = _make_bot(engine)
        update = _make_update("/help")
        with patch.object(bot, "_send_message", new_callable=AsyncMock) as mock_send:
            await bot._handle_update(update)
            mock_send.assert_called_once()
            msg = mock_send.call_args[0][0]
            assert "/kill" in msg

    @pytest.mark.asyncio
    async def test_handles_bot_suffix(self):
        """Commands like /kill@mybot should work."""
        engine = _make_engine()
        bot = _make_bot(engine)
        update = _make_update("/help@mybot")
        with patch.object(bot, "_send_message", new_callable=AsyncMock) as mock_send:
            await bot._handle_update(update)
            mock_send.assert_called_once()
            msg = mock_send.call_args[0][0]
            assert "/kill" in msg
