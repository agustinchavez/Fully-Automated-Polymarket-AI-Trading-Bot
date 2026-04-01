"""Tests for BotCommandHandler — shared command logic across platforms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from src.observability.bot_commands import BotCommandHandler, _parse_int


# ── Helpers ──────────────────────────────────────────────────────


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
    engine.config.risk.transaction_fee_pct = 0.02
    return engine


def _handler(engine=None) -> BotCommandHandler:
    return BotCommandHandler(engine)


# ── Dispatch ─────────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_known_command(self):
        h = _handler(_make_engine())
        result = await h.dispatch("status")
        assert "ENGINE STATUS" in result

    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self):
        h = _handler()
        result = await h.dispatch("foobar")
        assert "Unknown command" in result

    @pytest.mark.asyncio
    async def test_dispatch_report_with_args(self):
        engine = _make_engine()
        h = _handler(engine)
        # Patch _get_digest_generator to avoid needing real DB
        with patch.object(h, "_get_digest_generator", return_value=None):
            result = await h.dispatch("report", "14")
            assert "No engine/DB" in result

    @pytest.mark.asyncio
    async def test_dispatch_analyze_with_args(self):
        engine = _make_engine()
        h = _handler(engine)
        result = await h.dispatch("analyze", "60")
        assert "disabled" in result.lower() or "error" in result.lower()

    @pytest.mark.asyncio
    async def test_dispatch_all_commands_exist(self):
        """Every command in COMMANDS should dispatch without crashing."""
        h = _handler()
        for cmd in BotCommandHandler.COMMANDS:
            result = await h.dispatch(cmd)
            assert isinstance(result, str)
            assert len(result) > 0


# ── Kill Command ─────────────────────────────────────────────────


class TestCmdKill:
    @pytest.mark.asyncio
    async def test_kill_activates(self):
        engine = _make_engine()
        h = _handler(engine)
        result = await h.cmd_kill()
        assert "KILL SWITCH ACTIVATED" in result
        engine.drawdown.kill.assert_called_once_with("Bot /kill command")
        engine._persist_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_no_engine(self):
        h = _handler()
        result = await h.cmd_kill()
        assert "No engine" in result

    @pytest.mark.asyncio
    async def test_kill_exception_handled(self):
        engine = _make_engine()
        engine.drawdown.kill.side_effect = RuntimeError("boom")
        h = _handler(engine)
        result = await h.cmd_kill()
        assert "Kill failed" in result


# ── Status Command ───────────────────────────────────────────────


class TestCmdStatus:
    @pytest.mark.asyncio
    async def test_status_format(self):
        engine = _make_engine()
        h = _handler(engine)
        result = await h.cmd_status()
        assert "ENGINE STATUS" in result
        assert "Running: True" in result
        assert "Killed: False" in result
        assert "Cycles: 42" in result
        assert "$10,000.00" in result
        assert "$9,500.00" in result

    @pytest.mark.asyncio
    async def test_status_with_kill_reason(self):
        engine = _make_engine()
        engine.drawdown.state.kill_reason = "Daily loss limit"
        h = _handler(engine)
        result = await h.cmd_status()
        assert "Daily loss limit" in result

    @pytest.mark.asyncio
    async def test_status_no_engine(self):
        h = _handler()
        result = await h.cmd_status()
        assert "No engine" in result


# ── PnL Command ──────────────────────────────────────────────────


class TestCmdPnl:
    @pytest.mark.asyncio
    async def test_pnl_positive(self):
        engine = _make_engine()
        engine._db.get_daily_pnl.return_value = 150.0
        h = _handler(engine)
        result = await h.cmd_pnl()
        assert "+$150.00" in result
        assert "+1.5%" in result

    @pytest.mark.asyncio
    async def test_pnl_negative(self):
        engine = _make_engine()
        engine._db.get_daily_pnl.return_value = -200.0
        h = _handler(engine)
        result = await h.cmd_pnl()
        assert "-$200.00" in result
        assert "-2.0%" in result

    @pytest.mark.asyncio
    async def test_pnl_zero_bankroll(self):
        engine = _make_engine()
        engine.config.risk.bankroll = 0
        h = _handler(engine)
        result = await h.cmd_pnl()
        assert "P&L" in result

    @pytest.mark.asyncio
    async def test_pnl_no_engine(self):
        h = _handler()
        result = await h.cmd_pnl()
        assert "No engine" in result

    @pytest.mark.asyncio
    async def test_pnl_no_db(self):
        engine = _make_engine()
        engine._db = None
        h = _handler(engine)
        result = await h.cmd_pnl()
        assert "No engine/DB" in result


# ── Resume Command ───────────────────────────────────────────────


class TestCmdResume:
    @pytest.mark.asyncio
    async def test_resume_resets_kill(self):
        engine = _make_engine()
        engine.drawdown.state.is_killed = True
        engine.drawdown.state.kill_reason = "test"
        h = _handler(engine)
        result = await h.cmd_resume()
        assert "KILL SWITCH RESET" in result
        assert engine.drawdown.state.is_killed is False
        assert engine.drawdown.state.kill_reason == ""
        engine._db.reset_kill_switch.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_no_engine(self):
        h = _handler()
        result = await h.cmd_resume()
        assert "No engine" in result


# ── Help Command ─────────────────────────────────────────────────


class TestCmdHelp:
    @pytest.mark.asyncio
    async def test_help_lists_all_commands(self):
        h = _handler()
        result = await h.cmd_help()
        assert "BOT COMMANDS" in result
        for cmd in ("kill", "status", "pnl", "resume", "weekly",
                     "report", "insights", "models", "analyze", "provider"):
            assert cmd in result

    @pytest.mark.asyncio
    async def test_help_no_engine_still_works(self):
        h = _handler()
        result = await h.cmd_help()
        assert len(result) > 50


# ── Unknown Command ──────────────────────────────────────────────


class TestCmdUnknown:
    @pytest.mark.asyncio
    async def test_unknown_message(self):
        h = _handler()
        result = await h.cmd_unknown()
        assert "Unknown command" in result
        assert "help" in result.lower()


# ── Digest Commands ──────────────────────────────────────────────


class TestDigestCommands:
    @pytest.mark.asyncio
    async def test_weekly_no_db(self):
        h = _handler()
        result = await h.cmd_weekly()
        assert "No engine/DB" in result

    @pytest.mark.asyncio
    async def test_report_no_db(self):
        h = _handler()
        result = await h.cmd_report(7)
        assert "No engine/DB" in result

    @pytest.mark.asyncio
    async def test_insights_no_db(self):
        h = _handler()
        result = await h.cmd_insights()
        assert "No engine/DB" in result

    @pytest.mark.asyncio
    async def test_models_no_db(self):
        h = _handler()
        result = await h.cmd_models()
        assert "No engine/DB" in result


# ── AI Analysis Commands ─────────────────────────────────────────


class TestAnalyzeCommands:
    @pytest.mark.asyncio
    async def test_analyze_no_engine(self):
        h = _handler()
        result = await h.cmd_analyze()
        assert "No engine" in result

    @pytest.mark.asyncio
    async def test_analyze_disabled(self):
        engine = _make_engine()
        engine.config.analyst.enabled = False
        h = _handler(engine)
        result = await h.cmd_analyze()
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_provider_no_config(self):
        h = _handler()
        result = await h.cmd_provider()
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_provider_shows_config(self):
        engine = _make_engine()
        engine.config.analyst.enabled = True
        engine.config.analyst.provider = "anthropic"
        engine.config.analyst.model = "claude-sonnet-4-6"
        engine.config.analyst.rate_limit_hours = 24
        h = _handler(engine)
        result = await h.cmd_provider()
        assert "anthropic" in result
        assert "claude-sonnet-4-6" in result


# ── _parse_int ───────────────────────────────────────────────────


class TestParseInt:
    def test_valid_int(self):
        assert _parse_int("14", default=7, min_val=3, max_val=30) == 14

    def test_empty_string_returns_default(self):
        assert _parse_int("", default=7, min_val=3, max_val=30) == 7

    def test_whitespace_returns_default(self):
        assert _parse_int("  ", default=7, min_val=3, max_val=30) == 7

    def test_invalid_returns_default(self):
        assert _parse_int("abc", default=7, min_val=3, max_val=30) == 7

    def test_clamps_low(self):
        assert _parse_int("1", default=7, min_val=3, max_val=30) == 3

    def test_clamps_high(self):
        assert _parse_int("100", default=7, min_val=3, max_val=30) == 30
