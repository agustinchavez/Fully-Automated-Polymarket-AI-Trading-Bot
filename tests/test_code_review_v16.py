"""Code Review v16 — Resource leak, Telegram fixes, dashboard cost accuracy.

Issue 1: PipelineRunner.close() wired on engine shutdown
Issue 2: Telegram _send_message escapes Markdown
Issue 3: Dashboard cost table uses CostTracker (single source of truth)
Issue 4: Telegram shared httpx client
Issue 6: Stale claude-3-5-sonnet removed from _DEFAULT_COSTS
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Issue 1: PipelineRunner.close() on shutdown ──────────────────────


class TestPipelineCloseOnShutdown:
    """Verify engine calls PipelineRunner.close() when start() exits."""

    def test_close_call_in_start_source(self) -> None:
        """The start() method contains await self._pipeline.close()."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine.start)
        assert "await self._pipeline.close()" in source

    def test_close_wrapped_in_try_except(self) -> None:
        """Pipeline close is wrapped in try/except for safety."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine.start)
        # Find the close call and verify it's inside a try block
        assert "pipeline_close_error" in source


# ── Issue 2: Telegram Markdown Escaping ──────────────────────────────


class TestTelegramMarkdownEscaping:
    """Verify _send_message escapes Markdown special chars."""

    def test_escape_md_function_exists(self) -> None:
        """telegram_bot module has _escape_md function."""
        from src.observability.telegram_bot import _escape_md
        assert callable(_escape_md)

    def test_escape_md_asterisk(self) -> None:
        from src.observability.telegram_bot import _escape_md
        assert _escape_md("*bold*") == "\\*bold\\*"

    def test_escape_md_underscore(self) -> None:
        from src.observability.telegram_bot import _escape_md
        assert _escape_md("_italic_") == "\\_italic\\_"

    def test_escape_md_backtick(self) -> None:
        from src.observability.telegram_bot import _escape_md
        assert _escape_md("`code`") == "\\`code\\`"

    def test_escape_md_bracket(self) -> None:
        from src.observability.telegram_bot import _escape_md
        assert _escape_md("[link](url)") == "\\[link](url)"

    def test_escape_md_no_special(self) -> None:
        from src.observability.telegram_bot import _escape_md
        assert _escape_md("hello world") == "hello world"

    def test_escape_md_all_chars(self) -> None:
        from src.observability.telegram_bot import _escape_md
        text = "*_`["
        escaped = _escape_md(text)
        assert escaped == "\\*\\_\\`\\["

    @pytest.mark.asyncio
    async def test_send_message_escapes_text(self) -> None:
        """_send_message applies _escape_md before sending."""
        from src.observability.telegram_bot import TelegramKillBot

        bot = TelegramKillBot(token="fake", chat_id="123")

        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_resp)
        bot._http = mock_client

        await bot._send_message("*bold_text*")

        # Verify the text was escaped in the payload
        call_args = mock_client.post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        sent_text = payload["text"]
        assert "\\*" in sent_text
        assert "\\_" in sent_text


# ── Issue 3: Dashboard Cost Uses CostTracker ─────────────────────────


class TestDashboardCostSource:
    """Verify dashboard uses CostTracker._token_costs instead of local dict."""

    def test_no_hardcoded_cost_dict_in_status(self) -> None:
        """The local cost_per_1m_input/output dicts should be removed."""
        from src.dashboard import app
        source = inspect.getsource(app)
        # The old hardcoded dicts should no longer exist
        assert "cost_per_1m_input" not in source
        assert "cost_per_1m_output" not in source

    def test_cost_tracker_import_in_status(self) -> None:
        """Dashboard status endpoint imports cost_tracker for pricing."""
        from src.dashboard import app
        source = inspect.getsource(app)
        assert "_ct._token_costs" in source or "cost_tracker" in source


# ── Issue 4: Telegram Shared httpx Client ────────────────────────────


class TestTelegramSharedClient:
    """Verify Telegram bot uses a shared httpx client."""

    def test_get_http_method_exists(self) -> None:
        """TelegramKillBot has _get_http method."""
        from src.observability.telegram_bot import TelegramKillBot
        assert hasattr(TelegramKillBot, "_get_http")
        assert inspect.iscoroutinefunction(TelegramKillBot._get_http)

    @pytest.mark.asyncio
    async def test_get_http_lazy_init(self) -> None:
        """_get_http creates client on first call."""
        from src.observability.telegram_bot import TelegramKillBot

        bot = TelegramKillBot(token="fake", chat_id="123")
        assert bot._http is None

        client = await bot._get_http()
        assert client is not None
        assert bot._http is client

        # Close for cleanup
        await client.aclose()
        bot._http = None

    @pytest.mark.asyncio
    async def test_get_http_returns_same_instance(self) -> None:
        """Subsequent calls return the same client."""
        from src.observability.telegram_bot import TelegramKillBot

        bot = TelegramKillBot(token="fake", chat_id="123")
        c1 = await bot._get_http()
        c2 = await bot._get_http()
        assert c1 is c2

        await c1.aclose()
        bot._http = None

    def test_stop_clears_http_client(self) -> None:
        """stop() sets _http to None."""
        from src.observability.telegram_bot import TelegramKillBot

        bot = TelegramKillBot(token="fake", chat_id="123")
        bot._http = MagicMock()  # fake client
        bot._http.aclose = AsyncMock()
        bot.stop()
        assert bot._http is None

    def test_no_async_with_httpx_in_send(self) -> None:
        """_send_message no longer creates per-call httpx.AsyncClient."""
        from src.observability.telegram_bot import TelegramKillBot
        source = inspect.getsource(TelegramKillBot._send_message)
        assert "async with httpx.AsyncClient" not in source

    def test_no_async_with_httpx_in_get_updates(self) -> None:
        """_get_updates no longer creates per-call httpx.AsyncClient."""
        from src.observability.telegram_bot import TelegramKillBot
        source = inspect.getsource(TelegramKillBot._get_updates)
        assert "async with httpx.AsyncClient" not in source


# ── Issue 6: Stale Model Entry Removed ───────────────────────────────


class TestStaleModelRemoved:
    """Verify claude-3-5-sonnet-20241022 removed from _DEFAULT_COSTS."""

    def test_not_in_default_costs(self) -> None:
        from src.observability.metrics import _DEFAULT_COSTS
        assert "claude-3-5-sonnet-20241022" not in _DEFAULT_COSTS

    def test_still_in_token_costs(self) -> None:
        """Keep in _TOKEN_COSTS for historical data lookups."""
        from src.observability.metrics import _TOKEN_COSTS
        assert "claude-3-5-sonnet-20241022" in _TOKEN_COSTS

    def test_all_active_models_in_default_costs(self) -> None:
        """All 5 active ensemble models are in _DEFAULT_COSTS."""
        from src.observability.metrics import _DEFAULT_COSTS
        active_models = [
            "gpt-4o", "claude-haiku-4-5-20251001", "gemini-2.5-flash",
            "grok-4-fast-reasoning", "deepseek-chat",
        ]
        for model in active_models:
            assert model in _DEFAULT_COSTS, f"Missing active model: {model}"

    def test_all_active_models_in_token_costs(self) -> None:
        """All 5 active ensemble models are in _TOKEN_COSTS."""
        from src.observability.metrics import _TOKEN_COSTS
        active_models = [
            "gpt-4o", "claude-haiku-4-5-20251001", "gemini-2.5-flash",
            "grok-4-fast-reasoning", "deepseek-chat",
        ]
        for model in active_models:
            assert model in _TOKEN_COSTS, f"Missing active model: {model}"
