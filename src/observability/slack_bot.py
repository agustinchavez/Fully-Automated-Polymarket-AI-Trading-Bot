"""Slack kill-switch bot — remote engine control via Slack.

Uses Slack Bolt with Socket Mode (no public URL required).
Commands use ! prefix: !kill, !status, !pnl, !resume, !weekly,
!report, !insights, !models, !analyze, !provider, !help

Requires: pip install slack-bolt slack-sdk
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.observability.bot_commands import BotCommandHandler
from src.observability.logger import get_logger

log = get_logger(__name__)

_COMMAND_PATTERN = re.compile(
    r"^!(kill|status|pnl|resume|weekly|report|insights|models"
    r"|analyze|provider|help)\b"
)


class SlackKillBot:
    """Slack bot for kill switch control via Socket Mode."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        channel_id: str,
        engine: Any = None,
    ):
        self._bot_token = bot_token
        self._app_token = app_token
        self._channel_id = channel_id
        self._engine = engine
        self._commands = BotCommandHandler(engine)
        self._running = False
        self._handler: Any = None

    async def start(self) -> None:
        """Start the Slack bot in Socket Mode."""
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
        except ImportError:
            log.warning("slack_bot.slack_bolt_not_installed")
            return

        app = AsyncApp(token=self._bot_token)
        bot_ref = self

        @app.message(_COMMAND_PATTERN)
        async def handle_command(message: dict, say: Any) -> None:
            # Only respond in configured channel
            channel = message.get("channel", "")
            if channel != bot_ref._channel_id:
                return

            text = message.get("text", "").strip()
            parts = text.split(maxsplit=1)
            command = parts[0].lstrip("!").lower()
            args = parts[1] if len(parts) > 1 else ""

            response = await bot_ref._commands.dispatch(command, args)
            await say(response)

        self._handler = AsyncSocketModeHandler(app, self._app_token)
        self._running = True
        log.info("slack_bot.starting")
        try:
            await self._handler.start_async()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("slack_bot.connection_error", error=str(e))

    def stop(self) -> None:
        """Stop the Slack bot."""
        self._running = False
        if self._handler:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._handler.close_async())
            except RuntimeError:
                # No running loop — handler will be cleaned up on exit
                pass
        log.info("slack_bot.stopped")
