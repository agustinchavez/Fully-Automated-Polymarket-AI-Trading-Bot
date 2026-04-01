"""Discord kill-switch bot — remote engine control via Discord.

Commands use ! prefix: !kill, !status, !pnl, !resume, !weekly,
!report, !insights, !models, !analyze, !provider, !help

Requires: pip install discord.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.observability.bot_commands import BotCommandHandler
from src.observability.logger import get_logger

log = get_logger(__name__)

DISCORD_MAX_MESSAGE_LEN = 2000


class DiscordKillBot:
    """Discord bot for kill switch control via ! prefix commands."""

    def __init__(
        self,
        token: str,
        channel_id: str,
        engine: Any = None,
    ):
        self._token = token
        self._channel_id = int(channel_id) if channel_id else 0
        self._engine = engine
        self._commands = BotCommandHandler(engine)
        self._running = False
        self._client: Any = None

    async def start(self) -> None:
        """Start the Discord bot."""
        try:
            import discord
        except ImportError:
            log.warning("discord_bot.discord_py_not_installed")
            return

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        bot_ref = self

        @self._client.event
        async def on_ready() -> None:
            log.info("discord_bot.connected", user=str(self._client.user))

        @self._client.event
        async def on_message(message: Any) -> None:
            # Ignore own messages
            if message.author == self._client.user:
                return
            # Only respond in configured channel
            if message.channel.id != bot_ref._channel_id:
                return

            text = message.content.strip()
            if not text.startswith("!"):
                return

            parts = text.split(maxsplit=1)
            command = parts[0].lstrip("!").lower()
            args = parts[1] if len(parts) > 1 else ""

            response = await bot_ref._commands.dispatch(command, args)
            formatted = _to_discord_bold(response)
            await _send_chunked(message.channel, formatted)

        self._running = True
        log.info("discord_bot.starting")
        try:
            await self._client.start(self._token)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("discord_bot.connection_error", error=str(e))

    def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._client and not self._client.is_closed():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._client.close())
            except RuntimeError:
                # No running loop — client will be cleaned up on exit
                pass
        log.info("discord_bot.stopped")


def _to_discord_bold(text: str) -> str:
    """Convert Telegram-style *bold* to Discord **bold**.

    Telegram uses single asterisks for bold; Discord uses double.
    This does a simple conversion for the bot's output format.
    """
    import re
    return re.sub(r"\*([^*]+)\*", r"**\1**", text)


async def _send_chunked(channel: Any, text: str) -> None:
    """Send a message, splitting at Discord's 2000 char limit."""
    chunks = _split_message(text, DISCORD_MAX_MESSAGE_LEN)
    for chunk in chunks:
        await channel.send(chunk)


def _split_message(text: str, max_len: int) -> list[str]:
    """Split a long message into chunks at line boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        idx = text.rfind("\n", 0, max_len)
        if idx == -1:
            idx = max_len
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks
