"""Telegram kill-switch bot — remote engine control from mobile.

Commands:
  /kill    — activate kill switch + persist + alert
  /status  — engine running, kill state, drawdown, positions
  /pnl     — today's P&L
  /resume  — reset kill switch
  /weekly  — send weekly performance digest
  /report  — send digest for last N days (default 7)
  /insights — one-line actionable insight
  /models  — model accuracy table (30 days)
  /analyze — run AI analysis of bot performance
  /provider — show configured AI provider/model
  /help    — list commands
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.observability.bot_commands import BotCommandHandler
from src.observability.logger import get_logger

log = get_logger(__name__)

TELEGRAM_MAX_MESSAGE_LEN = 4096


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters."""
    for ch in ("*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


class TelegramKillBot:
    """Lightweight long-polling Telegram bot for kill switch control."""

    def __init__(
        self,
        token: str,
        chat_id: str,
        engine: Any = None,
    ):
        self._token = token
        self._chat_id = chat_id
        self._engine = engine
        self._commands = BotCommandHandler(engine)
        self._running = False
        self._offset = 0
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._scheduler: Any = None
        self._http: Any = None  # shared httpx.AsyncClient

    async def _get_http(self) -> Any:
        """Return shared httpx.AsyncClient, creating lazily on first call."""
        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(timeout=40)
        return self._http

    # ── Public API ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start long-polling loop."""
        self._running = True
        self._start_scheduler()
        log.info("telegram_bot.started")
        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("telegram_bot.poll_error", error=str(e))
                await asyncio.sleep(5)

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._scheduler:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        # Schedule shared client cleanup
        if self._http is not None:
            try:
                asyncio.ensure_future(self._http.aclose())
            except Exception:
                pass
            self._http = None
        log.info("telegram_bot.stopped")

    def _start_scheduler(self) -> None:
        """Start APScheduler for weekly digest cron job."""
        config = getattr(self._engine, "config", None) if self._engine else None
        digest_cfg = getattr(config, "digest", None)
        if not digest_cfg or not digest_cfg.enabled:
            return
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            self._scheduler = AsyncIOScheduler(timezone="UTC")
            self._scheduler.add_job(
                self._send_weekly_digest,
                trigger="cron",
                day_of_week=digest_cfg.schedule_day_of_week,
                hour=digest_cfg.schedule_hour,
                minute=0,
            )
            self._scheduler.start()
            log.info(
                "telegram_bot.scheduler_started",
                day=digest_cfg.schedule_day_of_week,
                hour=digest_cfg.schedule_hour,
            )
        except ImportError:
            log.warning("telegram_bot.apscheduler_not_installed")
        except Exception as e:
            log.warning("telegram_bot.scheduler_error", error=str(e))

    # ── Polling ───────────────────────────────────────────────────

    async def _get_updates(self) -> list[dict]:
        """Fetch updates via long polling."""
        try:
            import httpx  # noqa: F401 — ensure available
        except ImportError:
            log.warning("telegram_bot.httpx_not_installed")
            await asyncio.sleep(30)
            return []

        url = f"{self._base_url}/getUpdates"
        params = {"offset": self._offset, "timeout": 30}

        client = await self._get_http()
        resp = await client.get(url, params=params)
        data = resp.json()

        if not data.get("ok"):
            return []

        updates = data.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    async def _handle_update(self, update: dict) -> None:
        """Process a single update."""
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "").strip()

        # Security: only respond to configured chat_id
        if chat_id != self._chat_id:
            log.warning(
                "telegram_bot.unauthorized",
                chat_id=chat_id,
                expected=self._chat_id,
            )
            return

        if not text.startswith("/"):
            return

        # Parse command and args
        parts = text.split(maxsplit=1)
        raw_cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Strip /prefix and @botname suffix
        command = raw_cmd.lstrip("/")
        if "@" in command:
            command = command.split("@")[0]

        response = await self._commands.dispatch(command, args)
        await self._send_message(response)

    async def _send_message(self, text: str) -> bool:
        """Send a message to the configured chat, splitting if needed."""
        try:
            import httpx  # noqa: F401 — ensure available
        except ImportError:
            return False

        # Escape Markdown special chars to prevent HTTP 400 from Telegram API
        text = _escape_md(text)

        url = f"{self._base_url}/sendMessage"
        chunks = _split_message(text, TELEGRAM_MAX_MESSAGE_LEN)

        try:
            client = await self._get_http()
            for chunk in chunks:
                payload = {
                    "chat_id": self._chat_id,
                    "text": chunk,
                    "parse_mode": "Markdown",
                }
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    return False
            return True
        except Exception as e:
            log.warning("telegram_bot.send_error", error=str(e))
            return False

    # ── Scheduled Digest ──────────────────────────────────────────

    async def _send_weekly_digest(self) -> None:
        """Scheduled task: generate and send weekly digest."""
        try:
            response = await self._commands.cmd_weekly()
            await self._send_message(response)
            log.info("telegram_bot.weekly_digest_sent")
        except Exception as e:
            log.warning("telegram_bot.digest_error", error=str(e))


def _split_message(text: str, max_len: int) -> list[str]:
    """Split a long message into chunks at line boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find last newline within limit
        idx = text.rfind("\n", 0, max_len)
        if idx == -1:
            idx = max_len
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks
