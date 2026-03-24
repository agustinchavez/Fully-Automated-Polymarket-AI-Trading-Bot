"""Telegram kill-switch bot — remote engine control from mobile.

Commands:
  /kill    — activate kill switch + persist + alert
  /status  — engine running, kill state, drawdown, positions
  /pnl     — today's P&L
  /resume  — reset kill switch
  /help    — list commands
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


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
        self._running = False
        self._offset = 0
        self._base_url = f"https://api.telegram.org/bot{token}"

    # ── Public API ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start long-polling loop."""
        self._running = True
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
        log.info("telegram_bot.stopped")

    # ── Polling ───────────────────────────────────────────────────

    async def _get_updates(self) -> list[dict]:
        """Fetch updates via long polling."""
        try:
            import httpx
        except ImportError:
            log.warning("telegram_bot.httpx_not_installed")
            await asyncio.sleep(30)
            return []

        url = f"{self._base_url}/getUpdates"
        params = {"offset": self._offset, "timeout": 30}

        async with httpx.AsyncClient(timeout=40) as client:
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

        command = text.split()[0].lower()
        # Strip @botname suffix (e.g. /kill@mybot)
        if "@" in command:
            command = command.split("@")[0]

        handlers = {
            "/kill": self._cmd_kill,
            "/status": self._cmd_status,
            "/pnl": self._cmd_pnl,
            "/resume": self._cmd_resume,
            "/help": self._cmd_help,
        }

        handler = handlers.get(command, self._cmd_unknown)
        response = await handler()
        await self._send_message(response)

    async def _send_message(self, text: str) -> bool:
        """Send a message to the configured chat."""
        try:
            import httpx
        except ImportError:
            return False

        url = f"{self._base_url}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                return resp.status_code == 200
        except Exception as e:
            log.warning("telegram_bot.send_error", error=str(e))
            return False

    # ── Command Handlers ──────────────────────────────────────────

    async def _cmd_kill(self) -> str:
        """Activate kill switch."""
        if not self._engine:
            return "No engine connected."

        try:
            self._engine.drawdown.kill("Telegram /kill command")
            self._engine._persist_kill_switch()
            return (
                "KILL SWITCH ACTIVATED\n\n"
                "Reason: Telegram /kill command\n"
                "All trading halted. Use /resume to reset."
            )
        except Exception as e:
            return f"Kill failed: {e}"

    async def _cmd_status(self) -> str:
        """Return engine status summary."""
        if not self._engine:
            return "No engine connected."

        try:
            dd = self._engine.drawdown.state
            lines = [
                "ENGINE STATUS",
                f"Running: {self._engine.is_running}",
                f"Killed: {dd.is_killed}",
                f"Cycles: {self._engine._cycle_count}",
                f"Drawdown: {dd.drawdown_pct:.1%}",
                f"Peak: ${dd.peak_value:,.2f}",
                f"Current: ${dd.current_value:,.2f}",
            ]

            if dd.kill_reason:
                lines.append(f"Kill reason: {dd.kill_reason}")

            return "\n".join(lines)
        except Exception as e:
            return f"Status error: {e}"

    async def _cmd_pnl(self) -> str:
        """Return today's P&L."""
        if not self._engine or not self._engine._db:
            return "No engine/DB connected."

        try:
            daily_pnl = self._engine._db.get_daily_pnl()
            bankroll = self._engine.config.risk.bankroll
            pct = (daily_pnl / bankroll * 100) if bankroll > 0 else 0.0

            if daily_pnl >= 0:
                pnl_str = f"+${daily_pnl:.2f}"
            else:
                pnl_str = f"-${abs(daily_pnl):.2f}"

            return (
                f"TODAY'S P&L\n\n"
                f"P&L: {pnl_str} ({pct:+.1f}%)\n"
                f"Bankroll: ${bankroll:,.2f}"
            )
        except Exception as e:
            return f"P&L error: {e}"

    async def _cmd_resume(self) -> str:
        """Reset kill switch."""
        if not self._engine:
            return "No engine connected."

        try:
            self._engine.drawdown.state.is_killed = False
            self._engine.drawdown.state.kill_reason = ""

            if self._engine._db:
                self._engine._db.reset_kill_switch()

            return (
                "KILL SWITCH RESET\n\n"
                "Trading will resume on next cycle."
            )
        except Exception as e:
            return f"Resume failed: {e}"

    async def _cmd_help(self) -> str:
        """List available commands."""
        return (
            "TELEGRAM KILL BOT\n\n"
            "/kill - Activate kill switch\n"
            "/status - Engine status\n"
            "/pnl - Today's P&L\n"
            "/resume - Reset kill switch\n"
            "/help - This message"
        )

    async def _cmd_unknown(self) -> str:
        """Handle unknown commands."""
        return "Unknown command. Use /help to see available commands."
