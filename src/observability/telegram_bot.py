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
        self._scheduler: Any = None

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
        log.info("telegram_bot.stopped")

    def _start_scheduler(self) -> None:
        """Start APScheduler for weekly digest cron job."""
        if not self._engine:
            return
        config = getattr(self._engine, "config", None)
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
            "/weekly": self._cmd_weekly,
            "/report": self._cmd_report,
            "/insights": self._cmd_insights,
            "/models": self._cmd_models,
            "/analyze": self._cmd_analyze,
            "/provider": self._cmd_provider,
            "/help": self._cmd_help,
        }

        handler = handlers.get(command, self._cmd_unknown)
        # Commands that accept arguments
        if command == "/report":
            parts = text.split()
            days = 7
            if len(parts) > 1:
                try:
                    days = max(3, min(30, int(parts[1])))
                except ValueError:
                    pass
            response = await self._cmd_report(days)
        elif command == "/analyze":
            parts = text.split()
            days = 30
            if len(parts) > 1:
                try:
                    days = max(7, min(90, int(parts[1])))
                except ValueError:
                    pass
            response = await self._cmd_analyze(days)
        else:
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

    # ── Digest Commands ───────────────────────────────────────────

    def _get_digest_generator(self) -> Any:
        """Create a WeeklyDigestGenerator from the engine's DB."""
        from src.observability.reports import WeeklyDigestGenerator
        if not self._engine or not self._engine._db:
            return None
        bankroll = self._engine.config.risk.bankroll
        fee_pct = self._engine.config.risk.transaction_fee_pct
        return WeeklyDigestGenerator(
            conn=self._engine._db.conn,
            bankroll=bankroll,
            transaction_fee_pct=fee_pct,
        )

    async def _send_weekly_digest(self) -> None:
        """Scheduled task: generate and send weekly digest."""
        gen = self._get_digest_generator()
        if not gen:
            return
        try:
            days = 7
            config = getattr(self._engine, "config", None)
            digest_cfg = getattr(config, "digest", None)
            if digest_cfg:
                days = digest_cfg.lookback_days
            digest = gen.generate(days=days)
            message = gen.format_telegram(digest)
            parts = gen.split_message(message)
            for part in parts:
                await self._send_message(part)
            log.info("telegram_bot.weekly_digest_sent")
        except Exception as e:
            log.warning("telegram_bot.digest_error", error=str(e))

    async def _cmd_weekly(self) -> str:
        """Send the full weekly digest on demand."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=7)
            message = gen.format_telegram(digest)
            parts = gen.split_message(message)
            if len(parts) > 1:
                for part in parts[:-1]:
                    await self._send_message(part)
                return parts[-1]
            return message
        except Exception as e:
            return f"Digest error: {e}"

    async def _cmd_report(self, days: int = 7) -> str:
        """Send digest for last N days."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=days)
            message = gen.format_telegram(digest)
            parts = gen.split_message(message)
            if len(parts) > 1:
                for part in parts[:-1]:
                    await self._send_message(part)
                return parts[-1]
            return message
        except Exception as e:
            return f"Report error: {e}"

    async def _cmd_insights(self) -> str:
        """One-line summary of most actionable insight."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=7)
            return gen.format_short(digest)
        except Exception as e:
            return f"Insights error: {e}"

    async def _cmd_models(self) -> str:
        """Model accuracy table for last 30 days."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=30)
            if not digest.data_sufficient:
                return "Not enough data for model accuracy report."
            if not digest.model_accuracy:
                return "No model accuracy data available (need 10+ resolved forecasts)."
            lines = ["*Model Accuracy (30 days)*", ""]
            for m in digest.model_accuracy:
                lines.append(
                    f"{m.model_name}\n"
                    f"  Brier: {m.brier_score:.3f} | Dir: {m.directional_accuracy:.0f}% "
                    f"| {m.forecasts} forecasts"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Models error: {e}"

    # ── AI Analysis Commands ───────────────────────────────────────

    async def _cmd_analyze(self, days: int = 30) -> str:
        """Run AI analysis of bot performance."""
        if not self._engine or not self._engine._db:
            return "No engine/DB connected."
        config = getattr(self._engine, "config", None)
        analyst_cfg = getattr(config, "analyst", None)
        if not analyst_cfg or not analyst_cfg.enabled:
            return "AI analyst is disabled. Set analyst.enabled=true in config."
        try:
            from src.analytics.ai_analyst import AIAnalyst
            analyst = AIAnalyst(
                conn=self._engine._db.conn,
                config=analyst_cfg,
            )
            result = await analyst.analyse(days=days)
            if not result.data_sufficient:
                return result.summary
            badge = f"[{result.provider_used}/{result.model_used}]"
            lines = [f"*AI Analysis* {badge}", ""]
            if result.summary:
                lines.append(result.summary)
                lines.append("")
            if result.what_is_working:
                lines.append("*Working well:*")
                for item in result.what_is_working[:3]:
                    lines.append(f"  + {item}")
                lines.append("")
            if result.what_is_not_working:
                lines.append("*Needs improvement:*")
                for item in result.what_is_not_working[:3]:
                    lines.append(f"  - {item}")
                lines.append("")
            if result.recommendations:
                lines.append("*Recommendations:*")
                for r in result.recommendations[:3]:
                    lines.append(f"  {r.priority}. {r.action}")
            return "\n".join(lines)
        except Exception as e:
            return f"Analysis error: {e}"

    async def _cmd_provider(self) -> str:
        """Show configured AI provider and model."""
        config = getattr(self._engine, "config", None) if self._engine else None
        analyst_cfg = getattr(config, "analyst", None)
        if not analyst_cfg:
            return "AI analyst not configured."
        return (
            f"*AI Analyst Config*\n"
            f"Enabled: {analyst_cfg.enabled}\n"
            f"Provider: {analyst_cfg.provider}\n"
            f"Model: {analyst_cfg.model}\n"
            f"Rate limit: {analyst_cfg.rate_limit_hours}h"
        )

    async def _cmd_help(self) -> str:
        """List available commands."""
        return (
            "TELEGRAM BOT\n\n"
            "/kill - Activate kill switch\n"
            "/status - Engine status\n"
            "/pnl - Today's P&L\n"
            "/resume - Reset kill switch\n"
            "/weekly - Weekly performance digest\n"
            "/report N - Digest for last N days\n"
            "/insights - One-line insight\n"
            "/models - Model accuracy (30d)\n"
            "/analyze N - AI analysis (default 30d)\n"
            "/provider - Show AI provider config\n"
            "/help - This message"
        )

    async def _cmd_unknown(self) -> str:
        """Handle unknown commands."""
        return "Unknown command. Use /help to see available commands."
