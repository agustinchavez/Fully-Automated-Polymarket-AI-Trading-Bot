"""Shared command handler for all bot platforms (Telegram, Discord, Slack).

Contains the business logic for 11 bot commands. Each platform bot
(TelegramKillBot, DiscordKillBot, SlackKillBot) delegates to this class
and only handles transport-specific concerns (polling, sending messages).
"""

from __future__ import annotations

from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


class BotCommandHandler:
    """Platform-agnostic command handler for engine control bots."""

    COMMANDS = (
        "kill", "status", "pnl", "resume", "weekly",
        "report", "insights", "models", "analyze", "provider", "help",
    )

    def __init__(self, engine: Any = None):
        self._engine = engine

    async def dispatch(self, command: str, args: str = "") -> str:
        """Route a command to its handler and return the response text.

        Args:
            command: The command name (without prefix), e.g. "kill", "status".
            args: Optional argument string, e.g. "14" for "report 14".

        Returns:
            Plain text response suitable for any platform.
        """
        handlers = {
            "kill": self.cmd_kill,
            "status": self.cmd_status,
            "pnl": self.cmd_pnl,
            "resume": self.cmd_resume,
            "weekly": self.cmd_weekly,
            "report": self.cmd_report,
            "insights": self.cmd_insights,
            "models": self.cmd_models,
            "analyze": self.cmd_analyze,
            "provider": self.cmd_provider,
            "help": self.cmd_help,
        }

        handler = handlers.get(command)
        if not handler:
            return await self.cmd_unknown()

        # Commands that accept arguments
        if command == "report":
            days = _parse_int(args, default=7, min_val=3, max_val=30)
            return await self.cmd_report(days)
        if command == "analyze":
            days = _parse_int(args, default=30, min_val=7, max_val=90)
            return await self.cmd_analyze(days)

        return await handler()

    # ── Engine Control Commands ─────────────────────────────────────

    async def cmd_kill(self) -> str:
        """Activate kill switch."""
        if not self._engine:
            return "No engine connected."
        try:
            self._engine.drawdown.kill("Bot /kill command")
            self._engine._persist_kill_switch()
            return (
                "KILL SWITCH ACTIVATED\n\n"
                "Reason: Bot /kill command\n"
                "All trading halted. Use /resume to reset."
            )
        except Exception as e:
            return f"Kill failed: {e}"

    async def cmd_status(self) -> str:
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

    async def cmd_pnl(self) -> str:
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

    async def cmd_resume(self) -> str:
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

    # ── Digest Commands ─────────────────────────────────────────────

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

    async def cmd_weekly(self) -> str:
        """Send the full weekly digest on demand."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=7)
            return gen.format_telegram(digest)
        except Exception as e:
            return f"Digest error: {e}"

    async def cmd_report(self, days: int = 7) -> str:
        """Send digest for last N days."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=days)
            return gen.format_telegram(digest)
        except Exception as e:
            return f"Report error: {e}"

    async def cmd_insights(self) -> str:
        """One-line summary of most actionable insight."""
        gen = self._get_digest_generator()
        if not gen:
            return "No engine/DB connected."
        try:
            digest = gen.generate(days=7)
            return gen.format_short(digest)
        except Exception as e:
            return f"Insights error: {e}"

    async def cmd_models(self) -> str:
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

    # ── AI Analysis Commands ────────────────────────────────────────

    async def cmd_analyze(self, days: int = 30) -> str:
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
                bot_config=config,
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

    async def cmd_provider(self) -> str:
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

    async def cmd_help(self) -> str:
        """List available commands."""
        return (
            "BOT COMMANDS\n\n"
            "kill - Activate kill switch\n"
            "status - Engine status\n"
            "pnl - Today's P&L\n"
            "resume - Reset kill switch\n"
            "weekly - Weekly performance digest\n"
            "report N - Digest for last N days\n"
            "insights - One-line insight\n"
            "models - Model accuracy (30d)\n"
            "analyze N - AI analysis (default 30d)\n"
            "provider - Show AI provider config\n"
            "help - This message"
        )

    async def cmd_unknown(self) -> str:
        """Handle unknown commands."""
        return "Unknown command. Use help to see available commands."


def _parse_int(s: str, *, default: int, min_val: int, max_val: int) -> int:
    """Parse an integer argument with clamping."""
    s = s.strip()
    if not s:
        return default
    try:
        return max(min_val, min(max_val, int(s)))
    except ValueError:
        return default
