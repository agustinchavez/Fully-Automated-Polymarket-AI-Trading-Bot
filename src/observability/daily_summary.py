"""Daily summary generation — aggregate daily P&L and trading metrics.

At the configured hour (default 18:00 UTC), generates a snapshot of
the day's activity and sends it via configured alert channels.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class DailySummary:
    """Snapshot of a single day's trading activity."""
    summary_date: str = ""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    positions_held: int = 0
    drawdown_pct: float = 0.0
    bankroll: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class DailySummaryGenerator:
    """Generate and persist daily trading summaries."""

    def __init__(self, conn: sqlite3.Connection, bankroll: float = 5000.0):
        self._conn = conn
        self._bankroll = bankroll

    def generate(self, date: str | None = None) -> DailySummary:
        """Generate a summary for the given date (YYYY-MM-DD), defaulting to today."""
        if date is None:
            date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

        summary = DailySummary(summary_date=date, bankroll=self._bankroll)

        # Realized P&L from performance_log
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                "FROM performance_log WHERE date(resolved_at) = ?",
                (date,),
            ).fetchone()
            if row:
                summary.realized_pnl = float(row["total"])
                summary.trades_closed = int(row["cnt"])
        except sqlite3.OperationalError:
            pass

        # Unrealized P&L from open positions
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as total, COUNT(*) as cnt "
                "FROM positions"
            ).fetchone()
            if row:
                summary.unrealized_pnl = float(row["total"])
                summary.positions_held = int(row["cnt"])
        except sqlite3.OperationalError:
            pass

        summary.total_pnl = summary.realized_pnl + summary.unrealized_pnl

        # Trades opened today
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM trades "
                "WHERE date(created_at) = ?",
                (date,),
            ).fetchone()
            if row:
                summary.trades_opened = int(row["cnt"])
        except sqlite3.OperationalError:
            pass

        # Best and worst trade P&L today
        try:
            row = self._conn.execute(
                "SELECT MAX(pnl) as best, MIN(pnl) as worst "
                "FROM performance_log WHERE date(resolved_at) = ?",
                (date,),
            ).fetchone()
            if row and row["best"] is not None:
                summary.best_trade_pnl = float(row["best"])
                summary.worst_trade_pnl = float(row["worst"])
        except sqlite3.OperationalError:
            pass

        # Drawdown from engine_state
        try:
            import json
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'drawdown'"
            ).fetchone()
            if row:
                dd = json.loads(row["value"])
                summary.drawdown_pct = dd.get("drawdown_pct", 0.0)
        except (sqlite3.OperationalError, Exception):
            pass

        return summary

    def persist(self, summary: DailySummary) -> None:
        """Save summary to daily_summaries table."""
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO daily_summaries
                    (summary_date, total_pnl, realized_pnl, unrealized_pnl,
                     trades_opened, trades_closed, positions_held,
                     drawdown_pct, bankroll, best_trade_pnl, worst_trade_pnl,
                     created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    summary.summary_date, summary.total_pnl,
                    summary.realized_pnl, summary.unrealized_pnl,
                    summary.trades_opened, summary.trades_closed,
                    summary.positions_held, summary.drawdown_pct,
                    summary.bankroll, summary.best_trade_pnl,
                    summary.worst_trade_pnl,
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("daily_summary.persist_error", error=str(e))

    @staticmethod
    def format_message(summary: DailySummary) -> str:
        """Format summary as a human-readable alert message."""
        pnl_pct = (summary.total_pnl / summary.bankroll * 100) if summary.bankroll > 0 else 0
        if summary.total_pnl >= 0:
            pnl_str = f"+${summary.total_pnl:.2f}"
        else:
            pnl_str = f"-${abs(summary.total_pnl):.2f}"

        lines = [
            f"Daily Summary — {summary.summary_date}",
            f"{'=' * 35}",
            f"Total P&L: {pnl_str} ({pnl_pct:+.1f}%)",
            f"  Realized:   ${summary.realized_pnl:.2f}",
            f"  Unrealized: ${summary.unrealized_pnl:.2f}",
            f"Trades: {summary.trades_opened} opened, {summary.trades_closed} closed",
            f"Positions held: {summary.positions_held}",
            f"Drawdown: {summary.drawdown_pct:.1%}",
            f"Bankroll: ${summary.bankroll:.2f}",
        ]

        if summary.best_trade_pnl != 0 or summary.worst_trade_pnl != 0:
            lines.append(
                f"Best/Worst: +${summary.best_trade_pnl:.2f} / ${summary.worst_trade_pnl:.2f}"
            )

        return "\n".join(lines)

    async def send_summary(
        self,
        summary: DailySummary,
        alert_manager: Any,
    ) -> None:
        """Send the formatted daily summary via configured alert channels."""
        message = self.format_message(summary)
        try:
            await alert_manager.send(
                level="info",
                title=f"Daily Summary — {summary.summary_date}",
                message=message,
                cooldown_key="daily_summary",
            )
            log.info("daily_summary.sent", date=summary.summary_date)
        except Exception as e:
            log.warning("daily_summary.send_error", error=str(e))
