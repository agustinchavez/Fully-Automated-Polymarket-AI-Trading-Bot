"""Run reports — generate JSON summaries and weekly digests.

Contains:
  - generate_run_report(): JSON run summary (original)
  - WeeklyDigestGenerator: multi-day performance digest for Telegram
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.observability.logger import get_logger
from src.observability.metrics import metrics

log = get_logger(__name__)

MIN_DAYS_FOR_DIGEST = 3
MIN_TRADES_FOR_MODEL_ACCURACY = 10
TELEGRAM_MAX_LEN = 4096


def _escape_md(text: str) -> str:
    """Escape Telegram MarkdownV1 special characters."""
    for ch in ("*", "_", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


# ── Digest Dataclasses ───────────────────────────────────────────────


@dataclass
class CategoryDigest:
    """Per-category performance summary."""
    category: str = ""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    avg_edge_at_entry: float = 0.0
    win_rate: float = 0.0


@dataclass
class ModelDigest:
    """Per-model forecast accuracy summary."""
    model_name: str = ""
    forecasts: int = 0
    brier_score: float = 0.0
    directional_accuracy: float = 0.0
    avg_error: float = 0.0


@dataclass
class FrictionDigest:
    """Edge-vs-realized friction analysis."""
    avg_edge_at_entry: float = 0.0
    avg_pnl_per_trade: float = 0.0
    friction_gap: float = 0.0
    fee_cost_total: float = 0.0


@dataclass
class WeeklyDigest:
    """Full weekly digest data."""
    period_start: str = ""
    period_end: str = ""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    total_trades_opened: int = 0
    total_trades_resolved: int = 0
    roi_pct: float = 0.0
    sharpe_7d: float = 0.0
    max_drawdown_pct: float = 0.0
    category_breakdown: list[CategoryDigest] = field(default_factory=list)
    model_accuracy: list[ModelDigest] = field(default_factory=list)
    friction_analysis: FrictionDigest = field(default_factory=FrictionDigest)
    best_trade: str = ""
    worst_trade: str = ""
    markets_evaluated: int = 0
    markets_traded_pct: float = 0.0
    data_days_available: int = 0
    bankroll: float = 0.0
    data_sufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "win_rate": self.win_rate,
            "total_trades_opened": self.total_trades_opened,
            "total_trades_resolved": self.total_trades_resolved,
            "roi_pct": self.roi_pct,
            "sharpe_7d": self.sharpe_7d,
            "max_drawdown_pct": self.max_drawdown_pct,
            "category_breakdown": [
                {"category": c.category, "trades": c.trades, "wins": c.wins,
                 "total_pnl": c.total_pnl, "avg_edge_at_entry": c.avg_edge_at_entry,
                 "win_rate": c.win_rate}
                for c in self.category_breakdown
            ],
            "model_accuracy": [
                {"model_name": m.model_name, "forecasts": m.forecasts,
                 "brier_score": m.brier_score, "directional_accuracy": m.directional_accuracy,
                 "avg_error": m.avg_error}
                for m in self.model_accuracy
            ],
            "friction_analysis": {
                "avg_edge_at_entry": self.friction_analysis.avg_edge_at_entry,
                "avg_pnl_per_trade": self.friction_analysis.avg_pnl_per_trade,
                "friction_gap": self.friction_analysis.friction_gap,
                "fee_cost_total": self.friction_analysis.fee_cost_total,
            },
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "markets_evaluated": self.markets_evaluated,
            "markets_traded_pct": self.markets_traded_pct,
            "data_days_available": self.data_days_available,
            "data_sufficient": self.data_sufficient,
        }


# ── WeeklyDigestGenerator ───────────────────────────────────────────


class WeeklyDigestGenerator:
    """Generate multi-day performance digests from the trading database."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        bankroll: float = 5000.0,
        transaction_fee_pct: float = 0.02,
    ) -> None:
        self._conn = conn
        self._bankroll = bankroll
        self._transaction_fee_pct = transaction_fee_pct

    def generate(self, days: int = 7) -> WeeklyDigest:
        """Generate a digest covering the last ``days`` days."""
        end_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        start_date = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        ).strftime("%Y-%m-%d")

        digest = WeeklyDigest(
            period_start=start_date,
            period_end=end_date,
            bankroll=self._bankroll,
        )

        # Check data availability
        digest.data_days_available = self._count_data_days(start_date, end_date)
        if digest.data_days_available < MIN_DAYS_FOR_DIGEST:
            digest.data_sufficient = False
            return digest
        digest.data_sufficient = True

        self._fill_pnl(digest, start_date, end_date)
        self._fill_win_rate(digest, start_date, end_date)
        self._fill_trades_opened(digest, start_date, end_date)
        self._fill_sharpe(digest, start_date, end_date)
        self._fill_categories(digest, start_date, end_date)
        self._fill_model_accuracy(digest, start_date, end_date)
        self._fill_friction(digest, start_date, end_date)
        self._fill_best_worst(digest, start_date, end_date)
        self._fill_markets_evaluated(digest, start_date, end_date)

        # Computed fields
        if self._bankroll > 0:
            digest.roi_pct = digest.total_pnl / self._bankroll * 100
        if digest.markets_evaluated > 0:
            digest.markets_traded_pct = (
                digest.total_trades_opened / digest.markets_evaluated * 100
            )

        return digest

    # ── Query helpers ────────────────────────────────────────────

    def _count_data_days(self, start: str, end: str) -> int:
        try:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT summary_date) as cnt "
                "FROM daily_summaries WHERE summary_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            return int(row["cnt"]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def _fill_pnl(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(total_pnl), 0) as total, "
                "COALESCE(SUM(realized_pnl), 0) as realized, "
                "COALESCE(MAX(drawdown_pct), 0) as max_dd "
                "FROM daily_summaries WHERE summary_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row:
                d.total_pnl = float(row["total"])
                d.realized_pnl = float(row["realized"])
                d.max_drawdown_pct = float(row["max_dd"])
        except sqlite3.OperationalError:
            pass

        try:
            row = self._conn.execute(
                "SELECT unrealized_pnl FROM daily_summaries "
                "WHERE summary_date BETWEEN ? AND ? "
                "ORDER BY summary_date DESC LIMIT 1",
                (start, end),
            ).fetchone()
            if row:
                d.unrealized_pnl = float(row["unrealized_pnl"])
        except sqlite3.OperationalError:
            pass

    def _fill_win_rate(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins "
                "FROM performance_log WHERE date(resolved_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row and int(row["total"]) > 0:
                d.total_trades_resolved = int(row["total"])
                wins = int(row["wins"])
                d.win_rate = wins / d.total_trades_resolved * 100
        except sqlite3.OperationalError:
            pass

    def _fill_trades_opened(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM trades "
                "WHERE date(created_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row:
                d.total_trades_opened = int(row["cnt"])
        except sqlite3.OperationalError:
            pass

    def _fill_sharpe(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            rows = self._conn.execute(
                "SELECT total_pnl FROM daily_summaries "
                "WHERE summary_date BETWEEN ? AND ? ORDER BY summary_date",
                (start, end),
            ).fetchall()
            if len(rows) >= MIN_DAYS_FOR_DIGEST:
                pnls = [float(r["total_pnl"]) for r in rows]
                mean_pnl = sum(pnls) / len(pnls)
                variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
                std = math.sqrt(variance) if variance > 0 else 0.0
                d.sharpe_7d = round(mean_pnl / std, 2) if std > 0 else 0.0
        except sqlite3.OperationalError:
            pass

    def _fill_categories(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            rows = self._conn.execute(
                "SELECT f.market_type as category, "
                "COUNT(*) as trades, "
                "SUM(CASE WHEN p.pnl > 0 THEN 1 ELSE 0 END) as wins, "
                "SUM(p.pnl) as total_pnl, "
                "AVG(f.edge) as avg_edge "
                "FROM performance_log p "
                "JOIN forecasts f ON p.market_id = f.market_id "
                "WHERE date(p.resolved_at) BETWEEN ? AND ? "
                "GROUP BY f.market_type ORDER BY total_pnl DESC",
                (start, end),
            ).fetchall()
            for row in rows:
                cat = CategoryDigest(
                    category=row["category"] or "UNKNOWN",
                    trades=int(row["trades"]),
                    wins=int(row["wins"]),
                    total_pnl=float(row["total_pnl"]),
                    avg_edge_at_entry=float(row["avg_edge"] or 0),
                )
                if cat.trades > 0:
                    cat.win_rate = cat.wins / cat.trades * 100
                d.category_breakdown.append(cat)
        except sqlite3.OperationalError:
            pass

    def _fill_model_accuracy(self, d: WeeklyDigest, start: str, end: str) -> None:
        """Query model_forecast_log -- uses recorded_at (no resolved_at column)."""
        try:
            rows = self._conn.execute(
                "SELECT model_name, "
                "COUNT(*) as forecasts, "
                "AVG((forecast_prob - actual_outcome) "
                "   * (forecast_prob - actual_outcome)) as brier, "
                "AVG(ABS(forecast_prob - actual_outcome)) as avg_error "
                "FROM model_forecast_log "
                "WHERE actual_outcome IS NOT NULL "
                "AND date(recorded_at) BETWEEN ? AND ? "
                "GROUP BY model_name",
                (start, end),
            ).fetchall()
            for row in rows:
                count = int(row["forecasts"])
                if count < MIN_TRADES_FOR_MODEL_ACCURACY:
                    continue
                md = ModelDigest(
                    model_name=row["model_name"],
                    forecasts=count,
                    brier_score=round(float(row["brier"] or 0), 4),
                    avg_error=round(float(row["avg_error"] or 0), 4),
                )
                # Directional accuracy
                dir_row = self._conn.execute(
                    "SELECT SUM(CASE WHEN "
                    "(forecast_prob >= 0.5 AND actual_outcome = 1) OR "
                    "(forecast_prob < 0.5 AND actual_outcome = 0) "
                    "THEN 1 ELSE 0 END) as correct, COUNT(*) as total "
                    "FROM model_forecast_log "
                    "WHERE actual_outcome IS NOT NULL "
                    "AND date(recorded_at) BETWEEN ? AND ? "
                    "AND model_name = ?",
                    (start, end, row["model_name"]),
                ).fetchone()
                if dir_row and int(dir_row["total"]) > 0:
                    md.directional_accuracy = round(
                        int(dir_row["correct"]) / int(dir_row["total"]) * 100, 1
                    )
                d.model_accuracy.append(md)
        except sqlite3.OperationalError:
            pass

    def _fill_friction(self, d: WeeklyDigest, start: str, end: str) -> None:
        """Friction = avg edge at entry - avg realized ROI. Uses config fee_pct."""
        try:
            row = self._conn.execute(
                "SELECT AVG(f.edge) as avg_edge, "
                "AVG(p.pnl / NULLIF(p.stake_usd, 0)) as avg_roi, "
                "SUM(p.stake_usd) as total_stake "
                "FROM performance_log p "
                "JOIN forecasts f ON p.market_id = f.market_id "
                "WHERE date(p.resolved_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row and row["avg_edge"] is not None:
                avg_edge = float(row["avg_edge"])
                avg_roi = float(row["avg_roi"] or 0)
                total_stake = float(row["total_stake"] or 0)
                d.friction_analysis = FrictionDigest(
                    avg_edge_at_entry=round(avg_edge * 100, 2),
                    avg_pnl_per_trade=round(avg_roi * 100, 2),
                    friction_gap=round((avg_edge - avg_roi) * 100, 2),
                    fee_cost_total=round(
                        total_stake * self._transaction_fee_pct * 2, 2
                    ),
                )
        except sqlite3.OperationalError:
            pass

    def _fill_best_worst(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            best = self._conn.execute(
                "SELECT question, pnl FROM performance_log "
                "WHERE date(resolved_at) BETWEEN ? AND ? "
                "ORDER BY pnl DESC LIMIT 1",
                (start, end),
            ).fetchone()
            if best:
                d.best_trade = f"{_escape_md(best['question'][:60])} +${best['pnl']:.2f}"
        except sqlite3.OperationalError:
            pass

        try:
            worst = self._conn.execute(
                "SELECT question, pnl FROM performance_log "
                "WHERE date(resolved_at) BETWEEN ? AND ? "
                "ORDER BY pnl ASC LIMIT 1",
                (start, end),
            ).fetchone()
            if worst:
                d.worst_trade = f"{_escape_md(worst['question'][:60])} ${worst['pnl']:.2f}"
        except sqlite3.OperationalError:
            pass

    def _fill_markets_evaluated(self, d: WeeklyDigest, start: str, end: str) -> None:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM forecasts "
                "WHERE date(created_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row:
                d.markets_evaluated = int(row["cnt"])
        except sqlite3.OperationalError:
            pass

    # ── Formatting ───────────────────────────────────────────────

    def format_telegram(self, digest: WeeklyDigest) -> str:
        """Format digest as a Telegram-friendly Markdown message."""
        if not digest.data_sufficient:
            return (
                "*Weekly Digest*\n\n"
                f"Not enough data yet. Need at least {MIN_DAYS_FOR_DIGEST} days "
                f"of trading data (have {digest.data_days_available}).\n\n"
                "Keep paper trading and check back later."
            )

        pnl_sign = "+" if digest.total_pnl >= 0 else ""
        lines = [
            f"*Weekly Digest -- {digest.period_start} to {digest.period_end}*",
            "",
            "*P&L*",
            (
                f"Net: {pnl_sign}${digest.total_pnl:.2f} "
                f"({pnl_sign}{digest.roi_pct:.1f}%) | "
                f"Realized: ${digest.realized_pnl:.2f} | "
                f"Unrealized: ${digest.unrealized_pnl:.2f}"
            ),
            f"Sharpe (7d): {digest.sharpe_7d} | Max drawdown: {digest.max_drawdown_pct:.1%}",
        ]

        if digest.total_trades_resolved > 0:
            wins = int(digest.win_rate / 100 * digest.total_trades_resolved)
            lines.append(
                f"Win rate: {wins}/{digest.total_trades_resolved} "
                f"({digest.win_rate:.1f}%) on "
                f"{digest.total_trades_resolved} resolved trades"
            )

        if digest.category_breakdown:
            lines.append("")
            lines.append("*Categories*")
            for cat in digest.category_breakdown[:5]:
                pnl_s = (
                    f"+${cat.total_pnl:.2f}"
                    if cat.total_pnl >= 0
                    else f"${cat.total_pnl:.2f}"
                )
                lines.append(
                    f"{cat.category:12s} {pnl_s}  {cat.win_rate:.0f}% WR  "
                    f"edge: {cat.avg_edge_at_entry:.1%}"
                )

        if digest.model_accuracy:
            lines.append("")
            total_fc = sum(m.forecasts for m in digest.model_accuracy)
            lines.append(f"*Model Accuracy* ({total_fc} forecasts with outcomes)")
            for m in digest.model_accuracy:
                lines.append(
                    f"{m.model_name:20s} Brier: {m.brier_score:.3f}  "
                    f"Dir: {m.directional_accuracy:.0f}%"
                )

        fa = digest.friction_analysis
        if fa.avg_edge_at_entry != 0:
            lines.append("")
            lines.append("*Friction*")
            lines.append(
                f"Avg edge at entry: {fa.avg_edge_at_entry:.1f}% | "
                f"Avg realised: {fa.avg_pnl_per_trade:.1f}%"
            )
            lines.append(
                f"Gap: {fa.friction_gap:.1f}%  |  "
                f"Total fees: ${fa.fee_cost_total:.2f}"
            )

        if digest.best_trade:
            lines.append("")
            lines.append(f"Best: {digest.best_trade}")
        if digest.worst_trade:
            lines.append(f"Worst: {digest.worst_trade}")

        if digest.markets_evaluated > 0:
            lines.append("")
            lines.append(
                f"Markets evaluated: {digest.markets_evaluated} | "
                f"Traded: {digest.total_trades_opened} "
                f"({digest.markets_traded_pct:.1f}%)"
            )

        return "\n".join(lines)

    @staticmethod
    def split_message(text: str, max_len: int = TELEGRAM_MAX_LEN) -> list[str]:
        """Split a long message at line boundaries for Telegram limits."""
        if len(text) <= max_len:
            return [text]

        parts: list[str] = []
        current: list[str] = []
        current_len = 0

        for line in text.split("\n"):
            line_len = len(line) + 1
            if current_len + line_len > max_len and current:
                parts.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len

        if current:
            parts.append("\n".join(current))

        return parts

    def format_short(self, digest: WeeklyDigest) -> str:
        """One-line summary for /insights command."""
        if not digest.data_sufficient:
            return (
                f"Not enough data "
                f"({digest.data_days_available}/{MIN_DAYS_FOR_DIGEST} days)."
            )
        pnl_sign = "+" if digest.total_pnl >= 0 else ""
        return (
            f"{pnl_sign}${digest.total_pnl:.2f} "
            f"({pnl_sign}{digest.roi_pct:.1f}%) | "
            f"WR {digest.win_rate:.0f}% | Sharpe {digest.sharpe_7d}"
        )


# ── Original Run Report ──────────────────────────────────────────────


def generate_run_report(
    run_id: str,
    forecasts: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    output_dir: str = "reports/",
) -> Path:
    """Generate a JSON run report and write to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "summary": {
            "markets_scanned": len(forecasts),
            "trades_executed": len(trades),
            "trades_skipped": len(forecasts) - len(trades),
        },
        "metrics": metrics.snapshot(),
        "forecasts": forecasts,
        "trades": trades,
    }

    filepath = out / f"run_{run_id}.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("report.generated", path=str(filepath), markets=len(forecasts))
    return filepath
