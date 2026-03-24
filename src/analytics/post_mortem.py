"""Post-resolution analysis pipeline — learn from every resolved market.

Automatically analyzes resolved markets to determine:
  - Was the forecast correct?
  - Which model was closest / furthest?
  - Was the position size appropriate?
  - Generates weekly summaries with actionable insights.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class TradeAnalysis:
    """Analysis of a single resolved trade."""
    market_id: str
    question: str = ""
    category: str = ""
    forecast_prob: float = 0.0
    actual_outcome: float = 0.0
    was_correct: bool = False
    confidence_error: float = 0.0
    was_confident_and_wrong: bool = False
    best_model: str = ""
    worst_model: str = ""
    model_errors: dict[str, float] = field(default_factory=dict)
    evidence_quality: float = 0.0
    evidence_sources: list[str] = field(default_factory=list)
    position_size_appropriate: str = ""
    pnl: float = 0.0
    edge_at_entry: float = 0.0
    holding_hours: float = 0.0
    analyzed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "category": self.category,
            "forecast_prob": self.forecast_prob,
            "actual_outcome": self.actual_outcome,
            "was_correct": self.was_correct,
            "confidence_error": self.confidence_error,
            "was_confident_and_wrong": self.was_confident_and_wrong,
            "best_model": self.best_model,
            "worst_model": self.worst_model,
            "model_errors": self.model_errors,
            "evidence_quality": self.evidence_quality,
            "evidence_sources": self.evidence_sources,
            "position_size_appropriate": self.position_size_appropriate,
            "pnl": self.pnl,
            "edge_at_entry": self.edge_at_entry,
            "holding_hours": self.holding_hours,
            "analyzed_at": self.analyzed_at,
        }


@dataclass
class WeeklySummary:
    """Aggregated weekly performance summary."""
    period_start: str = ""
    period_end: str = ""
    total_resolved: int = 0
    correct_count: int = 0
    accuracy_pct: float = 0.0
    top_winning_categories: list[dict[str, Any]] = field(default_factory=list)
    top_losing_categories: list[dict[str, Any]] = field(default_factory=list)
    most_accurate_model: str = ""
    least_accurate_model: str = ""
    best_evidence_sources: list[str] = field(default_factory=list)
    worst_evidence_sources: list[str] = field(default_factory=list)
    confident_wrong_count: int = 0
    avg_position_sizing_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "total_resolved": self.total_resolved,
            "correct_count": self.correct_count,
            "accuracy_pct": self.accuracy_pct,
            "top_winning_categories": self.top_winning_categories,
            "top_losing_categories": self.top_losing_categories,
            "most_accurate_model": self.most_accurate_model,
            "least_accurate_model": self.least_accurate_model,
            "best_evidence_sources": self.best_evidence_sources,
            "worst_evidence_sources": self.worst_evidence_sources,
            "confident_wrong_count": self.confident_wrong_count,
            "avg_position_sizing_score": self.avg_position_sizing_score,
        }


class PostMortemAnalyzer:
    """Analyzes resolved markets for continuous learning."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def analyze_market(self, market_id: str) -> TradeAnalysis | None:
        """Analyze a single resolved market and save results."""
        try:
            row = self._conn.execute(
                "SELECT * FROM performance_log WHERE market_id = ?",
                (market_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None

        if not row:
            return None

        forecast_prob = float(row["forecast_prob"] or 0)
        actual_outcome = float(row["actual_outcome"] if row["actual_outcome"] is not None else 0)

        # Determine correctness
        if actual_outcome == 1.0:
            was_correct = forecast_prob >= 0.5
        elif actual_outcome == 0.0:
            was_correct = forecast_prob < 0.5
        else:
            was_correct = False

        confidence_error = abs(forecast_prob - actual_outcome)

        # Confident and wrong: forecast far from 0.5 but outcome opposite
        was_confident_and_wrong = (
            not was_correct
            and (forecast_prob >= 0.75 or forecast_prob <= 0.25)
        )

        # Model errors
        model_errors = self._get_model_errors(market_id)
        best_model = ""
        worst_model = ""
        if model_errors:
            best_model = min(model_errors, key=model_errors.get)
            worst_model = max(model_errors, key=model_errors.get)

        # Position size appropriateness
        position_size_appropriate = self._compute_position_appropriateness(
            forecast_prob=forecast_prob,
            actual_outcome=actual_outcome,
            stake_usd=float(row["stake_usd"] or 0),
            bankroll=5000.0,
            kelly_fraction=0.25,
        )

        analysis = TradeAnalysis(
            market_id=market_id,
            question=row["question"] or "",
            category=row["category"] or "",
            forecast_prob=forecast_prob,
            actual_outcome=actual_outcome,
            was_correct=was_correct,
            confidence_error=round(confidence_error, 4),
            was_confident_and_wrong=was_confident_and_wrong,
            best_model=best_model,
            worst_model=worst_model,
            model_errors=model_errors,
            evidence_quality=float(row["evidence_quality"] or 0),
            position_size_appropriate=position_size_appropriate,
            pnl=float(row["pnl"] or 0),
            edge_at_entry=float(row["edge_at_entry"] or 0),
            holding_hours=float(row["holding_hours"] or 0),
            analyzed_at=_now_iso(),
        )

        self.save_analysis(analysis)

        log.info(
            "post_mortem.analyzed",
            market_id=market_id[:8],
            was_correct=was_correct,
            confident_wrong=was_confident_and_wrong,
            best_model=best_model,
        )

        return analysis

    def analyze_all_pending(self) -> list[TradeAnalysis]:
        """Find and analyze all resolved markets not yet analyzed."""
        try:
            rows = self._conn.execute("""
                SELECT p.market_id
                FROM performance_log p
                LEFT JOIN trade_analysis t ON p.market_id = t.market_id
                WHERE p.actual_outcome IS NOT NULL
                  AND t.market_id IS NULL
            """).fetchall()
        except sqlite3.OperationalError:
            return []

        results = []
        for row in rows:
            analysis = self.analyze_market(row["market_id"])
            if analysis:
                results.append(analysis)

        log.info("post_mortem.batch_complete", analyzed=len(results))
        return results

    def generate_weekly_summary(self, lookback_days: int = 7) -> WeeklySummary:
        """Generate an aggregated weekly summary from trade analyses."""
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        start = now - dt.timedelta(days=lookback_days)
        period_start = start.isoformat()
        period_end = now.isoformat()

        try:
            rows = self._conn.execute("""
                SELECT * FROM trade_analysis
                WHERE analyzed_at >= ?
                ORDER BY analyzed_at DESC
            """, (period_start,)).fetchall()
        except sqlite3.OperationalError:
            return WeeklySummary(period_start=period_start, period_end=period_end)

        if not rows:
            return WeeklySummary(period_start=period_start, period_end=period_end)

        total = len(rows)
        correct = sum(1 for r in rows if r["was_correct"])
        confident_wrong = sum(1 for r in rows if r["was_confident_and_wrong"])

        # Category breakdown
        cat_pnl: dict[str, float] = {}
        for r in rows:
            cat = r["category"] or "UNKNOWN"
            cat_pnl[cat] = cat_pnl.get(cat, 0.0) + float(r["pnl"] or 0)

        sorted_cats = sorted(cat_pnl.items(), key=lambda x: x[1], reverse=True)
        top_winning = [
            {"category": c, "pnl": round(p, 2)}
            for c, p in sorted_cats[:3] if p > 0
        ]
        top_losing = [
            {"category": c, "pnl": round(p, 2)}
            for c, p in sorted_cats[-3:] if p < 0
        ]

        # Model accuracy from model_forecast_log
        most_accurate = ""
        least_accurate = ""
        try:
            model_rows = self._conn.execute("""
                SELECT model_name,
                       AVG((forecast_prob - actual_outcome) *
                           (forecast_prob - actual_outcome)) as brier
                FROM model_forecast_log
                WHERE recorded_at >= ?
                GROUP BY model_name
                HAVING COUNT(*) >= 3
            """, (period_start,)).fetchall()
            if model_rows:
                sorted_models = sorted(model_rows, key=lambda x: float(x["brier"]))
                most_accurate = sorted_models[0]["model_name"]
                least_accurate = sorted_models[-1]["model_name"]
        except sqlite3.OperationalError:
            pass

        # Position sizing score
        sizing_scores = {"appropriate": 1.0, "too_small": 0.5, "too_large": 0.0}
        sizing_values = [
            sizing_scores.get(r["position_size_appropriate"], 0.5) for r in rows
        ]
        avg_sizing = sum(sizing_values) / len(sizing_values) if sizing_values else 0.0

        return WeeklySummary(
            period_start=period_start,
            period_end=period_end,
            total_resolved=total,
            correct_count=correct,
            accuracy_pct=round(correct / total * 100, 1) if total > 0 else 0.0,
            top_winning_categories=top_winning,
            top_losing_categories=top_losing,
            most_accurate_model=most_accurate,
            least_accurate_model=least_accurate,
            confident_wrong_count=confident_wrong,
            avg_position_sizing_score=round(avg_sizing, 2),
        )

    def save_analysis(self, analysis: TradeAnalysis) -> None:
        """Upsert a trade analysis into the database."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO trade_analysis
                    (market_id, question, category, forecast_prob,
                     actual_outcome, was_correct, confidence_error,
                     was_confident_and_wrong, best_model, worst_model,
                     model_errors_json, evidence_quality,
                     evidence_sources_json, position_size_appropriate,
                     pnl, edge_at_entry, holding_hours, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.market_id, analysis.question, analysis.category,
                analysis.forecast_prob, analysis.actual_outcome,
                int(analysis.was_correct), analysis.confidence_error,
                int(analysis.was_confident_and_wrong),
                analysis.best_model, analysis.worst_model,
                json.dumps(analysis.model_errors),
                analysis.evidence_quality,
                json.dumps(analysis.evidence_sources),
                analysis.position_size_appropriate,
                analysis.pnl, analysis.edge_at_entry,
                analysis.holding_hours, analysis.analyzed_at,
            ))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("post_mortem.save_error", error=str(e))

    def _get_model_errors(self, market_id: str) -> dict[str, float]:
        """Get per-model absolute errors for a market."""
        try:
            rows = self._conn.execute("""
                SELECT model_name, forecast_prob, actual_outcome
                FROM model_forecast_log
                WHERE market_id = ?
            """, (market_id,)).fetchall()
        except sqlite3.OperationalError:
            return {}

        errors: dict[str, float] = {}
        for r in rows:
            prob = float(r["forecast_prob"] or 0)
            outcome = float(r["actual_outcome"] if r["actual_outcome"] is not None else 0)
            errors[r["model_name"]] = round(abs(prob - outcome), 4)
        return errors

    @staticmethod
    def _compute_position_appropriateness(
        forecast_prob: float,
        actual_outcome: float,
        stake_usd: float,
        bankroll: float,
        kelly_fraction: float,
    ) -> str:
        """Compare actual stake vs optimal Kelly stake."""
        # Compute optimal Kelly edge
        edge = abs(forecast_prob - 0.5) * 2  # simplified edge
        if edge <= 0:
            return "appropriate"

        # Kelly optimal stake
        p = forecast_prob if actual_outcome == 1.0 else (1.0 - forecast_prob)
        q = 1.0 - p
        if q <= 0:
            return "appropriate"
        kelly_optimal = max(0, (p - q) / 1.0) * bankroll * kelly_fraction

        if kelly_optimal <= 0:
            if stake_usd > 0:
                return "too_large"
            return "appropriate"

        ratio = stake_usd / kelly_optimal
        if ratio < 0.5:
            return "too_small"
        elif ratio > 2.0:
            return "too_large"
        return "appropriate"
