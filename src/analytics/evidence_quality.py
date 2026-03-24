"""Evidence source quality tracking — learn which domains are reliable.

Tracks which search domains correlate with correct forecasts over time.
After enough resolved markets, ranks domains and auto-adjusts search
weighting to prefer reliable sources.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class SourceQualityRecord:
    """Quality metrics for a single evidence source domain."""
    domain: str
    times_cited: int = 0
    times_correct: int = 0
    correct_forecast_rate: float = 0.0
    avg_evidence_quality: float = 0.0
    avg_authority: float = 0.0
    quality_trend: str = "stable"
    effective_weight: float = 1.0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "times_cited": self.times_cited,
            "times_correct": self.times_correct,
            "correct_forecast_rate": self.correct_forecast_rate,
            "avg_evidence_quality": self.avg_evidence_quality,
            "avg_authority": self.avg_authority,
            "quality_trend": self.quality_trend,
            "effective_weight": self.effective_weight,
            "last_updated": self.last_updated,
        }


class EvidenceQualityTracker:
    """Tracks evidence source quality across resolved markets."""

    def __init__(self, conn: sqlite3.Connection, min_citations: int = 5):
        self._conn = conn
        self._min_citations = min_citations

    def record_source_outcome(
        self,
        domain: str,
        was_correct: bool,
        evidence_quality: float = 0.0,
        authority: float = 0.0,
    ) -> None:
        """Record an outcome for a domain citation."""
        try:
            existing = self._conn.execute(
                "SELECT * FROM evidence_source_quality WHERE domain = ?",
                (domain,),
            ).fetchone()

            if existing:
                new_cited = int(existing["times_cited"]) + 1
                new_correct = int(existing["times_correct"]) + (1 if was_correct else 0)
                new_rate = new_correct / new_cited if new_cited > 0 else 0.0
                # Running average for quality and authority
                old_quality = float(existing["avg_evidence_quality"] or 0)
                old_authority = float(existing["avg_authority"] or 0)
                old_cited = int(existing["times_cited"])
                new_avg_quality = (old_quality * old_cited + evidence_quality) / new_cited
                new_avg_authority = (old_authority * old_cited + authority) / new_cited

                self._conn.execute("""
                    UPDATE evidence_source_quality
                    SET times_cited = ?,
                        times_correct = ?,
                        correct_forecast_rate = ?,
                        avg_evidence_quality = ?,
                        avg_authority = ?,
                        effective_weight = ?,
                        last_updated = ?
                    WHERE domain = ?
                """, (
                    new_cited, new_correct, round(new_rate, 4),
                    round(new_avg_quality, 4), round(new_avg_authority, 4),
                    round(self._compute_weight(new_rate, new_avg_authority, new_cited), 4),
                    _now_iso(), domain,
                ))
            else:
                rate = 1.0 if was_correct else 0.0
                self._conn.execute("""
                    INSERT INTO evidence_source_quality
                        (domain, times_cited, times_correct,
                         correct_forecast_rate, avg_evidence_quality,
                         avg_authority, effective_weight, last_updated)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                """, (
                    domain, 1 if was_correct else 0,
                    rate, round(evidence_quality, 4),
                    round(authority, 4),
                    round(self._compute_weight(rate, authority, 1), 4),
                    _now_iso(),
                ))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("evidence_quality.record_error", error=str(e))

    def get_domain_rankings(
        self, min_citations: int | None = None
    ) -> list[SourceQualityRecord]:
        """Get all domains sorted by effective weight (descending)."""
        threshold = min_citations if min_citations is not None else self._min_citations
        try:
            rows = self._conn.execute("""
                SELECT * FROM evidence_source_quality
                WHERE times_cited >= ?
                ORDER BY effective_weight DESC
            """, (threshold,)).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            SourceQualityRecord(
                domain=r["domain"],
                times_cited=int(r["times_cited"]),
                times_correct=int(r["times_correct"]),
                correct_forecast_rate=float(r["correct_forecast_rate"] or 0),
                avg_evidence_quality=float(r["avg_evidence_quality"] or 0),
                avg_authority=float(r["avg_authority"] or 0),
                quality_trend=r["quality_trend"] or "stable",
                effective_weight=float(r["effective_weight"] or 1.0),
                last_updated=r["last_updated"] or "",
            )
            for r in rows
        ]

    def get_effective_weight(self, domain: str) -> float:
        """Get the effective weight multiplier for a domain (0.5–1.5)."""
        try:
            row = self._conn.execute(
                "SELECT effective_weight FROM evidence_source_quality WHERE domain = ?",
                (domain,),
            ).fetchone()
        except sqlite3.OperationalError:
            return 1.0

        if not row:
            return 1.0
        return float(row["effective_weight"] or 1.0)

    def get_top_sources(self, n: int = 10) -> list[str]:
        """Get the top N highest-quality domains."""
        rankings = self.get_domain_rankings()
        return [r.domain for r in rankings[:n]]

    def get_blocklist(self, threshold: float = 0.3) -> list[str]:
        """Get domains below the quality threshold."""
        try:
            rows = self._conn.execute("""
                SELECT domain FROM evidence_source_quality
                WHERE correct_forecast_rate < ?
                  AND times_cited >= ?
            """, (threshold, self._min_citations)).fetchall()
        except sqlite3.OperationalError:
            return []

        return [r["domain"] for r in rows]

    @staticmethod
    def _compute_weight(
        correct_rate: float,
        authority: float,
        times_cited: int,
    ) -> float:
        """Compute effective weight from quality signals.

        Returns a value in [0.5, 1.5]:
          - Base weight = 1.0
          - correct_rate > 0.6 → boost up to +0.3
          - correct_rate < 0.4 → penalty up to -0.3
          - authority > 0.7 → boost +0.1
          - Low citation count → dampen toward 1.0
        """
        weight = 1.0

        # Accuracy component (±0.3)
        if correct_rate > 0.6:
            weight += min(0.3, (correct_rate - 0.6) * 0.75)
        elif correct_rate < 0.4:
            weight -= min(0.3, (0.4 - correct_rate) * 0.75)

        # Authority component (+0.1)
        if authority > 0.7:
            weight += 0.1

        # Confidence dampening: with few citations, pull toward 1.0
        if times_cited < 10:
            confidence = times_cited / 10.0
            weight = 1.0 + (weight - 1.0) * confidence

        return max(0.5, min(1.5, weight))
