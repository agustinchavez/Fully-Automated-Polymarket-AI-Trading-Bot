"""Whale quality scoring — multi-dimensional ranking of tracked wallets.

Scores each whale on five dimensions (each 0–100):
  1. Historical ROI (trailing 90 days)
  2. Calibration quality (entry prices before favorable moves, weighted by magnitude)
  3. Category specialization (concentrated profitability in best domain)
  4. Consistency (Sharpe ratio of individual trades)
  5. Timing score (% entries preceding favorable 24h price moves)

Only whales above the configured percentile (default 60th) contribute to
conviction signals. Disabled by default — gated behind
wallet_scanner.whale_quality_scoring_enabled.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from src.observability.logger import get_logger

log = get_logger(__name__)

# Category keywords for specialization scoring
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Politics": ["president", "election", "vote", "congress", "senate", "governor", "democrat", "republican", "biden", "trump"],
    "Crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "token", "defi", "nft", "solana", "sol"],
    "Sports": ["nba", "nfl", "mlb", "soccer", "football", "basketball", "tennis", "golf", "championship", "super bowl"],
    "Finance": ["stock", "interest rate", "fed", "inflation", "gdp", "recession", "s&p", "nasdaq", "dow"],
    "Entertainment": ["oscar", "grammy", "movie", "tv", "show", "emmy", "netflix", "disney"],
    "Science": ["climate", "temperature", "nasa", "space", "ai", "artificial intelligence"],
    "Geopolitics": ["war", "conflict", "nato", "sanctions", "china", "russia", "ukraine", "israel"],
}


@dataclass
class WhaleQualityScore:
    """Multi-dimensional quality score for a tracked whale wallet."""

    address: str
    name: str = ""
    # Dimension scores (each 0-100)
    historical_roi: float = 0.0
    calibration_quality: float = 0.0
    category_specialization: float = 0.0
    consistency: float = 0.0
    timing_score: float = 0.0
    # Composite
    composite_score: float = 0.0
    percentile: float = 0.0
    # Metadata
    best_category: str = ""
    trade_count_90d: int = 0
    sharpe_ratio: float = 0.0
    scored_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "historical_roi": round(self.historical_roi, 1),
            "calibration_quality": round(self.calibration_quality, 1),
            "category_specialization": round(self.category_specialization, 1),
            "consistency": round(self.consistency, 1),
            "timing_score": round(self.timing_score, 1),
            "composite_score": round(self.composite_score, 1),
            "percentile": round(self.percentile, 1),
            "best_category": self.best_category,
            "trade_count_90d": self.trade_count_90d,
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "scored_at": self.scored_at,
        }


def _compute_composite(score: WhaleQualityScore) -> float:
    """Weighted blend of dimension scores."""
    return (
        0.25 * score.historical_roi
        + 0.25 * score.calibration_quality
        + 0.15 * score.category_specialization
        + 0.20 * score.consistency
        + 0.15 * score.timing_score
    )


def _categorize_market(title: str) -> str:
    """Classify a market title into a category."""
    lower = title.lower() if title else ""
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return "Other"


class WhaleScorer:
    """Scores tracked whale wallets on quality dimensions."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lookback_days: int = 90,
        timing_favorable_threshold: float = 0.02,
    ):
        self._conn = conn
        self._lookback_days = lookback_days
        self._timing_threshold = timing_favorable_threshold

    def score_all(
        self,
        wallets: list[Any],
    ) -> list[WhaleQualityScore]:
        """Score all tracked wallets and assign percentiles.

        Args:
            wallets: list of TrackedWallet dataclass instances.

        Returns:
            Scores sorted by composite_score descending, with percentiles.
        """
        scores = [self.score_wallet(w) for w in wallets]

        # Assign percentiles
        if len(scores) > 1:
            scores.sort(key=lambda s: s.composite_score)
            for i, s in enumerate(scores):
                s.percentile = round(i / (len(scores) - 1) * 100, 1)
        elif scores:
            scores[0].percentile = 100.0

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def score_wallet(self, wallet: Any) -> WhaleQualityScore:
        """Score a single wallet on all dimensions."""
        address = wallet.address
        name = getattr(wallet, "name", "")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        roi = self._compute_historical_roi(address)
        cal, timing = self._compute_calibration(address)
        spec, best_cat = self._compute_category_specialization(address)
        consist, sharpe = self._compute_consistency(address)

        # Count trades in lookback
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=self._lookback_days)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM wallet_deltas WHERE wallet_address = ? AND detected_at >= ?",
                (address, cutoff),
            ).fetchone()
            trade_count = row[0] if row else 0
        except Exception:
            trade_count = 0

        score = WhaleQualityScore(
            address=address,
            name=name,
            historical_roi=roi,
            calibration_quality=cal,
            category_specialization=spec,
            consistency=consist,
            timing_score=timing,
            best_category=best_cat,
            trade_count_90d=trade_count,
            sharpe_ratio=sharpe,
            scored_at=now,
        )
        score.composite_score = round(_compute_composite(score), 1)
        return score

    # ── Dimension Computations ─────────────────────────────────────

    def _compute_historical_roi(self, address: str) -> float:
        """Trailing ROI from wallet_deltas value changes.

        Normalizes: -50% ROI → 0, +100% ROI → 100, linear interpolation.
        """
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=self._lookback_days)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        try:
            rows = self._conn.execute(
                """SELECT action, value_change_usd
                   FROM wallet_deltas
                   WHERE wallet_address = ? AND detected_at >= ?""",
                (address, cutoff),
            ).fetchall()
        except Exception:
            return 0.0

        if not rows:
            return 0.0

        total_pnl = sum(r[1] for r in rows if r[1] is not None)
        # Use invested as sum of absolute values of new entries
        invested = sum(
            abs(r[1]) for r in rows
            if r[0] == "NEW_ENTRY" and r[1] is not None
        )
        if invested <= 0:
            invested = max(abs(total_pnl), 1)

        roi = total_pnl / invested  # -0.5 to +1.0 typical range

        # Normalize: -0.5 → 0, 0 → 33.3, +1.0 → 100
        normalized = ((roi + 0.5) / 1.5) * 100
        return max(0.0, min(100.0, round(normalized, 1)))

    def _compute_calibration(self, address: str) -> tuple[float, float]:
        """Compute calibration quality and timing score from price snapshots.

        Returns:
            (calibration_quality, timing_score) — both 0-100.
            Calibration weights by move magnitude; timing is binary.
        """
        try:
            rows = self._conn.execute(
                """SELECT entry_price, price_after_24h, direction, favorable_move
                   FROM whale_price_snapshots
                   WHERE wallet_address = ? AND price_24h_recorded = 1""",
                (address,),
            ).fetchall()
        except Exception:
            return (0.0, 0.0)

        if not rows:
            return (0.0, 0.0)

        favorable_count = 0
        magnitude_sum = 0.0

        for entry_price, price_24h, direction, favorable in rows:
            if entry_price <= 0:
                continue

            if favorable:
                favorable_count += 1

            # Magnitude of move (positive = favorable direction)
            if direction == "BULLISH":
                move = (price_24h - entry_price) / entry_price
            else:
                move = (entry_price - price_24h) / entry_price

            magnitude_sum += move

        total = len(rows)
        timing_score = (favorable_count / total * 100) if total > 0 else 0.0

        # Calibration: average magnitude normalized (0% → 50, ±10% → 0 or 100)
        avg_magnitude = magnitude_sum / total if total > 0 else 0.0
        calibration = 50.0 + avg_magnitude * 500  # scale: +10% → 100, -10% → 0
        calibration = max(0.0, min(100.0, calibration))

        return (round(calibration, 1), round(timing_score, 1))

    def _compute_category_specialization(
        self, address: str,
    ) -> tuple[float, str]:
        """Specialization score based on category concentration.

        Returns:
            (specialization_score, best_category) — score 0-100.
        """
        try:
            rows = self._conn.execute(
                """SELECT title, value_change_usd
                   FROM wallet_deltas
                   WHERE wallet_address = ?""",
                (address,),
            ).fetchall()
        except Exception:
            return (0.0, "")

        if not rows:
            return (0.0, "")

        # Group PnL by category
        cat_pnl: dict[str, float] = {}
        cat_count: dict[str, int] = {}
        for title, pnl in rows:
            cat = _categorize_market(title or "")
            cat_pnl[cat] = cat_pnl.get(cat, 0) + (pnl or 0)
            cat_count[cat] = cat_count.get(cat, 0) + 1

        if not cat_pnl:
            return (0.0, "")

        # Best category by PnL
        best_cat = max(cat_pnl, key=lambda c: cat_pnl[c])
        best_pnl = cat_pnl[best_cat]

        # Concentration: Herfindahl index of trade counts
        total_trades = sum(cat_count.values())
        if total_trades == 0:
            return (0.0, best_cat)

        hhi = sum((n / total_trades) ** 2 for n in cat_count.values())
        # hhi ranges from 1/N to 1.0 (perfect concentration)
        # Normalize: random (1/N) → 0, concentrated (1.0) → 100
        n_cats = len(cat_count)
        if n_cats <= 1:
            spec_score = 100.0 if best_pnl > 0 else 0.0
        else:
            min_hhi = 1.0 / n_cats
            spec_score = ((hhi - min_hhi) / (1.0 - min_hhi)) * 100

        # Bonus: only reward concentration if the best category is profitable
        if best_pnl <= 0:
            spec_score *= 0.3  # heavily discount unprofitable specialization

        return (round(max(0.0, min(100.0, spec_score)), 1), best_cat)

    def _compute_consistency(self, address: str) -> tuple[float, float]:
        """Consistency score based on Sharpe ratio of individual trades.

        Returns:
            (consistency_score, raw_sharpe) — score 0-100.
        """
        try:
            rows = self._conn.execute(
                """SELECT value_change_usd
                   FROM wallet_deltas
                   WHERE wallet_address = ? AND action IN ('EXIT', 'SIZE_DECREASE')""",
                (address,),
            ).fetchall()
        except Exception:
            return (0.0, 0.0)

        pnls = [r[0] for r in rows if r[0] is not None]
        if len(pnls) < 2:
            return (0.0, 0.0)

        mean = sum(pnls) / len(pnls)
        variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
        std = math.sqrt(variance) if variance > 0 else 0.001

        sharpe = mean / std  # risk-adjusted return

        # Normalize: Sharpe < 0 → 0, Sharpe >= 3 → 100
        normalized = max(0.0, min(100.0, sharpe / 3.0 * 100))
        return (round(normalized, 1), round(sharpe, 3))

    # ── Filtering ──────────────────────────────────────────────────

    def get_top_percentile(
        self,
        scores: list[WhaleQualityScore],
        threshold: float = 60.0,
    ) -> list[WhaleQualityScore]:
        """Return whales above the given percentile threshold."""
        return [s for s in scores if s.percentile >= threshold]

    def is_qualified(
        self,
        score: WhaleQualityScore,
        threshold_pct: float = 60.0,
    ) -> bool:
        """Check if a whale is above the quality threshold."""
        return score.percentile >= threshold_pct

    # ── Database Persistence ───────────────────────────────────────

    def save_scores(
        self,
        conn: sqlite3.Connection,
        scores: list[WhaleQualityScore],
    ) -> None:
        """Upsert quality scores into whale_quality_scores table."""
        for s in scores:
            conn.execute(
                """INSERT OR REPLACE INTO whale_quality_scores
                   (address, name, historical_roi, calibration_quality,
                    category_specialization, consistency, timing_score,
                    composite_score, percentile, best_category,
                    trade_count_90d, sharpe_ratio, scored_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (s.address, s.name, s.historical_roi, s.calibration_quality,
                 s.category_specialization, s.consistency, s.timing_score,
                 s.composite_score, s.percentile, s.best_category,
                 s.trade_count_90d, s.sharpe_ratio, s.scored_at),
            )
        conn.commit()
        log.info("whale_scorer.saved", count=len(scores))

    def record_entry_snapshot(
        self,
        conn: sqlite3.Connection,
        wallet_address: str,
        market_slug: str,
        outcome: str,
        entry_price: float,
        direction: str,
    ) -> None:
        """Record a price snapshot when a whale enters a position."""
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price,
                entry_time, direction, price_24h_recorded)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (wallet_address, market_slug, outcome, entry_price, now, direction),
        )

    def update_pending_snapshots(
        self,
        conn: sqlite3.Connection,
        price_fetcher: Callable[[str], float],
    ) -> int:
        """Update snapshots where 24h has elapsed.

        Args:
            price_fetcher: callable(market_slug) -> current_price.

        Returns:
            Number of snapshots updated.
        """
        cutoff = (
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        try:
            rows = conn.execute(
                """SELECT id, market_slug, entry_price, direction
                   FROM whale_price_snapshots
                   WHERE price_24h_recorded = 0 AND entry_time <= ?""",
                (cutoff,),
            ).fetchall()
        except Exception:
            return 0

        updated = 0
        for row_id, slug, entry_price, direction in rows:
            try:
                current_price = price_fetcher(slug)
            except Exception:
                continue

            if entry_price <= 0:
                continue

            # Determine if move was favorable
            if direction == "BULLISH":
                move_pct = (current_price - entry_price) / entry_price
            else:
                move_pct = (entry_price - current_price) / entry_price

            favorable = 1 if move_pct >= self._timing_threshold else 0

            conn.execute(
                """UPDATE whale_price_snapshots
                   SET price_after_24h = ?, price_24h_recorded = 1, favorable_move = ?
                   WHERE id = ?""",
                (current_price, favorable, row_id),
            )
            updated += 1

        if updated:
            conn.commit()
        log.info("whale_scorer.snapshots_updated", count=updated)
        return updated

    # ── Timing Report ──────────────────────────────────────────────

    def compute_wallet_timing_report(self, address: str) -> dict[str, Any]:
        """Detailed timing report for a single wallet."""
        try:
            rows = self._conn.execute(
                """SELECT entry_price, price_after_24h, direction, favorable_move
                   FROM whale_price_snapshots
                   WHERE wallet_address = ? AND price_24h_recorded = 1""",
                (address,),
            ).fetchall()
        except Exception:
            return {
                "total_entries": 0,
                "favorable_entries": 0,
                "timing_score": 0.0,
                "avg_favorable_move_pct": 0.0,
                "avg_unfavorable_move_pct": 0.0,
            }

        if not rows:
            return {
                "total_entries": 0,
                "favorable_entries": 0,
                "timing_score": 0.0,
                "avg_favorable_move_pct": 0.0,
                "avg_unfavorable_move_pct": 0.0,
            }

        favorable_moves = []
        unfavorable_moves = []
        favorable_count = 0

        for entry_price, price_24h, direction, favorable in rows:
            if entry_price <= 0:
                continue
            if direction == "BULLISH":
                move = (price_24h - entry_price) / entry_price
            else:
                move = (entry_price - price_24h) / entry_price

            if favorable:
                favorable_count += 1
                favorable_moves.append(move)
            else:
                unfavorable_moves.append(move)

        total = len(rows)
        return {
            "total_entries": total,
            "favorable_entries": favorable_count,
            "timing_score": round(favorable_count / total * 100, 1) if total > 0 else 0.0,
            "avg_favorable_move_pct": round(
                sum(favorable_moves) / len(favorable_moves) * 100, 2
            ) if favorable_moves else 0.0,
            "avg_unfavorable_move_pct": round(
                sum(unfavorable_moves) / len(unfavorable_moves) * 100, 2
            ) if unfavorable_moves else 0.0,
        }
