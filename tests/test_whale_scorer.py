"""Tests for Phase 7 — Enhanced Whale Intelligence.

Covers:
  - WhaleQualityScore dataclass (defaults, to_dict, composite computation)
  - WhaleScorer scoring dimensions (ROI, calibration, specialization, consistency, timing)
  - Percentile assignment and quality filtering
  - Database persistence (save_scores, record_entry_snapshot, update_pending_snapshots)
  - Migration 13 schema
  - Conviction quality filtering (qualified_addresses, quality_weights)
  - Config Phase 7 fields
  - Timing report
  - Dashboard endpoint
  - Enhanced engine thresholds
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from src.analytics.whale_scorer import (
    WhaleQualityScore,
    WhaleScorer,
    _categorize_market,
    _compute_composite,
)
from src.analytics.wallet_scanner import (
    ConvictionSignal,
    ScanResult,
    TrackedWallet,
    WalletDelta,
    WalletScanner,
    save_scan_result,
)
from src.config import BotConfig, WalletScannerConfig


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _create_test_db() -> sqlite3.Connection:
    """Create in-memory SQLite DB with Phase 7 schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracked_wallets (
            address TEXT PRIMARY KEY,
            name TEXT,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            active_positions INTEGER DEFAULT 0,
            total_volume REAL DEFAULT 0,
            score REAL DEFAULT 0,
            last_scanned TEXT
        );
        CREATE TABLE IF NOT EXISTS wallet_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_slug TEXT NOT NULL,
            title TEXT,
            condition_id TEXT,
            outcome TEXT,
            whale_count INTEGER DEFAULT 0,
            total_whale_usd REAL DEFAULT 0,
            avg_whale_price REAL DEFAULT 0,
            current_price REAL DEFAULT 0,
            conviction_score REAL DEFAULT 0,
            whale_names_json TEXT DEFAULT '[]',
            direction TEXT,
            signal_strength TEXT,
            detected_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_wallet_signals_unique
            ON wallet_signals(market_slug, outcome);
        CREATE TABLE IF NOT EXISTS wallet_deltas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_address TEXT NOT NULL,
            wallet_name TEXT,
            action TEXT NOT NULL,
            market_slug TEXT,
            title TEXT,
            outcome TEXT,
            size_change REAL DEFAULT 0,
            value_change_usd REAL DEFAULT 0,
            current_price REAL DEFAULT 0,
            detected_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_wallet_deltas_unique
            ON wallet_deltas(wallet_address, market_slug, outcome, action);
        CREATE TABLE IF NOT EXISTS whale_quality_scores (
            address TEXT PRIMARY KEY,
            name TEXT DEFAULT '',
            historical_roi REAL DEFAULT 0,
            calibration_quality REAL DEFAULT 0,
            category_specialization REAL DEFAULT 0,
            consistency REAL DEFAULT 0,
            timing_score REAL DEFAULT 0,
            composite_score REAL DEFAULT 0,
            percentile REAL DEFAULT 0,
            best_category TEXT DEFAULT '',
            trade_count_90d INTEGER DEFAULT 0,
            sharpe_ratio REAL DEFAULT 0,
            scored_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_wqs_composite
            ON whale_quality_scores(composite_score);
        CREATE TABLE IF NOT EXISTS whale_price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet_address TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            outcome TEXT DEFAULT '',
            entry_price REAL DEFAULT 0,
            entry_time TEXT NOT NULL,
            price_after_24h REAL DEFAULT 0,
            price_24h_recorded INTEGER DEFAULT 0,
            direction TEXT DEFAULT '',
            favorable_move INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_wps_wallet
            ON whale_price_snapshots(wallet_address);
        CREATE INDEX IF NOT EXISTS idx_wps_pending
            ON whale_price_snapshots(price_24h_recorded);
    """)
    return conn


def _make_wallet(address: str = "0xAAA", name: str = "TestWhale") -> TrackedWallet:
    return TrackedWallet(address=address, name=name, total_pnl=100_000)


def _insert_deltas(
    conn: sqlite3.Connection,
    address: str,
    entries: list[tuple[str, str, float]],
) -> None:
    """Insert wallet_deltas rows: [(action, title, value_change_usd), ...]."""
    now = dt.datetime.utcnow().isoformat() + "Z"
    for action, title, value in entries:
        conn.execute(
            """INSERT INTO wallet_deltas
               (wallet_address, wallet_name, action, market_slug, title,
                outcome, size_change, value_change_usd, current_price, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (address, "test", action, "slug-" + title.lower().replace(" ", "-"),
             title, "Yes", 10.0, value, 0.5, now),
        )
    conn.commit()


def _insert_snapshots(
    conn: sqlite3.Connection,
    address: str,
    snapshots: list[tuple[float, float, str, int]],
) -> None:
    """Insert whale_price_snapshots: [(entry_price, price_24h, direction, favorable), ...]."""
    past = (dt.datetime.utcnow() - dt.timedelta(hours=48)).isoformat() + "Z"
    for entry_price, price_24h, direction, favorable in snapshots:
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price, entry_time,
                price_after_24h, price_24h_recorded, direction, favorable_move)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (address, "test-market", "Yes", entry_price, past, price_24h, direction, favorable),
        )
    conn.commit()


# ═══════════════════════════════════════════════════════════════════
#  WhaleQualityScore DATACLASS
# ═══════════════════════════════════════════════════════════════════

class TestWhaleQualityScore:
    """Test WhaleQualityScore dataclass."""

    def test_defaults(self):
        s = WhaleQualityScore(address="0x1")
        assert s.historical_roi == 0.0
        assert s.composite_score == 0.0
        assert s.percentile == 0.0
        assert s.best_category == ""

    def test_to_dict_keys(self):
        s = WhaleQualityScore(address="0x1", name="Whale")
        d = s.to_dict()
        expected_keys = {
            "address", "name", "historical_roi", "calibration_quality",
            "category_specialization", "consistency", "timing_score",
            "composite_score", "percentile", "best_category",
            "trade_count_90d", "sharpe_ratio", "scored_at",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounding(self):
        s = WhaleQualityScore(
            address="0x1",
            historical_roi=72.456789,
            sharpe_ratio=1.23456789,
        )
        d = s.to_dict()
        assert d["historical_roi"] == 72.5
        assert d["sharpe_ratio"] == 1.235

    def test_composite_computation(self):
        s = WhaleQualityScore(
            address="0x1",
            historical_roi=80.0,
            calibration_quality=60.0,
            category_specialization=40.0,
            consistency=70.0,
            timing_score=50.0,
        )
        expected = 0.25 * 80 + 0.25 * 60 + 0.15 * 40 + 0.20 * 70 + 0.15 * 50
        assert abs(_compute_composite(s) - expected) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

class TestCategorizeMarket:
    """Test market title categorization."""

    def test_politics(self):
        assert _categorize_market("Will Biden win the election?") == "Politics"

    def test_crypto(self):
        assert _categorize_market("Bitcoin above $100k by year end") == "Crypto"

    def test_sports(self):
        assert _categorize_market("NBA finals winner") == "Sports"

    def test_unknown(self):
        assert _categorize_market("Will the new restaurant open by Friday") == "Other"

    def test_empty(self):
        assert _categorize_market("") == "Other"

    def test_case_insensitive(self):
        assert _categorize_market("TRUMP ELECTION") == "Politics"


# ═══════════════════════════════════════════════════════════════════
#  HISTORICAL ROI SCORING
# ═══════════════════════════════════════════════════════════════════

class TestHistoricalROI:
    """Test _compute_historical_roi dimension."""

    def test_high_roi(self):
        conn = _create_test_db()
        addr = "0xROI"
        # Insert profitable entries + exits
        _insert_deltas(conn, addr, [
            ("NEW_ENTRY", "Market A", -500),  # invested $500
            ("EXIT", "Market A", 800),        # exited with $800 profit
        ])
        scorer = WhaleScorer(conn)
        roi = scorer._compute_historical_roi(addr)
        # total_pnl = -500 + 800 = 300, invested = 500
        # roi = 300/500 = 0.6, normalized = (0.6+0.5)/1.5*100 = 73.3
        assert roi > 60

    def test_negative_roi(self):
        conn = _create_test_db()
        addr = "0xLOSER"
        _insert_deltas(conn, addr, [
            ("NEW_ENTRY", "Market B", -1000),
            ("EXIT", "Market B", -200),
        ])
        scorer = WhaleScorer(conn)
        roi = scorer._compute_historical_roi(addr)
        # total_pnl = -1200, invested = 1000
        # roi = -1.2, normalized = (-1.2+0.5)/1.5*100 = -46.7 → clamped to 0
        assert roi == 0.0

    def test_no_data(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        roi = scorer._compute_historical_roi("0xNONE")
        assert roi == 0.0

    def test_zero_invested(self):
        conn = _create_test_db()
        addr = "0xZERO"
        # Only exits, no NEW_ENTRY
        _insert_deltas(conn, addr, [
            ("EXIT", "Market C", 500),
        ])
        scorer = WhaleScorer(conn)
        roi = scorer._compute_historical_roi(addr)
        # invested = max(abs(500), 1) = 500, roi = 500/500 = 1.0
        assert roi == 100.0


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION + TIMING SCORING
# ═══════════════════════════════════════════════════════════════════

class TestCalibration:
    """Test _compute_calibration dimension."""

    def test_all_favorable(self):
        conn = _create_test_db()
        addr = "0xCAL"
        # All bullish entries where price went up
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),
            (0.40, 0.55, "BULLISH", 1),
            (0.45, 0.58, "BULLISH", 1),
        ])
        scorer = WhaleScorer(conn)
        cal, timing = scorer._compute_calibration(addr)
        assert timing == 100.0  # all favorable
        assert cal > 50.0  # positive magnitude → above midpoint

    def test_no_favorable(self):
        conn = _create_test_db()
        addr = "0xBAD"
        _insert_snapshots(conn, addr, [
            (0.50, 0.40, "BULLISH", 0),
            (0.60, 0.45, "BULLISH", 0),
        ])
        scorer = WhaleScorer(conn)
        cal, timing = scorer._compute_calibration(addr)
        assert timing == 0.0
        assert cal < 50.0  # negative magnitude

    def test_mixed(self):
        conn = _create_test_db()
        addr = "0xMIX"
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),
            (0.50, 0.40, "BULLISH", 0),
        ])
        scorer = WhaleScorer(conn)
        cal, timing = scorer._compute_calibration(addr)
        assert timing == 50.0  # 1 out of 2

    def test_no_snapshots(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        cal, timing = scorer._compute_calibration("0xNONE")
        assert cal == 0.0
        assert timing == 0.0

    def test_bearish_direction(self):
        conn = _create_test_db()
        addr = "0xBEAR"
        # Bearish: favorable if price goes down
        _insert_snapshots(conn, addr, [
            (0.60, 0.40, "BEARISH", 1),  # price dropped → favorable for bearish
        ])
        scorer = WhaleScorer(conn)
        cal, timing = scorer._compute_calibration(addr)
        assert timing == 100.0
        assert cal > 50.0


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY SPECIALIZATION SCORING
# ═══════════════════════════════════════════════════════════════════

class TestCategorySpecialization:
    """Test _compute_category_specialization dimension."""

    def test_focused_profitable(self):
        conn = _create_test_db()
        addr = "0xFOC"
        _insert_deltas(conn, addr, [
            ("NEW_ENTRY", "Trump election bet", -100),
            ("EXIT", "Biden election outcome", 200),
            ("EXIT", "Congress vote prediction", 150),
        ])
        scorer = WhaleScorer(conn)
        spec, cat = scorer._compute_category_specialization(addr)
        assert cat == "Politics"  # all politics
        assert spec > 0  # concentrated + profitable

    def test_diverse(self):
        conn = _create_test_db()
        addr = "0xDIV"
        _insert_deltas(conn, addr, [
            ("EXIT", "Trump election", 100),
            ("EXIT", "Bitcoin price bet", 100),
            ("EXIT", "NBA championship", 100),
            ("EXIT", "Stock market crash", 100),
        ])
        scorer = WhaleScorer(conn)
        spec, cat = scorer._compute_category_specialization(addr)
        # Spread across 4 categories → low specialization
        assert spec < 50

    def test_no_positions(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        spec, cat = scorer._compute_category_specialization("0xNONE")
        assert spec == 0.0
        assert cat == ""

    def test_single_category_unprofitable(self):
        conn = _create_test_db()
        addr = "0xLOSE"
        _insert_deltas(conn, addr, [
            ("EXIT", "Bitcoin crash bet", -200),
            ("EXIT", "Ethereum price bet", -100),
        ])
        scorer = WhaleScorer(conn)
        spec, cat = scorer._compute_category_specialization(addr)
        assert cat == "Crypto"
        # Unprofitable specialization gets discounted by 0.3
        assert spec < 50


# ═══════════════════════════════════════════════════════════════════
#  CONSISTENCY SCORING (SHARPE)
# ═══════════════════════════════════════════════════════════════════

class TestConsistency:
    """Test _compute_consistency dimension."""

    def test_high_sharpe(self):
        conn = _create_test_db()
        addr = "0xSHARP"
        # Consistent profits with low variance
        _insert_deltas(conn, addr, [
            ("EXIT", "M1", 100),
            ("EXIT", "M2", 110),
            ("EXIT", "M3", 105),
            ("EXIT", "M4", 95),
            ("EXIT", "M5", 108),
        ])
        scorer = WhaleScorer(conn)
        consist, sharpe = scorer._compute_consistency(addr)
        assert sharpe > 0
        assert consist > 50  # positive Sharpe → decent score

    def test_negative_sharpe(self):
        conn = _create_test_db()
        addr = "0xBAD"
        _insert_deltas(conn, addr, [
            ("EXIT", "M1", -100),
            ("EXIT", "M2", -50),
            ("EXIT", "M3", -80),
        ])
        scorer = WhaleScorer(conn)
        consist, sharpe = scorer._compute_consistency(addr)
        assert sharpe < 0
        assert consist == 0.0  # clamped to 0

    def test_zero_trades(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        consist, sharpe = scorer._compute_consistency("0xNONE")
        assert consist == 0.0
        assert sharpe == 0.0

    def test_single_trade(self):
        conn = _create_test_db()
        addr = "0xONE"
        _insert_deltas(conn, addr, [("EXIT", "M1", 100)])
        scorer = WhaleScorer(conn)
        consist, sharpe = scorer._compute_consistency(addr)
        # Need at least 2 trades for Sharpe
        assert consist == 0.0

    def test_only_new_entries_ignored(self):
        """Consistency only counts EXIT and SIZE_DECREASE, not NEW_ENTRY."""
        conn = _create_test_db()
        addr = "0xENTRY"
        _insert_deltas(conn, addr, [
            ("NEW_ENTRY", "M1", -500),
            ("NEW_ENTRY", "M2", -300),
        ])
        scorer = WhaleScorer(conn)
        consist, sharpe = scorer._compute_consistency(addr)
        assert consist == 0.0


# ═══════════════════════════════════════════════════════════════════
#  SCORE_ALL + PERCENTILES
# ═══════════════════════════════════════════════════════════════════

class TestScoreAll:
    """Test score_all() percentile assignment and sorting."""

    def test_ranking_order(self):
        conn = _create_test_db()
        addr1, addr2 = "0xA1", "0xA2"
        # Make addr1 much more profitable
        _insert_deltas(conn, addr1, [
            ("NEW_ENTRY", "M1", -100),
            ("EXIT", "M1", 500),
        ])
        _insert_deltas(conn, addr2, [
            ("NEW_ENTRY", "M2", -100),
            ("EXIT", "M2", -50),
        ])
        scorer = WhaleScorer(conn)
        scores = scorer.score_all([
            _make_wallet(addr1, "W1"),
            _make_wallet(addr2, "W2"),
        ])
        # Sorted by composite descending
        assert scores[0].address == addr1
        assert scores[1].address == addr2

    def test_percentile_assignment(self):
        conn = _create_test_db()
        wallets = []
        for i in range(5):
            addr = f"0x{i:04x}"
            # Make wallets with varying profitability
            _insert_deltas(conn, addr, [
                ("NEW_ENTRY", f"Market {i}", -(i + 1) * 100),
                ("EXIT", f"Market {i}", (i + 1) * 300),
                ("SIZE_DECREASE", f"Market {i}", (i + 1) * 50),
            ])
            wallets.append(_make_wallet(addr, f"W{i}"))
        scorer = WhaleScorer(conn)
        scores = scorer.score_all(wallets)
        # Percentiles should range from 0 to 100
        percentiles = sorted([s.percentile for s in scores])
        assert percentiles[0] == 0.0
        assert percentiles[-1] == 100.0
        assert len(set(percentiles)) > 1  # not all the same

    def test_single_wallet(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores = scorer.score_all([_make_wallet("0x1")])
        assert len(scores) == 1
        assert scores[0].percentile == 100.0

    def test_empty_wallets(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores = scorer.score_all([])
        assert scores == []


# ═══════════════════════════════════════════════════════════════════
#  QUALITY FILTERING
# ═══════════════════════════════════════════════════════════════════

class TestQualityFiltering:
    """Test get_top_percentile and is_qualified."""

    def test_top_percentile_filter(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores = [
            WhaleQualityScore(address="0x1", percentile=80),
            WhaleQualityScore(address="0x2", percentile=60),
            WhaleQualityScore(address="0x3", percentile=40),
            WhaleQualityScore(address="0x4", percentile=20),
        ]
        top = scorer.get_top_percentile(scores, threshold=60.0)
        assert len(top) == 2
        assert {s.address for s in top} == {"0x1", "0x2"}

    def test_boundary_included(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores = [WhaleQualityScore(address="0x1", percentile=60.0)]
        top = scorer.get_top_percentile(scores, threshold=60.0)
        assert len(top) == 1

    def test_empty_input(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        top = scorer.get_top_percentile([], threshold=60.0)
        assert top == []

    def test_is_qualified_true(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        score = WhaleQualityScore(address="0x1", percentile=75.0)
        assert scorer.is_qualified(score, threshold_pct=60.0) is True

    def test_is_qualified_false(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        score = WhaleQualityScore(address="0x1", percentile=50.0)
        assert scorer.is_qualified(score, threshold_pct=60.0) is False


# ═══════════════════════════════════════════════════════════════════
#  DATABASE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════

class TestDBPersistence:
    """Test save_scores, record_entry_snapshot, update_pending_snapshots."""

    def test_save_scores(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores = [
            WhaleQualityScore(address="0xA", name="Alpha", composite_score=85.0,
                              percentile=100.0, scored_at="2025-01-01T00:00:00Z"),
            WhaleQualityScore(address="0xB", name="Beta", composite_score=60.0,
                              percentile=50.0, scored_at="2025-01-01T00:00:00Z"),
        ]
        scorer.save_scores(conn, scores)
        rows = conn.execute("SELECT * FROM whale_quality_scores ORDER BY composite_score DESC").fetchall()
        assert len(rows) == 2
        assert dict(rows[0])["address"] == "0xA"
        assert dict(rows[0])["composite_score"] == 85.0

    def test_save_scores_upsert(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scores1 = [WhaleQualityScore(address="0xA", composite_score=50.0, scored_at="t1")]
        scorer.save_scores(conn, scores1)
        # Update with new score
        scores2 = [WhaleQualityScore(address="0xA", composite_score=90.0, scored_at="t2")]
        scorer.save_scores(conn, scores2)
        rows = conn.execute("SELECT * FROM whale_quality_scores").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["composite_score"] == 90.0

    def test_record_entry_snapshot(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scorer.record_entry_snapshot(
            conn, "0xA", "test-market", "Yes", 0.55, "BULLISH",
        )
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["wallet_address"] == "0xA"
        assert row["entry_price"] == 0.55
        assert row["price_24h_recorded"] == 0
        assert row["direction"] == "BULLISH"

    def test_update_pending_snapshots(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn, timing_favorable_threshold=0.02)
        # Insert a snapshot from 25 hours ago (past 24h cutoff)
        past = (dt.datetime.utcnow() - dt.timedelta(hours=25)).isoformat() + "Z"
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price, entry_time,
                price_24h_recorded, direction)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            ("0xA", "market-1", "Yes", 0.50, past, "BULLISH"),
        )
        conn.commit()

        # Price went from 0.50 to 0.55 → +10% → favorable (>= 2%)
        updated = scorer.update_pending_snapshots(conn, lambda slug: 0.55)
        assert updated == 1

        row = dict(conn.execute("SELECT * FROM whale_price_snapshots").fetchone())
        assert row["price_24h_recorded"] == 1
        assert row["price_after_24h"] == 0.55
        assert row["favorable_move"] == 1

    def test_update_pending_unfavorable(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn, timing_favorable_threshold=0.02)
        past = (dt.datetime.utcnow() - dt.timedelta(hours=25)).isoformat() + "Z"
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price, entry_time,
                price_24h_recorded, direction)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            ("0xA", "market-1", "Yes", 0.50, past, "BULLISH"),
        )
        conn.commit()

        # Price went from 0.50 to 0.49 → -2% → unfavorable
        updated = scorer.update_pending_snapshots(conn, lambda slug: 0.49)
        assert updated == 1
        row = dict(conn.execute("SELECT * FROM whale_price_snapshots").fetchone())
        assert row["favorable_move"] == 0

    def test_update_skips_recent(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        # Insert a snapshot from 1 hour ago (within 24h → should be skipped)
        recent = (dt.datetime.utcnow() - dt.timedelta(hours=1)).isoformat() + "Z"
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price, entry_time,
                price_24h_recorded, direction)
               VALUES (?, ?, ?, ?, ?, 0, ?)""",
            ("0xA", "market-1", "Yes", 0.50, recent, "BULLISH"),
        )
        conn.commit()
        updated = scorer.update_pending_snapshots(conn, lambda slug: 0.60)
        assert updated == 0

    def test_update_skips_already_recorded(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        past = (dt.datetime.utcnow() - dt.timedelta(hours=25)).isoformat() + "Z"
        conn.execute(
            """INSERT INTO whale_price_snapshots
               (wallet_address, market_slug, outcome, entry_price, entry_time,
                price_24h_recorded, direction, price_after_24h, favorable_move)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            ("0xA", "market-1", "Yes", 0.50, past, "BULLISH", 0.55, 1),
        )
        conn.commit()
        updated = scorer.update_pending_snapshots(conn, lambda slug: 0.60)
        assert updated == 0  # already recorded


# ═══════════════════════════════════════════════════════════════════
#  MIGRATION 13
# ═══════════════════════════════════════════════════════════════════

class TestMigration13:
    """Test that migration 13 creates the right tables."""

    def test_migration_creates_tables(self):
        from src.storage.migrations import run_migrations, SCHEMA_VERSION
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)
        # Check tables exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "whale_quality_scores" in tables
        assert "whale_price_snapshots" in tables

    def test_schema_version(self):
        from src.storage.migrations import run_migrations, SCHEMA_VERSION
        conn = sqlite3.connect(":memory:")
        run_migrations(conn)
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert version >= 13
        assert SCHEMA_VERSION >= 13


# ═══════════════════════════════════════════════════════════════════
#  CONVICTION QUALITY FILTER
# ═══════════════════════════════════════════════════════════════════

class TestConvictionQualityFilter:
    """Test _compute_conviction with qualified_addresses and quality_weights."""

    def _make_position(self, slug: str, outcome: str = "Yes", value: float = 100):
        pos = MagicMock()
        pos.market_slug = slug
        pos.outcome = outcome
        pos.current_value = value
        pos.avg_price = 0.5
        pos.size = 200
        pos.cur_price = 0.55
        pos.title = "Test Market"
        pos.condition_id = "cond1"
        return pos

    def test_unqualified_excluded(self):
        """Only qualified addresses should contribute to signals."""
        scanner = WalletScanner(
            wallets=[
                {"address": "0x1", "name": "Good"},
                {"address": "0x2", "name": "Bad"},
                {"address": "0x3", "name": "Good2"},
            ],
            min_whale_count=2,
        )
        pos = self._make_position("market-a")
        all_positions = {
            "0x1": [pos],
            "0x2": [pos],
            "0x3": [pos],
        }
        # Without filter: 3 whales on same market → signal
        signals_all = scanner._compute_conviction(all_positions, "now")
        assert len(signals_all) >= 1

        # With filter: only 0x1 and 0x3 qualified → 2 whales still enough
        signals_filtered = scanner._compute_conviction(
            all_positions, "now",
            qualified_addresses={"0x1", "0x3"},
        )
        assert len(signals_filtered) >= 1
        # Whale count should reflect qualified only
        assert signals_filtered[0].whale_count == 2

    def test_quality_weights_scale_conviction(self):
        """Quality weights should change conviction score via count_factor."""
        scanner = WalletScanner(
            wallets=[
                {"address": "0x1", "name": "HighQ"},
                {"address": "0x2", "name": "LowQ"},
            ],
            min_whale_count=2,
            min_conviction_score=0,
        )
        pos = self._make_position("market-b", value=1000)
        all_positions = {"0x1": [pos], "0x2": [pos]}

        # Without weights: count_factor = 2 * 25 = 50
        signals_no_wt = scanner._compute_conviction(all_positions, "now")

        # With weights: count_factor = (0.9 + 0.3) * 25 = 30
        signals_wt = scanner._compute_conviction(
            all_positions, "now",
            quality_weights={"0x1": 0.9, "0x2": 0.3},
        )

        assert len(signals_no_wt) >= 1
        assert len(signals_wt) >= 1
        # Weighted conviction should be lower
        assert signals_wt[0].conviction_score < signals_no_wt[0].conviction_score

    def test_filter_removes_all_whales(self):
        """If no qualified addresses match, no signals should be produced."""
        scanner = WalletScanner(
            wallets=[{"address": "0x1", "name": "W1"}],
            min_whale_count=1,
        )
        pos = self._make_position("market-c")
        signals = scanner._compute_conviction(
            {"0x1": [pos]}, "now",
            qualified_addresses=set(),  # empty → no one qualifies
        )
        assert len(signals) == 0

    def test_filter_disabled_includes_all(self):
        """When qualified_addresses is None, all wallets contribute."""
        scanner = WalletScanner(
            wallets=[{"address": "0x1", "name": "W1"}],
            min_whale_count=1,
            min_conviction_score=0,
        )
        pos = self._make_position("market-d", value=5000)
        signals = scanner._compute_conviction(
            {"0x1": [pos]}, "now",
            qualified_addresses=None,
        )
        assert len(signals) >= 1


# ═══════════════════════════════════════════════════════════════════
#  CONFIG PHASE 7
# ═══════════════════════════════════════════════════════════════════

class TestConfigPhase7:
    """Test Phase 7 config fields in WalletScannerConfig."""

    def test_defaults_disabled(self):
        cfg = WalletScannerConfig()
        assert cfg.whale_quality_scoring_enabled is False
        assert cfg.whale_quality_lookback_days == 90
        assert cfg.whale_quality_min_percentile == 60.0
        assert cfg.whale_timing_favorable_threshold == 0.02
        assert cfg.enhanced_min_whale_count == 3
        assert cfg.enhanced_conviction_edge_boost == 0.04
        assert cfg.enhanced_conviction_edge_penalty == 0.03

    def test_enabled_override(self):
        cfg = WalletScannerConfig(whale_quality_scoring_enabled=True)
        assert cfg.whale_quality_scoring_enabled is True

    def test_enhanced_thresholds(self):
        cfg = WalletScannerConfig(
            enhanced_min_whale_count=5,
            enhanced_conviction_edge_boost=0.06,
            enhanced_conviction_edge_penalty=0.05,
        )
        assert cfg.enhanced_min_whale_count == 5
        assert cfg.enhanced_conviction_edge_boost == 0.06

    def test_full_config_loads(self):
        cfg = BotConfig()
        assert hasattr(cfg.wallet_scanner, "whale_quality_scoring_enabled")
        assert cfg.wallet_scanner.whale_quality_scoring_enabled is False


# ═══════════════════════════════════════════════════════════════════
#  SAVE RESULT WITH SNAPSHOTS
# ═══════════════════════════════════════════════════════════════════

class TestSaveResultWithSnapshots:
    """Test save_scan_result creates price snapshots for NEW_ENTRY deltas."""

    def test_snapshots_for_entries(self):
        conn = _create_test_db()
        result = ScanResult(
            scanned_at="2025-01-01T00:00:00Z",
            deltas=[
                WalletDelta(
                    wallet_address="0x1", wallet_name="W1",
                    action="NEW_ENTRY", market_slug="test-mkt",
                    title="Test", outcome="Yes", current_price=0.55,
                    detected_at="2025-01-01T00:00:00Z",
                ),
            ],
        )
        save_scan_result(conn, result)
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["wallet_address"] == "0x1"
        assert row["entry_price"] == 0.55
        assert row["direction"] == "BULLISH"  # outcome="Yes" → BULLISH
        assert row["price_24h_recorded"] == 0

    def test_no_snapshots_for_exits(self):
        conn = _create_test_db()
        result = ScanResult(
            scanned_at="2025-01-01T00:00:00Z",
            deltas=[
                WalletDelta(
                    wallet_address="0x1", wallet_name="W1",
                    action="EXIT", market_slug="test-mkt",
                    title="Test", outcome="Yes", current_price=0.0,
                    detected_at="2025-01-01T00:00:00Z",
                ),
            ],
        )
        save_scan_result(conn, result)
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 0

    def test_quality_scores_saved(self):
        conn = _create_test_db()
        qs = WhaleQualityScore(
            address="0xA", name="Test", composite_score=75.0,
            percentile=80.0, scored_at="2025-01-01T00:00:00Z",
        )
        result = ScanResult(
            scanned_at="2025-01-01T00:00:00Z",
            quality_scores=[qs],
        )
        save_scan_result(conn, result)
        rows = conn.execute("SELECT * FROM whale_quality_scores").fetchall()
        assert len(rows) == 1
        assert dict(rows[0])["composite_score"] == 75.0

    def test_no_snapshot_for_zero_price(self):
        conn = _create_test_db()
        result = ScanResult(
            scanned_at="2025-01-01T00:00:00Z",
            deltas=[
                WalletDelta(
                    wallet_address="0x1", wallet_name="W1",
                    action="NEW_ENTRY", market_slug="test-mkt",
                    title="Test", outcome="Yes", current_price=0.0,
                    detected_at="2025-01-01T00:00:00Z",
                ),
            ],
        )
        save_scan_result(conn, result)
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 0  # price=0 → no snapshot


# ═══════════════════════════════════════════════════════════════════
#  TIMING REPORT
# ═══════════════════════════════════════════════════════════════════

class TestTimingReport:
    """Test compute_wallet_timing_report."""

    def test_report_structure(self):
        conn = _create_test_db()
        addr = "0xTIME"
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),
            (0.50, 0.45, "BULLISH", 0),
        ])
        scorer = WhaleScorer(conn)
        report = scorer.compute_wallet_timing_report(addr)
        assert "total_entries" in report
        assert "favorable_entries" in report
        assert "timing_score" in report
        assert "avg_favorable_move_pct" in report
        assert "avg_unfavorable_move_pct" in report

    def test_all_favorable(self):
        conn = _create_test_db()
        addr = "0xGOOD"
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),
            (0.40, 0.55, "BULLISH", 1),
        ])
        scorer = WhaleScorer(conn)
        report = scorer.compute_wallet_timing_report(addr)
        assert report["total_entries"] == 2
        assert report["favorable_entries"] == 2
        assert report["timing_score"] == 100.0
        assert report["avg_favorable_move_pct"] > 0

    def test_no_entries(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        report = scorer.compute_wallet_timing_report("0xNONE")
        assert report["total_entries"] == 0
        assert report["timing_score"] == 0.0

    def test_avg_calculations(self):
        conn = _create_test_db()
        addr = "0xAVG"
        # 2 favorable, 1 unfavorable
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),   # +20% move
            (0.50, 0.55, "BULLISH", 1),   # +10% move
            (0.50, 0.45, "BULLISH", 0),   # -10% move
        ])
        scorer = WhaleScorer(conn)
        report = scorer.compute_wallet_timing_report(addr)
        assert report["favorable_entries"] == 2
        # avg favorable: (20 + 10) / 2 = 15%
        assert abs(report["avg_favorable_move_pct"] - 15.0) < 0.5
        # avg unfavorable: -10%
        assert abs(report["avg_unfavorable_move_pct"] - (-10.0)) < 0.5

    def test_bearish_timing(self):
        conn = _create_test_db()
        addr = "0xBEAR"
        # Bearish: favorable if price dropped
        _insert_snapshots(conn, addr, [
            (0.60, 0.40, "BEARISH", 1),  # price dropped → favorable for bears
        ])
        scorer = WhaleScorer(conn)
        report = scorer.compute_wallet_timing_report(addr)
        assert report["favorable_entries"] == 1
        assert report["avg_favorable_move_pct"] > 0


# ═══════════════════════════════════════════════════════════════════
#  SCORE_WALLET FULL INTEGRATION
# ═══════════════════════════════════════════════════════════════════

class TestScoreWallet:
    """Test score_wallet() integration — all dimensions together."""

    def test_produces_all_dimensions(self):
        conn = _create_test_db()
        addr = "0xINTEG"
        _insert_deltas(conn, addr, [
            ("NEW_ENTRY", "Trump election bet", -200),
            ("EXIT", "Biden election outcome", 500),
            ("EXIT", "Congress vote prediction", 300),
        ])
        _insert_snapshots(conn, addr, [
            (0.50, 0.60, "BULLISH", 1),
        ])
        scorer = WhaleScorer(conn)
        s = scorer.score_wallet(_make_wallet(addr))
        assert s.address == addr
        assert s.historical_roi > 0
        assert s.composite_score > 0
        assert s.best_category == "Politics"
        assert s.scored_at != ""

    def test_empty_wallet(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        s = scorer.score_wallet(_make_wallet("0xEMPTY"))
        assert s.composite_score == 0.0
        assert s.trade_count_90d == 0


# ═══════════════════════════════════════════════════════════════════
#  ENHANCED ENGINE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════

class TestEnhancedThresholds:
    """Test that engine uses enhanced thresholds when quality scoring is enabled."""

    def test_enhanced_when_enabled(self):
        cfg = WalletScannerConfig(whale_quality_scoring_enabled=True)
        assert cfg.enhanced_conviction_edge_boost == 0.04
        assert cfg.enhanced_conviction_edge_penalty == 0.03
        # Engine should pick these when enabled
        if cfg.whale_quality_scoring_enabled:
            boost = cfg.enhanced_conviction_edge_boost
            penalty = cfg.enhanced_conviction_edge_penalty
        else:
            boost = cfg.conviction_edge_boost
            penalty = cfg.conviction_edge_penalty
        assert boost == 0.04
        assert penalty == 0.03

    def test_original_when_disabled(self):
        cfg = WalletScannerConfig(whale_quality_scoring_enabled=False)
        if cfg.whale_quality_scoring_enabled:
            boost = cfg.enhanced_conviction_edge_boost
        else:
            boost = cfg.conviction_edge_boost
        assert boost == 0.08  # original value

    def test_default_values(self):
        cfg = WalletScannerConfig()
        assert cfg.conviction_edge_boost == 0.08
        assert cfg.conviction_edge_penalty == 0.02
        assert cfg.enhanced_conviction_edge_boost == 0.04
        assert cfg.enhanced_conviction_edge_penalty == 0.03


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD ENDPOINT
# ═══════════════════════════════════════════════════════════════════

class TestDashboardQualityEndpoint:
    """Test /api/whale-quality-scores endpoint."""

    def test_endpoint_empty(self):
        from src.dashboard.app import app
        from src.storage.migrations import run_migrations
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-quality-scores")
                assert resp.status_code == 200
                data = resp.get_json()
                assert "scores" in data
                assert data["total_count"] == 0
                assert "percentile_threshold" in data
                assert "scoring_enabled" in data

    def test_endpoint_with_scores(self):
        from src.dashboard.app import app
        # Insert test scores directly into the DB
        from src.storage.migrations import run_migrations
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)
        conn.execute(
            """INSERT INTO whale_quality_scores
               (address, name, composite_score, percentile, scored_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("0xA", "Alpha", 85.0, 100.0, "2025-01-01T00:00:00Z"),
        )
        conn.commit()

        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-quality-scores")
                assert resp.status_code == 200
                data = resp.get_json()
                assert len(data["scores"]) == 1
                assert data["scores"][0]["address"] == "0xA"
                assert data["qualified_count"] == 1
                assert data["total_count"] == 1


# ═══════════════════════════════════════════════════════════════════
#  SCAN RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════

class TestScanResultQualityScores:
    """Test ScanResult.quality_scores field."""

    def test_default_empty(self):
        result = ScanResult(scanned_at="now")
        assert result.quality_scores == []

    def test_with_scores(self):
        qs = WhaleQualityScore(address="0x1", composite_score=70.0)
        result = ScanResult(scanned_at="now", quality_scores=[qs])
        assert len(result.quality_scores) == 1
        assert result.quality_scores[0].composite_score == 70.0


# ═══════════════════════════════════════════════════════════════════
#  ENTRY SNAPSHOT RECORDING
# ═══════════════════════════════════════════════════════════════════

class TestEntrySnapshotRecording:
    """Test record_entry_snapshot and snapshot creation flow."""

    def test_creates_row(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scorer.record_entry_snapshot(conn, "0xA", "market-x", "Yes", 0.65, "BULLISH")
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["market_slug"] == "market-x"
        assert row["entry_price"] == 0.65

    def test_direction_bearish(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scorer.record_entry_snapshot(conn, "0xA", "market-y", "No", 0.35, "BEARISH")
        row = dict(conn.execute("SELECT * FROM whale_price_snapshots").fetchone())
        assert row["direction"] == "BEARISH"

    def test_multiple_entries(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scorer.record_entry_snapshot(conn, "0xA", "m1", "Yes", 0.50, "BULLISH")
        scorer.record_entry_snapshot(conn, "0xA", "m2", "No", 0.40, "BEARISH")
        scorer.record_entry_snapshot(conn, "0xB", "m1", "Yes", 0.55, "BULLISH")
        rows = conn.execute("SELECT * FROM whale_price_snapshots").fetchall()
        assert len(rows) == 3

    def test_snapshot_not_yet_recorded(self):
        conn = _create_test_db()
        scorer = WhaleScorer(conn)
        scorer.record_entry_snapshot(conn, "0xA", "m1", "Yes", 0.50, "BULLISH")
        row = dict(conn.execute("SELECT * FROM whale_price_snapshots").fetchone())
        assert row["price_24h_recorded"] == 0
        assert row["price_after_24h"] == 0


# ═══════════════════════════════════════════════════════════════════
#  COMPOSITE SCORE FORMULA
# ═══════════════════════════════════════════════════════════════════

class TestCompositeFormula:
    """Verify composite score weights sum to 1.0 and edge cases."""

    def test_weights_sum_to_one(self):
        s = WhaleQualityScore(
            address="0x1",
            historical_roi=100,
            calibration_quality=100,
            category_specialization=100,
            consistency=100,
            timing_score=100,
        )
        composite = _compute_composite(s)
        assert abs(composite - 100.0) < 0.01  # all 100 → composite = 100

    def test_all_zeros(self):
        s = WhaleQualityScore(address="0x1")
        assert _compute_composite(s) == 0.0

    def test_partial_scores(self):
        s = WhaleQualityScore(
            address="0x1",
            historical_roi=50,
            calibration_quality=0,
            category_specialization=0,
            consistency=0,
            timing_score=0,
        )
        # Only ROI contributes: 0.25 * 50 = 12.5
        assert abs(_compute_composite(s) - 12.5) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD: QUALITY TIERS + SUMMARY (Batch B)
# ═══════════════════════════════════════════════════════════════════

def _create_dashboard_test_db():
    """Create a DB with migrations + dashboard tables for whale-activity tests."""
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    # Create extra tables that the dashboard _ensure_tables would create
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS whale_stars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            star_type TEXT NOT NULL,
            identifier TEXT NOT NULL,
            label TEXT DEFAULT '',
            starred_at TEXT DEFAULT (datetime('now')),
            UNIQUE(star_type, identifier)
        );
        CREATE TABLE IF NOT EXISTS whale_alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            detail_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(alert_type, message)
        );
        CREATE TABLE IF NOT EXISTS conviction_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_slug TEXT NOT NULL,
            outcome TEXT DEFAULT '',
            conviction_score REAL DEFAULT 0,
            whale_count INTEGER DEFAULT 0,
            total_whale_usd REAL DEFAULT 0,
            snapped_at TEXT DEFAULT (datetime('now'))
        );
    """)
    return conn


def _seed_whale_data(conn: sqlite3.Connection) -> None:
    """Seed whale wallets + quality scores for dashboard tests."""
    now = dt.datetime.utcnow().isoformat() + "Z"
    # Tracked wallets
    for i, (addr, name, pnl) in enumerate([
        ("0xA1", "AlphaWhale", 3_000_000),
        ("0xB2", "BetaWhale", 1_500_000),
        ("0xC3", "GammaWhale", 500_000),
    ]):
        conn.execute(
            """INSERT INTO tracked_wallets
               (address, name, total_pnl, win_rate, active_positions, total_volume, score, last_scanned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (addr, name, pnl, 0.65, 10, 100_000, 80 - i * 20, now),
        )
    # Quality scores
    for addr, name, composite, pctl, timing in [
        ("0xA1", "AlphaWhale", 85.0, 100.0, 80.0),
        ("0xB2", "BetaWhale", 60.0, 50.0, 55.0),
        ("0xC3", "GammaWhale", 35.0, 0.0, 30.0),
    ]:
        conn.execute(
            """INSERT INTO whale_quality_scores
               (address, name, composite_score, percentile, timing_score, scored_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (addr, name, composite, pctl, timing, now),
        )
    # Conviction signal
    conn.execute(
        """INSERT INTO wallet_signals
           (market_slug, title, condition_id, outcome, whale_count,
            total_whale_usd, avg_whale_price, current_price,
            conviction_score, whale_names_json, direction, signal_strength, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("test-market", "Test Market", "cond1", "Yes", 2,
         5000.0, 0.50, 0.55, 65.0,
         json.dumps(["AlphaWhale", "BetaWhale"]), "BULLISH", "MODERATE", now),
    )
    conn.commit()


class TestDashboardQualityEnrichment:
    """Test Batch B: quality tiers + summary in whale-activity response."""

    def test_quality_summary_present(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        _seed_whale_data(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                assert resp.status_code == 200
                data = resp.get_json()
                qs = data["quality_summary"]
                assert qs["scoring_enabled"] is True
                assert qs["total_scored"] == 3
                assert qs["qualified_count"] >= 1  # at least AlphaWhale (100th pctl)
                assert qs["best_timer"] == "AlphaWhale"  # highest timing_score
                assert qs["avg_timing_score"] > 0

    def test_quality_tiers_assigned(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        _seed_whale_data(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                wallets = data["tracked_wallets"]
                tiers = {w["name"]: w["quality_tier"] for w in wallets}
                assert tiers["AlphaWhale"] == "S-TIER"   # 100th percentile
                assert tiers["BetaWhale"] == "B-TIER"    # 50th percentile
                assert tiers["GammaWhale"] == "C-TIER"   # 0th percentile

    def test_disabled_shows_unscored(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        # Insert wallets but NO quality scores
        now = dt.datetime.utcnow().isoformat() + "Z"
        conn.execute(
            """INSERT INTO tracked_wallets (address, name, total_pnl, win_rate,
               active_positions, total_volume, score, last_scanned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("0xX", "NoScore", 100_000, 0.5, 5, 50_000, 60, now),
        )
        conn.commit()
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                w = data["tracked_wallets"][0]
                assert w["quality_tier"] == "UNSCORED"
                assert data["quality_summary"]["scoring_enabled"] is False

    def test_empty_response_has_quality(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        # No tracked_wallets table content → hits empty response
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                assert "quality_summary" in data
                assert "quality_distribution" in data


# ═══════════════════════════════════════════════════════════════════
#  DASHBOARD: WEIGHTED CONVICTION + DISTRIBUTION (Batch C)
# ═══════════════════════════════════════════════════════════════════

class TestDashboardQualityVisualization:
    """Test Batch C: weighted conviction and quality distribution."""

    def test_weighted_conviction_in_signals(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        _seed_whale_data(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                signals = data["conviction_signals"]
                assert len(signals) >= 1
                s = signals[0]
                assert "quality_weighted_conviction" in s
                # Weighted should differ from raw since whales have different quality
                assert isinstance(s["quality_weighted_conviction"], (int, float))

    def test_quality_distribution_buckets(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        _seed_whale_data(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                dist = data["quality_distribution"]
                assert isinstance(dist, list)
                assert len(dist) > 0
                # Each bucket should have 'bucket' and 'count'
                assert "bucket" in dist[0]
                assert "count" in dist[0]
                # Total counts should sum to total scored whales
                total = sum(b["count"] for b in dist)
                assert total == 3

    def test_weighted_conviction_without_scores(self):
        """When no quality scores exist, weighted conviction = raw conviction."""
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        now = dt.datetime.utcnow().isoformat() + "Z"
        # Insert wallet + signal but no quality scores
        conn.execute(
            """INSERT INTO tracked_wallets (address, name, total_pnl, win_rate,
               active_positions, total_volume, score, last_scanned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("0xY", "NoQScore", 100_000, 0.5, 5, 50_000, 60, now),
        )
        conn.execute(
            """INSERT INTO wallet_signals
               (market_slug, title, outcome, whale_count, total_whale_usd,
                conviction_score, whale_names_json, direction, signal_strength, detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("mkt-x", "Market X", "Yes", 1, 1000, 50.0,
             json.dumps(["NoQScore"]), "BULLISH", "MODERATE", now),
        )
        conn.commit()
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                s = data["conviction_signals"][0]
                # Without quality scores, weighted = raw
                assert s["quality_weighted_conviction"] == s["conviction_score"]

    def test_distribution_empty_when_no_scores(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        now = dt.datetime.utcnow().isoformat() + "Z"
        conn.execute(
            """INSERT INTO tracked_wallets (address, name, total_pnl, win_rate,
               active_positions, total_volume, score, last_scanned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("0xZ", "Empty", 50_000, 0.4, 3, 20_000, 40, now),
        )
        conn.commit()
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                assert data["quality_distribution"] == []

    def test_tier_counts_in_quality_summary(self):
        from src.dashboard.app import app
        conn = _create_dashboard_test_db()
        _seed_whale_data(conn)
        with patch("src.dashboard.app._get_conn", return_value=conn):
            with app.test_client() as client:
                resp = client.get("/api/whale-activity")
                data = resp.get_json()
                qs = data["quality_summary"]
                assert qs["total_scored"] == 3
                # At least 1 qualified (AlphaWhale at 100th percentile)
                assert qs["qualified_count"] >= 1
