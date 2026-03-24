"""Tests for Phase 8 Batch A: Post-mortem analysis + evidence quality tracking."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

import pytest

from src.analytics.post_mortem import (
    PostMortemAnalyzer,
    TradeAnalysis,
    WeeklySummary,
)
from src.analytics.evidence_quality import (
    EvidenceQualityTracker,
    SourceQualityRecord,
)


# ── Helpers ──────────────────────────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """In-memory DB with Phase 8 tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # performance_log (migration 5)
    conn.execute("""
        CREATE TABLE performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            question TEXT,
            category TEXT DEFAULT 'UNKNOWN',
            forecast_prob REAL,
            actual_outcome REAL,
            edge_at_entry REAL,
            confidence TEXT DEFAULT 'LOW',
            evidence_quality REAL DEFAULT 0,
            stake_usd REAL DEFAULT 0,
            entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0,
            pnl REAL DEFAULT 0,
            holding_hours REAL DEFAULT 0,
            resolved_at TEXT
        )
    """)

    # model_forecast_log (migration 5)
    conn.execute("""
        CREATE TABLE model_forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            market_id TEXT NOT NULL,
            category TEXT DEFAULT 'UNKNOWN',
            forecast_prob REAL,
            actual_outcome REAL,
            recorded_at TEXT
        )
    """)

    # trade_analysis (migration 14)
    conn.execute("""
        CREATE TABLE trade_analysis (
            market_id TEXT PRIMARY KEY,
            question TEXT DEFAULT '',
            category TEXT DEFAULT '',
            forecast_prob REAL DEFAULT 0,
            actual_outcome REAL DEFAULT 0,
            was_correct INTEGER DEFAULT 0,
            confidence_error REAL DEFAULT 0,
            was_confident_and_wrong INTEGER DEFAULT 0,
            best_model TEXT DEFAULT '',
            worst_model TEXT DEFAULT '',
            model_errors_json TEXT DEFAULT '{}',
            evidence_quality REAL DEFAULT 0,
            evidence_sources_json TEXT DEFAULT '[]',
            position_size_appropriate TEXT DEFAULT '',
            pnl REAL DEFAULT 0,
            edge_at_entry REAL DEFAULT 0,
            holding_hours REAL DEFAULT 0,
            analyzed_at TEXT
        )
    """)

    # evidence_source_quality (migration 14)
    conn.execute("""
        CREATE TABLE evidence_source_quality (
            domain TEXT PRIMARY KEY,
            times_cited INTEGER DEFAULT 0,
            times_correct INTEGER DEFAULT 0,
            correct_forecast_rate REAL DEFAULT 0,
            avg_evidence_quality REAL DEFAULT 0,
            avg_authority REAL DEFAULT 0,
            quality_trend TEXT DEFAULT 'stable',
            effective_weight REAL DEFAULT 1.0,
            last_updated TEXT
        )
    """)

    conn.commit()
    return conn


def _insert_perf_log(
    conn: sqlite3.Connection,
    market_id: str,
    question: str = "Test?",
    category: str = "MACRO",
    forecast_prob: float = 0.7,
    actual_outcome: float | None = 1.0,
    edge_at_entry: float = 0.05,
    confidence: str = "HIGH",
    evidence_quality: float = 0.6,
    stake_usd: float = 50.0,
    entry_price: float = 0.5,
    exit_price: float = 1.0,
    pnl: float = 50.0,
    holding_hours: float = 24.0,
    resolved_at: str = "2025-03-01T00:00:00Z",
) -> None:
    conn.execute("""
        INSERT INTO performance_log
            (market_id, question, category, forecast_prob, actual_outcome,
             edge_at_entry, confidence, evidence_quality, stake_usd,
             entry_price, exit_price, pnl, holding_hours, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        market_id, question, category, forecast_prob, actual_outcome,
        edge_at_entry, confidence, evidence_quality, stake_usd,
        entry_price, exit_price, pnl, holding_hours, resolved_at,
    ))
    conn.commit()


def _insert_model_forecast(
    conn: sqlite3.Connection,
    model_name: str,
    market_id: str,
    forecast_prob: float,
    actual_outcome: float = 1.0,
    category: str = "MACRO",
    recorded_at: str = "2025-03-01T00:00:00Z",
) -> None:
    conn.execute("""
        INSERT INTO model_forecast_log
            (model_name, market_id, category, forecast_prob,
             actual_outcome, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (model_name, market_id, category, forecast_prob, actual_outcome, recorded_at))
    conn.commit()


# ── TradeAnalysis Dataclass ──────────────────────────────────────


class TestTradeAnalysis:
    def test_defaults(self):
        ta = TradeAnalysis(market_id="m1")
        assert ta.market_id == "m1"
        assert ta.was_correct is False
        assert ta.was_confident_and_wrong is False
        assert ta.model_errors == {}
        assert ta.position_size_appropriate == ""

    def test_to_dict(self):
        ta = TradeAnalysis(
            market_id="m1",
            question="Will X?",
            was_correct=True,
            pnl=42.0,
        )
        d = ta.to_dict()
        assert d["market_id"] == "m1"
        assert d["was_correct"] is True
        assert d["pnl"] == 42.0
        assert "model_errors" in d

    def test_correct_flag(self):
        ta = TradeAnalysis(market_id="m1", was_correct=True)
        assert ta.was_correct is True

    def test_confident_and_wrong(self):
        ta = TradeAnalysis(
            market_id="m1",
            was_confident_and_wrong=True,
            confidence_error=0.8,
        )
        assert ta.was_confident_and_wrong is True
        assert ta.confidence_error == 0.8


# ── AnalyzeMarket ────────────────────────────────────────────────


class TestAnalyzeMarket:
    def test_correct_forecast(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.8, actual_outcome=1.0, pnl=40.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis is not None
        assert analysis.was_correct is True
        assert analysis.was_confident_and_wrong is False

    def test_wrong_forecast(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.7, actual_outcome=0.0, pnl=-50.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis is not None
        assert analysis.was_correct is False

    def test_confident_wrong(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.85, actual_outcome=0.0, pnl=-50.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis.was_confident_and_wrong is True
        assert analysis.confidence_error == pytest.approx(0.85, abs=0.001)

    def test_confident_wrong_low(self):
        """Low forecast (<=0.25) with outcome=1 should also be confident_and_wrong."""
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.15, actual_outcome=1.0, pnl=-50.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis.was_confident_and_wrong is True

    def test_model_ranking(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.7, actual_outcome=1.0)
        _insert_model_forecast(conn, "gpt-4o", "m1", 0.8, 1.0)
        _insert_model_forecast(conn, "claude", "m1", 0.3, 1.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis.best_model == "gpt-4o"  # error 0.2
        assert analysis.worst_model == "claude"  # error 0.7

    def test_no_model_data(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.7, actual_outcome=1.0)
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis.best_model == ""
        assert analysis.worst_model == ""
        assert analysis.model_errors == {}

    def test_position_sizing(self):
        conn = _create_test_db()
        _insert_perf_log(
            conn, "m1", forecast_prob=0.8, actual_outcome=1.0,
            stake_usd=50.0, pnl=50.0,
        )
        analyzer = PostMortemAnalyzer(conn)
        analysis = analyzer.analyze_market("m1")
        assert analysis.position_size_appropriate in ("appropriate", "too_small", "too_large")

    def test_nonexistent_market(self):
        conn = _create_test_db()
        analyzer = PostMortemAnalyzer(conn)
        assert analyzer.analyze_market("nonexistent") is None

    def test_saves_to_db(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", forecast_prob=0.7, actual_outcome=1.0)
        analyzer = PostMortemAnalyzer(conn)
        analyzer.analyze_market("m1")
        row = conn.execute(
            "SELECT * FROM trade_analysis WHERE market_id = 'm1'"
        ).fetchone()
        assert row is not None
        assert row["was_correct"] == 1


# ── AnalyzeAllPending ────────────────────────────────────────────


class TestAnalyzeAllPending:
    def test_processes_pending(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", actual_outcome=1.0)
        _insert_perf_log(conn, "m2", actual_outcome=0.0, forecast_prob=0.3)
        analyzer = PostMortemAnalyzer(conn)
        results = analyzer.analyze_all_pending()
        assert len(results) == 2

    def test_skips_already_analyzed(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", actual_outcome=1.0)
        analyzer = PostMortemAnalyzer(conn)
        results1 = analyzer.analyze_all_pending()
        assert len(results1) == 1
        results2 = analyzer.analyze_all_pending()
        assert len(results2) == 0

    def test_skips_unresolved(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", actual_outcome=None)
        analyzer = PostMortemAnalyzer(conn)
        results = analyzer.analyze_all_pending()
        assert len(results) == 0

    def test_empty_performance_log(self):
        conn = _create_test_db()
        analyzer = PostMortemAnalyzer(conn)
        results = analyzer.analyze_all_pending()
        assert results == []


# ── WeeklySummary ────────────────────────────────────────────────


class TestWeeklySummary:
    def test_defaults(self):
        summary = WeeklySummary()
        assert summary.total_resolved == 0
        assert summary.correct_count == 0

    def test_to_dict(self):
        summary = WeeklySummary(total_resolved=5, correct_count=3, accuracy_pct=60.0)
        d = summary.to_dict()
        assert d["total_resolved"] == 5
        assert d["accuracy_pct"] == 60.0

    def test_structure(self):
        conn = _create_test_db()
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        _insert_perf_log(conn, "m1", category="MACRO", actual_outcome=1.0, pnl=50.0)
        analyzer = PostMortemAnalyzer(conn)
        analyzer.analyze_market("m1")
        summary = analyzer.generate_weekly_summary(lookback_days=30)
        assert summary.total_resolved >= 1
        assert summary.correct_count >= 1
        assert summary.accuracy_pct > 0

    def test_category_ranking(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", category="MACRO", pnl=100.0, actual_outcome=1.0)
        _insert_perf_log(conn, "m2", category="SPORTS", pnl=-50.0, actual_outcome=0.0, forecast_prob=0.7)
        analyzer = PostMortemAnalyzer(conn)
        analyzer.analyze_market("m1")
        analyzer.analyze_market("m2")
        summary = analyzer.generate_weekly_summary(lookback_days=30)
        if summary.top_winning_categories:
            assert summary.top_winning_categories[0]["category"] == "MACRO"
        if summary.top_losing_categories:
            assert summary.top_losing_categories[0]["category"] == "SPORTS"

    def test_empty_period(self):
        conn = _create_test_db()
        analyzer = PostMortemAnalyzer(conn)
        summary = analyzer.generate_weekly_summary()
        assert summary.total_resolved == 0
        assert summary.top_winning_categories == []

    def test_model_accuracy(self):
        conn = _create_test_db()
        _insert_perf_log(conn, "m1", actual_outcome=1.0)
        _insert_perf_log(conn, "m2", actual_outcome=1.0)
        _insert_perf_log(conn, "m3", actual_outcome=1.0)
        now_iso = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        for mid in ("m1", "m2", "m3"):
            _insert_model_forecast(conn, "gpt-4o", mid, 0.9, 1.0, recorded_at=now_iso)
            _insert_model_forecast(conn, "claude", mid, 0.3, 1.0, recorded_at=now_iso)
        analyzer = PostMortemAnalyzer(conn)
        for mid in ("m1", "m2", "m3"):
            analyzer.analyze_market(mid)
        summary = analyzer.generate_weekly_summary(lookback_days=30)
        assert summary.most_accurate_model == "gpt-4o"
        assert summary.least_accurate_model == "claude"


# ── SourceQualityRecord ──────────────────────────────────────────


class TestSourceQualityRecord:
    def test_defaults(self):
        r = SourceQualityRecord(domain="example.com")
        assert r.times_cited == 0
        assert r.effective_weight == 1.0
        assert r.quality_trend == "stable"

    def test_to_dict(self):
        r = SourceQualityRecord(domain="reuters.com", times_cited=10)
        d = r.to_dict()
        assert d["domain"] == "reuters.com"
        assert d["times_cited"] == 10

    def test_quality_trend(self):
        r = SourceQualityRecord(domain="a.com", quality_trend="improving")
        assert r.quality_trend == "improving"


# ── EvidenceQualityTracker ───────────────────────────────────────


class TestEvidenceQualityTracker:
    def test_record_source(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn)
        tracker.record_source_outcome("reuters.com", True, 0.8, 1.0)
        row = conn.execute(
            "SELECT * FROM evidence_source_quality WHERE domain = 'reuters.com'"
        ).fetchone()
        assert row is not None
        assert row["times_cited"] == 1
        assert row["times_correct"] == 1

    def test_record_multiple(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn)
        tracker.record_source_outcome("reuters.com", True, 0.8, 1.0)
        tracker.record_source_outcome("reuters.com", False, 0.6, 1.0)
        tracker.record_source_outcome("reuters.com", True, 0.7, 1.0)
        row = conn.execute(
            "SELECT * FROM evidence_source_quality WHERE domain = 'reuters.com'"
        ).fetchone()
        assert row["times_cited"] == 3
        assert row["times_correct"] == 2
        assert abs(row["correct_forecast_rate"] - 2.0/3) < 0.01

    def test_get_rankings(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn, min_citations=2)
        # reuters: 2 correct out of 2
        tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        # reddit: 0 correct out of 2
        tracker.record_source_outcome("reddit.com", False, 0.3, 0.4)
        tracker.record_source_outcome("reddit.com", False, 0.3, 0.4)
        rankings = tracker.get_domain_rankings(min_citations=2)
        assert len(rankings) == 2
        assert rankings[0].domain == "reuters.com"  # higher weight

    def test_effective_weight(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn)
        # Domain not tracked yet
        assert tracker.get_effective_weight("unknown.com") == 1.0
        # Add a high-quality source
        for _ in range(10):
            tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        weight = tracker.get_effective_weight("reuters.com")
        assert weight > 1.0

    def test_top_sources(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn, min_citations=1)
        tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        tracker.record_source_outcome("ap.com", True, 0.8, 0.9)
        top = tracker.get_top_sources(n=2)
        assert len(top) == 2
        assert "reuters.com" in top

    def test_blocklist(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn, min_citations=2)
        # Bad source
        tracker.record_source_outcome("fake-news.com", False, 0.2, 0.2)
        tracker.record_source_outcome("fake-news.com", False, 0.1, 0.2)
        # Good source
        tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
        blocklist = tracker.get_blocklist(threshold=0.3)
        assert "fake-news.com" in blocklist
        assert "reuters.com" not in blocklist

    def test_min_citations_filter(self):
        conn = _create_test_db()
        tracker = EvidenceQualityTracker(conn, min_citations=5)
        tracker.record_source_outcome("sparse.com", True, 0.9, 1.0)
        rankings = tracker.get_domain_rankings()
        assert len(rankings) == 0  # not enough citations


# ── Weight Computation ───────────────────────────────────────────


class TestWeightComputation:
    def test_high_accuracy_boost(self):
        weight = EvidenceQualityTracker._compute_weight(0.9, 0.5, 20)
        assert weight > 1.0

    def test_low_accuracy_penalty(self):
        weight = EvidenceQualityTracker._compute_weight(0.1, 0.5, 20)
        assert weight < 1.0

    def test_authority_boost(self):
        w_low = EvidenceQualityTracker._compute_weight(0.5, 0.3, 20)
        w_high = EvidenceQualityTracker._compute_weight(0.5, 0.9, 20)
        assert w_high > w_low

    def test_low_citations_dampen(self):
        w_few = EvidenceQualityTracker._compute_weight(0.9, 0.5, 2)
        w_many = EvidenceQualityTracker._compute_weight(0.9, 0.5, 20)
        # With few citations, weight should be closer to 1.0
        assert abs(w_few - 1.0) < abs(w_many - 1.0)

    def test_bounds(self):
        assert EvidenceQualityTracker._compute_weight(1.0, 1.0, 100) <= 1.5
        assert EvidenceQualityTracker._compute_weight(0.0, 0.0, 100) >= 0.5


# ── Config ───────────────────────────────────────────────────────


class TestConfigContinuousLearning:
    def test_defaults_disabled(self):
        from src.config import ContinuousLearningConfig
        cfg = ContinuousLearningConfig()
        assert cfg.post_mortem_enabled is False
        assert cfg.evidence_tracking_enabled is False
        assert cfg.param_optimizer_enabled is False
        assert cfg.smart_retrain_enabled is False

    def test_enable_fields(self):
        from src.config import ContinuousLearningConfig
        cfg = ContinuousLearningConfig(
            post_mortem_enabled=True,
            evidence_tracking_enabled=True,
        )
        assert cfg.post_mortem_enabled is True
        assert cfg.evidence_tracking_enabled is True

    def test_bot_config_has_section(self):
        from src.config import BotConfig
        cfg = BotConfig()
        assert hasattr(cfg, "continuous_learning")
        assert cfg.continuous_learning.post_mortem_enabled is False


# ── Migration 14 ─────────────────────────────────────────────────


class TestMigration14:
    def test_tables_created(self):
        from src.storage.migrations import run_migrations
        conn = sqlite3.connect(":memory:")
        run_migrations(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [r[0] for r in tables]
        assert "trade_analysis" in table_names
        assert "evidence_source_quality" in table_names
        assert "param_optimization_runs" in table_names
        assert "param_optimization_results" in table_names
        assert "calibration_ab_results" in table_names
        conn.close()

    def test_indexes_exist(self):
        from src.storage.migrations import run_migrations
        conn = sqlite3.connect(":memory:")
        run_migrations(conn)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = [r[0] for r in indexes]
        assert "idx_ta_category" in index_names
        assert "idx_ta_confident_wrong" in index_names
        assert "idx_esq_weight" in index_names
        assert "idx_por_run" in index_names
        conn.close()

    def test_schema_version(self):
        from src.storage.migrations import SCHEMA_VERSION
        assert SCHEMA_VERSION >= 14


# ── Dashboard Endpoints ──────────────────────────────────────────


def _create_dashboard_test_db() -> sqlite3.Connection:
    """Create a DB with migrations for dashboard endpoint tests."""
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


class TestDashboardEndpoints:
    @pytest.fixture
    def client(self):
        from src.dashboard.app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_post_mortem_recent_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/post-mortem/recent")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["analyses"] == []
            assert data["total"] == 0

    def test_post_mortem_recent_with_data(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            _insert_perf_log(conn, "m1", actual_outcome=1.0)
            analyzer = PostMortemAnalyzer(conn)
            analyzer.analyze_market("m1")
            mock.return_value = conn
            resp = client.get("/api/post-mortem/recent")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total"] == 1
            assert data["analyses"][0]["market_id"] == "m1"

    def test_post_mortem_summary_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/post-mortem/summary")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total_resolved"] == 0

    def test_evidence_quality_empty(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            mock.return_value = conn
            resp = client.get("/api/evidence-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["rankings"] == []
            assert data["total_domains"] == 0

    def test_evidence_quality_with_data(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = _create_dashboard_test_db()
            tracker = EvidenceQualityTracker(conn, min_citations=1)
            # Insert 5+ citations to pass the default min_citations=5 filter
            for _ in range(6):
                tracker.record_source_outcome("reuters.com", True, 0.9, 1.0)
            mock.return_value = conn
            resp = client.get("/api/evidence-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["total_domains"] == 1
            assert data["rankings"][0]["domain"] == "reuters.com"

    def test_post_mortem_no_table(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            mock.return_value = conn
            resp = client.get("/api/post-mortem/recent")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["analyses"] == []

    def test_evidence_quality_no_table(self, client):
        with patch("src.dashboard.app._get_conn") as mock:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            mock.return_value = conn
            resp = client.get("/api/evidence-quality")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["rankings"] == []


# ── Position Sizing Logic ────────────────────────────────────────


class TestPositionSizing:
    def test_appropriate(self):
        result = PostMortemAnalyzer._compute_position_appropriateness(
            forecast_prob=0.7, actual_outcome=1.0,
            stake_usd=50.0, bankroll=5000.0, kelly_fraction=0.25,
        )
        assert result in ("appropriate", "too_small", "too_large")

    def test_too_large(self):
        result = PostMortemAnalyzer._compute_position_appropriateness(
            forecast_prob=0.55, actual_outcome=1.0,
            stake_usd=2000.0, bankroll=5000.0, kelly_fraction=0.25,
        )
        assert result == "too_large"

    def test_too_small(self):
        result = PostMortemAnalyzer._compute_position_appropriateness(
            forecast_prob=0.9, actual_outcome=1.0,
            stake_usd=1.0, bankroll=5000.0, kelly_fraction=0.25,
        )
        assert result == "too_small"

    def test_zero_edge(self):
        result = PostMortemAnalyzer._compute_position_appropriateness(
            forecast_prob=0.5, actual_outcome=1.0,
            stake_usd=50.0, bankroll=5000.0, kelly_fraction=0.25,
        )
        assert result == "appropriate"
