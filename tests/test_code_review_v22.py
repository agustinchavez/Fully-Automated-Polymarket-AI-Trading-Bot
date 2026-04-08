"""Code review v22 fixes — whale convergence edge, evidence tracking wire,
FRED PMI keyword fix, base rate coverage gaps (MACRO, GEOPOLITICS, TECH).

Tests cover:
  1. whale_convergence_min_edge > total fees (Bug 1)
  2. evidence_tracking_enabled wired into exit_finalizer (Bug 2)
  3. FRED 'pmi' no longer maps to MANEMP (Bug 3)
  4. MACRO base rate upside/threshold patterns (Gap 1)
  5. GEOPOLITICS base rate patterns (Gap 2)
  6. TECH release/leaderboard patterns (Gap 3)
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest


# ── Bug 1: whale_convergence_min_edge > total fees ────────────────


class TestWhaleConvergenceEdge:
    """whale_convergence_min_edge must be above total fees."""

    def test_whale_convergence_min_edge_is_005(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.wallet_scanner.whale_convergence_min_edge == 0.05

    def test_whale_convergence_above_total_fees(self) -> None:
        from src.config import load_config
        config = load_config()
        total_fees = config.risk.transaction_fee_pct + config.risk.exit_fee_pct
        assert config.wallet_scanner.whale_convergence_min_edge > total_fees

    def test_whale_convergence_below_standard_min_edge(self) -> None:
        """Still a meaningful relaxation from standard min_edge."""
        from src.config import load_config
        config = load_config()
        assert config.wallet_scanner.whale_convergence_min_edge < config.risk.min_edge


# ── Bug 2: evidence_tracking_enabled wired ────────────────────────


class TestEvidenceTrackingWire:
    """evidence_tracking_enabled is now read by exit_finalizer."""

    def test_exit_finalizer_has_evidence_quality_method(self) -> None:
        """ExitFinalizer has _record_evidence_quality method."""
        from src.execution.exit_finalizer import ExitFinalizer
        assert hasattr(ExitFinalizer, "_record_evidence_quality")

    def test_record_calibration_checks_evidence_tracking(self) -> None:
        """_record_calibration source references evidence_tracking_enabled."""
        from src.execution.exit_finalizer import ExitFinalizer
        source = inspect.getsource(ExitFinalizer._record_calibration)
        assert "evidence_tracking_enabled" in source

    def test_record_evidence_quality_extracts_domains(self) -> None:
        """_record_evidence_quality extracts domains from research_evidence_json."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE evidence_source_quality (
                domain TEXT PRIMARY KEY,
                times_cited INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                correct_forecast_rate REAL DEFAULT 0.0,
                avg_evidence_quality REAL DEFAULT 0.0,
                avg_authority REAL DEFAULT 0.0,
                quality_trend TEXT DEFAULT 'stable',
                effective_weight REAL DEFAULT 1.0,
                last_updated TEXT DEFAULT ''
            )
        """)
        db._conn = conn

        config = MagicMock()
        config.continuous_learning.evidence_tracking_enabled = True

        finalizer = ExitFinalizer.__new__(ExitFinalizer)
        finalizer._db = db
        finalizer._config = config
        finalizer._alert_manager = None

        # Build a mock forecast with research_evidence_json
        fc = MagicMock()
        fc.research_evidence_json = json.dumps({
            "evidence": [
                {"citation": {"url": "https://www.reuters.com/article/test",
                              "publisher": "Reuters", "date": "", "title": "Test"},
                 "text": "Evidence text", "confidence": 0.85},
                {"citation": {"url": "https://fred.stlouisfed.org/series/FEDFUNDS",
                              "publisher": "FRED", "date": "", "title": "Fed Funds"},
                 "text": "Rate data", "confidence": 0.90},
            ],
        })
        fc.evidence_quality = 0.75

        finalizer._record_evidence_quality(fc, 1.0, 0.65)

        # Check domains were recorded
        rows = conn.execute(
            "SELECT domain, times_cited, times_correct FROM evidence_source_quality"
        ).fetchall()
        domains = {r["domain"]: dict(r) for r in rows}
        assert "reuters.com" in domains
        assert "fred.stlouisfed.org" in domains
        assert domains["reuters.com"]["times_cited"] == 1
        assert domains["reuters.com"]["times_correct"] == 1  # 0.65 > 0.5, outcome=1.0

        conn.close()

    def test_record_evidence_quality_incorrect_forecast(self) -> None:
        """Incorrect forecast marks domains as not correct."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE evidence_source_quality (
                domain TEXT PRIMARY KEY,
                times_cited INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                correct_forecast_rate REAL DEFAULT 0.0,
                avg_evidence_quality REAL DEFAULT 0.0,
                avg_authority REAL DEFAULT 0.0,
                quality_trend TEXT DEFAULT 'stable',
                effective_weight REAL DEFAULT 1.0,
                last_updated TEXT DEFAULT ''
            )
        """)
        db._conn = conn

        finalizer = ExitFinalizer.__new__(ExitFinalizer)
        finalizer._db = db
        finalizer._config = MagicMock()
        finalizer._alert_manager = None

        fc = MagicMock()
        fc.research_evidence_json = json.dumps({
            "evidence": [
                {"citation": {"url": "https://bbc.com/news/test",
                              "publisher": "BBC", "date": "", "title": "Test"},
                 "text": "Evidence", "confidence": 0.8},
            ],
        })
        fc.evidence_quality = 0.6

        # forecast_prob=0.7 but outcome=0.0 → incorrect
        finalizer._record_evidence_quality(fc, 0.0, 0.7)

        row = conn.execute(
            "SELECT times_correct FROM evidence_source_quality WHERE domain = 'bbc.com'"
        ).fetchone()
        assert row["times_correct"] == 0

        conn.close()

    def test_record_evidence_quality_deduplicates_domains(self) -> None:
        """Same domain cited twice is only recorded once per resolution."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE evidence_source_quality (
                domain TEXT PRIMARY KEY,
                times_cited INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                correct_forecast_rate REAL DEFAULT 0.0,
                avg_evidence_quality REAL DEFAULT 0.0,
                avg_authority REAL DEFAULT 0.0,
                quality_trend TEXT DEFAULT 'stable',
                effective_weight REAL DEFAULT 1.0,
                last_updated TEXT DEFAULT ''
            )
        """)
        db._conn = conn

        finalizer = ExitFinalizer.__new__(ExitFinalizer)
        finalizer._db = db
        finalizer._config = MagicMock()
        finalizer._alert_manager = None

        fc = MagicMock()
        fc.research_evidence_json = json.dumps({
            "evidence": [
                {"citation": {"url": "https://reuters.com/a", "publisher": "R",
                              "date": "", "title": "A"}, "text": "A", "confidence": 0.8},
                {"citation": {"url": "https://reuters.com/b", "publisher": "R",
                              "date": "", "title": "B"}, "text": "B", "confidence": 0.9},
            ],
        })
        fc.evidence_quality = 0.7

        finalizer._record_evidence_quality(fc, 1.0, 0.6)

        row = conn.execute(
            "SELECT times_cited FROM evidence_source_quality WHERE domain = 'reuters.com'"
        ).fetchone()
        assert row["times_cited"] == 1  # not 2

        conn.close()


# ── Bug 3: FRED 'pmi' no longer maps to MANEMP ───────────────────


class TestFredPmiKeywordFix:
    """FRED 'pmi' keyword removed from MANEMP mapping."""

    def test_pmi_query_does_not_return_manemp(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will US manufacturing PMI exceed 50?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "MANEMP" not in series_ids

    def test_manufacturing_employment_query_returns_manemp(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("Will manufacturing employment increase?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "MANEMP" in series_ids

    def test_factory_workers_query_returns_manemp(self) -> None:
        from src.research.connectors.fred import FredConnector
        c = FredConnector(config=None)
        matched = c._match_series("How many factory workers are employed?", max_series=5)
        series_ids = [sid for sid, _ in matched]
        assert "MANEMP" in series_ids


# ── Gap 1: MACRO base rate upside/threshold patterns ──────────────


class TestMacroUpsidePatterns:
    """MACRO patterns now cover upside direction (CPI exceed, commodity, indicator)."""

    def test_cpi_exceed_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will CPI exceed 3% in April 2026?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.40

    def test_cpi_rise_above_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will CPI rise above 4%?", "MACRO")
        assert match is not None

    def test_oil_price_exceed_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will oil prices exceed $90 per barrel?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_crude_wti_above_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will WTI crude reach above $100?", "MACRO")
        assert match is not None

    def test_gdp_growth_exceed_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will GDP growth exceed 2% in Q2?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.45

    def test_jobless_claims_exceed_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will initial jobless claims exceed 250k?", "MACRO")
        assert match is not None

    def test_retail_sales_increase_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will retail sales increase above 2% in March?", "MACRO")
        assert match is not None

    def test_macro_pattern_count(self) -> None:
        """MACRO now has 15 patterns (12 original + 3 new)."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        macro = [p for p in registry.patterns if p.category == "MACRO"]
        assert len(macro) == 15


# ── Gap 2: GEOPOLITICS base rate patterns ─────────────────────────


class TestGeopoliticsPatterns:
    """GEOPOLITICS now has 4 dedicated base rate patterns."""

    def test_military_strike_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will Iran launch a military strike against Israel?", "GEOPOLITICS")
        assert match is not None
        assert match.base_rate == 0.20
        assert match.category == "GEOPOLITICS"

    def test_nuclear_test_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will North Korea conduct a nuclear test?", "GEOPOLITICS")
        assert match is not None
        assert match.base_rate == 0.10

    def test_invasion_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will Russia invade another neighboring country?", "GEOPOLITICS")
        assert match is not None
        assert match.base_rate == 0.08

    def test_troop_withdrawal_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will US troops withdraw from Syria?", "GEOPOLITICS")
        assert match is not None
        assert match.base_rate == 0.25

    def test_geopolitics_pattern_count(self) -> None:
        """GEOPOLITICS has 4 dedicated patterns."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        geo = [p for p in registry.patterns if p.category == "GEOPOLITICS"]
        assert len(geo) == 4

    def test_geopolitics_fallback_rate(self) -> None:
        """GEOPOLITICS has a category fallback base rate."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        rate = registry.get_category_base_rate("GEOPOLITICS")
        assert rate == 0.35


# ── Gap 3: TECH release/leaderboard patterns ──────────────────────


class TestTechPatterns:
    """TECH now covers release deadlines and AI leaderboard questions."""

    def test_product_release_by_date_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will OpenAI release GPT-5 by June 2026?", "TECHNOLOGY")
        assert match is not None
        assert match.base_rate == 0.45

    def test_ship_before_q4_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will Apple launch the Vision Pro 2 before Q4?", "TECHNOLOGY")
        assert match is not None

    def test_ai_model_leaderboard_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will OpenAI have the #1 AI model at the end of April?", "TECHNOLOGY")
        assert match is not None
        assert match.base_rate == 0.30

    def test_top_ai_benchmark_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will Claude lead the AI benchmark leaderboard?", "TECHNOLOGY")
        assert match is not None

    def test_tech_pattern_count(self) -> None:
        """TECHNOLOGY now has 8 patterns (6 original + 2 new)."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        tech = [p for p in registry.patterns if p.category == "TECHNOLOGY"]
        assert len(tech) == 8


# ── Total pattern count ───────────────────────────────────────────


class TestTotalPatternCount:
    """Registry now has 83 patterns (74 + 3 MACRO + 4 GEO + 2 TECH)."""

    def test_total_count(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        assert registry.pattern_count == 83
