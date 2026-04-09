"""Code review v23 fixes — recession 'by Q3' pattern broadening,
smart_retrain + param_optimizer config enables, v22 verification.

Tests cover:
  1. Recession pattern now matches 'by Q3', 'before end of year', etc.
  2. smart_retrain_enabled is true in loaded config
  3. param_optimizer_enabled is true in loaded config
  4. Google Trends connector already uses SerpAPI (no pytrends)
  5. No dead risk.whale_convergence_min_edge field exists
"""

from __future__ import annotations

import inspect

import pytest


# ── Fix 1: Recession 'by Q3' base rate pattern ─────────────────────


class TestRecessionPatternBroadened:
    """Recession pattern matches 'by Q3', 'before end of year', etc."""

    def test_recession_by_q3(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will the US enter a recession by Q3?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.15
        assert match.category == "MACRO"

    def test_recession_by_q1_2027(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will the economy enter a recession by Q1 2027?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.15

    def test_recession_before_end_of_year(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will we see a recession before end of year?", "MACRO")
        assert match is not None

    def test_recession_by_next_year(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will recession hit by next year?", "MACRO")
        assert match is not None

    def test_recession_within_year_still_works(self) -> None:
        """Original pattern still matches."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will there be a recession within a year?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.15

    def test_recession_within_12_months_still_works(self) -> None:
        """Original pattern still matches."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will the US enter recession within 12 months?", "MACRO")
        assert match is not None

    def test_total_pattern_count_84(self) -> None:
        """Pattern count is 84 (v24 added directional MACRO pattern)."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        assert registry.pattern_count == 84


# ── Fix 2: smart_retrain + param_optimizer enabled ──────────────────


class TestContinuousLearningEnabled:
    """smart_retrain and param_optimizer enabled in config."""

    def test_smart_retrain_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.smart_retrain_enabled is True

    def test_param_optimizer_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.param_optimizer_enabled is True

    def test_post_mortem_still_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.post_mortem_enabled is True

    def test_evidence_tracking_still_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.evidence_tracking_enabled is True

    def test_weekly_summary_still_enabled(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.continuous_learning.weekly_summary_enabled is True


# ── Verification: Google Trends uses SerpAPI ─────────────────────────


class TestGoogleTrendsSerpApi:
    """Google Trends connector already uses SerpAPI, not pytrends."""

    def test_no_pytrends_import(self) -> None:
        """google_trends.py does not import pytrends."""
        from src.research.connectors import google_trends
        source = inspect.getsource(google_trends)
        assert "pytrends" not in source

    def test_serpapi_in_source(self) -> None:
        """google_trends.py references serpapi."""
        from src.research.connectors import google_trends
        source = inspect.getsource(google_trends)
        assert "serpapi" in source.lower()

    def test_fetch_serpapi_method_exists(self) -> None:
        """Connector has _fetch_serpapi method."""
        from src.research.connectors.google_trends import GoogleTrendsConnector
        assert hasattr(GoogleTrendsConnector, "_fetch_serpapi")


# ── Verification: No dead risk.whale_convergence_min_edge ────────────


class TestNoDeadWhaleField:
    """risk section has no whale_convergence_min_edge field."""

    def test_risk_config_no_whale_field(self) -> None:
        """RiskConfig does not have whale_convergence_min_edge."""
        from src.config import RiskConfig
        assert not hasattr(RiskConfig.model_fields, "whale_convergence_min_edge") or \
            "whale_convergence_min_edge" not in RiskConfig.model_fields

    def test_wallet_scanner_has_whale_field(self) -> None:
        """WalletScannerConfig has whale_convergence_min_edge."""
        from src.config import WalletScannerConfig
        assert "whale_convergence_min_edge" in WalletScannerConfig.model_fields

    def test_wallet_scanner_value_is_005(self) -> None:
        """Loaded config reads 0.05 from config.yaml."""
        from src.config import load_config
        config = load_config()
        assert config.wallet_scanner.whale_convergence_min_edge == 0.05

    def test_pipeline_reads_wallet_scanner(self) -> None:
        """Pipeline reads whale_convergence_min_edge from wallet_scanner, not risk."""
        from src.engine.pipeline import PipelineRunner
        source = inspect.getsource(PipelineRunner)
        assert "wallet_scanner.whale_convergence_min_edge" in source
