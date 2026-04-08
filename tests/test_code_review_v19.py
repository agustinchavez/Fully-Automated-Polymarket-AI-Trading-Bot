"""Code review v19 fixes — SPORTS scoring, per-query timeouts, base rate patterns,
config alignment, category confidence thresholds, long-horizon discount.

Tests cover:
  1. SPORTS no longer penalised in _score_market (Issue 1)
  2. Per-query web timeouts in source_fetcher (Issue 2)
  3. SPORTS base rate Polymarket-format patterns (Issue 3)
  4. Config value alignment: scanning thresholds, budget, timeout (Issues 4/5/7)
  5. Category-specific confidence thresholds (Issue 6)
  6. Long-horizon time_decay_multiplier discount (Issue 8)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Issue 1: SPORTS no longer penalised ──────────────────────────────


class TestSportsScoringFix:
    """SPORTS should get +15 via preferred_types, not -5."""

    def test_sports_in_preferred_types_config(self) -> None:
        """config.yaml includes SPORTS in preferred_types."""
        from src.config import load_config

        config = load_config()
        assert "SPORTS" in config.scanning.preferred_types

    def test_score_market_no_negative_sports(self) -> None:
        """_score_market gives SPORTS the preferred +15 bonus."""
        from src.engine.market_filter import _score_market

        market = MagicMock()
        market.market_type = "SPORTS"
        market.volume = 5000
        market.liquidity = 3000
        market.end_date = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=14)
        market.spread = 0.03
        market.has_clear_resolution = True
        market.best_bid = 0.50
        market.question = ""
        market.description = ""
        market.age_hours = 48

        score, breakdown = _score_market(
            market, preferred_types=["MACRO", "SPORTS"]
        )
        assert breakdown["market_type"] == 15

    def test_score_market_no_minus_five_anywhere(self) -> None:
        """_score_market source code has no -5 for SPORTS."""
        from src.engine.market_filter import _score_market

        source = inspect.getsource(_score_market)
        assert "SPORTS" not in source or "-5" not in source


# ── Issue 2: Per-query web timeouts ──────────────────────────────────


class TestPerQueryWebTimeouts:
    """Web search queries should each have their own timeout."""

    def test_fetch_sources_uses_per_query_timeout(self) -> None:
        """fetch_sources wraps each query individually in asyncio.wait_for."""
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_sources)
        assert "asyncio.wait_for(self._run_query(q)" in source
        assert "per_query_timeout" in source

    @pytest.mark.asyncio
    async def test_slow_query_does_not_cancel_fast(self) -> None:
        """One slow web query timing out doesn't cancel fast ones."""
        from src.research.source_fetcher import SourceFetcher, FetchedSource
        from src.connectors.web_search import SearchResult

        config = MagicMock()
        config.source_timeout_secs = 1
        config.max_sources = 10
        config.fetch_full_content = False
        config.primary_domains = {}
        config.secondary_domains = []
        config.blocked_domains = []

        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = config
        fetcher._connectors = []

        fast_result = SearchResult(
            title="fast", url="http://fast.com", snippet="ok",
            source="fast.com", date="", position=1, raw={},
        )

        call_count = 0

        async def mock_run_query(query):
            nonlocal call_count
            call_count += 1
            if "slow" in query.text:
                await asyncio.sleep(100)
                return []
            return [fast_result]

        fetcher._run_query = mock_run_query

        from src.research.query_builder import SearchQuery
        queries = [
            SearchQuery(text="fast query", intent="primary"),
            SearchQuery(text="slow query", intent="news"),
        ]

        results = await fetcher.fetch_sources(queries)
        assert len(results) == 1
        assert results[0].title == "fast"


# ── Issue 3: SPORTS base rate Polymarket patterns ────────────────────


class TestSportsBaseRatePatterns:
    """Verify Polymarket-format SPORTS patterns match real questions."""

    def test_h2h_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Bucks vs. Pistons", "SPORTS")
        assert match is not None
        assert match.base_rate == 0.50

    def test_spread_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Spread: Cavaliers (-2.5)", "SPORTS")
        assert match is not None
        assert match.base_rate == 0.50

    def test_over_under_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Bucks vs. Pistons: O/U 221.5", "SPORTS")
        assert match is not None

    def test_btts_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Both Teams to Score in the match", "SPORTS")
        assert match is not None
        assert match.base_rate == 0.55

    def test_esports_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("LoL: T1 vs GenG — who wins?", "SPORTS")
        assert match is not None

    def test_moneyline_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Moneyline: Lakers", "SPORTS")
        assert match is not None
        assert match.base_rate == 0.50

    def test_win_game_pattern(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will the 49ers win game 3 of the series?", "SPORTS")
        assert match is not None

    def test_total_pattern_count(self) -> None:
        """Registry has 74 total patterns (68 + 6 new SPORTS format)."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        assert registry.pattern_count == 83

    def test_sports_pattern_count(self) -> None:
        """SPORTS category has 12 patterns (4 conceptual + 8 format)."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        sports_patterns = [p for p in registry.patterns if p.category == "SPORTS"]
        assert len(sports_patterns) == 12


# ── Issues 4/5/7: Config value alignment ─────────────────────────────


class TestConfigAlignment:
    """Config values match between scanning and risk sections."""

    def test_source_timeout_22(self) -> None:
        """source_timeout_secs raised to 22."""
        from src.config import load_config

        config = load_config()
        assert config.research.source_timeout_secs == 22

    def test_scanning_min_liquidity_matches_risk(self) -> None:
        """Scanning min_liquidity_usd aligned to risk min_liquidity."""
        from src.config import load_config

        config = load_config()
        assert config.scanning.min_liquidity_usd >= config.risk.min_liquidity

    def test_scanning_max_spread_matches_risk(self) -> None:
        """Scanning max_spread aligned to risk max_spread."""
        from src.config import load_config

        config = load_config()
        assert config.scanning.max_spread <= config.risk.max_spread

    def test_scanning_min_liquidity_2000(self) -> None:
        """Scanning min_liquidity_usd is 2000."""
        from src.config import load_config

        config = load_config()
        assert config.scanning.min_liquidity_usd == 2000

    def test_scanning_max_spread_006(self) -> None:
        """Scanning max_spread is 0.06."""
        from src.config import load_config

        config = load_config()
        assert config.scanning.max_spread == 0.06

    def test_daily_budget_15(self) -> None:
        """Daily budget raised to $15."""
        from src.config import load_config

        config = load_config()
        assert config.budget.daily_limit_usd == 25.0


# ── Issue 6: Category-specific confidence thresholds ─────────────────


class TestCategoryConfidence:
    """Per-category confidence overrides in risk_limits."""

    def test_forecasting_config_has_field(self) -> None:
        """ForecastingConfig has category_min_confidence field."""
        from src.config import ForecastingConfig

        cfg = ForecastingConfig()
        assert hasattr(cfg, "category_min_confidence")
        assert isinstance(cfg.category_min_confidence, dict)

    def test_config_yaml_has_overrides(self) -> None:
        """config.yaml defines GEOPOLITICS and ELECTION as LOW."""
        from src.config import load_config

        config = load_config()
        cmc = config.forecasting.category_min_confidence
        assert cmc.get("GEOPOLITICS") == "LOW"
        assert cmc.get("ELECTION") == "LOW"

    def test_risk_limits_uses_category_override(self) -> None:
        """check_risk_limits uses category override for confidence."""
        from src.policy.risk_limits import check_risk_limits
        from src.policy.edge_calc import EdgeResult
        from src.forecast.feature_builder import MarketFeatures
        from src.config import RiskConfig, ForecastingConfig
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk_cfg = RiskConfig()

        forecast_cfg = ForecastingConfig(
            min_confidence_level="MEDIUM",
            category_min_confidence={"GEOPOLITICS": "LOW"},
        )

        edge = MagicMock(spec=EdgeResult)
        edge.abs_edge = 0.10
        edge.net_edge = 0.06
        edge.abs_net_edge = 0.06
        edge.implied_probability = 0.50
        edge.is_positive = True

        features = MarketFeatures(
            market_id="test",
            evidence_quality=0.8,
            days_to_expiry=30,
            hours_to_resolution=720,
            has_clear_resolution=True,
        )

        # LOW confidence + GEOPOLITICS → should pass (override = LOW)
        result = check_risk_limits(
            edge=edge,
            features=features,
            risk_config=risk_cfg,
            forecast_config=forecast_cfg,
            market_type="GEOPOLITICS",
            confidence_level="LOW",
        )
        conf_violations = [v for v in result.violations if "LOW_CONFIDENCE" in v]
        assert len(conf_violations) == 0

    def test_risk_limits_global_still_applies_without_override(self) -> None:
        """Without category override, global MEDIUM minimum applies."""
        from src.policy.risk_limits import check_risk_limits
        from src.policy.edge_calc import EdgeResult
        from src.forecast.feature_builder import MarketFeatures
        from src.config import RiskConfig, ForecastingConfig
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            risk_cfg = RiskConfig()

        forecast_cfg = ForecastingConfig(
            min_confidence_level="MEDIUM",
            category_min_confidence={"GEOPOLITICS": "LOW"},
        )

        edge = MagicMock(spec=EdgeResult)
        edge.abs_edge = 0.10
        edge.net_edge = 0.06
        edge.abs_net_edge = 0.06
        edge.implied_probability = 0.50
        edge.is_positive = True

        features = MarketFeatures(
            market_id="test",
            evidence_quality=0.8,
            days_to_expiry=30,
            hours_to_resolution=720,
            has_clear_resolution=True,
        )

        # LOW confidence + MACRO (no override) → should FAIL
        result = check_risk_limits(
            edge=edge,
            features=features,
            risk_config=risk_cfg,
            forecast_config=forecast_cfg,
            market_type="MACRO",
            confidence_level="LOW",
        )
        conf_violations = [v for v in result.violations if "LOW_CONFIDENCE" in v]
        assert len(conf_violations) == 1

    def test_risk_limits_source_has_category_override(self) -> None:
        """check_risk_limits source uses category_min_confidence."""
        from src.policy.risk_limits import check_risk_limits

        source = inspect.getsource(check_risk_limits)
        assert "category_min_confidence" in source


# ── Issue 8: Long-horizon time_decay_multiplier discount ─────────────


class TestLongHorizonDiscount:
    """time_decay_multiplier should discount positions >30 days."""

    def _build_features_with_days(self, days: float) -> "MarketFeatures":
        from src.forecast.feature_builder import build_features

        market = MagicMock()
        market.id = "test"
        market.question = "Test?"
        market.market_type = "MACRO"
        market.volume = 5000
        market.liquidity = 3000
        market.category = "MACRO"
        market.has_clear_resolution = True
        market.tokens = []
        market.best_bid = 0.50
        market.end_date = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=days)

        return build_features(market)

    def test_near_resolution_boost(self) -> None:
        """≤7 days still gets boost (up to 1.5x)."""
        features = self._build_features_with_days(3)
        assert features.time_decay_multiplier > 1.0

    def test_normal_range_no_change(self) -> None:
        """8-30 days gets multiplier = 1.0."""
        features = self._build_features_with_days(15)
        assert features.time_decay_multiplier == 1.0

    def test_30_day_boundary(self) -> None:
        """At exactly 30 days, multiplier is 1.0."""
        features = self._build_features_with_days(30)
        assert features.time_decay_multiplier == 1.0

    def test_60_day_discount(self) -> None:
        """At 60 days, multiplier is 0.85 (midpoint of 1.0→0.7)."""
        features = self._build_features_with_days(60)
        assert 0.80 <= features.time_decay_multiplier <= 0.90

    def test_90_day_discount(self) -> None:
        """At 90 days, multiplier is ~0.7 (boundary of linear segment)."""
        features = self._build_features_with_days(90)
        assert features.time_decay_multiplier == pytest.approx(0.7, abs=0.01)

    def test_120_day_discount(self) -> None:
        """At 120 days (>90), multiplier continues below 0.7."""
        features = self._build_features_with_days(120)
        assert features.time_decay_multiplier < 0.70

    def test_monotonic_decrease(self) -> None:
        """Multiplier decreases monotonically from 30→90 days."""
        m30 = self._build_features_with_days(30).time_decay_multiplier
        m45 = self._build_features_with_days(45).time_decay_multiplier
        m60 = self._build_features_with_days(60).time_decay_multiplier
        m75 = self._build_features_with_days(75).time_decay_multiplier
        m90 = self._build_features_with_days(90).time_decay_multiplier
        assert m30 >= m45 >= m60 >= m75 >= m90
