"""Code review v18 fixes — per-connector timeouts, category cooldowns,
config sync, prompt v2, stake multipliers, base rate coverage.

Tests cover:
  1. Per-connector timeouts in source_fetcher (Issue 2)
  2. Category-specific research cooldowns (Issue 3)
  3. use_probability_space_costs default sync (Issue 4)
  4. Prompt v2 config switch (Improvement 1)
  5. Updated category_stake_multipliers (Improvement 2)
  6. Base rate coverage for CRYPTO, WEATHER, SCIENCE (Improvement 3)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# ── Issue 2: Per-connector timeouts ────────────────────────────────


class TestPerConnectorTimeouts:
    """Verify each connector gets its own timeout instead of one shared."""

    def test_source_fetcher_no_outer_wait_for(self) -> None:
        """fetch_structured_sources wraps each connector individually."""
        import inspect
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_structured_sources)
        # Should have per-connector asyncio.wait_for inside list comp
        assert "asyncio.wait_for(" in source
        # Should NOT have the old pattern of wrapping gather in wait_for
        assert "asyncio.wait_for(\n                asyncio.gather" not in source

    def test_per_connector_timeout_uses_config(self) -> None:
        """Per-connector timeout reads source_timeout_secs from config."""
        import inspect
        from src.research.source_fetcher import SourceFetcher

        source = inspect.getsource(SourceFetcher.fetch_structured_sources)
        assert "source_timeout_secs" in source

    @pytest.mark.asyncio
    async def test_slow_connector_does_not_cancel_fast(self) -> None:
        """One slow connector timing out doesn't cancel fast ones."""
        from src.research.source_fetcher import SourceFetcher, FetchedSource

        config = MagicMock()
        config.source_timeout_secs = 1  # 1 second timeout

        provider = MagicMock()
        fetcher = SourceFetcher.__new__(SourceFetcher)
        fetcher._config = config

        # Fast connector returns instantly
        fast_source = FetchedSource(title="fast", url="http://fast.com", snippet="ok")

        async def fast_fetch(q, t):
            return [fast_source]

        # Slow connector sleeps forever
        async def slow_fetch(q, t):
            await asyncio.sleep(100)
            return []

        fast_conn = MagicMock()
        fast_conn.name = "fast"
        fast_conn.fetch = fast_fetch
        fast_conn.relevant_categories.return_value = {"MACRO"}
        fast_conn.is_relevant.return_value = True

        slow_conn = MagicMock()
        slow_conn.name = "slow"
        slow_conn.fetch = slow_fetch
        slow_conn.relevant_categories.return_value = {"MACRO"}
        slow_conn.is_relevant.return_value = True

        fetcher._connectors = [fast_conn, slow_conn]

        results = await fetcher.fetch_structured_sources("test?", "MACRO")
        # Fast connector's result should be present
        assert len(results) == 1
        assert results[0].title == "fast"


# ── Issue 3: Category-specific cooldowns ───────────────────────────


class TestCategoryCooldowns:
    """Verify ResearchCache supports per-category cooldown overrides."""

    def test_default_cooldown_unchanged(self) -> None:
        """Without category overrides, default cooldown applies."""
        from src.engine.market_filter import ResearchCache

        cache = ResearchCache(cooldown_minutes=60)
        cache.mark_researched("m1")
        assert cache.was_recently_researched("m1")
        assert cache.was_recently_researched("m1", category="MACRO")

    def test_category_override_shorter(self) -> None:
        """SPORTS category with 20-min cooldown expires before 60-min default."""
        from src.engine.market_filter import ResearchCache

        cache = ResearchCache(
            cooldown_minutes=60,
            category_cooldown_minutes={"SPORTS": 0},  # 0 min = immediately expired
        )
        cache.mark_researched("m1")
        # SPORTS cooldown is 0 minutes — should NOT be recently researched
        assert not cache.was_recently_researched("m1", category="SPORTS")
        # Default cooldown still applies for other categories
        assert cache.was_recently_researched("m1", category="MACRO")
        assert cache.was_recently_researched("m1")

    def test_category_override_longer(self) -> None:
        """GEOPOLITICS with longer cooldown stays cached longer."""
        from src.engine.market_filter import ResearchCache

        cache = ResearchCache(
            cooldown_minutes=60,
            category_cooldown_minutes={"GEOPOLITICS": 120},
        )
        cache.mark_researched("m1")
        # Both should be recently researched (we just marked it)
        assert cache.was_recently_researched("m1", category="GEOPOLITICS")
        assert cache.was_recently_researched("m1", category="MACRO")

    def test_config_has_category_cooldown_field(self) -> None:
        """ScanningConfig has category_cooldown_minutes field."""
        from src.config import ScanningConfig

        cfg = ScanningConfig()
        assert hasattr(cfg, "category_cooldown_minutes")
        assert isinstance(cfg.category_cooldown_minutes, dict)

    def test_config_yaml_has_category_cooldowns(self) -> None:
        """config.yaml defines category-specific cooldowns."""
        from src.config import load_config

        config = load_config()
        ccm = config.scanning.category_cooldown_minutes
        assert ccm.get("SPORTS") == 20
        assert ccm.get("GEOPOLITICS") == 120

    def test_filter_markets_passes_category(self) -> None:
        """filter_markets passes category to was_recently_researched."""
        import inspect
        from src.engine.market_filter import filter_markets

        source = inspect.getsource(filter_markets)
        assert "category=" in source


# ── Issue 4: use_probability_space_costs default sync ──────────────


class TestProbSpaceCostsDefault:
    """Verify Python default matches config.yaml value."""

    def test_default_is_true(self) -> None:
        """RiskConfig default for use_probability_space_costs is True."""
        from src.config import RiskConfig
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = RiskConfig()
        assert cfg.use_probability_space_costs is True

    def test_config_yaml_matches(self) -> None:
        """config.yaml also has use_probability_space_costs: true."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        content = config_path.read_text()
        assert "use_probability_space_costs: true" in content


# ── Improvement 1: Prompt v2 ───────────────────────────────────────


class TestPromptV2Config:
    """Verify prompt v2 is enabled in config."""

    def test_config_yaml_prompt_v2(self) -> None:
        """config.yaml sets prompt_version: v2."""
        from src.config import load_config

        config = load_config()
        assert config.forecasting.prompt_version == "v2"

    def test_config_yaml_base_rate_enabled(self) -> None:
        """config.yaml sets base_rate_enabled: true."""
        from src.config import load_config

        config = load_config()
        assert config.forecasting.base_rate_enabled is True


# ── Improvement 2: Category stake multipliers ─────────────────────


class TestStakeMultipliers:
    """Verify updated category stake multipliers in config."""

    def test_sports_multiplier(self) -> None:
        from src.config import load_config

        config = load_config()
        m = config.risk.category_stake_multipliers
        assert m.get("SPORTS") == 1.3

    def test_geopolitics_multiplier(self) -> None:
        from src.config import load_config

        config = load_config()
        m = config.risk.category_stake_multipliers
        assert m.get("GEOPOLITICS") == 0.6

    def test_crypto_multiplier(self) -> None:
        from src.config import load_config

        config = load_config()
        m = config.risk.category_stake_multipliers
        assert m.get("CRYPTO") == 0.7

    def test_science_multiplier(self) -> None:
        from src.config import load_config

        config = load_config()
        m = config.risk.category_stake_multipliers
        assert m.get("SCIENCE") == 1.1


# ── Improvement 3: Base rate coverage ──────────────────────────────


class TestBaseRateCoverage:
    """Verify new base rate patterns for CRYPTO, WEATHER, SCIENCE."""

    def test_crypto_patterns_exist(self) -> None:
        """CRYPTO category has base rate patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        crypto_patterns = [p for p in registry.patterns if p.category == "CRYPTO"]
        assert len(crypto_patterns) >= 4

    def test_weather_patterns_exist(self) -> None:
        """WEATHER category has base rate patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        weather_patterns = [p for p in registry.patterns if p.category == "WEATHER"]
        assert len(weather_patterns) >= 4

    def test_science_patterns_exist(self) -> None:
        """SCIENCE category has base rate patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        science_patterns = [p for p in registry.patterns if p.category == "SCIENCE"]
        assert len(science_patterns) >= 4

    def test_crypto_ath_match(self) -> None:
        """CRYPTO ATH pattern matches correctly."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will Bitcoin reach a new all-time high by June?", "CRYPTO")
        assert match is not None
        assert match.category == "CRYPTO"
        assert match.base_rate == 0.25

    def test_weather_hurricane_match(self) -> None:
        """WEATHER hurricane pattern matches correctly."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will the hurricane reach category 4 intensity?", "WEATHER")
        assert match is not None
        assert match.category == "WEATHER"

    def test_science_fda_match(self) -> None:
        """SCIENCE FDA approval pattern matches correctly."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will the FDA approve the new cancer drug treatment?", "SCIENCE")
        assert match is not None
        assert match.category == "SCIENCE"

    def test_total_pattern_count(self) -> None:
        """Registry now has 73 seed patterns (68 + 5 new SPORTS format)."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        assert registry.pattern_count == 73

    def test_crypto_category_base_rate(self) -> None:
        """CRYPTO has a category-level fallback base rate."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        rate = registry.get_category_base_rate("CRYPTO")
        assert rate == 0.40

    def test_crypto_price_target_match(self) -> None:
        """CRYPTO price target pattern matches."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will Ethereum hit $5000 by end of year?", "CRYPTO")
        assert match is not None
        assert match.category == "CRYPTO"

    def test_science_clinical_trial_match(self) -> None:
        """SCIENCE clinical trial pattern matches."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will the phase 3 clinical trial succeed and hit primary endpoint?", "SCIENCE")
        assert match is not None
        assert match.category == "SCIENCE"

    def test_weather_enso_match(self) -> None:
        """WEATHER ENSO pattern matches."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will El Niño develop by September 2025?", "WEATHER")
        assert match is not None
        assert match.category == "WEATHER"
