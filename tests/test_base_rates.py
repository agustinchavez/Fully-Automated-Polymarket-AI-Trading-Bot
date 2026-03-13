"""Tests for base rate registry (Phase 2 — Batch A)."""

from __future__ import annotations

import pytest

from src.forecast.base_rates import (
    BaseRateMatch,
    BaseRatePattern,
    BaseRateRegistry,
    _CATEGORY_BASE_RATES,
    _SEED_PATTERNS,
)


@pytest.fixture
def registry() -> BaseRateRegistry:
    return BaseRateRegistry()


# ── Seed pattern tests ───────────────────────────────────────────────


class TestSeedPatterns:

    def test_seed_patterns_count(self) -> None:
        """At least 50 seed patterns are defined."""
        assert len(_SEED_PATTERNS) >= 50

    def test_all_categories_represented(self) -> None:
        """All major categories have at least one pattern."""
        categories = {p["category"] for p in _SEED_PATTERNS}
        expected = {"MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY", "SPORTS", "GENERAL"}
        assert expected.issubset(categories)

    def test_pattern_fields_complete(self) -> None:
        """Every pattern has all required fields."""
        required = {"pattern", "category", "description", "base_rate", "source"}
        for i, p in enumerate(_SEED_PATTERNS):
            for field in required:
                assert field in p, f"Pattern {i} missing field '{field}'"
            assert 0.0 < p["base_rate"] < 1.0, f"Pattern {i} base_rate out of (0,1)"

    def test_patterns_compile(self) -> None:
        """All regex patterns compile without errors."""
        for i, p in enumerate(_SEED_PATTERNS):
            pat = BaseRatePattern(**p)
            compiled = pat.compile()
            assert compiled is not None, f"Pattern {i} failed to compile"

    def test_category_base_rates_exist(self) -> None:
        """Category-level fallback rates exist for key categories."""
        assert "MACRO" in _CATEGORY_BASE_RATES
        assert "ELECTION" in _CATEGORY_BASE_RATES
        assert "CORPORATE" in _CATEGORY_BASE_RATES
        assert "UNKNOWN" in _CATEGORY_BASE_RATES


# ── Registry initialization ─────────────────────────────────────────


class TestRegistryInit:

    def test_patterns_loaded(self, registry: BaseRateRegistry) -> None:
        """Registry loads all seed patterns on init."""
        assert registry.pattern_count >= 50

    def test_patterns_property(self, registry: BaseRateRegistry) -> None:
        """patterns property returns a copy."""
        patterns = registry.patterns
        assert len(patterns) == registry.pattern_count
        # Modifying the list shouldn't affect the registry
        patterns.clear()
        assert registry.pattern_count >= 50


# ── Pattern matching ─────────────────────────────────────────────────


class TestMatchPatterns:

    def test_fed_rate_cut(self, registry: BaseRateRegistry) -> None:
        """Matches Fed rate cut question."""
        m = registry.match("Will the Federal Reserve cut interest rates in June?")
        assert m is not None
        assert m.base_rate == 0.25
        assert "Fed" in m.pattern_description or "rate" in m.pattern_description.lower()

    def test_fed_rate_hike(self, registry: BaseRateRegistry) -> None:
        """Matches Fed rate hike question."""
        m = registry.match("Will the Fed raise rates at the March meeting?")
        assert m is not None
        assert m.base_rate == 0.30

    def test_earnings_beat(self, registry: BaseRateRegistry) -> None:
        """Matches earnings beat question."""
        m = registry.match("Will Apple's earnings beat analyst estimates in Q4?")
        assert m is not None
        assert m.base_rate == 0.70

    def test_merger_completion(self, registry: BaseRateRegistry) -> None:
        """Matches merger completion question."""
        m = registry.match("Will the Microsoft-Activision merger deal be completed?")
        assert m is not None
        assert m.base_rate == 0.85

    def test_incumbent_wins(self, registry: BaseRateRegistry) -> None:
        """Matches incumbent reelection question."""
        m = registry.match("Will the incumbent president win reelection?")
        assert m is not None
        assert m.base_rate == 0.60

    def test_conviction(self, registry: BaseRateRegistry) -> None:
        """Matches criminal conviction question."""
        m = registry.match("Will the defendant be found guilty?")
        assert m is not None
        assert m.base_rate == 0.90

    def test_crypto_price(self, registry: BaseRateRegistry) -> None:
        """Matches crypto price target question."""
        m = registry.match("Will Bitcoin reach above $100,000 by December?")
        assert m is not None
        assert 0.0 < m.base_rate < 1.0

    def test_home_team_wins(self, registry: BaseRateRegistry) -> None:
        """Matches sports home team question."""
        m = registry.match("Will the home team win the game?")
        assert m is not None
        assert m.base_rate == 0.55

    def test_category_bonus(self, registry: BaseRateRegistry) -> None:
        """Category hint improves match confidence."""
        m_no_cat = registry.match("Will the Fed cut rates?")
        m_with_cat = registry.match("Will the Fed cut rates?", category="MACRO")
        assert m_no_cat is not None
        assert m_with_cat is not None
        assert m_with_cat.confidence >= m_no_cat.confidence

    def test_no_match(self, registry: BaseRateRegistry) -> None:
        """Returns None for unrecognizable questions."""
        m = registry.match("How many jellybeans are in the jar?")
        assert m is None

    def test_empty_question(self, registry: BaseRateRegistry) -> None:
        """Returns None for empty question."""
        assert registry.match("") is None

    def test_match_confidence_range(self, registry: BaseRateRegistry) -> None:
        """Confidence is always in [0, 1]."""
        m = registry.match("Will the Federal Reserve cut rates?")
        assert m is not None
        assert 0.0 <= m.confidence <= 1.0

    def test_match_has_source(self, registry: BaseRateRegistry) -> None:
        """Match includes source information."""
        m = registry.match("Will the Fed cut rates?")
        assert m is not None
        assert len(m.source) > 0

    def test_government_shutdown(self, registry: BaseRateRegistry) -> None:
        """Matches government shutdown question."""
        m = registry.match("Will there be a government shutdown?")
        assert m is not None
        assert m.base_rate == 0.20


# ── Category base rate fallback ──────────────────────────────────────


class TestCategoryBaseRate:

    def test_known_category(self, registry: BaseRateRegistry) -> None:
        """Returns base rate for known category."""
        rate = registry.get_category_base_rate("MACRO")
        assert rate == 0.40

    def test_election_category(self, registry: BaseRateRegistry) -> None:
        rate = registry.get_category_base_rate("ELECTION")
        assert rate == 0.50

    def test_unknown_category(self, registry: BaseRateRegistry) -> None:
        """Returns rate for UNKNOWN category."""
        rate = registry.get_category_base_rate("UNKNOWN")
        assert rate == 0.50

    def test_unrecognized_category(self, registry: BaseRateRegistry) -> None:
        """Returns None for completely unrecognized category."""
        rate = registry.get_category_base_rate("NONEXISTENT")
        assert rate is None


# ── Empirical rate updates ───────────────────────────────────────────


class TestEmpiricalUpdates:

    def test_update_basic(self, registry: BaseRateRegistry) -> None:
        """Empirical update stores rate and sample size."""
        registry.update_from_resolved("MACRO", 0.45, 50)
        rates = registry.get_empirical_rates()
        assert "MACRO" in rates
        assert rates["MACRO"]["rate"] == 0.45
        assert rates["MACRO"]["sample_size"] == 50

    def test_update_clamps_rate(self, registry: BaseRateRegistry) -> None:
        """Empirical rate is clamped to (0.01, 0.99)."""
        registry.update_from_resolved("MACRO", 1.5, 50)
        assert registry.get_empirical_rates()["MACRO"]["rate"] == 0.99
        registry.update_from_resolved("MACRO", -0.5, 50)
        assert registry.get_empirical_rates()["MACRO"]["rate"] == 0.01

    def test_empirical_overrides_category_fallback(self, registry: BaseRateRegistry) -> None:
        """Empirical rate overrides default category base rate."""
        registry.update_from_resolved("MACRO", 0.55, 100)
        rate = registry.get_category_base_rate("MACRO")
        assert rate == 0.55

    def test_empirical_needs_30_samples(self, registry: BaseRateRegistry) -> None:
        """Empirical rate with <30 samples doesn't override category rate."""
        registry.update_from_resolved("MACRO", 0.99, 10)
        rate = registry.get_category_base_rate("MACRO")
        # Should still be the default since sample_size < 30
        assert rate == 0.40

    def test_empirical_match_high_confidence(self, registry: BaseRateRegistry) -> None:
        """Empirical rate can override pattern match when confidence is higher."""
        registry.update_from_resolved("MACRO", 0.60, 200)
        # This question matches a seed pattern (Fed cut → 0.25)
        # but empirical MACRO rate has confidence 0.8 (200/100 capped) which
        # may or may not override depending on pattern match confidence
        m = registry.match("Will the Fed cut rates?", category="MACRO")
        assert m is not None
        # The pattern-specific match should still win since it's more specific
        assert m.base_rate in (0.25, 0.60)


# ── API / Dashboard helpers ──────────────────────────────────────────


class TestGetAllPatterns:

    def test_get_all_patterns(self, registry: BaseRateRegistry) -> None:
        """Returns all patterns as dicts."""
        patterns = registry.get_all_patterns()
        assert len(patterns) >= 50
        for p in patterns:
            assert "pattern" in p
            assert "category" in p
            assert "description" in p
            assert "base_rate" in p

    def test_get_empirical_rates_empty(self, registry: BaseRateRegistry) -> None:
        """Initially no empirical rates."""
        assert registry.get_empirical_rates() == {}
