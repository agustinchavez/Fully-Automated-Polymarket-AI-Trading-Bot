"""Code review v20 fixes — SPORTS pattern gap, VaR gate Kelly estimate,
evidence gating + cost control, extended time_decay curve.

Tests cover:
  1. 'Will [Team] win?' SPORTS base rate pattern (Issue 1)
  2. VaR gate uses preliminary Kelly estimate, not hardcoded 50 (Issue 2)
  3. evidence_model_gating enabled + max_markets_per_cycle reduced (Issue 3)
  4. Time decay continues beyond 90d to 0.30 floor (Issue 4)
"""

from __future__ import annotations

import datetime as dt
import inspect
from unittest.mock import MagicMock, patch

import pytest


# ── Issue 1: 'Will [Team] win?' SPORTS base rate pattern ────────────


class TestWillTeamWinPattern:
    """New pattern matches question-format SPORTS markets."""

    def test_will_arsenal_win(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will Arsenal FC win on 2026-04-15?", "SPORTS")
        assert match is not None
        assert match.base_rate == 0.50
        assert match.category == "SPORTS"

    def test_will_lakers_win(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will the Lakers win tonight?", "SPORTS")
        assert match is not None
        assert match.category == "SPORTS"

    def test_will_team_advance(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will Brazil advance to the World Cup final?", "SPORTS")
        assert match is not None

    def test_will_team_qualify(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will Liverpool qualify for Champions League?", "SPORTS")
        assert match is not None

    def test_total_pattern_count_74(self) -> None:
        """Registry now has 74 total patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        assert registry.pattern_count == 74

    def test_sports_pattern_count_12(self) -> None:
        """SPORTS category now has 12 patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        sports = [p for p in registry.patterns if p.category == "SPORTS"]
        assert len(sports) == 12


# ── Issue 2: VaR gate uses preliminary Kelly estimate ────────────────


class TestVarGateKellyEstimate:
    """VaR gate should use preliminary Kelly sizing, not hardcoded 50."""

    def test_no_hardcoded_50_in_stage_var_gate(self) -> None:
        """stage_var_gate no longer has size_usd=50.0 hardcoded."""
        from src.engine.pipeline import PipelineRunner

        source = inspect.getsource(PipelineRunner.stage_var_gate)
        # The old hardcoded line should be gone
        assert "size_usd=50.0" not in source

    def test_stage_var_gate_imports_position_sizer(self) -> None:
        """stage_var_gate imports calculate_position_size."""
        from src.engine.pipeline import PipelineRunner

        source = inspect.getsource(PipelineRunner.stage_var_gate)
        assert "calculate_position_size" in source

    def test_stage_var_gate_uses_estimated_size(self) -> None:
        """stage_var_gate uses estimated_size variable."""
        from src.engine.pipeline import PipelineRunner

        source = inspect.getsource(PipelineRunner.stage_var_gate)
        assert "estimated_size" in source

    def test_prelim_kelly_fallback_on_error(self) -> None:
        """If prelim Kelly fails, falls back to 50.0."""
        from src.engine.pipeline import PipelineRunner

        source = inspect.getsource(PipelineRunner.stage_var_gate)
        # Should have fallback initialisation
        assert "estimated_size = 50.0" in source
        # And exception handling
        assert "except Exception" in source


# ── Issue 3: Evidence model gating + max markets reduced ─────────────


class TestCostControlConfig:
    """evidence_model_gating enabled and max_markets_per_cycle reduced."""

    def test_evidence_model_gating_enabled(self) -> None:
        """config.yaml has evidence_model_gating_enabled: true."""
        from src.config import load_config

        config = load_config()
        assert config.ensemble.evidence_model_gating_enabled is True

    def test_max_markets_per_cycle_reduced(self) -> None:
        """max_markets_per_cycle is 5 (was 10)."""
        from src.config import load_config

        config = load_config()
        assert config.engine.max_markets_per_cycle == 5

    def test_max_markets_per_cycle_leq_5(self) -> None:
        """max_markets_per_cycle should be at most 5 for cost control."""
        from src.config import load_config

        config = load_config()
        assert config.engine.max_markets_per_cycle <= 5


# ── Issue 4: Extended time_decay beyond 90d ──────────────────────────


class TestExtendedTimeDecay:
    """time_decay_multiplier continues below 0.7 for >90d markets."""

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

    def test_91_days_just_below_70(self) -> None:
        """91 days gets slightly below 0.70."""
        features = self._build_features_with_days(91)
        assert features.time_decay_multiplier < 0.70
        assert features.time_decay_multiplier > 0.69

    def test_180_days_about_60(self) -> None:
        """180 days gets ~0.60."""
        features = self._build_features_with_days(180)
        assert 0.58 <= features.time_decay_multiplier <= 0.62

    def test_270_days_about_50(self) -> None:
        """270 days gets ~0.50."""
        features = self._build_features_with_days(270)
        assert 0.48 <= features.time_decay_multiplier <= 0.52

    def test_365_days_about_40(self) -> None:
        """365 days gets ~0.40."""
        features = self._build_features_with_days(365)
        assert 0.38 <= features.time_decay_multiplier <= 0.42

    def test_500_days_floor_30(self) -> None:
        """500+ days hits the 0.30 floor."""
        features = self._build_features_with_days(500)
        assert features.time_decay_multiplier == 0.30

    def test_monotonic_decrease_90_to_365(self) -> None:
        """Multiplier decreases monotonically from 90→365 days."""
        m90 = self._build_features_with_days(90).time_decay_multiplier
        m120 = self._build_features_with_days(120).time_decay_multiplier
        m180 = self._build_features_with_days(180).time_decay_multiplier
        m270 = self._build_features_with_days(270).time_decay_multiplier
        m365 = self._build_features_with_days(365).time_decay_multiplier
        assert m90 >= m120 >= m180 >= m270 >= m365

    def test_30d_range_unchanged(self) -> None:
        """8-30 day range still returns 1.0."""
        features = self._build_features_with_days(15)
        assert features.time_decay_multiplier == 1.0

    def test_near_resolution_boost_unchanged(self) -> None:
        """≤7 days still gets boost."""
        features = self._build_features_with_days(3)
        assert features.time_decay_multiplier > 1.0
