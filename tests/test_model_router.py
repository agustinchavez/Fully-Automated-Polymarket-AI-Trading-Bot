"""Tests for model tier routing."""

from __future__ import annotations

import pytest

from src.config import ModelTierConfig
from src.forecast.model_router import select_tier, TierDecision
from src.forecast.feature_builder import MarketFeatures


def _features(**overrides) -> MarketFeatures:
    defaults = dict(
        market_id="m1",
        question="Test",
        market_type="MACRO",
        implied_probability=0.60,
        spread_pct=0.03,
        volume_usd=5000.0,
        evidence_quality=0.7,
    )
    defaults.update(overrides)
    return MarketFeatures(**defaults)


def _tier_cfg(**overrides) -> ModelTierConfig:
    defaults = dict(
        enabled=True,
        scout_models=["gpt-4o-mini"],
        standard_models=["gpt-4o"],
        premium_models=["gpt-4o", "claude-sonnet-4-6", "gemini-2.5-flash", "grok-4-fast-reasoning", "deepseek-chat"],
        premium_min_volume_usd=10000.0,
        premium_min_edge=0.06,
        scout_max_evidence_quality=0.4,
    )
    defaults.update(overrides)
    return ModelTierConfig(**defaults)


class TestTierRouting:
    def test_premium_routing(self) -> None:
        """High volume + strong edge → premium (full ensemble)."""
        tier = select_tier(
            _features(volume_usd=20000.0),
            _tier_cfg(),
            rough_edge=0.10,  # 10% edge
        )
        assert tier.tier == "premium"
        assert len(tier.models) == 5

    def test_scout_routing(self) -> None:
        """Low evidence quality → scout (cheap single model)."""
        tier = select_tier(
            _features(evidence_quality=0.2),
            _tier_cfg(),
            rough_edge=0.03,
        )
        assert tier.tier == "scout"
        assert tier.models == ["gpt-4o-mini"]

    def test_standard_default(self) -> None:
        """Normal quality, moderate volume → standard."""
        tier = select_tier(
            _features(volume_usd=5000.0, evidence_quality=0.7),
            _tier_cfg(),
            rough_edge=0.03,
        )
        assert tier.tier == "standard"
        assert tier.models == ["gpt-4o"]

    def test_disabled_uses_premium(self) -> None:
        """When disabled, always returns premium (full ensemble)."""
        tier = select_tier(
            _features(evidence_quality=0.1),  # would be scout if enabled
            _tier_cfg(enabled=False),
            rough_edge=0.01,
        )
        assert tier.tier == "premium"
        assert len(tier.models) == 5

    def test_premium_needs_both_conditions(self) -> None:
        """High volume alone isn't enough — also need edge."""
        tier = select_tier(
            _features(volume_usd=50000.0),
            _tier_cfg(),
            rough_edge=0.02,  # below 6% threshold
        )
        assert tier.tier != "premium"

    def test_edge_alone_not_premium(self) -> None:
        """High edge alone isn't enough — also need volume."""
        tier = select_tier(
            _features(volume_usd=1000.0),
            _tier_cfg(),
            rough_edge=0.15,
        )
        assert tier.tier != "premium"

    def test_config_loads(self) -> None:
        from src.config import load_config
        cfg = load_config()
        assert cfg.model_tiers.enabled is True
        assert cfg.model_tiers.premium_min_volume_usd == 5000.0
