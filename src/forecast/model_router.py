"""Model tier router — select model quality based on opportunity characteristics.

Tiers:
  - Scout:    cheap single model for low-quality evidence (quick screen)
  - Standard: single frontier model for typical opportunities
  - Premium:  full multi-model ensemble for high-value opportunities
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import ModelTierConfig
from src.forecast.feature_builder import MarketFeatures
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class TierDecision:
    """Result of model tier selection."""
    tier: str                  # "scout" | "standard" | "premium"
    models: list[str]
    reason: str


def select_tier(
    features: MarketFeatures,
    tier_config: ModelTierConfig,
    rough_edge: float = 0.0,
) -> TierDecision:
    """Select which model tier to use for forecasting.

    Args:
        features: Market features (volume, evidence quality, etc.)
        tier_config: Tier configuration with thresholds and model lists.
        rough_edge: Pre-forecast rough edge estimate (implied_prob distance from 0.5).

    Returns:
        TierDecision with tier name, models, and reason.
    """
    if not tier_config.enabled:
        return TierDecision(
            tier="premium",
            models=tier_config.premium_models,
            reason="tier routing disabled",
        )

    # Premium: high-volume markets with meaningful edge potential
    if (
        features.volume_usd >= tier_config.premium_min_volume_usd
        and abs(rough_edge) >= tier_config.premium_min_edge
    ):
        return TierDecision(
            tier="premium",
            models=tier_config.premium_models,
            reason=(
                f"volume ${features.volume_usd:,.0f} >= ${tier_config.premium_min_volume_usd:,.0f} "
                f"and edge {abs(rough_edge):.2%} >= {tier_config.premium_min_edge:.2%}"
            ),
        )

    # Scout: low evidence quality — quick screen, don't waste tokens
    if features.evidence_quality < tier_config.scout_max_evidence_quality:
        return TierDecision(
            tier="scout",
            models=tier_config.scout_models,
            reason=f"evidence quality {features.evidence_quality:.2f} < {tier_config.scout_max_evidence_quality}",
        )

    # Standard: everything else
    return TierDecision(
        tier="standard",
        models=tier_config.standard_models,
        reason="default standard tier",
    )
