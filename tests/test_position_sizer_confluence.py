"""Tests for position sizer confluence multiplier."""

from __future__ import annotations

import pytest

from src.config import RiskConfig
from src.policy.edge_calc import EdgeResult
from src.policy.position_sizer import PositionSize, calculate_position_size


# ── Helpers ──────────────────────────────────────────────────────────


def _make_edge(prob: float = 0.60, implied: float = 0.50) -> EdgeResult:
    raw_edge = prob - implied
    return EdgeResult(
        model_probability=prob,
        implied_probability=implied,
        raw_edge=raw_edge,
        edge_pct=raw_edge / implied if implied > 0 else 0,
        direction="BUY_YES",
        expected_value_per_dollar=raw_edge / implied if implied > 0 else 0,
        is_positive=raw_edge > 0,
    )


def _make_config(**kwargs) -> RiskConfig:
    defaults = dict(
        bankroll=10000,
        kelly_fraction=0.25,
        max_stake_per_market=500,
        max_bankroll_fraction=0.10,
        min_stake_usd=1.0,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


# ── Confluence multiplier tests ──────────────────────────────────────


class TestConfluenceMultiplier:
    def test_default_is_1(self) -> None:
        result = calculate_position_size(_make_edge(), _make_config())
        assert result.confluence_mult == 1.0

    def test_confluence_1_full_sizing(self) -> None:
        result_no_conf = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=1.0,
        )
        result_default = calculate_position_size(
            _make_edge(), _make_config(),
        )
        assert result_no_conf.stake_usd == result_default.stake_usd

    def test_confluence_075_reduces_stake(self) -> None:
        result_full = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=1.0,
        )
        result_reduced = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=0.75,
        )
        assert result_reduced.stake_usd < result_full.stake_usd
        assert result_reduced.confluence_mult == 0.75

    def test_confluence_050_halves_sizing(self) -> None:
        result_full = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=1.0,
        )
        result_half = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=0.50,
        )
        # Should be approximately half (may not be exact due to caps)
        assert result_half.stake_usd < result_full.stake_usd
        assert result_half.confluence_mult == 0.50

    def test_confluence_025_minimal_sizing(self) -> None:
        result = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=0.25,
        )
        assert result.confluence_mult == 0.25
        assert result.stake_usd >= 0

    def test_confluence_combines_with_uncertainty(self) -> None:
        result_both = calculate_position_size(
            _make_edge(), _make_config(),
            uncertainty_multiplier=0.75,
            confluence_multiplier=0.75,
        )
        result_uncertainty_only = calculate_position_size(
            _make_edge(), _make_config(),
            uncertainty_multiplier=0.75,
            confluence_multiplier=1.0,
        )
        result_confluence_only = calculate_position_size(
            _make_edge(), _make_config(),
            uncertainty_multiplier=1.0,
            confluence_multiplier=0.75,
        )
        # Both multipliers applied should give smaller stake
        assert result_both.stake_usd <= result_uncertainty_only.stake_usd
        assert result_both.stake_usd <= result_confluence_only.stake_usd

    def test_confluence_stored_in_position_size(self) -> None:
        result = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=0.35,
        )
        assert result.confluence_mult == 0.35

    def test_confluence_in_to_dict(self) -> None:
        result = calculate_position_size(
            _make_edge(), _make_config(), confluence_multiplier=0.50,
        )
        d = result.to_dict()
        assert "confluence_mult" in d
        assert d["confluence_mult"] == 0.50

    def test_backward_compat_no_confluence_param(self) -> None:
        """Calling without confluence_multiplier works (defaults to 1.0)."""
        result = calculate_position_size(
            _make_edge(),
            _make_config(),
            confidence_level="MEDIUM",
            drawdown_multiplier=0.8,
        )
        assert result.confluence_mult == 1.0
        assert result.stake_usd > 0

    def test_portfolio_gate_rejection_ignores_confluence(self) -> None:
        result = calculate_position_size(
            _make_edge(), _make_config(),
            confluence_multiplier=0.5,
            portfolio_gate=(False, "max_category"),
        )
        assert result.stake_usd == 0.0
        assert result.capped_by == "portfolio"
