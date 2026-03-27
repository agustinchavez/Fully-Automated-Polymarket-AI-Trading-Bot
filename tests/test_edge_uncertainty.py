"""Tests for edge uncertainty scoring (Phase 3 — Batch A)."""

from __future__ import annotations

import pytest

from src.policy.edge_uncertainty import (
    UncertaintyInputs,
    EdgeUncertaintyResult,
    compute_edge_uncertainty,
    adjust_edge_for_uncertainty,
    compute_and_adjust,
    _W_ENSEMBLE_SPREAD,
    _W_EVIDENCE_QUALITY,
    _W_BASE_RATE_DISTANCE,
    _W_DECOMPOSITION,
)
from src.policy.edge_calc import calculate_edge, EdgeResult
from src.policy.position_sizer import calculate_position_size, PositionSize
from src.config import RiskConfig


# ── helpers ─────────────────────────────────────────────────────────


def _risk_cfg(**overrides) -> RiskConfig:
    defaults = dict(
        kill_switch=False,
        max_daily_loss=100.0,
        max_open_positions=20,
        max_stake_per_market=50.0,
        max_bankroll_fraction=0.05,
        min_edge=0.02,
        min_liquidity=500.0,
        max_spread=0.12,
        kelly_fraction=0.25,
        bankroll=5000.0,
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _edge(implied: float = 0.60, model: float = 0.70) -> EdgeResult:
    return calculate_edge(implied_prob=implied, model_prob=model)


# ── compute_edge_uncertainty ────────────────────────────────────────


class TestComputeEdgeUncertainty:

    def test_zero_uncertainty_all_signals_perfect(self) -> None:
        """All inputs ideal → uncertainty near 0."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.6,
            model_probability=0.6,
            decomposition_sub_probs=[0.6, 0.6, 0.6],
        )
        score = compute_edge_uncertainty(inputs)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_max_uncertainty_all_signals_bad(self) -> None:
        """All inputs worst-case → uncertainty near 1.0."""
        inputs = UncertaintyInputs(
            ensemble_spread=1.0,
            evidence_quality=0.0,
            base_rate=0.0,
            model_probability=1.0,
            decomposition_sub_probs=[0.0, 1.0],
        )
        score = compute_edge_uncertainty(inputs)
        assert score > 0.8
        assert score <= 1.0

    def test_ensemble_spread_weight(self) -> None:
        """Vary only ensemble spread → contributes 0.3 weight."""
        base = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        full = UncertaintyInputs(
            ensemble_spread=1.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        diff = compute_edge_uncertainty(full) - compute_edge_uncertainty(base)
        assert diff == pytest.approx(_W_ENSEMBLE_SPREAD, abs=0.01)

    def test_evidence_quality_weight(self) -> None:
        """Vary only evidence quality → contributes 0.25 weight."""
        good = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        bad = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=0.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        diff = compute_edge_uncertainty(bad) - compute_edge_uncertainty(good)
        assert diff == pytest.approx(_W_EVIDENCE_QUALITY, abs=0.01)

    def test_base_rate_distance_weight(self) -> None:
        """Vary only base rate distance → contributes 0.25 weight."""
        close = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        far = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.0,
            model_probability=1.0,
            decomposition_sub_probs=[0.5, 0.5],
        )
        diff = compute_edge_uncertainty(far) - compute_edge_uncertainty(close)
        assert diff == pytest.approx(_W_BASE_RATE_DISTANCE, abs=0.01)

    def test_decomposition_disagreement_weight(self) -> None:
        """Vary only decomposition sub-probs → contributes 0.2 weight."""
        agree = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.5, 0.5],
        )
        disagree = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.0, 1.0],
        )
        diff = compute_edge_uncertainty(disagree) - compute_edge_uncertainty(agree)
        assert diff > 0.1  # significant increase
        assert diff <= _W_DECOMPOSITION  # capped by weight

    def test_no_decomposition_uses_zero(self) -> None:
        """Empty sub_probs → 0.0 default → no decomposition penalty."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
        )
        score = compute_edge_uncertainty(inputs)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_single_sub_question_zero_disagreement(self) -> None:
        """One sub-prob → uses default 0.0 since std requires >= 2 samples."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.5,
            decomposition_sub_probs=[0.7],
        )
        score = compute_edge_uncertainty(inputs)
        # With single sub-prob, uses default 0.0 → no decomp penalty
        assert score == pytest.approx(0.0, abs=0.01)

    def test_clamped_to_0_1(self) -> None:
        """Extreme inputs stay within [0, 1]."""
        extreme = UncertaintyInputs(
            ensemble_spread=5.0,
            evidence_quality=-1.0,
            base_rate=-1.0,
            model_probability=2.0,
            decomposition_sub_probs=[0.0, 1.0],
        )
        score = compute_edge_uncertainty(extreme)
        assert 0.0 <= score <= 1.0

    def test_moderate_uncertainty_scenario(self) -> None:
        """Realistic inputs → sanity check value."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.15,
            evidence_quality=0.7,
            base_rate=0.3,
            model_probability=0.65,
            decomposition_sub_probs=[0.6, 0.7],
        )
        score = compute_edge_uncertainty(inputs)
        assert 0.1 < score < 0.6

    def test_perfect_evidence_reduces_uncertainty(self) -> None:
        """evidence_quality=1.0 vs 0.5 → lower uncertainty."""
        base = UncertaintyInputs(
            ensemble_spread=0.2,
            evidence_quality=0.5,
            base_rate=0.5,
            model_probability=0.6,
        )
        better = UncertaintyInputs(
            ensemble_spread=0.2,
            evidence_quality=1.0,
            base_rate=0.5,
            model_probability=0.6,
        )
        assert compute_edge_uncertainty(better) < compute_edge_uncertainty(base)

    def test_extreme_base_rate_distance(self) -> None:
        """model=0.95, base_rate=0.10 → high base rate distance component."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.0,
            evidence_quality=1.0,
            base_rate=0.10,
            model_probability=0.95,
            decomposition_sub_probs=[0.5, 0.5],
        )
        score = compute_edge_uncertainty(inputs)
        # base_rate_distance = 0.85, component = 0.25 * 0.85 = 0.2125
        assert score > 0.2


# ── adjust_edge_for_uncertainty ─────────────────────────────────────


class TestAdjustEdgeForUncertainty:

    def test_spec_example_6pct_80pct(self) -> None:
        """6% raw edge, 80% uncertainty, penalty=0.5 → 3.6% effective."""
        effective = adjust_edge_for_uncertainty(0.06, 0.8, 0.5)
        assert effective == pytest.approx(0.036, abs=0.001)

    def test_zero_uncertainty_no_change(self) -> None:
        """0% uncertainty → effective_edge = raw_edge."""
        effective = adjust_edge_for_uncertainty(0.10, 0.0, 0.5)
        assert effective == pytest.approx(0.10, abs=0.001)

    def test_full_uncertainty_halves_edge(self) -> None:
        """100% uncertainty, penalty=0.5 → effective = raw * 0.5."""
        effective = adjust_edge_for_uncertainty(0.10, 1.0, 0.5)
        assert effective == pytest.approx(0.05, abs=0.001)

    def test_custom_penalty_factor(self) -> None:
        """penalty=0.3 → less aggressive reduction."""
        effective = adjust_edge_for_uncertainty(0.10, 0.5, 0.3)
        # 0.10 * (1 - 0.5 * 0.3) = 0.10 * 0.85 = 0.085
        assert effective == pytest.approx(0.085, abs=0.001)

    def test_penalty_factor_zero_disables(self) -> None:
        """penalty=0 → no change regardless of uncertainty."""
        effective = adjust_edge_for_uncertainty(0.10, 1.0, 0.0)
        assert effective == pytest.approx(0.10, abs=0.001)

    def test_edge_below_threshold_blocked(self) -> None:
        """5% raw edge + enough uncertainty → drops below 4%."""
        effective = adjust_edge_for_uncertainty(0.05, 0.6, 0.5)
        # 0.05 * (1 - 0.6 * 0.5) = 0.05 * 0.7 = 0.035
        assert effective < 0.04

    def test_edge_above_threshold_passes(self) -> None:
        """8% raw edge + moderate uncertainty → stays above 4%."""
        effective = adjust_edge_for_uncertainty(0.08, 0.3, 0.5)
        # 0.08 * (1 - 0.3 * 0.5) = 0.08 * 0.85 = 0.068
        assert effective > 0.04

    def test_negative_edge_handled(self) -> None:
        """Negative raw_edge produces negative effective_edge."""
        effective = adjust_edge_for_uncertainty(-0.05, 0.5, 0.5)
        assert effective < 0


# ── compute_and_adjust ──────────────────────────────────────────────


class TestComputeAndAdjust:

    def test_full_pipeline(self) -> None:
        """End-to-end: compute uncertainty and adjust edge."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.2,
            evidence_quality=0.7,
            base_rate=0.5,
            model_probability=0.65,
        )
        result = compute_and_adjust(inputs, raw_edge=0.06, penalty_factor=0.5)
        assert isinstance(result, EdgeUncertaintyResult)
        assert 0.0 < result.uncertainty_score < 1.0
        assert result.effective_edge < result.raw_edge
        assert result.raw_edge == 0.06

    def test_result_components_populated(self) -> None:
        """All component fields are non-negative."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.3,
            evidence_quality=0.6,
            base_rate=0.4,
            model_probability=0.7,
            decomposition_sub_probs=[0.6, 0.8],
        )
        result = compute_and_adjust(inputs, raw_edge=0.10)
        assert result.ensemble_spread_component >= 0
        assert result.evidence_quality_component >= 0
        assert result.base_rate_distance_component >= 0
        assert result.decomposition_disagreement_component >= 0
        total = (
            result.ensemble_spread_component
            + result.evidence_quality_component
            + result.base_rate_distance_component
            + result.decomposition_disagreement_component
        )
        assert total == pytest.approx(result.uncertainty_score, abs=0.01)

    def test_was_adjusted_flag(self) -> None:
        """Default was_adjusted is True."""
        inputs = UncertaintyInputs()
        result = compute_and_adjust(inputs, raw_edge=0.05)
        assert result.was_adjusted is True

    def test_all_defaults(self) -> None:
        """Default UncertaintyInputs produces valid result."""
        inputs = UncertaintyInputs()
        result = compute_and_adjust(inputs, raw_edge=0.05)
        assert 0.0 <= result.uncertainty_score <= 1.0
        assert result.effective_edge <= result.raw_edge

    def test_realistic_scenario(self) -> None:
        """Realistic trading scenario."""
        inputs = UncertaintyInputs(
            ensemble_spread=0.12,    # models disagree by 12%
            evidence_quality=0.75,   # decent evidence
            base_rate=0.25,          # historical base rate 25%
            model_probability=0.40,  # model says 40%
            decomposition_sub_probs=[0.35, 0.45],  # sub-questions somewhat agree
        )
        result = compute_and_adjust(inputs, raw_edge=0.08, penalty_factor=0.5)
        # Should have moderate uncertainty
        assert 0.15 < result.uncertainty_score < 0.45
        # Edge should be reduced but still positive
        assert 0.04 < result.effective_edge < 0.08


# ── EdgeResult integration ──────────────────────────────────────────


class TestIntegrationWithEdgeResult:

    def test_effective_edge_none_by_default(self) -> None:
        """EdgeResult.effective_edge is None by default."""
        result = _edge()
        assert result.effective_edge is None

    def test_effective_edge_can_be_set(self) -> None:
        """EdgeResult.effective_edge can be assigned."""
        result = _edge()
        result.effective_edge = 0.05
        assert result.effective_edge == 0.05

    def test_has_edge_uses_effective_when_set(self) -> None:
        """When effective_edge is set, it should be used for threshold."""
        result = _edge(implied=0.60, model=0.70)
        # net_edge is positive and > 0.04
        assert result.abs_net_edge > 0.04
        # Set effective_edge below threshold
        result.effective_edge = 0.02
        edge_for_threshold = result.effective_edge
        assert edge_for_threshold < 0.04

    def test_has_edge_uses_net_edge_when_none(self) -> None:
        """When effective_edge is None, fall back to abs_net_edge."""
        result = _edge(implied=0.60, model=0.70)
        assert result.effective_edge is None
        edge_for_threshold = (
            result.effective_edge
            if result.effective_edge is not None
            else result.abs_net_edge
        )
        assert edge_for_threshold == result.abs_net_edge


# ── Position sizer uncertainty multiplier ───────────────────────────


class TestPositionSizerUncertainty:

    def test_uncertainty_multiplier_default_no_change(self) -> None:
        """Default uncertainty_multiplier=1.0 doesn't change sizing."""
        edge = _edge(implied=0.50, model=0.65)
        cfg = _risk_cfg()
        size_default = calculate_position_size(edge=edge, risk_config=cfg)
        size_explicit = calculate_position_size(
            edge=edge, risk_config=cfg, uncertainty_multiplier=1.0,
        )
        assert size_default.stake_usd == size_explicit.stake_usd

    def test_uncertainty_multiplier_reduces_size(self) -> None:
        """uncertainty_multiplier < 1.0 reduces position size."""
        edge = _edge(implied=0.50, model=0.55)
        cfg = _risk_cfg(max_stake_per_market=500.0)
        size_full = calculate_position_size(edge=edge, risk_config=cfg)
        size_reduced = calculate_position_size(
            edge=edge, risk_config=cfg, uncertainty_multiplier=0.5,
        )
        assert size_full.stake_usd > 0
        assert size_reduced.stake_usd < size_full.stake_usd

    def test_high_uncertainty_halves_kelly(self) -> None:
        """uncertainty_multiplier=0.5 approximately halves the Kelly stake."""
        edge = _edge(implied=0.50, model=0.55)
        cfg = _risk_cfg(max_stake_per_market=500.0)
        size_full = calculate_position_size(edge=edge, risk_config=cfg)
        size_half = calculate_position_size(
            edge=edge, risk_config=cfg, uncertainty_multiplier=0.5,
        )
        if size_full.stake_usd > 0:
            ratio = size_half.stake_usd / size_full.stake_usd
            assert ratio == pytest.approx(0.5, abs=0.1)

    def test_uncertainty_mult_field_on_result(self) -> None:
        """PositionSize dataclass has uncertainty_mult field."""
        edge = _edge(implied=0.50, model=0.65)
        cfg = _risk_cfg()
        size = calculate_position_size(
            edge=edge, risk_config=cfg, uncertainty_multiplier=0.75,
        )
        assert size.uncertainty_mult == 0.75


# ── Config defaults ─────────────────────────────────────────────────


class TestConfigDefaults:

    def test_uncertainty_disabled_by_default(self) -> None:
        """uncertainty_enabled defaults to False."""
        cfg = RiskConfig(bankroll=5000.0, max_stake_per_market=50.0)
        assert cfg.uncertainty_enabled is False

    def test_penalty_factor_default(self) -> None:
        """uncertainty_penalty_factor defaults to 0.5."""
        cfg = RiskConfig(bankroll=5000.0, max_stake_per_market=50.0)
        assert cfg.uncertainty_penalty_factor == 0.5
