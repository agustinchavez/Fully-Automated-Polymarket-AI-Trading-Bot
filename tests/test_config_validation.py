"""Tests for cross-field config validation (Phase 0B)."""

from __future__ import annotations

import warnings

import pytest

from src.config import (
    BotConfig,
    DrawdownConfig,
    EnsembleConfig,
    ModelTierConfig,
    RiskConfig,
    load_config,
)


class TestRiskConfigValidation:
    """Cross-field validation on RiskConfig."""

    def test_max_stake_exceeds_bankroll_raises(self) -> None:
        with pytest.raises(ValueError, match="max_stake_per_market.*bankroll"):
            RiskConfig(max_stake_per_market=6000, bankroll=5000)

    def test_max_stake_equal_bankroll_ok(self) -> None:
        rc = RiskConfig(max_stake_per_market=5000, bankroll=5000)
        assert rc.max_stake_per_market == 5000

    def test_min_edge_lte_fees_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RiskConfig(min_edge=0.04, transaction_fee_pct=0.02, exit_fee_pct=0.02)
            fee_warnings = [x for x in w if "profit margin" in str(x.message)]
            assert len(fee_warnings) == 1

    def test_min_edge_gt_fees_no_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RiskConfig(min_edge=0.06, transaction_fee_pct=0.02, exit_fee_pct=0.02)
            fee_warnings = [x for x in w if "profit margin" in str(x.message)]
            assert len(fee_warnings) == 0

    def test_volatility_multipliers_inverted_raises(self) -> None:
        with pytest.raises(ValueError, match="volatility_high_min_mult"):
            RiskConfig(volatility_high_min_mult=0.8, volatility_med_min_mult=0.6)

    def test_volatility_multipliers_equal_raises(self) -> None:
        with pytest.raises(ValueError, match="volatility_high_min_mult"):
            RiskConfig(volatility_high_min_mult=0.5, volatility_med_min_mult=0.5)

    def test_default_risk_config_valid(self) -> None:
        """Default values must pass all validations (may warn)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rc = RiskConfig()
            assert rc.bankroll > 0


class TestEnsembleConfigValidation:

    def test_enabled_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EnsembleConfig(enabled=True, models=[])

    def test_disabled_empty_models_ok(self) -> None:
        ec = EnsembleConfig(enabled=False, models=[])
        assert ec.enabled is False

    def test_min_models_exceeds_list_raises(self) -> None:
        with pytest.raises(ValueError, match="min_models_required"):
            EnsembleConfig(models=["gpt-4o"], min_models_required=3)

    def test_min_models_equal_list_ok(self) -> None:
        ec = EnsembleConfig(models=["gpt-4o", "claude"], min_models_required=2)
        assert ec.min_models_required == 2

    def test_invalid_aggregation_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregation"):
            EnsembleConfig(aggregation="mean")

    def test_valid_aggregation_methods(self) -> None:
        for method in ("trimmed_mean", "median", "weighted"):
            ec = EnsembleConfig(aggregation=method)
            assert ec.aggregation == method

    def test_trim_fraction_at_half_raises(self) -> None:
        with pytest.raises(ValueError, match="trim_fraction"):
            EnsembleConfig(trim_fraction=0.5)

    def test_trim_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="trim_fraction"):
            EnsembleConfig(trim_fraction=-0.1)

    def test_trim_fraction_zero_ok(self) -> None:
        ec = EnsembleConfig(trim_fraction=0.0)
        assert ec.trim_fraction == 0.0

    def test_defaults_valid(self) -> None:
        ec = EnsembleConfig()
        assert ec.enabled is True
        assert len(ec.models) == 3


class TestDrawdownConfigValidation:

    def test_thresholds_misordered_raises(self) -> None:
        with pytest.raises(ValueError, match="warning.*critical.*max"):
            DrawdownConfig(
                warning_drawdown_pct=0.15,
                critical_drawdown_pct=0.10,
                max_drawdown_pct=0.20,
            )

    def test_warning_equals_critical_raises(self) -> None:
        with pytest.raises(ValueError, match="warning.*critical.*max"):
            DrawdownConfig(
                warning_drawdown_pct=0.10,
                critical_drawdown_pct=0.10,
                max_drawdown_pct=0.20,
            )

    def test_critical_equals_max_raises(self) -> None:
        with pytest.raises(ValueError, match="warning.*critical.*max"):
            DrawdownConfig(
                warning_drawdown_pct=0.05,
                critical_drawdown_pct=0.20,
                max_drawdown_pct=0.20,
            )

    def test_correct_ordering_ok(self) -> None:
        dc = DrawdownConfig(
            warning_drawdown_pct=0.08,
            critical_drawdown_pct=0.12,
            max_drawdown_pct=0.20,
        )
        assert dc.warning_drawdown_pct < dc.critical_drawdown_pct < dc.max_drawdown_pct

    def test_defaults_valid(self) -> None:
        dc = DrawdownConfig()
        assert dc.warning_drawdown_pct < dc.critical_drawdown_pct < dc.max_drawdown_pct


class TestModelTierConfigValidation:

    def test_enabled_empty_scout_raises(self) -> None:
        with pytest.raises(ValueError, match="scout_models"):
            ModelTierConfig(enabled=True, scout_models=[])

    def test_enabled_empty_standard_raises(self) -> None:
        with pytest.raises(ValueError, match="standard_models"):
            ModelTierConfig(enabled=True, standard_models=[])

    def test_enabled_empty_premium_raises(self) -> None:
        with pytest.raises(ValueError, match="premium_models"):
            ModelTierConfig(enabled=True, premium_models=[])

    def test_disabled_empty_lists_ok(self) -> None:
        mtc = ModelTierConfig(
            enabled=False, scout_models=[], standard_models=[], premium_models=[],
        )
        assert mtc.enabled is False

    def test_defaults_valid(self) -> None:
        mtc = ModelTierConfig()
        assert len(mtc.scout_models) > 0
        assert len(mtc.standard_models) > 0
        assert len(mtc.premium_models) > 0


class TestBotConfigIntegration:
    """Full BotConfig validates all sub-configs."""

    def test_default_botconfig_valid(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = BotConfig()
            assert cfg.risk.bankroll > 0

    def test_load_config_valid(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = load_config()
            assert cfg.ensemble.enabled is True
            assert cfg.risk.bankroll > 0
