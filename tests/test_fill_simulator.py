"""Tests for Phase 6 Batch A: Realistic fill simulator."""

from __future__ import annotations

import math
import sqlite3

import pytest

from src.backtest.fill_simulator import FillSimulationConfig, FillSimulator, SimulatedFill
from src.backtest.models import BacktestTradeRecord
from src.config import BacktestConfig


# ── FillSimulationConfig ─────────────────────────────────────────


class TestFillSimulationConfig:
    def test_defaults(self):
        cfg = FillSimulationConfig()
        assert cfg.depth_multiplier == 1.0
        assert cfg.partial_fill_enabled is True
        assert cfg.delay_min_ms == 50
        assert cfg.delay_max_ms == 500
        assert abs(cfg.price_drift_vol - 0.001) < 1e-9
        assert abs(cfg.fee_entry_pct - 0.02) < 1e-9
        assert abs(cfg.fee_exit_pct - 0.02) < 1e-9

    def test_from_backtest_config(self):
        bc = BacktestConfig(
            realistic_fills_enabled=True,
            fill_sim_depth_multiplier=2.0,
            fill_sim_partial_fill_enabled=False,
            fill_sim_delay_min_ms=100,
            fill_sim_delay_max_ms=300,
            fill_sim_price_drift_vol=0.002,
            fill_sim_fee_entry_pct=0.01,
            fill_sim_fee_exit_pct=0.03,
        )
        cfg = FillSimulationConfig.from_backtest_config(bc)
        assert cfg.depth_multiplier == 2.0
        assert cfg.partial_fill_enabled is False
        assert cfg.delay_min_ms == 100
        assert cfg.delay_max_ms == 300
        assert abs(cfg.price_drift_vol - 0.002) < 1e-9
        assert abs(cfg.fee_entry_pct - 0.01) < 1e-9
        assert abs(cfg.fee_exit_pct - 0.03) < 1e-9


# ── FillSimulator Core ───────────────────────────────────────────


class TestFillSimulator:
    def _make_simulator(self, seed: int = 42, **kwargs) -> FillSimulator:
        cfg = FillSimulationConfig(**kwargs)
        return FillSimulator(cfg, seed=seed)

    def test_full_fill_when_depth_exceeds_order(self):
        sim = self._make_simulator()
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert abs(result.fill_rate - 1.0) < 1e-9
        assert abs(result.filled_size_usd - 100.0) < 0.01

    def test_partial_fill_thin_liquidity(self):
        sim = self._make_simulator()
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=200.0,
            available_liquidity_usd=100.0,
        )
        assert result.fill_rate < 1.0
        assert result.filled_size_usd < 200.0
        assert abs(result.fill_rate - 0.5) < 1e-9

    def test_partial_fill_disabled(self):
        sim = self._make_simulator(partial_fill_enabled=False)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=200.0,
            available_liquidity_usd=100.0,
        )
        assert abs(result.fill_rate - 1.0) < 1e-9
        assert abs(result.filled_size_usd - 200.0) < 0.01

    def test_price_impact_proportional(self):
        sim = self._make_simulator(
            price_drift_vol=0.0,  # disable drift for clean comparison
        )
        small = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=10.0,
            available_liquidity_usd=10000.0,
        )
        large = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=1000.0,
            available_liquidity_usd=10000.0,
        )
        assert large.price_impact_pct > small.price_impact_pct

    def test_price_impact_zero_for_tiny_order(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=0.01,
            available_liquidity_usd=100000.0,
        )
        # Impact should be negligible
        assert result.price_impact_pct < 0.001

    def test_price_impact_capped(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100000.0,
            available_liquidity_usd=1.0,
        )
        # Impact should not exceed 50%
        assert result.price_impact_pct <= 0.50

    def test_fill_delay_in_range(self):
        sim = self._make_simulator(delay_min_ms=100, delay_max_ms=200)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert 100 <= result.fill_delay_ms <= 200

    def test_price_drift_deterministic_with_seed(self):
        s1 = self._make_simulator(seed=123)
        s2 = self._make_simulator(seed=123)
        r1 = s1.simulate("BUY_YES", 0.50, "YES", 100.0, 10000.0)
        r2 = s2.simulate("BUY_YES", 0.50, "YES", 100.0, 10000.0)
        assert abs(r1.price_drift_during_delay - r2.price_drift_during_delay) < 1e-12
        assert abs(r1.pnl - r2.pnl) < 1e-6

    def test_fee_calculation(self):
        sim = self._make_simulator(
            price_drift_vol=0.0,
            fee_entry_pct=0.02,
            fee_exit_pct=0.02,
        )
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=100000.0,
        )
        # With near-zero impact and drift, entry fee ~ 100 * 0.02 = 2.0
        # Exit fee applies to payout, which is exit_price * contracts
        assert result.fee_paid_usd > 0

    def test_buy_yes_wins_with_impact(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert result.exit_price == 1.0
        # Should be profitable (bought at ~0.50, resolves to 1.0)
        assert result.pnl > 0

    def test_buy_yes_loses_with_impact(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="NO",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert result.exit_price == 0.0
        # Should lose money
        assert result.pnl < 0

    def test_buy_no_wins_with_impact(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_NO",
            entry_price=0.40,
            resolution="NO",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert result.exit_price == 1.0
        assert result.pnl > 0

    def test_buy_no_loses_with_impact(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_NO",
            entry_price=0.40,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert result.exit_price == 0.0
        assert result.pnl < 0

    def test_zero_liquidity_fallback(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=0.0,
        )
        # Should still produce a fill (fallback to generous depth)
        assert result.fill_rate > 0
        assert result.pnl != 0

    def test_high_price_no_overshoot(self):
        sim = self._make_simulator(price_drift_vol=0.0)
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.95,
            resolution="YES",
            stake_usd=50000.0,
            available_liquidity_usd=100.0,
        )
        # Entry price should be clamped to <= 0.99
        assert result.entry_price <= 0.99

    def test_zero_entry_price(self):
        sim = self._make_simulator()
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.0,
            resolution="YES",
            stake_usd=100.0,
            available_liquidity_usd=10000.0,
        )
        assert result.fill_rate == 0.0
        assert result.pnl == 0.0

    def test_zero_stake(self):
        sim = self._make_simulator()
        result = sim.simulate(
            direction="BUY_YES",
            entry_price=0.50,
            resolution="YES",
            stake_usd=0.0,
            available_liquidity_usd=10000.0,
        )
        assert result.fill_rate == 0.0
        assert result.pnl == 0.0


# ── Integration ──────────────────────────────────────────────────


class TestFillSimulatorIntegration:
    def test_simulated_fill_dataclass(self):
        fill = SimulatedFill(
            entry_price=0.52,
            exit_price=1.0,
            fill_rate=1.0,
            filled_size_usd=100.0,
            slippage_bps=40.0,
            price_impact_pct=0.02,
            fill_delay_ms=150,
            price_drift_during_delay=0.001,
            fee_paid_usd=3.50,
            pnl=85.0,
        )
        assert fill.entry_price == 0.52
        assert fill.pnl == 85.0

    def test_backtest_trade_record_new_fields(self):
        record = BacktestTradeRecord(
            run_id="test",
            market_condition_id="abc",
            slippage_bps=25.5,
            fill_rate=0.8,
            simulated_impact_pct=0.015,
            fill_delay_ms=200,
            fee_paid_usd=2.50,
        )
        assert abs(record.slippage_bps - 25.5) < 1e-9
        assert abs(record.fill_rate - 0.8) < 1e-9
        assert record.fill_delay_ms == 200
        assert abs(record.fee_paid_usd - 2.50) < 1e-9

    def test_backtest_trade_record_defaults(self):
        """Existing code creating records without new fields should still work."""
        record = BacktestTradeRecord(
            run_id="test",
            market_condition_id="abc",
        )
        assert record.slippage_bps == 0.0
        assert record.fill_rate == 1.0
        assert record.simulated_impact_pct == 0.0
        assert record.fill_delay_ms == 0
        assert record.fee_paid_usd == 0.0

    def test_deterministic_across_runs(self):
        cfg = FillSimulationConfig()
        s1 = FillSimulator(cfg, seed=999)
        s2 = FillSimulator(cfg, seed=999)

        results1 = [
            s1.simulate("BUY_YES", 0.60, "YES", 50.0, 5000.0)
            for _ in range(5)
        ]
        results2 = [
            s2.simulate("BUY_YES", 0.60, "YES", 50.0, 5000.0)
            for _ in range(5)
        ]
        for r1, r2 in zip(results1, results2):
            assert abs(r1.pnl - r2.pnl) < 1e-6
            assert abs(r1.entry_price - r2.entry_price) < 1e-12
            assert r1.fill_delay_ms == r2.fill_delay_ms

    def test_config_backward_compat(self):
        """BacktestConfig without new fields should work fine."""
        bc = BacktestConfig()
        assert bc.realistic_fills_enabled is False
        assert bc.fill_sim_depth_multiplier == 1.0

    def test_pnl_worse_with_realistic_fills(self):
        """Realistic fills should generally produce worse P&L than flat slippage."""
        sim = FillSimulator(
            FillSimulationConfig(price_drift_vol=0.0),
            seed=42,
        )

        # Simulate many trades with realistic fills
        total_realistic = 0.0
        total_flat = 0.0
        for i in range(20):
            fill = sim.simulate("BUY_YES", 0.50, "YES", 100.0, 5000.0)
            total_realistic += fill.pnl

            # Simple flat slippage PnL for comparison
            entry_with_slip = 0.50 + 0.005
            contracts = 100.0 / entry_with_slip
            flat_pnl = (1.0 - entry_with_slip) * contracts
            total_flat += flat_pnl

        # Realistic should be lower due to impact + fees
        assert total_realistic < total_flat


# ── Backtest migration ───────────────────────────────────────────


class TestBacktestMigration:
    def test_migration_2_adds_columns(self):
        from src.backtest.migrations import run_migrations, SCHEMA_VERSION

        assert SCHEMA_VERSION == 2

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)

        # Check the new columns exist
        cursor = conn.execute("PRAGMA table_info(backtest_trades)")
        columns = {row["name"] for row in cursor.fetchall()}
        assert "slippage_bps" in columns
        assert "fill_rate" in columns
        assert "simulated_impact_pct" in columns
        assert "fill_delay_ms" in columns
        assert "fee_paid_usd" in columns

        # Check schema version
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert ver == 2
        conn.close()

    def test_migration_idempotent(self):
        from src.backtest.migrations import run_migrations

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)
        run_migrations(conn)  # should not raise
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        assert ver == 2
        conn.close()
