"""Realistic fill simulator for backtesting.

Models orderbook-based fills with:
  - Price impact proportional to order size / available liquidity (sqrt model)
  - Partial fills when order exceeds available depth
  - Random fill delays (50-500ms) with price drift during delay
  - Fee modeling (entry + exit fees)

Disabled by default — gated behind backtest.realistic_fills_enabled.
When disabled, the replay engine uses the existing flat slippage model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class FillSimulationConfig:
    """Configuration for realistic fill simulation."""

    depth_multiplier: float = 1.0
    partial_fill_enabled: bool = True
    delay_min_ms: int = 50
    delay_max_ms: int = 500
    price_drift_vol: float = 0.001
    fee_entry_pct: float = 0.02
    fee_exit_pct: float = 0.02

    @classmethod
    def from_backtest_config(cls, cfg: Any) -> "FillSimulationConfig":
        """Build from a BacktestConfig instance."""
        return cls(
            depth_multiplier=cfg.fill_sim_depth_multiplier,
            partial_fill_enabled=cfg.fill_sim_partial_fill_enabled,
            delay_min_ms=cfg.fill_sim_delay_min_ms,
            delay_max_ms=cfg.fill_sim_delay_max_ms,
            price_drift_vol=cfg.fill_sim_price_drift_vol,
            fee_entry_pct=cfg.fill_sim_fee_entry_pct,
            fee_exit_pct=cfg.fill_sim_fee_exit_pct,
        )


@dataclass
class SimulatedFill:
    """Result of a realistic fill simulation."""

    entry_price: float
    exit_price: float
    fill_rate: float
    filled_size_usd: float
    slippage_bps: float
    price_impact_pct: float
    fill_delay_ms: int
    price_drift_during_delay: float
    fee_paid_usd: float
    pnl: float


class FillSimulator:
    """Simulates realistic order fills for backtesting.

    Uses a square-root price impact model (Kyle's lambda) which is more
    realistic than linear for thin prediction markets.
    """

    # Cap impact to avoid unrealistic prices
    _MAX_IMPACT_PCT = 0.50

    def __init__(
        self,
        config: FillSimulationConfig,
        seed: int | None = None,
    ):
        self._config = config
        self._rng = random.Random(seed)

    def simulate(
        self,
        direction: str,
        entry_price: float,
        resolution: str,
        stake_usd: float,
        available_liquidity_usd: float,
    ) -> SimulatedFill:
        """Simulate a realistic fill.

        Args:
            direction: "BUY_YES" or "BUY_NO"
            entry_price: market price for the token being bought
            resolution: "YES" or "NO" — known outcome
            stake_usd: intended stake in USD
            available_liquidity_usd: available liquidity at this price level

        Returns:
            SimulatedFill with effective entry, P&L, and fill metrics.
        """
        if entry_price <= 0 or stake_usd <= 0:
            exit_price = self._compute_exit_price(direction, resolution)
            return SimulatedFill(
                entry_price=entry_price,
                exit_price=exit_price,
                fill_rate=0.0,
                filled_size_usd=0.0,
                slippage_bps=0.0,
                price_impact_pct=0.0,
                fill_delay_ms=0,
                price_drift_during_delay=0.0,
                fee_paid_usd=0.0,
                pnl=0.0,
            )

        # 1. Compute available depth
        available_depth = available_liquidity_usd * self._config.depth_multiplier
        if available_depth <= 0:
            # Fallback: no liquidity data — use flat slippage (0.5%)
            available_depth = stake_usd * 10  # assume plenty of depth

        # 2. Compute fill rate
        if self._config.partial_fill_enabled and stake_usd > available_depth:
            fill_rate = available_depth / stake_usd
        else:
            fill_rate = 1.0
        filled_size_usd = stake_usd * fill_rate

        # 3. Compute price impact (sqrt model)
        impact_pct = self._compute_price_impact(filled_size_usd, available_depth)

        # 4. Compute fill delay
        delay_ms = self._compute_fill_delay()

        # 5. Compute price drift during delay
        drift = self._compute_price_drift(delay_ms, entry_price)

        # 6. Compute effective entry price
        effective_entry = entry_price + impact_pct * entry_price + drift
        effective_entry = max(0.01, min(0.99, effective_entry))

        # 7. Compute slippage in basis points
        slippage_bps = (
            (effective_entry - entry_price) / entry_price * 10000
            if entry_price > 0
            else 0.0
        )

        # 8. Compute exit price
        exit_price = self._compute_exit_price(direction, resolution)

        # 9. Compute fees
        entry_fee = filled_size_usd * self._config.fee_entry_pct
        # Exit fee on the payout, not the stake
        contracts = filled_size_usd / effective_entry if effective_entry > 0 else 0.0
        exit_payout = exit_price * contracts
        exit_fee = exit_payout * self._config.fee_exit_pct
        total_fees = entry_fee + exit_fee

        # 10. Compute P&L
        gross_pnl = (exit_price - effective_entry) * contracts
        pnl = gross_pnl - total_fees

        return SimulatedFill(
            entry_price=round(effective_entry, 6),
            exit_price=exit_price,
            fill_rate=round(fill_rate, 4),
            filled_size_usd=round(filled_size_usd, 2),
            slippage_bps=round(slippage_bps, 2),
            price_impact_pct=round(impact_pct, 6),
            fill_delay_ms=delay_ms,
            price_drift_during_delay=round(drift, 6),
            fee_paid_usd=round(total_fees, 4),
            pnl=round(pnl, 2),
        )

    def _compute_price_impact(
        self,
        order_size: float,
        available_depth: float,
    ) -> float:
        """Square-root price impact model.

        impact = sqrt(order_size / available_depth) * coefficient
        Capped at _MAX_IMPACT_PCT.
        """
        if available_depth <= 0:
            return 0.0
        ratio = order_size / available_depth
        impact = math.sqrt(ratio) * 0.1  # coefficient tuned for prediction markets
        return min(impact, self._MAX_IMPACT_PCT)

    def _compute_fill_delay(self) -> int:
        """Random fill delay in milliseconds, uniform [min, max]."""
        return self._rng.randint(
            self._config.delay_min_ms,
            self._config.delay_max_ms,
        )

    def _compute_price_drift(self, delay_ms: int, current_price: float) -> float:
        """Simulate price drift during fill delay.

        Uses simple random walk: drift = vol * sqrt(delay_secs) * Z
        where Z ~ N(0,1), clipped to [-3, 3] sigma.
        """
        if delay_ms <= 0 or self._config.price_drift_vol <= 0:
            return 0.0
        delay_secs = delay_ms / 1000.0
        z = max(-3.0, min(3.0, self._rng.gauss(0.0, 1.0)))
        drift = self._config.price_drift_vol * math.sqrt(delay_secs) * z
        return drift * current_price

    @staticmethod
    def _compute_exit_price(direction: str, resolution: str) -> float:
        """Compute exit price based on direction and resolution."""
        if direction == "BUY_YES":
            return 1.0 if resolution == "YES" else 0.0
        else:
            return 1.0 if resolution == "NO" else 0.0
