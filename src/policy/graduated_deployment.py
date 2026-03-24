"""Graduated capital deployment — staged bankroll scaling.

Stages:
  paper    → $0 real capital (paper trading only)
  week1    → $100 bankroll, $5 max stake (after preflight passes)
  week2    → $500 bankroll, $25 max stake (if Week 1 loss ≤ 10%)
  week3_4  → $2000 bankroll, $50 max stake (if cumulative P&L positive)
  month2_plus → base + (Sharpe × multiplier), capped at 5x base
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from src.config import BotConfig
from src.observability.logger import get_logger

log = get_logger(__name__)

STAGES = ["paper", "week1", "week2", "week3_4", "month2_plus"]


@dataclass
class DeploymentStage:
    """A single deployment stage with its constraints."""
    name: str
    bankroll: float
    max_stake: float
    min_duration_days: int = 7
    criteria: str = ""


class GraduatedDeploymentManager:
    """Manage staged bankroll scaling."""

    def __init__(self, config: BotConfig, conn: sqlite3.Connection | None = None):
        self._config = config
        self._conn = conn
        self._prod = config.production

    def get_stage_config(self, stage: str | None = None) -> DeploymentStage:
        """Get the configuration for a deployment stage."""
        stage = stage or self._prod.deployment_stage

        if stage == "paper":
            return DeploymentStage(
                name="paper",
                bankroll=self._config.risk.bankroll,
                max_stake=self._config.risk.max_stake_per_market,
                min_duration_days=self._prod.preflight_min_paper_days,
                criteria="Manual advance after preflight passes",
            )
        elif stage == "week1":
            return DeploymentStage(
                name="week1",
                bankroll=self._prod.week1_bankroll,
                max_stake=self._prod.week1_max_stake,
                min_duration_days=7,
                criteria=f"Week 1 loss ≤ {self._prod.week1_max_loss_pct:.0%}",
            )
        elif stage == "week2":
            return DeploymentStage(
                name="week2",
                bankroll=self._prod.week2_bankroll,
                max_stake=self._prod.week2_max_stake,
                min_duration_days=7,
                criteria="Cumulative P&L positive",
            )
        elif stage == "week3_4":
            return DeploymentStage(
                name="week3_4",
                bankroll=self._prod.week3_4_bankroll,
                max_stake=self._prod.week3_4_max_stake,
                min_duration_days=14,
                criteria="Cumulative P&L positive",
            )
        elif stage == "month2_plus":
            return DeploymentStage(
                name="month2_plus",
                bankroll=self._prod.week3_4_bankroll,  # base
                max_stake=self._prod.week3_4_max_stake,
                min_duration_days=30,
                criteria="Sharpe-scaled bankroll, capped at 5x base",
            )
        else:
            # Unknown stage, use paper defaults
            return DeploymentStage(
                name=stage,
                bankroll=self._config.risk.bankroll,
                max_stake=self._config.risk.max_stake_per_market,
            )

    def get_effective_bankroll(self, sharpe: float = 0.0) -> float:
        """Return the stage-appropriate bankroll."""
        stage = self._prod.deployment_stage

        if stage == "month2_plus" and sharpe > 0:
            base = self._prod.week3_4_bankroll
            scaled = base + (sharpe * base * 0.5)  # Sharpe scaling
            return min(scaled, base * 5)  # Cap at 5x

        sc = self.get_stage_config(stage)
        return sc.bankroll

    def get_effective_max_stake(self) -> float:
        """Return the stage-appropriate max stake."""
        sc = self.get_stage_config()
        return sc.max_stake

    def apply_stage_limits(self, sharpe: float = 0.0) -> tuple[float, float]:
        """Apply deployment stage limits. Returns (bankroll, max_stake)."""
        if not self._prod.enabled:
            return self._config.risk.bankroll, self._config.risk.max_stake_per_market

        bankroll = self.get_effective_bankroll(sharpe)
        max_stake = self.get_effective_max_stake()

        log.info(
            "deployment.stage_limits",
            stage=self._prod.deployment_stage,
            bankroll=bankroll,
            max_stake=max_stake,
        )

        return bankroll, max_stake

    def check_advancement(
        self,
        cumulative_pnl: float,
        sharpe: float = 0.0,
        days_in_stage: int = 0,
    ) -> str | None:
        """Check if the bot should advance to the next stage.

        Returns the next stage name, or None if no advancement.
        """
        current = self._prod.deployment_stage

        if current == "paper":
            return None  # Manual advance only

        if current == "week1":
            if days_in_stage < 7:
                return None
            # Week 1 loss must be ≤ threshold
            bankroll = self._prod.week1_bankroll
            if bankroll > 0:
                loss_pct = abs(min(0, cumulative_pnl)) / bankroll
                if loss_pct <= self._prod.week1_max_loss_pct:
                    return "week2"
            return None

        if current == "week2":
            if days_in_stage < 7:
                return None
            if cumulative_pnl > 0:
                return "week3_4"
            return None

        if current == "week3_4":
            if days_in_stage < 14:
                return None
            if cumulative_pnl > 0:
                return "month2_plus"
            return None

        return None  # month2_plus is the final stage

    def advance_stage(
        self,
        next_stage: str,
        cumulative_pnl: float = 0.0,
        sharpe: float = 0.0,
    ) -> None:
        """Record stage advancement and update config."""
        prev_stage = self._prod.deployment_stage
        self._prod.deployment_stage = next_stage

        log.info(
            "deployment.stage_advanced",
            from_stage=prev_stage,
            to_stage=next_stage,
            cumulative_pnl=cumulative_pnl,
            sharpe=sharpe,
        )

        if self._conn:
            try:
                sc = self.get_stage_config(next_stage)
                self._conn.execute(
                    """INSERT INTO deployment_stages
                        (stage, bankroll, max_stake, cumulative_pnl,
                         sharpe_ratio, advanced_reason, started_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        next_stage, sc.bankroll, sc.max_stake,
                        cumulative_pnl, sharpe,
                        f"Advanced from {prev_stage}",
                        dt.datetime.now(dt.timezone.utc).isoformat(),
                    ),
                )
                self._conn.commit()
            except sqlite3.OperationalError as e:
                log.warning("deployment.persist_error", error=str(e))

    def get_stage_history(self) -> list[dict[str, Any]]:
        """Get deployment stage history."""
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                "SELECT * FROM deployment_stages ORDER BY started_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []
