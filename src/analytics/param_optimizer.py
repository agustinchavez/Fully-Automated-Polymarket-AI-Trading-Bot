"""Automatic strategy parameter tuning via backtest perturbation.

Every optimization_interval_days, generates N random perturbations of
tunable parameters (±20% by default), evaluates each via a mini-backtest
over the last 30 days, and surfaces statistically significant improvements
in the dashboard for human approval.

Human-in-the-loop: parameter changes are NEVER auto-applied.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


# Tunable parameters: (config_section, field_name, min_value, max_value)
TUNABLE_PARAMS: dict[str, tuple[str, str, float, float]] = {
    "min_edge": ("risk", "min_edge", 0.01, 0.15),
    "kelly_fraction": ("risk", "kelly_fraction", 0.05, 0.50),
    "min_evidence_quality": ("forecasting", "min_evidence_quality", 0.3, 0.8),
    "stop_loss_pct": ("risk", "stop_loss_pct", 0.05, 0.40),
    "take_profit_pct": ("risk", "take_profit_pct", 0.10, 0.50),
    "max_stake_per_market": ("risk", "max_stake_per_market", 10.0, 200.0),
}


@dataclass
class OptimizationResult:
    """Result of a parameter optimization run."""
    run_id: str = ""
    status: str = "pending"  # pending | completed | failed | no_improvement
    num_perturbations: int = 0
    best_sharpe: float = 0.0
    baseline_sharpe: float = 0.0
    sharpe_improvement_pct: float = 0.0
    p_value: float = 1.0
    significance: str = "none"  # none | weak | moderate | strong
    best_config: dict[str, Any] = field(default_factory=dict)
    config_diff: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "num_perturbations": self.num_perturbations,
            "best_sharpe": self.best_sharpe,
            "baseline_sharpe": self.baseline_sharpe,
            "sharpe_improvement_pct": self.sharpe_improvement_pct,
            "p_value": self.p_value,
            "significance": self.significance,
            "best_config": self.best_config,
            "config_diff": self.config_diff,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ParameterOptimizer:
    """Generates and evaluates parameter perturbations."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @staticmethod
    def generate_perturbations(
        current_config: dict[str, float],
        n: int = 30,
        range_pct: float = 0.20,
        seed: int | None = None,
    ) -> list[dict[str, float]]:
        """Generate N random perturbations of the current parameters.

        Each param is varied uniformly within ±range_pct of its current
        value, clamped to the parameter's allowed [min, max].
        """
        rng = random.Random(seed)
        perturbations = []

        for _ in range(n):
            perturbed = {}
            for name, value in current_config.items():
                if name not in TUNABLE_PARAMS:
                    perturbed[name] = value
                    continue
                _, _, param_min, param_max = TUNABLE_PARAMS[name]
                low = max(param_min, value * (1 - range_pct))
                high = min(param_max, value * (1 + range_pct))
                perturbed[name] = round(rng.uniform(low, high), 6)
            perturbations.append(perturbed)

        return perturbations

    @staticmethod
    def extract_current_params(config: Any) -> dict[str, float]:
        """Extract current tunable parameter values from a BotConfig."""
        params: dict[str, float] = {}
        for name, (section, field_name, _, _) in TUNABLE_PARAMS.items():
            section_obj = getattr(config, section, None)
            if section_obj:
                params[name] = float(getattr(section_obj, field_name, 0))
        return params

    @staticmethod
    def compute_config_diff(
        baseline: dict[str, float],
        best: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compute the diff between baseline and best configs."""
        diff: dict[str, dict[str, float]] = {}
        for k in baseline:
            if k in best and abs(baseline[k] - best[k]) > 1e-9:
                diff[k] = {
                    "baseline": baseline[k],
                    "optimized": best[k],
                    "change_pct": round(
                        (best[k] - baseline[k]) / baseline[k] * 100
                        if baseline[k] != 0 else 0,
                        2,
                    ),
                }
        return diff

    @staticmethod
    def significance_level(p_value: float) -> str:
        """Map a p-value to a significance level."""
        if p_value < 0.01:
            return "strong"
        elif p_value < 0.05:
            return "moderate"
        elif p_value < 0.10:
            return "weak"
        return "none"

    @staticmethod
    def paired_ttest(values_a: list[float], values_b: list[float]) -> float:
        """Compute paired t-test p-value (reuses BacktestComparator pattern)."""
        n = min(len(values_a), len(values_b))
        if n < 3:
            return 1.0
        diffs = [values_b[i] - values_a[i] for i in range(n)]
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
        if var_d <= 0:
            return 1.0 if abs(mean_d) < 1e-12 else 0.0
        se = math.sqrt(var_d / n)
        if se == 0:
            return 1.0
        t_stat = mean_d / se
        # Approximate p-value using df=n-1
        df = n - 1
        # Simple approximation for two-sided test
        x = abs(t_stat)
        p = math.exp(-0.717 * x - 0.416 * x * x)
        return min(1.0, p)

    def save_run(self, result: OptimizationResult) -> None:
        """Persist an optimization run to the database."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO param_optimization_runs
                    (run_id, status, num_perturbations, best_sharpe,
                     baseline_sharpe, sharpe_improvement_pct, p_value,
                     significance, best_config_json, config_diff_json,
                     applied, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                result.run_id, result.status, result.num_perturbations,
                result.best_sharpe, result.baseline_sharpe,
                result.sharpe_improvement_pct, result.p_value,
                result.significance,
                json.dumps(result.best_config),
                json.dumps(result.config_diff),
                result.started_at, result.completed_at,
            ))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("param_optimizer.save_run_error", error=str(e))

    def save_perturbation_result(
        self,
        run_id: str,
        config: dict[str, float],
        sharpe: float,
        total_pnl: float = 0.0,
        win_rate: float = 0.0,
        max_drawdown: float = 0.0,
        brier: float = 1.0,
    ) -> None:
        """Save a single perturbation result."""
        try:
            self._conn.execute("""
                INSERT INTO param_optimization_results
                    (run_id, config_json, sharpe_ratio, total_pnl,
                     win_rate, max_drawdown_pct, brier_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, json.dumps(config), sharpe, total_pnl,
                win_rate, max_drawdown, brier, _now_iso(),
            ))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("param_optimizer.save_result_error", error=str(e))

    def get_pending_suggestions(self) -> list[dict[str, Any]]:
        """Get unapplied optimization suggestions with significant improvement."""
        try:
            rows = self._conn.execute("""
                SELECT * FROM param_optimization_runs
                WHERE applied = 0
                  AND significance != 'none'
                  AND sharpe_improvement_pct > 0
                ORDER BY sharpe_improvement_pct DESC
            """).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "sharpe_improvement_pct": float(r["sharpe_improvement_pct"]),
                "p_value": float(r["p_value"]),
                "significance": r["significance"],
                "best_config": json.loads(r["best_config_json"] or "{}"),
                "config_diff": json.loads(r["config_diff_json"] or "{}"),
                "completed_at": r["completed_at"] or "",
            }
            for r in rows
        ]

    def apply_suggestion(self, run_id: str) -> bool:
        """Mark a suggestion as applied (actual config write is done by caller)."""
        try:
            self._conn.execute(
                "UPDATE param_optimization_runs SET applied = 1 WHERE run_id = ?",
                (run_id,),
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            return False

    def get_all_runs(self) -> list[dict[str, Any]]:
        """Get all optimization runs."""
        try:
            rows = self._conn.execute("""
                SELECT * FROM param_optimization_runs
                ORDER BY completed_at DESC
            """).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            {
                "run_id": r["run_id"],
                "status": r["status"],
                "num_perturbations": int(r["num_perturbations"]),
                "best_sharpe": float(r["best_sharpe"]),
                "baseline_sharpe": float(r["baseline_sharpe"]),
                "sharpe_improvement_pct": float(r["sharpe_improvement_pct"]),
                "p_value": float(r["p_value"]),
                "significance": r["significance"],
                "applied": bool(r["applied"]),
                "started_at": r["started_at"] or "",
                "completed_at": r["completed_at"] or "",
            }
            for r in rows
        ]
