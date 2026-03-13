"""A/B comparison of two backtest runs with statistical significance.

Compares aggregate metrics (P&L, Brier, Sharpe, win-rate) and runs
a paired t-test on per-market P&L for markets traded in both runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from src.backtest.database import BacktestDatabase
from src.backtest.models import BacktestRunRecord, BacktestTradeRecord
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two backtest runs."""

    run_a_id: str = ""
    run_a_name: str = ""
    run_b_id: str = ""
    run_b_name: str = ""

    # Metric deltas (B - A)
    pnl_delta: float = 0.0
    brier_delta: float = 0.0
    sharpe_delta: float = 0.0
    win_rate_delta: float = 0.0
    max_drawdown_delta: float = 0.0

    # Per-run metrics for display
    run_a_metrics: dict[str, Any] = field(default_factory=dict)
    run_b_metrics: dict[str, Any] = field(default_factory=dict)

    # Statistical significance (paired t-test on overlapping market P&L)
    p_value: float = 1.0
    significance: str = "none"   # none | weak | moderate | strong
    overlapping_markets: int = 0

    # Config differences between runs
    config_diff: dict[str, Any] = field(default_factory=dict)

    # Per-category comparison
    category_comparison: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_a": {"id": self.run_a_id, "name": self.run_a_name},
            "run_b": {"id": self.run_b_id, "name": self.run_b_name},
            "deltas": {
                "pnl": round(self.pnl_delta, 2),
                "brier_score": round(self.brier_delta, 4),
                "sharpe_ratio": round(self.sharpe_delta, 4),
                "win_rate": round(self.win_rate_delta, 4),
                "max_drawdown_pct": round(self.max_drawdown_delta, 4),
            },
            "run_a_metrics": self.run_a_metrics,
            "run_b_metrics": self.run_b_metrics,
            "significance": {
                "p_value": round(self.p_value, 6),
                "level": self.significance,
                "overlapping_markets": self.overlapping_markets,
            },
            "config_diff": self.config_diff,
            "category_comparison": self.category_comparison,
        }


def _run_metrics(run: BacktestRunRecord) -> dict[str, Any]:
    return {
        "total_pnl": run.total_pnl,
        "brier_score": run.brier_score,
        "win_rate": run.win_rate,
        "sharpe_ratio": run.sharpe_ratio,
        "max_drawdown_pct": run.max_drawdown_pct,
        "markets_processed": run.markets_processed,
        "markets_traded": run.markets_traded,
    }


def _significance_level(p_value: float) -> str:
    if p_value < 0.01:
        return "strong"
    elif p_value < 0.05:
        return "moderate"
    elif p_value < 0.1:
        return "weak"
    return "none"


def _paired_ttest(values_a: list[float], values_b: list[float]) -> float:
    """Paired t-test returning p-value.

    Uses scipy if available, otherwise a manual implementation.
    Assumes equal-length inputs of paired observations.
    """
    n = len(values_a)
    if n < 2:
        return 1.0

    # Differences
    diffs = [b - a for a, b in zip(values_a, values_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)

    if var_d == 0.0:
        return 1.0 if mean_d == 0.0 else 0.0

    se = math.sqrt(var_d / n)
    t_stat = mean_d / se
    df = n - 1

    # Try scipy for exact p-value
    try:
        from scipy.stats import t as t_dist
        p_value = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df))
        return p_value
    except ImportError:
        pass

    # Manual approximation using the t-distribution CDF
    # (simplified using normal approximation for large df)
    if df > 30:
        # For large df, t ≈ normal
        p_value = 2.0 * _normal_sf(abs(t_stat))
        return p_value

    # Small-sample: use incomplete beta approximation
    x = df / (df + t_stat ** 2)
    p_value = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return p_value


def _normal_sf(x: float) -> float:
    """Survival function of the standard normal (1 - CDF)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximate regularized incomplete beta function for t-test p-value.

    Uses a continued fraction expansion (Lentz's method).
    """
    if x < 0 or x > 1:
        return 0.0
    if x == 0.0 or x == 1.0:
        return x

    # Use the continued fraction for I_x(a, b)
    # This is sufficient for our t-test needs
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        math.log(x) * a + math.log(1 - x) * b - ln_beta
    ) / a

    # Lentz's method for the continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # Even step
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return front * f


def _diff_configs(a_json: str, b_json: str) -> dict[str, Any]:
    """Find differences between two config JSON strings."""
    try:
        a = json.loads(a_json) if a_json else {}
        b = json.loads(b_json) if b_json else {}
    except json.JSONDecodeError:
        return {}

    diff: dict[str, Any] = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in sorted(all_keys):
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            diff[key] = {"a": va, "b": vb}
    return diff


class BacktestComparator:
    """Compare two backtest runs with statistical significance."""

    def __init__(self, db: BacktestDatabase):
        self._db = db

    def compare(self, run_id_a: str, run_id_b: str) -> ComparisonResult:
        """Compare two backtest runs.

        Returns a ComparisonResult with metric deltas, statistical
        significance from a paired t-test on overlapping markets,
        and config differences.
        """
        run_a = self._db.get_backtest_run(run_id_a)
        run_b = self._db.get_backtest_run(run_id_b)

        if not run_a or not run_b:
            missing = run_id_a if not run_a else run_id_b
            raise ValueError(f"Backtest run not found: {missing}")

        trades_a = self._db.get_backtest_trades(run_id_a)
        trades_b = self._db.get_backtest_trades(run_id_b)

        result = ComparisonResult(
            run_a_id=run_id_a,
            run_a_name=run_a.name,
            run_b_id=run_id_b,
            run_b_name=run_b.name,
            run_a_metrics=_run_metrics(run_a),
            run_b_metrics=_run_metrics(run_b),
        )

        # Metric deltas (B - A)
        result.pnl_delta = run_b.total_pnl - run_a.total_pnl
        result.brier_delta = run_b.brier_score - run_a.brier_score
        result.sharpe_delta = run_b.sharpe_ratio - run_a.sharpe_ratio
        result.win_rate_delta = run_b.win_rate - run_a.win_rate
        result.max_drawdown_delta = run_b.max_drawdown_pct - run_a.max_drawdown_pct

        # Config diff
        result.config_diff = _diff_configs(run_a.config_json, run_b.config_json)

        # Paired t-test on overlapping markets
        pnl_a_by_market = {t.market_condition_id: t.pnl for t in trades_a}
        pnl_b_by_market = {t.market_condition_id: t.pnl for t in trades_b}
        overlap = set(pnl_a_by_market.keys()) & set(pnl_b_by_market.keys())
        result.overlapping_markets = len(overlap)

        if len(overlap) >= 2:
            sorted_ids = sorted(overlap)
            vals_a = [pnl_a_by_market[mid] for mid in sorted_ids]
            vals_b = [pnl_b_by_market[mid] for mid in sorted_ids]
            result.p_value = _paired_ttest(vals_a, vals_b)
            result.significance = _significance_level(result.p_value)

        # Per-category comparison
        result.category_comparison = self._compare_categories(trades_a, trades_b)

        return result

    @staticmethod
    def _compare_categories(
        trades_a: list[BacktestTradeRecord],
        trades_b: list[BacktestTradeRecord],
    ) -> list[dict[str, Any]]:
        """Compare per-category stats between two runs."""
        def _cat_stats(trades: list[BacktestTradeRecord]) -> dict[str, dict[str, Any]]:
            by_cat: dict[str, list[float]] = {}
            for t in trades:
                cat = t.category or "UNKNOWN"
                by_cat.setdefault(cat, []).append(t.pnl)
            return {
                cat: {
                    "count": len(pnls),
                    "total_pnl": round(sum(pnls), 2),
                    "win_rate": round(
                        len([p for p in pnls if p > 0]) / len(pnls), 4
                    ) if pnls else 0.0,
                }
                for cat, pnls in by_cat.items()
            }

        cats_a = _cat_stats(trades_a)
        cats_b = _cat_stats(trades_b)
        all_cats = sorted(set(cats_a.keys()) | set(cats_b.keys()))

        result: list[dict[str, Any]] = []
        for cat in all_cats:
            a = cats_a.get(cat, {"count": 0, "total_pnl": 0.0, "win_rate": 0.0})
            b = cats_b.get(cat, {"count": 0, "total_pnl": 0.0, "win_rate": 0.0})
            result.append({
                "category": cat,
                "run_a": a,
                "run_b": b,
                "pnl_delta": round(b["total_pnl"] - a["total_pnl"], 2),
            })
        return result
