"""Execution strategy auto-selection based on liquidity and historical quality.

Rules:
  1. Thin liquidity (depth < threshold) -> iceberg to minimize market impact
  2. Large order (> pct of depth) -> TWAP to spread execution
  3. Learning mode: pick strategy with best historical slippage + fill rate
  4. Default: simple limit order

Disabled by default — gated behind execution.auto_strategy_selection_enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class StrategyRecommendation:
    """Recommended execution strategy with reasoning."""

    strategy: str               # "simple" | "twap" | "iceberg"
    reason: str
    confidence: float           # 0-1
    alternative: str = ""
    depth_usd: float = 0.0
    order_size_usd: float = 0.0
    order_pct_of_depth: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "alternative": self.alternative,
            "depth_usd": round(self.depth_usd, 2),
            "order_size_usd": round(self.order_size_usd, 2),
            "order_pct_of_depth": round(self.order_pct_of_depth, 4),
        }


class ExecutionStrategySelector:
    """Auto-selects execution strategy based on liquidity and performance."""

    def __init__(
        self,
        thin_depth_usd: float = 5000.0,
        large_order_pct: float = 0.10,
        learning_enabled: bool = False,
        min_samples: int = 10,
    ):
        self._thin_depth = thin_depth_usd
        self._large_order_pct = large_order_pct
        self._learning_enabled = learning_enabled
        self._min_samples = min_samples

    def select(
        self,
        order_size_usd: float,
        depth_usd: float,
        historical_quality: Any = None,
    ) -> StrategyRecommendation:
        """Select optimal execution strategy.

        Args:
            order_size_usd: Size of the order in USD.
            depth_usd: Total visible orderbook depth in USD.
            historical_quality: ExecutionQuality instance (optional).
        """
        order_pct = (
            order_size_usd / depth_usd if depth_usd > 0 else float("inf")
        )

        # Rule 1: Thin liquidity -> iceberg
        if depth_usd < self._thin_depth:
            strategy = "iceberg"
            reason = (
                f"Thin liquidity (${depth_usd:.0f} < ${self._thin_depth:.0f} threshold)"
            )
            alternative = "simple"
            confidence = 0.85

            log.info(
                "strategy_selector.selected",
                strategy=strategy,
                reason=reason,
                depth_usd=round(depth_usd, 2),
                order_size_usd=round(order_size_usd, 2),
            )
            return StrategyRecommendation(
                strategy=strategy,
                reason=reason,
                confidence=confidence,
                alternative=alternative,
                depth_usd=depth_usd,
                order_size_usd=order_size_usd,
                order_pct_of_depth=order_pct,
            )

        # Rule 2: Large order -> TWAP
        if order_pct > self._large_order_pct:
            strategy = "twap"
            reason = (
                f"Large order ({order_pct:.1%} of depth > "
                f"{self._large_order_pct:.0%} threshold)"
            )
            alternative = "iceberg"
            confidence = 0.80

            log.info(
                "strategy_selector.selected",
                strategy=strategy,
                reason=reason,
                order_pct=round(order_pct, 4),
            )
            return StrategyRecommendation(
                strategy=strategy,
                reason=reason,
                confidence=confidence,
                alternative=alternative,
                depth_usd=depth_usd,
                order_size_usd=order_size_usd,
                order_pct_of_depth=order_pct,
            )

        # Rule 3: Learning mode — use historical fill quality
        if self._learning_enabled and historical_quality is not None:
            learned = self._learn_from_quality(historical_quality)
            if learned is not None:
                log.info(
                    "strategy_selector.learned",
                    strategy=learned,
                    depth_usd=round(depth_usd, 2),
                )
                return StrategyRecommendation(
                    strategy=learned,
                    reason=f"Learning mode: {learned} has best historical fill quality",
                    confidence=0.70,
                    alternative="simple",
                    depth_usd=depth_usd,
                    order_size_usd=order_size_usd,
                    order_pct_of_depth=order_pct,
                )

        # Rule 4: Default -> simple
        log.info(
            "strategy_selector.default",
            depth_usd=round(depth_usd, 2),
            order_pct=round(order_pct, 4),
        )
        return StrategyRecommendation(
            strategy="simple",
            reason="Normal conditions — standard limit order",
            confidence=0.75,
            alternative="",
            depth_usd=depth_usd,
            order_size_usd=order_size_usd,
            order_pct_of_depth=order_pct,
        )

    def _learn_from_quality(self, quality: Any) -> str | None:
        """Use historical fill quality to recommend a strategy.

        Returns strategy name if learning produces a clear winner, None otherwise.
        """
        stats = getattr(quality, "strategy_stats", {})
        if not stats:
            return None

        # Require minimum samples
        best_strategy = None
        best_score = float("inf")

        for strat, strat_data in stats.items():
            count = strat_data.get("count", 0)
            if count < self._min_samples:
                continue

            fill_rate = strat_data.get("avg_fill_rate", 0)
            if fill_rate < 0.80:
                continue  # Skip strategies with poor fill rates

            avg_slippage = abs(strat_data.get("avg_slippage_bps", 0))
            if avg_slippage < best_score:
                best_score = avg_slippage
                best_strategy = strat

        return best_strategy
