"""Conditional market graph — Dutch book detection across related markets.

Detects monotonicity violations: if P(A) > P(B) when A is a strict subset
of B (e.g., "Will X happen by June?" vs "Will X happen by September?"),
one market is guaranteed mispriced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class MarketNode:
    """A market in the conditional graph."""
    id: str
    question: str
    prob: float
    event_slug: str = ""
    end_date: str = ""


@dataclass
class ConditionalViolation:
    """A detected conditional probability violation."""
    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    market_a_prob: float       # should be <= market_b_prob
    market_b_prob: float
    implied_conditional: float  # P(A) / P(B) — should be in [0, 1]
    violation_type: str         # 'impossible' | 'implausible'
    edge: float
    actionable: bool


class MarketGraph:
    """Graph of related markets for conditional arbitrage detection."""

    def __init__(self, fee_cost: float = 0.04) -> None:
        self._nodes: dict[str, MarketNode] = {}
        self._subset_pairs: list[tuple[MarketNode, MarketNode]] = []
        self._fee_cost = fee_cost

    def build_from_markets(self, markets: list[Any]) -> None:
        """Build graph from market data, grouping by event_slug."""
        self._nodes.clear()
        self._subset_pairs.clear()

        # Group markets by event
        events: dict[str, list[MarketNode]] = {}
        for m in markets:
            mid = getattr(m, "id", "") or getattr(m, "condition_id", "")
            question = getattr(m, "question", "") or ""
            prob = getattr(m, "implied_probability", 0.5)
            slug = getattr(m, "event_slug", "") or getattr(m, "slug", "") or ""
            end_date = getattr(m, "end_date", "") or ""

            if not mid:
                continue

            node = MarketNode(
                id=mid, question=question, prob=prob,
                event_slug=slug, end_date=end_date,
            )
            self._nodes[mid] = node

            if slug:
                events.setdefault(slug, []).append(node)

        # Find subset pairs within each event group
        for slug, nodes in events.items():
            if len(nodes) < 2:
                continue
            # Sort by end_date — earlier deadline is subset of later
            sorted_nodes = sorted(nodes, key=lambda n: n.end_date)
            for i, a in enumerate(sorted_nodes):
                for b in sorted_nodes[i + 1:]:
                    if a.end_date and b.end_date and a.end_date < b.end_date:
                        self._subset_pairs.append((a, b))

    def find_monotonicity_violations(self) -> list[ConditionalViolation]:
        """P(A) > P(B) when A is a subset of B is a guaranteed Dutch book."""
        violations = []
        for a, b in self._subset_pairs:
            if a.prob > b.prob:
                edge = a.prob - b.prob - self._fee_cost
                violations.append(ConditionalViolation(
                    market_a_id=a.id,
                    market_b_id=b.id,
                    market_a_question=a.question,
                    market_b_question=b.question,
                    market_a_prob=a.prob,
                    market_b_prob=b.prob,
                    implied_conditional=a.prob / max(b.prob, 0.001),
                    violation_type="impossible",
                    edge=edge,
                    actionable=edge > 0.01,
                ))
                log.info(
                    "market_graph.violation",
                    a_id=a.id, b_id=b.id,
                    a_prob=round(a.prob, 3), b_prob=round(b.prob, 3),
                    edge=round(edge, 4),
                )
        return violations

    def find_conditional_violations(self) -> list[ConditionalViolation]:
        """Find all conditional probability violations (superset of monotonicity)."""
        return self.find_monotonicity_violations()
