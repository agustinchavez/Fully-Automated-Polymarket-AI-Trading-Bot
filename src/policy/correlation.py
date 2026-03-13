"""Event correlation modeling — pairwise correlation scoring for positions.

Uses event_slug grouping from the Polymarket Gamma API to identify
correlated positions. Markets within the same event are highly correlated;
markets in the same category have mild correlation; unrelated markets
are independent.

Correlation scores:
  - Same event, same side (both YES or both NO): 0.8
  - Same event, different sides: 0.3
  - Same category, different event: 0.15
  - Unrelated: 0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class CorrelationPair:
    """Pairwise correlation between two positions."""
    market_id_a: str
    market_id_b: str
    correlation: float
    source: str  # same_event_same_side | same_event_diff_side | same_category | unrelated


class EventCorrelationScorer:
    """Compute correlation scores between positions based on event grouping."""

    def __init__(self, config: Any | None = None):
        if config is not None:
            self._same_event_same_side = getattr(config, "same_event_same_outcome_corr", 0.8)
            self._same_event_diff_side = getattr(config, "same_event_diff_outcome_corr", 0.3)
            self._same_category = getattr(config, "same_category_corr", 0.15)
        else:
            self._same_event_same_side = 0.8
            self._same_event_diff_side = 0.3
            self._same_category = 0.15

    def _pairwise_correlation(
        self,
        a_event: str,
        a_side: str,
        a_category: str,
        b_event: str,
        b_side: str,
        b_category: str,
    ) -> tuple[float, str]:
        """Compute correlation and source for a pair of positions."""
        if a_event and b_event and a_event == b_event:
            if a_side == b_side:
                return self._same_event_same_side, "same_event_same_side"
            else:
                return self._same_event_diff_side, "same_event_diff_side"
        if a_category and b_category and a_category == b_category:
            return self._same_category, "same_category"
        return 0.0, "unrelated"

    def compute_pairwise(
        self,
        positions: list[Any],
    ) -> list[CorrelationPair]:
        """Compute pairwise correlations for all position pairs."""
        pairs: list[CorrelationPair] = []
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = positions[i], positions[j]
                corr, source = self._pairwise_correlation(
                    a.event_slug, a.side, a.category,
                    b.event_slug, b.side, b.category,
                )
                pairs.append(CorrelationPair(
                    market_id_a=a.market_id,
                    market_id_b=b.market_id,
                    correlation=corr,
                    source=source,
                ))
        return pairs

    def build_correlation_matrix(
        self,
        positions: list[Any],
    ) -> tuple[list[str], list[list[float]]]:
        """Build NxN correlation matrix.

        Returns:
            (market_ids, matrix) where matrix[i][j] is the correlation
            between position i and position j. Diagonal = 1.0.
        """
        n = len(positions)
        ids = [p.market_id for p in positions]
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                a, b = positions[i], positions[j]
                corr, _ = self._pairwise_correlation(
                    a.event_slug, a.side, a.category,
                    b.event_slug, b.side, b.category,
                )
                matrix[i][j] = corr
                matrix[j][i] = corr

        return ids, matrix

    def score_new_position(
        self,
        existing: list[Any],
        new_event_slug: str,
        new_side: str,
        new_category: str,
    ) -> float:
        """Compute the max correlation between a proposed new position
        and existing positions. Returns 0-1."""
        if not existing:
            return 0.0

        max_corr = 0.0
        for pos in existing:
            corr, _ = self._pairwise_correlation(
                pos.event_slug, pos.side, pos.category,
                new_event_slug, new_side, new_category,
            )
            max_corr = max(max_corr, corr)

        return max_corr
