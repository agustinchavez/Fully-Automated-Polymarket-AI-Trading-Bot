"""Capital efficiency — rank markets by annualized edge.

A 5% edge on a market resolving in 3 days is far more valuable than
the same edge on a market resolving in 90 days. This module computes
annualized edge and ranks qualifying markets.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class RankedMarket:
    """Market ranked by capital efficiency."""
    market_id: str
    annualized_edge: float
    net_edge: float
    days_to_resolution: float
    rank: int = 0


def compute_annualized_edge(net_edge: float, days_to_resolution: float) -> float:
    """Compute annualized edge = edge / (days / 365).

    Examples:
      5% edge, 7 days  → 260.7%/yr
      5% edge, 90 days → 20.3%/yr

    Returns float('inf') if days_to_resolution <= 0 (resolves immediately).
    """
    if days_to_resolution <= 0:
        return float("inf")
    return net_edge / (days_to_resolution / 365.0)


def rank_markets_by_efficiency(
    candidates: list[dict[str, float]],
    min_annualized_edge: float = 0.50,
) -> list[RankedMarket]:
    """Rank qualifying markets by annualized edge, descending.

    Args:
        candidates: List of dicts with keys: market_id, net_edge, days_to_resolution.
        min_annualized_edge: Filter out markets below this annualized edge.

    Returns:
        Ranked list of RankedMarket objects, best first.
    """
    ranked: list[RankedMarket] = []
    for c in candidates:
        mid = c.get("market_id", "")
        ne = c.get("net_edge", 0.0)
        days = c.get("days_to_resolution", 0.0)
        ann = compute_annualized_edge(ne, days)
        if ann >= min_annualized_edge:
            ranked.append(RankedMarket(
                market_id=mid,
                annualized_edge=ann,
                net_edge=ne,
                days_to_resolution=days,
            ))

    ranked.sort(key=lambda x: x.annualized_edge, reverse=True)
    for i, rm in enumerate(ranked):
        rm.rank = i + 1

    if ranked:
        log.info(
            "capital_efficiency.ranked",
            total=len(ranked),
            best_market=ranked[0].market_id,
            best_annualized=round(ranked[0].annualized_edge, 3),
        )

    return ranked
