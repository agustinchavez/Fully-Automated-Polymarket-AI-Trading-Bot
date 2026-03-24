"""Linked-market / arbitrage detection.

Identifies markets that reference the same event and finds
pricing inconsistencies that can be exploited. Examples:

  - "Will X happen before Y?" and "Will Y happen before X?"
    should sum to ~1.0 after vig
  - Multi-outcome markets where all options should sum to ~1.0
  - Correlated markets with divergent pricing
  - Complementary binary markets where YES+NO < 1.0 (guaranteed profit)
  - Correlated event mispricings (implausible implied conditionals)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.connectors.polymarket_gamma import GammaMarket
from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    """An identified pricing inconsistency."""
    market_ids: list[str]
    questions: list[str]
    implied_probs: list[float]
    prob_sum: float  # should be ~1.0 for complementary markets
    arb_edge: float  # how much the sum deviates from 1.0
    arb_type: str  # "complementary", "multi_outcome", "correlated"
    description: str
    is_actionable: bool  # True if edge > transaction costs

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def detect_arbitrage(
    markets: list[GammaMarket],
    fee_bps: int = 200,
) -> list[ArbitrageOpportunity]:
    """Scan a list of markets for arbitrage opportunities.

    Strategies:
    1. Complementary binary markets (same event, opposite outcomes)
    2. Multi-outcome markets (all outcomes should sum to ~1.0)
    3. Similar-question markets with different pricing
    """
    opportunities: list[ArbitrageOpportunity] = []

    # Strategy 1: Group by event_slug and find complementary pairs
    event_groups: dict[str, list[GammaMarket]] = {}
    for m in markets:
        slug = m.slug.rsplit("-", 1)[0] if m.slug else m.id
        event_groups.setdefault(slug, []).append(m)

    for slug, group in event_groups.items():
        if len(group) < 2:
            continue

        # Check if probabilities sum correctly
        probs = []
        for m in group:
            yes_tokens = [t for t in m.tokens if t.outcome.lower() == "yes"]
            if yes_tokens:
                probs.append(yes_tokens[0].price)
            else:
                probs.append(m.best_bid)

        prob_sum = sum(probs)
        fee_cost = fee_bps / 10000 * len(group)

        # For complementary markets, sum should be ~1.0
        # If sum differs significantly, there's an opportunity
        deviation = abs(prob_sum - 1.0)
        edge = deviation - fee_cost

        if edge > 0.01 and len(group) >= 2:
            opportunities.append(ArbitrageOpportunity(
                market_ids=[m.id for m in group],
                questions=[m.question for m in group],
                implied_probs=probs,
                prob_sum=prob_sum,
                arb_edge=edge,
                arb_type="complementary",
                description=(
                    f"Event '{slug}': {len(group)} markets sum to "
                    f"{prob_sum:.3f} (deviation: {deviation:.3f}, "
                    f"net edge after fees: {edge:.3f})"
                ),
                is_actionable=edge > 0.02,
            ))

    # Strategy 2: Multi-outcome (markets with >2 tokens)
    for m in markets:
        if len(m.tokens) > 2:
            token_probs = [t.price for t in m.tokens]
            prob_sum = sum(token_probs)
            deviation = abs(prob_sum - 1.0)
            fee_cost = fee_bps / 10000 * 2  # buy/sell
            edge = deviation - fee_cost

            if edge > 0.01:
                opportunities.append(ArbitrageOpportunity(
                    market_ids=[m.id],
                    questions=[m.question],
                    implied_probs=token_probs,
                    prob_sum=prob_sum,
                    arb_edge=edge,
                    arb_type="multi_outcome",
                    description=(
                        f"Multi-outcome '{m.question[:60]}': "
                        f"{len(m.tokens)} outcomes sum to {prob_sum:.3f} "
                        f"(edge: {edge:.3f})"
                    ),
                    is_actionable=edge > 0.02,
                ))

    # Strategy 3: Similar questions with divergent pricing
    # Simple keyword-based similarity check
    _check_similar_questions(markets, opportunities, fee_bps)

    if opportunities:
        log.info(
            "arbitrage.detected",
            num_opportunities=len(opportunities),
            actionable=sum(1 for o in opportunities if o.is_actionable),
        )

    return sorted(opportunities, key=lambda x: x.arb_edge, reverse=True)


def _check_similar_questions(
    markets: list[GammaMarket],
    opportunities: list[ArbitrageOpportunity],
    fee_bps: int,
) -> None:
    """Find markets with similar questions but different prices."""
    # Extract key entities from questions
    market_entities: list[tuple[GammaMarket, set[str]]] = []
    for m in markets:
        words = set(m.question.lower().split())
        # Remove common words
        stop_words = {
            "will", "the", "a", "an", "be", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "by", "before", "after",
            "this", "that", "or", "and", "not", "no", "yes", "?", "how",
            "what", "when", "where", "which", "who",
        }
        entities = words - stop_words
        if len(entities) >= 2:
            market_entities.append((m, entities))

    # Compare pairs (O(n^2) but typically small n)
    seen = set()
    for i, (m1, e1) in enumerate(market_entities):
        for j, (m2, e2) in enumerate(market_entities):
            if i >= j:
                continue
            pair_key = (m1.id, m2.id)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Jaccard similarity
            intersection = e1 & e2
            union = e1 | e2
            if not union:
                continue
            similarity = len(intersection) / len(union)

            if similarity >= 0.5:
                # These markets are about similar topics — check price divergence
                p1 = m1.best_bid
                p2 = m2.best_bid
                price_diff = abs(p1 - p2)
                fee_cost = fee_bps / 10000 * 2

                if price_diff > fee_cost + 0.03:
                    opportunities.append(ArbitrageOpportunity(
                        market_ids=[m1.id, m2.id],
                        questions=[m1.question, m2.question],
                        implied_probs=[p1, p2],
                        prob_sum=p1 + p2,
                        arb_edge=price_diff - fee_cost,
                        arb_type="correlated",
                        description=(
                            f"Similar markets with {price_diff:.3f} price gap: "
                            f"'{m1.question[:40]}' ({p1:.2f}) vs "
                            f"'{m2.question[:40]}' ({p2:.2f})"
                        ),
                        is_actionable=price_diff > fee_cost + 0.05,
                    ))


# ── Complementary Binary Market Arbitrage ───────────────────────────


@dataclass
class ComplementaryArbOpportunity:
    """A single market where YES + NO < 1.0, offering guaranteed profit."""
    market_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    yes_price: float
    no_price: float
    combined_cost: float           # yes_price + no_price
    guaranteed_profit: float       # 1.0 - combined_cost
    fee_cost: float                # total fees for buying both sides
    net_profit: float              # guaranteed_profit - fee_cost
    is_actionable: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def detect_complementary_arb(
    markets: list[GammaMarket],
    threshold: float = 0.97,
    fee_bps: int = 200,
) -> list[ComplementaryArbOpportunity]:
    """Find markets where YES + NO < threshold (buy both → guaranteed profit).

    In a properly priced binary market, YES + NO = 1.0 (before vig).
    If YES + NO < 1.0, buying both sides guarantees profit at resolution:
        - One side resolves to $1.00, other to $0.00
        - Total payout = $1.00, cost = YES + NO < $1.00

    Args:
        markets: List of Polymarket markets with YES/NO tokens.
        threshold: Only flag markets where combined cost < threshold.
        fee_bps: Total fee in basis points (buy YES + buy NO).

    Returns:
        Sorted list of ComplementaryArbOpportunity by net_profit descending.
    """
    opportunities: list[ComplementaryArbOpportunity] = []
    fee_cost = fee_bps / 10000 * 2  # Fee for two trades (buy YES + buy NO)

    for m in markets:
        if len(m.tokens) != 2:
            continue

        yes_token = None
        no_token = None
        for t in m.tokens:
            outcome = t.outcome.lower()
            if outcome == "yes":
                yes_token = t
            elif outcome == "no":
                no_token = t

        if not yes_token or not no_token:
            continue

        combined_cost = yes_token.price + no_token.price
        if combined_cost >= threshold:
            continue

        guaranteed_profit = 1.0 - combined_cost
        net_profit = guaranteed_profit - fee_cost

        opportunities.append(ComplementaryArbOpportunity(
            market_id=m.condition_id or m.id,
            question=m.question,
            yes_token_id=yes_token.token_id,
            no_token_id=no_token.token_id,
            yes_price=yes_token.price,
            no_price=no_token.price,
            combined_cost=round(combined_cost, 4),
            guaranteed_profit=round(guaranteed_profit, 4),
            fee_cost=round(fee_cost, 4),
            net_profit=round(net_profit, 4),
            is_actionable=net_profit > 0,
        ))

    if opportunities:
        log.info(
            "arbitrage.complementary_detected",
            total=len(opportunities),
            actionable=sum(1 for o in opportunities if o.is_actionable),
        )

    return sorted(opportunities, key=lambda o: o.net_profit, reverse=True)


# ── Correlated Event Mispricing ─────────────────────────────────────

# Stop words for entity extraction (shared with market_matcher.py)
_STOP_WORDS: set[str] = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of", "by", "before", "after",
    "this", "that", "or", "and", "not", "no", "yes", "?", "how",
    "what", "when", "where", "which", "who", "do", "does", "has",
    "have", "been", "being", "if", "it", "its", "than", "then",
    "can", "could", "would", "should", "may", "might",
}


@dataclass
class CorrelatedMispricing:
    """A pair of related markets with implausible implied conditionals."""
    primary_market_id: str
    primary_question: str
    primary_prob: float
    secondary_market_id: str
    secondary_question: str
    secondary_prob: float
    implied_conditional: float     # P(secondary | primary) implied by prices
    divergence: float              # how far the conditional is from [0, 1]
    explanation: str
    is_actionable: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def _extract_entities(text: str) -> set[str]:
    """Extract meaningful entities from market question text."""
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s%$.]", " ", normalized)
    words = set(normalized.split()) - _STOP_WORDS
    return {w for w in words if len(w) >= 2}


def detect_correlated_mispricings(
    markets: list[GammaMarket],
    min_divergence: float = 0.10,
) -> list[CorrelatedMispricing]:
    """Find pairs of related markets with implausible implied probabilities.

    Uses entity overlap to identify related markets, then checks whether
    the price relationship makes logical sense. For example:
        - "Will Biden win?" at 40%
        - "Will a Democrat win?" at 55%
        → P(Biden wins | Democrat wins) = 40%/55% = 72.7%
        → P(Democrat wins | Biden wins) = 100% (subset relationship)

    If a more specific event has higher probability than its superset,
    that's a mispricing (divergence > 0).

    Args:
        markets: List of Polymarket markets.
        min_divergence: Minimum probability divergence to flag.

    Returns:
        Sorted list of CorrelatedMispricing by divergence descending.
    """
    mispricings: list[CorrelatedMispricing] = []

    # Pre-compute entities and YES prices
    market_data: list[tuple[GammaMarket, str, float, set[str]]] = []
    for m in markets:
        mid = m.condition_id or m.id
        # Get YES price
        yes_price = m.best_bid
        for t in m.tokens:
            if t.outcome.lower() == "yes":
                yes_price = t.price
                break
        entities = _extract_entities(m.question)
        if len(entities) >= 2:
            market_data.append((m, mid, yes_price, entities))

    seen: set[tuple[str, str]] = set()
    for i, (m1, id1, p1, e1) in enumerate(market_data):
        for j, (m2, id2, p2, e2) in enumerate(market_data):
            if i >= j:
                continue
            pair = (id1, id2)
            if pair in seen:
                continue
            seen.add(pair)

            # Check entity overlap (Jaccard)
            intersection = e1 & e2
            union = e1 | e2
            if not union:
                continue
            similarity = len(intersection) / len(union)

            if similarity < 0.3:
                continue

            # One market is a subset of the other if it has more
            # specific entities (superset of entities = subset of events)
            specific_1 = e1 - e2  # entities unique to m1
            specific_2 = e2 - e1  # entities unique to m2

            # Determine which is more specific
            if len(specific_1) > len(specific_2):
                specific_m, specific_id, specific_p, specific_q = m1, id1, p1, m1.question
                general_m, general_id, general_p, general_q = m2, id2, p2, m2.question
            elif len(specific_2) > len(specific_1):
                specific_m, specific_id, specific_p, specific_q = m2, id2, p2, m2.question
                general_m, general_id, general_p, general_q = m1, id1, p1, m1.question
            else:
                # Neither is more specific — check price divergence directly
                price_diff = abs(p1 - p2)
                if price_diff >= min_divergence and similarity >= 0.5:
                    mispricings.append(CorrelatedMispricing(
                        primary_market_id=id1,
                        primary_question=m1.question,
                        primary_prob=p1,
                        secondary_market_id=id2,
                        secondary_question=m2.question,
                        secondary_prob=p2,
                        implied_conditional=0.0,
                        divergence=round(price_diff, 4),
                        explanation=(
                            f"Similar markets with {price_diff:.1%} price gap: "
                            f"'{m1.question[:40]}' ({p1:.0%}) vs "
                            f"'{m2.question[:40]}' ({p2:.0%})"
                        ),
                        is_actionable=price_diff >= min_divergence + 0.05,
                    ))
                continue

            # Specific event is a subset of general event
            # P(specific) should be <= P(general)
            divergence = specific_p - general_p
            if divergence < min_divergence:
                continue

            # Implied conditional: P(specific | general) = P(specific) / P(general)
            implied_cond = specific_p / general_p if general_p > 0.01 else float("inf")

            mispricings.append(CorrelatedMispricing(
                primary_market_id=specific_id,
                primary_question=specific_q,
                primary_prob=specific_p,
                secondary_market_id=general_id,
                secondary_question=general_q,
                secondary_prob=general_p,
                implied_conditional=round(min(implied_cond, 10.0), 4),
                divergence=round(divergence, 4),
                explanation=(
                    f"Specific event '{specific_q[:40]}' ({specific_p:.0%}) is priced "
                    f"higher than general event '{general_q[:40]}' ({general_p:.0%}); "
                    f"implied P(specific|general) = {implied_cond:.1%}"
                ),
                is_actionable=divergence >= min_divergence + 0.05,
            ))

    if mispricings:
        log.info(
            "arbitrage.correlated_mispricings_detected",
            total=len(mispricings),
            actionable=sum(1 for m in mispricings if m.is_actionable),
        )

    return sorted(mispricings, key=lambda m: m.divergence, reverse=True)
