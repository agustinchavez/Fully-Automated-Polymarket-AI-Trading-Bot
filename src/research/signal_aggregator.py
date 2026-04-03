"""Signal aggregator — collects consensus and behavioral signals into a SignalStack.

Extracts ``consensus_signal`` and ``behavioral_signal`` dicts from
``FetchedSource.raw`` and assembles them into a structured ``SignalStack``
dataclass.  Provides:
- ``build_signal_stack()`` — populate from sources + Polymarket price
- ``render_signal_stack()`` — format for LLM prompt injection
- ``compute_signal_confluence()`` — Kelly sizing multiplier [0.25–1.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)


@dataclass
class SignalStack:
    """Aggregated consensus + behavioral signals for a market."""

    # ── Consensus signals ─────────────────────────────────────────
    kalshi_price: float | None = None
    kalshi_spread_pp: float | None = None
    metaculus_probability: float | None = None
    metaculus_forecasters: int = 0

    # ── Behavioral signals ────────────────────────────────────────
    wikipedia_spike_ratio: float | None = None
    wikipedia_article: str = ""
    google_trends_index: float | None = None
    google_trends_spike_ratio: float | None = None
    google_trends_narrative: str = ""
    reddit_sentiment: float | None = None
    reddit_post_count: int = 0

    # ── Computed ──────────────────────────────────────────────────
    consensus_divergence: float = 0.0
    recommended_kelly_multiplier: float = 1.0


def build_signal_stack(
    sources: list[FetchedSource],
    poly_price: float,
) -> SignalStack:
    """Build a ``SignalStack`` from research sources and the Polymarket price.

    Iterates through sources looking for ``raw["consensus_signal"]`` and
    ``raw["behavioral_signal"]`` dicts, populating the appropriate fields.
    """
    stack = SignalStack()

    consensus_prices: list[float] = []

    for source in sources:
        raw = source.raw or {}

        # ── Consensus signals ─────────────────────────────────────
        cs = raw.get("consensus_signal")
        if cs and isinstance(cs, dict):
            platform = cs.get("platform", "")
            price = cs.get("price")

            if platform == "kalshi" and price is not None:
                stack.kalshi_price = float(price)
                stack.kalshi_spread_pp = cs.get("spread_pp")
                consensus_prices.append(float(price))

            elif platform == "metaculus" and price is not None:
                stack.metaculus_probability = float(price)
                stack.metaculus_forecasters = cs.get("forecasters", 0)
                consensus_prices.append(float(price))

        # ── Behavioral signals ────────────────────────────────────
        bs = raw.get("behavioral_signal")
        if bs and isinstance(bs, dict):
            sig_source = bs.get("source", "")
            sig_type = bs.get("signal_type", "")

            if sig_source == "wikipedia" and sig_type == "attention_spike":
                stack.wikipedia_spike_ratio = bs.get("value")
                stack.wikipedia_article = bs.get("article", "")

            elif sig_source == "google_trends" and sig_type == "search_trend":
                stack.google_trends_spike_ratio = bs.get("value")
                stack.google_trends_index = bs.get("current_index")
                stack.google_trends_narrative = bs.get("narrative", "")

            elif sig_source == "reddit" and sig_type == "sentiment":
                stack.reddit_sentiment = bs.get("value")
                stack.reddit_post_count = bs.get("post_count", 0)

    # ── Compute consensus divergence ──────────────────────────────
    if consensus_prices:
        max_div = max(abs(p - poly_price) for p in consensus_prices)
        stack.consensus_divergence = round(max_div, 4)

    # ── Compute recommended Kelly multiplier ──────────────────────
    stack.recommended_kelly_multiplier = compute_signal_confluence(
        stack, poly_price,
    )

    return stack


def render_signal_stack(stack: SignalStack) -> str:
    """Render the signal stack as a prompt block for LLM injection.

    Returns an empty string if no signals are present.
    """
    sections: list[str] = []

    # ── Consensus section ─────────────────────────────────────────
    consensus_lines: list[str] = []
    if stack.kalshi_price is not None:
        spread_str = (
            f" (spread: {stack.kalshi_spread_pp:.1f}pp)"
            if stack.kalshi_spread_pp is not None
            else ""
        )
        consensus_lines.append(
            f"- Kalshi mid-price: {stack.kalshi_price:.1%}{spread_str}"
        )
    if stack.metaculus_probability is not None:
        consensus_lines.append(
            f"- Metaculus community: {stack.metaculus_probability:.1%}"
            f" ({stack.metaculus_forecasters} forecasters)"
        )

    if consensus_lines:
        sections.append(
            "CONSENSUS SIGNALS:\n" + "\n".join(consensus_lines)
        )

    # ── Behavioral section ────────────────────────────────────────
    behavioral_lines: list[str] = []
    if stack.wikipedia_spike_ratio is not None:
        article_str = (
            f" ({stack.wikipedia_article.replace('_', ' ')})"
            if stack.wikipedia_article
            else ""
        )
        behavioral_lines.append(
            f"- Wikipedia attention: {stack.wikipedia_spike_ratio:.1f}x spike{article_str}"
        )
    if stack.google_trends_spike_ratio is not None:
        idx_str = (
            f", index={stack.google_trends_index:.0f}"
            if stack.google_trends_index is not None
            else ""
        )
        behavioral_lines.append(
            f"- Google Trends: {stack.google_trends_spike_ratio:.1f}x spike{idx_str}"
        )
        if stack.google_trends_narrative:
            behavioral_lines.append(
                f"  Context: {stack.google_trends_narrative[:200]}"
            )
    if stack.reddit_sentiment is not None:
        direction = (
            "bullish" if stack.reddit_sentiment > 0.1
            else "bearish" if stack.reddit_sentiment < -0.1
            else "neutral"
        )
        behavioral_lines.append(
            f"- Reddit sentiment: {stack.reddit_sentiment:+.2f} ({direction},"
            f" {stack.reddit_post_count} posts)"
        )

    if behavioral_lines:
        sections.append(
            "BEHAVIORAL SIGNALS:\n" + "\n".join(behavioral_lines)
        )

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"


def compute_signal_confluence(
    stack: SignalStack,
    poly_price: float,
) -> float:
    """Compute a Kelly sizing multiplier based on signal agreement/divergence.

    Returns a value in [0.25, 1.0]:
    - 1.0  → signals agree with Polymarket (or no signals present)
    - 0.25 → signals strongly diverge from Polymarket (≥20pp)

    Logic:
    1. If no consensus signals → return 1.0 (no information to conflict)
    2. Compute average consensus price across available platforms
    3. Compare to poly_price:
       - divergence < 5pp  → 1.0  (agreement)
       - divergence 5–10pp → 0.75 (mild disagreement)
       - divergence 10–15pp → 0.50 (moderate disagreement)
       - divergence 15–20pp → 0.35 (strong disagreement)
       - divergence ≥ 20pp → 0.25 (severe disagreement)
    """
    consensus_prices: list[float] = []
    if stack.kalshi_price is not None:
        consensus_prices.append(stack.kalshi_price)
    if stack.metaculus_probability is not None:
        consensus_prices.append(stack.metaculus_probability)

    if not consensus_prices:
        return 1.0

    avg_consensus = sum(consensus_prices) / len(consensus_prices)
    divergence_pp = abs(avg_consensus - poly_price) * 100  # percentage points

    if divergence_pp < 5:
        return 1.0
    elif divergence_pp < 10:
        return 0.75
    elif divergence_pp < 15:
        return 0.50
    elif divergence_pp < 20:
        return 0.35
    else:
        return 0.25
