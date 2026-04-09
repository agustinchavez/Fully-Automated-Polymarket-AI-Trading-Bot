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

    # ── Order flow signals (Improvement 2) ────────────────────────
    order_flow_imbalance: float | None = None   # -1=sell-heavy, +1=buy-heavy
    vwap_divergence_pct: float | None = None    # current vs time-weighted avg
    smart_money_ratio: float | None = None      # 0-1, >0.6 = smart money dominant
    whale_net_direction: str = ""               # 'BUY' | 'SELL' | ''
    whale_total_usd: float = 0.0

    # ── Smart money signals (Improvement 4B) ───────────────────
    whale_count: int = 0
    whale_direction: str = ""         # 'BULLISH' | 'BEARISH' | 'MIXED' | ''
    whale_avg_entry: float | None = None
    whale_signal_strength: str = ""   # 'STRONG' | 'MODERATE' | 'WEAK' | ''
    whale_conviction_score: float = 0.0
    whale_best_performer_rate: float | None = None

    # ── Additional consensus signals (Improvement 6) ───────────
    manifold_probability: float | None = None
    manifold_traders: int = 0
    predictit_probability: float | None = None

    # ── Sports consensus (Sports Intelligence Layer) ────────────
    sportsbook_consensus: float | None = None
    sportsbook_spread_pp: float | None = None
    sportsbook_count: int = 0
    sportsbook_sharp_price: float | None = None  # Pinnacle-specific
    sports_context: str = ""  # form/H2H summary

    # ── Spotify charts (CULTURE) ────────────────────────────────
    spotify_artist: str = ""
    spotify_listeners_rank: int | None = None
    spotify_daily_rank: int | None = None
    spotify_monthly_listeners: str = ""

    # ── Kronos crypto price forecast ─────────────────────────────
    kronos_upside_prob: float | None = None
    kronos_volatility_prob: float | None = None
    kronos_symbol: str = ""

    # ── Calendar events (Improvement 8) ────────────────────────
    calendar_events: list[Any] = field(default_factory=list)

    # ── Computed ──────────────────────────────────────────────────
    consensus_divergence: float = 0.0
    recommended_kelly_multiplier: float = 1.0


def build_signal_stack(
    sources: list[FetchedSource],
    poly_price: float,
    micro_signals: Any | None = None,
    conviction_signals: list[Any] | None = None,
    calendar_events: list[Any] | None = None,
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

            elif platform == "manifold" and price is not None:
                stack.manifold_probability = float(price)
                stack.manifold_traders = cs.get("traders", 0)
                consensus_prices.append(float(price))

            elif platform == "predictit" and price is not None:
                stack.predictit_probability = float(price)
                consensus_prices.append(float(price))

            elif platform == "sportsbooks" and price is not None:
                stack.sportsbook_consensus = float(price)
                stack.sportsbook_spread_pp = cs.get("spread_pp")
                stack.sportsbook_count = cs.get("books", 0)
                stack.sportsbook_sharp_price = (
                    float(cs["sharp_book"])
                    if cs.get("sharp_book") is not None
                    else None
                )
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

            elif sig_source == "sports_stats" and sig_type == "sports_context":
                stack.sports_context = source.content or ""

            elif sig_source == "spotify_charts" and sig_type == "chart_position":
                stack.spotify_artist = bs.get("artist", "")
                rank = bs.get("monthly_listeners_rank")
                stack.spotify_listeners_rank = int(rank) if rank is not None else None
                daily = bs.get("daily_chart_rank")
                stack.spotify_daily_rank = int(daily) if daily is not None else None
                stack.spotify_monthly_listeners = bs.get("monthly_listeners", "")

            elif sig_source == "kronos" and sig_type == "crypto_price_forecast":
                stack.kronos_upside_prob = bs.get("upside_probability")
                stack.kronos_volatility_prob = bs.get("volatility_amplification")
                stack.kronos_symbol = bs.get("symbol", "")

    # ── Microstructure signals (Improvement 2) ────────────────────
    if micro_signals is not None:
        imb = getattr(micro_signals, "flow_imbalances", [])
        if imb:
            recent = imb[0] if imb else None
            if recent:
                stack.order_flow_imbalance = getattr(recent, "imbalance_ratio", None)
        stack.vwap_divergence_pct = getattr(micro_signals, "vwap_divergence_pct", None)
        stack.smart_money_ratio = getattr(micro_signals, "smart_money_ratio", None)
        whales = getattr(micro_signals, "whale_alerts", [])
        if whales:
            total = sum(getattr(w, "size_usd", 0) for w in whales)
            buys = sum(
                getattr(w, "size_usd", 0) for w in whales
                if getattr(w, "side", "") == "buy"
            )
            stack.whale_net_direction = (
                "BUY" if total > 0 and buys > total * 0.6
                else "SELL" if total > 0 and buys < total * 0.4
                else ""
            )
            stack.whale_total_usd = total

    # ── Smart money / conviction signals (Improvement 4B) ─────────
    if conviction_signals:
        for sig in conviction_signals:
            if getattr(sig, "whale_count", 0) > 0:
                stack.whale_count = sig.whale_count
                stack.whale_total_usd = getattr(sig, "total_whale_usd", 0.0)
                stack.whale_direction = getattr(sig, "direction", "")
                stack.whale_avg_entry = getattr(sig, "avg_whale_price", None)
                stack.whale_signal_strength = getattr(sig, "signal_strength", "")
                stack.whale_conviction_score = getattr(sig, "conviction_score", 0.0)
                break

    # ── Calendar events (Improvement 8) ───────────────────────────
    if calendar_events:
        stack.calendar_events = calendar_events

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

    if stack.manifold_probability is not None:
        traders_str = (
            f" ({stack.manifold_traders} traders)"
            if stack.manifold_traders
            else ""
        )
        consensus_lines.append(
            f"- Manifold community: {stack.manifold_probability:.1%}{traders_str}"
        )
    if stack.predictit_probability is not None:
        consensus_lines.append(
            f"- PredictIt: {stack.predictit_probability:.1%}"
        )
    if stack.sportsbook_consensus is not None:
        sharp_str = (
            f", Pinnacle sharp: {stack.sportsbook_sharp_price:.1%}"
            if stack.sportsbook_sharp_price is not None
            else ""
        )
        spread_str = (
            f", spread: {stack.sportsbook_spread_pp:.1f}pp"
            if stack.sportsbook_spread_pp is not None
            else ""
        )
        consensus_lines.append(
            f"- Sportsbook consensus: {stack.sportsbook_consensus:.1%}"
            f" ({stack.sportsbook_count} books{sharp_str}{spread_str})"
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
    if stack.sports_context:
        # Truncate to first 300 chars to keep prompt manageable
        behavioral_lines.append(
            f"- Sports context: {stack.sports_context[:300]}"
        )
    if stack.spotify_listeners_rank is not None or stack.spotify_daily_rank is not None:
        parts = [f"Spotify Charts: {stack.spotify_artist}"]
        if stack.spotify_listeners_rank is not None:
            parts.append(f"#{stack.spotify_listeners_rank} monthly listeners")
            if stack.spotify_monthly_listeners:
                parts.append(f"({stack.spotify_monthly_listeners})")
        if stack.spotify_daily_rank is not None:
            parts.append(f"#{stack.spotify_daily_rank} daily chart")
        behavioral_lines.append(f"- {' '.join(parts)}")

    if stack.kronos_upside_prob is not None:
        symbol_str = f" for {stack.kronos_symbol}" if stack.kronos_symbol else ""
        vol_str = (
            f", volatility amplification: {stack.kronos_volatility_prob:.0%}"
            if stack.kronos_volatility_prob is not None
            else ""
        )
        behavioral_lines.append(
            f"- Kronos foundation model{symbol_str}"
            f" (24h forecast, N=10 Monte Carlo paths):"
            f" upside probability {stack.kronos_upside_prob:.0%}{vol_str}"
        )

    if behavioral_lines:
        sections.append(
            "BEHAVIORAL SIGNALS:\n" + "\n".join(behavioral_lines)
        )

    # ── Order flow section (Improvement 2) ─────────────────────────
    flow_lines: list[str] = []
    if stack.order_flow_imbalance is not None:
        ofi = stack.order_flow_imbalance
        label = (
            "net BUY pressure" if ofi > 0.1
            else "net SELL pressure" if ofi < -0.1
            else "balanced flow"
        )
        flow_lines.append(f"- Order flow imbalance: {ofi:+.2f} ({label})")
    if stack.vwap_divergence_pct is not None:
        d = stack.vwap_divergence_pct
        flow_lines.append(
            f"- Price vs VWAP: {abs(d):.1f}% {'above' if d > 0 else 'below'}"
            " time-weighted avg"
        )
    if stack.smart_money_ratio is not None:
        smr = stack.smart_money_ratio
        label = (
            "smart money dominant (informed traders active)" if smr > 0.6
            else "retail dominant" if smr < 0.4
            else "mixed"
        )
        flow_lines.append(f"- Trade composition: {label} ({smr:.0%} smart money)")
    if stack.whale_net_direction and stack.whale_total_usd > 500:
        flow_lines.append(
            f"- Whale activity: ${stack.whale_total_usd:,.0f}"
            f" net {stack.whale_net_direction}"
        )
    if flow_lines:
        sections.append("ORDER FLOW SIGNALS:\n" + "\n".join(flow_lines))

    # ── Smart money section (Improvement 4B) ───────────────────────
    if stack.whale_count > 0 and stack.whale_direction:
        whale_lines: list[str] = []
        direction_text = {
            "BULLISH": "holding YES positions",
            "BEARISH": "holding NO positions",
            "MIXED": "split between YES and NO",
        }.get(stack.whale_direction, stack.whale_direction)
        whale_lines.append(
            f"- {stack.whale_count} tracked high-PnL traders {direction_text}"
        )
        if stack.whale_total_usd > 0:
            entry_str = (
                f" at avg entry {stack.whale_avg_entry:.2f}"
                if stack.whale_avg_entry is not None
                else ""
            )
            whale_lines.append(
                f"- Combined position: ${stack.whale_total_usd:,.0f}{entry_str}"
            )
        if stack.whale_signal_strength:
            whale_lines.append(
                f"- Conviction: {stack.whale_signal_strength}"
                f" (score: {stack.whale_conviction_score:.0f}/100)"
            )
        if stack.whale_best_performer_rate is not None:
            whale_lines.append(
                f"- Best performer win rate in this category:"
                f" {stack.whale_best_performer_rate:.0%}"
            )
        sections.append("SMART MONEY SIGNALS:\n" + "\n".join(whale_lines))

    # ── Upcoming events section (Improvement 8) ────────────────────
    if stack.calendar_events:
        event_lines: list[str] = []
        for evt in stack.calendar_events[:5]:
            name = getattr(evt, "name", str(evt))
            hours = getattr(evt, "hours_away", None)
            impact = getattr(evt, "impact", "")
            if hours is not None:
                event_lines.append(
                    f"- {name} in {hours:.0f}h"
                    + (f" [HIGH IMPACT]" if impact == "high" else "")
                )
        if event_lines:
            sections.append("UPCOMING EVENTS:\n" + "\n".join(event_lines))

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
    if stack.manifold_probability is not None:
        consensus_prices.append(stack.manifold_probability)
    if stack.predictit_probability is not None:
        consensus_prices.append(stack.predictit_probability)
    if stack.sportsbook_consensus is not None:
        consensus_prices.append(stack.sportsbook_consensus)

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
