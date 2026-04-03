"""Tests for SignalAggregator — build, render, and confluence scoring."""

from __future__ import annotations

import pytest

from src.research.signal_aggregator import (
    SignalStack,
    build_signal_stack,
    compute_signal_confluence,
    render_signal_stack,
)
from src.research.source_fetcher import FetchedSource


# ── Helpers ──────────────────────────────────────────────────────────


def _make_source(raw: dict) -> FetchedSource:
    """Create a minimal FetchedSource with given raw dict."""
    return FetchedSource(
        title="Test",
        url="https://example.com",
        snippet="test",
        publisher="Test",
        raw=raw,
    )


# ── build_signal_stack ───────────────────────────────────────────────


class TestBuildSignalStack:
    def test_empty_sources(self) -> None:
        stack = build_signal_stack([], poly_price=0.5)
        assert stack.kalshi_price is None
        assert stack.metaculus_probability is None
        assert stack.wikipedia_spike_ratio is None
        assert stack.consensus_divergence == 0.0
        assert stack.recommended_kelly_multiplier == 1.0

    def test_consensus_only_kalshi(self) -> None:
        sources = [
            _make_source({
                "consensus_signal": {
                    "platform": "kalshi",
                    "price": 0.55,
                    "spread_pp": 2.0,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.kalshi_price == 0.55
        assert stack.kalshi_spread_pp == 2.0
        assert stack.metaculus_probability is None
        assert stack.consensus_divergence == 0.05

    def test_consensus_only_metaculus(self) -> None:
        sources = [
            _make_source({
                "consensus_signal": {
                    "platform": "metaculus",
                    "price": 0.60,
                    "forecasters": 100,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.metaculus_probability == 0.60
        assert stack.metaculus_forecasters == 100
        assert stack.kalshi_price is None
        assert stack.consensus_divergence == 0.10

    def test_behavioral_only_wikipedia(self) -> None:
        sources = [
            _make_source({
                "behavioral_signal": {
                    "source": "wikipedia",
                    "signal_type": "attention_spike",
                    "value": 2.5,
                    "article": "Donald_Trump",
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.wikipedia_spike_ratio == 2.5
        assert stack.wikipedia_article == "Donald_Trump"
        assert stack.consensus_divergence == 0.0

    def test_behavioral_only_google_trends(self) -> None:
        sources = [
            _make_source({
                "behavioral_signal": {
                    "source": "google_trends",
                    "signal_type": "search_trend",
                    "value": 1.8,
                    "current_index": 75,
                    "narrative": "Rising interest in topic",
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.google_trends_spike_ratio == 1.8
        assert stack.google_trends_index == 75
        assert stack.google_trends_narrative == "Rising interest in topic"

    def test_behavioral_only_reddit(self) -> None:
        sources = [
            _make_source({
                "behavioral_signal": {
                    "source": "reddit",
                    "signal_type": "sentiment",
                    "value": 0.35,
                    "post_count": 12,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.reddit_sentiment == 0.35
        assert stack.reddit_post_count == 12

    def test_mixed_consensus_and_behavioral(self) -> None:
        sources = [
            _make_source({
                "consensus_signal": {
                    "platform": "kalshi",
                    "price": 0.48,
                    "spread_pp": 3.0,
                },
            }),
            _make_source({
                "consensus_signal": {
                    "platform": "metaculus",
                    "price": 0.52,
                    "forecasters": 50,
                },
            }),
            _make_source({
                "behavioral_signal": {
                    "source": "wikipedia",
                    "signal_type": "attention_spike",
                    "value": 1.2,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.kalshi_price == 0.48
        assert stack.metaculus_probability == 0.52
        assert stack.wikipedia_spike_ratio == 1.2
        # Divergence is max(|0.48-0.50|, |0.52-0.50|) = 0.02
        assert stack.consensus_divergence == 0.02

    def test_divergence_uses_max(self) -> None:
        sources = [
            _make_source({
                "consensus_signal": {
                    "platform": "kalshi",
                    "price": 0.70,
                },
            }),
            _make_source({
                "consensus_signal": {
                    "platform": "metaculus",
                    "price": 0.55,
                    "forecasters": 30,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        # max(|0.70-0.50|, |0.55-0.50|) = 0.20
        assert stack.consensus_divergence == 0.20

    def test_ignores_unknown_signal_types(self) -> None:
        sources = [
            _make_source({
                "behavioral_signal": {
                    "source": "unknown_platform",
                    "signal_type": "mystery",
                    "value": 999,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.wikipedia_spike_ratio is None
        assert stack.google_trends_spike_ratio is None
        assert stack.reddit_sentiment is None

    def test_sources_without_raw_dict(self) -> None:
        sources = [
            FetchedSource(
                title="Plain",
                url="https://example.com",
                snippet="no raw",
                publisher="Test",
            ),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.kalshi_price is None


# ── render_signal_stack ──────────────────────────────────────────────


class TestRenderSignalStack:
    def test_empty_stack_returns_empty_string(self) -> None:
        stack = SignalStack()
        assert render_signal_stack(stack) == ""

    def test_consensus_block(self) -> None:
        stack = SignalStack(
            kalshi_price=0.55,
            kalshi_spread_pp=2.0,
        )
        rendered = render_signal_stack(stack)
        assert "CONSENSUS SIGNALS:" in rendered
        assert "Kalshi mid-price: 55.0%" in rendered
        assert "spread: 2.0pp" in rendered
        assert "BEHAVIORAL SIGNALS:" not in rendered

    def test_metaculus_block(self) -> None:
        stack = SignalStack(
            metaculus_probability=0.60,
            metaculus_forecasters=150,
        )
        rendered = render_signal_stack(stack)
        assert "CONSENSUS SIGNALS:" in rendered
        assert "Metaculus community: 60.0%" in rendered
        assert "150 forecasters" in rendered

    def test_behavioral_block(self) -> None:
        stack = SignalStack(
            wikipedia_spike_ratio=2.5,
            wikipedia_article="Bitcoin",
        )
        rendered = render_signal_stack(stack)
        assert "BEHAVIORAL SIGNALS:" in rendered
        assert "2.5x spike" in rendered
        assert "Bitcoin" in rendered
        assert "CONSENSUS SIGNALS:" not in rendered

    def test_google_trends_with_narrative(self) -> None:
        stack = SignalStack(
            google_trends_spike_ratio=1.8,
            google_trends_index=75,
            google_trends_narrative="Interest surging due to news",
        )
        rendered = render_signal_stack(stack)
        assert "Google Trends: 1.8x spike" in rendered
        assert "index=75" in rendered
        assert "Interest surging" in rendered

    def test_reddit_bullish(self) -> None:
        stack = SignalStack(
            reddit_sentiment=0.45,
            reddit_post_count=20,
        )
        rendered = render_signal_stack(stack)
        assert "Reddit sentiment: +0.45 (bullish, 20 posts)" in rendered

    def test_reddit_bearish(self) -> None:
        stack = SignalStack(
            reddit_sentiment=-0.30,
            reddit_post_count=8,
        )
        rendered = render_signal_stack(stack)
        assert "bearish" in rendered

    def test_reddit_neutral(self) -> None:
        stack = SignalStack(
            reddit_sentiment=0.05,
            reddit_post_count=3,
        )
        rendered = render_signal_stack(stack)
        assert "neutral" in rendered

    def test_full_rendering_has_both_sections(self) -> None:
        stack = SignalStack(
            kalshi_price=0.55,
            metaculus_probability=0.60,
            metaculus_forecasters=100,
            wikipedia_spike_ratio=2.0,
            reddit_sentiment=0.20,
            reddit_post_count=15,
        )
        rendered = render_signal_stack(stack)
        assert "CONSENSUS SIGNALS:" in rendered
        assert "BEHAVIORAL SIGNALS:" in rendered
        assert "Kalshi" in rendered
        assert "Metaculus" in rendered
        assert "Wikipedia" in rendered
        assert "Reddit" in rendered

    def test_narrative_truncated_to_200_chars(self) -> None:
        stack = SignalStack(
            google_trends_spike_ratio=1.5,
            google_trends_narrative="x" * 300,
        )
        rendered = render_signal_stack(stack)
        # Narrative should be truncated in the Context line
        context_line = [
            l for l in rendered.split("\n") if "Context:" in l
        ]
        assert len(context_line) == 1
        # The narrative in Context line is truncated to 200 chars
        assert len(context_line[0]) < 220  # "  Context: " + 200


# ── compute_signal_confluence ────────────────────────────────────────


class TestComputeSignalConfluence:
    def test_no_signals_returns_1(self) -> None:
        stack = SignalStack()
        assert compute_signal_confluence(stack, 0.50) == 1.0

    def test_agreement_within_5pp(self) -> None:
        stack = SignalStack(kalshi_price=0.52)
        assert compute_signal_confluence(stack, 0.50) == 1.0

    def test_mild_disagreement_5_to_10pp(self) -> None:
        stack = SignalStack(kalshi_price=0.57)
        assert compute_signal_confluence(stack, 0.50) == 0.75

    def test_moderate_disagreement_10_to_15pp(self) -> None:
        stack = SignalStack(kalshi_price=0.62)
        assert compute_signal_confluence(stack, 0.50) == 0.50

    def test_strong_disagreement_15_to_20pp(self) -> None:
        stack = SignalStack(kalshi_price=0.67)
        assert compute_signal_confluence(stack, 0.50) == 0.35

    def test_severe_disagreement_over_20pp(self) -> None:
        stack = SignalStack(kalshi_price=0.75)
        assert compute_signal_confluence(stack, 0.50) == 0.25

    def test_uses_average_of_consensus(self) -> None:
        # Kalshi=0.55, Metaculus=0.65, avg=0.60 → 10pp divergence
        stack = SignalStack(
            kalshi_price=0.55,
            metaculus_probability=0.65,
            metaculus_forecasters=50,
        )
        assert compute_signal_confluence(stack, 0.50) == 0.50

    def test_behavioral_only_returns_1(self) -> None:
        stack = SignalStack(
            wikipedia_spike_ratio=3.0,
            reddit_sentiment=-0.5,
        )
        assert compute_signal_confluence(stack, 0.50) == 1.0

    def test_boundary_exactly_5pp(self) -> None:
        stack = SignalStack(kalshi_price=0.55)
        # 5pp exactly → falls into 5-10pp bucket
        assert compute_signal_confluence(stack, 0.50) == 0.75

    def test_divergence_11pp(self) -> None:
        stack = SignalStack(kalshi_price=0.61)
        assert compute_signal_confluence(stack, 0.50) == 0.50

    def test_divergence_21pp(self) -> None:
        stack = SignalStack(kalshi_price=0.71)
        assert compute_signal_confluence(stack, 0.50) == 0.25

    def test_recommended_multiplier_set_by_build(self) -> None:
        sources = [
            _make_source({
                "consensus_signal": {
                    "platform": "kalshi",
                    "price": 0.71,
                },
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        # 21pp divergence → 0.25
        assert stack.recommended_kelly_multiplier == 0.25
