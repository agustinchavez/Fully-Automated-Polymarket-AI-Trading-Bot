"""Tests for pipeline signal stack wiring (Phase 2 integration)."""

from __future__ import annotations

import pytest

from src.engine.loop import PipelineContext
from src.research.signal_aggregator import SignalStack, build_signal_stack
from src.research.source_fetcher import FetchedSource


# ── Helpers ──────────────────────────────────────────────────────────


def _make_ctx(**kwargs) -> PipelineContext:
    """Create a minimal PipelineContext with a mock market."""
    from unittest.mock import MagicMock

    market = MagicMock()
    market.id = "test"
    market.question = "Test?"
    market.category = "MACRO"
    market.market_type = "MACRO"
    defaults = dict(market=market, cycle_id=1, market_id="test", question="Test?")
    defaults.update(kwargs)
    return PipelineContext(**defaults)


def _make_source_with_signal(signal_type: str, data: dict) -> FetchedSource:
    return FetchedSource(
        title="Test",
        url="https://example.com",
        snippet="test",
        publisher="Test",
        raw={signal_type: data},
    )


# ── PipelineContext._signal_stack field ──────────────────────────────


class TestPipelineContextField:
    def test_signal_stack_defaults_to_none(self) -> None:
        ctx = _make_ctx()
        assert ctx._signal_stack is None

    def test_signal_stack_can_be_set(self) -> None:
        ctx = _make_ctx()
        stack = SignalStack(kalshi_price=0.55)
        ctx._signal_stack = stack
        assert ctx._signal_stack is not None
        assert ctx._signal_stack.kalshi_price == 0.55


# ── Signal stack build from sources ──────────────────────────────────


class TestSignalStackFromSources:
    def test_build_from_consensus_source(self) -> None:
        sources = [
            _make_source_with_signal("consensus_signal", {
                "platform": "kalshi",
                "price": 0.60,
                "spread_pp": 2.0,
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.kalshi_price == 0.60
        assert stack.consensus_divergence == 0.10

    def test_build_from_behavioral_source(self) -> None:
        sources = [
            _make_source_with_signal("behavioral_signal", {
                "source": "wikipedia",
                "signal_type": "attention_spike",
                "value": 2.0,
                "article": "Test_Article",
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.wikipedia_spike_ratio == 2.0

    def test_build_from_mixed_sources(self) -> None:
        sources = [
            _make_source_with_signal("consensus_signal", {
                "platform": "metaculus",
                "price": 0.55,
                "forecasters": 100,
            }),
            _make_source_with_signal("behavioral_signal", {
                "source": "reddit",
                "signal_type": "sentiment",
                "value": 0.3,
                "post_count": 10,
            }),
            FetchedSource(
                title="Plain",
                url="https://example.com",
                snippet="no signal",
                publisher="Regular",
            ),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.metaculus_probability == 0.55
        assert stack.reddit_sentiment == 0.3
        assert stack.consensus_divergence == 0.05


# ── Confluence multiplier in pipeline context ────────────────────────


class TestConfluenceInPipeline:
    def test_no_signal_stack_confluence_is_1(self) -> None:
        ctx = _make_ctx()
        conf = 1.0
        signal_stack = getattr(ctx, "_signal_stack", None)
        if signal_stack is not None:
            conf = getattr(signal_stack, "recommended_kelly_multiplier", 1.0)
        assert conf == 1.0

    def test_signal_stack_with_divergence_reduces_confluence(self) -> None:
        sources = [
            _make_source_with_signal("consensus_signal", {
                "platform": "kalshi",
                "price": 0.71,  # 21pp divergence → 0.25
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)

        ctx = _make_ctx()
        ctx._signal_stack = stack
        conf = getattr(ctx._signal_stack, "recommended_kelly_multiplier", 1.0)
        assert conf == 0.25

    def test_signal_stack_agreement_keeps_confluence_1(self) -> None:
        sources = [
            _make_source_with_signal("consensus_signal", {
                "platform": "kalshi",
                "price": 0.52,  # 2pp → 1.0
            }),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.recommended_kelly_multiplier == 1.0


# ── No connectors enabled → pipeline unchanged ──────────────────────


class TestNoConnectorsEnabled:
    def test_empty_sources_produce_empty_stack(self) -> None:
        stack = build_signal_stack([], poly_price=0.50)
        assert stack.kalshi_price is None
        assert stack.metaculus_probability is None
        assert stack.wikipedia_spike_ratio is None
        assert stack.google_trends_spike_ratio is None
        assert stack.reddit_sentiment is None
        assert stack.recommended_kelly_multiplier == 1.0

    def test_regular_sources_produce_empty_stack(self) -> None:
        sources = [
            FetchedSource(
                title="News article",
                url="https://reuters.com/article",
                snippet="some news",
                publisher="Reuters",
                extraction_method="search",
            ),
        ]
        stack = build_signal_stack(sources, poly_price=0.50)
        assert stack.recommended_kelly_multiplier == 1.0
        assert stack.consensus_divergence == 0.0
