"""Tests for sports signal integration in SignalStack."""

from __future__ import annotations

import pytest

from src.research.signal_aggregator import (
    SignalStack,
    build_signal_stack,
    compute_signal_confluence,
    render_signal_stack,
)
from src.research.source_fetcher import FetchedSource


# ── Helpers ────────────────────────────────────────────────────────

def _make_source(raw: dict | None = None, content: str = "") -> FetchedSource:
    return FetchedSource(
        title="test",
        url="https://test.com",
        snippet="test",
        publisher="test",
        date="",
        content=content,
        authority_score=1.0,
        query_intent="primary",
        extraction_method="api",
        content_length=len(content),
        raw=raw or {},
    )


# ═══════════════════════════════════════════════════════════════════
#  build_signal_stack — sportsbook consensus
# ═══════════════════════════════════════════════════════════════════


class TestBuildSignalStackSportsbooks:
    def test_sportsbook_consensus_populated(self):
        src = _make_source(raw={
            "consensus_signal": {
                "platform": "sportsbooks",
                "price": 0.62,
                "spread_pp": 3.2,
                "books": 4,
                "sharp_book": 0.618,
            },
        })
        stack = build_signal_stack([src], poly_price=0.55)
        assert stack.sportsbook_consensus == 0.62
        assert stack.sportsbook_spread_pp == 3.2
        assert stack.sportsbook_count == 4
        assert stack.sportsbook_sharp_price == 0.618

    def test_sportsbook_added_to_consensus_prices(self):
        """Sportsbook consensus should affect divergence calculation."""
        src = _make_source(raw={
            "consensus_signal": {
                "platform": "sportsbooks",
                "price": 0.80,
                "books": 3,
            },
        })
        stack = build_signal_stack([src], poly_price=0.50)
        # 30pp divergence → should reduce Kelly multiplier significantly
        assert stack.consensus_divergence == pytest.approx(0.30, abs=0.01)
        assert stack.recommended_kelly_multiplier < 0.50

    def test_no_sportsbook_leaves_defaults(self):
        stack = build_signal_stack([], poly_price=0.50)
        assert stack.sportsbook_consensus is None
        assert stack.sportsbook_count == 0
        assert stack.sportsbook_sharp_price is None

    def test_sportsbook_without_sharp_book(self):
        src = _make_source(raw={
            "consensus_signal": {
                "platform": "sportsbooks",
                "price": 0.55,
                "spread_pp": 2.0,
                "books": 2,
                "sharp_book": None,
            },
        })
        stack = build_signal_stack([src], poly_price=0.55)
        assert stack.sportsbook_consensus == 0.55
        assert stack.sportsbook_sharp_price is None


# ═══════════════════════════════════════════════════════════════════
#  build_signal_stack — sports context
# ═══════════════════════════════════════════════════════════════════


class TestBuildSignalStackSportsContext:
    def test_sports_context_populated(self):
        src = _make_source(
            raw={
                "behavioral_signal": {
                    "source": "sports_stats",
                    "signal_type": "sports_context",
                    "value": 0.67,
                    "home_form": "WWDLW",
                    "away_form": "LDWWL",
                },
            },
            content="Home form WWDLW, Away form LDWWL, H2H: 3-1-1",
        )
        stack = build_signal_stack([src], poly_price=0.50)
        assert "WWDLW" in stack.sports_context

    def test_no_sports_context(self):
        stack = build_signal_stack([], poly_price=0.50)
        assert stack.sports_context == ""


# ═══════════════════════════════════════════════════════════════════
#  render_signal_stack — sportsbook lines
# ═══════════════════════════════════════════════════════════════════


class TestRenderSportsbook:
    def test_sportsbook_in_consensus_section(self):
        stack = SignalStack(
            sportsbook_consensus=0.62,
            sportsbook_spread_pp=3.2,
            sportsbook_count=4,
            sportsbook_sharp_price=0.618,
        )
        rendered = render_signal_stack(stack)
        assert "CONSENSUS SIGNALS:" in rendered
        assert "Sportsbook consensus: 62.0%" in rendered
        assert "4 books" in rendered
        assert "Pinnacle sharp: 61.8%" in rendered
        assert "spread: 3.2pp" in rendered

    def test_sportsbook_without_sharp(self):
        stack = SignalStack(
            sportsbook_consensus=0.55,
            sportsbook_count=2,
        )
        rendered = render_signal_stack(stack)
        assert "Sportsbook consensus: 55.0%" in rendered
        assert "Pinnacle" not in rendered

    def test_sports_context_in_behavioral(self):
        stack = SignalStack(
            sports_context="Home form WWDLW, Away form LDWWL",
        )
        rendered = render_signal_stack(stack)
        assert "BEHAVIORAL SIGNALS:" in rendered
        assert "Sports context:" in rendered
        assert "WWDLW" in rendered

    def test_no_sports_signals_empty(self):
        stack = SignalStack()
        rendered = render_signal_stack(stack)
        assert rendered == ""


# ═══════════════════════════════════════════════════════════════════
#  compute_signal_confluence — sportsbook included
# ═══════════════════════════════════════════════════════════════════


class TestConfluenceSportsbook:
    def test_sportsbook_agrees(self):
        stack = SignalStack(sportsbook_consensus=0.52)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 1.0  # 2pp divergence → agreement

    def test_sportsbook_mild_divergence(self):
        stack = SignalStack(sportsbook_consensus=0.58)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 0.75  # 8pp divergence

    def test_sportsbook_moderate_divergence(self):
        stack = SignalStack(sportsbook_consensus=0.63)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 0.50  # 13pp divergence

    def test_sportsbook_severe_divergence(self):
        stack = SignalStack(sportsbook_consensus=0.75)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 0.25  # 25pp divergence

    def test_no_sportsbook_returns_one(self):
        stack = SignalStack()
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 1.0
