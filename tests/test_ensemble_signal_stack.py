"""Tests for ensemble prompt rendering with signal stack."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.forecast.ensemble import _build_prompt
from src.forecast.feature_builder import MarketFeatures
from src.research.evidence_extractor import EvidencePackage
from src.research.signal_aggregator import SignalStack


# ── Helpers ──────────────────────────────────────────────────────────


def _make_features() -> MarketFeatures:
    return MarketFeatures(
        question="Will Bitcoin exceed $100k by Dec 2026?",
        market_type="CRYPTO",
        volume_usd=500000,
        liquidity_usd=100000,
        spread_pct=0.02,
        days_to_expiry=90,
        price_momentum=0.01,
        evidence_quality=0.7,
        num_sources=5,
        implied_probability=0.45,
        top_bullets=["BTC hitting new highs", "Institutional adoption increasing"],
    )


def _make_evidence() -> EvidencePackage:
    return EvidencePackage(
        market_id="test",
        question="Will Bitcoin exceed $100k?",
        summary="Bitcoin showing strong momentum.",
        contradictions=[],
    )


# ── V1 prompt with signal stack ──────────────────────────────────────


class TestV1WithSignalStack:
    def test_without_signal_stack_no_consensus_block(self) -> None:
        prompt = _build_prompt(_make_features(), _make_evidence())
        assert "CONSENSUS SIGNALS:" not in prompt
        assert "BEHAVIORAL SIGNALS:" not in prompt

    def test_with_empty_signal_stack_no_block(self) -> None:
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=SignalStack(),
        )
        assert "CONSENSUS SIGNALS:" not in prompt

    def test_with_consensus_signals(self) -> None:
        stack = SignalStack(
            kalshi_price=0.55,
            kalshi_spread_pp=2.0,
        )
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=stack,
        )
        assert "CONSENSUS SIGNALS:" in prompt
        assert "Kalshi mid-price: 55.0%" in prompt
        assert "spread: 2.0pp" in prompt

    def test_with_behavioral_signals(self) -> None:
        stack = SignalStack(
            wikipedia_spike_ratio=2.5,
            wikipedia_article="Bitcoin",
        )
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=stack,
        )
        assert "BEHAVIORAL SIGNALS:" in prompt
        assert "2.5x spike" in prompt
        assert "Bitcoin" in prompt

    def test_signal_block_before_evidence(self) -> None:
        stack = SignalStack(kalshi_price=0.55)
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=stack,
        )
        consensus_pos = prompt.index("CONSENSUS SIGNALS:")
        evidence_pos = prompt.index("Bitcoin showing strong momentum")
        assert consensus_pos < evidence_pos

    def test_v1_rules_mention_consensus(self) -> None:
        prompt = _build_prompt(_make_features(), _make_evidence())
        assert "CONSENSUS SIGNALS are present" in prompt


# ── V2 prompt with signal stack ──────────────────────────────────────


class TestV2WithSignalStack:
    def test_v2_without_signal_stack(self) -> None:
        prompt = _build_prompt(
            _make_features(), _make_evidence(), prompt_version="v2",
        )
        assert "CONSENSUS SIGNALS:" not in prompt
        assert "HISTORICAL BASE RATE:" in prompt

    def test_v2_with_consensus_and_behavioral(self) -> None:
        stack = SignalStack(
            metaculus_probability=0.60,
            metaculus_forecasters=150,
            reddit_sentiment=0.3,
            reddit_post_count=20,
        )
        prompt = _build_prompt(
            _make_features(), _make_evidence(),
            prompt_version="v2", signal_stack=stack,
        )
        assert "CONSENSUS SIGNALS:" in prompt
        assert "Metaculus community: 60.0%" in prompt
        assert "BEHAVIORAL SIGNALS:" in prompt
        assert "Reddit sentiment" in prompt

    def test_v2_rules_mention_consensus(self) -> None:
        prompt = _build_prompt(
            _make_features(), _make_evidence(), prompt_version="v2",
        )
        assert "CONSENSUS SIGNALS are present" in prompt


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompat:
    def test_none_signal_stack_no_error(self) -> None:
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=None,
        )
        assert "MARKET QUESTION:" in prompt

    def test_prompt_still_has_standard_fields(self) -> None:
        stack = SignalStack(kalshi_price=0.50)
        prompt = _build_prompt(
            _make_features(), _make_evidence(), signal_stack=stack,
        )
        assert "MARKET QUESTION:" in prompt
        assert "MARKET TYPE:" in prompt
        assert "EVIDENCE SUMMARY:" in prompt
        assert "MARKET FEATURES:" in prompt
        assert "TASK:" in prompt

    def test_base_rate_info_with_signal_stack(self) -> None:
        base_rate = MagicMock()
        base_rate.base_rate = 0.30
        base_rate.pattern_description = "test pattern"
        base_rate.source = "test source"
        base_rate.confidence = 0.80

        stack = SignalStack(kalshi_price=0.55)
        prompt = _build_prompt(
            _make_features(), _make_evidence(),
            base_rate_info=base_rate,
            prompt_version="v2",
            signal_stack=stack,
        )
        assert "HISTORICAL BASE RATE:" in prompt
        assert "CONSENSUS SIGNALS:" in prompt
        assert "Kalshi" in prompt
