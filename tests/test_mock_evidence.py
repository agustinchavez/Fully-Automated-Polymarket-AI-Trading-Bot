"""Tests for mock evidence builder (Phase 1)."""

from __future__ import annotations

import pytest

from src.backtest.mock_evidence import build_mock_evidence
from src.backtest.models import HistoricalMarketRecord


def _make_hist(**kwargs) -> HistoricalMarketRecord:
    defaults = dict(
        condition_id="0x1",
        question="Will BTC hit 100k?",
        description="Bitcoin price prediction market for 2024.",
        category="Crypto",
        market_type="MACRO",
        resolution="YES",
    )
    defaults.update(kwargs)
    return HistoricalMarketRecord(**defaults)


class TestBuildMockEvidence:

    def test_basic_evidence(self) -> None:
        hist = _make_hist()
        ev = build_mock_evidence(hist)
        assert ev.market_id == "0x1"
        assert ev.question == "Will BTC hit 100k?"
        assert ev.quality_score == 0.5
        assert ev.num_sources == 1
        assert len(ev.bullets) >= 1

    def test_uses_description_as_summary(self) -> None:
        hist = _make_hist(description="Detailed market description here.")
        ev = build_mock_evidence(hist)
        assert "Detailed market description" in ev.summary

    def test_empty_description_uses_question(self) -> None:
        hist = _make_hist(description="")
        ev = build_mock_evidence(hist)
        assert ev.summary == hist.question
        assert len(ev.bullets) == 1
        assert ev.bullets[0].text == hist.question

    def test_custom_quality_score(self) -> None:
        hist = _make_hist()
        ev = build_mock_evidence(hist, quality_score=0.8)
        assert ev.quality_score == 0.8
        assert ev.llm_quality_score == 0.8

    def test_market_type_passed_through(self) -> None:
        hist = _make_hist(market_type="ELECTION")
        ev = build_mock_evidence(hist)
        assert ev.market_type == "ELECTION"

    def test_citation_publisher(self) -> None:
        hist = _make_hist()
        ev = build_mock_evidence(hist)
        assert ev.bullets[0].citation.publisher == "Polymarket"

    def test_long_description_splits_into_bullets(self) -> None:
        desc = (
            "First sentence about the market. "
            "Second sentence with more details. "
            "Third sentence with even more context. "
            "Fourth sentence providing background. "
            "Fifth sentence wrapping up."
        )
        hist = _make_hist(description=desc)
        ev = build_mock_evidence(hist)
        assert len(ev.bullets) >= 3

    def test_max_five_bullets(self) -> None:
        desc = ". ".join([f"Sentence number {i} with content" for i in range(20)])
        hist = _make_hist(description=desc)
        ev = build_mock_evidence(hist)
        assert len(ev.bullets) <= 5
