"""Tests for evidence_extractor.py — structured evidence extraction.

Covers:
- EvidencePackage structure and serialization
- compute_independent_quality (5 scoring dimensions)
- _build_package from parsed LLM JSON
- parse_evidence_from_raw public helper
- EvidenceExtractor.extract() with mocked LLM
- Edge cases: empty sources, malformed JSON, source index bounds
"""

from __future__ import annotations

import asyncio
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.research.evidence_extractor import (
    Citation,
    Contradiction,
    EvidenceBullet,
    EvidencePackage,
    IndependentQualityScore,
    compute_independent_quality,
    parse_evidence_from_raw,
)
from src.research.source_fetcher import FetchedSource


# ── Helpers ──────────────────────────────────────────────────────────


def _make_source(
    url: str = "https://reuters.com/article/123",
    title: str = "Test Article",
    publisher: str = "Reuters",
    authority_score: float = 0.7,
    content: str = "Full content of the article with sufficient length " * 20,
    date: str = "",
) -> FetchedSource:
    return FetchedSource(
        url=url,
        title=title,
        publisher=publisher,
        snippet="A test snippet",
        authority_score=authority_score,
        extraction_method="web",
        content_length=len(content),
        content=content,
        date=date,
    )


def _make_raw_json(
    n_bullets: int = 3,
    n_contradictions: int = 0,
    quality_score: float = 0.7,
    numeric_bullets: int = 1,
) -> dict:
    bullets = []
    for i in range(n_bullets):
        bullet = {
            "text": f"Evidence bullet {i}",
            "source_index": 0,
            "relevance": 0.8,
            "is_numeric": i < numeric_bullets,
            "metric_name": f"metric_{i}" if i < numeric_bullets else "",
            "metric_value": f"{i * 10}" if i < numeric_bullets else "",
            "metric_unit": "percent" if i < numeric_bullets else "",
            "metric_date": "2026-01-15" if i < numeric_bullets else "",
            "confidence": 0.7,
        }
        bullets.append(bullet)

    contradictions = []
    for i in range(n_contradictions):
        contradictions.append({
            "claim_a": f"Claim A {i}",
            "source_a_index": 0,
            "claim_b": f"Claim B {i}",
            "source_b_index": min(1, 0),
            "description": f"Disagreement {i}",
        })

    return {
        "bullets": bullets,
        "contradictions": contradictions,
        "quality_score": quality_score,
        "summary": "Test summary of evidence.",
    }


# ── EvidencePackage ─────────────────────────────────────────────────


class TestEvidencePackage:
    def test_default_fields(self):
        pkg = EvidencePackage(market_id="mkt-1", question="Will X?")
        assert pkg.quality_score == 0.0
        assert pkg.bullets == []
        assert pkg.contradictions == []
        assert pkg.num_sources == 0

    def test_to_dict_structure(self):
        pkg = EvidencePackage(
            market_id="mkt-1",
            question="Will X?",
            market_type="MACRO",
            quality_score=0.75,
            num_sources=3,
            summary="Summary text",
        )
        d = pkg.to_dict()
        assert d["market_id"] == "mkt-1"
        assert d["quality_score"] == 0.75
        assert d["num_sources"] == 3
        assert "independent_quality" in d

    def test_to_dict_with_bullets(self):
        bullet = EvidenceBullet(
            text="CPI rose 3.2%",
            citation=Citation(url="https://bls.gov", publisher="BLS", date="2026-01-15"),
            relevance=0.9,
            is_numeric=True,
            metric_name="CPI YoY",
            metric_value="3.2",
            metric_unit="percent",
            metric_date="2026-01-15",
            confidence=0.85,
        )
        pkg = EvidencePackage(
            market_id="mkt-1", question="Q", bullets=[bullet],
        )
        d = pkg.to_dict()
        assert len(d["evidence"]) == 1
        assert d["evidence"][0]["is_numeric"] is True
        assert d["evidence"][0]["metric_name"] == "CPI YoY"


# ── compute_independent_quality ────────────────────────────────────


class TestComputeIndependentQuality:
    def test_empty_sources_returns_zero(self):
        result = compute_independent_quality([], [], [])
        assert result.overall == 0.0
        assert result.recency_score == 0.0

    def test_high_authority_sources(self):
        sources = [
            _make_source(authority_score=0.95),  # .gov-like
            _make_source(authority_score=0.7),
        ]
        result = compute_independent_quality(sources, [], [])
        assert result.authority_score >= 0.8  # .gov bonus applied

    def test_contradiction_penalty(self):
        sources = [_make_source(), _make_source()]
        bullets = []
        contradictions = [
            Contradiction(
                claim_a="Claim A",
                source_a=Citation(url="a.com", publisher="A", date=""),
                claim_b="Claim B",
                source_b=Citation(url="b.com", publisher="B", date=""),
            ),
            Contradiction(
                claim_a="Claim C",
                source_a=Citation(url="a.com", publisher="A", date=""),
                claim_b="Claim D",
                source_b=Citation(url="b.com", publisher="B", date=""),
            ),
        ]
        result = compute_independent_quality(sources, bullets, contradictions)
        # 2 contradictions → agreement = max(0.2, 1.0 - 0.30) = 0.7
        assert result.agreement_score == 0.7

    def test_no_contradictions_full_agreement(self):
        sources = [_make_source()]
        result = compute_independent_quality(sources, [], [])
        assert result.agreement_score == 1.0

    def test_numeric_density_bonus(self):
        sources = [_make_source()]
        numeric_bullets = [
            EvidenceBullet(
                text="GDP grew 2.5%",
                citation=Citation(url="", publisher="", date=""),
                is_numeric=True,
            )
            for _ in range(3)
        ]
        non_numeric = [
            EvidenceBullet(
                text="Statement",
                citation=Citation(url="", publisher="", date=""),
                is_numeric=False,
            )
        ]
        result = compute_independent_quality(sources, numeric_bullets + non_numeric, [])
        assert result.numeric_density_score > 0.5

    def test_content_depth_with_long_content(self):
        sources = [
            _make_source(content="x" * 1000),  # > 500 chars
            _make_source(content="short"),  # < 500 chars
        ]
        result = compute_independent_quality(sources, [], [])
        assert result.content_depth_score > 0

    def test_recent_sources_score_high(self):
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        sources = [_make_source(date=today)]
        result = compute_independent_quality(sources, [], [])
        assert result.recency_score == 1.0

    def test_stale_sources_score_low(self):
        old_date = "2020-01-01"
        sources = [_make_source(date=old_date)]
        result = compute_independent_quality(sources, [], [])
        assert result.recency_score == 0.2  # > 30 days old

    def test_overall_is_weighted_sum(self):
        sources = [_make_source(date="", authority_score=0.5)]
        result = compute_independent_quality(sources, [], [])
        expected = (
            result.recency_score * 0.20
            + result.authority_score * 0.30
            + result.agreement_score * 0.20
            + result.numeric_density_score * 0.15
            + result.content_depth_score * 0.15
        )
        assert abs(result.overall - round(expected, 3)) < 0.01

    def test_breakdown_contains_weights(self):
        sources = [_make_source()]
        result = compute_independent_quality(sources, [], [])
        assert result.breakdown["recency_weight"] == 0.20
        assert result.breakdown["authority_weight"] == 0.30


# ── parse_evidence_from_raw / _build_package ───────────────────────


class TestParseEvidenceFromRaw:
    def test_basic_package(self):
        sources = [_make_source()]
        raw = _make_raw_json(n_bullets=2, quality_score=0.6)
        pkg = parse_evidence_from_raw("mkt-1", "Will X?", sources, raw)
        assert pkg.market_id == "mkt-1"
        assert len(pkg.bullets) == 2
        assert pkg.llm_quality_score == 0.6
        assert pkg.summary == "Test summary of evidence."

    def test_quality_is_blended(self):
        sources = [_make_source()]
        raw = _make_raw_json(quality_score=0.8)
        pkg = parse_evidence_from_raw("mkt-1", "Q", sources, raw)
        # final = 0.8 * 0.4 + independent * 0.6
        assert pkg.quality_score > 0  # Non-zero
        assert pkg.quality_score != pkg.llm_quality_score  # Blended, not raw

    def test_source_index_bounds_checking(self):
        sources = [_make_source()]
        raw = {
            "bullets": [
                {"text": "Valid", "source_index": 0, "relevance": 0.9},
                {"text": "Out of bounds", "source_index": 99, "relevance": 0.5},
            ],
            "quality_score": 0.5,
            "summary": "",
        }
        pkg = parse_evidence_from_raw("mkt-1", "Q", sources, raw)
        assert len(pkg.bullets) == 2
        assert pkg.bullets[0].citation.url == "https://reuters.com/article/123"
        assert pkg.bullets[1].citation.url == ""  # Out of bounds → empty

    def test_contradictions_parsed(self):
        sources = [_make_source(), _make_source(url="https://bbc.com")]
        raw = {
            "bullets": [],
            "contradictions": [
                {
                    "claim_a": "Growth up",
                    "source_a_index": 0,
                    "claim_b": "Growth down",
                    "source_b_index": 1,
                    "description": "Conflicting GDP reports",
                },
            ],
            "quality_score": 0.4,
            "summary": "Mixed signals.",
        }
        pkg = parse_evidence_from_raw("mkt-1", "Q", sources, raw)
        assert len(pkg.contradictions) == 1
        assert pkg.contradictions[0].claim_a == "Growth up"
        assert pkg.contradictions[0].source_b.url == "https://bbc.com"

    def test_empty_bullets(self):
        sources = [_make_source()]
        raw = {"bullets": [], "quality_score": 0.2, "summary": "No evidence."}
        pkg = parse_evidence_from_raw("mkt-1", "Q", sources, raw)
        assert len(pkg.bullets) == 0
        assert pkg.num_sources == 1


# ── EvidenceExtractor.extract() ────────────────────────────────────


class TestEvidenceExtractor:
    @pytest.mark.asyncio
    async def test_empty_sources_returns_zero_quality(self):
        from src.research.evidence_extractor import EvidenceExtractor

        config = MagicMock()
        config.llm_model = "gpt-4o-mini"
        config.evidence_model = "gpt-4o-mini"
        config.llm_max_tokens = 2048

        extractor = EvidenceExtractor(config)
        pkg = await extractor.extract("mkt-1", "Will X?", [], "MACRO")
        assert pkg.quality_score == 0.0
        assert "No sources" in pkg.summary

    @pytest.mark.asyncio
    async def test_llm_extraction_success(self):
        from src.research.evidence_extractor import EvidenceExtractor

        config = MagicMock()
        config.llm_model = "gpt-4o-mini"
        config.evidence_model = "gpt-4o-mini"
        config.llm_max_tokens = 2048

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"bullets": [{"text": "CPI rose 3.2%", "source_index": 0, '
                    '"relevance": 0.9, "is_numeric": true, "metric_name": "CPI", '
                    '"metric_value": "3.2", "metric_unit": "percent", '
                    '"metric_date": "2026-01-15", "confidence": 0.85}], '
                    '"contradictions": [], "quality_score": 0.8, "summary": "Strong evidence"}'
                )
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        extractor = EvidenceExtractor(config)

        with patch.object(
            extractor._llm.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            with patch("src.research.evidence_extractor.cost_tracker"):
                sources = [_make_source()]
                pkg = await extractor.extract("mkt-1", "Will CPI exceed 3%?", sources, "MACRO")

        assert len(pkg.bullets) == 1
        assert pkg.bullets[0].is_numeric is True
        assert pkg.llm_quality_score == 0.8
        assert pkg.quality_score > 0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self):
        from src.research.evidence_extractor import EvidenceExtractor

        config = MagicMock()
        config.llm_model = "gpt-4o-mini"
        config.evidence_model = "gpt-4o-mini"
        config.llm_max_tokens = 2048

        extractor = EvidenceExtractor(config)

        with patch.object(
            extractor._llm.chat.completions,
            "create",
            new=AsyncMock(side_effect=RuntimeError("API timeout")),
        ):
            sources = [_make_source()]
            pkg = await extractor.extract("mkt-1", "Q", sources, "MACRO")

        assert pkg.quality_score == 0.0
        assert "LLM extraction failed" in pkg.summary

    @pytest.mark.asyncio
    async def test_markdown_fences_stripped(self):
        from src.research.evidence_extractor import EvidenceExtractor

        config = MagicMock()
        config.llm_model = "gpt-4o-mini"
        config.evidence_model = "gpt-4o-mini"
        config.llm_max_tokens = 2048

        # LLM returns JSON wrapped in markdown fences
        json_content = (
            '```json\n{"bullets": [], "contradictions": [], '
            '"quality_score": 0.5, "summary": "Fenced"}\n```'
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json_content))]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        extractor = EvidenceExtractor(config)

        with patch.object(
            extractor._llm.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            with patch("src.research.evidence_extractor.cost_tracker"):
                sources = [_make_source()]
                pkg = await extractor.extract("mkt-1", "Q", sources, "MACRO")

        assert pkg.summary == "Fenced"
        assert pkg.llm_quality_score == 0.5
