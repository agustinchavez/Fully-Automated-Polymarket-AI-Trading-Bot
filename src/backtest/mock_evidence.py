"""Mock evidence builder for backtesting.

Creates minimal EvidencePackage from historical market metadata,
without web search. The forecaster relies on its parametric knowledge
plus the market description.
"""

from __future__ import annotations

from src.backtest.models import HistoricalMarketRecord
from src.research.evidence_extractor import (
    Citation,
    EvidenceBullet,
    EvidencePackage,
    IndependentQualityScore,
)


def build_mock_evidence(
    hist: HistoricalMarketRecord,
    quality_score: float = 0.5,
) -> EvidencePackage:
    """Build a minimal EvidencePackage from historical market metadata.

    Uses the market's description and question text as evidence.
    No web search is performed. Quality is set to the configured
    mock level so the forecaster applies appropriate uncertainty.
    """
    # Build summary from description or question
    summary = hist.description[:500] if hist.description else hist.question

    # Create bullets from description sentences
    bullets: list[EvidenceBullet] = []
    if hist.description:
        sentences = [
            s.strip() for s in hist.description.replace("\n", ". ").split(".")
            if s.strip() and len(s.strip()) > 10
        ]
        citation = Citation(
            url="",
            publisher="Polymarket",
            date=hist.created_at or "",
            title=hist.question[:100],
        )
        for sent in sentences[:5]:
            bullets.append(
                EvidenceBullet(
                    text=sent,
                    citation=citation,
                    relevance=0.5,
                )
            )

    # If no description, use the question itself as a bullet
    if not bullets:
        bullets.append(
            EvidenceBullet(
                text=hist.question,
                citation=Citation(
                    url="", publisher="Polymarket",
                    date=hist.created_at or "",
                    title=hist.question[:100],
                ),
                relevance=0.5,
            )
        )

    return EvidencePackage(
        market_id=hist.condition_id,
        question=hist.question,
        market_type=hist.market_type,
        bullets=bullets,
        quality_score=quality_score,
        llm_quality_score=quality_score,
        independent_quality=IndependentQualityScore(overall=quality_score),
        num_sources=1,
        summary=summary,
    )
