"""Tests for event correlation, VaR upgrade, and capital efficiency (Phase 3 — Batch B)."""

from __future__ import annotations

import pytest

from src.policy.correlation import EventCorrelationScorer, CorrelationPair
from src.policy.capital_efficiency import (
    compute_annualized_edge,
    rank_markets_by_efficiency,
    RankedMarket,
)
from src.policy.portfolio_risk import (
    PositionSnapshot,
    calculate_portfolio_var,
    check_var_gate,
)
from src.config import PortfolioConfig


# ── helpers ─────────────────────────────────────────────────────────


def _pos(
    market_id: str = "m1",
    event_slug: str = "evt-1",
    side: str = "YES",
    category: str = "MACRO",
    size_usd: float = 100.0,
    entry_price: float = 0.50,
    current_price: float = 0.50,
    question: str = "Test?",
) -> PositionSnapshot:
    return PositionSnapshot(
        market_id=market_id,
        question=question,
        category=category,
        event_slug=event_slug,
        side=side,
        size_usd=size_usd,
        entry_price=entry_price,
        current_price=current_price,
    )


# ── EventCorrelationScorer ──────────────────────────────────────────


class TestEventCorrelationScorer:

    def test_same_event_same_side_high_correlation(self) -> None:
        """Two YES positions in same event_slug → 0.8."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="election-2028", side="YES"),
            _pos(market_id="m2", event_slug="election-2028", side="YES"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert len(pairs) == 1
        assert pairs[0].correlation == 0.8
        assert pairs[0].source == "same_event_same_side"

    def test_same_event_diff_side_moderate_correlation(self) -> None:
        """YES and NO in same event → 0.3."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="election-2028", side="YES"),
            _pos(market_id="m2", event_slug="election-2028", side="NO"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert len(pairs) == 1
        assert pairs[0].correlation == 0.3
        assert pairs[0].source == "same_event_diff_side"

    def test_same_category_low_correlation(self) -> None:
        """Same category, different event → 0.15."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-2", category="MACRO"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert len(pairs) == 1
        assert pairs[0].correlation == 0.15
        assert pairs[0].source == "same_category"

    def test_unrelated_zero_correlation(self) -> None:
        """Different category, different event → 0.0."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-2", category="SPORTS"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert len(pairs) == 1
        assert pairs[0].correlation == 0.0
        assert pairs[0].source == "unrelated"

    def test_empty_positions(self) -> None:
        """Empty list → no pairs."""
        scorer = EventCorrelationScorer()
        assert scorer.compute_pairwise([]) == []

    def test_single_position(self) -> None:
        """Single position → no pairs."""
        scorer = EventCorrelationScorer()
        assert scorer.compute_pairwise([_pos()]) == []

    def test_three_positions_mixed(self) -> None:
        """Three positions → 3 pairs with different correlations."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", side="YES", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-1", side="NO", category="MACRO"),
            _pos(market_id="m3", event_slug="evt-2", side="YES", category="SPORTS"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert len(pairs) == 3
        # m1-m2: same event, diff side → 0.3
        p12 = [p for p in pairs if "m1" in (p.market_id_a, p.market_id_b) and "m2" in (p.market_id_a, p.market_id_b)][0]
        assert p12.correlation == 0.3
        # m1-m3: different event, different category → 0.0
        p13 = [p for p in pairs if "m1" in (p.market_id_a, p.market_id_b) and "m3" in (p.market_id_a, p.market_id_b)][0]
        assert p13.correlation == 0.0

    def test_custom_config_values(self) -> None:
        """Scorer uses config values for correlation coefficients."""
        cfg = PortfolioConfig(
            same_event_same_outcome_corr=0.9,
            same_event_diff_outcome_corr=0.4,
            same_category_corr=0.2,
        )
        scorer = EventCorrelationScorer(cfg)
        positions = [
            _pos(market_id="m1", event_slug="evt-1", side="YES"),
            _pos(market_id="m2", event_slug="evt-1", side="YES"),
        ]
        pairs = scorer.compute_pairwise(positions)
        assert pairs[0].correlation == 0.9

    def test_score_new_position_max(self) -> None:
        """score_new_position returns max correlation with existing."""
        scorer = EventCorrelationScorer()
        existing = [
            _pos(market_id="m1", event_slug="evt-1", side="YES", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-2", side="NO", category="SPORTS"),
        ]
        # Same event as m1
        score = scorer.score_new_position(existing, "evt-1", "YES", "MACRO")
        assert score == 0.8

    def test_score_new_position_empty(self) -> None:
        """score_new_position with empty existing → 0.0."""
        scorer = EventCorrelationScorer()
        assert scorer.score_new_position([], "evt-1", "YES", "MACRO") == 0.0


# ── Correlation Matrix ──────────────────────────────────────────────


class TestCorrelationMatrix:

    def test_diagonal_is_one(self) -> None:
        """Diagonal of correlation matrix is 1.0."""
        scorer = EventCorrelationScorer()
        positions = [_pos(market_id=f"m{i}") for i in range(3)]
        ids, matrix = scorer.build_correlation_matrix(positions)
        for i in range(3):
            assert matrix[i][i] == 1.0

    def test_symmetric(self) -> None:
        """Correlation matrix is symmetric."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-2", category="MACRO"),
            _pos(market_id="m3", event_slug="evt-1", side="NO", category="MACRO"),
        ]
        _, matrix = scorer.build_correlation_matrix(positions)
        for i in range(3):
            for j in range(3):
                assert matrix[i][j] == matrix[j][i]

    def test_dimensions_match(self) -> None:
        """Matrix dimensions match number of positions."""
        scorer = EventCorrelationScorer()
        positions = [_pos(market_id=f"m{i}", event_slug=f"evt-{i}") for i in range(4)]
        ids, matrix = scorer.build_correlation_matrix(positions)
        assert len(ids) == 4
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

    def test_identity_when_all_unrelated(self) -> None:
        """All different categories and events → identity matrix (off-diag = 0)."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-2", category="SPORTS"),
            _pos(market_id="m3", event_slug="evt-3", category="TECH"),
        ]
        _, matrix = scorer.build_correlation_matrix(positions)
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert matrix[i][j] == 1.0
                else:
                    assert matrix[i][j] == 0.0

    def test_correlated_pair_in_matrix(self) -> None:
        """Same-event positions have non-zero off-diagonal entries."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="election", side="YES"),
            _pos(market_id="m2", event_slug="election", side="YES"),
        ]
        _, matrix = scorer.build_correlation_matrix(positions)
        assert matrix[0][1] == 0.8
        assert matrix[1][0] == 0.8

    def test_values_bounded(self) -> None:
        """All correlation values in [0, 1]."""
        scorer = EventCorrelationScorer()
        positions = [
            _pos(market_id="m1", event_slug="evt-1", side="YES", category="MACRO"),
            _pos(market_id="m2", event_slug="evt-1", side="NO", category="MACRO"),
            _pos(market_id="m3", event_slug="evt-2", side="YES", category="MACRO"),
        ]
        _, matrix = scorer.build_correlation_matrix(positions)
        for row in matrix:
            for val in row:
                assert 0.0 <= val <= 1.0


# ── VaR with Correlation ───────────────────────────────────────────


class TestVarWithCorrelation:

    def test_independent_var_unchanged(self) -> None:
        """No correlation matrix → same as original behavior."""
        positions = [
            _pos(market_id="m1", size_usd=100.0, current_price=0.50),
            _pos(market_id="m2", size_usd=100.0, current_price=0.50),
        ]
        result_none = calculate_portfolio_var(positions, 5000.0, correlation_matrix=None)
        result_zero = calculate_portfolio_var(
            positions, 5000.0,
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert result_none["daily_var_95"] == pytest.approx(
            result_zero["daily_var_95"], abs=0.01,
        )

    def test_correlated_var_higher(self) -> None:
        """Same-event positions → VaR increases vs independent assumption."""
        positions = [
            _pos(market_id="m1", event_slug="election", size_usd=200.0, current_price=0.50),
            _pos(market_id="m2", event_slug="election", size_usd=200.0, current_price=0.50),
        ]
        result_indep = calculate_portfolio_var(positions, 5000.0, correlation_matrix=None)
        result_corr = calculate_portfolio_var(
            positions, 5000.0,
            correlation_matrix=[[1.0, 0.8], [0.8, 1.0]],
        )
        assert result_corr["daily_var_95"] > result_indep["daily_var_95"]

    def test_perfect_correlation_max_var(self) -> None:
        """correlation=1.0 → maximum VaR."""
        positions = [
            _pos(market_id="m1", size_usd=100.0, current_price=0.50),
            _pos(market_id="m2", size_usd=100.0, current_price=0.50),
        ]
        result_1 = calculate_portfolio_var(
            positions, 5000.0,
            correlation_matrix=[[1.0, 1.0], [1.0, 1.0]],
        )
        result_0 = calculate_portfolio_var(
            positions, 5000.0,
            correlation_matrix=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert result_1["daily_var_95"] > result_0["daily_var_95"]

    def test_single_position_no_cross_terms(self) -> None:
        """Single position → no cross-correlation terms."""
        positions = [_pos(market_id="m1", size_usd=100.0, current_price=0.60)]
        result = calculate_portfolio_var(
            positions, 5000.0,
            correlation_matrix=[[1.0]],
        )
        result_none = calculate_portfolio_var(positions, 5000.0, correlation_matrix=None)
        assert result["daily_var_95"] == pytest.approx(
            result_none["daily_var_95"], abs=0.01,
        )

    def test_var_95_and_99_both_computed(self) -> None:
        """Both 95% and 99% VaR are computed."""
        positions = [_pos(market_id="m1", size_usd=100.0, current_price=0.50)]
        result = calculate_portfolio_var(positions, 5000.0)
        assert "daily_var_95" in result
        assert "daily_var_99" in result
        assert result["daily_var_99"] > result["daily_var_95"]

    def test_empty_positions_zero_var(self) -> None:
        """Empty positions → zero VaR."""
        result = calculate_portfolio_var([], 5000.0)
        assert result["daily_var_95"] == 0.0
        assert result["daily_var_99"] == 0.0

    def test_var_gate_allows_uncorrelated(self) -> None:
        """Adding unrelated position doesn't breach VaR limit."""
        existing = [
            _pos(market_id="m1", event_slug="evt-1", size_usd=50.0, category="MACRO"),
        ]
        new = _pos(market_id="m2", event_slug="evt-2", size_usd=50.0, category="SPORTS")
        scorer = EventCorrelationScorer()
        allowed, reason, details = check_var_gate(
            existing, new, bankroll=5000.0, max_var_pct=0.50,
            correlation_scorer=scorer,
        )
        assert allowed
        assert reason == "ok"

    def test_var_gate_blocks_when_limit_exceeded(self) -> None:
        """Adding position that pushes VaR over tight limit → blocked."""
        # Create enough correlated positions to push VaR high
        existing = [
            _pos(market_id=f"m{i}", event_slug="big-event", side="YES",
                 size_usd=500.0, current_price=0.50, category="MACRO")
            for i in range(5)
        ]
        new = _pos(
            market_id="m_new", event_slug="big-event", side="YES",
            size_usd=500.0, current_price=0.50, category="MACRO",
        )
        scorer = EventCorrelationScorer()
        # Very tight limit
        allowed, reason, details = check_var_gate(
            existing, new, bankroll=5000.0, max_var_pct=0.01,
            correlation_scorer=scorer,
        )
        assert not allowed
        assert "Projected VaR" in reason

    def test_var_gate_without_scorer(self) -> None:
        """VaR gate works without correlation scorer (independent assumption)."""
        existing = [_pos(market_id="m1", size_usd=100.0)]
        new = _pos(market_id="m2", size_usd=100.0)
        allowed, reason, details = check_var_gate(
            existing, new, bankroll=5000.0, max_var_pct=0.50,
            correlation_scorer=None,
        )
        assert allowed

    def test_var_gate_details_populated(self) -> None:
        """VaR gate returns details dict with expected keys."""
        existing = [_pos(market_id="m1", size_usd=100.0)]
        new = _pos(market_id="m2", size_usd=100.0)
        _, _, details = check_var_gate(
            existing, new, bankroll=5000.0, max_var_pct=0.50,
        )
        assert "current_var" in details
        assert "projected_var" in details
        assert "var_limit" in details
        assert "var_increase" in details


# ── Capital Efficiency ──────────────────────────────────────────────


class TestCapitalEfficiency:

    def test_annualized_edge_7_days(self) -> None:
        """5% edge, 7 days → ~260.7%/yr."""
        result = compute_annualized_edge(0.05, 7.0)
        assert result == pytest.approx(0.05 / (7.0 / 365.0), abs=0.1)

    def test_annualized_edge_90_days(self) -> None:
        """5% edge, 90 days → ~20.3%/yr."""
        result = compute_annualized_edge(0.05, 90.0)
        assert result == pytest.approx(0.05 / (90.0 / 365.0), abs=0.01)

    def test_zero_days_returns_infinity(self) -> None:
        """0 days → infinite annualized return."""
        result = compute_annualized_edge(0.05, 0.0)
        assert result == float("inf")

    def test_ranking_order(self) -> None:
        """Markets ranked by annualized edge, best first."""
        candidates = [
            {"market_id": "slow", "net_edge": 0.05, "days_to_resolution": 90.0},
            {"market_id": "fast", "net_edge": 0.05, "days_to_resolution": 7.0},
            {"market_id": "mid", "net_edge": 0.05, "days_to_resolution": 30.0},
        ]
        ranked = rank_markets_by_efficiency(candidates, min_annualized_edge=0.0)
        assert ranked[0].market_id == "fast"
        assert ranked[-1].market_id == "slow"
        assert ranked[0].rank == 1
        assert ranked[-1].rank == 3

    def test_filter_below_minimum(self) -> None:
        """Markets below min_annualized_edge are excluded."""
        candidates = [
            {"market_id": "good", "net_edge": 0.05, "days_to_resolution": 7.0},
            {"market_id": "bad", "net_edge": 0.01, "days_to_resolution": 90.0},
        ]
        ranked = rank_markets_by_efficiency(candidates, min_annualized_edge=0.50)
        ids = [r.market_id for r in ranked]
        assert "good" in ids
        assert "bad" not in ids


# ── Config defaults ─────────────────────────────────────────────────


class TestConfigDefaults:

    def test_var_gate_disabled_by_default(self) -> None:
        """var_gate_enabled defaults to False."""
        cfg = PortfolioConfig()
        assert cfg.var_gate_enabled is False

    def test_max_var_pct_default(self) -> None:
        """max_portfolio_var_pct defaults to 0.10."""
        cfg = PortfolioConfig()
        assert cfg.max_portfolio_var_pct == 0.10

    def test_correlation_values_default(self) -> None:
        """Default correlation values match spec."""
        cfg = PortfolioConfig()
        assert cfg.same_event_same_outcome_corr == 0.8
        assert cfg.same_event_diff_outcome_corr == 0.3
        assert cfg.same_category_corr == 0.15

    def test_backward_compatible(self) -> None:
        """Existing portfolio config fields still work."""
        cfg = PortfolioConfig()
        assert cfg.max_category_exposure_pct == 0.35
        assert cfg.max_single_event_exposure_pct == 0.25
        assert cfg.max_correlated_positions == 4
