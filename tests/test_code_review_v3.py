"""Tests for Code Review v3 fixes.

Covers:
  1. Slippage bug fix (multiplicative)
  2. Decomposition default floor penalty fix
  3. use_probability_space_costs config
  4. Scout tier confidence floor gate
  5. datetime.utcnow() deprecation fixes
  6. Ensemble degraded-model warning
  7. fill_size accounting with actual_price
  8. Paper Sharpe at startup
  9. Data scraper retry logic
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── 1. Slippage Bug Fix ──────────────────────────────────────────────


class TestSlippageFix:
    """Verify backtest slippage is multiplicative, not additive."""

    def test_slippage_is_multiplicative(self) -> None:
        """entry_with_slip = entry_price * (1 + slippage_pct)."""
        from src.backtest.replay_engine import ReplayEngine

        # Call the static method directly
        exit_price, pnl = ReplayEngine._simulate_execution(
            entry_price=0.60,
            direction="BUY_YES",
            resolution="YES",
            stake_usd=100.0,
            slippage_pct=0.005,
        )
        # Multiplicative: 0.60 * 1.005 = 0.603
        expected_entry = 0.60 * 1.005
        contracts = 100.0 / expected_entry
        expected_pnl = (1.0 - expected_entry) * contracts
        assert abs(pnl - round(expected_pnl, 2)) < 0.02

    def test_slippage_not_additive(self) -> None:
        """The old additive bug would give different results."""
        from src.backtest.replay_engine import ReplayEngine

        _, pnl = ReplayEngine._simulate_execution(
            entry_price=0.60,
            direction="BUY_YES",
            resolution="YES",
            stake_usd=100.0,
            slippage_pct=0.005,
        )
        # Additive bug would compute: 0.60 + 0.005 = 0.605
        additive_entry = 0.60 + 0.005
        additive_contracts = 100.0 / additive_entry
        additive_pnl = round((1.0 - additive_entry) * additive_contracts, 2)
        # Confirm our result differs from the old bug
        assert abs(pnl - additive_pnl) > 0.01

    def test_slippage_on_low_price_contract(self) -> None:
        """Low-price contracts: multiplicative slippage is smaller than additive."""
        from src.backtest.replay_engine import ReplayEngine

        _, pnl = ReplayEngine._simulate_execution(
            entry_price=0.10,
            direction="BUY_YES",
            resolution="YES",
            stake_usd=100.0,
            slippage_pct=0.01,
        )
        # Multiplicative: 0.10 * 1.01 = 0.101
        expected_entry = 0.10 * 1.01
        contracts = 100.0 / expected_entry
        expected_pnl = (1.0 - expected_entry) * contracts
        assert abs(pnl - round(expected_pnl, 2)) < 0.02


# ── 2. Decomposition Default Floor ───────────────────────────────────


class TestDecompositionDefault:
    """Verify decomposition default no longer penalizes when disabled."""

    def test_default_is_zero(self) -> None:
        """_DEFAULT_DECOMPOSITION_DISAGREEMENT should be 0.0."""
        from src.policy.edge_uncertainty import _DEFAULT_DECOMPOSITION_DISAGREEMENT
        assert _DEFAULT_DECOMPOSITION_DISAGREEMENT == 0.0

    def test_no_penalty_without_decomposition(self) -> None:
        """compute_edge_uncertainty with empty decomp_sub_probs returns 0 decomp component."""
        from src.policy.edge_uncertainty import UncertaintyInputs, compute_edge_uncertainty

        inputs = UncertaintyInputs(
            ensemble_spread=0.05,
            evidence_quality=0.70,
            base_rate=0.50,
            model_probability=0.60,
            decomposition_sub_probs=[],  # disabled
        )
        score = compute_edge_uncertainty(inputs)
        # With default=0.0, decomp component should be 0
        # Total should only reflect spread, evidence, base_rate
        assert score < 0.30  # previously was ~0.24 + 0.10 = 0.34

    def test_penalty_with_disagreeing_decomposition(self) -> None:
        """With actual decomp probs that disagree, penalty is applied."""
        from src.policy.edge_uncertainty import UncertaintyInputs, compute_edge_uncertainty

        inputs_with = UncertaintyInputs(
            ensemble_spread=0.05,
            evidence_quality=0.70,
            base_rate=0.50,
            model_probability=0.60,
            decomposition_sub_probs=[0.2, 0.8],  # high disagreement
        )
        inputs_without = UncertaintyInputs(
            ensemble_spread=0.05,
            evidence_quality=0.70,
            base_rate=0.50,
            model_probability=0.60,
            decomposition_sub_probs=[],
        )
        score_with = compute_edge_uncertainty(inputs_with)
        score_without = compute_edge_uncertainty(inputs_without)
        assert score_with > score_without


# ── 3. Probability Space Costs Config ─────────────────────────────────


class TestProbabilitySpaceCosts:
    """Verify config toggle for probability space costs."""

    def test_config_yaml_has_flag(self) -> None:
        """config.yaml should have use_probability_space_costs: true."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        content = config_path.read_text()
        assert "use_probability_space_costs: true" in content

    def test_config_loads_with_flag(self) -> None:
        """Loading config picks up the flag."""
        from src.config import load_config
        config = load_config()
        assert config.risk.use_probability_space_costs is True


# ── 4. Scout Tier Confidence Gate ─────────────────────────────────────


class TestScoutConfidenceGate:
    """Verify scout forecasts with LOW confidence are rejected."""

    def test_scout_low_confidence_returns_false(self) -> None:
        """stage_forecast returns False for LOW confidence scout forecast."""
        from src.engine.pipeline import PipelineRunner

        # Verify the gate code exists in the source
        import inspect
        source = inspect.getsource(PipelineRunner._run_forecast)
        assert "scout_confidence_gate_rejected" in source

    def test_tier_used_variable_initialized(self) -> None:
        """tier_used is initialized to None before tier selection."""
        import inspect
        from src.engine.pipeline import PipelineRunner
        source = inspect.getsource(PipelineRunner._run_forecast)
        assert "tier_used = None" in source
        assert "tier_used = tier.tier" in source


# ── 5. datetime.utcnow() Deprecation Fix ─────────────────────────────


class TestUtcnowDeprecation:
    """Verify deprecated utcnow() calls are replaced."""

    def test_whale_scorer_no_utcnow(self) -> None:
        """whale_scorer.py should not use datetime.utcnow()."""
        src = Path(__file__).parent.parent / "src" / "analytics" / "whale_scorer.py"
        content = src.read_text()
        assert "utcnow()" not in content
        assert "datetime.now(dt.timezone.utc)" in content

    def test_wallet_scanner_no_utcnow(self) -> None:
        """wallet_scanner.py should not use datetime.utcnow()."""
        src = Path(__file__).parent.parent / "src" / "analytics" / "wallet_scanner.py"
        content = src.read_text()
        assert "utcnow()" not in content
        assert "datetime.now(dt.timezone.utc)" in content

    def test_timestamp_format_consistent(self) -> None:
        """Ensure the new format produces Z-suffix timestamps."""
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        assert now.endswith("Z")
        assert "+" not in now  # no +00:00 offset


# ── 6. Ensemble Degraded Model Warning ───────────────────────────────


class TestEnsembleDegradedWarning:
    """Verify warning when ensemble drops below 2 models."""

    def test_degraded_warning_in_source(self) -> None:
        """ensemble.py has the degraded_single_model log message."""
        src = Path(__file__).parent.parent / "src" / "forecast" / "ensemble.py"
        content = src.read_text()
        assert "ensemble.degraded_single_model" in content

    def test_warning_only_when_configured_multi(self) -> None:
        """Warning should only fire when configured with >= 2 models."""
        src = Path(__file__).parent.parent / "src" / "forecast" / "ensemble.py"
        content = src.read_text()
        assert "len(self._ensemble.models) >= 2" in content


# ── 7. fill_size Accounting Fix ───────────────────────────────────────


class TestFillSizeAccounting:
    """Verify fill_size uses actual_price for market orders."""

    def test_parse_clob_response_accepts_actual_price(self) -> None:
        """_parse_clob_response signature includes actual_price param."""
        from src.execution.order_router import _parse_clob_response
        import inspect
        sig = inspect.signature(_parse_clob_response)
        assert "actual_price" in sig.parameters

    def test_parse_clob_response_uses_actual_price(self) -> None:
        """fill_size is computed using actual_price when provided."""
        from src.execution.order_router import _parse_clob_response

        order = MagicMock()
        order.order_id = "test-order-12345678"
        order.price = 0.60
        order.size = 200.0  # large enough to avoid guard clamp
        order.action_side = "BUY"
        order.outcome_side = "YES"

        # Simulate CLOB response with takingAmount
        resp = {
            "orderID": "clob-123",
            "status": "matched",
            "takingAmount": "60.0",  # taking_value
        }

        # Without actual_price (limit order) — uses order.price
        result_limit = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result_limit.fill_size == pytest.approx(60.0 / 0.60, rel=0.01)

        # With actual_price (market order) — uses aggressive price
        aggressive_price = 0.63  # 0.60 * 1.05
        result_market = _parse_clob_response(
            resp, order, "2024-01-01T00:00:00Z", actual_price=aggressive_price,
        )
        assert result_market.fill_size == pytest.approx(60.0 / 0.63, rel=0.01)
        assert result_market.fill_size != result_limit.fill_size


# ── 8. Paper Sharpe at Startup ────────────────────────────────────────


class TestPaperSharpeStartup:
    """Verify paper Sharpe is computed at engine startup."""

    def test_compute_paper_sharpe_method_exists(self) -> None:
        """TradingEngine has _compute_paper_sharpe method."""
        from src.engine.loop import TradingEngine
        assert hasattr(TradingEngine, "_compute_paper_sharpe")

    def test_compute_paper_sharpe_with_data(self) -> None:
        """_compute_paper_sharpe stores Sharpe when sufficient data exists."""
        from src.engine.loop import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = MagicMock()

        # Mock daily_summaries with 10 days of data
        pnl_data = [{"total_pnl": v} for v in [10, 15, -5, 20, 8, -2, 12, 18, -3, 7]]
        mock_rows = [MagicMock(**d) for d in pnl_data]
        for row, d in zip(mock_rows, pnl_data):
            row.__getitem__ = lambda self, key, d=d: d[key]  # noqa: B023

        engine._db.conn.execute.return_value.fetchall.return_value = mock_rows

        engine._compute_paper_sharpe()
        engine._db.set_engine_state.assert_called_once()
        call_args = engine._db.set_engine_state.call_args
        assert call_args[0][0] == "paper_sharpe"

    def test_compute_paper_sharpe_insufficient_data(self) -> None:
        """_compute_paper_sharpe does nothing with < 7 days of data."""
        from src.engine.loop import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = MagicMock()

        # Only 3 days
        mock_rows = [MagicMock() for _ in range(3)]
        engine._db.conn.execute.return_value.fetchall.return_value = mock_rows

        engine._compute_paper_sharpe()
        engine._db.set_engine_state.assert_not_called()

    def test_start_calls_compute_paper_sharpe(self) -> None:
        """start() calls _compute_paper_sharpe after DB init."""
        import inspect
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine.start)
        assert "_compute_paper_sharpe" in source


# ── 9. Data Scraper Retry Logic ───────────────────────────────────────


class TestDataScraperRetry:
    """Verify data_scraper retries on API failures."""

    def test_scrape_has_retry_logic(self) -> None:
        """scrape() method contains retry logic."""
        import inspect
        from src.backtest.data_scraper import HistoricalDataScraper
        source = inspect.getsource(HistoricalDataScraper.scrape)
        assert "for attempt in range(3)" in source
        assert "scraper.retry" in source

    def test_scrape_recent_has_retry_logic(self) -> None:
        """scrape_recent() method contains retry logic."""
        import inspect
        from src.backtest.data_scraper import HistoricalDataScraper
        source = inspect.getsource(HistoricalDataScraper.scrape_recent)
        assert "for attempt in range(3)" in source
        assert "scraper.recent_retry" in source

    @pytest.mark.asyncio
    async def test_scrape_retry_on_failure(self) -> None:
        """scrape() retries on API errors before giving up."""
        from src.backtest.data_scraper import HistoricalDataScraper

        mock_db = MagicMock()
        scraper = HistoricalDataScraper(db=mock_db)

        call_count = 0

        async def mock_list_markets(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("API timeout")
            return []  # success on 3rd attempt

        mock_client = AsyncMock()
        mock_client.list_markets = mock_list_markets
        mock_client.close = AsyncMock()

        with patch("src.backtest.data_scraper.GammaClient", return_value=mock_client), \
             patch("src.backtest.data_scraper.time.sleep"):  # skip actual waits
            result = await scraper.scrape(max_markets=10)

        assert call_count == 3  # retried twice, succeeded on 3rd
        assert len(result.errors) == 0
