"""Tests for backtest Pydantic models (Phase 1)."""

from __future__ import annotations

import pytest

from src.backtest.models import (
    BacktestRunRecord,
    BacktestTradeRecord,
    HistoricalMarketRecord,
    LLMCacheRecord,
)


class TestHistoricalMarketRecord:

    def test_defaults(self) -> None:
        r = HistoricalMarketRecord(condition_id="abc", question="Will X happen?")
        assert r.condition_id == "abc"
        assert r.question == "Will X happen?"
        assert r.resolution == ""
        assert r.volume_usd == 0.0
        assert r.raw_json == "{}"

    def test_full_record(self) -> None:
        r = HistoricalMarketRecord(
            condition_id="0x123",
            question="Will BTC hit 100k?",
            description="Bitcoin price prediction market",
            category="Crypto",
            market_type="MACRO",
            resolution="YES",
            resolved_at="2024-12-01T00:00:00Z",
            volume_usd=50000.0,
            liquidity_usd=10000.0,
            slug="btc-100k",
        )
        assert r.resolution == "YES"
        assert r.volume_usd == 50000.0

    def test_serialization(self) -> None:
        r = HistoricalMarketRecord(
            condition_id="test", question="Q?", resolution="NO",
        )
        d = r.model_dump()
        assert d["condition_id"] == "test"
        assert d["resolution"] == "NO"
        reconstructed = HistoricalMarketRecord(**d)
        assert reconstructed == r


class TestLLMCacheRecord:

    def test_defaults(self) -> None:
        r = LLMCacheRecord(
            cache_key="abc123",
            market_question_hash="qhash",
            model_name="gpt-4o",
            prompt_hash="phash",
        )
        assert r.cache_key == "abc123"
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.response_json == "{}"

    def test_with_token_counts(self) -> None:
        r = LLMCacheRecord(
            cache_key="key",
            market_question_hash="qh",
            model_name="claude-3.5-sonnet",
            prompt_hash="ph",
            input_tokens=500,
            output_tokens=200,
            latency_ms=1234.5,
        )
        assert r.input_tokens == 500
        assert r.output_tokens == 200
        assert r.latency_ms == 1234.5

    def test_serialization(self) -> None:
        r = LLMCacheRecord(
            cache_key="k", market_question_hash="q",
            model_name="m", prompt_hash="p",
        )
        d = r.model_dump()
        assert d["model_name"] == "m"
        reconstructed = LLMCacheRecord(**d)
        assert reconstructed == r


class TestBacktestRunRecord:

    def test_defaults(self) -> None:
        r = BacktestRunRecord(run_id="run-001")
        assert r.run_id == "run-001"
        assert r.status == "pending"
        assert r.brier_score == 0.0
        assert r.total_pnl == 0.0

    def test_completed_run(self) -> None:
        r = BacktestRunRecord(
            run_id="run-002",
            name="baseline",
            status="completed",
            markets_processed=1000,
            markets_traded=250,
            total_pnl=150.50,
            brier_score=0.22,
            win_rate=0.58,
            sharpe_ratio=1.2,
            max_drawdown_pct=0.08,
        )
        assert r.status == "completed"
        assert r.win_rate == 0.58

    def test_serialization(self) -> None:
        r = BacktestRunRecord(run_id="r1", name="test")
        d = r.model_dump()
        reconstructed = BacktestRunRecord(**d)
        assert reconstructed == r


class TestBacktestTradeRecord:

    def test_defaults(self) -> None:
        r = BacktestTradeRecord(
            run_id="run-001", market_condition_id="0xabc",
        )
        assert r.model_probability == 0.5
        assert r.direction == ""
        assert r.forecast_correct is False

    def test_winning_trade(self) -> None:
        r = BacktestTradeRecord(
            run_id="run-001",
            market_condition_id="0xabc",
            direction="BUY_YES",
            model_probability=0.75,
            implied_probability=0.50,
            edge=0.25,
            stake_usd=100.0,
            entry_price=0.50,
            exit_price=1.0,
            pnl=100.0,
            resolution="YES",
            actual_outcome=1.0,
            forecast_correct=True,
        )
        assert r.forecast_correct is True
        assert r.pnl == 100.0
        assert r.direction == "BUY_YES"

    def test_losing_trade(self) -> None:
        r = BacktestTradeRecord(
            run_id="run-001",
            market_condition_id="0xdef",
            direction="BUY_YES",
            model_probability=0.70,
            entry_price=0.50,
            exit_price=0.0,
            pnl=-100.0,
            resolution="NO",
            actual_outcome=0.0,
            forecast_correct=False,
        )
        assert r.forecast_correct is False
        assert r.pnl == -100.0

    def test_serialization(self) -> None:
        r = BacktestTradeRecord(
            run_id="r1", market_condition_id="c1",
            forecast_correct=True,
        )
        d = r.model_dump()
        assert d["forecast_correct"] is True
        reconstructed = BacktestTradeRecord(**d)
        assert reconstructed == r
