"""Tests for backtest CLI commands (Phase 1 — Batch 4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


class TestBacktestGroup:

    def test_backtest_help(self, runner: CliRunner) -> None:
        """Backtest group shows help with subcommands."""
        result = runner.invoke(cli, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "scrape" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "compare" in result.output
        assert "cache-stats" in result.output

    def test_backtest_group_registered(self, runner: CliRunner) -> None:
        """Backtest is a registered subgroup of the CLI."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "backtest" in result.output


class TestBacktestScrape:

    @patch("src.backtest.database.BacktestDatabase")
    @patch("src.backtest.data_scraper.HistoricalDataScraper")
    def test_scrape_invokes_scraper(
        self, mock_scraper_cls: MagicMock, mock_db_cls: MagicMock, runner: CliRunner,
    ) -> None:
        """scrape command calls HistoricalDataScraper.scrape()."""
        from src.backtest.data_scraper import ScrapeResult

        mock_db = MagicMock()
        mock_db.count_historical_markets.return_value = 42
        mock_db_cls.return_value = mock_db

        scrape_result = ScrapeResult(
            total_fetched=100, new_inserted=80,
            duplicates_skipped=15, invalid_skipped=5,
            duration_secs=12.3,
        )
        mock_scraper = MagicMock()
        mock_scraper.scrape = AsyncMock(return_value=scrape_result)
        mock_scraper_cls.return_value = mock_scraper

        result = runner.invoke(cli, ["backtest", "scrape", "--max-markets", "100"])
        assert result.exit_code == 0
        assert "100" in result.output
        assert "80" in result.output
        mock_scraper.scrape.assert_awaited_once()
        mock_db.close.assert_called_once()

    @patch("src.backtest.database.BacktestDatabase")
    @patch("src.backtest.data_scraper.HistoricalDataScraper")
    def test_scrape_custom_min_volume(
        self, mock_scraper_cls: MagicMock, mock_db_cls: MagicMock, runner: CliRunner,
    ) -> None:
        """scrape command passes --min-volume to scraper."""
        from src.backtest.data_scraper import ScrapeResult

        mock_db = MagicMock()
        mock_db.count_historical_markets.return_value = 0
        mock_db_cls.return_value = mock_db

        mock_scraper = MagicMock()
        mock_scraper.scrape = AsyncMock(return_value=ScrapeResult())
        mock_scraper_cls.return_value = mock_scraper

        result = runner.invoke(cli, [
            "backtest", "scrape", "--max-markets", "10", "--min-volume", "5000",
        ])
        assert result.exit_code == 0
        mock_scraper_cls.assert_called_once_with(mock_db, min_volume=5000.0)


class TestBacktestList:

    @patch("src.backtest.database.BacktestDatabase")
    def test_list_empty(self, mock_db_cls: MagicMock, runner: CliRunner) -> None:
        """list shows message when no runs exist."""
        mock_db = MagicMock()
        mock_db.get_backtest_runs.return_value = []
        mock_db_cls.return_value = mock_db

        result = runner.invoke(cli, ["backtest", "list"])
        assert result.exit_code == 0
        assert "No backtest runs found" in result.output

    @patch("src.backtest.database.BacktestDatabase")
    def test_list_shows_runs(self, mock_db_cls: MagicMock, runner: CliRunner) -> None:
        """list shows table of backtest runs."""
        from src.backtest.models import BacktestRunRecord

        mock_db = MagicMock()
        mock_db.get_backtest_runs.return_value = [
            BacktestRunRecord(
                run_id="abc123",
                name="test-run",
                status="completed",
                markets_processed=50,
                markets_traded=20,
                total_pnl=123.45,
                brier_score=0.2100,
                sharpe_ratio=1.2300,
                started_at="2024-06-01T00:00:00",
            ),
        ]
        mock_db_cls.return_value = mock_db

        result = runner.invoke(cli, ["backtest", "list"])
        assert result.exit_code == 0
        assert "abc123" in result.output
        # Rich may truncate column values; check partial match
        assert "test-" in result.output
        assert "compl" in result.output


class TestBacktestCompare:

    @patch("src.backtest.database.BacktestDatabase")
    def test_compare_not_found(self, mock_db_cls: MagicMock, runner: CliRunner) -> None:
        """compare shows error when run not found."""
        mock_db = MagicMock()
        mock_db.get_backtest_run.return_value = None
        mock_db_cls.return_value = mock_db

        result = runner.invoke(cli, ["backtest", "compare", "aaa", "bbb"])
        assert result.exit_code == 0
        assert "Run not found" in result.output

    @patch("src.backtest.database.BacktestDatabase")
    def test_compare_shows_delta(self, mock_db_cls: MagicMock, runner: CliRunner) -> None:
        """compare shows side-by-side metrics with deltas."""
        from src.backtest.models import BacktestRunRecord

        run_a = BacktestRunRecord(
            run_id="aaa", name="baseline", status="completed",
            total_pnl=100.0, win_rate=0.6, brier_score=0.2,
            sharpe_ratio=1.0, max_drawdown_pct=0.1,
            markets_traded=50, started_at="2024-01-01T00:00:00",
        )
        run_b = BacktestRunRecord(
            run_id="bbb", name="improved", status="completed",
            total_pnl=200.0, win_rate=0.7, brier_score=0.15,
            sharpe_ratio=1.5, max_drawdown_pct=0.08,
            markets_traded=50, started_at="2024-01-01T00:00:00",
        )
        mock_db = MagicMock()
        mock_db.get_backtest_run.side_effect = lambda rid: run_a if rid == "aaa" else run_b
        mock_db_cls.return_value = mock_db

        result = runner.invoke(cli, ["backtest", "compare", "aaa", "bbb"])
        assert result.exit_code == 0
        assert "baseline" in result.output
        assert "improved" in result.output


class TestBacktestCacheStats:

    @patch("src.backtest.database.BacktestDatabase")
    def test_cache_stats(self, mock_db_cls: MagicMock, runner: CliRunner) -> None:
        """cache-stats shows LLM cache statistics."""
        mock_db = MagicMock()
        mock_db.get_llm_cache_stats.return_value = {
            "total_entries": 500,
            "distinct_models": 3,
        }
        mock_db_cls.return_value = mock_db

        result = runner.invoke(cli, ["backtest", "cache-stats"])
        assert result.exit_code == 0
        assert "500" in result.output
        assert "3" in result.output
