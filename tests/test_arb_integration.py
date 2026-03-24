"""Integration tests for arbitrage storage, migration, engine wiring, and dashboard.

Covers:
  - ArbitrageConfig loading + defaults + backward compat
  - Migration 11: arb tables created
  - Arb CRUD methods on Database
  - Arb summary stats
  - Engine loop integration (config gating, scanner init, get_status)
  - Dashboard API endpoints (/api/arbitrage, /api/arbitrage/summary, /api/arbitrage/matches)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config import ArbitrageConfig, BotConfig, load_config
from src.storage.migrations import run_migrations, SCHEMA_VERSION, _get_current_version
from src.storage.models import (
    ArbOpportunityRecord,
    ArbTradeRecord,
    ComplementaryArbRecord,
)
from src.storage.database import Database
from src.config import StorageConfig


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_arb.db")


@pytest.fixture
def db(tmp_db_path: str) -> Database:
    """Return a connected Database with all migrations applied."""
    database = Database(StorageConfig(sqlite_path=tmp_db_path))
    database.connect()
    yield database  # type: ignore[misc]
    database.close()


@pytest.fixture
def conn(tmp_db_path: str) -> sqlite3.Connection:
    """Return a raw connection with migrations applied."""
    c = sqlite3.connect(tmp_db_path)
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c  # type: ignore[misc]
    c.close()


# ── ArbitrageConfig Tests ────────────────────────────────────────────


class TestArbConfigIntegration:
    def test_disabled_by_default(self) -> None:
        config = BotConfig()
        assert config.arbitrage.enabled is False
        assert config.arbitrage.kalshi_paper_mode is True

    def test_defaults_match_spec(self) -> None:
        arb = ArbitrageConfig()
        assert arb.min_arb_edge == 0.03
        assert arb.polymarket_fee_pct == 0.02
        assert arb.kalshi_fee_pct == 0.02
        assert arb.max_arb_position_usd == 200.0
        assert arb.max_arb_positions_count == 5
        assert arb.complementary_threshold == 0.97
        assert arb.correlated_min_divergence == 0.10
        assert arb.match_min_confidence == 0.6

    def test_fee_warning(self) -> None:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ArbitrageConfig(enabled=True, min_arb_edge=0.03)
            fee_warnings = [x for x in w if "min_arb_edge" in str(x.message)]
            assert len(fee_warnings) == 1

    def test_backward_compat(self) -> None:
        """Loading default config still works (arbitrage section optional in YAML)."""
        config = load_config()
        assert hasattr(config, "arbitrage")
        assert config.arbitrage.enabled is False

    def test_botconfig_has_arbitrage_field(self) -> None:
        config = BotConfig()
        assert isinstance(config.arbitrage, ArbitrageConfig)


# ── Migration 11 Tests ───────────────────────────────────────────────


class TestArbMigration:
    def test_schema_version_bumped(self) -> None:
        assert SCHEMA_VERSION >= 11

    def test_tables_created(self, conn: sqlite3.Connection) -> None:
        version = _get_current_version(conn)
        assert version >= 11

        # Check arb_opportunities table exists
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arb_opportunities'"
        ).fetchall()
        assert len(rows) == 1

        # Check arb_trades table exists
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='arb_trades'"
        ).fetchall()
        assert len(rows) == 1

        # Check complementary_arb table exists
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='complementary_arb'"
        ).fetchall()
        assert len(rows) == 1

    def test_migration_idempotent(self, conn: sqlite3.Connection) -> None:
        """Running migrations twice doesn't fail."""
        run_migrations(conn)
        version = _get_current_version(conn)
        assert version >= 11

    def test_indexes_created(self, conn: sqlite3.Connection) -> None:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_arb%'"
        ).fetchall()
        index_names = {r[0] for r in indexes}
        assert "idx_arb_opp_created" in index_names
        assert "idx_arb_opp_actionable" in index_names
        assert "idx_arb_trades_arb_id" in index_names
        assert "idx_arb_trades_status" in index_names


# ── Storage CRUD Tests ───────────────────────────────────────────────


class TestArbStorageIntegration:
    def test_insert_arb_opportunity(self, db: Database) -> None:
        record = ArbOpportunityRecord(
            arb_id="arb-test-001",
            match_method="keyword",
            match_confidence=0.85,
            poly_market_id="poly-1",
            poly_question="Will it rain?",
            poly_yes_price=0.40,
            poly_no_price=0.60,
            kalshi_ticker="KXRAIN",
            kalshi_title="Rain tomorrow?",
            kalshi_yes_price=0.50,
            kalshi_no_price=0.50,
            spread=0.10,
            net_spread=0.06,
            direction="BUY_POLY_YES_SELL_KALSHI_YES",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_price=0.40,
            sell_price=0.50,
            total_fees=0.04,
            is_actionable=True,
        )
        db.insert_arb_opportunity(record)

        rows = db.get_arb_opportunities(limit=10)
        assert len(rows) == 1
        assert rows[0]["arb_id"] == "arb-test-001"
        assert rows[0]["is_actionable"] == 1

    def test_insert_arb_trade(self, db: Database) -> None:
        record = ArbTradeRecord(
            arb_id="arb-trade-001",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_market_id="poly-1",
            sell_market_id="KXRAIN",
            buy_price=0.40,
            sell_price=0.50,
            buy_fill_price=0.40,
            sell_fill_price=0.49,
            stake_usd=100.0,
            net_pnl=5.0,
            status="both_filled",
        )
        db.insert_arb_trade(record)

        rows = db.get_arb_trades(limit=10)
        assert len(rows) == 1
        assert rows[0]["arb_id"] == "arb-trade-001"
        assert rows[0]["status"] == "both_filled"
        assert rows[0]["net_pnl"] == 5.0

    def test_insert_complementary_arb(self, db: Database) -> None:
        record = ComplementaryArbRecord(
            market_id="market-comp-1",
            question="Will Bitcoin hit $100k?",
            yes_price=0.45,
            no_price=0.45,
            combined_cost=0.90,
            guaranteed_profit=0.10,
            fee_cost=0.04,
            net_profit=0.06,
            is_actionable=True,
        )
        db.insert_complementary_arb(record)

        rows = db.get_complementary_arb(limit=10)
        assert len(rows) == 1
        assert rows[0]["market_id"] == "market-comp-1"
        assert rows[0]["is_actionable"] == 1

    def test_get_arb_opportunities_actionable_only(self, db: Database) -> None:
        # Insert one actionable and one not
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="arb-act", is_actionable=True, spread=0.1, net_spread=0.06,
        ))
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="arb-not", is_actionable=False, spread=0.01, net_spread=-0.03,
        ))

        all_rows = db.get_arb_opportunities(limit=10, actionable_only=False)
        assert len(all_rows) == 2

        actionable_rows = db.get_arb_opportunities(limit=10, actionable_only=True)
        assert len(actionable_rows) == 1
        assert actionable_rows[0]["arb_id"] == "arb-act"

    def test_get_arb_summary(self, db: Database) -> None:
        # Insert mixed data
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="opp-1", is_actionable=True,
        ))
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="opp-2", is_actionable=False,
        ))
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="t-1", status="both_filled", net_pnl=10.0,
        ))
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="t-2", status="failed", net_pnl=-2.0,
        ))
        db.insert_complementary_arb(ComplementaryArbRecord(
            market_id="c-1", is_actionable=True,
        ))

        summary = db.get_arb_summary()
        assert summary["total_opportunities"] == 2
        assert summary["actionable_opportunities"] == 1
        assert summary["total_trades"] == 2
        assert summary["filled_trades"] == 1
        assert abs(summary["combined_pnl"] - 8.0) < 0.01
        assert summary["complementary_actionable"] == 1

    def test_empty_summary(self, db: Database) -> None:
        summary = db.get_arb_summary()
        assert summary["total_opportunities"] == 0
        assert summary["total_trades"] == 0
        assert summary["combined_pnl"] == 0.0

    def test_arb_opportunity_upsert(self, db: Database) -> None:
        """Inserting with same arb_id replaces the record."""
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="arb-dup", spread=0.05, is_actionable=False,
        ))
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="arb-dup", spread=0.15, is_actionable=True,
        ))
        rows = db.get_arb_opportunities(limit=10)
        assert len(rows) == 1
        assert rows[0]["spread"] == 0.15

    def test_multiple_trades(self, db: Database) -> None:
        for i in range(5):
            db.insert_arb_trade(ArbTradeRecord(
                arb_id=f"t-{i}", status="both_filled", net_pnl=float(i),
            ))
        rows = db.get_arb_trades(limit=3)
        assert len(rows) == 3


# ── Engine Integration Tests ─────────────────────────────────────────


class TestEngineArbIntegration:
    def test_engine_status_includes_arb_when_disabled(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig()
        engine = TradingEngine(config)
        status = engine.get_status()
        assert "arbitrage" in status
        assert status["arbitrage"]["enabled"] is False

    def test_engine_scanner_none_when_disabled(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig()
        engine = TradingEngine(config)
        assert engine._cross_platform_scanner is None

    def test_engine_scanner_created_when_enabled(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig(arbitrage=ArbitrageConfig(enabled=True))
        engine = TradingEngine(config)
        assert engine._cross_platform_scanner is not None

    def test_engine_status_active_positions(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig(arbitrage=ArbitrageConfig(enabled=True))
        engine = TradingEngine(config)
        status = engine.get_status()
        assert status["arbitrage"]["active_arb_positions"] == 0

    def test_engine_tracks_complementary_arb(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig()
        engine = TradingEngine(config)
        assert engine._latest_complementary_arb == []

    def test_engine_tracks_correlated_mispricings(self) -> None:
        from src.engine.loop import TradingEngine
        config = BotConfig()
        engine = TradingEngine(config)
        assert engine._latest_correlated_mispricings == []


# ── Dashboard API Tests ──────────────────────────────────────────────


class TestArbDashboardEndpoints:
    @pytest.fixture
    def client(self, tmp_db_path: str) -> Any:
        """Flask test client with a temp database."""
        import src.dashboard.app as dashboard_app
        dashboard_app._db_path = tmp_db_path

        # Run migrations on the temp db
        conn = sqlite3.connect(tmp_db_path)
        run_migrations(conn)
        conn.close()

        dashboard_app.app.config["TESTING"] = True
        with dashboard_app.app.test_client() as c:
            yield c

    def test_api_arbitrage_empty(self, client: Any) -> None:
        resp = client.get("/api/arbitrage")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "opportunities" in data
        assert "trades" in data
        assert "complementary" in data
        assert "summary" in data
        assert data["opportunities"] == []

    def test_api_arbitrage_with_data(self, client: Any, tmp_db_path: str) -> None:
        # Insert data directly
        db = Database(StorageConfig(sqlite_path=tmp_db_path))
        db.connect()
        db.insert_arb_opportunity(ArbOpportunityRecord(
            arb_id="arb-dash-1", is_actionable=True, spread=0.1,
        ))
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="arb-dash-1", status="both_filled", net_pnl=5.0,
        ))
        db.close()

        resp = client.get("/api/arbitrage")
        data = resp.get_json()
        assert len(data["opportunities"]) == 1
        assert len(data["trades"]) == 1
        assert data["summary"]["total_opportunities"] == 1

    def test_api_arbitrage_summary_empty(self, client: Any) -> None:
        resp = client.get("/api/arbitrage/summary")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_opportunities"] == 0
        assert data["combined_pnl"] == 0.0

    def test_api_arbitrage_matches_no_engine(self, client: Any) -> None:
        import src.dashboard.app as dashboard_app
        dashboard_app._engine_instance = None

        resp = client.get("/api/arbitrage/matches")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["matches"] == []

    def test_api_arbitrage_matches_no_scanner(self, client: Any) -> None:
        import src.dashboard.app as dashboard_app
        mock_engine = MagicMock()
        mock_engine._cross_platform_scanner = None
        dashboard_app._engine_instance = mock_engine

        resp = client.get("/api/arbitrage/matches")
        data = resp.get_json()
        assert data["matches"] == []
        assert data["enabled"] is False

        dashboard_app._engine_instance = None

    def test_api_arbitrage_matches_with_scanner(self, client: Any) -> None:
        import src.dashboard.app as dashboard_app
        from src.policy.cross_platform_arb import (
            CrossPlatformArbOpportunity,
            PairedTradeResult,
        )
        from src.connectors.market_matcher import MarketMatch

        mock_opp = CrossPlatformArbOpportunity(
            match=MarketMatch(
                polymarket_id="p1", polymarket_question="Q?",
                kalshi_ticker="KX1", kalshi_title="Q?",
                match_method="keyword", match_confidence=0.9,
            ),
            poly_yes_price=0.40, poly_no_price=0.60,
            kalshi_yes_price=0.50, kalshi_no_price=0.50,
            spread=0.10, net_spread=0.06,
            direction="BUY_POLY_YES_SELL_KALSHI_YES",
            buy_platform="polymarket", sell_platform="kalshi",
            buy_price=0.40, sell_price=0.50,
            total_fees=0.04, is_actionable=True,
        )

        mock_scanner = MagicMock()
        mock_scanner.opportunity_log = [mock_opp]
        mock_scanner.active_positions = []

        mock_engine = MagicMock()
        mock_engine._cross_platform_scanner = mock_scanner
        dashboard_app._engine_instance = mock_engine

        resp = client.get("/api/arbitrage/matches")
        data = resp.get_json()
        assert len(data["matches"]) == 1
        assert data["enabled"] is True
        assert data["active_positions"] == 0

        dashboard_app._engine_instance = None


# ── Paired Trade Logging Tests ───────────────────────────────────────


class TestPairedTradeLogging:
    def test_both_legs_logged(self, db: Database) -> None:
        """Both legs of a trade logged as a single record."""
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="paired-1",
            buy_platform="polymarket",
            sell_platform="kalshi",
            buy_market_id="poly-1",
            sell_market_id="KXRAIN",
            buy_price=0.40,
            sell_price=0.50,
            buy_fill_price=0.40,
            sell_fill_price=0.49,
            stake_usd=100.0,
            net_pnl=5.0,
            status="both_filled",
        ))

        rows = db.get_arb_trades(limit=10)
        assert len(rows) == 1
        trade = rows[0]
        assert trade["buy_platform"] == "polymarket"
        assert trade["sell_platform"] == "kalshi"
        assert trade["buy_fill_price"] == 0.40
        assert trade["sell_fill_price"] == 0.49

    def test_combined_pnl_aggregation(self, db: Database) -> None:
        """Total PnL aggregated across all trades."""
        for pnl in [10.0, -3.0, 5.5]:
            db.insert_arb_trade(ArbTradeRecord(
                arb_id=f"t-pnl-{pnl}",
                status="both_filled",
                net_pnl=pnl,
            ))

        summary = db.get_arb_summary()
        assert abs(summary["combined_pnl"] - 12.5) < 0.01

    def test_unwind_logged(self, db: Database) -> None:
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="unwind-1",
            status="unwound",
            unwind_reason="Sell leg failed: API timeout",
            net_pnl=-4.0,
        ))

        rows = db.get_arb_trades(limit=10)
        assert len(rows) == 1
        assert rows[0]["status"] == "unwound"
        assert "API timeout" in rows[0]["unwind_reason"]

    def test_failed_trade_logged(self, db: Database) -> None:
        db.insert_arb_trade(ArbTradeRecord(
            arb_id="fail-1",
            status="failed",
            unwind_reason="Invalid buy price",
            net_pnl=0.0,
        ))

        summary = db.get_arb_summary()
        assert summary["total_trades"] == 1
        assert summary["filled_trades"] == 0
