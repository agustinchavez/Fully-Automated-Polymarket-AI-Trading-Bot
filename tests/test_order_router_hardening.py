"""Tests for Phase 10 Batch A: OrderRouter hardening + OrderRecord + open_orders table."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import ExecutionConfig
from src.execution.order_builder import OrderSpec
from src.execution.order_router import OrderResult, OrderRouter, _parse_clob_response
from src.storage.models import OrderRecord


# ── Helpers ──────────────────────────────────────────────────────


def _make_order(**kwargs) -> OrderSpec:
    defaults = dict(
        order_id="test-order-1234",
        market_id="market-abc",
        token_id="token-xyz",
        side="BUY",
        order_type="limit",
        price=0.50,
        size=100.0,
        stake_usd=50.0,
        ttl_secs=300,
        dry_run=False,
    )
    defaults.update(kwargs)
    return OrderSpec(**defaults)


def _make_config(**kwargs) -> ExecutionConfig:
    defaults = dict(dry_run=False, max_retries=3, retry_backoff_secs=0.01)
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


def _make_db() -> sqlite3.Connection:
    from src.storage.migrations import run_migrations
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _make_database(conn: sqlite3.Connection):
    from src.config import StorageConfig
    from src.storage.database import Database
    db = Database(StorageConfig(sqlite_path=":memory:"))
    db._conn = conn
    return db


# ── CLOB Response Parsing ────────────────────────────────────────


class TestCLOBResponseParsing:
    def test_parse_matched_response(self):
        """status='matched' should produce status='filled'."""
        resp = {"orderID": "clob-123", "status": "matched", "success": True,
                "takingAmount": "50", "makingAmount": "0"}
        order = _make_order(price=0.50, size=100.0)
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.clob_order_id == "clob-123"
        assert result.fill_price == 0.50
        assert result.fill_size == 100.0  # 50 / 0.50

    def test_parse_live_response(self):
        """status='live' should produce status='submitted' with zero fills."""
        resp = {"orderID": "clob-456", "status": "live", "success": True,
                "takingAmount": "0", "makingAmount": "100"}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "submitted"
        assert result.fill_price == 0.0
        assert result.fill_size == 0.0

    def test_parse_delayed_response(self):
        """status='delayed' should map to 'pending'."""
        resp = {"orderID": "clob-789", "status": "delayed", "success": True}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "pending"

    def test_parse_missing_fields(self):
        """Missing orderID and takingAmount should not crash."""
        resp = {"success": True, "status": "live"}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.clob_order_id == ""
        assert result.fill_size == 0.0

    def test_parse_empty_response(self):
        """Empty dict should produce pending status."""
        resp = {}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "pending"
        assert result.fill_price == 0.0
        assert result.fill_size == 0.0

    def test_clob_order_id_extracted(self):
        resp = {"orderID": "exchange-id-999", "status": "live", "success": True}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.clob_order_id == "exchange-id-999"

    def test_fallback_order_id_when_missing(self):
        """Without orderID, clob_order_id should be empty."""
        resp = {"status": "live", "success": True}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.clob_order_id == ""
        assert result.order_id == "test-order-1234"

    def test_taking_amount_partial(self):
        """takingAmount < full order value should set partial fill."""
        resp = {"orderID": "clob-p1", "status": "matched", "success": True,
                "takingAmount": "25"}  # 25 / 0.50 = 50 tokens
        order = _make_order(price=0.50, size=100.0)
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.fill_size == 50.0

    def test_taking_amount_zero(self):
        """takingAmount='0' should keep fill_size=0.0 for live orders."""
        resp = {"orderID": "clob-z", "status": "live", "success": True,
                "takingAmount": "0"}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.fill_size == 0.0

    def test_raw_response_preserved(self):
        resp = {"orderID": "raw-test", "status": "live", "success": True,
                "extra_field": "hello"}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.raw_response["extra_field"] == "hello"

    def test_non_dict_response(self):
        """Non-dict response (string) should be wrapped safely."""
        resp = "OK"
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "pending"
        assert "raw" in result.raw_response

    def test_success_false_returns_failed(self):
        """success=False in response should return status='failed'."""
        resp = {"orderID": "clob-fail", "status": "live", "success": False,
                "errorMsg": "Insufficient balance"}
        order = _make_order()
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "failed"
        assert "Insufficient balance" in result.error

    def test_matched_no_taking_amount_full_fill(self):
        """Matched without takingAmount should assume full fill."""
        resp = {"orderID": "clob-full", "status": "matched", "success": True}
        order = _make_order(price=0.60, size=80.0)
        result = _parse_clob_response(resp, order, "2024-01-01T00:00:00Z")
        assert result.status == "filled"
        assert result.fill_price == 0.60
        assert result.fill_size == 80.0


# ── OrderRouter Dry Run ──────────────────────────────────────────


class TestOrderRouterDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_unchanged(self):
        """Paper mode should return status='simulated' with fill at order price."""
        clob = MagicMock()
        router = OrderRouter(clob, _make_config(dry_run=True))
        order = _make_order(dry_run=True)
        result = await router.submit_order(order)
        assert result.status == "simulated"
        assert result.fill_price == 0.50
        assert result.fill_size == 100.0

    @pytest.mark.asyncio
    async def test_dry_run_env_override(self):
        """is_live_trading_enabled()=False keeps paper mode."""
        clob = MagicMock()
        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=False):
            result = await router.submit_order(order)
        assert result.status == "simulated"

    @pytest.mark.asyncio
    async def test_dry_run_config_flag(self):
        """config.dry_run=True forces paper mode regardless."""
        clob = MagicMock()
        router = OrderRouter(clob, _make_config(dry_run=True))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "simulated"

    @pytest.mark.asyncio
    async def test_order_spec_dry_run_flag(self):
        """order.dry_run=True forces paper mode."""
        clob = MagicMock()
        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=True)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "simulated"

    @pytest.mark.asyncio
    async def test_dry_run_no_clob_call(self):
        """CLOB client should never be called in paper mode."""
        clob = MagicMock()
        router = OrderRouter(clob, _make_config(dry_run=True))
        order = _make_order(dry_run=True)
        await router.submit_order(order)
        clob.get_signing_client.assert_not_called()


# ── OrderRouter Live ─────────────────────────────────────────────


class TestOrderRouterLive:
    @pytest.mark.asyncio
    async def test_live_returns_filled_on_match(self):
        """Mocked CLOB returns matched → result is filled."""
        signing = MagicMock()
        signing.create_and_post_order.return_value = {
            "orderID": "clob-live-1", "status": "matched", "success": True,
            "takingAmount": "50",
        }
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "filled"
        assert result.clob_order_id == "clob-live-1"

    @pytest.mark.asyncio
    async def test_live_returns_submitted_on_live(self):
        """Mocked CLOB returns live → result is submitted."""
        signing = MagicMock()
        signing.create_and_post_order.return_value = {
            "orderID": "clob-live-2", "status": "live", "success": True,
        }
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "submitted"
        assert result.fill_size == 0.0

    @pytest.mark.asyncio
    async def test_live_retry_on_transient_error(self):
        """Exception on attempt 1 → success on attempt 2."""
        signing = MagicMock()
        signing.create_and_post_order.side_effect = [
            RuntimeError("timeout"),
            {"orderID": "clob-retry", "status": "live", "success": True},
        ]
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "submitted"
        assert signing.create_and_post_order.call_count == 2

    @pytest.mark.asyncio
    async def test_live_exhausted_retries(self):
        """All retries fail → status='failed'."""
        signing = MagicMock()
        signing.create_and_post_order.side_effect = RuntimeError("boom")
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False, max_retries=2))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        assert result.status == "failed"
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_live_market_order_slippage(self):
        """Market order uses aggressive pricing."""
        signing = MagicMock()
        signing.create_and_post_order.return_value = {
            "orderID": "clob-mkt", "status": "matched", "success": True,
            "takingAmount": "50",
        }
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False, slippage_tolerance=0.02))
        order = _make_order(dry_run=False, order_type="market", price=0.50)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            await router.submit_order(order)
        call_args = signing.create_and_post_order.call_args
        assert call_args.kwargs["price"] == pytest.approx(0.51, abs=0.001)

    @pytest.mark.asyncio
    async def test_live_limit_order_exact_price(self):
        """Limit order uses exact price."""
        signing = MagicMock()
        signing.create_and_post_order.return_value = {
            "orderID": "clob-lmt", "status": "live", "success": True,
        }
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False, order_type="limit", price=0.55)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            await router.submit_order(order)
        call_args = signing.create_and_post_order.call_args
        assert call_args.kwargs["price"] == 0.55

    @pytest.mark.asyncio
    async def test_live_result_to_dict(self):
        """OrderResult.to_dict() includes clob_order_id."""
        signing = MagicMock()
        signing.create_and_post_order.return_value = {
            "orderID": "clob-dict", "status": "live", "success": True,
        }
        clob = MagicMock()
        clob.get_signing_client.return_value = signing

        router = OrderRouter(clob, _make_config(dry_run=False))
        order = _make_order(dry_run=False)
        with patch("src.execution.order_router.is_live_trading_enabled", return_value=True):
            result = await router.submit_order(order)
        d = result.to_dict()
        assert "clob_order_id" in d
        assert d["clob_order_id"] == "clob-dict"


# ── OrderRecord Model ────────────────────────────────────────────


class TestOrderRecordModel:
    def test_defaults(self):
        rec = OrderRecord(order_id="o1", market_id="m1")
        assert rec.status == "pending"
        assert rec.filled_size == 0.0
        assert rec.avg_fill_price == 0.0
        assert rec.dry_run is True
        assert rec.ttl_secs == 0
        assert rec.error == ""

    def test_serialization(self):
        rec = OrderRecord(order_id="o2", market_id="m2", status="filled",
                          filled_size=50.0, avg_fill_price=0.55)
        d = rec.model_dump()
        rec2 = OrderRecord(**d)
        assert rec2.order_id == "o2"
        assert rec2.filled_size == 50.0

    def test_status_values(self):
        for status in ("pending", "submitted", "partial", "filled",
                       "cancelled", "expired", "failed"):
            rec = OrderRecord(order_id="o", market_id="m", status=status)
            assert rec.status == status

    def test_timestamps_auto_set(self):
        rec = OrderRecord(order_id="o3", market_id="m3")
        assert rec.created_at != ""
        assert rec.updated_at != ""
        assert "T" in rec.created_at  # ISO format


# ── open_orders Table ────────────────────────────────────────────


class TestOpenOrdersTable:
    def test_migration_16_creates_table(self):
        conn = _make_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "open_orders" in table_names

    def test_schema_version_is_18(self):
        conn = _make_db()
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == 18

    def test_insert_order(self):
        conn = _make_db()
        db = _make_database(conn)
        order = OrderRecord(
            order_id="ins-1", clob_order_id="clob-ins-1",
            market_id="m1", token_id="t1", side="BUY",
            order_type="limit", price=0.50, size=100.0,
            stake_usd=50.0, status="submitted",
        )
        db.insert_order(order)
        result = db.get_order("ins-1")
        assert result is not None
        assert result.clob_order_id == "clob-ins-1"
        assert result.status == "submitted"

    def test_update_order_status(self):
        conn = _make_db()
        db = _make_database(conn)
        order = OrderRecord(order_id="upd-1", market_id="m1", status="submitted")
        db.insert_order(order)
        db.update_order_status("upd-1", "filled", filled_size=100.0, avg_fill_price=0.50)
        updated = db.get_order("upd-1")
        assert updated is not None
        assert updated.status == "filled"
        assert updated.filled_size == 100.0
        assert updated.avg_fill_price == 0.50

    def test_get_open_orders_filter(self):
        conn = _make_db()
        db = _make_database(conn)
        db.insert_order(OrderRecord(order_id="f1", market_id="m1", status="submitted"))
        db.insert_order(OrderRecord(order_id="f2", market_id="m2", status="filled"))
        db.insert_order(OrderRecord(order_id="f3", market_id="m3", status="submitted"))
        submitted = db.get_open_orders(status="submitted")
        assert len(submitted) == 2
        assert all(o.status == "submitted" for o in submitted)

    def test_get_open_orders_all(self):
        conn = _make_db()
        db = _make_database(conn)
        db.insert_order(OrderRecord(order_id="a1", market_id="m1", status="submitted"))
        db.insert_order(OrderRecord(order_id="a2", market_id="m2", status="filled"))
        all_orders = db.get_open_orders()
        assert len(all_orders) == 2

    def test_get_order_not_found(self):
        conn = _make_db()
        db = _make_database(conn)
        result = db.get_order("nonexistent")
        assert result is None

    def test_get_submitted_orders(self):
        conn = _make_db()
        db = _make_database(conn)
        db.insert_order(OrderRecord(order_id="s1", market_id="m1", status="submitted"))
        db.insert_order(OrderRecord(order_id="s2", market_id="m2", status="pending"))
        submitted = db.get_submitted_orders()
        assert len(submitted) == 1
        assert submitted[0].order_id == "s1"

    def test_get_stale_orders(self):
        conn = _make_db()
        db = _make_database(conn)
        # Insert an order with a very old timestamp
        old_ts = "2020-01-01T00:00:00+00:00"
        db.insert_order(OrderRecord(
            order_id="stale-1", market_id="m1", status="submitted",
            created_at=old_ts,
        ))
        db.insert_order(OrderRecord(
            order_id="fresh-1", market_id="m2", status="submitted",
        ))
        stale = db.get_stale_orders(max_age_secs=60)
        assert len(stale) == 1
        assert stale[0].order_id == "stale-1"

    def test_dry_run_bool_conversion(self):
        """dry_run stored as INTEGER should be read back as bool."""
        conn = _make_db()
        db = _make_database(conn)
        db.insert_order(OrderRecord(order_id="dr-1", market_id="m1", dry_run=False))
        result = db.get_order("dr-1")
        assert result is not None
        assert result.dry_run is False
