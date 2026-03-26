"""Tests for Phase 10C Batch A: OrderSpec canonical direction fields."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.config import ExecutionConfig
from src.execution.order_builder import (
    OrderSpec,
    build_exit_order,
    build_order,
)
from src.policy.position_sizer import PositionSize


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(**overrides) -> ExecutionConfig:
    defaults = dict(
        default_order_type="limit",
        slippage_tolerance=0.005,
        limit_order_ttl_secs=120,
        dry_run=True,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _make_position(direction="BUY_YES", **overrides) -> PositionSize:
    defaults = dict(
        stake_usd=50.0,
        kelly_fraction_used=0.05,
        full_kelly_stake=100.0,
        capped_by="kelly",
        direction=direction,
        token_quantity=100.0,
    )
    defaults.update(overrides)
    return PositionSize(**defaults)


# ── OrderSpec Fields ─────────────────────────────────────────────


class TestOrderSpecFields:
    def test_action_side_defaults_empty(self):
        spec = OrderSpec(
            order_id="test", market_id="mkt", token_id="tok",
            side="BUY", order_type="limit", price=0.5,
            size=100, stake_usd=50, ttl_secs=120, dry_run=True,
        )
        assert spec.action_side == ""
        assert spec.outcome_side == ""

    def test_canonical_fields_stored(self):
        spec = OrderSpec(
            order_id="test", market_id="mkt", token_id="tok",
            side="BUY", order_type="limit", price=0.5,
            size=100, stake_usd=50, ttl_secs=120, dry_run=True,
            action_side="BUY", outcome_side="YES",
        )
        assert spec.action_side == "BUY"
        assert spec.outcome_side == "YES"

    def test_to_dict_includes_canonical_fields(self):
        spec = OrderSpec(
            order_id="test", market_id="mkt", token_id="tok",
            side="BUY", order_type="limit", price=0.5,
            size=100, stake_usd=50, ttl_secs=120, dry_run=True,
            action_side="BUY", outcome_side="NO",
        )
        d = spec.to_dict()
        assert d["action_side"] == "BUY"
        assert d["outcome_side"] == "NO"


# ── build_order Simple ───────────────────────────────────────────


class TestBuildOrderSimple:
    def test_buy_yes_canonical_fields(self):
        config = _make_config()
        pos = _make_position(direction="BUY_YES")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="simple")

        assert len(orders) == 1
        assert orders[0].action_side == "BUY"
        assert orders[0].outcome_side == "YES"

    def test_buy_no_canonical_fields(self):
        config = _make_config()
        pos = _make_position(direction="BUY_NO")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="simple")

        assert len(orders) == 1
        assert orders[0].action_side == "BUY"
        assert orders[0].outcome_side == "NO"


# ── build_order TWAP ─────────────────────────────────────────────


class TestBuildOrderTWAP:
    def test_twap_all_children_have_canonical_fields(self):
        config = _make_config()
        pos = _make_position(direction="BUY_YES")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="twap")

        assert len(orders) == 5  # default 5 slices
        for order in orders:
            assert order.action_side == "BUY"
            assert order.outcome_side == "YES"

    def test_twap_buy_no_children(self):
        config = _make_config()
        pos = _make_position(direction="BUY_NO")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="twap")

        for order in orders:
            assert order.action_side == "BUY"
            assert order.outcome_side == "NO"


# ── build_order Iceberg ──────────────────────────────────────────


class TestBuildOrderIceberg:
    def test_iceberg_both_parts_have_canonical_fields(self):
        config = _make_config()
        pos = _make_position(direction="BUY_YES")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="iceberg")

        assert len(orders) == 2  # visible + hidden
        for order in orders:
            assert order.action_side == "BUY"
            assert order.outcome_side == "YES"

    def test_iceberg_buy_no(self):
        config = _make_config()
        pos = _make_position(direction="BUY_NO")
        orders = build_order("mkt-1", "tok-1", pos, 0.50, config, execution_strategy="iceberg")

        for order in orders:
            assert order.action_side == "BUY"
            assert order.outcome_side == "NO"


# ── build_exit_order ─────────────────────────────────────────────


class TestBuildExitOrder:
    def test_sell_with_outcome_side(self):
        config = _make_config()
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config, outcome_side="YES")
        assert order.action_side == "SELL"
        assert order.outcome_side == "YES"
        assert order.side == "SELL"

    def test_sell_with_no_outcome(self):
        config = _make_config()
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config, outcome_side="NO")
        assert order.action_side == "SELL"
        assert order.outcome_side == "NO"

    def test_sell_without_outcome_side_warns(self, caplog):
        config = _make_config()
        with caplog.at_level(logging.WARNING):
            order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config)
        assert order.action_side == "SELL"
        assert order.outcome_side == ""

    def test_exit_reason_still_in_metadata(self):
        config = _make_config()
        order = build_exit_order(
            "mkt-1", "tok-1", 50.0, 0.70, config,
            exit_reason="STOP_LOSS", outcome_side="YES",
        )
        assert order.metadata["exit_reason"] == "STOP_LOSS"

    def test_to_dict_includes_sell_canonical(self):
        config = _make_config()
        order = build_exit_order("mkt-1", "tok-1", 50.0, 0.70, config, outcome_side="YES")
        d = order.to_dict()
        assert d["action_side"] == "SELL"
        assert d["outcome_side"] == "YES"


# ── Engine Integration ───────────────────────────────────────────


class TestEngineRouteExitOrder:
    @pytest.mark.asyncio
    async def test_route_exit_passes_outcome_side(self):
        """_route_exit_order passes outcome_side from position to build_exit_order."""
        from src.engine.loop import TradingEngine

        engine = TradingEngine.__new__(TradingEngine)
        engine._db = MagicMock()
        engine._exit_finalizer = MagicMock()
        engine.config = MagicMock()
        engine.config.execution.dry_run = False
        engine.config.execution.live_exit_routing = True
        engine.config.execution.default_order_type = "limit"
        engine.config.execution.slippage_tolerance = 0.005
        engine.config.execution.limit_order_ttl_secs = 120

        pos = MagicMock()
        pos.market_id = "mkt-exit"
        pos.token_id = "tok-exit"
        pos.size = 100.0
        pos.stake_usd = 50.0
        pos.direction = "BUY_YES"
        pos.outcome_side = "YES"

        mock_result = MagicMock()
        mock_result.status = "filled"
        mock_result.fill_price = 0.60
        mock_result.fill_size = 100.0
        mock_result.order_id = "ord-exit-1"
        mock_result.clob_order_id = "clob-exit-1"

        with patch("src.execution.order_builder.build_exit_order") as mock_build, \
             patch("src.connectors.polymarket_clob.CLOBClient") as mock_clob_cls, \
             patch("src.execution.order_router.OrderRouter") as mock_router_cls, \
             patch("src.engine.loop.is_live_trading_enabled", return_value=True):
            mock_build.return_value = OrderSpec(
                order_id="ord-exit-1", market_id="mkt-exit", token_id="tok-exit",
                side="SELL", order_type="limit", price=0.595,
                size=100.0, stake_usd=50.0, ttl_secs=120, dry_run=False,
                action_side="SELL", outcome_side="YES",
            )
            mock_router_cls.return_value.submit_order.return_value = mock_result

            await engine._route_exit_order(
                pos, current_price=0.60, pnl=10.0,
                exit_reason="STOP_LOSS", mkt_record=None,
            )

            # Verify outcome_side was passed to build_exit_order
            mock_build.assert_called_once()
            call_kwargs = mock_build.call_args
            assert call_kwargs.kwargs.get("outcome_side") == "YES" or \
                   (len(call_kwargs.args) > 7 and call_kwargs.args[7] == "YES") or \
                   call_kwargs[1].get("outcome_side") == "YES"
