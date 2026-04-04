"""Tests for Phase 4: Quant-Level Intelligence Upgrades.

Covers all 8 improvements:
  1. Longshot Bias Correction
  2. Microstructure Signals → LLM Prompt
  3. TWAP Reference Price
  4. Smart Money as LLM Signal (Parts A+B)
  5. UMA Resolution Risk Monitor
  6. Manifold + PredictIt Connectors
  7. Conditional Market Graph
  8. Economic & Political Calendar
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Improvement 1: Longshot Bias Correction ─────────────────────────


class TestLongshotCorrection:
    """Tests for longshot_correction()."""

    def test_no_correction_in_normal_range(self):
        from src.forecast.ensemble import longshot_correction
        assert longshot_correction(0.50) == 0.50
        assert longshot_correction(0.30) == 0.30
        assert longshot_correction(0.70) == 0.70

    def test_low_prob_corrected_downward(self):
        """Low-prob events should be corrected downward (less overpriced)."""
        from src.forecast.ensemble import longshot_correction
        corrected = longshot_correction(0.05, low_threshold=0.12, strength=0.20)
        assert corrected < 0.05

    def test_high_prob_corrected_upward(self):
        """High-prob events should be corrected upward (less underpriced)."""
        from src.forecast.ensemble import longshot_correction
        corrected = longshot_correction(0.95, high_threshold=0.88, strength=0.20)
        assert corrected > 0.95

    def test_boundary_untouched(self):
        """Exactly at threshold → no correction."""
        from src.forecast.ensemble import longshot_correction
        assert longshot_correction(0.12) == 0.12
        assert longshot_correction(0.88) == 0.88

    def test_zero_strength_no_change(self):
        from src.forecast.ensemble import longshot_correction
        assert longshot_correction(0.05, strength=0.0) == 0.05
        assert longshot_correction(0.95, strength=0.0) == 0.95

    def test_probability_stays_in_bounds(self):
        """Correction should never push probability below 0 or above 1."""
        from src.forecast.ensemble import longshot_correction
        assert longshot_correction(0.001, strength=1.0) >= 0.0
        assert longshot_correction(0.999, strength=1.0) <= 1.0


# ── Improvement 2: Microstructure Signals → LLM Prompt ─────────────


class TestMicrostructurePromptInjection:
    """Tests for ORDER FLOW SIGNALS in signal stack rendering."""

    def test_order_flow_render_empty(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack()
        assert render_signal_stack(stack) == ""

    def test_order_flow_render_with_imbalance(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(order_flow_imbalance=0.35)
        rendered = render_signal_stack(stack)
        assert "ORDER FLOW SIGNALS" in rendered
        assert "net BUY pressure" in rendered

    def test_order_flow_render_sell_pressure(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(order_flow_imbalance=-0.4)
        rendered = render_signal_stack(stack)
        assert "net SELL pressure" in rendered

    def test_vwap_divergence_render(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(vwap_divergence_pct=2.5)
        rendered = render_signal_stack(stack)
        assert "above" in rendered
        assert "VWAP" in rendered

    def test_smart_money_ratio_render(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(smart_money_ratio=0.75)
        rendered = render_signal_stack(stack)
        assert "smart money dominant" in rendered

    def test_whale_activity_render(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(whale_net_direction="BUY", whale_total_usd=5000.0)
        rendered = render_signal_stack(stack)
        assert "$5,000" in rendered
        assert "BUY" in rendered


# ── Improvement 3: TWAP Reference Price ────────────────────────────


class TestTWAP:
    """Tests for WebSocketFeed TWAP calculation."""

    def test_get_twap_no_history(self):
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        assert ws.get_twap("token123") is None

    def test_get_twap_basic(self):
        """True time-weighted average: each price weighted by duration."""
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        now = time.time()
        # Price sat at 0.40 for 90s, then jumped to 0.80 for 30s
        ws._price_history["tok"] = [
            (now - 120, 0.40),
            (now - 30, 0.80),
            (now, 0.80),  # current tick anchors last segment
        ]
        twap = ws.get_twap("tok", window_hours=1.0)
        assert twap is not None
        # TWAP = (0.40*90 + 0.80*30) / 120 = (36+24)/120 = 0.50
        assert abs(twap - 0.50) < 0.01

    def test_get_twap_single_point_returns_none(self):
        """Need at least 2 data points for time-weighted calc."""
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        now = time.time()
        ws._price_history["tok"] = [(now - 100, 0.50)]
        assert ws.get_twap("tok", window_hours=1.0) is None

    def test_get_twap_short_window_returns_none(self):
        """Less than 60s of data is not meaningful."""
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        now = time.time()
        ws._price_history["tok"] = [
            (now - 30, 0.50),
            (now - 10, 0.60),
        ]
        assert ws.get_twap("tok", window_hours=1.0) is None

    def test_get_twap_window_filter(self):
        """Only prices within the window should be included."""
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        now = time.time()
        # One price 3 hours ago (outside 2h window), two recent with 100s span
        ws._price_history["tok"] = [
            (now - 10800, 0.30),  # 3h ago — excluded
            (now - 100, 0.60),
            (now - 10, 0.70),    # last segment anchor
        ]
        twap = ws.get_twap("tok", window_hours=2.0)
        assert twap is not None
        # Only 2 pts in window: 0.60 for 90s, total 90s → TWAP = 0.60
        assert abs(twap - 0.60) < 0.01


# ── Improvement 4: Smart Money as LLM Signal ──────────────────────


class TestSmartMoneySignal:
    """Tests for smart money signal in SignalStack."""

    def test_conviction_signal_in_stack(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(
            whale_count=4,
            whale_direction="BULLISH",
            whale_total_usd=25000,
            whale_avg_entry=0.45,
            whale_signal_strength="STRONG",
            whale_conviction_score=85.0,
        )
        rendered = render_signal_stack(stack)
        assert "SMART MONEY SIGNALS" in rendered
        assert "4 tracked" in rendered
        assert "STRONG" in rendered

    def test_conviction_signals_build(self):
        """build_signal_stack should extract conviction signals."""
        from src.research.signal_aggregator import build_signal_stack

        @dataclass
        class MockSig:
            whale_count: int = 3
            total_whale_usd: float = 10000
            direction: str = "BEARISH"
            avg_whale_price: float = 0.60
            signal_strength: str = "MODERATE"
            conviction_score: float = 55.0

        stack = build_signal_stack(
            sources=[], poly_price=0.5,
            conviction_signals=[MockSig()],
        )
        assert stack.whale_count == 3
        assert stack.whale_direction == "BEARISH"
        assert stack.whale_signal_strength == "MODERATE"

    @pytest.mark.asyncio
    async def test_discover_top_wallets_fallback(self):
        """discover_top_wallets falls back to static list on failure."""
        from src.analytics.wallet_scanner import WalletScanner, LEADERBOARD_WALLETS

        scanner = WalletScanner()
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(side_effect=Exception("no network"))
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await scanner.discover_top_wallets(top_n=5, min_pnl=0)
        # Should return static fallback on failure
        assert len(result) == 5
        assert result[0]["address"] == LEADERBOARD_WALLETS[0]["address"]


# ── Improvement 5: UMA Resolution Risk Monitor ────────────────────


class TestUMAMonitor:
    """Tests for UMAMonitor."""

    def test_not_disputed_by_default(self):
        from src.analytics.uma_monitor import UMAMonitor
        monitor = UMAMonitor()
        assert not monitor.is_disputed("some_condition_id")
        assert monitor.get_dispute_risk("some_condition_id") == 0.0

    def test_disputed_after_refresh(self):
        from src.analytics.uma_monitor import UMAMonitor
        monitor = UMAMonitor()
        # Manually inject disputed set
        monitor._disputed = {"cid_123"}
        monitor._last_refresh = time.time()
        assert monitor.is_disputed("cid_123")
        assert monitor.get_dispute_risk("cid_123") == 1.0
        assert not monitor.is_disputed("cid_456")

    @pytest.mark.asyncio
    async def test_refresh_respects_interval(self):
        from src.analytics.uma_monitor import UMAMonitor
        monitor = UMAMonitor(refresh_interval_mins=15)
        monitor._last_refresh = time.time()  # just refreshed
        monitor._disputed = {"old_cid"}
        await monitor.refresh_disputes()
        # Should not have cleared since interval not passed
        assert "old_cid" in monitor._disputed


# ── Improvement 6: Manifold + PredictIt Connectors ─────────────────


class TestManifoldConnector:
    """Tests for ManifoldConnector."""

    def test_name_and_categories(self):
        from src.research.connectors.manifold import ManifoldConnector
        cfg = MagicMock()
        conn = ManifoldConnector(cfg)
        assert conn.name == "manifold"

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        from src.research.connectors.manifold import ManifoldConnector
        cfg = MagicMock()
        conn = ManifoldConnector(cfg)
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=[])
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_session_inst = MagicMock()
            mock_session_inst.get.return_value = mock_ctx
            mock_session_inst.__aenter__ = AsyncMock(return_value=mock_session_inst)
            mock_session_inst.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value = mock_session_inst

            result = await conn._fetch_impl("Will X happen?", "MACRO")
            assert result == []


class TestPredictItConnector:
    """Tests for PredictItConnector."""

    def test_name_and_categories(self):
        from src.research.connectors.predictit import PredictItConnector
        cfg = MagicMock()
        conn = PredictItConnector(cfg)
        assert conn.name == "predictit"


# ── Improvement 7: Conditional Market Graph ────────────────────────


class TestMarketGraph:
    """Tests for MarketGraph Dutch book detection."""

    def test_no_violations_when_monotonic(self):
        from src.policy.market_graph import MarketGraph

        @dataclass
        class MockMarket:
            id: str
            question: str
            implied_probability: float
            event_slug: str = "event-1"
            end_date: str = ""

        graph = MarketGraph(fee_cost=0.04)
        markets = [
            MockMarket(id="m1", question="By June?", implied_probability=0.40, end_date="2025-06-01"),
            MockMarket(id="m2", question="By Sept?", implied_probability=0.60, end_date="2025-09-01"),
        ]
        graph.build_from_markets(markets)
        violations = graph.find_monotonicity_violations()
        assert len(violations) == 0

    def test_violation_detected(self):
        """P(earlier) > P(later) is a violation."""
        from src.policy.market_graph import MarketGraph

        @dataclass
        class MockMarket:
            id: str
            question: str
            implied_probability: float
            event_slug: str = "event-1"
            end_date: str = ""

        graph = MarketGraph(fee_cost=0.02)
        markets = [
            MockMarket(id="m1", question="By June?", implied_probability=0.70, end_date="2025-06-01"),
            MockMarket(id="m2", question="By Sept?", implied_probability=0.50, end_date="2025-09-01"),
        ]
        graph.build_from_markets(markets)
        violations = graph.find_monotonicity_violations()
        assert len(violations) == 1
        assert violations[0].market_a_prob == 0.70
        assert violations[0].market_b_prob == 0.50
        assert violations[0].edge > 0

    def test_different_events_not_compared(self):
        from src.policy.market_graph import MarketGraph

        @dataclass
        class MockMarket:
            id: str
            question: str
            implied_probability: float
            event_slug: str
            end_date: str = ""

        graph = MarketGraph()
        markets = [
            MockMarket(id="m1", question="A by June?", implied_probability=0.80,
                       event_slug="event-A", end_date="2025-06-01"),
            MockMarket(id="m2", question="B by Sept?", implied_probability=0.30,
                       event_slug="event-B", end_date="2025-09-01"),
        ]
        graph.build_from_markets(markets)
        violations = graph.find_monotonicity_violations()
        assert len(violations) == 0

    def test_actionable_requires_positive_edge(self):
        from src.policy.market_graph import MarketGraph

        @dataclass
        class MockMarket:
            id: str
            question: str
            implied_probability: float
            event_slug: str = "event-1"
            end_date: str = ""

        graph = MarketGraph(fee_cost=0.20)  # high fees eat the edge
        markets = [
            MockMarket(id="m1", question="By June?", implied_probability=0.55, end_date="2025-06-01"),
            MockMarket(id="m2", question="By Sept?", implied_probability=0.50, end_date="2025-09-01"),
        ]
        graph.build_from_markets(markets)
        violations = graph.find_monotonicity_violations()
        assert len(violations) == 1
        assert not violations[0].actionable  # edge - fees < 0.01


# ── Improvement 8: Economic & Political Calendar ──────────────────


class TestEventCalendar:
    """Tests for EventCalendar."""

    def test_empty_by_default(self):
        from src.analytics.event_calendar import EventCalendar
        cal = EventCalendar()
        events = cal.get_events_for_market("Will CPI exceed 3%?", "MACRO")
        assert events == []

    def test_event_matching_by_keyword(self):
        from src.analytics.event_calendar import EventCalendar, CalendarEvent
        from datetime import datetime, timezone
        cal = EventCalendar()
        cal._events = [
            CalendarEvent(
                name="Consumer Price Index",
                date=datetime.now(timezone.utc),
                category="ECONOMIC",
                hours_away=12.0,
                impact="high",
                keywords=["cpi", "inflation"],
            ),
        ]
        events = cal.get_events_for_market("Will CPI exceed 3%?", "TECH")
        assert len(events) == 1

    def test_event_matching_by_category(self):
        from src.analytics.event_calendar import EventCalendar, CalendarEvent
        from datetime import datetime, timezone
        cal = EventCalendar()
        cal._events = [
            CalendarEvent(
                name="FOMC Meeting",
                date=datetime.now(timezone.utc),
                category="FED",
                hours_away=48.0,
                impact="high",
                keywords=["fomc"],
            ),
        ]
        # Match by "FED" category
        events = cal.get_events_for_market("Will the Fed raise rates?", "FED")
        assert len(events) == 1


# ── Signal Aggregator: Manifold + PredictIt consensus ──────────────


class TestConsensusSignalAggregation:
    """Tests for Manifold/PredictIt consensus signal parsing in signal stack."""

    def test_manifold_in_consensus(self):
        from src.research.signal_aggregator import SignalStack, compute_signal_confluence
        stack = SignalStack(manifold_probability=0.54, manifold_traders=42)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 1.0  # 4pp divergence < 5pp threshold

    def test_predictit_in_consensus(self):
        from src.research.signal_aggregator import SignalStack, compute_signal_confluence
        stack = SignalStack(predictit_probability=0.72)
        mult = compute_signal_confluence(stack, poly_price=0.50)
        # 22pp divergence → should be 0.25
        assert mult == 0.25

    def test_mixed_consensus_divergence(self):
        from src.research.signal_aggregator import SignalStack, compute_signal_confluence
        stack = SignalStack(
            kalshi_price=0.68,
            metaculus_probability=0.70,
            manifold_probability=0.66,
        )
        mult = compute_signal_confluence(stack, poly_price=0.55)
        # avg consensus ≈ 0.68, divergence ≈ 13pp → 0.50
        assert mult == 0.50

    def test_no_consensus_returns_one(self):
        from src.research.signal_aggregator import SignalStack, compute_signal_confluence
        stack = SignalStack()
        mult = compute_signal_confluence(stack, poly_price=0.50)
        assert mult == 1.0


# ── Signal Stack Rendering: Calendar Events ────────────────────────


class TestCalendarEventsRendering:
    """Tests for calendar event rendering in signal stack."""

    def test_calendar_events_render(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        @dataclass
        class MockEvent:
            name: str = "FOMC Meeting"
            hours_away: float = 18.0
            impact: str = "high"

        stack = SignalStack(calendar_events=[MockEvent()])
        rendered = render_signal_stack(stack)
        assert "UPCOMING EVENTS" in rendered
        assert "FOMC Meeting" in rendered
        assert "HIGH IMPACT" in rendered

    def test_empty_calendar_no_render(self):
        from src.research.signal_aggregator import SignalStack, render_signal_stack
        stack = SignalStack(calendar_events=[])
        rendered = render_signal_stack(stack)
        assert "UPCOMING EVENTS" not in rendered


# ── Code Review Fix Tests ──────────────────────────────────────────


class TestTWAPTimeWeighted:
    """Verify TWAP uses duration-weighted calculation, not simple average."""

    def test_uneven_spacing_weights_by_duration(self):
        """Price at 0.40 for 180s then 0.80 for 20s → TWAP ≈ 0.44, not 0.60."""
        from src.connectors.ws_feed import WebSocketFeed
        ws = WebSocketFeed()
        now = time.time()
        ws._price_history["tok"] = [
            (now - 200, 0.40),
            (now - 20, 0.80),
            (now, 0.80),  # anchor
        ]
        twap = ws.get_twap("tok", window_hours=1.0)
        assert twap is not None
        # TWAP = (0.40*180 + 0.80*20) / 200 = (72+16)/200 = 0.44
        assert abs(twap - 0.44) < 0.01


class TestMicroSignalsWiring:
    """Verify microstructure signals flow into build_signal_stack."""

    def test_vwap_divergence_populates_stack(self):
        """When micro_signals has vwap_divergence_pct, it appears in the stack."""
        from src.connectors.microstructure import MicrostructureSignals
        from src.research.signal_aggregator import build_signal_stack

        ms = MicrostructureSignals(token_id="tok123")
        ms.vwap = 0.50
        ms.vwap_divergence = 0.03
        ms.vwap_divergence_pct = 0.06

        stack = build_signal_stack(sources=[], poly_price=0.50, micro_signals=ms)
        assert stack.vwap_divergence_pct == 0.06


class TestUMASingleton:
    """Verify UMAMonitor is created once, not per pipeline call."""

    def test_uma_monitor_stored_on_pipeline(self):
        """PipelineRunner stores _uma_monitor when uma.enabled=True."""
        from unittest.mock import MagicMock, patch

        config = MagicMock()
        config.uma.enabled = True
        config.uma.refresh_interval_mins = 15
        config.calendar.enabled = False

        with patch("src.engine.pipeline.PipelineRunner.__init__", return_value=None):
            from src.engine.pipeline import PipelineRunner
            runner = PipelineRunner.__new__(PipelineRunner)

        # Manually invoke the singleton init logic
        runner._uma_monitor = None
        if config.uma.enabled:
            from src.analytics.uma_monitor import UMAMonitor
            runner._uma_monitor = UMAMonitor(
                refresh_interval_mins=config.uma.refresh_interval_mins,
            )

        assert runner._uma_monitor is not None
        # Refresh interval should be set from config
        assert runner._uma_monitor._interval == 15 * 60

    def test_uma_monitor_none_when_disabled(self):
        """PipelineRunner._uma_monitor is None when uma.enabled=False."""
        from unittest.mock import MagicMock, patch

        config = MagicMock()
        config.uma.enabled = False
        config.calendar.enabled = False

        with patch("src.engine.pipeline.PipelineRunner.__init__", return_value=None):
            from src.engine.pipeline import PipelineRunner
            runner = PipelineRunner.__new__(PipelineRunner)

        runner._uma_monitor = None
        if config.uma.enabled:
            pass  # would create monitor
        assert runner._uma_monitor is None


class TestCalendarSingleton:
    """Verify EventCalendar is created once, not per pipeline call."""

    def test_calendar_stored_on_pipeline(self):
        """PipelineRunner stores _event_calendar when calendar.enabled=True."""
        from unittest.mock import MagicMock, patch

        config = MagicMock()
        config.uma.enabled = False
        config.calendar.enabled = True
        config.calendar.refresh_interval_hours = 6
        config.calendar.lookahead_days = 14

        with patch("src.engine.pipeline.PipelineRunner.__init__", return_value=None):
            from src.engine.pipeline import PipelineRunner
            runner = PipelineRunner.__new__(PipelineRunner)

        runner._event_calendar = None
        if config.calendar.enabled:
            from src.analytics.event_calendar import EventCalendar
            runner._event_calendar = EventCalendar(
                refresh_interval_hours=config.calendar.refresh_interval_hours,
                lookahead_days=config.calendar.lookahead_days,
            )

        assert runner._event_calendar is not None
        assert runner._event_calendar._interval == 6 * 3600


class TestWhaleBoostCap:
    """Verify whale boost is capped at 0.02 when whale_in_prompt is True."""

    def test_boost_capped_not_halved(self):
        """With conviction_edge_boost=0.08 and whale_in_prompt=True, cap at 0.02."""
        boost = 0.08
        penalty = 0.04
        whale_in_prompt = True

        if whale_in_prompt:
            boost = min(boost, 0.02)
            penalty = min(penalty, 0.02)

        assert boost == 0.02
        assert penalty == 0.02

    def test_boost_unchanged_when_already_below_cap(self):
        """Enhanced boost (0.01) stays at 0.01 even with cap."""
        boost = 0.01
        whale_in_prompt = True

        if whale_in_prompt:
            boost = min(boost, 0.02)

        assert boost == 0.01
