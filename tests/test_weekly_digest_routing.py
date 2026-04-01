"""Tests for weekly digest routing via AlertManager."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.observability.reports import WeeklyDigestGenerator


# ── Helpers ──────────────────────────────────────────────────────


def _make_digest():
    """Create a minimal WeeklyDigest mock."""
    digest = MagicMock()
    digest.data_sufficient = True
    digest.period_start = "2025-01-01"
    digest.period_end = "2025-01-07"
    digest.total_pnl = 250.0
    digest.roi_pct = 2.5
    digest.realized_pnl = 200.0
    digest.unrealized_pnl = 50.0
    digest.sharpe_7d = "1.5"
    digest.max_drawdown_pct = 0.03
    digest.win_rate = 60.0
    digest.total_trades_resolved = 10
    digest.category_breakdown = []
    digest.model_accuracy = []
    digest.friction_analysis = MagicMock()
    digest.friction_analysis.avg_edge_at_entry = 0
    digest.best_trade = None
    digest.worst_trade = None
    digest.markets_evaluated = 0
    return digest


def _make_generator():
    """Create a WeeklyDigestGenerator with a mock connection."""
    gen = WeeklyDigestGenerator.__new__(WeeklyDigestGenerator)
    gen._conn = MagicMock()
    gen._bankroll = 10000.0
    gen._fee_pct = 0.02
    return gen


# ── format_plain ─────────────────────────────────────────────────


class TestFormatPlain:
    def test_strips_bold_markers(self):
        gen = _make_generator()
        digest = _make_digest()
        plain = gen.format_plain(digest)
        # No single-asterisk bold should remain
        # (but dollar signs and percentages are fine)
        assert "*Weekly Digest" not in plain
        assert "*P&L*" not in plain

    def test_content_preserved(self):
        gen = _make_generator()
        digest = _make_digest()
        plain = gen.format_plain(digest)
        assert "Weekly Digest" in plain
        assert "P&L" in plain
        assert "+$250.00" in plain

    def test_insufficient_data(self):
        gen = _make_generator()
        digest = _make_digest()
        digest.data_sufficient = False
        digest.data_days_available = 1
        plain = gen.format_plain(digest)
        assert "Not enough data" in plain
        assert "*" not in plain


# ── send_via_alert_manager ───────────────────────────────────────


class TestSendViaAlertManager:
    @pytest.mark.asyncio
    async def test_sends_through_alert_manager(self):
        gen = _make_generator()
        digest = _make_digest()
        alert_manager = MagicMock()
        alert_manager.send = AsyncMock()

        await gen.send_via_alert_manager(digest, alert_manager)

        alert_manager.send.assert_called_once()
        call_kwargs = alert_manager.send.call_args[1]
        assert call_kwargs["level"] == "info"
        assert "Weekly Digest" in call_kwargs["title"]
        assert call_kwargs["cooldown_key"] == "weekly_digest"
        assert call_kwargs["cooldown_secs"] == 3600

    @pytest.mark.asyncio
    async def test_message_is_plain_text(self):
        gen = _make_generator()
        digest = _make_digest()
        alert_manager = MagicMock()
        alert_manager.send = AsyncMock()

        await gen.send_via_alert_manager(digest, alert_manager)

        message = alert_manager.send.call_args[1]["message"]
        # No Telegram bold markers
        bold_pattern = re.compile(r"\*[^*]+\*")
        assert not bold_pattern.search(message)

    @pytest.mark.asyncio
    async def test_title_includes_dates(self):
        gen = _make_generator()
        digest = _make_digest()
        digest.period_start = "2025-03-01"
        digest.period_end = "2025-03-07"
        alert_manager = MagicMock()
        alert_manager.send = AsyncMock()

        await gen.send_via_alert_manager(digest, alert_manager)

        title = alert_manager.send.call_args[1]["title"]
        assert "2025-03-01" in title
        assert "2025-03-07" in title
