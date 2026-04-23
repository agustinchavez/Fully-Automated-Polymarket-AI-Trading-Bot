"""Tests for Batch C — AI Analyst (Analysis Layer Phase 3).

Covers:
  1. Provider routing (4 providers + aliases)
  2. Data gate (insufficient trades/days)
  3. Context assembly
  4. Prompt building
  5. Response parsing (clean JSON, fenced, embedded, failure)
  6. Provider call mocking (4 providers)
  7. Rate limiting
  8. Cached result get/set
  9. Telegram /analyze and /provider commands
  10. Dashboard API routes (GET cached, POST trigger, POST 429)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixture: in-memory DB with schema ────────────────────────────────


def _create_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with tables needed by the AI analyst."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE daily_summaries (
            summary_date TEXT PRIMARY KEY,
            total_pnl REAL DEFAULT 0,
            realized_pnl REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            drawdown_pct REAL DEFAULT 0,
            bankroll REAL DEFAULT 5000,
            trades_opened INTEGER DEFAULT 0,
            trades_closed INTEGER DEFAULT 0,
            positions_held INTEGER DEFAULT 0,
            best_trade_pnl REAL DEFAULT 0,
            worst_trade_pnl REAL DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE performance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL, question TEXT,
            category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
            actual_outcome REAL, edge_at_entry REAL,
            confidence TEXT DEFAULT 'LOW', evidence_quality REAL DEFAULT 0,
            stake_usd REAL DEFAULT 0, entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0, pnl REAL DEFAULT 0,
            holding_hours REAL DEFAULT 0, resolved_at TEXT
        );
        CREATE TABLE forecasts (
            id TEXT PRIMARY KEY, market_id TEXT NOT NULL, question TEXT,
            market_type TEXT, implied_probability REAL,
            model_probability REAL, edge REAL,
            confidence_level TEXT, evidence_quality REAL,
            num_sources INTEGER, decision TEXT, reasoning TEXT,
            evidence_json TEXT, invalidation_triggers_json TEXT,
            created_at TEXT
        );
        CREATE TABLE model_forecast_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL, market_id TEXT NOT NULL,
            category TEXT DEFAULT 'UNKNOWN', forecast_prob REAL,
            actual_outcome REAL, recorded_at TEXT
        );
        CREATE TABLE engine_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at REAL
        );
    """)
    return conn


def _populate_test_db(conn: sqlite3.Connection, days: int = 30, trades: int = 60) -> None:
    """Insert test data spanning ``days`` days with ``trades`` resolved trades."""
    today = dt.datetime.now(dt.timezone.utc)
    for i in range(days):
        d = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        pnl = 5.0 if i % 3 != 0 else -3.0
        conn.execute(
            "INSERT INTO daily_summaries "
            "(summary_date, total_pnl, realized_pnl, unrealized_pnl, "
            "drawdown_pct, bankroll) VALUES (?,?,?,?,?,?)",
            (d, pnl, pnl * 0.8, pnl * 0.2, 0.02 + i * 0.003, 5000.0),
        )

    for i in range(trades):
        d = (today - dt.timedelta(days=i % days)).strftime("%Y-%m-%d")
        mid = f"market_{i}"
        pnl = 4.0 if i % 3 != 0 else -2.5
        cat = ["CRYPTO", "MACRO", "POLITICS"][i % 3]
        conn.execute(
            "INSERT INTO performance_log "
            "(market_id, question, category, pnl, stake_usd, "
            "edge_at_entry, resolved_at) VALUES (?,?,?,?,?,?,?)",
            (mid, f"Will event {i} happen?", cat, pnl, 50.0, 0.08, d),
        )
        conn.execute(
            "INSERT INTO forecasts "
            "(id, market_id, question, market_type, edge, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (f"fc_{i}", mid, f"Will event {i} happen?", cat, 0.07 + i * 0.001, d),
        )

    for model in ["gpt-4o", "claude-sonnet", "gemini-pro"]:
        for i in range(20):
            d = (today - dt.timedelta(days=i % days)).strftime("%Y-%m-%d")
            prob = 0.6 + (i % 5) * 0.05
            outcome = 1.0 if i % 3 != 0 else 0.0
            conn.execute(
                "INSERT INTO model_forecast_log "
                "(model_name, market_id, forecast_prob, actual_outcome, recorded_at) "
                "VALUES (?,?,?,?,?)",
                (model, f"market_{i}", prob, outcome, d),
            )

    conn.commit()


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides):
    """Create a minimal AnalystConfig for tests."""
    from src.config import AnalystConfig
    defaults = {
        "enabled": True,
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "temperature": 0.3,
        "timeout_secs": 45,
        "min_resolved_trades": 50,
        "min_data_days": 28,
        "rate_limit_hours": 6,
        "schedule_enabled": False,
    }
    defaults.update(overrides)
    return AnalystConfig(**defaults)


SAMPLE_JSON_RESPONSE = json.dumps({
    "summary": "Bot is profitable with strong crypto performance.",
    "what_is_working": ["Crypto category trades", "Low Brier scores"],
    "what_is_not_working": ["High friction gap", "Macro category losses"],
    "confidence": "medium",
    "recommendations": [
        {
            "priority": 1,
            "action": "Increase crypto allocation",
            "rationale": "Highest win rate category",
            "config_change": "risk.max_position_pct: 0.15",
            "expected_impact": "+5% P&L",
        },
        {
            "priority": 2,
            "action": "Reduce macro exposure",
            "rationale": "Consistently negative P&L",
            "config_change": None,
            "expected_impact": "-2% drawdown",
        },
    ],
})


# ═══════════════════════════════════════════════════════════════
# 1. Provider Routing
# ═══════════════════════════════════════════════════════════════


class TestProviderRouting:
    def test_anthropic(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("anthropic") == "anthropic"

    def test_claude_alias(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("claude") == "anthropic"

    def test_openai(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("openai") == "openai"

    def test_gpt_alias(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("gpt") == "openai"

    def test_google(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("google") == "google"

    def test_gemini_alias(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("gemini") == "google"

    def test_deepseek(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        assert _route_provider("deepseek") == "deepseek"

    def test_unknown_raises(self) -> None:
        from src.analytics.ai_analyst import _route_provider
        with pytest.raises(ValueError, match="Unknown analyst provider"):
            _route_provider("unknown_provider")


# ═══════════════════════════════════════════════════════════════
# 2. Data Gate
# ═══════════════════════════════════════════════════════════════


class TestDataGate:
    def test_insufficient_trades(self) -> None:
        """Returns data_sufficient=False when too few trades."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        # Only populate 10 trades (below min_resolved_trades=50)
        _populate_test_db(conn, days=30, trades=10)
        cfg = _make_config(min_resolved_trades=50)
        analyst = AIAnalyst(conn=conn, config=cfg)
        result = asyncio.run(analyst.analyse(days=30))
        assert not result.data_sufficient
        assert "50" in result.summary  # mentions required trades

    def test_insufficient_days(self) -> None:
        """Returns data_sufficient=False when too few days."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        _populate_test_db(conn, days=5, trades=60)
        cfg = _make_config(min_data_days=28)
        analyst = AIAnalyst(conn=conn, config=cfg)
        result = asyncio.run(analyst.analyse(days=30))
        assert not result.data_sufficient


# ═══════════════════════════════════════════════════════════════
# 3. Context Assembly
# ═══════════════════════════════════════════════════════════════


class TestContextAssembly:
    def test_context_fields_populated(self) -> None:
        """Context has all fields when data is sufficient."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        ctx = analyst.assemble_context(days=30)
        assert ctx.data_quality["data_sufficient"]
        assert ctx.overall_summary  # non-empty
        assert ctx.category_stats  # non-empty
        assert ctx.model_stats  # non-empty

    def test_null_sentinel_excluded_from_model_stats(self) -> None:
        """Rows with actual_outcome=NULL are excluded from model accuracy queries."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        # Insert unresolved rows (NULL actual_outcome, like pipeline does now)
        today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        for i in range(50):
            conn.execute(
                "INSERT INTO model_forecast_log "
                "(model_name, market_id, forecast_prob, actual_outcome, recorded_at) "
                "VALUES (?,?,?,?,?)",
                ("gpt-4o", f"unresolved_{i}", 0.99, None, today),
            )
        conn.commit()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        ctx = analyst.assemble_context(days=30)
        # The 50 NULL rows should NOT inflate gpt-4o's Brier score
        # With the old -1.0 sentinel, (0.99 - (-1.0))^2 = 3.96 would corrupt the score
        assert "gpt-4o" in ctx.model_stats
        # Brier for resolved rows should be < 0.5 (reasonable)
        for line in ctx.model_stats.split("\n"):
            if "gpt-4o" in line and "Brier" in line:
                brier_str = line.split("Brier")[1].split(",")[0].strip()
                brier_val = float(brier_str)
                assert brier_val < 0.5, f"Brier {brier_val} too high — NULL rows may be leaking"

    def test_config_snapshot_with_bot_config(self) -> None:
        """Config snapshot contains trading config when bot_config is passed."""
        from src.analytics.ai_analyst import AIAnalyst
        from src.config import BotConfig
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config()
        bot_cfg = BotConfig()
        analyst = AIAnalyst(conn=conn, config=cfg, bot_config=bot_cfg)
        ctx = analyst.assemble_context(days=30)
        assert "min_edge=" in ctx.config_snapshot
        assert "kelly_fraction=" in ctx.config_snapshot
        assert "min_confidence=" in ctx.config_snapshot
        assert "ensemble_models=" in ctx.config_snapshot

    def test_config_snapshot_without_bot_config(self) -> None:
        """Config snapshot is empty when bot_config is not passed."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        ctx = analyst.assemble_context(days=30)
        assert ctx.config_snapshot == ""

    def test_context_empty_db(self) -> None:
        """Context marks data_sufficient=False for empty DB."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        ctx = analyst.assemble_context(days=30)
        assert not ctx.data_quality["data_sufficient"]
        assert ctx.overall_summary == ""


# ═══════════════════════════════════════════════════════════════
# 4. Prompt Building
# ═══════════════════════════════════════════════════════════════


class TestPromptBuilding:
    def test_prompt_includes_data(self) -> None:
        """Prompt includes context data."""
        from src.analytics.ai_analyst import AIAnalyst, AnalystContext
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        ctx = AnalystContext(
            period_start="2025-01-01",
            period_end="2025-01-31",
            overall_summary="P&L: $100, Win rate: 65%",
            category_stats="CRYPTO: 10 trades",
            model_stats="gpt-4o: Brier 0.200",
        )
        prompt = analyst._build_prompt(ctx)
        assert "P&L: $100" in prompt
        assert "CRYPTO: 10 trades" in prompt
        assert "gpt-4o: Brier 0.200" in prompt
        assert "JSON SCHEMA" in prompt


# ═══════════════════════════════════════════════════════════════
# 5. Response Parsing
# ═══════════════════════════════════════════════════════════════


class TestResponseParsing:
    def test_clean_json(self) -> None:
        """Parses clean JSON response correctly."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        result = analyst._parse_response(SAMPLE_JSON_RESPONSE)
        assert result.summary == "Bot is profitable with strong crypto performance."
        assert len(result.what_is_working) == 2
        assert len(result.what_is_not_working) == 2
        assert len(result.recommendations) == 2
        assert result.recommendations[0].priority == 1
        assert result.confidence == "medium"
        assert not result.parse_error

    def test_fenced_json(self) -> None:
        """Parses markdown-fenced JSON."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        fenced = f"```json\n{SAMPLE_JSON_RESPONSE}\n```"
        result = analyst._parse_response(fenced)
        assert result.summary == "Bot is profitable with strong crypto performance."
        assert not result.parse_error

    def test_embedded_json(self) -> None:
        """Parses JSON embedded in surrounding text."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        embedded = f"Here is the analysis:\n{SAMPLE_JSON_RESPONSE}\nEnd of analysis."
        result = analyst._parse_response(embedded)
        assert result.summary == "Bot is profitable with strong crypto performance."
        assert not result.parse_error

    def test_parse_failure(self) -> None:
        """Returns parse_error=True for non-JSON response."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        result = analyst._parse_response("I cannot provide JSON. Here is plain text.")
        assert result.parse_error
        assert "failed" in result.summary.lower() or "unparseable" in result.summary.lower()


# ═══════════════════════════════════════════════════════════════
# 6. Provider Calls (mocked)
# ═══════════════════════════════════════════════════════════════


class TestProviderCalls:
    def test_anthropic_call(self) -> None:
        """Anthropic provider calls AsyncAnthropic correctly."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(provider="anthropic")
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(return_value=SAMPLE_JSON_RESPONSE)
        with patch.dict(_PROVIDER_DISPATCH, {"anthropic": mock_call}):
            result = asyncio.run(analyst.analyse(days=30))
            mock_call.assert_awaited_once()
            assert result.provider_used == "anthropic"
            assert result.data_sufficient

    def test_openai_call(self) -> None:
        """OpenAI provider calls AsyncOpenAI correctly."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(provider="openai", model="gpt-4o")
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(return_value=SAMPLE_JSON_RESPONSE)
        with patch.dict(_PROVIDER_DISPATCH, {"openai": mock_call}):
            result = asyncio.run(analyst.analyse(days=30))
            mock_call.assert_awaited_once()
            assert result.provider_used == "openai"

    def test_google_call(self) -> None:
        """Google provider calls GenerativeModel correctly."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(provider="google", model="gemini-2.0-flash-latest")
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(return_value=SAMPLE_JSON_RESPONSE)
        with patch.dict(_PROVIDER_DISPATCH, {"google": mock_call}):
            result = asyncio.run(analyst.analyse(days=30))
            mock_call.assert_awaited_once()
            assert result.provider_used == "google"

    def test_deepseek_call(self) -> None:
        """DeepSeek provider calls AsyncOpenAI with custom base_url."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(provider="deepseek", model="deepseek-chat")
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(return_value=SAMPLE_JSON_RESPONSE)
        with patch.dict(_PROVIDER_DISPATCH, {"deepseek": mock_call}):
            result = asyncio.run(analyst.analyse(days=30))
            mock_call.assert_awaited_once()
            assert result.provider_used == "deepseek"


# ═══════════════════════════════════════════════════════════════
# 7. Rate Limiting
# ═══════════════════════════════════════════════════════════════


class TestRateLimiting:
    def test_rate_limit_allows_first_call(self) -> None:
        """First call is allowed (no prior timestamp)."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config(rate_limit_hours=6)
        analyst = AIAnalyst(conn=conn, config=cfg)
        assert analyst._check_rate_limit() is True

    def test_rate_limit_blocks_recent_call(self) -> None:
        """Blocks when last call was within rate_limit_hours."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config(rate_limit_hours=6)
        analyst = AIAnalyst(conn=conn, config=cfg)
        # Record a call 1 hour ago
        conn.execute(
            "INSERT INTO engine_state (key, value, updated_at) "
            "VALUES ('ai_analysis_last_call', ?, ?)",
            (str(time.time() - 3600), time.time()),
        )
        conn.commit()
        assert analyst._check_rate_limit() is False

    def test_rate_limit_allows_old_call(self) -> None:
        """Allows when last call was > rate_limit_hours ago."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config(rate_limit_hours=6)
        analyst = AIAnalyst(conn=conn, config=cfg)
        # Record a call 7 hours ago
        conn.execute(
            "INSERT INTO engine_state (key, value, updated_at) "
            "VALUES ('ai_analysis_last_call', ?, ?)",
            (str(time.time() - 7 * 3600), time.time()),
        )
        conn.commit()
        assert analyst._check_rate_limit() is True

    def test_analyse_returns_rate_limited(self) -> None:
        """analyse() returns rate-limited message when blocked."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(rate_limit_hours=6)
        analyst = AIAnalyst(conn=conn, config=cfg)
        # Record a very recent call
        conn.execute(
            "INSERT INTO engine_state (key, value, updated_at) "
            "VALUES ('ai_analysis_last_call', ?, ?)",
            (str(time.time()), time.time()),
        )
        conn.commit()
        result = asyncio.run(analyst.analyse(days=30))
        assert "rate limited" in result.summary.lower() or "Rate limited" in result.summary


# ═══════════════════════════════════════════════════════════════
# 8. Cached Result
# ═══════════════════════════════════════════════════════════════


class TestCachedResult:
    def test_cache_and_get(self) -> None:
        """Cache result and retrieve it."""
        from src.analytics.ai_analyst import AIAnalyst, AnalysisResult, Recommendation
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)

        result = AnalysisResult(
            summary="Test summary",
            what_is_working=["A", "B"],
            what_is_not_working=["C"],
            recommendations=[Recommendation(priority=1, action="Do X")],
            confidence="high",
            data_sufficient=True,
            provider_used="anthropic",
            model_used="claude-sonnet",
            generated_at="2025-01-01T00:00:00",
        )
        analyst._cache_result(result)
        cached = analyst.get_cached_result()
        assert cached is not None
        assert cached.summary == "Test summary"
        assert cached.provider_used == "anthropic"
        assert len(cached.what_is_working) == 2

    def test_get_cached_empty(self) -> None:
        """Returns None when no cached result exists."""
        from src.analytics.ai_analyst import AIAnalyst
        conn = _create_test_db()
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)
        assert analyst.get_cached_result() is None


# ═══════════════════════════════════════════════════════════════
# 9. Telegram Commands
# ═══════════════════════════════════════════════════════════════


class TestTelegramCommands:
    def test_analyze_disabled(self) -> None:
        """Returns disabled message when analyst is not enabled."""
        from src.observability.telegram_bot import TelegramKillBot
        engine = MagicMock()
        engine.config.analyst.enabled = False
        bot = TelegramKillBot(token="t", chat_id="1", engine=engine)
        result = asyncio.run(bot._commands.cmd_analyze(30))
        assert "disabled" in result.lower()

    def test_analyze_no_engine(self) -> None:
        """Returns error when no engine connected."""
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="t", chat_id="1", engine=None)
        result = asyncio.run(bot._commands.cmd_analyze(30))
        assert "No engine" in result

    def test_analyze_success(self) -> None:
        """Returns formatted analysis on success."""
        from src.observability.telegram_bot import TelegramKillBot
        from src.analytics.ai_analyst import AnalysisResult, Recommendation

        engine = MagicMock()
        engine.config.analyst.enabled = True
        engine._db.conn = _create_test_db()

        mock_result = AnalysisResult(
            summary="Performance is strong.",
            what_is_working=["Crypto trades"],
            what_is_not_working=["High fees"],
            recommendations=[Recommendation(priority=1, action="Lower fees")],
            data_sufficient=True,
            provider_used="anthropic",
            model_used="claude-sonnet",
        )

        with patch("src.analytics.ai_analyst.AIAnalyst.analyse", new_callable=AsyncMock) as mock_analyse:
            mock_analyse.return_value = mock_result
            bot = TelegramKillBot(token="t", chat_id="1", engine=engine)
            result = asyncio.run(bot._commands.cmd_analyze(30))
            assert "AI Analysis" in result
            assert "anthropic" in result
            assert "Performance is strong" in result
            assert "Crypto trades" in result
            assert "Lower fees" in result

    def test_provider_command(self) -> None:
        """Provider command returns config details."""
        from src.observability.telegram_bot import TelegramKillBot
        engine = MagicMock()
        engine.config.analyst.enabled = True
        engine.config.analyst.provider = "anthropic"
        engine.config.analyst.model = "claude-sonnet-4-6"
        engine.config.analyst.rate_limit_hours = 6
        bot = TelegramKillBot(token="t", chat_id="1", engine=engine)
        result = asyncio.run(bot._commands.cmd_provider())
        assert "anthropic" in result
        assert "claude-sonnet" in result
        assert "6h" in result

    def test_help_lists_analyze(self) -> None:
        """Help command includes /analyze and /provider."""
        from src.observability.telegram_bot import TelegramKillBot
        bot = TelegramKillBot(token="t", chat_id="1")
        result = asyncio.run(bot._commands.cmd_help())
        assert "analyze" in result
        assert "provider" in result


# ═══════════════════════════════════════════════════════════════
# 10. Dashboard API Routes
# ═══════════════════════════════════════════════════════════════


class TestDashboardRoutes:
    @pytest.fixture
    def app_client(self):
        """Create a Flask test client with a populated in-memory DB."""
        from src.dashboard.app import app

        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)

        def mock_get_conn():
            return conn

        with patch("src.dashboard.app._get_conn", side_effect=mock_get_conn):
            app.config["TESTING"] = True
            with app.test_client() as client:
                yield client

    def test_get_cached_no_data(self, app_client) -> None:
        """GET returns no-data response when nothing cached."""
        resp = app_client.get("/api/insights/ai-analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data_sufficient"] is False

    def test_get_cached_with_data(self, app_client) -> None:
        """GET returns cached analysis when present in engine_state."""
        from src.dashboard.app import _get_conn
        conn = _get_conn()
        cached = json.dumps({
            "summary": "Cached analysis",
            "what_is_working": ["A"],
            "what_is_not_working": [],
            "recommendations": [],
            "confidence": "high",
            "data_sufficient": True,
            "provider_used": "anthropic",
            "model_used": "claude-sonnet",
            "generated_at": "2025-01-01T00:00:00",
        })
        conn.execute(
            "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
            "VALUES ('ai_analysis_result', ?, ?)",
            (cached, time.time()),
        )
        conn.commit()
        resp = app_client.get("/api/insights/ai-analysis")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["summary"] == "Cached analysis"

    def test_post_disabled(self, app_client) -> None:
        """POST returns 400 when analyst is disabled."""
        with patch("src.dashboard.app._get_config") as mock_cfg:
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.analyst.enabled = False
            resp = app_client.post("/api/insights/ai-analysis")
            assert resp.status_code == 400

    def test_post_rate_limited(self, app_client) -> None:
        """POST returns 429 when rate limited."""
        with patch("src.dashboard.app._get_config") as mock_cfg:
            cfg = _make_config(enabled=True)
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.analyst = cfg
            with patch("src.analytics.ai_analyst.AIAnalyst._check_rate_limit", return_value=False):
                resp = app_client.post("/api/insights/ai-analysis")
                assert resp.status_code == 429


# ═══════════════════════════════════════════════════════════════
# 11. Timeout Handling
# ═══════════════════════════════════════════════════════════════


class TestTimeoutHandling:
    def test_timeout_returns_error_result(self) -> None:
        """analyse() returns timeout message on asyncio.TimeoutError."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config(timeout_secs=1)
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(side_effect=asyncio.TimeoutError())
        with patch.dict(_PROVIDER_DISPATCH, {"anthropic": mock_call}):
            result = asyncio.run(analyst.analyse(days=30))
            assert "timed out" in result.summary.lower()
            assert result.provider_used == "anthropic"


# ═══════════════════════════════════════════════════════════════
# 12. Cost Tracking
# ═══════════════════════════════════════════════════════════════


class TestCostTracking:
    def test_cost_recorded_on_success(self) -> None:
        """cost_tracker.record_call is invoked after successful analysis."""
        from src.analytics.ai_analyst import AIAnalyst, _PROVIDER_DISPATCH
        conn = _create_test_db()
        _populate_test_db(conn, days=30, trades=60)
        cfg = _make_config()
        analyst = AIAnalyst(conn=conn, config=cfg)

        mock_call = AsyncMock(return_value=SAMPLE_JSON_RESPONSE)
        with patch.dict(_PROVIDER_DISPATCH, {"anthropic": mock_call}):
            with patch("src.observability.metrics.cost_tracker") as mock_ct:
                result = asyncio.run(analyst.analyse(days=30))
                mock_ct.record_call.assert_called_once()
                call_args = mock_ct.record_call.call_args
                assert "analyst-anthropic" in call_args[0][0]
