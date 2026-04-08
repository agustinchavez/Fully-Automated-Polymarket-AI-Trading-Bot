"""Proactive multi-channel alerts — verify AlertManager is wired into key events.

Events covered:
  1. Kill switch (daily P&L) → AlertManager.send()
  2. Drawdown halt → AlertManager.drawdown_alert()
  3. Trade exit → AlertManager.pnl_alert()
  4. Critical invariant violation → AlertManager.error_alert()
  5. Calibration retrain → alert_callback in SmartRetrainManager
  6. Adaptive model weight shift → alert_callback in AdaptiveModelWeighter
  7. AI analyst completion → alert_callback in AIAnalyst
  8. ExitFinalizer carries alert_manager
"""

from __future__ import annotations

import inspect
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── 1. Kill Switch Alert ──────────────────────────────────────────────


class TestKillSwitchAlert:
    """Verify kill switch sends AlertManager notification."""

    def test_daily_pnl_kill_sends_alert(self) -> None:
        """_check_daily_pnl_kill() source contains alert_manager.send."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._check_daily_pnl_kill)
        assert "self._alert_manager" in source
        assert "kill_daily_pnl" in source


# ── 2. Drawdown Alert ─────────────────────────────────────────────────


class TestDrawdownAlert:
    """Verify drawdown halt sends AlertManager.drawdown_alert."""

    def test_drawdown_halt_sends_alert(self) -> None:
        """_run_cycle contains drawdown_alert() call."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._run_cycle)
        assert "drawdown_alert" in source


# ── 3. Trade Exit Alert ───────────────────────────────────────────────


class TestTradeExitAlert:
    """Verify trade exit sends AlertManager.pnl_alert."""

    def test_finalize_exit_sends_pnl_alert(self) -> None:
        """_finalize_exit() contains pnl_alert() call."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._finalize_exit)
        assert "pnl_alert" in source

    def test_pnl_alert_has_market_id(self) -> None:
        """pnl_alert receives market_id and reason."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._finalize_exit)
        assert "pos.market_id" in source
        assert "exit_reason" in source


# ── 4. Invariant Violation Alert ──────────────────────────────────────


class TestInvariantAlert:
    """Verify critical invariant violations send AlertManager.error_alert."""

    def test_invariant_check_sends_error_alert(self) -> None:
        """_maybe_check_invariants() calls error_alert for critical violations."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine._maybe_check_invariants)
        assert "error_alert" in source


# ── 5. Calibration Retrain Alert ─────────────────────────────────────


class TestRetrainAlert:
    """Verify SmartRetrainManager sends alert on AB test completion."""

    def test_retrain_accepts_alert_callback(self) -> None:
        """retrain_with_ab_test() accepts alert_callback param."""
        from src.analytics.smart_retrain import SmartRetrainManager
        sig = inspect.signature(SmartRetrainManager.retrain_with_ab_test)
        assert "alert_callback" in sig.parameters

    def test_retrain_calls_alert_callback(self) -> None:
        """When alert_callback is provided, it gets called."""
        from src.analytics.smart_retrain import SmartRetrainManager, RetrainTrigger

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("""
                CREATE TABLE calibration_history (
                    forecast_prob REAL, actual_outcome REAL,
                    recorded_at TEXT, market_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE calibration_ab_results (
                    test_id TEXT PRIMARY KEY,
                    calibrated_brier REAL, uncalibrated_brier REAL,
                    calibrated_count INTEGER, uncalibrated_count INTEGER,
                    calibration_helps INTEGER, delta_brier REAL,
                    trigger_reason TEXT, started_at TEXT, completed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE engine_state (
                    key TEXT PRIMARY KEY, value TEXT, updated_at REAL
                )
            """)
            # Insert enough samples for AB test
            for i in range(30):
                conn.execute(
                    "INSERT INTO calibration_history VALUES (?, ?, ?, ?)",
                    (0.5 + (i % 5) * 0.1, float(i % 2), str(i), f"m-{i}"),
                )
            conn.commit()

            manager = SmartRetrainManager(conn, ab_min_samples=10)
            trigger = RetrainTrigger(should_retrain=True, reason="test")

            callback = MagicMock()
            manager.retrain_with_ab_test(trigger, alert_callback=callback)

            callback.assert_called_once()
            call_kwargs = callback.call_args[1]
            assert "level" in call_kwargs
            assert "title" in call_kwargs
            assert "message" in call_kwargs
            assert "Retrain" in call_kwargs["title"]
            conn.close()

    def test_retrain_no_callback_no_error(self) -> None:
        """retrain_with_ab_test works fine without callback."""
        from src.analytics.smart_retrain import SmartRetrainManager, RetrainTrigger

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("""
                CREATE TABLE calibration_history (
                    forecast_prob REAL, actual_outcome REAL,
                    recorded_at TEXT, market_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE calibration_ab_results (
                    test_id TEXT PRIMARY KEY,
                    calibrated_brier REAL, uncalibrated_brier REAL,
                    calibrated_count INTEGER, uncalibrated_count INTEGER,
                    calibration_helps INTEGER, delta_brier REAL,
                    trigger_reason TEXT, started_at TEXT, completed_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE engine_state (
                    key TEXT PRIMARY KEY, value TEXT, updated_at REAL
                )
            """)
            for i in range(30):
                conn.execute(
                    "INSERT INTO calibration_history VALUES (?, ?, ?, ?)",
                    (0.5, float(i % 2), str(i), f"m-{i}"),
                )
            conn.commit()

            manager = SmartRetrainManager(conn, ab_min_samples=10)
            trigger = RetrainTrigger(should_retrain=True, reason="test")
            # Should not raise
            result = manager.retrain_with_ab_test(trigger)
            assert result.test_id != ""
            conn.close()


# ── 6. Adaptive Weight Shift Alert ───────────────────────────────────


class TestAdaptiveWeightAlert:
    """Verify AdaptiveModelWeighter sends alert on significant weight shift."""

    def test_get_weights_accepts_alert_callback(self) -> None:
        """get_weights() accepts alert_callback param."""
        from src.analytics.adaptive_weights import AdaptiveModelWeighter
        sig = inspect.signature(AdaptiveModelWeighter.get_weights)
        assert "alert_callback" in sig.parameters

    def test_get_weights_no_data_no_alert(self) -> None:
        """When no learned data, alert_callback is not called."""
        from src.analytics.adaptive_weights import AdaptiveModelWeighter
        from src.config import EnsembleConfig

        cfg = EnsembleConfig(models=["gpt-4o", "claude-sonnet-4-6"])
        weighter = AdaptiveModelWeighter(cfg)

        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("no table")

        callback = MagicMock()
        result = weighter.get_weights(conn, "MACRO", alert_callback=callback)
        callback.assert_not_called()
        assert not result.data_available

    def test_alert_callback_source_check(self) -> None:
        """get_weights source contains alert_callback and blend > 0.3 threshold."""
        from src.analytics.adaptive_weights import AdaptiveModelWeighter
        source = inspect.getsource(AdaptiveModelWeighter.get_weights)
        assert "alert_callback" in source
        assert "blend > 0.3" in source


# ── 7. AI Analyst Alert ──────────────────────────────────────────────


class TestAIAnalystAlert:
    """Verify AIAnalyst sends alert on successful analysis."""

    def test_analyse_accepts_alert_callback(self) -> None:
        """analyse() accepts alert_callback param."""
        from src.analytics.ai_analyst import AIAnalyst
        sig = inspect.signature(AIAnalyst.analyse)
        assert "alert_callback" in sig.parameters

    def test_analyse_source_has_alert(self) -> None:
        """analyse() source contains alert_callback call."""
        from src.analytics.ai_analyst import AIAnalyst
        source = inspect.getsource(AIAnalyst.analyse)
        assert "alert_callback" in source
        assert "AI Analyst Complete" in source


# ── 8. ExitFinalizer Alert Manager ───────────────────────────────────


class TestExitFinalizerAlertManager:
    """Verify ExitFinalizer carries alert_manager for retrain notifications."""

    def test_exit_finalizer_accepts_alert_manager(self) -> None:
        """ExitFinalizer.__init__ accepts alert_manager param."""
        from src.execution.exit_finalizer import ExitFinalizer
        sig = inspect.signature(ExitFinalizer.__init__)
        assert "alert_manager" in sig.parameters

    def test_exit_finalizer_stores_alert_manager(self) -> None:
        """ExitFinalizer stores alert_manager as attribute."""
        from src.execution.exit_finalizer import ExitFinalizer
        mock_am = MagicMock()
        ef = ExitFinalizer(MagicMock(), MagicMock(), alert_manager=mock_am)
        assert ef._alert_manager is mock_am

    def test_engine_wires_alert_manager_to_finalizer(self) -> None:
        """Engine loop sets alert_manager on exit_finalizer after init."""
        from src.engine.loop import TradingEngine
        source = inspect.getsource(TradingEngine.start)
        assert "exit_finalizer._alert_manager" in source

    def test_record_calibration_builds_alert_callback(self) -> None:
        """_record_calibration creates alert callback when alert_manager present."""
        from src.execution.exit_finalizer import ExitFinalizer
        source = inspect.getsource(ExitFinalizer._record_calibration)
        assert "_alert_cb" in source
        assert "alert_callback=_alert_cb" in source


# ── 9. CalibrationFeedbackLoop threads alert_callback ─────────────────


class TestCalibrationFeedbackAlertCallback:
    """Verify CalibrationFeedbackLoop threads alert_callback to smart retrain."""

    def test_record_resolution_accepts_alert_callback(self) -> None:
        """record_resolution() accepts alert_callback param."""
        from src.analytics.calibration_feedback import CalibrationFeedbackLoop
        sig = inspect.signature(CalibrationFeedbackLoop.record_resolution)
        assert "alert_callback" in sig.parameters

    def test_maybe_smart_retrain_accepts_alert_callback(self) -> None:
        """_maybe_smart_retrain() accepts alert_callback param."""
        from src.analytics.calibration_feedback import CalibrationFeedbackLoop
        sig = inspect.signature(CalibrationFeedbackLoop._maybe_smart_retrain)
        assert "alert_callback" in sig.parameters

    def test_record_resolution_threads_callback(self) -> None:
        """record_resolution passes alert_callback to _maybe_smart_retrain."""
        from src.analytics.calibration_feedback import CalibrationFeedbackLoop
        source = inspect.getsource(CalibrationFeedbackLoop.record_resolution)
        assert "alert_callback=alert_callback" in source


# ── 10. Dashboard AI Analyst Alert Callback ───────────────────────────


class TestDashboardAnalystAlert:
    """Verify dashboard POST endpoint passes alert callback to analyst."""

    def test_dashboard_analyst_has_alert_callback(self) -> None:
        """The POST endpoint creates an AlertManager-backed callback."""
        from src.dashboard import app
        source = inspect.getsource(app)
        assert "_analyst_alert" in source
        assert "alert_callback=_analyst_alert" in source
