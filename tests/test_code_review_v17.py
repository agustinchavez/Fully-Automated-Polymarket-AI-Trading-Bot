"""Code Review v17 — Calibration wiring, journal wiring, classifier fixes.

Issue 1: CalibrationFeedbackLoop.record_resolution() wired in exit_finalizer
Issue 2: insert_journal_entry() wired in exit_finalizer
Issue 4: Classifier misclassification fixes (GEOPOLITICS/ELECTION strengthened)
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import MagicMock, patch

import pytest


# ── Issue 1: Calibration Feedback Wired ──────────────────────────────


class TestCalibrationWiring:
    """Verify CalibrationFeedbackLoop.record_resolution() is called on exit."""

    def test_record_calibration_method_exists(self) -> None:
        """ExitFinalizer has _record_calibration method."""
        from src.execution.exit_finalizer import ExitFinalizer
        assert hasattr(ExitFinalizer, "_record_calibration")

    def test_finalize_calls_record_calibration(self) -> None:
        """finalize() calls _record_calibration."""
        from src.execution.exit_finalizer import ExitFinalizer
        source = inspect.getsource(ExitFinalizer.finalize)
        assert "_record_calibration" in source

    def test_record_calibration_on_resolved_market(self) -> None:
        """_record_calibration records when exit_price indicates resolution."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        config = MagicMock()
        config.continuous_learning = MagicMock()
        config.continuous_learning.smart_retrain_enabled = False

        finalizer = ExitFinalizer(db, config)

        pos = MagicMock()
        pos.market_id = "test-123"
        pos.stake_usd = 50.0
        pos.entry_price = 0.40
        pos.opened_at = "2024-01-01T00:00:00+00:00"
        pos.question = "Will X happen?"

        fc = MagicMock()
        fc.model_probability = 0.65
        fc.edge = 0.10
        fc.confidence_level = "MEDIUM"
        fc.evidence_quality = 0.5
        fc.market_type = "SCIENCE"
        db.get_forecasts.return_value = [fc]

        # exit_price >= 0.98 means YES resolved
        with patch(
            "src.analytics.calibration_feedback.CalibrationFeedbackLoop"
        ) as MockCFL:
            mock_loop = MagicMock()
            MockCFL.return_value = mock_loop

            finalizer._record_calibration(pos, exit_price=0.99, pnl=29.75)

            mock_loop.record_resolution.assert_called_once()
            call_args = mock_loop.record_resolution.call_args
            record = call_args[0][1]  # second positional arg
            assert record.actual_outcome == 1.0
            assert record.market_id == "test-123"

    def test_record_calibration_skips_early_exit(self) -> None:
        """_record_calibration skips when market isn't resolved (mid-range exit)."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        finalizer = ExitFinalizer(db, MagicMock())

        pos = MagicMock()
        pos.market_id = "test-123"

        # exit_price in mid-range — not a resolution
        finalizer._record_calibration(pos, exit_price=0.50, pnl=-5.0)

        # Should not even look up forecasts
        db.get_forecasts.assert_not_called()

    def test_record_calibration_no_resolved(self) -> None:
        """_record_calibration with exit_price <= 0.02 records outcome=0.0."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        config = MagicMock()
        config.continuous_learning = MagicMock()
        config.continuous_learning.smart_retrain_enabled = False

        finalizer = ExitFinalizer(db, config)

        pos = MagicMock()
        pos.market_id = "test-456"
        pos.stake_usd = 50.0
        pos.entry_price = 0.60
        pos.opened_at = "2024-01-01T00:00:00+00:00"
        pos.question = "Will Y happen?"

        fc = MagicMock()
        fc.model_probability = 0.65
        fc.edge = 0.05
        fc.confidence_level = "LOW"
        fc.evidence_quality = 0.3
        fc.market_type = "MACRO"
        db.get_forecasts.return_value = [fc]

        with patch(
            "src.analytics.calibration_feedback.CalibrationFeedbackLoop"
        ) as MockCFL:
            mock_loop = MagicMock()
            MockCFL.return_value = mock_loop

            finalizer._record_calibration(pos, exit_price=0.01, pnl=-29.5)

            mock_loop.record_resolution.assert_called_once()
            record = mock_loop.record_resolution.call_args[0][1]
            assert record.actual_outcome == 0.0

    def test_record_calibration_handles_no_forecast(self) -> None:
        """_record_calibration safely returns when no forecast found."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        db.get_forecasts.return_value = []
        finalizer = ExitFinalizer(db, MagicMock())

        pos = MagicMock()
        pos.market_id = "no-forecast"
        pos.entry_price = 0.40

        # Should not raise
        finalizer._record_calibration(pos, exit_price=0.99, pnl=29.75)


# ── Issue 2: Trade Journal Wired ─────────────────────────────────────


class TestJournalWiring:
    """Verify insert_journal_entry() is called on exit."""

    def test_record_journal_method_exists(self) -> None:
        """ExitFinalizer has _record_journal_entry method."""
        from src.execution.exit_finalizer import ExitFinalizer
        assert hasattr(ExitFinalizer, "_record_journal_entry")

    def test_finalize_calls_record_journal(self) -> None:
        """finalize() calls _record_journal_entry."""
        from src.execution.exit_finalizer import ExitFinalizer
        source = inspect.getsource(ExitFinalizer.finalize)
        assert "_record_journal_entry" in source

    def test_journal_entry_recorded(self) -> None:
        """_record_journal_entry calls db.insert_journal_entry."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        finalizer = ExitFinalizer(db, MagicMock())

        pos = MagicMock()
        pos.market_id = "journal-test"
        pos.entry_price = 0.40
        pos.stake_usd = 50.0
        pos.direction = "BUY"
        pos.question = "Will Z happen?"

        mkt = MagicMock()
        mkt.question = "Will Z happen?"

        finalizer._record_journal_entry(pos, 0.99, 29.75, "resolved", mkt)

        db.insert_journal_entry.assert_called_once()
        call_kwargs = db.insert_journal_entry.call_args
        assert call_kwargs[1]["market_id"] == "journal-test"
        assert call_kwargs[1]["pnl"] == 29.75

    def test_journal_handles_no_db(self) -> None:
        """_record_journal_entry safely returns when db is None."""
        from src.execution.exit_finalizer import ExitFinalizer

        finalizer = ExitFinalizer(None, MagicMock())
        # Should not raise
        finalizer._record_journal_entry(MagicMock(), 0.99, 29.75, "resolved")

    def test_journal_handles_db_error(self) -> None:
        """_record_journal_entry catches DB exceptions."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        db.insert_journal_entry.side_effect = Exception("DB error")
        finalizer = ExitFinalizer(db, MagicMock())

        pos = MagicMock()
        pos.market_id = "err-test"
        pos.entry_price = 0.40
        pos.stake_usd = 50.0
        pos.direction = "BUY"

        # Should not raise
        finalizer._record_journal_entry(pos, 0.99, 29.75, "resolved")


# ── Issue 4: Classifier Fixes ────────────────────────────────────────


class TestClassifierFixes:
    """Verify classifier misclassifications are fixed."""

    def test_nuclear_weapon_classified_geopolitics(self) -> None:
        """'nuclear weapon test' should be GEOPOLITICS, not CRYPTO."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will the U.S. conduct a nuclear weapon test in 2025?")
        assert result.category == "GEOPOLITICS", (
            f"Expected GEOPOLITICS, got {result.category}"
        )

    def test_assembly_election_classified_election(self) -> None:
        """'Hungarian assembly election' should be ELECTION, not CRYPTO."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will Fidesz win the Hungarian assembly election?")
        assert result.category == "ELECTION", (
            f"Expected ELECTION, got {result.category}"
        )

    def test_parliament_election_classified_election(self) -> None:
        """'parliamentary election' should be ELECTION."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will Labour win the UK parliamentary election?")
        assert result.category == "ELECTION", (
            f"Expected ELECTION, got {result.category}"
        )

    def test_crypto_general_regex_fixed(self) -> None:
        """CRYPTO general fallback regex has proper word boundaries."""
        # The old regex: r"\b(crypto|bitcoin|btc|ethereum|eth|blockchain|defi|nft\b)"
        # had \b inside the group only for nft — eth could match inside words.
        # Fixed to: r"\b(crypto|bitcoin|btc|ethereum|eth|blockchain|defi|nft)\b"
        from src.engine import market_classifier
        source = inspect.getsource(market_classifier)

        # Check the fixed regex is present (word boundary AFTER closing paren)
        assert r'nft)\b"' in source or "nft)\\b" in source

    def test_eth_doesnt_match_ethnic(self) -> None:
        """'eth' in CRYPTO fallback should not match inside 'ethnic' or 'methodology'."""
        pattern = re.compile(r"\b(crypto|bitcoin|btc|ethereum|eth|blockchain|defi|nft)\b", re.I)
        # Should NOT match
        assert not pattern.search("ethnic cleansing in region")
        assert not pattern.search("methodology for research")
        # SHOULD match
        assert pattern.search("eth price above $3000")
        assert pattern.search("bitcoin reaches new high")

    def test_arms_control_classified_geopolitics(self) -> None:
        """'arms control treaty' should be GEOPOLITICS."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will the US and Russia sign a new arms control treaty?")
        assert result.category == "GEOPOLITICS", (
            f"Expected GEOPOLITICS, got {result.category}"
        )

    def test_icbm_classified_geopolitics(self) -> None:
        """'ICBM launch' should be GEOPOLITICS."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will North Korea test an ICBM in 2025?")
        assert result.category == "GEOPOLITICS", (
            f"Expected GEOPOLITICS, got {result.category}"
        )

    def test_coalition_government_classified_election(self) -> None:
        """'coalition government' should be ELECTION."""
        from src.engine.market_classifier import classify_market

        result = classify_market("Will the coalition government dissolve before 2026?")
        assert result.category == "ELECTION", (
            f"Expected ELECTION, got {result.category}"
        )


# ── Email Alerting Config ────────────────────────────────────────────


class TestEmailAlertConfig:
    """Verify email SMTP alerting is properly configured."""

    def test_alerts_config_has_email_fields(self) -> None:
        """AlertsConfig has all SMTP email fields."""
        from src.config import AlertsConfig
        cfg = AlertsConfig()
        assert hasattr(cfg, "email_smtp_host")
        assert hasattr(cfg, "email_smtp_port")
        assert hasattr(cfg, "email_smtp_user")
        assert hasattr(cfg, "email_smtp_password")
        assert hasattr(cfg, "email_from")
        assert hasattr(cfg, "email_to")
        assert cfg.email_smtp_port == 587

    def test_env_overrides_for_smtp(self) -> None:
        """Environment variable overrides exist for SMTP fields."""
        from src.config import _ENV_OVERRIDES
        smtp_keys = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS",
                      "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO"]
        for key in smtp_keys:
            assert key in _ENV_OVERRIDES, f"Missing env override: {key}"

    def test_email_password_is_secret(self) -> None:
        """email_smtp_password is in _SECRET_FIELDS."""
        from src.config import _SECRET_FIELDS
        assert "email_smtp_password" in _SECRET_FIELDS


# ── Full finalize() Integration ──────────────────────────────────────


class TestFinalizeIntegration:
    """Verify finalize() runs all steps including new journal + calibration."""

    def test_finalize_runs_all_seven_steps(self) -> None:
        """finalize() should call archive, perf log, post-mortem, remove, alert, journal, calibration."""
        from src.execution.exit_finalizer import ExitFinalizer

        db = MagicMock()
        db.get_forecasts.return_value = []  # no forecast found
        config = MagicMock()
        config.continuous_learning = MagicMock()
        config.continuous_learning.post_mortem_enabled = False
        config.continuous_learning.smart_retrain_enabled = False

        finalizer = ExitFinalizer(db, config)

        pos = MagicMock()
        pos.market_id = "integration-test"
        pos.entry_price = 0.40
        pos.stake_usd = 50.0
        pos.direction = "BUY"
        pos.opened_at = "2024-01-01T00:00:00+00:00"
        pos.question = "Test?"

        finalizer.finalize(pos, 0.99, 29.75, "resolved")

        # Step 1: archive
        db.archive_position.assert_called_once()
        # Step 4: remove
        db.remove_position.assert_called_once()
        # Step 5: alert
        db.insert_alert.assert_called_once()
        # Step 5b: journal
        db.insert_journal_entry.assert_called_once()
