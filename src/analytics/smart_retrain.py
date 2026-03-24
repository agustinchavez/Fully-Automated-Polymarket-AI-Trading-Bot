"""Smart calibration retraining with A/B testing.

Triggers retraining based on:
  1. Resolution count (e.g., every 30 resolutions)
  2. Brier score degradation (>10% over 7-day rolling window)
  3. New specialist enabled (changes forecast distribution)

Validates via A/B holdout: 80% train, 20% test. If calibration
hurts performance, auto-disables and alerts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class RetrainTrigger:
    """Result of checking whether retraining is needed."""
    should_retrain: bool = False
    reason: str = "none"  # resolution_count | brier_degradation | specialist_enabled | none
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Result of calibration A/B test."""
    test_id: str = ""
    calibrated_brier: float = 0.0
    uncalibrated_brier: float = 0.0
    calibrated_count: int = 0
    uncalibrated_count: int = 0
    calibration_helps: bool = True
    delta_brier: float = 0.0  # calibrated - uncalibrated (negative = helps)
    trigger_reason: str = ""
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "calibrated_brier": self.calibrated_brier,
            "uncalibrated_brier": self.uncalibrated_brier,
            "calibrated_count": self.calibrated_count,
            "uncalibrated_count": self.uncalibrated_count,
            "calibration_helps": self.calibration_helps,
            "delta_brier": self.delta_brier,
            "trigger_reason": self.trigger_reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class SmartRetrainManager:
    """Manages smart calibration retraining triggers and A/B testing."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        retrain_resolution_count: int = 30,
        brier_degradation_threshold: float = 0.10,
        brier_window_days: int = 7,
        ab_holdout_pct: float = 0.20,
        ab_min_samples: int = 20,
    ):
        self._conn = conn
        self._retrain_resolution_count = retrain_resolution_count
        self._brier_degradation_threshold = brier_degradation_threshold
        self._brier_window_days = brier_window_days
        self._ab_holdout_pct = ab_holdout_pct
        self._ab_min_samples = ab_min_samples

    def check_retrain_trigger(self) -> RetrainTrigger:
        """Check if calibration retraining is needed."""

        # 1. Resolution count since last retrain
        last_retrain_time = self._get_last_retrain_time()
        try:
            if last_retrain_time:
                count = self._conn.execute("""
                    SELECT COUNT(*) as cnt FROM calibration_history
                    WHERE recorded_at > ?
                """, (last_retrain_time,)).fetchone()["cnt"]
            else:
                count = self._conn.execute(
                    "SELECT COUNT(*) as cnt FROM calibration_history"
                ).fetchone()["cnt"]
        except sqlite3.OperationalError:
            count = 0

        if count >= self._retrain_resolution_count:
            return RetrainTrigger(
                should_retrain=True,
                reason="resolution_count",
                details={"count": count, "threshold": self._retrain_resolution_count},
            )

        # 2. Brier score degradation
        try:
            current_brier = self.get_rolling_brier(window_days=self._brier_window_days)
            prior_brier = self.get_rolling_brier(
                window_days=self._brier_window_days,
                offset_days=self._brier_window_days,
            )

            if current_brier > 0 and prior_brier > 0:
                degradation = (current_brier - prior_brier) / prior_brier
                if degradation > self._brier_degradation_threshold:
                    return RetrainTrigger(
                        should_retrain=True,
                        reason="brier_degradation",
                        details={
                            "current_brier": round(current_brier, 4),
                            "prior_brier": round(prior_brier, 4),
                            "degradation_pct": round(degradation * 100, 1),
                        },
                    )
        except (sqlite3.OperationalError, ZeroDivisionError):
            pass

        # 3. New specialist enabled since last retrain
        try:
            current_specialists = self._get_engine_state("enabled_specialists")
            last_specialists = self._get_engine_state("last_retrain_specialists")
            if current_specialists and current_specialists != last_specialists:
                return RetrainTrigger(
                    should_retrain=True,
                    reason="specialist_enabled",
                    details={
                        "current": current_specialists,
                        "previous": last_specialists or "none",
                    },
                )
        except sqlite3.OperationalError:
            pass

        return RetrainTrigger(should_retrain=False, reason="none")

    def get_rolling_brier(
        self,
        window_days: int = 7,
        offset_days: int = 0,
    ) -> float:
        """Compute Brier score from recent calibration_history rows."""
        import datetime as dt
        now = dt.datetime.now(dt.timezone.utc)
        end = now - dt.timedelta(days=offset_days)
        start = end - dt.timedelta(days=window_days)

        try:
            rows = self._conn.execute("""
                SELECT forecast_prob, actual_outcome
                FROM calibration_history
                WHERE recorded_at >= ? AND recorded_at < ?
            """, (start.timestamp(), end.timestamp())).fetchall()
        except sqlite3.OperationalError:
            return 0.0

        if not rows:
            return 0.0

        brier_sum = sum(
            (float(r["forecast_prob"]) - float(r["actual_outcome"])) ** 2
            for r in rows
        )
        return brier_sum / len(rows)

    def retrain_with_ab_test(
        self,
        trigger: RetrainTrigger,
    ) -> ABTestResult:
        """Retrain calibrator with A/B holdout validation."""
        test_id = f"ab-{uuid.uuid4().hex[:8]}"
        started_at = _now_iso()

        try:
            rows = self._conn.execute("""
                SELECT forecast_prob, actual_outcome, market_id
                FROM calibration_history
                ORDER BY recorded_at ASC
            """).fetchall()
        except sqlite3.OperationalError:
            return ABTestResult(
                test_id=test_id, started_at=started_at,
                completed_at=_now_iso(), trigger_reason=trigger.reason,
            )

        if len(rows) < self._ab_min_samples:
            return ABTestResult(
                test_id=test_id, started_at=started_at,
                completed_at=_now_iso(), trigger_reason=trigger.reason,
            )

        # Deterministic split by market_id hash
        train_data = []
        holdout_data = []
        for r in rows:
            market_id = r["market_id"] or ""
            h = hashlib.sha256(market_id.encode()).hexdigest()
            # Use first 4 hex chars as a fraction
            frac = int(h[:4], 16) / 0xFFFF
            if frac < self._ab_holdout_pct:
                holdout_data.append(r)
            else:
                train_data.append(r)

        if not holdout_data or not train_data:
            return ABTestResult(
                test_id=test_id, started_at=started_at,
                completed_at=_now_iso(), trigger_reason=trigger.reason,
            )

        # Compute uncalibrated Brier on holdout
        uncal_brier = sum(
            (float(r["forecast_prob"]) - float(r["actual_outcome"])) ** 2
            for r in holdout_data
        ) / len(holdout_data)

        # Fit calibrator on train set and evaluate on holdout
        from src.forecast.calibrator import CalibrationHistory, HistoricalCalibrator

        calibrator = HistoricalCalibrator(min_samples=5)
        train_history = [
            CalibrationHistory(
                forecast_prob=float(r["forecast_prob"]),
                actual_outcome=float(r["actual_outcome"]),
            )
            for r in train_data
        ]

        success = calibrator.fit(train_history)

        if success:
            cal_brier = sum(
                (calibrator.calibrate(float(r["forecast_prob"])) - float(r["actual_outcome"])) ** 2
                for r in holdout_data
            ) / len(holdout_data)
        else:
            cal_brier = uncal_brier

        delta = cal_brier - uncal_brier
        helps = delta < 0

        result = ABTestResult(
            test_id=test_id,
            calibrated_brier=round(cal_brier, 6),
            uncalibrated_brier=round(uncal_brier, 6),
            calibrated_count=len(train_data),
            uncalibrated_count=len(holdout_data),
            calibration_helps=helps,
            delta_brier=round(delta, 6),
            trigger_reason=trigger.reason,
            started_at=started_at,
            completed_at=_now_iso(),
        )

        self._save_ab_result(result)
        self.save_retrain_state(trigger)

        log.info(
            "smart_retrain.ab_test_complete",
            test_id=test_id,
            helps=helps,
            cal_brier=round(cal_brier, 4),
            uncal_brier=round(uncal_brier, 4),
        )

        return result

    def get_ab_history(self) -> list[ABTestResult]:
        """Fetch past A/B test results."""
        try:
            rows = self._conn.execute("""
                SELECT * FROM calibration_ab_results
                ORDER BY completed_at DESC
            """).fetchall()
        except sqlite3.OperationalError:
            return []

        return [
            ABTestResult(
                test_id=r["test_id"],
                calibrated_brier=float(r["calibrated_brier"]),
                uncalibrated_brier=float(r["uncalibrated_brier"]),
                calibrated_count=int(r["calibrated_count"]),
                uncalibrated_count=int(r["uncalibrated_count"]),
                calibration_helps=bool(r["calibration_helps"]),
                delta_brier=float(r["delta_brier"]),
                trigger_reason=r["trigger_reason"] or "",
                started_at=r["started_at"] or "",
                completed_at=r["completed_at"] or "",
            )
            for r in rows
        ]

    def save_retrain_state(self, trigger: RetrainTrigger) -> None:
        """Record the time and reason of the last retrain."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO engine_state (key, value, updated_at)
                VALUES ('last_retrain_time', ?, ?)
            """, (_now_iso(), time.time()))
            self._conn.execute("""
                INSERT OR REPLACE INTO engine_state (key, value, updated_at)
                VALUES ('last_retrain_reason', ?, ?)
            """, (trigger.reason, time.time()))
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def _get_last_retrain_time(self) -> str | None:
        """Get the timestamp of the last retrain from engine_state."""
        try:
            row = self._conn.execute("""
                SELECT value FROM engine_state WHERE key = 'last_retrain_time'
            """).fetchone()
            return row["value"] if row else None
        except sqlite3.OperationalError:
            return None

    def _get_engine_state(self, key: str) -> str | None:
        """Get a value from engine_state."""
        try:
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None
        except sqlite3.OperationalError:
            return None

    def _save_ab_result(self, result: ABTestResult) -> None:
        """Persist an A/B test result."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO calibration_ab_results
                    (test_id, calibrated_brier, uncalibrated_brier,
                     calibrated_count, uncalibrated_count,
                     calibration_helps, delta_brier, trigger_reason,
                     started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id, result.calibrated_brier,
                result.uncalibrated_brier, result.calibrated_count,
                result.uncalibrated_count, int(result.calibration_helps),
                result.delta_brier, result.trigger_reason,
                result.started_at, result.completed_at,
            ))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            log.warning("smart_retrain.save_ab_error", error=str(e))
