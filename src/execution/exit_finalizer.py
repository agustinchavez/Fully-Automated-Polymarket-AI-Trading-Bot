"""Shared exit finalization — archives position, records analytics, cleans up.

Used by both the engine loop (_finalize_exit) and reconciliation
(SELL fill handling) to ensure consistent exit behavior:
  1. Archive position to closed_positions
  2. Record performance log
  3. Run post-mortem (if enabled)
  4. Remove open position
  5. Insert alert
"""

from __future__ import annotations

from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


class ExitFinalizer:
    """Consolidates the 5-step exit finalization into a single class."""

    def __init__(self, db: object, config: object):
        self._db = db
        self._config = config

    def finalize(
        self,
        pos: Any,
        exit_price: float,
        pnl: float,
        close_reason: str,
        mkt_record: Any = None,
    ) -> None:
        """Run full exit finalization for a position.

        Steps:
          1. archive_position (closed_positions table)
          2. record performance log
          3. run post-mortem if enabled
          4. remove open position
          5. insert alert
        """
        rounded_pnl = round(pnl, 4)
        reason_tag = close_reason.split(":")[0]

        # Step 1: Archive
        try:
            self._db.archive_position(
                pos=pos,
                exit_price=exit_price,
                pnl=rounded_pnl,
                close_reason=reason_tag,
            )
        except Exception as e:
            log.warning("exit_finalizer.archive_error", error=str(e))

        # Step 2: Performance log
        self._record_performance_log(pos, exit_price, rounded_pnl, mkt_record)

        # Step 3: Post-mortem
        self._maybe_run_post_mortem(pos.market_id)

        # Step 4: Remove open position
        try:
            self._db.remove_position(pos.market_id)
        except Exception as e:
            log.warning("exit_finalizer.remove_error", error=str(e))

        # Step 5: Alert
        try:
            self._db.insert_alert(
                "warning",
                f"Auto-exit {pos.market_id[:8]}: {close_reason} "
                f"(PNL ${pnl:.2f})",
                "engine",
            )
        except Exception as e:
            log.warning("exit_finalizer.alert_error", error=str(e))

    def _record_performance_log(
        self,
        pos: Any,
        exit_price: float,
        pnl: float,
        mkt_record: Any = None,
    ) -> None:
        """Write a closed position to the performance_log table for analytics."""
        if not self._db:
            return
        try:
            import datetime as _dt
            from src.storage.models import PerformanceLogRecord

            # Look up the most recent forecast for this market
            forecasts = self._db.get_forecasts(market_id=pos.market_id, limit=1)
            fc = forecasts[0] if forecasts else None

            # Compute holding hours
            holding_hours = 0.0
            try:
                opened = _dt.datetime.fromisoformat(
                    pos.opened_at.replace("Z", "+00:00")
                )
                now = _dt.datetime.now(_dt.timezone.utc)
                holding_hours = (now - opened).total_seconds() / 3600
            except Exception:
                pass

            # Determine category from market record or forecast
            category = "UNKNOWN"
            if mkt_record and getattr(mkt_record, "category", None):
                category = mkt_record.category
            elif fc and getattr(fc, "market_type", None):
                category = fc.market_type

            # Determine actual outcome from exit price (for resolved markets)
            actual_outcome = None
            if exit_price >= 0.98:
                actual_outcome = 1.0
            elif exit_price <= 0.02:
                actual_outcome = 0.0

            self._db.insert_performance_log(PerformanceLogRecord(
                market_id=pos.market_id,
                question=(
                    getattr(mkt_record, "question", "")
                    if mkt_record else getattr(pos, "question", "")
                ),
                category=category,
                forecast_prob=fc.model_probability if fc else 0.0,
                actual_outcome=actual_outcome,
                edge_at_entry=fc.edge if fc else 0.0,
                confidence=fc.confidence_level if fc else "LOW",
                evidence_quality=fc.evidence_quality if fc else 0.0,
                stake_usd=pos.stake_usd,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                holding_hours=round(holding_hours, 2),
            ))

            log.info(
                "exit_finalizer.performance_log_recorded",
                market_id=pos.market_id[:8],
                pnl=pnl,
                holding_hours=round(holding_hours, 1),
                category=category,
            )

            # Backfill model_forecast_log with actual outcome
            if actual_outcome is not None:
                resolved_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
                self._db.conn.execute(
                    "UPDATE model_forecast_log "
                    "SET actual_outcome = ?, resolved_at = ? "
                    "WHERE market_id = ? AND actual_outcome IS NULL",
                    (actual_outcome, resolved_at, pos.market_id),
                )
                self._db.conn.commit()
        except Exception as e:
            log.warning("exit_finalizer.performance_log_error", error=str(e))

    def _maybe_run_post_mortem(self, market_id: str) -> None:
        """Run post-mortem analysis on a resolved market (Phase 8)."""
        cl_cfg = getattr(self._config, "continuous_learning", None)
        if not cl_cfg or not getattr(cl_cfg, "post_mortem_enabled", False):
            return
        if not self._db:
            return
        try:
            from src.analytics.post_mortem import PostMortemAnalyzer

            conn = self._db._conn
            analyzer = PostMortemAnalyzer(conn)
            analysis = analyzer.analyze_market(market_id)
            if analysis and analysis.was_confident_and_wrong:
                log.warning(
                    "exit_finalizer.confident_wrong_trade",
                    market_id=market_id[:8],
                    forecast=analysis.forecast_prob,
                    outcome=analysis.actual_outcome,
                )
        except Exception as e:
            log.warning("exit_finalizer.post_mortem_error", error=str(e))
