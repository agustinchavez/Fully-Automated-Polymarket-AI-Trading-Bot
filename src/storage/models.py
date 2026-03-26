"""Database models — Pydantic models for storage records."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class MarketRecord(BaseModel):
    """Stored market data."""
    id: str
    condition_id: str = ""
    question: str = ""
    market_type: str = ""
    category: str = ""
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: str = ""
    resolution_source: str = ""
    first_seen: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    last_updated: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class ForecastRecord(BaseModel):
    """Stored forecast."""
    id: str = ""
    market_id: str
    question: str = ""
    market_type: str = ""
    implied_probability: float = 0.5
    model_probability: float = 0.5
    edge: float = 0.0
    confidence_level: str = "LOW"
    evidence_quality: float = 0.0
    num_sources: int = 0
    decision: str = "NO TRADE"
    reasoning: str = ""
    evidence_json: str = "[]"
    invalidation_triggers_json: str = "[]"
    research_evidence_json: str = "{}"
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class OrderRecord(BaseModel):
    """Tracked order lifecycle (Phase 10).

    Separates order state from confirmed fills. The ``trades`` table
    continues to represent confirmed executions while this model tracks
    the full order lifecycle: pending → submitted → partial → filled | cancelled | expired | failed.
    """
    order_id: str
    clob_order_id: str = ""
    market_id: str
    token_id: str = ""
    side: str = ""              # COMPAT: legacy field, prefer action_side + outcome_side
    order_type: str = ""
    price: float = 0.0
    size: float = 0.0
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    stake_usd: float = 0.0
    status: str = "pending"
    dry_run: bool = True
    ttl_secs: int = 0
    error: str = ""
    action_side: str = ""   # "BUY" or "SELL"
    outcome_side: str = ""  # "YES" or "NO"
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class TradeRecord(BaseModel):
    """Stored trade."""
    id: str = ""
    order_id: str
    market_id: str
    token_id: str = ""
    side: str = ""              # COMPAT: legacy field, prefer action_side + outcome_side
    price: float = 0.0
    size: float = 0.0
    stake_usd: float = 0.0
    status: str = ""
    dry_run: bool = True
    action_side: str = ""   # "BUY" or "SELL"
    outcome_side: str = ""  # "YES" or "NO"
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class PositionRecord(BaseModel):
    """Tracked open position."""
    market_id: str
    token_id: str = ""
    direction: str = ""         # COMPAT: legacy field, prefer action_side + outcome_side
    entry_price: float = 0.0
    size: float = 0.0
    stake_usd: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0
    action_side: str = ""   # "BUY" or "SELL"
    outcome_side: str = ""  # "YES" or "NO"
    opened_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    question: str = ""
    market_type: str = ""


class ClosedPositionRecord(BaseModel):
    """Archived closed position with full context."""
    id: str = ""
    market_id: str
    token_id: str = ""
    direction: str = ""         # COMPAT: legacy field, prefer action_side + outcome_side
    entry_price: float = 0.0
    exit_price: float = 0.0
    size: float = 0.0
    stake_usd: float = 0.0
    pnl: float = 0.0
    close_reason: str = ""
    action_side: str = ""   # "BUY" or "SELL"
    outcome_side: str = ""  # "YES" or "NO"
    question: str = ""
    market_type: str = ""
    opened_at: str = ""
    closed_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class PerformanceLogRecord(BaseModel):
    """Record for the performance_log table — one per resolved/closed trade."""
    market_id: str
    question: str = ""
    category: str = "UNKNOWN"
    forecast_prob: float = 0.0
    actual_outcome: float | None = None
    edge_at_entry: float = 0.0
    confidence: str = "LOW"
    evidence_quality: float = 0.0
    stake_usd: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    holding_hours: float = 0.0
    resolved_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class RegimeHistoryRecord(BaseModel):
    """Detected market regime snapshot."""
    timestamp: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    regime: str = "normal"  # normal | volatile | trending | mean_reverting
    confidence: float = 0.0
    volatility_30d: float = 0.0
    trend_strength: float = 0.0
    metadata_json: str = "{}"


class ModelForecastLogRecord(BaseModel):
    """Individual model forecast within an ensemble run."""
    market_id: str
    model_name: str
    model_probability: float = 0.5
    confidence_level: str = "LOW"
    reasoning: str = ""
    latency_ms: float = 0.0
    error: str = ""
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class CandidateRecord(BaseModel):
    """Market candidate discovered during scan, before research."""
    market_id: str
    question: str = ""
    market_type: str = ""
    score: float = 0.0
    volume: float = 0.0
    liquidity: float = 0.0
    implied_probability: float = 0.5
    spread: float = 0.0
    status: str = "pending"  # pending | researching | forecasted | traded | skipped
    discovered_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class AlertRecord(BaseModel):
    """Persisted alert for audit trail."""
    level: str = "info"
    title: str = ""
    message: str = ""
    channels_sent: str = "[]"  # JSON list
    data_json: str = "{}"
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class CalibrationHistoryRecord(BaseModel):
    """Calibration model training snapshot."""
    num_samples: int = 0
    brier_score: float = 0.0
    calibration_error: float = 0.0
    model_params_json: str = "{}"
    trained_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class ArbOpportunityRecord(BaseModel):
    """Cross-platform arb opportunity from scanner."""
    arb_id: str
    match_method: str = ""
    match_confidence: float = 0.0
    poly_market_id: str = ""
    poly_question: str = ""
    poly_yes_price: float = 0.0
    poly_no_price: float = 0.0
    kalshi_ticker: str = ""
    kalshi_title: str = ""
    kalshi_yes_price: float = 0.0
    kalshi_no_price: float = 0.0
    spread: float = 0.0
    net_spread: float = 0.0
    direction: str = ""
    buy_platform: str = ""
    sell_platform: str = ""
    buy_price: float = 0.0
    sell_price: float = 0.0
    total_fees: float = 0.0
    is_actionable: bool = False
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class ArbTradeRecord(BaseModel):
    """Paired arb trade — both legs logged as a unit."""
    arb_id: str
    buy_platform: str = ""
    sell_platform: str = ""
    buy_market_id: str = ""
    sell_market_id: str = ""
    buy_price: float = 0.0
    sell_price: float = 0.0
    buy_fill_price: float = 0.0
    sell_fill_price: float = 0.0
    stake_usd: float = 0.0
    net_pnl: float = 0.0
    status: str = ""
    unwind_reason: str = ""
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )


class ComplementaryArbRecord(BaseModel):
    """Complementary arb opportunity (YES+NO < 1.0)."""
    market_id: str
    question: str = ""
    yes_price: float = 0.0
    no_price: float = 0.0
    combined_cost: float = 0.0
    guaranteed_profit: float = 0.0
    fee_cost: float = 0.0
    net_profit: float = 0.0
    is_actionable: bool = False
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
