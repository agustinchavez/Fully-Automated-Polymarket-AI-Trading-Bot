"""Pydantic models for the backtesting subsystem."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HistoricalMarketRecord(BaseModel):
    """A resolved market scraped from the Polymarket Gamma API."""

    condition_id: str
    question: str
    description: str = ""
    category: str = ""
    market_type: str = ""
    resolution: str = ""           # "YES" or "NO"
    resolved_at: str = ""          # ISO timestamp
    created_at: str = ""           # Market creation time
    end_date: str = ""             # Stated end date
    volume_usd: float = 0.0
    liquidity_usd: float = 0.0
    slug: str = ""
    outcomes_json: str = "[]"      # JSON list of outcome names
    final_prices_json: str = "{}"  # JSON map: outcome -> final price
    tokens_json: str = "[]"        # JSON list of token data
    raw_json: str = "{}"           # Full raw API response
    scraped_at: str = ""           # When we scraped this record


class LLMCacheRecord(BaseModel):
    """Cached LLM response for deterministic replay."""

    cache_key: str                 # SHA-256 of (question + model + template_version)
    market_question_hash: str      # SHA-256 of question text alone
    model_name: str
    prompt_hash: str               # SHA-256 of the full rendered prompt
    response_json: str = "{}"      # Full parsed LLM response
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    created_at: str = ""


class BacktestRunRecord(BaseModel):
    """Metadata for a single backtest run."""

    run_id: str                    # UUID
    name: str = ""                 # User-provided or auto-generated
    config_json: str = "{}"        # Full BotConfig serialized
    config_diff_json: str = "{}"   # Only fields that differ from default
    start_date: str = ""           # Market resolution date range start
    end_date: str = ""             # Market resolution date range end
    status: str = "pending"        # pending | running | completed | failed
    markets_processed: int = 0
    markets_traded: int = 0
    total_pnl: float = 0.0
    brier_score: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    results_json: str = "{}"       # Full BacktestResult serialized
    started_at: str = ""
    completed_at: str = ""
    duration_secs: float = 0.0


class BacktestTradeRecord(BaseModel):
    """Individual simulated trade within a backtest run."""

    run_id: str
    market_condition_id: str
    question: str = ""
    category: str = ""
    direction: str = ""            # BUY_YES or BUY_NO
    model_probability: float = 0.5
    implied_probability: float = 0.5
    edge: float = 0.0
    confidence_level: str = "LOW"
    stake_usd: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0       # 1.0 or 0.0 based on resolution
    pnl: float = 0.0
    resolution: str = ""           # YES or NO
    actual_outcome: float = 0.0    # 1.0 or 0.0
    forecast_correct: bool = False
    created_at: str = ""
    # Phase 6: Realistic fill simulation fields
    slippage_bps: float = 0.0
    fill_rate: float = 1.0
    simulated_impact_pct: float = 0.0
    fill_delay_ms: int = 0
    fee_paid_usd: float = 0.0
