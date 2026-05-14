"""Shared configuration loader and Pydantic settings.

Supports:
  - YAML file loading with env var overrides
  - Hot-reload via file watcher
  - All subsystem configs: scanning, research, forecasting, risk,
    execution, storage, observability, engine, alerts, drawdown,
    portfolio, cache, timeline, microstructure, ensemble
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, List

import yaml
import warnings

from pydantic import BaseModel, Field, field_validator, model_validator


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ScanningConfig(BaseModel):
    min_volume_usd: float = 1000
    min_liquidity_usd: float = 500
    max_spread: float = 0.08
    max_days_to_expiry: int = 120
    max_market_age_hours: float = 720.0  # 30 days — most Polymarket markets are older
    categories: list[str] = Field(default_factory=list)
    batch_size: int = 100
    preferred_types: list[str] = Field(default_factory=lambda: ["MACRO", "ELECTION", "CORPORATE", "LEGAL", "TECHNOLOGY", "SCIENCE"])
    restricted_types: list[str] = Field(default_factory=lambda: ["WEATHER"])
    # Pre-research filter settings
    filter_min_score: int = 25
    filter_blocked_types: list[str] = Field(default_factory=lambda: ["UNKNOWN"])
    research_cooldown_minutes: int = 60
    category_cooldown_minutes: dict[str, int] = Field(default_factory=dict)


class ResearchConfig(BaseModel):
    max_sources: int = 10
    source_timeout_secs: int = 15
    primary_domains: dict[str, list[str]] = Field(default_factory=dict)
    secondary_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    min_corroborating_sources: int = 2
    search_provider: str = "fallback"
    fetch_full_content: bool = True
    max_content_length: int = 15000
    content_fetch_top_n: int = 5
    stale_days_penalty_threshold: int = 7
    stale_days_heavy_penalty: int = 30

    # Structured API connectors
    fred_enabled: bool = False
    congress_enabled: bool = False
    courtlistener_enabled: bool = False
    coingecko_enabled: bool = False
    openmeteo_enabled: bool = True
    gdelt_enabled: bool = True
    edgar_enabled: bool = True
    arxiv_enabled: bool = True
    openfda_enabled: bool = True
    worldbank_enabled: bool = True
    fred_max_series: int = 3
    congress_max_bills: int = 5
    coingecko_max_coins: int = 3
    gdelt_timespan_days: int = 7
    edgar_max_filings: int = 3
    arxiv_max_results: int = 5
    worldbank_mrv: int = 3
    openfda_api_key: str = ""

    # Phase 2: Behavioral signal connectors
    kalshi_prior_enabled: bool = False
    metaculus_enabled: bool = False
    metaculus_api_key: str = ""
    wikipedia_pageviews_enabled: bool = False
    google_trends_enabled: bool = False
    reddit_sentiment_enabled: bool = False
    pubmed_enabled: bool = False
    metaculus_min_forecasters: int = 5
    metaculus_min_jaccard: float = 0.20
    wikipedia_cache_ttl_secs: int = 14400  # 4 hours
    google_trends_volume_threshold_usd: float = 50000.0
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    pubmed_api_key: str = ""  # optional NCBI key
    pubmed_max_results: int = 5
    # Sports intelligence connectors
    sports_odds_enabled: bool = False
    sports_stats_enabled: bool = False
    sports_odds_api_key: str = ""         # The Odds API key
    sports_stats_api_key: str = ""        # API-Football key
    sports_book_weights: dict[str, float] = Field(default_factory=lambda: {
        "pinnacle": 0.40, "betfair": 0.30, "draftkings": 0.20, "fanduel": 0.10,
    })
    sports_min_books: int = 2             # min sportsbooks needed for consensus
    # Spotify charts (kworb.net scraper, no key needed)
    spotify_charts_enabled: bool = False
    # Binance futures funding rate (no key needed)
    crypto_futures_enabled: bool = False
    # DeFiLlama TVL (no key needed)
    defillama_enabled: bool = False
    # ACLED armed conflict data (free academic API key)
    acled_enabled: bool = False
    acled_api_key: str = ""
    acled_lookback_days: int = 30
    # GitHub activity (optional GITHUB_TOKEN for higher rate limits)
    github_activity_enabled: bool = False
    # Kronos crypto price forecast (requires torch, einops, safetensors)
    kronos_enabled: bool = False
    # Improvement 6: Additional consensus connectors
    manifold_enabled: bool = False
    predictit_enabled: bool = False


class ForecastingConfig(BaseModel):
    llm_model: str = "gpt-4o"
    evidence_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1500
    calibration_method: str = "platt"
    low_evidence_penalty: float = 0.15
    min_evidence_quality: float = 0.55
    min_confidence_level: str = "MEDIUM"  # Reject LOW confidence trades
    category_min_confidence: dict[str, str] = Field(default_factory=dict)  # per-category override
    evidence_fallback_models: list[str] = Field(
        default_factory=lambda: ["gpt-4o-mini", "claude-haiku-4-5-20251001"]
    )
    # Phase 2: Structured forecasting
    prompt_version: str = "v1"  # v1 (legacy) or v2 (structured reasoning chain)
    base_rate_enabled: bool = False  # inject base rates into prompt
    decomposition_enabled: bool = False  # enable question decomposition
    decomposition_model: str = "gpt-4o-mini"  # cheap model for decomposition
    max_sub_questions: int = 3  # max sub-questions per market


class LongshotConfig(BaseModel):
    """Longshot bias correction — documented in Betfair/PredictIt/Polymarket research."""
    enabled: bool = True
    low_threshold: float = 0.12
    high_threshold: float = 0.88
    correction_strength: float = 0.20   # 0 = no effect, 1 = maximum
    excluded_categories: list[str] = Field(default_factory=list)


class EnsembleConfig(BaseModel):
    """Multi-model ensemble configuration."""
    enabled: bool = True
    models: list[str] = Field(default_factory=lambda: [
        "gpt-4o", "claude-haiku-4-5-20251001", "gemini-2.5-flash",
        "grok-4-fast-reasoning", "deepseek-chat",
    ])
    aggregation: str = "median"  # trimmed_mean | median | weighted
    trim_fraction: float = 0.1
    weights: dict[str, float] = Field(default_factory=lambda: {
        "gpt-4o": 0.25,
        "claude-haiku-4-5-20251001": 0.25,
        "gemini-2.5-flash": 0.20,
        "grok-4-fast-reasoning": 0.15,
        "deepseek-chat": 0.15,
    })
    timeout_per_model_secs: int = 30
    min_models_required: int = 1
    fallback_model: str = "gpt-4o"
    deepseek_excluded_categories: list[str] = Field(
        default_factory=lambda: ["GEOPOLITICS", "ELECTION"]
    )
    evidence_model_gating_enabled: bool = False
    evidence_low_quality_threshold: float = 0.25   # below this: use 2 models
    evidence_medium_quality_threshold: float = 0.50  # below this: use 3 models
    longshot: LongshotConfig = Field(default_factory=LongshotConfig)

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "EnsembleConfig":
        if self.enabled and not self.models:
            raise ValueError("ensemble.models must be non-empty when ensemble is enabled")
        if self.models and self.min_models_required > len(self.models):
            raise ValueError(
                f"min_models_required ({self.min_models_required}) "
                f"> len(models) ({len(self.models)})"
            )
        valid_agg = {"trimmed_mean", "median", "weighted"}
        if self.aggregation not in valid_agg:
            raise ValueError(
                f"aggregation must be one of {valid_agg}, got {self.aggregation!r}"
            )
        if not (0.0 <= self.trim_fraction < 0.5):
            raise ValueError(
                f"trim_fraction must be in [0.0, 0.5), got {self.trim_fraction}"
            )
        return self


class ModelTierConfig(BaseModel):
    """Model tier routing — select model quality based on opportunity."""
    enabled: bool = True
    scout_models: list[str] = Field(default_factory=lambda: ["gpt-4o-mini"])
    standard_models: list[str] = Field(default_factory=lambda: ["gpt-4o"])
    premium_models: list[str] = Field(default_factory=lambda: [
        "gpt-4o", "claude-haiku-4-5-20251001", "gemini-2.5-flash",
    ])
    premium_min_volume_usd: float = 10000.0
    premium_min_edge: float = 0.06
    scout_max_evidence_quality: float = 0.4

    @model_validator(mode="after")
    def _non_empty_tiers(self) -> "ModelTierConfig":
        if self.enabled:
            if not self.scout_models:
                raise ValueError("scout_models must be non-empty when model_tiers is enabled")
            if not self.standard_models:
                raise ValueError("standard_models must be non-empty when model_tiers is enabled")
            if not self.premium_models:
                raise ValueError("premium_models must be non-empty when model_tiers is enabled")
        return self


class RiskConfig(BaseModel):
    max_stake_per_market: float = 50.0
    max_daily_loss: float = 500.0
    max_open_positions: int = 25
    min_edge: float = 0.04
    min_liquidity: float = 2000
    min_volume: float = 1000
    max_spread: float = 0.06
    kelly_fraction: float = 0.25
    max_bankroll_fraction: float = 0.05
    kill_switch: bool = False
    bankroll: float = 5000.0
    transaction_fee_pct: float = 0.02
    exit_fee_pct: float = 0.02  # Fee for selling before resolution
    gas_cost_usd: float = 0.01
    min_implied_probability: float = 0.05  # Block micro-probability markets
    stop_loss_pct: float = 0.20  # Exit when position loses 20%
    take_profit_pct: float = 0.30  # Exit when position gains 30%
    max_holding_hours: float = 240.0  # Auto-exit positions held longer than this (10 days)
    min_stake_usd: float = 1.0  # Minimum stake to avoid dust trades
    # Volatility thresholds for position sizing
    volatility_high_threshold: float = 0.15
    volatility_med_threshold: float = 0.10
    volatility_high_min_mult: float = 0.4
    volatility_med_min_mult: float = 0.6
    min_annualized_edge: float = 0.15  # Reject trades below 15% annualized return (spec suggests 0.50)
    # Phase 3: Edge uncertainty
    uncertainty_enabled: bool = False          # penalize edge based on forecast uncertainty
    uncertainty_penalty_factor: float = 0.5    # how much uncertainty penalizes edge (0-1)
    # Cost model: convert percentage-of-stake into probability space (cost_pct * price)
    use_probability_space_costs: bool = True
    # Improvement 3: TWAP edge reference
    use_twap_edge: bool = False
    twap_window_hours: float = 2.0
    twap_max_divergence: float = 0.08  # max pp TWAP can differ from spot
    category_stake_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "MACRO": 1.0,
            "CORPORATE": 0.75,
            "ELECTION": 0.50,
        }
    )

    @field_validator("kelly_fraction")
    @classmethod
    def _clamp_kelly(cls, v: float) -> float:
        if v < 0 or v > 1.0:
            raise ValueError(f"kelly_fraction must be in [0, 1], got {v}")
        return v

    @field_validator("bankroll")
    @classmethod
    def _positive_bankroll(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bankroll must be positive, got {v}")
        return v

    @field_validator("max_stake_per_market", "max_daily_loss")
    @classmethod
    def _positive_limits(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "RiskConfig":
        if self.max_stake_per_market > self.bankroll:
            raise ValueError(
                f"max_stake_per_market ({self.max_stake_per_market}) "
                f"must be <= bankroll ({self.bankroll})"
            )
        total_fees = self.transaction_fee_pct + self.exit_fee_pct
        if self.min_edge <= total_fees:
            warnings.warn(
                f"min_edge ({self.min_edge}) <= total fees "
                f"({self.transaction_fee_pct} + {self.exit_fee_pct} = {total_fees}). "
                f"Trades may have zero or negative profit margin.",
                UserWarning,
                stacklevel=2,
            )
        if self.volatility_high_min_mult >= self.volatility_med_min_mult:
            raise ValueError(
                f"volatility_high_min_mult ({self.volatility_high_min_mult}) "
                f"must be < volatility_med_min_mult ({self.volatility_med_min_mult})"
            )
        return self


class DrawdownConfig(BaseModel):
    """Drawdown management configuration."""
    enabled: bool = True
    max_drawdown_pct: float = 0.20
    warning_drawdown_pct: float = 0.10
    critical_drawdown_pct: float = 0.15
    auto_reduce_at_warning: float = 0.50
    auto_reduce_at_critical: float = 0.25
    auto_kill_at_max: bool = True
    heat_window_trades: int = 10
    heat_loss_streak_threshold: int = 3
    heat_reduction_factor: float = 0.50
    recovery_trades_required: int = 5
    snapshot_interval_minutes: int = 15

    @model_validator(mode="after")
    def _threshold_ordering(self) -> "DrawdownConfig":
        if not (self.warning_drawdown_pct < self.critical_drawdown_pct < self.max_drawdown_pct):
            raise ValueError(
                f"Drawdown thresholds must satisfy "
                f"warning ({self.warning_drawdown_pct}) < "
                f"critical ({self.critical_drawdown_pct}) < "
                f"max ({self.max_drawdown_pct})"
            )
        return self


class BudgetConfig(BaseModel):
    """API cost budget management."""
    enabled: bool = True
    daily_limit_usd: float = 5.0
    warning_pct: float = 0.80


class CircuitBreakerSettings(BaseModel):
    """Circuit breaker global settings."""
    enabled: bool = True
    default_failure_threshold: int = 5
    default_window_secs: float = 60.0
    default_recovery_timeout_secs: float = 30.0


class PortfolioConfig(BaseModel):
    """Portfolio-level risk configuration."""
    max_category_exposure_pct: float = 0.35
    max_single_event_exposure_pct: float = 0.25
    max_correlated_positions: int = 4
    correlation_similarity_threshold: float = 0.7
    rebalance_check_interval_minutes: int = 30
    category_limits: dict[str, float] = Field(default_factory=lambda: {
        "MACRO": 0.40, "ELECTION": 0.35, "CORPORATE": 0.30,
        "WEATHER": 0.15, "SPORTS": 0.15,
    })
    # Phase 3: Correlation-aware VaR gate
    var_gate_enabled: bool = False
    max_portfolio_var_pct: float = 0.10
    same_event_same_outcome_corr: float = 0.8
    same_event_diff_outcome_corr: float = 0.3
    same_category_corr: float = 0.15


class TimelineConfig(BaseModel):
    """Resolution timeline configuration."""
    near_resolution_hours: int = 48
    near_resolution_confidence_boost: float = 0.15
    early_market_uncertainty_penalty: float = 0.10
    early_market_days_threshold: int = 60
    exit_before_resolution_hours: int = 0  # Disabled — hold through resolution
    time_decay_urgency_start_days: int = 7
    time_decay_max_multiplier: float = 1.5


class MicrostructureConfig(BaseModel):
    """Market microstructure analysis configuration."""
    whale_size_threshold_usd: float = 2000.0
    flow_imbalance_windows: list[int] = Field(default_factory=lambda: [60, 240, 1440])
    depth_change_alert_pct: float = 0.30
    trade_acceleration_window_mins: int = 30
    trade_acceleration_threshold: float = 2.0
    vwap_lookback_trades: int = 100


class ExecutionConfig(BaseModel):
    dry_run: bool = True
    default_order_type: str = "limit"
    slippage_tolerance: float = 0.01
    limit_order_ttl_secs: int = 300
    max_retries: int = 3
    retry_backoff_secs: float = 2.0
    twap_enabled: bool = True
    twap_num_slices: int = 5
    twap_interval_secs: int = 30
    adaptive_pricing: bool = True
    queue_position_target: str = "mid"
    max_market_impact_pct: float = 0.10
    stale_order_cancel_secs: int = 600
    iceberg_threshold_usd: float = 500.0
    iceberg_show_pct: float = 0.20
    # Phase 6: Patience window
    patience_window_enabled: bool = False
    patience_window_max_secs: int = 300
    patience_check_interval_secs: int = 5
    edge_immediate_multiplier: float = 2.0
    edge_deterioration_cancel: bool = True
    # Phase 10: Live execution hardening
    live_exit_routing: bool = False
    reconciliation_enabled: bool = False
    reconciliation_interval_secs: int = 30
    stale_order_cancel_enabled: bool = False
    # Phase 6: Auto strategy selection
    auto_strategy_selection_enabled: bool = False
    auto_strategy_thin_depth_usd: float = 5000.0
    auto_strategy_large_order_pct: float = 0.10
    auto_strategy_learning_enabled: bool = False
    auto_strategy_min_samples: int = 10
    # Phase 10B: Invariant checks
    invariant_checks_enabled: bool = False
    invariant_check_interval_cycles: int = 10
    # Phase 10E: Execution plan orchestration
    plan_orchestration_enabled: bool = False


class StorageConfig(BaseModel):
    db_type: str = "sqlite"
    sqlite_path: str = "data/bot.db"


class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = True
    search_ttl_secs: int = 3600
    orderbook_ttl_secs: int = 30
    llm_response_ttl_secs: int = 1800
    market_list_ttl_secs: int = 300
    max_cache_size_mb: int = 100


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/bot.log"
    enable_metrics: bool = True
    reports_dir: str = "reports/"


class AlertsConfig(BaseModel):
    """Alerting configuration."""
    enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    slack_webhook_url: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""
    alert_on_trade: bool = True
    alert_on_risk_breach: bool = True
    alert_on_drawdown_warning: bool = True
    alert_on_system_error: bool = True
    alert_on_kill_switch: bool = True
    daily_summary_enabled: bool = True
    daily_summary_hour: int = 18
    min_alert_interval_secs: int = 60
    min_alert_level: str = "info"  # info | warning | critical
    zero_trade_alert_cycles: int = 20  # alert after N consecutive no-trade cycles

    @property
    def telegram_token(self) -> str:
        return self.telegram_bot_token

    @property
    def discord_webhook(self) -> str:
        return self.discord_webhook_url

    @property
    def slack_webhook(self) -> str:
        return self.slack_webhook_url


class WalletScannerConfig(BaseModel):
    """Whale / smart-money wallet scanner configuration."""
    enabled: bool = True
    scan_interval_minutes: int = 15
    min_whale_count: int = 1       # single whale with big $ = valid signal
    min_conviction_score: float = 15.0
    max_wallets: int = 20          # max wallets to track
    conviction_edge_boost: float = 0.08  # boost edge by 8% when whales agree
    conviction_edge_penalty: float = 0.02  # penalise edge when whales disagree
    whale_convergence_min_edge: float = 0.02  # lower min_edge when whale+model agree
    track_leaderboard: bool = True  # auto-track leaderboard wallets
    custom_wallets: list[str] = Field(default_factory=list)  # user-added wallet addresses
    # Phase 7: Enhanced Whale Intelligence
    whale_quality_scoring_enabled: bool = False
    whale_quality_lookback_days: int = 90
    whale_quality_min_percentile: float = 60.0
    whale_timing_lookback_hours: int = 24
    whale_timing_favorable_threshold: float = 0.02  # 2% move = favorable
    enhanced_min_whale_count: int = 3
    enhanced_conviction_edge_boost: float = 0.04
    enhanced_conviction_edge_penalty: float = 0.03
    # Improvement 4: Smart money as LLM signal
    whale_in_prompt: bool = False        # inject whale positions into LLM prompt
    wallet_auto_discover: bool = False   # auto-discover top wallets from leaderboard
    wallet_discover_top_n: int = 25
    wallet_min_pnl: float = 100_000.0


class UMAConfig(BaseModel):
    """UMA Oracle dispute resolution monitoring."""
    enabled: bool = False
    refresh_interval_mins: int = 15


class CalendarConfig(BaseModel):
    """Economic & political calendar awareness."""
    enabled: bool = False
    pre_event_size_reduction: float = 0.5  # Kelly multiplier when high-impact event < 24h
    refresh_interval_hours: int = 6
    lookahead_days: int = 14


class EngineConfig(BaseModel):
    """Main trading engine configuration."""
    scan_interval_minutes: int = 15
    research_interval_minutes: int = 30
    position_check_interval_minutes: int = 5
    max_concurrent_research: int = 3
    max_markets_per_cycle: int = 5
    auto_start: bool = False
    paper_mode: bool = True
    cycle_interval_secs: int = 300  # 5 minutes between full cycles


class BacktestConfig(BaseModel):
    """Backtesting engine configuration."""
    db_path: str = "data/backtest.db"
    default_implied_prob: float = 0.5       # When no historical price available
    default_slippage_pct: float = 0.005     # 0.5% slippage
    cache_llm_responses: bool = True
    mock_evidence_quality: float = 0.5      # Evidence quality for synthetic evidence
    max_markets_per_run: int = 0            # 0 = unlimited
    prompt_template_version: str = "v1"     # Cache key component
    # Phase 6: Realistic fill simulation
    realistic_fills_enabled: bool = False        # config-gated, disabled by default
    fill_sim_depth_multiplier: float = 1.0       # scales modeled available liquidity
    fill_sim_partial_fill_enabled: bool = True    # allow partial fills
    fill_sim_delay_min_ms: int = 50              # minimum fill delay in ms
    fill_sim_delay_max_ms: int = 500             # maximum fill delay in ms
    fill_sim_price_drift_vol: float = 0.001      # vol for price drift during delay
    fill_sim_fee_entry_pct: float = 0.02         # entry fee percentage
    fill_sim_fee_exit_pct: float = 0.02          # exit fee percentage


class SpecialistsConfig(BaseModel):
    """Domain-specific forecasting specialists (Phase 4)."""
    enabled: bool = False
    enabled_specialists: list[str] = Field(default_factory=list)
    weather_min_edge: float = 0.08
    weather_api_base: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    crypto_min_edge: float = 0.04
    crypto_candle_source: str = "binance"
    politics_polling_weight: float = 0.6


class ContinuousLearningConfig(BaseModel):
    """Continuous learning & self-improvement (Phase 8)."""
    # Post-mortem analysis
    post_mortem_enabled: bool = False
    confident_wrong_threshold: float = 0.75
    weekly_summary_enabled: bool = False
    # Evidence quality tracking
    evidence_tracking_enabled: bool = False
    min_citations_for_ranking: int = 5
    source_quality_min_markets: int = 100
    auto_weight_sources: bool = False
    # Parameter optimizer
    param_optimizer_enabled: bool = False
    optimization_interval_days: int = 7
    num_perturbations: int = 30
    perturbation_range_pct: float = 0.20
    min_sharpe_improvement_pct: float = 0.10
    significance_threshold: float = 0.05
    lookback_days: int = 30
    # Smart calibration retraining
    smart_retrain_enabled: bool = False
    retrain_resolution_count: int = 30
    brier_degradation_threshold: float = 0.10
    brier_window_days: int = 7
    ab_holdout_pct: float = 0.20
    ab_min_samples: int = 20
    auto_disable_bad_calibration: bool = False


class DigestConfig(BaseModel):
    """Weekly digest report configuration."""
    enabled: bool = True
    schedule_day_of_week: str = "mon"
    schedule_hour: int = 8
    lookback_days: int = 7
    min_data_days: int = 3
    split_long_messages: bool = True


class AnalystConfig(BaseModel):
    """AI analyst configuration (Phase 3 — multi-provider)."""
    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 1500
    temperature: float = 0.3
    timeout_secs: int = 45
    min_resolved_trades: int = 50
    min_data_days: int = 28
    rate_limit_hours: int = 6
    schedule_enabled: bool = False
    schedule_day_of_week: str = "mon"
    schedule_hour: int = 9

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        valid = {"anthropic", "openai", "google", "deepseek", "xai"}
        if v.lower() not in valid:
            raise ValueError(f"analyst.provider must be one of {valid}, got {v!r}")
        return v.lower()


class ProductionConfig(BaseModel):
    """Production deployment & live trading (Phase 9)."""
    enabled: bool = False
    # Kill switch enhancements
    daily_loss_kill_pct: float = 0.05       # auto-kill at -5% of bankroll
    daily_loss_kill_enabled: bool = True
    persist_kill_switch: bool = True
    require_manual_restart_after_kill: bool = True
    # Graduated deployment
    deployment_stage: str = "paper"          # paper|week1|week2|week3_4|month2_plus
    auto_advance_stages: bool = False
    week1_bankroll: float = 100.0
    week1_max_stake: float = 5.0
    week2_bankroll: float = 500.0
    week2_max_stake: float = 25.0
    week3_4_bankroll: float = 2000.0
    week3_4_max_stake: float = 50.0
    week1_max_loss_pct: float = 0.10
    # Pre-flight thresholds
    preflight_min_sharpe: float = 1.0
    preflight_min_paper_days: int = 30
    preflight_backtest_paper_tolerance: float = 0.25
    # Telegram kill bot
    telegram_kill_enabled: bool = False
    telegram_kill_token: str = ""
    telegram_kill_chat_id: str = ""
    # Discord kill bot
    discord_kill_enabled: bool = False
    discord_kill_token: str = ""
    discord_kill_channel_id: str = ""
    # Slack kill bot
    slack_kill_enabled: bool = False
    slack_kill_bot_token: str = ""     # xoxb-...
    slack_kill_app_token: str = ""     # xapp-...
    slack_kill_channel_id: str = ""
    # Sentry
    sentry_dsn: str = ""


class ArbitrageConfig(BaseModel):
    """Cross-platform and intra-platform arbitrage (Phase 5)."""
    enabled: bool = False
    # Kalshi connector
    kalshi_api_base: str = "https://api.elections.kalshi.com"
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_paper_mode: bool = True
    # Cross-platform arb
    scan_interval_secs: int = 60
    min_arb_edge: float = 0.03           # 3% minimum after fees
    polymarket_fee_pct: float = 0.02     # 2% round-trip
    kalshi_fee_pct: float = 0.02         # 2% round-trip
    max_arb_position_usd: float = 200.0
    max_arb_positions_count: int = 5
    execution_timeout_secs: int = 30
    # Intra-Polymarket arb
    complementary_threshold: float = 0.97  # YES+NO sum < this = opportunity
    correlated_min_divergence: float = 0.10
    # Market matching
    manual_mappings_json: str = "{}"     # JSON: {kalshi_ticker: poly_condition_id}
    match_min_confidence: float = 0.6
    # Paired trade
    unwind_on_partial_fill: bool = True

    @model_validator(mode="after")
    def _validate_arb_config(self) -> "ArbitrageConfig":
        if self.enabled:
            total_fees = self.polymarket_fee_pct + self.kalshi_fee_pct
            if self.min_arb_edge <= total_fees:
                import warnings
                warnings.warn(
                    f"min_arb_edge ({self.min_arb_edge}) <= total fees "
                    f"({self.polymarket_fee_pct} + {self.kalshi_fee_pct} = {total_fees}). "
                    "Arb trades may have zero or negative profit.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


_SECRET_FIELDS = frozenset({
    "telegram_bot_token", "discord_webhook_url", "slack_webhook_url",
    "email_smtp_password", "kalshi_api_key_id", "kalshi_private_key_path",
    "telegram_kill_token", "discord_kill_token",
    "slack_kill_bot_token", "slack_kill_app_token",
    "sentry_dsn",
    "fred_api_key", "coingecko_api_key", "congress_api_key",
    "courtlistener_api_key", "openfda_api_key",
    "reddit_client_id", "reddit_client_secret", "pubmed_api_key",
    "metaculus_api_key",
    "sports_odds_api_key", "sports_stats_api_key",
    "acled_api_key",
})


class BotConfig(BaseModel):
    scanning: ScanningConfig = Field(default_factory=ScanningConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    microstructure: MicrostructureConfig = Field(default_factory=MicrostructureConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    wallet_scanner: WalletScannerConfig = Field(default_factory=WalletScannerConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    model_tiers: ModelTierConfig = Field(default_factory=ModelTierConfig)
    circuit_breakers: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    specialists: SpecialistsConfig = Field(default_factory=SpecialistsConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    continuous_learning: ContinuousLearningConfig = Field(default_factory=ContinuousLearningConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    analyst: AnalystConfig = Field(default_factory=AnalystConfig)
    uma: UMAConfig = Field(default_factory=UMAConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)

    def redacted_dict(self) -> dict[str, Any]:
        """Return config dict with secret values masked."""
        raw = self.model_dump()
        _redact_secrets(raw)
        return raw


def _redact_secrets(d: dict[str, Any]) -> None:
    """Recursively mask secret fields in a dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            _redact_secrets(v)
        elif k in _SECRET_FIELDS and isinstance(v, str) and v:
            d[k] = v[:3] + "***"


# Env var prefix → config field mapping for overrides
_ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    # ENV_VAR_NAME: (section, field, cast_type)
    "BOT_BANKROLL": ("risk", "bankroll", float),
    "BOT_MAX_STAKE": ("risk", "max_stake_per_market", float),
    "BOT_MAX_DAILY_LOSS": ("risk", "max_daily_loss", float),
    "BOT_KELLY_FRACTION": ("risk", "kelly_fraction", float),
    "BOT_DRY_RUN": ("execution", "dry_run", lambda v: v.lower() in ("true", "1", "yes")),
    "BOT_LOG_LEVEL": ("observability", "log_level", str),
    "BOT_SCAN_INTERVAL": ("engine", "scan_interval_minutes", int),
    "BOT_CYCLE_INTERVAL": ("engine", "cycle_interval_secs", int),
    # SMTP / email alerts
    "SMTP_HOST": ("alerts", "email_smtp_host", str),
    "SMTP_PORT": ("alerts", "email_smtp_port", int),
    "SMTP_USER": ("alerts", "email_smtp_user", str),
    "SMTP_PASS": ("alerts", "email_smtp_password", str),
    "ALERT_EMAIL_FROM": ("alerts", "email_from", str),
    "ALERT_EMAIL_TO": ("alerts", "email_to", str),
    "OPENFDA_API_KEY": ("research", "openfda_api_key", str),
    "REDDIT_CLIENT_ID": ("research", "reddit_client_id", str),
    "REDDIT_CLIENT_SECRET": ("research", "reddit_client_secret", str),
    "NCBI_API_KEY": ("research", "pubmed_api_key", str),
    "METACULUS_API_KEY": ("research", "metaculus_api_key", str),
    "ODDS_API_KEY": ("research", "sports_odds_api_key", str),
    "API_FOOTBALL_KEY": ("research", "sports_stats_api_key", str),
    "ACLED_API_KEY": ("research", "acled_api_key", str),
}


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """Apply environment variable overrides to raw config dict."""
    for env_var, (section, field_name, cast) in _ENV_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val is not None:
            raw.setdefault(section, {})
            raw[section][field_name] = cast(val)  # type: ignore[operator]


def load_config(path: str | Path | None = None) -> BotConfig:
    """Load config from YAML file with env var overrides, falling back to defaults."""
    if path is None:
        path = _PROJECT_ROOT / "config.yaml"
    path = Path(path)
    if path.exists():
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        raw = {}
    _apply_env_overrides(raw)
    return BotConfig(**raw)


def is_live_trading_enabled() -> bool:
    """Check if live trading is explicitly enabled via env var."""
    return os.environ.get("ENABLE_LIVE_TRADING", "").lower() == "true"


class ConfigWatcher:
    """Watch config file for changes and hot-reload."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
        self._last_mtime: float = 0.0
        self._config: BotConfig = load_config(self._path)
        self._callbacks: List[Callable[[BotConfig], None]] = []
        self._update_mtime()

    def _update_mtime(self) -> None:
        if self._path.exists():
            self._last_mtime = self._path.stat().st_mtime

    @property
    def config(self) -> BotConfig:
        return self._config

    def on_change(self, callback: Callable[[BotConfig], None]) -> None:
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def check_and_reload(self) -> bool:
        """Check if config file changed and reload if so. Returns True if reloaded."""
        if not self._path.exists():
            return False
        current_mtime = self._path.stat().st_mtime
        if current_mtime > self._last_mtime:
            try:
                new_config = load_config(self._path)
                self._config = new_config
                self._last_mtime = current_mtime
                for cb in self._callbacks:
                    cb(new_config)
                return True
            except Exception:
                return False
        return False
