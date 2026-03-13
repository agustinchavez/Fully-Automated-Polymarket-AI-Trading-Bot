**POLYMARKET AI TRADING BOT**

Technical Implementation Spec

Incremental Improvement Plan for Profitability

Based on: dylanpersonguy/Fully-Autonomous-Polymarket-AI-Trading-Bot

Version 1.1 | March 2026

**How to use this document**

Each phase is self-contained and testable. Implement phases in order-each builds on the previous.

Within each phase, follow the numbered steps sequentially.

Every phase includes acceptance criteria so you know when it's done.

Code references use the existing src/ directory structure from the base repo.

Estimated effort assumes a single developer or AI coding agent (Claude Code, Cursor, etc.).

---

# **Reviewer Notes (v1.1)**

The following changes were made based on a code-level review of the current implementation:

**Reordered implementation priority.** The original Phase 0-9 sequential ordering is suboptimal. Some high-impact, low-effort items from later phases should be pulled forward, and some Phase 0 items are premature. The recommended implementation order is:

1. **Phase 0A (NEW): Cost Control & Budget Management** — ~2 days. Without cost caps, the bot can burn through API budgets silently. This is the first thing to implement.
2. **Phase 3.3: Capital-efficiency-weighted market selection** — ~2 hours. A simple annualized edge filter that immediately removes capital traps.
3. **Phase 2.5: Model upgrade** — ~5 minutes. A config.yaml one-liner, not a phase item.
4. **Phase 0.3: Configuration validation** — ~1 day. Prevents silent bad-config bugs.
5. **Phase 0.2: Error budget tracking** — ~1-2 days. Circuit breakers for degraded components.
6. **Phase 1: Backtesting engine** — 5-7 days. Unlocks validation for everything after it.
7. **Phase 2.1-2.4: Structured forecasting** — validated via backtesting.
8. **Phase 3.1-3.2, 3.4: Edge & risk upgrades** — validated via backtesting.
9. Everything else based on backtest results.

**Phase 0.1 (PostgreSQL) deprioritized.** SQLite with WAL mode handles single-process workloads well. PostgreSQL adds operational complexity (backups, connection management, hosting costs) without benefit until multi-instance deployment. Moved to Phase 5+.

**New Phase 0A added.** The original spec has no cost management. The existing CostTracker in `src/observability/metrics.py` tracks spend but cannot enforce limits. With 3 LLMs running at 480 cycles/day, costs can silently escalate. A budget cap system is critical infrastructure.

**Appendix B expanded.** The original cost estimates ($375-745/month) are based on rough per-call averages. The actual cost depends heavily on token counts, model selection, and caching behavior. The expanded appendix includes per-token cost modeling, a model tiering strategy, and a concrete budget cap implementation spec.

---

# **Phase 0A: Cost Control & Budget Management**

**Effort:** 2 days **Priority:** CRITICAL **Risk if skipped:** Uncontrolled API spend that exceeds trading profits

The bot currently tracks costs via `CostTracker` in `src/observability/metrics.py` but has no ability to enforce limits. With 3 LLMs queried in parallel across up to 480 cycles/day, a single misconfiguration (e.g., reducing `research_cooldown_minutes` from 60 to 5) can 12x daily spend overnight. Cost management must be built before scaling up any other functionality.

## **0A.1 - Implement a global daily budget cap**

The existing `CostTracker` records costs per cycle and cumulatively but never stops the bot from spending more. Add enforcement.

- Add config fields to `config.yaml` under a new `cost_management` section:
  ```yaml
  cost_management:
    daily_budget_usd: 5.00          # Hard cap — bot stops forecasting when hit
    warning_threshold_pct: 0.70     # Alert at 70% of daily budget
    cycle_budget_usd: 0.50          # Max spend per single cycle
    budget_reset_hour_utc: 0        # When the daily counter resets
    track_token_usage: true         # Log actual token counts, not estimates
  ```
- Modify `CostTracker` to support `check_budget() -> bool` and `remaining_daily_budget() -> float`. When `check_budget()` returns False, the engine loop must skip the forecasting phase entirely (market scanning and monitoring continue, but no LLM/search API calls are made).
- Add a `cost_management.budget_exhausted` structured log event and a dashboard alert when the cap is hit.
- On budget exhaustion: continue monitoring existing positions (P&L updates, stop-loss checks), but do NOT open new positions or run new research/forecasts until the next reset.

## **0A.2 - Implement model tiering for cost efficiency**

Not every market needs 3 frontier-tier LLMs. Route markets to different cost tiers based on expected edge and stake size.

- Define three cost tiers in config:
  ```yaml
  cost_management:
    model_tiers:
      scout:        # Low-cost screening — is this market worth deeper analysis?
        models: [gpt-4o-mini]
        max_cost_per_market: 0.005
        use_for: initial_screen
      standard:     # Normal ensemble for most markets
        models: [gpt-4o-mini, gemini-2.0-flash]
        max_cost_per_market: 0.02
        use_for: default
      premium:      # Full ensemble for high-edge opportunities
        models: [gpt-4o, claude-sonnet-4-5-20250929, gemini-2.0-flash]
        max_cost_per_market: 0.05
        use_for: high_edge
  ```
- Create `src/forecast/model_router.py` that selects the tier:
  - **Scout tier**: Used for initial market screening. Before running full research + ensemble on all 10 markets per cycle, run a single cheap LLM call (gpt-4o-mini, ~$0.0003/call) with a minimal prompt: market question + current price + basic stats. If the scout model estimates <3% edge, skip the market entirely. This filters out 60-70% of markets before expensive research begins.
  - **Standard tier**: Markets that pass the scout screen. Uses 2 models (gpt-4o-mini + gemini-2.0-flash) — 95% cheaper than the current 3-model frontier ensemble while maintaining ensemble diversity.
  - **Premium tier**: Markets where the standard tier detects >6% edge AND the potential stake exceeds $20. Escalates to the full 3-model frontier ensemble for maximum accuracy on high-value trades.
- The tier decision is logged in the forecast record for later analysis of tier accuracy vs. cost.

## **0A.3 - Use actual token counts instead of flat-rate estimates**

The current `_DEFAULT_COSTS` in `metrics.py` uses flat per-call estimates ($0.005 for GPT-4o). Real costs vary 10x depending on prompt size (evidence-heavy prompts can hit 3,000+ tokens).

- For OpenAI: read `resp.usage.prompt_tokens` and `resp.usage.completion_tokens` from the API response. Compute actual cost using current pricing: GPT-4o = $2.50/1M input + $10.00/1M output; GPT-4o-mini = $0.15/1M input + $0.60/1M output.
- For Anthropic: read `resp.usage.input_tokens` and `resp.usage.output_tokens`. Claude Sonnet 4.5 = $3.00/1M input + $15.00/1M output.
- For Google: use `resp.usage_metadata.prompt_token_count` and `resp.usage_metadata.candidates_token_count`. Gemini 2.0 Flash = $0.10/1M input + $0.40/1M output (free tier covers ~1,500 requests/day).
- Update `CostTracker.record_call()` to accept token counts and compute real costs. Keep the flat-rate estimates as fallback when token data is unavailable.
- Log per-call actual cost in structured logs. This data feeds the budget cap enforcement and the cost dashboard.

## **0A.4 - Add a cost dashboard widget**

- Add a "Cost Management" card to the Dashboard Overview tab showing: today's spend vs. budget (progress bar), cost breakdown by provider (pie chart), cost per trade (average over last 24h), projected monthly cost at current burn rate, and budget remaining.
- Add a cost history chart to the Analytics tab: daily cost over last 30 days, overlaid with daily P&L. This makes it immediately obvious whether the bot is spending more than it earns.

## **0A.5 - Use cheap models for evidence extraction**

The evidence extractor in `src/research/evidence_extractor.py` currently uses `gpt-4o` for structured extraction (pulling bullets, contradictions, quality scores from web sources). This is a structured-output task, not a reasoning task — it doesn't need a frontier model.

- Switch evidence extraction to `gpt-4o-mini` (200x cheaper per token than GPT-4o for output). Evidence extraction prompts are ~2,000 input tokens + ~500 output tokens. Cost drops from ~$0.005/call to ~$0.0006/call.
- Validate quality: run both models on 100 cached evidence packages and compare extraction quality. If gpt-4o-mini misses >5% of facts that gpt-4o catches, keep gpt-4o for evidence extraction. Based on benchmarks, gpt-4o-mini performs within 2-3% on structured extraction tasks.

**Acceptance Criteria - Phase 0A**

✅ Daily budget cap halts forecasting when hit, while position monitoring continues

✅ Model tiering correctly routes 60%+ of markets to scout/standard tiers, saving >50% on LLM costs

✅ Token-based cost tracking is within 5% of actual API invoices

✅ Cost dashboard shows real-time spend vs. budget with provider breakdown

✅ Evidence extraction on gpt-4o-mini produces quality within 5% of gpt-4o baseline

---

# **Phase 0: Foundation & Infrastructure Hardening**

**Effort:** 1-2 days **Priority:** HIGH **Risk if skipped:** Silent bad-config bugs, no circuit breakers

> **Note (v1.1):** Phase 0.1 (PostgreSQL) has been moved to Phase 5+. SQLite with WAL mode is sufficient for single-process paper trading and early live trading. PostgreSQL becomes necessary only for multi-instance deployment or sustained high-frequency trading (>1,000 trades/day). The remaining items in this phase (0.2 and 0.3) are still important and should be implemented before backtesting.

Before changing any trading logic, harden the infrastructure so that every subsequent phase has reliable data and safe failure modes.

## **0.1 - Upgrade storage from SQLite to PostgreSQL (DEFERRED — see Phase 5+)**

> **Deprioritized (v1.1).** SQLite with WAL mode handles single-process workloads at the bot's current throughput (~10 markets/cycle, 480 cycles/day). PostgreSQL adds operational overhead (installation, backups, connection pooling, hosting cost) without tangible benefit until: (a) running multiple bot instances against the same DB, or (b) sustained write throughput exceeds ~100 writes/second. Revisit when approaching live trading at scale.

Original spec retained for when this becomes relevant:

- Install PostgreSQL 16 and create a dedicated database (polybot) with a restricted user.
- Create src/storage/pg_adapter.py implementing the same interface as the existing SQLite storage module. Use asyncpg or psycopg3 for async support.
- Migrate the 10 existing SQLite schema migrations to Alembic migration files. Keep SQLite as a fallback option via config flag.
- Add connection pooling (min 5, max 20 connections) and a health check query in the /ready endpoint.
- Add automated daily pg_dump backups to a configurable path with 30-day rotation.

## **0.2 - Add structured logging & error budget tracking**

The existing structlog setup is good. Extend it with error budget tracking so you know when the system is degrading.

- Add an error_budget table tracking: component, error_count, total_count, window_start. Components: forecaster, research, execution, whale_scanner.
- Implement a circuit breaker per component: if error rate exceeds 20% over a 1-hour window, that component auto-disables and alerts. Re-enable after 15 minutes of clean operation.
- Add request/response latency tracking for all external API calls (LLM, search, Polymarket). Log p50/p95/p99 per cycle.

## **0.3 - Add a configuration validation layer**

Current config.yaml has no schema validation. Bad config should fail fast at startup, not silently cause bad trades.

- Create src/config/schema.py using Pydantic v2 models that mirror every config.yaml section.
- Add cross-field validation rules: max_stake_per_market must be &lt;= bankroll \* 0.1, min_edge must be &gt; estimated round-trip fee (currently ~2%), kelly_fraction must be between 0.05 and 0.5.
- Validate on startup. Reject invalid configs with clear error messages.

**Acceptance Criteria - Phase 0**

✅ Circuit breaker triggers correctly when a mock API returns 50% errors

✅ Invalid config.yaml values cause a clear startup failure, not silent runtime bugs

✅ Latency tracking logs p50/p95/p99 for all external API calls per cycle

# **Phase 1: Historical Backtesting Engine**

**Effort:** 5-7 days **Priority:** CRITICAL **Risk if skipped:** Flying blind-no way to validate changes before risking money

This is the single most important missing piece. Without backtesting, every subsequent improvement is a guess. Build a replay system that can run the full pipeline against historical Polymarket data.

## **1.1 - Historical data scraper**

- Create src/backtest/data_scraper.py that fetches resolved markets from Polymarket's Gamma API. Target fields: condition_id, question, description, outcome, resolution_date, category, final_price_history (OHLCV).
- Store in a historical_markets table with columns: condition_id, question, description, category, resolution (YES/NO), resolved_at, created_at, price_history_json, volume_usd, liquidity_usd.
- Backfill at least 6 months of resolved markets. Aim for 5,000+ markets minimum. Prioritize markets with >\$10K volume (these are the ones worth trading).
- Add a daily cron job that ingests newly resolved markets.

## **1.2 - LLM response cache for cost-efficient replay**

Running 3 LLMs across 5,000 markets would cost thousands of dollars. Build a deterministic cache.

- Create a forecast_cache table: market_id, model_name, prompt_hash (SHA-256 of the full prompt), response_json, created_at.
- Modify src/forecast/llm_forecaster.py to check the cache before making an API call. Cache hits return instantly at zero cost.
- For initial backfill, run forecasts on a random sample of 500 markets to seed the cache. This gives you enough data for statistical significance without excessive cost (~\$50-100 in API fees).

## **1.3 - Replay engine**

- Create src/backtest/replay_engine.py that accepts a date range and config override, then replays the full pipeline: market discovery (from DB) → classification → filtering → research (cached) → forecasting (cached) → calibration → risk checks → position sizing → simulated execution.
- Simulated execution must model realistic fills: use the historical orderbook midpoint +/- configurable slippage (default 0.5%), enforce minimum liquidity from historical data, and apply Polymarket's fee schedule.
- Output a BacktestResult object containing: total_pnl, win_rate, sharpe_ratio, max_drawdown, brier_score, trades_list (with per-trade detail), equity_curve (daily snapshots).
- Add a CLI command: bot backtest --start 2025-09-01 --end 2026-02-01 --config config.yaml

## **1.4 - Backtesting dashboard tab**

- Add a 10th dashboard tab (Backtest) that displays: equity curve chart, trade log table, performance metrics cards (same KPIs as Analytics tab), and a config diff showing what parameters were tested.
- Add an A/B comparison mode: run two configs side-by-side and display performance differences with statistical significance (paired t-test on daily returns).

**Acceptance Criteria - Phase 1**

✅ 5,000+ resolved historical markets in database

✅ Replay engine produces a full BacktestResult for a 3-month window in under 10 minutes

✅ Backtest P&L correlates within 20% of forward paper trading results over the same period

✅ A/B config comparison shows statistically significant differences when testing min_edge=4% vs 8%

# **Phase 2: Structured Forecasting Pipeline**

**Effort:** 5-7 days **Priority:** HIGH **Expected impact:** 15-30% improvement in Brier score

The base repo treats LLMs as black-box probability estimators: feed in search results, get a number. This phase restructures the forecasting to use superforecasting methodology, which dramatically improves calibration.

## **2.1 - Question decomposition module**

Instead of asking the LLM one monolithic question, break it into sub-forecasts that are individually easier to estimate.

- Create src/forecast/decomposer.py. For each market question, prompt a **cheap model** (gpt-4o-mini at ~$0.0003/call, or claude-haiku at ~$0.0002/call) to generate 3-5 sub-questions whose joint probability implies the main question. Example: 'Will the Fed cut rates in June?' decomposes into: (a) Will inflation be below 3% by May? (b) Will unemployment exceed 4.5%? (c) Will there be a financial stability event? (d) Will the Fed signal a cut in the May statement?
- **Cost note:** Decomposition is a structural task (breaking a question into parts), not a reasoning task. It does not require frontier models. Using gpt-4o-mini here costs <$0.15/day even at 480 cycles.
- Each sub-question gets its own research + forecast cycle. The main probability is computed as a weighted combination of sub-probabilities.
- **Cost warning:** If each sub-question triggers its own full research + 3-model ensemble cycle, decomposition multiplies total cost by 3-5x. Mitigate by: (a) sharing the parent market's research across sub-questions (don't re-search), (b) using only the standard tier (2 models) for sub-question forecasts, (c) capping at 3 sub-questions per market.
- Cache decompositions per market_id so they're only generated once.

## **2.2 - Base rate anchoring**

LLMs are notoriously bad at base rates. Force them to anchor on historical frequency data before adjusting.

- Create src/forecast/base_rates.py with a lookup table of base rates by category. Examples: 'Fed cuts rates at any given meeting' → historical frequency ~25%; 'Incumbent wins presidential election' → ~60%; 'Major tech company meets earnings estimates' → ~70%.
- For markets that match a known base rate pattern, inject the base rate into the LLM prompt: 'Historical base rate for this type of event is X%. Adjust from this anchor based on the specific evidence.'
- Seed the base rate table with 50+ common patterns. Allow it to self-update: after 30+ resolved markets in a category, compute the empirical base rate from historical data and update the table.

## **2.3 - Dynamic ensemble weighting**

Replace static weights (40/35/25) with per-category adaptive weights based on actual model performance.

- Create src/forecast/adaptive_weights.py. Track each model's Brier score per market category (11 categories) using an exponentially weighted moving average (decay factor 0.95).
- After each market resolution, update the per-model, per-category Brier score and recompute weights. Weight formula: w_i = (1 / brier_i) / sum(1 / brier_j) for all models j.
- Cold start: use the existing static weights until 20+ resolved markets per category. Then switch to adaptive.
- Add a dashboard widget showing per-model, per-category weights and Brier scores over time.

## **2.4 - Prompt engineering overhaul**

- Restructure the forecasting prompt to enforce a reasoning chain. Require the LLM to output: (1) base rate, (2) evidence for, (3) evidence against, (4) confidence-weighted adjustment, (5) final probability. Parse this structured output rather than just extracting a number.
- Add an explicit debiasing instruction: 'Do NOT anchor to the current market price. Estimate the probability as if you had no knowledge of what the market currently prices this at.'
- Require the LLM to rate its own confidence (HIGH/MEDIUM/LOW) and flag when evidence is insufficient. LOW confidence forecasts should be penalized more aggressively in calibration.

## **2.5 - Model upgrade**

> **Note (v1.1):** Swapping model versions is a config change, not an engineering task. Update `config.yaml` and the `_DEFAULT_COSTS` map in `metrics.py`:

- Replace `claude-3-5-sonnet-20241022` with `claude-sonnet-4-5-20250929` in `config.yaml` → `ensemble.models`. Update cost entry in `_DEFAULT_COSTS`. No code changes required — the existing `_route_model()` function in `ensemble.py` routes any model with "claude" in the name to the Anthropic provider.
- Consider `gemini-2.0-flash` as a replacement for `gemini-1.5-pro` in the standard tier — it's faster, cheaper ($0.10/1M input vs. $1.25/1M), and competitive on structured output tasks.
- A 4th model slot (e.g., DeepSeek-R1 via OpenAI-compatible endpoint, or a self-hosted Llama model) can increase ensemble diversity. Add a new provider branch in `_route_model()` if the model uses a non-standard API. Only add a 4th model to the **premium tier** to avoid cost multiplication.
- **Cost impact:** Upgrading to Claude Sonnet 4.5 increases per-call cost marginally ($3.00 vs. $3.00/1M input — same price as 3.5 Sonnet). Switching Gemini to 2.0 Flash *reduces* cost by ~12x.

**Acceptance Criteria - Phase 2**

✅ Question decomposition runs on >80% of markets (some may not decompose well)

✅ Brier score improves by >10% on backtested data vs. the monolithic prompt baseline

✅ Adaptive weights diverge meaningfully across categories (not all models weighted equally)

✅ Structured prompt output parses cleanly with <2% parse failures

# **Phase 3: Edge Estimation & Risk Model Upgrade**

**Effort:** 4-5 days **Priority:** HIGH **Expected impact:** Fewer false-positive trades, better capital allocation

The current bot trades when it estimates a 4% edge. But a 4% edge with high uncertainty is very different from a 4% edge with high confidence. This phase adds uncertainty quantification and smarter position sizing.

## **3.1 - Edge confidence intervals**

- Create src/policy/edge_uncertainty.py. For each forecast, compute: (a) ensemble spread (max - min model probability), (b) evidence quality score (already exists), (c) base rate distance (how far the forecast is from the base rate), (d) decomposition agreement (how consistent sub-question forecasts are).
- Combine these into an edge_uncertainty_score (0-1, where 1 = maximum uncertainty). Formula: uncertainty = 0.3 \* ensemble_spread + 0.25 \* (1 - evidence_quality) + 0.25 \* base_rate_distance + 0.2 \* decomposition_disagreement.
- Adjust the effective edge: effective_edge = raw_edge \* (1 - uncertainty \* 0.5). This means a 6% raw edge with 80% uncertainty becomes a 3.6% effective edge-below the 4% threshold, so the trade is blocked.

## **3.2 - Raise minimum edge with fee-awareness**

- Calculate true round-trip cost per trade: Polymarket CLOB maker/taker fees + gas (typically ~1.5-2.5% combined). Make this a config parameter: execution.estimated_round_trip_fee_pct.
- Change the min_edge check to: raw_edge > min_edge + estimated_round_trip_fee_pct. With min_edge=4% and fees=2%, the bot only trades when raw edge exceeds 6%.
- Backtest this change. You will likely see: fewer trades, but higher win rate and better P&L. The default 4% threshold is probably generating many noise trades.

## **3.3 - Capital-efficiency-weighted market selection**

A 5% edge on a market resolving in 3 days is far more valuable than the same edge on a market resolving in 90 days.

- Create src/policy/capital_efficiency.py. Compute annualized_edge = edge / (days_to_resolution / 365). A 5% edge resolving in 7 days = 260% annualized. A 5% edge resolving in 90 days = 20% annualized.
- Add a configurable minimum annualized edge (suggested default: 50%). This filters out low-edge, long-duration capital traps.
- Use annualized_edge as a ranking input when the bot must choose between multiple qualifying markets in the same cycle.

## **3.4 - Event correlation modeling**

Current approach uses blunt category caps (35% per category). Real correlation is event-level.

- Create src/policy/correlation.py. Use the Polymarket Gamma API's event grouping-markets within the same event (e.g., 'US Presidential Election 2028') are flagged as correlated.
- For markets in the same event, compute an implied correlation score. Two markets that both resolve based on the same underlying outcome (e.g., 'Trump wins' and 'Republicans win Senate') get correlation = 0.8. Markets in the same event but different outcomes get correlation = 0.3.
- Replace the category exposure cap with a correlation-aware portfolio VaR calculation: before placing a new trade, estimate how much the portfolio VaR increases. Block the trade if VaR exceeds the daily loss limit.

**Acceptance Criteria - Phase 3**

✅ Backtested trade count drops 30-50% while P&L improves (fewer but better trades)

✅ No single event has more than 30% of total portfolio exposure

✅ Annualized edge ranking correctly prioritizes short-duration markets

✅ Edge uncertainty score correlates negatively with trade win rate (higher uncertainty = lower win rate)

# **Phase 4: Domain-Specific Forecasting Models**

**Effort:** 7-10 days **Priority:** HIGH **Expected impact:** Genuine alpha in specific market categories

This is the highest-leverage phase for profitability. Instead of using a general LLM for all markets, plug in domain-specific models where real quantitative edges exist. An LLM reading search results cannot outpredict a weather ensemble model on weather markets, or a polling aggregator on election markets.

## **4.1 - Weather market specialist**

Weather prediction markets are the lowest-hanging fruit because physics-based ensemble models provide genuinely superior probability estimates that the market often underprices.

- Create src/forecast/specialists/weather.py. Integrate with Open-Meteo's free API to fetch 31-member GFS ensemble forecasts for target cities.
- For temperature threshold markets (e.g., 'Will NYC high exceed 75°F on June 15?'): count the fraction of ensemble members exceeding the threshold. That fraction IS your probability estimate (e.g., 27/31 = 87%).
- Compare ensemble probability to market price. Trade when |ensemble_prob - market_prob| > 8% (weather markets have higher noise, so use a wider threshold).
- This specialist completely bypasses the LLM pipeline. It's a direct: weather API → probability → edge check → trade.

## **4.2 - Crypto price market specialist**

BTC/ETH 5-minute and 15-minute up/down markets are high-frequency and dominated by technical analysis, not LLM reasoning.

- Create src/forecast/specialists/crypto_ta.py. Fetch real-time 1-minute candles from Binance/Coinbase via their free APIs.
- Compute a composite signal from: RSI (14-period), VWAP deviation, SMA crossover (9/21), momentum (rate of change), and market skew (bid/ask imbalance from the prediction market orderbook).
- Use a simple logistic regression trained on historical 5-min outcomes to convert the composite signal into a probability. Retrain weekly on the latest 2,000 candles.
- Time entries to T-15 seconds before the window closes for 5-minute markets. At this point, the price direction is largely locked in but the market may not have fully priced it.

## **4.3 - Election/politics specialist**

- Create src/forecast/specialists/politics.py. Integrate with polling aggregator APIs: FiveThirtyEight (or Silver Bulletin), RealClearPolitics, and 270toWin for US elections.
- For election markets, use a weighted polling average as the base probability. Adjust based on: time until election (polls become more predictive closer to election day), historical polling error distributions, and fundamentals (incumbency, economy).
- This specialist augments rather than replaces the LLM pipeline: it provides a strong base rate that the LLM then adjusts with current news/events.

## **4.4 - Specialist router**

- Create src/forecast/specialist_router.py. After market classification (existing regex classifier), route to the appropriate specialist if one exists. If no specialist matches, fall back to the general LLM ensemble pipeline.
- Specialist output format must match the general pipeline output: probability, confidence, evidence_quality, reasoning. This ensures the downstream risk checks and position sizing work identically.
- Add a config section: specialists.enabled: \[weather, crypto_ta, politics\]. Allow toggling individual specialists on/off.

**Acceptance Criteria - Phase 4**

✅ Weather specialist Brier score is <0.15 on backtested temperature markets

✅ Crypto TA specialist achieves >55% win rate on 5-minute BTC markets over 1,000+ trades

✅ Specialist router correctly routes >95% of markets to the right pipeline

✅ Each specialist can be independently disabled without affecting other pipelines

# **Phase 5: Cross-Platform Arbitrage**

**Effort:** 5-7 days **Priority:** MEDIUM-HIGH **Expected impact:** Low-risk, consistent small profits independent of forecasting accuracy

Arbitrage profits come from structural market inefficiencies rather than being smarter than the crowd. This is the most reliable source of profit if you can find the opportunities.

## **5.1 - Kalshi connector**

- Create src/connectors/kalshi_client.py implementing: market discovery (list active markets), price fetching (bid/ask/mid), order placement, and position management.
- Kalshi uses REST API with RSA-PSS authentication. Implement the signing flow per their API docs.
- Map Kalshi market identifiers to Polymarket condition_ids for overlapping markets. Many weather, economic, and political markets exist on both platforms.

## **5.2 - Arbitrage scanner**

- Create src/policy/cross_platform_arb.py. Every 60 seconds, fetch prices for all overlapping markets on both platforms.
- Compute the arbitrage spread: if Polymarket prices YES at \$0.65 and Kalshi prices the same event's YES at \$0.72, there's a 7% spread. Account for fees on both sides (~2% per platform round-trip = ~4% total).
- When spread > (total_fees + min_arb_edge), simultaneously buy YES on the cheaper platform and sell YES (or buy NO) on the expensive platform. Log both legs as a paired trade.
- Add arb-specific risk limits: max_arb_position_usd, max_arb_positions_count, and execution_timeout_seconds (if both legs don't fill within timeout, unwind).

## **5.3 - Intra-Polymarket arbitrage for complementary markets**

The existing bot has arbitrage detection but it's unclear how aggressively it's used. Expand it.

- Scan for complementary markets (YES + NO on the same event should sum to ~\$1.00). When they sum to <\$0.97, buy both sides for a guaranteed ~3% profit at resolution.
- Scan for correlated event mispricings: if 'Biden wins' is priced at 40% but 'Democrat wins' is priced at 55%, there's a 15% implied probability of a non-Biden Democrat winning. If that's implausible, one market is mispriced.

**Acceptance Criteria - Phase 5**

✅ Kalshi connector can list markets, fetch prices, and place paper orders

✅ Arb scanner identifies >5 opportunities per day (even if most are sub-threshold)

✅ Complementary market checker catches YES+NO sums deviating by >2%

✅ Paired arb trades are logged as a unit with combined P&L tracking

# **Phase 6: Execution Quality & Market Microstructure**

**Effort:** 3-4 days **Priority:** MEDIUM **Expected impact:** 1-3% P&L improvement from reduced slippage and better entry timing

## **6.1 - Realistic fill simulator for backtesting**

- Upgrade the backtest execution simulator. Current approach likely assumes fills at midpoint. Instead, model: partial fills based on historical orderbook depth, price impact proportional to order size / available liquidity, and random fill delays (50-500ms) with price movement during delay.
- Calibrate the simulator against actual paper trading fills. Collect 200+ paper trades and compare simulated vs. actual fill price, fill rate, and slippage.

## **6.2 - Smart entry timing**

- Upgrade the existing smart_entry module in src/analytics/. Instead of entering immediately when risk checks pass, implement a patience window: monitor the orderbook for up to 5 minutes after a trade signal, looking for a favorable price dip.
- Define entry conditions: enter immediately if edge > 2 \* min_edge (strong signal, don't wait); wait up to 5 minutes if edge is near threshold; cancel if edge deteriorates below min_edge during the wait.

## **6.3 - Execution analytics**

- Track per-strategy execution quality metrics: average slippage vs midpoint, fill rate (% of orders fully filled), time-to-fill distribution, realized spread vs. expected. Display these in the Trading dashboard tab.
- Use these metrics to auto-select execution strategy: if a market has thin liquidity (book depth &lt; \$5K), use Iceberg. If normal liquidity, use Simple. If large order (&gt;10% of book depth), use TWAP.

**Acceptance Criteria - Phase 6**

✅ Backtested P&L with realistic fill simulation is within 5% of paper trading results

✅ Smart entry timing saves >0.5% average per trade vs. immediate entry baseline

✅ Execution strategy auto-selection is logging correct strategy choices for >90% of trades

# **Phase 7: Enhanced Whale Intelligence**

**Effort:** 3-4 days **Priority:** MEDIUM **Expected impact:** Better signal extraction from smart money, filtered from noise traders

## **7.1 - Whale quality scoring**

- Not all whales are smart money. Create src/analytics/whale_scorer.py that scores each tracked wallet on: historical ROI (trailing 90 days), calibration quality (are their entry prices consistently before favorable moves?), market category specialization (some whales are only good at politics), and consistency (Sharpe ratio of their trades, not just total P&L).
- Only use conviction signals from whales scoring above the 60th percentile. Ignore noise from wallets that are large but uncalibrated.

## **7.2 - Whale timing analysis**

- Track not just what whales buy, but when they buy relative to market movement. A whale entering a position 2 hours before a big price move is much more informative than one entering after the move.
- Compute a whale_timing_score for each wallet: percentage of their entries that precede favorable 24-hour price moves. Use this as a quality signal in whale_scorer.py.

## **7.3 - Whale consensus threshold tuning**

- The current settings use min_whale_count=1 and conviction_edge_boost=0.08 (+8%). This is likely too aggressive-a single whale's position shouldn't boost edge by 8%.
- Backtest different configurations. Suggested starting point: min_whale_count=3, conviction_edge_boost=0.04, conviction_edge_penalty=0.03. More whales required, smaller per-signal effect.

**Acceptance Criteria - Phase 7**

✅ Whale quality scoring correlates positively with actual subsequent trade profitability

✅ Filtered whale signals (top 40% wallets only) outperform unfiltered signals on backtest

✅ Whale timing score identifies at least 10 wallets that consistently front-run price moves

# **Phase 8: Continuous Learning & Self-Improvement**

**Effort:** 4-5 days **Priority:** MEDIUM **Expected impact:** Compound improvement-the bot gets better over time rather than degrading

## **8.1 - Post-resolution analysis pipeline**

- Create src/analytics/post_mortem.py. When a market resolves, automatically analyze: (a) was the bot's forecast correct? (b) which model was closest? (c) which evidence sources were most predictive? (d) was the position size appropriate given the actual outcome?
- Store post-mortem results in a trade_analysis table. Flag cases where the bot was wrong AND confident (these are the most important to learn from).
- Generate a weekly summary: top 3 winning categories, top 3 losing categories, most/least accurate model, evidence sources that were most/least useful.

## **8.2 - Automatic strategy parameter tuning**

- Create src/analytics/param_optimizer.py. Every week, automatically run 20-50 backtest configurations using the latest 30 days of data with randomized parameter perturbations around the current config.
- Parameters to optimize: min_edge, kelly_fraction, min_evidence_quality, stop_loss_pct, take_profit_pct, max_stake_per_market. Vary each by ±20% around current value.
- If a parameter set improves Sharpe ratio by >10% with statistical significance (p < 0.05), suggest the change in the dashboard with a one-click apply button. Do NOT auto-apply-human approval required.

## **8.3 - Evidence source quality tracking**

- Track which search domains/sources correlate with correct forecasts. After 100+ resolved markets, rank domains by: frequency of citation, correlation with correct forecasts, and average evidence quality score when cited.
- Automatically upweight high-quality sources and downweight or blocklist consistently misleading ones. Integrate this into the domain authority scoring in the research engine.

## **8.4 - Calibration retraining automation**

- The existing calibration retrain triggers at 30 resolved markets. Make this smarter: retrain whenever (a) 30 new resolutions, OR (b) Brier score degrades by >10% over a 7-day rolling window, OR (c) a new specialist is enabled (which changes the forecast distribution).
- Add A/B testing for calibration: hold out 20% of markets and compare calibrated vs. uncalibrated forecasts. If calibration is hurting performance, auto-disable and alert.

**Acceptance Criteria - Phase 8**

✅ Post-mortem analysis runs automatically for >95% of resolved markets

✅ Weekly summary generates actionable insights (not just raw numbers)

✅ Parameter optimizer finds at least 1 statistically significant improvement per month

✅ Evidence source rankings stabilize after 200+ resolutions and match intuitive quality (e.g., AP > Reddit)

# **Phase 9: Production Deployment & Live Trading**

**Effort:** 3-4 days **Priority:** Final phase **Prerequisite:** Phases 0A, 0, 1-4 must be complete and backtested

## **9.1 - Paper-to-live transition checklist**

Do NOT go live until all of the following are true:

- Backtested Sharpe ratio > 1.0 over a 3-month window.
- Paper trading P&L is positive over at least 30 consecutive days.
- Backtested P&L and paper P&L agree within 25% (validates that the backtest is realistic).
- All risk checks pass a chaos test: inject random failures into each component and verify the bot halts gracefully.
- Database backups have been verified with a successful restore (SQLite or PostgreSQL, depending on setup).
- Cost budget caps are tested and verified to halt forecasting correctly.
- Alert channels (Telegram/Discord/Slack) are configured and tested.

## **9.2 - Graduated capital deployment**

- Week 1: Deploy with \$100 bankroll. Max stake \$5 per market. Monitor every trade.
- Week 2: If Week 1 P&L is positive or within -10%, increase to \$500 bankroll, \$25 max stake.
- Week 3-4: If cumulative P&L is positive, increase to \$2,000 bankroll, \$50 max stake.
- Month 2+: Scale based on realized Sharpe ratio. Never deploy more than you can afford to lose entirely.

## **9.3 - Production infrastructure**

- Deploy on a VPS with 99.9% uptime SLA (DigitalOcean, Hetzner, or AWS Lightsail). Minimum 2 vCPU, 4GB RAM.
- Use Docker Compose with the existing Dockerfile. Add a Watchtower container for automated image updates from your private registry.
- Set up UptimeRobot or Healthchecks.io to ping the /health endpoint every 60 seconds. Alert if 2 consecutive failures.
- Enable the existing Sentry integration for error tracking in production.

## **9.4 - Kill switch protocol**

- Ensure the dashboard kill switch is accessible from mobile (the Flask dashboard should be responsive or add a Telegram command: /kill).
- Add an automatic kill trigger: if daily P&L drops below -\$X (configurable, default = 5% of bankroll), halt all trading and send a priority alert.
- After a kill trigger, require manual intervention to restart. The bot should NOT auto-recover from a kill-a human needs to review what happened.

**Acceptance Criteria - Phase 9**

✅ Live trading with real capital is profitable over 30+ days

✅ Kill switch can be triggered from mobile in under 10 seconds

✅ Graduated deployment followed strictly-no jumping ahead

✅ UptimeRobot confirms >99.5% uptime over the first month

# **Appendix A: Implementation Priority Matrix**

> **Updated (v1.1):** Reordered to reflect cost control as a prerequisite, quick wins pulled forward, and PostgreSQL deferred.

If you have limited time, here's the priority order. Phase 0A and quick wins are non-negotiable. Phases 1-2 unlock validation. Phases 3-4 are where the money is. Phases 5-8 are optimizations.

| **Order** | **Phase** | **Name**                 | **Priority** | **Effort** | **Impact on P&L**              |
| --------- | --------- | ------------------------ | ------------ | ---------- | ------------------------------ |
| 1         | 0A (NEW)  | Cost Control & Budgets   | CRITICAL     | 2 days     | Prevents cost > profit         |
| 2         | 3.3       | Annualized Edge Filter   | CRITICAL     | 2 hours    | Removes capital traps          |
| 3         | 2.5       | Model Upgrade            | QUICK WIN    | 5 minutes  | Better models, same cost       |
| 4         | 0.3       | Config Validation        | HIGH         | 1 day      | Prevents silent config bugs    |
| 5         | 0.2       | Error Budget Tracking    | HIGH         | 1-2 days   | Circuit breakers               |
| 6         | 1         | Backtesting Engine       | CRITICAL     | 5-7 days   | Enables all other improvements |
| 7         | 2.1-2.4   | Structured Forecasting   | HIGH         | 5-7 days   | 15-30% Brier improvement       |
| 8         | 3.1-3.2   | Edge Confidence + Fees   | HIGH         | 2-3 days   | Fewer false trades             |
| 9         | 3.4       | Event Correlation        | HIGH         | 2 days     | Portfolio risk reduction       |
| 10        | 4         | Domain Specialists       | HIGH         | 7-10 days  | Genuine alpha source           |
| 11        | 8         | Continuous Learning      | MEDIUM       | 4-5 days   | Compound improvement           |
| 12        | 7         | Whale Intelligence       | MEDIUM       | 3-4 days   | Better signal filtering        |
| 13        | 6         | Execution Quality        | MEDIUM       | 3-4 days   | 1-3% slippage reduction        |
| 14        | 5         | Cross-Platform Arb       | MEDIUM       | 5-7 days   | Low-risk consistent profit     |
| 15        | 0.1       | PostgreSQL Upgrade       | LOW          | 2-3 days   | Only needed at scale           |
| 16        | 9         | Production Go-Live       | FINAL        | 3-4 days   | Actual money                   |

# **Appendix B: Detailed API Cost Analysis**

> **Updated (v1.1):** The original estimates used rough per-call averages. This expanded analysis uses actual token counts from the codebase prompts and current API pricing (March 2026) to model costs precisely, then presents three operating profiles.

## **B.1 — Per-Market Cost Breakdown (Current Implementation)**

The current pipeline makes these API calls per market:

| **Step** | **API Call** | **Model** | **Input Tokens** | **Output Tokens** | **Cost/Call** |
| --- | --- | --- | --- | --- | --- |
| Research (search) | 4-8 search queries | SerpAPI/Tavily | N/A | N/A | $0.005/query |
| Evidence extraction | 1 LLM call | gpt-4o | ~2,000 | ~500 | $0.010 |
| Ensemble forecast 1 | 1 LLM call | gpt-4o | ~1,000 | ~300 | $0.0055 |
| Ensemble forecast 2 | 1 LLM call | claude-3-5-sonnet | ~1,000 | ~300 | $0.0075 |
| Ensemble forecast 3 | 1 LLM call | gemini-1.5-pro | ~1,000 | ~300 | $0.0034 |
| **Total per market** | | | | | **~$0.046** |

With the 60-minute research cooldown, each unique market is researched once per hour. At 10 markets/cycle and 480 cycles/day, roughly 100-150 unique markets are processed daily (many are re-evaluated with cached research).

**Current daily cost:** ~$4.60-$6.90 (100-150 unique markets × $0.046)
**Current monthly cost:** ~$140-$210

The original spec's $375-745/month estimate was too high because it didn't account for the research cooldown cache, which prevents re-searching the same market every 3 minutes.

## **B.2 — Three Operating Profiles**

### **Profile A: Paper Trading / Development (Minimum Cost)**

Use during development, backtesting, and initial paper trading.

| **Setting** | **Value** |
| --- | --- |
| Ensemble | Disabled (single model) |
| Forecast model | gpt-4o-mini |
| Evidence extraction model | gpt-4o-mini |
| Search provider | Tavily (free tier: 1,000 searches/month) |
| Cycle interval | 300s (5 min) |
| Markets per cycle | 5 |
| Daily budget cap | $1.00 |

**Per-market cost:** ~$0.003 (gpt-4o-mini for extraction + forecast, free search)
**Daily cost:** ~$0.15-$0.30
**Monthly cost:** ~$5-$10

This profile is sufficient for validating pipeline logic, testing new features, and running paper trades. Forecast quality is lower, but that's acceptable for development.

### **Profile B: Standard Live Trading (Cost-Optimized)**

Post Phase 0A implementation with model tiering.

| **Setting** | **Value** |
| --- | --- |
| Scout tier | gpt-4o-mini (screens all markets) |
| Standard tier | gpt-4o-mini + gemini-2.0-flash (2-model ensemble) |
| Premium tier | gpt-4o + claude-sonnet-4-5 + gemini-2.0-flash (high-edge only) |
| Evidence extraction | gpt-4o-mini |
| Search provider | SerpAPI ($50/month for 5K searches) |
| Cycle interval | 180s (3 min) |
| Markets per cycle | 10 |
| Daily budget cap | $5.00 |

Cost flow per cycle (10 markets):
1. **Scout screen**: 10 markets × gpt-4o-mini = 10 × $0.0003 = $0.003
2. **Markets passing scout** (~3-4): Full research + standard ensemble
   - Research: 4 × $0.005 (search) + $0.0006 (extraction) = $0.021
   - Standard forecast: 2 models × $0.001 = $0.002
   - Per qualifying market: ~$0.023
3. **Markets escalated to premium** (~0-1): +$0.015 for 3rd frontier model

**Daily cost:** ~$1.50-$3.00
**Monthly cost:** ~$45-$90

This is **50-70% cheaper** than the current implementation while maintaining ensemble forecasting for markets that matter.

### **Profile C: Full Capacity (Maximum Accuracy)**

For validated profitable strategies with sufficient bankroll to justify the cost.

| **Setting** | **Value** |
| --- | --- |
| All tiers active | Scout → Standard → Premium routing |
| Premium tier | gpt-4o + claude-sonnet-4-5 + gemini-2.0-flash + DeepSeek-R1 |
| Evidence extraction | gpt-4o (maximum extraction quality) |
| Search provider | SerpAPI + Tavily (redundant) |
| Cycle interval | 120s (2 min) |
| Markets per cycle | 15 |
| Daily budget cap | $15.00 |

**Daily cost:** ~$8-$15
**Monthly cost:** ~$240-$450

Only justified when monthly trading P&L consistently exceeds $500+.

## **B.3 — Break-Even Analysis**

| **Profile** | **Monthly Cost** | **Bankroll Needed for Break-Even** | **Required Monthly Return** |
| --- | --- | --- | --- |
| A (Paper) | $5-$10 | N/A (development) | N/A |
| B (Standard) | $45-$90 | $1,000 | 4.5-9% |
| C (Full) | $240-$450 | $5,000 | 4.8-9% |

A 5-9% monthly return is ambitious but achievable for a well-calibrated prediction market bot. The key insight: **Profile B is the sweet spot.** It cuts costs by 50-70% vs. the original spec's estimates while maintaining ensemble quality on the trades that matter most.

## **B.4 — Cost Reduction Strategies (Ranked by Impact)**

| **Strategy** | **Savings** | **Effort** | **Quality Impact** |
| --- | --- | --- | --- |
| Switch evidence extraction to gpt-4o-mini | 40% of extraction cost | 5 min (config change) | <3% quality loss |
| Add scout tier (screen with cheap model) | 60% of total LLM cost | 1 day (Phase 0A.2) | None — more screening, not less accuracy |
| Switch Gemini to 2.0 Flash | 12x cheaper per Gemini call | 5 min (config change) | Comparable on structured output |
| Increase research cooldown to 120 min | 50% of search cost | Config change | Staler research on fast-moving markets |
| Reduce ensemble to 2 models (standard tier) | 33% of forecast cost | Config change | ~2-5% Brier score degradation |
| Cache LLM responses by prompt hash (Phase 1.2) | 80%+ on repeat markets | 1 day | Zero — identical prompts = identical answers |
| Batch evidence extraction (3 markets/call) | 60% of extraction cost | 0.5 day | None if prompt fits context window |

## **B.5 — Infrastructure Costs**

| **Service** | **Cost/Month** | **Notes** |
| --- | --- | --- |
| VPS hosting | $6-$20 | Hetzner CX22 ($6), DigitalOcean ($12-$20) |
| Domain + SSL | $0 | Let's Encrypt + free subdomain |
| Sentry (error tracking) | $0 | Free tier (5K events/month) |
| **Total infra** | **$6-$20/month** | |

**PostgreSQL is not included** — SQLite is sufficient for current scale (see Phase 0.1 note).

## **B.6 — Budget Cap Implementation Reference**

The budget cap system from Phase 0A.1 should integrate with `CostTracker` as follows:

```python
# In CostTracker (src/observability/metrics.py)
def record_call(self, api_name: str, input_tokens: int = 0,
                output_tokens: int = 0, count: int = 1) -> None:
    """Record an API call with actual token-based cost."""
    if input_tokens > 0:
        cost = self._compute_token_cost(api_name, input_tokens, output_tokens)
    else:
        cost = self._costs.get(api_name, 0.001) * count  # Fallback flat rate
    with self._lock:
        self._cycle_cost += cost
        self._daily_cost += cost
        self._total_cost += cost

def check_budget(self) -> bool:
    """Return False if daily budget is exhausted."""
    with self._lock:
        return self._daily_cost < self._daily_budget

def remaining_daily_budget(self) -> float:
    with self._lock:
        return max(0.0, self._daily_budget - self._daily_cost)
```

The engine loop checks `cost_tracker.check_budget()` before each forecast cycle. If exhausted, log the event and skip to position monitoring.

# **Appendix C: Key Files to Modify (Base Repo)**

Reference for where new modules plug into the existing codebase:

| **New Module**                   | **Plugs Into**                    | **Integration Point**             |
| -------------------------------- | --------------------------------- | --------------------------------- |
| src/forecast/model_router.py     | src/forecast/ensemble.py          | Tier selection before forecast    |
| src/observability/metrics.py     | (existing — modify CostTracker)   | Add budget cap + token tracking   |
| config.yaml: cost_management     | src/engine/loop.py                | Budget check before each cycle    |
| src/storage/pg_adapter.py        | src/storage/                      | Replace SQLite imports (deferred) |
| src/backtest/\*                  | New directory                     | CLI command + dashboard tab       |
| src/forecast/decomposer.py       | src/forecast/llm_forecaster.py    | Called before ensemble            |
| src/forecast/base_rates.py       | src/forecast/llm_forecaster.py    | Injected into prompt              |
| src/forecast/adaptive_weights.py | src/forecast/ensemble.py          | Replaces static weights           |
| src/forecast/specialists/\*      | src/forecast/specialist_router.py | New routing layer                 |
| src/policy/edge_uncertainty.py   | src/policy/edge_calc.py           | Modifies edge calculation         |
| src/policy/capital_efficiency.py | src/engine/trading_loop.py        | Market ranking input              |
| src/policy/correlation.py        | src/policy/portfolio_risk.py      | Replaces category caps            |
| src/connectors/kalshi_client.py  | src/connectors/                   | New connector                     |
| src/policy/cross_platform_arb.py | src/engine/trading_loop.py        | Parallel arb scanner              |
| src/analytics/post_mortem.py     | src/engine/trading_loop.py        | Post-resolution hook              |
| src/analytics/param_optimizer.py | Cron / scheduled task             | Weekly batch job                  |
| src/analytics/whale_scorer.py    | src/analytics/                    | Filters whale signals             |

# **Appendix D: Model Pricing Reference (March 2026)**

Quick reference for cost calculations. Prices change — verify against provider pricing pages.

| **Model** | **Input ($/1M tokens)** | **Output ($/1M tokens)** | **Typical Call Cost** | **Best For** |
| --- | --- | --- | --- | --- |
| gpt-4o-mini | $0.15 | $0.60 | $0.0003 | Scout screening, evidence extraction, decomposition |
| gpt-4o | $2.50 | $10.00 | $0.005 | Premium tier forecasting |
| claude-haiku-3.5 | $0.80 | $4.00 | $0.002 | Decomposition, structured output |
| claude-sonnet-4.5 | $3.00 | $15.00 | $0.008 | Premium tier forecasting |
| gemini-2.0-flash | $0.10 | $0.40 | $0.0003 | Standard tier ensemble, high-volume tasks |
| gemini-1.5-pro | $1.25 | $5.00 | $0.003 | Premium tier (if Gemini slot needed) |
| deepseek-r1 | $0.55 | $2.19 | $0.002 | 4th ensemble slot for diversity |

**Free tiers to exploit:**
- Google Gemini: 1,500 requests/day free for 2.0 Flash
- Tavily: 1,000 searches/month free
- SerpAPI: 100 searches/month free

_End of Document_

This is a living document. Update acceptance criteria as implementation reveals new constraints.
