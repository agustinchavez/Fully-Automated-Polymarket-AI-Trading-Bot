# Dashboard API Reference

All endpoints are served by the Flask dashboard at `http://localhost:2345`. When `DASHBOARD_API_KEY` is set, every request must include authentication via the `X-API-Key` header or `?api_key=` query parameter. Health/readiness/metrics endpoints are always open.

---

## Health & System

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | Open | Liveness probe. Always returns `200 OK`. |
| `/ready` | GET | Open | Readiness check. Verifies DB connectivity and engine state. |
| `/metrics` | GET | Open | Prometheus-compatible metrics export (counters, histograms). |

---

## Portfolio & Positions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/portfolio` | GET | Portfolio overview: net value, allocation breakdown, unrealized P&L. |
| `/api/positions` | GET | All open positions with current prices, entry prices, and P&L. |
| `/api/positions/<market_id>` | GET | Detailed view of a single position including trade history. |
| `/api/equity-curve` | GET | Historical equity progression as a time series. |
| `/api/equity-snapshots` | GET | Timestamp-based equity snapshots for charting. |
| `/api/var` | GET | Value at Risk calculations (parametric and historical methods). See [VaR](#value-at-risk-var). |
| `/api/portfolio-risk` | GET | Aggregate risk metrics: total exposure, category concentration, correlation. |
| `/api/performance` | GET | Performance metrics: win rate, [Sharpe ratio](#sharpe-ratio), ROI, [max drawdown](#drawdown). |

---

## Trading & Execution

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trades` | GET | All historical trades with status, P&L, and timestamps. |
| `/api/trade-detail/<market_id>` | GET | Full trade history for a specific market. |
| `/api/forecasts` | GET | Model probability forecasts with edge calculations. |
| `/api/candidates` | GET | Current cycle's candidate markets being evaluated. |
| `/api/execution-plans` | GET | All TWAP/Iceberg execution plans (active + completed). |
| `/api/execution-plans/active` | GET | Active execution plans only. |
| `/api/execution-plans/<plan_id>/cancel` | POST | Cancel an active TWAP or Iceberg execution plan. |
| `/api/execution-quality` | GET | Execution quality metrics: [slippage](#slippage), partial fills, strategy stats. |
| `/api/engine-status` | GET | Engine running status, uptime, current cycle count. |
| `/api/engine/start` | POST | Start the trading engine. |
| `/api/engine/stop` | POST | Stop the trading engine gracefully. |
| `/api/latency` | GET | Per-endpoint API latency percentiles (p50, p95, p99). |

---

## Risk & Decision Support

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/risk` | GET | Current risk config: limits, [min edge](#edge), [VaR](#value-at-risk-var) thresholds. |
| `/api/filter-stats` | GET | Pipeline rejection reasons and pass/fail counts per stage. |
| `/api/decision-log` | GET | All decisions with TRADE/SKIP/NO-TRADE outcomes and reasoning. |
| `/api/drawdown` | GET | Current [drawdown](#drawdown) level, history, and [heat system](#heat-system) state. |
| `/api/regime` | GET | Current [market regime](#market-regime) classification (Normal/Trending/etc.). |
| `/api/alerts` | GET | Alert log with severity levels (info, warning, critical). |
| `/api/invariant-checks` | GET | 12 data integrity checks validating internal consistency. |
| `/api/circuit-breakers` | GET | Per-provider [circuit breaker](#circuit-breaker) states (CLOSED/OPEN/HALF_OPEN). |
| `/api/market-types` | GET | Breakdown of markets by category (MACRO, ELECTION, CRYPTO, etc.). |
| `/api/market-detail/<slug>` | GET | Live market data from Polymarket by event slug. |
| `/api/event-calendar` | GET | Upcoming events and resolution deadlines. |
| `/api/event-triggers` | GET | Event-based signals that may affect open positions. |
| `/api/reconciliation` | GET | Trade vs. execution reconciliation report. |

---

## Whale Intelligence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/whale-activity` | GET | Top whale trades, conviction signals, and smart money index. |
| `/api/whale-profile/<address>` | GET | Individual whale metrics: ROI, win rate, specialization. |
| `/api/whale-quality-scores` | GET | Whale quality tiers (S/A/B/C) with 5-dimension scoring. |
| `/api/whale-stars` | GET | List of starred (bookmarked) whale wallets. |
| `/api/whale-stars` | POST | Add a whale wallet to your starred list. Body: `{"address": "0x..."}` |
| `/api/whale-mentor` | POST | Chat with the AI whale mentor. Body: `{"message": "..."}` |
| `/api/whale-mentor/history` | GET | Whale mentor conversation history. |
| `/api/whale-mentor/clear` | POST | Clear whale mentor conversation. |
| `/api/whales/liquid-scan/*` | GET | 7 sub-routes for the whale liquid market scanner pipeline. |

---

## Backtesting

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/backtest/runs` | GET | List all backtest runs with summary metrics. |
| `/api/backtest/runs/<run_id>` | GET | Detailed metrics for a specific run: [Sharpe](#sharpe-ratio), [Brier](#brier-score), [max drawdown](#drawdown). |
| `/api/backtest/runs/<run_id>/trades` | GET | Individual trades from a specific backtest run. |
| `/api/backtest/runs/<run_id>/calibration` | GET | [Calibration curve](#calibration) data for a backtest run. |
| `/api/backtest/compare` | GET | A/B comparison of two runs with paired t-test and significance levels. |
| `/api/backtest/cache-stats` | GET | LLM response cache hit rate and size. |
| `/api/backtest/markets` | GET | Markets available for backtesting (scraped resolved markets). |
| `/api/forecast/base-rates` | GET | [Base rate](#base-rate) registry: 56 patterns across 7 categories. |

---

## Insights & Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/paper-trading-progress` | GET | Trade counts (today/7d/30d), days running, and milestone progress toward Smart Retrain, AI Analyst, and Live Trading Gate. |
| `/api/insights/pnl-overview` | GET | P&L overview: KPIs, equity curve, daily bars. Query: `?days=30` |
| `/api/insights/category-breakdown` | GET | Per-category performance: win rate, P&L, edge by category. Query: `?days=30` |
| `/api/insights/model-accuracy` | GET | Per-model [Brier scores](#brier-score), [calibration](#calibration) curves. Query: `?days=30` |
| `/api/insights/friction` | GET | Friction waterfall: gross edge -> fees -> [slippage](#slippage) -> net P&L. Query: `?days=30` |
| `/api/insights/summary` | GET | Weekly digest summary (same data as the Telegram `/weekly` command). |
| `/api/insights/export` | GET | Export insights data as CSV or JSON. |
| `/api/insights/ai-analysis` | GET | Retrieve cached AI analyst report. |
| `/api/insights/ai-analysis` | POST | Trigger a new AI analyst report (rate-limited). |

---

## Continuous Learning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model-accuracy` | GET | Per-model [Brier scores](#brier-score) with sample sizes. |
| `/api/adaptive-weights` | GET | Category-level model weighting based on historical accuracy. |
| `/api/calibration/retrain-history` | GET | [Calibration](#calibration) retraining events and trigger reasons. |
| `/api/calibration-curve` | GET | [Calibration curve](#calibration) with probability buckets. |
| `/api/param-optimization/runs` | GET | Parameter optimizer run history. |
| `/api/param-optimization/suggestions` | GET | Suggested parameter configurations from optimizer. |
| `/api/param-optimization/apply/<run_id>` | POST | Apply an optimized parameter set (requires manual approval). |

---

## Post-Mortem & Evidence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/post-mortem/recent` | GET | Recent trade-by-trade analysis after market resolution. |
| `/api/post-mortem/summary` | GET | Weekly post-mortem summary with lessons learned. |
| `/api/evidence-quality` | GET | Domain-level evidence source quality rankings. |

---

## Arbitrage

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/arbitrage` | GET | Detected arbitrage opportunities (intra-Polymarket + cross-platform). |
| `/api/arbitrage/summary` | GET | Arbitrage stats: opportunity count, average spread, executed trades. |
| `/api/arbitrage/matches` | GET | Polymarket <-> Kalshi market matches with price comparison. |

---

## Kill Switch & Deployment

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kill-switch` | POST | Toggle the engine kill switch (emergency halt). |
| `/api/kill-switch/state` | GET | Current kill switch state and who triggered it. |
| `/api/kill-switch/reset` | POST | Reset the kill switch to allow trading to resume. |
| `/api/deployment-stage` | GET | Current graduated deployment stage (paper/week1/week2/week3_4/month2_plus). |

---

## Configuration & Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Current `config.yaml` contents (secrets masked). |
| `/api/config` | POST | Save config edits. Body: YAML string. |
| `/api/config/reload` | POST | Reload config from disk without restart. |
| `/api/config/reset` | POST | Reset config to defaults. |
| `/api/config/schema` | GET | JSON schema of all config sections and fields. |
| `/api/env` | GET | Environment variables (secret values masked). |
| `/api/env` | POST | Update environment variables. |
| `/api/flags` | POST | Toggle feature flags at runtime. |
| `/api/admin` | GET | Admin health dashboard: system score, uptime, DB size. |
| `/api/admin/log-tail` | GET | Last 100 lines of `bot.log`. |
| `/api/admin/db-vacuum` | POST | Compact the SQLite database. |
| `/api/admin/clear-cache` | POST | Clear all in-memory caches. |
| `/api/admin/rotate-logs` | POST | Rotate log files. |
| `/api/admin/backup-db` | POST | Create a database backup (stored in `data/backups/`). |
| `/api/admin/reset-metrics` | POST | Reset all metric counters to zero. |
| `/api/admin/test-alert` | POST | Send a test alert to all configured channels. |
| `/api/admin/export/<table>` | GET | Export any database table as CSV. |

---

## Data Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/journal` | GET | Trade journal entries (auto-generated on position close). |
| `/api/journal` | POST | Add a manual journal entry. Body: `{"market_id": "...", "note": "..."}` |
| `/api/watchlist` | GET | Markets on your watchlist. |
| `/api/watchlist` | POST | Add a market to watchlist. Body: `{"market_id": "..."}` |
| `/api/watchlist/<market_id>` | DELETE | Remove a market from watchlist. |
| `/api/daily-summaries` | GET | Daily P&L summaries with trade counts. |

---

## Wallets & Strategies

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/wallets` | GET | List all wallets with balances. |
| `/api/wallets` | POST | Create a new wallet. Body: `{"name": "...", "type": "paper"}` |
| `/api/wallets/<id>` | PUT | Update wallet settings. |
| `/api/wallets/<id>` | DELETE | Delete a wallet. |
| `/api/wallets/<id>/performance` | GET | Per-wallet P&L and equity curve. |
| `/api/strategies` | GET | List all strategies. |
| `/api/strategies` | POST | Create a new strategy. Body: `{"name": "...", "type": "ai_trading"}` |
| `/api/strategies/<id>` | PUT | Update strategy settings. |
| `/api/strategies/<id>` | DELETE | Delete a strategy. |
| `/api/strategies-overview` | GET | Strategy performance comparison grid. |
| `/api/strategy-wallets` | POST | Bind a strategy to a wallet. |
| `/api/strategy-wallets/toggle` | POST | Enable/disable a strategy-wallet binding. |

---

## Other

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/connector-status` | GET | Exchange and API connector health status. |
| `/api/audit` | GET | Audit log of configuration and system changes. |
| `/api/costs` | GET | LLM token usage and cost breakdown by model/provider. |
| `/api/export/<table>` | GET | Export any database table as CSV. |

---

## Authentication

When `DASHBOARD_API_KEY` is set in `.env`:

```bash
# Via header
curl -H "X-API-Key: your-key" http://localhost:2345/api/portfolio

# Via query parameter
curl http://localhost:2345/api/portfolio?api_key=your-key
```

When `DASHBOARD_API_KEY` is not set, all endpoints are open (suitable for local development).

---

## Error Responses

All endpoints return JSON. On error:

```json
{
  "error": "description of what went wrong"
}
```

Common HTTP status codes:
- `200` - Success
- `401` - Unauthorized (missing or invalid API key)
- `404` - Resource not found
- `500` - Internal server error

---

For definitions of terms like Brier score, Sharpe ratio, edge, drawdown, and more, see the [Glossary](GLOSSARY.md).
