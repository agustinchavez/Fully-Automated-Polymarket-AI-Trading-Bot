# Glossary

Key terms used throughout the bot, dashboard, and documentation.

---

## Forecasting & Accuracy

### Brier Score

A measure of forecast accuracy. Ranges from 0 (perfect) to 1 (worst possible).

**Formula:** `Brier = mean((forecast - outcome)^2)` for all resolved markets, where `outcome` is 1 (YES) or 0 (NO).

| Score | Quality |
|-------|---------|
| 0.00 | Perfect forecaster |
| 0.10 | Superforecaster level |
| 0.20 | Good — consistently beating the market |
| 0.25 | Random guessing (always predicting 50%) |
| 0.33+ | Worse than random |

The bot tracks Brier scores per model, per category, and overall. Lower is always better. The **Paper Trading Progress** milestone requires Brier < 0.25 to unlock live trading.

### Calibration

How well your predicted probabilities match actual outcomes. A well-calibrated forecaster who says "70% chance" should be right about 70% of the time.

The dashboard plots a **calibration curve**: predicted probability (x-axis) vs. actual resolution rate (y-axis). Perfect calibration is the diagonal line. Points above the line mean you're underconfident; below means overconfident.

The bot uses **Platt scaling** (logistic regression on past forecasts) to auto-correct systematic miscalibration.

### Base Rate

The historical frequency of an event type occurring, used as a starting point before incorporating specific evidence. Example: "Incumbent US presidents win re-election ~70% of the time" is a base rate.

The bot maintains a registry of 56 base rate patterns across 7 categories. When enabled, the LLM prompt starts with the relevant base rate, then adjusts based on evidence.

### Ensemble

Running multiple AI models in parallel and combining their forecasts. The bot uses 3 models by default:

| Model | Default Weight |
|-------|---------------|
| GPT-4o | 40% |
| Claude Sonnet 4.6 | 35% |
| Gemini 2.0 Flash | 25% |

Aggregation methods: **trimmed mean** (drops outliers), **median**, or **weighted average**. When models disagree significantly (>10% spread), a penalty is applied to increase uncertainty.

### Implied Probability

The probability derived from a market price. On Polymarket, a YES token priced at $0.65 implies a 65% probability of the event occurring.

---

## Edge & Trading

### Edge

The difference between what you believe the true probability is and what the market is pricing. Edge is your expected profit margin.

**Formula:** `edge = |model_probability - market_price| - fees`

Example: If your model says 72% and the market prices YES at 60%, your gross edge is 12 percentage points. After ~2% fees each way (entry + exit), net edge is ~8%.

The bot requires a minimum edge (default 4%) before placing any trade.

### Kelly Criterion

A mathematical formula for optimal bet sizing that maximizes long-term growth rate. It tells you what fraction of your bankroll to bet based on your edge and the odds.

**Formula:** `kelly_fraction = edge / odds`

The bot uses **fractional Kelly** (default 25% of full Kelly) to reduce variance, then applies 7 multipliers:

1. **Confidence** — higher confidence = bigger position
2. **Drawdown** — reduce size during losing streaks
3. **Timeline** — smaller bets near resolution (less time to recover)
4. **Volatility** — reduce in volatile markets
5. **Regime** — adjust for market conditions
6. **Category** — reduce in unfamiliar categories
7. **Liquidity** — smaller positions in illiquid markets

### Slippage

The difference between the expected execution price and the actual fill price. Caused by market impact (your order moving the price) and book depth.

Measured in **basis points (bps)**: 100 bps = 1%. Example: If you expected to buy at $0.60 and filled at $0.605, slippage is 50 bps.

The dashboard tracks slippage by strategy (Simple, TWAP, Iceberg) so you can see which execution method minimizes costs.

---

## Risk Management

### Drawdown

The decline from a portfolio's peak value to its subsequent trough, expressed as a percentage. It measures the worst loss you'd have experienced if you entered at the peak.

**Formula:** `drawdown_pct = (peak_equity - current_equity) / peak_equity`

Example: If your portfolio peaked at $5,500 and dropped to $5,000, drawdown is 9.1%.

### Heat System

A 4-level progressive risk reduction system tied to drawdown:

| Level | Drawdown | Position Sizing |
|-------|----------|----------------|
| Normal | < 10% | 100% (full Kelly fraction) |
| Warning | 10-15% | 50% sizing |
| Critical | 15-20% | 25% sizing |
| Max | >= 20% | 0% — all trading halted |

The heat level auto-resets as the portfolio recovers.

### Value at Risk (VaR)

An estimate of the maximum loss expected over a given time period at a given confidence level. The bot calculates:

- **VaR 95%** — you're 95% confident losses won't exceed this in a day
- **VaR 99%** — you're 99% confident losses won't exceed this

Example: "Daily VaR 95% = $150" means on 19 out of 20 trading days, you expect to lose less than $150.

When `var_gate_enabled=True`, the bot blocks new trades if projected portfolio VaR exceeds `max_portfolio_var_pct` of bankroll.

### Sharpe Ratio

A measure of risk-adjusted return. Higher is better.

**Formula:** `Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)`

| Score | Quality |
|-------|---------|
| < 0 | Losing money |
| 0 - 1 | Below average |
| 1 - 2 | Good |
| 2 - 3 | Very good |
| 3+ | Excellent |

Related ratios the bot also tracks:
- **Sortino** — like Sharpe but only penalizes downside volatility (more relevant for trading)
- **Calmar** — annualized return / max drawdown (how well you handle worst-case scenarios)

### Circuit Breaker

An automatic safety mechanism that temporarily disables an API provider after repeated failures. Uses a state machine:

1. **CLOSED** (normal) — requests flow normally
2. **OPEN** (tripped) — requests immediately fail without calling the API
3. **HALF_OPEN** (testing) — allows one test request to check if the service recovered

Each LLM provider (OpenAI, Anthropic, Google) and external API has its own circuit breaker. This prevents a failing API from slowing down the entire pipeline.

---

## Market Concepts

### Market Regime

The bot automatically classifies current market conditions:

| Regime | Description | Sizing Effect |
|--------|-------------|---------------|
| Normal | Average volatility and activity | 1.0x (baseline) |
| Trending | Strong directional movement | 0.8x |
| Mean-Reverting | Prices oscillating around a value | 1.1x |
| High Volatility | Unusual price swings | 0.6x |
| Low Activity | Below-average trading volume | 0.7x |

### Conviction Signal

A trading signal generated when multiple whale wallets take the same position in a market. Strength depends on:

- **Whale count** — how many whales agree
- **Dollar volume** — total capital deployed
- **Whale quality** — scored whales weighted higher (S-tier > C-tier)

When whales agree with your model: **+8% edge boost**. When they disagree: **-2% penalty**.

---

## Paper Trading Milestones

The **Paper Trading Progress** widget tracks three milestones:

### Smart Retrain
**Requirement:** 30 resolved trades

The calibration system auto-refits its correction curve after enough resolved data accumulates. This lets the model learn from its mistakes and systematically correct for biases.

### AI Analyst
**Requirements:** 50 resolved trades + 28 days running

Unlocks the AI analyst feature, which uses an LLM to review your trading performance and generate actionable recommendations (what's working, what needs improvement, specific strategy changes).

### Live Trading Gate
**Requirements:** 30 resolved trades + 28 days running + Brier score < 0.25

All three conditions must be met before the system considers you ready for live trading. This ensures you have enough data, enough time, and sufficient forecast accuracy before risking real capital.

---

## Execution Strategies

| Strategy | When Used | How It Works |
|----------|-----------|-------------|
| **Simple** | Small orders | Single limit order at target price |
| **TWAP** | Large orders (> thin book) | Splits into 5 time-weighted child orders to minimize market impact |
| **Iceberg** | Very large orders | Shows only 20% of true order size; hidden portion fills behind the scenes |
| **Adaptive** | Learning mode | Tracks which strategy performs best for current market conditions |

---

For the complete list of API endpoints, see the [API Reference](API_REFERENCE.md).
