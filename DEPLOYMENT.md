# Deployment & Live Trading Guide

## Quick Start (Paper Trading)

```bash
# 1. Clone and install
git clone https://github.com/agustinchavez/Fully-Automated-Polymarket-AI-Trading-Bot.git
cd Fully-Automated-Polymarket-AI-Trading-Bot
make dev

# 2. Configure
cp .env.example .env
# Edit .env with your API keys (at minimum: OPENAI_API_KEY)

# 3. Run
make dashboard
# Visit http://localhost:2345
```

## Configuration

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | GPT-4o for forecasting |
| `ANTHROPIC_API_KEY` | Optional | Claude Sonnet 4.6 for ensemble |
| `GOOGLE_API_KEY` | Optional | Gemini 2.0 Flash for ensemble |
| `TAVILY_API_KEY` | Optional | Paid search fallback (1,000/mo free tier) |
| `SERPAPI_KEY` | Optional | Paid search fallback (250/mo free tier) |
| `BING_API_KEY` | Optional | Paid search fallback |
| `SEARXNG_URL` | Optional | Self-hosted SearXNG URL (default: http://localhost:8080) |
| `FRED_API_KEY` | Optional | FRED economic data (free, instant signup) |
| `COINGECKO_API_KEY` | Optional | CoinGecko crypto prices (free demo) |
| `CONGRESS_API_KEY` | Optional | Congress.gov legislative data (free) |
| `COURTLISTENER_API_KEY` | Optional | CourtListener legal cases (free) |
| `DEEPSEEK_API_KEY` | Optional | DeepSeek AI analyst provider |
| `DISCORD_BOT_TOKEN` | Optional | Discord interactive kill bot |
| `DISCORD_CHANNEL_ID` | Optional | Discord channel for bot commands |
| `SLACK_BOT_TOKEN` | Optional | Slack interactive kill bot |
| `SLACK_APP_TOKEN` | Optional | Slack app-level token (Socket Mode) |
| `SLACK_CHANNEL_ID` | Optional | Slack channel for bot commands |
| `TELEGRAM_BOT_TOKEN` | Optional | Telegram interactive kill bot |
| `TELEGRAM_CHAT_ID` | Optional | Telegram chat for bot commands |
| `DISCORD_WEBHOOK_URL` | Optional | Discord webhook for alert notifications |
| `POLYMARKET_API_KEY` | Live only | Polymarket CLOB API |
| `POLYMARKET_API_SECRET` | Live only | CLOB API secret |
| `POLYMARKET_API_PASSPHRASE` | Live only | CLOB passphrase |
| `POLYMARKET_PRIVATE_KEY` | Live only | Polygon wallet key |
| `ENABLE_LIVE_TRADING` | Live only | Set `true` for real trades |
| `DASHBOARD_API_KEY` | Optional | Protect dashboard |
| `SENTRY_DSN` | Optional | Error tracking |

Web search uses **DuckDuckGo by default** (free, unlimited, no API key). Paid providers and self-hosted SearXNG are optional fallbacks.

### Runtime Config (`config.yaml`)

All settings are tunable via `config.yaml`. The file is hot-reloaded
every cycle — changes take effect without restarting.

Key settings to tune:
- `risk.bankroll` — your total capital
- `risk.max_stake_per_market` — max bet size
- `risk.min_edge` — minimum edge to trade (default 4%)
- `engine.cycle_interval_secs` — how often to scan (default 180s)
- `ensemble.enabled` — use multi-model forecasting

---

## Deployment Options

### Option 1: Direct (Development)

```bash
make dashboard        # Flask dev server
# or
make gunicorn         # Production WSGI server
```

### Option 2: Docker

```bash
# Build and run
docker compose up -d

# View logs
docker compose logs -f bot

# Stop
docker compose down
```

### Option 3: Systemd (Linux VPS)

Create `/etc/systemd/system/polymarket-bot.service`:

```ini
[Unit]
Description=Polymarket Trading Bot
After=network.target

[Service]
Type=simple
User=botuser
WorkingDirectory=/opt/polymarket-bot
EnvironmentFile=/opt/polymarket-bot/.env
ExecStart=/opt/polymarket-bot/.venv/bin/gunicorn \
    --bind 0.0.0.0:2345 \
    --workers 2 --threads 4 --timeout 120 \
    src.dashboard.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable polymarket-bot
sudo systemctl start polymarket-bot
```

---

## Going Live — Staged Deployment

### Step 0: Run Preflight Check

Before any live trading, run the built-in 7-check preflight gate:

```bash
bot production preflight
```

All checks must pass:
- Backtest Sharpe >= 1.0
- Paper trading >= 30 days
- Backtest-paper agreement (Sharpe delta < 1.0)
- Chaos tests passed
- DB backup exists and recent
- Budget caps configured
- Alert channel configured (Telegram, Discord, or Slack)

### Step 1: Paper Trading (Required First)

Paper trading runs by default. No configuration changes needed:

```bash
make dashboard
# Monitor via http://localhost:2345
# Review weekly digest each Monday
# Aim for 50+ resolved trades before proceeding
```

### Step 2: Graduated Live Deployment

The bot enforces a 5-stage graduated deployment system:

| Stage | Duration | Max Bankroll | Max Stake | Kelly Fraction |
|-------|----------|-------------|-----------|---------------|
| Paper | 30+ days | Unlimited (simulated) | Unlimited | 0.25 |
| Week 1 | 7 days | $100 | $5 | 0.10 |
| Week 2 | 7 days | $500 | $25 | 0.15 |
| Weeks 3-4 | 14 days | $2,000 | $50 | 0.20 |
| Month 2+ | Ongoing | Config value | Config value | 0.25 |

### Step 3: Configure for Live Trading

```bash
# In .env:
POLYMARKET_CHAIN_ID=137           # Polygon mainnet
ENABLE_LIVE_TRADING=true
```

```yaml
# In config.yaml:
execution:
  dry_run: false
engine:
  paper_mode: false
risk:
  bankroll: 100.0                 # Start with week1 limits
  max_stake_per_market: 5.0
  kelly_fraction: 0.10            # Conservative Kelly for week1
drawdown:
  max_drawdown_pct: 0.10          # Tight 10% drawdown for week1
```

### Step 4: Install CLOB Client

```bash
pip install py-clob-client
```

### Step 5: Run and Monitor

```bash
make dashboard
# Send !status to Discord/Slack or /status to Telegram to confirm running
# Review P&L daily during week1
```

---

## Monitoring

### Health Checks

- `GET /health` — liveness (always returns 200)
- `GET /ready` — readiness (checks DB, engine)
- `GET /metrics` — Prometheus-compatible metrics

### Dashboard Auth

Set `DASHBOARD_API_KEY` in `.env`, then access with:
- Header: `X-API-Key: your-key`
- Query: `?api_key=your-key`

### Database Backups

```bash
make backup            # Manual backup
# Backups stored in data/backups/ (max 10, auto-pruned)
```

### Error Tracking

Set `SENTRY_DSN` in `.env` for automatic exception reporting.

---

## API Cost Estimates

| Component | Cost per market | Notes |
|-----------|----------------|-------|
| Scout tier (gpt-4o-mini) | ~$0.001 | Low-value markets, quick filter |
| Standard tier (gpt-4o) | ~$0.01 | Most markets |
| Premium tier (ensemble) | ~$0.05 | High-value markets, 3 models |
| Web search (DuckDuckGo) | Free | Default provider, no API key |
| Web search (SerpAPI/Tavily) | ~$0.01-0.03 | Optional paid fallback, cached 2hr |
| Research connectors | Free | FRED, CoinGecko, Congress, etc. |
| **Typical cycle (10 markets)** | **~$0.10-0.50** | Depends on tier distribution |

Model tier router automatically assigns markets to cost-appropriate tiers.
Research connectors reduce web search costs by providing structured data first.
