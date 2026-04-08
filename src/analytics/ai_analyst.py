"""AI analyst — multi-provider analysis of bot performance.

Routes analysis requests to one of four providers (Anthropic, OpenAI,
Google, DeepSeek) based on config. Assembles context from the DB,
builds a provider-agnostic prompt, and parses the JSON response.

Rate-limited to 1 call per config.rate_limit_hours via engine_state.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from src.config import AnalystConfig, BotConfig
from src.observability.logger import get_logger

log = get_logger(__name__)

MIN_RESOLVED_TRADES_DEFAULT = 50
MIN_DATA_DAYS_DEFAULT = 28


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class Recommendation:
    priority: int = 1
    action: str = ""
    rationale: str = ""
    config_change: str | None = None
    expected_impact: str = ""


@dataclass
class AnalysisResult:
    summary: str = ""
    what_is_working: list[str] = field(default_factory=list)
    what_is_not_working: list[str] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    confidence: str = "low"
    data_sufficient: bool = False
    provider_used: str = ""
    model_used: str = ""
    parse_error: bool = False
    raw_response: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "what_is_working": self.what_is_working,
            "what_is_not_working": self.what_is_not_working,
            "recommendations": [
                {"priority": r.priority, "action": r.action,
                 "rationale": r.rationale, "config_change": r.config_change,
                 "expected_impact": r.expected_impact}
                for r in self.recommendations
            ],
            "confidence": self.confidence,
            "data_sufficient": self.data_sufficient,
            "provider_used": self.provider_used,
            "model_used": self.model_used,
            "parse_error": self.parse_error,
            "generated_at": self.generated_at,
        }


@dataclass
class AnalystContext:
    period_start: str = ""
    period_end: str = ""
    overall_summary: str = ""
    category_stats: str = ""
    model_stats: str = ""
    friction_summary: str = ""
    config_snapshot: str = ""
    data_quality: dict[str, Any] = field(default_factory=dict)


# ── Provider Routing ─────────────────────────────────────────────────


def _route_provider(provider: str) -> str:
    """Normalise provider string to canonical name."""
    p = provider.lower().strip()
    if p in ("anthropic", "claude"):
        return "anthropic"
    elif p in ("openai", "gpt"):
        return "openai"
    elif p in ("google", "gemini"):
        return "google"
    elif p in ("deepseek",):
        return "deepseek"
    else:
        raise ValueError(f"Unknown analyst provider: {provider!r}")


# ── Provider Implementations ─────────────────────────────────────────


async def _call_anthropic(prompt: str, cfg: AnalystConfig) -> str:
    import anthropic
    client = anthropic.AsyncAnthropic()
    resp = await asyncio.wait_for(
        client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system="You are a quantitative trading analyst. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=cfg.timeout_secs,
    )
    return resp.content[0].text if resp.content else "{}"


async def _call_openai(prompt: str, cfg: AnalystConfig) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a quantitative trading analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        ),
        timeout=cfg.timeout_secs,
    )
    return resp.choices[0].message.content or "{}"


async def _call_google(prompt: str, cfg: AnalystConfig) -> str:
    import google.generativeai as genai
    import os
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    model = genai.GenerativeModel(cfg.model)
    full_prompt = (
        "You are a quantitative trading analyst. "
        "Return only valid JSON.\n\n" + prompt
    )
    resp = await asyncio.wait_for(
        asyncio.to_thread(model.generate_content, full_prompt),
        timeout=cfg.timeout_secs,
    )
    return resp.text or "{}"


async def _call_deepseek(prompt: str, cfg: AnalystConfig) -> str:
    from openai import AsyncOpenAI
    import os
    client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com/v1",
    )
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": "You are a quantitative trading analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        ),
        timeout=cfg.timeout_secs,
    )
    return resp.choices[0].message.content or "{}"


_PROVIDER_DISPATCH = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "google": _call_google,
    "deepseek": _call_deepseek,
}


# ── AIAnalyst Class ──────────────────────────────────────────────────


class AIAnalyst:
    """Multi-provider AI analyst for bot performance analysis."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: AnalystConfig,
        bot_config: BotConfig | None = None,
    ) -> None:
        self._conn = conn
        self._config = config
        self._bot_config = bot_config

    def assemble_context(self, days: int = 30) -> AnalystContext:
        """Assemble performance data into a context for the AI prompt."""
        end = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime("%Y-%m-%d")

        ctx = AnalystContext(period_start=start, period_end=end)

        # Data quality check
        resolved_count = 0
        data_days = 0
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM performance_log "
                "WHERE date(resolved_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            resolved_count = int(row["cnt"]) if row else 0
        except sqlite3.OperationalError:
            pass

        try:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT summary_date) as cnt "
                "FROM daily_summaries WHERE summary_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            data_days = int(row["cnt"]) if row else 0
        except sqlite3.OperationalError:
            pass

        ctx.data_quality = {
            "total_resolved_trades": resolved_count,
            "data_days_available": data_days,
            "data_sufficient": (
                resolved_count >= self._config.min_resolved_trades
                and data_days >= self._config.min_data_days
            ),
        }

        if not ctx.data_quality["data_sufficient"]:
            return ctx

        # Overall summary
        try:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(total_pnl), 0) as pnl, "
                "COALESCE(MAX(drawdown_pct), 0) as dd "
                "FROM daily_summaries WHERE summary_date BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            win_row = self._conn.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins "
                "FROM performance_log WHERE date(resolved_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            total = int(win_row["total"]) if win_row else 0
            wins = int(win_row["wins"]) if win_row else 0
            wr = round(wins / total * 100, 1) if total > 0 else 0
            ctx.overall_summary = (
                f"P&L: ${float(row['pnl']):.2f}, "
                f"Win rate: {wr}% ({wins}/{total}), "
                f"Max DD: {float(row['dd']):.1%}"
            )
        except sqlite3.OperationalError:
            pass

        # Category stats
        try:
            rows = self._conn.execute(
                "SELECT f.market_type as cat, COUNT(*) as n, "
                "SUM(CASE WHEN p.pnl > 0 THEN 1 ELSE 0 END) as w, "
                "SUM(p.pnl) as pnl, AVG(f.edge) as edge "
                "FROM performance_log p "
                "JOIN forecasts f ON p.market_id = f.market_id "
                "WHERE date(p.resolved_at) BETWEEN ? AND ? "
                "GROUP BY f.market_type ORDER BY pnl DESC",
                (start, end),
            ).fetchall()
            lines = []
            for r in rows:
                n = int(r["n"])
                w = int(r["w"])
                wr = round(w / n * 100, 1) if n > 0 else 0
                lines.append(
                    f"  {r['cat'] or 'UNKNOWN'}: {n} trades, "
                    f"{wr}% wins, P&L ${float(r['pnl']):.2f}, "
                    f"edge {float(r['edge'] or 0):.1%}"
                )
            ctx.category_stats = "\n".join(lines)
        except sqlite3.OperationalError:
            pass

        # Model stats
        try:
            rows = self._conn.execute(
                "SELECT model_name, COUNT(*) as n, "
                "AVG((forecast_prob - actual_outcome) "
                "   * (forecast_prob - actual_outcome)) as brier "
                "FROM model_forecast_log "
                "WHERE actual_outcome IS NOT NULL "
                "AND date(recorded_at) BETWEEN ? AND ? "
                "GROUP BY model_name",
                (start, end),
            ).fetchall()
            lines = []
            for r in rows:
                lines.append(
                    f"  {r['model_name']}: Brier {float(r['brier'] or 0):.3f}, "
                    f"{int(r['n'])} forecasts"
                )
            ctx.model_stats = "\n".join(lines)
        except sqlite3.OperationalError:
            pass

        # Friction summary
        try:
            row = self._conn.execute(
                "SELECT AVG(f.edge) as edge, "
                "AVG(p.pnl / NULLIF(p.stake_usd, 0)) as roi "
                "FROM performance_log p "
                "JOIN forecasts f ON p.market_id = f.market_id "
                "WHERE date(p.resolved_at) BETWEEN ? AND ?",
                (start, end),
            ).fetchone()
            if row and row["edge"] is not None:
                edge = float(row["edge"])
                roi = float(row["roi"] or 0)
                ctx.friction_summary = (
                    f"Avg edge at entry: {edge:.1%}, "
                    f"Avg realised: {roi:.1%}, "
                    f"Gap: {(edge - roi):.1%}"
                )
        except sqlite3.OperationalError:
            pass

        # Config snapshot — trading config for actionable recommendations
        if self._bot_config:
            bc = self._bot_config
            ctx.config_snapshot = (
                f"Trading config: min_edge={bc.risk.min_edge}, "
                f"kelly_fraction={bc.risk.kelly_fraction}, "
                f"max_stake={bc.risk.max_stake_per_market}, "
                f"min_confidence={bc.forecasting.min_confidence_level}, "
                f"transaction_fee={bc.risk.transaction_fee_pct}, "
                f"ensemble_models={bc.ensemble.models}, "
                f"aggregation={bc.ensemble.aggregation}, "
                f"tier_routing={'on' if bc.model_tiers.enabled else 'off'}, "
                f"premium_min_edge={bc.model_tiers.premium_min_edge}"
            )
        else:
            ctx.config_snapshot = ""

        return ctx

    def _build_prompt(self, ctx: AnalystContext) -> str:
        """Build a provider-agnostic prompt with context data."""
        return f"""You are a quantitative trading analyst reviewing an automated prediction market bot.
Analyse the data below and return ONLY a JSON object matching the schema at the end.
Do not include any text outside the JSON object.

PERFORMANCE DATA (last {ctx.period_start} to {ctx.period_end}):
Overall: {ctx.overall_summary}

Category breakdown (sorted by P&L):
{ctx.category_stats}

Model accuracy (lower Brier = better):
{ctx.model_stats}

Friction: {ctx.friction_summary}

{ctx.config_snapshot}

RULES: Be specific. Reference actual categories and models from the data above.
Max 5 recommendations, ordered by expected impact.

JSON SCHEMA: {{ "summary": str, "what_is_working": [str],
  "what_is_not_working": [str], "confidence": "low"|"medium"|"high",
  "recommendations": [{{"priority": int, "action": str, "rationale": str,
  "config_change": str|null, "expected_impact": str}}] }}"""

    def _parse_response(self, raw: str) -> AnalysisResult:
        """Parse provider response, handling fences and partial JSON."""
        text = raw.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]

        # Try full JSON parse
        try:
            data = json.loads(text.strip())
            return self._deserialise(data)
        except json.JSONDecodeError:
            pass

        # Regex fallback: find first JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._deserialise(data)
            except json.JSONDecodeError:
                pass

        # Complete parse failure
        log.error(
            "ai_analyst.parse_failed",
            provider=self._config.provider,
            raw=raw[:200],
        )
        return AnalysisResult(
            summary="Analysis failed -- provider returned unparseable response.",
            data_sufficient=True,
            parse_error=True,
            raw_response=raw[:500],
        )

    def _deserialise(self, data: dict) -> AnalysisResult:
        """Convert a parsed JSON dict into an AnalysisResult."""
        recs = []
        for r in data.get("recommendations", []):
            recs.append(Recommendation(
                priority=r.get("priority", 1),
                action=r.get("action", ""),
                rationale=r.get("rationale", ""),
                config_change=r.get("config_change"),
                expected_impact=r.get("expected_impact", ""),
            ))
        return AnalysisResult(
            summary=data.get("summary", ""),
            what_is_working=data.get("what_is_working", []),
            what_is_not_working=data.get("what_is_not_working", []),
            recommendations=recs,
            confidence=data.get("confidence", "low"),
            data_sufficient=True,
        )

    def _check_rate_limit(self) -> bool:
        """Return True if a new call is allowed (rate limit not exceeded)."""
        try:
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'ai_analysis_last_call'"
            ).fetchone()
            if row:
                last_call = float(row["value"])
                elapsed_hours = (time.time() - last_call) / 3600
                if elapsed_hours < self._config.rate_limit_hours:
                    return False
        except (sqlite3.OperationalError, ValueError):
            pass
        return True

    def _record_rate_limit(self) -> None:
        """Record the timestamp of the current call."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
                "VALUES ('ai_analysis_last_call', ?, ?)",
                (str(time.time()), time.time()),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def _cache_result(self, result: AnalysisResult) -> None:
        """Cache the analysis result in engine_state."""
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO engine_state (key, value, updated_at) "
                "VALUES ('ai_analysis_result', ?, ?)",
                (json.dumps(result.to_dict()), time.time()),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass

    def get_cached_result(self) -> AnalysisResult | None:
        """Get the last cached analysis result."""
        try:
            row = self._conn.execute(
                "SELECT value FROM engine_state WHERE key = 'ai_analysis_result'"
            ).fetchone()
            if row:
                data = json.loads(row["value"])
                result = self._deserialise(data)
                result.provider_used = data.get("provider_used", "")
                result.model_used = data.get("model_used", "")
                result.generated_at = data.get("generated_at", "")
                return result
        except (sqlite3.OperationalError, json.JSONDecodeError):
            pass
        return None

    async def analyse(self, days: int = 30, alert_callback: Any | None = None) -> AnalysisResult:
        """Run a full analysis using the configured provider."""
        ctx = self.assemble_context(days)

        if not ctx.data_quality.get("data_sufficient", False):
            return AnalysisResult(
                data_sufficient=False,
                summary=(
                    f"Not enough data for analysis. "
                    f"Need {self._config.min_resolved_trades} resolved trades "
                    f"(have {ctx.data_quality.get('total_resolved_trades', 0)}) "
                    f"and {self._config.min_data_days} days "
                    f"(have {ctx.data_quality.get('data_days_available', 0)})."
                ),
            )

        if not self._check_rate_limit():
            return AnalysisResult(
                data_sufficient=True,
                summary="Rate limited. Try again later.",
                provider_used=self._config.provider,
            )

        prompt = self._build_prompt(ctx)
        provider = _route_provider(self._config.provider)
        call_fn = _PROVIDER_DISPATCH.get(provider)

        if not call_fn:
            return AnalysisResult(
                data_sufficient=True,
                summary=f"Unknown provider: {provider}",
                parse_error=True,
            )

        try:
            raw = await call_fn(prompt, self._config)
            result = self._parse_response(raw)
            result.provider_used = self._config.provider
            result.model_used = self._config.model
            result.generated_at = dt.datetime.now(dt.timezone.utc).isoformat()

            # Record rate limit and cache
            self._record_rate_limit()
            self._cache_result(result)

            # Track cost
            try:
                from src.observability.metrics import cost_tracker
                cost_tracker.record_call(
                    f"analyst-{self._config.provider}",
                    input_tokens=len(prompt.split()),
                    output_tokens=self._config.max_tokens,
                )
            except Exception:
                pass

            # Send alert with analysis summary
            if alert_callback and result.confidence in ("medium", "high"):
                try:
                    rec_lines = "\n".join(
                        f"  {i+1}. {r.action}"
                        for i, r in enumerate(result.recommendations[:3])
                    )
                    alert_callback(
                        level="info",
                        title=f"AI Analyst Complete ({result.confidence})",
                        message=(
                            f"{result.summary[:200]}\n\n"
                            f"Top recommendations:\n{rec_lines}"
                        ),
                    )
                except Exception as e:
                    log.warning("ai_analyst.alert_error", error=str(e))

            return result

        except asyncio.TimeoutError:
            return AnalysisResult(
                data_sufficient=True,
                summary=f"Analysis timed out after {self._config.timeout_secs}s.",
                provider_used=self._config.provider,
                model_used=self._config.model,
            )
        except Exception as e:
            log.error(
                "ai_analyst.call_failed",
                provider=self._config.provider,
                error=str(e),
            )
            return AnalysisResult(
                data_sufficient=True,
                summary=f"Analysis failed: {e}",
                provider_used=self._config.provider,
                model_used=self._config.model,
                parse_error=True,
            )
