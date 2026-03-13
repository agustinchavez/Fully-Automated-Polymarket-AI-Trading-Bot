"""Base rate registry for superforecasting-style anchoring.

Provides historical base rates for common prediction market question
patterns. LLMs are notoriously bad at base rates — forcing them to
anchor on historical frequency data before adjusting improves
calibration significantly.

Features:
  - 50+ seed patterns with regex matching across 7 categories
  - Self-updating empirical rates from resolved markets
  - Category-level fallback when no pattern matches
  - Confidence scoring for match quality
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class BaseRateMatch:
    """Result of matching a question to a base rate pattern."""
    base_rate: float
    pattern_description: str
    source: str
    confidence: float  # 0.0–1.0, how well the question matches
    category: str = ""
    sample_size: int = 0


@dataclass
class BaseRatePattern:
    """A single base rate pattern entry."""
    pattern: str  # regex pattern
    category: str
    description: str
    base_rate: float
    source: str
    sample_size: int = 0
    _compiled: re.Pattern | None = field(default=None, repr=False)

    def compile(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, re.IGNORECASE)
        return self._compiled


# ── Seed patterns (50+) ─────────────────────────────────────────────

_SEED_PATTERNS: list[dict[str, Any]] = [
    # ── MACRO (12 patterns) ──────────────────────────────────────────
    {
        "pattern": r"(fed|federal reserve).*(cut|lower|reduce|decrease).*rate",
        "category": "MACRO",
        "description": "Fed cuts rates at any given meeting",
        "base_rate": 0.25,
        "source": "FOMC historical decisions 1990-2024",
        "sample_size": 270,
    },
    {
        "pattern": r"(fed|federal reserve).*(raise|hike|increase).*rate",
        "category": "MACRO",
        "description": "Fed raises rates at any given meeting",
        "base_rate": 0.30,
        "source": "FOMC historical decisions 1990-2024",
        "sample_size": 270,
    },
    {
        "pattern": r"(fed|federal reserve).*(hold|pause|maintain|unchanged).*rate",
        "category": "MACRO",
        "description": "Fed holds rates at any given meeting",
        "base_rate": 0.45,
        "source": "FOMC historical decisions 1990-2024",
        "sample_size": 270,
    },
    {
        "pattern": r"gdp.*(beat|exceed|above|surpass).*estimate",
        "category": "MACRO",
        "description": "GDP growth beats consensus estimate",
        "base_rate": 0.55,
        "source": "BEA quarterly GDP vs consensus 2000-2024",
        "sample_size": 96,
    },
    {
        "pattern": r"inflation.*(above|exceed|over|higher).*target",
        "category": "MACRO",
        "description": "Inflation runs above central bank target",
        "base_rate": 0.40,
        "source": "CPI vs 2% target 1990-2024",
        "sample_size": 408,
    },
    {
        "pattern": r"recession.*within.*(year|12 month|next year)",
        "category": "MACRO",
        "description": "Economy enters recession within 12 months",
        "base_rate": 0.15,
        "source": "NBER recession dating 1950-2024",
        "sample_size": 74,
    },
    {
        "pattern": r"unemployment.*(rise|increase|above|exceed|over).*\d",
        "category": "MACRO",
        "description": "Unemployment rate rises above threshold",
        "base_rate": 0.35,
        "source": "BLS unemployment data 1990-2024",
        "sample_size": 408,
    },
    {
        "pattern": r"(cpi|consumer price).*(fall|drop|below|decrease|decline)",
        "category": "MACRO",
        "description": "CPI falls below threshold",
        "base_rate": 0.30,
        "source": "BLS CPI data 2000-2024",
        "sample_size": 288,
    },
    {
        "pattern": r"(treasury|bond) yield.*(rise|above|exceed|over|increase)",
        "category": "MACRO",
        "description": "Treasury yield rises above threshold",
        "base_rate": 0.45,
        "source": "US Treasury yield data 2000-2024",
        "sample_size": 288,
    },
    {
        "pattern": r"(s&p|s&p 500|stock market|dow|nasdaq).*(crash|drop|fall|decline).*\d+%",
        "category": "MACRO",
        "description": "Major stock index drops by significant percentage",
        "base_rate": 0.10,
        "source": "S&P 500 drawdown history 1950-2024",
        "sample_size": 74,
    },
    {
        "pattern": r"(government|federal) shutdown",
        "category": "MACRO",
        "description": "Government shutdown occurs",
        "base_rate": 0.20,
        "source": "US government shutdown history 1976-2024",
        "sample_size": 48,
    },
    {
        "pattern": r"(trade war|tariff).*(escalat|increas|impose|new)",
        "category": "MACRO",
        "description": "Trade tensions escalate with new tariffs",
        "base_rate": 0.35,
        "source": "US trade policy actions 2000-2024",
        "sample_size": 50,
    },

    # ── ELECTION (10 patterns) ───────────────────────────────────────
    {
        "pattern": r"(incumbent|sitting).*(win|reelect|re-elect|victory)",
        "category": "ELECTION",
        "description": "Incumbent wins reelection",
        "base_rate": 0.60,
        "source": "US presidential elections 1900-2024",
        "sample_size": 31,
    },
    {
        "pattern": r"(republican|democrat|gop|dem).*(win|take|flip|retain).*senate",
        "category": "ELECTION",
        "description": "Party wins/retains Senate control",
        "base_rate": 0.50,
        "source": "US Senate elections 1980-2024",
        "sample_size": 22,
    },
    {
        "pattern": r"(republican|democrat|gop|dem).*(win|take|flip|retain).*house",
        "category": "ELECTION",
        "description": "Party wins/retains House control",
        "base_rate": 0.50,
        "source": "US House elections 1980-2024",
        "sample_size": 22,
    },
    {
        "pattern": r"approval.*(above|over|exceed|higher).*50",
        "category": "ELECTION",
        "description": "Leader approval rating above 50%",
        "base_rate": 0.35,
        "source": "Gallup presidential approval 1945-2024",
        "sample_size": 950,
    },
    {
        "pattern": r"(win|victory).*(margin|lead).*(above|over|more|exceed).*5",
        "category": "ELECTION",
        "description": "Candidate wins by more than 5 points",
        "base_rate": 0.40,
        "source": "US presidential election margins 1900-2024",
        "sample_size": 31,
    },
    {
        "pattern": r"(primary|nomination).*(win|lead|front)",
        "category": "ELECTION",
        "description": "Frontrunner wins party nomination",
        "base_rate": 0.70,
        "source": "US primary elections 1972-2024",
        "sample_size": 26,
    },
    {
        "pattern": r"(electoral college|270|electoral votes)",
        "category": "ELECTION",
        "description": "Candidate reaches 270 electoral votes",
        "base_rate": 0.50,
        "source": "US presidential elections (binary)",
        "sample_size": 31,
    },
    {
        "pattern": r"(swing state|battleground).*(flip|win|go to)",
        "category": "ELECTION",
        "description": "Swing state flips to other party",
        "base_rate": 0.35,
        "source": "US swing state elections 2000-2024",
        "sample_size": 42,
    },
    {
        "pattern": r"(governor|mayor|senator).*(win|elect|reelect)",
        "category": "ELECTION",
        "description": "Specific candidate wins election",
        "base_rate": 0.50,
        "source": "US elections (generic binary)",
        "sample_size": 100,
    },
    {
        "pattern": r"(referendum|ballot measure|proposition).*(pass|approve|yes)",
        "category": "ELECTION",
        "description": "Ballot measure passes",
        "base_rate": 0.45,
        "source": "US ballot measures 2000-2024",
        "sample_size": 200,
    },

    # ── CORPORATE (10 patterns) ──────────────────────────────────────
    {
        "pattern": r"(earnings|revenue|profit).*(beat|exceed|above|surpass).*estimate",
        "category": "CORPORATE",
        "description": "Company beats earnings estimates",
        "base_rate": 0.70,
        "source": "S&P 500 earnings surprise rate 2000-2024",
        "sample_size": 12000,
    },
    {
        "pattern": r"(ceo|chief executive).*(resign|step down|depart|fired|replaced)",
        "category": "CORPORATE",
        "description": "CEO departure within specified period",
        "base_rate": 0.08,
        "source": "S&P 500 CEO turnover rate annualized",
        "sample_size": 500,
    },
    {
        "pattern": r"(ipo|initial public offering).*(above|over|exceed|higher).*offer",
        "category": "CORPORATE",
        "description": "IPO prices above offer price on first day",
        "base_rate": 0.65,
        "source": "US IPO first-day returns 2000-2024",
        "sample_size": 3000,
    },
    {
        "pattern": r"(merger|acquisition|deal|takeover).*(complet|clos|approv|finalize)",
        "category": "CORPORATE",
        "description": "Announced merger/acquisition completes",
        "base_rate": 0.85,
        "source": "US M&A completion rates 2000-2024",
        "sample_size": 5000,
    },
    {
        "pattern": r"(layoff|restructur|job cut|workforce reduction)",
        "category": "CORPORATE",
        "description": "Major company announces layoffs",
        "base_rate": 0.40,
        "source": "S&P 500 restructuring events per year",
        "sample_size": 500,
    },
    {
        "pattern": r"(stock|share).*(split|buyback|repurchase)",
        "category": "CORPORATE",
        "description": "Company announces stock split or buyback",
        "base_rate": 0.25,
        "source": "S&P 500 corporate actions per year",
        "sample_size": 500,
    },
    {
        "pattern": r"(dividend).*(cut|reduce|suspend|eliminat)",
        "category": "CORPORATE",
        "description": "Company cuts or suspends dividend",
        "base_rate": 0.05,
        "source": "S&P 500 dividend actions 2000-2024",
        "sample_size": 12000,
    },
    {
        "pattern": r"(bankrupt|insolvency|chapter 11|chapter 7)",
        "category": "CORPORATE",
        "description": "Company files for bankruptcy",
        "base_rate": 0.02,
        "source": "US corporate bankruptcy rates annualized",
        "sample_size": 5000,
    },
    {
        "pattern": r"(product|service|app).*(launch|release|ship).*on time",
        "category": "CORPORATE",
        "description": "Product launches on announced schedule",
        "base_rate": 0.60,
        "source": "Tech product launch timing 2015-2024",
        "sample_size": 200,
    },
    {
        "pattern": r"(stock|share price).*(above|over|reach|exceed|hit).*\$\d+",
        "category": "CORPORATE",
        "description": "Stock reaches specific price target",
        "base_rate": 0.45,
        "source": "Analyst price target hit rates 2010-2024",
        "sample_size": 5000,
    },

    # ── LEGAL (6 patterns) ───────────────────────────────────────────
    {
        "pattern": r"supreme court.*(overturn|strike|reverse|invalidat)",
        "category": "LEGAL",
        "description": "Supreme Court overturns precedent",
        "base_rate": 0.03,
        "source": "SCOTUS reversal rate historical",
        "sample_size": 500,
    },
    {
        "pattern": r"(conviction|guilty|convicted|found guilty)",
        "category": "LEGAL",
        "description": "Criminal defendant convicted",
        "base_rate": 0.90,
        "source": "US federal criminal conviction rate",
        "sample_size": 10000,
    },
    {
        "pattern": r"(antitrust|anti-trust|monopoly).*(action|suit|case|block)",
        "category": "LEGAL",
        "description": "Antitrust action succeeds",
        "base_rate": 0.30,
        "source": "DOJ/FTC antitrust outcomes 2000-2024",
        "sample_size": 100,
    },
    {
        "pattern": r"(indict|indictment|charged|criminal charge)",
        "category": "LEGAL",
        "description": "Public figure indicted/charged",
        "base_rate": 0.20,
        "source": "Historical indictment rates for public figures",
        "sample_size": 50,
    },
    {
        "pattern": r"(regulat|regulation|fda|sec|ftc).*(approv|pass|clear)",
        "category": "LEGAL",
        "description": "Regulatory approval granted",
        "base_rate": 0.65,
        "source": "FDA/SEC approval rates 2000-2024",
        "sample_size": 1000,
    },
    {
        "pattern": r"(ban|prohibit|outlaw|restrict).*(pass|enact|implement|enforc)",
        "category": "LEGAL",
        "description": "Proposed ban or restriction enacted",
        "base_rate": 0.25,
        "source": "US legislative ban proposals vs enactments",
        "sample_size": 200,
    },

    # ── TECHNOLOGY (6 patterns) ──────────────────────────────────────
    {
        "pattern": r"(ai|artificial intelligence).*(regulat|ban|restrict|law|legislat)",
        "category": "TECHNOLOGY",
        "description": "AI regulation or restriction enacted",
        "base_rate": 0.20,
        "source": "AI policy proposals vs enactments 2020-2024",
        "sample_size": 50,
    },
    {
        "pattern": r"(tech stock|technology).*(beat|outperform|exceed).*s&p",
        "category": "TECHNOLOGY",
        "description": "Tech sector beats S&P 500 in given year",
        "base_rate": 0.55,
        "source": "NASDAQ vs S&P 500 annual returns 1990-2024",
        "sample_size": 34,
    },
    {
        "pattern": r"(crypto|bitcoin|ethereum|btc|eth).*(above|over|reach|exceed|hit).*\$",
        "category": "TECHNOLOGY",
        "description": "Cryptocurrency reaches price target",
        "base_rate": 0.40,
        "source": "Crypto price target achievement 2015-2024",
        "sample_size": 200,
    },
    {
        "pattern": r"(data breach|hack|cyber attack|cybersecurity incident)",
        "category": "TECHNOLOGY",
        "description": "Major data breach at specified company",
        "base_rate": 0.15,
        "source": "Fortune 500 data breach frequency annualized",
        "sample_size": 500,
    },
    {
        "pattern": r"(twitter|x\.com|social media).*(ban|suspend|censor|restrict).*account",
        "category": "TECHNOLOGY",
        "description": "Social media platform takes specific moderation action",
        "base_rate": 0.30,
        "source": "Social media moderation actions 2020-2024",
        "sample_size": 100,
    },
    {
        "pattern": r"(self.driving|autonomous vehicle|driverless).*(approv|launch|deploy)",
        "category": "TECHNOLOGY",
        "description": "Autonomous vehicle approval/deployment milestone",
        "base_rate": 0.35,
        "source": "AV regulatory milestones 2018-2024",
        "sample_size": 30,
    },

    # ── SPORTS (6 patterns) ──────────────────────────────────────────
    {
        "pattern": r"(home team|home.*(win|advantage))",
        "category": "SPORTS",
        "description": "Home team wins the game",
        "base_rate": 0.55,
        "source": "Major US sports home win rate composite",
        "sample_size": 50000,
    },
    {
        "pattern": r"(favorite|favoured).*(win|cover|beat).*spread",
        "category": "SPORTS",
        "description": "Favorite covers the spread",
        "base_rate": 0.50,
        "source": "NFL/NBA spread coverage rate (by design ~50%)",
        "sample_size": 50000,
    },
    {
        "pattern": r"(over|total).*(hit|exceed|above|over).*\d+",
        "category": "SPORTS",
        "description": "Game total goes over set line",
        "base_rate": 0.50,
        "source": "NFL/NBA over/under hit rate (by design ~50%)",
        "sample_size": 50000,
    },
    {
        "pattern": r"(champion|championship|title|trophy|cup).*(win|defend|retain)",
        "category": "SPORTS",
        "description": "Defending champion retains title",
        "base_rate": 0.25,
        "source": "Major US sports repeat championship rate",
        "sample_size": 200,
    },
    {
        "pattern": r"(mvp|most valuable|award).*(win|receive|earn)",
        "category": "SPORTS",
        "description": "Specific player wins MVP/award",
        "base_rate": 0.15,
        "source": "Preseason MVP favorites conversion rate",
        "sample_size": 100,
    },
    {
        "pattern": r"(playoff|postseason|qualify|make.*(playoffs|postseason))",
        "category": "SPORTS",
        "description": "Team makes playoffs",
        "base_rate": 0.50,
        "source": "NFL/NBA/MLB playoff qualification rates",
        "sample_size": 5000,
    },

    # ── GENERAL (6 patterns) ─────────────────────────────────────────
    {
        "pattern": r"will .+ (by|before) (end of |)(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})",
        "category": "GENERAL",
        "description": "Event occurs by specific deadline",
        "base_rate": 0.30,
        "source": "Polymarket resolved markets with deadlines",
        "sample_size": 1000,
    },
    {
        "pattern": r"(war|conflict|military|invasion|attack) .*(start|begin|launch|escalat)",
        "category": "GENERAL",
        "description": "Military conflict starts or escalates",
        "base_rate": 0.15,
        "source": "Geopolitical conflict onset frequency",
        "sample_size": 100,
    },
    {
        "pattern": r"(cease.?fire|peace|treaty|agreement).*(sign|reach|achiev)",
        "category": "GENERAL",
        "description": "Peace agreement or ceasefire reached",
        "base_rate": 0.25,
        "source": "Conflict resolution outcomes historical",
        "sample_size": 100,
    },
    {
        "pattern": r"(pandemic|epidemic|outbreak|disease).*(spread|reach|infect)",
        "category": "GENERAL",
        "description": "Disease outbreak reaches specified threshold",
        "base_rate": 0.20,
        "source": "WHO disease outbreak data 2000-2024",
        "sample_size": 50,
    },
    {
        "pattern": r"(natural disaster|earthquake|hurricane|flood|wildfire)",
        "category": "GENERAL",
        "description": "Major natural disaster occurs",
        "base_rate": 0.25,
        "source": "FEMA/NOAA disaster frequency data",
        "sample_size": 200,
    },
    {
        "pattern": r"(sanction|embargo).*(impose|expand|lift|remov)",
        "category": "GENERAL",
        "description": "International sanctions imposed or modified",
        "base_rate": 0.35,
        "source": "US/EU sanctions actions 2000-2024",
        "sample_size": 100,
    },
]

# Category-level fallback base rates
_CATEGORY_BASE_RATES: dict[str, float] = {
    "MACRO": 0.40,
    "ELECTION": 0.50,
    "CORPORATE": 0.50,
    "LEGAL": 0.40,
    "TECHNOLOGY": 0.40,
    "SPORTS": 0.50,
    "WEATHER": 0.35,
    "SCIENCE": 0.40,
    "GENERAL": 0.35,
    "UNKNOWN": 0.50,
}


class BaseRateRegistry:
    """Registry of historical base rates for prediction market anchoring.

    Matches market questions against regex patterns to find relevant
    base rates. Supports self-updating from resolved market data.
    """

    def __init__(self) -> None:
        self._patterns: list[BaseRatePattern] = []
        self._empirical_rates: dict[str, dict[str, Any]] = {}
        self._load_seed_patterns()

    def _load_seed_patterns(self) -> None:
        """Load seed patterns from the built-in table."""
        for entry in _SEED_PATTERNS:
            self._patterns.append(BaseRatePattern(**entry))

    @property
    def patterns(self) -> list[BaseRatePattern]:
        """All registered patterns."""
        return list(self._patterns)

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    def match(self, question: str, category: str = "") -> BaseRateMatch | None:
        """Find the best matching base rate pattern for a question.

        Args:
            question: The market question to match.
            category: Optional market category to narrow matching.

        Returns:
            BaseRateMatch if a pattern matches, None otherwise.
        """
        if not question:
            return None

        best_match: BaseRateMatch | None = None
        best_score: float = 0.0

        for pat in self._patterns:
            # Category filter: prefer same-category patterns
            category_bonus = 0.1 if (category and pat.category == category) else 0.0

            try:
                compiled = pat.compile()
                m = compiled.search(question)
                if m:
                    # Score based on match span relative to question length
                    match_len = m.end() - m.start()
                    match_ratio = match_len / max(len(question), 1)
                    # Base confidence from match quality
                    confidence = min(0.95, 0.5 + match_ratio + category_bonus)

                    score = confidence
                    if score > best_score:
                        best_score = score
                        best_match = BaseRateMatch(
                            base_rate=pat.base_rate,
                            pattern_description=pat.description,
                            source=pat.source,
                            confidence=round(confidence, 3),
                            category=pat.category,
                            sample_size=pat.sample_size,
                        )
            except re.error:
                continue

        # Check empirical rates (from resolved markets)
        if category and category in self._empirical_rates:
            emp = self._empirical_rates[category]
            if emp.get("sample_size", 0) >= 30:
                emp_confidence = min(0.8, emp["sample_size"] / 100)
                # Empirical rate beats pattern match only if confidence is higher
                if emp_confidence > best_score:
                    best_match = BaseRateMatch(
                        base_rate=emp["rate"],
                        pattern_description=f"Empirical rate for {category}",
                        source=f"Resolved markets (n={emp['sample_size']})",
                        confidence=round(emp_confidence, 3),
                        category=category,
                        sample_size=emp["sample_size"],
                    )

        if best_match:
            log.info(
                "base_rate.matched",
                question=question[:80],
                base_rate=best_match.base_rate,
                pattern=best_match.pattern_description,
                confidence=best_match.confidence,
            )

        return best_match

    def get_category_base_rate(self, category: str) -> float | None:
        """Get a simple category-level base rate as fallback.

        Returns None if the category is not recognized.
        """
        # Check empirical first
        if category in self._empirical_rates:
            emp = self._empirical_rates[category]
            if emp.get("sample_size", 0) >= 30:
                return emp["rate"]

        return _CATEGORY_BASE_RATES.get(category)

    def update_from_resolved(
        self, category: str, resolution_rate: float, sample_size: int,
    ) -> None:
        """Update empirical base rates from resolved market data.

        Should be called after accumulating 30+ resolved markets in a
        category. The empirical rate supplements (and can override)
        seed patterns.

        Args:
            category: Market category.
            resolution_rate: Fraction of markets that resolved YES.
            sample_size: Number of resolved markets.
        """
        self._empirical_rates[category] = {
            "rate": round(max(0.01, min(0.99, resolution_rate)), 4),
            "sample_size": sample_size,
        }
        log.info(
            "base_rate.empirical_update",
            category=category,
            rate=round(resolution_rate, 4),
            sample_size=sample_size,
        )

    def get_all_patterns(self) -> list[dict[str, Any]]:
        """Return all patterns as dicts (for API/dashboard)."""
        result = []
        for p in self._patterns:
            result.append({
                "pattern": p.pattern,
                "category": p.category,
                "description": p.description,
                "base_rate": p.base_rate,
                "source": p.source,
                "sample_size": p.sample_size,
            })
        return result

    def get_empirical_rates(self) -> dict[str, dict[str, Any]]:
        """Return all empirical rates (for API/dashboard)."""
        return dict(self._empirical_rates)
