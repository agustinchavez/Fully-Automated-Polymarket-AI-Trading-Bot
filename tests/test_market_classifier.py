"""Tests for src.engine.market_classifier — advanced market classification."""

from __future__ import annotations

import pytest

from src.engine.market_classifier import (
    MarketClassification,
    classify_batch,
    classify_market,
    classify_and_log,
)


# ═══════════════════════════════════════════════════════════════
#  DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════


class TestMarketClassificationDataclass:
    """MarketClassification to_dict / from_dict round-trip."""

    def test_to_dict_contains_all_fields(self):
        mc = MarketClassification(
            category="MACRO", subcategory="fed_rates",
            researchability=92, researchability_reasons=["Has schedule"],
            primary_sources=["fed.gov"], search_strategy="official_data",
            recommended_queries=8, worth_researching=True,
            confidence=0.9, tags=["scheduled_event"],
        )
        d = mc.to_dict()
        assert d["category"] == "MACRO"
        assert d["subcategory"] == "fed_rates"
        assert d["researchability"] == 92
        assert d["recommended_queries"] == 8
        assert d["worth_researching"] is True
        assert "scheduled_event" in d["tags"]

    def test_from_dict_round_trip(self):
        mc = MarketClassification(
            category="ELECTION", subcategory="presidential",
            researchability=88,
        )
        d = mc.to_dict()
        mc2 = MarketClassification.from_dict(d)
        assert mc2.category == "ELECTION"
        assert mc2.subcategory == "presidential"
        assert mc2.researchability == 88

    def test_from_dict_none_input(self):
        mc = MarketClassification.from_dict(None)
        assert mc.category == "UNKNOWN"
        assert mc.worth_researching is False

    def test_from_dict_empty_dict(self):
        mc = MarketClassification.from_dict({})
        assert mc.category == "UNKNOWN"

    def test_from_dict_partial(self):
        mc = MarketClassification.from_dict({"category": "SPORTS"})
        assert mc.category == "SPORTS"
        assert mc.subcategory == "unknown"


# ═══════════════════════════════════════════════════════════════
#  MACRO CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestMacroClassification:

    def test_fed_rate_cut(self):
        c = classify_market("Will the Fed cut interest rates in June 2025?")
        assert c.category == "MACRO"
        assert c.subcategory == "fed_rates"
        assert c.researchability >= 85

    def test_fomc(self):
        c = classify_market("Will FOMC hold rates at the December meeting?")
        assert c.category == "MACRO"
        assert c.subcategory == "fed_rates"
        assert c.worth_researching is True

    def test_cpi_inflation(self):
        c = classify_market("Will CPI come in above 3.5% for January 2025?")
        assert c.category == "MACRO"
        assert c.subcategory == "inflation"
        assert c.researchability >= 85

    def test_inflation_general(self):
        c = classify_market("Will inflation exceed 4% by end of year?")
        assert c.category == "MACRO"
        assert c.subcategory == "inflation"

    def test_gdp_growth(self):
        c = classify_market("Will Q4 GDP growth exceed 3%?")
        assert c.category == "MACRO"
        assert c.subcategory == "gdp"

    def test_unemployment(self):
        c = classify_market("Will unemployment rise above 5% by July?")
        assert c.category == "MACRO"
        assert c.subcategory == "employment"

    def test_nonfarm_payrolls(self):
        c = classify_market("Will nonfarm payrolls exceed 200K?")
        assert c.category == "MACRO"
        assert c.subcategory == "employment"

    def test_tariff(self):
        c = classify_market("Will the US impose new tariffs on China?")
        assert c.category == "MACRO"
        assert c.subcategory == "trade"

    def test_recession(self):
        c = classify_market("Will the US enter a recession in 2025?")
        assert c.category == "MACRO"
        assert c.subcategory == "recession"

    def test_treasury_yield(self):
        c = classify_market("Will the 10-year treasury yield exceed 5%?")
        assert c.category == "MACRO"
        assert c.subcategory == "bonds"

    def test_macro_has_high_queries(self):
        c = classify_market("Will the Federal Reserve announce a rate hike?")
        assert c.recommended_queries >= 6

    def test_macro_has_scheduled_event_tag(self):
        c = classify_market("Will CPI exceed expectations for March?")
        assert "scheduled_event" in c.tags or "data_release" in c.tags


# ═══════════════════════════════════════════════════════════════
#  ELECTION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestElectionClassification:

    def test_presidential(self):
        c = classify_market("Will Trump win the 2024 presidential election?")
        assert c.category == "ELECTION"
        assert c.subcategory == "presidential"
        assert c.researchability >= 80

    def test_senate(self):
        c = classify_market("Will Democrats win the Senate in 2024?")
        assert c.category == "ELECTION"
        assert c.subcategory == "congressional"

    def test_governor(self):
        c = classify_market("Will the governor of Texas sign the bill?")
        assert c.category == "ELECTION"
        assert c.subcategory == "state_local"

    def test_cabinet_appointment(self):
        c = classify_market("Will the nominee for Secretary of State be confirmed?")
        assert c.category == "ELECTION"
        assert c.subcategory == "appointments"

    def test_legislation(self):
        c = classify_market("Will the immigration bill pass the House?")
        assert c.category == "ELECTION"
        assert c.subcategory == "legislation"

    def test_general_election(self):
        c = classify_market("Will voter turnout exceed 70% in the election?")
        assert c.category == "ELECTION"
        assert c.worth_researching is True


# ═══════════════════════════════════════════════════════════════
#  CRYPTO CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestCryptoClassification:

    def test_btc_price(self):
        c = classify_market("Will Bitcoin hit $100K by June 2025?")
        assert c.category == "CRYPTO"
        assert c.subcategory == "btc_price"
        assert c.researchability >= 50

    def test_eth_price(self):
        c = classify_market("Will Ethereum price reach $5000?")
        assert c.category == "CRYPTO"
        assert c.subcategory == "eth_price"

    def test_altcoin(self):
        c = classify_market("Will Dogecoin price hit $1?")
        assert c.category == "CRYPTO"
        assert c.subcategory == "altcoin_price"
        assert c.researchability < 60

    def test_crypto_regulation(self):
        c = classify_market("Will the SEC approve a spot Bitcoin ETF?")
        assert c.category == "CRYPTO"
        assert c.subcategory == "crypto_regulation"
        assert c.researchability >= 70

    def test_crypto_event(self):
        c = classify_market("Will the Bitcoin halving happen before May?")
        assert c.category == "CRYPTO"
        assert c.subcategory == "crypto_events"

    def test_crypto_low_queries(self):
        c = classify_market("Will Solana price pump to $500?")
        assert c.recommended_queries <= 4


# ═══════════════════════════════════════════════════════════════
#  CORPORATE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestCorporateClassification:

    def test_earnings(self):
        c = classify_market("Will Apple beat earnings estimates in Q2?")
        assert c.category == "CORPORATE"
        assert c.subcategory == "earnings"
        assert c.researchability >= 80

    def test_ipo(self):
        c = classify_market("Will Stripe IPO before December 2025?")
        assert c.category == "CORPORATE"
        assert c.subcategory == "ipo"

    def test_merger(self):
        c = classify_market("Will the Microsoft-Activision merger close?")
        assert c.category == "CORPORATE"
        assert c.subcategory == "mna"

    def test_layoffs(self):
        c = classify_market("Will Google announce another layoff round?")
        assert c.category == "CORPORATE"
        assert c.subcategory == "layoffs"


# ═══════════════════════════════════════════════════════════════
#  LEGAL CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestLegalClassification:

    def test_supreme_court(self):
        c = classify_market("Will the Supreme Court rule on abortion?")
        assert c.category == "LEGAL"
        assert c.subcategory == "court_cases"
        assert c.researchability >= 70

    def test_indictment(self):
        c = classify_market("Will there be an indictment by March?")
        assert c.category == "LEGAL"
        assert c.subcategory == "criminal"

    def test_antitrust(self):
        c = classify_market("Will the FTC block the deal?")
        assert c.category == "LEGAL"
        assert c.subcategory == "regulatory"


# ═══════════════════════════════════════════════════════════════
#  SCIENCE / TECH CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestScienceTechClassification:

    def test_fda_approval(self):
        c = classify_market("Will FDA approve the new drug?")
        assert c.category == "SCIENCE"
        assert c.subcategory == "pharma"
        assert c.researchability >= 75

    def test_spacex_launch(self):
        c = classify_market("Will SpaceX Starship reach orbit by 2025?")
        assert c.category == "SCIENCE"
        assert c.subcategory == "space"

    def test_ai_company(self):
        c = classify_market("Will OpenAI release GPT-5 this year?")
        assert c.category == "TECH"
        assert c.subcategory == "ai"


# ═══════════════════════════════════════════════════════════════
#  SPORTS CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestSportsClassification:

    def test_super_bowl(self):
        c = classify_market("Will the Chiefs win the Super Bowl?")
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"
        assert c.researchability <= 55

    def test_ufc(self):
        c = classify_market("Will the UFC champion defend the title?")
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_f1(self):
        c = classify_market("Will Verstappen win the F1 championship?")
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sports_low_queries(self):
        c = classify_market("Will the NBA finals go to game 7?")
        assert c.recommended_queries <= 4


# ═══════════════════════════════════════════════════════════════
#  WEATHER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestWeatherClassification:

    def test_hurricane(self):
        c = classify_market("Will a category 5 hurricane hit Florida?")
        assert c.category == "WEATHER"
        assert c.subcategory == "severe_weather"

    def test_temperature(self):
        c = classify_market("Will the temperature exceed 110°F in Phoenix?")
        assert c.category == "WEATHER"
        assert c.subcategory == "forecast"

    def test_earthquake(self):
        c = classify_market("Will an earthquake hit California this year?")
        assert c.category == "WEATHER"
        assert c.subcategory == "natural_disaster"
        assert c.researchability < 50


# ═══════════════════════════════════════════════════════════════
#  GEOPOLITICS CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestGeopoliticsClassification:

    def test_conflict(self):
        c = classify_market("Will there be a ceasefire in Gaza?")
        assert c.category == "GEOPOLITICS"
        assert c.subcategory == "conflict"

    def test_sanctions(self):
        c = classify_market("Will the US impose new sanctions on Russia?")
        assert c.category == "GEOPOLITICS"
        assert c.subcategory == "diplomacy"


# ═══════════════════════════════════════════════════════════════
#  SOCIAL MEDIA / CULTURE — SHOULD BE BLOCKED
# ═══════════════════════════════════════════════════════════════


class TestSocialMediaCulture:

    def test_twitter_post(self):
        c = classify_market("Will Elon Musk tweet about Dogecoin today?")
        assert c.category == "SOCIAL_MEDIA"
        assert c.worth_researching is False
        assert c.researchability < 25

    def test_follower_count(self):
        c = classify_market("Will MrBeast hit 200M subscriber count?")
        assert c.category == "SOCIAL_MEDIA"
        assert c.worth_researching is False

    def test_streamer(self):
        c = classify_market("Will this Twitch streamer break a record?")
        assert c.category == "SOCIAL_MEDIA"
        assert c.worth_researching is False

    def test_celebrity(self):
        c = classify_market("Will the celebrity couple announce a breakup?")
        assert c.category == "CULTURE"
        assert c.worth_researching is False

    def test_meme_coin(self):
        c = classify_market("Will this meme coin pump 10x?")
        assert c.category == "CULTURE"
        assert c.subcategory == "novelty"
        assert c.researchability < 15

    def test_entertainment_awards(self):
        c = classify_market("Will this movie win the Oscar for best picture?")
        assert c.category == "CULTURE"
        assert c.subcategory == "entertainment"
        assert c.worth_researching is True  # awards are researchable


# ═══════════════════════════════════════════════════════════════
#  UNKNOWN / FALLBACK
# ═══════════════════════════════════════════════════════════════


class TestUnknownFallback:

    def test_unknown_question(self):
        c = classify_market("Will something random happen?")
        assert c.category == "UNKNOWN"
        assert c.subcategory == "unknown"
        assert c.worth_researching is False
        assert c.confidence < 0.5

    def test_empty_question(self):
        c = classify_market("")
        assert c.category == "UNKNOWN"
        assert c.worth_researching is False

    def test_unknown_low_queries(self):
        c = classify_market("Will the thing do the stuff?")
        assert c.recommended_queries <= 4


# ═══════════════════════════════════════════════════════════════
#  RESEARCHABILITY SCORE RANGES
# ═══════════════════════════════════════════════════════════════


class TestResearchabilityRanges:
    """Verify the relative ordering of researchability scores."""

    def test_macro_higher_than_sports(self):
        macro = classify_market("Will the Fed cut rates?")
        sports = classify_market("Will the Chiefs win the Super Bowl?")
        assert macro.researchability > sports.researchability

    def test_election_higher_than_social_media(self):
        election = classify_market("Will Biden win the election?")
        social = classify_market("Will this TikTok trend go viral?")
        assert election.researchability > social.researchability

    def test_corporate_higher_than_novelty(self):
        corp = classify_market("Will Apple beat earnings estimates?")
        novelty = classify_market("Who wins the hot dog eating contest?")
        assert corp.researchability > novelty.researchability

    def test_scheduled_events_highest(self):
        fed = classify_market("Will the FOMC raise rates?")
        assert fed.researchability >= 85


# ═══════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════


class TestConvenienceFunctions:

    def test_classify_and_log(self):
        class FakeMarket:
            id = "m1"
            question = "Will the Fed cut rates?"
            description = ""
        c = classify_and_log(FakeMarket())
        assert c.category == "MACRO"
        assert c.subcategory == "fed_rates"

    def test_classify_batch(self):
        class FM:
            def __init__(self, q):
                self.question = q
                self.description = ""
        markets = [
            FM("Will the Fed cut rates?"),
            FM("Will Bitcoin hit 100K?"),
            FM("Will the Chiefs win the Super Bowl?"),
            FM("Something random"),
        ]
        breakdown = classify_batch(markets)
        assert "MACRO" in breakdown
        assert "CRYPTO" in breakdown
        assert "SPORTS" in breakdown
        assert sum(breakdown.values()) == 4

    def test_classify_batch_empty(self):
        breakdown = classify_batch([])
        assert breakdown == {}


# ═══════════════════════════════════════════════════════════════
#  DESCRIPTION FALLBACK
# ═══════════════════════════════════════════════════════════════


class TestDescriptionFallback:
    """If question doesn't match, description text can trigger rules."""

    def test_description_triggers_match(self):
        c = classify_market(
            "Will this happen?",
            description="This market tracks the Federal Reserve interest rate decision."
        )
        assert c.category == "MACRO"

    def test_question_takes_priority_for_confidence(self):
        c = classify_market(
            "Will the Fed cut rates?",
            description="Some extra context",
        )
        assert c.confidence >= 0.8


# ═══════════════════════════════════════════════════════════════
#  PLATFORM METADATA CLASSIFICATION
# ═══════════════════════════════════════════════════════════════


class TestPlatformMetadata:
    """Platform metadata from Gamma API should take priority over regex."""

    # ── sportsMarketType ──────────────────────────────────────

    def test_sportsmarkettype_moneyline(self):
        c = classify_market(
            "Will Real Sociedad win on 2026-04-06?",
            raw={"sportsMarketType": "moneyline"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"
        assert c.confidence == 0.95
        assert "platform_classified" in c.tags

    def test_sportsmarkettype_totals(self):
        c = classify_market(
            "Stevenage FC vs. Blackpool FC: O/U 2.5",
            raw={"sportsMarketType": "totals"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_both_teams_to_score(self):
        c = classify_market(
            "Will both teams score?",
            raw={"sportsMarketType": "both_teams_to_score"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_cricket(self):
        c = classify_market(
            "T20 Kalahari Tournament: Zambia vs Mozambique - Completed match?",
            raw={"sportsMarketType": "cricket_completed_match"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_esports_kills(self):
        c = classify_market(
            "Total Kills Over/Under 27.5 in Game 1?",
            raw={"sportsMarketType": "kill_over_under_game"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_tennis_sets(self):
        c = classify_market(
            "Upper Austria Ladies Linz: Katie Boulter vs Gabriela Ruse",
            raw={"sportsMarketType": "tennis_set_totals"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_spreads(self):
        c = classify_market(
            "Lakers -4.5 vs Celtics",
            raw={"sportsMarketType": "spreads"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_unknown_sportsmarkettype_defaults_game_outcome(self):
        c = classify_market(
            "Some new sport market type",
            raw={"sportsMarketType": "some_future_type"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    # ── feeType ───────────────────────────────────────────────

    def test_feetype_sports(self):
        c = classify_market(
            "Mitch Johnson wins NBA Coach of the Year?",
            raw={"feeType": "sports_fees_v2"},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_feetype_weather(self):
        c = classify_market(
            "Will the highest temperature in LA be 68-69F?",
            raw={"feeType": "weather_fees"},
        )
        assert c.category == "WEATHER"
        assert c.subcategory == "forecast"
        assert c.confidence == 0.95

    def test_feetype_crypto(self):
        c = classify_market(
            "Will Bitcoin be above $56,000 on April 7?",
            raw={"feeType": "crypto_fees_v2"},
        )
        assert c.category == "CRYPTO"
        assert c.subcategory == "price"

    def test_feetype_culture(self):
        c = classify_market(
            "Will Elon Musk post 400 tweets?",
            raw={"feeType": "culture_fees"},
        )
        assert c.category == "CULTURE"
        assert c.confidence == 0.95

    def test_feetype_finance(self):
        c = classify_market(
            "Will WTI Crude Oil hit $20?",
            raw={"feeType": "finance_prices_fees"},
        )
        assert c.category == "MACRO"

    def test_feetype_tech(self):
        c = classify_market(
            "Will qwen3.5 be the best AI model?",
            raw={"feeType": "tech_fees"},
        )
        assert c.category == "TECH"

    def test_feetype_unknown_falls_through(self):
        """Unknown feeType should fall through to regex."""
        c = classify_market(
            "Will the Fed cut rates?",
            raw={"feeType": "general_fees"},
        )
        assert c.category == "MACRO"  # regex kicks in

    # ── Event series slug ─────────────────────────────────────

    def test_series_slug_atp(self):
        c = classify_market(
            "Some tennis match nobody can classify",
            raw={"events": [{"series": [{"slug": "atp"}]}]},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_series_slug_league_of_legends(self):
        c = classify_market(
            "BO3 Game result?",
            raw={"events": [{"series": [{"slug": "league-of-legends"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_international_cricket(self):
        c = classify_market(
            "Match completed?",
            raw={"events": [{"series": [{"slug": "international-cricket"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_mlb(self):
        c = classify_market(
            "Some baseball question",
            raw={"events": [{"series": [{"slug": "mlb"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_soccer_prefix(self):
        c = classify_market(
            "Match result?",
            raw={"events": [{"series": [{"slug": "soccer-el1"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_la_liga(self):
        c = classify_market(
            "Match winner?",
            raw={"events": [{"series": [{"slug": "la-liga-2"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_ucl(self):
        c = classify_market(
            "Champions League match",
            raw={"events": [{"series": [{"slug": "ucl-2025"}]}]},
        )
        assert c.category == "SPORTS"

    def test_series_slug_no_match_falls_through(self):
        """Non-sport series slug should fall through to regex."""
        c = classify_market(
            "Will Bitcoin hit $100K?",
            raw={"events": [{"series": [{"slug": "btc-multi-strikes-weekly"}]}]},
        )
        assert c.category == "CRYPTO"  # regex kicks in

    # ── Priority: sportsMarketType > feeType > series ─────────

    def test_sportsmarkettype_overrides_feetype(self):
        """sportsMarketType is checked first, even if feeType disagrees."""
        c = classify_market(
            "Some question",
            raw={"sportsMarketType": "moneyline", "feeType": "weather_fees"},
        )
        assert c.category == "SPORTS"

    # ── No raw dict → regex fallback ──────────────────────────

    def test_no_raw_uses_regex(self):
        c = classify_market("Will the Fed cut rates in June?")
        assert c.category == "MACRO"

    def test_empty_raw_uses_regex(self):
        c = classify_market("Will the Fed cut rates?", raw={})
        assert c.category == "MACRO"

    def test_raw_none_uses_regex(self):
        c = classify_market("Will the Fed cut rates?", raw=None)
        assert c.category == "MACRO"

    # ── Confidence is highest for platform metadata ───────────

    def test_platform_confidence_higher_than_regex(self):
        regex_result = classify_market("Will Real Sociedad win?")
        platform_result = classify_market(
            "Will Real Sociedad win?",
            raw={"sportsMarketType": "moneyline"},
        )
        assert platform_result.confidence > regex_result.confidence

    # ── classify_and_log passes raw through ───────────────────

    def test_classify_and_log_uses_raw(self):
        class FakeMarket:
            question = "Unknown obscure matchup"
            description = ""
            raw = {"sportsMarketType": "moneyline"}
            id = "test-123"

        result = classify_and_log(FakeMarket())
        assert result.category == "SPORTS"
        assert result.confidence == 0.95

    # ── New sportsMarketType values ────────────────────────────

    def test_sportsmarkettype_child_moneyline(self):
        c = classify_market("Lakers vs Celtics", raw={"sportsMarketType": "child_moneyline"})
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_tennis_first_set_totals(self):
        c = classify_market("First set O/U?", raw={"sportsMarketType": "tennis_first_set_totals"})
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_sportsmarkettype_tennis_match_totals(self):
        c = classify_market("Match total?", raw={"sportsMarketType": "tennis_match_totals"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_tennis_first_set_winner(self):
        c = classify_market("First set winner?", raw={"sportsMarketType": "tennis_first_set_winner"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_cs2_kills(self):
        c = classify_market("Odd/even kills?", raw={"sportsMarketType": "cs2_odd_even_total_kills"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_cs2_rounds(self):
        c = classify_market("Odd/even rounds?", raw={"sportsMarketType": "cs2_odd_even_total_rounds"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_map_handicap(self):
        c = classify_market("Map handicap?", raw={"sportsMarketType": "map_handicap"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_dota2_rampage(self):
        c = classify_market("Will there be a rampage?", raw={"sportsMarketType": "dota2_rampage"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_dota2_ultra_kill(self):
        c = classify_market("Ultra kill?", raw={"sportsMarketType": "dota2_ultra_kill"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_dota2_daytime(self):
        c = classify_market("Game ends daytime?", raw={"sportsMarketType": "dota2_game_ends_daytime"})
        assert c.category == "SPORTS"

    def test_sportsmarkettype_dota2_barracks(self):
        c = classify_market("Both barracks?", raw={"sportsMarketType": "dota2_both_teams_barracks"})
        assert c.category == "SPORTS"

    # ── Event category (legacy markets) ────────────────────────

    def test_event_category_us_current_affairs(self):
        c = classify_market(
            "Will Congress pass the spending bill?",
            raw={"events": [{"category": "US-current-affairs"}]},
        )
        assert c.category == "ELECTION"
        assert c.subcategory == "general"
        assert c.confidence == 0.95
        assert "platform_classified" in c.tags

    def test_event_category_global_politics(self):
        c = classify_market(
            "Will the UK PM resign?",
            raw={"events": [{"category": "Global Politics"}]},
        )
        assert c.category == "ELECTION"
        assert c.subcategory == "international"

    def test_event_category_sports(self):
        c = classify_market(
            "Who wins the match?",
            raw={"events": [{"category": "Sports"}]},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_event_category_nba_playoffs(self):
        c = classify_market(
            "NBA playoff game result",
            raw={"events": [{"category": "NBA Playoffs"}]},
        )
        assert c.category == "SPORTS"
        assert c.subcategory == "game_outcome"

    def test_event_category_crypto(self):
        c = classify_market(
            "Will ETH hit $5000?",
            raw={"events": [{"category": "Crypto"}]},
        )
        assert c.category == "CRYPTO"
        assert c.subcategory == "price"

    def test_event_category_nfts(self):
        c = classify_market(
            "Will this NFT sell for $100k?",
            raw={"events": [{"category": "NFTs"}]},
        )
        assert c.category == "CRYPTO"
        assert c.subcategory == "nft"

    def test_event_category_business(self):
        c = classify_market(
            "What will Airbnb market cap be?",
            raw={"events": [{"category": "Business"}]},
        )
        assert c.category == "CORPORATE"

    def test_event_category_tech(self):
        c = classify_market(
            "Will GPT-5 launch this year?",
            raw={"events": [{"category": "Tech"}]},
        )
        assert c.category == "TECH"

    def test_event_category_science(self):
        c = classify_market(
            "Will they find evidence of life on Mars?",
            raw={"events": [{"category": "Science"}]},
        )
        assert c.category == "SCIENCE"

    def test_event_category_coronavirus(self):
        c = classify_market(
            "Will COVID cases spike?",
            raw={"events": [{"category": "Coronavirus"}]},
        )
        assert c.category == "SCIENCE"
        assert c.subcategory == "pandemic"

    def test_event_category_space(self):
        c = classify_market(
            "Will SpaceX land on Mars?",
            raw={"events": [{"category": "Space"}]},
        )
        assert c.category == "SCIENCE"
        assert c.subcategory == "space"

    def test_event_category_pop_culture(self):
        c = classify_market(
            "Will Bad Bunny release an album?",
            raw={"events": [{"category": "Pop-Culture "}]},
        )
        assert c.category == "SOCIAL_MEDIA"

    def test_event_category_chess(self):
        c = classify_market(
            "Who wins the chess championship?",
            raw={"events": [{"category": "Chess"}]},
        )
        assert c.category == "SPORTS"

    def test_event_category_olympics(self):
        c = classify_market(
            "Who wins gold in the 100m?",
            raw={"events": [{"category": "Olympics"}]},
        )
        assert c.category == "SPORTS"

    def test_event_category_unknown_falls_through(self):
        """Unknown event category should fall through to regex."""
        c = classify_market(
            "Will the Fed cut rates in June?",
            raw={"events": [{"category": "SomeNewCategory"}]},
        )
        assert c.category == "MACRO"  # regex kicks in

    # ── Priority: feeType > event.category ─────────────────────

    def test_feetype_overrides_event_category(self):
        """feeType is checked before event.category."""
        c = classify_market(
            "Some question",
            raw={
                "feeType": "crypto_fees_v2",
                "events": [{"category": "US-current-affairs"}],
            },
        )
        assert c.category == "CRYPTO"  # feeType wins

    def test_series_slug_overrides_event_category(self):
        """Series slug is checked before event.category."""
        c = classify_market(
            "Match result?",
            raw={
                "events": [{
                    "category": "Tech",
                    "series": [{"slug": "nba"}],
                }],
            },
        )
        assert c.category == "SPORTS"  # series slug wins

    # ── Fix B: feeType mapping corrections ────────────────────────

    def test_finance_prices_fees_maps_to_macro(self):
        """finance_prices_fees → MACRO/commodity (not CORPORATE)."""
        c = classify_market(
            "Will Natural Gas (NG) hit $2.00?",
            raw={"feeType": "finance_prices_fees"},
        )
        assert c.category == "MACRO"
        assert c.subcategory == "commodity"

    def test_culture_fees_maps_to_culture(self):
        """culture_fees → CULTURE/entertainment (not SOCIAL_MEDIA)."""
        c = classify_market(
            "Will The Weeknd have the most monthly Spotify listeners?",
            raw={"feeType": "culture_fees"},
        )
        assert c.category == "CULTURE"
        assert c.subcategory == "entertainment"

    # ── Fix C: tag-based classification ────────────────────────────

    def test_geopolitics_tag_classification(self):
        """Event with tags=[{slug:'geopolitics'}] → GEOPOLITICS."""
        c = classify_market(
            "Will Israel take military action in Gaza?",
            raw={
                "events": [{
                    "tags": [{"slug": "geopolitics"}],
                }],
            },
        )
        assert c.category == "GEOPOLITICS"
        assert c.subcategory == "international"

    def test_middle_east_tag_classification(self):
        """Event with tags=[{slug:'middle-east'}] → GEOPOLITICS/conflict."""
        c = classify_market(
            "Will Israel conduct military action in Beirut?",
            raw={
                "events": [{
                    "tags": [{"slug": "middle-east"}],
                }],
            },
        )
        assert c.category == "GEOPOLITICS"
        assert c.subcategory == "conflict"

    def test_science_tag_classification(self):
        """Event with tags=[{slug:'science'}] → SCIENCE/general."""
        c = classify_market(
            "Will there be exactly 2 earthquakes of magnitude 6.5?",
            raw={
                "events": [{
                    "tags": [{"slug": "science"}],
                }],
            },
        )
        assert c.category == "SCIENCE"
        assert c.subcategory == "general"

    def test_earthquakes_tag_classification(self):
        """Event with tags=[{slug:'earthquakes'}] → SCIENCE/seismic."""
        c = classify_market(
            "How many earthquakes will there be?",
            raw={
                "events": [{
                    "tags": [{"slug": "earthquakes"}],
                }],
            },
        )
        assert c.category == "SCIENCE"
        assert c.subcategory == "seismic"

    def test_culture_music_tag_classification(self):
        """Event with tags=[{slug:'music'}] → CULTURE/music."""
        c = classify_market(
            "Will Spotify listeners reach 100M?",
            raw={
                "events": [{
                    "tags": [{"slug": "music"}],
                }],
            },
        )
        assert c.category == "CULTURE"
        assert c.subcategory == "music"

    def test_elections_tag_classification(self):
        """Event with tags=[{slug:'elections'}] → ELECTION/general."""
        c = classify_market(
            "Will voter turnout be 74-77%?",
            raw={
                "events": [{
                    "tags": [{"slug": "elections"}],
                }],
            },
        )
        assert c.category == "ELECTION"
        assert c.subcategory == "general"

    def test_economics_tag_classification(self):
        """Event with tags=[{slug:'economics'}] → MACRO/economic_data."""
        c = classify_market(
            "Will GDP growth exceed 3%?",
            raw={
                "events": [{
                    "tags": [{"slug": "economics"}],
                }],
            },
        )
        assert c.category == "MACRO"
        assert c.subcategory == "economic_data"

    def test_commodities_tag_classification(self):
        """Event with tags=[{slug:'commodities'}] → MACRO/commodity."""
        c = classify_market(
            "Will WTI Crude Oil hit $150?",
            raw={
                "events": [{
                    "tags": [{"slug": "commodities"}],
                }],
            },
        )
        assert c.category == "MACRO"
        assert c.subcategory == "commodity"

    def test_tag_fallback_after_higher_priority(self):
        """Tags are only checked when feeType/sportsMarketType/series/category are absent."""
        c = classify_market(
            "Match?",
            raw={
                "feeType": "sports_fees_v2",
                "events": [{"tags": [{"slug": "science"}]}],
            },
        )
        assert c.category == "SPORTS"  # feeType wins over tags

    def test_tag_label_fallback(self):
        """Tags can use 'label' when 'slug' is missing."""
        c = classify_market(
            "Military operation?",
            raw={
                "events": [{
                    "tags": [{"label": "Geopolitics"}],
                }],
            },
        )
        assert c.category == "GEOPOLITICS"


# ═══════════════════════════════════════════════════════════════
#  CODE REVIEW V11 TESTS
# ═══════════════════════════════════════════════════════════════


class TestCommodityRegex:
    """Bug 3: Commodity regex in rich classifier (no platform metadata)."""

    def test_natural_gas_without_metadata(self):
        c = classify_market("Will Natural Gas (NG) hit $2.00?")
        assert c.category == "MACRO"
        assert c.subcategory == "commodity"

    def test_crude_oil_without_metadata(self):
        c = classify_market("Will WTI Crude Oil hit $150?")
        assert c.category == "MACRO"
        assert c.subcategory == "commodity"

    def test_gold_price_without_metadata(self):
        c = classify_market("Will gold price exceed $3000?")
        assert c.category == "MACRO"
        assert c.subcategory == "commodity"

    def test_commodity_researchability(self):
        c = classify_market("Will Brent crude hit $100?")
        assert c.researchability == 75
        assert "EIA" in c.primary_sources


class TestStreamingRegex:
    """Opt 1: Streaming/Spotify regex rule for CULTURE."""

    def test_spotify_streaming(self):
        c = classify_market("Will The Weeknd have the most Spotify monthly listeners?")
        assert c.category == "CULTURE"
        assert c.subcategory == "streaming"

    def test_billboard_charts(self):
        c = classify_market("Will Olivia Rodrigo reach number one on Billboard?")
        assert c.category == "CULTURE"
        assert c.subcategory == "streaming"

    def test_streaming_researchability(self):
        c = classify_market("Will Spotify top artist change this month?")
        assert c.researchability == 60
        assert c.search_strategy == "news_analysis"


class TestCultureConnectorRelevance:
    """Bug 2: CULTURE must be in broad connector relevant_categories."""

    def test_manifold_includes_culture(self):
        from src.research.connectors.manifold import ManifoldConnector
        c = ManifoldConnector()
        assert "CULTURE" in c.relevant_categories()

    def test_metaculus_includes_culture(self):
        from src.research.connectors.metaculus import MetaculusConnector
        c = MetaculusConnector()
        assert "CULTURE" in c.relevant_categories()

    def test_wikipedia_includes_culture(self):
        from src.research.connectors.wikipedia_pageviews import WikipediaPageviewsConnector
        c = WikipediaPageviewsConnector()
        assert "CULTURE" in c.relevant_categories()

    def test_kalshi_includes_culture(self):
        from src.research.connectors.kalshi_prior import KalshiPriorConnector
        c = KalshiPriorConnector()
        assert "CULTURE" in c.relevant_categories()

    def test_google_trends_includes_culture(self):
        from src.research.connectors.google_trends import GoogleTrendsConnector
        c = GoogleTrendsConnector()
        assert "CULTURE" in c.relevant_categories()

    def test_reddit_includes_culture(self):
        from src.research.connectors.reddit_sentiment import RedditSentimentConnector
        c = RedditSentimentConnector()
        assert "CULTURE" in c.relevant_categories()
