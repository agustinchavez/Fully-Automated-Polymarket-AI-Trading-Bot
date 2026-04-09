"""Connector registry — instantiate enabled research connectors from config.

Reads the ``*_enabled`` flags on ``ResearchConfig`` and returns a list
of ready-to-use connector instances.  Called from ``SourceFetcher``.
"""

from __future__ import annotations

from typing import Any

from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector

log = get_logger(__name__)


def get_enabled_connectors(config: Any) -> list[BaseResearchConnector]:
    """Return list of enabled connector instances based on config flags.

    Parameters
    ----------
    config:
        The ``ResearchConfig`` (or any object) — checked for ``*_enabled`` attrs.
    """
    connectors: list[BaseResearchConnector] = []

    if getattr(config, "openmeteo_enabled", False):
        try:
            from src.research.connectors.open_meteo import OpenMeteoConnector
            connectors.append(OpenMeteoConnector(config))
        except Exception as e:
            log.warning("registry.openmeteo_load_failed", error=str(e))

    if getattr(config, "fred_enabled", False):
        try:
            from src.research.connectors.fred import FredConnector
            connectors.append(FredConnector(config))
        except Exception as e:
            log.warning("registry.fred_load_failed", error=str(e))

    if getattr(config, "coingecko_enabled", False):
        try:
            from src.research.connectors.coingecko import CoinGeckoConnector
            connectors.append(CoinGeckoConnector(config))
        except Exception as e:
            log.warning("registry.coingecko_load_failed", error=str(e))

    if getattr(config, "congress_enabled", False):
        try:
            from src.research.connectors.congress import CongressConnector
            connectors.append(CongressConnector(config))
        except Exception as e:
            log.warning("registry.congress_load_failed", error=str(e))

    if getattr(config, "gdelt_enabled", False):
        try:
            from src.research.connectors.gdelt import GdeltConnector
            connectors.append(GdeltConnector(config))
        except Exception as e:
            log.warning("registry.gdelt_load_failed", error=str(e))

    if getattr(config, "courtlistener_enabled", False):
        try:
            from src.research.connectors.courtlistener import CourtListenerConnector
            connectors.append(CourtListenerConnector(config))
        except Exception as e:
            log.warning("registry.courtlistener_load_failed", error=str(e))

    if getattr(config, "edgar_enabled", False):
        try:
            from src.research.connectors.edgar import EdgarConnector
            connectors.append(EdgarConnector(config))
        except Exception as e:
            log.warning("registry.edgar_load_failed", error=str(e))

    if getattr(config, "arxiv_enabled", False):
        try:
            from src.research.connectors.arxiv_connector import ArxivConnector
            connectors.append(ArxivConnector(config))
        except Exception as e:
            log.warning("registry.arxiv_load_failed", error=str(e))

    if getattr(config, "openfda_enabled", False):
        try:
            from src.research.connectors.openfda import OpenFDAConnector
            connectors.append(OpenFDAConnector(config))
        except Exception as e:
            log.warning("registry.openfda_load_failed", error=str(e))

    if getattr(config, "worldbank_enabled", False):
        try:
            from src.research.connectors.worldbank import WorldBankConnector
            connectors.append(WorldBankConnector(config))
        except Exception as e:
            log.warning("registry.worldbank_load_failed", error=str(e))

    if getattr(config, "kalshi_prior_enabled", False):
        try:
            from src.research.connectors.kalshi_prior import KalshiPriorConnector
            connectors.append(KalshiPriorConnector(config))
        except Exception as e:
            log.warning("registry.kalshi_prior_load_failed", error=str(e))

    if getattr(config, "metaculus_enabled", False):
        try:
            from src.research.connectors.metaculus import MetaculusConnector
            connectors.append(MetaculusConnector(config))
        except Exception as e:
            log.warning("registry.metaculus_load_failed", error=str(e))

    if getattr(config, "wikipedia_pageviews_enabled", False):
        try:
            from src.research.connectors.wikipedia_pageviews import WikipediaPageviewsConnector
            connectors.append(WikipediaPageviewsConnector(config))
        except Exception as e:
            log.warning("registry.wikipedia_pageviews_load_failed", error=str(e))

    if getattr(config, "google_trends_enabled", False):
        try:
            from src.research.connectors.google_trends import GoogleTrendsConnector
            connectors.append(GoogleTrendsConnector(config))
        except Exception as e:
            log.warning("registry.google_trends_load_failed", error=str(e))

    if getattr(config, "pubmed_enabled", False):
        try:
            from src.research.connectors.pubmed import PubMedConnector
            connectors.append(PubMedConnector(config))
        except Exception as e:
            log.warning("registry.pubmed_load_failed", error=str(e))

    if getattr(config, "reddit_sentiment_enabled", False):
        try:
            from src.research.connectors.reddit_sentiment import RedditSentimentConnector
            connectors.append(RedditSentimentConnector(config))
        except Exception as e:
            log.warning("registry.reddit_sentiment_load_failed", error=str(e))

    if getattr(config, "manifold_enabled", False):
        try:
            from src.research.connectors.manifold import ManifoldConnector
            connectors.append(ManifoldConnector(config))
        except Exception as e:
            log.warning("registry.manifold_load_failed", error=str(e))

    if getattr(config, "predictit_enabled", False):
        try:
            from src.research.connectors.predictit import PredictItConnector
            connectors.append(PredictItConnector(config))
        except Exception as e:
            log.warning("registry.predictit_load_failed", error=str(e))

    if getattr(config, "sports_odds_enabled", False):
        try:
            from src.research.connectors.sports_odds import SportsOddsConnector
            connectors.append(SportsOddsConnector(config))
        except Exception as e:
            log.warning("registry.sports_odds_load_failed", error=str(e))

    if getattr(config, "sports_stats_enabled", False):
        try:
            from src.research.connectors.sports_stats import SportsStatsConnector
            connectors.append(SportsStatsConnector(config))
        except Exception as e:
            log.warning("registry.sports_stats_load_failed", error=str(e))

    if getattr(config, "spotify_charts_enabled", False):
        try:
            from src.research.connectors.spotify_charts import SpotifyChartsConnector
            connectors.append(SpotifyChartsConnector(config))
        except Exception as e:
            log.warning("registry.spotify_charts_load_failed", error=str(e))

    if getattr(config, "crypto_futures_enabled", False):
        try:
            from src.research.connectors.crypto_futures import CryptoFuturesConnector
            connectors.append(CryptoFuturesConnector(config))
        except Exception as e:
            log.warning("registry.crypto_futures_load_failed", error=str(e))

    if getattr(config, "defillama_enabled", False):
        try:
            from src.research.connectors.defillama import DeFiLlamaConnector
            connectors.append(DeFiLlamaConnector(config))
        except Exception as e:
            log.warning("registry.defillama_load_failed", error=str(e))

    if getattr(config, "acled_enabled", False):
        try:
            from src.research.connectors.acled import AcledConnector
            connectors.append(AcledConnector(config))
        except Exception as e:
            log.warning("registry.acled_load_failed", error=str(e))

    if getattr(config, "github_activity_enabled", False):
        try:
            from src.research.connectors.github_activity import GitHubActivityConnector
            connectors.append(GitHubActivityConnector(config))
        except Exception as e:
            log.warning("registry.github_activity_load_failed", error=str(e))

    if getattr(config, "kronos_enabled", False):
        try:
            from src.research.connectors.kronos_connector import KronosConnector
            connectors.append(KronosConnector(config))
        except Exception as e:
            log.warning("registry.kronos_load_failed", error=str(e))

    log.debug(
        "registry.connectors_loaded",
        count=len(connectors),
        names=[c.name for c in connectors],
    )
    return connectors
