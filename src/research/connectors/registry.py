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

    log.debug(
        "registry.connectors_loaded",
        count=len(connectors),
        names=[c.name for c in connectors],
    )
    return connectors
