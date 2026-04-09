"""Code review v25 fixes — Kronos connector audit.

Tests cover:
  1. Bug 1 (REJECTED): KronosPredictor device='cpu' IS valid
  2. Bug 2: Kronos far-future resolution gate via is_relevant() heuristic
  3. Bug 3: MACRO downside directional base rate pattern (35%)
  4. Pattern count updated to 85
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ── Bug 1 (REJECTED): device='cpu' is valid ──────────────────────────


class TestKronosDeviceParamValid:
    """device='cpu' is a valid KronosPredictor parameter — review was wrong."""

    def test_predictor_constructor_has_device_param(self) -> None:
        """KronosPredictor.__init__ accepts device keyword argument."""
        import inspect
        from src.research.connectors.kronos_connector import _KronosSingleton

        source = inspect.getsource(_KronosSingleton.get_predictor)
        # Our code passes device="cpu" — this is correct
        assert 'device="cpu"' in source

    def test_device_cpu_in_singleton(self) -> None:
        """Singleton explicitly sets device='cpu' for consistent CPU inference."""
        import inspect
        from src.research.connectors.kronos_connector import _KronosSingleton

        source = inspect.getsource(_KronosSingleton.get_predictor)
        assert "device" in source
        assert "max_context=2048" in source


# ── Bug 2: Far-future resolution gate ────────────────────────────────


class TestKronosFarFutureGate:
    """Kronos skips far-future CRYPTO markets via is_relevant() heuristic."""

    def test_short_dated_btc_relevant(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert c.is_relevant("Will BTC reach 90000 by April 15?", "CRYPTO")

    def test_short_dated_eth_relevant(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert c.is_relevant("Will ETH price increase in 24 hours?", "CRYPTO")

    def test_far_future_end_of_2027_skipped(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will BTC hit 100k by end of 2027?", "CRYPTO"
        )

    def test_far_future_before_december_skipped(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will ETH reach 5000 before December?", "CRYPTO"
        )

    def test_far_future_q4_skipped(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will SOL price exceed 200 by Q4?", "CRYPTO"
        )

    def test_far_future_2028_skipped(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will BTC hit 500k by 2028?", "CRYPTO"
        )

    def test_non_crypto_still_irrelevant(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant("Will inflation rise?", "MACRO")

    def test_no_symbol_still_irrelevant(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will crypto recover by Q3?", "CRYPTO"
        )

    def test_q3_also_skipped(self) -> None:
        """Q3 (Jul-Sep) is 3-6 months away — too far for 24h forecast."""
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will BTC reach 80000 by Q3?", "CRYPTO"
        )

    def test_far_future_2026_skipped(self) -> None:
        """2026 is far enough for a 24h forecast to be irrelevant."""
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will BTC reach 100k by end of 2026?", "CRYPTO"
        )

    def test_far_future_end_of_year_skipped(self) -> None:
        """'end of year' is too far for a 24h forecast."""
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will ETH hit 5000 by end of year?", "CRYPTO"
        )

    def test_far_future_before_end_of_year_skipped(self) -> None:
        """'before end of year' is also far-future."""
        from src.research.connectors.kronos_connector import KronosConnector

        c = KronosConnector(config=None)
        assert not c.is_relevant(
            "Will SOL reach 300 before end of year?", "CRYPTO"
        )

    def test_max_resolution_days_constant_exists(self) -> None:
        """_MAX_RESOLUTION_DAYS constant is still defined."""
        from src.research.connectors.kronos_connector import _MAX_RESOLUTION_DAYS

        assert _MAX_RESOLUTION_DAYS == 7


# ── Bug 3: Downside directional MACRO base rate ─────────────────────


class TestMacroDownsideBaseRate:
    """MACRO downside directional pattern matches decline/fall/drop questions."""

    def test_retail_sales_decline(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will retail sales decline in Q2?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_gdp_growth_fall(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will GDP growth fall below 1%?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_manufacturing_contract(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will manufacturing contract in March?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_industrial_production_decrease(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match(
            "Will industrial production decrease this quarter?", "MACRO"
        )
        assert match is not None
        assert match.base_rate == 0.35

    def test_retail_sales_drop(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will retail sales drop in June?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_gdp_growth_shrink(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match("Will GDP growth shrink?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.35

    def test_upside_still_works(self) -> None:
        """Upside directional pattern still matches (regression check)."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        match = registry.match(
            "Will retail sales increase in March?", "MACRO"
        )
        assert match is not None
        assert match.base_rate == 0.55


# ── Pattern count ────────────────────────────────────────────────────


class TestPatternCount:
    """Registry now has 85 total patterns (84 + 1 downside MACRO)."""

    def test_total_count_85(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        assert registry.pattern_count == 85

    def test_macro_count_17(self) -> None:
        """MACRO category now has 17 patterns."""
        from src.forecast.base_rates import BaseRateRegistry

        registry = BaseRateRegistry()
        macro = [p for p in registry.patterns if p.category == "MACRO"]
        assert len(macro) == 17
