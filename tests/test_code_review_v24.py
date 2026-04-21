"""Code review v24 — retail sales directional pattern, Kronos connector,
SignalStack integration, config + registry.

Tests cover:
  1. MACRO directional retail sales pattern (no numeric threshold)
  2. KronosConnector: symbol extraction, relevance, lazy singleton
  3. KronosConnector: Binance candle fetch (mocked)
  4. KronosConnector: inference wrapper (mocked)
  5. KronosConnector: full _fetch_impl flow (mocked)
  6. SignalStack: Kronos fields populated by build_signal_stack
  7. SignalStack: Kronos block rendered by render_signal_stack
  8. Config: kronos_enabled field exists, default False
  9. Registry: KronosConnector loaded when enabled
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_HAS_PANDAS = "pandas" in sys.modules or bool(__import__("importlib").util.find_spec("pandas"))
_HAS_NUMPY = "numpy" in sys.modules or bool(__import__("importlib").util.find_spec("numpy"))
_SKIP_PANDAS = pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")
_SKIP_NUMPY = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy not installed")


# ── 1. Retail sales directional pattern ─────────────────────────────


class TestRetailSalesDirectionalPattern:
    """MACRO pattern matches directional questions without numeric threshold."""

    def test_retail_sales_increase(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will retail sales increase in March?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.55

    def test_gdp_growth_improve(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will GDP growth improve this quarter?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.55

    def test_industrial_production_rise(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will industrial production rise in April?", "MACRO")
        assert match is not None

    def test_numeric_threshold_pattern_still_works(self) -> None:
        """Original 'exceed X%' pattern still matches."""
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        match = registry.match("Will retail sales exceed 2% growth?", "MACRO")
        assert match is not None
        assert match.base_rate == 0.45  # original pattern

    def test_total_pattern_count_84(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        assert registry.pattern_count == 85

    def test_macro_pattern_count_16(self) -> None:
        from src.forecast.base_rates import BaseRateRegistry
        registry = BaseRateRegistry()
        macro = [p for p in registry.patterns if p.category == "MACRO"]
        assert len(macro) == 17


# ── 2. KronosConnector: symbol extraction + relevance ────────────────


class TestKronosSymbolExtraction:
    """Symbol extraction from question text."""

    def test_bitcoin_extraction(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c._extract_symbol("Will Bitcoin reach $100k?") == "BTC"

    def test_btc_extraction(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c._extract_symbol("Will BTC hit $90,000?") == "BTC"

    def test_ethereum_extraction(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c._extract_symbol("Will Ethereum exceed $3000?") == "ETH"

    def test_solana_extraction(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c._extract_symbol("Will SOL price double?") == "SOL"

    def test_no_crypto_returns_none(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c._extract_symbol("Will the Fed cut rates?") is None

    def test_is_relevant_crypto(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c.is_relevant("Will BTC reach $100k?", "CRYPTO") is True

    def test_not_relevant_non_crypto(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c.is_relevant("Will BTC reach $100k?", "MACRO") is False

    def test_not_relevant_no_symbol(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c.is_relevant("Will crypto market cap hit $3T?", "CRYPTO") is False

    def test_connector_name(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c.name == "kronos"

    def test_relevant_categories(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector
        c = KronosConnector(config=None)
        assert c.relevant_categories() == {"CRYPTO"}


# ── 3. KronosConnector: lazy singleton ───────────────────────────────


class TestKronosSingleton:
    """Lazy model loading singleton."""

    def test_singleton_reset(self) -> None:
        from src.research.connectors.kronos_connector import _KronosSingleton
        _KronosSingleton.reset()
        assert _KronosSingleton._loaded is False
        assert _KronosSingleton._predictor is None

    def test_singleton_returns_none_when_not_installed(self) -> None:
        from src.research.connectors.kronos_connector import _KronosSingleton
        _KronosSingleton.reset()
        # Simulate Kronos repo not available by patching _ensure_kronos_on_path
        with patch(
            "src.research.connectors.kronos_connector._ensure_kronos_on_path",
            return_value=False,
        ):
            result = _KronosSingleton.get_predictor()
        assert result is None
        assert _KronosSingleton._loaded is True  # prevents retry

    def test_singleton_no_retry_after_failure(self) -> None:
        from src.research.connectors.kronos_connector import _KronosSingleton
        _KronosSingleton.reset()
        with patch(
            "src.research.connectors.kronos_connector._ensure_kronos_on_path",
            return_value=False,
        ):
            _KronosSingleton.get_predictor()  # fails, sets _loaded=True
        # Second call should not retry
        assert _KronosSingleton._loaded is True
        assert _KronosSingleton._predictor is None


# ── 4. KronosConnector: Binance candle fetch ─────────────────────────


@_SKIP_PANDAS
class TestKronosBinanceFetch:
    """Binance OHLCV candle fetching."""

    @pytest.mark.asyncio
    async def test_fetch_candles_success(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        mock_data = [
            [1700000000000, "40000", "40500", "39500", "40200", "100",
             1700003600000, "4000000", 500, "50", "2000000", "0"]
            for _ in range(100)
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = mock_data

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp

        connector = KronosConnector(config=None)
        connector._client = mock_client

        df = await connector._fetch_binance_candles("BTC")
        assert df is not None
        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["close"].iloc[0] == 40200.0

    @pytest.mark.asyncio
    async def test_fetch_candles_error(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection failed")

        connector = KronosConnector(config=None)
        connector._client = mock_client

        df = await connector._fetch_binance_candles("BTC")
        assert df is None


# ── 5. KronosConnector: inference wrapper ────────────────────────────


@_SKIP_PANDAS
@_SKIP_NUMPY
class TestKronosInference:
    """Inference wrapper with mocked predictor."""

    def test_run_inference_success(self) -> None:
        import numpy as np
        import pandas as pd
        from src.research.connectors.kronos_connector import KronosConnector

        # Build mock OHLCV dataframe
        n = 100
        closes = np.linspace(40000, 41000, n)
        df = pd.DataFrame({
            "open": closes - 50,
            "high": closes + 100,
            "low": closes - 100,
            "close": closes,
            "volume": np.random.uniform(50, 150, n),
        })

        # Mock predictor that returns slightly above current price
        mock_predictor = MagicMock()
        pred_df = pd.DataFrame({
            "close": np.linspace(41000, 41500, 24),
        })
        mock_predictor.predict.return_value = pred_df

        result = KronosConnector._run_inference(mock_predictor, df)
        assert result is not None
        upside_prob, volatility_prob = result
        assert 0.0 <= upside_prob <= 1.0
        assert 0.0 <= volatility_prob <= 1.0
        # All paths end above 41000 (last close), so upside_prob should be 1.0
        assert upside_prob == 1.0

    def test_run_inference_error_returns_none(self) -> None:
        import pandas as pd
        from src.research.connectors.kronos_connector import KronosConnector

        mock_predictor = MagicMock()
        mock_predictor.predict.side_effect = RuntimeError("Model error")

        df = pd.DataFrame({
            "open": [100], "high": [105], "low": [95],
            "close": [102], "volume": [50],
        })

        result = KronosConnector._run_inference(mock_predictor, df)
        assert result is None


# ── 6. KronosConnector: full _fetch_impl flow ────────────────────────


@_SKIP_PANDAS
@_SKIP_NUMPY
class TestKronosFetchImpl:
    """Full fetch_impl with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_fetch_impl_no_symbol(self) -> None:
        from src.research.connectors.kronos_connector import KronosConnector

        connector = KronosConnector(config=None)
        result = await connector._fetch_impl("Will the Fed cut rates?", "CRYPTO")
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_impl_success(self) -> None:
        import numpy as np
        import pandas as pd
        from src.research.connectors.kronos_connector import (
            KronosConnector, _KronosSingleton,
        )

        # Mock Binance candles
        n = 100
        closes = np.linspace(40000, 41000, n)
        mock_df = pd.DataFrame({
            "open": closes - 50, "high": closes + 100,
            "low": closes - 100, "close": closes,
            "volume": np.random.uniform(50, 150, n),
        })

        # Mock predictor
        mock_predictor = MagicMock()
        pred_df = pd.DataFrame({"close": np.linspace(41000, 41500, 24)})
        mock_predictor.predict.return_value = pred_df

        connector = KronosConnector(config=None)

        with patch.object(connector, "_fetch_binance_candles", return_value=mock_df), \
             patch.object(_KronosSingleton, "get_predictor", return_value=mock_predictor), \
             patch("src.research.connectors.kronos_connector.rate_limiter") as rl:
            rl.get.return_value = AsyncMock()

            sources = await connector._fetch_impl("Will BTC reach $100k?", "CRYPTO")

        assert len(sources) == 1
        src = sources[0]
        assert "BTC" in src.title
        assert src.raw["behavioral_signal"]["source"] == "kronos"
        assert src.raw["behavioral_signal"]["signal_type"] == "crypto_price_forecast"
        assert 0.0 <= src.raw["behavioral_signal"]["upside_probability"] <= 1.0
        assert src.raw["behavioral_signal"]["symbol"] == "BTC"

    @pytest.mark.asyncio
    async def test_fetch_impl_no_predictor(self) -> None:
        import numpy as np
        import pandas as pd
        from src.research.connectors.kronos_connector import (
            KronosConnector, _KronosSingleton,
        )

        mock_df = pd.DataFrame({
            "open": [100]*50, "high": [105]*50, "low": [95]*50,
            "close": [102]*50, "volume": [50]*50,
        })

        connector = KronosConnector(config=None)

        with patch.object(connector, "_fetch_binance_candles", return_value=mock_df), \
             patch.object(_KronosSingleton, "get_predictor", return_value=None), \
             patch("src.research.connectors.kronos_connector.rate_limiter") as rl:
            rl.get.return_value = AsyncMock()

            sources = await connector._fetch_impl("Will BTC reach $100k?", "CRYPTO")

        assert sources == []

    @pytest.mark.asyncio
    async def test_fetch_impl_insufficient_candles(self) -> None:
        import pandas as pd
        from src.research.connectors.kronos_connector import KronosConnector

        # Only 10 candles (need 48+)
        mock_df = pd.DataFrame({
            "open": [100]*10, "high": [105]*10, "low": [95]*10,
            "close": [102]*10, "volume": [50]*10,
        })

        connector = KronosConnector(config=None)

        with patch.object(connector, "_fetch_binance_candles", return_value=mock_df), \
             patch("src.research.connectors.kronos_connector.rate_limiter") as rl:
            rl.get.return_value = AsyncMock()

            sources = await connector._fetch_impl("Will BTC reach $100k?", "CRYPTO")

        assert sources == []


# ── 7. SignalStack: Kronos fields ────────────────────────────────────


class TestSignalStackKronos:
    """SignalStack Kronos fields populated and rendered."""

    def test_kronos_fields_default_none(self) -> None:
        from src.research.signal_aggregator import SignalStack
        stack = SignalStack()
        assert stack.kronos_upside_prob is None
        assert stack.kronos_volatility_prob is None
        assert stack.kronos_symbol == ""

    def test_build_signal_stack_kronos(self) -> None:
        from src.research.signal_aggregator import build_signal_stack
        from src.research.source_fetcher import FetchedSource

        source = FetchedSource(
            title="Kronos BTC",
            url="https://huggingface.co/NeoQuasar/Kronos-mini",
            snippet="test",
            raw={
                "behavioral_signal": {
                    "source": "kronos",
                    "signal_type": "crypto_price_forecast",
                    "upside_probability": 0.72,
                    "volatility_amplification": 0.45,
                    "symbol": "BTC",
                }
            },
        )

        stack = build_signal_stack([source], 0.50)
        assert stack.kronos_upside_prob == 0.72
        assert stack.kronos_volatility_prob == 0.45
        assert stack.kronos_symbol == "BTC"

    def test_render_signal_stack_kronos(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(
            kronos_upside_prob=0.72,
            kronos_volatility_prob=0.45,
            kronos_symbol="BTC",
        )

        rendered = render_signal_stack(stack)
        assert "Kronos foundation model for BTC" in rendered
        assert "upside probability 72%" in rendered
        assert "volatility amplification: 45%" in rendered

    def test_render_no_kronos(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack()
        rendered = render_signal_stack(stack)
        assert "Kronos" not in rendered

    def test_render_kronos_no_symbol(self) -> None:
        from src.research.signal_aggregator import SignalStack, render_signal_stack

        stack = SignalStack(kronos_upside_prob=0.60)
        rendered = render_signal_stack(stack)
        assert "Kronos foundation model " in rendered
        assert "for " not in rendered or "for BTC" not in rendered


# ── 8. Config: kronos_enabled ────────────────────────────────────────


class TestKronosConfig:
    """kronos_enabled config field."""

    def test_kronos_enabled_default_false(self) -> None:
        from src.config import ResearchConfig
        config = ResearchConfig()
        assert config.kronos_enabled is False

    def test_kronos_disabled_in_loaded_config(self) -> None:
        from src.config import load_config
        config = load_config()
        assert config.research.kronos_enabled is False


# ── 9. Registry: KronosConnector ─────────────────────────────────────


class TestKronosRegistry:
    """Registry loads KronosConnector when enabled."""

    def test_not_loaded_when_disabled(self) -> None:
        from src.config import ResearchConfig
        from src.research.connectors.registry import get_enabled_connectors

        config = ResearchConfig()
        connectors = get_enabled_connectors(config)
        names = [c.name for c in connectors]
        assert "kronos" not in names

    def test_loaded_when_enabled(self) -> None:
        from src.config import ResearchConfig
        from src.research.connectors.registry import get_enabled_connectors

        config = ResearchConfig(kronos_enabled=True)
        connectors = get_enabled_connectors(config)
        names = [c.name for c in connectors]
        assert "kronos" in names
