"""Kronos foundation model connector — crypto price forecast signal.

Uses the Kronos financial foundation model (AAAI 2026, MIT licensed) to generate
24-hour price forecasts for CRYPTO markets via Monte Carlo sampling of predicted
K-line sequences.  Only runs for short-dated CRYPTO markets (<=7 days to resolution).

The model is lazy-loaded on first use.  Requires ``kronos`` extras:
  pip install torch einops safetensors huggingface_hub

Rate-limited via the existing ``binance`` bucket (OHLCV data fetch).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from src.connectors.rate_limiter import rate_limiter
from src.observability.logger import get_logger
from src.research.connectors.base import BaseResearchConnector
from src.research.source_fetcher import FetchedSource

log = get_logger(__name__)

# ── Symbol mapping ─────────────────────────────────────────────────────

_COIN_SYMBOLS: dict[str, str] = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "bnb": "BNB",
    "binance coin": "BNB",
    "xrp": "XRP",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "cardano": "ADA",
    "ada": "ADA",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "polygon": "MATIC",
    "matic": "MATIC",
}

# ── Constants ──────────────────────────────────────────────────────────

_MODEL_ID = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-mini")
_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-2k"
_LOOKBACK_HOURS = 360  # 15 days of 1h candles
_FORECAST_HOURS = 24
_MONTE_CARLO_N = int(os.environ.get("KRONOS_MONTE_CARLO_N", "10"))
_MAX_RESOLUTION_DAYS = 7


# ── Lazy singleton for model loading ──────────────────────────────────


class _KronosSingleton:
    """Lazy-loaded singleton holding the Kronos model and tokenizer."""

    _predictor: Any = None
    _loaded: bool = False

    @classmethod
    def get_predictor(cls) -> Any:
        """Load model on first call; return None if unavailable."""
        if not cls._loaded:
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer

                tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_ID)
                model = Kronos.from_pretrained(_MODEL_ID)
                cls._predictor = KronosPredictor(
                    model, tokenizer, device="cpu", max_context=2048
                )
                cls._loaded = True
                log.info("kronos.model_loaded", model=_MODEL_ID)
            except Exception as e:
                log.warning("kronos.load_failed", error=str(e))
                cls._loaded = True  # prevent retry loops
        return cls._predictor

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._predictor = None
        cls._loaded = False


# ── Connector ──────────────────────────────────────────────────────────


class KronosConnector(BaseResearchConnector):
    """Kronos foundation model price forecast for CRYPTO markets."""

    @property
    def name(self) -> str:
        return "kronos"

    def relevant_categories(self) -> set[str]:
        return {"CRYPTO"}

    def is_relevant(self, question: str, market_type: str) -> bool:
        if market_type != "CRYPTO":
            return False
        return bool(self._extract_symbol(question))

    async def _fetch_impl(
        self,
        question: str,
        market_type: str,
    ) -> list[FetchedSource]:
        symbol = self._extract_symbol(question)
        if not symbol:
            return []

        # Fetch OHLCV from Binance
        await rate_limiter.get("binance").acquire()
        ohlcv_df = await self._fetch_binance_candles(symbol)
        if ohlcv_df is None or len(ohlcv_df) < 48:
            return []

        # Run Kronos inference in thread (sync PyTorch model)
        predictor = _KronosSingleton.get_predictor()
        if predictor is None:
            return []

        result = await asyncio.to_thread(self._run_inference, predictor, ohlcv_df)
        if result is None:
            return []

        upside_prob, volatility_prob = result

        return [
            self._make_source(
                title=f"Kronos Price Forecast: {symbol}",
                url=f"https://huggingface.co/{_MODEL_ID}",
                snippet=(
                    f"Kronos {symbol}: upside_prob={upside_prob:.1%}, "
                    f"volatility_amplification={volatility_prob:.1%} "
                    f"(24h, N={_MONTE_CARLO_N})"
                ),
                publisher="Kronos Foundation Model",
                content=(
                    f"Kronos ({_MODEL_ID}) 24-hour price forecast for {symbol}:\n"
                    f"  Upside probability: {upside_prob:.1%}\n"
                    f"  Volatility amplification probability: {volatility_prob:.1%}\n"
                    f"  Monte Carlo sample paths: {_MONTE_CARLO_N}\n"
                    f"  Lookback: {_LOOKBACK_HOURS}h ({_LOOKBACK_HOURS // 24}d)\n"
                    f"  Source: Kronos foundation model (AAAI 2026)"
                ),
                authority_score=0.75,
                raw={
                    "behavioral_signal": {
                        "source": "kronos",
                        "signal_type": "crypto_price_forecast",
                        "upside_probability": round(upside_prob, 4),
                        "volatility_amplification": round(volatility_prob, 4),
                        "symbol": symbol,
                        "horizon_hours": _FORECAST_HOURS,
                        "sample_paths": _MONTE_CARLO_N,
                    }
                },
            )
        ]

    # ── Inference ──────────────────────────────────────────────────────

    @staticmethod
    def _run_inference(
        predictor: Any,
        df: "pd.DataFrame",
    ) -> tuple[float, float] | None:
        """Run Kronos prediction (sync). Returns (upside_prob, volatility_prob)."""
        try:
            import numpy as np
            import pandas as pd

            last_price = float(df["close"].iloc[-1])
            n = len(df)

            # Build timestamps for input and forecast horizon
            x_ts = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="1h")
            y_ts = pd.date_range(
                start=x_ts[-1], periods=_FORECAST_HOURS + 1, freq="1h"
            )[1:]

            # Monte Carlo sampling
            final_prices: list[float] = []
            path_volatilities: list[float] = []
            for _ in range(_MONTE_CARLO_N):
                pred = predictor.predict(
                    df=df,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=_FORECAST_HOURS,
                    T=1.0,
                    top_p=0.9,
                    sample_count=1,
                )
                final_prices.append(float(pred["close"].iloc[-1]))
                # Per-path volatility: std of returns within the predicted path
                pred_returns = pred["close"].pct_change().dropna()
                path_volatilities.append(float(pred_returns.std()))

            paths_arr = np.array(final_prices)

            # Upside probability: fraction of paths ending above current price
            upside_prob = float(np.mean(paths_arr > last_price))

            # Volatility amplification: fraction of paths with higher vol
            # than recent historical vol
            hist_vol = float(df["close"].pct_change().dropna().std())
            vol_arr = np.array(path_volatilities)
            volatility_prob = float(np.mean(vol_arr > hist_vol)) if hist_vol > 0 else 0.5

            return upside_prob, volatility_prob

        except Exception as e:
            log.warning("kronos.inference_error", error=str(e))
            return None

    # ── Binance OHLCV fetch ────────────────────────────────────────────

    async def _fetch_binance_candles(self, symbol: str) -> "pd.DataFrame | None":
        """Fetch OHLCV 1h candles from Binance public API.

        Tries api.binance.com first, falls back to api.binance.us if blocked.
        """
        try:
            import pandas as pd

            client = self._get_client(timeout=15.0)
            params = {
                "symbol": f"{symbol}USDT",
                "interval": "1h",
                "limit": _LOOKBACK_HOURS,
            }

            # Try global endpoint first, fallback to US
            data = None
            for base_url in [
                "https://api.binance.com/api/v3/klines",
                "https://api.binance.us/api/v3/klines",
            ]:
                try:
                    resp = await client.get(base_url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception:
                    continue

            if data is None:
                log.warning("kronos.binance_all_endpoints_failed", symbol=symbol)
                return None

            df = pd.DataFrame(
                data,
                columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_vol", "trades", "taker_base",
                    "taker_quote", "ignore",
                ],
            )
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as e:
            log.warning("kronos.binance_fetch_error", symbol=symbol, error=str(e))
            return None

    # ── Symbol extraction ──────────────────────────────────────────────

    @staticmethod
    def _extract_symbol(question: str) -> str | None:
        """Extract crypto symbol from question text."""
        q = question.lower()
        for keyword, symbol in _COIN_SYMBOLS.items():
            if keyword in q:
                return symbol
        return None
